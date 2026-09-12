"""Parameter audit: prove the local inference path fits the 6B budget.

The challenge counts *total* learned parameters across every required encoder,
backbone, adapter, projector, head, and auxiliary generator. So we do not trust a
number from a model card. We read the actual checkpoint headers and sum the
tensor shapes ourselves.

safetensors layout: 8-byte little-endian header length N, then N bytes of JSON
mapping tensor name -> {dtype, shape, data_offsets}. That header is a few KB, so
an HTTP range request audits a repo without downloading any weights.

Two subtleties this handles:
  * Sharded checkpoints repeat nothing, but we still de-duplicate by tensor name.
  * Tied weights (lm_head sharing embed_tokens) must be counted ONCE. A tied
    lm_head is usually absent from the checkpoint; we detect the config flag and
    report it explicitly rather than silently counting or not counting it.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

from .config import PARAM_BUDGET, ModelSpec

_HEADER_PROBE = 8 * 1024 * 1024  # enough for any header we'll meet

# Order matters: first matching pattern wins.
_COMPONENT_RULES = [
    ("vision encoder", ("visual.blocks", "vision_tower", "vision_model", "visual.patch_embed",
                        "visual.rotary", "vision_encoder")),
    ("vision-language projector", ("visual.merger", "multi_modal_projector", "mm_projector",
                                   "connector", "perceiver", "resampler", "modality_projection")),
    ("token embeddings", ("embed_tokens", "wte", "word_embeddings")),
    ("language backbone", ("model.layers", "language_model", "model.norm", "transformer.h")),
    ("output head", ("lm_head", "score", "classifier")),
]


def classify(tensor_name: str) -> str:
    for component, prefixes in _COMPONENT_RULES:
        if any(p in tensor_name for p in prefixes):
            return component
    return "other"


@dataclass
class BudgetReport:
    model_id: str
    source: str
    total: int = 0
    components: dict[str, int] = field(default_factory=dict)
    dtypes: dict[str, int] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)
    tied_embeddings: bool | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def within_budget(self) -> bool:
        return self.total <= PARAM_BUDGET

    @property
    def headroom(self) -> int:
        return PARAM_BUDGET - self.total

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "source": self.source,
            "total_parameters": self.total,
            "total_parameters_b": round(self.total / 1e9, 4),
            "budget": PARAM_BUDGET,
            "within_budget": self.within_budget,
            "headroom": self.headroom,
            "components": dict(sorted(self.components.items(), key=lambda kv: -kv[1])),
            "dtypes": self.dtypes,
            "tied_embeddings": self.tied_embeddings,
            "files_audited": self.files,
            "notes": self.notes,
        }


# --------------------------------------------------------------- header reads
def _parse_header(blob: bytes) -> dict:
    if len(blob) < 8:
        raise ValueError("truncated safetensors header")
    (n,) = struct.unpack("<Q", blob[:8])
    if n <= 0 or 8 + n > len(blob):
        raise ValueError(f"header length {n} exceeds probe of {len(blob)} bytes")
    return json.loads(blob[8:8 + n])


def _numel(shape) -> int:
    total = 1
    for d in shape:
        total *= int(d)
    return total


def _accumulate(header: dict, seen: dict[str, tuple[int, str]]) -> None:
    for name, meta in header.items():
        if name == "__metadata__" or not isinstance(meta, dict):
            continue
        shape = meta.get("shape") or []
        if not shape:
            continue  # scalars carry no learned capacity worth counting
        seen.setdefault(name, (_numel(shape), str(meta.get("dtype", "?"))))


# ------------------------------------------------------------------ from hub
def _hub_headers(repo_id: str, revision: str = "main") -> tuple[dict[str, tuple[int, str]], list[str]]:
    from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
    import requests

    from huggingface_hub import HfApi

    api = HfApi()
    files = [f for f in api.list_repo_files(repo_id, revision=revision)
             if f.endswith(".safetensors")]
    if not files:
        raise RuntimeError(f"{repo_id} publishes no .safetensors files; cannot audit shapes")

    # An index file tells us the canonical name->shard map, which is the
    # authoritative de-duplicated tensor list for sharded checkpoints.
    seen: dict[str, tuple[int, str]] = {}
    for fname in sorted(files):
        url = hf_hub_url(repo_id, fname, revision=revision)
        meta = get_hf_file_metadata(url)
        probe = min(_HEADER_PROBE, meta.size or _HEADER_PROBE)
        r = requests.get(url, headers={"Range": f"bytes=0-{probe - 1}"}, timeout=60)
        r.raise_for_status()
        _accumulate(_parse_header(r.content), seen)
    return seen, sorted(files)


def _hub_config(repo_id: str, revision: str = "main") -> dict:
    try:
        from huggingface_hub import hf_hub_download

        with open(hf_hub_download(repo_id, "config.json", revision=revision)) as fh:
            return json.load(fh)
    except Exception:
        return {}


# ---------------------------------------------------------------- from local
def _local_headers(root: Path) -> tuple[dict[str, tuple[int, str]], list[str]]:
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"No .safetensors under {root}")
    seen: dict[str, tuple[int, str]] = {}
    for path in shards:
        with open(path, "rb") as fh:
            _accumulate(_parse_header(fh.read(_HEADER_PROBE)), seen)
    return seen, [p.name for p in shards]


def _local_config(root: Path) -> dict:
    cfg = root / "config.json"
    if cfg.exists():
        try:
            return json.loads(cfg.read_text())
        except json.JSONDecodeError:
            pass
    return {}


# -------------------------------------------------------------------- public
def audit(model: str | ModelSpec, revision: str = "main") -> BudgetReport:
    """Audit a model by HF repo id, ModelSpec, or local directory path."""
    spec = model if isinstance(model, ModelSpec) else None
    model_id = spec.hf_id if spec else str(model)

    local = Path(model_id)
    if local.is_dir():
        seen, files = _local_headers(local)
        cfg, source = _local_config(local), f"local checkpoint headers ({local})"
    else:
        seen, files = _hub_headers(model_id, revision)
        cfg, source = _hub_config(model_id, revision), "hub safetensors headers (range requests)"

    report = BudgetReport(model_id=model_id, source=source, files=files)
    for name, (count, dtype) in seen.items():
        report.total += count
        comp = classify(name)
        report.components[comp] = report.components.get(comp, 0) + count
        report.dtypes[dtype] = report.dtypes.get(dtype, 0) + count

    tied = cfg.get("tie_word_embeddings")
    if tied is None:
        tied = (cfg.get("text_config") or {}).get("tie_word_embeddings")
    report.tied_embeddings = tied

    if tied and "lm_head" not in " ".join(seen):
        report.notes.append(
            "Output head shares weights with the token embeddings (tie_word_embeddings=true) "
            "and is absent from the checkpoint. It is counted once, inside 'token embeddings'."
        )
    if "other" in report.components:
        report.notes.append(
            f"{report.components['other']:,} parameters did not match a component rule; they are "
            "still included in the total."
        )
    report.notes.append(
        "Counted from tensor shapes in the checkpoint headers, so this is TOTAL parameters "
        "and is unaffected by quantisation or by which experts are active at inference."
    )
    return report


def audit_pipeline(specs: list[ModelSpec], revision: str = "main") -> dict:
    """Audit every model on the inference path and sum them.

    Glance's default path is a single VLM, but this function is what makes the
    claim checkable: if a second model were ever added (a detector, an OCR head,
    an auxiliary generator), it would appear here and count against the budget.
    """
    reports = [audit(s, revision) for s in specs]
    total = sum(r.total for r in reports)
    return {
        "components_on_path": [r.to_dict() for r in reports],
        "path_total_parameters": total,
        "path_total_parameters_b": round(total / 1e9, 4),
        "budget": PARAM_BUDGET,
        "within_budget": total <= PARAM_BUDGET,
        "headroom": PARAM_BUDGET - total,
    }


def verify_runtime(report: BudgetReport, runtime_count: int, tol: float = 0.005) -> dict:
    """Cross-check the header audit against a live loaded model.

    torch's ``parameters()`` de-duplicates tied weights, so a match here confirms
    the header count was not double-counting. A mismatch is worth surfacing, not
    hiding, so we return the delta rather than asserting.
    """
    delta = runtime_count - report.total
    rel = abs(delta) / max(1, report.total)
    return {
        "header_total": report.total,
        "runtime_total": runtime_count,
        "delta": delta,
        "relative_delta": round(rel, 6),
        "agrees": rel <= tol,
    }