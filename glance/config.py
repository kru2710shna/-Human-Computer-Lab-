"""Model registry and global constants.

Every model Glance can run is declared here, together with the facts the rest of
the code needs: how it encodes box coordinates, what resize factor its vision
tower uses, and a rough parameter count used only for a pre-flight sanity check.
The *authoritative* parameter count always comes from `glance budget`, which
reads the real checkpoint headers (see glance/budget.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Hard limit from the challenge: total learned parameters on the local path.
PARAM_BUDGET = 6_000_000_000


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    # "abs"     -> boxes are absolute pixels of the image the model was fed
    # "rel1000" -> boxes are normalised to 0..1000 on both axes
    coord_mode: str
    # Side length that the vision tower's merged patch grid snaps to.
    factor: int
    approx_params: float
    mlx_id: str | None = None
    # Pixel budget for the image we feed the model. Lower = faster, less detail.
    max_pixels: int = 1280 * 28 * 28
    min_pixels: int = 256 * 28 * 28
    notes: str = ""
    extra: dict = field(default_factory=dict)


MODELS: dict[str, ModelSpec] = {
    # Default on Apple silicon with <=16 GB. Same 3.75B total parameters as the
    # bf16 checkpoint -- 4-bit quantisation changes bytes per weight, not the
    # number of learned weights -- but ~2.1 GB resident instead of ~7.5 GB.
    "qwen2.5-vl-3b": ModelSpec(
        key="qwen2.5-vl-3b",
        hf_id="Qwen/Qwen2.5-VL-3B-Instruct",   # audited for parameters
        coord_mode="abs",
        factor=28,
        approx_params=3.75e9,
        mlx_id="mlx-community/Qwen2.5-VL-3B-Instruct-4bit",  # what actually runs
        notes="Default. 3.75B total params (62.6% of the 6B cap). ~2.1 GB in 4-bit "
              "MLX, ~7.5 GB in bf16. The bf16 path needs a 16 GB machine.",
    ),
    # Fallback if MLX will not install, or for an even tighter memory budget.
    "qwen2-vl-2b": ModelSpec(
        key="qwen2-vl-2b",
        hf_id="Qwen/Qwen2-VL-2B-Instruct",
        coord_mode="rel1000",
        factor=28,
        approx_params=2.21e9,
        mlx_id="mlx-community/Qwen2-VL-2B-Instruct-4bit",
        max_pixels=1024 * 28 * 28,
        notes="2.21B total. ~1.3 GB in 4-bit, ~4.4 GB in bf16. Weaker grounding; "
              "used as the low-memory comparison point in the benchmark.",
    ),
    # Deliberately over budget. Registered so `glance budget` can demonstrate the
    # audit rejecting a non-compliant configuration (8.29B > 6B).
    "qwen2.5-vl-7b": ModelSpec(
        key="qwen2.5-vl-7b",
        hf_id="Qwen/Qwen2.5-VL-7B-Instruct",
        coord_mode="abs",
        factor=28,
        approx_params=8.29e9,
        notes="OVER BUDGET (8.29B > 6B). Negative control for the parameter audit.",
    ),
}

DEFAULT_MODEL = "qwen2.5-vl-3b"


def get_spec(key: str) -> ModelSpec:
    if key in MODELS:
        return MODELS[key]
    for spec in MODELS.values():
        if spec.hf_id == key:
            return spec
    raise KeyError(f"Unknown model '{key}'. Choose one of: {', '.join(MODELS)}")