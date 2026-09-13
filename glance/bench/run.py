"""Benchmark runner.

Answers the question the project hinges on: does the zoom-refine stage earn its
latency? Runs identical inputs through pipeline variants and reports accuracy,
latency, and confidence calibration side by side.

Protocol notes that matter for the numbers being trustworthy:
  * One warm-up call is discarded before timing anything. Measured on an M1, the
    first call costs ~4.3s against ~2.8s steady state; averaging it in would
    silently inflate every variant.
  * Variants see the SAME screens and the SAME targets, in the same order.
  * One target per screen by default, sampled deterministically from the seed, so
    reruns are comparable.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from PIL import Image

from ..backends import build_backend
from ..config import get_spec
from ..pipeline import Engine, GroundConfig
from ..telemetry import hardware_report, measure
from . import dataset, metrics

VARIANTS = {
    "coarse":        dict(refine=False, verify=False, fallback_tiles=False),
    "coarse+verify": dict(refine=False, verify=True,  fallback_tiles=False),
    "refine":        dict(refine=True,  verify=False, fallback_tiles=False),
    "full":          dict(refine=True,  verify=True,  fallback_tiles=False),
}


def _pick_targets(rec: dict, k: int, rng: random.Random) -> list[dict]:
    gt = rec["ground_truth"]
    if not gt:
        return []
    return rng.sample(gt, k=min(k, len(gt)))


def run_grounding(engine: Engine, records: list[dict], variants: list[str],
                  targets_per_screen: int = 1, seed: int = 0) -> dict:
    rng = random.Random(seed)
    plan = [(rec, t) for rec in records for t in _pick_targets(rec, targets_per_screen, rng)]

    # Warm-up, discarded.
    if plan:
        engine.ground(plan[0][0]["_image"], plan[0][1]["descriptor"],
                      refine=False, verify=False)

    results: dict[str, list[dict]] = {}
    for name in variants:
        cfg = VARIANTS[name]
        rows = []
        for rec, tgt in plan:
            img = rec["_image"]
            t0 = time.perf_counter()
            if cfg["fallback_tiles"]:
                hit = engine.locate(img, tgt["descriptor"], fallback_tiles=True)
            else:
                hit = engine.ground(img, tgt["descriptor"],
                                    refine=cfg["refine"], verify=cfg["verify"])
            dt = time.perf_counter() - t0

            row = {
                "screen": rec["id"], "scale": rec["scale"], "theme": rec["theme"],
                "role": tgt["role"], "target": tgt["descriptor"],
                "target_height": tgt["height"],
                "latency_s": round(dt, 3), "found": hit is not None,
                "confidence": hit.confidence if hit else None,
                "refined": hit.refined if hit else False,
                "box_kind": tgt.get("box_kind", "tight"),
            }
            if hit:
                row.update(metrics.score_one(hit.box, tuple(tgt["box"])))
                row["pred_box"] = [round(v, 1) for v in hit.box]
                row["true_box"] = tgt["box"]
            rows.append(row)
            print(f"  [{name}] {rec['id']:3d} {tgt['descriptor'][:38]:38s} "
                  f"{'HIT ' if row.get('click_hit') else 'miss'} "
                  f"iou={row.get('iou', 0):.2f} {dt:5.1f}s", flush=True)
        results[name] = rows
    return results


def run(n: int = 24, model: str = "qwen2.5-vl-3b", backend: str = "auto",
        out: str | Path = "runs", variants: list[str] | None = None,
        targets_per_screen: int = 1, size=(1280, 800), seed: int = 0,
        save_screens: bool = True) -> dict:
    variants = variants or list(VARIANTS)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    hw = hardware_report()
    print(json.dumps(hw, indent=2))

    records = dataset.build(n=n, out=(out / f"screens-{stamp}") if save_screens else None,
                            size=size, seed0=seed)

    be = build_backend(model, backend=backend)
    engine = Engine(be, GroundConfig())
    print(f"\nbackend: {be.describe()}\n")

    with measure("benchmark") as res:
        results = run_grounding(engine, records, variants,
                                targets_per_screen=targets_per_screen, seed=seed)

    summary = {}
    for name, rows in results.items():
        summary[name] = {
            "overall": metrics.aggregate(rows),
            "by_scale": metrics.by_group(rows, "scale"),
            "by_role": metrics.by_group(rows, "role"),
            "by_box_kind": metrics.by_group(rows, "box_kind"),
            
        }
        conf = metrics.confidence_analysis(rows)
        if any(c["acted_on"] for c in conf):
            summary[name]["confidence"] = conf

    payload = {
        "timestamp": stamp,
        "hardware": hw,
        "model": model,
        "parameters_total": get_spec(model).approx_params,
        "backend": be.describe(),
        "dataset": {"screens": n, "size": list(size), "targets_per_screen": targets_per_screen,
                    "source": "synthetic, generated by glance.bench.dataset"},
        "resources": res.to_dict(),
        "backend_stats": be.stats(),
        "summary": summary,
        "rows": {k: v for k, v in results.items()},
    }
    path = out / f"bench-{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))

    print("\n" + "=" * 78)
    print(f"{'variant':16s} {'click acc':>10s} {'IoU@50':>8s} {'IoU mean':>9s} "
          f"{'found':>7s} {'median s':>9s}")
    print("-" * 78)
    for name in variants:
        a = summary[name]["overall"]
        print(f"{name:16s} {a['click_accuracy']:>10.1%} {a['iou_at_50']:>8.1%} "
              f"{a['iou_mean']:>9.3f} {a['found_rate']:>7.1%} {a['latency_s_median']:>9.2f}")
    print("=" * 78)
    print(f"\nwrote {path}")
    return payload

DISTRACTORS = [
    "the Print button", "the Export to PDF button", "the Undo button",
    "the search box in the toolbar", "the user avatar in the top right",
    "the notification bell icon", "the zoom slider",
]


def run_distractors(engine: Engine, records: list[dict], n_per_screen: int = 1,
                    seed: int = 0) -> dict:
    """Ask for elements that are NOT on screen and check we return nothing.

    Without this the benchmark cannot distinguish a system that grounds well from
    one that always emits a plausible box. A false positive is worse than a miss:
    a miss stops the workflow, a false positive clicks the wrong thing.
    """
    rng = random.Random(seed + 977)
    rows = []
    for rec in records:
        present = {e["text"].lower() for e in rec["all_elements"]}
        pool = [d for d in DISTRACTORS
                if not any(w in present for w in d.lower().split() if len(w) > 3)]
        if not pool:
            continue
        for target in rng.sample(pool, k=min(n_per_screen, len(pool))):
            hit = engine.ground(rec["_image"], target, refine=False, verify=True)
            acted = bool(hit and (hit.confidence is None
                                  or hit.confidence >= engine.cfg.accept_threshold))
            rows.append({"screen": rec["id"], "target": target,
                         "returned_box": hit is not None,
                         "would_act": acted,
                         "confidence": hit.confidence if hit else None})
            print(f"  [absent] {rec['id']:3d} {target[:36]:36s} "
                  f"{'FALSE POSITIVE' if acted else 'correctly declined'}", flush=True)
    n = len(rows)
    return {
        "n": n,
        "returned_a_box": sum(r["returned_box"] for r in rows),
        "would_act_on_it": sum(r["would_act"] for r in rows),
        "correct_rejection_rate": round(1 - sum(r["would_act"] for r in rows) / n, 4) if n else 0.0,
        "rows": rows,
    }