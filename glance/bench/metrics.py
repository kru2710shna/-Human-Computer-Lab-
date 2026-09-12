"""Scoring.

Two metrics, because they answer different questions:

  * CLICK ACCURACY -- does the predicted centre land inside the true box? This is
    what matters for acting on a screen. A loose box whose centre is correct is a
    success; you click the right thing.
  * IoU -- how well does the predicted box match the true extent? This is what
    matters for cropping, redaction, and highlighting.

Our earlier Save-button run scored IoU 0.81 with a correct click. Reporting only
IoU would understate it; reporting only click accuracy would hide that the right
edge was 31px short. Both, always.
"""

from __future__ import annotations

from statistics import mean, median

from ..imaging import area, center, iou, point_in


def score_one(pred_box, true_box) -> dict:
    return {
        "iou": round(iou(pred_box, true_box), 4),
        "click_hit": point_in(center(pred_box), true_box),
        "centre_error_px": round(
            (
                (center(pred_box)[0] - center(true_box)[0]) ** 2
                + (center(pred_box)[1] - center(true_box)[1]) ** 2
            )
            ** 0.5,
            1,
        ),
        "area_ratio": round(area(pred_box) / max(1.0, area(true_box)), 3),
    }


def aggregate(rows: list[dict]) -> dict:
    """rows: per-trial dicts with keys found, click_hit, iou, latency_s, ..."""
    n = len(rows)
    if not n:
        return {"n": 0}
    found = [r for r in rows if r.get("found")]
    hits = [r for r in found if r.get("click_hit")]
    ious = [r["iou"] for r in found]
    lats = sorted(r["latency_s"] for r in rows)

    out = {
        "n": n,
        "found_rate": round(len(found) / n, 4),
        "click_accuracy": round(len(hits) / n, 4),
        "click_accuracy_when_found": round(len(hits) / len(found), 4) if found else 0.0,
        "iou_mean": round(mean(ious), 4) if ious else 0.0,
        "iou_median": round(median(ious), 4) if ious else 0.0,
        "iou_at_50": round(sum(1 for v in ious if v >= 0.5) / n, 4),
        "iou_at_75": round(sum(1 for v in ious if v >= 0.75) / n, 4),
        "latency_s_mean": round(mean(lats), 2),
        "latency_s_median": round(median(lats), 2),
        "latency_s_p90": round(lats[min(n - 1, int(0.9 * n))], 2),
    }
    # IoU is only meaningful where our ground-truth box is the element itself.
    # See Element.box_kind in bench/dataset.py for why.
    tight = [r for r in found if r.get("box_kind") == "tight"]
    if tight:
        t_ious = [r["iou"] for r in tight]
        out["tight_n"] = len(tight)
        out["tight_iou_mean"] = round(mean(t_ious), 4)
        out["tight_iou_at_50"] = round(
            sum(1 for v in t_ious if v >= 0.5) / len(tight), 4
        )
        out["tight_iou_at_75"] = round(
            sum(1 for v in t_ious if v >= 0.75) / len(tight), 4
        )
    if found:
        out["centre_error_px_median"] = round(
            median(r["centre_error_px"] for r in found), 1
        )
    return out


def by_group(rows: list[dict], key: str) -> dict:
    groups: dict = {}
    for r in rows:
        groups.setdefault(r.get(key), []).append(r)
    return {
        str(k): aggregate(v)
        for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))
    }


def confidence_analysis(
    rows: list[dict], thresholds=(0.0, 0.5, 0.75, 0.9)
) -> list[dict]:
    """Accuracy vs abstention: if we only act above a confidence threshold, how
    often are we right, and how often do we refuse to act?

    This is the honest way to present a confidence number. A model that is 70%
    accurate but knows which 70% is more useful than one that is 75% accurate and
    equally sure about everything.
    """
    scored = [r for r in rows if r.get("confidence") is not None]
    out = []
    for t in thresholds:
        acted = [r for r in scored if r["confidence"] >= t]
        hits = [r for r in acted if r.get("click_hit")]
        out.append(
            {
                "threshold": t,
                "acted_on": len(acted),
                "abstained": len(scored) - len(acted),
                "coverage": round(len(acted) / len(scored), 4) if scored else 0.0,
                "precision": round(len(hits) / len(acted), 4) if acted else 0.0,
            }
        )
    return out


def ocr_score(pred_lines: list[dict], true_texts: list[dict]) -> dict:
    """Text recall by normalised string match. Deliberately lenient on boxes:
    we are measuring reading, not localisation, and our synthetic text boxes are
    tight glyph bounds that no model would reproduce exactly."""
    norm = lambda s: "".join(ch.lower() for ch in s if ch.isalnum())
    got = {norm(l["text"]) for l in pred_lines if norm(l["text"])}
    want = [norm(t["text"]) for t in true_texts if norm(t["text"])]
    found = sum(1 for w in want if any(w == g or (len(w) > 3 and w in g) for g in got))
    return {
        "lines_expected": len(want),
        "lines_returned": len(got),
        "recall": round(found / len(want), 4) if want else 0.0,
        "spurious": max(0, len(got) - found),
    }
