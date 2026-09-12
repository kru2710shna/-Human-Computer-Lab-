"""The user-facing tasks, all built on pipeline.Engine.

Every task returns a plain dict so the CLI, the HTML report, and the benchmark can
consume the same structure. Coordinates in returned dicts are always ORIGINAL
screenshot pixels.

One structural note: the backends take a single image per call, so `diff` composes
the two screenshots into one side-by-side canvas and maps the model's boxes back to
whichever panel they landed in. That keeps the whole thing inside one forward pass
instead of needing a model that accepts image pairs.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from PIL import Image, ImageDraw, ImageFilter

from . import imaging as im
from .parsing import extract_json, parse_boxes, parse_steps
from .pipeline import GROUND_SYSTEM, Engine, Hit

Box = tuple[float, float, float, float]


# --------------------------------------------------------------------- describe
DESCRIBE_PROMPT = """Analyse this screenshot and reply with JSON only, no prose:
{
  "app": "<the application or website, or 'unknown'>",
  "screen": "<what screen or view this is>",
  "summary": "<one sentence on what the user is looking at>",
  "state": ["<notable state: unsaved changes, error shown, modal open, ...>"],
  "actions": ["<the main things a user could do from here>"],
  "regions": [{"name": "<e.g. sidebar, toolbar, main content>", "contains": "<short>"}]
}
Report only what is actually visible."""


def describe(engine: Engine, img: Image.Image, max_new_tokens: int = 600) -> dict:
    view = engine.view(img)
    raw = engine.ask(view, DESCRIBE_PROMPT, max_new_tokens=max_new_tokens)
    data = extract_json(raw)
    if not isinstance(data, dict):
        # Model answered in prose; keep it rather than discarding a usable answer.
        data = {"app": "unknown", "screen": "unknown", "summary": raw.strip()[:600],
                "state": [], "actions": [], "regions": []}
    return {
        "task": "describe",
        "image_size": list(img.size),
        "view_size": list(view.size),
        "result": data,
        "raw": raw,
    }


# -------------------------------------------------------------------------- ocr
OCR_PROMPT = """Read all visible text in this image.
Reply with JSON only:
[{"bbox_2d": [x1, y1, x2, y2], "text": "<exact text of this line>"}]
One entry per line of text. Preserve capitalisation and punctuation exactly.
Do not translate, summarise, or invent text."""


def ocr(engine: Engine, img: Image.Image, tiles: int = 1, min_chars: int = 1,
        max_new_tokens: int = 1400) -> dict:
    """Read text with boxes.

    `tiles` > 1 splits the screen into an NxN grid and reads each tile upscaled,
    which recovers small text a single downscaled pass misses. Cost is N^2 calls,
    so it defaults off.
    """
    items: list[dict] = []
    if tiles <= 1:
        view = engine.view(img)
        items = engine.boxes_on(view, OCR_PROMPT, max_new_tokens=max_new_tokens)
    else:
        w, h = img.size
        tw, th = w / tiles, h / tiles
        ox, oy = tw * 0.08, th * 0.08  # overlap so seam-straddling lines survive
        for r in range(tiles):
            for c in range(tiles):
                region = im.clamp_box((c * tw - ox, r * th - oy,
                                       (c + 1) * tw + ox, (r + 1) * th + oy), w, h)
                tile = engine.view(im.crop(img, region), upscale_to=engine.spec.max_pixels)
                for it in engine.boxes_on(tile, OCR_PROMPT, max_new_tokens=max_new_tokens):
                    if im.coverage(it["box"], region) >= 0.6:
                        items.append(it)

    lines = [{"box": it["box"], "text": (it.get("text") or it.get("label") or "").strip()}
             for it in items]
    lines = [l for l in lines if len(l["text"]) >= min_chars]
    lines = _dedupe_text(lines)
    lines = im.reading_order(lines, key=lambda l: l["box"])
    return {
        "task": "ocr",
        "image_size": list(img.size),
        "tiles": tiles,
        "lines": [{"box": [round(v, 1) for v in l["box"]], "text": l["text"]} for l in lines],
        "text": "\n".join(l["text"] for l in lines),
    }


def _dedupe_text(lines: list[dict], iou_thresh: float = 0.6) -> list[dict]:
    """Overlapping tiles produce the same line twice; keep the longer reading."""
    kept: list[dict] = []
    for line in sorted(lines, key=lambda l: -len(l["text"])):
        if any(im.iou(line["box"], k["box"]) > iou_thresh or
               (line["text"] and line["text"] == k["text"] and
                im.iou(line["box"], k["box"]) > 0.2)
               for k in kept):
            continue
        kept.append(line)
    return kept


# ------------------------------------------------------------------------- find
def find(engine: Engine, img: Image.Image, target: str, all_matches: bool = False,
         fallback_tiles: bool = True, verify: bool = True) -> dict:
    if all_matches:
        hits = engine.ground_all(img, target, verify=verify)
        return {"task": "find", "target": target, "image_size": list(img.size),
                "hits": [h.to_dict() for h in hits], "count": len(hits)}

    hit = engine.locate(img, target, fallback_tiles=fallback_tiles)
    return {
        "task": "find",
        "target": target,
        "image_size": list(img.size),
        "hits": [hit.to_dict()] if hit else [],
        "count": 1 if hit else 0,
        "found": hit is not None,
        "accepted": bool(hit and (hit.confidence is None or
                                  hit.confidence >= engine.cfg.accept_threshold)),
        "trace": hit.trace if hit else [],
    }


# ------------------------------------------------------------------------ guide
GUIDE_PROMPT = """The user asks: "{question}"

Looking at this screenshot, give the steps to accomplish that from this screen.
Reply with JSON only:
[{{"instruction": "<what the user does>", "target": "<the exact visible UI element to click, or null if no click>"}}]

Rules:
- Only reference elements that are actually visible in this screenshot.
- If the first step opens a menu, stop there: you cannot see inside a menu that is not open.
- At most {max_steps} steps."""


def guide(engine: Engine, img: Image.Image, question: str, max_steps: int = 6,
          ground_steps: bool = True) -> dict:
    """Plan steps, then ground each step's target to a real box on screen.

    The grounding pass is the honest part: a plan that names a button which is not
    on screen gets caught here and marked unresolved, rather than being presented
    as a confident instruction.
    """
    view = engine.view(img)
    raw = engine.ask(view, GUIDE_PROMPT.format(question=question, max_steps=max_steps),
                     max_new_tokens=700)
    steps = parse_steps(raw)[:max_steps]

    out_steps = []
    for i, step in enumerate(steps, 1):
        entry: dict[str, Any] = {"n": i, "instruction": step["instruction"],
                                 "target": step.get("target")}
        tgt = step.get("target")
        if ground_steps and tgt and tgt.lower() not in ("null", "none", ""):
            hit = engine.ground(img, tgt)
            if hit:
                entry["hit"] = hit.to_dict()
                entry["resolved"] = (hit.confidence is None or
                                     hit.confidence >= engine.cfg.accept_threshold)
            else:
                entry["hit"] = None
                entry["resolved"] = False
        else:
            entry["resolved"] = None  # nothing to ground
        out_steps.append(entry)

    resolved = sum(1 for s in out_steps if s.get("resolved") is True)
    groundable = sum(1 for s in out_steps if s.get("resolved") is not None)
    return {
        "task": "guide",
        "question": question,
        "image_size": list(img.size),
        "steps": out_steps,
        "resolved": resolved,
        "groundable": groundable,
        "raw": raw,
    }


# ------------------------------------------------------------------------- diff
DIFF_PROMPT = """This image shows two screenshots side by side, separated by a vertical line.
LEFT is BEFORE. RIGHT is AFTER.

Describe what changed. Reply with JSON only:
{"changes": [{"bbox_2d": [x1, y1, x2, y2], "side": "left"|"right", "kind": "added"|"removed"|"changed", "description": "<short>"}],
 "summary": "<one sentence>"}
Use coordinates in this combined image. Report meaningful UI changes, not
anti-aliasing or cursor position."""


def _compose(before: Image.Image, after: Image.Image, gap: int = 16) -> tuple[Image.Image, int, float]:
    """Side-by-side canvas. Returns (canvas, left_panel_width, scale_applied)."""
    h = max(before.size[1], after.size[1])
    scale = 1.0
    bw, ah = before.size[0], after.size[0]
    canvas = Image.new("RGB", (bw + gap + ah, h), (24, 24, 27))
    canvas.paste(before.convert("RGB"), (0, 0))
    canvas.paste(after.convert("RGB"), (bw + gap, 0))
    d = ImageDraw.Draw(canvas)
    d.line([(bw + gap // 2, 0), (bw + gap // 2, h)], fill=(220, 80, 80), width=3)
    return im.set_tf(canvas, (0.0, 0.0, 1.0, 1.0)), bw + gap, scale


def diff(engine: Engine, before: Image.Image, after: Image.Image,
         max_new_tokens: int = 900) -> dict:
    canvas, right_x0, _ = _compose(before, after)
    view = engine.view(canvas)
    raw = engine.ask(view, DIFF_PROMPT, max_new_tokens=max_new_tokens)

    data = extract_json(raw)
    summary = data.get("summary", "") if isinstance(data, dict) else ""
    items = engine.boxes_on(view, DIFF_PROMPT) if False else []
    # Re-parse from the raw text we already have, rather than re-calling the model.
    parsed = parse_boxes(raw)
    vw, vh = view.size
    changes = []
    for item in parsed:
        px = im.clamp_box(im.model_to_pixels(view, item["box"], engine.spec.coord_mode), vw, vh)
        canvas_box = im.to_original(view, px)
        cx = im.center(canvas_box)[0]
        side = "left" if cx < right_x0 else "right"
        if side == "left":
            local = canvas_box
        else:
            local = (canvas_box[0] - right_x0, canvas_box[1],
                     canvas_box[2] - right_x0, canvas_box[3])
            local = im.clamp_box(local, after.size[0], after.size[1])
        changes.append({
            "side": side,
            "box": [round(v, 1) for v in local],
            "canvas_box": [round(v, 1) for v in canvas_box],
            "description": item.get("label") or item.get("text") or "",
        })
    return {
        "task": "diff",
        "before_size": list(before.size),
        "after_size": list(after.size),
        "summary": summary,
        "changes": changes,
        "count": len(changes),
        "raw": raw,
        "_canvas": canvas,  # used by render; stripped before JSON serialisation
    }


# ----------------------------------------------------------------------- redact
PII_KINDS = {
    "email": "email addresses",
    "phone": "phone numbers",
    "name": "personal names of individuals",
    "address": "street or postal addresses",
    "card": "credit card or bank account numbers",
    "key": "API keys, access tokens, passwords, or secret strings",
    "id": "government ID numbers such as SSN or passport numbers",
}

REDACT_PROMPT = """Find every piece of sensitive personal information visible in this screenshot.
Look for: {kinds}.

Reply with JSON only:
[{{"bbox_2d": [x1, y1, x2, y2], "label": "<kind>", "text_content": "<the sensitive text>"}}]
Box only the sensitive value itself, not the whole row or its field label.
If there is none, reply with []."""


def redact(engine: Engine, img: Image.Image, kinds: list[str] | None = None,
           method: str = "blur", pad: float = 3.0, verify: bool = False,
           max_new_tokens: int = 900) -> dict:
    """Locate and mask sensitive values. Everything stays on this machine.

    Limitation worth stating: this is a VLM reading pixels, not a validated PII
    detector. It will miss things and it will flag things that are not sensitive.
    Treat the output as a first pass a human reviews, never as a compliance control.
    """
    kinds = kinds or list(PII_KINDS)
    desc = ", ".join(PII_KINDS.get(k, k) for k in kinds)
    view = engine.view(img)
    items = engine.boxes_on(view, REDACT_PROMPT.format(kinds=desc),
                            max_new_tokens=max_new_tokens, system=GROUND_SYSTEM)

    findings = []
    for it in items:
        box = im.expand_box(it["box"], img.size[0], img.size[1], scale=1.0, pad=pad)
        entry = {"box": [round(v, 1) for v in box],
                 "kind": (it.get("label") or "unknown").strip().lower(),
                 "text": it.get("text", "")}
        if verify:
            p = engine.verify(img, box, f"sensitive personal information ({entry['kind']})")
            entry["confidence"] = None if p is None else round(p, 4)
            if p is not None and p < engine.cfg.accept_threshold:
                entry["kept"] = False
                findings.append(entry)
                continue
        entry["kept"] = True
        findings.append(entry)

    keep = [f for f in findings if f.get("kept", True)]
    out = apply_masks(img, [tuple(f["box"]) for f in keep], method=method)
    return {
        "task": "redact",
        "image_size": list(img.size),
        "method": method,
        "findings": findings,
        "redacted_count": len(keep),
        "_image": out,  # stripped before JSON serialisation
    }


def apply_masks(img: Image.Image, boxes: list[Box], method: str = "blur") -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for box in boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in im.clamp_box(box, *img.size))
        if x2 <= x1 or y2 <= y1:
            continue
        if method == "blur":
            region = out.crop((x1, y1, x2, y2))
            radius = max(6, (x2 - x1) // 8)
            out.paste(region.filter(ImageFilter.GaussianBlur(radius)), (x1, y1))
        elif method == "pixelate":
            region = out.crop((x1, y1, x2, y2))
            small = region.resize((max(1, (x2 - x1) // 12), max(1, (y2 - y1) // 12)),
                                  Image.Resampling.BILINEAR)
            out.paste(small.resize(region.size, Image.Resampling.NEAREST), (x1, y1))
        else:  # solid
            draw.rectangle([x1, y1, x2, y2], fill=(18, 18, 20))
    return im.set_tf(out, im.get_tf(img))


# ------------------------------------------------------------------------- misc
def ask(engine: Engine, img: Image.Image, question: str, max_new_tokens: int = 512) -> dict:
    """Free-form VQA escape hatch."""
    view = engine.view(img)
    raw = engine.ask(view, question, max_new_tokens=max_new_tokens)
    return {"task": "ask", "question": question, "answer": raw, "image_size": list(img.size)}


def strip_images(payload: dict) -> dict:
    """Remove PIL objects so a result can be JSON-serialised."""
    return {k: v for k, v in payload.items() if not k.startswith("_")}