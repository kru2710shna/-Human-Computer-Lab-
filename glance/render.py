"""Visual output: annotated PNGs and a self-contained HTML report.

The report embeds the image as base64 and inlines all CSS, so a run is a single
file you can move anywhere. No CDN, no network — which matters for a tool whose
whole premise is that nothing leaves the machine.
"""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import imaging as im

Box = tuple[float, float, float, float]

# Confidence -> colour. Deliberately three bands, not a gradient: a continuous
# ramp implies a precision this probability does not have.
_HIGH = (64, 196, 140)
_MID = (232, 176, 72)
_LOW = (226, 92, 92)
_PLAIN = (96, 152, 232)


def _font(size: int):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _colour(conf: float | None):
    if conf is None:
        return _PLAIN
    if conf >= 0.75:
        return _HIGH
    if conf >= 0.5:
        return _MID
    return _LOW


def annotate(img: Image.Image, items: list[dict], show_click: bool = True,
             show_label: bool = True, width: int | None = None) -> Image.Image:
    """Draw boxes from a task result. Each item needs "box"; may have "label",
    "text", "confidence", "click"."""
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out, "RGBA")
    scale = max(out.size) / 1200
    lw = width or max(2, int(round(2.5 * scale)))
    fsize = max(11, int(round(13 * scale)))
    font = _font(fsize)

    for i, item in enumerate(items, 1):
        box = tuple(float(v) for v in item["box"])
        conf = item.get("confidence")
        col = _colour(conf)
        x1, y1, x2, y2 = im.clamp_box(box, *out.size)
        draw.rectangle([x1, y1, x2, y2], outline=col + (255,), width=lw)
        draw.rectangle([x1, y1, x2, y2], fill=col + (28,))

        if show_click:
            cx, cy = item.get("click") or im.center(box)
            r = max(3, lw + 1)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col + (255,))
            draw.ellipse([cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2],
                         outline=(255, 255, 255, 220), width=1)

        if show_label:
            label = item.get("label") or item.get("text") or item.get("description") or f"#{i}"
            if conf is not None:
                label = f"{label}  {conf:.2f}"
            label = label[:60]
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            pad = max(3, int(3 * scale))
            ly = y1 - th - 2 * pad
            above = ly >= 0
            if not above:
                ly = y1
            draw.rectangle([x1, ly, x1 + tw + 2 * pad, ly + th + 2 * pad], fill=col + (235,))
            draw.text((x1 + pad, ly + pad), label, fill=(16, 16, 18), font=font)
    return out


def save_annotated(img: Image.Image, items: list[dict], path: str | Path, **kw) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    annotate(img, items, **kw).save(p)
    return p


def _b64(img: Image.Image, max_side: int = 1800) -> str:
    im2 = img.convert("RGB")
    if max(im2.size) > max_side:
        r = max_side / max(im2.size)
        im2 = im2.resize((int(im2.size[0] * r), int(im2.size[1] * r)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im2.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


_CSS = """
:root{--ink:#e8e6e3;--dim:#8f8b85;--line:#2e2c2a;--bg:#151413;--panel:#1c1b1a;
--ok:#40c48c;--mid:#e8b048;--low:#e25c5c;--blue:#6098e8}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 "Helvetica Neue",Helvetica,Arial,sans-serif;padding:32px 24px}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:26px;font-weight:600;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--dim);font-size:14px;margin:0 0 28px}
h2{font-size:15px;font-weight:600;margin:32px 0 10px;padding-bottom:8px;
border-bottom:1px solid var(--line)}
img.shot{width:100%;border:1px solid var(--line);border-radius:3px;display:block}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;
background:var(--line);border:1px solid var(--line);margin:16px 0}
.cell{background:var(--panel);padding:12px 14px}
.cell .k{color:var(--dim);font-size:12px;margin-bottom:3px}
.cell .v{font-size:19px;font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th{text-align:left;color:var(--dim);font-weight:500;padding:7px 10px;
border-bottom:1px solid var(--line)}
td{padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
td.num{font-variant-numeric:tabular-nums;color:var(--dim);white-space:nowrap}
.chip{display:inline-block;padding:1px 7px;border-radius:2px;font-size:12px}
.ok{background:rgba(64,196,140,.16);color:var(--ok)}
.mid{background:rgba(232,176,72,.16);color:var(--mid)}
.low{background:rgba(226,92,92,.16);color:var(--low)}
pre{background:var(--panel);border:1px solid var(--line);padding:14px;
overflow-x:auto;font-size:12.5px;line-height:1.5;border-radius:3px}
details{margin-top:10px}
summary{cursor:pointer;color:var(--dim);font-size:13.5px;padding:5px 0}
.note{color:var(--dim);font-size:13px;margin:10px 0}
"""


def _chip(conf) -> str:
    if conf is None:
        return '<span class="chip">n/a</span>'
    cls = "ok" if conf >= 0.75 else ("mid" if conf >= 0.5 else "low")
    return f'<span class="chip {cls}">{conf:.2f}</span>'


def _cells(pairs) -> str:
    body = "".join(f'<div class="cell"><div class="k">{html.escape(str(k))}</div>'
                   f'<div class="v">{html.escape(str(v))}</div></div>' for k, v in pairs)
    return f'<div class="grid">{body}</div>'


def report_html(path: str | Path, title: str, image: Image.Image | None,
                result: dict, items: list[dict] | None = None,
                meta: dict | None = None, extra_sections: list[tuple[str, str]] | None = None
                ) -> Path:
    """Write a single self-contained HTML file."""
    items = items if items is not None else _items_from(result)
    shot = annotate(image, items) if (image is not None and items) else image

    parts = [f'<h1>{html.escape(title)}</h1>']
    sub = result.get("target") or result.get("question") or result.get("summary") or ""
    parts.append(f'<p class="sub">{html.escape(str(sub))[:200]}</p>')

    if meta:
        flat = []
        for k, v in meta.items():
            if isinstance(v, (dict, list)):
                continue
            flat.append((k.replace("_", " "), v))
        if flat:
            parts.append(_cells(flat))

    if shot is not None:
        parts.append('<h2>Screen</h2>')
        parts.append(f'<img class="shot" src="data:image/png;base64,{_b64(shot)}" alt="">')

    if items:
        rows = []
        for i, it in enumerate(items, 1):
            box = ", ".join(str(int(round(v))) for v in it["box"])
            lbl = it.get("label") or it.get("text") or it.get("description") or ""
            rows.append(f'<tr><td class="num">{i}</td><td>{html.escape(str(lbl))[:180]}</td>'
                        f'<td class="num">{box}</td><td>{_chip(it.get("confidence"))}</td></tr>')
        parts.append('<h2>Detections</h2><table><tr><th>#</th><th>Element</th>'
                     '<th>Box (x1, y1, x2, y2)</th><th>Confidence</th></tr>'
                     + "".join(rows) + '</table>')
        parts.append('<p class="note">Confidence is P(Yes) from a single verification '
                     'forward pass, renormalised over the Yes/No tokens. It is a model '
                     'belief, not a guarantee.</p>')

    for heading, body in (extra_sections or []):
        parts.append(f'<h2>{html.escape(heading)}</h2>{body}')

    clean = {k: v for k, v in result.items() if not k.startswith("_")}
    parts.append('<details><summary>Full result JSON</summary><pre>'
                 + html.escape(json.dumps(clean, indent=2, default=str)) + '</pre></details>')

    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>{html.escape(title)}</title><style>{_CSS}</style></head>'
           f'<body><div class="wrap">{"".join(parts)}</div></body></html>')

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc, encoding="utf-8")
    return p


def _items_from(result: dict) -> list[dict]:
    """Pull drawable boxes out of any task result."""
    task = result.get("task")
    if task == "find":
        return result.get("hits", [])
    if task == "ocr":
        return [{"box": l["box"], "text": l["text"]} for l in result.get("lines", [])]
    if task == "guide":
        return [dict(s["hit"], label=f'{s["n"]}. {s.get("target") or ""}')
                for s in result.get("steps", []) if s.get("hit")]
    if task == "redact":
        return [{"box": f["box"], "label": f["kind"], "confidence": f.get("confidence")}
                for f in result.get("findings", [])]
    if task == "diff":
        return [{"box": c["canvas_box"], "label": c["description"]}
                for c in result.get("changes", [])]
    return []