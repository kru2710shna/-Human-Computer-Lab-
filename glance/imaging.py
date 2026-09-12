"""Image I/O and coordinate bookkeeping.

Every image that flows through Glance carries an affine transform in
``img.info["glance_tf"] = (ox, oy, sx, sy)`` that maps its pixel coordinates back
to the ORIGINAL screenshot:  orig_x = ox + x * sx,  orig_y = oy + y * sy.

Resizing and cropping update this transform, so any box a model predicts on any
derived view (resized, zoomed crop, verification crop) can be mapped back to the
user's screenshot exactly, independent of backend.
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image

Box = tuple[float, float, float, float]

_TF_KEY = "glance_tf"


# ---------------------------------------------------------------- transforms
def get_tf(img: Image.Image) -> tuple[float, float, float, float]:
    return tuple(img.info.get(_TF_KEY, (0.0, 0.0, 1.0, 1.0)))


def set_tf(img: Image.Image, tf) -> Image.Image:
    img.info[_TF_KEY] = tuple(float(v) for v in tf)
    return img


def to_original(img: Image.Image, box: Box) -> Box:
    ox, oy, sx, sy = get_tf(img)
    x1, y1, x2, y2 = box
    return (ox + x1 * sx, oy + y1 * sy, ox + x2 * sx, oy + y2 * sy)


def from_original(img: Image.Image, box: Box) -> Box:
    ox, oy, sx, sy = get_tf(img)
    x1, y1, x2, y2 = box
    return ((x1 - ox) / sx, (y1 - oy) / sy, (x2 - ox) / sx, (y2 - oy) / sy)


def model_to_pixels(img: Image.Image, box: Box, coord_mode: str) -> Box:
    """Convert a model-space box to pixel space of the image the model saw."""
    if coord_mode == "abs":
        return box
    if coord_mode == "rel1000":
        w, h = img.size
        x1, y1, x2, y2 = box
        return (x1 * w / 1000, y1 * h / 1000, x2 * w / 1000, y2 * h / 1000)
    raise ValueError(f"unknown coord_mode {coord_mode}")


def pixels_to_model(img: Image.Image, box: Box, coord_mode: str) -> Box:
    if coord_mode == "abs":
        return box
    w, h = img.size
    x1, y1, x2, y2 = box
    return (x1 * 1000 / w, y1 * 1000 / h, x2 * 1000 / w, y2 * 1000 / h)


# ------------------------------------------------------------------- loading
def load_image(path: str | Path) -> Image.Image:
    img = Image.open(path)
    img.load()
    if img.mode != "RGB":
        # Flatten transparency onto white; screenshots with alpha are common.
        if img.mode in ("RGBA", "LA", "P"):
            rgba = img.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[-1])
            img = bg
        else:
            img = img.convert("RGB")
    return set_tf(img, (0.0, 0.0, 1.0, 1.0))


def capture_screen(monitor: int = 1) -> Image.Image:
    """Grab a screenshot locally. Needs `pip install mss`; on macOS grant your
    terminal 'Screen Recording' permission in System Settings > Privacy."""
    try:
        import mss
    except ImportError as e:
        raise RuntimeError("Screen capture needs the 'mss' package: pip install mss") from e
    with mss.mss() as sct:
        mons = sct.monitors
        if monitor >= len(mons):
            raise RuntimeError(f"Monitor {monitor} not found; {len(mons) - 1} available")
        shot = sct.grab(mons[monitor])
        img = Image.frombytes("RGB", shot.size, shot.rgb)
    return set_tf(img, (0.0, 0.0, 1.0, 1.0))


# ------------------------------------------------------------------ resizing
def fit_dims(w: int, h: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """The same rule the Qwen-VL processor applies ("smart resize"). Feeding an
    image that already satisfies it makes the processor's own resize a no-op, so
    we know exactly which pixels the model saw on every backend."""
    h_bar = max(factor, round(h / factor) * factor)
    w_bar = max(factor, round(w / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return int(w_bar), int(h_bar)


def prepare(img: Image.Image, factor: int, min_pixels: int, max_pixels: int,
            upscale_to: int | None = None) -> Image.Image:
    """Resize to model-friendly dims and update the coordinate transform.

    ``upscale_to`` lets zoom crops be enlarged towards a pixel budget, which is
    what gives the refinement pass its extra detail.
    """
    w, h = img.size
    lo = min_pixels
    if upscale_to:
        lo = max(lo, min(upscale_to, max_pixels))
    nw, nh = fit_dims(w, h, factor, lo, max_pixels)
    ox, oy, sx, sy = get_tf(img)
    out = img if (nw, nh) == (w, h) else img.resize((nw, nh), Image.Resampling.BICUBIC)
    out = out.copy() if out is img else out
    return set_tf(out, (ox, oy, sx * w / nw, sy * h / nh))


def crop(img: Image.Image, box: Box) -> Image.Image:
    """Crop in the image's own pixel space; keeps the mapping to the original."""
    w, h = img.size
    x1, y1, x2, y2 = clamp_box(box, w, h)
    x1, y1 = int(math.floor(x1)), int(math.floor(y1))
    x2, y2 = int(math.ceil(x2)), int(math.ceil(y2))
    x2, y2 = max(x2, x1 + 1), max(y2, y1 + 1)
    ox, oy, sx, sy = get_tf(img)
    out = img.crop((x1, y1, x2, y2))
    return set_tf(out, (ox + x1 * sx, oy + y1 * sy, sx, sy))


# ---------------------------------------------------------------- box maths
def clamp_box(box: Box, w: float, h: float) -> Box:
    x1, y1, x2, y2 = box
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return (max(0.0, min(x1, w)), max(0.0, min(y1, h)),
            max(0.0, min(x2, w)), max(0.0, min(y2, h)))


def expand_box(box: Box, w: float, h: float, scale: float = 1.0, pad: float = 0.0,
               min_size: float = 0.0) -> Box:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw = max((x2 - x1) * scale + 2 * pad, min_size)
    bh = max((y2 - y1) * scale + 2 * pad, min_size)
    return clamp_box((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), w, h)


def center(box: Box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = area((ix1, iy1, ix2, iy2))
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def coverage(inner: Box, outer: Box) -> float:
    """Fraction of `inner` covered by `outer`."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    a = area(inner)
    return area((ix1, iy1, ix2, iy2)) / a if a > 0 else 0.0


def point_in(pt: tuple[float, float], box: Box, tol: float = 0.0) -> bool:
    return box[0] - tol <= pt[0] <= box[2] + tol and box[1] - tol <= pt[1] <= box[3] + tol


def reading_order(items: list, key=lambda it: it["box"]) -> list:
    """Sort boxes top-to-bottom, then left-to-right within visual rows."""
    if not items:
        return items
    heights = sorted(max(1.0, key(it)[3] - key(it)[1]) for it in items)
    tol = heights[len(heights) // 2] * 0.5
    rows: list[list] = []
    for it in sorted(items, key=lambda it: center(key(it))[1]):
        cy = center(key(it))[1]
        if rows and abs(center(key(rows[-1][0]))[1] - cy) <= tol:
            rows[-1].append(it)
        else:
            rows.append([it])
    out = []
    for row in rows:
        out.extend(sorted(row, key=lambda it: key(it)[0]))
    return out