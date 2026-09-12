"""Turn free-form VLM text into structured data, defensively.

Small VLMs wrap JSON in code fences, truncate long lists when they hit the token
limit, swap key names, or emit bare coordinate lists. Each function here accepts
all of those and never raises on bad model output.
"""
from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)
_OBJ = re.compile(r"\{[^{}]*\}", re.S)
_NUM = r"(-?\d+(?:\.\d+)?)"
_QUAD = re.compile(rf"[\[\(]\s*{_NUM}\s*,\s*{_NUM}\s*,\s*{_NUM}\s*,\s*{_NUM}\s*[\]\)]")
_PAIR = re.compile(rf"[\[\(]\s*{_NUM}\s*,\s*{_NUM}\s*[\]\)]")

BOX_KEYS = ("bbox_2d", "bbox", "box", "bounding_box", "coordinates", "box_2d")
POINT_KEYS = ("point_2d", "point", "click", "coordinate")
TEXT_KEYS = ("text_content", "text", "content", "ocr", "value")
LABEL_KEYS = ("label", "name", "description", "type", "target")


def extract_json(text: str) -> Any | None:
    """Best-effort JSON extraction. Returns None when nothing parses.

    The tricky case is a list truncated by the token cap. `raw_decode` on
    '[{...}, {...' parses only the FIRST element and returns it as a bare dict,
    silently discarding every later complete object. So whenever the source
    looked like a list but we got a single object back, prefer the salvage scan.
    """
    if not text:
        return None
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    dec = json.JSONDecoder()
    for cand in candidates:
        cand = cand.strip()
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
        for i, ch in enumerate(cand):
            if ch not in "[{":
                continue
            try:
                val = dec.raw_decode(cand[i:])[0]
            except json.JSONDecodeError:
                continue
            if isinstance(val, dict) and ch == "[":
                # Opened a list, got one object: the list was cut short.
                return _salvage_objects(cand)
            return val
    return _salvage_objects(text) or None


def _salvage_objects(text: str) -> list:
    """Recover every complete {...} from output truncated by the token cap."""
    objs = []
    for m in _OBJ.finditer(text):
        try:
            objs.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            continue
    return objs


def _as_items(data: Any) -> list[dict]:
    if data is None:
        return []
    if isinstance(data, dict):
        for k in ("items", "elements", "results", "objects", "boxes", "steps", "lines"):
            if isinstance(data.get(k), list):
                return [d for d in data[k] if isinstance(d, (dict, list))]
        return [data]
    if isinstance(data, list):
        # A single bare box like [x1, y1, x2, y2]
        if len(data) == 4 and all(isinstance(v, (int, float)) for v in data):
            return [{"bbox_2d": data}]
        return [d for d in data if isinstance(d, (dict, list))]
    return []


def _first(d: dict, keys) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _nums(v: Any, n: int) -> list[float] | None:
    if isinstance(v, dict):
        v = ([v.get(k) for k in ("x1", "y1", "x2", "y2")] if n == 4
             else [v.get("x"), v.get("y")])
    if isinstance(v, str):
        v = re.findall(_NUM, v)
    if not isinstance(v, (list, tuple)):
        return None
    # Flatten [[x1,y1],[x2,y2]] before the length check -- otherwise a nested
    # box has len 2, fails the n==4 test, and gets misread as a point.
    if (n == 4 and len(v) == 2
            and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in v)):
        v = [*v[0], *v[1]]
    if len(v) >= n:
        try:
            return [float(x) for x in v[:n]]
        except (TypeError, ValueError):
            return None
    return None


def parse_boxes(text: str) -> list[dict]:
    """Return [{"box": (x1,y1,x2,y2), "label": str, "text": str}, ...].

    Points become tiny boxes flagged with ``"point": True``. Coordinates stay in
    MODEL space; the caller maps them to pixels.
    """
    out: list[dict] = []
    for item in _as_items(extract_json(text)):
        if isinstance(item, list):
            item = {"bbox_2d": item}
        box = _nums(_first(item, BOX_KEYS), 4)
        is_point = False
        if box is None:
            pt = _nums(_first(item, POINT_KEYS), 2)
            if pt is None:
                continue
            box, is_point = [pt[0] - 1, pt[1] - 1, pt[0] + 1, pt[1] + 1], True
        x1, y1, x2, y2 = box
        entry = {
            "box": (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)),
            "label": str(_first(item, LABEL_KEYS) or ""),
            "text": str(_first(item, TEXT_KEYS) or ""),
        }
        if is_point:
            entry["point"] = True
        out.append(entry)
    if out:
        return out
    # Regex fallback: "The button is at (10, 20, 60, 44)".
    for m in _QUAD.finditer(text or ""):
        x1, y1, x2, y2 = (float(g) for g in m.groups())
        out.append({"box": (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)),
                    "label": "", "text": ""})
    if not out:
        for m in _PAIR.finditer(text or ""):
            x, y = (float(g) for g in m.groups())
            out.append({"box": (x - 1, y - 1, x + 1, y + 1), "label": "", "text": "", "point": True})
    return out


def parse_yes_no(text: str) -> bool | None:
    """None means the model did not actually answer, which we treat as abstain."""
    t = (text or "").strip().lower()
    has_yes = bool(re.search(r"\byes\b|\btrue\b|\bcorrect\b", t))
    has_no = bool(re.search(r"\bno\b|\bfalse\b|\bincorrect\b", t))
    # Check ambiguity FIRST: a prefix match on "yes and no" would otherwise
    # read as a confident yes.
    if has_yes and has_no:
        return None
    m = re.match(r"^\W*(yes|no|true|false|correct|incorrect)\b", t)
    if m:
        return m.group(1) in ("yes", "true", "correct")
    if has_yes != has_no:
        return has_yes
    return None


def parse_steps(text: str) -> list[dict]:
    """Parse a guide plan into [{"instruction": str, "target": str|None}]."""
    steps = []
    for item in _as_items(extract_json(text)):
        if not isinstance(item, dict):
            continue
        instr = _first(item, ("instruction", "step", "action", "text", "description"))
        if not instr:
            continue
        target = _first(item, ("target", "element", "ui_element", "click"))
        steps.append({"instruction": str(instr).strip(),
                      "target": (str(target).strip() or None) if target else None})
    if steps:
        return steps
    for line in (text or "").splitlines():
        m = re.match(r"^\s*(?:\d+[.)]|[-*])\s+(.*\S)", line)
        if m:
            steps.append({"instruction": m.group(1), "target": None})
    return steps