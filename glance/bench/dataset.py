"""Synthetic UI screens with exact ground truth.

We render the screens ourselves, so every element's box is known to the pixel.
That removes annotation noise and any licensing question about scraping real app
screenshots.

The tradeoff, stated plainly: synthetic UIs are cleaner than real ones -- uniform
fonts, high contrast, no overlapping chrome, no photographic content. Absolute
accuracy measured here will be OPTIMISTIC relative to real screenshots. What this
dataset measures reliably is the *relative* comparison between pipeline variants
on identical inputs, which is the question the benchmark exists to answer.

Difficulty is controlled by element size and contrast, so we can report accuracy
as a function of how small the target is -- the axis where a 3B model struggles.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

Box = tuple[int, int, int, int]

THEMES = {
    "light": {"bg": (246, 247, 250), "panel": (255, 255, 255), "line": (223, 226, 233),
              "ink": (32, 34, 40), "dim": (122, 128, 140), "accent": (46, 108, 220),
              "accent_ink": (255, 255, 255), "danger": (204, 64, 64)},
    "dark": {"bg": (22, 23, 26), "panel": (30, 32, 36), "line": (52, 55, 62),
             "ink": (232, 234, 238), "dim": (138, 143, 154), "accent": (86, 140, 240),
             "accent_ink": (12, 14, 18), "danger": (226, 92, 92)},
}

NOUNS = ["Report", "Invoice", "Budget", "Roadmap", "Notes", "Contract", "Summary",
         "Backup", "Archive", "Draft", "Proposal", "Minutes"]
MENUS = ["File", "Edit", "View", "Insert", "Format", "Tools", "Window", "Help"]
ACTIONS = ["Save", "Cancel", "Export", "Delete", "Share", "Rename", "Duplicate",
           "Publish", "Archive", "Upload", "Refresh", "Settings"]
SIDEBAR = ["Inbox", "Starred", "Drafts", "Sent", "Archive", "Trash", "Spam", "All files"]


def _font(size: int):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


@dataclass
class Element:
    box: Box
    role: str          # button | menu | nav_item | field | tab
    text: str
    theme_role: str    # primary | secondary | danger | plain
    height: int

    @property
    def box_kind(self) -> str:
        """Whether our ground-truth box is something a model could reproduce.

        This distinction came out of a measured surprise. On tabs, menus, and
        sidebar rows the first benchmark showed 100% click accuracy alongside 0%
        IoU@50 -- the model was finding the right element and boxing the TEXT,
        while our ground truth was the full clickable ROW or the whole menu-bar
        height. IoU there measures our labelling convention, not the model.

        "tight"  -- the drawn rectangle IS the element (buttons, input fields).
                    IoU is meaningful; report it.
        "region" -- the element is a hit-area much larger than its visible text
                    (nav rows, menu-bar items, tabs). Score these on click
                    accuracy and containment, not IoU.
        """
        return "tight" if self.role in ("button", "field") else "region"

    def descriptor(self) -> str:
        """How a user would refer to this element in natural language."""
        if self.role == "button":
            colour = {"primary": "blue ", "danger": "red ", "secondary": "", "plain": ""}
            return f'the {colour.get(self.theme_role, "")}{self.text} button'.replace("  ", " ")
        if self.role == "menu":
            return f'the "{self.text}" menu in the menu bar'
        if self.role == "nav_item":
            return f'the "{self.text}" item in the sidebar'
        if self.role == "field":
            return f'the "{self.text}" input field'
        if self.role == "tab":
            return f'the "{self.text}" tab'
        return f'the {self.text} element'

    def to_dict(self) -> dict:
        return {"box": list(self.box), "role": self.role, "text": self.text,
                "theme_role": self.theme_role, "height": self.height,
                "box_kind": self.box_kind, "descriptor": self.descriptor()}

@dataclass
class Screen:
    image: Image.Image
    elements: list[Element]
    theme: str
    size: tuple[int, int]
    seed: int
    texts: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"seed": self.seed, "theme": self.theme, "size": list(self.size),
                "elements": [e.to_dict() for e in self.elements],
                "texts": self.texts}


def _text_box(draw, xy, text, font) -> Box:
    b = draw.textbbox(xy, text, font=font)
    return (int(b[0]), int(b[1]), int(b[2]), int(b[3]))


def _rounded(draw, box, radius, fill=None, outline=None, width=1):
    try:
        draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
    except AttributeError:
        draw.rectangle(box, fill=fill, outline=outline, width=width)


def generate_screen(seed: int, size: tuple[int, int] = (1280, 800),
                    theme: str | None = None, scale: float = 1.0) -> Screen:
    """Render one app-like screen. `scale` shrinks all controls, which is the
    difficulty knob: small targets are where a downscaled pass fails."""
    rng = random.Random(seed)
    theme = theme or rng.choice(list(THEMES))
    T = THEMES[theme]
    W, H = size
    img = Image.new("RGB", size, T["bg"])
    d = ImageDraw.Draw(img)
    els: list[Element] = []
    texts: list[dict] = []

    s = lambda v: max(1, int(round(v * scale)))
    f_menu = _font(s(13))
    f_body = _font(s(14))
    f_title = _font(s(20))
    f_small = _font(s(12))

    bar_h = s(30)
    d.rectangle([0, 0, W, bar_h], fill=T["panel"])
    d.line([0, bar_h, W, bar_h], fill=T["line"], width=1)
    x = s(16)
    for name in rng.sample(MENUS, k=rng.randint(4, 6)):
        tb = _text_box(d, (x, s(8)), name, f_menu)
        d.text((x, s(8)), name, fill=T["ink"], font=f_menu)
        pad = s(6)
        els.append(Element((tb[0] - pad, 0, tb[2] + pad, bar_h), "menu", name, "plain", bar_h))
        texts.append({"box": list(tb), "text": name})
        x = tb[2] + s(22)

    sb_w = s(200)
    d.rectangle([0, bar_h, sb_w, H], fill=T["panel"])
    d.line([sb_w, bar_h, sb_w, H], fill=T["line"], width=1)
    y = bar_h + s(20)
    for name in rng.sample(SIDEBAR, k=rng.randint(4, 6)):
        row = (s(10), y, sb_w - s(10), y + s(30))
        if rng.random() < 0.22:
            _rounded(d, row, s(5), fill=T["line"])
        d.text((s(22), y + s(7)), name, fill=T["ink"], font=f_body)
        els.append(Element(row, "nav_item", name, "plain", s(30)))
        texts.append({"box": list(_text_box(d, (s(22), y + s(7)), name, f_body)), "text": name})
        y += s(38)

    doc = f"{rng.choice(NOUNS)} {rng.randint(2020, 2026)}"
    tx, ty = sb_w + s(32), bar_h + s(28)
    d.text((tx, ty), doc, fill=T["ink"], font=f_title)
    texts.append({"box": list(_text_box(d, (tx, ty), doc, f_title)), "text": doc})

    y = ty + s(44)
    x = tx
    for name in rng.sample(["Overview", "Details", "History", "Comments", "Sharing"],
                           k=rng.randint(2, 4)):
        tb = _text_box(d, (x, y), name, f_body)
        d.text((x, y), name, fill=T["ink"] if rng.random() < 0.4 else T["dim"], font=f_body)
        pad = s(8)
        tab = (tb[0] - pad, y - s(6), tb[2] + pad, y + s(24))
        els.append(Element(tab, "tab", name, "plain", s(30)))
        texts.append({"box": list(tb), "text": name})
        x = tb[2] + s(28)
    d.line([tx, y + s(26), W - s(32), y + s(26)], fill=T["line"], width=1)

    y += s(52)
    for label in rng.sample(["Title", "Owner", "Due date", "Tags", "Description"],
                            k=rng.randint(2, 3)):
        d.text((tx, y), label, fill=T["dim"], font=f_small)
        texts.append({"box": list(_text_box(d, (tx, y), label, f_small)), "text": label})
        fb = (tx, y + s(18), tx + s(340), y + s(18) + s(30))
        _rounded(d, fb, s(4), fill=T["bg"], outline=T["line"], width=1)
        els.append(Element(fb, "field", label, "plain", s(30)))
        y += s(66)

    by = H - s(56)
    bx = W - s(32)
    for i, name in enumerate(reversed(rng.sample(ACTIONS, k=rng.randint(2, 4)))):
        tb = d.textbbox((0, 0), name, font=f_body)
        bw = (tb[2] - tb[0]) + s(34)
        bh = s(34)
        box = (bx - bw, by, bx, by + bh)
        if i == 0:
            role, fill, ink = "primary", T["accent"], T["accent_ink"]
        elif name == "Delete":
            role, fill, ink = "danger", T["danger"], (255, 255, 255)
        else:
            role, fill, ink = "secondary", T["panel"], T["ink"]
        _rounded(d, box, s(5), fill=fill,
                 outline=None if role == "primary" else T["line"], width=1)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        d.text((box[0] + (bw - tw) / 2, box[1] + (bh - th) / 2 - s(2)), name, fill=ink, font=f_body)
        els.append(Element(box, "button", name, role, bh))
        texts.append({"box": [int(box[0] + (bw - tw) / 2), int(box[1] + (bh - th) / 2),
                              int(box[0] + (bw + tw) / 2), int(box[1] + (bh + th) / 2)],
                      "text": name})
        bx = box[0] - s(12)

    return Screen(image=img, elements=els, theme=theme, size=size, seed=seed, texts=texts)


def _unique_targets(screen: Screen) -> list[Element]:
    """Only keep elements whose descriptor is unambiguous on this screen.

    If two things are both 'the Save button', a miss is not a model failure --
    it is a bad question. Removing them keeps the metric meaningful.
    """
    seen: dict[str, int] = {}
    for e in screen.elements:
        seen[e.descriptor()] = seen.get(e.descriptor(), 0) + 1
    return [e for e in screen.elements if seen[e.descriptor()] == 1]


def build(n: int = 40, out: str | Path | None = None, size=(1280, 800),
          seed0: int = 0, scales=(1.0, 0.75, 0.55)) -> list[dict]:
    """Generate n screens across difficulty levels. Returns task records."""
    out = Path(out) if out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(n):
        scale = scales[i % len(scales)]
        sc = generate_screen(seed0 + i, size=size, scale=scale)
        path = None
        if out:
            path = out / f"ui_{i:03d}.png"
            sc.image.save(path)
        targets = _unique_targets(sc)
        records.append({
            "id": i, "seed": sc.seed, "theme": sc.theme, "scale": scale,
            "path": str(path) if path else None,
            "size": list(sc.size),
            "ground_truth": [e.to_dict() for e in targets],
            "all_elements": [e.to_dict() for e in sc.elements],
            "texts": sc.texts,
            "_image": sc.image,
        })
    return records