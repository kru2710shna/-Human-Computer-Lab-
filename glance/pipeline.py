"""The Glance inference engine: zoom-refine grounding with logit-level verification.

A 3.75B VLM fed a downscaled 1440p screenshot sees UI text at a few pixels of
height and guesses. Rather than spend parameters we do not have on a bigger model,
we spend *compute* on a second look:

  1. COARSE   Ask on the whole (downscaled) screen. Get a rough box.
  2. ZOOM     Crop a generous region around that box from the ORIGINAL pixels and
              upscale it towards the model's pixel budget, so the target that was
              8 px tall is now 60 px tall. Re-ask inside the crop.
  3. VERIFY   One forward pass on the final crop reading P(Yes) vs P(No) for
              "is this <target>?". This is a real probability, not a generated
              word, so it can be thresholded and can abstain.

The coordinate ledger in imaging.py is what makes step 2 legal: every crop and
resize carries an affine transform back to original screenshot pixels, so the
refined box maps home exactly.

Honest limits, stated up front:
  * The zoom pass can only refine a target the coarse pass roughly found. If the
    coarse box lands on the wrong element entirely, zooming confirms the wrong
    thing. `search_tiles` exists to mitigate this, at real latency cost.
  * P(Yes) is a calibrated-ish confidence, not a guarantee. It is renormalised
    over the Yes/No token groups only, and the model is not temperature-tuned
    against a held-out set. Treat thresholds as tuned heuristics; bench/run.py
    reports the accuracy-vs-abstention tradeoff so they can be re-tuned.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

from . import imaging as im
from .backends.base import Backend
from .parsing import parse_boxes

Box = tuple[float, float, float, float]


# ------------------------------------------------------------------- prompts
GROUND_SYSTEM = (
    "You are a precise UI grounding assistant. You locate elements in screenshots "
    "and report their pixel bounding boxes. You never invent elements that are not "
    "visible."
)

GROUND_PROMPT = (
    'Locate "{target}" in this screenshot.\n'
    'Reply with JSON only, no prose:\n'
    '[{{"bbox_2d": [x1, y1, x2, y2], "label": "<what you found>"}}]\n'
    "Use the tightest box around the element itself, not its container. "
    "If it is genuinely not visible, reply with []."
)

REFINE_PROMPT = (
    'This is a zoomed-in crop of a screenshot. Locate "{target}" within THIS crop.\n'
    'Reply with JSON only:\n'
    '[{{"bbox_2d": [x1, y1, x2, y2], "label": "<what you found>"}}]\n'
    "Coordinates must be relative to this crop. If it is not in this crop, reply with []."
)

VERIFY_PROMPT = 'Does this image show "{target}"?'


@dataclass
class Hit:
    """One grounded element, in ORIGINAL screenshot pixel coordinates."""
    box: Box
    label: str = ""
    text: str = ""
    confidence: float | None = None
    refined: bool = False
    stage: str = "coarse"
    trace: list[dict] = field(default_factory=list)

    @property
    def point(self) -> tuple[float, float]:
        """Where to click."""
        return im.center(self.box)

    def to_dict(self) -> dict:
        x1, y1, x2, y2 = (round(v, 1) for v in self.box)
        cx, cy = (round(v, 1) for v in self.point)
        return {
            "box": [x1, y1, x2, y2],
            "click": [cx, cy],
            "label": self.label,
            "text": self.text,
            "confidence": None if self.confidence is None else round(self.confidence, 4),
            "refined": self.refined,
            "stage": self.stage,
        }


@dataclass
class GroundConfig:
    # Crop this many times the coarse box, so the refine pass has context.
    zoom_scale: float = 3.0
    zoom_pad: float = 24.0
    # Never crop smaller than this (a 12x12 icon needs surrounding context).
    zoom_min_size: float = 160.0
    # Skip the zoom when the coarse box already occupies this much of the screen.
    zoom_skip_area_frac: float = 0.30
    # Accept the refined box only if it sits mostly inside the region we cropped.
    refine_containment: float = 0.55
    verify: bool = True
    # Below this, we report the hit but flag it as low confidence.
    accept_threshold: float = 0.55
    max_new_tokens: int = 160


class Engine:
    """Stateless-per-call wrapper around a Backend. Safe to reuse."""

    def __init__(self, backend: Backend, config: GroundConfig | None = None):
        self.backend = backend
        self.cfg = config or GroundConfig()
        self.spec = backend.spec

    # ------------------------------------------------------------- plumbing
    def view(self, img: Image.Image, upscale_to: int | None = None) -> Image.Image:
        """Snap an image to a model-legal size, carrying its transform along."""
        return im.prepare(img, self.spec.factor, self.spec.min_pixels,
                          self.spec.max_pixels, upscale_to=upscale_to)

    def ask(self, view: Image.Image, prompt: str, max_new_tokens: int = 512,
            system: str | None = None) -> str:
        return self.backend.generate(view, prompt, max_new_tokens=max_new_tokens,
                                     system=system).text

    def boxes_on(self, view: Image.Image, prompt: str, max_new_tokens: int = 512,
                 system: str | None = None) -> list[dict]:
        """Ask for boxes and return them in ORIGINAL screenshot coordinates."""
        raw = self.ask(view, prompt, max_new_tokens=max_new_tokens, system=system)
        out = []
        vw, vh = view.size
        for item in parse_boxes(raw):
            px = im.model_to_pixels(view, item["box"], self.spec.coord_mode)
            px = im.clamp_box(px, vw, vh)
            if im.area(px) <= 0 and not item.get("point"):
                continue
            out.append({**item, "box": im.to_original(view, px), "view_box": px})
        return out

    # -------------------------------------------------------------- verify
        # -------------------------------------------------------------- verify
    def verify(self, img: Image.Image, box: Box, target: str) -> float | None:
        """P(Yes) that the region at `box` (original coords) is `target`.

        Deliberately uses the SMALLEST pixel budget. Measured on an M1 with the
        4-bit build: a verify crop at min_pixels costs 2.55s against 8.34s for a
        full-screen call, and doubling the budget adds 1.2s for no benefit --
        there is no fine detail to recover in a 96px crop of one button.
        """
        w, h = img.size
        region = im.expand_box(im.from_original(img, box), w, h,
                               scale=1.6, pad=12.0, min_size=96.0)
        crop = self.view(im.crop(img, region), upscale_to=self.spec.min_pixels)
        return self.backend.yes_probability(crop, VERIFY_PROMPT.format(target=target))

    # -------------------------------------------------------------- ground
    def ground(self, img: Image.Image, target: str,
               refine: bool = True, verify: bool | None = None) -> Hit | None:
        """Locate one element. Returns None when nothing was found at all."""
        verify = self.cfg.verify if verify is None else verify
        trace: list[dict] = []

        base = self.view(img)
        coarse = self.boxes_on(base, GROUND_PROMPT.format(target=target),
                               max_new_tokens=self.cfg.max_new_tokens, system=GROUND_SYSTEM)
        if not coarse:
            return None

        best = coarse[0]
        hit = Hit(box=best["box"], label=best.get("label", ""), stage="coarse")
        trace.append({"stage": "coarse", "box": [round(v, 1) for v in hit.box],
                      "candidates": len(coarse)})

        # --- zoom-refine ----------------------------------------------------
        img_area = img.size[0] * img.size[1]
        too_big = im.area(hit.box) > self.cfg.zoom_skip_area_frac * img_area
        if refine and not too_big:
            region = im.expand_box(hit.box, img.size[0], img.size[1],
                                   scale=self.cfg.zoom_scale, pad=self.cfg.zoom_pad,
                                   min_size=self.cfg.zoom_min_size)
            # Upscale the crop towards the FULL pixel budget: this is the whole
            # point -- the model now sees the element at many times the detail.
            zoom = self.view(im.crop(img, region), upscale_to=self.spec.max_pixels)
            found = self.boxes_on(zoom, REFINE_PROMPT.format(target=target),
                                  max_new_tokens=self.cfg.max_new_tokens, system=GROUND_SYSTEM)
            picked = None
            for cand in found:
                # Guard against the refine pass wandering outside the crop or
                # returning the whole crop as one giant box.
                cov = im.coverage(cand["box"], region)
                frac = im.area(cand["box"]) / max(1.0, im.area(region))
                if cov >= self.cfg.refine_containment and frac < 0.9:
                    picked = cand
                    break
            if picked:
                hit.box = picked["box"]
                hit.label = picked.get("label") or hit.label
                hit.refined = True
                hit.stage = "refined"
                trace.append({"stage": "refine", "box": [round(v, 1) for v in hit.box],
                              "zoom_px": list(zoom.size)})
            else:
                trace.append({"stage": "refine", "result": "rejected, kept coarse box",
                              "candidates": len(found)})
        elif refine and too_big:
            trace.append({"stage": "refine", "result": "skipped, coarse box covers most of screen"})

        # --- verify ---------------------------------------------------------
        if verify:
            p = self.verify(img, hit.box, target)
            hit.confidence = p
            trace.append({"stage": "verify", "p_yes": None if p is None else round(p, 4)})

        hit.trace = trace
        return hit

    def ground_all(self, img: Image.Image, target: str, max_items: int = 20,
                   verify: bool = False) -> list[Hit]:
        """Locate every instance of a target. No zoom pass: refining each of N
        boxes costs N extra calls, which is rarely worth it for a list view."""
        base = self.view(img)
        prompt = (
            f'Locate every instance of "{target}" in this screenshot.\n'
            'Reply with JSON only:\n'
            '[{"bbox_2d": [x1, y1, x2, y2], "label": "<short description>"}]\n'
            f"List at most {max_items}. If there are none, reply with []."
        )
        items = self.boxes_on(base, prompt, max_new_tokens=1024, system=GROUND_SYSTEM)
        hits = [Hit(box=it["box"], label=it.get("label", ""), text=it.get("text", ""),
                    stage="coarse") for it in items[:max_items]]
        hits = im.reading_order(hits, key=lambda h: h.box)
        if verify:
            for h in hits:
                h.confidence = self.verify(img, h.box, target)
        return hits

    def search_tiles(self, img: Image.Image, target: str, cols: int = 2, rows: int = 2,
                     overlap: float = 0.15) -> Hit | None:
        """Fallback for when the coarse pass finds nothing: split the screen into
        overlapping tiles and ask each one. Costs cols*rows calls, so this is opt-in.

        Overlapping tiles matter: an element straddling a tile seam is invisible to
        both halves without them.
        """
        w, h = img.size
        tw, th = w / cols, h / rows
        ox, oy = tw * overlap, th * overlap
        best: Hit | None = None
        for r in range(rows):
            for c in range(cols):
                region = im.clamp_box((c * tw - ox, r * th - oy,
                                       (c + 1) * tw + ox, (r + 1) * th + oy), w, h)
                tile = self.view(im.crop(img, region), upscale_to=self.spec.max_pixels)
                found = self.boxes_on(tile, REFINE_PROMPT.format(target=target),
                                      max_new_tokens=self.cfg.max_new_tokens,
                                      system=GROUND_SYSTEM)
                for cand in found:
                    if im.coverage(cand["box"], region) < 0.55:
                        continue
                    p = self.backend.yes_probability(
                        self.view(im.crop(img, im.expand_box(cand["box"], w, h, 1.6, 12, 96)),
                                  upscale_to=self.spec.min_pixels * 2),
                        VERIFY_PROMPT.format(target=target))
                    score = -1.0 if p is None else p
                    if best is None or score > (best.confidence or -1.0):
                        best = Hit(box=cand["box"], label=cand.get("label", ""),
                                   confidence=p, stage=f"tile[{r},{c}]", refined=True)
        return best

    def locate(self, img: Image.Image, target: str, fallback_tiles: bool = True) -> Hit | None:
        """Full grounding strategy: coarse -> zoom -> verify, with a tile sweep
        when the first attempt finds nothing or is unconvincing."""
        hit = self.ground(img, target)
        weak = hit is None or (hit.confidence is not None
                               and hit.confidence < self.cfg.accept_threshold)
        if weak and fallback_tiles:
            alt = self.search_tiles(img, target)
            if alt and (hit is None or (alt.confidence or 0) > (hit.confidence or 0)):
                alt.trace = (hit.trace if hit else []) + [{"stage": "tile_fallback", "used": True}]
                return alt
        return hit