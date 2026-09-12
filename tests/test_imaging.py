"""The coordinate ledger is load-bearing: every IoU number in the benchmark
depends on boxes mapping correctly from model space through resized views and
zoom crops back to original screenshot pixels. An off-by-one here would silently
shift every reported result, so these tests check the maths independently of any
model."""
from __future__ import annotations

import pytest
from PIL import Image

from glance import imaging as im


def blank(w=1280, h=800):
    return im.load_image_from(Image.new("RGB", (w, h), (250, 250, 252))) \
        if hasattr(im, "load_image_from") else im.set_tf(
            Image.new("RGB", (w, h), (250, 250, 252)), (0.0, 0.0, 1.0, 1.0))


def approx(a, b, tol=1e-6):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


class TestTransform:
    def test_identity_on_fresh_image(self):
        assert im.get_tf(blank()) == (0.0, 0.0, 1.0, 1.0)

    def test_round_trip_identity(self):
        img = blank()
        box = (100.0, 200.0, 340.0, 260.0)
        assert approx(im.from_original(img, im.to_original(img, box)), box)

    def test_resize_maps_home(self):
        """A box on a resized view must map back to the same region of the original."""
        img = blank(1280, 800)
        view = im.prepare(img, factor=28, min_pixels=256 * 28 * 28, max_pixels=640 * 28 * 28)
        assert view.size != img.size, "test is vacuous if no resize happened"
        # Centre of the view is the centre of the original, whatever the scale.
        vw, vh = view.size
        centre_box = (vw / 2 - 5, vh / 2 - 5, vw / 2 + 5, vh / 2 + 5)
        orig = im.to_original(view, centre_box)
        cx, cy = im.center(orig)
        assert abs(cx - 640) < 8 and abs(cy - 400) < 8

    def test_crop_offsets_accumulate(self):
        img = blank(1280, 800)
        region = (400.0, 300.0, 700.0, 500.0)
        c = im.crop(img, region)
        assert c.size == (300, 200)
        # (0,0) of the crop is (400,300) of the original.
        assert approx(im.to_original(c, (0, 0, 10, 10)), (400, 300, 410, 310))

    def test_crop_then_resize_then_home(self):
        """The full zoom-refine path: crop from original, upscale, map a box back."""
        img = blank(1280, 800)
        region = (900.0, 60.0, 1060.0, 104.0)      # a button
        zoom = im.prepare(im.crop(img, region), factor=28,
                          min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28,
                          upscale_to=1280 * 28 * 28)
        zw, zh = zoom.size
        assert zw > 160, "crop should have been upscaled"
        # A box covering the whole zoom view maps back to the whole region.
        back = im.to_original(zoom, (0, 0, zw, zh))
        assert approx(back, region, tol=1.5)

    def test_nested_crops(self):
        img = blank(1280, 800)
        a = im.crop(img, (200, 100, 800, 600))
        b = im.crop(a, (50, 50, 150, 150))
        assert approx(im.to_original(b, (0, 0, 10, 10)), (250, 150, 260, 160))


class TestFitDims:
    def test_snaps_to_factor(self):
        w, h = im.fit_dims(1000, 700, factor=28, min_pixels=100 * 28 * 28,
                           max_pixels=1280 * 28 * 28)
        assert w % 28 == 0 and h % 28 == 0

    def test_respects_max_pixels(self):
        mx = 640 * 28 * 28
        w, h = im.fit_dims(4000, 3000, factor=28, min_pixels=64 * 28 * 28, max_pixels=mx)
        assert w * h <= mx

    def test_respects_min_pixels(self):
        mn = 256 * 28 * 28
        w, h = im.fit_dims(60, 40, factor=28, min_pixels=mn, max_pixels=1280 * 28 * 28)
        assert w * h >= mn * 0.9

    def test_preserves_aspect_roughly(self):
        w, h = im.fit_dims(1600, 900, factor=28, min_pixels=64 * 28 * 28,
                           max_pixels=640 * 28 * 28)
        assert abs((w / h) - (1600 / 900)) < 0.1

    def test_large_screens_clamp_identically(self):
        """1440p and 4K should cost the same, which is what the latency data showed."""
        a = im.fit_dims(2560, 1600, 28, 256 * 28 * 28, 1280 * 28 * 28)
        b = im.fit_dims(3840, 2400, 28, 256 * 28 * 28, 1280 * 28 * 28)
        assert a == b


class TestBoxMaths:
    def test_iou_identical(self):
        b = (10, 10, 50, 50)
        assert im.iou(b, b) == pytest.approx(1.0)

    def test_iou_disjoint(self):
        assert im.iou((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0

    def test_iou_half_overlap(self):
        assert im.iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)

    def test_coverage_inner_fully_inside(self):
        assert im.coverage((10, 10, 20, 20), (0, 0, 100, 100)) == pytest.approx(1.0)

    def test_point_in(self):
        assert im.point_in((15, 15), (10, 10, 20, 20))
        assert not im.point_in((25, 15), (10, 10, 20, 20))

    def test_clamp_fixes_inverted(self):
        assert im.clamp_box((50, 40, 10, 20), 100, 100) == (10, 20, 50, 40)

    def test_expand_respects_bounds(self):
        b = im.expand_box((5, 5, 15, 15), 100, 100, scale=10.0)
        assert b[0] >= 0 and b[1] >= 0 and b[2] <= 100 and b[3] <= 100

    def test_expand_min_size(self):
        b = im.expand_box((50, 50, 52, 52), 200, 200, scale=1.0, min_size=96)
        assert (b[2] - b[0]) >= 95 and (b[3] - b[1]) >= 95


class TestCoordModes:
    def test_rel1000_scales(self):
        img = Image.new("RGB", (800, 600))
        got = im.model_to_pixels(img, (0, 0, 500, 500), "rel1000")
        assert approx(got, (0, 0, 400, 300))

    def test_abs_passthrough(self):
        img = Image.new("RGB", (800, 600))
        assert im.model_to_pixels(img, (10, 20, 30, 40), "abs") == (10, 20, 30, 40)

    def test_rel1000_round_trip(self):
        img = Image.new("RGB", (1260, 784))
        box = (100.0, 200.0, 400.0, 300.0)
        back = im.model_to_pixels(img, im.pixels_to_model(img, box, "rel1000"), "rel1000")
        assert approx(back, box, tol=1e-6)


class TestReadingOrder:
    def test_sorts_rows_then_columns(self):
        items = [
            {"box": (300, 10, 380, 30), "n": "b"},
            {"box": (10, 10, 90, 30), "n": "a"},
            {"box": (10, 100, 90, 120), "n": "c"},
        ]
        assert [i["n"] for i in im.reading_order(items)] == ["a", "b", "c"]

    def test_tolerates_slight_row_misalignment(self):
        items = [
            {"box": (200, 12, 280, 32), "n": "b"},
            {"box": (10, 10, 90, 30), "n": "a"},
        ]
        assert [i["n"] for i in im.reading_order(items)] == ["a", "b"]

    def test_empty(self):
        assert im.reading_order([]) == []