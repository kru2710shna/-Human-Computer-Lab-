"""Guards on the benchmark itself. If the dataset or metrics are wrong, every
number we report is wrong in a way no model test would catch."""
from __future__ import annotations

from glance.bench import dataset, metrics


class TestDataset:
    def test_deterministic(self):
        a = dataset.generate_screen(42)
        b = dataset.generate_screen(42)
        assert [e.to_dict() for e in a.elements] == [e.to_dict() for e in b.elements]

    def test_different_seeds_differ(self):
        a = dataset.generate_screen(1)
        b = dataset.generate_screen(2)
        assert [e.text for e in a.elements] != [e.text for e in b.elements]

    def test_boxes_inside_canvas(self):
        for seed in range(8):
            sc = dataset.generate_screen(seed, size=(1280, 800))
            for e in sc.elements:
                x1, y1, x2, y2 = e.box
                assert 0 <= x1 < x2 <= 1280, f"{e.text}: {e.box}"
                assert 0 <= y1 < y2 <= 800, f"{e.text}: {e.box}"

    def test_scale_shrinks_elements(self):
        big = dataset.generate_screen(7, scale=1.0)
        small = dataset.generate_screen(7, scale=0.55)
        bh = [e.height for e in big.elements if e.role == "button"]
        sh = [e.height for e in small.elements if e.role == "button"]
        assert bh and sh and min(bh) > min(sh)

    def test_box_kind_split(self):
        sc = dataset.generate_screen(3)
        kinds = {e.role: e.box_kind for e in sc.elements}
        assert kinds.get("button") == "tight"
        assert kinds.get("nav_item") == "region"

    def test_unique_targets_drops_duplicates(self):
        sc = dataset.generate_screen(5)
        descs = [e.descriptor() for e in dataset._unique_targets(sc)]
        assert len(descs) == len(set(descs))

    def test_build_records(self):
        recs = dataset.build(n=3, out=None)
        assert len(recs) == 3
        for r in recs:
            assert r["ground_truth"] and "_image" in r
            assert all("box_kind" in e for e in r["ground_truth"])


class TestMetrics:
    def test_perfect_match(self):
        s = metrics.score_one((10, 10, 50, 50), (10, 10, 50, 50))
        assert s["iou"] == 1.0 and s["click_hit"] and s["centre_error_px"] == 0.0

    def test_click_hit_with_poor_iou(self):
        """The tab/menu case: right element, wrong extent. Both facts must survive."""
        s = metrics.score_one((20, 20, 30, 30), (0, 0, 100, 100))
        assert s["click_hit"] and s["iou"] < 0.1

    def test_miss(self):
        s = metrics.score_one((0, 0, 10, 10), (500, 500, 600, 600))
        assert not s["click_hit"] and s["iou"] == 0.0

    def test_aggregate_empty(self):
        assert metrics.aggregate([])["n"] == 0

    def test_aggregate_counts_misses_against_accuracy(self):
        rows = [
            {"found": True, "click_hit": True, "iou": 0.9, "latency_s": 1.0,
             "centre_error_px": 1.0, "box_kind": "tight"},
            {"found": False, "latency_s": 1.0},
        ]
        a = metrics.aggregate(rows)
        assert a["click_accuracy"] == 0.5
        assert a["click_accuracy_when_found"] == 1.0

    def test_tight_only_iou(self):
        rows = [
            {"found": True, "click_hit": True, "iou": 0.9, "latency_s": 1.0,
             "centre_error_px": 1.0, "box_kind": "tight"},
            {"found": True, "click_hit": True, "iou": 0.1, "latency_s": 1.0,
             "centre_error_px": 1.0, "box_kind": "region"},
        ]
        a = metrics.aggregate(rows)
        assert a["iou_mean"] == 0.5           # all targets
        assert a["tight_iou_mean"] == 0.9     # only where IoU is meaningful

    def test_confidence_analysis_trades_coverage_for_precision(self):
        rows = [
            {"confidence": 0.95, "click_hit": True},
            {"confidence": 0.9, "click_hit": True},
            {"confidence": 0.3, "click_hit": False},
        ]
        out = {c["threshold"]: c for c in metrics.confidence_analysis(rows)}
        assert out[0.0]["precision"] < out[0.75]["precision"]
        assert out[0.75]["coverage"] < out[0.0]["coverage"]

    def test_ocr_recall(self):
        pred = [{"text": "Save"}, {"text": "Cancel"}]
        true = [{"text": "Save"}, {"text": "Cancel"}, {"text": "Export"}]
        assert metrics.ocr_score(pred, true)["recall"] == round(2 / 3, 4)