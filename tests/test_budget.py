"""The parameter audit is the constraint proof. These tests use synthetic headers
so they run offline and deterministically -- no network, no weights."""
from __future__ import annotations

import json
import struct

import pytest

from glance import budget
from glance.config import PARAM_BUDGET


def make_header(tensors: dict) -> bytes:
    blob = json.dumps(tensors).encode()
    return struct.pack("<Q", len(blob)) + blob


class TestHeaderParsing:
    def test_round_trip(self):
        h = {"model.layers.0.weight": {"dtype": "BF16", "shape": [128, 256]}}
        assert budget._parse_header(make_header(h)) == h

    def test_rejects_truncated(self):
        with pytest.raises(ValueError):
            budget._parse_header(b"\x00\x01")

    def test_rejects_oversized_length(self):
        with pytest.raises(ValueError):
            budget._parse_header(struct.pack("<Q", 10**9) + b"{}")

    def test_numel(self):
        assert budget._numel([128, 256]) == 32768
        assert budget._numel([10]) == 10

    def test_skips_metadata_and_scalars(self):
        seen = {}
        budget._accumulate({"__metadata__": {"format": "pt"},
                            "scalar": {"dtype": "F32", "shape": []},
                            "real": {"dtype": "BF16", "shape": [4, 4]}}, seen)
        assert list(seen) == ["real"]

    def test_deduplicates_by_name(self):
        """Sharded checkpoints must not double-count a repeated tensor name."""
        seen = {}
        t = {"w": {"dtype": "BF16", "shape": [100, 100]}}
        budget._accumulate(t, seen)
        budget._accumulate(t, seen)
        assert seen["w"][0] == 10000 and len(seen) == 1


class TestClassification:
    @pytest.mark.parametrize("name,expected", [
        ("visual.blocks.0.attn.qkv.weight", "vision encoder"),
        ("vision_tower.encoder.layer.3.weight", "vision encoder"),
        ("visual.merger.mlp.0.weight", "vision-language projector"),
        ("multi_modal_projector.linear_1.weight", "vision-language projector"),
        ("model.embed_tokens.weight", "token embeddings"),
        ("model.layers.12.self_attn.q_proj.weight", "language backbone"),
        ("lm_head.weight", "output head"),
        ("something.unexpected.weight", "other"),
    ])
    def test_component_rules(self, name, expected):
        assert budget.classify(name) == expected

    def test_projector_beats_encoder(self):
        """visual.merger contains 'visual' but is the projector, not the encoder.
        Rule order matters; this pins it."""
        assert budget.classify("visual.merger.ln_q.weight") == "vision-language projector"


class TestReport:
    def _report(self, total):
        r = budget.BudgetReport(model_id="test", source="synthetic")
        r.total = total
        return r

    def test_within_budget(self):
        assert self._report(3_754_622_976).within_budget

    def test_over_budget(self):
        r = self._report(8_292_166_656)
        assert not r.within_budget and r.headroom < 0

    def test_exactly_at_cap_passes(self):
        assert self._report(PARAM_BUDGET).within_budget

    def test_one_over_fails(self):
        assert not self._report(PARAM_BUDGET + 1).within_budget

    def test_dict_shape(self):
        d = self._report(1_000_000_000).to_dict()
        for k in ("total_parameters", "within_budget", "headroom", "components", "budget"):
            assert k in d


class TestRuntimeVerification:
    def test_agrees_when_equal(self):
        r = budget.BudgetReport(model_id="t", source="s")
        r.total = 3_754_622_976
        assert budget.verify_runtime(r, 3_754_622_976)["agrees"]

    def test_reports_disagreement_rather_than_raising(self):
        """A mismatch is a real finding about the checkpoint; surface it."""
        r = budget.BudgetReport(model_id="t", source="s")
        r.total = 3_754_622_976
        out = budget.verify_runtime(r, 3_000_000_000)
        assert not out["agrees"] and out["delta"] < 0