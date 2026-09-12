"""A 3B model does not emit clean JSON reliably. These cases are all things we
actually saw during development -- fenced output, truncated lists, echoed target
strings as labels, prose with coordinates. The parser must never raise."""
from __future__ import annotations

from glance import parsing as P


class TestExtractJson:
    def test_plain(self):
        assert P.extract_json('{"a": 1}') == {"a": 1}

    def test_fenced(self):
        assert P.extract_json('```json\n[{"a": 1}]\n```') == [{"a": 1}]

    def test_with_preamble(self):
        assert P.extract_json('Here is the result:\n[{"a": 1}]') == [{"a": 1}]

    def test_trailing_prose(self):
        assert P.extract_json('[{"a": 1}] Hope that helps!') == [{"a": 1}]

    def test_truncated_list_salvages_complete_objects(self):
        """Hitting max_new_tokens mid-list is common on OCR of a busy screen."""
        got = P.extract_json('[{"bbox_2d": [1,2,3,4], "text": "a"}, {"bbox_2d": [5,6,7')
        assert isinstance(got, list) and len(got) == 1

    def test_garbage_returns_none(self):
        assert P.extract_json("I cannot see any buttons.") is None

    def test_empty(self):
        assert P.extract_json("") is None


class TestParseBoxes:
    def test_standard(self):
        got = P.parse_boxes('[{"bbox_2d": [10, 20, 30, 40], "label": "Save"}]')
        assert got[0]["box"] == (10, 20, 30, 40) and got[0]["label"] == "Save"

    def test_alternate_key_names(self):
        for key in ("bbox", "box", "bounding_box", "coordinates", "box_2d"):
            got = P.parse_boxes(f'[{{"{key}": [1,2,3,4]}}]')
            assert got[0]["box"] == (1, 2, 3, 4), key

    def test_bare_box(self):
        assert P.parse_boxes("[10, 20, 30, 40]")[0]["box"] == (10, 20, 30, 40)

    def test_wrapped_in_object(self):
        got = P.parse_boxes('{"elements": [{"bbox_2d": [1,2,3,4]}]}')
        assert len(got) == 1

    def test_point_becomes_tiny_box(self):
        got = P.parse_boxes('[{"point_2d": [100, 200]}]')
        assert got[0].get("point") is True
        assert got[0]["box"][0] < 100 < got[0]["box"][2]

    def test_nested_pairs(self):
        got = P.parse_boxes('[{"bbox_2d": [[10, 20], [30, 40]]}]')
        assert got[0]["box"] == (10, 20, 30, 40)

    def test_normalises_inverted(self):
        assert P.parse_boxes('[{"bbox_2d": [30, 40, 10, 20]}]')[0]["box"] == (10, 20, 30, 40)

    def test_prose_fallback(self):
        got = P.parse_boxes("The button is at (900, 60, 1060, 104) on screen.")
        assert got[0]["box"] == (900, 60, 1060, 104)

    def test_floats(self):
        assert P.parse_boxes('[{"bbox_2d": [10.5, 20.25, 30.0, 40.75]}]')[0]["box"] == \
            (10.5, 20.25, 30.0, 40.75)

    def test_no_boxes_returns_empty(self):
        assert P.parse_boxes("[]") == []
        assert P.parse_boxes("I don't see it.") == []

    def test_never_raises_on_junk(self):
        for junk in ("", "null", "{{{", '{"bbox_2d": "not a box"}',
                     '[{"bbox_2d": [1,2]}]', '{"bbox_2d": null}'):
            P.parse_boxes(junk)  # must not raise

    def test_text_keys(self):
        got = P.parse_boxes('[{"bbox_2d": [1,2,3,4], "text_content": "Hello"}]')
        assert got[0]["text"] == "Hello"


class TestParseYesNo:
    def test_clear_answers(self):
        assert P.parse_yes_no("Yes") is True
        assert P.parse_yes_no("No.") is False
        assert P.parse_yes_no("yes, that is the Save button") is True

    def test_synonyms(self):
        assert P.parse_yes_no("True") is True
        assert P.parse_yes_no("Incorrect") is False

    def test_ambiguous_is_none(self):
        """None means abstain, which is different from 'no'."""
        assert P.parse_yes_no("Maybe") is None
        assert P.parse_yes_no("") is None
        assert P.parse_yes_no("yes and no") is None


class TestParseSteps:
    def test_json(self):
        got = P.parse_steps('[{"instruction": "Click File", "target": "File menu"}]')
        assert got[0]["instruction"] == "Click File"
        assert got[0]["target"] == "File menu"

    def test_null_target(self):
        assert P.parse_steps('[{"instruction": "Wait", "target": null}]')[0]["target"] is None

    def test_numbered_lines_fallback(self):
        got = P.parse_steps("1. Open the File menu\n2. Choose Export")
        assert len(got) == 2 and got[1]["instruction"] == "Choose Export"

    def test_bullets_fallback(self):
        assert len(P.parse_steps("- First thing\n- Second thing")) == 2

    def test_empty(self):
        assert P.parse_steps("") == []