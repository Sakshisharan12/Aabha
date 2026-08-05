"""
test_capable_model.py — Unit tests for Florence-2 wrapper normalization.

These tests exercise only pure logic (no model download / inference).
"""

import pytest
from capable_model import _normalize_detections


class TestNormalizeDetections:
    def test_new_dict_format(self):
        result = {"labels": ["poster", "dog"], "bboxes": [[1, 2, 100, 200], [5.5, 6.6, 50, 60]]}
        out = _normalize_detections(result)
        assert out == [
            {"label": "poster", "score": None, "box": [1, 2, 100, 200]},
            {"label": "dog", "score": None, "box": [5, 6, 50, 60]},
        ]

    def test_old_list_format(self):
        result = [{"label": "cat", "bbox": [10, 10, 90, 90]}, {"label": "bird", "bbox": [0, 0, 1, 1]}]
        out = _normalize_detections(result)
        assert len(out) == 2
        assert out[0]["label"] == "cat"
        assert out[0]["box"] == [10, 10, 90, 90]

    def test_max_objects_limit(self):
        result = {"labels": ["a", "b", "c", "d"], "bboxes": [[0, 0, 1, 1]] * 4}
        assert len(_normalize_detections(result, max_objects=2)) == 2

    def test_empty_result(self):
        assert _normalize_detections({}) == []
        assert _normalize_detections([]) == []

    def test_missing_bbox_fallback(self):
        result = [{"label": "thing"}]
        out = _normalize_detections(result)
        assert out[0]["box"] == [0, 0, 0, 0]