"""Tests for blurblur.detection module."""

from __future__ import annotations

from blurblur.detection import Detection


class TestDetection:
    def test_create_detection(self) -> None:
        d = Detection(x1=10, y1=20, x2=100, y2=200, confidence=0.9, class_id=0)
        assert d.x1 == 10
        assert d.y1 == 20
        assert d.x2 == 100
        assert d.y2 == 200
        assert d.confidence == 0.9
        assert d.class_id == 0

    def test_detection_is_dataclass(self) -> None:
        from dataclasses import is_dataclass

        assert is_dataclass(Detection)
