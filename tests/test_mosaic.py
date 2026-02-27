"""Tests for blurblur.mosaic module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from blurblur.detection import Detection
from blurblur.detector import YOLODetector
from blurblur.mosaic import MosaicProcessor


def _make_detector(input_size: int = 640) -> YOLODetector:
    """Create a YOLODetector with a mocked ONNX session."""
    with patch("onnxruntime.InferenceSession"):
        return YOLODetector("dummy.onnx", input_size=input_size)


class TestMosaicProcessorApplyMosaic:
    def test_mosaic_changes_pixels_in_region(self) -> None:
        person_det = _make_detector()
        plate_det = _make_detector()
        proc = MosaicProcessor(person_det, plate_det, mosaic_ratio=0.05)

        # Create a gradient image so mosaic will definitely change pixels
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        for i in range(200):
            img[i, :, :] = i
        original = img.copy()

        result = proc._apply_mosaic(img, (50, 50, 150, 150))
        # Pixels inside bbox should differ
        region_orig = original[50:150, 50:150]
        region_result = result[50:150, 50:150]
        assert not np.array_equal(region_orig, region_result)

    def test_mosaic_does_not_change_outside_region(self) -> None:
        person_det = _make_detector()
        plate_det = _make_detector()
        proc = MosaicProcessor(person_det, plate_det, mosaic_ratio=0.05)

        img = np.arange(200 * 200 * 3, dtype=np.uint8).reshape(200, 200, 3)
        original = img.copy()

        result = proc._apply_mosaic(img, (50, 50, 150, 150))
        # Top-left corner (outside bbox) should be unchanged
        assert np.array_equal(original[:50, :50], result[:50, :50])
        # Bottom-right corner (outside bbox)
        assert np.array_equal(original[150:, 150:], result[150:, 150:])


class TestMosaicProcessorProcessImage:
    def test_process_image_with_detections(self) -> None:
        person_det = MagicMock()
        plate_det = MagicMock()
        person_det.detect.return_value = [
            Detection(x1=10, y1=10, x2=50, y2=50, confidence=0.9, class_id=0),
        ]
        plate_det.detect.return_value = [
            Detection(x1=100, y1=100, x2=150, y2=150, confidence=0.8, class_id=0),
        ]
        proc = MosaicProcessor(person_det, plate_det, mosaic_ratio=0.05)

        img = np.zeros((200, 200, 3), dtype=np.uint8)
        result, person_count, plate_count = proc.process_image(img, conf_thresh=0.25, iou_thresh=0.45)
        assert person_count == 1
        assert plate_count == 1
        assert result.shape == img.shape

    def test_process_image_no_detections(self) -> None:
        person_det = MagicMock()
        plate_det = MagicMock()
        person_det.detect.return_value = []
        plate_det.detect.return_value = []
        proc = MosaicProcessor(person_det, plate_det, mosaic_ratio=0.05)

        img = np.zeros((200, 200, 3), dtype=np.uint8)
        original = img.copy()
        result, person_count, plate_count = proc.process_image(img, conf_thresh=0.25, iou_thresh=0.45)
        assert person_count == 0
        assert plate_count == 0
        assert np.array_equal(result, original)
