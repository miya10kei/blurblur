"""Tests for blurblur.detector module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
from numpy.typing import NDArray

from blurblur.detector import YOLODetector


def _make_detector(input_size: int = 640) -> YOLODetector:
    """Create a YOLODetector with a mocked ONNX session."""
    with patch("onnxruntime.InferenceSession"):
        return YOLODetector("dummy.onnx", input_size=input_size)


def _make_postprocess_output(
    boxes_xywh: list[list[float]],
    class_scores: list[list[float]],
) -> NDArray[np.float32]:
    """Build a fake YOLO output tensor (1, 4+num_classes, num_boxes).

    boxes_xywh: list of [cx, cy, w, h] in input-size coords (after preprocess)
    class_scores: list of per-class scores for each box
    """
    num_classes = len(class_scores[0]) if class_scores else 1
    num_boxes = len(boxes_xywh)
    # shape: (1, 4+num_classes, num_boxes)
    data = np.zeros((1, 4 + num_classes, num_boxes), dtype=np.float32)
    for i, (box, scores) in enumerate(zip(boxes_xywh, class_scores, strict=True)):
        data[0, :4, i] = box
        data[0, 4:, i] = scores
    return data


class TestYOLODetectorPreprocess:
    def test_output_shape(self) -> None:
        detector = _make_detector(input_size=640)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        blob, ratio, (pad_w, pad_h) = detector._preprocess(img)
        assert blob.shape == (1, 3, 640, 640)
        assert blob.dtype == np.float32

    def test_value_range(self) -> None:
        detector = _make_detector(input_size=640)
        img = np.full((480, 640, 3), 255, dtype=np.uint8)
        blob, _, _ = detector._preprocess(img)
        assert blob.min() >= 0.0
        assert blob.max() <= 1.0

    def test_square_image_no_padding(self) -> None:
        detector = _make_detector(input_size=640)
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        blob, ratio, (pad_w, pad_h) = detector._preprocess(img)
        assert ratio == 1.0
        assert pad_w == 0.0
        assert pad_h == 0.0

    def test_landscape_image_padding(self) -> None:
        detector = _make_detector(input_size=640)
        img = np.zeros((320, 640, 3), dtype=np.uint8)
        blob, ratio, (pad_w, pad_h) = detector._preprocess(img)
        assert ratio == 1.0
        assert pad_w == 0.0
        assert pad_h == 160.0  # (640 - 320) / 2

    def test_portrait_image_padding(self) -> None:
        detector = _make_detector(input_size=640)
        img = np.zeros((640, 320, 3), dtype=np.uint8)
        blob, ratio, (pad_w, pad_h) = detector._preprocess(img)
        assert ratio == 1.0
        assert pad_w == 160.0  # (640 - 320) / 2
        assert pad_h == 0.0

    def test_large_image_scaled_down(self) -> None:
        detector = _make_detector(input_size=640)
        img = np.zeros((1280, 1920, 3), dtype=np.uint8)
        blob, ratio, (pad_w, pad_h) = detector._preprocess(img)
        # ratio = 640 / 1920 = 1/3
        expected_ratio = 640 / 1920
        assert abs(ratio - expected_ratio) < 1e-6
        # resized: 1280*ratio=426.67 -> pad_h = (640 - 426.67)/2 ≈ 106.67
        assert blob.shape == (1, 3, 640, 640)

    def test_padding_color(self) -> None:
        detector = _make_detector(input_size=640)
        img = np.zeros((320, 640, 3), dtype=np.uint8)
        blob, _, (pad_w, pad_h) = detector._preprocess(img)
        # Top padding region should be 114/255
        expected = 114.0 / 255.0
        pad_top = int(pad_h)
        if pad_top > 0:
            assert abs(blob[0, 0, 0, 0] - expected) < 1e-3


class TestYOLODetectorPostprocess:
    def test_single_detection(self) -> None:
        detector = _make_detector(input_size=640)
        # Image 640x640, ratio=1, pad=(0,0)
        # Box at center: cx=320, cy=320, w=100, h=100 -> xyxy: 270,270,370,370
        output = _make_postprocess_output(
            boxes_xywh=[[320, 320, 100, 100]],
            class_scores=[[0.9]],
        )
        detections = detector._postprocess(output, ratio=1.0, pad_w=0.0, pad_h=0.0, conf_thresh=0.25, iou_thresh=0.45)
        assert len(detections) == 1
        d = detections[0]
        assert d.class_id == 0
        assert d.confidence > 0.8
        assert abs(d.x1 - 270) <= 1
        assert abs(d.y1 - 270) <= 1
        assert abs(d.x2 - 370) <= 1
        assert abs(d.y2 - 370) <= 1

    def test_confidence_filter(self) -> None:
        detector = _make_detector(input_size=640)
        output = _make_postprocess_output(
            boxes_xywh=[[320, 320, 100, 100], [100, 100, 50, 50]],
            class_scores=[[0.9], [0.1]],  # second box below threshold
        )
        detections = detector._postprocess(output, ratio=1.0, pad_w=0.0, pad_h=0.0, conf_thresh=0.25, iou_thresh=0.45)
        assert len(detections) == 1

    def test_empty_when_no_detections(self) -> None:
        detector = _make_detector(input_size=640)
        output = _make_postprocess_output(
            boxes_xywh=[[320, 320, 100, 100]],
            class_scores=[[0.05]],  # below threshold
        )
        detections = detector._postprocess(output, ratio=1.0, pad_w=0.0, pad_h=0.0, conf_thresh=0.25, iou_thresh=0.45)
        assert len(detections) == 0

    def test_coordinate_inverse_transform(self) -> None:
        detector = _make_detector(input_size=640)
        # Simulating a 1920x1280 image: ratio=640/1920=1/3, pad_w=0, pad_h≈106.67
        ratio = 640 / 1920
        new_h = int(1280 * ratio)  # 426
        pad_h = (640 - new_h) / 2  # 107.0
        pad_w = 0.0
        # A box at cx=320, cy=320, w=60, h=60 in preprocessed coords
        # xyxy in preprocessed: (290, 290, 350, 350)
        # Remove padding: (290-0, 290-107, 350-0, 350-107) = (290, 183, 350, 243)
        # Scale back: / ratio = (290/ratio, 183/ratio, 350/ratio, 243/ratio)
        output = _make_postprocess_output(
            boxes_xywh=[[320, 320, 60, 60]],
            class_scores=[[0.9]],
        )
        detections = detector._postprocess(
            output, ratio=ratio, pad_w=pad_w, pad_h=pad_h, conf_thresh=0.25, iou_thresh=0.45
        )
        assert len(detections) == 1
        d = detections[0]
        # Expected after inverse transform
        assert d.x1 > 0
        assert d.y1 > 0

    def test_target_classes_filter(self) -> None:
        detector = _make_detector(input_size=640)
        # 3 classes: person(0), car(1), plate(2)
        output = _make_postprocess_output(
            boxes_xywh=[[320, 320, 100, 100], [100, 100, 50, 50]],
            class_scores=[[0.1, 0.9, 0.05], [0.05, 0.05, 0.9]],
        )
        # Only want class 0 (person) - neither box has high class 0 score
        detections = detector._postprocess(
            output, ratio=1.0, pad_w=0.0, pad_h=0.0, conf_thresh=0.25, iou_thresh=0.45, target_classes=[0]
        )
        assert len(detections) == 0

        # Only want class 1 (car)
        detections = detector._postprocess(
            output, ratio=1.0, pad_w=0.0, pad_h=0.0, conf_thresh=0.25, iou_thresh=0.45, target_classes=[1]
        )
        assert len(detections) == 1
        assert detections[0].class_id == 1

    def test_nms_removes_overlapping(self) -> None:
        detector = _make_detector(input_size=640)
        # Two highly overlapping boxes
        output = _make_postprocess_output(
            boxes_xywh=[[320, 320, 100, 100], [325, 325, 100, 100]],
            class_scores=[[0.9], [0.85]],
        )
        detections = detector._postprocess(output, ratio=1.0, pad_w=0.0, pad_h=0.0, conf_thresh=0.25, iou_thresh=0.45)
        assert len(detections) == 1  # NMS should suppress one


class TestYOLODetectorDetect:
    def test_detect_calls_session_and_returns_detections(self) -> None:
        detector = _make_detector(input_size=640)
        # Mock the ONNX session's run method
        fake_output = _make_postprocess_output(
            boxes_xywh=[[320, 320, 100, 100]],
            class_scores=[[0.9]],
        )
        mock_input = MagicMock()
        mock_input.name = "images"
        detector.session.get_inputs.return_value = [mock_input]
        detector.session.run.return_value = [fake_output]

        img = np.zeros((640, 640, 3), dtype=np.uint8)
        detections = detector.detect(img, conf_thresh=0.25, iou_thresh=0.45)
        assert len(detections) == 1
        assert detections[0].class_id == 0
        detector.session.run.assert_called_once()

    def test_detect_with_target_classes(self) -> None:
        detector = _make_detector(input_size=640)
        # 2 classes, only class 1 has high score
        fake_output = _make_postprocess_output(
            boxes_xywh=[[320, 320, 100, 100]],
            class_scores=[[0.1, 0.9]],
        )
        mock_input = MagicMock()
        mock_input.name = "images"
        detector.session.get_inputs.return_value = [mock_input]
        detector.session.run.return_value = [fake_output]

        img = np.zeros((640, 640, 3), dtype=np.uint8)
        # target class 0 only -> should get nothing
        detections = detector.detect(img, conf_thresh=0.25, iou_thresh=0.45, target_classes=[0])
        assert len(detections) == 0

        # target class 1 -> should get 1
        detections = detector.detect(img, conf_thresh=0.25, iou_thresh=0.45, target_classes=[1])
        assert len(detections) == 1
