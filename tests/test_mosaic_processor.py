"""Tests for mosaic_processor module."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from numpy.typing import NDArray

from mosaic_processor import Detection, MosaicProcessor, YOLODetector, parse_args


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


def _make_detector(input_size: int = 640) -> YOLODetector:
    """Create a YOLODetector with a mocked ONNX session."""
    with patch("onnxruntime.InferenceSession"):
        return YOLODetector("dummy.onnx", input_size=input_size)


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


class TestParseArgs:
    def test_required_args(self) -> None:
        args = parse_args(
            [
                "--input-dir",
                "/tmp/in",
                "--output-dir",
                "/tmp/out",
                "--person-model",
                "person.onnx",
                "--plate-model",
                "plate.onnx",
            ]
        )
        assert args.input_dir == "/tmp/in"
        assert args.output_dir == "/tmp/out"
        assert args.person_model == "person.onnx"
        assert args.plate_model == "plate.onnx"

    def test_default_values(self) -> None:
        args = parse_args(
            [
                "--input-dir",
                "/tmp/in",
                "--output-dir",
                "/tmp/out",
                "--person-model",
                "person.onnx",
                "--plate-model",
                "plate.onnx",
            ]
        )
        assert args.conf_threshold == 0.25
        assert args.iou_threshold == 0.45
        assert args.mosaic_ratio == 0.05

    def test_custom_thresholds(self) -> None:
        args = parse_args(
            [
                "--input-dir",
                "/tmp/in",
                "--output-dir",
                "/tmp/out",
                "--person-model",
                "person.onnx",
                "--plate-model",
                "plate.onnx",
                "--conf-threshold",
                "0.5",
                "--iou-threshold",
                "0.6",
                "--mosaic-ratio",
                "0.1",
            ]
        )
        assert args.conf_threshold == 0.5
        assert args.iou_threshold == 0.6
        assert args.mosaic_ratio == 0.1


class TestMainIntegration:
    def test_main_processes_images(self, tmp_path: Any) -> None:
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Create test images
        for name in ["test1.jpg", "test2.png"]:
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(input_dir / name), img)

        with (
            patch("mosaic_processor.YOLODetector"),
            patch("mosaic_processor.MosaicProcessor") as mock_proc_cls,
        ):
            mock_proc = MagicMock()
            mock_proc.process_image.return_value = (np.zeros((100, 100, 3), dtype=np.uint8), 1, 2)
            mock_proc_cls.return_value = mock_proc

            from mosaic_processor import main

            main(
                [
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(output_dir),
                    "--person-model",
                    "person.onnx",
                    "--plate-model",
                    "plate.onnx",
                ]
            )

        assert output_dir.exists()
        output_files = list(output_dir.iterdir())
        assert len(output_files) == 2

    def test_main_skips_non_image_files(self, tmp_path: Any) -> None:
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Create a non-image file and an image file
        (input_dir / "readme.txt").write_text("hello")
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(input_dir / "test.jpg"), img)

        with (
            patch("mosaic_processor.YOLODetector"),
            patch("mosaic_processor.MosaicProcessor") as mock_proc_cls,
        ):
            mock_proc = MagicMock()
            mock_proc.process_image.return_value = (np.zeros((100, 100, 3), dtype=np.uint8), 0, 0)
            mock_proc_cls.return_value = mock_proc

            from mosaic_processor import main

            main(
                [
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(output_dir),
                    "--person-model",
                    "person.onnx",
                    "--plate-model",
                    "plate.onnx",
                ]
            )

        output_files = list(output_dir.iterdir())
        assert len(output_files) == 1
        assert output_files[0].name == "test.jpg"
