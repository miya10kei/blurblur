"""Tests for blurblur.cli module."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from blurblur.cli import main, parse_args


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
            patch("blurblur.cli.YOLODetector"),
            patch("blurblur.cli.MosaicProcessor") as mock_proc_cls,
        ):
            mock_proc = MagicMock()
            mock_proc.process_image.return_value = (np.zeros((100, 100, 3), dtype=np.uint8), 1, 2)
            mock_proc_cls.return_value = mock_proc

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
            patch("blurblur.cli.YOLODetector"),
            patch("blurblur.cli.MosaicProcessor") as mock_proc_cls,
        ):
            mock_proc = MagicMock()
            mock_proc.process_image.return_value = (np.zeros((100, 100, 3), dtype=np.uint8), 0, 0)
            mock_proc_cls.return_value = mock_proc

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
