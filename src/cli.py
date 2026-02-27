"""CLIエントリーポイント."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from .detection import IMAGE_EXTENSIONS
from .detector import YOLODetector
from .mosaic import MosaicProcessor


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="駐車場画像のナンバープレート・人物にモザイクをかけるCLIツール")
    parser.add_argument("--input-dir", required=True, help="入力画像ディレクトリ")
    parser.add_argument("--output-dir", required=True, help="出力画像ディレクトリ")
    parser.add_argument("--person-model", required=True, help="人物検出ONNXモデルパス")
    parser.add_argument("--plate-model", required=True, help="ナンバープレート検出ONNXモデルパス")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="検出信頼度の閾値")
    parser.add_argument("--iou-threshold", type=float, default=0.45, help="NMSのIoU閾値")
    parser.add_argument("--mosaic-ratio", type=float, default=0.05, help="モザイクの粗さ")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    person_detector = YOLODetector(args.person_model)
    plate_detector = YOLODetector(args.plate_model)
    processor = MosaicProcessor(person_detector, plate_detector, args.mosaic_ratio)

    image_files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    total = len(image_files)
    total_persons = 0
    total_plates = 0
    processed = 0
    start_time = time.time()

    for idx, img_path in enumerate(image_files, 1):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[{idx}/{total}] {img_path.name} - 読み込み失敗、スキップ", file=sys.stderr)
            continue

        result, person_count, plate_count = processor.process_image(
            image,  # type: ignore[arg-type]
            args.conf_threshold,
            args.iou_threshold,
        )
        cv2.imwrite(str(output_dir / img_path.name), result)

        total_persons += person_count
        total_plates += plate_count
        processed += 1
        print(f"[{idx}/{total}] {img_path.name} - person: {person_count}, plate: {plate_count}", file=sys.stderr)

    elapsed = time.time() - start_time
    print(f"\n処理完了: {processed}/{total}枚", file=sys.stderr)
    print(f"検出数 - 人物: {total_persons}, ナンバープレート: {total_plates}", file=sys.stderr)
    print(f"処理時間: {elapsed:.1f}秒", file=sys.stderr)
