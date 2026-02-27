"""駐車場画像のナンバープレート・人物を自動検出しモザイク処理するCLIツール."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int


class YOLODetector:
    def __init__(self, model_path: str, input_size: int = 640) -> None:
        self.input_size = input_size
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    def _preprocess(self, image: NDArray[np.uint8]) -> tuple[NDArray[np.float32], float, tuple[float, float]]:
        h, w = image.shape[:2]
        ratio = self.input_size / max(h, w)
        new_w, new_h = int(w * ratio), int(h * ratio)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = (self.input_size - new_w) / 2
        pad_h = (self.input_size - new_h) / 2
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        blob: NDArray[np.float32] = padded[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB, normalize
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # HWC -> CHW -> BCHW
        return blob, ratio, (pad_w, pad_h)

    def _postprocess(
        self,
        output: NDArray[np.float32],
        ratio: float,
        pad_w: float,
        pad_h: float,
        conf_thresh: float,
        iou_thresh: float,
        target_classes: list[int] | None = None,
    ) -> list[Detection]:
        # output shape: (1, 4+num_classes, num_boxes) -> (num_boxes, 4+num_classes)
        preds = output[0].T

        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]

        max_scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)

        # Confidence filter
        mask = max_scores >= conf_thresh
        if target_classes is not None:
            class_mask = np.isin(class_ids, target_classes)
            mask = mask & class_mask

        boxes_xywh = boxes_xywh[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return []

        # xywh -> xyxy
        boxes_xyxy = np.zeros_like(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2

        # Inverse transform: remove padding, then scale back
        boxes_xyxy[:, 0] = (boxes_xyxy[:, 0] - pad_w) / ratio
        boxes_xyxy[:, 1] = (boxes_xyxy[:, 1] - pad_h) / ratio
        boxes_xyxy[:, 2] = (boxes_xyxy[:, 2] - pad_w) / ratio
        boxes_xyxy[:, 3] = (boxes_xyxy[:, 3] - pad_h) / ratio

        # NMS
        nms_boxes = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes_xyxy]
        indices = cv2.dnn.NMSBoxes(nms_boxes, max_scores.tolist(), conf_thresh, iou_thresh)

        detections: list[Detection] = []
        for i in indices:
            idx = int(i)
            detections.append(
                Detection(
                    x1=int(round(boxes_xyxy[idx, 0])),
                    y1=int(round(boxes_xyxy[idx, 1])),
                    x2=int(round(boxes_xyxy[idx, 2])),
                    y2=int(round(boxes_xyxy[idx, 3])),
                    confidence=float(max_scores[idx]),
                    class_id=int(class_ids[idx]),
                )
            )
        return detections

    def detect(
        self,
        image: NDArray[np.uint8],
        conf_thresh: float,
        iou_thresh: float,
        target_classes: list[int] | None = None,
    ) -> list[Detection]:
        blob, ratio, (pad_w, pad_h) = self._preprocess(image)
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: blob})
        return self._postprocess(outputs[0], ratio, pad_w, pad_h, conf_thresh, iou_thresh, target_classes)


class MosaicProcessor:
    PERSON_CLASS_ID = 0
    PLATE_CLASS_ID = 0

    def __init__(self, person_detector: YOLODetector, plate_detector: YOLODetector, mosaic_ratio: float) -> None:
        self.person_detector = person_detector
        self.plate_detector = plate_detector
        self.mosaic_ratio = mosaic_ratio

    def _apply_mosaic(self, image: NDArray[np.uint8], bbox: tuple[int, int, int, int]) -> NDArray[np.uint8]:
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return image

        roi = image[y1:y2, x1:x2]
        rh, rw = roi.shape[:2]
        small_w = max(1, int(rw * self.mosaic_ratio))
        small_h = max(1, int(rh * self.mosaic_ratio))
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        mosaic = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
        image[y1:y2, x1:x2] = mosaic
        return image

    def process_image(
        self,
        image: NDArray[np.uint8],
        conf_thresh: float,
        iou_thresh: float,
    ) -> tuple[NDArray[np.uint8], int, int]:
        persons = self.person_detector.detect(image, conf_thresh, iou_thresh, target_classes=[self.PERSON_CLASS_ID])
        plates = self.plate_detector.detect(image, conf_thresh, iou_thresh, target_classes=[self.PLATE_CLASS_ID])

        for d in persons + plates:
            image = self._apply_mosaic(image, (d.x1, d.y1, d.x2, d.y2))

        return image, len(persons), len(plates)


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
            image, args.conf_threshold, args.iou_threshold  # type: ignore[arg-type]
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


if __name__ == "__main__":
    main()
