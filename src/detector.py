"""YOLOv8ベースのオブジェクト検出器."""

from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from .detection import Detection


class YOLODetector:
    def __init__(self, model_path: str, input_size: int = 640) -> None:
        self.input_size = input_size
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

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
