"""モザイク処理モジュール."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .detector import YOLODetector


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
