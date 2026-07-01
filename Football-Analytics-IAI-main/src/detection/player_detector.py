"""YOLOv8 wrapper for football object detection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from src.core.base import BaseDetector
from src.core.exceptions import ModelLoadError
from src.core.types import BoundingBox, Detection
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PlayerDetector(BaseDetector):
    """YOLOv8m detector for players, goalkeepers, ball, and referee."""

    def __init__(
        self,
        weights_path: str | Path,
        conf: float = 0.35,
        iou: float = 0.5,
        imgsz: int = 640,
        device: str | int = "cpu",
    ) -> None:
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.model: YOLO | None = None
        self.load_weights(weights_path)

    def load_weights(self, weights_path: str | Path) -> None:
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise ModelLoadError(f"Weights not found: {weights_path}")
        try:
            self.model = YOLO(str(weights_path))
            logger.info(f"[green]OK[/green] Loaded YOLOv8 detector: {weights_path.name}")
        except Exception as exc:
            raise ModelLoadError(f"Failed to load {weights_path}: {exc}") from exc

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self.model is None:
            raise ModelLoadError("Model not loaded")

        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        return self._results_to_detections(results)

    def _results_to_detections(self, results) -> list[Detection]:
        if self.model is None:
            raise ModelLoadError("Model not loaded")

        detections: list[Detection] = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            detections.append(
                Detection(
                    bbox=BoundingBox.from_xyxy(box.xyxy[0].cpu().numpy()),
                    confidence=float(box.conf[0]),
                    class_id=class_id,
                    class_name=self.model.names[class_id],
                )
            )
        return detections
