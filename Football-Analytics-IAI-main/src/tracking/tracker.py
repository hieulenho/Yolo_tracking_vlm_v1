"""Multi-object tracking with YOLOv8m detections and DeepSORT association."""
from __future__ import annotations

from pathlib import Path
from zlib import crc32

import numpy as np

from src.core.base import BaseTracker
from src.core.exceptions import ModelLoadError
from src.core.types import BoundingBox, TrackedDetection
from src.detection.player_detector import PlayerDetector
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DeepSORTTracker(BaseTracker):
    """
    Tracking-by-detection pipeline:

    1. YOLOv8m detects players, goalkeepers, ball, and referee per frame.
    2. DeepSORT predicts track motion with a Kalman filter.
    3. DeepSORT associates detections using motion and visual appearance.
    4. Confirmed tracks are returned with stable IDs for visualization.
    """

    def __init__(
        self,
        weights_path: str | Path,
        conf: float = 0.35,
        iou: float = 0.5,
        imgsz: int = 640,
        device: str | int = "cpu",
        max_age: int = 30,
        n_init: int = 3,
        max_cosine_distance: float = 0.2,
        nn_budget: int | None = 100,
        embedder: str = "mobilenet",
        half: bool = False,
        bgr: bool = True,
        embedder_gpu: bool = False,
        only_confirmed: bool = False,
    ) -> None:
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError as exc:
            raise ModelLoadError(
                "DeepSORT dependency is missing. Install it with "
                "`pip install deep-sort-realtime` or `pip install -r requirements.txt`."
            ) from exc

        self.detector = PlayerDetector(
            weights_path=weights_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
        )
        names = self.detector.model.names
        name_items = names.items() if isinstance(names, dict) else enumerate(names)
        self.class_id_by_name = {name: class_id for class_id, name in name_items}
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            embedder=embedder,
            half=half,
            bgr=bgr,
            embedder_gpu=embedder_gpu,
        )
        self.only_confirmed = only_confirmed
        logger.info("[green]OK[/green] Loaded tracker: YOLOv8m + DeepSORT")

    def update(self, frame: np.ndarray) -> list[TrackedDetection]:
        detections = self.detector.detect(frame)
        deepsort_inputs = []

        for det in detections:
            x1, y1, x2, y2 = det.bbox.to_xyxy()
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            if width == 0 or height == 0:
                continue
            deepsort_inputs.append(([x1, y1, width, height], det.confidence, det.class_name))

        tracks = self.tracker.update_tracks(deepsort_inputs, frame=frame)
        output: list[TrackedDetection] = []

        for track in tracks:
            if track.time_since_update > 0:
                continue
            if self.only_confirmed and not track.is_confirmed():
                continue

            x1, y1, x2, y2 = track.to_ltrb()
            class_name = track.get_det_class() or "object"
            confidence = track.get_det_conf()
            class_id = self.class_id_by_name.get(class_name, -1)

            output.append(
                TrackedDetection(
                    bbox=BoundingBox.from_xyxy([x1, y1, x2, y2]),
                    confidence=float(confidence) if confidence is not None else 0.0,
                    class_id=class_id,
                    class_name=class_name,
                    track_id=self._stable_track_id(track.track_id),
                )
            )

        return output

    @staticmethod
    def _stable_track_id(raw_id: object) -> int:
        raw = str(raw_id)
        if raw.isdigit():
            return int(raw)
        return crc32(raw.encode("utf-8")) & 0x7FFFFFFF


def tracker_factory(
    weights_path: str | Path,
    tracker: str = "deepsort",
    **kwargs,
) -> BaseTracker:
    """Create the supported tracker for this project."""
    if tracker.lower() != "deepsort":
        logger.warning("Only DeepSORT is supported now. Falling back to DeepSORT.")
    return DeepSORTTracker(weights_path=weights_path, **kwargs)
