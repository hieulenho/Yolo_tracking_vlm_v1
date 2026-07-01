"""Football player detection and multi-object tracking pipeline."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.core.types import FrameResult
from src.tracking.tracker import tracker_factory
from src.utils.io import load_yaml
from src.utils.logging import get_logger
from src.utils.visualization import draw_tracked_detections

logger = get_logger(__name__)


class TrackingPipeline:
    """
    End-to-end pipeline focused on YOLOv8m detection and DeepSORT tracking.

    Per frame:
    1. YOLOv8m detects football objects.
    2. DeepSORT assigns stable multi-object track IDs.
    3. The renderer draws thick boxes and high-contrast ID labels.
    """

    def __init__(self, config_path: str | Path):
        cfg = load_yaml(config_path)
        self.cfg = cfg

        det_cfg = cfg["detection"]
        trk_cfg = cfg.get("tracking", {})

        self.tracker = tracker_factory(
            weights_path=cfg["models"]["players_weights"],
            tracker=trk_cfg.get("tracker", "deepsort"),
            conf=det_cfg.get("conf_threshold", 0.35),
            iou=det_cfg.get("iou_threshold", 0.5),
            imgsz=det_cfg.get("imgsz", 640),
            device=det_cfg.get("device", "cpu"),
            max_age=trk_cfg.get("max_age", 30),
            n_init=trk_cfg.get("n_init", 3),
            max_cosine_distance=trk_cfg.get("max_cosine_distance", 0.2),
            nn_budget=trk_cfg.get("nn_budget", 100),
            embedder=trk_cfg.get("embedder", "mobilenet"),
            half=trk_cfg.get("half", False),
            bgr=trk_cfg.get("bgr", True),
            embedder_gpu=trk_cfg.get("embedder_gpu", False),
            only_confirmed=trk_cfg.get("only_confirmed", False),
        )

        vis_cfg = cfg.get("visualization", {})
        self.box_thickness = vis_cfg.get("box_thickness", 4)
        self.font_scale = vis_cfg.get("font_scale", 0.7)
        self.font_thickness = vis_cfg.get("font_thickness", 3)

        logger.info("[green]OK[/green] TrackingPipeline ready")

    def process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> FrameResult:
        result = FrameResult(frame_idx=frame_idx)
        result.detections = self.tracker.update(frame)
        return result

    def render(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        return draw_tracked_detections(
            frame,
            result.detections,
            box_thickness=self.box_thickness,
            font_scale=self.font_scale,
            font_thickness=self.font_thickness,
        )

    def process_video(
        self,
        video_path: str | Path,
        output_path: str | Path,
        progress: bool = True,
    ) -> None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise OSError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video: {width}x{height} @ {fps:.1f}fps, {total} frames")

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        iterator = range(total)
        if progress:
            iterator = tqdm(iterator, desc="DeepSORT tracking", unit="frame")

        for idx in iterator:
            ok, frame = cap.read()
            if not ok:
                break
            result = self.process_frame(frame, frame_idx=idx)
            writer.write(self.render(frame, result))

        cap.release()
        writer.release()
        logger.info(f"[green]OK[/green] Output saved: {output_path}")


FootballVideoProcessor = TrackingPipeline
