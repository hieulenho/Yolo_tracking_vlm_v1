"""Drawing helpers for football detection and tracking visualization."""
from __future__ import annotations

import colorsys
from typing import Sequence

import cv2
import numpy as np

from src.core.types import TrackedDetection


BALL_COLOR: tuple[int, int, int] = (25, 210, 60)
UNKNOWN_COLOR: tuple[int, int, int] = (80, 80, 80)
TEXT_COLOR: tuple[int, int, int] = (255, 255, 255)
TEXT_OUTLINE: tuple[int, int, int] = (0, 0, 0)


def _id_to_color(track_id: int) -> tuple[int, int, int]:
    """Map a track ID to one stable, saturated BGR color."""
    golden_ratio = 0.6180339887
    hue = (track_id * golden_ratio) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.9, 0.85)
    return (int(blue * 255), int(green * 255), int(red * 255))


def _label_for(det: TrackedDetection) -> str:
    if det.track_id >= 0:
        return f"ID {det.track_id} | {det.class_name}"
    return f"{det.class_name} {det.confidence:.2f}"


def _draw_label(
    frame: np.ndarray,
    label: str,
    anchor: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float,
    font_thickness: int,
) -> None:
    x, y = anchor
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    pad_x = 8
    pad_y = 6
    top = max(0, y - text_h - baseline - (pad_y * 2))
    bottom = top + text_h + baseline + (pad_y * 2)
    right = x + text_w + (pad_x * 2)

    cv2.rectangle(frame, (x, top), (right, bottom), TEXT_OUTLINE, cv2.FILLED)
    cv2.rectangle(frame, (x + 2, top + 2), (right - 2, bottom - 2), color, cv2.FILLED)

    text_origin = (x + pad_x, bottom - baseline - pad_y)
    cv2.putText(
        frame,
        label,
        text_origin,
        font,
        font_scale,
        TEXT_OUTLINE,
        font_thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        text_origin,
        font,
        font_scale,
        TEXT_COLOR,
        font_thickness,
        cv2.LINE_AA,
    )


def draw_tracked_detections(
    frame: np.ndarray,
    detections: Sequence[TrackedDetection],
    box_thickness: int = 4,
    font_scale: float = 0.7,
    font_thickness: int = 3,
) -> np.ndarray:
    """
    Draw DeepSORT tracks on a copy of the frame.

    Every confirmed track ID gets a deterministic color. Labels are deliberately
    thick and high-contrast so the ID remains readable on moving football video.
    """
    output = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det.bbox.to_xyxy()
        if det.track_id >= 0:
            color = _id_to_color(det.track_id)
        elif det.class_name == "ball":
            color = BALL_COLOR
        else:
            color = UNKNOWN_COLOR

        cv2.rectangle(output, (x1, y1), (x2, y2), TEXT_OUTLINE, box_thickness + 2)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, box_thickness)
        _draw_label(
            output,
            _label_for(det),
            (x1, y1),
            color,
            font_scale=font_scale,
            font_thickness=font_thickness,
        )

    return output


def plot_training_curves(results_csv: str, save_path: str | None = None):
    """Plot YOLO training curves from a results.csv file."""
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(df["epoch"], df["train/box_loss"], label="train")
    axes[0, 0].plot(df["epoch"], df["val/box_loss"], label="val")
    axes[0, 0].set_title("Box loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()

    axes[0, 1].plot(df["epoch"], df["train/cls_loss"], label="train")
    axes[0, 1].plot(df["epoch"], df["val/cls_loss"], label="val")
    axes[0, 1].set_title("Classification loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()

    axes[1, 0].plot(df["epoch"], df["metrics/mAP50(B)"])
    axes[1, 0].set_title("mAP@0.5")
    axes[1, 0].set_xlabel("Epoch")

    axes[1, 1].plot(df["epoch"], df["metrics/mAP50-95(B)"])
    axes[1, 1].set_title("mAP@0.5:0.95")
    axes[1, 1].set_xlabel("Epoch")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
