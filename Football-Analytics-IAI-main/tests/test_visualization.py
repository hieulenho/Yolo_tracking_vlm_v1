"""Tests for tracking visualization helpers."""
import numpy as np

from src.core.types import BoundingBox, TrackedDetection
from src.utils.visualization import _id_to_color, draw_tracked_detections


def _track(track_id: int) -> TrackedDetection:
    return TrackedDetection(
        bbox=BoundingBox(20, 20, 80, 120),
        confidence=0.9,
        class_id=0,
        class_name="player",
        track_id=track_id,
    )


def test_track_ids_get_distinct_stable_colors():
    assert _id_to_color(7) == _id_to_color(7)
    assert _id_to_color(7) != _id_to_color(8)


def test_draw_tracked_detections_changes_frame():
    frame = np.zeros((160, 160, 3), dtype=np.uint8)
    annotated = draw_tracked_detections(frame, [_track(1), _track(2)])

    assert annotated.shape == frame.shape
    assert np.count_nonzero(annotated) > 0
    assert np.array_equal(frame, np.zeros_like(frame))
