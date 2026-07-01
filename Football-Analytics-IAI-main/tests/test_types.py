"""Tests for core detection and tracking data types."""
import numpy as np
import pytest

from src.core.types import BoundingBox, FrameResult, TrackedDetection


class TestBoundingBox:
    def test_basic(self):
        bbox = BoundingBox(10, 20, 50, 100)
        assert bbox.width == 40
        assert bbox.height == 80
        assert bbox.area == 3200
        assert bbox.center == (30, 60)

    def test_invalid(self):
        with pytest.raises(ValueError):
            BoundingBox(50, 20, 10, 100)

    def test_from_xyxy(self):
        bbox = BoundingBox.from_xyxy(np.array([10, 20, 50, 100]))
        assert bbox.x1 == 10
        assert bbox.x2 == 50

    def test_to_xyxy(self):
        bbox = BoundingBox(10.5, 20.3, 50.7, 100.9)
        assert bbox.to_xyxy() == (10, 20, 50, 100)


class TestFrameResult:
    def test_empty(self):
        result = FrameResult(frame_idx=0)
        assert result.detections == []

    def test_filter_by_class(self):
        player = TrackedDetection(
            bbox=BoundingBox(0, 0, 10, 10),
            confidence=0.9,
            class_id=0,
            class_name="player",
            track_id=1,
        )
        ball = TrackedDetection(
            bbox=BoundingBox(20, 20, 30, 30),
            confidence=0.8,
            class_id=2,
            class_name="ball",
            track_id=2,
        )
        result = FrameResult(frame_idx=0, detections=[player, ball])
        assert result.get_detections_by_class("player") == [player]
        assert result.get_detections_by_class("ball") == [ball]
