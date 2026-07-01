"""
Typed data structures using dataclasses.

Vì sao dùng dataclass thay vì dict?
- Type safety: IDE catch lỗi compile-time
- Self-documenting: nhìn class biết ngay structure
- Immutable option (frozen=True) khi cần
- Validation tự động qua __post_init__
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np


class ObjectClass(IntEnum):
    """Class id mapping cho player detector."""

    PLAYER = 0
    GOALKEEPER = 1
    BALL = 2
    REFEREE = 3


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bbox theo định dạng xyxy (corner)."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"Invalid bbox: ({self.x1}, {self.y1}, {self.x2}, {self.y2})")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @classmethod
    def from_xyxy(cls, xyxy: np.ndarray | list) -> BoundingBox:
        return cls(*[float(v) for v in xyxy])

    def to_xyxy(self) -> tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))


@dataclass
class Detection:
    """1 detection từ object detector."""

    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str


@dataclass
class TrackedDetection(Detection):
    """Detection có track_id (sau khi qua tracker)."""

    track_id: int = -1


@dataclass
class FrameResult:
    """
    Toàn bộ output xử lý cho 1 frame.

    Dùng làm interface chuẩn giữa pipeline và downstream consumers
    (visualization, app, evaluation, etc.).
    """

    frame_idx: int
    detections: list[TrackedDetection] = field(default_factory=list)

    def get_detections_by_class(self, class_name: str) -> list[TrackedDetection]:
        return [d for d in self.detections if d.class_name == class_name]
