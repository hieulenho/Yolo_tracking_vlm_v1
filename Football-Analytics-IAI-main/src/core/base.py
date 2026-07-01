"""Abstract interfaces for detection and tracking components."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from src.core.types import Detection, TrackedDetection


class BaseDetector(ABC):
    """Object detector interface."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run object detection on one BGR frame."""

    @abstractmethod
    def load_weights(self, weights_path: str | Path) -> None:
        """Load model weights."""


class BaseTracker(ABC):
    """Multi-object tracker interface returning detections with track IDs."""

    @abstractmethod
    def update(self, frame: np.ndarray) -> list[TrackedDetection]:
        """Update tracking state from one BGR frame."""
