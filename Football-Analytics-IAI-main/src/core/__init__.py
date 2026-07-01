"""Core types and exceptions."""
from src.core.exceptions import ConfigError, FootballAnalyticsError, ModelLoadError
from src.core.types import BoundingBox, Detection, FrameResult, ObjectClass, TrackedDetection

__all__ = [
    "BoundingBox",
    "ConfigError",
    "Detection",
    "FootballAnalyticsError",
    "FrameResult",
    "ModelLoadError",
    "ObjectClass",
    "TrackedDetection",
]
