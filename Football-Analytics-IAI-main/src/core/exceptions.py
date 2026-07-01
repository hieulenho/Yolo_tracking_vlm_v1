"""Custom exception hierarchy.

Lợi ích: caller có thể catch specific exception thay vì catch broad Exception,
dễ debug, và message lỗi rõ ràng hơn.
"""


class FootballAnalyticsError(Exception):
    """Base exception cho tất cả lỗi của project."""


class ModelLoadError(FootballAnalyticsError):
    """Lỗi khi load model weights."""


class ConfigError(FootballAnalyticsError):
    """Lỗi config invalid."""
