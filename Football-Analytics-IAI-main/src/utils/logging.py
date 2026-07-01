"""
Structured logging với rich formatting.

Thay vì dùng `print` lung tung, dùng logger để:
- Có level (DEBUG, INFO, WARNING, ERROR)
- Format thống nhất, có timestamp, module
- Dễ tắt log của 1 module mà không ảnh hưởng khác
- Output ra file dễ hơn
"""
from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    show_path: bool = False,
) -> None:
    """
    Setup global logging config. Gọi 1 lần ở entry point (script main).

    Args:
        level: log level — DEBUG | INFO | WARNING | ERROR
        log_file: nếu set, ghi log ra cả file
        show_path: hiện đường dẫn file source trong log (debug)
    """
    handlers: list[logging.Handler] = [
        RichHandler(rich_tracebacks=True, show_path=show_path, markup=True)
    ]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Get module-level logger. Convention: get_logger(__name__) ở đầu mỗi file."""
    return logging.getLogger(name)
