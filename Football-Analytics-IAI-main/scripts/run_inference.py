"""Run YOLOv8m + DeepSORT inference on a football video."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.video_processor import TrackingPipeline
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Football Player Detection and Multi-Object Tracking "
            "using YOLOv8m and DeepSORT"
        )
    )
    parser.add_argument("--config", default="configs/inference.yaml", help="Path to config YAML")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", default="output.mp4", help="Output video path")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level=args.log_level)

    pipeline = TrackingPipeline(args.config)
    pipeline.process_video(args.video, args.output)
    logger.info(f"[bold green]Done[/bold green] Output: {args.output}")


if __name__ == "__main__":
    main()
