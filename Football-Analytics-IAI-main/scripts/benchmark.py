"""Benchmark FPS của detection model trên 1 video."""
from __future__ import annotations

import argparse

import cv2
from rich.console import Console
from rich.table import Table

from src.detection.player_detector import PlayerDetector
from src.utils.io import save_json
from src.utils.logging import get_logger, setup_logging
from src.utils.metrics import benchmark_inference

logger = get_logger(__name__)
console = Console()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--device", default=0)
    p.add_argument("--output-json", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()

    # Sample frames từ video
    cap = cv2.VideoCapture(args.video)
    frames = []
    while len(frames) < args.max_frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    logger.info(f"Loaded {len(frames)} frames cho benchmark")

    detector = PlayerDetector(args.weights, device=args.device)
    result = benchmark_inference(detector.detect, frames)

    table = Table(title="Inference Benchmark")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in result.to_dict().items():
        table.add_row(k, str(v))
    console.print(table)

    if args.output_json:
        save_json(result.to_dict(), args.output_json)


if __name__ == "__main__":
    main()
