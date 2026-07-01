"""Train YOLOv8 cho player detection."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO players detector")
    p.add_argument("--data", required=True, help="Path tới data yaml")
    p.add_argument("--model", default="yolov8s.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--project", default="runs/players")
    p.add_argument("--name", default="exp")
    p.add_argument("--device", default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--log-file", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(level="INFO", log_file=args.log_file)
    logger.info(f"[bold]Training players detector[/bold]: {args.model} on {args.data}")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        resume=args.resume,
        patience=args.patience,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        save=True,
        save_period=5,
        plots=True,
    )
    logger.info(f"[bold green]Done[/bold green]: weights in {args.project}/{args.name}/weights/")


if __name__ == "__main__":
    main()
