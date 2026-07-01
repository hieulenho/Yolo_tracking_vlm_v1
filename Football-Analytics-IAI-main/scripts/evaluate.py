"""
Evaluate YOLO trên val set. Export metrics ra JSON cho báo cáo.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table
from ultralytics import YOLO

from src.utils.io import save_json
from src.utils.logging import get_logger, setup_logging
from src.utils.metrics import yolo_metrics_to_dict

logger = get_logger(__name__)
console = Console()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=0)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()
    logger.info(f"Evaluating {args.weights} on {args.split} split")

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
        plots=True,
        verbose=False,
    )
    result = yolo_metrics_to_dict(metrics)

    table = Table(title="Evaluation Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("mAP@0.5", f"{result['mAP50']:.4f}")
    table.add_row("mAP@0.5:0.95", f"{result['mAP50_95']:.4f}")
    table.add_row("Precision", f"{result['precision']:.4f}")
    table.add_row("Recall", f"{result['recall']:.4f}")
    console.print(table)

    if result["per_class"]:
        per_cls = Table(title="Per-class", show_header=True)
        per_cls.add_column("Class", style="cyan")
        per_cls.add_column("mAP@0.5", style="green")
        per_cls.add_column("mAP@0.5:0.95", style="green")
        for name, m in result["per_class"].items():
            per_cls.add_row(name, f"{m['mAP50']:.4f}", f"{m['mAP50_95']:.4f}")
        console.print(per_cls)

    if args.output_json:
        save_json(result, args.output_json)
        logger.info(f"Saved metrics → {args.output_json}")


if __name__ == "__main__":
    main()
