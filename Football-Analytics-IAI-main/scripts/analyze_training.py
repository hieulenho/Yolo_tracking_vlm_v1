"""Generate player-detector training plots from a YOLO results.csv file."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", default="reports/figures/players_results.csv")
    parser.add_argument("--output-dir", default="reports/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_csv = Path(args.results_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Players Model - Learning Curves (YOLOv8m)", fontsize=14, fontweight="bold")

    axes[0, 0].plot(df["epoch"], df["train/box_loss"], label="train")
    axes[0, 0].plot(df["epoch"], df["val/box_loss"], label="val")
    axes[0, 0].set_title("Box loss")
    axes[0, 0].legend()

    axes[0, 1].plot(df["epoch"], df["train/cls_loss"], label="train")
    axes[0, 1].plot(df["epoch"], df["val/cls_loss"], label="val")
    axes[0, 1].set_title("Class loss")
    axes[0, 1].legend()

    axes[1, 0].plot(df["epoch"], df["metrics/mAP50(B)"], color="#1f77b4")
    axes[1, 0].set_title("mAP@0.5")

    axes[1, 1].plot(df["epoch"], df["metrics/mAP50-95(B)"], color="#2ca02c")
    axes[1, 1].set_title("mAP@0.5:0.95")

    for ax in axes.ravel():
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    save_path = output_dir / "players_learning_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
