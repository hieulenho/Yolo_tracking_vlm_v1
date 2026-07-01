"""Run YOLOv8m + DeepSORT on one SportsMOT sequence.

Expected SportsMOT layout:
    dataset/train/VIDEO_NAME/
        img1/000001.jpg
        gt/gt.txt
        seqinfo.ini
"""
from __future__ import annotations

import argparse
import configparser
import json
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.video_processor import TrackingPipeline
from src.utils.logging import setup_logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", required=True, help="Path to one SportsMOT sequence")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--output-dir", default="runs/sportsmot")
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_seqinfo(sequence_dir: Path) -> dict:
    seqinfo_path = sequence_dir / "seqinfo.ini"
    if not seqinfo_path.exists():
        raise FileNotFoundError(f"Missing seqinfo.ini: {seqinfo_path}")

    cfg = configparser.ConfigParser()
    cfg.read(seqinfo_path)
    section = cfg["Sequence"]
    return {
        "name": section.get("name", sequence_dir.name),
        "im_dir": section.get("imDir", "img1"),
        "frame_rate": section.getint("frameRate", fallback=25),
        "seq_length": section.getint("seqLength", fallback=0),
        "width": section.getint("imWidth", fallback=0),
        "height": section.getint("imHeight", fallback=0),
        "im_ext": section.get("imExt", ".jpg"),
    }


def detection_to_mot_row(frame_idx: int, det) -> str:
    x1, y1, x2, y2 = det.bbox.to_xyxy()
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return (
        f"{frame_idx},{det.track_id},{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},"
        f"{det.confidence:.6f},-1,-1,-1"
    )


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)

    sequence_dir = Path(args.sequence_dir)
    seqinfo = load_seqinfo(sequence_dir)
    image_dir = sequence_dir / seqinfo["im_dir"]
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    output_dir = Path(args.output_dir)
    pred_dir = output_dir / "preds"
    video_dir = output_dir / "videos"
    summary_dir = output_dir / "summaries"
    pred_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    pred_path = pred_dir / f"{seqinfo['name']}.txt"
    video_path = video_dir / f"{seqinfo['name']}_tracked.mp4"
    summary_path = summary_dir / f"{seqinfo['name']}.json"

    pipeline = TrackingPipeline(args.config)
    seq_length = seqinfo["seq_length"]
    if args.max_frames is not None:
        seq_length = min(seq_length, args.max_frames)

    writer = None
    if args.render_video:
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            seqinfo["frame_rate"],
            (seqinfo["width"], seqinfo["height"]),
        )

    mot_rows: list[str] = []
    total_tracks_seen: set[int] = set()
    processed_frames = 0

    iterator = tqdm(range(1, seq_length + 1), desc=seqinfo["name"], unit="frame")
    for frame_idx in iterator:
        frame_path = image_dir / f"{frame_idx:06d}{seqinfo['im_ext']}"
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise FileNotFoundError(f"Cannot read frame: {frame_path}")

        result = pipeline.process_frame(frame, frame_idx=frame_idx)
        for det in result.detections:
            if det.track_id < 0:
                continue
            mot_rows.append(detection_to_mot_row(frame_idx, det))
            total_tracks_seen.add(det.track_id)

        if writer is not None:
            writer.write(pipeline.render(frame, result))

        processed_frames += 1

    if writer is not None:
        writer.release()

    pred_path.write_text("\n".join(mot_rows) + ("\n" if mot_rows else ""), encoding="utf-8")
    summary = {
        "sequence": seqinfo["name"],
        "frames": processed_frames,
        "detections": len(mot_rows),
        "unique_track_ids": len(total_tracks_seen),
        "prediction_file": str(pred_path),
        "rendered_video": str(video_path) if args.render_video else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved predictions: {pred_path}")
    if args.render_video:
        print(f"Saved video: {video_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
