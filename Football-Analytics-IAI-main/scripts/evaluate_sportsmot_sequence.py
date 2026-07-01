"""Evaluate one SportsMOT sequence prediction against gt/gt.txt.

This is a lightweight local evaluator for quick comparison. It reports common
MOT-style metrics (MOTA, IDF1, precision, recall, ID switches). For official
SportsMOT HOTA leaderboard comparison, use TrackEval with the same prediction
file exported by run_sportsmot_sequence.py.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", required=True, help="SportsMOT sequence with gt/gt.txt")
    parser.add_argument("--pred", required=True, help="Prediction txt in MOT format")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def load_mot(path: Path, has_score: bool) -> dict[int, list[dict]]:
    frames: dict[int, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            frame_id = int(float(row[0]))
            track_id = int(float(row[1]))
            x, y, w, h = [float(v) for v in row[2:6]]
            score = float(row[6]) if has_score and len(row) > 6 else 1.0
            if score <= 0:
                continue
            frames[frame_id].append(
                {"id": track_id, "bbox": np.array([x, y, x + w, y + h], dtype=float)}
            )
    return frames


def iou_matrix(gt: list[dict], pred: list[dict]) -> np.ndarray:
    matrix = np.zeros((len(gt), len(pred)), dtype=float)
    for i, gt_item in enumerate(gt):
        gx1, gy1, gx2, gy2 = gt_item["bbox"]
        g_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
        for j, pred_item in enumerate(pred):
            px1, py1, px2, py2 = pred_item["bbox"]
            p_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
            ix1 = max(gx1, px1)
            iy1 = max(gy1, py1)
            ix2 = min(gx2, px2)
            iy2 = min(gy2, py2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = g_area + p_area - inter
            matrix[i, j] = inter / union if union > 0 else 0.0
    return matrix


def match_frame(gt: list[dict], pred: list[dict], threshold: float):
    if not gt or not pred:
        return [], list(range(len(gt))), list(range(len(pred)))

    ious = iou_matrix(gt, pred)
    rows, cols = linear_sum_assignment(-ious)
    matches = []
    matched_gt = set()
    matched_pred = set()

    for row, col in zip(rows, cols):
        if ious[row, col] >= threshold:
            matches.append((row, col, float(ious[row, col])))
            matched_gt.add(row)
            matched_pred.add(col)

    unmatched_gt = [idx for idx in range(len(gt)) if idx not in matched_gt]
    unmatched_pred = [idx for idx in range(len(pred)) if idx not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


def evaluate(gt_by_frame: dict[int, list[dict]], pred_by_frame: dict[int, list[dict]], threshold: float):
    all_frames = sorted(set(gt_by_frame) | set(pred_by_frame))
    total_gt = sum(len(v) for v in gt_by_frame.values())
    total_pred = sum(len(v) for v in pred_by_frame.values())

    tp = fp = fn = id_switches = 0
    iou_sum = 0.0
    last_match_for_gt: dict[int, int] = {}
    global_pair_counts: dict[tuple[int, int], int] = defaultdict(int)

    for frame_id in all_frames:
        gt = gt_by_frame.get(frame_id, [])
        pred = pred_by_frame.get(frame_id, [])
        matches, unmatched_gt, unmatched_pred = match_frame(gt, pred, threshold)

        tp += len(matches)
        fn += len(unmatched_gt)
        fp += len(unmatched_pred)

        for gt_idx, pred_idx, iou in matches:
            gt_id = gt[gt_idx]["id"]
            pred_id = pred[pred_idx]["id"]
            if gt_id in last_match_for_gt and last_match_for_gt[gt_id] != pred_id:
                id_switches += 1
            last_match_for_gt[gt_id] = pred_id
            global_pair_counts[(gt_id, pred_id)] += 1
            iou_sum += iou

    mota = 1.0 - ((fp + fn + id_switches) / total_gt) if total_gt else 0.0
    motp = iou_sum / tp if tp else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    idtp = compute_global_idtp(global_pair_counts)
    idfp = total_pred - idtp
    idfn = total_gt - idtp
    idf1 = (2 * idtp) / ((2 * idtp) + idfp + idfn) if total_gt + total_pred else 0.0

    return {
        "iou_threshold": threshold,
        "gt_detections": total_gt,
        "pred_detections": total_pred,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "id_switches": id_switches,
        "MOTA": mota,
        "MOTP": motp,
        "precision": precision,
        "recall": recall,
        "IDF1": idf1,
    }


def compute_global_idtp(pair_counts: dict[tuple[int, int], int]) -> int:
    if not pair_counts:
        return 0

    gt_ids = sorted({pair[0] for pair in pair_counts})
    pred_ids = sorted({pair[1] for pair in pair_counts})
    gt_index = {track_id: idx for idx, track_id in enumerate(gt_ids)}
    pred_index = {track_id: idx for idx, track_id in enumerate(pred_ids)}
    matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=float)

    for (gt_id, pred_id), count in pair_counts.items():
        matrix[gt_index[gt_id], pred_index[pred_id]] = count

    rows, cols = linear_sum_assignment(-matrix)
    return int(sum(matrix[row, col] for row, col in zip(rows, cols)))


def main() -> None:
    args = parse_args()
    sequence_dir = Path(args.sequence_dir)
    gt_path = sequence_dir / "gt" / "gt.txt"
    pred_path = Path(args.pred)
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing ground truth: {gt_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")

    gt = load_mot(gt_path, has_score=False)
    pred = load_mot(pred_path, has_score=True)
    metrics = evaluate(gt, pred, args.iou_threshold)

    print(json.dumps(metrics, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
