"""Create a detailed SportsMOT benchmark report against ground truth.

This script reads MOT-format predictions from ``runs/sportsmot/preds`` and
compares them with each sequence's ``gt/gt.txt``. It produces:

- per-sequence benchmark metrics
- overall benchmark metrics
- ID-switch event details
- per-ground-truth-ID identity stability details
- per-frame error details

The HOTA/AssA/DetA values here are local @IoU=0.50 estimates. Official
SportsMOT leaderboard HOTA should still be computed with TrackEval.
"""
from __future__ import annotations

import argparse
import configparser
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table
from scipy.optimize import linear_sum_assignment

console = Console(width=220)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw", help="SportsMOT data root")
    parser.add_argument(
        "--splits-file",
        default="data/raw/splits_txt/football.txt",
        help="Sequence-name list, one sequence per line",
    )
    parser.add_argument("--pred-dir", default="runs/sportsmot/preds")
    parser.add_argument("--output-dir", default="runs/sportsmot/summaries")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to scan",
    )
    parser.add_argument(
        "--top-id-issues",
        type=int,
        default=20,
        help="Rows to print from the worst GT identity table",
    )
    parser.add_argument(
        "--top-frame-issues",
        type=int,
        default=20,
        help="Rows to print from the worst frame-error table",
    )
    parser.add_argument(
        "--top-switch-events",
        type=int,
        default=20,
        help="Rows to print from the ID-switch event table",
    )
    return parser.parse_args()


def load_mot(path: Path, has_score: bool) -> dict[int, list[dict[str, Any]]]:
    frames: dict[int, list[dict[str, Any]]] = defaultdict(list)
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
                {
                    "id": track_id,
                    "bbox": np.array([x, y, x + w, y + h], dtype=float),
                }
            )
    return frames


def load_seq_length(sequence_dir: Path, fallback_frames: set[int]) -> int:
    seqinfo_path = sequence_dir / "seqinfo.ini"
    if seqinfo_path.exists():
        cfg = configparser.ConfigParser()
        cfg.read(seqinfo_path)
        if cfg.has_section("Sequence"):
            seq_length = cfg["Sequence"].getint("seqLength", fallback=0)
            if seq_length > 0:
                return seq_length
    return max(fallback_frames) if fallback_frames else 0


def iou_matrix(gt: list[dict[str, Any]], pred: list[dict[str, Any]]) -> np.ndarray:
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


def match_frame(
    gt: list[dict[str, Any]],
    pred: list[dict[str, Any]],
    threshold: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
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


def compute_idtp(pair_counts: Counter[tuple[Any, Any]]) -> int:
    if not pair_counts:
        return 0

    gt_ids = sorted({pair[0] for pair in pair_counts}, key=str)
    pred_ids = sorted({pair[1] for pair in pair_counts}, key=str)
    gt_index = {track_id: idx for idx, track_id in enumerate(gt_ids)}
    pred_index = {track_id: idx for idx, track_id in enumerate(pred_ids)}
    matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=float)

    for (gt_id, pred_id), count in pair_counts.items():
        matrix[gt_index[gt_id], pred_index[pred_id]] = count

    rows, cols = linear_sum_assignment(-matrix)
    return int(sum(matrix[row, col] for row, col in zip(rows, cols)))


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def pct(value: float) -> float:
    return value * 100.0


def rounded(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def association_accuracy(
    pair_counts: Counter[tuple[int, int]],
    gt_matched_counts: Counter[int],
    pred_matched_counts: Counter[int],
) -> float:
    total_tp = sum(pair_counts.values())
    if total_tp == 0:
        return 0.0

    weighted_sum = 0.0
    for (gt_id, pred_id), count in pair_counts.items():
        denominator = gt_matched_counts[gt_id] + pred_matched_counts[pred_id] - count
        weighted_sum += count * safe_div(count, denominator)
    return weighted_sum / total_tp


def analyze_sequence(
    split: str,
    sequence: str,
    sequence_dir: Path,
    pred_path: Path,
    iou_threshold: float,
) -> dict[str, Any]:
    gt_by_frame = load_mot(sequence_dir / "gt" / "gt.txt", has_score=False)
    pred_by_frame = load_mot(pred_path, has_score=True)
    all_frame_ids = set(gt_by_frame) | set(pred_by_frame)
    seq_length = load_seq_length(sequence_dir, all_frame_ids)

    total_gt = sum(len(v) for v in gt_by_frame.values())
    total_pred = sum(len(v) for v in pred_by_frame.values())

    tp = fp = fn = id_switches = fragments = 0
    iou_sum = 0.0

    gt_occurrences: Counter[int] = Counter()
    pred_occurrences: Counter[int] = Counter()
    switches_by_gt: Counter[int] = Counter()
    fragments_by_gt: Counter[int] = Counter()
    pair_counts: Counter[tuple[int, int]] = Counter()
    gt_matched_counts: Counter[int] = Counter()
    pred_matched_counts: Counter[int] = Counter()

    last_pred_for_gt: dict[int, int] = {}
    gt_has_been_matched: dict[int, bool] = defaultdict(bool)
    gt_was_matched_last_present_frame: dict[int, bool] = defaultdict(bool)

    id_switch_events: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []

    for frame_id in range(1, seq_length + 1):
        gt = gt_by_frame.get(frame_id, [])
        pred = pred_by_frame.get(frame_id, [])
        matches, unmatched_gt, unmatched_pred = match_frame(gt, pred, iou_threshold)

        for item in gt:
            gt_occurrences[item["id"]] += 1
        for item in pred:
            pred_occurrences[item["id"]] += 1

        frame_id_switches = 0
        frame_iou_sum = 0.0
        matched_gt_ids: set[int] = set()

        tp += len(matches)
        fn += len(unmatched_gt)
        fp += len(unmatched_pred)

        for gt_idx, pred_idx, iou in matches:
            gt_id = int(gt[gt_idx]["id"])
            pred_id = int(pred[pred_idx]["id"])
            matched_gt_ids.add(gt_id)

            previous_pred_id = last_pred_for_gt.get(gt_id)
            if previous_pred_id is not None and previous_pred_id != pred_id:
                id_switches += 1
                frame_id_switches += 1
                switches_by_gt[gt_id] += 1
                id_switch_events.append(
                    {
                        "split": split,
                        "sequence": sequence,
                        "frame": frame_id,
                        "gt_id": gt_id,
                        "previous_pred_id": previous_pred_id,
                        "new_pred_id": pred_id,
                        "iou": rounded(iou),
                    }
                )

            if gt_has_been_matched[gt_id] and not gt_was_matched_last_present_frame[gt_id]:
                fragments += 1
                fragments_by_gt[gt_id] += 1

            gt_has_been_matched[gt_id] = True
            gt_was_matched_last_present_frame[gt_id] = True
            last_pred_for_gt[gt_id] = pred_id

            pair_counts[(gt_id, pred_id)] += 1
            gt_matched_counts[gt_id] += 1
            pred_matched_counts[pred_id] += 1
            iou_sum += iou
            frame_iou_sum += iou

        for gt_idx in unmatched_gt:
            gt_id = int(gt[gt_idx]["id"])
            gt_was_matched_last_present_frame[gt_id] = False

        frame_gt = len(gt)
        frame_pred = len(pred)
        frame_tp = len(matches)
        frame_fp = len(unmatched_pred)
        frame_fn = len(unmatched_gt)
        frame_error = frame_fp + frame_fn + frame_id_switches
        frame_rows.append(
            {
                "split": split,
                "sequence": sequence,
                "frame": frame_id,
                "gt_count": frame_gt,
                "pred_count": frame_pred,
                "true_positives": frame_tp,
                "false_positives": frame_fp,
                "false_negatives": frame_fn,
                "id_switches": frame_id_switches,
                "mean_iou": rounded(safe_div(frame_iou_sum, frame_tp)),
                "error_count": frame_error,
                "error_vs_gt_pct": rounded(pct(safe_div(frame_error, frame_gt))),
            }
        )

    idtp = compute_idtp(pair_counts)
    idfp = total_pred - idtp
    idfn = total_gt - idtp

    mota = 1.0 - safe_div(fp + fn + id_switches, total_gt)
    motp = safe_div(iou_sum, tp)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    idf1 = safe_div(2 * idtp, (2 * idtp) + idfp + idfn)
    deta = safe_div(tp, tp + fp + fn)
    assa = association_accuracy(pair_counts, gt_matched_counts, pred_matched_counts)
    hota = math.sqrt(deta * assa) if deta > 0 and assa > 0 else 0.0

    per_gt_rows = []
    for gt_id in sorted(gt_occurrences):
        pred_counts = Counter(
            {pred_id: count for (pair_gt_id, pred_id), count in pair_counts.items() if pair_gt_id == gt_id}
        )
        matched = sum(pred_counts.values())
        missed = gt_occurrences[gt_id] - matched
        dominant_pred_id = ""
        dominant_pred_matches = 0
        purity = 0.0
        if pred_counts:
            dominant_pred_id, dominant_pred_matches = pred_counts.most_common(1)[0]
            purity = safe_div(dominant_pred_matches, matched)
        pred_ids_seen = ";".join(
            f"{pred_id}:{count}" for pred_id, count in pred_counts.most_common()
        )
        identity_mismatch = 1.0 - purity if matched else 0.0

        per_gt_rows.append(
            {
                "split": split,
                "sequence": sequence,
                "gt_id": gt_id,
                "gt_detections": gt_occurrences[gt_id],
                "matched_detections": matched,
                "missed_detections": missed,
                "coverage_pct": rounded(pct(safe_div(matched, gt_occurrences[gt_id]))),
                "dominant_pred_id": dominant_pred_id,
                "dominant_pred_matches": dominant_pred_matches,
                "purity_pct": rounded(pct(purity)),
                "identity_mismatch_pct": rounded(pct(identity_mismatch)),
                "id_switches": switches_by_gt[gt_id],
                "fragments": fragments_by_gt[gt_id],
                "pred_ids_seen": pred_ids_seen,
            }
        )

    metrics = {
        "split": split,
        "sequence": sequence,
        "frames": seq_length,
        "gt_ids": len(gt_occurrences),
        "pred_ids": len(pred_occurrences),
        "gt_detections": total_gt,
        "pred_detections": total_pred,
        "pred_minus_gt": total_pred - total_gt,
        "pred_vs_gt_pct": rounded(pct(safe_div(total_pred - total_gt, total_gt))),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "id_switches": id_switches,
        "fragments": fragments,
        "MOTA_pct": rounded(pct(mota)),
        "MOTP_pct": rounded(pct(motp)),
        "IDF1_pct": rounded(pct(idf1)),
        "precision_pct": rounded(pct(precision)),
        "recall_pct": rounded(pct(recall)),
        "DetA50_pct": rounded(pct(deta)),
        "AssA50_pct": rounded(pct(assa)),
        "HOTA50_pct": rounded(pct(hota)),
        "miss_rate_pct": rounded(pct(safe_div(fn, total_gt))),
        "fp_vs_gt_pct": rounded(pct(safe_div(fp, total_gt))),
        "fp_rate_pct": rounded(pct(safe_div(fp, total_pred))),
        "id_switch_vs_gt_pct": rounded(pct(safe_div(id_switches, total_gt))),
        "fragments_vs_gt_pct": rounded(pct(safe_div(fragments, total_gt))),
        "total_error_vs_gt_pct": rounded(pct(safe_div(fp + fn + id_switches, total_gt))),
        "iou_threshold": iou_threshold,
    }

    return {
        "metrics": metrics,
        "id_switch_events": id_switch_events,
        "per_gt_rows": per_gt_rows,
        "frame_rows": frame_rows,
        "pair_counts": pair_counts,
        "gt_matched_counts": gt_matched_counts,
        "pred_matched_counts": pred_matched_counts,
        "iou_sum": iou_sum,
    }


def discover_sequences(data_root: Path, splits_file: Path, splits: list[str]) -> list[dict[str, Any]]:
    seq_names = [
        line.strip()
        for line in splits_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sequences = []
    for split in splits:
        for sequence in seq_names:
            sequence_dir = data_root / split / sequence
            if (sequence_dir / "gt" / "gt.txt").exists():
                sequences.append({"split": split, "sequence": sequence, "sequence_dir": sequence_dir})
    return sequences


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_overall_summary(
    sequence_results: list[dict[str, Any]],
    iou_threshold: float,
) -> dict[str, Any]:
    total_gt = sum(r["metrics"]["gt_detections"] for r in sequence_results)
    total_pred = sum(r["metrics"]["pred_detections"] for r in sequence_results)
    tp = sum(r["metrics"]["true_positives"] for r in sequence_results)
    fp = sum(r["metrics"]["false_positives"] for r in sequence_results)
    fn = sum(r["metrics"]["false_negatives"] for r in sequence_results)
    id_switches = sum(r["metrics"]["id_switches"] for r in sequence_results)
    fragments = sum(r["metrics"]["fragments"] for r in sequence_results)
    iou_sum = sum(r["iou_sum"] for r in sequence_results)

    global_pairs: Counter[tuple[tuple[str, int], tuple[str, int]]] = Counter()
    global_gt_matched: Counter[tuple[str, int]] = Counter()
    global_pred_matched: Counter[tuple[str, int]] = Counter()
    for result in sequence_results:
        sequence = result["metrics"]["sequence"]
        for (gt_id, pred_id), count in result["pair_counts"].items():
            global_pairs[((sequence, gt_id), (sequence, pred_id))] += count
            global_gt_matched[(sequence, gt_id)] += count
            global_pred_matched[(sequence, pred_id)] += count

    idtp = compute_idtp(global_pairs)
    idfp = total_pred - idtp
    idfn = total_gt - idtp

    mota = 1.0 - safe_div(fp + fn + id_switches, total_gt)
    motp = safe_div(iou_sum, tp)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    idf1 = safe_div(2 * idtp, (2 * idtp) + idfp + idfn)
    deta = safe_div(tp, tp + fp + fn)
    assa = association_accuracy(global_pairs, global_gt_matched, global_pred_matched)
    hota = math.sqrt(deta * assa) if deta > 0 and assa > 0 else 0.0

    return {
        "sequence_count": len(sequence_results),
        "gt_detections": total_gt,
        "pred_detections": total_pred,
        "pred_minus_gt": total_pred - total_gt,
        "pred_vs_gt_pct": rounded(pct(safe_div(total_pred - total_gt, total_gt))),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "id_switches": id_switches,
        "fragments": fragments,
        "MOTA_pct": rounded(pct(mota)),
        "MOTP_pct": rounded(pct(motp)),
        "IDF1_pct": rounded(pct(idf1)),
        "precision_pct": rounded(pct(precision)),
        "recall_pct": rounded(pct(recall)),
        "DetA50_pct": rounded(pct(deta)),
        "AssA50_pct": rounded(pct(assa)),
        "HOTA50_pct": rounded(pct(hota)),
        "miss_rate_pct": rounded(pct(safe_div(fn, total_gt))),
        "fp_vs_gt_pct": rounded(pct(safe_div(fp, total_gt))),
        "fp_rate_pct": rounded(pct(safe_div(fp, total_pred))),
        "id_switch_vs_gt_pct": rounded(pct(safe_div(id_switches, total_gt))),
        "fragments_vs_gt_pct": rounded(pct(safe_div(fragments, total_gt))),
        "total_error_vs_gt_pct": rounded(pct(safe_div(fp + fn + id_switches, total_gt))),
        "iou_threshold": iou_threshold,
        "note": "HOTA50/AssA50/DetA50 are local one-threshold estimates, not official TrackEval leaderboard HOTA.",
    }


def print_overall_table(summary: dict[str, Any]) -> None:
    table = Table(title="Overall Benchmark vs Ground Truth")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    rows = [
        ("Sequences", summary["sequence_count"]),
        ("GT detections", summary["gt_detections"]),
        ("Pred detections", summary["pred_detections"]),
        ("Pred - GT", summary["pred_minus_gt"]),
        ("Pred vs GT %", summary["pred_vs_gt_pct"]),
        ("TP", summary["true_positives"]),
        ("FP", summary["false_positives"]),
        ("FN", summary["false_negatives"]),
        ("MOTA %", summary["MOTA_pct"]),
        ("MOTP %", summary["MOTP_pct"]),
        ("IDF1 %", summary["IDF1_pct"]),
        ("Precision %", summary["precision_pct"]),
        ("Recall %", summary["recall_pct"]),
        ("HOTA@0.50 local %", summary["HOTA50_pct"]),
        ("DetA@0.50 local %", summary["DetA50_pct"]),
        ("AssA@0.50 local %", summary["AssA50_pct"]),
        ("ID switches", summary["id_switches"]),
        ("ID switches / GT %", summary["id_switch_vs_gt_pct"]),
        ("Fragments", summary["fragments"]),
        ("Fragments / GT %", summary["fragments_vs_gt_pct"]),
        ("Miss rate %", summary["miss_rate_pct"]),
        ("FP vs GT %", summary["fp_vs_gt_pct"]),
        ("Total error vs GT %", summary["total_error_vs_gt_pct"]),
    ]
    for name, value in rows:
        table.add_row(name, str(value))
    console.print(table)


def print_sequence_table(rows: list[dict[str, Any]]) -> None:
    table = Table(title="Per-Sequence Metrics")
    for column in [
        "split",
        "sequence",
        "GT",
        "Pred",
        "MOTA%",
        "IDF1%",
        "HOTA50%",
        "Prec%",
        "Rec%",
        "IDs",
        "IDs/GT%",
        "Frag",
        "Miss%",
        "FP/GT%",
    ]:
        table.add_column(column, justify="right" if column not in {"split", "sequence"} else "left")

    for row in rows:
        table.add_row(
            row["split"],
            row["sequence"],
            str(row["gt_detections"]),
            str(row["pred_detections"]),
            str(row["MOTA_pct"]),
            str(row["IDF1_pct"]),
            str(row["HOTA50_pct"]),
            str(row["precision_pct"]),
            str(row["recall_pct"]),
            str(row["id_switches"]),
            str(row["id_switch_vs_gt_pct"]),
            str(row["fragments"]),
            str(row["miss_rate_pct"]),
            str(row["fp_vs_gt_pct"]),
        )
    console.print(table)


def print_id_issue_table(rows: list[dict[str, Any]], limit: int) -> None:
    table = Table(title=f"Worst GT IDs by Identity Instability (top {limit})")
    for column in [
        "split",
        "sequence",
        "gt_id",
        "gt_det",
        "matched",
        "miss%",
        "dom_pred",
        "purity%",
        "id_mismatch%",
        "IDs",
        "Frag",
    ]:
        table.add_column(column, justify="right" if column not in {"split", "sequence"} else "left")

    sorted_rows = sorted(
        rows,
        key=lambda r: (
            r["id_switches"],
            r["identity_mismatch_pct"],
            r["missed_detections"],
        ),
        reverse=True,
    )
    for row in sorted_rows[:limit]:
        miss_pct = rounded(100.0 - row["coverage_pct"])
        table.add_row(
            row["split"],
            row["sequence"],
            str(row["gt_id"]),
            str(row["gt_detections"]),
            str(row["matched_detections"]),
            str(miss_pct),
            str(row["dominant_pred_id"]),
            str(row["purity_pct"]),
            str(row["identity_mismatch_pct"]),
            str(row["id_switches"]),
            str(row["fragments"]),
        )
    console.print(table)


def print_frame_issue_table(rows: list[dict[str, Any]], limit: int) -> None:
    table = Table(title=f"Worst Frames by Error vs GT (top {limit})")
    for column in [
        "split",
        "sequence",
        "frame",
        "GT",
        "Pred",
        "TP",
        "FP",
        "FN",
        "IDs",
        "err",
        "err/GT%",
        "mIoU",
    ]:
        table.add_column(column, justify="right" if column not in {"split", "sequence"} else "left")

    sorted_rows = sorted(
        rows,
        key=lambda r: (r["error_count"], r["error_vs_gt_pct"], r["false_negatives"]),
        reverse=True,
    )
    for row in sorted_rows[:limit]:
        table.add_row(
            row["split"],
            row["sequence"],
            str(row["frame"]),
            str(row["gt_count"]),
            str(row["pred_count"]),
            str(row["true_positives"]),
            str(row["false_positives"]),
            str(row["false_negatives"]),
            str(row["id_switches"]),
            str(row["error_count"]),
            str(row["error_vs_gt_pct"]),
            str(row["mean_iou"]),
        )
    console.print(table)


def print_switch_event_table(rows: list[dict[str, Any]], limit: int) -> None:
    table = Table(title=f"ID Switch Events (top {limit})")
    for column in ["split", "sequence", "frame", "gt_id", "prev_pred", "new_pred", "IoU"]:
        table.add_column(column, justify="right" if column not in {"split", "sequence"} else "left")

    for row in rows[:limit]:
        table.add_row(
            row["split"],
            row["sequence"],
            str(row["frame"]),
            str(row["gt_id"]),
            str(row["previous_pred_id"]),
            str(row["new_pred_id"]),
            str(row["iou"]),
        )
    console.print(table)


def write_markdown_report(
    path: Path,
    summary: dict[str, Any],
    sequence_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SportsMOT Local Benchmark Report",
        "",
        "This report compares MOT predictions with `gt/gt.txt` ground truth.",
        "",
        "> HOTA50, AssA50, and DetA50 are local @IoU=0.50 estimates, not official TrackEval leaderboard HOTA.",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## Per Sequence",
            "",
            "| Split | Sequence | GT | Pred | MOTA% | IDF1% | HOTA50% | Precision% | Recall% | IDs | IDs/GT% | Frag | Miss% | FP/GT% |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sequence_rows:
        lines.append(
            "| {split} | {sequence} | {gt_detections} | {pred_detections} | "
            "{MOTA_pct} | {IDF1_pct} | {HOTA50_pct} | {precision_pct} | "
            "{recall_pct} | {id_switches} | {id_switch_vs_gt_pct} | "
            "{fragments} | {miss_rate_pct} | {fp_vs_gt_pct} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    splits_file = Path(args.splits_file)
    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = discover_sequences(data_root, splits_file, args.splits)
    sequence_results = []
    missing_predictions = []

    for item in sequences:
        split = item["split"]
        sequence = item["sequence"]
        sequence_dir = item["sequence_dir"]
        pred_path = pred_dir / f"{sequence}.txt"
        if not pred_path.exists():
            missing_predictions.append({"split": split, "sequence": sequence, "pred": str(pred_path)})
            continue
        sequence_results.append(
            analyze_sequence(split, sequence, sequence_dir, pred_path, args.iou_threshold)
        )

    if not sequence_results:
        raise FileNotFoundError(f"No prediction files found in {pred_dir}")

    sequence_rows = [r["metrics"] for r in sequence_results]
    id_switch_rows = [row for r in sequence_results for row in r["id_switch_events"]]
    per_gt_rows = [row for r in sequence_results for row in r["per_gt_rows"]]
    frame_rows = [row for r in sequence_results for row in r["frame_rows"]]
    overall_summary = build_overall_summary(sequence_results, args.iou_threshold)

    write_csv(output_dir / "benchmark_per_sequence.csv", sequence_rows)
    write_csv(output_dir / "benchmark_id_switch_events.csv", id_switch_rows)
    write_csv(output_dir / "benchmark_per_gt_identity.csv", per_gt_rows)
    write_csv(output_dir / "benchmark_frame_errors.csv", frame_rows)
    write_csv(output_dir / "benchmark_missing_predictions.csv", missing_predictions)
    (output_dir / "benchmark_overall_summary.json").write_text(
        json.dumps(overall_summary, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(output_dir / "benchmark_report.md", overall_summary, sequence_rows)

    print_overall_table(overall_summary)
    print_sequence_table(sequence_rows)
    print_id_issue_table(per_gt_rows, args.top_id_issues)
    print_frame_issue_table(frame_rows, args.top_frame_issues)
    print_switch_event_table(id_switch_rows, args.top_switch_events)

    console.print(f"[green]Saved:[/green] {output_dir / 'benchmark_per_sequence.csv'}")
    console.print(f"[green]Saved:[/green] {output_dir / 'benchmark_per_gt_identity.csv'}")
    console.print(f"[green]Saved:[/green] {output_dir / 'benchmark_id_switch_events.csv'}")
    console.print(f"[green]Saved:[/green] {output_dir / 'benchmark_frame_errors.csv'}")
    console.print(f"[green]Saved:[/green] {output_dir / 'benchmark_overall_summary.json'}")
    console.print(f"[green]Saved:[/green] {output_dir / 'benchmark_report.md'}")

    if missing_predictions:
        console.print(
            f"[yellow]Warning:[/yellow] {len(missing_predictions)} sequences have no prediction txt. "
            f"See {output_dir / 'benchmark_missing_predictions.csv'}"
        )


if __name__ == "__main__":
    main()
