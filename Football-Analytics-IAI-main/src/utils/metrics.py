"""
Custom metrics for evaluation và reporting.

YOLO đã có mAP nội bộ, nhưng ta cần thêm:
- Per-class metrics export ra DataFrame (cho báo cáo)
- FPS benchmark
- Inference latency distribution
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class BenchmarkResult:
    """Kết quả benchmark inference."""

    n_frames: int
    total_time_s: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    fps: float

    def to_dict(self) -> dict:
        return {
            "n_frames": self.n_frames,
            "total_time_s": round(self.total_time_s, 2),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "fps": round(self.fps, 2),
        }


def benchmark_inference(
    inference_fn: Callable[[np.ndarray], object],
    frames: list[np.ndarray],
    warmup: int = 5,
) -> BenchmarkResult:
    """
    Benchmark 1 hàm inference trên list frame.

    Args:
        inference_fn: function nhận 1 frame, return detection (hoặc gì đó).
        frames: list các frame test.
        warmup: số frame warmup (loại khỏi tính toán).
    """
    # Warmup
    for f in frames[:warmup]:
        inference_fn(f)

    latencies = []
    for f in frames[warmup:]:
        t0 = time.perf_counter()
        inference_fn(f)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    lat = np.array(latencies)
    total_s = lat.sum() / 1000
    return BenchmarkResult(
        n_frames=len(lat),
        total_time_s=total_s,
        mean_latency_ms=float(lat.mean()),
        p50_latency_ms=float(np.percentile(lat, 50)),
        p95_latency_ms=float(np.percentile(lat, 95)),
        p99_latency_ms=float(np.percentile(lat, 99)),
        fps=len(lat) / total_s if total_s > 0 else 0.0,
    )


def yolo_metrics_to_dict(metrics) -> dict:
    """Chuyển YOLO val metrics object → flat dict cho export."""
    return {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class": {
            name: {
                "mAP50": float(metrics.box.ap50[i]),
                "mAP50_95": float(metrics.box.ap[i]),
            }
            for i, name in enumerate(metrics.names.values())
        }
        if hasattr(metrics.box, "ap50")
        else {},
    }
