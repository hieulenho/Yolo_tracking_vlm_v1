# Football Player Detection and Multi-Object Tracking

This project focuses on football player detection and multi-object tracking using a fine-tuned YOLOv8m detector and DeepSORT.

The pipeline is intentionally scoped to tracking quality: detect football objects in each frame, associate them through time, assign stable IDs, and render each ID with a distinct high-contrast color.

## Core Features

- YOLOv8m detection for `player`, `goalkeeper`, `ball`, and `referee` classes.
- DeepSORT tracking with Kalman motion prediction and visual appearance association.
- Stable per-object `track_id` output through `TrackedDetection`.
- Thick bounding boxes and bold labels in the rendered video.
- Deterministic per-ID colors so different tracked objects are easy to follow.
- Lightweight CLI for video inference.

## Results

### Player Detector

| Model | Task | mAP50 | mAP50-95 | Precision | Recall | Epochs |
|---|---|---:|---:|---:|---:|---:|
| YOLOv8m | Players detection, imgsz=640 | 0.809 | 0.528 | 0.920 | 0.800 | 100 |
| YOLOv8m | Players detection, imgsz=1280 | 0.922 | 0.638 | 0.967 | 0.898 | 100 |

### Per-Class Analysis, YOLOv8m imgsz=640

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| player | 0.969 | 0.978 | 0.988 | 0.710 |
| goalkeeper | 0.908 | 0.899 | 0.934 | 0.669 |
| ball | 0.832 | 0.525 | 0.513 | 0.251 |

Ball detection improves from 0.513 to 0.805 mAP50 when using imgsz=1280.

## Architecture

```text
Video frame
    |
    v
YOLOv8m PlayerDetector
    |
    v
Detections: bbox, class, confidence
    |
    v
DeepSORT Tracker
    |
    v
TrackedDetection: bbox, class, confidence, track_id
    |
    v
Video renderer: thick boxes, bold ID labels, per-ID colors
```

DeepSORT combines a Kalman filter, Hungarian assignment, IoU/motion matching, and appearance embeddings. This helps preserve IDs when players move quickly, overlap, or briefly become partially occluded.

## Project Structure

```text
football-analytics/
├── configs/
│   ├── data/players.yaml
│   ├── train/players_yolov8m.yaml
│   └── inference.yaml
├── models/
│   └── players_best.pt
├── scripts/
│   ├── train_players_yolo.py
│   ├── evaluate.py
│   ├── benchmark.py
│   ├── analyze_training.py
│   └── run_inference.py
├── src/
│   ├── core/
│   ├── detection/
│   ├── tracking/
│   ├── pipeline/
│   └── utils/
└── tests/
```

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Windows, Python 3.12 with a compatible PyTorch build is recommended.

## Training

```bash
python scripts/train_players_yolo.py \
    --data configs/data/players.yaml \
    --model yolov8m.pt \
    --epochs 100 \
    --batch 8
```

## Inference

```bash
python scripts/run_inference.py \
    --video input.mp4 \
    --output output.mp4 \
    --config configs/inference.yaml
```

The output video contains YOLO detections tracked by DeepSORT. Every tracked object receives a label such as `ID 12 | player` and a unique color.

## Configuration

Tracking parameters live in `configs/inference.yaml`:

```yaml
tracking:
  tracker: deepsort
  max_age: 30
  n_init: 3
  max_cosine_distance: 0.2
  nn_budget: 100
  embedder_gpu: false
  only_confirmed: false
```

- `max_age`: how long a track can survive without a matched detection.
- `n_init`: number of consecutive matches before a track is confirmed.
- `max_cosine_distance`: appearance distance threshold for matching.
- `nn_budget`: maximum gallery size for appearance embeddings.
- `only_confirmed`: set `true` to hide tentative early tracks.

## Evaluate

```bash
python scripts/evaluate.py \
    --weights models/players_best.pt \
    --data configs/data/players.yaml
```

## SportsMOT Benchmark

Use SportsMOT football sequences when you need a benchmark video with ground
truth tracking labels. See [docs/sportsmot_benchmark.md](docs/sportsmot_benchmark.md).

## Tests

```bash
pytest tests/
```

## Scope

This repository is scoped to player/object detection and multi-object tracking. Older side workflows were removed so the codebase stays centered on YOLOv8m + DeepSORT tracking.

## License

MIT
