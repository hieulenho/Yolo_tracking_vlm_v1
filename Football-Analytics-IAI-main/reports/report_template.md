# Football Player Detection and Multi-Object Tracking using YOLOv8m and DeepSORT

## 1. Introduction

This project builds a focused football video analysis pipeline for detecting players and tracking them across frames. The system uses YOLOv8m as the object detector and DeepSORT as the multi-object tracker.

The project concentrates on stable object identities, readable track labels, and clear visualization of player movement in the original broadcast frame.

## 2. Objectives

- Detect football objects from video frames using YOLOv8m.
- Track detected objects over time using DeepSORT.
- Assign stable track IDs to players, goalkeepers, ball, and referee detections.
- Render each ID with a distinct color and bold high-contrast label.
- Evaluate detection performance and provide a simple reproducible inference workflow.

## 3. Methodology

### 3.1 YOLOv8m Detection

YOLOv8m is fine-tuned for football-object detection. The detector outputs bounding boxes, confidence scores, and class labels for each frame. These detections are passed directly into the tracker.

The target classes are:

- player
- goalkeeper
- ball
- referee

### 3.2 DeepSORT Tracking

DeepSORT follows the tracking-by-detection paradigm. It receives YOLO detections frame by frame and maintains object identities using:

- Kalman filter motion prediction
- Hungarian assignment
- bounding-box overlap
- appearance embeddings

This is especially useful in football videos because players frequently cross paths, accelerate quickly, and become partially occluded.

### 3.3 Visualization

Each confirmed DeepSORT track is rendered with:

- a deterministic color based on `track_id`
- a thick bounding box
- a bold label such as `ID 7 | player`
- a dark outline and filled label background for readability

The goal is to make tracking IDs easy to inspect in the output video.

## 4. Results Summary

| Model | Task | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---:|---:|---:|---:|
| YOLOv8m | Players detection, imgsz=640 | 0.809 | 0.528 | 0.920 | 0.800 |
| YOLOv8m | Players detection, imgsz=1280 | 0.922 | 0.638 | 0.967 | 0.898 |

The strongest gains came from increasing image size to 1280, especially for the ball class.

## 5. Inference Pipeline

```text
Video frame
  -> YOLOv8m detector
  -> frame detections
  -> DeepSORT tracker
  -> tracked detections with IDs
  -> annotated output video
```

Run inference with:

```bash
python scripts/run_inference.py --video input.mp4 --output output.mp4
```

## 6. Conclusion

The repository is now centered on football player detection and multi-object tracking. Removing unrelated projection and classification modules makes the codebase easier to understand, test, and extend for tracking-focused experiments.
