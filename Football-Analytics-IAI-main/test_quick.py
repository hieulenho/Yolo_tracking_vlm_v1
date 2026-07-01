"""Quick single-frame smoke test for the YOLOv8m player detector."""
from __future__ import annotations

import cv2
from ultralytics import YOLO

VIDEO_PATH = "test_video_city_chel_21_22.mp4"
WEIGHTS_PATH = "models/players_best.pt"


def main() -> None:
    print("Loading YOLOv8m player detector...")
    model = YOLO(WEIGHTS_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise SystemExit(f"Cannot open video: {VIDEO_PATH}")

    print("\n=== Player Detection ===")
    results = model.predict(frame, conf=0.35, verbose=False)[0]
    for box in results.boxes:
        class_name = model.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        print(f"  {class_name}: {confidence:.2f}")
    print(f"Total: {len(results.boxes)} objects")

    cv2.imwrite("test_output.jpg", results.plot())
    print("\nSaved: test_output.jpg")


if __name__ == "__main__":
    main()
