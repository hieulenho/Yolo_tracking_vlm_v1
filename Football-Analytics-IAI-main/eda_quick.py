from pathlib import Path
from collections import Counter
import cv2
import matplotlib.pyplot as plt
import numpy as np

# ── Config ──────────────────────────────────────────
DATASET = Path("data/raw/players")
SAVE_DIR = Path("reports/figures")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
CLASS_NAMES = {0: "player", 1: "goalkeeper", 2: "ball"}

# ── 1. Đếm ảnh ──────────────────────────────────────
train_imgs = list((DATASET / "images/train").glob("*.jpg"))
val_imgs   = list((DATASET / "images/val").glob("*.jpg"))
print(f"Train: {len(train_imgs)} images")
print(f"Val:   {len(val_imgs)} images")

# ── 2. Class distribution ────────────────────────────
counter = Counter()
for txt in (DATASET / "labels/train").glob("*.txt"):
    for line in txt.read_text().strip().splitlines():
        if line:
            counter[int(line.split()[0])] += 1

labels = [CLASS_NAMES[k] for k in sorted(counter)]
values = [counter[k] for k in sorted(counter)]

plt.figure(figsize=(7, 4))
bars = plt.bar(labels, values, color=["#4C72B0","#DD8452","#55A868"])
plt.title("Class distribution (train set)")
plt.ylabel("Bbox count")
for bar, v in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, v + 20, str(v), ha="center")
plt.tight_layout()
plt.savefig(SAVE_DIR / "players_class_distribution.png", dpi=150)
plt.close()
print("✓ Saved: players_class_distribution.png")

# ── 3. Bbox size distribution ────────────────────────
widths, heights = [], []
for txt in (DATASET / "labels/train").glob("*.txt"):
    for line in txt.read_text().strip().splitlines():
        if line:
            parts = line.split()
            widths.append(float(parts[3]))
            heights.append(float(parts[4]))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(widths,  bins=40, color="#4C72B0", edgecolor="white")
axes[0].set_title("Bbox width (normalized)")
axes[0].set_xlabel("Width")
axes[1].hist(heights, bins=40, color="#DD8452", edgecolor="white")
axes[1].set_title("Bbox height (normalized)")
axes[1].set_xlabel("Height")
plt.suptitle(f"Bbox size distribution  (n={len(widths)} bboxes)")
plt.tight_layout()
plt.savefig(SAVE_DIR / "players_bbox_distribution.png", dpi=150)
plt.close()
print("✓ Saved: players_bbox_distribution.png")
print(f"  Mean width:  {np.mean(widths):.4f}")
print(f"  Mean height: {np.mean(heights):.4f}")

# ── 4. Sample images với bbox ────────────────────────
COLORS = [(255,80,80),(255,165,0),(80,200,80)]
sample_imgs = train_imgs[:6]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, img_path in zip(axes.flat, sample_imgs):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    label_path = DATASET / "labels/train" / (img_path.stem + ".txt")
    if label_path.exists():
        for line in label_path.read_text().strip().splitlines():
            if not line: continue
            cls_id, cx, cy, bw, bh = map(float, line.split())
            cls_id = int(cls_id)
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            cv2.rectangle(img, (x1,y1), (x2,y2), COLORS[cls_id % 3], 2)
            cv2.putText(img, CLASS_NAMES.get(cls_id, str(cls_id)),
                        (x1, max(15, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        COLORS[cls_id % 3], 1)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(img_path.name, fontsize=8)
    ax.axis("off")
plt.suptitle("Sample training images with labels")
plt.tight_layout()
plt.savefig(SAVE_DIR / "players_samples.png", dpi=120)
plt.close()
print("✓ Saved: players_samples.png")

print("\n✓ EDA done! Check reports/figures/")