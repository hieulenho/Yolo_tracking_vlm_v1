# Football Player Tracking

Dự án này dùng YOLOv8m để phát hiện cầu thủ và DeepSORT để gán ID theo thời gian. Đầu ra chính là video đã tracking và các file benchmark so với ground truth theo định dạng SportsMOT/MOTChallenge.

Pipeline hiện tại tập trung vào:

- phát hiện `player`, `goalkeeper`, `ball`, `referee`;
- tracking nhiều đối tượng bằng DeepSORT;
- xuất file prediction dạng MOT;
- so sánh prediction với `gt/gt.txt`;
- tổng hợp các chỉ số như HOTA, AssA, DetA, MOTA, IDF1, IDs và Frag.

## Cài đặt

```powershell
cd F:\Football-Analytics-IAI-main\Football-Analytics-IAI-main

python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Model mặc định nằm tại:

```text
models/players_best.pt
```

Cấu hình inference nằm tại:

```text
configs/inference.yaml
```

Trong cấu hình hiện tại, model đang chạy bằng CPU:

```yaml
detection:
  device: cpu
```

Nếu đã cài PyTorch CUDA, có thể đổi `device: cpu` thành `device: 0`.

## Tracking video riêng

Ví dụ với hai video gốc trong `F:\videos`:

```powershell
.\.venv\Scripts\python.exe scripts\run_inference.py `
  --video F:\videos\1.mp4 `
  --output F:\videos\1_tracked.mp4 `
  --config configs\inference.yaml

.\.venv\Scripts\python.exe scripts\run_inference.py `
  --video F:\videos\2.mp4 `
  --output F:\videos\2_tracked.mp4 `
  --config configs\inference.yaml
```

Hai file kết quả:

```text
F:\videos\1_tracked.mp4
F:\videos\2_tracked.mp4
```

Lưu ý: video riêng chỉ có thể xem kết quả tracking. Muốn tính benchmark thì cần ground truth.

## Benchmark SportsMOT

Dataset đang dùng có dạng:

```text
data/raw/
  splits_txt/football.txt
  train/<sequence>/
    img1/
    gt/gt.txt
    seqinfo.ini
  val/<sequence>/
    img1/
    gt/gt.txt
    seqinfo.ini
```

Chạy tracking và evaluate cho toàn bộ sequence có trong `football.txt`:

```powershell
$seqs = Get-Content data\raw\splits_txt\football.txt

foreach ($split in @("train", "val")) {
  foreach ($seq in $seqs) {
    $seqDir = "data\raw\$split\$seq"
    $pred = "runs\sportsmot\preds\$seq.txt"
    $trackSummary = "runs\sportsmot\summaries\$seq.json"
    $metrics = "runs\sportsmot\summaries\${seq}_metrics.json"

    if (-not (Test-Path "$seqDir\gt\gt.txt")) {
      continue
    }

    if (Test-Path $metrics) {
      Write-Host "SKIP metrics exists: $split/$seq"
      continue
    }

    if ((Test-Path $pred) -and (Test-Path $trackSummary)) {
      Write-Host "EVALUATE existing pred: $split/$seq"
    } else {
      Write-Host "RUN tracking: $split/$seq"

      .\.venv\Scripts\python.exe scripts\run_sportsmot_sequence.py `
        --sequence-dir $seqDir `
        --output-dir runs\sportsmot

      if ($LASTEXITCODE -ne 0) {
        throw "Tracking failed: $split/$seq"
      }
    }

    .\.venv\Scripts\python.exe scripts\evaluate_sportsmot_sequence.py `
      --sequence-dir $seqDir `
      --pred $pred `
      --output-json $metrics

    if ($LASTEXITCODE -ne 0) {
      throw "Evaluation failed: $split/$seq"
    }
  }
}
```

Kiểm tra đã đủ benchmark chưa:

```powershell
(Get-ChildItem runs\sportsmot\summaries -Filter *_metrics.json).Count
```

Với bộ dữ liệu hiện tại, số lượng đầy đủ là `30`.

## Tổng hợp kết quả

Sau khi đã có prediction và metrics, chạy:

```powershell
.\.venv\Scripts\python.exe scripts\report_sportsmot_benchmark.py `
  --data-root data\raw `
  --splits-file data\raw\splits_txt\football.txt `
  --pred-dir runs\sportsmot\preds `
  --output-dir runs\sportsmot\summaries
```

Script này in bảng tổng kết trên terminal và ghi các file:

```text
runs/sportsmot/summaries/benchmark_report.md
runs/sportsmot/summaries/benchmark_overall_summary.json
runs/sportsmot/summaries/benchmark_per_sequence.csv
runs/sportsmot/summaries/benchmark_per_gt_identity.csv
runs/sportsmot/summaries/benchmark_id_switch_events.csv
runs/sportsmot/summaries/benchmark_frame_errors.csv
```

Các file chi tiết:

- `benchmark_per_sequence.csv`: kết quả từng sequence.
- `benchmark_per_gt_identity.csv`: từng ID ground truth được match với những predicted ID nào.
- `benchmark_id_switch_events.csv`: frame xảy ra ID switch, ID cũ và ID mới.
- `benchmark_frame_errors.csv`: frame có nhiều FP, FN hoặc ID switch.

## Kết quả hiện tại

Kết quả local trên 30 sequence football, IoU threshold `0.50`:

| Metric | Value |
|---|---:|
| GT detections | 262343 |
| Pred detections | 294258 |
| True positives | 242800 |
| False positives | 51458 |
| False negatives | 19543 |
| HOTA@0.50 | 63.8945 |
| AssA@0.50 | 52.7633 |
| DetA@0.50 | 77.3739 |
| MOTA | 71.1008 |
| MOTP | 80.5728 |
| IDF1 | 55.8504 |
| Precision | 82.5126 |
| Recall | 92.5506 |
| IDs | 4814 |
| Frag | 5051 |
| Miss rate | 7.4494 |
| FP vs GT | 19.6148 |
| IDs vs GT | 1.8350 |
| Frag vs GT | 1.9253 |

`HOTA@0.50`, `AssA@0.50` và `DetA@0.50` là kết quả local ở một ngưỡng IoU. Nếu cần điểm giống leaderboard CodaLab/SportsMOT, cần chạy official TrackEval hoặc submit prediction lên CodaLab.

## Các script chính

```text
scripts/run_inference.py              tracking video mp4
scripts/run_sportsmot_sequence.py     tracking một sequence SportsMOT
scripts/evaluate_sportsmot_sequence.py so prediction với gt/gt.txt
scripts/report_sportsmot_benchmark.py tổng hợp benchmark và lỗi ID
scripts/train_players_yolo.py         train YOLO detector
scripts/evaluate.py                   evaluate detector
```

## Cấu hình tracking

Một số tham số thường chỉnh trong `configs/inference.yaml`:

```yaml
detection:
  conf_threshold: 0.35
  iou_threshold: 0.5
  imgsz: 640
  device: cpu

tracking:
  max_age: 30
  n_init: 3
  max_cosine_distance: 0.2
  nn_budget: 100
  only_confirmed: false
```

Gợi ý chỉnh nhanh:

- tăng `conf_threshold` nếu prediction bị nhiều false positive;
- giảm `conf_threshold` nếu bị miss nhiều cầu thủ;
- tăng `imgsz` nếu cần bắt vật thể nhỏ tốt hơn, đổi lại thời gian chạy lâu hơn;
- tăng `max_age` nếu ID hay mất khi cầu thủ bị che khuất ngắn.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest tests
```

## Phạm vi

Repo này chỉ tập trung vào detection và multi-object tracking trong bóng đá. Phần phân tích chiến thuật, nhận diện đội hình hoặc sự kiện trận đấu chưa nằm trong phạm vi hiện tại.
