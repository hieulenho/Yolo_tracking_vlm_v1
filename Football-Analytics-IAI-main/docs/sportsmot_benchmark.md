# SportsMOT Benchmark Workflow

SportsMOT is the recommended benchmark for this project because it provides
football sequences, `gt/gt.txt` annotations, and MOTChallenge-style evaluation.

## Download

The official SportsMOT repository provides:

- CodaLab full dataset download: https://codalab.lisn.upsaclay.fr/competitions/12424#participate
- Official GitHub and format description: https://github.com/MCG-NJU/SportsMOT

The OneDrive example link may be blocked in some environments. If that happens,
download the example or full dataset manually in your browser, then extract it to:

```text
data/benchmarks/sportsmot/
```

Expected layout:

```text
data/benchmarks/sportsmot/
+-- splits_txt/
|   +-- football.txt
|   +-- train.txt
|   +-- val.txt
+-- dataset/
    +-- train/
    |   +-- v_-6Os86HzwCs_c001/
    |       +-- img1/
    |       +-- gt/gt.txt
    |       +-- seqinfo.ini
    +-- val/
```

## Recommended First Sequences

Start with:

- `v_-6Os86HzwCs_c001`
- `v_2j7kLB-vEEk_c001`

Use sequences from `train` or `val` because they include ground truth.

## Run Tracking

```powershell
python scripts/run_sportsmot_sequence.py `
  --sequence-dir "data\benchmarks\sportsmot\dataset\train\v_-6Os86HzwCs_c001" `
  --output-dir "runs\sportsmot" `
  --render-video
```

This writes:

```text
runs/sportsmot/preds/v_-6Os86HzwCs_c001.txt
runs/sportsmot/videos/v_-6Os86HzwCs_c001_tracked.mp4
runs/sportsmot/summaries/v_-6Os86HzwCs_c001.json
```

## Evaluate Locally

```powershell
python scripts/evaluate_sportsmot_sequence.py `
  --sequence-dir "data\benchmarks\sportsmot\dataset\train\v_-6Os86HzwCs_c001" `
  --pred "runs\sportsmot\preds\v_-6Os86HzwCs_c001.txt" `
  --output-json "runs\sportsmot\summaries\v_-6Os86HzwCs_c001_metrics.json"
```

The local evaluator reports:

- MOTA
- MOTP
- IDF1
- precision
- recall
- ID switches

For official SportsMOT HOTA comparison, feed the same prediction txt into the
SportsMOT/TrackEval evaluator.
