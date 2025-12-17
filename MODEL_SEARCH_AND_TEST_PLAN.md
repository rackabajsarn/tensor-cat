# Model Search Strategy + Test Plan (CatCam local model)

This repo has two distinct model paths:

- **Local ESP32 model**: `train_simple_model.py` (grayscale 96×96, exported full-int8 TFLite; ESP32 crop fixed 192×192 from QVGA 320×240).
- **Server model**: `train_model.py` / `test_model.py` (MobileNetV2 224×224 etc).

This document targets the **local ESP32 model**.

---

## 1) Goals & Success Criteria

### Primary objective
Detect **prey** reliably. The stated aspirational goal is **$\ge 0.90$ prey recall**.

### Why recall alone is dangerous
A trivial threshold can get near-100% recall by predicting prey for everything. Therefore, define a *constrained* objective.

### Recommended success gates (offline)
Use a **held-out test set** (never used for tuning) to decide success.

Suggested gates (tune as needed):
- `prey_recall_test >= 0.90`
- `prey_precision_test >= 0.60` (or a max FP/minute target derived from real camera cadence)
- `prey_f1_test >= 0.70`
- `val_pr_auc >= 0.50` as a sanity check (PR-AUC is more informative than accuracy under imbalance)

### Deployment constraints (device)
Track (from `reports/metrics.json`):
- **TFLite size** (`tflite_details.file_size_kb`) → flash / OTA constraints
- **Arena estimate** (`tflite_details.arena_estimate_kb`) → RAM constraints
- **Latency** (measured on-device; see test plan)

If a winning model needs more RAM, try **PSRAM** and then measure end-to-end speed.

---

## 2) Test Data Strategy (required for trustworthy results)

### A) Split types
Use 3 splits:
- **Train**: used to fit weights
- **Val**: used to select checkpoint + threshold and to compare sweeps
- **Test**: used once at the end to confirm final performance

### B) Leakage prevention
Avoid splitting frames from the *same event* across train/val/test. If filenames encode timestamps/bursts, split by event group rather than individual frames.

### C) Recommended split stability
Use a fixed split across sweeps so results are comparable.

`train_simple_model.py` supports `--split_manifest` so a sweep can share identical train/val/test lists.

---

## 3) Metrics to Track (what “good” means)

### Core classification metrics
For prey:
- Recall, Precision, F1

Also:
- Confusion matrix counts (TP/FP/FN/TN)
- PR-AUC and ROC-AUC (diagnostics)
- Majority-class baseline accuracy (to avoid being fooled by ~0.70 “stuck” accuracy)

### Thresholding metrics
- The chosen `prey_threshold` (probability-space)
- `prey_threshold_stats` (FP/TP/FN and predicted positive rate)

### Model/deployment metrics
- `tflite_details.file_size_kb`
- `tflite_details.arena_estimate_kb`
- Output mode: `training_params.export_output` (`probs` vs `logits_margin`)

---

## 4) Model Search Strategy (practical phases)

### Phase 0 — Data sanity (fast)
Goal: confirm labels and preprocessing are not the bottleneck.

- Inspect a random sample of each label.
- Use the dumped preprocessed previews under each run:
  - `models/local/<run_id>/reports/images/train_set/`
  - `models/local/<run_id>/reports/images/val_set/`
- Enable misclassification dumps:
  - Run with `--dump_misclassified 30`
  - Review `models/local/<run_id>/reports/images/misclassified_val/`

If FNs/Fps look ambiguous or mislabeled, fix dataset first.

### Phase 1 — Reproducible baseline
Run a baseline training with fixed seed and fixed split.
Capture:
- metrics.json
- model size + arena
- misclassified examples

### Phase 2 — Parameter sweep (cheap search)
Use automated sweeps for:
- `width_mult` (capacity)
- `dropout` (regularization)
- `learning_rate`, `lr_schedule`, `warmup_epochs`
- `augment` (off/light/medium)
- `label_smoothing`
- `use_class_weights`
- Export mode (`logits_margin` is preferred for stable on-device thresholding)

Tooling:
- `optimization/sweep_simple_model.py` writes a leaderboard JSON/CSV.

### Phase 3 — Focused refinement (expensive)
Take top 3–5 configs and:
- increase epochs
- run multiple seeds
- confirm robustness of thresholding
- evaluate on held-out test split

### Phase 4 — On-device validation
Deploy the finalist model and verify:
- classification correctness on real captures
- speed/latency
- memory headroom (internal vs PSRAM)

---

## 5) Automated Sweep Usage

From repo root:

- Run a small sweep (example):
  - `C:/Users/madso/source/repos/tensor-cat/.venv/Scripts/python.exe optimization/sweep_simple_model.py --runs 12 --epochs 25 --prefer_recall --export_output logits_margin`

Artifacts:
- `optimization/simple_sweep_<timestamp>.json`
- `optimization/simple_sweep_<timestamp>.csv`
- Per-run outputs in `models/local/<run_id>/...`

Interpretation:
- Sort by recall/F1 first, then confirm PR-AUC and confusion matrix aren’t pathological.

---

## 6) Extensive Test Plan (checklist)

### Dataset integrity tests
- EXIF label parse success rate (spot-check + count failures)
- Class balance per split
- No duplicate filenames across splits
- No “event leakage” across splits (if grouping exists)

### Preprocessing tests
- Verify crop policy:
  - legacy images (>=480 min dim) → crop 384 → resize 96
  - new images (>=240 min dim) → crop 192 → resize 96
- Visual check using dumped `train_set/` + `val_set/`.

### Training regression tests
- Re-run baseline config weekly; compare key metrics deltas:
  - prey recall/precision/F1
  - PR-AUC
  - threshold stats

### Sweep validation tests
- Run sweep with fixed seed/split; ensure leaderboard is deterministic.
- Repeat top config across 3 seeds; ensure recall doesn’t collapse.

### Final test-set acceptance
- Lock the model only if it hits the gates on the test split.
- Archive the run folder and export artifacts.

### On-device functional tests
- Correct model metadata loaded (threshold + output mode)
- Correct crop path used (ESP32 fixed crop)
- Sanity-run with known “prey” and “no prey” images

### Performance tests (ESP32)
Measure for both internal RAM and PSRAM (if used):
- inference latency (ms)
- frames per minute throughput for your pipeline
- heap usage / free heap before/after
- arena allocation success

### Field tests
- Run for 24–72 hours with logging:
  - count prey detections
  - manually verify a sample of detections and misses
  - compute estimated FP/hour and FN/day

---

## 7) What to do if recall 0.90 is not reachable
If the probability distributions for prey vs not_prey are still heavily overlapping:
- Add more **hard negatives** (non-prey that look like prey)
- Add more **diverse prey positives** (lighting angles, distances)
- Consider revisiting label policy (ambiguous frames)
- Increase model capacity (e.g., `--width_mult 1.0` or higher) and test PSRAM

In practice, dataset quality/coverage often dominates architecture tweaks for this kind of small edge model.
