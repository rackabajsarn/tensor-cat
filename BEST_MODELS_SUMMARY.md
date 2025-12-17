# Best ESP-local models (current)

This note captures the exact settings + results for the two exported ESP32-local models derived from the same trained checkpoint.

## Source training run

- Version/run: `models/local/ms_best_catonly_seed45`
- Classes: `['prey', 'not_prey']`
- Negative policy: `cat_entering_only`
- Split manifest: `optimization/split_manifest_seed42_cat_entering_only.json`

### Key metrics (validation split used for threshold selection)

- Prey precision: **0.3690**
- Prey recall: **0.9118**
- Prey F1: **0.5254**
- Accuracy: **0.4717**

### Training parameters (repro settings)

- `epochs`: 35
- `learning_rate`: 5e-4
- `batch_size`: 32
- `seed`: 45
- `class_count`: 2
- `negative_policy`: `cat_entering_only`
- `max_samples_per_class`: 0
- `width_mult`: 2.0
- `dropout_rate`: 0.2
- `val_split`: 0.2
- `early_stop_patience`: 6
- `weight_decay`: 1e-5
- `augment`: `light`
- `use_class_weights`: true
- `label_smoothing`: 0.0
- `lr_schedule`: `cosine`
- `warmup_epochs`: 0

### Thresholds (from the run)

- Probability threshold (`export_output=probs`): `prey_threshold = 0.5176522731781006`
- Equivalent margin threshold (`export_output=logits_margin`): `prey_logit_margin_threshold = 0.07063845065389901`

## Exported model A: uint8 probs (2 outputs)

- Path: `models/local/esp_from_ms_best_catonly_seed45_probs/model/model_quant.tflite`
- Input: `uint8 [1, 96, 96, 1]`
- Output: `uint8 [1, 2]` (softmax)
- Size: **~28.06 KB**
- Ops: `CAST`, `CONV_2D`, `DEPTHWISE_CONV_2D`, `FULLY_CONNECTED`, `MEAN`, `QUANTIZE`, `SOFTMAX`
- Use in ESP metadata:
  - `number_of_labels = 2`
  - `threshold_value = 0.5176522731781006`

## Exported model B: int8 logits margin (1 output)

- Path: `models/local/esp_from_ms_best_catonly_seed45_margin/model/model_quant.tflite`
- Input: `uint8 [1, 96, 96, 1]`
- Output: `int8 [1, 1]` (prey_logit - not_prey_logit)
- Size: **~29.04 KB**
- Ops: `CAST`, `CONV_2D`, `DEPTHWISE_CONV_2D`, `FULLY_CONNECTED`, `MEAN`, `QUANTIZE`, `STRIDED_SLICE`, `SUB`
- Use in ESP metadata:
  - `number_of_labels = 1`
  - `threshold_value = 0.07063845065389901`
