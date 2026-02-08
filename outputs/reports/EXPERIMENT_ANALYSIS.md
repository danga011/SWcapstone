# Experiment Analysis -- Detailed Technical Breakdown

> Generated: 2026-02-08 (read-only analysis, no experiments rerun)

---

## 1. Numerical Results Table

### summary.csv (6 rows)

| experiment_id | split | accuracy | precision | recall | f1 | auroc | pr_auc | latency_ms | heatmap_call_rate |
|---|---|---|---|---|---|---|---|---|---|
| padim_baseline | test_mix | 0.340 | 0.340 | 1.000 | 0.507 | 0.314 | 0.292 | 388 | 1.00 |
| padim_baseline | neu_test | 1.000 | 1.000 | 1.000 | 1.000 | -- | 1.000 | 373 | 1.00 |
| gate_effnetb0_patchcore_r18 | test_mix | 0.660 | 0.000 | 0.000 | 0.000 | 0.522 | 0.383 | 28 | 0.00 |
| gate_effnetb0_patchcore_r18 | neu_test | 0.000 | 0.000 | 0.000 | 0.000 | -- | 1.000 | 50 | 0.00 |
| gate_mnv3_patchcore_r18 | test_mix | 0.660 | 0.000 | 0.000 | 0.000 | 0.500 | 0.670 | 4 | 0.00 |
| gate_mnv3_patchcore_r18 | neu_test | 0.000 | 0.000 | 0.000 | 0.000 | -- | 1.000 | 4 | 0.00 |

---

## 2. Threshold Sweep Summary

- **Total combinations evaluated:** 176 (88 per gate model)
- **Splits evaluated:** test_mix, val_mix
- **T_low range tested:** 0.001 to 0.05
- **T_high range tested:** 0.01 to 0.50
- **Calibration:** sigmoid with temperature=50
- **Outcome:** Every single combination yields `heatmap_call_rate = 0.0` and `recall = 0.0`

This proves that **all gate scores are below 0.001** for every sample. The sweep is exhaustive evidence of total score collapse.

---

## 3. Dataset Split Composition

```
train_normal : 2,584 samples (100% normal)  -- Kolektor: 2,071 + MVTec: 513
val_mix      :   838 samples (66% normal)   -- Kolektor: 607 + MVTec: 231
test_mix     :   839 samples (66% normal)   -- Kolektor: 635 + MVTec: 204
neu_test     : 1,800 samples (100% anomaly) -- NEU only
```

Key implication: The gate classifier is trained on `train_normal` (0% anomaly) but expected to distinguish normal from anomaly. This is a fundamental data pipeline bug.

---

## 4. Heatmap Visual Analysis

### 4.1 test_mix Sample Predictions (padim_baseline)

Reviewed 10 sample prediction visualizations (Original | Heatmap | Overlay):
- **Heatmaps are diffuse and non-localizing.** Activation blobs cover large portions of each image without focusing on defect regions.
- **Normal and anomalous samples produce similar heatmap patterns.** Both show center-heavy or broadly spread activations.
- **No visible correlation between heatmap intensity and actual defect location.**
- This explains the below-random AUROC (0.314): the model's scoring direction is inverted relative to ground truth.

### 4.2 neu_test Sample Predictions (padim_baseline)

Reviewed 10 NEU steel defect samples:
- Heatmaps show somewhat more structured patterns (linear activations on scratch/crack defects).
- However, since all NEU samples are anomalous and the model predicts everything as anomaly, this evaluation doesn't test discrimination ability.

### 4.3 Gate Model Sample Predictions

Both EfficientNet-B0 and MobileNetV3 confusion matrices show identical all-normal predictions:
```
                Predicted Normal    Predicted Anomaly
True Normal           554                  0
True Anomaly          285                  0
```
The heatmap columns in their sample_predictions images are blank (all zeros) because the heatmap model was never invoked.

---

## 5. Code Architecture Notes

### 5.1 Score Flow in Cascade Evaluation

```
Input Image
    |
    v
[Gate Model] --> gate_prob (softmax[:, 1] or calibrated sigmoid)
    |
    |-- if gate_prob < T_low:  final_score = gate_prob   (confident normal)
    |-- if gate_prob > T_high: final_score = gate_prob   (confident anomaly)
    |-- else:
    |      [Heatmap Model] --> raw_heatmap_score
    |      final_score = 0.5 + 0.5 * (raw / batch_max)  (maps to [0.5, 1.0])
    |
    v
prediction = (final_score >= 0.5) ? anomaly : normal
```

### 5.2 PatchCore Implementation Details

- Backbone: torchvision ResNet18 (ImageNet pretrained, frozen)
- Feature extraction: layer2 (128-d, 28x28) + layer3 (256-d, 14x14), upsampled and concatenated
- Random projection: SparseRandomProjection (eps=0.9) for dimensionality reduction
- Memory bank: Random coreset subsampling (no greedy coreset)
- Distance: k-NN (k=9) using mean of k smallest squared Euclidean distances
- Post-processing: Gaussian smoothing (sigma=4) + bilinear upsampling to 224x224
- Image-level score: max of smoothed heatmap

### 5.3 Gate Model Implementation Details

- Backbone: timm models (efficientnet_b0 or mobilenetv3_small_100)
- Output: 2-class logits, softmax for probability
- Training: AdamW (lr=0.001, wd=0.0001), CosineAnnealingLR (T_max=20)
- Loss: CrossEntropyLoss (unweighted)
- Epochs: 20 with best-val-accuracy checkpoint

---

## 6. Bugs and Issues Catalog

| ID | Severity | Location | Description |
|---|---|---|---|
| BUG-1 | **Critical** | `run_experiments.py:310` | Gate trained on normal-only data; cannot learn anomaly class |
| BUG-2 | **High** | `run_experiments.py:197` | Batch-level max normalization makes scores inconsistent across batches |
| BUG-3 | **Moderate** | `run_experiments.py:170` | Baseline forces all samples into uncertain range (gate_prob=0.5), guaranteeing all predictions are anomaly |
| BUG-4 | **Low** | `exp_padim.yaml` | Config named "padim_baseline" but uses PatchCore model |
| BUG-5 | **Low** | `gate.py:78` | CrossEntropyLoss has no class weights despite imbalanced val_mix (66/34 split) |

---

## 7. What NOT to Re-run

- Do not re-run threshold sweeps on existing gate models. All scores are collapsed; no threshold can fix this.
- Do not re-run the padim_baseline with the same coreset_sampling_ratio=0.01. The AUROC < 0.5 indicates the model representation is fundamentally insufficient at this ratio.
- Do not try to calibrate existing gate checkpoints. They have never seen anomaly data; fine-tuning won't help.

---

*End of analysis.*
