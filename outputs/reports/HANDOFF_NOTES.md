# CASCADE Anomaly Detection -- Handoff Notes

> Generated: 2026-02-08 (night-mode analysis, no experiments rerun)
> Repository: `cpst`
> Status: **Phase 1 experiments complete; gate models failed; baseline works but needs tuning**

---

## 1. Executive Summary

Three experiments were completed and evaluated:

| Experiment | Gate | Heatmap | test_mix Recall | test_mix AUROC | Heatmap Call Rate | Avg Latency |
|---|---|---|---|---|---|---|
| `padim_baseline` | None | PatchCore (R18, coreset 1%) | **1.000** | 0.314 | 1.00 | 388 ms |
| `gate_effnetb0_patchcore_r18` | EfficientNet-B0 | PatchCore (R18, coreset 10%) | 0.000 | 0.522 | 0.00 | 28 ms |
| `gate_mnv3_patchcore_r18` | MobileNetV3-Small | PatchCore (R18, coreset 10%) | 0.000 | 0.500 | 0.00 | 4 ms |

**Bottom line:** The gate models are completely non-functional (zero recall, zero heatmap calls). The PatchCore baseline catches every anomaly but flags everything as anomalous (precision = 0.34, all 839 test samples predicted anomaly). A threshold sweep covering 176 (T_low, T_high) combinations per model found no operating point that activates the heatmap.

---

## 2. Root Cause Analysis: Gate Failure

### 2.1 The Training Data Problem (Critical Bug)

The root cause is in `src/experiments/run_experiments.py`, lines 310-318:

```python
train_loader, val_loader, _ = create_dataloaders(
    metadata_csv=metadata_csv,
    train_split="train_normal",   # <-- THIS IS THE PROBLEM
    val_split="val_mix",
    ...
)
gate_model = self.train_gate_model(train_loader, val_loader)
```

The gate model (a 2-class supervised classifier) is trained on `train_normal`, which contains **2,584 samples -- all labeled normal (class 0)**. There are zero anomaly (class 1) samples in training. The model learns to predict class 0 for everything with near-perfect confidence.

### 2.2 How Score Collapse Manifests

- `predict_proba()` returns softmax probabilities. With no class-1 training data, the anomaly-class logit is always strongly negative, so `probs[:, 1]` collapses to approximately 0.0 for all inputs.
- `get_calibrated_scores()` computes `sigmoid((logit[1] - logit[0]) / temperature)`. Since `logit[1] << logit[0]` for every sample, the sigmoid input is a large negative number, and the output is approximately 0.0 regardless of temperature.

### 2.3 Cascade Consequence Chain

1. Gate score for all samples ≈ 0.0
2. `gate_score < T_low` is **True** for every sample, even when T_low = 0.001
3. All samples classified as "confident normal" → heatmap never called (`heatmap_call_rate = 0.0`)
4. `final_score = gate_prob ≈ 0.0 < 0.5` → prediction = 0 (normal) for everything
5. Accuracy = 554/839 = 0.660 = exact normal-class proportion ✓
6. Recall = 0/285 = 0.0 ✓

### 2.4 Evidence: Threshold Sweep is Invariant

The `threshold_sweep_results.csv` contains 176 rows (88 per model). Every single row shows:
- `heatmap_call_rate = 0.0`
- `recall = 0.0`
- `accuracy = 0.6603` (test_mix) or `0.6611` (val_mix)

Thresholds tested ranged from T_low = 0.001 to 0.05 and T_high = 0.01 to 0.5. None activated the heatmap, confirming total score collapse below 0.001.

### 2.5 Confusion Matrices Confirm

Both gate model confusion matrices on test_mix show the identical pattern:
```
Predicted:    Normal   Anomaly
True Normal:    554       0
True Anomaly:   285       0
```
Every sample is predicted normal. The heatmap model is never invoked.

---

## 3. Baseline (PatchCore) Analysis

### 3.1 Performance Summary

| Split | Accuracy | Precision | Recall | F1 | AUROC | PR-AUC | Latency |
|---|---|---|---|---|---|---|---|
| test_mix | 0.340 | 0.340 | 1.000 | 0.507 | 0.314 | 0.292 | 388 ms |
| neu_test | 1.000 | 1.000 | 1.000 | 1.000 | -- | 1.000 | 373 ms |

### 3.2 Why It Predicts Everything as Anomalous

The baseline runs PatchCore on every sample (no gate). In the `evaluate()` function, when `gate_model is None`:
- `gate_probs` is set to 0.5 for all samples (neutral)
- All samples fall in the `[T_low, T_high]` = `[0.3, 0.7]` uncertain range
- Heatmap runs on every sample → `heatmap_call_rate = 1.0`
- Final score = `0.5 + 0.5 * (heatmap_score / max_score)` → always ≥ 0.5
- **Every sample gets predicted as anomaly**

This means the confusion matrix is: TN=0, FP=554, FN=0, TP=285 → precision = 285/839 = 0.340.

### 3.3 The ROC Curve Tells the Real Story

The ROC curve for test_mix shows AUC = 0.314, which is **below the 0.5 random baseline**. This means the PatchCore anomaly scores are **inversely correlated** with true anomaly status -- normal samples tend to get *higher* anomaly scores than actual anomalies.

Possible causes:
1. **Coreset too small**: `coreset_sampling_ratio = 0.01` keeps only 1% of patches. With ~2,584 training images and (14x14 = 196 patches each), total patches ≈ 506K, coreset ≈ 5,060 patches. This may be too sparse to represent the normal distribution.
2. **Multi-domain confusion**: Training normal data mixes Kolektor (industrial metal, 2,071 samples) and MVTec (grid/metal_nut/tile, 513 samples). These have very different textures. A single PatchCore model may learn a poor "normal" representation when domains are mixed.
3. **Score normalization in code**: The batch-level max normalization (`batch_heatmap_scores.max()`) makes scores inconsistent across batches.

### 3.4 Heatmap Visual Quality

From the sample_predictions images:
- **test_mix**: Heatmaps show broad, diffuse activations (large blobs) without clearly localizing defects. Both normal and anomalous samples produce similar-looking heatmaps with center-focused hot regions. This confirms poor score separation.
- **neu_test**: All NEU samples (100% anomaly) are correctly scored as anomalous. Heatmaps show more coherent patterns on steel surface defects, though still quite diffuse.

### 3.5 The neu_test "Perfect Score" Is Misleading

neu_test contains 1,800 samples -- **all anomalous**. Since the model predicts everything as anomaly, it achieves recall=1.0, precision=1.0, accuracy=1.0. This is trivially correct and does not validate model quality.

---

## 4. Code-Level Issues Found

### 4.1 Critical: Gate Training Data Bug
**File:** `src/experiments/run_experiments.py:310`
**Issue:** `train_split="train_normal"` feeds only normal samples to a 2-class classifier.
**Fix:** Create a `train_mix` split with both classes, or restructure the gate to train on `val_mix`.

### 4.2 Moderate: Batch-Level Score Normalization
**File:** `src/experiments/run_experiments.py:197` and `src/experiments/threshold_sweep.py:149`
```python
0.5 + 0.5 * (batch_heatmap_scores / (batch_heatmap_scores.max() + 1e-8))
```
The `max()` is computed per-batch, not globally. This means the same raw heatmap score gets different normalized values depending on what other samples are in the batch. For consistent evaluation, compute a global max from the validation set.

### 4.3 Minor: Config Naming Confusion
**File:** `configs/exp_padim.yaml`
```yaml
experiment_id: "padim_baseline"    # Name says PaDiM
heatmap_model:
  name: "patchcore"                # But actually uses PatchCore
```
The experiment is named "padim_baseline" but uses PatchCore as the heatmap model. This is confusing for handoff.

### 4.4 Minor: Cascade Always Predicts Anomaly When Gate is None
When `gate_model is None`, the code sets `gate_probs = 0.5`, which falls inside the default `[T_low=0.3, T_high=0.7]` range, so the heatmap result is always mapped to `[0.5, 1.0]` → always predicted anomaly. The baseline should instead use the raw heatmap score with its own threshold.

---

## 5. Available Artifacts

### 5.1 Trained Models
```
models/padim_baseline/heatmap_model.pkl          # PatchCore (coreset=0.01)
models/gate_effnetb0_patchcore_r18/gate_model.pt  # NON-FUNCTIONAL
models/gate_effnetb0_patchcore_r18/heatmap_model.pkl
models/gate_mnv3_patchcore_r18/gate_model.pt      # NON-FUNCTIONAL
models/gate_mnv3_patchcore_r18/heatmap_model.pkl
```

### 5.2 Evaluation Plots (per experiment, per split)
```
models/<experiment>/confusion_matrix_<split>.png
models/<experiment>/roc_curve_<split>.png
models/<experiment>/pr_curve_<split>.png
models/<experiment>/sample_predictions_<split>.png
```

### 5.3 CSV Reports
```
outputs/reports/summary.csv                    # Main results (6 rows)
outputs/reports/threshold_sweep_results.csv    # 176 threshold combos
outputs/reports/threshold_sweep_best.csv       # Best per model (2 rows)
```

### 5.4 Environment Snapshots
```
outputs/reports/pip_freeze_20260205_1645.txt
outputs/reports/pip_freeze_before_sleep_20260205_2156.txt
```

### 5.5 Dataset Index
```
outputs/metadata.csv       # Full indexed dataset with splits
outputs/split_summary.txt  # Human-readable summary
```

---

## 6. Recommended Next Steps (Priority Order)

### Priority 1: Fix PatchCore Baseline (No Gate)

This is the fastest path to a working system.

1. **Increase coreset_sampling_ratio** from 0.01 to 0.10 or 0.25. The gated experiments already use 0.10, so compare performance.
2. **Train per-domain models**: Separate PatchCore models for Kolektor vs MVTec. Their normal textures are fundamentally different.
3. **Use raw heatmap score with calibrated threshold**: Instead of the `0.5 + 0.5 * (score / max)` normalization, compute global percentiles on the validation set and select a threshold that maximizes F1 or achieves a target recall.
4. **Evaluate with proper ROC**: The current AUROC of 0.314 suggests inverted scoring. Check if the score direction is correct (higher = more anomalous).

### Priority 2: Fix Gate Training

If a cascade architecture is still desired:

1. **Create a `train_mix` split** containing both normal and anomaly samples for gate training. The simplest approach: modify `src/data/index_dataset.py` to create this split, or train the gate on `val_mix`.
2. **Apply class-weight balancing**: `CrossEntropyLoss(weight=[1.0, 1.94])` to account for the 66/34 normal/anomaly imbalance.
3. **Validate gate independently**: Before plugging into the cascade, confirm the standalone gate achieves AUROC > 0.70 on the validation set.
4. **Fix score normalization**: Replace batch-max normalization with a global-max or percentile-based approach.

### Priority 3: Alternative Gate Architectures

If supervised gate training proves unreliable:

- **Autoencoder gate**: Train on `train_normal` only. High reconstruction error → uncertain → call heatmap. This avoids the need for anomaly labels at the gate stage.
- **Feature-distance gate**: Use the PatchCore memory bank itself as a fast gate. Compute a lightweight distance (e.g., top-1 instead of top-9 neighbors) as a pre-screen.
- **Quality gate**: Use simple image statistics (edge density, histogram entropy) as a sub-millisecond pre-filter.

### Priority 4: Evaluation Infrastructure

- Add a validation-set score distribution analysis script to visualize gate score histograms (normal vs anomaly) before plugging into the cascade.
- Add global score normalization to the evaluation pipeline.
- Fix the `padim_baseline` naming in configs.

---

## 7. Quick-Start for Next Engineer

```bash
# 1. Clone and setup
git clone <repo-url> cpst && cd cpst
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Dataset must be at ./final_dataset/ (not in repo)
# Structure: final_dataset/dataset_type/{Kolektor,MVTec,NEU}/...

# 3. Index dataset (if metadata.csv is missing)
python -m src.data.index_dataset --config configs/dataset.yaml

# 4. Run a single experiment
python -m src.experiments.run_experiments --configs configs/exp_padim.yaml

# 5. Run threshold sweep (after gated experiments exist)
python -m src.experiments.threshold_sweep --config configs/threshold_sweep.yaml

# 6. View results
cat outputs/reports/summary.csv
python -m src.dashboard  # Streamlit dashboard
```

---

## 8. Key Codebase Files

| File | Purpose |
|---|---|
| `src/experiments/run_experiments.py` | Main experiment runner (train + eval cascade) |
| `src/experiments/threshold_sweep.py` | Grid search over T_low/T_high thresholds |
| `src/models/gate.py` | Gate model (supervised binary classifier) |
| `src/models/heatmap.py` | PatchCore and PaDiM implementations |
| `src/data/dataset.py` | PyTorch dataset and dataloader factory |
| `src/data/index_dataset.py` | Dataset scanning and split creation |
| `src/eval/metrics.py` | Metrics computation and plot generation |
| `src/serve.py` | FastAPI inference server |
| `src/dashboard.py` | Streamlit results dashboard |
| `configs/exp_*.yaml` | Experiment configurations |
| `configs/threshold_sweep*.yaml` | Threshold sweep configurations |

---

## 9. Decision Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-02-05 | Train gate on `train_normal` only | Design intent was unsupervised, but gate is supervised -- **this was a mistake** |
| 2026-02-05 | Use PatchCore (not PaDiM) for baseline | PatchCore generally outperforms PaDiM on diverse datasets |
| 2026-02-05 | Set baseline `coreset_sampling_ratio=0.01` | Faster training, but sacrificed representation quality |
| 2026-02-05 | Threshold sweep with calibrated scores | Temperature=50 broadens sigmoid, but doesn't help when all scores ≈ 0 |
| 2026-02-08 | Stop threshold sweep early | No combination activated heatmap; further sweeping is futile without fixing gate |

---

*End of handoff notes.*
