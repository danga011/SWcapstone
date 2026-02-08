# Next Steps Checklist

> For the next MLOps engineer picking up this project.
> Generated: 2026-02-08

---

## Before You Start

- [ ] Read `outputs/reports/HANDOFF_NOTES.md` for full context
- [ ] Read `outputs/reports/EXPERIMENT_ANALYSIS.md` for technical details
- [ ] Verify dataset is at `./final_dataset/` (not included in repo)
- [ ] Run `pip install -r requirements.txt` (check against `outputs/reports/pip_freeze_before_sleep_20260205_2156.txt`)
- [ ] Confirm `outputs/metadata.csv` exists (if not, run dataset indexing script)

---

## Phase 1: Fix PatchCore Baseline (Estimated: 2-4 hours)

- [ ] **1a.** Retrain PatchCore with `coreset_sampling_ratio=0.10` (config already exists in gated experiments)
- [ ] **1b.** Check if AUROC improves above 0.50 (current = 0.314, which is below random)
- [ ] **1c.** If still poor, try per-domain models:
  - Train PatchCore on Kolektor normals only, evaluate on Kolektor test samples
  - Train PatchCore on MVTec normals only, evaluate on MVTec test samples
- [ ] **1d.** Fix score normalization: replace `batch_heatmap_scores.max()` with a global max computed on val_mix
- [ ] **1e.** Use raw heatmap scores directly with a threshold calibrated on val_mix (instead of the 0.5+0.5*normalized formula)
- [ ] **1f.** Re-evaluate and document updated metrics

## Phase 2: Fix Gate Training (Estimated: 3-5 hours)

- [ ] **2a.** Create a `train_mix` split in `src/data/index_dataset.py` with both normal and anomaly samples
- [ ] **2b.** Modify `run_experiments.py:310` to use `train_mix` for gate training
- [ ] **2c.** Add class weights to CrossEntropyLoss: approximately `weight=[1.0, 1.94]` for the 66/34 imbalance
- [ ] **2d.** Train gate model and evaluate standalone (not in cascade) -- must achieve AUROC > 0.70
- [ ] **2e.** Plot gate score histogram (normal vs anomaly) on val_mix to verify separation
- [ ] **2f.** Only then plug into cascade and run threshold sweep

## Phase 3: Cascade Integration (Estimated: 2-3 hours)

- [ ] **3a.** Fix batch-level score normalization (BUG-2)
- [ ] **3b.** Fix baseline evaluation to not force all predictions to anomaly (BUG-3)
- [ ] **3c.** Run threshold sweep with corrected gate + corrected normalization
- [ ] **3d.** Select operating point meeting: recall >= 0.95, heatmap_call_rate in [0.20, 0.60]
- [ ] **3e.** Update summary.csv and generate new plots

## Phase 4: Cleanup (Estimated: 1 hour)

- [ ] **4a.** Rename `padim_baseline` to `patchcore_baseline` in configs and artifacts
- [ ] **4b.** Archive old non-functional gate model checkpoints
- [ ] **4c.** Update Streamlit dashboard if new experiments are added
- [ ] **4d.** Document final chosen configuration and operating point

---

## Optional: Alternative Architectures

If the supervised gate approach continues to underperform:

- [ ] **Alt-A.** Autoencoder gate: train reconstruction model on `train_normal`, use reconstruction error as gate score
- [ ] **Alt-B.** Feature-distance gate: reuse PatchCore memory bank with k=1 for fast screening
- [ ] **Alt-C.** Quality gate: simple image statistics (edge density, histogram entropy) for sub-ms filtering

---

## Files to Modify (Minimal Changes)

| File | Change | Priority |
|---|---|---|
| `src/experiments/run_experiments.py:310` | Change `train_split` to `train_mix` | Critical |
| `src/experiments/run_experiments.py:197` | Global max normalization | High |
| `src/data/index_dataset.py` | Add `train_mix` split creation | High |
| `src/models/gate.py:78` | Add class weights to loss | Medium |
| `configs/exp_padim.yaml` | Rename experiment_id | Low |

---

*End of checklist.*
