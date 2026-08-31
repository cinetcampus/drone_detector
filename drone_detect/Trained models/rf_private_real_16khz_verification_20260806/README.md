# RF Private Real-only 16 kHz Verification Bundle

This minimal bundle reproduces frozen Random Forest inference on the private
real-only external test set. It does not retrain the model or retune the
threshold.

## Included

- 1 real positive WAV and 10 real negative WAVs.
- Private real-only manifest and preprocessing audits.
- Canonical 16 kHz `spectral_mfcc_v1` feature bundle.
- Train-fitted scaler and frozen Random Forest weights.
- Frozen evaluation JSON and model metrics.
- Self-contained inference/evaluation notebook.
- Short technical report and per-file SHA-256 inventory.

Public training data, public feature bundles, caches, alternative models, and
V3 experiments are intentionally excluded.

## Environment

Validated with Python 3.13 and:

```bash
python -m pip install \
  numpy==2.4.6 scipy==1.18.0 joblib==1.5.3 scikit-learn==1.9.0 \
  jupyterlab
```

## Run

1. Extract the ZIP and change directory to its top-level folder.
2. Open `notebooks/random_forest_16khz_inference_walkthrough.ipynb`.
3. Run all cells from top to bottom.

The notebook performs two independent checks:

1. Scores the frozen `private_real_features.npz` bundle.
2. Rebuilds the same 1,705 windows from the 11 raw WAV files:
   mono downmix -> whole-file `resample_poly` to 16 kHz -> 1.0 s windows with
   0.5 s stride -> 100 features -> frozen scaler -> frozen RF/threshold.

Expected Private real-only result:

| Metric | Expected |
| --- | ---: |
| Positive windows | 1,113 |
| Negative windows | 592 |
| Precision | 0.9473684211 |
| Recall | 0.8247978437 |
| F1 | 0.8818443804 |
| F2 | 0.8467072496 |
| FPR | 0.0861486486 |
| ROC-AUC | 0.9526526189 |
| PR-AUC | 0.9698483033 |
| Confusion matrix | TN=541, FP=51, FN=195, TP=918 |

Successful execution prints both:

```text
Private real-only reference reconciliation: PASS
Raw-WAV vs canonical bundle reconciliation: PASS
```

## Frozen contract

- Preprocessing: `mono_resample_poly_16khz_v1`
- Feature set: `spectral_mfcc_v1` (100 features)
- Model: canonical Random Forest
- Threshold: validation-selected `threshold_f2 = 0.4085200850214452`
- No private row is used for fitting, model selection, or threshold tuning.

## Caveats

- This is an external stress test with only one independent positive recording
  and ten negative recordings.
- Overlapping windows are correlated; window counts are not independent trials.
- The raw files are marked `private_research_only`. Confirm that the professor
  or data owner permits sharing with the intended recipient before transfer.
- Only load the included `.joblib` file if you trust the source of this ZIP.

Use `SHA256SUMS.txt` to verify files after transfer.
