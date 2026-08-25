# Canonical 16 kHz Drone Audio Classification

## Technical summary

- All audio is standardized to **mono 16 kHz** before windowing.
- The model uses **100 engineered spectral/MFCC features** per one-second
  window.
- Training and threshold selection use public train/validation data only.
- The private real-only test contains **real positive + real negative data**;
  it does not contain public negatives.
- The selected operational model is the canonical **Random Forest**, using the
  frozen F2 threshold `0.4085200850214452`.

## End-to-end pipeline

```text
raw audio
  -> downmix mono
  -> resample complete recording to 16 kHz
  -> 1.0 s windows, 0.5 s stride
  -> extract 100 spectral/MFCC features
  -> apply scaler fitted on public train only
  -> Random Forest P(drone)
  -> drone if P(drone) >= 0.4085200850214452
```

## Audio preprocessing

Canonical profile: `mono_resample_poly_16khz_v1`.

1. Decode the complete recording.
2. Average channels to mono.
3. Resample once to 16 kHz with `scipy.signal.resample_poly`.
4. Cut complete 1.0-second windows after resampling; stride is 0.5 seconds.

One window therefore contains **16,000 samples**. The pipeline does not add
denoising, loudness normalization, silence removal, or processed WAV copies.
Legacy native-rate artifacts cannot be mixed with canonical 16 kHz artifacts.

## Feature extraction

Feature set: `spectral_mfcc_v1`.

| Feature group | Count | Main contents |
| --- | ---: | --- |
| Basic waveform | 5 | Mean, RMS, peak amplitude, zero-crossing rate |
| Spectral | 15 | Centroid, bandwidth, rolloff, flatness, band energies/ratios |
| MFCC | 80 | MFCC mean/std and delta mean/std |
| **Total** | **100** | One feature vector per window |

The scaler was fitted on **274,728 public training windows only**, then reused
unchanged for validation and all test sets. Feature extraction reported zero
non-finite replacements.

## Dataset construction

| Split | Origin | Drone | No drone | Total | Role |
| --- | --- | ---: | ---: | ---: | --- |
| Train | Public | 109,725 | 165,003 | 274,728 | Fit scaler/model |
| Validation | Public | 21,787 | 37,511 | 59,298 | Select model/threshold |
| Public test | Public | 23,031 | 44,213 | 67,244 | Frozen binary evaluation |
| Professor positive-only | Private real | 1,113 | 0 | 1,113 | Frozen recall evaluation |
| Private real-only | Private real | 1,113 | 592 | 1,705 | Frozen external stress test |

Private real-only reuses the same 1,113 positive windows as professor
positive-only. These two rows are therefore not independent and must not be
summed as unique windows.

Key distinction:

```text
public test = public positive + public negative
private real-only = real positive + real negative
```

Professor and private data are never used for fitting, scaler fitting,
early stopping, model selection, or threshold tuning.

## Private real-only

| Component | Recordings | Windows | Source rate | Processed rate |
| --- | ---: | ---: | ---: | ---: |
| Real positive drone | 1 | 1,113 | 16 kHz | 16 kHz |
| Real negative traffic | 5 | 316 | 48 kHz | 16 kHz |
| Real negative forest/background | 5 | 276 | 48 kHz | 16 kHz |
| **Total** | **11** | **1,705** | 16/48 kHz | **16 kHz** |

The positive recording is the same recording used in the professor
positive-only evaluation. Consequently, professor recall and private
positive-class recall are identical. The set is an external stress test, not a
deployment estimate, because it contains only one independent positive
recording and overlapping windows are correlated.

## Top 3 model finalists

All percentages below use each model's frozen validation-selected F2
threshold. Real-only results are post-freeze robustness diagnostics, not tuning
inputs.

| Model | Val F2 | Public F2 | Real precision | Real recall | Real F1 | Real F2 | Real FPR | Model size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Random Forest** | 99.42% | 99.39% | **94.74%** | **82.48%** | **88.18%** | **84.67%** | **8.61%** | 14.1 MB |
| HistGradientBoosting | **99.60%** | **99.54%** | 88.31% | 76.01% | 81.70% | 78.19% | 18.92% | 920 KB |
| Shallow XGBoost | 99.47% | 99.43% | 92.12% | 71.43% | 80.47% | 74.79% | 11.49% | **647 KB** |

The compressed-RF ablation is represented by the canonical RF family rather
than listed as a separate finalist: it reduced size only from 14.1 to 11.9 MB
and increased real FPR from 8.61% to 9.97%.

### Best model: Random Forest

**Pick: canonical Random Forest.**

Configuration: 80 trees, maximum depth 16, minimum 5 samples per leaf,
`sqrt` features per split, `balanced_subsample`, seed `20260619`.

Random Forest is larger, but it provides the best current real-world balance:

- highest real-only precision, recall, F1, and F2 among the three finalists;
- lowest real-only FPR;
- no additional XGBoost runtime dependency;
- canonical model/scaler/preprocessing contract already frozen and verified.

The smaller boosted-tree models remain research candidates. They should only
replace RF after evaluation on new independent positive recordings.

Operational artifact:
`artifacts/v2_16khz/models/v2a/models/random_forest.joblib`.
