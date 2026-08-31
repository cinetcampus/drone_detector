Minimal inference bundle for the two current best drone-audio models.

Files:

- tiny_mlp_es.joblib
- random_forest.joblib
- v2_spectral_mfcc_scaler.json

Input contract:

- Both models require spectral_mfcc_v1 features.
- Feature shape must be (n_windows, 100).
- Raw features must be scaled with v2_spectral_mfcc_scaler.json before scoring.
- Do not feed raw audio directly into the .joblib files.

The .joblib model packages store the model estimator, feature_set,
feature_names, labels, threshold, threshold_name, thresholds, and
deployability metadata.

Current threshold_f2 values:

- tiny_mlp_es: 0.23674599028587018
- random_forest: 0.3243004537352851

Expected runtime dependencies:

- numpy
- scipy, for matching feature extraction
- joblib
- scikit-learn

Not included:

- raw audio
- feature bundle .npz
- metrics, audit, or comparison reports
