"""Inference for trained model bundles in Trained models/.

Bundle contract:
- .joblib contains estimator, threshold, feature_set, feature_names, labels
- scaler JSON contains feature_names, mean, std for spectral_mfcc_v1
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import librosa
import numpy as np

import config


def _safe_div(num: float, den: float, eps: float = 1e-12) -> float:
    return float(num / (den + eps))


def extract_spectral_mfcc_v1(audio: np.ndarray, sr: int) -> dict[str, float]:
    y = audio.astype(np.float32, copy=False)
    if y.ndim != 1:
        y = y.flatten()

    n_fft = config.N_FFT
    hop = config.HOP_LENGTH
    n_mfcc = config.N_MFCC

    abs_y = np.abs(y)
    rms = float(np.sqrt(np.mean(np.square(y))))
    peak_abs = float(np.max(abs_y)) if abs_y.size else 0.0

    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y, frame_length=n_fft, hop_length=hop)))

    S_complex = librosa.stft(y=y, n_fft=n_fft, hop_length=hop)
    S_mag = np.abs(S_complex)
    S_power = np.square(S_mag)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(S=S_mag, sr=sr)))
    spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(S=S_mag, sr=sr)))
    spectral_rolloff = float(
        np.mean(librosa.feature.spectral_rolloff(S=S_mag, sr=sr, roll_percent=0.85))
    )
    spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(S=S_mag)))
    spectral_contrast = float(np.mean(librosa.feature.spectral_contrast(S=S_mag, sr=sr)))

    mean_spectrum = np.mean(S_power, axis=1) if S_power.ndim == 2 else np.array([], dtype=np.float32)
    dominant_frequency = float(freqs[int(np.argmax(mean_spectrum))]) if mean_spectrum.size else 0.0

    low_mask = freqs < 300.0
    mid_mask = (freqs >= 300.0) & (freqs < 3000.0)
    high_mask = freqs >= 3000.0

    low_energy = float(np.sum(mean_spectrum[low_mask])) if mean_spectrum.size else 0.0
    mid_energy = float(np.sum(mean_spectrum[mid_mask])) if mean_spectrum.size else 0.0
    high_energy = float(np.sum(mean_spectrum[high_mask])) if mean_spectrum.size else 0.0
    total_energy = low_energy + mid_energy + high_energy

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=config.N_MELS,
    )
    delta_mfcc = librosa.feature.delta(mfcc)

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    delta_mean = np.mean(delta_mfcc, axis=1)
    delta_std = np.std(delta_mfcc, axis=1)

    feats: dict[str, float] = {
        "mean": float(np.mean(y)) if y.size else 0.0,
        "mean_abs": float(np.mean(abs_y)) if abs_y.size else 0.0,
        "rms": rms,
        "peak_abs": peak_abs,
        "zero_crossing_rate": zcr,
        "spectral_centroid_hz": spectral_centroid,
        "spectral_bandwidth_hz": spectral_bandwidth,
        "spectral_rolloff_85_hz": spectral_rolloff,
        "spectral_flatness": spectral_flatness,
        "spectral_contrast_db": spectral_contrast,
        "dominant_frequency_hz": dominant_frequency,
        "low_band_energy": low_energy,
        "mid_band_energy": mid_energy,
        "high_band_energy": high_energy,
        "low_total_ratio": _safe_div(low_energy, total_energy),
        "mid_total_ratio": _safe_div(mid_energy, total_energy),
        "high_total_ratio": _safe_div(high_energy, total_energy),
        "low_mid_ratio": _safe_div(low_energy, mid_energy),
        "low_high_ratio": _safe_div(low_energy, high_energy),
        "mid_high_ratio": _safe_div(mid_energy, high_energy),
    }

    for i in range(n_mfcc):
        feats[f"mfcc_{i + 1:02d}_mean"] = float(mfcc_mean[i])
    for i in range(n_mfcc):
        feats[f"mfcc_{i + 1:02d}_std"] = float(mfcc_std[i])
    for i in range(n_mfcc):
        feats[f"delta_mfcc_{i + 1:02d}_mean"] = float(delta_mean[i])
    for i in range(n_mfcc):
        feats[f"delta_mfcc_{i + 1:02d}_std"] = float(delta_std[i])

    return feats


class BundleModelDetector:
    def __init__(self, model_path: Path | str, scaler_path: Path | str):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.threshold = 0.5
        self.threshold_name = "threshold"
        self.feature_set = "unknown"
        self.labels: dict[str, str] = {}

        self._estimator = None
        self._model_feature_names: list[str] = []
        self._scaler_feature_names: list[str] = []
        self._scaler_mean: np.ndarray | None = None
        self._scaler_std: np.ndarray | None = None

    def load(self):
        bundle = joblib.load(self.model_path)
        if not isinstance(bundle, dict):
            raise ValueError("Model bundle must be a dict")

        self._estimator = bundle.get("estimator") or bundle.get("model")
        if self._estimator is None:
            raise ValueError("Model bundle missing estimator/model")

        self.threshold = float(bundle.get("threshold", 0.5))
        self.threshold_name = str(bundle.get("threshold_name", "threshold"))
        self.feature_set = str(bundle.get("feature_set", "unknown"))
        self.labels = dict(bundle.get("labels", {}))
        self._model_feature_names = list(bundle.get("feature_names", []))

        with open(self.scaler_path, "r", encoding="utf-8") as f:
            scaler = json.load(f)

        self._scaler_feature_names = list(scaler["feature_names"])
        self._scaler_mean = np.array(scaler["mean"], dtype=np.float32)
        self._scaler_std = np.array(scaler["std"], dtype=np.float32)
        self._scaler_std = np.where(self._scaler_std == 0.0, 1.0, self._scaler_std)

        if len(self._scaler_feature_names) != len(self._scaler_mean):
            raise ValueError("Scaler feature_names and mean length mismatch")
        if len(self._scaler_feature_names) != len(self._scaler_std):
            raise ValueError("Scaler feature_names and std length mismatch")

        if self._model_feature_names and self._model_feature_names != self._scaler_feature_names:
            raise ValueError("Model feature_names do not match scaler feature_names")

    def _positive_class_index(self) -> int:
        classes = list(getattr(self._estimator, "classes_", []))
        if not classes:
            return 0
        if "drone" in classes:
            return classes.index("drone")
        if 1 in classes:
            return classes.index(1)
        if True in classes:
            return classes.index(True)
        return len(classes) - 1

    def _feature_vector(self, audio_chunk: np.ndarray) -> np.ndarray:
        raw = extract_spectral_mfcc_v1(audio_chunk, sr=config.SAMPLE_RATE)
        ordered = np.array([raw[name] for name in self._scaler_feature_names], dtype=np.float32)
        return (ordered - self._scaler_mean) / self._scaler_std

    def is_anomaly(self, audio_chunk: np.ndarray) -> tuple[bool, float]:
        if self._estimator is None:
            raise RuntimeError("Model bundle not loaded")
        x = self._feature_vector(audio_chunk).reshape(1, -1)
        probs = self._estimator.predict_proba(x)[0]
        positive_idx = self._positive_class_index()
        score = float(probs[positive_idx])
        return score >= self.threshold, score
