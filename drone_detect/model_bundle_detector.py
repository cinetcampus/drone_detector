"""Inference for trained model bundles in Trained models/.

Bundle contract:
- .joblib contains estimator, threshold, feature_set, feature_names, labels
- scaler JSON contains feature_names, mean, std for spectral_mfcc_v1
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
from scipy import fft as scipy_fft

import config


EPSILON = 1e-12
MEL_FILTER_COUNT = 40
LOW_BAND_HZ = (20.0, 500.0)
MID_BAND_HZ = (500.0, 2000.0)


def _safe_div(num: float, den: float, eps: float = 1e-12) -> float:
    return float(num / (den + eps))


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _hz_to_mel(value):
    return 2595.0 * np.log10(1.0 + np.asarray(value) / 700.0)


def _mel_to_hz(value):
    return 700.0 * (10.0 ** (np.asarray(value) / 2595.0) - 1.0)


def _mel_filterbank(sample_rate_hz: int, n_fft: int, mel_count: int) -> np.ndarray:
    min_mel = _hz_to_mel(0.0)
    max_mel = _hz_to_mel(sample_rate_hz / 2.0)
    mel_points = np.linspace(min_mel, max_mel, mel_count + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate_hz).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    filters = np.zeros((mel_count, n_fft // 2 + 1), dtype=np.float64)
    for index in range(1, mel_count + 1):
        left, center, right = bins[index - 1], bins[index], bins[index + 1]
        if center == left:
            center = min(center + 1, n_fft // 2)
        if right == center:
            right = min(right + 1, n_fft // 2)
        for bin_index in range(left, center):
            filters[index - 1, bin_index] = (bin_index - left) / max(center - left, 1)
        for bin_index in range(center, right):
            filters[index - 1, bin_index] = (right - bin_index) / max(right - center, 1)
    return filters


def _framed_audio(samples: np.ndarray, sample_rate_hz: int) -> np.ndarray:
    frame_length = max(1, int(round(0.025 * sample_rate_hz)))
    hop_length = max(1, int(round(0.010 * sample_rate_hz)))
    if len(samples) < frame_length:
        samples = np.pad(samples, (0, frame_length - len(samples)))
    frame_starts = range(0, len(samples) - frame_length + 1, hop_length)
    frames = [samples[start : start + frame_length] for start in frame_starts]
    if not frames:
        frames = [np.pad(samples, (0, frame_length - len(samples)))]
    return np.vstack(frames)


def _band_energy(freqs: np.ndarray, power: np.ndarray, low_hz: float, high_hz: float) -> float:
    mask = (freqs >= low_hz) & (freqs < high_hz)
    return float(np.sum(power[mask]))


def _v1_basic_features(samples: np.ndarray) -> tuple[float, ...]:
    abs_samples = np.abs(samples)
    signs = samples >= 0
    crossings = np.count_nonzero(signs[1:] != signs[:-1]) if len(samples) > 1 else 0
    zero_crossing_rate = crossings / (len(samples) - 1) if len(samples) > 1 else 0.0
    return (
        float(np.mean(samples)),
        float(np.mean(abs_samples)),
        float(np.sqrt(np.mean(samples * samples))),
        float(np.max(abs_samples)),
        float(zero_crossing_rate),
    )


def _spectral_features(samples: np.ndarray, sample_rate_hz: int) -> tuple[float, ...]:
    windowed = samples * np.hanning(len(samples))
    magnitudes = np.abs(np.fft.rfft(windowed))
    power = magnitudes * magnitudes
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate_hz)
    total_power = float(np.sum(power))
    if total_power <= EPSILON:
        return (0.0,) * 15

    centroid = float(np.sum(freqs * power) / total_power)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total_power))
    cumulative = np.cumsum(power)
    rolloff_index = int(np.searchsorted(cumulative, 0.85 * total_power, side="left"))
    rolloff_index = min(rolloff_index, len(freqs) - 1)
    rolloff = float(freqs[rolloff_index])
    flatness = float(np.exp(np.mean(np.log(power + EPSILON))) / (np.mean(power) + EPSILON))

    positive = magnitudes[magnitudes > EPSILON]
    if positive.size:
        low_percentile = float(np.percentile(positive, 5))
        high_percentile = float(np.percentile(positive, 95))
        spectral_contrast = 20.0 * math.log10((high_percentile + EPSILON) / (low_percentile + EPSILON))
    else:
        spectral_contrast = 0.0

    dominant_frequency = float(freqs[int(np.argmax(power))])

    low_energy = _band_energy(freqs, power, LOW_BAND_HZ[0], LOW_BAND_HZ[1])
    mid_energy = _band_energy(freqs, power, MID_BAND_HZ[0], MID_BAND_HZ[1])
    high_energy = _band_energy(freqs, power, MID_BAND_HZ[1], sample_rate_hz / 2.0)

    return (
        centroid,
        bandwidth,
        rolloff,
        flatness,
        spectral_contrast,
        dominant_frequency,
        low_energy,
        mid_energy,
        high_energy,
        low_energy / (total_power + EPSILON),
        mid_energy / (total_power + EPSILON),
        high_energy / (total_power + EPSILON),
        low_energy / (mid_energy + EPSILON),
        low_energy / (high_energy + EPSILON),
        mid_energy / (high_energy + EPSILON),
    )


def _mfcc_features(samples: np.ndarray, sample_rate_hz: int) -> tuple[float, ...]:
    frames = _framed_audio(samples, sample_rate_hz)
    frame_length = frames.shape[1]
    n_fft = _next_power_of_two(frame_length)
    window = np.hanning(frame_length)
    spectrum = np.fft.rfft(frames * window, n=n_fft, axis=1)
    power = (np.abs(spectrum) ** 2) / n_fft

    filters = _mel_filterbank(sample_rate_hz, n_fft, MEL_FILTER_COUNT)
    mel_energy = np.maximum(power @ filters.T, EPSILON)
    log_mel = np.log(mel_energy)

    coefficient_count = int(config.N_MFCC)
    coefficients = scipy_fft.dct(log_mel, type=2, norm="ortho", axis=1)[:, :coefficient_count]
    if coefficients.shape[0] > 1:
        deltas = np.diff(coefficients, axis=0)
    else:
        deltas = np.zeros_like(coefficients)

    return tuple(
        [float(value) for value in np.mean(coefficients, axis=0)]
        + [float(value) for value in np.std(coefficients, axis=0)]
        + [float(value) for value in np.mean(deltas, axis=0)]
        + [float(value) for value in np.std(deltas, axis=0)]
    )


def extract_spectral_mfcc_v1(audio: np.ndarray, sr: int) -> dict[str, float]:
    y = audio.astype(np.float64, copy=False)
    if y.ndim != 1:
        y = y.flatten()

    values = _v1_basic_features(y) + _spectral_features(y, sr) + _mfcc_features(y, sr)
    names = [
        "mean",
        "mean_abs",
        "rms",
        "peak_abs",
        "zero_crossing_rate",
        "spectral_centroid_hz",
        "spectral_bandwidth_hz",
        "spectral_rolloff_85_hz",
        "spectral_flatness",
        "spectral_contrast_db",
        "dominant_frequency_hz",
        "low_band_energy",
        "mid_band_energy",
        "high_band_energy",
        "low_total_ratio",
        "mid_total_ratio",
        "high_total_ratio",
        "low_mid_ratio",
        "low_high_ratio",
        "mid_high_ratio",
    ]
    coefficient_count = int(config.N_MFCC)
    names.extend([f"mfcc_{i + 1:02d}_mean" for i in range(coefficient_count)])
    names.extend([f"mfcc_{i + 1:02d}_std" for i in range(coefficient_count)])
    names.extend([f"delta_mfcc_{i + 1:02d}_mean" for i in range(coefficient_count)])
    names.extend([f"delta_mfcc_{i + 1:02d}_std" for i in range(coefficient_count)])

    feats = {}
    for name, value in zip(names, values):
        feats[name] = float(value) if np.isfinite(value) else 0.0
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
