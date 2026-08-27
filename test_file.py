#!/usr/bin/env python3
"""Run the bundle detector against a WAV file instead of the live microphone."""

import argparse
import time
from collections import deque
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

import config
from detector_client import send_detection
from model_bundle_detector import BundleModelDetector

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("audio_file", help="WAV file to replay through the detector")
parser.add_argument(
    "--real-time",
    action="store_true",
    help=(
        "Pace chunks to arrive at the same cadence a live mic capture would "
        "(one new chunk every DETECTION_STRIDE_SECONDS) instead of running "
        "as fast as possible."
    ),
)
args = parser.parse_args()

audio, sr = sf.read(args.audio_file, dtype="float32")
if audio.ndim > 1:
    audio = audio[:, 0]

if sr != config.SAMPLE_RATE:
    from scipy.signal import resample_poly

    g = gcd(config.SAMPLE_RATE, sr)
    audio = resample_poly(audio, config.SAMPLE_RATE // g, sr // g).astype(np.float32)
    print(f"Resampled {sr} Hz → {config.SAMPLE_RATE} Hz")

project_root = Path(__file__).resolve().parent
model_path = Path(config.MODEL_BUNDLE_PATH)
scaler_path = Path(config.MODEL_SCALER_PATH)
if not model_path.is_absolute():
    model_path = project_root / model_path
if not scaler_path.is_absolute():
    scaler_path = project_root / scaler_path

detector = BundleModelDetector(model_path=model_path, scaler_path=scaler_path)
detector.load()

threshold_override = getattr(config, "BUNDLE_SCORE_THRESHOLD_OVERRIDE", None)
if threshold_override is not None:
    detector.threshold = float(threshold_override)

print(f"Model : {model_path.name}")
print(f"Threshold ({detector.threshold_name}): {detector.threshold:.4f}")
print(f"Audio : {len(audio) / config.SAMPLE_RATE:.2f}s @ {config.SAMPLE_RATE} Hz")
print()

chunk_samples = int(config.SAMPLE_RATE * config.CHUNK_DURATION)
stride_samples = int(config.SAMPLE_RATE * config.DETECTION_STRIDE_SECONDS)

detection_window_chunks = int(getattr(config, "DETECTION_WINDOW_CHUNKS", 5))
detection_min_anomalies = int(getattr(config, "DETECTION_MIN_ANOMALIES_IN_WINDOW", 2))
detection_cooldown = float(getattr(config, "DETECTION_COOLDOWN_SECONDS", 30.0))
detection_min_accuracy = float(getattr(config, "DETECTION_MIN_ACCURACY", 60.0)) / 100.0
print(
    f"Detection trigger: {detection_min_anomalies}/{detection_window_chunks} chunks, min_accuracy={detection_min_accuracy * 100:.0f}%, cooldown={detection_cooldown:.1f}s"
)
if args.real_time:
    print(f"Real-time mode: pacing to {config.DETECTION_STRIDE_SECONDS:.2f}s/chunk")
print()

pos = 0
total = 0
detections = 0
events_sent = 0
anomalies_in_window: deque[int] = deque(maxlen=detection_window_chunks)
score_window: deque[float] = deque(maxlen=detection_window_chunks)
# Simulate wall-clock time from audio timestamps to honour cooldown against recording time.
last_detection_audio_time = -detection_cooldown
start_wall = time.monotonic()
while pos + chunk_samples <= len(audio):
    if args.real_time:
        target_wall = start_wall + total * config.DETECTION_STRIDE_SECONDS
        sleep_for = target_wall - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
    chunk = audio[pos : pos + chunk_samples]
    is_anomaly, score = detector.is_anomaly(chunk)
    t = pos / config.SAMPLE_RATE
    label = "DRONE   <<<<" if is_anomaly else "no_drone"
    print(f"t={t:6.2f}s  score={score:.4f}  {label}")
    if is_anomaly:
        detections += 1
    anomalies_in_window.append(1 if is_anomaly else 0)
    score_window.append(score)
    anomaly_scores = [s for a, s in zip(anomalies_in_window, score_window) if a]
    accuracy = sum(anomaly_scores) / len(anomaly_scores) if anomaly_scores else 0.0
    if (
        sum(anomalies_in_window) >= detection_min_anomalies
        and accuracy >= detection_min_accuracy
        and t - last_detection_audio_time >= detection_cooldown
    ):
        ok = send_detection(accuracy * 100, "Drone", "detector1")
        status = "sent" if ok else "FAILED"
        print(
            f"  → Detection event {status} (window={sum(anomalies_in_window)}/{detection_window_chunks}, accuracy={accuracy * 100:.1f}%)"
        )
        last_detection_audio_time = t
        events_sent += 1
    pos += stride_samples
    total += 1

print()
print(
    f"Result: {detections}/{total} windows detected as drone ({100 * detections / max(total, 1):.1f}%)"
)
print(f"Detection events sent: {events_sent}")
