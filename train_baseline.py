#!/usr/bin/env python3
"""Record normal background audio and train the anomaly detector baseline."""

import sys
import numpy as np
import sounddevice as sd

import config
from features import extract_features
from detector import AnomalyDetector


def main():
    duration = config.BASELINE_DURATION
    if len(sys.argv) > 1:
        duration = int(sys.argv[1])

    chunk_samples = int(config.SAMPLE_RATE * config.CHUNK_DURATION)

    print(f"Recording {duration}s of NORMAL background audio...")
    print("Keep the environment quiet/normal. Press Ctrl+C to stop early.")
    print()

    try:
        audio = sd.rec(
            int(config.SAMPLE_RATE * duration),
            samplerate=config.SAMPLE_RATE,
            channels=config.CHANNELS,
            dtype="float32",
            device=config.DEVICE_INDEX,
        )
        sd.wait()
    except KeyboardInterrupt:
        sd.stop()
        audio = audio[: len(audio)]
        print("\nStopped early.")

    audio = audio.flatten()
    print(f"Recorded {len(audio) / config.SAMPLE_RATE:.1f}s of audio.")

    # Split into chunks and extract features
    n_chunks = len(audio) // chunk_samples
    if n_chunks < 5:
        print("ERROR: Not enough audio. Record at least 5 seconds.")
        sys.exit(1)

    features_list = []
    for i in range(n_chunks):
        chunk = audio[i * chunk_samples : (i + 1) * chunk_samples]
        feat = extract_features(chunk)
        features_list.append(feat)

    features = np.array(features_list)
    print(f"Extracted features from {n_chunks} chunks ({features.shape[1]} features each).")

    # Train detector
    detector = AnomalyDetector()
    detector.train(features)
    detector.save()

    print(f"\nBaseline model saved to '{config.BASELINE_FILE}'.")
    print(f"Threshold score: {detector.threshold:.4f}")
    print("\nNow run: python3 detector.py")


if __name__ == "__main__":
    main()
