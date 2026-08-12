"""Example: call notify_detection() from your existing model script."""
import os
import random
import time
import requests

DETECTOR_URL = os.getenv("DETECTOR_URL", "http://127.0.0.1:8000/api/detection")
DETECTOR_TOKEN = os.getenv("DETECTOR_TOKEN", "")


def notify_detection(accuracy: float, label: str = "Drone", latitude=None, longitude=None):
    headers = {"X-Detector-Token": DETECTOR_TOKEN} if DETECTOR_TOKEN else {}
    response = requests.post(
        DETECTOR_URL,
        json={
            "accuracy": accuracy,
            "label": label,
            "latitude": latitude,
            "longitude": longitude,
        },
        headers=headers,
        timeout=2,
    )
    response.raise_for_status()


if __name__ == "__main__":
    print("Sending randomized detections every 10 seconds. Press Ctrl+C to stop.")
    try:
        while True:
            accuracy = round(random.uniform(70.0, 100.0), 1)
            notify_detection(accuracy, "Drone")
            print(f"Detection sent, accuracy={accuracy}%")
            time.sleep(10)
    except KeyboardInterrupt:
        print("Stopped")
