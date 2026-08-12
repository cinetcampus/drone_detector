"""Drop-in helper: import and call send_detection() from your model script.

Usage:
    from detector_client import send_detection

    # when your model detects something:
    send_detection(accuracy=93.4, label="Drone", latitude=62.2324, longitude=25.7369)
"""
import os
from typing import Optional

import requests

# Osoite, johon web.py kuuntelee tunnistuksia. Ohita ympäristömuuttujilla tarvittaessa.
DETECTOR_URL = os.getenv("DETECTOR_URL", "http://127.0.0.1:8000/api/detection")
DETECTOR_TOKEN = os.getenv("DETECTOR_TOKEN", "")


def send_detection(
    accuracy: float,
    label: str = "Detection",
    detector: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    timeout: float = 2.0,
) -> bool:
    """Lähetä yksi tunnistus web-sivulle. Palauttaa True onnistuessa.

    accuracy  : tunnistuksen tarkkuus prosentteina (esim. 93.4)
    label     : mitä tunnistettiin (esim. "Drone", "person")
    detector  : tunnistimen/mallin nimi (esim. "yolov8"); oletuksena label
    latitude  : sensorin sijainnin leveysaste (valinnainen)
    longitude : sensorin sijainnin pituusaste (valinnainen)
    """
    headers = {"X-Detector-Token": DETECTOR_TOKEN} if DETECTOR_TOKEN else {}
    payload = {
        "accuracy": accuracy,
        "label": label,
        "detector": detector or label,
        "latitude": latitude,
        "longitude": longitude,
    }
    try:
        response = requests.post(DETECTOR_URL, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        return True
    except requests.RequestException as exc:
        # Älä kaada tunnistussilmukkaa, jos web-palvelin ei ole tavoitettavissa.
        print(f"[detector_client] Failed to send detection: {exc}")
        return False


if __name__ == "__main__":
    # Pikatesti: käynnistä web.py ensin, aja sitten tämä.
    ok = send_detection(accuracy=88.8, label="Drone", detector="test")
    print("Sent" if ok else "Send failed")
