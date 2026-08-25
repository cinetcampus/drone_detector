#!/usr/bin/env python3
"""Real-time audio anomaly monitoring."""

import logging
import os
import queue
import signal
import sys
import argparse
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd

import joblib
from sklearn.ensemble import IsolationForest

import config
from features import extract_features
from model_bundle_detector import BundleModelDetector
from measurement_packet import MeasurementFormat, MeasurementPacket, MeasurementType
from detector_client import send_detection, send_confidence

# Allow importing RadioLib wrapper from rpi-lorawan subdirectory.
PROJECT_ROOT = Path(__file__).resolve().parent
RPI_LORAWAN_DIR = PROJECT_ROOT / "rpi-lorawan"
if str(RPI_LORAWAN_DIR) not in sys.path:
    sys.path.append(str(RPI_LORAWAN_DIR))

try:
    from radiolib_client import RadioLibClient, RadioLibError
except Exception:  # pragma: no cover - optional at runtime
    RadioLibClient = None
    RadioLibError = RuntimeError

# Setup logging: one file per process (named by PID) plus a "latest" symlink,
# so a single long-lived log can't grow unbounded or blur together unrelated runs.
LOG_DIR = Path(config.LOG_DIR)
if not LOG_DIR.is_absolute():
    LOG_DIR = (PROJECT_ROOT / LOG_DIR).resolve()
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE_PATH = LOG_DIR / f"detections_{os.getpid()}.log"
LATEST_LOG_LINK = LOG_DIR / "detections-latest.log"
try:
    if LATEST_LOG_LINK.is_symlink() or LATEST_LOG_LINK.exists():
        LATEST_LOG_LINK.unlink()
    LATEST_LOG_LINK.symlink_to(LOG_FILE_PATH.name)
except OSError:
    pass  # symlinks unsupported on this filesystem; logging still works without it

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE_PATH),
    ],
)
logger = logging.getLogger(__name__)

# Audio queue
audio_queue: queue.Queue[np.ndarray] = queue.Queue()
running = True


def signal_handler(sig, frame):
    global running
    running = False
    print("\nStopping...")


def audio_callback(indata, frames, time_info, status):
    if status:
        logger.warning(f"Audio status: {status}")
    audio_queue.put(indata.copy())


def init_lora_client():
    """Initialize LoRa client for periodic anomaly count uplinks."""
    if not config.LORA_SEND_ENABLED:
        logger.info("LoRa reporting disabled by config.")
        return None
    if RadioLibClient is None:
        logger.warning("LoRa reporting disabled: radiolib_client import failed.")
        return None

    binary_path = Path(config.LORA_BINARY_PATH)
    if not binary_path.is_absolute():
        binary_path = PROJECT_ROOT / binary_path
    working_dir = Path(config.LORA_WORKING_DIR)
    if not working_dir.is_absolute():
        working_dir = PROJECT_ROOT / working_dir

    if not binary_path.exists():
        logger.warning(f"LoRa reporting disabled: binary not found at {binary_path}")
        return None

    try:
        client = RadioLibClient(
            binary_path=str(binary_path),
            working_dir=str(working_dir),
            join_eui=config.LORAWAN_JOIN_EUI,
            dev_eui=config.LORAWAN_DEV_EUI,
            app_key=config.LORAWAN_APP_KEY,
            nwk_key=config.LORAWAN_NWK_KEY,
            spi_bus=config.LORA_SPI_BUS,
            pin_nss=config.LORA_PIN_NSS,
            pin_dio0=config.LORA_PIN_DIO0,
            pin_rst=config.LORA_PIN_RST,
            adr_enabled=config.LORAWAN_ADR_ENABLED,
            uplink_dr=config.LORAWAN_UPLINK_DR
            if not config.LORAWAN_ADR_ENABLED
            else None,
        )
    except Exception as e:
        logger.warning(f"LoRa reporting disabled: init failed ({e})")
        return None

    logger.info("LoRa reporting enabled.")
    return client


def send_anomaly_report(lora_client, per_chunk_predictions: list[int]):
    """Send Observations TimeSeries payload like MCU send_results()."""
    predictions = [1 if int(v) > 0 else 0 for v in per_chunk_predictions]

    # Packet payload length is uint8 (max 255 bytes). For one TIME_SERIES entry we have:
    # [type:1][count:1][values:N] => N must be <= 253.
    max_values_per_packet = 253
    chunks: list[list[int]] = [
        predictions[i : i + max_values_per_packet]
        for i in range(0, len(predictions), max_values_per_packet)
    ]
    if not chunks:
        chunks = [[]]

    for idx, chunk in enumerate(chunks, start=1):
        packet = MeasurementPacket()
        packet.add_array8(
            MeasurementType.OBSERVATIONS,
            MeasurementFormat.TIME_SERIES,
            chunk,
        )
        payload = packet.to_bytes()

        if lora_client is None:
            logger.info(
                f"Minute anomaly report part {idx}/{len(chunks)}: "
                f"observations={len(chunk)} anomalies={sum(chunk)} "
                f"(LoRa disabled), payload={payload.hex().upper()}"
            )
            continue

        try:
            result = lora_client.send_uplink(payload, fport=config.ANOMALY_REPORT_FPORT)
            logger.info(
                f"Minute anomaly report part {idx}/{len(chunks)} sent: "
                f"observations={len(chunk)} anomalies={sum(chunk)}, "
                f"payload={payload.hex().upper()}, FCntUp={result.fcnt_up}"
            )
        except (RadioLibError, FileNotFoundError) as e:
            logger.error(f"Failed to send minute anomaly report part {idx}: {e}")


class AnomalyDetector:
    """Baseline IsolationForest anomaly detector for normal audio."""

    def __init__(self):
        self.model: IsolationForest | None = None
        self.threshold: float = 0.0
        self._model_path = Path(config.BASELINE_FILE)

    def train(self, features: np.ndarray) -> None:
        self.model = IsolationForest(contamination=0.05, random_state=42)
        self.model.fit(features)
        scores = -self.model.score_samples(features)
        self.threshold = float(
            np.mean(scores) + config.THRESHOLD_MULTIPLIER * np.std(scores)
        )

    def save(self) -> None:
        if self.model is None:
            raise RuntimeError("Model not trained yet.")
        joblib.dump(
            {"model": self.model, "threshold": self.threshold}, self._model_path
        )

    def load(self) -> None:
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Baseline model not found at '{self._model_path}'. "
                "Run train_baseline.py first."
            )
        data = joblib.load(self._model_path)
        self.model = data["model"]
        self.threshold = float(data["threshold"])

    def is_anomaly(self, features: np.ndarray) -> tuple[bool, float]:
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        score = float(-self.model.score_samples(features.reshape(1, -1))[0])
        return score > self.threshold, score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time audio anomaly monitoring")
    parser.add_argument(
        "--aggregate-seconds",
        type=float,
        default=config.ANOMALY_REPORT_INTERVAL_SECONDS,
        help=(
            "Aggregation/report interval in seconds for anomaly count uplinks "
            f"(default: {config.ANOMALY_REPORT_INTERVAL_SECONDS})"
        ),
    )
    args = parser.parse_args()
    if args.aggregate_seconds <= 0:
        parser.error("--aggregate-seconds must be > 0")
    return args


def main():
    global running
    signal.signal(signal.SIGINT, signal_handler)
    args = parse_args()
    report_interval = float(args.aggregate_seconds)

    backend = str(getattr(config, "DETECTOR_BACKEND", "baseline")).strip().lower()
    if backend == "bundle":
        model_path = Path(config.MODEL_BUNDLE_PATH)
        scaler_path = Path(config.MODEL_SCALER_PATH)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        if not scaler_path.is_absolute():
            scaler_path = PROJECT_ROOT / scaler_path

        detector = BundleModelDetector(model_path=model_path, scaler_path=scaler_path)
        try:
            detector.load()
        except Exception as e:
            print(f"ERROR: Failed to load model bundle: {e}")
            sys.exit(1)

        threshold_override = getattr(config, "BUNDLE_SCORE_THRESHOLD_OVERRIDE", None)
        if threshold_override is not None:
            detector.threshold = float(threshold_override)

        def detect_chunk(chunk: np.ndarray) -> tuple[bool, float]:
            return detector.is_anomaly(chunk)

        threshold_txt = f"{detector.threshold_name}={detector.threshold:.4f}"
        backend_txt = f"bundle ({model_path.name})"
    else:
        detector = AnomalyDetector()
        try:
            detector.load()
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

        def detect_chunk(chunk: np.ndarray) -> tuple[bool, float]:
            features = extract_features(chunk)
            return detector.is_anomaly(features)

        threshold_txt = f"threshold={detector.threshold:.4f}"
        backend_txt = "baseline (IsolationForest)"

    logger.info("Audio anomaly detector started.")
    logger.info(f"Detector backend: {backend_txt}")
    logger.info(f"Device: {config.DEVICE_INDEX}, Rate: {config.SAMPLE_RATE} Hz")
    logger.info(f"Detector threshold: {threshold_txt}")
    logger.info(f"Aggregation/report interval: {report_interval:.1f}s")
    debug_live = bool(getattr(config, "DEBUG_LIVE_PREDICTIONS", False))
    debug_every_n = max(1, int(getattr(config, "DEBUG_LIVE_EVERY_N_CHUNKS", 1)))
    debug_live_levels = bool(getattr(config, "DEBUG_LIVE_SHOW_AUDIO_LEVELS", False))
    if debug_live:
        logger.info(f"Live prediction debug enabled (every {debug_every_n} chunk(s)).")
    logger.info("Listening... (Ctrl+C to stop)")

    chunk_samples = int(config.SAMPLE_RATE * config.CHUNK_DURATION)
    stride_samples = int(
        config.SAMPLE_RATE
        * getattr(config, "DETECTION_STRIDE_SECONDS", config.CHUNK_DURATION)
    )
    if stride_samples <= 0:
        print("ERROR: DETECTION_STRIDE_SECONDS must be > 0")
        sys.exit(1)
    if stride_samples > chunk_samples:
        print("ERROR: DETECTION_STRIDE_SECONDS cannot be greater than CHUNK_DURATION")
        sys.exit(1)
    logger.info(
        f"Window={config.CHUNK_DURATION:.2f}s stride={stride_samples / config.SAMPLE_RATE:.2f}s "
        f"overlap={(1.0 - stride_samples / max(chunk_samples, 1)) * 100:.0f}%"
    )

    detection_window_chunks = int(getattr(config, "DETECTION_WINDOW_CHUNKS", 5))
    detection_min_anomalies = int(getattr(config, "DETECTION_MIN_ANOMALIES_IN_WINDOW", 2))
    detection_cooldown = float(getattr(config, "DETECTION_COOLDOWN_SECONDS", 30.0))
    detection_min_accuracy = float(getattr(config, "DETECTION_MIN_ACCURACY", 60.0)) / 100.0
    logger.info(
        f"Detection trigger: {detection_min_anomalies}/{detection_window_chunks} chunks, "
        f"min_accuracy={detection_min_accuracy*100:.0f}%, cooldown={detection_cooldown:.1f}s"
    )

    buffer = np.array([], dtype=np.float32)
    anomaly_count = 0
    total_chunks = 0
    anomaly_window: deque[int] = deque(maxlen=detection_window_chunks)
    score_window: deque[float] = deque(maxlen=detection_window_chunks)
    last_detection_time = 0.0  # epoch zero allows first event to fire immediately
    last_confidence_send = 0.0
    # Peak accuracy seen since the last sent event. Kept up to date every chunk
    # (even while cooldown blocks sending) so that when cooldown lifts we report
    # the strongest reading during the approach instead of a stale/instantaneous
    # one. Reset whenever the drone drops back out of range, so it can't bleed
    # into an unrelated later pass.
    peak_accuracy_since_send = 0.0

    aggregate_predictions: list[int] = []
    last_report_ts = time.monotonic()
    lora_client = init_lora_client()

    with sd.InputStream(
        device=config.DEVICE_INDEX,
        samplerate=config.SAMPLE_RATE,
        channels=config.CHANNELS,
        dtype="float32",
        blocksize=4096,
        callback=audio_callback,
    ):
        while running:
            now = time.monotonic()
            while now - last_report_ts >= report_interval:
                send_anomaly_report(lora_client, aggregate_predictions)
                aggregate_predictions = []
                last_report_ts += report_interval

            try:
                data = audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            buffer = np.append(buffer, data.flatten())

            while len(buffer) >= chunk_samples:
                chunk = buffer[:chunk_samples]
                buffer = buffer[stride_samples:]
                total_chunks += 1
                rms = float(np.sqrt(np.mean(chunk**2)))
                peak = float(np.max(np.abs(chunk))) if len(chunk) else 0.0

                is_anomaly, score = detect_chunk(chunk)

                if debug_live and (total_chunks % debug_every_n == 0):
                    if debug_live_levels:
                        logger.info(
                            f"LIVE chunk={total_chunks} score={score:.4f} "
                            f"prediction={'drone' if is_anomaly else 'no_drone'} "
                            f"rms={rms:.4f} peak={peak:.4f}"
                        )
                    else:
                        logger.info(
                            f"LIVE chunk={total_chunks} score={score:.4f} "
                            f"prediction={'drone' if is_anomaly else 'no_drone'}"
                        )

                if is_anomaly:
                    anomaly_count += 1
                    logger.warning(
                        f"DRONE detected! Score: {score:.4f}, "
                        f"RMS: {rms:.4f}, "
                        f"Count: {anomaly_count}/{total_chunks}"
                    )

                anomaly_window.append(1 if is_anomaly else 0)
                score_window.append(score)
                anomaly_scores = [s for a, s in zip(anomaly_window, score_window) if a]
                accuracy = sum(anomaly_scores) / len(anomaly_scores) if anomaly_scores else 0.0
                live_confidence = sum(score_window) / len(score_window) if score_window else 0.0
                now_mono = time.monotonic()

                trigger_condition = (
                    sum(anomaly_window) >= detection_min_anomalies
                    and accuracy >= detection_min_accuracy
                )
                if trigger_condition:
                    peak_accuracy_since_send = max(peak_accuracy_since_send, accuracy)
                else:
                    peak_accuracy_since_send = 0.0

                if trigger_condition and now_mono - last_detection_time >= detection_cooldown:
                    report_accuracy = peak_accuracy_since_send
                    send_detection(report_accuracy * 100, "Drone", "detector1")
                    last_detection_time = now_mono
                    peak_accuracy_since_send = 0.0
                    logger.info(
                        f"Detection event sent (window={sum(anomaly_window)}/{detection_window_chunks}, "
                        f"accuracy={report_accuracy*100:.1f}%)"
                    )

                # Send live confidence update ~once per second (non-blocking).
                is_drone_now = sum(anomaly_window) >= detection_min_anomalies
                if now_mono - last_confidence_send >= 1.0:
                    threading.Thread(
                        target=send_confidence,
                        args=(live_confidence * 100, is_drone_now),
                        daemon=True,
                    ).start()
                    last_confidence_send = now_mono

                aggregate_predictions.append(1 if is_anomaly else 0)

    logger.info(
        f"Stopped. Total: {total_chunks} chunks, "
        f"{anomaly_count} drones detected ({100 * anomaly_count / max(total_chunks, 1):.1f}%)"
    )


if __name__ == "__main__":
    main()
