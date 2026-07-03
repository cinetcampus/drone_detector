#!/usr/bin/env python3
"""Real-time audio anomaly monitoring."""

import logging
import queue
import signal
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

import config
from features import extract_features
from model_bundle_detector import BundleModelDetector
from measurement_packet import MeasurementFormat, MeasurementPacket, MeasurementType

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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_FILE),
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
            uplink_dr=config.LORAWAN_UPLINK_DR if not config.LORAWAN_ADR_ENABLED else None,
        )
    except Exception as e:
        logger.warning(f"LoRa reporting disabled: init failed ({e})")
        return None

    logger.info("LoRa reporting enabled.")
    return client


def send_anomaly_report(lora_client, per_chunk_predictions: list[int]):
    """Send Observations TimeSeries payload like MCU send_results()."""
    predictions = [1 if int(v) > 0 else 0 for v in per_chunk_predictions]

    # TimeSeries count field is uint8, so split long windows into multiple packets.
    chunks: list[list[int]] = [predictions[i : i + 255] for i in range(0, len(predictions), 255)]
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
    buffer = np.array([], dtype=np.float32)
    anomaly_count = 0
    total_chunks = 0
  
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
                buffer = buffer[chunk_samples:]
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

                aggregate_predictions.append(1 if is_anomaly else 0)

    logger.info(
        f"Stopped. Total: {total_chunks} chunks, "
        f"{anomaly_count} drones detected ({100*anomaly_count/max(total_chunks,1):.1f}%)"
    )


if __name__ == "__main__":
    main()
