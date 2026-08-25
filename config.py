"""Audio anomaly detector configuration."""

# Audio settings
DEVICE_INDEX = 0  # USB Audio Device (card 0)
# DEVICE_INDEX = 5  # Logitech headset
# SAMPLE_RATE = 48000
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 1.0  # seconds per analysis window
DETECTION_STRIDE_SECONDS = (
    0.5  # seconds between consecutive windows (0.5 => 50% overlap)
)

# Feature extraction
N_MELS = 64
N_MFCC = 20
HOP_LENGTH = 512
N_FFT = 2048

# Anomaly detection
THRESHOLD_MULTIPLIER = 2.5  # How many std devs from mean = anomaly
MIN_ANOMALY_DURATION = 0.5  # Minimum duration to trigger (seconds)

# Detector backend: "bundle" (Trained models) or "baseline" (IsolationForest)
DETECTOR_BACKEND = "bundle"
# DETECTOR_BACKEND = "baseline"

# Trained model bundle settings (used when DETECTOR_BACKEND == "bundle")
# MODEL_BUNDLE_PATH = "Trained models/tiny_mlp_es.joblib"
# MODEL_BUNDLE_PATH = "Trained models/random_forest.joblib"
# MODEL_SCALER_PATH = "Trained models/v2_spectral_mfcc_scaler.json"
MODEL_BUNDLE_PATH = "Trained models/rf_private_real_16khz_verification_20260806/artifacts/v2_16khz/models/v2a/models/random_forest.joblib"
MODEL_SCALER_PATH = "Trained models/rf_private_real_16khz_verification_20260806/artifacts/v2_16khz/features/scaler.json"
# Optional override for bundle score threshold (None = use model's own threshold)
BUNDLE_SCORE_THRESHOLD_OVERRIDE = None

# Detection event triggering (controls when an event is sent to the web UI)
# A detection event fires when at least DETECTION_MIN_ANOMALIES_IN_WINDOW of the
# last DETECTION_WINDOW_CHUNKS chunks are anomalies, subject to a cooldown.
DETECTION_WINDOW_CHUNKS = 5  # sliding window size in chunks
DETECTION_MIN_ANOMALIES_IN_WINDOW = 2  # anomalies required in the window to trigger
DETECTION_COOLDOWN_SECONDS = 30.0  # minimum seconds between consecutive events
DETECTION_MIN_ACCURACY = 60.0  # minimum mean anomaly-chunk score (%) to fire an event

# Reporting
ANOMALY_REPORT_INTERVAL_SECONDS = 5  # Send anomaly count once per minute
ANOMALY_REPORT_FPORT = 1

# Debug
DEBUG_LIVE_PREDICTIONS = True  # Log score + decision for each processed chunk
DEBUG_LIVE_EVERY_N_CHUNKS = 1  # 1 = every chunk, 10 = every 10th chunk
DEBUG_LIVE_SHOW_AUDIO_LEVELS = True  # Include RMS/peak levels in live debug logs

# LoRaWAN sender (used by detector.py)
LORA_SEND_ENABLED = False
LORA_BINARY_PATH = "rpi-lorawan/rpi_radiolib/rpi-sx1261"
LORA_WORKING_DIR = "rpi-lorawan/rpi_radiolib"

# LoRaWAN OTAA credentials (replace with real values)
LORAWAN_JOIN_EUI = "CAFECAFEFEEEDFDF"
LORAWAN_DEV_EUI = "CAFECAFEFEEEDFDF"
LORAWAN_APP_KEY = "CAFECAFEFEEEDFDFCAFECAFEFEEEDFDF"
LORAWAN_NWK_KEY = "CAFECAFEFEEEDFDFCAFECAFEFEEEDFDF"

# Radio settings
LORA_SPI_BUS = 0
LORA_PIN_NSS = 7
LORA_PIN_DIO0 = 22
LORA_PIN_RST = 25
LORAWAN_ADR_ENABLED = False
LORAWAN_UPLINK_DR = 5

# Paths
BASELINE_FILE = "baseline_model.pkl"
# One log file per detector.py process (named by PID), plus a "latest" symlink.
# Shared with scripts/detector-start's tmux stdout/stderr capture, so both files
# for a given run end up side by side in the same directory with matching PIDs.
LOG_DIR = "../logs"

# Training
BASELINE_DURATION = 30  # seconds of normal audio to record for training
