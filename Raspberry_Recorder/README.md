# Raspberry Recorder

Standalone Flask app for recording audio on the Raspberry Pi via a physical
button (with OLED status display), and browsing/downloading/deleting the
resulting recordings from a web UI. Part of the
[Drone_Detector](../README.md) project — independent of `drone_detect`, used
to capture field recordings (e.g. for later batch-testing the detector
against).

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`gpiozero`/`lgpio` (button input) and `adafruit-blinka` +
`adafruit-circuitpython-ssd1306` + `Pillow` (I2C OLED display) are optional at
runtime — the app degrades gracefully (logs a warning, skips that feature) if
the hardware or its libraries aren't available, so this also runs on a
non-Pi machine for UI development.

## Run

```bash
python3 app.py
```

Serves on `0.0.0.0:5000`. Recordings are written to `recordings/` (gitignored).

## Hardware

- **Mic**: captured via ALSA at `ALSA_DEVICE = "plughw:3,0"` (USB Audio
  Device), `SAMPLE_RATE = 48000`, mono — see the constants near the top of
  `app.py` if the card index differs on your Pi.
- **Buttons** (GPIO, pull-up): `GPIO_RECORD_PIN = 5` (start/stop recording),
  `GPIO_WIFI_PIN = 6` (toggle WiFi radio on/off — same logic as
  `scripts/wifi-on`/`wifi-off`/`wifi-status`), `GPIO_STATUS_PIN = 12` (show
  status on the OLED).
- **Display**: optional 128x32 I2C SSD1306 OLED, shows recording/WiFi status.

## API (`app.py`)

- `GET /` — web UI.
- `GET /api/status` — current recording/WiFi state.
- `POST /api/record/start`, `POST /api/record/stop` — control recording.
- `GET /api/recordings` — list recordings.
- `GET /api/recordings/<filename>` — download one file.
- `DELETE /api/recordings/<filename>` — delete one file.
- `POST /api/recordings/delete-many` — bulk delete.
- `POST /api/recordings/<filename>/rename` — rename a recording.
- `POST /api/recordings/download-zip` — download several recordings as a zip.
