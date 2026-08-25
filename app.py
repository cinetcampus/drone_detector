import io
import logging
import os
import signal
import socket
import subprocess
import threading
import time
import zipfile
from datetime import datetime

import numpy as np
import scipy.io.wavfile as wavfile
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

try:
    from gpiozero import Button, Device
    from gpiozero.exc import BadPinFactory
    try:
        from gpiozero.pins.lgpio import LGPIOFactory
    except Exception:
        LGPIOFactory = None
except Exception:
    Button = None
    Device = None
    BadPinFactory = Exception
    LGPIOFactory = None

try:
    import board
    from PIL import Image, ImageDraw, ImageFont
    from adafruit_ssd1306 import SSD1306_I2C
except Exception:
    board = None
    Image = None
    ImageDraw = None
    ImageFont = None
    SSD1306_I2C = None

log = logging.getLogger("recorder")
log.setLevel(logging.DEBUG)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
log.addHandler(_handler)

app = Flask(__name__)

RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
print("RECORDING_DIR " + RECORDINGS_DIR)
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# USB Audio Device (hw:0"
ALSA_DEVICE = "plughw:CARD=Headset,DEV=0"
SAMPLE_RATE = 48000
CHANNELS = 1

# GPIO button pins
GPIO_RECORD_PIN = 5
GPIO_WIFI_PIN = 6
GPIO_STATUS_PIN = 12

# OLED display
OLED_WIDTH = 128
OLED_HEIGHT = 32

# Recording state
_recording = {
    "active": False,
    "process": None,
    "filepath": None,
    "start_time": None,
}

# Wi-Fi state
_wifi = {
    "enabled": None,
    "last_error": None,
}

_state_lock = threading.RLock()
_display_notice = {"text": "", "until": 0}

# Hardware handles
_buttons = []
_oled = None
_oled_font = None


def _safe_filepath(filename):
    """Return a safe absolute path inside RECORDINGS_DIR, or None."""
    name = secure_filename(filename)
    if not name:
        return None
    path = os.path.join(RECORDINGS_DIR, name)
    if os.path.commonpath([RECORDINGS_DIR, os.path.realpath(path)]) != RECORDINGS_DIR:
        return None
    return path


def _run_command(command):
    return subprocess.run(command, capture_output=True, text=True)


def _wifi_enabled_from_nmcli():
    result = _run_command(["nmcli", "-t", "-f", "WIFI", "g"])
    if result.returncode != 0:
        return None
    value = result.stdout.strip().lower()
    if value == "enabled":
        return True
    if value == "disabled":
        return False
    return None


def _wifi_enabled_from_rfkill():
    result = _run_command(["rfkill", "list", "wifi"])
    if result.returncode != 0:
        return None

    text = result.stdout.lower()
    if "soft blocked: yes" in text:
        return False
    if "soft blocked: no" in text:
        return True
    return None


def detect_wifi_enabled():
    state = _wifi_enabled_from_nmcli()
    if state is not None:
        return state
    state = _wifi_enabled_from_rfkill()
    if state is not None:
        return state
    return None


def set_wifi_enabled(enabled):
    command_sets = []
    if enabled:
        command_sets = [
            [["nmcli", "radio", "wifi", "on"]],
            [["rfkill", "unblock", "wifi"]],
        ]
    else:
        command_sets = [
            [["nmcli", "radio", "wifi", "off"]],
            [["rfkill", "block", "wifi"]],
        ]

    errors = []
    for command_group in command_sets:
        group_ok = True
        for command in command_group:
            try:
                result = _run_command(command)
            except FileNotFoundError:
                group_ok = False
                errors.append(f"Komentoa ei löydy: {' '.join(command)}")
                break

            if result.returncode != 0:
                group_ok = False
                stderr = (result.stderr or "").strip() or "tuntematon virhe"
                errors.append(f"{' '.join(command)} -> {stderr}")
                break

        if group_ok:
            break

    time.sleep(0.3)
    detected = detect_wifi_enabled()
    _wifi["enabled"] = detected

    if detected is None:
        _wifi["last_error"] = "Wi-Fi-tilaa ei voitu lukea"
        return False, _wifi["last_error"]

    if detected != enabled:
        _wifi["last_error"] = "Wi-Fi-tila ei paivittynyt"
        return False, _wifi["last_error"]

    _wifi["last_error"] = None
    return True, "Wi-Fi paivitetty"


def _wifi_label():
    if _wifi["enabled"] is True:
        return "ON"
    if _wifi["enabled"] is False:
        return "OFF"
    return "?"


def _get_local_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except Exception:
        return "-"
    finally:
        sock.close()


def _set_display_notice(text, seconds=4):
    _display_notice["text"] = text
    _display_notice["until"] = time.time() + seconds


def _configure_gpio_backend():
    if Device is None:
        return

    if os.environ.get("GPIOZERO_PIN_FACTORY"):
        log.info("GPIOZERO_PIN_FACTORY=%s", os.environ.get("GPIOZERO_PIN_FACTORY"))
        return

    if LGPIOFactory is None:
        log.warning("lgpio-pinta puuttuu: asenna 'pip install lgpio' tai 'sudo apt install python3-lgpio'")
        return

    try:
        Device.pin_factory = LGPIOFactory()
        log.info("GPIO pin factory: lgpio")
    except BadPinFactory as exc:
        log.warning("GPIO pin factoryn alustus epaonnistui: %s", exc)
    except Exception as exc:
        log.warning("GPIO backendin alustus epaonnistui: %s", exc)


def _init_oled():
    global _oled, _oled_font
    if board is None or SSD1306_I2C is None:
        log.warning(
            "OLED-kirjastoja ei löydy, nayttoa ei alustettu. Asenna: pip install adafruit-blinka adafruit-circuitpython-ssd1306 pillow"
        )
        return

    try:
        i2c = board.I2C()
        _oled = SSD1306_I2C(OLED_WIDTH, OLED_HEIGHT, i2c)
        _oled.fill(0)
        _oled.show()
        _oled_font = ImageFont.load_default()
        log.info("OLED alustettu")
    except Exception as exc:
        _oled = None
        log.warning("OLED alustus epaonnistui: %s", exc)


def _draw_oled(lines):
    if _oled is None or Image is None or ImageDraw is None:
        return
    image = Image.new("1", (_oled.width, _oled.height))
    draw = ImageDraw.Draw(image)
    y = 0
    for line in lines[:4]:
        draw.text((0, y), line[:21], font=_oled_font, fill=255)
        y += 10
    _oled.image(image)
    _oled.show()


def update_oled_status():
    with _state_lock:
        if _recording["active"] and _recording["start_time"]:
            elapsed = int(time.time() - _recording["start_time"])
            mm = elapsed // 60
            ss = elapsed % 60
            line2 = f"REC {mm:02d}:{ss:02d}"
        else:
            line2 = "REC idle"

        line3 = f"WiFi: {_wifi_label()}"
        notice_active = time.time() < _display_notice["until"]
        line4 = _display_notice["text"] if notice_active else "5: Rec | 6: WiFi"
        _draw_oled([line2, line3, line4])


def _display_loop():
    while True:
        try:
            update_oled_status()
        except Exception as exc:
            log.debug("OLED paivitys epaonnistui: %s", exc)
        time.sleep(1)


def _start_recording_internal():
    with _state_lock:
        if _recording["active"]:
            return False, {"error": "Aanitys on jo kaynnissa"}

        filename = datetime.now().strftime("recording_%Y%m%d_%H%M%S.wav")
        filepath = os.path.join(RECORDINGS_DIR, filename)

        log.info("Aanitys alkaa - laite: %s, tiedosto: %s", ALSA_DEVICE, filename)

        proc = subprocess.Popen(
            [
                "arecord",
                "-D",
                ALSA_DEVICE,
                "-f",
                "S16_LE",
                "-r",
                str(SAMPLE_RATE),
                "-c",
                str(CHANNELS),
                "-t",
                "wav",
                filepath,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # arecord fails almost instantly (e.g. "Device or resource busy") if
        # the mic is already held by another process, such as the detector.
        # Give it a brief moment to fail before reporting success, otherwise
        # the UI/OLED shows "recording" while no file is actually written.
        time.sleep(0.3)
        if proc.poll() is not None:
            stderr_output = proc.stderr.read().decode(errors="replace").strip() if proc.stderr else ""
            log.error("Aanitys ei kaynnistynyt (mikrofoni varattu?): %s", stderr_output or "tuntematon virhe")
            if os.path.exists(filepath):
                os.remove(filepath)
            _set_display_notice("Mic busy!", 5)
            return False, {
                "error": "Mikrofoni ei ole kaytettavissa (toinen ohjelma kayttaa sita?)",
                "detail": stderr_output,
            }

        _recording["active"] = True
        _recording["process"] = proc
        _recording["filepath"] = filepath
        _recording["start_time"] = time.time()
        _set_display_notice("Recording started", 2)
        return True, {"status": "recording"}


def _stop_recording_internal():
    with _state_lock:
        if not _recording["active"]:
            return False, {"error": "Aanitys ei ole kaynnissa"}

        proc = _recording["process"]
        if proc and proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=5)

        _recording["active"] = False
        filepath = _recording["filepath"]
        _recording["process"] = None
        _recording["filepath"] = None
        _recording["start_time"] = None

    if not filepath or not os.path.isfile(filepath):
        return False, {"error": "Aanitiedostoa ei löydy"}

    duration = 0
    try:
        rate, data = wavfile.read(filepath)
        duration = len(data) / rate
        log.info(
            "Aanitys tallennettu: %s (%.1f s, %s)",
            os.path.basename(filepath),
            duration,
            _format_size(os.path.getsize(filepath)),
        )
    except Exception as exc:
        log.error("WAV-tiedoston lukuvirhe: %s", exc)

    size = os.path.getsize(filepath)
    _set_display_notice("Saved " + os.path.basename(filepath)[:12], 4)
    return True, {"status": "saved", "filename": os.path.basename(filepath), "size": size}


def _handle_record_button():
    def worker():
        if _recording["active"]:
            ok, payload = _stop_recording_internal()
            if not ok:
                log.warning("REC-painike: %s", payload.get("error"))
        else:
            ok, payload = _start_recording_internal()
            if not ok:
                log.warning("REC-painike: %s", payload.get("error"))
        update_oled_status()

    threading.Thread(target=worker, daemon=True).start()


def _handle_wifi_button():
    def worker():
        current = _wifi["enabled"]
        if current is None:
            current = detect_wifi_enabled()
        target = not bool(current)
        ok, message = set_wifi_enabled(target)
        state_text = "ON" if target else "OFF"
        if ok:
            _set_display_notice(f"WiFi {state_text}", 3)
            log.info("Wi-Fi asetettu: %s", state_text)
        else:
            _set_display_notice("WiFi toggle fail", 3)
            log.warning("Wi-Fi vaihto epaonnistui: %s", message)
        update_oled_status()

    threading.Thread(target=worker, daemon=True).start()


def _handle_status_button():
    ip = _get_local_ip()
    file_count = len([f for f in os.listdir(RECORDINGS_DIR) if f.lower().endswith(".wav")])
    _set_display_notice(f"IP {ip} F:{file_count}", 5)
    update_oled_status()


def init_hardware_controls():
    _wifi["enabled"] = detect_wifi_enabled()
    _configure_gpio_backend()
    _init_oled()

    if Button is None:
        log.warning("gpiozero-kirjastoa ei löydy, painikkeita ei alustettu")
    else:
        try:
            rec_btn = Button(GPIO_RECORD_PIN, pull_up=True, bounce_time=0.2)
            wifi_btn = Button(GPIO_WIFI_PIN, pull_up=True, bounce_time=0.2)
            status_btn = Button(GPIO_STATUS_PIN, pull_up=True, bounce_time=0.2)
            rec_btn.when_pressed = _handle_record_button
            wifi_btn.when_pressed = _handle_wifi_button
            status_btn.when_pressed = _handle_status_button
            _buttons.extend([rec_btn, wifi_btn, status_btn])
            log.info("GPIO-painikkeet alustettu (GPIO %s, %s, %s)", GPIO_RECORD_PIN, GPIO_WIFI_PIN, GPIO_STATUS_PIN)
        except Exception as exc:
            log.warning("GPIO-alustus epaonnistui: %s", exc)
            log.warning("Varmista lgpio asennus ja kayttooikeudet /dev/gpiochip* laitteisiin")

    threading.Thread(target=_display_loop, daemon=True).start()
    update_oled_status()


# ── Routes ───────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    elapsed = 0
    if _recording["active"] and _recording["start_time"]:
        elapsed = time.time() - _recording["start_time"]
    return jsonify(
        {
            "is_recording": _recording["active"],
            "elapsed": elapsed,
        }
    )


@app.route("/api/record/start", methods=["POST"])
def start_recording():
    ok, payload = _start_recording_internal()
    update_oled_status()
    if not ok:
        return jsonify(payload), 400
    return jsonify(payload)


@app.route("/api/record/stop", methods=["POST"])
def stop_recording():
    ok, payload = _stop_recording_internal()
    update_oled_status()
    if not ok:
        return jsonify(payload), 400
    return jsonify(payload)


@app.route("/api/recordings")
def list_recordings():
    files = []
    for f in sorted(os.listdir(RECORDINGS_DIR), reverse=True):
        if not f.lower().endswith(".wav"):
            continue
        fp = os.path.join(RECORDINGS_DIR, f)
        stat = os.stat(fp)
        files.append(
            {
                "filename": f,
                "size": stat.st_size,
                "created": stat.st_mtime,
            }
        )
    return jsonify(files)


@app.route("/api/recordings/download-zip", methods=["POST"])
def download_zip():
    data = request.get_json(silent=True) or {}
    filenames = data.get("filenames", [])
    if not filenames or not isinstance(filenames, list):
        return jsonify({"error": "Tiedostonimiä ei annettu"}), 400

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in filenames:
            filepath = _safe_filepath(name)
            if filepath and os.path.isfile(filepath):
                zf.write(filepath, os.path.basename(filepath))
    buf.seek(0)
    return send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name="recordings.zip",
    )


@app.route("/api/recordings/<filename>")
def get_recording(filename):
    filepath = _safe_filepath(filename)
    if not filepath or not os.path.isfile(filepath):
        return jsonify({"error": "Tiedostoa ei löydy"}), 404
    return send_file(filepath, mimetype="audio/wav")


@app.route("/api/recordings/<filename>", methods=["DELETE"])
def delete_recording(filename):
    filepath = _safe_filepath(filename)
    if not filepath or not os.path.isfile(filepath):
        return jsonify({"error": "Tiedostoa ei löydy"}), 404
    os.remove(filepath)
    log.info("Tiedosto poistettu: %s", filename)
    return jsonify({"status": "deleted"})


@app.route("/api/recordings/delete-many", methods=["POST"])
def delete_many_recordings():
    data = request.get_json(silent=True) or {}
    filenames = data.get("filenames", [])
    if not filenames or not isinstance(filenames, list):
        return jsonify({"error": "Tiedostonimiä ei annettu"}), 400
    deleted = []
    for name in filenames:
        filepath = _safe_filepath(name)
        if filepath and os.path.isfile(filepath):
            os.remove(filepath)
            log.info("Tiedosto poistettu: %s", name)
            deleted.append(name)
    return jsonify({"deleted": deleted})


@app.route("/api/recordings/<filename>/rename", methods=["POST"])
def rename_recording(filename):
    filepath = _safe_filepath(filename)
    if not filepath or not os.path.isfile(filepath):
        return jsonify({"error": "Tiedostoa ei löydy"}), 404

    data = request.get_json(silent=True) or {}
    new_name = data.get("name", "").strip()
    if not new_name:
        return jsonify({"error": "Uusi nimi puuttuu"}), 400

    if not new_name.lower().endswith(".wav"):
        new_name += ".wav"

    new_path = _safe_filepath(new_name)
    if not new_path:
        return jsonify({"error": "Virheellinen tiedostonimi"}), 400
    if os.path.exists(new_path):
        return jsonify({"error": "Samanniminen tiedosto on jo olemassa"}), 409

    os.rename(filepath, new_path)
    log.info("Tiedosto nimetty uudelleen: %s -> %s", filename, os.path.basename(new_path))
    return jsonify({"status": "renamed", "filename": os.path.basename(new_path)})


def _format_size(b):
    if b < 1024:
        return f"{b} B"
    if b < 1024 * 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b / (1024 * 1024):.1f} MB"


if __name__ == "__main__":
    # Verify arecord is available
    try:
        result = subprocess.run(["arecord", "-L"], capture_output=True, text=True)
        if result.returncode == 0:
            log.info("ALSA-laite: %s", ALSA_DEVICE)
        else:
            log.warning("arecord -L palautti virheen: %s", result.stderr.strip())
    except FileNotFoundError:
        log.error("arecord ei löydy! Asenna: sudo apt install alsa-utils")

    init_hardware_controls()
    debug_enabled = os.environ.get("FLASK_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
    app.run(host="0.0.0.0", port=5000, debug=debug_enabled, use_reloader=debug_enabled)
