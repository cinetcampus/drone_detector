import logging
import os
import signal
import subprocess
import time
from datetime import datetime

import numpy as np
import scipy.io.wavfile as wavfile
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

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

# Recording state
_recording = {
    "active": False,
    "process": None,
    "filepath": None,
    "start_time": None,
}


def _safe_filepath(filename):
    """Return a safe absolute path inside RECORDINGS_DIR, or None."""
    name = secure_filename(filename)
    if not name:
        return None
    path = os.path.join(RECORDINGS_DIR, name)
    if os.path.commonpath([RECORDINGS_DIR, os.path.realpath(path)]) != RECORDINGS_DIR:
        return None
    return path


# ── Routes ───────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    elapsed = 0
    if _recording["active"] and _recording["start_time"]:
        elapsed = time.time() - _recording["start_time"]
    return jsonify({"is_recording": _recording["active"], "elapsed": elapsed})


@app.route("/api/record/start", methods=["POST"])
def start_recording():
    if _recording["active"]:
        return jsonify({"error": "Äänitys on jo käynnissä"}), 400

    filename = datetime.now().strftime("recording_%Y%m%d_%H%M%S.wav")
    filepath = os.path.join(RECORDINGS_DIR, filename)

    log.info("Äänitys alkaa — laite: %s, polku: %s, tiedosto: %s", ALSA_DEVICE, filepath, filename)

    proc = subprocess.Popen(
        [
            "arecord",
            "-D", ALSA_DEVICE,
            "-f", "S16_LE",
            "-r", str(SAMPLE_RATE),
            "-c", str(CHANNELS),
            "-t", "wav",
            filepath,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    _recording["active"] = True
    _recording["process"] = proc
    _recording["filepath"] = filepath
    _recording["start_time"] = time.time()

    return jsonify({"status": "recording"})


@app.route("/api/record/stop", methods=["POST"])
def stop_recording():
    if not _recording["active"]:
        return jsonify({"error": "Äänitys ei ole käynnissä"}), 400

    proc = _recording["process"]
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        print("Recording stopped")
        proc.wait(timeout=5)
    else: 
      print("_recording=", proc.poll())
    
    _recording["active"] = False
    filepath = _recording["filepath"]
    _recording["process"] = None
    _recording["filepath"] = None

    if not filepath or not os.path.isfile(filepath):
        return jsonify({"error": "Äänitiedostoa ei löydy"}), 500

    duration = 0
    try:
        rate, data = wavfile.read(filepath)
        duration = len(data) / rate
        log.info("Äänitys tallennettu: %s (%.1f s, %s)",
                 os.path.basename(filepath), duration,
                 _format_size(os.path.getsize(filepath)))
    except Exception as e:
        log.error("WAV-tiedoston lukuvirhe: %s", e)

    size = os.path.getsize(filepath)
    return jsonify({"status": "saved", "filename": os.path.basename(filepath), "size": size})


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
        log.info("ALSA-laite: %s", ALSA_DEVICE)
    except FileNotFoundError:
        log.error("arecord ei löydy! Asenna: sudo apt install alsa-utils")
    app.run(host="0.0.0.0", port=5000, debug=True)
