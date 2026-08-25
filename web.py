import asyncio
import json
import os
import random
import subprocess
import urllib.request
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()
event_queue: asyncio.Queue = asyncio.Queue()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "..", "scripts")
DEBUG = os.getenv("DETECTOR_DEBUG", "0") == "1"
SHOW_STOP_DETECTOR = os.getenv("DETECTOR_SHOW_STOP", "1") == "1"
DETECTOR_TOKEN = os.getenv("DETECTOR_TOKEN", "")


class Detection(BaseModel):
    accuracy: float
    label: str = "Detection"
    detector: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class Confidence(BaseModel):
    confidence: float   # 0–100, mean anomaly score of the current window
    is_drone: bool      # true when window meets anomaly-count threshold (ignores cooldown)


async def send_detection(detection: Detection):
    """Kutsu tätä, kun tekoälymalli tekee tunnistuksen."""
    await event_queue.put({
        "type": "detection",
        "accuracy": detection.accuracy,
        "label": detection.label,
        "detector": detection.detector or detection.label,
        "latitude": detection.latitude,
        "longitude": detection.longitude,
    })


@app.post("/api/detection")
async def post_detection(
    detection: Detection,
    x_detector_token: Optional[str] = Header(default=None),
):
    if DETECTOR_TOKEN and x_detector_token != DETECTOR_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid detector token")
    await send_detection(detection)
    return {"status": "queued"}


@app.post("/api/confidence")
async def post_confidence(
    confidence: Confidence,
    x_detector_token: Optional[str] = Header(default=None),
):
    if DETECTOR_TOKEN and x_detector_token != DETECTOR_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid detector token")
    await event_queue.put({
        "type": "confidence",
        "confidence": confidence.confidence,
        "is_drone": confidence.is_drone,
    })
    return {"status": "queued"}


@app.get("/api/config")
async def config():
    return {
        "debug": DEBUG,
        "showStopDetector": SHOW_STOP_DETECTOR,
    }


@app.post("/api/debug/detection")
async def debug_detection():
    await send_detection(Detection(
        accuracy=round(random.uniform(84.0, 99.9), 1),
        label=random.choice(["person", "dog", "car", "bird", "drone"]),
        detector="debug",
    ))
    return {"status": "queued"}


@app.get("/api/events")
async def events(request: Request):
    async def event_stream():
        yield "event: connected\ndata: {}\n\n"
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=15)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"  # heartbeat, keeps the connection open
                continue
            yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/detector/status")
async def detector_status():
    result = subprocess.run(["tmux", "has-session", "-t", "drone-detect"], capture_output=True)
    return {"running": result.returncode == 0}


@app.post("/api/detector/start")
async def detector_start():
    result = subprocess.run(
        [os.path.join(SCRIPTS_DIR, "detector-start")], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=(result.stdout + result.stderr).strip())
    return {"status": "started"}


@app.post("/api/detector/stop")
async def detector_stop():
    result = subprocess.run(
        [os.path.join(SCRIPTS_DIR, "detector-stop")], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=(result.stdout + result.stderr).strip())
    return {"status": "stopped"}

@app.get("/tiles/{z}/{x}/{filename}")
async def tile_proxy(z: int, x: int, filename: str):
    if not filename.endswith(".png"):
        raise HTTPException(status_code=400)
    y = filename[:-4]
    tile_path = os.path.join(BASE_DIR, "static", "tiles", str(z), str(x), filename)
    if os.path.exists(tile_path):
        return FileResponse(tile_path, media_type="image/png")
    try:
        url = f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        req = urllib.request.Request(url, headers={"User-Agent": "DroneDetector/1.0 (tile proxy)"})
        data = await asyncio.to_thread(lambda: urllib.request.urlopen(req).read())
        os.makedirs(os.path.dirname(tile_path), exist_ok=True)
        with open(tile_path, "wb") as f:
            f.write(data)
        return Response(content=data, media_type="image/png")
    except Exception:
        raise HTTPException(status_code=503, detail="Tile unavailable offline")

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


# Tarjoile staattinen web-sivu static-hakemistosta.
app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="web")