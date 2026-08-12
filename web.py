import asyncio
import json
import os
import random
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()
event_queue: asyncio.Queue = asyncio.Queue()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEBUG = os.getenv("DETECTOR_DEBUG", "0") == "1"
DETECTOR_TOKEN = os.getenv("DETECTOR_TOKEN", "")


class Detection(BaseModel):
    accuracy: float
    label: str = "Detection"
    detector: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


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


@app.get("/api/config")
async def config():
    return {"debug": DEBUG}


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


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


# Tarjoile staattinen web-sivu static-hakemistosta.
app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="web")