# Detector UI

FastAPI web app that shows live detections on a map. Part of the
[Drone_Detector](../README.md) project — receives events over HTTP from
`drone_detect` (or any other detector) and pushes them to connected browsers
over Server-Sent Events.

## Install

```bash
./install.sh
```

Creates a `.venv` and installs `requirements.txt` (`fastapi`, `uvicorn`,
`requests`). `scripts/detector-ui-start` assumes this venv is named `.venv`.

## Run

```bash
source .venv/bin/activate
uvicorn web:app --reload
```

Or via the project-level wrapper: `scripts/detector-ui-start` (add
`-demo`/set `DETECTOR_SHOW_STOP=0` to hide the stop-detector button during a
live demo — see `scripts/detector-ui-start-demo`).

## API (`web.py`)

- `POST /api/detection` — report a detection event (`accuracy`, `label`,
  `detector`, optional `latitude`/`longitude`). Broadcast to all connected
  clients.
- `POST /api/confidence` — report the current live confidence/score
  (`confidence`, `is_drone`, optional `min_accuracy`) for the UI's gauge.
- `GET /api/events` — Server-Sent Events stream browsers subscribe to (with
  a 15s keep-alive heartbeat).
- `GET /api/config` — feature flags for the frontend (`debug`, `showStopDetector`).
- `POST /api/debug/detection` — fires a randomized fake detection, for UI testing.
- `GET /api/detector/status`, `POST /api/detector/start`, `POST /api/detector/stop`
  — checks/controls the `drone-detect` tmux session via `scripts/detector-start`
  and `scripts/detector-stop`.
- `GET /tiles/{z}/{x}/{filename}` — offline-friendly OpenStreetMap tile proxy;
  caches fetched tiles under `static/tiles/` so the map still works without
  internet on a later run.

Both `POST` endpoints accept an optional `X-Detector-Token` header, checked
against the `DETECTOR_TOKEN` environment variable when set.

## Environment variables

- `DETECTOR_TOKEN` — if set, required on `/api/detection` and `/api/confidence`.
- `DETECTOR_DEBUG` — `1` enables the debug endpoint/UI affordances.
- `DETECTOR_SHOW_STOP` — `0` hides the stop-detector button in the UI.

## Sending detections from a model script

- `detector_client.py` — the helper `drone_detect/detector.py` actually
  imports (`send_detection()`); reads `DETECTOR_URL`/`CONFIDENCE_URL`/
  `DETECTOR_TOKEN` from the environment, defaulting to `http://127.0.0.1:8000`.
- `model_example.py` — minimal standalone example (`notify_detection()`) for
  wiring up a different model script, plus a `__main__` demo that sends
  randomized detections every 10 seconds.

## Static frontend

`static/` is served directly at `/` (Leaflet-based map + event log). `static/images`
and `static/tiles` are gitignored — tiles are populated at runtime by the tile
proxy above.
