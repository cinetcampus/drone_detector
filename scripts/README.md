# Scripts

Operational tooling for running and managing the [Drone_Detector](../README.md)
demo. Everything here locates its siblings by relative path
(`$(dirname "$0")/..`), so it works regardless of where the repo is checked out.

## Demo control

- `detector-start` / `detector-stop` — start/stop `drone_detect/detector.py`
  in a `drone-detect` tmux session; logs go to `logs/drone-detect_NNNN.log`
  (symlinked as `logs/drone-detect-latest.log`).
- `detector-ui-start` / `detector-ui-stop` — same, for `detector_ui` (runs
  `uvicorn web:app` on port 8000) in a `detector-ui` tmux session.
- `detector-ui-start-demo` — like `detector-ui-start`, but hides the stop-detector
  button in the UI (`DETECTOR_SHOW_STOP=0`) for a live demo.
- `detector-all-start` / `detector-all-stop` — both of the above together.
- `detector-status` — shows whether each tmux session is running and tails
  its latest log.
- `detector-test` — runs `drone_detect/test_file.py` against
  `test_samples/positive_test.wav` in the running `detector-ui` session's
  environment, prints the result.

## Analysis tools

- `batch-test` — opens `drone_detect/batch_test.py`'s browser UI for
  re-runnable batch testing of the model against a directory of recordings.
- `log-viewer` — opens `drone_detect/log_viewer.py`'s browser UI over session
  logs in `logs/`.

## WiFi (run on the Pi)

- `wifi-on` / `wifi-off` — enable/disable the WiFi radio (needs `sudo`; see
  each script's header comment for why). Same effect as the hardware WiFi
  button in `Raspberry_Recorder/app.py`.
- `wifi-status` — reports whether WiFi is currently enabled.

## Pi deployment / sync (laptop \<-> Pi)

These sync this single combined repo with one matching repo on a Raspberry
Pi. Run `pi-git-setup.sh` once on the Pi before using the rest.

- `pi-git-setup.sh` — run once, on the Pi: creates one bare repo + checkout
  for the whole project under `~/repos/drone_detector.git` /
  `~/workspace/drone_detector`, with a post-receive hook that also
  initializes the `rpi-lorawan` submodule.
- `pi-sync-push` — push committed local changes to the Pi.
- `pi-sync-pull` — bring changes made directly on the Pi back into the
  laptop's history (snapshotting any dirty Pi working tree first), then
  pushes the reconciled result back.
- `pi-sync-status` — shows whether the laptop and Pi agree on commits and
  whether the Pi has uncommitted changes.
- `pi-sync-logs` — rsyncs session log files down from the Pi's `logs/` into
  the laptop's `logs/`.
