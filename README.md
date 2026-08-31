# Drone Detector

Field demo system for detecting drones from audio, built from four components that
used to live in separate repositories (their full commit history is preserved under
each subdirectory below).

## Components

- **`drone_detect/`** — the detection engine: audio capture, feature extraction,
  trained models, and the batch-testing / log-viewing tools used to evaluate them.
  Includes `rpi-lorawan` as a git submodule (LoRaWAN radio client).
- **`detector_ui/`** — web UI client that displays live detections.
- **`Raspberry_Recorder/`** — standalone audio recorder app used on the Pi.
- **`scripts/`** — operational tooling to start/stop/inspect the demo (`detector-start`,
  `detector-ui-start`, `detector-status`, `log-viewer`, `batch-test`, ...) plus the
  `wifi-on`/`wifi-off`/`wifi-status` helpers used on the Pi.

## Getting started

Each component has its own `requirements.txt` / `install.sh` — see the subdirectory.
`drone_detect` also needs its submodule:

```bash
git submodule update --init --recursive
```

Once each component's dependencies are installed (typically into a `.venv/` inside
that component's directory), use the wrappers in `scripts/` to run the demo, e.g.
`scripts/detector-all-start` / `scripts/detector-all-stop` / `scripts/detector-status`.

## Pi-sync scripts

`scripts/pi-sync-pull`, `pi-sync-push`, `pi-sync-status`, and `pi-git-setup.sh` were
written for the previous layout, where each of the four components above was an
independent repo with its own `pi` git remote kept in sync with a matching repo on the
Raspberry Pi. They haven't been updated for the merged layout yet — adapting them to
sync this single repo with one repo on the Pi is a natural next step if development
continues.
