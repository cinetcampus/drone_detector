# Drone Detector

Field demo system for detecting drones from audio, built from four components that
used to live in separate repositories (their full commit history is preserved under
each subdirectory below).

## Components

- **[`drone_detect/`](drone_detect/README.md)** — the detection engine: audio capture,
  feature extraction, trained models, and the batch-testing / log-viewing tools used
  to evaluate them. Includes `rpi-lorawan` as a git submodule (LoRaWAN radio client).
- **[`detector_ui/`](detector_ui/README.md)** — web UI client that displays live detections.
- **[`Raspberry_Recorder/`](Raspberry_Recorder/README.md)** — standalone audio recorder
  app used on the Pi.
- **[`scripts/`](scripts/README.md)** — operational tooling to start/stop/inspect the demo
  (`detector-start`, `detector-ui-start`, `detector-status`, `log-viewer`, `batch-test`,
  ...) plus the `wifi-on`/`wifi-off`/`wifi-status` helpers used on the Pi.

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

`scripts/pi-sync-pull`, `pi-sync-push`, `pi-sync-status`, and `pi-git-setup.sh` sync
this single combined repo with one matching repo on a Raspberry Pi (see
[`scripts/README.md`](scripts/README.md#pi-deployment--sync-laptop---pi)). They
previously synced each of the four components as an independent repo — that only
applied to the old per-repo layout and has been superseded. They target a **new** Pi
setup: the current field Pi still runs the old four-repo layout and hasn't been
migrated, so run `pi-git-setup.sh` on a fresh Pi before using these.
