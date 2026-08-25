#!/usr/bin/env python3
"""Analyze drone-detect log files.

Parses detector.py's log output (one or more files, each possibly containing
several restarted sessions) and reports, per session:
  - duration, chunk count, drone-chunk rate, score distribution
  - a coarse ASCII timeline of anomaly score over time
  - the detection events actually sent, and gaps between them
  - a replay of the trigger logic against the recorded per-chunk scores,
    so alternate window/threshold/cooldown settings (e.g. the new
    peak-hold logic) can be evaluated against a real flight without
    re-flying the drone.

Standard library only, so it can run on a log pulled off the Pi without
needing the project venv.

Usage:
    ./analyze_logs.py logs/drone-detect.log
    ./analyze_logs.py logs/*.log --cooldown 8
    ./analyze_logs.py logs/drone-detect.log --no-replay
"""
from __future__ import annotations

import argparse
import re
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
LIVE_RE = re.compile(
    r"LIVE chunk=(\d+) score=([\d.]+) prediction=(\w+)(?: rms=([\d.]+) peak=([\d.]+))?"
)
EVENT_RE = re.compile(r"Detection event sent \(window=(\d+)/(\d+), accuracy=([\d.]+)%\)")
STARTED_RE = re.compile(r"Audio anomaly detector started\.")
TRIGGER_RE = re.compile(
    r"Detection trigger: (\d+)/(\d+) chunks, min_accuracy=([\d.]+)%, cooldown=([\d.]+)s"
)
DEVICE_RE = re.compile(r"Device: (\S+), Rate: (\d+) Hz")
OVERFLOW_RE = re.compile(r"\[WARNING\] Audio status: (.+)")

DATE_FMT = "%Y-%m-%d %H:%M:%S,%f"

BLOCKS = " ▁▂▃▄▅▆▇█"


@dataclass
class Chunk:
    ts: datetime
    idx: int
    score: float
    is_anomaly: bool
    rms: float | None
    peak: float | None


@dataclass
class Event:
    ts: datetime
    window: int
    window_size: int
    accuracy_pct: float


@dataclass
class Session:
    label: str
    start_ts: datetime | None = None
    min_anomalies: int | None = None
    window_chunks: int | None = None
    min_accuracy_pct: float | None = None
    cooldown_s: float | None = None
    sample_rate: int | None = None
    chunks: list[Chunk] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    audio_warnings: list[tuple[datetime, str]] = field(default_factory=list)


def parse_log(path: Path) -> list[Session]:
    sessions: list[Session] = []
    current: Session | None = None
    with path.open(errors="replace") as f:
        for line in f:
            m = TS_RE.match(line)
            if not m:
                continue
            ts = datetime.strptime(m.group(1), DATE_FMT)

            if STARTED_RE.search(line):
                current = Session(label=f"{path.name} @ {ts:%Y-%m-%d %H:%M:%S}", start_ts=ts)
                sessions.append(current)
                continue
            if current is None:
                current = Session(label=f"{path.name} @ {ts:%Y-%m-%d %H:%M:%S}", start_ts=ts)
                sessions.append(current)

            if m2 := TRIGGER_RE.search(line):
                current.min_anomalies = int(m2.group(1))
                current.window_chunks = int(m2.group(2))
                current.min_accuracy_pct = float(m2.group(3))
                current.cooldown_s = float(m2.group(4))
                continue
            if m2 := DEVICE_RE.search(line):
                current.sample_rate = int(m2.group(2))
                continue
            if m2 := LIVE_RE.search(line):
                current.chunks.append(
                    Chunk(
                        ts=ts,
                        idx=int(m2.group(1)),
                        score=float(m2.group(2)),
                        is_anomaly=m2.group(3) == "drone",
                        rms=float(m2.group(4)) if m2.group(4) else None,
                        peak=float(m2.group(5)) if m2.group(5) else None,
                    )
                )
                continue
            if m2 := EVENT_RE.search(line):
                current.events.append(
                    Event(ts, int(m2.group(1)), int(m2.group(2)), float(m2.group(3)))
                )
                continue
            if m2 := OVERFLOW_RE.search(line):
                current.audio_warnings.append((ts, m2.group(1)))
                continue
    return sessions


def simulate(
    chunks: list[Chunk],
    window_chunks: int,
    min_anomalies: int,
    min_accuracy_pct: float,
    cooldown_s: float,
    peak_hold: bool,
) -> list[Event]:
    """Replay detector.py's trigger logic over recorded chunks.

    Mirrors detector.py's main loop (accuracy = mean score of the anomalous
    chunks in the sliding window; optional peak-hold reporting, see
    detector.py's peak_accuracy_since_send).
    """
    min_accuracy = min_accuracy_pct / 100.0
    anomaly_window: deque[int] = deque(maxlen=window_chunks)
    score_window: deque[float] = deque(maxlen=window_chunks)
    events: list[Event] = []
    last_detection_ts: datetime | None = None
    peak_accuracy = 0.0

    for c in chunks:
        anomaly_window.append(1 if c.is_anomaly else 0)
        score_window.append(c.score)
        anomaly_scores = [s for a, s in zip(anomaly_window, score_window) if a]
        accuracy = sum(anomaly_scores) / len(anomaly_scores) if anomaly_scores else 0.0

        trigger = sum(anomaly_window) >= min_anomalies and accuracy >= min_accuracy
        if peak_hold:
            peak_accuracy = max(peak_accuracy, accuracy) if trigger else 0.0

        cooldown_ok = (
            last_detection_ts is None
            or (c.ts - last_detection_ts).total_seconds() >= cooldown_s
        )
        if trigger and cooldown_ok:
            reported = peak_accuracy if peak_hold else accuracy
            events.append(Event(c.ts, sum(anomaly_window), window_chunks, reported * 100))
            last_detection_ts = c.ts
            if peak_hold:
                peak_accuracy = 0.0

    return events


def fmt_td(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:04.1f}s"


def ascii_timeline(chunks: list[Chunk], bins: int = 80) -> str:
    if not chunks:
        return ""
    t0, t1 = chunks[0].ts, chunks[-1].ts
    span = (t1 - t0).total_seconds() or 1.0
    buckets = [[] for _ in range(bins)]
    for c in chunks:
        i = min(bins - 1, int((c.ts - t0).total_seconds() / span * bins))
        buckets[i].append(c.score)
    maxscore = max((c.score for c in chunks), default=1.0) or 1.0
    chars = []
    for b in buckets:
        if not b:
            chars.append(" ")
            continue
        level = max(b) / maxscore
        idx = min(len(BLOCKS) - 1, int(level * (len(BLOCKS) - 1)))
        chars.append(BLOCKS[idx])
    return "".join(chars)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def print_session_summary(sess: Session) -> None:
    print(f"\n=== Session: {sess.label} ===")
    if not sess.chunks:
        print("  (no chunk data)")
        return

    duration = (sess.chunks[-1].ts - sess.chunks[0].ts).total_seconds()
    scores = [c.score for c in sess.chunks]
    drone_chunks = [c for c in sess.chunks if c.is_anomaly]
    cfg_bits = []
    if sess.window_chunks is not None:
        cfg_bits.append(f"{sess.min_anomalies}/{sess.window_chunks} chunks")
    if sess.min_accuracy_pct is not None:
        cfg_bits.append(f"min_accuracy={sess.min_accuracy_pct:.0f}%")
    if sess.cooldown_s is not None:
        cfg_bits.append(f"cooldown={sess.cooldown_s:.1f}s")
    if sess.sample_rate:
        cfg_bits.append(f"{sess.sample_rate} Hz")

    print(f"  Config: {', '.join(cfg_bits) if cfg_bits else 'unknown'}")
    print(f"  Duration: {fmt_td(duration)}  ({len(sess.chunks)} chunks)")
    print(
        f"  Drone-flagged chunks: {len(drone_chunks)}/{len(sess.chunks)} "
        f"({100 * len(drone_chunks) / len(sess.chunks):.1f}%)"
    )
    print(
        f"  Score: min={min(scores):.3f} mean={statistics.fmean(scores):.3f} "
        f"median={statistics.median(scores):.3f} p90={percentile(scores, 0.90):.3f} "
        f"p95={percentile(scores, 0.95):.3f} max={max(scores):.3f}"
    )
    if sess.audio_warnings:
        print(f"  Audio warnings: {len(sess.audio_warnings)} (dropped/overflowed input)")

    print(f"  Score timeline (low->high over session span):")
    print(f"    {ascii_timeline(sess.chunks)}")

    print(f"  Detection events actually sent: {len(sess.events)}")
    if sess.events:
        gaps = [
            (b.ts - a.ts).total_seconds() for a, b in zip(sess.events, sess.events[1:])
        ]
        accs = [e.accuracy_pct for e in sess.events]
        print(
            f"    accuracy: min={min(accs):.1f}% mean={statistics.fmean(accs):.1f}% "
            f"max={max(accs):.1f}%"
        )
        if gaps:
            print(
                f"    gaps between events: min={fmt_td(min(gaps))} "
                f"mean={fmt_td(statistics.fmean(gaps))} max={fmt_td(max(gaps))}"
            )
        for e in sess.events:
            print(f"    {e.ts:%H:%M:%S}  window={e.window}/{e.window_size}  accuracy={e.accuracy_pct:.1f}%")


def print_replay_comparison(
    sess: Session,
    window_chunks: int,
    min_anomalies: int,
    min_accuracy_pct: float,
    cooldown_s: float,
) -> None:
    if not sess.chunks:
        return
    baseline = None
    if sess.window_chunks is not None:
        baseline = simulate(
            sess.chunks,
            sess.window_chunks,
            sess.min_anomalies,
            sess.min_accuracy_pct,
            sess.cooldown_s,
            peak_hold=False,
        )
    candidate = simulate(
        sess.chunks, window_chunks, min_anomalies, min_accuracy_pct, cooldown_s, peak_hold=True
    )

    print(f"  Replay comparison:")
    if baseline is not None:
        match = "" if len(baseline) else " (no events)"
        note = " [validates parser]" if len(baseline) == len(sess.events) else " [differs from logged events - check log for gaps/rotation]"
        print(
            f"    as originally run  ({sess.min_anomalies}/{sess.window_chunks}, "
            f"min_accuracy={sess.min_accuracy_pct:.0f}%, cooldown={sess.cooldown_s:.0f}s, no peak-hold): "
            f"{len(baseline)} events{match}{note}"
        )
    print(
        f"    candidate settings ({min_anomalies}/{window_chunks}, "
        f"min_accuracy={min_accuracy_pct:.0f}%, cooldown={cooldown_s:.0f}s, peak-hold): "
        f"{len(candidate)} events"
    )
    if candidate:
        accs = [e.accuracy_pct for e in candidate]
        print(
            f"      accuracy: min={min(accs):.1f}% mean={statistics.fmean(accs):.1f}% max={max(accs):.1f}%"
        )
        for e in candidate:
            print(f"      {e.ts:%H:%M:%S}  window={e.window}/{e.window_size}  accuracy={e.accuracy_pct:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logfiles", nargs="+", type=Path, help="detector.py log file(s) to analyze")
    parser.add_argument("--window", type=int, default=5, help="candidate DETECTION_WINDOW_CHUNKS (default: 5)")
    parser.add_argument("--min-anomalies", type=int, default=2, help="candidate DETECTION_MIN_ANOMALIES_IN_WINDOW (default: 2)")
    parser.add_argument("--min-accuracy", type=float, default=60.0, help="candidate DETECTION_MIN_ACCURACY %% (default: 60)")
    parser.add_argument("--cooldown", type=float, default=8.0, help="candidate DETECTION_COOLDOWN_SECONDS (default: 8)")
    parser.add_argument("--no-replay", action="store_true", help="skip the replay/what-if comparison")
    args = parser.parse_args()

    any_sessions = False
    for path in args.logfiles:
        if not path.exists():
            print(f"skip: {path} not found")
            continue
        for sess in parse_log(path):
            if not sess.chunks and not sess.events:
                continue
            any_sessions = True
            print_session_summary(sess)
            if not args.no_replay:
                print_replay_comparison(
                    sess, args.window, args.min_anomalies, args.min_accuracy, args.cooldown
                )

    if not any_sessions:
        print("No sessions with chunk data found in the given log file(s).")


if __name__ == "__main__":
    main()
