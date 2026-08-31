#!/usr/bin/env python3
"""Batch-test recordings against the trained model, browsable and re-runnable.

Start the server, then from the browser: pick a directory (searched
recursively, grouped by subfolder e.g. positive/drone_a vs positive/drone_b
vs negative), hit Run, and watch the report appear once the batch finishes.
Every finished run is persisted to --storage-dir (default: ./batch_runs) and
reloaded on the next startup, so previous batches show up in the history
list without re-running the model.

Report rendering (grouped listing, score chart, replay-with-different-
settings form) is shared with log_viewer.py so an offline batch and a real
flight log look identical.

Usage:
    ./batch_test.py                    # just serve; run batches from the browser
    ./batch_test.py /path/to/recordings  # also kick off one run immediately
    ./batch_test.py --port 9000
"""
from __future__ import annotations

import argparse
import html
import json
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import gcd
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import config
from analyze_logs import Chunk, Event, Session, fmt_td, simulate
from log_viewer import DEFAULT_CANDIDATE, PAGE_CSS, render_detail
from model_bundle_detector import BundleModelDetector

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_STORAGE_DIR = PROJECT_ROOT / "batch_runs"
# Sessions built from an offline file have no real wall-clock; analyze_logs'
# duration/cooldown math only ever looks at deltas, so an arbitrary shared
# epoch is enough, and it's what chunk/event timestamps get serialized
# relative to.
EPOCH = datetime(2000, 1, 1)

GROUP_CSS = """
  details.group { margin: 0.6rem 0; }
  details.group > summary { cursor: pointer; font-size: 1.05rem; padding: 0.3rem 0; }
  details.group > summary:hover { color: #2a6f97; }
  details.group table { margin-top: 0.5rem; }
  .group-body { margin: 0.4rem 0 0.4rem 1rem; padding-left: 0.9rem;
                border-left: 2px solid #e5e5e5; }
  form.run-form { margin: 1rem 0; padding: 0.75rem 1rem; background: #f6f6f6;
        border: 1px solid #e5e5e5; border-radius: 6px; }
  form.run-form .form-title { margin-bottom: 0.6rem; }
  form.run-form .form-row { display: flex; align-items: center; gap: 0.6rem; margin: 0.5rem 0; }
  form.run-form .form-row label { flex: 0 0 5rem; font-size: 0.9rem; color: #444; }
  form.run-form input[type=text] { width: 26rem; max-width: 70%; }
  form.run-form button { margin-top: 0.5rem; }
  .error { color: #b00020; }
  .status-running { color: #a15c00; }
  .status-failed { color: #b00020; }
  .breadcrumb { font-size: 0.95rem; }
  .breadcrumb a { margin: 0 0.15rem; }
  ul.browse-list { list-style: none; padding: 0; margin: 0.75rem 0; }
  ul.browse-list li { padding: 3px 0; }
  ul.browse-list a { text-decoration: none; }
  ul.browse-list a:hover { text-decoration: underline; }
  .run-actions { display: flex; align-items: center; gap: 1.25rem; flex-wrap: wrap;
                 margin: 0.5rem 0 1rem; }
  .rename-form { display: flex; align-items: center; gap: 0.4rem; }
  .rename-form label { font-size: 0.9rem; color: #444; }
  .rename-form input[type=text] { width: 16rem; max-width: 60vw; }
  .delete-link { color: #b00020; }
  button.danger { background: #b00020; color: #fff; border: none; padding: 0.5rem 1rem;
                  border-radius: 4px; cursor: pointer; font-size: 0.9rem; }
  button.danger:hover { background: #8e001a; }
"""


def bt_page(title: str, body: str, refresh: bool = False) -> bytes:
    refresh_tag = '<meta http-equiv="refresh" content="3">' if refresh else ""
    return f"""<!doctype html>
<html><head><meta charset="utf-8">{refresh_tag}<title>{html.escape(title)}</title>
<style>{PAGE_CSS}{GROUP_CSS}</style></head>
<body>{body}</body></html>""".encode("utf-8")


@dataclass
class Recording:
    id: int
    path: Path
    rel_path: str
    session: Session


@dataclass
class DirGroup:
    name: str
    subdirs: dict[str, "DirGroup"]
    recordings: list[Recording]


@dataclass
class BatchRun:
    run_id: str
    directory: str
    started_at: datetime
    name: str = ""
    finished_at: datetime | None = None
    status: str = "running"  # "running" | "done" | "failed"
    error: str | None = None
    total_files: int = 0
    recordings: list[Recording] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        return self.name or self.directory


# --- directory tree grouping -------------------------------------------------


def build_tree(recordings: list[Recording]) -> DirGroup:
    root = DirGroup(name="", subdirs={}, recordings=[])
    for r in recordings:
        node = root
        for part in Path(r.rel_path).parent.parts:
            node = node.subdirs.setdefault(part, DirGroup(name=part, subdirs={}, recordings=[]))
        node.recordings.append(r)
    return root


def collect_all(node: DirGroup) -> list[Recording]:
    result = list(node.recordings)
    for sub in node.subdirs.values():
        result.extend(collect_all(sub))
    return result


def find_audio_files(root: Path) -> list[Path]:
    files = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        if any(part.startswith(".") for part in p.relative_to(root).parts):
            continue
        files.append(p)
    return sorted(files)


# --- model inference ----------------------------------------------------------


def load_detector() -> BundleModelDetector:
    model_path = Path(config.MODEL_BUNDLE_PATH)
    scaler_path = Path(config.MODEL_SCALER_PATH)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not scaler_path.is_absolute():
        scaler_path = PROJECT_ROOT / scaler_path

    detector = BundleModelDetector(model_path=model_path, scaler_path=scaler_path)
    detector.load()
    threshold_override = getattr(config, "BUNDLE_SCORE_THRESHOLD_OVERRIDE", None)
    if threshold_override is not None:
        detector.threshold = float(threshold_override)
    return detector


def run_file(detector: BundleModelDetector, path: Path, rel_path: str) -> Session:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != config.SAMPLE_RATE:
        g = gcd(config.SAMPLE_RATE, sr)
        audio = resample_poly(audio, config.SAMPLE_RATE // g, sr // g).astype(np.float32)

    chunk_samples = int(config.SAMPLE_RATE * config.CHUNK_DURATION)
    stride_samples = int(config.SAMPLE_RATE * config.DETECTION_STRIDE_SECONDS)
    window_chunks = int(getattr(config, "DETECTION_WINDOW_CHUNKS", 5))
    min_anomalies = int(getattr(config, "DETECTION_MIN_ANOMALIES_IN_WINDOW", 2))
    cooldown_s = float(getattr(config, "DETECTION_COOLDOWN_SECONDS", 30.0))
    min_accuracy_pct = float(getattr(config, "DETECTION_MIN_ACCURACY", 60.0))

    chunks: list[Chunk] = []
    pos = 0
    idx = 0
    while pos + chunk_samples <= len(audio):
        chunk = audio[pos : pos + chunk_samples]
        is_anomaly, score = detector.is_anomaly(chunk)
        idx += 1
        rms = float(np.sqrt(np.mean(chunk**2)))
        peak = float(np.max(np.abs(chunk))) if len(chunk) else 0.0
        chunks.append(
            Chunk(
                ts=EPOCH + timedelta(seconds=pos / config.SAMPLE_RATE),
                idx=idx,
                score=score,
                is_anomaly=is_anomaly,
                rms=rms,
                peak=peak,
            )
        )
        pos += stride_samples

    events = simulate(
        chunks, window_chunks, min_anomalies, min_accuracy_pct, cooldown_s, peak_hold=True
    )

    return Session(
        label=rel_path,
        start_ts=EPOCH,
        min_anomalies=min_anomalies,
        window_chunks=window_chunks,
        min_accuracy_pct=min_accuracy_pct,
        cooldown_s=cooldown_s,
        sample_rate=config.SAMPLE_RATE,
        model_threshold_pct=detector.threshold * 100.0,
        chunks=chunks,
        events=events,
    )


# --- persistence ---------------------------------------------------------------


def session_to_dict(rel_path: str, sess: Session) -> dict:
    return {
        "rel_path": rel_path,
        "min_anomalies": sess.min_anomalies,
        "window_chunks": sess.window_chunks,
        "min_accuracy_pct": sess.min_accuracy_pct,
        "cooldown_s": sess.cooldown_s,
        "sample_rate": sess.sample_rate,
        "model_threshold_pct": sess.model_threshold_pct,
        "chunks": [
            [(c.ts - EPOCH).total_seconds(), c.score, c.is_anomaly, c.rms, c.peak]
            for c in sess.chunks
        ],
        "events": [
            [(e.ts - EPOCH).total_seconds(), e.window, e.window_size, e.accuracy_pct]
            for e in sess.events
        ],
    }


def session_from_dict(d: dict) -> Session:
    chunks = [
        Chunk(
            ts=EPOCH + timedelta(seconds=t),
            idx=i + 1,
            score=score,
            is_anomaly=is_anomaly,
            rms=rms,
            peak=peak,
        )
        for i, (t, score, is_anomaly, rms, peak) in enumerate(d["chunks"])
    ]
    events = [
        Event(ts=EPOCH + timedelta(seconds=t), window=w, window_size=ws, accuracy_pct=acc)
        for (t, w, ws, acc) in d["events"]
    ]
    return Session(
        label=d["rel_path"],
        start_ts=EPOCH,
        min_anomalies=d["min_anomalies"],
        window_chunks=d["window_chunks"],
        min_accuracy_pct=d["min_accuracy_pct"],
        cooldown_s=d["cooldown_s"],
        sample_rate=d["sample_rate"],
        model_threshold_pct=d.get("model_threshold_pct"),
        chunks=chunks,
        events=events,
    )


def run_to_dict(run: BatchRun) -> dict:
    return {
        "run_id": run.run_id,
        "directory": run.directory,
        "name": run.name,
        "started_at": run.started_at.isoformat(),
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "status": run.status,
        "error": run.error,
        "total_files": run.total_files,
        "recordings": [session_to_dict(r.rel_path, r.session) for r in run.recordings],
    }


def run_from_dict(d: dict) -> BatchRun:
    recordings = [
        Recording(
            id=i,
            path=Path(d["directory"]) / rd["rel_path"],
            rel_path=rd["rel_path"],
            session=session_from_dict(rd),
        )
        for i, rd in enumerate(d["recordings"])
    ]
    return BatchRun(
        run_id=d["run_id"],
        directory=d["directory"],
        name=d.get("name", ""),
        started_at=datetime.fromisoformat(d["started_at"]),
        finished_at=datetime.fromisoformat(d["finished_at"]) if d.get("finished_at") else None,
        status=d["status"],
        error=d.get("error"),
        total_files=d.get("total_files", len(recordings)),
        recordings=recordings,
    )


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


class RunRegistry:
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._runs: dict[str, BatchRun] = {}
        for f in sorted(self.storage_dir.glob("*.json")):
            try:
                self._runs[f.stem] = run_from_dict(json.loads(f.read_text()))
            except Exception as e:
                print(f"WARNING: failed to load {f}: {e}")

    def add(self, run: BatchRun) -> None:
        with self._lock:
            self._runs[run.run_id] = run

    def get(self, run_id: str) -> BatchRun | None:
        with self._lock:
            return self._runs.get(run_id)

    def all_sorted(self) -> list[BatchRun]:
        with self._lock:
            return sorted(self._runs.values(), key=lambda r: r.started_at, reverse=True)

    def save(self, run: BatchRun) -> None:
        path = self.storage_dir / f"{run.run_id}.json"
        path.write_text(json.dumps(run_to_dict(run)))

    def delete(self, run_id: str) -> bool:
        with self._lock:
            if run_id not in self._runs:
                return False
            del self._runs[run_id]
        (self.storage_dir / f"{run_id}.json").unlink(missing_ok=True)
        return True


# --- background execution ------------------------------------------------------


def execute_run(run: BatchRun, detector: BundleModelDetector, registry: RunRegistry) -> None:
    root = Path(run.directory)
    files = find_audio_files(root)
    run.total_files = len(files)
    if not files:
        run.status = "failed"
        run.error = f"No audio files ({', '.join(sorted(AUDIO_EXTENSIONS))}) found under {root}"
        run.finished_at = datetime.now()
        return

    print(f"[{run.run_id}] {len(files)} recording(s) under {root}")
    recordings: list[Recording] = []
    start = time.monotonic()
    for i, path in enumerate(files, start=1):
        rel = path.relative_to(root).as_posix()
        try:
            session = run_file(detector, path, rel)
        except Exception as e:
            print(f"[{run.run_id}] [{i}/{len(files)}] {rel}: FAILED ({e})")
            continue
        n_chunks = len(session.chunks)
        n_drone = sum(1 for c in session.chunks if c.is_anomaly)
        pct = 100 * n_drone / n_chunks if n_chunks else 0.0
        print(
            f"[{run.run_id}] [{i}/{len(files)}] {rel}: {n_chunks} chunks, "
            f"{pct:.1f}% drone-flagged, {len(session.events)} event(s)"
        )
        recordings.append(Recording(id=len(recordings), path=path, rel_path=rel, session=session))
        run.recordings = list(recordings)  # visible to in-progress page views

    elapsed = time.monotonic() - start
    run.status = "done"
    run.finished_at = datetime.now()
    print(f"[{run.run_id}] Processed {len(recordings)}/{len(files)} recording(s) in {fmt_td(elapsed)}")
    try:
        registry.save(run)
    except Exception as e:
        print(f"[{run.run_id}] WARNING: failed to persist results: {e}")


def start_run(
    directory: str, registry: RunRegistry, detector: BundleModelDetector, name: str = ""
) -> BatchRun:
    run = BatchRun(run_id=make_run_id(), directory=directory, name=name, started_at=datetime.now())
    registry.add(run)
    threading.Thread(target=execute_run, args=(run, detector, registry), daemon=True).start()
    return run


# --- rendering -------------------------------------------------------------------


def group_summary_html(recordings: list[Recording]) -> str:
    if not recordings:
        return "<span class='muted'>(empty)</span>"
    total_chunks = sum(len(r.session.chunks) for r in recordings)
    total_drone = sum(sum(1 for c in r.session.chunks if c.is_anomaly) for r in recordings)
    total_events = sum(len(r.session.events) for r in recordings)
    pct = f"{100 * total_drone / total_chunks:.1f}%" if total_chunks else "-"
    n = len(recordings)
    return (
        f"<span class='muted'>{n} recording{'s' if n != 1 else ''}, "
        f"drone-flagged {pct}, {total_events} event(s)</span>"
    )


def render_file_table(run_id: str, recordings: list[Recording]) -> str:
    if not recordings:
        return ""
    rows = []
    for r in recordings:
        sess = r.session
        duration = "-"
        drone_pct = "-"
        if sess.chunks:
            secs = (sess.chunks[-1].ts - sess.chunks[0].ts).total_seconds() + config.CHUNK_DURATION
            duration = fmt_td(secs)
            drone_n = sum(1 for c in sess.chunks if c.is_anomaly)
            drone_pct = f"{100 * drone_n / len(sess.chunks):.1f}%"
        name = Path(r.rel_path).name
        rows.append(
            f"<tr><td><a href='/run/{run_id}/recording/{r.id}'>{html.escape(name)}</a></td>"
            f"<td>{duration}</td><td>{len(sess.chunks)}</td>"
            f"<td>{drone_pct}</td><td>{len(sess.events)}</td></tr>"
        )
    return (
        "<table><thead><tr><th>File</th><th>Duration</th>"
        "<th>Chunks</th><th>Drone-flagged</th><th>Events</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_group(run_id: str, node: DirGroup) -> str:
    parts = [render_file_table(run_id, node.recordings)]
    for name in sorted(node.subdirs):
        sub = node.subdirs[name]
        summary = group_summary_html(collect_all(sub))
        parts.append(
            "<details class='group' open>"
            f"<summary><b>{html.escape(name)}/</b> {summary}</summary>"
            f"<div class='group-body'>{render_group(run_id, sub)}</div>"
            "</details>"
        )
    return "".join(parts)


def run_form(prefill_directory: str = "", prefill_name: str = "", show_browse: bool = True) -> str:
    browse_path = prefill_directory or str(Path.home())
    browse_link = (
        f" <a href='/browse?path={quote(browse_path)}'>Browse…</a>" if show_browse else ""
    )
    return f"""
    <form class="run-form" method="post" action="/run">
      <div class="form-title"><b>Run a new batch:</b>
        <span class="muted">(searched recursively)</span></div>
      <div class="form-row">
        <label for="run-directory">Directory</label>
        <input id="run-directory" type="text" name="directory" placeholder="/path/to/recordings"
               value="{html.escape(prefill_directory)}" required>{browse_link}
      </div>
      <div class="form-row">
        <label for="run-name">Name</label>
        <input id="run-name" type="text" name="name" placeholder="optional, e.g. small drone hover test"
               value="{html.escape(prefill_name)}">
      </div>
      <button type="submit">Run</button>
    </form>
    """


def render_breadcrumb(path: Path) -> str:
    parts = path.parts
    crumbs = []
    accum = Path(parts[0])
    crumbs.append(f"<a href='/browse?path={quote(str(accum))}'>{html.escape(parts[0])}</a>")
    for part in parts[1:]:
        accum = accum / part
        crumbs.append(f"<a href='/browse?path={quote(str(accum))}'>{html.escape(part)}</a>")
    return "<span class='breadcrumb'>" + " / ".join(crumbs) + "</span>"


def render_browse(path: Path) -> bytes:
    error_html = ""
    entries: list[Path] = []
    try:
        entries = sorted(
            (p for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")),
            key=lambda p: p.name.lower(),
        )
    except PermissionError:
        error_html = "<p class='error'>Permission denied listing this directory.</p>"
    except (FileNotFoundError, NotADirectoryError):
        error_html = "<p class='error'>Directory not found.</p>"

    parent = path.parent
    up_link = (
        f"<li><a href='/browse?path={quote(str(parent))}'>.. (up)</a></li>" if parent != path else ""
    )
    rows = "".join(
        f"<li><a href='/browse?path={quote(str(p))}'>{html.escape(p.name)}/</a></li>" for p in entries
    )
    list_html = f"<ul class='browse-list'>{up_link}{rows}</ul>"

    select_form = f"""
    <form method="get" action="/">
      <input type="hidden" name="directory" value="{html.escape(str(path))}">
      <button type="submit">Select this directory</button>
    </form>
    """

    quick_links = (
        f"<a href='/browse?path={quote(str(Path.home()))}'>~ home</a> &nbsp;|&nbsp; "
        f"<a href='/browse?path=/'>/ root</a>"
    )

    body = (
        "<a class='backlink' href='/'>&larr; back</a>"
        f"<h1>Browse for a recordings directory</h1>"
        f"<p>{render_breadcrumb(path)}</p>"
        f"<p class='muted'>{quick_links}</p>"
        f"{error_html}"
        f"{select_form}"
        f"{list_html}"
    )
    return bt_page(f"Browse {path}", body)


def render_home(registry: RunRegistry, error: str | None, prefill_directory: str = "") -> bytes:
    error_html = f"<p class='error'>{html.escape(error)}</p>" if error else ""
    runs = registry.all_sorted()
    rows = []
    for run in runs:
        started = run.started_at.strftime("%Y-%m-%d %H:%M:%S")
        if run.status == "running":
            status = (
                f"<span class='status-running'>running… "
                f"({len(run.recordings)}/{run.total_files or '?'})</span>"
            )
        elif run.status == "failed":
            status = f"<span class='status-failed'>failed: {html.escape(run.error or '')}</span>"
        else:
            total_events = sum(len(r.session.events) for r in run.recordings)
            status = f"{len(run.recordings)} recording(s), {total_events} event(s)"
        directory_line = (
            f"<br><span class='muted'>{html.escape(run.directory)}</span>" if run.name else ""
        )
        rows.append(
            f"<tr><td><a href='/run/{run.run_id}'>{html.escape(run.display_name)}</a>"
            f"{directory_line}</td>"
            f"<td>{started}</td><td>{status}</td>"
            f"<td><a class='delete-link' href='/run/{run.run_id}/delete'>Delete…</a></td></tr>"
        )
    table = (
        "<table><thead><tr><th>Run</th><th>Started</th><th>Result</th><th></th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        if rows
        else "<p class='muted'>No batch runs yet -- start one above.</p>"
    )
    body = f"<h1>Drone detector batch tests</h1>{run_form(prefill_directory)}{error_html}{table}"
    return bt_page("Batch tests", body, refresh=any(r.status == "running" for r in runs))


def render_run(run: BatchRun) -> bytes:
    header = f"<a class='backlink' href='/'>&larr; all runs</a><h1>{html.escape(run.display_name)}</h1>"
    if run.name:
        header += f"<p class='muted'>{html.escape(run.directory)}</p>"
    header += f"""
    <div class="run-actions">
      <form class="rename-form" method="post" action="/run/{run.run_id}/rename">
        <label for="rename-input">Name</label>
        <input id="rename-input" type="text" name="name" value="{html.escape(run.name)}"
               placeholder="(none)">
        <button type="submit">Rename</button>
      </form>
      <a class="delete-link" href="/run/{run.run_id}/delete">Delete this run…</a>
    </div>
    """
    if run.status == "running":
        header += (
            f"<p class='status-running'>Running… "
            f"{len(run.recordings)}/{run.total_files or '?'} processed.</p>"
        )
    elif run.status == "failed":
        header += f"<p class='status-failed'>Failed: {html.escape(run.error or 'unknown error')}</p>"
    header += run_form(run.directory, run.name, show_browse=False)

    tree = build_tree(run.recordings)
    body_content = (
        render_group(run.run_id, tree) if run.recordings else "<p class='muted'>No results yet.</p>"
    )
    return bt_page(run.display_name, header + body_content, refresh=(run.status == "running"))


def render_delete_confirm(run: BatchRun) -> bytes:
    back = f"<a class='backlink' href='/run/{run.run_id}'>&larr; back</a>"
    if run.status == "running":
        body = (
            back
            + f"<h1>Delete {html.escape(run.display_name)}?</h1>"
            + "<p class='error'>This run is still in progress and can't be deleted until it finishes.</p>"
        )
        return bt_page("Delete run", body)

    body = (
        back
        + f"<h1>Delete {html.escape(run.display_name)}?</h1>"
        + f"<p class='muted'>{html.escape(run.directory)}</p>"
        + f"<p>This permanently deletes the persisted results "
        + f"({len(run.recordings)} recording(s)). This cannot be undone.</p>"
        + f"""
        <form method="post" action="/run/{run.run_id}/delete">
          <button type="submit" class="danger">Yes, delete this run</button>
        </form>
        <p><a href="/run/{run.run_id}">Cancel</a></p>
        """
    )
    return bt_page("Delete run", body)


def render_recording(run: BatchRun, entry: Recording, candidate: dict) -> bytes:
    return render_detail(entry.rel_path, entry.session, candidate, f"/run/{run.run_id}", "this run")


# --- HTTP server -----------------------------------------------------------------


def make_handler(registry: RunRegistry, detector: BundleModelDetector):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # noqa: A003 - quiet default access log
            pass

        def _send(self, body: bytes, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _redirect(self, location: str):
            self.send_response(303)
            self.send_header("Location", location)
            self.end_headers()

        def _candidate_from_qs(self, qs: dict) -> dict:
            candidate = dict(DEFAULT_CANDIDATE)
            try:
                if "window" in qs:
                    candidate["window"] = int(qs["window"][0])
                if "min_anomalies" in qs:
                    candidate["min_anomalies"] = int(qs["min_anomalies"][0])
                if "min_accuracy" in qs:
                    candidate["min_accuracy"] = float(qs["min_accuracy"][0])
                if "cooldown" in qs:
                    candidate["cooldown"] = float(qs["cooldown"][0])
            except ValueError:
                pass  # ignore malformed overrides, fall back to defaults
            return candidate

        def do_GET(self):
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            parts = [p for p in parsed.path.split("/") if p]

            if not parts:
                self._send(
                    render_home(
                        registry,
                        qs.get("error", [None])[0],
                        qs.get("directory", [""])[0],
                    )
                )
                return

            if parts == ["browse"]:
                raw_path = qs.get("path", [str(Path.home())])[0]
                self._send(render_browse(Path(raw_path).expanduser()))
                return

            if parts[0] == "run" and len(parts) == 2:
                run = registry.get(parts[1])
                if run is None:
                    self._send(bt_page("Not found", "<p>Run not found.</p>"), status=404)
                    return
                self._send(render_run(run))
                return

            if parts[0] == "run" and len(parts) == 3 and parts[2] == "delete":
                run = registry.get(parts[1])
                if run is None:
                    self._send(bt_page("Not found", "<p>Run not found.</p>"), status=404)
                    return
                self._send(render_delete_confirm(run))
                return

            if parts[0] == "run" and len(parts) == 4 and parts[2] == "recording":
                run = registry.get(parts[1])
                if run is None:
                    self._send(bt_page("Not found", "<p>Run not found.</p>"), status=404)
                    return
                try:
                    entry = run.recordings[int(parts[3])]
                except (ValueError, IndexError):
                    self._send(bt_page("Not found", "<p>Recording not found.</p>"), status=404)
                    return
                self._send(render_recording(run, entry, self._candidate_from_qs(qs)))
                return

            self._send(bt_page("Not found", "<p>Not found.</p>"), status=404)

        def do_POST(self):
            parsed = urlparse(self.path)
            parts = [p for p in parsed.path.split("/") if p]
            length = int(self.headers.get("Content-Length", 0))
            form = parse_qs(self.rfile.read(length).decode("utf-8")) if length else {}

            if parsed.path == "/run":
                directory = form.get("directory", [""])[0].strip()
                name = form.get("name", [""])[0].strip()
                if not directory:
                    self._redirect("/?error=" + quote("Directory is required"))
                    return
                root = Path(directory).expanduser()
                if not root.is_dir():
                    self._redirect("/?error=" + quote(f"Not a directory: {directory}"))
                    return
                run = start_run(str(root.resolve()), registry, detector, name=name)
                self._redirect(f"/run/{run.run_id}")
                return

            if parts[0] == "run" and len(parts) == 3 and parts[2] == "rename":
                run = registry.get(parts[1])
                if run is None:
                    self._send(bt_page("Not found", "<p>Run not found.</p>"), status=404)
                    return
                run.name = form.get("name", [""])[0].strip()
                if run.status != "running":
                    registry.save(run)
                self._redirect(f"/run/{run.run_id}")
                return

            if parts[0] == "run" and len(parts) == 3 and parts[2] == "delete":
                run = registry.get(parts[1])
                if run is None:
                    self._send(bt_page("Not found", "<p>Run not found.</p>"), status=404)
                    return
                if run.status == "running":
                    self._redirect(f"/run/{run.run_id}/delete")
                    return
                registry.delete(run.run_id)
                self._redirect("/")
                return

            self._send(bt_page("Not found", "<p>Not found.</p>"), status=404)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "recordings_dir",
        nargs="?",
        type=Path,
        help="optional: also start one run immediately on startup (or start runs from the browser)",
    )
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=DEFAULT_STORAGE_DIR,
        help=f"where to persist batch results (default: {DEFAULT_STORAGE_DIR})",
    )
    parser.add_argument("--no-browser", action="store_true", help="don't auto-open a browser tab")
    args = parser.parse_args()

    registry = RunRegistry(args.storage_dir.resolve())
    print(f"Loaded {len(registry.all_sorted())} previous run(s) from {registry.storage_dir}")

    detector = load_detector()
    print(f"Model: {Path(config.MODEL_BUNDLE_PATH).name}  threshold={detector.threshold:.4f}")

    if args.recordings_dir is not None:
        root = args.recordings_dir.resolve()
        if not root.is_dir():
            print(f"ERROR: {root} is not a directory")
            raise SystemExit(1)
        run = start_run(str(root), registry, detector)
        print(f"Started batch run {run.run_id} for {root}")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(registry, detector))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"\nServing at {url} (Ctrl+C to stop)")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
