#!/usr/bin/env python3
"""Small local web UI over analyze_logs.py, for browsing session logs without
the CLI.

Lists every session found in a logs directory (default: ../logs, the
directory scripts/pi-sync-logs fills from the Pi) and lets you open one to
see its summary and a score-over-time chart in a browser.

Re-scans the logs directory on every request, so re-running
scripts/pi-sync-logs and reloading the page picks up new sessions
immediately -- no server restart needed.

Standard library only.

Usage:
    ./log_viewer.py                # serves ../logs on http://localhost:8765
    ./log_viewer.py /path/to/logs --port 9000
"""
from __future__ import annotations

import argparse
import html
import io
import re
import webbrowser
from contextlib import redirect_stdout
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from analyze_logs import (
    Session,
    ascii_timeline,
    parse_log,
    print_replay_comparison,
    print_session_summary,
)

TRAILING_NUM_RE = re.compile(r"_(\d+)\.log$")
# Superseded by the stdout-redirect log now that detector.py logs to stdout
# only (see detector.py's logging setup) -- skip so historical field logs
# don't show each session twice.
LEGACY_DUPLICATE_RE = re.compile(r"^detections(_\d+)?(-latest)?\.log$")

DEFAULT_CANDIDATE = {"window": 5, "min_anomalies": 2, "min_accuracy": 60.0, "cooldown": 8.0}

PAGE_CSS = """
  body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
         margin: 2rem auto; max-width: 900px; color: #1a1a1a; }
  h1 { font-size: 1.4rem; }
  h1 a { color: inherit; text-decoration: none; }
  table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #e5e5e5; font-size: 0.92rem; }
  th { color: #555; font-weight: 600; }
  tr:hover td { background: #fafafa; }
  a { color: #2a6f97; }
  .muted { color: #777; }
  pre { background: #f6f6f6; border: 1px solid #e5e5e5; border-radius: 6px;
        padding: 1rem; overflow-x: auto; font-size: 0.85rem; line-height: 1.4; }
  form.candidate { margin: 1rem 0; padding: 0.75rem 1rem; background: #f6f6f6;
        border: 1px solid #e5e5e5; border-radius: 6px; }
  form.candidate label { margin-right: 1rem; font-size: 0.85rem; }
  form.candidate input { width: 4.5rem; }
  .backlink { display: inline-block; margin-bottom: 1rem; }
  .chart-toggle { display: none; }
  .chart-legend { margin: 0.4rem 0; font-size: 0.85rem; }
  .chart-legend label { margin-right: 1.25rem; cursor: pointer; user-select: none; }
  .swatch { display: inline-block; width: 20px; height: 0; margin-right: 5px;
            vertical-align: middle; border-top-width: 3px; }
  .swatch-model { border-top-style: dotted; border-top-color: #e07b00; }
  .swatch-event { border-top-style: dashed; border-top-color: #999; }
  #toggle-model-thr:not(:checked) ~ .chart-wrap .model-threshold-line { display: none; }
  #toggle-event-thr:not(:checked) ~ .chart-wrap .event-threshold-line { display: none; }
  #toggle-model-thr:not(:checked) ~ .chart-legend .legend-model { opacity: 0.4; }
  #toggle-event-thr:not(:checked) ~ .chart-legend .legend-event { opacity: 0.4; }
"""


@dataclass
class Entry:
    id: int
    file: Path
    session: Session
    seq: int | None


def trailing_seq(path: Path) -> int | None:
    m = TRAILING_NUM_RE.search(path.name)
    return int(m.group(1)) if m else None


def scan(logs_dir: Path) -> list[Entry]:
    entries: list[Entry] = []
    files = sorted(
        p
        for p in logs_dir.glob("*.log")
        if not p.is_symlink() and not LEGACY_DUPLICATE_RE.match(p.name)
    )
    for f in files:
        for sess in parse_log(f):
            if not sess.chunks and not sess.events:
                continue
            entries.append(Entry(id=0, file=f, session=sess, seq=trailing_seq(f)))
    # Sort by the session's own recorded timestamp. Filenames from before
    # the sequence-counter naming fix are PID-based, which is not
    # chronological, so it can only be a tiebreaker, not the primary key.
    entries.sort(key=lambda e: (e.session.start_ts, e.seq if e.seq is not None else -1))
    for i, e in enumerate(entries):
        e.id = i
    return entries


def page(title: str, body: str) -> bytes:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>{PAGE_CSS}</style></head>
<body>{body}</body></html>""".encode("utf-8")


def render_index(entries: list[Entry]) -> bytes:
    rows = []
    for e in entries:
        sess = e.session
        start = f"{sess.start_ts:%Y-%m-%d %H:%M:%S}" if sess.start_ts else "-"
        duration = "-"
        drone_pct = "-"
        if sess.chunks:
            secs = (sess.chunks[-1].ts - sess.chunks[0].ts).total_seconds()
            duration = f"{secs:.0f}s" if secs < 60 else f"{secs / 60:.1f}m"
            drone_n = sum(1 for c in sess.chunks if c.is_anomaly)
            drone_pct = f"{100 * drone_n / len(sess.chunks):.1f}%"
        rows.append(
            f"<tr><td><a href='/session/{e.id}'>{html.escape(e.file.name)}</a></td>"
            f"<td>{start}</td><td>{duration}</td><td>{len(sess.chunks)}</td>"
            f"<td>{drone_pct}</td><td>{len(sess.events)}</td></tr>"
        )
    table = (
        "<table><thead><tr><th>File</th><th>Started</th><th>Duration</th>"
        "<th>Chunks</th><th>Drone-flagged</th><th>Events sent</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        if rows
        else "<p class='muted'>No session logs found. Run scripts/pi-sync-logs first.</p>"
    )
    body = f"<h1>Drone detector session logs</h1>{table}"
    return page("Session logs", body)


def render_svg_chart(sess: Session, width: int = 760, height: int = 260) -> str:
    """Score-over-time chart: fixed 0-100% scale (scores are always a
    probability, so this never needs to rescale to the data -- rescaling to
    the observed max is what made a quiet, no-drone session's noise fill the
    whole chart and look meaningful when it wasn't) plus the model's own
    per-chunk decision threshold and the app-level event-trigger threshold,
    each independently toggleable via the checkbox-hack (no JS in this tool).
    """
    if not sess.chunks:
        return ""
    margin_left = 34
    margin_y = 6
    plot_h = height - 2 * margin_y

    t0, t1 = sess.chunks[0].ts, sess.chunks[-1].ts
    span = (t1 - t0).total_seconds() or 1.0

    def y_for_frac(frac: float) -> float:
        return margin_y + plot_h - frac * plot_h

    buckets = min(400, len(sess.chunks))
    bucket_vals: list[list[float]] = [[] for _ in range(buckets)]
    for c in sess.chunks:
        i = min(buckets - 1, int((c.ts - t0).total_seconds() / span * buckets))
        bucket_vals[i].append(c.score)

    points = []
    for i, vals in enumerate(bucket_vals):
        x = margin_left + width * i / (buckets - 1) if buckets > 1 else margin_left
        v = max(vals) if vals else 0.0
        points.append(f"{x:.1f},{y_for_frac(v):.1f}")
    polyline = " ".join(points)

    total_w = margin_left + width
    grid = []
    for pct in (0, 25, 50, 75, 100):
        gy = y_for_frac(pct / 100.0)
        grid.append(
            f'<line x1="{margin_left}" y1="{gy:.1f}" x2="{total_w}" y2="{gy:.1f}" '
            f'stroke="#eee" stroke-width="1" />'
        )
        grid.append(
            f'<text x="{margin_left - 5}" y="{gy + 3:.1f}" font-size="10" fill="#888" '
            f'text-anchor="end">{pct}</text>'
        )

    model_threshold_line = ""
    if sess.model_threshold_pct is not None:
        ty = y_for_frac(sess.model_threshold_pct / 100.0)
        model_threshold_line = (
            f'<line class="model-threshold-line" x1="{margin_left}" y1="{ty:.1f}" '
            f'x2="{total_w}" y2="{ty:.1f}" stroke="#e07b00" stroke-width="1.5" '
            f'stroke-dasharray="1,3" stroke-linecap="round" />'
        )

    event_threshold_line = ""
    if sess.min_accuracy_pct is not None:
        ty = y_for_frac(sess.min_accuracy_pct / 100.0)
        event_threshold_line = (
            f'<line class="event-threshold-line" x1="{margin_left}" y1="{ty:.1f}" '
            f'x2="{total_w}" y2="{ty:.1f}" stroke="#999" stroke-width="1.5" '
            f'stroke-dasharray="6,3" />'
        )

    marks = []
    for e in sess.events:
        x = margin_left + width * (e.ts - t0).total_seconds() / span
        marks.append(
            f'<line x1="{x:.1f}" y1="{margin_y}" x2="{x:.1f}" y2="{margin_y + plot_h}" '
            f'stroke="#e63946" stroke-width="1.5" opacity="0.75" />'
        )

    svg = (
        f'<svg viewBox="0 0 {total_w} {height}" width="100%" height="{height}" '
        f'preserveAspectRatio="none" style="background:#fafafa;border:1px solid #e5e5e5;'
        f'border-radius:6px;">'
        f"{''.join(grid)}"
        f"{event_threshold_line}{model_threshold_line}"
        f'<polyline points="{polyline}" fill="none" stroke="#2a6f97" stroke-width="1.5" />'
        f"{''.join(marks)}"
        "</svg>"
    )

    toggles = []
    if sess.model_threshold_pct is not None:
        toggles.append(
            '<input type="checkbox" id="toggle-model-thr" class="chart-toggle" checked>'
        )
    if sess.min_accuracy_pct is not None:
        toggles.append(
            '<input type="checkbox" id="toggle-event-thr" class="chart-toggle" checked>'
        )
    legend_items = []
    if sess.model_threshold_pct is not None:
        legend_items.append(
            "<label for='toggle-model-thr' class='legend-model'>"
            "<span class='swatch swatch-model'></span>"
            f"model threshold ({sess.model_threshold_pct:.1f}%)</label>"
        )
    if sess.min_accuracy_pct is not None:
        legend_items.append(
            "<label for='toggle-event-thr' class='legend-event'>"
            "<span class='swatch swatch-event'></span>"
            f"event threshold ({sess.min_accuracy_pct:.0f}%)</label>"
        )

    return (
        "".join(toggles)
        + f"<div class='chart-legend'>{''.join(legend_items)}</div>"
        + f"<div class='chart-wrap'>{svg}</div>"
    )


def render_detail(
    title: str,
    sess: Session,
    candidate: dict,
    back_href: str = "/",
    back_label: str = "all sessions",
) -> bytes:
    """Shared session/recording detail page: chart, replay form, summary text.

    Used by log_viewer's per-session page and batch_test's per-recording page
    alike, since both just need to show the same things about a Session.
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_session_summary(sess)
        print_replay_comparison(
            sess,
            candidate["window"],
            candidate["min_anomalies"],
            candidate["min_accuracy"],
            candidate["cooldown"],
        )
    summary_text = buf.getvalue().strip()

    chart = render_svg_chart(sess)
    chart_legend = (
        "<p class='muted'>Blue: score over time (fixed 0-100% scale). "
        "Red: a detection event was actually sent.</p>"
        if chart
        else ""
    )

    form = f"""
    <form class="candidate" method="get">
      <b>Replay with:</b>
      <label>window <input type="number" name="window" value="{candidate['window']}"></label>
      <label>min anomalies <input type="number" name="min_anomalies" value="{candidate['min_anomalies']}"></label>
      <label>min accuracy % <input type="number" step="0.1" name="min_accuracy" value="{candidate['min_accuracy']}"></label>
      <label>cooldown s <input type="number" step="0.1" name="cooldown" value="{candidate['cooldown']}"></label>
      <button type="submit">Re-run</button>
    </form>
    """

    body = (
        f"<a class='backlink' href='{back_href}'>&larr; {back_label}</a>"
        f"<h1>{html.escape(title)}</h1>"
        f"{chart}{chart_legend}"
        f"{form}"
        f"<pre>{html.escape(summary_text)}</pre>"
    )
    return page(title, body)


def render_session(entry: Entry, candidate: dict) -> bytes:
    return render_detail(entry.file.name, entry.session, candidate)


def make_handler(logs_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # noqa: A003 - quiet default access log
            pass

        def _send(self, body: bytes, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            entries = scan(logs_dir)

            if parsed.path == "/":
                self._send(render_index(entries))
                return

            if parsed.path.startswith("/session/"):
                try:
                    session_id = int(parsed.path.rsplit("/", 1)[-1])
                    entry = entries[session_id]
                except (ValueError, IndexError):
                    self._send(page("Not found", "<p>Session not found.</p>"), status=404)
                    return

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

                self._send(render_session(entry, candidate))
                return

            self._send(page("Not found", "<p>Not found.</p>"), status=404)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "logs_dir", nargs="?", type=Path, default=Path(__file__).resolve().parent.parent / "logs",
        help="directory of session logs to serve (default: ../logs)",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="don't auto-open a browser tab")
    args = parser.parse_args()

    logs_dir = args.logs_dir.resolve()
    if not logs_dir.is_dir():
        print(f"ERROR: {logs_dir} is not a directory")
        raise SystemExit(1)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(logs_dir))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Serving logs from {logs_dir}")
    print(f"Open {url} (Ctrl+C to stop)")
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
