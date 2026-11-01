#!/usr/bin/env python3
"""
VocalKeep Phase 2: a local web UI. Drag a video in, get the speech-only one back.

    python web/app.py                 # http://127.0.0.1:8610
    python web/app.py --port 9000

Standard library only. Jobs run one at a time (one GPU), finished files are
deleted an hour after they finish. Binds to localhost by default: nothing
leaves this machine.
"""
import argparse
import itertools
import json
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.separation import remove_music_from_video, DEFAULT_MODEL  # noqa: E402  core/__init__ sets the ROCm env first

INDEX = Path(__file__).with_name("index.html")
MAX_BYTES = 20 * 1024 ** 3                               # a full movie; GPU work is chunked anyway
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
MODELS = {"htdemucs", "htdemucs_ft"}
KEEP_SECONDS = 3600
WORK = os.environ.get("VOCALKEEP_WORKDIR") or None          # e.g. a big disk; default: the system temp dir
SEQ = itertools.count()


class Job:
    def __init__(self, workdir: Path, src: Path, model: str):
        self.id, self.seq = uuid.uuid4().hex, next(SEQ)
        self.workdir, self.src, self.model = workdir, src, model
        self.out = workdir / f"{src.stem}_vocalkeep.mp4"
        self.state, self.device, self.error = "queued", None, None
        self.started = self.finished = None

    def status(self, ahead: int) -> dict:
        end = self.finished or time.time()
        return {"state": self.state, "device": self.device, "error": self.error, "ahead": ahead,
                "seconds": round(end - self.started, 1) if self.started else 0}


JOBS: dict[str, Job] = {}
PENDING: "queue.Queue[Job]" = queue.Queue()
LOCK = threading.Lock()


def worker():
    """The single GPU worker: takes jobs in upload order."""
    while True:
        job = PENDING.get()
        job.state, job.started = "running", time.time()
        try:
            result = remove_music_from_video(job.src, job.out, model_name=job.model)
            job.device = result.device_used.note
            job.state = "done"
        except Exception as e:                               # report it in the page, keep serving
            job.error, job.state = str(e).strip().splitlines()[-1][:500], "error"
        job.finished = time.time()
        job.src.unlink(missing_ok=True)


def janitor():
    """Deletes finished jobs' files an hour after they finish."""
    while True:
        time.sleep(60)
        now = time.time()
        with LOCK:
            old = [j for j in JOBS.values() if j.finished and now - j.finished > KEEP_SECONDS]
            for j in old:
                del JOBS[j.id]
        for j in old:
            shutil.rmtree(j.workdir, ignore_errors=True)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, data: dict):
        self._send(code, json.dumps(data).encode(), "application/json")

    def _job(self) -> Job | None:
        parts = self.path.split("/")                         # ['', 'api', 'jobs', <id>, ('file')]
        with LOCK:
            return JOBS.get(parts[3]) if len(parts) > 3 else None

    def do_GET(self):
        if self.path == "/":
            return self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
        if not self.path.startswith("/api/jobs/"):
            return self._json(404, {"error": "not found"})
        job = self._job()
        if job is None:
            return self._json(404, {"error": "unknown or expired job"})
        if self.path.endswith("/file"):
            if job.state != "done":
                return self._json(409, {"error": "not finished"})
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(job.out.stat().st_size))
            self.send_header("Content-Disposition", f'attachment; filename="{job.out.name}"')
            self.end_headers()
            with open(job.out, "rb") as f:
                shutil.copyfileobj(f, self.wfile)
            return
        with LOCK:
            ahead = sum(1 for j in JOBS.values() if j.state in ("queued", "running") and j.seq < job.seq)
        self._json(200, job.status(ahead))

    def do_POST(self):
        if self.path != "/api/jobs":
            return self._json(404, {"error": "not found"})
        size = int(self.headers.get("Content-Length") or 0)
        name = Path(unquote(self.headers.get("X-Filename", ""))).name    # basename only: no path tricks
        model = self.headers.get("X-Model", DEFAULT_MODEL)
        if not 0 < size <= MAX_BYTES:
            return self._json(413, {"error": "file must be 1 byte to 20 GB"})
        if Path(name).suffix.lower() not in VIDEO_EXT:
            return self._json(415, {"error": "not a supported video type"})
        if model not in MODELS:
            return self._json(400, {"error": "unknown model"})
        workdir = Path(tempfile.mkdtemp(prefix="vocalkeep_", dir=WORK))     # on disk: /tmp is often RAM (tmpfs)
        src = workdir / name
        with open(src, "wb") as f:                           # stream to disk, never the whole file in RAM
            left = size
            while left:
                chunk = self.rfile.read(min(left, 1 << 20))
                if not chunk:
                    shutil.rmtree(workdir, ignore_errors=True)
                    return self._json(400, {"error": "upload cut short"})
                f.write(chunk)
                left -= len(chunk)
        job = Job(workdir, src, model)
        with LOCK:
            JOBS[job.id] = job
        PENDING.put(job)
        self._json(200, {"id": job.id})

    def log_message(self, fmt, *args):                       # quiet: only errors reach the terminal
        if args and str(args[1])[:1] in "45":
            super().log_message(fmt, *args)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8610)
    args = ap.parse_args()
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=janitor, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"VocalKeep running at http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:                                                 # nothing left behind in /tmp on exit
        with LOCK:
            for j in JOBS.values():
                shutil.rmtree(j.workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
