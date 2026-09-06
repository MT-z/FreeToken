"""Wire log -- a diagnostic, off unless FREETOKEN_WIRE_LOG is set.

Records every request and response (method, query, headers, capped bodies, timing) into
one file, and with FREETOKEN_WIRE_BODY_DIR also drops each full request body into
``<dir>/<start-stamp>-<pid>/req-<NNNNN>.json`` for byte-exact diffing of consecutive
requests. Written for the prefix-cache investigation (.claude/REVIEW-prefix-cache.md).

Three rules, each learned the hard way:

* Nothing here runs on the event loop's time. Lines and bodies go through a bounded
  queue to one writer thread (the request_logger pattern); a slow or full disk drops
  records instead of stalling requests or SSE streams.
* Nothing is overwritten. Each process writes under its own run directory and creates
  body files exclusively, so a restart into the same FREETOKEN_WIRE_BODY_DIR can never
  replace the files an older log still points at.
* Nothing fails silently. Dropped and failed writes are counted and reported inside the
  log itself, and a summary line closes it, so a capture that lost data says so.
"""

from __future__ import annotations

import atexit
import itertools
import os
import queue
import threading
import time
from typing import Any

SKIP_HEADERS = {"authorization", "x-api-key", "cookie", "anthropic-auth-token"}
_MAX_QUEUE = 4096


def headers_line(items) -> str:
    return " ".join(f"{k}={v}" for k, v in items if k.lower() not in SKIP_HEADERS)


def stamp(clock=time) -> str:
    return clock.strftime("%H:%M:%S", clock.localtime()) + f".{int(clock.time() * 1000) % 1000:03d}"


class WireLog:
    """One writer thread per instance; every public method returns immediately."""

    def __init__(self, log_path: str | None, body_dir: str | None, *, max_queue: int = _MAX_QUEUE) -> None:
        self.log_path = log_path
        self.body_root = body_dir
        self.run_dir: str | None = None  # decided at first body, one per process
        self._seq = itertools.count(1)
        self._queue: queue.Queue[tuple[Any, ...] | None] = queue.Queue(maxsize=max_queue)
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()
        self._fh = None
        self.written_lines = 0
        self.written_bodies = 0
        self.dropped = 0  # queue full: the record never reached the writer
        self.failed = 0  # the writer could not write it (disk, permissions, name taken)
        self._dropped_reported = 0

    # ------------------------------------------------------------------ producers
    def enabled(self) -> bool:
        return bool(self.log_path)

    def line(self, text: str) -> None:
        if self.log_path:
            self._put(("line", text))

    def body(self, raw: bytes) -> str | None:
        """Queue one request body; returns the path it will be written to (for the log
        line), or None when body capture is off."""
        if not self.log_path or not self.body_root:
            return None
        name = os.path.join(self._run_directory(), f"req-{next(self._seq):05d}.json")
        self._put(("body", name, raw))
        return name

    def _run_directory(self) -> str:
        if self.run_dir is None:
            self.run_dir = os.path.join(
                self.body_root, time.strftime("%Y%m%dT%H%M%S", time.localtime()) + f"-{os.getpid()}"
            )
        return self.run_dir

    def _put(self, item: tuple[Any, ...]) -> None:
        self._ensure_worker()
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            with self._lock:
                self.dropped += 1

    def _ensure_worker(self) -> None:
        if self._worker is not None:
            return
        with self._lock:
            if self._worker is not None:
                return
            self._worker = threading.Thread(target=self._loop, name="wire-log-writer", daemon=True)
            self._worker.start()
            atexit.register(self.close)

    # ------------------------------------------------------------------ the writer
    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    break
                self._write(item)
                self._report_drops()
            finally:
                self._queue.task_done()

    def _write(self, item: tuple[Any, ...]) -> None:
        try:
            if item[0] == "line":
                self._write_line(item[1])
                self.written_lines += 1
            elif item[0] == "summary":
                self._try_line(self.summary())  # computed by the writer, after everything before it
            else:
                _, name, raw = item
                os.makedirs(os.path.dirname(name), exist_ok=True)
                with open(name, "xb") as fh:  # exclusive: never replace an existing capture
                    fh.write(raw)
                self.written_bodies += 1
        except Exception as exc:  # noqa: BLE001 -- the writer must outlive any one failure
            self.failed += 1
            if item[0] != "line":
                self._try_line(f"wire: could not write {item[1]}: {exc!r}")

    def _write_line(self, text: str) -> None:
        if self._fh is None:
            self._fh = open(self.log_path, "a", encoding="utf-8")  # noqa: SIM115 -- kept open by the writer
        self._fh.write(text + "\n")
        self._fh.flush()

    def _try_line(self, text: str) -> None:
        try:
            self._write_line(text)
        except Exception:  # noqa: BLE001 -- if the log itself is gone, only the counters remain
            self.failed += 1

    def _report_drops(self) -> None:
        with self._lock:
            dropped = self.dropped
        if dropped != self._dropped_reported:
            self._dropped_reported = dropped
            self._try_line(f"wire: dropped {dropped} records so far (queue full; slow disk?)")

    # ------------------------------------------------------------------ lifecycle
    def summary(self) -> str:
        return (
            f"wire: {self.written_lines} lines, {self.written_bodies} bodies written"
            f"{f' under {self.run_dir}' if self.run_dir else ''}; "
            f"{self.dropped} dropped, {self.failed} failed"
        )

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until everything queued so far has been handled. True if it was."""
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.005)
        return self._queue.unfinished_tasks == 0

    def close(self, timeout: float = 5.0) -> None:
        """Write the summary, stop the writer. Idempotent; registered with atexit."""
        worker, self._worker = self._worker, None
        if worker is None:
            return
        try:
            self._queue.put(("summary",), timeout=timeout)
            self._queue.put(None, timeout=timeout)
        except queue.Full:
            pass
        worker.join(timeout)
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None


WIRE = WireLog(os.environ.get("FREETOKEN_WIRE_LOG"), os.environ.get("FREETOKEN_WIRE_BODY_DIR"))
BODY_CAP = int(os.environ.get("FREETOKEN_WIRE_BODY_CAP", "4096"))
