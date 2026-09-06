"""The wire log (FREETOKEN_WIRE_LOG / FREETOKEN_WIRE_BODY_DIR), a diagnostic that must not
overwrite, must not block, and must not lose data silently.

Run:  PYTHONPATH=python <venv>/bin/python -m pytest tests/server/test_wire_log.py -v
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PY = os.path.join(_ROOT, "python")
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from freetoken.server.wire_log import WireLog, headers_line


def _log(tmp_path, **kw) -> WireLog:
    return WireLog(str(tmp_path / "wire.log"), str(tmp_path / "bodies"), **kw)


def test_disabled_without_a_log_path_does_nothing(tmp_path):
    w = WireLog(None, str(tmp_path / "bodies"))
    assert w.enabled() is False
    w.line("x")
    assert w.body(b"{}") is None
    assert w._worker is None and not os.path.exists(tmp_path / "bodies")


def test_lines_and_bodies_land_after_flush_and_bodies_live_under_a_run_directory(tmp_path):
    w = _log(tmp_path)
    w.line("first")
    name = w.body(b'{"a": 1}')
    assert w.flush() and name is not None
    assert (tmp_path / "wire.log").read_text(encoding="utf-8").splitlines() == ["first"]
    assert Path(name).read_bytes() == b'{"a": 1}'
    run_dir = os.path.dirname(name)
    assert os.path.dirname(run_dir) == str(tmp_path / "bodies")
    assert os.path.basename(run_dir).endswith(f"-{os.getpid()}")
    assert os.path.basename(name) == "req-00001.json"
    w.close()


def test_a_restart_into_the_same_directory_never_replaces_earlier_captures(tmp_path):
    first = _log(tmp_path)
    old = first.body(b"old run")
    assert first.flush()
    first.close()
    second = _log(tmp_path)  # a new process would get a new stamp; force distinct dirs here
    second.run_dir = os.path.join(second.body_root, "later-run")
    new = second.body(b"new run")
    assert second.flush()
    assert Path(old).read_bytes() == b"old run"  # untouched
    assert os.path.basename(new) == "req-00001.json" and new != old
    assert Path(new).read_bytes() == b"new run"
    second.close()


def test_an_existing_body_file_is_never_overwritten_and_the_failure_is_logged(tmp_path):
    w = _log(tmp_path)
    w.run_dir = str(tmp_path / "bodies" / "run")
    os.makedirs(w.run_dir)
    taken = os.path.join(w.run_dir, "req-00001.json")
    Path(taken).write_bytes(b"keep me")
    assert w.body(b"intruder") == taken
    assert w.flush()
    assert Path(taken).read_bytes() == b"keep me"
    assert w.failed == 1
    assert any("could not write" in line and "req-00001.json" in line
               for line in (tmp_path / "wire.log").read_text(encoding="utf-8").splitlines())
    w.close()


def test_a_full_queue_drops_instead_of_blocking_and_says_so_in_the_log(tmp_path, monkeypatch):
    w = _log(tmp_path, max_queue=1)
    monkeypatch.setattr(w, "_ensure_worker", lambda: None)  # hold the writer back
    for text in ("one", "two", "three"):
        w.line(text)
    assert w.dropped == 2  # the queue held one; nothing blocked, nothing raised
    monkeypatch.undo()
    w._ensure_worker()
    assert w.flush()
    lines = (tmp_path / "wire.log").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "one"
    assert any(line.startswith("wire: dropped 2 records") for line in lines)
    w.close()


def test_close_writes_a_summary_with_the_counts(tmp_path):
    w = _log(tmp_path)
    w.line("a")
    w.body(b"b")
    w.close()
    lines = (tmp_path / "wire.log").read_text(encoding="utf-8").splitlines()
    assert lines[-1].startswith("wire: 1 lines, 1 bodies written under ")
    assert lines[-1].endswith("; 0 dropped, 0 failed")
    w.close()  # idempotent


def test_headers_line_skips_credentials():
    items = [("Authorization", "secret"), ("x-api-key", "k"), ("content-type", "json")]
    assert headers_line(items) == "content-type=json"


def test_the_middleware_logs_through_the_writer_and_never_touches_the_disk_itself(tmp_path, monkeypatch):
    # End to end through the real app: a POST to a path that does not exist still passes the
    # middleware, so the request line, the capped body and the body file all appear -- after
    # the writer has drained, not before.
    from fastapi.testclient import TestClient
    from freetoken.server import api_server

    w = _log(tmp_path)
    monkeypatch.setattr(api_server, "_WIRE", w)
    with TestClient(api_server.app) as client:
        r = client.post("/nowhere", json={"model": "x", "messages": []}, headers={"x-api-key": "secret"})
    assert r.status_code == 404
    assert w.flush()
    text = (tmp_path / "wire.log").read_text(encoding="utf-8")
    assert ">>> " in text and "POST /nowhere" in text and "<<< " in text and " 404 " in text
    assert "req-body-file " in text and "secret" not in text
    bodies = [os.path.join(d, f) for d, _, fs in os.walk(tmp_path / "bodies") for f in fs]
    assert len(bodies) == 1 and json.loads(Path(bodies[0]).read_bytes()) == {"model": "x", "messages": []}
    w.close()
