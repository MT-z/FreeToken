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
    # No `with`: the lifespan hooks would shut down whatever _GLOBAL_STATE another test
    # left behind; the middleware needs neither startup nor state.
    client = TestClient(api_server.app)
    r = client.post("/nowhere", json={"model": "x", "messages": []}, headers={"x-api-key": "secret"})
    assert r.status_code == 404
    assert w.flush()
    text = (tmp_path / "wire.log").read_text(encoding="utf-8")
    assert ">>> " in text and "POST /nowhere" in text and "<<< " in text and " 404 " in text
    assert "req-body-file " in text and "secret" not in text
    bodies = [os.path.join(d, f) for d, _, fs in os.walk(tmp_path / "bodies") for f in fs]
    assert len(bodies) == 1 and json.loads(Path(bodies[0]).read_bytes()) == {"model": "x", "messages": []}
    w.close()


def test_the_apps_shutdown_hook_closes_the_log_so_a_sigterm_stop_keeps_its_records(tmp_path, monkeypatch):
    # uvicorn runs the lifespan shutdown on SIGTERM/SIGINT; atexit does not run when the
    # process dies by the signal. So the hook, not atexit, is what drains the queue and
    # writes the closing summary on the exit that actually happens.
    import threading

    from fastapi.testclient import TestClient
    from freetoken.server import api_server

    w = _log(tmp_path)
    monkeypatch.setattr(api_server, "_WIRE", w)
    monkeypatch.setattr(api_server, "_GLOBAL_STATE", None)
    monkeypatch.setattr(api_server, "_SHUTTING_DOWN", threading.Event())
    with TestClient(api_server.app) as client:
        client.post("/nowhere", json={"model": "x"})
    # No explicit close(): the lifespan exit must have done it.
    assert w._worker is None
    lines = (tmp_path / "wire.log").read_text(encoding="utf-8").splitlines()
    assert lines[-1].startswith("wire: ") and "1 bodies written" in lines[-1] and lines[-1].endswith("0 dropped, 0 failed")
    bodies = [f for _, _, fs in os.walk(tmp_path / "bodies") for f in fs]
    assert bodies == ["req-00001.json"]


def test_the_shutdown_hook_closes_the_log_even_when_the_states_teardown_raises(tmp_path, monkeypatch):
    # shutdown() stops the tokenizer sockets and terminates workers; if it raises, the log
    # must still be closed, or the fix above silently comes undone on exactly that exit.
    import threading
    from types import SimpleNamespace

    import pytest
    from fastapi.testclient import TestClient
    from freetoken.server import api_server

    def boom():
        raise RuntimeError("teardown failed")

    w = _log(tmp_path)
    monkeypatch.setattr(api_server, "_WIRE", w)
    monkeypatch.setattr(api_server, "_GLOBAL_STATE", SimpleNamespace(shutdown=boom))
    monkeypatch.setattr(api_server, "_SHUTTING_DOWN", threading.Event())
    with pytest.raises(RuntimeError, match="teardown failed"), TestClient(api_server.app) as client:
        client.post("/nowhere", json={"model": "x"})
    assert w._worker is None
    lines = (tmp_path / "wire.log").read_text(encoding="utf-8").splitlines()
    assert lines[-1].startswith("wire: ") and "1 bodies written" in lines[-1]


def test_importing_the_module_changes_no_signal_handler_and_starts_nothing():
    # Measured in a fresh interpreter with the variables unset, because this process has
    # long since imported the module. An import must not touch SIGINT/SIGTERM/SIGHUP.
    import subprocess

    code = (
        "import signal, threading, os\n"
        "for v in ('FREETOKEN_WIRE_LOG', 'FREETOKEN_WIRE_BODY_DIR'): os.environ.pop(v, None)\n"
        "sigs = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)\n"
        "before = [signal.getsignal(s) for s in sigs]; n = threading.active_count()\n"
        "import freetoken.server.wire_log as W\n"
        "after = [signal.getsignal(s) for s in sigs]\n"
        "assert before == after, (before, after)\n"
        "assert threading.active_count() == n\n"
        "assert W.WIRE._worker is None and not W.WIRE.enabled()\n"
        "print('clean')\n"
    )
    env = {**os.environ, "PYTHONPATH": _PY}
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120, check=False)
    assert r.returncode == 0 and r.stdout.strip() == "clean", r.stderr[-800:]


def test_close_spends_at_most_one_timeout_even_when_the_queue_is_full_and_the_writer_is_stuck(tmp_path, monkeypatch):
    # Full queue, no consumer: both puts would each wait the whole timeout and the join a
    # third, tripling the stall the docstring promises to bound. One deadline for all three.
    import threading
    import time

    w = _log(tmp_path, max_queue=1)
    monkeypatch.setattr(w, "_ensure_worker", lambda: None)
    w.line("a")
    w.line("b")  # dropped; the queue now holds one item nobody will take
    stuck = threading.Thread(target=time.sleep, args=(2.0,), daemon=True)
    stuck.start()
    w._worker = stuck
    t = time.perf_counter()
    w.close(timeout=0.3)
    elapsed = time.perf_counter() - t
    assert 0.25 <= elapsed < 0.6, elapsed  # one budget (~0.3), not three (~0.9)
    assert w._closed and w._worker is None


def test_a_record_after_close_is_counted_as_dropped_not_queued_into_the_void(tmp_path):
    w = _log(tmp_path)
    w.line("before")
    w.close()
    w.line("after")
    assert w.body(b"late") is not None  # the caller still gets a name for its log line
    assert w.dropped == 2 and w._worker is None and w._queue.unfinished_tasks == 0


def test_sighup_reaches_the_stop_signal_chain_which_closes_the_log_before_the_process_dies(tmp_path, monkeypatch):
    # uvicorn handles SIGINT and SIGTERM only; a closed terminal sends SIGHUP, which goes
    # straight to the chain api_server installs before uvicorn (the pidfile's). That chain
    # must drain and close the wire log before re-raising the signal with the default action.
    import signal

    from freetoken.server import api_server

    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
    real_kill = os.kill
    sent: list[tuple[int, int]] = []

    def fake_kill(pid, sig):  # let liveness probes through, record the re-raise
        if sig == 0:
            return real_kill(pid, 0)
        sent.append((pid, sig))

    monkeypatch.setattr(os, "kill", fake_kill)
    try:
        signal.signal(signal.SIGHUP, signal.SIG_DFL)  # what a foreground serve has for SIGHUP
        w = _log(tmp_path)
        monkeypatch.setattr(api_server, "_WIRE", w)
        pidfile = tmp_path / "serve.pid"
        pidfile.write_text(str(os.getpid()))
        api_server._install_stop_signal_handlers(str(pidfile))
        w.line("queued before the hang-up")
        w.body(b"{}")
        signal.getsignal(signal.SIGHUP)(signal.SIGHUP, None)  # deliver it by hand
        assert sent == [(os.getpid(), signal.SIGHUP)] and signal.getsignal(signal.SIGHUP) is signal.SIG_DFL
        assert not pidfile.exists()
        assert w._worker is None  # closed by the chain, not by a later atexit
        lines = (tmp_path / "wire.log").read_text(encoding="utf-8").splitlines()
        assert lines[0] == "queued before the hang-up"
        assert lines[-1].startswith("wire: 1 lines, 1 bodies written") and lines[-1].endswith("0 dropped, 0 failed")
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)


def test_the_stop_signal_chain_protects_the_log_even_when_no_pidfile_could_be_written(tmp_path, monkeypatch):
    # The chain is installed whether or not the pidfile was written; with none, SIGHUP still
    # drains and closes the log before the default action, and nothing tries to unlink.
    import signal

    from freetoken.server import api_server

    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
    real_kill = os.kill
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: real_kill(pid, 0) if sig == 0 else sent.append((pid, sig)))
    try:
        signal.signal(signal.SIGHUP, signal.SIG_DFL)
        w = _log(tmp_path)
        monkeypatch.setattr(api_server, "_WIRE", w)
        api_server._install_stop_signal_handlers(None)
        w.line("no pidfile, still protected")
        signal.getsignal(signal.SIGHUP)(signal.SIGHUP, None)
        assert sent == [(os.getpid(), signal.SIGHUP)] and w._worker is None
        lines = (tmp_path / "wire.log").read_text(encoding="utf-8").splitlines()
        assert lines[0] == "no pidfile, still protected" and lines[-1].startswith("wire: 1 lines, 0 bodies written")
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)
