"""The serve's pidfile: written, refused, taken over, and -- the part that took a strace to
see -- released when the process dies BY the stopping signal rather than by returning.

uvicorn catches SIGTERM, shuts down, restores the previous handler and re-raises, so the
process ends by signal and no ``finally`` in the caller runs. The release therefore has to
live on the signal path too. No server is started here: the handler is invoked directly
with the re-raise captured, which is the same call uvicorn's ``raise_signal`` would make.
"""
from __future__ import annotations

import os
import signal
import subprocess

import pytest
from freetoken.server.api_server import (
    _install_stop_signal_handlers,
    _pid_in,
    _pidfile,
    _release_pidfile,
)


@pytest.fixture
def restore_signals():
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def _dead_pid() -> int:
    """A pid that certainly names no process: a child that has already been reaped."""
    child = subprocess.Popen(["true"])
    child.wait()
    return child.pid


def _capture_reraise(monkeypatch) -> list[tuple[int, int]]:
    """Record real signals sent to ourselves; let ``kill(pid, 0)`` liveness probes through,
    because ``_pid_in`` relies on them to decide whether the file is still ours."""
    real_kill = os.kill
    sent: list[tuple[int, int]] = []

    def fake_kill(pid, sig):
        if sig == 0:
            return real_kill(pid, 0)
        sent.append((pid, sig))

    monkeypatch.setattr(os, "kill", fake_kill)
    return sent


def test_pidfile_holds_our_pid_and_is_removed_on_normal_exit(tmp_path):
    path = tmp_path / "serve.pid"
    with _pidfile(str(path)):
        assert int(path.read_text()) == os.getpid()
    assert not path.exists()


def test_a_live_pid_in_the_file_refuses_a_second_serve(tmp_path):
    path = tmp_path / "serve.pid"
    path.write_text(f"{os.getpid()}\n")  # alive: it is us
    with pytest.raises(RuntimeError, match="already running on pid"):
        with _pidfile(str(path)):
            pass
    assert int(path.read_text()) == os.getpid(), "the refused start must not touch the file"


def test_a_stale_pid_is_taken_over(tmp_path):
    path = tmp_path / "serve.pid"
    path.write_text(f"{_dead_pid()}\n")
    with _pidfile(str(path)):
        assert int(path.read_text()) == os.getpid()
    assert not path.exists()


def test_release_is_idempotent_and_spares_a_successors_file(tmp_path):
    path = tmp_path / "serve.pid"
    path.write_text(f"{os.getpid()}\n")
    _release_pidfile(str(path))
    _release_pidfile(str(path))  # already gone: no error
    assert not path.exists()
    path.write_text("1\n")  # pid 1 is alive and not us
    _release_pidfile(str(path))
    assert _pid_in(str(path)) == 1, "a file naming another live process is not ours to remove"


def test_signal_release_unlinks_then_reraises_with_the_default_handler(tmp_path, monkeypatch, restore_signals):
    path = tmp_path / "serve.pid"
    path.write_text(f"{os.getpid()}\n")
    signal.signal(signal.SIGTERM, signal.SIG_DFL)  # what a non-shell serve has before uvicorn
    _install_stop_signal_handlers(str(path))
    handler = signal.getsignal(signal.SIGTERM)
    assert callable(handler) and handler is not signal.SIG_DFL

    sent = _capture_reraise(monkeypatch)
    handler(signal.SIGTERM, None)  # uvicorn's raise_signal, after it restored us

    assert not path.exists(), "the pidfile is released on the signal path"
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL, "the default action is restored..."
    assert sent == [(os.getpid(), signal.SIGTERM)], "...and the signal re-raised, so the exit is still 'killed by SIGTERM'"


def test_signal_release_chains_to_a_callable_previous_handler(tmp_path, monkeypatch, restore_signals):
    path = tmp_path / "serve.pid"
    path.write_text(f"{os.getpid()}\n")
    seen: list[int] = []
    signal.signal(signal.SIGHUP, lambda signum, frame: seen.append(signum))  # e.g. the shell stop handler
    _install_stop_signal_handlers(str(path))
    sent = _capture_reraise(monkeypatch)

    signal.getsignal(signal.SIGHUP)(signal.SIGHUP, None)

    assert not path.exists()
    assert seen == [signal.SIGHUP]
    assert sent == [], "with a handler in place it chains; it must not re-raise on top"
