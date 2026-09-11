"""The bound on a stop: uvicorn waits forever for in-flight generations without one.

uvicorn's timeout_graceful_shutdown defaults to None and server.py:309 spins until
server_state.connections drains, so a SIGTERM cannot complete while a request is still
generating. Measured on this box (freetoken-systest results/20260911-shutdown-hang.txt):
0.5 s with no client and 0.5 s with an idle keep-alive socket, 22 s with a request generating
4,096 tokens and 48 s at the 32,768 Claude Code sends. daemon/server.py:218 already bounds its
own at 3 s; the serve did not bound its own at all.
"""
from types import SimpleNamespace

import pytest

import freetoken.server.api_server as api_server
from freetoken.server.args import ServerArgs


@pytest.fixture
def _state(monkeypatch):
    def set_grace(value):
        monkeypatch.setattr(api_server, "_GLOBAL_STATE",
                            SimpleNamespace(config=SimpleNamespace(shutdown_grace_seconds=value)))
    return set_grace


def test_the_configured_grace_reaches_uvicorn(_state):
    _state(45)
    assert api_server._shutdown_grace() == 45


def test_zero_means_uvicorns_unbounded_wait(_state):
    """0 is the escape hatch back to the old behaviour, and uvicorn spells that None."""
    _state(0)
    assert api_server._shutdown_grace() is None


def test_a_serve_without_global_state_still_has_a_bound(monkeypatch):
    """_serve_and_run_shell builds its uvicorn Server from a helper that never sees the config,
    so the fallback must be the default rather than None -- both paths have to agree, or a stop
    completes on one and hangs on the other."""
    monkeypatch.setattr(api_server, "_GLOBAL_STATE", None)
    assert api_server._shutdown_grace() == ServerArgs.shutdown_grace_seconds
    assert api_server._shutdown_grace() is not None


def test_the_default_is_a_bound_and_the_flag_exists():
    import contextlib
    import io

    from freetoken.server.args import parse_args

    assert ServerArgs.shutdown_grace_seconds > 0, "the default must bound the wait"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.suppress(SystemExit):
        parse_args(["--help"])
    assert "--shutdown-grace-seconds" in buf.getvalue()
