"""--no-system-in-place / FREETOKEN_SYSTEM_IN_PLACE: how the serve decides where
mid-conversation system-role messages go, and how it tells the adapter.

Run:  PYTHONPATH=python <venv>/bin/python -m pytest tests/server/test_system_in_place_args.py -v
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PY = os.path.join(_ROOT, "python")
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from freetoken.server import anthropic_api as A  # noqa: E402
from freetoken.server import api_server  # noqa: E402
from freetoken.server.args import ServerArgs, parse_args  # noqa: E402

ENV = "FREETOKEN_SYSTEM_IN_PLACE"


class _Config:
    def __init__(self, data: dict) -> None:
        self._data = data

    def to_dict(self) -> dict:
        return self._data


def _parse(argv: list[str]) -> ServerArgs:
    # The parser reads the checkpoint config to infer parsers; keep the test off the disk.
    config = _Config({"architectures": ["Qwen2ForCausalLM"], "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: config):
        args, _run_shell = parse_args(["--model", "/models/anon", *argv])
    return args


def test_default_is_in_place_and_the_flag_turns_it_off():
    assert ServerArgs.system_in_place is True
    assert _parse([]).system_in_place is True
    assert _parse(["--no-system-in-place"]).system_in_place is False


def test_serve_resolves_flag_and_environment_as_either_may_turn_it_off(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert api_server._system_placement(SimpleNamespace(system_in_place=True)) is True
    assert api_server._system_placement(SimpleNamespace(system_in_place=False)) is False
    monkeypatch.setenv(ENV, "0")
    assert api_server._system_placement(SimpleNamespace(system_in_place=True)) is False
    monkeypatch.setenv(ENV, "1")
    assert api_server._system_placement(SimpleNamespace(system_in_place=False)) is False
    # A config from before the field existed (scheduler stubs) means the default.
    assert api_server._system_placement(SimpleNamespace()) is True


def test_configured_placement_wins_over_the_environment(monkeypatch):
    # Once the serve has configured the adapter, the environment no longer matters;
    # unconfigured (tests, offline tools) it falls back to the environment.
    monkeypatch.setattr(A, "_SYSTEM_IN_PLACE", None)
    monkeypatch.setenv(ENV, "0")
    assert A._system_in_place() is False
    monkeypatch.setattr(A, "_SYSTEM_IN_PLACE", True)
    assert A._system_in_place() is True
    monkeypatch.setattr(A, "_SYSTEM_IN_PLACE", False)
    monkeypatch.delenv(ENV, raising=False)
    assert A._system_in_place() is False
    A.configure_system_placement(True)
    assert A._system_in_place() is True
    monkeypatch.setattr(A, "_SYSTEM_IN_PLACE", None)  # leave the module as found
