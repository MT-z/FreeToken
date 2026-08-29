"""`--sampling-override`: state which of a model card's presets this deployment serves.

``generation_config.json`` holds exactly one recommendation, but a model card often gives
several for different tasks -- Ornith-1.5 recommends temperature 1.0 in general and 0.6 for
precise coding, and only the first is in the checkpoint. That would not matter if clients sent
their own sampling fields, but the claude CLI sends only ``model``/``max_tokens``/``stream``/
``messages``/``system``/``tools``, so whatever the server resolves is what the model runs at.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from freetoken.server.args import parse_args
from freetoken.server.generation import resolve_sampling

ANON_PATH = "/models/anon"
_CFG = {"architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe"}
# parse_args reads the config both attribute-wise and via .to_dict() (the parser cascade).
CONFIG = SimpleNamespace(to_dict=lambda: _CFG, **_CFG)


def _args(*extra: str):
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: CONFIG):
        args, _ = parse_args(["--model", ANON_PATH, *extra])
    return args


# ------------------------------------------------------------------------ parsing
def test_absent_by_default():
    assert _args().sampling_override == {}


def test_single_and_repeated_keys():
    assert _args("--sampling-override", "temperature=0.6").sampling_override == {
        "temperature": 0.6
    }
    got = _args("--sampling-override", "temperature=0.6",
                "--sampling-override", "top_p=0.9").sampling_override
    assert got == {"temperature": 0.6, "top_p": 0.9}


def test_values_keep_their_types():
    got = _args("--sampling-override", "top_k=40").sampling_override
    assert got == {"top_k": 40} and isinstance(got["top_k"], int)


@pytest.mark.parametrize("bad,why", [
    ("temp=0.6", "an unknown key is a typo, not a new knob"),
    ("temperature", "no '=' at all"),
    ("temperature=abc", "unparseable value"),
    ("top_k=0.5", "int key given a float"),
])
def test_bad_overrides_fail_at_startup(bad, why):
    """Loudly, at parse time: a silently dropped override would serve the wrong preset for the
    life of the process with nothing in the log to say so."""
    with pytest.raises(ValueError, match="sampling-override"):
        _args("--sampling-override", bad)


# -------------------------------------------------------------- effect on a request
def _params(model_sampling: dict, **request_fields):
    fields = {"temperature": None, "top_k": None, "top_p": None,
              "max_tokens": 16, "ignore_eos": False}
    fields.update(request_fields)
    return resolve_sampling(model_sampling=model_sampling, **fields)


def test_override_replaces_the_checkpoint_recommendation():
    """What the layering is for: generation_config says 1.0, the deployment serves 0.6."""
    checkpoint = {"temperature": 1.0, "top_k": 20, "top_p": 0.95}
    merged = {**checkpoint, "temperature": 0.6}
    assert _params(merged).temperature == pytest.approx(0.6)
    # the keys it does not name are still the checkpoint's
    assert _params(merged).top_k == 20
    assert _params(merged).top_p == pytest.approx(0.95)


def test_an_explicit_request_value_still_wins():
    """The override fills *unspecified* fields only -- a client that does send temperature
    must keep getting exactly what it asked for."""
    merged = {"temperature": 0.6, "top_k": 20, "top_p": 0.95}
    assert _params(merged, temperature=0.0).temperature == pytest.approx(0.0)
    assert _params(merged, top_k=5).top_k == 5


def test_no_checkpoint_defaults_still_takes_the_override():
    """--sampling-defaults none plus an override: the override is the whole default."""
    assert _params({"temperature": 0.6}).temperature == pytest.approx(0.6)
    assert _params({}).temperature == pytest.approx(0.0)  # framework default
