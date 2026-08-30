"""`--template-kwarg`: reach a chat template's controls when no protocol carries them.

A model card's reasoning controls are often template variables and nothing else. Qwen3.8-27B
grades its thinking as `reasoning_effort` in (low, medium, xhigh) and **defaults to xhigh**,
its longest; Ornith-1.5 offers only the `enable_thinking` on/off switch. Neither is a field in
the OpenAI or Anthropic wire format, and the claude CLI sends
`thinking: {"type": "adaptive"}` -- which names no state, so the adapter produces no kwargs at
all and the template's own default is the only reachable setting. That is how a long agent
session ends up on xhigh with no way to say otherwise.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import freetoken.server.generation as gen
from freetoken.core import SamplingParams
from freetoken.server.args import parse_args

ANON_PATH = "/models/anon"
_CFG = {"architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5"}
CONFIG = SimpleNamespace(to_dict=lambda: _CFG, **_CFG)


@pytest.fixture(autouse=True)
def _clean_defaults():
    """The defaults are process-wide; a leaked one would silently steer other tests."""
    gen.set_template_defaults({})
    yield
    gen.set_template_defaults({})


def _args(*extra: str):
    with patch("freetoken.utils.cached_load_hf_config", lambda _p: CONFIG):
        args, _ = parse_args(["--model", ANON_PATH, *extra])
    return args


def _spec(**kw):
    return gen.GenSpec(messages=[{"role": "user", "content": "hi"}],
                       sampling_params=SamplingParams(), **kw)


# ------------------------------------------------------------------------------- parsing
def test_absent_by_default():
    assert _args().template_kwarg == {}


def test_booleans_are_parsed_as_booleans():
    """`enable_thinking` is tested with `is false` in every template that has it; the string
    "false" is truthy in Jinja and would turn thinking ON."""
    got = _args("--template-kwarg", "enable_thinking=false").template_kwarg
    assert got == {"enable_thinking": False} and got["enable_thinking"] is False
    assert _args("--template-kwarg", "enable_thinking=true").template_kwarg["enable_thinking"] is True


def test_non_boolean_values_stay_strings():
    got = _args("--template-kwarg", "reasoning_effort=low").template_kwarg
    assert got == {"reasoning_effort": "low"} and isinstance(got["reasoning_effort"], str)


def test_repeated_flags_accumulate():
    got = _args("--template-kwarg", "enable_thinking=false",
                "--template-kwarg", "reasoning_effort=low").template_kwarg
    assert got == {"enable_thinking": False, "reasoning_effort": "low"}


def test_a_value_less_flag_fails_at_startup():
    with pytest.raises(ValueError, match="template-kwarg"):
        _args("--template-kwarg", "enable_thinking")


# ------------------------------------------------------------------- effect on a GenSpec
def test_defaults_reach_a_spec_that_states_nothing():
    """The claude CLI's shape: `adaptive` names no state, so the adapter builds {}."""
    gen.set_template_defaults({"enable_thinking": False})
    assert _spec().chat_template_kwargs == {"enable_thinking": False}


def test_a_request_that_states_a_mode_still_wins():
    """A client that does ask for thinking must get it, whatever the deployment prefers."""
    gen.set_template_defaults({"enable_thinking": False})
    got = _spec(chat_template_kwargs={"enable_thinking": True, "thinking_mode": "enabled"})
    assert got.chat_template_kwargs["enable_thinking"] is True


def test_unrelated_request_keys_are_preserved_alongside_the_defaults():
    gen.set_template_defaults({"reasoning_effort": "low"})
    got = _spec(chat_template_kwargs={"thinking_mode": "enabled"})
    assert got.chat_template_kwargs == {"reasoning_effort": "low", "thinking_mode": "enabled"}


def test_no_defaults_leaves_the_spec_untouched():
    assert _spec().chat_template_kwargs == {}
    assert _spec(chat_template_kwargs={"a": 1}).chat_template_kwargs == {"a": 1}


def test_turning_thinking_off_also_disables_the_reasoning_parser():
    """The gate downstream reads chat_template_kwargs directly. Merging anywhere later than
    GenSpec would leave it looking at the unmerged dict, so the parser would keep hunting for
    a <think> block the template just suppressed."""
    gen.set_template_defaults({"enable_thinking": False})
    spec = _spec()
    # the gate, spelled exactly as generation.py spells it
    force_reasoning = (spec.chat_template_kwargs or {}).get("enable_thinking") is not False
    assert force_reasoning is False
