"""Presence penalty: the protocol contract, and the one behaviour the greedy path can lose.

PR #393 replaced this branch's ``seen``-bitmap implementation with its LogitsPlan, which
covers presence + frequency + repetition + logit_bias + min_tokens. The arithmetic and the
batching edges (prompt exclusion, ragged padding) are pinned by
``tests/engine/test_logits_processors.py`` against that implementation, so the tests that
poked at ``args.penalties`` / ``args.seen`` are gone with the code they described.

What is kept here is what #393 does not assert:
  * the deployment-default path -- an omitted penalty must still fall through to
    ``--sampling-override presence_penalty=...``, which is how a model card's
    anti-repetition half gets served at all (Ornith-1.5: temperature 1.0 + 1.5).
    ``pick()`` reads a None sentinel, so the request model must default to None, not 0.0;
  * temperature=0 takes the argmax shortcut, and the penalty has to be applied before it,
    or a greedy deployment loops with the flag silently doing nothing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

from freetoken.core import SamplingParams
from freetoken.engine.sample import Sampler

DEV = torch.device("cuda")
VOCAB = 128


def _sampler() -> Sampler:
    return Sampler(DEV, VOCAB)


def _req(prompt: list[int], generated: list[int], penalty: float, temperature: float = 1.0):
    """A Req as #393's plan builder reads it: it recovers the prompt length from
    ``max_device_len - output_len``, so the stub carries those rather than prompt_len."""
    max_tokens = 4
    return SimpleNamespace(
        sampling_params=SamplingParams(
            temperature=temperature, presence_penalty=penalty, max_tokens=max_tokens
        ),
        input_ids=torch.tensor(prompt + generated, dtype=torch.int32),
        output_len=max_tokens,
        max_device_len=len(prompt) + max_tokens,
    )


def _batch(*reqs):
    return SimpleNamespace(reqs=list(reqs), size=len(reqs))


# ------------------------------------------------------------------- the greedy shortcut
def test_greedy_requests_still_get_the_penalty():
    """temperature=0 takes the argmax shortcut; the penalty must be applied before it, or a
    greedy deployment would loop with the flag silently doing nothing."""
    s = _sampler()
    logits = torch.zeros(1, VOCAB, device=DEV)
    logits[0, 11] = 5.0
    req = _req([0], [11], 8.0, temperature=0.0)
    args = s.prepare(_batch(req))
    assert args.temperatures is None, "this request must take the greedy path"
    assert int(s.sample(logits, args)[0]) != 11


# ------------------------------------------------------- the OpenAI protocol carries it
def test_an_openai_request_can_set_the_penalty():
    """/v1/chat/completions has a presence_penalty field and this server accepts it, so
    ignoring it was a silent drop -- the client set a value and got the deployment default."""
    from freetoken.server.generation import resolve_sampling

    sp = resolve_sampling(
        temperature=None, top_k=None, top_p=None, max_tokens=None, ignore_eos=False,
        model_sampling={"presence_penalty": 1.5}, presence_penalty=0.25,
    )
    assert sp.presence_penalty == 0.25, "an explicit request value wins"


def test_an_omitted_penalty_falls_through_to_the_deployment_default():
    """The reason the field is `float | None` and not `float = 0.0`: a request that never
    mentioned it must not override --sampling-override presence_penalty=1.5 with a protocol
    default of 0.0, which would disable the model card's anti-repetition recommendation for
    every client that does not know to ask for it."""
    from freetoken.server.generation import resolve_sampling

    sp = resolve_sampling(
        temperature=None, top_k=None, top_p=None, max_tokens=None, ignore_eos=False,
        model_sampling={"presence_penalty": 1.5}, presence_penalty=None,
    )
    assert sp.presence_penalty == 1.5


def test_an_explicit_zero_is_not_an_omission():
    """A client that deliberately sends 0.0 is turning the penalty OFF, and must be able to."""
    from freetoken.server.generation import resolve_sampling

    sp = resolve_sampling(
        temperature=None, top_k=None, top_p=None, max_tokens=None, ignore_eos=False,
        model_sampling={"presence_penalty": 1.5}, presence_penalty=0.0,
    )
    assert sp.presence_penalty == 0.0


def test_the_request_model_defaults_to_none_not_zero():
    from freetoken.server.api_models import ChatCompletionRequest, CompletionRequest

    assert ChatCompletionRequest(model="m", messages=[]).presence_penalty is None
    assert CompletionRequest(model="m", prompt="x").presence_penalty is None
