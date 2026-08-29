"""Presence penalty: a flat logit subtraction on tokens the request already generated.

The motivating failure is a reasoning model looping in a long agent turn -- repeating a
paragraph verbatim until it hits max_tokens. Model cards pair a high temperature with a
presence penalty to prevent exactly that (Ornith-1.5: temperature 1.0 + presence_penalty 1.5),
but generation_config.json can only carry the temperature, so the penalty half was missing.

Two properties matter and neither is obvious from the arithmetic:
  * the PROMPT is excluded -- penalizing a 100k-token agent prompt would cover most of the
    vocabulary, and the penalty would stop meaning "you already said this";
  * padding a ragged batch must not leak -- pad with a real token id and every short request
    in the batch silently penalizes it.
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


def _req(prompt: list[int], generated: list[int], penalty: float):
    """A Req as the sampler reads it: input_ids is prompt+generated, prompt_len splits them."""
    return SimpleNamespace(
        sampling_params=SamplingParams(temperature=1.0, presence_penalty=penalty),
        prompt_len=len(prompt),
        input_ids=torch.tensor(prompt + generated, dtype=torch.int32),
    )


def _batch(*reqs):
    return SimpleNamespace(reqs=list(reqs), size=len(reqs))


# --------------------------------------------------------------------- which tokens count
def test_only_generated_tokens_are_marked():
    _, seen = _sampler()._presence(_batch(_req([5, 6, 7], [11, 12, 11], 1.5)))
    assert seen[0].nonzero().flatten().tolist() == [11, 12]


def test_prompt_tokens_are_never_penalized():
    """The whole point: an agent prompt is most of the context and none of it is repetition."""
    _, seen = _sampler()._presence(_batch(_req([5, 6, 7], [11], 1.5)))
    assert seen[0, 5] == 0 and seen[0, 6] == 0 and seen[0, 7] == 0


def test_ragged_batch_padding_does_not_leak():
    """Pad with a real id and the short request in the batch penalizes that id for free."""
    _, seen = _sampler()._presence(
        _batch(_req([0], [11], 1.5), _req([0], [20, 21, 22, 23], 1.5))
    )
    assert seen[0].nonzero().flatten().tolist() == [11]
    assert seen[1].nonzero().flatten().tolist() == [20, 21, 22, 23]


# ------------------------------------------------------------------------ the cheap paths
def test_zero_penalty_allocates_nothing():
    assert _sampler()._presence(_batch(_req([0], [11], 0.0))) == (None, None)


def test_nothing_generated_yet_allocates_nothing():
    """Before the first sampled token there is nothing that could be a repeat."""
    assert _sampler()._presence(_batch(_req([1, 2], [], 1.5))) == (None, None)


def test_one_penalized_request_arms_the_whole_batch():
    pens, seen = _sampler()._presence(_batch(_req([0], [11], 0.0), _req([0], [22], 1.5)))
    assert pens.tolist() == [0.0, 1.5]
    # the un-penalized request still gets a (harmless) row, scaled by its own zero penalty
    assert seen.shape == (2, VOCAB)


# ------------------------------------------------------------------ the shift, and its effect
def test_the_shift_is_exactly_the_penalty():
    s = _sampler()
    args = s.prepare(_batch(_req([0], [11, 12], 2.0)))
    shifted = torch.zeros(1, VOCAB, device=DEV) - args.penalties[:, None] * args.seen
    assert shifted[0, 11].item() == pytest.approx(-2.0)
    assert shifted[0, 12].item() == pytest.approx(-2.0)
    assert shifted[0, 13].item() == pytest.approx(0.0)


def test_a_strong_penalty_beats_a_dominant_logit():
    """End to end through sample(): the runaway token is the most likely one by far, and the
    penalty has to be what stops it being picked."""
    s = _sampler()
    logits = torch.zeros(1, VOCAB, device=DEV)
    logits[0, 11] = 5.0

    penalized = _sampler().prepare(_batch(_req([0], [11], 8.0)))
    picks = [int(s.sample(logits.clone(), penalized)[0]) for _ in range(40)]
    assert picks.count(11) <= 2, picks.count(11)

    free = _sampler().prepare(_batch(_req([0], [11], 0.0)))
    picks_free = [int(s.sample(logits.clone(), free)[0]) for _ in range(40)]
    # exp(5)/(exp(5)+127) ~= 54% at temperature 1, so this is a wide but decisive margin
    assert picks_free.count(11) >= 12, picks_free.count(11)


def test_greedy_requests_still_get_the_penalty():
    """temperature=0 takes the argmax shortcut; the penalty must be applied before it, or a
    greedy deployment would loop with the flag silently doing nothing."""
    s = _sampler()
    logits = torch.zeros(1, VOCAB, device=DEV)
    logits[0, 11] = 5.0
    req = _req([0], [11], 8.0)
    req.sampling_params = SamplingParams(temperature=0.0, presence_penalty=8.0)
    args = s.prepare(_batch(req))
    assert args.temperatures is None, "this request must take the greedy path"
    assert int(s.sample(logits, args)[0]) != 11
