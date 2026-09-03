"""What a chunk carrying an unsplittable image span costs.

The terminal-vs-transient test for such a span compares the cost against the widest chunk the
server could ever schedule. Charging only the span's own width makes a span that no pass can
hold look retryable, and ``try_add_one`` returning None breaks the admission loop -- so the
whole prefill queue stalls behind a request that will never be admitted.
"""
from __future__ import annotations

from freetoken.scheduler.prefill import PrefillAdder

UNIT = 64  # page_size on a hybrid model; 1 (no-op) everywhere else


def _adder() -> PrefillAdder:
    return PrefillAdder.__new__(PrefillAdder)  # the cost is pure arithmetic, reads no state


def test_a_mid_prompt_span_pays_the_head_and_the_tail():
    # span [100, 356): the chunk starts at 64 (the boundary below 100) and, being non-final,
    # ends at 384 (the first boundary at or after 356).
    assert _adder()._span_chunk_cost(100, 356, 100_000, UNIT) == 384 - 64


def test_a_span_at_the_prompt_end_pays_no_tail():
    # The chunk holding it reaches the prompt end, and a final chunk may end unaligned.
    assert _adder()._span_chunk_cost(100, 356, 356, UNIT) == 356 - 64


def test_an_aligned_span_costs_exactly_itself():
    assert _adder()._span_chunk_cost(128, 384, 100_000, UNIT) == 256


def test_unit_one_charges_exactly_the_span():
    # Non-hybrid models align to 1, where the old arithmetic was already right.
    assert _adder()._span_chunk_cost(100, 356, 100_000, 1) == 256


def test_the_cheaper_end_wins():
    # Prompt ends 8 tokens past the span: stopping there beats running on to the boundary.
    assert _adder()._span_chunk_cost(100, 356, 364, UNIT) == 364 - 64
