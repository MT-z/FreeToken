"""Per-token termination, and a gamma=1 verify cycle committing through the same code.

``_process_last_data`` used to carry the append / stop-evaluation / retire-or-publish body
inline, one token per request per forward. A speculative cycle produces up to gamma+1
tokens from a single forward, and the spec (Sec 8.3) requires them to go through the
*existing* per-token rules in token order rather than a copy of them -- ``_match_stop_str``
decodes the tail of ``req.input_ids``, ``hit_length`` moves one position per token, and
``toolcall_anchor_len`` is the history length AT its token. So the body is now
``_commit_token``, called once by the normal drain and up to twice by
``_commit_speculative``.

These tests drive the real unbound ``Scheduler`` methods against CPU-built hybrid managers,
like the repo's other scheduler tests. The tokenizer is a fake (one char per token id): what
is under test is WHEN a stop string is visible, not how text decodes.

Every test here fails against at least one single-line mutation of the rules it covers; the
normal-path and speculative-path tests are deliberately not independent, because sharing one
implementation is the property being checked.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import PrefillManager
from freetoken.scheduler.scheduler import Scheduler
from freetoken.scheduler.table import TableManager

UID = 7
EOS = ord("Z")
PROMPT = torch.tensor([ord(c) for c in "hello"], dtype=torch.int32)


def _setup(*, eos=(EOS,)):
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    pool = LinearStatePool(group=g, num_slots=16, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    tm = TableManager(max_running_reqs=4, page_table=pt)
    dm = DecodeManager(page_size=1)
    published: list[tuple[int, bool]] = []
    real_cache_req = cm.cache_req

    def cache_req(req, *, finished):           # record, then run the real thing
        published.append((req.cached_len, finished))
        return real_cache_req(req, finished=finished)

    cm.cache_req = cache_req
    sent: list = []
    stub = SimpleNamespace(
        cache_manager=cm, table_manager=tm, decode_manager=dm,
        prefill_manager=PrefillManager(cm, tm, dm),
        finished_reqs=set(), eos_token_ids=set(eos), toolcall_anchor_id=None,
        tokenizer=SimpleNamespace(decode=lambda ids: "".join(chr(i) for i in ids)),
        config=SimpleNamespace(page_size=1),
        status_reporter=SimpleNamespace(report_batch=lambda *_, **__: None),
        send_result=sent.extend,
        _kv_usage_pages=cm.page_usage, _mamba_slot_usage=lambda: None,
        _swa_token_usage=lambda: None, _gpu_mem_bytes=lambda: 0,
        _pending_abort_acks=set(), _last_data=None,
    )
    for name in ("_free_req_resources", "_retire_req", "_commit_token",
                 "_commit_speculative", "_publish_replies", "_match_stop_str"):
        fn = getattr(Scheduler, name)
        setattr(stub, name, (lambda f: lambda *a, **kw: f(stub, *a, **kw))(fn))
    return pool, cm, tm, dm, stub, sent, published


def _req(cm, tm, *, output_len=4, **sp):
    mr = cm.match_req(SimpleNamespace(input_ids=PROMPT, input_len=len(PROMPT)))
    req = Req(input_ids=PROMPT.clone(), table_idx=tm.allocate(), cached_len=0,
              output_len=output_len, uid=UID,
              sampling_params=SamplingParams(max_tokens=output_len, **sp),
              cache_handle=mr.cuda_handle)
    req.linear_slot_idx = cm.linear_state_pool.alloc(1)[0]
    req.mamba_track_slots = tuple(cm.linear_state_pool.alloc(2))
    req.mamba_track_seqlens = (None, None)
    req.mamba_next_track_idx = 1
    cm.lock(mr.cuda_handle)
    return req


def _step(cm, stub, req, token, *, publish_prefix=False, reply=None, done=None):
    """One normal forward+drain: allocate the extend window, advance, commit one token."""
    cm.allocate_paged([req])          # scheduler, at batch build
    req.complete_one()                # engine, inside forward_batch
    with cm.lazy_free_region():
        return stub._commit_token(
            req, torch.tensor(token, dtype=torch.int32),
            reply=reply if reply is not None else [],
            new_finished_reqs=done if done is not None else set(),
            publish_prefix=publish_prefix,
        )


def _step_raw(cm, stub, req, token):
    """Commit one token with the request left exactly as the caller set it up."""
    with cm.lazy_free_region():
        return stub._commit_token(
            req, torch.tensor(token, dtype=torch.int32),
            reply=[], new_finished_reqs=set(), publish_prefix=False)


def _cycle(cm, stub, req, tokens):
    """One gamma=1 verify cycle: two positions computed, `tokens` committed."""
    back = req.device_len
    req.device_len = req.cached_len + 2      # verify's extend window (harness does the same)
    cm.allocate_paged([req])
    req.device_len = back
    return stub._commit_speculative(
        req, torch.tensor(tokens, dtype=torch.int32), batch=Batch(reqs=[req], phase="decode")
    )


# --------------------------------------------------------------------- normal path

def test_eos_finishes_with_stop_and_retires_the_request():
    _pool, cm, tm, dm, stub, _sent, _pub = _setup()
    req = _req(cm, tm)
    dm.filter_reqs([req])
    reply, done = [], set()
    assert not _step(cm, stub, req, ord("a"), reply=reply, done=done)
    assert _step(cm, stub, req, EOS, reply=reply, done=done)
    assert [(m.next_token, m.finished, m.finish_reason) for m in reply] == [
        (ord("a"), False, None), (EOS, True, "stop")]
    assert req in done and req not in dm.running_reqs and req.table_idx == -1


def test_budget_exhaustion_finishes_with_length():
    _pool, cm, tm, dm, stub, _sent, _pub = _setup()
    req = _req(cm, tm, output_len=2)
    dm.filter_reqs([req])
    reply = []
    assert not _step(cm, stub, req, ord("a"), reply=reply)
    assert _step(cm, stub, req, ord("b"), reply=reply)
    assert reply[-1].finish_reason == "length"


def test_eos_on_the_last_slot_reports_stop_not_length():
    """Both conditions fire on the same token; the comment above the block says EOS wins."""
    _pool, cm, tm, dm, stub, _sent, _pub = _setup()
    req = _req(cm, tm, output_len=1)
    dm.filter_reqs([req])
    reply = []
    assert _step(cm, stub, req, EOS, reply=reply)
    assert reply[-1].finish_reason == "stop" and reply[-1].matched_stop is None


def test_stop_string_is_invisible_until_its_last_token_is_appended():
    """`_match_stop_str` reads req.input_ids, so it can only see appended tokens. "END"
    must not match on 'E' or 'EN', and must match on the 'D' -- which is what forces the
    speculative path to evaluate token by token instead of after both appends."""
    _pool, cm, tm, dm, stub, _sent, _pub = _setup()
    req = _req(cm, tm, output_len=4, stop_strs=["END"])
    dm.filter_reqs([req])
    reply = []
    assert not _step(cm, stub, req, ord("E"), reply=reply)
    assert not _step(cm, stub, req, ord("N"), reply=reply)
    assert _step(cm, stub, req, ord("D"), reply=reply)
    assert [m.matched_stop for m in reply] == [None, None, "END"]
    assert reply[-1].finish_reason == "stop"


def test_anchor_is_not_recorded_on_the_token_that_finishes():
    _pool, cm, tm, dm, stub, _sent, _pub = _setup()
    stub.toolcall_anchor_id = ord("T")
    req = _req(cm, tm, output_len=2)
    dm.filter_reqs([req])
    _step(cm, stub, req, ord("a"))
    assert req.toolcall_anchor_len is None
    assert _step(cm, stub, req, ord("T"))          # the anchor token also exhausts the budget
    assert req.toolcall_anchor_len is None, "an anchor on a finished request has no reuse point"


def test_anchor_records_the_history_length_at_its_own_token():
    _pool, cm, tm, dm, stub, _sent, _pub = _setup()
    stub.toolcall_anchor_id = ord("T")
    req = _req(cm, tm, output_len=4)
    dm.filter_reqs([req])
    _step(cm, stub, req, ord("T"))
    assert req.toolcall_anchor_len == len(PROMPT) + 1
    _step(cm, stub, req, ord("b"))
    assert req.toolcall_anchor_len == len(PROMPT) + 1, "set once, at the first opener"


def test_prefill_publishes_the_prefix_and_a_decode_step_does_not():
    _pool, cm, tm, dm, stub, _sent, published = _setup()
    req = _req(cm, tm)
    dm.filter_reqs([req])
    _step(cm, stub, req, ord("a"), publish_prefix=True)
    assert published == [(len(PROMPT), False)]
    _step(cm, stub, req, ord("b"), publish_prefix=False)
    assert published == [(len(PROMPT), False)], "a decode step publishes nothing"


# ------------------------------------------------------------------ speculative path

def test_accepted_pair_commits_both_tokens_in_order():
    _pool, cm, tm, dm, stub, sent, _pub = _setup()
    req = _req(cm, tm, output_len=4)
    dm.filter_reqs([req])
    assert _cycle(cm, stub, req, [ord("a"), ord("b")]) == 2
    assert [m.next_token for m in sent] == [ord("a"), ord("b")]
    assert req.input_ids.tolist() == PROMPT.tolist() + [ord("a"), ord("b")]
    # the commit contract: history one ahead of the computed frontier, KV one past that
    assert req.cached_len == req.input_ids.numel() - 1 == len(PROMPT) + 1
    assert req.device_len == req.input_ids.numel()


def test_bonus_is_never_appended_when_the_draft_ends_the_request():
    """EOS on t_{p+2}. The bonus is not retracted after the fact -- it never enters the
    history, the reply stream, or the committed KV range."""
    _pool, cm, tm, dm, stub, sent, _pub = _setup()
    req = _req(cm, tm, output_len=4)
    dm.filter_reqs([req])
    assert _cycle(cm, stub, req, [EOS, ord("b")]) == 1
    assert [(m.next_token, m.finish_reason) for m in sent] == [(EOS, "stop")]
    assert req.input_ids.tolist() == PROMPT.tolist() + [EOS]
    assert req not in dm.running_reqs and req.table_idx == -1


def test_one_slot_left_makes_the_draft_the_last_token():
    """Remaining budget 1. No gamma_eff logic in the commit: hit_length fires on the draft
    because complete_one moved the counter, and the bonus never runs."""
    _pool, cm, tm, dm, stub, sent, _pub = _setup()
    req = _req(cm, tm, output_len=1)
    dm.filter_reqs([req])
    assert _cycle(cm, stub, req, [ord("a"), ord("b")]) == 1
    assert [(m.next_token, m.finish_reason) for m in sent] == [(ord("a"), "length")]


def test_two_slots_left_publishes_both_and_stops_on_the_bonus():
    _pool, cm, tm, dm, stub, sent, _pub = _setup()
    req = _req(cm, tm, output_len=2)
    dm.filter_reqs([req])
    assert _cycle(cm, stub, req, [ord("a"), ord("b")]) == 2
    assert [(m.next_token, m.finish_reason) for m in sent] == [
        (ord("a"), None), (ord("b"), "length")]


def test_a_stop_string_completed_by_the_draft_stops_before_the_bonus():
    """The pair would both be published if the stop check ran after both appends. "EN"+"D":
    committing in token order ends the request on the draft."""
    _pool, cm, tm, dm, stub, sent, _pub = _setup()
    req = _req(cm, tm, output_len=4, stop_strs=["END"])
    dm.filter_reqs([req])
    _step(cm, stub, req, ord("E"))
    _step(cm, stub, req, ord("N"))
    assert _cycle(cm, stub, req, [ord("D"), ord("x")]) == 1
    assert [(m.next_token, m.matched_stop) for m in sent] == [(ord("D"), "END")]
    assert req.input_ids.tolist()[-1] == ord("D")


def test_a_stop_string_completed_by_the_bonus_publishes_both():
    _pool, cm, tm, dm, stub, sent, _pub = _setup()
    req = _req(cm, tm, output_len=4, stop_strs=["END"])
    dm.filter_reqs([req])
    _step(cm, stub, req, ord("E"))
    assert _cycle(cm, stub, req, [ord("N"), ord("D")]) == 2
    assert [(m.next_token, m.matched_stop) for m in sent] == [
        (ord("N"), None), (ord("D"), "END")]


def test_rejected_cycle_leaves_the_draft_position_outside_the_committed_range():
    """One committed token from a two-position verify. cached_len must stop at the
    corrected token's own position, so the KV the verify wrote from the draft is outside
    [0, cached_len) -- recomputed by the next forward, and never published as prefix."""
    _pool, cm, tm, dm, stub, sent, published = _setup()
    req = _req(cm, tm, output_len=4)
    dm.filter_reqs([req])
    assert _cycle(cm, stub, req, [ord("x")]) == 1
    assert req.input_ids.tolist() == PROMPT.tolist() + [ord("x")]
    assert req.cached_len == len(PROMPT)
    assert published == [], "mid-decode commits publish nothing"
    # and the next forward recomputes exactly that position
    assert req.device_len - req.cached_len == 1


def test_an_aborted_cycle_publishes_no_token_and_no_prefix():
    _pool, cm, tm, dm, stub, sent, published = _setup()
    req = _req(cm, tm, output_len=4)
    dm.filter_reqs([req])
    req.aborted = True
    assert _cycle(cm, stub, req, [ord("a"), ord("b")]) == 0
    assert sent == [] and req.input_ids.tolist() == PROMPT.tolist()
    assert published == [(0, True)], "released, and the release publishes nothing new"
    assert req.table_idx == -1 and req not in dm.running_reqs


# ------------------------------------------------------- the contract refuses bad callers

def test_speculative_commit_refuses_a_stale_frontier():
    """verify_forward does not call complete_one; the speculative commit makes that call and
    checks it landed. The check lives there, not in _commit_token: the normal drain runs
    under overlap scheduling, where the NEXT batch's complete_one has already moved
    cached_len one past the history by the time this token is committed."""
    _pool, cm, tm, dm, stub, _sent, _pub = _setup()
    req = _req(cm, tm)
    dm.filter_reqs([req])
    req.complete_one()                      # a forward already advanced it ...
    try:
        _cycle(cm, stub, req, [ord("a")])   # ... and the cycle advances it again
    except AssertionError as e:
        assert "frontier" in str(e), e
    else:
        raise AssertionError("a stale frontier was accepted")


def test_the_normal_drain_accepts_an_overlapped_request():
    """The real cost of getting the contract wrong: overlap_loop launches batch N+1 BEFORE
    draining batch N, so at commit time cached_len is one AHEAD of the history. An assert
    demanding the strict relation here takes down ordinary serving -- it did, on a chunked
    prefill's commit, the first time a real generation was run through it."""
    _pool, cm, tm, dm, stub, _sent, _pub = _setup()
    req = _req(cm, tm)
    dm.filter_reqs([req])
    cm.allocate_paged([req])
    req.complete_one()                      # this batch's forward
    req.complete_one()                      # the NEXT batch, launched before this drain
    assert req.cached_len == req.input_ids.numel() + 1
    assert not _step_raw(cm, stub, req, ord("a"))


def test_speculative_commit_refuses_a_logprobs_request():
    _pool, cm, tm, dm, stub, _sent, _pub = _setup()
    req = _req(cm, tm, logprobs=True)
    dm.filter_reqs([req])
    try:
        _cycle(cm, stub, req, [ord("a")])
    except AssertionError as e:
        assert "logprobs" in str(e)
    else:
        raise AssertionError("a logprobs request went through the verify commit")
