"""P2b integration: CacheManager hybrid path (match_req -> cache_req donate -> prefix hit).
CPU, real LinearStatePool + page_table, hand-built Reqs. Exercises the two-currency wiring
without the full scheduler/engine."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager


def _pool(num_slots=16):
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    return LinearStatePool(group=g, num_slots=num_slots, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)


def _pend(ids):
    # int32 to match production Req.input_ids dtype (fast_compare_key needs consistent dtype)
    t = torch.tensor(ids, dtype=torch.int32)
    return SimpleNamespace(input_ids=t, input_len=len(ids), mm_embeds=None)


def test_hybrid_cache_manager_donate_then_hit():
    pool = _pool()
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)
    assert cm.is_hybrid

    # cold match on an empty tree
    mr = cm.match_req(_pend([1, 2, 3, 4, 5]))
    assert mr.cuda_handle.cached_len == 0 and mr.mamba_value is None

    # admit req A: allocate live + ping-pong, stage KV pages, mark a ×N snapshot at boundary 4
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    page_table[0, :4] = torch.tensor([100, 101, 102, 103], dtype=torch.int32)
    reqA = Req(input_ids=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32), table_idx=0,
               cached_len=4, output_len=1, uid=0, sampling_params=SamplingParams(),
               cache_handle=mr.cuda_handle)
    reqA.linear_slot_idx, reqA.mamba_ping_pong = live, pp
    reqA.mamba_next_track_idx = 1            # flipped from 0 in build_fla_metadata; frozen = pp[0]
    reqA.mamba_last_track_seqlen = 4
    cm.lock(mr.cuda_handle)

    clone_before = set(pool._free_slots)
    free_before = pool.num_free_slots
    cm.cache_req(reqA, finished=False)       # copy-on-donate: the request keeps pp[0]; the tree
    # gets a PRIVATE CLONE of it, allocated from the free list. The request's slot is NOT
    # replaced (no shared slot id), so the only free-list delta is the clone itself.
    assert pool.num_free_slots == free_before - 1  # one clone alloc'd (node was new: not freed back)
    clone_ids = clone_before - set(pool._free_slots)
    assert len(clone_ids) == 1
    clone_id = next(iter(clone_ids))
    assert reqA.mamba_ping_pong[0] == pp[0]        # request keeps its own slot (no replacement alloc)

    # req B shares the [1,2,3,4] prefix -> HIT: the tree's snapshot is the private CLONE,
    # not the request's own slot.
    mrB = cm.match_req(_pend([1, 2, 3, 4, 9]))
    assert mrB.cuda_handle.cached_len == 4
    assert mrB.mamba_value == clone_id
    assert mrB.cuda_handle.get_matched_indices().tolist() == [100, 101, 102, 103]


def test_hybrid_finish_donates_live_slot():
    pool = _pool()
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)

    mr = cm.match_req(_pend([7, 8, 9, 10]))
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    page_table[1, :3] = torch.tensor([200, 201, 202], dtype=torch.int32)
    req = Req(input_ids=torch.tensor([7, 8, 9, 10], dtype=torch.int32), table_idx=1,
              cached_len=3, output_len=1, uid=1, sampling_params=SamplingParams(),
              cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    cm.lock(mr.cuda_handle)

    clone_before = set(pool._free_slots)
    cm.cache_req(req, finished=True)         # copy-on-donate: the tree gets a PRIVATE CLONE of
    # the live (final-state) slot; the request's live + ping-pong slots are freed.
    clone_after = set(pool._free_slots)
    clone_ids = clone_before - clone_after
    assert len(clone_ids) == 1                       # exactly one CLONE allocated for the node
    clone_id = next(iter(clone_ids))
    mr2 = cm.match_req(_pend([7, 8, 9, 10]))
    assert mr2.cuda_handle.cached_len == 3 and mr2.mamba_value == clone_id


def test_free_req_slots_idempotent():
    """C2: a finish/abort double-free of the same request must NOT push its GDN slots twice."""
    pool = _pool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    req = Req(input_ids=torch.tensor([1, 2, 3], dtype=torch.int32), table_idx=0, cached_len=2,
              output_len=1, uid=0, sampling_params=SamplingParams(), cache_handle=None)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    base = pool.num_free_slots
    cm._free_req_slots(req)
    assert pool.num_free_slots == base + 3        # live + 2 ping-pong returned once
    cm._free_req_slots(req)                        # second free (abort/finish race)
    assert pool.num_free_slots == base + 3         # idempotent: nothing pushed twice


def test_rebuild_reclaims_donated_gdn_slots():
    """C5: a runtime cache rebuild must return the discarded tree's GDN snapshot slots (idle)."""
    pool = _pool(num_slots=16)
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    mr = cm.match_req(_pend([7, 8, 9, 10]))
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    pt[1, :3] = torch.tensor([200, 201, 202], dtype=torch.int32)
    req = Req(input_ids=torch.tensor([7, 8, 9, 10], dtype=torch.int32), table_idx=1, cached_len=3,
              output_len=1, uid=1, sampling_params=SamplingParams(), cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    cm.lock(mr.cuda_handle)
    cm.cache_req(req, finished=True)              # donates `live` to the tree, frees ping-pong
    assert pool.num_free_slots < pool.num_slots - 1   # a slot is now tree-owned
    cm.rebuild(64, pt)                            # idle rebuild discards the tree
    assert pool.num_free_slots == pool.num_slots - 1  # all GDN slots reclaimed (no leak)


def test_prefill_chunk_ends_on_a_page_boundary():
    """A hybrid chunk must end page-aligned: the snapshot commit skips any other boundary."""
    from freetoken.scheduler.prefill import ChunkedReq, PrefillAdder
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pool = _pool()
    pt = torch.zeros(4, 512, dtype=torch.int32)
    cm = CacheManager(64, 64, pt, "hybrid_radix", linear_state_pool=pool)
    assert cm.prefill_chunk_align == 64
    tm = TableManager(max_running_reqs=4, page_table=pt)
    pending = PendingReq(0, torch.arange(300, dtype=torch.int32), SamplingParams(max_tokens=1))

    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    req = adder.try_add_one(pending)
    assert isinstance(req, ChunkedReq) and req.extend_len == 64

    # a budget below one page keeps the unaligned chunk rather than stalling the request
    adder = PrefillAdder(token_budget=40, reserved_size=0, cache_manager=cm, table_manager=tm)
    assert adder.try_add_one(pending).extend_len == 40


def test_naive_cache_does_not_align_prefill_chunks():
    """The alignment hook is hybrid-only; every other cache keeps the raw budget chunk."""
    from freetoken.scheduler.prefill import PrefillAdder
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pt = torch.zeros(4, 512, dtype=torch.int32)
    cm = CacheManager(64, 64, pt, "radix")
    assert cm.prefill_chunk_align == 1
    tm = TableManager(max_running_reqs=4, page_table=pt)
    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    pending = PendingReq(0, torch.arange(300, dtype=torch.int32), SamplingParams(max_tokens=1))
    assert adder.try_add_one(pending).extend_len == 100


def test_pool_sizing_covers_4mr_floor():
    """C6: pool must reserve the 4-slot-per-request non-evictable floor even at a tiny ratio."""
    from types import SimpleNamespace
    from freetoken.kvcache.linear_state_pool import _linear_pool_num_slots
    for mr in (1, 8, 64):
        c = SimpleNamespace(max_running_req=mr, cache_type="hybrid_radix",
                            linear_state_cache_ratio=0.1)
        assert _linear_pool_num_slots(c) >= 4 * mr + 1, (mr, _linear_pool_num_slots(c))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")


def _slot_val(pool, slot):
    return pool.recurrent_states[0, slot].flatten()[0].item()


def _admit_cold(cm, pool, pt, ids, row):
    """Hand-build a request that was admitted COLD (no tree match) and has finished its final
    prefill chunk: all `len(ids)` tokens computed into pages taken off the manager's free list.
    Returns (req, ping_pong, pages)."""
    mr = cm.match_req(_pend(ids))
    assert mr.cuda_handle.cached_len == 0 and mr.mamba_value is None
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    pool.recurrent_states[0, pp[0]].fill_(1.0)   # face 0 -> value 1.0
    pool.recurrent_states[0, pp[1]].fill_(2.0)   # face 1 -> value 2.0
    pages = cm.free_slots[: len(ids)].clone()
    cm.free_slots = cm.free_slots[len(ids) :]
    pt[row, : len(ids)] = pages
    req = Req(input_ids=torch.tensor(ids, dtype=torch.int32), table_idx=row, cached_len=0,
              output_len=1, uid=row, sampling_params=SamplingParams(), cache_handle=mr.cuda_handle)
    req.complete_one()                            # the prefill forward ran: cached_len == len(ids)
    assert req.cached_len == len(ids)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    cm.lock(mr.cuda_handle)
    return req, pp, pages.tolist()


def test_hybrid_chunk_commit_donates_the_previous_face_too():
    """Chunked prompt [0,6)+[6,10): chunk 1 tracked boundary 4 into face 0 and skipped cache_req
    (the continuation carried it as mamba_prev_track_seqlen); chunk 2 tracked boundary 8 into
    face 1. The final-chunk commit must donate BOTH faces, so a branch inside the final chunk
    (divergence at 6) reuses up to 4 instead of finding no live snapshot at all."""
    pool = _pool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    ids = list(range(1, 11))
    req, pp, pages = _admit_cold(cm, pool, pt, ids, row=0)
    req.mamba_prev_track_seqlen = 4      # face 0 (written by chunk 1, next 0 -> 1)
    req.mamba_last_track_seqlen = 8      # face 1 (written by chunk 2, next 1 -> 0)
    req.mamba_next_track_idx = 0
    free_slots_before, free_pages_before = pool.num_free_slots, len(cm.free_slots)
    cm.cache_req(req, finished=False)
    assert pool.num_free_slots == free_slots_before - 2      # two private clones, nothing else
    assert len(cm.free_slots) == free_pages_before            # cold tree: no dup pages to free
    assert req.mamba_ping_pong == pp                          # request keeps both faces
    assert req.mamba_prev_track_seqlen is None and req.mamba_last_track_seqlen is None
    assert req.cache_handle.cached_len == 8
    # branch inside the final chunk -> the previous face's boundary
    m4 = cm.match_req(_pend(ids[:6] + [99]))
    assert m4.cuda_handle.cached_len == 4
    assert _slot_val(pool, m4.mamba_value) == 1.0             # clone of face 0, not face 0 itself
    assert m4.mamba_value not in pp
    assert m4.cuda_handle.get_matched_indices().tolist() == pages[:4]
    # branch after the last boundary -> the frozen face, as before
    m8 = cm.match_req(_pend(ids[:9] + [99]))
    assert m8.cuda_handle.cached_len == 8
    assert _slot_val(pool, m8.mamba_value) == 2.0
    assert m8.cuda_handle.get_matched_indices().tolist() == pages[:8]
    # integrity balances only once no request is in flight: finish it (live-slot donate)
    cm.cache_req(req, finished=True)
    cm.check_integrity()


def test_hybrid_chunk_commit_donates_only_the_previous_face_when_the_final_chunk_crossed_none():
    """A final chunk shorter than one GDN chunk tracks nothing (L is None): the commit used to
    return early and the carried boundary died with the request. Now the un-flipped face
    (1 - next) is donated on its own."""
    pool = _pool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    ids = list(range(1, 11))
    req, _pp, _pages = _admit_cold(cm, pool, pt, ids, row=0)
    req.mamba_prev_track_seqlen = 4      # face 0 (chunk 1 wrote it, next 0 -> 1); no write since
    req.mamba_last_track_seqlen = None
    req.mamba_next_track_idx = 1
    free_before = pool.num_free_slots
    cm.cache_req(req, finished=False)
    assert pool.num_free_slots == free_before - 1
    assert req.cache_handle.cached_len == 4
    assert req.mamba_prev_track_seqlen is None
    m = cm.match_req(_pend(ids[:6] + [99]))
    assert m.cuda_handle.cached_len == 4 and _slot_val(pool, m.mamba_value) == 1.0
    cm.cache_req(req, finished=True)
    cm.check_integrity()


def test_hybrid_chunk_commit_dedup_floor_with_a_pre_existing_shorter_branch():
    """Concurrent shape: A was admitted cold, then B donated [1..3] before A's final chunk
    committed. A's previous-face donation (boundary 4) walks B's nodes for [0,3) and builds
    [3,4) on A's own page; its frozen-face donation (8) then walks THAT node and builds [4,8).
    The dedup floor must be 3 -- the first insert's prefix_len -- not the second insert's 4:
    freeing against 4 would free A's page 3 from under the tree node that now owns it. The old
    single-donation commit could not reach 4 at all (a branch at 6 fell back to B's 3)."""
    pool = _pool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    ids = list(range(1, 11))
    reqA, _ppA, pagesA = _admit_cold(cm, pool, pt, ids, row=0)
    # B: [1..4] with a tracked boundary at 3, committed while A is still in flight
    reqB, _ppB, pagesB = _admit_cold(cm, pool, pt, ids[:4], row=1)
    reqB.mamba_last_track_seqlen, reqB.mamba_next_track_idx = 3, 1
    cm.cache_req(reqB, finished=False)
    assert cm.match_req(_pend(ids[:6] + [99])).cuda_handle.cached_len == 3
    reqA.mamba_prev_track_seqlen, reqA.mamba_last_track_seqlen, reqA.mamba_next_track_idx = 4, 8, 0
    free_slots_before, free_pages_before = pool.num_free_slots, len(cm.free_slots)
    cm.cache_req(reqA, finished=False)
    assert pool.num_free_slots == free_slots_before - 2       # both clones taken (new nodes)
    assert len(cm.free_slots) == free_pages_before + 3         # A's dups [0,3) freed, page 3 kept
    expect = pagesB[:3] + pagesA[3:8]
    assert pt[0, :8].tolist() == expect
    m4 = cm.match_req(_pend(ids[:6] + [99]))
    assert m4.cuda_handle.cached_len == 4                      # was 3: the previous face reaches it
    assert _slot_val(pool, m4.mamba_value) == 1.0
    assert m4.cuda_handle.get_matched_indices().tolist() == expect[:4]
    m8 = cm.match_req(_pend(ids[:9] + [99]))
    assert m8.cuda_handle.cached_len == 8
    assert m8.cuda_handle.get_matched_indices().tolist() == expect
    assert _slot_val(pool, m8.mamba_value) == 2.0              # A's face 1 clone at 8
    cm.cache_req(reqA, finished=True)
    cm.cache_req(reqB, finished=True)
    cm.check_integrity()


def test_prefill_continuation_carries_the_uncommitted_track_boundary():
    """The scheduler skips cache_req for intermediate chunks, so the boundary a chunk tracked
    lives only in its ping-pong face. PrefillAdder must hand it to the continuation as
    mamba_prev_track_seqlen -- and pass an older one through when a chunk tracked nothing."""
    from freetoken.scheduler.prefill import ChunkedReq, PrefillAdder
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pool = _pool()
    pt = torch.zeros(4, 512, dtype=torch.int32)
    cm = CacheManager(64, 64, pt, "hybrid_radix", linear_state_pool=pool)
    tm = TableManager(max_running_reqs=4, page_table=pt)
    pending = PendingReq(0, torch.arange(300, dtype=torch.int32), SamplingParams(max_tokens=1))

    def add(budget):
        return PrefillAdder(token_budget=budget, reserved_size=0, cache_manager=cm,
                            table_manager=tm).try_add_one(pending)

    chunk = add(128)
    assert isinstance(chunk, ChunkedReq) and chunk.extend_len == 128
    assert chunk.mamba_prev_track_seqlen is None              # fresh admission carries nothing
    # forward: build_fla_metadata tracked 64 into face 0 and flipped; scheduler skipped cache_req
    chunk.mamba_last_track_seqlen, chunk.mamba_next_track_idx = 64, 1
    chunk.cached_len = chunk.device_len
    pending.chunked_req = chunk
    cont = add(64)
    assert isinstance(cont, ChunkedReq) and cont.cached_len == 128
    assert cont.mamba_prev_track_seqlen == 64 and cont.mamba_last_track_seqlen is None
    assert cont.mamba_next_track_idx == 1 and cont.mamba_ping_pong == chunk.mamba_ping_pong
    # this chunk crossed no boundary (extend 64: c = 63 // 64 = 0): the older one passes through
    cont.cached_len = cont.device_len
    pending.chunked_req = cont
    last = add(512)
    assert not isinstance(last, ChunkedReq) and last.cached_len == 192
    assert last.mamba_prev_track_seqlen == 64 and last.mamba_next_track_idx == 1


# ---------------------------------------------------------------- fork-boundary tracking (c)

def test_match_reports_the_token_match_beyond_the_snapshot_truncation():
    """Tree holds [1..8] with snapshots at 4 and 8 (the two-face donation above). A request
    sharing [1..6] matches 6 tokens but can only resume from 4: cached_len 4, tok_match 6.
    A cold request reports 0/0; one whose match ends exactly on a snapshot reports equal."""
    pool = _pool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    ids = list(range(1, 11))
    req, _pp, _pages = _admit_cold(cm, pool, pt, ids, row=0)
    req.mamba_prev_track_seqlen, req.mamba_last_track_seqlen, req.mamba_next_track_idx = 4, 8, 0
    cm.cache_req(req, finished=False)
    m = cm.match_req(_pend(ids[:6] + [99, 100]))          # key = first 7 ids -> matches 6
    assert (m.cuda_handle.cached_len, m.tok_match) == (4, 6)
    m = cm.match_req(_pend(ids[:4] + [99, 100]))          # match ends on the snapshot
    assert (m.cuda_handle.cached_len, m.tok_match) == (4, 4)
    m = cm.match_req(_pend([50, 51, 52]))                 # cold
    assert (m.cuda_handle.cached_len, m.tok_match) == (0, 0)


def test_prefill_admission_records_the_fork_and_carries_it_across_chunks():
    """PrefillAdder turns tok_match > cached_len into Req.mamba_fork_len (None when the match
    ends on a snapshot or the request is cold) and hands it on to continuation chunks."""
    from freetoken.scheduler.prefill import ChunkedReq, PrefillAdder
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pool = _pool(num_slots=32)               # four admissions x 3 slots + donor + clones
    pt = torch.zeros(8, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    tm = TableManager(max_running_reqs=8, page_table=pt)
    ids = list(range(1, 11))
    donor, _pp, _pages = _admit_cold(cm, pool, pt, ids, row=0)
    donor.mamba_prev_track_seqlen, donor.mamba_last_track_seqlen, donor.mamba_next_track_idx = 4, 8, 0
    cm.cache_req(donor, finished=False)

    def admit(prompt, budget):
        pend = PendingReq(1, torch.tensor(prompt, dtype=torch.int32), SamplingParams(max_tokens=1))
        adder = PrefillAdder(token_budget=budget, reserved_size=0, cache_manager=cm, table_manager=tm)
        return pend, adder.try_add_one(pend)

    # branches at 6: resumes from 4, fork recorded at 6
    _, req = admit(ids[:6] + [99, 100], budget=64)
    assert (req.cached_len, req.mamba_fork_len) == (4, 6)
    # match ends exactly on the snapshot: nothing to fork
    _, req = admit(ids[:4] + [99, 100], budget=64)
    assert (req.cached_len, req.mamba_fork_len) == (4, None)
    # cold: nothing to fork
    _, req = admit([50, 51, 52, 53], budget=64)
    assert (req.cached_len, req.mamba_fork_len) == (0, None)
    # chunked: the fork rides along on the continuation until a chunk consumes it
    pend, chunk = admit(ids[:6] + list(range(100, 130)), budget=2)   # 36 tokens, chunk of 2
    assert isinstance(chunk, ChunkedReq) and chunk.mamba_fork_len == 6
    chunk.cached_len = chunk.device_len
    pend.chunked_req = chunk
    cont = PrefillAdder(token_budget=2, reserved_size=0, cache_manager=cm, table_manager=tm).try_add_one(pend)
    assert cont.mamba_fork_len == 6
    chunk.mamba_fork_len = None                                        # a forward consumed it
    cont2 = PrefillAdder(token_budget=2, reserved_size=0, cache_manager=cm, table_manager=tm).try_add_one(pend)
    assert cont2.mamba_fork_len is None


def _track_boundary(*, cached_len, extend_len, fork, pool):
    """Run _build_track_metadata on one hand-built request; return (boundary, remaining fork)."""
    from freetoken import core
    from freetoken.attention.linear import _build_track_metadata
    from freetoken.core import Context, set_global_ctx

    core._GLOBAL_CTX = None  # test-only: the builder reads the state pool off the ctx
    set_global_ctx(Context(page_size=1, linear_state_pool=pool))
    try:
        r = SimpleNamespace(cached_len=cached_len, extend_len=extend_len, mamba_fork_len=fork,
                            mamba_ping_pong=(1, 2), mamba_next_track_idx=0,
                            mamba_last_track_seqlen=None)
        cu = torch.tensor([0, extend_len], dtype=torch.int32)
        md = _build_track_metadata([r], cu, torch.device("cpu"), {})
        if md["track_dst"] is None:
            return None, r.mamba_fork_len
        return r.mamba_last_track_seqlen, r.mamba_fork_len
    finally:
        core._GLOBAL_CTX = None


def test_track_boundary_moves_to_the_fork_when_the_extend_spans_it():
    """Default: the deepest ×64 boundary strictly inside the extend. With a fork inside the
    extend: the deepest ×64 boundary at or before the fork, and the fork is consumed. A fork
    beyond this extend is left for a later chunk; one within the first 64 tokens (no boundary
    at or before it) or already behind the extend falls back to the default and is dropped."""
    pool = _pool()

    def T(**kw):
        return _track_boundary(pool=pool, **kw)
    assert T(cached_len=0, extend_len=8192, fork=None) == (8128, None)           # (b) today
    assert T(cached_len=90048, extend_len=6018, fork=None) == (96064, None)
    # 96k probe, 2nd request: cached 90,048, extend 6,018, fork 96,047 -> 90,048 + 93*64
    assert T(cached_len=90048, extend_len=6018, fork=96047) == (96000, None)
    # fan-out sibling, chunk 16,384..24,576 spanning the 24,152 fork -> 16,384 + 121*64
    assert T(cached_len=16384, extend_len=8192, fork=24152) == (24128, None)
    # fork lies in a later chunk: default boundary, fork kept for that chunk
    assert T(cached_len=8192, extend_len=8192, fork=24152) == (16320, 24152)
    # fork exactly at the extend end is not strictly inside: default, kept for the next chunk
    assert T(cached_len=8192, extend_len=8192, fork=16384) == (16320, 16384)
    # fork within the first 64 tokens of the extend: nothing to track at it -> default, dropped
    assert T(cached_len=90048, extend_len=6018, fork=90100) == (96064, None)
    # fork already behind this extend (stale): default, dropped
    assert T(cached_len=90048, extend_len=6018, fork=90000) == (96064, None)
    # an extend of one GDN chunk or less tracks nothing, fork or not
    assert T(cached_len=96000, extend_len=64, fork=96047) == (None, 96047)
