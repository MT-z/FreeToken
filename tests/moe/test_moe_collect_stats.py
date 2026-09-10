"""--moe-collect-stats: the flag, and the report it produces.

The counters themselves are accumulated device-side inside ``ensure_experts`` and were
already covered; what was missing until this flag existed was any way to turn them on from
the command line or read them back. These tests cover that wiring -- the flag reaching
``ServerArgs``, and the emit formatting the numbers and resetting the window afterwards.
"""

import contextlib
import io
from types import MethodType, SimpleNamespace

import pytest

from freetoken.engine.engine import MOE_STATS_INTERVAL, Engine
from freetoken.server.args import ServerArgs, parse_args


def test_flag_is_registered_and_defaults_off():
    """``--help`` short-circuits before the model is resolved, so this needs no checkpoint."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.suppress(SystemExit):
        parse_args(["--help"])
    assert "--moe-collect-stats" in buf.getvalue()
    # Off unless asked for: the counters ride in the decode CUDA graph and cost throughput.
    assert ServerArgs.moe_collect_stats is False


class _StubCache:
    """Just enough cache to exercise the emit: the four readers plus the window reset."""

    def __init__(self, decode_target="gpu", layer_calls=512):
        self.collect_stats = True
        self.decode_target = decode_target
        self._layer_calls = layer_calls
        self.reset_calls = 0

    def decode_miss_stats(self):
        return {
            "layer_calls": self._layer_calls,
            "active_per_layer": 8.0,
            "missing_per_layer": 2.0,
            "miss_rate": 0.25,
            "fetched_per_layer": 1.5,
            "cpu_per_layer": 0.5,
            "fetch_rate": 0.75,
            "prefill_hit_rows": 0,
            "prefill_rows": 0,
        }

    def decode_miss_stats_per_layer(self):
        return {
            "per_layer": [
                {"layer": 0, "steps": 4, "miss_rate": 0.5},
                {"layer": 1, "steps": 4, "miss_rate": 0.1},
                # steps == 0 means the layer never ran in this window; it must not be
                # ranked as a 0.0-miss-rate "best" layer.
                {"layer": 2, "steps": 0, "miss_rate": 0.0},
            ]
        }

    def decode_routing_stats(self):
        return {
            "slots_per_layer": 56.7,
            "working_set_mean": 173.1,
            "working_set_max": 243,
            "experts_for_90pct": 92.3,
            "oracle_hit_at_slots": 0.764,
            "norm_entropy": 0.813,
        }

    def reset_stats(self):
        self.reset_calls += 1


@pytest.fixture(autouse=True)
def _no_inherited_dump_path(monkeypatch):
    """The emit path reads FREETOKEN_MOE_FREQ_OUT, and a histogram sweep leaves it exported.

    Without this the suite passes in a clean shell and fails in the one where this branch is
    actually used, which reads as a code regression rather than as leakage.
    """
    monkeypatch.delenv("FREETOKEN_MOE_FREQ_OUT", raising=False)


def _engine_stub(cache):
    """Just enough Engine for _emit_moe_stats, with the dump path bound from the real class.

    Binding rather than re-declaring is the point: a stub that reimplements the engine side
    keeps passing while production changes underneath it.
    """
    from freetoken.distributed import DistributedInfo

    engine = SimpleNamespace(moe_offload_cache=cache, _emit_moe_stats=None,
                             _moe_freq_out_failed=False,
                             tp_info=DistributedInfo(rank=0, size=1))
    engine._dump_routing_histogram = MethodType(Engine._dump_routing_histogram, engine)
    return engine


def _emit(cache, caplog):
    with caplog.at_level("INFO"):
        Engine._emit_moe_stats(_engine_stub(cache))
    return "\n".join(r.getMessage() for r in caplog.records)


def test_emit_reports_and_resets_the_window(caplog):
    cache = _StubCache()
    out = _emit(cache, caplog)
    assert "miss_rate=0.250" in out
    # The oracle bound is the whole point of the report: it says how much room a different
    # eviction policy could possibly have.
    assert "oracle_hit=0.764" in out
    assert "(realized 0.750)" in out
    # Ranked worst-first, and the layer that never ran is left out entirely.
    assert "L0=0.500, L1=0.100" in out
    assert "L2" not in out
    assert cache.reset_calls == 1


def test_hybrid_split_only_reported_for_hybrid(caplog):
    assert "fetch_rate" not in _emit(_StubCache(decode_target="gpu"), caplog)
    caplog.clear()
    assert "fetch_rate=0.750" in _emit(_StubCache(decode_target="hybrid"), caplog)


def test_idle_window_emits_nothing_and_keeps_counters(caplog):
    """No decode ran, so there is nothing to report -- and nothing to reset either."""
    cache = _StubCache(layer_calls=0)
    assert _emit(cache, caplog) == ""
    assert cache.reset_calls == 0


def test_interval_is_a_sane_window():
    assert MOE_STATS_INTERVAL >= 1


class _DumpableCache(_StubCache):
    """_StubCache plus the fields routing_histogram reads, and the real method bound.

    The payload is assembled by production, not restated here: if the cache grows a field the
    dump needs, this fails loudly on the missing attribute instead of quietly writing a file
    that no longer matches what the engine writes.
    """

    def __init__(self, **kw):
        import torch

        from freetoken.moe.offload_cache import OffloadMoeCache

        super().__init__(**kw)
        self.num_layers, self.num_experts, self.cache_size = 2, 4, 6
        # Distinct per tensor so a swapped key in the payload cannot pass.
        self.decode_freq = torch.full((2, 4), 7, dtype=torch.int64)
        self.decode_miss_freq = torch.full((2, 4), 3, dtype=torch.int64)
        self.prefill_miss_freq = torch.full((2, 4), 5, dtype=torch.int64)
        # The real-rows pair, one column wider: the last column is the pad-row sentinel.
        self.decode_freq_real = torch.full((2, 5), 6, dtype=torch.int64)
        self.decode_miss_freq_real = torch.full((2, 5), 2, dtype=torch.int64)
        self.prefill_chunks = 11
        self.prefill_hit_rows = 13
        self.prefill_total_rows = 17
        self.lru_stats = torch.tensor([[19, 23, 29]], dtype=torch.int64)
        self.cpu_layer_ids = frozenset()
        self._prefill_hit_d2d_active = True
        # stat_* so the hybrid branch of window_totals has something real to read.
        self.stat_active = torch.tensor(31, dtype=torch.int64)
        self.stat_missing = torch.tensor(37, dtype=torch.int64)
        self.stat_calls = torch.tensor(41, dtype=torch.int64)
        self.routing_histogram = MethodType(OffloadMoeCache.routing_histogram, self)
        self.window_totals = MethodType(OffloadMoeCache.window_totals, self)


def _emit_with_dump(cache, caplog, path, monkeypatch, engine=None):
    if engine is None:
        engine = _engine_stub(cache)
    else:
        engine.moe_offload_cache = cache
        engine._dump_routing_histogram = MethodType(Engine._dump_routing_histogram, engine)
    monkeypatch.setenv("FREETOKEN_MOE_FREQ_OUT", str(path))
    with caplog.at_level("INFO"):
        Engine._emit_moe_stats(engine)
    return engine, "\n".join(r.getMessage() for r in caplog.records)


def test_an_unwritable_dump_path_warns_once_and_does_not_stop_decode(caplog, tmp_path, monkeypatch):
    """The dump runs inside the decode loop, so it must not be able to kill a serve.

    An unwritable FREETOKEN_MOE_FREQ_OUT is a typo in an env var. Before this was handled,
    torch.save raised out of _emit_moe_stats and took the serve with it -- the instrument
    killing the thing it was measuring. It must warn and carry on instead, and warn once
    rather than every 256 decode steps.
    """
    bad = tmp_path / "no-such-dir" / "freq.pt"  # parent does not exist
    engine, out = _emit_with_dump(_DumpableCache(), caplog, bad, monkeypatch)

    # The report itself still came out: decode was not interrupted.
    assert "miss_rate=0.250" in out
    assert "cannot be written" in out
    assert engine._moe_freq_out_failed is True
    assert not bad.exists()

    caplog.clear()
    _, out2 = _emit_with_dump(_DumpableCache(), caplog, bad, monkeypatch, engine=engine)
    assert "miss_rate=0.250" in out2, "the report must keep coming"
    assert "cannot be written" not in out2, "warned twice; it should latch"


def test_a_writable_dump_path_still_writes(caplog, tmp_path, monkeypatch):
    """The guard must not have turned the instrument off for everyone."""
    import torch

    good = tmp_path / "freq.pt"
    engine, _ = _emit_with_dump(_DumpableCache(), caplog, good, monkeypatch)
    assert good.exists()
    assert engine._moe_freq_out_failed is False
    saved = torch.load(good)
    # Distinguishable values per tensor: a swapped key or an inverted ratio has to show up.
    assert int(saved["decode_freq"][0, 0]) == 7
    assert int(saved["decode_miss_freq"][0, 0]) == 3
    assert int(saved["prefill_miss_freq"][0, 0]) == 5
    assert (saved["window_active"], saved["window_missing"]) == (19, 23)
    assert saved["prefill_hit_rows_window"] == 13 and saved["prefill_total_rows_window"] == 17
    assert saved["decode_target"] == "gpu" and saved["prefill_hit_d2d_active"] is True
    assert saved["decode_counted_layers"] == [0, 1]
    assert not (tmp_path / "freq.pt.tmp").exists()
    assert not list(tmp_path.glob("freq.pt.tmp.*")), "left its scratch file behind"


def test_a_failed_write_does_not_destroy_the_previous_dump(caplog, tmp_path, monkeypatch):
    """The dump is overwritten every MOE_STATS_INTERVAL steps and is polled from outside.

    Writing straight to the destination opens and truncates it before serializing, so a
    reader that arrives mid-write gets a truncated file and a writer that fails mid-write
    leaves one. On 2026-09-10 three readers took three different token counts from one live
    dump; the workaround lived in the consumer (copy on mtime change, re-check, hope). This
    puts it in the writer: save beside the destination and os.replace, which is atomic within
    a filesystem.
    """
    import torch

    dest = tmp_path / "freq.pt"
    _emit_with_dump(_DumpableCache(), caplog, dest, monkeypatch)
    first = torch.load(dest)
    assert first["cache_size"] == 6

    real_save = torch.save

    def torn_save(obj, path, *a, **kw):
        # Fails the way a real one does: the file is opened and partly written first.
        open(path, "wb").write(b"\x80\x04partial")
        raise RuntimeError("disk full")

    caplog.clear()
    monkeypatch.setattr(torch, "save", torn_save)
    engine, out = _emit_with_dump(_DumpableCache(), caplog, dest, monkeypatch)
    monkeypatch.setattr(torch, "save", real_save)

    assert "cannot be written" in out
    # The previous dump is still there and still loadable.
    assert torch.load(dest)["cache_size"] == 6, "the failed write destroyed the last good dump"
    assert not list(tmp_path.glob("freq.pt.tmp.*")), "left its scratch file behind"

    # A full disk is not a typo. This assertion used to read `is True` -- the test pinned the
    # very behaviour the source comment says it avoids, because permanence was decided from
    # the exception type and torch.save reports both with RuntimeError.
    assert engine._moe_freq_out_failed is False, "one full disk ended collection for the run"
    assert "will retry next report" in out


def test_a_transient_write_failure_recovers_at_the_next_report(caplog, tmp_path, monkeypatch):
    """The consequence of not latching: collection resumes on its own.

    A multi-hour run is the case that matters -- the dump is the whole deliverable, and a
    single ENOSPC must not leave it frozen at an hours-old snapshot.
    """
    import torch

    dest = tmp_path / "freq.pt"
    real_save = torch.save

    def full_disk(obj, path, *a, **kw):
        raise RuntimeError("No space left on device")

    monkeypatch.setattr(torch, "save", full_disk)
    engine, _ = _emit_with_dump(_DumpableCache(), caplog, dest, monkeypatch)
    assert not dest.exists()
    assert engine._moe_freq_out_failed is False

    monkeypatch.setattr(torch, "save", real_save)
    caplog.clear()
    _, out = _emit_with_dump(_DumpableCache(), caplog, dest, monkeypatch, engine=engine)
    assert dest.exists(), "the dump never came back after the disk was freed"
    assert torch.load(dest)["cache_size"] == 6
    assert "cannot be written" not in out


def test_hybrid_window_totals_reach_the_dump(caplog, tmp_path, monkeypatch):
    """ensure_experts_hybrid runs a different kernel and never writes lru_stats.

    Reading lru_stats unconditionally made the dump's window totals three zeros on hybrid,
    beside a log line that showed misses -- and decode_target in the payload could not
    recover them, because the values themselves were gone. Both now go through window_totals.
    """
    import torch

    dest = tmp_path / "freq.pt"
    cache = _DumpableCache(decode_target="hybrid")
    # lru_stats holds the gpu-path numbers; the hybrid path's live in stat_*.
    _emit_with_dump(cache, caplog, dest, monkeypatch)
    payload = torch.load(dest)
    assert (payload["window_active"], payload["window_missing"],
            payload["window_layer_calls"]) == (31, 37, 41)
    assert payload["window_missing"] != 23, "read lru_stats, which hybrid never writes"

    caplog.clear()
    gpu = _DumpableCache(decode_target="gpu")
    _emit_with_dump(gpu, caplog, dest, monkeypatch)
    assert torch.load(dest)["window_missing"] == 23, "the gpu path must still read lru_stats"


def test_the_payload_is_a_snapshot_not_a_live_view():
    """prefill_miss_freq lives on the host, where .cpu() returns the tensor itself.

    The other two histograms are CUDA and .cpu() copies them, so the asymmetry is invisible at
    the call site -- and the payload exists precisely to freeze what gets serialized.
    """
    import torch

    cache = _DumpableCache()
    payload = cache.routing_histogram()
    before = payload["prefill_miss_freq"].clone()

    cache.prefill_miss_freq += 100

    assert torch.equal(payload["prefill_miss_freq"], before), "the payload aliases the counter"


def test_only_rank_zero_writes_the_dump(caplog, tmp_path, monkeypatch):
    """One Engine per TP rank, one FREETOKEN_MOE_FREQ_OUT between them.

    Without a gate every rank writes the same destination through the same temp name, which
    is exactly what the temp-and-rename was added to prevent.
    """
    from freetoken.distributed import DistributedInfo

    dest = tmp_path / "freq.pt"
    engine = _engine_stub(_DumpableCache())
    engine.tp_info = DistributedInfo(rank=1, size=2)
    _emit_with_dump(_DumpableCache(), caplog, dest, monkeypatch, engine=engine)
    assert not dest.exists(), "a non-primary rank wrote the dump"

    engine.tp_info = DistributedInfo(rank=0, size=2)
    _emit_with_dump(_DumpableCache(), caplog, dest, monkeypatch, engine=engine)
    assert dest.exists(), "rank 0 must still write it"
