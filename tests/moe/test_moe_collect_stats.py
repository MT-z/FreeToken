"""--moe-collect-stats: the flag, and the report it produces.

The counters themselves are accumulated device-side inside ``ensure_experts`` and were
already covered; what was missing until this flag existed was any way to turn them on from
the command line or read them back. These tests cover that wiring -- the flag reaching
``ServerArgs``, and the emit formatting the numbers and resetting the window afterwards.
"""

import contextlib
import io
from types import SimpleNamespace

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


def _emit(cache, caplog):
    engine = SimpleNamespace(moe_offload_cache=cache, _emit_moe_stats=None)
    with caplog.at_level("INFO"):
        Engine._emit_moe_stats(engine)
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
    """_StubCache plus the tensors FREETOKEN_MOE_FREQ_OUT reads."""

    def __init__(self, **kw):
        import torch

        super().__init__(**kw)
        self.decode_freq = torch.zeros((2, 4), dtype=torch.int64)
        self.decode_miss_freq = torch.zeros((2, 4), dtype=torch.int64)
        self.prefill_miss_freq = torch.zeros((2, 4), dtype=torch.int64)
        self.prefill_chunks = 0
        self.prefill_hit_rows = 0
        self.prefill_total_rows = 0
        self.cache_size = 6
        self.num_layers = 2
        self.num_experts = 4


def _emit_with_dump(cache, caplog, path, engine=None):
    import os

    engine = engine or SimpleNamespace(
        moe_offload_cache=cache, _emit_moe_stats=None, _moe_freq_out_failed=False
    )
    old = os.environ.get("FREETOKEN_MOE_FREQ_OUT")
    os.environ["FREETOKEN_MOE_FREQ_OUT"] = str(path)
    try:
        with caplog.at_level("INFO"):
            Engine._emit_moe_stats(engine)
    finally:
        if old is None:
            os.environ.pop("FREETOKEN_MOE_FREQ_OUT", None)
        else:
            os.environ["FREETOKEN_MOE_FREQ_OUT"] = old
    return engine, "\n".join(r.getMessage() for r in caplog.records)


def test_an_unwritable_dump_path_warns_once_and_does_not_stop_decode(caplog, tmp_path):
    """The dump runs inside the decode loop, so it must not be able to kill a serve.

    An unwritable FREETOKEN_MOE_FREQ_OUT is a typo in an env var. Before this was handled,
    torch.save raised out of _emit_moe_stats and took the serve with it -- the instrument
    killing the thing it was measuring. It must warn and carry on instead, and warn once
    rather than every 256 decode steps.
    """
    bad = tmp_path / "no-such-dir" / "freq.pt"  # parent does not exist
    engine, out = _emit_with_dump(_DumpableCache(), caplog, bad)

    # The report itself still came out: decode was not interrupted.
    assert "miss_rate=0.250" in out
    assert "cannot be written" in out
    assert engine._moe_freq_out_failed is True
    assert not bad.exists()

    caplog.clear()
    _, out2 = _emit_with_dump(_DumpableCache(), caplog, bad, engine=engine)
    assert "miss_rate=0.250" in out2, "the report must keep coming"
    assert "cannot be written" not in out2, "warned twice; it should latch"


def test_a_writable_dump_path_still_writes(caplog, tmp_path):
    """The guard must not have turned the instrument off for everyone."""
    import torch

    good = tmp_path / "freq.pt"
    engine, _ = _emit_with_dump(_DumpableCache(), caplog, good)
    assert good.exists()
    assert engine._moe_freq_out_failed is False
    saved = torch.load(good)
    assert saved["cache_size"] == 6
    assert saved["decode_freq"].shape == (2, 4)


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
    _emit_with_dump(_DumpableCache(), caplog, dest)
    first = torch.load(dest)
    assert first["cache_size"] == 6

    real_save = torch.save

    def torn_save(obj, path, *a, **kw):
        # Fails the way a real one does: the file is opened and partly written first.
        open(path, "wb").write(b"\x80\x04partial")
        raise RuntimeError("disk full")

    caplog.clear()
    monkeypatch.setattr(torch, "save", torn_save)
    engine, out = _emit_with_dump(_DumpableCache(), caplog, dest)
    monkeypatch.setattr(torch, "save", real_save)

    assert "cannot be written" in out
    assert engine._moe_freq_out_failed is True
    # The point: the previous dump is still there and still loadable.
    assert torch.load(dest)["cache_size"] == 6, "the failed write destroyed the last good dump"
    assert not (tmp_path / "freq.pt.tmp").exists(), "left its scratch file behind"
