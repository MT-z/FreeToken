from contextlib import contextmanager

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.layers.quantization import QuantKind


def _init_tp():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _bf16_offload_layer(layer_id: int, num_experts: int, top_k: int, hidden_size: int, intermediate_size: int):
    """A bf16 offload layer on the fused kernel."""
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.layers.quantization import NoQuantConfig

    return OffloadMoELayer(
        layer_id, num_experts, top_k, hidden_size, intermediate_size,
        quant_config=NoQuantConfig(), prefix=f"model.layers.{layer_id}.mlp.experts",
    )


def _make_layer_and_cache():
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    layer = _bf16_offload_layer(0, 4, 2, 8, 16)
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=4,
        cache_size=6,
        device=torch.device("cpu"),
    )
    cache.set_bank_sources({"gate_up": [torch.randn(4, 32, 8)], "down": [torch.randn(4, 8, 16)]})
    layer.offload_cache = cache
    return layer, cache


def test_dummy_expert_banks_follow_the_kernel_layout(monkeypatch):

    from freetoken.kernel import backend
    from freetoken.layers.quantization import QuantBackend, QuantConfig, set_quant_backend
    from freetoken.moe.expert_banks import build_expert_banks

    _init_tp()
    L, E, H, I = 3, 4, 64, 32
    monkeypatch.setattr(backend, "device_capability", lambda: (0, 0))
    monkeypatch.setattr(backend, "is_vllm_installed", lambda: False)
    monkeypatch.setattr(backend, "is_flashinfer_installed", lambda: False)

    def _bound(quant):
        layer = _bf16_offload_layer(0, E, 2, H, I) if quant is None else None
        if layer is None:
            from freetoken.layers.moe import OffloadMoELayer

            layer = OffloadMoELayer(0, E, 2, H, I, quant_config=quant, prefix="model.layers.0.mlp.experts")
        return layer

    bf16 = _bound(None)
    banks = build_expert_banks(bf16.quant_method, L, None, device=torch.device("cpu"), dummy=True)
    assert banks.kind is QuantKind.NONE and set(banks.sources) == {"gate_up", "down"}
    assert len(banks.sources["gate_up"]) == L and all(t.shape == (E, 2 * I, H) for t in banks.sources["gate_up"])
    assert all(t.shape == (E, H, I) for t in banks.sources["down"])

    set_quant_backend(QuantBackend.parse("moe.nvfp4=triton"))
    quant = QuantConfig.from_hf({"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4", "ignore": ["lm_head"]}})
    nvfp4 = _bound(quant)
    banks = build_expert_banks(nvfp4.quant_method, L, None, device=torch.device("cpu"), dummy=True)
    assert banks.kind is QuantKind.NVFP4 and banks.kernel == "triton"
    assert {len(layers) for layers in banks.sources.values()} == {L}
    assert {t.shape[0] for layers in banks.sources.values() for t in layers} == {E}
    assert torch.all(banks.sources["gate_up_scale"][0].float() == 1.0)
    assert torch.all(banks.sources["gate_up_global"][0].float() > 0)


def test_offload_moe_layer_prefill_forward_uses_single_layer_cache_view(monkeypatch):
    layer, cache = _make_layer_and_cache()
    topk_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    topk_ids = torch.tensor([[2, 1]], dtype=torch.int32)
    hidden_states = torch.randn(1, 8)
    router_logits = torch.randn(1, 4)
    calls = {}

    monkeypatch.setattr(
        "freetoken.layers.moe.fused_topk",
        lambda *, hidden_states, gating_output, topk, renormalize: (topk_weights, topk_ids),
    )
    monkeypatch.setattr(cache, "materialize_layer", lambda layer_id: calls.setdefault("layer_id", layer_id))
    monkeypatch.setattr(cache, "copy_missing", lambda: calls.setdefault("copied", True))

    def fake_fused(
        hidden_states,
        w1,
        w2,
        got_topk_weights,
        got_topk_ids,
        activation,
        apply_router_weight_on_input,
        act_alpha=1.0,
        act_limit=float("inf"),
    ):
        calls["w1"] = w1
        calls["w2"] = w2
        calls["topk_weights"] = got_topk_weights
        calls["topk_ids"] = got_topk_ids.clone()
        return hidden_states

    monkeypatch.setattr("freetoken.moe.fused.fused_experts_impl", fake_fused)

    out = layer.prefill_forward(hidden_states, router_logits)

    assert out is hidden_states
    assert calls["layer_id"] == 0
    assert calls["copied"] is True
    assert calls["w1"].shape[0] == layer.num_experts
    assert calls["w2"].shape[0] == layer.num_experts
    assert calls["w1"].data_ptr() == cache.bank_caches["gate_up"].data_ptr()
    assert calls["w2"].data_ptr() == cache.bank_caches["down"].data_ptr()
    assert calls["topk_weights"] is topk_weights
    assert calls["topk_ids"].dtype == torch.int32
    # slot == expert id after materialize, so the routing ids pass through unmapped
    assert calls["topk_ids"].tolist() == [[2, 1]]


def test_offload_moe_layer_prefill_overlap_prefetches_layers_into_two_buffers(monkeypatch):
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    num_layers = 3
    num_experts = 4
    layers = [_bf16_offload_layer(layer_id, num_experts, 2, 8, 16) for layer_id in range(num_layers)]
    cache = OffloadMoeCache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=8,
        device=torch.device("cpu"),
        prefill_overlap=True,
    )
    gate_up_source = list(torch.arange(num_layers * num_experts * 32 * 8, dtype=torch.float32).reshape(
        num_layers * num_experts, 32, 8
    ).split(num_experts))
    down_source = list(torch.arange(num_layers * num_experts * 8 * 16, dtype=torch.float32).reshape(
        num_layers * num_experts, 8, 16
    ).split(num_experts))
    cache.set_bank_sources({"gate_up": gate_up_source, "down": down_source})
    for layer in layers:
        layer.offload_cache = cache

    topk_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    topk_ids = torch.tensor([[2, 1]], dtype=torch.int32)
    hidden_states = torch.randn(1, 8)
    router_logits = torch.randn(1, num_experts)
    fused_calls = []

    monkeypatch.setattr(
        "freetoken.layers.moe.fused_topk",
        lambda *, hidden_states, gating_output, topk, renormalize: (
            topk_weights,
            topk_ids.clone(),
        ),
    )

    def unexpected_fast_index_copy(*args, **kwargs):
        raise AssertionError("prefill overlap should use direct async copy")

    monkeypatch.setattr("freetoken.kernel.fast_index_copy_jit", unexpected_fast_index_copy)

    def fake_fused(
        hidden_states,
        w1,
        w2,
        got_topk_weights,
        got_topk_ids,
        activation,
        apply_router_weight_on_input,
        act_alpha=1.0,
        act_limit=float("inf"),
    ):
        layer_id = len(fused_calls)
        fused_calls.append(
            {
                "w1_ptr": w1.data_ptr(),
                "w2_ptr": w2.data_ptr(),
                "w1": w1.clone(),
                "w2": w2.clone(),
                "topk_weights": got_topk_weights,
                "topk_ids": got_topk_ids.clone(),
            }
        )
        return hidden_states + layer_id

    monkeypatch.setattr("freetoken.moe.fused.fused_experts_impl", fake_fused)

    out = hidden_states
    for layer in layers:
        out = layer.prefill_forward(out, router_logits)

    assert torch.allclose(out, hidden_states + 3)
    for layer_id in range(num_layers):
        assert fused_calls[layer_id]["topk_weights"] is topk_weights
        assert fused_calls[layer_id]["topk_ids"].tolist() == [[2, 1]]
        assert torch.equal(fused_calls[layer_id]["w1"], gate_up_source[layer_id])
        assert torch.equal(fused_calls[layer_id]["w2"], down_source[layer_id])

    assert fused_calls[0]["w1_ptr"] == fused_calls[2]["w1_ptr"]
    assert fused_calls[0]["w2_ptr"] == fused_calls[2]["w2_ptr"]
    assert fused_calls[0]["w1_ptr"] != fused_calls[1]["w1_ptr"]
    assert fused_calls[0]["w2_ptr"] != fused_calls[1]["w2_ptr"]
    prefill_gate_up_buffer, prefill_down_buffer = cache.prefill_bank_buffers
    assert prefill_gate_up_buffer.data_ptr() == cache.bank_caches["gate_up"].data_ptr()
    assert prefill_down_buffer.data_ptr() == cache.bank_caches["down"].data_ptr()


def test_offload_moe_cache_prefill_overlap_requires_two_layer_slots():
    from freetoken.moe.offload_cache import OffloadMoeCache

    with pytest.raises(AssertionError):
        OffloadMoeCache(
            num_layers=3,
            num_experts=4,
            cache_size=7,
            device=torch.device("cpu"),
            prefill_overlap=True,
        )


def test_offload_moe_cache_marlin_rejects_slot_count_beyond_kernel_limit():
    from freetoken.moe.offload_cache import OffloadMoeCache

    with pytest.raises(ValueError, match="992"):
        OffloadMoeCache(
            num_layers=2,
            num_experts=8,
            cache_size=1024,
            device=torch.device("cpu"),
            quant_format="nvfp4_marlin",
        )


def test_prefill_overlap_prefetch_invalidates_borrowed_unified_cache_slots():
    from freetoken.moe.offload_cache import OffloadMoeCache

    num_layers = 3
    num_experts = 4
    cache = OffloadMoeCache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=8,
        device=torch.device("cpu"),
        prefill_overlap=True,
    )
    gate_up_source = list(torch.arange(num_layers * num_experts * 32 * 8, dtype=torch.float32).reshape(
        num_layers * num_experts, 32, 8
    ).split(num_experts))
    down_source = list(torch.arange(num_layers * num_experts * 8 * 16, dtype=torch.float32).reshape(
        num_layers * num_experts, 8, 16
    ).split(num_experts))
    cache.set_bank_sources({"gate_up": gate_up_source, "down": down_source})

    old_layers = torch.tensor([2, 2, 1, 1], dtype=torch.int32)
    old_experts = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    cache.id_of_slot[:num_experts] = old_layers * num_experts + old_experts
    cache.usage[:num_experts] = torch.arange(1, num_experts + 1, dtype=torch.int64)
    for slot, (layer_id, expert_id) in enumerate(zip(old_layers.tolist(), old_experts.tolist())):
        cache.slot_for_id[layer_id, expert_id] = slot

    cache.prefetch_prefill_layer(0)

    assert cache.id_of_slot[:num_experts].tolist() == [-1] * num_experts
    assert cache.usage[:num_experts].tolist() == [0] * num_experts
    for layer_id, expert_id in zip(old_layers.tolist(), old_experts.tolist()):
        assert int(cache.slot_for_id[layer_id, expert_id].item()) == -1
    assert torch.equal(cache.bank_caches["gate_up"][:num_experts], gate_up_source[0])
    assert torch.equal(cache.bank_caches["down"][:num_experts], down_source[0])


def test_prefill_overlap_waits_for_previous_prefill_release_after_begin(monkeypatch):
    from freetoken.moe.offload_cache import OffloadMoeCache

    num_layers = 2
    num_experts = 4
    cache = OffloadMoeCache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=8,
        device=torch.device("cpu"),
        prefill_overlap=True,
    )
    gate_up_source = list(torch.zeros(num_layers * num_experts, 32, 8).split(num_experts))
    down_source = list(torch.zeros(num_layers * num_experts, 8, 16).split(num_experts))
    cache.set_bank_sources({"gate_up": gate_up_source, "down": down_source})

    class FakeStream:
        def __init__(self):
            self.waited = []

        def wait_event(self, event):
            self.waited.append(event.name)

    class FakeEvent:
        def __init__(self, name):
            self.name = name

        def record(self, stream=None):
            pass

    @contextmanager
    def fake_cuda_stream(stream):
        yield

    copy_stream = FakeStream()
    cache.prefill_copy_stream = copy_stream
    cache.prefill_begin_event = FakeEvent("begin")
    cache.prefill_ready_events = [FakeEvent("ready0"), FakeEvent("ready1")]
    cache.prefill_release_events = [FakeEvent("release0"), FakeEvent("release1")]
    monkeypatch.setattr("torch.cuda.stream", fake_cuda_stream)
    monkeypatch.setattr("torch.cuda.current_stream", lambda device=None: object())

    cache.prefetch_prefill_layer(0)
    cache.release_prefill_layer(0)
    cache.begin_prefill()
    cache.prefetch_prefill_layer(0)

    # begin_prefill fences the copy stream behind the compute stream (so a prefetch
    # cannot race the preceding decode batch), then the buffer reuse waits on the
    # previous prefill's release event.
    assert copy_stream.waited == ["begin", "release0"]


def test_offload_moe_layer_decode_forward_uses_remapped_slot_ids(monkeypatch):
    layer, cache = _make_layer_and_cache()
    topk_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    topk_ids = torch.tensor([[2, 1]], dtype=torch.int32)
    hidden_states = torch.randn(1, 8)
    router_logits = torch.randn(1, 4)
    calls = {}

    monkeypatch.setattr(
        "freetoken.layers.moe.fused_topk",
        lambda *, hidden_states, gating_output, topk, renormalize: (topk_weights, topk_ids),
    )

    def fake_ensure(layer_id, expert_ids):
        calls["ensure_layer_id"] = layer_id
        calls["ensure_expert_ids"] = expert_ids.clone()
        expert_ids.copy_(torch.tensor([[5, 0]], dtype=torch.int32))

    monkeypatch.setattr(cache, "ensure_experts", fake_ensure)
    monkeypatch.setattr(cache, "copy_missing", lambda: calls.setdefault("copied", True))

    def fake_fused_decode(
        hidden_states,
        w1,
        w2,
        got_topk_weights,
        got_topk_ids,
        activation,
        apply_router_weight_on_input,
        act_alpha=1.0,
        act_limit=float("inf"),
    ):
        calls["w1"] = w1
        calls["w2"] = w2
        calls["topk_weights"] = got_topk_weights
        calls["topk_ids"] = got_topk_ids.clone()
        return hidden_states

    monkeypatch.setattr("freetoken.moe.fused.fused_experts_decode_impl", fake_fused_decode)

    out = layer.decode_forward(hidden_states, router_logits)

    assert out is hidden_states
    assert calls["ensure_layer_id"] == 0
    assert calls["ensure_expert_ids"].tolist() == [[2, 1]]
    assert calls["copied"] is True
    assert calls["w1"] is cache.bank_caches["gate_up"]
    assert calls["w2"] is cache.bank_caches["down"]
    assert calls["topk_weights"] is topk_weights
    assert calls["topk_ids"].dtype == torch.int32
    assert calls["topk_ids"].tolist() == [[5, 0]]



def test_lru_gpu_cache_assigns_unique_slots_for_large_miss_batch():
    import pytest
    from freetoken.moe.offload_cache import OffloadMoeCache

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the GPU offload cache kernel")

    cache = OffloadMoeCache(
        num_layers=40,
        num_experts=256,
        cache_size=1664,
        device=torch.device("cuda"),
    )
    expert_ids = torch.arange(256, dtype=torch.int32, device="cuda").view(32, 8)

    cache.ensure_experts(0, expert_ids)
    torch.cuda.synchronize()

    assert int(cache.num_indices.item()) == 256
    assert expert_ids.min().item() >= 0
    assert expert_ids.max().item() < cache.cache_size
    evict_slots = cache.evict_slots[:256]
    assert evict_slots.min().item() >= 0
    assert evict_slots.max().item() < cache.cache_size
    assert torch.unique(evict_slots).numel() == evict_slots.numel()
    assert cache.src_indices[:256].tolist() == list(range(256))


def test_adjust_config_converts_moe_cache_rate_to_cache_size(monkeypatch):
    from types import SimpleNamespace

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    import freetoken.engine.engine as engine_module
    from freetoken.engine.engine import _adjust_config

    # This test exercises the discrete-GPU offload path regardless of the host
    # running the suite (GB10 reports cudaDevAttrIntegrated=1).
    monkeypatch.setattr(engine_module, "_is_unified_memory_gpu", lambda index=None: False)

    config = EngineConfig(
        model_path="/tmp/freetoken-test-model",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.float16,
        attention_backend="fi",
        moe_cache_rate=0.3,
    )
    object.__setattr__(
        config,
        "model_config",
        SimpleNamespace(
            has_swa_attention=False,
            has_linear_attention=False,
            is_moe=True,
            num_layers=10,
            num_moe_layers=10,
            num_experts=8,
            expert_quant="none",
            moe_strategy="auto",
        ),
    )

    _adjust_config(config)

    from freetoken.moe import is_offload_moe_strategy

    assert config.moe_cache_size == 24
    # Family, not member: a box with a benchbw profile resolves bf16 experts to hybrid.
    assert is_offload_moe_strategy(config.moe_strategy)


def test_graph_capture_reuses_warm_offload_cache_before_capture(monkeypatch):
    import freetoken.core as core
    from freetoken.core import Context, Req, get_global_ctx
    from freetoken.engine.graph import GraphRunner

    events = []
    _init_tp()
    monkeypatch.setattr(core, "_GLOBAL_CTX", Context(page_size=1))

    class FakeGraph:
        def pool(self):
            return "pool"

    @contextmanager
    def fake_cuda_graph(graph, pool=None, stream=None):
        events.append("graph_enter")
        yield
        events.append("graph_exit")

    class FakeAttnBackend:
        def init_capture_graph(self, max_seq_len, bs_list):
            pass

        def prepare_for_capture(self, batch):
            pass

    class FakeModel:
        def forward(self):
            events.append("forward")
            batch = get_global_ctx().batch
            return torch.zeros(batch.size, 3)

    class FakeOffloadCache:
        def reset(self):
            events.append("reset")

    monkeypatch.setattr("torch.cuda.CUDAGraph", FakeGraph)
    monkeypatch.setattr("torch.cuda.graph", fake_cuda_graph)
    monkeypatch.setattr("torch.cuda.synchronize", lambda device=None: None)
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: None)
    monkeypatch.setattr("torch.cuda.reset_peak_memory_stats", lambda device=None: None)
    monkeypatch.setattr("freetoken.engine.graph.get_free_memory", lambda device: 1024)

    dummy_req = Req(
        input_ids=torch.tensor([0], dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=1,
        uid=-1,
        sampling_params=None,
        cache_handle=None,
    )
    GraphRunner(
        stream=None,
        device=torch.device("cpu"),
        model=FakeModel(),
        attn_backend=FakeAttnBackend(),
        cuda_graph_bs=[1],
        cuda_graph_max_bs=None,
        free_memory=1024,
        max_seq_len=1,
        vocab_size=3,
        dummy_req=dummy_req,
        moe_offload_cache=FakeOffloadCache(),
    )

    assert events == [
        "reset",
        "forward",
        "graph_enter",
        "forward",
        "graph_exit",
        "reset",
        "reset",
    ]


def test_nvfp4_materialize_keeps_bookkeeping_consistent_across_requests():
    """Regression: a full-layer prefill loads the layer's experts into slots [0, E).
    If that overwrite does not invalidate the previous owners' mappings, a later
    decode "hits" a stale slot_for_id entry and silently reads another expert's
    weights. materialize_layer must keep bookkeeping == slot contents."""
    import pytest
    from freetoken.moe.offload_cache import OffloadMoeCache

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the GPU offload cache kernel")

    L, E, S = 2, 8, 8
    OUT, IN = 64, 512  # keep rows >= 128B so the fast_index_copy JIT has a kernel
    dev = torch.device("cuda")

    def bank(out, inner, dtype):
        # one independently allocated [E, out, inner] tensor per layer (the per-layer host
        # bank contract); row idx within layer l keeps the old flat fingerprint l*E+idx.
        layers = []
        for l in range(L):
            t = torch.zeros(E, out, inner, dtype=dtype)
            for e in range(E):
                t[e].view(torch.uint8).fill_(l * E + e)
            layers.append(t)
        return layers

    def pinned(layers):
        return [t.pin_memory() for t in layers]

    cache = OffloadMoeCache(
        num_layers=L, num_experts=E, cache_size=S, device=dev, quant_format="nvfp4"
    )
    cache.set_bank_sources(
        {
            "gate_up_packed": pinned(bank(OUT, IN // 2, torch.uint8)),
            "gate_up_scale": pinned(bank(OUT, IN // 16, torch.float8_e4m3fn)),
            "gate_up_global": pinned([t.squeeze(-1).contiguous() for t in bank(OUT, 1, torch.float16)]),
            "down_packed": pinned(bank(OUT, IN // 2, torch.uint8)),
            "down_scale": pinned(bank(OUT, IN // 16, torch.float8_e4m3fn)),
            "down_global": pinned([t.squeeze(-1).contiguous() for t in bank(OUT, 1, torch.float16)]),
        }
    )
    cache.reset()

    def fingerprint(slot):  # which source row's bytes live in this slot?
        return int(cache.bank_caches["gate_up_packed"][slot].view(torch.uint8).flatten()[0].item())

    # Request A, decode: layer 0 loads experts 3 and 5 somewhere in the cache.
    ids = torch.tensor([3, 5], dtype=torch.int32, device=dev)
    cache.ensure_experts(0, ids)
    cache.copy_missing()
    torch.cuda.synchronize()
    assert [fingerprint(s) for s in ids.tolist()] == [3, 5]

    # Request B, prefill: layer 1 is materialized into slots [0, E), overwriting
    # every slot (S == E), including the ones decode A used.
    cache.materialize_layer(1)
    cache.copy_missing()
    torch.cuda.synchronize()
    # The layer's experts fill slots [0, E) bijectively and the bookkeeping agrees.
    assert [fingerprint(s) for s in range(E)] == [E + e for e in range(E)]
    assert cache.slot_for_id[1].tolist() == list(range(E))

    # Request B, decode: layer 0 routes to experts 3/5 again. Their old slots were
    # overwritten, so this must be a miss + reload -- never a stale hit serving
    # layer-1 bytes.
    ids2 = torch.tensor([3, 5], dtype=torch.int32, device=dev)
    cache.ensure_experts(0, ids2)
    cache.copy_missing()
    torch.cuda.synchronize()
    assert [fingerprint(s) for s in ids2.tolist()] == [3, 5]

    # The prefilled layer's own experts still resolve to correct bytes (S == E, so
    # the layer-0 reload above evicted two layer-1 slots -- hit or miss, the
    # bookkeeping must never serve another expert's bytes).
    ids3 = torch.tensor([1, 2], dtype=torch.int32, device=dev)
    cache.ensure_experts(1, ids3)
    cache.copy_missing()
    torch.cuda.synchronize()
    assert [fingerprint(s) for s in ids3.tolist()] == [E + 1, E + 2]


def test_offload_cache_rebuild_resizes_and_preserves_sources():
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    cache = OffloadMoeCache(num_layers=1, num_experts=4, cache_size=6, device=torch.device("cpu"))
    gate_up = torch.randn(4, 32, 8)
    down = torch.randn(4, 8, 16)
    cache.set_bank_sources({"gate_up": [gate_up], "down": [down]})

    cache.rebuild(10)

    assert cache.cache_size == 10
    # host sources preserved (same objects, not reloaded)
    assert cache.bank_sources["gate_up"][0] is gate_up
    assert cache.bank_sources["down"][0] is down
    # GPU slot caches resized to the new cache_size, row shape unchanged
    assert cache.bank_caches["gate_up"].shape == (10, 32, 8)
    assert cache.bank_caches["down"].shape == (10, 8, 16)
    # bookkeeping resized + reset
    assert cache.id_of_slot.shape == (10,)
    assert cache.usage.shape == (10,)
    assert torch.all(cache.slot_for_id == -1)
    assert torch.all(cache.id_of_slot == -1)


def test_offload_cache_rebuild_disables_prefill_overlap_when_too_small():
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    cache = OffloadMoeCache(
        num_layers=1, num_experts=4, cache_size=8, device=torch.device("cpu"),
        prefill_overlap=True,
    )
    cache.set_bank_sources({"gate_up": [torch.randn(4, 32, 8)], "down": [torch.randn(4, 8, 16)]})
    assert cache.prefill_overlap is True

    cache.rebuild(5)  # 5 < 2*num_experts (8) -> overlap must auto-disable

    assert cache.cache_size == 5
    assert cache.prefill_overlap is False
    assert cache.prefill_bank_buffers == []


def test_offload_cache_rebuild_keeps_overlap_at_boundary():
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    cache = OffloadMoeCache(
        num_layers=1, num_experts=4, cache_size=8, device=torch.device("cpu"),
        prefill_overlap=True,
    )
    cache.set_bank_sources({"gate_up": [torch.randn(4, 32, 8)], "down": [torch.randn(4, 8, 16)]})
    cache.rebuild(8)  # exactly 2*num_experts -> overlap stays on
    assert cache.prefill_overlap is True
    assert cache.cache_size == 8


def test_offload_cache_validate_rebuild_enforces_marlin_cap_and_floor():
    # The constructor caps nvfp4_marlin slots at 992; a runtime rebuild must enforce the
    # same upper cap (and the num_experts floor), else marlin decode kernels later break.
    from freetoken.moe.offload_cache import MARLIN_MAX_CACHE_SIZE, OffloadMoeCache

    _init_tp()
    marlin = OffloadMoeCache(
        num_layers=1, num_experts=8, cache_size=16,
        device=torch.device("cpu"), quant_format="nvfp4_marlin",
    )
    with pytest.raises(ValueError, match="992"):
        marlin.validate_rebuild(MARLIN_MAX_CACHE_SIZE + 1)
    marlin.validate_rebuild(MARLIN_MAX_CACHE_SIZE)  # exactly at the cap: allowed

    bf16 = OffloadMoeCache(num_layers=1, num_experts=4, cache_size=6, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="num_experts"):
        bf16.validate_rebuild(3)  # below the num_experts floor


def _make_split_cache(num_layers=2, locked=(1,), prefill_overlap=False, device="cpu"):
    """A [gate_up, down] bf16 cache with the given layers LOCKED (rest pinned)."""
    from freetoken.moe.host_banks import HostResidency
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    dev = torch.device(device)
    cache = OffloadMoeCache(
        num_layers=num_layers, num_experts=4, cache_size=8,
        device=dev, prefill_overlap=prefill_overlap,
    )
    cache.cpu_layer_ids = frozenset(locked)
    src_dev = dev if dev.type == "cuda" else torch.device("cpu")
    sources = {
        # CUDA-resident pinned-layer sources keep _build_copy_plan's device_ptr happy in the CUDA variant; locked layers stay host tensors (never translated)
        "gate_up": [
            torch.randn(4, 32, 8, device=torch.device("cpu") if i in locked else src_dev)
            for i in range(num_layers)
        ],
        "down": [
            torch.randn(4, 8, 16, device=torch.device("cpu") if i in locked else src_dev)
            for i in range(num_layers)
        ],
    }
    residency = [
        HostResidency.LOCKED.value if i in locked else HostResidency.PINNED.value
        for i in range(num_layers)
    ]
    cache.set_bank_sources(sources, layer_residency=residency)
    return cache, sources


def test_set_bank_sources_locked_layer_requires_cpu_layer_ids():
    # a layer without a device address can only decode on the CPU executor; labeling it LOCKED outside cpu_layer_ids is a wiring bug and must fail loudly
    from freetoken.moe.host_banks import HostResidency
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    cache = OffloadMoeCache(
        num_layers=2, num_experts=4, cache_size=8, device=torch.device("cpu"),
    )
    sources = {
        "gate_up": [torch.randn(4, 32, 8) for _ in range(2)],
        "down": [torch.randn(4, 8, 16) for _ in range(2)],
    }
    with pytest.raises(ValueError, match="cpu_layer_ids"):
        cache.set_bank_sources(
            sources,
            layer_residency=[HostResidency.PINNED.value, HostResidency.LOCKED.value],
        )


def test_set_bank_sources_locked_layer_rejects_prefill_overlap():
    # prefill overlap DMAs from registered banks; a LOCKED layer cannot feed it
    from freetoken.moe.host_banks import HostResidency
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    cache = OffloadMoeCache(
        num_layers=2, num_experts=4, cache_size=8, device=torch.device("cpu"),
        prefill_overlap=True,
    )
    cache.cpu_layer_ids = frozenset({1})
    sources = {
        "gate_up": [torch.randn(4, 32, 8) for _ in range(2)],
        "down": [torch.randn(4, 8, 16) for _ in range(2)],
    }
    with pytest.raises(ValueError, match="[Pp]refill overlap"):
        cache.set_bank_sources(
            sources,
            layer_residency=[HostResidency.PINNED.value, HostResidency.LOCKED.value],
        )


def test_locked_layer_prefill_materialize_copies_whole_layer_pageable():
    # the only movement a LOCKED layer needs: copy_missing's pageable branch copies the whole layer into slots [0, E) with position == expert id
    # stage the state materialize_layer would (its kernel is CUDA-only; the fixture cache lives on the CPU)
    cache, sources = _make_split_cache(num_layers=2, locked=(1,))

    cache._pending_src_layer = 1
    cache._pending_whole_layer = True
    cache.copy_missing()

    gate_up_cache, down_cache = (c for _, c in cache.banks)
    assert torch.equal(gate_up_cache[:4], sources["gate_up"][1])
    assert torch.equal(down_cache[:4], sources["down"][1])
    # (The pinned layers' staged JIT path is covered by the mocked tests above.)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_copy_plan_skips_locked_layers_and_keeps_fused_path():
    # _build_copy_plan must not resolve a device alias for a LOCKED layer; its descriptor row stays a 0 placeholder while the pinned layers keep the fused path
    cache, _ = _make_split_cache(num_layers=2, locked=(1,), device="cuda")

    assert cache._copy_fused_ok
    assert (cache._copy_src_ptrs[1] == 0).all(), "locked layer row must stay 0"
    assert (cache._copy_src_ptrs[0] != 0).all(), "pinned layer rows must resolve"


def test_locked_layer_copy_missing_rejects_ensure_experts_staging():
    # the pageable branch presumes materialize_layer's position == expert id; staging via ensure_experts (LRU slot remap) on a locked layer must fail loudly, not gather other experts' weights
    # stage the state ensure_experts would (its kernel is CUDA-only; the fixture cache lives on the CPU)
    cache, _ = _make_split_cache(num_layers=2, locked=(1,))

    cache._pending_src_layer = 1
    cache._pending_whole_layer = False
    with pytest.raises(RuntimeError, match="unpinned"):
        cache.copy_missing()


def test_requested_residency_routes_layer_settles(monkeypatch):
    # the ambient plan installed by load_expert_banks must route each layer's banks by label at both slow-path settle points (PinPipeline layer sink, list-valued pin_banks) and record that it was consulted
    # without a plan everything pins
    import freetoken.moe.host_banks as hb

    settled = []
    monkeypatch.setattr(hb.HostBank, "pin", lambda self: settled.append("pin"))
    monkeypatch.setattr(hb.HostBank, "lock", lambda self: settled.append("lock"))
    banks = {
        "gate_up": [hb.HostBank((4,), torch.uint8) for _ in range(3)],
        "down": [hb.HostBank((4,), torch.uint8) for _ in range(3)],
    }
    labels = [
        hb.HostResidency.PINNED.value,
        hb.HostResidency.LOCKED.value,
        hb.HostResidency.PAGEABLE.value,
    ]

    with hb.requested_residency(labels) as plan:
        with hb.PinPipeline() as pins:
            for layer_id in range(3):
                pins(layer_id, {name: per[layer_id] for name, per in banks.items()})
    # the single drain thread settles FIFO: layer 0 pins, layer 1 locks, layer 2 passes
    assert settled == ["pin", "pin", "lock", "lock"]
    assert plan.applied

    settled.clear()
    with hb.requested_residency(labels) as plan:
        hb.pin_banks(banks)
    assert settled == ["pin", "lock", "pin", "lock"]  # per name: layer 0 pin, 1 lock, 2 skip
    assert plan.applied

    settled.clear()
    hb.pin_banks(banks)  # no ambient plan -> every layer pins
    assert settled == ["pin"] * 6


def test_echo_residency_stamps_honored_requests_only():
    # load_expert_banks stamps the request onto the provider's ExpertBanks only when a settle point consulted the plan; an unconsulted plan keeps None (the engine's degrade signal)
    from freetoken.moe.expert_banks import ExpertBanks, _echo_residency
    from freetoken.moe.host_banks import HostResidency, _ResidencyPlan

    labels = [HostResidency.PINNED.value, HostResidency.LOCKED.value]
    banks = ExpertBanks("bf16", {"gate_up": [], "down": []})

    plan = _ResidencyPlan(labels)
    plan.residency_for(1)  # a settle point consulted the plan
    assert _echo_residency(banks, labels, plan).layer_residency == labels

    stale = _ResidencyPlan(labels)  # never consulted -> keep None + warn
    assert _echo_residency(banks, labels, stale).layer_residency is None
    assert _echo_residency(banks, None, None) is banks


def test_lock_failure_downgrades_echoed_residency(monkeypatch):
    # a failed mlock leaves the bank pageable; the plan and the echoed labels must report that instead of the requested LOCKED
    import freetoken.moe.host_banks as hb
    from freetoken.moe.expert_banks import ExpertBanks, _echo_residency

    def boom(addr, nbytes):
        raise OSError(12, "mlock denied")

    monkeypatch.setattr(hb, "_os_lock", boom)
    monkeypatch.setattr(hb, "_os_lock_failed", False)
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")  # keep the pinned layer off CUDA
    labels = [hb.HostResidency.PINNED.value, hb.HostResidency.LOCKED.value]

    banks = {"gate_up": [hb.HostBank((4,), torch.uint8) for _ in range(2)]}
    with hb.requested_residency(labels) as plan:
        hb.pin_banks(banks)
    assert plan.actual == {1: hb.HostResidency.PAGEABLE.value}
    echoed = _echo_residency(ExpertBanks("bf16", {}), labels, plan)
    assert echoed.layer_residency == [
        hb.HostResidency.PINNED.value, hb.HostResidency.PAGEABLE.value,
    ]

    monkeypatch.setattr(hb, "_os_lock_failed", False)
    with hb.requested_residency(labels) as plan2:
        with hb.PinPipeline() as pins:
            pins(1, {"gate_up": hb.HostBank((4,), torch.uint8)})
    assert plan2.actual == {1: hb.HostResidency.PAGEABLE.value}


def test_rebuild_resets_the_miss_counters_with_decode_freq():
    """A rebuild resizes the slot cache, so miss counts taken under the old size cannot be
    mixed with picks taken under the new one.

    ``decode_freq`` was already reset here; ``decode_miss_freq`` / ``prefill_miss_freq`` /
    ``prefill_chunks`` were added later and were left accumulating across the resize
    (external review, 2026-09-10). The visible damage is arithmetic: the placement cost
    function divides misses by picks, and a stale numerator over a fresh denominator gives
    miss > pick -- a per-token transfer cost larger than the number of routing decisions
    that could have caused one. They are one generation; reset together or not at all.
    """
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(
        num_layers=2,
        num_experts=4,
        cache_size=6,
        device=torch.device("cpu"),
    )
    cache.set_bank_sources(
        {
            "gate_up": [torch.randn(4, 32, 8), torch.randn(4, 32, 8)],
            "down": [torch.randn(4, 8, 16), torch.randn(4, 8, 16)],
        }
    )
    cache.decode_freq += 7
    cache.decode_miss_freq += 3
    cache.prefill_miss_freq += 5
    cache.prefill_chunks = 11
    cache.prefill_hit_rows = 13
    cache.prefill_total_rows = 17

    cache.rebuild(5)

    # decode_freq is the reference: whatever it does, the miss counters do too.
    assert int(cache.decode_freq.sum()) == 0
    assert int(cache.decode_miss_freq.sum()) == 0, "stale misses over fresh picks -> miss > pick"
    assert int(cache.prefill_miss_freq.sum()) == 0
    assert cache.prefill_chunks == 0
    assert cache.prefill_hit_rows == 0
    assert cache.prefill_total_rows == 0


def test_no_counter_takes_its_device_from_process_state():
    """Every tensor the cache allocates must name its device, not inherit torch's default.

    prefill_miss_freq did not (2026-09-10, found by review). It is accumulated against
    _prefill_slot_snapshot, which is pinned host memory, so host is the right home for it --
    but leaving that to the default made it the one tensor here whose device depends on
    process state. Under torch.set_default_device("cuda") it would allocate on the GPU while
    its accumulation partner stayed on the host, and the += in begin_prefill would raise at
    serve time. Nothing in python/ sets a default device, so the defect was latent.

    A CUDA fixture is not needed to catch it, and would not be enough anyway: the accumulation
    is gated on device.type == "cuda", so a CPU-constructed cache can never reach it. What is
    testable is the allocation. Setting the default to "meta" -- a device nothing here would
    ever want -- makes any tensor that leaves the choice to torch land somewhere visible,
    which covers tensors this test does not name yet.
    """
    from freetoken.moe.offload_cache import OffloadMoeCache

    torch.set_default_device("meta")
    try:
        cache = OffloadMoeCache(
            num_layers=2, num_experts=4, cache_size=6, device=torch.device("cpu")
        )
    finally:
        torch.set_default_device(None)

    stray = sorted(
        name
        for name, value in vars(cache).items()
        if torch.is_tensor(value) and value.device.type == "meta"
    )
    assert not stray, f"allocated on torch's default device instead of naming one: {stray}"


def test_count_routing_separates_picks_from_fetches():
    """decode_freq counts picks, decode_miss_freq counts transfers. The whole branch is this.

    Two picks of the same expert in one step are one fetch, and a pick to a resident expert is
    none -- which is why decode_miss_freq and not decode_freq is proportional to bytes. Nothing
    asserted that. Three mutations used to survive the suite: scatter_ -> scatter_add_ (which
    collapses the distinction), the prefill hit rule 2*E -> 0, and deleting the call outright.
    """
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(
        num_layers=2, num_experts=4, cache_size=6, device=torch.device("cpu")
    )
    cache.collect_decode_freq = True
    # expert 1 resident in slot 3, expert 3 in slot 0; 0 and 2 are not.
    cache.slot_for_id[0] = torch.tensor([-1, 3, -1, 0], dtype=cache.slot_for_id.dtype)

    cache._count_routing(0, torch.tensor([0, 0, 1, 2], dtype=torch.int32))

    assert cache.decode_freq[0].tolist() == [2, 1, 1, 0], "picks: expert 0 was chosen twice"
    assert cache.decode_miss_freq[0].tolist() == [1, 0, 1, 0], (
        "fetches: 0 and 2 were absent (once each, not twice for 0); 1 was resident"
    )
    assert cache.decode_freq[1].sum() == 0, "another layer's row must not move"

    # Second step, same routing, expert 0 now resident: a pick, not a fetch.
    cache.slot_for_id[0, 0] = 5
    cache._count_routing(0, torch.tensor([0, 0, 1, 2], dtype=torch.int32))
    assert cache.decode_freq[0].tolist() == [4, 2, 2, 0]
    assert cache.decode_miss_freq[0].tolist() == [1, 0, 2, 0], "a resident pick moves nothing"

    assert (cache.decode_miss_freq <= cache.decode_freq).all(), "miss > pick is impossible"

    # NOT covered here: the prefill side's own hit rule (snapshot < 2*num_experts). Its
    # accumulation is gated on device.type == "cuda", so a CPU cache cannot reach it and
    # flipping the threshold to < 0 still passes this suite. It is checked end-to-end instead,
    # by freetoken-systest tools/prefill-identity.py against the engine's own row counters.


def test_reset_clears_the_routing_histogram(monkeypatch):
    """reset() is called by warmup and by graph capture to undo their synthetic residency.

    It did not clear the counters, so that traffic stayed in the histogram for the whole run --
    on a config where _warmup_prefill fires, two all-miss prefill chunks in every cell before
    any real request, pulling exactly the skew the instrument measures toward uniform.

    reset_cache is a CUDA kernel, so it is stubbed; what is under test is the wiring.
    """
    from freetoken.moe import offload_kernels
    from freetoken.moe.offload_cache import OffloadMoeCache

    monkeypatch.setattr(offload_kernels, "reset_cache", lambda cache: None)
    cache = OffloadMoeCache(
        num_layers=2, num_experts=4, cache_size=6, device=torch.device("cpu")
    )
    cache.decode_freq += 7
    cache.decode_miss_freq += 3
    cache.prefill_miss_freq += 5
    cache.prefill_chunks = 11

    cache.reset()

    assert int(cache.decode_freq.sum()) == 0
    assert int(cache.decode_miss_freq.sum()) == 0
    assert int(cache.prefill_miss_freq.sum()) == 0
    assert cache.prefill_chunks == 0


def test_ensure_experts_actually_calls_the_counter(monkeypatch):
    """The counting is only as good as the call site; deleting it used to pass everything.

    ensure_experts' kernel is CUDA, so it is stubbed -- what is under test is that the wiring
    from ensure_experts / ensure_experts_hybrid to _count_routing is still there, and that it
    runs BEFORE the kernel (which rewrites expert_ids into slots in place).
    """
    from freetoken.moe import offload_kernels
    from freetoken.moe.offload_cache import OffloadMoeCache

    seen = []

    def fake(cache, layer_id, expert_ids, *a, **kw):
        # By now the counters must already have moved, or the ids they read were slots.
        seen.append(int(cache.decode_freq.sum()))

    monkeypatch.setattr(offload_kernels, "ensure_experts", fake)
    monkeypatch.setattr(offload_kernels, "ensure_experts_hybrid", fake)
    cache = OffloadMoeCache(
        num_layers=2, num_experts=4, cache_size=6, device=torch.device("cpu")
    )
    cache.collect_decode_freq = True

    cache.ensure_experts(0, torch.tensor([0, 1], dtype=torch.int32))
    assert seen == [2], "counted after the kernel, or not at all"

    cache.ensure_experts_hybrid(1, torch.tensor([2, 3], dtype=torch.int32))
    assert seen == [2, 4], "the hybrid call site does not count"


def test_reset_clears_both_counter_generations(monkeypatch):
    """``reset()`` means "this traffic never happened", and that has to cover both.

    Its callers are warmup and CUDA-graph capture (engine/graph.py), which reset residency
    precisely so their synthetic traffic does not count. It cleared the run-cumulative
    histogram and left the window counters (``lru_stats``, ``prefill_hit_rows`` /
    ``prefill_total_rows``, ``stat_*``) holding that same traffic -- so the first snapshot
    after startup carried a histogram of real picks beside window totals that still included
    warmup, and the identity ``delta decode_miss_freq.sum() == window_missing`` failed on
    exactly that window (external review, 2026-09-11). The end-to-end check that verified the
    identity over 211 windows could not see it: it reads dumps that already exist on disk, so
    the contaminated first window was never in the sample.

    ``reset_stats`` stays the every-report case and must NOT touch the histogram -- that
    direction is covered by test_reset_clears_the_routing_histogram's sibling.
    """
    from freetoken.moe import offload_kernels
    from freetoken.moe.offload_cache import OffloadMoeCache

    # reset() delegates residency to a Triton kernel, which needs a live driver; the change
    # under test is the Python either side of it. Stubbing the kernel keeps this a CPU test
    # and keeps it honest about what it covers -- residency reset is not asserted here.
    monkeypatch.setattr(offload_kernels, "reset_cache", lambda cache: None)

    cache = OffloadMoeCache(
        num_layers=2, num_experts=4, cache_size=6, device=torch.device("cpu")
    )
    cache.set_bank_sources(
        {
            "gate_up": [torch.randn(4, 32, 8), torch.randn(4, 32, 8)],
            "down": [torch.randn(4, 8, 16), torch.randn(4, 8, 16)],
        }
    )
    cache.decode_freq += 7
    cache.decode_miss_freq += 3
    cache.prefill_miss_freq += 5
    cache.prefill_chunks = 11
    # The window generation, which reset() used to leave behind.
    cache.lru_stats += 19
    cache.prefill_hit_rows = 13
    cache.prefill_total_rows = 17
    cache.stat_missing += 23
    cache.stat_active += 29
    cache.stat_calls += 31

    cache.reset()

    assert int(cache.decode_freq.sum()) == 0
    assert int(cache.decode_miss_freq.sum()) == 0
    assert int(cache.prefill_miss_freq.sum()) == 0
    assert cache.prefill_chunks == 0
    assert int(cache.lru_stats.sum()) == 0, "window totals survived a reset"
    assert cache.prefill_hit_rows == 0 and cache.prefill_total_rows == 0
    assert int(cache.stat_missing.sum()) == 0, "hybrid window totals survived a reset"
    assert int(cache.stat_active.sum()) == 0
    assert int(cache.stat_calls.sum()) == 0


def test_window_totals_follow_the_decode_target():
    """``ensure_experts_hybrid`` runs a different kernel and never writes ``lru_stats``.

    The log line branched on ``decode_target`` and the dump did not, so a hybrid run wrote
    three zeros into the snapshot beside a log line showing misses -- and ``decode_target``
    in the payload could not recover them, because the values were gone. One getter now, and
    both callers use it.
    """
    from freetoken.moe.offload_cache import OffloadMoeCache

    def build(target):
        c = OffloadMoeCache(
            num_layers=2, num_experts=4, cache_size=6, device=torch.device("cpu"),
            decode_target=target,
        )
        c.lru_stats += torch.tensor([[2, 3, 5]], dtype=torch.int64)
        c.stat_active += 7
        c.stat_missing += 11
        c.stat_calls += 13
        return c

    assert build("gpu").window_totals() == (4, 6, 10)  # lru_stats.sum(0) over 2 layers
    assert build("hybrid").window_totals() == (7, 11, 13)


def test_real_row_counters_exclude_the_padded_rows():
    """pad_batch's dummy rows are counted; the _real pair is the same counters without them.

    A batch of 3 into the bs=4 graph makes exactly one row in four synthetic, and those picks
    land in decode_freq like any other. Keeping BOTH counters is what makes the contamination
    checkable from a single run: their difference must equal the padded rows' picks exactly,
    which no two-run comparison can establish (turning graphs off to remove the padding also
    changes the kernels, and with them the greedy output).
    """
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(
        num_layers=1, num_experts=8, cache_size=8, device=torch.device("cpu")
    )
    # 4 rows of 2 picks; rows 0-2 are real, row 3 is pad_batch's dummy.
    ids = torch.tensor([[0, 1], [1, 2], [2, 3], [6, 7]], dtype=torch.int32)
    cache._real_rows.fill_(3)
    cache._count_routing(0, ids)

    assert cache.decode_freq[0].tolist() == [1, 2, 2, 1, 0, 0, 1, 1]
    # The dummy row's two picks are gone from the real counter and sit in the sentinel column.
    assert cache.decode_freq_real[0][:8].tolist() == [1, 2, 2, 1, 0, 0, 0, 0]
    assert int(cache.decode_freq_real[0][8]) == 2

    # The identity the whole design rests on: the gap IS the padded rows' picks.
    gap = int(cache.decode_freq.sum()) - int(cache.decode_freq_real[0][:8].sum())
    assert gap == 2, "the two counters differ by something other than the padded rows"

    # And with nothing padded, the two counters agree exactly.
    cache.reset_routing_freq()
    cache._real_rows.fill_(4)
    cache._count_routing(0, ids)
    assert cache.decode_freq[0].tolist() == cache.decode_freq_real[0][:8].tolist()
    assert int(cache.decode_freq_real[0][8]) == 0


def test_real_rows_defaults_to_masking_nothing():
    """A caller that never sets _real_rows must get the old behaviour, not empty counters.

    Only GraphRunner.pad_batch sets it, and only while collecting; every other path -- unit
    tests, a rebuild, an eager forward that skipped pad_batch -- would otherwise silently
    produce a histogram of zeros that looks like "no traffic".
    """
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(
        num_layers=1, num_experts=8, cache_size=8, device=torch.device("cpu")
    )
    ids = torch.tensor([[0, 1], [6, 7]], dtype=torch.int32)
    cache._count_routing(0, ids)  # _real_rows untouched
    assert cache.decode_freq[0].tolist() == cache.decode_freq_real[0][:8].tolist()
    assert int(cache.decode_freq_real[0][8]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the counting kernel is CUDA-only")
def test_the_kernel_matches_the_reference_counter():
    """One Triton launch must produce exactly what fifteen tensor ops produced.

    The kernel exists because the tensor-op form was dispatch-bound -- 0.92-0.97 us per launch,
    15 launches per layer per step inside the captured decode graph. Speed is not worth a
    different histogram, so this compares all four counters plus the pad sentinel against
    _count_routing_ref on the same inputs, across batch shapes and residency patterns, with
    the pad boundary swept over every row count including both ends.
    """
    from freetoken.moe.offload_cache import OffloadMoeCache

    dev = torch.device("cuda")
    E, TOP_K = 64, 8
    torch.manual_seed(20260911)
    for nrow in (1, 2, 3, 4, 8):
        for real_rows in range(0, nrow + 1):
            ids = torch.randint(0, E, (nrow, TOP_K), dtype=torch.int32, device=dev)
            slots = torch.where(
                torch.rand((2, E), device=dev) < 0.4,
                torch.full((2, E), -1, dtype=torch.int32, device=dev),
                torch.randint(0, 128, (2, E), dtype=torch.int32, device=dev),
            )

            def build():
                c = OffloadMoeCache(num_layers=2, num_experts=E, cache_size=128, device=dev)
                c.slot_for_id.copy_(slots)
                c._real_rows.fill_(real_rows)
                return c

            k, r = build(), build()
            k._count_routing(1, ids)          # cuda -> the kernel
            r._count_routing_ref(1, ids)      # the tensor ops, explicitly
            where = f"nrow={nrow} real_rows={real_rows}"
            assert torch.equal(k.decode_freq, r.decode_freq), f"decode_freq: {where}"
            assert torch.equal(k.decode_miss_freq, r.decode_miss_freq), f"miss: {where}"
            assert torch.equal(k.decode_freq_real, r.decode_freq_real), f"freq_real: {where}"
            assert torch.equal(k.decode_miss_freq_real, r.decode_miss_freq_real), f"miss_real: {where}"
            # And the identity the sentinel exists for, at every boundary.
            pad = int(k.decode_freq_real[1][E])
            assert pad == (nrow - real_rows) * TOP_K, f"sentinel: {where}"
