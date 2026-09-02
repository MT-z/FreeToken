"""effective_context_length: what /v1/models and /v1/stats advertise as the context window."""
from types import SimpleNamespace

from freetoken.server.stats import derive_model_card, effective_context_length


def _config(ceiling):
    return SimpleNamespace(
        max_seq_len=ceiling,
        served_model_name="m",
        model_config=SimpleNamespace(has_linear_attention=False, has_swa_attention=False, is_moe=False),
    )


def test_ceiling_alone_when_nothing_else_is_known():
    assert effective_context_length(_config(262144)) == 262144
    assert effective_context_length(_config(262144), cache_pools=None, last_rebuild=None) == 262144


def test_pool_limit_wins_over_the_ceiling():
    assert effective_context_length(_config(262144), {"max_seq_len": 8256}) == 8256


def test_rebuild_wins_over_the_load_time_pool_only_when_it_succeeded():
    pools = {"max_seq_len": 8256}
    assert effective_context_length(_config(262144), pools, {"status": "ok", "max_seq_len": 131072}) == 131072
    assert effective_context_length(_config(262144), pools, {"status": "failed", "max_seq_len": 131072}) == 8256
    assert effective_context_length(_config(262144), pools, {"status": "busy", "max_seq_len": 131072}) == 8256


def test_older_engine_pool_sizes_are_multiplied_out():
    assert effective_context_length(_config(262144), {"num_pages": 2049, "page_size": 64}) == 131136


def test_never_past_the_ceiling_and_never_raises():
    assert effective_context_length(_config(4096), {"max_seq_len": 8256}) == 4096
    assert effective_context_length(SimpleNamespace(), {"max_seq_len": 8256}) is None  # no max_seq_len
    assert effective_context_length(_config(0), {"max_seq_len": 8256}) is None


def test_stats_model_card_uses_the_same_number():
    card = derive_model_card(_config(262144), {"max_seq_len": 8256}, None)
    assert card["ctx"] == 8256
    assert derive_model_card(_config(262144))["ctx"] == 262144  # old call shape still works
