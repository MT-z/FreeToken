from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from freetoken.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from freetoken.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None
    # Presence penalty, when any request in the batch asked for one: ``penalties`` [bs] and
    # ``seen`` [bs, vocab] (1.0 where that request has already emitted the token). Both None
    # when no request wants it, so the common path allocates and computes nothing.
    penalties: torch.Tensor | None = None
    seen: torch.Tensor | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    from freetoken.kernel.backend import is_flashinfer_installed

    if is_flashinfer_installed():
        import flashinfer.sampling as sampling
    else:
        import freetoken.kernel.triton.sampling as sampling

    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def _presence(self, batch: Batch) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """``(penalties [bs], seen [bs, vocab])`` for the presence penalty, or ``(None, None)``.

        ``seen`` is rebuilt each step rather than carried between them. It is not small at
        width -- 4.6 MiB at bs=8 but 74 MiB at bs=128 on a 152k vocab -- yet the rebuild is
        not what costs: zeros+scatter+H2D does not grow from bs=64 to bs=128 (0.41 -> 0.34
        ms), the caching allocator serving the repeated ``zeros`` out of a block it already
        holds. The dominant term is the apply in ``sample`` below, which upcasts the whole
        bf16 logits tensor to float32 and reads ``seen``: 0.58 ms of the 0.92 ms total at
        bs=128. So a persistent per-request bitmap would buy only the cheaper half and still
        owe slot-reuse, abort and finish bookkeeping; the lever is fusing the penalty into
        the sampling kernel, which drops the materialization and the float32 copy both. Only
        tokens the request itself generated count -- ``input_ids[:prompt_len]`` is the prompt.
        """
        pens = [r.sampling_params.presence_penalty for r in batch.reqs]
        if not any(pens):
            return None, None
        gen = [r.input_ids[r.prompt_len:] for r in batch.reqs]
        width = max((int(g.numel()) for g in gen), default=0)
        if width == 0:  # nothing generated yet: no token can be a repeat
            return None, None
        # Pad with the sentinel column vocab_size, scatter, then drop it -- padding with a
        # real id (0) would penalize that token for every short request in the batch.
        idx = torch.full((len(gen), width), self.vocab_size, dtype=torch.int64)
        for i, g in enumerate(gen):
            idx[i, : g.numel()] = g.to(torch.int64)
        idx = idx.pin_memory().to(self.device, non_blocking=True)
        seen = torch.zeros((len(gen), self.vocab_size + 1), dtype=torch.float32,
                           device=self.device)
        seen.scatter_(1, idx, 1.0)
        return make_device_tensor(pens, torch.float32, self.device), seen[:, : self.vocab_size]

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        penalties, seen = self._presence(batch)
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None, penalties=penalties, seen=seen)

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p,
                                 penalties=penalties, seen=seen)

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.penalties is not None:
                # On the raw logits, before the temperature divide -- the penalty is a fixed
                # logit offset, so scaling it by 1/temperature would make its strength depend
                # on the temperature (OpenAI and vLLM both apply it here).
                logits = logits.float() - args.penalties[:, None] * args.seen
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
