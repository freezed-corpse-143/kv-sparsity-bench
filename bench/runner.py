"""Central benchmark runner with KV-cache eviction."""

import time

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .core import HookManager
from .strategies import EvictionStrategy, BaselineStrategy
from .tasks import Task


class BenchmarkRunner:
    """Runs generation with attention-score capture and KV-cache eviction."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        strategy: EvictionStrategy,
        config: dict,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.config = config
        self._evicts = not isinstance(strategy, BaselineStrategy)
        self.score_buffer = strategy.score_buffer
        self.hook_manager = HookManager(self.score_buffer)
        self.hook_manager.install(model)

    def run(self, task: Task) -> dict:
        return task.run(self)

    def generate_with_eviction(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
    ) -> tuple[torch.Tensor, dict]:
        device = self.model.device
        self.strategy.reset()
        self.score_buffer.reset(device)

        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True, output_attentions=True)
        prefill_time = time.perf_counter() - start
        cache = outputs.past_key_values
        plen = input_ids.size(1)
        n_layers = len(cache)

        # alive_ids[layer][cache_pos] = original_token_id
        alive_ids = [list(range(plen)) for _ in range(n_layers)]

        generated = input_ids.clone()
        gen_times = []
        kv_sizes = []

        for step in range(max_new_tokens):
            last = generated[:, -1:]
            start = time.perf_counter()
            with torch.no_grad():
                out = self.model(last, past_key_values=cache,
                                 use_cache=True, output_attentions=True)
            gen_times.append(time.perf_counter() - start)
            tid = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, tid], dim=1)

            new_id = plen + step

            if self._evicts and step < max_new_tokens - 1:
                for li in range(n_layers):
                    k, v = cache.layers[li].keys, cache.layers[li].values
                    sz, budget = k.size(2), self.strategy.budget_controller.get_budget(
                        step + plen, li)
                    if sz > budget:
                        keep = self._pick_keep(li, alive_ids[li], budget)
                        if keep and len(keep) < sz:
                            kt = torch.tensor(keep, device=k.device, dtype=torch.long)
                            cache.layers[li].keys = k[:, :, kt, :]
                            cache.layers[li].values = v[:, :, kt, :]
                            alive_ids[li] = [alive_ids[li][i] for i in keep]
                    alive_ids[li].append(new_id)
            else:
                for li in range(n_layers):
                    if len(alive_ids[li]) <= plen + step:
                        alive_ids[li].append(new_id)

            for li in range(n_layers):
                kv_sizes.append(cache.layers[li].keys.size(2))

        return generated, {
            "prefill_time_ms": prefill_time * 1000,
            "avg_gen_time_ms": (sum(gen_times) / len(gen_times)) * 1000,
            "prefill_len": plen,
            "gen_len": max_new_tokens,
            "tokens_per_sec": max_new_tokens / max(sum(gen_times), 1e-6),
            "avg_kv_size": sum(kv_sizes) / len(kv_sizes) if kv_sizes else 0,
        }

    def _pick_keep(self, li: int, alive: list[int], budget: int) -> list[int]:
        """Return cache positions to keep, based on accumulated attention scores."""
        if not alive:
            return []
        scores = self.score_buffer.scores[li].sum(dim=0)  # (max_seq,)
        ps = [(i, scores[oid].item() if oid < scores.size(0) else 0.0)
              for i, oid in enumerate(alive)]
        k = min(budget, len(ps))
        return sorted([p[0] for p in sorted(ps, key=lambda x: -x[1])[:k]])
