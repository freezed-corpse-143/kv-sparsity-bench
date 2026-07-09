"""Central benchmark runner orchestrating model, strategy, and tasks."""

import time

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .core import HookManager
from .strategies import EvictionStrategy, StrategyRegistry
from .tasks import Task


class BenchmarkRunner:
    """Orchestrates model loading, strategy hooking, and task execution."""

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
        # Detect if strategy actually evicts (check class, not call)
        from .strategies.baseline import BaselineStrategy
        self._evicts = not isinstance(strategy, BaselineStrategy)
        self.score_buffer = strategy.score_buffer
        self.hook_manager = HookManager(self.score_buffer)
        self.hook_manager.install(model)

    def run(self, task: Task) -> dict:
        return task.run(self)

    def generate_with_eviction(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 32,
    ) -> tuple[torch.Tensor, dict]:
        device = self.model.device
        self.strategy.reset()
        self.score_buffer.reset(device)

        # Prefill
        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_attentions=True,
            )
        prefill_time = time.perf_counter() - start
        past_key_values = outputs.past_key_values
        prefill_len = input_ids.size(1)

        generated = input_ids.clone()
        gen_times = []
        kv_sizes = []

        for step in range(max_new_tokens):
            last_token = generated[:, -1:]

            start = time.perf_counter()
            with torch.no_grad():
                outputs = self.model(
                    input_ids=last_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=True,
                )
            step_time = time.perf_counter() - start
            gen_times.append(step_time)

            logits = outputs.logits
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            # Apply eviction (skip for baseline)
            full_pkv = outputs.past_key_values
            if full_pkv is not None and step < max_new_tokens - 1 and self._evicts:
                compressed = []
                for layer_idx, layer_data in enumerate(full_pkv):
                    k = layer_data[0] if isinstance(layer_data, tuple) else layer_data
                    v = layer_data[1] if isinstance(layer_data, tuple) and len(layer_data) > 1 else k

                    keep_idx = self.strategy.decide_eviction(layer_idx, step + prefill_len)
                    if keep_idx is not None and len(keep_idx) < k.size(2):
                        keep_t = torch.tensor(keep_idx, device=k.device, dtype=torch.long)
                        k = k[:, :, keep_t, :]
                        v = v[:, :, keep_t, :]
                        kv_sizes.append(k.size(2))

                    if isinstance(layer_data, tuple):
                        compressed.append((k, v, *list(layer_data[2:])))
                    else:
                        compressed.append(k)

                past_key_values = type(full_pkv)(compressed)
            else:
                past_key_values = full_pkv
                if full_pkv is not None:
                    for layer_data in full_pkv:
                        k = layer_data[0] if isinstance(layer_data, tuple) else layer_data
                        kv_sizes.append(k.size(2))

        metrics = {
            "prefill_time_ms": prefill_time * 1000,
            "avg_gen_time_ms": (sum(gen_times) / len(gen_times)) * 1000,
            "prefill_len": prefill_len,
            "gen_len": max_new_tokens,
            "tokens_per_sec": max_new_tokens / max(sum(gen_times), 1e-6),
            "avg_kv_size": sum(kv_sizes) / max(len(kv_sizes), 1) if kv_sizes else 0,
        }

        return generated, metrics
