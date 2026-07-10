"""RocketKV-light: Two-stage KV cache compression.

Reference: RocketKV: Accelerating Long-Context LLM Inference via
Two-Stage KV Cache Compression (ICML 2025, NVIDIA).

Stage 1 — Coarse-grain eviction: 1D average pooling on attention scores
to identify important token regions. Keep top pooled regions + recent window.

Stage 2 — Fine-grain top-k: On remaining tokens, keep top-k by score.
"""

import math
import torch
import torch.nn.functional as F

from .base import EvictionStrategy, StrategyRegistry


@StrategyRegistry.register_decorator("rocketkv")
class RocketKVStrategy(EvictionStrategy):
    """Two-stage KV cache compression.

    Stage 1 pools attention scores with kernel_size=63 to find important
    token regions, keeps top ones + recent_window tokens.
    Stage 2 keeps top-k by score from the remaining budget.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        sc = config.get("strategy", {})
        self.kernel_size = sc.get("kernel_size", 63)
        self.window_size = sc.get("window_size", 32)
        # Stage 2 top-k ratio (fraction of total budget)
        self.stage2_ratio = sc.get("stage2_ratio", 0.5)
        self._step = 0

    def reset(self):
        super().reset()
        self._step = 0

    def decide_eviction(
        self, layer_idx: int, step: int,
    ) -> list[int]:
        budget = self.budget_controller.get_budget(step, layer_idx)
        if budget <= 0:
            return []

        scores = self.score_buffer.scores
        if scores is None or scores.count_nonzero().item() == 0:
            return list(range(min(budget, step + 1)))

        # Get per-layer scores summed across heads
        layer_scores = scores[layer_idx].sum(dim=0)  # (max_seq,)
        seq_len = min(step + 1, layer_scores.size(0))
        active = layer_scores[:seq_len]

        if seq_len <= budget:
            return list(range(seq_len))

        # Stage 1: Coarse-grain eviction via 1D pooling
        # Pad to multiple of kernel_size for even pooling
        pad = (self.kernel_size - seq_len % self.kernel_size) % self.kernel_size
        if pad > 0:
            pooled = F.avg_pool1d(
                active.view(1, 1, -1),
                kernel_size=self.kernel_size,
                stride=self.kernel_size,
            )
        else:
            pooled = active.view(1, 1, -1)
            if pooled.size(-1) > 0:
                pooled = F.avg_pool1d(
                    pooled, kernel_size=self.kernel_size,
                    stride=self.kernel_size,
                )

        pooled = pooled[0, 0]  # (n_regions,)

        # Budget split: stage1 budget for coarse regions, stage2 for fine top-k
        stage2_budget = max(4, int(budget * self.stage2_ratio))
        stage1_budget = budget - stage2_budget

        if stage1_budget <= 0:
            # Pure top-k fallback
            return active.topk(budget).indices.tolist()

        # Pick top pooled regions
        n_regions = min(pooled.size(0), max(1, stage1_budget // self.kernel_size))
        top_regions = pooled.topk(n_regions).indices.tolist()

        # Map region indices to token indices
        keep_set: set[int] = set()
        for r in top_regions:
            start = r * self.kernel_size
            end = min(start + self.kernel_size, seq_len)
            keep_set.update(range(start, end))

        # Add recent window
        recent_start = max(0, seq_len - self.window_size)
        keep_set.update(range(recent_start, seq_len))

        # Stage 2: fine-grain top-k from remaining
        if len(keep_set) > budget:
            # Sort by score and take top budget
            keep_list = sorted(keep_set)
            scores_list = [active[i].item() for i in keep_list]
            ranked = sorted(
                range(len(keep_list)),
                key=lambda i: scores_list[i],
                reverse=True,
            )[:budget]
            keep_set = {keep_list[i] for i in ranked}

        return sorted(keep_set)
