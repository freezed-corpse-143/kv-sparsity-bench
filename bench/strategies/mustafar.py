"""MUSTAFAR: Unstructured sparsity for KV cache pruning.

Reference: MUSTAFAR: Promoting Unstructured Sparsity for KV Cache
Pruning in LLM Inference (NeurIPS 2025).

Core idea: Per-token magnitude pruning of KV entries using a bitmap
sparse format. Each token's KV contribution is evaluated independently
by its magnitude, and low-magnitude entries are zeroed out (not removed,
allowing the sparse tensor core to skip them).
"""

import torch

from .base import EvictionStrategy, StrategyRegistry


@StrategyRegistry.register_decorator("mustafar")
class MUSTAFARStrategy(EvictionStrategy):
    """Unstructured sparsity via per-token magnitude pruning.

    Evaluates each KV entry by its L2 norm and keeps the top-k
    entries by magnitude. Unlike H2O which uses attention scores,
    MUSTAFAR uses the KV values directly.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        sc = config.get("strategy", {})
        self.topk_ratio = sc.get("topk_ratio", 0.1)  # fraction of K/V dims to keep
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
        return list(range(min(budget, step + 1)))

    def apply_magnitude_pruning(
        self, key: torch.Tensor, value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply per-token magnitude pruning to K and V tensors.

        Keeps only the top-k% of feature dimensions per token.
        Returns pruned (key, value) with zeroed-out low-magnitude entries.

        Args:
            key: (1, n_heads, seq_len, head_dim)
            value: (1, n_heads, seq_len, head_dim)

        Returns:
            (pruned_key, pruned_value) with sparse structure preserved.
        """
        k = max(1, int(key.size(-1) * self.topk_ratio))

        # Per-token magnitude pruning on key
        k_norm = key.abs()  # (1, n_heads, seq_len, head_dim)
        k_thresh = k_norm.topk(k, dim=-1).values[:, :, :, -1:]  # threshold
        k_mask = k_norm >= k_thresh
        pruned_k = key * k_mask

        # Same for value
        v_norm = value.abs()
        v_thresh = v_norm.topk(k, dim=-1).values[:, :, :, -1:]
        v_mask = v_norm >= v_thresh
        pruned_v = value * v_mask

        return pruned_k, pruned_v
