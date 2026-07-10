"""CurDKV: Value-guided KV compression via CUR decomposition.

Reference: Value-Guided KV Compression for LLMs via Approximated
CUR Decomposition (NeurIPS 2025).

Core idea: Use CUR matrix decomposition on the value matrix V to compute
leverage scores, which guide which KV entries to keep. Tokens with high
leverage scores (high row/column norms in V) contribute more to the
attention output and should be retained.
"""

import torch

from .base import EvictionStrategy, StrategyRegistry


@StrategyRegistry.register_decorator("curdkv")
class CurDKVStrategy(EvictionStrategy):
    """Value-guided eviction using CUR leverage scores.

    Combines attention scores with value-matrix column norms to decide
    which KV entries to keep. Tokens with high value magnitude and high
    attention accumulation are prioritized.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        sc = config.get("strategy", {})
        # Weight for value-based score vs attention-based score
        self.value_weight = sc.get("value_weight", 0.4)
        # Sample rate for leverage score computation (reduce overhead)
        self.sample_rate = sc.get("sample_rate", 0.5)
        self._step = 0

        # Cache value norms computed during generation
        self._value_norms: dict[int, torch.Tensor] = {}

    def reset(self):
        super().reset()
        self._step = 0
        self._value_norms = {}

    def update_value_norms(self, layer_idx: int, value: torch.Tensor):
        """Store column norms of the value matrix for leverage scoring.

        Args:
            layer_idx: Layer index.
            value: (1, n_heads, seq_len, head_dim) value tensor.
        """
        # Norm of each value vector across the head dimension
        # (1, n_heads, seq_len, head_dim) -> (seq_len,)
        norms = value[0].norm(dim=-1).mean(dim=0)  # avg across heads
        self._value_norms[layer_idx] = norms

    def decide_eviction(
        self, layer_idx: int, step: int,
    ) -> list[int]:
        budget = self.budget_controller.get_budget(step, layer_idx)
        if budget <= 0:
            return []

        scores = self.score_buffer.scores
        if scores is None or scores.count_nonzero().item() == 0:
            return list(range(min(budget, step + 1)))

        seq_len = min(step + 1, scores.size(-1))
        if seq_len <= budget:
            return list(range(seq_len))

        # Attention-based scores (summed across heads)
        attn_scores = scores[layer_idx].sum(dim=0)[:seq_len]  # (seq_len,)

        # Value-based leverage scores
        value_norms = self._value_norms.get(layer_idx)
        if value_norms is not None and value_norms.size(0) >= seq_len:
            v_scores = value_norms[:seq_len]
            # Normalize
            v_scores = v_scores / (v_scores.max() + 1e-8)
        else:
            v_scores = torch.ones(seq_len)

        # Normalize attention scores
        a_scores = attn_scores
        if a_scores.max() > 0:
            a_scores = a_scores / a_scores.max()

        # Combined score: weighted sum
        combined = (1 - self.value_weight) * a_scores + self.value_weight * v_scores

        # Keep top-k by combined score
        return combined.topk(budget).indices.tolist()
