"""SnapKV: cluster prompt KV and keep centroids + recent window.

Reference: SnapKV: LLM Knows What You Are Looking For (ACL 2024).
"""

import torch

from .base import EvictionStrategy, StrategyRegistry


@StrategyRegistry.register_decorator("snapkv")
class SnapKVStrategy(EvictionStrategy):
    """Keep cluster centroids from prompt + recent window during decoding.

    During prefill, prompt KV entries are clustered. During generation,
    only the centroids + last N recent tokens are retained.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        strategy_cfg = config.get("strategy", {})
        self.prompt_cluster_ratio = strategy_cfg.get("prompt_cluster_ratio", 0.2)
        self.recent_window = strategy_cfg.get("recent_window", 16)
        self._step = 0
        self._prompt_len = 0
        self._cluster_centroids: list[torch.Tensor] | None = None

    def reset(self):
        super().reset()
        self._step = 0
        self._prompt_len = 0
        self._cluster_centroids = None

    def set_prompt_len(self, length: int):
        """Record prompt length and compute centroids."""
        self._prompt_len = length
        self._compute_centroids()

    def _compute_centroids(self):
        """Compute cluster centroids from accumulated scores.

        Simplified version: use score buffer to pick representative tokens
        instead of full KMeans clustering on the KV values.
        """
        budget = self.budget_controller.get_budget(0, 0)
        n_cluster = max(4, int(budget * self.prompt_cluster_ratio))
        self.n_cluster = n_cluster

    def decide_eviction(
        self, layer_idx: int, step: int,
    ) -> list[int] | torch.Tensor:
        budget = self.budget_controller.get_budget(step, layer_idx)
        n_recent = min(self.recent_window, step)

        # Get top prompt tokens from score buffer
        scores = self.score_buffer.scores
        if scores is not None and self._prompt_len > 0:
            # Sum scores across all layers and heads for prompt tokens
            agg = scores[:, :, :self._prompt_len].sum(dim=(0, 1))
            top_prompt = agg.topk(self.n_cluster).indices.tolist()
        else:
            top_prompt = list(range(min(self.n_cluster, self._prompt_len)))

        # Recent decoding tokens
        recent = list(range(max(0, step - n_recent), step + 1))

        # Combine and deduplicate
        keep = sorted(set(top_prompt + recent))
        return keep
