"""H2O: Heavy-Hitter Oracle — keep top-k tokens by accumulated attention score.

Reference: H2O: Heavy-Hitter Oracle for Efficient Generative Inference of
Large Language Models (ACL 2024).
"""

import torch

from .base import EvictionStrategy, StrategyRegistry


@StrategyRegistry.register_decorator("h2o")
class H2OStrategy(EvictionStrategy):
    """Keep tokens with the highest cumulative attention scores.

    Accumulates softmax attention scores across all generation steps,
    then retains only the top-k tokens per layer at each eviction step.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.evict_interval = config.get("strategy", {}).get("evict_interval", 1)
        self._step = 0

    def reset(self):
        super().reset()
        self._step = 0

    def decide_eviction(
        self, layer_idx: int, step: int,
    ) -> list[int] | torch.Tensor:
        budget = self.budget_controller.get_budget(step, layer_idx)
        # Get top-k indices from accumulated score buffer
        indices = self.score_buffer.get_topk(budget, layer=layer_idx)
        return indices.tolist()

    def should_evict(self, step: int) -> bool:
        """Only evict every N steps to reduce overhead."""
        return step > 0 and step % self.evict_interval == 0
