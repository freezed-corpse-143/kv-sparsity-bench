"""Baseline strategy: no eviction, keep all KV entries."""

from typing import Any

import torch

from .base import EvictionStrategy, StrategyRegistry


@StrategyRegistry.register_decorator("baseline")
class BaselineStrategy(EvictionStrategy):
    """Keep all KV tokens — no compression. Reference line."""

    def decide_eviction(
        self, layer_idx: int, step: int,
    ) -> list[int] | torch.Tensor:
        raise NotImplementedError(
            "Baseline never evicts; use the runner's no-op path."
        )
