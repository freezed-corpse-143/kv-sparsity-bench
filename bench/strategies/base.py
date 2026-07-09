"""Eviction strategy base class and registry."""

from abc import ABC, abstractmethod

import torch

from bench.core.score_buffer import ScoreBuffer
from bench.core.cache_manager import CacheManager
from bench.core.budget_controller import BudgetController


class EvictionStrategy(ABC):
    """Base class for all KV-cache eviction/compression strategies.

    Each strategy implements decide_eviction() — the core logic that
    determines which token positions to retain at each layer.
    Shared infrastructure (ScoreBuffer, CacheManager, BudgetController)
    is provided by the framework.
    """

    def __init__(self, config: dict):
        self.config = config
        self.name = self.__class__.__name__
        self.score_buffer = ScoreBuffer(config)
        self.cache_manager = CacheManager(config)
        self.budget_controller = BudgetController(config)

    @abstractmethod
    def decide_eviction(
        self, layer_idx: int, step: int,
    ) -> list[int] | torch.Tensor:
        """Return indices of tokens to retain for this layer at this step."""

    def post_init(self):
        """Called after model and hooks are installed. Override for per-run setup."""
        pass

    def reset(self):
        """Called at the start of each new sequence."""
        self.cache_manager.reset()


class StrategyRegistry:
    """Registry mapping strategy names to classes."""

    _strategies: dict[str, type[EvictionStrategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_cls: type[EvictionStrategy]):
        cls._strategies[name] = strategy_cls

    @classmethod
    def get(cls, name: str) -> type[EvictionStrategy]:
        if name not in cls._strategies:
            available = ", ".join(cls._strategies.keys())
            raise KeyError(
                f"Unknown strategy '{name}'. Available: {available}"
            )
        return cls._strategies[name]

    @classmethod
    def register_decorator(cls, name: str):
        def wrapper(klass):
            cls.register(name, klass)
            return klass
        return wrapper
