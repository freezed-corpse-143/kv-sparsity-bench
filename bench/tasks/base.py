"""Abstract task interface for benchmark evaluations."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..runner import BenchmarkRunner


class Task(ABC):
    """A single evaluation task (e.g., perplexity, longbench, needle)."""

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def run(self, runner: "BenchmarkRunner") -> dict:
        """Execute the task and return metrics dict."""
        ...
