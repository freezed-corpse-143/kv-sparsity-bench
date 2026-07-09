"""Determine how many KV entries to retain at each step."""


class BudgetController:
    """Controls per-layer and per-step token retention budgets.

    The budget can be:
    - Fixed ratio: keep budget_ratio * max_cache_capacity tokens.
    - Dynamic: grows with sequence length up to a cap.
    """

    def __init__(self, config: dict):
        strategy_cfg = config.get("strategy", {})
        self.budget_ratio = strategy_cfg.get("budget_ratio", 0.2)
        self.max_budget = config.get("max_cache_capacity", 2048)
        self.min_budget = strategy_cfg.get("min_budget", 64)

    def get_budget(self, step: int, layer: int | None = None) -> int:
        """Number of tokens to retain at this step.

        Args:
            step: Current generation step (0 = first new token).
            layer: Optional per-layer budget (unused in uniform budgeting).

        Returns:
            Integer budget >= min_budget.
        """
        base = max(self.min_budget, int(self.max_budget * self.budget_ratio))
        return min(base, step + 1)

    def total_cache_capacity(self) -> int:
        """Maximum number of KV entries allowed across all layers."""
        return max(self.min_budget, int(self.max_budget * self.budget_ratio))
