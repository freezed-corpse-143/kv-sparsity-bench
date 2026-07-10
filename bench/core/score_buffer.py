"""Unified attention score accumulator across layers, heads, and steps."""

import torch


class ScoreBuffer:
    def __init__(self, config: dict):
        mc = config.get("model", {})
        self.n_layers = mc.get("n_layers", 32)
        self.n_heads = mc.get("n_heads", 32)
        self.max_seq = config.get("max_seq_length", 8192)
        self._scores: torch.Tensor | None = None

    def reset(self, device: torch.device = torch.device("cpu")):
        self._scores = torch.zeros(
            self.n_layers, self.n_heads, self.max_seq,
            dtype=torch.float32, device=device,
        )

    @property
    def scores(self) -> torch.Tensor:
        assert self._scores is not None
        return self._scores

    def update(self, layer: int, head: int, attn_weights: torch.Tensor):
        sl = attn_weights.size(-1)
        cap = self.scores.size(-1)
        if sl > cap:
            sl = cap
        s = attn_weights[0, 0, :sl].sum(dim=0)[:sl]
        n = min(s.size(0), cap)
        self.scores[layer, head, :n] += s[:n]

    def get_topk(self, k: int, layer: int | None = None) -> torch.Tensor:
        agg = (
            self.scores[layer].sum(dim=0)
            if layer is not None
            else self.scores.sum(dim=(0, 1))
        )
        ek = min(k, agg.size(-1))
        return agg.topk(ek).indices
