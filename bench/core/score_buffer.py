"""Unified attention score accumulator across layers, heads, and steps."""

import torch


class ScoreBuffer:
    """Accumulates attention scores across all layers and heads.

    Many strategies (H2O, SnapKV, CurDKV) need to track which tokens have
    historically received high attention. This buffer provides a single shared
    accumulator that strategies query via get_topk().
    """

    def __init__(self, config: dict):
        model_cfg = config.get("model", {})
        self.n_layers = model_cfg.get("n_layers", 32)
        self.n_heads = model_cfg.get("n_heads", 32)
        self.max_seq = config.get("max_seq_length", 8192)
        # scores[layer, head, seq_len]
        self._scores: torch.Tensor | None = None
        self._step = 0

    def reset(self, device: torch.device = torch.device("cpu")):
        """Zero out the buffer for a new sequence."""
        self._scores = torch.zeros(
            self.n_layers, self.n_heads, self.max_seq,
            dtype=torch.float32, device=device,
        )
        self._step = 0

    @property
    def scores(self) -> torch.Tensor:
        assert self._scores is not None, "ScoreBuffer not initialized; call reset() first"
        return self._scores

    def update(self, layer: int, head: int, attn_weights: torch.Tensor):
        """Accumulate new attention softmax scores.

        attn_weights: (1, 1, q_len, kv_len) post-softmax.
        """
        seq_len = attn_weights.size(-1)
        # Accumulate across query positions (mean over q_len)
        scores = attn_weights[0, 0].sum(dim=0)  # (kv_len,)
        self.scores[layer, head, :seq_len] += scores

    def get_topk(
        self, k: int, layer: int | None = None,
    ) -> torch.Tensor:
        """Return indices of top-k tokens by accumulated score.

        Args:
            k: Number of tokens to keep.
            layer: If set, sum only that layer's heads; else sum all layers.

        Returns:
            Tensor of top-k token indices (sorted descending by score).
        """
        agg = (
            self.scores[layer].sum(dim=0)
            if layer is not None
            else self.scores.sum(dim=(0, 1))
        )
        effective_k = min(k, agg.size(-1))
        return agg.topk(effective_k).indices

    def advance_step(self):
        self._step += 1
