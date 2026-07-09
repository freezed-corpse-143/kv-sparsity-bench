"""Track token liveness and compress KV tensors."""

import torch


class CacheManager:
    """Tracks which KV entries are alive and compresses tensors after eviction.

    Each layer has an alive_mask[seq_len] — True means the token is retained.
    compress() applies the mask and returns the sliced (key, value).
    """

    def __init__(self, config: dict):
        self.n_layers = config.get("model", {}).get("n_layers", 32)
        self._alive_mask: list[torch.Tensor | None] = [None] * self.n_layers

    def reset(self):
        self._alive_mask = [None] * self.n_layers

    def compress(
        self,
        layer: int,
        keep_indices: list[int] | torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice key/value to keep only the given token indices.

        Args:
            layer: Layer index.
            keep_indices: Token positions to retain.
            key: (1, n_heads, seq_len, head_dim)
            value: (1, n_heads, seq_len, head_dim)

        Returns:
            (compressed_key, compressed_value) with seq_len = len(keep_indices).
        """
        device = key.device
        n_tokens = key.size(2)
        mask = torch.zeros(n_tokens, dtype=torch.bool, device=device)
        mask[keep_indices] = True
        self._alive_mask[layer] = mask
        return key[:, :, keep_indices, :], value[:, :, keep_indices, :]
