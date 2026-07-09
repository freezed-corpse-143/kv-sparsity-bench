"""Inject attention-score capture hooks into HuggingFace model layers."""

import torch
from transformers import PreTrainedModel

from .score_buffer import ScoreBuffer


class HookManager:
    """Captures attention softmax scores from model decoder layers.

    Uses forward hooks on each decoder layer's attention module to record
    per-head softmax scores after each forward pass.
    """

    def __init__(self, score_buffer: ScoreBuffer):
        self.score_buffer = score_buffer
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._captured_attentions: list[list[torch.Tensor]] = []

    def install(self, model: PreTrainedModel):
        """Register hooks on each decoder layer's self-attention.

        Requires model to use attn_implementation='eager' and
        output_attentions=True so that the forward returns softmax weights.
        """
        self._captured_attentions = []
        layers = _get_layers(model)

        for layer_idx, layer in enumerate(layers):
            attn = _get_attention_module(layer)
            if attn is None:
                continue

            # Register forward hook that runs AFTER attention forward
            handle = attn.register_forward_hook(
                self._make_attn_hook(layer_idx)
            )
            self._handles.append(handle)

    def _make_attn_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            # Output from HuggingFace eager attention is typically
            # (attn_output, present_kv, attn_weights) when output_attentions=True
            # attn_weights is (1, n_heads, q_len, kv_len)
            if isinstance(output, tuple) and len(output) >= 3:
                attn_weights = output[2]
                if attn_weights is not None and isinstance(attn_weights, torch.Tensor):
                    # attn_weights: (batch, n_heads, q_len, kv_len)
                    # Average over query positions to get per-token scores
                    n_heads = min(4, attn_weights.size(1))  # sample heads
                    for h in range(n_heads):
                        self.score_buffer.update(
                            layer_idx, h,
                            attn_weights[0:1, h : h + 1].detach().cpu(),
                        )
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @property
    def captured(self) -> list[list[torch.Tensor]]:
        return self._captured_attentions


def _get_layers(model: PreTrainedModel) -> list:
    """Get decoder layers regardless of model architecture."""
    for attr in ("model", "transformer", "encoder", "decoder"):
        sub = getattr(model, attr, None)
        if sub is not None:
            for layers_attr in ("layers", "h", "block", "blocks"):
                layers = getattr(sub, layers_attr, None)
                if layers is not None:
                    return layers
    for attr in ("layers", "h", "block", "blocks"):
        layers = getattr(model, attr, None)
        if layers is not None:
            return layers
    raise ValueError("Could not find decoder layers in model")


def _get_attention_module(layer) -> object | None:
    """Get the self-attention module from a decoder layer."""
    for attr in ("self_attn", "attention", "attn"):
        attn = getattr(layer, attr, None)
        if attn is not None:
            return attn
    return None
