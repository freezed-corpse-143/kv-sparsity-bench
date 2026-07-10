"""Inject attention-score capture hooks into HuggingFace model layers."""

import torch
from transformers import PreTrainedModel

from .score_buffer import ScoreBuffer


class HookManager:
    """Captures attention softmax scores via forward hooks on attention modules."""

    def __init__(self, score_buffer: ScoreBuffer):
        self.score_buffer = score_buffer
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._captured_attentions: list[list[torch.Tensor]] = []

    def install(self, model: PreTrainedModel):
        self._captured_attentions = []
        layers = _get_layers(model)

        for layer_idx, layer in enumerate(layers):
            attn = _get_attention_module(layer)
            if attn is None:
                continue
            handle = attn.register_forward_hook(
                self._make_attn_hook(layer_idx)
            )
            self._handles.append(handle)

    def _make_attn_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            # GPT2: (hidden_states, attentions)
            # LLaMA/Qwen: (hidden_states, present_kv, attentions)
            attn_weights = None
            if isinstance(output, tuple):
                if len(output) >= 3:
                    attn_weights = output[2]
                elif len(output) == 2:
                    attn_weights = output[1]

            if attn_weights is not None and isinstance(attn_weights, torch.Tensor):
                n_heads = min(4, attn_weights.size(1))
                for h in range(n_heads):
                    self.score_buffer.update(
                        layer_idx, h,
                        attn_weights[0:1, h: h + 1].detach().cpu(),
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
    for attr in ("self_attn", "attention", "attn"):
        attn = getattr(layer, attr, None)
        if attn is not None:
            return attn
    return None
