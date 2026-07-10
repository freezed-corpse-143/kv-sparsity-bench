"""Inject attention-score capture hooks into HuggingFace model layers."""

import torch
from transformers import PreTrainedModel

from .score_buffer import ScoreBuffer


class HookManager:
    def __init__(self, score_buffer: ScoreBuffer):
        self.score_buffer = score_buffer
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def install(self, model: PreTrainedModel):
        layers = _get_layers(model)
        for layer_idx, layer in enumerate(layers):
            attn = _get_attention_module(layer)
            if attn is None:
                continue
            handle = attn.register_forward_hook(self._make_hook(layer_idx))
            self._handles.append(handle)

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            # GPT2: (hidden, attn) | LLaMA/Qwen: (hidden, kv, attn)
            aw = None
            if isinstance(output, tuple):
                if len(output) >= 3:
                    aw = output[2]
                elif len(output) == 2:
                    aw = output[1]
            if aw is not None and isinstance(aw, torch.Tensor):
                nh = min(4, aw.size(1))
                for h in range(nh):
                    self.score_buffer.update(layer_idx, h, aw[0:1, h:h+1].detach())
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _get_layers(model: PreTrainedModel) -> list:
    for attr in ("model", "transformer"):
        sub = getattr(model, attr, None)
        if sub is not None:
            for la in ("layers", "h", "block"):
                layers = getattr(sub, la, None)
                if layers is not None:
                    return layers
    for la in ("layers", "h", "block"):
        layers = getattr(model, la, None)
        if layers is not None:
            return layers
    raise ValueError("Could not find decoder layers")


def _get_attention_module(layer) -> object | None:
    for attr in ("self_attn", "attention", "attn"):
        attn = getattr(layer, attr, None)
        if attn is not None:
            return attn
    return None
