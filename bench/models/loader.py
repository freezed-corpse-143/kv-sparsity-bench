"""Load HuggingFace models for benchmark evaluation."""

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from bench.core.hook_manager import _get_layers


class SimpleTokenizer:
    """Character-level tokenizer for tiny test models. No HF hub access needed."""

    def __init__(self):
        self.vocab = {chr(i): i - 32 for i in range(32, 127)}
        self.vocab["<|endoftext|>"] = len(self.vocab)
        self.inv = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = self.vocab["<|endoftext|>"]
        self.eos_token_id = self.vocab["<|endoftext|>"]
        self.bos_token_id = self.vocab["<|endoftext|>"]

    def encode(self, text: str) -> list[int]:
        eos = self.vocab["<|endoftext|>"]
        return [self.vocab.get(c, eos) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.inv.get(i, "") for i in ids)

    def __call__(self, text: str, **kwargs):
        import torch
        max_length = kwargs.get("max_length", 2048)
        truncation = kwargs.get("truncation", False)
        return_tr = kwargs.get("return_tensors", None)
        ids = self.encode(text)
        if truncation and len(ids) > max_length:
            ids = ids[:max_length]
        result = {"input_ids": torch.tensor([ids])}
        if return_tr == "pt":
            result["input_ids"] = torch.tensor([ids])
        return result


class ModelLoader:
    """Loads and configures a HuggingFace model for KV-cache benchmarks."""

    @staticmethod
    def from_config(config: dict) -> tuple:
        model_cfg = config.get("model", {})
        model_name = model_cfg.get("name", "Qwen/Qwen2.5-0.5B")
        dtype_name = model_cfg.get("dtype", "bfloat16")
        device = model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype_name, torch.bfloat16)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            attn_implementation="eager",
            trust_remote_code=True,
        )
        model = model.to(device)
        model.eval()
        model.config.output_attentions = True

        # Store n_layers / n_heads for framework
        n_layers = ModelLoader._get_n_layers(model)
        model.config.n_layers = n_layers
        n_heads = getattr(
            model.config, "num_attention_heads",
            getattr(model.config, "num_heads", 32),
        )
        model.config.n_heads = n_heads

        # Tokenizer — try HF hub, fallback to simple char tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            print("  [tokenizer fallback: using char-level tokenizer]")
            tokenizer = SimpleTokenizer()

        return model, tokenizer

    @staticmethod
    def _get_n_layers(model) -> int:
        cfg = model.config
        for attr in ("num_hidden_layers", "num_layers", "n_layer"):
            v = getattr(cfg, attr, None)
            if v is not None:
                return v
        return len(_get_layers(model))
