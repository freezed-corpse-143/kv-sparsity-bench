"""Perplexity evaluation with and without KV-cache eviction."""

import math
import os
from pathlib import Path

import torch

from .base import Task

_SYNTHETIC_CORPUS = """
The quick brown fox jumps over the lazy dog. Machine learning is transforming
how we process natural language. KV cache compression reduces memory usage in
transformer inference. Attention mechanisms allow models to focus on relevant
input parts. As sequence lengths grow the KV cache becomes a memory bottleneck.
Various compression techniques have been proposed including H2O SnapKV and RocketKV.
""".strip()


class PerplexityTask(Task):
    """Compute perplexity via sliding window, optionally with KV eviction."""

    def __init__(self, config: dict):
        super().__init__(config)
        task_cfg = config.get("tasks", [])
        self.cfg = {}
        for t in (task_cfg if isinstance(task_cfg, list) else [task_cfg]):
            if isinstance(t, dict) and t.get("name") == "perplexity":
                self.cfg = t
                break
        self.max_length = config.get("max_seq_length", 2048)
        self.stride = self.cfg.get("stride", 512)
        self.corpus_file = self.cfg.get("corpus_file", None)
        sn = config.get("strategy", {}).get("name", "baseline")
        self.use_eviction = sn not in ("baseline", None)

    def run(self, runner) -> dict:
        runner.score_buffer.reset(runner.model.device)
        return (self._run_with_eviction(runner) if self.use_eviction
                else self._run_standard(runner))

    def _run_standard(self, runner) -> dict:
        model, device = runner.model, runner.model.device
        ids = self._tokenize(runner.tokenizer, device)
        sl = ids.size(1)
        if sl < 64:
            return {"perplexity": float("inf"), "n_tokens": 0}
        nll, nt = 0.0, 0
        for s in range(0, sl, self.stride):
            e = min(s + self.max_length, sl)
            if e - s < 64:
                break
            c = ids[:, s:e]
            lb = c.clone()
            lb[:, :-1] = -100
            with torch.no_grad():
                nll += model(c, labels=lb).loss.item() * (e - s)
                nt += e - s
            if e == sl:
                break
        ppl = math.exp(nll / nt) if nt > 0 else float("inf")
        return {"perplexity": round(ppl, 4), "n_tokens": nt, "seq_len": sl, "mode": "standard"}

    def _run_with_eviction(self, runner) -> dict:
        model, device = runner.model, runner.model.device
        ids = self._tokenize(runner.tokenizer, device)
        sl = ids.size(1)
        if sl < 64:
            return {"perplexity": float("inf"), "n_tokens": 0}
        runner.strategy.reset()
        cache = None
        nll, nt, steps = 0.0, 0, 0
        kv_sizes = []

        for s in range(0, sl, self.stride):
            e = min(s + self.max_length, sl)
            if e - s < 64:
                break
            with torch.no_grad():
                out = model(ids[:, s:e], past_key_values=cache,
                            use_cache=True, output_attentions=True)
                loss = out.loss
                if loss is None:
                    slb = ids[:, s+1:e+1]
                    loss = torch.nn.CrossEntropyLoss()(
                        out.logits[:, :-1].reshape(-1, out.logits.size(-1)),
                        slb.reshape(-1))
            nll += loss.item() * (e - s)
            nt += e - s
            steps += 1
            cache = out.past_key_values
            if cache is not None:
                for li in range(len(cache)):
                    k = cache.layers[li].keys
                    v = cache.layers[li].values
                    sz = k.size(2)
                    bgt = runner.strategy.budget_controller.get_budget(steps, li)
                    if sz > bgt:
                        sc = runner.score_buffer.scores[li].sum(dim=0)
                        ps = [(i, sc[i].item() if i < sc.size(0) else 0.0) for i in range(sz)]
                        keep = [p[0] for p in sorted(ps, key=lambda x: -x[1])[:bgt]]
                        if keep:
                            kt = torch.tensor(keep, device=k.device, dtype=torch.long)
                            cache.layers[li].keys = k[:, :, kt, :]
                            cache.layers[li].values = v[:, :, kt, :]
                    kv_sizes.append(cache.layers[li].keys.size(2))
            if e == sl:
                break

        avg = sum(kv_sizes) / len(kv_sizes) if kv_sizes else 0
        ppl = math.exp(nll / nt) if nt > 0 else float("inf")
        return {"perplexity": round(ppl, 4), "n_tokens": nt, "seq_len": sl,
                "mode": "eviction", "avg_kv_size": round(avg, 1)}

    def _tokenize(self, tokenizer, device):
        text = (Path(self.corpus_file).read_text(encoding="utf-8")
                if self.corpus_file and os.path.exists(self.corpus_file)
                else _SYNTHETIC_CORPUS)
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=self.max_length * 4)
        return (enc["input_ids"] if isinstance(enc, dict) else enc.input_ids).to(device)
