"""Perplexity evaluation using local text corpus."""

import math
import os
from pathlib import Path

import torch

from .base import Task


# Synthetic test corpus (1000 tokens of varied English text)
_SYNTHETIC_CORPUS = """
The quick brown fox jumps over the lazy dog. Machine learning is transforming
how we process natural language. KV cache compression reduces memory usage in
transformer inference. Attention mechanisms allow models to focus on relevant
parts of the input sequence. Large language models have demonstrated remarkable
capabilities in text generation, translation, and summarization.

The key innovation of transformers is the self-attention mechanism, which computes
a weighted sum of all positions in the input sequence. The weights are determined
by the compatibility between queries and keys. During autoregressive generation,
the model produces one token at a time, and the key-value cache stores previously
computed representations to avoid redundant computation.

As sequence lengths grow, the KV cache becomes a memory bottleneck. Each layer
stores keys and values for every token, consuming O(n * d * h) memory where n
is the sequence length, d is the dimension, and h is the number of heads. For
long sequences of 100K tokens or more, this can require tens of gigabytes.

Various compression techniques have been proposed. Token eviction methods like
H2O and SnapKV discard unimportant tokens based on attention scores. Structured
sparsity methods like MUSTAFAR prune less important entries. CUR decomposition
approximates the KV matrix with a low-rank factorization.

H2O (Heavy-Hitter Oracle) accumulates attention scores across generation steps
and retains only the top-k tokens by cumulative score. SnapKV goes further by
clustering prompt tokens and keeping cluster centroids plus a recent window.
RocketKV from NVIDIA uses a two-stage design: coarse page-level eviction followed
by fine-grained top-k selection.

CurDKV applies CUR matrix decomposition to the value matrix, using leverage scores
to guide eviction decisions. This achieves higher accuracy than score-based methods
on long-context tasks. MUSTAFAR promotes unstructured sparsity through per-token
magnitude pruning with a bitmap storage format.

Evaluating these methods requires measuring perplexity on held-out text, accuracy on
downstream tasks like question answering and summarization, and retrieval precision
on needle-in-haystack tests. The compression ratio measures how many KV entries
are retained relative to the full cache.

This benchmark framework provides a unified interface for implementing and comparing
these strategies. Each strategy subclasses EvictionStrategy and implements the
decide_eviction method, which returns the indices of tokens to retain at each layer.
""".strip()


class PerplexityTask(Task):
    """Compute perplexity on a text corpus using sliding-window evaluation.

    Uses a local synthetic corpus by default, or loads from a text file.
    """

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

    def run(self, runner) -> dict:
        model = runner.model
        tokenizer = runner.tokenizer
        device = model.device
        # Load text
        if self.corpus_file and os.path.exists(self.corpus_file):
            text = Path(self.corpus_file).read_text(encoding="utf-8")
        else:
            text = _SYNTHETIC_CORPUS
        encodings = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.max_length * 4,
        )
        # Handle both dict return (SimpleTokenizer) and object return (HF tokenizer)
        if isinstance(encodings, dict):
            input_ids = encodings["input_ids"].to(device)
        else:
            input_ids = encodings.input_ids.to(device)
        seq_len = input_ids.size(1)

        if seq_len < 64:
            return {"perplexity": float("inf"), "n_tokens": 0}

        nll_sum = 0.0
        n_tokens = 0

        for start in range(0, seq_len, self.stride):
            end = min(start + self.max_length, seq_len)
            if end - start < 64:
                break

            chunk_ids = input_ids[:, start:end]
            labels = chunk_ids.clone()
            labels[:, :-1] = -100

            with torch.no_grad():
                outputs = model(chunk_ids, labels=labels)
                nll = outputs.loss.item() * (end - start)
                nll_sum += nll
                n_tokens += (end - start)

            if end == seq_len:
                break

        ppl = math.exp(nll_sum / n_tokens) if n_tokens > 0 else float("inf")

        return {
            "perplexity": round(ppl, 4),
            "nll_sum": round(nll_sum, 2),
            "n_tokens": n_tokens,
            "seq_len": seq_len,
        }
