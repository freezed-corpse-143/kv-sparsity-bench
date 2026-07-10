"""Needle-in-Haystack: long-context retrieval accuracy evaluation.

Tests whether the model can retrieve a specific piece of information
("needle") from a large body of distractor text ("haystack") at
various depths and context lengths.
"""

import random
import math
from typing import Any

import torch

from .base import Task


class NeedleInHaystackTask(Task):
    """Measure retrieval accuracy across context depths and lengths.

    Inserts a unique fact (needle) at different positions (depths) within
    a long distractor text (haystack), then queries the model about it.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        task_cfg = config.get("tasks", [])
        self.cfg = {}
        for t in (task_cfg if isinstance(task_cfg, list) else [task_cfg]):
            if isinstance(t, dict) and t.get("name") in ("needle", "needle_in_haystack"):
                self.cfg = t
                break

        self.max_length = config.get("max_seq_length", 4096)
        self.depths = self.cfg.get("depths", [0.1, 0.3, 0.5, 0.7, 0.9])
        self.lengths = self.cfg.get("lengths", [2048, 4096])
        self.n_needle_types = self.cfg.get("n_needle_types", 3)
        self.seed = self.cfg.get("seed", 42)

    def run(self, runner) -> dict:
        model = runner.model
        tokenizer = runner.tokenizer
        strategy = runner.strategy
        device = model.device
        runner.score_buffer.reset(device)
        strategy.reset()

        rng = random.Random(self.seed)
        results = []

        for ctx_len in self.lengths:
            if ctx_len > self.max_length:
                continue
            for depth in self.depths:
                for _ in range(self.n_needle_types):
                    acc = self._run_single(
                        runner, ctx_len, depth, rng)
                    results.append({
                        "context_length": ctx_len,
                        "depth": depth,
                        "accuracy": acc,
                    })

        # Aggregate
        avg_acc = sum(r["accuracy"] for r in results) / max(len(results), 1)
        return {
            "accuracy": round(avg_acc, 4),
            "n_samples": len(results),
            "details": results,
        }

    def _run_single(
        self, runner, ctx_len: int, depth: float, rng: random.Random,
    ) -> float:
        """Run one needle-in-haystack trial. Returns accuracy (0 or 1)."""
        model = runner.model
        tokenizer = runner.tokenizer
        strategy = runner.strategy
        device = model.device
        tokenizer.pad_token = getattr(tokenizer, "pad_token",
                                       getattr(tokenizer, "eos_token", None))

        # Create needle fact
        needle_num = rng.randint(10000, 99999)
        needle_city = rng.choice(["Paris", "Tokyo", "Berlin", "London", "Rome"])
        needle = (
            f"The secret code is {needle_num}. "
            f"Remember: {needle_city} is the target city."
        )

        # Create haystack (repeated filler text)
        haystack_sentence = (
            "The quick brown fox jumps over the lazy dog near the riverbank. "
            "Several large oak trees provide shade on hot summer afternoons. "
        )
        n_repeat = math.ceil(ctx_len / len(haystack_sentence))
        haystack = (haystack_sentence * n_repeat)[:ctx_len]

        # Insert needle at specified depth
        insert_pos = int(len(haystack) * depth)
        context = haystack[:insert_pos] + needle + haystack[insert_pos:]

        # Truncate
        context = context[:ctx_len]

        # Query
        query = (
            f"\n\nBased on the text above, what is the secret code "
            f"and which is the target city?"
        )
        prompt = context + query

        enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=self.max_length)
        input_ids = (enc["input_ids"] if isinstance(enc, dict)
                     else enc.input_ids).to(device)

        if input_ids.size(1) < 64:
            return 0.0

        # Generate with eviction
        strategy.reset()
        runner.score_buffer.reset(device)
        max_new = 20
        gen_len = 0
        cache = None
        generated = input_ids.clone()

        for step in range(max_new):
            last = generated[:, -1:]
            with torch.no_grad():
                out = model(last, past_key_values=cache,
                            use_cache=True, output_attentions=True)
            tid = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, tid], dim=1)
            gen_len += 1
            cache = out.past_key_values

            # Check for EOS
            if tid.item() == getattr(tokenizer, "eos_token_id", -1):
                break

            # Evict
            if cache is not None:
                for li in range(len(cache)):
                    k = cache.layers[li].keys
                    v = cache.layers[li].values
                    sz = k.size(2)
                    bgt = strategy.budget_controller.get_budget(
                        input_ids.size(1) + step, li)
                    if sz > bgt:
                        sc = runner.score_buffer.scores[li].sum(dim=0)
                        ps = [(i, sc[i].item() if i < sc.size(0) else 0.0)
                              for i in range(sz)]
                        keep = [p[0] for p in sorted(ps, key=lambda x: -x[1])[:bgt]]
                        if keep:
                            kt = torch.tensor(keep, device=k.device, dtype=torch.long)
                            cache.layers[li].keys = k[:, :, kt, :]
                            cache.layers[li].values = v[:, :, kt, :]

        # Check if answer is correct
        output_text = tokenizer.decode(
            generated[0, input_ids.size(1):], skip_special_tokens=True)
        correct = str(needle_num) in output_text and needle_city in output_text
        return 1.0 if correct else 0.0
