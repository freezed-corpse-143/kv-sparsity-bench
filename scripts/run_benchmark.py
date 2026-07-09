#!/usr/bin/env python3
"""CLI entry point for KV-Cache Sparsity Benchmark."""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.models import ModelLoader
from bench.runner import BenchmarkRunner
from bench.strategies import StrategyRegistry
from bench.tasks import PerplexityTask


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="KV-Cache Sparsity Benchmark"
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--strategy", default=None,
        help="Override strategy name",
    )
    parser.add_argument(
        "--task", default="perplexity",
        choices=["perplexity", "longbench", "needle"],
        help="Task to run",
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Override dataset subset size",
    )
    args = parser.parse_args()

    # Load and merge config
    config = load_config(args.config)
    if args.strategy:
        config["strategy"]["name"] = args.strategy
    if args.subset:
        for t in config.get("tasks", []):
            if isinstance(t, dict):
                t["subset"] = args.subset

    # Load model first (so we know n_layers / n_heads)
    print(f"Loading model: {config['model']['name']}")
    model, tokenizer = ModelLoader.from_config(config)
    n_layers = getattr(model.config, "n_layers",
                       getattr(model.config, "num_hidden_layers", 32))
    n_heads = getattr(model.config, "num_attention_heads",
                      getattr(model.config, "num_heads", 32))
    print(f"  Layers: {n_layers}, Heads: {n_heads}")

    # Inject architecture into config for ScoreBuffer
    config.setdefault("model", {})
    config["model"]["n_layers"] = n_layers
    config["model"]["n_heads"] = n_heads

    # Create strategy
    strategy_name = config["strategy"]["name"]
    print(f"Strategy: {strategy_name}")
    strategy_cls = StrategyRegistry.get(strategy_name)
    strategy = strategy_cls(config)

    # Create runner
    runner = BenchmarkRunner(model, tokenizer, strategy, config)

    # Run task
    task_map = {"perplexity": PerplexityTask}
    task_cls = task_map.get(args.task, PerplexityTask)
    task = task_cls(config)
    print(f"Running task: {args.task}")
    results = runner.run(task)

    # Output
    out_dir = Path(config.get("output", {}).get("dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{strategy_name}_{args.task}.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to: {out_path}")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
