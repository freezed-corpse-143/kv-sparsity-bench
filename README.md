# KV-Cache Sparsity Benchmark

> LLM 推理加速中的 KV-Cache 稀疏化算法评测框架

[![Loop Ready](https://img.shields.io/badge/loop--ready-100%25-success)](https://github.com/cobusgreyling/loop-engineering)

## 概述

大语言模型（LLM）的自回归生成依赖 KV cache 来避免重复计算。但随着序列增长（100K+ tokens），KV cache 占用数十 GB 显存，成为推理吞吐的核心瓶颈。

本项目构建统一的评测框架，覆盖 2024-2026 年 8 篇核心前沿论文的稀疏化/压缩算法。

## 快速开始

```bash
# 安装
uv venv
uv pip install -r requirements.txt
uv run python -m pip install torch --index-url https://download.pytorch.org/whl/cu128

# 运行 baseline 评测
uv run python scripts/run_benchmark.py --strategy baseline --task perplexity

# 运行 H2O 评测
uv run python scripts/run_benchmark.py --strategy h2o --task perplexity
```

## 架构

```
kv-sparsity-bench/
├── bench/                 # 核心库
│   ├── core/              # 共享基础设施 (ScoreBuffer, HookManager, ...)
│   ├── strategies/        # 稀疏策略（可插拔）
│   ├── models/            # 模型加载
│   ├── tasks/             # 评测任务 (PPL / LongBench / Needle)
│   ├── metrics/           # 指标收集 & 可视化
│   └── runner.py          # 统一运行入口
├── scripts/               # CLI
├── configs/               # YAML 配置
├── papers/                # 论文 PDF
└── results/               # 评测结果
```

## 已实现策略

| 策略 | 核心 | 行数 | 压缩比 (10% budget) |
|---|---|---|---|
| Baseline | 参考线，无压缩 | — | 1× |
| H2O | Attention score 排序 eviction | ~30 | **5×** |
| SnapKV | Prompt 聚类 + 窗口保留 | ~60 | 按 cluster 压缩 |

## 实验结果

| 策略 | Budget | KV 大小 | 压缩比 | 吞吐 (tok/s) |
|---|---|---|---|---|
| Baseline | — | 1033 | 1× | 707 |
| H2O | 30% | 614 | 1.7× | 684 |
| H2O | 20% | 409 | 2.5× | 707 |
| H2O | 10% | 204 | **5×** | 697 |

## 评测任务

- **Perplexity**: 滑动窗口 PPL 评测
- **LongBench** (开发中): 多领域长文本下游任务
- **Needle-in-Haystack** (开发中): 长上下文检索精度

## 配置

```yaml
# configs/default.yaml
model:
  name: "models/tiny-test-v2"
  dtype: float32
  device: "cpu"

strategy:
  name: h2o
  budget_ratio: 0.3
  min_budget: 64
```

## Loop 运维

本项目使用 [loop-engineering](https://github.com/cobusgreyling/loop-engineering) 自动化运维：

```bash
# 每日进度追踪
/loop 1d $loop-triage

# 审计
npx @cobusgreyling/loop-audit .
```

## 论文收录

| 论文 | Venue |
|---|---|
| H2O: Heavy-Hitter Oracle | ACL 2024 |
| SnapKV: LLM Knows What You Need | ACL 2024 |
| RocketKV: Two-Stage Compression | ICML 2025, NVIDIA |
| AttentionPredictor: Temporal Patterns | NeurIPS 2025 |
| SmallKV: Small Model Assisted | NeurIPS 2025 |
| CurDKV: CUR Decomposition | NeurIPS 2025 |
| MUSTAFAR: Unstructured Sparsity | NeurIPS 2025 |
| Mnemosyne: Cache Hit Order Fitting | AAAI 2026 |

## 文档

- [实验报告](docs/report.md)
- [架构设计](docs/plans/2026-07-09-kv-sparsity-bench-design.md)
- [技术路线图](docs/roadmap.md)
