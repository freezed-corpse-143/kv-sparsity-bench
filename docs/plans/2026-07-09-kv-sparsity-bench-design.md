# KV-Cache Sparsity Benchmark — 架构设计

> 更新：2026-07-09

## 1. 项目目标

构建可复现的 KV-cache 稀疏化/压缩算法评测框架，覆盖 Token Eviction + 结构剪枝路线（不含量化压缩）。支持：

- **精度 vs 压缩率曲线** — Perplexity / LongBench F1 / Needle 检索准确率
- **端到端推理加速** — latency / throughput profiling
- **多模型、多策略、多任务的统一评测**

## 2. 整体结构

```
kv-sparsity-bench/
├── bench/                          # 核心库
│   ├── __init__.py
│   │
│   ├── strategies/                 # 稀疏策略（可插拔）
│   │   ├── __init__.py
│   │   ├── base.py                 # EvictionStrategy 基类
│   │   ├── h2o.py
│   │   ├── snapkv.py
│   │   ├── rocketkv.py
│   │   ├── curdkv.py
│   │   ├── mustafar.py
│   │   └── mnemosyne.py
│   │
│   ├── core/                       # 共享基础设施
│   │   ├── __init__.py
│   │   ├── score_buffer.py         # ScoreBuffer
│   │   ├── cache_manager.py        # CacheManager
│   │   ├── budget_controller.py    # BudgetController
│   │   └── hook_manager.py         # HookManager
│   │
│   ├── models/                     # 模型加载
│   │   ├── __init__.py
│   │   └── loader.py               # ModelLoader
│   │
│   ├── tasks/                      # 评测任务
│   │   ├── __init__.py
│   │   ├── base.py                 # Task 基类
│   │   ├── perplexity.py           # PPL @ PG19
│   │   ├── longbench.py            # LongBench 下游任务
│   │   └── needle.py               # Needle-in-Haystack
│   │
│   ├── metrics/                    # 指标收集 & 报告
│   │   ├── __init__.py
│   │   ├── collector.py            # MetricsCollector
│   │   └── reporter.py             # ReportBuilder + 出图
│   │
│   └── runner.py                   # BenchmarkRunner 统一入口
│
├── scripts/                        # CLI
│   ├── run_benchmark.py            # 启动评测
│   └── plot_results.py             # 单独出图
│
├── configs/                        # YAML 配置
│   ├── default.yaml
│   └── strategies/
│       ├── h2o.yaml
│       ├── snapkv.yaml
│       └── ...
│
├── data/                           # 数据集缓存（gitignored）
├── results/                        # 结果输出（gitignored）
│
├── papers/                         # PDF 论文
├── docs/
│   ├── roadmap.md
│   └── plans/
│       └── 2026-07-09-kv-sparsity-bench-design.md
│
├── pyproject.toml                  # 项目元数据 + 依赖
├── requirements.txt
└── README.md
```

## 3. 核心类设计

### 3.1 EvictionStrategy（基类）

```python
class EvictionStrategy(ABC):
    """所有稀疏策略的基类。
    
    子类只需实现 decide_eviction() 方法——仅决定"保留哪些 tokens"。
    共享基础设施由 core/ 中的组件统一管理。
    """

    def __init__(self, config: dict):
        self.config = config
        self.score_buffer = ScoreBuffer(config)
        self.cache_manager = CacheManager(config)
        self.budget_controller = BudgetController(config)

    @abstractmethod
    def decide_eviction(
        self,
        layer_idx: int,
        step: int,
    ) -> list[int]:
        """返回该层这一步要保留的 token 索引列表。"""
        ...

    def on_attn_step(
        self,
        layer_idx: int,
        head_idx: int,
        attn_scores: torch.Tensor,
    ):
        """默认 hook：将 attention scores 累入 ScoreBuffer。"""
        self.score_buffer.update(layer_idx, head_idx, attn_scores)

    def on_generation_step(
        self,
        layer_idx: int,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """由 HookManager 在每步生成后调用。
        
        返回该层 evict 后的 (key, value)。
        """
        keep_indices = self.decide_eviction(layer_idx, step)
        return self.cache_manager.compress(layer_idx, keep_indices)
```

### 3.2 ScoreBuffer

```python
class ScoreBuffer:
    """累积各层、各 head 的 attention scores。"""

    def __init__(self, config: dict):
        self.n_layers = config["model"]["n_layers"]
        self.n_heads = config["model"]["n_heads"]
        self.max_seq = config.get("max_seq_length", 8192)
        # scores[layer, head, token_idx] — 浮点累加
        self.scores = torch.zeros(
            self.n_layers, self.n_heads, self.max_seq,
            dtype=torch.float32,
        )

    def update(self, layer: int, head: int, scores: torch.Tensor):
        """累加新步的 attention softmax scores。"""
        seq_len = scores.size(-1)
        self.scores[layer, head, :seq_len] += scores

    def get_topk(self, k: int, layer: int | None = None) -> torch.Tensor:
        """返回全局或特定层的 top-k token 索引。"""
        if layer is not None:
            agg = self.scores[layer].sum(dim=0)  # 跨 head 求和
        else:
            agg = self.scores.sum(dim=(0, 1))    # 跨层+head
        return agg.topk(k).indices
```

### 3.3 CacheManager

```python
class CacheManager:
    """追踪每个 token 的存活状态，执行 KV 压缩与对齐。"""

    def __init__(self, config: dict):
        self.n_layers = config["model"]["n_layers"]
        self.alive_masks: list[torch.Tensor | None] = [None] * self.n_layers
        self.keep_counts: list[int] = [0] * self.n_layers

    def compress(
        self, layer: int, keep_indices: list[int],
        key: torch.Tensor, value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """按 keep_indices 压缩该层 KV，返回压缩后的 (K, V)。"""
        mask = torch.zeros(key.size(2), dtype=torch.bool, device=key.device)
        mask[keep_indices] = True
        self.alive_masks[layer] = mask
        new_k = key[:, :, keep_indices, :]
        new_v = value[:, :, keep_indices, :]
        self.keep_counts[layer] = len(keep_indices)
        return new_k, new_v
```

### 3.4 BudgetController

```python
class BudgetController:
    """控制每层/每步保留多少 tokens。"""

    def __init__(self, config: dict):
        ratio = config["strategy"].get("budget_ratio", 0.2)
        self.budget_ratio = ratio
        self.max_budget = config.get("max_cache_capacity", 2048)

    def get_budget(self, step: int, layer: int | None = None) -> int:
        """当前步应该保留的 token 数。"""
        base = int(self.max_budget * self.budget_ratio)
        # 可被子类覆写以实现动态预算
        return min(base, step + 1)  # 不超过已生成的 token 数
```

### 3.5 HookManager

```python
class HookManager:
    """注入到 HuggingFace 模型的 attention forward 中。

    在每层 attention 计算完 KV 和 softmax scores 后，调用 strategy.on_attn_step()
    收集分数；在生成每步末尾调用 strategy.on_generation_step() 执行 eviction。
    """

    def install(self, model: PreTrainedModel, strategy: EvictionStrategy):
        """替换各层的 attention forward。"""
        ...

    def remove(self):
        """恢复原始 forward。"""
        ...
```

## 4. 各策略速览

| 策略 | 核心逻辑 | 行数估计 |
|---|---|---|
| **H2O** | 累加 scores → `get_topk(budget)` | ~20 |
| **SnapKV** | prompt 聚类 + window buffer → 替换 budget | ~40 |
| **RocketKV-light** | 分页 coarse evict + fine top-k 稀疏 attn | ~60 |
| **CurDKV** | CUR 分解 V → leverage score → evict | ~80 |
| **MUSTAFAR** | KV 幅度剪枝 + bitmap mask | ~50 |
| **Mnemosyne** | 重排序 + 多级 cache（hot/warm/cold） | ~60 |

## 5. 评测任务

| 任务 | 数据集 | 核心指标 | 记录方式 |
|---|---|---|---|
| Perplexity | PG19 | PPL @ 各压缩比 | 曲线 (compression vs PPL) |
| LongBench | LongBench (6 类) | F1 / ROUGE-L / Accuracy | 按任务类别分组柱状图 |
| Needle-in-Haystack | 合成 | 检索准确率 | 热力图 (depth × context_len) |

每个任务接受 `List[EvictionStrategy]`，输出 `List[Dict[str, float]]`，由 ReportBuilder 汇总成报告。

## 6. 执行流

```
config.yaml
    │
    ▼
ModelLoader.from_config(config) → HuggingFace model (eval mode)
    │
    ▼
StrategyRegistry.get(config.strategy.name) → EvictionStrategy
    │
    ▼
HookManager.install(model, strategy)
    │
    ▼
for task in config.tasks:
    │
    ├─ Task.load_data(data/...) → dataset
    ├─ for sample in dataset:
    │     Runner.prefill(sample.prompt)
    │     Runner.generate(sample.expected_output_len)
    │     MetricsCollector.record(ppl / f1 / acc / compression_ratio / latency)
    │
    └─ ReportBuilder.build(results) → markdown + figures
    │
    ▼
results/{model}_{strategy}_{timestamp}/
    ├── report.md
    └── figures/
```

## 7. 配置文件格式

```yaml
# configs/default.yaml
model:
  name: "meta-llama/Llama-3.1-8B"
  dtype: bfloat16
  device: cuda:0
  n_layers: 32
  n_heads: 32

strategy:
  name: h2o
  budget_ratio: 0.2
  max_cache_capacity: 2048

tasks:
  - name: perplexity
    dataset: "pg19"
    max_length: 8192
    stride: 512
    subset: 100   # 采样 100 个样本加速开发
  - name: needle
    depths: [0.1, 0.3, 0.5, 0.7, 0.9]
    lengths: [2048, 4096, 8192]
  - name: longbench
    categories: ["qa", "summarization", "classification"]

output:
  dir: results/
  save_checkpoints: true
```

## 8. 依赖（初步）

```
transformers>=4.40.0
torch>=2.2.0
datasets>=2.18.0
numpy
matplotlib
seaborn
pyyaml
tqdm
```

## 9. 渐进式实现顺序

| 阶段 | 内容 | 交付物 |
|---|---|---|
| **P0** 骨架 | core/ + strategy 基类 + HookManager + baseline | `--strategy baseline` 能跑 PPL |
| **P1** 经典策略 | H2O + SnapKV | PPL + Needle 结果 |
| **P2** 二阶段 | RocketKV-light | 与 P1 同任务对比 |
| **P3** 前沿策略 | CurDKV + MUSTAFAR | 完整 PPL+LongBench+Needle |
| **P4** 你的论文 | Mnemosyne | 全量报告 |
| **P5** 工程化 | 可视化、profiling、README | 可复现 benchmark 项目 |
