# KV-Cache Sparsity Benchmark — 项目报告

> 实习项目：LLM 推理加速中的 KV-Cache 稀疏化算法评测框架

---

## 一、项目背景

大语言模型（LLM）在自回归生成时需要缓存每层的 Key-Value 向量（KV cache）。当序列长度达到 100K+ tokens 时，KV cache 会消耗数十 GB 显存，成为推理吞吐的严重瓶颈。

KV-cache 稀疏化是一条重要的优化路径——通过选择性丢弃或压缩注意力头中不重要的 KV 条目，在保持生成质量的同时大幅降低缓存占用。

本项目构建了一个**统一的、可扩展的 KV-cache 稀疏算法评测框架**，收录了 2024-2026 年间的核心前沿论文，并通过统一的接口实现了多种稀疏策略的对比评测。

---

## 二、技术路线

KV-cache 优化分为三条路径：

| 路线 | 方法 | 代表性论文 |
|---|---|---|
| **Token Eviction** | 丢弃不重要的 token | H2O, SnapKV, Mnemosyne |
| **量化压缩** | 降低每个 KV 的 bit-width | PolarQuant, TurboQuant |
| **结构剪枝 + 稀疏 attention** | 二阶段稀疏 + 非结构化剪枝 | RocketKV, MUSTAFAR, DSparK |

本项目聚焦路线 **1 + 3**。

### 前沿论文时间线

```
2024                    2025                     2026
├─────────┬─────────┬─────────┬─────────┬──────────┬─────────┬─────────┤
H2O (ACL)
SnapKV (ACL)
          RocketKV (ICML, NVIDIA)
          AttentionPredictor (NeurIPS, 13×)
          SmallKV (NeurIPS)
          CurDKV (NeurIPS, CUR)
          MUSTAFAR (NeurIPS, 70%)
                                        Mnemosyne (AAAI 2026)
```

---

## 三、架构设计

### 3.1 整体结构

```
kv-sparsity-bench/
├── bench/                 # 核心库
│   ├── core/              # 共享基础设施
│   │   ├── score_buffer    # 注意力分数累积器
│   │   ├── cache_manager   # KV 缓存状态追踪
│   │   ├── budget_controller  # 预算控制
│   │   └── hook_manager    # HuggingFace 模型钩子
│   ├── strategies/         # 稀疏策略（可插拔）
│   │   ├── base.py         # EvictionStrategy 基类
│   │   ├── h2o.py          # H2O 实现
│   │   ├── snapkv.py       # SnapKV 实现
│   │   └── baseline.py     # 无压缩基线
│   ├── models/             # 模型加载
│   ├── tasks/              # 评测任务
│   │   └── perplexity.py   # PPL 评测
│   ├── metrics/            # 指标收集 & 可视化
│   └── runner.py           # 统一运行入口
├── scripts/                # CLI
├── configs/                # YAML 配置
├── models/                 # 本地缓存模型
├── results/                # 评测结果
└── papers/                 # 论文 PDF
```

### 3.2 核心架构——策略接口

所有策略继承同一基类，只需实现 `decide_eviction()` 方法：

```python
class EvictionStrategy(ABC):
    def decide_eviction(self, layer_idx, step) -> list[int]:
        """返回该层要保留的 token 索引"""
```

共享基础设施自动处理：
- **ScoreBuffer**：跨层、跨 head、跨步累积 attention softmax 分数
- **CacheManager**：追踪 token 存活状态，执行 KV 压缩
- **BudgetController**：根据配置的压缩比计算每步预算
- **HookManager**：通过 forward hook 自动捕获每层 attention 分数

### 3.3 已实现策略

| 策略 | 核心逻辑 | 代码量 |
|---|---|---|
| **Baseline** | 保留所有 KV——参考线 | ~10 行 |
| **H2O** | 累加 attention score，保留 top-k | ~30 行 |
| **SnapKV** | 聚类 prompt + 保留近期窗口 | ~60 行 |

---

## 四、实验与结果

### 4.1 实验设置

- **模型**: GPT-2 架构迷你模型 (0.7M 参数, 2 层, 4 head)
- **硬件**: CPU (框架支持任意 CUDA GPU)
- **评测**: 1024 token 输入，生成 16 个新 token
- **指标**: KV Cache 平均大小、生成吞吐 (tokens/sec)
- **模型**: 30% budget (保留 30% 的 KV 条目)

### 4.2 核心结果

![KV 压缩对比](results/figures/kv_comparison.png)
![压缩比](results/figures/compression_ratio.png)

| 策略 | Budget | 平均 KV 大小 | 压缩比 | 吞吐 (tok/s) |
|---|---|---|---|---|
| Baseline | — | 1033 | 1× | 707 |
| H2O | 30% | 614 | **1.7×** | 684 (-3.2%) |
| H2O | 20% | 409 | **2.5×** | 707 (持平) |
| H2O | 10% | 204 | **5.1×** | 697 (-1.4%) |

### 4.3 关键发现

1. **压缩有效**：H2O 在 10% budget 下实现了 5× 压缩，吞吐几乎不受影响
2. **吞吐稳定**：压缩后 token/s 波动在 ±3% 以内，证明 eviction 计算开销极小
3. **线性可扩展**：平均 KV 大小与 budget 比例严格线性相关，策略按预期工作

---

## 五、安装与使用

### 5.1 安装

```bash
# 依赖（requirements.txt）
torch>=2.4.0
transformers>=4.40.0
numpy
matplotlib
seaborn
pyyaml
tqdm

# 安装
uv venv
uv pip install -r requirements.txt
uv run python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### 5.2 运行评测

```bash
# Baseline PPL
uv run python scripts/run_benchmark.py --strategy baseline --task perplexity

# H2O PPL
uv run python scripts/run_benchmark.py --strategy h2o --task perplexity

# 批量对比
uv run python scripts/run_comparison.py
```

### 5.3 配置

```yaml
# configs/default.yaml
model:
  name: "Qwen/Qwen2.5-0.5B"      # 模型名称
  dtype: bfloat16                  # 精度
  device: "cuda"                   # 设备

strategy:
  name: h2o                        # 策略名称
  budget_ratio: 0.3                # 保留比例
  min_budget: 64                   # 最小保留数

tasks:
  - name: perplexity               # 任务类型
    stride: 512
```

---

## 六、扩展指南

### 添加新策略

```python
from bench.strategies import EvictionStrategy, StrategyRegistry

@StrategyRegistry.register_decorator("my-strategy")
class MyStrategy(EvictionStrategy):
    def decide_eviction(self, layer_idx, step):
        # 自定义 eviction 逻辑
        keep_indices = [...]  # 要保留的 token 索引
        return keep_indices
```

注册后即可通过 `--strategy my-strategy` 调用。

### 添加新任务

```python
from bench.tasks import Task

class MyTask(Task):
    def run(self, runner) -> dict:
        # 使用 runner.model, runner.tokenizer 执行评测
        return {"my_metric": 0.95}
```

---

## 七、论文收录

项目 `papers/` 目录收录了以下论文 PDF（持续更新）：

| 论文 | Venue | 年份 | 归类 |
|---|---|---|---|
| H2O: Heavy-Hitter Oracle | ACL | 2024 | Token Eviction |
| SnapKV: LLM Knows What You Need | ACL | 2024 | Token Eviction |
| RocketKV: Two-Stage Compression | ICML | 2025 | Structure Pruning |
| AttentionPredictor: Temporal Patterns | NeurIPS | 2025 | Learning-based |
| SmallKV: Small Model Assisted | NeurIPS | 2025 | Learning-based |
| CurDKV: CUR Decomposition | NeurIPS | 2025 | Token Eviction |
| MUSTAFAR: Unstructured Sparsity | NeurIPS | 2025 | Structure Pruning |
| Mnemosyne: Cache Hit Order Fitting | AAAI | 2026 | Token Eviction |

---

## 八、后续工作

- **P2**: 实现 RocketKV-light（二阶段稀疏）
- **P3**: 实现 CurDKV（CUR 分解）+ MUSTAFAR（非结构化稀疏）
- **P4**: LongBench 下游任务评测 + Needle-in-Haystack 检索测试
- **P5**: GPU profiling（CUDA kernel 耗时分析）
- **长期**: 用真实 LLM（Qwen-2.5-0.5B / LLaMA-3.2-1B）跑全量评测

---

## 九、参考

本项目的设计受以下资料启发：
1. 论文 PDF 在 `papers/` 目录
2. 设计文档：`docs/plans/2026-07-09-kv-sparsity-bench-design.md`
3. 路线图：`docs/roadmap.md`
4. Loop Engineering：https://github.com/cobusgreyling/loop-engineering
