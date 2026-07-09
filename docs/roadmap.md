# KV-Cache 稀疏/压缩前沿技术路线图

> 更新：2026-07-09

## 全景分类

KV-cache 优化技术分为三条路径：

```
KV 优化
├── ① Token Eviction（丢弃不重要 token）—— H2O, SnapKV, MNemosyne
├── ② 量化压缩（降低每个 KV 的 bit-width）—— PolarQuant, TurboQuant
└── ③ 结构剪枝 + 稀疏 attention—— RocketKV, MUSTAFAR, DSparK
```

本项目聚焦 **① + ③**（稀疏路线），量化压缩归 `quant-pareto` 项目。

---

## 前沿论文时间线

```
2024                                   2025                                    2026
├─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┤
H2O (ACL) │                                                                              
SnapKV    │                                                                              
          │          RocketKV (ICML, NVIDIA)                                            
          │          AttentionPredictor (NeurIPS, 13×)                                    
          │          SmallKV (NeurIPS, small-model assisted)                              
          │          CurDKV (NeurIPS, CUR decomposition)                                  
          │          KVzip (NeurIPS, context reconstruction)                              
          │          MUSTAFAR (NeurIPS, 70% unstructured sparsity)                        
          │          Mnemosyne (AAAI 2026, 你的论文)                                      
          │                          
                      DSparK-style 动态稀疏 (2025)
```

## 各技术速览

### 第一代（2024）：基于 Attention Score 的 Eviction

| 方法 | 核心 | 算法 | 局限 |
|---|---|---|---|
| **H2O** | Heavy Hitter Oracle | 累计 attention score 排序，保留 top-k | 只看分数不看语义，长序列退化 |
| **SnapKV** | 窗口+聚类 | prompt 部分做 cluster 压缩，decoding 留最近窗口 | 聚类粒度粗 |

### 第二代（ICML 2025）：二阶段设计（从硬件来）

| 方法 | 核心 | 算法 | 亮点 |
|---|---|---|---|
| **RocketKV** (NVIDIA) | 粗粒度 eviction + 细粒度 top-k | Stage1: 分页 evict → Stage2: 辅助 KT cache 稀疏 attention | **400× 压缩 + 3.7× 加速**，已整合 TensorRT-LLM |
| **DSparK-style** | 永久 evict + per-head top-k | two-stage dynamic sparsity | 30-40% mem 减，3-4× 解码加速 |

### 第三代（NeurIPS 2025）：学习/模型辅助

| 方法 | 核心 | 算法 | 亮点 |
|---|---|---|---|
| **AttentionPredictor** | 轻量 conv 模型预测 attention | 捕捉 attention 时空模式 + 跨 token 预取 | **13× 压缩 + 5.6× 加速** |
| **SmallKV** | 小模型补偿大模型 | 小模型提供全局 attention 信号，补偿边际 token | 1.75-2.56× 吞吐提升 |
| **CurDKV** | CUR 矩阵分解 | value-guided leverage score 排序 evict | **比 SnapKV 高 9.6% 精度**，40% 延迟降低 |
| **KVzip** | 上下文重建 | query-agnostic，重建已缓存上下文 | 3-4× 压缩，2× attention 加速 |
| **MUSTAFAR** | 非结构化稀疏 | per-token 幅度剪枝 + bitmap 稀疏格式 | **70% 稀疏无 finetune**，2.2× 吞吐 |

---

## 项目覆盖策略

> 目标：用你能实现的代码量，覆盖尽可能多的技术空间

| 已确定实现 | 理由 |
|---|---|
| **baseline** (no eviction) | 参考线 |
| **H2O** | 最经典 baseline，15 行核心逻辑 |
| **SnapKV** | 另一个经典，对比 cluster eviction vs per-token |
| **RocketKV-light** (二阶段) | 覆盖第二代设计空间 |
| **Mnemosyne** (你的) | 覆盖重排序+分级，与论文呼应 |

| 待确认是否实现 | 理由 |
|---|---|
| **CurDKV** | CUR 分解代码量可控（~80 行），且 NeurIPS 2025，面试信号强 |
| **MUSTAFAR** | 非结构化稀疏是芯片设计的核心话题（sparse tensor core） |
| **AttentionPredictor** | 需要训练小模型，代码量大，放到 v2 |

---

## 芯片视角解读

这些技术对芯片设计的启示：

| 技术 | 硬件含义 |
|---|---|
| RocketKV 二阶段 | 芯片需要支持**粗粒度 eviction 控制** + **细粒度稀疏 attention 引擎** |
| MUSTAFAR 非结构化稀疏 | 芯片需要 **sparse tensor core**（类比 NVIDIA Ampere 的稀疏支持） |
| CurDKV leverage score | chip 需要**快速低精度 score 预估器**（类似 prefetcher 的 confidence 逻辑） |
| SmallKV 小模型补偿 | chip 可以设计**轻量级 lossy 加速器 + 主核验证**的异构架构 |
| DSparK per-head top-k | chip 需要 **per-head 稀疏 mask**，意味着更复杂的 scheduler |

---

## 参考论文

```bibtex
@inproceedings{h2o2024,
  title={H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models},
  booktitle={ACL 2024}
}
@inproceedings{snapkv2024,
  title={SnapKV: LLM Knows What You are Looking for},
  booktitle={ACL 2024}
}
@inproceedings{rocketkv2025,
  title={RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression},
  booktitle={ICML 2025}
}
@inproceedings{attentionpredictor2025,
  title={AttentionPredictor: Temporal Patterns Matter for KV Cache Compression},
  booktitle={NeurIPS 2025}
}
@inproceedings{smallkv2025,
  title={SmallKV: Small Model Assisted Compensation of KV Cache Compression},
  booktitle={NeurIPS 2025}
}
@inproceedings{curdkv2025,
  title={Value-Guided KV Compression for LLMs via Approximated CUR Decomposition},
  booktitle={NeurIPS 2025}
}
@inproceedings{mustafar2025,
  title={MUSTAFAR: Promoting Unstructured Sparsity for KV Cache Pruning in LLM Inference},
  booktitle={NeurIPS 2025}
}
@inproceedings{mnemosyne2026,
  title={Mnemosyne: Accelerating Multi-Hop Question Answering via Cache Hit Order Fitting},
  booktitle={AAAI 2026}
}
```
