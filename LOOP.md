# Loop Configuration — KV-Cache Sparsity Benchmark (Claude Code)

## Active Loops

| Pattern | Cadence | Status | Command |
| Daily Triage | 1d | L1 report-only | `/loop 1d Run $loop-triage` |
| Benchmark Runner | on-demand | L1 | `/loop "run benchmark --strategy h2o --task perplexity"` |

## Project Context

- **目标**: 实现 KV-cache 稀疏算法并评测（精度 vs 压缩率 vs 性能）
- **P0-P5 渐进**: 见 STATE.md High Priority
- **论文**: papers/ 目录下 7 篇 PDF
- **设计**: docs/plans/2026-07-09-kv-sparsity-bench-design.md

## Human Gates

- No auto-fix until L2 checklist complete
- All high-risk paths: human review required (see docs/safety.md denylist)

## Worktrees

- Use `isolation: worktree` when spawning implementer sub-agents (L2+).
- One worktree per fix attempt; discard after verifier REJECT.

## Connectors (MCP)

- MCP optional for L1 report-only loops.
- For L2+: GitHub MCP to read CI/issues; scope connectors to read + comment only until trusted.

## Budget

- Max sub-agent spawns per run: 0 (L1)
- Review STATE.md daily

## Links

- Pattern: [daily-triage](../../patterns/daily-triage.md)
- Checklist: [loop-design-checklist](../../docs/loop-design-checklist.md)