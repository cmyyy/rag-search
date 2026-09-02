# vaultrag — Vault RAG 检索引擎

rag-search 插件的核心引擎：把个人知识库（Obsidian vault / 任意笔记目录）变成可检索的语义知识库——混合检索 + 排序 + 置信度守卫，返回带来源标注的相关笔记块。

> **完整报告见 [`../docs/rag-search-overview.md`](../docs/rag-search-overview.md)**（设计/原理/参数/评测/运行实例，面试可讲）；参数统计背书见 `evals/parameter_audit.md`。

## 一句话原理（当前状态 2026-08-30）

```
用户查询
  → [长度门槛] <2 字直接跳过（"好""嗯"确认消息；英文缩写"MoA"不误杀）
  → 混合检索：BM25 中文 bigram（本地）+ bge-m3 向量（云端）各取 top-48
  → 块级 RRF 融合（k=60，每笔记取最优块）→ 8 候选
  → cross-encoder rerank（bge-reranker-v2-m3）对前 8 候选打分（行边界截断）
  → [CRAG 三档] top1 ≥ 0.60 Correct（注入）/ 0.02~0.60 Ambiguous（低置信，不注入）/ < 0.02 Incorrect（拦截）
  → 注入 top-8 块 × 每块 600 字符（工具返回文本，带来源标注）
  → 每次调用写结构化 trace（JSONL）到 vault/.smart-env/vaultrag/trace.jsonl
  → 任何失败 → 返回 None/skipped → fail-open（绝不打断调用方）
```

## 结构

- `__init__.py` — `VaultRAGEngine`：`search()` 唯一检索入口（单一事实源）；guard 三档判定；trace
- `retriever.py` — 索引（split_markdown 按标题切块）+ BM25（纯 numpy 中文 bigram）+ 块级 RRF 融合
- `embedding.py` — 云端 embedding（bge-m3）/ rerank（bge-reranker-v2-m3）客户端，fail-open
- `evals/` — 评测体系（93 条 golden cases × 四层）+ 参数审计

## 设计决策

| 决策 | 选择 | 为什么 |
|---|---|---|
| 单一事实源 | `search()` 唯一入口（工具/评测共用） | 杜绝双份逻辑漂移（曾有两路实现不一致的教训）；select_context（context engine 时代遗留）已删 |
| 混合检索 | BM25 + 向量 → 块级 RRF 融合 | BM25 补精确匹配（缩写/术语/ID）、向量补语义；RRF 规避量纲不可比 |
| 块级融合 | 融合键 (笔记, 块)，每笔记取最优块 | 修复"笔记对了、块错了"（by_source last-wins）——L4 失败主因 |
| rerank 双重职责 | cross-encoder 一次调用：guard 打分 + hits 重排 | 零额外成本；分数断层区分（正确块 0.92 vs 次名 0.04） |
| guard 三档 | CRAG 思想 + 保守化（Ambiguous 不注入） | 宁缺毋滥：低置信交给 agent 弹性；CRAG 原设计有 web search 兜底，我们没有 |
| 阈值来源 | 0.60/0.02 由知识库统计双峰分布定 | 可解释、换库重跑 `evals/parameter_audit.py` 重定 |
| 行边界截断 | `truncate_hit()`：截断回退最近行尾 | 硬截 99% 落在行中间（172/174）、18% 在代码块内——行边界让注入文本干净 |
| 打分项数 8 | rerank 只对 RRF 前 8 打分 | 相关块 94% 在 RRF top-8；打分 16 只多救 2% 却慢 4 倍 |
| embedding 云端 | API 调用，不跑本地模型 | 零本地负担（不下载模型/不占内存）；成本仅工具主动调用时产生 |
| fail-open | 任何失败返回 None/skipped | 检索是增强不是依赖，绝不打断对话 |
| 可审计 | 每次调用写 trace（query/verdict/分数/原因） | 出问题逐条回放 |

## 评测与性能（当前）

- 93 条 golden cases（查询 + 期望笔记 + AI 简洁答案）× 四层评测：字面命中 / 守卫拦截 / 注入线敏感性 / DeepEval LLM-as-judge
- 结果：单篇命中 91%、无关 0 误放、L4 语义覆盖 98% / 排序 94%（详见 `evals/eval_report_v2.md`）
- 单次查询热 3~5s（embedding + rerank 云端）；索引加载 ~7s；混合检索本地 0.02s

## 复现

```bash
python vaultrag/evals/study_eval.py      # 全量评测（3 次运行约 40 分钟，含 L4 judge）
python vaultrag/evals/parameter_audit.py # 参数统计审计（换库重跑重定参数）
```
