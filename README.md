# rag-search

Hermes Agent 插件：`rag_search` 工具 —— 在本地知识库（个人笔记/文档库）做 **RAG 语义检索**，返回带来源标注的相关笔记片段。

让 agent 自主判断何时使用：工具描述中性，不预设调用策略；`search_files` 等字面检索覆盖不了的问题，agent 会自行调用它做语义兜底。

## 结构

```
rag-search/
├── __init__.py      注册入口：rag_search 工具 + /llm-wiki-init 索引命令
├── tools.py         schema + handler + check_fn（检索流程编排）
├── plugin.yaml      插件声明（provides_tools: [rag_search]）
└── vaultrag/        检索引擎（VaultIndex 混合检索 + CRAG guard + 评测体系）
```

## 安装

```bash
# 1. 放到 Hermes 的插件目录（仓库 plugins/ 或 ~/.hermes/plugins/）
git clone git@github.com:cmyyy/rag-search.git <hermes>/plugins/rag-search

# 2. 启用工具集
hermes tools enable rag
```

未配置 vault 路径时 `check_fn` 返回 False，工具自动不出现——配好即用。

## 配置

`config.yaml` 的 `context.vaultrag` 段（优先级：显式参数 > config.yaml > 环境变量）：

```yaml
context:
  vaultrag:
    vault_path: D:/llmwiki/llm-wiki   # 知识库根目录（Obsidian vault / 任意笔记目录）
    embedding:
      base_url: https://api.siliconflow.cn/v1
      model: BAAI/bge-m3
```

环境变量兜底：

| 变量 | 默认值 | 作用 |
|------|--------|------|
| `OBSIDIAN_VAULT_PATH` | — | vault 路径（config 未设时） |
| `EMBEDDING_API_KEY` | — | embedding 服务 key（必填） |
| `EMBEDDING_BASE_URL` | `https://api.siliconflow.cn/v1` | embedding 端点 |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | embedding 模型 |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | rerank 模型（guard 用） |

## 使用

首次调用自动懒加载建索引（扫描 vault + chunk + BM25 索引）；也可手动初始化：

```
/llm-wiki-init <vault路径>
```

工具参数：`rag_search(query: string, top_k: int = 4)`——自然语言查询，返回排序后的命中片段 + 来源标注。

## 特性

- **混合检索**：BM25 字面 + bge-m3 向量 → 块级 RRF 融合（每篇笔记取最优块，2026-08-28）
- **CRAG guard 三档**：rerank 分数判 Correct（top1 ≥ 0.60，注入）/ Ambiguous（0.02~0.60，低置信**不注入**）/ Incorrect（< 0.02，拦截）——宁可无结果，不错注入；margin 判据已移除（2026-08-28，rerank 全量重排后失真）
- **rerank 全量重排**：cross-encoder 分数参与排序（2026-08-28 起，原来只当守卫门槛）
- **查询不做增强**：缩写展开 / query2doc 已删除（2026-08-26 消融实验：hit@1 零贡献、无条件增强误放行生活负样本）——guard 用原始查询保守判定，短技术查询被拦时由 agent 换词重查
- **多概念降级**：含比较/分隔符的查询放宽拦截线（0.15）
- **fail-safe**：索引未就绪 / embedding 失败 / guard 不过，均有明确提示返回，不抛错；rerank API 失败回退 RRF 序
- **可配置通用**：不绑定任何特定知识库，配好 vault 路径即用

## 评测

`vaultrag/evals/`：93 条 golden cases（查询 + 期望笔记 + 人写答案），四层评测（字面命中 / 守卫拦截 / 注入线敏感性 / DeepEval LLM-as-judge 语义判定），报告 `eval_report_v2.md` 含修复前后对比。复现：`python evals/study_eval.py`（3 次运行约 40 分钟）。

## 注意

- embedding / rerank 需要外部 API（SiliconFlow 等 OpenAI 兼容端点），费用按调用计——工具只在 agent 主动调用时产生成本
- guard 阈值带评测依据（见 `vaultrag/evals/`）；检索判定逻辑集中在引擎 `search()`（select_context 与工具共用，单一事实源），改动只动一处
