"""rag_search tool — RAG 语义检索（仿 spotify 插件的 tools.py 结构，2026-08-25）。

在本地知识库（个人笔记/文档 vault）做语义检索，返回带来源标注的相关片段。
agent 根据问题性质自行决定是否使用（工具描述中性，不预设调用策略）。

检索与 guard 能力复用 vaultrag context engine（VaultRAGEngine）——
单一实现，不重复造轮子。引擎未安装/不可用时 check_fn 返回 False，工具不出现。
vault 路径由 context.vaultrag.vault_path（config.yaml）或 OBSIDIAN_VAULT_PATH
环境变量配置；未配置时工具自动不可用。
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_engine: Optional[Any] = None


def _get_engine():
    """懒加载单例引擎（延迟导入：插件加载期引擎可能未就绪，首次调用时才解析）。"""
    global _engine
    if _engine is None:
        from .vaultrag import VaultRAGEngine

        _engine = VaultRAGEngine()
    return _engine


def _check_rag_available() -> bool:
    """工具门控：vault 已配置且目录存在（轻量检查，不建索引）。"""
    try:
        eng = _get_engine()
        return bool(eng.vault_root) and os.path.isdir(eng.vault_root)
    except Exception:
        return False


RAG_SEARCH_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": (
        "在本地知识库（个人笔记/文档库）做语义检索，"
        "返回相关笔记片段及来源标注。"
    ),
    "properties": {
        "query": {
            "type": "string",
            "description": "检索问题或关键词（自然语言；会做缩写展开与查询增强）",
        },
        "top_k": {
            "type": "integer",
            "description": "返回命中条数",
            "default": 4,
            "minimum": 1,
            "maximum": 8,
        },
    },
    "required": ["query"],
}


def _rag_search(args: dict, **kwargs) -> str:
    """在本地知识库（个人笔记/文档 vault）做语义检索，返回命中片段（带来源标注）。

    handler 签名遵循工具框架约定（args dict + **kwargs，见 registry.py 调用点）。
    guard 不过会明确告知，不硬给结果（fail-safe：宁可无结果，不错注入）。
    """
    query = str(args.get("query", "") or "").strip() if isinstance(args, dict) else str(args or "")
    try:
        top_k = int(args.get("top_k", 4) or 4) if isinstance(args, dict) else 4
    except (TypeError, ValueError):
        top_k = 4

    # 引擎常量延迟导入（复用引擎的检索参数，单一来源）
    from .vaultrag import (
        _HYBRID_RECALL,
        _MAX_CHARS_PER_HIT,
        _MIN_QUERY_CHARS,
        _MIN_SCORE,
    )

    engine = _get_engine()

    # 1. 索引就绪（懒加载，与 select_context 一致）
    if not engine._index_ready:
        engine._index_ready = engine.index.ensure_index()
    if not engine._index_ready:
        return "[rag] 知识库索引未就绪——请先初始化知识库索引"

    # 2. 缩写展开 + 长度门槛（与 select_context 一致）
    q = engine._expand_abbrev((query or "").strip())
    if not q:
        return "[rag] 查询为空"
    if len(q) < _MIN_QUERY_CHARS:
        return "[rag] 查询过短（1 字确认类消息无检索价值）"

    # 3. 混合检索（向量 + BM25 → RRF）
    qv = engine.embedding.embed_query(q)
    if qv is None:
        return "[rag] embedding 调用失败（检查 embedding 配置/网络）"
    candidates = engine.index.hybrid_search(q, qv, top_k=_HYBRID_RECALL)
    if not candidates:
        return "[rag] 无命中"

    # 4. 过滤自动生成的 index 页（MOC：关键词齐全但无答案，rerank 易误判高分）
    pool = [c for c in candidates if Path(c["source"]).stem != "index"]
    if not pool:
        return "[rag] 命中均为目录页，无实质内容"
    hits = pool[:top_k]

    # 5. guard 判定（与引擎 select_context 保持一致）：
    #    query2doc 增强查询 → rerank top1/top2 → 分数 + margin 三档评估
    guard_query = engine._enhance_guard_query(q, pool)
    cand_texts = [c["text"][:_MAX_CHARS_PER_HIT] for c in pool]
    guard_scores = engine.embedding.rerank(guard_query, cand_texts, top_n=2)
    if guard_scores:
        top1 = float(guard_scores[0]["score"])
        top2 = float(guard_scores[1]["score"]) if len(guard_scores) > 1 else 0.0
    else:
        # rerank 不可用：RRF 分数近似（量纲不同，仅 margin 兜底）
        top1 = float(hits[0]["score"]) if hits else 0.0
        top2 = float(hits[1]["score"]) if len(hits) > 1 else 0.0
    margin = top1 - top2

    # 多概念查询降级（比较标记 + 分隔符 → 阈值放宽；负样本无标记不触发）
    if engine._is_multi_concept(q):
        _min_score = 0.15
        _ambiguous_margin = 0.05
    else:
        _min_score = _MIN_SCORE
        _ambiguous_margin = 0.15

    if top1 < _min_score:
        return (
            f"[rag] 检索无把握（top1 分数 {top1:.3f} 低于阈值 {_min_score}），"
            "未返回结果——知识库中可能没有相关内容"
        )
    verdict = "correct" if (top1 >= 0.40 and margin >= _ambiguous_margin) else "ambiguous"

    # 6. 组装返回（带来源标注，供 agent 核对/引用）
    lines = [f"[rag] 命中 {len(hits)} 条（verdict={verdict}, top1={top1:.3f}）:"]
    for i, h in enumerate(hits, 1):
        src = h.get("source", "?")
        text = h.get("text", "")[:_MAX_CHARS_PER_HIT]
        lines.append(f"{i}. 来源 {src}（score={float(h.get('score', 0)):.3f}）:\n{text}")
    return "\n".join(lines)
