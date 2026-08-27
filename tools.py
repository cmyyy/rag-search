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
    检索核心全部委托 engine.search()（单一事实源，与 select_context 共用），
    本函数只做参数解析与结果格式化（2026-08-26 重构：消除双份逻辑漂移）。
    guard 不过会明确告知，不硬给结果（fail-safe：宁可无结果，不错注入）。
    """
    query = str(args.get("query", "") or "").strip() if isinstance(args, dict) else str(args or "")
    try:
        top_k = int(args.get("top_k", 4) or 4) if isinstance(args, dict) else 4
    except (TypeError, ValueError):
        top_k = 4

    from .vaultrag import _MAX_CHARS_PER_HIT

    engine = _get_engine()
    result = engine.search(query, top_k=top_k)

    if result is None:
        return "[rag] 检索失败（引擎异常，fail-open）——请稍后重试或改用其他检索方式"

    reason = result["reason"]
    if reason == "index-not-ready":
        return "[rag] 知识库索引未就绪——请先初始化知识库索引"
    if reason == "no-query":
        return "[rag] 查询为空"
    if reason == "length-gate":
        return "[rag] 查询过短（1 字确认类消息无检索价值）"
    if reason == "embedding-failed":
        return "[rag] embedding 调用失败（检查 embedding 配置/网络）"
    if reason == "no-candidates":
        return "[rag] 无命中"
    if reason == "no-pool":
        return "[rag] 命中均为目录页，无实质内容"
    if reason == "score-below-threshold":
        return (
            f"[rag] 检索无把握（top1 分数 {result['top1_score']:.3f} 低于阈值），"
            "未返回结果——知识库中可能没有相关内容"
        )

    hits = result["hits"]
    if not hits:
        return "[rag] 无命中"
    verdict = result["verdict"]
    top1 = result["top1_score"]

    # 组装返回（带来源标注，供 agent 核对/引用）
    lines = [f"[rag] 命中 {len(hits)} 条（verdict={verdict}, top1={top1:.3f}）:"]
    for i, h in enumerate(hits, 1):
        src = h.get("source", "?")
        text = h.get("text", "")[:_MAX_CHARS_PER_HIT]
        lines.append(f"{i}. 来源 {src}（score={float(h.get('score', 0)):.3f}）:\n{text}")
    return "\n".join(lines)
