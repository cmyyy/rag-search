"""knowledge-search 插件 — rag_search 工具（仿 spotify 插件结构，2026-08-25）。

- tools.py：schema + handler + check_fn（检索实现，复用 vaultrag 引擎）
- __init__.py：_TOOLS 元组 + register(ctx) 注册循环 + system prompt 提示
- plugin.yaml：插件声明（provides_tools）

不依赖 context.engine 是否选中 vaultrag 引擎（context_engine/ 子目录被
通用插件发现排除，故工具注册独立成顶层插件）。

通用化：vault 路径由 context.vaultrag.vault_path（config.yaml）或
OBSIDIAN_VAULT_PATH 环境变量配置；未配置时 check_fn 返回 False，工具
不出现——别人装这个插件，配好 vault 路径即可直接用，与知识库名无关。

提示词策略：只告知"有本地知识库可检索"，不教 agent 什么时候用——
使用时机由 agent 根据工具描述自行判断。
"""
from typing import Any, Mapping

from .tools import RAG_SEARCH_SCHEMA, _check_rag_available, _rag_search

__all__ = ["register"]

# (name, schema, handler, check_fn, emoji, description)
_TOOLS = (
    (
        "rag_search",
        RAG_SEARCH_SCHEMA,
        _rag_search,
        _check_rag_available,
        "📚",
        "在本地知识库（个人笔记/文档库）做语义检索，返回相关笔记片段及来源标注。",
    ),
)


def _kb_hint(_session_info: Mapping[str, Any] | None = None) -> str:
    """渲染知识库提示（每新会话一次）。只告知存在性，不预设使用时机。

    规模信息来自已加载的索引矩阵（size()），未加载时用泛称——
    system prompt 渲染不能被索引构建阻塞。
    """
    size_desc = ""
    try:
        from .tools import _get_engine

        n = _get_engine().index.size()
        if n:
            size_desc = f"（约 {n} 个内容片段）"
    except Exception:
        pass
    return "本地知识库可检索（工具 rag_search）" + size_desc + "：这是你的个人笔记/文档库。"


def register(ctx) -> None:
    """注册 rag_search 工具 + 知识库提示（插件加载器调用一次）。"""
    for name, schema, handler, check_fn, emoji, description in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="knowledge",
            schema=schema,
            handler=handler,
            check_fn=check_fn,
            description=description,
            emoji=emoji,
        )
    ctx.register_system_prompt_section(
        id="knowledge_kb_hint",
        content=_kb_hint,
        position="after_memory",
        max_chars=1000,
    )
