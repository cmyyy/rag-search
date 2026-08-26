"""rag-search 插件 — rag_search 工具（仿 spotify 插件结构，2026-08-25）。

- tools.py：schema + handler + check_fn（检索实现，复用 vaultrag 引擎）
- __init__.py：_TOOLS 元组 + register(ctx) 注册循环 + 索引命令
- plugin.yaml：插件声明（provides_tools）

不依赖 context.engine 是否选中 vaultrag 引擎（context_engine/ 子目录被
通用插件发现排除，故工具注册独立成顶层插件）。

通用化：vault 路径由 context.vaultrag.vault_path（config.yaml）或
OBSIDIAN_VAULT_PATH 环境变量配置；未配置时 check_fn 返回 False，工具
不出现——别人装这个插件，配好 vault 路径即可直接用，与知识库名无关。

提示策略：无 system prompt 注入——存在性靠工具名 + 描述传达
（rag_search），使用时机由 agent 自行判断（不教调用策略）。
"""
from typing import Any

from .tools import RAG_SEARCH_SCHEMA, _check_rag_available, _rag_search
from .vaultrag import _cmd_llm_wiki_init

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


def register(ctx) -> None:
    """注册 rag_search 工具 + 知识库索引命令（插件加载器调用一次）。"""
    for name, schema, handler, check_fn, emoji, description in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="rag",
            schema=schema,
            handler=handler,
            check_fn=check_fn,
            description=description,
            emoji=emoji,
        )
    ctx.register_command(
        "llm-wiki-init",
        _cmd_llm_wiki_init,
        description="初始化知识库索引（扫描 vault + 建索引 + 写入 context 配置）",
        args_hint="<vault路径>",
    )
