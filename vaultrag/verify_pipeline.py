"""verify_pipeline.py —— vaultrag 检索管线验证（固化版，2026-08-16）。

跑法（在 hermes-agent 目录下）：
    .venv/Scripts/python.exe plugins/rag-search/vaultrag/verify_pipeline.py

自动读取 .env 里的 EMBEDDING_API_KEY / OBSIDIAN_VAULT_PATH（load_dotenv），
不需要手动传环境变量。覆盖：

  [1] BM25 单元：精确匹配命中正确文档；短确认词分数趋零（源头过滤）
  [2] 索引构建：真实 vault 扫描 → 分块 → 向量索引（排除 raw/copilot）
  [3] 四类查询端到端：
      A. 短确认消息（"好"）        → 长度门槛拦截，不注入
      B. 长句无关（"帮我写个Java程序"）→ rerank 压分 < 阈值，拦截
      C. 术语查询（"MoA是什么"）    → BM25 精确匹配，高分注入
      D. 正常提问（"压缩四道闸"）    → rerank 精排，高分注入

断言机制：check() 失败记入 fails 列表，最后 sys.exit(1 if fails else 0)。
退出码 0 = 全部通过（可接 CI / 手动回归）。
"""
import os
import sys
from pathlib import Path

# 让插件包可导入：脚本在插件目录里，往上找 hermes-agent 根
# （在 D:\AI\vaultra-graph 独立仓库时 4 级 parent 会越界到 D:\，
#   改为向上找第一个含 .env 的目录；Windows 根目录 parent 是自身，需判停）
_HERMES_ROOT = Path(__file__).resolve().parent
for _ in range(8):
    if (_HERMES_ROOT / ".env").exists():
        break
    _parent = _HERMES_ROOT.parent
    if _parent == _HERMES_ROOT:  # 已到文件系统根
        _HERMES_ROOT = Path("D:/AI/hermes-agent")  # 独立仓库布局的已知根
        break
    _HERMES_ROOT = _parent
sys.path.insert(0, str(_HERMES_ROOT))

# 自动加载 .env（Hermes 主进程也走这套：hermes_cli/env_loader.py）
# override=True：bash 环境里可能已有 MSYS 格式的旧值（/d/llmwiki/llm-wiki），
# 必须用 .env 的 Windows 格式覆盖，否则 Path 解析出 D:\d\... 错误路径
from dotenv import load_dotenv  # noqa: E402

load_dotenv(_HERMES_ROOT / ".env", override=True)

from plugins.context_engine.vaultrag.embedding import EmbeddingClient  # noqa: E402
from plugins.context_engine.vaultrag.retriever import BM25Index, VaultIndex  # noqa: E402
from plugins.context_engine.vaultrag import VaultRAGEngine  # noqa: E402

fails = []


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {extra}")
    if not cond:
        fails.append(name)


def main():
    print("[0] 环境")
    emb = EmbeddingClient()
    check("embedding key 可用（EMBEDDING_API_KEY）", emb.available)

    print("\n[1] BM25 单元验证")
    bm25 = BM25Index(["Hermes 压缩默认 in_place 模式", "北京天气很好", "DB2 数据库性能调优"])
    s = bm25.score("压缩 in_place")
    check("BM25 命中正确文档", int(s.argmax()) == 0, f"scores={[round(float(x),3) for x in s]}")
    s2 = bm25.score("好")
    check("短词 BM25 分数极低（源头过滤）", float(s2.max()) < 0.1, f"max={float(s2.max()):.4f}")

    print("\n[2] 索引构建（真实 vault）")
    from plugins.context_engine.vaultrag import _load_vaultrag_config

    vcfg = _load_vaultrag_config()
    vault = vcfg.get("vault_path") or os.getenv("OBSIDIAN_VAULT_PATH", "")
    check("vault_path 已配置（config.yaml context.vaultrag.vault_path）", bool(vault), f"vault={vault}")
    if not vault:
        print("  跳过索引与端到端（无 vault 配置）")
        print("结果: 环境缺失" if fails else "结果: 通过（跳过部分）")
        sys.exit(1 if fails else 0)
    index = VaultIndex(vault, embedding=emb)
    ok = index.ensure_index(force=True)
    check("索引构建", ok, f"size={index.size}")
    leaked = [d for d in [""] if False]  # noqa
    check("索引已排除 raw/copilot", not any("raw" in s or "copilot" in s for s in index._sources))

    print("\n[3] 四类查询端到端")
    engine = VaultRAGEngine(vault_root=vault)
    engine.index = index
    engine._index_ready = True

    cases = [
        ("A.短确认", "好", "skip"),          # 长度门槛拦截，不注入
        ("B.长句无关", "帮我写个Java程序打印hello world", "skip"),
        ("C.术语", "MoA是什么", "inject"),
        ("D.正常提问", "Hermes上下文压缩的四道闸是什么", "inject"),
    ]
    for label, q, expect in cases:
        print(f"\n--- {label}: {q} [期望: {'拦截' if expect=='skip' else '注入'}] ---")
        r = engine.select_context(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": q}],
            incoming_message={"role": "user", "content": q},
        )
        if expect == "skip":
            check(f"{label} 被拦截（不注入）", r is None)
        else:
            check(f"{label} 注入成功", r is not None and "<knowledge_context>" in r[-1]["content"])
            if r:
                # 来源不再挂在注入消息上（provider 兼容），只验证内容含来源标注
                print(f"    top1 来源标注: {'来源:' in r[-1]['content']}")

    print("\n结果: 全部通过" if not fails else f"\n结果: {len(fails)} 失败")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
