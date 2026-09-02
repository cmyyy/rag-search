"""面包屑+标签消融（2026-08-30）：with_breadcrumb=True（现状）vs False（纯文本）。

说明：split_markdown 的 tags 追加被 with_breadcrumb 门控（`if with_breadcrumb and tags`），
所以 with_breadcrumb=False = 无面包屑 + 无 tags（纯文本切块）。
指标：相关块覆盖率（78 条正样本，期望笔记块在候选 top-8 的比例）+ 负样本 top1 上界。
缓存键含 with_breadcrumb，两组索引自动区分，不污染生产缓存。
"""
import sys
from pathlib import Path

sys.path.insert(0, r"D:\AI\hermes-agent")
sys.path.insert(0, r"D:\AI\hermes-agent\plugins\rag-search")
sys.path.insert(0, r"D:\AI\hermes-agent\plugins\rag-search\vaultrag")

from dotenv import load_dotenv
load_dotenv(r"D:\AI\hermes-agent\.env", override=True)

from vaultrag import VaultRAGEngine
from vaultrag.retriever import VaultIndex
from vaultrag.embedding import EmbeddingClient
from evals.eval_queries import get_queries

VAULT = r"D:\llmwiki\llm-wiki"
queries = [q for q in get_queries() if q["type"] in ("single-hop", "multi-hop", "abbreviation")]
negs = [q for q in get_queries() if q["type"] == "negative"]
embedding = EmbeddingClient()

def eval_index(with_breadcrumb: bool):
    idx = VaultIndex(VAULT, embedding=embedding)
    ok = idx.ensure_index(force=True, with_breadcrumb=with_breadcrumb)
    if not ok:
        return None
    print(f"  [{'面包屑+标签' if with_breadcrumb else '纯文本'}] 索引 {len(idx._texts)} 块", flush=True)
    in_top8 = not_in = 0
    for q in queries:
        qv = embedding.embed_query(q["query"])
        pool = idx.hybrid_search(q["query"], qv, top_k=8)
        for exp in q["expected_notes"]:
            rank = next((i + 1 for i, c in enumerate(pool) if Path(c["source"]).stem == exp), None)
            if rank and rank <= 8:
                in_top8 += 1
            else:
                not_in += 1
    neg_tops = []
    for q in negs:
        qv = embedding.embed_query(q["query"])
        pool = idx.hybrid_search(q["query"], qv, top_k=8)
        if pool:
            neg_tops.append(pool[0]["score"])
    total = in_top8 + not_in
    print(f"  相关块 top-8 覆盖: {in_top8}/{total}（{in_top8 / total:.0%}）", flush=True)
    print(f"  负样本 top1: max={max(neg_tops):.4f} p75={sorted(neg_tops)[len(neg_tops) * 3 // 4]:.4f}", flush=True)
    return {"blocks": len(idx._texts), "cover": in_top8 / total, "n": total, "neg_max": max(neg_tops)}

print("=== 组 A：有面包屑+标签（现状）===", flush=True)
a = eval_index(True)
print("=== 组 B：无面包屑+标签（纯文本）===", flush=True)
b = eval_index(False)
print(f"\n对比: 覆盖 {a['cover']:.0%} vs {b['cover']:.0%} | 块数 {a['blocks']} vs {b['blocks']} | 负样本上界 {a['neg_max']:.4f} vs {b['neg_max']:.4f}")
