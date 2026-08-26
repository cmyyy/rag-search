"""候选多样性验证：按笔记去重 vs 原始 RRF，对比全体 100 条指标。

回答关键问题：按笔记去重会不会伤害已有的成功案例？
对比：原始 hybrid top5 vs 去重后 top5（同笔记只留最高分块）
"""
import sys
sys.path.insert(0, r"D:\AI\hermes-agent")
sys.path.insert(0, r"D:\AI\hermes-agent\plugins\context_engine\vaultrag")
from dotenv import load_dotenv
load_dotenv(r"D:\AI\hermes-agent\.env", override=True)

from pathlib import Path
from retriever import VaultIndex, BM25Index, scan_vault
from evals.eval_queries import get_queries
from embedding import EmbeddingClient


def build_index():
    emb = EmbeddingClient()
    docs = scan_vault(r"D:\llmwiki\llm-wiki")
    texts, sources = [], []
    for d in docs:
        for title, body in d["chunks"]:
            texts.append(f"{title}\n{body}" if title else body)
            sources.append(d["path"])
    idx = VaultIndex.__new__(VaultIndex)
    idx._texts = texts
    idx._sources = sources
    idx.bm25 = BM25Index(texts)
    idx.embedding = emb
    idx._matrix = emb.embed_texts(texts)
    return idx


def hybrid_ranked(index, query: str, top_k: int = 16):
    """混合检索返回 RRF 排序的 top-k 全列表（含 source）。"""
    qv = index.embedding.embed_query(query)
    if qv is None:
        return []
    return index.hybrid_search(query, qv, top_k=top_k)


def dedup_by_note(hits):
    """按笔记去重：同笔记只保留 RRF 分最高的块，保持顺序。"""
    seen, out = set(), []
    for h in hits:
        stem = Path(h["source"]).stem
        if stem in seen:
            continue
        seen.add(stem)
        out.append(h)
    return out


def metrics(stems_fn, queries):
    hit1 = recall = mrr = total = 0
    for q in queries:
        expected = set(q["expected_notes"])
        stems = stems_fn(q)
        total += 1
        if stems and stems[0] in expected:
            hit1 += 1
        top5 = stems[:5]
        hit_exp = [e for e in expected if e in top5]
        recall += len(hit_exp) / len(expected)
        for rank, s in enumerate(stems[:10], 1):
            if s in expected:
                mrr += 1.0 / rank
                break
    return hit1 / total, recall / total, mrr / total


idx = build_index()
queries = [q for q in get_queries(None) if q["type"] != "negative"]

def original_top5(q):
    return [Path(h["source"]).stem for h in hybrid_ranked(idx, q["query"])[:5]]

def dedup_top5(q):
    hits = hybrid_ranked(idx, q["query"], top_k=16)
    return [Path(h["source"]).stem for h in dedup_by_note(hits)[:5]]

h1a, r5a, mrr_a = metrics(original_top5, queries)
h1b, r5b, mrr_b = metrics(dedup_top5, queries)

print(f'查询集: {len(queries)} 条正样本（有面包屑索引）')
print()
print(f'{"方法":<24} {"Hit@1":>8} {"Recall@5":>10} {"MRR@10":>8}')
print(f'{"原始 RRF top5":<24} {h1a*100:>7.1f}% {r5a*100:>9.1f}% {mrr_a:>8.3f}')
print(f'{"按笔记去重 top5":<24} {h1b*100:>7.1f}% {r5b*100:>9.1f}% {mrr_b:>8.3f}')
print()
print(f'差异: Hit@1 {h1b-h1a:+.1%} | Recall@5 {r5b-r5a:+.1%} | MRR {mrr_b-mrr_a:+.3f}')
