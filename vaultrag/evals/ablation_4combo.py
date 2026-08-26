"""4 组合消融：有/无面包屑 × 有/无阈值自适应，指标 Hit@1 + Recall@5 + MRR。

基于 100 条查询集（eval_queries.py），在混合检索（hybrid_no_rerank）上交叉：
  组合① 无面包屑 + 固定阈值（旧版）
  组合② 有面包屑 + 固定阈值
  组合③ 无面包屑 + 自适应阈值
  组合④ 有面包屑 + 自适应阈值（新版）

注意：阈值自适应影响 guard（拦截判定），面包屑影响检索排序（top-k 内容）。
Recall@k/MRR/Hit@1 反映排序质量（面包屑贡献）；拦截率反映 guard（阈值贡献）。
"""
import sys, json
sys.path.insert(0, r"D:\AI\hermes-agent")
sys.path.insert(0, r"D:\AI\hermes-agent\plugins\context_engine\vaultrag")

from dotenv import load_dotenv
load_dotenv(r"D:\AI\hermes-agent\.env", override=True)

from pathlib import Path
import numpy as np
from retriever import scan_vault, BM25Index, VaultIndex
from evals.eval_queries import get_queries


def build_index(with_breadcrumb: bool, chunk_opt: bool = True):
    """按开关构建索引（真实 embedding）。
    with_breadcrumb: 面包屑+tags
    chunk_opt: 分块优化（超长二次切+段落兜底）；False = 旧版纯标题切块（max_chunk_chars 极大）
    """
    from embedding import EmbeddingClient
    emb = EmbeddingClient()
    docs = scan_vault(r"D:\llmwiki\llm-wiki", with_breadcrumb=with_breadcrumb,
                      max_chunk_chars=1200 if chunk_opt else 10**9)
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
    idx._vault_hash = f"ablation4-{'y' if with_breadcrumb else 'n'}-{'o' if chunk_opt else 'p'}"
    return idx


def hybrid_topk(index, query: str, top_k: int = 10):
    """混合检索返回 top-k 全列表（source stem 列表）。"""
    qv = index.embedding.embed_query(query)
    if qv is None:
        return []
    hits = index.hybrid_search(query, qv, top_k=top_k)
    return [Path(h["source"]).stem for h in hits]


def hybrid_rerank_topk(index, query: str, top_k: int = 5):
    """混合检索 + cross-encoder rerank 后返回 top-k（source stem 列表）。"""
    from evals.run_eval import VariantRunner, _HYBRID_RECALL, _TOP_K
    qv = index.embedding.embed_query(query)
    if qv is None:
        return []
    candidates = index.hybrid_search(query, qv, top_k=_HYBRID_RECALL)
    if not candidates:
        return []
    # 过滤 index 页（生产适配）
    pool = [c for c in candidates if Path(c["source"]).stem != "index"]
    if not pool:
        return []
    texts = [c["text"][:600] for c in pool]
    rr = index.embedding.rerank(query, texts, top_n=_TOP_K)
    if rr is None:
        return []
    hits = [{**pool[r["index"]], "score": r["score"]} for r in rr]
    return [Path(h["source"]).stem for h in hits[:top_k]]


def compute_metrics(index, queries, top_k=5):
    """对查询集算 Hit@1 / Recall@k / Precision@k / MRR。

    Precision@k = top-k 里命中的期望笔记数 / k（分母固定 k）——
    衡量"前 k 个结果里有多少是相关的"（Recall 高时检查是否靠混入无关堆出来的）。
    """
    hit1 = recall = prec = mrr = 0
    total = 0
    for q in queries:
        if q["type"] == "negative":
            continue  # 负样本单独算拦截
        expected = set(q["expected_notes"])
        if not expected:
            continue
        stems = hybrid_topk(index, q["query"], top_k=10)
        total += 1
        # Hit@1
        if stems and stems[0] in expected:
            hit1 += 1
        # Recall@k：top-k 里命中期望的比例（期望多篇，算覆盖比例）
        topk = stems[:top_k]
        hit_expected = [e for e in expected if e in topk]
        recall += len(hit_expected) / len(expected)
        # Precision@k：命中数 / k（分母固定）
        prec += len(hit_expected) / top_k
        # MRR：第一个命中期望的位置
        for rank, s in enumerate(stems[:10], 1):
            if s in expected:
                mrr += 1.0 / rank
                break
    return {
        "hit@1": hit1 / total if total else 0,
        f"recall@{top_k}": recall / total if total else 0,
        f"precision@{top_k}": prec / total if total else 0,
        "mrr@10": mrr / total if total else 0,
        "n": total,
    }


def main():
    queries = get_queries(None)
    neg = [q for q in queries if q["type"] == "negative"]
    pos = [q for q in queries if q["type"] != "negative"]
    print(f"查询集: 共 {len(queries)}（正样本 {len(pos)} / 负样本 {len(neg)}）\n")

    # 四组合
    combos = [
        ("① 无面包屑", False),
        ("② 有面包屑", True),
    ]
    print("=== 混合检索（无 rerank）排序质量 ===")
    print(f"{'组合':<14} {'Hit@1':>8} {'Recall@5':>10} {'P@5':>8} {'MRR@10':>8}")
    for label, with_bc in combos:
        idx = build_index(with_bc)
        m = compute_metrics(idx, pos)
        print(f"{label:<14} {m['hit@1']*100:>7.1f}% {m['recall@5']*100:>9.1f}% {m['precision@5']*100:>7.1f}% {m['mrr@10']:>8.3f}")


if __name__ == "__main__":
    main()
