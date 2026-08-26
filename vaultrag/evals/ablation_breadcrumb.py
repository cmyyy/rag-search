"""消融实验：面包屑+frontmatter 的贡献（有 vs 无）。

对比两个配置（同查询集、同检索逻辑，仅切块不同）：
- with_breadcrumb=True  （当前实现：面包屑路径 + tags 注入）
- with_breadcrumb=False （旧版：纯文本切块，无面包屑无 tags）

指标：多跳/单跳 hit@1、负样本拦截、top1 分数。变体：pure_bm25 + hybrid（无 rerank，避免 rerank 噪声）。
"""
import sys
sys.path.insert(0, r"D:\AI\vaultra-graph")
from dotenv import load_dotenv
load_dotenv(r"D:\AI\hermes-agent\.env", override=True)

import numpy as np
from pathlib import Path
from embedding import EmbeddingClient
from retriever import scan_vault, BM25Index, VaultIndex
from evals.eval_queries import get_queries
from evals.run_eval import VariantRunner


def build_index(with_breadcrumb: bool):
    """独立构建两套索引（不依赖 .npz 缓存）。"""
    emb = EmbeddingClient()
    docs = scan_vault(r"D:\llmwiki\llm-wiki", with_breadcrumb=with_breadcrumb)
    texts, sources = [], []
    for d in docs:
        for title, body in d["chunks"]:
            texts.append(f"{title}\n{body}" if title else body)
            sources.append(d["path"])
    # 混合检索需要向量；纯 BM25 不需要
    class _Idx:
        pass
    idx = _Idx()
    idx._texts = texts
    idx._sources = sources
    idx.bm25 = BM25Index(texts)
    idx.emb = emb
    # 复刻 hybrid_search 需要 _matrix/_note_chunks；这里用简化：纯 BM25 + 向量 top 取并集 RRF
    from retriever import VaultIndex as VI
    idx2 = VI.__new__(VI)
    idx2._texts = texts
    idx2._sources = sources
    idx2._note_chunks = idx2._index_notes(sources)
    idx2.bm25 = BM25Index(texts)
    idx2.embedding = emb
    idx2._matrix = emb.embed_texts(texts)  # 真实向量矩阵（云端 bge-m3）
    idx2._vault_hash = "ablation-" + ("y" if with_breadcrumb else "n")
    print(f"    向量矩阵: {None if idx2._matrix is None else idx2._matrix.shape}")
    return idx2, len(texts)


def main():
    all_q = get_queries("multi-hop") + get_queries("single-hop") + get_queries("negative")
    print(f"查询集: {len(all_q)} 条（多跳 {len(get_queries('multi-hop'))} / 单跳 {len(get_queries('single-hop'))} / 无关 {len(get_queries('negative'))}）\n")

    for with_bc in (True, False):
        label = "有面包屑+tags" if with_bc else "无面包屑（旧版）"
        print(f"=== {label} ===")
        idx, nchunks = build_index(with_bc)
        print(f"  切块数: {nchunks}")
        runner = VariantRunner(idx, idx.embedding)

        # 纯 BM25 变体
        bm25_hit, bm25_total = 0, 0
        for q in get_queries("multi-hop") + get_queries("single-hop"):
            try:
                r = runner.run_pure_bm25(q["query"])
                hit = r["top1_source"] in q["expected_notes"]
                bm25_hit += hit
                bm25_total += 1
            except Exception as e:
                print(f"    ✗ {q['id']} {str(e)[:50]}")
        print(f"  纯BM25 hit@1: {bm25_hit}/{bm25_total} = {bm25_hit/bm25_total*100:.1f}%" if bm25_total else "  N/A")

        # 混合（无 rerank）变体
        hyb_hit, hyb_total = 0, 0
        for q in get_queries("multi-hop") + get_queries("single-hop"):
            try:
                r = runner.run_hybrid_no_rerank(q["query"])
                hit = r["top1_source"] in q["expected_notes"]
                hyb_hit += hit
                hyb_total += 1
            except Exception:
                pass
        print(f"  混合(无rerank) hit@1: {hyb_hit}/{hyb_total} = {hyb_hit/hyb_total*100:.1f}%" if hyb_total else "  N/A")
        print()


if __name__ == "__main__":
    main()
