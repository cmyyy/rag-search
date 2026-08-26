"""chunk_size 参数扫描：max_chunk_chars ∈ {800, 1200, 1600}，100 条查询集。

注意：每档块不同 → embedding 矩阵不同 → 必须重建。加失败重试防限流。
"""
import sys, time
sys.path.insert(0, r"D:\AI\hermes-agent")
sys.path.insert(0, r"D:\AI\hermes-agent\plugins\context_engine\vaultrag")
from dotenv import load_dotenv
load_dotenv(r"D:\AI\hermes-agent\.env", override=True)

from evals.ablation_4combo import build_index, compute_metrics
from evals.eval_queries import get_queries
from pathlib import Path

queries = get_queries(None)
pos = [q for q in queries if q["type"] != "negative"]
print(f"查询集: {len(queries)} 条（正样本 {len(pos)}）")
print()
print(f"{'max_chunk_chars':<18} {'Hit@1':>8} {'Recall@5':>10} {'MRR@10':>8} {'块数':>6}")
print("-" * 55)

# 用旧分块（max_chunk_chars 极大）作为对照 + 3 档扫描
for mcc in [10**9, 1600, 1200, 800]:
    for attempt in range(3):  # 重试防限流
        try:
            # 自定义 build：传 max_chunk_chars
            from retriever import VaultIndex, BM25Index, scan_vault
            from embedding import EmbeddingClient
            emb = EmbeddingClient()
            docs = scan_vault(r"D:\llmwiki\llm-wiki", with_breadcrumb=True, max_chunk_chars=mcc)
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
            idx._vault_hash = f"scan-{mcc}"
            m = compute_metrics(idx, pos)
            label = "旧分块(无上限)" if mcc == 10**9 else f"{mcc} 字符"
            print(f"{label:<18} {m['hit@1']*100:>7.1f}% {m['recall@5']*100:>9.1f}% {m['mrr@10']:>8.3f} {len(texts):>6}")
            break
        except Exception as e:
            print(f"  [重试 {attempt+1}] {mcc}: {str(e)[:60]}")
            time.sleep(10)
