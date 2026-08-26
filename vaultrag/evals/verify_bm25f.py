"""BM25F 字段加权验证：header_weight ∈ {0, 2, 3, 5}，硬核指标对比。

不依赖生产配置（直接用引擎路径）——测 BM25 header 加权对排序的影响。
"""
import sys, re
sys.path.insert(0, r"D:\AI\hermes-agent")
sys.path.insert(0, r"D:\AI\hermes-agent\plugins\context_engine")
sys.path.insert(0, r"D:\AI\hermes-agent\plugins\context_engine\vaultrag")
from dotenv import load_dotenv
load_dotenv(r"D:\AI\hermes-agent\.env", override=True)

from pathlib import Path
from retriever import VaultIndex, BM25Index, scan_vault
from embedding import EmbeddingClient
from evals.eval_queries import get_queries


def build_index_with_hw(hw):
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
    idx.bm25 = BM25Index(texts, header_weight=hw)
    idx.embedding = emb
    idx._matrix = emb.embed_texts(texts)
    return idx


queries = get_queries(None)

def run(hw):
    idx = build_index_with_hw(hw)
    stats = {"single-hop": [0,0], "multi-hop": [0,0], "abbreviation": [0,0], "negative": [0,0]}
    for q in queries:
        typ = q["type"]
        exp = set(q["expected_notes"])
        qv = idx.embedding.embed_query(q["query"])
        if qv is None:
            continue
        cands = idx.hybrid_search(q["query"], qv, top_k=8)
        stems = [Path(c["source"]).stem for c in cands]
        if typ == "negative":
            stats["negative"][1] += 1
            continue  # 负样本由 guard 拦，这里只测排序
        stats[typ][1] += 1
        if typ in ("single-hop", "abbreviation"):
            if stems and stems[0] in exp:
                stats[typ][0] += 1
        elif typ == "multi-hop":
            if exp.issubset(set(stems)):
                stats["multi-hop"][0] += 1
    return stats

print("=== BM25F header_weight 扫描（检索层 top8，无 guard）===")
print(f'{"hw":<4} {"single":>8} {"multi":>8} {"abbrev":>8}')
for hw in (0, 2, 3, 5):
    s = run(hw)
    print(f'{hw:<4} {s["single-hop"][0]}/{s["single-hop"][1]:<5} {s["multi-hop"][0]}/{s["multi-hop"][1]:<5} {s["abbreviation"][0]}/{s["abbreviation"][1]}')
