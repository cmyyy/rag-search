"""轻量对比：论文方法（Naive RAG / BM25 / Hybrid）vs vaultrag 完整版。

在 llm-wiki 自有集（36 条查询）上跑，方法与论文对齐：
- Naive RAG（Gao 2023，LightRAG 论文基线）= pure_vector
- BM25 RAG（MultiHop-RAG 官方方法）= pure_bm25
- Hybrid（MultiHop-RAG 官方 hybrid_retriever）= hybrid_no_rerank
- vaultrag 完整版 = hybrid_rerank（含面包屑+tags+rerank+guard）
"""
import sys
sys.path.insert(0, r"D:\AI\vaultra-graph")
from dotenv import load_dotenv
load_dotenv(r"D:\AI\hermes-agent\.env", override=True)

from embedding import EmbeddingClient
from retriever import VaultIndex
from evals.eval_queries import get_queries
from evals.run_eval import VariantRunner

emb = EmbeddingClient()
index = VaultIndex(r"D:\llmwiki\llm-wiki", embedding=emb)
ok = index.ensure_index()
print(f"索引: {'OK' if ok else 'FAIL'} size={index.size}")
runner = VariantRunner(index, emb)

all_q = get_queries(None)
types = ["multi-hop", "single-hop", "abbreviation", "concept-link"]

# 论文方法 → 变体名映射
variants = {
    "Naive RAG (纯向量)": "run_pure_vector",
    "BM25 RAG": "run_pure_bm25",
    "Hybrid (BM25+向量)": "run_hybrid_no_rerank",
    "vaultrag 完整版": "run_hybrid_rerank",
}

results = {v: {t: {"hit": 0, "total": 0} for t in types} for v in variants}
neg_results = {v: {"blocked": 0, "total": 0} for v in variants}

for q in all_q:
    qt = q["type"]
    for label, method in variants.items():
        try:
            r = getattr(runner, method)(q["query"])
        except Exception:
            continue
        if qt == "negative":
            neg_results[label]["total"] += 1
            # 负样本拦截：仅 rerank 变体有 guard（verdict 三档），
            # 其他变体（Naive/BM25/Hybrid）无 guard 机制——标 N/A 不判分
            verdict = r.get("verdict")
            if verdict is not None:
                if verdict == "incorrect":
                    neg_results[label]["blocked"] += 1
        else:
            results[label][qt]["total"] += 1
            if r["top1_source"] in q["expected_notes"]:
                results[label][qt]["hit"] += 1

print("\n=== 方法对比（36 条查询，hit@1）===")
print(f"{'方法':<22} {'多跳':>10} {'单跳':>10} {'缩写':>10} {'概念关联':>10} {'负样本拦截':>10}")
for label in variants:
    mh = results[label]["multi-hop"]; sh = results[label]["single-hop"]
    ab = results[label]["abbreviation"]; lk = results[label]["concept-link"]
    ng = neg_results[label]
    def pct(c):
        return f"{c['hit']}/{c['total']}" if c['total'] else "-"
    ng_pct = f"{ng['blocked']}/{ng['total']}" if ng['total'] else "-"
    print(f"{label:<22} {pct(mh):>10} {pct(sh):>10} {pct(ab):>10} {pct(lk):>10} {ng_pct:>10}")

# 汇总（非负样本）
print("\n=== 汇总（非负样本 hit@1）===")
for label in variants:
    hit = sum(results[label][t]["hit"] for t in types)
    total = sum(results[label][t]["total"] for t in types)
    print(f"{label:<22} {hit}/{total} = {hit/total*100:.1f}%" if total else f"{label}: N/A")
