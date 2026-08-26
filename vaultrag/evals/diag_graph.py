"""诊断：graph 扩展在多跳查询上的实际效果（扩展命中 vs 扩展未命中）。"""
import sys, json
sys.path.insert(0, r"D:\AI\vaultra-graph")
from dotenv import load_dotenv
load_dotenv(r"D:\AI\hermes-agent\.env", override=True)

from evals.eval_queries import get_queries
from embedding import EmbeddingClient
from retriever import VaultIndex
from evals.run_eval import VariantRunner

vault = r"D:\llmwiki\llm-wiki"
emb = EmbeddingClient()
index = VaultIndex(vault, embedding=emb)
ok = index.ensure_index()
print(f"索引: {'OK' if ok else 'FAIL'} size={index.size}")
runner = VariantRunner(index, emb)

queries = get_queries("multi-hop")
for q in queries[:10]:
    d = runner.run_graph(q)
    gs = d["graph_stats"]
    print(f"[{q['id']}] {q['query'][:36]}")
    print(f"    top1={d['top1_source']} score={d['top1_score']} {d['verdict']}")
    print(f"    命中笔记={gs.get('hit_notes')} 扩展笔记={gs.get('expanded_notes')} 扩展块={gs.get('expanded_chunks')} pool={gs.get('pool_size')}")
