"""诊断2：完整 run_graph 抓原始 400 异常来源。"""
import sys, traceback
sys.path.insert(0, r"D:\AI\vaultra-graph")
from dotenv import load_dotenv
load_dotenv(r"D:\AI\hermes-agent\.env", override=True)

from embedding import EmbeddingClient
from retriever import VaultIndex
from evals.eval_queries import get_queries
from evals.run_eval import VariantRunner

emb = EmbeddingClient()
index = VaultIndex(r"D:\llmwiki\llm-wiki", embedding=emb)
index.ensure_index()
runner = VariantRunner(index, emb)

q = get_queries("multi-hop")[0]
print("query:", q["query"])
try:
    d = runner.run_graph(q["query"])
    print("OK:", d["top1_source"], d["top1_score"], d["verdict"])
    print("stats:", d.get("graph_stats"))
except Exception:
    traceback.print_exc()
