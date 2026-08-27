"""L4 only runner：跑一次生产路径 + DeepEval judge，结果写 l4_result.json。"""
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, r"D:\AI\hermes-agent")
try:
    from dotenv import load_dotenv
    load_dotenv(r"D:\AI\hermes-agent\.env", override=True)
except Exception:
    pass

from study_eval import judge_semantic, run_production_once
from vaultrag import VaultRAGEngine
from evals.eval_queries import get_queries

engine = VaultRAGEngine()
queries = get_queries(None)
results = run_production_once(engine, queries, [])
out = judge_semantic(engine, queries, results)
print("L4 结果:", json.dumps(out, ensure_ascii=False, indent=2))
(_HERE / "l4_result.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
print("已写入 l4_result.json")
