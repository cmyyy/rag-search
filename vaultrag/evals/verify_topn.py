"""方案 A 验证：注入 top-N 4 vs 8（monkey-patch vaultrag._TOP_K），对比硬核指标。"""
import sys, re
sys.path.insert(0, r"D:\AI\hermes-agent")
sys.path.insert(0, r"D:\AI\hermes-agent\plugins\context_engine")
sys.path.insert(0, r"D:\AI\hermes-agent\plugins\context_engine\vaultrag")
from dotenv import load_dotenv
load_dotenv(r"D:\AI\hermes-agent\.env", override=True)

import vaultrag
from evals.eval_queries import get_queries

queries = get_queries(None)

def run_with_topk(topk):
    vaultrag._TOP_K = topk  # monkey-patch 注入条数
    engine = vaultrag.VaultRAGEngine()
    stats = {"single-hop": [0,0], "multi-hop": [0,0], "abbreviation": [0,0], "negative": [0,0]}
    for q in queries:
        qid, typ, query, expected = q["id"], q["type"], q["query"], q["expected_notes"]
        try:
            r = engine.select_context(
                [{"role":"system","content":"You are helpful."},{"role":"user","content":query}],
                conversation_messages=[], incoming_message=query, budget_tokens=8000)
        except Exception:
            r = None
        exp_set = set(expected)
        if typ == "negative":
            stats["negative"][1] += 1
            if r:
                stats["negative"][0] += 1  # 误报
            continue
        stats[typ][1] += 1
        if not r:
            continue  # guard 拦截（未注入）→ 不命中
        txt = str(r)
        srcs = re.findall(r"来源:.*?([^\\\\/]+)\.md", txt)
        srcs = [s.strip() for s in srcs]
        top1 = srcs[0] if srcs else None
        if typ in ("single-hop", "abbreviation") and top1 in exp_set:
            stats[typ][0] += 1
        if typ == "multi-hop" and exp_set.issubset(set(srcs)):
            stats["multi-hop"][0] += 1
    return stats

print("=== 方案 A：注入 top-N 4 vs 8 ===")
print(f"{'指标':<28} {'top4':>8} {'top8':>8}")
for topk in (4, 8):
    s = run_with_topk(topk)
    if topk == 4:
        s4 = s
    else:
        s8 = s
for label, key in [("single-hop Top1", "single-hop"), ("multi-hop 完全命中", "multi-hop"), ("abbreviation Top1", "abbreviation"), ("negative 误报", "negative")]:
    a = s4[key]; b = s8[key]
    print(f"{label:<28} {a[0]}/{a[1]:<5} {b[0]}/{b[1]}")
