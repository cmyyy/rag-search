"""硬核指标（用户要求，2026-08-22）：4 个绝对数字，不平均。

生产配置（有面包屑 + max_chunk_chars=1600）：
1. single-hop Top1 命中数（目标 ≥30/35）
2. multi-hop 完全命中数（所有 expected 都在 top5，目标 ≥16/20）
3. abbreviation 命中数（目标 ≥13/15）
4. negative 误报数（目标 =0/15，用生产 guard 真实判定）

方法：直接用 VaultRAGEngine.select_context 走完整生产路径（含 guard），
正样本看注入来源、负样本看是否拦截。不走评测变体（避免口径偏差）。
"""
import sys, re
sys.path.insert(0, r"D:\AI\hermes-agent")
sys.path.insert(0, r"D:\AI\hermes-agent\plugins\context_engine")
sys.path.insert(0, r"D:\AI\hermes-agent\plugins\context_engine\vaultrag")
from dotenv import load_dotenv
load_dotenv(r"D:\AI\hermes-agent\.env", override=True)

from vaultrag import VaultRAGEngine
from evals.eval_queries import get_queries

engine = VaultRAGEngine()

def call_engine(q):
    """走生产 select_context，返回 (是否注入, 注入来源列表)。"""
    try:
        r = engine.select_context(
            [{"role": "system", "content": "You are helpful."},
             {"role": "user", "content": q}],
            conversation_messages=[],
            incoming_message=q,
            budget_tokens=4000,
        )
    except Exception as e:
        return False, [f"ERR:{str(e)[:50]}"]
    if not r:
        return False, []
    txt = str(r)
    # 提取注入来源（<knowledge_context> 里的 来源: xxx.md）——文件名取最后一个 \ 后的 stem
    srcs = re.findall(r"来源:.*?([^\\\\/]+)\.md", txt)
    srcs = [s.strip() for s in srcs]
    return True, srcs

# ---- 4 类统计 ----
queries = get_queries(None)
stats = {"single-hop": {"top1_hit": 0, "total": 0},
         "multi-hop": {"full_hit": 0, "total": 0},
         "abbreviation": {"top1_hit": 0, "total": 0},
         "concept-link": {"top1_hit": 0, "total": 0, "full_hit": 0},
         "negative": {"false_pos": 0, "total": 0}}

print("=== 逐条结果 ===")
for q in queries:
    qid, typ, query, expected = q["id"], q["type"], q["query"], q["expected_notes"]
    injected, srcs = call_engine(query)
    exp_set = set(expected)

    if typ == "negative":
        stats["negative"]["total"] += 1
        if injected:
            stats["negative"]["false_pos"] += 1
            print(f"[{qid}] ❌ 误报（注入了）: {query[:30]}")
        else:
            print(f"[{qid}] ✅ 拦截")
    else:
        stats[typ]["total"] += 1
        # top1 命中：注入的第一个来源 in expected
        top1 = srcs[0] if srcs else None
        hit1 = top1 in exp_set if top1 else False
        # 完全命中：所有 expected 都在注入来源里（top5 内）
        src_set = set(srcs)
        full = exp_set.issubset(src_set)
        if typ == "single-hop" and hit1:
            stats["single-hop"]["top1_hit"] += 1
        if typ == "multi-hop" and full:
            stats["multi-hop"]["full_hit"] += 1
        if typ == "abbreviation" and hit1:
            stats["abbreviation"]["top1_hit"] += 1
        if typ == "concept-link":
            if hit1:
                stats["concept-link"]["top1_hit"] += 1
            if full:
                stats["concept-link"]["full_hit"] += 1
        mark = "✓" if (typ == "single-hop" and hit1) or (typ == "multi-hop" and full) or (typ == "abbreviation" and hit1) or (typ == "concept-link" and full) else "✗"
        print(f"[{qid}] {mark} {query[:28]} top1={top1 or '无'} 命中全部={full}")

print()
print("=== 硬核指标（生产配置：面包屑+1600）===")
for k, v in stats.items():
    print(f"{k}: {v}")

s = stats
print()
print("=== 目标对比 ===")
print(f"single-hop Top1: {s['single-hop']['top1_hit']}/{s['single-hop']['total']}（目标 ≥30/35）")
print(f"multi-hop 完全命中: {s['multi-hop']['full_hit']}/{s['multi-hop']['total']}（目标 ≥28/35）")
print(f"abbreviation Top1: {s['abbreviation']['top1_hit']}/{s['abbreviation']['total']}（目标 ≥13/15）")
print(f"negative 误报: {s['negative']['false_pos']}/{s['negative']['total']}（目标 0/15）")
