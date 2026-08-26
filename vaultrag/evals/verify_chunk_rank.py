"""分块优化对期望笔记排名的影响：旧分块 vs 优化分块，对比 top-16 内期望笔记的排名分布。

回答：优化分块后，期望笔记的排名是否提升（更靠前）？
"""
import sys
sys.path.insert(0, r"D:\AI\hermes-agent")
sys.path.insert(0, r"D:\AI\hermes-agent\plugins\context_engine\vaultrag")
from dotenv import load_dotenv
load_dotenv(r"D:\AI\hermes-agent\.env", override=True)

from pathlib import Path
from evals.ablation_4combo import build_index, hybrid_topk
from evals.eval_queries import get_queries
from collections import defaultdict

pos = [q for q in get_queries(None) if q["type"] != "negative"]

def rank_of_expected(idx, q, top_k=16):
    """返回期望笔记在 top-k 里的排名（1-based）；不在返回 None。"""
    stems = hybrid_topk(idx, q["query"], top_k=top_k)
    ranks = {}
    for e in q["expected_notes"]:
        if e in stems:
            ranks[e] = stems.index(e) + 1
        else:
            ranks[e] = None
    return ranks

print("=== 期望笔记排名（优化分块，top-16 内）===")
print(f'{"查询":<10} {"期望笔记":<42} {"优化分块排名":>10}')
details = []
idx = build_index(True, True)  # 建一次索引复用
for q in pos:
    r_new = rank_of_expected(idx, q)  # 优化分块
    for e in q["expected_notes"]:
        rn = r_new[e]
        details.append((q["id"], e, rn))

# 统计
in_top5 = sum(1 for _, _, r in details if r is not None and r <= 5)
in_top16 = sum(1 for _, _, r in details if r is not None and r <= 16)
total = len(details)
print(f'\n期望笔记总数: {total}')
print(f'在 top-5 内: {in_top5} ({in_top5/total*100:.1f}%)')
print(f'在 top-16 内: {in_top16} ({in_top16/total*100:.1f}%)')
print(f'不在 top-16: {total-in_top16} ({(total-in_top16)/total*100:.1f}%)')
print(f'\n=== top-16 外（召回失败）===')
for qid, e, r in details:
    if r is None:
        print(f'[{qid}] {e}')
print(f'\n=== 排名 >5 但 <=16（排序问题，扩 top-N 可救）===')
for qid, e, r in details:
    if r is not None and r > 5:
        print(f'[{qid}] {e}: 第 {r} 位')
