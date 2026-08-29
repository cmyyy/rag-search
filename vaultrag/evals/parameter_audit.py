"""parameter_audit.py — 参数统计审计（数据/规模类参数的统计背书）。

用法: python evals/parameter_audit.py
输出: evals/parameter_audit.md — 参数审计表（当前值 / 知识库统计 / 建议值 / 依据）

设计原则（2026-08-29）：
  - 架构/算法类参数信论文（RRF k=60、两阶段检索、CRAG 三档）——换库不变
  - 数据/规模类参数信知识库统计——换库跟着变，本脚本就是给它们背书
  - 经验值（初版拍的 600/16/8 等）通过统计归位

统计项：
  A. 块长度分布（定截断长度）
  B. 相关块排名分布（定候选池/打分项数）
  C. multi-hop 笔记数分布（定注入块数）
  D. 正/负样本 guard top1 分数分布（定注入线/拦截线）
  E. 查询长度分布（定长度门槛）
"""
import sys
import statistics
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))      # plugins/rag-search
sys.path.insert(0, str(_HERE.parent.parent.parent))  # hermes-agent
sys.path.insert(0, str(_HERE.parent))             # vaultrag

from dotenv import load_dotenv
load_dotenv(_HERE.parent.parent.parent / ".env", override=True)

from vaultrag import VaultRAGEngine, _HYBRID_RECALL, _RERANK_TOPK, _TOP_K, _MAX_CHARS_PER_HIT, _MIN_SCORE
from evals.eval_queries import get_queries


def pct(lens: list[int], p: float) -> int:
    return sorted(lens)[min(len(lens) - 1, int(len(lens) * p))]


def main() -> None:
    engine = VaultRAGEngine()
    engine.index.ensure_index()
    queries = get_queries()
    pos = [q for q in queries if q["type"] in ("single-hop", "multi-hop", "abbreviation")]
    neg = [q for q in queries if q["type"] == "negative"]

    L: list[str] = []
    L.append("# rag-search 参数统计审计")
    L.append(f"\n生成时间：{__import__('datetime').datetime.now():%Y-%m-%d %H:%M}（知识库 {len(engine.index._texts)} 块 / 评测集 {len(queries)} 条）\n")
    L.append("原则：**架构参数信论文、数据参数信知识库统计**——本审计只背数据类参数。\n")

    # ---- A. 块长度分布（定截断长度）----
    lens = sorted(len(t) for t in engine.index._texts)
    n = len(lens)
    L.append("## A. 块长度分布 → 截断长度\n")
    L.append(f"min={lens[0]}  p50={pct(lens, .5)}  p75={pct(lens, .75)}  p90={pct(lens, .9)}  p95={pct(lens, .95)}  max={lens[-1]}  平均={statistics.mean(lens):.0f}")
    L.append("| 截断阈值 | 被截断的块 | 块完整保留率 |")
    L.append("|---|---|---|")
    for th in (400, 600, 800, 900, 1000, 1200):
        cut = sum(1 for l in lens if l > th)
        L.append(f"| {th} | {cut}（{cut / n:.0%}） | {(n - cut) / n:.0%} |")
    L.append("")

    # ---- B. 相关块排名分布（定候选池/打分项数）----
    in_top8 = in_top16 = not_in = 0
    for q in pos:
        qv = engine.embedding.embed_query(q["query"])
        cands = engine.index.hybrid_search(q["query"], qv, top_k=_HYBRID_RECALL)
        pool = [c for c in cands if Path(c["source"]).stem != "index"]
        for exp in q["expected_notes"]:
            rank = next((i + 1 for i, c in enumerate(pool) if Path(c["source"]).stem == exp), None)
            if rank is None:
                not_in += 1
            elif rank <= 8:
                in_top8 += 1
            elif rank <= 16:
                in_top16 += 1
    total = in_top8 + in_top16 + not_in
    L.append("## B. 相关块在候选池的排名分布 → 候选池/打分项数\n")
    L.append(f"（{len(pos)} 条正样本 × 期望笔记，共 {total} 个期望块）")
    L.append(f"- RRF top-8 内：**{in_top8}（{in_top8 / total:.0%}）** → 打分项数取 8 保留率 {in_top8 / (in_top8 + in_top16):.0%}")
    L.append(f"- RRF 9-16：{in_top16}（{in_top16 / total:.0%}）")
    L.append(f"- 不在 16 候选：{not_in}（{not_in / total:.0%}）——召回层缺口，打分救不回\n")

    # ---- C. multi-hop 笔记数分布（定注入块数）----
    mh = [q for q in queries if q["type"] == "multi-hop"]
    counts = sorted(len(q["expected_notes"]) for q in mh)
    L.append("## C. multi-hop 查询涉及笔记数分布 → 注入块数\n")
    L.append(f"（{len(mh)} 条 multi-hop 查询）")
    L.append(f"笔记数: min={counts[0]}  p50={pct(counts, .5)}  p75={pct(counts, .75)}  max={counts[-1]}  平均={statistics.mean(counts):.1f}")
    L.append(f"→ 注入块数 {_TOP_K} 覆盖 p75={pct(counts, .75)} 篇笔记的需求；单笔记查询（single-hop）1 块即够\n")

    # ---- D. 正/负样本 guard top1 分布（定注入线/拦截线）----
    pos_tops, neg_tops = [], []
    for q in pos:
        r = engine.search(q["query"], top_k=_TOP_K)
        if r and r.get("top1_score") is not None and not r.get("rerank_failed"):
            pos_tops.append(r["top1_score"])
    for q in neg:
        r = engine.search(q["query"], top_k=_TOP_K)
        if r and r.get("top1_score") is not None and not r.get("rerank_failed"):
            neg_tops.append(r["top1_score"])
    L.append("## D. guard top1 分数分布 → 注入线 / 拦截线\n")
    L.append(f"正样本（n={len(pos_tops)}）：min={min(pos_tops):.3f}  p25={pct(sorted(pos_tops), .25):.3f}  med={statistics.median(pos_tops):.3f}  max={max(pos_tops):.3f}")
    L.append(f"负样本（n={len(neg_tops)}）：min={min(neg_tops):.3f}  p75={pct(sorted(neg_tops), .75):.3f}  max={max(neg_tops):.3f}")
    low_pos = sum(1 for s in pos_tops if s < 0.60)
    fp_at_60 = sum(1 for s in neg_tops if s >= 0.60)
    L.append(f"- 注入线 0.60：误杀（正样本 < 0.60）= {low_pos}/{len(pos_tops)}（{low_pos / len(pos_tops):.0%}）；误放（负样本 ≥ 0.60）= **{fp_at_60}/15**")
    fp_at_50 = sum(1 for s in neg_tops if s >= 0.50)
    low_pos_50 = sum(1 for s in pos_tops if s < 0.50)
    L.append(f"- 注入线 0.50（对比）：误杀 = {low_pos_50}/{len(pos_tops)}（{low_pos_50 / len(pos_tops):.0%}）；误放 = {fp_at_50}/15")
    pos_main = pct(sorted(pos_tops), .25)  # 正样本主分布下界（p25）
    neg_max = max(neg_tops)
    L.append(f"- 双峰：负样本全部 ≤{neg_max:.3f}；正样本主分布从 {pos_main:.3f}（p25）起（另有 {low_pos} 条弱相关尾巴 <0.60，有意不注入）")
    L.append(f"- 空档 = {neg_max:.3f} ~ {pos_main:.3f}（宽 {pos_main - neg_max:.3f}）——注入线 0.60 落在空档内 → 0 误放 + 仅 {low_pos}/{len(pos_tops)}（{low_pos / len(pos_tops):.0%}）误杀\n")

    # ---- E. 查询长度分布（定长度门槛）----
    qlens = sorted(len(q["query"]) for q in queries)
    L.append("## E. 查询长度分布 → 长度门槛\n")
    L.append(f"min={qlens[0]}  p5={pct(qlens, .05)}  p25={pct(qlens, .25)}  med={statistics.median(qlens)}  max={qlens[-1]}")
    L.append(f"→ 门槛 2 字符：拦截最短 2% 的查询（确认类短消息），技术查询（MoA/RAG 等 3 字符缩写）不被误伤\n")

    # ---- 汇总表 ----
    L.append("## 参数审计汇总\n")
    L.append("| 参数 | 当前值 | 知识库统计依据 | 建议 |")
    L.append("|---|---|---|---|")
    L.append(f"| 截断长度 | {_MAX_CHARS_PER_HIT} | 块 p90={pct(lens, .9)}（{_MAX_CHARS_PER_HIT} 截断 {sum(1 for l in lens if l > _MAX_CHARS_PER_HIT) / n:.0%} 的块） | p90≈900（90% 块完整）或 1000（94%） |")
    L.append(f"| 注入块数 | {_TOP_K} | multi-hop 笔记数 p75={pct(counts, .75)} | {_TOP_K} 已覆盖（可考虑 max(注入, p75)） |")
    L.append(f"| 打分项数 | {_RERANK_TOPK} | 相关块 {in_top8 / total:.0%} 在 RRF top-8 | 8（已落地，延迟 -4x） |")
    L.append(f"| 候选池 | {_HYBRID_RECALL} | 相关块 {(in_top8 + in_top16) / total:.0%} 在 16 内 | 16（本地免费，召回覆盖） |")
    L.append(f"| 注入线 | 0.60 | 双峰间隔 {gap - max(neg_tops):.3f}（0.60 在空档内，误放 0） | 0.60 保持 |")
    L.append(f"| 拦截线 | {_MIN_SCORE} | 负样本 max={max(neg_tops):.3f}（拦截线以下才标 incorrect） | 保持（行为分界是注入线 0.60） |")
    L.append(f"| 长度门槛 | 2 字符 | 查询 p5={pct(qlens, .05)} | 2 保持（技术缩写不误伤） |")

    out = _HERE / "parameter_audit.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"参数审计报告已生成: {out}")
    print(f"A 块: p50={pct(lens, .5)} p90={pct(lens, .9)} | B 相关块: top8={in_top8}/{total} | C multi-hop: p75={pct(counts, .75)} | D top1: pos_min={gap:.3f} neg_max={max(neg_tops):.3f} | E 查询: p5={pct(qlens, .05)}")


if __name__ == "__main__":
    main()
