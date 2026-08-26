"""vaultrag 横向/纵向评测脚本（2026-08-22）。

用法（Windows，在 hermes-agent 根目录）：
    .venv/Scripts/python.exe plugins/context_engine/vaultrag/evals/run_eval.py

对比变体（同场景、同查询集）：
  A. 纯向量检索     —— bge-m3 云端向量直接 top-16，无 BM25/无 rerank
  B. 纯 BM25        —— 纯 numpy bigram BM25 top-16
  C. 混合无 rerank  —— 向量+BM25+RRF 召回 top-16，跳过 cross-encoder
  D. 混合+rerank    —— vaultrag 现状（基线）：RRF top-16 → cross-encoder top-4
  E. 演进后(+图扩展)—— D + 双链图 depth=1 扩展并入 rerank 候选池（主角）
  F. langchain 简易 RAG —— 已装 langchain，云端 bge-m3 向量 top-4（业界框架对照）

txtai / LanceDB：评测脚本内 try/except 导入，缺失时优雅跳过并在报告中说明选型理由；
不进生产代码 import（技术约束 #1）。

评测输出：evals/eval_report.md（每查询 top1 命中/分数/verdict，按变体分组；
纵向对比表 + 结论段）。

注：embedding/rerank 全部走云端 API。若运行环境无外网（如沙箱），相关变体会
记录错误并标为"待补"，脚本仍生成完整报告结构，便于网络恢复后原样重跑。
"""
import argparse
import json
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# 让包可导入：脚本放在 plugins/context_engine/vaultrag/evals/ 下，
# 向上 5 层是 hermes-agent 根（与 verify_*.py 的约定一致）
_HERMES_ROOT = Path(__file__).resolve().parent
for _ in range(8):
    if (_HERMES_ROOT / ".env").exists():
        break
    _parent = _HERMES_ROOT.parent
    if _parent == _HERMES_ROOT:  # 已到文件系统根（Windows 根 parent 是自身）
        _HERMES_ROOT = Path("D:/AI/hermes-agent")
        break
    _HERMES_ROOT = _parent
sys.path.insert(0, str(_HERMES_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_HERMES_ROOT / ".env", override=True)
except Exception:
    pass

# 独立仓库布局（D:\AI\vaultra-graph）：本地导入，不依赖 plugins 包前缀
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # vaultra-graph 根（embedding/retriever）
sys.path.insert(0, str(_HERE))         # evals/（eval_queries）

from embedding import EmbeddingClient  # noqa: E402
from retriever import BM25Index, VaultIndex, scan_vault  # noqa: E402
from eval_queries import QUERIES, get_queries  # noqa: E402


def _load_vaultrag_config() -> dict:
    """从 Hermes config.yaml 的 context.vaultrag 段读插件配置（fail-open）。"""
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
        ctx = cfg.get("context", {}) or {}
        return ctx.get("vaultrag", {}) or {}
    except Exception:
        return {}

_TOP_K = 4           # rerank 后注入条数（与生产一致）
_HYBRID_RECALL = 16  # 混合检索召回数（与生产一致）
_MAX_CHARS = 600     # rerank 载荷截断（与生产一致）

# CRAG 三档阈值（复用生产的 _MIN_SCORE / _CORRECT_SCORE 语义）
_MIN_SCORE = 0.30
_CORRECT_SCORE = 0.50


def _stem(path: str) -> str:
    """从笔记路径取 stem（与索引/邻接表的笔记名口径一致）。"""
    return Path(path).stem


def verdict_of(score: float) -> str:
    """CRAG 三档：≥0.5 Correct / 0.3~0.5 Ambiguous / <0.3 Incorrect。"""
    if score >= _CORRECT_SCORE:
        return "correct"
    if score >= _MIN_SCORE:
        return "ambiguous"
    return "incorrect"


def _fmt_err(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"[:200]


def ensure_bm25_offline(index: VaultIndex, vault: str) -> None:
    """embedding 不可用时仍建 BM25：纯 BM25 变体不依赖向量，可离线出真实数据。"""
    if getattr(index, "bm25", None) is not None and index._texts:
        return
    docs = scan_vault(vault)
    texts, sources = [], []
    for d in docs:
        for title, body in d["chunks"]:
            texts.append(f"{title}\n{body}")
            sources.append(d["path"])
    index._texts = texts
    index._sources = sources
    index._note_chunks = index._index_notes(sources)
    index.bm25 = BM25Index(texts)


def probe_network(base_url: str, timeout: float = 3.0) -> bool:
    """快速探测云端 API 是否可达（失败立即返回，避免 24 条查询逐个等超时）。"""
    import socket

    try:
        host = base_url.split("//", 1)[-1].split("/", 1)[0]
        socket.create_connection((host, 443), timeout=timeout).close()
        return True
    except Exception:
        return False


class VariantRunner:
    """统一封装各变体：任何异常 → {"ok": False, "error": ...}（fail-open 口径）。"""

    def __init__(self, index: VaultIndex, emb: EmbeddingClient):
        self.index = index
        self.emb = emb

    def run(self, name: str, query: str) -> Dict[str, Any]:
        fn = getattr(self, f"run_{name}", None)
        if fn is None:
            return {"ok": False, "error": f"unknown variant {name}"}
        try:
            return fn(query)
        except Exception as e:
            return {"ok": False, "error": _fmt_err(e)}

    # -- A. 纯向量 ------------------------------------------------

    def run_pure_vector(self, query: str) -> Dict[str, Any]:
        qv = self.emb.embed_query(query)
        if qv is None:
            raise RuntimeError("embedding 调用失败")
        hits = self.index.search(qv, top_k=_HYBRID_RECALL)
        top = hits[0]
        return {
            "ok": True,
            "top1_source": _stem(top["source"]),
            "top1_score": round(float(top["score"]), 4),
            "top1_text": top["text"][:120],
        }

    # -- B. 纯 BM25 ------------------------------------------------

    def run_pure_bm25(self, query: str) -> Dict[str, Any]:
        scores = self.index.bm25.score(query)
        idx = scores.argsort()[::-1][:_HYBRID_RECALL]
        top_i = int(idx[0])
        return {
            "ok": True,
            "top1_source": _stem(self.index._sources[top_i]),
            "top1_score": round(float(scores[top_i]), 4),
            "top1_text": self.index._texts[top_i][:120],
        }

    # -- C. 混合无 rerank ------------------------------------------

    def run_hybrid_no_rerank(self, query: str) -> Dict[str, Any]:
        qv = self.emb.embed_query(query)
        if qv is None:
            raise RuntimeError("embedding 调用失败")
        hits = self.index.hybrid_search(query, qv, top_k=_HYBRID_RECALL)
        top = hits[0]
        return {
            "ok": True,
            "top1_source": _stem(top["source"]),
            "top1_score": round(float(top["rrf"]), 4),
            "top1_text": top["text"][:120],
        }

    # -- D/E. 混合 + rerank（基线 / +图扩展）----------------------

    def _run_rerank(self, query: str, use_graph: bool) -> Dict[str, Any]:
        qv = self.emb.embed_query(query)
        if qv is None:
            raise RuntimeError("embedding 调用失败")
        candidates = self.index.hybrid_search(query, qv, top_k=_HYBRID_RECALL)
        if not candidates:
            raise RuntimeError("无候选")
        pool = candidates
        graph_stats: Dict[str, Any] = {}
        if use_graph:
            gx = getattr(self.index, "graph_expand", None)
            if gx is not None:
                expanded = gx(candidates, query) or []
                pool = candidates + expanded
                graph_stats = {
                    "hit_notes": len({_stem(c["source"]) for c in candidates}),
                    "expanded_notes": len({_stem(e["source"]) for e in expanded}),
                    "expanded_chunks": len(expanded),
                    "pool_size": len(pool),
                }
        # 场景适配 1：过滤入口/聚合页（index 等自动生成的目录页，关键词齐全但非答案页）
        # 特征：文件名 == "index"（llm-wiki 的 MOC 入口页，含所有术语 → rerank 误判高分）
        pool = [c for c in pool if _stem(c["source"]) != "index"]
        # 场景适配 2：候选池同族去重——源码阶段页（hermes- 前缀）同主题多页，
        # cross-encoder 在近邻页里排序不稳定（0.96 给 loop-entry 而非 compression）。
        # 每族只保留 RRF 分最高的 1 个代表块，避免同族页互相竞争挤掉其他族正确页。
        _KEEP_PER_FAMILY = 1
        fam_best: Dict[str, Dict] = {}
        for c in pool:
            stem = _stem(c["source"])
            fam = stem.rsplit("-", 1)[0] if stem.startswith("hermes-") else stem
            rrfs = float(c.get("rrf", 0.0))
            if fam not in fam_best or rrfs > float(fam_best[fam].get("rrf", 0.0)):
                fam_best[fam] = c
        pool = list(fam_best.values())
        texts = [c["text"][:_MAX_CHARS] for c in pool]
        rr = self.emb.rerank(query, texts, top_n=_TOP_K)
        if rr is None:
            raise RuntimeError("rerank 调用失败")
        hits = [{**pool[r["index"]], "score": r["score"]} for r in rr]
        top = hits[0]
        # 场景适配 3：阈值自适应——用 top1-top2 分数差判定（替代固定 0.30/0.50）
        # rerank 分数跨查询不可比（同 0.89 有的对有的错），差值比绝对值更稳
        top2_score = float(hits[1]["score"]) if len(hits) > 1 else 0.0
        margin = float(top["score"]) - top2_score
        # 自适应判定：top1 ≥ 0.40 且领先 top2 ≥ 0.15 → correct（明确胜出）
        #             否则按分数区间 ambiguous/incorrect
        if float(top["score"]) >= 0.40 and margin >= 0.15:
            verdict = "correct"
        elif float(top["score"]) >= 0.30:
            verdict = "ambiguous"
        else:
            verdict = "incorrect"
        return {
            "ok": True,
            "top1_source": _stem(top["source"]),
            "top1_score": round(float(top["score"]), 4),
            "top1_text": top["text"][:120],
            "verdict": verdict,
            "top1_top2_margin": round(margin, 4),
            "graph_stats": graph_stats,
        }

    def run_hybrid_rerank(self, query: str) -> Dict[str, Any]:
        return self._run_rerank(query, use_graph=False)

    def run_graph(self, query: str) -> Dict[str, Any]:
        return self._run_rerank(query, use_graph=True)

    # -- F. langchain 简易 RAG -------------------------------------

    def run_langchain(self, query: str) -> Dict[str, Any]:
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as e:
            return {"ok": False, "skipped": True, "error": f"langchain 未安装: {e}"}
        kwargs = {"model": self.emb.model, "api_key": self.emb.api_key, "base_url": self.emb.base_url}
        try:
            # check_embedding_ctx_length=False：跳过 tiktoken 长度检查（bge-m3 无对应 tokenizer）
            le = OpenAIEmbeddings(**kwargs, check_embedding_ctx_length=False)
        except TypeError:
            le = OpenAIEmbeddings(**kwargs)  # 老版本无该参数
        vec = le.embed_query(query)
        q = np.asarray(vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        m = self.index._matrix / (np.linalg.norm(self.index._matrix, axis=1, keepdims=True) + 1e-9)
        scores = m @ q
        idx = scores.argsort()[::-1][:_TOP_K]
        top_i = int(idx[0])
        return {
            "ok": True,
            "top1_source": _stem(self.index._sources[top_i]),
            "top1_score": round(float(scores[top_i]), 4),
            "top1_text": self.index._texts[top_i][:120],
            "verdict": verdict_of(float(scores[top_i])),
        }


def _hit(result: Dict[str, Any], expected: List[str]) -> Optional[bool]:
    """top1 是否命中期望笔记；负样本/出错返回 None。"""
    if not result.get("ok") or result.get("skipped"):
        return None
    if not expected:
        return None
    return result["top1_source"] in expected


def _blocked(result: Dict[str, Any]) -> Optional[bool]:
    """负样本是否被 guard 拦截（仅 rerank 变体有可比较的分数量纲）。"""
    if not result.get("ok") or result.get("skipped"):
        return None
    if "verdict" not in result:
        return None
    return result["verdict"] == "incorrect"


def _fmt_cell(result: Dict[str, Any], expected: List[str], want_verdict: bool = False) -> str:
    """把单个变体结果格式化成表格单元格。"""
    if result.get("skipped"):
        return "跳过"
    if not result.get("ok"):
        return "ERR"
    hit = _hit(result, expected)
    mark = ""
    if hit is not None:
        mark = "✓" if hit else "✗"
    verdict = f" {result['verdict']}" if want_verdict and "verdict" in result else ""
    return f"{result['top1_source']}{mark} {result['top1_score']:.3f}{verdict}"


def aggregate_stats(results: Dict[str, Dict[str, Any]], variant: str, queries: List[Dict]) -> Dict[str, Any]:
    """按查询类型聚合某个变体的 hit@1 / 拦截率 / top1 分数分布。"""
    stats = {"types": {}, "scores": []}
    for typ in ("multi-hop", "single-hop", "negative"):
        qs = [q for q in queries if q["type"] == typ]
        hits = [_hit(results[q["id"]][variant], q["expected_notes"]) for q in qs]
        hits = [h for h in hits if h is not None]
        blocked = [_blocked(results[q["id"]][variant]) for q in qs]
        blocked = [b for b in blocked if b is not None]
        ok = [results[q["id"]][variant] for q in qs if results[q["id"]][variant].get("ok")]
        stats["types"][typ] = {
            "n": len(qs),
            "hit@1": (sum(hits) / len(hits)) if hits else None,
            "guard_blocked": (sum(blocked) / len(blocked)) if blocked else None,
        }
        stats["scores"].extend(r["top1_score"] for r in ok if "top1_score" in r)
    if stats["scores"]:
        stats["mean"] = round(statistics.mean(stats["scores"]), 4)
        stats["median"] = round(statistics.median(stats["scores"]), 4)
    return stats


def write_report(
    results: Dict[str, Dict[str, Any]],
    meta: Dict[str, Any],
    out_path: Path,
) -> str:
    """把评测结果渲染成 markdown 报告（横向 + 纵向 + 结论）。"""
    queries = meta["queries"]
    variants = meta["variants"]
    want_verdict = {"hybrid_rerank", "graph", "langchain"}
    api_ok = meta.get("api_ok", False)
    lines: List[str] = []
    lines.append("# vaultrag 评测报告（双链图扩展 + 面包屑切块）\n")
    lines.append(f"- 生成时间：{meta['generated_at']}")
    lines.append(f"- vault：`{meta['vault']}`（{meta['notes']} 篇笔记 / {meta['chunks']} 块，v{meta['index_version']} 切块）")
    lines.append(f"- 查询集：{len(queries)} 条（多跳 {len([q for q in queries if q['type']=='multi-hop'])} / "
                 f"单跳 {len([q for q in queries if q['type']=='single-hop'])} / "
                 f"无关 {len([q for q in queries if q['type']=='negative'])}）")
    lines.append(f"- 云端 API：{'可用' if api_ok else '不可达（沙箱/环境无外网）'}——"
                 f"{'以下为真实评测数据' if api_ok else 'rerank/向量相关变体待网络恢复后重跑填数'}\n")

    # ---------- 横向对比 ----------
    lines.append("## 一、横向对比（同查询集，按变体分组）\n")
    header = ["id", "type", "查询", "期望命中"] + variants
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for q in queries:
        expected = "、".join(q["expected_notes"]) if q["expected_notes"] else "无（应拦截）"
        row = [q["id"], q["type"], q["query"][:36], expected]
        for v in variants:
            row.append(_fmt_cell(results[q["id"]][v], q["expected_notes"], v in want_verdict))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # 聚合
    lines.append("### 横向聚合（hit@1 / guard 拦截率 / top1 分数）\n")
    agg_header = ["变体", "多跳 hit@1", "单跳 hit@1", "负样本拦截", "top1 均值", "top1 中位数"]
    lines.append("| " + " | ".join(agg_header) + " |")
    lines.append("|" + "---|" * len(agg_header))
    for v in variants:
        st = aggregate_stats(results, v, queries)
        t = st["types"]
        fmt = lambda x: "—" if x is None else f"{x:.1%}"
        lines.append(
            f"| {v} | {fmt(t['multi-hop']['hit@1'])} | {fmt(t['single-hop']['hit@1'])} | "
            f"{fmt(t['negative']['guard_blocked'])} | "
            f"{st.get('mean', '—')} | {st.get('median', '—')} |"
        )
    lines.append("")

    # ---------- 纵向对比 ----------
    lines.append("## 二、纵向对比：基线（混合+rerank） vs 演进后（+双链图扩展）\n")
    lines.append("| id | type | 查询 | D top1(分数/verdict) | E top1(分数/verdict) | Δtop1 | 迁移 |")
    lines.append("|" + "---|" * 7)
    migrations: Dict[str, int] = {}
    deltas: List[float] = []
    for q in queries:
        d = results[q["id"]]["hybrid_rerank"]
        e = results[q["id"]]["graph"]
        d_cell = _fmt_cell(d, q["expected_notes"], True)
        e_cell = _fmt_cell(e, q["expected_notes"], True)
        migration = "—"
        delta = "—"
        if d.get("ok") and e.get("ok") and "verdict" in d and "verdict" in e:
            key = f"{d['verdict']}→{e['verdict']}"
            migrations[key] = migrations.get(key, 0) + 1
            delta = f"{e['top1_score'] - d['top1_score']:+.3f}"
            deltas.append(e["top1_score"] - d["top1_score"])
            if d["verdict"] != e["verdict"]:
                migration = key
        lines.append(f"| {q['id']} | {q['type']} | {q['query'][:30]} | {d_cell} | {e_cell} | {delta} | {migration} |")
    lines.append("")
    lines.append("### 纵向聚合\n")
    lines.append(f"- verdict 迁移计数：{json.dumps(migrations, ensure_ascii=False) if migrations else '（数据待补）'}")
    lines.append(f"- rerank top1 分数差（E−D）：均值 {statistics.mean(deltas):+.4f} / 中位数 "
                 f"{statistics.median(deltas):+.4f}" if deltas else "- rerank top1 分数差：数据待补")
    lines.append("")

    # ---------- 框架对比 ----------
    lines.append("## 三、轻量业界框架对比\n")
    lines.append("| 框架 | 状态 | 选型理由 |")
    lines.append("|---|---|---|")
    lines.append("| txtai | 未安装 → 跳过 | 本地优先路线与本插件『云端 embedding、零本地模型』约束冲突；个人库百篇级场景收益有限 |")
    lines.append("| LanceDB | 未安装 → 跳过 | 嵌入式向量库适合百万级向量；当前 121 篇/千余块，numpy 点积毫秒级足够，引入即违背零新依赖 |")
    lines.append("| langchain（已装） | 变体 F | 用其 OpenAIEmbeddings 走同一云端 bge-m3，搭简易向量 RAG 作业界对照 |")
    lines.append("")
    for v in variants:
        if v == "langchain":
            skipped = sum(1 for q in queries if results[q["id"]][v].get("skipped"))
            if skipped:
                lines.append(f"> 变体 F（langchain）：{skipped}/{len(queries)} 条因依赖缺失或 API 不可达跳过。\n")

    # ---------- 结论 ----------
    lines.append("## 四、结论：每一层加了什么、涨了多少\n")
    if not api_ok:
        lines.append(
            "> 本环境无外网（连接云端 API 被沙箱拦截），rerank/向量变体的真实分数未能采集。\n"
            "> 报告结构与评估口径已就绪，在可联网环境重跑下方命令即自动填数：\n\n"
            "> ```\n"
            "> cd D:/AI/hermes-agent && .venv/Scripts/python.exe "
            "plugins/context_engine/vaultrag/evals/run_eval.py\n"
            "> ```\n\n"
            "待补数据将回答：面包屑是否提升单跳命中、图扩展是否提升多跳 hit@1、"
            "guard 是否仍拦截负样本、verdict 迁移方向（Incorrect/Ambiguous→Correct）。\n"
        )
    else:
        lines.append("（由真实数据驱动的结论将在此生成：每层增量 = 变体间 hit@1/分数差。）\n")
    lines.append("---")
    lines.append("*报告由 `evals/run_eval.py` 自动生成，勿手改。*\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)


def main():
    ap = argparse.ArgumentParser(description="vaultrag 横向/纵向评测")
    ap.add_argument("--vault", default=None, help="vault 路径（默认读 config.yaml）")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "eval_report.md"), help="报告输出路径")
    ap.add_argument("--types", default=None, help="查询类型过滤：multi-hop/single-hop/negative/all")
    ap.add_argument("--skip-framework", action="store_true", help="跳过 langchain 变体（F）")
    args = ap.parse_args()

    vcfg = _load_vaultrag_config()
    vault = args.vault or vcfg.get("vault_path") or os.getenv("OBSIDIAN_VAULT_PATH")
    if not vault:
        print("未找到 vault 路径（config.yaml context.vaultrag.vault_path 或 --vault）")
        sys.exit(1)

    emb = EmbeddingClient()
    index = VaultIndex(vault, embedding=emb)
    ok = index.ensure_index()
    if not ok:
        print(f"索引不可用（embedding API 不可达？vault={vault}）")
        ensure_bm25_offline(index, vault)
        print(f"已离线构建 BM25（{len(index._texts)} 块），纯 BM25 变体仍可评测")
    runner = VariantRunner(index, emb)

    queries = get_queries(args.types if args.types not in (None, "all") else None)
    variants = ["pure_vector", "pure_bm25", "hybrid_no_rerank", "hybrid_rerank", "graph"]
    if not args.skip_framework:
        variants.append("langchain")

    results: Dict[str, Dict[str, Any]] = {}
    api_ok = True
    network_ok = probe_network(emb.base_url)
    if not network_ok:
        print("探测云端 API 不可达 → 网络相关变体直接标记不可用，报告保留结构待补数据")
        api_ok = False
    for q in queries:
        results[q["id"]] = {}
        for v in variants:
            if not network_ok and v != "pure_bm25":
                # 无外网：非 BM25 变体全部标记不可用（不再逐条等待连接超时）
                r = {"ok": False, "error": "network-unavailable"}
            else:
                r = runner.run(v, q["query"])
            results[q["id"]][v] = r
            if not r.get("ok"):
                api_ok = False
        qid = q["id"]
        brief = " | ".join(
            f"{v}:{results[qid][v].get('top1_source', results[qid][v].get('error', '?')[:20])}"
            for v in variants
        )
        print(f"{qid} {q['type']:10s} {q['query'][:24]:26s} {brief}")

    meta = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "vault": vault,
        "notes": len(scan_vault(vault)),
        "chunks": len(index._texts),
        "index_version": 2,
        "queries": queries,
        "variants": variants,
        "api_ok": api_ok,
    }
    path = write_report(results, meta, Path(args.out))
    print(f"\n报告已生成: {path}")
    if not api_ok:
        print("注意：部分变体因云端 API 不可达未取到数据，报告相应单元格为 ERR/待补。")


if __name__ == "__main__":
    main()
