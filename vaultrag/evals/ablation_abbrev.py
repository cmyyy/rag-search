"""缩写展开消融（2026-08-26）：query 原样 vs 展开，abbreviation 类查询对比。

回答：若 agent 传裸缩写（不自行展开），引擎的缩写展开机制是否必要。
  A = query 原样（模拟 agent 传裸缩写，引擎不展开）
  B = query 展开（引擎 _expand_abbrev 后的查询）
对比 hit@1（top1 命中 expected_notes）+ guard verdict 分布。

词表快照来源：vaultrag/__init__.py 的 _ABBREV_EXPANSIONS（改引擎时同步）。
"""
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # vaultrag 根（embedding/retriever）
sys.path.insert(0, str(_HERE))         # evals/（eval_queries/run_eval）

try:
    from dotenv import load_dotenv

    load_dotenv(r"D:\AI\hermes-agent\.env", override=True)
except Exception:
    pass

from embedding import EmbeddingClient
from retriever import VaultIndex
from eval_queries import get_queries
from run_eval import VariantRunner, _hit, _load_vaultrag_config

_ABBREV_EXPANSIONS = {
    "MCP": "Model Context Protocol",
    "TTS": "Text-to-Speech 语音合成",
    "AOP": "Aspect Oriented Programming 面向切面编程",
    "CRAG": "Corrective RAG 纠正式检索增强",
    "HyDE": "Hypothetical Document Embeddings 假设文档嵌入",
    "RRF": "Reciprocal Rank Fusion 倒数排名融合",
    "MoA": "Mixture of Agents 多智能体协作",
    "MoE": "Mixture of Experts 专家混合",
    "BM25": "BM25 词法检索 关键词匹配",
    "KV": "Key-Value 键值",
    "OAuth": "OAuth 认证协议",
    "LLM": "Large Language Model 大语言模型",
    "RAG": "Retrieval Augmented Generation 检索增强生成",
    "SQL": "Structured Query Language 结构化查询语言",
    "TCC": "Try Confirm Cancel 分布式事务",
    "Saga": "Saga 分布式事务长事务",
}


def expand_query(query: str) -> str:
    if not query:
        return query
    expanded = []
    for abbr, full in _ABBREV_EXPANSIONS.items():
        if re.search(rf"(?<![A-Za-z]){re.escape(abbr)}(?![A-Za-z])", query):
            expanded.append(f"{abbr}({full})")
    if not expanded:
        return query
    return query + " " + " ".join(expanded)


def main() -> None:
    vcfg = _load_vaultrag_config()
    vault = vcfg.get("vault_path") or "D:/llmwiki/llm-wiki"
    emb = EmbeddingClient()
    index = VaultIndex(vault, embedding=emb)
    if not index.ensure_index():
        print("索引不可用"); return
    runner = VariantRunner(index, emb)
    queries = get_queries("abbreviation")
    print(f"abbreviation 类查询: {len(queries)} 条\n")
    rows = []
    for q in queries:
        qid, qtext = q["id"], q["query"]
        expected = q.get("expected_notes", [])
        ra = runner.run_hybrid_rerank(qtext)
        rb = runner.run_hybrid_rerank(expand_query(qtext))
        rows.append((qid, qtext, ra, rb, expected))

    # 汇总
    hit_a = sum(1 for _, _, ra, _, ex in rows if _hit(ra, ex) is True)
    hit_b = sum(1 for _, _, _, rb, ex in rows if _hit(rb, ex) is True)
    miss_a = sum(1 for _, _, ra, _, ex in rows if _hit(ra, ex) is False)
    miss_b = sum(1 for _, _, _, rb, ex in rows if _hit(rb, ex) is False)
    guard_a = sum(1 for _, _, ra, _, _ in rows if ra.get("verdict") == "incorrect")
    guard_b = sum(1 for _, _, _, rb, _ in rows if rb.get("verdict") == "incorrect")
    amb_a = sum(1 for _, _, ra, _, _ in rows if ra.get("verdict") == "ambiguous")
    amb_b = sum(1 for _, _, _, rb, _ in rows if rb.get("verdict") == "ambiguous")

    print("=" * 90)
    print(f"{'id':<8}{'A 原样 hit/verdict':<28}{'B 展开 hit/verdict':<28}差异")
    print("=" * 90)
    for qid, qtext, ra, rb, ex in rows:
        ha, hb = _hit(ra, ex), _hit(rb, ex)
        diff = (ha != hb) or (ra.get("verdict") != rb.get("verdict"))
        mark = " <<< 差异" if diff else ""
        print(f"{qid:<8}{str(ha):<8}{ra.get('verdict'):<18}{str(hb):<8}{rb.get('verdict'):<18}{mark}")
    print("=" * 90)
    print(f"\n汇总（{len(rows)} 条 abbreviation 查询）:")
    print(f"  hit@1 : A(原样)={hit_a}  B(展开)={hit_b}   (+{hit_b - hit_a})")
    print(f"  miss  : A={miss_a}  B={miss_b}")
    print(f"  guard incorrect 拦截: A={guard_a}  B={guard_b}")
    print(f"  ambiguous        : A={amb_a}  B={amb_b}")


if __name__ == "__main__":
    main()
