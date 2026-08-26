"""verify_guard_upgrade.py —— vaultrag "生产级检索 guard"升级验证（2026-08-19）。

跑法（在 hermes-agent 目录下）：
    .venv/Scripts/python.exe plugins/rag-search/vaultrag/verify_guard_upgrade.py

纯离线验证（不碰网络，不调真实 embedding/rerank API）——用伪造的
EmbeddingClient + VaultIndex 驱动 select_context 管线，逐项断言 4 项改动：

  [1] 改动1：长度门槛 4 → 2（"MoA" 3 字符不再被拦；"好" 1 字仍被拦）
  [2] 改动2：CRAG 三档评估（correct/ambiguous/incorrect 判定边界）
  [3] 改动3：结构化 trace JSONL（每次调用落盘、字段齐全、fail-open）

断言机制：check() 失败记入 fails，最后 sys.exit(1 if fails else 0)。
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# 让插件包可导入：脚本在插件目录里，往上找 hermes-agent 根
_HERMES_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_HERMES_ROOT))

# 自动加载 .env（override=True 覆盖 MSYS 旧值）——与 verify_pipeline.py 一致
from dotenv import load_dotenv  # noqa: E402

load_dotenv(_HERMES_ROOT / ".env", override=True)

from plugins.context_engine.vaultrag import VaultRAGEngine  # noqa: E402
from plugins.context_engine.vaultrag import _MIN_QUERY_CHARS, _MIN_SCORE, _CORRECT_SCORE  # noqa: E402

fails = []


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {extra}")
    if not cond:
        fails.append(name)


class FakeEmbedding:
    """伪造 EmbeddingClient：可控 rerank 分数，不碰网络。"""

    def __init__(self, rerank_scores=None):
        self.rerank_scores = list(rerank_scores) if rerank_scores else [0.9, 0.8, 0.7, 0.6]
        self.embed_called = 0
        self.rerank_called = 0
        self.api_key = "sk-test"
        self.base_url = "https://fake.invalid/v1"

    @property
    def available(self):
        return True

    def embed_query(self, q):
        self.embed_called += 1
        return np.array([1.0, 0.0], dtype=np.float32)

    def rerank(self, query, documents, top_n=4):
        self.rerank_called += 1
        return [
            {"index": i, "score": float(s)}
            for i, s in enumerate(self.rerank_scores[:top_n])
        ]


class FakeIndex:
    """伪造 VaultIndex：固定召回 4 条候选，不碰真实 vault/网络。"""

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def ensure_index(self, force=False):
        return True

    def hybrid_search(self, query, query_vec, top_k=16):
        return [
            {"text": f"doc{i}", "source": f"note{i}.md", "score": 0.9 - i * 0.1, "rrf": 0.0}
            for i in range(4)
        ]


# 共享临时目录：所有引擎的 trace 都落到同一个 trace.jsonl，方便汇总断言
SHARED_TMP = Path(tempfile.mkdtemp(prefix="vaultrag_guard_"))
TRACE_DIR = SHARED_TMP / ".smart-env" / "vaultrag"
TRACE_FILE = TRACE_DIR / "trace.jsonl"


def make_engine(rerank_scores=None):
    emb = FakeEmbedding(rerank_scores=rerank_scores)
    engine = VaultRAGEngine(vault_root=str(SHARED_TMP), embedding=emb)
    engine.index = FakeIndex(TRACE_DIR)  # 替换真实 VaultIndex（离线）
    engine._index_ready = True           # 跳过真实索引构建
    return engine, emb


def run_query(engine, q):
    return engine.select_context(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": q}],
        incoming_message={"role": "user", "content": q},
    )


def read_traces():
    if not TRACE_FILE.exists():
        return []
    return [json.loads(line) for line in TRACE_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def last_trace():
    traces = read_traces()
    return traces[-1] if traces else None


def main():
    print("[0] 常量")
    check("改动1: _MIN_QUERY_CHARS == 2", _MIN_QUERY_CHARS == 2, f"(={_MIN_QUERY_CHARS})")
    check("改动2: _MIN_SCORE == 0.30", _MIN_SCORE == 0.30, f"(={_MIN_SCORE})")
    check("改动2: _CORRECT_SCORE == 0.50", _CORRECT_SCORE == 0.50, f"(={_CORRECT_SCORE})")

    print("\n[1] 改动1: 长度门槛 4 → 2")
    # 3 字符英文缩写 "MoA" 必须不被拦
    e1, emb1 = make_engine(rerank_scores=[0.9, 0.8, 0.7, 0.6])
    r = run_query(e1, "MoA")
    check("'MoA'(3字符) 不被长度门槛拦截 → 注入成功", r is not None and "<knowledge_context>" in r[-1]["content"])
    check("'MoA' 触发了检索(embed/rerank 被调用)", emb1.embed_called == 1 and emb1.rerank_called == 1)
    # 1 字确认消息 "好" 仍被拦
    e2, emb2 = make_engine(rerank_scores=[0.9, 0.8, 0.7, 0.6])
    r2 = run_query(e2, "好")
    check("'好'(1字符) 仍被长度门槛拦截 → 不注入", r2 is None)
    check("'好' 未触发检索(embed 未被调用)", emb2.embed_called == 0)

    print("\n[2] 改动2: CRAG 三档评估")
    # Correct: top1 ≥ 0.5
    ec, _ = make_engine(rerank_scores=[0.85, 0.6, 0.55, 0.5])
    rc = run_query(ec, "Correct 查询")
    check("Correct(≥0.5) 注入成功", rc is not None and "<knowledge_context>" in rc[-1]["content"])
    check("Correct verdict 记录为 correct", last_trace()["verdict"] == "correct", f"(={last_trace()['verdict']})")
    # Ambiguous: 0.3 ≤ top1 < 0.5
    ea, _ = make_engine(rerank_scores=[0.42, 0.35, 0.31, 0.30])
    ra = run_query(ea, "Ambiguous 查询")
    check("Ambiguous(0.3~0.5) 仍注入(低置信)", ra is not None and "<knowledge_context>" in ra[-1]["content"])
    check("Ambiguous verdict 记录为 ambiguous", last_trace()["verdict"] == "ambiguous", f"(={last_trace()['verdict']})")
    # Incorrect: top1 < 0.3
    ei, _ = make_engine(rerank_scores=[0.12, 0.08, 0.05, 0.03])
    ri = run_query(ei, "Incorrect 查询")
    check("Incorrect(<0.3) 不注入", ri is None)
    t = last_trace()
    check("Incorrect verdict 记录为 incorrect", t["verdict"] == "incorrect", f"(={t['verdict']})")
    check("Incorrect reason = score-below-threshold", t["reason"] == "score-below-threshold", f"(={t['reason']})")
    # 注入消息只保留 role+content 两键（严格 provider 兼容）
    if ra is not None:
        check("注入消息只含 role+content 两键", set(ra[-1].keys()) == {"role", "content"}, f"keys={list(ra[-1].keys())}")

    print("\n[3] 改动3: 结构化 trace (RAGOps)")
    traces = read_traces()
    check("trace.jsonl 已生成", TRACE_FILE.exists() and len(traces) > 0, f"(共 {len(traces)} 条)")
    required = {"timestamp", "query", "length_gate_pass", "recalled", "rerank_top_scores", "verdict", "injected_count", "reason"}
    all_ok = all(required <= set(t.keys()) for t in traces)
    check("每条 trace 字段齐全(8 个必需键)", all_ok)
    check("每条 trace 都是合法 JSON", len(traces) == len(read_traces()))
    # 具体条目断言
    verdicts = [t["verdict"] for t in traces]
    check("trace 覆盖 correct/ambiguous/incorrect/skipped", {"correct", "ambiguous", "incorrect", "skipped"} <= set(verdicts), f"={set(verdicts)}")
    length_gate_trace = next((t for t in traces if t["reason"] == "length-gate"), None)
    check("length-gate 条目: verdict=skipped", length_gate_trace is not None and length_gate_trace["verdict"] == "skipped")
    check("length-gate 条目: length_gate_pass=false", length_gate_trace is not None and length_gate_trace["length_gate_pass"] is False)
    inj_trace = next((t for t in traces if t["injected_count"] > 0), None)
    check("注入条目: injected_count>0 且 rerank_top_scores 非空", inj_trace is not None and inj_trace["injected_count"] > 0 and len(inj_trace["rerank_top_scores"]) > 0)
    check("trace 含时间戳", all(t["timestamp"] for t in traces))
    # fail-open：把 trace 目录变成不可写场景，检索不应崩
    e3, _ = make_engine(rerank_scores=[0.9, 0.8, 0.7, 0.6])
    e3.index.cache_dir = Path(SHARED_TMP) / "no_such_dir" / "deep" / "readonly"  # 父目录不存在 → 写失败
    r3 = run_query(e3, "trace 写失败也应注入")
    check("trace 写失败不影响检索(fail-open)", r3 is not None and "<knowledge_context>" in r3[-1]["content"])

    print("\n结果: 全部通过" if not fails else f"\n结果: {len(fails)} 失败 -> {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
