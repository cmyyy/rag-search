import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(r"D:\AI\hermes-agent\.env", override=True)
from openai import AsyncOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM


_HERE = Path(__file__).resolve().parent
sys.path.insert(0, r"D:\AI\hermes-agent")
sys.path.insert(0, r"D:\AI\hermes-agent\plugins\rag-search")
sys.path.insert(0, str(_HERE.parent))

from vaultrag import VaultRAGEngine, _TOP_K, truncate_hit
from vaultrag.retriever import VaultIndex
from evals.eval_queries import get_queries


class AsyncDeepSeekLLM(DeepEvalBaseLLM):
    """真异步 judge（消融专用，2026-08-30）：AsyncOpenAI + 重试——run_async=True 真并行。

    study_eval 的 DeepSeekLLM.a_generate 是同步包装（伪异步），并行不加速。
    """
    def __init__(self):
        self._model = os.environ.get("DEEPSEEK_JUDGE_MODEL", "deepseek-v4-flash")
        self._client = AsyncOpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            timeout=60,
        )

    def load_model(self):
        return self._client

    def generate(self, prompt: str, **kwargs) -> str:
        """同步兜底（run_async=True 时 DeepEval 只调 a_generate；此处防误用）。"""
        import asyncio
        return asyncio.run(self.a_generate(prompt, **kwargs))

    async def a_generate(self, prompt: str, **kwargs) -> str:
        import asyncio
        # DeepEval judge 需要严格 JSON（对齐 study_eval DeepSeekLLM）：
        # 强制 json_object；提示无 json 字样被拒时回退普通模式
        try:
            r = await self._client.chat.completions.create(
                model=self._model, messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=8192,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}}, **kwargs)
            content = r.choices[0].message.content
            if content:
                return content
        except Exception:
            pass
        last_err = None
        for attempt, delay in ((1, 1.5), (2, 3), (3, 6)):
            try:
                r = await self._client.chat.completions.create(
                    model=self._model, messages=[{"role": "user", "content": prompt}],
                    temperature=0.0, max_tokens=8192,
                    extra_body={"thinking": {"type": "disabled"}}, **kwargs)
                content = r.choices[0].message.content
                if content:
                    return content
                last_err = RuntimeError("empty content")
            except Exception as e:
                last_err = e
                await asyncio.sleep(delay)
        raise last_err

    def get_model_name(self) -> str:
        return self._model


def run_l4(with_breadcrumb: bool) -> dict:
    from deepeval import evaluate
    from deepeval.evaluate.configs import AsyncConfig, ErrorConfig
    from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
    from deepeval.test_case import LLMTestCase

    engine = VaultRAGEngine()
    idx = VaultIndex(engine.vault_root, embedding=engine.embedding)
    ok = idx.ensure_index(force=True, with_breadcrumb=with_breadcrumb)
    if not ok:
        return {"error": "index build failed"}
    engine.index = idx
    engine._index_ready = True

    answers = json.loads((_HERE / "answers.json").read_text(encoding="utf-8"))
    queries = [q for q in get_queries() if q["type"] != "negative"]

    judge = AsyncDeepSeekLLM()
    metric_prec = ContextualPrecisionMetric(threshold=0.5, model=judge, verbose_mode=False)
    metric_rec = ContextualRecallMetric(threshold=0.5, model=judge, verbose_mode=False)

    test_cases = []
    for q in queries:
        sr = engine.search(q["query"], top_k=_TOP_K)
        if sr is None or sr["verdict"] != "correct" or not sr.get("hits"):
            continue
        ctx = [truncate_hit(h["text"]) for h in sr["hits"]]
        ans = answers.get(q["id"], {}).get("answer", "")
        if not ans or not ctx:
            continue
        test_cases.append(LLMTestCase(
            input=q["query"], actual_output=ctx[0][:300],
            expected_output=ans, retrieval_context=ctx,
        ))

    print(f"  [{'面包屑+标签' if with_breadcrumb else '纯文本'}] L4 用例 {len(test_cases)} 条", flush=True)
    if not test_cases:
        return {"n": 0}
    results = evaluate(test_cases=test_cases, metrics=[metric_prec, metric_rec],
                       async_config=AsyncConfig(run_async=True),
                       error_config=ErrorConfig(ignore_errors=True, skip_on_missing_params=True))
    # 聚合全部 case 的 metrics（2026-08-30 修复：test_results 是逐 case 列表，
    # 只取 [0] 会把第一个 case 的分数当整组均值——之前"全 1.0"就是这个 bug）
    precs, recs, fails = [], [], 0
    for tr in results.test_results:
        for m in tr.metrics_data:
            if m.name == "Contextual Precision":
                if m.score is not None:
                    precs.append(m.score)
                else:
                    fails += 1
            elif m.name == "Contextual Recall":
                if m.score is not None:
                    recs.append(m.score)
                else:
                    fails += 1
    import statistics
    out = {"n": len(test_cases), "precision": statistics.mean(precs) if precs else None,
           "recall": statistics.mean(recs) if recs else None, "judge_fail": fails}
    print(f"  [{'面包屑+标签' if with_breadcrumb else '纯文本'}] Precision={out['precision']:.3f} Recall={out['recall']:.3f} (n={out['n']}, judge_fail={fails})", flush=True)
    return out


if __name__ == "__main__":
    a = run_l4(True)
    b = run_l4(False)
    print(f"\nL4 对比: Precision {a.get('precision'):.3f} vs {b.get('precision'):.3f} | "
          f"Recall {a.get('recall'):.3f} vs {b.get('recall'):.3f}")
