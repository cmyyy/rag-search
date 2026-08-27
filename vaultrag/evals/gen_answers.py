"""为 golden cases 生成"人写答案"（expected_answer）——C 方案（2026-08-27）。

背景：golden cases 原本只有「查询 → 期望命中笔记」标注；L4 语义评测把笔记
前 800 字符当 expected_output，是设计将就（拆句多、输出长、失败率高、口径偏
"笔记内容覆盖"而非"答案覆盖"）。本脚本为每条正样本查询生成简洁答案
（3-5 句，完全基于笔记内容），存 answers.json，供 judge_semantic 使用。

用法：python evals/gen_answers.py   （生成 answers.json，含 85 条正样本答案）
"""
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))   # plugins/rag-search（vaultrag 包）
sys.path.insert(0, str(_HERE.parent))          # vaultrag 根
sys.path.insert(0, str(_HERE))                 # evals/
try:
    from dotenv import load_dotenv
    load_dotenv(r"D:\AI\hermes-agent\.env", override=True)
except Exception:
    pass

from eval_queries import get_queries

OUT = _HERE / "answers.json"

SYSTEM_PROMPT = (
    "你是知识库问答助手。基于给定的笔记内容，用 3-5 句简洁的中文回答用户的问题。"
    "要求：1) 答案完全基于笔记内容，不编造、不扩展、不联想；"
    "2) 直接给出答案本身，不要写'根据笔记'、'笔记中提到'之类的引导语；"
    "3) 简明扼要，3-5 句话。"
)


def read_note(vault_root: str, stem: str, limit: int = 2500) -> str:
    for md in Path(vault_root).rglob("*.md"):
        if md.stem == stem:
            return md.read_text(encoding="utf-8", errors="ignore")[:limit]
    return ""


def generate_answer(client, query: str, note_text: str) -> str:
    user_prompt = f"用户的问题：{query}\n\n笔记内容：\n{note_text}\n\n请给出答案："
    r = client.chat.completions.create(
        model=os.environ.get("DEEPSEEK_JUDGE_MODEL", "deepseek-v4-flash"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=600,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return (r.choices[0].message.content or "").strip()


def main():
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com/v1",
        timeout=60,
    )
    engine = __import__("vaultrag").VaultRAGEngine()
    vault_root = engine.vault_root

    queries = [q for q in get_queries(None) if q["type"] != "negative"]
    print(f"正样本 {len(queries)} 条，开始生成答案（DeepSeek v4-flash，thinking 关）...")

    answers = {}
    fails = 0
    for i, q in enumerate(queries, 1):
        note = read_note(vault_root, q["expected_notes"][0])
        if not note:
            print(f"  [{i}] 笔记未找到: {q['expected_notes'][0]}（跳过）")
            fails += 1
            continue
        for attempt in range(3):
            try:
                ans = generate_answer(client, q["query"], note)
                if ans:
                    answers[q["id"]] = {"query": q["query"], "answer": ans}
                    break
            except Exception as e:
                print(f"  [{i}] 生成失败（{str(e)[:50]}），重试 {attempt + 1}")
                time.sleep(2 * (attempt + 1))
        else:
            print(f"  [{i}] 生成失败: {q['id']}")
            fails += 1
        if i % 10 == 0:
            print(f"  进度 {i}/{len(queries)}")
        time.sleep(0.2)

    OUT.write_text(json.dumps(answers, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"完成：{len(answers)}/{len(queries)} 条，失败 {fails} → {OUT}")


if __name__ == "__main__":
    main()
