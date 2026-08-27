"""vaultrag — Vault RAG context engine（个人知识库检索注入）。

一句话：把 Obsidian vault 变成 Hermes 的"外接知识库"——用户提问时，
先向量检索相关笔记片段，注入请求上下文，模型基于自己的笔记回答。

工作方式（挂在 Hermes 的 context engine 钩子上）：
  1. 配置 context.engine = vaultrag（config.yaml）
  2. 每回合请求前，Hermes 调用本引擎的 select_context(api_messages, ...)
  3. 我们：取 incoming_message（用户本轮问题）→ 向量化 → vault 检索 top-k
     → 把命中片段拼成 "Knowledge context" 消息插进请求列表 → 返回新列表
  4. 任何失败（无 key / 索引不可用 / 异常）→ 返回 None → Hermes 原样放行
     （fail-open，绝不打断正常对话）

与内置 compressor 的关系：
  - 本引擎是"选择"上下文（select_context），不是"压缩"上下文（compress）
  - 压缩职责委托给内置 ContextCompressor：config 里 compression.* 的
    阈值/策略全部继续生效，本引擎不重复实现
"""
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine

from .embedding import EmbeddingClient
from .retriever import VaultIndex, scan_vault

logger = logging.getLogger(__name__)

# 注入的消息角色：system 前缀保持字节稳定（prompt cache），
# 检索结果放独立 user 消息——不碰 system、不改历史（cache 契约）。
_INJECT_ROLE = "user"
_TOP_K = 8
_MAX_CHARS_PER_HIT = 600
# 查询质量门槛（防噪音注入，2026-08-16 实测"好"/"改"等短消息命中全无关）：
# ⚠ 2026-08-19 修复 bug：门槛从 4 降到 2——"MoA"/"RAG" 这类 3 字符英文缩写实查询
#   被 len() 按字符数误杀（英文 1 字母 = 1 字符）。降到 2 只挡"好/行/改/继续"这类
#   1 字确认消息（中文单字确认 < 2 才被拦）。
_MIN_QUERY_CHARS = 2   # 查询少于 2 个字符直接跳过（1 字确认消息）
# CRAG 式三档评估阈值（复用 rerank 的 top1 分数，零额外成本，2026-08-19）：
# top1 分数阈值 → Incorrect，不注入。
# 2026-08-23 评测驱动修正 0.30 → 0.02：全量 84 正样本 + 15 负样本 rerank top1 分布实测——
#   负样本 max = 0.015（15 条全部 ≤ 0.015），正样本中位 0.563、p10 0.076；
#   0.30 阈值误杀 33% 正样本（28/84，其中 6 条 top1 就是正确笔记但分数 0.02-0.21），
#   0.02 阈值误杀仅 5/84（全是检索本身失败，放行也无益）且误放负样本 0/15。
# 0.02 由负样本分布上界 0.015 决定（自然分隔点），非对正样本拟合。
_MIN_SCORE = 0.02
_CORRECT_SCORE = 0.50  # top1 ≥ 0.50 → Correct，正常注入；[0.30, 0.50) → Ambiguous
# 混合检索 + rerank 参数：
_HYBRID_RECALL = 16    # 混合检索召回数（向量+BM25 各取 48 再 RRF 到 16）


def _load_vaultrag_config() -> Dict[str, Any]:
    """从 config.yaml 的 context.vaultrag 段读插件配置（fail-open）。

    结构：
      context:
        engine: vaultrag
        vaultrag:
          vault_path: D:/llmwiki/llm-wiki
          embedding:
            base_url: https://api.siliconflow.cn/v1
            model: BAAI/bge-m3

    读不到/配置损坏 → 返回空 dict，插件走环境变量/默认值兜底，不崩。
    """
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
        ctx = cfg.get("context", {}) or {}
        vcfg = ctx.get("vaultrag", {}) or {}
        return vcfg if isinstance(vcfg, dict) else {}
    except Exception:
        return {}


class VaultRAGEngine(ContextEngine):
    name = "vaultrag"

    def __init__(self, vault_root: Optional[str] = None, embedding: Optional[EmbeddingClient] = None):
        # 配置优先级：显式参数 > config.yaml (context.vaultrag.*) > 环境变量
        # （行为配置走 config.yaml，.env 只放密钥 —— Hermes 插件规范）
        vcfg = _load_vaultrag_config()
        self.vault_root = (
            vault_root
            or vcfg.get("vault_path")
            or os.getenv("OBSIDIAN_VAULT_PATH")  # 向后兼容
        )
        emb_cfg = vcfg.get("embedding", {}) or {}
        self.embedding = embedding or EmbeddingClient(
            base_url=emb_cfg.get("base_url"),
            model=emb_cfg.get("model"),
        )
        self.index = VaultIndex(self.vault_root, embedding=self.embedding)
        self._index_ready = False
        # 委托压缩：内置 ContextCompressor 处理 should_compress/compress。
        # ⚠ 2026-08-17 踩坑教训（两层）：
        #  ① host 不会给外部引擎挂 delegate（agent_init 注释 "External engines own
        #     compaction policy"，attach_delegate 全仓无调用方）——之前 _delegate=None
        #     导致 should_compress 恒 False、compress 原样返回 = 压缩被完全禁用；
        #  ② ContextCompressor.__init__ 必传 model 等参数（无参实例化报错），
        #     委托必须在 host 第一次 update_model() 时用传入参数创建。
        # 修复：懒创建——update_model() 是 host 引擎选择后必调用的（agent_init.py:2593），
        # 在那里用真实参数实例化委托，后续转发全部压缩职责。
        self._delegate = None
        self._delegate_kwargs = None

    def _ensure_delegate(self):
        """懒创建委托：用 host update_model 传入的参数实例化内置 ContextCompressor。

        ⚠ 参数映射：update_model 的 kwargs 是 host 视角的字段名
        （context_length），而 ContextCompressor.__init__ 用的是
        config_context_length——直接透传会炸（2026-08-17 实测）。
        """
        if self._delegate is not None:
            return self._delegate
        kw = self._delegate_kwargs or {}
        if not kw:
            return None
        try:
            from agent.context_compressor import ContextCompressor
            from hermes_cli.config import load_config_readonly

            build_kwargs = {
                "model": kw.get("model", ""),
                "base_url": kw.get("base_url", ""),
                "api_key": kw.get("api_key", ""),
                "provider": kw.get("provider", ""),
                "api_mode": kw.get("api_mode", ""),
                "config_context_length": kw.get("context_length"),
            }
            # ⚠ 2026-08-17：内置路径（agent_init.py:2604-2626）构造时传 22 个参数
            # （全部来自 config.yaml compression 段 + agent 属性）；host 对插件引擎
            # 只调 update_model 且不传 max_tokens——委托必须自己从 config.yaml 读
            # compression 段补全，否则压缩触发点/行为与内置路径不一致。
            try:
                cfg = load_config_readonly() or {}
                comp = (cfg.get("compression") or {}) if isinstance(cfg, dict) else {}
                build_kwargs.update({
                    "threshold_percent": float(comp.get("threshold", 0.50)),
                    "protect_first_n": int(comp.get("protect_first_n", 3)),
                    "protect_last_n": int(comp.get("protect_last_n", 20)),
                    "summary_target_ratio": float(comp.get("target_ratio", 0.20)),
                    "tail_mode": comp.get("tail_mode", "legacy") or "legacy",
                    "model_thresholds": comp.get("model_thresholds"),
                    "threshold_tokens_cap": comp.get("threshold_tokens"),
                    "proactive_prune_tokens": int(comp.get("proactive_prune_tokens", 0)),
                    "proactive_prune_min_result_chars": int(comp.get("proactive_prune_min_result_chars", 8000)),
                    "proactive_prune_min_reclaim_tokens": int(comp.get("proactive_prune_min_reclaim_tokens", 4096)),
                    "min_tail_user_messages": int(comp.get("min_tail_user_messages", 1)),
                    "abort_on_summary_failure": bool(comp.get("abort_on_summary_failure", False)),
                })
                # max_tokens：host 不给插件引擎传，从 config model 段读（若配置）
                model_cfg = cfg.get("model") if isinstance(cfg, dict) else None
                if isinstance(model_cfg, dict):
                    mt = model_cfg.get("max_tokens")
                    if mt:
                        build_kwargs["max_tokens"] = int(mt)
            except Exception as e:
                logger.warning("[vaultrag] 读取 compression 配置失败，用默认值: %s", e)
            self._delegate = ContextCompressor(**build_kwargs)
            logger.info("[vaultrag] 压缩委托已创建: %s", build_kwargs.get("model"))
        except Exception as e:
            logger.warning("[vaultrag] 压缩委托创建失败: %s", e)
            self._delegate = None
        return self._delegate

    def update_model(self, *args, **kwargs):
        """转发给委托：host 切模型/改上下文长度时，压缩策略同步更新。

        host 在引擎选择后必调 update_model()（agent_init.py:2593），参数含
        model/context_length/base_url/api_key/provider/api_mode——委托在此创建。
        """
        self._delegate_kwargs = kwargs
        delegate = self._ensure_delegate()
        if delegate is not None and hasattr(delegate, "update_model"):
            try:
                return delegate.update_model(*args, **kwargs)
            except Exception as e:
                logger.warning("[vaultrag] 压缩委托 update_model 失败: %s", e)
        return None

    def bind_session_state(self, *args, **kwargs):
        """转发给委托：会话绑定（cooldown/统计状态持久化）。"""
        delegate = self._ensure_delegate()
        if delegate is not None and hasattr(delegate, "bind_session_state"):
            try:
                return delegate.bind_session_state(*args, **kwargs)
            except Exception as e:
                logger.warning("[vaultrag] 压缩委托 bind_session_state 失败: %s", e)
        return None

    def on_session_start(self, *args, **kwargs):
        """转发给委托：会话开始事件（压缩状态重置/延续）。"""
        delegate = self._ensure_delegate()
        if delegate is not None and hasattr(delegate, "on_session_start"):
            try:
                return delegate.on_session_start(*args, **kwargs)
            except Exception as e:
                logger.warning("[vaultrag] 压缩委托 on_session_start 失败: %s", e)
        return None

    # -- ContextEngine 必需接口 ------------------------------------------

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """转发给委托：每次 API 调用后的 token 用量喂给压缩器。

        ⚠ 2026-08-17：之前这里是 pass——delegate 永远拿不到真实用量，
        should_compress 的阈值判断基于过期/零数据。host 每次调用都走
        引擎的 update_from_response，必须转发，压缩阈值才有效。
        """
        delegate = self._ensure_delegate()
        if delegate is not None and hasattr(delegate, "update_from_response"):
            try:
                return delegate.update_from_response(usage)
            except Exception as e:
                logger.warning("[vaultrag] 压缩委托 update_from_response 失败: %s", e)
        return None

    def should_compress(self, prompt_tokens: int = None) -> bool:
        delegate = self._ensure_delegate()
        return delegate.should_compress(prompt_tokens) if delegate else False

    # -- 属性转发：preflight/UI 读这些属性做显示与预检 ----------------
    # ⚠ 2026-08-17：ContextEngine 基类把 threshold_tokens/context_length
    # 定义为类属性默认 0（context_engine.py:106-107），VaultRAGEngine 不覆盖
    # 的话，turn_context preflight 日志显示 ">= 0 threshold (ctx 0)"——
    # 功能没坏（should_compress 走委托的真实值），但显示误导。
    # 转发到委托，让显示与真实判断一致。

    @property
    def threshold_tokens(self) -> int:
        delegate = self._ensure_delegate()
        return delegate.threshold_tokens if delegate is not None else 0

    @property
    def context_length(self) -> int:
        delegate = self._ensure_delegate()
        return delegate.context_length if delegate is not None else 0

    @property
    def protect_first_n(self) -> int:
        delegate = self._ensure_delegate()
        return delegate.protect_first_n if delegate is not None else 3

    @property
    def protect_last_n(self) -> int:
        delegate = self._ensure_delegate()
        return delegate.protect_last_n if delegate is not None else 6

    def compress(self, messages, current_tokens=None, focus_topic=None, force=False, memory_context=""):
        delegate = self._ensure_delegate()
        if delegate is not None:
            return delegate.compress(
                messages,
                current_tokens=current_tokens,
                focus_topic=focus_topic,
                force=force,
                memory_context=memory_context,
            )
        return messages

    def attach_delegate(self, delegate: Any) -> None:
        """由宿主注入内置压缩器作为委托（agent_init 在引擎选择后组装）。"""
        self._delegate = delegate

    # -- 核心：每回合检索注入 -------------------------------------------

    # 多概念查询检测（2026-08-23，Adaptive RAG 查询复杂度感知的理论落地）：
    # 比较/关系标记 + 多分句/分隔符 → 该查询期望多个概念，无单块能覆盖整句，
    # guard 阈值应放宽（见 3.6b）。规则可解释，负样本（生活话题）不触发。
    _COMPARE_MARKS = ("区别", "关系", "联系", "异同", "分别", "对比", "vs", "versus", "差别", "不同")
    _CONJUNCTIONS = ("和", "与", "、", "+", "／", "/", "以及", "跟")

    @classmethod
    def _is_multi_concept(cls, query: str) -> bool:
        """多概念查询检测：含比较标记 且 含分隔符（并列结构）→ True。

        例："A 和 B 有什么区别" → True（区别 + 和）
             "Causal Coupling guard 解决的是什么问题" → False（无比较标记）
             生活话题负样本（"装修选乳胶漆"）→ False（无比较标记）
        """
        if not query:
            return False
        q = query.lower()
        has_compare = any(m in q for m in cls._COMPARE_MARKS)
        if not has_compare:
            return False
        has_conj = any(c in q for c in cls._CONJUNCTIONS)
        return has_conj

    @classmethod
    def _enhance_guard_query(cls, query: str, pool: List[Dict]) -> str:
        """构造 guard rerank 用的查询（2026-08-26 最终态：不做增强）。

        演进记录：
        - query2doc 增强（query + top1 标题词，AOP 0.121→0.271）曾依赖
          "命中缩写词表"做门控（仅技术查询增强，防生活话题误放行）。
        - 2026-08-26 缩写展开机制删除（消融 evals/ablation_abbrev.py：
          hit@1 零贡献，向量层已覆盖缩写语义）→ 门控信号消失。
        - 无条件增强实测 5/5 生活负样本误放行（rerank 自匹配虚高），
          不可行；英文/大写缩写门控会误伤 NBA 类生活查询。
        - 最终决策：guard 用原始查询 rerank（最保守）。短技术查询
          （AOP/TTS 类）裸喂时可能被拦（假阴性）——由 agent 弹性吸收：
          LLM 理解缩写、可换词重查或用 search_files 兜底，引擎不做
          LLM 该做的事。负样本拦截 100% 保持（硬指标）。
        """
        return query

    def search(self, query: str, top_k: int = _TOP_K) -> Optional[Dict[str, Any]]:
        """统一检索入口——select_context 与 rag_search 工具共用（单一事实源）。

        混合检索（BM25 + 向量 → RRF）→ 过滤 index 页 → guard 三档判定。
        返回结构化结果（hits/verdict/top1_score/margin/reason），拦截或失败也返回
        dict（verdict=incorrect/skipped + reason），None 仅限异常兜底（fail-open）。

        2026-08-26 修正：rerank top_n=2——原 select_context 用 top_n=1 使 top2 恒为 0、
        margin 恒等于 top1、margin 判定实际失效；工具 handler 却用真实 top2。
        现统一 top_n=2，margin 真实生效，两路行为一致。
        """
        try:
            if not self._index_ready:
                self._index_ready = self.index.ensure_index()
            if not self._index_ready:
                return {"query": query, "hits": [], "verdict": "skipped", "top1_score": 0.0,
                        "top2_score": 0.0, "margin": 0.0, "multi_concept": False, "reason": "index-not-ready"}
            q = (query or "").strip()
            if not q:
                return {"query": q, "hits": [], "verdict": "skipped", "top1_score": 0.0,
                        "top2_score": 0.0, "margin": 0.0, "multi_concept": False, "reason": "no-query"}
            if len(q) < _MIN_QUERY_CHARS:
                return {"query": q, "hits": [], "verdict": "skipped", "top1_score": 0.0,
                        "top2_score": 0.0, "margin": 0.0, "multi_concept": False, "reason": "length-gate"}
            qv = self.embedding.embed_query(q)
            if qv is None:
                return {"query": q, "hits": [], "verdict": "skipped", "top1_score": 0.0,
                        "top2_score": 0.0, "margin": 0.0, "multi_concept": False, "reason": "embedding-failed"}
            candidates = self.index.hybrid_search(q, qv, top_k=_HYBRID_RECALL)
            if not candidates:
                return {"query": q, "hits": [], "verdict": "skipped", "top1_score": 0.0,
                        "top2_score": 0.0, "margin": 0.0, "multi_concept": False, "reason": "no-candidates"}
            # 过滤 index 页（MOC：关键词齐全但无答案 → rerank 误判高分）
            pool = [c for c in candidates if Path(c["source"]).stem != "index"]
            if not pool:
                return {"query": q, "hits": [], "verdict": "skipped", "top1_score": 0.0,
                        "top2_score": 0.0, "margin": 0.0, "multi_concept": False, "reason": "no-pool"}
            hits = pool[:top_k]
            # guard：rerank 打分（top_n=2 取真实 top2，margin 才有意义），失败退回 RRF 分数
            guard_query = self._enhance_guard_query(q, pool)
            cand_texts = [c["text"][:_MAX_CHARS_PER_HIT] for c in pool]
            guard_scores = self.embedding.rerank(guard_query, cand_texts, top_n=2)
            if guard_scores:
                top1 = float(guard_scores[0]["score"])
                top2 = float(guard_scores[1]["score"]) if len(guard_scores) > 1 else 0.0
            else:
                top1 = float(hits[0]["score"]) if hits else 0.0
                top2 = float(hits[1]["score"]) if len(hits) > 1 else 0.0
            margin = top1 - top2
            multi = self._is_multi_concept(q)
            min_score = 0.15 if multi else _MIN_SCORE
            amb_margin = 0.05 if multi else 0.15
            if top1 < min_score:
                return {"query": q, "hits": [], "verdict": "incorrect", "top1_score": top1,
                        "top2_score": top2, "margin": margin, "multi_concept": multi,
                        "reason": "score-below-threshold"}
            verdict = "correct" if (top1 >= 0.40 and margin >= amb_margin) else "ambiguous"
            return {"query": q, "hits": hits, "verdict": verdict, "top1_score": top1,
                    "top2_score": top2, "margin": margin, "multi_concept": multi, "reason": "ok"}
        except Exception as e:
            logger.warning("[vaultrag] search failed, fail-open: %s", e)
            return None

    def select_context(self, request_messages, *, conversation_messages=None, incoming_message=None, budget_tokens=0):
        """检索 vault，把命中片段注入请求。返回新消息列表；失败返回 None（fail-open）。"""
        # 结构化 trace（RAGOps，2026-08-19）：每次调用记一条完整决策链，
        # 用于离线分析"为什么不注入某查询"。逐字段填充，任何退出点都写 trace
        # （fail-open：写失败不影响检索）。
        trace = {
            "query": "",
            "length_gate_pass": False,
            "recalled": 0,
            "expanded_from": [],
            "graph_stats": {},
            "rerank_top_scores": [],
            "verdict": "skipped",
            "injected_count": 0,
            "reason": "",
        }
        try:
            # 1. 索引就绪（懒加载：首次调用构建/加载缓存）
            if not self._index_ready:
                self._index_ready = self.index.ensure_index()
            if not self._index_ready:
                trace["reason"] = "index-not-ready"
                self._emit_trace(trace)
                return None

            # 2. 拿本轮用户问题（注入的检索辅助消息不算）
            query = self._extract_query(incoming_message, request_messages)
            trace["query"] = query
            if not query:
                trace["reason"] = "no-query"
                self._emit_trace(trace)
                return None

            # 2.5 查询质量门槛（防噪音注入）：1 字确认消息（"好""嗯""谢谢"等）
            #     信息量不足 → 直接跳过。阈值已降到 2（见 _MIN_QUERY_CHARS 注释）。
            if len(query) < _MIN_QUERY_CHARS:
                trace["reason"] = "length-gate"
                self._emit_trace(trace)
                return None
            trace["length_gate_pass"] = True

            # 3. 统一检索（单一事实源：select_context 与 rag_search 工具共用 search；
            #    混合检索/过滤/guard 判定全部在 search 内，杜绝双份代码漂移，2026-08-26）
            result = self.search(query, top_k=_TOP_K)
            if result is None:
                trace["reason"] = "exception"
                self._emit_trace(trace)
                return None

            hits = result["hits"]
            top1 = result["top1_score"]
            trace["recalled"] = len(hits)
            trace["guard_top1_score"] = round(top1, 4)
            trace["top1_top2_margin"] = round(result["margin"], 4)
            if result["multi_concept"]:
                trace["multi_concept"] = True

            if result["verdict"] in ("incorrect", "skipped"):
                trace["verdict"] = result["verdict"]
                trace["reason"] = result["reason"]
                self._emit_trace(trace)
                return None
            if not hits:
                trace["reason"] = "no-hits"
                self._emit_trace(trace)
                return None
            verdict = result["verdict"]

            # 4. 拼注入消息（克隆消息列表，绝不修改原列表）
            injected = self._build_injection(hits)
            new_messages = list(request_messages)
            new_messages.append(
                {
                    "role": _INJECT_ROLE,
                    "content": injected,
                    # 注意：不带任何自定义字段（如 metadata）——
                    # 严格 OpenAI 兼容 provider 会拒收未知键，
                    # 来源信息只进日志（2026-08-16 实测 DeepSeek 宽容，但不赌）
                }
            )
            if verdict == "ambiguous":
                logger.info(
                    "[vaultrag] low-confidence injection (verdict=ambiguous, query=%r, top=%.3f)",
                    query[:40], top1,
                )
            logger.info(
                "[vaultrag] injected %d hits (query=%r, top=%.3f) sources=%s",
                len(hits), query[:40], top1,
                [h["source"] for h in hits],
            )
            trace["verdict"] = verdict
            trace["injected_count"] = len(hits)
            self._emit_trace(trace)
            return new_messages
        except Exception as e:
            logger.warning("[vaultrag] select_context failed, passing through: %s", e)
            try:
                if not trace["reason"]:
                    trace["reason"] = "exception"
                trace["verdict"] = "skipped"
                self._emit_trace(trace)
            except Exception:
                pass  # trace 本身也失败 → 彻底静默（fail-open）
            return None

    # -- 辅助 -----------------------------------------------------------

    def _emit_trace(self, trace: Dict[str, Any]) -> None:
        """结构化 trace（RAGOps）：追加写 JSONL，fail-open（写失败不影响检索）。

        每次 select_context 调用都写一条完整检索决策链（谁被拦、为什么、
        召回多少、top 分数、判定档位），落盘到 vault/.smart-env/vaultrag/trace.jsonl
        （与索引缓存同目录），用于离线分析"为什么不注入某查询"。
        """
        try:
            import json
            from datetime import datetime

            record = dict(trace)
            record["timestamp"] = datetime.now().astimezone().isoformat(timespec="milliseconds")
            trace_path = self.index.cache_dir / "trace.jsonl"
            with open(trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass  # fail-open：trace 写失败绝不影响对话

    @staticmethod
    def _extract_query(incoming_message, request_messages) -> str:
        """取本轮用户问题：优先 incoming_message，回退到列表里最后一条 user 消息。"""
        if isinstance(incoming_message, dict):
            content = incoming_message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()[:500]
        for m in reversed(request_messages):
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                return m["content"].strip()[:500]
        return ""

    @staticmethod
    def _build_injection(hits) -> str:
        """把命中片段拼成一段给模型的上下文（带出处，模型可引用）。"""
        parts = ["<knowledge_context>", "以下是用户知识库中与问题相关的片段（按相关度排序）："]
        for i, h in enumerate(hits, 1):
            text = h["text"].replace("\r", "")[:_MAX_CHARS_PER_HIT]
            parts.append(f"[{i}] 来源: {h['source']}\n{text}")
        parts.append("</knowledge_context>")
        return "\n\n".join(parts)


# =====================================================================
# 斜杠命令：/llm-wiki-init <vault路径>
# 用法：/llm-wiki-init D:/llmwiki/llm-wiki
# 效果：校验路径 → 扫描建索引 → 写 config.yaml 的 context.vaultrag.vault_path
#       → 返回结果文本。之后 context.engine=vaultrag 生效即可检索该 vault。
# =====================================================================

def _cmd_llm_wiki_init(raw_args: str) -> str:
    """注册到全局斜杠命令表的 handler：fn(raw_args: str) -> str。"""
    vault = (raw_args or "").strip().strip("\"'")
    if not vault:
        return (
            "用法: /llm-wiki-init <vault路径>\n"
            "示例: /llm-wiki-init D:/llmwiki/llm-wiki\n"
            "效果: 扫描 vault 建索引 + 写入 context.vaultrag.vault_path 配置。"
        )
    from pathlib import Path

    root = Path(vault)
    if not root.is_dir():
        return f"❌ 路径不存在或不是目录: {vault}"
    md_files = list(root.rglob("*.md"))
    if not md_files:
        return f"❌ 目录下没有 .md 文件: {vault}"

    # 1. 扫描 + 建索引（强制重建）
    emb = EmbeddingClient()
    if not emb.available:
        return "❌ EMBEDDING_API_KEY 未配置（.env），无法建索引"
    idx = VaultIndex(vault, embedding=emb)
    if not idx.ensure_index(force=True):
        return f"❌ 索引构建失败（embedding 调用异常？），vault={vault}"
    total_chunks = idx.size
    docs = scan_vault(vault)
    total_docs = len(docs)

    # 2. 写配置（force=True：跳过未知键警告，自定义键允许写入）
    #    注：只写 vault_path，不写 context.engine——引擎已迁入本插件
    #    （rag-search/vaultrag/），context.engine=vaultrag 已无法被
    #    load_context_engine 解析（2026-08-26 迁移修复）。
    try:
        from hermes_cli.config import set_config_value

        set_config_value("context.vaultrag.vault_path", str(root), force=True)
    except Exception as e:
        return f"❌ 索引已建（{total_docs} 篇 / {total_chunks} 块），但配置写入失败: {e}"

    return (
        f"✅ llm-wiki 初始化完成\n"
        f"  vault: {root}\n"
        f"  笔记: {total_docs} 篇 → {total_chunks} 块\n"
        f"  已写入 context.vaultrag.vault_path（rag_search 工具读取）\n"
        f"  重启 Hermes 后生效（或新会话工具集加载时生效）"
    )


def register(ctx):
    """插件注册入口（load_context_engine 的 register(ctx) 模式，保留兼容）。

    注册 /llm-wiki-init 斜杠命令 + 引擎实例。
    注：本引擎已随知识检索工具迁移为 plugins/knowledge-search/vaultrag/
    （rag_search 工具由 knowledge-search 插件注册；引擎形态闲置，
    context.engine 默认 compressor 不加载本引擎）。
    """
    ctx.register_context_engine(VaultRAGEngine())
    ctx.register_command(
        "llm-wiki-init",
        _cmd_llm_wiki_init,
        description="初始化 llm-wiki vault（扫描建索引 + 写入 context 配置）",
        args_hint="<vault路径>",
    )
