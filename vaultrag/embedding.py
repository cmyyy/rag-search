"""embedding.py — 云端 embedding 客户端（OpenAI 兼容接口，零本地模型依赖）。

设计原则（面试点）：
  - 不跑本地小模型：对个人用户零负担（不用下载模型、不占内存、不用装
    sentence-transformers 全家桶）。embedding 交给云端 OpenAI 兼容 API。
  - Provider 可插拔：只依赖 OpenAI 兼容的 /v1/embeddings 接口，
    换厂商（硅基流动/智谱/阿里/OpenAI）只改 .env 三个配置，代码不动。
  - 批量接口：vault 索引时一次传多段文本，省 API 调用次数。

配置（.env）：
  EMBEDDING_API_KEY    必填，如硅基流动 sk-xxx
  EMBEDDING_BASE_URL   可选，默认 https://api.siliconflow.cn/v1
  EMBEDDING_MODEL      可选，默认 BAAI/bge-m3（1024 维，中文强，免费）
"""
import os
from typing import Dict, List, Optional

import numpy as np


class EmbeddingClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY", "")
        self.base_url = base_url or os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
        self.model = model or os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        self._client = None
        self._batch_size = 64

    def _lazy_client(self):
        """延迟初始化 openai 客户端：Hermes 主进程没配 key 时，导入插件也不报错。"""
        if self._client is None:
            if not self.api_key:
                raise RuntimeError("EMBEDDING_API_KEY 未配置（.env）")
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def embed_texts(self, texts: List[str]) -> Optional[np.ndarray]:
        """批量向量化，返回 (n, dim) float32 矩阵。失败返回 None（fail-open）。"""
        if not texts:
            return None
        try:
            client = self._lazy_client()
            vectors = []
            for i in range(0, len(texts), self._batch_size):
                batch = texts[i : i + self._batch_size]
                resp = client.embeddings.create(model=self.model, input=batch)
                vectors.extend([d.embedding for d in resp.data])
            return np.asarray(vectors, dtype=np.float32)
        except Exception as e:
            print(f"[vaultrag] embedding 调用失败: {e}")
            return None

    def embed_query(self, text: str) -> Optional[np.ndarray]:
        """单个查询向量化，返回 (dim,) 或 None。"""
        arr = self.embed_texts([text])
        return arr[0] if arr is not None else None

    # -- rerank（cross-encoder 精排）-----------------------------------

    def rerank(self, query: str, documents: List[str], top_n: int = 4) -> Optional[List[Dict]]:
        """cross-encoder 重排：query 与每篇文档逐对细读打分。

        为什么比 embedding 相似度强（2026-08-16 实测）：
          embedding 是"各自压成向量再比"（bi-encoder），丢失词级交互；
          reranker 把 query+文档拼一起让模型细读（cross-encoder），
          区分度高一档——实测同一查询，相关文档 0.63 vs 无关文档 0.005。

        硅基流动 /v1/rerank 是自定义端点（openai SDK 无 rerank 方法），
        用 urllib 直接调，零新依赖。失败返回 None（fail-open）。
        """
        if not documents:
            return None
        try:
            import json
            import urllib.request

            payload = json.dumps(
                {"model": os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
                 "query": query, "documents": documents, "top_n": top_n},
                ensure_ascii=False,
            ).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url.rstrip('/')}/rerank",
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
            return [
                {"index": r["index"], "score": float(r["relevance_score"])}
                for r in resp.get("results", [])
            ]
        except Exception as e:
            print(f"[vaultrag] rerank 调用失败: {e}")
            return None
