"""retriever.py — Obsidian vault 扫描 + 分块 + 向量索引 + top-k 检索。

设计（面试点）：
  - 不依赖 chromadb/faiss：笔记规模（几千篇）下 numpy 矩阵点积足够，
    零重依赖（插件运行在 Hermes venv，只有 numpy + openai）。
  - 索引持久化：embedding 结果存 .npz（向量矩阵 + 文本清单 + 文件哈希），
    只有 vault 文件变化才重建，避免每次启动重新调 API。
  - 分块策略：按 Markdown 标题（## / ###）切块，块与笔记双向可追溯
    （检索命中 → 返回"笔记路径 + 块内容"，模型能引用出处）。
  - 双链图（2026-08-22 新增）：解析 [[wikilink]] 建笔记级邻接表，
    混合检索命中后沿图 depth=1 扩展，把关联笔记的块并入 rerank 候选池，
    由 cross-encoder 过滤无关——弥补"纯文本块检索无块间关联"的短板。
  - fail-open：任何异常返回空列表，调用方（select_context）原样放行。
"""
import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from .embedding import EmbeddingClient
except ImportError:  # 顶层运行（独立仓库评测）：绝对导入
    from embedding import EmbeddingClient

# Markdown 标题级分块：## 和 ### 开头的行作为块边界
_HEADING_RE = re.compile(r"^(#{2,3})\s+(.+)$")
# Obsidian wikilink：[[目标|别名]] 或 [[目标#锚点]]
_WIKILINK_RE = re.compile(r"\[\[([^\[\]]+?)\]\]")
# 索引版本号：切块逻辑/文本形态变化时 bump，旧 .npz 缓存自然失效（文件名带版本）
# v3 = 2026-08-22：分块参数（with_breadcrumb/max_chunk_chars）纳入缓存键，避免不同参数共用缓存
_INDEX_VERSION = 3
# 忽略的目录/文件
# raw/ = 原始文章存档（非知识笔记），copilot/ = 工具模板，均不进知识库索引
_IGNORE_DIRS = {
    ".obsidian", ".smart-env", ".git", "node_modules", ".trash",
    "raw", "copilot", "templates", "attachments",
}
_IGNORE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".excalidraw", ".canvas"}
# 元数据/操作日志文件：不是知识笔记，检索命中无意义（index=目录页，log=操作日志，SCHEMA=结构规范）
_IGNORE_FILES = {"index", "log", "SCHEMA"}


def _is_ignored(path: Path) -> bool:
    if path.suffix.lower() in _IGNORE_EXT:
        return True
    if path.stem in _IGNORE_FILES:
        return True
    return any(part in _IGNORE_DIRS for part in path.parts)


def _parse_frontmatter(text: str) -> Tuple[Dict, str]:
    """解析 Obsidian YAML frontmatter，返回 ({title, tags}, 正文)。

    零依赖手写解析（不引入 yaml 库）：只认两件事——
      - title: 笔记标题（面包屑第一段，缺失时回退文件名）
      - tags:  行内数组 `[ai, llm]` 或块状列表（`tags:` 下一行起 `- xxx`）
    其余字段（created/updated/source/status 等）与检索无关，忽略。
    正文 = frontmatter 结束标记（---/...）之后的内容，避免 YAML 污染切块。
    """
    fm: Dict = {"title": "", "tags": []}
    if not text.startswith("---"):
        return fm, text
    lines = text.splitlines()
    if len(lines) < 2:
        return fm, text
    i = 1
    in_tags_list = False
    while i < len(lines):
        line = lines[i].strip()
        if line in ("---", "..."):
            break
        if in_tags_list:
            if line.startswith("-"):
                item = line[1:].strip().strip("\"'")
                if item:
                    fm["tags"].append(item)
                i += 1
                continue
            in_tags_list = False
        if line.startswith("title:"):
            fm["title"] = line[len("title:") :].strip().strip("\"'")
        elif line.startswith("tags:"):
            rest = line[len("tags:") :].strip()
            if rest.startswith("["):
                fm["tags"] = [
                    t.strip().strip("\"'") for t in rest.strip("[]").split(",") if t.strip()
                ]
            elif not rest:
                in_tags_list = True
        i += 1
    body = "\n".join(lines[i + 1 :])
    return fm, body


def normalize_wikilink(target: str) -> str:
    """把 wikilink 目标归一化为笔记名（去路径/后缀/锚点/空白）。

    Obsidian 允许 `[[文件夹/笔记|别名]]`、`[[笔记#锚点]]`：
      - 目标名取首个 `|` 前（别名只是锚文本，检索用不到）；
      - 忽略 `#` section 形式（实测 vault 无此用法，直接截断）；
      - 归一化：去掉 .md 后缀、取路径 basename、去首尾空白。
    """
    t = target.split("|", 1)[0].split("#", 1)[0].strip()
    t = t.replace("\\", "/").rstrip("/")
    base = t.rsplit("/", 1)[-1]
    if base.lower().endswith(".md"):
        base = base[:-3]
    return base.strip()


def parse_wikilinks(text: str) -> List[str]:
    """正则解析全部 wikilink，返回去重（保序）后的归一化笔记名列表。"""
    seen: Dict[str, None] = {}
    for m in _WIKILINK_RE.finditer(text):
        name = normalize_wikilink(m.group(1))
        if name:
            seen.setdefault(name, None)
    return list(seen)


def split_markdown(text: str, note_title: str = "", tags: Optional[List[str]] = None, with_breadcrumb: bool = True, max_chunk_chars: int = 1600) -> List[Tuple[str, str]]:
    """按 ## / ### 标题切块，返回 [(面包屑标题, 块内容)]。

    维护标题栈生成面包屑路径：`笔记标题 > ## 章节 > ### 小节`——
    命中块时模型能直接看到"这段属于哪篇笔记的哪个章节"，跨笔记同名词不混淆。
    每块末尾追加 frontmatter 的 tags（如 `[tags: ai, llm]`），补语义分类信号。
    无标题的正文归入"未分节"块。块保留标题行，让检索结果自带上下文。

    *with_breadcrumb=False* 时退化为旧版纯文本切块（无面包屑、无 tags），
    供消融实验量化面包屑/frontmatter 的贡献。

    2026-08-22 优化（实测驱动）：
    - 超长块二次切分：块 > max_chunk_chars 时按段落（空行）再切，避免大章节整块
      （实测 22.6% 块 >1500 字符，最大 13876，稀释向量/占满注入预算）
    - 段落兜底：无 ## / ### 结构时按空行分块（实测 13.4% 笔记整篇单块，精度差）
    """
    lines = text.splitlines()
    chunks: List[Tuple[str, str]] = []
    stack: List[str] = [note_title] if note_title else []
    cur_lines: List[str] = []
    has_heading = False

    def breadcrumb() -> str:
        """由标题栈拼面包屑；无任何标题时回退"未分节"（兼容旧行为）。"""
        parts = [s for s in stack if s]
        return " > ".join(parts) if parts else "未分节"

    def emit(title: str, body: str):
        if with_breadcrumb and tags:
            body = body + f"\n[tags: {', '.join(tags)}]"
        chunks.append((title, body))

    def flush():
        title = breadcrumb() if with_breadcrumb else ""
        body = "\n".join(cur_lines).strip()
        if not body:
            return
        # 超长块二次切分：按段落（空行分隔）切成 ≤max_chunk_chars 的块
        if len(body) > max_chunk_chars:
            paras = [p.strip() for p in body.split("\n\n") if p.strip()]
            cur = ""
            for p in paras:
                if len(cur) + len(p) > max_chunk_chars and cur:
                    emit(title, cur)
                    cur = p
                else:
                    cur = f"{cur}\n\n{p}" if cur else p
            if cur:
                emit(title, cur)
        else:
            emit(title, body)

    for line in lines:
        m = _HEADING_RE.match(line.strip())
        if m:
            flush()
            has_heading = True
            level = len(m.group(1))
            heading = m.group(2).strip()
            if level == 2:
                # ## 章节：重置栈（笔记标题 之下只挂这一层章节）
                stack = [note_title, heading] if note_title else [heading]
            else:
                # ### 小节：保留当前 ## 章节（若有），在其下叠一层小节
                if len(stack) >= 2:
                    stack = stack[:2] + [heading]
                else:
                    stack = [note_title, heading] if note_title else [heading]
            cur_lines = [line]
        else:
            cur_lines.append(line)
    flush()

    # 段落兜底：整篇无 ## / ### 结构时，把"未分节"大块按空行再切（治 13.4% 单块笔记）
    if not has_heading and len(chunks) == 1 and len(chunks[0][1]) > max_chunk_chars:
        title, body = chunks[0]
        paras = [p.strip() for p in body.split("\n\n") if p.strip()]
        chunks = [(title, p) for p in paras] or [chunks[0]]
    return chunks


def scan_vault(vault_root: str, with_breadcrumb: bool = True, max_chunk_chars: int = 1600) -> List[Dict]:
    """扫描 vault 下所有 .md，返回 [{path, title, links, chunks:[(面包屑, body)]}]。

    title 优先取 frontmatter.title（面包屑第一段），缺失时回退文件名。
    links = 归一化后的 wikilink 目标（供双链邻接表使用，Day2-3 特性）。
    frontmatter 本身不进切块，只贡献 title/tags。
    """
    root = Path(vault_root)
    docs = []
    if not root.is_dir():
        return docs
    for md in sorted(root.rglob("*.md")):
        if _is_ignored(md):
            continue
        try:
            text = md.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if text.startswith("\ufeff"):
            text = text[1:]  # 去掉 UTF-8 BOM，避免污染 frontmatter 判定
        fm, body = _parse_frontmatter(text)
        note_title = fm.get("title") or md.stem
        chunks = split_markdown(body, note_title=note_title, tags=fm.get("tags") or None, with_breadcrumb=with_breadcrumb, max_chunk_chars=max_chunk_chars)
        if chunks:
            docs.append({
                "path": str(md),
                "title": note_title,
                "links": parse_wikilinks(text),
                "chunks": chunks,
            })
    return docs


class BM25Index:
    """BM25 关键词检索（纯 numpy，零依赖）。

    为什么加 BM25（2026-08-16，arXiv 2604.01733 基准 + 实测）：
      - 语义检索补不了精确匹配：缩写/术语/ID（"MoA""四道闸"）靠关键词
      - 短确认消息（"好""改"）在文档里几乎不出现 → BM25 分数天然趋近零，
        源头就召回不到——比事后长度判断更根本的噪音过滤
      - 同一基准里 BM25 多数指标甚至优于 text-embedding-3-large

    实现：中文按字符 bigram 切分（不用 jieba，零依赖），英文按词。
    """

    def __init__(self, texts: List[str], header_weight: float = 0.0):
        """BM25 关键词检索（纯 numpy，零依赖）。

        为什么加 BM25（2026-08-16，arXiv 2604.01733 基准 + 实测）：
          - 语义检索补不了精确匹配：缩写/术语/ID（"MoA""四道闸"）靠关键词
          - 短确认消息（"好""改"）在文档里几乎不出现 → BM25 分数天然趋近零，
            源头就召回不到——比事后长度判断更根本的噪音过滤
          - 同一基准里 BM25 多数指标甚至优于 text-embedding-3-large

        header_weight（BM25F 思想，2026-08-23 subagent 研究落地）：
        >0 时 header（面包屑/标题路径，texts 每项第一行）命中按权重加倍——
        标题词是本库最可靠匹配信号（面包屑消融 +4.7pp 已证）。0 = 无加权。
        """
        self.header_weight = header_weight
        self.doc_headers = [t.split("\n", 1)[0] if t else "" for t in texts]
        self.doc_bodies = [t.split("\n", 1)[1] if "\n" in t else "" for t in texts]
        self.doc_terms = [self._tokenize(t) for t in texts]
        self.doc_len = [len(t) for t in self.doc_terms]
        self.avgdl = sum(self.doc_len) / max(1, len(self.doc_len))
        self.N = len(self.doc_terms)
        # IDF：每个词出现在多少文档里
        df = {}
        for terms in self.doc_terms:
            for term in set(terms):
                df[term] = df.get(term, 0) + 1
        self.df = df
        self.k1 = 1.5
        self.b = 0.75

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """中英文混合分词：中文按字符 bigram，英文/数字按单词。"""
        import re as _re
        text = text.lower()
        tokens = []
        # 连续 CJK 字符段 → bigram
        for seg in _re.findall(r"[\u4e00-\u9fff]+", text):
            seg_tokens = list(seg) if len(seg) <= 2 else [seg[i:i+2] for i in range(len(seg)-1)]
            tokens.extend(seg_tokens)
        # 英文/数字词
        tokens.extend(_re.findall(r"[a-z0-9]+", text))
        return tokens

    def score(self, query: str) -> np.ndarray:
        """对每个文档算 BM25 分，返回 (N,) 数组。

        header_weight > 0 时：header（面包屑/标题）命中的 tf 按权重放大——
        BM25F 思想：标题/头部是强信号字段，命中应比正文更值钱。
        """
        q_terms = self._tokenize(query)
        scores = np.zeros(self.N, dtype=np.float32)
        if not q_terms:
            return scores
        # header 词频（BM25F：header 命中 × header_weight）
        header_terms = [self._tokenize(h) for h in self.doc_headers] if self.header_weight > 0 else None
        for term in q_terms:
            idf = np.log(1 + (self.N - self.df.get(term, 0) + 0.5) / (self.df.get(term, 0) + 0.5))
            for i, terms in enumerate(self.doc_terms):
                tf = terms.count(term)
                if tf == 0:
                    continue
                if header_terms is not None:
                    htf = header_terms[i].count(term)
                    tf = tf + htf * self.header_weight
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * tf * (self.k1 + 1) / denom
        return scores


class VaultIndex:
    """向量索引：构建 → 持久化（.npz）→ 检索。"""

    def __init__(self, vault_root: str, cache_dir: Optional[str] = None, embedding: Optional[EmbeddingClient] = None):
        self.vault_root = vault_root
        self.embedding = embedding or EmbeddingClient()
        # 缓存目录：默认 vault/.smart-env/vaultrag（与 Smart Connections 同族，互不干扰）
        self.cache_dir = Path(cache_dir or os.path.join(vault_root, ".smart-env", "vaultrag"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._matrix: Optional[np.ndarray] = None
        self._texts: List[str] = []       # 与矩阵行一一对应：检索命中 → 文本
        self._sources: List[str] = []     # 对应笔记路径
        self._vault_hash = ""
        self._note_links: Dict[str, List[str]] = {}  # 双链图：source → [目标 source]（2026-08-30）
        self._hub_sources: set = set()  # 汇总型笔记（出链多且内容薄）——检索不注入（2026-08-30）

    # -- 构建 ----------------------------------------------------------

    def _compute_vault_hash(self, docs: List[Dict], with_breadcrumb: bool = True, max_chunk_chars: int = 1600) -> str:
        """内容指纹：文件路径 + 行数 + 大小 + 分块参数。

        分块参数（with_breadcrumb/max_chunk_chars）纳入 hash——
        参数变了缓存自然失效，避免不同参数复用旧矩阵。
        """
        h = hashlib.md5()
        h.update(f"bc={with_breadcrumb}|mcc={max_chunk_chars}|".encode())
        for d in docs:
            p = Path(d["path"])
            try:
                stat = p.stat()
                h.update(f"{d['path']}|{stat.st_size}|{stat.st_mtime_ns}".encode())
            except OSError:
                pass
        return h.hexdigest()[:16]

    def ensure_index(self, force: bool = False, with_breadcrumb: bool = True, max_chunk_chars: int = 1600) -> bool:
        """确保索引可用：vault 没变就加载缓存，变了就重建。返回是否可用。

        with_breadcrumb/max_chunk_chars 纳入缓存键——参数变了自动重建，
        不会复用旧矩阵（2026-08-22 修复：之前不同参数共用缓存）。
        """
        if not self.embedding.available:
            return False
        docs = scan_vault(self.vault_root, with_breadcrumb=with_breadcrumb, max_chunk_chars=max_chunk_chars)
        if not docs:
            return False
        self._vault_hash = self._compute_vault_hash(docs, with_breadcrumb=with_breadcrumb, max_chunk_chars=max_chunk_chars)

        cache_file = self.cache_dir / f"index_v{_INDEX_VERSION}_{self._vault_hash}.npz"
        if not force and cache_file.exists():
            return self._load(cache_file)
        return self._build(docs, cache_file)

    def _build(self, docs: List[Dict], cache_file: Path) -> bool:
        """把全部分块向量化，构建矩阵并落盘。"""
        texts, sources = [], []
        for d in docs:
            for title, body in d["chunks"]:
                texts.append(f"{title}\n{body}")
                sources.append(d["path"])
        if not texts:
            return False

        matrix = self.embedding.embed_texts(texts)
        if matrix is None:
            return False

        self._matrix = matrix
        self._texts = texts
        self._sources = sources
        # 双链图（2026-08-30）：wikilink 目标按 stem 匹配到实际笔记路径；
        # 汇总型（hub）判定用原始出链数（links 全量，非仅存在的目标）
        stem2src = {}
        for d in docs:
            stem2src.setdefault(Path(d["path"]).stem, []).append(d["path"])
        self._note_links = {}
        raw_link_count = {}
        char_by_src = {}
        for d in docs:
            raw_link_count[d["path"]] = len(d.get("links") or [])
            char_by_src[d["path"]] = sum(len(b) for _, b in d["chunks"])
            targets = []
            for name in d.get("links") or []:
                for src in stem2src.get(name, []):
                    if src != d["path"]:
                        targets.append(src)
            if targets:
                self._note_links[d["path"]] = targets
        # hub 阈值（可解释，2026-08-30）：出链 >= 8 且内容 < 2000 字符 = 链接列表型笔记
        self._hub_sources = {p_ for p_, n in raw_link_count.items()
                             if n >= self.HUB_MIN_LINKS and char_by_src.get(p_, 0) < self.HUB_MAX_CHARS}
        # BM25 索引：构建很快（bigram），每次启动重建，不持久化
        self.bm25 = BM25Index(texts, header_weight=3)
        try:
            np.savez_compressed(
                cache_file,
                matrix=matrix,
                texts=np.array(texts, dtype=object),
                sources=np.array(sources, dtype=object),
                note_links=np.array([self._note_links], dtype=object),
                hub_sources=np.array([sorted(self._hub_sources)], dtype=object),
            )
        except Exception:
            pass
        return True

    def _load(self, cache_file: Path) -> bool:
        try:
            data = np.load(cache_file, allow_pickle=True)
            self._matrix = data["matrix"]
            self._texts = list(data["texts"])
            self._sources = list(data["sources"])
            # 双链图（2026-08-30）：旧缓存无 note_links 字段时保持空图（图增强自动关闭）
            try:
                self._note_links = dict(data["note_links"][0])
            except (KeyError, IndexError, ValueError):
                self._note_links = {}
            try:
                self._hub_sources = set(data["hub_sources"][0])
            except (KeyError, IndexError, ValueError):
                self._hub_sources = set()
            self.bm25 = BM25Index(self._texts, header_weight=3)
            return True
        except Exception:
            return False

    # -- 检索 ----------------------------------------------------------

    def search(self, query_vec: np.ndarray, top_k: int = 4) -> List[Dict]:
        """余弦相似度 top-k 检索，返回 [{text, source, score, chunk_id}]。"""
        if self._matrix is None or query_vec is None:
            return []
        # 归一化后点积 = 余弦相似度
        q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        m = self._matrix / (np.linalg.norm(self._matrix, axis=1, keepdims=True) + 1e-9)
        scores = m @ q
        k = min(top_k, len(scores))
        idx = np.argsort(scores)[::-1][:k]
        return [
            {"text": self._texts[i], "source": self._sources[i],
             "score": float(scores[i]), "chunk_id": int(i)}
            for i in idx
        ]

    HUB_MIN_LINKS = 8     # 汇总型判定：出链下限
    HUB_MAX_CHARS = 2000  # 汇总型判定：内容上限（字符）

    def hybrid_search(self, query: str, query_vec: np.ndarray, top_k: int = 8,
                      max_chunks_per_note: int = 1, graph_expand: bool = True) -> List[Dict]:
        """混合检索：向量 top-k + BM25 top-k → 块级 RRF 融合 → 返回 top_k。

        RRF（Reciprocal Rank Fusion，滑铁卢+Google 2019）：
          对每个块，score = Σ 1/(rank + k)，k 默认 60。
          不直接比较两种检索的原始分数（尺度不同），而是按名次融合——
          两个列表里都排前面的块自然胜出。

        2026-08-28 修复（块级融合）：原实现以 source（笔记路径）为融合键，
        每篇笔记只保留一个块，且 by_source 字典后写覆盖返回"BM25 末位块"
        （往往最不相关）→ L4 评测 67% 失败（正确笔记 + 错误块）。
        现改为 (source, chunk_id) 块级融合 + 每笔记最多 max_chunks_per_note
        块（注入位多样性约束，可解释），返回每笔记融合分最高的块。
        """
        if self._matrix is None or query_vec is None:
            return []
        dense = self.search(query_vec, top_k=top_k * 3)
        bm25_scores = self.bm25.score(query)
        # BM25 单独取 top_k*3
        bm25_idx = np.argsort(bm25_scores)[::-1][: top_k * 3]
        bm25 = [
            {"text": self._texts[i], "source": self._sources[i],
             "score": float(bm25_scores[i]), "chunk_id": int(i)}
            for i in bm25_idx
        ]

        k_rrf = 60.0
        fusion = {}
        for rank, item in enumerate(dense + bm25):
            key = (item["source"], item["chunk_id"])   # 块级融合键
            fusion[key] = fusion.get(key, 0.0) + 1.0 / (rank + k_rrf)
        # 按融合分排序；每笔记最多 max_chunks_per_note 块
        ranked = sorted(fusion.items(), key=lambda kv: kv[1], reverse=True)
        by_chunk = {(item["source"], item["chunk_id"]): item for item in dense + bm25}
        per_note = {}
        results = []
        for (src, cid), rrf_score in ranked:
            if per_note.get(src, 0) >= max_chunks_per_note:
                continue
            item = by_chunk.get((src, cid))
            if item is None:
                continue
            per_note[src] = per_note.get(src, 0) + 1
            results.append({**item, "rrf": rrf_score})
            if len(results) >= top_k:
                break

        if not graph_expand or not self._note_links:
            return results

        # 2026-08-30 双向链接增强（ponytail 最小版）：
        # 1) 汇总型笔记（hub：出链多且内容薄，_build 时判定）过滤——与 index 页同族，rerank 易误判高分
        # 2) 一跳展开：非 hub 命中笔记的出链目标补进候选（每目标 1 块），multi-hop 找全
        hub_srcs = self._hub_sources if graph_expand else set()
        kept = [r for r in results if r["source"] not in hub_srcs]

        # 一跳展开（只从保留下来的命中笔记出发，每目标取融合分最高的块）
        if len(kept) < top_k:
            seen = {r["source"] for r in kept}
            added = []
            for r in kept[:2]:  # 只从前 2 个命中笔记展开，控制规模
                for target in self._note_links.get(r["source"], [])[:4]:  # 每笔记最多 4 目标
                    if target in seen:
                        continue
                    seen.add(target)
                    # 该目标笔记在 by_chunk 里融合分最高的块
                    best = max((item for item in by_chunk.values()
                                if item["source"] == target and item["source"] not in hub_srcs),
                               key=lambda it: fusion.get((it["source"], it["chunk_id"]), 0.0), default=None)
                    if best is not None:
                        added.append({**best, "rrf": fusion.get((best["source"], best["chunk_id"]), 0.0)})
            kept = kept + added
        return kept[:top_k]

    @property
    def size(self) -> int:
        return len(self._texts) if self._matrix is not None else 0
