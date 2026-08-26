# vaultrag — Vault RAG Context Engine

个人知识库检索插件：把 Obsidian vault 变成 Hermes 的"外接知识库"。用户提问时，
先向量检索 vault 中相关笔记片段，注入请求上下文，模型基于自己的笔记回答。

## 一句话原理

```
用户提问
  → [长度门槛] <2 字直接跳过（"好""嗯"这类 1 字确认消息；英文缩写"MoA"不误杀）
  → 混合检索：向量 top-48 + BM25 top-48 → RRF 融合 → 召回 16
  → Rerank 精排（cross-encoder，bge-reranker-v2-m3）→ top-4
  → [三档评估] CRAG 式：top1 ≥0.5 Correct / 0.3~0.5 Ambiguous / <0.3 Incorrect
  → 命中片段拼成 <knowledge_context> 注入请求（独立 user 消息）
  → 模型基于笔记回答（自带来源标注）
  → 每次调用写一条结构化 trace（JSONL）到 vault/.smart-env/vaultrag/trace.jsonl
  → 任何失败 → 返回 None → 原样放行（fail-open）
```

## 安装 / 启用

1. 插件位于 `plugins/context_engine/vaultrag/`（repo-shipped 目录，Hermes 自动发现）
2. `.env` 配置密钥：

   ```
   EMBEDDING_API_KEY=sk-xxx          # 硅基流动（bge-m3 免费）或任意 OpenAI 兼容端点
   ```

3. `config.yaml` 配置行为（vault 路径 + embedding 服务）：

   ```yaml
   context:
     engine: vaultrag
     vaultrag:
       vault_path: D:/llmwiki/llm-wiki     # Obsidian vault 路径
       embedding:
         base_url: https://api.siliconflow.cn/v1   # OpenAI 兼容端点
         model: BAAI/bge-m3                          # embedding 模型
       ```

4. 重启 Hermes，生效。

配置读取优先级：显式参数 > config.yaml (`context.vaultrag.*`) > 环境变量（`OBSIDIAN_VAULT_PATH` / `EMBEDDING_*`，向后兼容）。行为配置走 config.yaml，`.env` 只放密钥（Hermes 插件规范）。

## 设计决策（面试可讲）

| 决策 | 选择 | 为什么 |
|------|------|--------|
| embedding | 云端 API，不跑本地小模型 | 对个人用户零负担：不下载模型、不占内存、不装 sentence-transformers 全家桶 |
| provider 抽象 | OpenAI 兼容接口 + .env 配置 | 换厂商（硅基/智谱/阿里/OpenAI）只改配置，代码不动 |
| 检索管线 | 混合检索（BM25+向量）+ RRF 融合 + cross-encoder rerank 精排 | 行业标准架构（arXiv 2604.01733：两阶段管线 Recall@5 0.695→0.816）；BM25 补精确匹配，rerank 补语义精排 |
| 噪音过滤 | 长度门槛 + BM25 源头过滤 + rerank 分数阈值 | 三层防线：短确认消息直接跳过；"好"这类词 BM25 分数趋零召回不到；长句无关由 rerank 压分拦截 |
| 向量检索 | numpy 矩阵点积，不用 chromadb/faiss | 笔记规模（千级块）下足够；插件零重依赖（仅 numpy+openai，Hermes venv 自带） |
| 分块 | 按 Markdown 标题（##/###）切块 | 块自带上下文；命中返回"笔记路径+块内容"，模型可引用出处 |
| 索引持久化 | .npz 缓存 + 文件哈希指纹 | vault 没变就加载缓存，不重复调 API；变了自动重建 |
| 注入位置 | 独立 user 消息，不碰 system/历史 | 保住 prompt cache 契约（system 字节稳定），列表只用于本次请求 |
| 压缩委托 | should_compress/compress 委托内置 ContextCompressor | 插件只管"选择"上下文，压缩策略不重复实现 |

## 实测记录（2026-08-16）

### 检索质量升级对比（纯向量 vs 混合+rerank）

| 查询 | 旧方案（纯向量） | 新方案（混合+rerank） |
|------|----------------|---------------------|
| "好"（短确认） | ❌ 注入噪音（0.536 命中飞书笔记） | ✅ 长度门槛拦截 |
| "帮我写个Java程序"（长句无关） | ❌ 会注入 | ✅ rerank 压到 0.030 < 0.30，拦截 |
| "MoA是什么"（术语） | 一般 | ✅ top1=0.835（BM25 精确匹配） |
| "压缩四道闸"（正常提问） | top=0.537 | ✅ top1=0.908（rerank 精排） |

### 端到端注入（2026-08-16）

```
20:18:58 [vaultrag] injected 4 hits (query='空响应7级阶梯是什么', top=0.537)
```

命中 hermes-compression-per-turn-state-reset.md / hermes-pre-api-pressure-check.md 等
深度笔记，回答可直接引用笔记结构。索引规模：91 篇笔记 → 984 块（已排除 raw/copilot）。

## 已知限制 / 优化方向

- **索引不自动增量**：vault 新增/修改笔记后，下次请求时哈希变化自动重建
  （首次构建需全量 embedding，几百块约 1-2 分钟）。
- **rerank 增加一次 API 调用**：每回合多 1 次 /rerank（bge-reranker 免费，毫秒级），
  对免费模型无感知，付费模型可考虑仅对 >N 字提问触发。
- **BM25 分词简单**：中文按字符 bigram（未用 jieba），术语匹配够用，
  追求极致可换 jieba 分词（+1 依赖）。

## 排障记录（2026-08-17）

### ⚠ 压缩被禁用的坑（已修复）

**现象**：启用 vaultrag 后，上下文超过压缩阈值（50 万）不压缩。

**根因（两层）**：
1. host 选外部引擎后把 `agent.context_compressor` 整体替换成插件引擎
   （agent_init.py:2566），注释明说 "External engines own compaction policy"——
   **host 不负责给外部引擎挂压缩委托**，`attach_delegate` 全仓无调用方。
   插件 `_delegate=None` → `should_compress` 恒 False、`compress` 原样返回
   = 压缩被完全禁用。
2. `ContextCompressor.__init__` 必传 `model` 且参数名是 `config_context_length`
   （不是 host update_model 的 `context_length`），无参/透传实例化都会炸。

**修复**：懒创建委托——在 host 必调的 `update_model()` 里用真实参数
实例化内置 ContextCompressor（`context_length`→`config_context_length` 映射），
并转发 `update_model` / `bind_session_state` / `on_session_start` /
`update_from_response` / `should_compress` / `compress` 全套接口，
让压缩策略与内置引擎完全一致。

**验证**：模拟 host 调用链（update_model → update_from_response(550K) →
should_compress → compress），9 项断言全过，550K 正确触发压缩。

**教训**：写 context engine 插件时，压缩职责必须自己扛——host 只调
update_model 传参数，其余全靠插件实现。查"压缩失效"先确认
`should_compress` 是否基于真实用量（update_from_response 是否转发）。

### ⚠ preflight 显示 ">= 0 threshold (ctx 0)" 的坑（已修复）

**现象**：压缩实际正常触发（日志 `Context compression triggered (589773 >= 500000)`），
但 UI/preflight 提示显示 `~589,773 tokens >= 0 threshold (ctx 0)`。

**根因**：`ContextEngine` 基类把 `threshold_tokens` / `context_length` 定义为
**类属性默认 0**（context_engine.py:106-107）。插件引擎继承基类但不覆盖，
turn_context preflight 读 `agent.context_compressor.threshold_tokens`
（turn_context.py:976）时拿到基类默认 0——**功能没坏（should_compress 走
委托真实值），显示撒谎**。`protect_last_n` 同理会拿基类默认 6（真实 20），
影响 preflight 尾部保护估算。

**修复**：引擎加 4 个属性转发（`threshold_tokens` / `context_length` /
`protect_first_n` / `protect_last_n`）→ 委托。验证：500000 / 1000000 / 3 / 20
全对，9 项回归全过。

**教训**：写 ContextEngine 插件时，基类的**类属性默认值**（不只是方法）
也要检查——preflight/UI 直接读这些属性做显示与预检，默认 0 会误导。
