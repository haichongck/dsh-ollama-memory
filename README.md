# dsh-ollama-memory

[中文](#中文) · [English](#english)

## 中文

给 [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)（dsh）加**跨会话长期记忆**的插件，全程本地、不联网：

**两层结构**

| 层 | 载体 | 特点 |
|---|---|---|
| 核心记忆 | `~/.dsh/ollama-memory/memory.md` | 称呼、偏好等必须每轮在场的事实，自动注入系统提示；有字节上限，超限拒写并告警 |
| 向量库 | `~/.dsh/ollama-memory/memory.db`（SQLite） | 项目约定、决定、踩坑、进度；用本机 Ollama 嵌入模型向量化，按需语义检索 |

**检索排序** = 余弦相似度 × 时间衰减（半衰期可配）× 类别权重（约束/决定权重更高）。

**记忆治理**

- 结构化类别：`note / constraint / decision / artifact / progress`
- 存入去重：与既有最高余弦 ≥ 阈值（默认 0.95）视为同一件事，不重复存
- 作废取代：`memory_supersede` 用新条目取代旧条目，旧条目退出检索但可查
- 容量治理：活跃条目超上限自动归档最旧的
- 产物过时检测：记忆内容里的文件路径被改动/删除时，检索结果标注 `stale`

**工具面**：`memory_save / memory_search / memory_list / memory_supersede / memory_delete / memory_core_get / memory_core_set`

### 依赖

- **Ollama**（本机服务，默认 `http://127.0.0.1:11434`）
- 一个已拉取的嵌入模型，默认 `qwen3-embedding:0.6b`：`ollama pull qwen3-embedding:0.6b`
- Node ≥ 22.5（内置 `node:sqlite`）

### 安装

dsh 0.2.1+：**插件 → 发现**里添加市场索引源后搜索 `dsh-ollama-memory` 安装；或手动：

```sh
dsh plugin --profile <p> add npm:dsh-ollama-memory
```

### 配置（安装后在插件卡片上直接改）

| 字段 | 默认 | 说明 |
|---|---|---|
| `ollamaUrl` | `http://127.0.0.1:11434` | Ollama 地址 |
| `embedModel` | `qwen3-embedding:0.6b` | 嵌入模型（需已 pull） |
| `dbPath` / `corePath` | `~/.dsh/ollama-memory/…` | 向量库 / 核心记忆路径 |
| `topN` | 5 | 检索默认条数 |
| `coreMaxBytes` | 1024 | 核心记忆字节上限 |
| `maxMemories` | 2000 | 活跃条目上限（超出归档最旧） |
| `dedupThreshold` | 0.95 | 存入去重余弦阈值 |
| `coreWarnRatio` | 0.8 | 核心记忆水位告警线 |
| `timeDecayDays` | 180 | 检索时间衰减半衰期（天） |

Ollama 不可用时所有工具返回 `{ok:false, error}` 并说明原因，不影响会话本身。

MIT License.

## English

Two-layer long-term memory for DeepSeek Harness, fully local: a small **core memory** file (always injected into the system prompt, byte-capped) plus a **SQLite vector store** embedded by a local Ollama model. Retrieval ranks by cosine × time-decay × kind-weight (`constraint`/`decision` weighted higher). Includes save-time dedup, supersede semantics (new entry retires the old one), capacity governance (archive oldest over limit) and stale-file detection in memory contents. Requires Ollama with an embedding model (default `qwen3-embedding:0.6b`) and Node ≥ 22.5. All settings are editable on the plugin card after install. MIT licensed.
