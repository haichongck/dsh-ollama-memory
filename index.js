/**
 * dsh-ollama-memory —— 双层长期记忆：核心记忆文件 + 向量库（含结构化与治理）。
 *
 * 核心层：memory.md（默认 ≤1KB）只放称呼、位置、偏好这类核心事实，
 * 通过 systemPrompt 服务在每次组装时动态读入系统提示，模型开场即知。
 * 模型用 memory_core_set 整体重写该文件（超限拒写，逼它蒸馏压缩），
 * 用 memory_core_get 读取当前内容。
 *
 * 向量层：memory_save / memory_search / memory_list / memory_delete / memory_supersede
 * 操作 SQLite（node:sqlite），向量由 Ollama /api/embed 生成。检索用
 * 「余弦相似度 × 时间衰减 × 类别权重」排序，嵌入失败的条目不参与语义检索。
 *
 * 治理（学自 dsh-soul 的实测修复，见记忆 #156）：
 *   · 去重：存入前与既有条目比余弦，≥ dedupThreshold 视为同一件事，不再重复存；
 *   · 容量：超过 maxMemories 时把最旧的活跃条目归档（active=0），不物删；
 *   · 作废：memory_supersede 用新条目取代旧条目，旧条目标记 superseded_by 并退出检索；
 *   · 指纹：内容里出现文件路径时记 mtime，检索时比对，文件被改过就标 stale；
 *   · 水位：核心记忆写到 coreWarnRatio 以上时在工具结果里告警。
 *
 * 生命周期纪律：数据库连接与全部状态在 apply() 内创建，
 * 返回的 disposer 关闭连接并撤销系统提示段；模块顶层不持有可变状态。
 */
import { defineTool } from '@deepseek-ai/dsh-tools';
import z from '@deepseek-ai/schemastery';
import { DatabaseSync } from 'node:sqlite';
import { mkdirSync, readFileSync, writeFileSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import os from 'node:os';

export const name = 'ollama-memory';
export const inject = ['tools', 'systemPrompt'];

/** 记忆类别：决定检索权重，也决定「作废」的语义。 */
const KINDS = ['note', 'constraint', 'decision', 'artifact', 'progress'];
/** 各类别在检索里的权重（约束/决定更该被记住）。 */
const KIND_WEIGHT = { note: 1.0, constraint: 1.15, decision: 1.1, artifact: 0.95, progress: 1.0 };

/**
 * 配置的运行时校验 schema；默认值同时是文档。
 * 官方规则：dsh-settings 只把标了 .volatile() 的字段投影成设置表单
 * （见 dsh-settings volatileForm），所以全部标 volatile，装完即可在
 * 插件卡片上直接改 Ollama 地址、嵌入模型等。
 */
export const Config = z.object({
  /** Ollama 服务地址（OpenAI 兼容与原生 API 共用同一端口）。 */
  ollamaUrl: z.string().default('http://127.0.0.1:11434').volatile(),
  /** 嵌入模型名，需已在 Ollama 中拉取（ollama pull <model>）。 */
  embedModel: z.string().default('qwen3-embedding:0.6b').volatile(),
  /** SQLite 数据库路径，~ 展开为用户主目录。 */
  dbPath: z.string().default('~/.dsh/ollama-memory/memory.db').volatile(),
  /** memory_search 默认返回条数。 */
  topN: z.natural().default(5).volatile(),
  /** 核心记忆文件路径，~ 展开为用户主目录。 */
  corePath: z.string().default('~/.dsh/ollama-memory/memory.md').volatile(),
  /** 核心记忆文件字节上限，超过拒写。 */
  coreMaxBytes: z.natural().default(1024).volatile(),
  /** 活跃记忆条数上限；超出时把最旧的活跃条目标记为归档（active=0）。 */
  maxMemories: z.natural().default(2000).volatile(),
  /** 存入去重阈值：与既有条目的最高余弦 ≥ 该值就视为同一件事。 */
  dedupThreshold: z.number().default(0.95).volatile(),
  /** 核心记忆水位告警线（占 coreMaxBytes 的比例）。 */
  coreWarnRatio: z.number().default(0.8).volatile(),
  /** 检索时间衰减的半衰期（天）：越旧权重越低。 */
  timeDecayDays: z.number().default(180).volatile(),
});

function expandPath(p) {
  if (p.startsWith('~')) return join(os.homedir(), p.slice(1));
  return p;
}

/**
 * 调 Ollama 原生嵌入接口。
 * POST {url}/api/embed  {model, input: [texts]} → {embeddings: [[float]]}
 * 旧版 Ollama（<0.5）没有 /api/embed，回落到 /api/embeddings（单条 input）。
 */
async function embedTexts(baseUrl, model, texts, signal) {
  const root = baseUrl.replace(/\/+$/, '');
  /** 发一次嵌入请求；连接失败给出可执行的排查提示，不再只抛 "fetch failed"。 */
  const post = async (path, payload) => {
    try {
      return await fetch(`${root}${path}`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(payload),
        signal,
      });
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      throw new Error(
        `Ollama 嵌入服务不可达（${root}）：${detail}。` +
        `请确认 Ollama 正在运行，且模型已拉取（ollama pull ${model}）。`
      );
    }
  };
  let res = await post('/api/embed', { model, input: texts });
  if (res.ok) {
    const data = await res.json();
    if (Array.isArray(data.embeddings) && data.embeddings.length === texts.length) {
      return data.embeddings;
    }
    throw new Error('Ollama /api/embed 返回了意外的载荷结构');
  }
  const body = await res.text().catch(() => '');
  // 模型不存在要直接说清楚，不能当成「旧版接口」去回落（否则报错绕弯）。
  if (res.status === 404 && /not found|no such model|model/i.test(body)) {
    throw new Error(`Ollama 没有模型 ${model}：请先 ollama pull ${model}。响应：${body.slice(0, 200)}`);
  }
  if (res.status === 404) {
    // 旧版 Ollama（<0.5）没有 /api/embed，回落到 /api/embeddings（单条 input）
    const vectors = [];
    for (const text of texts) {
      const r2 = await post('/api/embeddings', { model, prompt: text });
      if (!r2.ok) {
        const b2 = await r2.text().catch(() => '');
        throw new Error(`Ollama /api/embeddings HTTP ${String(r2.status)}: ${b2.slice(0, 300)}`);
      }
      const d2 = await r2.json();
      if (!Array.isArray(d2.embedding)) throw new Error('Ollama /api/embeddings 返回了意外的载荷结构');
      vectors.push(d2.embedding);
    }
    return vectors;
  }
  throw new Error(`Ollama /api/embed HTTP ${String(res.status)}: ${body.slice(0, 300)}`);
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

/** 时间衰减：半衰期 timeDecayDays 天的指数衰减，1 天内≈1。 */
function timeDecay(createdAt, halfLifeDays) {
  const then = Date.parse(`${String(createdAt).replace(' ', 'T')}Z`);
  if (!Number.isFinite(then)) return 1;
  const ageDays = Math.max(0, (Date.now() - then) / 86_400_000);
  return Math.pow(0.5, ageDays / Math.max(1, halfLifeDays));
}

/**
 * 从内容里抽第一个像文件路径的片段（产物指纹用）。
 * 只认绝对路径与 ~ 开头，去掉中文标点尾巴。
 */
function extractArtifact(content) {
  const m = String(content).match(/(~|\/home\/[\w.-]+|\/Users\/[\w.-]+|\/tmp|\/mnt|\/workspace|\/srv|\/opt)[\w./@+-]*/);
  if (!m) return null;
  let p = m[0].replace(/[。，、；：)\]】]+$/, '');
  if (p.startsWith('~')) p = join(os.homedir(), p.slice(1));
  return p;
}

/** 产物指纹：返回 { path, mtime } 或 null；路径不存在时 mtime 为 null。 */
function artifactFingerprint(content) {
  const path = extractArtifact(content);
  if (!path) return null;
  try {
    return { path, mtime: statSync(path).mtimeMs };
  } catch {
    return { path, mtime: null };
  }
}

/** 已记录的产物是否已过时（文件被改过 / 被删）。 */
function artifactStale(row) {
  if (!row.artifact_path) return false;
  const fp = artifactFingerprint(row.artifact_path);
  if (!fp || fp.mtime === null) return true; // 文件不在了
  return row.artifact_mtime == null || Math.abs(fp.mtime - Number(row.artifact_mtime)) > 1;
}

function textBlock(text) {
  return [{ type: 'text', text }];
}

/**
 * 把查询切成可用于 LIKE 的词项（降级检索用）。
 * 英文/数字按分隔符切；纯中文再切二元（「北京路书」→ 北京/京路/路书），
 * 这样「北京路书」也能命中「北京五日路书已交付」。
 */
function queryTerms(query) {
  const terms = new Set();
  for (const raw of String(query).split(/[\s,，、;；:：.。!！?？()（）\[\]【】'"“”‘’]+/)) {
    if (!raw) continue;
    if (/^[\u4e00-\u9fff]+$/.test(raw)) {
      if (raw.length <= 2) terms.add(raw);
      else for (let i = 0; i < raw.length - 1; i += 1) terms.add(raw.slice(i, i + 2));
    } else {
      terms.add(raw.toLowerCase());
    }
  }
  return [...terms].slice(0, 12);
}

/** 按词项拼 OR 条件：每个词项命中 content 或 tags 任一即可。 */
function termCondition(query, alias) {
  const terms = queryTerms(query);
  if (terms.length === 0) return { sql: '0', params: [] };
  const parts = terms.map(() => `(content LIKE ? OR tags LIKE ?)`);
  const params = [];
  for (const t of terms) { params.push(`%${t}%`, `%${t}%`); }
  return { sql: parts.join(' OR '), params };
}

/**
 * 建表 / 迁移。老库只有 (id, content, tags, embedding, created_at)，
 * 新列用 ALTER TABLE 补齐，缺哪补哪，可重复执行。
 */
function migrate(db) {
  db.exec(`
    CREATE TABLE IF NOT EXISTS memories (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      content TEXT NOT NULL,
      tags TEXT NOT NULL DEFAULT '',
      embedding TEXT,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
  `);
  const cols = new Set(db.prepare('PRAGMA table_info(memories)').all().map((c) => c.name));
  const add = (name, ddl) => {
    if (!cols.has(name)) db.exec(`ALTER TABLE memories ADD COLUMN ${ddl}`);
  };
  add('kind', "kind TEXT NOT NULL DEFAULT 'note'");
  add('session', 'session TEXT');
  add('seq', 'seq INTEGER');
  add('superseded_by', 'superseded_by INTEGER');
  add('artifact_path', 'artifact_path TEXT');
  add('artifact_mtime', 'artifact_mtime INTEGER');
  add('active', 'active INTEGER NOT NULL DEFAULT 1');
  // 归档数据（active=0 且不是被作废的）在旧库里不存在，这里保证可查。
  db.exec('CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(active, id)');
}

/**
 * 注册插件：打开 SQLite，注册核心记忆层（系统提示段 + 读写工具）与向量层工具。
 *
 * @param ctx - Cordis 上下文，其 `tools` 注册表接收工具定义，`systemPrompt` 接收提示段。
 * @param config - 已由 {@link Config} 校验并填充默认值的配置。
 */
export function apply(ctx, config) {
  const dbPath = expandPath(config.dbPath);
  mkdirSync(dirname(dbPath), { recursive: true });
  const db = new DatabaseSync(dbPath);
  migrate(db);

  const baseUrl = config.ollamaUrl;
  const model = config.embedModel;
  const defaultTopN = config.topN ?? 5;
  const corePath = expandPath(config.corePath);
  const coreMaxBytes = config.coreMaxBytes ?? 1024;
  const maxMemories = Math.max(1, config.maxMemories ?? 2000);
  const dedupThreshold = Math.min(1, Math.max(0, config.dedupThreshold ?? 0.95));
  const coreWarnRatio = Math.min(1, Math.max(0, config.coreWarnRatio ?? 0.8));
  const halfLifeDays = Math.max(1, config.timeDecayDays ?? 180);

  const disposers = [];

  /** 读取核心记忆，返回内容/字节/水位。 */
  function readCore() {
    try {
      const content = readFileSync(corePath, 'utf8');
      return { content, bytes: Buffer.byteLength(content, 'utf8') };
    } catch {
      return { content: '', bytes: 0 };
    }
  }

  /** 水位告警文案（超过 coreWarnRatio 时给一句可执行的提醒）。 */
  function coreWarning(bytes) {
    if (bytes < Math.floor(coreMaxBytes * coreWarnRatio)) return '';
    const pct = Math.round((bytes / coreMaxBytes) * 100);
    return `⚠️ 核心记忆已用 ${String(bytes)}/${String(coreMaxBytes)} 字节（${String(pct)}%）。` +
      '建议用 memory_core_get 看一遍，把仍有效的核心事实蒸馏压缩后整体覆盖；细节请挪到 memory_save。';
  }

  /** 容量治理：活跃条数超过上限时，把最旧的活跃条目标记为归档（不物删）。 */
  function enforceCapacity() {
    const { c } = db.prepare('SELECT COUNT(*) AS c FROM memories WHERE active = 1').get();
    if (c <= maxMemories) return 0;
    const overflow = c - maxMemories;
    const victims = db
      .prepare('SELECT id FROM memories WHERE active = 1 ORDER BY id ASC LIMIT ?')
      .all(overflow);
    const mark = db.prepare('UPDATE memories SET active = 0 WHERE id = ?');
    for (const v of victims) mark.run(Number(v.id));
    return victims.length;
  }

  // ---------- 核心记忆层：系统提示段 ----------
  // 每次组装系统提示时同步读文件；空文件或不存在时贡献空文本（不占位）。
  // order 取大值，排在内置段之后，不打乱核心提示顺序。
  disposers.push(ctx.systemPrompt.section({
    name: 'ollama-memory-core',
    order: 1000,
    text: () => {
      const { content } = readCore();
      const trimmed = content.trim();
      if (!trimmed) return '';
      return `\n## 长期核心记忆\n以下是关于用户的长期核心事实，始终适用：\n${trimmed}\n`;
    },
  }));

  // ---------- memory_core_get ----------
  disposers.push(ctx.tools.register(defineTool({
    name: 'memory_core_get',
    description: '读取核心记忆文件（memory.md）的当前全文与水位。核心记忆是称呼、位置、偏好这类' +
      '必须始终在场的事实，已自动注入系统提示；重写前先用它看当前内容。结果含 ok、content、bytes、maxBytes、ratio 与 error（失败时）。',
    parameters: {},
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          ok: { type: 'boolean', required: true },
          content: { type: 'string', required: true },
          bytes: { type: 'number', required: true },
          maxBytes: { type: 'number', required: true },
          ratio: { type: 'number', required: true },
          warn: { type: 'string' },
          error: { type: 'string' },
        },
      },
      render: (_args, value) => textBlock(
        !value.ok
          ? `读取核心记忆失败：${value.error ?? '未知错误'}`
          : `${value.content ? `核心记忆（${String(value.bytes)}/${String(value.maxBytes)} 字节，${String(Math.round(value.ratio * 100))}%）：\n${value.content}` : '核心记忆文件还是空的。'}${value.warn ? `\n${value.warn}` : ''}`
      ),
    },
    timeoutMs: 10_000,
    isConcurrencySafe: () => true,
    async execute() {
      try {
        const { content, bytes } = readCore();
        return {
          ok: true,
          content,
          bytes,
          maxBytes: coreMaxBytes,
          ratio: coreMaxBytes > 0 ? bytes / coreMaxBytes : 0,
          warn: coreWarning(bytes),
        };
      } catch (error) {
        return { ok: false, content: '', bytes: 0, maxBytes: coreMaxBytes, ratio: 0, error: error instanceof Error ? error.message : String(error) };
      }
    },
  })));

  // ---------- memory_core_set ----------
  disposers.push(ctx.tools.register(defineTool({
    name: 'memory_core_set',
    description: `整体重写核心记忆文件（memory.md），上限 ${String(coreMaxBytes)} 字节，超限拒写。` +
      '只放称呼、位置、偏好、长期约定这类必须每轮在场的事实；细节、历史、过程一律用 memory_save 进向量库。' +
      '重写前先 memory_core_get 看当前内容，把仍有效的核心事实蒸馏压缩后整体覆盖。' +
      '结果含 ok、bytes、maxBytes、ratio、warn 与 error（超限时 error 说明当前与上限字节数）。',
    parameters: {
      content: { type: 'string', required: true, description: '核心记忆文件的新全文（蒸馏压缩后的核心事实）。' },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          ok: { type: 'boolean', required: true },
          bytes: { type: 'number' },
          maxBytes: { type: 'number', required: true },
          ratio: { type: 'number', required: true },
          warn: { type: 'string' },
          error: { type: 'string' },
        },
      },
      render: (_args, value) => textBlock(
        value.ok
          ? `核心记忆已更新（${String(value.bytes)}/${String(value.maxBytes)} 字节，${String(Math.round(value.ratio * 100))}%）。${value.warn ? `\n${value.warn}` : ''}`
          : `核心记忆未写入：${value.error ?? '未知错误'}`
      ),
    },
    timeoutMs: 10_000,
    async execute(args) {
      const bytes = Buffer.byteLength(args.content, 'utf8');
      if (bytes > coreMaxBytes) {
        return {
          ok: false,
          maxBytes: coreMaxBytes,
          ratio: bytes / coreMaxBytes,
          error: `内容 ${String(bytes)} 字节，超过上限 ${String(coreMaxBytes)} 字节。请蒸馏压缩：合并同类项、删过时事实、只留核心。`,
        };
      }
      try {
        mkdirSync(dirname(corePath), { recursive: true });
        writeFileSync(corePath, args.content, 'utf8');
        return {
          ok: true,
          bytes,
          maxBytes: coreMaxBytes,
          ratio: bytes / coreMaxBytes,
          warn: coreWarning(bytes),
        };
      } catch (error) {
        return { ok: false, maxBytes: coreMaxBytes, ratio: 0, error: error instanceof Error ? error.message : String(error) };
      }
    },
  })));

  // ---------- memory_save ----------
  disposers.push(ctx.tools.register(defineTool({
    name: 'memory_save',
    description: '把一条事实、偏好或结论存入向量长期记忆（Ollama 嵌入 + SQLite 存储），跨会话可用，按需语义检索。' +
      '适合项目约定、重要决定、踩过的坑、过程细节。内容应自包含、一句话能说清；tags 用逗号分隔的关键词（可空）。' +
      `kind 取 ${KINDS.join(' / ')}（约束与决定检索权重更高）。与既有条目余弦 ≥ ${String(dedupThreshold)} 时视为同一件事，不再重复存。` +
      '称呼、位置、偏好这类必须每轮在场的核心事实改用 memory_core_set 写核心记忆文件。' +
      '结果含 ok、id、duplicated 与 error（失败时）。',
    parameters: {
      content: { type: 'string', required: true, description: '要记住的内容，自包含的一句话或短段落。' },
      tags: { type: 'string', description: '逗号分隔的关键词标签，可省略。' },
      kind: { type: 'string', enum: [...KINDS], description: `记忆类别，默认 note。${KINDS.join(' / ')}。`, default: 'note' },
      session: { type: 'string', description: '来源会话 id（可选，便于日后按会话回捞）。' },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          ok: { type: 'boolean', required: true },
          id: { type: 'number' },
          duplicated: { type: 'boolean', required: true },
          archived: { type: 'number' },
          error: { type: 'string' },
        },
      },
      render: (_args, value) => textBlock(
        !value.ok
          ? `存入记忆失败：${value.error ?? '未知错误'}`
          : value.duplicated
            ? `已存在相同记忆（#${String(value.id)}），未重复存入。`
            : `已存入记忆 #${String(value.id)}。${value.archived ? `\n容量治理：归档了 ${String(value.archived)} 条最旧的活跃记忆。` : ''}`
      ),
    },
    timeoutMs: 30_000,
    async execute(args, exec) {
      try {
        const content = String(args.content);
        const [vector] = await embedTexts(baseUrl, model, [content], exec.signal);

        // 去重：与既有活跃条目比余弦，命中就不再存（省得同一件事堆几十遍）。
        const existing = db
          .prepare('SELECT id, embedding FROM memories WHERE active = 1 AND superseded_by IS NULL')
          .all();
        let best = null;
        let bestScore = -1;
        for (const row of existing) {
          if (!row.embedding) continue;
          let v;
          try {
            v = JSON.parse(row.embedding);
          } catch {
            continue; // 损坏的向量行跳过，不让单条坏数据毁掉整次去重
          }
          const s = cosine(vector, v);
          if (s > bestScore) { bestScore = s; best = row; }
        }
        if (best && bestScore >= dedupThreshold) {
          return { ok: true, id: Number(best.id), duplicated: true, archived: 0 };
        }

        const fp = artifactFingerprint(content);
        const info = db
          .prepare(
            'INSERT INTO memories (content, tags, embedding, kind, session, artifact_path, artifact_mtime, active) ' +
            'VALUES (?, ?, ?, ?, ?, ?, ?, 1)'
          )
          .run(
            content,
            args.tags ?? '',
            JSON.stringify(vector),
            KINDS.includes(args.kind) ? args.kind : 'note',
            args.session ?? null,
            fp ? fp.path : null,
            fp && fp.mtime != null ? Math.round(fp.mtime) : null
          );
        const archived = enforceCapacity();
        return { ok: true, id: Number(info.lastInsertRowid), duplicated: false, archived };
      }
      catch (error) {
        return { ok: false, duplicated: false, archived: 0, error: error instanceof Error ? error.message : String(error) };
      }
    },
  })));

  // ---------- memory_supersede ----------
  disposers.push(ctx.tools.register(defineTool({
    name: 'memory_supersede',
    description: '用一条新记忆取代旧记忆（作废语义）：把新内容作为新条目写入，并把旧条目标记为已作废（superseded_by 指向新条目，退出检索）。' +
      '旧条目不物理删除，memory_list 仍可看到它的作废状态。适用于「方案改了 / 结论被推翻 / 进度更新」。' +
      '结果含 ok、oldId、newId 与 error（失败时）。',
    parameters: {
      oldId: { type: 'number', required: true, description: '被取代的旧记忆 id。' },
      content: { type: 'string', required: true, description: '取代它的新内容（自包含）。' },
      tags: { type: 'string', description: '逗号分隔的关键词标签，可省略。' },
      kind: { type: 'string', enum: [...KINDS], description: '新记忆的类别，默认沿用旧条目的类别。' },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          ok: { type: 'boolean', required: true },
          oldId: { type: 'number', required: true },
          newId: { type: 'number' },
          error: { type: 'string' },
        },
      },
      render: (_args, value) => textBlock(
        value.ok
          ? `已用 #${String(value.newId)} 取代记忆 #${String(value.oldId)}（旧条目已作废，不再参与检索）。`
          : `取代失败：${value.error ?? '未知错误'}`
      ),
    },
    timeoutMs: 30_000,
    async execute(args, exec) {
      try {
        const old = db.prepare('SELECT id, kind FROM memories WHERE id = ?').get(args.oldId);
        if (!old) return { ok: false, oldId: args.oldId, error: `没有 id 为 ${String(args.oldId)} 的记忆` };
        const content = String(args.content);
        const [vector] = await embedTexts(baseUrl, model, [content], exec.signal);
        const fp = artifactFingerprint(content);
        const kind = KINDS.includes(args.kind) ? args.kind : String(old.kind || 'note');
        const info = db
          .prepare(
            'INSERT INTO memories (content, tags, embedding, kind, artifact_path, artifact_mtime, active) ' +
            'VALUES (?, ?, ?, ?, ?, ?, 1)'
          )
          .run(
            content,
            args.tags ?? '',
            JSON.stringify(vector),
            kind,
            fp ? fp.path : null,
            fp && fp.mtime != null ? Math.round(fp.mtime) : null
          );
        const newId = Number(info.lastInsertRowid);
        db.prepare('UPDATE memories SET superseded_by = ?, active = 0 WHERE id = ?').run(newId, Number(args.oldId));
        const archived = enforceCapacity();
        if (archived > 0) {
          // 归档只针对活跃条目，这里仅作信息透出。
        }
        return { ok: true, oldId: Number(args.oldId), newId };
      }
      catch (error) {
        return { ok: false, oldId: Number(args.oldId), error: error instanceof Error ? error.message : String(error) };
      }
    },
  })));

  // ---------- memory_search ----------
  disposers.push(ctx.tools.register(defineTool({
    name: 'memory_search',
    description: '按语义检索本机长期记忆：用 Ollama 嵌入模型对查询做向量化，与已存记忆比对后按 ' +
      '「余弦相似度 × 时间衰减 × 类别权重」排序（约束/决定权重更高，越旧越低）。' +
      '已作废（被取代/归档）的条目不参与检索。内容里带文件路径的条目会标注是否已过时（文件被改过/被删）。' +
      '回答涉及用户偏好、历史决定、项目背景的问题前先搜一下。结果含 ok、query、items（id/content/tags/kind/score/createdAt/stale）与 error（失败时）。',
    parameters: {
      query: { type: 'string', required: true, description: '自然语言查询。' },
      limit: { type: 'integer', description: `返回条数，默认 ${String(defaultTopN)}。`, default: defaultTopN },
      includeArchived: { type: 'boolean', description: '连已归档/已作废的条目一起返回，默认 false。', default: false },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          ok: { type: 'boolean', required: true },
          query: { type: 'string', required: true },
          degraded: { type: 'boolean', description: '嵌入服务不可用，本次退到关键词匹配。', default: false },
          items: {
            type: 'array',
            required: true,
            items: {
              type: 'object',
              additionalProperties: false,
              properties: {
                id: { type: 'number', required: true },
                content: { type: 'string', required: true },
                tags: { type: 'string', required: true },
                kind: { type: 'string', required: true },
                score: { type: 'number', required: true },
                createdAt: { type: 'string', required: true },
                stale: { type: 'boolean', required: true },
              },
            },
          },
          error: { type: 'string' },
        },
      },
      render: (_args, value) => {
        if (!value.ok) return textBlock(`记忆检索失败：${value.error ?? '未知错误'}`);
        if (value.degraded) {
          const lines = value.items.map((it) =>
            `#${String(it.id)} (${it.kind}${it.stale ? '，产物可能已过时' : ''}) ${it.content}${it.tags ? `（${it.tags}）` : ''}`
          );
          return textBlock(
            `⚠️ 嵌入服务不可用，已降级为关键词匹配（${value.error}）。\n` +
            (lines.length ? `与 "${value.query}" 相关的记忆：\n${lines.join('\n')}` : `没有匹配到 "${value.query}"。`)
          );
        }
        if (value.items.length === 0) return textBlock(`未找到与 "${value.query}" 相关的记忆。`);
        const lines = value.items.map((it) =>
          `#${String(it.id)} [${it.score.toFixed(3)}] (${it.kind}${it.stale ? '，产物可能已过时' : ''}) ${it.content}${it.tags ? `（${it.tags}）` : ''}`
        );
        return textBlock(`与 "${value.query}" 相关的记忆：\n${lines.join('\n')}`);
      },
    },
    timeoutMs: 30_000,
    isConcurrencySafe: () => true,
    async execute(args, exec) {
      const limit = Math.max(1, Math.min(args.limit ?? defaultTopN, 50));
      const includeArchived = args.includeArchived === true;
      let queryVector;
      try {
        [queryVector] = await embedTexts(baseUrl, model, [args.query], exec.signal);
      } catch (error) {
        // 嵌入不可用（Ollama 挂了 / 模型没拉）时降级到关键词匹配，至少还能搜到东西。
        const message = error instanceof Error ? error.message : String(error);
        try {
          const cond = termCondition(args.query);
          const base = 'SELECT id, content, tags, created_at, kind, artifact_path, artifact_mtime FROM memories WHERE ';
          const tail = ' ORDER BY id DESC LIMIT ?';
          const scope = includeArchived ? '' : 'active = 1 AND superseded_by IS NULL AND ';
          const rows = cond.sql === '0'
            ? []
            : db.prepare(base + scope + '(' + cond.sql + ')' + tail).all(...cond.params, limit);
          return {
            ok: true,
            query: args.query,
            degraded: true,
            error: message,
            items: rows.map((r) => ({
              id: Number(r.id),
              content: r.content,
              tags: r.tags,
              kind: r.kind,
              score: 0,
              createdAt: r.created_at,
              stale: artifactStale(r),
            })),
          };
        } catch {
          return { ok: false, query: args.query, degraded: true, items: [], error: message };
        }
      }
      try {
        const rows = db
          .prepare(
            'SELECT id, content, tags, embedding, created_at, kind, artifact_path, artifact_mtime ' +
            'FROM memories' +
            (includeArchived ? ' ORDER BY id' : ' WHERE active = 1 AND superseded_by IS NULL ORDER BY id')
          )
          .all();
        const scored = [];
        for (const row of rows) {
          if (!row.embedding) continue; // 嵌入失败的条目不参与语义检索
          let vector;
          try {
            vector = JSON.parse(row.embedding);
          } catch {
            continue; // 损坏的向量行跳过，不让单条坏数据毁掉整次检索
          }
          const sim = cosine(queryVector, vector);
          const decay = timeDecay(row.created_at, halfLifeDays);
          const weight = KIND_WEIGHT[row.kind] ?? 1;
          scored.push({
            id: Number(row.id),
            content: row.content,
            tags: row.tags,
            kind: row.kind,
            score: sim * decay * weight,
            similarity: sim,
            createdAt: row.created_at,
            stale: artifactStale(row),
          });
        }
        scored.sort((a, b) => b.score - a.score);
        const items = scored.slice(0, limit).map(({ similarity, ...rest }) => rest);
        return { ok: true, query: args.query, degraded: false, items };
      }
      catch (error) {
        return { ok: false, query: args.query, items: [], error: error instanceof Error ? error.message : String(error) };
      }
    },
  })));

  // ---------- memory_list ----------
  disposers.push(ctx.tools.register(defineTool({
    name: 'memory_list',
    description: '按时间倒序列出本机长期记忆（不做语义排序），含类别与作废/归档状态。' +
      '结果含 ok、items（id/content/tags/kind/active/supersededBy/createdAt）与 error（失败时）。',
    parameters: {
      limit: { type: 'integer', description: '返回条数，默认 20。', default: 20 },
      all: { type: 'boolean', description: '连已归档/已作废的一并列出来，默认 false。', default: false },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          ok: { type: 'boolean', required: true },
          items: {
            type: 'array',
            required: true,
            items: {
              type: 'object',
              additionalProperties: false,
              properties: {
                id: { type: 'number', required: true },
                content: { type: 'string', required: true },
                tags: { type: 'string', required: true },
                kind: { type: 'string', required: true },
                active: { type: 'boolean', required: true },
                supersededBy: { type: 'number' },
                createdAt: { type: 'string', required: true },
              },
            },
          },
          error: { type: 'string' },
        },
      },
      render: (_args, value) => {
        if (!value.ok) return textBlock(`列出记忆失败：${value.error ?? '未知错误'}`);
        if (value.items.length === 0) return textBlock('记忆库还是空的。');
        const lines = value.items.map((it) =>
          `#${String(it.id)} [${it.kind}]${it.active ? '' : (it.supersededBy ? ` 已作废→#${String(it.supersededBy)}` : ' 已归档')} ${it.createdAt} ${it.content}${it.tags ? `（${it.tags}）` : ''}`
        );
        return textBlock(`最近的记忆：\n${lines.join('\n')}`);
      },
    },
    timeoutMs: 10_000,
    isConcurrencySafe: () => true,
    async execute(args) {
      try {
        const limit = Math.max(1, Math.min(args.limit ?? 20, 200));
        const rows = db
          .prepare(
            'SELECT id, content, tags, kind, active, superseded_by, created_at FROM memories' +
            (args.all === true ? ' ORDER BY id DESC LIMIT ?' : ' WHERE active = 1 AND superseded_by IS NULL ORDER BY id DESC LIMIT ?')
          )
          .all(limit);
        return {
          ok: true,
          items: rows.map((r) => ({
            id: Number(r.id),
            content: r.content,
            tags: r.tags,
            kind: r.kind,
            active: Number(r.active) === 1,
            ...(r.superseded_by != null ? { supersededBy: Number(r.superseded_by) } : {}),
            createdAt: r.created_at,
          })),
        };
      }
      catch (error) {
        return { ok: false, items: [], error: error instanceof Error ? error.message : String(error) };
      }
    },
  })));

  // ---------- memory_delete ----------
  disposers.push(ctx.tools.register(defineTool({
    name: 'memory_delete',
    description: '按 id 物理删除一条本机长期记忆（不可恢复）。仅是作废/归档请改用 memory_supersede 或等容量治理。' +
      'id 来自 memory_search / memory_list 的结果。结果含 ok、id 与 error（失败时）。',
    parameters: {
      id: { type: 'number', required: true, description: '要删除的记忆 id。' },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          ok: { type: 'boolean', required: true },
          id: { type: 'number', required: true },
          error: { type: 'string' },
        },
      },
      render: (_args, value) => textBlock(
        value.ok ? `已删除记忆 #${String(value.id)}。` : `删除记忆失败：${value.error ?? '未知错误'}`
      ),
    },
    timeoutMs: 10_000,
    async execute(args) {
      try {
        const info = db.prepare('DELETE FROM memories WHERE id = ?').run(args.id);
        if (Number(info.changes) === 0) {
          return { ok: false, id: args.id, error: `没有 id 为 ${String(args.id)} 的记忆` };
        }
        return { ok: true, id: args.id };
      }
      catch (error) {
        return { ok: false, id: args.id, error: error instanceof Error ? error.message : String(error) };
      }
    },
  })));

  return () => {
    for (const dispose of disposers) {
      try {
        dispose();
      }
      catch {
        // 卸载路径不因单个撤销失败而中断。
      }
    }
    try {
      db.close();
    }
    catch {
      // 同上。
    }
  };
}
