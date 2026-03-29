# Análise de Referência: qmd

> Implementação de referência em `referencias/qmd/`
> Todos os caminhos relativos a `referencias/qmd/`

---

## Visão Geral

**qmd** é uma aplicação CLI/MCP de busca semântica sobre documentos locais (markdown, código, notas). Diferente do `node-llama-cpp` (que é uma biblioteca de bindings), o qmd é um **produto final** — uma ferramenta de RAG (Retrieval-Augmented Generation) completa que usa `node-llama-cpp` como dependência para todas as operações de LLM.

### Stack Tecnológica

| Camada | Tecnologia |
|---|---|
| Runtime | Bun / Node.js (cross-runtime) |
| LLM Backend | `node-llama-cpp` (alto nível, gerencia GGML internamente) |
| Banco de dados | SQLite (`bun:sqlite` / `better-sqlite3`) |
| Busca vetorial | `sqlite-vec` (extensão SQLite) |
| Busca textual | FTS5 (tokenizer Porter + Unicode61) |
| Configuração | YAML (`~/.config/qmd/index.yml`) |

### Diferenças Fundamentais vs bun-llama-cpp

| Aspecto | qmd | bun-llama-cpp |
|---|---|---|
| Natureza | Aplicação (CLI + MCP server) | Biblioteca (FFI bindings) |
| FFI | `node-llama-cpp` high-level API | Raw `bun:ffi` + C shims |
| Threading | Múltiplos embedding contexts | Single Worker + serial queue |
| Escopo | RAG completo (embed + rerank + search) | Inferência de texto |

O valor do qmd como referência está na **orquestração de pipeline** — como coordena três modelos separados, gerencia ciclo de vida com ref-counting, e implementa busca híbrida com reranking.

---

## Arquitetura de LLM

### Três Modelos Separados

O qmd opera com três modelos GGUF distintos, cada um com lifecycle independente:

```typescript
// src/llm.ts:196-199
const DEFAULT_EMBED_MODEL = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";
const DEFAULT_RERANK_MODEL = "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf";
const DEFAULT_GENERATE_MODEL = "hf:tobil/qmd-query-expansion-1.7B-gguf/qmd-query-expansion-1.7B-q4_k_m.gguf";
```

| Modelo | Default | Propósito | Tamanho |
|---|---|---|---|
| Embed | embeddinggemma-300M Q8 | Embeddings de documentos e queries | ~300MB |
| Rerank | Qwen3-Reranker-0.6B Q8 | Cross-encoder reranking | ~600MB |
| Generate | Custom fine-tuned 1.7B Q4_K_M | Expansão de queries | ~1.7GB |

### Lazy Loading com Promise Dedup Guards

Cada modelo tem um **guard de promise** para evitar alocação dupla de VRAM quando múltiplas operações tentam carregar o mesmo modelo simultaneamente:

```typescript
// src/llm.ts:577-601
private async ensureEmbedModel(): Promise<LlamaModel> {
  // 1. Já carregado? Retorna imediatamente.
  if (this.embedModel) return this.embedModel;

  // 2. Já em carregamento? Espera a mesma promise.
  if (this.embedModelLoadPromise) return await this.embedModelLoadPromise;

  // 3. Primeiro a chegar? Cria promise de carregamento.
  this.embedModelLoadPromise = (async () => {
    const llama = await this.ensureLlama();
    const modelPath = await this.resolveModel(this.embedModelUri);
    const model = await llama.loadModel({ modelPath });
    this.embedModel = model;
    this.touchActivity();
    return model;
  })();

  try { return await this.embedModelLoadPromise; }
  finally { this.embedModelLoadPromise = null; }
  // ↑ finally limpa a promise in-flight mas mantém o modelo cacheado
}
```

**Padrão chave**: O `finally` limpa a promise de carregamento (permitindo futuras re-tentativas em caso de erro), mas o modelo resolvido permanece em `this.embedModel`. Este padrão é replicado para os três tipos de modelo (`ensureEmbedModel`, `ensureGenerateModel`, `ensureRerankModel`).

### Inactivity Lifecycle

```
Model loading → touchActivity() → timer de inatividade inicia
                                   ↓ (5 min default)
                                   canUnloadLLM()? → sim → unloadIdleResources()
                                                  → não → reagenda timer
```

Estratégia de disposal em duas camadas:

```typescript
// src/llm.ts:490-533
async unloadIdleResources(): Promise<void> {
  // Dispose contexts primeiro (objetos pesados por sessão)
  for (const ctx of this.embedContexts) await ctx.dispose();
  this.embedContexts = [];
  for (const ctx of this.rerankContexts) await ctx.dispose();
  this.rerankContexts = [];

  // Opcionalmente dispose modelos também (opt-in)
  if (this.disposeModelsOnInactivity) {
    if (this.embedModel) { await this.embedModel.dispose(); this.embedModel = null; }
    if (this.generateModel) { await this.generateModel.dispose(); this.generateModel = null; }
    if (this.rerankModel) { await this.rerankModel.dispose(); this.rerankModel = null; }
  }
  // Instância llama é mantida viva — é leve
}
```

- **Default**: Dispose apenas contexts (mantém modelos carregados — evita VRAM thrash)
- **Opt-in** (`disposeModelsOnInactivity: true`): Também dispose modelos
- O timer usa `.unref()` para não manter o processo vivo

### LLMSession Pattern

O sistema de sessões provê **garantias de lifecycle** com ref-counting:

```typescript
// src/llm.ts:1471-1483
export async function withLLMSession<T>(
  fn: (session: ILLMSession) => Promise<T>,
  options?: LLMSessionOptions
): Promise<T> {
  const manager = getSessionManager();
  const session = new LLMSession(manager, options);
  try { return await fn(session); }
  finally { session.release(); }
}
```

Características da sessão:

| Feature | Implementação |
|---|---|
| Ref-counting | `_activeSessionCount` previne idle disposal durante operações ativas |
| In-flight tracking | `_inFlightOperations` para rastreamento fino de operações |
| Auto-abort | `maxDuration` (default 10 min) com `AbortController` |
| Signal linking | Conecta ao `AbortSignal` do chamador |
| Non-singleton | `withLLMSessionForLlm()` para instâncias não-singleton |

A verificação de segurança antes do unload:

```typescript
// src/llm.ts:1297-1299
canUnload(): boolean {
  return this._activeSessionCount === 0 && this._inFlightOperations === 0;
}
```

**Força**: Previne o bug clássico onde o timer de inatividade dispara no meio de um batch de embedding.
**Fraqueza**: Sem sistema de prioridades — uma sessão longa de indexação bloqueia cleanup de idle mesmo se sessões de busca precisam de recursos.

---

## Pipeline de Embeddings

### Formatação Assimétrica Query/Document

Este é o aspecto mais crítico do pipeline de embeddings. Modelos de embedding para retrieval aprendem representações **diferentes** para queries e documentos:

```typescript
// src/llm.ts:29-58
export function isQwen3EmbeddingModel(modelUri: string): boolean {
  return /qwen.*embed/i.test(modelUri) || /embed.*qwen/i.test(modelUri);
}

// Formatação de QUERY
export function formatQueryForEmbedding(query: string, modelUri?: string): string {
  if (isQwen3EmbeddingModel(uri)) {
    return `Instruct: Retrieve relevant documents for the given query\nQuery: ${query}`;
  }
  // Nomic/embeddinggemma (default)
  return `task: search result | query: ${query}`;
}

// Formatação de DOCUMENTO
export function formatDocForEmbedding(text: string, title?: string, modelUri?: string): string {
  if (isQwen3EmbeddingModel(uri)) {
    return title ? `${title}\n${text}` : text;  // Qwen3: texto cru
  }
  // Nomic/embeddinggemma (default)
  return `title: ${title || "none"} | text: ${text}`;
}
```

| Modelo | Query Format | Document Format |
|---|---|---|
| Nomic/embeddinggemma | `task: search result \| query: {q}` | `title: {t} \| text: {d}` |
| Qwen3-Embedding | `Instruct: Retrieve...\nQuery: {q}` | Texto cru (opcional título) |

A detecção do modelo é feita por regex no URI — simples mas funcional.

### Pool de Contextos Paralelos

O qmd cria múltiplos `LlamaEmbeddingContext` para embeddings em paralelo:

```typescript
// src/llm.ts:611-629
private async computeParallelism(perContextMB: number): Promise<number> {
  if (llama.gpu) {
    const vram = await llama.getVramState();
    const freeMB = vram.free / (1024 * 1024);
    const maxByVram = Math.floor((freeMB * 0.25) / perContextMB);
    return Math.max(1, Math.min(8, maxByVram));  // 1-8 contextos
  }
  // CPU: pelo menos 4 threads por contexto
  const cores = llama.cpuMathCores || 4;
  return Math.max(1, Math.min(4, Math.floor(cores / 4)));
}
```

- **GPU**: Até 8 contextos, limitado a 25% da VRAM livre
- **CPU**: Até 4 contextos, mínimo 4 threads cada
- **Fallback**: Se a criação de contexto falha, usa quantos conseguiu criar

O batch embedding distribui textos entre contextos:

```typescript
// src/llm.ts:918-946 — Múltiplos contextos: distribui textos
const chunkSize = Math.ceil(texts.length / n);
const chunks = Array.from({ length: n }, (_, i) =>
  texts.slice(i * chunkSize, (i + 1) * chunkSize)
);
const chunkResults = await Promise.all(
  chunks.map(async (chunk, i) => {
    const ctx = contexts[i]!;
    // cada contexto processa seu slice sequencialmente
    for (const text of chunk) {
      const embedding = await ctx.getEmbeddingFor(safeText);
      // ...
    }
  })
);
return chunkResults.flat();
```

### Truncação de Contexto

```typescript
// src/llm.ts:841-855
private async truncateToContextSize(text: string): Promise<{ text: string; truncated: boolean }> {
  const maxTokens = this.embedModel.trainContextSize;
  const tokens = this.embedModel.tokenize(text);
  if (tokens.length <= maxTokens) return { text, truncated: false };
  const safeLimit = Math.max(1, maxTokens - 4); // margem de 4 tokens BOS/EOS
  const truncatedTokens = tokens.slice(0, safeLimit);
  return { text: this.embedModel.detokenize(truncatedTokens), truncated: true };
}
```

**Força**: Previne crashes GGML com inputs excessivamente grandes.
**Fraqueza**: Truncação silenciosa (apenas `console.warn`) — consumidores downstream não sabem que conteúdo foi perdido.

---

## Smart Chunking

### Configuração

```typescript
// src/store.ts:51-59
export const CHUNK_SIZE_TOKENS = 900;
export const CHUNK_OVERLAP_TOKENS = Math.floor(CHUNK_SIZE_TOKENS * 0.15);  // 135 tokens
export const CHUNK_SIZE_CHARS = CHUNK_SIZE_TOKENS * 4;    // 3600 chars (fallback)
export const CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * 4;  // 540 chars
export const CHUNK_WINDOW_TOKENS = 200;  // janela de busca por break points
```

### Detecção de Break Points

O sistema pré-escaneia documentos para pontos de quebra estruturais com prioridades pontuadas:

```typescript
// src/store.ts:97-110
export const BREAK_PATTERNS: [RegExp, number, string][] = [
  [/\n#{1}(?!#)/g, 100, 'h1'],     // Melhor ponto de quebra
  [/\n#{2}(?!#)/g, 90, 'h2'],
  [/\n#{3}(?!#)/g, 80, 'h3'],
  [/\n```/g, 80, 'codeblock'],     // Fronteira de code block = prioridade h3
  [/\n(?:---|\*\*\*|___)\s*\n/g, 60, 'hr'],  // Linha horizontal
  [/\n\n+/g, 20, 'blank'],         // Parágrafo
  [/\n[-*]\s/g, 5, 'list'],        // Item de lista
  [/\n\d+\.\s/g, 5, 'numlist'],    // Item numerado
  [/\n/g, 1, 'newline'],           // Último recurso
];
```

### Decay Quadrático por Distância

O algoritmo de seleção de ponto de corte usa decaimento quadrático — break points mais distantes do alvo perdem pontuação, mas headings distantes ainda vencem newlines próximas:

```typescript
// src/store.ts:188-224
export function findBestCutoff(breakPoints, targetCharPos, windowChars, decayFactor, codeFences) {
  for (const bp of breakPoints) {
    if (bp.pos < windowStart) continue;
    if (bp.pos > targetCharPos) break;

    // Skip dentro de code fences
    if (isInsideCodeFence(bp.pos, codeFences)) continue;

    const distance = targetCharPos - bp.pos;
    // Decaimento quadrático: suave no início, íngreme no fim
    // Na posição alvo:    multiplicador = 1.0
    // A 50% para trás:    multiplicador = 0.825
    // Na borda da janela: multiplicador = 0.3
    const normalizedDist = distance / windowChars;
    const multiplier = 1.0 - (normalizedDist * normalizedDist) * decayFactor;
    const finalScore = bp.score * multiplier;
  }
}
```

**Design chave**: Um heading (`score=100`) distante na janela (`multiplier=0.6`) produz `score=60`, que ainda vence um newline (`score=1`) na posição exata (`multiplier=1.0`, `score=1`). O decaimento quadrático torna headings "gravitacionalmente atrativos".

### Proteção de Code Fences

```typescript
// src/store.ts:144-174
export function findCodeFences(text: string): CodeFenceRegion[] {
  const regions: CodeFenceRegion[] = [];
  const fencePattern = /\n```/g;
  let inFence = false;
  let fenceStart = 0;

  for (const match of text.matchAll(fencePattern)) {
    if (!inFence) {
      fenceStart = match.index!;
      inFence = true;
    } else {
      regions.push({ start: fenceStart, end: match.index! + match[0].length });
      inFence = false;
    }
  }
  // Fence não fechada estende até o fim do documento
  if (inFence) regions.push({ start: fenceStart, end: text.length });
  return regions;
}
```

Break points dentro de code fences são ignorados, prevenindo splits no meio de código.

### Chunking Token-Accurate (Two-Pass)

```typescript
// src/store.ts:2091-2137
export async function chunkDocumentByTokens(content, maxTokens=900, ...) {
  // Pass 1: Chunk em espaço de caracteres com estimativa conservadora
  //   - Usa ~3 chars/token (conservador para conteúdo misto)
  //   - Aplica smart break points com findBestCutoff()
  //   - Aplica overlap de 15%

  // Pass 2: Verifica tokens reais
  //   - Tokeniza cada chunk
  //   - Se excede maxTokens, re-split com ratio chars/token real
  //   - Usa ratio medido para precisão na re-divisão
}
```

A abordagem de dois passes é inteligente: chunking baseado em caracteres com detecção estrutural primeiro (rápido, determinístico), depois verificação com tokenizer real com re-splitting se necessário.

---

## Armazenamento Vetorial

### Camada de Banco de Dados (`src/db.ts`)

Camada de compatibilidade cross-runtime:

```typescript
// src/db.ts:19-56
if (isBun) {
  const BunDatabase = (await import("bun:sqlite")).Database;

  // macOS: Apple SQLite tem SQLITE_OMIT_LOAD_EXTENSION — swap para Homebrew
  if (process.platform === "darwin") {
    const homebrewPaths = [
      "/opt/homebrew/opt/sqlite/lib/libsqlite3.dylib",  // Apple Silicon
      "/usr/local/opt/sqlite/lib/libsqlite3.dylib",     // Intel
    ];
    for (const p of homebrewPaths) {
      try { BunDatabase.setCustomSQLite(p); break; } catch {}
    }
  }
  _Database = BunDatabase;
  // Testa se extensions funcionam antes de confiar
  try {
    const { getLoadablePath } = await import("sqlite-vec");
    const testDb = new BunDatabase(":memory:");
    testDb.loadExtension(vecPath);  // ← teste real
    _sqliteVecLoad = (db: any) => db.loadExtension(vecPath);
  } catch {
    _sqliteVecLoad = null;  // Graceful fallback — FTS continua funcionando
  }
} else {
  _Database = (await import("better-sqlite3")).default;
}
```

### Schema de Banco de Dados

```sql
-- Armazenamento content-addressable (fonte de verdade)
CREATE TABLE content (
  hash TEXT PRIMARY KEY,          -- SHA-256 do conteúdo
  doc TEXT NOT NULL,
  created_at TEXT NOT NULL
);

-- Camada de filesystem (paths virtuais → hashes)
CREATE TABLE documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  collection TEXT NOT NULL,
  path TEXT NOT NULL,
  title TEXT NOT NULL,
  hash TEXT NOT NULL,
  active INTEGER NOT NULL DEFAULT 1,   -- soft delete
  FOREIGN KEY (hash) REFERENCES content(hash),
  UNIQUE(collection, path)
);

-- Metadata de vetores (quais chunks têm embeddings)
CREATE TABLE content_vectors (
  hash TEXT NOT NULL,
  seq INTEGER NOT NULL DEFAULT 0,     -- número sequencial do chunk
  pos INTEGER NOT NULL DEFAULT 0,     -- offset em caracteres no documento
  model TEXT NOT NULL,
  PRIMARY KEY (hash, seq)
);

-- Tabela virtual sqlite-vec (vetores reais)
CREATE VIRTUAL TABLE vectors_vec USING vec0(
  hash_seq TEXT PRIMARY KEY,          -- "{hash}_{seq}"
  embedding float[{dimensions}] distance_metric=cosine
);

-- Full-text search
CREATE VIRTUAL TABLE documents_fts USING fts5(
  filepath, title, body,
  tokenize='porter unicode61'
);
```

**Decisões de design**:
- **Content-addressable**: Documentos referenciam conteúdo por hash SHA-256. Mesmo conteúdo compartilhado entre coleções.
- **Dual key para vetores**: `hash_seq` = `{content_hash}_{chunk_seq}` — liga embeddings de chunks aos documentos fonte.
- **Distância coseno**: `distance_metric=cosine` na definição do vec0.
- **FTS triggers**: Auto-sync `documents_fts` via triggers SQL em INSERT/UPDATE/DELETE.

### Busca Vetorial — Two-Step Query

```typescript
// src/store.ts:2820-2904 — Consulta em dois passos (JOINs travam sqlite-vec!)
export async function searchVec(db, query, model, limit, collectionName, session, precomputedEmbedding) {
  // Passo 1: Busca vetorial pura (SEM JOINs — eles travam sqlite-vec)
  const vecResults = db.prepare(`
    SELECT hash_seq, distance
    FROM vectors_vec
    WHERE embedding MATCH ? AND k = ?
  `).all(new Float32Array(embedding), limit * 3);  // 3× over-fetch

  // Passo 2: Query separada para dados do documento
  const docRows = db.prepare(`
    SELECT cv.hash, cv.pos, d.title, content.doc as body, ...
    FROM content_vectors cv
    JOIN documents d ON d.hash = cv.hash
    JOIN content ON content.hash = d.hash
    WHERE cv.hash || '_' || cv.seq IN (${placeholders})
  `).all(...hashSeqs);

  // Dedup por filepath, mantém melhor distância
  // Score = 1 - cosine_distance (cosine similarity)
}
```

**Bug crítico evitado**: Tabelas virtuais sqlite-vec travam com JOINs na mesma query. O approach de dois passos é mandatório.

### Criação Dinâmica da Tabela Vetorial

```typescript
// src/store.ts:962-977
function ensureVecTableInternal(db, dimensions) {
  // Verifica schema existente
  // Se dimensões erradas, métrica errada, ou hash_seq faltando → DROP + recria
  db.exec(`CREATE VIRTUAL TABLE vectors_vec USING vec0(
    hash_seq TEXT PRIMARY KEY,
    embedding float[${dimensions}] distance_metric=cosine
  )`);
}
```

A dimensão é determinada em runtime a partir do primeiro embedding bem-sucedido — sistema agnóstico ao modelo.

---

## Reranking

### Arquitetura Cross-Encoder

O qmd usa o Qwen3-Reranker-0.6B como cross-encoder, com otimizações agressivas de VRAM:

```typescript
// src/llm.ts:759-762
// Budget: 800 tokens (chunk) + 200 tokens (overhead de template) + query ≈ 1100 tokens
// 2048 por margem de segurança. 17× menos que auto (40960).
private static readonly RERANK_CONTEXT_SIZE = 2048;
```

Otimizações:
- **flashAttention**: ~20% menos VRAM por contexto (com fallback se não suportado)
- **Context size cap**: 2048 tokens (vs auto 40960) — redução de 17× em VRAM
- **Múltiplos contextos paralelos**: Até 4, limitados por VRAM

### Fluxo de Reranking

```typescript
// src/llm.ts:1106-1199
async rerank(query, documents, options) {
  // 1. Truncar documentos para caber na janela de contexto
  const maxDocTokens = RERANK_CONTEXT_SIZE - TEMPLATE_OVERHEAD - queryTokens;

  // 2. Deduplicar textos de chunks idênticos (evita scoring redundante)
  const textToDocs = new Map<string, { file, index }[]>();

  // 3. Distribuir entre contextos paralelos
  const chunkSize = Math.ceil(texts.length / activeContexts.length);

  // 4. Ranking paralelo
  const allScores = await Promise.all(
    chunks.map((chunk, i) => activeContexts[i]!.rankAll(query, chunk))
  );

  // 5. Fan out scores deduplicados de volta para documentos originais
}
```

### Cache Content-Addressed

```typescript
// src/store.ts:3007-3050
// Chave de cache = hash(query + model + chunk_text)
// NÃO usa file path — mesmo texto de chunk recebe mesmo score independente do arquivo fonte
const cacheKey = getCacheKey("rerank", { query: rerankQuery, model, chunk: doc.text });
```

**Força**: Cache content-addressed significa que chunks idênticos (comum em coleções com duplicatas) são pontuados uma única vez.

### Reranking Intent-Aware

```typescript
// src/store.ts:3009
const rerankQuery = intent ? `${intent}\n\n${query}` : query;
```

Quando um intent é fornecido (ex: "tempos de carregamento de páginas web"), ele é prepend ao rerank query, direcionando o julgamento de relevância do cross-encoder.

---

## Query Pipeline Completo

### `hybridQuery()` — O Orquestrador Principal

Pipeline de 8 etapas em `src/store.ts:3712-3990`:

```
Etapa 1: Sonda BM25
  ├── searchFTS(query) → top 20 resultados
  ├── Sinal forte? (score ≥ 0.85 E gap ≥ 0.15 do #2)
  │   └── SIM → pula expansão (economia ~500ms)
  └── Intent fornecido? → desabilita bypass de sinal forte

Etapa 2: Expansão de Query (LLM-powered)
  ├── LlamaCpp.expandQuery() usando modelo fine-tuned 1.7B
  ├── Gramática restrita: "lex: ...\nvec: ...\nhyde: ..."
  ├── Filtros: deve conter termos da query original, sem duplicatas
  └── Fallback: vec + lex + hyde com query original

Etapa 3: Busca Type-Routed
  ├── lex queries → searchFTS() (instantâneo, síncrono)
  ├── vec/hyde queries → batch embedBatch() → searchVec() com embeddings pré-computados
  └── Query original → ambos FTS (da etapa 1) e busca vetorial

Etapa 4: Fusão RRF (Reciprocal Rank Fusion)
  ├── reciprocalRankFusion() com k=60
  ├── Primeiras 2 listas ganham peso 2× (queries originais)
  ├── Bônus top-rank: +0.05 para rank 1, +0.02 para ranks 2-3
  └── Slice para candidateLimit (default 40)

Etapa 5: Seleção de Chunks + Best-Chunk
  ├── chunkDocument() cada corpo de candidato
  ├── Pontua chunks por overlap de termos da query
  ├── Termos de intent contribuem a peso 0.5×
  └── Seleciona melhor chunk por documento para reranking

Etapa 6: Cross-Encoder Reranking
  ├── rerank(query, best_chunks) — NÃO corpos completos
  ├── Check de cache por texto de chunk
  └── Contextos paralelos para chunks não-cacheados

Etapa 7: Blending Position-Aware
  ├── RRF rank 1-3:  peso = 0.75 (protege top resultados de retrieval)
  ├── RRF rank 4-10: peso = 0.60
  ├── RRF rank 11+:  peso = 0.40 (deixa reranker dominar)
  └── blendedScore = rrfWeight × (1/rrfRank) + (1-rrfWeight) × rerankScore

Etapa 8: Dedup + Filtro + Slice
  └── Dedup por filepath → filtra por minScore → slice para limit
```

### Reciprocal Rank Fusion — Implementação

```typescript
// src/store.ts:3056-3099
export function reciprocalRankFusion(
  resultLists: RankedResult[][],
  weights: number[] = [],
  k: number = 60
): RankedResult[] {
  const scores = new Map<string, { result, rrfScore, topRank }>();

  for (let listIdx = 0; listIdx < resultLists.length; listIdx++) {
    const weight = weights[listIdx] ?? 1.0;
    for (let rank = 0; rank < list.length; rank++) {
      const rrfContribution = weight / (k + rank + 1);
      // Acumula scores por filepath
    }
  }

  // Bônus top-rank
  for (const entry of scores.values()) {
    if (entry.topRank === 0) entry.rrfScore += 0.05;
    else if (entry.topRank <= 2) entry.rrfScore += 0.02;
  }
}
```

### Strong Signal Bypass

```typescript
// src/store.ts:228-229
export const STRONG_SIGNAL_MIN_SCORE = 0.85;
export const STRONG_SIGNAL_MIN_GAP = 0.15;
```

Quando BM25 encontra um resultado claramente dominante (`score ≥ 0.85` com `gap ≥ 0.15` do segundo), a expansão de query é pulada — economia de ~500ms em matches óbvios de keyword.

**Desabilitado quando intent é fornecido** — o match BM25 óbvio pode não alinhar com o intent do chamador.

### Expansão de Query com Gramática

```typescript
// src/llm.ts:1024-1031
const grammar = await llama.createGrammar({
  grammar: `
    root ::= line+
    line ::= type ": " content "\\n"
    type ::= "lex" | "vec" | "hyde"
    content ::= [^\\n]+
  `
});
// Gera output estruturado como:
// lex: termos de busca keyword
// vec: variação semântica da query
// hyde: documento hipotético que responderia a query
```

---

## Manutenção e Collections

### Organização de Collections (`src/collections.ts`)

Padrão de armazenamento dual:
- **Arquivo YAML** (`~/.config/qmd/index.yml`): Fonte de verdade externa, editável por humanos
- **SQLite `store_collections`**: Cópia no banco para queries auto-contidas

```yaml
# ~/.config/qmd/index.yml
global_context: "Base de conhecimento pessoal"
collections:
  notes:
    path: /Users/me/notes
    pattern: "**/*.md"
    ignore: ["Sessions/**"]
    context:
      /: "Notas gerais"
      /2024: "Notas de 2024"
    includeByDefault: true
```

A sincronização é hash-based — se o JSON do config não mudou, o sync é pulado:

```typescript
// store.ts:syncConfigToDb
const hash = createHash('sha256').update(JSON.stringify(config)).digest('hex');
if (existingHash === hash) return;  // skip se não mudou
```

### Re-indexação Incremental

```
1. fast-glob scan filesystem → filtra hidden files → filtra ignore patterns
2. Para cada arquivo:
   - Lê conteúdo, computa SHA-256 hash
   - handelize(path) → path normalizado (lowercase, dashes)
   - Compara com existente: inalterado/atualizado/novo
   - insertContent() + insertDocument() (upsert com ON CONFLICT)
3. Desativa documentos não mais no filesystem (soft delete)
4. cleanupOrphanedContent()
```

**Dedup content-addressable**: Se dois arquivos têm conteúdo idêntico, compartilham o mesmo hash de conteúdo e embeddings.

### Operações de Manutenção (`src/maintenance.ts`)

```typescript
export class Maintenance {
  vacuum(): void;                    // SQLite VACUUM — recupera espaço
  cleanupOrphanedContent(): number;  // Deleta content não referenciado por docs ativos
  cleanupOrphanedVectors(): number;  // Deleta embeddings de content removido
  clearLLMCache(): number;           // Purga cache de expansão + rerank
  deleteInactiveDocs(): number;      // Remove documentos soft-deleted
  clearEmbeddings(): void;           // Opção nuclear — força re-embedding completo
}
```

### Pipeline de Embedding em Batch

```typescript
// src/store.ts:1303-1447
// 1. getPendingEmbeddingDocs() → docs sem embeddings (LEFT JOIN em content_vectors)
// 2. buildEmbeddingBatches() → divide em batches por contagem E tamanho em bytes
// 3. Para cada batch:
//    a. Carrega corpos de documentos da tabela content
//    b. chunkDocumentByTokens() → divide em chunks de 900 tokens com 15% overlap
//    c. formatDocForEmbedding() → aplica formatação modelo-específica
//    d. session.embedBatch() em grupos de 32 chunks
//    e. Fallback: se batch falha, tenta chunks individuais
//    f. insertEmbedding() → grava em content_vectors E vectors_vec
```

Limites de batch:

| Parâmetro | Valor | Razão |
|---|---|---|
| `DEFAULT_EMBED_MAX_DOCS_PER_BATCH` | 64 | Previne OOM em coleções grandes |
| `DEFAULT_EMBED_MAX_BATCH_BYTES` | 64 MB | Teto de memória por batch |
| `BATCH_SIZE` (embedding) | 32 chunks | Por chamada embedBatch() |
| Pool de embed contexts | 1-8 (GPU), 1-4 (CPU) | Baseado em recursos disponíveis |

---

## Pontos Fortes ✅

1. **Smart chunking com proteção de code fences**: Quebras structure-aware com decaimento quadrático. Headings distantes vencem newlines próximas — o chunking respeita a estrutura do documento.

2. **Busca híbrida (BM25 + vetorial + reranking)**: Pipeline de 8 etapas que combina o melhor de keyword search (FTS5 com Porter stemming) com busca semântica (embeddings) e reranking cross-encoder.

3. **Formatação assimétrica query/document**: Prompts diferentes para queries e documentos com detecção automática de modelo (nomic vs qwen3) — essencial para retrieval-focused embeddings.

4. **Dedup content-addressable**: Mesmo conteúdo em múltiplas coleções compartilha embeddings via hash SHA-256.

5. **Pipeline de reranking eficiente**: Context cap de 2048 tokens (17× economia VRAM vs auto), dedup de textos idênticos antes do scoring, cache content-addressed.

6. **Session lifecycle com ref-counting**: `withLLMSession()` previne idle disposal durante operações ativas. `canUnload()` verifica sessões E operações in-flight.

7. **Promise dedup guards**: Pattern elegante que previne carregamento duplo de modelos — múltiplas chamadas concurrent esperam a mesma promise.

8. **Strong signal bypass**: Pula expansão cara quando BM25 já encontrou resultado claramente dominante — economia de ~500ms.

9. **Blending position-aware**: Top resultados de retrieval são protegidos contra discordância do reranker — resultados rank 1-3 mantêm peso 0.75 do RRF.

10. **Batch embedding com fallback**: Se o batch inteiro falha, tenta chunks individuais — resiliência sem perda de dados.

---

## Pontos Fracos ❌

1. **Embedding single-threaded por contexto**: Apesar do pool de contextos, cada contexto processa sequencialmente. Node-llama-cpp não expõe batch processing real (cada `getEmbeddingFor()` é uma chamada separada).

2. **Dependência pesada de node-llama-cpp**: API de alto nível que abstrai GGML. Cada versão de node-llama-cpp pode quebrar APIs internas. Upgrade friction significativo.

3. **Singleton global**: `getDefaultLlamaCpp()` impõe um conjunto de modelos por processo. Múltiplas configurações de índice no mesmo processo precisam de overrides explícitos via `store.llm`.

4. **Truncação silenciosa**: Documentos excedendo a janela de contexto são truncados com apenas `console.warn`. Consumidores downstream não sabem que conteúdo foi perdido.

5. **Chunking fallback baseado em caracteres**: Token-accurate chunking requer async + modelo. A estimativa de caracteres (4 chars/token) é imprecisa para código (~2 chars/token).

6. **Seleção de chunk por keyword overlap**: O melhor chunk por documento é selecionado por sobreposição de keywords, não pelo reranker. Se a heurística de keyword escolhe o chunk errado, a qualidade do reranking degrada.

7. **Limitações sqlite-vec**: Tabela virtual não pode ter JOINs (workaround de dois passos mandatório). Sem suporte a pré-filtragem por coleção no nível vetorial — over-fetch 3× e depois filtra.

8. **Sem streaming**: Embedding e reranking são totalmente batch — sem resultados incrementais.

9. **Re-indexação não-incremental**: Precisa escanear todos os documentos para encontrar embeddings faltantes (apesar do dedup por hash prevenir re-embedding de conteúdo inalterado).

10. **Sem prioridade de sessões**: Uma sessão longa de indexação bloqueia cleanup de idle mesmo se sessões de busca precisam de recursos.

---

## Lições para bun-llama-cpp

### Padrões a Adotar

**1. Promise-based load deduplication**

O pattern de guard com promise previne carregamento duplo de modelos — diretamente aplicável ao nosso Worker:

```typescript
// Padrão extraído do qmd — adaptar para Worker messages
private modelLoadPromise: Promise<void> | null = null;

async ensureModel(): Promise<void> {
  if (this.modelLoaded) return;
  if (this.modelLoadPromise) return await this.modelLoadPromise;

  this.modelLoadPromise = this.doLoad();
  try { await this.modelLoadPromise; }
  finally { this.modelLoadPromise = null; }
}
```

**2. Inactivity timer com session ref-counting**

Disposal seguro de recursos idle:

```typescript
// Conceito: Não descarrega se há operações ativas
class SessionManager {
  private count = 0;
  acquire() { this.count++; }
  release() { this.count = Math.max(0, this.count - 1); }
  canUnload() { return this.count === 0; }
}
```

**3. Formatação assimétrica de embeddings**

Se/quando adicionarmos API de embeddings, a distinção query/document é essencial. Sem ela, a qualidade de retrieval degrada significativamente.

**4. Two-pass chunking**

Char-based structure detection (rápido, determinístico) + token verification (preciso, async). Ideal para pipeline de RAG futuro.

**5. Content-addressable storage**

SHA-256 para dedup de conteúdo. Evita re-computar embeddings para conteúdo idêntico em contextos diferentes.

**6. Busca híbrida com RRF**

Combinar FTS (keyword, instantâneo) com vetorial (semântico, mais lento) via Reciprocal Rank Fusion produz resultados superiores a qualquer um isolado.

**7. Cross-encoder reranking com context cap**

O cap de 2048 tokens (vs 40960 auto) reduz VRAM 17× com perda mínima de qualidade para chunks de ~800 tokens. Otimização dramática que habilita reranking em hardware consumer.

### O que Não Adotar

- **node-llama-cpp como dependência**: API de alto nível com abstração excessiva para nosso caso. Nosso FFI direto via `bun:ffi` + C shims é mais eficiente e controlável.
- **Global singleton pattern**: Dificulta testes e configurações múltiplas. Nosso padrão de instância explícita é superior.
- **Truncação silenciosa**: Sempre comunicar ao consumidor quando dados são perdidos.