# Embeddings: Análise Completa e Estratégia

> Documento temático comparando as abordagens de embedding de node-llama-cpp e qmd,
> mapeando o estado atual do bun-llama-cpp e propondo uma estratégia de implementação em camadas.

---

## O Que São Embeddings e Por Que Importam

Embeddings são representações vetoriais densas de texto em um espaço de alta dimensão (tipicamente 384–4096 floats). Textos semanticamente similares produzem vetores próximos, permitindo busca por significado ao invés de palavras-chave.

### Casos de Uso

| Caso de Uso | Como Funciona |
|---|---|
| **RAG** (Retrieval-Augmented Generation) | Indexa documentos como vetores → busca os mais relevantes → alimenta o LLM com contexto |
| **Semantic Search** | Converte query em vetor → busca vizinhos mais próximos no índice |
| **Clustering** | Agrupa documentos por similaridade vetorial (k-means, HDBSCAN) |
| **Deduplicação** | Detecta conteúdo duplicado/near-duplicate via similaridade de cosseno |
| **Classificação** | Usa embeddings como features para classificadores downstream |
| **Reranking** | Cross-encoder avalia relevância query↔documento diretamente |

**Por que embeddings são o gap #1 do bun-llama-cpp**: sem eles, não é possível construir pipelines RAG — o caso de uso mais comum para LLMs locais em aplicações reais.

---

## Como llama.cpp Suporta Embeddings

### API Nativa

O llama.cpp oferece suporte nativo a embeddings via quatro funções principais:

```c
// Dimensão do vetor de embedding do modelo
int32_t llama_model_n_embd(const struct llama_model * model);

// Tipo de pooling configurado no contexto
enum llama_pooling_type llama_pooling_type(const struct llama_context * ctx);

// Embeddings poolados por sequência (mean, CLS, last, rank)
float * llama_get_embeddings_seq(struct llama_context * ctx, llama_seq_id seq_id);

// Embeddings do token na posição i (sem pooling)
float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
```

Além disso, `llama_encode()` é usado no lugar de `llama_decode()` para computar embeddings — executa o forward pass sem gerar tokens:

```c
// Forward pass sem KV cache (ideal para embeddings)
int32_t llama_encode(struct llama_context * ctx, struct llama_batch batch);
```

### Pooling Types

O pooling determina como os embeddings de tokens individuais são combinados em um vetor único por sequência:

```c
enum llama_pooling_type {
    LLAMA_POOLING_TYPE_UNSPECIFIED = -1,
    LLAMA_POOLING_TYPE_NONE = 0,   // Sem pooling — usa embedding do último token
    LLAMA_POOLING_TYPE_MEAN = 1,   // Média de todos os tokens
    LLAMA_POOLING_TYPE_CLS  = 2,   // Usa embedding do token [CLS]
    LLAMA_POOLING_TYPE_LAST = 3,   // Usa embedding do último token
    LLAMA_POOLING_TYPE_RANK = 4,   // Cross-encoder — produz logit de relevância
};
```

### Context Params para Embeddings

```c
struct llama_context_params params = llama_context_default_params();
params.embeddings = true;                         // Habilita extração de embeddings
params.pooling_type = LLAMA_POOLING_TYPE_MEAN;   // Ou CLS, LAST, RANK
```

### Modelos de Embedding vs Modelos Generativos

| Aspecto | Modelo de Embedding | Modelo Generativo |
|---|---|---|
| Arquitetura | Encoder-only (BERT, nomic-bert) | Decoder-only (LLaMA, Mistral) |
| Saída | Vetor float[n_embd] | Next-token probabilities |
| Pooling | MEAN ou CLS (tipicamente) | N/A |
| Treino | Contrastive learning (query↔doc) | Autoregressive (next token) |
| Tamanho típico | 100M–600M params | 1B–70B+ params |
| VRAM | ~200MB–1GB | 2GB–40GB+ |
| Exemplos GGUF | embeddinggemma-300M, nomic-embed-text | llama-3, qwen2.5, mistral |

> **Nota**: Modelos generativos *podem* produzir embeddings (via pooling do último token), mas modelos dedicados de embedding são muito mais eficientes e produzem vetores de melhor qualidade para retrieval.

---

## Como node-llama-cpp Implementa

### LlamaEmbeddingContext

A implementação de node-llama-cpp é um wrapper fino sobre um `LlamaContext` padrão com a flag `_embeddings: true`.

#### Criação

```typescript
// API do consumidor
const embeddingCtx = await model.createEmbeddingContext({
    contextSize: 512,   // ou "auto"
    batchSize: 128,
    threads: 6,
});

// Internamente: cria LlamaContext com embedding mode
const llamaContext = await _model.createContext({
    contextSize, batchSize, threads,
    _embeddings: true   // ← flag chave: seta context_params.embeddings = true
});
```

#### Execução Serial via withLock

Cada chamada a `getEmbeddingFor()` é serializada por `withLock`:

```
"Hello world"
     │
     ▼
tokenize(input)  →  [15043, 3186]
     │
     ▼
add BOS/EOS  →  [1, 15043, 3186, 2]
     │
     ▼
withLock (serial queue)
     │
     ▼
eraseContextTokenRanges()  ← limpa KV cache anterior
     │
     ▼
sequence.evaluate(tokens, {_noSampling: true})  ← forward pass sem geração
     │
     ▼
_ctx.getEmbedding(tokenCount)  ← extrai do contexto nativo
     │
     ▼
LlamaEmbedding({ vector: Float64Array → number[] })
```

#### Extração de Vetor e LlamaEmbedding

O vetor é extraído no addon C++ via estratégia de fallback:

```cpp
// 1. Tenta embeddings poolados (mean/cls/rank)
const auto* embeddings = pooling_type == LLAMA_POOLING_TYPE_NONE
    ? NULL
    : llama_get_embeddings_seq(ctx, 0);   // sequência 0

// 2. Fallback: embedding do último token
if (embeddings == NULL) {
    embeddings = llama_get_embeddings_ith(ctx, inputTokensLength - 1);
}

// 3. Copia para Float64Array
Napi::Float64Array result = Napi::Float64Array::New(env, n_embd);
for (size_t i = 0; i < n_embd; i++) {
    result[i] = embeddings[i];
}
```

A classe `LlamaEmbedding` encapsula o vetor com imutabilidade:

```typescript
class LlamaEmbedding {
    readonly vector: readonly number[];  // Object.freeze() + slice()

    calculateCosineSimilarity(other): number {
        // dot(a,b) / (||a|| × ||b||)
    }

    toJSON(): LlamaEmbeddingJSON;
    static fromJSON(json): LlamaEmbedding;
}
```

### LlamaRankingContext

#### Conceito: Cross-Encoder

Diferente de embeddings (bi-encoder: codifica query e documento separadamente), o ranking usa **cross-encoder**: query e documento são concatenados e avaliados juntos. O modelo produz um logit único convertido em probabilidade de relevância via sigmoid.

#### Criação com POOLING_TYPE_RANK

```typescript
const llamaContext = await _model.createContext({
    contextSize, batchSize, threads,
    _embeddings: true,    // habilita embeddings
    _ranking: true         // seta pooling_type = LLAMA_POOLING_TYPE_RANK (4)
});
```

Validações na criação:
1. Rejeita modelos encoder-decoder (ex: T5)
2. Verifica presença de tensor `cls.weight` ou `cls.output.weight`
3. Requer token SEP ou EOS, ou template de reranking válido

#### Template Handling (Query/Document)

```typescript
// Com template do modelo (metadata: tokenizer.chat_template.rerank):
// "<query>{{query}}</query><document>{{document}}</document>"

// Sem template — formato default:
// [BOS] query [EOS] [SEP] document [EOS]
const input = [
    bos, ...queryTokens, eos,
    sep, ...documentTokens, eos
];
```

#### Sigmoid Scoring

```typescript
// Extrai UM ÚNICO valor (maxVectorSize=1)
const embedding = ctx.getEmbedding(input.length, 1);
const logit = embedding[0];
const probability = 1 / (1 + Math.exp(-logit));  // sigmoid → 0..1
```

#### Batch Ranking

```typescript
public async rankAndSort(query, documents) {
    const scores = await this.rankAll(query, documents);
    return documents
        .map((doc, i) => ({ document: doc, score: scores[i] }))
        .sort((a, b) => b.score - a.score);
}
```

### Limitações do node-llama-cpp

| # | Limitação | Impacto |
|---|-----------|---------|
| 1 | **Sem batch embedding real** | Cada texto é avaliado separadamente com clear total do KV cache |
| 2 | **Sem normalização L2** | Muitos modelos de embedding esperam vetores normalizados |
| 3 | **Serial apesar de Promise.all** | `withLock` serializa todas as chamadas — `Promise.all` é ilusão de paralelismo |
| 4 | **Sem pooling configurável** | Pooling type é fixo na criação do contexto |
| 5 | **Float64 overhead** | Usa Float64Array (64-bit) quando o modelo produz float32 |
| 6 | **Sem caching** | Mesmo texto recomputado toda vez (sem memoização) |
| 7 | **Cosine-only** | Não oferece dot product, distância euclidiana, ou outras métricas |
| 8 | **Bug no ranking template** | `unshift` ao invés de `push` para end token (linha 203) |

---

## Como qmd Implementa

### Pipeline de Embedding

O qmd é uma aplicação completa de RAG que usa node-llama-cpp como camada de LLM. Sua pipeline de embedding adiciona camadas sofisticadas acima da API base.

#### Formatação Assimétrica (Query vs Doc)

A formatação assimétrica é **essencial** para embeddings de retrieval — o modelo aprende representações diferentes para queries e documentos:

```typescript
// Detecção de modelo
function isQwen3EmbeddingModel(modelUri: string): boolean {
    return /qwen.*embed/i.test(modelUri) || /embed.*qwen/i.test(modelUri);
}

// Query formatting
// Nomic/embeddinggemma: "task: search result | query: {query}"
// Qwen3:               "Instruct: Retrieve relevant documents...\nQuery: {query}"

// Document formatting
// Nomic/embeddinggemma: "title: {title} | text: {text}"
// Qwen3:               raw text (prefixo de título opcional)
```

#### Batch Processing com Limites

```typescript
const DEFAULT_EMBED_MAX_DOCS_PER_BATCH = 64;      // Máx docs por batch
const DEFAULT_EMBED_MAX_BATCH_BYTES = 64 * 1024 * 1024;  // 64MB por batch
const BATCH_SIZE = 32;  // Chunks por chamada embedBatch()
```

Pipeline do batch:
```
1. getPendingEmbeddingDocs() → docs sem embeddings (LEFT JOIN)
2. buildEmbeddingBatches() → divide por contagem E tamanho em bytes
3. Para cada batch:
   a. Carrega corpo dos docs
   b. chunkDocumentByTokens() → chunks de 900 tokens, 15% overlap
   c. formatDocForEmbedding() → formatação específica do modelo
   d. session.embedBatch() em grupos de 32 chunks
   e. Fallback: se batch falha, retenta chunks individuais
   f. insertEmbedding() → grava em content_vectors + vectors_vec
```

#### Chunking Inteligente (900 Tokens, 15% Overlap)

```typescript
const CHUNK_SIZE_TOKENS = 900;
const CHUNK_OVERLAP_TOKENS = Math.floor(CHUNK_SIZE_TOKENS * 0.15);  // 135 tokens
```

O sistema usa **break points com prioridade por score**:

| Padrão | Score | Tipo |
|---|---|---|
| `# Heading 1` | 100 | h1 (melhor ponto de corte) |
| `## Heading 2` | 90 | h2 |
| `### Heading 3` / `` ``` `` | 80 | h3 / code block boundary |
| Linha em branco dupla | 20 | Parágrafo |
| Item de lista `- ` / `* ` | 5 | Lista |
| Quebra de linha simples | 1 | Último recurso |

**Decay quadrático por distância**: Um heading longe do ponto alvo ainda vence um newline perto. Headings são "gravitacionalmente atrativos" como pontos de corte.

**Proteção de code fences**: Break points dentro de blocos de código são ignorados.

### Reranking

#### Cross-Encoder com Context Cap 2048

```typescript
const RERANK_CONTEXT_SIZE = 2048;  // 17× menos VRAM que auto (40960)
// Budget: 800 tokens (chunk) + 200 tokens (template overhead) + query ≈ 1100 tokens
```

#### Parallel Contexts para Throughput

```typescript
async computeParallelism(perContextMB: number): Promise<number> {
    if (llama.gpu) {
        const freeMB = vram.free / (1024 * 1024);
        const maxByVram = Math.floor((freeMB * 0.25) / perContextMB);
        return Math.max(1, Math.min(8, maxByVram));
    }
    // CPU: divide cores entre contextos. Mín 4 threads por contexto.
    return Math.max(1, Math.min(4, Math.floor(cores / 4)));
}
```

- GPU: até 8 contextos paralelos (25% da VRAM livre)
- CPU: até 4 contextos (mín 4 threads cada)

#### Content-Addressed Cache

```typescript
// Chave de cache = hash(query + model + chunk_text)
// NÃO usa file path — mesmo chunk text recebe mesmo score
const cacheKey = getCacheKey("rerank", { query, model, chunk: doc.text });
```

### Armazenamento

#### sqlite-vec com Cosine Distance

```sql
CREATE VIRTUAL TABLE vectors_vec USING vec0(
    hash_seq TEXT PRIMARY KEY,              -- "{content_hash}_{chunk_seq}"
    embedding float[{dimensions}] distance_metric=cosine
);
```

#### Two-Step Queries (Workaround para Bug do sqlite-vec)

```typescript
// Step 1: busca vetorial pura (SEM JOINs — eles travam sqlite-vec)
const vecResults = db.prepare(`
    SELECT hash_seq, distance FROM vectors_vec
    WHERE embedding MATCH ? AND k = ?
`).all(new Float32Array(embedding), limit * 3);

// Step 2: query separada para dados do documento
const docRows = db.prepare(`
    SELECT cv.hash, d.title, content.doc ...
    FROM content_vectors cv JOIN documents d ON d.hash = cv.hash
    WHERE cv.hash || '_' || cv.seq IN (${placeholders})
`).all(...hashSeqs);
```

#### SHA-256 Dedup

Armazenamento content-addressable: documentos são referenciados por hash SHA-256 do conteúdo. Mesmo conteúdo em coleções diferentes compartilha embeddings.

### Limitações do qmd

| # | Limitação | Impacto |
|---|-----------|---------|
| 1 | **Depende de node-llama-cpp** | Herda todas as limitações (sem batch real, serial, Float64) |
| 2 | **Single process** | Um conjunto de modelos por processo (singleton global) |
| 3 | **Sem streaming de embeddings** | Operações totalmente batch — sem resultados incrementais |
| 4 | **Truncation silenciosa** | Documentos excedem context window sem aviso ao consumidor |
| 5 | **Chunking por caracteres como fallback** | Estimativa de 4 chars/token é imprecisa para código (~2 chars/token) |
| 6 | **sqlite-vec não suporta JOINs** | Workaround de two-step obrigatório, over-fetch 3× |

---

## Estado Atual do bun-llama-cpp

### 0% de Suporte a Embeddings

O gap-analysis é claro: **embeddings têm 0% de cobertura**. Das ~180 funções em `llama.h`, nenhuma das 4 funções de embedding está vinculada:

| Função | Status | Propósito |
|---|---|---|
| `llama_model_n_embd` | ❌ Não vinculada | Dimensão do vetor (384, 768, 4096...) |
| `llama_pooling_type` | ❌ Não vinculada | Tipo de pooling ativo no contexto |
| `llama_get_embeddings_seq` | ❌ Não vinculada | Embedding poolado por sequência |
| `llama_get_embeddings_ith` | ❌ Não vinculada | Embedding do token na posição i |
| `llama_encode` | ❌ Não vinculada | Forward pass sem KV cache |

### FFI Functions Necessárias

**Novas bindings diretas em libllama** (sem shim — retornam escalares ou ponteiros):

```typescript
// Adicionar ao dlopen de libllama em ffi.ts
llama_model_n_embd:       { args: [FFIType.ptr],              returns: FFIType.i32 },
llama_pooling_type:       { args: [FFIType.ptr],              returns: FFIType.i32 },
llama_get_embeddings_seq: { args: [FFIType.ptr, FFIType.i32], returns: FFIType.ptr },
llama_get_embeddings_ith: { args: [FFIType.ptr, FFIType.i32], returns: FFIType.ptr },
```

**Novos shims C necessários** (para structs by-value):

```c
// Em llama_shims.c — setters de context_params

void shim_ctx_params_set_embeddings(struct llama_context_params *p, bool v) {
    p->embeddings = v;
}

void shim_ctx_params_set_pooling_type(struct llama_context_params *p, int32_t type) {
    p->pooling_type = (enum llama_pooling_type)type;
}

// Encode (forward pass para embeddings — como decode mas sem KV cache)
int32_t shim_encode(struct llama_context *ctx, struct llama_batch *batch) {
    return llama_encode(ctx, *batch);
}
```

### Leitura de Float Pointers via bun:ffi

A parte mais delicada da implementação: `llama_get_embeddings_seq/ith` retornam `float*` apontando para memória interna do llama.cpp. O Bun oferece `toArrayBuffer()` para converter ponteiros em buffers legíveis:

```typescript
import { toArrayBuffer } from 'bun:ffi'

function readEmbedding(ptr: number, n_embd: number): Float32Array {
    // ptr = ponteiro retornado por llama_get_embeddings_seq/ith
    // n_embd = dimensão do vetor (ex: 768)
    const buf = toArrayBuffer(ptr, 0, n_embd * 4)  // 4 bytes por float32
    return new Float32Array(buf)
}
```

### O Que o Gap-Analysis Identificou

Do gap-analysis (Feature #1):

> **Dificuldade**: Medium  
> **Impacto**: 🔴 High — required for RAG  
> **O que falta**: `llama_set_embeddings`, `llama_encode`, `llama_get_embeddings_ith/seq`. Need `embeddings: true` on context params, `llama_encode()` instead of `llama_decode()`, float pointer read-back. New shims: `shim_ctx_params_set_embeddings`, `shim_encode`, `shim_get_embeddings_ith`.

---

## Estratégia Proposta

### Nível 1: Embedding Básico

O MVP: computar um vetor de embedding para um texto.

#### Novo Message Type no Worker Protocol

```typescript
// Em types.ts — adicionar ao WorkerMessage union type
type WorkerMessage =
    | { type: 'init'; ... }
    | { type: 'infer'; ... }
    | { type: 'embed'; text: string; id: number }   // ← NOVO
    | { type: 'shutdown' }

// Resposta do worker
type WorkerResponse =
    | { type: 'token'; ... }
    | { type: 'embedding'; id: number; vector: Float32Array }  // ← NOVO
    | { type: 'done'; ... }
    | { type: 'error'; ... }
```

#### Novas FFI Bindings

```typescript
// Em ffi.ts — interface LibLlama
llama_model_n_embd: (model: number) => number
llama_pooling_type: (ctx: number) => number
llama_get_embeddings_seq: (ctx: number, seqId: number) => number  // retorna ptr
llama_get_embeddings_ith: (ctx: number, i: number) => number      // retorna ptr

// Em ffi.ts — interface LibShims
shim_ctx_params_set_embeddings: (buf: Buffer, v: boolean) => void
shim_ctx_params_set_pooling_type: (buf: Buffer, type: number) => void
shim_encode: (ctx: number, batchBuf: Buffer) => number
```

#### EmbeddingContext Mode no Contexto Existente

Na inicialização do modelo, uma flag `embeddings: true` no `ModelConfig` ativa o modo embedding:

```typescript
// Em inference.ts — modificação do initModel()
if (config.embeddings) {
    S.shim_ctx_params_set_embeddings(ctxParamsBuf, true)
    if (config.poolingType !== undefined) {
        S.shim_ctx_params_set_pooling_type(ctxParamsBuf, config.poolingType)
    }
}
```

#### API Pública

```typescript
// Uso simples
const llm = await LlamaModel.load('./nomic-embed-text.gguf', {
    embeddings: true,
    poolingType: 1,  // MEAN
})

const vector = await llm.embed('Hello world')
// → Float32Array(768) [0.023, -0.041, 0.089, ...]

await llm.dispose()
```

Internamente, `embed()` segue o mesmo pattern de `infer()`:
1. Enfileira na SerialQueue
2. Envia mensagem `{ type: 'embed', text, id }` ao Worker
3. Worker tokeniza, faz `llama_encode` (não `llama_decode`), extrai embedding
4. Worker responde com `{ type: 'embedding', vector }` via `postMessage` (transferindo o ArrayBuffer)

### Nível 2: Batch Embeddings

#### True Batch Embedding

O llama.cpp suporta múltiplas sequências em um único batch. Em vez de processar um texto por vez (como faz node-llama-cpp), podemos adicionar tokens de múltiplos textos no mesmo batch com `seq_id` diferentes:

```typescript
// Worker side — batch embedding
function embedBatch(texts: string[], libs: Libraries, state: LlamaState): Float32Array[] {
    const { L, S } = libs
    const n_embd = L.llama_model_n_embd(state.model)

    for (let i = 0; i < texts.length; i++) {
        const tokens = tokenize(texts[i], libs, state)
        for (let j = 0; j < tokens.length; j++) {
            // Cada texto recebe seu próprio seq_id
            S.shim_batch_add(state.batchBuf, tokens[j], j, /*seq_id=*/ i, j === tokens.length - 1)
        }
    }

    S.shim_encode(state.ctx, state.batchBuf)  // Um único forward pass!

    const results: Float32Array[] = []
    for (let i = 0; i < texts.length; i++) {
        const ptr = L.llama_get_embeddings_seq(state.ctx, i)
        results.push(readEmbedding(ptr, n_embd))
    }
    return results
}
```

#### Zero-Copy Float Array Transfer (SharedArrayBuffer)

Para evitar cópias ao transferir embeddings do Worker para a main thread:

```typescript
// Worker: escreve diretamente no SharedArrayBuffer
const shared = new SharedArrayBuffer(textsCount * n_embd * 4)
const view = new Float32Array(shared)

for (let i = 0; i < textsCount; i++) {
    const ptr = L.llama_get_embeddings_seq(state.ctx, i)
    const embedding = readEmbedding(ptr, n_embd)
    view.set(embedding, i * n_embd)  // copia do ponteiro C para shared memory
}

// Main thread: lê diretamente — sem transferência
```

#### API Pública

```typescript
const vectors = await llm.embedBatch([
    'first document text',
    'second document text',
    'third document text',
])
// → Float32Array[] — um vetor por texto

// Com limites automáticos baseados no context size
const vectors = await llm.embedBatch(largeArrayOfTexts, {
    maxBatchSize: 32,      // Máx textos por forward pass
    onProgress: (done, total) => { ... },
})
```

### Nível 3: Reranking

#### Ranking Context com POOLING_TYPE_RANK

```typescript
const ranker = await LlamaModel.load('./qwen3-reranker-0.6b.gguf', {
    embeddings: true,
    poolingType: 4,  // LLAMA_POOLING_TYPE_RANK
})
```

#### Template-Based Query/Document Encoding

```typescript
// Construção do input para cross-encoder
function buildRankingInput(
    query: string, document: string,
    libs: Libraries, state: LlamaState,
    template?: string
): Int32Array {
    if (template) {
        // Template do modelo: "<query>{{query}}</query><document>{{document}}</document>"
        return tokenizeTemplate(template, query, document, libs, state)
    }

    // Formato default: [BOS] query [EOS] [SEP] document [EOS]
    const vocab = state.vocab
    const bos = L.llama_vocab_bos(vocab)
    const eos = L.llama_vocab_eos(vocab)
    // SEP = token 102 para modelos BERT (hardcoded como fallback)
    return new Int32Array([bos, ...queryTokens, eos, sep, ...docTokens, eos])
}
```

#### API Pública

```typescript
const results = await ranker.rank('what is machine learning?', [
    'Machine learning is a subset of artificial intelligence...',
    'The weather forecast for tomorrow shows rain...',
    'Deep learning uses neural networks with many layers...',
])
// → [
//     { document: 'Machine learning is a subset...', score: 0.94 },
//     { document: 'Deep learning uses neural...', score: 0.87 },
//     { document: 'The weather forecast...', score: 0.03 },
//   ]
```

Internamente:
1. Constrói input concatenado (query + documento) para cada par
2. Faz `llama_encode` com `POOLING_TYPE_RANK`
3. Extrai logit único via `llama_get_embeddings_seq(ctx, seq_id)`
4. Aplica sigmoid: `score = 1 / (1 + exp(-logit))`
5. Ordena por score decrescente

### Nível 4: Inovações (Além das Referências)

#### Embedding Pool: Workers Dedicados para Embeddings

Nem node-llama-cpp nem qmd resolvem o problema fundamental: **embeddings bloqueiam inferência**.

```
┌─ Main Thread ─────────────────────────────────────────┐
│                                                       │
│  LlamaModel (generativo)    LlamaEmbedPool            │
│  ┌──────────────┐          ┌───────────────────┐      │
│  │ Worker #0    │          │ Worker #1 (embed) │      │
│  │ infer()      │          │ Worker #2 (embed) │      │
│  │ (não bloqueia│          │ Worker #3 (embed) │      │
│  │  embeddings) │          └───────────────────┘      │
│  └──────────────┘          Round-robin dispatch        │
│                            n_embd floats via transfer  │
└───────────────────────────────────────────────────────┘
```

- Pool de N workers dedicados a embeddings (separados da inferência)
- Round-robin dispatch com backpressure
- Modelo de embedding carregado em cada worker (compartilha memória via mmap do OS)
- Inferência nunca é bloqueada por embedding e vice-versa

```typescript
// API conceitual
const pool = await LlamaEmbedPool.create('./nomic-embed-text.gguf', {
    workers: 4,          // 4 workers paralelos
    poolingType: 1,      // MEAN
})

// Processa 1000 documentos em paralelo real
const vectors = await pool.embedBatch(documents, {
    onProgress: (done, total) => console.log(`${done}/${total}`),
})
```

#### SIMD-Accelerated Similarity

Cosine similarity em JavaScript puro é O(n) com overhead de interpretação. Via bun:ffi, podemos usar operações SIMD nativas:

```c
// Em llama_shims.c — similaridade vetorial via SIMD (ARM NEON / x86 SSE)
float shim_cosine_similarity(const float *a, const float *b, int32_t n) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    // O compilador auto-vetoriza com -O2 + NEON/SSE
    for (int i = 0; i < n; i++) {
        dot    += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
    return dot / (sqrtf(norm_a) * sqrtf(norm_b));
}

float shim_dot_product(const float *a, const float *b, int32_t n) {
    float dot = 0.0f;
    for (int i = 0; i < n; i++) dot += a[i] * b[i];
    return dot;
}
```

**Benchmark esperado**: 5-10× mais rápido que JS para vetores de dimensão 768+, especialmente em batch (milhares de comparações).

#### Streaming Embeddings

Gera embeddings incrementalmente durante inferência, permitindo RAG em tempo real:

```typescript
// Conceito: embedding parcial atualizado a cada N tokens gerados
const stream = llm.inferWithEmbeddings(prompt, {
    embeddingInterval: 50,  // emite embedding atualizado a cada 50 tokens
    onEmbedding: (partialEmbedding, tokensSoFar) => {
        // Pode fazer retrieval adicional baseado no que está sendo gerado
        // "O modelo está falando sobre X → buscar mais contexto sobre X"
    },
    onToken: (text) => process.stdout.write(text),
})
```

#### Hybrid Model: Embedding + Generativo em Uma Lib

A maioria dos frameworks RAG requer orquestração externa entre modelo de embedding e modelo generativo. bun-llama-cpp pode oferecer isso nativamente:

```typescript
// Uma única lib gerencia ambos os modelos
const rag = await LlamaRAG.create({
    embedModel: './nomic-embed-text.gguf',
    genModel: './llama-3.2-3b.gguf',
    embedWorkers: 2,
})

// Indexa documentos
await rag.index(documents)

// Query com retrieval + geração automática
const answer = await rag.query('What is bun-llama-cpp?', {
    topK: 5,
    onToken: (text) => process.stdout.write(text),
})
```

Internamente:
1. Embedding pool computa vetor da query
2. Busca top-K documentos similares (cosine similarity SIMD)
3. Worker de inferência gera resposta com contexto dos documentos

#### Quantized Embeddings

Matryoshka embeddings e binary embeddings para economia drástica de memória:

```typescript
// Matryoshka: trunca vetor para dimensão menor (modelo treinado para isso)
const full = await llm.embed(text)          // Float32Array(768)
const half = full.slice(0, 384)             // Ainda útil! Matryoshka property
const quarter = full.slice(0, 192)          // Menos preciso, 4× menos memória

// Binary: quantiza para 1 bit por dimensão
function toBinaryEmbedding(vec: Float32Array): Uint8Array {
    const bits = new Uint8Array(Math.ceil(vec.length / 8))
    for (let i = 0; i < vec.length; i++) {
        if (vec[i] > 0) bits[i >> 3] |= (1 << (i & 7))
    }
    return bits  // 768 dims → 96 bytes (vs 3072 bytes para float32)
}

// Hamming distance para busca ultra-rápida em binary embeddings
function hammingDistance(a: Uint8Array, b: Uint8Array): number {
    let dist = 0
    for (let i = 0; i < a.length; i++) {
        let xor = a[i] ^ b[i]
        while (xor) { dist++; xor &= xor - 1 }  // popcount
    }
    return dist
}
```

#### Adaptive Chunking

Em vez de chunks fixos de 900 tokens, usa o próprio modelo de embedding para detectar boundaries semânticos:

```typescript
// Conceito: chunking baseado em queda de similaridade
async function adaptiveChunk(text: string, llm: LlamaModel): Promise<string[]> {
    const sentences = splitSentences(text)
    const embeddings = await llm.embedBatch(sentences)

    const chunks: string[] = []
    let currentChunk: string[] = [sentences[0]]

    for (let i = 1; i < sentences.length; i++) {
        const similarity = cosineSimilarity(embeddings[i - 1], embeddings[i])

        if (similarity < 0.5 || currentTokenCount > 800) {
            // Queda de similaridade = mudança de tópico = boundary de chunk
            chunks.push(currentChunk.join(' '))
            currentChunk = [sentences[i]]
        } else {
            currentChunk.push(sentences[i])
        }
    }
    chunks.push(currentChunk.join(' '))
    return chunks
}
```

---

## Impacto na Arquitetura

### Novos Shims C Necessários

| Shim | Assinatura | Propósito |
|---|---|---|
| `shim_ctx_params_set_embeddings` | `(params*, bool) → void` | Habilitar modo embedding |
| `shim_ctx_params_set_pooling_type` | `(params*, int32_t) → void` | Configurar pooling (MEAN/CLS/RANK) |
| `shim_encode` | `(ctx*, batch*) → int32_t` | Forward pass sem KV cache |
| `shim_cosine_similarity` | `(float*, float*, int32_t) → float` | Similaridade SIMD (nível 4) |
| `shim_set_causal_attn` | `(ctx*, bool) → void` | Desativar atenção causal (reranking) |

### Mudanças no Worker Protocol

```typescript
// Novos tipos de mensagem
type WorkerMessage =
    | { type: 'init'; config: ModelConfig }     // existente (+ embeddings flag)
    | { type: 'infer'; ... }                    // existente
    | { type: 'embed'; text: string; id: number }           // NOVO
    | { type: 'embedBatch'; texts: string[]; id: number }   // NOVO
    | { type: 'rank'; query: string; documents: string[]; id: number }  // NOVO
    | { type: 'shutdown' }                      // existente
```

### Nova API Pública

| Método | Tipo de Retorno | Nível |
|---|---|---|
| `llm.embed(text)` | `Promise<Float32Array>` | 1 |
| `llm.embedBatch(texts[])` | `Promise<Float32Array[]>` | 2 |
| `llm.rank(query, documents[])` | `Promise<{document, score}[]>` | 3 |
| `LlamaEmbedPool.create(model, opts)` | Pool de embedding workers | 4 |

### Considerações de Memória

| Cenário | Memória | Cálculo |
|---|---|---|
| 1 embedding (768 dims) | 3 KB | 768 × 4 bytes |
| 1000 embeddings | 3 MB | 1000 × 3 KB |
| 100K embeddings | 300 MB | 100000 × 3 KB |
| Batch de 32 textos (context) | ~50 MB | KV cache + batch buffer |

**Float32 vs Float64**: Usar `Float32Array` (como os dados nativos) economiza 50% de memória comparado com a abordagem do node-llama-cpp que converte para `Float64Array`.

---

## Comparação Final

| Feature | node-llama-cpp | qmd | bun-llama-cpp (proposto) |
|---|---|---|---|
| **Embedding básico** | ✅ `getEmbeddingFor()` | ✅ via node-llama-cpp | 🔲 Nível 1 |
| **Batch embedding** | ❌ Serial (faux-parallel) | ✅ Multi-context pool | 🔲 Nível 2 (true batch via seq_id) |
| **Reranking** | ✅ Cross-encoder | ✅ Parallel contexts | 🔲 Nível 3 |
| **Pooling configurável** | ⚠️ Fixo na criação | ⚠️ Fixo na criação | 🔲 Runtime configurable |
| **Normalização L2** | ❌ | ❌ | 🔲 Opção built-in |
| **Float precision** | Float64 (desperdício) | Float64 (herda) | Float32 (nativo) |
| **Caching de embeddings** | ❌ | ✅ Content-addressed | 🔲 Opcional |
| **Similarity SIMD** | ❌ JS puro | ❌ JS puro | 🔲 Nível 4 (C via FFI) |
| **Embedding pool** | ❌ | ⚠️ Multi-context, single worker | 🔲 Nível 4 (multi-worker) |
| **Chunking inteligente** | ❌ (não é responsabilidade) | ✅ Break-point scoring | ❌ (não é responsabilidade) |
| **Formatação assimétrica** | ❌ | ✅ Nomic/Qwen3 | ❌ (responsabilidade do consumidor) |
| **AbortSignal** | ⚠️ Só na criação | ❌ | 🔲 Durante embedding |
| **Separação embed/infer** | ❌ (mesma thread) | ❌ (mesma thread) | 🔲 Nível 4 (worker pool separado) |
| **Binary/quantized embeddings** | ❌ | ❌ | 🔲 Nível 4 |
| **Hybrid RAG** | ❌ (consumidor orquestra) | ✅ (built-in pipeline) | 🔲 Nível 4 (lib-level) |

### Vantagem Competitiva

O bun-llama-cpp pode se diferenciar em três eixos:

1. **True batch via multi-sequence**: Diferente de node-llama-cpp (serial) ou qmd (multi-context), usar múltiplos `seq_id` no mesmo batch permite embedding de vários textos em um único forward pass — potencialmente 10-30× mais rápido.

2. **Worker pool separado**: Embeddings nunca bloqueiam inferência e vice-versa. Nenhuma outra lib JS oferece isso.

3. **Zero overhead de conversão**: Float32 nativo do modelo → Float32Array no JS, sem conversão para Float64. Metade da memória, transferência zero-copy via SharedArrayBuffer.
