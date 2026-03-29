# Estratégia Consolidada: bun-llama-cpp

> Documento de estratégia consolidando todas as análises de referência, análises temáticas e gap analysis
> em um plano de ação unificado para transformar bun-llama-cpp na lib JS/TS mais performática para llama.cpp.

---

## Sumário Executivo

### Onde estamos hoje

O bun-llama-cpp cobre **~37% da API do llama.cpp** (66 de ~180 funções vinculadas). A lib oferece hoje:

- Carregamento de modelo GGUF com presets (small/medium/large)
- Inferência serial com streaming de tokens via `onToken` callback
- Cadeia de samplers configurável (top-k, top-p, min-p, temperature, seed)
- Cancelamento via `SharedArrayBuffer` + `Atomics` (funciona mesmo com event loop bloqueado)
- Worker thread isolado — FFI nunca bloqueia a main thread
- Shutdown gracioso com liberação de buffers Metal/GPU

A arquitetura atual é **single model, single context, single sequence, serial queue**.

### Onde queremos chegar

A lib JS/TS mais performática para llama.cpp, com:

- **Embeddings** (RAG, busca semântica, reranking)
- **Sequences paralelas** com prefix sharing e continuous batching
- **Multi-modelo** com gestão inteligente de VRAM
- **Grammar/constrained generation** para JSON mode e structured output
- **API composable** — primitivas que se combinam, não monolito

### Vantagem competitiva

| Aspecto | node-llama-cpp | bun-llama-cpp |
|---------|---------------|---------------|
| Binding | N-API C++ addon (compilação cmake-js por plataforma) | `bun:ffi` direto (`dlopen`, zero compilação nativa) |
| Overhead de FFI call | ~50ns (N-API marshalling) | ~10ns (chamada direta) |
| Thread model | Main thread (N-API thread-safe functions) | Worker thread isolado — main thread nunca bloqueia |
| Abort durante FFI | Callback-based (requer event loop) | `SharedArrayBuffer` + `Atomics` (bypassa event loop bloqueado) |
| Embedding format | Float64Array (2× desperdício de memória) | Float32Array nativo (zero conversão) |
| Portabilidade | Binário pré-compilado por plataforma | `dlopen` universal — zero build step nativo |

**A combinação de bun:ffi + Workers + SharedArrayBuffer cria uma stack com latência mínima e controle máximo que nenhum outro binding JS oferece.**

---

## Análise das Referências

### node-llama-cpp — O Que Faz Bem

1. **VRAM awareness completo** — `MemoryOrchestrator` com reservation-based tracking previne overcommit antes de alocar. Três orchestrators (VRAM/RAM/Swap) consultam estado real do hardware. Padrão essencial para qualquer sistema multi-modelo.

2. **Auto GPU layers com binary search** — O modo `gpuLayers: "auto"` faz binary search com scoring: estima VRAM via parsing GGUF, pontua configurações por layers GPU + context size, e retorna a melhor que cabe na memória. Gold standard que devemos replicar.

3. **Hierarquia de objetos clara** — `Llama → Model → Context → Sequence → Session` é uma progressão natural que escala de uso simples a complexo. Cada nível tem responsabilidades bem definidas.

4. **Batch scheduling sofisticado** — Fila producer-consumer com estratégias plugáveis (`maximumParallelism` distribui budget igualmente; `firstInFirstOut` prioriza completar sequences). Retry com shrink automático (até 16×, -16% cada) na criação de contexto.

5. **Context shift automático** — Quando a sequence enche, apaga tokens do início preservando BOS. Suporta estratégia custom. Detalhe crítico que eles acertam.

6. **Detecção de capabilities via GGUF** — Parse offline de tensors e metadados determina se modelo suporta embedding, ranking, encoder/decoder, pooling type — tudo sem carregar o modelo. Permite decisões inteligentes antes de alocar VRAM.

7. **Speculative decoding completo** — Draft model + input lookup prediction com validação por batch. O `InputLookupTokenPredictor` é zero-cost para tarefas grounded no input (sumarização, code editing).

### node-llama-cpp — O Que Não Faz Bem

1. **Sem batch embedding real** — Cada texto é avaliado separadamente com clear completo do KV cache. Para 1000 textos, são 1000 forward passes sequenciais. Uma abordagem com múltiplos `seq_id` no mesmo batch seria dramaticamente mais rápida.

2. **KV cache não compartilhado entre sequences** — Alocação `contextSize × N` desperdiça memória quando sequences compartilham prefixo. Não implementa prefix sharing via `llama_memory_seq_cp()`. Com 8 sequences e system prompt de 500 tokens, processa 4000 tokens quando bastaria processar 500 + 7 cópias O(1).

3. **Over-engineering do disposal** — 3 camadas (DisposeGuard + AsyncDisposeAggregator + WeakRef/FinalizationRegistry) tornam o fluxo de cleanup difícil de raciocinar. O modelo Worker + `terminate()` é inerentemente mais simples e seguro.

4. **`Promise.all` enganoso no ranking** — `rankAll()` usa `Promise.all` que parece paralelo mas executa serialmente por causa do `withLock`. API misleading que induz o consumidor a acreditar que há paralelismo.

5. **Float64 desnecessário** — Embeddings armazenados como `number[]` (64-bit) quando o modelo produz float32. Desperdício de 2× memória sem ganho de precisão.

6. **Sem eviction ou prioridade de modelos** — Se VRAM enche, o usuário deve manualmente fazer dispose. Sem LRU, sem prioridade, sem eviction automática.

7. **N-API requer compilação** — cmake-js + binários pré-compilados por plataforma. Frágil em ambientes novos, edge, ou containers. Nosso `dlopen` universal elimina esse problema.

### qmd — O Que Faz Bem

1. **Pipeline de orquestração 3 modelos** — Embed (300M) → Rerank (600M) → Generate (1.7B) com lifecycle independente. Demonstra o padrão correto para RAG end-to-end.

2. **Lazy loading com promise dedup** — O guard de promise previne carregamento duplo de VRAM quando múltiplas operações tentam carregar o mesmo modelo. 10 requests durante loading esperam a mesma promise. Padrão diretamente aplicável ao nosso Worker.

3. **Inactivity lifecycle com ref-counting** — Timer de 5 min + `_activeSessionCount` previne disposal durante operações ativas. `canUnload()` verifica sessions E operações in-flight. Seguro e eficiente.

4. **Formatação assimétrica query/document** — Modelos de embedding para retrieval aprendem representações diferentes para queries e documentos. Detecção automática de modelo (nomic vs qwen3) com formatação específica. Essencial para qualidade de retrieval.

5. **Smart chunking com proteção de code fences** — Break points com prioridade por score (h1=100, h2=90, parágrafo=20), decaimento quadrático por distância, proteção de code blocks. Chunking two-pass: char-based primeiro (rápido), token verification depois (preciso).

6. **Busca híbrida 8 etapas** — BM25 + vetorial + reranking com RRF fusion, strong signal bypass (~500ms economia), blending position-aware. Referência completa de como orquestrar um pipeline RAG.

7. **Cache content-addressed** — Chunks idênticos compartilham scores de reranking via hash SHA-256. Dedup previne scoring redundante, economizando compute significativo.

### qmd — O Que Não Faz Bem

1. **Dependência pesada de node-llama-cpp** — Herda todas as limitações (serial, sem batch real, Float64). Upgrade friction significativo — cada versão pode quebrar APIs internas.

2. **Singleton global** — `getDefaultLlamaCpp()` impõe um conjunto de modelos por processo. Múltiplas configurações de índice no mesmo processo precisam de overrides explícitos.

3. **Embedding single-threaded por contexto** — Apesar do pool de contextos (1-8), cada contexto processa sequencialmente via `getEmbeddingFor()`. Não é true batch.

4. **Truncação silenciosa** — Documentos excedendo a janela de contexto são truncados com apenas `console.warn`. Consumidores downstream não sabem que conteúdo foi perdido.

5. **Sem streaming** — Embedding e reranking são totalmente batch, sem resultados incrementais. Para coleções grandes, o usuário espera sem feedback.

6. **Sem prioridade entre sessões** — Uma sessão longa de indexação bloqueia cleanup de idle mesmo se sessões de busca precisam de recursos.

7. **sqlite-vec com limitações de JOIN** — Tabela virtual não suporta JOINs (workaround de dois passos mandatório). Over-fetch 3× para compensar.

---

## Princípios de Design do bun-llama-cpp

### 1. Performance-first

FFI direto via `bun:ffi`, zero-copy com `SharedArrayBuffer`, Float32 nativo. Cada camada de abstração é opcional — o consumidor que não precisa dela não paga por ela. Latência de chamada FFI ~5× menor que N-API.

### 2. Controle fino

O consumidor escolhe o nível de abstração:
- **Simples**: `LlamaModel.load()` → `infer()` → `dispose()` (já funciona hoje)
- **Intermediário**: Embeddings, grammar, métricas de performance
- **Avançado**: Sequences paralelas, scheduling, prefix sharing, pipeline cascading

Cada nível é opt-in. Quem só quer inferência simples não precisa saber sobre sequences.

### 3. Composable

Primitivas que se combinam ao invés de monolito:
- `LlamaModel` (inferência/embedding single)
- `ModelRegistry` (coordenação multi-modelo)
- `LlamaEmbedPool` (pool de workers dedicados a embedding)
- `ModelPipeline` (orquestração embed→rerank→generate)

Cada componente funciona independente e combina com os outros.

### 4. Bun-native

Explorar tudo que Bun oferece:
- **Workers** com `SharedArrayBuffer` para abort e métricas zero-copy
- **`bun:ffi`** com `toArrayBuffer()` para leitura de float pointers
- **`Bun.file`** para operações de arquivo
- **`bun:sqlite`** para storage de embeddings (futuro)
- **JavaScriptCore** — tight loops mais rápidos que V8 para batch construction e scheduling

---

## Features a Implementar

### Fase 1: Fundamentos (quick wins)

Features de alto impacto com baixa dificuldade — resultados imediatos sem mudanças arquiteturais.

#### Repetition penalties no sampler chain

```typescript
// Novo config
interface SamplerConfig {
  // ... existentes (topK, topP, minP, temp, seed) ...
  repeatPenalty?: number     // penalidade por repetição (default: 1.1)
  frequencyPenalty?: number  // penalidade por frequência
  presencePenalty?: number   // penalidade por presença
  penaltyLastN?: number      // janela de lookback (default: 64)
}
```

**Implementação**: Um novo shim `shim_sampler_init_penalties(last_n, repeat, freq, present)` + adição na cadeia de samplers antes do distribution sampler. Impacto direto na qualidade de geração — repetição é o problema #1 relatado por usuários de LLMs locais.

#### Model metadata/info

Vincular funções diretas do llama.cpp (sem shim — retornam escalares):

- `llama_model_n_params` — total de parâmetros
- `llama_model_n_embd` — dimensão de embedding
- `llama_model_n_ctx_train` — context window de treino
- `llama_model_n_layer` — número de layers
- `llama_model_desc` — descrição textual
- `llama_model_size` — tamanho em bytes

```typescript
const info = await llm.getModelInfo()
// { nParams: 8_000_000_000, nEmbd: 4096, nCtxTrain: 131072, nLayers: 32, desc: "LLaMA v3.2", size: 4_500_000_000 }
```

Essencial para auto-sizing de GPU layers, validação de modelo, e UX.

#### Chat template support

`llama_chat_apply_template` permite formatar mensagens de chat usando o template embutido no modelo (Llama 3, ChatML, etc.). Novo shim para construir array de `llama_chat_message`:

```typescript
const formatted = await llm.applyTemplate([
  { role: 'system', content: 'You are a helpful assistant.' },
  { role: 'user', content: 'Hello!' },
])
// → "<|begin_of_text|><|start_header_id|>system..."
```

**Impacto**: Elimina a necessidade do consumidor conhecer o formato de template de cada modelo. Table-stakes para qualquer API de chat.

#### FIM tokens (Fill-in-Middle)

Vincular tokens especiais para code completion: `llama_vocab_fim_pre/suf/mid/pad/rep/sep`. Já existem no vocab API, apenas não estão vinculados. Essencial para integrações de IDE e code assistants.

#### Performance metrics

Vincular `llama_perf_context` e `llama_perf_sampler` para métricas de performance:

```typescript
const result = await llm.infer(prompt, { onToken, maxTokens: 200 })
console.log(result.metrics)
// { promptTokens: 150, generatedTokens: 200, promptMs: 340, generateMs: 2800, tokensPerSec: 71.4 }
```

Custo quase zero, valor alto para debugging e benchmarking.

### Fase 2: Embeddings

O gap #1 do projeto — sem embeddings, não é possível construir pipelines RAG.

#### EmbeddingContext mode

Uma flag `embeddings: true` no `ModelConfig` ativa o modo embedding:

```typescript
const embed = await LlamaModel.load('./nomic-embed-text.gguf', {
  embeddings: true,
  poolingType: 1,  // MEAN (ou CLS=2, LAST=3, RANK=4)
})

const vector = await embed.embed('Hello world')
// → Float32Array(768) [0.023, -0.041, 0.089, ...]
```

**Novos shims**: `shim_ctx_params_set_embeddings`, `shim_ctx_params_set_pooling_type`, `shim_encode` (forward pass sem KV cache).

**Novas bindings diretas**: `llama_model_n_embd`, `llama_pooling_type`, `llama_get_embeddings_seq`, `llama_get_embeddings_ith`.

**Leitura de float pointer**: `toArrayBuffer(ptr, 0, n_embd * 4)` → `Float32Array`. Zero conversão, metade da memória vs node-llama-cpp (Float64).

#### Batch embedding

Diferente de node-llama-cpp (serial) e qmd (multi-context), usar múltiplos `seq_id` no mesmo batch para true batch embedding:

```typescript
const vectors = await embed.embedBatch([
  'first document text',
  'second document text',
  'third document text',
])
// → Float32Array[] — um único forward pass para todos os textos!
```

Um único `llama_encode()` processa todos os textos com `seq_id` diferentes — potencialmente **10-30× mais rápido** que a abordagem sequencial do node-llama-cpp.

#### Reranking com cross-encoder

Context mode com `POOLING_TYPE_RANK` para cross-encoder scoring:

```typescript
const ranker = await LlamaModel.load('./qwen3-reranker-0.6b.gguf', {
  embeddings: true,
  poolingType: 4,  // RANK
})

const results = await ranker.rank('what is machine learning?', [
  'Machine learning is a subset of AI...',
  'The weather forecast shows rain...',
])
// → [{ document: '...', score: 0.94 }, { document: '...', score: 0.03 }]
```

Logit → sigmoid → probabilidade de relevância 0..1. Ordenação automática por score.

#### Cosine similarity built-in

Normalização L2 opcional (`normalize: true`) e funções de similaridade:

```typescript
import { cosineSimilarity, dotProduct } from 'bun-llama-cpp'

const sim = cosineSimilarity(vectorA, vectorB)  // -1..1
const dot = dotProduct(vectorA, vectorB)
```

Na Fase 5, estas funções serão aceleradas via SIMD (FFI para C com auto-vectorization).

### Fase 3: Sequences Paralelas

Transformar o bun-llama-cpp de serial para concurrent — throughput multiplicado.

#### Multi-sequence no mesmo context

O shim `shim_batch_add` já aceita `seq_id` — hoje é chamado sempre com 0. Expor via config:

```typescript
const llm = await LlamaModel.load('./model.gguf', {
  nSeqMax: 4,     // 4 sequences paralelas
  nCtx: 4096,     // context per sequence
})

// 4 requests simultâneas — todas executam em paralelo real
const [r1, r2, r3, r4] = await Promise.all([
  llm.infer("User: Hello",  { onToken: t1, maxTokens: 100 }),
  llm.infer("User: Hi",     { onToken: t2, maxTokens: 200 }),
  llm.infer("User: Hey",    { onToken: t3, maxTokens: 150 }),
  llm.infer("User: Howdy",  { onToken: t4, maxTokens: 50 }),
])
```

**Componentes internos**: `SequenceAllocator` (pool de slots com acquire/release), `BatchScheduler` (constrói batches multi-sequence), `per-sequence samplers`.

#### KV cache prefix sharing (INOVAÇÃO)

**node-llama-cpp não faz isso.** Quando múltiplas sequences compartilham system prompt:

```
Sem sharing:  500 tokens × 8 sequences = 4000 tokens processados
Com sharing:  500 tokens × 1 + 7 cópias O(1) = ~500 tokens processados
Speedup: ~8× no prefill do system prompt
```

Via `llama_memory_seq_cp()`, o KV cache do prefix é copiado instantaneamente (referência ao mesmo bloco — copy-on-write semântico). Economia de **~87% no KV cache** do system prompt.

```typescript
await llm.warmup("You are a helpful assistant...")
// Pré-computa KV cache do system prompt; novas sequences copiam em O(1)
```

#### Smart batch scheduling

Estratégias plugáveis:
- **Round-robin** (default): cada sequence processa tokens proporcionalmente ao batch budget
- **Priority**: sequences de alta prioridade processam primeiro
- **Adaptive**: ajusta batch size baseado em utilização e latência

#### Continuous batching (INOVAÇÃO)

**Nenhuma lib JS faz isso.** Inspirado em vLLM/TGI — novas requests entram no batch enquanto outras estão gerando:

```
Static batching (node-llama-cpp):
  Request A ──► [prefill A] [gen A] [gen A] [gen A] ──► done
  Request B ──► [  wait   ] [  wait  ] [prefill B] [gen B] ──► done

Continuous batching (bun-llama-cpp):
  Request A ──► [prefill A] [gen A] [gen A] [gen A] ──► done
  Request B ──► [  wait   ] [prefill B] [gen B] [gen B] ──► done
                               ↑ entra no próximo batch, sem esperar A terminar
```

A diferença fundamental: no static batching, Request B espera A terminar. No continuous batching, B entra no próximo batch step — latência de resposta dramática menor em cenários de server.

### Fase 4: Multi-Modelo

Coordenação de múltiplos modelos com gestão inteligente de recursos.

#### Model registry / worker pool

```typescript
const registry = new ModelRegistry()

await registry.load('embed', './nomic-embed.gguf', { preset: 'small' })
await registry.load('rerank', './qwen3-reranker.gguf', { preset: 'small' })
await registry.load('generate', './llama-3.2.gguf', { preset: 'large' })

const embedModel = registry.get('embed')
const genModel = registry.get('generate')

await registry.unload('embed')     // Libera apenas embedding
await registry.disposeAll()        // Shutdown completo
```

Cada modelo em seu próprio Bun Worker. O registry é um wrapper fino de coordenação — baixa complexidade, alto valor.

#### VRAM tracking (Apple Silicon aware)

No Apple Silicon, GPU e CPU compartilham o mesmo pool de memória (unified memory). O tracker deve:

1. Consultar estado real via `sysctl hw.memsize` + `vm_stat`
2. Manter reservation-based tracking (reserve antes de alocar, libere depois)
3. Evitar double-counting (VRAM = RAM em unified memory)
4. Reservar 30% para OS, Metal driver, e outros processos

```typescript
// VRAM budget para Apple Silicon
// M1 (8GB):     ~5.6GB para modelos
// M1 Pro (16GB): ~11.2GB → Embed 300M + Gen 7B Q4
// M2 Pro (32GB): ~22.4GB → Embed 300M + Rerank 600M + Gen 13B Q4
// M3 Max (64GB): ~44.8GB → Embed 1B + Rerank 1B + Gen 70B Q4
```

#### Lazy loading com lifecycle management

Inspirado no qmd, com melhorias:

```typescript
// Promise dedup — previne double-allocation
// Inactivity timer — libera VRAM de modelos idle
// Session ref-counting — previne disposal durante operações ativas
// Prioridade: persistent > normal > evictable
```

Diferencial vs qmd: timer configurável por modelo, prioridade de eviction, e support a preloading hints.

#### Pipeline cascading (INOVAÇÃO)

Uma única chamada orquestra embed → rerank → generate:

```typescript
const pipeline = new ModelPipeline(embedModel, rerankModel, genModel, vectorStore)

const result = await pipeline.query('Como funciona o garbage collector do V8?', {
  topK: 5,
  rerank: true,
  onToken: (text) => process.stdout.write(text),
})
// → { answer: '...', sources: [...], tokenCount: 342 }
```

Internamente: embedding da query → busca vetorial → reranking → geração com contexto. Nenhuma outra lib JS oferece isso como primitiva.

### Fase 5: Advanced

Features para uso avançado e edge cases.

#### Grammar/constrained generation (JSON mode)

`llama_sampler_init_grammar(vocab, grammar_str, grammar_root)` permite geração restrita a uma gramática GBNF:

```typescript
const result = await llm.infer(prompt, {
  onToken,
  grammar: `root ::= "{" ws "name" ws ":" ws string "," ws "age" ws ":" ws number "}"
             string ::= "\\"" [^"\\\\]* "\\""
             number ::= [0-9]+
             ws ::= [ \\t\\n]*`,
})
// Garante que a saída é JSON válido
```

Alternativa simplificada:

```typescript
const result = await llm.infer(prompt, { onToken, jsonMode: true })
// Usa grammar pré-definida para JSON genérico
```

**Impacto**: Table-stakes para agent/tool-use patterns. Structured output é requisito para qualquer integração com código.

#### Speculative decoding

Draft model gera candidatos, main model valida em batch:

```typescript
const llm = await LlamaModel.load('./llama-3.2-3b.gguf', {
  speculative: {
    draftModel: './llama-3.2-1b.gguf',
    maxPredictions: 8,
    minConfidence: 0.6,
  },
})
```

Requer multi-modelo + batch decode. Potencial de 2-3× speedup para modelos grandes.

#### KV cache quantization

Redução de memória do KV cache via tipos quantizados:

```typescript
const llm = await LlamaModel.load('./model.gguf', {
  kvCacheType: 'q8_0',  // 50% menos memória (vs f16 default)
  // kvCacheType: 'q4_0'  // 75% menos memória
})
```

Para modelo 7B com context 4096 e 8 sequences:
- F16/F16: 512 MB
- Q8/Q8: 256 MB (50% redução)
- Q4/Q4: 128 MB (75% redução)

#### Context window management (shift/trim)

Quando `pos >= n_ctx`, automaticamente:
1. Detecta overflow
2. `llama_memory_seq_rm` para dropar os N tokens mais antigos
3. `llama_memory_seq_add` com delta -N para shift posições
4. Preserva BOS token sempre

```typescript
const llm = await LlamaModel.load('./model.gguf', {
  contextShift: true,  // habilita shift automático
  contextShiftSize: 256,  // remove 256 tokens quando overflow
})
```

Permite geração "infinita" sem o consumidor gerenciar context window.

---

## Inovações Originais (Além das Referências)

Consolidação das inovações propostas nos documentos temáticos. Estas são as features que diferenciam bun-llama-cpp de qualquer outra lib JS:

### 1. KV Cache Prefix Sharing

**node-llama-cpp não faz isso.** Quando múltiplas sequences compartilham system prompt, o prefill é feito UMA VEZ e o KV cache é copiado instantaneamente para as demais via `llama_memory_seq_cp()`. Economia de ~87% de KV cache no system prompt e ~8× speedup no prefill para cenários de server.

**Impacto**: Alto — qualquer cenário de server com system prompt compartilhado (chatbots, assistentes, API endpoints).

### 2. Continuous Batching

**Nenhuma lib JS faz isso.** Inspirado em vLLM e TGI, novas requests entram no batch sem esperar as anteriores terminarem. Enquanto node-llama-cpp usa static batching (Request B espera A), bun-llama-cpp insere B no próximo step do batch loop. Redução drástica de latência P99 em cenários de alta concorrência.

**Impacto**: Muito alto — transformador para uso em server. Diferenciador competitivo claro.

### 3. Pipeline Cascading API

Uma única chamada orquestra embed → busca vetorial → rerank → generate. Nenhuma outra lib JS oferece isso como primitiva — o consumidor sempre precisa orquestrar manualmente. Com lazy loading e promise dedup, modelos são carregados sob demanda.

**Impacto**: Alto — reduz de ~50 linhas de código para 1 chamada. Developer experience transformadora para RAG.

### 4. Zero-Copy Embedding Transfer

`SharedArrayBuffer` para transferência de float arrays entre Worker e main thread sem cópia. O Worker escreve embeddings diretamente no shared memory; a main thread lê sem `postMessage`. Combinado com Float32 nativo (vs Float64 do node-llama-cpp), resulta em **4× menos overhead** que a alternativa.

**Impacto**: Médio-alto — significativo para batch embedding de milhares de documentos.

### 5. Embedding Worker Pool

Nem node-llama-cpp nem qmd resolvem o problema fundamental: **embeddings bloqueiam inferência**. Com Workers separados, embedding e inferência nunca competem por thread time:

```
Worker #0 (generativo): infer() — nunca bloqueado por embeddings
Worker #1-3 (embedding): embed() — round-robin dispatch com backpressure
```

O modelo de embedding é carregado em cada worker (compartilha memória via mmap do OS). Inferência nunca é degradada por batch de embeddings.

**Impacto**: Alto — essencial para aplicações RAG em produção.

### 6. SIMD Similarity via FFI

Cosine similarity e dot product via C compilado com auto-vectorization (ARM NEON no Apple Silicon). **5-10× mais rápido** que JS puro para vetores de dimensão 768+, especialmente em batch (milhares de comparações).

```c
// O compilador auto-vetoriza com -O2 + NEON
float shim_cosine_similarity(const float *a, const float *b, int32_t n);
```

**Impacto**: Médio — relevante quando o gargalo é busca vetorial em-memória (sem sqlite-vec).

### 7. SharedArrayBuffer Metrics

Métricas zero-copy via `SharedArrayBuffer` + `Atomics`. Worker atualiza atomicamente; main thread lê sem `postMessage`:

```typescript
// [0] = active_sequences, [1] = pending_requests,
// [2] = total_tokens_generated, [3] = batch_utilization_percent
const active = Atomics.load(metrics, 0)  // Zero latência
```

**Impacto**: Baixo-médio — valor para observabilidade e debugging em produção.

### 8. Preemptive Scheduling

Pausar sequences de baixa prioridade quando memória aperta. Escolhe a "vítima" com menor prioridade e mais progresso (menos trabalho perdido). O KV cache é limpo e a sequence pode ser restaurada via re-prefill com prefix sharing.

**Impacto**: Médio — relevante para server com muitas sequences concorrentes e VRAM limitada.

---

## Vantagens Competitivas vs Alternativas

| Feature | node-llama-cpp | llamafile | bun-llama-cpp (proposto) |
|---------|---------------|-----------|--------------------------|
| **Binding type** | N-API C++ addon | Servidor HTTP embutido | `bun:ffi` direto (dlopen) |
| **Build step nativo** | cmake-js por plataforma | Distribuído como executável | Zero (apenas C shim leve) |
| **FFI latency** | ~50ns (N-API) | N/A (HTTP) | ~10ns (chamada direta) |
| **Thread model** | Main thread | Processo separado | Worker thread isolado |
| **Embedding format** | Float64Array | JSON over HTTP | Float32Array nativo |
| **Batch embedding** | ❌ Sequencial | ❌ Um por request HTTP | ✅ True batch via multi-seq_id |
| **Prefix sharing** | ❌ | ❌ | ✅ `llama_memory_seq_cp` |
| **Continuous batching** | ❌ Static | ❌ | ✅ Inspirado em vLLM |
| **Abort durante FFI** | Callback (precisa event loop) | HTTP timeout | `SharedArrayBuffer` + Atomics |
| **Multi-modelo** | ✅ Shared backend | ❌ Um por processo | ✅ Workers isolados |
| **VRAM tracking** | ✅ MemoryOrchestrator | ❌ | ✅ Reservation-based |
| **Grammar/JSON mode** | ✅ | ✅ | 🔜 Fase 5 |
| **Speculative decoding** | ✅ | ❌ | 🔜 Fase 5 |
| **Metrics zero-copy** | ❌ | ❌ | ✅ SharedArrayBuffer |
| **Pipeline cascading** | ❌ Manual | ❌ | ✅ embed→rerank→generate |
| **Context shift** | ✅ Auto | ❌ | 🔜 Fase 5 |
| **LoRA adapters** | ✅ | ❌ | 🔜 Futuro |

**Legenda**: ✅ Implementado | 🔜 Planejado | ❌ Não suportado

### Onde bun-llama-cpp já vence

- **Performance pura**: FFI 5× mais rápido, Worker não bloqueia main thread, Float32 nativo
- **Zero build step**: Sem cmake, sem compilação por plataforma — `dlopen` universal
- **Abort robusto**: `SharedArrayBuffer` bypassa event loop bloqueado — algo que N-API não consegue

### Onde bun-llama-cpp vai vencer (com o roadmap)

- **Throughput**: Continuous batching + prefix sharing = latência P99 drasticamente menor
- **Embedding performance**: True batch via multi-seq_id — uma ordem de magnitude mais rápido
- **Developer experience**: Pipeline cascading — RAG em uma chamada

### Onde node-llama-cpp ainda vence (e quando fecharemos)

- **Feature completeness**: Grammar, speculative decoding, LoRA → Fase 5
- **VRAM awareness**: MemoryOrchestrator completo → Fase 4
- **Cross-platform GPU**: CUDA/Vulkan detection → Futuro (foco inicial em Metal)

---

## Riscos e Mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|-------|--------------|---------|-----------|
| **llama.cpp API instável** | Alta | Alto | Shim layer absorve mudanças — apenas shims precisam ser atualizados, não a API pública. Versionamento do shim atrelado a releases do llama.cpp. |
| **Bun Worker limitations** | Média | Médio | Workers são threads reais com event loop próprio. Limitação principal: `dlopen` não é compartilhável entre Workers (cada Worker faz seu próprio). Mitigation: OS mmap sharing de model weights é automático. |
| **VRAM management complexity** | Alta | Alto | Começar conservador (60% do total). Reservation-based tracking previne overcommit. Fallback: reduzir GPU layers automaticamente quando VRAM insuficiente. |
| **Metal assertion failures em multi-worker** | Média | Alto | Cada Worker tem seu próprio Metal command queue. Testar empiricamente com 2-3 workers. Se Metal não suporta, serializar loads via mutex na main thread. |
| **Continuous batching correctness** | Alta | Alto | Implementar em fases: 1) multi-sequence estático, 2) dynamic admission, 3) preemption. Cada fase validada com testes antes de avançar. |
| **SharedArrayBuffer de embedding — tamanho** | Baixa | Baixo | 1000 embeddings de 768 dims = 3MB — trivial. Para 100K, considerar streaming em batches de 1000. |
| **Worker overhead de memória** | Baixa | Baixo | Cada Bun Worker consome ~30-50MB base. Para 3 modelos = ~150MB overhead — aceitável. |
| **`llama_backend_init` em múltiplos Workers** | Baixa | Baixo | Cada Worker chama independentemente. llama.cpp suporta múltiplas inicializações — são processos isolados. |

---

## Roadmap Sugerido

### Sequência de implementação

```
Fase 1 ─── Fase 2 ─── Fase 3 ─── Fase 4 ─── Fase 5
(quick     (embed     (parallel  (multi     (advanced)
 wins)      dings)     seqs)      model)
```

### Dependências

```
Fase 1 (fundamentos)
  ├── Repetition penalties ← sem dependências
  ├── Model metadata ← sem dependências
  ├── Chat templates ← sem dependências
  ├── FIM tokens ← sem dependências
  └── Performance metrics ← sem dependências

Fase 2 (embeddings)
  ├── Embedding básico ← requer novos shims + FFI bindings
  ├── Batch embedding ← requer embedding básico + multi-seq support
  ├── Reranking ← requer embedding básico + POOLING_TYPE_RANK
  └── Cosine similarity ← sem dependências (JS puro; SIMD na Fase 5)

Fase 3 (parallel sequences)
  ├── Multi-sequence context ← requer novos shims (n_seq_max, memory_seq_*)
  ├── Prefix sharing ← requer multi-sequence + llama_memory_seq_cp
  ├── Batch scheduler ← requer multi-sequence
  └── Continuous batching ← requer batch scheduler

Fase 4 (multi-modelo)
  ├── Model registry ← sem dependências técnicas (usa API existente)
  ├── VRAM tracker ← requer model metadata (Fase 1)
  ├── Lazy loading + lifecycle ← requer registry
  └── Pipeline cascading ← requer embeddings (Fase 2) + registry

Fase 5 (advanced)
  ├── Grammar/JSON mode ← requer novo shim
  ├── Speculative decoding ← requer multi-modelo (Fase 4) + batch decode
  ├── KV cache quantization ← requer novos shims (type_k, type_v)
  ├── Context shift ← requer llama_memory_seq_rm/add (Fase 3)
  └── SIMD similarity ← requer shim C com NEON/SSE
```

### Justificativa da ordem

1. **Fase 1 primeiro**: Quick wins para feedback loop rápido. Cada feature é independente e pode ser entregue individualmente. Demonstra velocidade de execução.

2. **Embeddings antes de sequences**: Mais demanda do mercado (RAG é o caso de uso #1 para LLMs locais). Cada feature de embedding funciona de forma standalone — valor entregue incrementalmente.

3. **Sequences paralelas depois**: Maior impacto técnico mas exige mudanças arquiteturais significativas no Worker. As inovações (prefix sharing, continuous batching) são o maior diferenciador competitivo.

4. **Multi-modelo requer embeddings**: Pipeline cascading (a inovação principal de Fase 4) depende de embedding support. Model registry pode ser feito antes, mas o valor completo requer Fase 2.

5. **Advanced por último**: Grammar e speculative são features de polish. KV quantization e context shift são incrementais sobre a infraestrutura das fases anteriores.

---

## Conclusão

O bun-llama-cpp está em posição única para se tornar a melhor lib JS/TS para llama.cpp. As fundações já são sólidas:

- **FFI direto via `bun:ffi`** — latência 5× menor que N-API, sem build step nativo
- **Worker thread isolado** — main thread nunca bloqueia, abort funciona mesmo durante FFI
- **Arquitetura de shims** — absorve instabilidade da API do llama.cpp

As inovações propostas não existem em nenhuma outra lib JS:

- **Continuous batching** (vLLM-inspired) — nenhuma lib JS faz
- **KV cache prefix sharing** — node-llama-cpp desperdiça ~87% do KV cache do system prompt
- **True batch embedding** via multi-seq_id — uma ordem de magnitude mais rápido que sequencial
- **Pipeline cascading** — RAG end-to-end em uma chamada
- **Embedding worker pool** — embeddings nunca bloqueiam inferência

O roadmap em 5 fases entrega valor incrementalmente: quick wins primeiro para feedback rápido, embeddings para capturar o caso de uso mais demandado, e then as inovações de throughput que criam o fosso competitivo.

A combinação de `bun:ffi` (zero overhead) + `SharedArrayBuffer` (comunicação zero-copy) + continuous batching (throughput de server) posiciona bun-llama-cpp como a opção definitiva para inferência local em JavaScript/TypeScript.

---

*Documento consolidado a partir das análises de referência (node-llama-cpp, qmd), análises temáticas (parallel sequences, embeddings, multi-model), e gap analysis do projeto.*
