# Análise de Referência: node-llama-cpp

> Documento de referência técnica para desenvolvedores avançados.
> Analisa arquitetura, decisões de design e trade-offs do node-llama-cpp
> como baseline para o projeto bun-llama-cpp.

---

## Visão Geral

### O que é

[node-llama-cpp](https://github.com/withcatai/node-llama-cpp) é a binding Node.js mais madura para llama.cpp. Mantido pela comunidade withcatai, oferece uma API TypeScript completa sobre o runtime nativo, cobrindo inferência, embeddings, ranking, speculative decoding e gerenciamento de VRAM.

### Stack Tecnológica

| Camada | Tecnologia |
|--------|-----------|
| Runtime | Node.js (≥18) |
| Bindings nativos | N-API C++ addon (compilado via cmake-js) |
| Linguagem | TypeScript estrito |
| Build nativo | cmake-js com detecção automática de GPU (Metal/CUDA/Vulkan) |
| Formato de modelo | GGUF (com parser de metadados próprio em JS) |

### Arquitetura Geral

Usa **N-API C++ addon** — compilação nativa por plataforma que expõe classes C++ como objetos JavaScript via `Napi::ObjectWrap`. Isso contrasta diretamente com nosso approach de **bun:ffi + dlopen**, que evita compilação mas exige C shims para structs passadas por valor.

```
┌─────────────────────────────────────────────────────────┐
│  TypeScript API Layer                                    │
│  Llama → LlamaModel → LlamaContext → Sequence → Session │
├─────────────────────────────────────────────────────────┤
│  N-API C++ Addon                                         │
│  AddonModel / AddonContext / AddonSampler                 │
├─────────────────────────────────────────────────────────┤
│  llama.cpp (libllama)                                    │
│  Compilado junto com o addon via cmake                   │
└─────────────────────────────────────────────────────────┘
```

---

## Arquitetura de Bindings

### N-API C++ Addon

O addon é um módulo `.node` compilado que encapsula toda a interação com llama.cpp em classes C++:

- **`AddonModel`** — wrapa `llama_model`, expõe tokenização, metadados, tamanho de embedding
- **`AddonContext`** — wrapa `llama_context`, gerencia batch, decode, extração de embeddings
- **`AddonSampler`** — wrapa cadeia de samplers com configuração dinâmica

**Vantagem**: acesso direto a structs C sem wrappers. O addon pode chamar `llama_model_default_params()` e acessar campos diretamente.

**Desvantagem**: requer compilação nativa por plataforma. O projeto mantém binários prebuilt + fallback para build from source com cmake-js.

### Comparação com nosso approach FFI

| Aspecto | node-llama-cpp (N-API) | bun-llama-cpp (FFI) |
|---------|----------------------|---------------------|
| Compilação | cmake-js por plataforma | Apenas C shim leve |
| Acesso a structs | Direto (C++ nativo) | Via shim setters (`shim_ctx_params_set_*`) |
| Overhead de chamada | ~50ns (N-API) | ~10ns (bun:ffi direto) |
| Portabilidade | Binário por plataforma | dlopen universal |
| Thread safety | Main thread (N-API thread-safe functions) | Worker thread isolado |
| Debugging | gdb/lldb no addon | Mais difícil (FFI opaco) |

### O Padrão Addon*

Cada classe nativa segue um padrão consistente:

```cpp
// AddonContext.cpp (simplificado)
class AddonContext : public Napi::ObjectWrap<AddonContext> {
    llama_context* ctx;
    llama_model* model;  // referência ao model pai
    bool disposed = false;

    // Construção
    AddonContext(Napi::CallbackInfo& info) {
        auto params = llama_context_default_params();
        // Parse options do JS → params
        ctx = llama_new_context_with_model(model, params);
    }

    // Embedding: configura flags diretamente
    if (options.Has("embeddings"))
        context_params.embeddings = true;
    if (options.Has("ranking"))
        context_params.pooling_type = LLAMA_POOLING_TYPE_RANK;
};
```

Nosso equivalente com FFI exige um shim C para cada campo de struct:

```c
// llama_shims.c
void shim_ctx_params_set_n_ctx(void* buf, uint32_t n) {
    ((struct llama_context_params*)buf)->n_ctx = n;
}
```

Mais verbose, mas elimina a dependência de compilação cmake.

---

## Modelo e Contexto

### Hierarquia de Objetos

```
Llama (backend singleton-like)
  ├── LlamaModel #1 (um por arquivo GGUF)
  │   ├── LlamaContext #1a (inferência)
  │   │   ├── LlamaContextSequence #1a-1
  │   │   └── LlamaContextSequence #1a-2
  │   ├── LlamaContext #1b (inferência)
  │   ├── LlamaEmbeddingContext (embeddings)
  │   └── LlamaRankingContext (reranking)
  └── LlamaModel #2
      └── ...
```

**`Llama`** — instância de backend. Criada via `getLlama()` (factory, **não singleton** — cada chamada cria nova instância). Gerencia:
- Detecção de GPU (Metal/CUDA/Vulkan)
- Orquestradores de memória (VRAM/RAM/Swap)
- Thread pool compartilhado
- Carregamento e cache de binários nativos

**`LlamaModel`** — um modelo carregado. Cada modelo tem:
- Handle nativo (`AddonModel`)
- Tokenizer
- Metadados GGUF parseados
- LoRA adapters carregados
- Contagem de GPU layers

**`LlamaContext`** — contexto de inferência. Cada contexto possui:
- KV cache próprio (tamanho = `contextSize × sequences`)
- Configuração de batch
- Pool de sequence IDs
- Fila de decode (producer-consumer)

**`LlamaContextSequence`** — stream de geração independente dentro de um contexto. Cada sequence tem:
- ID nativo (mapeado para KV cache partition)
- Histórico de tokens (`_contextTokens`)
- Posição no KV cache (`_nextTokenIndex`)
- Lock de avaliação independente
- Token predictor (speculative decoding)

### Como Carrega Modelos

```typescript
// 1. Factory method (VRAM-aware)
const model = await llama.loadModel({
    modelPath: "model.gguf",
    gpuLayers: "auto",  // binary search pelo melhor layer count
});

// Internamente:
static async _create(options, {_llama}) {
    // 1. Parse GGUF metadata (offline, sem carregar modelo)
    const ggufInsights = new GgufInsights(fileInfo, _llama);

    // 2. Resolve GPU layers (auto = binary search por VRAM)
    const gpuLayers = await ggufInsights.configurationResolver
        .resolveModelGpuLayers(options.gpuLayers, {
            getVramState: () => _llama._vramOrchestrator.getMemoryState()
        });

    // 3. Estima requisitos de memória
    const estimation = ggufInsights.estimateModelResourceRequirements({gpuLayers});

    // 4. Reserva VRAM + RAM (previne overcommit)
    const vramReservation = _llama._vramOrchestrator.reserveMemory(estimation.gpuVram);

    // 5. Carrega modelo nativo
    const modelLoaded = await model._model.init();

    // 6. Libera reserva (memória real agora tracked pelo OS)
    vramReservation.dispose();
}
```

**Carregamento serializado**: um mutex global (`_memoryLock`) garante que apenas um modelo/contexto carrega por vez. Isso previne race conditions de VRAM (dois loads vendo "4GB livre" e ambos alocando 3GB).

### Como Gerencia Contextos

Contextos são criados via `model.createContext()`, também serializado pelo mesmo mutex. O KV cache é alocado com `contextSize × sequences` cells — cada sequence recebe uma partição fixa e independente.

**Retry com shrinking**: se a criação falha (OOM), retenta até 16 vezes, reduzindo o `contextSize` em 16% a cada tentativa:

```typescript
// Padrão resiliente para auto-sizing
for (let attempt = 0; attempt < 16; attempt++) {
    try {
        return await createContext(contextSize);
    } catch (e) {
        contextSize = Math.floor(contextSize * 0.84);
    }
}
```

---

## Sequences Paralelas (Mesmo Modelo)

### Conceito

Múltiplas sequences permitem gerar N respostas simultaneamente dentro de um único contexto, compartilhando o mesmo modelo e GPU. Cada sequence é um stream de tokens independente com seu próprio estado.

### Pool de IDs com GC-based Reclamation

```typescript
private _popSequenceId(): number | null {
    // 1. Tenta IDs reciclados primeiro
    if (this._unusedSequenceIds.length > 0)
        return this._unusedSequenceIds.shift()!;

    // 2. Incrementa counter
    if (this._nextGeneratedSequenceId < this._totalSequences) {
        return this._nextGeneratedSequenceId++;
    }

    // 3. Pool esgotado
    return null;
}
```

Reclamação via **`FinalizationRegistry`** (GC cleanup) + `dispose()` explícito:

```typescript
// No construtor da sequence:
this._gcRegistry = new FinalizationRegistry(
    this._context._reclaimUnusedSequenceId
);
this._gcRegistry.register(this, sequenceId);

// Quando reclamado: ctx.disposeSequence(sequenceId) limpa o KV cache
```

**Observação**: `FinalizationRegistry` é um safety net — o ideal é sempre chamar `dispose()` explicitamente. O GC não garante timing.

### KV Cache Particionado

```typescript
// Total KV cache = contextSize × número de sequences
contextSize: padSafeContextSize(this._contextSize * this._totalSequences, "up")
```

Cada sequence recebe `contextSize` cells independentes. **Não há compartilhamento de KV cache entre sequences** — cada uma é uma partição isolada. Isso é simples mas desperdiça memória quando sequences compartilham prefixo (ex: system prompt idêntico).

### Batch Scheduling

O sistema de batch é uma **fila producer-consumer** com estratégias plugáveis:

**Producer** — cada sequence submete tokens para decodificação:

```typescript
this._queuedDecodes.push({
    sequenceId,
    tokens,
    logits,                     // quais posições precisam de logit output
    firstTokenSequenceIndex,    // onde no KV cache escrever
    evaluationPriority,         // 1-10
    response: [accept, reject], // resolução de Promise
    logitDataMapper,            // callback para sampling
});
this._scheduleDecode();  // agenda processamento
```

**Consumer** — `dispatchPendingBatch()` processa a fila:

```
1. Resolve estratégia de priorização
2. Reserva threads
3. Enquanto fila tem itens:
   a. Ordena por estratégia → getOrderedQueuedDecodes()
   b. Encaixa itens no budget do batch → fitQueuedDecodesToABatch()
   c. Inicializa batch nativo → ctx.initBatch(batchSize)
   d. Para cada item: ctx.addToBatch(seqId, firstIndex, tokens, logitIndexes)
   e. ctx.decodeBatch()  ← computação GPU real
   f. Processa resultados de logit (sampling)
   g. Executa afterBatchAction callbacks
```

**Dispatch scheduling**: despacha imediatamente quando todas as sequences têm trabalho pendente. Caso contrário, adia para o próximo ciclo do event loop (`setImmediate`):

```typescript
if (this._queuedDecodeSequenceIds.size === this._totalSequences)
    dispatch();  // todas as sequences prontas → dispatch imediato
else if (dispatchSchedule === "nextCycle")
    setImmediate(dispatch);  // espera uma tick
```

### Estratégias de Priorização

**`maximumParallelism` (padrão)**:
1. Divide budget igualmente: `batchSize / numItems` tokens por item
2. Redistribui sobras em 3 passes round-robin
3. Pass final: dá resto para itens de maior prioridade

Garante que toda sequence processa ao menos alguns tokens — maximiza paralelismo.

**`firstInFirstOut`**:
1. Ordena por `evaluationPriority` (descendente)
2. Aloca tokens até encher o batch

Processa menos sequences mas completa cada uma mais rápido.

### Geração Paralela com Async Generators

Cada sequence expõe um `AsyncGenerator` para geração:

```typescript
async *_evaluate(tokens, metadata, options) {
    while (true) {
        const evaluatorLock = await acquireLock([this._lock, "evaluate"]);

        // Submete para batch queue (pode esperar outras sequences)
        const decodeResult = await this._decodeTokens(
            evalTokens, logitsArray, priority, tokenMeter, contextShift,
            (batchLogitIndex) => {
                sampler.applyConfig(samplerConfig);
                return ctx.sampleToken(batchLogitIndex, sampler._sampler);
            }
        );

        // Yield token gerado
        const replacementToken = yield { token: nextToken };

        // Caller pode substituir o próximo token (guided generation)
        evalTokens = replacementToken ?? [nextToken];
    }
}
```

Diferentes sequences naturalmente terminam em momentos diferentes — o batch system lida com sequences entrando/saindo da fila dinamicamente. Não há `max_tokens` explícito no nível da sequence — o caller controla quando parar de iterar.

---

## Embeddings

### LlamaEmbeddingContext

Wrapper fino sobre `LlamaContext` com flag `_embeddings: true`. Reutiliza o pipeline de avaliação mas pula o sampling de tokens.

```typescript
const embeddingCtx = await model.createEmbeddingContext({
    contextSize: 512,
    batchSize: 128,
    threads: 6,  // default fixo (vs adaptativo para inferência)
});
```

**Diferenças do contexto de inferência**:

| Aspecto | Inferência | Embedding |
|---------|-----------|-----------|
| Flag `_embeddings` | `false` | `true` |
| Sampling | Sim (gera tokens) | Não (`_noSampling: true`) |
| Output | Probabilidades de next token | Vetor float[] |
| Threads default | Adaptativo | 6 (fixo) |
| Sequences | Múltiplas | Uma |

### Pipeline de Embedding

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
sequence.evaluate(tokens, {_noSampling: true})  ← forward pass only
     │
     ▼
_ctx.getEmbedding(tokenCount)  ← extrai do contexto nativo
     │
     ▼
LlamaEmbedding({ vector: Float64Array → number[] })
```

### FFI Calls para Embeddings

O C++ addon usa estas funções do llama.cpp:

```cpp
// Determina dimensão do embedding (ex: 384, 768, 4096)
const int n_embd = llama_model_n_embd(model->model);

// Determina estratégia de pooling
const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

// Estratégia 1: embeddings pooled (mean/cls/rank)
const auto* embeddings = llama_get_embeddings_seq(ctx, 0);  // sequence 0

// Estratégia 2: fallback para embeddings do último token
if (embeddings == NULL)
    embeddings = llama_get_embeddings_ith(ctx, inputTokensLength - 1);

// Copia para Float64Array
Napi::Float64Array result = Napi::Float64Array::New(env, resultSize);
for (size_t i = 0; i < resultSize; i++)
    result[i] = embeddings[i];
```

### Pooling Types

```typescript
enum GgufMetadataArchitecturePoolingType {
    unspecified = -1,
    none = 0,    // sem pooling — usa último token
    mean = 1,    // média de todos os token embeddings
    cls = 2,     // usa embedding do token [CLS]
    last = 3,    // usa embedding do último token
    rank = 4     // modo cross-encoder ranking
}
```

### LlamaEmbedding — Classe de Dados

```typescript
class LlamaEmbedding {
    public readonly vector: readonly number[];

    constructor(options) {
        // Imutável: frozen + copied
        this.vector = Object.freeze(options.vector.slice());
    }

    calculateCosineSimilarity(other): number {
        // dot(a,b) / (||a|| × ||b||)
        // Edge cases: vetores vazios → 0, ambos zero → 1
    }

    toJSON() { return { type: "embedding", vector: [...this.vector] }; }
    static fromJSON(json) { return new LlamaEmbedding(json); }
}
```

**Nota**: não aplica normalização L2 — os embeddings brutos do llama.cpp são retornados como estão. A normalização é implícita na divisão pela magnitude no cosine similarity.

---

## Ranking/Reranking

### LlamaRankingContext

Contexto de **cross-encoder** — diferente de embeddings (bi-encoder: encode query e documentos separadamente), ranking concatena query + documento e avalia juntos. O modelo produz um único logit convertido em probabilidade de relevância via sigmoid.

### Criação

```typescript
const rankingCtx = await model.createRankingContext({
    contextSize: 512,
    batchSize: 128,
});
// Internamente: _embeddings: true + _ranking: true
// _ranking → pooling_type = LLAMA_POOLING_TYPE_RANK
```

**Validações na criação**:
1. Rejeita modelos encoder-decoder (ex: T5)
2. Verifica suporte a ranking via scan de tensors (`cls.weight` ou `cls.output.weight`)
3. Valida tokens especiais (EOS ou SEP obrigatórios)
4. Parse de template de ranking da metadata GGUF (`chat_template.rerank`)

### Detecção de Suporte a Ranking

```typescript
public get supportsRanking() {
    const layers = this._ggufFileInfo.fullTensorInfo ?? [];

    // Scan reverso (otimização: cls layers ficam no final)
    for (let i = layers.length - 1; i >= 0; i--) {
        if (tensor.name === "cls.weight" || tensor.name === "cls.output.weight") {
            return (this.tokens.sepToken != null || this.tokens.eosToken != null)
                && !(this.hasEncoder && this.hasDecoder);
        }
    }
    return false;  // sem layer cls
}
```

**Critérios** (todos devem ser atendidos):
1. ✅ Modelo tem tensor `cls.weight` ou `cls.output.weight`
2. ✅ Modelo tem token SEP ou EOS
3. ✅ Modelo **não é** encoder-decoder

### Template de Input

**Com template** (da metadata `chat_template.rerank`):

```typescript
// Template: "<query>{{query}}</query><document>{{document}}</document>"
// Template parts são tokenizados COM special tokens
// Query/document são tokenizados SEM special tokens
```

**Sem template** (formato default):

```
[BOS] query [EOS] [SEP] document [EOS]
```

### Scoring: Logit → Sigmoid

```typescript
private async _evaluateRankingForInput(input: Token[]): Promise<number> {
    return withLock([this, "evaluate"], async () => {
        // Limpa KV cache
        await this._sequence.eraseContextTokenRanges([...]);

        // Avalia sem sampling
        const iterator = this._sequence.evaluate(input, { _noSampling: true });
        for await (const token of iterator) { break; }

        // Extrai SINGLE valor (maxVectorSize=1)
        const embedding = this._llamaContext._ctx.getEmbedding(input.length, 1);
        const logit = embedding[0]!;

        // Sigmoid: 1 / (1 + e^(-logit))
        return 1 / (1 + Math.exp(-logit));
    });
}
```

**Diferença crítica do embedding**: `getEmbedding(input.length, 1)` — o segundo argumento `1` limita o vetor a um único valor. Esse é o logit de relevância do cross-encoder.

### Batch Ranking

```typescript
public async rankAll(query, documents) {
    return await Promise.all(
        documents.map(doc => this._evaluateRankingForInput(
            this._getEvaluationInput(query, doc)
        ))
    );
}

public async rankAndSort(query, documents) {
    const scores = await this.rankAll(query, documents);
    return documents
        .map((doc, i) => ({ document: doc, score: scores[i]! }))
        .sort((a, b) => b.score - a.score);  // highest first
}
```

**Atenção**: `Promise.all` **parece** paralelo mas executa serialmente por causa do `withLock` — cada chamada enfileira e espera sua vez. É um padrão enganoso.

### Bug Encontrado

No `LlamaRankingContext.ts` linha 203, o end token é prepended ao invés de appended:

```typescript
// BUG: deveria ser .push(endToken)
if (endToken != null && resolvedInput.at(-1) !== endToken)
    resolvedInput.unshift(endToken);  // ← ERRADO
```

---

## Multi-Modelo e VRAM

### MemoryOrchestrator

Três orquestradores de memória rastreiam reservas independentemente:

```typescript
static async _create(params) {
    // 1. VRAM — queries GPU via bindings nativos
    const vramOrchestrator = new MemoryOrchestrator(() => {
        const {total, used, unifiedSize} = bindings.getGpuVramInfo();
        return { total, free: Math.max(0, total - used), unifiedSize };
    });

    // 2. RAM — queries OS
    const ramOrchestrator = new MemoryOrchestrator(() => {
        const used = process.memoryUsage().rss;
        return { total: os.totalmem(), free: total - used };
    });

    // 3. Swap — queries bindings nativos
    const swapOrchestrator = new MemoryOrchestrator(() => {
        const {total, free} = bindings.getSwapInfo();
        return { total, free, unifiedSize: 0 };
    });
}
```

### Reservation-Based Tracking

```typescript
class MemoryOrchestrator {
    private _reservedMemory: number = 0;

    reserveMemory(bytes: number): MemoryReservation {
        this._reservedMemory += bytes;
        return MemoryReservation._create(bytes, () => {
            this._reservedMemory -= bytes;
        });
    }

    async getMemoryState() {
        const {free, total} = this._getMemoryState();
        return {
            free: Math.max(0, free - this._reservedMemory),
            total
        };
    }
}
```

**Como funciona para multi-modelo**:
1. Modelo A começa a carregar → reserva 3GB VRAM
2. Modelo B começa a carregar → vê `free` reduzido → aloca menos GPU layers
3. Modelo A termina → libera reserva → tracking real pelo OS
4. `_getMemoryState()` sempre consulta estado **real** do hardware

### Auto GPU Layers (Binary Search)

O modo `gpuLayers: "auto"` usa binary search com scoring:

```typescript
function getBestGpuLayersForFreeVram({ggufInsights, freeVram, ...}) {
    // Binary search de maxLayers até minLayers
    // Para cada candidato:
    //   1. Calcula VRAM do modelo = estimateModelResourceRequirements({gpuLayers})
    //   2. Se fitContext, calcula VRAM do contexto também
    //   3. Pontua a configuração:
    //      - Score de GPU layers (mais = melhor, todas = bônus)
    //      - Score de context size (maior = melhor, ponderado por % GPU)
    //   4. Retorna configuração com maior score que cabe na VRAM
}
```

**Modos de `gpuLayers`**:

| Modo | Comportamento |
|------|--------------|
| `"auto"` (default) | Binary search para layer count ótimo |
| `"max"` | Todas as layers na GPU; throw se impossível |
| `number` | Contagem exata; validada contra VRAM |
| `{min, max, fitContext}` | Range com fitting de contexto |

### Apple Silicon Unified Memory

No Apple Silicon, GPU e CPU compartilham memória. O uso de VRAM é contabilizado duas vezes. Um helper calcula a porção RAM do uso de VRAM:

```typescript
function getRamUsageFromUnifiedVram(vramUsage, vramState) {
    const onlyVramSize = vramState.total - vramState.unifiedSize;
    return Math.min(
        vramState.unifiedSize,
        Math.max(0, vramUsage - Math.max(0, onlyVramSize - existingUsage))
    );
}
```

### Safety Padding

```typescript
// VRAM: min(8% do total, 1.2GB) — headroom para driver/OS
const vramPadding = Math.min(totalVram * 0.08, 1.2 * 1024**3);

// RAM: min(25%, 6GB) no macOS; min(25%, 1GB) no Linux
const ramPadding = platform === "linux"
    ? Math.min(totalRam * 0.25, 1 * 1024**3)
    : Math.min(totalRam * 0.25, 6 * 1024**3);
```

---

## Resource Management

### Arquitetura de Disposal em 3 Camadas

#### Camada 1: DisposeGuard (Reference Counting)

Previne disposal prematuro via contagem de referências:

```typescript
class DisposeGuard {
    private _preventionHandles: number = 0;

    createPreventDisposalHandle(): DisposalPreventionHandle {
        this._preventionHandles++;
        return DisposalPreventionHandle._create(() => {
            this._preventionHandles--;
            this._activateLocksIfNeeded();
        });
    }

    async acquireDisposeLock(): Promise<void> {
        // Espera até que todos os handles sejam liberados
        if (this._preventionHandles > 0)
            await new Promise(resolve => this._callbacks.push(resolve));
    }
}
```

**Hierarquia de guards**:

```
Llama._backendDisposeGuard
  ↑ parent of
LlamaModel._backendModelDisposeGuard
  ↑ parent of
LlamaContext._backendContextDisposeGuard
```

Quando um model cria um context, ele segura um prevention handle no guard do model, que por sua vez segura um no guard do backend. Isso previne disposal prematuro do backend.

#### Camada 2: AsyncDisposeAggregator (Cleanup Ordenado)

Cada objeto registra steps de cleanup em ordem:

```typescript
// Ordem de disposal do LlamaModel:
this._disposeAggregator.add(() => { this._disposedState.disposed = true; });
this._disposeAggregator.add(this.onDispose.dispatchEvent);
this._disposeAggregator.add(async () => {
    await this._backendModelDisposeGuard.acquireDisposeLock();  // Espera contexts
    await this._model.dispose();                                // Free native memory
    this._llamaPreventDisposalHandle.dispose();                 // Release backend lock
});
```

#### Camada 3: WeakRef + FinalizationRegistry (GC Safety Net)

```typescript
// Models escutam disposal do parent Llama via WeakRef
function disposeModelIfReferenced(modelRef: WeakRef<LlamaModel>) {
    const model = modelRef.deref();
    if (model != null) void model.dispose();
}

// Sequences usam FinalizationRegistry para cleanup por GC
this._gcRegistry = new FinalizationRegistry(
    this._context._reclaimUnusedSequenceId
);
```

### Cascade Disposal

Quando `Llama.dispose()` é chamado:
1. Dispara evento `onDispose`
2. Todos os models recebem o evento → chamam `dispose()` próprio
3. Cada model espera seus contexts terminarem (via DisposeGuard)
4. Cada context dispõe suas sequences
5. Memória nativa é liberada bottom-up
6. Prevention handles são liberados upward

Quando um model individual é disposto:

```typescript
await model1.dispose();
// → contexts do model1 são dispostos
// → handle nativo do model1 liberado
// → prevention handle no backend Llama liberado
// → model2 completamente não-afetado
// → backend Llama continua vivo (model2 ainda tem prevention handle)
```

### Segurança na Saída do Processo

```typescript
// Registra disposal no beforeExit
process.once("beforeExit", () => {
    for (const ref of disposableRefs) {
        const target = ref.deref();
        if (target) target[Symbol.asyncDispose]();
    }
});
```

---

## Speculative Decoding

### Conceito

Speculative decoding avalia `1 + N` tokens em um único batch (1 real + N preditos). Se a predição na posição `i` estava correta, o resultado em cache é usado diretamente sem outra avaliação.

### Token Predictors

**Classe base abstrata**:

```typescript
abstract class TokenPredictor {
    abstract reset(params): Promise<void> | void;
    abstract pushTokens(tokens: Token[]): void;
    abstract predictTokens(): Promise<Token[]> | Token[];
    stop(): void {}
    dispose(): void {}
}
```

### DraftSequenceTokenPredictor

Usa um modelo menor (draft) para predizer tokens especulativamente:

- Mantém uma `LlamaContextSequence` de um modelo separado (menor)
- A sequence draft espelha o estado da sequence alvo
- Gera tokens usando o draft model
- Threshold de confiança (`minConfidence: 0.6`) — para de predizer quando confiança cai

```typescript
// Prediction loop
while (canIterate) {
    const {token, confidence} = await draftIterator.next();
    if (confidence < this._minConfidence) {
        this._waitForPredictionExhaustion = true;
        break;
    }
    this._predictedTokens.push(token);
}
```

### InputLookupTokenPredictor

Preditor zero-cost baseado em prompt-lookup decoding:

1. Procura os últimos tokens gerados como padrão nos tokens de input (prompt)
2. Se encontrar, prediz os tokens que seguem esse padrão no input
3. Útil para tarefas grounded no input (sumarização, modificação de código)

```typescript
predictTokens() {
    // Encontra match mais longo de tokens recentes no input
    const [matchStartIndex, matchLength] = this._findLongestPatternIndex(
        this._inputTokens,
        this._stateTokens
    );
    // Retorna tokens que seguem o match
    return this._inputTokens.slice(
        matchStartIndex + matchLength,
        matchStartIndex + matchLength + this._predictionMaxLength
    );
}
```

### Loop de Validação

```typescript
async *_speculativeEvaluate(tokens, metadata, options) {
    while (true) {
        // FAST PATH: usa predições pré-validadas
        if (loadedTokenPredictions.length > 0) {
            nextToken = predictions.shift().output;
            yield { token: nextToken };
            continue;
        }

        // Obtém predições
        for (const token of await tokenPredictor.predictTokens()) {
            evalTokens.push(token);
            logitsArray[evalTokens.length - 1] = true;  // logits para TODAS as posições
        }

        // Avalia TODOS os tokens em um batch
        const decodeResult = await this._decodeTokens(evalTokens, logitsArray);

        // Valida predições
        for (let i = 1; i < evalTokens.length; i++) {
            const resultToken = decodeResult[i];
            if (lastOutput === evalTokens[i]) {
                // ✅ Correta — cacheia resultado
                this._loadedTokenPredictions.push([evalTokens[i], resultToken]);
                this._validatedTokenPredictions++;
            } else {
                // ❌ Errada — apaga do KV cache
                this._refutedTokenPredictions++;
                await this._eraseContextTokenRanges([...]);
                break;
            }
        }
    }
}
```

**Estatísticas rastreadas por sequence**:

```typescript
tokenPredictions: {
    used: number,       // validadas E consumidas
    unused: number,     // validadas mas não consumidas
    validated: number,  // predições corretas
    refuted: number     // predições erradas
}
```

---

## Pontos Fortes ✅

1. **VRAM awareness completo** — estimação baseada em GGUF antes de carregar, reservation-based tracking, auto GPU layers com binary search. Previne OOM de forma robusta.

2. **Hierarquia de objetos clara** — `Llama → Model → Context → Sequence → Session` é uma progressão natural que escala de simples a complexo.

3. **Batch scheduling sofisticado** — fila producer-consumer com estratégias plugáveis (`maximumParallelism` vs `firstInFirstOut`), partial processing, prioridade por sequence.

4. **Disposal robusto** — 3 camadas (DisposeGuard + AsyncDisposeAggregator + WeakRef/FinalizationRegistry) garantem cleanup correto mesmo com GC imprevisível.

5. **Speculative decoding completo** — suporta tanto draft model quanto input lookup, com validação por batch e cache de predições.

6. **Context shift automático** — quando a sequence enche, apaga tokens do início preservando BOS. Suporta estratégia custom.

7. **Detecção inteligente de modelos** — parse de metadados GGUF para detectar embedding support, ranking support, encoder/decoder, pooling type — tudo offline.

8. **Retry com shrinking** — criação de contexto retenta até 16x reduzindo contextSize em 16%. Resiliente em ambientes com VRAM limitada.

9. **API unificada para embeddings e ranking** — reutiliza o pipeline de avaliação existente com flags diferentes, sem duplicação de código.

10. **Cross-platform** — Metal, CUDA, Vulkan com detecção automática e fallback para CPU. Build from source como último recurso.

---

## Pontos Fracos ❌

1. **Sem batch embedding** — cada texto é avaliado separadamente com clear completo do KV cache. Para 1000 textos, são 1000 forward passes sequenciais. Um verdadeiro batch (múltiplos textos em um batch) seria dramaticamente mais rápido.

2. **KV cache não compartilhado entre sequences** — alocação `contextSize × N` desperdiça memória quando sequences compartilham prefixo (ex: system prompt). Não implementa prefix sharing via `llama_kv_cache_seq_cp()`.

3. **Over-engineering do disposal** — 3 camadas (DisposeGuard + AsyncDisposeAggregator + WeakRef) tornam o fluxo de cleanup difícil de raciocinar. Muitos edge cases e callbacks nested.

4. **`Promise.all` enganoso no ranking** — `rankAll()` usa `Promise.all` que *parece* paralelo mas executa serialmente por causa do `withLock`. API misleading.

5. **Sem normalização L2** — embeddings retornados brutos. Muitos modelos de embedding esperam normalização L2 para cosine similarity funcionar corretamente.

6. **Float64 desnecessário** — embeddings armazenados como `number[]` (64-bit) quando o modelo produz float32. Desperdício de 2× memória.

7. **Sem eviction ou prioridade de modelos** — se VRAM enche, o usuário deve manualmente fazer dispose. Sem LRU, sem prioridade de "manter na GPU".

8. **Carregamento serial obrigatório** — o mutex global impede carregar dois modelos simultaneamente, mesmo em GPUs diferentes ou com VRAM sobrando.

9. **N-API requer compilação** — cmake-js + binários pré-compilados por plataforma. Frágil em ambientes novos ou edge.

10. **Sem métricas de batch** — nenhum logging de utilização de batch, tempos de espera, ou decisões de scheduling. Caixa preta.

11. **InputLookupTokenPredictor conservador demais** — default `maxTokens: 3` é muito baixo. Para tarefas de código, predizer 8-16 tokens do prompt lookup seria significativamente mais benéfico.

12. **Bug no ranking template** — `unshift` ao invés de `push` para end token na construção do input de ranking.

---

## Lições para bun-llama-cpp

### O que Adotar ✅

1. **MemoryOrchestrator pattern** — essencial para qualquer sistema multi-modelo. O padrão reservation-based é elegante, portável e previne overcommit. Adaptar para funcionar entre Workers.

2. **Pool de sequence IDs** — simples e efetivo. Usar recycling com `FinalizationRegistry` como safety net, mas priorizar `dispose()` explícito.

3. **Batch queue producer-consumer** — o pattern com estratégias plugáveis é essencial para multi-sequence. Implementar dentro do Worker.

4. **Auto GPU layers com GGUF estimation** — vale a pena para `gpuLayers: "auto"`. Binary search com scoring é o gold standard.

5. **Context shift com preservação de BOS** — detalhe importante que eles acertam. Sempre preservar o token BOS ao fazer shift.

6. **Retry com shrinking na criação de contexto** — pattern resiliente e barato de implementar.

7. **Detecção de capabilities via GGUF metadata** — parse offline de tensors e metadados para saber se modelo suporta embedding, ranking, etc.

### O que Fazer Diferente 🔄

1. **Batch embedding real** — implementar avaliação de múltiplos textos em um único batch via sequences distintas. Dramaticamente mais rápido que sequencial.

2. **KV cache com prefix sharing** — usar `llama_kv_cache_seq_cp()` para copiar prefixo compartilhado (system prompt) entre sequences. Economia significativa de compute.

3. **Float32Array para embeddings** — usar `Float32Array` direto do FFI ao invés de converter para `number[]`. 2× menos memória, zero overhead.

4. **Disposal simples** — nosso modelo Worker + `terminate()` já é mais simples e seguro. Para multi-modelo, coordenar via main thread sem 3 camadas de indireção. `using` + `Symbol.asyncDispose` é suficiente.

5. **Worker thread ao invés de main thread** — nossa arquitetura de Worker isolado é superior para thread safety. O FFI roda no Worker, nunca bloqueia o main thread. node-llama-cpp roda tudo no main thread via N-API thread-safe functions, o que é mais complexo.

6. **Normalização L2 opcional** — oferecer `normalize: true` como opção em embeddings. Muitos modelos precisam disso.

7. **Métricas de batch** — adicionar observabilidade: tokens/segundo, utilização de batch, tempos de espera. Custo quase zero, valor alto para debugging.

8. **InputLookupPredictor mais agressivo** — default de 8-16 tokens para tarefas de código. O custo de erro é baixo (apenas erase do KV cache), mas o benefício de acerto é alto.

### Decisões Arquiteturais Chave

| Decisão | node-llama-cpp | Nossa recomendação |
|---------|---------------|-------------------|
| Binding | N-API addon | bun:ffi + C shim (manter) |
| Thread model | Main thread | Worker isolado (manter) |
| Sequence management | Event loop JS | Worker message loop (adaptar) |
| VRAM tracking | MemoryOrchestrator | Equivalente no main thread (adotar) |
| Disposal | 3-layer DisposeGuard | Worker.terminate + dispose msg (simplificar) |
| Embedding format | Float64Array | Float32Array (melhorar) |
| Batch embedding | Sequencial | Batch real (melhorar) |
| Prefix sharing | Não implementado | llama_kv_cache_seq_cp (implementar) |

---

*Análise gerada a partir do source code em `referencias/node-llama-cpp/` e dos documentos de análise do projeto.*
