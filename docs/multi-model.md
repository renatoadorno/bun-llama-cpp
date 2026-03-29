# Multi-Modelo: Análise e Estratégia

> Análise de como orquestrar múltiplos modelos LLM em paralelo no bun-llama-cpp,
> com base nas implementações de referência (node-llama-cpp e qmd) e proposta de
> arquitetura própria otimizada para Bun Workers e Apple Silicon.

---

## O Problema

Aplicações modernas de IA raramente dependem de um único modelo. Um pipeline RAG típico exige **três modelos distintos** operando em conjunto:

| Modelo | Função | Tamanho típico |
|--------|--------|----------------|
| Embedding | Converter texto em vetores para busca semântica | 300M–1B parâmetros |
| Reranking | Cross-encoder que pontua relevância query↔documento | 0.5B–1B parâmetros |
| Geração | Produzir texto (respostas, expansão de query) | 1B–70B parâmetros |

### Cenários que exigem multi-modelo

1. **RAG Pipeline**: embed query → buscar vetores → rerankar resultados → gerar resposta
2. **Speculative Decoding**: modelo draft pequeno gera candidatos, modelo principal valida em batch
3. **Model Routing**: detectar tipo de tarefa e rotear para modelo especializado (coding, chat, análise)
4. **A/B Testing**: dois modelos de geração carregados simultaneamente para comparação

### Desafios fundamentais

- **Limites de VRAM**: Apple Silicon unifica CPU/GPU memory — um modelo de 8GB + outro de 4GB = 12GB do pool unificado de 16–32GB
- **Coordenação de memória**: sem tracking centralizado, dois modelos podem tentar alocar mais VRAM do que existe
- **Lifecycle management**: modelos idle consumindo memória que poderia ser usada por modelos ativos
- **Ordem de disposal**: liberar um modelo enquanto outro o referencia causa crashes de Metal/GPU

---

## Como node-llama-cpp Implementa

O node-llama-cpp usa uma arquitetura sofisticada de N-API addon rodando na main thread, com coordenação centralizada via um objeto `Llama` compartilhado.

### Hierarquia de objetos

```
Llama (backend compartilhado, reusável)
  ├── LlamaModel #1 (lifecycle independente)
  │   ├── LlamaContext #1a
  │   │   ├── LlamaContextSequence #1a-1
  │   │   └── LlamaContextSequence #1a-2
  │   └── LlamaContext #1b
  ├── LlamaModel #2 (lifecycle independente)
  │   └── LlamaContext #2a
  └── LlamaModel #3 ...
```

### Trio de MemoryOrchestrators

Três orchestrators rastreiam reservas em paralelo:

```typescript
// VRAM — consulta bindings nativos
const vramOrchestrator = new MemoryOrchestrator(() => {
    const {total, used, unifiedSize} = bindings.getGpuVramInfo();
    return { total, free: Math.max(0, total - used), unifiedSize };
});

// RAM — consulta OS
const ramOrchestrator = new MemoryOrchestrator(() => ({
    total: os.totalmem(),
    free: Math.max(0, os.totalmem() - process.memoryUsage().rss),
}));

// Swap — consulta bindings nativos
const swapOrchestrator = new MemoryOrchestrator(() => {
    const {total, free} = bindings.getSwapInfo();
    return { total, free };
});
```

### Reservation-based VRAM tracking

O padrão central: **reservar antes de alocar**.

1. Modelo A começa a carregar → reserva 3GB de VRAM → orchestrator subtrai 3GB do `free`
2. Modelo B começa a carregar → vê VRAM reduzida → aloca menos GPU layers
3. Modelo A termina de carregar → libera reserva → tracking real do OS assume

Um **global lock** (`_memoryLock`) serializa todas as operações de load — dois loads simultâneos nunca consultam VRAM ao mesmo tempo, eliminando race conditions TOCTOU.

### Auto GPU layers via binary search

```
Para cada candidato de layers (max → min):
  1. Estimar VRAM do modelo = ggufInsights.estimateModelResourceRequirements({gpuLayers})
  2. Se fitContext, calcular VRAM do contexto também
  3. Pontuar configuração:
     - Mais layers GPU = pontuação maior
     - Todas as layers GPU = bônus
     - Context size maior = pontuação maior
  4. Retornar configuração de maior pontuação que cabe na VRAM livre
```

### Apple Silicon unified memory

No Apple Silicon, GPU e CPU compartilham o mesmo pool de memória. O node-llama-cpp trata isso via `unifiedSize`:

```typescript
function getRamUsageFromUnifiedVram(vramUsage, vramState) {
    const onlyVramSize = vramState.total - vramState.unifiedSize;
    return Math.min(
        vramState.unifiedSize,
        Math.max(0, vramUsage - Math.max(0, onlyVramSize - existingUsage))
    );
}
```

Isso evita double-counting: alocar 4GB de "VRAM" em Apple Silicon realmente consome 4GB do pool unificado, não 4GB de VRAM + 4GB de RAM.

### Cascading disposal

Três camadas de segurança:

1. **DisposeGuard** — reference counting com parent chain. Modelo segura handle do backend; contexto segura handle do modelo. Ninguém é destruído antes de seus dependentes.
2. **AsyncDisposeAggregator** — cleanup ordenado: marca disposed → dispara evento → espera dependentes → libera memória nativa → libera handle do pai.
3. **WeakRef + FinalizationRegistry** — rede de segurança do GC. Se um sequence não for explicitamente disposed, o GC eventualmente reclama o ID.

### Pontos Fortes

| Aspecto | Detalhe |
|---------|---------|
| VRAM awareness | Estimativa completa via GGUF antes de carregar; previne OOM |
| Auto GPU layers | Algoritmo de scoring com binary search encontra configuração ótima |
| Sistema de reserva | MemoryOrchestrator previne overcommit em loads concorrentes |
| Disposal hierárquico | DisposeGuard com parent chain garante ordem correta de cleanup |
| Unified memory | Tratamento especial para Apple Silicon evita double-counting |
| Loading serializado | Global lock previne race conditions TOCTOU |
| Suporte a AbortSignal | Loading de modelo pode ser cancelado mid-stream |

### Pontos Fracos

| Aspecto | Detalhe |
|---------|---------|
| Sem pool de modelos | Cada load lê do disco; sem cache de modelos carregados |
| Sem eviction | Se VRAM está cheia, usuário deve manualmente dar dispose |
| Sem prioridade | Todos os modelos tratados igualmente; sem "keep in VRAM" |
| Loading serial | Global lock significa que modelos carregam um por vez |
| Tolerância de estimativa | Testes mostram ±200–330MB de erro; pode falhar com VRAM apertada |
| Sem migração dinâmica | Não pode mover layers entre GPU/CPU após loading |
| Disposal complexo | 3 camadas (DisposeGuard + AsyncDisposeAggregator + WeakRef) é difícil de raciocinar |
| Dependência N-API | Requer compilação nativa ou binários prebuilt por plataforma |

---

## Como qmd Orquestra 3 Modelos

O qmd é uma aplicação concreta que usa node-llama-cpp para operar 3 modelos simultaneamente em um pipeline RAG completo.

### Três modelos, três URIs

```typescript
const DEFAULT_EMBED_MODEL    = "embeddinggemma-300M-Q8_0.gguf";   // 300M params
const DEFAULT_RERANK_MODEL   = "qwen3-reranker-0.6b-q8_0.gguf";  // 600M params
const DEFAULT_GENERATE_MODEL = "qmd-query-expansion-1.7B-q4_k_m.gguf"; // 1.7B params
```

Cada modelo tem propósito e lifecycle distintos.

### Lazy loading com promise dedup

O padrão central evita double-allocation de VRAM:

```typescript
private async ensureEmbedModel(): Promise<LlamaModel> {
  if (this.embedModel) return this.embedModel;             // Já carregado
  if (this.embedModelLoadPromise) return await this.embedModelLoadPromise; // Em loading

  this.embedModelLoadPromise = (async () => {
    const model = await llama.loadModel({ modelPath });
    this.embedModel = model;
    return model;
  })();

  try { return await this.embedModelLoadPromise; }
  finally { this.embedModelLoadPromise = null; }  // Limpa promise, mantém modelo
}
```

Se 10 requests chegam enquanto o modelo está carregando, todas aguardam a mesma promise. Sem dedup, 10 loads concorrentes tentariam alocar VRAM 10x.

### Inactivity lifecycle

```
Model load → touchActivity() → timer inicia (5 min default)
                                 ↓ timeout
                                 canUnloadLLM()? → sim → unloadIdleResources()
                                                → não → reagenda timer
```

Estratégia em duas camadas:
- **Default**: dispose apenas contexts (mantém modelos carregados — evita VRAM thrashing)
- **Opt-in** (`disposeModelsOnInactivity: true`): dispose modelos também

### Session ref-counting

```typescript
export async function withLLMSession<T>(
  fn: (session: ILLMSession) => Promise<T>
): Promise<T> {
  const session = new LLMSession(manager);
  try { return await fn(session); }     // _activeSessionCount++
  finally { session.release(); }         // _activeSessionCount--
}
```

O ref-counting previne o cenário onde o inactivity timer dispara no meio de um batch de embeddings. Enquanto `_activeSessionCount > 0`, nenhum modelo é descarregado.

### Pontos Fortes

| Aspecto | Detalhe |
|---------|---------|
| Lazy loading | Modelos só são carregados quando necessários |
| Promise dedup | Previne double-allocation em loads concorrentes |
| Inactivity timer | Libera VRAM de modelos idle automaticamente |
| Ref-counting | Previne disposal prematuro durante operações ativas |
| Context pool | Múltiplos embedding contexts em paralelo (até 8) |
| Content-addressable cache | Chunks idênticos compartilham scores de reranking |

### Pontos Fracos

| Aspecto | Detalhe |
|---------|---------|
| Sem prioridade entre modelos | Um indexing longo bloqueia cleanup para search |
| Global singleton | `getDefaultLlamaCpp()` impede múltiplas instâncias independentes |
| Sem preloading | Primeiro request sempre paga latência de load |
| Sem hot-swap | Trocar modelo exige dispose + reload completo |
| Hardcoded timer | 5 min de inactivity não é configurável por modelo |

---

## Estado Atual do bun-llama-cpp

### Arquitetura atual

```
Main Thread                    Worker Thread
┌─────────────┐               ┌──────────────────┐
│ LlamaModel  │  postMessage  │ llm.worker.ts    │
│ ┌─────────┐ │ ────────────► │ ┌──────────────┐ │
│ │SerialQ  │ │               │ │ FFI calls    │ │
│ │(Promise)│ │ SharedArray   │ │ (synchronous │ │
│ │         │ │ Buffer abort  │ │  blocking)   │ │
│ └─────────┘ │ ◄───────────► │ └──────────────┘ │
└─────────────┘               └──────────────────┘
```

### Limitações para multi-modelo

| Aspecto | Estado |
|---------|--------|
| Modelos por instância | 1 modelo, 1 worker, 1 contexto |
| Coordenação entre modelos | Inexistente — cada `LlamaModel.load()` é independente |
| VRAM tracking | Nenhum — confia nos defaults do llama.cpp |
| Auto GPU layers | Não suportado — usa valor fixo ou default |
| Lifecycle management | Manual — `dispose()` ou nada |
| Disposal coordination | `worker.terminate()` simples, sem hierarchy |

### O que já funciona (em teoria)

Múltiplos `LlamaModel.load()` **podem** funcionar concorrentemente porque cada um cria seu próprio Worker com seu próprio `dlopen`. Porém:

- **Sem coordenação de VRAM**: dois loads simultâneos podem ambos ver 8GB livres e alocar 6GB cada → crash
- **`llama_backend_init` por worker**: cada worker inicializa o backend independentemente — funciona porque são processos separados
- **OS mmap sharing**: múltiplos workers carregando o mesmo arquivo GGUF **compartilham páginas físicas** via mmap do OS. Isso é automático e transparente, mas não é tracking — é otimismo

---

## Estratégia Proposta

### Nível 1: Multi-Worker (Fundação)

**Objetivo**: Cada modelo em seu próprio Bun Worker, com registry centralizado na main thread.

```typescript
// API proposta
const registry = new ModelRegistry();

await registry.load('embed', './models/nomic-embed.gguf', {
  preset: 'small', gpuLayers: 33
});
await registry.load('generate', './models/llama-3.gguf', {
  preset: 'large', gpuLayers: 'auto'
});

const embedModel = registry.get('embed');
const genModel = registry.get('generate');

// Uso independente
const embedResult = await embedModel.embed('texto para embedding');
const genResult = await genModel.infer('prompt', { onToken: console.log });

// Dispose seletivo
await registry.unload('embed');     // Libera apenas o modelo de embedding
await registry.disposeAll();        // Shutdown completo
```

**Implementação**:

```typescript
class ModelRegistry {
  private models = new Map<string, LlamaModel>();
  private loadOrder: string[] = [];

  async load(name: string, path: string, config?: ModelConfig): Promise<void> {
    if (this.models.has(name)) throw new Error(`Model '${name}' already loaded`);
    const model = await LlamaModel.load(path, config);
    this.models.set(name, model);
    this.loadOrder.push(name);
  }

  get(name: string): LlamaModel {
    const model = this.models.get(name);
    if (!model) throw new Error(`Model '${name}' not found`);
    return model;
  }

  async unload(name: string): Promise<void> {
    const model = this.models.get(name);
    if (model) {
      await model.dispose();
      this.models.delete(name);
    }
  }

  async disposeAll(): Promise<void> {
    // Dispose em ordem reversa de loading
    for (const name of [...this.loadOrder].reverse()) {
      await this.unload(name);
    }
  }
}
```

**Complexidade**: Baixa. Usa a infraestrutura existente de `LlamaModel` como está. O registry é um wrapper fino de coordenação.

### Nível 2: Resource Management

**Objetivo**: Tracking de VRAM para prevenir overcommit e calcular GPU layers automaticamente.

#### VRAM Tracker para Apple Silicon

```typescript
class VramTracker {
  private reservations = new Map<string, number>();

  // Consulta VRAM real via sysctl ou Metal API
  async getState(): Promise<VramState> {
    // Apple Silicon: memória unificada
    // total = RAM do sistema (compartilhada entre CPU e GPU)
    // used = alocações Metal ativas
    const total = os.totalmem();
    const reserved = [...this.reservations.values()]
      .reduce((sum, r) => sum + r, 0);
    return { total, reserved, available: total - reserved };
  }

  reserve(modelName: string, bytes: number): void {
    this.reservations.set(modelName, bytes);
  }

  release(modelName: string): void {
    this.reservations.delete(modelName);
  }
}
```

#### Auto GPU layer calculation

```typescript
async function calculateGpuLayers(
  modelPath: string,
  vramTracker: VramTracker
): Promise<number> {
  const state = await vramTracker.getState();
  const available = state.available;

  // Heurística simplificada (sem parsing GGUF completo):
  // - Ler tamanho do arquivo como proxy de memória necessária
  // - Estimar bytes/layer baseado em tamanho total / n_layers
  // - Binary search pelo máximo de layers que cabem
  const fileSize = (await Bun.file(modelPath).stat()).size;
  const estimatedFullGpu = fileSize * 1.1; // 10% overhead

  if (estimatedFullGpu <= available * 0.8) {
    return 999; // Todas as layers na GPU (llama.cpp clipa ao máximo)
  }

  // Degradação proporcional
  return Math.floor(999 * (available * 0.8 / estimatedFullGpu));
}
```

#### Graceful degradation

Quando VRAM está cheia:
1. Reduzir GPU layers do próximo modelo a carregar
2. Se não cabe mesmo com 0 GPU layers, reportar erro com sugestão de unload
3. Sistema de prioridade: embedding models são baratos e ficam carregados; generation models podem ser swapped

```typescript
interface ModelPriority {
  name: string;
  priority: 'persistent' | 'normal' | 'evictable';
}

// Quando VRAM insuficiente para novo load:
// 1. Evictar modelos 'evictable' primeiro
// 2. Depois 'normal'
// 3. Nunca evictar 'persistent' automaticamente
```

### Nível 3: Smart Lifecycle

**Objetivo**: Gestão inteligente de ciclo de vida inspirada no qmd, mas com melhorias.

#### Lazy loading com promise dedup

```typescript
class LazyModel {
  private model: LlamaModel | null = null;
  private loadPromise: Promise<LlamaModel> | null = null;
  private lastUsed = 0;

  constructor(
    private path: string,
    private config: ModelConfig,
    private registry: ModelRegistry
  ) {}

  async ensure(): Promise<LlamaModel> {
    if (this.model) {
      this.lastUsed = Date.now();
      return this.model;
    }
    if (this.loadPromise) return this.loadPromise;

    this.loadPromise = (async () => {
      const model = await LlamaModel.load(this.path, this.config);
      this.model = model;
      this.lastUsed = Date.now();
      return model;
    })();

    try { return await this.loadPromise; }
    finally { this.loadPromise = null; }
  }

  get idleTime(): number {
    return this.model ? Date.now() - this.lastUsed : 0;
  }
}
```

#### Inactivity timer + session ref-counting

```typescript
class LifecycleManager {
  private timers = new Map<string, Timer>();
  private sessionCounts = new Map<string, number>();

  startSession(modelName: string): () => void {
    const count = (this.sessionCounts.get(modelName) ?? 0) + 1;
    this.sessionCounts.set(modelName, count);
    this.clearTimer(modelName);

    // Retorna função de release
    return () => {
      const newCount = (this.sessionCounts.get(modelName) ?? 1) - 1;
      this.sessionCounts.set(modelName, newCount);
      if (newCount <= 0) this.scheduleEviction(modelName);
    };
  }

  private scheduleEviction(modelName: string, delayMs = 5 * 60_000) {
    this.timers.set(modelName, setTimeout(() => {
      if ((this.sessionCounts.get(modelName) ?? 0) <= 0) {
        this.registry.unload(modelName);
      }
    }, delayMs));
  }
}
```

#### Hot-swap: trocar modelo sem recriar worker

Conceito: carregar o novo modelo antes de descartar o antigo, garantindo zero-downtime.

```typescript
async function hotSwap(
  registry: ModelRegistry,
  name: string,
  newPath: string,
  newConfig: ModelConfig
): Promise<void> {
  // 1. Carregar novo modelo com nome temporário
  const tempName = `${name}_swap_${Date.now()}`;
  await registry.load(tempName, newPath, newConfig);

  // 2. Atomic swap na referência
  const oldModel = registry.get(name);
  registry.rename(tempName, name);

  // 3. Aguardar requests in-flight do modelo antigo
  await oldModel.drain(); // Espera SerialQueue esvaziar

  // 4. Dispose do antigo
  await oldModel.dispose();
}
```

#### Model preloading hints

```typescript
// Antecipar quais modelos serão necessários
registry.preloadHint('rerank');  // Inicia loading em background
// ... algum tempo depois ...
const reranker = await registry.get('rerank'); // Já está carregado (ou quase)
```

### Nível 4: Inovações (Além das Referências)

#### Worker Pool com Affinity

Para sistemas com múltiplas GPUs (futuro suporte CUDA/Vulkan), workers podem ter afinidade com GPUs específicas:

```typescript
class WorkerPool {
  private pools = new Map<number, Worker[]>(); // GPU ID → Workers

  async assignModel(model: string, gpuId: number): Promise<Worker> {
    const pool = this.pools.get(gpuId) ?? [];
    const worker = pool.find(w => !w.isBusy) ?? await this.spawnWorker(gpuId);
    return worker;
  }
}
```

No Apple Silicon com unified memory isso é menos relevante, mas para servidores Linux com múltiplas GPUs NVIDIA seria essencial.

#### Model Cascading: Pipeline Automático embed→rerank→generate

**Esta é a inovação principal.** Uma única chamada orquestra os 3 modelos:

```typescript
class ModelPipeline {
  constructor(
    private embedModel: LazyModel,
    private rerankModel: LazyModel,
    private generateModel: LazyModel,
    private vectorStore: VectorStore
  ) {}

  async query(text: string, options?: PipelineOptions): Promise<PipelineResult> {
    // Fase 1: Embedding da query
    const embed = await this.embedModel.ensure();
    const queryVector = await embed.embed(text);

    // Fase 2: Busca vetorial
    const candidates = await this.vectorStore.search(queryVector, options?.topK ?? 20);

    // Fase 3: Reranking (opcional, ativado por default)
    let ranked = candidates;
    if (options?.rerank !== false && candidates.length > 1) {
      const reranker = await this.rerankModel.ensure();
      ranked = await reranker.rerank(text, candidates);
    }

    // Fase 4: Geração com contexto
    const context = ranked.slice(0, options?.contextDocs ?? 5)
      .map(d => d.text).join('\n\n');
    const gen = await this.generateModel.ensure();
    const response = await gen.infer(
      `Context:\n${context}\n\nQuestion: ${text}\n\nAnswer:`,
      { onToken: options?.onToken ?? (() => {}), maxTokens: options?.maxTokens ?? 512 }
    );

    return {
      answer: response.text,
      sources: ranked.slice(0, options?.contextDocs ?? 5),
      embeddings: queryVector,
      tokenCount: response.tokenCount,
    };
  }
}

// API de uso
const pipeline = new ModelPipeline(embedModel, rerankModel, genModel, store);
const result = await pipeline.query('Como funciona o garbage collector do V8?');
```

#### Shared Vocabulary Optimization

Modelos da mesma família (e.g., Llama 3 8B e Llama 3 70B) compartilham o mesmo tokenizer. Em vez de carregar vocabulário duplicado:

```typescript
class TokenizerCache {
  private cache = new Map<string, VocabHandle>();

  getOrCreate(modelFamily: string, vocabPtr: number): VocabHandle {
    if (this.cache.has(modelFamily)) return this.cache.get(modelFamily)!;
    const handle = new VocabHandle(vocabPtr);
    this.cache.set(modelFamily, handle);
    return handle;
  }
}
```

Nota: cada Worker tem seu próprio `dlopen`, então este cache seria útil apenas dentro do mesmo Worker. Para cross-worker, seria necessário `SharedArrayBuffer` com o vocabulário serializado.

#### Dynamic Model Routing

Roteamento automático baseado no tipo de tarefa:

```typescript
class ModelRouter {
  async route(input: string, taskHint?: TaskType): Promise<LlamaModel> {
    const task = taskHint ?? this.detectTask(input);

    switch (task) {
      case 'embedding': return this.registry.get('embed');
      case 'reranking': return this.registry.get('rerank');
      case 'code':      return this.registry.get('code-gen');
      case 'chat':      return this.registry.get('chat-gen');
      default:          return this.registry.get('default-gen');
    }
  }

  private detectTask(input: string): TaskType {
    // Heurísticas simples (sem LLM, sem overhead):
    if (input.length < 100 && !input.includes('\n')) return 'embedding';
    if (input.includes('```') || input.includes('function ')) return 'code';
    return 'chat';
  }
}
```

#### Zero-Downtime Model Swap

Troca de modelo sem interromper requests in-flight:

```
1. Load novo modelo (em Worker separado)
2. Quando pronto, redirecionar novas requests para novo modelo
3. Aguardar requests do modelo antigo finalizarem (drain)
4. Dispose do modelo antigo
5. Liberar Worker antigo
```

```
Timeline:
  Model A: ████████████ [serving] ▓▓▓ [draining] ✗ [disposed]
  Model B:        ░░░░░ [loading] ████████████████ [serving] ...
  Requests:  →A →A →A →A →A  →B →B →B →B →B →B →B
                         ↑ swap point
```

Nenhuma request é dropped. O swap point é atômico (troca de referência no registry).

#### Memory-Mapped Model Sharing

Múltiplos `LlamaModel` apontando para o mesmo arquivo GGUF **já compartilham memória física** via `mmap` do OS. Isso é transparente e automático:

```
Worker 1: dlopen → llama_model_load("model.gguf") → mmap(fd) → pages A,B,C,D
Worker 2: dlopen → llama_model_load("model.gguf") → mmap(fd) → pages A,B,C,D (mesmas!)
                                                                 ↑ OS reutiliza
```

**O que verificar**: que o llama.cpp realmente usa mmap (e não malloc+read). Pela análise do gap-analysis: "llama.cpp internally memory-maps model weights, so multiple workers loading the same model file will share physical RAM via OS mmap." ✅

**O que não é compartilhado**: KV cache e compute buffers — esses são per-context e sempre alocados separadamente.

---

## Impacto na Arquitetura

### Novos componentes

| Componente | Responsabilidade |
|------------|------------------|
| `ModelRegistry` | Registro centralizado de modelos por nome, load/unload coordenado |
| `VramTracker` | Tracking de reservas de VRAM, consulta estado real via sysctl/Metal |
| `LifecycleManager` | Inactivity timers, session ref-counting, eviction automática |
| `ModelPipeline` | Orquestração de pipeline embed→rerank→generate |
| `LazyModel` | Wrapper com lazy loading, promise dedup, idle tracking |
| `ModelRouter` | Roteamento automático de requests para modelo apropriado |

### Mudanças no código existente

| Arquivo | Mudança |
|---------|---------|
| `src/model.ts` | Adicionar `drain()` para aguardar queue esvaziar. Expor metadata do modelo (n_params, n_layers). |
| `src/types.ts` | Novos tipos: `ModelPriority`, `VramState`, `PipelineOptions`, `PipelineResult` |
| `src/index.ts` | Exportar `ModelRegistry`, `ModelPipeline`, `LazyModel` |
| `src/worker/ffi.ts` | Bindings para `llama_model_n_params`, `llama_model_n_layer` (metadata para auto GPU layers) |
| `src/worker/llm.worker.ts` | Suporte a mensagens `embed` e `rerank` (depende de embedding support) |

### Inter-worker communication

Workers não compartilham ponteiros FFI. Toda coordenação acontece via main thread:

```
Main Thread (ModelRegistry)
  ├── postMessage → Worker 1 (embed model)
  ├── postMessage → Worker 2 (rerank model)
  └── postMessage → Worker 3 (generation model)
```

O pipeline orquestra chamadas sequenciais: embed → busca vetorial (main thread) → rerank → generate.

### VRAM monitoring

Para Apple Silicon, a abordagem prática é:

```bash
# Memória do sistema (unificada)
sysctl hw.memsize                    # Total
vm_stat | grep "Pages free"          # Livre
```

Via FFI ou `Bun.spawn`, podemos consultar periodicamente. Não existe API Metal pública para VRAM diretamente — no Apple Silicon, VRAM = RAM unificada.

---

## Comparação

| Feature | node-llama-cpp | qmd | bun-llama-cpp (proposto) |
|---------|---------------|-----|--------------------------|
| Multi-modelo | ✅ Via shared `Llama` backend | ✅ 3 modelos com lazy loading | ✅ Via `ModelRegistry` + Workers |
| VRAM tracking | ✅ `MemoryOrchestrator` trio | ✅ Herda de node-llama-cpp | 🔶 `VramTracker` simplificado |
| Auto GPU layers | ✅ Binary search com GGUF parsing | ✅ Herda de node-llama-cpp | 🔶 Heurística baseada em filesize |
| Unified memory | ✅ `getRamUsageFromUnifiedVram` | ✅ Herda | 🔶 Via `sysctl` + heurísticas |
| Lazy loading | ❌ Eager por default | ✅ Promise dedup guards | ✅ `LazyModel` com promise dedup |
| Inactivity eviction | ❌ Manual disposal | ✅ 5 min timer + ref-counting | ✅ Configurável por modelo |
| Session ref-counting | ❌ N/A | ✅ `_activeSessionCount` | ✅ `LifecycleManager` |
| Model priority | ❌ Todos iguais | ❌ Todos iguais | ✅ persistent/normal/evictable |
| Pipeline cascading | ❌ Manual pelo consumer | ✅ Hardcoded em `hybridQuery` | ✅ `ModelPipeline` genérico |
| Hot-swap | ❌ Não suportado | ❌ Não suportado | ✅ Zero-downtime swap |
| Model routing | ❌ Manual | ❌ Hardcoded | ✅ `ModelRouter` automático |
| Worker isolation | N/A (main thread) | N/A (main thread) | ✅ Cada modelo em Worker isolado |
| Thread safety | Via N-API thread-safe functions | Via node-llama-cpp | Via Bun Worker boundary |
| mmap sharing | ✅ Automático (mesmo processo) | ✅ Automático | ✅ Automático (OS-level) |
| Disposal hierarchy | ✅ 3 camadas (DisposeGuard etc.) | ✅ Herda | 🔶 Simplificado (Worker terminate) |

### Legenda

- ✅ Implementado / planejado com design completo
- 🔶 Abordagem simplificada (trade-off consciente)
- ❌ Não suportado

---

## Considerações Práticas

### VRAM Budget em Apple Silicon

| Chip | RAM Total | Budget p/ Modelos (70%) | Exemplo de Alocação |
|------|-----------|------------------------|---------------------|
| M1 (8GB) | 8GB | ~5.6GB | 1 modelo small (Q4) |
| M1 Pro (16GB) | 16GB | ~11.2GB | Embed 300M + Gen 7B Q4 |
| M2 Pro (32GB) | 32GB | ~22.4GB | Embed 300M + Rerank 600M + Gen 13B Q4 |
| M3 Max (64GB) | 64GB | ~44.8GB | Embed 1B + Rerank 1B + Gen 70B Q4 |

Regra prática: reservar 30% para OS, Metal driver, e outros processos.

### Ordem de Implementação Recomendada

1. **ModelRegistry** (Nível 1) — baixo risco, alto valor. Funciona com a API existente.
2. **LazyModel + LifecycleManager** (Nível 3) — requer apenas wrapping de `LlamaModel`.
3. **VramTracker** (Nível 2) — requer FFI ou shell-out para consultar memória.
4. **ModelPipeline** (Nível 4) — requer embedding support (gap #1 do gap-analysis).
5. **ModelRouter, Hot-swap** (Nível 4) — polish features, implementar por último.

### Riscos

| Risco | Mitigação |
|-------|-----------|
| Metal assertion failures em multi-worker | Testar empiricamente; cada Worker tem seu próprio Metal command queue |
| VRAM estimation inaccuracies | Começar conservador (60% do total); ajustar com dados reais |
| Worker overhead (memória base por worker) | Cada Bun Worker consome ~30–50MB base; para 3 modelos = ~150MB overhead |
| `llama_backend_init` multiple times | Cada Worker chama independentemente; llama.cpp suporta isso |
| Promise dedup edge cases | Timeout no load promise; retry com backoff |

---

*Documento gerado a partir da análise de node-llama-cpp (`docs/analysis-node-llama-cpp-multi-model-vram.md`), qmd (`referencias/qmd/qmd-embedding-pipeline-analysis.md`), e source code atual (`src/model.ts`, `src/worker/llm.worker.ts`).*
