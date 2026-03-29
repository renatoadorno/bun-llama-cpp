# Sequences Paralelas: Análise e Estratégia

> Documento temático comparando abordagens de sequences paralelas e propondo estratégias além das referências existentes.

---

## O Problema

### Por que sequences paralelas são importantes

Em cenários de produção, um único modelo precisa atender múltiplas requisições simultaneamente:

- **Server scenarios** — Múltiplos usuários fazendo chat com o mesmo modelo. Sem paralelismo, cada request espera a anterior terminar (tempo de resposta O(n) com n requests na fila).
- **Batch processing** — Gerar embeddings, classificações ou respostas para dezenas/centenas de prompts. Serial é inaceitável.
- **Beam search** — Múltiplas hipóteses exploradas em paralelo no mesmo context.
- **A/B de samplers** — Mesma prompt, diferentes configs de sampling, avaliação simultânea.

O gargalo não é CPU — é o **KV cache na GPU**. Cada sequence precisa de seu próprio estado de atenção (keys + values para cada token processado). O desafio é gerenciar memória de KV cache eficientemente entre sequences concorrentes.

### O que llama.cpp oferece nativamente

llama.cpp suporta sequences paralelas via API de batch:

```c
// Context com N sequences
llama_context_params params = llama_context_default_params();
params.n_ctx = 4096;      // context window total
params.n_seq_max = 8;     // até 8 sequences paralelas
params.n_batch = 512;     // tokens por batch de decode

// Cada token no batch tem um seq_id
llama_batch_add(batch, token_id, position, seq_id, logits);

// Um único decode processa tokens de TODAS sequences
llama_decode(ctx, batch);

// Gerenciamento de KV cache por sequence
llama_memory_seq_rm(mem, seq_id, p0, p1);    // remover range
llama_memory_seq_cp(mem, src, dst, p0, p1);  // copiar (prefix sharing!)
llama_memory_seq_add(mem, seq_id, p0, p1, delta); // shift posições
llama_memory_seq_keep(mem, seq_id);           // manter apenas uma seq
```

A chave é que `llama_decode()` processa tokens de **múltiplas sequences em um único batch GPU**, maximizando utilização do hardware.

---

## Como node-llama-cpp Implementa

### Abordagem: Pool-based, KV partitioned

node-llama-cpp usa um modelo de **partições fixas de KV cache**, onde cada sequence recebe um bloco independente de tamanho `contextSize`.

```
KV Cache Total = contextSize × n_sequences
┌──────────────┬──────────────┬──────────────┐
│   Seq 0      │   Seq 1      │   Seq 2      │
│  4096 cells  │  4096 cells  │  4096 cells  │
└──────────────┴──────────────┴──────────────┘
Total: 12288 cells → 12288 × 2 × n_heads × d_head bytes
```

### Código-chave e decisões de design

**1. Alocação de sequence IDs — Pool com reciclagem:**

```ts
// Pool-based: IDs reciclados via FinalizationRegistry
private _popSequenceId(): number | null {
  if (this._unusedSequenceIds.length > 0)
    return this._unusedSequenceIds.shift()!;
  if (this._nextGeneratedSequenceId < this._totalSequences) {
    const sequenceId = this._nextGeneratedSequenceId++;
    return sequenceId;
  }
  return null; // sem slots livres
}
```

**2. Batch scheduling — Producer-consumer com priorização:**

```ts
// Estratégia "maximumParallelism" (default):
// 1. Divide batch igualmente: minTokens = batchSize / numItems
// 2. Cada item recebe min(tokenCount, minTokens)
// 3. Redistribui sobras em 3 passes round-robin
// Garante que TODAS sequences progridem a cada batch
```

**3. Dispatch imediato quando todas sequences têm trabalho:**

```ts
private _scheduleDecode() {
  // Otimização: dispatch imediato quando todas sequences estão prontas
  if (this._queuedDecodeSequenceIds.size === this._totalSequences)
    dispatch();  // sem esperar next tick
  else
    setImmediate(dispatch);  // espera mais sequences enfileirarem
}
```

### Pontos fortes

| Aspecto | Detalhe |
|---------|---------|
| Robusto | Retry com shrink automático em OOM (até 16x, -16% cada) |
| Flexível | Estratégias plugáveis (maximumParallelism, FIFO, custom) |
| KV quantizado | Suporte experimental a Q4/Q8 para key/value types |
| Context shift | Erase automático com preservação de BOS token |
| Speculative | Suporte a draft model + prompt-lookup prediction |

### Pontos fracos

| Limitação | Impacto |
|-----------|---------|
| **KV não compartilhado** | `contextSize × N` de memória, mesmo com prompts idênticos |
| **Sem prefix sharing** | Cada sequence re-processa system prompt independentemente |
| **JS event loop** | Batch dispatch no event loop principal — latência de scheduling |
| **Sampler por avaliação** | Aloca/desaloca sampler a cada `.evaluate()` call |
| **Context fixo** | Sem redimensionamento dinâmico após criação |

---

## Estado Atual do bun-llama-cpp

### Single sequence, serial queue

O bun-llama-cpp atual é estritamente serial:

```
Request A ──► [SerialQueue] ──► Worker (prefill + generate) ──► Response A
                                    │
Request B ────────── espera ─────────┘──► prefill + generate ──► Response B
```

**Código atual em `inference.ts`:**

```ts
// Prefill: seq_id sempre 0
for (let i = 0; i < tokens.length; i++) {
  S.shim_batch_add(batchBuf, tokens[i]!, i, 0, i === tokens.length - 1)
  //                                        ↑ seq_id fixo em 0
}

// Generation: seq_id sempre 0
S.shim_batch_add(batchBuf, token, pos, 0, true)
//                                     ↑ seq_id fixo em 0
```

**Código atual em `model.ts`:**

```ts
class LlamaModel {
  private queue = new SerialQueue()
  // Uma única fila — requests nunca rodam em paralelo

  async infer(prompt, options) {
    return this.queue.enqueue(() => this.doInfer(prompt, options))
    // doInfer bloqueia o worker inteiro até completar
  }
}
```

### O batch API já suporta seq_id mas não é usado

O shim `shim_batch_add` aceita `seq_id` como parâmetro:

```ts
// Em ffi.ts — já expõe seq_id
shim_batch_add: {
  args: [FFIType.ptr, FFIType.i32, FFIType.i32, FFIType.i32, FFIType.bool],
  //                  token_id     position     seq_id       logits
  returns: FFIType.void,
}
```

O `shim_batch_init` aceita `n_seq_max`:

```ts
shim_batch_init: {
  args: [FFIType.ptr, FFIType.i32, FFIType.i32, FFIType.i32],
  //                  n_tokens     embd         n_seq_max
  returns: FFIType.void,
}
```

Mas ambos são chamados com valores fixos (`seq_id = 0`, `n_seq_max = 1`).

### Gap específico

| Componente | Estado | Necessário |
|------------|--------|------------|
| `n_seq_max` no context params | Não configurável (default = 1) | Shim `shim_ctx_params_set_n_seq_max` |
| Sequence slot management | Inexistente | Sequence allocator no Worker |
| Batch multi-sequence | Suportado pela FFI, não usado | Batch scheduler |
| KV cache per-sequence | Não gerenciado | `llama_memory_seq_rm/cp/add/keep` |
| Concurrent API | SerialQueue bloqueia | Sequence-aware dispatcher |
| Per-sequence sampler | Sampler único global | Sampler pool ou per-request |

---

## Estratégia Proposta para bun-llama-cpp

### Nível 1: Básico (implementar primeiro)

#### Multi-sequence no mesmo context

Configurar o context para suportar múltiplas sequences:

```ts
// Nova config
interface ModelConfig {
  // ... existentes ...
  nSeqMax?: number  // default: 1 (compatibilidade)
}

// Em initModel():
S.shim_ctx_params_set_n_seq_max(cpBuf, config.nSeqMax)
S.shim_ctx_params_set_n_ctx(cpBuf, config.nCtx * config.nSeqMax) // KV para todas
S.shim_batch_init(batchBuf, config.nCtx, 0, config.nSeqMax)
```

#### Sequence slot allocation

```ts
// No Worker — gerenciador de slots
interface SequenceSlot {
  seqId: number
  state: 'idle' | 'prefilling' | 'generating'
  position: number       // próxima posição no KV cache
  samplerPtr: number     // sampler próprio
  requestId: string      // request que está usando
}

class SequenceAllocator {
  private slots: Map<number, SequenceSlot> = new Map()
  private freeIds: number[] = []

  constructor(nSeqMax: number) {
    for (let i = 0; i < nSeqMax; i++) {
      this.freeIds.push(i)
    }
  }

  acquire(requestId: string): SequenceSlot | null {
    const seqId = this.freeIds.pop()
    if (seqId === undefined) return null

    const slot: SequenceSlot = {
      seqId,
      state: 'idle',
      position: 0,
      samplerPtr: 0, // inicializado separadamente
      requestId,
    }
    this.slots.set(seqId, slot)
    return slot
  }

  release(seqId: number): void {
    const slot = this.slots.get(seqId)
    if (!slot) return

    // Limpar KV cache desta sequence
    // llama_memory_seq_rm(mem, seqId, -1, -1) → limpa tudo
    this.slots.delete(seqId)
    this.freeIds.push(seqId)
  }
}
```

#### KV cache partitioning

```ts
// Novo shim necessário
export interface LibShimsExtended extends LibShims {
  shim_ctx_params_set_n_seq_max: (buf: Buffer, n: number) => void
}

// Novas FFI bindings necessárias
export interface LibLlamaExtended extends LibLlama {
  llama_memory_seq_rm:   (mem: number, seqId: number, p0: number, p1: number) => boolean
  llama_memory_seq_cp:   (mem: number, src: number, dst: number, p0: number, p1: number) => void
  llama_memory_seq_add:  (mem: number, seqId: number, p0: number, p1: number, delta: number) => void
  llama_memory_seq_keep: (mem: number, seqId: number) => void
}
```

### Nível 2: Otimizações

#### Shared KV cache prefix (o que node-llama-cpp NÃO faz)

**O problema:**

Em cenários de server, múltiplos usuários compartilham o mesmo system prompt:

```
Seq 0: [system_prompt | user_msg_A | assistant_response_A]
Seq 1: [system_prompt | user_msg_B | assistant_response_B]
Seq 2: [system_prompt | user_msg_C | assistant_response_C]
         ↑ idêntico — processado 3x, armazenado 3x no KV cache
```

**A solução: prefix sharing via `llama_memory_seq_cp`:**

```ts
// Pseudo-código: prefill compartilhado
async function sharedPrefixPrefill(
  systemPrompt: string,
  sequences: SequenceSlot[],
) {
  const tokens = tokenize(systemPrompt)

  // 1. Prefill UMA VEZ na sequence 0
  batchClear(batchBuf)
  for (let i = 0; i < tokens.length; i++) {
    batchAdd(batchBuf, tokens[i], i, sequences[0].seqId, false)
  }
  decode(ctxPtr, batchBuf)

  // 2. COPIAR KV cache para outras sequences (quase instantâneo)
  const mem = getMemory(ctxPtr)
  for (let i = 1; i < sequences.length; i++) {
    llama_memory_seq_cp(
      mem,
      sequences[0].seqId,   // source
      sequences[i].seqId,   // destination
      0,                    // from position 0
      tokens.length         // até o fim do system prompt
    )
    sequences[i].position = tokens.length
  }

  // Economia: prefill de 500 tokens × 8 sequences
  // node-llama-cpp: 4000 tokens processados (500 × 8)
  // bun-llama-cpp:  500 tokens processados + 7 cópias O(1)
  // Speedup: ~8x no prefill do system prompt
}
```

**Economia de memória:**

Com KV cache compartilhado, as posições do system prompt ocupam memória **uma única vez** na GPU. Para um system prompt de 500 tokens com 8 sequences em um modelo 7B (32 heads, 128 dim):

```
Sem sharing: 500 × 8 × 2 × 32 × 128 × 2 bytes = 128 MB de KV cache
Com sharing: 500 × 1 × 2 × 32 × 128 × 2 bytes = 16 MB + overhead cópia
Economia: ~87% no KV cache do system prompt
```

> **Nota:** O `llama_memory_seq_cp` no llama.cpp cria referências ao mesmo bloco de KV cache — não copia os dados. A cópia real só acontece quando uma sequence diverge (copy-on-write semântico na prática, dependendo da implementação do backend de memória).

#### Smart batch scheduling

**Round-robin vs priority-based:**

```ts
type SchedulingStrategy = 'round-robin' | 'priority' | 'adaptive'

interface BatchScheduler {
  schedule(
    activeSlots: SequenceSlot[],
    batchBudget: number,
  ): Map<number, number>  // seqId → tokensToProcess
}

// Round-robin: cada sequence processa 1 token por batch
class RoundRobinScheduler implements BatchScheduler {
  schedule(slots, budget) {
    const result = new Map<number, number>()
    const perSeq = Math.max(1, Math.floor(budget / slots.length))
    for (const slot of slots) {
      result.set(slot.seqId, slot.state === 'prefilling'
        ? Math.min(slot.remainingPrefill, perSeq)
        : 1  // generation: sempre 1 token
      )
    }
    return result
  }
}

// Priority-based: sequences de alta prioridade processam primeiro
class PriorityScheduler implements BatchScheduler {
  schedule(slots, budget) {
    const sorted = [...slots].sort((a, b) => b.priority - a.priority)
    const result = new Map<number, number>()
    let remaining = budget

    for (const slot of sorted) {
      if (remaining <= 0) break
      const tokens = slot.state === 'prefilling'
        ? Math.min(slot.remainingPrefill, remaining)
        : 1
      result.set(slot.seqId, tokens)
      remaining -= tokens
    }
    return result
  }
}
```

**Adaptive batch size:**

```ts
// Ajustar batch size baseado em utilização
class AdaptiveBatcher {
  private utilizationHistory: number[] = []

  adjustBatchSize(currentBatch: number, activeSeqs: number): number {
    const utilization = activeSeqs / this.maxSeqs

    if (utilization > 0.8 && this.avgDecodeTime < this.targetLatency) {
      // Alta utilização, latência ok → aumentar batch
      return Math.min(currentBatch * 1.5, this.maxBatch)
    }
    if (utilization < 0.3) {
      // Baixa utilização → reduzir batch para economizar memória
      return Math.max(currentBatch * 0.7, this.minBatch)
    }
    return currentBatch
  }
}
```

### Nível 3: Inovações (além das referências)

#### Continuous batching (inspirado em vLLM/TGI)

node-llama-cpp e a maioria das bibliotecas usam **static batching**: todas sequences no batch precisam estar no mesmo estágio (prefill ou generation). Novas requests esperam o batch atual terminar.

**Continuous batching** permite inserir novas requests **enquanto outras geram**:

```ts
// Pseudo-código do loop de continuous batching
async function continuousBatchingLoop(
  ctx: number,
  allocator: SequenceAllocator,
  requestQueue: AsyncQueue<InferRequest>,
) {
  const activeSlots: SequenceSlot[] = []

  while (true) {
    // 1. Admitir novas requests (se há slots livres)
    while (allocator.hasFreeSlots() && requestQueue.hasItems()) {
      const request = requestQueue.dequeue()
      const slot = allocator.acquire(request.id)
      if (!slot) break

      slot.tokens = tokenize(request.prompt)
      slot.state = 'prefilling'
      slot.remainingPrefill = slot.tokens.length
      activeSlots.push(slot)
    }

    if (activeSlots.length === 0) {
      // Nenhum trabalho — esperar nova request
      await requestQueue.waitForItem()
      continue
    }

    // 2. Construir batch misto (prefill + generation)
    batchClear(batchBuf)
    for (const slot of activeSlots) {
      if (slot.state === 'prefilling') {
        // Adicionar chunk do prefill (pode ser parcial)
        const chunk = slot.tokens.slice(
          slot.position,
          slot.position + PREFILL_CHUNK_SIZE
        )
        for (let i = 0; i < chunk.length; i++) {
          const isLast = (slot.position + i + 1 >= slot.tokens.length)
          batchAdd(batchBuf, chunk[i], slot.position + i, slot.seqId, isLast)
        }
        slot.position += chunk.length
        if (slot.position >= slot.tokens.length) {
          slot.state = 'generating'
        }
      } else {
        // Adicionar token de generation
        batchAdd(batchBuf, slot.lastToken, slot.position, slot.seqId, true)
        slot.position++
      }
    }

    // 3. Decode único para TODOS (prefill + generation juntos)
    decode(ctxPtr, batchBuf)

    // 4. Sample para sequences em generation
    for (const slot of activeSlots) {
      if (slot.state === 'generating') {
        const token = sampleToken(slot.samplerPtr, ctxPtr, slot.batchIndex)
        if (isEog(token)) {
          // Sequence terminou — liberar slot, notificar caller
          finalizeSlot(slot)
          allocator.release(slot.seqId)
          activeSlots.splice(activeSlots.indexOf(slot), 1)
        } else {
          slot.lastToken = token
          streamToken(slot.requestId, tokenToPiece(token))
        }
      }
    }

    // 5. LOOP — volta ao passo 1, podendo admitir novas requests
  }
}
```

**Diferença fundamental do static batching:**

```
Static batching (node-llama-cpp):
  Request A ──► [prefill A] [gen A] [gen A] [gen A] ──► done
  Request B ──► [  wait   ] [  wait  ] [prefill B] [gen B] ──► done

Continuous batching (proposto):
  Request A ──► [prefill A] [gen A] [gen A] [gen A] ──► done
  Request B ──► [  wait   ] [prefill B] [gen B] [gen B] ──► done
                              ↑ entra no próximo batch, sem esperar A terminar
```

#### Preemptive scheduling

```ts
// Pausar sequences de baixa prioridade quando memória aperta
interface PreemptionPolicy {
  shouldPreempt(slot: SequenceSlot, memoryPressure: number): boolean
  selectVictim(activeSlots: SequenceSlot[]): SequenceSlot | null
}

class MemoryPressurePreemption implements PreemptionPolicy {
  shouldPreempt(slot: SequenceSlot, memoryPressure: number): boolean {
    // Preemptar quando KV cache > 90% e sequence é de baixa prioridade
    return memoryPressure > 0.9 && slot.priority < 5
  }

  selectVictim(slots: SequenceSlot[]): SequenceSlot | null {
    // Escolher a sequence com menor prioridade e mais progresso
    // (menos trabalho perdido ao pausar uma que está quase pronta)
    return slots
      .filter(s => s.state === 'generating')
      .sort((a, b) => {
        // Prioridade ascendente, progresso descendente
        if (a.priority !== b.priority) return a.priority - b.priority
        return b.tokensGenerated - a.tokensGenerated
      })[0] ?? null
  }
}

// Implementação da preempção
async function preemptSequence(
  slot: SequenceSlot,
  state: LlamaState,
): Promise<PreemptedState> {
  // 1. Salvar estado: tokens gerados até agora + posição
  const saved: PreemptedState = {
    requestId: slot.requestId,
    generatedTokens: [...slot.generatedTokens],
    prompt: slot.originalPrompt,
    position: slot.position,
    priority: slot.priority,
  }

  // 2. Limpar KV cache desta sequence
  L.llama_memory_seq_rm(mem, slot.seqId, -1, -1)

  // 3. Liberar slot
  allocator.release(slot.seqId)

  // 4. Quando houver slot livre, restaurar
  // Re-prefill é necessário mas pode usar prefix sharing
  return saved
}
```

#### KV cache compression

**Quantização do KV cache (Q4/Q8):**

llama.cpp suporta nativamente tipos de KV cache quantizado via context params:

```ts
// Configuração via context params (já suportado pelo llama.cpp)
// Novos shims necessários:
// shim_ctx_params_set_type_k(buf, ggml_type)
// shim_ctx_params_set_type_v(buf, ggml_type)

interface KVCacheConfig {
  keyType: 'f16' | 'q8_0' | 'q4_0'   // default: f16
  valueType: 'f16' | 'q8_0' | 'q4_0' // default: f16
}

// Impacto na memória para modelo 7B, ctx 4096, 8 sequences:
// F16/F16: 4096 × 8 × 2 × 32 × 128 × 2 = 512 MB
// Q8/Q8:   4096 × 8 × 2 × 32 × 128 × 1 = 256 MB (50% redução)
// Q4/Q4:   4096 × 8 × 2 × 32 × 128 × 0.5 = 128 MB (75% redução)
```

**Selective attention (manter apenas tokens importantes):**

```ts
// Conceito: não manter TODOS os tokens no KV cache.
// Tokens com baixa attention score podem ser evicted.
//
// Implementação requer instrumentação do attention:
// 1. Periodicamente, calcular attention scores médios por token
// 2. Evictar tokens com score < threshold
// 3. Manter "anchor tokens" (BOS, separadores, últimos N)
//
// Isso é ALÉM do que llama.cpp oferece nativamente.
// Requer modificação do decode loop para track attention ou
// uso de modelos que suportam GQA/MQA com attention sinks.

interface SelectiveEvictionPolicy {
  anchorTokens: number      // manter primeiros N tokens sempre
  recentWindow: number      // manter últimos N tokens sempre
  evictionThreshold: number // score abaixo do qual evictar
  checkInterval: number     // verificar a cada N decodes
}

function evictLowAttentionTokens(
  mem: number,
  seqId: number,
  policy: SelectiveEvictionPolicy,
  totalTokens: number,
): void {
  // Manter: [0..anchorTokens] + [totalTokens-recentWindow..totalTokens]
  // Evictar seletivamente no meio baseado em heurística
  const evictStart = policy.anchorTokens
  const evictEnd = totalTokens - policy.recentWindow

  // Heurística simples: evictar metade dos tokens do meio
  // (tokens mais antigos, exceto anchors)
  const midpoint = Math.floor((evictStart + evictEnd) / 2)
  L.llama_memory_seq_rm(mem, seqId, evictStart, midpoint)
  L.llama_memory_seq_add(mem, seqId, midpoint, evictEnd, -(midpoint - evictStart))
}
```

#### Speculative prefill

**Pré-computar prefill de prompts frequentes:**

```ts
// Hot cache para system prompts — KV cache pre-computado mantido em memória
class PrefillCache {
  private cache = new Map<string, {
    tokens: number[]
    seqId: number       // sequence reservada com KV cache pronto
    refCount: number    // quantas sequences estão usando
    lastUsed: number
  }>()

  // Reservar um slot permanente para system prompts frequentes
  async warmup(systemPrompt: string, allocator: SequenceAllocator) {
    const tokens = tokenize(systemPrompt)
    const key = this.hashTokens(tokens)

    if (this.cache.has(key)) return

    // Alocar slot dedicado e fazer prefill
    const slot = allocator.acquireReserved()
    if (!slot) return

    batchClear(batchBuf)
    for (let i = 0; i < tokens.length; i++) {
      batchAdd(batchBuf, tokens[i], i, slot.seqId, false)
    }
    decode(ctxPtr, batchBuf)

    this.cache.set(key, {
      tokens,
      seqId: slot.seqId,
      refCount: 0,
      lastUsed: Date.now(),
    })
  }

  // Quando nova request chega com este system prompt:
  // Copiar KV em vez de re-computar
  async applyToSequence(
    systemPrompt: string,
    targetSeqId: number,
  ): Promise<number> { // retorna posição após prefix
    const key = this.hashTokens(tokenize(systemPrompt))
    const cached = this.cache.get(key)
    if (!cached) return 0

    // Copiar KV cache instantaneamente
    L.llama_memory_seq_cp(
      mem, cached.seqId, targetSeqId,
      0, cached.tokens.length
    )
    cached.refCount++
    cached.lastUsed = Date.now()

    return cached.tokens.length
  }

  private hashTokens(tokens: number[]): string {
    // Hash rápido dos primeiros + últimos tokens + length
    const sample = [
      ...tokens.slice(0, 8),
      tokens.length,
      ...tokens.slice(-8),
    ]
    return sample.join(',')
  }
}
```

---

## Impacto na Arquitetura

### Mudanças necessárias no Worker

```
ANTES (single sequence):
┌─────────────────────────────────────────┐
│ Worker                                   │
│  ┌──────────┐  ┌─────────────────────┐  │
│  │ FFI libs  │  │ LlamaState (único)  │  │
│  └──────────┘  │  modelPtr            │  │
│                │  ctxPtr              │  │
│                │  samplerPtr          │  │
│                │  batchBuf            │  │
│                └─────────────────────┘  │
│  message → initModel → runInference     │
│          (bloqueante, serial)            │
└─────────────────────────────────────────┘

DEPOIS (multi-sequence):
┌──────────────────────────────────────────────────┐
│ Worker                                            │
│  ┌──────────┐  ┌──────────────────────────────┐  │
│  │ FFI libs  │  │ SharedState                   │  │
│  └──────────┘  │  modelPtr, ctxPtr, batchBuf   │  │
│                │  memPtr (KV cache handle)      │  │
│                └──────────────────────────────┘  │
│  ┌───────────────┐  ┌─────────────────────────┐  │
│  │ SeqAllocator   │  │ Slots                    │  │
│  │ acquire/release│  │ [0]: {sampler, pos, ...} │  │
│  └───────────────┘  │ [1]: {sampler, pos, ...} │  │
│                     │ [N]: {sampler, pos, ...} │  │
│  ┌───────────────┐  └─────────────────────────┘  │
│  │ BatchScheduler │                               │
│  │ buildBatch()   │                               │
│  │ dispatchDecode │                               │
│  └───────────────┘                               │
│  ┌───────────────┐                               │
│  │ PrefillCache   │ ← hot system prompt cache     │
│  └───────────────┘                               │
│                                                   │
│  message loop:                                    │
│    'infer'    → acquire slot → enqueue            │
│    'cancel'   → abort sequence                    │
│    'shutdown' → cleanup all                       │
│                                                   │
│  batch loop (async, continuous):                  │
│    buildBatch() → decode() → sample() → stream    │
└──────────────────────────────────────────────────┘
```

### Novas FFI functions necessárias

```ts
// === Novas bindings em libllama ===
llama_memory_seq_rm:   (mem, seq_id, p0, p1) => boolean
llama_memory_seq_cp:   (mem, seq_src, seq_dst, p0, p1) => void
llama_memory_seq_add:  (mem, seq_id, p0, p1, delta) => void
llama_memory_seq_keep: (mem, seq_id) => void
llama_memory_can_shift:(mem) => boolean   // suporte a context shift

// === Novos shims ===
shim_ctx_params_set_n_seq_max:  (buf, n) => void
shim_ctx_params_set_n_batch:    (buf, n) => void
shim_ctx_params_set_flash_attn: (buf, enable) => void
shim_ctx_params_set_type_k:     (buf, ggml_type) => void  // KV quantization
shim_ctx_params_set_type_v:     (buf, ggml_type) => void
```

### Mudanças no protocolo Worker ↔ Main thread

```ts
// === Mensagens Main → Worker ===
type WorkerRequest =
  | { type: 'init'; modelPath: string; config: ResolvedConfig }
  | { type: 'infer'; id: string; prompt: string; maxTokens: number;
      abortFlag: Int32Array; priority?: number }
  | { type: 'cancel'; id: string }        // ← NOVO: cancelar request específica
  | { type: 'warmup'; systemPrompt: string } // ← NOVO: pré-aquecer cache
  | { type: 'shutdown' }

// === Mensagens Worker → Main ===
type WorkerResponse =
  | { type: 'ready' }
  | { type: 'token'; id: string; text: string }
  | { type: 'done'; id: string; tokenCount: number }
  | { type: 'aborted'; id: string; tokenCount: number }
  | { type: 'queued'; id: string; position: number }  // ← NOVO: request enfileirada
  | { type: 'slot_acquired'; id: string; seqId: number } // ← NOVO
  | { type: 'error'; id?: string; message: string }
```

### Impacto na API pública

```ts
class LlamaModel {
  // Sem mudança na interface básica — backward compatible
  async infer(prompt: string, options: InferOptions): Promise<InferResult>

  // NOVA: inferência com prioridade
  async infer(prompt: string, options: InferOptions & {
    priority?: number        // 1-10, default 5
  }): Promise<InferResult>

  // NOVA: pré-aquecer system prompt
  async warmup(systemPrompt: string): Promise<void>

  // NOVA: métricas de utilização
  get stats(): {
    activeSequences: number
    totalSequences: number
    kvCacheUtilization: number
    batchUtilization: number
  }

  // EXISTENTE: agora suporta múltiplas chamadas simultâneas
  // (antes: SerialQueue bloqueava; agora: dispatch para slots)
}

// Uso:
const llm = await LlamaModel.load('./model.gguf', {
  nSeqMax: 4,          // 4 sequences paralelas
  nCtx: 4096,          // context per sequence
  kvCacheType: 'q8_0', // KV quantizado
})

// Pré-aquecer system prompt
await llm.warmup("You are a helpful assistant...")

// 4 requests simultâneas — todas executam em paralelo
const [r1, r2, r3, r4] = await Promise.all([
  llm.infer("User: Hello",  { onToken: console.log, maxTokens: 100 }),
  llm.infer("User: Hi",     { onToken: console.log, maxTokens: 200 }),
  llm.infer("User: Hey",    { onToken: console.log, maxTokens: 150 }),
  llm.infer("User: Howdy",  { onToken: console.log, maxTokens: 50 }),
])
```

---

## Vantagens do Bun para esta Implementação

O Bun oferece vantagens específicas que tornam esta implementação mais eficiente que node-llama-cpp:

| Vantagem Bun | Impacto |
|---|---|
| **`bun:ffi` direto** | Sem overhead de N-API/addon build. `dlopen` + chamadas diretas. Menos latência por FFI call (~10x mais rápido que N-API). |
| **Workers nativos** | Thread isolation sem overhead. Cada Worker é uma thread real com seu próprio event loop. |
| **`SharedArrayBuffer`** | Já usado para abort. Pode ser estendido para shared metrics, batch status, e sinalização entre main ↔ worker sem `postMessage`. |
| **`Bun.ArrayBufferSink`** | Acumulação eficiente de tokens sem alocações repetidas. |
| **Performance geral** | Bun's JS engine (JavaScriptCore) é mais rápido em tight loops — relevante para batch construction e scheduling. |

```ts
// Exemplo: shared metrics via SharedArrayBuffer (zero-copy)
const metricsBuf = new SharedArrayBuffer(32)
const metrics = new Int32Array(metricsBuf)
// [0] = active_sequences
// [1] = pending_requests
// [2] = total_tokens_generated
// [3] = batch_utilization_percent
// [4] = kv_cache_utilization_percent

// Worker atualiza atomicamente:
Atomics.store(metrics, 0, activeSlots.length)
Atomics.add(metrics, 2, tokensGenerated)

// Main thread lê sem postMessage:
const active = Atomics.load(metrics, 0)
```

---

## Comparação: node-llama-cpp vs Proposta bun-llama-cpp

| Feature | node-llama-cpp | bun-llama-cpp (proposto) |
|---|---|---|
| **Multi-sequence** | ✅ Pool-based, N partições fixas | ✅ Pool-based, N slots dinâmicos |
| **KV cache model** | Particionado: `ctxSize × N` | Particionado + shared prefix |
| **Prefix sharing** | ❌ Cada seq re-processa system prompt | ✅ `seq_cp` para compartilhar prefix |
| **Batch scheduling** | maximumParallelism / FIFO | Round-robin / Priority / Adaptive |
| **Continuous batching** | ❌ Static batching | ✅ Novas requests entram mid-batch |
| **Preemption** | ❌ Sem preempção | ✅ Memory-pressure based |
| **KV cache quantization** | ✅ Experimental (Q4/Q8 types) | ✅ Via context params |
| **Speculative decoding** | ✅ Draft model + prompt lookup | 🔜 Fase futura |
| **Context shift** | ✅ Erase beginning + custom | ✅ Erase + selective eviction |
| **System prompt cache** | ❌ Re-avalia sempre | ✅ PrefillCache com hot slots |
| **Metrics/observability** | ❌ Limitado | ✅ SharedArrayBuffer metrics |
| **FFI overhead** | N-API (C++ addon, build necessário) | bun:ffi (dlopen direto, zero build) |
| **Abort mechanism** | Callback-based (event loop) | SharedArrayBuffer + Atomics (bypassa event loop bloqueado) |
| **Scheduling latency** | JS event loop (`setImmediate`) | Worker loop direto (sem yield ao event loop) |

### Resumo estratégico

**node-llama-cpp** priorizou:
- Robustez (retry, fallback, checkpoints)
- Flexibilidade (estratégias plugáveis, grammar, LoRA)
- Ecossistema (N-API, npm, TypeScript nativo)

**bun-llama-cpp** deve priorizar:
- **Performance** — FFI direto, continuous batching, shared metrics
- **Eficiência de memória** — Prefix sharing, KV quantization, selective eviction
- **Throughput de server** — Continuous batching, preemptive scheduling, hot cache
- **Simplicidade** — API mínima com defaults inteligentes, sem complexidade desnecessária

A combinação de `bun:ffi` (latência FFI mínima) + `SharedArrayBuffer` (comunicação zero-copy) + continuous batching (inspirado em vLLM) pode posicionar bun-llama-cpp como a opção mais performante para cenários de inferência server-side em JavaScript/TypeScript.
