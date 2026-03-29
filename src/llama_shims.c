#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "llama.h"

/* ── Size queries ──────────────────────────────────────────────────── */
size_t shim_sizeof_model_params(void)         { return sizeof(struct llama_model_params);         }
size_t shim_sizeof_context_params(void)       { return sizeof(struct llama_context_params);       }
size_t shim_sizeof_batch(void)                { return sizeof(struct llama_batch);                }
size_t shim_sizeof_sampler_chain_params(void) { return sizeof(struct llama_sampler_chain_params); }

/* ── Default-params fill ───────────────────────────────────────────── */
void shim_model_default_params(struct llama_model_params *out) {
    *out = llama_model_default_params();
}
void shim_context_default_params(struct llama_context_params *out) {
    *out = llama_context_default_params();
}
void shim_sampler_chain_default_params(struct llama_sampler_chain_params *out) {
    *out = llama_sampler_chain_default_params();
}

/* ── Field setters ─────────────────────────────────────────────────── */
void shim_model_params_set_n_gpu_layers(struct llama_model_params *p, int32_t n) {
    p->n_gpu_layers = n;
}
void shim_ctx_params_set_n_ctx(struct llama_context_params *p, uint32_t n) {
    p->n_ctx = n;
}
void shim_ctx_params_set_n_threads(struct llama_context_params *p, int32_t n) {
    p->n_threads       = n;
    p->n_threads_batch = n;
}

/* ── Model / context creation ──────────────────────────────────────── */
struct llama_model *shim_model_load_from_file(
    const char *path, const struct llama_model_params *params)
{
    return llama_model_load_from_file(path, *params);
}
struct llama_context *shim_init_from_model(
    struct llama_model *model, const struct llama_context_params *params)
{
    return llama_init_from_model(model, *params);
}

/* ── Batch helpers ─────────────────────────────────────────────────── */
void shim_batch_init(struct llama_batch *out,
                     int32_t n_tokens, int32_t embd, int32_t n_seq_max) {
    *out = llama_batch_init(n_tokens, embd, n_seq_max);
}

void shim_batch_clear(struct llama_batch *batch) {
    batch->n_tokens = 0;
}

/* id=token_id, pos=position, seq_id=sequence index, logits=compute output */
void shim_batch_add(struct llama_batch *batch,
                    int32_t id, int32_t pos, int32_t seq_id, bool logits) {
    int n = batch->n_tokens;
    batch->token   [n]    = id;
    batch->pos     [n]    = pos;
    batch->n_seq_id[n]    = 1;
    batch->seq_id  [n][0] = seq_id;
    batch->logits  [n]    = logits ? 1 : 0;
    batch->n_tokens++;
}

void shim_batch_free(struct llama_batch *batch) {
    llama_batch_free(*batch);
}

int32_t shim_decode(struct llama_context *ctx, struct llama_batch *batch) {
    return llama_decode(ctx, *batch);
}

/* ── Sampler chain ─────────────────────────────────────────────────── */
struct llama_sampler *shim_sampler_chain_init(
    const struct llama_sampler_chain_params *params) {
    return llama_sampler_chain_init(*params);
}

/* size_t wrappers — bun:ffi cannot reliably pass size_t (8 bytes on arm64) */
struct llama_sampler *shim_sampler_init_top_p(float p, int32_t min_keep) {
    return llama_sampler_init_top_p(p, (size_t)min_keep);
}
struct llama_sampler *shim_sampler_init_min_p(float p, int32_t min_keep) {
    return llama_sampler_init_min_p(p, (size_t)min_keep);
}
