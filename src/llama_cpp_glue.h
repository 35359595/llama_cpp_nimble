#ifndef MICRO_RUNNER_LLAMA_CPP_GLUE_H
#define MICRO_RUNNER_LLAMA_CPP_GLUE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct mr_model_params;
struct mr_context_params;
struct llama_model;
struct llama_context;

typedef struct mr_device_info {
    const char * name;
    const char * description;
    const char * backend;
    size_t memory_free;
    size_t memory_total;
    int type;
} mr_device_info;

int mr_llama_backend_init(void);
void mr_llama_backend_free(void);

bool mr_llama_supports_gpu_offload(void);

struct mr_model_params * mr_model_params_new(void);
void mr_model_params_free(struct mr_model_params * params);
void mr_model_params_set_gpu_layers(struct mr_model_params * params, int32_t n_gpu_layers);
void mr_model_params_set_main_gpu(struct mr_model_params * params, int32_t main_gpu);
void mr_model_params_set_use_mmap(struct mr_model_params * params, bool use_mmap);
void mr_model_params_set_use_mlock(struct mr_model_params * params, bool use_mlock);

struct mr_context_params * mr_context_params_new(void);
void mr_context_params_free(struct mr_context_params * params);
void mr_context_params_set_n_ctx(struct mr_context_params * params, uint32_t n_ctx);
void mr_context_params_set_n_threads(struct mr_context_params * params, int32_t n_threads);
void mr_context_params_set_n_threads_batch(struct mr_context_params * params, int32_t n_threads_batch);

struct llama_model * mr_model_load(const char * path, const struct mr_model_params * params);
void mr_model_free(struct llama_model * model);

struct llama_context * mr_context_new(struct llama_model * model, const struct mr_context_params * params);
void mr_context_free(struct llama_context * ctx);

int32_t mr_model_n_vocab(const struct llama_model * model);
int32_t mr_context_n_ctx(const struct llama_context * ctx);

int32_t mr_tokenize(
    const struct llama_model * model,
    const char * text,
    int32_t * tokens,
    int32_t max_tokens,
    bool add_bos,
    bool special);

int32_t mr_token_to_piece(
    const struct llama_model * model,
    int32_t token,
    char * buf,
    int32_t buf_size,
    bool special);

int32_t mr_token_bos(const struct llama_model * model);
int32_t mr_token_eos(const struct llama_model * model);
bool mr_token_is_eog(const struct llama_model * model, int32_t token);

const char * mr_model_chat_template(const struct llama_model * model);

int32_t mr_chat_apply_template(
    const char * tmpl,
    char ** roles,
    char ** contents,
    size_t n_msg,
    bool add_ass,
    char * buf,
    int32_t length);

void mr_kv_cache_clear(struct llama_context * ctx);

int32_t mr_decode_tokens(
    struct llama_context * ctx,
    const int32_t * tokens,
    int32_t n_tokens);

const float * mr_get_logits(struct llama_context * ctx);

size_t mr_list_devices(mr_device_info * out, size_t max);

#ifdef __cplusplus
}
#endif

#endif
