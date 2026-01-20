#include "llama_cpp_glue.h"

#include "llama.h"
#include "ggml-backend.h"

#include <cstring>
#include <vector>

struct mr_model_params {
    llama_model_params params;
};

struct mr_context_params {
    llama_context_params params;
};

int mr_llama_backend_init(void) {
    llama_backend_init();
    return 0;
}

void mr_llama_backend_free(void) {
    llama_backend_free();
}

bool mr_llama_supports_gpu_offload(void) {
    return llama_supports_gpu_offload();
}

struct mr_model_params * mr_model_params_new(void) {
    mr_model_params * params = new mr_model_params();
    params->params = llama_model_default_params();
    return params;
}

void mr_model_params_free(struct mr_model_params * params) {
    delete params;
}

void mr_model_params_set_gpu_layers(struct mr_model_params * params, int32_t n_gpu_layers) {
    if (!params) {
        return;
    }
    params->params.n_gpu_layers = n_gpu_layers;
}

void mr_model_params_set_main_gpu(struct mr_model_params * params, int32_t main_gpu) {
    if (!params) {
        return;
    }
    params->params.main_gpu = main_gpu;
}

void mr_model_params_set_use_mmap(struct mr_model_params * params, bool use_mmap) {
    if (!params) {
        return;
    }
    params->params.use_mmap = use_mmap;
}

void mr_model_params_set_use_mlock(struct mr_model_params * params, bool use_mlock) {
    if (!params) {
        return;
    }
    params->params.use_mlock = use_mlock;
}

void mr_model_params_set_use_direct_io(struct mr_model_params * params, bool use_direct_io) {
    if (!params) {
        return;
    }
    params->params.use_direct_io = use_direct_io;
}

struct mr_context_params * mr_context_params_new(void) {
    mr_context_params * params = new mr_context_params();
    params->params = llama_context_default_params();
    return params;
}

void mr_context_params_free(struct mr_context_params * params) {
    delete params;
}

void mr_context_params_set_n_ctx(struct mr_context_params * params, uint32_t n_ctx) {
    if (!params) {
        return;
    }
    params->params.n_ctx = n_ctx;
}

void mr_context_params_set_n_threads(struct mr_context_params * params, int32_t n_threads) {
    if (!params) {
        return;
    }
    params->params.n_threads = n_threads;
}

void mr_context_params_set_n_threads_batch(struct mr_context_params * params, int32_t n_threads_batch) {
    if (!params) {
        return;
    }
    params->params.n_threads_batch = n_threads_batch;
}

struct llama_model * mr_model_load(const char * path, const struct mr_model_params * params) {
    llama_model_params model_params = llama_model_default_params();
    if (params) {
        model_params = params->params;
    }
    return llama_load_model_from_file(path, model_params);
}

void mr_model_free(struct llama_model * model) {
    if (model) {
        llama_free_model(model);
    }
}

struct llama_context * mr_context_new(struct llama_model * model, const struct mr_context_params * params) {
    if (!model) {
        return nullptr;
    }
    llama_context_params ctx_params = llama_context_default_params();
    if (params) {
        ctx_params = params->params;
    }
    return llama_new_context_with_model(model, ctx_params);
}

void mr_context_free(struct llama_context * ctx) {
    if (ctx) {
        llama_free(ctx);
    }
}

int32_t mr_model_n_vocab(const struct llama_model * model) {
    if (!model) {
        return 0;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return llama_n_vocab(vocab);
}

int32_t mr_context_n_ctx(const struct llama_context * ctx) {
    if (!ctx) {
        return 0;
    }
    return llama_n_ctx(ctx);
}

int32_t mr_tokenize(
    const struct llama_model * model,
    const char * text,
    int32_t * tokens,
    int32_t max_tokens,
    bool add_bos,
    bool special) {
    if (!model || !text || !tokens || max_tokens <= 0) {
        return 0;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return llama_tokenize(vocab, text, (int32_t) std::strlen(text), tokens, max_tokens, add_bos, special);
}

int32_t mr_token_to_piece(
    const struct llama_model * model,
    int32_t token,
    char * buf,
    int32_t buf_size,
    bool special) {
    if (!model || !buf || buf_size <= 0) {
        return -1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return llama_token_to_piece(vocab, token, buf, buf_size, 0, special);
}

int32_t mr_token_bos(const struct llama_model * model) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return llama_token_bos(vocab);
}

int32_t mr_token_eos(const struct llama_model * model) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return llama_token_eos(vocab);
}

bool mr_token_is_eog(const struct llama_model * model, int32_t token) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return llama_token_is_eog(vocab, token);
}

const char * mr_model_chat_template(const struct llama_model * model) {
    if (!model) {
        return nullptr;
    }
    return llama_model_chat_template(model, nullptr);
}

int32_t mr_chat_apply_template(
    const char * tmpl,
    char ** roles,
    char ** contents,
    size_t n_msg,
    bool add_ass,
    char * buf,
    int32_t length) {
    if (!tmpl || !roles || !contents || !buf || length <= 0) {
        return -1;
    }
    std::vector<llama_chat_message> messages;
    messages.reserve(n_msg);
    for (size_t i = 0; i < n_msg; ++i) {
        llama_chat_message msg;
        msg.role = roles[i];
        msg.content = contents[i];
        messages.push_back(msg);
    }
    return llama_chat_apply_template(tmpl, messages.data(), n_msg, add_ass, buf, length);
}

void mr_kv_cache_clear(struct llama_context * ctx) {
    if (!ctx) {
        return;
    }
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_clear(mem, true);
}

int32_t mr_decode_tokens(
    struct llama_context * ctx,
    const int32_t * tokens,
    int32_t n_tokens) {
    if (!ctx || !tokens || n_tokens <= 0) {
        return -1;
    }
    llama_batch batch = llama_batch_get_one((llama_token *) tokens, n_tokens);
    return llama_decode(ctx, batch);
}

const float * mr_get_logits(struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }
    return llama_get_logits(ctx);
}

size_t mr_list_devices(mr_device_info * out, size_t max) {
    size_t count = ggml_backend_dev_count();
    if (!out || max == 0) {
        return count;
    }
    size_t written = 0;
    for (size_t i = 0; i < count && written < max; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_dev_props props;
        std::memset(&props, 0, sizeof(props));
        ggml_backend_dev_get_props(dev, &props);

        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        const char * backend_name = ggml_backend_reg_name(reg);

        int type = 4; // unknown
        switch (props.type) {
            case GGML_BACKEND_DEVICE_TYPE_CPU:
                type = 0;
                break;
            case GGML_BACKEND_DEVICE_TYPE_GPU:
                type = 1;
                break;
            case GGML_BACKEND_DEVICE_TYPE_IGPU:
                type = 2;
                break;
            case GGML_BACKEND_DEVICE_TYPE_ACCEL:
                type = 3;
                break;
            default:
                type = 4;
                break;
        }

        out[written].name = props.name;
        out[written].description = props.description;
        out[written].backend = backend_name;
        out[written].memory_free = props.memory_free;
        out[written].memory_total = props.memory_total;
        out[written].type = type;
        written++;
    }
    return written;
}
