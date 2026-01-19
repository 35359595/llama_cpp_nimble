import std/os

const thisDir = currentSourcePath().parentDir()
const vendorDir = thisDir / ".." / "vendor" / "llama.cpp"
const includeDir = vendorDir / "include"
const ggmlIncludeDir = vendorDir / "ggml" / "include"

{.passC: "-I" & thisDir.}
{.passC: "-I" & includeDir.}
{.passC: "-I" & ggmlIncludeDir.}
{.compile: "llama_cpp_glue.cpp".}

when defined(linux):
  {.passC: "-std=c++17".}
  {.passL: "-lstdc++".}

type
  MrModelParams* = pointer
  MrContextParams* = pointer
  LlamaModel* = pointer
  LlamaContext* = pointer

  MrDeviceInfo* {.bycopy, importc: "mr_device_info", header: "llama_cpp_glue.h".} = object
    name*: cstring
    description*: cstring
    backend*: cstring
    memory_free*: csize_t
    memory_total*: csize_t
    deviceType* {.importc: "type".}: cint

proc mr_llama_backend_init*(): cint {.importc, header: "llama_cpp_glue.h".}
proc mr_llama_backend_free*() {.importc, header: "llama_cpp_glue.h".}
proc mr_llama_supports_gpu_offload*(): bool {.importc, header: "llama_cpp_glue.h".}

proc mr_model_params_new*(): MrModelParams {.importc, header: "llama_cpp_glue.h".}
proc mr_model_params_free*(params: MrModelParams) {.importc, header: "llama_cpp_glue.h".}
proc mr_model_params_set_gpu_layers*(params: MrModelParams, n_gpu_layers: cint) {.importc, header: "llama_cpp_glue.h".}
proc mr_model_params_set_main_gpu*(params: MrModelParams, main_gpu: cint) {.importc, header: "llama_cpp_glue.h".}
proc mr_model_params_set_use_mmap*(params: MrModelParams, use_mmap: bool) {.importc, header: "llama_cpp_glue.h".}
proc mr_model_params_set_use_mlock*(params: MrModelParams, use_mlock: bool) {.importc, header: "llama_cpp_glue.h".}
proc mr_model_params_set_use_direct_io*(params: MrModelParams, use_direct_io: bool) {.importc, header: "llama_cpp_glue.h".}

proc mr_context_params_new*(): MrContextParams {.importc, header: "llama_cpp_glue.h".}
proc mr_context_params_free*(params: MrContextParams) {.importc, header: "llama_cpp_glue.h".}
proc mr_context_params_set_n_ctx*(params: MrContextParams, n_ctx: cuint) {.importc, header: "llama_cpp_glue.h".}
proc mr_context_params_set_n_threads*(params: MrContextParams, n_threads: cint) {.importc, header: "llama_cpp_glue.h".}
proc mr_context_params_set_n_threads_batch*(params: MrContextParams, n_threads_batch: cint) {.importc, header: "llama_cpp_glue.h".}

proc mr_model_load*(path: cstring, params: MrModelParams): LlamaModel {.importc, header: "llama_cpp_glue.h".}
proc mr_model_free*(model: LlamaModel) {.importc, header: "llama_cpp_glue.h".}

proc mr_context_new*(model: LlamaModel, params: MrContextParams): LlamaContext {.importc, header: "llama_cpp_glue.h".}
proc mr_context_free*(ctx: LlamaContext) {.importc, header: "llama_cpp_glue.h".}

proc mr_model_n_vocab*(model: LlamaModel): cint {.importc, header: "llama_cpp_glue.h".}
proc mr_context_n_ctx*(ctx: LlamaContext): cint {.importc, header: "llama_cpp_glue.h".}

proc mr_tokenize*(
  model: LlamaModel,
  text: cstring,
  tokens: ptr cint,
  max_tokens: cint,
  add_bos: bool,
  special: bool
): cint {.importc, header: "llama_cpp_glue.h".}

proc mr_token_to_piece*(
  model: LlamaModel,
  token: cint,
  buf: cstring,
  buf_size: cint,
  special: bool
): cint {.importc, header: "llama_cpp_glue.h".}

proc mr_token_bos*(model: LlamaModel): cint {.importc, header: "llama_cpp_glue.h".}
proc mr_token_eos*(model: LlamaModel): cint {.importc, header: "llama_cpp_glue.h".}
proc mr_token_is_eog*(model: LlamaModel, token: cint): bool {.importc, header: "llama_cpp_glue.h".}

proc mr_model_chat_template*(model: LlamaModel): cstring {.importc, header: "llama_cpp_glue.h".}

proc mr_chat_apply_template*(
  tmpl: cstring,
  roles: ptr cstring,
  contents: ptr cstring,
  n_msg: csize_t,
  add_ass: bool,
  buf: cstring,
  length: cint
): cint {.importc, header: "llama_cpp_glue.h".}

proc mr_kv_cache_clear*(ctx: LlamaContext) {.importc, header: "llama_cpp_glue.h".}

proc mr_decode_tokens*(ctx: LlamaContext, tokens: ptr cint, n_tokens: cint): cint {.importc, header: "llama_cpp_glue.h".}

proc mr_get_logits*(ctx: LlamaContext): ptr cfloat {.importc, header: "llama_cpp_glue.h".}

proc mr_list_devices*(outDevices: ptr MrDeviceInfo, max: csize_t): csize_t {.importc, header: "llama_cpp_glue.h".}
