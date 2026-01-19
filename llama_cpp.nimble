# Package
version       = "0.1.0"
author        = "micro_runner"
description   = "Minimal llama.cpp glue bindings for Nim"
license       = "MIT"

# Directories
srcDir = "src"

import os
let baseDir = thisDir()
let llamaDir = baseDir / "vendor" / "llama.cpp"
let buildDir = llamaDir / "build"

when defined(linux):
  switch("passL", "-L" & buildDir / "src")
  switch("passL", "-L" & buildDir / "ggml" / "src")
  switch("passL", "-lllama")
  switch("passL", "-lggml")
  switch("passL", "-lggml-cpu")
  switch("passL", "-lggml-base")
  switch("passL", "-lggml-vulkan")
  switch("passL", "-lvulkan")
  switch("passL", "-fopenmp")
  switch("passL", "-lpthread")
  switch("passL", "-ldl")
  switch("passL", "-lm")
