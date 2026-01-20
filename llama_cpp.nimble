# Package
version       = "0.1.2"
author        = "micro_runner"
description   = "Minimal llama.cpp glue bindings for Nim"
license       = "MIT"

# Directories
skipDirs = @["vendor"]
installFiles = @[
  "llama_cpp.nim",
  "llama_cpp_glue.cpp",
  "llama_cpp_glue.h",
  "scripts/build_llama.sh",
  "scripts/glslc",
  "README.md",
  "LICENSE"
]
