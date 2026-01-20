# llama_cpp Nimble Package

Minimal Nim bindings and C++ glue for llama.cpp.

## Build llama.cpp

```
./src/scripts/build_llama.sh
```

The build script will clone llama.cpp into `vendor/llama.cpp` if it is missing.
For Vulkan builds, install the Vulkan SDK (for `glslc`) or `glslangValidator`.

## Use in another project

Add this to your `.nimble`:

```
requires "https://github.com/35359595/llama_cpp_nimble"
```

Then import:

```
import llama_cpp
```

## Notes

- The build script installs to `scripts/` in Nimble installs.
- Static libs are expected under `vendor/llama.cpp/build`.
- This package is intentionally minimal; extend the glue as needed.
