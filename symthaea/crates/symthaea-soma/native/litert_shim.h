// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// LiteRT-LM C shim: thin C wrapper around the LiteRT-LM C++ API.
// Called from Rust via FFI (litert_bridge.rs).
//
// Build (Android NDK r28b+):
//   bazel build --config=android_arm64 //native:litert_shim
// Or with CMake:
//   cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
//         -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 ..

#ifndef LITERT_SHIM_H
#define LITERT_SHIM_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Create a LiteRT-LM engine from a .litertlm model file.
/// Returns an opaque engine pointer, or NULL on failure.
/// @param model_path  Path to the .litertlm model file.
/// @param use_gpu     If true, prefer GPU backend (Mali on Pixel 8 Pro).
void* litert_create_engine(const char* model_path, bool use_gpu);

/// Check if the engine is initialized and ready for inference.
bool litert_is_ready(const void* engine);

/// Generate text from a prompt.
/// Returns a newly allocated C string (caller must free with litert_free_response),
/// or NULL on failure.
/// @param engine      Engine pointer from litert_create_engine().
/// @param prompt      Input prompt (null-terminated UTF-8).
/// @param max_tokens  Maximum tokens to generate.
char* litert_generate(void* engine, const char* prompt, uint32_t max_tokens);

/// Callback type for streaming generation.
/// @param token    The next generated token (null-terminated UTF-8).
/// @param context  User-provided context pointer (passed through unchanged).
typedef void (*litert_stream_callback)(const char* token, void* context);

/// Generate text with per-token streaming callback.
/// The callback is invoked for each generated token.
/// Returns the full concatenated response (caller must free with litert_free_response),
/// or NULL on failure.
char* litert_generate_stream(
    void* engine,
    const char* prompt,
    uint32_t max_tokens,
    litert_stream_callback callback,
    void* context
);

/// Free a response string returned by litert_generate() or litert_generate_stream().
void litert_free_response(char* response);

/// Free the engine and all associated resources.
void litert_free_engine(void* engine);

#ifdef __cplusplus
}
#endif

#endif // LITERT_SHIM_H
