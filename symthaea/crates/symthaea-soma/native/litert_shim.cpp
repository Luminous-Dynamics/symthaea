// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// LiteRT-LM C shim implementation.
// Wraps the LiteRT-LM C++ API (ModelAssets, Engine, Conversation)
// into C-compatible functions for Rust FFI.
//
// Requires: LiteRT-LM SDK headers + linked library.
// Build: see litert_shim.h for instructions.

#include "litert_shim.h"

#include <cstdlib>
#include <cstring>
#include <string>

// LiteRT-LM C++ API headers
// These come from the LiteRT-LM SDK (github.com/google-ai-edge/LiteRT-LM)
#ifdef LITERT_LM_AVAILABLE
#include "litert_lm/engine.h"
#include "litert_lm/conversation.h"
#include "litert_lm/model_assets.h"
#include "litert_lm/engine_settings.h"
#include "litert_lm/conversation_config.h"
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Internal engine wrapper (hides C++ types from C interface)
// ──────────────��──────────────────────────────────────────────────────────────

struct LiteRTEngine {
#ifdef LITERT_LM_AVAILABLE
    std::unique_ptr<litert::lm::Engine> engine;
    std::unique_ptr<litert::lm::ModelAssets> model_assets;
#endif
    bool ready;
};

// ─────────────────────────────────────────────────────────────────────────────
// Public C API
// ─────────────────────────────────────────────────────────────────────────────

extern "C" {

void* litert_create_engine(const char* model_path, bool use_gpu) {
    if (!model_path) return nullptr;

    auto* wrapper = new (std::nothrow) LiteRTEngine();
    if (!wrapper) return nullptr;
    wrapper->ready = false;

#ifdef LITERT_LM_AVAILABLE
    // Load model assets from file
    auto model_assets = litert::lm::ModelAssets::Create(model_path);
    if (!model_assets) {
        delete wrapper;
        return nullptr;
    }

    // Configure backend (GPU preferred for Tensor G3 Mali)
    auto backend = use_gpu
        ? litert::lm::Backend::kGpu
        : litert::lm::Backend::kCpu;
    auto settings = litert::lm::EngineSettings::CreateDefault(*model_assets, backend);
    if (!settings) {
        delete wrapper;
        return nullptr;
    }

    // Create engine
    auto engine = litert::lm::Engine::CreateEngine(*settings);
    if (!engine) {
        delete wrapper;
        return nullptr;
    }

    wrapper->model_assets = std::move(model_assets);
    wrapper->engine = std::move(engine);
    wrapper->ready = true;
#else
    // SDK not linked — engine creation is a no-op stub
    (void)model_path;
    (void)use_gpu;
#endif

    return wrapper;
}

bool litert_is_ready(const void* engine) {
    if (!engine) return false;
    return static_cast<const LiteRTEngine*>(engine)->ready;
}

char* litert_generate(void* engine, const char* prompt, uint32_t max_tokens) {
    if (!engine || !prompt) return nullptr;
    auto* wrapper = static_cast<LiteRTEngine*>(engine);
    if (!wrapper->ready) return nullptr;

#ifdef LITERT_LM_AVAILABLE
    // Create a conversation for this generation
    auto conv_config = litert::lm::ConversationConfig::CreateDefault(*wrapper->engine);
    if (!conv_config) return nullptr;

    auto conv = litert::lm::Conversation::Create(*conv_config);
    if (!conv) return nullptr;

    // Build the request message (simple text prompt)
    std::string request = std::string("{\"text\": \"") + prompt + "\"}";
    auto response = conv->SendMessage(request);
    if (response.empty()) return nullptr;

    // Allocate a C string copy for the caller
    char* result = static_cast<char*>(malloc(response.size() + 1));
    if (result) {
        memcpy(result, response.c_str(), response.size() + 1);
    }
    return result;
#else
    (void)prompt;
    (void)max_tokens;
    return nullptr;
#endif
}

char* litert_generate_stream(
    void* engine,
    const char* prompt,
    uint32_t max_tokens,
    litert_stream_callback callback,
    void* context
) {
    if (!engine || !prompt) return nullptr;
    auto* wrapper = static_cast<LiteRTEngine*>(engine);
    if (!wrapper->ready) return nullptr;

#ifdef LITERT_LM_AVAILABLE
    auto conv_config = litert::lm::ConversationConfig::CreateDefault(*wrapper->engine);
    if (!conv_config) return nullptr;

    auto conv = litert::lm::Conversation::Create(*conv_config);
    if (!conv) return nullptr;

    // Streaming: accumulate tokens and invoke callback for each
    std::string accumulated;
    std::string request = std::string("{\"text\": \"") + prompt + "\"}";

    // Use streaming SendMessage with per-token callback
    auto on_token = [&accumulated, callback, context](const std::string& token) {
        accumulated += token;
        if (callback) {
            callback(token.c_str(), context);
        }
    };

    // Note: exact streaming API depends on LiteRT-LM SDK version.
    // SendMessage with callback is the documented pattern.
    conv->SendMessage(request, on_token);

    if (accumulated.empty()) return nullptr;

    char* result = static_cast<char*>(malloc(accumulated.size() + 1));
    if (result) {
        memcpy(result, accumulated.c_str(), accumulated.size() + 1);
    }
    return result;
#else
    (void)prompt;
    (void)max_tokens;
    (void)callback;
    (void)context;
    return nullptr;
#endif
}

void litert_free_response(char* response) {
    free(response);
}

void litert_free_engine(void* engine) {
    if (engine) {
        delete static_cast<LiteRTEngine*>(engine);
    }
}

} // extern "C"
