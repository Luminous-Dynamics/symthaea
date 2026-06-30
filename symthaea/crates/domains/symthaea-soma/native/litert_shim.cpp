// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// LiteRT-LM C shim implementation.
// Wraps the LiteRT-LM C API (engine.h) into our simplified interface
// for Rust FFI (litert_bridge.rs).
//
// When LITERT_LM_AVAILABLE is defined (set by build-jni.sh when SDK is present),
// this links against the real LiteRT-LM library. Otherwise, all functions
// return null/false (stub mode).

#include "litert_shim.h"

#include <cstdlib>
#include <cstring>
#include <string>

#ifdef LITERT_LM_AVAILABLE
#include "c/engine.h"  // LiteRT-LM C API
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Internal engine wrapper
// ─────────────────────────────────────────────────────────────────────────────

struct LiteRTEngine {
#ifdef LITERT_LM_AVAILABLE
    LiteRtLmEngine* engine;
    LiteRtLmSession* session;
    LiteRtLmEngineSettings* settings;
    LiteRtLmSessionConfig* session_config;
#endif
    bool ready;
};

// Helper: duplicate a string for return to caller (must be freed with litert_free_response)
static char* dup_string(const char* s) {
    if (!s) return nullptr;
    size_t len = strlen(s);
    char* copy = static_cast<char*>(malloc(len + 1));
    if (copy) memcpy(copy, s, len + 1);
    return copy;
}

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
    // Suppress verbose logging
    litert_lm_set_min_log_level(2);  // ERROR only

    // Create engine settings
    wrapper->settings = litert_lm_engine_settings_create(model_path);
    if (!wrapper->settings) { delete wrapper; return nullptr; }

    // Configure for mobile: reasonable token limits
    litert_lm_engine_settings_set_max_num_tokens(wrapper->settings, 2048);

    // Create engine
    wrapper->engine = litert_lm_engine_create(wrapper->settings);
    if (!wrapper->engine) {
        litert_lm_engine_settings_delete(wrapper->settings);
        delete wrapper;
        return nullptr;
    }

    // Create session config with default parameters
    wrapper->session_config = litert_lm_session_config_create();
    if (!wrapper->session_config) {
        litert_lm_engine_delete(wrapper->engine);
        litert_lm_engine_settings_delete(wrapper->settings);
        delete wrapper;
        return nullptr;
    }

    // Default: max 256 output tokens, top-k sampling
    litert_lm_session_config_set_max_output_tokens(wrapper->session_config, 256);
    SamplerParams sampler = { kTopK, 40, 0.9f, 0.7f };
    litert_lm_session_config_set_sampler_params(wrapper->session_config, sampler);

    // Create session
    wrapper->session = litert_lm_engine_create_session(
        wrapper->engine, wrapper->session_config);
    if (!wrapper->session) {
        litert_lm_session_config_delete(wrapper->session_config);
        litert_lm_engine_delete(wrapper->engine);
        litert_lm_engine_settings_delete(wrapper->settings);
        delete wrapper;
        return nullptr;
    }

    wrapper->ready = true;
#else
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
    // Build text input
    InputData input;
    input.type = kInputText;
    input.data = prompt;
    input.size = strlen(prompt);

    // Override max tokens if specified
    if (max_tokens > 0 && max_tokens != 256) {
        litert_lm_session_config_set_max_output_tokens(
            wrapper->session_config, max_tokens);
    }

    // Generate
    LiteRtLmResponses* responses = litert_lm_session_generate_content(
        wrapper->session, &input, 1);
    if (!responses) return nullptr;

    // Extract first candidate
    const char* text = litert_lm_responses_get_response_text_at(responses, 0);
    char* result = dup_string(text);

    litert_lm_responses_delete(responses);
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
    InputData input;
    input.type = kInputText;
    input.data = prompt;
    input.size = strlen(prompt);

    if (max_tokens > 0 && max_tokens != 256) {
        litert_lm_session_config_set_max_output_tokens(
            wrapper->session_config, max_tokens);
    }

    // Accumulate chunks for the return value
    struct StreamState {
        std::string accumulated;
        litert_stream_callback user_callback;
        void* user_context;
    };

    StreamState state;
    state.user_callback = callback;
    state.user_context = context;

    // LiteRT-LM streaming callback
    auto sdk_callback = [](void* cb_data, const char* chunk,
                           bool is_final, const char* error_msg) {
        auto* s = static_cast<StreamState*>(cb_data);
        if (error_msg) return;  // skip errors
        if (chunk) {
            s->accumulated += chunk;
            if (s->user_callback) {
                s->user_callback(chunk, s->user_context);
            }
        }
    };

    int rc = litert_lm_session_generate_content_stream(
        wrapper->session, &input, 1, sdk_callback, &state);

    if (rc != 0 || state.accumulated.empty()) return nullptr;

    return dup_string(state.accumulated.c_str());
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
    if (!engine) return;
    auto* wrapper = static_cast<LiteRTEngine*>(engine);

#ifdef LITERT_LM_AVAILABLE
    if (wrapper->session) litert_lm_session_delete(wrapper->session);
    if (wrapper->session_config) litert_lm_session_config_delete(wrapper->session_config);
    if (wrapper->engine) litert_lm_engine_delete(wrapper->engine);
    if (wrapper->settings) litert_lm_engine_settings_delete(wrapper->settings);
#endif

    delete wrapper;
}

} // extern "C"
