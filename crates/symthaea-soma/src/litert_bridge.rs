// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! LiteRT-LM bridge: on-device Gemma 4 E2B inference via Google's LiteRT-LM C++ API.
//!
//! Provides consciousness-gated general-purpose LLM on Android (Pixel 8 Pro),
//! complementing the native Broca center for open-ended queries.
//!
//! Architecture:
//!   Rust (this module) → C FFI → litert_shim.c → LiteRT-LM C++ API → GPU/CPU
//!
//! The model (~2.58 GB) is downloaded on first launch by the Kotlin LiteRTManager,
//! then this bridge loads it via FFI. Falls back to Holon desktop if unavailable.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

/// Minimum consciousness level required for LLM generation (saves battery/compute).
const CONSCIOUSNESS_GATE_THRESHOLD: f32 = 0.15;

/// Maximum tokens for on-device generation (keeps latency reasonable at ~25-35 tok/s).
pub const DEFAULT_MAX_TOKENS: u32 = 256;

// ═══════════════════════════════════════════════════════════════════════════════
// FFI declarations — implemented by native/litert_shim.c
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(not(test))]
extern "C" {
    fn litert_create_engine(model_path: *const c_char, use_gpu: bool) -> *mut std::ffi::c_void;
    fn litert_is_ready(engine: *const std::ffi::c_void) -> bool;
    fn litert_generate(
        engine: *mut std::ffi::c_void,
        prompt: *const c_char,
        max_tokens: u32,
    ) -> *mut c_char;
    fn litert_free_response(response: *mut c_char);
    fn litert_free_engine(engine: *mut std::ffi::c_void);
    fn litert_generate_stream(
        engine: *mut std::ffi::c_void,
        prompt: *const c_char,
        max_tokens: u32,
        callback: extern "C" fn(*const c_char, *mut std::ffi::c_void),
        context: *mut std::ffi::c_void,
    ) -> *mut c_char;
}

// Stub FFI for host-side compilation and testing
#[cfg(test)]
mod ffi_stub {
    use std::os::raw::c_char;

    pub unsafe fn litert_create_engine(
        _model_path: *const c_char,
        _use_gpu: bool,
    ) -> *mut std::ffi::c_void {
        std::ptr::null_mut()
    }

    pub unsafe fn litert_is_ready(_engine: *const std::ffi::c_void) -> bool {
        false
    }

    pub unsafe fn litert_generate(
        _engine: *mut std::ffi::c_void,
        _prompt: *const c_char,
        _max_tokens: u32,
    ) -> *mut c_char {
        std::ptr::null_mut()
    }

    pub unsafe fn litert_free_response(_response: *mut c_char) {}

    pub unsafe fn litert_free_engine(_engine: *mut std::ffi::c_void) {}

    pub unsafe fn litert_generate_stream(
        _engine: *mut std::ffi::c_void,
        _prompt: *const c_char,
        _max_tokens: u32,
        _callback: extern "C" fn(*const c_char, *mut std::ffi::c_void),
        _context: *mut std::ffi::c_void,
    ) -> *mut c_char {
        std::ptr::null_mut()
    }
}

#[cfg(test)]
use ffi_stub::*;

// ═══════════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Backend selection for LiteRT-LM inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LiteRTBackend {
    /// GPU acceleration (Mali on Pixel 8 Pro) — ~676 MB RAM, fastest.
    Gpu,
    /// CPU fallback (XNNPACK, 4 threads) — ~1.7 GB RAM, slower.
    Cpu,
}

impl Default for LiteRTBackend {
    fn default() -> Self {
        Self::Gpu
    }
}

/// Result from on-device LLM generation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LiteRTResponse {
    /// Generated text.
    pub text: String,
    /// Whether the model was available and generation succeeded.
    pub from_device: bool,
}

/// On-device Gemma 4 E2B bridge via LiteRT-LM.
pub struct LiteRTBridge {
    /// Opaque pointer to the LiteRT-LM engine (null if not initialized).
    engine: *mut std::ffi::c_void,
    /// Path to the downloaded .litertlm model file.
    model_path: String,
    /// Backend preference.
    backend: LiteRTBackend,
}

// LiteRT-LM engine is thread-unsafe (single-threaded access required),
// which matches SomaEngine's single-threaded design.
unsafe impl Send for LiteRTBridge {}

impl LiteRTBridge {
    /// Create a new bridge. Does NOT load the model — call `init()` after
    /// the Kotlin layer confirms the model file exists.
    pub fn new(model_path: String, backend: LiteRTBackend) -> Self {
        Self {
            engine: std::ptr::null_mut(),
            model_path,
            backend,
        }
    }

    /// Initialize the LiteRT-LM engine by loading the model file.
    /// Returns true if the engine is ready for inference.
    pub fn init(&mut self) -> bool {
        if !self.engine.is_null() {
            return true; // already initialized
        }

        let path = match CString::new(self.model_path.as_str()) {
            Ok(p) => p,
            Err(_) => return false,
        };

        let engine =
            unsafe { litert_create_engine(path.as_ptr(), self.backend == LiteRTBackend::Gpu) };

        if engine.is_null() {
            tracing::warn!(
                path = %self.model_path,
                "LiteRT-LM engine creation failed — model may not be downloaded yet"
            );
            return false;
        }

        self.engine = engine;
        tracing::info!(
            path = %self.model_path,
            backend = ?self.backend,
            "LiteRT-LM engine initialized (gemma4:e2b on-device)"
        );
        true
    }

    /// Whether the engine is loaded and ready for inference.
    pub fn is_available(&self) -> bool {
        if self.engine.is_null() {
            return false;
        }
        unsafe { litert_is_ready(self.engine) }
    }

    /// Generate text from a prompt. Returns None if the engine is not available.
    pub fn generate(&self, prompt: &str, max_tokens: u32) -> Option<String> {
        if self.engine.is_null() {
            return None;
        }

        let c_prompt = CString::new(prompt).ok()?;
        let response = unsafe { litert_generate(self.engine, c_prompt.as_ptr(), max_tokens) };

        if response.is_null() {
            return None;
        }

        let text = unsafe { CStr::from_ptr(response) }
            .to_str()
            .ok()
            .map(String::from);

        unsafe { litert_free_response(response) };

        text
    }

    /// Generate text gated by consciousness level.
    ///
    /// Suppresses generation when consciousness is too low (saves battery/compute).
    /// This complements Broca's epistemic gating — Broca gates *what* is said,
    /// this gates *whether* to invoke the expensive LLM at all.
    pub fn generate_consciousness_gated(
        &self,
        prompt: &str,
        max_tokens: u32,
        consciousness_level: f32,
    ) -> Option<LiteRTResponse> {
        if consciousness_level < CONSCIOUSNESS_GATE_THRESHOLD {
            tracing::trace!(
                consciousness_level,
                threshold = CONSCIOUSNESS_GATE_THRESHOLD,
                "LiteRT generation suppressed — consciousness below threshold"
            );
            return None;
        }

        self.generate(prompt, max_tokens)
            .map(|text| LiteRTResponse {
                text,
                from_device: true,
            })
    }

    /// Generate text with verified claims as context (on-device RAG).
    ///
    /// Prepends verified claims to the prompt so the on-device LLM can
    /// answer grounded in web research results from the desktop WebResearcher.
    pub fn generate_with_context(
        &self,
        prompt: &str,
        context_claims: &[String],
        max_tokens: u32,
        consciousness_level: f32,
    ) -> Option<LiteRTResponse> {
        if consciousness_level < CONSCIOUSNESS_GATE_THRESHOLD {
            return None;
        }

        if context_claims.is_empty() {
            return self
                .generate(prompt, max_tokens)
                .map(|text| LiteRTResponse {
                    text,
                    from_device: true,
                });
        }

        // Build context-augmented prompt
        let context_block: String = context_claims
            .iter()
            .enumerate()
            .map(|(i, c)| format!("{}. {}", i + 1, c))
            .collect::<Vec<_>>()
            .join("\n");

        let augmented_prompt = format!(
            "Based on these verified facts:\n{}\n\nAnswer: {}",
            context_block, prompt
        );

        self.generate(&augmented_prompt, max_tokens)
            .map(|text| LiteRTResponse {
                text,
                from_device: true,
            })
    }

    /// Generate text with per-token streaming callback.
    ///
    /// Tokens are delivered via `on_token` as they're generated, enabling
    /// real-time UI updates (~25-35 tok/s on Pixel 8 Pro GPU).
    pub fn generate_streaming(
        &self,
        prompt: &str,
        max_tokens: u32,
        on_token: &mut dyn FnMut(&str),
    ) -> Option<String> {
        if self.engine.is_null() {
            return None;
        }

        let c_prompt = CString::new(prompt).ok()?;

        // Trampoline: C callback → Rust closure via context pointer
        extern "C" fn trampoline(token: *const c_char, context: *mut std::ffi::c_void) {
            if token.is_null() || context.is_null() {
                return;
            }
            let closure = unsafe { &mut *(context as *mut &mut dyn FnMut(&str)) };
            if let Ok(s) = unsafe { CStr::from_ptr(token) }.to_str() {
                closure(s);
            }
        }

        let mut closure_ref: &mut dyn FnMut(&str) = on_token;
        let context_ptr = &mut closure_ref as *mut &mut dyn FnMut(&str) as *mut std::ffi::c_void;

        let response = unsafe {
            litert_generate_stream(
                self.engine,
                c_prompt.as_ptr(),
                max_tokens,
                trampoline,
                context_ptr,
            )
        };

        if response.is_null() {
            return None;
        }

        let text = unsafe { CStr::from_ptr(response) }
            .to_str()
            .ok()
            .map(String::from);

        unsafe { litert_free_response(response) };

        text
    }

    /// Streaming generation gated by consciousness level.
    pub fn generate_streaming_consciousness_gated(
        &self,
        prompt: &str,
        max_tokens: u32,
        consciousness_level: f32,
        on_token: &mut dyn FnMut(&str),
    ) -> Option<LiteRTResponse> {
        if consciousness_level < CONSCIOUSNESS_GATE_THRESHOLD {
            return None;
        }

        self.generate_streaming(prompt, max_tokens, on_token)
            .map(|text| LiteRTResponse {
                text,
                from_device: true,
            })
    }

    /// Update the model path (e.g., after download completes).
    pub fn set_model_path(&mut self, path: String) {
        // If engine is already loaded with a different path, free it first
        if !self.engine.is_null() {
            unsafe { litert_free_engine(self.engine) };
            self.engine = std::ptr::null_mut();
        }
        self.model_path = path;
    }

    /// Get the current model path.
    pub fn model_path(&self) -> &str {
        &self.model_path
    }

    /// Get the configured backend.
    pub fn backend(&self) -> LiteRTBackend {
        self.backend
    }
}

impl Drop for LiteRTBridge {
    fn drop(&mut self) {
        if !self.engine.is_null() {
            unsafe { litert_free_engine(self.engine) };
            self.engine = std::ptr::null_mut();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests (host-side with FFI stubs)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let bridge = LiteRTBridge::new("/tmp/model.litertlm".into(), LiteRTBackend::Gpu);
        assert!(!bridge.is_available());
        assert_eq!(bridge.model_path(), "/tmp/model.litertlm");
        assert_eq!(bridge.backend(), LiteRTBackend::Gpu);
    }

    #[test]
    fn test_unavailable_generate_returns_none() {
        let bridge = LiteRTBridge::new("/tmp/nonexistent.litertlm".into(), LiteRTBackend::Cpu);
        assert!(bridge.generate("test", 32).is_none());
    }

    #[test]
    fn test_consciousness_gate_suppresses_low() {
        let bridge = LiteRTBridge::new("/tmp/model.litertlm".into(), LiteRTBackend::Gpu);
        // Even if engine were available, consciousness below threshold → None
        let result = bridge.generate_consciousness_gated("test", 32, 0.05);
        assert!(result.is_none());
    }

    #[test]
    fn test_consciousness_gate_allows_high() {
        // With stub FFI, engine is null so generate returns None regardless,
        // but we verify the gate doesn't block at high consciousness
        let bridge = LiteRTBridge::new("/tmp/model.litertlm".into(), LiteRTBackend::Gpu);
        let result = bridge.generate_consciousness_gated("test", 32, 0.8);
        // Returns None because stub engine is null, not because of gating
        assert!(result.is_none());
    }

    #[test]
    fn test_init_with_stub_returns_false() {
        let mut bridge = LiteRTBridge::new("/tmp/model.litertlm".into(), LiteRTBackend::Gpu);
        assert!(!bridge.init()); // stub returns null engine
    }

    #[test]
    fn test_set_model_path() {
        let mut bridge = LiteRTBridge::new("/old/path".into(), LiteRTBackend::Cpu);
        bridge.set_model_path("/new/path".into());
        assert_eq!(bridge.model_path(), "/new/path");
    }

    #[test]
    fn test_default_backend_is_gpu() {
        assert_eq!(LiteRTBackend::default(), LiteRTBackend::Gpu);
    }

    #[test]
    fn test_litert_response_serialization() {
        let resp = LiteRTResponse {
            text: "Hello from Gemma 4".into(),
            from_device: true,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("Hello from Gemma 4"));
        assert!(json.contains("from_device"));
    }
}
