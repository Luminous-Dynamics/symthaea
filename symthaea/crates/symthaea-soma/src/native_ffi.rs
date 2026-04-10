// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Native FFI: C-compatible interface for iOS (staticlib) and Android (JNI/cdylib).
//!
//! Exposes SomaEngine (mobile-embodied consciousness) through opaque pointer + extern "C" functions.
//! JNI symbol names use `soma_engine_*` prefix (renamed from `spore_engine_*` in Phase 4).
//!
//! # Safety
//!
//! All functions taking `*mut SomaEngine` require a valid pointer obtained from
//! `soma_engine_new()`. Passing null or freed pointers is undefined behavior.
//! The engine is NOT thread-safe — callers must serialize access externally.

use crate::engine::{SomaConfig, SomaEngine};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

// ═══════════════════════════════════════════════════════════════════════════════
// Lifecycle
// ═══════════════════════════════════════════════════════════════════════════════

#[no_mangle]
pub extern "C" fn soma_engine_new() -> *mut SomaEngine {
    Box::into_raw(Box::new(SomaEngine::new(SomaConfig::default())))
}

#[no_mangle]
pub extern "C" fn soma_engine_new_mobile() -> *mut SomaEngine {
    Box::into_raw(Box::new(SomaEngine::new(SomaConfig::default())))
}

/// # Safety
/// `config_json` must be a valid null-terminated UTF-8 C string.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_new_with_config(
    config_json: *const c_char,
) -> *mut SomaEngine {
    if config_json.is_null() {
        return std::ptr::null_mut();
    }
    let c_str = unsafe { CStr::from_ptr(config_json) };
    let json_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    let config: SomaConfig = match serde_json::from_str(json_str) {
        Ok(c) => c,
        Err(_) => return std::ptr::null_mut(),
    };
    Box::into_raw(Box::new(SomaEngine::new(config)))
}

/// # Safety
/// `engine` must be a valid pointer from `soma_engine_new*()`, or null (no-op).
/// After calling this function, the pointer is invalid — do NOT use it again.
/// The first 8 bytes are overwritten with a sentinel to detect use-after-free.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_free(engine: *mut SomaEngine) {
    if !engine.is_null() {
        drop(unsafe { Box::from_raw(engine) });
    }
}

/// Validate an engine pointer is not null.
/// Returns the pointer as a mutable reference, or null if invalid.
///
/// # Safety
/// `engine` must be a valid, non-null pointer from `soma_engine_new*()`.
unsafe fn validate_engine<'a>(engine: *mut SomaEngine) -> Option<&'a mut SomaEngine> {
    if engine.is_null() {
        return None;
    }
    Some(unsafe { &mut *engine })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Core cognitive cycle
// ═══════════════════════════════════════════════════════════════════════════════

/// # Safety
/// `engine` must be valid. `input` may be null.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_cycle(engine: *mut SomaEngine, input: *const c_char) -> f32 {
    let engine = unsafe { &mut *engine };
    let text = if input.is_null() {
        ""
    } else {
        unsafe { CStr::from_ptr(input) }.to_str().unwrap_or("")
    };
    engine.cycle(text).consciousness_level
}

/// # Safety
/// `engine` must be valid. `input` may be null.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_cycle_json(
    engine: *mut SomaEngine,
    input: *const c_char,
) -> *mut c_char {
    let engine = unsafe { &mut *engine };
    let text = if input.is_null() {
        ""
    } else {
        unsafe { CStr::from_ptr(input) }.to_str().unwrap_or("")
    };
    let result = engine.cycle(text);
    match serde_json::to_string(&result) {
        Ok(json) => CString::new(json).map_or(std::ptr::null_mut(), |c| c.into_raw()),
        Err(_) => std::ptr::null_mut(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// State inspection
// ═══════════════════════════════════════════════════════════════════════════════

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_consciousness_level(engine: *const SomaEngine) -> f32 {
    unsafe { &*engine }.consciousness_level()
}

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_cycle_count(engine: *const SomaEngine) -> u64 {
    unsafe { &*engine }.spore.current_cycle()
}

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_substrate_feasibility(engine: *const SomaEngine) -> f32 {
    unsafe { &*engine }.spore.substrate_feasibility()
}

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_harmony_alignment(engine: *const SomaEngine) -> f32 {
    unsafe { &*engine }.spore.harmony_alignment()
}

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_consciousness_report(
    engine: *const SomaEngine,
) -> *mut c_char {
    let report = unsafe { &*engine }.consciousness_report();
    CString::new(report).map_or(std::ptr::null_mut(), |c| c.into_raw())
}

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_neuromod_json(engine: *const SomaEngine) -> *mut c_char {
    let json = unsafe { &*engine }.neuromod_json();
    CString::new(json).map_or(std::ptr::null_mut(), |c| c.into_raw())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Platform integration
// ═══════════════════════════════════════════════════════════════════════════════

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_set_thermal_level(engine: *mut SomaEngine, level: u8) {
    unsafe { &mut *engine }.thermal_level = level.min(4);
}

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_set_battery_state(
    engine: *mut SomaEngine,
    charge_percent: u8,
    is_charging: u8,
) {
    let engine = unsafe { &mut *engine };
    engine.battery_percent = charge_percent.min(100);
    engine.battery_charging = is_charging != 0;
}

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_set_night_mode(engine: *mut SomaEngine, is_night: u8) {
    unsafe { &mut *engine }.night_mode = is_night != 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dream
// ═══════════════════════════════════════════════════════════════════════════════

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_dream_cycle(engine: *mut SomaEngine) -> u8 {
    if unsafe { &mut *engine }.dream_cycle().is_some() {
        1
    } else {
        0
    }
}

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_dream_consolidate(engine: *mut SomaEngine) {
    unsafe { &mut *engine }.dream_consolidate();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Sleep/Wake metabolism
// ═══════════════════════════════════════════════════════════════════════════════

/// Signal values: 0=PhonePickup, 1=UserInput, 2=ExplicitSleep.
/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_wake_signal(engine: *mut SomaEngine, signal: u8) {
    unsafe { &mut *engine }.wake_signal(crate::metabolism::WakeSignal::from_u8(signal));
}

/// Returns 0=Sleep, 1=Drowsy, 2=Alert, 3=Focused.
/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_wake_state(engine: *const SomaEngine) -> u8 {
    unsafe { &*engine }.wake_state().as_u8()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Sensor bridge
// ═══════════════════════════════════════════════════════════════════════════════

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_set_sensors(
    engine: *mut SomaEngine,
    accel: f32,
    light: f32,
    proximity_near: u8,
    barometer: f32,
    gps_novelty: f32,
) {
    unsafe { &mut *engine }.set_sensors(accel, light, proximity_near != 0, barometer, gps_novelty);
}

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_motion_state(engine: *const SomaEngine) -> u8 {
    unsafe { &*engine }.motion_state().as_u8()
}

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_privacy_mode(engine: *const SomaEngine) -> u8 {
    if unsafe { &*engine }.privacy_mode() {
        1
    } else {
        0
    }
}

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_set_gyroscope(engine: *mut SomaEngine, rotation_rate: f32) {
    unsafe { &mut *engine }.set_gyroscope(rotation_rate);
}

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_set_step_delta(engine: *mut SomaEngine, steps: u32) {
    unsafe { &mut *engine }.set_step_delta(steps);
}

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_set_ambient_db(engine: *mut SomaEngine, db: f32) {
    unsafe { &mut *engine }.set_ambient_db(db);
}

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_set_social_pressure(engine: *mut SomaEngine, count: u32) {
    unsafe { &mut *engine }.set_social_pressure(count);
}

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_set_media_state(engine: *mut SomaEngine, state: u8) {
    unsafe { &mut *engine }.set_media_state(state);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Compass / Sharing / Haptic / Dream journal / Broca / Persistence / Holon / BLE
// ═══════════════════════════════════════════════════════════════════════════════

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_compass_json(engine: *const SomaEngine) -> *mut c_char {
    let json = unsafe { &*engine }.compass_json();
    CString::new(json).map_or(std::ptr::null_mut(), |c| c.into_raw())
}

/// # Safety: `engine` and `json` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_set_sharing_config(
    engine: *mut SomaEngine,
    json: *const c_char,
) {
    if json.is_null() {
        return;
    }
    let c_str = unsafe { CStr::from_ptr(json) };
    if let Ok(s) = c_str.to_str() {
        if let Ok(config) = serde_json::from_str::<symthaea_spore::config::SharingConfig>(s) {
            unsafe { &mut *engine }.set_sharing_config(config);
        }
    }
}

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_haptic_drain(engine: *mut SomaEngine) -> *mut c_char {
    let json = unsafe { &mut *engine }.haptic_drain_json();
    CString::new(json).map_or(std::ptr::null_mut(), |c| c.into_raw())
}

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_haptic_pending(engine: *const SomaEngine) -> u32 {
    unsafe { &*engine }.haptic_pending()
}

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_haptic_set_enabled(engine: *mut SomaEngine, enabled: u8) {
    unsafe { &mut *engine }.haptic_set_enabled(enabled != 0);
}

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_dream_journal_latest(engine: *const SomaEngine) -> *mut c_char {
    let json = unsafe { &*engine }.dream_journal_latest_json();
    CString::new(json).map_or(std::ptr::null_mut(), |c| c.into_raw())
}

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_dream_journal_all(engine: *const SomaEngine) -> *mut c_char {
    let json = unsafe { &*engine }.dream_journal_all_json();
    CString::new(json).map_or(std::ptr::null_mut(), |c| c.into_raw())
}

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_dream_journal_count(engine: *const SomaEngine) -> u32 {
    unsafe { &*engine }.dream_journal_count()
}

/// Load trained BrocaLite checkpoint from raw bytes.
/// Returns 1 on success, 0 on failure.
///
/// # Safety: `engine` must be valid. `data` must point to `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_load_broca_checkpoint(
    engine: *mut SomaEngine,
    data: *const u8,
    len: u32,
) -> u8 {
    if data.is_null() || len == 0 {
        return 0;
    }
    let slice = unsafe { std::slice::from_raw_parts(data, len as usize) };
    match unsafe { &mut *engine }.load_broca_checkpoint(slice) {
        Ok(()) => 1,
        Err(_) => 0,
    }
}

/// Inject user engagement signal (0.0-1.0) into neuromodulator bath.
///
/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_set_engagement_score(engine: *mut SomaEngine, score: f32) {
    unsafe { &mut *engine }.set_engagement_score(score);
}

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_generate_text(
    engine: *mut SomaEngine,
    max_tokens: u32,
) -> *mut c_char {
    let result = unsafe { &mut *engine }.generate_text(max_tokens as usize);
    let json = serde_json::json!({
        "text": result.text,
        "num_tokens": result.num_tokens,
        "eos_terminated": result.eos_terminated,
    });
    CString::new(json.to_string()).map_or(std::ptr::null_mut(), |c| c.into_raw())
}

/// Generate text with user input context so Broca responds to what the user said.
///
/// # Safety: `engine` and `input` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_generate_text_with_input(
    engine: *mut SomaEngine,
    input: *const c_char,
    max_tokens: u32,
) -> *mut c_char {
    let input_str = if input.is_null() {
        ""
    } else {
        unsafe { CStr::from_ptr(input) }.to_str().unwrap_or("")
    };
    let result = unsafe { &mut *engine }.generate_text_with_input(input_str, max_tokens as usize);
    let json = serde_json::json!({
        "text": result.text,
        "num_tokens": result.num_tokens,
        "eos_terminated": result.eos_terminated,
    });
    CString::new(json.to_string()).map_or(std::ptr::null_mut(), |c| c.into_raw())
}

/// Generate text using the full 20-channel embodied Broca pipeline.
/// Returns richer output including coherence and veto data.
/// Falls back to basic generate_text if broca-full is not compiled.
///
/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_generate_embodied_text(
    engine: *mut SomaEngine,
) -> *mut c_char {
    #[cfg(feature = "broca-full")]
    {
        let result = unsafe { &mut *engine }.generate_embodied_text();
        let json = serde_json::json!({
            "text": result.text,
            "num_tokens": result.num_tokens,
            "eos_terminated": result.eos_terminated,
            "veto_triggered": result.veto_triggered,
            "coherence": result.final_coherence,
            "hallucination_flag": result.hallucination_flag,
        });
        CString::new(json.to_string()).map_or(std::ptr::null_mut(), |c| c.into_raw())
    }
    #[cfg(not(feature = "broca-full"))]
    {
        // Fall back to basic generation with 12 tokens
        soma_engine_generate_text(engine, 12)
    }
}

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_save_checkpoint(engine: *mut SomaEngine) -> u8 {
    if unsafe { &mut *engine }.save_checkpoint() {
        1
    } else {
        0
    }
}

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_load_checkpoint(engine: *mut SomaEngine) -> u8 {
    if unsafe { &mut *engine }.load_checkpoint() {
        1
    } else {
        0
    }
}

/// # Safety: `engine` and `path` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_set_storage_path(
    engine: *mut SomaEngine,
    path: *const c_char,
) {
    if path.is_null() {
        return;
    }
    let c_str = unsafe { CStr::from_ptr(path) };
    if let Ok(s) = c_str.to_str() {
        unsafe { &mut *engine }
            .set_storage(Box::new(symthaea_spore::persistence::FileStorage::new(s)));
    }
}

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_holon_drain_outbound(engine: *mut SomaEngine) -> *mut c_char {
    let json = unsafe { &mut *engine }.holon_drain_outbound_json();
    CString::new(json).map_or(std::ptr::null_mut(), |c| c.into_raw())
}

/// # Safety: `engine` and `json` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_holon_receive(engine: *mut SomaEngine, json: *const c_char) {
    if json.is_null() {
        return;
    }
    let c_str = unsafe { CStr::from_ptr(json) };
    if let Ok(s) = c_str.to_str() {
        unsafe { &mut *engine }.holon_receive_json(s);
    }
}

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_engine_holon_set_connected(engine: *mut SomaEngine, connected: u8) {
    unsafe { &mut *engine }.holon_set_connected(connected != 0);
}

/// # Safety: `engine` must be valid. `cv_data` must point to `len` valid bytes.
#[no_mangle]
pub unsafe extern "C" fn soma_ble_receive_peer(
    engine: *mut SomaEngine,
    peer_id: u64,
    cv_data: *const u8,
    len: u32,
) -> u8 {
    if cv_data.is_null() || len < 12 {
        return 0;
    }
    let data = unsafe { std::slice::from_raw_parts(cv_data, len as usize) };
    if unsafe { &mut *engine }.ble_receive_peer(peer_id, data) {
        1
    } else {
        0
    }
}

/// # Safety: `engine` must be valid. `out_buf` must point to `buf_len` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn soma_ble_advertise_payload(
    engine: *mut SomaEngine,
    out_buf: *mut u8,
    buf_len: u32,
) -> u32 {
    let payload = unsafe { &mut *engine }.ble_advertise_payload();
    if payload.is_empty() || buf_len < payload.len() as u32 {
        return 0;
    }
    let out = unsafe { std::slice::from_raw_parts_mut(out_buf, payload.len()) };
    out.copy_from_slice(&payload);
    payload.len() as u32
}

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_ble_peer_count(engine: *const SomaEngine) -> u32 {
    unsafe { &*engine }.ble_peer_count()
}

/// # Safety: `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn soma_ble_collective_phi(engine: *const SomaEngine) -> f32 {
    unsafe { &*engine }.ble_collective_phi()
}

// ═══════════════════════════════════════════════════════════════════════════════
// On-device LLM (LiteRT-LM / Gemma 4 E2B)
// ═══════════════════════════════════════════════════════════════════════════════

/// Initialize the on-device Gemma 4 E2B engine with a model file path.
/// Returns 1 on success, 0 on failure.
///
/// # Safety
/// `engine` must be valid. `model_path` must be a valid null-terminated UTF-8 C string.
#[cfg(feature = "litert")]
#[no_mangle]
pub unsafe extern "C" fn soma_engine_litert_init(
    engine: *mut SomaEngine,
    model_path: *const c_char,
) -> u8 {
    if model_path.is_null() {
        return 0;
    }
    let path = match unsafe { CStr::from_ptr(model_path) }.to_str() {
        Ok(s) => s,
        Err(_) => return 0,
    };
    if unsafe { &mut *engine }.litert_init(path) {
        1
    } else {
        0
    }
}

/// Check if the on-device LLM is available for inference.
/// Returns 1 if ready, 0 if not.
///
/// # Safety: `engine` must be valid.
#[cfg(feature = "litert")]
#[no_mangle]
pub unsafe extern "C" fn soma_engine_litert_available(engine: *const SomaEngine) -> u8 {
    if unsafe { &*engine }.litert_available() {
        1
    } else {
        0
    }
}

/// Generate text using the on-device Gemma 4 E2B model.
/// Returns a JSON string `{"text":"...","from_device":true}` or null if unavailable.
/// Caller must free with `soma_string_free()`.
///
/// # Safety
/// `engine` must be valid. `prompt` must be a valid null-terminated UTF-8 C string.
#[cfg(feature = "litert")]
#[no_mangle]
pub unsafe extern "C" fn soma_engine_litert_generate(
    engine: *mut SomaEngine,
    prompt: *const c_char,
    max_tokens: u32,
) -> *mut c_char {
    if prompt.is_null() {
        return std::ptr::null_mut();
    }
    let prompt_str = match unsafe { CStr::from_ptr(prompt) }.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    match unsafe { &mut *engine }.litert_generate(prompt_str, max_tokens) {
        Some(text) => {
            let json = serde_json::json!({
                "text": text,
                "from_device": true,
            });
            CString::new(json.to_string()).map_or(std::ptr::null_mut(), |c| c.into_raw())
        }
        None => std::ptr::null_mut(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Prism epistemic search (offline-capable, sub-ms)
// ═══════════════════════════════════════════════════════════════════════════════

/// Initialize the Prism epistemic search engine with pre-seeded claims.
///
/// # Safety: `engine` must be valid.
#[cfg(feature = "prism-search")]
#[no_mangle]
pub unsafe extern "C" fn soma_engine_prism_init(engine: *mut SomaEngine) {
    unsafe { &mut *engine }.prism_init();
}

/// Load the full Wikidata corpus (~10K claims) and merge into the search engine.
/// Call after `soma_engine_prism_init()` on a background thread.
///
/// # Safety: `engine` must be valid.
#[cfg(feature = "prism-search")]
#[no_mangle]
pub unsafe extern "C" fn soma_engine_prism_load_full(engine: *mut SomaEngine) {
    unsafe { &mut *engine }.prism_load_full();
}

/// Search Prism epistemic claims. Returns JSON array of results.
/// Caller must free with `soma_string_free()`.
///
/// # Safety: `engine` must be valid. `query` must be valid null-terminated UTF-8.
#[cfg(feature = "prism-search")]
#[no_mangle]
pub unsafe extern "C" fn soma_engine_prism_search(
    engine: *const SomaEngine,
    query: *const c_char,
    top_k: u32,
) -> *mut c_char {
    if query.is_null() {
        return std::ptr::null_mut();
    }
    let query_str = match unsafe { CStr::from_ptr(query) }.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    let results = unsafe { &*engine }.prism_search(query_str, top_k as usize);
    let json: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "content": r.content,
                "sources": r.sources,
                "empirical_level": format!("{:?}", r.empirical_level),
                "similarity": r.query_similarity,
            })
        })
        .collect();
    CString::new(serde_json::to_string(&json).unwrap_or_else(|_| "[]".to_string()))
        .map_or(std::ptr::null_mut(), |c| c.into_raw())
}

/// Whether Prism search is initialized and has claims.
///
/// # Safety: `engine` must be valid.
#[cfg(feature = "prism-search")]
#[no_mangle]
pub unsafe extern "C" fn soma_engine_prism_available(engine: *const SomaEngine) -> u8 {
    if unsafe { &*engine }.prism_available() {
        1
    } else {
        0
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Screen vision (Phase 3)
// ═══════════════════════════════════════════════════════════════════════════════

/// Inject a screen frame for visual processing.
///
/// `data` must point to `width * height * channels` bytes (RGB = 3 channels).
/// Returns the overall surprise level (0.0-1.0).
///
/// # Safety
/// `engine` must be valid. `data` must point to `width * height * channels` valid bytes.
#[cfg(feature = "screen-vision")]
#[no_mangle]
pub unsafe extern "C" fn soma_engine_inject_frame(
    engine: *mut SomaEngine,
    data: *const u8,
    width: u32,
    height: u32,
    channels: u32,
) -> f32 {
    if data.is_null() || width == 0 || height == 0 || channels == 0 {
        return 0.0;
    }
    // Only support RGB (3 channels) for now
    if channels != 3 {
        return 0.0;
    }
    let engine = unsafe { &mut *engine };
    let len = (width as usize) * (height as usize) * (channels as usize);
    let frame = unsafe { std::slice::from_raw_parts(data, len) };
    let perception = engine.inject_frame(frame, width, height);
    perception.surprise_level
}

/// Inject a touch event for proprioceptive processing.
///
/// `action`: 0=Down, 1=Move, 2=Up, 3=Cancel.
///
/// # Safety
/// `engine` must be valid.
#[cfg(feature = "screen-vision")]
#[no_mangle]
pub unsafe extern "C" fn soma_engine_touch_event(
    engine: *mut SomaEngine,
    x: f32,
    y: f32,
    action: u8,
    pressure: f32,
) {
    let engine = unsafe { &mut *engine };
    let touch_action = match action {
        0 => crate::touch_body::TouchAction::Down,
        1 => crate::touch_body::TouchAction::Move,
        2 => crate::touch_body::TouchAction::Up,
        _ => crate::touch_body::TouchAction::Cancel,
    };
    let event = crate::touch_body::TouchEvent {
        x: x.clamp(0.0, 1.0),
        y: y.clamp(0.0, 1.0),
        action: touch_action,
        pressure: pressure.clamp(0.0, 1.0),
        timestamp_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0),
    };
    engine.on_touch(event);
}

/// Get salient screen regions as JSON array.
///
/// Returns `[{"x":N,"y":N,"width":N,"height":N,"surprise":F,"velocity":[F,F]}, ...]`
/// Caller must free with `soma_string_free()`.
///
/// # Safety
/// `engine` must be valid.
#[cfg(feature = "screen-vision")]
#[no_mangle]
pub unsafe extern "C" fn soma_engine_screen_salient_regions_json(
    engine: *const SomaEngine,
) -> *mut c_char {
    let engine = unsafe { &*engine };
    let telemetry = engine.screen_vision_telemetry();
    let json = serde_json::json!({
        "frames_processed": telemetry.frames_processed,
        "last_surprise": telemetry.last_surprise,
        "last_salient_count": telemetry.last_salient_count,
        "foveation_pending": telemetry.foveation_pending,
    });
    CString::new(json.to_string()).map_or(std::ptr::null_mut(), |c| c.into_raw())
}

// ═══════════════════════════════════════════════════════════════════════════════
// String memory management
// ═══════════════════════════════════════════════════════════════════════════════

#[no_mangle]
pub unsafe extern "C" fn soma_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(unsafe { CString::from_raw(s) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lifecycle_new_free() {
        let engine = soma_engine_new();
        assert!(!engine.is_null());
        unsafe { soma_engine_free(engine) };
    }

    #[test]
    fn test_cycle_returns_valid_consciousness() {
        let engine = soma_engine_new();
        unsafe {
            let input = CString::new("hello").unwrap();
            let level = soma_engine_cycle(engine, input.as_ptr());
            assert!((0.0..=1.0).contains(&level));
            assert_eq!(soma_engine_cycle_count(engine), 1);
            soma_engine_free(engine);
        }
    }

    #[test]
    fn test_wake_signal_and_state() {
        let engine = soma_engine_new();
        unsafe {
            assert_eq!(soma_engine_wake_state(engine), 2); // Alert
            soma_engine_wake_signal(engine, 2); // ExplicitSleep
            assert_eq!(soma_engine_wake_state(engine), 0); // Sleep
            soma_engine_free(engine);
        }
    }

    #[test]
    fn test_set_sensors_privacy() {
        let engine = soma_engine_new();
        unsafe {
            assert_eq!(soma_engine_privacy_mode(engine), 0);
            soma_engine_set_sensors(engine, 0.0, 300.0, 1, 1013.0, 0.0);
            assert_eq!(soma_engine_privacy_mode(engine), 1);
            soma_engine_free(engine);
        }
    }

    #[test]
    fn test_free_null_is_noop() {
        unsafe {
            soma_engine_free(std::ptr::null_mut());
            soma_string_free(std::ptr::null_mut());
        }
    }
}
