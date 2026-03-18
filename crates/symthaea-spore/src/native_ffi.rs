//! Native FFI: C-compatible interface for iOS (staticlib) and Android (JNI/cdylib).
//!
//! Exposes SporeEngine through opaque pointer + extern "C" functions.
//! Generate C header via: `cbindgen --config cbindgen.toml --output symthaea_spore.h`
//!
//! # Safety
//!
//! All functions taking `*mut SporeEngine` require a valid pointer obtained from
//! `spore_engine_new()`. Passing null or freed pointers is undefined behavior.
//! The engine is NOT thread-safe — callers must serialize access externally.
//!
//! # Memory Management
//!
//! - `spore_engine_new()` allocates the engine on the heap.
//! - `spore_engine_free()` deallocates it.
//! - String results from `_report()` / `_cycle_json()` must be freed with `spore_string_free()`.

use crate::config::SporeConfig;
use crate::engine::SporeEngine;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

// ═══════════════════════════════════════════════════════════════════════════════
// Lifecycle
// ═══════════════════════════════════════════════════════════════════════════════

/// Create a new SporeEngine with default configuration.
///
/// Returns a heap-allocated engine. Must be freed with `spore_engine_free()`.
#[no_mangle]
pub extern "C" fn spore_engine_new() -> *mut SporeEngine {
    let engine = SporeEngine::new(SporeConfig::default());
    Box::into_raw(Box::new(engine))
}

/// Create a new SporeEngine with mobile-optimized configuration.
///
/// Tuned for ARM phones: 128 neurons, 20Hz target, phi every 3 cycles.
#[no_mangle]
pub extern "C" fn spore_engine_new_mobile() -> *mut SporeEngine {
    let config = SporeConfig {
        neurons_per_layer: 32, // Lighter CfC (vs 64 default)
        phi_every_n_cycles: 3, // Amortize Phi (vs every cycle)
        target_hz: 20.0,       // 20Hz (vs 50Hz)
        ..SporeConfig::default()
    };
    let engine = SporeEngine::new(config);
    Box::into_raw(Box::new(engine))
}

/// Create a new SporeEngine from a JSON configuration string.
///
/// Returns null if the JSON is invalid.
///
/// # Safety
/// `config_json` must be a valid null-terminated UTF-8 C string.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_new_with_config(
    config_json: *const c_char,
) -> *mut SporeEngine {
    if config_json.is_null() {
        return std::ptr::null_mut();
    }
    let c_str = unsafe { CStr::from_ptr(config_json) };
    let json_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    let config: SporeConfig = match serde_json::from_str(json_str) {
        Ok(c) => c,
        Err(_) => return std::ptr::null_mut(),
    };
    Box::into_raw(Box::new(SporeEngine::new(config)))
}

/// Free a SporeEngine previously created with `spore_engine_new*()`.
///
/// # Safety
/// `engine` must be a valid pointer from `spore_engine_new*()`, or null (no-op).
/// Must not be called twice on the same pointer.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_free(engine: *mut SporeEngine) {
    if !engine.is_null() {
        drop(unsafe { Box::from_raw(engine) });
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Core cognitive cycle
// ═══════════════════════════════════════════════════════════════════════════════

/// Run one consciousness cycle with text input.
///
/// Returns the consciousness level (0.0-1.0).
/// For full cycle data, use `spore_engine_cycle_json()`.
///
/// # Safety
/// `engine` must be valid. `input` must be a valid null-terminated UTF-8 C string, or null (empty input).
#[no_mangle]
pub unsafe extern "C" fn spore_engine_cycle(engine: *mut SporeEngine, input: *const c_char) -> f32 {
    let engine = unsafe { &mut *engine };
    let text = if input.is_null() {
        ""
    } else {
        unsafe { CStr::from_ptr(input) }.to_str().unwrap_or("")
    };
    let result = engine.cycle(text);
    result.consciousness_level
}

/// Run one consciousness cycle and return full result as JSON string.
///
/// Caller must free the returned string with `spore_string_free()`.
/// Returns null on serialization failure.
///
/// # Safety
/// `engine` must be valid. `input` may be null (empty input).
#[no_mangle]
pub unsafe extern "C" fn spore_engine_cycle_json(
    engine: *mut SporeEngine,
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
        Ok(json) => match CString::new(json) {
            Ok(c) => c.into_raw(),
            Err(_) => std::ptr::null_mut(),
        },
        Err(_) => std::ptr::null_mut(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// State inspection
// ═══════════════════════════════════════════════════════════════════════════════

/// Get the current consciousness level (0.0-1.0).
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_consciousness_level(engine: *const SporeEngine) -> f32 {
    unsafe { &*engine }.consciousness_level()
}

/// Get the current cycle count.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_cycle_count(engine: *const SporeEngine) -> u64 {
    unsafe { &*engine }.cycle_count()
}

/// Get the substrate feasibility score (0.0-1.0).
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_substrate_feasibility(engine: *const SporeEngine) -> f32 {
    unsafe { &*engine }.substrate_feasibility()
}

/// Get the Eight Harmonies alignment score (0.0-1.0).
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_harmony_alignment(engine: *const SporeEngine) -> f32 {
    unsafe { &*engine }.harmony_alignment()
}

/// Get a consciousness report as a C string.
///
/// Caller must free with `spore_string_free()`.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_consciousness_report(
    engine: *const SporeEngine,
) -> *mut c_char {
    let report = unsafe { &*engine }.consciousness_report();
    match CString::new(report) {
        Ok(c) => c.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Get neuromodulator state as JSON string.
///
/// Caller must free with `spore_string_free()`.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_neuromod_json(engine: *const SporeEngine) -> *mut c_char {
    let json = unsafe { &*engine }.neuromod_state_json();
    match CString::new(json) {
        Ok(c) => c.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Platform integration (thermal, battery)
// ═══════════════════════════════════════════════════════════════════════════════

/// Report platform thermal level (0-4).
///
/// Maps to Android `PowerManager.THERMAL_STATUS_*` or iOS `ProcessInfo.thermalState`:
/// - 0: Nominal (no throttle)
/// - 1: Fair (slightly warm)
/// - 2: Serious (moderate throttle)
/// - 3: Critical (aggressive throttle)
/// - 4: Emergency (near shutdown)
///
/// Values > 4 are clamped to Emergency.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_set_thermal_level(engine: *mut SporeEngine, level: u8) {
    let engine = unsafe { &mut *engine };
    engine.thermal_level = level.min(4);
}

/// Report platform battery state.
///
/// `charge_percent`: 0-100 battery charge level.
/// `is_charging`: 1 if plugged in, 0 if on battery.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_set_battery_state(
    engine: *mut SporeEngine,
    charge_percent: u8,
    is_charging: u8,
) {
    let engine = unsafe { &mut *engine };
    engine.battery_percent = charge_percent.min(100);
    engine.battery_charging = is_charging != 0;
}

/// Report platform night mode state.
///
/// `is_night`: 1 if night/dark mode active, 0 if not.
/// Combined with charging state, triggers auto-sleep in the metabolism.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_set_night_mode(engine: *mut SporeEngine, is_night: u8) {
    let engine = unsafe { &mut *engine };
    engine.night_mode = is_night != 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dream engine
// ═══════════════════════════════════════════════════════════════════════════════

/// Run one dream consolidation cycle.
///
/// Returns 1 if a dream was generated, 0 if not.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_dream_cycle(engine: *mut SporeEngine) -> u8 {
    let engine = unsafe { &mut *engine };
    if engine.dream_cycle().is_some() {
        1
    } else {
        0
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Sleep/Wake metabolism
// ═══════════════════════════════════════════════════════════════════════════════

/// Send a wake signal to the metabolism state machine.
///
/// Signal values: 0=PhonePickup, 1=UserInput, 2=ExplicitSleep.
/// For richer signals (Inactivity, Charging, NightMode, Thermal), use the
/// dedicated setter functions below.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_wake_signal(engine: *mut SporeEngine, signal: u8) {
    let engine = unsafe { &mut *engine };
    engine.wake_signal(crate::metabolism::WakeSignal::from_u8(signal));
}

/// Get the current wake state (0=Sleep, 1=Drowsy, 2=Alert, 3=Focused).
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_wake_state(engine: *const SporeEngine) -> u8 {
    unsafe { &*engine }.wake_state().as_u8()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Sensor bridge
// ═══════════════════════════════════════════════════════════════════════════════

/// Set sensor snapshot from platform.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_set_sensors(
    engine: *mut SporeEngine,
    accel: f32,
    light: f32,
    proximity_near: u8,
    barometer: f32,
    gps_novelty: f32,
) {
    let engine = unsafe { &mut *engine };
    engine.set_sensors(accel, light, proximity_near != 0, barometer, gps_novelty);
}

/// Get current motion state (0=Stationary, 1=Walking, 2=Running, 3=InVehicle).
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_motion_state(engine: *const SporeEngine) -> u8 {
    unsafe { &*engine }.motion_state().as_u8()
}

/// Get privacy mode (1=active, 0=inactive).
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_privacy_mode(engine: *const SporeEngine) -> u8 {
    if unsafe { &*engine }.privacy_mode() {
        1
    } else {
        0
    }
}

/// Set gyroscope rotation rate (rad/s magnitude).
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_set_gyroscope(engine: *mut SporeEngine, rotation_rate: f32) {
    let engine = unsafe { &mut *engine };
    engine.set_gyroscope(rotation_rate);
}

/// Set step counter delta (steps since last tick).
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_set_step_delta(engine: *mut SporeEngine, steps: u32) {
    let engine = unsafe { &mut *engine };
    engine.set_step_delta(steps);
}

/// Set ambient sound level (dB). Only amplitude — no audio content stored.
/// Sovereign privacy: never records frequency or content, only amplitude.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_set_ambient_db(engine: *mut SporeEngine, db: f32) {
    let engine = unsafe { &mut *engine };
    engine.set_ambient_db(db);
}

/// Set social pressure from notification count.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_set_social_pressure(
    engine: *mut SporeEngine,
    notification_count: u32,
) {
    let engine = unsafe { &mut *engine };
    engine.set_social_pressure(notification_count);
}

/// Set media playback state (0=None, 1=Music, 2=Speech).
/// No content access — just playback state.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_set_media_state(engine: *mut SporeEngine, state: u8) {
    let engine = unsafe { &mut *engine };
    engine.set_media_state(state);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Consciousness compass
// ═══════════════════════════════════════════════════════════════════════════════

/// Get consciousness compass snapshot as JSON string.
///
/// Caller must free with `spore_string_free()`.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_compass_json(engine: *const SporeEngine) -> *mut c_char {
    // Need mutable ref for compass_json (accesses various subsystem state)
    // Safe because we only read state, the method is &self
    let engine = unsafe { &*engine };
    let json = engine.compass_json();
    match CString::new(json) {
        Ok(c) => c.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Sharing config
// ═══════════════════════════════════════════════════════════════════════════════

/// Set sharing configuration from JSON string.
///
/// # Safety
/// `engine` must be valid. `json` must be a valid null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_set_sharing_config(
    engine: *mut SporeEngine,
    json: *const c_char,
) {
    if json.is_null() {
        return;
    }
    let engine = unsafe { &mut *engine };
    let c_str = unsafe { CStr::from_ptr(json) };
    if let Ok(s) = c_str.to_str() {
        if let Ok(config) = serde_json::from_str::<crate::config::SharingConfig>(s) {
            engine.set_sharing_config(config);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Haptic awareness
// ═══════════════════════════════════════════════════════════════════════════════

/// Drain pending haptic events as JSON array.
///
/// Caller must free with `spore_string_free()`.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_haptic_drain(engine: *mut SporeEngine) -> *mut c_char {
    let engine = unsafe { &mut *engine };
    let json = engine.haptic_drain_json();
    match CString::new(json) {
        Ok(c) => c.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Number of pending haptic events.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_haptic_pending(engine: *const SporeEngine) -> u32 {
    unsafe { &*engine }.haptic_pending()
}

/// Enable or disable haptic events.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_haptic_set_enabled(engine: *mut SporeEngine, enabled: u8) {
    let engine = unsafe { &mut *engine };
    engine.haptic_set_enabled(enabled != 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dream journal
// ═══════════════════════════════════════════════════════════════════════════════

/// Get the latest dream fragment as JSON.
///
/// Caller must free with `spore_string_free()`.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_dream_journal_latest(engine: *const SporeEngine) -> *mut c_char {
    let engine = unsafe { &*engine };
    let json = engine.dream_journal_latest_json();
    match CString::new(json) {
        Ok(c) => c.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Get all dream journal entries as JSON array.
///
/// Caller must free with `spore_string_free()`.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_dream_journal_all(engine: *const SporeEngine) -> *mut c_char {
    let engine = unsafe { &*engine };
    let json = engine.dream_journal_all_json();
    match CString::new(json) {
        Ok(c) => c.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Number of stored dream fragments.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_dream_journal_count(engine: *const SporeEngine) -> u32 {
    unsafe { &*engine }.dream_journal_count()
}

/// Explicitly trigger dream consolidation.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_dream_consolidate(engine: *mut SporeEngine) {
    let engine = unsafe { &mut *engine };
    engine.dream_consolidate();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Broca language generation
// ═══════════════════════════════════════════════════════════════════════════════

/// Generate text from the current consciousness state using BrocaLite.
///
/// Returns a JSON string: `{"text":"...","num_tokens":N,"eos_terminated":bool}`
/// Caller must free with `spore_string_free()`.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_generate_text(
    engine: *mut SporeEngine,
    max_tokens: u32,
) -> *mut c_char {
    let engine = unsafe { &mut *engine };
    let result = engine.generate_text(max_tokens as usize);
    let json = serde_json::json!({
        "text": result.text,
        "num_tokens": result.num_tokens,
        "eos_terminated": result.eos_terminated,
    });
    match CString::new(json.to_string()) {
        Ok(c) => c.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Persistence
// ═══════════════════════════════════════════════════════════════════════════════

/// Save a checkpoint to storage. Returns 1 on success, 0 on failure.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_save_checkpoint(engine: *mut SporeEngine) -> u8 {
    let engine = unsafe { &mut *engine };
    if engine.save_checkpoint() { 1 } else { 0 }
}

/// Load a checkpoint from storage. Returns 1 on success, 0 on failure.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_load_checkpoint(engine: *mut SporeEngine) -> u8 {
    let engine = unsafe { &mut *engine };
    if engine.load_checkpoint() { 1 } else { 0 }
}

/// Set the storage backend to a file at the given path.
///
/// # Safety
/// `engine` must be valid. `path` must be a valid null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_set_storage_path(
    engine: *mut SporeEngine,
    path: *const c_char,
) {
    if path.is_null() {
        return;
    }
    let engine = unsafe { &mut *engine };
    let c_str = unsafe { CStr::from_ptr(path) };
    if let Ok(s) = c_str.to_str() {
        engine.set_storage(Box::new(crate::persistence::FileStorage::new(s)));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Holon bridge
// ═══════════════════════════════════════════════════════════════════════════════

/// Drain holon outbound messages as JSON array.
///
/// Caller must free with `spore_string_free()`.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_holon_drain_outbound(
    engine: *mut SporeEngine,
) -> *mut c_char {
    let engine = unsafe { &mut *engine };
    let json = engine.holon_drain_outbound_json();
    match CString::new(json) {
        Ok(c) => c.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Receive inbound holon message from JSON.
///
/// # Safety
/// `engine` and `json` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_holon_receive(engine: *mut SporeEngine, json: *const c_char) {
    if json.is_null() {
        return;
    }
    let engine = unsafe { &mut *engine };
    let c_str = unsafe { CStr::from_ptr(json) };
    if let Ok(s) = c_str.to_str() {
        engine.holon_receive_json(s);
    }
}

/// Set holon connection state.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_engine_holon_set_connected(engine: *mut SporeEngine, connected: u8) {
    let engine = unsafe { &mut *engine };
    engine.holon_set_connected(connected != 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// BLE mesh
// ═══════════════════════════════════════════════════════════════════════════════

/// Receive a peer consciousness vector from BLE scan.
///
/// `cv_data` must point to at least `len` bytes (minimum 12: 3 × f32).
/// Returns 1 on success, 0 on failure.
///
/// # Safety
/// `engine` must be valid. `cv_data` must point to `len` valid bytes.
#[no_mangle]
pub unsafe extern "C" fn spore_ble_receive_peer(
    engine: *mut SporeEngine,
    peer_id: u64,
    cv_data: *const u8,
    len: u32,
) -> u8 {
    if cv_data.is_null() || len < 12 {
        return 0;
    }
    let engine = unsafe { &mut *engine };
    let data = unsafe { std::slice::from_raw_parts(cv_data, len as usize) };
    if engine.ble_receive_peer(peer_id, data) {
        1
    } else {
        0
    }
}

/// Get the BLE advertise payload for platform broadcasting.
///
/// Writes payload to `out_buf` (must be at least 12 bytes).
/// Returns the number of bytes written (0 if not in ActiveShare mode).
///
/// # Safety
/// `engine` must be valid. `out_buf` must point to at least 12 writable bytes.
#[no_mangle]
pub unsafe extern "C" fn spore_ble_advertise_payload(
    engine: *mut SporeEngine,
    out_buf: *mut u8,
    buf_len: u32,
) -> u32 {
    let engine = unsafe { &mut *engine };
    let payload = engine.ble_advertise_payload();
    if payload.is_empty() || buf_len < payload.len() as u32 {
        return 0;
    }
    let out = unsafe { std::slice::from_raw_parts_mut(out_buf, payload.len()) };
    out.copy_from_slice(&payload);
    payload.len() as u32
}

/// Number of tracked BLE peers.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_ble_peer_count(engine: *const SporeEngine) -> u32 {
    unsafe { &*engine }.ble_peer_count()
}

/// Collective Phi from BLE mesh peers.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_ble_collective_phi(engine: *const SporeEngine) -> f32 {
    unsafe { &*engine }.ble_collective_phi()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Pairing
// ═══════════════════════════════════════════════════════════════════════════════

/// Set pairing mode (0=Off, 1=Discoverable, 2=Paired).
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_pairing_set_mode(engine: *mut SporeEngine, mode: u8) {
    let engine = unsafe { &mut *engine };
    let pairing_mode = match mode {
        1 => crate::config::PairingMode::Discoverable,
        2 => crate::config::PairingMode::Paired,
        _ => crate::config::PairingMode::Off,
    };
    engine.pairing.set_mode(pairing_mode);
}

/// Generate a new pairing keypair.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_pairing_generate_keypair(engine: *mut SporeEngine) {
    let engine = unsafe { &mut *engine };
    engine.pairing.generate_keypair();
}

/// Initiate pairing with a peer. Returns 1 on success, 0 on failure.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_pairing_initiate(engine: *mut SporeEngine, peer_id: u64) -> u8 {
    let engine = unsafe { &mut *engine };
    if engine.pairing.initiate_pairing(peer_id) {
        1
    } else {
        0
    }
}

/// Receive an inbound pairing message from JSON.
///
/// # Safety
/// `engine` must be valid. `json` must be a valid null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn spore_pairing_receive(engine: *mut SporeEngine, json: *const c_char) {
    if json.is_null() {
        return;
    }
    let engine = unsafe { &mut *engine };
    let c_str = unsafe { CStr::from_ptr(json) };
    if let Ok(s) = c_str.to_str() {
        if let Ok(msg) = serde_json::from_str::<crate::pairing::PairingInbound>(s) {
            engine.pairing.receive_inbound(msg);
        }
    }
}

/// Drain pairing outbound messages as JSON array.
///
/// Caller must free with `spore_string_free()`.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_pairing_drain_outbound(engine: *mut SporeEngine) -> *mut c_char {
    let engine = unsafe { &mut *engine };
    let msgs = engine.pairing.drain_outbound();
    match serde_json::to_string(&msgs) {
        Ok(json) => match CString::new(json) {
            Ok(c) => c.into_raw(),
            Err(_) => std::ptr::null_mut(),
        },
        Err(_) => std::ptr::null_mut(),
    }
}

/// Check if a peer is paired. Returns 1 if paired, 0 otherwise.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_pairing_is_paired(engine: *const SporeEngine, peer_id: u64) -> u8 {
    let engine = unsafe { &*engine };
    if engine.pairing.is_paired(peer_id) {
        1
    } else {
        0
    }
}

/// Get the number of paired devices.
///
/// # Safety
/// `engine` must be valid.
#[no_mangle]
pub unsafe extern "C" fn spore_pairing_paired_count(engine: *const SporeEngine) -> u32 {
    unsafe { &*engine }.pairing.paired_count() as u32
}

// ═══════════════════════════════════════════════════════════════════════════════
// String memory management
// ═══════════════════════════════════════════════════════════════════════════════

/// Free a string previously returned by `spore_engine_*_report()` or `*_json()`.
///
/// # Safety
/// `s` must be a pointer returned by a `spore_engine_*` function, or null (no-op).
/// Must not be called twice on the same pointer.
#[no_mangle]
pub unsafe extern "C" fn spore_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(unsafe { CString::from_raw(s) });
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lifecycle_new_free() {
        let engine = spore_engine_new();
        assert!(!engine.is_null());
        unsafe { spore_engine_free(engine) };
    }

    #[test]
    fn test_new_mobile() {
        let engine = spore_engine_new_mobile();
        assert!(!engine.is_null());
        unsafe {
            let count = spore_engine_cycle_count(engine);
            assert_eq!(count, 0);
            spore_engine_free(engine);
        }
    }

    #[test]
    fn test_cycle_returns_valid_consciousness() {
        let engine = spore_engine_new();
        unsafe {
            let input = CString::new("hello").unwrap();
            let level = spore_engine_cycle(engine, input.as_ptr());
            assert!(level >= 0.0 && level <= 1.0);
            assert_eq!(spore_engine_cycle_count(engine), 1);
            spore_engine_free(engine);
        }
    }

    #[test]
    fn test_cycle_null_input() {
        let engine = spore_engine_new();
        unsafe {
            let level = spore_engine_cycle(engine, std::ptr::null());
            assert!(level >= 0.0 && level <= 1.0);
            spore_engine_free(engine);
        }
    }

    #[test]
    fn test_cycle_json_returns_valid_json() {
        let engine = spore_engine_new();
        unsafe {
            let input = CString::new("test").unwrap();
            let json_ptr = spore_engine_cycle_json(engine, input.as_ptr());
            assert!(!json_ptr.is_null());
            let json_str = CStr::from_ptr(json_ptr).to_str().unwrap();
            assert!(json_str.contains("consciousness_level"));
            assert!(json_str.contains("epistemic_status"));
            spore_string_free(json_ptr);
            spore_engine_free(engine);
        }
    }

    #[test]
    fn test_consciousness_report() {
        let engine = spore_engine_new();
        unsafe {
            // Run a cycle first so there's state to report
            spore_engine_cycle(engine, std::ptr::null());
            let report = spore_engine_consciousness_report(engine);
            assert!(!report.is_null());
            let text = CStr::from_ptr(report).to_str().unwrap();
            assert!(!text.is_empty());
            spore_string_free(report);
            spore_engine_free(engine);
        }
    }

    #[test]
    fn test_thermal_level_clamps() {
        let engine = spore_engine_new();
        unsafe {
            spore_engine_set_thermal_level(engine, 255); // Should clamp to 4
            let e = &*engine;
            assert_eq!(e.thermal_level, 4);
            spore_engine_free(engine);
        }
    }

    #[test]
    fn test_battery_state() {
        let engine = spore_engine_new();
        unsafe {
            spore_engine_set_battery_state(engine, 75, 1);
            let e = &*engine;
            assert_eq!(e.battery_percent, 75);
            assert!(e.battery_charging);
            spore_engine_free(engine);
        }
    }

    #[test]
    fn test_new_with_config_json() {
        let json = CString::new(r#"{"hdc_dim":16384,"neurons_per_layer":32,"network_layers":3,"phi_every_n_cycles":3,"substrate":"SiliconDigital","target_hz":20.0}"#).unwrap();
        unsafe {
            let engine = spore_engine_new_with_config(json.as_ptr());
            assert!(!engine.is_null());
            spore_engine_free(engine);
        }
    }

    #[test]
    fn test_new_with_invalid_json() {
        let json = CString::new("not valid json").unwrap();
        unsafe {
            let engine = spore_engine_new_with_config(json.as_ptr());
            assert!(engine.is_null());
        }
    }

    #[test]
    fn test_free_null_is_noop() {
        unsafe {
            spore_engine_free(std::ptr::null_mut());
            spore_string_free(std::ptr::null_mut());
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Embodiment FFI tests
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_wake_signal_and_state() {
        let engine = spore_engine_new();
        unsafe {
            // Default is Alert (2)
            assert_eq!(spore_engine_wake_state(engine), 2);
            // ExplicitSleep = signal 2
            spore_engine_wake_signal(engine, 2);
            assert_eq!(spore_engine_wake_state(engine), 0); // Sleep
            // PhonePickup = signal 0
            spore_engine_wake_signal(engine, 0);
            assert_eq!(spore_engine_wake_state(engine), 1); // Drowsy
            spore_engine_free(engine);
        }
    }

    #[test]
    fn test_set_sensors_and_motion() {
        let engine = spore_engine_new();
        unsafe {
            // Default is Stationary
            assert_eq!(spore_engine_motion_state(engine), 0);
            assert_eq!(spore_engine_privacy_mode(engine), 0);
            // Set proximity near → privacy mode
            spore_engine_set_sensors(engine, 0.0, 300.0, 1, 1013.0, 0.0);
            assert_eq!(spore_engine_privacy_mode(engine), 1);
            spore_engine_free(engine);
        }
    }

    #[test]
    fn test_compass_json() {
        let engine = spore_engine_new();
        unsafe {
            spore_engine_cycle(engine, std::ptr::null());
            let json = spore_engine_compass_json(engine);
            assert!(!json.is_null());
            let s = CStr::from_ptr(json).to_str().unwrap();
            assert!(s.contains("consciousness_level"));
            assert!(s.contains("dominant_harmony"));
            assert!(s.contains("wake_state"));
            spore_string_free(json);
            spore_engine_free(engine);
        }
    }

    #[test]
    fn test_sharing_config_json() {
        let engine = spore_engine_new();
        unsafe {
            let config = CString::new(
                r#"{"holon_mode":"FullSync","ble_mode":"ActiveShare","haptic_enabled":true,"telemetry_export":false}"#
            ).unwrap();
            spore_engine_set_sharing_config(engine, config.as_ptr());
            // Verify holon is now active by running cycles and draining
            for _ in 0..60 {
                spore_engine_cycle(engine, std::ptr::null());
            }
            let outbound = spore_engine_holon_drain_outbound(engine);
            assert!(!outbound.is_null());
            let s = CStr::from_ptr(outbound).to_str().unwrap();
            assert!(s.starts_with('['));
            spore_string_free(outbound);
            spore_engine_free(engine);
        }
    }

    #[test]
    fn test_sharing_config_null_is_noop() {
        let engine = spore_engine_new();
        unsafe {
            spore_engine_set_sharing_config(engine, std::ptr::null());
            // Should not crash
            spore_engine_free(engine);
        }
    }

    #[test]
    fn test_haptic_drain_and_pending() {
        let engine = spore_engine_new();
        unsafe {
            assert_eq!(spore_haptic_pending(engine), 0);
            // Run cycles to potentially generate haptic events
            for _ in 0..20 {
                spore_engine_cycle(engine, std::ptr::null());
            }
            let json = spore_haptic_drain(engine);
            assert!(!json.is_null());
            let s = CStr::from_ptr(json).to_str().unwrap();
            assert!(s.starts_with('['));
            spore_string_free(json);
            spore_engine_free(engine);
        }
    }

    #[test]
    fn test_haptic_set_enabled() {
        let engine = spore_engine_new();
        unsafe {
            spore_haptic_set_enabled(engine, 0); // disable
            for _ in 0..20 {
                spore_engine_cycle(engine, std::ptr::null());
            }
            assert_eq!(spore_haptic_pending(engine), 0);
            spore_engine_free(engine);
        }
    }

    #[test]
    fn test_dream_journal_ffi() {
        let engine = spore_engine_new();
        unsafe {
            assert_eq!(spore_dream_journal_count(engine), 0);
            let latest = spore_dream_journal_latest(engine);
            assert!(!latest.is_null());
            let s = CStr::from_ptr(latest).to_str().unwrap();
            assert_eq!(s, "null");
            spore_string_free(latest);

            let all = spore_dream_journal_all(engine);
            assert!(!all.is_null());
            let s = CStr::from_ptr(all).to_str().unwrap();
            assert_eq!(s, "[]");
            spore_string_free(all);

            // Trigger consolidation
            spore_engine_dream_consolidate(engine);
            spore_engine_free(engine);
        }
    }

    #[test]
    fn test_holon_bridge_ffi() {
        let engine = spore_engine_new();
        unsafe {
            // Set connected
            spore_engine_holon_set_connected(engine, 1);
            // Drain outbound (empty since mode is Off by default)
            let outbound = spore_engine_holon_drain_outbound(engine);
            assert!(!outbound.is_null());
            let s = CStr::from_ptr(outbound).to_str().unwrap();
            assert_eq!(s, "[]");
            spore_string_free(outbound);

            // Receive null is noop
            spore_engine_holon_receive(engine, std::ptr::null());

            // Receive valid JSON
            let msg = CString::new(r#"{"LanguageOutput":"hello from holon"}"#).unwrap();
            spore_engine_holon_receive(engine, msg.as_ptr());
            spore_engine_free(engine);
        }
    }

    #[test]
    fn test_ble_peer_ffi() {
        let engine = spore_engine_new();
        unsafe {
            assert_eq!(spore_ble_peer_count(engine), 0);
            assert_eq!(spore_ble_collective_phi(engine), 0.0);

            // Receive peer with valid CV data (3 x f32 = 12 bytes)
            let mut cv_data = Vec::new();
            cv_data.extend_from_slice(&0.7f32.to_le_bytes());
            cv_data.extend_from_slice(&0.3f32.to_le_bytes());
            cv_data.extend_from_slice(&0.5f32.to_le_bytes());

            // Need to enable BLE mode first
            let config = CString::new(
                r#"{"holon_mode":"Off","ble_mode":"ActiveShare","haptic_enabled":true,"telemetry_export":false}"#
            ).unwrap();
            spore_engine_set_sharing_config(engine, config.as_ptr());

            let result = spore_ble_receive_peer(engine, 42, cv_data.as_ptr(), 12);
            assert_eq!(result, 1);
            assert_eq!(spore_ble_peer_count(engine), 1);
            assert!(spore_ble_collective_phi(engine) > 0.0);

            spore_engine_free(engine);
        }
    }

    #[test]
    fn test_ble_receive_null_data() {
        let engine = spore_engine_new();
        unsafe {
            let result = spore_ble_receive_peer(engine, 1, std::ptr::null(), 0);
            assert_eq!(result, 0);
            spore_engine_free(engine);
        }
    }

    #[test]
    fn test_ble_receive_short_data() {
        let engine = spore_engine_new();
        unsafe {
            let data = [1u8, 2, 3];
            let result = spore_ble_receive_peer(engine, 1, data.as_ptr(), 3);
            assert_eq!(result, 0);
            spore_engine_free(engine);
        }
    }

    #[test]
    fn test_ble_advertise_payload_ffi() {
        let engine = spore_engine_new();
        unsafe {
            let config = CString::new(
                r#"{"holon_mode":"Off","ble_mode":"ActiveShare","haptic_enabled":true,"telemetry_export":false}"#
            ).unwrap();
            spore_engine_set_sharing_config(engine, config.as_ptr());

            spore_engine_cycle(engine, std::ptr::null());

            let mut buf = [0u8; 16];
            let written = spore_ble_advertise_payload(engine, buf.as_mut_ptr(), 16);
            assert_eq!(written, 12); // 3 × f32

            spore_engine_free(engine);
        }
    }

    #[test]
    fn test_ble_advertise_buf_too_small() {
        let engine = spore_engine_new();
        unsafe {
            let config = CString::new(
                r#"{"holon_mode":"Off","ble_mode":"ActiveShare","haptic_enabled":true,"telemetry_export":false}"#
            ).unwrap();
            spore_engine_set_sharing_config(engine, config.as_ptr());
            spore_engine_cycle(engine, std::ptr::null());

            let mut buf = [0u8; 4]; // too small
            let written = spore_ble_advertise_payload(engine, buf.as_mut_ptr(), 4);
            assert_eq!(written, 0);

            spore_engine_free(engine);
        }
    }

    #[test]
    fn test_night_mode_with_charging_triggers_sleep() {
        let engine = spore_engine_new();
        unsafe {
            assert_eq!(spore_engine_wake_state(engine), 2); // Alert
            spore_engine_set_night_mode(engine, 1);
            spore_engine_set_battery_state(engine, 80, 1); // charging
            // Run a cycle to trigger edge-detected forwarding
            spore_engine_cycle(engine, std::ptr::null());
            assert_eq!(spore_engine_wake_state(engine), 0); // Sleep
            spore_engine_free(engine);
        }
    }
}
