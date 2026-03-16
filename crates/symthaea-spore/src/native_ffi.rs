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
}
