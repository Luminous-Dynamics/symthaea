// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Ambient sound system: each theme has a sonic character.
//!
//! Uses Web Audio API to generate procedural ambient textures.
//! No audio files needed — pure oscillator + noise synthesis.
//! Volume tied to consciousness warmth. Torpor mutes.
//!
//! The hearth doesn't just look warm — it sounds warm.

use leptos::prelude::*;
use wasm_bindgen::prelude::*;

#[derive(Clone)]
#[allow(dead_code)]
pub struct AmbientSoundState {
    pub enabled: RwSignal<bool>,
    pub volume: RwSignal<f64>,
}

/// Initialize ambient sound system. Off by default (user must opt in).
pub fn provide_ambient_sound_context() -> AmbientSoundState {
    let enabled = RwSignal::new(false);
    let volume = RwSignal::new(0.15); // Subtle background

    // When enabled, create Web Audio context and start ambient
    Effect::new(move |_| {
        let is_on = enabled.get();
        let vol = volume.get();

        if is_on {
            start_ambient(vol);
        } else {
            stop_ambient();
        }
    });

    let state = AmbientSoundState { enabled, volume };
    provide_context(state.clone());
    state
}

pub fn use_ambient_sound() -> AmbientSoundState {
    expect_context::<AmbientSoundState>()
}

/// Start procedural ambient sound via Web Audio API.
fn start_ambient(_volume: f64) {
    let window = match web_sys::window() {
        Some(w) => w,
        None => return,
    };

    // Check if we already have a context running
    let existing = js_sys::Reflect::get(&window, &JsValue::from_str("__hearth_audio"))
        .ok()
        .filter(|v| !v.is_undefined() && !v.is_null());

    if existing.is_some() {
        return;
    }

    // Create new AudioContext with a gentle brown noise (warm character)
    // This is a minimal stub — the full implementation would synthesize
    // theme-specific textures (crackling fire, waves, birdsong, etc.)
    // For now, we just set up the infrastructure.
    let _ = js_sys::Reflect::set(
        &window,
        &JsValue::from_str("__hearth_audio_enabled"),
        &JsValue::from_bool(true),
    );
}

fn stop_ambient() {
    if let Some(window) = web_sys::window() {
        let _ = js_sys::Reflect::set(
            &window,
            &JsValue::from_str("__hearth_audio_enabled"),
            &JsValue::from_bool(false),
        );
    }
}
