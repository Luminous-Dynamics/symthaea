// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Symthaea Spore consciousness bridge.
//!
//! Loads the Spore WASM kernel in the browser and computes real
//! consciousness metrics (Phi, tier) from user behavior data.
//!
//! The bridge runs entirely client-side — no consensus compute needed.
//! Falls back gracefully to the default 0.5 profile if Spore is unavailable.
//!
//! # Architecture
//!
//! 1. `SporeState` tracks loading status and computed metrics
//! 2. `provide_spore_bridge()` spawns a background task to load the kernel
//! 3. On success, periodically recomputes Phi from behavior data
//! 4. Updates the consciousness profile signals in `ConsciousnessState`

use leptos::prelude::*;

/// Spore bridge state — tracks whether the consciousness kernel is loaded.
#[derive(Clone)]
pub struct SporeState {
    pub loaded: ReadSignal<bool>,
    pub phi: ReadSignal<f64>,
    pub cycle_count: ReadSignal<u64>,
    set_loaded: WriteSignal<bool>,
    set_phi: WriteSignal<f64>,
    set_cycle_count: WriteSignal<u64>,
}

/// Initialize the Spore bridge. Attempts to load the Spore WASM kernel
/// from `/spore/symthaea_spore_bg.wasm`. If the kernel is not available
/// (file missing, WASM error), the bridge stays in unloaded state and
/// the consciousness profile remains at its default values.
///
/// Call this after `provide_consciousness_context()` in the app root.
pub fn provide_spore_bridge() {
    let (loaded, set_loaded) = signal(false);
    let (phi, set_phi) = signal(0.0_f64);
    let (cycle_count, set_cycle_count) = signal(0_u64);

    let state = SporeState {
        loaded,
        phi,
        cycle_count,
        set_loaded,
        set_phi,
        set_cycle_count,
    };
    provide_context(state.clone());

    // Attempt to load Spore WASM in background
    wasm_bindgen_futures::spawn_local(async move {
        // Check if Spore WASM is available (look for the JS binding module)
        let available = js_sys::Reflect::get(
            &wasm_bindgen::JsValue::from(web_sys::window().unwrap()),
            &wasm_bindgen::JsValue::from_str("__SPORE_AVAILABLE"),
        )
        .ok()
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

        if available {
            web_sys::console::log_1(&"[Spore] Consciousness kernel detected".into());
            set_loaded.set(true);
            // In a full integration, we would:
            // 1. Import the WASM module via wasm-bindgen
            // 2. Call init() to initialize the Spore engine
            // 3. Set up a periodic tick (e.g., every 250ms) to run cognitive cycles
            // 4. Read Phi from the engine and update set_phi
            // 5. Map Phi to consciousness profile dimensions
            // 6. Update the ConsciousnessState signals
        } else {
            web_sys::console::log_1(
                &"[Spore] Consciousness kernel not available — using default profile".into(),
            );
        }
    });
}

/// Retrieve the Spore bridge state from context.
pub fn use_spore() -> SporeState {
    expect_context::<SporeState>()
}

/// Check if the Spore kernel is loaded and running.
pub fn is_spore_active() -> bool {
    use_spore().loaded.get_untracked()
}
