// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Homeostatic Void detection.
//!
//! When the care board is clear and there are no pending decisions,
//! the UI enters "homeostasis" mode. The chrome fades, the Bevy canvas
//! goes full-screen, and renders a meditative particle field driven by
//! the Sacred Stillness harmony.
//!
//! Being idle is not punished — it regenerates Phi faster.
//! The organism rewards stillness.

use leptos::prelude::*;

/// Reactive state for homeostasis detection.
#[derive(Clone)]
#[allow(dead_code)]
pub struct HomeostasisState {
    /// Number of pending care tasks.
    pub pending_care_tasks: ReadSignal<u32>,
    /// Number of open decisions awaiting votes.
    pub pending_decisions: ReadSignal<u32>,
    /// Whether the UI is in homeostatic void mode.
    pub in_homeostasis: ReadSignal<bool>,
    /// How long (seconds) the user has been in homeostasis continuously.
    pub homeostasis_duration_secs: ReadSignal<f64>,
}

/// Initialize homeostasis detection and provide as Leptos context.
///
/// Pages update the pending_care_tasks and pending_decisions signals
/// as data loads. When both reach zero, homeostasis activates.
pub fn provide_homeostasis_context() -> HomeostasisState {
    let (pending_care, set_pending_care) = signal(0u32);
    let (pending_decisions, set_pending_decisions) = signal(0u32);
    let (in_homeostasis, set_in_homeostasis) = signal(false);
    let (duration, set_duration) = signal(0.0_f64);

    // Derived: homeostasis when nothing is pending
    Effect::new(move |_| {
        let care = pending_care.get();
        let decisions = pending_decisions.get();
        let is_home = care == 0 && decisions == 0;
        set_in_homeostasis.set(is_home);

        // Reset duration counter when leaving homeostasis
        if !is_home {
            set_duration.set(0.0);
        }

        // Set CSS variable for homeostasis-aware styling
        set_css_var("--homeostasis", if is_home { "1" } else { "0" });
    });

    let state = HomeostasisState {
        pending_care_tasks: pending_care,
        pending_decisions,
        in_homeostasis,
        homeostasis_duration_secs: duration,
    };

    // Expose write signals for pages to update
    provide_context(set_pending_care);
    provide_context(set_pending_decisions);
    provide_context(state.clone());

    state
}

fn set_css_var(name: &str, value: &str) {
    use wasm_bindgen::JsCast;
    if let Some(window) = web_sys::window() {
        if let Some(document) = window.document() {
            if let Some(root) = document.document_element() {
                if let Some(el) = root.dyn_ref::<web_sys::HtmlElement>() {
                    let _ = el.style().set_property(name, value);
                }
            }
        }
    }
}

pub fn use_homeostasis() -> HomeostasisState {
    expect_context::<HomeostasisState>()
}
