// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Consciousness + Thermodynamic → CSS custom property bridge.
//!
//! Updates CSS variables reactively based on consciousness profile,
//! coupled with thermodynamic state. Torpor dims all consciousness-driven
//! effects — the organism enters visual hibernation when energy is low.

use leptos::prelude::*;
use crate::consciousness_provider::use_consciousness;
use crate::thermodynamic::use_thermodynamic;

/// Wire consciousness and thermodynamic signals to CSS custom properties.
pub fn init_consciousness_ui() {
    let consciousness = use_consciousness();
    let thermo = use_thermodynamic();

    Effect::new(move |_| {
        let profile = consciousness.profile.get();
        let warmth = profile.combined_score();
        let energy = thermo.device_energy.get();
        let torpor = thermo.torpor_level.get();

        // Base consciousness values
        set_css_var("--consciousness-warmth", &format!("{:.3}", warmth));
        set_css_var("--consciousness-bond-glow", &format!("{:.3}", profile.community));

        // Animation speed: consciousness drives intent, energy constrains capacity
        // In torpor, animations slow dramatically to conserve Joules
        let anim_speed = (0.5 + warmth * 0.5) * energy * (1.0 - torpor * 0.8);
        set_css_var("--consciousness-animation-speed", &format!("{:.3}", anim_speed.max(0.1)));

        // Primary color saturation dims with torpor
        // At full torpor, the warm amber becomes a muted grey-brown
        let saturation = (1.0 - torpor * 0.7) * 100.0;
        set_css_var("--primary-saturation", &format!("{:.0}%", saturation));

        // Glow intensity: consciousness * energy * network health
        let network = thermo.network_health.get();
        let glow = profile.community * energy * network;
        set_css_var("--effective-glow", &format!("{:.3}", glow));
    });
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
