// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Consciousness + Thermodynamic → CSS custom property bridge.
//!
//! Civic indigo glow modulated by consciousness and device energy.
//! Torpor dims all governance-driven effects — deliberation slows
//! when the organism's substrate is depleted.

use leptos::prelude::*;
use crate::consciousness_provider::use_consciousness;
use crate::thermodynamic::use_thermodynamic;

pub fn init_consciousness_ui() {
    let consciousness = use_consciousness();
    let thermo = use_thermodynamic();

    Effect::new(move |_| {
        let profile = consciousness.profile.get();
        let warmth = profile.combined_score();
        let energy = thermo.device_energy.get();
        let torpor = thermo.torpor_level.get();

        set_css_var("--consciousness-warmth", &format!("{:.3}", warmth));
        set_css_var("--consciousness-bond-glow", &format!("{:.3}", profile.community));

        let anim_speed = (0.5 + warmth * 0.5) * energy * (1.0 - torpor * 0.8);
        set_css_var("--consciousness-animation-speed", &format!("{:.3}", anim_speed.max(0.1)));

        let saturation = (1.0 - torpor * 0.7) * 100.0;
        set_css_var("--primary-saturation", &format!("{:.0}%", saturation));

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
