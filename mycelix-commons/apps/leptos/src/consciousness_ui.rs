// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::consciousness_provider::use_consciousness;
use crate::thermodynamic::use_thermodynamic;

pub fn init_consciousness_ui() {
    let c = use_consciousness();
    let t = use_thermodynamic();
    Effect::new(move |_| {
        let p = c.profile.get();
        let warmth = p.combined_score();
        let energy = t.device_energy.get();
        let torpor = t.torpor_level.get();
        let speed = (0.5 + warmth * 0.5) * energy * (1.0 - torpor * 0.8);
        set_css("--consciousness-warmth", &format!("{warmth:.3}"));
        set_css("--consciousness-animation-speed", &format!("{:.3}", speed.max(0.1)));
        set_css("--effective-glow", &format!("{:.3}", p.community * energy));
    });
}

fn set_css(name: &str, value: &str) {
    use wasm_bindgen::JsCast;
    if let Some(el) = web_sys::window().and_then(|w| w.document()).and_then(|d| d.document_element()).and_then(|r| r.dyn_into::<web_sys::HtmlElement>().ok()) {
        let _ = el.style().set_property(name, value);
    }
}
