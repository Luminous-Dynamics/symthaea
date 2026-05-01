// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Consciousness + Thermodynamic -> CSS custom property bridge.
//!
//! Updates CSS variables reactively based on consciousness profile,
//! coupled with thermodynamic state. Torpor dims all consciousness-driven
//! effects — the organism enters visual hibernation when energy is low.

use crate::consciousness::use_consciousness;
use crate::thermodynamic::use_thermodynamic;
use crate::util::set_css_var;
use leptos::prelude::*;

/// Wire consciousness and thermodynamic signals to CSS custom properties.
///
/// Call once after both `provide_consciousness_context()` and
/// `provide_thermodynamic_context()` have been called.
pub fn init_consciousness_ui() {
    let consciousness = use_consciousness();
    let thermo = use_thermodynamic();

    Effect::new(move |_| {
        let profile = consciousness.profile.get();
        let warmth = crate::consciousness::combined_score(&profile);
        let energy = thermo.device_energy.get();
        let torpor = thermo.torpor_level.get();

        // Base consciousness values
        set_css_var("--civic-warmth", &format!("{:.3}", warmth));
        set_css_var(
            "--civic-bond-glow",
            &format!("{:.3}", profile.semantic_resonance),
        );

        // Animation speed: consciousness drives intent, energy constrains capacity
        let anim_speed = (0.5 + warmth * 0.5) * energy * (1.0 - torpor * 0.8);
        set_css_var(
            "--civic-animation-speed",
            &format!("{:.3}", anim_speed.max(0.1)),
        );

        // Primary color saturation dims with torpor
        let saturation = (1.0 - torpor * 0.7) * 100.0;
        set_css_var("--primary-saturation", &format!("{:.0}%", saturation));

        // Glow intensity: consciousness * energy * network health
        let network = thermo.network_health.get();
        let glow = profile.semantic_resonance * energy * network;
        set_css_var("--effective-glow", &format!("{:.3}", glow));
    });
}
