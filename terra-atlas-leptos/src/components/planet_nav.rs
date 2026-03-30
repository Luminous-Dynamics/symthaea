// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;

use crate::state::globe_state::GlobeState;

#[derive(Clone, Copy, PartialEq)]
pub enum PlanetTarget {
    Earth,
    Moon,
    Venus,
    Mars,
    Jupiter,
    Saturn,
    Sun,
}

impl PlanetTarget {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Earth => "Earth",
            Self::Moon => "Moon",
            Self::Venus => "Venus",
            Self::Mars => "Mars",
            Self::Jupiter => "Jupiter",
            Self::Saturn => "Saturn",
            Self::Sun => "Sun",
        }
    }

    pub fn icon(&self) -> &'static str {
        match self {
            Self::Earth => "\u{1F30D}",  // 🌍
            Self::Moon => "\u{1F319}",   // 🌙
            Self::Venus => "\u{2640}",   // ♀
            Self::Mars => "\u{2642}",    // ♂
            Self::Jupiter => "\u{2643}", // ♃
            Self::Saturn => "\u{2644}",  // ♄
            Self::Sun => "\u{2609}",     // ☉
        }
    }

    /// Camera distance to frame this body nicely
    pub fn camera_distance(&self) -> f32 {
        match self {
            Self::Earth => 4.2,
            Self::Moon => 1.5,
            Self::Venus => 1.2,
            Self::Mars => 1.0,
            Self::Jupiter => 3.0,
            Self::Saturn => 3.5,
            Self::Sun => 5.0,
        }
    }

    /// Orbital parameters matching the renderer's body definitions
    pub fn orbit_params(&self) -> (f32, f32, f32) {
        // (distance, speed, offset) matching CelestialBody in mod.rs
        match self {
            Self::Earth => (0.0, 0.0, 0.0),        // at origin
            Self::Moon => (3.5, 0.05, 0.0),
            Self::Venus => (8.0, 0.015, 1.2),
            Self::Mars => (12.0, 0.01, 2.5),
            Self::Jupiter => (25.0, 0.005, 4.0),
            Self::Saturn => (35.0, 0.003, 5.5),
            Self::Sun => (20.0, 0.02, 0.0),
        }
    }
}

#[component]
pub fn PlanetNav() -> impl IntoView {
    let globe_state = expect_context::<GlobeState>();

    let planets = [
        PlanetTarget::Sun,
        PlanetTarget::Venus,
        PlanetTarget::Earth,
        PlanetTarget::Moon,
        PlanetTarget::Mars,
        PlanetTarget::Jupiter,
        PlanetTarget::Saturn,
    ];

    view! {
        <div class="planet-nav">
            {planets.into_iter().map(|planet| {
                let gs = globe_state.clone();
                let is_active = move || gs.focused_planet.get() == Some(planet);
                let on_click = move |_| {
                    if gs.focused_planet.get() == Some(planet) {
                        // Click again to return to Earth overview
                        gs.focused_planet.set(None);
                    } else {
                        gs.focused_planet.set(Some(planet));
                    }
                };

                view! {
                    <button
                        class=move || if is_active() { "planet-btn active" } else { "planet-btn" }
                        on:click=on_click
                        title=planet.label()
                    >
                        <span class="planet-icon">{planet.icon()}</span>
                        <span class="planet-label">{planet.label()}</span>
                    </button>
                }
            }).collect::<Vec<_>>()}
        </div>
    }
}
