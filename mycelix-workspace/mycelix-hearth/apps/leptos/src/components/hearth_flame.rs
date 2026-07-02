// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Reusable campfire flame component — the visual heartbeat of the Hearth.
//!
//! Two modes:
//! - `Full`: Large, centered flames for the founding ceremony
//! - `Ambient`: Small, corner-positioned glow for daily life

use leptos::prelude::*;

/// How the flames should render.
#[derive(Clone, Copy, PartialEq)]
pub enum FlameMode {
    /// Large, centered — used during the founding ceremony
    Full,
    /// Small, bottom-right corner — ambient presence during daily use
    Ambient,
}

/// The campfire flame element — 5 flames, 8 embers, warm radial glow.
#[component]
pub fn HearthFlame(
    #[prop(default = FlameMode::Ambient)]
    mode: FlameMode,
) -> impl IntoView {
    let class = match mode {
        FlameMode::Full => "campfire-container campfire-full",
        FlameMode::Ambient => "campfire-container campfire-ambient",
    };

    view! {
        <div class=class aria-hidden="true">
            <div class="flame flame-1" />
            <div class="flame flame-2" />
            <div class="flame flame-3" />
            <div class="flame flame-4" />
            <div class="flame flame-5" />
            <div class="ember-particles">
                <div class="ember e1" />
                <div class="ember e2" />
                <div class="ember e3" />
                <div class="ember e4" />
                <div class="ember e5" />
                <div class="ember e6" />
                <div class="ember e7" />
                <div class="ember e8" />
            </div>
            <div class="campfire-glow" />
        </div>
    }
}
