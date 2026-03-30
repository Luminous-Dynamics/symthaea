// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;

mod ambient_sound;
mod app;
mod circadian;
mod components;
mod consciousness_provider;
mod consciousness_ui;
pub mod hearth_actions;
mod hearth_context;
mod holochain;
mod homeostasis;
pub mod mock_data;
mod onboarding;
pub mod record_bridge;
mod pages;
mod signal_listener;
mod simulated_life;
mod soma_bridge;
mod themes;
mod thermodynamic;
mod toasts;
mod types;
pub mod visualization;

fn main() {
    console_error_panic_hook::set_once();
    mount_to_body(app::App);
}
