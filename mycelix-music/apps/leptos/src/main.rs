// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;

mod app;
mod audio_bridge;
mod components;
mod consciousness_viz;
mod journey;
mod live_synth;
mod pages;
mod synth_engine;
mod types;
mod visualization;
mod wav_export;

fn main() {
    console_error_panic_hook::set_once();
    mount_to_body(app::App);
}
