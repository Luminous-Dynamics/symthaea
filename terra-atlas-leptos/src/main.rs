// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
mod app;
mod components;
mod data;
mod renderer;
mod state;

fn main() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Debug);
    log::info!("Terra Atlas — Leptos WebGL Globe starting");

    leptos::mount::mount_to_body(app::App);
}
