// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Prism entry point.

use prism_ui::app::App;

fn main() {
    console_error_panic_hook::set_once();
    _ = console_log::init_with_level(log::Level::Info);
    log::info!("Prism starting...");
    leptos::mount::mount_to_body(App);
}
