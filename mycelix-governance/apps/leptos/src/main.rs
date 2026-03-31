// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;

mod app;
mod components;
mod consciousness_provider;
mod consciousness_ui;
mod contexts;
mod holochain;
mod homeostasis;
mod mock_data;
mod pages;
mod themes;
mod thermodynamic;
mod toasts;

fn main() {
    console_error_panic_hook::set_once();
    mount_to_body(app::App);
}
