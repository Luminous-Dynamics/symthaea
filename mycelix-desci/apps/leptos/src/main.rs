// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;

mod api;
mod app;
mod components;
mod holochain;
mod pages;
mod types;

fn main() {
    console_error_panic_hook::set_once();
    mount_to_body(app::App);
}
