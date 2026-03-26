// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;

mod app;
mod cognitive_adaptivity;
mod components;
mod consciousness;
mod holochain;
mod learning_engine;
mod pages;
mod role;

fn main() {
    console_error_panic_hook::set_once();
    mount_to_body(app::App);
}
