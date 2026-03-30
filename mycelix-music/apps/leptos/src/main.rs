// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;

mod app;
mod components;
mod pages;
mod types;
mod visualization;

fn main() {
    console_error_panic_hook::set_once();
    mount_to_body(app::App);
}
