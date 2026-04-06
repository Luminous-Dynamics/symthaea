// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;

mod app;
mod components;
mod contexts;
mod mock_data;
mod pages;
mod themes;

fn main() {
    console_error_panic_hook::set_once();
    mount_to_body(app::App);
}
