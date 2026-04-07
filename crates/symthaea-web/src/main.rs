// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;

mod app;
mod components;
pub mod i18n;
mod pages;
mod service_worker;
mod state;
mod worker;

fn main() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Debug).expect("logger");

    service_worker::register();

    mount_to_body(app::App);
}
