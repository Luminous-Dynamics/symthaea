// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use leptos::prelude::*;

mod app;
mod components;
pub mod i18n;
mod pages;
mod worker;

fn main() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Debug).expect("logger");
    mount_to_body(app::App);
}
