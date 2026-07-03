// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;

mod achievements;
mod app;
mod components;
mod craft;
mod curriculum;
mod games;
mod graph_cache;
mod holochain;
mod i18n;
mod katex;
mod ledger;
mod location;
mod mesh;
mod pages;
mod persistence;
mod role;
mod search;
mod social_proof;
mod student_profile;
mod study_tracker;
mod tauri_bridge;
mod theme;
mod tutor;

fn main() {
    console_error_panic_hook::set_once();
    holochain::apply_conductor_url_override();
    mount_to_body(app::App);
}
