// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;

mod achievements;
mod adaptivity_provider;
mod app;
mod cognitive_adaptivity;
mod components;
mod consciousness;
mod consciousness_ui;
mod curriculum;
mod games;
mod graph_cache;
mod holochain;
mod i18n;
mod katex;
mod learning_engine;
mod pages;
mod persistence;
mod role;
mod search;
mod social_proof;
mod student_profile;
mod study_tracker;
mod theme;

fn main() {
    console_error_panic_hook::set_once();
    mount_to_body(app::App);
}
