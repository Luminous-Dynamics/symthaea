// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mycelix Portal — unified interface for all Mycelix clusters.
//! One First Breath. Many domains. Consciousness-gated access.

use leptos::prelude::*;

mod app;
mod identity;
mod nav;

fn main() {
    console_error_panic_hook::set_once();
    leptos::mount::mount_to_body(app::App);
}
