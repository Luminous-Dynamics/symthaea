// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use mycelix_identity_ui::app::App;

fn main() {
    console_error_panic_hook::set_once();
    mount_to_body(App);
}
