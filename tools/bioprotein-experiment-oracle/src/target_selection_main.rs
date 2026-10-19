// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
#[path = "target_selection.rs"]
mod target_selection;
fn main() {
    std::process::exit(target_selection::run_cli());
}
