// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symtropy_sim_bridge::{run_mk0_bootstrapper_scenario, Mk0ScenarioConfig};

fn main() {
    let report = run_mk0_bootstrapper_scenario(Mk0ScenarioConfig::default());
    let json =
        serde_json::to_string_pretty(&report).expect("mk0 bootstrapper report must serialize");
    println!("{json}");
}
