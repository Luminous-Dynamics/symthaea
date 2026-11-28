// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Run reproducible PCE v0.2 reference and random baselines.
//!
//! Usage:
//!   cargo run -p symthaea-psych-bench --example cognitive_ecology_baselines
//!   cargo run -p symthaea-psych-bench --example cognitive_ecology_baselines -- --jsonl

use symthaea_psych_bench::benchmarks::cognitive_ecology::{
    ScenarioManifest, receipts_to_jsonl, run_random_policy, run_reference_policy,
};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let jsonl = args.iter().any(|arg| arg == "--jsonl");

    for seed in [42_u64, 43_u64] {
        let manifest = ScenarioManifest {
            seed,
            ..ScenarioManifest::default()
        };
        let reference = run_reference_policy(&manifest, false)?;
        let recovery = run_reference_policy(&manifest, true)?;
        let random = run_random_policy(&manifest)?;

        let summary = json!({
            "scenario_id": manifest.scenario_id,
            "generator_version": manifest.generator_version,
            "seed": seed,
            "layout": format!("{:?}", manifest.layout().filter_location),
            "manifest_digest": manifest.digest()?,
            "reference": reference.score,
            "recovery": recovery.score,
            "random": random.score,
            "reference_final_state_digest": reference.final_state_digest,
            "recovery_final_state_digest": recovery.final_state_digest,
            "random_final_state_digest": random.final_state_digest,
        });

        if jsonl {
            println!("{}", serde_json::to_string(&summary)?);
        } else {
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
    }

    Ok(())
}
