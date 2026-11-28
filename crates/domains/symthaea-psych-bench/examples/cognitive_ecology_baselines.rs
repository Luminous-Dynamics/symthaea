// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Run reproducible PCE v0.3 reference, random and safe-oracle baselines.
//!
//! Usage:
//!   cargo run -p symthaea-psych-bench --example cognitive_ecology_baselines
//!   cargo run -p symthaea-psych-bench --example cognitive_ecology_baselines -- --jsonl

use symthaea_psych_bench::benchmarks::cognitive_ecology::{
    ScenarioManifest, run_cost_oracle, run_random_policy, run_reference_policy, run_search_oracle,
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
        let oracle = run_search_oracle(&manifest)?;
        let cost_oracle = run_cost_oracle(&manifest)?;

        let summary = json!({
            "scenario_id": manifest.scenario_id.clone(),
            "generator_version": manifest.generator_version.clone(),
            "seed": seed,
            "layout": format!("{:?}", manifest.layout().filter_location),
            "manifest_digest": manifest.digest()?,
            "reference": reference.score,
            "recovery": recovery.score,
            "random": random.score,
            "search_oracle": oracle.score,
            "cost_oracle": cost_oracle.score,
            "reference_final_state_digest": reference.final_state_digest,
            "recovery_final_state_digest": recovery.final_state_digest,
            "random_final_state_digest": random.final_state_digest,
            "oracle_final_state_digest": oracle.final_state_digest,
            "cost_oracle_final_state_digest": cost_oracle.final_state_digest,
        });

        if jsonl {
            println!("{}", serde_json::to_string(&summary)?);
        } else {
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
    }

    Ok(())
}
