// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Run the frozen synthetic manufacturing active-learning comparison.
//!
//! Example:
//! cargo run -p symthaea-manufacturing-active-learning-bench --example run_reference_campaigns
use serde::Serialize;
use symthaea_manufacturing_active_learning_bench::{
    CampaignManifest, CampaignTranscript, EvaluationReport, Policy, ScenarioFamily,
    evaluate_transcript, run_campaign,
};
use std::error::Error;

const PAIRED_SEEDS: u32 = 30;

#[derive(Serialize)]
struct RunArtifact {
    paired_seed_index: u32,
    scenario: ScenarioFamily,
    policy: Policy,
    manifest_digest: String,
    transcript: CampaignTranscript,
    independent_evaluation: EvaluationReport,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut artifacts = Vec::new();
    for paired_seed_index in 0..PAIRED_SEEDS {
        for (scenario_index, scenario) in ScenarioFamily::ALL.into_iter().enumerate() {
            // Policy arms share both the hidden oracle seed and the selector seed for each
            // paired scenario/seed. Neither hidden scenario labels nor oracle seeds enter
            // the public selector view.
            let oracle_seed =
                0x4d46_4700_u64 + u64::from(paired_seed_index) * 5 + scenario_index as u64;
            let manifest = CampaignManifest::reference(scenario, oracle_seed, 20);
            let manifest_digest = manifest.digest()?;
            let selector_seed = 0x5eed_2026_u64 + u64::from(paired_seed_index);

            for policy in Policy::ALL {
                let transcript = run_campaign(&manifest, policy, selector_seed)?;
                let independent_evaluation = evaluate_transcript(&manifest, &transcript)?;
                artifacts.push(RunArtifact {
                    paired_seed_index,
                    scenario,
                    policy,
                    manifest_digest: manifest_digest.clone(),
                    transcript,
                    independent_evaluation,
                });
            }
        }
    }

    // Exact normalized JSON artifact. This is a synthetic policy comparison, not
    // evidence of physical manufacturing performance or a separately built verifier.
    println!("{}", serde_json::to_string_pretty(&artifacts)?);
    Ok(())
}
