// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Run the small frozen synthetic manufacturing active-learning comparison.
//!
//! Example:
//! cargo run -p symthaea-manufacturing-active-learning-bench --example run_reference_campaigns
use serde::Serialize;
use symthaea_manufacturing_active_learning_bench::{
    CampaignManifest, CampaignTranscript, EvaluationReport, Policy, ScenarioFamily,
    evaluate_transcript, run_campaign,
};
use std::error::Error;

#[derive(Serialize)]
struct RunArtifact {
    scenario: ScenarioFamily,
    policy: Policy,
    manifest_digest: String,
    transcript: CampaignTranscript,
    independent_evaluation: EvaluationReport,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut artifacts = Vec::new();
    for (scenario_index, scenario) in ScenarioFamily::ALL.into_iter().enumerate() {
        // Same oracle seed within one scenario and the same budget across policy arms.
        // Oracle seed and scenario are held outside the public selector view.
        let manifest = CampaignManifest::reference(
            scenario,
            0x4d46_4700_u64 + scenario_index as u64,
            20,
        );
        let manifest_digest = manifest.digest()?;

        for policy in Policy::ALL {
            let transcript = run_campaign(&manifest, policy, 0x5eed_2026)?;
            let independent_evaluation = evaluate_transcript(&manifest, &transcript)?;
            artifacts.push(RunArtifact {
                scenario,
                policy,
                manifest_digest: manifest_digest.clone(),
                transcript,
                independent_evaluation,
            });
        }
    }

    // Exact normalized JSON artifact. This is a deterministic synthetic run, not
    // evidence of physical manufacturing performance or a separately built verifier.
    println!("{}", serde_json::to_string_pretty(&artifacts)?);
    Ok(())
}
