// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Source-SHA-bound synthetic known-answer runner for family non-inferiority.
//!
//! This runner verifies three statistical control outcomes:
//! - all protected families improve and clear their non-inferiority margins;
//! - one severe family regression is not hidden by an improvement elsewhere;
//! - an unchanged candidate remains inconclusive when evidence is too weak.
//!
//! It is a synthetic evaluator contract, not deployment evidence or a power study.

use serde_json::json;
use symthaea_core::family_noninferiority::{
    FamilyGuardrailObservation, FamilyGuardrailPlan, FamilyGuardrailSpec,
    FamilyGuardrailStatus, FamilyNonInferiorityGate,
};

fn plan(prefix: &str) -> Result<FamilyGuardrailPlan, Box<dyn std::error::Error>> {
    let families = ["bearing.cross-load", "pump.low-flow"]
        .into_iter()
        .map(|family_id| FamilyGuardrailSpec {
            family_id: family_id.to_owned(),
            non_inferiority_margin: 0.05,
            minimum_observations: 100,
            maximum_observations: 200,
            episode_ids: (0..200)
                .map(|index| format!("{prefix}:{family_id}:{index:04}"))
                .collect(),
        })
        .collect();
    Ok(FamilyGuardrailPlan::new(families)?)
}

fn run_case(
    name: &str,
    improvements: [f64; 2],
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let prefix = format!("family-guardrail-known-answer-{name}");
    let plan = plan(&prefix)?;
    let family_ids = ["bearing.cross-load", "pump.low-flow"];
    let mut gate = FamilyNonInferiorityGate::new(plan, 0.05, 1)?;
    let mut final_report = gate.report()?;

    for (family_index, family_id) in family_ids.iter().enumerate() {
        for episode_index in 0..200 {
            let improvement = improvements[family_index];
            let baseline_loss = 0.5;
            let candidate_loss = baseline_loss - improvement;
            if !(0.0..=1.0).contains(&candidate_loss) {
                return Err(format!("invalid fixture candidate loss {candidate_loss}").into());
            }
            final_report = gate.observe(FamilyGuardrailObservation {
                episode_id: format!("{prefix}:{family_id}:{episode_index:04}"),
                family_id: (*family_id).to_owned(),
                baseline_loss,
                candidate_loss,
            })?;
            if matches!(
                final_report.status,
                FamilyGuardrailStatus::RegressionDetected
                    | FamilyGuardrailStatus::AllFamiliesNonInferior
                    | FamilyGuardrailStatus::InconclusiveAtBudget
            ) {
                break;
            }
        }
        if gate.is_terminal() {
            break;
        }
    }

    Ok(json!({
        "name": name,
        "status": format!("{:?}", final_report.status),
        "report_digest": final_report.report_digest,
        "plan_digest": final_report.plan_digest,
        "total_observations": final_report.total_observation_count,
        "families": final_report.families.iter().map(|family| json!({
            "family_id": family.family_id,
            "observation_count": family.observation_count,
            "mean_improvement": family.mean_improvement,
            "lower_confidence_bound": family.lower_confidence_bound,
            "upper_confidence_bound": family.upper_confidence_bound,
            "non_inferiority_margin": family.non_inferiority_margin,
            "status": format!("{:?}", family.status)
        })).collect::<Vec<_>>()
    }))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let positive = run_case("positive", [0.8, 0.8])?;
    let hidden_regression = run_case("hidden-regression", [0.9, -0.9])?;
    let no_change = run_case("no-change", [0.0, 0.0])?;

    if positive["status"] != "AllFamiliesNonInferior" {
        return Err("positive control failed to pass every family".into());
    }
    if hidden_regression["status"] != "RegressionDetected" {
        return Err("negative control failed to detect a family regression".into());
    }
    if no_change["status"] != "InconclusiveAtBudget" {
        return Err("no-change control must not be promoted as proven non-inferior".into());
    }

    let output = json!({
        "schema_version": "hlb-family-noninferiority-known-answers-v1",
        "status": "synthetic_known_answer_fixture_only",
        "source_sha": std::env::var("SYMTHAEA_SOURCE_SHA").ok(),
        "campaign_alpha": 0.05,
        "candidate_ordinal": 1,
        "simultaneous_family_bounds": true,
        "cases": [positive, hidden_regression, no_change]
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
