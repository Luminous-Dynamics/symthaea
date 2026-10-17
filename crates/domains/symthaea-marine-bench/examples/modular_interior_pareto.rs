// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Demonstrates evidence-gated ranking of synthetic modular interior layouts.
//! The placeholder data should keep these candidates out of an evidence-backed frontier.

use std::error::Error;

use symthaea_marine_bench::modular_interior::{
    validate_configuration, Configuration,
};
use symthaea_marine_bench::modular_interior_pareto::{
    rank_candidates, verify_pareto_report, CandidateSubmission,
    ObjectiveDirection, ObjectiveEvidenceStatus, ObjectiveObservation,
};

fn submission(candidate_id: &str, json: &str) -> Result<CandidateSubmission, Box<dyn Error>> {
    let configuration: Configuration = serde_json::from_str(json)?;
    let validation_receipt = validate_configuration(&configuration)?;
    let declared_mass = validation_receipt.body.weight_ledger.known_subtotal_mass_kg;
    Ok(CandidateSubmission {
        candidate_id: candidate_id.to_owned(),
        configuration,
        validation_receipt,
        objectives: vec![ObjectiveObservation {
            metric_id: "installed_module_mass".to_owned(),
            value: Some(declared_mass),
            unit: "kg".to_owned(),
            direction: ObjectiveDirection::Minimize,
            evidence_status: ObjectiveEvidenceStatus::DesignAssumption,
            comparison_basis: "synthetic fixture masses; no physical configuration validation".to_owned(),
            evidence_refs: vec!["SYNTHETIC-MOD-001".to_owned()],
        }],
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let liveaboard = include_str!("../fixtures/modular-interiors/liveaboard-custom-layout-v1.json");
    let expedition = include_str!("../fixtures/modular-interiors/expedition-research-layout-v1.json");
    let submissions = vec![
        submission("liveaboard-synthetic", liveaboard)?,
        submission("expedition-synthetic", expedition)?,
    ];
    let report = rank_candidates(submissions.clone())?;
    if !verify_pareto_report(&submissions, &report) {
        return Err("Pareto report failed its input/receipt integrity check".into());
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
