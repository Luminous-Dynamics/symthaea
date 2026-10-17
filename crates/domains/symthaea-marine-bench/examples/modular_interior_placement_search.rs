// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Demonstrates a tiny joint placement search on synthetic configuration data.
//! This is a geometry-screening example, not a vessel-safety qualification.

use std::error::Error;

use serde::Serialize;
use symthaea_marine_bench::modular_interior::{Configuration, PositionMm};
use symthaea_marine_bench::modular_interior_placement_search::{
    search_placements, verify_placement_search_report, PlacementChoice, PlacementDomain,
    PlacementSearchLimits,
};

#[derive(Serialize)]
struct CandidateSummary {
    candidate_id: String,
    choice_ids: Vec<String>,
    moved_instances: Vec<String>,
    disposition: String,
    interaction_findings: Vec<String>,
    physical_overlaps: usize,
    out_of_bounds_placements: usize,
}

#[derive(Serialize)]
struct ExampleSummary {
    search_version: String,
    synthetic_fixture: bool,
    candidates_evaluated: usize,
    search_complete: bool,
    truncated_by_candidate_cap: bool,
    candidate_count_lower_bound: usize,
    receipt_verifies_by_recomputation: bool,
    report_sha256: String,
    candidates: Vec<CandidateSummary>,
    claim_ceiling: String,
}

fn choice(choice_id: &str, x_mm: i64) -> PlacementChoice {
    PlacementChoice {
        choice_id: choice_id.to_owned(),
        deck_id: "MAIN".to_owned(),
        position_mm: PositionMm { x_mm, y_mm: 1100, z_mm: 0 },
        rotation_z_deg: 0,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let json = include_str!("../fixtures/modular-interiors/liveaboard-custom-layout-v1.json");
    let baseline: Configuration = serde_json::from_str(json)?;
    let domains = vec![
        PlacementDomain {
            instance_id: "CABIN-01".to_owned(),
            choices: vec![choice("CABIN-X600", 600)],
        },
        PlacementDomain {
            instance_id: "WETROOM-01".to_owned(),
            choices: vec![choice("WETROOM-X2900", 2900)],
        },
    ];
    let limits = PlacementSearchLimits {
        max_domains: 8,
        max_choices_per_domain: 8,
        max_total_choices: 32,
        max_candidates: 100,
        max_moved_instances: 3,
    };
    let report = search_placements(&baseline, &domains, limits.clone())?;
    let receipt_verifies_by_recomputation =
        verify_placement_search_report(&baseline, &domains, &limits, &report);

    let candidates = report.body.candidates.iter().map(|candidate| CandidateSummary {
        candidate_id: candidate.candidate_id.clone(),
        choice_ids: candidate.selected_choice_ids.clone(),
        moved_instances: candidate.moved_instance_ids.clone(),
        disposition: format!("{:?}", candidate.disposition),
        interaction_findings: candidate.interaction_findings.iter()
            .map(|finding| format!("{}|{}", finding.code, finding.subject))
            .collect(),
        physical_overlaps: candidate.validation_receipt.body.geometry.physical_overlaps,
        out_of_bounds_placements: candidate.validation_receipt.body.geometry.out_of_bounds_placements,
    }).collect();

    let summary = ExampleSummary {
        search_version: report.body.search_version.clone(),
        synthetic_fixture: true,
        candidates_evaluated: report.body.candidates.len(),
        search_complete: report.body.search_complete,
        truncated_by_candidate_cap: report.body.truncated_by_candidate_cap,
        candidate_count_lower_bound: report.body.candidate_count_lower_bound,
        receipt_verifies_by_recomputation,
        report_sha256: report.report_sha256,
        candidates,
        claim_ceiling: report.body.claim_ceiling,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);
    if !receipt_verifies_by_recomputation {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "recomputed placement-search receipt did not verify",
        ).into());
    }
    Ok(())
}
