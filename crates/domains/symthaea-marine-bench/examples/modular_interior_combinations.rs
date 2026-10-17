// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Demonstrates bounded combinations and interaction diagnostics on synthetic data.
//! A generated result is not a vessel safety or qualification decision.

use std::error::Error;

use serde::Serialize;
use symthaea_marine_bench::modular_interior::Configuration;
use symthaea_marine_bench::modular_interior_combinations::{
    search_combinations, verify_combination_search_report, CombinationDisposition,
    SearchLimits,
};
use symthaea_marine_bench::modular_interior_variants::ModuleVariant;

#[derive(Serialize)]
struct CandidateSummary {
    combination_id: String,
    variant_ids: Vec<String>,
    target_module_ids: Vec<String>,
    disposition: CombinationDisposition,
    validator_disposition: Option<String>,
    interaction_findings: Vec<String>,
}

#[derive(Serialize)]
struct ExampleSummary {
    search_version: String,
    input_evidence_note: String,
    candidates_evaluated: usize,
    search_complete: bool,
    truncated_by_candidate_cap: bool,
    receipt_verifies_by_recomputation: bool,
    report_sha256: String,
    candidates: Vec<CandidateSummary>,
    claim_ceiling: String,
}

fn module_variant(
    baseline: &Configuration,
    module_id: &str,
    variant_id: &str,
    additional_ac_watts: f64,
) -> Result<ModuleVariant, Box<dyn Error>> {
    let mut replacement = baseline.module_catalog.iter()
        .find(|module| module.module_id == module_id)
        .cloned()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("fixture has no module {module_id}")))?;
    replacement.revision = replacement.revision.checked_add(1)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("revision overflow for {module_id}")))?;
    replacement.name = format!("{} / combination-search example", replacement.name);

    if additional_ac_watts != 0.0 {
        let port = replacement.ports.iter_mut()
            .find(|port| port.port_id == "AC_IN")
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("module {module_id} has no AC_IN port")))?;
        let watts = port.rating.value.ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("module {module_id} has no declared AC rating")))?;
        port.rating.value = Some(watts + additional_ac_watts);
    }

    Ok(ModuleVariant { variant_id: variant_id.to_owned(), replacement })
}

fn main() -> Result<(), Box<dyn Error>> {
    let fixture = include_str!("../fixtures/modular-interiors/liveaboard-custom-layout-v1.json");
    let baseline: Configuration = serde_json::from_str(fixture)?;
    let variants = vec![
        module_variant(&baseline, "COMPACT-GALLEY", "GALLEY-HIGH-LOAD", 2_000.0)?,
        module_variant(&baseline, "FLEX-WORKSPACE", "WORKSPACE-HIGH-LOAD", 2_000.0)?,
    ];
    let limits = SearchLimits {
        max_input_variants: 16,
        max_candidates: 100,
        max_changed_modules: 3,
    };
    let report = search_combinations(&baseline, &variants, limits.clone())?;
    let receipt_verifies_by_recomputation =
        verify_combination_search_report(&baseline, &variants, &limits, &report);

    let candidates = report.body.candidates.iter().map(|candidate| CandidateSummary {
        combination_id: candidate.combination_id.clone(),
        variant_ids: candidate.selected_variant_ids.clone(),
        target_module_ids: candidate.target_module_ids.clone(),
        disposition: candidate.disposition,
        validator_disposition: candidate.validation_receipt.as_ref()
            .map(|receipt| receipt.body.overall_disposition.clone()),
        interaction_findings: candidate.interaction_findings.iter()
            .map(|finding| format!("{}|{}", finding.code, finding.subject))
            .collect(),
    }).collect();

    let summary = ExampleSummary {
        search_version: report.body.search_version.clone(),
        input_evidence_note: "Synthetic workflow fixture with assumption-based inputs; not measured vessel data.".to_owned(),
        candidates_evaluated: report.body.candidates.len(),
        search_complete: report.body.search_complete,
        truncated_by_candidate_cap: report.body.truncated_by_candidate_cap,
        receipt_verifies_by_recomputation,
        report_sha256: report.report_sha256,
        candidates,
        claim_ceiling: report.body.claim_ceiling,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);
    if !receipt_verifies_by_recomputation {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "recomputed search receipt did not verify").into());
    }
    Ok(())
}
