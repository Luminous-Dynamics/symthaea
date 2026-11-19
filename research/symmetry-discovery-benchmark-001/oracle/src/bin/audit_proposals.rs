// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Independent audit adapter for a finite-group proposal file.
//!
//! The submitted group labels are proposals only. Exact stabilizers, subgroup
//! axioms, false promotions, missed symmetries and witnesses are recomputed
//! from the embedded frozen manifest before the JSON receipt is emitted.

use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::process::ExitCode;

use symmetry_discovery_benchmark_oracle::Manifest;

fn required_string(value: &Value, key: &str) -> Result<String, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("missing or invalid string field {key}"))
}

fn required_array<'a>(value: &'a Value, key: &str) -> Result<&'a Vec<Value>, String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing or invalid array field {key}"))
}

fn run(input_path: &str) -> Result<Value, String> {
    let bytes = fs::read(input_path)
        .map_err(|error| format!("could not read proposal input {input_path}: {error}"))?;
    let input: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid proposal input JSON: {error}"))?;

    if required_string(&input, "schema_version")? != "0.1.0" {
        return Err("unsupported proposal input schema_version".to_owned());
    }
    if required_string(&input, "benchmark_id")?
        != "LUMINOUS-SYMM-QUALIFIED-DISCOVERY-001"
    {
        return Err("unexpected benchmark_id".to_owned());
    }

    let execution_id = required_string(&input, "execution_id")?;
    let split = required_string(&input, "split")?;
    let benchmark_track = required_string(&input, "benchmark_track")?;
    let method = input
        .get("method")
        .ok_or_else(|| "missing method metadata".to_owned())?;
    for field in [
        "name",
        "version",
        "source_revision",
        "configuration_digest_sha256",
    ] {
        let _ = required_string(method, field)?;
    }

    let manifest = Manifest::parse()?;
    manifest.verify_group_axioms()?;
    manifest.validate_all_cases()?;

    let expected_ids = manifest.case_ids_for_split_and_track(&split, &benchmark_track)?;
    if expected_ids.is_empty() {
        return Err(format!(
            "no frozen cases for split={split} track={benchmark_track}"
        ));
    }

    let proposals = required_array(&input, "proposals")?;
    let mut seen = BTreeSet::new();
    let mut case_results = Vec::with_capacity(proposals.len());
    let mut exact_matches = 0usize;
    let mut false_promotions = 0usize;
    let mut missed_symmetries = 0usize;
    let mut invalid_group_proposals = 0usize;

    for proposal in proposals {
        let case_id = required_string(proposal, "case_id")?;
        if !seen.insert(case_id.clone()) {
            return Err(format!("duplicate proposal case_id {case_id}"));
        }
        let (case_track, case_split) = manifest.case_track_and_split(&case_id)?;
        if case_track != benchmark_track || case_split != split {
            return Err(format!(
                "proposal {case_id} is outside the declared split/track"
            ));
        }

        let proposed_subgroup_ids = required_array(proposal, "proposed_subgroup_ids")?
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| format!("{case_id}: subgroup IDs must be strings"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let audit = manifest.audit_case_proposal(&case_id, &proposed_subgroup_ids)?;
        manifest.verify_case_proposal_audit(&case_id, &audit)?;

        let proposal_set: BTreeSet<String> =
            proposed_subgroup_ids.iter().cloned().collect();
        let exact_set: BTreeSet<String> =
            audit.exact_stabilizer.iter().cloned().collect();
        let exact_match = proposal_set == exact_set;
        let false_promotion = !audit.false_promotion_elements.is_empty();
        let false_elimination = !audit.missed_symmetry_elements.is_empty();

        exact_matches += if exact_match { 1 } else { 0 };
        false_promotions += if false_promotion { 1 } else { 0 };
        missed_symmetries += if false_elimination { 1 } else { 0 };
        invalid_group_proposals += if audit.group_axioms_satisfied { 0 } else { 1 };

        case_results.push(json!({
            "case_id": case_id,
            "proposed_subgroup_ids": audit.proposed_elements,
            "exact_stabilizer": audit.exact_stabilizer,
            "proposal_exact_match": exact_match,
            "false_exact_promotion": false_promotion,
            "false_elimination": false_elimination,
            "group_axioms_satisfied": audit.group_axioms_satisfied,
            "axiom_violations": audit.axiom_violations,
            "true_positive_elements": audit.true_positive_elements,
            "false_promotion_elements": audit.false_promotion_elements,
            "missed_symmetry_elements": audit.missed_symmetry_elements,
            "true_negative_elements": audit.true_negative_elements,
            "decisions": audit.decisions,
            "audit_replayed": true
        }));
    }

    let expected_set: BTreeSet<String> = expected_ids.iter().cloned().collect();
    let completed = seen.len() == expected_ids.len() && seen == expected_set;
    Ok(json!({
        "schema_version": "0.1.0",
        "benchmark_id": "LUMINOUS-SYMM-QUALIFIED-DISCOVERY-001",
        "execution_id": execution_id,
        "execution_status": if completed { "completed" } else { "partial" },
        "split": split,
        "benchmark_track": benchmark_track,
        "method": method,
        "case_manifest_binding": "embedded_exact_manifest_at_oracle_source_revision",
        "denominator": {
            "cases_in_scope": expected_ids.len(),
            "cases_attempted": seen.len(),
            "completed": case_results.len(),
            "abstained": 0,
            "failed": 0,
            "out_of_domain": 0
        },
        "metrics": {
            "proposal_exact_match_rate": if case_results.is_empty() { 0.0 } else { exact_matches as f64 / case_results.len() as f64 },
            "false_exact_promotion_case_rate": if case_results.is_empty() { 0.0 } else { false_promotions as f64 / case_results.len() as f64 },
            "false_elimination_case_rate": if case_results.is_empty() { 0.0 } else { missed_symmetries as f64 / case_results.len() as f64 },
            "invalid_group_proposal_case_rate": if case_results.is_empty() { 0.0 } else { invalid_group_proposals as f64 / case_results.len() as f64 }
        },
        "verification": {
            "oracle_audits_replayed": case_results.len(),
            "false_exact_promotion_cases": false_promotions,
            "false_elimination_cases": missed_symmetries,
            "invalid_group_proposal_cases": invalid_group_proposals
        },
        "case_results": case_results
    }))
}

fn main() -> ExitCode {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    if arguments.len() != 1 {
        eprintln!("usage: audit-proposals <proposal-input.json>");
        return ExitCode::from(2);
    }
    match run(&arguments[0]) {
        Ok(receipt) => {
            match serde_json::to_string_pretty(&receipt) {
                Ok(encoded) => {
                    println!("{encoded}");
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("could not encode audit receipt: {error}");
                    ExitCode::FAILURE
                }
            }
        }
        Err(error) => {
            eprintln!("PROPOSAL_AUDIT_FAIL: {error}");
            ExitCode::FAILURE
        }
    }
}
