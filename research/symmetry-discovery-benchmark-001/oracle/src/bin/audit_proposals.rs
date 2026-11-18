// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! JSON adapter for independent symmetry-proposal auditing.
//!
//! Usage:
//! cargo run --locked --manifest-path research/symmetry-discovery-benchmark-001/oracle/Cargo.toml --bin audit-proposals -- submission.json
//!
//! The submission must contain exactly one prediction or explicit abstention
//! for every case in the declared track/split. The output contains exact
//! element-level witnesses but is an audit artifact, not a benchmark score.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::process::ExitCode;

use symmetry_discovery_benchmark_oracle::{Manifest, SubgroupProposalAuditV1};

const BENCHMARK_ID: &str = "LUMINOUS-SYMM-QUALIFIED-DISCOVERY-001";

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum DispositionV1 {
    Predicted,
    Abstained,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct MethodMetadataV1 {
    name: String,
    version: String,
    source_revision: String,
    configuration_digest_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PredictionV1 {
    case_id: String,
    disposition: DispositionV1,
    #[serde(default)]
    proposed_elements: Vec<String>,
    #[serde(default)]
    abstention_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SubmissionV1 {
    schema_version: String,
    benchmark_id: String,
    benchmark_track: String,
    split: String,
    method: MethodMetadataV1,
    predictions: Vec<PredictionV1>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseAuditV1 {
    case_id: String,
    disposition: DispositionV1,
    abstention_reason: Option<String>,
    exact_stabilizer: Vec<String>,
    exact_subgroup_match: Option<bool>,
    proposal_audit: Option<SubgroupProposalAuditV1>,
}

#[derive(Debug, Clone, Serialize)]
struct SummaryV1 {
    cases_in_scope: usize,
    predicted: usize,
    abstained: usize,
    exact_subgroup_matches: usize,
    false_promotions: usize,
    missed_symmetries: usize,
    invalid_group_proposals: usize,
}

#[derive(Debug, Clone, Serialize)]
struct AuditReportV1 {
    schema_version: String,
    benchmark_id: String,
    benchmark_track: String,
    split: String,
    method: MethodMetadataV1,
    cases: Vec<CaseAuditV1>,
    summary: SummaryV1,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_submission(submission: &SubmissionV1) -> Result<(), String> {
    if submission.schema_version != "0.1.0" {
        return Err("schema_version must be 0.1.0".to_owned());
    }
    if submission.benchmark_id != BENCHMARK_ID {
        return Err("unexpected benchmark_id".to_owned());
    }
    if ![
        "exact_finite_structural_stabilizer",
        "finite_reference_group_regression_discovery",
    ]
    .contains(&submission.benchmark_track.as_str())
    {
        return Err("unsupported benchmark_track".to_owned());
    }
    if !["discovery", "calibration", "final_holdout"].contains(&submission.split.as_str()) {
        return Err("split must be discovery, calibration or final_holdout".to_owned());
    }
    if !nonempty(&submission.method.name)
        || !nonempty(&submission.method.version)
        || !nonempty(&submission.method.source_revision)
    {
        return Err("method name, version and source_revision must be nonempty".to_owned());
    }
    if !valid_sha256(&submission.method.configuration_digest_sha256) {
        return Err("configuration_digest_sha256 must be 64 lowercase hexadecimal characters"
            .to_owned());
    }
    if submission.predictions.is_empty() {
        return Err("predictions must not be empty".to_owned());
    }

    let mut seen = BTreeSet::new();
    for prediction in &submission.predictions {
        if !nonempty(&prediction.case_id) || !seen.insert(&prediction.case_id) {
            return Err(format!(
                "prediction case_id is empty or duplicated: {}",
                prediction.case_id
            ));
        }
        match prediction.disposition {
            DispositionV1::Predicted => {
                if prediction.proposed_elements.is_empty() {
                    return Err(format!(
                        "predicted case {} must propose at least one element",
                        prediction.case_id
                    ));
                }
                if prediction.abstention_reason.is_some() {
                    return Err(format!(
                        "predicted case {} must not have an abstention_reason",
                        prediction.case_id
                    ));
                }
                let mut unique = BTreeSet::new();
                for element_id in &prediction.proposed_elements {
                    if !unique.insert(element_id) {
                        return Err(format!(
                            "case {} duplicates proposed element {}",
                            prediction.case_id, element_id
                        ));
                    }
                }
            }
            DispositionV1::Abstained => {
                if !prediction.proposed_elements.is_empty() {
                    return Err(format!(
                        "abstained case {} must have no proposed_elements",
                        prediction.case_id
                    ));
                }
                if !prediction
                    .abstention_reason
                    .as_deref()
                    .is_some_and(nonempty)
                {
                    return Err(format!(
                        "abstained case {} requires an abstention_reason",
                        prediction.case_id
                    ));
                }
            }
        }
    }
    Ok(())
}

fn run(submission: SubmissionV1) -> Result<AuditReportV1, String> {
    validate_submission(&submission)?;
    let oracle = Manifest::parse()?;
    oracle.verify_group_axioms()?;

    let expected_case_ids = oracle.case_ids_for_split_track(
        &submission.split,
        &submission.benchmark_track,
    )?;
    let expected: BTreeSet<String> = expected_case_ids.iter().cloned().collect();
    let observed: BTreeSet<String> = submission
        .predictions
        .iter()
        .map(|prediction| prediction.case_id.clone())
        .collect();
    if expected != observed {
        let missing: Vec<String> = expected.difference(&observed).cloned().collect();
        let extra: Vec<String> = observed.difference(&expected).cloned().collect();
        return Err(format!(
            "submission coverage mismatch: missing={} extra={} (first missing: {:?}; first extra: {:?})",
            missing.len(),
            extra.len(),
            missing.first(),
            extra.first()
        ));
    }

    let prediction_by_id: BTreeMap<&str, &PredictionV1> = submission
        .predictions
        .iter()
        .map(|prediction| (prediction.case_id.as_str(), prediction))
        .collect();

    let mut cases = Vec::with_capacity(expected_case_ids.len());
    let mut predicted = 0usize;
    let mut abstained = 0usize;
    let mut exact_subgroup_matches = 0usize;
    let mut false_promotions = 0usize;
    let mut missed_symmetries = 0usize;
    let mut invalid_group_proposals = 0usize;

    for case_id in &expected_case_ids {
        let prediction = prediction_by_id[case_id.as_str()];
        let case = oracle.case_by_id(case_id)?;
        let exact_certificate = match submission.benchmark_track.as_str() {
            "exact_finite_structural_stabilizer" => oracle.structural_certificate(case)?,
            "finite_reference_group_regression_discovery" => oracle.polynomial_certificate(case)?,
            _ => unreachable!("track is validated above"),
        };
        let exact_stabilizer = exact_certificate.accepted_subgroup;

        match prediction.disposition {
            DispositionV1::Predicted => {
                predicted += 1;
                let audit = oracle.audit_proposed_subgroup(case, &prediction.proposed_elements)?;
                false_promotions += audit.false_promotion_elements.len();
                missed_symmetries += audit.missed_symmetry_elements.len();
                if !audit.group_axioms_satisfied {
                    invalid_group_proposals += 1;
                }
                let mut proposed = prediction.proposed_elements.clone();
                proposed.sort();
                let exact_subgroup_match = proposed == exact_stabilizer;
                exact_subgroup_matches += if exact_subgroup_match { 1 } else { 0 };
                cases.push(CaseAuditV1 {
                    case_id: case_id.clone(),
                    disposition: DispositionV1::Predicted,
                    abstention_reason: None,
                    exact_stabilizer,
                    exact_subgroup_match: Some(exact_subgroup_match),
                    proposal_audit: Some(audit),
                });
            }
            DispositionV1::Abstained => {
                abstained += 1;
                cases.push(CaseAuditV1 {
                    case_id: case_id.clone(),
                    disposition: DispositionV1::Abstained,
                    abstention_reason: prediction.abstention_reason.clone(),
                    exact_stabilizer,
                    exact_subgroup_match: None,
                    proposal_audit: None,
                });
            }
        }
    }

    Ok(AuditReportV1 {
        schema_version: "0.1.0".to_owned(),
        benchmark_id: BENCHMARK_ID.to_owned(),
        benchmark_track: submission.benchmark_track,
        split: submission.split,
        method: submission.method,
        cases,
        summary: SummaryV1 {
            cases_in_scope: expected_case_ids.len(),
            predicted,
            abstained,
            exact_subgroup_matches,
            false_promotions,
            missed_symmetries,
            invalid_group_proposals,
        },
    })
}

fn main() -> ExitCode {
    let Some(input_path) = env::args_os().nth(1) else {
        eprintln!("usage: audit-proposals <submission.json>");
        return ExitCode::from(2);
    };
    let input = match fs::read_to_string(&input_path) {
        Ok(input) => input,
        Err(error) => {
            eprintln!("PROPOSAL_AUDIT_INPUT_ERROR: {error}");
            return ExitCode::from(2);
        }
    };
    let submission: SubmissionV1 = match serde_json::from_str(&input) {
        Ok(submission) => submission,
        Err(error) => {
            eprintln!("PROPOSAL_AUDIT_JSON_ERROR: {error}");
            return ExitCode::from(2);
        }
    };
    match run(submission) {
        Ok(report) => match serde_json::to_string_pretty(&report) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("PROPOSAL_AUDIT_SERIALIZATION_ERROR: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("PROPOSAL_AUDIT_REJECTED: {error}");
            ExitCode::from(1)
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn smoke_submission() -> SubmissionV1 {
        serde_json::from_str(include_str!(
            "../../../fixtures/v0.1/proposal_oracle_selftest.json"
        ))
        .expect("frozen proposal smoke submission parses")
    }

    #[test]
    fn complete_discovery_submission_is_audited_against_every_frozen_case() {
        let report = run(smoke_submission()).expect("complete proposal submission audits");
        assert_eq!(report.summary.cases_in_scope, 300);
        assert_eq!(report.summary.predicted, 300);
        assert_eq!(report.summary.abstained, 0);
        assert_eq!(report.summary.exact_subgroup_matches, 300);
        assert_eq!(report.summary.false_promotions, 0);
        assert_eq!(report.summary.missed_symmetries, 0);
        assert_eq!(report.summary.invalid_group_proposals, 0);
        assert_eq!(report.cases.len(), 300);
    }

    #[test]
    fn incomplete_submission_is_rejected_instead_of_shrinking_the_denominator() {
        let mut submission = smoke_submission();
        submission.predictions.pop();
        let error = run(submission).unwrap_err();
        assert!(error.contains("submission coverage mismatch"));
        assert!(error.contains("missing=1"));
    }

    #[test]
    fn explicit_abstention_is_counted_and_does_not_become_a_false_negative() {
        let mut submission = smoke_submission();
        let prediction = &mut submission.predictions[0];
        prediction.disposition = DispositionV1::Abstained;
        prediction.proposed_elements.clear();
        prediction.abstention_reason = Some("synthetic adapter abstention test".to_owned());
        let report = run(submission).unwrap();
        assert_eq!(report.summary.cases_in_scope, 300);
        assert_eq!(report.summary.predicted, 299);
        assert_eq!(report.summary.abstained, 1);
        assert_eq!(report.summary.exact_subgroup_matches, 299);
        assert_eq!(report.summary.false_promotions, 0);
        assert_eq!(report.summary.missed_symmetries, 0);
    }

    #[test]
    fn unknown_proposed_group_elements_fail_closed() {
        let mut submission = smoke_submission();
        submission.predictions[0].proposed_elements[0] = "invented-element".to_owned();
        assert!(run(submission)
            .unwrap_err()
            .contains("unknown D4 element invented-element"));
    }
}
