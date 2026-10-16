// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dependency-free integrity checks for the proposed INTX-PROGRAM-001 fixture corpus.
//!
//! This validates fixture shape and stated safety invariants only. It does not execute
//! Symthaea's action runtime, prove connector correctness, or qualify physical equipment.

use std::collections::{BTreeMap, BTreeSet};

pub const CORPUS: &str = include_str!("../fixtures/cases.tsv");

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Case {
    pub id: String,
    pub dimension: String,
    pub scenario: String,
    pub expected_state: String,
    pub dispatch: String,
    pub reconcile: String,
}

pub fn parse_cases(input: &str) -> Result<Vec<Case>, String> {
    let mut lines = input.lines();
    let header = lines.next().ok_or_else(|| "missing header".to_string())?;
    if header != "id|dimension|scenario|expected_state|dispatch|reconcile" {
        return Err("unexpected corpus header".to_string());
    }

    let mut cases = Vec::new();
    for (offset, line) in lines.enumerate() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('|').collect();
        if fields.len() != 6 {
            return Err(format!("line {}: expected 6 pipe-delimited fields", offset + 2));
        }
        if fields.iter().any(|field| field.trim().is_empty()) {
            return Err(format!("line {}: empty field", offset + 2));
        }
        if fields.iter().any(|field| field.contains(',') || field.contains('\t')) {
            return Err(format!(
                "line {}: fields must use stable identifiers, not embedded delimiters",
                offset + 2
            ));
        }
        cases.push(Case {
            id: fields[0].to_string(),
            dimension: fields[1].to_string(),
            scenario: fields[2].to_string(),
            expected_state: fields[3].to_string(),
            dispatch: fields[4].to_string(),
            reconcile: fields[5].to_string(),
        });
    }
    Ok(cases)
}

pub fn validate_cases(cases: &[Case]) -> Result<(), String> {
    if cases.len() != 50 {
        return Err(format!("expected 50 cases, got {}", cases.len()));
    }

    let mut ids = BTreeSet::new();
    let mut dimensions = BTreeMap::<&str, usize>::new();
    let valid_dimensions = ["identity", "artifact", "authority", "transport", "lifecycle"];
    let valid_states = [
        "denied",
        "observation_admitted",
        "release_blocked",
        "artifact_exported",
        "offline_validation_passed",
        "preflight_admitted",
        "not_dispatched",
        "outcome_unknown",
        "remote_rejected",
        "deduplicated",
        "discrepancy",
        "synthetic_effect_verified",
        "requalification_required",
    ];

    for case in cases {
        if !ids.insert(&case.id) {
            return Err(format!("duplicate case id: {}", case.id));
        }
        if !valid_dimensions.contains(&case.dimension.as_str()) {
            return Err(format!("unknown dimension: {}", case.dimension));
        }
        if !valid_states.contains(&case.expected_state.as_str()) {
            return Err(format!("unknown expected state: {}", case.expected_state));
        }
        if !["none", "occurred", "may_have_occurred"].contains(&case.dispatch.as_str()) {
            return Err(format!("unknown dispatch class for {}", case.id));
        }
        if !["yes", "no"].contains(&case.reconcile.as_str()) {
            return Err(format!("invalid reconcile value for {}", case.id));
        }
        if !case.id.starts_with(&format!("{}-", match case.dimension.as_str() {
            "identity" => "C0",
            "artifact" => "C1",
            "authority" => "C2",
            "transport" => "C3",
            "lifecycle" => "C4",
            _ => unreachable!("dimension was validated"),
        })) {
            return Err(format!("case id and dimension disagree: {}", case.id));
        }

        *dimensions.entry(case.dimension.as_str()).or_default() += 1;

        if ["denied", "release_blocked", "requalification_required", "not_dispatched"]
            .contains(&case.expected_state.as_str())
            && case.dispatch != "none"
        {
            return Err(format!("blocked case could dispatch: {}", case.id));
        }
        if case.expected_state == "outcome_unknown"
            && (case.dispatch != "may_have_occurred" || case.reconcile != "yes")
        {
            return Err(format!("ambiguous outcome must require reconciliation: {}", case.id));
        }
        if case.expected_state == "discrepancy"
            && (case.dispatch != "occurred" || case.reconcile != "yes")
        {
            return Err(format!("discrepancy must reconcile after dispatch: {}", case.id));
        }
        if case.expected_state == "synthetic_effect_verified"
            && (case.dispatch != "occurred" || case.reconcile != "yes")
        {
            return Err(format!(
                "synthetic verification requires dispatch and reconciliation evidence: {}",
                case.id
            ));
        }
    }

    for dimension in valid_dimensions {
        let count = dimensions.get(dimension).copied().unwrap_or_default();
        if count != 10 {
            return Err(format!("dimension {dimension} should contain 10 cases, got {count}"));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_corpus_has_fifty_unique_cases_across_five_dimensions() {
        let cases = parse_cases(CORPUS).expect("fixture corpus parses");
        validate_cases(&cases).expect("fixture corpus satisfies shape and safety invariants");
        assert_eq!(cases.first().map(|case| case.id.as_str()), Some("C0-01"));
        assert_eq!(cases.last().map(|case| case.id.as_str()), Some("C4-10"));
    }

    #[test]
    fn ambiguous_outcomes_cannot_be_misclassified_as_failures() {
        let mut cases = parse_cases(CORPUS).expect("fixture corpus parses");
        let case = cases
            .iter_mut()
            .find(|case| case.id == "C3-02")
            .expect("ambiguous timeout fixture exists");
        case.expected_state = "remote_rejected".to_string();
        assert!(validate_cases(&cases).is_err());
    }

    #[test]
    fn blocked_states_cannot_request_dispatch() {
        let mut cases = parse_cases(CORPUS).expect("fixture corpus parses");
        let case = cases
            .iter_mut()
            .find(|case| case.id == "C2-05")
            .expect("safety-invariant fixture exists");
        case.dispatch = "occurred".to_string();
        assert!(validate_cases(&cases).is_err());
    }

    #[test]
    fn changed_case_identity_cannot_be_silently_accepted() {
        let mut cases = parse_cases(CORPUS).expect("fixture corpus parses");
        let case = cases
            .iter_mut()
            .find(|case| case.id == "C1-01")
            .expect("artifact fixture exists");
        case.id = "C9-99".to_string();
        assert!(validate_cases(&cases).is_err());
    }

    #[test]
    fn malformed_fixture_rows_fail_closed() {
        assert!(parse_cases("id|dimension|scenario|expected_state|dispatch|reconcile\nC0-01|identity").is_err());
    }
}
