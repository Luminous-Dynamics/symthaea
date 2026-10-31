//! Fail-closed validation for the frozen Symthaea architecture-engineering v0.1 matrix.
//!
//! This crate validates the *design fixture*, not experiment results. Measured results
//! belong in separate, provenance-bearing run receipts; this frozen matrix intentionally
//! requires every measurement to remain null and every record to remain NOT_RUN.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

use serde_json::Value;

pub mod paired_evaluation;
pub mod synthetic_search;
pub mod multi_fidelity;
pub mod multi_fidelity_benchmark;
pub mod analytical_benchmark;
pub mod pruning_audit;
pub mod budgeted_audit;
pub mod run_wide_pruning_risk;
pub mod decision_focused_voc;
pub mod voc_qualification;

const MATRIX_SCHEMA: &str = "symthaea.architecture-engineering.candidate-matrix";
const MATRIX_VERSION: &str = "0.1.0";
const TRACKING_ISSUE: &str = "https://github.com/Luminous-Dynamics/symthaea/issues/8888";
const EXPECTED_ARCHITECTURE_CANDIDATES: usize = 7;
const EXPECTED_SEARCH_POLICIES: usize = 3;
const EXPECTED_NEGATIVE_CONTROLS: usize = 3;

/// Summary returned only after the complete frozen matrix passes validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixSummary {
    pub schema: String,
    pub version: String,
    pub architecture_candidate_ids: Vec<String>,
    pub search_policy_ids: Vec<String>,
    pub negative_control_ids: Vec<String>,
}

/// A deterministic validation failure with a JSON-path-like location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationError {
    pub path: String,
    pub message: String,
}

impl ValidationError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for ValidationError {}

/// Validate the frozen v0.1 candidate matrix from UTF-8 JSON.
///
/// This validator intentionally rejects measured values in the design matrix. It prevents
/// the frozen candidate definitions from becoming an informal mutable leaderboard; measured
/// outcomes must be stored separately with run identity, raw outputs, costs, and uncertainty.
pub fn validate_frozen_candidate_matrix(input: &str) -> Result<MatrixSummary, ValidationError> {
    let root: Value = serde_json::from_str(input)
        .map_err(|error| ValidationError::new("$", format!("invalid JSON: {error}")))?;

    if !root.is_object() {
        return Err(ValidationError::new("$", "expected a JSON object"));
    }

    let schema = required_string(&root, "schema", "$")?;
    if schema != MATRIX_SCHEMA {
        return Err(ValidationError::new(
            "$.schema",
            format!("expected {MATRIX_SCHEMA:?}, found {schema:?}"),
        ));
    }

    let version = required_string(&root, "version", "$")?;
    if version != MATRIX_VERSION {
        return Err(ValidationError::new(
            "$.version",
            format!("this validator accepts frozen version {MATRIX_VERSION:?}, found {version:?}"),
        ));
    }

    let tracking_issue = required_string(&root, "tracking_issue", "$")?;
    if tracking_issue != TRACKING_ISSUE {
        return Err(ValidationError::new(
            "$.tracking_issue",
            format!("expected the canonical tracking issue {TRACKING_ISSUE:?}"),
        ));
    }

    let qualification = required_string(&root, "qualification_state", "$")?;
    if qualification != "DESIGN_ONLY_NOT_IMPLEMENTED_NOT_EXECUTED_NOT_QUALIFIED_NOT_PASS" {
        return Err(ValidationError::new(
            "$.qualification_state",
            "the v0.1 design fixture must retain its explicit not-qualified state",
        ));
    }

    let evaluation_policy = root
        .get("evaluation_policy")
        .and_then(Value::as_object)
        .ok_or_else(|| ValidationError::new("$.evaluation_policy", "expected an object"))?;
    if evaluation_policy
        .get("search_reads_final_test")
        .and_then(Value::as_bool)
        != Some(false)
    {
        return Err(ValidationError::new(
            "$.evaluation_policy.search_reads_final_test",
            "must be false; search may not read the locked final test set",
        ));
    }
    if evaluation_policy
        .get("null_means_not_measured")
        .and_then(Value::as_bool)
        != Some(true)
    {
        return Err(ValidationError::new(
            "$.evaluation_policy.null_means_not_measured",
            "must be true",
        ));
    }

    let mut all_ids = HashSet::new();
    let architecture_candidate_ids = validate_not_run_records(
        &root,
        "architecture_candidates",
        Some("measurements"),
        EXPECTED_ARCHITECTURE_CANDIDATES,
        &mut all_ids,
    )?;
    let architecture_id_set: HashSet<&str> = architecture_candidate_ids
        .iter()
        .map(String::as_str)
        .collect();

    // Every ablation control must refer to a defined architecture candidate and cannot
    // point to itself. Dangling IDs are usually a typo that would invalidate comparisons.
    let architecture_candidates = required_array(&root, "architecture_candidates", "$")?;
    for (index, candidate) in architecture_candidates.iter().enumerate() {
        let path = format!("$.architecture_candidates[{index}].control_ids");
        let controls = candidate
            .get("control_ids")
            .and_then(Value::as_array)
            .ok_or_else(|| ValidationError::new(&path, "expected an array"))?;
        let candidate_id = candidate
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| ValidationError::new(format!("$.architecture_candidates[{index}].id"), "expected a string"))?;
        for (control_index, control) in controls.iter().enumerate() {
            let control_id = control.as_str().ok_or_else(|| {
                ValidationError::new(format!("{path}[{control_index}]"), "expected a string ID")
            })?;
            if !architecture_id_set.contains(control_id) {
                return Err(ValidationError::new(
                    format!("{path}[{control_index}]"),
                    format!("unknown architecture control ID {control_id:?}"),
                ));
            }
            if control_id == candidate_id {
                return Err(ValidationError::new(
                    format!("{path}[{control_index}]"),
                    "a candidate cannot be its own control",
                ));
            }
        }
    }

    let search_policy_ids = validate_not_run_records(
        &root,
        "search_policy_candidates",
        Some("metrics"),
        EXPECTED_SEARCH_POLICIES,
        &mut all_ids,
    )?;
    let negative_control_ids = validate_not_run_records(
        &root,
        "negative_controls",
        None,
        EXPECTED_NEGATIVE_CONTROLS,
        &mut all_ids,
    )?;

    let references = required_array(&root, "research_references", "$")?;
    if references.is_empty()
        || references
            .iter()
            .any(|reference| reference.as_str().is_none_or(|value| !value.starts_with("https://")))
    {
        return Err(ValidationError::new(
            "$.research_references",
            "expected one or more HTTPS source URLs",
        ));
    }

    Ok(MatrixSummary {
        schema: schema.to_owned(),
        version: version.to_owned(),
        architecture_candidate_ids,
        search_policy_ids,
        negative_control_ids,
    })
}

fn required_string<'a>(
    root: &'a Value,
    key: &str,
    parent_path: &str,
) -> Result<&'a str, ValidationError> {
    root.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| ValidationError::new(format!("{parent_path}.{key}"), "expected a string"))
}

fn required_array<'a>(
    root: &'a Value,
    key: &str,
    parent_path: &str,
) -> Result<&'a Vec<Value>, ValidationError> {
    root.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| ValidationError::new(format!("{parent_path}.{key}"), "expected an array"))
}

fn validate_not_run_records(
    root: &Value,
    array_key: &str,
    measurement_key: Option<&str>,
    expected_count: usize,
    all_ids: &mut HashSet<String>,
) -> Result<Vec<String>, ValidationError> {
    let records = required_array(root, array_key, "$")?;
    if records.len() != expected_count {
        return Err(ValidationError::new(
            format!("$.{array_key}"),
            format!("expected {expected_count} frozen records, found {}", records.len()),
        ));
    }

    let mut ids = Vec::with_capacity(records.len());
    for (index, record) in records.iter().enumerate() {
        let path = format!("$.{array_key}[{index}]");
        if !record.is_object() {
            return Err(ValidationError::new(&path, "expected an object"));
        }

        let id = record
            .get("id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| ValidationError::new(format!("{path}.id"), "expected a non-empty string"))?;
        if !all_ids.insert(id.to_owned()) {
            return Err(ValidationError::new(
                format!("{path}.id"),
                format!("duplicate ID {id:?}"),
            ));
        }

        if record.get("status").and_then(Value::as_str) != Some("NOT_RUN") {
            return Err(ValidationError::new(
                format!("{path}.status"),
                "frozen v0.1 records must remain NOT_RUN",
            ));
        }

        if let Some(metric_key) = measurement_key {
            let metric_path = format!("{path}.{metric_key}");
            let metrics = record
                .get(metric_key)
                .and_then(Value::as_object)
                .ok_or_else(|| ValidationError::new(&metric_path, "expected an object"))?;
            if metrics.is_empty() {
                return Err(ValidationError::new(&metric_path, "expected declared metric fields"));
            }
            if let Some((name, _)) = metrics.iter().find(|(_, value)| !value.is_null()) {
                return Err(ValidationError::new(
                    format!("{metric_path}.{name}"),
                    "must remain null in the frozen design matrix; put measurements in a run receipt",
                ));
            }
        }

        ids.push(id.to_owned());
    }

    ids.sort_unstable();
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FROZEN_MATRIX: &str = include_str!(
        "../../../../docs/research/fixtures/ARCHITECTURE_ENGINEERING_CANDIDATES_V0_1.json"
    );

    #[test]
    fn validates_the_frozen_candidate_matrix() {
        let summary = validate_frozen_candidate_matrix(FROZEN_MATRIX)
            .expect("the committed v0.1 design matrix should validate");

        assert_eq!(summary.schema, MATRIX_SCHEMA);
        assert_eq!(summary.version, MATRIX_VERSION);
        assert_eq!(summary.architecture_candidate_ids.len(), EXPECTED_ARCHITECTURE_CANDIDATES);
        assert_eq!(summary.search_policy_ids.len(), EXPECTED_SEARCH_POLICIES);
        assert_eq!(summary.negative_control_ids.len(), EXPECTED_NEGATIVE_CONTROLS);
        assert!(summary.architecture_candidate_ids.contains(&"AE-BASE-01".to_owned()));
        assert!(summary.search_policy_ids.contains(&"SEARCH-RANDOM-01".to_owned()));
    }

    #[test]
    fn rejects_a_metric_inserted_into_the_frozen_design_matrix() {
        let mutated = FROZEN_MATRIX.replacen(
            "\"held_out_nll\": null",
            "\"held_out_nll\": 0.42",
            1,
        );
        let error = validate_frozen_candidate_matrix(&mutated)
            .expect_err("a frozen matrix must not become an unproven leaderboard");
        assert!(error.to_string().contains("must remain null"));
    }

    #[test]
    fn rejects_duplicate_candidate_ids() {
        let mutated = FROZEN_MATRIX.replacen(
            "\"id\": \"AE-LG-01\"",
            "\"id\": \"AE-BASE-01\"",
            1,
        );
        let error = validate_frozen_candidate_matrix(&mutated)
            .expect_err("duplicate candidate IDs must fail closed");
        assert!(error.to_string().contains("duplicate ID"));
    }

    #[test]
    fn rejects_search_access_to_the_locked_final_test_set() {
        let mutated = FROZEN_MATRIX.replace(
            "\"search_reads_final_test\": false",
            "\"search_reads_final_test\": true",
        );
        let error = validate_frozen_candidate_matrix(&mutated)
            .expect_err("search must not access the locked final test set");
        assert!(error.to_string().contains("search may not read"));
    }

    #[test]
    fn rejects_malformed_json() {
        let error = validate_frozen_candidate_matrix("{not-json")
            .expect_err("malformed JSON must fail closed");
        assert!(error.to_string().contains("invalid JSON"));
    }
}
