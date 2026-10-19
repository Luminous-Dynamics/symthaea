// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Synthetic-first expected-value-of-sample-information (EVSI) oracle.
//!
//! This crate ranks research plans by expected improvement in downstream decisions
//! minus experiment cost, not by assay accuracy, variance reduction, or yield alone.
//! It does not authorize experiments, predict biology, establish food safety, or validate a TEA/LCA.

pub mod robust;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

pub const RUNNER_VERSION: &str = "0.1.0";
pub const EXPECTED_CORPUS_ID: &str = "BIOPROTEIN-EXPERIMENT-VALUE-CORPUS-V1";
pub const EXPECTED_SCHEMA_VERSION: &str = "1.0.0";
pub const PROBABILITY_SCALE: u32 = 10_000;
const UTILITY_ABS_LIMIT: i64 = 1_000_000;
const CLAIM_CEILING: &str = "Synthetic decision-theory contract only; no biological outcome, assay validity, food safety, authorization, real production economics, or lifecycle superiority is established.";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResultStatus {
    Selected,
    NoPositiveEvsi,
    Blocked,
    Indeterminate,
    InvalidInput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Authorization {
    Permitted,
    Restricted,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceState {
    Complete,
    Incomplete,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub state_id: String,
    pub prior_weight: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Action {
    pub action_id: String,
    pub utilities: BTreeMap<String, i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeLikelihoods {
    pub state_id: String,
    pub outcomes: BTreeMap<String, u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Experiment {
    pub experiment_id: String,
    pub cost_utility: i64,
    pub authorization: Authorization,
    pub evidence_state: EvidenceState,
    pub scenario_digest: String,
    pub likelihoods: Vec<OutcomeLikelihoods>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Expected {
    pub status: ResultStatus,
    pub selected_experiment_id: Option<String>,
    pub reason_codes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub case_id: String,
    pub scenario_digest: String,
    pub probability_scale: u32,
    pub states: Vec<State>,
    pub actions: Vec<Action>,
    pub experiments: Vec<Experiment>,
    pub expected: Expected,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Corpus {
    pub corpus_id: String,
    pub schema_version: String,
    pub claim_ceiling: String,
    pub cases: Vec<Case>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentValue {
    /// EVSI numerator in utility * probability_scale^2 units, stored as a decimal string.
    pub experiment_id: String,
    pub gross_evsi_numerator: String,
    pub net_evsi_numerator: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExcludedExperiment {
    pub experiment_id: String,
    pub reason_code: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaseEvaluation {
    pub case_id: String,
    pub status: ResultStatus,
    pub selected_experiment_id: Option<String>,
    pub reason_codes: Vec<String>,
    pub experiment_values: Vec<ExperimentValue>,
    pub excluded_experiments: Vec<ExcludedExperiment>,
}

impl CaseEvaluation {
    pub fn matches_expected(&self, expected: &Expected) -> bool {
        self.status == expected.status
            && self.selected_experiment_id == expected.selected_experiment_id
            && self.reason_codes == expected.reason_codes
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaseReceipt {
    pub case_id: String,
    pub status: ResultStatus,
    pub selected_experiment_id: Option<String>,
    pub reason_codes: Vec<String>,
    pub experiment_values: Vec<ExperimentValue>,
    pub excluded_experiments: Vec<ExcludedExperiment>,
    pub expected_contract_match: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunReceipt {
    pub runner_version: String,
    pub corpus_id: String,
    pub schema_version: String,
    pub corpus_sha256: String,
    pub declared_head: String,
    pub actual_head: String,
    pub exact_head_match: bool,
    pub case_count: usize,
    pub matched_case_count: usize,
    pub failed_case_ids: Vec<String>,
    pub cases: Vec<CaseReceipt>,
    pub synthetic_only_claim_ceiling: String,
    pub contract_passed: bool,
}

fn validate_case(case: &Case) -> Vec<String> {
    let mut errors = Vec::new();
    if case.case_id.trim().is_empty() || case.scenario_digest.trim().is_empty() {
        errors.push("invalid_missing_case_identity".to_owned());
    }
    if case.probability_scale != PROBABILITY_SCALE {
        errors.push("invalid_probability_scale".to_owned());
    }
    if case.states.is_empty() {
        errors.push("invalid_empty_states".to_owned());
    }
    if case.actions.is_empty() {
        errors.push("invalid_empty_actions".to_owned());
    }

    let mut state_ids = BTreeSet::new();
    let mut prior_sum = 0_u64;
    for state in &case.states {
        if state.state_id.trim().is_empty() || !state_ids.insert(state.state_id.as_str()) {
            errors.push("invalid_duplicate_or_empty_state_id".to_owned());
        }
        prior_sum += u64::from(state.prior_weight);
    }
    if prior_sum != u64::from(PROBABILITY_SCALE) {
        errors.push("invalid_prior_sum".to_owned());
    }

    let mut action_ids = BTreeSet::new();
    for action in &case.actions {
        if action.action_id.trim().is_empty() || !action_ids.insert(action.action_id.as_str()) {
            errors.push("invalid_duplicate_or_empty_action_id".to_owned());
        }
        let utility_states: BTreeSet<&str> = action.utilities.keys().map(String::as_str).collect();
        if utility_states != state_ids {
            errors.push("invalid_action_utility_coverage".to_owned());
        }
        if action
            .utilities
            .values()
            .any(|value| value.unsigned_abs() > UTILITY_ABS_LIMIT as u64)
        {
            errors.push("invalid_utility_bound".to_owned());
        }
    }

    let mut experiment_ids = BTreeSet::new();
    for experiment in &case.experiments {
        if experiment.experiment_id.trim().is_empty()
            || !experiment_ids.insert(experiment.experiment_id.as_str())
        {
            errors.push("invalid_duplicate_or_empty_experiment_id".to_owned());
        }
        if experiment.cost_utility < 0 || experiment.cost_utility > UTILITY_ABS_LIMIT {
            errors.push("invalid_experiment_cost".to_owned());
        }
        if experiment.scenario_digest.trim().is_empty() {
            errors.push("invalid_missing_experiment_scenario".to_owned());
        }

        let mut seen_likelihood_states = BTreeSet::new();
        for row in &experiment.likelihoods {
            if !state_ids.contains(row.state_id.as_str())
                || !seen_likelihood_states.insert(row.state_id.as_str())
            {
                errors.push("invalid_likelihood_state_coverage".to_owned());
            }
        }
        if seen_likelihood_states != state_ids {
            errors.push("invalid_likelihood_state_coverage".to_owned());
        }

        let outcome_sets: Vec<BTreeSet<&str>> = experiment
            .likelihoods
            .iter()
            .map(|row| row.outcomes.keys().map(String::as_str).collect())
            .collect();
        if outcome_sets.is_empty() || outcome_sets.iter().any(|set| set != &outcome_sets[0]) {
            errors.push("invalid_outcome_coverage".to_owned());
        }
        for row in &experiment.likelihoods {
            let sum: u64 = row.outcomes.values().map(|weight| u64::from(*weight)).sum();
            if row.outcomes.is_empty() || sum != u64::from(PROBABILITY_SCALE) {
                errors.push("invalid_outcome_probability_sum".to_owned());
            }
        }
    }

    errors.sort();
    errors.dedup();
    errors
}

fn excluded(experiment: &Experiment, reason_code: &str) -> ExcludedExperiment {
    ExcludedExperiment {
        experiment_id: experiment.experiment_id.clone(),
        reason_code: reason_code.to_owned(),
    }
}

fn invalid_evaluation(case: &Case, reason_codes: Vec<String>) -> CaseEvaluation {
    CaseEvaluation {
        case_id: case.case_id.clone(),
        status: ResultStatus::InvalidInput,
        selected_experiment_id: None,
        reason_codes,
        experiment_values: Vec::new(),
        excluded_experiments: Vec::new(),
    }
}

/// Exact finite EVSI with integer probabilities.
/// A gross EVSI numerator uses denominator probability_scale squared.
/// Utility and experiment cost share the same declared abstract unit.
pub fn evaluate_case(case: &Case) -> CaseEvaluation {
    let errors = validate_case(case);
    if !errors.is_empty() {
        return invalid_evaluation(case, errors);
    }

    let scale = i128::from(PROBABILITY_SCALE);
    let scale_squared = scale * scale;

    // Expected utility without new information, numerator denominator = scale.
    let baseline_numerator = case
        .actions
        .iter()
        .map(|action| {
            case.states.iter().fold(0_i128, |sum, state| {
                sum + i128::from(state.prior_weight) * i128::from(action.utilities[&state.state_id])
            })
        })
        .max()
        .unwrap_or(0);

    let mut values = Vec::<(String, i128, i128)>::new();
    let mut excluded_experiments = Vec::new();
    let mut unresolved_gate_seen = false;

    for experiment in &case.experiments {
        if experiment.authorization == Authorization::Restricted {
            excluded_experiments.push(excluded(experiment, "authorization_restricted"));
            continue;
        }
        if experiment.evidence_state == EvidenceState::Incomplete {
            excluded_experiments.push(excluded(experiment, "experiment_evidence_incomplete"));
            continue;
        }
        if experiment.scenario_digest != case.scenario_digest {
            excluded_experiments.push(excluded(experiment, "scenario_mismatch"));
            continue;
        }
        if experiment.authorization == Authorization::Unknown {
            unresolved_gate_seen = true;
            excluded_experiments.push(excluded(experiment, "authorization_unknown"));
            continue;
        }
        if experiment.evidence_state == EvidenceState::Unknown {
            unresolved_gate_seen = true;
            excluded_experiments.push(excluded(experiment, "experiment_evidence_unknown"));
            continue;
        }

        let rows_by_state: BTreeMap<&str, &OutcomeLikelihoods> = experiment
            .likelihoods
            .iter()
            .map(|row| (row.state_id.as_str(), row))
            .collect();
        let mut outcome_names: Vec<&str> = experiment.likelihoods[0]
            .outcomes
            .keys()
            .map(String::as_str)
            .collect();
        outcome_names.sort_unstable();

        // Sum over possible outcomes, each taking the best action for that outcome.
        let post_information_numerator = outcome_names.iter().fold(0_i128, |outer_sum, outcome| {
            let best_action_numerator = case
                .actions
                .iter()
                .map(|action| {
                    case.states.iter().fold(0_i128, |inner_sum, state| {
                        let likelihood = rows_by_state[state.state_id.as_str()].outcomes[*outcome];
                        inner_sum
                            + i128::from(state.prior_weight)
                                * i128::from(likelihood)
                                * i128::from(action.utilities[&state.state_id])
                    })
                })
                .max()
                .unwrap_or(0);
            outer_sum + best_action_numerator
        });

        let gross_evsi_numerator = post_information_numerator - baseline_numerator * scale;
        let net_evsi_numerator =
            gross_evsi_numerator - i128::from(experiment.cost_utility) * scale_squared;
        values.push((
            experiment.experiment_id.clone(),
            gross_evsi_numerator,
            net_evsi_numerator,
        ));
    }

    values.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
    excluded_experiments.sort_by(|a, b| a.experiment_id.cmp(&b.experiment_id));
    let experiment_values: Vec<ExperimentValue> = values
        .iter()
        .map(|(id, gross, net)| ExperimentValue {
            experiment_id: id.clone(),
            gross_evsi_numerator: gross.to_string(),
            net_evsi_numerator: net.to_string(),
        })
        .collect();

    if let Some((experiment_id, _, _)) = values.iter().find(|(_, _, net)| *net > 0) {
        return CaseEvaluation {
            case_id: case.case_id.clone(),
            status: ResultStatus::Selected,
            selected_experiment_id: Some(experiment_id.clone()),
            reason_codes: vec![
                "positive_net_evsi".to_owned(),
                "highest_net_evsi_selected".to_owned(),
            ],
            experiment_values,
            excluded_experiments,
        };
    }

    if unresolved_gate_seen {
        return CaseEvaluation {
            case_id: case.case_id.clone(),
            status: ResultStatus::Indeterminate,
            selected_experiment_id: None,
            reason_codes: vec!["unresolved_experiment_gate_may_change_decision".to_owned()],
            experiment_values,
            excluded_experiments,
        };
    }

    if experiment_values.is_empty() {
        CaseEvaluation {
            case_id: case.case_id.clone(),
            status: ResultStatus::Blocked,
            selected_experiment_id: None,
            reason_codes: vec!["no_eligible_experiment".to_owned()],
            experiment_values,
            excluded_experiments,
        }
    } else {
        CaseEvaluation {
            case_id: case.case_id.clone(),
            status: ResultStatus::NoPositiveEvsi,
            selected_experiment_id: None,
            reason_codes: vec!["no_positive_net_evsi".to_owned()],
            experiment_values,
            excluded_experiments,
        }
    }
}

pub fn evaluate_corpus(corpus: &Corpus) -> Vec<CaseEvaluation> {
    corpus.cases.iter().map(evaluate_case).collect()
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn run_cli() -> i32 {
    let mut expected_head: Option<String> = None;
    let mut receipt_path: Option<PathBuf> = None;
    let mut corpus_path: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--expected-head" => expected_head = args.next(),
            "--receipt" => receipt_path = args.next().map(PathBuf::from),
            "--corpus" => corpus_path = args.next().map(PathBuf::from),
            _ => {
                eprintln!("unknown argument: {arg}");
                return 2;
            }
        }
    }
    let Some(expected_head) = expected_head else {
        eprintln!("missing --expected-head");
        return 2;
    };
    let Some(receipt_path) = receipt_path else {
        eprintln!("missing --receipt");
        return 2;
    };
    let Some(corpus_path) = corpus_path else {
        eprintln!("missing --corpus");
        return 2;
    };

    let actual_head = match Command::new("git").args(["rev-parse", "HEAD"]).output() {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).trim().to_owned()
        }
        _ => {
            eprintln!("could not establish actual Git HEAD");
            return 2;
        }
    };
    if actual_head != expected_head {
        eprintln!("exact-head mismatch: expected={expected_head} actual={actual_head}");
        return 2;
    }

    let corpus_bytes = match fs::read(&corpus_path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("cannot read corpus {}: {error}", corpus_path.display());
            return 2;
        }
    };
    let corpus: Corpus = match serde_json::from_slice(&corpus_bytes) {
        Ok(corpus) => corpus,
        Err(error) => {
            eprintln!("cannot parse corpus: {error}");
            return 2;
        }
    };
    if corpus.corpus_id != EXPECTED_CORPUS_ID || corpus.schema_version != EXPECTED_SCHEMA_VERSION {
        eprintln!("unexpected corpus ID or schema version");
        return 2;
    }

    let results = evaluate_corpus(&corpus);
    let cases: Vec<CaseReceipt> = corpus
        .cases
        .iter()
        .zip(results.iter())
        .map(|(case, result)| CaseReceipt {
            case_id: result.case_id.clone(),
            status: result.status,
            selected_experiment_id: result.selected_experiment_id.clone(),
            reason_codes: result.reason_codes.clone(),
            experiment_values: result.experiment_values.clone(),
            excluded_experiments: result.excluded_experiments.clone(),
            expected_contract_match: result.matches_expected(&case.expected),
        })
        .collect();
    let failed_case_ids: Vec<String> = cases
        .iter()
        .filter(|receipt| !receipt.expected_contract_match)
        .map(|receipt| receipt.case_id.clone())
        .collect();
    let matched_case_count = cases.len() - failed_case_ids.len();
    let contract_passed = failed_case_ids.is_empty();
    let exact_head_match = actual_head == expected_head;
    let receipt = RunReceipt {
        runner_version: RUNNER_VERSION.to_owned(),
        corpus_id: corpus.corpus_id,
        schema_version: corpus.schema_version,
        corpus_sha256: sha256_hex(&corpus_bytes),
        declared_head: expected_head,
        exact_head_match,
        actual_head,
        case_count: cases.len(),
        matched_case_count,
        failed_case_ids,
        cases,
        synthetic_only_claim_ceiling: CLAIM_CEILING.to_owned(),
        contract_passed,
    };
    if let Some(parent) = receipt_path.parent()
        && let Err(error) = fs::create_dir_all(parent)
    {
        eprintln!("cannot create receipt directory: {error}");
        return 2;
    }
    let receipt_json = match serde_json::to_vec_pretty(&receipt) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("cannot serialize receipt: {error}");
            return 2;
        }
    };
    if let Err(error) = fs::write(&receipt_path, &receipt_json) {
        eprintln!("cannot write receipt {}: {error}", receipt_path.display());
        return 2;
    }
    print!("{}", String::from_utf8_lossy(&receipt_json));
    println!();
    if contract_passed { 0 } else { 1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CORPUS_JSON: &str = include_str!(
        "../../../docs/research/bioprotein-experiments/experiment-value-corpus-v1.json"
    );

    fn fixture() -> Corpus {
        serde_json::from_str(CORPUS_JSON).expect("frozen experiment-value corpus parses")
    }

    #[test]
    fn frozen_corpus_matches_every_expected_disposition_and_reason() {
        let corpus = fixture();
        assert_eq!(corpus.corpus_id, EXPECTED_CORPUS_ID);
        assert_eq!(corpus.schema_version, EXPECTED_SCHEMA_VERSION);
        let results = evaluate_corpus(&corpus);
        assert_eq!(results.len(), corpus.cases.len());
        for (case, result) in corpus.cases.iter().zip(results.iter()) {
            assert!(
                result.matches_expected(&case.expected),
                "case {} mismatch: expected {:?}, actual {:?}",
                case.case_id,
                case.expected,
                result
            );
        }
    }

    #[test]
    fn illustrative_positive_case_has_exact_integer_value() {
        let case = fixture()
            .cases
            .into_iter()
            .find(|case| case.case_id == "EVSI-001")
            .expect("EVSI-001 exists");
        let result = evaluate_case(&case);
        assert_eq!(result.status, ResultStatus::Selected);
        assert_eq!(result.selected_experiment_id.as_deref(), Some("assay-a"));
        assert_eq!(
            result.experiment_values[0].gross_evsi_numerator,
            "350000000"
        );
        assert_eq!(result.experiment_values[0].net_evsi_numerator, "250000000");
    }

    #[test]
    fn unknown_authorization_is_never_treated_as_permission() {
        let mut case = fixture()
            .cases
            .into_iter()
            .find(|case| case.case_id == "EVSI-005")
            .expect("EVSI-005 exists");
        case.experiments[0].authorization = Authorization::Unknown;
        let result = evaluate_case(&case);
        assert_eq!(result.status, ResultStatus::Indeterminate);
        assert_eq!(result.selected_experiment_id, None);
        assert_eq!(
            result.excluded_experiments[0].reason_code,
            "authorization_unknown"
        );
    }

    #[test]
    fn experiment_order_does_not_change_winner() {
        let mut case = fixture()
            .cases
            .into_iter()
            .find(|case| case.case_id == "EVSI-007")
            .expect("EVSI-007 exists");
        let first = evaluate_case(&case);
        case.experiments.reverse();
        let second = evaluate_case(&case);
        assert_eq!(first.status, second.status);
        assert_eq!(first.selected_experiment_id, second.selected_experiment_id);
        assert_eq!(first.reason_codes, second.reason_codes);
        assert_eq!(first.selected_experiment_id.as_deref(), Some("assay-a"));
    }

    #[test]
    fn independent_test_outcomes_have_zero_gross_evsi() {
        let case = fixture()
            .cases
            .into_iter()
            .find(|case| case.case_id == "EVSI-003")
            .expect("EVSI-003 exists");
        let result = evaluate_case(&case);
        assert_eq!(result.status, ResultStatus::NoPositiveEvsi);
        assert_eq!(result.experiment_values[0].gross_evsi_numerator, "0");
    }

    #[test]
    fn state_action_and_likelihood_row_order_do_not_change_values() {
        let mut case = fixture()
            .cases
            .into_iter()
            .find(|case| case.case_id == "EVSI-006")
            .expect("EVSI-006 exists");
        let before = evaluate_case(&case);
        case.states.reverse();
        case.actions.reverse();
        for experiment in &mut case.experiments {
            experiment.likelihoods.reverse();
        }
        let after = evaluate_case(&case);
        assert_eq!(before.status, after.status);
        assert_eq!(before.selected_experiment_id, after.selected_experiment_id);
        assert_eq!(before.experiment_values, after.experiment_values);
    }

    #[test]
    fn extreme_signed_utility_is_rejected_without_overflow() {
        let mut case = fixture()
            .cases
            .into_iter()
            .find(|case| case.case_id == "EVSI-001")
            .expect("EVSI-001 exists");
        case.actions[0]
            .utilities
            .insert("favorable".to_owned(), i64::MIN);
        let result = evaluate_case(&case);
        assert_eq!(result.status, ResultStatus::InvalidInput);
        assert!(
            result
                .reason_codes
                .iter()
                .any(|reason| reason == "invalid_utility_bound")
        );
    }
}
