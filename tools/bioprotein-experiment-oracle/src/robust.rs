// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Robust EVSI stress suite over explicit joint uncertainty scenarios.
//!
//! Scenarios bundle dependent assumptions rather than factorizing every input as
//! independent. Results are stress-test summaries, not calibrated beliefs, real TEA/LCA,
//! experiment authorization, biological prediction, or proof of robustness outside
//! the declared scenario set.

use super::{evaluate_case, sha256_hex, Case, CaseEvaluation, ResultStatus};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

pub const ROBUST_RUNNER_VERSION: &str = "0.1.0";
pub const ROBUST_CORPUS_ID: &str = "BIOPROTEIN-ROBUST-EVSI-CORPUS-V1";
pub const ROBUST_SCHEMA_VERSION: &str = "1.0.0";
pub const SCENARIO_WEIGHT_SCALE: u32 = 10_000;
const CLAIM_CEILING: &str = "Synthetic robustness stress test only. Scenario weights and likelihoods are invented fixture values, not calibrated probabilities. No biological result, food safety, authorization, real economics, environmental performance, or out-of-sample robustness is established.";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorrelationModel {
    JointScenarioSet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UncertaintyClass {
    ParameterShift,
    CorrelatedJointShock,
    StructuralModelAlternative,
    DistributionShift,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RobustStatus {
    RobustWinner,
    ScenarioSensitive,
    NoPositiveEvsi,
    Indeterminate,
    InvalidInput,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobustExpected {
    pub status: RobustStatus,
    pub robust_winner: Option<String>,
    pub scenario_winners: Vec<Option<String>>,
    pub reason_codes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobustScenario {
    pub scenario_id: String,
    pub weight: u32,
    pub uncertainty_class: UncertaintyClass,
    pub assumption_tags: Vec<String>,
    pub case: Case,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobustFamily {
    pub family_id: String,
    pub correlation_model: CorrelationModel,
    pub decision_contract_digest: String,
    pub scenarios: Vec<RobustScenario>,
    pub expected: RobustExpected,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobustCorpus {
    pub corpus_id: String,
    pub schema_version: String,
    pub scenario_weight_scale: u32,
    pub family_count: usize,
    pub claim_ceiling: String,
    pub families: Vec<RobustFamily>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScenarioResult {
    pub scenario_id: String,
    pub weight: u32,
    pub status: ResultStatus,
    pub selected_experiment_id: Option<String>,
    pub reason_codes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateRobustness {
    pub experiment_id: String,
    pub eligible_scenario_count: usize,
    pub eligible_scenario_weight: u64,
    /// Exact sum of scenario_weight multiplied by net EVSI numerator over eligible scenarios.
    /// Divide by eligible_scenario_weight for the exact weighted mean numerator.
    pub weighted_net_evsi_numerator: String,
    pub minimum_net_evsi_numerator: Option<String>,
    pub maximum_net_evsi_numerator: Option<String>,
    pub positive_scenario_weight: u64,
    pub nonpositive_scenario_weight: u64,
    pub eligible_in_all_scenarios: bool,
    pub positive_in_all_scenarios: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobustEvaluation {
    pub family_id: String,
    pub status: RobustStatus,
    pub robust_winner: Option<String>,
    pub reason_codes: Vec<String>,
    pub scenario_results: Vec<ScenarioResult>,
    pub candidates: Vec<CandidateRobustness>,
}

impl RobustEvaluation {
    pub fn matches_expected(&self, expected: &RobustExpected) -> bool {
        self.status == expected.status
            && self.robust_winner == expected.robust_winner
            && self.scenario_results
                .iter()
                .map(|result| result.selected_experiment_id.clone())
                .collect::<Vec<_>>()
                == expected.scenario_winners
            && self.reason_codes == expected.reason_codes
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobustFamilyReceipt {
    pub family_id: String,
    pub evaluation: RobustEvaluation,
    pub expected_contract_match: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobustRunReceipt {
    pub runner_version: String,
    pub corpus_id: String,
    pub schema_version: String,
    pub corpus_sha256: String,
    pub declared_head: String,
    pub actual_head: String,
    pub exact_head_match: bool,
    pub family_count: usize,
    pub matched_family_count: usize,
    pub failed_family_ids: Vec<String>,
    pub families: Vec<RobustFamilyReceipt>,
    pub synthetic_only_claim_ceiling: String,
    pub contract_passed: bool,
}

fn invalid_family(family_id: &str, reason: &str) -> RobustEvaluation {
    RobustEvaluation {
        family_id: family_id.to_owned(),
        status: RobustStatus::InvalidInput,
        robust_winner: None,
        reason_codes: vec![reason.to_owned()],
        scenario_results: Vec::new(),
        candidates: Vec::new(),
    }
}

fn action_signature(case: &Case) -> BTreeMap<String, BTreeMap<String, i64>> {
    case.actions
        .iter()
        .map(|action| (action.action_id.clone(), action.utilities.clone()))
        .collect()
}

fn state_signature(case: &Case) -> BTreeSet<String> {
    case.states.iter().map(|state| state.state_id.clone()).collect()
}

fn experiment_signature(case: &Case) -> BTreeSet<String> {
    case.experiments
        .iter()
        .map(|experiment| experiment.experiment_id.clone())
        .collect()
}

fn validate_family(family: &RobustFamily) -> Vec<String> {
    let mut errors = Vec::new();
    if family.family_id.trim().is_empty() || family.decision_contract_digest.trim().is_empty() {
        errors.push("family_identity_missing".to_owned());
    }
    if family.scenarios.len() < 2 {
        errors.push("too_few_scenarios".to_owned());
    }

    let mut scenario_ids = BTreeSet::new();
    let mut case_ids = BTreeSet::new();
    let mut scenario_digests = BTreeSet::new();
    let mut weight_sum = 0_u64;
    for scenario in &family.scenarios {
        if scenario.scenario_id.trim().is_empty()
            || !scenario_ids.insert(scenario.scenario_id.as_str())
        {
            errors.push("duplicate_or_empty_scenario_id".to_owned());
        }
        if scenario.case.case_id.trim().is_empty()
            || !case_ids.insert(scenario.case.case_id.as_str())
        {
            errors.push("duplicate_or_empty_case_id".to_owned());
        }
        if scenario.weight == 0 || scenario.weight > SCENARIO_WEIGHT_SCALE {
            errors.push("invalid_scenario_weight".to_owned());
        }
        weight_sum += u64::from(scenario.weight);
        if scenario.assumption_tags.is_empty()
            || scenario.assumption_tags.iter().any(|tag| tag.trim().is_empty())
        {
            errors.push("scenario_assumption_tags_missing".to_owned());
        }
        if !scenario_digests.insert(scenario.case.scenario_digest.as_str()) {
            errors.push("duplicate_scenario_digest".to_owned());
        }
    }
    if weight_sum != u64::from(SCENARIO_WEIGHT_SCALE) {
        errors.push("scenario_weight_sum_mismatch".to_owned());
    }

    if let Some(reference) = family.scenarios.first() {
        let reference_actions = action_signature(&reference.case);
        let reference_states = state_signature(&reference.case);
        let reference_experiments = experiment_signature(&reference.case);
        for scenario in &family.scenarios {
            if action_signature(&scenario.case) != reference_actions {
                errors.push("decision_utility_contract_mismatch".to_owned());
            }
            if state_signature(&scenario.case) != reference_states {
                errors.push("decision_state_contract_mismatch".to_owned());
            }
            if experiment_signature(&scenario.case) != reference_experiments {
                errors.push("candidate_set_mismatch".to_owned());
            }
        }
    }

    errors.sort();
    errors.dedup();
    errors
}

fn invalid_reason_for_evaluation(evaluation: &CaseEvaluation) -> Option<&str> {
    if evaluation.status == ResultStatus::InvalidInput {
        Some(
            evaluation
                .reason_codes
                .first()
                .map(String::as_str)
                .unwrap_or("invalid_case"),
        )
    } else {
        None
    }
}

pub fn evaluate_family(family: &RobustFamily) -> RobustEvaluation {
    let errors = validate_family(family);
    if let Some(reason) = errors.first() {
        return invalid_family(&family.family_id, reason);
    }

    let mut scenarios: Vec<&RobustScenario> = family.scenarios.iter().collect();
    scenarios.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));

    let evaluated: Vec<(&RobustScenario, CaseEvaluation)> = scenarios
        .into_iter()
        .map(|scenario| (scenario, evaluate_case(&scenario.case)))
        .collect();

    for (_, evaluation) in &evaluated {
        if let Some(reason) = invalid_reason_for_evaluation(evaluation) {
            return invalid_family(&family.family_id, reason);
        }
    }

    let scenario_results: Vec<ScenarioResult> = evaluated
        .iter()
        .map(|(scenario, evaluation)| ScenarioResult {
            scenario_id: scenario.scenario_id.clone(),
            weight: scenario.weight,
            status: evaluation.status,
            selected_experiment_id: evaluation.selected_experiment_id.clone(),
            reason_codes: evaluation.reason_codes.clone(),
        })
        .collect();

    let all_selected_same = scenario_results.first().is_some_and(|first| {
        first.status == ResultStatus::Selected
            && first.selected_experiment_id.is_some()
            && scenario_results.iter().all(|result| {
                result.status == ResultStatus::Selected
                    && result.selected_experiment_id == first.selected_experiment_id
            })
    });

    let status = if all_selected_same {
        RobustStatus::RobustWinner
    } else if scenario_results.iter().any(|result| {
        result.status == ResultStatus::Indeterminate || result.status == ResultStatus::Blocked
    }) {
        RobustStatus::Indeterminate
    } else if scenario_results
        .iter()
        .all(|result| result.status == ResultStatus::NoPositiveEvsi)
    {
        RobustStatus::NoPositiveEvsi
    } else {
        RobustStatus::ScenarioSensitive
    };

    let (robust_winner, reason_codes) = match status {
        RobustStatus::RobustWinner => (
            scenario_results[0].selected_experiment_id.clone(),
            vec!["winner_stable_across_joint_scenarios".to_owned()],
        ),
        RobustStatus::ScenarioSensitive => (
            None,
            vec!["decision_not_stable_across_joint_scenarios".to_owned()],
        ),
        RobustStatus::NoPositiveEvsi => (
            None,
            vec!["no_positive_net_evsi_in_any_scenario".to_owned()],
        ),
        RobustStatus::Indeterminate => (
            None,
            vec!["unresolved_hard_gate_or_evidence_in_at_least_one_scenario".to_owned()],
        ),
        RobustStatus::InvalidInput => unreachable!("invalid scenario exits before family classification"),
    };

    let candidate_ids: Vec<String> = family
        .scenarios
        .first()
        .map(|scenario| {
            let mut ids: Vec<String> = scenario
                .case
                .experiments
                .iter()
                .map(|experiment| experiment.experiment_id.clone())
                .collect();
            ids.sort();
            ids
        })
        .unwrap_or_default();

    let mut candidates = Vec::with_capacity(candidate_ids.len());
    for experiment_id in candidate_ids {
        let mut weighted_sum = 0_i128;
        let mut eligible_scenario_weight = 0_u64;
        let mut eligible_scenario_count = 0_usize;
        let mut positive_scenario_weight = 0_u64;
        let mut nonpositive_scenario_weight = 0_u64;
        let mut minimum: Option<i128> = None;
        let mut maximum: Option<i128> = None;
        let mut numeric_error = false;

        for (scenario, evaluation) in &evaluated {
            let value = evaluation
                .experiment_values
                .iter()
                .find(|value| value.experiment_id == experiment_id);
            let Some(value) = value else {
                continue;
            };
            let Ok(net) = value.net_evsi_numerator.parse::<i128>() else {
                numeric_error = true;
                break;
            };
            let Some(weighted) = i128::from(scenario.weight).checked_mul(net) else {
                numeric_error = true;
                break;
            };
            let Some(sum) = weighted_sum.checked_add(weighted) else {
                numeric_error = true;
                break;
            };
            weighted_sum = sum;
            eligible_scenario_weight += u64::from(scenario.weight);
            eligible_scenario_count += 1;
            minimum = Some(minimum.map_or(net, |current| current.min(net)));
            maximum = Some(maximum.map_or(net, |current| current.max(net)));
            if net > 0 {
                positive_scenario_weight += u64::from(scenario.weight);
            } else {
                nonpositive_scenario_weight += u64::from(scenario.weight);
            }
        }

        if numeric_error {
            return invalid_family(&family.family_id, "robust_value_arithmetic_or_parse_failure");
        }

        candidates.push(CandidateRobustness {
            experiment_id,
            eligible_scenario_count,
            eligible_scenario_weight,
            weighted_net_evsi_numerator: weighted_sum.to_string(),
            minimum_net_evsi_numerator: minimum.map(|value| value.to_string()),
            maximum_net_evsi_numerator: maximum.map(|value| value.to_string()),
            positive_scenario_weight,
            nonpositive_scenario_weight,
            eligible_in_all_scenarios: eligible_scenario_count == family.scenarios.len()
                && eligible_scenario_weight == u64::from(SCENARIO_WEIGHT_SCALE),
            positive_in_all_scenarios: eligible_scenario_count == family.scenarios.len()
                && eligible_scenario_weight == u64::from(SCENARIO_WEIGHT_SCALE)
                && positive_scenario_weight == u64::from(SCENARIO_WEIGHT_SCALE),
        });
    }

    RobustEvaluation {
        family_id: family.family_id.clone(),
        status,
        robust_winner,
        reason_codes,
        scenario_results,
        candidates,
    }
}

pub fn evaluate_corpus(corpus: &RobustCorpus) -> Vec<RobustEvaluation> {
    corpus.families.iter().map(evaluate_family).collect()
}

pub fn validate_corpus_header(corpus: &RobustCorpus) -> Result<(), &'static str> {
    if corpus.corpus_id != ROBUST_CORPUS_ID || corpus.schema_version != ROBUST_SCHEMA_VERSION {
        return Err("corpus_id_or_schema_version_mismatch");
    }
    if corpus.scenario_weight_scale != SCENARIO_WEIGHT_SCALE {
        return Err("scenario_weight_scale_mismatch");
    }
    if corpus.family_count != corpus.families.len() {
        return Err("family_count_mismatch");
    }
    if corpus.claim_ceiling.trim().is_empty() || corpus.families.is_empty() {
        return Err("claim_ceiling_or_families_missing");
    }
    let mut family_ids = BTreeSet::new();
    for family in &corpus.families {
        if family.family_id.trim().is_empty() || !family_ids.insert(family.family_id.as_str()) {
            return Err("duplicate_or_empty_family_id");
        }
    }
    Ok(())
}

pub fn run_cli() -> i32 {
    let mut expected_head = None::<String>;
    let mut receipt_path = None::<PathBuf>;
    let mut suite_path = None::<PathBuf>;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--expected-head" => expected_head = args.next(),
            "--receipt" => receipt_path = args.next().map(PathBuf::from),
            "--suite" => suite_path = args.next().map(PathBuf::from),
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
    let Some(suite_path) = suite_path else {
        eprintln!("missing --suite");
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

    let suite_bytes = match fs::read(&suite_path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("cannot read suite {}: {error}", suite_path.display());
            return 2;
        }
    };
    let corpus: RobustCorpus = match serde_json::from_slice(&suite_bytes) {
        Ok(corpus) => corpus,
        Err(error) => {
            eprintln!("cannot parse robust suite: {error}");
            return 2;
        }
    };
    if let Err(error) = validate_corpus_header(&corpus) {
        eprintln!("invalid robust suite header: {error}");
        return 2;
    }

    let evaluations = evaluate_corpus(&corpus);
    let families: Vec<RobustFamilyReceipt> = corpus
        .families
        .iter()
        .zip(evaluations.iter())
        .map(|(family, evaluation)| RobustFamilyReceipt {
            family_id: family.family_id.clone(),
            evaluation: evaluation.clone(),
            expected_contract_match: evaluation.matches_expected(&family.expected),
        })
        .collect();
    let failed_family_ids: Vec<String> = families
        .iter()
        .filter(|family| !family.expected_contract_match)
        .map(|family| family.family_id.clone())
        .collect();
    let matched_family_count = families.len() - failed_family_ids.len();
    let contract_passed = failed_family_ids.is_empty();
    let receipt = RobustRunReceipt {
        runner_version: ROBUST_RUNNER_VERSION.to_owned(),
        corpus_id: corpus.corpus_id,
        schema_version: corpus.schema_version,
        corpus_sha256: sha256_hex(&suite_bytes),
        declared_head: expected_head,
        actual_head,
        exact_head_match: true,
        family_count: families.len(),
        matched_family_count,
        failed_family_ids,
        families,
        synthetic_only_claim_ceiling: CLAIM_CEILING.to_owned(),
        contract_passed,
    };

    if let Some(parent) = receipt_path.parent()
        && let Err(error) = fs::create_dir_all(parent)
    {
        eprintln!("cannot create receipt directory: {error}");
        return 2;
    }
    let receipt_bytes = match serde_json::to_vec_pretty(&receipt) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("cannot serialize robust receipt: {error}");
            return 2;
        }
    };
    if let Err(error) = fs::write(&receipt_path, &receipt_bytes) {
        eprintln!("cannot write receipt {}: {error}", receipt_path.display());
        return 2;
    }
    print!("{}", String::from_utf8_lossy(&receipt_bytes));
    println!();
    if contract_passed { 0 } else { 1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CORPUS_JSON: &str = include_str!(
        "../../../docs/research/bioprotein-experiments/robust-evsi-corpus-v1.json"
    );

    fn corpus() -> RobustCorpus {
        serde_json::from_str(CORPUS_JSON).expect("robust EVSI corpus parses")
    }

    #[test]
    fn all_joint_uncertainty_families_match_frozen_expected_contract() {
        let suite = corpus();
        validate_corpus_header(&suite).expect("suite header is valid");
        assert_eq!(suite.family_count, 8);
        for (family, evaluation) in suite.families.iter().zip(evaluate_corpus(&suite)) {
            assert!(
                evaluation.matches_expected(&family.expected),
                "family {} mismatch: expected {:?}, actual {:?}",
                family.family_id,
                family.expected,
                evaluation
            );
        }
    }

    #[test]
    fn stable_winner_requires_selection_in_every_joint_scenario() {
        let family = corpus()
            .families
            .into_iter()
            .find(|family| family.family_id == "ROBUST-001")
            .expect("stable family exists");
        let result = evaluate_family(&family);
        assert_eq!(result.status, RobustStatus::RobustWinner);
        assert_eq!(result.robust_winner.as_deref(), Some("assay-a"));
        assert!(
            result
                .candidates
                .iter()
                .find(|candidate| candidate.experiment_id == "assay-a")
                .is_some_and(|candidate| candidate.positive_in_all_scenarios)
        );
    }

    #[test]
    fn correlated_cost_and_prior_shifts_are_not_factorized_as_independent() {
        let family = corpus()
            .families
            .into_iter()
            .find(|family| family.family_id == "ROBUST-002")
            .expect("correlated family exists");
        let result = evaluate_family(&family);
        assert_eq!(result.status, RobustStatus::ScenarioSensitive);
        assert_eq!(
            result
                .scenario_results
                .iter()
                .map(|scenario| scenario.selected_experiment_id.clone())
                .collect::<Vec<_>>(),
            vec![
                Some("assay-a".to_owned()),
                Some("assay-b".to_owned()),
                Some("assay-b".to_owned())
            ]
        );
    }

    #[test]
    fn unknown_authorization_prevents_robust_recommendation() {
        let family = corpus()
            .families
            .into_iter()
            .find(|family| family.family_id == "ROBUST-005")
            .expect("permission-gate family exists");
        let result = evaluate_family(&family);
        assert_eq!(result.status, RobustStatus::Indeterminate);
        assert_eq!(result.robust_winner, None);
    }

    #[test]
    fn scenario_order_does_not_change_decision_or_candidate_summary() {
        let mut family = corpus()
            .families
            .into_iter()
            .find(|family| family.family_id == "ROBUST-002")
            .expect("correlated family exists");
        let before = evaluate_family(&family);
        family.scenarios.reverse();
        let after = evaluate_family(&family);
        assert_eq!(before, after);
    }

    #[test]
    fn malformed_scenario_weight_mass_fails_closed() {
        let family = corpus()
            .families
            .into_iter()
            .find(|family| family.family_id == "ROBUST-008")
            .expect("malformed weight family exists");
        let result = evaluate_family(&family);
        assert_eq!(result.status, RobustStatus::InvalidInput);
        assert_eq!(result.reason_codes, vec!["scenario_weight_sum_mismatch"]);
        assert_eq!(result.robust_winner, None);
    }

    #[test]
    fn candidate_set_mismatch_is_not_ranked_as_comparable() {
        let mut family = corpus()
            .families
            .into_iter()
            .find(|family| family.family_id == "ROBUST-001")
            .expect("stable family exists");
        family.scenarios[0].case.experiments.pop();
        let result = evaluate_family(&family);
        assert_eq!(result.status, RobustStatus::InvalidInput);
        assert_eq!(result.reason_codes, vec!["candidate_set_mismatch"]);
    }

    #[test]
    fn corpus_identity_and_family_count_are_checked() {
        let mut suite = corpus();
        suite.family_count -= 1;
        assert_eq!(validate_corpus_header(&suite), Err("family_count_mismatch"));
        let mut suite = corpus();
        suite.families[1].family_id = suite.families[0].family_id.clone();
        assert_eq!(validate_corpus_header(&suite), Err("duplicate_or_empty_family_id"));
    }
}
