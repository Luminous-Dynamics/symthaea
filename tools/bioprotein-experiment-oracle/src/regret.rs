// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fixed-budget comparison of research-selection strategies by decision regret.
//!
//! All score proxies and scenarios are synthetic. The oracle compares single-experiment
//! choices under one explicit budget; it does not model combined-test synergy, biological
//! outcomes, food safety, production economics, or lifecycle impacts.

use super::{Case, CaseEvaluation, ResultStatus, evaluate_case, sha256_hex};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

pub const REGRET_RUNNER_VERSION: &str = "0.1.0";
pub const REGRET_CORPUS_ID: &str = "BIOPROTEIN-DECISION-REGRET-CORPUS-V1";
pub const REGRET_SCHEMA_VERSION: &str = "1.0.0";
pub const REGRET_WEIGHT_SCALE: u32 = 10_000;
const SCORE_ABS_LIMIT: i64 = 1_000_000;
const CLAIM_CEILING: &str = "Synthetic fixed-budget decision-regret benchmark only. Scores, utilities, likelihoods, scenario weights and costs are invented. No empirical strategy superiority, biological prediction, experiment authorization, food safety, real economics or lifecycle outcome is established.";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegretStatus {
    Valid,
    InvalidInput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegretStrategy {
    NetEvsi,
    YieldOnly,
    LocalSensitivityOnly,
    EqualAllocationRoundRobin,
    SeededRandom,
}

impl RegretStrategy {
    pub const ALL: [Self; 5] = [
        Self::NetEvsi,
        Self::YieldOnly,
        Self::LocalSensitivityOnly,
        Self::EqualAllocationRoundRobin,
        Self::SeededRandom,
    ];

    pub fn key(self) -> &'static str {
        match self {
            Self::NetEvsi => "net_evsi",
            Self::YieldOnly => "yield_only",
            Self::LocalSensitivityOnly => "local_sensitivity_only",
            Self::EqualAllocationRoundRobin => "equal_allocation_round_robin",
            Self::SeededRandom => "seeded_random",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaselineScore {
    pub experiment_id: String,
    pub expected_output_score: i64,
    pub local_sensitivity_score: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegretScenario {
    pub scenario_id: String,
    pub weight: u32,
    pub budget_utility: i64,
    pub assumption_tags: Vec<String>,
    pub case: Case,
    pub baseline_scores: Vec<BaselineScore>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedScenarioSelections {
    pub scenario_id: String,
    pub selected_by_strategy: BTreeMap<String, Option<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegretExpected {
    pub status: RegretStatus,
    pub scenario_selections: Vec<ExpectedScenarioSelections>,
    pub weighted_regret_numerators: BTreeMap<String, String>,
    pub maximum_regret_numerators: BTreeMap<String, String>,
    pub reason_codes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegretFamily {
    pub family_id: String,
    pub random_seed: u64,
    pub scenarios: Vec<RegretScenario>,
    pub expected: RegretExpected,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegretCorpus {
    pub corpus_id: String,
    pub schema_version: String,
    pub scenario_weight_scale: u32,
    pub family_count: usize,
    pub claim_ceiling: String,
    pub families: Vec<RegretFamily>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StrategyChoice {
    pub strategy: RegretStrategy,
    pub selected_experiment_id: Option<String>,
    pub selected_net_evsi_numerator: String,
    pub regret_numerator: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScenarioRegretResult {
    pub scenario_id: String,
    pub weight: u32,
    pub budget_utility: i64,
    pub eligible_experiment_ids: Vec<String>,
    pub oracle_experiment_id: Option<String>,
    pub oracle_net_evsi_numerator: String,
    pub choices: Vec<StrategyChoice>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StrategySummary {
    pub strategy: RegretStrategy,
    /// Exact sum of scenario weight multiplied by regret numerator.
    /// Divide by the corpus weight scale for the weighted mean numerator.
    pub weighted_regret_numerator: String,
    pub maximum_regret_numerator: String,
    pub zero_regret_scenario_weight: u64,
    pub selected_scenario_weight_by_experiment: BTreeMap<String, u64>,
    pub no_experiment_scenario_weight: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegretEvaluation {
    pub family_id: String,
    pub status: RegretStatus,
    pub reason_codes: Vec<String>,
    pub scenario_results: Vec<ScenarioRegretResult>,
    pub strategy_summaries: Vec<StrategySummary>,
}

impl RegretEvaluation {
    pub fn matches_expected(&self, expected: &RegretExpected) -> bool {
        let choices: Vec<ExpectedScenarioSelections> = self
            .scenario_results
            .iter()
            .map(|scenario| ExpectedScenarioSelections {
                scenario_id: scenario.scenario_id.clone(),
                selected_by_strategy: scenario
                    .choices
                    .iter()
                    .map(|choice| {
                        (
                            choice.strategy.key().to_owned(),
                            choice.selected_experiment_id.clone(),
                        )
                    })
                    .collect(),
            })
            .collect();
        let weighted: BTreeMap<String, String> = self
            .strategy_summaries
            .iter()
            .map(|summary| {
                (
                    summary.strategy.key().to_owned(),
                    summary.weighted_regret_numerator.clone(),
                )
            })
            .collect();
        let maximum: BTreeMap<String, String> = self
            .strategy_summaries
            .iter()
            .map(|summary| {
                (
                    summary.strategy.key().to_owned(),
                    summary.maximum_regret_numerator.clone(),
                )
            })
            .collect();
        self.status == expected.status
            && choices == expected.scenario_selections
            && weighted == expected.weighted_regret_numerators
            && maximum == expected.maximum_regret_numerators
            && self.reason_codes == expected.reason_codes
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegretFamilyReceipt {
    pub family_id: String,
    pub evaluation: RegretEvaluation,
    pub expected_contract_match: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegretRunReceipt {
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
    pub families: Vec<RegretFamilyReceipt>,
    pub synthetic_only_claim_ceiling: String,
    pub contract_passed: bool,
}

fn invalid_family(family_id: &str, reason: &str) -> RegretEvaluation {
    RegretEvaluation {
        family_id: family_id.to_owned(),
        status: RegretStatus::InvalidInput,
        reason_codes: vec![reason.to_owned()],
        scenario_results: Vec::new(),
        strategy_summaries: Vec::new(),
    }
}

fn action_signature(case: &Case) -> BTreeMap<String, BTreeMap<String, i64>> {
    case.actions
        .iter()
        .map(|action| (action.action_id.clone(), action.utilities.clone()))
        .collect()
}

fn state_signature(case: &Case) -> BTreeSet<String> {
    case.states
        .iter()
        .map(|state| state.state_id.clone())
        .collect()
}

fn experiment_signature(case: &Case) -> BTreeSet<String> {
    case.experiments
        .iter()
        .map(|experiment| experiment.experiment_id.clone())
        .collect()
}

fn validate_family(family: &RegretFamily) -> Vec<String> {
    let mut errors = Vec::new();
    if family.family_id.trim().is_empty() || family.scenarios.len() < 2 {
        errors.push("family_identity_or_scenarios_missing".to_owned());
    }
    let mut scenario_ids = BTreeSet::new();
    let mut case_ids = BTreeSet::new();
    let mut scenario_digests = BTreeSet::new();
    let mut weight_sum = 0_u64;
    let mut reference_budget = None;

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
        if scenario.weight == 0 || scenario.weight > REGRET_WEIGHT_SCALE {
            errors.push("invalid_scenario_weight".to_owned());
        }
        weight_sum += u64::from(scenario.weight);
        if scenario.budget_utility < 0 || scenario.budget_utility > SCORE_ABS_LIMIT {
            errors.push("invalid_budget_utility".to_owned());
        }
        if let Some(budget) = reference_budget {
            if budget != scenario.budget_utility {
                errors.push("fixed_budget_contract_mismatch".to_owned());
            }
        } else {
            reference_budget = Some(scenario.budget_utility);
        }
        if scenario.assumption_tags.is_empty()
            || scenario
                .assumption_tags
                .iter()
                .any(|tag| tag.trim().is_empty())
        {
            errors.push("scenario_assumption_tags_missing".to_owned());
        }
        if !scenario_digests.insert(scenario.case.scenario_digest.as_str()) {
            errors.push("duplicate_scenario_digest".to_owned());
        }

        let candidate_ids = experiment_signature(&scenario.case);
        let mut score_ids = BTreeSet::<String>::new();
        for score in &scenario.baseline_scores {
            if score.experiment_id.trim().is_empty()
                || !score_ids.insert(score.experiment_id.clone())
            {
                errors.push("duplicate_or_empty_baseline_score_id".to_owned());
            }
            if score.expected_output_score.unsigned_abs() > SCORE_ABS_LIMIT as u64
                || score.local_sensitivity_score.unsigned_abs() > SCORE_ABS_LIMIT as u64
            {
                errors.push("baseline_score_out_of_bounds".to_owned());
            }
        }
        if score_ids != candidate_ids {
            errors.push("baseline_score_candidate_set_mismatch".to_owned());
        }
    }

    if weight_sum != u64::from(REGRET_WEIGHT_SCALE) {
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

fn stable_hash(seed: u64, value: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64 ^ seed;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let mut mixed = hash.wrapping_add(0x9e3779b97f4a7c15);
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d049bb133111eb);
    mixed ^ (mixed >> 31)
}

fn choose_by_score(
    eligible_ids: &[String],
    scores: &BTreeMap<String, BaselineScore>,
    get_score: fn(&BaselineScore) -> i64,
) -> Option<String> {
    eligible_ids
        .iter()
        .min_by(|left, right| {
            let left_score = scores.get(*left).map(get_score).unwrap_or(i64::MIN);
            let right_score = scores.get(*right).map(get_score).unwrap_or(i64::MIN);
            right_score.cmp(&left_score).then_with(|| left.cmp(right))
        })
        .cloned()
}

fn invalid_case_evaluation(evaluation: &CaseEvaluation) -> Option<&str> {
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

pub fn evaluate_family(family: &RegretFamily) -> RegretEvaluation {
    let errors = validate_family(family);
    if let Some(reason) = errors.first() {
        return invalid_family(&family.family_id, reason);
    }

    let mut ordered: Vec<&RegretScenario> = family.scenarios.iter().collect();
    ordered.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    let mut evaluated: Vec<(&RegretScenario, CaseEvaluation)> = Vec::with_capacity(ordered.len());
    for scenario in ordered {
        let evaluation = evaluate_case(&scenario.case);
        if let Some(reason) = invalid_case_evaluation(&evaluation) {
            return invalid_family(&family.family_id, reason);
        }
        evaluated.push((scenario, evaluation));
    }

    let mut scenario_results = Vec::with_capacity(evaluated.len());
    for (scenario_index, (scenario, evaluation)) in evaluated.iter().enumerate() {
        let values: BTreeMap<String, i128> = evaluation
            .experiment_values
            .iter()
            .filter_map(|value| {
                value
                    .net_evsi_numerator
                    .parse::<i128>()
                    .ok()
                    .map(|net| (value.experiment_id.clone(), net))
            })
            .collect();

        let cost_by_id: BTreeMap<&str, i64> = scenario
            .case
            .experiments
            .iter()
            .map(|experiment| (experiment.experiment_id.as_str(), experiment.cost_utility))
            .collect();
        let mut eligible_ids: Vec<String> = values
            .keys()
            .filter(|id| {
                cost_by_id
                    .get(id.as_str())
                    .is_some_and(|cost| *cost <= scenario.budget_utility)
            })
            .cloned()
            .collect();
        eligible_ids.sort();

        let mut oracle: Option<(String, i128)> = None;
        for experiment_id in &eligible_ids {
            let net = values[experiment_id];
            if net <= 0 {
                continue;
            }
            if oracle.as_ref().is_none_or(|(current_id, current_net)| {
                net > *current_net || (net == *current_net && experiment_id < current_id)
            }) {
                oracle = Some((experiment_id.clone(), net));
            }
        }
        let oracle_experiment_id = oracle.as_ref().map(|entry| entry.0.clone());
        let oracle_value = oracle.as_ref().map_or(0_i128, |entry| entry.1);

        let score_map: BTreeMap<String, BaselineScore> = scenario
            .baseline_scores
            .iter()
            .cloned()
            .map(|score| (score.experiment_id.clone(), score))
            .collect();

        let yield_selection = choose_by_score(&eligible_ids, &score_map, |score| {
            score.expected_output_score
        });
        let sensitivity_selection = choose_by_score(&eligible_ids, &score_map, |score| {
            score.local_sensitivity_score
        });
        let round_robin_selection = if eligible_ids.is_empty() {
            None
        } else {
            Some(eligible_ids[scenario_index % eligible_ids.len()].clone())
        };
        let random_selection = if eligible_ids.is_empty() {
            None
        } else {
            let pick = (stable_hash(family.random_seed, &scenario.scenario_id)
                % eligible_ids.len() as u64) as usize;
            Some(eligible_ids[pick].clone())
        };

        let selections = [
            (RegretStrategy::NetEvsi, oracle_experiment_id.clone()),
            (RegretStrategy::YieldOnly, yield_selection),
            (RegretStrategy::LocalSensitivityOnly, sensitivity_selection),
            (
                RegretStrategy::EqualAllocationRoundRobin,
                round_robin_selection,
            ),
            (RegretStrategy::SeededRandom, random_selection),
        ];

        let mut choices = Vec::with_capacity(RegretStrategy::ALL.len());
        for (strategy, selected_experiment_id) in selections {
            let selected_value = selected_experiment_id
                .as_ref()
                .and_then(|id| values.get(id))
                .copied()
                .unwrap_or(0);
            let regret = oracle_value.checked_sub(selected_value);
            let Some(regret) = regret else {
                return invalid_family(&family.family_id, "regret_arithmetic_overflow");
            };
            if regret < 0 {
                return invalid_family(&family.family_id, "baseline_outperforms_oracle");
            }
            choices.push(StrategyChoice {
                strategy,
                selected_experiment_id,
                selected_net_evsi_numerator: selected_value.to_string(),
                regret_numerator: regret.to_string(),
            });
        }

        scenario_results.push(ScenarioRegretResult {
            scenario_id: scenario.scenario_id.clone(),
            weight: scenario.weight,
            budget_utility: scenario.budget_utility,
            eligible_experiment_ids: eligible_ids,
            oracle_experiment_id,
            oracle_net_evsi_numerator: oracle_value.to_string(),
            choices,
        });
    }

    let mut strategy_summaries = Vec::with_capacity(RegretStrategy::ALL.len());
    for strategy in RegretStrategy::ALL {
        let mut weighted_regret = 0_i128;
        let mut maximum_regret = 0_i128;
        let mut zero_regret_scenario_weight = 0_u64;
        let mut selected_scenario_weight_by_experiment = BTreeMap::<String, u64>::new();
        let mut no_experiment_scenario_weight = 0_u64;

        for scenario in &scenario_results {
            let choice = scenario
                .choices
                .iter()
                .find(|choice| choice.strategy == strategy)
                .expect("all strategy choices are emitted");
            let regret = match choice.regret_numerator.parse::<i128>() {
                Ok(value) => value,
                Err(_) => return invalid_family(&family.family_id, "regret_value_parse_failure"),
            };
            let Some(weighted) = i128::from(scenario.weight).checked_mul(regret) else {
                return invalid_family(&family.family_id, "regret_arithmetic_overflow");
            };
            let Some(total) = weighted_regret.checked_add(weighted) else {
                return invalid_family(&family.family_id, "regret_arithmetic_overflow");
            };
            weighted_regret = total;
            maximum_regret = maximum_regret.max(regret);
            if regret == 0 {
                zero_regret_scenario_weight += u64::from(scenario.weight);
            }
            if let Some(id) = &choice.selected_experiment_id {
                *selected_scenario_weight_by_experiment
                    .entry(id.clone())
                    .or_default() += u64::from(scenario.weight);
            } else {
                no_experiment_scenario_weight += u64::from(scenario.weight);
            }
        }
        strategy_summaries.push(StrategySummary {
            strategy,
            weighted_regret_numerator: weighted_regret.to_string(),
            maximum_regret_numerator: maximum_regret.to_string(),
            zero_regret_scenario_weight,
            selected_scenario_weight_by_experiment,
            no_experiment_scenario_weight,
        });
    }

    RegretEvaluation {
        family_id: family.family_id.clone(),
        status: RegretStatus::Valid,
        reason_codes: vec!["fixed_budget_scenario_regret_evaluated".to_owned()],
        scenario_results,
        strategy_summaries,
    }
}

pub fn evaluate_corpus(corpus: &RegretCorpus) -> Vec<RegretEvaluation> {
    corpus.families.iter().map(evaluate_family).collect()
}

pub fn validate_corpus_header(corpus: &RegretCorpus) -> Result<(), &'static str> {
    if corpus.corpus_id != REGRET_CORPUS_ID || corpus.schema_version != REGRET_SCHEMA_VERSION {
        return Err("corpus_id_or_schema_version_mismatch");
    }
    if corpus.scenario_weight_scale != REGRET_WEIGHT_SCALE {
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
    let mut corpus_path = None::<PathBuf>;
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
    let corpus: RegretCorpus = match serde_json::from_slice(&corpus_bytes) {
        Ok(corpus) => corpus,
        Err(error) => {
            eprintln!("cannot parse regret corpus: {error}");
            return 2;
        }
    };
    if let Err(error) = validate_corpus_header(&corpus) {
        eprintln!("invalid regret corpus header: {error}");
        return 2;
    }

    let evaluations = evaluate_corpus(&corpus);
    let families: Vec<RegretFamilyReceipt> = corpus
        .families
        .iter()
        .zip(evaluations.iter())
        .map(|(family, evaluation)| RegretFamilyReceipt {
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
    let receipt = RegretRunReceipt {
        runner_version: REGRET_RUNNER_VERSION.to_owned(),
        corpus_id: corpus.corpus_id,
        schema_version: corpus.schema_version,
        corpus_sha256: sha256_hex(&corpus_bytes),
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
            eprintln!("cannot serialize regret receipt: {error}");
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
        "../../../docs/research/bioprotein-experiments/decision-regret-corpus-v1.json"
    );

    fn corpus() -> RegretCorpus {
        serde_json::from_str(CORPUS_JSON).expect("decision-regret corpus parses")
    }

    #[test]
    fn frozen_corpus_matches_strategy_selections_and_regret_metrics() {
        let suite = corpus();
        validate_corpus_header(&suite).expect("regret corpus header validates");
        assert_eq!(suite.family_count, 5);
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
    fn net_evsi_oracle_has_zero_regret_by_construction() {
        let suite = corpus();
        for evaluation in evaluate_corpus(&suite) {
            if evaluation.status != RegretStatus::Valid {
                continue;
            }
            let summary = evaluation
                .strategy_summaries
                .iter()
                .find(|summary| summary.strategy == RegretStrategy::NetEvsi)
                .expect("net EVSI summary exists");
            assert_eq!(summary.weighted_regret_numerator, "0");
            assert_eq!(summary.maximum_regret_numerator, "0");
        }
    }

    #[test]
    fn budget_filter_prevents_heuristic_from_buying_unaffordable_candidate() {
        let family = corpus()
            .families
            .into_iter()
            .find(|family| family.family_id == "REGRET-002")
            .expect("budget family exists");
        let evaluation = evaluate_family(&family);
        for scenario in &evaluation.scenario_results {
            assert!(
                !scenario
                    .eligible_experiment_ids
                    .iter()
                    .any(|id| id == "assay-a")
            );
            let yield_choice = scenario
                .choices
                .iter()
                .find(|choice| choice.strategy == RegretStrategy::YieldOnly)
                .expect("yield baseline choice exists");
            assert_ne!(
                yield_choice.selected_experiment_id.as_deref(),
                Some("assay-a")
            );
        }
    }

    #[test]
    fn round_robin_gives_equal_candidate_slots_when_all_are_feasible() {
        let family = corpus()
            .families
            .into_iter()
            .find(|family| family.family_id == "REGRET-001")
            .expect("allocation family exists");
        let evaluation = evaluate_family(&family);
        let selected: Vec<String> = evaluation
            .scenario_results
            .iter()
            .filter_map(|scenario| {
                scenario
                    .choices
                    .iter()
                    .find(|choice| choice.strategy == RegretStrategy::EqualAllocationRoundRobin)
                    .and_then(|choice| choice.selected_experiment_id.clone())
            })
            .collect();
        assert_eq!(selected, vec!["assay-a", "assay-b", "assay-c"]);
    }

    #[test]
    fn seeded_random_baseline_is_order_invariant() {
        let mut family = corpus()
            .families
            .into_iter()
            .find(|family| family.family_id == "REGRET-001")
            .expect("allocation family exists");
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
            .find(|family| family.family_id == "REGRET-005")
            .expect("invalid weight family exists");
        let result = evaluate_family(&family);
        assert_eq!(result.status, RegretStatus::InvalidInput);
        assert_eq!(result.reason_codes, vec!["scenario_weight_sum_mismatch"]);
    }

    #[test]
    fn no_positive_evsi_option_preserves_zero_regret_oracle() {
        let family = corpus()
            .families
            .into_iter()
            .find(|family| family.family_id == "REGRET-003")
            .expect("no-positive family exists");
        let result = evaluate_family(&family);
        assert!(
            result
                .scenario_results
                .iter()
                .all(|scenario| scenario.oracle_experiment_id.is_none())
        );
    }

    #[test]
    fn regret_is_nonnegative_and_uses_exact_integer_numerators() {
        for evaluation in evaluate_corpus(&corpus()) {
            for scenario in &evaluation.scenario_results {
                for choice in &scenario.choices {
                    assert!(choice.regret_numerator.parse::<i128>().unwrap() >= 0);
                }
            }
        }
    }
}
