// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Known-truth scorer for deterministic acoustic experiment-selection benchmarks.
//!
//! The ranking policy receives only ExperimentOption metadata. Oracle outcomes
//! are kept in a separate case and are consumed only by the scorer, after a
//! budgeted plan is formed. This separation prevents the policy from selecting
//! directly on the answer in a synthetic benchmark.
//!
//! Question-value units, option estimates and validation obligations are
//! synthetic benchmark inputs, not measured real-world value, probabilities,
//! or calibration guarantees. Benchmark reports should identify the generator,
//! seed, objective schema, fidelity costs and validation-cost convention.

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::experiment_selection::{
    rank_experiments, select_under_budget, BudgetedExperimentPlan, ExperimentOption,
    ExperimentSelectionError, ExperimentSelectionPolicy,
};

/// A synthetic decision question with fixed oracle value in the benchmark profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OracleQuestion {
    pub id: String,
    pub value_units: u64,
}

impl OracleQuestion {
    pub fn new(id: impl Into<String>, value_units: u64) -> Self {
        Self {
            id: id.into(),
            value_units,
        }
    }
}

/// Known oracle outcome for one experiment option.
///
/// This type belongs to the scorer-only data path. Do not copy its fields into
/// ExperimentOption or expose it to a selection policy during a benchmark run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntheticExperimentOutcome {
    pub option_id: String,
    pub resolved_question_ids: Vec<String>,
    /// Additional validation cost required before using this result to promote
    /// a candidate to a final engineering decision.
    pub deferred_validation_cost_units: u64,
}

impl SyntheticExperimentOutcome {
    pub fn new(
        option_id: impl Into<String>,
        resolved_question_ids: Vec<String>,
        deferred_validation_cost_units: u64,
    ) -> Self {
        Self {
            option_id: option_id.into(),
            resolved_question_ids,
            deferred_validation_cost_units,
        }
    }
}

/// Known-truth case for scoring a budgeted selection plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntheticBenchmarkCase {
    pub scenario_id: String,
    pub evaluation_profile_id: String,
    pub cost_unit_id: String,
    pub questions: Vec<OracleQuestion>,
    pub outcomes: Vec<SyntheticExperimentOutcome>,
}

impl SyntheticBenchmarkCase {
    pub fn new(
        scenario_id: impl Into<String>,
        evaluation_profile_id: impl Into<String>,
        cost_unit_id: impl Into<String>,
        questions: Vec<OracleQuestion>,
        outcomes: Vec<SyntheticExperimentOutcome>,
    ) -> Self {
        Self {
            scenario_id: scenario_id.into(),
            evaluation_profile_id: evaluation_profile_id.into(),
            cost_unit_id: cost_unit_id.into(),
            questions,
            outcomes,
        }
    }
}

/// Reproducible metrics for one policy's budgeted plan on a known-truth case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntheticBenchmarkReceipt {
    pub scenario_id: String,
    pub evaluation_profile_id: String,
    pub cost_unit_id: String,
    pub policy: ExperimentSelectionPolicy,
    pub budget_units: u64,
    pub selected_option_ids: Vec<String>,
    pub resolved_question_ids: Vec<String>,
    pub realized_decision_value_units: u64,
    pub immediate_spend_units: u64,
    pub deferred_validation_cost_units: u64,
    pub end_to_end_cost_units: u64,
    pub unused_budget_units: u64,
}

/// Invalid case truth, plan integrity, or synthetic benchmark accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyntheticBenchmarkError {
    Selection(ExperimentSelectionError),
    EmptyScenarioId,
    EmptyEvaluationProfileId,
    EmptyCostUnitId,
    EmptyQuestionId,
    DuplicateQuestionId,
    EmptyOptionId,
    DuplicateOptionId,
    EmptyDesignId,
    EmptyFidelityId,
    ZeroEstimatedCost,
    OptionContractMismatch,
    DuplicateOutcomeId,
    MissingOutcome,
    UnknownOutcome,
    EmptyResolvedQuestionId,
    DuplicateResolvedQuestionId,
    UnknownResolvedQuestion,
    DuplicateSelectedOption,
    DuplicateSkippedOption,
    UnknownPlannedOption,
    PlanOptionPartitionMismatch,
    PlanIndexIdMismatch,
    PlanSpendMismatch,
    PlanBudgetAccountingMismatch,
    PlanExceedsBudget,
    ValueOverflow,
    ValidationCostOverflow,
    EndToEndCostOverflow,
}

impl fmt::Display for SyntheticBenchmarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Selection(error) => return write!(f, "experiment selection failed: {error}"),
            Self::EmptyScenarioId => "scenario identifier must not be empty",
            Self::EmptyEvaluationProfileId => "evaluation profile identifier must not be empty",
            Self::EmptyCostUnitId => "cost unit identifier must not be empty",
            Self::EmptyQuestionId => "oracle question identifiers must not be empty",
            Self::DuplicateQuestionId => "oracle question identifiers must be unique",
            Self::EmptyOptionId => "experiment option identifiers must not be empty",
            Self::DuplicateOptionId => "experiment option identifiers must be unique",
            Self::EmptyDesignId => "design identifiers must not be empty",
            Self::EmptyFidelityId => "fidelity identifiers must not be empty",
            Self::ZeroEstimatedCost => "estimated experiment costs must be greater than zero",
            Self::OptionContractMismatch => {
                "option cost unit or evaluation profile does not match the case"
            }
            Self::DuplicateOutcomeId => "oracle outcome identifiers must be unique",
            Self::MissingOutcome => "every eligible option requires an oracle outcome",
            Self::UnknownOutcome => "oracle outcome refers to an unknown option",
            Self::EmptyResolvedQuestionId => "resolved question identifiers must not be empty",
            Self::DuplicateResolvedQuestionId => {
                "one oracle outcome cannot resolve the same question twice"
            }
            Self::UnknownResolvedQuestion => "oracle outcome refers to an unknown question",
            Self::DuplicateSelectedOption => "selected option identifiers must be unique",
            Self::DuplicateSkippedOption => "skipped option identifiers must be unique",
            Self::UnknownPlannedOption => "plan refers to an unknown option identifier",
            Self::PlanOptionPartitionMismatch => {
                "selected and skipped options must partition the complete option pool"
            }
            Self::PlanIndexIdMismatch => "plan indices and selected identifiers disagree",
            Self::PlanSpendMismatch => "plan spend differs from selected option costs",
            Self::PlanBudgetAccountingMismatch => {
                "plan spend plus unused budget must equal the declared budget"
            }
            Self::PlanExceedsBudget => "plan exceeds the declared immediate budget",
            Self::ValueOverflow => "synthetic decision value overflowed",
            Self::ValidationCostOverflow => "deferred validation cost overflowed",
            Self::EndToEndCostOverflow => "end-to-end cost overflowed",
        };
        f.write_str(message)
    }
}

impl std::error::Error for SyntheticBenchmarkError {}

impl From<ExperimentSelectionError> for SyntheticBenchmarkError {
    fn from(error: ExperimentSelectionError) -> Self {
        Self::Selection(error)
    }
}

/// Run one deterministic baseline and score its plan against separate oracle truth.
///
/// This is a small policy harness, not Bayesian optimization and not proof that
/// any heuristic generalizes to physical acoustics. It enables controlled
/// comparisons on synthetic cases whose outcomes and trade-offs are known.
pub fn run_synthetic_benchmark(
    options: &[ExperimentOption],
    case: &SyntheticBenchmarkCase,
    budget_units: u64,
    policy: ExperimentSelectionPolicy,
) -> Result<SyntheticBenchmarkReceipt, SyntheticBenchmarkError> {
    validate_case_contract(case)?;
    let ranking = rank_experiments(
        options,
        &case.evaluation_profile_id,
        &case.cost_unit_id,
        policy,
    )?;
    let plan = select_under_budget(options, &ranking, budget_units)?;
    score_synthetic_plan(options, case, &plan)
}

/// Score an existing plan using hidden-to-policy synthetic oracle outcomes.
///
/// Each resolved question's value is counted at most once, even if several
/// selected experiments reveal the same fact. Immediate experiment spend and
/// deferred high-fidelity validation cost are reported separately and together.
pub fn score_synthetic_plan(
    options: &[ExperimentOption],
    case: &SyntheticBenchmarkCase,
    plan: &BudgetedExperimentPlan,
) -> Result<SyntheticBenchmarkReceipt, SyntheticBenchmarkError> {
    validate_case_contract(case)?;
    if plan.evaluation_profile_id != case.evaluation_profile_id
        || plan.cost_unit_id != case.cost_unit_id
    {
        return Err(SyntheticBenchmarkError::OptionContractMismatch);
    }

    let option_map = validate_options(options, case)?;
    let question_map = validate_questions(&case.questions)?;
    let outcome_map = validate_outcomes(&case.outcomes, &option_map, &question_map)?;
    validate_plan(options, plan, &option_map)?;

    let mut resolved_question_ids = HashSet::new();
    let mut deferred_validation_cost_units = 0_u64;
    for option_id in &plan.selected_option_ids {
        let outcome = outcome_map
            .get(option_id.as_str())
            .ok_or(SyntheticBenchmarkError::MissingOutcome)?;
        deferred_validation_cost_units = deferred_validation_cost_units
            .checked_add(outcome.deferred_validation_cost_units)
            .ok_or(SyntheticBenchmarkError::ValidationCostOverflow)?;
        for question_id in &outcome.resolved_question_ids {
            resolved_question_ids.insert(question_id.as_str());
        }
    }

    let mut resolved_question_ids: Vec<String> = resolved_question_ids
        .into_iter()
        .map(str::to_owned)
        .collect();
    resolved_question_ids.sort();

    let mut realized_decision_value_units = 0_u64;
    for question_id in &resolved_question_ids {
        let question = question_map
            .get(question_id.as_str())
            .ok_or(SyntheticBenchmarkError::UnknownResolvedQuestion)?;
        realized_decision_value_units = realized_decision_value_units
            .checked_add(question.value_units)
            .ok_or(SyntheticBenchmarkError::ValueOverflow)?;
    }

    let end_to_end_cost_units = plan
        .estimated_cost_units
        .checked_add(deferred_validation_cost_units)
        .ok_or(SyntheticBenchmarkError::EndToEndCostOverflow)?;

    Ok(SyntheticBenchmarkReceipt {
        scenario_id: case.scenario_id.clone(),
        evaluation_profile_id: case.evaluation_profile_id.clone(),
        cost_unit_id: case.cost_unit_id.clone(),
        policy: plan.policy,
        budget_units: plan.budget_units,
        selected_option_ids: plan.selected_option_ids.clone(),
        resolved_question_ids,
        realized_decision_value_units,
        immediate_spend_units: plan.estimated_cost_units,
        deferred_validation_cost_units,
        end_to_end_cost_units,
        unused_budget_units: plan.unused_budget_units,
    })
}

fn validate_case_contract(
    case: &SyntheticBenchmarkCase,
) -> Result<(), SyntheticBenchmarkError> {
    if case.scenario_id.trim().is_empty() {
        return Err(SyntheticBenchmarkError::EmptyScenarioId);
    }
    if case.evaluation_profile_id.trim().is_empty() {
        return Err(SyntheticBenchmarkError::EmptyEvaluationProfileId);
    }
    if case.cost_unit_id.trim().is_empty() {
        return Err(SyntheticBenchmarkError::EmptyCostUnitId);
    }
    Ok(())
}

fn validate_options<'a>(
    options: &'a [ExperimentOption],
    case: &SyntheticBenchmarkCase,
) -> Result<HashMap<&'a str, &'a ExperimentOption>, SyntheticBenchmarkError> {
    let mut option_map = HashMap::with_capacity(options.len());
    for option in options {
        if option.id.trim().is_empty() {
            return Err(SyntheticBenchmarkError::EmptyOptionId);
        }
        if option.design_id.trim().is_empty() {
            return Err(SyntheticBenchmarkError::EmptyDesignId);
        }
        if option.fidelity_id.trim().is_empty() {
            return Err(SyntheticBenchmarkError::EmptyFidelityId);
        }
        if option.estimated_cost_units == 0 {
            return Err(SyntheticBenchmarkError::ZeroEstimatedCost);
        }
        if option.evaluation_profile_id != case.evaluation_profile_id
            || option.cost_unit_id != case.cost_unit_id
        {
            return Err(SyntheticBenchmarkError::OptionContractMismatch);
        }
        if option_map.insert(option.id.as_str(), option).is_some() {
            return Err(SyntheticBenchmarkError::DuplicateOptionId);
        }
    }
    Ok(option_map)
}

fn validate_questions<'a>(
    questions: &'a [OracleQuestion],
) -> Result<HashMap<&'a str, &'a OracleQuestion>, SyntheticBenchmarkError> {
    let mut question_map = HashMap::with_capacity(questions.len());
    for question in questions {
        if question.id.trim().is_empty() {
            return Err(SyntheticBenchmarkError::EmptyQuestionId);
        }
        if question_map
            .insert(question.id.as_str(), question)
            .is_some()
        {
            return Err(SyntheticBenchmarkError::DuplicateQuestionId);
        }
    }
    Ok(question_map)
}

fn validate_outcomes<'a>(
    outcomes: &'a [SyntheticExperimentOutcome],
    options: &HashMap<&str, &ExperimentOption>,
    questions: &HashMap<&str, &OracleQuestion>,
) -> Result<HashMap<&'a str, &'a SyntheticExperimentOutcome>, SyntheticBenchmarkError> {
    let mut outcome_map = HashMap::with_capacity(outcomes.len());
    for outcome in outcomes {
        if outcome.option_id.trim().is_empty() {
            return Err(SyntheticBenchmarkError::EmptyOptionId);
        }
        if !options.contains_key(outcome.option_id.as_str()) {
            return Err(SyntheticBenchmarkError::UnknownOutcome);
        }
        if outcome_map
            .insert(outcome.option_id.as_str(), outcome)
            .is_some()
        {
            return Err(SyntheticBenchmarkError::DuplicateOutcomeId);
        }

        let mut seen_questions = HashSet::with_capacity(outcome.resolved_question_ids.len());
        for question_id in &outcome.resolved_question_ids {
            if question_id.trim().is_empty() {
                return Err(SyntheticBenchmarkError::EmptyResolvedQuestionId);
            }
            if !seen_questions.insert(question_id.as_str()) {
                return Err(SyntheticBenchmarkError::DuplicateResolvedQuestionId);
            }
            if !questions.contains_key(question_id.as_str()) {
                return Err(SyntheticBenchmarkError::UnknownResolvedQuestion);
            }
        }
    }

    if outcome_map.len() != options.len() {
        return Err(SyntheticBenchmarkError::MissingOutcome);
    }
    Ok(outcome_map)
}

fn validate_plan(
    options: &[ExperimentOption],
    plan: &BudgetedExperimentPlan,
    option_map: &HashMap<&str, &ExperimentOption>,
) -> Result<(), SyntheticBenchmarkError> {
    if plan.selected_option_indices.len() != plan.selected_option_ids.len() {
        return Err(SyntheticBenchmarkError::PlanIndexIdMismatch);
    }

    let mut selected = HashSet::with_capacity(plan.selected_option_ids.len());
    let mut selected_cost = 0_u64;
    for (index, option_id) in plan
        .selected_option_indices
        .iter()
        .zip(&plan.selected_option_ids)
    {
        let option = options
            .get(*index)
            .ok_or(SyntheticBenchmarkError::PlanIndexIdMismatch)?;
        if option.id != *option_id {
            return Err(SyntheticBenchmarkError::PlanIndexIdMismatch);
        }
        if !selected.insert(option_id.as_str()) {
            return Err(SyntheticBenchmarkError::DuplicateSelectedOption);
        }
        let known = option_map
            .get(option_id.as_str())
            .ok_or(SyntheticBenchmarkError::UnknownPlannedOption)?;
        selected_cost = selected_cost
            .checked_add(known.estimated_cost_units)
            .ok_or(SyntheticBenchmarkError::PlanSpendMismatch)?;
    }

    let mut skipped = HashSet::with_capacity(plan.skipped_option_ids.len());
    for option_id in &plan.skipped_option_ids {
        if !skipped.insert(option_id.as_str()) {
            return Err(SyntheticBenchmarkError::DuplicateSkippedOption);
        }
        if !option_map.contains_key(option_id.as_str()) {
            return Err(SyntheticBenchmarkError::UnknownPlannedOption);
        }
        if selected.contains(option_id.as_str()) {
            return Err(SyntheticBenchmarkError::PlanOptionPartitionMismatch);
        }
    }

    if selected.len() + skipped.len() != options.len()
        || option_map
            .keys()
            .any(|id| !selected.contains(id) && !skipped.contains(id))
    {
        return Err(SyntheticBenchmarkError::PlanOptionPartitionMismatch);
    }
    if selected_cost != plan.estimated_cost_units {
        return Err(SyntheticBenchmarkError::PlanSpendMismatch);
    }
    if selected_cost > plan.budget_units {
        return Err(SyntheticBenchmarkError::PlanExceedsBudget);
    }
    if plan.budget_units - selected_cost != plan.unused_budget_units {
        return Err(SyntheticBenchmarkError::PlanBudgetAccountingMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment_selection::{
        rank_experiments, select_under_budget, ExperimentOption, ExperimentSelectionPolicy,
    };

    const SCENARIO: &str = "known-truth-acoustic-v1";
    const PROFILE: &str = "synthetic-acoustic-profile-v1";
    const COST: &str = "synthetic-cost-unit";

    fn fixture() -> (Vec<ExperimentOption>, SyntheticBenchmarkCase) {
        let options = vec![
            ExperimentOption::new("cheap-noise", "design-a", "analytic", 1, 0, 0, 4)
                .with_contract(PROFILE, COST),
            ExperimentOption::new("target-test", "design-b", "bench", 4, 2, 2, 8)
                .with_contract(PROFILE, COST),
            ExperimentOption::new("heldout-transfer", "design-c", "high-fidelity", 8, 1, 5, 20)
                .with_contract(PROFILE, COST),
            ExperimentOption::new("expensive-decoy", "design-d", "simulation", 6, 0, 0, 1)
                .with_contract(PROFILE, COST),
        ];
        let case = SyntheticBenchmarkCase::new(
            SCENARIO,
            PROFILE,
            COST,
            vec![
                OracleQuestion::new("ambient-noise", 2),
                OracleQuestion::new("pipe-material", 30),
                OracleQuestion::new("false-alarm", 25),
                OracleQuestion::new("cross-day-transfer", 50),
                OracleQuestion::new("calibration", 10),
                OracleQuestion::new("decoy-detail", 1),
            ],
            vec![
                SyntheticExperimentOutcome::new("cheap-noise", vec!["ambient-noise".into()], 20),
                SyntheticExperimentOutcome::new(
                    "target-test",
                    vec!["pipe-material".into(), "false-alarm".into()],
                    6,
                ),
                SyntheticExperimentOutcome::new(
                    "heldout-transfer",
                    vec!["cross-day-transfer".into(), "calibration".into()],
                    2,
                ),
                SyntheticExperimentOutcome::new("expensive-decoy", vec!["decoy-detail".into()], 30),
            ],
        );
        (options, case)
    }

    #[test]
    fn decision_value_counts_unique_oracle_questions_only_once() {
        let (options, case) = fixture();
        let mut overlapping = case.clone();
        overlapping.outcomes[1].resolved_question_ids.push("ambient-noise".into());
        overlapping.questions[0].value_units = 2;
        let ranking = rank_experiments(
            &options,
            PROFILE,
            COST,
            ExperimentSelectionPolicy::CheapestFirst,
        )
        .unwrap();
        let plan = select_under_budget(&options, &ranking, 5).unwrap();
        let receipt = score_synthetic_plan(&options, &overlapping, &plan).unwrap();

        assert_eq!(receipt.resolved_question_ids, vec!["ambient-noise", "false-alarm", "pipe-material"]);
        assert_eq!(receipt.realized_decision_value_units, 57);
    }

    #[test]
    fn validation_debt_is_not_hidden_inside_immediate_spend() {
        let (options, case) = fixture();
        let receipt = run_synthetic_benchmark(
            &options,
            &case,
            10,
            ExperimentSelectionPolicy::CheapestFirst,
        )
        .unwrap();

        assert_eq!(receipt.immediate_spend_units, 5);
        assert_eq!(receipt.deferred_validation_cost_units, 26);
        assert_eq!(receipt.end_to_end_cost_units, 31);
        assert_eq!(receipt.unused_budget_units, 5);
    }

    #[test]
    fn uncertainty_proxy_can_improve_decision_value_at_same_end_to_end_cost() {
        let (options, case) = fixture();
        let cheapest = run_synthetic_benchmark(
            &options,
            &case,
            9,
            ExperimentSelectionPolicy::CheapestFirst,
        )
        .unwrap();
        let proxy = run_synthetic_benchmark(
            &options,
            &case,
            9,
            ExperimentSelectionPolicy::UncertaintyReductionPerCost,
        )
        .unwrap();

        assert_eq!(cheapest.end_to_end_cost_units, proxy.end_to_end_cost_units);
        assert!(proxy.realized_decision_value_units > cheapest.realized_decision_value_units);
        assert_eq!(proxy.realized_decision_value_units, 62);
        assert_eq!(cheapest.realized_decision_value_units, 57);
    }

    #[test]
    fn repeated_scoring_produces_identical_receipts() {
        let (options, case) = fixture();
        let first = run_synthetic_benchmark(
            &options,
            &case,
            9,
            ExperimentSelectionPolicy::SeededPseudoRandom { seed: 17 },
        )
        .unwrap();
        let second = run_synthetic_benchmark(
            &options,
            &case,
            9,
            ExperimentSelectionPolicy::SeededPseudoRandom { seed: 17 },
        )
        .unwrap();

        assert_eq!(first, second);
    }

    #[test]
    fn missing_oracle_outcomes_fail_closed() {
        let (options, mut case) = fixture();
        case.outcomes.pop();

        assert_eq!(
            run_synthetic_benchmark(
                &options,
                &case,
                9,
                ExperimentSelectionPolicy::CheapestFirst,
            ),
            Err(SyntheticBenchmarkError::MissingOutcome)
        );
    }

    #[test]
    fn unknown_question_and_duplicate_question_references_fail_closed() {
        let (options, mut case) = fixture();
        case.outcomes[0].resolved_question_ids = vec!["not-in-case".into()];

        assert_eq!(
            run_synthetic_benchmark(
                &options,
                &case,
                9,
                ExperimentSelectionPolicy::CheapestFirst,
            ),
            Err(SyntheticBenchmarkError::UnknownResolvedQuestion)
        );

        let (options, mut case) = fixture();
        case.outcomes[0].resolved_question_ids =
            vec!["ambient-noise".into(), "ambient-noise".into()];
        assert_eq!(
            run_synthetic_benchmark(
                &options,
                &case,
                9,
                ExperimentSelectionPolicy::CheapestFirst,
            ),
            Err(SyntheticBenchmarkError::DuplicateResolvedQuestionId)
        );
    }

    #[test]
    fn mismatched_contract_and_zero_cost_fail_closed() {
        let (mut options, case) = fixture();
        options[0].cost_unit_id = "other-unit".into();
        assert_eq!(
            run_synthetic_benchmark(
                &options,
                &case,
                9,
                ExperimentSelectionPolicy::CheapestFirst,
            ),
            Err(SyntheticBenchmarkError::OptionContractMismatch)
        );

        let (mut options, case) = fixture();
        options[0].estimated_cost_units = 0;
        assert_eq!(
            run_synthetic_benchmark(
                &options,
                &case,
                9,
                ExperimentSelectionPolicy::CheapestFirst,
            ),
            Err(SyntheticBenchmarkError::Selection(
                ExperimentSelectionError::ZeroEstimatedCost
            ))
        );
    }

    #[test]
    fn plan_from_other_option_pool_is_rejected() {
        let (options, case) = fixture();
        let ranking = rank_experiments(
            &options,
            PROFILE,
            COST,
            ExperimentSelectionPolicy::CheapestFirst,
        )
        .unwrap();
        let plan = select_under_budget(&options, &ranking, 9).unwrap();
        let mut changed = options.clone();
        changed[0].estimated_cost_units += 1;

        assert_eq!(
            score_synthetic_plan(&changed, &case, &plan),
            Err(SyntheticBenchmarkError::PlanSpendMismatch)
        );
    }

    #[test]
    fn overflowing_value_and_validation_cost_fail_closed() {
        let (options, mut case) = fixture();
        case.questions[0].value_units = u64::MAX;
        case.questions[1].value_units = 1;
        let plan = {
            let ranking = rank_experiments(
                &options,
                PROFILE,
                COST,
                ExperimentSelectionPolicy::CheapestFirst,
            )
            .unwrap();
            select_under_budget(&options, &ranking, 5).unwrap()
        };
        assert_eq!(
            score_synthetic_plan(&options, &case, &plan),
            Err(SyntheticBenchmarkError::ValueOverflow)
        );

        let (options, mut case) = fixture();
        case.outcomes[0].deferred_validation_cost_units = u64::MAX;
        case.outcomes[1].deferred_validation_cost_units = 1;
        assert_eq!(
            run_synthetic_benchmark(
                &options,
                &case,
                5,
                ExperimentSelectionPolicy::CheapestFirst,
            ),
            Err(SyntheticBenchmarkError::ValidationCostOverflow)
        );
    }
}
