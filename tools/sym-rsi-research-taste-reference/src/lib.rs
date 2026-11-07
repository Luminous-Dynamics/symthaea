//! Research-taste synthetic fixture validation and a finite-state EVSI reference.
//!
//! This validates the declared test pack and computes a preregistered decision
//! quantity. It does not judge open-ended research proposals, authenticate
//! evidence, predict real outcomes, or establish scientific truth.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;

const FIXTURE_SCHEMA_VERSION: &str = "sym-rsi-research-taste-decision-fixtures-v0";
const FIXTURE_STATUS: &str = "SYNTHETIC_POLICY_CASES_NOT_EXPERT_GOLD_OR_BENCHMARK_RESULTS";
const EXPECTED_FIXTURES: usize = 16;
const PROBABILITY_TOLERANCE: f64 = 1e-9;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    DirectionSelection,
    ApproachDesign,
    ExperimentDesign,
    ResultInterpretation,
    Transfer,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionFixture {
    pub fixture_id: String,
    pub stage: Stage,
    pub scenario: String,
    pub facts: Value,
    pub expected_action: String,
    pub required_response: Vec<String>,
    pub prohibited_inference: Vec<String>,
    pub evidence_to_request: Vec<String>,
    pub why: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationPolicy {
    pub adjudication: String,
    pub scoring: String,
    pub denominator: String,
    pub no_single_oracle: String,
    pub transfer: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionFixturePack {
    pub schema_version: String,
    pub status: String,
    pub source_issue: String,
    pub purpose: String,
    pub fixtures: Vec<DecisionFixture>,
    pub evaluation_policy: EvaluationPolicy,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct FixtureValidationReport {
    pub valid_fixture_pack: bool,
    pub fixture_count: usize,
    pub stage_counts: [usize; 5],
    pub issues: Vec<String>,
    pub limitations: Vec<String>,
}

const FIXTURE_CONTRACT: [(&str, Stage, &str); EXPECTED_FIXTURES] = [
    ("RT-D01", Stage::DirectionSelection, "select_A_for_pilot"),
    ("RT-D02", Stage::DirectionSelection, "deprioritize_as_duplicate"),
    ("RT-D03", Stage::DirectionSelection, "run_low_cost_information_pilot"),
    ("RT-A01", Stage::ApproachDesign, "return_for_mechanism_and_falsifiability"),
    ("RT-A02", Stage::ApproachDesign, "accept_for_discriminating_pilot"),
    ("RT-A03", Stage::ApproachDesign, "classify_as_reproduction_not_novelty"),
    ("RT-E01", Stage::ExperimentDesign, "redesign_non_discriminating_experiment"),
    ("RT-E02", Stage::ExperimentDesign, "run_preregistered_paired_test"),
    ("RT-E03", Stage::ExperimentDesign, "stage_low_cost_screen_before_confirmation"),
    ("RT-I01", Stage::ResultInterpretation, "abstain_or_replicate"),
    ("RT-I02", Stage::ResultInterpretation, "reject_current_mechanism_and_test_alternative"),
    ("RT-I03", Stage::ResultInterpretation, "invalidate_result_and_requalify"),
    ("RT-T01", Stage::Transfer, "run_blinded_mechanism_transfer_test"),
    ("RT-T02", Stage::Transfer, "reject_transfer_claim"),
    ("RT-T03", Stage::Transfer, "reject_successor_promotion"),
    ("RT-T04", Stage::Transfer, "reject_research_state_continuity_claim"),
];

fn stage_index(stage: Stage) -> usize {
    match stage {
        Stage::DirectionSelection => 0,
        Stage::ApproachDesign => 1,
        Stage::ExperimentDesign => 2,
        Stage::ResultInterpretation => 3,
        Stage::Transfer => 4,
    }
}

fn non_empty(value: &str) -> bool {
    !value.trim().is_empty()
}

pub fn validate_decision_fixtures(pack: &DecisionFixturePack) -> FixtureValidationReport {
    let mut issues = Vec::new();
    let mut stage_counts = [0usize; 5];
    let mut fixture_ids = BTreeSet::new();

    if pack.schema_version != FIXTURE_SCHEMA_VERSION {
        issues.push("SCHEMA_VERSION: unsupported fixture schema".to_owned());
    }
    if pack.status != FIXTURE_STATUS {
        issues.push("STATUS: pack must remain explicitly synthetic".to_owned());
    }
    if pack.source_issue != "#10877" {
        issues.push("SOURCE_ISSUE: expected #10877".to_owned());
    }
    if !non_empty(&pack.purpose) {
        issues.push("PURPOSE: missing".to_owned());
    }
    for (name, value) in [
        ("adjudication", pack.evaluation_policy.adjudication.as_str()),
        ("scoring", pack.evaluation_policy.scoring.as_str()),
        ("denominator", pack.evaluation_policy.denominator.as_str()),
        ("no_single_oracle", pack.evaluation_policy.no_single_oracle.as_str()),
        ("transfer", pack.evaluation_policy.transfer.as_str()),
    ] {
        if !non_empty(value) {
            issues.push(format!("POLICY_FIELD: {name}"));
        }
    }
    if pack.fixtures.len() != EXPECTED_FIXTURES {
        issues.push(format!(
            "FIXTURE_COUNT: expected {EXPECTED_FIXTURES}, found {}",
            pack.fixtures.len()
        ));
    }

    for fixture in &pack.fixtures {
        if !fixture_ids.insert(fixture.fixture_id.clone()) {
            issues.push(format!("DUPLICATE_FIXTURE_ID: {}", fixture.fixture_id));
        }
        if !non_empty(&fixture.scenario) {
            issues.push(format!("SCENARIO_MISSING: {}", fixture.fixture_id));
        }
        if !fixture.facts.is_object() || fixture.facts.as_object().is_some_and(|o| o.is_empty()) {
            issues.push(format!("FACTS_MISSING: {}", fixture.fixture_id));
        }
        if !non_empty(&fixture.why) {
            issues.push(format!("RATIONALE_MISSING: {}", fixture.fixture_id));
        }
        for (field, values) in [
            ("required_response", fixture.required_response.as_slice()),
            ("prohibited_inference", fixture.prohibited_inference.as_slice()),
            ("evidence_to_request", fixture.evidence_to_request.as_slice()),
        ] {
            if values.is_empty() || values.iter().any(|value| !non_empty(value)) {
                issues.push(format!("{field}: empty or invalid for {}", fixture.fixture_id));
            }
            if values.iter().collect::<BTreeSet<_>>().len() != values.len() {
                issues.push(format!("{field}: duplicate entries for {}", fixture.fixture_id));
            }
        }

        stage_counts[stage_index(fixture.stage)] += 1;

        match FIXTURE_CONTRACT.iter().find(|(id, _, _)| *id == fixture.fixture_id) {
            Some((_, expected_stage, expected_action)) => {
                if fixture.stage != *expected_stage {
                    issues.push(format!("STAGE_MISMATCH: {}", fixture.fixture_id));
                }
                if fixture.expected_action != *expected_action {
                    issues.push(format!("ACTION_MISMATCH: {}", fixture.fixture_id));
                }
            }
            None => issues.push(format!("UNKNOWN_FIXTURE_ID: {}", fixture.fixture_id)),
        }
    }

    let expected_stage_counts = [3, 3, 3, 3, 4];
    if stage_counts != expected_stage_counts {
        issues.push(format!(
            "STAGE_COUNTS: expected {expected_stage_counts:?}, found {stage_counts:?}"
        ));
    }
    for (id, _, _) in FIXTURE_CONTRACT {
        if !fixture_ids.contains(id) {
            issues.push(format!("REQUIRED_FIXTURE_MISSING: {id}"));
        }
    }

    issues.sort();
    issues.dedup();

    FixtureValidationReport {
        valid_fixture_pack: issues.is_empty(),
        fixture_count: pack.fixtures.len(),
        stage_counts,
        issues,
        limitations: vec![
            "A valid synthetic fixture pack does not establish that an agent makes the expected decision."
                .to_owned(),
            "Expected actions are rule-derived policy targets, not expert gold labels or scientific outcomes."
                .to_owned(),
            "This validator does not authenticate evidence, task custody, or domain-scientific truth."
                .to_owned(),
        ],
    }
}

pub fn parse_and_validate_fixtures(
    contents: &str,
) -> Result<FixtureValidationReport, serde_json::Error> {
    let pack: DecisionFixturePack = serde_json::from_str(contents)?;
    Ok(validate_decision_fixtures(&pack))
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentOutcome {
    pub outcome_id: String,
    /// P(this outcome | each state), in the same order as states.
    pub likelihood_by_state: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvsiProblem {
    pub decision_id: String,
    pub states: Vec<String>,
    /// Prior probability for each state. Must sum to one.
    pub prior: Vec<f64>,
    pub actions: Vec<String>,
    /// Utility[action_index][state_index].
    pub utilities: Vec<Vec<f64>>,
    pub outcomes: Vec<ExperimentOutcome>,
    /// Direct resource cost converted into utility units.
    pub direct_experiment_cost: f64,
    /// Best foregone alternative use of the committed resources, net of direct cost.
    pub opportunity_cost: f64,
    /// Non-negative downside loss for each outcome, in utility units.
    pub downside_loss_by_outcome: Vec<f64>,
    /// Confidence level for upper-tail CVaR (for example, 0.95).
    pub cvar_confidence: f64,
    /// Maximum admissible CVaR downside loss, in utility units.
    pub max_cvar_downside_loss: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct OutcomeDecision {
    pub outcome_id: String,
    pub predictive_probability: f64,
    pub posterior: Vec<f64>,
    pub best_action: String,
    pub expected_utility_after_observation: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct EvsiReport {
    pub decision_id: String,
    pub best_action_without_experiment: String,
    pub expected_utility_without_experiment: f64,
    pub expected_utility_with_experiment: f64,
    pub expected_value_of_sample_information: f64,
    pub direct_experiment_cost: f64,
    pub opportunity_cost: f64,
    pub net_evsi: f64,
    pub expected_downside_loss: f64,
    pub cvar_confidence: f64,
    pub cvar_downside_loss: f64,
    pub max_cvar_downside_loss: f64,
    pub risk_limit_passed: bool,
    pub decision_eligible: bool,
    pub outcome_decisions: Vec<OutcomeDecision>,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvsiError {
    EmptyIdentifier(&'static str),
    InsufficientStates,
    InsufficientActions,
    InsufficientOutcomes,
    DimensionMismatch(&'static str),
    DuplicateIdentifier(&'static str, String),
    InvalidProbability(&'static str),
    ProbabilityMassMismatch(&'static str),
    NonFiniteUtility,
    InvalidCost,
    InvalidRisk(&'static str),
}

impl std::fmt::Display for EvsiError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for EvsiError {}

fn duplicate_or_empty(values: &[String], kind: &'static str) -> Result<(), EvsiError> {
    let mut seen = BTreeSet::new();
    for value in values {
        if !non_empty(value) {
            return Err(EvsiError::EmptyIdentifier(kind));
        }
        if !seen.insert(value) {
            return Err(EvsiError::DuplicateIdentifier(kind, value.clone()));
        }
    }
    Ok(())
}

fn approx_one(value: f64) -> bool {
    (value - 1.0).abs() <= PROBABILITY_TOLERANCE
}

fn validate_evsi(problem: &EvsiProblem) -> Result<(), EvsiError> {
    if !non_empty(&problem.decision_id) {
        return Err(EvsiError::EmptyIdentifier("decision_id"));
    }
    if problem.states.len() < 2 {
        return Err(EvsiError::InsufficientStates);
    }
    if problem.actions.is_empty() {
        return Err(EvsiError::InsufficientActions);
    }
    if problem.outcomes.len() < 2 {
        return Err(EvsiError::InsufficientOutcomes);
    }
    duplicate_or_empty(&problem.states, "state")?;
    duplicate_or_empty(&problem.actions, "action")?;

    if problem.prior.len() != problem.states.len() {
        return Err(EvsiError::DimensionMismatch("prior"));
    }
    if problem.utilities.len() != problem.actions.len()
        || problem.utilities.iter().any(|row| row.len() != problem.states.len())
    {
        return Err(EvsiError::DimensionMismatch("utilities"));
    }
    if problem.utilities.iter().flatten().any(|value| !value.is_finite()) {
        return Err(EvsiError::NonFiniteUtility);
    }
    if problem.prior.iter().any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0) {
        return Err(EvsiError::InvalidProbability("prior"));
    }
    if !approx_one(problem.prior.iter().sum()) {
        return Err(EvsiError::ProbabilityMassMismatch("prior"));
    }

    let outcome_ids: Vec<String> = problem.outcomes.iter().map(|o| o.outcome_id.clone()).collect();
    duplicate_or_empty(&outcome_ids, "outcome")?;
    for outcome in &problem.outcomes {
        if outcome.likelihood_by_state.len() != problem.states.len() {
            return Err(EvsiError::DimensionMismatch("likelihood_by_state"));
        }
        if outcome
            .likelihood_by_state
            .iter()
            .any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
        {
            return Err(EvsiError::InvalidProbability("likelihood"));
        }
    }
    for state_index in 0..problem.states.len() {
        let mass: f64 = problem
            .outcomes
            .iter()
            .map(|outcome| outcome.likelihood_by_state[state_index])
            .sum();
        if !approx_one(mass) {
            return Err(EvsiError::ProbabilityMassMismatch("outcomes_given_state"));
        }
    }
    if !problem.direct_experiment_cost.is_finite()
        || problem.direct_experiment_cost < 0.0
        || !problem.opportunity_cost.is_finite()
        || problem.opportunity_cost < 0.0
    {
        return Err(EvsiError::InvalidCost);
    }
    if problem.downside_loss_by_outcome.len() != problem.outcomes.len() {
        return Err(EvsiError::DimensionMismatch("downside_loss_by_outcome"));
    }
    if problem
        .downside_loss_by_outcome
        .iter()
        .any(|loss| !loss.is_finite() || *loss < 0.0)
    {
        return Err(EvsiError::InvalidRisk("downside_loss_by_outcome"));
    }
    if !problem.cvar_confidence.is_finite()
        || !(0.0..1.0).contains(&problem.cvar_confidence)
        || !problem.max_cvar_downside_loss.is_finite()
        || problem.max_cvar_downside_loss < 0.0
    {
        return Err(EvsiError::InvalidRisk("cvar_policy"));
    }

    Ok(())
}

fn best_action(problem: &EvsiProblem, probabilities: &[f64]) -> (usize, f64) {
    let mut best_index = 0usize;
    let mut best_value = f64::NEG_INFINITY;
    for (action_index, utilities) in problem.utilities.iter().enumerate() {
        let expected: f64 = utilities
            .iter()
            .zip(probabilities)
            .map(|(utility, probability)| utility * probability)
            .sum();
        if expected > best_value {
            best_value = expected;
            best_index = action_index;
        }
    }
    (best_index, best_value)
}

/// Compute finite-state expected value of sample information (EVSI).
///
/// EVSI is the expected improvement in the best downstream decision after
/// observing an experiment, before subtracting direct or opportunity cost.
/// Downside risk is reported and gated separately; it is not hidden inside an
/// arbitrary risk-adjusted utility penalty. This is not Shannon information
/// gain: an informative observation can have zero EVSI if it cannot improve
/// the optimal decision.
fn weighted_upper_tail_cvar(losses: &[f64], probabilities: &[f64], confidence: f64) -> f64 {
    let total_probability: f64 = probabilities.iter().sum();
    let tail_mass = 1.0 - confidence;
    let mut weighted: Vec<(f64, f64)> = losses
        .iter()
        .copied()
        .zip(probabilities.iter().map(|p| p / total_probability))
        .filter(|(_, probability)| *probability > 0.0)
        .collect();
    weighted.sort_by(|left, right| right.0.total_cmp(&left.0));

    let mut remaining = tail_mass;
    let mut tail_loss = 0.0;
    for (loss, probability) in weighted {
        let included = probability.min(remaining);
        tail_loss += loss * included;
        remaining -= included;
        if remaining <= PROBABILITY_TOLERANCE {
            break;
        }
    }
    tail_loss / tail_mass
}

/// Compute finite-state expected value of sample information (EVSI).
///
/// EVSI is the expected improvement in the best downstream decision after
/// observing an experiment, before subtracting direct or opportunity cost.
/// Downside risk is reported and gated separately; it is not hidden inside an
/// arbitrary risk-adjusted utility penalty. This is not Shannon information
/// gain: an informative observation can have zero EVSI if it cannot improve
/// the optimal decision.
pub fn calculate_evsi(problem: &EvsiProblem) -> Result<EvsiReport, EvsiError> {
    validate_evsi(problem)?;

    let (prior_action, prior_value) = best_action(problem, &problem.prior);
    let mut outcomes = Vec::with_capacity(problem.outcomes.len());
    let mut value_with_experiment = 0.0;

    for outcome in &problem.outcomes {
        let predictive_probability: f64 = problem
            .prior
            .iter()
            .zip(&outcome.likelihood_by_state)
            .map(|(prior, likelihood)| prior * likelihood)
            .sum();

        if predictive_probability <= 0.0 {
            outcomes.push(OutcomeDecision {
                outcome_id: outcome.outcome_id.clone(),
                predictive_probability,
                posterior: vec![0.0; problem.states.len()],
                best_action: "UNDEFINED_ZERO_PROBABILITY_OUTCOME".to_owned(),
                expected_utility_after_observation: 0.0,
            });
            continue;
        }

        let posterior: Vec<f64> = problem
            .prior
            .iter()
            .zip(&outcome.likelihood_by_state)
            .map(|(prior, likelihood)| prior * likelihood / predictive_probability)
            .collect();

        let (action_index, expected_utility) = best_action(problem, &posterior);
        value_with_experiment += predictive_probability * expected_utility;
        outcomes.push(OutcomeDecision {
            outcome_id: outcome.outcome_id.clone(),
            predictive_probability,
            posterior,
            best_action: problem.actions[action_index].clone(),
            expected_utility_after_observation: expected_utility,
        });
    }

    let mut evsi = value_with_experiment - prior_value;
    if evsi < 0.0 && evsi.abs() <= 1e-8 {
        evsi = 0.0;
    }
    if evsi < 0.0 {
        return Err(EvsiError::ProbabilityMassMismatch("negative_evsi_numerical_or_model_error"));
    }

    let outcome_probabilities: Vec<f64> = outcomes
        .iter()
        .map(|outcome| outcome.predictive_probability)
        .collect();
    let expected_downside_loss: f64 = problem
        .downside_loss_by_outcome
        .iter()
        .zip(&outcome_probabilities)
        .map(|(loss, probability)| loss * probability)
        .sum();
    let cvar_downside_loss = weighted_upper_tail_cvar(
        &problem.downside_loss_by_outcome,
        &outcome_probabilities,
        problem.cvar_confidence,
    );
    let risk_limit_passed =
        cvar_downside_loss <= problem.max_cvar_downside_loss + PROBABILITY_TOLERANCE;
    let net_evsi = evsi - problem.direct_experiment_cost - problem.opportunity_cost;

    Ok(EvsiReport {
        decision_id: problem.decision_id.clone(),
        best_action_without_experiment: problem.actions[prior_action].clone(),
        expected_utility_without_experiment: prior_value,
        expected_utility_with_experiment: value_with_experiment,
        expected_value_of_sample_information: evsi,
        direct_experiment_cost: problem.direct_experiment_cost,
        opportunity_cost: problem.opportunity_cost,
        net_evsi,
        expected_downside_loss,
        cvar_confidence: problem.cvar_confidence,
        cvar_downside_loss,
        max_cvar_downside_loss: problem.max_cvar_downside_loss,
        risk_limit_passed,
        decision_eligible: net_evsi > 0.0 && risk_limit_passed,
        outcome_decisions: outcomes,
        limitations: vec![
            "EVSI is conditional on the declared state space, priors, likelihoods, utility and cost model."
                .to_owned(),
            "Direct cost and foregone-alternative opportunity cost are separate, declared utility-unit inputs."
                .to_owned(),
            "Downside losses and CVaR are conditional on the supplied outcome-loss model and risk threshold."
                .to_owned(),
            "A passing risk gate does not capture unmodeled outcomes or establish that the experiment is safe."
                .to_owned(),
            "It does not measure entropy reduction, scientific truth, or model calibration.".to_owned(),
            "The result is not causal evidence that the experiment will deliver the modeled outcomes."
                .to_owned(),
        ],
    })
}

pub fn parse_and_calculate_evsi(contents: &str) -> Result<EvsiReport, Box<dyn std::error::Error>> {
    let problem: EvsiProblem = serde_json::from_str(contents)?;
    Ok(calculate_evsi(&problem)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const FIXTURES: &str =
        include_str!("../../../docs/research/fixtures/SYM_RSI_RESEARCH_TASTE_DECISION_FIXTURES_V0_2026-11-07.json");
    const EVSI_EXAMPLE: &str =
        include_str!("../../../docs/research/fixtures/SYM_RSI_RESEARCH_TASTE_EVSI_EXAMPLE_V0_2026-11-07.json");

    fn valid_pack() -> DecisionFixturePack {
        serde_json::from_str(FIXTURES).expect("checked-in decision fixtures parse")
    }

    fn valid_problem() -> EvsiProblem {
        EvsiProblem {
            decision_id: "weather-umbrella".to_owned(),
            states: vec!["sun".to_owned(), "rain".to_owned()],
            prior: vec![0.5, 0.5],
            actions: vec!["umbrella".to_owned(), "no_umbrella".to_owned()],
            utilities: vec![vec![8.0, 6.0], vec![10.0, 0.0]],
            outcomes: vec![
                ExperimentOutcome { outcome_id: "forecast_sun".to_owned(), likelihood_by_state: vec![0.8, 0.2] },
                ExperimentOutcome { outcome_id: "forecast_rain".to_owned(), likelihood_by_state: vec![0.2, 0.8] },
            ],
            direct_experiment_cost: 0.1,
            opportunity_cost: 0.0,
            downside_loss_by_outcome: vec![0.0, 0.0],
            cvar_confidence: 0.95,
            max_cvar_downside_loss: 0.0,
        }
    }

    fn pack_value() -> Value {
        serde_json::to_value(valid_pack()).expect("fixture pack serializes")
    }

    #[test]
    fn accepts_checked_in_decision_fixture_pack() {
        let report = validate_decision_fixtures(&valid_pack());
        assert!(report.valid_fixture_pack, "{:?}", report.issues);
        assert_eq!(report.fixture_count, 16);
        assert_eq!(report.stage_counts, [3, 3, 3, 3, 4]);
    }

    #[test]
    fn rejects_wrong_synthetic_status() {
        let mut value = pack_value();
        value["status"] = Value::String("EXPERT_GOLD".to_owned());
        let pack: DecisionFixturePack = serde_json::from_value(value).expect("mutated pack parses");
        let report = validate_decision_fixtures(&pack);
        assert!(report.issues.iter().any(|issue| issue.starts_with("STATUS:")));
    }

    #[test]
    fn rejects_duplicate_fixture_id() {
        let mut value = pack_value();
        let id = value["fixtures"][0]["fixture_id"].clone();
        value["fixtures"][1]["fixture_id"] = id;
        let pack: DecisionFixturePack = serde_json::from_value(value).expect("mutated pack parses");
        let report = validate_decision_fixtures(&pack);
        assert!(report.issues.iter().any(|issue| issue.starts_with("DUPLICATE_FIXTURE_ID:")));
    }

    #[test]
    fn rejects_changed_expected_action() {
        let mut value = pack_value();
        value["fixtures"][0]["expected_action"] = Value::String("accept_everything".to_owned());
        let pack: DecisionFixturePack = serde_json::from_value(value).expect("mutated pack parses");
        let report = validate_decision_fixtures(&pack);
        assert!(report.issues.iter().any(|issue| issue.starts_with("ACTION_MISMATCH:")));
    }

    #[test]
    fn rejects_missing_prohibited_inference() {
        let mut value = pack_value();
        value["fixtures"][0]["prohibited_inference"] = Value::Array(Vec::new());
        let pack: DecisionFixturePack = serde_json::from_value(value).expect("mutated pack parses");
        let report = validate_decision_fixtures(&pack);
        assert!(report.issues.iter().any(|issue| issue.starts_with("prohibited_inference:")));
    }

    #[test]
    fn checked_in_evsi_example_matches_reference_calculation() {
        let report = parse_and_calculate_evsi(EVSI_EXAMPLE).expect("checked-in EVSI example parses");
        assert!((report.expected_value_of_sample_information - 0.2).abs() < 1e-9);
        assert!((report.net_evsi - 0.1).abs() < 1e-9);
    }

    #[test]
    fn computes_positive_evsi_and_net_value() {
        let report = calculate_evsi(&valid_problem()).expect("valid finite decision problem");
        assert!((report.expected_value_of_sample_information - 0.2).abs() < 1e-9);
        assert!((report.net_evsi - 0.1).abs() < 1e-9);
        assert_eq!(report.best_action_without_experiment, "umbrella");
        assert_eq!(report.outcome_decisions[0].best_action, "no_umbrella");
        assert_eq!(report.outcome_decisions[1].best_action, "umbrella");
    }

    #[test]
    fn informative_observation_has_zero_evsi_when_it_cannot_change_the_decision() {
        let mut problem = valid_problem();
        problem.utilities = vec![vec![10.0, 10.0], vec![0.0, 0.0]];
        let report = calculate_evsi(&problem).expect("valid finite decision problem");
        assert!(report.expected_value_of_sample_information.abs() < 1e-9);
        assert!(report.net_evsi < 0.0);
        assert!(report.outcome_decisions.iter().all(|item| item.best_action == "umbrella"));
    }

    #[test]
    fn expensive_experiment_has_negative_net_evsi() {
        let mut problem = valid_problem();
        problem.direct_experiment_cost = 1.0;
        let report = calculate_evsi(&problem).expect("valid finite decision problem");
        assert!(report.expected_value_of_sample_information > 0.0);
        assert!(report.net_evsi < 0.0);
    }

    #[test]
    fn keeps_direct_and_opportunity_cost_separate_from_gross_evsi() {
        let mut problem = valid_problem();
        problem.direct_experiment_cost = 0.05;
        problem.opportunity_cost = 0.20;
        let report = calculate_evsi(&problem).expect("valid finite decision problem");
        assert!((report.expected_value_of_sample_information - 0.2).abs() < 1e-9);
        assert!((report.net_evsi + 0.05).abs() < 1e-9);
        assert!(!report.decision_eligible);
    }

    #[test]
    fn positive_net_value_can_fail_the_downside_risk_gate() {
        let mut problem = valid_problem();
        problem.direct_experiment_cost = 0.05;
        problem.downside_loss_by_outcome = vec![0.0, 10.0];
        problem.max_cvar_downside_loss = 1.0;
        let report = calculate_evsi(&problem).expect("valid risk model");
        assert!(report.net_evsi > 0.0);
        assert!((report.expected_downside_loss - 5.0).abs() < 1e-9);
        assert!((report.cvar_downside_loss - 10.0).abs() < 1e-9);
        assert!(!report.risk_limit_passed);
        assert!(!report.decision_eligible);
    }

    #[test]
    fn weighted_cvar_uses_only_the_declared_upper_tail_mass() {
        let mut problem = valid_problem();
        problem.outcomes = vec![
            ExperimentOutcome { outcome_id: "rare_severe".to_owned(), likelihood_by_state: vec![0.02, 0.02] },
            ExperimentOutcome { outcome_id: "moderate".to_owned(), likelihood_by_state: vec![0.08, 0.08] },
            ExperimentOutcome { outcome_id: "benign".to_owned(), likelihood_by_state: vec![0.90, 0.90] },
        ];
        problem.downside_loss_by_outcome = vec![100.0, 10.0, 0.0];
        problem.max_cvar_downside_loss = 50.0;
        let report = calculate_evsi(&problem).expect("valid three-outcome risk model");
        assert!((report.cvar_downside_loss - 46.0).abs() < 1e-9);
        assert!(report.risk_limit_passed);
    }

    #[test]
    fn rejects_downside_loss_dimension_mismatch() {
        let mut problem = valid_problem();
        problem.downside_loss_by_outcome.pop();
        assert_eq!(
            calculate_evsi(&problem).expect_err("loss vector must match outcome count"),
            EvsiError::DimensionMismatch("downside_loss_by_outcome")
        );
    }

    #[test]
    fn rejects_invalid_cvar_policy() {
        let mut problem = valid_problem();
        problem.cvar_confidence = 1.0;
        assert_eq!(
            calculate_evsi(&problem).expect_err("CVaR confidence must be below one"),
            EvsiError::InvalidRisk("cvar_policy")
        );
    }

    #[test]
    fn rejects_prior_with_wrong_probability_mass() {
        let mut problem = valid_problem();
        problem.prior = vec![0.8, 0.8];
        assert_eq!(
            calculate_evsi(&problem).expect_err("prior mass exceeds one"),
            EvsiError::ProbabilityMassMismatch("prior")
        );
    }

    #[test]
    fn rejects_outcome_likelihoods_that_do_not_cover_each_state() {
        let mut problem = valid_problem();
        problem.outcomes[0].likelihood_by_state = vec![0.8, 0.8];
        problem.outcomes[1].likelihood_by_state = vec![0.1, 0.1];
        assert_eq!(
            calculate_evsi(&problem).expect_err("likelihood mass is incomplete"),
            EvsiError::ProbabilityMassMismatch("outcomes_given_state")
        );
    }

    #[test]
    fn rejects_duplicate_action_names() {
        let mut problem = valid_problem();
        problem.actions[1] = problem.actions[0].clone();
        assert!(matches!(
            calculate_evsi(&problem),
            Err(EvsiError::DuplicateIdentifier("action", _))
        ));
    }

    #[test]
    fn rejects_dimension_mismatch() {
        let mut problem = valid_problem();
        problem.utilities[0].pop();
        assert_eq!(
            calculate_evsi(&problem).expect_err("utility dimensions mismatch"),
            EvsiError::DimensionMismatch("utilities")
        );
    }

    #[test]
    fn zero_probability_observation_is_preserved_and_marked_undefined() {
        let mut problem = valid_problem();
        problem.prior = vec![1.0, 0.0];
        problem.outcomes = vec![
            ExperimentOutcome { outcome_id: "certain".to_owned(), likelihood_by_state: vec![1.0, 0.0] },
            ExperimentOutcome { outcome_id: "impossible".to_owned(), likelihood_by_state: vec![0.0, 1.0] },
        ];
        let report = calculate_evsi(&problem).expect("valid outcome model");
        assert_eq!(report.outcome_decisions[1].best_action, "UNDEFINED_ZERO_PROBABILITY_OUTCOME");
        assert_eq!(report.outcome_decisions[1].predictive_probability, 0.0);
    }
}
