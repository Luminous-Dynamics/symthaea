use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    env, fs,
    process::ExitCode,
};

const VERSION: &str = "0.1.0";
const EXPECTED_SCENARIOS: usize = 8;
const MAX_GAIN_UNITS: i64 = 1_000_000_000_000;
type Check<T> = Result<T, String>;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum MeasurementKind {
    Assay,
    ContaminantAssay,
    MoistureCheck,
    ComminutionEnergy,
    RecoveryTest,
    IdentityCheck,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Corpus {
    suite_id: String,
    suite_version: String,
    evidence_class: String,
    claim_ceiling: String,
    scenarios: Vec<Scenario>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Scenario {
    scenario_id: String,
    budget_units: u32,
    actions: Vec<MeasurementAction>,
    evaluation_only: EvaluationOnly,
    expected: ExpectedSelections,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct MeasurementAction {
    action_id: String,
    measurement_kind: MeasurementKind,
    available: bool,
    availability_reason: String,
    cost_units: u32,
    uncertainty_reduction_units: u64,
    estimated_gross_decision_gain_units: i64,
    fixed_priority: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct EvaluationOnly {
    realized_gross_decision_gain_units: Vec<RealizedGain>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RealizedGain {
    action_id: String,
    gross_gain_units: i64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ExpectedSelections {
    fixed_order_action_id: Option<String>,
    uncertainty_first_action_id: Option<String>,
    decision_aware_action_id: Option<String>,
    oracle_action_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ScenarioReceipt {
    scenario_id: String,
    fixed_order_action_id: Option<String>,
    uncertainty_first_action_id: Option<String>,
    decision_aware_action_id: Option<String>,
    oracle_action_id: Option<String>,
    fixed_order_realized_net_value_units: i64,
    uncertainty_first_realized_net_value_units: i64,
    decision_aware_realized_net_value_units: i64,
    oracle_realized_net_value_units: i64,
    fixed_order_regret_units: i64,
    uncertainty_first_regret_units: i64,
    decision_aware_regret_units: i64,
    expected_control_matched: bool,
    blocked_action_selections: usize,
    budget_violations: usize,
}

#[derive(Debug, Clone, Serialize)]
struct Receipt {
    result: String,
    suite_id: String,
    suite_version: String,
    validator_version: String,
    source_commit: String,
    corpus_sha256: String,
    scenario_count: usize,
    expected_controls_matched: usize,
    total_fixed_order_regret_units: i64,
    total_uncertainty_first_regret_units: i64,
    total_decision_aware_regret_units: i64,
    total_oracle_net_value_units: i64,
    blocked_action_selections: usize,
    budget_violations: usize,
    scenarios: Vec<ScenarioReceipt>,
}

fn eligible_actions(budget_units: u32, actions: &[MeasurementAction]) -> Vec<&MeasurementAction> {
    actions
        .iter()
        .filter(|action| action.available && action.cost_units <= budget_units)
        .collect()
}

fn estimated_net_value(action: &MeasurementAction) -> i64 {
    action.estimated_gross_decision_gain_units - i64::from(action.cost_units)
}

fn choose_fixed_order(
    budget_units: u32,
    actions: &[MeasurementAction],
) -> Option<String> {
    let mut candidates = eligible_actions(budget_units, actions);
    candidates.sort_by(|a, b| {
        a.fixed_priority
            .cmp(&b.fixed_priority)
            .then_with(|| a.action_id.cmp(&b.action_id))
    });
    candidates.first().map(|action| action.action_id.clone())
}

fn choose_uncertainty_first(
    budget_units: u32,
    actions: &[MeasurementAction],
) -> Option<String> {
    let mut candidates = eligible_actions(budget_units, actions);
    candidates.sort_by(|a, b| {
        b.uncertainty_reduction_units
            .cmp(&a.uncertainty_reduction_units)
            .then_with(|| a.cost_units.cmp(&b.cost_units))
            .then_with(|| a.action_id.cmp(&b.action_id))
    });
    candidates.first().map(|action| action.action_id.clone())
}

fn choose_decision_aware(
    budget_units: u32,
    actions: &[MeasurementAction],
) -> Option<String> {
    let mut candidates = eligible_actions(budget_units, actions);
    candidates.sort_by(|a, b| {
        estimated_net_value(b)
            .cmp(&estimated_net_value(a))
            .then_with(|| a.cost_units.cmp(&b.cost_units))
            .then_with(|| a.action_id.cmp(&b.action_id))
    });
    candidates
        .first()
        .filter(|action| estimated_net_value(action) > 0)
        .map(|action| action.action_id.clone())
}

fn realized_gross_gain(evaluation: &EvaluationOnly, action_id: &str) -> i64 {
    evaluation
        .realized_gross_decision_gain_units
        .iter()
        .find(|gain| gain.action_id == action_id)
        .expect("validated evaluation key must contain every action")
        .gross_gain_units
}

fn realized_net_value(
    actions: &[MeasurementAction],
    evaluation: &EvaluationOnly,
    action_id: Option<&str>,
) -> i64 {
    let Some(action_id) = action_id else {
        return 0;
    };
    let action = actions
        .iter()
        .find(|action| action.action_id == action_id)
        .expect("selected action must exist in validated corpus");
    realized_gross_gain(evaluation, action_id) - i64::from(action.cost_units)
}

fn choose_oracle(
    budget_units: u32,
    actions: &[MeasurementAction],
    evaluation: &EvaluationOnly,
) -> Option<String> {
    let mut candidates = eligible_actions(budget_units, actions);
    candidates.sort_by(|a, b| {
        let a_value = realized_gross_gain(evaluation, &a.action_id) - i64::from(a.cost_units);
        let b_value = realized_gross_gain(evaluation, &b.action_id) - i64::from(b.cost_units);
        b_value
            .cmp(&a_value)
            .then_with(|| a.cost_units.cmp(&b.cost_units))
            .then_with(|| a.action_id.cmp(&b.action_id))
    });
    candidates
        .first()
        .filter(|action| {
            realized_gross_gain(evaluation, &action.action_id) - i64::from(action.cost_units) > 0
        })
        .map(|action| action.action_id.clone())
}

fn validate_expected_action(
    scenario: &Scenario,
    action_id: &Option<String>,
    label: &str,
) -> Check<()> {
    if let Some(action_id) = action_id {
        let action = scenario
            .actions
            .iter()
            .find(|action| action.action_id == *action_id)
            .ok_or_else(|| format!("{}: expected {} references unknown action", scenario.scenario_id, label))?;
        if !action.available || action.cost_units > scenario.budget_units {
            return Err(format!(
                "{}: expected {} action is blocked or over budget",
                scenario.scenario_id, label
            ));
        }
    }
    Ok(())
}

fn validate_corpus(corpus: &Corpus) -> Check<()> {
    if corpus.suite_id != "ORE-VOI-001" || corpus.suite_version != VERSION {
        return Err("unexpected suite identity or version".to_owned());
    }
    if corpus.evidence_class != "synthetic_contract" || corpus.scenarios.len() != EXPECTED_SCENARIOS {
        return Err("unexpected evidence class or scenario count".to_owned());
    }
    if corpus.claim_ceiling.trim().is_empty() {
        return Err("claim ceiling must not be empty".to_owned());
    }

    let mut scenario_ids = BTreeSet::new();
    for scenario in &corpus.scenarios {
        if scenario.scenario_id.trim().is_empty() || !scenario_ids.insert(&scenario.scenario_id) {
            return Err("scenario IDs must be non-empty and unique".to_owned());
        }
        if scenario.actions.is_empty() || scenario.actions.len() > 100 {
            return Err(format!("{}: invalid action count", scenario.scenario_id));
        }

        let mut action_ids = BTreeSet::new();
        for action in &scenario.actions {
            if action.action_id.trim().is_empty() || !action_ids.insert(&action.action_id) {
                return Err(format!("{}: action IDs must be non-empty and unique", scenario.scenario_id));
            }
            if action.cost_units == 0 {
                return Err(format!("{}: test costs must be positive", action.action_id));
            }
            if action.estimated_gross_decision_gain_units < 0
                || action.estimated_gross_decision_gain_units > MAX_GAIN_UNITS
            {
                return Err(format!("{}: estimated gain is outside the supported range", action.action_id));
            }
            if action.available != (action.availability_reason == "available") {
                return Err(format!("{}: availability and reason disagree", action.action_id));
            }
            if !action.available && action.availability_reason.trim().is_empty() {
                return Err(format!("{}: blocked action requires a reason", action.action_id));
            }
        }

        let mut realized_ids = BTreeSet::new();
        for gain in &scenario.evaluation_only.realized_gross_decision_gain_units {
            if !action_ids.contains(&gain.action_id) || !realized_ids.insert(&gain.action_id) {
                return Err(format!("{}: evaluation keys must be known and unique", scenario.scenario_id));
            }
            if gain.gross_gain_units < 0 || gain.gross_gain_units > MAX_GAIN_UNITS {
                return Err(format!("{}: realized gain is outside the supported range", gain.action_id));
            }
        }
        if realized_ids != action_ids {
            return Err(format!("{}: evaluator must provide one gain per action", scenario.scenario_id));
        }

        validate_expected_action(scenario, &scenario.expected.fixed_order_action_id, "fixed")?;
        validate_expected_action(scenario, &scenario.expected.uncertainty_first_action_id, "uncertainty")?;
        validate_expected_action(scenario, &scenario.expected.decision_aware_action_id, "decision-aware")?;
        validate_expected_action(scenario, &scenario.expected.oracle_action_id, "oracle")?;
    }

    Ok(())
}

fn evaluate_scenario(scenario: &Scenario) -> ScenarioReceipt {
    let fixed = choose_fixed_order(scenario.budget_units, &scenario.actions);
    let uncertainty = choose_uncertainty_first(scenario.budget_units, &scenario.actions);
    let decision_aware = choose_decision_aware(scenario.budget_units, &scenario.actions);
    let oracle = choose_oracle(
        scenario.budget_units,
        &scenario.actions,
        &scenario.evaluation_only,
    );

    let fixed_value = realized_net_value(
        &scenario.actions,
        &scenario.evaluation_only,
        fixed.as_deref(),
    );
    let uncertainty_value = realized_net_value(
        &scenario.actions,
        &scenario.evaluation_only,
        uncertainty.as_deref(),
    );
    let decision_value = realized_net_value(
        &scenario.actions,
        &scenario.evaluation_only,
        decision_aware.as_deref(),
    );
    let oracle_value = realized_net_value(
        &scenario.actions,
        &scenario.evaluation_only,
        oracle.as_deref(),
    );

    let blocked_action_selections = [&fixed, &uncertainty, &decision_aware, &oracle]
        .iter()
        .filter_map(|selection| selection.as_ref())
        .filter(|id| {
            scenario
                .actions
                .iter()
                .find(|action| action.action_id == **id)
                .is_some_and(|action| !action.available)
        })
        .count();
    let budget_violations = [&fixed, &uncertainty, &decision_aware, &oracle]
        .iter()
        .filter_map(|selection| selection.as_ref())
        .filter(|id| {
            scenario
                .actions
                .iter()
                .find(|action| action.action_id == **id)
                .is_some_and(|action| action.cost_units > scenario.budget_units)
        })
        .count();

    let expected_control_matched = fixed == scenario.expected.fixed_order_action_id
        && uncertainty == scenario.expected.uncertainty_first_action_id
        && decision_aware == scenario.expected.decision_aware_action_id
        && oracle == scenario.expected.oracle_action_id;

    ScenarioReceipt {
        scenario_id: scenario.scenario_id.clone(),
        fixed_order_action_id: fixed,
        uncertainty_first_action_id: uncertainty,
        decision_aware_action_id: decision_aware,
        oracle_action_id: oracle,
        fixed_order_realized_net_value_units: fixed_value,
        uncertainty_first_realized_net_value_units: uncertainty_value,
        decision_aware_realized_net_value_units: decision_value,
        oracle_realized_net_value_units: oracle_value,
        fixed_order_regret_units: oracle_value - fixed_value,
        uncertainty_first_regret_units: oracle_value - uncertainty_value,
        decision_aware_regret_units: oracle_value - decision_value,
        expected_control_matched,
        blocked_action_selections,
        budget_violations,
    }
}

fn source_commit_from_args(args: &[String]) -> Check<String> {
    if args.len() != 4 || args[2] != "--source-commit" {
        return Err("usage: ore-voi-bench <corpus.json> --source-commit <40-hex-sha>".to_owned());
    }
    let source_commit = args[3].to_ascii_lowercase();
    if source_commit.len() != 40 || !source_commit.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("source commit must be a 40-character hexadecimal SHA".to_owned());
    }
    Ok(source_commit)
}

fn run() -> Check<ExitCode> {
    let args: Vec<String> = env::args().collect();
    let source_commit = source_commit_from_args(&args)?;
    let bytes = fs::read(&args[1]).map_err(|error| format!("cannot read corpus: {error}"))?;
    let corpus: Corpus =
        serde_json::from_slice(&bytes).map_err(|error| format!("invalid corpus JSON: {error}"))?;
    validate_corpus(&corpus)?;

    let scenarios: Vec<ScenarioReceipt> = corpus.scenarios.iter().map(evaluate_scenario).collect();
    let expected_controls_matched = scenarios
        .iter()
        .filter(|result| result.expected_control_matched)
        .count();
    let blocked_action_selections = scenarios.iter().map(|result| result.blocked_action_selections).sum();
    let budget_violations = scenarios.iter().map(|result| result.budget_violations).sum();
    let total_fixed_order_regret_units = scenarios.iter().map(|result| result.fixed_order_regret_units).sum();
    let total_uncertainty_first_regret_units =
        scenarios.iter().map(|result| result.uncertainty_first_regret_units).sum();
    let total_decision_aware_regret_units =
        scenarios.iter().map(|result| result.decision_aware_regret_units).sum();
    let total_oracle_net_value_units =
        scenarios.iter().map(|result| result.oracle_realized_net_value_units).sum();
    let all_controls_matched = expected_controls_matched == corpus.scenarios.len()
        && blocked_action_selections == 0
        && budget_violations == 0;

    let receipt = Receipt {
        result: if all_controls_matched { "validated" } else { "control_mismatch" }.to_owned(),
        suite_id: corpus.suite_id,
        suite_version: corpus.suite_version,
        validator_version: VERSION.to_owned(),
        source_commit,
        corpus_sha256: format!("{:x}", Sha256::digest(&bytes)),
        scenario_count: scenarios.len(),
        expected_controls_matched,
        total_fixed_order_regret_units,
        total_uncertainty_first_regret_units,
        total_decision_aware_regret_units,
        total_oracle_net_value_units,
        blocked_action_selections,
        budget_violations,
        scenarios,
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&receipt)
            .map_err(|error| format!("cannot serialize receipt: {error}"))?
    );
    Ok(if all_controls_matched { ExitCode::SUCCESS } else { ExitCode::from(2) })
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> Corpus {
        serde_json::from_str(include_str!("../../../docs/tech/ore-voi-benchmark-corpus-v0.1.json"))
            .expect("frozen synthetic corpus must deserialize")
    }

    #[test]
    fn frozen_corpus_matches_all_known_answers() {
        let fixture = corpus();
        validate_corpus(&fixture).expect("corpus metadata must be valid");
        let results: Vec<ScenarioReceipt> = fixture.scenarios.iter().map(evaluate_scenario).collect();
        assert_eq!(EXPECTED_SCENARIOS, results.len());
        assert!(results.iter().all(|result| result.expected_control_matched));
        assert!(results.iter().all(|result| result.blocked_action_selections == 0));
        assert!(results.iter().all(|result| result.budget_violations == 0));
    }

    #[test]
    fn uncertainty_reduction_is_not_decision_value() {
        let fixture = corpus();
        let scenario = fixture
            .scenarios
            .iter()
            .find(|scenario| scenario.scenario_id == "UNCERTAINTY_IS_NOT_DECISION_VALUE")
            .expect("known-answer scenario exists");
        assert_eq!(
            Some("HIGH_UNCERTAINTY".to_owned()),
            choose_uncertainty_first(scenario.budget_units, &scenario.actions)
        );
        assert_eq!(
            Some("DECISION_TARGETED".to_owned()),
            choose_decision_aware(scenario.budget_units, &scenario.actions)
        );
    }

    #[test]
    fn a_miscalibrated_estimator_is_not_hidden_by_the_oracle() {
        let fixture = corpus();
        let scenario = fixture
            .scenarios
            .iter()
            .find(|scenario| scenario.scenario_id == "ESTIMATOR_MISCALIBRATION_MUST_REMAIN_VISIBLE")
            .expect("miscalibration scenario exists");
        let result = evaluate_scenario(scenario);
        assert_eq!(Some("NOISY_HIGH_GAIN_TEST".to_owned()), result.decision_aware_action_id);
        assert_eq!(Some("ROBUST_RECOVERY_TEST".to_owned()), result.oracle_action_id);
        assert_eq!(14, result.decision_aware_regret_units);
    }

    #[test]
    fn unavailable_and_over_budget_actions_are_never_selectable() {
        let fixture = corpus();
        let scenario = fixture
            .scenarios
            .iter()
            .find(|scenario| scenario.scenario_id == "UNQUALIFIED_SAMPLE_MUST_BE_BLOCKED")
            .expect("blocked evidence scenario exists");
        let choices = [
            choose_fixed_order(scenario.budget_units, &scenario.actions),
            choose_uncertainty_first(scenario.budget_units, &scenario.actions),
            choose_decision_aware(scenario.budget_units, &scenario.actions),
            choose_oracle(scenario.budget_units, &scenario.actions, &scenario.evaluation_only),
        ];
        assert!(choices.iter().all(|choice| choice.as_deref() != Some("UNQUALIFIED_IDENTITY_CHECK")));
    }

    #[test]
    fn no_positive_net_value_returns_no_test() {
        let fixture = corpus();
        let scenario = fixture
            .scenarios
            .iter()
            .find(|scenario| scenario.scenario_id == "NO_POSITIVE_NET_DECISION_VALUE")
            .expect("no-value scenario exists");
        assert_eq!(None, choose_decision_aware(scenario.budget_units, &scenario.actions));
        assert_eq!(None, choose_oracle(scenario.budget_units, &scenario.actions, &scenario.evaluation_only));
    }

    #[test]
    fn unknown_json_fields_fail_closed() {
        let raw = include_str!("../../../docs/tech/ore-voi-benchmark-corpus-v0.1.json")
            .replacen("\"suite_id\":", "\"unexpected_authority\":true,\"suite_id\":", 1);
        assert!(serde_json::from_str::<Corpus>(&raw).is_err());
    }
}
