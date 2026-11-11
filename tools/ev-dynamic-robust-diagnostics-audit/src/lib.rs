use jsonschema::{Validator, draft202012};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const CORPUS_VERSION: &str = "ev-dynamic-diagnostic-corpus/0.1";
const PROBABILITY_TOLERANCE: f64 = 1e-9;
const MAX_MODELS: usize = 4;
const MAX_TESTS: usize = 2;
const MAX_OUTCOMES: usize = 3;
const MAX_ACTIONS: usize = 2;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Corpus {
    pub corpus_version: String,
    pub problems: Vec<Problem>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Problem {
    pub problem_id: String,
    pub loss_unit: String,
    pub states: Vec<String>,
    pub actions: Vec<TerminalAction>,
    pub tests: Vec<TestSpec>,
    pub worlds: Vec<WorldModel>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TerminalAction {
    pub action_id: String,
    pub loss_by_state: Vec<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TestSpec {
    pub test_id: String,
    pub outcomes: Vec<String>,
    pub duration_seconds: f64,
    pub test_cost: f64,
    pub downtime_cost: f64,
    pub safety_approved: bool,
    pub calibration_valid: bool,
    pub supported_domain: bool,
    pub equipment_available: bool,
}

impl TestSpec {
    fn eligible(&self) -> bool {
        self.safety_approved
            && self.calibration_valid
            && self.supported_domain
            && self.equipment_available
    }

    fn cost(&self) -> f64 {
        self.test_cost + self.downtime_cost
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorldModel {
    pub model_id: String,
    pub prior: Vec<f64>,
    pub test_models: Vec<TestModel>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TestModel {
    pub test_id: String,
    /// Transition during test duration: row = pre-test state, column = state
    /// after elapsed-time degradation has occurred.
    pub time_transition: Vec<Vec<f64>>,
    /// Test-induced state perturbation: row = state after elapsed time,
    /// column = final state after the test has perturbed the system.
    pub perturbation_transition: Vec<Vec<f64>>,
    /// Observation likelihoods by final post-test state. Each row aligns with
    /// the corresponding TestSpec.outcomes order.
    pub observation_by_post_state: Vec<Vec<f64>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct BranchPlan {
    pub first_outcome: String,
    pub second_test: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TerminalDecision {
    pub history: String,
    pub action_id: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct PolicyPlan {
    pub first_test: Option<String>,
    pub branches: Vec<BranchPlan>,
    pub terminal_decisions: Vec<TerminalDecision>,
    pub policy_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct WorldEvaluation {
    pub model_id: String,
    pub expected_total_loss: f64,
    pub expected_tests: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProblemReport {
    pub problem_id: String,
    pub dynamic_policy: PolicyPlan,
    pub dynamic_robust_loss: f64,
    pub dynamic_worlds: Vec<WorldEvaluation>,
    pub static_policy: PolicyPlan,
    pub static_model_robust_loss: f64,
    pub static_policy_actual_dynamic_loss: f64,
    pub robust_loss_improvement: f64,
    pub dynamic_candidate_count: usize,
    pub static_candidate_count: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct AuditReport {
    pub corpus_version: String,
    pub input_sha256: String,
    pub canonical_corpus_sha256: String,
    pub problem_count: usize,
    pub reports: Vec<ProblemReport>,
}

#[derive(Clone)]
struct CandidatePolicy {
    first_test: Option<usize>,
    second_by_first_outcome: Vec<Option<usize>>,
    terminal_action_by_history: BTreeMap<String, usize>,
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn compile_schema(bytes: &[u8]) -> Result<Validator, String> {
    let schema: Value =
        serde_json::from_slice(bytes).map_err(|error| format!("schema JSON: {error}"))?;
    if schema.get("$schema").and_then(Value::as_str)
        != Some("https://json-schema.org/draft/2020-12/schema")
    {
        return Err("schema must declare JSON Schema Draft 2020-12".to_owned());
    }
    if !draft202012::meta::is_valid(&schema) {
        return Err("schema fails Draft 2020-12 meta-schema validation".to_owned());
    }
    draft202012::options()
        .should_validate_formats(true)
        .build(&schema)
        .map_err(|error| format!("schema compilation: {error}"))
}

fn validate_schema(validator: &Validator, value: &Value) -> Result<(), String> {
    let errors: Vec<String> = validator.iter_errors(value).map(|error| error.to_string()).collect();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!("dynamic diagnostic corpus schema errors: {}", errors.join("; ")))
    }
}

fn is_finite_nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn probability_row(values: &[f64], label: &str) -> Result<(), String> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite() || *value < 0.0 || *value > 1.0) {
        return Err(format!("{label} contains an invalid probability"));
    }
    let sum: f64 = values.iter().sum();
    if (sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(format!("{label} sums to {sum}, expected 1"));
    }
    Ok(())
}

fn matrix_2x2(matrix: &[Vec<f64>], label: &str) -> Result<(), String> {
    if matrix.len() != 2 || matrix.iter().any(|row| row.len() != 2) {
        return Err(format!("{label} must be a 2x2 transition matrix"));
    }
    for (index, row) in matrix.iter().enumerate() {
        probability_row(row, &format!("{label}[{index}]"))?;
    }
    Ok(())
}

fn transition_dynamic(model: &TestModel, initial_state: usize) -> [f64; 2] {
    let mut output = [0.0; 2];
    for intermediate in 0..2 {
        for final_state in 0..2 {
            output[final_state] += model.time_transition[initial_state][intermediate]
                * model.perturbation_transition[intermediate][final_state];
        }
    }
    output
}

fn transition_static(_model: &TestModel, initial_state: usize) -> [f64; 2] {
    if initial_state == 0 {
        [1.0, 0.0]
    } else {
        [0.0, 1.0]
    }
}

fn validate_problem(problem: &Problem) -> Result<(), String> {
    if problem.problem_id.trim().is_empty() || problem.loss_unit.trim().is_empty() {
        return Err("problem identity and loss unit are required".to_owned());
    }
    if problem.states.len() != 2 || problem.states.iter().any(|state| state.trim().is_empty()) {
        return Err("v0.1 requires exactly two non-empty latent states".to_owned());
    }
    if problem.actions.is_empty() || problem.actions.len() > MAX_ACTIONS {
        return Err(format!("v0.1 supports one or two terminal actions only"));
    }
    if problem.tests.len() > MAX_TESTS {
        return Err(format!("v0.1 supports at most {MAX_TESTS} tests"));
    }
    if problem.worlds.is_empty() || problem.worlds.len() > MAX_MODELS {
        return Err(format!("v0.1 requires one to {MAX_MODELS} ambiguity models"));
    }

    let mut action_ids = BTreeSet::new();
    for action in &problem.actions {
        if action.action_id.trim().is_empty()
            || !action_ids.insert(action.action_id.clone())
            || action.loss_by_state.len() != 2
            || action.loss_by_state.iter().any(|value| !is_finite_nonnegative(*value))
        {
            return Err(format!("terminal action {} is malformed", action.action_id));
        }
    }

    let mut test_ids = BTreeSet::new();
    for test in &problem.tests {
        if test.test_id.trim().is_empty() || !test_ids.insert(test.test_id.clone()) {
            return Err(format!("test identity is empty or duplicated: {}", test.test_id));
        }
        if test.outcomes.len() < 2 || test.outcomes.len() > MAX_OUTCOMES {
            return Err(format!("test {} must declare two or three outcomes", test.test_id));
        }
        let mut outcomes = BTreeSet::new();
        if test.outcomes.iter().any(|value| value.trim().is_empty() || !outcomes.insert(value.clone())) {
            return Err(format!("test {} has empty or duplicate outcome labels", test.test_id));
        }
        if !is_finite_nonnegative(test.duration_seconds)
            || !is_finite_nonnegative(test.test_cost)
            || !is_finite_nonnegative(test.downtime_cost)
        {
            return Err(format!("test {} has invalid duration or cost", test.test_id));
        }
    }

    let mut model_ids = BTreeSet::new();
    for world in &problem.worlds {
        if world.model_id.trim().is_empty() || !model_ids.insert(world.model_id.clone()) {
            return Err(format!("ambiguity model identity is empty or duplicated: {}", world.model_id));
        }
        if world.prior.len() != 2 {
            return Err(format!("model {} prior must contain exactly two states", world.model_id));
        }
        probability_row(&world.prior, &format!("model {} prior", world.model_id))?;
        if world.test_models.len() != problem.tests.len() {
            return Err(format!("model {} does not define every test exactly once", world.model_id));
        }
        let mut model_test_ids = BTreeSet::new();
        for test_model in &world.test_models {
            if !model_test_ids.insert(test_model.test_id.clone()) {
                return Err(format!("model {} duplicates test model {}", world.model_id, test_model.test_id));
            }
            let spec = problem.tests.iter().find(|test| test.test_id == test_model.test_id)
                .ok_or_else(|| format!("model {} references unknown test {}", world.model_id, test_model.test_id))?;
            matrix_2x2(&test_model.time_transition, "time transition")?;
            matrix_2x2(&test_model.perturbation_transition, "test perturbation transition")?;
            if test_model.observation_by_post_state.len() != 2 {
                return Err(format!("test {} must define observation likelihoods for both post-states", spec.test_id));
            }
            for (state, row) in test_model.observation_by_post_state.iter().enumerate() {
                if row.len() != spec.outcomes.len() {
                    return Err(format!("test {} state {state} likelihoods do not match its outcome labels", spec.test_id));
                }
                probability_row(row, &format!("test {} state {state} likelihoods", spec.test_id))?;
            }
        }
    }
    Ok(())
}

fn test_model<'a>(world: &'a WorldModel, test_id: &str) -> Result<&'a TestModel, String> {
    world.test_models.iter().find(|item| item.test_id == test_id)
        .ok_or_else(|| format!("model {} lacks test {}", world.model_id, test_id))
}

fn test_spec<'a>(problem: &'a Problem, test_id: &str) -> Result<&'a TestSpec, String> {
    problem.tests.iter().find(|item| item.test_id == test_id)
        .ok_or_else(|| format!("problem {} lacks test {}", problem.problem_id, test_id))
}

fn terminal_action<'a>(problem: &'a Problem, action_index: usize) -> Result<&'a TerminalAction, String> {
    problem.actions.get(action_index)
        .ok_or_else(|| "policy references unknown terminal action".to_owned())
}

fn history_first(test_id: &str, outcome: &str) -> String {
    format!("{test_id}={outcome}")
}

fn history_second(first_test: &str, first_outcome: &str, second_test: &str, second_outcome: &str) -> String {
    format!("{}|{}={}", history_first(first_test, first_outcome), second_test, second_outcome)
}

fn build_leaves(
    problem: &Problem,
    first_test: Option<usize>,
    second_by_first_outcome: &[Option<usize>],
) -> Result<Vec<String>, String> {
    let Some(first_index) = first_test else {
        return Ok(vec!["NO_TEST".to_owned()]);
    };
    let first = &problem.tests[first_index];
    if second_by_first_outcome.len() != first.outcomes.len() {
        return Err("policy branch count does not match the first test outcomes".to_owned());
    }
    let mut leaves = Vec::new();
    for (outcome_index, second_index) in second_by_first_outcome.iter().enumerate() {
        let first_outcome = &first.outcomes[outcome_index];
        match second_index {
            None => leaves.push(history_first(&first.test_id, first_outcome)),
            Some(second_index) => {
                let second = problem.tests.get(*second_index)
                    .ok_or_else(|| "policy references an unknown second test".to_owned())?;
                for second_outcome in &second.outcomes {
                    leaves.push(history_second(
                        &first.test_id,
                        first_outcome,
                        &second.test_id,
                        second_outcome,
                    ));
                }
            }
        }
    }
    leaves.sort();
    Ok(leaves)
}

fn to_policy_plan(policy: &CandidatePolicy, problem: &Problem) -> Result<PolicyPlan, String> {
    let first_test = policy.first_test.map(|index| problem.tests[index].test_id.clone());
    let mut branches = Vec::new();
    if let Some(first_index) = policy.first_test {
        let first = &problem.tests[first_index];
        for (outcome_index, second_index) in policy.second_by_first_outcome.iter().enumerate() {
            branches.push(BranchPlan {
                first_outcome: first.outcomes[outcome_index].clone(),
                second_test: second_index.map(|index| problem.tests[index].test_id.clone()),
            });
        }
    }
    let terminal_decisions = policy.terminal_action_by_history.iter().map(|(history, action_index)| {
        Ok(TerminalDecision {
            history: history.clone(),
            action_id: terminal_action(problem, *action_index)?.action_id.clone(),
        })
    }).collect::<Result<Vec<_>, String>>()?;
    let mut plan = PolicyPlan {
        first_test,
        branches,
        terminal_decisions,
        policy_sha256: String::new(),
    };
    let bytes = serde_json::to_vec(&plan).map_err(|error| error.to_string())?;
    plan.policy_sha256 = sha256(&bytes);
    Ok(plan)
}

fn evaluate_policy_world(
    problem: &Problem,
    world: &WorldModel,
    policy: &CandidatePolicy,
    dynamic: bool,
) -> Result<(f64, f64), String> {
    let mut expected_loss = 0.0;
    let mut expected_tests = 0.0;
    for initial_state in 0..2 {
        let prior = world.prior[initial_state];
        if prior == 0.0 {
            continue;
        }
        let Some(first_index) = policy.first_test else {
            let action_index = *policy.terminal_action_by_history.get("NO_TEST")
                .ok_or_else(|| "no-test policy lacks a terminal decision".to_owned())?;
            expected_loss += prior * terminal_action(problem, action_index)?.loss_by_state[initial_state];
            continue;
        };

        let first_spec = &problem.tests[first_index];
        let first_model = test_model(world, &first_spec.test_id)?;
        let first_transition = if dynamic {
            transition_dynamic(first_model, initial_state)
        } else {
            transition_static(first_model, initial_state)
        };
        for first_state in 0..2 {
            let after_first_transition = prior * first_transition[first_state];
            if after_first_transition == 0.0 {
                continue;
            }
            for first_outcome_index in 0..first_spec.outcomes.len() {
                let outcome_probability =
                    first_model.observation_by_post_state[first_state][first_outcome_index];
                let path_probability = after_first_transition * outcome_probability;
                if path_probability == 0.0 {
                    continue;
                }
                let first_outcome = &first_spec.outcomes[first_outcome_index];
                let second_index = policy.second_by_first_outcome.get(first_outcome_index).copied().flatten();
                let Some(second_index) = second_index else {
                    let history = history_first(&first_spec.test_id, first_outcome);
                    let action_index = *policy.terminal_action_by_history.get(&history)
                        .ok_or_else(|| format!("policy lacks terminal decision for {history}"))?;
                    expected_loss += path_probability
                        * (first_spec.cost() + terminal_action(problem, action_index)?.loss_by_state[first_state]);
                    expected_tests += path_probability;
                    continue;
                };

                let second_spec = &problem.tests[second_index];
                let second_model = test_model(world, &second_spec.test_id)?;
                let second_transition = if dynamic {
                    transition_dynamic(second_model, first_state)
                } else {
                    transition_static(second_model, first_state)
                };
                for final_state in 0..2 {
                    let after_second_transition = path_probability * second_transition[final_state];
                    if after_second_transition == 0.0 {
                        continue;
                    }
                    for second_outcome_index in 0..second_spec.outcomes.len() {
                        let second_probability =
                            second_model.observation_by_post_state[final_state][second_outcome_index];
                        let full_probability = after_second_transition * second_probability;
                        if full_probability == 0.0 {
                            continue;
                        }
                        let second_outcome = &second_spec.outcomes[second_outcome_index];
                        let history = history_second(
                            &first_spec.test_id,
                            first_outcome,
                            &second_spec.test_id,
                            second_outcome,
                        );
                        let action_index = *policy.terminal_action_by_history.get(&history)
                            .ok_or_else(|| format!("policy lacks terminal decision for {history}"))?;
                        let action_loss = terminal_action(problem, action_index)?.loss_by_state[final_state];
                        expected_loss += full_probability
                            * (first_spec.cost() + second_spec.cost() + action_loss);
                        expected_tests += full_probability * 2.0;
                    }
                }
            }
        }
    }
    if !expected_loss.is_finite() || !expected_tests.is_finite() {
        return Err("non-finite dynamic policy expectation".to_owned());
    }
    Ok((expected_loss, expected_tests))
}

fn enumerate_decision_maps(
    leaves: &[String],
    action_count: usize,
    mut visitor: impl FnMut(BTreeMap<String, usize>) -> Result<(), String>,
) -> Result<(), String> {
    let combinations = action_count.pow(leaves.len() as u32);
    for mut code in 0..combinations {
        let mut map = BTreeMap::new();
        for leaf in leaves {
            let action = code % action_count;
            code /= action_count;
            map.insert(leaf.clone(), action);
        }
        visitor(map)?;
    }
    Ok(())
}

fn plan_key(policy: &CandidatePolicy, problem: &Problem) -> Result<String, String> {
    let plan = to_policy_plan(policy, problem)?;
    serde_json::to_string(&plan).map_err(|error| error.to_string())
}

fn optimize(problem: &Problem, dynamic: bool) -> Result<(CandidatePolicy, f64, usize), String> {
    let eligible: Vec<usize> = problem.tests.iter().enumerate()
        .filter_map(|(index, test)| if test.eligible() { Some(index) } else { None })
        .collect();
    let mut best: Option<(CandidatePolicy, f64, String)> = None;
    let mut candidate_count = 0usize;

    // STOP/NO_TEST is always present.
    let no_test_leaves = vec!["NO_TEST".to_owned()];
    enumerate_decision_maps(&no_test_leaves, problem.actions.len(), |decision_map| {
        let policy = CandidatePolicy {
            first_test: None,
            second_by_first_outcome: Vec::new(),
            terminal_action_by_history: decision_map,
        };
        consider_candidate(problem, dynamic, policy, &mut best, &mut candidate_count)
    })?;

    for first_index in &eligible {
        let first = &problem.tests[*first_index];
        let second_eligible: Vec<usize> = eligible.iter().copied()
            .filter(|index| index != first_index)
            .collect();
        // v0.1 is deliberately bounded to at most one follow-up test.
        if second_eligible.len() > 1 {
            return Err("v0.1 planner supports at most two total eligible tests".to_owned());
        }
        let second_index = second_eligible.first().copied();
        let branch_combinations = 1usize << first.outcomes.len();
        for branch_code in 0..branch_combinations {
            let second_by_first_outcome: Vec<Option<usize>> = (0..first.outcomes.len())
                .map(|index| {
                    if (branch_code & (1usize << index)) != 0 { second_index } else { None }
                })
                .collect();
            let leaves = build_leaves(problem, Some(*first_index), &second_by_first_outcome)?;
            enumerate_decision_maps(&leaves, problem.actions.len(), |decision_map| {
                let policy = CandidatePolicy {
                    first_test: Some(*first_index),
                    second_by_first_outcome: second_by_first_outcome.clone(),
                    terminal_action_by_history: decision_map,
                };
                consider_candidate(problem, dynamic, policy, &mut best, &mut candidate_count)
            })?;
        }
    }

    let (policy, loss, _) = best.ok_or_else(|| "no admissible policy was enumerated".to_owned())?;
    Ok((policy, loss, candidate_count))
}

fn consider_candidate(
    problem: &Problem,
    dynamic: bool,
    policy: CandidatePolicy,
    best: &mut Option<(CandidatePolicy, f64, String)>,
    candidate_count: &mut usize,
) -> Result<(), String> {
    *candidate_count += 1;
    let mut worst_loss = 0.0f64;
    for world in &problem.worlds {
        let (loss, _) = evaluate_policy_world(problem, world, &policy, dynamic)?;
        worst_loss = worst_loss.max(loss);
    }
    let key = plan_key(&policy, problem)?;
    let replace = match best {
        None => true,
        Some((_, best_loss, best_key)) => {
            worst_loss < *best_loss - PROBABILITY_TOLERANCE
                || ((worst_loss - *best_loss).abs() <= PROBABILITY_TOLERANCE && key < *best_key)
        }
    };
    if replace {
        *best = Some((policy, worst_loss, key));
    }
    Ok(())
}

fn canonicalize(corpus: &mut Corpus) {
    corpus.problems.sort_by(|left, right| left.problem_id.cmp(&right.problem_id));
    for problem in &mut corpus.problems {
        problem.tests.sort_by(|left, right| left.test_id.cmp(&right.test_id));
        problem.actions.sort_by(|left, right| left.action_id.cmp(&right.action_id));
        problem.worlds.sort_by(|left, right| left.model_id.cmp(&right.model_id));
        for world in &mut problem.worlds {
            world.test_models.sort_by(|left, right| left.test_id.cmp(&right.test_id));
        }
    }
}

fn solve_problem(problem: &Problem) -> Result<ProblemReport, String> {
    validate_problem(problem)?;
    let (dynamic_policy, dynamic_robust_loss, dynamic_candidate_count) =
        optimize(problem, true)?;
    let (static_policy, static_model_robust_loss, static_candidate_count) =
        optimize(problem, false)?;
    let mut dynamic_worlds = Vec::new();
    let mut static_policy_actual_dynamic_loss = 0.0f64;
    for world in &problem.worlds {
        let (dynamic_loss, dynamic_tests) =
            evaluate_policy_world(problem, world, &dynamic_policy, true)?;
        dynamic_worlds.push(WorldEvaluation {
            model_id: world.model_id.clone(),
            expected_total_loss: dynamic_loss,
            expected_tests: dynamic_tests,
        });
        let (static_actual_loss, _) =
            evaluate_policy_world(problem, world, &static_policy, true)?;
        static_policy_actual_dynamic_loss =
            static_policy_actual_dynamic_loss.max(static_actual_loss);
    }
    let improvement = static_policy_actual_dynamic_loss - dynamic_robust_loss;
    Ok(ProblemReport {
        problem_id: problem.problem_id.clone(),
        dynamic_policy: to_policy_plan(&dynamic_policy, problem)?,
        dynamic_robust_loss,
        dynamic_worlds,
        static_policy: to_policy_plan(&static_policy, problem)?,
        static_model_robust_loss,
        static_policy_actual_dynamic_loss,
        robust_loss_improvement: improvement,
        dynamic_candidate_count,
        static_candidate_count,
    })
}

pub fn audit_corpus(schema_bytes: &[u8], corpus_bytes: &[u8]) -> Result<AuditReport, String> {
    let validator = compile_schema(schema_bytes)?;
    let json: Value = serde_json::from_slice(corpus_bytes)
        .map_err(|error| format!("corpus JSON: {error}"))?;
    validate_schema(&validator, &json)?;
    let mut corpus: Corpus =
        serde_json::from_value(json).map_err(|error| format!("corpus decode: {error}"))?;
    if corpus.corpus_version != CORPUS_VERSION || corpus.problems.is_empty() {
        return Err("unsupported or empty dynamic diagnostic corpus".to_owned());
    }
    canonicalize(&mut corpus);
    let mut ids = BTreeSet::new();
    for problem in &corpus.problems {
        if !ids.insert(problem.problem_id.clone()) {
            return Err(format!("duplicate problem identity {}", problem.problem_id));
        }
    }
    let canonical = serde_json::to_vec(&corpus)
        .map_err(|error| format!("canonical corpus serialization: {error}"))?;
    let reports = corpus.problems.iter().map(solve_problem).collect::<Result<Vec<_>, _>>()?;
    Ok(AuditReport {
        corpus_version: corpus.corpus_version,
        input_sha256: sha256(corpus_bytes),
        canonical_corpus_sha256: sha256(&canonical),
        problem_count: reports.len(),
        reports,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const SCHEMA: &[u8] =
        include_bytes!("../../../schemas/transport/ev_dynamic_robust_diagnostic_v0_1.schema.json");
    const CORPUS: &[u8] =
        include_bytes!("../../../fixtures/ev_battery/dynamic_robust_diagnostic_v0_1.json");

    fn report() -> AuditReport {
        audit_corpus(SCHEMA, CORPUS).expect("frozen dynamic robust diagnostic corpus")
    }

    #[test]
    fn frozen_corpus_replays_all_expected_dynamic_known_truth_cases() {
        let report = report();
        assert_eq!(report.problem_count, 4);
        assert_eq!(report.canonical_corpus_sha256.len(), 64);
        assert!(report.reports.iter().all(|row| {
            row.dynamic_robust_loss.is_finite()
                && row.static_policy_actual_dynamic_loss.is_finite()
                && row.dynamic_policy.policy_sha256.len() == 64
        }));
    }

    #[test]
    fn state_drift_during_a_long_test_can_reverse_the_static_choice() {
        let report = report();
        let row = report.reports.iter()
            .find(|row| row.problem_id == "long-test-crosses-service-threshold")
            .expect("drift problem");
        assert_ne!(row.dynamic_policy.first_test, row.static_policy.first_test);
        assert!(row.robust_loss_improvement > 0.0);
    }

    #[test]
    fn test_induced_state_perturbation_is_separate_from_elapsed_time() {
        let report = report();
        let row = report.reports.iter()
            .find(|row| row.problem_id == "pulse-perturbs-state")
            .expect("perturbation problem");
        assert!(row.dynamic_candidate_count > 0);
        assert!(row.dynamic_robust_loss.is_finite());
    }

    #[test]
    fn model_ambiguity_uses_one_policy_across_all_worlds() {
        let report = report();
        let row = report.reports.iter()
            .find(|row| row.problem_id == "likelihood-ambiguity")
            .expect("ambiguity problem");
        assert_eq!(row.dynamic_worlds.len(), 2);
        assert!(row.dynamic_robust_loss >= row.dynamic_worlds[0].expected_total_loss - PROBABILITY_TOLERANCE);
        assert!(row.dynamic_robust_loss >= row.dynamic_worlds[1].expected_total_loss - PROBABILITY_TOLERANCE);
    }

    #[test]
    fn test_failure_is_an_observation_outcome_not_silently_dropped() {
        let report = report();
        let row = report.reports.iter()
            .find(|row| row.problem_id == "timeout-is-observed")
            .expect("timeout problem");
        let has_timeout_branch = row.dynamic_policy.branches.iter()
            .any(|branch| branch.first_outcome == "timeout");
        assert!(has_timeout_branch || row.dynamic_policy.first_test.is_none());
    }

    #[test]
    fn unapproved_or_uncalibrated_tests_are_never_selected() {
        let report = report();
        for row in &report.reports {
            let all_selected: Vec<&str> = row.dynamic_policy.first_test.iter().map(String::as_str)
                .chain(row.dynamic_policy.branches.iter().filter_map(|branch| branch.second_test.as_deref()))
                .collect();
            assert!(!all_selected.iter().any(|test| test.starts_with("unsafe-") || test.starts_with("uncalibrated-")));
        }
    }

    #[test]
    fn model_and_test_order_permutations_preserve_canonical_semantics() {
        let original = report();
        let mut corpus: Corpus = serde_json::from_slice(CORPUS).expect("corpus");
        corpus.problems.reverse();
        for problem in &mut corpus.problems {
            problem.tests.reverse();
            problem.actions.reverse();
            problem.worlds.reverse();
            for world in &mut problem.worlds {
                world.test_models.reverse();
            }
        }
        let bytes = serde_json::to_vec(&corpus).expect("permuted corpus");
        let replay = audit_corpus(SCHEMA, &bytes).expect("permuted audit");
        assert_eq!(original.canonical_corpus_sha256, replay.canonical_corpus_sha256);
        assert_eq!(
            serde_json::to_value(original.reports).unwrap(),
            serde_json::to_value(replay.reports).unwrap()
        );
    }

    #[test]
    fn malformed_transition_or_observation_probabilities_are_rejected() {
        let mut corpus: Corpus = serde_json::from_slice(CORPUS).expect("corpus");
        corpus.problems[0].worlds[0].test_models[0].time_transition[0] = vec![0.2, 0.2];
        let bytes = serde_json::to_vec(&corpus).expect("invalid corpus");
        assert!(audit_corpus(SCHEMA, &bytes).is_err());
    }

    #[test]
    fn unknown_joint_likelihood_fields_are_rejected_not_silently_ignored() {
        let mut corpus: Value = serde_json::from_slice(CORPUS).expect("corpus JSON");
        corpus["problems"][0]["worlds"][0]["test_models"][0]["joint_likelihood_ref"] =
            Value::String("unqualified-shared-sensor-model".into());
        let bytes = serde_json::to_vec(&corpus).expect("tagged corpus");
        assert!(audit_corpus(SCHEMA, &bytes).is_err());
    }
}
