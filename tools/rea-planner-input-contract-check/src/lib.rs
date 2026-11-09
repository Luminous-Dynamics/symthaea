//! Schema and semantic checks for the REA planner input contract v0.1.
//!
//! The checker validates planner inputs without invoking the planner. It keeps
//! synthetic mathematical fixtures separate from observed-Evidence integration.

use jsonschema::draft202012;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

const SCHEMA: &str =
    include_str!("../../../docs/research/rea-planner-input-contract-v0-1.schema.json");
const EXAMPLE: &str =
    include_str!("../../../docs/research/rea-planner-input-contract-v0-1-example-math-only.json");
const SUM_TOLERANCE: f64 = 1.0e-9;

const POLICY_AGENT: &str = include_str!(
    "../../../docs/research/rea-planner-policy-disagreement-v0-1-agent-input.json"
);
const POLICY_GRADER_SCHEMA: &str = include_str!(
    "../../../docs/research/rea-planner-policy-disagreement-v0-1-grader.schema.json"
);
const POLICY_GRADER: &str =
    include_str!("../../../docs/research/rea-planner-policy-disagreement-v0-1-grader.json");


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractError(pub String);

impl std::fmt::Display for ContractError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ContractError {}

fn object<'a>(
    value: &'a Value,
    key: &str,
) -> Result<&'a serde_json::Map<String, Value>, ContractError> {
    value
        .get(key)
        .and_then(Value::as_object)
        .ok_or_else(|| ContractError(format!("{key} must be an object")))
}

fn array<'a>(value: &'a Value, key: &str) -> Result<&'a Vec<Value>, ContractError> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| ContractError(format!("{key} must be an array")))
}

fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, ContractError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| ContractError(format!("{key} must be a string")))
}

fn number(value: &Value, key: &str) -> Result<f64, ContractError> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite())
        .ok_or_else(|| ContractError(format!("{key} must be a finite number")))
}

fn string_set(values: &[Value], context: &str) -> Result<BTreeSet<String>, ContractError> {
    let mut result = BTreeSet::new();
    for item in values {
        let value = item
            .as_str()
            .ok_or_else(|| ContractError(format!("{context} must contain only strings")))?;
        if !result.insert(value.to_owned()) {
            return Err(ContractError(format!(
                "{context} contains duplicate {value}"
            )));
        }
    }
    Ok(result)
}

fn string_vec(values: &[Value], context: &str) -> Result<Vec<String>, ContractError> {
    let mut result = Vec::with_capacity(values.len());
    let mut seen = BTreeSet::new();
    for item in values {
        let value = item
            .as_str()
            .ok_or_else(|| ContractError(format!("{context} must contain only strings")))?;
        if !seen.insert(value.to_owned()) {
            return Err(ContractError(format!(
                "{context} contains duplicate {value}"
            )));
        }
        result.push(value.to_owned());
    }
    Ok(result)
}

fn normalized_probability_rows(
    rows: &[Value],
    expected_outcomes: &[String],
    hypothesis_ids: &BTreeSet<String>,
    context: &str,
) -> Result<(), ContractError> {
    let mut row_ids = BTreeSet::new();
    for row in rows {
        let hypothesis_id = string(row, "hypothesis_id")?;
        if !hypothesis_ids.contains(hypothesis_id) {
            return Err(ContractError(format!(
                "{context} references unknown hypothesis {hypothesis_id}"
            )));
        }
        if !row_ids.insert(hypothesis_id.to_owned()) {
            return Err(ContractError(format!(
                "{context} contains duplicate row for {hypothesis_id}"
            )));
        }
        let declared_order = string_vec(
            array(row, "outcome_order")?,
            &format!("{context}/{hypothesis_id} outcome_order"),
        )?;
        if declared_order != expected_outcomes {
            return Err(ContractError(format!(
                "{context}/{hypothesis_id} outcome_order differs from the experiment's declared outcome order"
            )));
        }
        let probabilities = array(row, "outcome_probabilities")?;
        if probabilities.len() != expected_outcomes.len() {
            return Err(ContractError(format!(
                "{context}/{hypothesis_id} has {} probabilities for {} outcomes",
                probabilities.len(),
                expected_outcomes.len()
            )));
        }
        let mut total = 0.0;
        for probability in probabilities {
            let probability = probability
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| {
                    ContractError(format!(
                        "{context}/{hypothesis_id} contains non-finite probability"
                    ))
                })?;
            if !(0.0..=1.0).contains(&probability) {
                return Err(ContractError(format!(
                    "{context}/{hypothesis_id} probability outside [0,1]"
                )));
            }
            total += probability;
        }
        if (total - 1.0).abs() > SUM_TOLERANCE {
            return Err(ContractError(format!(
                "{context}/{hypothesis_id} probability mass is {total}, not 1"
            )));
        }
    }
    if &row_ids != hypothesis_ids {
        return Err(ContractError(format!(
            "{context} hypothesis rows do not exactly match the declared hypothesis set"
        )));
    }
    Ok(())
}

fn has_independent_root(
    id: &str,
    evidence: &HashMap<String, &Value>,
    visiting: &mut HashSet<String>,
    memo: &mut HashMap<String, bool>,
) -> Result<bool, ContractError> {
    if let Some(result) = memo.get(id) {
        return Ok(*result);
    }
    if !visiting.insert(id.to_owned()) {
        return Err(ContractError(format!(
            "cycle in Evidence derivation graph at {id}"
        )));
    }
    let record = evidence
        .get(id)
        .ok_or_else(|| ContractError(format!("unknown Evidence ID {id}")))?;
    let origin = string(record, "origin")?;
    let parents = array(record, "parent_evidence_ids")?;
    let result = match origin {
        "independent_observation" => {
            if !parents.is_empty() {
                return Err(ContractError(format!(
                    "independent observation {id} cannot have derived parents"
                )));
            }
            true
        }
        "independent_derivation" => {
            if parents.is_empty() {
                false
            } else {
                let mut rooted = false;
                for parent in parents {
                    let parent = parent.as_str().ok_or_else(|| {
                        ContractError(format!("parent Evidence ID must be a string for {id}"))
                    })?;
                    if !evidence.contains_key(parent) {
                        return Err(ContractError(format!(
                            "Evidence {id} references missing parent {parent}"
                        )));
                    }
                    rooted |= has_independent_root(parent, evidence, visiting, memo)?;
                }
                rooted
            }
        }
        "model_generated" | "synthetic" | "external_reference" => false,
        other => return Err(ContractError(format!("unknown Evidence origin {other}"))),
    };
    visiting.remove(id);
    memo.insert(id.to_owned(), result);
    Ok(result)
}

pub fn validate_contract(value: &Value) -> Result<(), ContractError> {
    let mode = string(value, "qualification_mode")?;
    if !matches!(mode, "math_only_synthetic" | "planner_integration_observed") {
        return Err(ContractError(format!(
            "unsupported qualification mode {mode}"
        )));
    }

    let mut evidence = HashMap::<String, &Value>::new();
    for record in array(value, "evidence_catalog")? {
        let id = string(record, "evidence_id")?;
        if evidence.insert(id.to_owned(), record).is_some() {
            return Err(ContractError(format!("duplicate Evidence ID {id}")));
        }
    }
    for (id, record) in &evidence {
        let parents = array(record, "parent_evidence_ids")?;
        let parent_ids = string_set(parents, &format!("Evidence {id} parents"))?;
        for parent in parent_ids {
            if !evidence.contains_key(&parent) {
                return Err(ContractError(format!(
                    "Evidence {id} references missing parent {parent}"
                )));
            }
        }
        let origin = string(record, "origin")?;
        match mode {
            "math_only_synthetic" if origin != "synthetic" => {
                return Err(ContractError(format!(
                    "math_only_synthetic evidence {id} must remain labelled synthetic"
                )));
            }
            "planner_integration_observed"
                if !matches!(origin, "independent_observation" | "independent_derivation") =>
            {
                return Err(ContractError(format!(
                    "planner integration evidence {id} cannot be rooted only in {origin}"
                )));
            }
            _ => {}
        }
    }

    let hypotheses = array(value, "hypotheses")?;
    let mut hypothesis_ids = BTreeSet::new();
    let mut prior_total = 0.0;
    for hypothesis in hypotheses {
        let id = string(hypothesis, "id")?;
        if !hypothesis_ids.insert(id.to_owned()) {
            return Err(ContractError(format!("duplicate hypothesis ID {id}")));
        }
        let prior = number(hypothesis, "prior_probability")?;
        if prior <= 0.0 || prior > 1.0 {
            return Err(ContractError(format!(
                "hypothesis {id} prior must be in (0,1]"
            )));
        }
        prior_total += prior;

        let supports = array(hypothesis, "supporting_evidence_ids")?;
        let support_ids = string_set(supports, &format!("{id} supports"))?;
        if support_ids.is_empty() {
            return Err(ContractError(format!("hypothesis {id} is ungrounded")));
        }
        for evidence_id in support_ids {
            if !evidence.contains_key(&evidence_id) {
                return Err(ContractError(format!(
                    "hypothesis {id} references missing support Evidence {evidence_id}"
                )));
            }
        }
        let contradictions = array(hypothesis, "contradicting_evidence_ids")?;
        for evidence_id in string_set(contradictions, &format!("{id} contradictions"))? {
            if !evidence.contains_key(&evidence_id) {
                return Err(ContractError(format!(
                    "hypothesis {id} references missing contradiction Evidence {evidence_id}"
                )));
            }
        }

        if mode == "planner_integration_observed" {
            let mut grounded = false;
            let mut visiting = HashSet::new();
            let mut memo = HashMap::new();
            for evidence_id in array(hypothesis, "supporting_evidence_ids")? {
                let evidence_id = evidence_id.as_str().ok_or_else(|| {
                    ContractError(format!("support Evidence ID must be a string in {id}"))
                })?;
                grounded |= has_independent_root(evidence_id, &evidence, &mut visiting, &mut memo)?;
            }
            if !grounded {
                return Err(ContractError(format!(
                    "hypothesis {id} lacks an independently rooted supporting Evidence record"
                )));
            }
        }
    }
    if (prior_total - 1.0).abs() > SUM_TOLERANCE {
        return Err(ContractError(format!(
            "hypothesis prior mass is {prior_total}, not 1"
        )));
    }

    let experiments = array(value, "experiments")?;
    let mut experiment_ids = BTreeSet::new();
    let mut outcome_map = BTreeMap::<String, Vec<String>>::new();
    for experiment in experiments {
        let id = string(experiment, "id")?;
        if !experiment_ids.insert(id.to_owned()) {
            return Err(ContractError(format!("duplicate experiment ID {id}")));
        }
        let outcomes = string_vec(array(experiment, "outcomes")?, &format!("{id} outcomes"))?;
        if outcomes.len() < 2 {
            return Err(ContractError(format!(
                "{id} requires at least two outcomes"
            )));
        }
        outcome_map.insert(id.to_owned(), outcomes.clone());
        normalized_probability_rows(
            array(experiment, "point_likelihoods")?,
            &outcomes,
            &hypothesis_ids,
            &format!("{id} point likelihoods"),
        )?;
    }

    let scenarios = array(value, "likelihood_scenarios")?;
    let mut scenario_ids = BTreeSet::new();
    let mut scenario_total = 0.0;
    for scenario in scenarios {
        let id = string(scenario, "scenario_id")?;
        if !scenario_ids.insert(id.to_owned()) {
            return Err(ContractError(format!("duplicate likelihood scenario {id}")));
        }
        let probability = number(scenario, "scenario_probability")?;
        if probability <= 0.0 || probability > 1.0 {
            return Err(ContractError(format!(
                "scenario {id} probability must be in (0,1]"
            )));
        }
        scenario_total += probability;

        let matrices = array(scenario, "experiments")?;
        let mut matrix_ids = BTreeSet::new();
        for matrix in matrices {
            let experiment_id = string(matrix, "experiment_id")?;
            if !experiment_ids.contains(experiment_id) {
                return Err(ContractError(format!(
                    "scenario {id} references unknown experiment {experiment_id}"
                )));
            }
            if !matrix_ids.insert(experiment_id.to_owned()) {
                return Err(ContractError(format!(
                    "scenario {id} duplicates experiment {experiment_id}"
                )));
            }
            let outcome_order = outcome_map.get(experiment_id).ok_or_else(|| {
                ContractError(format!(
                    "experiment {experiment_id} has no outcome ordering"
                ))
            })?;
            normalized_probability_rows(
                array(matrix, "likelihoods")?,
                outcome_order,
                &hypothesis_ids,
                &format!("{id}/{experiment_id} likelihoods"),
            )?;
        }
        if matrix_ids != experiment_ids {
            return Err(ContractError(format!(
                "scenario {id} does not define exactly one matrix for every experiment"
            )));
        }
    }
    if (scenario_total - 1.0).abs() > SUM_TOLERANCE {
        return Err(ContractError(format!(
            "likelihood scenario mass is {scenario_total}, not 1"
        )));
    }

    let mut decision_ids = BTreeSet::new();
    for decision in array(value, "terminal_decisions")? {
        let id = string(decision, "id")?;
        if !decision_ids.insert(id.to_owned()) {
            return Err(ContractError(format!("duplicate terminal decision {id}")));
        }
        let utilities = array(decision, "utilities")?;
        let mut utility_ids = BTreeSet::new();
        for utility in utilities {
            let hypothesis_id = string(utility, "hypothesis_id")?;
            if !hypothesis_ids.contains(hypothesis_id) {
                return Err(ContractError(format!(
                    "decision {id} utility references unknown hypothesis {hypothesis_id}"
                )));
            }
            if !utility_ids.insert(hypothesis_id.to_owned()) {
                return Err(ContractError(format!(
                    "decision {id} duplicates utility for {hypothesis_id}"
                )));
            }
            let _utility = number(utility, "utility")?;
        }
        if utility_ids != hypothesis_ids {
            return Err(ContractError(format!(
                "decision {id} does not define one utility for every hypothesis"
            )));
        }
    }

    if mode == "planner_integration_observed" {
        for section in ["prior", "likelihood_models"] {
            let status = string(
                object(value, "input_provenance")?
                    .get(section)
                    .ok_or_else(|| ContractError(format!("input_provenance.{section} missing")))?,
                "calibration_status",
            )?;
            if status == "synthetic_known_answer" {
                return Err(ContractError(format!(
                    "planner integration cannot use synthetic known-answer {section}"
                )));
            }
        }
    }

    Ok(())
}

/// Validate the Draft 2020-12 schema, example instance and cross-field rules.
pub fn validate_embedded_example() -> Result<(), ContractError> {
    let schema: Value = serde_json::from_str(SCHEMA)
        .map_err(|error| ContractError(format!("schema JSON parse failed: {error}")))?;
    draft202012::meta::validate(&schema)
        .map_err(|error| ContractError(format!("Draft 2020-12 schema invalid: {error}")))?;

    let example: Value = serde_json::from_str(EXAMPLE)
        .map_err(|error| ContractError(format!("example JSON parse failed: {error}")))?;
    let validator = draft202012::options()
        .build(&schema)
        .map_err(|error| ContractError(format!("schema compilation failed: {error}")))?;
    let errors: Vec<String> = validator
        .iter_errors(&example)
        .map(|error| error.to_string())
        .collect();
    if !errors.is_empty() {
        return Err(ContractError(format!(
            "example fails JSON Schema validation: {}",
            errors.join("; ")
        )));
    }
    validate_contract(&example)
}


/// Validate the three-action policy-disagreement fixture and its grader-only
/// effects. The public planner API is intentionally not called: this is a
/// synthetic mathematical oracle, not observed-Evidence integration.
pub fn validate_policy_disagreement_fixture() -> Result<(), ContractError> {
    let schema: Value = serde_json::from_str(SCHEMA)
        .map_err(|error| ContractError(format!("planner schema parse failed: {error}")))?;
    draft202012::meta::validate(&schema)
        .map_err(|error| ContractError(format!("planner schema invalid: {error}")))?;
    let agent: Value = serde_json::from_str(POLICY_AGENT)
        .map_err(|error| ContractError(format!("policy agent input parse failed: {error}")))?;
    let validator = draft202012::options()
        .build(&schema)
        .map_err(|error| ContractError(format!("planner schema compilation failed: {error}")))?;
    let errors: Vec<String> = validator
        .iter_errors(&agent)
        .map(|error| error.to_string())
        .collect();
    if !errors.is_empty() {
        return Err(ContractError(format!(
            "policy agent input fails planner schema: {}",
            errors.join("; ")
        )));
    }
    validate_contract(&agent)?;

    let grader_schema: Value = serde_json::from_str(POLICY_GRADER_SCHEMA)
        .map_err(|error| ContractError(format!("policy grader schema parse failed: {error}")))?;
    draft202012::meta::validate(&grader_schema)
        .map_err(|error| ContractError(format!("policy grader schema invalid: {error}")))?;
    let grader: Value = serde_json::from_str(POLICY_GRADER)
        .map_err(|error| ContractError(format!("policy grader parse failed: {error}")))?;
    let grader_validator = draft202012::options()
        .build(&grader_schema)
        .map_err(|error| ContractError(format!("policy grader schema compilation failed: {error}")))?;
    let grader_errors: Vec<String> = grader_validator
        .iter_errors(&grader)
        .map(|error| error.to_string())
        .collect();
    if !grader_errors.is_empty() {
        return Err(ContractError(format!(
            "policy grader fails schema: {}",
            grader_errors.join("; ")
        )));
    }

    if string(&agent, "case_id")? != string(&grader, "case_id")? {
        return Err(ContractError(
            "agent and grader case IDs must match".to_owned(),
        ));
    }
    for forbidden in [
        "candidate_effects",
        "expected_policy_rankings",
        "oracle_policy",
        "world_state_oracle",
    ] {
        if agent.get(forbidden).is_some() {
            return Err(ContractError(format!(
                "grader-only field {forbidden} leaked into agent-visible planner input"
            )));
        }
    }

    let experiments = array(&agent, "experiments")?;
    let mut experiment_outcomes = BTreeMap::<String, BTreeSet<String>>::new();
    for experiment in experiments {
        let id = string(experiment, "id")?;
        let outcomes = string_set(array(experiment, "outcomes")?, &format!("{id} outcomes"))?;
        experiment_outcomes.insert(id.to_owned(), outcomes);
    }

    let obligations = array(&grader, "obligations")?;
    let mut obligation_classes = BTreeMap::<String, BTreeSet<String>>::new();
    for obligation in obligations {
        let id = string(obligation, "obligation_id")?;
        let classes = string_set(
            array(obligation, "admissible_evidence_classes")?,
            &format!("{id} admissible evidence classes"),
        )?;
        obligation_classes.insert(id.to_owned(), classes);
    }

    let mut grader_candidates = BTreeSet::new();
    for candidate in array(&grader, "candidate_effects")? {
        let candidate_id = string(candidate, "experiment_id")?;
        if !grader_candidates.insert(candidate_id.to_owned()) {
            return Err(ContractError(format!(
                "duplicate grader candidate {candidate_id}"
            )));
        }
        let allowed_outcomes = experiment_outcomes.get(candidate_id).ok_or_else(|| {
            ContractError(format!(
                "grader references unknown agent-visible experiment {candidate_id}"
            ))
        })?;
        for effect in array(candidate, "effects")? {
            let outcome = string(effect, "outcome")?;
            if !allowed_outcomes.contains(outcome) {
                return Err(ContractError(format!(
                    "grader effect references undeclared outcome {outcome} for {candidate_id}"
                )));
            }
            let obligation_id = string(effect, "obligation_id")?;
            let classes = obligation_classes.get(obligation_id).ok_or_else(|| {
                ContractError(format!(
                    "grader effect references unknown obligation {obligation_id}"
                ))
            })?;
            let evidence_class = string(effect, "evidence_class")?;
            if !classes.contains(evidence_class) {
                return Err(ContractError(format!(
                    "grader effect uses inadmissible evidence class {evidence_class} for {obligation_id}"
                )));
            }
        }
    }
    if grader_candidates.len() != experiment_outcomes.len()
        || grader_candidates.iter().any(|id| !experiment_outcomes.contains_key(id))
    {
        return Err(ContractError(
            "agent and grader candidate identity sets must match exactly".to_owned(),
        ));
    }

    let rankings = object(&grader, "expected_policy_rankings")?;
    for key in [
        "point_eig_best",
        "expected_scenario_eig_best",
        "lower_tail_cvar_eig_best",
        "expected_evsi_best",
        "obligation_closure_per_second_best",
    ] {
        let id = rankings
            .get(key)
            .and_then(Value::as_str)
            .ok_or_else(|| ContractError(format!("expected policy ranking {key} must be a string")))?;
        if !experiment_outcomes.contains_key(id) {
            return Err(ContractError(format!(
                "policy ranking {key} references unknown candidate {id}"
            )));
        }
    }
    for item in array(&Value::Object(rankings.clone()), "combined_gated_sequence")? {
        let id = item
            .as_str()
            .ok_or_else(|| ContractError("combined policy sequence must contain strings".to_owned()))?;
        if !experiment_outcomes.contains_key(id) {
            return Err(ContractError(format!(
                "combined policy sequence references unknown candidate {id}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
fn entropy_bits(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .copied()
        .filter(|probability| *probability > 0.0)
        .map(|probability| -probability * probability.log2())
        .sum()
}

#[cfg(test)]
fn hypothesis_priors(agent: &Value) -> Result<(Vec<String>, Vec<f64>), ContractError> {
    let mut ids = Vec::new();
    let mut priors = Vec::new();
    for hypothesis in array(agent, "hypotheses")? {
        ids.push(string(hypothesis, "id")?.to_owned());
        priors.push(number(hypothesis, "prior_probability")?);
    }
    Ok((ids, priors))
}

#[cfg(test)]
fn eig_from_rows(
    hypothesis_ids: &[String],
    priors: &[f64],
    rows: &[Value],
) -> Result<f64, ContractError> {
    let prior_entropy = entropy_bits(priors);
    let mut expected_posterior_entropy = 0.0;
    let outcomes = array(rows.first().ok_or_else(|| {
        ContractError("EIG requires at least one likelihood row".to_owned())
    })?, "outcome_probabilities")?
    .len();

    for outcome_index in 0..outcomes {
        let mut predictive = 0.0;
        let mut posterior = vec![0.0; hypothesis_ids.len()];
        for (hypothesis_index, hypothesis_id) in hypothesis_ids.iter().enumerate() {
            let row = rows
                .iter()
                .find(|row| row.get("hypothesis_id").and_then(Value::as_str) == Some(hypothesis_id))
                .ok_or_else(|| ContractError(format!("missing likelihood row for {hypothesis_id}")))?;
            let probability = array(row, "outcome_probabilities")?[outcome_index]
                .as_f64()
                .ok_or_else(|| ContractError("outcome probability must be numeric".to_owned()))?;
            let joint = priors[hypothesis_index] * probability;
            predictive += joint;
            posterior[hypothesis_index] = joint;
        }
        if predictive > 0.0 {
            for probability in &mut posterior {
                *probability /= predictive;
            }
            expected_posterior_entropy += predictive * entropy_bits(&posterior);
        }
    }
    Ok(prior_entropy - expected_posterior_entropy)
}

#[cfg(test)]
fn best_decision_utility(
    hypothesis_ids: &[String],
    posterior: &[f64],
    decisions: &[Value],
) -> Result<f64, ContractError> {
    let mut best = f64::NEG_INFINITY;
    for decision in decisions {
        let mut expected = 0.0;
        for (index, hypothesis_id) in hypothesis_ids.iter().enumerate() {
            let utility = array(decision, "utilities")?
                .iter()
                .find(|item| item.get("hypothesis_id").and_then(Value::as_str) == Some(hypothesis_id))
                .ok_or_else(|| ContractError(format!("missing decision utility for {hypothesis_id}")))?;
            expected += posterior[index] * number(utility, "utility")?;
        }
        best = best.max(expected);
    }
    Ok(best)
}

#[cfg(test)]
fn evsi_from_rows(
    hypothesis_ids: &[String],
    priors: &[f64],
    rows: &[Value],
    decisions: &[Value],
) -> Result<f64, ContractError> {
    let baseline = best_decision_utility(hypothesis_ids, priors, decisions)?;
    let outcomes = array(rows.first().ok_or_else(|| {
        ContractError("EVSI requires at least one likelihood row".to_owned())
    })?, "outcome_probabilities")?
    .len();
    let mut expected_posterior_best = 0.0;

    for outcome_index in 0..outcomes {
        let mut predictive = 0.0;
        let mut posterior = vec![0.0; hypothesis_ids.len()];
        for (hypothesis_index, hypothesis_id) in hypothesis_ids.iter().enumerate() {
            let row = rows
                .iter()
                .find(|row| row.get("hypothesis_id").and_then(Value::as_str) == Some(hypothesis_id))
                .ok_or_else(|| ContractError(format!("missing likelihood row for {hypothesis_id}")))?;
            let probability = array(row, "outcome_probabilities")?[outcome_index]
                .as_f64()
                .ok_or_else(|| ContractError("outcome probability must be numeric".to_owned()))?;
            let joint = priors[hypothesis_index] * probability;
            predictive += joint;
            posterior[hypothesis_index] = joint;
        }
        if predictive > 0.0 {
            for probability in &mut posterior {
                *probability /= predictive;
            }
            expected_posterior_best += predictive
                * best_decision_utility(hypothesis_ids, &posterior, decisions)?;
        }
    }
    Ok(expected_posterior_best - baseline)
}

#[cfg(test)]
fn lower_tail_cvar(values: &[(f64, f64)], alpha: f64) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(|left, right| left.0.total_cmp(&right.0));
    let mut remaining = alpha;
    let mut weighted = 0.0;
    for (value, probability) in ordered {
        let included = remaining.min(probability);
        weighted += included * value;
        remaining -= included;
        if remaining <= 0.0 {
            break;
        }
    }
    weighted / alpha
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example() -> Value {
        serde_json::from_str(EXAMPLE).expect("checked-in example must parse")
    }

    #[test]
    fn draft_2020_12_schema_and_synthetic_example_validate() {
        validate_embedded_example().unwrap();
    }

    #[test]
    fn prior_mass_must_sum_to_one() {
        let mut value = example();
        value["hypotheses"][0]["prior_probability"] = serde_json::json!(0.6);
        assert!(validate_contract(&value).is_err());
    }

    #[test]
    fn likelihood_rows_must_match_hypotheses_and_outcomes() {
        let mut value = example();
        value["experiments"][0]["point_likelihoods"][0]["outcome_probabilities"] =
            serde_json::json!([0.9]);
        assert!(validate_contract(&value).is_err());
    }

    #[test]
    fn scenario_weights_must_sum_to_one() {
        let mut value = example();
        value["likelihood_scenarios"][0]["scenario_probability"] = serde_json::json!(0.6);
        assert!(validate_contract(&value).is_err());
    }

    #[test]
    fn terminal_decisions_require_complete_utility_matrices() {
        let mut value = example();
        value["terminal_decisions"][0]["utilities"]
            .as_array_mut()
            .unwrap()
            .pop();
        assert!(validate_contract(&value).is_err());
    }

    #[test]
    fn synthetic_evidence_cannot_enter_observed_planner_mode() {
        let mut value = example();
        value["qualification_mode"] = Value::String("planner_integration_observed".to_owned());
        assert!(validate_contract(&value).is_err());
    }

    #[test]
    fn evidence_references_must_resolve() {
        let mut value = example();
        value["hypotheses"][0]["supporting_evidence_ids"][0] =
            Value::String(format!("ev_{}", "f".repeat(64)));
        assert!(validate_contract(&value).is_err());
    }

    #[test]
    fn scenario_must_cover_every_experiment() {
        let mut value = example();
        value["likelihood_scenarios"][0]["experiments"]
            .as_array_mut()
            .unwrap()
            .clear();
        assert!(validate_contract(&value).is_err());
    }

    fn binary_entropy(probability: f64) -> f64 {
        [probability, 1.0 - probability]
            .into_iter()
            .filter(|p| *p > 0.0)
            .map(|p| -p * p.log2())
            .sum()
    }

    fn best_expected_utility(probability_vulnerable: f64) -> f64 {
        let mitigate = probability_vulnerable * 3.0 + (1.0 - probability_vulnerable) * -6.0;
        let no_action = probability_vulnerable * -20.0;
        mitigate.max(no_action)
    }

    #[test]
    fn synthetic_math_example_matches_hand_derived_eig_and_evsi() {
        let prior_vulnerable = 0.5;
        let prior_entropy = binary_entropy(prior_vulnerable);
        assert!((prior_entropy - 1.0).abs() < 1.0e-12);

        let nominal_eig = prior_entropy - 0.5 * binary_entropy(0.9) - 0.5 * binary_entropy(0.1);
        let conservative_eig =
            prior_entropy - 0.5 * binary_entropy(0.75) - 0.5 * binary_entropy(0.25);
        assert!((nominal_eig - 0.531_004_406_410_718_8).abs() < 1.0e-12);
        assert!((conservative_eig - 0.188_721_875_540_867_17).abs() < 1.0e-12);
        assert!(((nominal_eig + conservative_eig) / 2.0 - 0.359_863_140_975_793).abs() < 1.0e-12);

        let prior_best_utility = best_expected_utility(prior_vulnerable);
        assert!((prior_best_utility - -1.5).abs() < 1.0e-12);

        let nominal_posterior_best =
            0.5 * best_expected_utility(0.9) + 0.5 * best_expected_utility(0.1);
        let conservative_posterior_best =
            0.5 * best_expected_utility(0.75) + 0.5 * best_expected_utility(0.25);

        let nominal_evsi = nominal_posterior_best - prior_best_utility;
        let conservative_evsi = conservative_posterior_best - prior_best_utility;
        assert!((nominal_evsi - 1.55).abs() < 1.0e-12);
        assert!(conservative_evsi.abs() < 1.0e-12);
        assert!(((nominal_evsi + conservative_evsi) / 2.0 - 0.775).abs() < 1.0e-12);
    }

    #[test]
    fn likelihood_probability_vector_must_bind_exact_outcome_order() {
        let mut value = example();
        value["likelihood_scenarios"][0]["experiments"][0]["likelihoods"][0]["outcome_order"] =
            serde_json::json!(["not_reproduced", "reproduced"]);
        assert!(validate_contract(&value).is_err());
    }

    #[test]
    fn point_likelihood_probability_vector_must_bind_exact_outcome_order() {
        let mut value = example();
        value["experiments"][0]["point_likelihoods"][0]["outcome_order"] =
            serde_json::json!(["not_reproduced", "reproduced"]);
        assert!(validate_contract(&value).is_err());
    }


    fn policy_agent() -> Value {
        serde_json::from_str(POLICY_AGENT).expect("policy agent input must parse")
    }

    fn policy_grader() -> Value {
        serde_json::from_str(POLICY_GRADER).expect("policy grader input must parse")
    }

    fn experiment<'a>(agent: &'a Value, id: &str) -> &'a Value {
        array(agent, "experiments")
            .unwrap()
            .iter()
            .find(|item| item.get("id").and_then(Value::as_str) == Some(id))
            .unwrap()
    }

    fn scenario_rows<'a>(scenario: &'a Value, experiment_id: &str) -> &'a [Value] {
        let matrix = array(scenario, "experiments")
            .unwrap()
            .iter()
            .find(|item| item.get("experiment_id").and_then(Value::as_str) == Some(experiment_id))
            .unwrap();
        array(matrix, "likelihoods").unwrap()
    }


    fn normalized_planner_cost(experiment: &Value, base_config: &Value) -> f64 {
        let weights = &base_config["cost_weights"];
        weights["time"].as_f64().unwrap()
            * experiment["estimated_time_seconds"].as_f64().unwrap()
            / base_config["max_time_seconds"].as_f64().unwrap()
            + weights["energy"].as_f64().unwrap()
                * experiment["estimated_energy_joules"].as_f64().unwrap()
                / base_config["max_energy_joules"].as_f64().unwrap()
            + weights["risk"].as_f64().unwrap()
                * experiment["estimated_risk"].as_f64().unwrap()
                / base_config["max_risk"].as_f64().unwrap()
    }

    fn scenario_named<'a>(agent: &'a Value, id: &str) -> &'a Value {
        array(agent, "likelihood_scenarios")
            .unwrap()
            .iter()
            .find(|item| item.get("scenario_id").and_then(Value::as_str) == Some(id))
            .unwrap()
    }

    #[test]
    fn policy_disagreement_agent_and_grader_views_validate() {
        validate_policy_disagreement_fixture().unwrap();
    }

    #[test]
    fn policy_objectives_have_distinct_exact_known_answer_rankings() {
        let agent = policy_agent();
        let hypothesis_ids_and_priors = hypothesis_priors(&agent).unwrap();
        let (hypothesis_ids, priors) = hypothesis_ids_and_priors;
        let decisions = array(&agent, "terminal_decisions").unwrap();
        let nominal = scenario_named(&agent, "nominal");
        let conservative = scenario_named(&agent, "conservative");
        let scenario_weights = [
            number(nominal, "scenario_probability").unwrap(),
            number(conservative, "scenario_probability").unwrap(),
        ];

        let mut point_eig = BTreeMap::new();
        let mut expected_scenario_eig = BTreeMap::new();
        let mut robust_lower_tail_eig = BTreeMap::new();
        let mut expected_evsi = BTreeMap::new();

        for item in array(&agent, "experiments").unwrap() {
            let id = string(item, "id").unwrap().to_owned();
            let point_rows = array(item, "point_likelihoods").unwrap();
            point_eig.insert(
                id.clone(),
                eig_from_rows(&hypothesis_ids, &priors, point_rows).unwrap(),
            );

            let nominal_eig = eig_from_rows(
                &hypothesis_ids,
                &priors,
                scenario_rows(nominal, &id),
            )
            .unwrap();
            let conservative_eig = eig_from_rows(
                &hypothesis_ids,
                &priors,
                scenario_rows(conservative, &id),
            )
            .unwrap();

            expected_scenario_eig.insert(
                id.clone(),
                scenario_weights[0] * nominal_eig
                    + scenario_weights[1] * conservative_eig,
            );
            robust_lower_tail_eig.insert(
                id.clone(),
                lower_tail_cvar(
                    &[
                        (nominal_eig, scenario_weights[0]),
                        (conservative_eig, scenario_weights[1]),
                    ],
                    0.25,
                ),
            );

            let nominal_evsi =
                evsi_from_rows(&hypothesis_ids, &priors, scenario_rows(nominal, &id), decisions)
                    .unwrap();
            let conservative_evsi = evsi_from_rows(
                &hypothesis_ids,
                &priors,
                scenario_rows(conservative, &id),
                decisions,
            )
            .unwrap();
            expected_evsi.insert(
                id,
                scenario_weights[0] * nominal_evsi
                    + scenario_weights[1] * conservative_evsi,
            );
        }

        let max_key = |values: &BTreeMap<String, f64>| {
            values
                .iter()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(key, _)| key.clone())
                .expect("each policy score map contains every candidate")
        };
        assert_eq!(max_key(&point_eig), "static_function_probe");
        assert_eq!(max_key(&expected_scenario_eig), "static_function_probe");
        assert_eq!(max_key(&robust_lower_tail_eig), "controlled_reproduction");
        assert_eq!(max_key(&expected_evsi), "static_function_probe");

        // Existing point/robust planners divide the selected information gain
        // by this normalized cost. The expected winners remain distinct after
        // applying the exact implementation's cost normalization.
        let point_base = &agent["planner_configs"]["point"];
        let robust_base = &agent["planner_configs"]["robust"]["base"];
        let point_utility = |id: &str| {
            let cost = normalized_planner_cost(experiment(&agent, id), point_base);
            point_eig[id] / cost
        };
        let robust_utility = |id: &str| {
            let cost = normalized_planner_cost(experiment(&agent, id), robust_base);
            robust_lower_tail_eig[id] / cost
        };
        assert!(point_utility("static_function_probe") > point_utility("controlled_reproduction"));
        assert!(point_utility("static_function_probe") > point_utility("verify_deployment_manifest"));
        assert!(robust_utility("controlled_reproduction") > robust_utility("static_function_probe"));
        assert!(robust_utility("controlled_reproduction") > robust_utility("verify_deployment_manifest"));
        assert!((normalized_planner_cost(experiment(&agent, "static_function_probe"), point_base)
            - 0.383_333_333_333_333_3).abs() < 1.0e-12);
        assert!((normalized_planner_cost(experiment(&agent, "verify_deployment_manifest"), point_base)
            - 0.103_333_333_333_333_35).abs() < 1.0e-12);
        assert!((normalized_planner_cost(experiment(&agent, "controlled_reproduction"), point_base)
            - 0.733_333_333_333_333_4).abs() < 1.0e-12);

        let grader = policy_grader();
        let golden = &grader["golden_values"];
        assert!((golden["static_probe_normalized_cost"].as_f64().unwrap()
            - normalized_planner_cost(experiment(&agent, "static_function_probe"), point_base))
            .abs() < 1.0e-12);
        assert!((golden["manifest_normalized_cost"].as_f64().unwrap()
            - normalized_planner_cost(experiment(&agent, "verify_deployment_manifest"), point_base))
            .abs() < 1.0e-12);
        assert!((golden["reproduction_normalized_cost"].as_f64().unwrap()
            - normalized_planner_cost(experiment(&agent, "controlled_reproduction"), point_base))
            .abs() < 1.0e-12);
        assert!((golden["static_probe_point_planner_utility"].as_f64().unwrap()
            - point_utility("static_function_probe"))
            .abs() < 1.0e-12);
        assert!((golden["reproduction_point_planner_utility"].as_f64().unwrap()
            - point_utility("controlled_reproduction"))
            .abs() < 1.0e-12);
        assert!((golden["static_probe_robust_lower_tail_utility"].as_f64().unwrap()
            - robust_utility("static_function_probe"))
            .abs() < 1.0e-12);
        assert!((golden["reproduction_robust_lower_tail_utility"].as_f64().unwrap()
            - robust_utility("controlled_reproduction"))
            .abs() < 1.0e-12);

        assert!((point_eig["static_function_probe"] - 0.531_004_406_410_718_9).abs() < 1.0e-12);
        assert!((point_eig["controlled_reproduction"] - 0.278_071_905_112_637_7).abs() < 1.0e-12);
        assert!((expected_scenario_eig["static_function_probe"] - 0.265_646_482_328_954_5).abs() < 1.0e-12);
        assert!((expected_scenario_eig["controlled_reproduction"] - 0.233_396_890_326_752_43).abs() < 1.0e-12);
        assert!((robust_lower_tail_eig["static_function_probe"] - 0.000_288_558_247_190_096_04).abs() < 1.0e-12);
        assert!((robust_lower_tail_eig["controlled_reproduction"] - 0.188_721_875_540_867_17).abs() < 1.0e-12);
        assert!((expected_evsi["static_function_probe"] - 0.775).abs() < 1.0e-12);
        assert!((expected_evsi["controlled_reproduction"] - 0.05).abs() < 1.0e-12);
    }

    #[test]
    fn proof_obligation_closure_per_second_prefers_manifest_then_reproduction() {
        let agent = policy_agent();
        let grader = policy_grader();
        let effects = array(&grader, "candidate_effects").unwrap();
        let mut closure_rate = BTreeMap::new();

        for candidate in effects {
            let id = string(candidate, "experiment_id").unwrap();
            let experiment = experiment(&agent, id);
            let time = number(experiment, "estimated_time_seconds").unwrap();
            let mut closes = BTreeSet::new();
            for effect in array(candidate, "effects").unwrap() {
                if string(effect, "resulting_status").unwrap() == "satisfied" {
                    closes.insert(string(effect, "obligation_id").unwrap().to_owned());
                }
            }
            closure_rate.insert(id.to_owned(), closes.len() as f64 / time);
        }
        let winner = closure_rate
            .iter()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .unwrap()
            .0;
        assert_eq!(winner, "verify_deployment_manifest");
        assert!((closure_rate["verify_deployment_manifest"] - 0.5).abs() < 1.0e-12);
        assert!((closure_rate["controlled_reproduction"] - 0.125).abs() < 1.0e-12);
        assert_eq!(
            array(
                grader.get("expected_policy_rankings").unwrap(),
                "combined_gated_sequence"
            )
            .unwrap()
            .iter()
            .map(|item| item.as_str().unwrap())
            .collect::<Vec<_>>(),
            vec!["verify_deployment_manifest", "controlled_reproduction"]
        );
    }

    #[test]
    fn predicted_outcomes_do_not_make_claim_ready_but_observed_effects_can() {
        let grader = policy_grader();
        let mut states = BTreeMap::from([
            ("deployment_binding".to_owned(), "unknown".to_owned()),
            ("vulnerability_behavior".to_owned(), "unknown".to_owned()),
        ]);

        let ready = |states: &BTreeMap<String, String>| {
            states.values().all(|status| status == "satisfied")
        };
        assert!(!ready(&states));

        // Candidate rankings and predicted likelihoods are not observations.
        assert!(!ready(&states));

        let apply = |candidate_id: &str, outcome: &str, states: &mut BTreeMap<String, String>| {
            let candidate = array(&grader, "candidate_effects")
                .unwrap()
                .iter()
                .find(|item| item["experiment_id"].as_str() == Some(candidate_id))
                .unwrap();
            for effect in array(candidate, "effects").unwrap() {
                if effect["outcome"].as_str() == Some(outcome) {
                    let obligation = effect["obligation_id"].as_str().unwrap();
                    let result = effect["resulting_status"].as_str().unwrap();
                    states.insert(obligation.to_owned(), result.to_owned());
                }
            }
        };

        apply(
            "verify_deployment_manifest",
            "digest_match",
            &mut states,
        );
        assert_eq!(states["deployment_binding"], "satisfied");
        assert_eq!(states["vulnerability_behavior"], "unknown");
        assert!(!ready(&states));

        apply("controlled_reproduction", "reproduced", &mut states);
        assert!(ready(&states));
    }

    #[test]
    fn policy_agent_schema_rejects_grader_oracle_leakage() {
        let schema: Value = serde_json::from_str(SCHEMA).unwrap();
        let mut agent = policy_agent();
        let validator = draft202012::options().build(&schema).unwrap();
        assert!(validator.is_valid(&agent));
        // The base planner contract permits no grader-only property.
        agent["world_state_oracle"] = Value::String("vulnerable".to_owned());
        assert!(!validator.is_valid(&agent));
    }

    #[test]
    fn policy_grader_rejects_effects_for_undeclared_candidate_outcomes() {
        let schema: Value = serde_json::from_str(POLICY_GRADER_SCHEMA).unwrap();
        draft202012::meta::validate(&schema).unwrap();
        let validator = draft202012::options().build(&schema).unwrap();
        let mut grader = policy_grader();
        grader["candidate_effects"][1]["effects"][0]["outcome"] =
            Value::String("undeclared_outcome".to_owned());
        assert!(!validator.is_valid(&grader));
    }

    #[test]
    fn policy_input_rejects_same_length_outcome_reordering() {
        let mut agent = policy_agent();
        agent["experiments"][1]["point_likelihoods"][0]["outcome_order"] =
            serde_json::json!(["digest_mismatch", "digest_match"]);
        assert!(validate_contract(&agent).is_err());
    }

    #[test]
    fn policy_grader_exact_golden_values_cannot_be_mutated() {
        let schema: Value = serde_json::from_str(POLICY_GRADER_SCHEMA).unwrap();
        let validator = draft202012::options().build(&schema).unwrap();
        let mut grader = policy_grader();
        grader["golden_values"]["static_probe_expected_evsi_units"] =
            serde_json::json!(0.76);
        assert!(!validator.is_valid(&grader));
    }

}
