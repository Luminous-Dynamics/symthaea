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
            return Err(ContractError(format!("{context} contains duplicate {value}")));
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
                ContractError(format!("experiment {experiment_id} has no outcome ordering"))
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
        let mitigate = probability_vulnerable * 3.0
            + (1.0 - probability_vulnerable) * -6.0;
        let no_action = probability_vulnerable * -20.0;
        mitigate.max(no_action)
    }

    #[test]
    fn synthetic_math_example_matches_hand_derived_eig_and_evsi() {
        let prior_vulnerable = 0.5;
        let prior_entropy = binary_entropy(prior_vulnerable);
        assert!((prior_entropy - 1.0).abs() < 1.0e-12);

        let nominal_eig =
            prior_entropy - 0.5 * binary_entropy(0.9) - 0.5 * binary_entropy(0.1);
        let conservative_eig =
            prior_entropy - 0.5 * binary_entropy(0.75) - 0.5 * binary_entropy(0.25);
        assert!((nominal_eig - 0.531_004_406_410_718_8).abs() < 1.0e-12);
        assert!((conservative_eig - 0.188_721_875_540_867_17).abs() < 1.0e-12);
        assert!(
            ((nominal_eig + conservative_eig) / 2.0 - 0.359_863_140_975_793).abs() < 1.0e-12
        );

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

}
