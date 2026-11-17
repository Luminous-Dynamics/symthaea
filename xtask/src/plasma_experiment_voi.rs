use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

const PROBABILITY_TOLERANCE: f64 = 1.0e-9;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: String,
    decision_id: String,
    evidence_class: String,
    decision_loss_unit: String,
    measurement_cost_unit: String,
    available_budget: f64,
    states: Vec<State>,
    actions: Vec<Action>,
    measurements: Vec<Measurement>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct State {
    state_id: String,
    prior_probability: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Action {
    action_id: String,
    losses: Vec<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Measurement {
    measurement_id: String,
    cost: f64,
    outcomes: Vec<Outcome>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Outcome {
    outcome_id: String,
    likelihood_by_state: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OutcomeReport {
    pub outcome_id: String,
    pub probability: f64,
    pub posterior_by_state: Vec<f64>,
    pub best_action: Option<String>,
    pub posterior_expected_loss: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MeasurementReport {
    pub measurement_id: String,
    pub cost: f64,
    pub affordable: bool,
    pub expected_loss_after_measurement: f64,
    pub expected_value_of_sample_information: f64,
    pub net_expected_value: f64,
    pub outcomes: Vec<OutcomeReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DecisionReport {
    pub decision_id: String,
    pub evidence_class: String,
    pub claim_ceiling: String,
    pub decision_loss_unit: String,
    pub available_budget: f64,
    pub prior_expected_loss: f64,
    pub best_action_before_measurement: String,
    pub disposition: String,
    pub selected_measurement: Option<String>,
    pub measurements: Vec<MeasurementReport>,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn close_enough(left: f64, right: f64) -> bool {
    (left - right).abs() <= PROBABILITY_TOLERANCE
}

fn validate_probability_vector(values: &[f64], context: &str) -> Result<(), String> {
    if values.is_empty() {
        return Err(format!("{context}: probability vector is empty"));
    }
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0 || *value > 1.0)
    {
        return Err(format!(
            "{context}: probabilities must be finite and in [0, 1]"
        ));
    }
    let sum = values.iter().sum::<f64>();
    if !close_enough(sum, 1.0) {
        return Err(format!("{context}: probabilities sum to {sum}, not 1"));
    }
    Ok(())
}

fn best_action(actions: &[Action], state_probabilities: &[f64]) -> (String, f64) {
    actions
        .iter()
        .map(|action| {
            let expected_loss = action
                .losses
                .iter()
                .zip(state_probabilities.iter())
                .map(|(loss, probability)| loss * probability)
                .sum::<f64>();
            (action.action_id.clone(), expected_loss)
        })
        .min_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        })
        .expect("validated manifests contain at least one action")
}

fn evaluate_measurement(
    measurement: &Measurement,
    states: &[State],
    actions: &[Action],
    prior_expected_loss: f64,
    budget: f64,
) -> MeasurementReport {
    let mut expected_loss_after_measurement = 0.0;
    let mut outcomes = Vec::with_capacity(measurement.outcomes.len());

    for outcome in &measurement.outcomes {
        let probability = states
            .iter()
            .zip(outcome.likelihood_by_state.iter())
            .map(|(state, likelihood)| state.prior_probability * likelihood)
            .sum::<f64>();

        if probability <= f64::EPSILON {
            outcomes.push(OutcomeReport {
                outcome_id: outcome.outcome_id.clone(),
                probability: 0.0,
                posterior_by_state: vec![0.0; states.len()],
                best_action: None,
                posterior_expected_loss: None,
            });
            continue;
        }

        let posterior_by_state = states
            .iter()
            .zip(outcome.likelihood_by_state.iter())
            .map(|(state, likelihood)| state.prior_probability * likelihood / probability)
            .collect::<Vec<_>>();
        let posterior_sum = posterior_by_state.iter().sum::<f64>();
        let normalized_posterior = posterior_by_state
            .iter()
            .map(|value| value / posterior_sum)
            .collect::<Vec<_>>();
        let (selected_action, posterior_expected_loss) =
            best_action(actions, &normalized_posterior);

        expected_loss_after_measurement += probability * posterior_expected_loss;
        outcomes.push(OutcomeReport {
            outcome_id: outcome.outcome_id.clone(),
            probability,
            posterior_by_state: normalized_posterior,
            best_action: Some(selected_action),
            posterior_expected_loss: Some(posterior_expected_loss),
        });
    }

    // Information cannot increase the optimal expected loss. Clamp only
    // floating-point noise at zero; all inputs were validated above.
    let expected_value_of_sample_information =
        (prior_expected_loss - expected_loss_after_measurement).max(0.0);
    let net_expected_value = expected_value_of_sample_information - measurement.cost;

    MeasurementReport {
        measurement_id: measurement.measurement_id.clone(),
        cost: measurement.cost,
        affordable: measurement.cost <= budget,
        expected_loss_after_measurement,
        expected_value_of_sample_information,
        net_expected_value,
        outcomes,
    }
}

pub fn evaluate_manifest_json(source: &str) -> Result<DecisionReport, String> {
    let manifest: Manifest =
        serde_json::from_str(source).map_err(|error| format!("invalid EVSI manifest: {error}"))?;

    if manifest.schema_version != "luminous.plasma-experiment-voi.v0" {
        return Err("unsupported EVSI schema version".to_owned());
    }
    if !nonempty(&manifest.decision_id) || manifest.evidence_class != "SYNTHETIC" {
        return Err("v0 EVSI evaluator accepts synthetic manifests only".to_owned());
    }
    if !nonempty(&manifest.decision_loss_unit)
        || manifest.decision_loss_unit != manifest.measurement_cost_unit
    {
        return Err(
            "decision loss and measurement cost must use the same declared unit".to_owned(),
        );
    }
    if !manifest.available_budget.is_finite() || manifest.available_budget < 0.0 {
        return Err("available budget must be finite and non-negative".to_owned());
    }
    if manifest.states.len() < 2 || manifest.actions.len() < 2 || manifest.measurements.is_empty() {
        return Err("at least two states, two actions and one measurement are required".to_owned());
    }

    let mut state_ids = BTreeSet::new();
    for state in &manifest.states {
        if !nonempty(&state.state_id) || !state_ids.insert(state.state_id.as_str()) {
            return Err("state IDs must be unique and non-empty".to_owned());
        }
    }
    let priors = manifest
        .states
        .iter()
        .map(|state| state.prior_probability)
        .collect::<Vec<_>>();
    validate_probability_vector(&priors, "prior")?;

    let mut action_ids = BTreeSet::new();
    for action in &manifest.actions {
        if !nonempty(&action.action_id) || !action_ids.insert(action.action_id.as_str()) {
            return Err("action IDs must be unique and non-empty".to_owned());
        }
        if action.losses.len() != manifest.states.len()
            || action
                .losses
                .iter()
                .any(|loss| !loss.is_finite() || *loss < 0.0)
        {
            return Err(format!(
                "action {} must have one finite non-negative loss per state",
                action.action_id
            ));
        }
    }

    let mut measurement_ids = BTreeSet::new();
    for measurement in &manifest.measurements {
        if !nonempty(&measurement.measurement_id)
            || !measurement_ids.insert(measurement.measurement_id.as_str())
        {
            return Err("measurement IDs must be unique and non-empty".to_owned());
        }
        if !measurement.cost.is_finite() || measurement.cost < 0.0 {
            return Err(format!(
                "measurement {} has an invalid cost",
                measurement.measurement_id
            ));
        }
        if measurement.outcomes.len() < 2 {
            return Err(format!(
                "measurement {} must declare at least two possible outcomes",
                measurement.measurement_id
            ));
        }
        let mut outcome_ids = BTreeSet::new();
        let mut likelihood_sums = vec![0.0; manifest.states.len()];
        for outcome in &measurement.outcomes {
            if !nonempty(&outcome.outcome_id) || !outcome_ids.insert(outcome.outcome_id.as_str()) {
                return Err(format!(
                    "measurement {} has empty or duplicate outcome IDs",
                    measurement.measurement_id
                ));
            }
            if outcome.likelihood_by_state.len() != manifest.states.len()
                || outcome.likelihood_by_state.iter().any(|likelihood| {
                    !likelihood.is_finite() || *likelihood < 0.0 || *likelihood > 1.0
                })
            {
                return Err(format!(
                    "measurement {} outcome {} must have one valid likelihood per state",
                    measurement.measurement_id, outcome.outcome_id
                ));
            }
            for (index, likelihood) in outcome.likelihood_by_state.iter().enumerate() {
                likelihood_sums[index] += likelihood;
            }
        }
        for (index, sum) in likelihood_sums.iter().enumerate() {
            if !close_enough(*sum, 1.0) {
                return Err(format!(
                    "measurement {} likelihoods for state {} sum to {}, not 1",
                    measurement.measurement_id, manifest.states[index].state_id, sum
                ));
            }
        }
    }

    let (best_action_before_measurement, prior_expected_loss) =
        best_action(&manifest.actions, &priors);

    let measurements = manifest
        .measurements
        .iter()
        .map(|measurement| {
            evaluate_measurement(
                measurement,
                &manifest.states,
                &manifest.actions,
                prior_expected_loss,
                manifest.available_budget,
            )
        })
        .collect::<Vec<_>>();

    let selected_measurement = measurements
        .iter()
        .filter(|measurement| measurement.affordable && measurement.net_expected_value > 0.0)
        .max_by(|left, right| {
            left.net_expected_value
                .total_cmp(&right.net_expected_value)
                .then_with(|| right.measurement_id.cmp(&left.measurement_id))
        })
        .map(|measurement| measurement.measurement_id.clone());

    let disposition = if selected_measurement.is_some() {
        "SELECT_MEASUREMENT"
    } else {
        "ABSTAIN_NO_POSITIVE_NET_EVSI"
    };

    Ok(DecisionReport {
        decision_id: manifest.decision_id,
        evidence_class: manifest.evidence_class,
        claim_ceiling: "SYNTHETIC_ONLY_NO_PHYSICAL_MEASUREMENT_OR_ROUTE_SUPERIORITY".to_owned(),
        decision_loss_unit: manifest.decision_loss_unit,
        available_budget: manifest.available_budget,
        prior_expected_loss,
        best_action_before_measurement,
        disposition: disposition.to_owned(),
        selected_measurement,
        measurements,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const FIXTURE: &str =
        include_str!("../plasma-audit-harness/fixtures/plasma-voi-synthetic-v0.json");

    fn fixture_value() -> Value {
        serde_json::from_str(FIXTURE).expect("synthetic EVSI fixture must parse")
    }

    #[test]
    fn selects_highest_positive_net_evsi_within_budget() {
        let report = evaluate_manifest_json(FIXTURE).expect("fixture should evaluate");
        assert_eq!(report.disposition, "SELECT_MEASUREMENT");
        assert_eq!(
            report.selected_measurement.as_deref(),
            Some("high_information_measurement")
        );
        assert!((report.prior_expected_loss - 0.5).abs() < 1.0e-9);
        let best = report
            .measurements
            .iter()
            .find(|item| item.measurement_id == "high_information_measurement")
            .unwrap();
        assert!((best.expected_loss_after_measurement - 0.1).abs() < 1.0e-9);
        assert!((best.expected_value_of_sample_information - 0.4).abs() < 1.0e-9);
        assert!((best.net_expected_value - 0.35).abs() < 1.0e-9);
    }

    #[test]
    fn rejects_measurement_when_budget_is_too_small() {
        let mut value = fixture_value();
        value["available_budget"] = serde_json::json!(0.01);
        let report = evaluate_manifest_json(&value.to_string()).unwrap();
        assert_eq!(report.disposition, "ABSTAIN_NO_POSITIVE_NET_EVSI");
        assert_eq!(report.selected_measurement, None);
    }

    #[test]
    fn abstains_when_all_measurements_have_negative_net_value() {
        let mut value = fixture_value();
        for measurement in value["measurements"].as_array_mut().unwrap() {
            measurement["cost"] = serde_json::json!(0.6);
        }
        let report = evaluate_manifest_json(&value.to_string()).unwrap();
        assert_eq!(report.disposition, "ABSTAIN_NO_POSITIVE_NET_EVSI");
    }

    #[test]
    fn rejects_prior_probabilities_that_do_not_sum_to_one() {
        let mut value = fixture_value();
        value["states"][0]["prior_probability"] = serde_json::json!(0.7);
        assert!(
            evaluate_manifest_json(&value.to_string())
                .unwrap_err()
                .contains("prior")
        );
    }

    #[test]
    fn rejects_measurement_likelihoods_that_do_not_sum_to_one_per_state() {
        let mut value = fixture_value();
        value["measurements"][0]["outcomes"][0]["likelihood_by_state"][0] = serde_json::json!(0.7);
        assert!(
            evaluate_manifest_json(&value.to_string())
                .unwrap_err()
                .contains("likelihoods for state")
        );
    }

    #[test]
    fn rejects_outcome_likelihood_dimension_mismatch() {
        let mut value = fixture_value();
        value["measurements"][0]["outcomes"][0]["likelihood_by_state"] = serde_json::json!([0.9]);
        assert!(
            evaluate_manifest_json(&value.to_string())
                .unwrap_err()
                .contains("one valid likelihood per state")
        );
    }

    #[test]
    fn rejects_mismatched_decision_and_measurement_cost_units() {
        let mut value = fixture_value();
        value["measurement_cost_unit"] = serde_json::json!("EUR");
        assert!(
            evaluate_manifest_json(&value.to_string())
                .unwrap_err()
                .contains("same declared unit")
        );
    }

    #[test]
    fn no_information_has_zero_evsi_even_when_free() {
        let mut value = fixture_value();
        value["measurements"] = serde_json::json!([{
            "measurement_id": "uninformative",
            "cost": 0.0,
            "outcomes": [
                {"outcome_id": "a", "likelihood_by_state": [0.5, 0.5]},
                {"outcome_id": "b", "likelihood_by_state": [0.5, 0.5]}
            ]
        }]);
        let report = evaluate_manifest_json(&value.to_string()).unwrap();
        assert_eq!(report.disposition, "ABSTAIN_NO_POSITIVE_NET_EVSI");
        assert_eq!(
            report.measurements[0].expected_value_of_sample_information,
            0.0
        );
    }
}
