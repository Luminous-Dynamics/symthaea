#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use symthaea_semiconductor::benchmark::{
    METROLOGY_ASSAY_MODEL_SCHEMA_V1, MetrologyAssayModelV1, MetrologyAssayOutcomeV1,
    MetrologyDecisionActionV1, MetrologyDecisionLossModelV1, MetrologyDecisionStateV1,
    compute_expected_value_of_sample_information_v1,
};

fn fixture() -> MetrologyDecisionLossModelV1 {
    MetrologyDecisionLossModelV1 {
        model_id: "two-state-asymmetric-loss".into(),
        revision: 1,
        states: vec![
            MetrologyDecisionStateV1 {
                state_id: "good".into(),
                prior_probability: 0.90,
            },
            MetrologyDecisionStateV1 {
                state_id: "defective".into(),
                prior_probability: 0.10,
            },
        ],
        actions: vec![
            MetrologyDecisionActionV1 {
                action_id: "accept".into(),
                loss_by_state: BTreeMap::from([
                    ("good".into(), 0.0),
                    ("defective".into(), 100.0),
                ]),
            },
            MetrologyDecisionActionV1 {
                action_id: "hold".into(),
                loss_by_state: BTreeMap::from([
                    ("good".into(), 5.0),
                    ("defective".into(), 0.0),
                ]),
            },
        ],
        assay: MetrologyAssayModelV1 {
            schema: METROLOGY_ASSAY_MODEL_SCHEMA_V1.into(),
            model_id: "assay-with-95pct-sensitivity".into(),
            revision: 1,
            outcomes: vec![
                MetrologyAssayOutcomeV1 {
                    outcome_id: "pass".into(),
                    likelihood_by_state: BTreeMap::from([
                        ("good".into(), 0.98),
                        ("defective".into(), 0.05),
                    ]),
                },
                MetrologyAssayOutcomeV1 {
                    outcome_id: "fail".into(),
                    likelihood_by_state: BTreeMap::from([
                        ("good".into(), 0.02),
                        ("defective".into(), 0.95),
                    ]),
                },
            ],
        },
    }
}

/// A separately written posterior-risk enumeration. It deliberately does not
/// call the production expected-optimal-loss helper.
fn reference_evsi(model: &MetrologyDecisionLossModelV1) -> (f64, f64) {
    let prior: BTreeMap<String, f64> = model.states
        .iter()
        .map(|state| (state.state_id.clone(), state.prior_probability))
        .collect();

    let expected_loss = |probabilities: &BTreeMap<String, f64>| {
        model.actions
            .iter()
            .map(|action| {
                model.states
                    .iter()
                    .map(|state| probabilities[&state.state_id] * action.loss_by_state[&state.state_id])
                    .sum::<f64>()
            })
            .fold(f64::INFINITY, f64::min)
    };

    let risk_before = expected_loss(&prior);
    let mut risk_after = 0.0;
    for outcome in &model.assay.outcomes {
        let probability = model.states
            .iter()
            .map(|state| {
                state.prior_probability
                    * outcome.likelihood_by_state[&state.state_id]
            })
            .sum::<f64>();
        if probability == 0.0 {
            continue;
        }
        let posterior = model.states
            .iter()
            .map(|state| {
                (
                    state.state_id.clone(),
                    state.prior_probability
                        * outcome.likelihood_by_state[&state.state_id]
                        / probability,
                )
            })
            .collect::<BTreeMap<_, _>>();
        risk_after += probability * expected_loss(&posterior);
    }
    (risk_before, risk_after)
}

#[test]
fn evsi_matches_independent_posterior_decision_risk_oracle() {
    let model = fixture();
    let expected = reference_evsi(&model);
    let actual = compute_expected_value_of_sample_information_v1(&model).unwrap();
    assert!((expected.0 - actual.0).abs() < 1.0e-12);
    assert!((expected.1 - actual.1).abs() < 1.0e-12);
    assert!(actual.1 < actual.0);
}

#[test]
fn permuting_assay_outcomes_does_not_change_expected_decision_risk() {
    let mut model = fixture();
    let original = compute_expected_value_of_sample_information_v1(&model).unwrap();
    model.assay.outcomes.reverse();
    let reversed = compute_expected_value_of_sample_information_v1(&model).unwrap();
    assert!((original.0 - reversed.0).abs() < 1.0e-12);
    assert!((original.1 - reversed.1).abs() < 1.0e-12);
}
