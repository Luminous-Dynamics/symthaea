use std::collections::BTreeMap;
use symthaea_semiconductor::benchmark::{
    CAUSAL_PORTFOLIO_INPUT_SCHEMA_V1, CandidateOutcomeLikelihoodV1,
    CausalInformationPortfolioInputV1, CausalMeasurementCandidateV1,
    CausalPortfolioObjectiveWeightsV1, HypothesisPriorV1,
    optimize_causal_information_portfolio_v1,
};

fn digest(ch: char) -> String {
    std::iter::repeat_n(ch, 64).collect()
}

fn binary_candidate(
    id: &str,
    group: &str,
    p_alarm_h1: f64,
    p_alarm_h2: f64,
    cost: f64,
) -> CausalMeasurementCandidateV1 {
    CausalMeasurementCandidateV1 {
        candidate_id: id.into(),
        dependence_group_id: group.into(),
        physical_measurement_cost: cost,
        capacity_units: 1,
        expected_result_delay_seconds: 5,
        randomized_audit: false,
        audit_design_sha256: None,
        audit_inclusion_probability: None,
        calibration_qualified: true,
        source_path_qualified: true,
        safety_preconditions_satisfied: true,
        expected_predictive_risk_reduction: 0.0,
        likelihood_model_sha256: digest('a'),
        outcome_schema_sha256: digest('b'),
        outcomes: vec![
            CandidateOutcomeLikelihoodV1 {
                outcome_id: "alarm".into(),
                probability_by_hypothesis: BTreeMap::from([
                    ("h1".into(), p_alarm_h1),
                    ("h2".into(), p_alarm_h2),
                ]),
            },
            CandidateOutcomeLikelihoodV1 {
                outcome_id: "quiet".into(),
                probability_by_hypothesis: BTreeMap::from([
                    ("h1".into(), 1.0 - p_alarm_h1),
                    ("h2".into(), 1.0 - p_alarm_h2),
                ]),
            },
        ],
    }
}

fn input(
    candidates: Vec<CausalMeasurementCandidateV1>,
    budget: f64,
    independence_declared: bool,
) -> CausalInformationPortfolioInputV1 {
    CausalInformationPortfolioInputV1 {
        schema: CAUSAL_PORTFOLIO_INPUT_SCHEMA_V1.into(),
        portfolio_id: "independent-oracle-v1".into(),
        hypotheses: vec![
            HypothesisPriorV1 {
                hypothesis_id: "h1".into(),
                prior_probability: 0.5,
            },
            HypothesisPriorV1 {
                hypothesis_id: "h2".into(),
                prior_probability: 0.5,
            },
        ],
        decision_actions: Vec::new(),
        candidates,
        physical_cost_budget: budget,
        maximum_capacity_units: 2,
        minimum_randomized_audit_count: 0,
        conditional_independence_assumption_declared: independence_declared,
        conditional_independence_assumption_sha256: independence_declared.then(|| digest('c')),
        objective_weights: CausalPortfolioObjectiveWeightsV1 {
            information_gain_weight: 1.0,
            decision_loss_reduction_weight: 0.0,
            predictive_risk_reduction_weight: 0.0,
            physical_cost_weight: 0.0,
            delay_weight: 0.0,
        },
    }
}

fn binary_entropy(p: f64) -> f64 {
    let term = |x: f64| if x <= 0.0 { 0.0 } else { -x * x.log2() };
    term(p) + term(1.0 - p)
}

#[test]
fn exact_optimizer_matches_independent_exhaustive_two_candidate_oracle() {
    let value = input(
        vec![
            binary_candidate("strong", "sensor-a", 0.9, 0.1, 1.0),
            binary_candidate("weak", "sensor-b", 0.7, 0.3, 1.0),
        ],
        2.0,
        true,
    );
    let report = optimize_causal_information_portfolio_v1(&value).unwrap();
    let selected = report.selected_portfolio.as_ref().unwrap();

    // Independent 4-outcome enumeration for the two conditionally independent tests.
    let outcome_pairs = [
        (0.9, 0.7),
        (0.9, 0.3),
        (0.1, 0.7),
        (0.1, 0.3),
    ];
    let mut expected_posterior_entropy = 0.0;
    for (p_h1_a, p_h1_b) in outcome_pairs {
        let p_h2_a = if p_h1_a == 0.9 { 0.1 } else { 0.9 };
        let p_h2_b = if p_h1_b == 0.7 { 0.3 } else { 0.7 };
        let probability_h1 = 0.5 * p_h1_a * p_h1_b;
        let probability_h2 = 0.5 * p_h2_a * p_h2_b;
        let marginal = probability_h1 + probability_h2;
        let posterior_h1 = probability_h1 / marginal;
        expected_posterior_entropy += marginal * binary_entropy(posterior_h1);
    }
    let expected_gain = 1.0 - expected_posterior_entropy;
    assert!((selected.expected_hypothesis_information_gain_bits - expected_gain).abs() < 1.0e-12);
    assert_eq!(selected.candidate_ids, vec!["strong", "weak"]);
    assert!(report.conditional_independence_assumption_used);
    assert!(!report.source_likelihood_authenticity_proven);
    assert!(!report.physical_measurement_executed);
}

#[test]
fn no_independence_declaration_limits_portfolio_to_one_measurement() {
    let value = input(
        vec![
            binary_candidate("a", "sensor-a", 0.9, 0.1, 1.0),
            binary_candidate("b", "sensor-b", 0.8, 0.2, 1.0),
        ],
        2.0,
        false,
    );
    let report = optimize_causal_information_portfolio_v1(&value).unwrap();
    assert_eq!(report.selected_portfolio.unwrap().candidate_ids.len(), 1);
    assert!(!report.conditional_independence_assumption_used);
}
