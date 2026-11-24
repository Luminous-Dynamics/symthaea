#![forbid(unsafe_code)]

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

use symthaea_semiconductor::benchmark::{
    METROLOGY_ASSAY_MODEL_SCHEMA_V1, METROLOGY_CANDIDATE_SCHEMA_V1,
    METROLOGY_DECISION_LOSS_INPUT_SCHEMA_V1, METROLOGY_DECISION_LOSS_POLICY_SCHEMA_V1,
    METROLOGY_READINESS_SCHEMA_V1, MetrologyAssayModelV1, MetrologyAssayOutcomeV1,
    MetrologyCandidateV1, MetrologyDecisionActionV1, MetrologyDecisionLossInputV1,
    MetrologyDecisionLossModelV1, MetrologyDecisionLossPolicyV1, MetrologyDecisionStateV1,
    MetrologyReadinessEvidenceV1, MetrologyReadinessStatusV1,
    compute_expected_value_of_sample_information_v1, optimize_metrology_decision_loss_portfolio_v1,
};

fn hash_json<T: Serialize>(value: &T) -> String {
    format!("{:x}", Sha256::digest(serde_json::to_vec(value).unwrap()))
}

fn fixture(evidence_quality: f64) -> MetrologyDecisionLossModelV1 {
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
                loss_by_state: BTreeMap::from([("good".into(), 0.0), ("defective".into(), 100.0)]),
            },
            MetrologyDecisionActionV1 {
                action_id: "hold".into(),
                loss_by_state: BTreeMap::from([("good".into(), 5.0), ("defective".into(), 0.0)]),
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
                        ("defective".into(), 1.0 - evidence_quality),
                    ]),
                },
                MetrologyAssayOutcomeV1 {
                    outcome_id: "fail".into(),
                    likelihood_by_state: BTreeMap::from([
                        ("good".into(), 0.02),
                        ("defective".into(), evidence_quality),
                    ]),
                },
            ],
        },
    }
}

/// A separately written posterior-risk enumeration. It deliberately does not
/// call the production expected-optimal-loss helper.
fn reference_evsi(model: &MetrologyDecisionLossModelV1) -> (f64, f64) {
    let prior: BTreeMap<String, f64> = model
        .states
        .iter()
        .map(|state| (state.state_id.clone(), state.prior_probability))
        .collect();

    let expected_loss = |probabilities: &BTreeMap<String, f64>| {
        model
            .actions
            .iter()
            .map(|action| {
                model
                    .states
                    .iter()
                    .map(|state| {
                        probabilities[&state.state_id] * action.loss_by_state[&state.state_id]
                    })
                    .sum::<f64>()
            })
            .fold(f64::INFINITY, f64::min)
    };

    let risk_before = expected_loss(&prior);
    let mut risk_after = 0.0;
    for outcome in &model.assay.outcomes {
        let probability = model
            .states
            .iter()
            .map(|state| state.prior_probability * outcome.likelihood_by_state[&state.state_id])
            .sum::<f64>();
        if probability == 0.0 {
            continue;
        }
        let posterior = model
            .states
            .iter()
            .map(|state| {
                (
                    state.state_id.clone(),
                    state.prior_probability * outcome.likelihood_by_state[&state.state_id]
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
    let model = fixture(0.95);
    let expected = reference_evsi(&model);
    let actual = compute_expected_value_of_sample_information_v1(&model).unwrap();
    assert!((expected.0 - actual.0).abs() < 1.0e-12);
    assert!((expected.1 - actual.1).abs() < 1.0e-12);
    assert!(actual.1 < actual.0);
}

#[test]
fn permuting_assay_outcomes_does_not_change_expected_decision_risk() {
    let mut model = fixture(0.95);
    let original = compute_expected_value_of_sample_information_v1(&model).unwrap();
    model.assay.outcomes.reverse();
    let reversed = compute_expected_value_of_sample_information_v1(&model).unwrap();
    assert!((original.0 - reversed.0).abs() < 1.0e-12);
    assert!((original.1 - reversed.1).abs() < 1.0e-12);
}

fn digest(ch: char) -> String {
    std::iter::repeat_n(ch, 64).collect()
}

fn readiness(instrument: &str, measurand: &str, unit: &str) -> MetrologyReadinessEvidenceV1 {
    let mut value = MetrologyReadinessEvidenceV1 {
        schema: METROLOGY_READINESS_SCHEMA_V1.into(),
        status: MetrologyReadinessStatusV1::Qualified,
        instrument_id: instrument.into(),
        measurand_id: measurand.into(),
        unit: unit.into(),
        certificate_id: "oracle-certificate".into(),
        certificate_sha256: digest('a'),
        calibration_record_sha256: digest('b'),
        measurement_system_profile_sha256: digest('c'),
        valid_from_unix_seconds: 0,
        valid_until_unix_seconds: 10_000,
        revoked_at_unix_seconds: None,
        expected_standard_uncertainty_upper_bound: 0.01,
        report_sha256: String::new(),
    };
    let mut canonical = value.clone();
    canonical.report_sha256.clear();
    value.report_sha256 = hash_json(&canonical);
    value
}

fn portfolio_candidate(
    id: &str,
    group: &str,
    cost_units: u32,
    measurement_cost: f64,
    turnaround: u64,
    quality: f64,
) -> MetrologyCandidateV1 {
    MetrologyCandidateV1 {
        schema: METROLOGY_CANDIDATE_SCHEMA_V1.into(),
        candidate_id: id.into(),
        measurement_event_id: format!("event-{id}"),
        group_key: group.into(),
        instrument_id: "tool-A".into(),
        measurand_id: "film-thickness".into(),
        unit: "nm".into(),
        planned_measurement_at_unix_seconds: 100,
        decision_deadline_unix_seconds: 1_000,
        turnaround_seconds: turnaround,
        measurement_cost,
        measurement_cost_units: cost_units,
        readiness: readiness("tool-A", "film-thickness", "nm"),
        decision_loss_model: fixture(quality),
    }
}

fn independent_portfolio_oracle(
    candidates: &[MetrologyCandidateV1],
    policy: &MetrologyDecisionLossPolicyV1,
) -> (Vec<String>, f64, u32, BTreeMap<String, u32>) {
    let mut ordered = candidates.to_vec();
    ordered.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));

    let nets: Vec<f64> = ordered
        .iter()
        .map(|candidate| {
            let (before, after) = reference_evsi(&candidate.decision_loss_model);
            before
                - after
                - candidate.measurement_cost
                - policy.delay_cost_per_second * candidate.turnaround_seconds as f64
        })
        .collect();

    let mut best_ids = Vec::new();
    let mut best_value = f64::NEG_INFINITY;
    let mut best_cost = 0_u32;
    let mut best_groups = BTreeMap::new();
    let mut found = false;

    for mask in 0_u64..(1_u64 << ordered.len()) {
        let mut ids = Vec::new();
        let mut value = 0.0;
        let mut cost = 0_u32;
        let mut groups: BTreeMap<String, u32> = BTreeMap::new();
        let mut feasible = true;

        for (index, candidate) in ordered.iter().enumerate() {
            if (mask & (1_u64 << index)) == 0 {
                continue;
            }
            if nets[index] <= 0.0 {
                feasible = false;
                break;
            }
            cost = match cost.checked_add(candidate.measurement_cost_units) {
                Some(total) if total <= policy.measurement_budget_units => total,
                _ => {
                    feasible = false;
                    break;
                }
            };
            value += nets[index];
            ids.push(candidate.candidate_id.clone());
            *groups.entry(candidate.group_key.clone()).or_default() += 1;
        }

        if !feasible
            || policy
                .minimum_group_coverage
                .iter()
                .any(|(group, minimum)| groups.get(group).copied().unwrap_or(0) < *minimum)
        {
            continue;
        }

        if !found
            || value > best_value + 1.0e-10
            || ((value - best_value).abs() <= 1.0e-10 && ids < best_ids)
        {
            found = true;
            best_ids = ids;
            best_value = value;
            best_cost = cost;
            best_groups = groups;
        }
    }

    assert!(
        found,
        "oracle fixture must have at least one feasible portfolio"
    );
    (best_ids, best_value, best_cost, best_groups)
}

#[test]
fn exact_portfolio_optimizer_matches_independent_exhaustive_subset_oracle() {
    let candidates = vec![
        portfolio_candidate("d", "chamber-2", 2, 0.20, 10, 0.80),
        portfolio_candidate("b", "chamber-1", 2, 0.15, 10, 0.90),
        portfolio_candidate("c", "chamber-2", 1, 0.10, 10, 0.95),
        portfolio_candidate("a", "chamber-1", 1, 0.10, 10, 0.95),
    ];
    let policy = MetrologyDecisionLossPolicyV1 {
        schema: METROLOGY_DECISION_LOSS_POLICY_SCHEMA_V1.into(),
        policy_id: "independent-portfolio-oracle".into(),
        revision: 1,
        measurement_budget_units: 3,
        delay_cost_per_second: 0.01,
        minimum_group_coverage: BTreeMap::from([("chamber-1".into(), 1), ("chamber-2".into(), 1)]),
        maximum_exact_candidates: 20,
    };
    let input = MetrologyDecisionLossInputV1 {
        schema: METROLOGY_DECISION_LOSS_INPUT_SCHEMA_V1.into(),
        input_id: "independent-portfolio-fixture".into(),
        policy: policy.clone(),
        candidates: candidates.clone(),
    };

    let expected = independent_portfolio_oracle(&candidates, &policy);
    let actual = optimize_metrology_decision_loss_portfolio_v1(&input).unwrap();

    assert_eq!(actual.selected_candidate_ids, expected.0);
    assert!((actual.expected_net_loss_reduction - expected.1).abs() < 1.0e-10);
    assert_eq!(actual.total_measurement_cost_units, expected.2);
    assert_eq!(actual.selected_group_counts, expected.3);
}
