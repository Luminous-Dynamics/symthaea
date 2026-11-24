#![forbid(unsafe_code)]

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

use symthaea_semiconductor::benchmark::{
    METROLOGY_ASSAY_MODEL_SCHEMA_V1, METROLOGY_DECISION_LOSS_POLICY_SCHEMA_V1,
    METROLOGY_READINESS_SCHEMA_V1, ROBUST_EVSI_CANDIDATE_SCHEMA_V1,
    ROBUST_EVSI_INPUT_SCHEMA_V1, ROBUST_EVSI_SCENARIO_SCHEMA_V1, MetrologyAssayModelV1,
    MetrologyAssayOutcomeV1, MetrologyCandidateV1, MetrologyDecisionActionV1,
    MetrologyDecisionLossModelV1, MetrologyDecisionLossPolicyV1, MetrologyDecisionStateV1,
    MetrologyReadinessEvidenceV1, MetrologyReadinessStatusV1, RobustEvsiCandidateV1,
    RobustEvsiInputV1, RobustEvsiPortfolioStatusV1, RobustEvsiScenarioV1,
    optimize_robust_metrology_evsi_portfolio_v1, seal_robust_evsi_candidate_v1,
    seal_robust_evsi_scenario_v1,
};

fn digest(ch: char) -> String {
    std::iter::repeat_n(ch, 64).collect()
}

fn hash_json<T: Serialize>(value: &T) -> String {
    format!("{:x}", Sha256::digest(serde_json::to_vec(value).unwrap()))
}

fn model(quality: f64, prior_defect: f64) -> MetrologyDecisionLossModelV1 {
    MetrologyDecisionLossModelV1 {
        model_id: "independent-two-state-model".into(),
        revision: 1,
        states: vec![
            MetrologyDecisionStateV1 {
                state_id: "good".into(),
                prior_probability: 1.0 - prior_defect,
            },
            MetrologyDecisionStateV1 {
                state_id: "defective".into(),
                prior_probability: prior_defect,
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
            model_id: "independent-assay".into(),
            revision: 1,
            outcomes: vec![
                MetrologyAssayOutcomeV1 {
                    outcome_id: "pass".into(),
                    likelihood_by_state: BTreeMap::from([
                        ("good".into(), 0.98),
                        ("defective".into(), 1.0 - quality),
                    ]),
                },
                MetrologyAssayOutcomeV1 {
                    outcome_id: "fail".into(),
                    likelihood_by_state: BTreeMap::from([
                        ("good".into(), 0.02),
                        ("defective".into(), quality),
                    ]),
                },
            ],
        },
    }
}

fn reference_bayes_risk(model: &MetrologyDecisionLossModelV1) -> (f64, f64) {
    let expected_loss = |posterior: &BTreeMap<String, f64>| {
        model
            .actions
            .iter()
            .map(|action| {
                model
                    .states
                    .iter()
                    .map(|state| {
                        posterior[&state.state_id] * action.loss_by_state[&state.state_id]
                    })
                    .sum::<f64>()
            })
            .fold(f64::INFINITY, f64::min)
    };
    let prior = model
        .states
        .iter()
        .map(|state| (state.state_id.clone(), state.prior_probability))
        .collect::<BTreeMap<_, _>>();
    let before = expected_loss(&prior);
    let mut after = 0.0;
    for outcome in &model.assay.outcomes {
        let outcome_probability = model
            .states
            .iter()
            .map(|state| {
                state.prior_probability * outcome.likelihood_by_state[&state.state_id]
            })
            .sum::<f64>();
        if outcome_probability <= 0.0 {
            continue;
        }
        let posterior = model
            .states
            .iter()
            .map(|state| {
                (
                    state.state_id.clone(),
                    state.prior_probability
                        * outcome.likelihood_by_state[&state.state_id]
                        / outcome_probability,
                )
            })
            .collect::<BTreeMap<_, _>>();
        after += outcome_probability * expected_loss(&posterior);
    }
    (before, after)
}

fn readiness() -> MetrologyReadinessEvidenceV1 {
    let mut value = MetrologyReadinessEvidenceV1 {
        schema: METROLOGY_READINESS_SCHEMA_V1.into(),
        status: MetrologyReadinessStatusV1::Qualified,
        instrument_id: "tool-A".into(),
        measurand_id: "critical-dimension".into(),
        unit: "nm".into(),
        certificate_id: "oracle-certificate".into(),
        certificate_sha256: digest('a'),
        calibration_record_sha256: digest('b'),
        measurement_system_profile_sha256: digest('c'),
        valid_from_unix_seconds: 0,
        valid_until_unix_seconds: 10_000,
        revoked_at_unix_seconds: None,
        expected_standard_uncertainty_upper_bound: 0.02,
        report_sha256: String::new(),
    };
    let mut canonical = value.clone();
    canonical.report_sha256.clear();
    value.report_sha256 = hash_json(&canonical);
    value
}

fn scenario(
    id: &str,
    weight: Option<f64>,
    measurement_cost: f64,
    delay_cost_per_second: f64,
    quality: f64,
    prior_defect: f64,
) -> RobustEvsiScenarioV1 {
    let mut value = RobustEvsiScenarioV1 {
        schema: ROBUST_EVSI_SCENARIO_SCHEMA_V1.into(),
        scenario_id: id.into(),
        scenario_weight: weight,
        measurement_cost,
        delay_cost_per_second,
        decision_loss_model: model(quality, prior_defect),
        source_provenance_sha256: digest('d'),
        scenario_sha256: String::new(),
    };
    value.scenario_sha256 = seal_robust_evsi_scenario_v1(&value).unwrap();
    value
}

fn candidate(id: &str, group: &str, scenarios: Vec<RobustEvsiScenarioV1>) -> RobustEvsiCandidateV1 {
    let mut value = RobustEvsiCandidateV1 {
        schema: ROBUST_EVSI_CANDIDATE_SCHEMA_V1.into(),
        candidate: MetrologyCandidateV1 {
            schema: "luminous.semi-bench.metrology-candidate.v1".into(),
            candidate_id: id.into(),
            measurement_event_id: format!("measurement-{id}"),
            group_key: group.into(),
            instrument_id: "tool-A".into(),
            measurand_id: "critical-dimension".into(),
            unit: "nm".into(),
            planned_measurement_at_unix_seconds: 100,
            decision_deadline_unix_seconds: 2_000,
            turnaround_seconds: 20,
            measurement_cost: 0.1,
            measurement_cost_units: 1,
            readiness: readiness(),
            decision_loss_model: model(0.95, 0.10),
        },
        central_scenario_id: "central".into(),
        scenarios,
        scenario_set_complete: true,
        scenario_set_sha256: String::new(),
    };
    value.scenario_set_sha256 = seal_robust_evsi_candidate_v1(&value).unwrap();
    value
}

fn policy(budget: u32) -> MetrologyDecisionLossPolicyV1 {
    MetrologyDecisionLossPolicyV1 {
        schema: METROLOGY_DECISION_LOSS_POLICY_SCHEMA_V1.into(),
        policy_id: "independent-robust-evsi-oracle".into(),
        revision: 1,
        measurement_budget_units: budget,
        delay_cost_per_second: 0.01,
        minimum_group_coverage: BTreeMap::new(),
        maximum_exact_candidates: 20,
    }
}

fn input(candidates: Vec<RobustEvsiCandidateV1>, budget: u32) -> RobustEvsiInputV1 {
    RobustEvsiInputV1 {
        schema: ROBUST_EVSI_INPUT_SCHEMA_V1.into(),
        input_id: "independent-robust-portfolio".into(),
        base_policy: policy(budget),
        maximum_scenarios_per_candidate: 8,
        uncertainty_set_complete: true,
        candidates,
    }
}

fn independent_robust_value(candidate: &RobustEvsiCandidateV1) -> (String, f64) {
    candidate
        .scenarios
        .iter()
        .map(|scenario| {
            let (before, after) = reference_bayes_risk(&scenario.decision_loss_model);
            let value = before - after
                - scenario.measurement_cost
                - scenario.delay_cost_per_second
                    * candidate.candidate.turnaround_seconds as f64;
            (scenario.scenario_id.clone(), value)
        })
        .min_by(|left, right| left.1.total_cmp(&right.1).then_with(|| left.0.cmp(&right.0)))
        .unwrap()
}

fn independent_portfolio(
    candidates: &[RobustEvsiCandidateV1],
    budget: u32,
    quotas: &BTreeMap<String, u32>,
) -> (Vec<String>, f64) {
    let mut ordered = candidates.to_vec();
    ordered.sort_by(|left, right| {
        left.candidate.candidate_id.cmp(&right.candidate.candidate_id)
    });
    let values = ordered
        .iter()
        .map(|candidate| independent_robust_value(candidate).1)
        .collect::<Vec<_>>();
    let mut best_ids = Vec::new();
    let mut best_value = f64::NEG_INFINITY;
    let mut found = false;

    for mask in 0_u64..(1_u64 << ordered.len()) {
        let mut ids = Vec::new();
        let mut value = 0.0;
        let mut used = 0_u32;
        let mut groups = BTreeMap::new();
        let mut feasible = true;
        for (index, candidate) in ordered.iter().enumerate() {
            if (mask & (1_u64 << index)) == 0 {
                continue;
            }
            if values[index] <= 0.0 {
                feasible = false;
                break;
            }
            used += candidate.candidate.measurement_cost_units;
            if used > budget {
                feasible = false;
                break;
            }
            value += values[index];
            ids.push(candidate.candidate.candidate_id.clone());
            *groups.entry(candidate.candidate.group_key.clone()).or_insert(0_u32) += 1;
        }
        if !feasible || quotas.iter().any(|(group, minimum)| {
            groups.get(group).copied().unwrap_or(0) < *minimum
        }) {
            continue;
        }
        if !found
            || value > best_value + 1.0e-10
            || ((value - best_value).abs() <= 1.0e-10 && ids < best_ids)
        {
            found = true;
            best_ids = ids;
            best_value = value;
        }
    }
    assert!(found, "reference fixture should have a feasible robust portfolio");
    (best_ids, best_value)
}

#[test]
fn robust_candidate_minimum_matches_independent_scenario_oracle() {
    let value = candidate(
        "a",
        "chamber-1",
        vec![
            scenario("central", Some(0.6), 0.1, 0.01, 0.95, 0.10),
            scenario("lower-prior-low-assay", Some(0.2), 0.3, 0.02, 0.75, 0.03),
            scenario("upper-prior-high-assay", Some(0.2), 0.2, 0.01, 0.99, 0.20),
        ],
    );
    let expected = independent_robust_value(&value);
    let report = optimize_robust_metrology_evsi_portfolio_v1(&input(vec![value], 1)).unwrap();
    let candidate_report = &report.candidate_reports[0];
    assert_eq!(candidate_report.worst_case_scenario_id.as_deref(), Some(expected.0.as_str()));
    assert!(
        (candidate_report.worst_case_net_expected_loss_reduction.unwrap() - expected.1).abs()
            < 1.0e-10
    );
}

#[test]
fn robust_portfolio_matches_independent_exhaustive_subset_oracle() {
    let candidates = vec![
        candidate(
            "a",
            "chamber-1",
            vec![
                scenario("central", None, 0.1, 0.01, 0.95, 0.10),
                scenario("stress", None, 0.2, 0.02, 0.85, 0.15),
            ],
        ),
        candidate(
            "b",
            "chamber-2",
            vec![
                scenario("central", None, 0.1, 0.01, 0.95, 0.10),
                scenario("stress", None, 0.2, 0.02, 0.85, 0.15),
            ],
        ),
        candidate(
            "c",
            "chamber-1",
            vec![
                scenario("central", None, 0.1, 0.01, 0.95, 0.10),
                scenario("stress", None, 0.3, 0.02, 0.70, 0.20),
            ],
        ),
        candidate(
            "d",
            "chamber-2",
            vec![
                scenario("central", None, 0.1, 0.01, 0.90, 0.10),
                scenario("stress", None, 0.25, 0.02, 0.75, 0.15),
            ],
        ),
    ];
    let quotas = BTreeMap::from([
        ("chamber-1".into(), 1),
        ("chamber-2".into(), 1),
    ]);
    let expected = independent_portfolio(&candidates, 3, &quotas);
    let mut portfolio_input = input(candidates.clone(), 3);
    portfolio_input.base_policy.minimum_group_coverage = quotas;
    let report = optimize_robust_metrology_evsi_portfolio_v1(&portfolio_input).unwrap();
    assert_eq!(report.status, RobustEvsiPortfolioStatusV1::Feasible);
    assert_eq!(report.selected_candidate_ids, expected.0);
    assert!((report.worst_case_expected_net_loss_reduction - expected.1).abs() < 1.0e-10);
}
