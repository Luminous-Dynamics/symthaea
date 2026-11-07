// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Independent tiny reference cases for online categorical likelihood-ratio drift monitoring.

use symthaea_operations_research::aviation_online_drift::{
    DriftAlternative, DriftEvidence, DriftMonitorStatus, DriftScenarioProbability,
    OnlineDriftMonitor, PROBABILITY_SCALE_PPM,
};

fn probability(id: &str, ppm: u32) -> DriftScenarioProbability {
    DriftScenarioProbability {
        scenario_id: id.into(),
        probability_ppm: ppm,
    }
}

fn fresh_monitor(alpha: u32) -> OnlineDriftMonitor {
    OnlineDriftMonitor::new(
        vec![probability("routine", 900_000), probability("disruption", 100_000)],
        vec![
            DriftAlternative {
                alternative_id: "balanced-shift".into(),
                mixture_weight_ppm: 700_000,
                probabilities: vec![probability("routine", 500_000), probability("disruption", 500_000)],
            },
            DriftAlternative {
                alternative_id: "moderate-shift".into(),
                mixture_weight_ppm: 300_000,
                probabilities: vec![probability("routine", 800_000), probability("disruption", 200_000)],
            },
        ],
        alpha,
    ).unwrap()
}

#[test]
fn exact_one_step_null_expectation_of_mixture_e_value_is_one() {
    // Under the fixed nominal distribution p, E_p[q(X)/p(X)] = sum_x q(x) = 1.
    // The same remains true for any mixture of predeclared normalized alternatives.
    let mut common = fresh_monitor(50_000);
    let mut disrupted = fresh_monitor(50_000);
    let e_common = common.observe("common", 1, "routine").unwrap().log_e_value.exp();
    let e_disrupted = disrupted.observe("disrupted", 1, "disruption").unwrap().log_e_value.exp();
    let expected = 0.9 * e_common + 0.1 * e_disrupted;
    assert!((expected - 1.0).abs() < 1e-12);
}

#[test]
fn alternative_mixture_weights_change_the_e_value_by_exact_weighting() {
    let mut monitor = fresh_monitor(50_000);
    let e = monitor.observe("event", 1, "disruption").unwrap().log_e_value.exp();
    let expected = 0.7 * (0.5 / 0.1) + 0.3 * (0.2 / 0.1);
    assert!((e - expected).abs() < 1e-12);
}

#[test]
fn sequential_evidence_crosses_only_after_predeclared_boundary() {
    let mut monitor = fresh_monitor(100_000);
    assert_eq!(monitor.observe("e1", 1, "disruption").unwrap().status, DriftMonitorStatus::InControl);
    let report = monitor.observe("e2", 2, "disruption").unwrap();
    assert_eq!(report.status, DriftMonitorStatus::DriftAlarm);
    assert_eq!(report.evidence, DriftEvidence::MixtureLikelihoodRatio);
    assert!(report.log_e_value >= report.log_alarm_threshold);
}

#[test]
fn scenario_outside_nominal_support_is_a_distinct_immediate_alarm() {
    let mut monitor = fresh_monitor(50_000);
    let report = monitor.observe("e1", 1, "compound-unmodeled").unwrap();
    assert_eq!(report.status, DriftMonitorStatus::SupportShiftAlarm);
    assert_eq!(report.evidence, DriftEvidence::ScenarioOutsideNominalSupport);
}

#[test]
fn probability_scale_is_one_million_ppm() {
    assert_eq!(PROBABILITY_SCALE_PPM, 1_000_000);
}
