// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Tiny-grid exhaustive oracle for distributionally robust aviation evaluation.
//!
//! This oracle intentionally enumerates feasible discrete distributions instead
//! of sharing the production evaluator's greedy probability-transfer algorithm.
//! It validates the TV-ball objective on tiny, hand-checkable cases only.

use symthaea_operations_research::aviation_distributional_robustness::{
    evaluate_tv_robust_cost_delta, ScenarioCostDelta,
};

const GRID_PPM: u32 = 100_000;

/// Enumerate all probability vectors on a 10-point grid and independently
/// maximize candidate-minus-baseline expected cost subject to the TV constraint.
fn brute_force_tv_max(
    scenarios: &[ScenarioCostDelta],
    radius_ppm: u32,
) -> Option<i128> {
    assert!(scenarios.len() <= 4, "oracle is only for tiny fixtures");
    assert_eq!(1_000_000 % GRID_PPM, 0);
    assert_eq!(radius_ppm % GRID_PPM, 0);
    assert!(scenarios
        .iter()
        .all(|s| s.nominal_probability_ppm % GRID_PPM == 0));

    let mut q = vec![0_u32; scenarios.len()];
    let mut best: Option<i128> = None;
    enumerate_probability_vectors(
        scenarios,
        radius_ppm,
        0,
        1_000_000,
        &mut q,
        &mut best,
    );
    best
}

fn enumerate_probability_vectors(
    scenarios: &[ScenarioCostDelta],
    radius_ppm: u32,
    index: usize,
    remaining_ppm: u32,
    q: &mut [u32],
    best: &mut Option<i128>,
) {
    if index + 1 == scenarios.len() {
        if remaining_ppm % GRID_PPM != 0 {
            return;
        }
        q[index] = remaining_ppm;
        let tv_l1: u64 = scenarios
            .iter()
            .zip(q.iter())
            .map(|(scenario, probability)| {
                u64::from(
                    scenario
                        .nominal_probability_ppm
                        .abs_diff(*probability),
                )
            })
            .sum();
        if tv_l1 > 2 * u64::from(radius_ppm) {
            return;
        }
        let objective: i128 = scenarios
            .iter()
            .zip(q.iter())
            .map(|(scenario, probability)| {
                i128::from(scenario.candidate_minus_baseline_cost_units)
                    * i128::from(*probability)
            })
            .sum();
        *best = Some(best.map_or(objective, |old| old.max(objective)));
        return;
    }

    let mut p = 0;
    while p <= remaining_ppm {
        q[index] = p;
        enumerate_probability_vectors(
            scenarios,
            radius_ppm,
            index + 1,
            remaining_ppm - p,
            q,
            best,
        );
        p += GRID_PPM;
    }
}

fn fixture() -> Vec<ScenarioCostDelta> {
    vec![
        ScenarioCostDelta {
            scenario_id: "routine".into(),
            nominal_probability_ppm: 600_000,
            candidate_minus_baseline_cost_units: -100,
        },
        ScenarioCostDelta {
            scenario_id: "capacity".into(),
            nominal_probability_ppm: 300_000,
            candidate_minus_baseline_cost_units: 0,
        },
        ScenarioCostDelta {
            scenario_id: "compound-weather".into(),
            nominal_probability_ppm: 100_000,
            candidate_minus_baseline_cost_units: 20,
        },
    ]
}

#[test]
fn greedy_tv_evaluator_matches_independent_exhaustive_oracle_at_multiple_radii() {
    let scenarios = fixture();
    for radius in [0, 100_000, 200_000, 500_000, 1_000_000] {
        let production = evaluate_tv_robust_cost_delta(&scenarios, radius).unwrap();
        let oracle = brute_force_tv_max(&scenarios, radius)
            .expect("nominal distribution is always in the ambiguity set");
        assert_eq!(
            production.worst_case_expected_delta_numerator_ppm, oracle,
            "mismatch at TV radius {radius} ppm"
        );
    }
}

#[test]
fn exhaustive_oracle_checks_zero_probability_stress_scenario() {
    let mut scenarios = fixture();
    scenarios.push(ScenarioCostDelta {
        scenario_id: "zero-probability-stress".into(),
        nominal_probability_ppm: 0,
        candidate_minus_baseline_cost_units: 50,
    });
    for radius in [0, 100_000, 300_000, 1_000_000] {
        let production = evaluate_tv_robust_cost_delta(&scenarios, radius).unwrap();
        let oracle = brute_force_tv_max(&scenarios, radius).unwrap();
        assert_eq!(production.worst_case_expected_delta_numerator_ppm, oracle);
    }
}

#[test]
fn oracle_is_insensitive_to_scenario_row_order() {
    let scenarios = fixture();
    let mut reversed = scenarios.clone();
    reversed.reverse();
    for radius in [0, 200_000, 500_000, 1_000_000] {
        let a = evaluate_tv_robust_cost_delta(&scenarios, radius).unwrap();
        let b = evaluate_tv_robust_cost_delta(&reversed, radius).unwrap();
        assert_eq!(a.worst_case_expected_delta_numerator_ppm, b.worst_case_expected_delta_numerator_ppm);
        assert_eq!(a.nominal_expected_delta_numerator_ppm, b.nominal_expected_delta_numerator_ppm);
    }
}
