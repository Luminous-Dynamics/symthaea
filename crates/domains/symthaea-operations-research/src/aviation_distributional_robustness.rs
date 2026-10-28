// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Total-variation distributionally robust evaluation of paired recovery decisions.
//!
//! Each scenario carries the candidate-minus-baseline cost difference and a
//! nominal probability. The evaluator computes nominal expected delta, then the
//! worst expected delta over all probability distributions within a total
//! variation (TV) radius of the nominal distribution.
//!
//! This is a finite-support ambiguity set. Results are conditional on the
//! scenario support, nominal probabilities, cost model and declared TV radius;
//! it does not guarantee protection against omitted scenarios or arbitrary shift.
//!
//! Probabilities use integer parts per million (ppm), summing exactly to 1_000_000.
//! TV radius is also expressed in ppm of moved probability mass, i.e. radius rho
//! corresponds to an L1 distance no greater than 2*rho. Positive cost delta means
//! the candidate is worse than baseline; negative means the candidate is better.

use std::collections::BTreeSet;

/// Probability scale for exact integer probability arithmetic.
pub const PROBABILITY_SCALE_PPM: u32 = 1_000_000;

/// Cost difference for one frozen scenario.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScenarioCostDelta {
    /// Stable ID bound by the caller to an immutable scenario/input digest.
    pub scenario_id: String,
    /// Nominal probability in parts per million. All rows must sum to one million.
    pub nominal_probability_ppm: u32,
    /// Candidate objective cost minus baseline objective cost in frozen integer units.
    pub candidate_minus_baseline_cost_units: i64,
}

/// Explicit failure when a distributionally robust comparison cannot be evaluated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistributionalRobustError {
    EmptyScenarioSet,
    EmptyScenarioId,
    DuplicateScenarioId { scenario_id: String },
    ProbabilityMassMismatch { actual_ppm: u64 },
    InvalidTvRadius { radius_ppm: u32 },
    ArithmeticOverflow { context: &'static str },
    TvRadiusMismatch { report_radius_ppm: u32, requested_radius_ppm: u32 },
}

/// Nominal and worst-case probabilities for one scenario.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScenarioProbability {
    pub scenario_id: String,
    pub nominal_probability_ppm: u32,
    pub worst_case_probability_ppm: u32,
    pub candidate_minus_baseline_cost_units: i64,
}

/// Distributionally robust paired cost report.
#[derive(Debug, Clone, PartialEq)]
pub struct DistributionalRobustReport {
    pub scenario_count: usize,
    pub tv_radius_ppm: u32,
    /// Exact numerator: expected cost delta = numerator / PROBABILITY_SCALE_PPM.
    pub nominal_expected_delta_numerator_ppm: i128,
    /// Exact numerator over the worst distribution in the declared TV ball.
    pub worst_case_expected_delta_numerator_ppm: i128,
    pub nominal_expected_delta_units: f64,
    pub worst_case_expected_delta_units: f64,
    pub worst_single_scenario_delta_units: i64,
    /// Deterministic tie-break: lexicographically smallest scenario ID among ties.
    pub worst_single_scenario_id: String,
    pub scenarios_improved: usize,
    pub scenarios_regressed: usize,
    pub scenarios_unchanged: usize,
    /// Sorted by scenario ID, independent of input row order.
    pub probabilities: Vec<ScenarioProbability>,
}

/// Predeclared acceptance thresholds. Expected cost is compared using exact
/// ppm-numerator arithmetic; the single-scenario ceiling is also explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistributionalRobustConstraints {
    pub tv_radius_ppm: u32,
    /// Candidate-minus-baseline expected cost must not exceed this integer unit value.
    pub max_worst_case_expected_delta_units: i64,
    /// Maximum allowed positive cost delta in any one scenario in the finite support.
    pub max_single_scenario_regression_units: u64,
}

/// Reasons a candidate fails a distributionally robust acceptance gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistributionalExclusion {
    WorstCaseExpectedDeltaExceeded {
        actual_numerator_ppm: i128,
        maximum_units: i64,
    },
    SingleScenarioRegressionExceeded {
        scenario_id: String,
        actual_units: u64,
        maximum_units: u64,
    },
}

/// Result of applying predeclared thresholds to a report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DistributionalRobustAssessment {
    pub eligible: bool,
    pub exclusions: Vec<DistributionalExclusion>,
}

fn add_checked(a: i128, b: i128, context: &'static str) -> Result<i128, DistributionalRobustError> {
    a.checked_add(b)
        .ok_or(DistributionalRobustError::ArithmeticOverflow { context })
}

fn weighted_delta_numerator(
    scenarios: &[ScenarioCostDelta],
    probabilities_ppm: &[u32],
    order_by_id: &[usize],
) -> Result<i128, DistributionalRobustError> {
    let mut total = 0_i128;
    for index in order_by_id {
        let product = i128::from(probabilities_ppm[*index])
            .checked_mul(i128::from(scenarios[*index].candidate_minus_baseline_cost_units))
            .ok_or(DistributionalRobustError::ArithmeticOverflow {
                context: "probability-weighted cost delta",
            })?;
        total = add_checked(total, product, "weighted cost-delta sum")?;
    }
    Ok(total)
}

/// Evaluate the worst expected candidate-minus-baseline cost over a finite TV ball.
///
/// The exact optimizer for a finite loss vector moves probability mass from
/// lowest-cost-delta support points to highest-cost-delta support points until
/// the TV budget, donor mass or receiver capacity is exhausted. A greedy
/// transfer is optimal for this one-dimensional expectation problem because
/// every unit of mass should move from the smallest delta to the largest delta.
///
/// This function does not construct or validate the scenario set's realism,
/// estimate the nominal probabilities, or prove that the TV radius is calibrated.
pub fn evaluate_tv_robust_cost_delta(
    scenarios: &[ScenarioCostDelta],
    tv_radius_ppm: u32,
) -> Result<DistributionalRobustReport, DistributionalRobustError> {
    if scenarios.is_empty() {
        return Err(DistributionalRobustError::EmptyScenarioSet);
    }
    if tv_radius_ppm > PROBABILITY_SCALE_PPM {
        return Err(DistributionalRobustError::InvalidTvRadius {
            radius_ppm: tv_radius_ppm,
        });
    }

    let mut ids = BTreeSet::new();
    let mut probability_sum = 0_u64;
    for scenario in scenarios {
        if scenario.scenario_id.trim().is_empty() {
            return Err(DistributionalRobustError::EmptyScenarioId);
        }
        if !ids.insert(scenario.scenario_id.as_str()) {
            return Err(DistributionalRobustError::DuplicateScenarioId {
                scenario_id: scenario.scenario_id.clone(),
            });
        }
        probability_sum += u64::from(scenario.nominal_probability_ppm);
    }
    if probability_sum != u64::from(PROBABILITY_SCALE_PPM) {
        return Err(DistributionalRobustError::ProbabilityMassMismatch {
            actual_ppm: probability_sum,
        });
    }

    let mut by_cost: Vec<usize> = (0..scenarios.len()).collect();
    by_cost.sort_by(|a, b| {
        scenarios[*a]
            .candidate_minus_baseline_cost_units
            .cmp(&scenarios[*b].candidate_minus_baseline_cost_units)
            .then_with(|| scenarios[*a].scenario_id.cmp(&scenarios[*b].scenario_id))
    });
    let mut by_id: Vec<usize> = (0..scenarios.len()).collect();
    by_id.sort_by(|a, b| scenarios[*a].scenario_id.cmp(&scenarios[*b].scenario_id));

    let nominal: Vec<u32> = scenarios
        .iter()
        .map(|scenario| scenario.nominal_probability_ppm)
        .collect();
    let nominal_numerator = weighted_delta_numerator(scenarios, &nominal, &by_id)?;
    let mut worst_probabilities: Vec<u32> = nominal.clone();
    let mut remaining_mass = u64::from(tv_radius_ppm);
    let mut low = 0_usize;
    let mut high = by_cost.len() - 1;

    while remaining_mass > 0 && low < high {
        while low < high && worst_probabilities[by_cost[low]] == 0 {
            low += 1;
        }
        while low < high
            && worst_probabilities[by_cost[high]] == PROBABILITY_SCALE_PPM
        {
            high -= 1;
        }
        if low >= high {
            break;
        }

        let donor = by_cost[low];
        let receiver = by_cost[high];
        let donor_mass = u64::from(worst_probabilities[donor]);
        let receiver_capacity =
            u64::from(PROBABILITY_SCALE_PPM - worst_probabilities[receiver]);
        let moved = remaining_mass.min(donor_mass).min(receiver_capacity);

        if moved == 0 {
            break;
        }
        worst_probabilities[donor] -= moved as u32;
        worst_probabilities[receiver] += moved as u32;
        remaining_mass -= moved;

        if worst_probabilities[donor] == 0 {
            low += 1;
        }
        if worst_probabilities[receiver] == PROBABILITY_SCALE_PPM {
            high = high.saturating_sub(1);
        }
    }

    let worst_numerator = weighted_delta_numerator(scenarios, &worst_probabilities, &by_id)?;
    let mut worst_single_delta = i64::MIN;
    let mut worst_single_id = String::new();
    let mut improved = 0;
    let mut regressed = 0;
    let mut unchanged = 0;

    // Iterate by ID so equal-cost ties always select the same reported scenario.
    for index in &by_id {
        let delta = scenarios[*index].candidate_minus_baseline_cost_units;
        match delta.cmp(&0) {
            std::cmp::Ordering::Less => improved += 1,
            std::cmp::Ordering::Greater => regressed += 1,
            std::cmp::Ordering::Equal => unchanged += 1,
        }
        if delta > worst_single_delta || worst_single_id.is_empty() {
            worst_single_delta = delta;
            worst_single_id = scenarios[*index].scenario_id.clone();
        }
    }

    let probabilities = by_id
        .iter()
        .map(|index| ScenarioProbability {
            scenario_id: scenarios[*index].scenario_id.clone(),
            nominal_probability_ppm: scenarios[*index].nominal_probability_ppm,
            worst_case_probability_ppm: worst_probabilities[*index],
            candidate_minus_baseline_cost_units: scenarios[*index]
                .candidate_minus_baseline_cost_units,
        })
        .collect();

    Ok(DistributionalRobustReport {
        scenario_count: scenarios.len(),
        tv_radius_ppm,
        nominal_expected_delta_numerator_ppm: nominal_numerator,
        worst_case_expected_delta_numerator_ppm: worst_numerator,
        nominal_expected_delta_units: nominal_numerator as f64
            / f64::from(PROBABILITY_SCALE_PPM),
        worst_case_expected_delta_units: worst_numerator as f64
            / f64::from(PROBABILITY_SCALE_PPM),
        worst_single_scenario_delta_units: worst_single_delta,
        worst_single_scenario_id: worst_single_id,
        scenarios_improved: improved,
        scenarios_regressed: regressed,
        scenarios_unchanged: unchanged,
        probabilities,
    })
}

/// Apply acceptance thresholds. A risk report computed at one TV radius cannot
/// be reused at another radius. Expected-value gating uses exact integer
/// numerator arithmetic; the f64 convenience fields are for reporting only.
pub fn assess_tv_robust_candidate(
    report: &DistributionalRobustReport,
    constraints: DistributionalRobustConstraints,
) -> Result<DistributionalRobustAssessment, DistributionalRobustError> {
    if constraints.tv_radius_ppm > PROBABILITY_SCALE_PPM {
        return Err(DistributionalRobustError::InvalidTvRadius {
            radius_ppm: constraints.tv_radius_ppm,
        });
    }
    if report.tv_radius_ppm != constraints.tv_radius_ppm {
        return Err(DistributionalRobustError::TvRadiusMismatch {
            report_radius_ppm: report.tv_radius_ppm,
            requested_radius_ppm: constraints.tv_radius_ppm,
        });
    }

    let maximum_numerator = i128::from(constraints.max_worst_case_expected_delta_units)
        .checked_mul(i128::from(PROBABILITY_SCALE_PPM))
        .ok_or(DistributionalRobustError::ArithmeticOverflow {
            context: "expected-cost threshold scaling",
        })?;
    let mut exclusions = Vec::new();

    if report.worst_case_expected_delta_numerator_ppm > maximum_numerator {
        exclusions.push(DistributionalExclusion::WorstCaseExpectedDeltaExceeded {
            actual_numerator_ppm: report.worst_case_expected_delta_numerator_ppm,
            maximum_units: constraints.max_worst_case_expected_delta_units,
        });
    }

    if report.worst_single_scenario_delta_units > 0
        && report.worst_single_scenario_delta_units as u64
            > constraints.max_single_scenario_regression_units
    {
        exclusions.push(DistributionalExclusion::SingleScenarioRegressionExceeded {
            scenario_id: report.worst_single_scenario_id.clone(),
            actual_units: report.worst_single_scenario_delta_units as u64,
            maximum_units: constraints.max_single_scenario_regression_units,
        });
    }

    Ok(DistributionalRobustAssessment {
        eligible: exclusions.is_empty(),
        exclusions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scenarios() -> Vec<ScenarioCostDelta> {
        vec![
            ScenarioCostDelta {
                scenario_id: "healthy".into(),
                nominal_probability_ppm: 600_000,
                candidate_minus_baseline_cost_units: -100,
            },
            ScenarioCostDelta {
                scenario_id: "ordinary".into(),
                nominal_probability_ppm: 300_000,
                candidate_minus_baseline_cost_units: 0,
            },
            ScenarioCostDelta {
                scenario_id: "disruption".into(),
                nominal_probability_ppm: 100_000,
                candidate_minus_baseline_cost_units: 20,
            },
        ]
    }

    #[test]
    fn total_variation_shift_has_hand_checkable_expected_value() {
        let report = evaluate_tv_robust_cost_delta(&scenarios(), 200_000).unwrap();
        assert_eq!(report.nominal_expected_delta_numerator_ppm, -58_000_000);
        assert_eq!(report.worst_case_expected_delta_numerator_ppm, -34_000_000);
        assert_eq!(report.worst_case_expected_delta_units, -34.0);
        let q = report.probabilities;
        assert_eq!(q[0].scenario_id, "disruption");
        assert_eq!(q[0].worst_case_probability_ppm, 300_000);
        assert_eq!(q[1].scenario_id, "healthy");
        assert_eq!(q[1].worst_case_probability_ppm, 400_000);
        assert_eq!(q[2].scenario_id, "ordinary");
        assert_eq!(q[2].worst_case_probability_ppm, 300_000);
    }

    #[test]
    fn zero_tv_radius_preserves_nominal_distribution() {
        let report = evaluate_tv_robust_cost_delta(&scenarios(), 0).unwrap();
        assert_eq!(
            report.worst_case_expected_delta_numerator_ppm,
            report.nominal_expected_delta_numerator_ppm
        );
        for row in report.probabilities {
            assert_eq!(row.nominal_probability_ppm, row.worst_case_probability_ppm);
        }
    }

    #[test]
    fn full_tv_radius_can_concentrate_mass_on_the_worst_scenario() {
        let report = evaluate_tv_robust_cost_delta(&scenarios(), PROBABILITY_SCALE_PPM).unwrap();
        assert_eq!(report.worst_case_expected_delta_units, 20.0);
        assert_eq!(
            report.probabilities.iter().find(|row| row.scenario_id == "disruption")
                .map(|row| row.worst_case_probability_ppm),
            Some(1_000_000)
        );
    }

    #[test]
    fn identical_scenario_costs_have_no_distribution_shift_penalty() {
        let scenarios = vec![
            ScenarioCostDelta { scenario_id: "A".into(), nominal_probability_ppm: 250_000, candidate_minus_baseline_cost_units: 7 },
            ScenarioCostDelta { scenario_id: "B".into(), nominal_probability_ppm: 750_000, candidate_minus_baseline_cost_units: 7 },
        ];
        let report = evaluate_tv_robust_cost_delta(&scenarios, 700_000).unwrap();
        assert_eq!(report.nominal_expected_delta_units, 7.0);
        assert_eq!(report.worst_case_expected_delta_units, 7.0);
    }

    #[test]
    fn rejects_probability_mass_that_does_not_sum_to_one() {
        let mut bad = scenarios();
        bad[0].nominal_probability_ppm -= 1;
        assert_eq!(
            evaluate_tv_robust_cost_delta(&bad, 10),
            Err(DistributionalRobustError::ProbabilityMassMismatch { actual_ppm: 999_999 })
        );
    }

    #[test]
    fn rejects_duplicate_and_empty_ids() {
        let mut bad = scenarios();
        bad[1].scenario_id = bad[0].scenario_id.clone();
        assert!(matches!(
            evaluate_tv_robust_cost_delta(&bad, 10),
            Err(DistributionalRobustError::DuplicateScenarioId { .. })
        ));
        bad[1].scenario_id.clear();
        assert_eq!(
            evaluate_tv_robust_cost_delta(&bad, 10),
            Err(DistributionalRobustError::EmptyScenarioId)
        );
    }

    #[test]
    fn rejects_invalid_tv_radius() {
        assert!(matches!(
            evaluate_tv_robust_cost_delta(&scenarios(), 1_000_001),
            Err(DistributionalRobustError::InvalidTvRadius { .. })
        ));
    }

    #[test]
    fn result_is_deterministic_under_input_row_permutation() {
        let base = scenarios();
        let a = evaluate_tv_robust_cost_delta(&base, 200_000).unwrap();
        let mut reversed = base;
        reversed.reverse();
        let b = evaluate_tv_robust_cost_delta(&reversed, 200_000).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn robust_gate_rejects_when_distribution_shift_flips_expected_benefit() {
        let report = evaluate_tv_robust_cost_delta(&scenarios(), 500_000).unwrap();
        assert_eq!(report.nominal_expected_delta_units, -58.0);
        assert_eq!(report.worst_case_expected_delta_units, 2.0);
        let assessment = assess_tv_robust_candidate(
            &report,
            DistributionalRobustConstraints {
                tv_radius_ppm: 500_000,
                max_worst_case_expected_delta_units: 0,
                max_single_scenario_regression_units: 20,
            },
        )
        .unwrap();
        assert!(!assessment.eligible);
        assert!(assessment.exclusions.iter().any(|x| matches!(
            x,
            DistributionalExclusion::WorstCaseExpectedDeltaExceeded { .. }
        )));
    }

    #[test]
    fn robust_gate_accepts_when_benefit_survives_the_declared_tv_ball() {
        let report = evaluate_tv_robust_cost_delta(&scenarios(), 200_000).unwrap();
        let assessment = assess_tv_robust_candidate(
            &report,
            DistributionalRobustConstraints {
                tv_radius_ppm: 200_000,
                max_worst_case_expected_delta_units: 0,
                max_single_scenario_regression_units: 20,
            },
        )
        .unwrap();
        assert!(assessment.eligible);
        assert!(assessment.exclusions.is_empty());
    }

    #[test]
    fn rejects_gate_with_mismatched_tv_radius() {
        let report = evaluate_tv_robust_cost_delta(&scenarios(), 200_000).unwrap();
        assert!(matches!(
            assess_tv_robust_candidate(
                &report,
                DistributionalRobustConstraints {
                    tv_radius_ppm: 500_000,
                    max_worst_case_expected_delta_units: 0,
                    max_single_scenario_regression_units: 20,
                },
            ),
            Err(DistributionalRobustError::TvRadiusMismatch { .. })
        ));
    }

    #[test]
    fn positive_worst_scenario_is_reported_with_stable_id() {
        let mut items = scenarios();
        items.push(ScenarioCostDelta {
            scenario_id: "adverse".into(),
            nominal_probability_ppm: 0,
            candidate_minus_baseline_cost_units: 20,
        });
        let report = evaluate_tv_robust_cost_delta(&items, 100_000).unwrap();
        assert_eq!(report.worst_single_scenario_id, "adverse");
        assert_eq!(report.worst_single_scenario_delta_units, 20);
    }

}
