// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Paired, fail-closed cost and tail-risk comparison for aviation recovery policies.
//!
//! Outcomes must be paired by identical scenario ID and scored under the same
//! frozen objective and cost units. Only verifier-accepted feasible outcomes
//! are scored: rejected and indeterminate plans fail closed instead of being
//! assigned an attractive penalty that might hide a hard-constraint failure.
//!
//! This module computes descriptive paired metrics. It does not construct the
//! scenarios, verify plan constraints, estimate confidence intervals, or prove
//! that synthetic costs correspond to real airline costs. Those are separate
//! evidence and evaluation responsibilities.

use std::collections::BTreeSet;

/// Verification disposition for one plan on one scenario.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutcomeVerification {
    /// An external or separately maintained checker accepted modeled feasibility.
    VerifiedFeasible,
    /// The checker rejected the plan or found a hard-constraint violation.
    Rejected,
    /// The checker could not establish feasibility, for example due to missing evidence.
    Indeterminate,
}

/// A scored policy outcome tied to one immutable scenario identifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScenarioOutcome {
    /// Stable ID identifying the exact scenario and its immutable input digest.
    pub scenario_id: String,
    /// Objective cost in the frozen benchmark's integer cost units.
    pub cost: u64,
    /// Must be accepted before this outcome can enter the cost comparison.
    pub verification: OutcomeVerification,
}

/// Which side of a comparison failed validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicySide {
    Baseline,
    Candidate,
}

/// Why a paired robust-cost comparison was refused.
#[derive(Debug, Clone, PartialEq)]
pub enum ComparisonError {
    EmptyScenarioSet,
    LengthMismatch {
        baseline: usize,
        candidate: usize,
    },
    InvalidTailFraction {
        tail_fraction: f64,
    },
    DuplicateScenarioId {
        side: PolicySide,
        scenario_id: String,
    },
    ScenarioIdMismatch {
        index: usize,
        baseline_id: String,
        candidate_id: String,
    },
    OutcomeNotVerified {
        side: PolicySide,
        index: usize,
        scenario_id: String,
        verification: OutcomeVerification,
    },
}

/// Paired cost and tail-risk metrics for two policies on the same cases.
#[derive(Debug, Clone, PartialEq)]
pub struct RobustnessReport {
    pub scenario_count: usize,
    /// Upper-tail fraction, e.g. 0.05 means the worst 5% of scenarios.
    pub tail_fraction: f64,
    pub baseline_mean_cost: f64,
    pub candidate_mean_cost: f64,
    /// Mean of paired (candidate cost - baseline cost); negative favors candidate.
    pub mean_cost_delta: f64,
    pub baseline_tail_mean_cost: f64,
    pub candidate_tail_mean_cost: f64,
    /// Mean paired cost delta in its worst upper-tail fraction; positive is worse.
    pub tail_mean_cost_delta: f64,
    pub worst_case_baseline_cost: u64,
    pub worst_case_candidate_cost: u64,
    /// Largest per-scenario (candidate cost - baseline cost); positive is regression.
    pub worst_paired_cost_delta: i128,
    pub regression_scenarios: usize,
    pub improvement_scenarios: usize,
    pub unchanged_scenarios: usize,
}

/// Compare two policies on identically identified, verifier-accepted outcomes.
///
/// The same scenario IDs must appear in the same order. A caller must first
/// validate both policies' plans; this function refuses to rank any rejected
/// or indeterminate plan as a cheaper alternative. Tail fraction is the
/// fraction of scenario probability mass in the upper cost tail, not a
/// percentile confidence level. Fractional sample mass is used when needed:
/// with 10 equally weighted scenarios and tail fraction 0.05, this estimates
/// the empirical worst-5% tail using half the weight of the single worst case.
///
/// This is a descriptive comparison only. Use cluster-aware paired inference
/// and a preregistered test protocol before making a statistical superiority claim.
pub fn compare_robust_costs(
    baseline: &[ScenarioOutcome],
    candidate: &[ScenarioOutcome],
    tail_fraction: f64,
) -> Result<RobustnessReport, ComparisonError> {
    if baseline.is_empty() {
        return Err(ComparisonError::EmptyScenarioSet);
    }
    if baseline.len() != candidate.len() {
        return Err(ComparisonError::LengthMismatch {
            baseline: baseline.len(),
            candidate: candidate.len(),
        });
    }
    if !tail_fraction.is_finite() || tail_fraction <= 0.0 || tail_fraction > 1.0 {
        return Err(ComparisonError::InvalidTailFraction { tail_fraction });
    }

    validate_unique_ids(baseline, PolicySide::Baseline)?;
    validate_unique_ids(candidate, PolicySide::Candidate)?;

    for (index, (base, cand)) in baseline.iter().zip(candidate.iter()).enumerate() {
        if base.scenario_id != cand.scenario_id {
            return Err(ComparisonError::ScenarioIdMismatch {
                index,
                baseline_id: base.scenario_id.clone(),
                candidate_id: cand.scenario_id.clone(),
            });
        }
        if base.verification != OutcomeVerification::VerifiedFeasible {
            return Err(ComparisonError::OutcomeNotVerified {
                side: PolicySide::Baseline,
                index,
                scenario_id: base.scenario_id.clone(),
                verification: base.verification,
            });
        }
        if cand.verification != OutcomeVerification::VerifiedFeasible {
            return Err(ComparisonError::OutcomeNotVerified {
                side: PolicySide::Candidate,
                index,
                scenario_id: cand.scenario_id.clone(),
                verification: cand.verification,
            });
        }
    }

    let baseline_costs: Vec<u64> = baseline.iter().map(|outcome| outcome.cost).collect();
    let candidate_costs: Vec<u64> = candidate.iter().map(|outcome| outcome.cost).collect();
    let paired_deltas: Vec<f64> = baseline
        .iter()
        .zip(candidate.iter())
        .map(|(base, cand)| (i128::from(cand.cost) - i128::from(base.cost)) as f64)
        .collect();

    let count = baseline.len() as f64;
    let baseline_mean_cost = mean_u64(&baseline_costs);
    let candidate_mean_cost = mean_u64(&candidate_costs);
    let mean_cost_delta = paired_deltas.iter().sum::<f64>() / count;
    let tail_mean_cost_delta = upper_tail_mean_f64(&paired_deltas, tail_fraction);

    let mut regression_scenarios = 0;
    let mut improvement_scenarios = 0;
    let mut unchanged_scenarios = 0;
    let mut worst_paired_cost_delta = i128::MIN;

    for (base, cand) in baseline.iter().zip(candidate.iter()) {
        let delta = i128::from(cand.cost) - i128::from(base.cost);
        worst_paired_cost_delta = worst_paired_cost_delta.max(delta);
        match cand.cost.cmp(&base.cost) {
            std::cmp::Ordering::Greater => regression_scenarios += 1,
            std::cmp::Ordering::Less => improvement_scenarios += 1,
            std::cmp::Ordering::Equal => unchanged_scenarios += 1,
        }
    }

    Ok(RobustnessReport {
        scenario_count: baseline.len(),
        tail_fraction,
        baseline_mean_cost,
        candidate_mean_cost,
        mean_cost_delta,
        baseline_tail_mean_cost: upper_tail_mean_u64(&baseline_costs, tail_fraction),
        candidate_tail_mean_cost: upper_tail_mean_u64(&candidate_costs, tail_fraction),
        tail_mean_cost_delta,
        worst_case_baseline_cost: *baseline_costs.iter().max().expect("nonempty after validation"),
        worst_case_candidate_cost: *candidate_costs.iter().max().expect("nonempty after validation"),
        worst_paired_cost_delta,
        regression_scenarios,
        improvement_scenarios,
        unchanged_scenarios,
    })
}

fn validate_unique_ids(
    outcomes: &[ScenarioOutcome],
    side: PolicySide,
) -> Result<(), ComparisonError> {
    let mut ids = BTreeSet::new();
    for outcome in outcomes {
        if !ids.insert(outcome.scenario_id.as_str()) {
            return Err(ComparisonError::DuplicateScenarioId {
                side,
                scenario_id: outcome.scenario_id.clone(),
            });
        }
    }
    Ok(())
}

fn mean_u64(values: &[u64]) -> f64 {
    values.iter().map(|value| *value as f64).sum::<f64>() / values.len() as f64
}

fn upper_tail_mean_u64(values: &[u64], tail_fraction: f64) -> f64 {
    let floats: Vec<f64> = values.iter().map(|value| *value as f64).collect();
    upper_tail_mean_f64(&floats, tail_fraction)
}

fn upper_tail_mean_f64(values: &[f64], tail_fraction: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| right.total_cmp(left));

    let tail_mass = sorted.len() as f64 * tail_fraction;
    let full_count = tail_mass.floor() as usize;
    let fractional_mass = tail_mass - full_count as f64;
    let full_sum = sorted.iter().take(full_count).sum::<f64>();
    let fractional_sum = if fractional_mass > 0.0 && full_count < sorted.len() {
        sorted[full_count] * fractional_mass
    } else {
        0.0
    };

    (full_sum + fractional_sum) / tail_mass
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcomes(prefix: &str, costs: &[u64]) -> Vec<ScenarioOutcome> {
        costs
            .iter()
            .enumerate()
            .map(|(index, cost)| ScenarioOutcome {
                scenario_id: format!("{prefix}-{index}"),
                cost: *cost,
                verification: OutcomeVerification::VerifiedFeasible,
            })
            .collect()
    }

    fn paired(costs: &[u64]) -> Vec<ScenarioOutcome> {
        outcomes("scenario", costs)
    }

    fn recost(source: &[ScenarioOutcome], costs: &[u64]) -> Vec<ScenarioOutcome> {
        source
            .iter()
            .zip(costs.iter())
            .map(|(outcome, cost)| ScenarioOutcome {
                scenario_id: outcome.scenario_id.clone(),
                cost: *cost,
                verification: outcome.verification,
            })
            .collect()
    }

    #[test]
    fn rejects_empty_scenario_set() {
        assert_eq!(
            compare_robust_costs(&[], &[], 0.05),
            Err(ComparisonError::EmptyScenarioSet)
        );
    }

    #[test]
    fn rejects_mismatched_lengths() {
        assert_eq!(
            compare_robust_costs(&paired(&[1, 2]), &paired(&[1]), 0.5),
            Err(ComparisonError::LengthMismatch {
                baseline: 2,
                candidate: 1,
            })
        );
    }

    #[test]
    fn rejects_invalid_tail_fractions() {
        for tail in [0.0, -0.1, 1.01, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                compare_robust_costs(&paired(&[1]), &paired(&[1]), tail),
                Err(ComparisonError::InvalidTailFraction { .. })
            ));
        }
    }

    #[test]
    fn rejects_scenario_misalignment_instead_of_comparing_unpaired_cases() {
        let baseline = paired(&[10, 20]);
        let mut candidate = recost(&baseline, &[9, 19]);
        candidate.swap(0, 1);
        assert!(matches!(
            compare_robust_costs(&baseline, &candidate, 0.5),
            Err(ComparisonError::ScenarioIdMismatch { index: 0, .. })
        ));
    }

    #[test]
    fn rejects_duplicate_scenario_ids() {
        let baseline = vec![
            ScenarioOutcome {
                scenario_id: "same".into(),
                cost: 10,
                verification: OutcomeVerification::VerifiedFeasible,
            },
            ScenarioOutcome {
                scenario_id: "same".into(),
                cost: 20,
                verification: OutcomeVerification::VerifiedFeasible,
            },
        ];
        assert!(matches!(
            compare_robust_costs(&baseline, &baseline, 0.5),
            Err(ComparisonError::DuplicateScenarioId { side: PolicySide::Baseline, .. })
        ));
    }

    #[test]
    fn refuses_to_score_rejected_or_indeterminate_plans() {
        for status in [OutcomeVerification::Rejected, OutcomeVerification::Indeterminate] {
            let baseline = paired(&[10]);
            let mut candidate = recost(&baseline, &[1]);
            candidate[0].verification = status;
            assert!(matches!(
                compare_robust_costs(&baseline, &candidate, 0.5),
                Err(ComparisonError::OutcomeNotVerified {
                    side: PolicySide::Candidate,
                    index: 0,
                    ..
                })
            ));
        }
    }

    #[test]
    fn identical_policies_have_zero_paired_delta() {
        let baseline = paired(&[8, 12, 20, 40]);
        let report = compare_robust_costs(&baseline, &baseline, 0.25).unwrap();
        assert_eq!(report.mean_cost_delta, 0.0);
        assert_eq!(report.tail_mean_cost_delta, 0.0);
        assert_eq!(report.worst_paired_cost_delta, 0);
        assert_eq!(report.unchanged_scenarios, 4);
    }

    #[test]
    fn captures_regressions_hidden_by_a_better_mean() {
        let baseline = paired(&[500, 100, 100, 100, 100, 100, 100, 100, 100, 100]);
        let candidate = recost(&baseline, &[600, 80, 80, 80, 80, 80, 80, 80, 80, 80]);
        let report = compare_robust_costs(&baseline, &candidate, 0.2).unwrap();

        assert!(report.mean_cost_delta < 0.0);
        assert!(report.candidate_mean_cost < report.baseline_mean_cost);
        assert!(report.candidate_tail_mean_cost > report.baseline_tail_mean_cost);
        assert!(report.tail_mean_cost_delta > 0.0);
        assert_eq!(report.worst_paired_cost_delta, 100);
        assert_eq!(report.regression_scenarios, 1);
        assert_eq!(report.improvement_scenarios, 9);
    }

    #[test]
    fn reports_fractional_empirical_tail_mass() {
        let baseline = paired(&[10, 20, 30, 40]);
        let candidate = recost(&baseline, &[10, 20, 30, 100]);
        let report = compare_robust_costs(&baseline, &candidate, 0.125);
        let report = report.unwrap();
        assert_eq!(report.baseline_tail_mean_cost, 40.0);
        assert_eq!(report.candidate_tail_mean_cost, 100.0);
        assert_eq!(report.tail_mean_cost_delta, 60.0);
    }

    #[test]
    fn full_tail_fraction_equals_mean_cost() {
        let baseline = paired(&[10, 20, 30, 40]);
        let report = compare_robust_costs(&baseline, &baseline, 1.0).unwrap();
        assert_eq!(report.baseline_tail_mean_cost, report.baseline_mean_cost);
    }

    #[test]
    fn paired_cost_delta_does_not_overflow_unsigned_subtraction() {
        let baseline = paired(&[u64::MAX]);
        let candidate = recost(&baseline, &[0]);
        let report = compare_robust_costs(&baseline, &candidate, 1.0).unwrap();
        assert!(report.mean_cost_delta < 0.0);
        assert_eq!(report.worst_paired_cost_delta, -i128::from(u64::MAX));
    }
}
