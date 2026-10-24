//! Pareto-frontier reporting for matched-cohort quantum/classical optimizer comparisons.
//!
//! Points must be aggregated over the same frozen cohort and use the same predeclared
//! normalization scale for every selected cost metric. This avoids disguising a
//! preference-dependent scalar risk as a universal ranking.

use core::fmt;

use crate::mis_cost_risk::{MisCostMetric, MisEvidenceClass};

/// A normalized aggregate for one cost dimension on a frozen benchmark cohort.
#[derive(Debug, Clone, PartialEq)]
pub struct MisNormalizedCostValue {
    /// Resource dimension.
    pub metric: MisCostMetric,
    /// Mean amount divided by one shared, preregistered reference amount.
    pub normalized_mean_amount: f64,
    /// Three-letter uppercase currency code, required for the money metric only.
    pub currency_code: Option<String>,
}

/// One policy's aggregated result on a named benchmark cohort.
#[derive(Debug, Clone, PartialEq)]
pub struct MisParetoPoint {
    /// Identifier for the identical frozen workload cohort and aggregation protocol.
    pub cohort_id: String,
    /// Unique method/policy identifier.
    pub policy_id: String,
    /// Evidence class used to produce this aggregate; shown, never inferred from policy name.
    pub evidence_class: MisEvidenceClass,
    /// Mean normalized solution-quality loss in [0, 1], lower is better.
    pub normalized_quality_loss: f64,
    /// Infeasibility rate in [0, 1], lower is better.
    pub infeasible_rate: f64,
    /// Target-miss rate in [0, 1], lower is better.
    pub target_miss_rate: f64,
    /// Normalized mean cost for every selected metric, using shared cohort scales.
    pub costs: Vec<MisNormalizedCostValue>,
}

/// Errors from Pareto-frontier construction.
#[derive(Debug, Clone, PartialEq)]
pub enum MisParetoError {
    /// No candidate point was provided.
    EmptyPoints,
    /// At least one cost metric must be selected.
    EmptyMetrics,
    /// A cost dimension is selected more than once.
    DuplicateMetric { /// Cost metric.
        metric: MisCostMetric,
    },
    /// Cohort IDs are empty or differ across policies.
    MixedCohorts,
    /// Policy ID is missing or duplicated.
    DuplicateOrEmptyPolicyId,
    /// Outcome loss or rates are non-finite or outside [0, 1].
    InvalidOutcomeRates { /// Invalid policy.
        policy_id: String,
    },
    /// A point does not provide exactly the requested cost metrics.
    MismatchedCostMetrics { /// Policy missing or adding a dimension.
        policy_id: String,
    },
    /// A normalized cost amount is negative or non-finite.
    InvalidNormalizedCost {
        /// Invalid policy.
        policy_id: String,
        /// Invalid dimension.
        metric: MisCostMetric,
    },
    /// Currency is absent/invalid for money, or present for a non-money metric.
    InvalidCurrency {
        /// Invalid policy.
        policy_id: String,
        /// Invalid dimension.
        metric: MisCostMetric,
    },
    /// Policies use different currencies for monetary comparison.
    CurrencyMismatch,
    /// Dominance tolerance is negative or non-finite.
    InvalidTolerance,
}

impl fmt::Display for MisParetoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPoints => write!(f, "at least one Pareto point is required"),
            Self::EmptyMetrics => write!(f, "at least one cost metric is required"),
            Self::DuplicateMetric { metric } => write!(f, "Pareto metric {} is duplicated", metric.label()),
            Self::MixedCohorts => write!(f, "Pareto points must share one nonempty cohort ID"),
            Self::DuplicateOrEmptyPolicyId => write!(f, "policy IDs must be nonempty and unique"),
            Self::InvalidOutcomeRates { policy_id } => write!(f, "policy {policy_id} has invalid loss/rate values"),
            Self::MismatchedCostMetrics { policy_id } => write!(f, "policy {policy_id} does not provide exactly the requested metrics"),
            Self::InvalidNormalizedCost { policy_id, metric } => write!(f, "policy {policy_id} has invalid normalized cost for {}", metric.label()),
            Self::InvalidCurrency { policy_id, metric } => write!(f, "policy {policy_id} has invalid currency metadata for {}", metric.label()),
            Self::CurrencyMismatch => write!(f, "Pareto points use different monetary currencies"),
            Self::InvalidTolerance => write!(f, "Pareto tolerance must be finite and nonnegative"),
        }
    }
}

impl std::error::Error for MisParetoError {}

/// Returns input indices on the Pareto frontier, preserving input order.
///
/// Lower quality loss, infeasibility, target-miss rate, and selected costs are better.
/// A point is dominated if another point is no worse on every dimension and better by
/// more than the tolerance on at least one dimension. Cost amounts must already be
/// normalized with the same predeclared scale for all policies in the cohort.
///
/// For the money metric, every point must use the same explicit currency. This
/// function does not infer exchange rates, normalize independently per policy, or
/// validate that the caller used the same held-out tasks to build each aggregate.
pub fn mis_cost_pareto_frontier(
    points: &[MisParetoPoint],
    metrics: &[MisCostMetric],
    tolerance: f64,
) -> Result<Vec<usize>, MisParetoError> {
    if points.is_empty() {
        return Err(MisParetoError::EmptyPoints);
    }
    if metrics.is_empty() {
        return Err(MisParetoError::EmptyMetrics);
    }
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(MisParetoError::InvalidTolerance);
    }

    for (i, metric) in metrics.iter().enumerate() {
        if metrics[..i].contains(metric) {
            return Err(MisParetoError::DuplicateMetric { metric: *metric });
        }
    }

    let cohort_id = points[0].cohort_id.trim();
    if cohort_id.is_empty() || points.iter().any(|p| p.cohort_id.trim() != cohort_id) {
        return Err(MisParetoError::MixedCohorts);
    }
    for (i, point) in points.iter().enumerate() {
        if point.policy_id.trim().is_empty()
            || points[..i].iter().any(|earlier| earlier.policy_id == point.policy_id) {
            return Err(MisParetoError::DuplicateOrEmptyPolicyId);
        }
        if !valid_rate(point.normalized_quality_loss)
            || !valid_rate(point.infeasible_rate)
            || !valid_rate(point.target_miss_rate) {
            return Err(MisParetoError::InvalidOutcomeRates {
                policy_id: point.policy_id.clone(),
            });
        }
        if point.costs.len() != metrics.len() {
            return Err(MisParetoError::MismatchedCostMetrics {
                policy_id: point.policy_id.clone(),
            });
        }
        for (j, cost) in point.costs.iter().enumerate() {
            if point.costs[..j].iter().any(|prev| prev.metric == cost.metric)
                || !metrics.contains(&cost.metric) {
                return Err(MisParetoError::MismatchedCostMetrics {
                    policy_id: point.policy_id.clone(),
                });
            }
            if !cost.normalized_mean_amount.is_finite() || cost.normalized_mean_amount < 0.0 {
                return Err(MisParetoError::InvalidNormalizedCost {
                    policy_id: point.policy_id.clone(),
                    metric: cost.metric,
                });
            }
            match (cost.metric, cost.currency_code.as_deref()) {
                (MisCostMetric::MoneyMicrounits, Some(code))
                    if code.len() == 3 && code.bytes().all(|b| b.is_ascii_uppercase()) => {}
                (MisCostMetric::MoneyMicrounits, _) | (_, Some(_)) => {
                    return Err(MisParetoError::InvalidCurrency {
                        policy_id: point.policy_id.clone(),
                        metric: cost.metric,
                    });
                }
                _ => {}
            }
        }
        for metric in metrics {
            if !point.costs.iter().any(|cost| cost.metric == *metric) {
                return Err(MisParetoError::MismatchedCostMetrics {
                    policy_id: point.policy_id.clone(),
                });
            }
        }
    }

    if metrics.contains(&MisCostMetric::MoneyMicrounits) {
        let currencies = points.iter().map(|point| {
            point.costs.iter()
                .find(|cost| cost.metric == MisCostMetric::MoneyMicrounits)
                .and_then(|cost| cost.currency_code.as_deref())
        });
        let mut seen: Option<&str> = None;
        for currency in currencies {
            let Some(currency) = currency else { return Err(MisParetoError::CurrencyMismatch); };
            match seen {
                Some(expected) if expected != currency => return Err(MisParetoError::CurrencyMismatch),
                None => seen = Some(currency),
                _ => {}
            }
        }
    }

    let mut frontier = Vec::new();
    for candidate_index in 0..points.len() {
        let dominated = (0..points.len()).any(|other_index| {
            other_index != candidate_index
                && dominates(&points[other_index], &points[candidate_index], metrics, tolerance)
        });
        if !dominated {
            frontier.push(candidate_index);
        }
    }
    Ok(frontier)
}

fn valid_rate(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn dominates(
    a: &MisParetoPoint,
    b: &MisParetoPoint,
    metrics: &[MisCostMetric],
    tolerance: f64,
) -> bool {
    let scalar_a = [a.normalized_quality_loss, a.infeasible_rate, a.target_miss_rate];
    let scalar_b = [b.normalized_quality_loss, b.infeasible_rate, b.target_miss_rate];
    if scalar_a.iter().zip(scalar_b).any(|(x, y)| *x > y + tolerance) {
        return false;
    }
    let mut strictly_better = scalar_a.iter().zip(scalar_b).any(|(x, y)| *x < y - tolerance);
    for metric in metrics {
        let Some(x) = a.costs.iter().find(|c| c.metric == *metric) else { return false; };
        let Some(y) = b.costs.iter().find(|c| c.metric == *metric) else { return false; };
        if x.normalized_mean_amount > y.normalized_mean_amount + tolerance {
            return false;
        }
        strictly_better |= x.normalized_mean_amount < y.normalized_mean_amount - tolerance;
    }
    strictly_better
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cost(metric: MisCostMetric, value: f64) -> MisNormalizedCostValue {
        MisNormalizedCostValue { metric, normalized_mean_amount: value, currency_code: None }
    }

    fn point(id: &str, loss: f64, infeasible: f64, miss: f64, shots: f64, wall: f64) -> MisParetoPoint {
        MisParetoPoint {
            cohort_id: "qoblib-snapshot-1".to_owned(),
            policy_id: id.to_owned(),
            evidence_class: MisEvidenceClass::SyntheticArms,
            normalized_quality_loss: loss,
            infeasible_rate: infeasible,
            target_miss_rate: miss,
            costs: vec![
                cost(MisCostMetric::PhysicalShots, shots),
                cost(MisCostMetric::WallClockMillis, wall),
            ],
        }
    }

    fn metrics() -> [MisCostMetric; 2] {
        [MisCostMetric::PhysicalShots, MisCostMetric::WallClockMillis]
    }

    #[test]
    fn removes_only_points_dominated_on_all_dimensions() {
        let points = vec![
            point("balanced", 0.15, 0.0, 0.1, 0.6, 0.7),
            point("dominated", 0.20, 0.0, 0.1, 0.8, 0.9),
            point("quality-first", 0.05, 0.0, 0.0, 1.1, 0.8),
            point("cost-first", 0.25, 0.0, 0.2, 0.4, 0.5),
        ];
        assert_eq!(mis_cost_pareto_frontier(&points, &metrics(), 1e-9).unwrap(), vec![0, 2, 3]);
    }

    #[test]
    fn refuses_mismatched_cohorts_and_cost_dimensions() {
        let mut points = vec![
            point("a", 0.1, 0.0, 0.0, 0.5, 0.5),
            point("b", 0.2, 0.0, 0.0, 0.6, 0.6),
        ];
        points[1].cohort_id = "different-split".to_owned();
        assert_eq!(mis_cost_pareto_frontier(&points, &metrics(), 0.0), Err(MisParetoError::MixedCohorts));
        points[1].cohort_id = points[0].cohort_id.clone();
        points[1].costs.pop();
        assert_eq!(
            mis_cost_pareto_frontier(&points, &metrics(), 0.0),
            Err(MisParetoError::MismatchedCostMetrics { policy_id: "b".to_owned() })
        );
    }

    #[test]
    fn monetary_frontier_requires_same_explicit_currency() {
        let mut a = point("a", 0.1, 0.0, 0.0, 0.5, 0.5);
        let mut b = point("b", 0.2, 0.0, 0.0, 0.6, 0.6);
        a.costs = vec![MisNormalizedCostValue {
            metric: MisCostMetric::MoneyMicrounits,
            normalized_mean_amount: 0.3,
            currency_code: Some("USD".to_owned()),
        }];
        b.costs = vec![MisNormalizedCostValue {
            metric: MisCostMetric::MoneyMicrounits,
            normalized_mean_amount: 0.4,
            currency_code: Some("EUR".to_owned()),
        }];
        assert_eq!(
            mis_cost_pareto_frontier(&[a, b], &[MisCostMetric::MoneyMicrounits], 0.0),
            Err(MisParetoError::CurrencyMismatch)
        );
    }

    #[test]
    fn validates_rates_tolerance_and_duplicate_metrics() {
        let points = vec![point("a", 0.1, 0.0, 0.0, 0.5, 0.5)];
        assert_eq!(
            mis_cost_pareto_frontier(&points, &[MisCostMetric::PhysicalShots, MisCostMetric::PhysicalShots], 0.0),
            Err(MisParetoError::DuplicateMetric { metric: MisCostMetric::PhysicalShots })
        );
        assert_eq!(
            mis_cost_pareto_frontier(&points, &metrics(), f64::NAN),
            Err(MisParetoError::InvalidTolerance)
        );
        let mut invalid = points[0].clone();
        invalid.target_miss_rate = 1.2;
        assert_eq!(
            mis_cost_pareto_frontier(&[invalid], &metrics(), 0.0),
            Err(MisParetoError::InvalidOutcomeRates { policy_id: "a".to_owned() })
        );
    }
}
