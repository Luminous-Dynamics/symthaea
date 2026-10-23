//! Cohort-level accounting for reusable fixed setup and per-instance cost.
//!
//! Hybrid quantum workflows may pay a one-time cost to generate a reusable expectation
//! table, compile/calibrate a circuit family, or create shared preprocessing artifacts.
//! This module keeps fixed setup and marginal per-instance resource use separate and
//! computes cohort totals and a transparent break-even size for each resource dimension.
//!
//! It does not combine unlike dimensions into one score. Compare shots with shots,
//! wall time with wall time, and money only in the same explicitly named currency.
//! Overlapping time categories (e.g. wall time and stage timings) are separate views,
//! not additive terms. Cost profiles are caller-supplied; this code does not measure or
//! attest their provenance.

use core::fmt;
use std::collections::BTreeMap;

/// An independently reported cost/resource dimension.
///
/// Values are raw integer units. The unit is part of the variant name so incompatible
/// dimensions are not silently added together.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MisAmortizedMetric {
    /// Candidate observations (not interchangeable with physical shots).
    CandidateObservations,
    /// Unique physical device shots.
    PhysicalShots,
    /// Submitted/executed circuit batches.
    CircuitBatches,
    /// Compilation/transpilation time, milliseconds.
    CompilationMillis,
    /// Calibration overhead, milliseconds.
    CalibrationMillis,
    /// Classical preprocessing time, milliseconds.
    PreprocessingMillis,
    /// Classical solver/inference execution time, milliseconds.
    ClassicalComputeMillis,
    /// Queue/wait time, milliseconds.
    QueueMillis,
    /// End-to-end elapsed wall time, milliseconds.
    WallClockMillis,
    /// Active quantum processing unit execution time, milliseconds.
    QpuExecutionMillis,
    /// Energy, millijoules, when measured or explicitly estimated.
    EnergyMillijoules,
    /// Direct cost in millionths of a currency unit; requires an ISO-style uppercase
    /// three-letter currency code in MisAmortizedCostEntry.currency_code.
    MoneyMicrounits,
}

impl MisAmortizedMetric {
    /// Stable machine-readable label for manifests and receipts.
    pub const fn label(self) -> &'static str {
        match self {
            Self::CandidateObservations => "candidate_observations",
            Self::PhysicalShots => "physical_shots",
            Self::CircuitBatches => "circuit_batches",
            Self::CompilationMillis => "compilation_ms",
            Self::CalibrationMillis => "calibration_ms",
            Self::PreprocessingMillis => "preprocessing_ms",
            Self::ClassicalComputeMillis => "classical_compute_ms",
            Self::QueueMillis => "queue_ms",
            Self::WallClockMillis => "wall_clock_ms",
            Self::QpuExecutionMillis => "qpu_execution_ms",
            Self::EnergyMillijoules => "energy_mj",
            Self::MoneyMicrounits => "money_microunits",
        }
    }
}

/// Fixed plus marginal cost for one resource dimension of one policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MisAmortizedCostEntry {
    /// Cost dimension; each dimension is compared independently.
    pub metric: MisAmortizedMetric,
    /// Required only for MoneyMicrounits. All compared money entries must use the same
    /// uppercase three-letter currency code. Non-money entries must set this to None.
    pub currency_code: Option<String>,
    /// One-time setup amount attributable to the declared method/configuration.
    ///
    /// Use Some(0) when measured/configured as zero. None means unrecorded and fails closed.
    pub fixed_setup_units: Option<u64>,
    /// Marginal amount consumed for each completed instance served by the method.
    ///
    /// Use Some(0) when measured/configured as zero. None means unrecorded and fails closed.
    pub marginal_units_per_instance: Option<u64>,
}

/// Cost profile for one method over a reusable cohort.
///
/// Fixed setup must correspond to a declared amortization scope: for example one
/// fixed-angle/topology table serving a known family of graphs. Do not charge setup
/// once per graph if it was actually paid once for the whole cohort; do not omit it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MisAmortizedCostProfile {
    /// Stable policy/method identifier.
    pub arm_id: String,
    /// One entry for every resource dimension selected in the protocol.
    pub costs: Vec<MisAmortizedCostEntry>,
}

/// Per-resource result; no heterogeneous scalar aggregation is performed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MisAmortizedMetricResult {
    /// Independently compared dimension.
    pub metric: MisAmortizedMetric,
    /// Currency code, only for money.
    pub currency_code: Option<String>,
    /// Fixed setup cost of the candidate.
    pub candidate_fixed_setup_units: u64,
    /// Candidate marginal cost per instance.
    pub candidate_marginal_units_per_instance: u64,
    /// Total candidate cost for the requested cohort size.
    pub candidate_total_units: u128,
    /// Fixed setup cost of the baseline.
    pub baseline_fixed_setup_units: u64,
    /// Baseline marginal cost per instance.
    pub baseline_marginal_units_per_instance: u64,
    /// Total baseline cost for the requested cohort size.
    pub baseline_total_units: u128,
    /// True if candidate total is less than or equal to baseline total at this cohort size.
    pub candidate_no_more_expensive_at_requested_size: bool,
    /// For a method with no-greater marginal cost, the smallest positive cohort size at
    /// which its total cost is no greater than the baseline total. None means its marginal
    /// cost is higher, so larger cohorts do not produce a durable amortization win.
    /// This is a cost crossover, not evidence of solution-quality equivalence.
    pub candidate_break_even_cohort_size: Option<u64>,
}

/// Result of comparing a candidate and baseline at one requested cohort size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MisAmortizedCostComparison {
    /// Candidate method identifier.
    pub candidate_arm_id: String,
    /// Baseline method identifier.
    pub baseline_arm_id: String,
    /// Number of graph instances/jobs served by the shared setup costs.
    pub cohort_size: u64,
    /// One result per cost dimension, sorted by stable metric enum order.
    pub metrics: Vec<MisAmortizedMetricResult>,
}

/// Validation or accounting error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MisAmortizedCostError {
    /// An arm identifier is blank.
    EmptyArmId,
    /// A profile has no cost dimensions.
    EmptyCostProfile { arm_id: String },
    /// The same resource dimension appears more than once in a profile.
    DuplicateMetric {
        /// Arm that contains the duplicate.
        arm_id: String,
        /// Duplicated dimension.
        metric: MisAmortizedMetric,
    },
    /// One or both profiles are missing a resource dimension present in the other.
    MetricSetMismatch,
    /// Currency metadata is missing, malformed, or attached to a non-money metric.
    InvalidCurrencyMetadata {
        /// Arm containing invalid metadata.
        arm_id: String,
        /// Metric with invalid metadata.
        metric: MisAmortizedMetric,
    },
    /// Both arms use different currency codes for monetary comparison.
    CurrencyMismatch {
        /// Candidate currency.
        candidate: String,
        /// Baseline currency.
        baseline: String,
    },
    /// A configured cost dimension has a missing fixed setup measurement.
    MissingFixedSetup {
        /// Arm containing the missing value.
        arm_id: String,
        /// Metric with the missing value.
        metric: MisAmortizedMetric,
    },
    /// A configured cost dimension has a missing marginal measurement.
    MissingMarginalCost {
        /// Arm containing the missing value.
        arm_id: String,
        /// Metric with the missing value.
        metric: MisAmortizedMetric,
    },
    /// Candidate and baseline identifiers must differ.
    SameArmCompared,
    /// Cohort size must be positive.
    ZeroCohortSize,
    /// Arithmetic exceeded the supported total representation.
    ArithmeticOverflow {
        /// Metric whose total overflowed.
        metric: MisAmortizedMetric,
    },
}

impl fmt::Display for MisAmortizedCostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyArmId => write!(f, "arm identifier must not be empty"),
            Self::EmptyCostProfile { arm_id } => write!(f, "cost profile for {arm_id:?} has no metrics"),
            Self::DuplicateMetric { arm_id, metric } => {
                write!(f, "duplicate metric {} in arm {arm_id:?}", metric.label())
            }
            Self::MetricSetMismatch => write!(f, "candidate and baseline must record the same cost dimensions"),
            Self::InvalidCurrencyMetadata { arm_id, metric } => {
                write!(f, "invalid currency metadata for metric {} in arm {arm_id:?}", metric.label())
            }
            Self::CurrencyMismatch { candidate, baseline } => {
                write!(f, "money units are incompatible: candidate currency {candidate}, baseline currency {baseline}")
            }
            Self::MissingFixedSetup { arm_id, metric } => {
                write!(f, "missing fixed setup cost for metric {} in arm {arm_id:?}", metric.label())
            }
            Self::MissingMarginalCost { arm_id, metric } => {
                write!(f, "missing marginal cost for metric {} in arm {arm_id:?}", metric.label())
            }
            Self::SameArmCompared => write!(f, "candidate and baseline identifiers must differ"),
            Self::ZeroCohortSize => write!(f, "cohort size must be positive"),
            Self::ArithmeticOverflow { metric } => {
                write!(f, "total cost arithmetic overflow for metric {}", metric.label())
            }
        }
    }
}

impl std::error::Error for MisAmortizedCostError {}

impl MisAmortizedCostProfile {
    /// Validate identity, duplicate dimensions and currency metadata.
    pub fn validate(&self) -> Result<(), MisAmortizedCostError> {
        if self.arm_id.trim().is_empty() {
            return Err(MisAmortizedCostError::EmptyArmId);
        }
        if self.costs.is_empty() {
            return Err(MisAmortizedCostError::EmptyCostProfile {
                arm_id: self.arm_id.clone(),
            });
        }
        let mut seen = std::collections::BTreeSet::new();
        for entry in &self.costs {
            if !seen.insert(entry.metric) {
                return Err(MisAmortizedCostError::DuplicateMetric {
                    arm_id: self.arm_id.clone(),
                    metric: entry.metric,
                });
            }
            match (entry.metric, entry.currency_code.as_deref()) {
                (MisAmortizedMetric::MoneyMicrounits, Some(code))
                    if code.len() == 3 && code.bytes().all(|byte| byte.is_ascii_uppercase()) => {}
                (MisAmortizedMetric::MoneyMicrounits, _) => {
                    return Err(MisAmortizedCostError::InvalidCurrencyMetadata {
                        arm_id: self.arm_id.clone(),
                        metric: entry.metric,
                    });
                }
                (_, None) => {}
                (_, Some(_)) => {
                    return Err(MisAmortizedCostError::InvalidCurrencyMetadata {
                        arm_id: self.arm_id.clone(),
                        metric: entry.metric,
                    });
                }
            }
        }
        Ok(())
    }

    fn indexed(&self) -> BTreeMap<MisAmortizedMetric, &MisAmortizedCostEntry> {
        self.costs.iter().map(|entry| (entry.metric, entry)).collect()
    }
}

/// Compare total costs at a cohort size and compute the fixed-cost payback point.
///
/// Each dimension is kept separate, and candidate/baseline profiles must declare the
/// same dimensions. No missing measurement defaults to zero. Totals are:
///
/// total(N) = fixed_setup + N * marginal_per_instance
///
/// A break-even size is reported only when candidate marginal cost is no greater than
/// baseline marginal cost. In that case it is the smallest positive N for which candidate
/// total cost is <= baseline total cost; setup savings count as break-even at N=1. If the
/// candidate's marginal cost is higher, no durable amortization win is reported, although
/// its total may still be lower for a small requested cohort.
///
/// This compares resource consumption only. It does not establish that solution quality,
/// feasibility, or success probability is equivalent between the two methods. Wall-clock
/// amortization assumes a fixed execution regime; when jobs run concurrently, measure
/// actual end-to-end latency for each cohort size rather than assuming a linear formula.
pub fn compare_mis_amortized_costs(
    candidate: &MisAmortizedCostProfile,
    baseline: &MisAmortizedCostProfile,
    cohort_size: u64,
) -> Result<MisAmortizedCostComparison, MisAmortizedCostError> {
    if cohort_size == 0 {
        return Err(MisAmortizedCostError::ZeroCohortSize);
    }
    candidate.validate()?;
    baseline.validate()?;
    if candidate.arm_id == baseline.arm_id {
        return Err(MisAmortizedCostError::SameArmCompared);
    }

    let candidate_by_metric = candidate.indexed();
    let baseline_by_metric = baseline.indexed();
    if candidate_by_metric.keys().ne(baseline_by_metric.keys()) {
        return Err(MisAmortizedCostError::MetricSetMismatch);
    }

    let mut metrics = Vec::with_capacity(candidate.costs.len());
    for (&metric, candidate_entry) in &candidate_by_metric {
        let baseline_entry = baseline_by_metric
            .get(&metric)
            .expect("metric-set equality was checked");

        if metric == MisAmortizedMetric::MoneyMicrounits {
            let candidate_currency = candidate_entry.currency_code.as_deref()
                .expect("profile validation checked money currency");
            let baseline_currency = baseline_entry.currency_code.as_deref()
                .expect("profile validation checked money currency");
            if candidate_currency != baseline_currency {
                return Err(MisAmortizedCostError::CurrencyMismatch {
                    candidate: candidate_currency.to_owned(),
                    baseline: baseline_currency.to_owned(),
                });
            }
        }

        let candidate_fixed = candidate_entry.fixed_setup_units.ok_or_else(|| {
            MisAmortizedCostError::MissingFixedSetup {
                arm_id: candidate.arm_id.clone(),
                metric,
            }
        })?;
        let candidate_marginal = candidate_entry.marginal_units_per_instance.ok_or_else(|| {
            MisAmortizedCostError::MissingMarginalCost {
                arm_id: candidate.arm_id.clone(),
                metric,
            }
        })?;
        let baseline_fixed = baseline_entry.fixed_setup_units.ok_or_else(|| {
            MisAmortizedCostError::MissingFixedSetup {
                arm_id: baseline.arm_id.clone(),
                metric,
            }
        })?;
        let baseline_marginal = baseline_entry.marginal_units_per_instance.ok_or_else(|| {
            MisAmortizedCostError::MissingMarginalCost {
                arm_id: baseline.arm_id.clone(),
                metric,
            }
        })?;

        let candidate_total = u128::from(candidate_fixed)
            .checked_add(
                u128::from(candidate_marginal)
                    .checked_mul(u128::from(cohort_size))
                    .ok_or(MisAmortizedCostError::ArithmeticOverflow { metric })?,
            )
            .ok_or(MisAmortizedCostError::ArithmeticOverflow { metric })?;
        let baseline_total = u128::from(baseline_fixed)
            .checked_add(
                u128::from(baseline_marginal)
                    .checked_mul(u128::from(cohort_size))
                    .ok_or(MisAmortizedCostError::ArithmeticOverflow { metric })?,
            )
            .ok_or(MisAmortizedCostError::ArithmeticOverflow { metric })?;

        let break_even = if candidate_marginal > baseline_marginal {
            None
        } else if candidate_fixed <= baseline_fixed {
            Some(1)
        } else if candidate_marginal == baseline_marginal {
            // Higher fixed cost with no marginal savings is never amortized away.
            None
        } else {
            let setup_premium = u128::from(candidate_fixed - baseline_fixed);
            let marginal_saving = u128::from(baseline_marginal - candidate_marginal);
            let cohort = setup_premium
                .checked_add(marginal_saving - 1)
                .ok_or(MisAmortizedCostError::ArithmeticOverflow { metric })?
                / marginal_saving;
            Some(u64::try_from(cohort).map_err(|_| {
                MisAmortizedCostError::ArithmeticOverflow { metric }
            })?)
        };

        metrics.push(MisAmortizedMetricResult {
            metric,
            currency_code: candidate_entry.currency_code.clone(),
            candidate_fixed_setup_units: candidate_fixed,
            candidate_marginal_units_per_instance: candidate_marginal,
            candidate_total_units: candidate_total,
            baseline_fixed_setup_units: baseline_fixed,
            baseline_marginal_units_per_instance: baseline_marginal,
            baseline_total_units: baseline_total,
            candidate_no_more_expensive_at_requested_size: candidate_total <= baseline_total,
            candidate_break_even_cohort_size: break_even,
        });
    }

    Ok(MisAmortizedCostComparison {
        candidate_arm_id: candidate.arm_id.clone(),
        baseline_arm_id: baseline.arm_id.clone(),
        cohort_size,
        metrics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(
        metric: MisAmortizedMetric,
        fixed: u64,
        marginal: u64,
    ) -> MisAmortizedCostEntry {
        MisAmortizedCostEntry {
            metric,
            currency_code: None,
            fixed_setup_units: Some(fixed),
            marginal_units_per_instance: Some(marginal),
        }
    }

    fn profile(arm_id: &str, costs: Vec<MisAmortizedCostEntry>) -> MisAmortizedCostProfile {
        MisAmortizedCostProfile { arm_id: arm_id.to_owned(), costs }
    }

    #[test]
    fn computes_fixed_setup_break_even_for_reused_expectation_table() {
        let candidate = profile(
            "qaoa-informed",
            vec![entry(MisAmortizedMetric::WallClockMillis, 1_001, 2)],
        );
        let baseline = profile(
            "classical-portfolio",
            vec![entry(MisAmortizedMetric::WallClockMillis, 1, 12)],
        );
        let at_100 = compare_mis_amortized_costs(&candidate, &baseline, 100).unwrap();
        let result = &at_100.metrics[0];
        assert_eq!(result.candidate_total_units, 1_201);
        assert_eq!(result.baseline_total_units, 1_201);
        assert!(result.candidate_no_more_expensive_at_requested_size);
        assert_eq!(result.candidate_break_even_cohort_size, Some(100));

        let at_99 = compare_mis_amortized_costs(&candidate, &baseline, 99).unwrap();
        assert!(!at_99.metrics[0].candidate_no_more_expensive_at_requested_size);
    }

    #[test]
    fn reports_no_durable_break_even_when_candidate_marginal_is_higher() {
        let candidate = profile(
            "candidate",
            vec![entry(MisAmortizedMetric::PhysicalShots, 0, 11)],
        );
        let baseline = profile(
            "baseline",
            vec![entry(MisAmortizedMetric::PhysicalShots, 100, 10)],
        );
        let result = compare_mis_amortized_costs(&candidate, &baseline, 1).unwrap();
        assert!(result.metrics[0].candidate_no_more_expensive_at_requested_size);
        assert_eq!(result.metrics[0].candidate_break_even_cohort_size, None);
        assert!(!compare_mis_amortized_costs(&candidate, &baseline, 101).unwrap()
            .metrics[0].candidate_no_more_expensive_at_requested_size);
    }

    #[test]
    fn candidate_with_lower_setup_and_marginal_cost_breaks_even_at_one() {
        let candidate = profile("candidate", vec![entry(MisAmortizedMetric::CircuitBatches, 1, 1)]);
        let baseline = profile("baseline", vec![entry(MisAmortizedMetric::CircuitBatches, 5, 2)]);
        let result = compare_mis_amortized_costs(&candidate, &baseline, 1).unwrap();
        assert_eq!(result.metrics[0].candidate_break_even_cohort_size, Some(1));
        assert!(result.metrics[0].candidate_no_more_expensive_at_requested_size);
    }

    #[test]
    fn missing_configured_measurement_never_defaults_to_zero() {
        let candidate = profile(
            "candidate",
            vec![MisAmortizedCostEntry {
                metric: MisAmortizedMetric::QpuExecutionMillis,
                currency_code: None,
                fixed_setup_units: None,
                marginal_units_per_instance: Some(0),
            }],
        );
        let baseline = profile(
            "baseline",
            vec![entry(MisAmortizedMetric::QpuExecutionMillis, 0, 100)],
        );
        assert_eq!(
            compare_mis_amortized_costs(&candidate, &baseline, 10),
            Err(MisAmortizedCostError::MissingFixedSetup {
                arm_id: "candidate".to_owned(),
                metric: MisAmortizedMetric::QpuExecutionMillis,
            })
        );
    }

    #[test]
    fn rejects_currency_mismatch_and_nonmoney_currency_metadata() {
        let mut candidate_money = entry(MisAmortizedMetric::MoneyMicrounits, 10, 2);
        candidate_money.currency_code = Some("USD".to_owned());
        let mut baseline_money = entry(MisAmortizedMetric::MoneyMicrounits, 5, 1);
        baseline_money.currency_code = Some("EUR".to_owned());
        let candidate = profile("candidate", vec![candidate_money]);
        let baseline = profile("baseline", vec![baseline_money]);
        assert_eq!(
            compare_mis_amortized_costs(&candidate, &baseline, 10),
            Err(MisAmortizedCostError::CurrencyMismatch {
                candidate: "USD".to_owned(),
                baseline: "EUR".to_owned(),
            })
        );

        let invalid = profile(
            "invalid",
            vec![MisAmortizedCostEntry {
                metric: MisAmortizedMetric::WallClockMillis,
                currency_code: Some("USD".to_owned()),
                fixed_setup_units: Some(1),
                marginal_units_per_instance: Some(1),
            }],
        );
        assert_eq!(
            invalid.validate(),
            Err(MisAmortizedCostError::InvalidCurrencyMetadata {
                arm_id: "invalid".to_owned(),
                metric: MisAmortizedMetric::WallClockMillis,
            })
        );
    }

    #[test]
    fn rejects_duplicate_metrics_mismatched_metric_sets_and_same_arm() {
        let duplicated = profile(
            "duplicate",
            vec![
                entry(MisAmortizedMetric::PhysicalShots, 1, 2),
                entry(MisAmortizedMetric::PhysicalShots, 1, 2),
            ],
        );
        assert_eq!(
            duplicated.validate(),
            Err(MisAmortizedCostError::DuplicateMetric {
                arm_id: "duplicate".to_owned(),
                metric: MisAmortizedMetric::PhysicalShots,
            })
        );
        let candidate = profile("candidate", vec![entry(MisAmortizedMetric::PhysicalShots, 0, 1)]);
        let baseline = profile("baseline", vec![entry(MisAmortizedMetric::QpuExecutionMillis, 0, 1)]);
        assert_eq!(compare_mis_amortized_costs(&candidate, &baseline, 2), Err(MisAmortizedCostError::MetricSetMismatch));
        assert_eq!(compare_mis_amortized_costs(&candidate, &candidate, 2), Err(MisAmortizedCostError::SameArmCompared));
    }

    #[test]
    fn result_is_independent_of_input_metric_order_and_keeps_dimensions_separate() {
        let candidate = profile(
            "candidate",
            vec![
                entry(MisAmortizedMetric::QpuExecutionMillis, 100, 2),
                entry(MisAmortizedMetric::WallClockMillis, 300, 4),
            ],
        );
        let baseline = profile(
            "baseline",
            vec![
                entry(MisAmortizedMetric::WallClockMillis, 100, 30),
                entry(MisAmortizedMetric::QpuExecutionMillis, 0, 0),
            ],
        );
        let result = compare_mis_amortized_costs(&candidate, &baseline, 10).unwrap();
        assert_eq!(result.metrics.len(), 2);
        assert_eq!(result.metrics[0].metric, MisAmortizedMetric::QpuExecutionMillis);
        assert_eq!(result.metrics[1].metric, MisAmortizedMetric::WallClockMillis);
        assert_eq!(result.metrics[0].candidate_total_units, 120);
        assert_eq!(result.metrics[0].baseline_total_units, 0);
        assert!(!result.metrics[0].candidate_no_more_expensive_at_requested_size);
        assert_eq!(result.metrics[1].candidate_total_units, 340);
        assert_eq!(result.metrics[1].baseline_total_units, 400);
        assert!(result.metrics[1].candidate_no_more_expensive_at_requested_size);
    }

    #[test]
    fn rejects_zero_cohort_and_empty_or_blank_profiles() {
        let candidate = profile("candidate", vec![entry(MisAmortizedMetric::PhysicalShots, 0, 0)]);
        let baseline = profile("baseline", vec![entry(MisAmortizedMetric::PhysicalShots, 0, 0)]);
        assert_eq!(compare_mis_amortized_costs(&candidate, &baseline, 0), Err(MisAmortizedCostError::ZeroCohortSize));
        assert_eq!(profile("none", vec![]).validate(), Err(MisAmortizedCostError::EmptyCostProfile { arm_id: "none".to_owned() }));
        assert_eq!(profile("  ", vec![entry(MisAmortizedMetric::PhysicalShots, 0, 0)]).validate(), Err(MisAmortizedCostError::EmptyArmId));
    }

    #[test]
    fn higher_fixed_cost_without_marginal_savings_never_breaks_even() {
        let candidate = profile(
            "candidate",
            vec![entry(MisAmortizedMetric::CompilationMillis, 101, 5)],
        );
        let baseline = profile(
            "baseline",
            vec![entry(MisAmortizedMetric::CompilationMillis, 100, 5)],
        );
        let result = compare_mis_amortized_costs(&candidate, &baseline, 10).unwrap();
        assert!(!result.metrics[0].candidate_no_more_expensive_at_requested_size);
        assert_eq!(result.metrics[0].candidate_break_even_cohort_size, None);
    }

}
