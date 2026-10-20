#![forbid(unsafe_code)]

//! Evidence-first benchmark metrics for the AtomFab research track.
//!
//! This crate intentionally does not implement an atomistic solver, interatomic
//! potential, dataset loader, or instrument driver. It supplies small, auditable
//! metric primitives so research adapters can share validation and reporting
//! behavior. Metrics are not evidence that a physical process works.

pub mod benchmark_registry;
pub mod campaign;
pub mod contract;

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

/// One scalar prediction paired with its reference value, in identical units.
#[derive(Debug, Clone, Copy)]
pub struct ScalarComparison {
    pub predicted: f64,
    pub reference: f64,
}

/// Error metrics for a set of predictions against reference values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorSummary {
    pub count: usize,
    pub mae: f64,
    pub rmse: f64,
    pub max_absolute_error: f64,
}

/// Pairwise agreement between predicted and reference energetic orderings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrderingSummary {
    /// Number of non-tied reference pairs evaluated.
    pub comparable_pairs: usize,
    /// Number of those pairs whose predicted order matched the reference.
    pub correctly_ordered_pairs: usize,
    /// Pairwise ordering accuracy in [0, 1].
    pub accuracy: f64,
}

/// A finite-prefix, top-weighted overlap score for two candidate rankings.
///
/// This is an explicitly normalized RBO-style metric, not a claim to reproduce
/// every extrapolated RBO variant. The persistence parameter controls how
/// strongly shallow prefixes dominate the score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopWeightedOverlapSummary {
    /// Number of prefix depths evaluated.
    pub evaluated_depths: usize,
    /// Normalized weighted prefix overlap in [0, 1].
    pub normalized_score: f64,
    /// Persistence weight, strictly between zero and one.
    pub persistence: f64,
}

/// One process-feasibility prediction compared with an independent reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeasibilityComparison {
    pub predicted_feasible: bool,
    pub reference_feasible: bool,
}

/// Safety-relevant process-feasibility classification metrics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeasibilitySummary {
    pub sample_count: usize,
    pub predicted_feasible_count: usize,
    pub false_feasible_count: usize,
    pub missed_feasible_count: usize,
    /// False-feasible predictions divided by all predicted-feasible cases.
    /// None means the system accepted no cases; it does not mean a zero rate.
    pub false_feasible_rate: Option<f64>,
}

/// One prediction interval and the observed reference value.
#[derive(Debug, Clone, Copy)]
pub struct IntervalComparison {
    pub lower: f64,
    pub upper: f64,
    pub observed: f64,
}

/// Empirical metrics for supplied prediction intervals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntervalSummary {
    pub count: usize,
    /// Fraction of observations contained in their intervals, in [0, 1].
    pub empirical_coverage: f64,
    /// Mean interval width in the supplied measurement units.
    pub mean_width: f64,
}

/// Invalid inputs or metrics that cannot be represented as finite values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkError {
    EmptyInput,
    NonFiniteValue {
        index: usize,
        field: &'static str,
    },
    InvalidInterval {
        index: usize,
    },
    InvalidTieTolerance,
    InvalidPersistence,
    RankingLengthMismatch,
    DuplicateRankingId {
        ranking: &'static str,
        index: usize,
    },
    NoComparablePairs,
    ZeroReferenceStructures,
    InvalidRecoveryCounts,
    NonFiniteMetric,
    ZeroEvaluationCount,
}

impl fmt::Display for BenchmarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "benchmark input must not be empty"),
            Self::NonFiniteValue { index, field } => {
                write!(f, "sample {index} has a non-finite {field} value")
            }
            Self::InvalidInterval { index } => {
                write!(f, "sample {index} has a lower interval bound above its upper bound")
            }
            Self::InvalidTieTolerance => write!(f, "tie tolerance must be finite and non-negative"),
            Self::InvalidPersistence => write!(f, "ranking persistence must be finite and strictly between zero and one"),
            Self::RankingLengthMismatch => write!(f, "predicted and reference rankings must have the same non-zero length"),
            Self::DuplicateRankingId { ranking, index } => write!(f, "{ranking} ranking repeats an identifier at index {index}"),
            Self::NoComparablePairs => write!(f, "no non-tied reference pairs are available"),
            Self::ZeroReferenceStructures => write!(f, "reference structure count must be greater than zero"),
            Self::InvalidRecoveryCounts => write!(f, "recovered structure count cannot exceed reference count"),
            Self::NonFiniteMetric => write!(f, "metric calculation produced a non-finite value"),
            Self::ZeroEvaluationCount => {
                write!(f, "evaluation counts must be greater than zero")
            }
        }
    }
}

impl Error for BenchmarkError {}

/// Compute prediction errors, rejecting empty input and non-finite calculations.
///
/// The predicted and reference values must use the same units and reference
/// method. Report results per material family as well as in aggregate; an
/// aggregate error alone is not evidence of transferability.
pub fn summarize_errors(samples: &[ScalarComparison]) -> Result<ErrorSummary, BenchmarkError> {
    if samples.is_empty() {
        return Err(BenchmarkError::EmptyInput);
    }

    let mut absolute_total = 0.0;
    let mut squared_total = 0.0;
    let mut max_absolute_error: f64 = 0.0;

    for (index, sample) in samples.iter().enumerate() {
        if !sample.predicted.is_finite() {
            return Err(BenchmarkError::NonFiniteValue {
                index,
                field: "predicted",
            });
        }
        if !sample.reference.is_finite() {
            return Err(BenchmarkError::NonFiniteValue {
                index,
                field: "reference",
            });
        }

        let error = sample.predicted - sample.reference;
        let absolute_error = error.abs();
        let squared_error = error * error;
        if !error.is_finite() || !squared_error.is_finite() {
            return Err(BenchmarkError::NonFiniteMetric);
        }

        absolute_total += absolute_error;
        squared_total += squared_error;
        max_absolute_error = max_absolute_error.max(absolute_error);
        if !absolute_total.is_finite() || !squared_total.is_finite() {
            return Err(BenchmarkError::NonFiniteMetric);
        }
    }

    let count = samples.len();
    let divisor = count as f64;
    let mae = absolute_total / divisor;
    let rmse = (squared_total / divisor).sqrt();
    if !mae.is_finite() || !rmse.is_finite() || !max_absolute_error.is_finite() {
        return Err(BenchmarkError::NonFiniteMetric);
    }

    Ok(ErrorSummary {
        count,
        mae,
        rmse,
        max_absolute_error,
    })
}

/// Measure empirical coverage and mean width of a set of prediction intervals.
///
/// Coverage is descriptive, not a calibration guarantee. The nominal coverage
/// level and the dataset split must be declared separately by the experiment.
pub fn summarize_intervals(
    samples: &[IntervalComparison],
) -> Result<IntervalSummary, BenchmarkError> {
    if samples.is_empty() {
        return Err(BenchmarkError::EmptyInput);
    }

    let mut covered = 0usize;
    let mut width_total = 0.0;

    for (index, sample) in samples.iter().enumerate() {
        for (field, value) in [
            ("lower", sample.lower),
            ("upper", sample.upper),
            ("observed", sample.observed),
        ] {
            if !value.is_finite() {
                return Err(BenchmarkError::NonFiniteValue { index, field });
            }
        }
        if sample.lower > sample.upper {
            return Err(BenchmarkError::InvalidInterval { index });
        }

        let width = sample.upper - sample.lower;
        if !width.is_finite() {
            return Err(BenchmarkError::NonFiniteMetric);
        }
        width_total += width;
        if !width_total.is_finite() {
            return Err(BenchmarkError::NonFiniteMetric);
        }

        if sample.observed >= sample.lower && sample.observed <= sample.upper {
            covered += 1;
        }
    }

    let count = samples.len();
    let summary = IntervalSummary {
        count,
        empirical_coverage: covered as f64 / count as f64,
        mean_width: width_total / count as f64,
    };
    if !summary.empirical_coverage.is_finite() || !summary.mean_width.is_finite() {
        return Err(BenchmarkError::NonFiniteMetric);
    }
    Ok(summary)
}

/// Compare pairwise predicted energies to reference energies.
///
/// Pairs tied within the non-negative reference tie tolerance are excluded.
/// A predicted tie for a non-tied reference pair counts as incorrect. This
/// measures energetic ordering only; it does not establish structural recovery
/// or physical validity.
pub fn pairwise_ordering_accuracy(
    samples: &[ScalarComparison],
    tie_tolerance: f64,
) -> Result<OrderingSummary, BenchmarkError> {
    if samples.is_empty() {
        return Err(BenchmarkError::EmptyInput);
    }
    if !tie_tolerance.is_finite() || tie_tolerance < 0.0 {
        return Err(BenchmarkError::InvalidTieTolerance);
    }
    for (index, sample) in samples.iter().enumerate() {
        if !sample.predicted.is_finite() {
            return Err(BenchmarkError::NonFiniteValue { index, field: "predicted" });
        }
        if !sample.reference.is_finite() {
            return Err(BenchmarkError::NonFiniteValue { index, field: "reference" });
        }
    }

    let mut comparable_pairs = 0usize;
    let mut correctly_ordered_pairs = 0usize;
    for i in 0..samples.len() {
        for j in (i + 1)..samples.len() {
            let reference_delta = samples[i].reference - samples[j].reference;
            let predicted_delta = samples[i].predicted - samples[j].predicted;
            if !reference_delta.is_finite() || !predicted_delta.is_finite() {
                return Err(BenchmarkError::NonFiniteMetric);
            }
            if reference_delta.abs() <= tie_tolerance {
                continue;
            }

            comparable_pairs += 1;
            let predicted_is_tied = predicted_delta.abs() <= tie_tolerance;
            let agrees = !predicted_is_tied
                && ((reference_delta < 0.0 && predicted_delta < 0.0)
                    || (reference_delta > 0.0 && predicted_delta > 0.0));
            if agrees {
                correctly_ordered_pairs += 1;
            }
        }
    }

    if comparable_pairs == 0 {
        return Err(BenchmarkError::NoComparablePairs);
    }

    Ok(OrderingSummary {
        comparable_pairs,
        correctly_ordered_pairs,
        accuracy: correctly_ordered_pairs as f64 / comparable_pairs as f64,
    })
}

/// Measure dangerous false-feasible predictions against independent labels.
///
/// The rate denominator is the number of cases the system called feasible.
/// If it predicts no feasible cases, the rate is None rather than zero, so a
/// system that abstains on everything cannot appear perfectly safe by default.
pub fn summarize_feasibility(
    samples: &[FeasibilityComparison],
) -> Result<FeasibilitySummary, BenchmarkError> {
    if samples.is_empty() {
        return Err(BenchmarkError::EmptyInput);
    }

    let predicted_feasible_count = samples.iter().filter(|s| s.predicted_feasible).count();
    let false_feasible_count = samples
        .iter()
        .filter(|s| s.predicted_feasible && !s.reference_feasible)
        .count();
    let missed_feasible_count = samples
        .iter()
        .filter(|s| !s.predicted_feasible && s.reference_feasible)
        .count();
    let false_feasible_rate = if predicted_feasible_count == 0 {
        None
    } else {
        Some(false_feasible_count as f64 / predicted_feasible_count as f64)
    };

    Ok(FeasibilitySummary {
        sample_count: samples.len(),
        predicted_feasible_count,
        false_feasible_count,
        missed_feasible_count,
        false_feasible_rate,
    })
}

/// Compute normalized finite-prefix top-weighted overlap for ranked candidate IDs.
///
/// Both rankings must contain the same number of unique candidate identifiers.
/// At each depth, overlap is the fraction of top-depth candidates shared by both
/// rankings. The score is the persistence-weighted average across depths using
/// weights proportional to `(1 - p) * p^(depth - 1)`. Values near one indicate
/// similar top-ranked lists. Larger persistence values give deeper prefixes
/// more weight. The score is descriptive and does not establish physical
/// validity or reproduce the extrapolated form of every published RBO metric.
pub fn top_weighted_ranking_overlap(
    predicted_ranking: &[u64],
    reference_ranking: &[u64],
    persistence: f64,
) -> Result<TopWeightedOverlapSummary, BenchmarkError> {
    if predicted_ranking.is_empty() || reference_ranking.is_empty() {
        return Err(BenchmarkError::EmptyInput);
    }
    if predicted_ranking.len() != reference_ranking.len() {
        return Err(BenchmarkError::RankingLengthMismatch);
    }
    if !persistence.is_finite() || !(0.0..1.0).contains(&persistence) {
        return Err(BenchmarkError::InvalidPersistence);
    }

    for (ranking_name, ranking) in [
        ("predicted", predicted_ranking),
        ("reference", reference_ranking),
    ] {
        let mut seen = BTreeSet::new();
        for (index, identifier) in ranking.iter().copied().enumerate() {
            if !seen.insert(identifier) {
                return Err(BenchmarkError::DuplicateRankingId {
                    ranking: ranking_name,
                    index,
                });
            }
        }
    }

    let mut predicted_prefix = BTreeSet::new();
    let mut reference_prefix = BTreeSet::new();
    let mut weighted_overlap = 0.0;
    let mut total_weight = 0.0;
    for depth in 1..=predicted_ranking.len() {
        predicted_prefix.insert(predicted_ranking[depth - 1]);
        reference_prefix.insert(reference_ranking[depth - 1]);
        let shared = predicted_prefix.intersection(&reference_prefix).count();
        let prefix_overlap = shared as f64 / depth as f64;
        let weight = (1.0 - persistence) * persistence.powf((depth - 1) as f64);
        weighted_overlap += weight * prefix_overlap;
        total_weight += weight;
        if !weighted_overlap.is_finite() || !total_weight.is_finite() {
            return Err(BenchmarkError::NonFiniteMetric);
        }
    }

    if total_weight <= 0.0 {
        return Err(BenchmarkError::NonFiniteMetric);
    }
    let normalized_score = weighted_overlap / total_weight;
    if !normalized_score.is_finite() || !(0.0..=1.0).contains(&normalized_score) {
        return Err(BenchmarkError::NonFiniteMetric);
    }
    Ok(TopWeightedOverlapSummary {
        evaluated_depths: predicted_ranking.len(),
        normalized_score,
        persistence,
    })
}

/// Fraction of known reference structures recovered by a search.
///
/// Structure identity and the definition of a match are domain-specific and
/// must be established by the caller before passing these counts.
pub fn structure_recovery_fraction(
    reference_structures: u64,
    recovered_structures: u64,
) -> Result<f64, BenchmarkError> {
    if reference_structures == 0 {
        return Err(BenchmarkError::ZeroReferenceStructures);
    }
    if recovered_structures > reference_structures {
        return Err(BenchmarkError::InvalidRecoveryCounts);
    }
    Ok(recovered_structures as f64 / reference_structures as f64)
}

/// Compute the acceleration factor for two campaigns that reached the same
/// preregistered target on the same benchmark: reference queries divided by
/// strategy queries. The caller must establish target equivalence and matching
/// effective query costs; this function cannot prove either.
pub fn acceleration_factor(
    reference_evaluations: u64,
    strategy_evaluations: u64,
) -> Result<f64, BenchmarkError> {
    if reference_evaluations == 0 || strategy_evaluations == 0 {
        return Err(BenchmarkError::ZeroEvaluationCount);
    }

    let factor = reference_evaluations as f64 / strategy_evaluations as f64;
    if !factor.is_finite() {
        return Err(BenchmarkError::NonFiniteMetric);
    }
    Ok(factor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_errors_match_hand_calculated_values() {
        let summary = summarize_errors(&[
            ScalarComparison {
                predicted: 1.0,
                reference: 1.0,
            },
            ScalarComparison {
                predicted: 3.0,
                reference: 1.0,
            },
        ])
        .expect("finite non-empty sample should summarize");

        assert_eq!(summary.count, 2);
        assert!((summary.mae - 1.0).abs() < 1e-12);
        assert!((summary.rmse - 2.0_f64.sqrt()).abs() < 1e-12);
        assert!((summary.max_absolute_error - 2.0).abs() < 1e-12);
    }

    #[test]
    fn scalar_errors_reject_empty_and_non_finite_input() {
        assert_eq!(summarize_errors(&[]), Err(BenchmarkError::EmptyInput));
        assert_eq!(
            summarize_errors(&[ScalarComparison {
                predicted: f64::NAN,
                reference: 0.0,
            }]),
            Err(BenchmarkError::NonFiniteValue {
                index: 0,
                field: "predicted",
            })
        );
    }

    #[test]
    fn interval_metrics_report_coverage_and_width() {
        let summary = summarize_intervals(&[
            IntervalComparison {
                lower: 0.0,
                upper: 2.0,
                observed: 1.0,
            },
            IntervalComparison {
                lower: 0.0,
                upper: 1.0,
                observed: 2.0,
            },
        ])
        .expect("valid intervals should summarize");

        assert_eq!(summary.count, 2);
        assert!((summary.empirical_coverage - 0.5).abs() < 1e-12);
        assert!((summary.mean_width - 1.5).abs() < 1e-12);
    }

    #[test]
    fn interval_metrics_reject_invalid_intervals() {
        assert_eq!(
            summarize_intervals(&[IntervalComparison {
                lower: 2.0,
                upper: 1.0,
                observed: 1.5,
            }]),
            Err(BenchmarkError::InvalidInterval { index: 0 })
        );
    }

    #[test]
    fn feasibility_metrics_expose_false_accepts_and_missed_opportunities() {
        let summary = summarize_feasibility(&[
            FeasibilityComparison { predicted_feasible: true, reference_feasible: true },
            FeasibilityComparison { predicted_feasible: true, reference_feasible: false },
            FeasibilityComparison { predicted_feasible: false, reference_feasible: true },
            FeasibilityComparison { predicted_feasible: false, reference_feasible: false },
        ])
        .expect("non-empty cases should be summarized");
        assert_eq!(summary.sample_count, 4);
        assert_eq!(summary.predicted_feasible_count, 2);
        assert_eq!(summary.false_feasible_count, 1);
        assert_eq!(summary.missed_feasible_count, 1);
        assert_eq!(summary.false_feasible_rate, Some(0.5));
    }

    #[test]
    fn feasibility_metrics_do_not_reward_abstaining_on_every_case() {
        let summary = summarize_feasibility(&[
            FeasibilityComparison { predicted_feasible: false, reference_feasible: true },
            FeasibilityComparison { predicted_feasible: false, reference_feasible: false },
        ])
        .expect("non-empty cases should be summarized");
        assert_eq!(summary.false_feasible_count, 0);
        assert_eq!(summary.false_feasible_rate, None);
        assert_eq!(summarize_feasibility(&[]), Err(BenchmarkError::EmptyInput));
    }

    #[test]
    fn ordering_accuracy_detects_rank_inversions() {
        let summary = pairwise_ordering_accuracy(
            &[
                ScalarComparison { predicted: 1.0, reference: 1.0 },
                ScalarComparison { predicted: 3.0, reference: 2.0 },
                ScalarComparison { predicted: 2.0, reference: 3.0 },
            ],
            1e-12,
        )
        .expect("non-tied references should be comparable");

        assert_eq!(summary.comparable_pairs, 3);
        assert_eq!(summary.correctly_ordered_pairs, 2);
        assert!((summary.accuracy - (2.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn ordering_accuracy_rejects_tolerance_and_all_tied_references() {
        assert_eq!(
            pairwise_ordering_accuracy(
                &[ScalarComparison { predicted: 1.0, reference: 1.0 }],
                -1.0,
            ),
            Err(BenchmarkError::InvalidTieTolerance)
        );
        assert_eq!(
            pairwise_ordering_accuracy(
                &[
                    ScalarComparison { predicted: 1.0, reference: 2.0 },
                    ScalarComparison { predicted: 2.0, reference: 2.0 },
                ],
                0.0,
            ),
            Err(BenchmarkError::NoComparablePairs)
        );
    }

    #[test]
    fn top_weighted_overlap_is_one_for_identical_rankings() {
        let summary = top_weighted_ranking_overlap(&[10, 20, 30], &[10, 20, 30], 0.9)
            .expect("unique equal-length rankings should be comparable");
        assert_eq!(summary.evaluated_depths, 3);
        assert_eq!(summary.persistence, 0.9);
        assert!((summary.normalized_score - 1.0).abs() < 1e-12);
    }

    #[test]
    fn top_weighted_overlap_penalizes_a_top_rank_inversion() {
        let predicted = top_weighted_ranking_overlap(&[20, 10, 30, 40], &[10, 20, 30, 40], 0.9)
            .expect("valid rankings should compare");
        let lower_persistence =
            top_weighted_ranking_overlap(&[20, 10, 30, 40], &[10, 20, 30, 40], 0.2)
                .expect("valid rankings should compare");
        assert!(predicted.normalized_score < 1.0);
        assert!(predicted.normalized_score > 0.0);
        assert!(lower_persistence.normalized_score > predicted.normalized_score);
    }

    #[test]
    fn top_weighted_overlap_rejects_bad_rankings() {
        assert_eq!(
            top_weighted_ranking_overlap(&[], &[], 0.9),
            Err(BenchmarkError::EmptyInput)
        );
        assert_eq!(
            top_weighted_ranking_overlap(&[1, 2], &[1], 0.9),
            Err(BenchmarkError::RankingLengthMismatch)
        );
        assert_eq!(
            top_weighted_ranking_overlap(&[1, 1], &[1, 2], 0.9),
            Err(BenchmarkError::DuplicateRankingId {
                ranking: "predicted",
                index: 1,
            })
        );
        assert_eq!(
            top_weighted_ranking_overlap(&[1, 2], &[1, 2], 1.0),
            Err(BenchmarkError::InvalidPersistence)
        );
    }

    #[test]
    fn structure_recovery_fraction_validates_counts() {
        assert!((structure_recovery_fraction(4, 3).expect("valid counts") - 0.75).abs() < 1e-12);
        assert_eq!(
            structure_recovery_fraction(0, 0),
            Err(BenchmarkError::ZeroReferenceStructures)
        );
        assert_eq!(
            structure_recovery_fraction(2, 3),
            Err(BenchmarkError::InvalidRecoveryCounts)
        );
    }

    #[test]
    fn acceleration_factor_requires_positive_counts() {
        assert!((acceleration_factor(50, 10).expect("positive counts") - 5.0).abs() < 1e-12);
        assert_eq!(
            acceleration_factor(0, 10),
            Err(BenchmarkError::ZeroEvaluationCount)
        );
    }
}
