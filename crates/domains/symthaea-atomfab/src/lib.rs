#![forbid(unsafe_code)]

//! Evidence-first benchmark metrics for the AtomFab research track.
//!
//! This crate intentionally does not implement an atomistic solver, interatomic
//! potential, dataset loader, or instrument driver. It supplies small, auditable
//! metric primitives so research adapters can share validation and reporting
//! behavior. Metrics are not evidence that a physical process works.

pub mod benchmark_registry;
pub mod campaign;
pub mod campaign_statistics;
pub mod contract;
pub mod decision_frontier;
pub mod evidence_router;
pub mod failure_modes;
pub mod published_ranking;

use std::collections::HashSet;
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

/// Separate discovery-set agreement from ranking agreement among shared candidates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatchedRankingSummary {
    pub predicted_candidate_count: usize,
    pub reference_candidate_count: usize,
    pub shared_candidate_count: usize,
    /// Jaccard overlap of the predicted and reference candidate sets, in [0, 1].
    pub candidate_jaccard: f64,
    /// Ranking overlap calculated only after restricting both rankings to shared IDs.
    /// None means no candidate IDs were shared, so ranking agreement is undefined.
    pub shared_candidate_ranking: Option<TopWeightedOverlapSummary>,
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

/// One scalar prediction and reference, paired with an uncertainty score used
/// to decide whether a model should abstain from recommending the prediction.
#[derive(Debug, Clone, Copy)]
pub struct UncertaintyComparison {
    pub predicted: f64,
    pub reference: f64,
    /// Non-negative uncertainty score; its units and construction are model-defined.
    pub uncertainty: f64,
}

/// One point on a risk-coverage curve after retaining the least-uncertain cases.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RiskCoveragePoint {
    /// Number of samples retained at or below the uncertainty threshold.
    pub accepted_count: usize,
    /// Fraction of the full evaluation set retained, in (0, 1].
    pub coverage: f64,
    /// Mean absolute prediction error on retained samples.
    pub mean_absolute_error: f64,
    /// Worst absolute prediction error on retained samples.
    pub max_absolute_error: f64,
    /// Largest uncertainty value among retained samples.
    pub uncertainty_threshold: f64,
}

/// Discrete uncertainty-selective evaluation with a conservative stepwise AURC.
#[derive(Debug, Clone, PartialEq)]
pub struct RiskCoverageSummary {
    pub count: usize,
    /// Stepwise area under risk-coverage: sum of each coverage increment times
    /// the risk at the increment's endpoint. Lower is better for fixed units.
    pub area_under_curve: f64,
    /// Points are ordered by ascending coverage and include every distinct
    /// uncertainty threshold; equal uncertainty values enter as one group.
    pub points: Vec<RiskCoveragePoint>,
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
    NonFiniteValue { index: usize, field: &'static str },
    InvalidInterval { index: usize },
    InvalidUncertainty { index: usize },
    InvalidTieTolerance,
    InvalidPersistence,
    RankingLengthMismatch,
    DuplicateRankingId { ranking: &'static str, index: usize },
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
                write!(
                    f,
                    "sample {index} has a lower interval bound above its upper bound"
                )
            }
            Self::InvalidUncertainty { index } => write!(
                f,
                "sample {index} has an invalid (negative or non-finite) uncertainty"
            ),
            Self::InvalidTieTolerance => write!(f, "tie tolerance must be finite and non-negative"),
            Self::InvalidPersistence => write!(
                f,
                "ranking persistence must be finite and strictly between zero and one"
            ),
            Self::RankingLengthMismatch => write!(
                f,
                "predicted and reference rankings must have the same non-zero length"
            ),
            Self::DuplicateRankingId { ranking, index } => {
                write!(
                    f,
                    "{ranking} ranking repeats an identifier at index {index}"
                )
            }
            Self::NoComparablePairs => write!(f, "no non-tied reference pairs are available"),
            Self::ZeroReferenceStructures => {
                write!(f, "reference structure count must be greater than zero")
            }
            Self::InvalidRecoveryCounts => {
                write!(f, "recovered structure count cannot exceed reference count")
            }
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

/// Evaluate abstention using a risk-coverage curve sorted by increasing uncertainty.
///
/// Samples with equal uncertainty are admitted together, so the result does not
/// depend on input ordering or use arbitrary tie-breaking to improve the score.
/// The reported AURC is a discrete right-step integral:
/// `sum((accepted_count - previous_count) / N * risk_at_this_point)`.
///
/// This is a descriptive benchmark primitive. It does not establish that the
/// uncertainty score is calibrated; thresholds and score construction must be
/// frozen on validation data before reporting test-set performance.
pub fn summarize_risk_coverage(
    samples: &[UncertaintyComparison],
) -> Result<RiskCoverageSummary, BenchmarkError> {
    if samples.is_empty() {
        return Err(BenchmarkError::EmptyInput);
    }

    let mut ordered = samples.to_vec();
    for (index, sample) in ordered.iter().enumerate() {
        for (field, value) in [
            ("predicted", sample.predicted),
            ("reference", sample.reference),
        ] {
            if !value.is_finite() {
                return Err(BenchmarkError::NonFiniteValue { index, field });
            }
        }
        if !sample.uncertainty.is_finite() || sample.uncertainty < 0.0 {
            return Err(BenchmarkError::InvalidUncertainty { index });
        }
        let error = sample.predicted - sample.reference;
        if !error.is_finite() {
            return Err(BenchmarkError::NonFiniteMetric);
        }
    }

    ordered.sort_by(|a, b| {
        a.uncertainty
            .total_cmp(&b.uncertainty)
            .then_with(|| a.predicted.total_cmp(&b.predicted))
            .then_with(|| a.reference.total_cmp(&b.reference))
    });
    let count = ordered.len();
    let denominator = count as f64;
    let mut points = Vec::new();
    let mut accepted_count = 0usize;
    let mut absolute_error_total = 0.0;
    let mut max_absolute_error: f64 = 0.0;
    let mut weighted_risk_area = 0.0;

    while accepted_count < count {
        let threshold = ordered[accepted_count].uncertainty;
        let previous_count = accepted_count;
        while accepted_count < count && ordered[accepted_count].uncertainty == threshold {
            let error =
                (ordered[accepted_count].predicted - ordered[accepted_count].reference).abs();
            absolute_error_total += error;
            max_absolute_error = max_absolute_error.max(error);
            if !absolute_error_total.is_finite() {
                return Err(BenchmarkError::NonFiniteMetric);
            }
            accepted_count += 1;
        }

        let mean_absolute_error = absolute_error_total / accepted_count as f64;
        let coverage = accepted_count as f64 / denominator;
        let coverage_increment = (accepted_count - previous_count) as f64 / denominator;
        weighted_risk_area += coverage_increment * mean_absolute_error;
        if !mean_absolute_error.is_finite()
            || !coverage.is_finite()
            || !weighted_risk_area.is_finite()
            || !(0.0..=1.0).contains(&coverage)
        {
            return Err(BenchmarkError::NonFiniteMetric);
        }
        points.push(RiskCoveragePoint {
            accepted_count,
            coverage,
            mean_absolute_error,
            max_absolute_error,
            uncertainty_threshold: threshold,
        });
    }

    Ok(RiskCoverageSummary {
        count,
        area_under_curve: weighted_risk_area,
        points,
    })
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

fn validate_unique_ranking(
    ranking_name: &'static str,
    ranking: &[u64],
) -> Result<(), BenchmarkError> {
    let mut seen = HashSet::new();
    for (index, identifier) in ranking.iter().copied().enumerate() {
        if !seen.insert(identifier) {
            return Err(BenchmarkError::DuplicateRankingId {
                ranking: ranking_name,
                index,
            });
        }
    }
    Ok(())
}

/// Compute normalized finite-prefix top-weighted overlap for ranked candidate IDs.
///
/// Both rankings must contain the same number of unique candidate identifiers
/// drawn from a common, canonically identified candidate universe. The function
/// can validate uniqueness and length, but cannot prove that callers assigned
/// the same identifier to equivalent structures. At each depth, overlap is the
/// fraction of top-depth candidates shared by both rankings. The score is the
/// persistence-weighted average across depths using
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
    if !persistence.is_finite() || persistence <= 0.0 || persistence >= 1.0 {
        return Err(BenchmarkError::InvalidPersistence);
    }

    validate_unique_ranking("predicted", predicted_ranking)?;
    validate_unique_ranking("reference", reference_ranking)?;

    // At prefix depth d, a candidate contributes to the overlap exactly when
    // both its predicted and reference rank are <= d. Count each shared ID at
    // its max rank once, then accumulate the prefix counts. This avoids
    // repeatedly intersecting two growing sets (quadratic work for long lists).
    // Hash-based uniqueness/indexing keeps the expected work linear in list length.
    let len = predicted_ranking.len();
    let reference_positions: std::collections::HashMap<u64, usize> = reference_ranking
        .iter()
        .copied()
        .enumerate()
        .map(|(index, identifier)| (identifier, index + 1))
        .collect();
    let mut shared_starting_at = vec![0usize; len + 1];
    for (index, identifier) in predicted_ranking.iter().copied().enumerate() {
        if let Some(reference_rank) = reference_positions.get(&identifier).copied() {
            let first_shared_depth = (index + 1).max(reference_rank);
            shared_starting_at[first_shared_depth] += 1;
        }
    }

    let mut shared_prefix_count = 0usize;
    let mut weighted_overlap = 0.0;
    let mut total_weight = 0.0;
    for (depth, shared_at_depth) in shared_starting_at.iter().enumerate().skip(1) {
        shared_prefix_count += *shared_at_depth;
        let prefix_overlap = shared_prefix_count as f64 / depth as f64;
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

/// Compare candidate discovery agreement separately from ranking agreement.
///
/// Candidate-set Jaccard is calculated on the original sets. The top-weighted
/// ranking overlap is then calculated on each ranking filtered to the intersection
/// of canonical candidate IDs. This avoids treating newly discovered or missing
/// candidates as if they were ranking errors. Structure canonicalization and ID
/// stability remain caller-owned preconditions.
pub fn matched_candidate_ranking_overlap(
    predicted_ranking: &[u64],
    reference_ranking: &[u64],
    persistence: f64,
) -> Result<MatchedRankingSummary, BenchmarkError> {
    if predicted_ranking.is_empty() || reference_ranking.is_empty() {
        return Err(BenchmarkError::EmptyInput);
    }
    if !persistence.is_finite() || persistence <= 0.0 || persistence >= 1.0 {
        return Err(BenchmarkError::InvalidPersistence);
    }
    validate_unique_ranking("predicted", predicted_ranking)?;
    validate_unique_ranking("reference", reference_ranking)?;

    let predicted_ids: HashSet<u64> = predicted_ranking.iter().copied().collect();
    let reference_ids: HashSet<u64> = reference_ranking.iter().copied().collect();
    let shared_candidate_count = predicted_ids.intersection(&reference_ids).count();

    let union_count = predicted_ids.len() + reference_ids.len() - shared_candidate_count;
    let candidate_jaccard = shared_candidate_count as f64 / union_count as f64;
    let predicted_shared: Vec<u64> = predicted_ranking
        .iter()
        .copied()
        .filter(|identifier| reference_ids.contains(identifier))
        .collect();
    let reference_shared: Vec<u64> = reference_ranking
        .iter()
        .copied()
        .filter(|identifier| predicted_ids.contains(identifier))
        .collect();
    let shared_candidate_ranking = if shared_candidate_count == 0 {
        None
    } else {
        Some(top_weighted_ranking_overlap(
            &predicted_shared,
            &reference_shared,
            persistence,
        )?)
    };

    if !candidate_jaccard.is_finite() || !(0.0..=1.0).contains(&candidate_jaccard) {
        return Err(BenchmarkError::NonFiniteMetric);
    }
    Ok(MatchedRankingSummary {
        predicted_candidate_count: predicted_ranking.len(),
        reference_candidate_count: reference_ranking.len(),
        shared_candidate_count,
        candidate_jaccard,
        shared_candidate_ranking,
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
    fn risk_coverage_orders_by_uncertainty_and_reports_stepwise_area() {
        let summary = summarize_risk_coverage(&[
            UncertaintyComparison {
                predicted: 10.0,
                reference: 0.0,
                uncertainty: 0.9,
            },
            UncertaintyComparison {
                predicted: 1.0,
                reference: 0.0,
                uncertainty: 0.1,
            },
            UncertaintyComparison {
                predicted: 2.0,
                reference: 0.0,
                uncertainty: 0.2,
            },
        ])
        .expect("finite non-negative uncertainty should summarize");

        assert_eq!(summary.count, 3);
        assert_eq!(summary.points.len(), 3);
        assert_eq!(summary.points[0].accepted_count, 1);
        assert_eq!(summary.points[0].uncertainty_threshold, 0.1);
        assert!((summary.points[0].mean_absolute_error - 1.0).abs() < 1e-12);
        assert_eq!(summary.points[1].accepted_count, 2);
        assert!((summary.points[1].mean_absolute_error - 1.5).abs() < 1e-12);
        assert_eq!(summary.points[2].accepted_count, 3);
        assert!((summary.points[2].mean_absolute_error - (13.0 / 3.0)).abs() < 1e-12);
        let expected_area = (1.0 / 3.0) * 1.0 + (1.0 / 3.0) * 1.5 + (1.0 / 3.0) * (13.0 / 3.0);
        assert!((summary.area_under_curve - expected_area).abs() < 1e-12);
    }

    #[test]
    fn risk_coverage_groups_equal_uncertainty_and_is_input_order_invariant() {
        let a = UncertaintyComparison {
            predicted: 1.0,
            reference: 0.0,
            uncertainty: 0.2,
        };
        let b = UncertaintyComparison {
            predicted: 9.0,
            reference: 0.0,
            uncertainty: 0.2,
        };
        let c = UncertaintyComparison {
            predicted: 2.0,
            reference: 0.0,
            uncertainty: 0.8,
        };
        let first = summarize_risk_coverage(&[a, b, c]).expect("valid samples");
        let second =
            summarize_risk_coverage(&[c, b, a]).expect("permutation should not change result");
        assert_eq!(first, second);
        assert_eq!(first.points.len(), 2);
        assert_eq!(first.points[0].accepted_count, 2);
        assert!((first.points[0].mean_absolute_error - 5.0).abs() < 1e-12);
        assert_eq!(first.points[1].accepted_count, 3);
    }

    #[test]
    fn risk_coverage_tied_scores_are_bitwise_deterministic_under_permutation() {
        let samples = [
            UncertaintyComparison {
                predicted: 1.0e16,
                reference: 0.0,
                uncertainty: 0.5,
            },
            UncertaintyComparison {
                predicted: 1.0,
                reference: 0.0,
                uncertainty: 0.5,
            },
            UncertaintyComparison {
                predicted: 1.0,
                reference: 0.0,
                uncertainty: 0.5,
            },
        ];
        let forward = summarize_risk_coverage(&samples).expect("valid samples");
        let reverse = summarize_risk_coverage(&[samples[2], samples[1], samples[0]])
            .expect("permuted valid samples");
        assert_eq!(forward, reverse);
    }

    #[test]
    fn risk_coverage_rejects_empty_negative_uncertainty_and_non_finite_values() {
        assert_eq!(
            summarize_risk_coverage(&[]),
            Err(BenchmarkError::EmptyInput)
        );
        assert_eq!(
            summarize_risk_coverage(&[UncertaintyComparison {
                predicted: 0.0,
                reference: 0.0,
                uncertainty: -0.1,
            }]),
            Err(BenchmarkError::InvalidUncertainty { index: 0 })
        );
        assert_eq!(
            summarize_risk_coverage(&[UncertaintyComparison {
                predicted: 0.0,
                reference: 0.0,
                uncertainty: f64::NAN,
            }]),
            Err(BenchmarkError::InvalidUncertainty { index: 0 })
        );
        assert_eq!(
            summarize_risk_coverage(&[UncertaintyComparison {
                predicted: f64::INFINITY,
                reference: 0.0,
                uncertainty: 0.2,
            }]),
            Err(BenchmarkError::NonFiniteValue {
                index: 0,
                field: "predicted",
            })
        );
    }

    #[test]
    fn feasibility_metrics_expose_false_accepts_and_missed_opportunities() {
        let summary = summarize_feasibility(&[
            FeasibilityComparison {
                predicted_feasible: true,
                reference_feasible: true,
            },
            FeasibilityComparison {
                predicted_feasible: true,
                reference_feasible: false,
            },
            FeasibilityComparison {
                predicted_feasible: false,
                reference_feasible: true,
            },
            FeasibilityComparison {
                predicted_feasible: false,
                reference_feasible: false,
            },
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
            FeasibilityComparison {
                predicted_feasible: false,
                reference_feasible: true,
            },
            FeasibilityComparison {
                predicted_feasible: false,
                reference_feasible: false,
            },
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
                ScalarComparison {
                    predicted: 1.0,
                    reference: 1.0,
                },
                ScalarComparison {
                    predicted: 3.0,
                    reference: 2.0,
                },
                ScalarComparison {
                    predicted: 2.0,
                    reference: 3.0,
                },
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
                &[ScalarComparison {
                    predicted: 1.0,
                    reference: 1.0
                }],
                -1.0,
            ),
            Err(BenchmarkError::InvalidTieTolerance)
        );
        assert_eq!(
            pairwise_ordering_accuracy(
                &[
                    ScalarComparison {
                        predicted: 1.0,
                        reference: 2.0
                    },
                    ScalarComparison {
                        predicted: 2.0,
                        reference: 2.0
                    },
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
        assert!(predicted.normalized_score > lower_persistence.normalized_score);
    }

    #[test]
    fn top_weighted_overlap_matches_brute_force_prefix_intersections() {
        use std::collections::BTreeSet;

        let rankings: [&[u64]; 6] = [
            &[1, 2, 3, 4],
            &[4, 3, 2, 1],
            &[1, 3, 5, 7],
            &[2, 4, 6, 8],
            &[2, 1, 3, 4],
            &[8, 1, 6, 4],
        ];
        let persistence: f64 = 0.94;

        for predicted in rankings {
            for reference in rankings {
                let mut predicted_prefix = BTreeSet::new();
                let mut reference_prefix = BTreeSet::new();
                let mut weighted_total = 0.0;
                let mut weight_total = 0.0;

                for depth in 1..=predicted.len() {
                    predicted_prefix.insert(predicted[depth - 1]);
                    reference_prefix.insert(reference[depth - 1]);
                    let overlap = predicted_prefix.intersection(&reference_prefix).count() as f64
                        / depth as f64;
                    let weight = (1.0 - persistence) * persistence.powf((depth - 1) as f64);
                    weighted_total += weight * overlap;
                    weight_total += weight;
                }
                let expected = weighted_total / weight_total;
                let actual = top_weighted_ranking_overlap(predicted, reference, persistence)
                    .expect("unique equal-length rankings should compare")
                    .normalized_score;
                assert!(
                    (actual - expected).abs() < 1.0e-12,
                    "rankings {predicted:?} vs {reference:?}: {actual} != {expected}"
                );
            }
        }
    }

    #[test]
    fn top_weighted_overlap_is_zero_for_disjoint_rankings() {
        let summary = top_weighted_ranking_overlap(&[1, 2, 3], &[4, 5, 6], 0.94)
            .expect("equal-length unique rankings should compare");
        assert_eq!(summary.normalized_score, 0.0);
    }

    #[test]
    fn top_weighted_overlap_handles_large_candidate_rankings() {
        let predicted: Vec<u64> = (0..20_000).collect();
        let mut reference = predicted.clone();
        reference.reverse();
        let summary = top_weighted_ranking_overlap(&predicted, &reference, 0.94)
            .expect("large valid rankings should compare");
        assert_eq!(summary.evaluated_depths, 20_000);
        assert!(summary.normalized_score.is_finite());
        assert!((0.0..=1.0).contains(&summary.normalized_score));
    }

    #[test]
    fn matched_ranking_overlap_separates_discovery_from_ranking() {
        let summary = matched_candidate_ranking_overlap(&[1, 2, 3], &[1, 2, 4], 0.94)
            .expect("a non-empty shared candidate set should compare");
        assert_eq!(summary.predicted_candidate_count, 3);
        assert_eq!(summary.reference_candidate_count, 3);
        assert_eq!(summary.shared_candidate_count, 2);
        assert!((summary.candidate_jaccard - 0.5).abs() < 1e-12);
        assert!(
            (summary
                .shared_candidate_ranking
                .expect("shared candidate set has a defined ranking")
                .normalized_score
                - 1.0)
                .abs()
                < 1e-12
        );

        let inverted = matched_candidate_ranking_overlap(&[1, 2, 3], &[2, 1, 4], 0.94)
            .expect("shared IDs should remain comparable");
        assert!((inverted.candidate_jaccard - summary.candidate_jaccard).abs() < 1e-12);
        assert!(
            inverted
                .shared_candidate_ranking
                .expect("shared candidate set has a defined ranking")
                .normalized_score
                < 1.0
        );
    }

    #[test]
    fn matched_ranking_overlap_keeps_discovery_jaccard_when_no_candidates_are_shared() {
        let summary = matched_candidate_ranking_overlap(&[1, 2], &[3, 4], 0.94)
            .expect("disjoint non-empty candidate sets still have defined Jaccard");
        assert_eq!(summary.shared_candidate_count, 0);
        assert_eq!(summary.candidate_jaccard, 0.0);
        assert_eq!(summary.shared_candidate_ranking, None);

        assert_eq!(
            matched_candidate_ranking_overlap(&[1, 1], &[1, 2], 0.94),
            Err(BenchmarkError::DuplicateRankingId {
                ranking: "predicted",
                index: 1,
            })
        );
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
            top_weighted_ranking_overlap(&[1, 2], &[1, 2], 0.0),
            Err(BenchmarkError::InvalidPersistence)
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

#[cfg(test)]
mod workcell_fault_corpus_tests {
    use serde_json::Value;
    use std::collections::BTreeSet;

    #[test]
    fn frozen_workcell_fault_corpus_has_unique_ids_and_fail_closed_oracles() {
        let corpus: Value =
            serde_json::from_str(include_str!("../fixtures/workcell-fault-corpus-v1.json"))
                .expect("shared workcell fault corpus must be valid JSON");

        assert_eq!(corpus["schema_version"], 1);
        assert_eq!(corpus["corpus_id"], "atomfab-workcell-fault-corpus-v1");
        assert_eq!(corpus["status"], "frozen_software_regression_suite");

        let cases = corpus["cases"]
            .as_array()
            .expect("corpus cases must be an array");
        assert!(
            cases.len() >= 15,
            "v1 requires the frozen fault/control cases"
        );

        let mut ids = BTreeSet::new();
        let mut accepted_controls = 0usize;
        for case in cases {
            let id = case["case_id"]
                .as_str()
                .expect("every case needs an identifier");
            assert!(ids.insert(id.to_owned()), "duplicate case identifier: {id}");

            let expected = &case["expected"];
            let decision = expected["decision"]
                .as_str()
                .expect("every case needs an expected decision");
            let action_authorized = expected["simulation_action_authorized"]
                .as_bool()
                .expect("every case needs an action-authorization oracle");
            let feasibility = expected["feasibility_status"]
                .as_str()
                .expect("every case needs a feasibility disposition");
            let hard_violation = expected["hard_constraint_violation"]
                .as_bool()
                .expect("every case needs a hard-constraint oracle");

            if decision != "accept" {
                assert!(
                    !action_authorized,
                    "non-accepting case {id} must not authorize a simulation action"
                );
                assert!(
                    expected["reason_codes"]
                        .as_array()
                        .is_some_and(|codes| !codes.is_empty()),
                    "non-accepting case {id} must explain its disposition"
                );
            }
            if feasibility == "feasible" {
                assert_eq!(
                    decision, "accept",
                    "case {id} cannot be feasible without an accepted decision"
                );
            }
            if hard_violation {
                assert_ne!(
                    decision, "accept",
                    "hard-constraint violation accepted: {id}"
                );
                assert_eq!(
                    feasibility, "infeasible",
                    "hard-constraint violation must remain visible: {id}"
                );
            }
            if case["control"].as_bool() == Some(true) && decision == "accept" {
                assert!(
                    action_authorized,
                    "accepted control {id} should authorize only the simulated action"
                );
                accepted_controls += 1;
            }
        }

        for required in [
            "stale-observation",
            "non-finite-observation",
            "sensor-drift-outside-calibration",
            "actuator-saturation",
            "invalid-planner-gain",
            "tool-obstruction",
            "actuator-count-mismatch",
            "bridge-timeout",
            "malformed-planner-output",
            "model-out-of-domain",
            "replay-divergence",
            "synthetic-as-measurement",
            "physical-actuation-without-interlock",
            "bounded-sensor-drift-control",
            "valid-simulation-control",
        ] {
            assert!(
                ids.contains(required),
                "required v1 case is missing: {required}"
            );
        }
        assert!(
            accepted_controls >= 2,
            "the corpus needs accepted controls so reject-all cannot pass"
        );
    }
}
