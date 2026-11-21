//! Repeatability-calibrated similarity metrics for CETB research.
//!
//! This crate deliberately contains no data loader, correlation estimator or clinical
//! classifier. It operates on already-computed correlation matrices and participant-level
//! within-/between-condition similarities. Participant-level inference is performed only
//! after repeated segment measurements have been reduced within participant.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetricError {
    EmptyInput,
    LengthMismatch { left: usize, right: usize },
    NonFiniteValue { index: usize },
    NegativeWeight { index: usize },
    ZeroTotalMass,
    TooFewRegions,
    NonSquareMatrix,
    MatrixSizeMismatch { left: usize, right: usize },
    AsymmetricMatrix { row: usize, col: usize },
    CorrelationOutOfRange { row: usize, col: usize },
    InsufficientSelectedEdges { selected: usize, minimum: usize },
    TooFewParticipants { n: usize },
    InvalidSimilarity { index: usize },
    InvalidBootstrapDraws { draws: usize },
    InvalidAlpha,
}

impl fmt::Display for MetricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "edge-weight input is empty"),
            Self::LengthMismatch { left, right } =>
                write!(f, "edge-vector lengths differ: {left} != {right}"),
            Self::NonFiniteValue { index } =>
                write!(f, "non-finite value at edge index {index}"),
            Self::NegativeWeight { index } =>
                write!(f, "negative edge weight at index {index}"),
            Self::ZeroTotalMass => write!(f, "weighted Jaccard is undefined for zero total mass"),
            Self::TooFewRegions => write!(f, "at least two regions are required"),
            Self::NonSquareMatrix => write!(f, "correlation matrix must be square"),
            Self::MatrixSizeMismatch { left, right } =>
                write!(f, "correlation matrix dimensions differ: {left} != {right}"),
            Self::AsymmetricMatrix { row, col } =>
                write!(f, "matrix is not symmetric at ({row}, {col})"),
            Self::CorrelationOutOfRange { row, col } =>
                write!(f, "correlation outside [-1, 1] at ({row}, {col})"),
            Self::InsufficientSelectedEdges { selected, minimum } =>
                write!(f, "only {selected} edges selected; minimum required is {minimum}"),
            Self::TooFewParticipants { n } =>
                write!(f, "at least two participants are required, got {n}"),
            Self::InvalidSimilarity { index } =>
                write!(f, "similarity at participant index {index} is outside [0, 1] or non-finite"),
            Self::InvalidBootstrapDraws { draws } =>
                write!(f, "at least 100 bootstrap draws are required, got {draws}"),
            Self::InvalidAlpha => write!(f, "alpha must be finite and in (0, 0.5)"),
        }
    }
}

impl std::error::Error for MetricError {}

/// Pair-aware WJ comparison modes for signed correlation matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrelationWjMode {
    /// Use every upper-triangle edge by absolute magnitude.
    Absolute,
    /// Include only edges positive in both matrices; compare their positive magnitudes.
    PositiveEdgesInBoth,
    /// Include only edges negative in both matrices; compare their absolute magnitudes.
    NegativeEdgesInBoth,
    /// Shift every correlation from [-1, 1] to [0, 2], so sign disagreement lowers overlap.
    SignedShift,
}

/// Validate a square, symmetric correlation matrix. Diagonal values are ignored.
fn validate_correlation_matrix(matrix: &[Vec<f64>]) -> Result<usize, MetricError> {
    let n = matrix.len();
    if n < 2 {
        return Err(MetricError::TooFewRegions);
    }
    if matrix.iter().any(|row| row.len() != n) {
        return Err(MetricError::NonSquareMatrix);
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let a = matrix[i][j];
            let b = matrix[j][i];
            if !a.is_finite() || !b.is_finite() {
                return Err(MetricError::NonFiniteValue { index: i * n + j });
            }
            if a.abs() > 1.0 + 1e-10 || b.abs() > 1.0 + 1e-10 {
                return Err(MetricError::CorrelationOutOfRange { row: i, col: j });
            }
            if (a - b).abs() > 1e-8 {
                return Err(MetricError::AsymmetricMatrix { row: i, col: j });
            }
        }
    }
    Ok(n)
}

/// WJ similarity between two aligned correlation matrices.
///
/// Positive-only and negative-only modes use a shared sign mask: an edge is included
/// only if it has the requested sign in BOTH matrices. This matches the pairwise
/// semantics of the published ds006623 revision analysis. SignedShift instead includes
/// every edge after adding 1.0 to each correlation. The ROI ordering must be identical
/// in both matrices; numeric dimensions alone cannot detect a parcel-order mismatch.
pub fn weighted_jaccard_correlation_matrices(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    mode: CorrelationWjMode,
    minimum_selected_edges: usize,
) -> Result<f64, MetricError> {
    let n_a = validate_correlation_matrix(a)?;
    let n_b = validate_correlation_matrix(b)?;
    if n_a != n_b {
        return Err(MetricError::MatrixSizeMismatch { left: n_a, right: n_b });
    }

    let mut weights_a = Vec::with_capacity(n_a * (n_a - 1) / 2);
    let mut weights_b = Vec::with_capacity(n_a * (n_a - 1) / 2);

    for i in 0..n_a {
        for j in (i + 1)..n_a {
            let x = a[i][j].clamp(-1.0, 1.0);
            let y = b[i][j].clamp(-1.0, 1.0);
            let pair = match mode {
                CorrelationWjMode::Absolute => Some((x.abs(), y.abs())),
                CorrelationWjMode::PositiveEdgesInBoth if x > 0.0 && y > 0.0 => Some((x, y)),
                CorrelationWjMode::NegativeEdgesInBoth if x < 0.0 && y < 0.0 => Some((-x, -y)),
                CorrelationWjMode::SignedShift => Some((x + 1.0, y + 1.0)),
                _ => None,
            };
            if let Some((wa, wb)) = pair {
                weights_a.push(wa);
                weights_b.push(wb);
            }
        }
    }

    if weights_a.len() < minimum_selected_edges.max(1) {
        return Err(MetricError::InsufficientSelectedEdges {
            selected: weights_a.len(),
            minimum: minimum_selected_edges.max(1),
        });
    }
    weighted_jaccard(&weights_a, &weights_b)
}

/// Weighted Jaccard similarity for two aligned, non-negative edge-weight vectors.
///
/// A zero denominator is an error rather than a fabricated similarity. This matters
/// for sign-filtered edge sets when neither matrix contains selected edges.
pub fn weighted_jaccard(a: &[f64], b: &[f64]) -> Result<f64, MetricError> {
    if a.is_empty() || b.is_empty() {
        return Err(MetricError::EmptyInput);
    }
    if a.len() != b.len() {
        return Err(MetricError::LengthMismatch { left: a.len(), right: b.len() });
    }

    let mut min_sum = 0.0;
    let mut max_sum = 0.0;
    for (index, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        if !x.is_finite() || !y.is_finite() {
            return Err(MetricError::NonFiniteValue { index });
        }
        if x < 0.0 || y < 0.0 {
            return Err(MetricError::NegativeWeight { index });
        }
        min_sum += x.min(y);
        max_sum += x.max(y);
    }
    if !max_sum.is_finite() || max_sum <= 0.0 {
        return Err(MetricError::ZeroTotalMass);
    }
    let score = min_sum / max_sum;
    if !score.is_finite() || !(0.0..=1.0 + 1e-12).contains(&score) {
        return Err(MetricError::ZeroTotalMass);
    }
    Ok(score.min(1.0))
}

#[derive(Debug, Clone, PartialEq)]
pub struct RcafSummary {
    pub participant_count: usize,
    pub median_within_similarity: f64,
    pub median_between_similarity: f64,
    /// Median of within_i - between_i, preserving participant pairing.
    pub median_paired_gap: f64,
    pub positive_gap_fraction: f64,
    /// Descriptive normalized excess distance; not a variance decomposition.
    pub median_excess_distance_fraction: Option<f64>,
    /// Participants with a defined normalized excess-distance fraction.
    pub excess_distance_participant_count: usize,
    pub gap_ci_lower: f64,
    pub gap_ci_upper: f64,
    pub bootstrap_draws: usize,
}

/// Summarize participant-paired within-state and between-state similarities.
///
/// Each input element must already be a participant-level summary after repeated segments
/// have been aggregated within that participant. Do not pass segment windows as independent
/// observations. The percentile bootstrap resamples participants, not segments.
pub fn summarize_rcaf(
    within: &[f64],
    between: &[f64],
    bootstrap_draws: usize,
    alpha: f64,
    seed: u64,
) -> Result<RcafSummary, MetricError> {
    if within.len() != between.len() {
        return Err(MetricError::LengthMismatch { left: within.len(), right: between.len() });
    }
    let n = within.len();
    if n < 2 {
        return Err(MetricError::TooFewParticipants { n });
    }
    if bootstrap_draws < 100 {
        return Err(MetricError::InvalidBootstrapDraws { draws: bootstrap_draws });
    }
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 0.5 {
        return Err(MetricError::InvalidAlpha);
    }

    for i in 0..n {
        if !within[i].is_finite() || !(0.0..=1.0).contains(&within[i]) {
            return Err(MetricError::InvalidSimilarity { index: i });
        }
        if !between[i].is_finite() || !(0.0..=1.0).contains(&between[i]) {
            return Err(MetricError::InvalidSimilarity { index: i });
        }
    }

    let gaps: Vec<f64> = within.iter().zip(between.iter()).map(|(w, b)| w - b).collect();
    let excess: Vec<f64> = within.iter().zip(between.iter())
        .filter_map(|(w, b)| {
            let denominator = 1.0 - b;
            if denominator > f64::EPSILON {
                Some((w - b) / denominator)
            } else {
                None
            }
        })
        .collect();

    let mut rng = XorShift64::new(seed);
    let mut boot = Vec::with_capacity(bootstrap_draws);
    let mut sample = Vec::with_capacity(n);
    for _ in 0..bootstrap_draws {
        sample.clear();
        for _ in 0..n {
            let index = (rng.next() % n as u64) as usize;
            sample.push(gaps[index]);
        }
        boot.push(median(&sample));
    }
    boot.sort_by(f64::total_cmp);

    Ok(RcafSummary {
        participant_count: n,
        median_within_similarity: median(within),
        median_between_similarity: median(between),
        median_paired_gap: median(&gaps),
        positive_gap_fraction: gaps.iter().filter(|&&gap| gap > 0.0).count() as f64 / n as f64,
        median_excess_distance_fraction: if excess.is_empty() { None } else { Some(median(&excess)) },
        excess_distance_participant_count: excess.len(),
        gap_ci_lower: quantile_sorted(&boot, alpha / 2.0),
        gap_ci_upper: quantile_sorted(&boot, 1.0 - alpha / 2.0),
        bootstrap_draws,
    })
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

fn quantile_sorted(sorted: &[f64], probability: f64) -> f64 {
    let position = (sorted.len() - 1) as f64 * probability;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let fraction = position - lower as f64;
        sorted[lower] + fraction * (sorted[upper] - sorted[lower])
    }
}

/// Small deterministic PRNG so bootstrap fixtures require no dependency.
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed } }
    }

    fn next(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighted_jaccard_matches_hand_fixture() {
        let score = weighted_jaccard(&[0.2, 0.5], &[0.1, 0.4]).unwrap();
        assert!((score - 5.0 / 7.0).abs() < 1e-12);
    }

    #[test]
    fn identical_vectors_have_similarity_one() {
        assert_eq!(weighted_jaccard(&[0.2, 0.5], &[0.2, 0.5]).unwrap(), 1.0);
    }

    #[test]
    fn zero_mass_fails_closed() {
        assert_eq!(weighted_jaccard(&[0.0, 0.0], &[0.0, 0.0]), Err(MetricError::ZeroTotalMass));
    }

    #[test]
    fn invalid_edge_vectors_are_rejected() {
        assert!(matches!(weighted_jaccard(&[0.2], &[0.2, 0.3]), Err(MetricError::LengthMismatch { .. })));
        assert_eq!(weighted_jaccard(&[f64::NAN], &[0.2]), Err(MetricError::NonFiniteValue { index: 0 }));
        assert_eq!(weighted_jaccard(&[-0.1], &[0.2]), Err(MetricError::NegativeWeight { index: 0 }));
    }

    #[test]
    fn pair_aware_sign_masks_match_the_published_definition() {
        let a = vec![vec![1.0, 0.5, -0.4], vec![0.5, 1.0, 0.6], vec![-0.4, 0.6, 1.0]];
        let b = vec![vec![1.0, -0.5, -0.2], vec![-0.5, 1.0, 0.6], vec![-0.2, 0.6, 1.0]];
        // Only the shared positive edge (1,2) is compared: 0.6 vs 0.6.
        assert_eq!(weighted_jaccard_correlation_matrices(&a, &b, CorrelationWjMode::PositiveEdgesInBoth, 1).unwrap(), 1.0);
        // Shared negative edge (0,2): |-.4| vs |-.2| = 0.5 WJ.
        assert!((weighted_jaccard_correlation_matrices(&a, &b, CorrelationWjMode::NegativeEdgesInBoth, 1).unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn invalid_matrices_and_empty_sign_masks_fail_closed() {
        let asymmetric = vec![vec![1.0, 0.4], vec![0.3, 1.0]];
        assert!(matches!(weighted_jaccard_correlation_matrices(&asymmetric, &asymmetric, CorrelationWjMode::Absolute, 1), Err(MetricError::AsymmetricMatrix { .. })));
        let no_shared_positive = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert!(matches!(weighted_jaccard_correlation_matrices(&no_shared_positive, &no_shared_positive, CorrelationWjMode::PositiveEdgesInBoth, 1), Err(MetricError::InsufficientSelectedEdges { .. })));
    }

    #[test]
    fn rcaf_summary_is_paired_and_deterministic() {
        let within = [0.54, 0.55, 0.56, 0.53, 0.57];
        let between = [0.48, 0.49, 0.50, 0.47, 0.51];
        let a = summarize_rcaf(&within, &between, 2000, 0.05, 42).unwrap();
        let b = summarize_rcaf(&within, &between, 2000, 0.05, 42).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.participant_count, 5);
        assert!((a.median_paired_gap - 0.06).abs() < 1e-12);
        assert_eq!(a.positive_gap_fraction, 1.0);
        assert!(a.gap_ci_lower <= a.median_paired_gap);
        assert!(a.gap_ci_upper >= a.median_paired_gap);
    }

    #[test]
    fn rcaf_rejects_unpaired_or_invalid_similarity() {
        assert!(matches!(summarize_rcaf(&[0.5], &[0.4, 0.3], 100, 0.05, 1), Err(MetricError::LengthMismatch { .. })));
        assert_eq!(summarize_rcaf(&[f64::NAN, 0.5], &[0.4, 0.3], 100, 0.05, 1), Err(MetricError::InvalidSimilarity { index: 0 }));
        assert!(matches!(summarize_rcaf(&[0.5, 0.6], &[0.4, 0.3], 99, 0.05, 1), Err(MetricError::InvalidBootstrapDraws { .. })));
    }

    #[test]
    fn reported_floor_example_has_descriptive_normalization() {
        let within = [0.544, 0.544, 0.544];
        let between = [0.486, 0.486, 0.486];
        let summary = summarize_rcaf(&within, &between, 1000, 0.05, 7).unwrap();
        let expected = (0.544 - 0.486) / (1.0 - 0.486);
        assert!((summary.median_excess_distance_fraction.unwrap() - expected).abs() < 1e-12);
    }
}
