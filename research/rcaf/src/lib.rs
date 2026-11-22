//! Repeatability-calibrated similarity metrics for CETB research.
//!
//! This crate contains no data loader or clinical classifier. It includes explicit reference
//! Pearson connectivity and matrix-similarity implementations for synthetic fixtures and
//! qualification harnesses; empirical comparisons must pin and identify the upstream connectivity
//! estimator and preprocessing pipeline. Participant-level inference is performed only after
//! repeated segment measurements have been reduced within participant.

pub mod estimator_response;
pub mod factorial_counterfactual;
pub mod factorial_dose_response;
pub mod groupwise_null_calibration;
pub mod held_out_null_calibration;
pub mod motion_transient;
pub mod simulation;
pub mod spatial_calibration;
pub mod spatial_nuisance;

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
    InvalidSignThreshold,
    RoiLabelCountMismatch { labels: usize, regions: usize },
    RoiOrderMismatch,
    EmptyRoiLabel { index: usize },
    DuplicateRoiLabel { index: usize },
    EmptyParticipantId { index: usize },
    DuplicateParticipantId { index: usize },
    InvalidSimulationConfig { reason: &'static str },
}

impl fmt::Display for MetricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "edge-weight input is empty"),
            Self::LengthMismatch { left, right } => {
                write!(f, "edge-vector lengths differ: {left} != {right}")
            }
            Self::NonFiniteValue { index } => write!(f, "non-finite value at edge index {index}"),
            Self::NegativeWeight { index } => write!(f, "negative edge weight at index {index}"),
            Self::ZeroTotalMass => write!(f, "weighted Jaccard is undefined for zero total mass"),
            Self::TooFewRegions => write!(f, "at least two regions are required"),
            Self::NonSquareMatrix => write!(f, "correlation matrix must be square"),
            Self::MatrixSizeMismatch { left, right } => {
                write!(f, "correlation matrix dimensions differ: {left} != {right}")
            }
            Self::AsymmetricMatrix { row, col } => {
                write!(f, "matrix is not symmetric at ({row}, {col})")
            }
            Self::CorrelationOutOfRange { row, col } => {
                write!(f, "correlation outside [-1, 1] at ({row}, {col})")
            }
            Self::InsufficientSelectedEdges { selected, minimum } => write!(
                f,
                "only {selected} edges selected; minimum required is {minimum}"
            ),
            Self::TooFewParticipants { n } => {
                write!(f, "at least two participants are required, got {n}")
            }
            Self::InvalidSimilarity { index } => write!(
                f,
                "similarity at participant index {index} is outside [0, 1] or non-finite"
            ),
            Self::InvalidBootstrapDraws { draws } => {
                write!(f, "at least 100 bootstrap draws are required, got {draws}")
            }
            Self::InvalidAlpha => write!(f, "alpha must be finite and in (0, 0.5)"),
            Self::InvalidSignThreshold => write!(f, "sign threshold must be finite and in [0, 1]"),
            Self::RoiLabelCountMismatch { labels, regions } => write!(
                f,
                "ROI label count {labels} does not match matrix dimension {regions}"
            ),
            Self::RoiOrderMismatch => {
                write!(f, "ROI label order differs between compared matrices")
            }
            Self::EmptyRoiLabel { index } => write!(f, "ROI label at index {index} is empty"),
            Self::DuplicateRoiLabel { index } => write!(f, "duplicate ROI label at index {index}"),
            Self::EmptyParticipantId { index } => {
                write!(f, "participant ID at index {index} is empty")
            }
            Self::DuplicateParticipantId { index } => {
                write!(f, "duplicate participant ID at index {index}")
            }
            Self::InvalidSimulationConfig { reason } => {
                write!(f, "invalid injection-recovery configuration: {reason}")
            }
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
    for (i, row) in matrix.iter().enumerate() {
        for ((j, a), row_j) in row
            .iter()
            .enumerate()
            .skip(i + 1)
            .zip(matrix.iter().skip(i + 1))
        {
            let a = *a;
            let b = row_j[i];
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

fn validate_roi_labels(labels: &[String], regions: usize) -> Result<(), MetricError> {
    if labels.len() != regions {
        return Err(MetricError::RoiLabelCountMismatch {
            labels: labels.len(),
            regions,
        });
    }
    let mut seen = std::collections::HashSet::with_capacity(labels.len());
    for (index, label) in labels.iter().enumerate() {
        if label.trim().is_empty() {
            return Err(MetricError::EmptyRoiLabel { index });
        }
        if !seen.insert(label) {
            return Err(MetricError::DuplicateRoiLabel { index });
        }
    }
    Ok(())
}

/// WJ comparison that verifies exact ROI labels and order before comparing matrices.
///
/// This is the preferred entry point for empirical data. The unlabeled matrix function is
/// retained for fixtures and callers that already enforce alignment externally.
pub fn weighted_jaccard_labeled_correlation_matrices(
    a_roi_labels: &[String],
    a: &[Vec<f64>],
    b_roi_labels: &[String],
    b: &[Vec<f64>],
    mode: CorrelationWjMode,
    minimum_selected_edges: usize,
) -> Result<f64, MetricError> {
    let n_a = validate_correlation_matrix(a)?;
    let n_b = validate_correlation_matrix(b)?;
    validate_roi_labels(a_roi_labels, n_a)?;
    validate_roi_labels(b_roi_labels, n_b)?;
    if a_roi_labels != b_roi_labels {
        return Err(MetricError::RoiOrderMismatch);
    }
    weighted_jaccard_correlation_matrices(a, b, mode, minimum_selected_edges)
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
        return Err(MetricError::MatrixSizeMismatch {
            left: n_a,
            right: n_b,
        });
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
        return Err(MetricError::LengthMismatch {
            left: a.len(),
            right: b.len(),
        });
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
    /// Exact two-sided participant-level sign test on paired gaps; ties are excluded.
    pub sign_test: SignTestSummary,
}

/// A within/between pair attached to one pseudonymous participant key.
#[derive(Debug, Clone, PartialEq)]
pub struct ParticipantSimilarity {
    pub participant_id: String,
    pub within_similarity: f64,
    pub between_similarity: f64,
}

/// Safely summarize paired participant records after validating unique participant identity.
///
/// This is the preferred participant-level API. It prevents accidental pairing by row order
/// across separately assembled within and between vectors.
pub fn summarize_rcaf_by_participant(
    observations: &[ParticipantSimilarity],
    bootstrap_draws: usize,
    alpha: f64,
    seed: u64,
) -> Result<RcafSummary, MetricError> {
    let mut seen = std::collections::HashSet::with_capacity(observations.len());
    for (index, observation) in observations.iter().enumerate() {
        if observation.participant_id.trim().is_empty() {
            return Err(MetricError::EmptyParticipantId { index });
        }
        if !seen.insert(&observation.participant_id) {
            return Err(MetricError::DuplicateParticipantId { index });
        }
    }
    let within: Vec<f64> = observations
        .iter()
        .map(|item| item.within_similarity)
        .collect();
    let between: Vec<f64> = observations
        .iter()
        .map(|item| item.between_similarity)
        .collect();
    summarize_rcaf(&within, &between, bootstrap_draws, alpha, seed)
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
        return Err(MetricError::LengthMismatch {
            left: within.len(),
            right: between.len(),
        });
    }
    let n = within.len();
    if n < 2 {
        return Err(MetricError::TooFewParticipants { n });
    }
    if bootstrap_draws < 100 {
        return Err(MetricError::InvalidBootstrapDraws {
            draws: bootstrap_draws,
        });
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

    let gaps: Vec<f64> = within
        .iter()
        .zip(between.iter())
        .map(|(w, b)| w - b)
        .collect();
    let sign_test = exact_sign_test(&gaps)?;
    let excess: Vec<f64> = within
        .iter()
        .zip(between.iter())
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
            let index = rng.index_below(n as u64) as usize;
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
        median_excess_distance_fraction: if excess.is_empty() {
            None
        } else {
            Some(median(&excess))
        },
        excess_distance_participant_count: excess.len(),
        gap_ci_lower: quantile_sorted(&boot, alpha / 2.0),
        gap_ci_upper: quantile_sorted(&boot, 1.0 - alpha / 2.0),
        bootstrap_draws,
        sign_test,
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
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    fn next(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Rejection sampling removes modulo bias from bootstrap indices.
    fn index_below(&mut self, upper: u64) -> u64 {
        let threshold = upper.wrapping_neg() % upper;
        loop {
            let value = self.next();
            if value >= threshold {
                return value % upper;
            }
        }
    }
}

/// Participant-level exact sign test for the median paired gap.
///
/// The null is that positive and negative non-tied participant gaps are equally likely.
/// Exact zero gaps are excluded from the binomial denominator. This is a distribution-free
/// complementary diagnostic, not a randomized treatment-assignment test and not a causal test.
#[derive(Debug, Clone, PartialEq)]
pub struct SignTestSummary {
    pub positive_count: usize,
    pub negative_count: usize,
    pub tie_count: usize,
    pub non_tied_count: usize,
    pub p_two_sided_exact: f64,
}

/// Compute a two-sided exact binomial sign-test p-value for participant-level differences.
pub fn exact_sign_test(differences: &[f64]) -> Result<SignTestSummary, MetricError> {
    if differences.is_empty() {
        return Err(MetricError::EmptyInput);
    }
    let mut positive_count = 0usize;
    let mut negative_count = 0usize;
    let mut tie_count = 0usize;
    for (index, value) in differences.iter().enumerate() {
        if !value.is_finite() {
            return Err(MetricError::NonFiniteValue { index });
        }
        if *value > 0.0 {
            positive_count += 1;
        } else if *value < 0.0 {
            negative_count += 1;
        } else {
            tie_count += 1;
        }
    }
    let non_tied_count = positive_count + negative_count;
    let p_two_sided_exact = if non_tied_count == 0 {
        1.0
    } else {
        let lower_tail =
            binomial_symmetric_lower_tail(non_tied_count, positive_count.min(negative_count));
        (2.0 * lower_tail).min(1.0)
    };
    Ok(SignTestSummary {
        positive_count,
        negative_count,
        tie_count,
        non_tied_count,
        p_two_sided_exact,
    })
}

/// P(X <= k) for X ~ Binomial(n, 0.5), evaluated from the central mass downward.
fn binomial_symmetric_lower_tail(n: usize, k: usize) -> f64 {
    let mode = n / 2;
    let mut log_mode_mass = -(n as f64) * std::f64::consts::LN_2;
    for j in 1..=mode {
        log_mode_mass += ((n - j + 1) as f64).ln() - (j as f64).ln();
    }
    let mut mass = log_mode_mass.exp();
    let mut tail = if mode <= k { mass } else { 0.0 };
    for j in (1..=mode).rev() {
        mass *= j as f64 / (n - j + 1) as f64;
        if j - 1 <= k {
            tail += mass;
        }
    }
    tail
}

/// Counts how often aligned correlation edges change polarity or enter/leave a neutral zone.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeSignTransitionSummary {
    pub total_edges: usize,
    pub stable_positive: usize,
    pub stable_negative: usize,
    pub stable_neutral: usize,
    pub positive_to_negative: usize,
    pub negative_to_positive: usize,
    pub positive_to_neutral: usize,
    pub negative_to_neutral: usize,
    pub neutral_to_positive: usize,
    pub neutral_to_negative: usize,
    pub polarity_reversal_count: usize,
    pub sign_state_change_count: usize,
    pub polarity_reversal_fraction: f64,
    pub sign_state_change_fraction: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SignState {
    Negative,
    Neutral,
    Positive,
}

fn classify_sign(value: f64, neutral_threshold: f64) -> SignState {
    if value > neutral_threshold {
        SignState::Positive
    } else if value < -neutral_threshold {
        SignState::Negative
    } else {
        SignState::Neutral
    }
}

/// Sign-transition analysis that verifies exact ROI labels and order before comparison.
pub fn summarize_edge_sign_transitions_labeled(
    a_roi_labels: &[String],
    a: &[Vec<f64>],
    b_roi_labels: &[String],
    b: &[Vec<f64>],
    neutral_threshold: f64,
) -> Result<EdgeSignTransitionSummary, MetricError> {
    let n_a = validate_correlation_matrix(a)?;
    let n_b = validate_correlation_matrix(b)?;
    validate_roi_labels(a_roi_labels, n_a)?;
    validate_roi_labels(b_roi_labels, n_b)?;
    if a_roi_labels != b_roi_labels {
        return Err(MetricError::RoiOrderMismatch);
    }
    summarize_edge_sign_transitions(a, b, neutral_threshold)
}

/// Summarize sign transitions over aligned upper-triangle edges.
///
/// A non-zero neutral_threshold treats correlations in [-threshold, +threshold] as neutral.
/// Run this over a predeclared threshold sensitivity grid; never choose the threshold after
/// inspecting which value best supports a state contrast. ROI ordering must be identical.
pub fn summarize_edge_sign_transitions(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    neutral_threshold: f64,
) -> Result<EdgeSignTransitionSummary, MetricError> {
    if !neutral_threshold.is_finite() || !(0.0..=1.0).contains(&neutral_threshold) {
        return Err(MetricError::InvalidSignThreshold);
    }
    let n_a = validate_correlation_matrix(a)?;
    let n_b = validate_correlation_matrix(b)?;
    if n_a != n_b {
        return Err(MetricError::MatrixSizeMismatch {
            left: n_a,
            right: n_b,
        });
    }

    let mut summary = EdgeSignTransitionSummary {
        total_edges: n_a * (n_a - 1) / 2,
        stable_positive: 0,
        stable_negative: 0,
        stable_neutral: 0,
        positive_to_negative: 0,
        negative_to_positive: 0,
        positive_to_neutral: 0,
        negative_to_neutral: 0,
        neutral_to_positive: 0,
        neutral_to_negative: 0,
        polarity_reversal_count: 0,
        sign_state_change_count: 0,
        polarity_reversal_fraction: 0.0,
        sign_state_change_fraction: 0.0,
    };

    for i in 0..n_a {
        for j in (i + 1)..n_a {
            let left = classify_sign(a[i][j], neutral_threshold);
            let right = classify_sign(b[i][j], neutral_threshold);
            match (left, right) {
                (SignState::Positive, SignState::Positive) => summary.stable_positive += 1,
                (SignState::Negative, SignState::Negative) => summary.stable_negative += 1,
                (SignState::Neutral, SignState::Neutral) => summary.stable_neutral += 1,
                (SignState::Positive, SignState::Negative) => summary.positive_to_negative += 1,
                (SignState::Negative, SignState::Positive) => summary.negative_to_positive += 1,
                (SignState::Positive, SignState::Neutral) => summary.positive_to_neutral += 1,
                (SignState::Negative, SignState::Neutral) => summary.negative_to_neutral += 1,
                (SignState::Neutral, SignState::Positive) => summary.neutral_to_positive += 1,
                (SignState::Neutral, SignState::Negative) => summary.neutral_to_negative += 1,
            }
        }
    }

    summary.polarity_reversal_count = summary.positive_to_negative + summary.negative_to_positive;
    summary.sign_state_change_count = summary.total_edges
        - summary.stable_positive
        - summary.stable_negative
        - summary.stable_neutral;
    summary.polarity_reversal_fraction =
        summary.polarity_reversal_count as f64 / summary.total_edges as f64;
    summary.sign_state_change_fraction =
        summary.sign_state_change_count as f64 / summary.total_edges as f64;
    Ok(summary)
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
        assert_eq!(
            weighted_jaccard(&[0.0, 0.0], &[0.0, 0.0]),
            Err(MetricError::ZeroTotalMass)
        );
    }

    #[test]
    fn invalid_edge_vectors_are_rejected() {
        assert!(matches!(
            weighted_jaccard(&[0.2], &[0.2, 0.3]),
            Err(MetricError::LengthMismatch { .. })
        ));
        assert_eq!(
            weighted_jaccard(&[f64::NAN], &[0.2]),
            Err(MetricError::NonFiniteValue { index: 0 })
        );
        assert_eq!(
            weighted_jaccard(&[-0.1], &[0.2]),
            Err(MetricError::NegativeWeight { index: 0 })
        );
    }

    #[test]
    fn pair_aware_sign_masks_match_the_published_definition() {
        let a = vec![
            vec![1.0, 0.5, -0.4],
            vec![0.5, 1.0, 0.6],
            vec![-0.4, 0.6, 1.0],
        ];
        let b = vec![
            vec![1.0, -0.5, -0.2],
            vec![-0.5, 1.0, 0.6],
            vec![-0.2, 0.6, 1.0],
        ];
        // Only the shared positive edge (1,2) is compared: 0.6 vs 0.6.
        assert_eq!(
            weighted_jaccard_correlation_matrices(
                &a,
                &b,
                CorrelationWjMode::PositiveEdgesInBoth,
                1
            )
            .unwrap(),
            1.0
        );
        // Shared negative edge (0,2): |-.4| vs |-.2| = 0.5 WJ.
        assert!(
            (weighted_jaccard_correlation_matrices(
                &a,
                &b,
                CorrelationWjMode::NegativeEdgesInBoth,
                1
            )
            .unwrap()
                - 0.5)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn invalid_matrices_and_empty_sign_masks_fail_closed() {
        let asymmetric = vec![vec![1.0, 0.4], vec![0.3, 1.0]];
        assert!(matches!(
            weighted_jaccard_correlation_matrices(
                &asymmetric,
                &asymmetric,
                CorrelationWjMode::Absolute,
                1
            ),
            Err(MetricError::AsymmetricMatrix { .. })
        ));
        let no_shared_positive = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert!(matches!(
            weighted_jaccard_correlation_matrices(
                &no_shared_positive,
                &no_shared_positive,
                CorrelationWjMode::PositiveEdgesInBoth,
                1
            ),
            Err(MetricError::InsufficientSelectedEdges { .. })
        ));
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
        assert!((a.sign_test.p_two_sided_exact - 0.0625).abs() < 1e-12);
    }

    #[test]
    fn rcaf_rejects_unpaired_or_invalid_similarity() {
        assert!(matches!(
            summarize_rcaf(&[0.5], &[0.4, 0.3], 100, 0.05, 1),
            Err(MetricError::LengthMismatch { .. })
        ));
        assert_eq!(
            summarize_rcaf(&[f64::NAN, 0.5], &[0.4, 0.3], 100, 0.05, 1),
            Err(MetricError::InvalidSimilarity { index: 0 })
        );
        assert!(matches!(
            summarize_rcaf(&[0.5, 0.6], &[0.4, 0.3], 99, 0.05, 1),
            Err(MetricError::InvalidBootstrapDraws { .. })
        ));
    }

    #[test]
    fn exact_sign_test_is_paired_tie_aware_and_exact() {
        let all_positive = exact_sign_test(&[1.0, 2.0, 3.0, 4.0, 0.0]).unwrap();
        assert_eq!(all_positive.positive_count, 4);
        assert_eq!(all_positive.negative_count, 0);
        assert_eq!(all_positive.tie_count, 1);
        assert_eq!(all_positive.non_tied_count, 4);
        assert!((all_positive.p_two_sided_exact - 0.125).abs() < 1e-12);

        let balanced = exact_sign_test(&[1.0, -1.0, 0.0, 2.0, -2.0]).unwrap();
        assert_eq!(balanced.positive_count, 2);
        assert_eq!(balanced.negative_count, 2);
        assert_eq!(balanced.tie_count, 1);
        assert_eq!(balanced.p_two_sided_exact, 1.0);

        let all_tied = exact_sign_test(&[0.0, 0.0]).unwrap();
        assert_eq!(all_tied.non_tied_count, 0);
        assert_eq!(all_tied.p_two_sided_exact, 1.0);
        assert_eq!(
            exact_sign_test(&[f64::NAN]),
            Err(MetricError::NonFiniteValue { index: 0 })
        );
    }

    #[test]
    fn edge_sign_transitions_expose_polarity_changes_and_neutral_edges() {
        let a = vec![
            vec![1.0, 0.5, -0.4],
            vec![0.5, 1.0, 0.01],
            vec![-0.4, 0.01, 1.0],
        ];
        let b = vec![
            vec![1.0, -0.5, -0.1],
            vec![-0.5, 1.0, 0.4],
            vec![-0.1, 0.4, 1.0],
        ];
        let transitions = summarize_edge_sign_transitions(&a, &b, 0.05).unwrap();
        assert_eq!(transitions.total_edges, 3);
        assert_eq!(transitions.positive_to_negative, 1);
        assert_eq!(transitions.stable_negative, 1);
        assert_eq!(transitions.neutral_to_positive, 1);
        assert_eq!(transitions.polarity_reversal_count, 1);
        assert_eq!(transitions.sign_state_change_count, 2);
        assert!((transitions.polarity_reversal_fraction - 1.0 / 3.0).abs() < 1e-12);
        assert!((transitions.sign_state_change_fraction - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(
            summarize_edge_sign_transitions(&a, &b, f64::NAN),
            Err(MetricError::InvalidSignThreshold)
        );
    }

    #[test]
    fn labeled_matrix_comparison_rejects_roi_order_mismatch() {
        let a = vec![vec![1.0, 0.2], vec![0.2, 1.0]];
        let b = a.clone();
        let labels_a = vec!["ROI-1".to_string(), "ROI-2".to_string()];
        let labels_b = vec!["ROI-2".to_string(), "ROI-1".to_string()];
        assert_eq!(
            weighted_jaccard_labeled_correlation_matrices(
                &labels_a,
                &a,
                &labels_b,
                &b,
                CorrelationWjMode::Absolute,
                1
            ),
            Err(MetricError::RoiOrderMismatch)
        );
        assert_eq!(
            weighted_jaccard_labeled_correlation_matrices(
                &["ROI-1".to_string()],
                &a,
                &labels_a,
                &b,
                CorrelationWjMode::Absolute,
                1
            ),
            Err(MetricError::RoiLabelCountMismatch {
                labels: 1,
                regions: 2
            })
        );
    }

    #[test]
    fn participant_keyed_summary_rejects_duplicate_or_empty_ids() {
        let valid = vec![
            ParticipantSimilarity {
                participant_id: "p1".into(),
                within_similarity: 0.6,
                between_similarity: 0.5,
            },
            ParticipantSimilarity {
                participant_id: "p2".into(),
                within_similarity: 0.7,
                between_similarity: 0.6,
            },
        ];
        let summary = summarize_rcaf_by_participant(&valid, 100, 0.05, 7).unwrap();
        assert_eq!(summary.participant_count, 2);
        assert!((summary.median_paired_gap - 0.1).abs() < 1e-12);

        let duplicate = vec![valid[0].clone(), valid[0].clone()];
        assert_eq!(
            summarize_rcaf_by_participant(&duplicate, 100, 0.05, 7),
            Err(MetricError::DuplicateParticipantId { index: 1 })
        );
        let empty = vec![
            ParticipantSimilarity {
                participant_id: " ".into(),
                within_similarity: 0.6,
                between_similarity: 0.5,
            },
            valid[1].clone(),
        ];
        assert_eq!(
            summarize_rcaf_by_participant(&empty, 100, 0.05, 7),
            Err(MetricError::EmptyParticipantId { index: 0 })
        );
    }

    #[test]
    fn exact_sign_test_has_expected_resolution_at_n23() {
        let gaps = vec![0.1; 23];
        let result = exact_sign_test(&gaps).unwrap();
        assert!((result.p_two_sided_exact - 2.0 / 8_388_608.0).abs() < 1e-18);
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
