// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Language-profiled phonological feature and tone error metrics.
//!
//! Inspired by Chen, Adeleke & Downey (AfricaNLP 2026):
//! https://aclanthology.org/2026.africanlp-main.14/
//!
//! This module intentionally provides no universal phonological inventory or
//! G2P. Callers supply versioned feature vectors and tone dimensions from a
//! language-appropriate, independently governed feature profile.
//!
//! Alignment is global Needleman-Wunsch with a caller-declared gap penalty.
//! The default penalty is 1.0. Exact parity with an external implementation
//! requires matching its gap and tie policies.

use std::collections::HashSet;
use std::fmt;

/// One segment represented by sparse phonological feature values.
///
/// Values must be -1 (feature absent), 0 (not applicable/unspecified), or
/// +1 (feature present). Dimension meanings come from the caller-owned profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhonologicalFeatureVector {
    values: Vec<i8>,
}

impl PhonologicalFeatureVector {
    /// Construct a feature vector, rejecting values outside {-1, 0, +1}.
    pub fn new(values: Vec<i8>) -> Result<Self, FeatureMetricError> {
        if values.is_empty() {
            return Err(FeatureMetricError::EmptyFeatureVector);
        }
        if let Some((index, value)) = values
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !matches!(value, -1..=1))
        {
            return Err(FeatureMetricError::InvalidFeatureValue { index, value });
        }
        Ok(Self { values })
    }

    /// Exact signed feature values in caller-defined dimension order.
    pub fn values(&self) -> &[i8] {
        &self.values
    }

    /// Number of feature dimensions.
    pub fn dimensions(&self) -> usize {
        self.values.len()
    }

    fn has_active_value(&self, dimensions: impl Iterator<Item = usize>) -> bool {
        dimensions.any(|index| self.values[index] != 0)
    }
}

/// Scoring configuration tied to one versioned phonological feature profile.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureScoringConfig {
    profile_id: String,
    profile_revision: String,
    dimension_count: usize,
    tone_dimensions: Vec<usize>,
    gap_penalty: f64,
}

impl FeatureScoringConfig {
    /// Construct a profile-bound scoring configuration.
    ///
    /// An empty tone dimension list is valid for a non-tonal profile; TER will
    /// be non-estimable rather than reported as zero.
    pub fn new(
        profile_id: impl Into<String>,
        profile_revision: impl Into<String>,
        dimension_count: usize,
        tone_dimensions: Vec<usize>,
        gap_penalty: f64,
    ) -> Result<Self, FeatureMetricError> {
        let profile_id = profile_id.into();
        let profile_revision = profile_revision.into();

        if profile_id.trim().is_empty() {
            return Err(FeatureMetricError::EmptyProfileId);
        }
        if profile_revision.trim().is_empty() {
            return Err(FeatureMetricError::EmptyProfileRevision);
        }
        if dimension_count == 0 {
            return Err(FeatureMetricError::ZeroFeatureDimensions);
        }
        if !gap_penalty.is_finite() || gap_penalty < 0.0 {
            return Err(FeatureMetricError::InvalidGapPenalty);
        }

        let mut seen = HashSet::with_capacity(tone_dimensions.len());
        for &index in &tone_dimensions {
            if index >= dimension_count {
                return Err(FeatureMetricError::ToneDimensionOutOfRange {
                    index,
                    dimension_count,
                });
            }
            if !seen.insert(index) {
                return Err(FeatureMetricError::DuplicateToneDimension(index));
            }
        }

        Ok(Self {
            profile_id,
            profile_revision,
            dimension_count,
            tone_dimensions,
            gap_penalty,
        })
    }

    /// Stable feature-profile identifier.
    pub fn profile_id(&self) -> &str { &self.profile_id }

    /// Immutable feature-profile revision.
    pub fn profile_revision(&self) -> &str { &self.profile_revision }

    /// Number of feature dimensions.
    pub fn dimension_count(&self) -> usize { self.dimension_count }

    /// Caller-declared tone dimensions.
    pub fn tone_dimensions(&self) -> &[usize] { &self.tone_dimensions }

    /// Alignment gap penalty.
    pub fn gap_penalty(&self) -> f64 { self.gap_penalty }
}

/// Alignment operation over phonological feature segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureAlignmentOp {
    /// Identical feature vectors.
    Match,
    /// Different feature vectors.
    Substitution,
    /// A reference segment has no hypothesis segment.
    Deletion,
    /// A hypothesis segment has no reference segment.
    Insertion,
}

/// One column in the deterministic global alignment.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureAlignmentColumn {
    /// Reference segment index, or None for an insertion.
    pub reference_index: Option<usize>,
    /// Hypothesis segment index, or None for a deletion.
    pub hypothesis_index: Option<usize>,
    /// Operation selected by the backtrace.
    pub operation: FeatureAlignmentOp,
    /// Full-feature cost for this column.
    pub feature_cost: f64,
    /// Tone-only cost for this column.
    pub tone_cost: f64,
    /// Number of active full-feature dimensions in a paired column.
    pub active_feature_dimensions: usize,
    /// Number of active tone dimensions in a paired column.
    pub active_tone_dimensions: usize,
}

/// A normalized metric numerator and reference-segment denominator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatureMetricValue {
    /// Sum of alignment costs.
    pub numerator: f64,
    /// Number of reference segments used as denominator.
    pub denominator: usize,
    /// Numerator divided by denominator.
    pub value: f64,
}

/// Why a metric could not be estimated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureMetricNonEstimableReason {
    /// FER has no active phonological feature values in the reference.
    NoActiveReferenceFeatures,
    /// No tone dimensions were declared for this profile.
    NoToneDimensionsConfigured,
    /// No active tone values occur in the reference segments.
    NoActiveReferenceToneFeatures,
}

/// Result from language-profiled phonological evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct PhonologicalFeatureResult {
    /// Feature-profile identifier.
    pub profile_id: String,
    /// Feature-profile revision.
    pub profile_revision: String,
    /// Number of feature dimensions.
    pub dimension_count: usize,
    /// Tone dimensions used by TER.
    pub tone_dimensions: Vec<usize>,
    /// Gap penalty used for alignment and missing-segment costs.
    pub gap_penalty: f64,
    /// Number of reference segments.
    pub reference_segments: usize,
    /// Number of hypothesis segments.
    pub hypothesis_segments: usize,
    /// Exact feature-vector matches.
    pub matches: usize,
    /// Non-identical aligned segment pairs.
    pub substitutions: usize,
    /// Reference segments aligned to a gap.
    pub deletions: usize,
    /// Hypothesis segments aligned to a gap.
    pub insertions: usize,
    /// Aligned pairs with no active full-feature dimensions.
    pub unscorable_feature_pairs: usize,
    /// Reference segments carrying at least one active feature.
    pub active_reference_feature_segments: usize,
    /// Reference segments carrying at least one active tone feature.
    pub active_reference_tone_segments: usize,
    /// FER, or None with a reason when the reference has no active features.
    pub fer: Option<FeatureMetricValue>,
    /// FER non-estimability reason, if any.
    pub fer_non_estimable: Option<FeatureMetricNonEstimableReason>,
    /// TER, or None when tone is not applicable/estimable.
    pub ter: Option<FeatureMetricValue>,
    /// TER non-estimability reason, if any.
    pub ter_non_estimable: Option<FeatureMetricNonEstimableReason>,
    /// Alignment columns in source order.
    pub alignment: Vec<FeatureAlignmentColumn>,
}

/// Invalid input or configuration for phonological feature scoring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeatureMetricError {
    /// A feature vector must contain at least one dimension.
    EmptyFeatureVector,
    /// A feature value must be -1, 0, or +1.
    InvalidFeatureValue { index: usize, value: i8 },
    /// A feature profile identifier is required.
    EmptyProfileId,
    /// A feature profile revision is required.
    EmptyProfileRevision,
    /// At least one feature dimension is required.
    ZeroFeatureDimensions,
    /// A segment has the wrong vector width.
    FeatureDimensionMismatch {
        segment_index: usize,
        expected: usize,
        actual: usize,
        is_reference: bool,
    },
    /// Tone dimension is outside the feature profile.
    ToneDimensionOutOfRange { index: usize, dimension_count: usize },
    /// Tone dimension was listed more than once.
    DuplicateToneDimension(usize),
    /// Gap penalty must be finite and non-negative.
    InvalidGapPenalty,
    /// FER and TER require a non-empty reference.
    EmptyReference,
}

impl fmt::Display for FeatureMetricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyFeatureVector => write!(f, "phonological feature vector is empty"),
            Self::InvalidFeatureValue { index, value } => write!(
                f,
                "feature value {value} at dimension {index} is outside {{-1, 0, +1}}"
            ),
            Self::EmptyProfileId => write!(f, "feature profile ID must be non-empty"),
            Self::EmptyProfileRevision => write!(f, "feature profile revision must be non-empty"),
            Self::ZeroFeatureDimensions => write!(f, "feature profile has zero dimensions"),
            Self::FeatureDimensionMismatch { segment_index, expected, actual, is_reference } =>
                write!(f, "{} segment {segment_index} has {actual} dimensions; expected {expected}",
                    if *is_reference { "reference" } else { "hypothesis" }),
            Self::ToneDimensionOutOfRange { index, dimension_count } =>
                write!(f, "tone dimension {index} is outside feature profile width {dimension_count}"),
            Self::DuplicateToneDimension(index) =>
                write!(f, "tone dimension {index} was declared more than once"),
            Self::InvalidGapPenalty => write!(f, "gap penalty must be finite and non-negative"),
            Self::EmptyReference => write!(f, "feature scoring requires a non-empty reference"),
        }
    }
}
impl std::error::Error for FeatureMetricError {}

/// Compute FER and TER using a caller-supplied language feature profile.
///
/// Alignment uses full-feature masked substitution costs and the declared gap
/// penalty. Ties resolve diagonal first, then deletion, then insertion. FER
/// and TER use the same alignment.
///
/// For paired segments, distance is the fraction of active dimensions whose
/// values differ. A dimension is active when either side is non-zero. A
/// dimension zero on both sides is not applicable and excluded.
///
/// Insertions/deletions contribute the configured gap penalty to FER. They
/// contribute the same penalty to TER only when the gapped segment carries an
/// active tone feature. TER is non-estimable when no reference segment carries
/// an active tone feature, or no tone dimensions are configured.
///
/// No G2P, tokenization, normalization, or inventory is performed here.
///
/// # Errors
///
/// Returns a typed error for empty reference, malformed vectors/profile,
/// invalid tone dimensions, or invalid gap penalty.
pub fn phonological_feature_error_rates(
    reference: &[PhonologicalFeatureVector],
    hypothesis: &[PhonologicalFeatureVector],
    config: &FeatureScoringConfig,
) -> Result<PhonologicalFeatureResult, FeatureMetricError> {
    if reference.is_empty() {
        return Err(FeatureMetricError::EmptyReference);
    }
    validate_widths(reference, config.dimension_count, true)?;
    validate_widths(hypothesis, config.dimension_count, false)?;

    let n = reference.len();
    let m = hypothesis.len();
    let mut dp = vec![vec![0.0_f64; m + 1]; n + 1];

    for i in 1..=n { dp[i][0] = dp[i - 1][0] + config.gap_penalty; }
    for j in 1..=m { dp[0][j] = dp[0][j - 1] + config.gap_penalty; }

    for i in 1..=n {
        for j in 1..=m {
            let sub_cost = masked_cost(&reference[i - 1], &hypothesis[j - 1], 0..config.dimension_count).0;
            let substitution = dp[i - 1][j - 1] + sub_cost;
            let deletion = dp[i - 1][j] + config.gap_penalty;
            let insertion = dp[i][j - 1] + config.gap_penalty;
            dp[i][j] = substitution.min(deletion).min(insertion);
        }
    }

    let mut reverse_columns = Vec::with_capacity(n.max(m));
    let mut i = n;
    let mut j = m;
    while i > 0 || j > 0 {
        if i > 0 && j > 0 {
            let (feature_cost, active_feature_dimensions) =
                masked_cost(&reference[i - 1], &hypothesis[j - 1], 0..config.dimension_count);
            if close(dp[i][j], dp[i - 1][j - 1] + feature_cost) {
                let (tone_cost, active_tone_dimensions) =
                    masked_cost(&reference[i - 1], &hypothesis[j - 1], config.tone_dimensions.iter().copied());
                let operation = if reference[i - 1] == hypothesis[j - 1] {
                    FeatureAlignmentOp::Match
                } else {
                    FeatureAlignmentOp::Substitution
                };
                reverse_columns.push(FeatureAlignmentColumn {
                    reference_index: Some(i - 1),
                    hypothesis_index: Some(j - 1),
                    operation,
                    feature_cost,
                    tone_cost,
                    active_feature_dimensions,
                    active_tone_dimensions,
                });
                i -= 1;
                j -= 1;
                continue;
            }
        }

        if i > 0 && close(dp[i][j], dp[i - 1][j] + config.gap_penalty) {
            let segment = &reference[i - 1];
            let tone_active = segment.has_active_value(config.tone_dimensions.iter().copied());
            reverse_columns.push(FeatureAlignmentColumn {
                reference_index: Some(i - 1),
                hypothesis_index: None,
                operation: FeatureAlignmentOp::Deletion,
                feature_cost: config.gap_penalty,
                tone_cost: if tone_active { config.gap_penalty } else { 0.0 },
                active_feature_dimensions: config.dimension_count,
                active_tone_dimensions: usize::from(tone_active),
            });
            i -= 1;
            continue;
        }

        if j > 0 && close(dp[i][j], dp[i][j - 1] + config.gap_penalty) {
            let segment = &hypothesis[j - 1];
            let tone_active = segment.has_active_value(config.tone_dimensions.iter().copied());
            reverse_columns.push(FeatureAlignmentColumn {
                reference_index: None,
                hypothesis_index: Some(j - 1),
                operation: FeatureAlignmentOp::Insertion,
                feature_cost: config.gap_penalty,
                tone_cost: if tone_active { config.gap_penalty } else { 0.0 },
                active_feature_dimensions: config.dimension_count,
                active_tone_dimensions: usize::from(tone_active),
            });
            j -= 1;
            continue;
        }

        unreachable!("Needleman-Wunsch backtrace has no predecessor");
    }
    reverse_columns.reverse();

    let mut feature_numerator = 0.0;
    let mut tone_numerator = 0.0;
    let mut matches = 0;
    let mut substitutions = 0;
    let mut deletions = 0;
    let mut insertions = 0;
    let mut unscorable_feature_pairs = 0;

    for column in &reverse_columns {
        feature_numerator += column.feature_cost;
        tone_numerator += column.tone_cost;
        match column.operation {
            FeatureAlignmentOp::Match => matches += 1,
            FeatureAlignmentOp::Substitution => substitutions += 1,
            FeatureAlignmentOp::Deletion => deletions += 1,
            FeatureAlignmentOp::Insertion => insertions += 1,
        }
        if column.reference_index.is_some()
            && column.hypothesis_index.is_some()
            && column.active_feature_dimensions == 0
        {
            unscorable_feature_pairs += 1;
        }
    }

    let active_reference_feature_segments = reference.iter()
        .filter(|segment| segment.values.iter().any(|&value| value != 0))
        .count();
    let active_reference_tone_segments = reference.iter()
        .filter(|segment| segment.has_active_value(config.tone_dimensions.iter().copied()))
        .count();

    let (fer, fer_non_estimable) = if active_reference_feature_segments == 0 {
        (None, Some(FeatureMetricNonEstimableReason::NoActiveReferenceFeatures))
    } else {
        (Some(FeatureMetricValue {
            numerator: feature_numerator,
            denominator: n,
            value: feature_numerator / n as f64,
        }), None)
    };

    let (ter, ter_non_estimable) = if config.tone_dimensions.is_empty() {
        (None, Some(FeatureMetricNonEstimableReason::NoToneDimensionsConfigured))
    } else if active_reference_tone_segments == 0 {
        (None, Some(FeatureMetricNonEstimableReason::NoActiveReferenceToneFeatures))
    } else {
        (Some(FeatureMetricValue {
            numerator: tone_numerator,
            denominator: n,
            value: tone_numerator / n as f64,
        }), None)
    };

    Ok(PhonologicalFeatureResult {
        profile_id: config.profile_id.clone(),
        profile_revision: config.profile_revision.clone(),
        dimension_count: config.dimension_count,
        tone_dimensions: config.tone_dimensions.clone(),
        gap_penalty: config.gap_penalty,
        reference_segments: n,
        hypothesis_segments: m,
        matches,
        substitutions,
        deletions,
        insertions,
        unscorable_feature_pairs,
        active_reference_feature_segments,
        active_reference_tone_segments,
        fer,
        fer_non_estimable,
        ter,
        ter_non_estimable,
        alignment: reverse_columns,
    })
}

fn validate_widths(
    segments: &[PhonologicalFeatureVector],
    expected_width: usize,
    is_reference: bool,
) -> Result<(), FeatureMetricError> {
    for (segment_index, segment) in segments.iter().enumerate() {
        if segment.values.len() != expected_width {
            return Err(FeatureMetricError::FeatureDimensionMismatch {
                segment_index,
                expected: expected_width,
                actual: segment.values.len(),
                is_reference,
            });
        }
    }
    Ok(())
}

fn masked_cost(
    reference: &PhonologicalFeatureVector,
    hypothesis: &PhonologicalFeatureVector,
    dimensions: impl Iterator<Item = usize>,
) -> (f64, usize) {
    let mut active = 0usize;
    let mut mismatches = 0usize;
    for index in dimensions {
        let left = reference.values[index];
        let right = hypothesis.values[index];
        if left != 0 || right != 0 {
            active += 1;
            if left != right { mismatches += 1; }
        }
    }
    if active == 0 { (0.0, 0) } else { (mismatches as f64 / active as f64, active) }
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-12
}

#[cfg(test)]
mod tests {
    use super::{
        FeatureAlignmentOp, FeatureMetricError, FeatureMetricNonEstimableReason,
        FeatureScoringConfig, PhonologicalFeatureVector, phonological_feature_error_rates,
    };

    fn v(values: &[i8]) -> PhonologicalFeatureVector {
        PhonologicalFeatureVector::new(values.to_vec()).expect("valid vector")
    }

    fn cfg(tone: Vec<usize>, gap: f64) -> FeatureScoringConfig {
        FeatureScoringConfig::new("synthetic-profile", "v1", 4, tone, gap).expect("valid config")
    }

    #[test]
    fn identical_features_have_zero_fer_and_ter() {
        let r = phonological_feature_error_rates(
            &[v(&[1, -1, 1, -1])], &[v(&[1, -1, 1, -1])], &cfg(vec![3], 1.0)
        ).unwrap();
        assert_eq!(r.fer.unwrap().value, 0.0);
        assert_eq!(r.ter.unwrap().value, 0.0);
        assert_eq!(r.matches, 1);
    }

    #[test]
    fn one_of_four_active_feature_mismatches_costs_one_quarter() {
        let r = phonological_feature_error_rates(
            &[v(&[1, 1, -1, 1])], &[v(&[1, -1, -1, 1])], &cfg(vec![], 1.0)
        ).unwrap();
        assert!((r.fer.unwrap().value - 0.25).abs() < 1e-12);
    }

    #[test]
    fn dimensions_zero_on_both_sides_are_excluded() {
        let r = phonological_feature_error_rates(
            &[v(&[1, 0, 0, -1])], &[v(&[1, 0, 0, -1])], &cfg(vec![], 1.0)
        ).unwrap();
        assert_eq!(r.alignment[0].active_feature_dimensions, 2);
        assert_eq!(r.fer.unwrap().value, 0.0);
    }

    #[test]
    fn zero_against_active_value_is_a_mismatch() {
        let r = phonological_feature_error_rates(
            &[v(&[1, 0, 0, 0])], &[v(&[1, 1, 0, 0])], &cfg(vec![], 1.0)
        ).unwrap();
        assert_eq!(r.alignment[0].active_feature_dimensions, 2);
        assert!((r.fer.unwrap().value - 0.5).abs() < 1e-12);
    }

    #[test]
    fn tone_dimension_is_scored_separately() {
        let r = phonological_feature_error_rates(
            &[v(&[1, 1, 0, 0])], &[v(&[1, -1, 0, 0])], &cfg(vec![1], 1.0)
        ).unwrap();
        assert!((r.fer.unwrap().value - 0.5).abs() < 1e-12);
        assert_eq!(r.ter.unwrap().value, 1.0);
    }

    #[test]
    fn segmental_error_does_not_change_tone_error() {
        let r = phonological_feature_error_rates(
            &[v(&[1, 1, 0, 0])], &[v(&[-1, 1, 0, 0])], &cfg(vec![1], 1.0)
        ).unwrap();
        assert_eq!(r.fer.unwrap().value, 0.5);
        assert_eq!(r.ter.unwrap().value, 0.0);
    }

    #[test]
    fn tone_is_not_estimable_when_reference_has_no_active_tone() {
        let r = phonological_feature_error_rates(
            &[v(&[1, 0, 0, 0]), v(&[-1, 0, 0, 0])],
            &[v(&[1, 0, 0, 0]), v(&[-1, 0, 0, 0])],
            &cfg(vec![1], 1.0)
        ).unwrap();
        assert_eq!(r.ter, None);
        assert_eq!(r.ter_non_estimable, Some(FeatureMetricNonEstimableReason::NoActiveReferenceToneFeatures));
    }

    #[test]
    fn no_tone_dimensions_is_not_reported_as_zero() {
        let r = phonological_feature_error_rates(
            &[v(&[1, 1, 0, 0])], &[v(&[1, 1, 0, 0])], &cfg(vec![], 1.0)
        ).unwrap();
        assert_eq!(r.ter, None);
        assert_eq!(r.ter_non_estimable, Some(FeatureMetricNonEstimableReason::NoToneDimensionsConfigured));
    }

    #[test]
    fn deletion_uses_declared_gap_penalty() {
        let r = phonological_feature_error_rates(
            &[v(&[1, 0, 0, 0]), v(&[-1, 0, 0, 0])],
            &[v(&[1, 0, 0, 0])],
            &cfg(vec![], 1.0)
        ).unwrap();
        assert_eq!(r.deletions, 1);
        assert_eq!(r.fer.unwrap().value, 0.5);
    }

    #[test]
    fn tone_deletion_penalizes_only_a_tone_bearing_segment() {
        let r = phonological_feature_error_rates(
            &[v(&[1, 1, 0, 0]), v(&[-1, 0, 0, 0])],
            &[v(&[-1, 0, 0, 0])],
            &cfg(vec![1], 1.0)
        ).unwrap();
        assert_eq!(r.deletions, 1);
        assert_eq!(r.ter.unwrap().value, 0.5);
    }

    #[test]
    fn insertions_use_the_declared_gap_penalty() {
        let r = phonological_feature_error_rates(
            &[v(&[1, 0, 0, 0])],
            &[v(&[1, 0, 0, 0]), v(&[-1, 0, 0, 0])],
            &cfg(vec![], 1.5)
        ).unwrap();
        assert_eq!(r.insertions, 1);
        assert_eq!(r.fer.unwrap().value, 1.5);
    }

    #[test]
    fn insertion_of_tone_bearing_segment_contributes_to_ter() {
        let r = phonological_feature_error_rates(
            &[v(&[1, 1, 0, 0])],
            &[v(&[1, 1, 0, 0]), v(&[-1, -1, 0, 0])],
            &cfg(vec![1], 1.0)
        ).unwrap();
        assert_eq!(r.ter.unwrap().value, 1.0);
    }

    #[test]
    fn feature_dimension_mismatch_is_rejected() {
        let error = phonological_feature_error_rates(
            &[v(&[1, 0, 0, 0])], &[v(&[1, 0, 0])], &cfg(vec![], 1.0)
        );
        assert_eq!(error, Err(FeatureMetricError::FeatureDimensionMismatch {
            segment_index: 0, expected: 4, actual: 3, is_reference: false
        }));
    }

    #[test]
    fn empty_reference_is_rejected() {
        let error = phonological_feature_error_rates(
            &[], &[v(&[1, 0, 0, 0])], &cfg(vec![], 1.0)
        );
        assert_eq!(error, Err(FeatureMetricError::EmptyReference));
    }

    #[test]
    fn invalid_values_are_rejected() {
        assert_eq!(
            PhonologicalFeatureVector::new(vec![2, 0, 0, 0]),
            Err(FeatureMetricError::InvalidFeatureValue { index: 0, value: 2 })
        );
    }

    #[test]
    fn duplicate_or_out_of_range_tone_dimensions_are_rejected() {
        assert_eq!(
            FeatureScoringConfig::new("p", "v1", 4, vec![1, 1], 1.0),
            Err(FeatureMetricError::DuplicateToneDimension(1))
        );
        assert_eq!(
            FeatureScoringConfig::new("p", "v1", 4, vec![4], 1.0),
            Err(FeatureMetricError::ToneDimensionOutOfRange { index: 4, dimension_count: 4 })
        );
    }

    #[test]
    fn invalid_gap_penalty_is_rejected() {
        assert_eq!(
            FeatureScoringConfig::new("p", "v1", 4, vec![], f64::NAN),
            Err(FeatureMetricError::InvalidGapPenalty)
        );
        assert_eq!(
            FeatureScoringConfig::new("p", "v1", 4, vec![], -0.5),
            Err(FeatureMetricError::InvalidGapPenalty)
        );
    }

    #[test]
    fn alignment_is_deterministic() {
        let reference = vec![v(&[1, 0, 0, 0]), v(&[1, 1, 0, 0]), v(&[-1, 0, 0, 0])];
        let hypothesis = vec![v(&[-1, 0, 0, 0])];
        let cfg = cfg(vec![1], 1.0);
        let first = phonological_feature_error_rates(&reference, &hypothesis, &cfg).unwrap();
        let second = phonological_feature_error_rates(&reference, &hypothesis, &cfg).unwrap();
        assert_eq!(first, second);
    }
}
