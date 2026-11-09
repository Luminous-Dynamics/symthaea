// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical joint causal-set signature for simulation-calibrated inference.
//!
//! The signature combines the complete normalized open-interval abundance
//! profile (including incomparable-pair mass) with the exact longest-chain
//! length. It intentionally does not append the Myrheim ordering fraction or
//! BDG linear bracket as independent features: both are deterministic
//! functions of the same interval histogram and would double-count evidence.
//!
//! This module extracts observables only. It does not define a likelihood,
//! infer a dimension, or claim that the profile is a continuum geometry
//! distance. Joint likelihoods must be calibrated across whole sprinklings,
//! preserving the dependence among pair intervals.

use crate::causal_set_nd::{CausalSetNd, CausalSetNdError};
use crate::interval_profile::{IntervalAbundanceProfile, IntervalProfileError};
use crate::{CausalSet2D, CausalSetError};
use std::error::Error;
use std::fmt;

/// Supported candidate spacetime dimensions for dimension-conditioned chain scaling.
pub const MIN_SIGNATURE_SPACETIME_DIMENSION: usize = 2;
pub const MAX_SIGNATURE_SPACETIME_DIMENSION: usize = 4;

/// Invalid source, profile request, or dimension scaling in a joint signature.
#[derive(Clone, Debug, PartialEq)]
pub enum JointSignatureError {
    InvalidProfileBinCount,
    InvalidProfileLength { expected: usize, actual: usize },
    InvalidProfileValues,
    InsufficientEvents,
    InvalidSpacetimeDimension { dimension: usize },
    NonFiniteScaledChain,
    IntervalProfile(IntervalProfileError),
    CausalSet1p1(CausalSetError),
    CausalSetNd(CausalSetNdError),
}

impl fmt::Display for JointSignatureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidProfileBinCount => {
                write!(
                    f,
                    "joint signature requires a positive interval-profile bin count"
                )
            }
            Self::InvalidProfileLength { expected, actual } => {
                write!(
                    f,
                    "interval profile has {actual} values; expected {expected}"
                )
            }
            Self::InvalidProfileValues => {
                write!(
                    f,
                    "interval profile must be a finite normalized composition"
                )
            }
            Self::InsufficientEvents => {
                write!(f, "joint signature requires at least two events")
            }
            Self::InvalidSpacetimeDimension { dimension } => write!(
                f,
                "spacetime dimension {dimension} is outside {MIN_SIGNATURE_SPACETIME_DIMENSION}..={MAX_SIGNATURE_SPACETIME_DIMENSION}"
            ),
            Self::NonFiniteScaledChain => {
                write!(
                    f,
                    "dimension-normalized longest-chain statistic is not finite"
                )
            }
            Self::IntervalProfile(error) => write!(f, "interval profile failed: {error}"),
            Self::CausalSet1p1(error) => write!(f, "1+1D causal set failed: {error}"),
            Self::CausalSetNd(error) => write!(f, "N-dimensional causal set failed: {error}"),
        }
    }
}

impl Error for JointSignatureError {}

impl From<IntervalProfileError> for JointSignatureError {
    fn from(value: IntervalProfileError) -> Self {
        Self::IntervalProfile(value)
    }
}

impl From<CausalSetError> for JointSignatureError {
    fn from(value: CausalSetError) -> Self {
        Self::CausalSet1p1(value)
    }
}

impl From<CausalSetNdError> for JointSignatureError {
    fn from(value: CausalSetNdError) -> Self {
        Self::CausalSetNd(value)
    }
}

/// Joint finite-causal-set observables for whole-sprinkling calibration.
///
/// `normalized_interval_profile` contains K bins for normalized open-interval
/// cardinality plus one dedicated incomparable-pair mass. The latter means
/// that the profile already preserves the total related-pair fraction.
///
/// `longest_chain_length` is kept as an integer here. Its scaling depends on
/// the candidate dimension and belongs in `features_for_dimension`, where it
/// can be computed separately for every candidate model.
#[derive(Clone, Debug, PartialEq)]
pub struct JointCausalSetSignature {
    pub event_count: usize,
    pub profile_bin_count: usize,
    pub normalized_interval_profile: Vec<f64>,
    pub longest_chain_length: usize,
}

impl JointCausalSetSignature {
    /// Build from a 1+1D Minkowski causal set.
    pub fn from_causal_set_1p1(
        causet: &CausalSet2D,
        profile_bin_count: usize,
    ) -> Result<Self, JointSignatureError> {
        if profile_bin_count == 0 {
            return Err(JointSignatureError::InvalidProfileBinCount);
        }
        let profile = IntervalAbundanceProfile::from_causal_set_1p1(causet)?;
        let normalized_interval_profile = profile.normalized_bins(profile_bin_count)?;
        Ok(Self {
            event_count: causet.len(),
            profile_bin_count,
            normalized_interval_profile,
            longest_chain_length: causet.longest_chain_length(),
        })
    }

    /// Build from a dimension-generic 2+1D or 3+1D Minkowski causal set.
    pub fn from_causal_set_nd(
        causet: &CausalSetNd,
        profile_bin_count: usize,
    ) -> Result<Self, JointSignatureError> {
        if profile_bin_count == 0 {
            return Err(JointSignatureError::InvalidProfileBinCount);
        }
        let profile = IntervalAbundanceProfile::from_causal_set_nd(causet)?;
        let normalized_interval_profile = profile.normalized_bins(profile_bin_count)?;
        Ok(Self {
            event_count: causet.len(),
            profile_bin_count,
            normalized_interval_profile,
            longest_chain_length: causet.longest_chain_length(),
        })
    }

    /// Construct a canonical signature from a validated raw interval histogram
    /// and longest-chain length. This is useful for independent implementations
    /// and exact small-poset fixtures.
    pub fn from_histogram(
        event_count: usize,
        histogram: Vec<u64>,
        longest_chain_length: usize,
        profile_bin_count: usize,
    ) -> Result<Self, JointSignatureError> {
        if profile_bin_count == 0 {
            return Err(JointSignatureError::InvalidProfileBinCount);
        }
        if longest_chain_length > event_count {
            return Err(JointSignatureError::NonFiniteScaledChain);
        }
        let profile = IntervalAbundanceProfile::from_histogram(event_count, histogram)?;
        let normalized_interval_profile = profile.normalized_bins(profile_bin_count)?;
        Ok(Self {
            event_count,
            profile_bin_count,
            normalized_interval_profile,
            longest_chain_length,
        })
    }

    /// Return the complete feature vector for a candidate spacetime dimension.
    ///
    /// The vector is:
    /// `[normalized interval profile bins..., L / N^(1/D)]`.
    ///
    /// No ordering fraction or BDG bracket is appended, because those are
    /// deterministic summaries of the profile's raw interval histogram.
    /// Whole-sprinkling calibration must preserve covariance among these
    /// features; pair intervals are not independent observations.
    pub fn features_for_dimension(
        &self,
        spacetime_dimension: usize,
    ) -> Result<Vec<f64>, JointSignatureError> {
        if !(MIN_SIGNATURE_SPACETIME_DIMENSION..=MAX_SIGNATURE_SPACETIME_DIMENSION)
            .contains(&spacetime_dimension)
        {
            return Err(JointSignatureError::InvalidSpacetimeDimension {
                dimension: spacetime_dimension,
            });
        }
        if self.event_count == 0 {
            return Err(JointSignatureError::NonFiniteScaledChain);
        }
        let scale = (self.event_count as f64).powf(1.0 / spacetime_dimension as f64);
        let scaled_chain = self.longest_chain_length as f64 / scale;
        if !scaled_chain.is_finite() || scaled_chain < 0.0 {
            return Err(JointSignatureError::NonFiniteScaledChain);
        }

        let mut features = self.normalized_interval_profile.clone();
        features.push(scaled_chain);
        Ok(features)
    }

    /// Return a non-redundant, simplex-aware feature vector for joint
    /// synthetic-likelihood calibration.
    ///
    /// The normalized interval profile has K+1 components but only K degrees
    /// of freedom because its entries sum to one. This method applies an
    /// isometric log-ratio (ILR) transform after a declared Jeffreys-style
    /// half-count pseudocount (0.5 / choose(N,2)) is added to each component.
    /// It then appends the candidate-dimension-normalized longest chain:
    ///
    /// [ILR(profile)[0..K], L / N^(1/D)].
    ///
    /// The smoothing rule is fixed by event count and does not depend on the
    /// observed profile. ILR coordinates preserve Aitchison geometry and
    /// remove the singular simplex direction. Calibration must still use
    /// whole-sprinkling replicates because interval observations are dependent.
    pub fn ilr_features_for_dimension(
        &self,
        spacetime_dimension: usize,
    ) -> Result<Vec<f64>, JointSignatureError> {
        if !(MIN_SIGNATURE_SPACETIME_DIMENSION..=MAX_SIGNATURE_SPACETIME_DIMENSION)
            .contains(&spacetime_dimension)
        {
            return Err(JointSignatureError::InvalidSpacetimeDimension {
                dimension: spacetime_dimension,
            });
        }
        if self.event_count < 2 {
            return Err(JointSignatureError::InsufficientEvents);
        }
        let expected = self.profile_bin_count + 1;
        if self.normalized_interval_profile.len() != expected {
            return Err(JointSignatureError::InvalidProfileLength {
                expected,
                actual: self.normalized_interval_profile.len(),
            });
        }
        if self
            .normalized_interval_profile
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(JointSignatureError::InvalidProfileValues);
        }
        let profile_sum = self.normalized_interval_profile.iter().sum::<f64>();
        if !profile_sum.is_finite() || (profile_sum - 1.0).abs() > 1.0e-8 {
            return Err(JointSignatureError::InvalidProfileValues);
        }

        let total_pairs = self.event_count as f64 * (self.event_count - 1) as f64 / 2.0;
        let pseudocount = 0.5 / total_pairs;
        let log_composition: Vec<f64> = self
            .normalized_interval_profile
            .iter()
            .map(|value| (value + pseudocount).ln())
            .collect();
        let categories = log_composition.len();
        let mut ilr = Vec::with_capacity(categories);
        let mut prefix_log_sum = 0.0;
        for prefix_count in 1..categories {
            prefix_log_sum += log_composition[prefix_count - 1];
            let count = prefix_count as f64;
            let coordinate = (count / (count + 1.0)).sqrt()
                * (prefix_log_sum / count - log_composition[prefix_count]);
            if !coordinate.is_finite() {
                return Err(JointSignatureError::InvalidProfileValues);
            }
            ilr.push(coordinate);
        }

        let scale = (self.event_count as f64).powf(1.0 / spacetime_dimension as f64);
        let scaled_chain = self.longest_chain_length as f64 / scale;
        if !scaled_chain.is_finite() || scaled_chain < 0.0 {
            return Err(JointSignatureError::NonFiniteScaledChain);
        }
        ilr.push(scaled_chain);
        Ok(ilr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_plus_one_signature_contains_profile_and_scaled_chain_without_duplicate_fraction() {
        let causet = CausalSet2D::sprinkle_fixed_cardinality(0x51A7, 256, 2.0).unwrap();
        let signature = JointCausalSetSignature::from_causal_set_1p1(&causet, 8).unwrap();

        assert_eq!(signature.event_count, 256);
        assert_eq!(signature.profile_bin_count, 8);
        assert_eq!(signature.normalized_interval_profile.len(), 9);
        assert_eq!(
            signature.longest_chain_length,
            causet.longest_chain_length()
        );

        let profile_sum: f64 = signature.normalized_interval_profile.iter().sum();
        assert!((profile_sum - 1.0).abs() < 1.0e-12);

        let features = signature.features_for_dimension(2).unwrap();
        assert_eq!(features.len(), 10);
        let expected_chain = signature.longest_chain_length as f64 / 256.0_f64.sqrt();
        assert!((features[9] - expected_chain).abs() < 1.0e-12);
    }

    #[test]
    fn nd_signature_is_candidate_dimension_conditioned() {
        let causet = CausalSetNd::sprinkle_fixed_cardinality(0xC0FFEE, 192, 3, 1.0).unwrap();
        let signature = JointCausalSetSignature::from_causal_set_nd(&causet, 6).unwrap();

        assert_eq!(signature.normalized_interval_profile.len(), 7);
        assert_eq!(
            signature.longest_chain_length,
            causet.longest_chain_length()
        );

        let d3 = signature.features_for_dimension(3).unwrap();
        let d4 = signature.features_for_dimension(4).unwrap();
        assert_eq!(d3.len(), 8);
        assert_eq!(d4.len(), 8);
        assert_eq!(&d3[..7], &d4[..7]);
        assert!(d3[7] < d4[7]);
    }

    #[test]
    fn ilr_features_remove_simplex_redundancy_and_smooth_zero_bins() {
        let signature = JointCausalSetSignature {
            event_count: 64,
            profile_bin_count: 4,
            normalized_interval_profile: vec![0.0, 0.0, 0.2, 0.3, 0.5],
            longest_chain_length: 12,
        };
        let features = signature.ilr_features_for_dimension(2).unwrap();
        assert_eq!(features.len(), 5);
        assert!(features.iter().all(|value| value.is_finite()));
        assert!((features[4] - 12.0 / 64.0_f64.sqrt()).abs() < 1.0e-12);
    }

    #[test]
    fn ilr_features_reject_non_normalized_or_malformed_profiles() {
        let non_normalized = JointCausalSetSignature {
            event_count: 64,
            profile_bin_count: 2,
            normalized_interval_profile: vec![0.2, 0.2, 0.2],
            longest_chain_length: 8,
        };
        assert_eq!(
            non_normalized.ilr_features_for_dimension(2),
            Err(JointSignatureError::InvalidProfileValues)
        );

        let wrong_length = JointCausalSetSignature {
            event_count: 64,
            profile_bin_count: 3,
            normalized_interval_profile: vec![0.5, 0.5],
            longest_chain_length: 8,
        };
        assert!(matches!(
            wrong_length.ilr_features_for_dimension(2),
            Err(JointSignatureError::InvalidProfileLength { .. })
        ));
    }

    #[test]
    fn histogram_constructor_rejects_invalid_chain_and_profile() {
        assert!(matches!(
            JointCausalSetSignature::from_histogram(4, vec![1, 1, 0], 5, 4),
            Err(JointSignatureError::NonFiniteScaledChain)
        ));
        assert!(matches!(
            JointCausalSetSignature::from_histogram(4, vec![1, 1, 0], 2, 0),
            Err(JointSignatureError::InvalidProfileBinCount)
        ));
    }

    #[test]
    fn out_of_range_dimension_fails_closed() {
        let causet = CausalSet2D::sprinkle_fixed_cardinality(7, 32, 1.0).unwrap();
        let signature = JointCausalSetSignature::from_causal_set_1p1(&causet, 4).unwrap();
        assert!(matches!(
            signature.features_for_dimension(5),
            Err(JointSignatureError::InvalidSpacetimeDimension { dimension: 5 })
        ));
    }
}
