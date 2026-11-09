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
use crate::{CausalSet2D, CausalSetError};
use crate::interval_profile::{IntervalAbundanceProfile, IntervalProfileError};
use std::error::Error;
use std::fmt;

/// Supported candidate spacetime dimensions for dimension-conditioned chain scaling.
pub const MIN_SIGNATURE_SPACETIME_DIMENSION: usize = 2;
pub const MAX_SIGNATURE_SPACETIME_DIMENSION: usize = 4;

/// Invalid source, profile request, or dimension scaling in a joint signature.
#[derive(Clone, Debug, PartialEq)]
pub enum JointSignatureError {
    InvalidProfileBinCount,
    InvalidSpacetimeDimension {
        dimension: usize,
    },
    NonFiniteScaledChain,
    IntervalProfile(IntervalProfileError),
    CausalSet1p1(CausalSetError),
    CausalSetNd(CausalSetNdError),
}

impl fmt::Display for JointSignatureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidProfileBinCount => {
                write!(f, "joint signature requires a positive interval-profile bin count")
            }
            Self::InvalidSpacetimeDimension { dimension } => write!(
                f,
                "spacetime dimension {dimension} is outside {MIN_SIGNATURE_SPACETIME_DIMENSION}..={MAX_SIGNATURE_SPACETIME_DIMENSION}"
            ),
            Self::NonFiniteScaledChain => {
                write!(f, "dimension-normalized longest-chain statistic is not finite")
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 1p1_signature_contains_profile_and_scaled_chain_without_duplicate_fraction() {
        let causet = CausalSet2D::sprinkle_fixed_cardinality(0x51A7, 256, 2.0).unwrap();
        let signature = JointCausalSetSignature::from_causal_set_1p1(&causet, 8).unwrap();

        assert_eq!(signature.event_count, 256);
        assert_eq!(signature.profile_bin_count, 8);
        assert_eq!(signature.normalized_interval_profile.len(), 9);
        assert_eq!(signature.longest_chain_length, causet.longest_chain_length());

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
        assert_eq!(signature.longest_chain_length, causet.longest_chain_length());

        let d3 = signature.features_for_dimension(3).unwrap();
        let d4 = signature.features_for_dimension(4).unwrap();
        assert_eq!(d3.len(), 8);
        assert_eq!(d4.len(), 8);
        assert_eq!(&d3[..7], &d4[..7]);
        assert!(d3[7] < d4[7]);
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
