// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Normalized interval-abundance profiles and statistical profile distances.
//!
//! This module compares finite causal-set interval histograms without claiming
//! that a generic distribution distance is a Lorentzian-geometry distance.
//! The profile preserves both (a) the distribution of open-interval
//! cardinalities conditional on comparable endpoint pairs and (b) the
//! fraction of incomparable endpoint pairs.

use crate::causal_set_nd::CausalSetNd;
use crate::{CausalSet2D, CausalSetError};
use std::error::Error;
use std::fmt;

/// Errors while validating or comparing interval-abundance profiles.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IntervalProfileError {
    PairCountOverflow {
        event_count: usize,
    },
    HistogramLength {
        expected: usize,
        actual: usize,
    },
    HistogramCountOverflow,
    HistogramExceedsPairCount {
        related_pairs: u64,
        total_pairs: u64,
    },
    NoPairSamples,
    ZeroProfileBins,
    IncompatibleBinCount {
        left: usize,
        right: usize,
    },
}

impl fmt::Display for IntervalProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PairCountOverflow { event_count } => {
                write!(f, "unordered pair count overflows for {event_count} events")
            }
            Self::HistogramLength { expected, actual } => write!(
                f,
                "interval histogram has {actual} bins; expected {expected}"
            ),
            Self::HistogramCountOverflow => write!(f, "interval histogram count sum overflowed"),
            Self::HistogramExceedsPairCount {
                related_pairs,
                total_pairs,
            } => write!(
                f,
                "histogram contains {related_pairs} related pairs; only {total_pairs} pairs exist"
            ),
            Self::NoPairSamples => {
                write!(f, "at least two events are required for a normalized pair profile")
            }
            Self::ZeroProfileBins => write!(f, "normalized profile requires at least one bin"),
            Self::IncompatibleBinCount { left, right } => write!(
                f,
                "profile bin count mismatch: left has {left}, right has {right}"
            ),
        }
    }
}

impl Error for IntervalProfileError {}

/// Exact histogram plus normalized information for one finite causal set.
///
/// `histogram[k]` counts unordered comparable endpoint pairs whose open order
/// interval contains exactly k events. `incomparable_pairs` counts all
/// unordered endpoint pairs that are not causally related.
///
/// The profile intentionally stores raw counts as well as derived
/// probabilities. Comparisons across event densities must report the
/// normalization convention; this object does not turn a finite sprinkling
/// into a continuum geometry by itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntervalAbundanceProfile {
    event_count: usize,
    total_pairs: u64,
    related_pairs: u64,
    incomparable_pairs: u64,
    histogram: Vec<u64>,
}

impl IntervalAbundanceProfile {
    /// Construct a profile from a validated open-interval histogram.
    pub fn from_histogram(
        event_count: usize,
        histogram: Vec<u64>,
    ) -> Result<Self, IntervalProfileError> {
        let expected = event_count.saturating_sub(1);
        if histogram.len() != expected {
            return Err(IntervalProfileError::HistogramLength {
                expected,
                actual: histogram.len(),
            });
        }

        let total_pairs = if event_count < 2 {
            0
        } else {
            let product = event_count.checked_mul(event_count - 1).ok_or(
                IntervalProfileError::PairCountOverflow { event_count },
            )?;
            u64::try_from(product / 2)
                .map_err(|_| IntervalProfileError::PairCountOverflow { event_count })?
        };

        let related_pairs = histogram.iter().try_fold(0_u64, |sum, count| {
            sum.checked_add(*count)
                .ok_or(IntervalProfileError::HistogramCountOverflow)
        })?;
        if related_pairs > total_pairs {
            return Err(IntervalProfileError::HistogramExceedsPairCount {
                related_pairs,
                total_pairs,
            });
        }

        Ok(Self {
            event_count,
            total_pairs,
            related_pairs,
            incomparable_pairs: total_pairs - related_pairs,
            histogram,
        })
    }

    /// Build a profile from the crate's exact 1+1D causal-set implementation.
    pub fn from_causal_set_1p1(
        causet: &CausalSet2D,
    ) -> Result<Self, IntervalProfileError> {
        Self::from_histogram(causet.len(), causet.interval_size_histogram())
    }

    /// Build a profile from the dimension-generic 2+1D or 3+1D implementation.
    pub fn from_causal_set_nd(
        causet: &CausalSetNd,
    ) -> Result<Self, IntervalProfileError> {
        Self::from_histogram(causet.len(), causet.interval_size_histogram())
    }

    pub fn event_count(&self) -> usize {
        self.event_count
    }

    pub fn total_pairs(&self) -> u64 {
        self.total_pairs
    }

    pub fn related_pairs(&self) -> u64 {
        self.related_pairs
    }

    pub fn incomparable_pairs(&self) -> u64 {
        self.incomparable_pairs
    }

    pub fn related_pair_fraction(&self) -> Option<f64> {
        (self.total_pairs > 0)
            .then_some(self.related_pairs as f64 / self.total_pairs as f64)
    }

    pub fn histogram(&self) -> &[u64] {
        &self.histogram
    }

    /// Produce a fixed-width probability profile for comparing event sets of
    /// different cardinalities.
    ///
    /// Interior cardinality k is mapped to normalized size k/(N-2), with
    /// linear interpolation between adjacent bins. The final element is a
    /// dedicated incomparable-pair mass. The output sums to one up to
    /// floating-point rounding.
    ///
    /// This is a declared finite-size normalization, not a universal
    /// continuum scaling law. Comparisons should also retain the raw
    /// histogram, event density and total spacetime-volume convention.
    pub fn normalized_bins(
        &self,
        profile_bin_count: usize,
    ) -> Result<Vec<f64>, IntervalProfileError> {
        if profile_bin_count == 0 {
            return Err(IntervalProfileError::ZeroProfileBins);
        }
        if self.total_pairs == 0 {
            return Err(IntervalProfileError::NoPairSamples);
        }

        let mut probabilities = vec![0.0; profile_bin_count + 1];
        let denominator = self.total_pairs as f64;
        let maximum_interior = self.event_count.saturating_sub(2);

        for (interior_count, pair_count) in self.histogram.iter().copied().enumerate() {
            if pair_count == 0 {
                continue;
            }
            let normalized_size = if maximum_interior == 0 {
                0.0
            } else {
                interior_count as f64 / maximum_interior as f64
            };
            let position = normalized_size * (profile_bin_count - 1) as f64;
            let lower = position.floor() as usize;
            let upper = position.ceil() as usize;
            let mass = pair_count as f64 / denominator;
            if lower == upper {
                probabilities[lower] += mass;
            } else {
                let upper_weight = position - lower as f64;
                probabilities[lower] += mass * (1.0 - upper_weight);
                probabilities[upper] += mass * upper_weight;
            }
        }

        probabilities[profile_bin_count] =
            self.incomparable_pairs as f64 / denominator;
        Ok(probabilities)
    }

    /// Total-variation distance between fixed-width normalized profiles.
    ///
    /// Range: [0, 1]. This is a statistical distribution distance, not a
    /// Lorentzian Gromov–Hausdorff distance or a physical closeness theorem.
    pub fn total_variation_distance(
        &self,
        other: &Self,
        profile_bin_count: usize,
    ) -> Result<f64, IntervalProfileError> {
        let left = self.normalized_bins(profile_bin_count)?;
        let right = other.normalized_bins(profile_bin_count)?;
        Ok(0.5
            * left
                .iter()
                .zip(right.iter())
                .map(|(p, q)| (p - q).abs())
                .sum::<f64>())
    }

    /// Hellinger distance between fixed-width normalized profiles.
    ///
    /// Range: [0, 1] for discrete probability vectors with this convention.
    pub fn hellinger_distance(
        &self,
        other: &Self,
        profile_bin_count: usize,
    ) -> Result<f64, IntervalProfileError> {
        let left = self.normalized_bins(profile_bin_count)?;
        let right = other.normalized_bins(profile_bin_count)?;
        let squared = left
            .iter()
            .zip(right.iter())
            .map(|(p, q)| (p.sqrt() - q.sqrt()).powi(2))
            .sum::<f64>();
        Ok((0.5 * squared).sqrt())
    }

    /// Jensen–Shannon distance (square root of base-2 JS divergence).
    ///
    /// Range: [0, 1]. This generic statistical distance is not the published
    /// Surya Lorentzian closeness function; it is a transparent baseline.
    pub fn jensen_shannon_distance(
        &self,
        other: &Self,
        profile_bin_count: usize,
    ) -> Result<f64, IntervalProfileError> {
        let left = self.normalized_bins(profile_bin_count)?;
        let right = other.normalized_bins(profile_bin_count)?;
        let divergence = left
            .iter()
            .zip(right.iter())
            .map(|(p, q)| {
                let midpoint = 0.5 * (p + q);
                let left_term = if *p > 0.0 {
                    0.5 * p * (p / midpoint).ln() / std::f64::consts::LN_2
                } else {
                    0.0
                };
                let right_term = if *q > 0.0 {
                    0.5 * q * (q / midpoint).ln() / std::f64::consts::LN_2
                } else {
                    0.0
                };
                left_term + right_term
            })
            .sum::<f64>();
        Ok(divergence.max(0.0).sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal_set_nd::{CausalSetNd, EventNd};

    #[test]
    fn total_order_profile_preserves_interval_cardinalities() {
        let profile = IntervalAbundanceProfile::from_histogram(4, vec![3, 2, 1]).unwrap();
        assert_eq!(profile.total_pairs(), 6);
        assert_eq!(profile.related_pairs(), 6);
        assert_eq!(profile.incomparable_pairs(), 0);
        assert_eq!(profile.related_pair_fraction(), Some(1.0));
        assert_eq!(profile.histogram(), &[3, 2, 1]);
    }

    #[test]
    fn antichain_profile_preserves_incomparable_mass() {
        let profile = IntervalAbundanceProfile::from_histogram(4, vec![0, 0, 0]).unwrap();
        assert_eq!(profile.related_pairs(), 0);
        assert_eq!(profile.incomparable_pairs(), 6);
        assert_eq!(profile.related_pair_fraction(), Some(0.0));
        let bins = profile.normalized_bins(8).unwrap();
        assert_eq!(bins[8], 1.0);
        assert!(bins[..8].iter().all(|mass| *mass == 0.0));
    }

    #[test]
    fn normalized_profile_sums_to_one() {
        let profile = IntervalAbundanceProfile::from_histogram(5, vec![4, 3, 2, 1]).unwrap();
        let bins = profile.normalized_bins(16).unwrap();
        assert!((bins.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn identical_profiles_have_zero_statistical_distance() {
        let left = IntervalAbundanceProfile::from_histogram(5, vec![4, 3, 2, 1]).unwrap();
        let right = IntervalAbundanceProfile::from_histogram(5, vec![4, 3, 2, 1]).unwrap();
        assert_eq!(left.total_variation_distance(&right, 16).unwrap(), 0.0);
        assert_eq!(left.hellinger_distance(&right, 16).unwrap(), 0.0);
        assert_eq!(left.jensen_shannon_distance(&right, 16).unwrap(), 0.0);
    }

    #[test]
    fn distances_are_symmetric_and_bounded() {
        let chain = IntervalAbundanceProfile::from_histogram(5, vec![4, 3, 2, 1]).unwrap();
        let antichain = IntervalAbundanceProfile::from_histogram(5, vec![0, 0, 0, 0]).unwrap();

        let tv = chain.total_variation_distance(&antichain, 32).unwrap();
        let hellinger = chain.hellinger_distance(&antichain, 32).unwrap();
        let js = chain.jensen_shannon_distance(&antichain, 32).unwrap();

        assert!((0.0..=1.0).contains(&tv));
        assert!((0.0..=1.0).contains(&hellinger));
        assert!((0.0..=1.0).contains(&js));
        assert_eq!(tv, antichain.total_variation_distance(&chain, 32).unwrap());
        assert_eq!(hellinger, antichain.hellinger_distance(&chain, 32).unwrap());
        assert_eq!(js, antichain.jensen_shannon_distance(&chain, 32).unwrap());
    }

    #[test]
    fn zero_or_one_event_causets_have_no_normalized_pair_distribution() {
        let empty = IntervalAbundanceProfile::from_histogram(0, vec![]).unwrap();
        let singleton = IntervalAbundanceProfile::from_histogram(1, vec![]).unwrap();
        assert_eq!(empty.total_pairs(), 0);
        assert_eq!(
            singleton.normalized_bins(8),
            Err(IntervalProfileError::NoPairSamples)
        );
    }

    #[test]
    fn malformed_histograms_and_overflow_fail_closed() {
        assert_eq!(
            IntervalAbundanceProfile::from_histogram(4, vec![1, 2]),
            Err(IntervalProfileError::HistogramLength {
                expected: 3,
                actual: 2
            })
        );
        assert!(matches!(
            IntervalAbundanceProfile::from_histogram(3, vec![u64::MAX, 1]),
            Err(IntervalProfileError::HistogramCountOverflow)
        ));
        assert!(matches!(
            IntervalAbundanceProfile::from_histogram(3, vec![4, 0]),
            Err(IntervalProfileError::HistogramExceedsPairCount { .. })
        ));
        assert_eq!(
            IntervalAbundanceProfile::from_histogram(3, vec![1, 0])
                .unwrap()
                .normalized_bins(0),
            Err(IntervalProfileError::ZeroProfileBins)
        );
    }

    #[test]
    fn profile_builders_match_under_same_event_set() {
        let events = (0..4)
            .map(|index| EventNd {
                id: index,
                coordinates: vec![index as f64, 0.0],
            })
            .collect();
        let nd = CausalSetNd::new(2, events).unwrap();
        let nd_profile = IntervalAbundanceProfile::from_causal_set_nd(&nd).unwrap();
        let profile = IntervalAbundanceProfile::from_histogram(4, vec![3, 2, 1]).unwrap();
        assert_eq!(nd_profile, profile);
    }

    #[test]
    fn one_plus_one_builder_uses_exact_causal_set_histogram() {
        let causet = CausalSet2D::new(
            (0..4)
                .map(|index| crate::Event2D {
                    id: index,
                    u: index as f64,
                    v: index as f64,
                })
                .collect(),
            4.0,
            4.0,
        )
        .unwrap();
        let profile = IntervalAbundanceProfile::from_causal_set_1p1(&causet).unwrap();
        assert_eq!(profile.histogram(), &[3, 2, 1]);
        assert_eq!(profile.related_pairs(), 6);
    }
}
