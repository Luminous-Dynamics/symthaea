// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validated finite probability distributions.
//!
//! Construction is explicit:
//! - new validates a vector that is already normalized within a caller-selected
//!   tolerance and never rescales it.
//! - normalized explicitly normalizes non-negative finite weights.
//!
//! This prevents malformed posterior or Markov state vectors from silently
//! entering downstream entropy, transport, or trajectory calculations.

use std::error::Error;
use std::fmt;

/// Errors encountered while constructing a finite probability distribution.
#[derive(Debug, Clone, PartialEq)]
pub enum DistributionError {
    /// A probability vector must contain at least one state.
    Empty,
    /// The requested normalization tolerance must be finite and lie in [0, 1).
    InvalidTolerance { tolerance: f64 },
    /// A probability or weight is not finite.
    NonFiniteEntry { index: usize, value: f64 },
    /// A probability or weight is negative.
    NegativeEntry { index: usize, value: f64 },
    /// Explicit normalization was requested but total mass is zero.
    ZeroMass,
    /// Probabilities do not sum to one within the requested tolerance.
    NotNormalized { sum: f64, tolerance: f64 },
    /// Normalization arithmetic overflowed or produced a non-finite result.
    NormalizationOverflow,
}

impl fmt::Display for DistributionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "probability distribution is empty"),
            Self::InvalidTolerance { tolerance } => {
                write!(f, "invalid normalization tolerance: {tolerance}")
            }
            Self::NonFiniteEntry { index, value } => {
                write!(f, "probability at index {index} is not finite: {value}")
            }
            Self::NegativeEntry { index, value } => {
                write!(f, "probability at index {index} is negative: {value}")
            }
            Self::ZeroMass => write!(f, "cannot normalize a zero-mass vector"),
            Self::NotNormalized { sum, tolerance } => write!(
                f,
                "probability mass sums to {sum}, outside tolerance {tolerance}"
            ),
            Self::NormalizationOverflow => write!(f, "probability normalization overflowed"),
        }
    }
}

impl Error for DistributionError {}

/// A non-empty finite probability vector whose entries are non-negative and
/// sum to one within the tolerance used at construction.
#[derive(Debug, Clone, PartialEq)]
pub struct FiniteDistribution {
    probabilities: Vec<f64>,
}

impl FiniteDistribution {
    /// Validate an already-normalized probability vector without changing it.
    pub fn new(
        probabilities: Vec<f64>,
        tolerance: f64,
    ) -> Result<Self, DistributionError> {
        validate_tolerance(tolerance)?;
        validate_weights(&probabilities)?;

        let sum = probabilities.iter().sum::<f64>();
        if !sum.is_finite() {
            return Err(DistributionError::NormalizationOverflow);
        }
        if (sum - 1.0).abs() > tolerance {
            return Err(DistributionError::NotNormalized { sum, tolerance });
        }

        Ok(Self { probabilities })
    }

    /// Explicitly normalize non-negative finite weights to unit mass.
    ///
    /// This is intentionally separate from new: callers must opt into
    /// rescaling rather than accidentally hiding invalid probability mass
    /// inside a validation path.
    pub fn normalized(weights: Vec<f64>) -> Result<Self, DistributionError> {
        validate_weights(&weights)?;

        let sum = weights.iter().sum::<f64>();
        if !sum.is_finite() {
            return Err(DistributionError::NormalizationOverflow);
        }
        if sum <= 0.0 {
            return Err(DistributionError::ZeroMass);
        }

        let probabilities = weights
            .into_iter()
            .map(|weight| weight / sum)
            .collect::<Vec<_>>();
        if probabilities.iter().any(|value| !value.is_finite()) {
            return Err(DistributionError::NormalizationOverflow);
        }

        // Explicit normalization can accumulate a few ulps of summation
        // error; validate the result without rescaling it a second time.
        Self::new(probabilities, 1e-12)
    }

    /// Construct a uniform distribution over a positive number of states.
    pub fn uniform(state_count: usize) -> Result<Self, DistributionError> {
        if state_count == 0 {
            return Err(DistributionError::Empty);
        }
        let probability = 1.0 / state_count as f64;
        Self::new(vec![probability; state_count], 1e-12)
    }

    /// Number of states in this distribution.
    pub fn len(&self) -> usize {
        self.probabilities.len()
    }

    /// A valid finite distribution is never empty.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Borrow the probabilities without copying.
    pub fn as_slice(&self) -> &[f64] {
        &self.probabilities
    }

    /// Consume the wrapper and return its validated probability vector.
    pub fn into_vec(self) -> Vec<f64> {
        self.probabilities
    }

    /// Sum of probability mass (near one within construction tolerance).
    pub fn total_mass(&self) -> f64 {
        self.probabilities.iter().sum()
    }
}

fn validate_tolerance(tolerance: f64) -> Result<(), DistributionError> {
    if !tolerance.is_finite() || !(0.0..1.0).contains(&tolerance) {
        return Err(DistributionError::InvalidTolerance { tolerance });
    }
    Ok(())
}

fn validate_weights(weights: &[f64]) -> Result<(), DistributionError> {
    if weights.is_empty() {
        return Err(DistributionError::Empty);
    }
    for (index, &value) in weights.iter().enumerate() {
        if !value.is_finite() {
            return Err(DistributionError::NonFiniteEntry { index, value });
        }
        if value < 0.0 {
            return Err(DistributionError::NegativeEntry { index, value });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_without_silently_renormalizing() {
        let distribution = FiniteDistribution::new(vec![0.25, 0.75], 1e-12).unwrap();
        assert_eq!(distribution.as_slice(), &[0.25, 0.75]);
        assert!(matches!(
            FiniteDistribution::new(vec![0.25, 0.70], 1e-12),
            Err(DistributionError::NotNormalized { .. })
        ));
    }

    #[test]
    fn explicit_normalization_is_a_separate_operation() {
        let distribution = FiniteDistribution::normalized(vec![2.0, 6.0]).unwrap();
        assert!((distribution.as_slice()[0] - 0.25).abs() < 1e-15);
        assert!((distribution.as_slice()[1] - 0.75).abs() < 1e-15);
    }

    #[test]
    fn rejects_empty_negative_non_finite_and_zero_mass() {
        assert_eq!(
            FiniteDistribution::new(Vec::new(), 1e-9),
            Err(DistributionError::Empty)
        );
        assert!(matches!(
            FiniteDistribution::new(vec![1.0, f64::NAN], 1e-9),
            Err(DistributionError::NonFiniteEntry { index: 1, .. })
        ));
        assert!(matches!(
            FiniteDistribution::new(vec![1.1, -0.1], 1e-9),
            Err(DistributionError::NegativeEntry { index: 1, .. })
        ));
        assert_eq!(
            FiniteDistribution::normalized(vec![0.0, 0.0]),
            Err(DistributionError::ZeroMass)
        );
    }

    #[test]
    fn rejects_invalid_tolerances() {
        assert!(matches!(
            FiniteDistribution::new(vec![1.0], f64::NAN),
            Err(DistributionError::InvalidTolerance { .. })
        ));
        assert!(matches!(
            FiniteDistribution::new(vec![1.0], -1.0),
            Err(DistributionError::InvalidTolerance { .. })
        ));
        assert!(matches!(
            FiniteDistribution::new(vec![1.0], 1.0),
            Err(DistributionError::InvalidTolerance { .. })
        ));
    }

    #[test]
    fn uniform_requires_positive_state_count() {
        assert_eq!(
            FiniteDistribution::uniform(0),
            Err(DistributionError::Empty)
        );
        assert_eq!(FiniteDistribution::uniform(4).unwrap().as_slice(), &[0.25; 4]);
    }
}
