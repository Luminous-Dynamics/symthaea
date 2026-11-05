// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Discrete-time Markov chains with validated probability inputs.

use std::error::Error;
use std::fmt;

use crate::distribution::{DistributionError, FiniteDistribution};

/// Default absolute tolerance for normalized probability mass.
pub const DEFAULT_PROBABILITY_TOLERANCE: f64 = 1e-9;

/// Typed failures from finite Markov-chain construction or propagation.
#[derive(Debug, Clone, PartialEq)]
pub enum MarkovError {
    /// A transition matrix must contain at least one state.
    EmptyMatrix,
    /// The probability tolerance must be finite and lie in [0, 1).
    InvalidTolerance { tolerance: f64 },
    /// A transition row has the wrong number of entries.
    NonSquareMatrix {
        row: usize,
        actual: usize,
        expected: usize,
    },
    /// A transition entry is negative or non-finite.
    InvalidTransitionProbability {
        row: usize,
        column: usize,
        value: f64,
    },
    /// A transition row does not sum to one within the requested tolerance.
    RowNotNormalized {
        row: usize,
        sum: f64,
        tolerance: f64,
    },
    /// A supplied state distribution is invalid.
    InvalidDistribution(DistributionError),
    /// A distribution has a different state count from the chain.
    DistributionDimensionMismatch {
        actual: usize,
        expected: usize,
    },
}

impl fmt::Display for MarkovError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "transition matrix is empty"),
            Self::InvalidTolerance { tolerance } => {
                write!(f, "invalid probability tolerance: {tolerance}")
            }
            Self::NonSquareMatrix {
                row,
                actual,
                expected,
            } => write!(
                f,
                "transition row {row} has {actual} entries; expected {expected}"
            ),
            Self::InvalidTransitionProbability {
                row,
                column,
                value,
            } => write!(
                f,
                "transition probability at ({row}, {column}) is invalid: {value}"
            ),
            Self::RowNotNormalized {
                row,
                sum,
                tolerance,
            } => write!(
                f,
                "transition row {row} sums to {sum}; expected one within {tolerance}"
            ),
            Self::InvalidDistribution(error) => write!(f, "invalid state distribution: {error}"),
            Self::DistributionDimensionMismatch { actual, expected } => write!(
                f,
                "state distribution has {actual} entries; chain has {expected} states"
            ),
        }
    }
}

impl Error for MarkovError {}

impl From<DistributionError> for MarkovError {
    fn from(value: DistributionError) -> Self {
        Self::InvalidDistribution(value)
    }
}

/// A discrete-time Markov chain given by a row-stochastic transition matrix:
/// P[i][j] = probability of moving from state i to state j.
#[derive(Debug, Clone)]
pub struct MarkovChain {
    p: Vec<Vec<f64>>,
    n: usize,
}

impl MarkovChain {
    /// Construct from a transition matrix using the default probability
    /// tolerance. For typed errors use try_new.
    ///
    /// This compatibility constructor preserves the historical String error
    /// type while delegating to the stricter checked constructor.
    pub fn new(p: Vec<Vec<f64>>) -> Result<MarkovChain, String> {
        Self::try_new(p).map_err(|error| error.to_string())
    }

    /// Construct from a row-stochastic transition matrix.
    ///
    /// Every entry must be finite and non-negative; every row must sum to one
    /// within the default absolute tolerance. No row is silently normalized.
    pub fn try_new(p: Vec<Vec<f64>>) -> Result<Self, MarkovError> {
        Self::try_new_with_tolerance(p, DEFAULT_PROBABILITY_TOLERANCE)
    }

    /// Construct with an explicit absolute row-sum tolerance.
    pub fn try_new_with_tolerance(
        p: Vec<Vec<f64>>,
        tolerance: f64,
    ) -> Result<Self, MarkovError> {
        if !tolerance.is_finite() || !(0.0..1.0).contains(&tolerance) {
            return Err(MarkovError::InvalidTolerance { tolerance });
        }

        let n = p.len();
        if n == 0 {
            return Err(MarkovError::EmptyMatrix);
        }

        for (row_index, row) in p.iter().enumerate() {
            if row.len() != n {
                return Err(MarkovError::NonSquareMatrix {
                    row: row_index,
                    actual: row.len(),
                    expected: n,
                });
            }

            for (column, &value) in row.iter().enumerate() {
                if !value.is_finite() || value < 0.0 {
                    return Err(MarkovError::InvalidTransitionProbability {
                        row: row_index,
                        column,
                        value,
                    });
                }
            }

            let sum = row.iter().sum::<f64>();
            if !sum.is_finite() || (sum - 1.0).abs() > tolerance {
                return Err(MarkovError::RowNotNormalized {
                    row: row_index,
                    sum,
                    tolerance,
                });
            }
        }

        Ok(Self { p, n })
    }

    /// Number of states.
    pub fn size(&self) -> usize {
        self.n
    }

    /// Borrow the transition matrix.
    pub fn transition(&self) -> &[Vec<f64>] {
        &self.p
    }

    /// Advance a validated distribution one step: next = dist multiplied by P.
    pub fn step_distribution(
        &self,
        distribution: &FiniteDistribution,
    ) -> Result<FiniteDistribution, MarkovError> {
        if distribution.len() != self.n {
            return Err(MarkovError::DistributionDimensionMismatch {
                actual: distribution.len(),
                expected: self.n,
            });
        }

        let mut next = vec![0.0; self.n];
        for (i, &mass) in distribution.as_slice().iter().enumerate() {
            if mass == 0.0 {
                continue;
            }
            for (j, next_mass) in next.iter_mut().enumerate() {
                *next_mass += mass * self.p[i][j];
            }
        }

        // Validate the result without silently renormalizing it. If transition
        // row mass error accumulates, return an explicit error rather than
        // silently changing the propagated state.
        Ok(FiniteDistribution::new(
            next,
            DEFAULT_PROBABILITY_TOLERANCE,
        )?)
    }

    /// Fallible one-step propagation from a raw slice.
    pub fn try_step(&self, distribution: &[f64]) -> Result<Vec<f64>, MarkovError> {
        let distribution = FiniteDistribution::new(
            distribution.to_vec(),
            DEFAULT_PROBABILITY_TOLERANCE,
        )?;
        Ok(self.step_distribution(&distribution)?.into_vec())
    }

    /// Compatibility wrapper for one-step propagation.
    ///
    /// Prefer try_step to handle malformed distributions without a panic.
    #[deprecated(note = "use try_step for typed validation errors")]
    pub fn step(&self, distribution: &[f64]) -> Vec<f64> {
        self.try_step(distribution)
            .unwrap_or_else(|error| panic!("invalid Markov state distribution: {error}"))
    }

    /// Fallible distribution after a declared number of transitions.
    pub fn try_distribution_after(
        &self,
        initial: &[f64],
        steps: usize,
    ) -> Result<FiniteDistribution, MarkovError> {
        let mut distribution = FiniteDistribution::new(
            initial.to_vec(),
            DEFAULT_PROBABILITY_TOLERANCE,
        )?;
        if distribution.len() != self.n {
            return Err(MarkovError::DistributionDimensionMismatch {
                actual: distribution.len(),
                expected: self.n,
            });
        }

        for _ in 0..steps {
            distribution = self.step_distribution(&distribution)?;
        }
        Ok(distribution)
    }

    /// Compatibility wrapper for n-step propagation.
    ///
    /// Prefer try_distribution_after to handle malformed distributions.
    #[deprecated(note = "use try_distribution_after for typed validation errors")]
    pub fn distribution_after(&self, initial: &[f64], steps: usize) -> Vec<f64> {
        self.try_distribution_after(initial, steps)
            .unwrap_or_else(|error| panic!("invalid Markov distribution propagation: {error}"))
            .into_vec()
    }

    /// Fallible stationary distribution by power iteration from a uniform
    /// initial state. Convergence is guaranteed only under suitable chain
    /// conditions; the returned vector is a finite-iteration approximation.
    pub fn try_stationary_distribution(
        &self,
        iterations: usize,
    ) -> Result<FiniteDistribution, MarkovError> {
        let mut distribution = FiniteDistribution::uniform(self.n)?;
        for _ in 0..iterations {
            let next = self.step_distribution(&distribution)?;
            let delta = next
                .as_slice()
                .iter()
                .zip(distribution.as_slice())
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>();
            distribution = next;
            if delta < 1e-15 {
                break;
            }
        }
        Ok(distribution)
    }

    /// Compatibility stationary-distribution API.
    ///
    /// Prefer try_stationary_distribution when the caller needs a typed error.
    pub fn stationary_distribution(&self, iterations: usize) -> Vec<f64> {
        self.try_stationary_distribution(iterations)
            .expect("a validated transition matrix preserves a valid uniform distribution")
            .into_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_matrix() {
        assert!(MarkovChain::new(vec![vec![0.5, 0.4]]).is_err());
        assert!(MarkovChain::new(vec![vec![1.0, 0.0], vec![0.3, 0.3]]).is_err());
    }

    #[test]
    fn rejects_non_finite_and_negative_transition_entries() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.01] {
            assert!(matches!(
                MarkovChain::try_new(vec![vec![1.0, 0.0], vec![invalid, 1.0 - invalid]]),
                Err(MarkovError::InvalidTransitionProbability { .. })
            ));
        }
    }

    #[test]
    fn rejects_bad_dimensions_and_invalid_tolerance() {
        assert!(matches!(
            MarkovChain::try_new(vec![vec![1.0, 0.0], vec![1.0]]),
            Err(MarkovError::NonSquareMatrix { .. })
        ));
        assert!(matches!(
            MarkovChain::try_new_with_tolerance(vec![vec![1.0]], f64::NAN),
            Err(MarkovError::InvalidTolerance { .. })
        ));
        assert!(matches!(
            MarkovChain::try_new_with_tolerance(vec![vec![1.0]], 1.0),
            Err(MarkovError::InvalidTolerance { .. })
        ));
    }

    #[test]
    fn two_state_stationary() {
        let chain = MarkovChain::try_new(vec![vec![0.9, 0.1], vec![0.5, 0.5]]).unwrap();
        let stationary = chain.try_stationary_distribution(1000).unwrap();
        assert!((stationary.as_slice()[0] - 5.0 / 6.0).abs() < 1e-9);
        assert!((stationary.as_slice()[1] - 1.0 / 6.0).abs() < 1e-9);
        assert!((stationary.total_mass() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn distribution_converges_to_stationary() {
        let chain = MarkovChain::try_new(vec![vec![0.9, 0.1], vec![0.5, 0.5]]).unwrap();
        let distribution = chain.try_distribution_after(&[1.0, 0.0], 200).unwrap();
        assert!((distribution.as_slice()[0] - 5.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn propagation_rejects_wrong_length_non_normalized_and_invalid_inputs() {
        let chain = MarkovChain::try_new(vec![vec![0.9, 0.1], vec![0.5, 0.5]]).unwrap();

        assert!(matches!(
            chain.try_step(&[1.0]),
            Err(MarkovError::DistributionDimensionMismatch {
                actual: 1,
                expected: 2
            })
        ));
        assert!(matches!(
            chain.try_step(&[0.0, 0.0]),
            Err(MarkovError::InvalidDistribution(
                DistributionError::NotNormalized { .. }
            ))
        ));
        assert!(matches!(
            chain.try_step(&[1.0, f64::NAN]),
            Err(MarkovError::InvalidDistribution(
                DistributionError::NonFiniteEntry { .. }
            ))
        ));
        assert!(matches!(
            chain.try_step(&[1.1, -0.1]),
            Err(MarkovError::InvalidDistribution(
                DistributionError::NegativeEntry { .. }
            ))
        ));
        assert!(matches!(
            chain.try_distribution_after(&[1.0], 0),
            Err(MarkovError::DistributionDimensionMismatch { .. })
        ));
    }
}
