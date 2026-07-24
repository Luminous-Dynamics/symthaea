// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Analytical minimum-distance reliability under independent symbol errata.
//!
//! Each transmitted symbol independently becomes either a clean symbol, an
//! unknown error, or a known erasure. The three outcomes are mutually
//! exclusive. Dynamic programming computes the exact floating representation
//! of the errata-weight distribution, where unknown errors cost two units and
//! known erasures cost one. A block code is guaranteed to recover whenever
//! `2e + s < d_min`.

use std::fmt;

use crate::{
    channel::Probability,
    parameters::BlockCodeParameters,
};

/// Invalid analytical model or block-code dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReliabilityError {
    ErrataProbabilityExceedsOne {
        error_numerator: u64,
        error_denominator: u64,
        erasure_numerator: u64,
        erasure_denominator: u64,
    },
    ZeroCodewordSymbols,
    ZeroMinimumDistance,
    MinimumDistanceExceedsSingletonBound {
        minimum_distance: usize,
        codeword_symbols: usize,
    },
    DistributionSizeOverflow { codeword_symbols: usize },
}

impl fmt::Display for ReliabilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ErrataProbabilityExceedsOne {
                error_numerator,
                error_denominator,
                erasure_numerator,
                erasure_denominator,
            } => write!(
                f,
                "error probability {error_numerator}/{error_denominator} plus erasure probability {erasure_numerator}/{erasure_denominator} exceeds one"
            ),
            Self::ZeroCodewordSymbols => write!(f, "reliability analysis requires a non-empty codeword"),
            Self::ZeroMinimumDistance => write!(f, "minimum distance must be non-zero"),
            Self::MinimumDistanceExceedsSingletonBound {
                minimum_distance,
                codeword_symbols,
            } => write!(
                f,
                "minimum distance {minimum_distance} exceeds n + 1 = {} for codeword length {codeword_symbols}",
                codeword_symbols + 1
            ),
            Self::DistributionSizeOverflow { codeword_symbols } => write!(
                f,
                "errata-weight distribution for {codeword_symbols} symbols overflows usize"
            ),
        }
    }
}

impl std::error::Error for ReliabilityError {}

/// Exact rational probabilities for mutually exclusive unknown errors and erasures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndependentErrataModel {
    unknown_error_probability: Probability,
    known_erasure_probability: Probability,
}

impl IndependentErrataModel {
    pub fn new(
        unknown_error_probability: Probability,
        known_erasure_probability: Probability,
    ) -> Result<Self, ReliabilityError> {
        let error_scaled = u128::from(unknown_error_probability.numerator())
            * u128::from(known_erasure_probability.denominator());
        let erasure_scaled = u128::from(known_erasure_probability.numerator())
            * u128::from(unknown_error_probability.denominator());
        let common_denominator = u128::from(unknown_error_probability.denominator())
            * u128::from(known_erasure_probability.denominator());
        if error_scaled > common_denominator.saturating_sub(erasure_scaled) {
            return Err(ReliabilityError::ErrataProbabilityExceedsOne {
                error_numerator: unknown_error_probability.numerator(),
                error_denominator: unknown_error_probability.denominator(),
                erasure_numerator: known_erasure_probability.numerator(),
                erasure_denominator: known_erasure_probability.denominator(),
            });
        }
        Ok(Self {
            unknown_error_probability,
            known_erasure_probability,
        })
    }

    #[must_use]
    pub const fn unknown_error_probability(self) -> Probability {
        self.unknown_error_probability
    }

    #[must_use]
    pub const fn known_erasure_probability(self) -> Probability {
        self.known_erasure_probability
    }

    #[must_use]
    pub fn clean_probability(self) -> f64 {
        1.0
            - self.unknown_error_probability.as_f64()
            - self.known_erasure_probability.as_f64()
    }

    /// Stable input-only description suitable for preregistration.
    #[must_use]
    pub fn identifier(self) -> String {
        format!(
            "independent-errata-perr{}-{}-perasure{}-{}",
            self.unknown_error_probability.numerator(),
            self.unknown_error_probability.denominator(),
            self.known_erasure_probability.numerator(),
            self.known_erasure_probability.denominator(),
        )
    }
}

/// Probability mass indexed by minimum-distance errata weight `2e + s`.
#[derive(Debug, Clone, PartialEq)]
pub struct ErrataWeightDistribution {
    codeword_symbols: usize,
    model: IndependentErrataModel,
    probability_by_weight: Vec<f64>,
}

impl ErrataWeightDistribution {
    /// Compute the independent categorical distribution in `O(n²)` time.
    pub fn new(
        codeword_symbols: usize,
        model: IndependentErrataModel,
    ) -> Result<Self, ReliabilityError> {
        if codeword_symbols == 0 {
            return Err(ReliabilityError::ZeroCodewordSymbols);
        }
        let maximum_weight = codeword_symbols
            .checked_mul(2)
            .ok_or(ReliabilityError::DistributionSizeOverflow { codeword_symbols })?;
        let distribution_len = maximum_weight
            .checked_add(1)
            .ok_or(ReliabilityError::DistributionSizeOverflow { codeword_symbols })?;
        let mut probability_by_weight = vec![0.0; distribution_len];
        probability_by_weight[0] = 1.0;

        let clean = model.clean_probability();
        let error = model.unknown_error_probability().as_f64();
        let erasure = model.known_erasure_probability().as_f64();
        let mut active_maximum = 0usize;

        for _ in 0..codeword_symbols {
            let mut next = vec![0.0; distribution_len];
            for weight in 0..=active_maximum {
                let mass = probability_by_weight[weight];
                next[weight] += mass * clean;
                next[weight + 1] += mass * erasure;
                next[weight + 2] += mass * error;
            }
            probability_by_weight = next;
            active_maximum += 2;
        }

        Ok(Self {
            codeword_symbols,
            model,
            probability_by_weight,
        })
    }

    #[must_use]
    pub const fn codeword_symbols(&self) -> usize {
        self.codeword_symbols
    }

    #[must_use]
    pub const fn model(&self) -> IndependentErrataModel {
        self.model
    }

    #[must_use]
    pub fn maximum_weight(&self) -> usize {
        self.probability_by_weight.len() - 1
    }

    #[must_use]
    pub fn probability_at(&self, weight: usize) -> f64 {
        self.probability_by_weight
            .get(weight)
            .copied()
            .unwrap_or(0.0)
    }

    #[must_use]
    pub fn probability_within(&self, maximum_weight: usize) -> f64 {
        self.probability_by_weight
            .iter()
            .take(maximum_weight.saturating_add(1))
            .sum::<f64>()
            .clamp(0.0, 1.0)
    }

    #[must_use]
    pub fn total_probability(&self) -> f64 {
        self.probability_by_weight.iter().sum()
    }

    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.probability_by_weight
    }
}

/// Guaranteed-recovery probability implied by minimum distance.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockCodeReliabilityEstimate {
    pub parameters: BlockCodeParameters,
    pub model: IndependentErrataModel,
    pub guaranteed_recovery_probability: f64,
    pub outside_guarantee_probability: f64,
    pub expected_unknown_errors: f64,
    pub expected_known_erasures: f64,
    pub expected_errata_weight: f64,
}

impl BlockCodeReliabilityEstimate {
    /// Stable input manifest; computed floating results are intentionally omitted.
    #[must_use]
    pub fn manifest(&self) -> String {
        format!(
            "symthaea-coding-reliability-v1;family={:?};n={};d={};model={}",
            self.parameters.family,
            self.parameters.codeword_symbols,
            self.parameters.minimum_distance,
            self.model.identifier(),
        )
    }
}

/// Analyze the algebraic guarantee `2e + s < d_min` for fixed parameters.
pub fn estimate_block_code_reliability(
    parameters: BlockCodeParameters,
    model: IndependentErrataModel,
) -> Result<BlockCodeReliabilityEstimate, ReliabilityError> {
    validate_parameters(parameters.codeword_symbols, parameters.minimum_distance)?;
    let distribution = ErrataWeightDistribution::new(parameters.codeword_symbols, model)?;
    let guaranteed_recovery_probability =
        distribution.probability_within(parameters.minimum_distance - 1);
    let outside_guarantee_probability = (1.0 - guaranteed_recovery_probability).clamp(0.0, 1.0);
    let expected_unknown_errors = parameters.codeword_symbols as f64
        * model.unknown_error_probability().as_f64();
    let expected_known_erasures = parameters.codeword_symbols as f64
        * model.known_erasure_probability().as_f64();

    Ok(BlockCodeReliabilityEstimate {
        parameters,
        model,
        guaranteed_recovery_probability,
        outside_guarantee_probability,
        expected_unknown_errors,
        expected_known_erasures,
        expected_errata_weight: expected_unknown_errors * 2.0 + expected_known_erasures,
    })
}

fn validate_parameters(
    codeword_symbols: usize,
    minimum_distance: usize,
) -> Result<(), ReliabilityError> {
    if codeword_symbols == 0 {
        return Err(ReliabilityError::ZeroCodewordSymbols);
    }
    if minimum_distance == 0 {
        return Err(ReliabilityError::ZeroMinimumDistance);
    }
    if minimum_distance > codeword_symbols.saturating_add(1) {
        return Err(ReliabilityError::MinimumDistanceExceedsSingletonBound {
            minimum_distance,
            codeword_symbols,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        parameters::{HAMMING74_PARAMETERS, HAMMING84_PARAMETERS},
        reed_solomon::{ReedSolomon, ReedSolomonConfig},
    };

    fn probability(numerator: u64, denominator: u64) -> Probability {
        Probability::new(numerator, denominator).unwrap()
    }

    #[test]
    fn exact_rational_model_rejects_overfull_categorical_mass() {
        assert!(IndependentErrataModel::new(probability(1, 3), probability(2, 3)).is_ok());
        assert_eq!(
            IndependentErrataModel::new(probability(1, 2), probability(2, 3)),
            Err(ReliabilityError::ErrataProbabilityExceedsOne {
                error_numerator: 1,
                error_denominator: 2,
                erasure_numerator: 2,
                erasure_denominator: 3,
            })
        );
    }

    #[test]
    fn errata_weight_distribution_preserves_probability_mass() {
        let model = IndependentErrataModel::new(probability(1, 20), probability(1, 10)).unwrap();
        for symbols in [1usize, 2, 7, 31, 255] {
            let distribution = ErrataWeightDistribution::new(symbols, model).unwrap();
            assert!((distribution.total_probability() - 1.0).abs() < 1.0e-12);
        }
    }

    #[test]
    fn unknown_error_only_case_matches_binomial_closed_form() {
        let model = IndependentErrataModel::new(probability(1, 10), probability(0, 1)).unwrap();
        let estimate = estimate_block_code_reliability(HAMMING74_PARAMETERS, model).unwrap();
        let p = 0.1f64;
        let expected = (1.0 - p).powi(7) + 7.0 * p * (1.0 - p).powi(6);
        assert!((estimate.guaranteed_recovery_probability - expected).abs() < 1.0e-15);
    }

    #[test]
    fn stronger_distance_never_reduces_the_guaranteed_probability() {
        let model = IndependentErrataModel::new(probability(1, 50), probability(1, 100)).unwrap();
        let h74 = estimate_block_code_reliability(HAMMING74_PARAMETERS, model).unwrap();
        let h84 = estimate_block_code_reliability(HAMMING84_PARAMETERS, model).unwrap();
        assert!(h84.guaranteed_recovery_probability >= h74.guaranteed_recovery_probability);
    }

    #[test]
    fn reed_solomon_parameters_feed_the_same_minimum_distance_bound() {
        let codec = ReedSolomon::new(ReedSolomonConfig::aes(16)).unwrap();
        let parameters = codec.parameters(64).unwrap();
        let model = IndependentErrataModel::new(probability(1, 100), probability(1, 200)).unwrap();
        let estimate = estimate_block_code_reliability(parameters, model).unwrap();
        assert_eq!(parameters.minimum_distance, 17);
        assert!(estimate.guaranteed_recovery_probability > 0.99);
        assert_eq!(
            estimate.manifest(),
            "symthaea-coding-reliability-v1;family=ReedSolomon;n=80;d=17;model=independent-errata-perr1-100-perasure1-200"
        );
    }

    #[test]
    fn all_clean_and_all_error_extremes_are_exact() {
        let clean = IndependentErrataModel::new(probability(0, 1), probability(0, 1)).unwrap();
        let all_error = IndependentErrataModel::new(probability(1, 1), probability(0, 1)).unwrap();
        assert_eq!(
            estimate_block_code_reliability(HAMMING84_PARAMETERS, clean)
                .unwrap()
                .guaranteed_recovery_probability,
            1.0
        );
        assert_eq!(
            estimate_block_code_reliability(HAMMING84_PARAMETERS, all_error)
                .unwrap()
                .guaranteed_recovery_probability,
            0.0
        );
    }
}
