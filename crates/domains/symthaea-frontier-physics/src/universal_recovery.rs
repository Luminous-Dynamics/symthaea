// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

#![allow(clippy::needless_range_loop)]
// Fixed 2x8 and 8x8 contractions use explicit index loops.

//! An exact error-set recovery baseline for the three-qubit repetition code.
//!
//! The code space is span{|000>, |111>} and the declared error set is
//! {I, X_0, X_1, X_2}. A single syndrome recovery corrects every stochastic
//! channel in this family, without learning its probability vector.
//!
//! This is an exact Knill-Laflamme control and an executable universal-decoder
//! fixture. It is not yet the general approximate error-set Petz construction
//! or the full theorem/metric implementation from arXiv:2608.28962.

const PHYSICAL_DIMENSION: usize = 8;
const LOGICAL_DIMENSION: usize = 2;
const ERROR_COUNT: usize = 4;
const CODEWORDS: [usize; LOGICAL_DIMENSION] = [0, 7];
const ERROR_MASKS: [usize; ERROR_COUNT] = [0, 4, 2, 1];

/// Failure while constructing or validating an error-set recovery fixture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UniversalRecoveryError {
    InvalidTolerance(f64),
    NonFiniteProbability { index: usize, value: f64 },
    NegativeProbability { index: usize, value: f64 },
    ProbabilitySum { sum: f64, tolerance: f64 },
}

impl core::fmt::Display for UniversalRecoveryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidTolerance(value) => {
                write!(f, "tolerance must be positive and finite, got {value}")
            }
            Self::NonFiniteProbability { index, value } => {
                write!(f, "probability {index} is non-finite: {value}")
            }
            Self::NegativeProbability { index, value } => {
                write!(f, "probability {index} is negative: {value}")
            }
            Self::ProbabilitySum { sum, tolerance } => write!(
                f,
                "error probabilities sum to {sum}, outside unit normalization tolerance {tolerance}"
            ),
        }
    }
}

impl std::error::Error for UniversalRecoveryError {}

/// Four syndrome Kraus operators, each mapping the physical 8D space to the
/// logical 2D code space. Index order is [syndrome][logical output][physical input].
pub type ThreeQubitBitFlipRecovery = [[[f64; PHYSICAL_DIMENSION]; LOGICAL_DIMENSION]; ERROR_COUNT];

/// Diagnostics for one stochastic channel supported on the fixed error set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UniversalBitFlipRecoverySample {
    pub probabilities: [f64; ERROR_COUNT],
    pub error_set_size: usize,
    pub code_dimension: usize,
    pub qec_matrix_knill_laflamme_residual: f64,
    pub channel_trace_preservation_residual: f64,
    pub recovery_trace_preservation_residual: f64,
    pub logical_recovery_residual: f64,
    pub entanglement_fidelity: f64,
}

/// The basis permutation for each member of {I, X_0, X_1, X_2}.
///
/// Qubit 0 is the most-significant bit and qubit 2 is the least-significant bit.
fn error_maps() -> [[usize; PHYSICAL_DIMENSION]; ERROR_COUNT] {
    core::array::from_fn(|error| core::array::from_fn(|basis| basis ^ ERROR_MASKS[error]))
}

/// Construct one fixed syndrome decoder. It is independent of all channel
/// probabilities and is reused for every member of the declared channel family.
pub fn three_qubit_bit_flip_recovery_kraus() -> ThreeQubitBitFlipRecovery {
    let maps = error_maps();
    let mut recovery = [[[0.0; PHYSICAL_DIMENSION]; LOGICAL_DIMENSION]; ERROR_COUNT];
    for syndrome in 0..ERROR_COUNT {
        for logical in 0..LOGICAL_DIMENSION {
            let physical = maps[syndrome][CODEWORDS[logical]];
            recovery[syndrome][logical][physical] = 1.0;
        }
    }
    recovery
}

/// Evaluate a stochastic bit-flip channel using the same fixed universal
/// recovery for every probability vector.
///
/// The channel is N(rho)=sum_e p_e X_e rho X_e, with X_0=I and
/// X_1,X_2,X_3 the single-qubit bit flips. Probabilities are validated but
/// never normalized or repaired.
pub fn simulate_three_qubit_bit_flip_universal_recovery(
    probabilities: [f64; ERROR_COUNT],
    tolerance: f64,
) -> Result<UniversalBitFlipRecoverySample, UniversalRecoveryError> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(UniversalRecoveryError::InvalidTolerance(tolerance));
    }
    for (index, probability) in probabilities.iter().copied().enumerate() {
        if !probability.is_finite() {
            return Err(UniversalRecoveryError::NonFiniteProbability {
                index,
                value: probability,
            });
        }
        if probability < 0.0 {
            return Err(UniversalRecoveryError::NegativeProbability {
                index,
                value: probability,
            });
        }
    }
    let probability_sum: f64 = probabilities.iter().sum();
    if (probability_sum - 1.0).abs() > tolerance {
        return Err(UniversalRecoveryError::ProbabilitySum {
            sum: probability_sum,
            tolerance,
        });
    }

    let maps = error_maps();
    let recovery = three_qubit_bit_flip_recovery_kraus();

    // A_QEC[(i,k),(j,l)] = <c_i| E_k^dagger E_l |c_j>.
    // Exact Knill-Laflamme requires delta_(k,l) delta_(i,j) for this
    // Hilbert-Schmidt orthogonal unitary error set.
    let mut qec_matrix_knill_laflamme_residual: f64 = 0.0;
    for logical_row in 0..LOGICAL_DIMENSION {
        for error_row in 0..ERROR_COUNT {
            for logical_column in 0..LOGICAL_DIMENSION {
                for error_column in 0..ERROR_COUNT {
                    let after_column = maps[error_column][CODEWORDS[logical_column]];
                    let after_row = maps[error_row][after_column];
                    let actual: f64 = if after_row == CODEWORDS[logical_row] {
                        1.0
                    } else {
                        0.0
                    };
                    let expected: f64 = if error_row == error_column && logical_row == logical_column {
                        1.0
                    } else {
                        0.0
                    };
                    qec_matrix_knill_laflamme_residual =
                        qec_matrix_knill_laflamme_residual.max((actual - expected).abs());
                }
            }
        }
    }

    // The noise channel is trace-preserving because every X_e is unitary
    // and the probability vector is normalized.
    let channel_trace_preservation_residual = (probability_sum - 1.0).abs();

    // Verify sum_s R_s^dagger R_s = I_8 for the single fixed decoder.
    let mut recovery_trace_preservation_residual: f64 = 0.0;
    for physical_row in 0..PHYSICAL_DIMENSION {
        for physical_column in 0..PHYSICAL_DIMENSION {
            let mut actual = 0.0;
            for syndrome in 0..ERROR_COUNT {
                for logical in 0..LOGICAL_DIMENSION {
                    actual += recovery[syndrome][logical][physical_row]
                        * recovery[syndrome][logical][physical_column];
                }
            }
            let expected = if physical_row == physical_column {
                1.0
            } else {
                0.0
            };
            recovery_trace_preservation_residual =
                recovery_trace_preservation_residual.max((actual - expected).abs());
        }
    }

    // For every syndrome s and channel error e, R_s X_e V is
    // sqrt(p_e) I_2 when s=e, and zero otherwise. This checks the complete
    // logical Kraus family, not only one input state.
    let mut logical_recovery_residual: f64 = 0.0;
    let mut fidelity_numerator = 0.0;
    for syndrome in 0..ERROR_COUNT {
        for error in 0..ERROR_COUNT {
            let amplitude = probabilities[error].sqrt();
            let mut trace = 0.0;
            for logical_output in 0..LOGICAL_DIMENSION {
                for logical_input in 0..LOGICAL_DIMENSION {
                    let physical = maps[error][CODEWORDS[logical_input]];
                    let actual = recovery[syndrome][logical_output][physical] * amplitude;
                    let expected = if syndrome == error && logical_output == logical_input {
                        amplitude
                    } else {
                        0.0
                    };
                    logical_recovery_residual =
                        logical_recovery_residual.max((actual - expected).abs());
                    if logical_output == logical_input {
                        trace += actual;
                    }
                }
            }
            fidelity_numerator += trace * trace;
        }
    }
    let entanglement_fidelity = fidelity_numerator / (LOGICAL_DIMENSION * LOGICAL_DIMENSION) as f64;

    Ok(UniversalBitFlipRecoverySample {
        probabilities,
        error_set_size: ERROR_COUNT,
        code_dimension: LOGICAL_DIMENSION,
        qec_matrix_knill_laflamme_residual,
        channel_trace_preservation_residual,
        recovery_trace_preservation_residual,
        logical_recovery_residual,
        entanglement_fidelity,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        UniversalRecoveryError, simulate_three_qubit_bit_flip_universal_recovery,
        three_qubit_bit_flip_recovery_kraus,
    };

    const TOLERANCE: f64 = 1.0e-12;

    #[test]
    fn single_fixed_recovery_corrects_each_error_set_member() {
        for error in 0..4 {
            let mut probabilities = [0.0; 4];
            probabilities[error] = 1.0;
            let sample =
                simulate_three_qubit_bit_flip_universal_recovery(probabilities, TOLERANCE).unwrap();
            assert_eq!(sample.qec_matrix_knill_laflamme_residual, 0.0);
            assert_eq!(sample.channel_trace_preservation_residual, 0.0);
            assert_eq!(sample.recovery_trace_preservation_residual, 0.0);
            assert_eq!(sample.logical_recovery_residual, 0.0);
            assert_eq!(sample.entanglement_fidelity, 1.0);
        }
    }

    #[test]
    fn one_recovery_corrects_uniform_and_asymmetric_noise_mixtures() {
        for probabilities in [
            [0.25, 0.25, 0.25, 0.25],
            [0.70, 0.10, 0.15, 0.05],
            [0.01, 0.33, 0.33, 0.33],
            [0.0, 0.0, 0.0, 1.0],
        ] {
            let sample =
                simulate_three_qubit_bit_flip_universal_recovery(probabilities, TOLERANCE).unwrap();
            assert!(sample.qec_matrix_knill_laflamme_residual <= TOLERANCE);
            assert!(sample.recovery_trace_preservation_residual <= TOLERANCE);
            assert!(sample.logical_recovery_residual <= TOLERANCE);
            assert!((sample.entanglement_fidelity - 1.0).abs() <= TOLERANCE);
        }
    }

    #[test]
    fn fixed_decoder_is_independent_of_the_channel_probability_vector() {
        let first = three_qubit_bit_flip_recovery_kraus();
        let _ =
            simulate_three_qubit_bit_flip_universal_recovery([0.80, 0.10, 0.05, 0.05], TOLERANCE)
                .unwrap();
        let second = three_qubit_bit_flip_recovery_kraus();
        let _ =
            simulate_three_qubit_bit_flip_universal_recovery([0.05, 0.15, 0.30, 0.50], TOLERANCE)
                .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn frozen_probability_family_has_unit_fidelity_for_every_member() {
        for case in 0..64 {
            let weights = [
                ((case * 3 + 1) % 17 + 1) as f64,
                ((case * 5 + 4) % 17 + 1) as f64,
                ((case * 7 + 2) % 17 + 1) as f64,
                ((case * 11 + 6) % 17 + 1) as f64,
            ];
            let total: f64 = weights.iter().sum();
            let probabilities = weights.map(|weight| weight / total);
            let sample =
                simulate_three_qubit_bit_flip_universal_recovery(probabilities, TOLERANCE).unwrap();
            assert!((sample.entanglement_fidelity - 1.0).abs() <= TOLERANCE);
            assert!(sample.logical_recovery_residual <= TOLERANCE);
        }
    }

    #[test]
    fn invalid_probability_vectors_are_rejected_without_renormalization() {
        assert!(matches!(
            simulate_three_qubit_bit_flip_universal_recovery([0.2, 0.2, 0.2, 0.2], TOLERANCE),
            Err(UniversalRecoveryError::ProbabilitySum { .. })
        ));
        assert!(matches!(
            simulate_three_qubit_bit_flip_universal_recovery([0.4, -0.1, 0.4, 0.3], TOLERANCE),
            Err(UniversalRecoveryError::NegativeProbability { .. })
        ));
        assert!(matches!(
            simulate_three_qubit_bit_flip_universal_recovery([0.2, f64::NAN, 0.4, 0.4], TOLERANCE),
            Err(UniversalRecoveryError::NonFiniteProbability { .. })
        ));
        assert!(matches!(
            simulate_three_qubit_bit_flip_universal_recovery([0.25; 4], 0.0),
            Err(UniversalRecoveryError::InvalidTolerance(_))
        ));
    }
}
