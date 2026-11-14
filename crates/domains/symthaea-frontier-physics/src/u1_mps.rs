// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

#![allow(clippy::needless_range_loop)]
// Fixed physical/virtual tensor-index contractions are explicit by design.

//! Complex U(1)-symmetric Matrix Product States.
//!
//! Each virtual bond basis vector carries an integer prefix charge. The
//! physical qubit basis has charges q(0)=0 and q(1)=1. A tensor element
//! A[left, physical, right] is admissible only when
//!
//!     right_charge = left_charge + physical_charge.
//!
//! The first virtual bond has one charge-zero state and the final virtual
//! bond has one state carrying the declared total charge. This makes total
//! charge conservation structural rather than a post-hoc expectation-value
//! check.
//!
//! This first implementation provides validated complex tensors, exact
//! product-state construction, exact small-system dense expansion, and
//! contraction-based norm evaluation. It does not yet implement two-site gate
//! application, charge-block SVD, TEBD, truncation or discarded-weight bounds.

use num_complex::Complex64;

/// A single-site qubit tensor with explicit integer charges on both virtual bonds.
///
/// Data layout is row-major with index ((left * 2) + physical) * right_dim + right.
#[derive(Debug, Clone, PartialEq)]
pub struct U1MpsTensor {
    left_charges: Vec<i32>,
    right_charges: Vec<i32>,
    data: Vec<Complex64>,
}

/// Validation or numerical failure in a complex U(1) MPS.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum U1MpsError {
    EmptyChain,
    EmptyBond {
        site: usize,
        side: &'static str,
    },
    DimensionOverflow,
    EntryCount {
        expected: usize,
        actual: usize,
    },
    NonFiniteTensorEntry {
        site: usize,
        index: usize,
    },
    InvalidTolerance(f64),
    LeftBoundary {
        dimension: usize,
        charge: Option<i32>,
    },
    RightBoundary {
        dimension: usize,
        actual_charge: Option<i32>,
        expected_charge: i32,
    },
    BondChargeMismatch {
        bond: usize,
    },
    ChargeSelectionRule {
        site: usize,
        left_charge: i32,
        physical: usize,
        right_charge: i32,
        amplitude: f64,
        tolerance: f64,
    },
    DenseExpansionTooLarge {
        sites: usize,
        maximum_sites: usize,
    },
    NonFiniteContraction(&'static str),
    ComplexNormResidual {
        imaginary_part: f64,
        tolerance: f64,
    },
    NegativeNormSquared(f64),
}

impl core::fmt::Display for U1MpsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyChain => write!(f, "an MPS must contain at least one site"),
            Self::EmptyBond { site, side } => {
                write!(f, "site {site} has an empty {side} virtual bond")
            }
            Self::DimensionOverflow => write!(f, "MPS tensor dimensions overflow usize"),
            Self::EntryCount { expected, actual } => {
                write!(f, "MPS tensor needs {expected} entries, received {actual}")
            }
            Self::NonFiniteTensorEntry { site, index } => {
                write!(f, "MPS tensor entry {index} at site {site} is non-finite")
            }
            Self::InvalidTolerance(value) => {
                write!(f, "tolerance must be positive and finite, got {value}")
            }
            Self::LeftBoundary { dimension, charge } => write!(
                f,
                "left boundary must be one charge-zero state, got dimension {dimension}, charge {charge:?}"
            ),
            Self::RightBoundary {
                dimension,
                actual_charge,
                expected_charge,
            } => write!(
                f,
                "right boundary must be one state with charge {expected_charge}, got dimension {dimension}, charge {actual_charge:?}"
            ),
            Self::BondChargeMismatch { bond } => {
                write!(
                    f,
                    "right/left charge labels disagree on virtual bond {bond}"
                )
            }
            Self::ChargeSelectionRule {
                site,
                left_charge,
                physical,
                right_charge,
                amplitude,
                tolerance,
            } => write!(
                f,
                "U(1) selection rule failed at site {site}: qL={left_charge}, s={physical}, qR={right_charge}, forbidden amplitude={amplitude}, tolerance={tolerance}"
            ),
            Self::DenseExpansionTooLarge {
                sites,
                maximum_sites,
            } => write!(
                f,
                "dense expansion of {sites} sites exceeds configured maximum {maximum_sites}"
            ),
            Self::NonFiniteContraction(name) => {
                write!(f, "MPS contraction produced non-finite {name}")
            }
            Self::ComplexNormResidual {
                imaginary_part,
                tolerance,
            } => write!(
                f,
                "MPS norm has imaginary residual {imaginary_part}, tolerance {tolerance}"
            ),
            Self::NegativeNormSquared(value) => {
                write!(f, "MPS norm squared is negative: {value}")
            }
        }
    }
}

impl std::error::Error for U1MpsError {}

impl U1MpsTensor {
    /// Construct one complex qubit tensor with left/right virtual charge labels.
    ///
    /// Entries in forbidden charge sectors are retained as supplied but are
    /// rejected by ComplexU1Mps::new if they exceed the declared tolerance.
    pub fn new(
        left_charges: Vec<i32>,
        right_charges: Vec<i32>,
        data: Vec<Complex64>,
    ) -> Result<Self, U1MpsError> {
        if left_charges.is_empty() {
            return Err(U1MpsError::EmptyBond {
                site: 0,
                side: "left",
            });
        }
        if right_charges.is_empty() {
            return Err(U1MpsError::EmptyBond {
                site: 0,
                side: "right",
            });
        }
        let expected = left_charges
            .len()
            .checked_mul(2)
            .and_then(|value| value.checked_mul(right_charges.len()))
            .ok_or(U1MpsError::DimensionOverflow)?;
        if data.len() != expected {
            return Err(U1MpsError::EntryCount {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            left_charges,
            right_charges,
            data,
        })
    }

    pub fn left_charges(&self) -> &[i32] {
        &self.left_charges
    }

    pub fn right_charges(&self) -> &[i32] {
        &self.right_charges
    }

    pub const fn physical_dimension(&self) -> usize {
        2
    }

    pub fn left_dimension(&self) -> usize {
        self.left_charges.len()
    }

    pub fn right_dimension(&self) -> usize {
        self.right_charges.len()
    }

    pub fn data(&self) -> &[Complex64] {
        &self.data
    }

    pub fn get(&self, left: usize, physical: usize, right: usize) -> Complex64 {
        self.data[(left * 2 + physical) * self.right_dimension() + right]
    }

    fn max_forbidden_amplitude(&self) -> (f64, usize, usize, usize) {
        let mut maximum = 0.0;
        let mut witness = (0, 0, 0);
        for left in 0..self.left_dimension() {
            for physical in 0..2 {
                for right in 0..self.right_dimension() {
                    if self.right_charges[right] != self.left_charges[left] + physical as i32 {
                        let amplitude = self.get(left, physical, right).norm();
                        if amplitude > maximum {
                            maximum = amplitude;
                            witness = (left, physical, right);
                        }
                    }
                }
            }
        }
        (maximum, witness.0, witness.1, witness.2)
    }
}

/// A complex open-boundary MPS with exact U(1) prefix-charge labels.
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexU1Mps {
    tensors: Vec<U1MpsTensor>,
    total_charge: i32,
    selection_rule_residual: f64,
}

impl ComplexU1Mps {
    /// Validate a complete open-boundary MPS and its charge-selection rules.
    pub fn new(
        tensors: Vec<U1MpsTensor>,
        total_charge: i32,
        tolerance: f64,
    ) -> Result<Self, U1MpsError> {
        if tensors.is_empty() {
            return Err(U1MpsError::EmptyChain);
        }
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(U1MpsError::InvalidTolerance(tolerance));
        }

        for (site, tensor) in tensors.iter().enumerate() {
            for (index, value) in tensor.data.iter().enumerate() {
                if !value.re.is_finite() || !value.im.is_finite() {
                    return Err(U1MpsError::NonFiniteTensorEntry { site, index });
                }
            }
        }

        let first = &tensors[0];
        if first.left_dimension() != 1 || first.left_charges[0] != 0 {
            return Err(U1MpsError::LeftBoundary {
                dimension: first.left_dimension(),
                charge: first.left_charges.first().copied(),
            });
        }
        let last = tensors.last().expect("nonempty tensor chain");
        if last.right_dimension() != 1 || last.right_charges[0] != total_charge {
            return Err(U1MpsError::RightBoundary {
                dimension: last.right_dimension(),
                actual_charge: last.right_charges.first().copied(),
                expected_charge: total_charge,
            });
        }

        for bond in 0..(tensors.len() - 1) {
            if tensors[bond].right_charges != tensors[bond + 1].left_charges {
                return Err(U1MpsError::BondChargeMismatch { bond });
            }
        }

        let mut selection_rule_residual: f64 = 0.0;
        for (site, tensor) in tensors.iter().enumerate() {
            let (amplitude, left, physical, right) = tensor.max_forbidden_amplitude();
            selection_rule_residual = selection_rule_residual.max(amplitude);
            if amplitude > tolerance {
                return Err(U1MpsError::ChargeSelectionRule {
                    site,
                    left_charge: tensor.left_charges[left],
                    physical,
                    right_charge: tensor.right_charges[right],
                    amplitude,
                    tolerance,
                });
            }
        }

        Ok(Self {
            tensors,
            total_charge,
            selection_rule_residual,
        })
    }

    /// Construct an exact computational-basis product state.
    ///
    /// Bit order is most-significant-site first. true denotes physical charge one.
    pub fn product_state(bits: &[bool]) -> Result<Self, U1MpsError> {
        if bits.is_empty() {
            return Err(U1MpsError::EmptyChain);
        }
        let mut prefix_charge = 0_i32;
        let mut tensors = Vec::with_capacity(bits.len());
        for &bit in bits {
            let next_charge = prefix_charge + if bit { 1 } else { 0 };
            let mut data = vec![Complex64::new(0.0, 0.0); 2];
            data[usize::from(bit)] = Complex64::new(1.0, 0.0);
            tensors.push(U1MpsTensor::new(
                vec![prefix_charge],
                vec![next_charge],
                data,
            )?);
            prefix_charge = next_charge;
        }
        Self::new(tensors, prefix_charge, 0.0_f64.max(1.0e-14))
    }

    pub fn site_count(&self) -> usize {
        self.tensors.len()
    }

    pub const fn total_charge(&self) -> i32 {
        self.total_charge
    }

    pub fn tensors(&self) -> &[U1MpsTensor] {
        &self.tensors
    }

    pub fn selection_rule_residual(&self) -> f64 {
        self.selection_rule_residual
    }

    pub fn maximum_bond_dimension(&self) -> usize {
        self.tensors
            .iter()
            .map(|tensor| tensor.left_dimension().max(tensor.right_dimension()))
            .max()
            .unwrap_or(0)
    }

    /// Expand to a dense computational-basis vector for exact small-N overlap tests.
    ///
    /// The caller must explicitly choose the maximum supported number of sites.
    /// No normalization or amplitude thresholding is performed.
    pub fn to_dense(&self, maximum_sites: usize) -> Result<Vec<Complex64>, U1MpsError> {
        if self.site_count() > maximum_sites || self.site_count() >= usize::BITS as usize {
            return Err(U1MpsError::DenseExpansionTooLarge {
                sites: self.site_count(),
                maximum_sites,
            });
        }
        let state_dimension = 1_usize
            .checked_shl(self.site_count() as u32)
            .ok_or(U1MpsError::DimensionOverflow)?;
        let mut dense = vec![Complex64::new(0.0, 0.0); state_dimension];

        for basis in 0..state_dimension {
            let mut contracted = vec![Complex64::new(1.0, 0.0)];
            for (site, tensor) in self.tensors.iter().enumerate() {
                let shift = self.site_count() - 1 - site;
                let physical = (basis >> shift) & 1;
                let mut next = vec![Complex64::new(0.0, 0.0); tensor.right_dimension()];
                for left in 0..tensor.left_dimension() {
                    for right in 0..tensor.right_dimension() {
                        next[right] += contracted[left] * tensor.get(left, physical, right);
                    }
                }
                contracted = next;
            }
            dense[basis] = contracted[0];
        }
        Ok(dense)
    }

    /// Compute the MPS norm by contracting the double-layer transfer matrices.
    pub fn norm_squared(&self, tolerance: f64) -> Result<f64, U1MpsError> {
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(U1MpsError::InvalidTolerance(tolerance));
        }
        let mut environment = vec![Complex64::new(1.0, 0.0)];
        let mut environment_dimension = 1;
        for tensor in &self.tensors {
            if environment_dimension != tensor.left_dimension() {
                return Err(U1MpsError::NonFiniteContraction(
                    "inconsistent transfer-environment dimension",
                ));
            }
            let right_dimension = tensor.right_dimension();
            let environment_entries = right_dimension
                .checked_mul(right_dimension)
                .ok_or(U1MpsError::DimensionOverflow)?;
            let mut next = vec![Complex64::new(0.0, 0.0); environment_entries];
            for bra_left in 0..environment_dimension {
                for ket_left in 0..environment_dimension {
                    let env = environment[bra_left * environment_dimension + ket_left];
                    for physical in 0..2 {
                        for bra_right in 0..right_dimension {
                            let bra = tensor.get(bra_left, physical, bra_right).conj();
                            for ket_right in 0..right_dimension {
                                let ket = tensor.get(ket_left, physical, ket_right);
                                next[bra_right * right_dimension + ket_right] += env * bra * ket;
                            }
                        }
                    }
                }
            }
            if next
                .iter()
                .any(|value| !value.re.is_finite() || !value.im.is_finite())
            {
                return Err(U1MpsError::NonFiniteContraction("transfer environment"));
            }
            environment = next;
            environment_dimension = right_dimension;
        }

        let norm = environment[0];
        if !norm.re.is_finite() || !norm.im.is_finite() {
            return Err(U1MpsError::NonFiniteContraction("final norm"));
        }
        if norm.im.abs() > tolerance * norm.re.abs().max(1.0) {
            return Err(U1MpsError::ComplexNormResidual {
                imaginary_part: norm.im,
                tolerance: tolerance * norm.re.abs().max(1.0),
            });
        }
        if norm.re < -tolerance {
            return Err(U1MpsError::NegativeNormSquared(norm.re));
        }
        Ok(norm.re.max(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::{ComplexU1Mps, U1MpsError, U1MpsTensor};
    use num_complex::Complex64;

    #[test]
    fn product_state_preserves_declared_total_charge_and_dense_bit_order() {
        let state = ComplexU1Mps::product_state(&[true, false, true, false]).unwrap();
        assert_eq!(state.site_count(), 4);
        assert_eq!(state.total_charge(), 2);
        assert_eq!(state.maximum_bond_dimension(), 1);
        assert_eq!(state.selection_rule_residual(), 0.0);
        assert_eq!(state.norm_squared(1.0e-12).unwrap(), 1.0);

        let dense = state.to_dense(8).unwrap();
        for (index, amplitude) in dense.iter().enumerate() {
            if index == 10 {
                assert_eq!(*amplitude, Complex64::new(1.0, 0.0));
            } else {
                assert_eq!(*amplitude, Complex64::new(0.0, 0.0));
            }
        }
    }

    #[test]
    fn entangled_three_qubit_one_excitation_w_state_has_charge_one() {
        let root_two_thirds = (2.0_f64 / 3.0).sqrt();
        let root_one_third = (1.0_f64 / 3.0).sqrt();
        let inverse_root_two = 1.0 / 2.0_f64.sqrt();

        // Site 0: left charge 0; right charges 0 and 1.
        let mut first = vec![Complex64::new(0.0, 0.0); 4];
        first[0] = Complex64::new(root_two_thirds, 0.0);
        first[3] = Complex64::new(root_one_third, 0.0);
        let first = U1MpsTensor::new(vec![0], vec![0, 1], first).unwrap();

        // Site 1: virtual charge paths 0 -> {0,1} and 1 -> 1.
        let mut second = vec![Complex64::new(0.0, 0.0); 2 * 2 * 2];
        second[0] = Complex64::new(inverse_root_two, 0.0);
        second[3] = Complex64::new(inverse_root_two, 0.0);
        second[5] = Complex64::new(1.0, 0.0);
        let second = U1MpsTensor::new(vec![0, 1], vec![0, 1], second).unwrap();

        // Site 2 closes either remaining path to total charge one.
        let mut third = vec![Complex64::new(0.0, 0.0); 2 * 2];
        third[1] = Complex64::new(1.0, 0.0);
        third[2] = Complex64::new(1.0, 0.0);
        let third = U1MpsTensor::new(vec![0, 1], vec![1], third).unwrap();

        let state = ComplexU1Mps::new(vec![first, second, third], 1, 1.0e-14).unwrap();
        assert_eq!(state.total_charge(), 1);
        assert_eq!(state.maximum_bond_dimension(), 2);
        assert!((state.norm_squared(1.0e-12).unwrap() - 1.0).abs() <= 1.0e-12);

        let dense = state.to_dense(4).unwrap();
        let expected = 1.0 / 3.0_f64.sqrt();
        for basis in 0..8 {
            let should_be_nonzero = matches!(basis, 1 | 2 | 4);
            if should_be_nonzero {
                assert!((dense[basis].re - expected).abs() <= 1.0e-12);
                assert!(dense[basis].im.abs() <= 1.0e-12);
            } else {
                assert!(dense[basis].norm() <= 1.0e-12);
            }
        }
    }

    #[test]
    fn constructor_rejects_amplitude_outside_u1_selection_rule() {
        let tensor = U1MpsTensor::new(
            vec![0],
            vec![0],
            vec![Complex64::new(0.0, 0.0), Complex64::new(0.25, 0.0)],
        )
        .unwrap();
        assert!(matches!(
            ComplexU1Mps::new(vec![tensor], 0, 1.0e-12),
            Err(U1MpsError::ChargeSelectionRule { .. })
        ));
    }

    #[test]
    fn constructor_rejects_internal_bond_charge_mismatch_and_wrong_total_charge() {
        let first = U1MpsTensor::new(
            vec![0],
            vec![0, 1],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        )
        .unwrap();
        let second = U1MpsTensor::new(
            vec![1, 0],
            vec![1],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        )
        .unwrap();
        assert!(matches!(
            ComplexU1Mps::new(vec![first.clone(), second.clone()], 1, 1.0e-12),
            Err(U1MpsError::BondChargeMismatch { .. })
        ));
        assert!(matches!(
            ComplexU1Mps::new(vec![first], 0, 1.0e-12),
            Err(U1MpsError::RightBoundary { .. })
        ));
    }

    #[test]
    fn rejects_nonfinite_tensor_entries_empty_chains_and_large_dense_expansions() {
        assert!(matches!(
            ComplexU1Mps::new(Vec::new(), 0, 1.0e-12),
            Err(U1MpsError::EmptyChain)
        ));

        let nonfinite = U1MpsTensor::new(
            vec![0],
            vec![0],
            vec![Complex64::new(f64::NAN, 0.0), Complex64::new(0.0, 0.0)],
        )
        .unwrap();
        assert!(matches!(
            ComplexU1Mps::new(vec![nonfinite], 0, 1.0e-12),
            Err(U1MpsError::NonFiniteTensorEntry { .. })
        ));

        let state = ComplexU1Mps::product_state(&[false, false]).unwrap();
        assert!(matches!(
            state.to_dense(1),
            Err(U1MpsError::DenseExpansionTooLarge { .. })
        ));
    }
}
