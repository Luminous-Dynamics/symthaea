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
//! product-state construction, charge-block dense-to-MPS SVD factorization,
//! exact small-system dense expansion, and contraction-based norm evaluation.
//! It does not yet implement two-site gate
//! application, charge-block SVD, TEBD, truncation or discarded-weight bounds.

use nalgebra::DMatrix;
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
    EmptyBond { site: usize, side: &'static str },
    DimensionOverflow,
    EntryCount { expected: usize, actual: usize },
    NonFiniteTensorEntry { site: usize, index: usize },
    InvalidTolerance(f64),
    LeftBoundary { dimension: usize, charge: Option<i32> },
    RightBoundary {
        dimension: usize,
        actual_charge: Option<i32>,
        expected_charge: i32,
    },
    BondChargeMismatch { bond: usize },
    ChargeSelectionRule {
        site: usize,
        left_charge: i32,
        physical: usize,
        right_charge: i32,
        amplitude: f64,
        tolerance: f64,
    },
    DenseExpansionTooLarge { sites: usize, maximum_sites: usize },
    NonFiniteContraction(&'static str),
    ComplexNormResidual { imaginary_part: f64, tolerance: f64 },
    NegativeNormSquared(f64),
    InvalidTotalCharge { total_charge: i32, site_count: usize },
    ZeroDenseState,
    GateChargeLeakage {
        output_charge: u32,
        input_charge: u32,
        amplitude: f64,
        tolerance: f64,
    },
    GateUnitarityResidual { residual: f64, tolerance: f64 },
    TwoSiteBondMismatch,
    TwoSiteShapeMismatch,
    TwoSiteDenseExpansionRequiresOpenBoundaries,
    DenseChargeLeakage { basis_index: usize, amplitude: f64, tolerance: f64 },
    SvdFailed { rows: usize, columns: usize },
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
                write!(f, "right/left charge labels disagree on virtual bond {bond}")
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
            Self::InvalidTotalCharge { total_charge, site_count } => write!(
                f,
                "total charge {total_charge} is invalid for a {site_count}-site qubit MPS"
            ),
            Self::ZeroDenseState => write!(f, "the dense input has no nonzero amplitude in the declared charge sector"),
            Self::GateChargeLeakage {
                output_charge,
                input_charge,
                amplitude,
                tolerance,
            } => write!(
                f,
                "two-site gate couples charge {input_charge} to {output_charge} with amplitude {amplitude}, tolerance {tolerance}"
            ),
            Self::GateUnitarityResidual { residual, tolerance } => write!(
                f,
                "two-site gate unitarity residual {residual} exceeds tolerance {tolerance}"
            ),
            Self::TwoSiteBondMismatch => write!(f, "adjacent MPS tensors have different virtual charge labels"),
            Self::TwoSiteShapeMismatch => write!(f, "two-site state tensor dimensions are inconsistent"),
            Self::TwoSiteDenseExpansionRequiresOpenBoundaries => {
                write!(f, "two-site dense expansion requires one-dimensional external boundary bonds")
            }
            Self::DenseChargeLeakage { basis_index, amplitude, tolerance } => write!(
                f,
                "dense basis amplitude {basis_index} has forbidden charge-sector magnitude {amplitude}, tolerance {tolerance}"
            ),
            Self::SvdFailed { rows, columns } => {
                write!(f, "charge-block SVD failed to converge for {rows}x{columns} block")
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
    dense_input_charge_leakage_max: f64,
    dense_input_charge_leakage_weight: f64,
    dense_reconstruction_max_residual: Option<f64>,
    dense_reconstruction_l2_residual: Option<f64>,
    dense_svd_rank_by_charge_and_cut: Option<Vec<Vec<(i32, usize)>>>,
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
            dense_input_charge_leakage_max: 0.0,
            dense_input_charge_leakage_weight: 0.0,
            dense_reconstruction_max_residual: None,
            dense_reconstruction_l2_residual: None,
            dense_svd_rank_by_charge_and_cut: None,
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

    pub fn dense_input_charge_leakage_max(&self) -> f64 {
        self.dense_input_charge_leakage_max
    }

    pub fn dense_input_charge_leakage_weight(&self) -> f64 {
        self.dense_input_charge_leakage_weight
    }

    pub fn dense_reconstruction_max_residual(&self) -> Option<f64> {
        self.dense_reconstruction_max_residual
    }

    pub fn dense_reconstruction_l2_residual(&self) -> Option<f64> {
        self.dense_reconstruction_l2_residual
    }

    pub fn dense_svd_rank_by_charge_and_cut(&self) -> Option<&[Vec<(i32, usize)>]> {
        self.dense_svd_rank_by_charge_and_cut.as_deref()
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


    /// Factor an exact dense fixed-charge state into a complex U(1)-symmetric MPS.
    ///
    /// The decomposition performs a full SVD independently in each intermediate
    /// charge sector at every cut. It applies no bond cap, singular-value cutoff,
    /// normalization or truncation. All returned singular directions are retained,
    /// including numerically zero directions; callers can therefore compare the
    /// dense reconstruction directly against the source amplitudes.
    ///
    /// Bit order is most-significant-site first. Dense amplitudes outside the
    /// declared total-charge sector are rejected when they exceed tolerance.
    pub fn from_dense_fixed_charge(
        amplitudes: &[Complex64],
        site_count: usize,
        total_charge: i32,
        tolerance: f64,
    ) -> Result<Self, U1MpsError> {
        if site_count == 0 {
            return Err(U1MpsError::EmptyChain);
        }
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(U1MpsError::InvalidTolerance(tolerance));
        }
        if site_count > i32::MAX as usize || total_charge < 0 || total_charge > site_count as i32 {
            return Err(U1MpsError::InvalidTotalCharge {
                total_charge,
                site_count,
            });
        }
        if site_count >= usize::BITS as usize {
            return Err(U1MpsError::DenseExpansionTooLarge {
                sites: site_count,
                maximum_sites: usize::BITS as usize - 1,
            });
        }
        let dense_dimension = 1_usize
            .checked_shl(site_count as u32)
            .ok_or(U1MpsError::DimensionOverflow)?;
        if amplitudes.len() != dense_dimension {
            return Err(U1MpsError::EntryCount {
                expected: dense_dimension,
                actual: amplitudes.len(),
            });
        }
        let mut dense_input_charge_leakage_max: f64 = 0.0;
        let mut dense_input_charge_leakage_weight = 0.0;
        for (index, amplitude) in amplitudes.iter().enumerate() {
            if !amplitude.re.is_finite() || !amplitude.im.is_finite() {
                return Err(U1MpsError::NonFiniteTensorEntry { site: 0, index });
            }
            if (index.count_ones() as i32) != total_charge {
                let forbidden = amplitude.norm();
                if !forbidden.is_finite() {
                    return Err(U1MpsError::NonFiniteTensorEntry { site: 0, index });
                }
                dense_input_charge_leakage_max = dense_input_charge_leakage_max.max(forbidden);
                dense_input_charge_leakage_weight += forbidden * forbidden;
                if !dense_input_charge_leakage_weight.is_finite() {
                    return Err(U1MpsError::NonFiniteContraction(
                        "dense input charge-leakage weight",
                    ));
                }
                if forbidden > tolerance {
                    return Err(U1MpsError::DenseChargeLeakage {
                        basis_index: index,
                        amplitude: forbidden,
                        tolerance,
                    });
                }
            }
        }

        // The current residual is a matrix from the left virtual basis to the
        // remaining computational-basis suffix. Initially there is one left
        // state with prefix charge zero and the full dense state as its row.
        let mut current = DMatrix::<Complex64>::from_row_slice(1, dense_dimension, amplitudes);
        let mut left_charges = vec![0_i32];
        let mut tensors = Vec::with_capacity(site_count);
        let mut svd_rank_by_charge_and_cut = Vec::with_capacity(site_count.saturating_sub(1));

        for site in 0..(site_count - 1) {
            let suffix_sites = site_count - site - 1;
            let suffix_dimension = 1_usize
                .checked_shl(suffix_sites as u32)
                .ok_or(U1MpsError::DimensionOverflow)?;

            // Group local (left, physical) rows by their resulting prefix charge.
            let mut row_groups: Vec<(i32, Vec<(usize, usize)>)> = Vec::new();
            for left in 0..left_charges.len() {
                for physical in 0..2 {
                    let charge = left_charges[left] + physical as i32;
                    if let Some((_, rows)) = row_groups.iter_mut().find(|(q, _)| *q == charge) {
                        rows.push((left, physical));
                    } else {
                        row_groups.push((charge, vec![(left, physical)]));
                    }
                }
            }

            struct ChargeBlockSvd {
                charge: i32,
                rows: Vec<(usize, usize)>,
                columns: Vec<usize>,
                u: DMatrix<Complex64>,
                v_t: DMatrix<Complex64>,
                singular_values: Vec<f64>,
            }

            let mut blocks = Vec::new();
            let max_block_dimension = left_charges.len().saturating_mul(2).max(suffix_dimension);
            let max_iterations = 10_000usize.max(
                64usize.saturating_mul(max_block_dimension.saturating_mul(max_block_dimension)),
            );

            for (charge, rows) in row_groups {
                let suffix_charge = total_charge - charge;
                if suffix_charge < 0 || suffix_charge > suffix_sites as i32 {
                    continue;
                }
                let columns: Vec<usize> = (0..suffix_dimension)
                    .filter(|suffix| suffix.count_ones() as i32 == suffix_charge)
                    .collect();
                if rows.is_empty() || columns.is_empty() {
                    continue;
                }

                let mut block = DMatrix::<Complex64>::zeros(rows.len(), columns.len());
                for (row_index, &(left, physical)) in rows.iter().enumerate() {
                    for (column_index, &suffix) in columns.iter().enumerate() {
                        let full_suffix = physical * suffix_dimension + suffix;
                        block[(row_index, column_index)] = current[(left, full_suffix)];
                    }
                }

                let (block_rows, block_columns) = (block.nrows(), block.ncols());
                let decomposition = block
                    .try_svd(true, true, tolerance, max_iterations)
                    .ok_or(U1MpsError::SvdFailed {
                        rows: block_rows,
                        columns: block_columns,
                    })?;
                let u = decomposition.u.ok_or(U1MpsError::SvdFailed {
                    rows: block_rows,
                    columns: block_columns,
                })?;
                let v_t = decomposition.v_t.ok_or(U1MpsError::SvdFailed {
                    rows: block_rows,
                    columns: block_columns,
                })?;
                // Remove exact zero singular directions only. Every positive
                // singular value is retained, however small; no cutoff is applied.
                let kept_indices: Vec<usize> = decomposition
                    .singular_values
                    .iter()
                    .enumerate()
                    .filter_map(|(index, value)| (*value > 0.0).then_some(index))
                    .collect();
                if kept_indices.is_empty() {
                    continue;
                }
                let kept_u = DMatrix::<Complex64>::from_fn(rows.len(), kept_indices.len(), |r, col| {
                    u[(r, kept_indices[col])]
                });
                let kept_v_t =
                    DMatrix::<Complex64>::from_fn(kept_indices.len(), columns.len(), |r, col| {
                        v_t[(kept_indices[r], col)]
                    });
                let singular_values = kept_indices
                    .iter()
                    .map(|&index| decomposition.singular_values[index])
                    .collect();
                blocks.push(ChargeBlockSvd {
                    charge,
                    rows,
                    columns,
                    u: kept_u,
                    v_t: kept_v_t,
                    singular_values,
                });
            }

            let cut_ranks: Vec<(i32, usize)> = blocks
                .iter()
                .map(|block| (block.charge, block.singular_values.len()))
                .collect();
            svd_rank_by_charge_and_cut.push(cut_ranks);
            let next_bond_dimension = blocks.iter().try_fold(0_usize, |sum, block| {
                sum.checked_add(block.singular_values.len())
                    .ok_or(U1MpsError::DimensionOverflow)
            })?;
            if next_bond_dimension == 0 {
                return Err(U1MpsError::ZeroDenseState);
            }

            let tensor_entries = left_charges
                .len()
                .checked_mul(2)
                .and_then(|value| value.checked_mul(next_bond_dimension))
                .ok_or(U1MpsError::DimensionOverflow)?;
            let mut tensor_data = vec![Complex64::new(0.0, 0.0); tensor_entries];
            let next_entries = next_bond_dimension
                .checked_mul(suffix_dimension)
                .ok_or(U1MpsError::DimensionOverflow)?;
            let mut next_current =
                DMatrix::<Complex64>::zeros(next_bond_dimension, suffix_dimension);
            let mut next_charges = Vec::with_capacity(next_bond_dimension);
            let mut offset = 0;

            for block in blocks {
                let block_rank = block.singular_values.len();
                for _ in 0..block_rank {
                    next_charges.push(block.charge);
                }
                for singular_index in 0..block_rank {
                    let new_bond = offset + singular_index;
                    for (row_index, &(left, physical)) in block.rows.iter().enumerate() {
                        let index =
                            (left * 2 + physical) * next_bond_dimension + new_bond;
                        tensor_data[index] = block.u[(row_index, singular_index)];
                    }
                    let singular_value = block.singular_values[singular_index];
                    for (column_index, &suffix) in block.columns.iter().enumerate() {
                        next_current[(new_bond, suffix)] =
                            block.v_t[(singular_index, column_index)] * singular_value;
                    }
                }
                offset += block_rank;
            }

            debug_assert_eq!(next_current.len(), next_entries);
            let tensor = U1MpsTensor::new(left_charges, next_charges.clone(), tensor_data)?;
            tensors.push(tensor);
            left_charges = next_charges;
            current = next_current;
        }

        if current.ncols() != 2 {
            return Err(U1MpsError::NonFiniteContraction(
                "final dense-to-MPS suffix dimension",
            ));
        }
        let mut last_data = vec![Complex64::new(0.0, 0.0); left_charges.len() * 2];
        for left in 0..left_charges.len() {
            for physical in 0..2 {
                if left_charges[left] + physical as i32 == total_charge {
                    last_data[left * 2 + physical] = current[(left, physical)];
                } else {
                    let forbidden = current[(left, physical)].norm();
                    if forbidden > tolerance {
                        return Err(U1MpsError::DenseChargeLeakage {
                            basis_index: 0,
                            amplitude: forbidden,
                            tolerance,
                        });
                    }
                }
            }
        }
        let last = U1MpsTensor::new(left_charges, vec![total_charge], last_data)?;
        tensors.push(last);

        let mut state = Self::new(tensors, total_charge, tolerance)?;
        state.dense_input_charge_leakage_max = dense_input_charge_leakage_max;
        state.dense_input_charge_leakage_weight = dense_input_charge_leakage_weight;
        let reconstructed = state.to_dense(site_count)?;
        let mut maximum_residual: f64 = 0.0;
        let mut residual_squared = 0.0;
        for (actual, expected) in reconstructed.iter().zip(amplitudes.iter()) {
            let residual = (*actual - *expected).norm();
            maximum_residual = maximum_residual.max(residual);
            residual_squared += residual * residual;
        }
        state.dense_reconstruction_max_residual = Some(maximum_residual);
        state.dense_reconstruction_l2_residual = Some(residual_squared.sqrt());
        state.dense_svd_rank_by_charge_and_cut = Some(svd_rank_by_charge_and_cut);
        Ok(state)
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
            if next.iter().any(|value| !value.re.is_finite() || !value.im.is_finite()) {
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



/// A validated two-qubit gate that preserves total U(1) excitation number.
///
/// Basis order is |00>, |01>, |10>, |11>. The matrix is row-major
/// (output_basis, input_basis). Its only nontrivial block may be the U(2)
/// action on {|01>,|10>}; |00> and |11> acquire independent phases.
#[derive(Debug, Clone, PartialEq)]
pub struct U1TwoSiteGate {
    matrix: [[Complex64; 4]; 4],
    charge_conservation_residual: f64,
    unitarity_residual: f64,
}

/// The exact two-site tensor obtained after contracting an MPS bond and
/// applying one U(1)-conserving two-qubit gate.
///
/// Data layout is ((left * 4) + physical_pair) * right_dim + right, with
/// physical_pair = first_qubit * 2 + second_qubit.
#[derive(Debug, Clone, PartialEq)]
pub struct U1TwoSiteState {
    left_charges: Vec<i32>,
    right_charges: Vec<i32>,
    data: Vec<Complex64>,
    selection_rule_residual: f64,
}

impl U1TwoSiteGate {
    /// Construct and validate a charge-conserving 4x4 unitary.
    pub fn new(
        matrix: [[Complex64; 4]; 4],
        tolerance: f64,
    ) -> Result<Self, U1MpsError> {
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(U1MpsError::InvalidTolerance(tolerance));
        }
        for row in 0..4 {
            for column in 0..4 {
                let value = matrix[row][column];
                if !value.re.is_finite() || !value.im.is_finite() {
                    return Err(U1MpsError::NonFiniteTensorEntry {
                        site: 0,
                        index: row * 4 + column,
                    });
                }
            }
        }

        let mut charge_conservation_residual: f64 = 0.0;
        for output in 0..4 {
            for input in 0..4 {
                if output.count_ones() != input.count_ones() {
                    charge_conservation_residual =
                        charge_conservation_residual.max(matrix[output][input].norm());
                }
            }
        }
        if charge_conservation_residual > tolerance {
            let witness = (0..4)
                .flat_map(|output| (0..4).map(move |input| (output, input)))
                .find(|(output, input)| {
                    output.count_ones() != input.count_ones()
                        && matrix[*output][*input].norm() > tolerance
                })
                .unwrap_or((0, 0));
            return Err(U1MpsError::GateChargeLeakage {
                output_charge: witness.0.count_ones(),
                input_charge: witness.1.count_ones(),
                amplitude: matrix[witness.0][witness.1].norm(),
                tolerance,
            });
        }

        let mut unitarity_residual: f64 = 0.0;
        for row in 0..4 {
            for column in 0..4 {
                let mut overlap = Complex64::new(0.0, 0.0);
                for inner in 0..4 {
                    overlap += matrix[inner][row].conj() * matrix[inner][column];
                }
                let expected = if row == column {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                unitarity_residual =
                    unitarity_residual.max((overlap - expected).norm());
            }
        }
        if unitarity_residual > tolerance {
            return Err(U1MpsError::GateUnitarityResidual {
                residual: unitarity_residual,
                tolerance,
            });
        }

        Ok(Self {
            matrix,
            charge_conservation_residual,
            unitarity_residual,
        })
    }

    pub fn matrix(&self) -> &[[Complex64; 4]; 4] {
        &self.matrix
    }

    pub fn charge_conservation_residual(&self) -> f64 {
        self.charge_conservation_residual
    }

    pub fn unitarity_residual(&self) -> f64 {
        self.unitarity_residual
    }

    /// Exact identity gate.
    pub fn identity() -> Self {
        let mut matrix = [[Complex64::new(0.0, 0.0); 4]; 4];
        for index in 0..4 {
            matrix[index][index] = Complex64::new(1.0, 0.0);
        }
        Self {
            matrix,
            charge_conservation_residual: 0.0,
            unitarity_residual: 0.0,
        }
    }

    /// Exact two-qubit SWAP, which preserves total excitation number.
    pub fn swap() -> Self {
        let mut matrix = [[Complex64::new(0.0, 0.0); 4]; 4];
        matrix[0][0] = Complex64::new(1.0, 0.0);
        matrix[1][2] = Complex64::new(1.0, 0.0);
        matrix[2][1] = Complex64::new(1.0, 0.0);
        matrix[3][3] = Complex64::new(1.0, 0.0);
        Self {
            matrix,
            charge_conservation_residual: 0.0,
            unitarity_residual: 0.0,
        }
    }
}

impl U1TwoSiteState {
    /// Construct a checked two-site tensor with external virtual-charge labels.
    pub fn new(
        left_charges: Vec<i32>,
        right_charges: Vec<i32>,
        data: Vec<Complex64>,
        tolerance: f64,
    ) -> Result<Self, U1MpsError> {
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(U1MpsError::InvalidTolerance(tolerance));
        }
        if left_charges.is_empty() {
            return Err(U1MpsError::EmptyBond {
                site: 0,
                side: "left",
            });
        }
        if right_charges.is_empty() {
            return Err(U1MpsError::EmptyBond {
                site: 1,
                side: "right",
            });
        }
        let expected = left_charges
            .len()
            .checked_mul(4)
            .and_then(|value| value.checked_mul(right_charges.len()))
            .ok_or(U1MpsError::DimensionOverflow)?;
        if data.len() != expected {
            return Err(U1MpsError::EntryCount {
                expected,
                actual: data.len(),
            });
        }

        let right_dimension = right_charges.len();
        let mut selection_rule_residual: f64 = 0.0;
        let mut selection_rule_witness = (0_usize, 0_usize, 0_usize);
        for left in 0..left_charges.len() {
            for physical_pair in 0..4 {
                for right in 0..right_dimension {
                    let index = (left * 4 + physical_pair) * right_dimension + right;
                    let value = data[index];
                    if !value.re.is_finite() || !value.im.is_finite() {
                        return Err(U1MpsError::NonFiniteTensorEntry { site: 0, index });
                    }
                    if right_charges[right]
                        != left_charges[left] + physical_pair.count_ones() as i32
                    {
                        let amplitude = value.norm();
                        if amplitude > selection_rule_residual {
                            selection_rule_residual = amplitude;
                            selection_rule_witness = (left, physical_pair, right);
                        }
                    }
                }
            }
        }
        if selection_rule_residual > tolerance {
            let (left, physical_pair, right) = selection_rule_witness;
            return Err(U1MpsError::ChargeSelectionRule {
                site: 0,
                left_charge: left_charges[left],
                physical: physical_pair,
                right_charge: right_charges[right],
                amplitude: selection_rule_residual,
                tolerance,
            });
        }
        Ok(Self {
            left_charges,
            right_charges,
            data,
            selection_rule_residual,
        })
    }

    pub fn left_charges(&self) -> &[i32] {
        &self.left_charges
    }

    pub fn right_charges(&self) -> &[i32] {
        &self.right_charges
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

    pub fn selection_rule_residual(&self) -> f64 {
        self.selection_rule_residual
    }

    pub fn get(&self, left: usize, physical_pair: usize, right: usize) -> Complex64 {
        self.data[(left * 4 + physical_pair) * self.right_dimension() + right]
    }

    /// Contract two adjacent site tensors into one exact two-site state.
    pub fn from_adjacent_tensors(
        left: &U1MpsTensor,
        right: &U1MpsTensor,
        tolerance: f64,
    ) -> Result<Self, U1MpsError> {
        if left.right_charges != right.left_charges {
            return Err(U1MpsError::TwoSiteBondMismatch);
        }
        let expected = left
            .left_dimension()
            .checked_mul(4)
            .and_then(|value| value.checked_mul(right.right_dimension()))
            .ok_or(U1MpsError::DimensionOverflow)?;
        let mut data = vec![Complex64::new(0.0, 0.0); expected];
        for left_index in 0..left.left_dimension() {
            for first_physical in 0..2 {
                for second_physical in 0..2 {
                    let physical_pair = first_physical * 2 + second_physical;
                    for right_index in 0..right.right_dimension() {
                        let mut value = Complex64::new(0.0, 0.0);
                        for bond in 0..left.right_dimension() {
                            value += left.get(left_index, first_physical, bond)
                                * right.get(bond, second_physical, right_index);
                        }
                        let index = (left_index * 4 + physical_pair)
                            * right.right_dimension()
                            + right_index;
                        data[index] = value;
                    }
                }
            }
        }
        Self::new(
            left.left_charges.clone(),
            right.right_charges.clone(),
            data,
            tolerance,
        )
    }

    /// Apply a validated two-site gate without splitting or truncating the bond.
    pub fn apply_gate(
        &self,
        gate: &U1TwoSiteGate,
        tolerance: f64,
    ) -> Result<Self, U1MpsError> {
        let right_dimension = self.right_dimension();
        let entries = self
            .left_dimension()
            .checked_mul(4)
            .and_then(|value| value.checked_mul(right_dimension))
            .ok_or(U1MpsError::DimensionOverflow)?;
        let mut output = vec![Complex64::new(0.0, 0.0); entries];
        for left in 0..self.left_dimension() {
            for output_pair in 0..4 {
                for right in 0..right_dimension {
                    let mut value = Complex64::new(0.0, 0.0);
                    for input_pair in 0..4 {
                        value += gate.matrix[output_pair][input_pair]
                            * self.get(left, input_pair, right);
                    }
                    let index = (left * 4 + output_pair) * right_dimension + right;
                    output[index] = value;
                }
            }
        }
        Self::new(
            self.left_charges.clone(),
            self.right_charges.clone(),
            output,
            tolerance,
        )
    }

    /// Return the four computational-basis amplitudes for open-boundary tensors.
    pub fn to_dense_two_qubit(&self) -> Result<[Complex64; 4], U1MpsError> {
        if self.left_dimension() != 1 || self.right_dimension() != 1 {
            return Err(U1MpsError::TwoSiteDenseExpansionRequiresOpenBoundaries);
        }
        Ok(core::array::from_fn(|index| self.get(0, index, 0)))
    }
}


#[cfg(test)]
mod tests {
    use super::{
        ComplexU1Mps, U1MpsError, U1MpsTensor, U1TwoSiteGate, U1TwoSiteState,
    };
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
            vec![
                Complex64::new(f64::NAN, 0.0),
                Complex64::new(0.0, 0.0),
            ],
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

    #[test]
    fn dense_fixed_charge_w_state_factorizes_by_charge_blocks_without_truncation() {
        let amplitude = 1.0 / 3.0_f64.sqrt();
        let mut dense = vec![Complex64::new(0.0, 0.0); 8];
        dense[1] = Complex64::new(amplitude, 0.0);
        dense[2] = Complex64::new(amplitude, 0.0);
        dense[4] = Complex64::new(amplitude, 0.0);

        let state = ComplexU1Mps::from_dense_fixed_charge(&dense, 3, 1, 1.0e-12).unwrap();
        assert_eq!(state.total_charge(), 1);
        assert_eq!(state.dense_input_charge_leakage_max(), 0.0);
        assert_eq!(state.dense_input_charge_leakage_weight(), 0.0);
        assert!(state.dense_reconstruction_max_residual().unwrap() <= 1.0e-10);
        assert!(state.dense_reconstruction_l2_residual().unwrap() <= 1.0e-10);
        assert!(state.maximum_bond_dimension() <= 2);
        assert!((state.norm_squared(1.0e-12).unwrap() - 1.0).abs() <= 1.0e-10);

        let reconstructed = state.to_dense(4).unwrap();
        for index in 0..dense.len() {
            assert!(
                (reconstructed[index] - dense[index]).norm() <= 1.0e-10,
                "basis={index}, reconstructed={:?}, expected={:?}",
                reconstructed[index],
                dense[index]
            );
        }
    }

    #[test]
    fn dense_fixed_charge_product_state_factorizes_exactly() {
        let mut dense = vec![Complex64::new(0.0, 0.0); 16];
        dense[10] = Complex64::new(0.75, -0.25);
        let state = ComplexU1Mps::from_dense_fixed_charge(&dense, 4, 2, 1.0e-12).unwrap();
        assert_eq!(state.total_charge(), 2);
        assert_eq!(state.maximum_bond_dimension(), 1);
        assert_eq!(state.dense_svd_rank_by_charge_and_cut().unwrap().len(), 3);
        assert!((state.norm_squared(1.0e-12).unwrap() - 0.625).abs() <= 1.0e-12);
        assert_eq!(state.to_dense(8).unwrap(), dense);
    }

    #[test]
    fn dense_fixed_charge_factorization_rejects_charge_leakage_and_bad_dimensions() {
        let mut dense = vec![Complex64::new(0.0, 0.0); 8];
        dense[1] = Complex64::new(1.0 / 3.0_f64.sqrt(), 0.0);
        dense[2] = Complex64::new(1.0 / 3.0_f64.sqrt(), 0.0);
        dense[4] = Complex64::new(1.0 / 3.0_f64.sqrt(), 0.0);
        dense[0] = Complex64::new(1.0e-3, 0.0);
        assert!(matches!(
            ComplexU1Mps::from_dense_fixed_charge(&dense, 3, 1, 1.0e-6),
            Err(U1MpsError::DenseChargeLeakage { .. })
        ));

        assert!(matches!(
            ComplexU1Mps::from_dense_fixed_charge(&dense[..7], 3, 1, 1.0e-12),
            Err(U1MpsError::EntryCount { .. })
        ));
        assert!(matches!(
            ComplexU1Mps::from_dense_fixed_charge(&dense, 3, 4, 1.0e-12),
            Err(U1MpsError::InvalidTotalCharge { .. })
        ));
    }


    #[test]
    fn dense_factorization_reports_tolerated_out_of_sector_input_leakage() {
        let amplitude = 1.0 / 3.0_f64.sqrt();
        let mut dense = vec![Complex64::new(0.0, 0.0); 8];
        dense[1] = Complex64::new(amplitude, 0.0);
        dense[2] = Complex64::new(amplitude, 0.0);
        dense[4] = Complex64::new(amplitude, 0.0);
        dense[0] = Complex64::new(1.0e-13, 0.0);

        let state = ComplexU1Mps::from_dense_fixed_charge(&dense, 3, 1, 1.0e-12).unwrap();
        assert!((state.dense_input_charge_leakage_max() - 1.0e-13).abs() <= 1.0e-25);
        assert!((state.dense_input_charge_leakage_weight() - 1.0e-26).abs() <= 1.0e-38);
        assert!((state.dense_reconstruction_max_residual().unwrap() - 1.0e-13).abs() <= 1.0e-25);
    }


    #[test]
    fn dense_fixed_charge_factorization_preserves_tiny_positive_schmidt_weight() {
        let tiny = 1.0e-18;
        let large = (1.0 - tiny * tiny).sqrt();
        let mut dense = vec![Complex64::new(0.0, 0.0); 4];
        dense[1] = Complex64::new(large, 0.0);
        dense[2] = Complex64::new(tiny, 0.0);
        let state = ComplexU1Mps::from_dense_fixed_charge(&dense, 2, 1, 1.0e-12).unwrap();
        assert_eq!(state.maximum_bond_dimension(), 2);
        let reconstructed = state.to_dense(4).unwrap();
        assert!((reconstructed[2].re - tiny).abs() <= 1.0e-30);
        assert!(state.dense_reconstruction_max_residual().unwrap() <= 1.0e-15);
    }

    #[test]
    fn dense_factorization_reports_tolerated_out_of_sector_weight_and_reconstruction_error() {
        let mut dense = vec![Complex64::new(0.0, 0.0); 4];
        dense[1] = Complex64::new(1.0, 0.0);
        dense[0] = Complex64::new(1.0e-13, 0.0);
        let state = ComplexU1Mps::from_dense_fixed_charge(&dense, 2, 1, 1.0e-12).unwrap();
        assert!((state.dense_input_charge_leakage_max() - 1.0e-13).abs() <= 1.0e-25);
        assert!((state.dense_input_charge_leakage_weight() - 1.0e-26).abs() <= 1.0e-38);
        assert!((state.dense_reconstruction_max_residual().unwrap() - 1.0e-13).abs() <= 1.0e-25);
        assert!((state.dense_reconstruction_l2_residual().unwrap() - 1.0e-13).abs() <= 1.0e-25);
    }

    #[test]
    fn dense_factorization_rejects_zero_state_instead_of_fabricating_a_bond() {
        let dense = vec![Complex64::new(0.0, 0.0); 4];
        assert_eq!(
            ComplexU1Mps::from_dense_fixed_charge(&dense, 2, 1, 1.0e-12),
            Err(U1MpsError::ZeroDenseState)
        );
    }


    fn single_excitation_state() -> ComplexU1Mps {
        let amplitude = 1.0 / 2.0_f64.sqrt();
        let dense = [
            Complex64::new(0.0, 0.0),
            Complex64::new(amplitude, 0.0),
            Complex64::new(amplitude, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        ComplexU1Mps::from_dense_fixed_charge(&dense, 2, 1, 1.0e-12).unwrap()
    }

    fn one_excitation_rotation() -> U1TwoSiteGate {
        let mut matrix = [[Complex64::new(0.0, 0.0); 4]; 4];
        matrix[0][0] = Complex64::new(1.0, 0.0);
        matrix[3][3] = Complex64::new(1.0, 0.0);
        let cosine = 1.0 / 2.0_f64.sqrt();
        let sine = 1.0 / 2.0_f64.sqrt();
        matrix[1][1] = Complex64::new(cosine, 0.0);
        matrix[1][2] = Complex64::new(sine, 0.0);
        matrix[2][1] = Complex64::new(-sine, 0.0);
        matrix[2][2] = Complex64::new(cosine, 0.0);
        U1TwoSiteGate::new(matrix, 1.0e-12).unwrap()
    }

    #[test]
    fn two_site_gate_contraction_reconstructs_adjacent_mps_tensors() {
        let state = single_excitation_state();
        let pair = U1TwoSiteState::from_adjacent_tensors(
            &state.tensors()[0],
            &state.tensors()[1],
            1.0e-12,
        )
        .unwrap();
        let dense = pair.to_dense_two_qubit().unwrap();
        assert_eq!(dense, state.to_dense(2).unwrap().try_into().unwrap());
        assert!(pair.selection_rule_residual() <= 1.0e-12);
    }

    #[test]
    fn u1_two_site_rotation_preserves_charge_and_matches_dense_known_answer() {
        let state = single_excitation_state();
        let pair = U1TwoSiteState::from_adjacent_tensors(
            &state.tensors()[0],
            &state.tensors()[1],
            1.0e-12,
        )
        .unwrap();
        let gate = one_excitation_rotation();
        assert!(gate.charge_conservation_residual() <= 1.0e-12);
        assert!(gate.unitarity_residual() <= 1.0e-12);

        let output = pair.apply_gate(&gate, 1.0e-12).unwrap();
        let dense = output.to_dense_two_qubit().unwrap();
        assert!((dense[0].norm()).abs() <= 1.0e-12);
        assert!((dense[1].re - 1.0).abs() <= 1.0e-12);
        assert!(dense[2].norm() <= 1.0e-12);
        assert!(dense[3].norm() <= 1.0e-12);
        assert!(output.selection_rule_residual() <= 1.0e-12);
    }

    #[test]
    fn u1_swap_gate_exchanges_single_excitation_basis_states() {
        let dense = [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let pair = U1TwoSiteState::new(vec![0], vec![1], dense.to_vec(), 1.0e-12).unwrap();
        let swapped = pair.apply_gate(&U1TwoSiteGate::swap(), 1.0e-12).unwrap();
        let actual = swapped.to_dense_two_qubit().unwrap();
        assert_eq!(actual[1], Complex64::new(0.0, 0.0));
        assert_eq!(actual[2], Complex64::new(1.0, 0.0));
        assert_eq!(actual[0], Complex64::new(0.0, 0.0));
        assert_eq!(actual[3], Complex64::new(0.0, 0.0));
    }

    #[test]
    fn gate_constructor_rejects_charge_nonconserving_and_nonunitary_matrices() {
        let half = 1.0 / 2.0_f64.sqrt();
        let nonconserving = [
            [
                Complex64::new(half, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(half, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(half, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(half, 0.0),
            ],
            [
                Complex64::new(half, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-half, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(half, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-half, 0.0),
            ],
        ];
        assert!(matches!(
            U1TwoSiteGate::new(nonconserving, 1.0e-12),
            Err(U1MpsError::GateChargeLeakage { .. })
        ));

        let mut nonunitary = U1TwoSiteGate::identity().matrix().to_owned();
        nonunitary[3][3] = Complex64::new(0.5, 0.0);
        assert!(matches!(
            U1TwoSiteGate::new(nonunitary, 1.0e-12),
            Err(U1MpsError::GateUnitarityResidual { .. })
        ));
    }

    #[test]
    fn adjacent_two_site_contraction_rejects_bond_charge_mismatch() {
        let left = U1MpsTensor::new(
            vec![0],
            vec![0],
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        )
        .unwrap();
        let right = U1MpsTensor::new(
            vec![1],
            vec![1],
            vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        )
        .unwrap();
        assert!(matches!(
            U1TwoSiteState::from_adjacent_tensors(&left, &right, 1.0e-12),
            Err(U1MpsError::TwoSiteBondMismatch)
        ));
    }

    #[test]
    fn two_site_state_rejects_forbidden_pair_charge_and_bad_dense_boundary() {
        assert!(matches!(
            U1TwoSiteState::new(
                vec![0],
                vec![0],
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
                1.0e-12
            ),
            Err(U1MpsError::ChargeSelectionRule { .. })
        ));

        let non_open = U1TwoSiteState::new(
            vec![0, 1],
            vec![1],
            vec![Complex64::new(0.0, 0.0); 2 * 4],
            1.0e-12,
        )
        .unwrap();
        assert!(matches!(
            non_open.to_dense_two_qubit(),
            Err(U1MpsError::TwoSiteDenseExpansionRequiresOpenBoundaries)
        ));
    }


    fn complex_zero() -> Complex64 {
        Complex64::new(0.0, 0.0)
    }

    #[test]
    fn two_site_identity_and_swap_gates_preserve_charge_and_dense_amplitudes() {
        let product = ComplexU1Mps::product_state(&[false, true]).unwrap();
        let two_site =
            U1TwoSiteState::from_adjacent_tensors(&product.tensors()[0], &product.tensors()[1], 1.0e-12)
                .unwrap();
        assert_eq!(two_site.to_dense_two_qubit().unwrap(), [
            complex_zero(),
            Complex64::new(1.0, 0.0),
            complex_zero(),
            complex_zero(),
        ]);

        let identity = two_site.apply_gate(&U1TwoSiteGate::identity(), 1.0e-12).unwrap();
        assert_eq!(identity.to_dense_two_qubit().unwrap(), two_site.to_dense_two_qubit().unwrap());

        let swapped = two_site.apply_gate(&U1TwoSiteGate::swap(), 1.0e-12).unwrap();
        assert_eq!(swapped.to_dense_two_qubit().unwrap(), [
            complex_zero(),
            complex_zero(),
            Complex64::new(1.0, 0.0),
            complex_zero(),
        ]);
        assert_eq!(swapped.left_charges(), &[0]);
        assert_eq!(swapped.right_charges(), &[1]);
        assert_eq!(swapped.selection_rule_residual(), 0.0);
    }

    #[test]
    fn complex_u1_gate_applies_nontrivial_unitary_within_single_excitation_sector() {
        let inverse_root_two = 1.0 / 2.0_f64.sqrt();
        let mut matrix = [[complex_zero(); 4]; 4];
        matrix[0][0] = Complex64::new(0.0, 1.0);
        matrix[3][3] = Complex64::new(-1.0, 0.0);
        matrix[1][1] = Complex64::new(inverse_root_two, 0.0);
        matrix[1][2] = Complex64::new(0.0, inverse_root_two);
        matrix[2][1] = Complex64::new(0.0, inverse_root_two);
        matrix[2][2] = Complex64::new(inverse_root_two, 0.0);
        let gate = U1TwoSiteGate::new(matrix, 1.0e-12).unwrap();
        assert!(gate.charge_conservation_residual() <= 1.0e-12);
        assert!(gate.unitarity_residual() <= 1.0e-12);

        let plus = U1TwoSiteState::new(
            vec![0],
            vec![1],
            vec![
                complex_zero(),
                Complex64::new(inverse_root_two, 0.0),
                Complex64::new(inverse_root_two, 0.0),
                complex_zero(),
            ],
            1.0e-12,
        )
        .unwrap();
        let evolved = plus.apply_gate(&gate, 1.0e-12).unwrap();
        let dense = evolved.to_dense_two_qubit().unwrap();
        assert!(dense[0].norm() <= 1.0e-12);
        assert!((dense[1] - Complex64::new(inverse_root_two, inverse_root_two)).norm() <= 1.0e-12);
        assert!((dense[2] - Complex64::new(inverse_root_two, inverse_root_two)).norm() <= 1.0e-12);
        assert!(dense[3].norm() <= 1.0e-12);
    }

    #[test]
    fn two_site_gate_constructor_rejects_charge_leaking_and_nonunitary_matrices() {
        let mut charge_leaking = [[complex_zero(); 4]; 4];
        charge_leaking[0][1] = Complex64::new(1.0, 0.0);
        charge_leaking[1][0] = Complex64::new(1.0, 0.0);
        charge_leaking[2][2] = Complex64::new(1.0, 0.0);
        charge_leaking[3][3] = Complex64::new(1.0, 0.0);
        assert!(matches!(
            U1TwoSiteGate::new(charge_leaking, 1.0e-12),
            Err(U1MpsError::GateChargeLeakage { .. })
        ));

        let mut nonunitary = [[complex_zero(); 4]; 4];
        for index in 0..4 {
            nonunitary[index][index] = Complex64::new(1.0, 0.0);
        }
        nonunitary[2][2] = Complex64::new(2.0, 0.0);
        assert!(matches!(
            U1TwoSiteGate::new(nonunitary, 1.0e-12),
            Err(U1MpsError::GateUnitarityResidual { .. })
        ));
    }

    #[test]
    fn two_site_state_rejects_forbidden_charge_sector_and_nonfinite_values() {
        let forbidden = U1TwoSiteState::new(
            vec![0],
            vec![0],
            vec![
                complex_zero(),
                Complex64::new(0.2, 0.0),
                complex_zero(),
                complex_zero(),
            ],
            1.0e-12,
        );
        assert!(matches!(
            forbidden,
            Err(U1MpsError::ChargeSelectionRule {
                left_charge: 0,
                physical: 1,
                right_charge: 0,
                ..
            })
        ));

        let nonfinite = U1TwoSiteState::new(
            vec![0],
            vec![0],
            vec![
                Complex64::new(f64::NAN, 0.0),
                complex_zero(),
                complex_zero(),
                complex_zero(),
            ],
            1.0e-12,
        );
        assert!(matches!(
            nonfinite,
            Err(U1MpsError::NonFiniteTensorEntry { .. })
        ));
    }

    #[test]
    fn adjacent_tensor_contraction_rejects_mismatched_virtual_charge_bonds() {
        let left = U1MpsTensor::new(
            vec![0],
            vec![0],
            vec![Complex64::new(1.0, 0.0), complex_zero()],
        )
        .unwrap();
        let right = U1MpsTensor::new(
            vec![1],
            vec![1],
            vec![complex_zero(), Complex64::new(1.0, 0.0)],
        )
        .unwrap();
        assert_eq!(
            U1TwoSiteState::from_adjacent_tensors(&left, &right, 1.0e-12),
            Err(U1MpsError::TwoSiteBondMismatch)
        );
    }

}
