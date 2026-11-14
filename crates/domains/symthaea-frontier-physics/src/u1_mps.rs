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
    InvalidTotalCharge {
        total_charge: i32,
        site_count: usize,
    },
    ZeroDenseState,
    GateChargeLeakage {
        output_charge: u32,
        input_charge: u32,
        amplitude: f64,
        tolerance: f64,
    },
    GateUnitarityResidual {
        residual: f64,
        tolerance: f64,
    },
    GateChargeConservationResidual {
        residual: f64,
        tolerance: f64,
    },
    TwoSiteHamiltonianHermiticityResidual {
        residual: f64,
        tolerance: f64,
    },
    TwoSiteHamiltonianChargeLeakage {
        output_charge: u32,
        input_charge: u32,
        amplitude: f64,
        tolerance: f64,
    },
    TwoSiteBondMismatch,
    TwoSiteShapeMismatch,
    TwoSiteDenseExpansionRequiresOpenBoundaries,
    TwoSiteSvdFailed {
        rows: usize,
        columns: usize,
    },
    DenseChargeLeakage {
        basis_index: usize,
        amplitude: f64,
        tolerance: f64,
    },
    SvdFailed {
        rows: usize,
        columns: usize,
    },
    InvalidTruncationPolicy {
        max_bond_dimension: usize,
        relative_cutoff: f64,
        tolerance: f64,
    },
    InvalidErrorBudget {
        max_bond_dimension: usize,
        maximum_absolute_local_l2_error: f64,
        tolerance: f64,
    },
    TruncationBudgetUnreachable {
        requested_absolute_l2_error: f64,
        maximum_bond_dimension: usize,
        minimum_required_bond_dimension: usize,
        best_achievable_l2_error: f64,
    },
    InvalidTeBDParity(u8),
    TeBDGateOutOfRange {
        bond_index: usize,
        site_count: usize,
    },
    TeBDGateWrongParity {
        bond_index: usize,
        parity: u8,
    },
    DuplicateTeBDGate {
        bond_index: usize,
    },
    TeBDLayerSiteCountMismatch {
        layer_sites: usize,
        state_sites: usize,
    },
    TruncationLedgerStateMismatch,
    CanonicalGaugeNotQualified {
        residual: f64,
        tolerance: f64,
    },
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
            Self::InvalidTotalCharge {
                total_charge,
                site_count,
            } => write!(
                f,
                "total charge {total_charge} is invalid for a {site_count}-site qubit MPS"
            ),
            Self::ZeroDenseState => write!(
                f,
                "the dense input has no nonzero amplitude in the declared charge sector"
            ),
            Self::GateChargeLeakage {
                output_charge,
                input_charge,
                amplitude,
                tolerance,
            } => write!(
                f,
                "two-site gate couples charge {input_charge} to {output_charge} with amplitude {amplitude}, tolerance {tolerance}"
            ),
            Self::GateUnitarityResidual {
                residual,
                tolerance,
            } => write!(
                f,
                "two-site gate unitarity residual {residual} exceeds tolerance {tolerance}"
            ),
            Self::GateChargeConservationResidual {
                residual,
                tolerance,
            } => write!(
                f,
                "two-site gate charge-conservation residual {residual} exceeds tolerance {tolerance}"
            ),
            Self::TwoSiteHamiltonianHermiticityResidual { residual, tolerance } => write!(
                f,
                "two-site Hamiltonian Hermiticity residual {residual} exceeds tolerance {tolerance}"
            ),
            Self::TwoSiteHamiltonianChargeLeakage {
                output_charge,
                input_charge,
                amplitude,
                tolerance,
            } => write!(
                f,
                "two-site Hamiltonian couples excitation sector {input_charge} to {output_charge} with amplitude {amplitude}, tolerance {tolerance}"
            ),
            Self::TwoSiteBondMismatch => write!(
                f,
                "adjacent MPS tensors have different virtual charge labels"
            ),
            Self::TwoSiteShapeMismatch => {
                write!(f, "two-site state tensor dimensions are inconsistent")
            }
            Self::TwoSiteDenseExpansionRequiresOpenBoundaries => {
                write!(
                    f,
                    "two-site dense expansion requires one-dimensional external boundary bonds"
                )
            }
            Self::TwoSiteSvdFailed { rows, columns } => write!(
                f,
                "charge-block two-site SVD failed to converge for {rows}x{columns} block"
            ),
            Self::InvalidTruncationPolicy {
                max_bond_dimension,
                relative_cutoff,
                tolerance,
            } => write!(
                f,
                "invalid truncation policy: max_bond={max_bond_dimension}, relative_cutoff={relative_cutoff}, tolerance={tolerance}"
            ),
            Self::InvalidErrorBudget {
                max_bond_dimension,
                maximum_absolute_local_l2_error,
                tolerance,
            } => write!(
                f,
                "invalid absolute L2 error budget {maximum_absolute_local_l2_error} with max bond {max_bond_dimension} and tolerance {tolerance}"
            ),
            Self::TruncationBudgetUnreachable {
                requested_absolute_l2_error,
                maximum_bond_dimension,
                minimum_required_bond_dimension,
                best_achievable_l2_error,
            } => write!(
                f,
                "local L2 error budget {requested_absolute_l2_error} is unreachable with max bond {maximum_bond_dimension}; minimum Schmidt rank is {minimum_required_bond_dimension}, best achievable error is {best_achievable_l2_error}"
            ),
            Self::InvalidTeBDParity(parity) => {
                write!(f, "TEBD layer parity must be 0 or 1, got {parity}")
            }
            Self::TeBDGateOutOfRange {
                bond_index,
                site_count,
            } => write!(
                f,
                "TEBD gate bond {bond_index} is out of range for {site_count} sites"
            ),
            Self::TeBDGateWrongParity { bond_index, parity } => write!(
                f,
                "TEBD gate bond {bond_index} does not belong to parity-{parity} layer"
            ),
            Self::DuplicateTeBDGate { bond_index } => {
                write!(f, "TEBD layer contains duplicate gate on bond {bond_index}")
            }
            Self::TeBDLayerSiteCountMismatch {
                layer_sites,
                state_sites,
            } => write!(
                f,
                "TEBD layer was built for {layer_sites} sites, state has {state_sites}"
            ),
            Self::TruncationLedgerStateMismatch => write!(
                f,
                "truncation ledger is bound to a different input MPS state"
            ),
            Self::CanonicalGaugeNotQualified {
                residual,
                tolerance,
            } => write!(
                f,
                "canonical gauge residual {residual} exceeds qualification tolerance {tolerance}"
            ),
            Self::DenseChargeLeakage {
                basis_index,
                amplitude,
                tolerance,
            } => write!(
                f,
                "dense basis amplitude {basis_index} has forbidden charge-sector magnitude {amplitude}, tolerance {tolerance}"
            ),
            Self::SvdFailed { rows, columns } => {
                write!(
                    f,
                    "charge-block SVD failed to converge for {rows}x{columns} block"
                )
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

/// Measured mixed-canonical gauge receipt for a two-site center at a bond.
///
/// Sites [0,bond) are left-isometric and sites (bond+1,N) are right-isometric.
#[derive(Debug, Clone, PartialEq)]
pub struct U1CanonicalGaugeReceipt {
    bond: usize,
    site_count: usize,
    total_charge: i32,
    left_isometry_residuals: Vec<(usize, f64)>,
    right_isometry_residuals: Vec<(usize, f64)>,
    maximum_left_isometry_residual: f64,
    maximum_right_isometry_residual: f64,
    left_exterior_operator_norm_upper_bound: f64,
    right_exterior_operator_norm_upper_bound: f64,
    factorization_max_residual: f64,
    factorization_l2_residual: f64,
    state_reconstruction_l2_residual: f64,
    exact_zero_singular_values_omitted: usize,
    selection_rule_residual: f64,
    input_norm_squared: f64,
    output_norm_squared: f64,
    relative_norm_drift: f64,
    qualified: bool,
}

impl U1CanonicalGaugeReceipt {
    fn required_left_exterior_bound(&self) -> f64 {
        self.left_isometry_residuals
            .iter()
            .fold(1.0, |bound, (_, residual)| bound * (1.0 + residual).sqrt())
    }

    fn required_right_exterior_bound(&self) -> f64 {
        self.right_isometry_residuals
            .iter()
            .fold(1.0, |bound, (_, residual)| bound * (1.0 + residual).sqrt())
    }

    pub fn bond(&self) -> usize { self.bond }
    pub fn site_count(&self) -> usize { self.site_count }
    pub fn total_charge(&self) -> i32 { self.total_charge }
    pub fn left_isometry_residuals(&self) -> &[(usize, f64)] { &self.left_isometry_residuals }
    pub fn right_isometry_residuals(&self) -> &[(usize, f64)] { &self.right_isometry_residuals }
    pub fn maximum_left_isometry_residual(&self) -> f64 { self.maximum_left_isometry_residual }
    pub fn maximum_right_isometry_residual(&self) -> f64 { self.maximum_right_isometry_residual }
    pub fn left_exterior_operator_norm_upper_bound(&self) -> f64 {
        self.left_exterior_operator_norm_upper_bound
    }
    pub fn right_exterior_operator_norm_upper_bound(&self) -> f64 {
        self.right_exterior_operator_norm_upper_bound
    }
    pub fn factorization_max_residual(&self) -> f64 { self.factorization_max_residual }
    pub fn factorization_l2_residual(&self) -> f64 { self.factorization_l2_residual }
    pub fn state_reconstruction_l2_residual(&self) -> f64 {
        self.state_reconstruction_l2_residual
    }
    pub fn exact_zero_singular_values_omitted(&self) -> usize {
        self.exact_zero_singular_values_omitted
    }
    pub fn selection_rule_residual(&self) -> f64 { self.selection_rule_residual }
    pub fn input_norm_squared(&self) -> f64 { self.input_norm_squared }
    pub fn output_norm_squared(&self) -> f64 { self.output_norm_squared }
    pub fn relative_norm_drift(&self) -> f64 { self.relative_norm_drift }
    pub fn is_qualified(&self) -> bool { self.qualified }

    pub fn qualifies(&self, tolerance: f64) -> bool {
        tolerance.is_finite()
            && tolerance > 0.0
            && self.site_count >= 2
            && self.bond < self.site_count - 1
            && self.qualified
            && self.maximum_left_isometry_residual.is_finite()
            && self.maximum_left_isometry_residual >= 0.0
            && self.maximum_left_isometry_residual <= tolerance
            && self.maximum_right_isometry_residual.is_finite()
            && self.maximum_right_isometry_residual >= 0.0
            && self.maximum_right_isometry_residual <= tolerance
            && self.factorization_max_residual.is_finite()
            && self.factorization_max_residual >= 0.0
            && self.factorization_max_residual <= tolerance
            && self.factorization_l2_residual.is_finite()
            && self.factorization_l2_residual >= 0.0
            && self.factorization_l2_residual <= tolerance
            && self.state_reconstruction_l2_residual.is_finite()
            && self.state_reconstruction_l2_residual >= 0.0
            && self.state_reconstruction_l2_residual <= tolerance
            && self.relative_norm_drift.is_finite()
            && self.relative_norm_drift >= 0.0
            && self.relative_norm_drift <= tolerance
            && self.selection_rule_residual.is_finite()
            && self.selection_rule_residual >= 0.0
            && self.selection_rule_residual <= tolerance
            && self.left_exterior_operator_norm_upper_bound.is_finite()
            && self.left_exterior_operator_norm_upper_bound
                + 8.0 * f64::EPSILON * self.required_left_exterior_bound().max(1.0)
                >= self.required_left_exterior_bound()
            && self.right_exterior_operator_norm_upper_bound.is_finite()
            && self.right_exterior_operator_norm_upper_bound
                + 8.0 * f64::EPSILON * self.required_right_exterior_bound().max(1.0)
                >= self.required_right_exterior_bound()
            && self.exterior_operator_norm_product_upper_bound().is_finite()
            && self.left_isometry_residuals.len() == self.bond
            && self
                .left_isometry_residuals
                .iter()
                .enumerate()
                .all(|(index, (site, residual))| {
                    *site == index
                        && residual.is_finite()
                        && *residual >= 0.0
                        && *residual <= tolerance
                })
            && self.right_isometry_residuals.len()
                == self.site_count.saturating_sub(self.bond + 2)
            && self
                .right_isometry_residuals
                .iter()
                .enumerate()
                .all(|(index, (site, residual))| {
                    *site == self.bond + 2 + index
                        && residual.is_finite()
                        && *residual >= 0.0
                        && *residual <= tolerance
                })
            && self.input_norm_squared.is_finite()
            && self.input_norm_squared > 0.0
            && self.output_norm_squared.is_finite()
            && self.output_norm_squared > 0.0
    }

    pub fn exterior_operator_norm_product_upper_bound(&self) -> f64 {
        self.left_exterior_operator_norm_upper_bound * self.right_exterior_operator_norm_upper_bound
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct CanonicalizationStats {
    max_residual: f64,
    residual_squared: f64,
    exact_zero_singular_values_omitted: usize,
}

fn left_isometry_residual(tensor: &U1MpsTensor) -> Result<f64, U1MpsError> {
    let dimension = tensor.right_dimension();
    let mut squared = 0.0;
    for row in 0..dimension {
        for column in 0..dimension {
            let mut gram = Complex64::new(0.0, 0.0);
            for left in 0..tensor.left_dimension() {
                for physical in 0..2 {
                    gram +=
                        tensor.get(left, physical, row).conj() * tensor.get(left, physical, column);
                }
            }
            let expected = if row == column {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            squared += (gram - expected).norm_sqr();
        }
    }
    let residual = squared.sqrt();
    if !residual.is_finite() {
        return Err(U1MpsError::NonFiniteContraction(
            "left-canonical isometry residual",
        ));
    }
    Ok(residual)
}

fn right_isometry_residual(tensor: &U1MpsTensor) -> Result<f64, U1MpsError> {
    let dimension = tensor.left_dimension();
    let mut squared = 0.0;
    for row in 0..dimension {
        for column in 0..dimension {
            let mut gram = Complex64::new(0.0, 0.0);
            for physical in 0..2 {
                for right in 0..tensor.right_dimension() {
                    gram += tensor.get(row, physical, right)
                        * tensor.get(column, physical, right).conj();
                }
            }
            let expected = if row == column {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            squared += (gram - expected).norm_sqr();
        }
    }
    let residual = squared.sqrt();
    if !residual.is_finite() {
        return Err(U1MpsError::NonFiniteContraction(
            "right-canonical isometry residual",
        ));
    }
    Ok(residual)
}

fn left_canonicalize_site(
    tensors: &mut [U1MpsTensor],
    site: usize,
    tolerance: f64,
) -> Result<CanonicalizationStats, U1MpsError> {
    if site + 1 >= tensors.len() {
        return Err(U1MpsError::TwoSiteShapeMismatch);
    }
    if tensors[site].right_charges != tensors[site + 1].left_charges {
        return Err(U1MpsError::TwoSiteBondMismatch);
    }

    struct Block {
        charge: i32,
        rows: Vec<(usize, usize)>,
        columns: Vec<usize>,
        u: DMatrix<Complex64>,
        v_t: DMatrix<Complex64>,
        singular_values: Vec<f64>,
    }

    let current = &tensors[site];
    let next = &tensors[site + 1];
    let old_right_dimension = current.right_dimension();
    let mut row_groups: Vec<(i32, Vec<(usize, usize)>)> = Vec::new();
    for left in 0..current.left_dimension() {
        for physical in 0..2 {
            let charge = current.left_charges[left] + physical as i32;
            if let Some((_, rows)) = row_groups.iter_mut().find(|(q, _)| *q == charge) {
                rows.push((left, physical));
            } else {
                row_groups.push((charge, vec![(left, physical)]));
            }
        }
    }

    let max_dimension = current
        .left_dimension()
        .saturating_mul(2)
        .max(old_right_dimension);
    let max_iterations =
        10_000usize.max(64usize.saturating_mul(max_dimension.saturating_mul(max_dimension)));
    let mut blocks = Vec::new();
    let mut stats = CanonicalizationStats::default();

    for (charge, rows) in row_groups {
        let columns = current
            .right_charges
            .iter()
            .enumerate()
            .filter_map(|(index, candidate)| (*candidate == charge).then_some(index))
            .collect::<Vec<_>>();
        if rows.is_empty() || columns.is_empty() {
            continue;
        }
        let mut block = DMatrix::<Complex64>::zeros(rows.len(), columns.len());
        for (row_index, &(left, physical)) in rows.iter().enumerate() {
            for (column_index, &right) in columns.iter().enumerate() {
                block[(row_index, column_index)] = current.get(left, physical, right);
            }
        }

        let row_count = block.nrows();
        let column_count = block.ncols();
        let decomposition = block
            .clone()
            .try_svd(true, true, tolerance, max_iterations)
            .ok_or(U1MpsError::SvdFailed {
                rows: row_count,
                columns: column_count,
            })?;
        let u = decomposition.u.ok_or(U1MpsError::SvdFailed {
            rows: row_count,
            columns: column_count,
        })?;
        let v_t = decomposition.v_t.ok_or(U1MpsError::SvdFailed {
            rows: row_count,
            columns: column_count,
        })?;

        let mut kept = Vec::new();
        for (index, value) in decomposition.singular_values.iter().enumerate() {
            if !value.is_finite() {
                return Err(U1MpsError::NonFiniteContraction(
                    "left-canonical singular value",
                ));
            }
            if *value > 0.0 {
                kept.push(index);
            } else {
                stats.exact_zero_singular_values_omitted += 1;
            }
        }
        if kept.is_empty() {
            continue;
        }

        let kept_u = DMatrix::<Complex64>::from_fn(rows.len(), kept.len(), |row, column| {
            u[(row, kept[column])]
        });
        let kept_v_t = DMatrix::<Complex64>::from_fn(kept.len(), columns.len(), |row, column| {
            v_t[(kept[row], column)]
        });
        let singular_values = kept
            .iter()
            .map(|&index| decomposition.singular_values[index])
            .collect::<Vec<_>>();

        for row in 0..rows.len() {
            for column in 0..columns.len() {
                let mut reconstructed = Complex64::new(0.0, 0.0);
                for mode in 0..kept.len() {
                    reconstructed +=
                        kept_u[(row, mode)] * singular_values[mode] * kept_v_t[(mode, column)];
                }
                let residual = (reconstructed - block[(row, column)]).norm();
                stats.max_residual = stats.max_residual.max(residual);
                stats.residual_squared += residual * residual;
            }
        }
        blocks.push(Block {
            charge,
            rows,
            columns,
            u: kept_u,
            v_t: kept_v_t,
            singular_values,
        });
    }

    let new_dimension = blocks.iter().try_fold(0_usize, |sum, block| {
        sum.checked_add(block.singular_values.len())
            .ok_or(U1MpsError::DimensionOverflow)
    })?;
    if new_dimension == 0 {
        return Err(U1MpsError::ZeroDenseState);
    }
    let current_left_dimension = current.left_dimension();
    let next_right_dimension = next.right_dimension();
    let current_entries = current_left_dimension
        .checked_mul(2)
        .and_then(|value| value.checked_mul(new_dimension))
        .ok_or(U1MpsError::DimensionOverflow)?;
    let transfer_entries = new_dimension
        .checked_mul(old_right_dimension)
        .ok_or(U1MpsError::DimensionOverflow)?;
    let next_entries = new_dimension
        .checked_mul(2)
        .and_then(|value| value.checked_mul(next_right_dimension))
        .ok_or(U1MpsError::DimensionOverflow)?;
    let mut current_data = vec![Complex64::new(0.0, 0.0); current_entries];
    let mut transfer_data = vec![Complex64::new(0.0, 0.0); transfer_entries];
    let mut next_charges = Vec::with_capacity(new_dimension);
    let mut offset = 0;

    for block in blocks {
        let rank = block.singular_values.len();
        next_charges.extend(core::iter::repeat_n(block.charge, rank));
        for mode in 0..rank {
            let new_index = offset + mode;
            for (row_index, &(left, physical)) in block.rows.iter().enumerate() {
                let index = (left * 2 + physical) * new_dimension + new_index;
                current_data[index] = block.u[(row_index, mode)];
            }
            for (column_index, &old_right) in block.columns.iter().enumerate() {
                let index = new_index * old_right_dimension + old_right;
                transfer_data[index] =
                    block.singular_values[mode] * block.v_t[(mode, column_index)];
            }
        }
        offset += rank;
    }

    let mut next_data = vec![Complex64::new(0.0, 0.0); next_entries];
    for new_left in 0..new_dimension {
        for physical in 0..2 {
            for right in 0..next_right_dimension {
                let mut value = Complex64::new(0.0, 0.0);
                for old_left in 0..old_right_dimension {
                    value += transfer_data[new_left * old_right_dimension + old_left]
                        * next.get(old_left, physical, right);
                }
                let index = (new_left * 2 + physical) * next_right_dimension + right;
                next_data[index] = value;
            }
        }
    }

    let new_current = U1MpsTensor::new(
        current.left_charges.clone(),
        next_charges.clone(),
        current_data,
    )?;
    let new_next = U1MpsTensor::new(next_charges, next.right_charges.clone(), next_data)?;
    tensors[site] = new_current;
    tensors[site + 1] = new_next;
    Ok(stats)
}

fn right_canonicalize_site(
    tensors: &mut [U1MpsTensor],
    site: usize,
    tolerance: f64,
) -> Result<CanonicalizationStats, U1MpsError> {
    if site == 0 || site >= tensors.len() {
        return Err(U1MpsError::TwoSiteShapeMismatch);
    }
    if tensors[site - 1].right_charges != tensors[site].left_charges {
        return Err(U1MpsError::TwoSiteBondMismatch);
    }

    struct Block {
        charge: i32,
        rows: Vec<usize>,
        columns: Vec<(usize, usize)>,
        u: DMatrix<Complex64>,
        v_t: DMatrix<Complex64>,
        singular_values: Vec<f64>,
    }

    let previous = &tensors[site - 1];
    let current = &tensors[site];
    let old_left_dimension = current.left_dimension();
    let mut row_groups: Vec<(i32, Vec<usize>)> = Vec::new();
    for left in 0..old_left_dimension {
        let charge = current.left_charges[left];
        if let Some((_, rows)) = row_groups.iter_mut().find(|(q, _)| *q == charge) {
            rows.push(left);
        } else {
            row_groups.push((charge, vec![left]));
        }
    }

    let max_dimension = old_left_dimension.max(current.right_dimension().saturating_mul(2));
    let max_iterations =
        10_000usize.max(64usize.saturating_mul(max_dimension.saturating_mul(max_dimension)));
    let mut blocks = Vec::new();
    let mut stats = CanonicalizationStats::default();

    for (charge, rows) in row_groups {
        let mut columns = Vec::new();
        for physical in 0..2 {
            for right in 0..current.right_dimension() {
                if current.right_charges[right] == charge + physical as i32 {
                    columns.push((physical, right));
                }
            }
        }
        if rows.is_empty() || columns.is_empty() {
            continue;
        }

        let mut block = DMatrix::<Complex64>::zeros(rows.len(), columns.len());
        for (row_index, &left) in rows.iter().enumerate() {
            for (column_index, &(physical, right)) in columns.iter().enumerate() {
                block[(row_index, column_index)] = current.get(left, physical, right);
            }
        }

        let row_count = block.nrows();
        let column_count = block.ncols();
        let decomposition = block
            .clone()
            .try_svd(true, true, tolerance, max_iterations)
            .ok_or(U1MpsError::SvdFailed {
                rows: row_count,
                columns: column_count,
            })?;
        let u = decomposition.u.ok_or(U1MpsError::SvdFailed {
            rows: row_count,
            columns: column_count,
        })?;
        let v_t = decomposition.v_t.ok_or(U1MpsError::SvdFailed {
            rows: row_count,
            columns: column_count,
        })?;

        let mut kept = Vec::new();
        for (index, value) in decomposition.singular_values.iter().enumerate() {
            if !value.is_finite() {
                return Err(U1MpsError::NonFiniteContraction(
                    "right-canonical singular value",
                ));
            }
            if *value > 0.0 {
                kept.push(index);
            } else {
                stats.exact_zero_singular_values_omitted += 1;
            }
        }
        if kept.is_empty() {
            continue;
        }

        let kept_u = DMatrix::<Complex64>::from_fn(rows.len(), kept.len(), |row, column| {
            u[(row, kept[column])]
        });
        let kept_v_t = DMatrix::<Complex64>::from_fn(kept.len(), columns.len(), |row, column| {
            v_t[(kept[row], column)]
        });
        let singular_values = kept
            .iter()
            .map(|&index| decomposition.singular_values[index])
            .collect::<Vec<_>>();

        for row in 0..rows.len() {
            for column in 0..columns.len() {
                let mut reconstructed = Complex64::new(0.0, 0.0);
                for mode in 0..kept.len() {
                    reconstructed +=
                        kept_u[(row, mode)] * singular_values[mode] * kept_v_t[(mode, column)];
                }
                let residual = (reconstructed - block[(row, column)]).norm();
                stats.max_residual = stats.max_residual.max(residual);
                stats.residual_squared += residual * residual;
            }
        }

        blocks.push(Block {
            charge,
            rows,
            columns,
            u: kept_u,
            v_t: kept_v_t,
            singular_values,
        });
    }

    let new_dimension = blocks.iter().try_fold(0_usize, |sum, block| {
        sum.checked_add(block.singular_values.len())
            .ok_or(U1MpsError::DimensionOverflow)
    })?;
    if new_dimension == 0 {
        return Err(U1MpsError::ZeroDenseState);
    }

    let previous_left_dimension = previous.left_dimension();
    let current_right_dimension = current.right_dimension();
    let previous_entries = previous_left_dimension
        .checked_mul(2)
        .and_then(|value| value.checked_mul(new_dimension))
        .ok_or(U1MpsError::DimensionOverflow)?;
    let current_entries = new_dimension
        .checked_mul(2)
        .and_then(|value| value.checked_mul(current_right_dimension))
        .ok_or(U1MpsError::DimensionOverflow)?;
    let mut previous_data = vec![Complex64::new(0.0, 0.0); previous_entries];
    let mut current_data = vec![Complex64::new(0.0, 0.0); current_entries];
    let mut new_left_charges = Vec::with_capacity(new_dimension);
    let mut offset = 0;

    for block in blocks {
        let rank = block.singular_values.len();
        new_left_charges.extend(core::iter::repeat_n(block.charge, rank));
        for mode in 0..rank {
            let new_left = offset + mode;
            for (column_index, &(physical, right)) in block.columns.iter().enumerate() {
                let index = (new_left * 2 + physical) * current_right_dimension + right;
                current_data[index] = block.v_t[(mode, column_index)];
            }
            for (row_index, &old_left) in block.rows.iter().enumerate() {
                let transfer = block.u[(row_index, mode)] * block.singular_values[mode];
                for previous_left in 0..previous_left_dimension {
                    for previous_physical in 0..2 {
                        let index =
                            (previous_left * 2 + previous_physical) * new_dimension + new_left;
                        previous_data[index] +=
                            previous.get(previous_left, previous_physical, old_left) * transfer;
                    }
                }
            }
        }
        offset += rank;
    }

    let new_previous = U1MpsTensor::new(
        previous.left_charges.clone(),
        new_left_charges.clone(),
        previous_data,
    )?;
    let new_current = U1MpsTensor::new(
        new_left_charges,
        current.right_charges.clone(),
        current_data,
    )?;
    tensors[site - 1] = new_previous;
    tensors[site] = new_current;
    Ok(stats)
}

impl ComplexU1Mps {
    fn state_fingerprint(&self) -> u128 {
        // FNV-1a 128-bit fingerprint over dimensions, charge labels and exact
        // IEEE-754 tensor bits. This binds a continuation ledger to the exact
        // MPS value without retaining a second full state in memory.
        // It is an integrity/misuse check, not a cryptographic authenticator.
        let mut hash = 0x6c62272e07bb014262b821756295c58du128;
        const PRIME: u128 = 0x0000000001000000000000000000013b;
        let mut mix = |value: u64| {
            for byte in value.to_le_bytes() {
                hash ^= u128::from(byte);
                hash = hash.wrapping_mul(PRIME);
            }
        };
        mix(self.total_charge as u64);
        mix(self.tensors.len() as u64);
        for tensor in &self.tensors {
            mix(tensor.left_dimension() as u64);
            mix(tensor.right_dimension() as u64);
            for charge in &tensor.left_charges { mix(*charge as u64); }
            for charge in &tensor.right_charges { mix(*charge as u64); }
            for value in &tensor.data {
                mix(value.re.to_bits());
                mix(value.im.to_bits());
            }
        }
        hash
    }

    /// Stable norm of the difference between two same-sector MPS.
    ///
    /// The two networks are embedded into a direct-sum virtual space. The
    /// subtraction is applied at the first site, then the two paths propagate
    /// in block-diagonal form and are recombined only at the right boundary.
    /// This computes ||A-B||² without subtracting nearly equal O(1) norm and
    /// overlap scalars.
    fn difference_norm_squared(&self, other: &Self, tolerance: f64) -> Result<f64, U1MpsError> {
        if self.site_count() != other.site_count() || self.total_charge != other.total_charge {
            return Err(U1MpsError::TwoSiteShapeMismatch);
        }
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(U1MpsError::InvalidTolerance(tolerance));
        }

        let site_count = self.site_count();
        let mut difference_tensors = Vec::with_capacity(site_count);

        if site_count == 1 {
            let left = &self.tensors[0];
            let right = &other.tensors[0];
            if left.left_charges != right.left_charges || left.right_charges != right.right_charges
            {
                return Err(U1MpsError::TwoSiteBondMismatch);
            }
            let data = (0..left.left_dimension())
                .flat_map(|left_index| {
                    (0..2).flat_map(move |physical| {
                        (0..left.right_dimension()).map(move |right_index| {
                            left.get(left_index, physical, right_index)
                                - right.get(left_index, physical, right_index)
                        })
                    })
                })
                .collect();
            difference_tensors.push(U1MpsTensor::new(
                left.left_charges.clone(),
                left.right_charges.clone(),
                data,
            )?);
        } else {
            // First site: share the charge-zero left boundary and branch into
            // the A and -B right virtual spaces.
            let first_a = &self.tensors[0];
            let first_b = &other.tensors[0];
            let first_right_dimension = first_a
                .right_dimension()
                .checked_add(first_b.right_dimension())
                .ok_or(U1MpsError::DimensionOverflow)?;
            let first_entries = 2_usize
                .checked_mul(first_right_dimension)
                .ok_or(U1MpsError::DimensionOverflow)?;
            let mut first_data = vec![Complex64::new(0.0, 0.0); first_entries];
            for physical in 0..2 {
                for right in 0..first_a.right_dimension() {
                    first_data[physical * first_right_dimension + right] =
                        first_a.get(0, physical, right);
                }
                for right in 0..first_b.right_dimension() {
                    first_data
                        [physical * first_right_dimension + first_a.right_dimension() + right] =
                        -first_b.get(0, physical, right);
                }
            }
            let mut first_right_charges = first_a.right_charges.clone();
            first_right_charges.extend_from_slice(&first_b.right_charges);
            difference_tensors.push(U1MpsTensor::new(
                first_a.left_charges.clone(),
                first_right_charges,
                first_data,
            )?);

            // Middle sites: propagate A and B in separate diagonal blocks.
            for site in 1..(site_count - 1) {
                let tensor_a = &self.tensors[site];
                let tensor_b = &other.tensors[site];
                let left_dimension = tensor_a
                    .left_dimension()
                    .checked_add(tensor_b.left_dimension())
                    .ok_or(U1MpsError::DimensionOverflow)?;
                let right_dimension = tensor_a
                    .right_dimension()
                    .checked_add(tensor_b.right_dimension())
                    .ok_or(U1MpsError::DimensionOverflow)?;
                let entries = left_dimension
                    .checked_mul(2)
                    .and_then(|value| value.checked_mul(right_dimension))
                    .ok_or(U1MpsError::DimensionOverflow)?;
                let mut data = vec![Complex64::new(0.0, 0.0); entries];

                for left in 0..tensor_a.left_dimension() {
                    for physical in 0..2 {
                        for right in 0..tensor_a.right_dimension() {
                            let index = (left * 2 + physical) * right_dimension + right;
                            data[index] = tensor_a.get(left, physical, right);
                        }
                    }
                }
                for left in 0..tensor_b.left_dimension() {
                    for physical in 0..2 {
                        for right in 0..tensor_b.right_dimension() {
                            let index = ((tensor_a.left_dimension() + left) * 2 + physical)
                                * right_dimension
                                + tensor_a.right_dimension()
                                + right;
                            data[index] = tensor_b.get(left, physical, right);
                        }
                    }
                }

                let mut left_charges = tensor_a.left_charges.clone();
                left_charges.extend_from_slice(&tensor_b.left_charges);
                let mut right_charges = tensor_a.right_charges.clone();
                right_charges.extend_from_slice(&tensor_b.right_charges);
                difference_tensors.push(U1MpsTensor::new(left_charges, right_charges, data)?);
            }

            // Final site: add the two branch amplitudes at the common
            // total-charge right boundary.
            let final_a = self.tensors.last().expect("nonempty MPS");
            let final_b = other.tensors.last().expect("nonempty MPS");
            let final_left_dimension = final_a
                .left_dimension()
                .checked_add(final_b.left_dimension())
                .ok_or(U1MpsError::DimensionOverflow)?;
            let final_entries = final_left_dimension
                .checked_mul(2)
                .ok_or(U1MpsError::DimensionOverflow)?;
            let mut final_data = vec![Complex64::new(0.0, 0.0); final_entries];
            for left in 0..final_a.left_dimension() {
                for physical in 0..2 {
                    final_data[left * 2 + physical] = final_a.get(left, physical, 0);
                }
            }
            for left in 0..final_b.left_dimension() {
                for physical in 0..2 {
                    final_data[(final_a.left_dimension() + left) * 2 + physical] =
                        final_b.get(left, physical, 0);
                }
            }
            let mut final_left_charges = final_a.left_charges.clone();
            final_left_charges.extend_from_slice(&final_b.left_charges);
            difference_tensors.push(U1MpsTensor::new(
                final_left_charges,
                final_a.right_charges.clone(),
                final_data,
            )?);
        }

        let difference = Self::new(difference_tensors, self.total_charge, tolerance)?;
        difference.norm_squared(tolerance)
    }

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

    /// Move the MPS into mixed-canonical form around a two-site center.
    ///
    /// Sites before the bond are left-isometric; sites after the bond plus one
    /// are right-isometric. All factorizations are charge-blocked and retain
    /// every positive singular value. The center pair remains available for a
    /// local two-site operation.
    pub fn canonicalize_for_bond(
        &mut self,
        bond: usize,
        tolerance: f64,
    ) -> Result<U1CanonicalGaugeReceipt, U1MpsError> {
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(U1MpsError::InvalidTolerance(tolerance));
        }
        if self.site_count() < 2 || bond >= self.site_count() - 1 {
            return Err(U1MpsError::TwoSiteShapeMismatch);
        }

        let input_norm_squared = self.norm_squared(tolerance)?;
        if input_norm_squared <= 0.0 {
            return Err(U1MpsError::ZeroDenseState);
        }
        let mut tensors = self.tensors.clone();
        let mut factorization_stats = CanonicalizationStats::default();

        for site in 0..bond {
            let stats = left_canonicalize_site(&mut tensors, site, tolerance)?;
            factorization_stats.max_residual =
                factorization_stats.max_residual.max(stats.max_residual);
            factorization_stats.residual_squared += stats.residual_squared;
            factorization_stats.exact_zero_singular_values_omitted +=
                stats.exact_zero_singular_values_omitted;
        }

        for site in ((bond + 2)..self.site_count()).rev() {
            let stats = right_canonicalize_site(&mut tensors, site, tolerance)?;
            factorization_stats.max_residual =
                factorization_stats.max_residual.max(stats.max_residual);
            factorization_stats.residual_squared += stats.residual_squared;
            factorization_stats.exact_zero_singular_values_omitted +=
                stats.exact_zero_singular_values_omitted;
        }

        // Validate a candidate first. On any failure the original MPS is unchanged.
        let candidate = Self::new(tensors, self.total_charge, tolerance)?;
        let output_norm_squared = candidate.norm_squared(tolerance)?;
        let difference_squared = self.difference_norm_squared(&candidate, tolerance)?;
        if difference_squared < 0.0 {
            return Err(U1MpsError::NegativeNormSquared(difference_squared));
        }
        let state_reconstruction_l2_residual = difference_squared.sqrt();
        // This is relative to the input norm even for subnormalized states.
        // A max(..., 1) denominator would silently turn it into an absolute
        // criterion for physically small states.
        let relative_norm_drift =
            (output_norm_squared - input_norm_squared).abs() / input_norm_squared;

        let mut left_isometry_residuals = Vec::with_capacity(bond);
        let mut right_isometry_residuals =
            Vec::with_capacity(self.site_count().saturating_sub(bond + 2));
        let mut maximum_left_isometry_residual: f64 = 0.0;
        let mut maximum_right_isometry_residual: f64 = 0.0;
        let mut left_exterior_operator_norm_upper_bound = 1.0;
        let mut right_exterior_operator_norm_upper_bound = 1.0;

        for site in 0..bond {
            let residual = left_isometry_residual(&candidate.tensors[site])?;
            maximum_left_isometry_residual = maximum_left_isometry_residual.max(residual);
            left_exterior_operator_norm_upper_bound *= (1.0 + residual).sqrt();
            left_isometry_residuals.push((site, residual));
        }
        for site in ((bond + 2)..self.site_count()).rev() {
            let residual = right_isometry_residual(&candidate.tensors[site])?;
            maximum_right_isometry_residual = maximum_right_isometry_residual.max(residual);
            right_exterior_operator_norm_upper_bound *= (1.0 + residual).sqrt();
            right_isometry_residuals.push((site, residual));
        }
        right_isometry_residuals.sort_by_key(|(site, _)| *site);

        let factorization_l2_residual = factorization_stats.residual_squared.sqrt();
        let qualified = maximum_left_isometry_residual <= tolerance
            && maximum_right_isometry_residual <= tolerance
            && factorization_stats.max_residual <= tolerance
            && factorization_l2_residual <= tolerance
            && state_reconstruction_l2_residual <= tolerance
            && relative_norm_drift <= tolerance
            && candidate.selection_rule_residual <= tolerance
            && left_exterior_operator_norm_upper_bound.is_finite()
            && right_exterior_operator_norm_upper_bound.is_finite();

        let receipt = U1CanonicalGaugeReceipt {
            bond,
            site_count: self.site_count(),
            total_charge: self.total_charge,
            left_isometry_residuals,
            right_isometry_residuals,
            maximum_left_isometry_residual,
            maximum_right_isometry_residual,
            left_exterior_operator_norm_upper_bound,
            right_exterior_operator_norm_upper_bound,
            factorization_max_residual: factorization_stats.max_residual,
            factorization_l2_residual,
            state_reconstruction_l2_residual,
            exact_zero_singular_values_omitted: factorization_stats
                .exact_zero_singular_values_omitted,
            selection_rule_residual: candidate.selection_rule_residual,
            input_norm_squared,
            output_norm_squared,
            relative_norm_drift,
            qualified,
        };
        *self = candidate;
        Ok(receipt)
    }

    /// Mixed-canonicalize the MPS around a two-site center at (bond, bond+1).
    ///
    /// Sites before bond are left-isometric; sites after bond+1 are
    /// right-isometric. The active two-site center is not canonicalized.
    /// Charge-block SVD sweeps retain every positive singular value and omit
    /// exact zeros only.
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
                let decomposition = block.try_svd(true, true, tolerance, max_iterations).ok_or(
                    U1MpsError::SvdFailed {
                        rows: block_rows,
                        columns: block_columns,
                    },
                )?;
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
                let kept_u =
                    DMatrix::<Complex64>::from_fn(rows.len(), kept_indices.len(), |r, col| {
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
                        let index = (left * 2 + physical) * next_bond_dimension + new_bond;
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

    /// Contract the overlap <self|other> without dense expansion.
    pub fn inner_product(&self, other: &Self) -> Result<Complex64, U1MpsError> {
        if self.site_count() != other.site_count() {
            return Err(U1MpsError::TwoSiteShapeMismatch);
        }
        let mut environment = vec![Complex64::new(1.0, 0.0)];
        let mut bra_dimension = 1_usize;
        let mut ket_dimension = 1_usize;

        for (bra_tensor, ket_tensor) in self.tensors.iter().zip(other.tensors.iter()) {
            if bra_dimension != bra_tensor.left_dimension()
                || ket_dimension != ket_tensor.left_dimension()
            {
                return Err(U1MpsError::NonFiniteContraction(
                    "inconsistent overlap transfer-environment dimension",
                ));
            }
            let next_bra_dimension = bra_tensor.right_dimension();
            let next_ket_dimension = ket_tensor.right_dimension();
            let next_entries = next_bra_dimension
                .checked_mul(next_ket_dimension)
                .ok_or(U1MpsError::DimensionOverflow)?;
            let mut next = vec![Complex64::new(0.0, 0.0); next_entries];

            for bra_left in 0..bra_dimension {
                for ket_left in 0..ket_dimension {
                    let env = environment[bra_left * ket_dimension + ket_left];
                    for physical in 0..2 {
                        for bra_right in 0..next_bra_dimension {
                            let bra = bra_tensor.get(bra_left, physical, bra_right).conj();
                            for ket_right in 0..next_ket_dimension {
                                let ket = ket_tensor.get(ket_left, physical, ket_right);
                                next[bra_right * next_ket_dimension + ket_right] += env * bra * ket;
                            }
                        }
                    }
                }
            }
            if next
                .iter()
                .any(|value| !value.re.is_finite() || !value.im.is_finite())
            {
                return Err(U1MpsError::NonFiniteContraction(
                    "overlap transfer environment",
                ));
            }
            environment = next;
            bra_dimension = next_bra_dimension;
            ket_dimension = next_ket_dimension;
        }

        if bra_dimension != 1 || ket_dimension != 1 || environment.len() != 1 {
            return Err(U1MpsError::TwoSiteShapeMismatch);
        }
        let overlap = environment[0];
        if !overlap.re.is_finite() || !overlap.im.is_finite() {
            return Err(U1MpsError::NonFiniteContraction("MPS overlap"));
        }
        Ok(overlap)
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
    /// Operator-norm bound for replacing a tolerance-qualified source
    /// Hamiltonian by its exact Hermitian, U(1)-block-diagonal projection.
    hamiltonian_projection_error_operator_upper_bound: f64,
}



/// Hermitian nearest-neighbour two-qubit Hamiltonian restricted to U(1) sectors.
///
/// Basis order is |00>, |01>, |10>, |11>. The only coupled block is the
/// single-excitation sector {|01>,|10>}; zero- and two-excitation sectors
/// contribute independent real energies.
#[derive(Debug, Clone, PartialEq)]
pub struct U1TwoSiteHamiltonian {
    matrix: [[Complex64; 4]; 4],
    hermiticity_residual: f64,
    charge_conservation_residual: f64,
    /// Frobenius residual from the closest Hermitian block-diagonal U(1) model.
    projection_residual_frobenius: f64,
}

impl U1TwoSiteHamiltonian {
    /// Construct a finite Hermitian two-site Hamiltonian with zero U(1) flux.
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

        let mut hermiticity_residual: f64 = 0.0;
        let mut charge_conservation_residual: f64 = 0.0;
        let mut charge_witness = (0_usize, 0_usize);
        for row in 0_usize..4 {
            for column in 0_usize..4 {
                hermiticity_residual = hermiticity_residual
                    .max((matrix[row][column] - matrix[column][row].conj()).norm());
                if row.count_ones() != column.count_ones() {
                    let amplitude = matrix[row][column].norm();
                    if amplitude > charge_conservation_residual {
                        charge_conservation_residual = amplitude;
                        charge_witness = (row, column);
                    }
                }
            }
        }
        if hermiticity_residual > tolerance {
            return Err(U1MpsError::TwoSiteHamiltonianHermiticityResidual {
                residual: hermiticity_residual,
                tolerance,
            });
        }
        if charge_conservation_residual > tolerance {
            return Err(U1MpsError::TwoSiteHamiltonianChargeLeakage {
                output_charge: charge_witness.0.count_ones(),
                input_charge: charge_witness.1.count_ones(),
                amplitude: charge_conservation_residual,
                tolerance,
            });
        }

        let mut projected = [[Complex64::new(0.0, 0.0); 4]; 4];
        projected[0][0] = Complex64::new(matrix[0][0].re, 0.0);
        projected[3][3] = Complex64::new(matrix[3][3].re, 0.0);
        projected[1][1] = Complex64::new(matrix[1][1].re, 0.0);
        projected[2][2] = Complex64::new(matrix[2][2].re, 0.0);
        let projected_coupling = 0.5 * (matrix[1][2] + matrix[2][1].conj());
        projected[1][2] = projected_coupling;
        projected[2][1] = projected_coupling.conj();
        let projection_residual_frobenius = matrix
            .iter()
            .zip(projected.iter())
            .flat_map(|(row, projected_row)| {
                row.iter().zip(projected_row.iter())
            })
            .map(|(value, projected_value)| (*value - *projected_value).norm_sqr())
            .sum::<f64>()
            .sqrt();
        if !projection_residual_frobenius.is_finite() {
            return Err(U1MpsError::NonFiniteContraction(
                "Hamiltonian symmetry/Hermiticity projection residual",
            ));
        }

        Ok(Self {
            matrix,
            hermiticity_residual,
            charge_conservation_residual,
            projection_residual_frobenius,
        })
    }

    pub fn matrix(&self) -> &[[Complex64; 4]; 4] {
        &self.matrix
    }

    pub fn hermiticity_residual(&self) -> f64 {
        self.hermiticity_residual
    }

    pub fn charge_conservation_residual(&self) -> f64 {
        self.charge_conservation_residual
    }

    pub fn projection_residual_frobenius(&self) -> f64 {
        self.projection_residual_frobenius
    }

    /// Compute exp(-i dt H) analytically, block-by-block in the U(1) sectors.
    ///
    /// The single-excitation block is a closed-form 2x2 Hermitian exponential;
    /// this avoids a generic 4x4 matrix exponential and preserves block
    /// structure by construction.
    pub fn evolution_gate(
        &self,
        dt: f64,
        gate_tolerance: f64,
    ) -> Result<U1TwoSiteGate, U1MpsError> {
        if !dt.is_finite() {
            return Err(U1MpsError::NonFiniteContraction(
                "two-site Hamiltonian time step",
            ));
        }
        if !gate_tolerance.is_finite() || gate_tolerance <= 0.0 {
            return Err(U1MpsError::InvalidTolerance(gate_tolerance));
        }

        let h00 = self.matrix[0][0].re;
        let h11 = self.matrix[1][1].re;
        let h22 = self.matrix[2][2].re;
        let h33 = self.matrix[3][3].re;
        let mean = 0.5 * (h11 + h22);
        let difference = 0.5 * (h11 - h22);
        let coupling = 0.5 * (self.matrix[1][2] + self.matrix[2][1].conj());
        let radius = (difference * difference + coupling.norm_sqr()).sqrt();
        let angle = dt * radius;
        let cosine = angle.cos();
        let sine_over_radius = if radius <= f64::EPSILON * (1.0 + mean.abs()) {
            dt
        } else {
            angle.sin() / radius
        };
        let phase_single = Complex64::from_polar(1.0, -dt * mean);
        let imaginary_factor = Complex64::new(0.0, -sine_over_radius);

        let mut gate = [[Complex64::new(0.0, 0.0); 4]; 4];
        gate[0][0] = Complex64::from_polar(1.0, -dt * h00);
        gate[3][3] = Complex64::from_polar(1.0, -dt * h33);
        gate[1][1] =
            phase_single * (Complex64::new(cosine, 0.0) + imaginary_factor * difference);
        gate[1][2] = phase_single * imaginary_factor * coupling;
        gate[2][1] = phase_single * imaginary_factor * coupling.conj();
        gate[2][2] =
            phase_single * (Complex64::new(cosine, 0.0) - imaginary_factor * difference);

        let gate = U1TwoSiteGate::new(gate, gate_tolerance)?;
        // Duhamel's bound for H = H_projected + Delta, with H_projected
        // Hermitian and ||Delta||_2 <= ||Delta||_F:
        // ||exp(-i H dt)-exp(-i H_projected dt)||
        //   <= |dt| ||Delta|| exp(|dt| ||Delta||).
        // This remains conservative even when the supplied H is only
        // approximately Hermitian or U(1)-conserving.
        let projection_residual = self.projection_residual_frobenius;
        let exponential_growth = (dt.abs() * projection_residual).exp();
        let projection_operator_error_upper_bound =
            dt.abs() * projection_residual * exponential_growth;
        gate.with_hamiltonian_projection_error(projection_operator_error_upper_bound)
    }
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
    pub fn new(matrix: [[Complex64; 4]; 4], tolerance: f64) -> Result<Self, U1MpsError> {
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
        for output in 0_usize..4 {
            for input in 0_usize..4 {
                if output.count_ones() != input.count_ones() {
                    charge_conservation_residual =
                        charge_conservation_residual.max(matrix[output][input].norm());
                }
            }
        }
        if charge_conservation_residual > tolerance {
            let witness = (0_usize..4)
                .flat_map(|output| (0_usize..4).map(move |input| (output, input)))
                .find(|(output, input)| {
                    (*output).count_ones() != (*input).count_ones()
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
                unitarity_residual = unitarity_residual.max((overlap - expected).norm());
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
            hamiltonian_projection_error_operator_upper_bound: 0.0,
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

    pub fn hamiltonian_projection_error_operator_upper_bound(&self) -> f64 {
        self.hamiltonian_projection_error_operator_upper_bound
    }

    fn with_hamiltonian_projection_error(
        mut self,
        bound: f64,
    ) -> Result<Self, U1MpsError> {
        if !bound.is_finite() || bound < 0.0 {
            return Err(U1MpsError::NonFiniteContraction(
                "Hamiltonian projection error bound",
            ));
        }
        self.hamiltonian_projection_error_operator_upper_bound = bound;
        Ok(self)
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
            hamiltonian_projection_error_operator_upper_bound: 0.0,
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
            hamiltonian_projection_error_operator_upper_bound: 0.0,
        }
    }
}

/// Receipts from an exact charge-block split of a two-site state.
///
/// Exactly-zero singular directions are omitted as nullspace. Every positive
/// singular value is retained; this operation has no bond cap or cutoff.
#[derive(Debug, Clone, PartialEq)]
pub struct U1TwoSiteSplitDiagnostics {
    pub left_tensor: U1MpsTensor,
    pub right_tensor: U1MpsTensor,
    /// (intermediate charge, retained positive singular-value count).
    pub rank_by_charge: Vec<(i32, usize)>,
    pub retained_singular_values: usize,
    pub exact_zero_singular_values_omitted: usize,
    pub reconstruction_max_residual: f64,
    pub reconstruction_l2_residual: f64,
}

/// Policy for an explicitly approximate U(1)-preserving two-site split.
///
/// The relative cutoff keeps singular values at least this fraction of the
/// largest singular value. The maximum bond dimension is global across charge
/// sectors, not a per-sector cap. No implicit renormalization occurs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct U1TwoSiteTruncationPolicy {
    pub max_bond_dimension: usize,
    pub relative_singular_value_cutoff: f64,
    pub tolerance: f64,
}

/// Diagnostics from an approximate two-site charge-block split.
#[derive(Debug, Clone, PartialEq)]
pub struct U1TwoSiteTruncationDiagnostics {
    pub left_tensor: U1MpsTensor,
    pub right_tensor: U1MpsTensor,
    pub retained_rank_by_charge: Vec<(i32, usize)>,
    pub total_singular_value_count: usize,
    pub retained_singular_value_count: usize,
    pub cutoff_discarded_singular_value_count: usize,
    pub bond_cap_discarded_singular_value_count: usize,
    pub total_schmidt_weight: f64,
    pub retained_schmidt_weight: f64,
    pub cutoff_discarded_weight: f64,
    pub bond_cap_discarded_weight: f64,
    pub discarded_weight: f64,
    pub relative_discarded_weight: f64,
    pub pre_truncation_norm_squared: f64,
    pub post_truncation_norm_squared: f64,
    /// Scalar which would restore the original norm if applied; not applied.
    pub renormalization_factor_to_restore_original_norm: f64,
    /// True only when the cap forced otherwise cutoff-eligible modes out.
    pub bond_cap_binding: bool,
    pub reconstruction_max_residual: f64,
    pub reconstruction_l2_residual: f64,
}



/// Absolute local two-site L2 error budget for adaptive Schmidt compression.
///
/// Unlike a relative cutoff, this policy is stated in output-state amplitude
/// units. The implementation finds the smallest global rank across all charge
/// sectors whose discarded Schmidt tail meets the budget, subject to a hard
/// maximum bond dimension. It never renormalizes the compressed state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct U1TwoSiteErrorBudgetPolicy {
    pub maximum_bond_dimension: usize,
    pub maximum_absolute_local_l2_error: f64,
    pub tolerance: f64,
}

/// Result of minimum-rank, error-budget-driven two-site compression.
#[derive(Debug, Clone, PartialEq)]
pub struct U1TwoSiteBudgetedSplit {
    pub diagnostics: U1TwoSiteTruncationDiagnostics,
    pub requested_absolute_local_l2_error: f64,
    pub achieved_absolute_local_l2_error: f64,
    pub minimum_required_bond_dimension: usize,
    pub budget_slack: f64,
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
                        let index = (left_index * 4 + physical_pair) * right.right_dimension()
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
    pub fn apply_gate(&self, gate: &U1TwoSiteGate, tolerance: f64) -> Result<Self, U1MpsError> {
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

    /// Split this two-site tensor into two U(1)-symmetric site tensors.
    ///
    /// The matrix is block-diagonalized by the intermediate prefix charge q:
    /// rows satisfy q_left + s_first = q and columns satisfy
    /// q_right = q + s_second. Each charge block is factorized independently.
    ///
    /// No bond cap, positive singular-value cutoff, normalization or
    /// renormalization is applied. Exactly-zero singular directions are omitted;
    /// every positive singular value is retained, however small.
    pub fn split_exact(&self, tolerance: f64) -> Result<U1TwoSiteSplitDiagnostics, U1MpsError> {
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(U1MpsError::InvalidTolerance(tolerance));
        }

        struct ChargeBlock {
            charge: i32,
            rows: Vec<(usize, usize)>,
            columns: Vec<(usize, usize)>,
            left_vectors: DMatrix<Complex64>,
            right_vectors_t: DMatrix<Complex64>,
            singular_values: Vec<f64>,
        }

        let mut row_groups: Vec<(i32, Vec<(usize, usize)>)> = Vec::new();
        for left in 0..self.left_dimension() {
            for first_physical in 0..2 {
                let charge = self.left_charges[left] + first_physical as i32;
                if let Some((_, rows)) = row_groups.iter_mut().find(|(q, _)| *q == charge) {
                    rows.push((left, first_physical));
                } else {
                    row_groups.push((charge, vec![(left, first_physical)]));
                }
            }
        }

        let max_block_dimension = self
            .left_dimension()
            .saturating_mul(2)
            .max(self.right_dimension().saturating_mul(2));
        let max_iterations = 10_000usize
            .max(64usize.saturating_mul(max_block_dimension.saturating_mul(max_block_dimension)));
        let mut blocks = Vec::new();
        let mut exact_zero_singular_values_omitted = 0_usize;

        for (charge, rows) in row_groups {
            let mut columns = Vec::new();
            for second_physical in 0..2 {
                for right in 0..self.right_dimension() {
                    if self.right_charges[right] == charge + second_physical as i32 {
                        columns.push((second_physical, right));
                    }
                }
            }
            if rows.is_empty() || columns.is_empty() {
                continue;
            }

            let mut block = DMatrix::<Complex64>::zeros(rows.len(), columns.len());
            for (row_index, &(left, first_physical)) in rows.iter().enumerate() {
                for (column_index, &(second_physical, right)) in columns.iter().enumerate() {
                    let physical_pair = first_physical * 2 + second_physical;
                    block[(row_index, column_index)] = self.get(left, physical_pair, right);
                }
            }

            let block_rows = block.nrows();
            let block_columns = block.ncols();
            let decomposition = block.try_svd(true, true, tolerance, max_iterations).ok_or(
                U1MpsError::TwoSiteSvdFailed {
                    rows: block_rows,
                    columns: block_columns,
                },
            )?;
            let left_vectors = decomposition.u.ok_or(U1MpsError::TwoSiteSvdFailed {
                rows: block_rows,
                columns: block_columns,
            })?;
            let right_vectors_t = decomposition.v_t.ok_or(U1MpsError::TwoSiteSvdFailed {
                rows: block_rows,
                columns: block_columns,
            })?;

            let mut kept_indices = Vec::new();
            for (index, singular_value) in decomposition.singular_values.iter().enumerate() {
                if !singular_value.is_finite() {
                    return Err(U1MpsError::NonFiniteContraction(
                        "two-site charge-block singular value",
                    ));
                }
                if *singular_value > 0.0 {
                    kept_indices.push(index);
                } else {
                    exact_zero_singular_values_omitted += 1;
                }
            }
            if kept_indices.is_empty() {
                continue;
            }

            let kept_left =
                DMatrix::<Complex64>::from_fn(rows.len(), kept_indices.len(), |row, column| {
                    left_vectors[(row, kept_indices[column])]
                });
            let kept_right_t =
                DMatrix::<Complex64>::from_fn(kept_indices.len(), columns.len(), |row, column| {
                    right_vectors_t[(kept_indices[row], column)]
                });
            let singular_values = kept_indices
                .iter()
                .map(|&index| decomposition.singular_values[index])
                .collect();

            blocks.push(ChargeBlock {
                charge,
                rows,
                columns,
                left_vectors: kept_left,
                right_vectors_t: kept_right_t,
                singular_values,
            });
        }

        let total_rank = blocks.iter().try_fold(0_usize, |sum, block| {
            sum.checked_add(block.singular_values.len())
                .ok_or(U1MpsError::DimensionOverflow)
        })?;
        if total_rank == 0 {
            return Err(U1MpsError::ZeroDenseState);
        }

        let left_entries = self
            .left_dimension()
            .checked_mul(2)
            .and_then(|value| value.checked_mul(total_rank))
            .ok_or(U1MpsError::DimensionOverflow)?;
        let right_entries = total_rank
            .checked_mul(2)
            .and_then(|value| value.checked_mul(self.right_dimension()))
            .ok_or(U1MpsError::DimensionOverflow)?;
        let mut left_data = vec![Complex64::new(0.0, 0.0); left_entries];
        let mut right_data = vec![Complex64::new(0.0, 0.0); right_entries];
        let mut middle_charges = Vec::with_capacity(total_rank);
        let mut rank_by_charge = Vec::with_capacity(blocks.len());
        let mut offset = 0_usize;

        for block in &blocks {
            let rank = block.singular_values.len();
            rank_by_charge.push((block.charge, rank));
            middle_charges.extend(core::iter::repeat_n(block.charge, rank));

            for singular_index in 0..rank {
                let middle = offset + singular_index;
                for (row_index, &(left, first_physical)) in block.rows.iter().enumerate() {
                    let index = (left * 2 + first_physical) * total_rank + middle;
                    left_data[index] = block.left_vectors[(row_index, singular_index)];
                }

                for (column_index, &(second_physical, right)) in block.columns.iter().enumerate() {
                    let index = (middle * 2 + second_physical) * self.right_dimension() + right;
                    right_data[index] = block.singular_values[singular_index]
                        * block.right_vectors_t[(singular_index, column_index)];
                }
            }
            offset += rank;
        }

        let left_tensor =
            U1MpsTensor::new(self.left_charges.clone(), middle_charges.clone(), left_data)?;
        let right_tensor =
            U1MpsTensor::new(middle_charges, self.right_charges.clone(), right_data)?;

        let mut reconstruction_max_residual: f64 = 0.0;
        let mut reconstruction_squared = 0.0;
        for left in 0..self.left_dimension() {
            for first_physical in 0..2 {
                for second_physical in 0..2 {
                    let physical_pair = first_physical * 2 + second_physical;
                    for right in 0..self.right_dimension() {
                        let mut reconstructed = Complex64::new(0.0, 0.0);
                        for middle in 0..total_rank {
                            reconstructed += left_tensor.get(left, first_physical, middle)
                                * right_tensor.get(middle, second_physical, right);
                        }
                        let residual =
                            (reconstructed - self.get(left, physical_pair, right)).norm();
                        reconstruction_max_residual = reconstruction_max_residual.max(residual);
                        reconstruction_squared += residual * residual;
                    }
                }
            }
        }
        if !reconstruction_max_residual.is_finite() || !reconstruction_squared.is_finite() {
            return Err(U1MpsError::NonFiniteContraction(
                "two-site split reconstruction",
            ));
        }

        Ok(U1TwoSiteSplitDiagnostics {
            left_tensor,
            right_tensor,
            rank_by_charge,
            retained_singular_values: total_rank,
            exact_zero_singular_values_omitted,
            reconstruction_max_residual,
            reconstruction_l2_residual: reconstruction_squared.sqrt(),
        })
    }

    /// Apply an explicit relative singular-value cutoff and global bond cap.
    ///
    /// The exact charge-block split is computed first. Singular values are
    /// recovered as row norms of the singular-value-weighted V-adjoint factors.
    /// Global selection ranks modes across all charge sectors, then copies the
    /// selected basis vectors without mixing their charge labels.
    ///
    /// The returned state is deliberately not renormalized. Discarded squared
    /// Schmidt weight and the factor that would restore the original norm are
    /// reported separately.
    pub fn split_truncated(
        &self,
        policy: U1TwoSiteTruncationPolicy,
    ) -> Result<U1TwoSiteTruncationDiagnostics, U1MpsError> {
        if policy.max_bond_dimension == 0
            || !policy.relative_singular_value_cutoff.is_finite()
            || !(0.0..=1.0).contains(&policy.relative_singular_value_cutoff)
            || !policy.tolerance.is_finite()
            || policy.tolerance <= 0.0
        {
            return Err(U1MpsError::InvalidTruncationPolicy {
                max_bond_dimension: policy.max_bond_dimension,
                relative_cutoff: policy.relative_singular_value_cutoff,
                tolerance: policy.tolerance,
            });
        }

        let exact = self.split_exact(policy.tolerance)?;
        let full_rank = exact.retained_singular_values;
        let middle_charges = exact.left_tensor.right_charges();
        let mut singular_values = vec![0.0_f64; full_rank];
        let mut total_schmidt_weight = 0.0;
        for middle in 0..full_rank {
            let mut row_norm_squared = 0.0;
            for physical in 0..2 {
                for right in 0..exact.right_tensor.right_dimension() {
                    row_norm_squared += exact.right_tensor.get(middle, physical, right).norm_sqr();
                }
            }
            let singular_value = row_norm_squared.sqrt();
            if !singular_value.is_finite() {
                return Err(U1MpsError::NonFiniteContraction(
                    "two-site Schmidt singular value",
                ));
            }
            singular_values[middle] = singular_value;
            total_schmidt_weight += singular_value * singular_value;
        }
        if !total_schmidt_weight.is_finite() || total_schmidt_weight <= 0.0 {
            return Err(U1MpsError::ZeroDenseState);
        }

        let maximum_singular_value = singular_values.iter().copied().fold(0.0, f64::max);
        let cutoff_threshold = policy.relative_singular_value_cutoff * maximum_singular_value;
        let mut cutoff_eligible = Vec::new();
        let mut cutoff_discarded_weight = 0.0;
        let mut cutoff_discarded_singular_value_count = 0;
        for (index, singular_value) in singular_values.iter().copied().enumerate() {
            if singular_value >= cutoff_threshold {
                cutoff_eligible.push(index);
            } else {
                cutoff_discarded_weight += singular_value * singular_value;
                cutoff_discarded_singular_value_count += 1;
            }
        }

        cutoff_eligible.sort_by(|left, right| {
            singular_values[*right]
                .total_cmp(&singular_values[*left])
                .then_with(|| left.cmp(right))
        });
        let bond_cap_binding = cutoff_eligible.len() > policy.max_bond_dimension;
        let retained_sorted = cutoff_eligible
            .iter()
            .copied()
            .take(policy.max_bond_dimension)
            .collect::<Vec<_>>();
        let mut bond_cap_discarded_weight = 0.0;
        let mut bond_cap_discarded_singular_value_count = 0;
        for index in cutoff_eligible
            .iter()
            .copied()
            .skip(policy.max_bond_dimension)
        {
            bond_cap_discarded_weight += singular_values[index] * singular_values[index];
            bond_cap_discarded_singular_value_count += 1;
        }
        let mut retained = retained_sorted;
        retained.sort_unstable();

        let retained_rank = retained.len();
        if retained_rank == 0 {
            return Err(U1MpsError::ZeroDenseState);
        }
        let retained_schmidt_weight = retained
            .iter()
            .map(|index| singular_values[*index] * singular_values[*index])
            .sum::<f64>();
        let discarded_weight = cutoff_discarded_weight + bond_cap_discarded_weight;
        let relative_discarded_weight = discarded_weight / total_schmidt_weight;
        let renormalization_factor_to_restore_original_norm =
            (total_schmidt_weight / retained_schmidt_weight).sqrt();

        let left_entries = exact
            .left_tensor
            .left_dimension()
            .checked_mul(2)
            .and_then(|value| value.checked_mul(retained_rank))
            .ok_or(U1MpsError::DimensionOverflow)?;
        let right_entries = retained_rank
            .checked_mul(2)
            .and_then(|value| value.checked_mul(exact.right_tensor.right_dimension()))
            .ok_or(U1MpsError::DimensionOverflow)?;
        let mut left_data = vec![Complex64::new(0.0, 0.0); left_entries];
        let mut right_data = vec![Complex64::new(0.0, 0.0); right_entries];
        let mut retained_middle_charges = Vec::with_capacity(retained_rank);
        for (new_middle, old_middle) in retained.iter().copied().enumerate() {
            retained_middle_charges.push(middle_charges[old_middle]);
            for left in 0..exact.left_tensor.left_dimension() {
                for first_physical in 0..2 {
                    let index = (left * 2 + first_physical) * retained_rank + new_middle;
                    left_data[index] = exact.left_tensor.get(left, first_physical, old_middle);
                }
            }
            for second_physical in 0..2 {
                for right in 0..exact.right_tensor.right_dimension() {
                    let index = (new_middle * 2 + second_physical)
                        * exact.right_tensor.right_dimension()
                        + right;
                    right_data[index] = exact.right_tensor.get(old_middle, second_physical, right);
                }
            }
        }

        let left_tensor = U1MpsTensor::new(
            exact.left_tensor.left_charges().to_vec(),
            retained_middle_charges.clone(),
            left_data,
        )?;
        let right_tensor = U1MpsTensor::new(
            retained_middle_charges.clone(),
            exact.right_tensor.right_charges().to_vec(),
            right_data,
        )?;

        let retained_state_entries = self
            .left_dimension()
            .checked_mul(4)
            .and_then(|value| value.checked_mul(self.right_dimension()))
            .ok_or(U1MpsError::DimensionOverflow)?;
        let mut retained_state_data = vec![Complex64::new(0.0, 0.0); retained_state_entries];
        let mut reconstruction_max_residual: f64 = 0.0;
        let mut reconstruction_squared = 0.0;
        for left in 0..self.left_dimension() {
            for first_physical in 0..2 {
                for second_physical in 0..2 {
                    let pair = first_physical * 2 + second_physical;
                    for right in 0..self.right_dimension() {
                        let mut value = Complex64::new(0.0, 0.0);
                        for middle in 0..retained_rank {
                            value += left_tensor.get(left, first_physical, middle)
                                * right_tensor.get(middle, second_physical, right);
                        }
                        let index = (left * 4 + pair) * self.right_dimension() + right;
                        retained_state_data[index] = value;
                        let residual = (value - self.get(left, pair, right)).norm();
                        reconstruction_max_residual = reconstruction_max_residual.max(residual);
                        reconstruction_squared += residual * residual;
                    }
                }
            }
        }
        if !discarded_weight.is_finite()
            || !relative_discarded_weight.is_finite()
            || !renormalization_factor_to_restore_original_norm.is_finite()
            || !reconstruction_squared.is_finite()
        {
            return Err(U1MpsError::NonFiniteContraction(
                "two-site truncation diagnostics",
            ));
        }
        let _retained_state = U1TwoSiteState::new(
            self.left_charges.clone(),
            self.right_charges.clone(),
            retained_state_data,
            policy.tolerance,
        )?;

        let mut retained_rank_by_charge = Vec::new();
        for charge in retained_middle_charges.iter().copied() {
            if !retained_rank_by_charge
                .iter()
                .any(|(existing, _)| *existing == charge)
            {
                let count = retained_middle_charges
                    .iter()
                    .filter(|candidate| **candidate == charge)
                    .count();
                retained_rank_by_charge.push((charge, count));
            }
        }

        Ok(U1TwoSiteTruncationDiagnostics {
            left_tensor,
            right_tensor,
            retained_rank_by_charge,
            total_singular_value_count: full_rank,
            retained_singular_value_count: retained_rank,
            cutoff_discarded_singular_value_count,
            bond_cap_discarded_singular_value_count,
            total_schmidt_weight,
            retained_schmidt_weight,
            cutoff_discarded_weight,
            bond_cap_discarded_weight,
            discarded_weight,
            relative_discarded_weight,
            pre_truncation_norm_squared: total_schmidt_weight,
            post_truncation_norm_squared: retained_schmidt_weight,
            renormalization_factor_to_restore_original_norm,
            bond_cap_binding,
            reconstruction_max_residual,
            reconstruction_l2_residual: reconstruction_squared.sqrt(),
        })
    }



    /// Choose the smallest global Schmidt rank that satisfies an absolute L2
    /// projection-error budget, failing closed when the hard bond cap is too
    /// small or the measured reconstruction residual exceeds the budget.
    ///
    /// The Schmidt spectrum is collected across all intermediate U(1) charge
    /// sectors and sorted globally. The minimum rank is the first k for which
    /// the sum of squared discarded singular values is at most the budget
    /// squared. The final result is checked against the measured two-site
    /// reconstruction residual, which also captures floating-point
    /// factorization error and tolerated forbidden-sector components.
    ///
    /// Numerical comparison slack is tolerance times the input norm. It is not
    /// added to the reported achieved error and does not renormalize the output.
    pub fn split_with_error_budget(
        &self,
        policy: U1TwoSiteErrorBudgetPolicy,
    ) -> Result<U1TwoSiteBudgetedSplit, U1MpsError> {
        if policy.maximum_bond_dimension == 0
            || !policy.maximum_absolute_local_l2_error.is_finite()
            || policy.maximum_absolute_local_l2_error < 0.0
            || !policy.tolerance.is_finite()
            || policy.tolerance <= 0.0
        {
            return Err(U1MpsError::InvalidErrorBudget {
                max_bond_dimension: policy.maximum_bond_dimension,
                maximum_absolute_local_l2_error: policy.maximum_absolute_local_l2_error,
                tolerance: policy.tolerance,
            });
        }

        let exact = self.split_exact(policy.tolerance)?;
        let rank = exact.retained_singular_values;
        let mut singular_values = Vec::with_capacity(rank);
        let mut total_weight = 0.0_f64;
        for middle in 0..rank {
            let mut squared = 0.0;
            for physical in 0..2 {
                for right in 0..exact.right_tensor.right_dimension() {
                    squared += exact.right_tensor.get(middle, physical, right).norm_sqr();
                }
            }
            let singular_value = squared.sqrt();
            if !singular_value.is_finite() {
                return Err(U1MpsError::NonFiniteContraction(
                    "error-budget Schmidt singular value",
                ));
            }
            singular_values.push(singular_value);
            total_weight += singular_value * singular_value;
        }
        if !total_weight.is_finite() {
            return Err(U1MpsError::NonFiniteContraction(
                "error-budget Schmidt weight",
            ));
        }
        let input_norm = total_weight.sqrt();
        if !input_norm.is_finite() {
            return Err(U1MpsError::NonFiniteContraction(
                "error-budget input norm",
            ));
        }
        if input_norm <= 0.0 {
            return Err(U1MpsError::ZeroDenseState);
        }

        singular_values.sort_by(f64::total_cmp);
        singular_values.reverse();
        let comparison_slack = policy.tolerance * input_norm;
        let allowed_error = policy.maximum_absolute_local_l2_error + comparison_slack;
        let allowed_squared = allowed_error * allowed_error;
        let mut retained_weight = 0.0;
        let mut minimum_required_bond_dimension = 0_usize;
        while minimum_required_bond_dimension < singular_values.len()
            && total_weight - retained_weight > allowed_squared
        {
            retained_weight += singular_values[minimum_required_bond_dimension].powi(2);
            minimum_required_bond_dimension += 1;
        }
        minimum_required_bond_dimension = minimum_required_bond_dimension.max(1);

        let achievable_rank = policy.maximum_bond_dimension.min(rank).max(1);
        let requested_rank = minimum_required_bond_dimension.min(achievable_rank);
        let diagnostics = self.split_truncated(U1TwoSiteTruncationPolicy {
            max_bond_dimension: requested_rank,
            relative_singular_value_cutoff: 0.0,
            tolerance: policy.tolerance,
        })?;
        let achieved = diagnostics.reconstruction_l2_residual;

        if minimum_required_bond_dimension > policy.maximum_bond_dimension
            || achieved > allowed_error
        {
            let best = if requested_rank == achievable_rank {
                achieved
            } else {
                self.split_truncated(U1TwoSiteTruncationPolicy {
                    max_bond_dimension: achievable_rank,
                    relative_singular_value_cutoff: 0.0,
                    tolerance: policy.tolerance,
                })?
                .reconstruction_l2_residual
            };
            return Err(U1MpsError::TruncationBudgetUnreachable {
                requested_absolute_l2_error: policy.maximum_absolute_local_l2_error,
                maximum_bond_dimension: policy.maximum_bond_dimension,
                minimum_required_bond_dimension,
                best_achievable_l2_error: best,
            });
        }

        Ok(U1TwoSiteBudgetedSplit {
            diagnostics,
            requested_absolute_local_l2_error: policy.maximum_absolute_local_l2_error,
            achieved_absolute_local_l2_error: achieved,
            minimum_required_bond_dimension,
            budget_slack: policy.maximum_absolute_local_l2_error - achieved,
        })
    }


    /// Return the four computational-basis amplitudes for open-boundary tensors.
    pub fn to_dense_two_qubit(&self) -> Result<[Complex64; 4], U1MpsError> {
        if self.left_dimension() != 1 || self.right_dimension() != 1 {
            return Err(U1MpsError::TwoSiteDenseExpansionRequiresOpenBoundaries);
        }
        Ok(core::array::from_fn(|index| self.get(0, index, 0)))
    }
}

/// Cumulative local L2 projection ledger for a sequence of two-site splits.
///
/// The sum of local sqrt(discarded_weight) values is a global state-vector
/// error bound only when every split is performed in a mixed-canonical gauge
/// with orthonormal exterior left/right block bases, and intervening gates are
/// unitary. The current TEBD scheduler does not verify that canonical-gauge
/// precondition, so global qualification remains false.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct U1MpsTruncationLedger {
    gate_count: usize,
    accumulated_discarded_weight: f64,
    cumulative_local_l2_error_bound: f64,
    cumulative_canonicalization_l2_error_bound: f64,
    cumulative_gate_quality_l2_error_bound: f64,
    cumulative_hamiltonian_projection_l2_error_bound: f64,
    cumulative_projection_l2_error_bound: f64,
    maximum_single_gate_relative_discarded_weight: f64,
    canonical_gauge_receipt_count: usize,
    global_state_error_bound_qualified: bool,
    global_state_error_bound_upper: Option<f64>,
    input_state_fingerprint: Option<u128>,
}

fn gate_quality_error_bounds_from_residuals(
    unitarity_residual: f64,
    charge_residual: f64,
    pre_gate_norm_squared: f64,
) -> Result<(f64, f64), U1MpsError> {
    if !pre_gate_norm_squared.is_finite() || pre_gate_norm_squared < 0.0 {
        return Err(U1MpsError::NonFiniteContraction(
            "pre-gate norm in canonical error ledger",
        ));
    }
    // U1TwoSiteGate stores the maximum absolute entry of U†U-I.
    // For a 4x4 matrix, ||R||_2 <= ||R||_F <= 4 * max_ij |R_ij|.
    // Use this operator-norm bound before converting it into a bound on ||U||.
    let unitarity_operator_residual_upper = 4.0 * unitarity_residual;
    // There are exactly ten matrix entries coupling distinct excitation sectors.
    // Their Frobenius norm, and hence operator norm, is bounded by sqrt(10)*max.
    let off_sector_operator_error = 10.0_f64.sqrt() * charge_residual;
    let gate_operator_norm_upper = (1.0 + unitarity_operator_residual_upper).sqrt();
    // Let D = G - E be the block-diagonal part of G, with E its off-sector part.
    // ||D†D-I|| <= ||G†G-I|| + 2||G||||E|| + ||E||².
    let block_unitarity_residual = unitarity_operator_residual_upper
        + 2.0 * gate_operator_norm_upper * off_sector_operator_error
        + off_sector_operator_error * off_sector_operator_error;
    if !block_unitarity_residual.is_finite() || block_unitarity_residual >= 1.0 {
        return Err(U1MpsError::GateUnitarityResidual {
            residual: block_unitarity_residual,
            tolerance: 1.0,
        });
    }
    let polar_distance = ((1.0 + block_unitarity_residual).sqrt() - 1.0)
        .max(1.0 - (1.0 - block_unitarity_residual).sqrt());
    let quality_error =
        (off_sector_operator_error + polar_distance) * pre_gate_norm_squared.sqrt();
    if !gate_operator_norm_upper.is_finite() || !quality_error.is_finite() {
        return Err(U1MpsError::NonFiniteContraction(
            "gate quality operator bound",
        ));
    }
    Ok((gate_operator_norm_upper, quality_error))
}

fn gate_quality_error_bounds(
    gate: &U1TwoSiteGate,
    pre_gate_norm_squared: f64,
) -> Result<(f64, f64), U1MpsError> {
    gate_quality_error_bounds_from_residuals(
        gate.unitarity_residual(),
        gate.charge_conservation_residual(),
        pre_gate_norm_squared,
    )
}


impl U1MpsTruncationLedger {
    pub fn gate_count(&self) -> usize { self.gate_count }
    pub fn accumulated_discarded_weight(&self) -> f64 { self.accumulated_discarded_weight }
    pub fn cumulative_local_l2_error_bound(&self) -> f64 { self.cumulative_local_l2_error_bound }
    pub fn cumulative_canonicalization_l2_error_bound(&self) -> f64 {
        self.cumulative_canonicalization_l2_error_bound
    }
    pub fn cumulative_gate_quality_l2_error_bound(&self) -> f64 {
        self.cumulative_gate_quality_l2_error_bound
    }
    pub fn cumulative_hamiltonian_projection_l2_error_bound(&self) -> f64 {
        self.cumulative_hamiltonian_projection_l2_error_bound
    }
    pub fn cumulative_projection_l2_error_bound(&self) -> f64 {
        self.cumulative_projection_l2_error_bound
    }
    pub fn maximum_single_gate_relative_discarded_weight(&self) -> f64 {
        self.maximum_single_gate_relative_discarded_weight
    }
    pub fn canonical_gauge_receipt_count(&self) -> usize {
        self.canonical_gauge_receipt_count
    }
    pub fn global_state_error_bound_qualified(&self) -> bool {
        self.global_state_error_bound_qualified
    }
    pub fn global_state_error_bound_upper(&self) -> Option<f64> {
        self.global_state_error_bound_upper
    }

    fn record_local(
        &mut self,
        diagnostics: &U1TwoSiteTruncationDiagnostics,
    ) -> Result<(), U1MpsError> {
        if !diagnostics.discarded_weight.is_finite()
            || diagnostics.discarded_weight < 0.0
            || !diagnostics.relative_discarded_weight.is_finite()
            || !(0.0..=1.0 + f64::EPSILON).contains(&diagnostics.relative_discarded_weight)
        {
            return Err(U1MpsError::NonFiniteContraction("truncation ledger input"));
        }
        let next_count = self
            .gate_count
            .checked_add(1)
            .ok_or(U1MpsError::DimensionOverflow)?;
        let next_weight = self.accumulated_discarded_weight + diagnostics.discarded_weight;
        if !diagnostics.reconstruction_l2_residual.is_finite()
            || diagnostics.reconstruction_l2_residual < 0.0
        {
            return Err(U1MpsError::NonFiniteContraction(
                "two-site reconstruction residual in truncation ledger",
            ));
        }
        // Use the measured pre/post two-site tensor residual, not merely the
        // ideal discarded Schmidt weight. The measured residual also includes
        // finite-precision factorization error and tolerated forbidden-sector
        // components that are absent from the Schmidt-weight sum.
        let next_bound =
            self.cumulative_local_l2_error_bound + diagnostics.reconstruction_l2_residual;
        if !next_weight.is_finite() || !next_bound.is_finite() {
            return Err(U1MpsError::NonFiniteContraction(
                "cumulative truncation ledger",
            ));
        }
        self.gate_count = next_count;
        self.accumulated_discarded_weight = next_weight;
        self.cumulative_local_l2_error_bound = next_bound;
        self.maximum_single_gate_relative_discarded_weight = self
            .maximum_single_gate_relative_discarded_weight
            .max(diagnostics.relative_discarded_weight);
        Ok(())
    }

    /// Record local tensor truncation diagnostics without claiming a global
    /// state-vector bound.
    fn record(
        &mut self,
        diagnostics: &U1TwoSiteTruncationDiagnostics,
    ) -> Result<(), U1MpsError> {
        self.record_local(diagnostics)?;
        self.global_state_error_bound_qualified = false;
        self.global_state_error_bound_upper = None;
        Ok(())
    }

    /// Compatibility helper for callers that have already established an
    /// exactly unitary, charge-conserving two-site gate. Production TEBD should
    /// use record_with_canonical_gauge_and_gate so gate defects are included.
    fn record_with_canonical_gauge(
        &mut self,
        diagnostics: &U1TwoSiteTruncationDiagnostics,
        receipt: &U1CanonicalGaugeReceipt,
        tolerance: f64,
    ) -> Result<(), U1MpsError> {
        self.record_with_gate_error_bounds(
            diagnostics,
            receipt,
            0.0,
            0.0,
            0.0,
            receipt.output_norm_squared,
            tolerance,
        )
    }

    /// Record a truncation with a measured mixed-canonical gauge receipt and
    /// explicitly measured gate quality. The matrix residuals are converted
    /// into conservative operator-error bounds before they enter the ledger.
    fn record_with_canonical_gauge_and_gate(
        &mut self,
        diagnostics: &U1TwoSiteTruncationDiagnostics,
        receipt: &U1CanonicalGaugeReceipt,
        gate: &U1TwoSiteGate,
        tolerance: f64,
    ) -> Result<(), U1MpsError> {
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(U1MpsError::InvalidTolerance(tolerance));
        }
        let unitarity_residual = gate.unitarity_residual();
        let charge_residual = gate.charge_conservation_residual();
        if !unitarity_residual.is_finite() || unitarity_residual > tolerance {
            return Err(U1MpsError::GateUnitarityResidual {
                residual: unitarity_residual,
                tolerance,
            });
        }
        if !charge_residual.is_finite() || charge_residual > tolerance {
            return Err(U1MpsError::GateChargeConservationResidual {
                residual: charge_residual,
                tolerance,
            });
        }
        self.record_with_gate_error_bounds(
            diagnostics,
            receipt,
            unitarity_residual,
            charge_residual,
            gate.hamiltonian_projection_error_operator_upper_bound(),
            receipt.output_norm_squared,
            tolerance,
        )
    }

    fn record_with_gate_error_bounds(
        &mut self,
        diagnostics: &U1TwoSiteTruncationDiagnostics,
        receipt: &U1CanonicalGaugeReceipt,
        gate_unitarity_residual: f64,
        gate_charge_residual: f64,
        hamiltonian_projection_operator_error_upper_bound: f64,
        pre_gate_norm_squared: f64,
        tolerance: f64,
    ) -> Result<(), U1MpsError> {
        let prior_count = self.gate_count;
        let prior_qualified = self.global_state_error_bound_qualified;
        let prior_bound = self.global_state_error_bound_upper;
        let prior_canonicalization_error = self.cumulative_canonicalization_l2_error_bound;
        let prior_gate_quality_error = self.cumulative_gate_quality_l2_error_bound;
        let prior_hamiltonian_projection_error =
            self.cumulative_hamiltonian_projection_l2_error_bound;
        let prior_projection_error = self.cumulative_projection_l2_error_bound;

        if !gate_unitarity_residual.is_finite()
            || !(0.0..1.0).contains(&gate_unitarity_residual)
            || !gate_charge_residual.is_finite()
            || gate_charge_residual < 0.0
        {
            self.global_state_error_bound_qualified = false;
            self.global_state_error_bound_upper = None;
            return Err(U1MpsError::NonFiniteContraction(
                "gate quality residual in canonical error ledger",
            ));
        }

        if !hamiltonian_projection_operator_error_upper_bound.is_finite()
            || hamiltonian_projection_operator_error_upper_bound < 0.0
        {
            return Err(U1MpsError::NonFiniteContraction(
                "Hamiltonian projection operator-error bound",
            ));
        }
        let (gate_operator_norm_upper, gate_quality_error) =
            gate_quality_error_bounds_from_residuals(
                gate_unitarity_residual,
                gate_charge_residual,
                pre_gate_norm_squared,
            )?;
        let hamiltonian_projection_error = hamiltonian_projection_operator_error_upper_bound
            * pre_gate_norm_squared.sqrt();
        if !receipt.qualifies(tolerance) {
            self.record_local(diagnostics)?;
            self.canonical_gauge_receipt_count = self
                .canonical_gauge_receipt_count
                .checked_add(1)
                .ok_or(U1MpsError::DimensionOverflow)?;
            self.global_state_error_bound_qualified = false;
            self.global_state_error_bound_upper = None;
            return Ok(());
        }

        let projection_error = diagnostics.reconstruction_l2_residual
            * receipt.exterior_operator_norm_product_upper_bound();
        let canonicalization_error = receipt.state_reconstruction_l2_residual;
        // Canonicalization and prior-state errors occur before G and must be
        // propagated by the certified operator-norm bound of the actual gate.
        let contribution = canonicalization_error * gate_operator_norm_upper
            + gate_quality_error
            + hamiltonian_projection_error
            + projection_error;
        if !contribution.is_finite()
            || !gate_operator_norm_upper.is_finite()
            || !gate_quality_error.is_finite()
            || !hamiltonian_projection_error.is_finite()
        {
            return Err(U1MpsError::NonFiniteContraction(
                "canonical-gauge global truncation contribution",
            ));
        }

        self.record_local(diagnostics)?;
        self.canonical_gauge_receipt_count = self
            .canonical_gauge_receipt_count
            .checked_add(1)
            .ok_or(U1MpsError::DimensionOverflow)?;

        let next_bound = if prior_count == 0 {
            Some(contribution)
        } else if prior_qualified {
            prior_bound
                .map(|bound| bound * gate_operator_norm_upper + contribution)
                .filter(|bound| bound.is_finite())
        } else {
            None
        };

        if let Some(bound) = next_bound {
            self.global_state_error_bound_qualified = true;
            self.global_state_error_bound_upper = Some(bound);
            self.cumulative_canonicalization_l2_error_bound =
                prior_canonicalization_error * gate_operator_norm_upper
                    + canonicalization_error * gate_operator_norm_upper;
            self.cumulative_gate_quality_l2_error_bound =
                prior_gate_quality_error * gate_operator_norm_upper + gate_quality_error;
            self.cumulative_hamiltonian_projection_l2_error_bound =
                prior_hamiltonian_projection_error * gate_operator_norm_upper
                    + hamiltonian_projection_error;
            self.cumulative_projection_l2_error_bound =
                prior_projection_error * gate_operator_norm_upper + projection_error;
        } else {
            self.global_state_error_bound_qualified = false;
            self.global_state_error_bound_upper = None;
        }
        Ok(())
    }
}

/// One nearest-neighbour two-site gate attached to a bond index.
#[derive(Debug, Clone, PartialEq)]
pub struct U1TeBDGate {
    bond_index: usize,
    gate: U1TwoSiteGate,
}

impl U1TeBDGate {
    pub fn new(bond_index: usize, gate: U1TwoSiteGate) -> Self {
        Self { bond_index, gate }
    }

    pub const fn bond_index(&self) -> usize {
        self.bond_index
    }

    pub fn gate(&self) -> &U1TwoSiteGate {
        &self.gate
    }
}

/// A deterministic even- or odd-bond layer of disjoint nearest-neighbour gates.
#[derive(Debug, Clone, PartialEq)]
pub struct U1TeBDLayer {
    site_count: usize,
    parity: u8,
    gates: Vec<U1TeBDGate>,
}

/// Result of one TEBD layer, including its output MPS and per-gate error receipts.
/// Per-gate evidence bundle. Fields are kept together so truncation,
/// canonical-gauge and gate-quality receipts cannot be detached by index.
#[derive(Debug, Clone, PartialEq)]
pub struct U1TeBDGateReceipt {
    pub bond_index: usize,
    pub gate_unitarity_residual: f64,
    pub gate_charge_conservation_residual: f64,
    pub gate_quality_error_upper_bound: f64,
    pub canonical_gauge: U1CanonicalGaugeReceipt,
    pub truncation: U1TwoSiteTruncationDiagnostics,
    pub local_projection_error_upper_bound: f64,
    pub global_l2_error_contribution: Option<f64>,
    pub global_l2_error_bound_after_gate: Option<f64>,
    pub global_projection_error_upper_bound: f64,
    pub hamiltonian_projection_operator_error_upper_bound: f64,
    pub hamiltonian_projection_error_upper_bound: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct U1TeBDLayerResult {
    pub state: ComplexU1Mps,
    pub parity: u8,
    pub bond_indices: Vec<usize>,
    pub gate_truncations: Vec<U1TwoSiteTruncationDiagnostics>,
    pub canonical_gauge_receipts: Vec<U1CanonicalGaugeReceipt>,
    pub gate_receipts: Vec<U1TeBDGateReceipt>,
    pub truncation_ledger: U1MpsTruncationLedger,
    pub input_norm_squared: f64,
    pub output_norm_squared: f64,
    pub input_total_charge: i32,
    pub output_total_charge: i32,
    pub output_selection_rule_residual: f64,
    /// False until a mixed-canonical exterior-gauge receipt is available.
    pub global_l2_bound_qualified: bool,
}

impl U1TeBDLayer {
    /// Build a parity layer. The gate list is sorted by bond index and validated
    /// for bounds, parity and uniqueness. For a fixed parity, gates are disjoint.
    pub fn new(
        site_count: usize,
        parity: u8,
        mut gates: Vec<U1TeBDGate>,
    ) -> Result<Self, U1MpsError> {
        if site_count == 0 {
            return Err(U1MpsError::EmptyChain);
        }
        if parity > 1 {
            return Err(U1MpsError::InvalidTeBDParity(parity));
        }
        for gate in &gates {
            if gate.bond_index >= site_count.saturating_sub(1) {
                return Err(U1MpsError::TeBDGateOutOfRange {
                    bond_index: gate.bond_index,
                    site_count,
                });
            }
            if gate.bond_index % 2 != parity as usize {
                return Err(U1MpsError::TeBDGateWrongParity {
                    bond_index: gate.bond_index,
                    parity,
                });
            }
        }
        gates.sort_by_key(U1TeBDGate::bond_index);
        for pair in gates.windows(2) {
            if pair[0].bond_index == pair[1].bond_index {
                return Err(U1MpsError::DuplicateTeBDGate {
                    bond_index: pair[0].bond_index,
                });
            }
        }
        Ok(Self {
            site_count,
            parity,
            gates,
        })
    }

    pub const fn site_count(&self) -> usize {
        self.site_count
    }

    pub const fn parity(&self) -> u8 {
        self.parity
    }

    pub fn gates(&self) -> &[U1TeBDGate] {
        &self.gates
    }

    pub fn bond_indices(&self) -> Vec<usize> {
        self.gates.iter().map(U1TeBDGate::bond_index).collect()
    }

    /// Execute the layer as sorted disjoint two-site gate + truncation operations.
    ///
    /// No renormalization is performed. Each gate's exact unitary application,
    /// charge-sector split and approximate compression is recorded in order.
    /// The L2 ledger is valid under the assumptions documented by
    /// U1MpsTruncationLedger: unitary gates and orthogonal Schmidt projections.
    pub fn apply(
        &self,
        input: &ComplexU1Mps,
        policy: U1TwoSiteTruncationPolicy,
    ) -> Result<U1TeBDLayerResult, U1MpsError> {
        self.apply_with_ledger(input, policy, U1MpsTruncationLedger::default())
    }

    fn apply_with_ledger(
        &self,
        input: &ComplexU1Mps,
        policy: U1TwoSiteTruncationPolicy,
        mut truncation_ledger: U1MpsTruncationLedger,
    ) -> Result<U1TeBDLayerResult, U1MpsError> {
        if input.site_count() != self.site_count {
            return Err(U1MpsError::TeBDLayerSiteCountMismatch {
                layer_sites: self.site_count,
                state_sites: input.site_count(),
            });
        }
        if let Some(expected_fingerprint) = truncation_ledger.input_state_fingerprint {
            if expected_fingerprint != input.state_fingerprint() {
                return Err(U1MpsError::TruncationLedgerStateMismatch);
            }
        } else if truncation_ledger.gate_count != 0 {
            return Err(U1MpsError::TruncationLedgerStateMismatch);
        }
        let input_norm_squared = input.norm_squared(policy.tolerance)?;
        let input_total_charge = input.total_charge;
        let mut state = input.clone();
        let mut gate_truncations = Vec::with_capacity(self.gates.len());
        let mut canonical_gauge_receipts = Vec::with_capacity(self.gates.len());
        let mut gate_receipts = Vec::with_capacity(self.gates.len());

        for entry in &self.gates {
            let bond = entry.bond_index;
            let gauge_receipt = state.canonicalize_for_bond(bond, policy.tolerance)?;
            let pair = U1TwoSiteState::from_adjacent_tensors(
                &state.tensors[bond],
                &state.tensors[bond + 1],
                policy.tolerance,
            )?;
            let evolved_pair = pair.apply_gate(&entry.gate, policy.tolerance)?;
            let truncation = evolved_pair.split_truncated(policy)?;
            let prior_global_bound = truncation_ledger.global_state_error_bound_upper;
            let (gate_operator_norm_upper, gate_quality_error_upper_bound) =
                gate_quality_error_bounds(&entry.gate, gauge_receipt.output_norm_squared)?;
            let global_projection_error_upper_bound = truncation.reconstruction_l2_residual
                * gauge_receipt.exterior_operator_norm_product_upper_bound();
            truncation_ledger.record_with_canonical_gauge_and_gate(
                &truncation,
                &gauge_receipt,
                &entry.gate,
                policy.tolerance,
            )?;
            let propagated_prior_bound =
                prior_global_bound.unwrap_or(0.0) * gate_operator_norm_upper;
            let global_l2_error_contribution = truncation_ledger
                .global_state_error_bound_upper
                .map(|after| after - propagated_prior_bound);
            let global_l2_error_bound_after_gate =
                truncation_ledger.global_state_error_bound_upper;
            let local_projection_error_upper_bound = truncation.reconstruction_l2_residual;
            let hamiltonian_projection_operator_error_upper_bound =
                entry.gate.hamiltonian_projection_error_operator_upper_bound();
            let hamiltonian_projection_error_upper_bound =
                hamiltonian_projection_operator_error_upper_bound
                    * gauge_receipt.output_norm_squared.sqrt();
            state.tensors[bond] = truncation.left_tensor.clone();
            state.tensors[bond + 1] = truncation.right_tensor.clone();
            state = ComplexU1Mps::new(state.tensors.clone(), input_total_charge, policy.tolerance)?;
            canonical_gauge_receipts.push(gauge_receipt.clone());
            gate_receipts.push(U1TeBDGateReceipt {
                bond_index: bond,
                gate_unitarity_residual: entry.gate.unitarity_residual(),
                gate_charge_conservation_residual: entry.gate.charge_conservation_residual(),
                gate_quality_error_upper_bound,
                canonical_gauge: gauge_receipt,
                truncation: truncation.clone(),
                local_projection_error_upper_bound,
                global_l2_error_contribution,
                global_l2_error_bound_after_gate,
                global_projection_error_upper_bound,
                hamiltonian_projection_operator_error_upper_bound,
                hamiltonian_projection_error_upper_bound,
            });
            gate_truncations.push(truncation);
        }

        let output_norm_squared = state.norm_squared(policy.tolerance)?;
        truncation_ledger.input_state_fingerprint = Some(state.state_fingerprint());
        let global_l2_bound_qualified = truncation_ledger.global_state_error_bound_qualified;
        Ok(U1TeBDLayerResult {
            output_selection_rule_residual: state.selection_rule_residual(),
            output_total_charge: state.total_charge(),
            state,
            parity: self.parity,
            bond_indices: self.bond_indices(),
            gate_truncations,
            canonical_gauge_receipts,
            gate_receipts,
            truncation_ledger,
            input_norm_squared,
            output_norm_squared,
            input_total_charge,
            global_l2_bound_qualified,
        })
    }
}



/// One nearest-neighbour two-site Hamiltonian term in a U(1) chain.
#[derive(Debug, Clone, PartialEq)]
pub struct U1TeBDHamiltonianTerm {
    bond_index: usize,
    hamiltonian: U1TwoSiteHamiltonian,
}

impl U1TeBDHamiltonianTerm {
    pub fn new(bond_index: usize, hamiltonian: U1TwoSiteHamiltonian) -> Self {
        Self { bond_index, hamiltonian }
    }

    pub const fn bond_index(&self) -> usize {
        self.bond_index
    }

    pub fn hamiltonian(&self) -> &U1TwoSiteHamiltonian {
        &self.hamiltonian
    }
}

/// Second-order nearest-neighbour TEBD stepper using Strang composition.
///
/// A full step applies exp(-i dt H_even/2), exp(-i dt H_odd), then
/// exp(-i dt H_even/2). Same-parity terms act on disjoint bonds and commute
/// within each layer. The truncation ledger is passed through all three
/// layers rather than reset at each composition boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct U1TeBDStrangStepper {
    site_count: usize,
    terms: Vec<U1TeBDHamiltonianTerm>,
}

/// Result of one second-order Strang TEBD time step.
///
/// The global L2 receipt covers representation, canonicalization, gate quality
/// and local truncation errors. Trotter error is not bounded by this receipt;
/// it requires a step-refinement study or an independent commutator bound.
#[derive(Debug, Clone, PartialEq)]
pub struct U1TeBDStepResult {
    pub state: ComplexU1Mps,
    pub time_step: f64,
    pub splitting_order: u8,
    pub layer_parities: Vec<u8>,
    pub layer_bond_indices: Vec<Vec<usize>>,
    pub gate_receipts: Vec<U1TeBDGateReceipt>,
    pub truncation_ledger: U1MpsTruncationLedger,
    pub input_norm_squared: f64,
    pub output_norm_squared: f64,
    pub input_total_charge: i32,
    pub output_total_charge: i32,
    pub output_selection_rule_residual: f64,
    pub global_l2_bound_qualified: bool,
    pub time_discretization_error_qualified: bool,
}

impl U1TeBDStrangStepper {
    /// Construct a fixed nearest-neighbour Hamiltonian decomposition.
    ///
    /// Every bond index must be unique and in range. Terms are stored in
    /// deterministic ascending bond order.
    pub fn new(
        site_count: usize,
        mut terms: Vec<U1TeBDHamiltonianTerm>,
    ) -> Result<Self, U1MpsError> {
        if site_count == 0 {
            return Err(U1MpsError::EmptyChain);
        }
        for term in &terms {
            if term.bond_index >= site_count.saturating_sub(1) {
                return Err(U1MpsError::TeBDGateOutOfRange {
                    bond_index: term.bond_index,
                    site_count,
                });
            }
        }
        terms.sort_by_key(U1TeBDHamiltonianTerm::bond_index);
        for pair in terms.windows(2) {
            if pair[0].bond_index == pair[1].bond_index {
                return Err(U1MpsError::DuplicateTeBDGate {
                    bond_index: pair[0].bond_index,
                });
            }
        }
        Ok(Self { site_count, terms })
    }

    pub const fn site_count(&self) -> usize {
        self.site_count
    }

    pub fn terms(&self) -> &[U1TeBDHamiltonianTerm] {
        &self.terms
    }

    /// Apply one symmetric second-order even/odd/even step.
    ///
    /// Trotter error is intentionally separate from the compression ledger.
    /// Callers should establish time-discretization convergence independently.
    pub fn apply(
        &self,
        input: &ComplexU1Mps,
        dt: f64,
        policy: U1TwoSiteTruncationPolicy,
    ) -> Result<U1TeBDStepResult, U1MpsError> {
        self.apply_with_ledger(
            input,
            dt,
            policy,
            U1MpsTruncationLedger::default(),
        )
    }

    /// Apply one Strang step while continuing an existing error ledger.
    ///
    /// Pass the ledger from the prior time step to keep the global state-vector
    /// bound cumulative across the full evolution. A fresh ledger is a new
    /// qualification segment and does not include earlier time steps.
    pub fn apply_with_ledger(
        &self,
        input: &ComplexU1Mps,
        dt: f64,
        policy: U1TwoSiteTruncationPolicy,
        initial_ledger: U1MpsTruncationLedger,
    ) -> Result<U1TeBDStepResult, U1MpsError> {
        if !dt.is_finite() {
            return Err(U1MpsError::NonFiniteContraction("TEBD Strang time step"));
        }
        if input.site_count() != self.site_count {
            return Err(U1MpsError::TeBDLayerSiteCountMismatch {
                layer_sites: self.site_count,
                state_sites: input.site_count(),
            });
        }

        let mut even_half_gates = Vec::new();
        let mut odd_full_gates = Vec::new();
        for term in &self.terms {
            let gate = if term.bond_index % 2 == 0 {
                term.hamiltonian.evolution_gate(0.5 * dt, policy.tolerance)?
            } else {
                term.hamiltonian.evolution_gate(dt, policy.tolerance)?
            };
            let entry = U1TeBDGate::new(term.bond_index, gate);
            if term.bond_index % 2 == 0 {
                even_half_gates.push(entry);
            } else {
                odd_full_gates.push(entry);
            }
        }

        let even_first = U1TeBDLayer::new(self.site_count, 0, even_half_gates.clone())?;
        let odd = U1TeBDLayer::new(self.site_count, 1, odd_full_gates)?;
        let even_last = U1TeBDLayer::new(self.site_count, 0, even_half_gates)?;

        let first = even_first.apply_with_ledger(input, policy, initial_ledger)?;
        let second = odd.apply_with_ledger(
            &first.state,
            policy,
            first.truncation_ledger.clone(),
        )?;
        let third = even_last.apply_with_ledger(
            &second.state,
            policy,
            second.truncation_ledger.clone(),
        )?;

        let mut gate_receipts = first.gate_receipts;
        gate_receipts.extend(second.gate_receipts);
        gate_receipts.extend(third.gate_receipts);
        let layer_bond_indices = vec![
            even_first.bond_indices(),
            odd.bond_indices(),
            even_last.bond_indices(),
        ];
        let input_norm_squared = input.norm_squared(policy.tolerance)?;
        let output_norm_squared = third.state.norm_squared(policy.tolerance)?;
        let output_selection_rule_residual = third.state.selection_rule_residual();
        let output_total_charge = third.state.total_charge();
        let global_l2_bound_qualified =
            third.truncation_ledger.global_state_error_bound_qualified;

        Ok(U1TeBDStepResult {
            state: third.state,
            time_step: dt,
            splitting_order: 2,
            layer_parities: vec![0, 1, 0],
            layer_bond_indices,
            gate_receipts,
            truncation_ledger: third.truncation_ledger,
            input_norm_squared,
            output_norm_squared,
            input_total_charge: input.total_charge(),
            output_total_charge,
            output_selection_rule_residual,
            global_l2_bound_qualified,
            time_discretization_error_qualified: false,
        })
    }
}


#[cfg(test)]
mod tests {
    use super::{
        ComplexU1Mps, U1CanonicalGaugeReceipt, U1MpsError, U1MpsTensor, U1MpsTruncationLedger,
        U1TeBDGate, U1TeBDGateReceipt, U1TeBDHamiltonianTerm, U1TeBDLayer,
        U1TeBDStepResult, U1TeBDStrangStepper, U1TwoSiteGate, U1TwoSiteHamiltonian,
        U1TwoSiteState, U1TwoSiteBudgetedSplit, U1TwoSiteErrorBudgetPolicy,
        U1TwoSiteTruncationPolicy,
        left_isometry_residual, right_isometry_residual,
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
        assert!(state.maximum_bond_dimension() <= 2);
        assert_eq!(state.dense_svd_rank_by_charge_and_cut().unwrap().len(), 3);
        assert!(state.dense_reconstruction_max_residual().unwrap() <= 1.0e-12);
        assert!((state.norm_squared(1.0e-12).unwrap() - 0.625).abs() <= 1.0e-12);
        let reconstructed = state.to_dense(8).unwrap();
        for (actual, expected) in reconstructed.iter().zip(dense.iter()) {
            assert!((*actual - *expected).norm() <= 1.0e-12);
        }
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
        let tiny: f64 = 1.0e-18;
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
        let dense_vec = state.to_dense(2).unwrap();
        let dense_from_state: [Complex64; 4] = dense_vec.try_into().unwrap();
        assert_eq!(dense, dense_from_state);
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
        let two_site = U1TwoSiteState::from_adjacent_tensors(
            &product.tensors()[0],
            &product.tensors()[1],
            1.0e-12,
        )
        .unwrap();
        assert_eq!(
            two_site.to_dense_two_qubit().unwrap(),
            [
                complex_zero(),
                Complex64::new(1.0, 0.0),
                complex_zero(),
                complex_zero(),
            ]
        );

        let identity = two_site
            .apply_gate(&U1TwoSiteGate::identity(), 1.0e-12)
            .unwrap();
        assert_eq!(
            identity.to_dense_two_qubit().unwrap(),
            two_site.to_dense_two_qubit().unwrap()
        );

        let swapped = two_site
            .apply_gate(&U1TwoSiteGate::swap(), 1.0e-12)
            .unwrap();
        assert_eq!(
            swapped.to_dense_two_qubit().unwrap(),
            [
                complex_zero(),
                complex_zero(),
                Complex64::new(1.0, 0.0),
                complex_zero(),
            ]
        );
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
        assert!((dense[1] - Complex64::new(0.5, 0.5)).norm() <= 1.0e-12);
        assert!((dense[2] - Complex64::new(0.5, 0.5)).norm() <= 1.0e-12);
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

    #[test]
    fn exact_charge_block_split_rebuilds_phase_gated_entangled_w_state() {
        let root_two_thirds = (2.0_f64 / 3.0).sqrt();
        let root_one_third = (1.0_f64 / 3.0).sqrt();
        let inverse_root_two = 1.0 / 2.0_f64.sqrt();

        let mut first_data = vec![complex_zero(); 4];
        first_data[0] = Complex64::new(root_two_thirds, 0.0);
        first_data[3] = Complex64::new(root_one_third, 0.0);
        let first = U1MpsTensor::new(vec![0], vec![0, 1], first_data).unwrap();

        let mut second_data = vec![complex_zero(); 8];
        second_data[0] = Complex64::new(inverse_root_two, 0.0);
        second_data[3] = Complex64::new(inverse_root_two, 0.0);
        second_data[5] = Complex64::new(1.0, 0.0);
        let second = U1MpsTensor::new(vec![0, 1], vec![0, 1], second_data).unwrap();

        let mut third_data = vec![complex_zero(); 4];
        third_data[1] = Complex64::new(1.0, 0.0);
        third_data[2] = Complex64::new(1.0, 0.0);
        let third = U1MpsTensor::new(vec![0, 1], vec![1], third_data).unwrap();

        let initial = ComplexU1Mps::new(
            vec![first.clone(), second.clone(), third.clone()],
            1,
            1.0e-12,
        )
        .unwrap();
        let two_site = U1TwoSiteState::from_adjacent_tensors(&first, &second, 1.0e-12).unwrap();

        let mut phase_gate = [[complex_zero(); 4]; 4];
        phase_gate[0][0] = Complex64::new(1.0, 0.0);
        phase_gate[1][1] = Complex64::new(1.0, 0.0);
        phase_gate[2][2] = Complex64::new(-1.0, 0.0);
        phase_gate[3][3] = Complex64::new(1.0, 0.0);
        let gate = U1TwoSiteGate::new(phase_gate, 1.0e-12).unwrap();

        let evolved_two_site = two_site.apply_gate(&gate, 1.0e-12).unwrap();
        let split = evolved_two_site.split_exact(1.0e-12).unwrap();
        assert_eq!(split.rank_by_charge, vec![(0, 1), (1, 1)]);
        assert_eq!(split.retained_singular_values, 2);
        assert!(split.reconstruction_max_residual <= 1.0e-12);
        assert!(split.reconstruction_l2_residual <= 1.0e-12);

        let rebuilt = ComplexU1Mps::new(
            vec![split.left_tensor, split.right_tensor, third],
            1,
            1.0e-12,
        )
        .unwrap();
        assert!((rebuilt.norm_squared(1.0e-12).unwrap() - 1.0).abs() <= 1.0e-12);

        let mut expected = initial.to_dense(4).unwrap();
        expected[4] = -expected[4];
        let actual = rebuilt.to_dense(4).unwrap();
        for index in 0..expected.len() {
            assert!(
                (actual[index] - expected[index]).norm() <= 1.0e-10,
                "basis={index}, actual={:?}, expected={:?}",
                actual[index],
                expected[index]
            );
        }
    }

    #[test]
    fn exact_two_site_split_retains_tiny_positive_schmidt_amplitude() {
        let tiny: f64 = 1.0e-18;
        let large = (1.0 - tiny * tiny).sqrt();
        let state = U1TwoSiteState::new(
            vec![0],
            vec![1],
            vec![
                complex_zero(),
                Complex64::new(large, 0.0),
                Complex64::new(tiny, 0.0),
                complex_zero(),
            ],
            1.0e-12,
        )
        .unwrap();

        let split = state.split_exact(1.0e-12).unwrap();
        assert_eq!(split.retained_singular_values, 2);
        assert!(split.rank_by_charge.contains(&(0, 1)));
        assert!(split.rank_by_charge.contains(&(1, 1)));
        assert!(split.reconstruction_max_residual <= 1.0e-15);
        let small_amplitude = split.left_tensor.get(0, 1, 1) * split.right_tensor.get(1, 0, 0);
        assert!((small_amplitude.re - tiny).abs() <= 1.0e-30);
    }

    #[test]
    fn exact_two_site_split_omits_only_exact_zero_singular_directions() {
        // The q=0 block is [[1, 1], [1, 1]], with one exact zero
        // singular value and one positive singular value.
        let mut entries = vec![complex_zero(); 2 * 4 * 2];
        entries[0] = Complex64::new(1.0, 0.0);
        entries[1] = Complex64::new(1.0, 0.0);
        entries[8] = Complex64::new(1.0, 0.0);
        entries[9] = Complex64::new(1.0, 0.0);
        let state = U1TwoSiteState::new(vec![0, 0], vec![0, 0], entries, 1.0e-12).unwrap();
        let split = state.split_exact(1.0e-12).unwrap();
        assert_eq!(split.rank_by_charge, vec![(0, 1)]);
        assert_eq!(split.retained_singular_values, 1);
        assert_eq!(split.exact_zero_singular_values_omitted, 1);
        assert!(split.reconstruction_max_residual <= 1.0e-12);
    }

    #[test]
    fn exact_two_site_split_rejects_zero_tensor_and_invalid_tolerance() {
        let zero = U1TwoSiteState::new(vec![0], vec![0], vec![complex_zero(); 4], 1.0e-12).unwrap();
        assert_eq!(zero.split_exact(1.0e-12), Err(U1MpsError::ZeroDenseState));
        assert_eq!(
            zero.split_exact(0.0),
            Err(U1MpsError::InvalidTolerance(0.0))
        );
    }

    fn single_excitation_pair(large: f64, small: f64) -> U1TwoSiteState {
        U1TwoSiteState::new(
            vec![0],
            vec![1],
            vec![
                complex_zero(),
                Complex64::new(large, 0.0),
                Complex64::new(small, 0.0),
                complex_zero(),
            ],
            1.0e-12,
        )
        .unwrap()
    }

    #[test]
    fn truncated_split_applies_global_bond_cap_and_reports_discarded_weight() {
        let state = single_excitation_pair(0.8, 0.6);
        let result = state
            .split_truncated(U1TwoSiteTruncationPolicy {
                max_bond_dimension: 1,
                relative_singular_value_cutoff: 0.0,
                tolerance: 1.0e-12,
            })
            .unwrap();

        assert_eq!(result.total_singular_value_count, 2);
        assert_eq!(result.retained_singular_value_count, 1);
        assert_eq!(result.bond_cap_discarded_singular_value_count, 1);
        assert_eq!(result.cutoff_discarded_singular_value_count, 0);
        assert!(result.bond_cap_binding);
        assert!((result.total_schmidt_weight - 1.0).abs() <= 1.0e-12);
        assert!((result.retained_schmidt_weight - 0.64).abs() <= 1.0e-12);
        assert!((result.bond_cap_discarded_weight - 0.36).abs() <= 1.0e-12);
        assert!((result.discarded_weight - 0.36).abs() <= 1.0e-12);
        assert!((result.relative_discarded_weight - 0.36).abs() <= 1.0e-12);
        assert!((result.pre_truncation_norm_squared - 1.0).abs() <= 1.0e-12);
        assert!((result.post_truncation_norm_squared - 0.64).abs() <= 1.0e-12);
        assert!((result.renormalization_factor_to_restore_original_norm - 1.25).abs() <= 1.0e-12);
        assert!(result.reconstruction_l2_residual >= 0.6 - 1.0e-12);
    }

    #[test]
    fn truncated_split_relative_cutoff_is_separate_from_bond_cap() {
        let state = single_excitation_pair(0.8, 0.6);
        let result = state
            .split_truncated(U1TwoSiteTruncationPolicy {
                max_bond_dimension: 8,
                relative_singular_value_cutoff: 0.8,
                tolerance: 1.0e-12,
            })
            .unwrap();
        assert_eq!(result.retained_singular_value_count, 1);
        assert_eq!(result.cutoff_discarded_singular_value_count, 1);
        assert_eq!(result.bond_cap_discarded_singular_value_count, 0);
        assert!(!result.bond_cap_binding);
        assert!((result.cutoff_discarded_weight - 0.36).abs() <= 1.0e-12);
        assert!(result.bond_cap_discarded_weight.abs() <= 1.0e-12);
    }

    #[test]
    fn truncated_split_keeps_u1_charge_and_does_not_renormalize() {
        let state = single_excitation_pair(0.8, 0.6);
        let result = state
            .split_truncated(U1TwoSiteTruncationPolicy {
                max_bond_dimension: 1,
                relative_singular_value_cutoff: 0.0,
                tolerance: 1.0e-12,
            })
            .unwrap();
        assert_eq!(result.left_tensor.right_dimension(), 1);
        assert_eq!(result.right_tensor.left_dimension(), 1);
        assert_eq!(
            result.left_tensor.right_charges(),
            result.right_tensor.left_charges()
        );
        let mut norm_squared = 0.0;
        for first_physical in 0..2 {
            for second_physical in 0..2 {
                let mut amplitude = Complex64::new(0.0, 0.0);
                for middle in 0..result.left_tensor.right_dimension() {
                    amplitude += result.left_tensor.get(0, first_physical, middle)
                        * result.right_tensor.get(middle, second_physical, 0);
                }
                norm_squared += amplitude.norm_sqr();
            }
        }
        assert!(norm_squared <= 1.0 + 1.0e-12);
        assert!((norm_squared - 0.64).abs() <= 1.0e-12);
    }

    #[test]
    fn truncation_ledger_sums_per_gate_l2_residuals_conservatively() {
        let state = single_excitation_pair(0.8, 0.6);
        let result = state
            .split_truncated(U1TwoSiteTruncationPolicy {
                max_bond_dimension: 1,
                relative_singular_value_cutoff: 0.0,
                tolerance: 1.0e-12,
            })
            .unwrap();
        let mut ledger = U1MpsTruncationLedger::default();
        ledger.record(&result).unwrap();
        ledger.record(&result).unwrap();
        assert_eq!(ledger.gate_count, 2);
        assert!((ledger.accumulated_discarded_weight - 0.72).abs() <= 1.0e-12);
        assert!((ledger.cumulative_local_l2_error_bound - 1.2).abs() <= 1.0e-12);
        assert!(!ledger.global_state_error_bound_qualified);
        assert!((ledger.maximum_single_gate_relative_discarded_weight - 0.36).abs() <= 1.0e-12);
    }

    #[test]
    fn truncation_ledger_uses_measured_residual_for_tolerated_charge_leakage() {
        // |00> is outside the declared total-charge-one sector, but its
        // amplitude is below the constructor tolerance. It contributes no
        // Schmidt weight to the charge-block SVD, yet the reconstruction
        // residual must still be counted by the ledger.
        let state = U1TwoSiteState::new(
            vec![0],
            vec![1],
            vec![
                Complex64::new(1.0e-13, 0.0),
                Complex64::new(1.0, 0.0),
                complex_zero(),
                complex_zero(),
            ],
            1.0e-12,
        )
        .unwrap();
        let diagnostics = state
            .split_truncated(U1TwoSiteTruncationPolicy {
                max_bond_dimension: 2,
                relative_singular_value_cutoff: 0.0,
                tolerance: 1.0e-12,
            })
            .unwrap();
        assert_eq!(diagnostics.discarded_weight, 0.0);
        assert!(diagnostics.reconstruction_l2_residual >= 1.0e-13);

        let mut ledger = U1MpsTruncationLedger::default();
        ledger.record(&diagnostics).unwrap();
        assert!(
            ledger.cumulative_local_l2_error_bound >= diagnostics.reconstruction_l2_residual
        );
        assert!(!ledger.global_state_error_bound_qualified);
    }

    #[test]
    fn truncation_ledger_rejects_nonfinite_receipts() {
        let mut ledger = U1MpsTruncationLedger::default();
        let mut state = U1TwoSiteState::new(
            vec![0],
            vec![1],
            vec![
                complex_zero(),
                Complex64::new(1.0, 0.0),
                complex_zero(),
                complex_zero(),
            ],
            1.0e-12,
        )
        .unwrap()
        .split_truncated(U1TwoSiteTruncationPolicy {
            max_bond_dimension: 2,
            relative_singular_value_cutoff: 0.0,
            tolerance: 1.0e-12,
        })
        .unwrap();
        state.discarded_weight = f64::NAN;
        assert!(ledger.record(&state).is_err());
    }

    #[test]
    fn truncated_split_rejects_invalid_policy_fields() {
        let state = single_excitation_pair(0.8, 0.6);
        for policy in [
            U1TwoSiteTruncationPolicy {
                max_bond_dimension: 0,
                relative_singular_value_cutoff: 0.0,
                tolerance: 1.0e-12,
            },
            U1TwoSiteTruncationPolicy {
                max_bond_dimension: 1,
                relative_singular_value_cutoff: -0.1,
                tolerance: 1.0e-12,
            },
            U1TwoSiteTruncationPolicy {
                max_bond_dimension: 1,
                relative_singular_value_cutoff: 1.1,
                tolerance: 1.0e-12,
            },
            U1TwoSiteTruncationPolicy {
                max_bond_dimension: 1,
                relative_singular_value_cutoff: 0.0,
                tolerance: f64::NAN,
            },
        ] {
            assert!(matches!(
                state.split_truncated(policy),
                Err(U1MpsError::InvalidTruncationPolicy { .. })
            ));
        }
    }

    #[test]
    fn tebd_even_layer_executes_sorted_disjoint_swaps_and_preserves_charge() {
        let input = ComplexU1Mps::product_state(&[true, false, true, false]).unwrap();
        let layer = U1TeBDLayer::new(
            4,
            0,
            vec![
                U1TeBDGate::new(2, U1TwoSiteGate::swap()),
                U1TeBDGate::new(0, U1TwoSiteGate::swap()),
            ],
        )
        .unwrap();
        assert_eq!(layer.bond_indices(), vec![0, 2]);

        let result = layer
            .apply(
                &input,
                U1TwoSiteTruncationPolicy {
                    max_bond_dimension: 4,
                    relative_singular_value_cutoff: 0.0,
                    tolerance: 1.0e-12,
                },
            )
            .unwrap();
        assert_eq!(result.input_total_charge, 2);
        assert_eq!(result.output_total_charge, 2);
        assert!(result.output_selection_rule_residual <= 1.0e-12);
        assert!((result.input_norm_squared - 1.0).abs() <= 1.0e-12);
        assert!((result.output_norm_squared - 1.0).abs() <= 1.0e-10);
        assert_eq!(result.truncation_ledger.gate_count, 2);
        assert_eq!(result.gate_truncations.len(), 2);
        assert_eq!(result.canonical_gauge_receipts.len(), 2);
        assert!(result.global_l2_bound_qualified);
        assert!(
            result
                .truncation_ledger
                .global_state_error_bound_upper
                .is_some()
        );

        let dense = result.state.to_dense(8).unwrap();
        assert!((dense[5] - Complex64::new(1.0, 0.0)).norm() <= 1.0e-10);
        for (index, amplitude) in dense.iter().enumerate() {
            if index != 5 {
                assert!(amplitude.norm() <= 1.0e-10);
            }
        }
    }

    #[test]
    fn tebd_odd_layer_executes_one_gate_and_tracks_local_discarded_weight() {
        let input = ComplexU1Mps::product_state(&[false, true, false]).unwrap();
        let layer =
            U1TeBDLayer::new(3, 1, vec![U1TeBDGate::new(1, U1TwoSiteGate::swap())]).unwrap();
        let result = layer
            .apply(
                &input,
                U1TwoSiteTruncationPolicy {
                    max_bond_dimension: 1,
                    relative_singular_value_cutoff: 0.0,
                    tolerance: 1.0e-12,
                },
            )
            .unwrap();
        assert_eq!(result.bond_indices, vec![1]);
        assert_eq!(result.truncation_ledger.gate_count, 1);
        assert_eq!(result.output_total_charge, 1);
        assert!((result.output_norm_squared - result.input_norm_squared).abs() <= 1.0e-10);
        assert!(result.gate_truncations[0].discarded_weight <= 1.0e-12);
        assert_eq!(result.canonical_gauge_receipts.len(), 1);
        assert!(result.global_l2_bound_qualified);
        assert!(
            result
                .truncation_ledger
                .global_state_error_bound_upper
                .is_some()
        );
        let dense = result.state.to_dense(4).unwrap();
        assert!((dense[1] - Complex64::new(1.0, 0.0)).norm() <= 1.0e-10);
    }

    #[test]
    fn tebd_layer_rejects_bad_parity_duplicate_and_out_of_range_bonds() {
        assert!(matches!(
            U1TeBDLayer::new(4, 2, Vec::new()),
            Err(U1MpsError::InvalidTeBDParity(2))
        ));
        assert!(matches!(
            U1TeBDLayer::new(4, 0, vec![U1TeBDGate::new(1, U1TwoSiteGate::swap())]),
            Err(U1MpsError::TeBDGateWrongParity { .. })
        ));
        assert!(matches!(
            U1TeBDLayer::new(
                4,
                0,
                vec![
                    U1TeBDGate::new(0, U1TwoSiteGate::swap()),
                    U1TeBDGate::new(0, U1TwoSiteGate::identity()),
                ]
            ),
            Err(U1MpsError::DuplicateTeBDGate { bond_index: 0 })
        ));
        assert!(matches!(
            U1TeBDLayer::new(4, 0, vec![U1TeBDGate::new(4, U1TwoSiteGate::swap())]),
            Err(U1MpsError::TeBDGateOutOfRange { .. })
        ));
    }

    #[test]
    fn tebd_layer_rejects_state_with_mismatched_site_count() {
        let input = ComplexU1Mps::product_state(&[false, true, false]).unwrap();
        let layer =
            U1TeBDLayer::new(4, 0, vec![U1TeBDGate::new(0, U1TwoSiteGate::swap())]).unwrap();
        assert!(matches!(
            layer.apply(
                &input,
                U1TwoSiteTruncationPolicy {
                    max_bond_dimension: 4,
                    relative_singular_value_cutoff: 0.0,
                    tolerance: 1.0e-12,
                }
            ),
            Err(U1MpsError::TeBDLayerSiteCountMismatch {
                layer_sites: 4,
                state_sites: 3
            })
        ));
    }

    fn four_site_two_excitation_dense_fixture() -> Vec<Complex64> {
        let mut dense = vec![complex_zero(); 16];
        dense[3] = Complex64::new(0.3, 0.1);
        dense[5] = Complex64::new(-0.2, 0.4);
        dense[6] = Complex64::new(0.5, -0.1);
        dense[9] = Complex64::new(0.1, 0.2);
        dense[10] = Complex64::new(-0.3, -0.2);
        dense[12] = Complex64::new(0.4, 0.3);
        let norm = dense
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt();
        for value in &mut dense {
            *value /= norm;
        }
        dense
    }

    #[test]
    fn canonicalization_preserves_dense_state_and_qualifies_all_two_site_centers() {
        let dense = four_site_two_excitation_dense_fixture();
        let original = ComplexU1Mps::from_dense_fixed_charge(&dense, 4, 2, 1.0e-12).unwrap();
        let original_norm = original.norm_squared(1.0e-12).unwrap();

        for bond in 0..3 {
            let mut canonical = original.clone();
            let receipt = canonical.canonicalize_for_bond(bond, 1.0e-10).unwrap();
            assert!(
                receipt.qualifies(1.0e-10),
                "bond={bond}, receipt={receipt:?}"
            );
            assert_eq!(receipt.bond, bond);
            assert_eq!(receipt.site_count, 4);
            assert_eq!(receipt.total_charge, 2);
            assert_eq!(receipt.left_isometry_residuals.len(), bond);
            assert_eq!(receipt.right_isometry_residuals.len(), 2 - bond);
            assert!(receipt.maximum_left_isometry_residual <= 1.0e-10);
            assert!(receipt.maximum_right_isometry_residual <= 1.0e-10);
            assert!(receipt.factorization_max_residual <= 1.0e-10);
            assert!(receipt.state_reconstruction_l2_residual <= 1.0e-9);
            assert!(receipt.relative_norm_drift <= 1.0e-10);
            assert!(receipt.exterior_operator_norm_product_upper_bound() >= 1.0);
            assert!(
                receipt
                    .exterior_operator_norm_product_upper_bound()
                    .is_finite()
            );

            let reconstructed = canonical.to_dense(8).unwrap();
            for (index, (actual, expected)) in reconstructed.iter().zip(dense.iter()).enumerate() {
                assert!(
                    (*actual - *expected).norm() <= 1.0e-9,
                    "bond={bond}, basis={index}, actual={actual:?}, expected={expected:?}"
                );
            }
            assert!((canonical.norm_squared(1.0e-12).unwrap() - original_norm).abs() <= 1.0e-10);
            assert_eq!(canonical.total_charge(), original.total_charge());
            assert!(canonical.selection_rule_residual() <= 1.0e-12);
        }
    }

    #[test]
    fn canonicalization_receipt_is_bound_to_measured_external_isometries() {
        let dense = four_site_two_excitation_dense_fixture();
        let mut state = ComplexU1Mps::from_dense_fixed_charge(&dense, 4, 2, 1.0e-12).unwrap();
        let receipt: U1CanonicalGaugeReceipt = state.canonicalize_for_bond(1, 1.0e-10).unwrap();

        for &(site, residual) in &receipt.left_isometry_residuals {
            assert_eq!(site, 0);
            assert!(
                (left_isometry_residual(&state.tensors()[site]).unwrap() - residual).abs()
                    <= 1.0e-14
            );
        }
        for &(site, residual) in &receipt.right_isometry_residuals {
            assert_eq!(site, 3);
            assert!(
                (right_isometry_residual(&state.tensors()[site]).unwrap() - residual).abs()
                    <= 1.0e-14
            );
        }
        assert!(receipt.qualifies(1.0e-10));
        assert!(!receipt.qualifies(f64::NAN));
        assert!(!receipt.qualifies(0.0));

        let mut missing_site = receipt.clone();
        missing_site.right_isometry_residuals.pop();
        assert!(!missing_site.qualifies(1.0e-10));
        let mut misbound_site = receipt.clone();
        misbound_site.right_isometry_residuals[0].0 = 2;
        assert!(!misbound_site.qualifies(1.0e-10));
    }

    #[test]
    fn mps_overlap_matches_norm_and_survives_gauge_rebasing() {
        let dense = four_site_two_excitation_dense_fixture();
        let original = ComplexU1Mps::from_dense_fixed_charge(&dense, 4, 2, 1.0e-12).unwrap();
        let mut rebased = original.clone();
        let receipt = rebased.canonicalize_for_bond(2, 1.0e-10).unwrap();
        let overlap = original.inner_product(&rebased).unwrap();
        let original_norm = original.norm_squared(1.0e-12).unwrap();
        let rebased_norm = rebased.norm_squared(1.0e-12).unwrap();

        assert!(receipt.qualifies(1.0e-10));
        assert!((overlap.re - original_norm).abs() <= 1.0e-10);
        assert!(overlap.im.abs() <= 1.0e-10);
        assert!((original_norm - rebased_norm).abs() <= 1.0e-10);
        assert!(original.inner_product(&original).unwrap().im.abs() <= 1.0e-12);
    }

    #[test]
    fn canonicalization_rejects_invalid_center_without_mutating_state() {
        let dense = four_site_two_excitation_dense_fixture();
        let mut state = ComplexU1Mps::from_dense_fixed_charge(&dense, 4, 2, 1.0e-12).unwrap();
        let before = state.to_dense(8).unwrap();

        assert_eq!(
            state.canonicalize_for_bond(3, 1.0e-10),
            Err(U1MpsError::TwoSiteShapeMismatch)
        );
        assert!(matches!(
            state.canonicalize_for_bond(0, f64::NAN),
            Err(U1MpsError::InvalidTolerance(value)) if value.is_nan()
        ));
        let after = state.to_dense(8).unwrap();
        for (actual, expected) in after.iter().zip(before.iter()) {
            assert!((*actual - *expected).norm() <= 1.0e-14);
        }
    }

    fn canonical_w_state() -> ComplexU1Mps {
        let amplitude = 1.0 / 3.0_f64.sqrt();
        let mut dense = vec![Complex64::new(0.0, 0.0); 8];
        dense[1] = Complex64::new(amplitude, 0.0);
        dense[2] = Complex64::new(amplitude, 0.0);
        dense[4] = Complex64::new(amplitude, 0.0);
        ComplexU1Mps::from_dense_fixed_charge(&dense, 3, 1, 1.0e-12).unwrap()
    }

    #[test]
    fn mixed_canonicalization_qualifies_both_two_site_centers_and_preserves_dense_state() {
        let original = canonical_w_state();
        let expected_dense = original.to_dense(4).unwrap();
        let expected_norm = original.norm_squared(1.0e-12).unwrap();

        for bond in [0, 1] {
            let mut candidate = original.clone();
            let receipt: U1CanonicalGaugeReceipt =
                candidate.canonicalize_for_bond(bond, 1.0e-10).unwrap();

            assert_eq!(receipt.bond, bond);
            assert_eq!(receipt.site_count, 3);
            assert_eq!(receipt.total_charge, 1);
            assert!(receipt.qualifies(1.0e-10), "receipt={receipt:?}");
            assert_eq!(
                receipt
                    .left_isometry_residuals
                    .iter()
                    .map(|(site, _)| *site)
                    .collect::<Vec<_>>(),
                (0..bond).collect::<Vec<_>>()
            );
            assert_eq!(
                receipt
                    .right_isometry_residuals
                    .iter()
                    .map(|(site, _)| *site)
                    .collect::<Vec<_>>(),
                ((bond + 2)..3).collect::<Vec<_>>()
            );
            assert!(receipt.left_exterior_operator_norm_upper_bound >= 1.0);
            assert!(receipt.right_exterior_operator_norm_upper_bound >= 1.0);
            assert!(receipt.state_reconstruction_l2_residual <= 1.0e-10);
            assert!((receipt.output_norm_squared - expected_norm).abs() <= 1.0e-10);

            let actual_dense = candidate.to_dense(4).unwrap();
            for (actual, expected) in actual_dense.iter().zip(expected_dense.iter()) {
                assert!((*actual - *expected).norm() <= 1.0e-10);
            }
        }
    }

    #[test]
    fn canonical_gauge_receipt_fails_closed_for_invalid_tolerance_or_bond() {
        let mut state = canonical_w_state();
        let original_tensors = state.tensors().to_vec();
        assert!(matches!(
            state.canonicalize_for_bond(0, 0.0),
            Err(U1MpsError::InvalidTolerance(0.0))
        ));
        assert!(matches!(
            state.canonicalize_for_bond(2, 1.0e-12),
            Err(U1MpsError::TwoSiteShapeMismatch)
        ));
        assert_eq!(state.tensors(), original_tensors.as_slice());
        let receipt = U1CanonicalGaugeReceipt {
            bond: 0,
            site_count: 3,
            total_charge: 1,
            left_isometry_residuals: Vec::new(),
            right_isometry_residuals: Vec::new(),
            maximum_left_isometry_residual: 0.0,
            maximum_right_isometry_residual: 0.0,
            left_exterior_operator_norm_upper_bound: f64::NAN,
            right_exterior_operator_norm_upper_bound: 1.0,
            factorization_max_residual: 0.0,
            factorization_l2_residual: 0.0,
            state_reconstruction_l2_residual: 0.0,
            exact_zero_singular_values_omitted: 0,
            selection_rule_residual: 0.0,
            input_norm_squared: 1.0,
            output_norm_squared: 1.0,
            relative_norm_drift: 0.0,
            qualified: true,
        };
        assert!(!receipt.qualifies(1.0e-12));
    }

    #[test]
    fn canonicalization_is_idempotent_at_a_fixed_two_site_center() {
        let mut state = canonical_w_state();
        let first = state.canonicalize_for_bond(1, 1.0e-10).unwrap();
        let dense_after_first = state.to_dense(4).unwrap();
        let second = state.canonicalize_for_bond(1, 1.0e-10).unwrap();
        let dense_after_second = state.to_dense(4).unwrap();
        assert!(first.qualifies(1.0e-10));
        assert!(second.qualifies(1.0e-10));
        for (first_value, second_value) in dense_after_first.iter().zip(dense_after_second.iter()) {
            assert!((*first_value - *second_value).norm() <= 1.0e-10);
        }
    }

    #[test]
    fn tebd_global_error_bound_matches_dense_two_site_discarded_component() {
        let mut dense = vec![complex_zero(); 4];
        dense[1] = Complex64::new(0.8, 0.0);
        dense[2] = Complex64::new(0.6, 0.0);
        let input = ComplexU1Mps::from_dense_fixed_charge(&dense, 2, 1, 1.0e-12).unwrap();
        let layer =
            U1TeBDLayer::new(2, 0, vec![U1TeBDGate::new(0, U1TwoSiteGate::identity())]).unwrap();
        let result = layer
            .apply(
                &input,
                U1TwoSiteTruncationPolicy {
                    max_bond_dimension: 1,
                    relative_singular_value_cutoff: 0.0,
                    tolerance: 1.0e-10,
                },
            )
            .unwrap();

        assert!(result.global_l2_bound_qualified);
        assert_eq!(result.canonical_gauge_receipts.len(), 1);
        assert!(result.canonical_gauge_receipts[0].qualifies(1.0e-10));
        assert!((result.gate_truncations[0].discarded_weight - 0.36).abs() <= 1.0e-10);
        let certified_bound = result
            .truncation_ledger
            .global_state_error_bound_upper
            .expect("qualified gauge must produce global error bound");
        assert!((certified_bound - 0.6).abs() <= 1.0e-10);

        let before = input.to_dense(4).unwrap();
        let after = result.state.to_dense(4).unwrap();
        let actual_error = before
            .iter()
            .zip(after.iter())
            .map(|(left, right)| (*left - *right).norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!((actual_error - 0.6).abs() <= 1.0e-10);
        assert!(certified_bound + 1.0e-10 >= actual_error);
        assert!((result.output_norm_squared - 0.64).abs() <= 1.0e-10);
    }

    #[test]
    fn tebd_rejects_a_gate_constructed_with_a_looser_unitarity_tolerance() {
        let mut matrix = [[complex_zero(); 4]; 4];
        for index in 0..4 {
            matrix[index][index] = Complex64::new(1.0, 0.0);
        }
        matrix[1][1] = Complex64::new(1.0 + 1.0e-7, 0.0);
        let loose_gate = U1TwoSiteGate::new(matrix, 1.0e-6).unwrap();
        assert!(loose_gate.unitarity_residual() > 1.0e-10);
        let input = ComplexU1Mps::product_state(&[false, true]).unwrap();
        let layer = U1TeBDLayer::new(2, 0, vec![U1TeBDGate::new(0, loose_gate)]).unwrap();
        assert!(matches!(
            layer.apply(
                &input,
                U1TwoSiteTruncationPolicy {
                    max_bond_dimension: 2,
                    relative_singular_value_cutoff: 0.0,
                    tolerance: 1.0e-10,
                }
            ),
            Err(U1MpsError::GateUnitarityResidual { .. })
        ));
    }

    #[test]
    fn tebd_global_error_receipt_includes_gate_unitarity_defect() {
        let epsilon = 2.5e-8;
        let mut matrix = [[complex_zero(); 4]; 4];
        for index in 0..4 {
            matrix[index][index] = Complex64::new(1.0, 0.0);
        }
        matrix[1][1] = Complex64::new(1.0 + epsilon, 0.0);
        let gate = U1TwoSiteGate::new(matrix, 1.0e-6).unwrap();
        let input = ComplexU1Mps::product_state(&[false, true]).unwrap();
        let layer = U1TeBDLayer::new(2, 0, vec![U1TeBDGate::new(0, gate)]).unwrap();
        let result = layer
            .apply(
                &input,
                U1TwoSiteTruncationPolicy {
                    max_bond_dimension: 2,
                    relative_singular_value_cutoff: 0.0,
                    tolerance: 1.0e-6,
                },
            )
            .unwrap();
        assert!(result.global_l2_bound_qualified);
        assert_eq!(result.gate_receipts.len(), 1);
        let receipt: &U1TeBDGateReceipt = &result.gate_receipts[0];
        assert_eq!(receipt.bond_index, 0);
        assert!(receipt.gate_unitarity_residual > 0.0);
        assert!(receipt.gate_quality_error_upper_bound >= epsilon * (1.0 - 1.0e-6));
        assert!(receipt.global_l2_error_contribution.is_some());

        let before = input.to_dense(4).unwrap();
        let after = result.state.to_dense(4).unwrap();
        let actual_error = before
            .iter()
            .zip(after.iter())
            .map(|(left, right)| (*left - *right).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let bound = result
            .truncation_ledger
            .global_state_error_bound_upper
            .unwrap();
        assert!(bound + 1.0e-12 >= actual_error);
        assert!(
            (result
                .truncation_ledger
                .cumulative_gate_quality_l2_error_bound
                - receipt.gate_quality_error_upper_bound)
                .abs()
                <= 1.0e-12
        );
    }

    #[test]
    fn tebd_global_error_bound_propagates_through_nonunitary_gate_norms() {
        let epsilon = 1.0e-3;
        let mut matrix = [[complex_zero(); 4]; 4];
        for index in 0..4 {
            matrix[index][index] = Complex64::new(1.0, 0.0);
        }
        matrix[2][2] = Complex64::new(1.0 + epsilon, 0.0);
        let gate = U1TwoSiteGate::new(matrix, 1.0e-2).unwrap();
        let input = ComplexU1Mps::product_state(&[true, false, true, false]).unwrap();
        let layer = U1TeBDLayer::new(
            4,
            0,
            vec![
                U1TeBDGate::new(0, gate.clone()),
                U1TeBDGate::new(2, gate),
            ],
        )
        .unwrap();
        let result = layer
            .apply(
                &input,
                U1TwoSiteTruncationPolicy {
                    max_bond_dimension: 4,
                    relative_singular_value_cutoff: 0.0,
                    tolerance: 1.0e-2,
                },
            )
            .unwrap();

        assert!(result.global_l2_bound_qualified);
        assert_eq!(result.truncation_ledger.gate_count, 2);
        assert_eq!(result.gate_receipts.len(), 2);
        let first = &result.gate_receipts[0];
        let second = &result.gate_receipts[1];
        assert!(first.global_l2_error_bound_after_gate.is_some());
        assert!(second.global_l2_error_bound_after_gate.is_some());

        let first_bound = first.global_l2_error_bound_after_gate.unwrap();
        let second_bound = second.global_l2_error_bound_after_gate.unwrap();
        let second_gate_operator_norm =
            (1.0 + second.gate_unitarity_residual).sqrt();
        let propagated_first = first_bound * second_gate_operator_norm;
        let second_increment = second.global_l2_error_contribution.unwrap();
        assert!(second_increment >= 0.0);
        assert!(
            (second_bound - propagated_first - second_increment).abs() <= 1.0e-12,
            "second bound {second_bound}, propagated {propagated_first}, increment {second_increment}"
        );

        let before = input.to_dense(8).unwrap();
        let after = result.state.to_dense(8).unwrap();
        let actual_error = before
            .iter()
            .zip(after.iter())
            .map(|(left, right)| (*left - *right).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let bound = result
            .truncation_ledger
            .global_state_error_bound_upper
            .unwrap();
        assert!(bound + 1.0e-12 >= actual_error);
        let component_sum = result.truncation_ledger.cumulative_canonicalization_l2_error_bound
            + result.truncation_ledger.cumulative_gate_quality_l2_error_bound
            + result.truncation_ledger.cumulative_projection_l2_error_bound;
        assert!((component_sum - bound).abs() <= 1.0e-12);
    }



    #[test]
    fn gate_quality_operator_bound_accounts_for_entrywise_to_spectral_norm_conversion() {
        // In the {|01>, |10>} sector choose G = sqrt(I + a * ones).
        // The max entry of G†G-I is a, while its spectral norm is 2a.
        let a: f64 = 0.02;
        let delta = (1.0 + 2.0 * a).sqrt() - 1.0;
        let mut matrix = [[complex_zero(); 4]; 4];
        matrix[0][0] = Complex64::new(1.0, 0.0);
        matrix[3][3] = Complex64::new(1.0, 0.0);
        matrix[1][1] = Complex64::new(1.0 + 0.5 * delta, 0.0);
        matrix[1][2] = Complex64::new(0.5 * delta, 0.0);
        matrix[2][1] = Complex64::new(0.5 * delta, 0.0);
        matrix[2][2] = Complex64::new(1.0 + 0.5 * delta, 0.0);
        let gate = U1TwoSiteGate::new(matrix, 0.1).unwrap();
        assert!((gate.unitarity_residual() - a).abs() <= 1.0e-12);

        let (operator_norm_upper, quality_error_upper) =
            super::gate_quality_error_bounds(&gate, 1.0).unwrap();
        let actual_operator_norm = (1.0 + 2.0 * a).sqrt();
        assert!(operator_norm_upper >= actual_operator_norm);
        assert!(quality_error_upper >= actual_operator_norm - 1.0);
        assert!(operator_norm_upper.is_finite());
        assert!(quality_error_upper.is_finite());
    }

    #[test]
    fn hamiltonian_projection_error_uses_deltas_own_duhamel_growth_bound() {
        let delta: f64 = 0.05;
        let dt: f64 = 2.0;
        let mut matrix = [[complex_zero(); 4]; 4];
        // A small anti-Hermitian component makes H differ from its Hermitian
        // U(1)-preserving projection by exactly |delta| in Frobenius norm.
        matrix[0][0] = Complex64::new(0.0, delta);
        let hamiltonian = U1TwoSiteHamiltonian::new(matrix, 0.2).unwrap();
        let gate = hamiltonian.evolution_gate(dt, 1.0e-12).unwrap();
        let expected = dt.abs() * delta * (dt.abs() * delta).exp();
        assert!(
            (gate.hamiltonian_projection_error_operator_upper_bound() - expected).abs()
                <= 1.0e-12
        );
        // The true scalar-channel amplitude error is exp(delta*dt)-1.
        let true_error = (delta * dt).exp() - 1.0;
        assert!(gate.hamiltonian_projection_error_operator_upper_bound() >= true_error);
    }

    #[test]
    fn two_site_hamiltonian_exponential_matches_analytic_single_excitation_rabi_gate() {
        let coupling = 0.7;
        let dt = 0.31;
        let mut hamiltonian = [[complex_zero(); 4]; 4];
        hamiltonian[0][0] = Complex64::new(0.2, 0.0);
        hamiltonian[3][3] = Complex64::new(-0.3, 0.0);
        hamiltonian[1][1] = Complex64::new(0.1, 0.0);
        hamiltonian[2][2] = Complex64::new(-0.1, 0.0);
        hamiltonian[1][2] = Complex64::new(coupling, 0.0);
        hamiltonian[2][1] = Complex64::new(coupling, 0.0);
        let model = U1TwoSiteHamiltonian::new(hamiltonian, 1.0e-12).unwrap();
        let gate = model.evolution_gate(dt, 1.0e-12).unwrap();
        let matrix = gate.matrix();
        let difference: f64 = 0.1;
        let radius = (difference * difference + coupling * coupling).sqrt();
        let cosine = (dt * radius).cos();
        let sine_over_radius = (dt * radius).sin() / radius;
        assert!((matrix[0][0] - Complex64::from_polar(1.0, -dt * 0.2)).norm() <= 1.0e-12);
        assert!((matrix[3][3] - Complex64::from_polar(1.0, dt * 0.3)).norm() <= 1.0e-12);
        assert!(
            (matrix[1][2] - Complex64::new(0.0, -sine_over_radius * coupling)).norm()
                <= 1.0e-12
        );
        assert!(
            (matrix[2][1] - Complex64::new(0.0, -sine_over_radius * coupling)).norm()
                <= 1.0e-12
        );
        assert!(
            (matrix[1][1]
                - Complex64::new(cosine, 0.0)
                - Complex64::new(0.0, -sine_over_radius * difference))
                .norm()
                <= 1.0e-12
        );
        assert!(
            (matrix[2][2]
                - Complex64::new(cosine, 0.0)
                + Complex64::new(0.0, -sine_over_radius * difference))
                .norm()
                <= 1.0e-12
        );
        assert!(gate.unitarity_residual() <= 1.0e-12);
        assert_eq!(gate.charge_conservation_residual(), 0.0);
    }

    #[test]
    fn two_site_hamiltonian_rejects_nonhermitian_charge_leaking_and_nonfinite_inputs() {
        let mut nonhermitian = [[complex_zero(); 4]; 4];
        nonhermitian[1][2] = Complex64::new(0.5, 0.2);
        nonhermitian[2][1] = Complex64::new(0.5, 0.2);
        assert!(matches!(
            U1TwoSiteHamiltonian::new(nonhermitian, 1.0e-12),
            Err(U1MpsError::TwoSiteHamiltonianHermiticityResidual { .. })
        ));

        let mut charge_leaking = [[complex_zero(); 4]; 4];
        charge_leaking[0][1] = Complex64::new(0.1, 0.0);
        charge_leaking[1][0] = Complex64::new(0.1, 0.0);
        assert!(matches!(
            U1TwoSiteHamiltonian::new(charge_leaking, 1.0e-12),
            Err(U1MpsError::TwoSiteHamiltonianChargeLeakage { .. })
        ));

        let mut nonfinite = [[complex_zero(); 4]; 4];
        nonfinite[2][2] = Complex64::new(f64::NAN, 0.0);
        assert!(matches!(
            U1TwoSiteHamiltonian::new(nonfinite, 1.0e-12),
            Err(U1MpsError::NonFiniteTensorEntry { .. })
        ));
    }

    #[test]
    fn two_site_hamiltonian_exponential_rejects_nonfinite_time_step() {
        let model = U1TwoSiteHamiltonian::new([[complex_zero(); 4]; 4], 1.0e-12).unwrap();
        assert!(model.evolution_gate(f64::INFINITY, 1.0e-12).is_err());
        assert!(model.evolution_gate(0.1, f64::NAN).is_err());
    }



    fn nearest_neighbour_hopping_term(coupling: f64) -> U1TwoSiteHamiltonian {
        let mut matrix = [[complex_zero(); 4]; 4];
        matrix[1][2] = Complex64::new(coupling, 0.0);
        matrix[2][1] = Complex64::new(coupling, 0.0);
        U1TwoSiteHamiltonian::new(matrix, 1.0e-12).unwrap()
    }

    fn one_excitation_chain_exact(time: f64, coupling: f64) -> Vec<Complex64> {
        let angle = 2.0_f64.sqrt() * coupling * time;
        vec![
            Complex64::new((0.5 * angle).cos().powi(2), 0.0),
            Complex64::new(0.0, -(1.0 / 2.0_f64.sqrt()) * angle.sin()),
            Complex64::new(-(0.5 * angle).sin().powi(2), 0.0),
        ]
    }

    #[test]
    fn strang_stepper_builds_even_odd_even_layers_and_preserves_global_ledger() {
        let stepper = U1TeBDStrangStepper::new(
            3,
            vec![
                U1TeBDHamiltonianTerm::new(1, nearest_neighbour_hopping_term(0.7)),
                U1TeBDHamiltonianTerm::new(0, nearest_neighbour_hopping_term(0.7)),
            ],
        )
        .unwrap();
        assert_eq!(
            stepper
                .terms()
                .iter()
                .map(U1TeBDHamiltonianTerm::bond_index)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        let input = ComplexU1Mps::product_state(&[true, false, false]).unwrap();
        let result = stepper
            .apply(
                &input,
                0.2,
                U1TwoSiteTruncationPolicy {
                    max_bond_dimension: 4,
                    relative_singular_value_cutoff: 0.0,
                    tolerance: 1.0e-10,
                },
            )
            .unwrap();
        assert_eq!(result.splitting_order, 2);
        assert_eq!(result.layer_parities, vec![0, 1, 0]);
        assert_eq!(result.layer_bond_indices, vec![vec![0], vec![1], vec![0]]);
        assert_eq!(result.input_total_charge, 1);
        assert_eq!(result.output_total_charge, 1);
        assert!(result.output_selection_rule_residual <= 1.0e-10);
        assert!(result.global_l2_bound_qualified);
        assert!(!result.time_discretization_error_qualified);
        assert_eq!(result.truncation_ledger.gate_count, 3);
        assert_eq!(result.gate_receipts.len(), 3);

        let dense = result.state.to_dense(4).unwrap();
        let exact = one_excitation_chain_exact(0.2, 0.7);
        let actual = [dense[4], dense[2], dense[1]];
        let error = actual
            .iter()
            .zip(exact.iter())
            .map(|(left, right)| (*left - *right).norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 5.0e-3, "Strang one-step error {error}");
    }

    #[test]
    fn strang_stepper_shows_second_order_global_convergence_against_dense_solution() {
        let coupling = 0.55;
        let total_time = 0.6;
        let stepper = U1TeBDStrangStepper::new(
            3,
            vec![
                U1TeBDHamiltonianTerm::new(0, nearest_neighbour_hopping_term(coupling)),
                U1TeBDHamiltonianTerm::new(1, nearest_neighbour_hopping_term(coupling)),
            ],
        )
        .unwrap();
        let input = ComplexU1Mps::product_state(&[true, false, false]).unwrap();
        let policy = U1TwoSiteTruncationPolicy {
            max_bond_dimension: 4,
            relative_singular_value_cutoff: 0.0,
            tolerance: 1.0e-11,
        };

        let evolve = |steps: usize| {
            let dt = total_time / steps as f64;
            let mut state = input.clone();
            for _ in 0..steps {
                state = stepper.apply(&state, dt, policy).unwrap().state;
            }
            state
        };
        let coarse = evolve(8).to_dense(4).unwrap();
        let fine = evolve(16).to_dense(4).unwrap();
        let exact = one_excitation_chain_exact(total_time, coupling);
        let dense_components = |state: &[Complex64]| [state[4], state[2], state[1]];
        let coarse_error = dense_components(&coarse)
            .iter()
            .zip(exact.iter())
            .map(|(left, right)| (*left - *right).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let fine_error = dense_components(&fine)
            .iter()
            .zip(exact.iter())
            .map(|(left, right)| (*left - *right).norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(coarse_error / fine_error > 3.2);
        assert!(coarse_error / fine_error < 4.8);
        assert!(fine_error < 5.0e-4);
    }

    #[test]
    fn hamiltonian_projection_residual_is_measured_and_attached_to_evolution_gate() {
        let mut matrix = [[complex_zero(); 4]; 4];
        matrix[0][0] = Complex64::new(0.2, 1.0e-8);
        matrix[3][3] = Complex64::new(-0.3, 0.0);
        matrix[1][1] = Complex64::new(0.1, 2.0e-8);
        matrix[2][2] = Complex64::new(-0.1, 0.0);
        matrix[1][2] = Complex64::new(0.7, 1.0e-8);
        matrix[2][1] = Complex64::new(0.7, 0.0);
        matrix[0][1] = Complex64::new(1.0e-8, 0.0);
        matrix[1][0] = Complex64::new(1.0e-8, 0.0);

        let hamiltonian = U1TwoSiteHamiltonian::new(matrix, 1.0e-6).unwrap();
        assert!(hamiltonian.hermiticity_residual() > 0.0);
        assert!(hamiltonian.charge_conservation_residual() > 0.0);
        assert!(hamiltonian.projection_residual_frobenius() > 0.0);

        let gate = hamiltonian.evolution_gate(0.3, 1.0e-12).unwrap();
        assert!(gate.hamiltonian_projection_error_operator_upper_bound() > 0.0);
        assert!(gate.hamiltonian_projection_error_operator_upper_bound().is_finite());
        assert!(gate.unitarity_residual() <= 1.0e-12);
        assert_eq!(gate.charge_conservation_residual(), 0.0);
    }

    #[test]
    fn strang_global_error_ledger_accounts_for_hamiltonian_model_projection() {
        let mut matrix = [[complex_zero(); 4]; 4];
        matrix[0][0] = Complex64::new(0.1, 1.0e-8);
        matrix[1][1] = Complex64::new(0.0, 1.0e-8);
        matrix[2][2] = Complex64::new(0.0, 0.0);
        matrix[1][2] = Complex64::new(0.4, 0.0);
        matrix[2][1] = Complex64::new(0.4, 0.0);
        matrix[0][1] = Complex64::new(1.0e-8, 0.0);
        matrix[1][0] = Complex64::new(1.0e-8, 0.0);
        let hamiltonian = U1TwoSiteHamiltonian::new(matrix, 1.0e-6).unwrap();

        let stepper = U1TeBDStrangStepper::new(
            2,
            vec![U1TeBDHamiltonianTerm::new(0, hamiltonian)],
        )
        .unwrap();
        let input = ComplexU1Mps::product_state(&[true, false]).unwrap();
        let result = stepper
            .apply(
                &input,
                0.2,
                U1TwoSiteTruncationPolicy {
                    max_bond_dimension: 2,
                    relative_singular_value_cutoff: 0.0,
                    tolerance: 1.0e-6,
                },
            )
            .unwrap();
        assert!(result.global_l2_bound_qualified);
        assert!(result.gate_receipts.len() == 2);
        assert!(
            result
                .truncation_ledger
                .cumulative_hamiltonian_projection_l2_error_bound
                > 0.0
        );
        let component_sum = result
            .truncation_ledger
            .cumulative_canonicalization_l2_error_bound
            + result.truncation_ledger.cumulative_gate_quality_l2_error_bound
            + result
                .truncation_ledger
                .cumulative_hamiltonian_projection_l2_error_bound
            + result.truncation_ledger.cumulative_projection_l2_error_bound;
        let bound = result.truncation_ledger.global_state_error_bound_upper.unwrap();
        assert!((component_sum - bound).abs() <= 1.0e-12);
        assert!(result.gate_receipts.iter().all(|receipt| {
            receipt.hamiltonian_projection_operator_error_upper_bound.is_finite()
                && receipt.hamiltonian_projection_error_upper_bound.is_finite()
        }));
    }

    #[test]
    fn strang_stepper_continues_global_error_ledger_across_time_steps() {
        let coupling = 0.8;
        let stepper = U1TeBDStrangStepper::new(
            3,
            vec![
                U1TeBDHamiltonianTerm::new(0, nearest_neighbour_hopping_term(coupling)),
                U1TeBDHamiltonianTerm::new(1, nearest_neighbour_hopping_term(coupling)),
            ],
        )
        .unwrap();
        let input = ComplexU1Mps::product_state(&[true, false, false]).unwrap();
        let policy = U1TwoSiteTruncationPolicy {
            max_bond_dimension: 1,
            relative_singular_value_cutoff: 0.0,
            tolerance: 1.0e-10,
        };
        let first = stepper.apply(&input, 0.25, policy).unwrap();
        let first_bound = first.truncation_ledger.global_state_error_bound_upper.unwrap();
        assert_eq!(first.truncation_ledger.gate_count, 3);
        let second = stepper
            .apply_with_ledger(
                &first.state,
                0.25,
                policy,
                first.truncation_ledger.clone(),
            )
            .unwrap();
        assert_eq!(second.truncation_ledger.gate_count, 6);
        assert!(second.global_l2_bound_qualified);
        assert!(second.truncation_ledger.global_state_error_bound_upper.unwrap() >= first_bound);
        assert_eq!(
            second.truncation_ledger.canonical_gauge_receipt_count,
            6
        );
    }

    #[test]
    fn canonical_gauge_receipt_rejects_understated_operator_norm_bound() {
        let mut state = canonical_w_state();
        let mut receipt = state.canonicalize_for_bond(0, 1.0e-10).unwrap();
        assert!(receipt.qualifies(1.0e-10));
        receipt.left_exterior_operator_norm_upper_bound = 1.0;
        assert!(!receipt.qualifies(1.0e-10));
    }

    #[test]
    fn canonical_gauge_receipt_rejects_negative_residuals_even_if_qualified_flag_is_set() {
        let mut state = canonical_w_state();
        let mut receipt = state.canonicalize_for_bond(0, 1.0e-10).unwrap();
        receipt.maximum_left_isometry_residual = -1.0;
        assert!(!receipt.qualifies(1.0e-10));
    }

    #[test]
    fn truncation_ledger_cannot_be_reused_with_a_different_mps_state() {
        let stepper = U1TeBDStrangStepper::new(
            3,
            vec![
                U1TeBDHamiltonianTerm::new(0, nearest_neighbour_hopping_term(0.4)),
                U1TeBDHamiltonianTerm::new(1, nearest_neighbour_hopping_term(0.4)),
            ],
        )
        .unwrap();
        let input = ComplexU1Mps::product_state(&[true, false, false]).unwrap();
        let policy = U1TwoSiteTruncationPolicy {
            max_bond_dimension: 2,
            relative_singular_value_cutoff: 0.0,
            tolerance: 1.0e-10,
        };
        let first = stepper.apply(&input, 0.1, policy).unwrap();
        let unrelated = ComplexU1Mps::product_state(&[false, true, false]).unwrap();
        assert!(matches!(
            stepper.apply_with_ledger(
                &unrelated,
                0.1,
                policy,
                first.truncation_ledger.clone()
            ),
            Err(U1MpsError::TruncationLedgerStateMismatch)
        ));
        // The continuation is accepted when paired with the exact output state
        // to which the ledger was bound.
        assert!(
            stepper
                .apply_with_ledger(
                    &first.state,
                    0.1,
                    policy,
                    first.truncation_ledger.clone()
                )
                .is_ok()
        );
    }

    #[test]
    fn strang_stepper_rejects_duplicate_and_out_of_range_hamiltonian_bonds() {
        let hamiltonian = nearest_neighbour_hopping_term(0.3);
        assert!(matches!(
            U1TeBDStrangStepper::new(
                3,
                vec![
                    U1TeBDHamiltonianTerm::new(0, hamiltonian.clone()),
                    U1TeBDHamiltonianTerm::new(0, hamiltonian.clone()),
                ]
            ),
            Err(U1MpsError::DuplicateTeBDGate { bond_index: 0 })
        ));
        assert!(matches!(
            U1TeBDStrangStepper::new(
                3,
                vec![U1TeBDHamiltonianTerm::new(3, hamiltonian)]
            ),
            Err(U1MpsError::TeBDGateOutOfRange { .. })
        ));
    }

    #[test]
    fn tebd_global_error_bound_stays_disabled_when_any_gauge_receipt_fails() {
        let input = canonical_w_state();
        let mut ledger = U1MpsTruncationLedger::default();
        let diagnostics = single_excitation_pair(0.8, 0.6)
            .split_truncated(U1TwoSiteTruncationPolicy {
                max_bond_dimension: 1,
                relative_singular_value_cutoff: 0.0,
                tolerance: 1.0e-12,
            })
            .unwrap();
        let mut bad_receipt = input.clone();
        let mut receipt = bad_receipt.canonicalize_for_bond(0, 1.0e-10).unwrap();
        receipt.qualified = false;
        ledger
            .record_with_canonical_gauge(&diagnostics, &receipt, 1.0e-10)
            .unwrap();
        assert!(!ledger.global_state_error_bound_qualified);
        assert!(ledger.global_state_error_bound_upper.is_none());
    }

    #[test]
    fn mixed_canonical_sweeps_preserve_complex_dense_state_and_report_isometries() {
        let mut dense = vec![Complex64::new(0.0, 0.0); 16];
        dense[3] = Complex64::new(0.3, 0.1);
        dense[5] = Complex64::new(-0.2, 0.25);
        dense[6] = Complex64::new(0.4, -0.1);
        dense[9] = Complex64::new(0.1, 0.3);
        dense[10] = Complex64::new(-0.35, 0.2);
        dense[12] = Complex64::new(0.2, -0.15);
        let norm = dense
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt();
        for value in &mut dense {
            *value /= norm;
        }

        let mut state = ComplexU1Mps::from_dense_fixed_charge(&dense, 4, 2, 1.0e-12).unwrap();
        let before = state.to_dense(8).unwrap();
        let input_norm = state.norm_squared(1.0e-12).unwrap();
        let receipt = state.canonicalize_for_bond(1, 1.0e-10).unwrap();

        assert_eq!(receipt.bond, 1);
        assert_eq!(receipt.site_count, 4);
        assert_eq!(receipt.total_charge, 2);
        assert_eq!(receipt.left_isometry_residuals.len(), 1);
        assert_eq!(receipt.right_isometry_residuals.len(), 1);
        assert!(receipt.maximum_left_isometry_residual <= 1.0e-10);
        assert!(receipt.maximum_right_isometry_residual <= 1.0e-10);
        assert!(receipt.factorization_max_residual <= 1.0e-10);
        assert!(receipt.relative_norm_drift <= 1.0e-10);
        assert!(receipt.qualifies(1.0e-10), "canonical receipt: {receipt:?}");
        assert!(receipt.exterior_operator_norm_product_upper_bound() >= 1.0);
        assert_eq!(state.total_charge(), 2);
        assert!(state.selection_rule_residual() <= 1.0e-10);
        assert!((state.norm_squared(1.0e-10).unwrap() - input_norm).abs() <= 1.0e-10);

        let after = state.to_dense(8).unwrap();
        for (index, (actual, expected)) in after.iter().zip(before.iter()).enumerate() {
            assert!(
                (*actual - *expected).norm() <= 1.0e-10,
                "basis={index}, actual={actual:?}, expected={expected:?}"
            );
        }
    }

    #[test]
    fn mixed_canonical_sweep_handles_left_and_right_boundary_centers() {
        let dense = [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let mut left_center = ComplexU1Mps::from_dense_fixed_charge(&dense, 2, 1, 1.0e-12).unwrap();
        let left_receipt = left_center.canonicalize_for_bond(0, 1.0e-10).unwrap();
        assert!(left_receipt.left_isometry_residuals.is_empty());
        assert_eq!(left_receipt.right_isometry_residuals.len(), 0);
        assert!(left_receipt.qualifies(1.0e-10));

        let mut right_center =
            ComplexU1Mps::from_dense_fixed_charge(&dense, 2, 1, 1.0e-12).unwrap();
        let right_receipt = right_center.canonicalize_for_bond(0, 1.0e-10).unwrap();
        assert!(right_receipt.qualifies(1.0e-10));
        assert!(right_center.selection_rule_residual() <= 1.0e-10);
    }

    #[test]
    fn mixed_canonicalization_rejects_invalid_center_and_tolerance_without_mutation() {
        let mut state = ComplexU1Mps::product_state(&[true, false, true, false]).unwrap();
        let before = state.to_dense(8).unwrap();
        assert!(matches!(
            state.canonicalize_for_bond(3, 1.0e-12),
            Err(U1MpsError::TwoSiteShapeMismatch)
        ));
        assert!(matches!(
            state.canonicalize_for_bond(0, 0.0),
            Err(U1MpsError::InvalidTolerance(0.0))
        ));
        assert_eq!(state.to_dense(8).unwrap(), before);
    }

    fn budget_test_state(scale: f64) -> U1TwoSiteState {
        U1TwoSiteState::new(
            vec![0],
            vec![1],
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(0.8 * scale, 0.0),
                Complex64::new(0.6 * scale, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            1.0e-12,
        )
        .unwrap()
    }

    #[test]
    fn error_budget_selects_minimum_global_rank_not_a_fixed_relative_cutoff() {
        let state = budget_test_state(1.0);
        let result: U1TwoSiteBudgetedSplit = state
            .split_with_error_budget(U1TwoSiteErrorBudgetPolicy {
                maximum_bond_dimension: 2,
                maximum_absolute_local_l2_error: 0.61,
                tolerance: 1.0e-12,
            })
            .unwrap();
        assert_eq!(result.minimum_required_bond_dimension, 1);
        assert_eq!(result.diagnostics.retained_singular_value_count, 1);
        assert!((result.achieved_absolute_local_l2_error - 0.6).abs() <= 1.0e-12);
        assert!((result.budget_slack - 0.01).abs() <= 1.0e-12);
    }

    #[test]
    fn error_budget_retains_an_extra_sector_when_the_requested_error_is_tighter() {
        let state = budget_test_state(1.0);
        let result = state
            .split_with_error_budget(U1TwoSiteErrorBudgetPolicy {
                maximum_bond_dimension: 2,
                maximum_absolute_local_l2_error: 0.59,
                tolerance: 1.0e-12,
            })
            .unwrap();
        assert_eq!(result.minimum_required_bond_dimension, 2);
        assert_eq!(result.diagnostics.retained_singular_value_count, 2);
        assert!(result.achieved_absolute_local_l2_error <= 0.59 + 1.0e-12);
    }

    #[test]
    fn error_budget_fails_closed_when_the_hard_bond_cap_cannot_meet_target() {
        let state = budget_test_state(1.0);
        assert!(matches!(
            state.split_with_error_budget(U1TwoSiteErrorBudgetPolicy {
                maximum_bond_dimension: 1,
                maximum_absolute_local_l2_error: 0.59,
                tolerance: 1.0e-12,
            }),
            Err(U1MpsError::TruncationBudgetUnreachable {
                requested_absolute_l2_error: 0.59,
                maximum_bond_dimension: 1,
                minimum_required_bond_dimension: 2,
                best_achievable_l2_error,
            }) if (best_achievable_l2_error - 0.6).abs() <= 1.0e-12
        ));
    }

    #[test]
    fn error_budget_is_scale_aware_and_reports_absolute_error() {
        let state = budget_test_state(2.0);
        let result = state
            .split_with_error_budget(U1TwoSiteErrorBudgetPolicy {
                maximum_bond_dimension: 1,
                maximum_absolute_local_l2_error: 1.21,
                tolerance: 1.0e-12,
            })
            .unwrap();
        assert_eq!(result.minimum_required_bond_dimension, 1);
        assert!((result.achieved_absolute_local_l2_error - 1.2).abs() <= 1.0e-12);
        assert!(result.budget_slack >= 0.0);
    }

    #[test]
    fn error_budget_rejects_nonfinite_negative_and_zero_capacity_policies() {
        let state = budget_test_state(1.0);
        for policy in [
            U1TwoSiteErrorBudgetPolicy {
                maximum_bond_dimension: 0,
                maximum_absolute_local_l2_error: 0.1,
                tolerance: 1.0e-12,
            },
            U1TwoSiteErrorBudgetPolicy {
                maximum_bond_dimension: 2,
                maximum_absolute_local_l2_error: -0.1,
                tolerance: 1.0e-12,
            },
            U1TwoSiteErrorBudgetPolicy {
                maximum_bond_dimension: 2,
                maximum_absolute_local_l2_error: f64::NAN,
                tolerance: 1.0e-12,
            },
            U1TwoSiteErrorBudgetPolicy {
                maximum_bond_dimension: 2,
                maximum_absolute_local_l2_error: 0.1,
                tolerance: f64::INFINITY,
            },
        ] {
            assert!(matches!(
                state.split_with_error_budget(policy),
                Err(U1MpsError::InvalidErrorBudget { .. })
            ));
        }
    }


    #[test]
    fn error_budget_rejects_overflowed_schmidt_weight_as_nonfinite_not_zero() {
        let state = U1TwoSiteState::new(
            vec![0],
            vec![1],
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0e200, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            1.0e-12,
        )
        .unwrap();
        assert!(matches!(
            state.split_with_error_budget(U1TwoSiteErrorBudgetPolicy {
                maximum_bond_dimension: 2,
                maximum_absolute_local_l2_error: 0.0,
                tolerance: 1.0e-12,
            }),
            Err(U1MpsError::NonFiniteContraction("error-budget Schmidt weight"))
        ));
    }

}
