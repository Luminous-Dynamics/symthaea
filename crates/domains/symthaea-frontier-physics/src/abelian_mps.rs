// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fixed-rank Abelian charge vectors and symmetry-block MPS structure.
//!
//! This module generalizes the scalar U(1) selection rule to a product of K
//! commuting integer-valued charges. It supplies structural validation and
//! two-site block planning; it does not yet implement generic K-charge SVD,
//! truncation, TEBD, or non-Abelian fusion trees.

use core::fmt;
use std::collections::BTreeMap;

use nalgebra::DMatrix;
use num_complex::Complex64;

/// A K-component integer charge under U(1)^K.
///
/// Components are ordered and named by the caller's convention. For example,
/// K=2 could represent (particle number, twice-Sz), or (spin-up count,
/// spin-down count). This type deliberately does not prescribe units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AbelianCharge<const K: usize> {
    components: [i32; K],
}

impl<const K: usize> AbelianCharge<K> {
    /// Construct a charge vector. K=0 is rejected because it carries no symmetry.
    pub fn new(components: [i32; K]) -> Result<Self, AbelianChargeError<K>> {
        if K == 0 {
            return Err(AbelianChargeError::ZeroChargeRank);
        }
        Ok(Self { components })
    }

    pub const fn zero() -> Self {
        Self {
            components: [0; K],
        }
    }

    pub const fn components(&self) -> &[i32; K] {
        &self.components
    }

    pub const fn rank(&self) -> usize {
        K
    }

    pub fn is_zero(&self) -> bool {
        self.components.iter().all(|component| *component == 0)
    }

    pub fn checked_add(
        self,
        rhs: Self,
    ) -> Result<Self, AbelianChargeError<K>> {
        if K == 0 {
            return Err(AbelianChargeError::ZeroChargeRank);
        }
        let mut result = [0_i32; K];
        for (index, output) in result.iter_mut().enumerate() {
            *output = self.components[index]
                .checked_add(rhs.components[index])
                .ok_or(AbelianChargeError::ChargeOverflow { component: index })?;
        }
        Ok(Self { components: result })
    }

    pub fn checked_sub(
        self,
        rhs: Self,
    ) -> Result<Self, AbelianChargeError<K>> {
        if K == 0 {
            return Err(AbelianChargeError::ZeroChargeRank);
        }
        let mut result = [0_i32; K];
        for (index, output) in result.iter_mut().enumerate() {
            *output = self.components[index]
                .checked_sub(rhs.components[index])
                .ok_or(AbelianChargeError::ChargeOverflow { component: index })?;
        }
        Ok(Self { components: result })
    }
}

/// Structural or numerical error in an Abelian charge-labelled tensor network.
#[derive(Debug, Clone, PartialEq)]
pub enum AbelianChargeError<const K: usize> {
    ZeroChargeRank,
    InvalidTolerance(f64),
    ChargeOverflow {
        component: usize,
    },
    EmptyMps,
    EmptyBond {
        site: usize,
        side: &'static str,
    },
    EmptyPhysicalBasis {
        site: usize,
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
    ChargeSelectionRule {
        site: usize,
        left_charge: AbelianCharge<K>,
        physical_charge: AbelianCharge<K>,
        right_charge: AbelianCharge<K>,
        amplitude: f64,
        tolerance: f64,
    },
    LeftBoundary {
        dimension: usize,
        charge: Option<AbelianCharge<K>>,
    },
    RightBoundary {
        dimension: usize,
        charge: Option<AbelianCharge<K>>,
        expected: AbelianCharge<K>,
    },
    BondChargeMismatch {
        bond: usize,
    },
    NoAllowedChargeBlocks,
    SvdFailed {
        rows: usize,
        columns: usize,
    },
    NonFiniteSingularValue,
    ZeroTwoSiteState,
    InvalidTruncationPolicy {
        max_bond_dimension: usize,
        relative_cutoff: f64,
        tolerance: f64,
    },
}

impl<const K: usize> fmt::Display for AbelianChargeError<K> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl<const K: usize> std::error::Error for AbelianChargeError<K> {}

/// A rank-3 complex tensor with an arbitrary local physical charge basis.
///
/// Tensor layout is ((left * physical_dimension) + physical) * right_dimension
/// + right. Every allowed nonzero entry obeys q_right = q_left + q_physical.
#[derive(Debug, Clone, PartialEq)]
pub struct AbelianChargeTensor<const K: usize> {
    left_charges: Vec<AbelianCharge<K>>,
    physical_charges: Vec<AbelianCharge<K>>,
    right_charges: Vec<AbelianCharge<K>>,
    data: Vec<Complex64>,
    selection_rule_residual: f64,
}

impl<const K: usize> AbelianChargeTensor<K> {
    pub fn new(
        site: usize,
        left_charges: Vec<AbelianCharge<K>>,
        physical_charges: Vec<AbelianCharge<K>>,
        right_charges: Vec<AbelianCharge<K>>,
        data: Vec<Complex64>,
        tolerance: f64,
    ) -> Result<Self, AbelianChargeError<K>> {
        if K == 0 {
            return Err(AbelianChargeError::ZeroChargeRank);
        }
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(AbelianChargeError::InvalidTolerance(tolerance));
        }
        if left_charges.is_empty() {
            return Err(AbelianChargeError::EmptyBond {
                site,
                side: "left",
            });
        }
        if physical_charges.is_empty() {
            return Err(AbelianChargeError::EmptyPhysicalBasis { site });
        }
        if right_charges.is_empty() {
            return Err(AbelianChargeError::EmptyBond {
                site,
                side: "right",
            });
        }
        let expected = left_charges
            .len()
            .checked_mul(physical_charges.len())
            .and_then(|value| value.checked_mul(right_charges.len()))
            .ok_or(AbelianChargeError::DimensionOverflow)?;
        if data.len() != expected {
            return Err(AbelianChargeError::EntryCount {
                expected,
                actual: data.len(),
            });
        }
        for (index, value) in data.iter().enumerate() {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(AbelianChargeError::NonFiniteTensorEntry { site, index });
            }
        }

        let mut selection_rule_residual: f64 = 0.0;
        let right_dimension = right_charges.len();
        let physical_dimension = physical_charges.len();
        for (left_index, left_charge) in left_charges.iter().copied().enumerate() {
            for (physical_index, physical_charge) in
                physical_charges.iter().copied().enumerate()
            {
                let expected_right = left_charge.checked_add(physical_charge)?;
                for (right_index, right_charge) in right_charges.iter().copied().enumerate() {
                    if right_charge != expected_right {
                        let index = (left_index * physical_dimension + physical_index)
                            * right_dimension
                            + right_index;
                        selection_rule_residual =
                            selection_rule_residual.max(data[index].norm());
                    }
                }
            }
        }
        if selection_rule_residual > tolerance {
            for (left_index, left_charge) in left_charges.iter().copied().enumerate() {
                for (physical_index, physical_charge) in
                    physical_charges.iter().copied().enumerate()
                {
                    let expected_right = left_charge.checked_add(physical_charge)?;
                    for (right_index, right_charge) in right_charges.iter().copied().enumerate() {
                        if right_charge != expected_right {
                            let index = (left_index * physical_dimension + physical_index)
                                * right_dimension
                                + right_index;
                            let amplitude = data[index].norm();
                            if amplitude > tolerance {
                                return Err(AbelianChargeError::ChargeSelectionRule {
                                    site,
                                    left_charge,
                                    physical_charge,
                                    right_charge,
                                    amplitude,
                                    tolerance,
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(Self {
            left_charges,
            physical_charges,
            right_charges,
            data,
            selection_rule_residual,
        })
    }

    pub fn left_charges(&self) -> &[AbelianCharge<K>] {
        &self.left_charges
    }

    pub fn physical_charges(&self) -> &[AbelianCharge<K>] {
        &self.physical_charges
    }

    pub fn right_charges(&self) -> &[AbelianCharge<K>] {
        &self.right_charges
    }

    pub fn left_dimension(&self) -> usize {
        self.left_charges.len()
    }

    pub fn physical_dimension(&self) -> usize {
        self.physical_charges.len()
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

    pub fn get(&self, left: usize, physical: usize, right: usize) -> Complex64 {
        self.data[(left * self.physical_dimension() + physical) * self.right_dimension() + right]
    }
}

/// An open-boundary MPS in a fixed U(1)^K sector.
#[derive(Debug, Clone, PartialEq)]
pub struct AbelianMps<const K: usize> {
    tensors: Vec<AbelianChargeTensor<K>>,
    total_charge: AbelianCharge<K>,
    selection_rule_residual: f64,
}

impl<const K: usize> AbelianMps<K> {
    pub fn new(
        tensors: Vec<AbelianChargeTensor<K>>,
        total_charge: AbelianCharge<K>,
        tolerance: f64,
    ) -> Result<Self, AbelianChargeError<K>> {
        if K == 0 {
            return Err(AbelianChargeError::ZeroChargeRank);
        }
        if tensors.is_empty() {
            return Err(AbelianChargeError::EmptyMps);
        }
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(AbelianChargeError::InvalidTolerance(tolerance));
        }
        let first = &tensors[0];
        if first.left_dimension() != 1 || !first.left_charges()[0].is_zero() {
            return Err(AbelianChargeError::LeftBoundary {
                dimension: first.left_dimension(),
                charge: first.left_charges().first().copied(),
            });
        }
        let last = tensors.last().ok_or(AbelianChargeError::EmptyMps)?;
        if last.right_dimension() != 1 || last.right_charges()[0] != total_charge {
            return Err(AbelianChargeError::RightBoundary {
                dimension: last.right_dimension(),
                charge: last.right_charges().first().copied(),
                expected: total_charge,
            });
        }
        for (bond, adjacent) in tensors.windows(2).enumerate() {
            if adjacent[0].right_charges() != adjacent[1].left_charges() {
                return Err(AbelianChargeError::BondChargeMismatch { bond });
            }
        }
        let selection_rule_residual = tensors
            .iter()
            .map(AbelianChargeTensor::selection_rule_residual)
            .reduce(f64::max)
            .unwrap_or(0.0);
        if selection_rule_residual > tolerance {
            return Err(AbelianChargeError::NoAllowedChargeBlocks);
        }

        Ok(Self {
            tensors,
            total_charge,
            selection_rule_residual,
        })
    }

    pub fn tensors(&self) -> &[AbelianChargeTensor<K>] {
        &self.tensors
    }

    pub fn total_charge(&self) -> AbelianCharge<K> {
        self.total_charge
    }

    pub fn site_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn selection_rule_residual(&self) -> f64 {
        self.selection_rule_residual
    }

    pub fn maximum_bond_dimension(&self) -> usize {
        self.tensors
            .iter()
            .map(AbelianChargeTensor::right_dimension)
            .max()
            .unwrap_or(0)
    }
}

/// One independent intermediate-charge block of a two-site tensor split.
#[derive(Debug, Clone, PartialEq)]
pub struct AbelianChargeBlock<const K: usize> {
    pub intermediate_charge: AbelianCharge<K>,
    /// (left virtual index, first-site physical index).
    pub rows: Vec<(usize, usize)>,
    /// (second-site physical index, right virtual index).
    pub columns: Vec<(usize, usize)>,
}

/// Deterministic charge-sector partition for a two-site tensor.
///
/// Rows are grouped by q_mid = q_left + q_physical_1. Columns are grouped by
/// q_mid = q_right - q_physical_2. Only equal intermediate charge vectors are
/// paired, so a later SVD cannot mix distinct Abelian sectors.
#[derive(Debug, Clone, PartialEq)]
pub struct AbelianChargeBlockPlan<const K: usize> {
    pub blocks: Vec<AbelianChargeBlock<K>>,
}



/// Diagnostics from an exact U(1)^K charge-block two-site SVD.
///
/// Exactly-zero singular directions are omitted; every positive singular
/// value is retained. No cutoff, bond cap or renormalization is performed.
#[derive(Debug, Clone, PartialEq)]
pub struct AbelianTwoSiteSplit<const K: usize> {
    pub left_tensor: AbelianChargeTensor<K>,
    pub right_tensor: AbelianChargeTensor<K>,
    pub rank_by_charge: Vec<(AbelianCharge<K>, usize)>,
    pub total_singular_value_count: usize,
    pub retained_singular_value_count: usize,
    pub exact_zero_singular_values_omitted: usize,
    pub reconstruction_max_residual: f64,
    pub reconstruction_l2_residual: f64,
}



/// Explicit approximate-compression policy across all U(1)^K sectors.
///
/// The relative singular-value cutoff keeps sigma_i >= r * sigma_max.
/// The maximum bond dimension is global across all charge vectors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AbelianTwoSiteTruncationPolicy {
    pub max_bond_dimension: usize,
    pub relative_singular_value_cutoff: f64,
    pub tolerance: f64,
}

/// Receipts from global cutoff/cap truncation across Abelian charge sectors.
///
/// The output tensors are not renormalized. A restoring factor is reported but
/// not applied.
#[derive(Debug, Clone, PartialEq)]
pub struct AbelianTwoSiteTruncation<const K: usize> {
    pub left_tensor: AbelianChargeTensor<K>,
    pub right_tensor: AbelianChargeTensor<K>,
    pub retained_rank_by_charge: Vec<(AbelianCharge<K>, usize)>,
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
    pub renormalization_factor_to_restore_original_norm: f64,
    pub bond_cap_binding: bool,
    pub reconstruction_max_residual: f64,
    pub reconstruction_l2_residual: f64,
}

impl<const K: usize> AbelianChargeBlockPlan<K> {
    pub fn from_adjacent_tensors(
        first: &AbelianChargeTensor<K>,
        second: &AbelianChargeTensor<K>,
    ) -> Result<Self, AbelianChargeError<K>> {
        if K == 0 {
            return Err(AbelianChargeError::ZeroChargeRank);
        }
        if first.right_charges() != second.left_charges() {
            return Err(AbelianChargeError::BondChargeMismatch { bond: 0 });
        }

        let mut row_groups: BTreeMap<AbelianCharge<K>, Vec<(usize, usize)>> = BTreeMap::new();
        for (left, left_charge) in first.left_charges().iter().enumerate() {
            for (physical, physical_charge) in first.physical_charges().iter().enumerate() {
                let charge = left_charge.checked_add(*physical_charge)?;
                row_groups.entry(charge).or_default().push((left, physical));
            }
        }

        let mut column_groups: BTreeMap<AbelianCharge<K>, Vec<(usize, usize)>> = BTreeMap::new();
        for (physical, physical_charge) in second.physical_charges().iter().enumerate() {
            for (right, right_charge) in second.right_charges().iter().enumerate() {
                let charge = right_charge.checked_sub(*physical_charge)?;
                column_groups.entry(charge).or_default().push((physical, right));
            }
        }

        let mut blocks = Vec::new();
        for (charge, rows) in row_groups {
            if let Some(columns) = column_groups.get(&charge) {
                if !rows.is_empty() && !columns.is_empty() {
                    blocks.push(AbelianChargeBlock {
                        intermediate_charge: charge,
                        rows,
                        columns: columns.clone(),
                    });
                }
            }
        }
        if blocks.is_empty() {
            return Err(AbelianChargeError::NoAllowedChargeBlocks);
        }
        Ok(Self { blocks })
    }

    /// Factor adjacent charge-labelled site tensors using an independent SVD
    /// for every intermediate K-vector charge sector.
    ///
    /// The two-site matrix is assembled from the exact contraction of the
    /// supplied tensors. No dense full-chain expansion is performed.
    pub fn split_exact(
        first: &AbelianChargeTensor<K>,
        second: &AbelianChargeTensor<K>,
        tolerance: f64,
    ) -> Result<AbelianTwoSiteSplit<K>, AbelianChargeError<K>> {
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(AbelianChargeError::InvalidTolerance(tolerance));
        }
        let plan = Self::from_adjacent_tensors(first, second)?;

        struct BlockFactor<const K: usize> {
            charge: AbelianCharge<K>,
            rows: Vec<(usize, usize)>,
            columns: Vec<(usize, usize)>,
            left_vectors: DMatrix<Complex64>,
            right_vectors_t: DMatrix<Complex64>,
            singular_values: Vec<f64>,
        }

        let maximum_block_dimension = plan
            .blocks
            .iter()
            .map(|block| block.rows.len().max(block.columns.len()))
            .max()
            .unwrap_or(0);
        let maximum_iterations = 10_000_usize.max(
            64_usize.saturating_mul(
                maximum_block_dimension.saturating_mul(maximum_block_dimension),
            ),
        );

        let mut factors = Vec::with_capacity(plan.blocks.len());
        let mut rank_by_charge = Vec::with_capacity(plan.blocks.len());
        let mut total_singular_value_count = 0_usize;
        let mut exact_zero_singular_values_omitted = 0_usize;
        let mut reconstruction_max_residual: f64 = 0.0;
        let mut reconstruction_squared = 0.0;

        for block in &plan.blocks {
            let mut matrix =
                DMatrix::<Complex64>::zeros(block.rows.len(), block.columns.len());
            for (row_index, &(left, first_physical)) in block.rows.iter().enumerate() {
                for (column_index, &(second_physical, right)) in
                    block.columns.iter().enumerate()
                {
                    let mut value = Complex64::new(0.0, 0.0);
                    for middle in 0..first.right_dimension() {
                        if first.right_charges()[middle] == block.intermediate_charge {
                            value += first.get(left, first_physical, middle)
                                * second.get(middle, second_physical, right);
                        }
                    }
                    matrix[(row_index, column_index)] = value;
                }
            }

            let row_count = matrix.nrows();
            let column_count = matrix.ncols();
            let decomposition = matrix
                .try_svd(true, true, tolerance, maximum_iterations)
                .ok_or(AbelianChargeError::SvdFailed {
                    rows: row_count,
                    columns: column_count,
                })?;
            let u = decomposition.u.ok_or(AbelianChargeError::SvdFailed {
                rows: row_count,
                columns: column_count,
            })?;
            let v_t = decomposition.v_t.ok_or(AbelianChargeError::SvdFailed {
                rows: row_count,
                columns: column_count,
            })?;

            total_singular_value_count = total_singular_value_count
                .checked_add(decomposition.singular_values.len())
                .ok_or(AbelianChargeError::DimensionOverflow)?;
            let mut kept_indices = Vec::new();
            for (index, singular_value) in decomposition.singular_values.iter().enumerate() {
                if !singular_value.is_finite() {
                    return Err(AbelianChargeError::NonFiniteSingularValue);
                }
                if *singular_value > 0.0 {
                    kept_indices.push(index);
                } else {
                    exact_zero_singular_values_omitted += 1;
                }
            }
            rank_by_charge.push((block.intermediate_charge, kept_indices.len()));
            if kept_indices.is_empty() {
                continue;
            }

            let left_vectors = DMatrix::<Complex64>::from_fn(
                block.rows.len(),
                kept_indices.len(),
                |row, column| u[(row, kept_indices[column])],
            );
            let right_vectors_t = DMatrix::<Complex64>::from_fn(
                kept_indices.len(),
                block.columns.len(),
                |row, column| v_t[(kept_indices[row], column)],
            );
            let singular_values = kept_indices
                .iter()
                .map(|&index| decomposition.singular_values[index])
                .collect::<Vec<_>>();

            factors.push(BlockFactor {
                charge: block.intermediate_charge,
                rows: block.rows.clone(),
                columns: block.columns.clone(),
                left_vectors,
                right_vectors_t,
                singular_values,
            });
        }

        let retained_singular_value_count = factors.iter().try_fold(
            0_usize,
            |sum, factor| {
                sum.checked_add(factor.singular_values.len())
                    .ok_or(AbelianChargeError::DimensionOverflow)
            },
        )?;
        if retained_singular_value_count == 0 {
            return Err(AbelianChargeError::ZeroTwoSiteState);
        }

        let left_entries = first
            .left_dimension()
            .checked_mul(first.physical_dimension())
            .and_then(|value| value.checked_mul(retained_singular_value_count))
            .ok_or(AbelianChargeError::DimensionOverflow)?;
        let right_entries = retained_singular_value_count
            .checked_mul(second.physical_dimension())
            .and_then(|value| value.checked_mul(second.right_dimension()))
            .ok_or(AbelianChargeError::DimensionOverflow)?;
        let mut left_data = vec![Complex64::new(0.0, 0.0); left_entries];
        let mut right_data = vec![Complex64::new(0.0, 0.0); right_entries];
        let mut middle_charges = Vec::with_capacity(retained_singular_value_count);
        let mut offset = 0_usize;

        for factor in &factors {
            let rank = factor.singular_values.len();
            middle_charges.extend(core::iter::repeat_n(factor.charge, rank));
            for mode in 0..rank {
                let new_middle = offset + mode;
                for (row_index, &(left, physical)) in factor.rows.iter().enumerate() {
                    let index = (left * first.physical_dimension() + physical)
                        * retained_singular_value_count
                        + new_middle;
                    left_data[index] = factor.left_vectors[(row_index, mode)];
                }
                for (column_index, &(physical, right)) in factor.columns.iter().enumerate() {
                    let index = (new_middle * second.physical_dimension() + physical)
                        * second.right_dimension()
                        + right;
                    right_data[index] = factor.singular_values[mode]
                        * factor.right_vectors_t[(mode, column_index)];
                }
            }
            offset += rank;
        }

        let left_tensor = AbelianChargeTensor::new(
            0,
            first.left_charges().to_vec(),
            first.physical_charges().to_vec(),
            middle_charges.clone(),
            left_data,
            tolerance,
        )?;
        let right_tensor = AbelianChargeTensor::new(
            1,
            middle_charges,
            second.physical_charges().to_vec(),
            second.right_charges().to_vec(),
            right_data,
            tolerance,
        )?;

        for left in 0..first.left_dimension() {
            for first_physical in 0..first.physical_dimension() {
                for second_physical in 0..second.physical_dimension() {
                    for right in 0..second.right_dimension() {
                        let mut original = Complex64::new(0.0, 0.0);
                        for middle in 0..first.right_dimension() {
                            original += first.get(left, first_physical, middle)
                                * second.get(middle, second_physical, right);
                        }
                        let mut reconstructed = Complex64::new(0.0, 0.0);
                        for middle in 0..retained_singular_value_count {
                            reconstructed += left_tensor.get(left, first_physical, middle)
                                * right_tensor.get(middle, second_physical, right);
                        }
                        let residual = (reconstructed - original).norm();
                        reconstruction_max_residual =
                            reconstruction_max_residual.max(residual);
                        reconstruction_squared += residual * residual;
                    }
                }
            }
        }
        if !reconstruction_max_residual.is_finite() || !reconstruction_squared.is_finite() {
            return Err(AbelianChargeError::NonFiniteSingularValue);
        }

        Ok(AbelianTwoSiteSplit {
            left_tensor,
            right_tensor,
            rank_by_charge,
            total_singular_value_count,
            retained_singular_value_count,
            exact_zero_singular_values_omitted,
            reconstruction_max_residual,
            reconstruction_l2_residual: reconstruction_squared.sqrt(),
        })
    }


    /// Apply a relative singular-value cutoff, then one global bond cap across
    /// all charge vectors. No implicit renormalization is performed.
    pub fn split_truncated(
        first: &AbelianChargeTensor<K>,
        second: &AbelianChargeTensor<K>,
        policy: AbelianTwoSiteTruncationPolicy,
    ) -> Result<AbelianTwoSiteTruncation<K>, AbelianChargeError<K>> {
        if policy.max_bond_dimension == 0
            || !policy.relative_singular_value_cutoff.is_finite()
            || !(0.0..=1.0).contains(&policy.relative_singular_value_cutoff)
            || !policy.tolerance.is_finite()
            || policy.tolerance <= 0.0
        {
            return Err(AbelianChargeError::InvalidTruncationPolicy {
                max_bond_dimension: policy.max_bond_dimension,
                relative_cutoff: policy.relative_singular_value_cutoff,
                tolerance: policy.tolerance,
            });
        }

        let exact = Self::split_exact(first, second, policy.tolerance)?;
        let full_rank = exact.retained_singular_value_count;
        let middle_charges = exact.left_tensor.right_charges().to_vec();
        let mut singular_values = vec![0.0_f64; full_rank];
        let mut total_schmidt_weight = 0.0;
        for middle in 0..full_rank {
            let mut row_norm_squared = 0.0;
            for physical in 0..exact.right_tensor.physical_dimension() {
                for right in 0..exact.right_tensor.right_dimension() {
                    row_norm_squared +=
                        exact.right_tensor.get(middle, physical, right).norm_sqr();
                }
            }
            let value = row_norm_squared.sqrt();
            if !value.is_finite() {
                return Err(AbelianChargeError::NonFiniteSingularValue);
            }
            singular_values[middle] = value;
            total_schmidt_weight += value * value;
        }
        if !total_schmidt_weight.is_finite() || total_schmidt_weight <= 0.0 {
            return Err(AbelianChargeError::ZeroTwoSiteState);
        }

        let maximum_singular_value = singular_values.iter().copied().fold(0.0, f64::max);
        let cutoff_threshold =
            policy.relative_singular_value_cutoff * maximum_singular_value;
        let mut cutoff_eligible = Vec::new();
        let mut cutoff_discarded_weight = 0.0;
        let mut cutoff_discarded_singular_value_count = 0;
        for (index, value) in singular_values.iter().copied().enumerate() {
            if value >= cutoff_threshold {
                cutoff_eligible.push(index);
            } else {
                cutoff_discarded_weight += value * value;
                cutoff_discarded_singular_value_count += 1;
            }
        }

        cutoff_eligible.sort_by(|left, right| {
            singular_values[*right]
                .total_cmp(&singular_values[*left])
                .then_with(|| left.cmp(right))
        });
        let bond_cap_binding = cutoff_eligible.len() > policy.max_bond_dimension;
        let retained_ranked = cutoff_eligible
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
        let mut retained = retained_ranked;
        retained.sort_unstable();
        let retained_rank = retained.len();
        if retained_rank == 0 {
            return Err(AbelianChargeError::ZeroTwoSiteState);
        }

        let retained_schmidt_weight = retained
            .iter()
            .map(|index| singular_values[*index] * singular_values[*index])
            .sum::<f64>();
        let discarded_weight = cutoff_discarded_weight + bond_cap_discarded_weight;
        let relative_discarded_weight = discarded_weight / total_schmidt_weight;
        let renormalization_factor_to_restore_original_norm =
            (total_schmidt_weight / retained_schmidt_weight).sqrt();
        if !retained_schmidt_weight.is_finite()
            || retained_schmidt_weight <= 0.0
            || !discarded_weight.is_finite()
            || !relative_discarded_weight.is_finite()
            || !renormalization_factor_to_restore_original_norm.is_finite()
        {
            return Err(AbelianChargeError::NonFiniteSingularValue);
        }

        let left_entries = exact
            .left_tensor
            .left_dimension()
            .checked_mul(exact.left_tensor.physical_dimension())
            .and_then(|value| value.checked_mul(retained_rank))
            .ok_or(AbelianChargeError::DimensionOverflow)?;
        let right_entries = retained_rank
            .checked_mul(exact.right_tensor.physical_dimension())
            .and_then(|value| value.checked_mul(exact.right_tensor.right_dimension()))
            .ok_or(AbelianChargeError::DimensionOverflow)?;
        let mut left_data = vec![Complex64::new(0.0, 0.0); left_entries];
        let mut right_data = vec![Complex64::new(0.0, 0.0); right_entries];
        let mut retained_middle_charges = Vec::with_capacity(retained_rank);

        for (new_middle, old_middle) in retained.iter().copied().enumerate() {
            retained_middle_charges.push(middle_charges[old_middle]);
            for left in 0..exact.left_tensor.left_dimension() {
                for physical in 0..exact.left_tensor.physical_dimension() {
                    let index =
                        (left * exact.left_tensor.physical_dimension() + physical)
                            * retained_rank
                            + new_middle;
                    left_data[index] = exact.left_tensor.get(left, physical, old_middle);
                }
            }
            for physical in 0..exact.right_tensor.physical_dimension() {
                for right in 0..exact.right_tensor.right_dimension() {
                    let index =
                        (new_middle * exact.right_tensor.physical_dimension() + physical)
                            * exact.right_tensor.right_dimension()
                            + right;
                    right_data[index] =
                        exact.right_tensor.get(old_middle, physical, right);
                }
            }
        }

        let left_tensor = AbelianChargeTensor::new(
            0,
            exact.left_tensor.left_charges().to_vec(),
            exact.left_tensor.physical_charges().to_vec(),
            retained_middle_charges.clone(),
            left_data,
            policy.tolerance,
        )?;
        let right_tensor = AbelianChargeTensor::new(
            1,
            retained_middle_charges.clone(),
            exact.right_tensor.physical_charges().to_vec(),
            exact.right_tensor.right_charges().to_vec(),
            right_data,
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

        let mut reconstruction_max_residual: f64 = 0.0;
        let mut reconstruction_squared = 0.0;
        for left in 0..first.left_dimension() {
            for first_physical in 0..first.physical_dimension() {
                for second_physical in 0..second.physical_dimension() {
                    for right in 0..second.right_dimension() {
                        let mut original = Complex64::new(0.0, 0.0);
                        for middle in 0..first.right_dimension() {
                            original += first.get(left, first_physical, middle)
                                * second.get(middle, second_physical, right);
                        }
                        let mut reconstructed = Complex64::new(0.0, 0.0);
                        for middle in 0..retained_rank {
                            reconstructed += left_tensor.get(left, first_physical, middle)
                                * right_tensor.get(middle, second_physical, right);
                        }
                        let residual = (reconstructed - original).norm();
                        reconstruction_max_residual =
                            reconstruction_max_residual.max(residual);
                        reconstruction_squared += residual * residual;
                    }
                }
            }
        }
        if !reconstruction_max_residual.is_finite() || !reconstruction_squared.is_finite() {
            return Err(AbelianChargeError::NonFiniteSingularValue);
        }

        Ok(AbelianTwoSiteTruncation {
            left_tensor,
            right_tensor,
            retained_rank_by_charge,
            total_singular_value_count: exact.total_singular_value_count,
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

}

#[cfg(test)]
mod tests {
    use super::{
        AbelianCharge, AbelianChargeBlockPlan, AbelianChargeError, AbelianChargeTensor, AbelianMps,
        AbelianTwoSiteTruncationPolicy,
    };
    use num_complex::Complex64;

    fn q(first: i32, second: i32) -> AbelianCharge<2> {
        AbelianCharge::new([first, second]).unwrap()
    }

    fn zero() -> Complex64 {
        Complex64::new(0.0, 0.0)
    }



    fn two_sector_schmidt_pair() -> (
        AbelianChargeTensor<2>,
        AbelianChargeTensor<2>,
    ) {
        let vacuum = q(0, 0);
        let up = q(1, 1);
        let down = q(1, -1);
        let double = q(2, 0);
        let physical = vec![vacuum, up, down, double];
        let bond = vec![vacuum, up, down, double];

        let mut first_data = vec![zero(); 16];
        first_data[5] = Complex64::new(0.8, 0.0);
        first_data[10] = Complex64::new(0.6, 0.0);
        let first = AbelianChargeTensor::new(
            0,
            vec![vacuum],
            physical.clone(),
            bond.clone(),
            first_data,
            1.0e-12,
        )
        .unwrap();

        let mut second_data = vec![zero(); 16];
        second_data[6] = Complex64::new(1.0, 0.0);
        second_data[9] = Complex64::new(1.0, 0.0);
        let second = AbelianChargeTensor::new(
            1,
            bond,
            physical,
            vec![double],
            second_data,
            1.0e-12,
        )
        .unwrap();
        (first, second)
    }

    fn contracted_two_site_norm_squared<const K: usize>(
        left: &AbelianChargeTensor<K>,
        right: &AbelianChargeTensor<K>,
    ) -> f64 {
        let mut norm_squared = 0.0;
        for left_index in 0..left.left_dimension() {
            for first_physical in 0..left.physical_dimension() {
                for second_physical in 0..right.physical_dimension() {
                    for right_index in 0..right.right_dimension() {
                        let mut amplitude = Complex64::new(0.0, 0.0);
                        for middle in 0..left.right_dimension() {
                            amplitude += left.get(left_index, first_physical, middle)
                                * right.get(middle, second_physical, right_index);
                        }
                        norm_squared += amplitude.norm_sqr();
                    }
                }
            }
        }
        norm_squared
    }

    #[test]
    fn charge_vectors_add_subtract_and_detect_component_overflow() {
        let particle_spin = q(2, -1);
        let local = q(1, 1);
        assert_eq!(particle_spin.checked_add(local).unwrap(), q(3, 0));
        assert_eq!(particle_spin.checked_sub(local).unwrap(), q(1, -2));
        assert!(matches!(
            q(i32::MAX, 0).checked_add(q(1, 0)),
            Err(AbelianChargeError::ChargeOverflow { component: 0 })
        ));
        assert!(matches!(
            q(i32::MIN, 0).checked_sub(q(1, 0)),
            Err(AbelianChargeError::ChargeOverflow { component: 0 })
        ));
        assert!(matches!(
            AbelianCharge::<0>::new([]),
            Err(AbelianChargeError::ZeroChargeRank)
        ));
        let maximum = AbelianCharge::new([i32::MAX, 0]).unwrap();
        let increment = q(1, 0);
        assert!(matches!(
            AbelianChargeTensor::new(
                0,
                vec![maximum],
                vec![increment],
                vec![maximum],
                vec![zero()],
                1.0e-12,
            ),
            Err(AbelianChargeError::ChargeOverflow { component: 0 })
        ));
    }

    #[test]
    fn two_charge_mps_enforces_both_particle_number_and_spin_selection_rules() {
        let vacuum = q(0, 0);
        let spin_up = q(1, 1);
        let spin_down = q(1, -1);
        let double = q(2, 0);
        let physical = vec![vacuum, spin_up, spin_down, double];

        let left_bond = vec![vacuum, spin_up, spin_down, double];
        let mut first_data = vec![zero(); 16];
        first_data[0] = Complex64::new(1.0, 0.0);
        first_data[5] = Complex64::new(0.5, 0.0);
        first_data[10] = Complex64::new(0.5, 0.0);
        let first = AbelianChargeTensor::new(
            0,
            vec![vacuum],
            physical.clone(),
            left_bond.clone(),
            first_data,
            1.0e-12,
        )
        .unwrap();

        let mut second_data = vec![zero(); 16];
        // |vacuum> -> |double> and the two single-occupancy exchange paths.
        second_data[3] = Complex64::new(1.0, 0.0);
        second_data[6] = Complex64::new(1.0, 0.0);
        second_data[9] = Complex64::new(1.0, 0.0);
        second_data[12] = Complex64::new(1.0, 0.0);
        let second = AbelianChargeTensor::new(
            1,
            left_bond,
            physical,
            vec![double],
            second_data,
            1.0e-12,
        )
        .unwrap();

        let state = AbelianMps::new(vec![first, second], double, 1.0e-12).unwrap();
        assert_eq!(state.site_count(), 2);
        assert_eq!(state.total_charge(), double);
        assert_eq!(state.selection_rule_residual(), 0.0);
    }

    #[test]
    fn two_site_block_plan_partitions_joint_charge_vectors_without_cross_sector_pairs() {
        let vacuum = q(0, 0);
        let up = q(1, 1);
        let down = q(1, -1);
        let double = q(2, 0);
        let physical = vec![vacuum, up, down, double];
        let bond = vec![vacuum, up, down, double];

        let first = AbelianChargeTensor::new(
            0,
            vec![vacuum],
            physical.clone(),
            bond.clone(),
            vec![zero(); 16],
            1.0e-12,
        )
        .unwrap();
        let second = AbelianChargeTensor::new(
            1,
            bond,
            physical,
            vec![double],
            vec![zero(); 16],
            1.0e-12,
        )
        .unwrap();

        let plan = AbelianChargeBlockPlan::from_adjacent_tensors(&first, &second).unwrap();
        assert_eq!(plan.blocks.len(), 4);
        let charges = plan
            .blocks
            .iter()
            .map(|block| block.intermediate_charge)
            .collect::<Vec<_>>();
        assert_eq!(charges, vec![vacuum, down, up, double]);
        for block in &plan.blocks {
            for &(left, first_physical) in &block.rows {
                assert_eq!(
                    first.left_charges()[left]
                        .checked_add(first.physical_charges()[first_physical])
                        .unwrap(),
                    block.intermediate_charge
                );
            }
            for &(second_physical, right) in &block.columns {
                assert_eq!(
                    second.right_charges()[right]
                        .checked_sub(second.physical_charges()[second_physical])
                        .unwrap(),
                    block.intermediate_charge
                );
            }
        }
    }

    #[test]
    fn joint_charge_tensor_rejects_forbidden_cross_sector_amplitude() {
        let vacuum = q(0, 0);
        let up = q(1, 1);
        let mut data = vec![zero(); 2];
        data[1] = Complex64::new(0.25, 0.0);
        assert!(matches!(
            AbelianChargeTensor::new(
                0,
                vec![vacuum],
                vec![vacuum],
                vec![vacuum, up],
                data,
                1.0e-12,
            ),
            Err(AbelianChargeError::ChargeSelectionRule { .. })
        ));
    }

    #[test]
    fn joint_charge_mps_rejects_bond_mismatch_and_wrong_total_charge() {
        let vacuum = q(0, 0);
        let up = q(1, 1);
        let first = AbelianChargeTensor::new(
            0,
            vec![vacuum],
            vec![up],
            vec![up],
            vec![Complex64::new(1.0, 0.0)],
            1.0e-12,
        )
        .unwrap();
        let last = AbelianChargeTensor::new(
            1,
            vec![up],
            vec![vacuum],
            vec![up],
            vec![Complex64::new(1.0, 0.0)],
            1.0e-12,
        )
        .unwrap();
        assert!(AbelianMps::new(vec![first.clone(), last.clone()], up, 1.0e-12).is_ok());

        let mismatched_bond = AbelianChargeTensor::new(
            1,
            vec![vacuum],
            vec![up],
            vec![up],
            vec![Complex64::new(1.0, 0.0)],
            1.0e-12,
        )
        .unwrap();
        assert!(matches!(
            AbelianMps::new(vec![first.clone(), mismatched_bond], up, 1.0e-12),
            Err(AbelianChargeError::BondChargeMismatch { bond: 0 })
        ));

        assert!(matches!(
            AbelianMps::new(vec![first, last], vacuum, 1.0e-12),
            Err(AbelianChargeError::RightBoundary { .. })
        ));
    }

    #[test]
    fn exact_u1k_two_site_svd_reconstructs_state_and_reports_each_charge_rank() {
        let vacuum = q(0, 0);
        let up = q(1, 1);
        let down = q(1, -1);
        let double = q(2, 0);
        let physical = vec![vacuum, up, down, double];
        let bond = vec![vacuum, up, down, double];

        let mut first_data = vec![zero(); 16];
        first_data[0] = Complex64::new(1.0, 0.0);
        first_data[5] = Complex64::new(0.5, 0.0);
        first_data[10] = Complex64::new(0.5, 0.0);
        let first = AbelianChargeTensor::new(
            0,
            vec![vacuum],
            physical.clone(),
            bond.clone(),
            first_data,
            1.0e-12,
        )
        .unwrap();

        let mut second_data = vec![zero(); 16];
        second_data[3] = Complex64::new(1.0, 0.0);
        second_data[6] = Complex64::new(1.0, 0.0);
        second_data[9] = Complex64::new(1.0, 0.0);
        second_data[12] = Complex64::new(1.0, 0.0);
        let second = AbelianChargeTensor::new(
            1,
            bond,
            physical,
            vec![double],
            second_data,
            1.0e-12,
        )
        .unwrap();

        let split =
            AbelianChargeBlockPlan::split_exact(&first, &second, 1.0e-12).unwrap();
        assert_eq!(
            split.rank_by_charge,
            vec![(vacuum, 1), (down, 1), (up, 1), (double, 0)]
        );
        assert_eq!(split.total_singular_value_count, 4);
        assert_eq!(split.retained_singular_value_count, 3);
        assert_eq!(split.exact_zero_singular_values_omitted, 1);
        assert!(split.reconstruction_max_residual <= 1.0e-12);
        assert!(split.reconstruction_l2_residual <= 1.0e-12);

        let rebuilt =
            AbelianMps::new(vec![split.left_tensor, split.right_tensor], double, 1.0e-12)
                .unwrap();
        assert_eq!(rebuilt.total_charge(), double);
        assert_eq!(rebuilt.selection_rule_residual(), 0.0);
    }

    #[test]
    fn exact_u1k_two_site_svd_retains_tiny_positive_schmidt_value() {
        let vacuum = q(0, 0);
        let up = q(1, 1);
        let down = q(1, -1);
        let double = q(2, 0);
        let physical = vec![vacuum, up, down, double];
        let bond = vec![vacuum, up, down, double];
        let tiny: f64 = 1.0e-18;

        let mut first_data = vec![zero(); 16];
        first_data[5] = Complex64::new(1.0, 0.0);
        first_data[10] = Complex64::new(tiny, 0.0);
        let first = AbelianChargeTensor::new(
            0,
            vec![vacuum],
            physical.clone(),
            bond.clone(),
            first_data,
            1.0e-12,
        )
        .unwrap();

        let mut second_data = vec![zero(); 16];
        second_data[6] = Complex64::new(1.0, 0.0);
        second_data[9] = Complex64::new(1.0, 0.0);
        let second = AbelianChargeTensor::new(
            1,
            bond,
            physical,
            vec![double],
            second_data,
            1.0e-12,
        )
        .unwrap();

        let split =
            AbelianChargeBlockPlan::split_exact(&first, &second, 1.0e-12).unwrap();
        assert_eq!(split.rank_by_charge, vec![(vacuum, 0), (down, 1), (up, 1), (double, 0)]);
        assert_eq!(split.retained_singular_value_count, 2);
        let tiny_middle = split
            .left_tensor
            .right_charges()
            .iter()
            .position(|charge| *charge == down)
            .unwrap();
        let tiny_amplitude = split.left_tensor.get(0, 2, tiny_middle)
            * split.right_tensor.get(tiny_middle, 1, 0);
        assert!((tiny_amplitude.norm() - tiny).abs() <= 1.0e-30);
        assert!(split.reconstruction_max_residual <= 1.0e-15);
    }

    #[test]
    fn exact_u1k_two_site_svd_rejects_zero_state_and_mismatched_internal_bond() {
        let vacuum = q(0, 0);
        let physical = vec![vacuum];
        let first = AbelianChargeTensor::new(
            0,
            vec![vacuum],
            physical.clone(),
            vec![vacuum],
            vec![zero()],
            1.0e-12,
        )
        .unwrap();
        let second = AbelianChargeTensor::new(
            1,
            vec![vacuum],
            physical.clone(),
            vec![vacuum],
            vec![zero()],
            1.0e-12,
        )
        .unwrap();
        assert!(matches!(
            AbelianChargeBlockPlan::split_exact(&first, &second, 1.0e-12),
            Err(AbelianChargeError::ZeroTwoSiteState)
        ));

        let up = q(1, 1);
        let mismatched = AbelianChargeTensor::new(
            1,
            vec![up],
            physical,
            vec![vacuum],
            vec![zero()],
            1.0e-12,
        )
        .unwrap();
        assert!(matches!(
            AbelianChargeBlockPlan::split_exact(&first, &mismatched, 1.0e-12),
            Err(AbelianChargeError::BondChargeMismatch { .. })
        ));
    }


    #[test]
    fn u1k_truncation_applies_global_bond_cap_across_charge_vectors() {
        let (first, second) = two_sector_schmidt_pair();
        let result = AbelianChargeBlockPlan::split_truncated(
            &first,
            &second,
            AbelianTwoSiteTruncationPolicy {
                max_bond_dimension: 1,
                relative_singular_value_cutoff: 0.0,
                tolerance: 1.0e-12,
            },
        )
        .unwrap();

        assert_eq!(result.total_singular_value_count, 4);
        assert_eq!(result.retained_singular_value_count, 1);
        assert_eq!(result.cutoff_discarded_singular_value_count, 0);
        assert_eq!(result.bond_cap_discarded_singular_value_count, 1);
        assert!(result.bond_cap_binding);
        assert!((result.total_schmidt_weight - 1.0).abs() <= 1.0e-12);
        assert!((result.retained_schmidt_weight - 0.64).abs() <= 1.0e-12);
        assert!((result.bond_cap_discarded_weight - 0.36).abs() <= 1.0e-12);
        assert!((result.discarded_weight - 0.36).abs() <= 1.0e-12);
        assert!((result.relative_discarded_weight - 0.36).abs() <= 1.0e-12);
        assert!((result.renormalization_factor_to_restore_original_norm - 1.25).abs() <= 1.0e-12);
        assert_eq!(result.retained_rank_by_charge.len(), 1);
        assert!(result.reconstruction_l2_residual >= 0.6 - 1.0e-12);
    }

    #[test]
    fn u1k_relative_cutoff_and_global_bond_cap_have_separate_loss_receipts() {
        let (first, second) = two_sector_schmidt_pair();
        let result = AbelianChargeBlockPlan::split_truncated(
            &first,
            &second,
            AbelianTwoSiteTruncationPolicy {
                max_bond_dimension: 8,
                relative_singular_value_cutoff: 0.8,
                tolerance: 1.0e-12,
            },
        )
        .unwrap();

        assert_eq!(result.retained_singular_value_count, 1);
        assert_eq!(result.cutoff_discarded_singular_value_count, 1);
        assert_eq!(result.bond_cap_discarded_singular_value_count, 0);
        assert!(!result.bond_cap_binding);
        assert!((result.cutoff_discarded_weight - 0.36).abs() <= 1.0e-12);
        assert!(result.bond_cap_discarded_weight.abs() <= 1.0e-12);
    }

    #[test]
    fn u1k_truncation_preserves_charge_vectors_and_does_not_renormalize() {
        let (first, second) = two_sector_schmidt_pair();
        let result = AbelianChargeBlockPlan::split_truncated(
            &first,
            &second,
            AbelianTwoSiteTruncationPolicy {
                max_bond_dimension: 1,
                relative_singular_value_cutoff: 0.0,
                tolerance: 1.0e-12,
            },
        )
        .unwrap();

        assert_eq!(
            result.left_tensor.right_charges(),
            result.right_tensor.left_charges()
        );
        assert_eq!(
            contracted_two_site_norm_squared(&result.left_tensor, &result.right_tensor),
            result.post_truncation_norm_squared
        );
        assert!((result.post_truncation_norm_squared - 0.64).abs() <= 1.0e-12);
        assert!(result.post_truncation_norm_squared < result.pre_truncation_norm_squared);
        let rebuilt = AbelianMps::new(
            vec![result.left_tensor, result.right_tensor],
            q(2, 0),
            1.0e-12,
        )
        .unwrap();
        assert_eq!(rebuilt.selection_rule_residual(), 0.0);
    }

    #[test]
    fn u1k_truncation_rejects_invalid_global_policy() {
        let (first, second) = two_sector_schmidt_pair();
        for policy in [
            AbelianTwoSiteTruncationPolicy {
                max_bond_dimension: 0,
                relative_singular_value_cutoff: 0.0,
                tolerance: 1.0e-12,
            },
            AbelianTwoSiteTruncationPolicy {
                max_bond_dimension: 1,
                relative_singular_value_cutoff: -0.1,
                tolerance: 1.0e-12,
            },
            AbelianTwoSiteTruncationPolicy {
                max_bond_dimension: 1,
                relative_singular_value_cutoff: 1.1,
                tolerance: 1.0e-12,
            },
            AbelianTwoSiteTruncationPolicy {
                max_bond_dimension: 1,
                relative_singular_value_cutoff: 0.0,
                tolerance: f64::NAN,
            },
        ] {
            assert!(matches!(
                AbelianChargeBlockPlan::split_truncated(&first, &second, policy),
                Err(AbelianChargeError::InvalidTruncationPolicy { .. })
            ));
        }
    }

}
