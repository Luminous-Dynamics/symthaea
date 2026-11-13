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
}

#[cfg(test)]
mod tests {
    use super::{
        AbelianCharge, AbelianChargeBlockPlan, AbelianChargeError, AbelianChargeTensor, AbelianMps,
    };
    use num_complex::Complex64;

    fn q(first: i32, second: i32) -> AbelianCharge<2> {
        AbelianCharge::new([first, second]).unwrap()
    }

    fn zero() -> Complex64 {
        Complex64::new(0.0, 0.0)
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
}
