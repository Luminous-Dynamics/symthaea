use super::{
    ApproximateSymmetryInputV1, ApproximateSymmetryReceiptV1, EnvelopeBudgetV1, EnvelopeErrorV1,
    qualify_approximate_symmetry, verify_receipt,
};

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exact affine derivation of one-step symmetry defect and domain bounds.
//!
//! All matrices and box endpoints are integers. The certified norm is L-infinity.
//! This module derives bounds for affine systems only; it is not a nonlinear
//! interval-analysis engine.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegerAffineMapV1 {
    /// Row-major integer matrix.
    pub matrix: Vec<Vec<i128>>,
    pub offset: Vec<i128>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegerBoxV1 {
    pub lower: Vec<i128>,
    pub upper: Vec<i128>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineSymmetryDomainV1 {
    pub contract_id: String,
    pub step_map: IntegerAffineMapV1,
    pub action: IntegerAffineMapV1,
    pub action_order: u32,
    /// K: reference trajectories must remain in this box.
    pub reference_region: IntegerBoxV1,
    /// D: Lipschitz and residual estimates apply throughout this larger box.
    pub certificate_domain: IntegerBoxV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineSymmetryWitnessV1 {
    pub schema_version: u32,
    pub contract_id: String,
    pub dimension: usize,
    pub one_step_equivariance_defect_linf: u128,
    pub step_lipschitz_linf: u128,
    pub domain_boundary_margin_linf: u128,
    pub reference_region_invariant: bool,
    pub action_has_declared_finite_order: bool,
    pub certificate_domain_contains_reference_region: bool,
    pub norm: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AffineErrorV1 {
    EmptyContractId,
    InvalidDimension,
    ShapeMismatch,
    ReversedBoxInterval { axis: usize },
    ReferenceRegionOutsideCertificateDomain { axis: usize },
    ReferenceRegionNotInvariant { map: String, axis: usize },
    InvalidActionOrder,
    ActionOrderMismatch,
    ArithmeticOverflow,
}

fn checked_add(left: i128, right: i128) -> Result<i128, AffineErrorV1> {
    left.checked_add(right)
        .ok_or(AffineErrorV1::ArithmeticOverflow)
}

fn checked_sub(left: i128, right: i128) -> Result<i128, AffineErrorV1> {
    left.checked_sub(right)
        .ok_or(AffineErrorV1::ArithmeticOverflow)
}

fn checked_mul(left: i128, right: i128) -> Result<i128, AffineErrorV1> {
    left.checked_mul(right)
        .ok_or(AffineErrorV1::ArithmeticOverflow)
}

fn abs_u128(value: i128) -> Result<u128, AffineErrorV1> {
    value
        .checked_abs()
        .map(|value| value as u128)
        .ok_or(AffineErrorV1::ArithmeticOverflow)
}

fn validate_map(map: &IntegerAffineMapV1, dimension: usize) -> Result<(), AffineErrorV1> {
    if map.matrix.len() != dimension
        || map.matrix.iter().any(|row| row.len() != dimension)
        || map.offset.len() != dimension
    {
        return Err(AffineErrorV1::ShapeMismatch);
    }
    Ok(())
}

fn validate_box(region: &IntegerBoxV1, dimension: usize) -> Result<(), AffineErrorV1> {
    if region.lower.len() != dimension || region.upper.len() != dimension {
        return Err(AffineErrorV1::ShapeMismatch);
    }
    for axis in 0..dimension {
        if region.lower[axis] > region.upper[axis] {
            return Err(AffineErrorV1::ReversedBoxInterval { axis });
        }
    }
    Ok(())
}

fn identity(dimension: usize) -> Vec<Vec<i128>> {
    let mut matrix = vec![vec![0_i128; dimension]; dimension];
    for (axis, row) in matrix.iter_mut().enumerate() {
        row[axis] = 1;
    }
    matrix
}

fn matrix_multiply(
    left: &[Vec<i128>],
    right: &[Vec<i128>],
) -> Result<Vec<Vec<i128>>, AffineErrorV1> {
    let dimension = left.len();
    let mut product = vec![vec![0_i128; dimension]; dimension];
    for row in 0..dimension {
        for column in 0..dimension {
            let mut value = 0_i128;
            for inner in 0..dimension {
                value = checked_add(
                    value,
                    checked_mul(left[row][inner], right[inner][column])?,
                )?;
            }
            product[row][column] = value;
        }
    }
    Ok(product)
}

fn matrix_vector(
    matrix: &[Vec<i128>],
    vector: &[i128],
) -> Result<Vec<i128>, AffineErrorV1> {
    let mut output = vec![0_i128; matrix.len()];
    for row in 0..matrix.len() {
        let mut value = 0_i128;
        for column in 0..vector.len() {
            value = checked_add(
                value,
                checked_mul(matrix[row][column], vector[column])?,
            )?;
        }
        output[row] = value;
    }
    Ok(output)
}

fn add_vectors(left: &[i128], right: &[i128]) -> Result<Vec<i128>, AffineErrorV1> {
    left.iter()
        .zip(right)
        .map(|(a, b)| checked_add(*a, *b))
        .collect()
}

fn affine_image_interval(
    map: &IntegerAffineMapV1,
    region: &IntegerBoxV1,
    output_axis: usize,
) -> Result<(i128, i128), AffineErrorV1> {
    let mut minimum = map.offset[output_axis];
    let mut maximum = map.offset[output_axis];
    for input_axis in 0..region.lower.len() {
        let coefficient = map.matrix[output_axis][input_axis];
        let lower_product = checked_mul(coefficient, region.lower[input_axis])?;
        let upper_product = checked_mul(coefficient, region.upper[input_axis])?;
        minimum = checked_add(minimum, lower_product.min(upper_product))?;
        maximum = checked_add(maximum, lower_product.max(upper_product))?;
    }
    Ok((minimum, maximum))
}

fn verify_invariant_region(
    map: &IntegerAffineMapV1,
    region: &IntegerBoxV1,
    map_name: &str,
) -> Result<(), AffineErrorV1> {
    for axis in 0..region.lower.len() {
        let (minimum, maximum) = affine_image_interval(map, region, axis)?;
        if minimum < region.lower[axis] || maximum > region.upper[axis] {
            return Err(AffineErrorV1::ReferenceRegionNotInvariant {
                map: map_name.to_owned(),
                axis,
            });
        }
    }
    Ok(())
}

fn verify_action_order(
    action: &IntegerAffineMapV1,
    declared_order: u32,
) -> Result<(), AffineErrorV1> {
    if declared_order == 0 || declared_order > 64 {
        return Err(AffineErrorV1::InvalidActionOrder);
    }
    let dimension = action.matrix.len();
    let mut power_matrix = identity(dimension);
    let mut power_offset = vec![0_i128; dimension];
    for _ in 0..declared_order {
        power_matrix = matrix_multiply(&action.matrix, &power_matrix)?;
        power_offset = add_vectors(
            &matrix_vector(&action.matrix, &power_offset)?,
            &action.offset,
        )?;
    }
    if power_matrix != identity(dimension) || power_offset.iter().any(|value| *value != 0) {
        return Err(AffineErrorV1::ActionOrderMismatch);
    }
    Ok(())
}

fn derive_commutator(
    step_map: &IntegerAffineMapV1,
    action: &IntegerAffineMapV1,
) -> Result<(Vec<Vec<i128>>, Vec<i128>), AffineErrorV1> {
    // Phi(Tx) - T(Phi(x)) = (A G - G A)x + (A c + b - G b - c).
    let ag = matrix_multiply(&step_map.matrix, &action.matrix)?;
    let ga = matrix_multiply(&action.matrix, &step_map.matrix)?;
    let mut linear = vec![vec![0_i128; ag.len()]; ag.len()];
    for row in 0..ag.len() {
        for column in 0..ag.len() {
            linear[row][column] = checked_sub(ag[row][column], ga[row][column])?;
        }
    }
    let ac = matrix_vector(&step_map.matrix, &action.offset)?;
    let gb = matrix_vector(&action.matrix, &step_map.offset)?;
    let first = add_vectors(&ac, &step_map.offset)?;
    let second = add_vectors(&gb, &action.offset)?;
    let constant = first
        .iter()
        .zip(second)
        .map(|(left, right)| checked_sub(*left, right))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((linear, constant))
}

fn box_max_abs(region: &IntegerBoxV1, axis: usize) -> Result<u128, AffineErrorV1> {
    Ok(abs_u128(region.lower[axis])?.max(abs_u128(region.upper[axis])?))
}

fn residual_linf_bound(
    linear: &[Vec<i128>],
    constant: &[i128],
    region: &IntegerBoxV1,
) -> Result<u128, AffineErrorV1> {
    let mut maximum = 0_u128;
    for row in 0..linear.len() {
        let mut row_bound = abs_u128(constant[row])?;
        for column in 0..linear.len() {
            let term = abs_u128(linear[row][column])?
                .checked_mul(box_max_abs(region, column)?)
                .ok_or(AffineErrorV1::ArithmeticOverflow)?;
            row_bound = row_bound
                .checked_add(term)
                .ok_or(AffineErrorV1::ArithmeticOverflow)?;
        }
        maximum = maximum.max(row_bound);
    }
    Ok(maximum)
}

fn lipschitz_linf_bound(matrix: &[Vec<i128>]) -> Result<u128, AffineErrorV1> {
    let mut maximum = 0_u128;
    for row in matrix {
        let mut row_sum = 0_u128;
        for value in row {
            row_sum = row_sum
                .checked_add(abs_u128(*value)?)
                .ok_or(AffineErrorV1::ArithmeticOverflow)?;
        }
        maximum = maximum.max(row_sum);
    }
    Ok(maximum)
}

fn domain_margin(
    reference: &IntegerBoxV1,
    domain: &IntegerBoxV1,
) -> Result<u128, AffineErrorV1> {
    let mut margin = u128::MAX;
    for axis in 0..reference.lower.len() {
        if reference.lower[axis] < domain.lower[axis]
            || reference.upper[axis] > domain.upper[axis]
        {
            return Err(AffineErrorV1::ReferenceRegionOutsideCertificateDomain { axis });
        }
        let lower_gap = reference.lower[axis]
            .checked_sub(domain.lower[axis])
            .ok_or(AffineErrorV1::ArithmeticOverflow)? as u128;
        let upper_gap = domain.upper[axis]
            .checked_sub(reference.upper[axis])
            .ok_or(AffineErrorV1::ArithmeticOverflow)? as u128;
        margin = margin.min(lower_gap).min(upper_gap);
    }
    Ok(margin)
}

/// Derive one-step defect, Lipschitz and domain-margin bounds for an exact
/// integer affine map/action pair over a forward-invariant reference box K
/// strictly inside a larger certificate box D.
///
/// Phi and T must both map K into K. The one-step defect is bounded over K;
/// the Phi Lipschitz constant is global for the affine map and therefore
/// applies to the perturbed trajectory while it remains in D. The returned
/// margin is the L-infinity distance from K to the boundary of D.
pub fn derive_affine_witness(
    contract: &AffineSymmetryDomainV1,
) -> Result<AffineSymmetryWitnessV1, AffineErrorV1> {
    if contract.contract_id.trim().is_empty() {
        return Err(AffineErrorV1::EmptyContractId);
    }
    let dimension = contract.reference_region.lower.len();
    if dimension == 0 || dimension > 32 {
        return Err(AffineErrorV1::InvalidDimension);
    }
    validate_map(&contract.step_map, dimension)?;
    validate_map(&contract.action, dimension)?;
    validate_box(&contract.reference_region, dimension)?;
    validate_box(&contract.certificate_domain, dimension)?;
    verify_action_order(&contract.action, contract.action_order)?;
    verify_invariant_region(&contract.step_map, &contract.reference_region, "Phi")?;
    verify_invariant_region(&contract.action, &contract.reference_region, "T")?;
    let margin = domain_margin(&contract.reference_region, &contract.certificate_domain)?;
    let (linear, constant) = derive_commutator(&contract.step_map, &contract.action)?;
    let defect = residual_linf_bound(&linear, &constant, &contract.reference_region)?;
    let lipschitz = lipschitz_linf_bound(&contract.step_map.matrix)?;

    Ok(AffineSymmetryWitnessV1 {
        schema_version: 1,
        contract_id: contract.contract_id.clone(),
        dimension,
        one_step_equivariance_defect_linf: defect,
        step_lipschitz_linf: lipschitz,
        domain_boundary_margin_linf: margin,
        reference_region_invariant: true,
        action_has_declared_finite_order: true,
        certificate_domain_contains_reference_region: true,
        norm: "L-infinity".to_owned(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineTaskBoundsV1 {
    pub task_contract_id: String,
    pub state_unit_id: String,
    pub objective_unit_id: String,
    pub constraint_unit_id: String,
    /// Fixed-point scale for the state/objective/constraint units.
    pub scale: u128,
    pub initial_state_error: u128,
    pub horizon_steps: u32,
    pub objective_lipschitz_scaled: u128,
    pub direct_objective_defect: u128,
    pub constraint_lipschitz_scaled: u128,
    pub direct_constraint_defect: u128,
    pub constraint_slack: u128,
    pub max_state_error: u128,
    pub max_objective_error: u128,
    pub max_constraint_error: u128,
    pub objective_gap: Option<u128>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineEnvelopeReceiptV1 {
    pub schema_version: u32,
    pub affine_witness: AffineSymmetryWitnessV1,
    pub envelope_receipt: ApproximateSymmetryReceiptV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AffineTaskErrorV1 {
    Affine(AffineErrorV1),
    Envelope(EnvelopeErrorV1),
    ArithmeticOverflow,
}

impl From<AffineErrorV1> for AffineTaskErrorV1 {
    fn from(value: AffineErrorV1) -> Self {
        Self::Affine(value)
    }
}

impl From<EnvelopeErrorV1> for AffineTaskErrorV1 {
    fn from(value: EnvelopeErrorV1) -> Self {
        Self::Envelope(value)
    }
}

fn scaled(value: u128, scale: u128) -> Result<u128, AffineTaskErrorV1> {
    value
        .checked_mul(scale)
        .ok_or(AffineTaskErrorV1::ArithmeticOverflow)
}

/// Derive the affine witness and feed its exact defect, Lipschitz factor and
/// domain margin into the finite-horizon envelope. Objective and constraint
/// bounds remain explicit task inputs with their own units.
pub fn qualify_affine_task(
    contract: &AffineSymmetryDomainV1,
    task: &AffineTaskBoundsV1,
    budget: EnvelopeBudgetV1,
) -> Result<AffineEnvelopeReceiptV1, AffineTaskErrorV1> {
    let witness = derive_affine_witness(contract)?;
    let scale = task.scale;
    let input = ApproximateSymmetryInputV1 {
        contract_id: task.task_contract_id.clone(),
        state_unit_id: task.state_unit_id.clone(),
        objective_unit_id: task.objective_unit_id.clone(),
        constraint_unit_id: task.constraint_unit_id.clone(),
        defect_evidence_id: format!("{}::affine-commutator-v1", contract.contract_id),
        lipschitz_evidence_id: format!("{}::affine-lipschitz-v1", contract.contract_id),
        domain_evidence_id: format!("{}::invariant-box-margin-v1", contract.contract_id),
        scale,
        initial_state_error: task.initial_state_error,
        one_step_equivariance_defect: scaled(
            witness.one_step_equivariance_defect_linf,
            scale,
        )?,
        step_lipschitz_scaled: scaled(witness.step_lipschitz_linf, scale)?,
        horizon_steps: task.horizon_steps,
        domain_invariance_verified: witness.reference_region_invariant
            && witness.action_has_declared_finite_order
            && witness.certificate_domain_contains_reference_region,
        domain_boundary_margin: scaled(witness.domain_boundary_margin_linf, scale)?,
        objective_lipschitz_scaled: task.objective_lipschitz_scaled,
        direct_objective_defect: task.direct_objective_defect,
        constraint_lipschitz_scaled: task.constraint_lipschitz_scaled,
        direct_constraint_defect: task.direct_constraint_defect,
        constraint_slack: task.constraint_slack,
        max_state_error: task.max_state_error,
        max_objective_error: task.max_objective_error,
        max_constraint_error: task.max_constraint_error,
        objective_gap: task.objective_gap,
    };
    let envelope_receipt = qualify_approximate_symmetry(&input, budget)?;
    Ok(AffineEnvelopeReceiptV1 {
        schema_version: 1,
        affine_witness: witness,
        envelope_receipt,
    })
}

pub fn verify_affine_task_receipt(
    contract: &AffineSymmetryDomainV1,
    task: &AffineTaskBoundsV1,
    receipt: &AffineEnvelopeReceiptV1,
) -> Result<(), AffineTaskErrorV1> {
    if receipt.schema_version != 1 {
        return Err(AffineTaskErrorV1::Envelope(
            EnvelopeErrorV1::ReceiptMismatch,
        ));
    }
    let expected = qualify_affine_task(
        contract,
        task,
        receipt.envelope_receipt.budget,
    )?;
    if *receipt != expected {
        return Err(AffineTaskErrorV1::Envelope(
            EnvelopeErrorV1::ReceiptMismatch,
        ));
    }
    verify_receipt(
        &ApproximateSymmetryInputV1 {
            contract_id: task.task_contract_id.clone(),
            state_unit_id: task.state_unit_id.clone(),
            objective_unit_id: task.objective_unit_id.clone(),
            constraint_unit_id: task.constraint_unit_id.clone(),
            defect_evidence_id: format!("{}::affine-commutator-v1", contract.contract_id),
            lipschitz_evidence_id: format!("{}::affine-lipschitz-v1", contract.contract_id),
            domain_evidence_id: format!("{}::invariant-box-margin-v1", contract.contract_id),
            scale: task.scale,
            initial_state_error: task.initial_state_error,
            one_step_equivariance_defect: scaled(
                receipt.affine_witness.one_step_equivariance_defect_linf,
                task.scale,
            )?,
            step_lipschitz_scaled: scaled(
                receipt.affine_witness.step_lipschitz_linf,
                task.scale,
            )?,
            horizon_steps: task.horizon_steps,
            domain_invariance_verified: receipt.affine_witness.reference_region_invariant
                && receipt.affine_witness.action_has_declared_finite_order
                && receipt.affine_witness.certificate_domain_contains_reference_region,
            domain_boundary_margin: scaled(
                receipt.affine_witness.domain_boundary_margin_linf,
                task.scale,
            )?,
            objective_lipschitz_scaled: task.objective_lipschitz_scaled,
            direct_objective_defect: task.direct_objective_defect,
            constraint_lipschitz_scaled: task.constraint_lipschitz_scaled,
            direct_constraint_defect: task.direct_constraint_defect,
            constraint_slack: task.constraint_slack,
            max_state_error: task.max_state_error,
            max_objective_error: task.max_objective_error,
            max_constraint_error: task.max_constraint_error,
            objective_gap: task.objective_gap,
        },
        &receipt.envelope_receipt,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract() -> AffineSymmetryDomainV1 {
        AffineSymmetryDomainV1 {
            contract_id: "affine-swap-projection-v1".to_owned(),
            step_map: IntegerAffineMapV1 {
                matrix: vec![vec![1, 0], vec![0, 0]],
                offset: vec![0, 0],
            },
            action: IntegerAffineMapV1 {
                matrix: vec![vec![0, 1], vec![1, 0]],
                offset: vec![0, 0],
            },
            action_order: 2,
            reference_region: IntegerBoxV1 {
                lower: vec![-1, -1],
                upper: vec![1, 1],
            },
            certificate_domain: IntegerBoxV1 {
                lower: vec![-3, -3],
                upper: vec![3, 3],
            },
        }
    }

    #[test]
    fn derived_affine_bounds_feed_error_envelope_and_replay_as_one_receipt() {
        let contract = contract();
        let task = AffineTaskBoundsV1 {
            task_contract_id: "projection-task-v1".to_owned(),
            state_unit_id: "metre".to_owned(),
            objective_unit_id: "joule".to_owned(),
            constraint_unit_id: "metre-squared".to_owned(),
            scale: 1_000,
            initial_state_error: 0,
            horizon_steps: 2,
            objective_lipschitz_scaled: 1_000,
            direct_objective_defect: 0,
            constraint_lipschitz_scaled: 1_000,
            direct_constraint_defect: 0,
            constraint_slack: 3_000,
            max_state_error: 2_000,
            max_objective_error: 2_000,
            max_constraint_error: 2_000,
            objective_gap: Some(5_000),
        };
        let receipt = qualify_affine_task(
            &contract,
            &task,
            EnvelopeBudgetV1::default(),
        )
        .unwrap();
        assert_eq!(receipt.affine_witness.one_step_equivariance_defect_linf, 1);
        assert_eq!(receipt.affine_witness.step_lipschitz_linf, 1);
        assert_eq!(receipt.envelope_receipt.report.state_error_bound, 2_000);
        assert_eq!(receipt.envelope_receipt.report.status, super::super::EnvelopeStatusV1::WithinDeclaredErrorBudgets);
        assert!(receipt.envelope_receipt.report.epsilon_lift_candidate);
        assert!(!receipt.envelope_receipt.report.hard_prune_authorized);
        verify_affine_task_receipt(&contract, &task, &receipt).unwrap();
    }

    #[test]
    fn affine_error_bound_exceeding_derived_domain_margin_abstains() {
        let contract = contract();
        let task = AffineTaskBoundsV1 {
            task_contract_id: "projection-task-three-steps-v1".to_owned(),
            state_unit_id: "metre".to_owned(),
            objective_unit_id: "joule".to_owned(),
            constraint_unit_id: "metre-squared".to_owned(),
            scale: 1_000,
            initial_state_error: 0,
            horizon_steps: 3,
            objective_lipschitz_scaled: 1_000,
            direct_objective_defect: 0,
            constraint_lipschitz_scaled: 1_000,
            direct_constraint_defect: 0,
            constraint_slack: 4_000,
            max_state_error: 3_000,
            max_objective_error: 3_000,
            max_constraint_error: 3_000,
            objective_gap: Some(10_000),
        };
        let receipt = qualify_affine_task(
            &contract,
            &task,
            EnvelopeBudgetV1::default(),
        )
        .unwrap();
        assert_eq!(receipt.envelope_receipt.report.state_error_bound, 3_000);
        assert_eq!(
            receipt.envelope_receipt.report.status,
            super::super::EnvelopeStatusV1::Abstain
        );
        assert!(!receipt.envelope_receipt.report.epsilon_lift_candidate);
    }

    #[test]
    fn affine_oracle_derives_exact_commutator_lipschitz_and_domain_margin() {
        let witness = derive_affine_witness(&contract()).unwrap();
        assert_eq!(witness.one_step_equivariance_defect_linf, 1);
        assert_eq!(witness.step_lipschitz_linf, 1);
        assert_eq!(witness.domain_boundary_margin_linf, 2);
        assert!(witness.reference_region_invariant);
        assert!(witness.action_has_declared_finite_order);
        assert!(witness.certificate_domain_contains_reference_region);
    }

    #[test]
    fn finite_order_action_is_checked_not_assumed() {
        let mut input = contract();
        input.action.matrix = vec![vec![1, 1], vec![0, 1]];
        assert_eq!(
            derive_affine_witness(&input),
            Err(AffineErrorV1::ActionOrderMismatch)
        );
    }

    #[test]
    fn noninvariant_reference_region_is_rejected() {
        let mut input = contract();
        input.step_map.matrix = vec![vec![2, 0], vec![0, 0]];
        assert_eq!(
            derive_affine_witness(&input),
            Err(AffineErrorV1::ReferenceRegionNotInvariant {
                map: "Phi".to_owned(),
                axis: 0
            })
        );
    }

    #[test]
    fn arithmetic_overflow_in_affine_lipschitz_bound_fails_closed() {
        let mut input = contract();
        input.step_map.matrix = vec![vec![i128::MIN, i128::MIN], vec![0, 0]];
        input.action.matrix = vec![vec![1, 0], vec![0, 1]];
        input.action_order = 1;
        input.reference_region = IntegerBoxV1 {
            lower: vec![0, 0],
            upper: vec![0, 0],
        };
        input.certificate_domain = IntegerBoxV1 {
            lower: vec![-1, -1],
            upper: vec![1, 1],
        };
        assert_eq!(
            derive_affine_witness(&input),
            Err(AffineErrorV1::ArithmeticOverflow)
        );
    }

    #[test]
    fn reference_region_must_fit_inside_certificate_domain() {
        let mut input = contract();
        input.certificate_domain.lower[0] = 0;
        assert_eq!(
            derive_affine_witness(&input),
            Err(AffineErrorV1::ReferenceRegionOutsideCertificateDomain { axis: 0 })
        );
    }
}
