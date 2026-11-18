// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Fixed-point finite-horizon error envelopes for approximate discrete symmetries.
//!
//! This crate propagates caller-supplied defect, Lipschitz and domain bounds.
//! It does not prove those assumptions and never authorizes exact hard pruning.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnvelopeBudgetV1 {
    pub max_horizon_steps: u32,
}

impl Default for EnvelopeBudgetV1 {
    fn default() -> Self {
        Self {
            max_horizon_steps: 10_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApproximateSymmetryInputV1 {
    pub contract_id: String,
    /// All fixed-point quantities use this positive integer scale.
    pub scale: u128,
    pub initial_state_error: u128,
    /// Uniform bound on ||Phi(Tx) - T Phi(x)|| over the declared domain.
    pub one_step_equivariance_defect: u128,
    /// Lipschitz factor L multiplied by scale.
    pub step_lipschitz_scaled: u128,
    pub horizon_steps: u32,
    /// This must come from a separate domain-invariance qualification.
    pub domain_invariance_verified: bool,
    /// Lower bound on the reference trajectory's distance to the domain edge.
    pub domain_boundary_margin: u128,
    /// Objective Lipschitz factor multiplied by scale.
    pub objective_lipschitz_scaled: u128,
    /// Direct bound on objective equivariance defect.
    pub direct_objective_defect: u128,
    /// Constraint-function Lipschitz factor multiplied by scale.
    pub constraint_lipschitz_scaled: u128,
    /// Direct bound on constraint equivariance defect.
    pub direct_constraint_defect: u128,
    /// Lower bound on constraint slack on the reference trajectory.
    pub constraint_slack: u128,
    /// Declared error budgets.
    pub max_state_error: u128,
    pub max_objective_error: u128,
    pub max_constraint_error: u128,
    /// Optional gap between the two objective values being compared.
    pub objective_gap: Option<u128>,
}

impl ApproximateSymmetryInputV1 {
    pub fn validate(&self, budget: EnvelopeBudgetV1) -> Result<(), EnvelopeErrorV1> {
        if self.contract_id.trim().is_empty() {
            return Err(EnvelopeErrorV1::EmptyContractId);
        }
        if self.scale == 0 {
            return Err(EnvelopeErrorV1::ZeroScale);
        }
        if self.horizon_steps > budget.max_horizon_steps {
            return Err(EnvelopeErrorV1::HorizonBudgetExceeded {
                requested: self.horizon_steps,
                maximum: budget.max_horizon_steps,
            });
        }
        Ok(())
    }

    /// Versioned canonical bytes for deterministic receipt replay.
    /// These bytes are not a cryptographic digest or signature.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = b"luminous.approx_symmetry_envelope.v1\0".to_vec();
        append_bytes(&mut bytes, self.contract_id.as_bytes());
        for value in [
            self.scale,
            self.initial_state_error,
            self.one_step_equivariance_defect,
            self.step_lipschitz_scaled,
            u128::from(self.horizon_steps),
            u128::from(self.domain_invariance_verified),
            self.domain_boundary_margin,
            self.objective_lipschitz_scaled,
            self.direct_objective_defect,
            self.constraint_lipschitz_scaled,
            self.direct_constraint_defect,
            self.constraint_slack,
            self.max_state_error,
            self.max_objective_error,
            self.max_constraint_error,
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        match self.objective_gap {
            Some(gap) => {
                bytes.push(1);
                bytes.extend_from_slice(&gap.to_le_bytes());
            }
            None => bytes.push(0),
        }
        bytes
    }
}

fn append_bytes(target: &mut Vec<u8>, value: &[u8]) {
    target.extend_from_slice(&(value.len() as u64).to_le_bytes());
    target.extend_from_slice(value);
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvelopeErrorV1 {
    EmptyContractId,
    ZeroScale,
    HorizonBudgetExceeded { requested: u32, maximum: u32 },
    ArithmeticOverflow,
    ReceiptMismatch,
}

impl fmt::Display for EnvelopeErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for EnvelopeErrorV1 {}

fn ceil_mul_div(value: u128, factor_scaled: u128, scale: u128) -> Result<u128, EnvelopeErrorV1> {
    let product = value
        .checked_mul(factor_scaled)
        .ok_or(EnvelopeErrorV1::ArithmeticOverflow)?;
    let quotient = product / scale;
    let remainder = product % scale;
    if remainder == 0 {
        Ok(quotient)
    } else {
        quotient
            .checked_add(1)
            .ok_or(EnvelopeErrorV1::ArithmeticOverflow)
    }
}

fn checked_add(left: u128, right: u128) -> Result<u128, EnvelopeErrorV1> {
    left.checked_add(right)
        .ok_or(EnvelopeErrorV1::ArithmeticOverflow)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvelopeStatusV1 {
    ExactEquivarianceCandidate,
    WithinDeclaredErrorBudgets,
    Abstain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbstentionReasonV1 {
    DomainInvarianceUnverified,
    DomainMarginExceeded,
    StateErrorBudgetExceeded,
    ObjectiveErrorBudgetExceeded,
    ConstraintErrorBudgetExceeded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionStabilityV1 {
    Stable,
    Unresolved,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApproximateSymmetryReportV1 {
    pub schema_version: u32,
    pub status: EnvelopeStatusV1,
    pub state_error_bound: u128,
    pub objective_error_bound: u128,
    pub constraint_error_bound: u128,
    pub domain_margin_remaining: Option<u128>,
    pub constraint_slack_remaining: Option<u128>,
    pub objective_order: DecisionStabilityV1,
    pub constraint_feasibility: DecisionStabilityV1,
    pub abstention_reasons: Vec<AbstentionReasonV1>,
    /// This reference deliberately never grants exact pruning authority.
    pub hard_prune_authorized: bool,
    /// True only when the approximate result is inside all declared budgets,
    /// domain and constraint margins, and the objective gap is stable.
    pub epsilon_lift_candidate: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApproximateSymmetryReceiptV1 {
    pub schema_version: u32,
    pub input_contract_bytes: Vec<u8>,
    pub budget: EnvelopeBudgetV1,
    pub report: ApproximateSymmetryReportV1,
}

fn is_zero_defect(input: &ApproximateSymmetryInputV1, state_error: u128, objective_error: u128, constraint_error: u128) -> bool {
    state_error == 0
        && objective_error == 0
        && constraint_error == 0
        && input.one_step_equivariance_defect == 0
        && input.direct_objective_defect == 0
        && input.direct_constraint_defect == 0
        && input.initial_state_error == 0
}

fn compute_report(
    input: &ApproximateSymmetryInputV1,
    budget: EnvelopeBudgetV1,
) -> Result<ApproximateSymmetryReportV1, EnvelopeErrorV1> {
    input.validate(budget)?;
    let mut state_error = input.initial_state_error;
    for _ in 0..input.horizon_steps {
        state_error = checked_add(
            ceil_mul_div(state_error, input.step_lipschitz_scaled, input.scale)?,
            input.one_step_equivariance_defect,
        )?;
    }

    let objective_error = checked_add(
        ceil_mul_div(state_error, input.objective_lipschitz_scaled, input.scale)?,
        input.direct_objective_defect,
    )?;
    let constraint_error = checked_add(
        ceil_mul_div(state_error, input.constraint_lipschitz_scaled, input.scale)?,
        input.direct_constraint_defect,
    )?;

    let domain_margin_remaining =
        input.domain_boundary_margin.checked_sub(state_error);
    let constraint_slack_remaining = input.constraint_slack.checked_sub(constraint_error);
    let objective_order = match input.objective_gap {
        Some(gap) if objective_error.checked_mul(2).is_some_and(|twice| gap > twice) => {
            DecisionStabilityV1::Stable
        }
        _ => DecisionStabilityV1::Unresolved,
    };
    let constraint_feasibility = if input.constraint_slack > constraint_error {
        DecisionStabilityV1::Stable
    } else {
        DecisionStabilityV1::Unresolved
    };

    let mut abstention_reasons = Vec::new();
    if !input.domain_invariance_verified {
        abstention_reasons.push(AbstentionReasonV1::DomainInvarianceUnverified);
    }
    if domain_margin_remaining.is_none() {
        abstention_reasons.push(AbstentionReasonV1::DomainMarginExceeded);
    }
    if state_error > input.max_state_error {
        abstention_reasons.push(AbstentionReasonV1::StateErrorBudgetExceeded);
    }
    if objective_error > input.max_objective_error {
        abstention_reasons.push(AbstentionReasonV1::ObjectiveErrorBudgetExceeded);
    }
    if constraint_error > input.max_constraint_error {
        abstention_reasons.push(AbstentionReasonV1::ConstraintErrorBudgetExceeded);
    }

    let status = if !abstention_reasons.is_empty() {
        EnvelopeStatusV1::Abstain
    } else if is_zero_defect(input, state_error, objective_error, constraint_error) {
        EnvelopeStatusV1::ExactEquivarianceCandidate
    } else {
        EnvelopeStatusV1::WithinDeclaredErrorBudgets
    };

    let epsilon_lift_candidate = status != EnvelopeStatusV1::Abstain
        && constraint_feasibility == DecisionStabilityV1::Stable
        && objective_order == DecisionStabilityV1::Stable;

    Ok(ApproximateSymmetryReportV1 {
        schema_version: 1,
        status,
        state_error_bound: state_error,
        objective_error_bound: objective_error,
        constraint_error_bound: constraint_error,
        domain_margin_remaining,
        constraint_slack_remaining,
        objective_order,
        constraint_feasibility,
        abstention_reasons,
        hard_prune_authorized: false,
        epsilon_lift_candidate,
    })
}

pub fn qualify_approximate_symmetry(
    input: &ApproximateSymmetryInputV1,
    budget: EnvelopeBudgetV1,
) -> Result<ApproximateSymmetryReceiptV1, EnvelopeErrorV1> {
    let report = compute_report(input, budget)?;
    Ok(ApproximateSymmetryReceiptV1 {
        schema_version: 1,
        input_contract_bytes: input.canonical_bytes(),
        budget,
        report,
    })
}

pub fn verify_receipt(
    input: &ApproximateSymmetryInputV1,
    receipt: &ApproximateSymmetryReceiptV1,
) -> Result<(), EnvelopeErrorV1> {
    if receipt.schema_version != 1 || receipt.input_contract_bytes != input.canonical_bytes() {
        return Err(EnvelopeErrorV1::ReceiptMismatch);
    }
    let expected = qualify_approximate_symmetry(input, receipt.budget)?;
    if *receipt != expected {
        return Err(EnvelopeErrorV1::ReceiptMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> ApproximateSymmetryInputV1 {
        ApproximateSymmetryInputV1 {
            contract_id: "discrete-approx-symmetry-three-step-v1".to_owned(),
            scale: 1_000,
            initial_state_error: 0,
            one_step_equivariance_defect: 2,
            step_lipschitz_scaled: 1_500,
            horizon_steps: 3,
            domain_invariance_verified: true,
            domain_boundary_margin: 10,
            objective_lipschitz_scaled: 4_000,
            direct_objective_defect: 1,
            constraint_lipschitz_scaled: 2_000,
            direct_constraint_defect: 0,
            constraint_slack: 21,
            max_state_error: 10,
            max_objective_error: 41,
            max_constraint_error: 20,
            objective_gap: Some(83),
        }
    }

    #[test]
    fn fixed_point_recurrence_rounds_up_at_every_horizon_step() {
        let receipt =
            qualify_approximate_symmetry(&fixture(), EnvelopeBudgetV1::default()).unwrap();
        assert_eq!(receipt.report.state_error_bound, 10);
        assert_eq!(receipt.report.objective_error_bound, 41);
        assert_eq!(receipt.report.constraint_error_bound, 20);
        assert_eq!(receipt.report.status, EnvelopeStatusV1::WithinDeclaredErrorBudgets);
        assert_eq!(receipt.report.objective_order, DecisionStabilityV1::Stable);
        assert_eq!(
            receipt.report.constraint_feasibility,
            DecisionStabilityV1::Stable
        );
        assert!(receipt.report.epsilon_lift_candidate);
        assert!(!receipt.report.hard_prune_authorized);
    }

    #[test]
    fn unit_lipschitz_accumulates_constant_one_step_defect_linearly() {
        let mut input = fixture();
        input.horizon_steps = 5;
        input.step_lipschitz_scaled = input.scale;
        input.one_step_equivariance_defect = 3;
        input.initial_state_error = 0;
        input.max_state_error = 15;
        input.domain_boundary_margin = 15;
        input.objective_lipschitz_scaled = 0;
        input.direct_objective_defect = 0;
        input.max_objective_error = 0;
        input.objective_gap = Some(1);
        input.constraint_lipschitz_scaled = 0;
        input.constraint_slack = 1;
        input.direct_constraint_defect = 0;
        input.max_constraint_error = 0;
        let receipt =
            qualify_approximate_symmetry(&input, EnvelopeBudgetV1::default()).unwrap();
        assert_eq!(receipt.report.state_error_bound, 15);
    }

    #[test]
    fn expansive_map_grows_the_bound_geometrically() {
        let mut input = fixture();
        input.horizon_steps = 3;
        input.step_lipschitz_scaled = 2_000;
        input.one_step_equivariance_defect = 1;
        input.max_state_error = 7;
        input.domain_boundary_margin = 7;
        input.objective_lipschitz_scaled = 0;
        input.direct_objective_defect = 0;
        input.max_objective_error = 0;
        input.objective_gap = Some(1);
        input.constraint_lipschitz_scaled = 0;
        input.direct_constraint_defect = 0;
        input.max_constraint_error = 0;
        input.constraint_slack = 1;
        let receipt =
            qualify_approximate_symmetry(&input, EnvelopeBudgetV1::default()).unwrap();
        assert_eq!(receipt.report.state_error_bound, 7);
    }

    #[test]
    fn domain_invariance_and_boundary_margin_are_mandatory() {
        let mut input = fixture();
        input.domain_invariance_verified = false;
        let receipt =
            qualify_approximate_symmetry(&input, EnvelopeBudgetV1::default()).unwrap();
        assert_eq!(receipt.report.status, EnvelopeStatusV1::Abstain);
        assert!(receipt.report.abstention_reasons.contains(
            &AbstentionReasonV1::DomainInvarianceUnverified
        ));

        input.domain_invariance_verified = true;
        input.domain_boundary_margin = 9;
        let receipt =
            qualify_approximate_symmetry(&input, EnvelopeBudgetV1::default()).unwrap();
        assert_eq!(receipt.report.status, EnvelopeStatusV1::Abstain);
        assert!(receipt.report.abstention_reasons.contains(
            &AbstentionReasonV1::DomainMarginExceeded
        ));
    }

    #[test]
    fn insufficient_constraint_slack_blocks_epsilon_lift_candidate() {
        let mut input = fixture();
        input.constraint_slack = input.max_constraint_error;
        let receipt =
            qualify_approximate_symmetry(&input, EnvelopeBudgetV1::default()).unwrap();
        assert_eq!(
            receipt.report.constraint_feasibility,
            DecisionStabilityV1::Unresolved
        );
        assert!(!receipt.report.epsilon_lift_candidate);
    }

    #[test]
    fn insufficient_objective_gap_blocks_ranking_claim() {
        let mut input = fixture();
        input.objective_gap = Some(input.objective_error_bound_placeholder());
        let receipt =
            qualify_approximate_symmetry(&input, EnvelopeBudgetV1::default()).unwrap();
        assert_eq!(receipt.report.objective_order, DecisionStabilityV1::Unresolved);
        assert!(!receipt.report.epsilon_lift_candidate);
    }

    #[test]
    fn arithmetic_and_horizon_budgets_fail_closed() {
        let mut input = fixture();
        input.step_lipschitz_scaled = u128::MAX;
        input.scale = 1;
        input.initial_state_error = 2;
        assert_eq!(
            qualify_approximate_symmetry(&input, EnvelopeBudgetV1::default()),
            Err(EnvelopeErrorV1::ArithmeticOverflow)
        );

        input = fixture();
        input.horizon_steps = 11;
        assert_eq!(
            qualify_approximate_symmetry(&input, EnvelopeBudgetV1 { max_horizon_steps: 10 }),
            Err(EnvelopeErrorV1::HorizonBudgetExceeded {
                requested: 11,
                maximum: 10
            })
        );
    }

    #[test]
    fn exact_zero_defect_is_only_an_equivariance_candidate_not_prune_authority() {
        let mut input = fixture();
        input.one_step_equivariance_defect = 0;
        input.initial_state_error = 0;
        input.direct_objective_defect = 0;
        input.direct_constraint_defect = 0;
        input.horizon_steps = 8;
        let receipt =
            qualify_approximate_symmetry(&input, EnvelopeBudgetV1::default()).unwrap();
        assert_eq!(receipt.report.status, EnvelopeStatusV1::ExactEquivarianceCandidate);
        assert!(!receipt.report.hard_prune_authorized);
    }

    #[test]
    fn receipt_replay_rejects_mutated_inputs_and_forged_bounds() {
        let input = fixture();
        let receipt =
            qualify_approximate_symmetry(&input, EnvelopeBudgetV1::default()).unwrap();
        verify_receipt(&input, &receipt).unwrap();

        let mut forged = receipt.clone();
        forged.report.state_error_bound = 0;
        assert_eq!(verify_receipt(&input, &forged), Err(EnvelopeErrorV1::ReceiptMismatch));

        let mut changed = input.clone();
        changed.one_step_equivariance_defect += 1;
        assert_eq!(verify_receipt(&changed, &receipt), Err(EnvelopeErrorV1::ReceiptMismatch));
    }
}
