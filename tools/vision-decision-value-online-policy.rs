// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Online, budget-constrained escalation policy for decision-value-of-perception research.
//!
//! This is a cognitive-side reference policy, not a production vision integration.
//! The allocator accepts only a pre-escalation value estimate and declared resource
//! bounds. It cannot receive evaluator counterfactual losses or an escalated output.
//! The evaluator-only arithmetic belongs in tools/vision-decision-value-reference.rs.
//!
//! Standalone tests (no Cargo workspace membership or third-party dependencies):
//! rustc --edition=2024 --test tools/vision-decision-value-online-policy.rs -o /tmp/vision-dvpc-online-tests
//! /tmp/vision-dvpc-online-tests
//! rustc --edition=2024 -O tools/vision-decision-value-online-policy.rs -o /tmp/vision-dvpc-online
//! /tmp/vision-dvpc-online

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

pub type EvidenceDigest = [u8; 32];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationChoice {
    KeepCheap,
    Escalate,
    Abstain,
}

impl AllocationChoice {
    fn label(self) -> &'static str {
        match self {
            Self::KeepCheap => "keep_cheap",
            Self::Escalate => "escalate",
            Self::Abstain => "abstain",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationReason {
    PositiveConservativeValue,
    ValueBelowThreshold,
    ValueEstimateMissing,
    ValueEstimateInvalid,
    EscalationCostUnknown,
    BudgetUnavailable,
    CheapPathNotViable,
}

impl AllocationReason {
    fn label(self) -> &'static str {
        match self {
            Self::PositiveConservativeValue => "positive_conservative_value",
            Self::ValueBelowThreshold => "value_below_threshold",
            Self::ValueEstimateMissing => "value_estimate_missing",
            Self::ValueEstimateInvalid => "value_estimate_invalid",
            Self::EscalationCostUnknown => "escalation_cost_unknown",
            Self::BudgetUnavailable => "budget_unavailable",
            Self::CheapPathNotViable => "cheap_path_not_viable",
        }
    }
}

/// A calibrated estimate computed from permitted information available before
/// escalation. net_benefit_lower_bound is expressed in the named task-loss profile,
/// after any predeclared compute penalty in that same loss unit has been applied.
/// It is an estimate, never an evaluator counterfactual.
#[derive(Debug, Clone, PartialEq)]
pub struct PreEscalationValueEstimateV1 {
    pub estimator_id: String,
    pub estimator_artifact_digest: EvidenceDigest,
    pub loss_profile_id: String,
    pub source_input_digest: EvidenceDigest,
    pub net_benefit_lower_bound: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PreEscalationInputV1 {
    /// Opaque unique trial identifier. It must not encode hidden condition labels.
    pub trial_id: u64,
    pub loss_profile_id: String,
    /// Digest of the causal prefix and permitted pre-escalation features.
    pub input_digest: EvidenceDigest,
    /// Declared by the current cheap path/task contract, not by evaluator truth.
    pub cheap_path_safe: bool,
    pub value_estimate: Option<PreEscalationValueEstimateV1>,
    /// Conservative upper bound in declared resource units; not implicitly energy.
    pub escalation_cost_upper_bound_units: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocationError {
    InvalidConfig,
    InvalidTrialId,
    EmptyLossProfile(u64),
    MismatchedLossProfile(u64),
    ZeroInputDigest(u64),
    DuplicateTrialId(u64),
    UnknownReservation(u64),
    AccountingInvariant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetStatus {
    WithinBudget,
    RunBudgetExceeded,
    ReservationOverrun,
    CostAccountingUnknown,
}

impl BudgetStatus {
    fn label(self) -> &'static str {
        match self {
            Self::WithinBudget => "within_budget",
            Self::RunBudgetExceeded => "run_budget_exceeded",
            Self::ReservationOverrun => "reservation_overrun",
            Self::CostAccountingUnknown => "cost_accounting_unknown",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AllocationCommitV1 {
    pub trial_id: u64,
    pub choice: AllocationChoice,
    pub reason: AllocationReason,
    pub estimate_lower_bound: Option<f64>,
    pub expected_cost_units: Option<u64>,
    pub reservation_id: Option<u64>,
    pub available_before_units: Option<u64>,
    pub input_digest: EvidenceDigest,
}

/// Actual charge is recorded after execution by the host/evaluator adapter.
/// Unknown charge blocks further escalation. A charge above the reservation is
/// recorded as a reservation overrun and also blocks later escalation.
#[derive(Debug, Clone)]
struct Reservation {
    trial_id: u64,
    expected_units: u64,
}

#[derive(Debug, Clone)]
pub struct BudgetLedger {
    capacity_units: u64,
    charged_units: u64,
    reserved_units: u64,
    next_reservation_id: u64,
    reservations: BTreeMap<u64, Reservation>,
    seen_trials: BTreeSet<u64>,
    run_exceeded: bool,
    reservation_overrun: bool,
    accounting_unknown: bool,
}

impl BudgetLedger {
    pub fn new(capacity_units: u64) -> Self {
        Self {
            capacity_units,
            charged_units: 0,
            reserved_units: 0,
            next_reservation_id: 1,
            reservations: BTreeMap::new(),
            seen_trials: BTreeSet::new(),
            run_exceeded: false,
            reservation_overrun: false,
            accounting_unknown: false,
        }
    }

    pub fn capacity_units(&self) -> u64 {
        self.capacity_units
    }

    pub fn charged_units(&self) -> u64 {
        self.charged_units
    }

    pub fn reserved_units(&self) -> u64 {
        self.reserved_units
    }

    pub fn status(&self) -> BudgetStatus {
        if self.accounting_unknown {
            BudgetStatus::CostAccountingUnknown
        } else if self.run_exceeded {
            BudgetStatus::RunBudgetExceeded
        } else if self.reservation_overrun {
            BudgetStatus::ReservationOverrun
        } else {
            BudgetStatus::WithinBudget
        }
    }

    /// None means there is no trustworthy remaining budget. Reserved work counts
    /// against the cap before the expensive operation is allowed to begin.
    pub fn available_units(&self) -> Option<u64> {
        if self.status() != BudgetStatus::WithinBudget {
            return None;
        }
        self.capacity_units
            .checked_sub(self.charged_units)?
            .checked_sub(self.reserved_units)
    }

    fn mark_trial_seen(&mut self, trial_id: u64) -> Result<(), AllocationError> {
        if trial_id == 0 {
            return Err(AllocationError::InvalidTrialId);
        }
        if !self.seen_trials.insert(trial_id) {
            return Err(AllocationError::DuplicateTrialId(trial_id));
        }
        Ok(())
    }

    fn reserve(&mut self, trial_id: u64, expected_units: u64) -> Option<(u64, u64)> {
        if expected_units == 0 || self.status() != BudgetStatus::WithinBudget {
            return None;
        }
        let available_before = self.available_units()?;
        if expected_units > available_before {
            return None;
        }
        let new_reserved = self.reserved_units.checked_add(expected_units)?;
        let reservation_id = self.next_reservation_id;
        self.next_reservation_id = self.next_reservation_id.checked_add(1)?;
        self.reserved_units = new_reserved;
        self.reservations.insert(
            reservation_id,
            Reservation {
                trial_id,
                expected_units,
            },
        );
        Some((reservation_id, available_before))
    }

    /// Settle a prior escalation with host-accounted actual cost.
    /// This cannot authorize a decision retroactively: the allocation was already
    /// committed and budget-reserved before work began.
    pub fn settle(
        &mut self,
        reservation_id: u64,
        actual_units: Option<u64>,
    ) -> Result<BudgetStatus, AllocationError> {
        let reservation = self
            .reservations
            .remove(&reservation_id)
            .ok_or(AllocationError::UnknownReservation(reservation_id))?;
        if self.reserved_units < reservation.expected_units {
            return Err(AllocationError::AccountingInvariant);
        }
        self.reserved_units -= reservation.expected_units;

        match actual_units {
            None => {
                self.accounting_unknown = true;
            }
            Some(actual) => {
                self.charged_units = match self.charged_units.checked_add(actual) {
                    Some(total) => total,
                    None => {
                        self.run_exceeded = true;
                        u64::MAX
                    }
                };
                if actual > reservation.expected_units {
                    self.reservation_overrun = true;
                }
                if self.charged_units > self.capacity_units {
                    self.run_exceeded = true;
                }
            }
        }

        let _settled_trial_id = reservation.trial_id;
        Ok(self.status())
    }
}

#[derive(Debug, Clone)]
pub struct DecisionValueAllocatorV1 {
    loss_profile_id: String,
    minimum_net_benefit: f64,
    budget: BudgetLedger,
}

impl DecisionValueAllocatorV1 {
    pub fn new(
        loss_profile_id: impl Into<String>,
        minimum_net_benefit: f64,
        run_budget_units: u64,
    ) -> Result<Self, AllocationError> {
        let loss_profile_id = loss_profile_id.into();
        if loss_profile_id.trim().is_empty()
            || !minimum_net_benefit.is_finite()
            || minimum_net_benefit < 0.0
            || run_budget_units == 0
        {
            return Err(AllocationError::InvalidConfig);
        }
        Ok(Self {
            loss_profile_id,
            minimum_net_benefit,
            budget: BudgetLedger::new(run_budget_units),
        })
    }

    pub fn budget(&self) -> &BudgetLedger {
        &self.budget
    }

    /// Decide using only the current causal prefix and a pre-escalation estimate.
    /// No evaluator outcome, alternate-mode output, or future-frame type is accepted.
    pub fn decide(
        &mut self,
        input: PreEscalationInputV1,
    ) -> Result<AllocationCommitV1, AllocationError> {
        if input.loss_profile_id.trim().is_empty() {
            return Err(AllocationError::EmptyLossProfile(input.trial_id));
        }
        if input.loss_profile_id != self.loss_profile_id {
            return Err(AllocationError::MismatchedLossProfile(input.trial_id));
        }
        if input.input_digest.iter().all(|byte| *byte == 0) {
            return Err(AllocationError::ZeroInputDigest(input.trial_id));
        }
        self.budget.mark_trial_seen(input.trial_id)?;

        let estimate_result = match input.value_estimate.as_ref() {
            None => Err(EstimateFailure::Missing),
            Some(estimate) => validate_estimate(
                estimate,
                &self.loss_profile_id,
                &input.input_digest,
            ),
        };

        if let Ok(lower_bound) = estimate_result {
            if lower_bound > self.minimum_net_benefit {
                let Some(expected_cost) = input.escalation_cost_upper_bound_units else {
                    return Ok(self.fallback_commit(
                        input,
                        Some(lower_bound),
                        AllocationReason::EscalationCostUnknown,
                        None,
                    ));
                };
                if expected_cost == 0 {
                    return Ok(self.fallback_commit(
                        input,
                        Some(lower_bound),
                        AllocationReason::EscalationCostUnknown,
                        None,
                    ));
                }
                if let Some((reservation_id, available_before)) =
                    self.budget.reserve(input.trial_id, expected_cost)
                {
                    return Ok(AllocationCommitV1 {
                        trial_id: input.trial_id,
                        choice: AllocationChoice::Escalate,
                        reason: AllocationReason::PositiveConservativeValue,
                        estimate_lower_bound: Some(lower_bound),
                        expected_cost_units: Some(expected_cost),
                        reservation_id: Some(reservation_id),
                        available_before_units: Some(available_before),
                        input_digest: input.input_digest,
                    });
                }
                return Ok(self.fallback_commit(
                    input,
                    Some(lower_bound),
                    AllocationReason::BudgetUnavailable,
                    Some(expected_cost),
                ));
            }

            let expected_cost_units = input.escalation_cost_upper_bound_units;
            return Ok(self.fallback_commit(
                input,
                Some(lower_bound),
                AllocationReason::ValueBelowThreshold,
                expected_cost_units,
            ));
        }

        let reason = match estimate_result {
            Err(EstimateFailure::Missing) => AllocationReason::ValueEstimateMissing,
            Err(EstimateFailure::Invalid) => AllocationReason::ValueEstimateInvalid,
            Ok(_) => unreachable!(),
        };
        let expected_cost_units = input.escalation_cost_upper_bound_units;
        Ok(self.fallback_commit(
            input,
            None,
            reason,
            expected_cost_units,
        ))
    }

    fn fallback_commit(
        &self,
        input: PreEscalationInputV1,
        estimate_lower_bound: Option<f64>,
        reason: AllocationReason,
        expected_cost_units: Option<u64>,
    ) -> AllocationCommitV1 {
        let choice = if input.cheap_path_safe {
            AllocationChoice::KeepCheap
        } else {
            AllocationChoice::Abstain
        };
        AllocationCommitV1 {
            trial_id: input.trial_id,
            choice,
            reason: if choice == AllocationChoice::Abstain {
                AllocationReason::CheapPathNotViable
            } else {
                reason
            },
            estimate_lower_bound,
            expected_cost_units,
            reservation_id: None,
            available_before_units: self.budget.available_units(),
            input_digest: input.input_digest,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EstimateFailure {
    Missing,
    Invalid,
}

fn validate_estimate(
    estimate: &PreEscalationValueEstimateV1,
    expected_loss_profile: &str,
    expected_input_digest: &EvidenceDigest,
) -> Result<f64, EstimateFailure> {
    if estimate.estimator_id.trim().is_empty()
        || estimate.estimator_artifact_digest.iter().all(|byte| *byte == 0)
        || estimate.loss_profile_id != expected_loss_profile
        || estimate.source_input_digest != *expected_input_digest
        || !estimate.net_benefit_lower_bound.is_finite()
    {
        return Err(EstimateFailure::Invalid);
    }
    Ok(estimate.net_benefit_lower_bound)
}

fn digest(byte: u8) -> EvidenceDigest {
    [byte; 32]
}

fn estimate(trial: u64, input_byte: u8, benefit: f64) -> Option<PreEscalationValueEstimateV1> {
    Some(PreEscalationValueEstimateV1 {
        estimator_id: "fixture-pre-escalation-value-v1".to_owned(),
        estimator_artifact_digest: digest(91),
        loss_profile_id: "visual-object-persistence-v1".to_owned(),
        source_input_digest: digest(input_byte),
        net_benefit_lower_bound: benefit,
    })
}

fn input(
    trial_id: u64,
    input_byte: u8,
    lower_bound: Option<f64>,
    cost: Option<u64>,
    cheap_path_safe: bool,
) -> PreEscalationInputV1 {
    PreEscalationInputV1 {
        trial_id,
        loss_profile_id: "visual-object-persistence-v1".to_owned(),
        input_digest: digest(input_byte),
        cheap_path_safe,
        value_estimate: lower_bound.and_then(|value| estimate(trial_id, input_byte, value)),
        escalation_cost_upper_bound_units: cost,
    }
}

fn render_report_row(
    commit: &AllocationCommitV1,
    actual_cost: Option<u64>,
    budget_status: BudgetStatus,
) -> String {
    let lower_bound = commit
        .estimate_lower_bound
        .map(|value| format!("{value:.6}"))
        .unwrap_or_else(|| "UNKNOWN".to_owned());
    let expected_cost = commit
        .expected_cost_units
        .map(|value| value.to_string())
        .unwrap_or_else(|| "UNKNOWN".to_owned());
    let actual = actual_cost
        .map(|value| value.to_string())
        .unwrap_or_else(|| "UNKNOWN".to_owned());
    format!(
        "ROW\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        commit.trial_id,
        commit.choice.label(),
        commit.reason.label(),
        lower_bound,
        expected_cost,
        actual,
        budget_status.label()
    )
}

/// Deterministic mechanism fixture only. The scenarios are hand-authored and do
/// not estimate performance on real visual tasks.
fn run_reference_scenario() -> Result<String, AllocationError> {
    let mut allocator =
        DecisionValueAllocatorV1::new("visual-object-persistence-v1", 0.10, 16)?;
    let mut rows = Vec::new();

    let cases = [
        input(1, 1, Some(0.80), Some(5), true),
        input(2, 2, Some(-0.20), Some(5), true),
        input(3, 3, None, Some(2), true),
        input(4, 4, Some(0.70), Some(13), false),
        input(5, 5, Some(0.60), Some(12), true),
        input(6, 6, Some(0.90), Some(1), false),
    ];

    for case in cases {
        let commit = allocator.decide(case)?;
        let actual_cost = if let Some(reservation_id) = commit.reservation_id {
            let actual = if commit.trial_id == 1 { 4 } else { 13 };
            allocator.budget.settle(reservation_id, Some(actual))?;
            Some(actual)
        } else {
            None
        };
        rows.push(render_report_row(&commit, actual_cost, allocator.budget.status()));
    }

    let mut output = String::new();
    output.push_str("META\tformat\tVIS-DVPC-ONLINE-001-v0.1\n");
    output.push_str("META\tpolicy\tpre-escalation-value-lower-bound-plus-hard-budget-reservation\n");
    output.push_str("META\ttruth_visibility\tproducer-input-excludes-counterfactual-losses-future-frames-and-evaluator-truth\n");
    output.push_str("META\tresource_units\tabstract-declared-units-not-measured-energy\n");
    output.push_str("ROW_HEADER\ttrial_id\tchoice\treason\testimate_lower_bound\texpected_cost_units\tactual_cost_units\tbudget_status\n");
    for row in rows {
        output.push_str(&row);
        output.push('\n');
    }
    Ok(output)
}

fn main() {
    match run_reference_scenario() {
        Ok(report) => print!("{report}"),
        Err(error) => {
            eprintln!("VIS-DVPC-ONLINE-001 failed closed: {error:?}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allocator(budget: u64) -> DecisionValueAllocatorV1 {
        DecisionValueAllocatorV1::new("visual-object-persistence-v1", 0.10, budget).unwrap()
    }

    #[test]
    fn positive_value_reserves_budget_before_escalation() {
        let mut policy = allocator(10);
        let commit = policy.decide(input(1, 1, Some(0.20), Some(4), true)).unwrap();
        assert_eq!(commit.choice, AllocationChoice::Escalate);
        assert_eq!(commit.reason, AllocationReason::PositiveConservativeValue);
        assert_eq!(commit.available_before_units, Some(10));
        assert_eq!(commit.reservation_id, Some(1));
        assert_eq!(policy.budget().reserved_units(), 4);
        assert_eq!(policy.budget().available_units(), Some(6));
    }

    #[test]
    fn negative_or_neutral_value_stays_cheap() {
        let mut policy = allocator(10);
        for (id, value) in [(1, -0.30), (2, 0.10)] {
            let commit = policy.decide(input(id, id as u8, Some(value), Some(3), true)).unwrap();
            assert_eq!(commit.choice, AllocationChoice::KeepCheap);
            assert_eq!(commit.reason, AllocationReason::ValueBelowThreshold);
        }
        assert_eq!(policy.budget().reserved_units(), 0);
    }

    #[test]
    fn missing_estimate_fails_closed_to_cheap_or_abstain() {
        let mut policy = allocator(10);
        let safe = policy.decide(input(1, 1, None, Some(3), true)).unwrap();
        let unsafe_cheap = policy.decide(input(2, 2, None, Some(3), false)).unwrap();
        assert_eq!(safe.choice, AllocationChoice::KeepCheap);
        assert_eq!(safe.reason, AllocationReason::ValueEstimateMissing);
        assert_eq!(unsafe_cheap.choice, AllocationChoice::Abstain);
        assert_eq!(unsafe_cheap.reason, AllocationReason::CheapPathNotViable);
    }

    #[test]
    fn input_digest_mismatch_invalidates_estimate() {
        let mut policy = allocator(10);
        let mut item = input(1, 1, Some(0.9), Some(3), true);
        item.value_estimate.as_mut().unwrap().source_input_digest = digest(2);
        let commit = policy.decide(item).unwrap();
        assert_eq!(commit.choice, AllocationChoice::KeepCheap);
        assert_eq!(commit.reason, AllocationReason::ValueEstimateInvalid);
        assert_eq!(policy.budget().reserved_units(), 0);
    }

    #[test]
    fn wrong_estimator_artifact_or_non_finite_value_invalidates_estimate() {
        let mut policy = allocator(10);
        let mut artifact = input(1, 1, Some(0.9), Some(3), true);
        artifact.value_estimate.as_mut().unwrap().estimator_artifact_digest = digest(0);
        assert_eq!(
            policy.decide(artifact).unwrap().reason,
            AllocationReason::ValueEstimateInvalid
        );

        let mut nan = input(2, 2, Some(0.9), Some(3), true);
        nan.value_estimate.as_mut().unwrap().net_benefit_lower_bound = f64::NAN;
        assert_eq!(
            policy.decide(nan).unwrap().reason,
            AllocationReason::ValueEstimateInvalid
        );
    }

    #[test]
    fn escalation_requires_known_nonzero_cost() {
        let mut policy = allocator(10);
        let unknown = policy.decide(input(1, 1, Some(0.9), None, true)).unwrap();
        let zero = policy.decide(input(2, 2, Some(0.9), Some(0), true)).unwrap();
        assert_eq!(unknown.choice, AllocationChoice::KeepCheap);
        assert_eq!(unknown.reason, AllocationReason::EscalationCostUnknown);
        assert_eq!(zero.choice, AllocationChoice::KeepCheap);
        assert_eq!(zero.reason, AllocationReason::EscalationCostUnknown);
    }

    #[test]
    fn no_escalation_when_reservation_would_exceed_budget() {
        let mut policy = allocator(5);
        let commit = policy.decide(input(1, 1, Some(0.9), Some(6), false)).unwrap();
        assert_eq!(commit.choice, AllocationChoice::Abstain);
        assert_eq!(commit.reason, AllocationReason::CheapPathNotViable);
        assert_eq!(policy.budget().reserved_units(), 0);
    }

    #[test]
    fn actual_cost_overrun_blocks_future_escalation() {
        let mut policy = allocator(20);
        let first = policy.decide(input(1, 1, Some(0.9), Some(3), true)).unwrap();
        let reservation = first.reservation_id.unwrap();
        assert_eq!(
            policy.budget.settle(reservation, Some(5)).unwrap(),
            BudgetStatus::ReservationOverrun
        );
        let next = policy.decide(input(2, 2, Some(0.9), Some(1), false)).unwrap();
        assert_eq!(next.choice, AllocationChoice::Abstain);
        assert_eq!(next.reason, AllocationReason::CheapPathNotViable);
        assert_eq!(policy.budget().available_units(), None);
    }

    #[test]
    fn unknown_actual_charge_blocks_future_escalation() {
        let mut policy = allocator(20);
        let first = policy.decide(input(1, 1, Some(0.9), Some(3), true)).unwrap();
        policy.budget.settle(first.reservation_id.unwrap(), None).unwrap();
        assert_eq!(policy.budget().status(), BudgetStatus::CostAccountingUnknown);
        let next = policy.decide(input(2, 2, Some(0.9), Some(1), false)).unwrap();
        assert_eq!(next.choice, AllocationChoice::Abstain);
        assert_eq!(policy.budget().available_units(), None);
    }

    #[test]
    fn run_budget_overrun_blocks_future_escalation() {
        let mut policy = allocator(5);
        let first = policy.decide(input(1, 1, Some(0.9), Some(5), true)).unwrap();
        policy.budget.settle(first.reservation_id.unwrap(), Some(6)).unwrap();
        assert_eq!(policy.budget().status(), BudgetStatus::RunBudgetExceeded);
        let next = policy.decide(input(2, 2, Some(0.9), Some(1), false)).unwrap();
        assert_eq!(next.choice, AllocationChoice::Abstain);
    }

    #[test]
    fn duplicate_trials_and_double_settlement_are_rejected() {
        let mut policy = allocator(10);
        let first = policy.decide(input(1, 1, Some(0.9), Some(3), true)).unwrap();
        assert_eq!(
            policy.decide(input(1, 2, Some(0.9), Some(3), true)),
            Err(AllocationError::DuplicateTrialId(1))
        );
        let id = first.reservation_id.unwrap();
        policy.budget.settle(id, Some(3)).unwrap();
        assert_eq!(
            policy.budget.settle(id, Some(3)),
            Err(AllocationError::UnknownReservation(id))
        );
    }

    #[test]
    fn different_loss_profiles_are_rejected() {
        let mut policy = allocator(10);
        let mut item = input(1, 1, Some(0.9), Some(3), true);
        item.loss_profile_id = "different-profile".to_owned();
        assert_eq!(
            policy.decide(item),
            Err(AllocationError::MismatchedLossProfile(1))
        );
    }

    #[test]
    fn output_is_deterministic_and_exposes_no_counterfactual_field() {
        let first = run_reference_scenario().unwrap();
        let second = run_reference_scenario().unwrap();
        assert_eq!(first, second);
        assert!(first.contains("VIS-DVPC-ONLINE-001-v0.1"));
        assert!(first.contains("reservation_overrun"));
        assert!(!first.contains("cheap_task_loss"));
        assert!(!first.contains("escalated_task_loss"));
        assert!(!first.contains("truth_revision"));
    }
}
