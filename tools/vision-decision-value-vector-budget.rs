// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! VIS-DVPC-007: online decision-value allocation with independent resource budgets.
//!
//! This is a standalone research reference, not a production vision integration.
//! The policy sees only a pre-escalation value estimate, a causal-input digest,
//! declared upper bounds, and resource availability. Evaluator outcomes and
//! escalated outputs are intentionally absent from the decision interface.
//!
//! Run the focused tests without Cargo or third-party dependencies:
//!   rustc --edition=2024 --test tools/vision-decision-value-vector-budget.rs -o /tmp/vis-dvpc-007-tests
//!   /tmp/vis-dvpc-007-tests

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

pub type EvidenceDigest = [u8; 32];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResourceVector {
    pub latency_us: u64,
    pub energy_uj: u64,
}

impl ResourceVector {
    pub const fn new(latency_us: u64, energy_uj: u64) -> Self {
        Self { latency_us, energy_uj }
    }

    fn is_positive(self) -> bool {
        self.latency_us > 0 && self.energy_uj > 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObservedResourceCost {
    /// None means the host could not produce a trustworthy latency measurement.
    pub latency_us: Option<u64>,
    /// None means the host could not produce a trustworthy energy measurement.
    pub energy_uj: Option<u64>,
}

impl ObservedResourceCost {
    pub const fn known(latency_us: u64, energy_uj: u64) -> Self {
        Self { latency_us: Some(latency_us), energy_uj: Some(energy_uj) }
    }
}

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

#[derive(Debug, Clone, PartialEq)]
pub struct PreEscalationValueEstimateV1 {
    pub estimator_id: String,
    pub estimator_artifact_digest: EvidenceDigest,
    pub loss_profile_id: String,
    pub source_input_digest: EvidenceDigest,
    /// Lower bound on downstream benefit in the frozen task-loss profile.
    pub net_benefit_lower_bound: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PreEscalationInputV1 {
    /// Opaque identifier; it must not encode evaluator-only condition labels.
    pub trial_id: u64,
    pub loss_profile_id: String,
    /// Digest of the causal prefix and all permitted pre-escalation features.
    pub input_digest: EvidenceDigest,
    /// Declared by the cheap path/task contract, not evaluator truth.
    pub cheap_path_safe: bool,
    pub value_estimate: Option<PreEscalationValueEstimateV1>,
    /// Per-resource conservative upper bounds; no scalarized proxy is accepted.
    pub escalation_cost_upper_bound: Option<ResourceVector>,
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
    pub fn label(self) -> &'static str {
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
    pub expected_cost: Option<ResourceVector>,
    pub reservation_id: Option<u64>,
    /// Both values are captured before reserving this action.
    pub available_before: Option<ResourceVector>,
    pub input_digest: EvidenceDigest,
}

impl AllocationCommitV1 {
    /// Stable tab-separated receipt. All strings are fixed labels or numeric fields.
    pub fn to_tsv(&self) -> String {
        format!(
            "COMMIT\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            self.trial_id,
            self.choice.label(),
            self.reason.label(),
            self.estimate_lower_bound
                .map(|value| format!("{value:.9}"))
                .unwrap_or_else(|| "NA".to_owned()),
            option_u64(self.expected_cost.map(|cost| cost.latency_us)),
            option_u64(self.expected_cost.map(|cost| cost.energy_uj)),
            self.reservation_id
                .map(|value| value.to_string())
                .unwrap_or_else(|| "NA".to_owned()),
            option_u64(self.available_before.map(|value| value.latency_us)),
            option_u64(self.available_before.map(|value| value.energy_uj)),
            digest_hex(&self.input_digest),
        )
    }
}

#[derive(Debug, Clone)]
struct Reservation {
    trial_id: u64,
    expected: ResourceVector,
}

/// Two-resource ledger: latency and energy are accounted independently, while
/// each reservation is an atomic pair. Any measurement uncertainty or breach
/// blocks *new* escalation until the run is restarted or externally adjudicated.
#[derive(Debug, Clone)]
pub struct VectorBudgetLedgerV1 {
    capacity: ResourceVector,
    charged: ResourceVector,
    reserved: ResourceVector,
    next_reservation_id: u64,
    reservations: BTreeMap<u64, Reservation>,
    seen_trials: BTreeSet<u64>,
    run_exceeded: bool,
    reservation_overrun: bool,
    accounting_unknown: bool,
}

impl VectorBudgetLedgerV1 {
    pub fn new(capacity: ResourceVector) -> Self {
        Self {
            capacity,
            charged: ResourceVector::default(),
            reserved: ResourceVector::default(),
            next_reservation_id: 1,
            reservations: BTreeMap::new(),
            seen_trials: BTreeSet::new(),
            run_exceeded: false,
            reservation_overrun: false,
            accounting_unknown: false,
        }
    }

    pub fn capacity(&self) -> ResourceVector {
        self.capacity
    }

    pub fn charged(&self) -> ResourceVector {
        self.charged
    }

    pub fn reserved(&self) -> ResourceVector {
        self.reserved
    }

    pub fn outstanding_reservations(&self) -> usize {
        self.reservations.len()
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

    /// None denotes a fail-closed state, not an unlimited resource budget.
    pub fn available(&self) -> Option<ResourceVector> {
        if self.status() != BudgetStatus::WithinBudget {
            return None;
        }
        Some(ResourceVector {
            latency_us: self.capacity.latency_us
                .checked_sub(self.charged.latency_us)?
                .checked_sub(self.reserved.latency_us)?,
            energy_uj: self.capacity.energy_uj
                .checked_sub(self.charged.energy_uj)?
                .checked_sub(self.reserved.energy_uj)?,
        })
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

    /// Admit both resource reservations or mutate neither resource ledger.
    fn reserve(
        &mut self,
        trial_id: u64,
        expected: ResourceVector,
    ) -> Option<(u64, ResourceVector)> {
        if !expected.is_positive() || self.status() != BudgetStatus::WithinBudget {
            return None;
        }
        let available_before = self.available()?;
        if expected.latency_us > available_before.latency_us
            || expected.energy_uj > available_before.energy_uj
        {
            return None;
        }

        // Preflight every fallible arithmetic operation before changing state.
        let new_reserved = ResourceVector {
            latency_us: self.reserved.latency_us.checked_add(expected.latency_us)?,
            energy_uj: self.reserved.energy_uj.checked_add(expected.energy_uj)?,
        };
        let next_reservation_id = self.next_reservation_id.checked_add(1)?;
        let reservation_id = self.next_reservation_id;

        self.reserved = new_reserved;
        self.next_reservation_id = next_reservation_id;
        self.reservations.insert(reservation_id, Reservation { trial_id, expected });
        Some((reservation_id, available_before))
    }

    /// Settle actual host-measured costs. Missing either measurement frees the
    /// reservation but marks the whole run untrusted and blocks future escalation.
    pub fn settle(
        &mut self,
        reservation_id: u64,
        actual: ObservedResourceCost,
    ) -> Result<BudgetStatus, AllocationError> {
        let reservation = self
            .reservations
            .get(&reservation_id)
            .cloned()
            .ok_or(AllocationError::UnknownReservation(reservation_id))?;

        if self.reserved.latency_us < reservation.expected.latency_us
            || self.reserved.energy_uj < reservation.expected.energy_uj
        {
            return Err(AllocationError::AccountingInvariant);
        }

        let (new_latency_charge, latency_overflow) =
            add_charge(self.charged.latency_us, actual.latency_us);
        let (new_energy_charge, energy_overflow) =
            add_charge(self.charged.energy_uj, actual.energy_uj);

        // Mutations begin only after the reservation and accounting invariants pass.
        self.reservations.remove(&reservation_id);
        self.reserved.latency_us -= reservation.expected.latency_us;
        self.reserved.energy_uj -= reservation.expected.energy_uj;
        self.charged.latency_us = new_latency_charge;
        self.charged.energy_uj = new_energy_charge;

        if actual.latency_us.is_none() || actual.energy_uj.is_none() {
            self.accounting_unknown = true;
        }
        if latency_overflow || energy_overflow {
            self.run_exceeded = true;
        }
        if actual.latency_us.is_some_and(|cost| cost > reservation.expected.latency_us)
            || actual.energy_uj.is_some_and(|cost| cost > reservation.expected.energy_uj)
        {
            self.reservation_overrun = true;
        }
        if self.charged.latency_us > self.capacity.latency_us
            || self.charged.energy_uj > self.capacity.energy_uj
        {
            self.run_exceeded = true;
        }

        let _settled_trial_id = reservation.trial_id;
        Ok(self.status())
    }
}

#[derive(Debug, Clone)]
pub struct VectorBudgetAllocatorV1 {
    loss_profile_id: String,
    minimum_net_benefit: f64,
    budget: VectorBudgetLedgerV1,
}

impl VectorBudgetAllocatorV1 {
    pub fn new(
        loss_profile_id: impl Into<String>,
        minimum_net_benefit: f64,
        capacity: ResourceVector,
    ) -> Result<Self, AllocationError> {
        let loss_profile_id = loss_profile_id.into();
        if loss_profile_id.trim().is_empty()
            || !minimum_net_benefit.is_finite()
            || minimum_net_benefit < 0.0
        {
            return Err(AllocationError::InvalidConfig);
        }
        Ok(Self {
            loss_profile_id,
            minimum_net_benefit,
            budget: VectorBudgetLedgerV1::new(capacity),
        })
    }

    pub fn budget(&self) -> &VectorBudgetLedgerV1 {
        &self.budget
    }

    pub fn settle(
        &mut self,
        reservation_id: u64,
        actual: ObservedResourceCost,
    ) -> Result<BudgetStatus, AllocationError> {
        self.budget.settle(reservation_id, actual)
    }

    /// Decide from pre-escalation evidence only. No ground-truth labels or expensive
    /// mode outputs enter this interface.
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
            Some(value) => validate_estimate(value, &self.loss_profile_id, &input.input_digest),
        };

        if let Ok(lower_bound) = estimate_result {
            if lower_bound > self.minimum_net_benefit {
                let Some(expected_cost) = input.escalation_cost_upper_bound else {
                    return Ok(self.fallback_commit(
                        input,
                        Some(lower_bound),
                        AllocationReason::EscalationCostUnknown,
                        None,
                    ));
                };
                if !expected_cost.is_positive() {
                    return Ok(self.fallback_commit(
                        input,
                        Some(lower_bound),
                        AllocationReason::EscalationCostUnknown,
                        Some(expected_cost),
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
                        expected_cost: Some(expected_cost),
                        reservation_id: Some(reservation_id),
                        available_before: Some(available_before),
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

            let expected_cost = input.escalation_cost_upper_bound;
            return Ok(self.fallback_commit(
                input,
                Some(lower_bound),
                AllocationReason::ValueBelowThreshold,
                expected_cost,
            ));
        }

        let reason = match estimate_result {
            Err(EstimateFailure::Missing) => AllocationReason::ValueEstimateMissing,
            Err(EstimateFailure::Invalid) => AllocationReason::ValueEstimateInvalid,
            Ok(_) => unreachable!(),
        };
        let expected_cost = input.escalation_cost_upper_bound;
        Ok(self.fallback_commit(input, None, reason, expected_cost))
    }

    fn fallback_commit(
        &self,
        input: PreEscalationInputV1,
        estimate_lower_bound: Option<f64>,
        reason: AllocationReason,
        expected_cost: Option<ResourceVector>,
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
            expected_cost,
            reservation_id: None,
            available_before: self.budget.available(),
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

fn add_charge(current: u64, actual: Option<u64>) -> (u64, bool) {
    match actual {
        None => (current, false),
        Some(amount) => match current.checked_add(amount) {
            Some(total) => (total, false),
            None => (u64::MAX, true),
        },
    }
}

fn digest_hex(digest: &EvidenceDigest) -> String {
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push_str(&format!("{byte:02x}"));
    }
    output
}

fn option_u64(value: Option<u64>) -> String {
    value.map(|number| number.to_string()).unwrap_or_else(|| "NA".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOSS_PROFILE: &str = "collision-weighted-task-loss-v1";

    fn digest(byte: u8) -> EvidenceDigest {
        [byte; 32]
    }

    fn candidate(
        trial_id: u64,
        cost: Option<ResourceVector>,
        benefit: f64,
        cheap_path_safe: bool,
    ) -> PreEscalationInputV1 {
        let input_digest = digest((trial_id as u8).wrapping_add(1));
        let value_estimate = Some(PreEscalationValueEstimateV1 {
            estimator_id: "fixture-value-router-v1".to_owned(),
            estimator_artifact_digest: digest(0xA5),
            loss_profile_id: LOSS_PROFILE.to_owned(),
            source_input_digest: input_digest,
            net_benefit_lower_bound: benefit,
        });
        PreEscalationInputV1 {
            trial_id,
            loss_profile_id: LOSS_PROFILE.to_owned(),
            input_digest,
            cheap_path_safe,
            value_estimate,
            escalation_cost_upper_bound: cost,
        }
    }

    fn allocator(latency_us: u64, energy_uj: u64) -> VectorBudgetAllocatorV1 {
        VectorBudgetAllocatorV1::new(
            LOSS_PROFILE,
            0.01,
            ResourceVector::new(latency_us, energy_uj),
        )
        .unwrap()
    }

    #[test]
    fn latency_limit_rejects_atomically_without_reserving_energy() {
        let mut policy = allocator(10, 100);
        let commit = policy
            .decide(candidate(1, Some(ResourceVector::new(11, 50)), 0.2, true))
            .unwrap();
        assert_eq!(commit.choice, AllocationChoice::KeepCheap);
        assert_eq!(commit.reason, AllocationReason::BudgetUnavailable);
        assert_eq!(policy.budget().reserved(), ResourceVector::default());
        assert_eq!(policy.budget().available(), Some(ResourceVector::new(10, 100)));
    }

    #[test]
    fn energy_limit_rejects_atomically_without_reserving_latency() {
        let mut policy = allocator(100, 10);
        let commit = policy
            .decide(candidate(1, Some(ResourceVector::new(50, 11)), 0.2, true))
            .unwrap();
        assert_eq!(commit.choice, AllocationChoice::KeepCheap);
        assert_eq!(commit.reason, AllocationReason::BudgetUnavailable);
        assert_eq!(policy.budget().reserved(), ResourceVector::default());
        assert_eq!(policy.budget().available(), Some(ResourceVector::new(100, 10)));
    }

    #[test]
    fn outstanding_reservations_count_against_both_caps_before_execution() {
        let mut policy = allocator(100, 100);
        let first = policy
            .decide(candidate(1, Some(ResourceVector::new(60, 60)), 0.2, true))
            .unwrap();
        assert_eq!(first.choice, AllocationChoice::Escalate);
        let second = policy
            .decide(candidate(2, Some(ResourceVector::new(50, 50)), 0.2, true))
            .unwrap();
        assert_eq!(second.choice, AllocationChoice::KeepCheap);
        assert_eq!(second.reason, AllocationReason::BudgetUnavailable);
        assert_eq!(policy.budget().reserved(), ResourceVector::new(60, 60));
        assert_eq!(policy.budget().available(), Some(ResourceVector::new(40, 40)));

        let status = policy
            .settle(first.reservation_id.unwrap(), ObservedResourceCost::known(40, 30))
            .unwrap();
        assert_eq!(status, BudgetStatus::WithinBudget);
        assert_eq!(policy.budget().charged(), ResourceVector::new(40, 30));
        assert_eq!(policy.budget().reserved(), ResourceVector::default());
        assert_eq!(policy.budget().available(), Some(ResourceVector::new(60, 70)));
    }

    #[test]
    fn unknown_single_resource_frees_pair_and_blocks_future_escalation() {
        let mut policy = allocator(100, 100);
        let first = policy
            .decide(candidate(1, Some(ResourceVector::new(40, 30)), 0.2, true))
            .unwrap();
        let status = policy
            .settle(
                first.reservation_id.unwrap(),
                ObservedResourceCost { latency_us: Some(25), energy_uj: None },
            )
            .unwrap();
        assert_eq!(status, BudgetStatus::CostAccountingUnknown);
        assert_eq!(policy.budget().reserved(), ResourceVector::default());
        assert_eq!(policy.budget().charged(), ResourceVector::new(25, 0));
        assert_eq!(policy.budget().available(), None);

        let second = policy
            .decide(candidate(2, Some(ResourceVector::new(10, 10)), 0.9, true))
            .unwrap();
        assert_eq!(second.choice, AllocationChoice::KeepCheap);
        assert_eq!(second.reason, AllocationReason::BudgetUnavailable);
    }

    #[test]
    fn latency_reservation_overrun_blocks_all_future_escalations() {
        let mut policy = allocator(100, 100);
        let first = policy
            .decide(candidate(1, Some(ResourceVector::new(10, 20)), 0.2, true))
            .unwrap();
        let status = policy
            .settle(first.reservation_id.unwrap(), ObservedResourceCost::known(11, 10))
            .unwrap();
        assert_eq!(status, BudgetStatus::ReservationOverrun);
        assert_eq!(policy.budget().charged(), ResourceVector::new(11, 10));
        assert_eq!(policy.budget().available(), None);
    }

    #[test]
    fn run_budget_overrun_is_distinct_from_per_action_overrun() {
        let mut policy = allocator(100, 100);
        let first = policy
            .decide(candidate(1, Some(ResourceVector::new(10, 10)), 0.2, true))
            .unwrap();
        let status = policy
            .settle(first.reservation_id.unwrap(), ObservedResourceCost::known(101, 10))
            .unwrap();
        assert_eq!(status, BudgetStatus::RunBudgetExceeded);
        assert_eq!(policy.budget().charged(), ResourceVector::new(101, 10));
    }

    #[test]
    fn unknown_and_duplicate_settlement_are_rejected_without_mutation() {
        let mut policy = allocator(100, 100);
        assert_eq!(
            policy.settle(99, ObservedResourceCost::known(1, 1)),
            Err(AllocationError::UnknownReservation(99))
        );
        assert_eq!(policy.budget().reserved(), ResourceVector::default());

        let first = policy
            .decide(candidate(1, Some(ResourceVector::new(10, 10)), 0.2, true))
            .unwrap();
        let id = first.reservation_id.unwrap();
        policy.settle(id, ObservedResourceCost::known(5, 5)).unwrap();
        assert_eq!(
            policy.settle(id, ObservedResourceCost::known(5, 5)),
            Err(AllocationError::UnknownReservation(id))
        );
        assert_eq!(policy.budget().charged(), ResourceVector::new(5, 5));
    }

    #[test]
    fn charge_overflow_saturates_and_reports_overflow() {
        assert_eq!(add_charge(u64::MAX, Some(1)), (u64::MAX, true));
        assert_eq!(add_charge(u64::MAX, None), (u64::MAX, false));
    }

    #[test]
    fn unsafe_cheap_path_abstains_when_estimate_is_missing() {
        let mut policy = allocator(100, 100);
        let mut input = candidate(1, Some(ResourceVector::new(10, 10)), 0.2, false);
        input.value_estimate = None;
        let commit = policy.decide(input).unwrap();
        assert_eq!(commit.choice, AllocationChoice::Abstain);
        assert_eq!(commit.reason, AllocationReason::CheapPathNotViable);
        assert_eq!(policy.budget().reserved(), ResourceVector::default());
    }

    #[test]
    fn cost_estimates_must_be_positive_in_both_dimensions() {
        let mut policy = allocator(100, 100);
        let commit = policy
            .decide(candidate(1, Some(ResourceVector::new(10, 0)), 0.2, true))
            .unwrap();
        assert_eq!(commit.choice, AllocationChoice::KeepCheap);
        assert_eq!(commit.reason, AllocationReason::EscalationCostUnknown);
    }

    #[test]
    fn duplicate_trials_and_zero_trial_ids_reject() {
        let mut policy = allocator(100, 100);
        assert_eq!(
            policy.decide(candidate(0, Some(ResourceVector::new(10, 10)), 0.2, true)),
            Err(AllocationError::InvalidTrialId)
        );
        policy
            .decide(candidate(1, Some(ResourceVector::new(10, 10)), 0.2, true))
            .unwrap();
        assert_eq!(
            policy.decide(candidate(1, Some(ResourceVector::new(10, 10)), 0.2, true)),
            Err(AllocationError::DuplicateTrialId(1))
        );
    }

    #[test]
    fn commitment_receipt_is_deterministic_and_binds_input_digest() {
        let mut policy = allocator(100, 100);
        let commit = policy
            .decide(candidate(7, Some(ResourceVector::new(10, 20)), 0.2, true))
            .unwrap();
        let first = commit.to_tsv();
        let second = commit.to_tsv();
        assert_eq!(first, second);
        let fields: Vec<&str> = first.trim_end().split('\t').collect();
        assert_eq!(fields.len(), 11);
        assert_eq!(
            &fields[..10],
            [
                "COMMIT", "7", "escalate", "positive_conservative_value", "0.200000000",
                "10", "20", "1", "100", "100",
            ]
        );
        assert_eq!(fields[10], "08".repeat(32).as_str());
    }

    #[test]
    fn invalid_or_mismatched_estimates_fail_closed() {
        let mut policy = allocator(100, 100);
        let mut input = candidate(1, Some(ResourceVector::new(10, 10)), 0.2, true);
        input.value_estimate.as_mut().unwrap().source_input_digest = digest(0xFF);
        let commit = policy.decide(input).unwrap();
        assert_eq!(commit.choice, AllocationChoice::KeepCheap);
        assert_eq!(commit.reason, AllocationReason::ValueEstimateInvalid);
        assert_eq!(policy.budget().reserved(), ResourceVector::default());
    }

    #[test]
    fn zero_estimate_does_not_escalate_and_does_not_reserve() {
        let mut policy = allocator(100, 100);
        let commit = policy
            .decide(candidate(1, Some(ResourceVector::new(10, 10)), 0.01, true))
            .unwrap();
        assert_eq!(commit.choice, AllocationChoice::KeepCheap);
        assert_eq!(commit.reason, AllocationReason::ValueBelowThreshold);
        assert_eq!(policy.budget().reserved(), ResourceVector::default());
    }
}
