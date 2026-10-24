// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! VIS-DVPC-015: all-in online latency/energy reservation ledger.
//!
//! This reference ledger reserves allocator overhead before a route decision,
//! then separately reserves escalation resources if requested. It does not
//! estimate values or invoke a vision model; compose it around VIS-DVPC-007.
//!
//! Run:
//!   rustc --edition=2024 --test tools/vision-dvpc-all-in-budget.rs -o /tmp/vis-dvpc-015-tests
//!   /tmp/vis-dvpc-015-tests
//!
//! The configured bounds must be calibrated against target hardware. Passing
//! tests prove accounting mechanics only, not bound validity or SOTA.

#![deny(unsafe_code)]

use std::collections::BTreeMap;
use std::fmt::Write as FmtWrite;

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

    fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            latency_us: self.latency_us.checked_add(other.latency_us)?,
            energy_uj: self.energy_uj.checked_add(other.energy_uj)?,
        })
    }

    fn checked_sub(self, other: Self) -> Option<Self> {
        Some(Self {
            latency_us: self.latency_us.checked_sub(other.latency_us)?,
            energy_uj: self.energy_uj.checked_sub(other.energy_uj)?,
        })
    }

    fn componentwise_le(self, other: Self) -> bool {
        self.latency_us <= other.latency_us && self.energy_uj <= other.energy_uj
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LedgerConfig {
    pub run_budget: ResourceVector,
    pub allocator_upper_bound: ResourceVector,
    pub escalation_upper_bound: ResourceVector,
}

impl LedgerConfig {
    pub fn validate(self) -> Result<Self, String> {
        if !self.run_budget.is_positive()
            || !self.allocator_upper_bound.is_positive()
            || !self.escalation_upper_bound.is_positive()
        {
            return Err("run budget and both resource upper bounds must be positive in latency and energy".to_owned());
        }
        if !self.allocator_upper_bound.componentwise_le(self.run_budget) {
            return Err("allocator upper bound exceeds the all-in run budget".to_owned());
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunState {
    Healthy,
    Contaminated,
}

impl RunState {
    fn label(self) -> &'static str {
        match self {
            Self::Healthy => "HEALTHY",
            Self::Contaminated => "CONTAMINATED",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActionState {
    NotRequested,
    Reserved,
    RejectedBudget,
}

impl ActionState {
    fn label(self) -> &'static str {
        match self {
            Self::NotRequested => "NOT_REQUESTED",
            Self::Reserved => "RESERVED",
            Self::RejectedBudget => "REJECTED_BUDGET",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecisionState {
    AllocatorReserved,
    SettledCheap,
    SettledEscalated,
    RejectedAllocatorBudget,
    RejectedRunContaminated,
    Contaminated,
}

impl DecisionState {
    fn label(self) -> &'static str {
        match self {
            Self::AllocatorReserved => "ALLOCATOR_RESERVED",
            Self::SettledCheap => "SETTLED_CHEAP",
            Self::SettledEscalated => "SETTLED_ESCALATED",
            Self::RejectedAllocatorBudget => "REJECTED_ALLOCATOR_BUDGET",
            Self::RejectedRunContaminated => "REJECTED_RUN_CONTAMINATED",
            Self::Contaminated => "CONTAMINATED",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DecisionRecord {
    action_state: ActionState,
    decision_state: DecisionState,
    allocator_actual: Option<ResourceVector>,
    escalation_actual: Option<ResourceVector>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LedgerSnapshot {
    pub state: RunState,
    pub spent: ResourceVector,
    pub reserved: ResourceVector,
    pub remaining: ResourceVector,
    pub decisions: usize,
    pub settled_cheap: usize,
    pub settled_escalated: usize,
    pub rejected_allocator_budget: usize,
    pub rejected_run_contaminated: usize,
    pub rejected_escalation_budget: usize,
    pub contaminated_decisions: usize,
}

#[derive(Debug, Clone)]
pub struct AllInBudgetLedger {
    config: LedgerConfig,
    state: RunState,
    spent: ResourceVector,
    reserved: ResourceVector,
    decisions: BTreeMap<String, DecisionRecord>,
}

impl AllInBudgetLedger {
    pub fn new(config: LedgerConfig) -> Result<Self, String> {
        Ok(Self {
            config: config.validate()?,
            state: RunState::Healthy,
            spent: ResourceVector::default(),
            reserved: ResourceVector::default(),
            decisions: BTreeMap::new(),
        })
    }

    fn validate_id(id: &str) -> Result<(), String> {
        if id.is_empty()
            || !id.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':')
            })
        {
            return Err("decision_id must be a non-empty opaque token".to_owned());
        }
        Ok(())
    }

    fn fits(&self, extra: ResourceVector) -> bool {
        self.spent
            .checked_add(self.reserved)
            .and_then(|used| used.checked_add(extra))
            .is_some_and(|needed| needed.componentwise_le(self.config.run_budget))
    }

    fn mark_contaminated(&mut self, decision_id: &str) {
        self.state = RunState::Contaminated;
        if let Some(record) = self.decisions.get_mut(decision_id) {
            record.decision_state = DecisionState::Contaminated;
        }
    }

    /// Reserve the allocator's maximum latency+energy before running it.
    /// A budget rejection is retained as a terminal receipt for this ID.
    pub fn reserve_allocator(&mut self, decision_id: &str) -> Result<bool, String> {
        Self::validate_id(decision_id)?;
        if self.decisions.contains_key(decision_id) {
            return Err(format!("duplicate decision_id {decision_id}"));
        }
        if self.state == RunState::Contaminated {
            self.decisions.insert(
                decision_id.to_owned(),
                DecisionRecord {
                    action_state: ActionState::NotRequested,
                    decision_state: DecisionState::RejectedRunContaminated,
                    allocator_actual: None,
                    escalation_actual: None,
                },
            );
            return Ok(false);
        }

        if !self.fits(self.config.allocator_upper_bound) {
            self.decisions.insert(
                decision_id.to_owned(),
                DecisionRecord {
                    action_state: ActionState::NotRequested,
                    decision_state: DecisionState::RejectedAllocatorBudget,
                    allocator_actual: None,
                    escalation_actual: None,
                },
            );
            return Ok(false);
        }

        self.reserved = self
            .reserved
            .checked_add(self.config.allocator_upper_bound)
            .ok_or_else(|| "internal reserved-resource overflow".to_owned())?;
        self.decisions.insert(
            decision_id.to_owned(),
            DecisionRecord {
                action_state: ActionState::NotRequested,
                decision_state: DecisionState::AllocatorReserved,
                allocator_actual: None,
                escalation_actual: None,
            },
        );
        Ok(true)
    }

    /// Reserve a full-mode escalation only after the allocator reservation
    /// exists. Returning false means the caller must not run the escalation.
    pub fn reserve_escalation(&mut self, decision_id: &str) -> Result<bool, String> {
        let record = self
            .decisions
            .get(decision_id)
            .ok_or_else(|| format!("unknown decision_id {decision_id}"))?;
        if record.decision_state != DecisionState::AllocatorReserved {
            return Err("escalation reservation requires an unsettled allocator reservation".to_owned());
        }
        if record.action_state != ActionState::NotRequested {
            return Err("escalation reservation may be requested only once".to_owned());
        }
        if self.state == RunState::Contaminated {
            self.decisions.get_mut(decision_id).expect("record exists").action_state =
                ActionState::RejectedBudget;
            return Ok(false);
        }

        if !self.fits(self.config.escalation_upper_bound) {
            self.decisions.get_mut(decision_id).expect("record exists").action_state =
                ActionState::RejectedBudget;
            return Ok(false);
        }

        self.reserved = self
            .reserved
            .checked_add(self.config.escalation_upper_bound)
            .ok_or_else(|| "internal reserved-resource overflow".to_owned())?;
        self.decisions.get_mut(decision_id).expect("record exists").action_state =
            ActionState::Reserved;
        Ok(true)
    }

    /// Settle measured allocator cost and, iff separately reserved, measured
    /// escalation cost. Unknown measurement or unreserved work contaminates the
    /// run; overruns are charged and permanently block all future admissions.
    pub fn settle(
        &mut self,
        decision_id: &str,
        allocator_actual: Option<ResourceVector>,
        escalation_actual: Option<ResourceVector>,
    ) -> Result<(), String> {
        let record = self
            .decisions
            .get(decision_id)
            .ok_or_else(|| format!("unknown decision_id {decision_id}"))?;
        if record.decision_state != DecisionState::AllocatorReserved {
            return Err("only an open allocator reservation can be settled".to_owned());
        }

        let action_state = record.action_state;
        let allocator_bound = self.config.allocator_upper_bound;
        let escalation_bound = self.config.escalation_upper_bound;

        let Some(allocator_cost) = allocator_actual else {
            self.mark_contaminated(decision_id);
            return Err("allocator measurement missing; run contaminated and outstanding reservation retained".to_owned());
        };
        if !allocator_cost.is_positive() {
            self.mark_contaminated(decision_id);
            return Err("allocator measurement must be positive in both resources".to_owned());
        }

        let (escalation_cost, escalation_reserved) = match (action_state, escalation_actual) {
            (ActionState::Reserved, Some(cost)) if cost.is_positive() => (cost, true),
            (ActionState::Reserved, None) => {
                self.mark_contaminated(decision_id);
                return Err("escalation measurement missing after reserved work; run contaminated".to_owned());
            }
            (ActionState::Reserved, Some(_)) => {
                self.mark_contaminated(decision_id);
                return Err("escalation measurement must be positive in both resources; run contaminated".to_owned());
            }
            (ActionState::NotRequested | ActionState::RejectedBudget, None) => {
                (ResourceVector::default(), false)
            }
            (ActionState::NotRequested | ActionState::RejectedBudget, Some(_)) => {
                self.mark_contaminated(decision_id);
                return Err("unreserved escalation work observed; run contaminated".to_owned());
            }
        };

        let mut reserved_to_release = allocator_bound;
        if escalation_reserved {
            reserved_to_release = match reserved_to_release.checked_add(escalation_bound) {
                Some(sum) => sum,
                None => {
                    self.mark_contaminated(decision_id);
                    return Err("reservation arithmetic overflow; run contaminated".to_owned());
                }
            };
        }
        let actual_total = match allocator_cost.checked_add(escalation_cost) {
            Some(sum) => sum,
            None => {
                self.mark_contaminated(decision_id);
                return Err("actual cost arithmetic overflow; run contaminated".to_owned());
            }
        };
        let next_spent = match self.spent.checked_add(actual_total) {
            Some(sum) => sum,
            None => {
                self.mark_contaminated(decision_id);
                return Err("spent-resource arithmetic overflow; run contaminated".to_owned());
            }
        };
        let next_reserved = match self.reserved.checked_sub(reserved_to_release) {
            Some(value) => value,
            None => {
                self.mark_contaminated(decision_id);
                return Err("reserved-resource ledger underflow; run contaminated".to_owned());
            }
        };

        let allocator_overrun = !allocator_cost.componentwise_le(allocator_bound);
        let escalation_overrun =
            escalation_reserved && !escalation_cost.componentwise_le(escalation_bound);
        let run_budget_overrun = !next_spent.checked_add(next_reserved).is_some_and(|used| {
            used.componentwise_le(self.config.run_budget)
        });

        self.spent = next_spent;
        self.reserved = next_reserved;
        let record = self.decisions.get_mut(decision_id).expect("record exists");
        record.allocator_actual = Some(allocator_cost);
        record.escalation_actual = if escalation_reserved { Some(escalation_cost) } else { None };

        if allocator_overrun || escalation_overrun || run_budget_overrun {
            self.mark_contaminated(decision_id);
            return Err("measured cost exceeded a reserved bound or all-in run budget; run contaminated".to_owned());
        }

        let record = self.decisions.get_mut(decision_id).expect("record exists");
        record.decision_state = if escalation_reserved {
            DecisionState::SettledEscalated
        } else {
            DecisionState::SettledCheap
        };
        Ok(())
    }

    pub fn snapshot(&self) -> LedgerSnapshot {
        let mut settled_cheap = 0usize;
        let mut settled_escalated = 0usize;
        let mut rejected_allocator_budget = 0usize;
        let mut rejected_run_contaminated = 0usize;
        let mut rejected_escalation_budget = 0usize;
        let mut contaminated_decisions = 0usize;
        for record in self.decisions.values() {
            match record.decision_state {
                DecisionState::SettledCheap => settled_cheap += 1,
                DecisionState::SettledEscalated => settled_escalated += 1,
                DecisionState::RejectedAllocatorBudget => rejected_allocator_budget += 1,
                DecisionState::RejectedRunContaminated => rejected_run_contaminated += 1,
                DecisionState::Contaminated => contaminated_decisions += 1,
                DecisionState::AllocatorReserved => {}
            }
            if record.action_state == ActionState::RejectedBudget {
                rejected_escalation_budget += 1;
            }
        }
        let remaining = ResourceVector {
            latency_us: self.config.run_budget.latency_us.saturating_sub(
                self.spent.latency_us.saturating_add(self.reserved.latency_us),
            ),
            energy_uj: self.config.run_budget.energy_uj.saturating_sub(
                self.spent.energy_uj.saturating_add(self.reserved.energy_uj),
            ),
        };
        LedgerSnapshot {
            state: self.state,
            spent: self.spent,
            reserved: self.reserved,
            remaining,
            decisions: self.decisions.len(),
            settled_cheap,
            settled_escalated,
            rejected_allocator_budget,
            rejected_run_contaminated,
            rejected_escalation_budget,
            contaminated_decisions,
        }
    }

    /// Stable line-oriented receipt; identifiers are sorted lexicographically.
    pub fn receipt(&self) -> String {
        let snapshot = self.snapshot();
        let mut out = String::new();
        let _ = writeln!(out, "LEDGER\tformat\tVIS-DVPC-ALL-IN-BUDGET-001-v1");
        let _ = writeln!(out, "LEDGER\trun_state\t{}", snapshot.state.label());
        let _ = writeln!(out, "LEDGER\tbudget_latency_us\t{}", self.config.run_budget.latency_us);
        let _ = writeln!(out, "LEDGER\tbudget_energy_uj\t{}", self.config.run_budget.energy_uj);
        let _ = writeln!(out, "LEDGER\tspent_latency_us\t{}", snapshot.spent.latency_us);
        let _ = writeln!(out, "LEDGER\tspent_energy_uj\t{}", snapshot.spent.energy_uj);
        let _ = writeln!(out, "LEDGER\treserved_latency_us\t{}", snapshot.reserved.latency_us);
        let _ = writeln!(out, "LEDGER\treserved_energy_uj\t{}", snapshot.reserved.energy_uj);
        let _ = writeln!(out, "LEDGER\tremaining_latency_us\t{}", snapshot.remaining.latency_us);
        let _ = writeln!(out, "LEDGER\tremaining_energy_uj\t{}", snapshot.remaining.energy_uj);
        let _ = writeln!(out, "LEDGER\tdecisions\t{}", snapshot.decisions);
        let _ = writeln!(out, "LEDGER\tsettled_cheap\t{}", snapshot.settled_cheap);
        let _ = writeln!(out, "LEDGER\tsettled_escalated\t{}", snapshot.settled_escalated);
        let _ = writeln!(out, "LEDGER\trejected_allocator_budget\t{}", snapshot.rejected_allocator_budget);
        let _ = writeln!(out, "LEDGER\trejected_run_contaminated\t{}", snapshot.rejected_run_contaminated);
        let _ = writeln!(out, "LEDGER\trejected_escalation_budget\t{}", snapshot.rejected_escalation_budget);
        let _ = writeln!(out, "LEDGER\tcontaminated_decisions\t{}", snapshot.contaminated_decisions);
        let _ = writeln!(out, "LEDGER\tperformance_claim\tNONE");
        for (decision_id, record) in &self.decisions {
            let _ = writeln!(
                out,
                "DECISION\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                decision_id,
                record.decision_state.label(),
                record.action_state.label(),
                optional_component(record.allocator_actual, |v| v.latency_us),
                optional_component(record.allocator_actual, |v| v.energy_uj),
                optional_component(record.escalation_actual, |v| v.latency_us),
                optional_component(record.escalation_actual, |v| v.energy_uj),
                self.config.allocator_upper_bound.latency_us,
                self.config.allocator_upper_bound.energy_uj,
                self.config.escalation_upper_bound.latency_us,
                self.config.escalation_upper_bound.energy_uj,
            );
        }
        out
    }
}

fn optional_component(value: Option<ResourceVector>, select: impl FnOnce(ResourceVector) -> u64) -> String {
    value.map(select).map(|number| number.to_string()).unwrap_or_else(|| "-".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(latency: u64, energy: u64) -> LedgerConfig {
        LedgerConfig {
            run_budget: ResourceVector::new(latency, energy),
            allocator_upper_bound: ResourceVector::new(10, 3),
            escalation_upper_bound: ResourceVector::new(20, 7),
        }
    }

    #[test]
    fn allocator_cost_is_reserved_before_decision_runs() {
        let mut ledger = AllInBudgetLedger::new(config(100, 100)).unwrap();
        assert!(ledger.reserve_allocator("d1").unwrap());
        let snapshot = ledger.snapshot();
        assert_eq!(snapshot.reserved, ResourceVector::new(10, 3));
        assert_eq!(snapshot.remaining, ResourceVector::new(90, 97));
        assert_eq!(snapshot.spent, ResourceVector::default());
    }

    #[test]
    fn allocator_reservation_is_componentwise_and_atomic() {
        let mut ledger = AllInBudgetLedger::new(config(9, 100)).unwrap();
        assert!(!ledger.reserve_allocator("d1").unwrap());
        assert_eq!(ledger.snapshot().reserved, ResourceVector::default());
        assert_eq!(ledger.snapshot().rejected_allocator_budget, 1);
        let mut ledger2 = AllInBudgetLedger::new(config(100, 2)).unwrap();
        assert!(!ledger2.reserve_allocator("d1").unwrap());
        assert_eq!(ledger2.snapshot().reserved, ResourceVector::default());
    }

    #[test]
    fn escalation_reservation_counts_allocator_reserve_before_admission() {
        let mut ledger = AllInBudgetLedger::new(config(29, 100)).unwrap();
        assert!(ledger.reserve_allocator("d1").unwrap());
        assert!(!ledger.reserve_escalation("d1").unwrap());
        assert_eq!(ledger.snapshot().reserved, ResourceVector::new(10, 3));
        assert_eq!(ledger.snapshot().rejected_escalation_budget, 1);
        ledger.settle("d1", Some(ResourceVector::new(8, 2)), None).unwrap();
        assert_eq!(ledger.snapshot().spent, ResourceVector::new(8, 2));
    }

    #[test]
    fn settled_cheap_path_releases_allocator_slack() {
        let mut ledger = AllInBudgetLedger::new(config(100, 100)).unwrap();
        assert!(ledger.reserve_allocator("d1").unwrap());
        ledger.settle("d1", Some(ResourceVector::new(7, 2)), None).unwrap();
        let snapshot = ledger.snapshot();
        assert_eq!(snapshot.spent, ResourceVector::new(7, 2));
        assert_eq!(snapshot.reserved, ResourceVector::default());
        assert_eq!(snapshot.remaining, ResourceVector::new(93, 98));
        assert_eq!(snapshot.settled_cheap, 1);
    }

    #[test]
    fn full_path_reserves_and_settles_both_costs() {
        let mut ledger = AllInBudgetLedger::new(config(100, 100)).unwrap();
        assert!(ledger.reserve_allocator("d1").unwrap());
        assert!(ledger.reserve_escalation("d1").unwrap());
        assert_eq!(ledger.snapshot().reserved, ResourceVector::new(30, 10));
        ledger.settle(
            "d1",
            Some(ResourceVector::new(8, 2)),
            Some(ResourceVector::new(14, 5)),
        ).unwrap();
        let snapshot = ledger.snapshot();
        assert_eq!(snapshot.spent, ResourceVector::new(22, 7));
        assert_eq!(snapshot.reserved, ResourceVector::default());
        assert_eq!(snapshot.settled_escalated, 1);
    }

    #[test]
    fn energy_exhaustion_blocks_escalation_even_if_latency_fits() {
        let mut ledger = AllInBudgetLedger::new(config(1_000, 9)).unwrap();
        assert!(ledger.reserve_allocator("d1").unwrap());
        assert!(!ledger.reserve_escalation("d1").unwrap());
        ledger.settle("d1", Some(ResourceVector::new(5, 2)), None).unwrap();
        assert_eq!(ledger.snapshot().settled_cheap, 1);
        assert_eq!(ledger.snapshot().rejected_escalation_budget, 1);
    }

    #[test]
    fn allocator_overrun_contaminates_and_blocks_future_decisions() {
        let mut ledger = AllInBudgetLedger::new(config(100, 100)).unwrap();
        assert!(ledger.reserve_allocator("d1").unwrap());
        assert!(ledger.settle("d1", Some(ResourceVector::new(11, 2)), None).is_err());
        assert_eq!(ledger.snapshot().state, RunState::Contaminated);
        assert!(!ledger.reserve_allocator("d2").unwrap());
        assert_eq!(ledger.snapshot().rejected_run_contaminated, 1);
    }

    #[test]
    fn escalation_overrun_contaminates_even_when_other_resource_has_slack() {
        let mut ledger = AllInBudgetLedger::new(config(1_000, 1_000)).unwrap();
        assert!(ledger.reserve_allocator("d1").unwrap());
        assert!(ledger.reserve_escalation("d1").unwrap());
        assert!(ledger.settle(
            "d1",
            Some(ResourceVector::new(8, 2)),
            Some(ResourceVector::new(14, 8)),
        ).is_err());
        assert_eq!(ledger.snapshot().state, RunState::Contaminated);
        assert_eq!(ledger.snapshot().spent, ResourceVector::new(22, 10));
        assert_eq!(ledger.snapshot().reserved, ResourceVector::default());
    }

    #[test]
    fn unknown_measurement_contaminates_and_keeps_reservation_visible() {
        let mut ledger = AllInBudgetLedger::new(config(100, 100)).unwrap();
        assert!(ledger.reserve_allocator("d1").unwrap());
        assert!(ledger.settle("d1", None, None).is_err());
        let snapshot = ledger.snapshot();
        assert_eq!(snapshot.state, RunState::Contaminated);
        assert_eq!(snapshot.reserved, ResourceVector::new(10, 3));
        assert_eq!(snapshot.remaining, ResourceVector::new(90, 97));
    }

    #[test]
    fn unreserved_escalation_is_never_settled_as_valid_work() {
        let mut ledger = AllInBudgetLedger::new(config(100, 100)).unwrap();
        assert!(ledger.reserve_allocator("d1").unwrap());
        assert!(ledger.settle(
            "d1",
            Some(ResourceVector::new(5, 2)),
            Some(ResourceVector::new(10, 3)),
        ).is_err());
        assert_eq!(ledger.snapshot().state, RunState::Contaminated);
    }

    #[test]
    fn duplicate_ids_and_double_settlement_fail() {
        let mut ledger = AllInBudgetLedger::new(config(100, 100)).unwrap();
        assert!(ledger.reserve_allocator("d1").unwrap());
        assert!(ledger.reserve_allocator("d1").is_err());
        ledger.settle("d1", Some(ResourceVector::new(5, 2)), None).unwrap();
        assert!(ledger.settle("d1", Some(ResourceVector::new(5, 2)), None).is_err());
    }

    #[test]
    fn reservations_account_for_other_outstanding_decisions() {
        let mut ledger = AllInBudgetLedger::new(config(40, 20)).unwrap();
        assert!(ledger.reserve_allocator("d1").unwrap());
        assert!(ledger.reserve_escalation("d1").unwrap());
        assert!(!ledger.reserve_allocator("d2").unwrap());
        assert_eq!(ledger.snapshot().reserved, ResourceVector::new(30, 10));
    }

    #[test]
    fn receipt_is_deterministic_and_exposes_both_resource_dimensions() {
        let mut ledger = AllInBudgetLedger::new(config(100, 100)).unwrap();
        assert!(ledger.reserve_allocator("z").unwrap());
        assert!(ledger.reserve_escalation("z").unwrap());
        ledger.settle("z", Some(ResourceVector::new(8, 2)), Some(ResourceVector::new(14, 5))).unwrap();
        assert_eq!(ledger.receipt(), ledger.receipt());
        assert!(ledger.receipt().contains("LEDGER\tspent_latency_us\t22"));
        assert!(ledger.receipt().contains("LEDGER\tspent_energy_uj\t7"));
        assert!(ledger.receipt().contains("LEDGER\tperformance_claim\tNONE"));
    }

    #[test]
    fn invalid_configuration_and_ids_fail_closed() {
        let invalid_config = LedgerConfig {
            run_budget: ResourceVector::new(100, 100),
            allocator_upper_bound: ResourceVector::new(0, 3),
            escalation_upper_bound: ResourceVector::new(20, 7),
        };
        assert!(AllInBudgetLedger::new(invalid_config).is_err());
        assert!(AllInBudgetLedger::new(config(5, 100)).is_err());
        let mut ledger = AllInBudgetLedger::new(config(100, 100)).unwrap();
        assert!(ledger.reserve_allocator("../d1").is_err());
    }

    #[test]
    fn resource_arithmetic_overflow_is_detected() {
        assert!(ResourceVector::new(u64::MAX, 1).checked_add(ResourceVector::new(1, 0)).is_none());
        let mut ledger = AllInBudgetLedger::new(LedgerConfig {
            run_budget: ResourceVector::new(u64::MAX, u64::MAX),
            allocator_upper_bound: ResourceVector::new(u64::MAX - 1, 3),
            escalation_upper_bound: ResourceVector::new(1, 1),
        }).unwrap();
        assert!(ledger.reserve_allocator("d1").unwrap());
        assert!(ledger.settle("d1", Some(ResourceVector::new(u64::MAX, 2)), None).is_err());
        assert_eq!(ledger.snapshot().state, RunState::Contaminated);
    }
}
