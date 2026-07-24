// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded service accounting for stakeholders competing for scarce mission capacity.
//!
//! Fairness evidence never authorizes motion. It can only identify sustained
//! under-service and reduce discretionary work until a higher-level planner or
//! accountable operator resolves the conflict.

use crate::objective_budget::ConflictObjective;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const FAIRNESS_LEDGER_SCHEMA_VERSION: u16 = 1;
pub const MAX_FAIRNESS_STAKEHOLDERS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StakeholderId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ServiceAccount {
    pub requested_units: f64,
    pub served_units: f64,
    pub deferred_events: u64,
    pub last_requested_step: u64,
    pub last_served_step: Option<u64>,
}

impl ServiceAccount {
    pub const fn empty() -> Self {
        Self {
            requested_units: 0.0,
            served_units: 0.0,
            deferred_events: 0,
            last_requested_step: 0,
            last_served_step: None,
        }
    }

    pub fn service_ratio(self) -> f64 {
        if self.requested_units <= f64::EPSILON {
            1.0
        } else {
            (self.served_units / self.requested_units).clamp(0.0, 1.0)
        }
    }

    fn validate(self) -> bool {
        self.requested_units.is_finite()
            && self.requested_units >= 0.0
            && self.served_units.is_finite()
            && self.served_units >= 0.0
            && self.served_units <= self.requested_units + f64::EPSILON
            && self
                .last_served_step
                .is_none_or(|served| served <= self.last_requested_step)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ServiceKey {
    pub stakeholder: StakeholderId,
    pub objective: ConflictObjective,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FairnessAssessment {
    pub minimum_service_ratio: f64,
    pub jain_index: f64,
    pub underserved: Vec<ServiceKey>,
    pub tracked_accounts: usize,
}

impl FairnessAssessment {
    pub fn nominal() -> Self {
        Self {
            minimum_service_ratio: 1.0,
            jain_index: 1.0,
            underserved: Vec::new(),
            tracked_accounts: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairnessLedger {
    schema_version: u16,
    minimum_expected_ratio: f64,
    accounts: BTreeMap<ServiceKey, ServiceAccount>,
    rejected_accounts: u64,
    last: FairnessAssessment,
}

impl FairnessLedger {
    pub fn new(minimum_expected_ratio: f64) -> Self {
        Self {
            schema_version: FAIRNESS_LEDGER_SCHEMA_VERSION,
            minimum_expected_ratio: minimum_expected_ratio.clamp(0.0, 1.0),
            accounts: BTreeMap::new(),
            rejected_accounts: 0,
            last: FairnessAssessment::nominal(),
        }
    }

    pub fn validate(&self) -> bool {
        self.schema_version == FAIRNESS_LEDGER_SCHEMA_VERSION
            && self.minimum_expected_ratio.is_finite()
            && (0.0..=1.0).contains(&self.minimum_expected_ratio)
            && self.accounts.len() <= MAX_FAIRNESS_STAKEHOLDERS
            && self.accounts.values().copied().all(ServiceAccount::validate)
            && self.last.minimum_service_ratio.is_finite()
            && self.last.jain_index.is_finite()
            && self.last.underserved.len() <= MAX_FAIRNESS_STAKEHOLDERS
    }

    pub fn record(
        &mut self,
        step: u64,
        stakeholder: StakeholderId,
        objective: ConflictObjective,
        requested_units: f64,
        served_units: f64,
    ) -> bool {
        if !requested_units.is_finite()
            || !served_units.is_finite()
            || requested_units <= 0.0
            || served_units < 0.0
            || served_units > requested_units
        {
            return false;
        }
        let key = ServiceKey {
            stakeholder,
            objective,
        };
        if !self.accounts.contains_key(&key) && self.accounts.len() >= MAX_FAIRNESS_STAKEHOLDERS {
            self.rejected_accounts = self.rejected_accounts.saturating_add(1);
            return false;
        }
        let account = self.accounts.entry(key).or_insert(ServiceAccount::empty());
        account.requested_units += requested_units;
        account.served_units += served_units;
        account.last_requested_step = step;
        if served_units > 0.0 {
            account.last_served_step = Some(step);
        }
        if served_units + f64::EPSILON < requested_units {
            account.deferred_events = account.deferred_events.saturating_add(1);
        }
        self.last = self.assess();
        true
    }

    pub fn assess(&self) -> FairnessAssessment {
        if self.accounts.is_empty() {
            return FairnessAssessment::nominal();
        }
        let mut minimum_service_ratio = 1.0f64;
        let mut sum = 0.0f64;
        let mut sum_squares = 0.0f64;
        let mut underserved = Vec::new();
        for (key, account) in &self.accounts {
            let ratio = account.service_ratio();
            minimum_service_ratio = minimum_service_ratio.min(ratio);
            sum += ratio;
            sum_squares += ratio * ratio;
            if ratio + f64::EPSILON < self.minimum_expected_ratio {
                underserved.push(*key);
            }
        }
        let count = self.accounts.len() as f64;
        let jain_index = if sum_squares <= f64::EPSILON {
            0.0
        } else {
            ((sum * sum) / (count * sum_squares)).clamp(0.0, 1.0)
        };
        FairnessAssessment {
            minimum_service_ratio,
            jain_index,
            underserved,
            tracked_accounts: self.accounts.len(),
        }
    }

    pub fn last(&self) -> &FairnessAssessment {
        &self.last
    }

    pub fn rejected_accounts(&self) -> u64 {
        self.rejected_accounts
    }

    pub fn account(
        &self,
        stakeholder: StakeholderId,
        objective: ConflictObjective,
    ) -> Option<ServiceAccount> {
        self.accounts
            .get(&ServiceKey {
                stakeholder,
                objective,
            })
            .copied()
    }
}

impl Default for FairnessLedger {
    fn default() -> Self {
        Self::new(0.25)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn under_service_is_measured_without_authorizing_action() {
        let mut ledger = FairnessLedger::new(0.5);
        assert!(ledger.record(
            1,
            StakeholderId(7),
            ConflictObjective::PeerAssistance,
            10.0,
            1.0,
        ));
        assert_eq!(ledger.last().underserved.len(), 1);
        assert!(ledger.last().minimum_service_ratio < 0.5);
    }

    #[test]
    fn malformed_service_claim_is_rejected() {
        let mut ledger = FairnessLedger::default();
        assert!(!ledger.record(
            1,
            StakeholderId(1),
            ConflictObjective::MissionWork,
            1.0,
            2.0,
        ));
    }
}
