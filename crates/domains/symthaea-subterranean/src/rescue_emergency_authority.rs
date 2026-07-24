// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded emergency-rescue authorization when a subject cannot communicate.
//!
//! The crate does not diagnose incapacity and does not verify signatures. It
//! consumes externally authenticated assertions, requires two distinct
//! hardware-backed roles, binds them to one rescue case, and expires authority.

use crate::operator_protocol::{AuthenticationLevel, OperatorId};
use crate::rescue::RescueCaseId;
use crate::team::AgentId;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const EMERGENCY_RESCUE_AUTHORITY_SCHEMA_VERSION: u16 = 1;
pub const MAX_EMERGENCY_RESCUE_AUTHORIZATIONS: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EmergencyRescueRole {
    SafetyOfficer,
    IndependentWitness,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmergencyRescueApproval {
    pub operator: OperatorId,
    pub role: EmergencyRescueRole,
    pub authentication: AuthenticationLevel,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmergencyRescueAuthorization {
    pub schema_version: u16,
    pub authorization_id: u64,
    pub subject: AgentId,
    pub case_id: RescueCaseId,
    pub epoch: u32,
    pub sequence: u64,
    pub issued_step: u64,
    pub expires_step: u64,
    pub immediate_threat: bool,
    pub communication_unavailable: bool,
    pub approvals: Vec<EmergencyRescueApproval>,
}

impl EmergencyRescueAuthorization {
    pub fn validate(&self) -> bool {
        if self.schema_version != EMERGENCY_RESCUE_AUTHORITY_SCHEMA_VERSION
            || self.authorization_id == 0
            || self.subject == AgentId::SURFACE_CONTROL
            || self.case_id.0 == 0
            || self.expires_step < self.issued_step
            || !self.immediate_threat
            || !self.communication_unavailable
        {
            return false;
        }
        let verified: Vec<&EmergencyRescueApproval> = self
            .approvals
            .iter()
            .filter(|approval| {
                approval.operator.is_valid()
                    && approval.authentication == AuthenticationLevel::HardwareBacked
            })
            .collect();
        let operators: BTreeSet<OperatorId> = verified
            .iter()
            .map(|approval| approval.operator)
            .collect();
        let roles: BTreeSet<EmergencyRescueRole> = verified
            .iter()
            .map(|approval| approval.role)
            .collect();
        operators.len() >= 2
            && roles.contains(&EmergencyRescueRole::SafetyOfficer)
            && roles.contains(&EmergencyRescueRole::IndependentWitness)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmergencyRescueAuthorizationRejection {
    InvalidAuthorization,
    Replay,
    EpochRegression,
    Expired,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmergencyRescueAuthorityLedger {
    records: BTreeMap<(AgentId, RescueCaseId), EmergencyRescueAuthorization>,
}

impl EmergencyRescueAuthorityLedger {
    pub fn new() -> Self {
        Self {
            records: BTreeMap::new(),
        }
    }

    pub fn ingest(
        &mut self,
        authorization: EmergencyRescueAuthorization,
        current_step: u64,
    ) -> Result<(), EmergencyRescueAuthorizationRejection> {
        if !authorization.validate() {
            return Err(EmergencyRescueAuthorizationRejection::InvalidAuthorization);
        }
        if current_step > authorization.expires_step {
            return Err(EmergencyRescueAuthorizationRejection::Expired);
        }
        let key = (authorization.subject, authorization.case_id);
        if let Some(previous) = self.records.get(&key) {
            if authorization.epoch < previous.epoch {
                return Err(EmergencyRescueAuthorizationRejection::EpochRegression);
            }
            if authorization.epoch == previous.epoch
                && authorization.sequence <= previous.sequence
            {
                return Err(EmergencyRescueAuthorizationRejection::Replay);
            }
        }
        if self.records.len() >= MAX_EMERGENCY_RESCUE_AUTHORIZATIONS
            && !self.records.contains_key(&key)
        {
            let oldest = self
                .records
                .iter()
                .min_by_key(|(_, record)| (record.expires_step, record.issued_step))
                .map(|(key, _)| *key);
            if let Some(oldest) = oldest {
                self.records.remove(&oldest);
            }
        }
        self.records.insert(key, authorization);
        Ok(())
    }

    pub fn permits(
        &self,
        subject: AgentId,
        case_id: RescueCaseId,
        current_step: u64,
    ) -> bool {
        self.records
            .get(&(subject, case_id))
            .is_some_and(|authorization| {
                current_step <= authorization.expires_step && authorization.validate()
            })
    }

    pub fn expire(&mut self, current_step: u64) {
        self.records
            .retain(|_, authorization| current_step <= authorization.expires_step);
    }

    pub fn validate(&self) -> bool {
        self.records.len() <= MAX_EMERGENCY_RESCUE_AUTHORIZATIONS
            && self.records.iter().all(|((subject, case_id), authorization)| {
                *subject == authorization.subject
                    && *case_id == authorization.case_id
                    && authorization.validate()
            })
    }
}

impl Default for EmergencyRescueAuthorityLedger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn authorization() -> EmergencyRescueAuthorization {
        EmergencyRescueAuthorization {
            schema_version: EMERGENCY_RESCUE_AUTHORITY_SCHEMA_VERSION,
            authorization_id: 4,
            subject: AgentId::new(2),
            case_id: RescueCaseId(9),
            epoch: 1,
            sequence: 1,
            issued_step: 20,
            expires_step: 40,
            immediate_threat: true,
            communication_unavailable: true,
            approvals: vec![
                EmergencyRescueApproval {
                    operator: OperatorId::new(11),
                    role: EmergencyRescueRole::SafetyOfficer,
                    authentication: AuthenticationLevel::HardwareBacked,
                },
                EmergencyRescueApproval {
                    operator: OperatorId::new(12),
                    role: EmergencyRescueRole::IndependentWitness,
                    authentication: AuthenticationLevel::HardwareBacked,
                },
            ],
        }
    }

    #[test]
    fn two_distinct_hardware_backed_roles_are_required() {
        let mut invalid = authorization();
        invalid.approvals[1].operator = OperatorId::new(11);
        assert!(!invalid.validate());
        assert!(authorization().validate());
    }

    #[test]
    fn authority_expires_and_cannot_be_replayed() {
        let mut ledger = EmergencyRescueAuthorityLedger::new();
        let value = authorization();
        assert!(ledger.ingest(value.clone(), 20).is_ok());
        assert_eq!(
            ledger.ingest(value, 21),
            Err(EmergencyRescueAuthorizationRejection::Replay)
        );
        assert!(ledger.permits(AgentId::new(2), RescueCaseId(9), 40));
        assert!(!ledger.permits(AgentId::new(2), RescueCaseId(9), 41));
    }
}
