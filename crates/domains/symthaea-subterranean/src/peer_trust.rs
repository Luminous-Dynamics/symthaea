// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Peer authentication trust ledger.
//!
//! Team-level authority decisions (leadership-lease tallying, occupancy
//! yielding, rescue handoff) must not be steerable by an unauthenticated or
//! replayed peer identity. This module tracks which peers currently hold a
//! valid, deployment-scoped authentication assertion, and is the sole gate
//! other team subsystems consult before trusting a peer-reported claim. It
//! does not perform cryptographic verification itself -- `authentication_verified`
//! and `hardware_backed` are asserted by an external identity/attestation
//! layer -- but it does enforce deployment scoping, schema validity, and
//! replay/supersession ordering, matching this crate's `team.rs` convention
//! of deterministic identity/epoch/sequence checks without claiming
//! cryptographic authenticity itself.

use crate::team::AgentId;
use serde::{Deserialize, Serialize};

pub const PEER_TRUST_SCHEMA_VERSION: u16 = 1;
pub const MAX_TRUSTED_PEERS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerAuthenticationAssertion {
    pub schema_version: u16,
    pub agent_id: AgentId,
    pub deployment_id: u64,
    pub epoch: u64,
    pub sequence: u64,
    pub issued_step: u64,
    pub expires_step: u64,
    pub authentication_verified: bool,
    pub hardware_backed: bool,
}

impl PeerAuthenticationAssertion {
    pub const fn validate(&self) -> bool {
        self.schema_version == PEER_TRUST_SCHEMA_VERSION && self.issued_step <= self.expires_step
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PeerTrustRejection {
    Malformed,
    SchemaVersionMismatch,
    WrongDeployment,
    NotAuthenticated,
    Superseded,
    DirectoryFull,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PeerAuthenticationOutcome {
    Trusted,
    TrustedNotHardwareBacked,
    Untrusted,
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct PeerTrustPolicy {
    /// When set, a peer without a hardware-backed assertion is treated as
    /// untrusted rather than merely lower-confidence.
    pub require_hardware_backed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct TrustedPeerRecord {
    assertion: PeerAuthenticationAssertion,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PeerTrustSupervisor {
    schema_version: u16,
    deployment_id: u64,
    records: Vec<TrustedPeerRecord>,
}

impl PeerTrustSupervisor {
    pub fn new(deployment_id: u64) -> Self {
        Self {
            schema_version: PEER_TRUST_SCHEMA_VERSION,
            deployment_id,
            records: Vec::new(),
        }
    }

    pub fn validate(&self) -> bool {
        self.schema_version == PEER_TRUST_SCHEMA_VERSION
            && self.records.len() <= MAX_TRUSTED_PEERS
            && self
                .records
                .iter()
                .all(|record| record.assertion.validate())
    }

    /// Ingest a peer's authentication assertion. Staleness relative to "now"
    /// is deliberately not checked here -- ingestion only records what was
    /// asserted and when it was superseded; freshness is evaluated at query
    /// time in [`Self::outcome`], since only the caller knows the current step.
    pub fn ingest(
        &mut self,
        assertion: PeerAuthenticationAssertion,
    ) -> Result<(), PeerTrustRejection> {
        if !assertion.validate() {
            return Err(PeerTrustRejection::Malformed);
        }
        if assertion.schema_version != PEER_TRUST_SCHEMA_VERSION {
            return Err(PeerTrustRejection::SchemaVersionMismatch);
        }
        if assertion.deployment_id != self.deployment_id {
            return Err(PeerTrustRejection::WrongDeployment);
        }
        if !assertion.authentication_verified {
            return Err(PeerTrustRejection::NotAuthenticated);
        }
        if let Some(existing) = self
            .records
            .iter()
            .position(|record| record.assertion.agent_id == assertion.agent_id)
        {
            let current = self.records[existing].assertion;
            let superseded = assertion.epoch < current.epoch
                || (assertion.epoch == current.epoch && assertion.sequence <= current.sequence);
            if superseded {
                return Err(PeerTrustRejection::Superseded);
            }
            self.records[existing] = TrustedPeerRecord { assertion };
            return Ok(());
        }
        if self.records.len() >= MAX_TRUSTED_PEERS {
            return Err(PeerTrustRejection::DirectoryFull);
        }
        self.records.push(TrustedPeerRecord { assertion });
        Ok(())
    }

    pub fn prune_expired(&mut self, current_step: u64) {
        self.records
            .retain(|record| current_step <= record.assertion.expires_step);
    }

    pub fn outcome(
        &self,
        agent_id: AgentId,
        current_step: u64,
        policy: PeerTrustPolicy,
    ) -> PeerAuthenticationOutcome {
        match self
            .records
            .iter()
            .find(|record| record.assertion.agent_id == agent_id)
        {
            Some(record) if current_step <= record.assertion.expires_step => {
                if record.assertion.hardware_backed {
                    PeerAuthenticationOutcome::Trusted
                } else if policy.require_hardware_backed {
                    PeerAuthenticationOutcome::Untrusted
                } else {
                    PeerAuthenticationOutcome::TrustedNotHardwareBacked
                }
            }
            _ => PeerAuthenticationOutcome::Untrusted,
        }
    }

    pub fn is_trusted(
        &self,
        agent_id: AgentId,
        current_step: u64,
        policy: PeerTrustPolicy,
    ) -> bool {
        matches!(
            self.outcome(agent_id, current_step, policy),
            PeerAuthenticationOutcome::Trusted
                | PeerAuthenticationOutcome::TrustedNotHardwareBacked
        )
    }

    pub fn trusted_count(&self, current_step: u64, policy: PeerTrustPolicy) -> usize {
        self.records
            .iter()
            .filter(|record| current_step <= record.assertion.expires_step)
            .filter(|record| !policy.require_hardware_backed || record.assertion.hardware_backed)
            .count()
    }

    pub const fn deployment_id(&self) -> u64 {
        self.deployment_id
    }

    pub fn reset(&mut self) {
        self.records.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assertion(agent: u64, deployment: u64, sequence: u64) -> PeerAuthenticationAssertion {
        PeerAuthenticationAssertion {
            schema_version: PEER_TRUST_SCHEMA_VERSION,
            agent_id: AgentId::new(agent),
            deployment_id: deployment,
            epoch: 1,
            sequence,
            issued_step: 1,
            expires_step: 100,
            authentication_verified: true,
            hardware_backed: true,
        }
    }

    #[test]
    fn wrong_deployment_is_rejected() {
        let mut supervisor = PeerTrustSupervisor::new(7);
        assert_eq!(
            supervisor.ingest(assertion(2, 8, 1)),
            Err(PeerTrustRejection::WrongDeployment)
        );
    }

    #[test]
    fn replayed_sequence_is_rejected_and_trust_persists_past_expiry_query() {
        let mut supervisor = PeerTrustSupervisor::new(7);
        assert_eq!(supervisor.ingest(assertion(2, 7, 5)), Ok(()));
        assert_eq!(
            supervisor.ingest(assertion(2, 7, 5)),
            Err(PeerTrustRejection::Superseded)
        );
        let policy = PeerTrustPolicy::default();
        assert!(supervisor.is_trusted(AgentId::new(2), 50, policy));
        assert!(!supervisor.is_trusted(AgentId::new(2), 200, policy));
    }
}
