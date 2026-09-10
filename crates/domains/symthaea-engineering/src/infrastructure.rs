// SPDX-License-Identifier: AGPL-3.0-or-later
//! Thin, domain-neutral infrastructure vocabulary for off-Earth and remote
//! infrastructure coordination.
//!
//! These types are deliberately descriptive. They do not grant actuator
//! authority and they do not replace domain-specific physics, control, or
//! interoperability standards.

use serde::{Deserialize, Serialize};

/// Stable identifier for a physical or virtual infrastructure asset.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AssetId(pub String);

impl AssetId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}

/// Broad resource classes used for cross-domain planning.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ResourceKind {
    Energy,
    Power,
    Water,
    Oxygen,
    Propellant,
    Bandwidth,
    Compute,
    Storage,
    CargoCapacity,
    ThermalRejection,
    HabitationVolume,
    Material(String),
    DomainSpecific(String),
}

/// Evidence status for interoperability or qualification claims.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvidenceStatus {
    Declared,
    Modeled,
    Tested,
    Qualified,
}

/// Reference to an interface or external interoperability contract.
///
/// `standard` is an identifier only. A matching string is never sufficient to
/// establish compatibility; `evidence_status` records how strongly the claim
/// has been established.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InterfaceRef {
    pub standard: String,
    pub profile: Option<String>,
    pub evidence_status: EvidenceStatus,
}

/// A bounded service offered by an infrastructure asset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServiceCapability {
    pub provider: AssetId,
    pub service: String,
    pub resource: ResourceKind,
    pub capacity: f64,
    pub available: f64,
    pub unit: String,
    pub interfaces: Vec<InterfaceRef>,
}

impl ServiceCapability {
    /// Returns false for malformed/non-finite capacity declarations.
    pub fn is_well_formed(&self) -> bool {
        self.capacity.is_finite()
            && self.available.is_finite()
            && self.capacity >= 0.0
            && self.available >= 0.0
            && self.available <= self.capacity
            && !self.service.trim().is_empty()
            && !self.unit.trim().is_empty()
    }
}

/// Provenance for a measured or derived observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationSource {
    pub source_id: String,
    pub method: String,
}

/// Time-bound observation with explicit uncertainty.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InfrastructureObservation {
    pub asset: AssetId,
    pub quantity: String,
    pub value: f64,
    pub unit: String,
    pub uncertainty: f64,
    pub observed_at_unix_ms: u64,
    pub source: ObservationSource,
}

impl InfrastructureObservation {
    pub fn is_well_formed(&self) -> bool {
        self.value.is_finite()
            && self.uncertainty.is_finite()
            && self.uncertainty >= 0.0
            && !self.quantity.trim().is_empty()
            && !self.unit.trim().is_empty()
            && !self.source.source_id.trim().is_empty()
            && !self.source.method.trim().is_empty()
    }
}

/// Severity of an identified hazard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HazardSeverity {
    Advisory,
    Caution,
    Critical,
    Catastrophic,
}

/// Evidence-bearing hazard description.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hazard {
    pub asset: AssetId,
    pub hazard_id: String,
    pub description: String,
    pub severity: HazardSeverity,
    pub evidence_refs: Vec<String>,
}

/// A temporary resource allocation. A reservation is never actuator authority.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Reservation {
    pub reservation_id: String,
    pub requester: AssetId,
    pub capability_provider: AssetId,
    pub resource: ResourceKind,
    pub amount: f64,
    pub unit: String,
    pub valid_from_unix_ms: u64,
    pub valid_until_unix_ms: u64,
}

impl Reservation {
    pub fn is_well_formed(&self) -> bool {
        self.amount.is_finite()
            && self.amount > 0.0
            && self.valid_until_unix_ms > self.valid_from_unix_ms
            && !self.reservation_id.trim().is_empty()
            && !self.unit.trim().is_empty()
    }
}

/// A descriptive action proposal. This type is intentionally non-executable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandProposal {
    pub proposal_id: String,
    pub target: AssetId,
    pub operation: String,
    pub rationale: String,
    pub evidence_refs: Vec<String>,
}

/// Narrow authority grant bound to one proposal/target/operation and time span.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Authorization {
    pub authorization_id: String,
    pub proposal_id: String,
    pub target: AssetId,
    pub operation: String,
    pub issuer: String,
    pub valid_from_unix_ms: u64,
    pub valid_until_unix_ms: u64,
    pub policy_ref: String,
    pub evidence_refs: Vec<String>,
}

impl Authorization {
    pub fn authorizes(&self, proposal: &CommandProposal, now_unix_ms: u64) -> bool {
        self.proposal_id == proposal.proposal_id
            && self.target == proposal.target
            && self.operation == proposal.operation
            && now_unix_ms >= self.valid_from_unix_ms
            && now_unix_ms < self.valid_until_unix_ms
            && !self.issuer.trim().is_empty()
            && !self.policy_ref.trim().is_empty()
    }
}

/// Local-controller result for an authorized operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExecutionDisposition {
    Accepted,
    Rejected,
    Completed,
    Failed,
}

/// Evidence emitted by a local execution boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionReceipt {
    pub receipt_id: String,
    pub authorization_id: String,
    pub target: AssetId,
    pub operation: String,
    pub disposition: ExecutionDisposition,
    pub recorded_at_unix_ms: u64,
    pub evidence_refs: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proposal() -> CommandProposal {
        CommandProposal {
            proposal_id: "proposal-1".into(),
            target: AssetId::new("rover-7"),
            operation: "hold-position".into(),
            rationale: "navigation uncertainty exceeded envelope".into(),
            evidence_refs: vec!["obs-42".into()],
        }
    }

    #[test]
    fn capability_rejects_overcommitted_availability() {
        let capability = ServiceCapability {
            provider: AssetId::new("grid-a"),
            service: "dc-power".into(),
            resource: ResourceKind::Power,
            capacity: 40.0,
            available: 41.0,
            unit: "kW".into(),
            interfaces: vec![],
        };
        assert!(!capability.is_well_formed());
    }

    #[test]
    fn authorization_is_exactly_bound_and_expires() {
        let proposal = proposal();
        let auth = Authorization {
            authorization_id: "auth-1".into(),
            proposal_id: proposal.proposal_id.clone(),
            target: proposal.target.clone(),
            operation: proposal.operation.clone(),
            issuer: "operator-a".into(),
            valid_from_unix_ms: 100,
            valid_until_unix_ms: 200,
            policy_ref: "policy-v1".into(),
            evidence_refs: vec![],
        };

        assert!(!auth.authorizes(&proposal, 99));
        assert!(auth.authorizes(&proposal, 100));
        assert!(auth.authorizes(&proposal, 199));
        assert!(!auth.authorizes(&proposal, 200));

        let mut changed = proposal.clone();
        changed.operation = "drive".into();
        assert!(!auth.authorizes(&changed, 150));
    }

    #[test]
    fn reservation_requires_positive_bounded_window() {
        let reservation = Reservation {
            reservation_id: "r-1".into(),
            requester: AssetId::new("isru-1"),
            capability_provider: AssetId::new("grid-a"),
            resource: ResourceKind::Energy,
            amount: 10.0,
            unit: "kWh".into(),
            valid_from_unix_ms: 10,
            valid_until_unix_ms: 11,
        };
        assert!(reservation.is_well_formed());
    }
}
