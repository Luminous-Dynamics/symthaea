// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Domain-neutral infrastructure coordination vocabulary.
//!
//! This crate models assets, resources, services, evidence-bearing interface
//! claims, reservations, proposed actions, narrow authorizations, and execution
//! receipts. It deliberately contains no flight-control, robotics, networking,
//! physics, or hardware drivers.
//!
//! # Authority boundary
//!
//! A [`CommandProposal`] is descriptive and non-executable. An
//! [`Authorization`] is narrowly bound to a proposal, target, operation, and
//! validity window, but a local controller still retains final command
//! acceptance and immediate physical safety authority.

#![deny(unsafe_code)]

pub mod transport;

use serde::{Deserialize, Serialize};

/// Stable identifier for a physical or virtual infrastructure asset.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AssetId(pub String);

impl AssetId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn is_well_formed(&self) -> bool {
        !self.0.trim().is_empty()
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

/// Strength of evidence behind an interface/conformance claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvidenceStatus {
    /// An implementer has declared compatibility; no independent evidence.
    Declared,
    /// Compatibility has been exercised in a model/simulation only.
    Modeled,
    /// Compatibility has been exercised by a relevant test.
    Tested,
    /// Compatibility has passed the separately defined qualification process.
    Qualified,
}

/// Reference to an internal or external interoperability contract.
///
/// Matching `standard`/`profile` strings do not by themselves establish
/// compatibility. `evidence_status` records the evidence class supporting the
/// claim; detailed evidence remains outside this leaf crate.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InterfaceRef {
    pub standard: String,
    pub profile: Option<String>,
    pub evidence_status: EvidenceStatus,
}

impl InterfaceRef {
    pub fn is_well_formed(&self) -> bool {
        !self.standard.trim().is_empty()
            && self
                .profile
                .as_ref()
                .is_none_or(|profile| !profile.trim().is_empty())
    }
}

/// A bounded service offered by an infrastructure asset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServiceCapability {
    pub provider: AssetId,
    pub service: String,
    pub resource: ResourceKind,
    pub capacity: f64,
    pub available: f64,
    /// Explicit engineering unit label for the scalar values above.
    ///
    /// This v0.1 leaf crate intentionally does not choose a units library for
    /// every domain. Adapters should normalize/validate units before execution.
    pub unit: String,
    pub interfaces: Vec<InterfaceRef>,
}

impl ServiceCapability {
    pub fn is_well_formed(&self) -> bool {
        self.provider.is_well_formed()
            && self.capacity.is_finite()
            && self.available.is_finite()
            && self.capacity >= 0.0
            && self.available >= 0.0
            && self.available <= self.capacity
            && !self.service.trim().is_empty()
            && !self.unit.trim().is_empty()
            && self.interfaces.iter().all(InterfaceRef::is_well_formed)
    }
}

/// Provenance for a measured or derived observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationSource {
    pub source_id: String,
    pub method: String,
}

/// Time-bound observation with explicit non-negative uncertainty.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InfrastructureObservation {
    pub observation_id: String,
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
        !self.observation_id.trim().is_empty()
            && self.asset.is_well_formed()
            && self.value.is_finite()
            && self.uncertainty.is_finite()
            && self.uncertainty >= 0.0
            && !self.quantity.trim().is_empty()
            && !self.unit.trim().is_empty()
            && !self.source.source_id.trim().is_empty()
            && !self.source.method.trim().is_empty()
    }
}

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

impl Hazard {
    pub fn is_well_formed(&self) -> bool {
        self.asset.is_well_formed()
            && !self.hazard_id.trim().is_empty()
            && !self.description.trim().is_empty()
    }
}

/// Temporary resource allocation. This never implies actuator authority.
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
        !self.reservation_id.trim().is_empty()
            && self.requester.is_well_formed()
            && self.capability_provider.is_well_formed()
            && self.amount.is_finite()
            && self.amount > 0.0
            && self.valid_until_unix_ms > self.valid_from_unix_ms
            && !self.unit.trim().is_empty()
    }

    pub fn is_active_at(&self, now_unix_ms: u64) -> bool {
        now_unix_ms >= self.valid_from_unix_ms && now_unix_ms < self.valid_until_unix_ms
    }
}

/// Descriptive action proposal. Deliberately contains no executable callback,
/// transport, device handle, or hardware command object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandProposal {
    pub proposal_id: String,
    pub target: AssetId,
    pub operation: String,
    pub rationale: String,
    pub evidence_refs: Vec<String>,
}

impl CommandProposal {
    pub fn is_well_formed(&self) -> bool {
        !self.proposal_id.trim().is_empty()
            && self.target.is_well_formed()
            && !self.operation.trim().is_empty()
            && !self.rationale.trim().is_empty()
    }
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
    pub fn is_well_formed(&self) -> bool {
        !self.authorization_id.trim().is_empty()
            && !self.proposal_id.trim().is_empty()
            && self.target.is_well_formed()
            && !self.operation.trim().is_empty()
            && !self.issuer.trim().is_empty()
            && self.valid_until_unix_ms > self.valid_from_unix_ms
            && !self.policy_ref.trim().is_empty()
    }

    /// Exact proposal binding with a half-open validity interval `[from, until)`.
    pub fn authorizes(&self, proposal: &CommandProposal, now_unix_ms: u64) -> bool {
        self.is_well_formed()
            && proposal.is_well_formed()
            && self.proposal_id == proposal.proposal_id
            && self.target == proposal.target
            && self.operation == proposal.operation
            && now_unix_ms >= self.valid_from_unix_ms
            && now_unix_ms < self.valid_until_unix_ms
    }
}

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

impl ExecutionReceipt {
    pub fn is_well_formed(&self) -> bool {
        !self.receipt_id.trim().is_empty()
            && !self.authorization_id.trim().is_empty()
            && self.target.is_well_formed()
            && !self.operation.trim().is_empty()
    }
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
    fn malformed_interface_claim_is_rejected() {
        let interface = InterfaceRef {
            standard: " ".into(),
            profile: None,
            evidence_status: EvidenceStatus::Declared,
        };
        assert!(!interface.is_well_formed());
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
        assert!(reservation.is_active_at(10));
        assert!(!reservation.is_active_at(11));
    }

    #[test]
    fn non_finite_measurements_fail_closed() {
        let observation = InfrastructureObservation {
            observation_id: "obs-1".into(),
            asset: AssetId::new("tank-1"),
            quantity: "pressure".into(),
            value: f64::NAN,
            unit: "Pa".into(),
            uncertainty: 1.0,
            observed_at_unix_ms: 10,
            source: ObservationSource {
                source_id: "sensor-1".into(),
                method: "transducer".into(),
            },
        };
        assert!(!observation.is_well_formed());
    }
}
