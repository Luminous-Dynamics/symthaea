// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed rover/habitat/suitport service advertisement for SX-019.
//!
//! A service node advertises individual capabilities. No capability is implied
//! by another one, and discovering a node never grants pressure-boundary or
//! actuator authority. The types are intentionally transport-agnostic so local
//! offline docking can use the same contract as a network-discovered node.

use serde::{Deserialize, Serialize};

use crate::space_exosuit::ExosuitEvidenceLevel;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SuitServiceKind {
    MechanicalCapture,
    PressureBoundary,
    ElectricalCharge,
    DataDiagnostics,
    OxygenReplenish,
    CoolantService,
    PlssRegeneration,
    DustDecontamination,
    ToolAndSpareInventory,
    EmergencyShelter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuitServiceNodeKind {
    Habitat,
    PressurizedRover,
    UnpressurizedSupportRover,
    FixedSuitport,
    MobileServiceNode,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SuitServiceCapability {
    pub kind: SuitServiceKind,
    pub available: bool,
    /// Generic declared throughput. Units are defined by `kind` and the
    /// associated transfer request, not inferred by this descriptor.
    pub max_rate: f64,
    /// Confidence in the latest local capability observation [0,1].
    pub confidence: f64,
    pub evidence: ExosuitEvidenceLevel,
}

impl SuitServiceCapability {
    pub fn is_valid(&self) -> bool {
        self.max_rate.is_finite()
            && self.max_rate >= 0.0
            && self.confidence.is_finite()
            && (0.0..=1.0).contains(&self.confidence)
            && (!self.available || self.max_rate > 0.0 || matches!(
                self.kind,
                SuitServiceKind::MechanicalCapture
                    | SuitServiceKind::PressureBoundary
                    | SuitServiceKind::DataDiagnostics
                    | SuitServiceKind::EmergencyShelter
            ))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SuitServiceNodeAdvertisement {
    pub node_id: String,
    pub kind: SuitServiceNodeKind,
    /// Local identity/authentication result. Discovery alone is insufficient.
    pub identity_verified: bool,
    /// Age of the local advertisement/observation, seconds.
    pub observation_age_s: f64,
    pub capabilities: Vec<SuitServiceCapability>,
    pub evidence: ExosuitEvidenceLevel,
}

impl SuitServiceNodeAdvertisement {
    pub fn is_valid(&self) -> bool {
        !self.node_id.trim().is_empty()
            && self.observation_age_s.is_finite()
            && self.observation_age_s >= 0.0
            && self.capabilities.iter().all(SuitServiceCapability::is_valid)
            && !has_duplicate_capability(&self.capabilities)
    }

    pub fn capability(&self, kind: SuitServiceKind) -> Option<&SuitServiceCapability> {
        self.capabilities.iter().find(|capability| capability.kind == kind)
    }

    pub fn advertises_available(&self, kind: SuitServiceKind) -> bool {
        self.capability(kind).is_some_and(|capability| capability.available)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SuitServicePolicy {
    pub max_observation_age_s: f64,
    pub min_identity_confidence: f64,
    pub min_capability_confidence: f64,
    pub evidence: ExosuitEvidenceLevel,
}

impl SuitServicePolicy {
    pub fn simulation_reference() -> Self {
        Self {
            max_observation_age_s: 10.0,
            min_identity_confidence: 1.0,
            min_capability_confidence: 0.90,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.max_observation_age_s.is_finite()
            && self.max_observation_age_s >= 0.0
            && self.min_identity_confidence.is_finite()
            && (0.0..=1.0).contains(&self.min_identity_confidence)
            && self.min_capability_confidence.is_finite()
            && (0.0..=1.0).contains(&self.min_capability_confidence)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceAdmissionDenial {
    InvalidNode,
    InvalidPolicy,
    IdentityUnverified,
    StaleAdvertisement,
    CapabilityMissing,
    CapabilityUnavailable,
    CapabilityConfidenceLow,
}

pub fn admit_service(
    node: &SuitServiceNodeAdvertisement,
    kind: SuitServiceKind,
    policy: SuitServicePolicy,
) -> Result<SuitServiceCapability, ServiceAdmissionDenial> {
    if !node.is_valid() {
        return Err(ServiceAdmissionDenial::InvalidNode);
    }
    if !policy.is_valid() {
        return Err(ServiceAdmissionDenial::InvalidPolicy);
    }
    if !node.identity_verified {
        return Err(ServiceAdmissionDenial::IdentityUnverified);
    }
    if node.observation_age_s > policy.max_observation_age_s {
        return Err(ServiceAdmissionDenial::StaleAdvertisement);
    }
    let capability = node
        .capability(kind)
        .ok_or(ServiceAdmissionDenial::CapabilityMissing)?;
    if !capability.available {
        return Err(ServiceAdmissionDenial::CapabilityUnavailable);
    }
    if capability.confidence < policy.min_capability_confidence {
        return Err(ServiceAdmissionDenial::CapabilityConfidenceLow);
    }
    Ok(capability.clone())
}

fn has_duplicate_capability(capabilities: &[SuitServiceCapability]) -> bool {
    for (index, capability) in capabilities.iter().enumerate() {
        if capabilities[index + 1..]
            .iter()
            .any(|other| other.kind == capability.kind)
        {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node() -> SuitServiceNodeAdvertisement {
        SuitServiceNodeAdvertisement {
            node_id: "rover-7".into(),
            kind: SuitServiceNodeKind::PressurizedRover,
            identity_verified: true,
            observation_age_s: 1.0,
            capabilities: vec![SuitServiceCapability {
                kind: SuitServiceKind::ElectricalCharge,
                available: true,
                max_rate: 800.0,
                confidence: 0.99,
                evidence: ExosuitEvidenceLevel::Simulation,
            }],
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    #[test]
    fn capabilities_are_individually_admitted() {
        let node = node();
        assert!(admit_service(
            &node,
            SuitServiceKind::ElectricalCharge,
            SuitServicePolicy::simulation_reference()
        )
        .is_ok());
        assert_eq!(
            admit_service(
                &node,
                SuitServiceKind::OxygenReplenish,
                SuitServicePolicy::simulation_reference()
            ),
            Err(ServiceAdmissionDenial::CapabilityMissing)
        );
    }

    #[test]
    fn stale_or_unverified_nodes_fail_closed() {
        let mut stale = node();
        stale.observation_age_s = 60.0;
        assert_eq!(
            admit_service(
                &stale,
                SuitServiceKind::ElectricalCharge,
                SuitServicePolicy::simulation_reference()
            ),
            Err(ServiceAdmissionDenial::StaleAdvertisement)
        );

        let mut unverified = node();
        unverified.identity_verified = false;
        assert_eq!(
            admit_service(
                &unverified,
                SuitServiceKind::ElectricalCharge,
                SuitServicePolicy::simulation_reference()
            ),
            Err(ServiceAdmissionDenial::IdentityUnverified)
        );
    }
}
