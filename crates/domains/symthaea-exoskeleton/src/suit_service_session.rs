// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Thin composition layer for SX-019 service sessions.
//!
//! This module derives resource-transfer permissions from individually admitted
//! service capabilities and derives suitport decontamination readiness from the
//! existing dust twin. It does not open pressure boundaries or command transfer
//! hardware.

use serde::{Deserialize, Serialize};

use crate::dust::{DustDegradationTwin, DustZone};
use crate::resource_transfer::{ResourceTransferPermissions, ResourceTransferRequest};
use crate::space_exosuit::ExosuitEvidenceLevel;
use crate::suit_service_interface::{
    admit_service, ServiceAdmissionDenial, SuitServiceKind, SuitServiceNodeAdvertisement,
    SuitServicePolicy,
};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SuitServiceSessionPolicy {
    pub service_policy: SuitServicePolicy,
    /// Maximum retained contamination permitted at the external suitport
    /// interface before the nominal state machine may consider decon complete.
    pub max_suitport_dust_g_m2: f64,
    /// Maximum suitport-interface seal-risk proxy [0,1].
    pub max_suitport_seal_risk: f64,
    pub evidence: ExosuitEvidenceLevel,
}

impl SuitServiceSessionPolicy {
    pub fn simulation_reference() -> Self {
        Self {
            service_policy: SuitServicePolicy::simulation_reference(),
            max_suitport_dust_g_m2: 0.5,
            max_suitport_seal_risk: 0.05,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.service_policy.is_valid()
            && self.max_suitport_dust_g_m2.is_finite()
            && self.max_suitport_dust_g_m2 >= 0.0
            && self.max_suitport_seal_risk.is_finite()
            && (0.0..=1.0).contains(&self.max_suitport_seal_risk)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceNegotiationFailure {
    pub kind: SuitServiceKind,
    pub denial: ServiceAdmissionDenial,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServiceNegotiationResult {
    pub permissions: ResourceTransferPermissions,
    pub failures: Vec<ServiceNegotiationFailure>,
    pub request_valid: bool,
    pub node_valid: bool,
    pub policy_valid: bool,
    pub all_required_services_admitted: bool,
    pub evidence: ExosuitEvidenceLevel,
}

pub fn negotiate_resource_services(
    node: &SuitServiceNodeAdvertisement,
    request: &ResourceTransferRequest,
    policy: SuitServicePolicy,
) -> ServiceNegotiationResult {
    let mut permissions = ResourceTransferPermissions::none();
    let mut failures = Vec::new();
    let request_valid = request.is_valid();
    let node_valid = node.is_valid();
    let policy_valid = policy.is_valid();

    if request_valid && node_valid && policy_valid {
        let needs_charge = requested(&request.electrical_energy_wh);
        let needs_oxygen =
            requested(&request.primary_oxygen_l) || requested(&request.secondary_oxygen_l);
        let needs_coolant = requested(&request.thermal_service_j);
        let needs_regeneration = requested(&request.co2_regeneration_l)
            || requested(&request.humidity_regeneration);

        admit_if_required(
            node,
            policy,
            needs_charge,
            SuitServiceKind::ElectricalCharge,
            &mut permissions.electrical_charge,
            &mut failures,
        );
        admit_if_required(
            node,
            policy,
            needs_oxygen,
            SuitServiceKind::OxygenReplenish,
            &mut permissions.oxygen_replenish,
            &mut failures,
        );
        admit_if_required(
            node,
            policy,
            needs_coolant,
            SuitServiceKind::CoolantService,
            &mut permissions.coolant_service,
            &mut failures,
        );
        admit_if_required(
            node,
            policy,
            needs_regeneration,
            SuitServiceKind::PlssRegeneration,
            &mut permissions.plss_regeneration,
            &mut failures,
        );
    }

    ServiceNegotiationResult {
        permissions,
        request_valid,
        node_valid,
        policy_valid,
        all_required_services_admitted: request_valid
            && node_valid
            && policy_valid
            && failures.is_empty(),
        failures,
        evidence: node.evidence.min(policy.evidence),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SuitportDeconAssessment {
    pub ready: bool,
    pub retained_dust_g_m2: f64,
    pub seal_risk: f64,
    pub evidence: ExosuitEvidenceLevel,
}

pub fn assess_suitport_decon(
    dust: &DustDegradationTwin,
    policy: SuitServiceSessionPolicy,
) -> SuitportDeconAssessment {
    if !policy.is_valid() {
        return SuitportDeconAssessment {
            ready: false,
            retained_dust_g_m2: f64::NAN,
            seal_risk: f64::NAN,
            evidence: policy.evidence,
        };
    }

    let state = dust.state(DustZone::SuitportInterface);
    let finite = state.retained_dust_g_m2.is_finite() && state.seal_risk.is_finite();
    let ready = finite
        && state.retained_dust_g_m2 <= policy.max_suitport_dust_g_m2
        && state.seal_risk <= policy.max_suitport_seal_risk;

    SuitportDeconAssessment {
        ready,
        retained_dust_g_m2: state.retained_dust_g_m2,
        seal_risk: state.seal_risk,
        evidence: policy.evidence,
    }
}

fn requested(amount: &crate::resource_transfer::MeteredAmount) -> bool {
    amount.source_amount > 0.0 || amount.suit_amount > 0.0
}

fn admit_if_required(
    node: &SuitServiceNodeAdvertisement,
    policy: SuitServicePolicy,
    required: bool,
    kind: SuitServiceKind,
    permission: &mut bool,
    failures: &mut Vec<ServiceNegotiationFailure>,
) {
    if !required {
        return;
    }
    match admit_service(node, kind, policy) {
        Ok(_) => *permission = true,
        Err(denial) => failures.push(ServiceNegotiationFailure { kind, denial }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dust::{DustExposure, DUST_ZONE_COUNT};
    use crate::resource_transfer::MeteredAmount;
    use crate::suit_service_interface::{SuitServiceCapability, SuitServiceNodeKind};

    fn metered(value: f64) -> MeteredAmount {
        MeteredAmount {
            source_amount: value,
            suit_amount: value,
            max_relative_disagreement: 0.02,
        }
    }

    fn request() -> ResourceTransferRequest {
        ResourceTransferRequest {
            duration_s: 60.0,
            electrical_energy_wh: metered(10.0),
            primary_oxygen_l: metered(20.0),
            secondary_oxygen_l: MeteredAmount::zero(),
            thermal_service_j: MeteredAmount::zero(),
            co2_regeneration_l: MeteredAmount::zero(),
            humidity_regeneration: MeteredAmount::zero(),
            completion_fraction: 1.0,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    fn node() -> SuitServiceNodeAdvertisement {
        SuitServiceNodeAdvertisement {
            node_id: "hab-1".into(),
            kind: SuitServiceNodeKind::Habitat,
            identity_verified: true,
            identity_confidence: 1.0,
            observation_age_s: 1.0,
            capabilities: vec![
                SuitServiceCapability {
                    kind: SuitServiceKind::ElectricalCharge,
                    available: true,
                    max_rate: 1_000.0,
                    confidence: 0.99,
                    evidence: ExosuitEvidenceLevel::Simulation,
                },
                SuitServiceCapability {
                    kind: SuitServiceKind::OxygenReplenish,
                    available: true,
                    max_rate: 100.0,
                    confidence: 0.99,
                    evidence: ExosuitEvidenceLevel::Simulation,
                },
            ],
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    #[test]
    fn only_requested_services_must_be_admitted() {
        let result = negotiate_resource_services(
            &node(),
            &request(),
            SuitServicePolicy::simulation_reference(),
        );
        assert!(result.all_required_services_admitted);
        assert!(result.request_valid);
        assert!(result.node_valid);
        assert!(result.policy_valid);
        assert!(result.permissions.electrical_charge);
        assert!(result.permissions.oxygen_replenish);
        assert!(!result.permissions.coolant_service);
        assert!(!result.permissions.plss_regeneration);
    }

    #[test]
    fn malformed_request_cannot_pass_utility_negotiation() {
        let mut malformed = request();
        malformed.duration_s = f64::NAN;
        let result = negotiate_resource_services(
            &node(),
            &malformed,
            SuitServicePolicy::simulation_reference(),
        );
        assert!(!result.request_valid);
        assert!(!result.all_required_services_admitted);
        assert_eq!(result.permissions, ResourceTransferPermissions::none());
    }

    #[test]
    fn missing_required_service_is_explicit_failure() {
        let mut node = node();
        node.capabilities
            .retain(|capability| capability.kind != SuitServiceKind::OxygenReplenish);
        let result = negotiate_resource_services(
            &node,
            &request(),
            SuitServicePolicy::simulation_reference(),
        );
        assert!(!result.all_required_services_admitted);
        assert!(!result.permissions.oxygen_replenish);
        assert_eq!(result.failures.len(), 1);
    }

    #[test]
    fn dirty_suitport_interface_does_not_self_certify_clean() {
        let mut dust = DustDegradationTwin::simulation_reference();
        let mut exposure = DustExposure::uniform(0.0, 60.0);
        exposure.deposition_g_m2 = [0.0; DUST_ZONE_COUNT];
        exposure.deposition_g_m2[DustZone::SuitportInterface.index()] = 10.0;
        dust.config_mut_for_trade_study().eds[DustZone::SuitportInterface.index()].enabled = false;
        dust.step(exposure).unwrap();
        let assessment = assess_suitport_decon(
            &dust,
            SuitServiceSessionPolicy::simulation_reference(),
        );
        assert!(!assessment.ready);
    }
}
