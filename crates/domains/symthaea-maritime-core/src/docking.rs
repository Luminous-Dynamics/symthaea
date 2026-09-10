// SPDX-License-Identifier: AGPL-3.0-or-later
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

fn canonical_field(value: &str) -> bool {
    !value.trim().is_empty()
        && value.trim() == value
        && !value.chars().any(char::is_control)
}

/// Observed phase of a resident-vehicle docking sequence.
///
/// The phase is evidence, not authentication. Service permission also requires
/// fresh local mechanical/electrical interlock facts and an independently
/// authenticated client identity at the point of use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DockingPhase {
    Undocked,
    Approaching,
    Captured,
    Aligned,
    Latched,
    ElectricallyIsolated,
    Authenticated,
    ServiceReady,
    Servicing,
    ReleaseQualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DockingObservation {
    pub dock_id: String,
    pub host_platform_id: String,
    pub client_platform_id: String,
    pub interface_schema: String,
    pub phase: DockingPhase,
    pub faulted: bool,
    /// Opaque binding to the observation/interlock evidence behind this state.
    pub evidence_binding: String,
}

impl DockingObservation {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !canonical_field(&self.dock_id)
            || !canonical_field(&self.host_platform_id)
            || !canonical_field(&self.client_platform_id)
            || !canonical_field(&self.interface_schema)
            || !canonical_field(&self.evidence_binding)
        {
            return Err("docking observation contains a malformed canonical field");
        }
        Ok(())
    }
}

/// Generic services a resident maritime platform may negotiate at a compatible dock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MaritimeServiceKind {
    ElectricalPower,
    DataTransfer,
    TimeSynchronization,
    Telemetry,
    Diagnostics,
    ThermalService,
    FluidService,
    RoboticHandling,
}

/// Evidence-bearing service agreement for one client at one dock.
///
/// This is deliberately connector-neutral: a wet-mate connector, inductive
/// charger, optical link or future interface can implement the same upper-level
/// contract without changing maritime-core.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaritimeServiceContract {
    pub contract_id: String,
    pub dock_id: String,
    pub client_platform_id: String,
    pub interface_schema: String,
    pub allowed_services: BTreeSet<MaritimeServiceKind>,
    pub evidence_binding: String,
}

impl MaritimeServiceContract {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !canonical_field(&self.contract_id)
            || !canonical_field(&self.dock_id)
            || !canonical_field(&self.client_platform_id)
            || !canonical_field(&self.interface_schema)
            || !canonical_field(&self.evidence_binding)
        {
            return Err("service contract contains a malformed canonical field");
        }
        if self.allowed_services.is_empty() {
            return Err("service contract must allow at least one service");
        }
        Ok(())
    }
}

/// Fresh local facts used to authorize the start of service.
///
/// This type intentionally has no serde implementation. Portable JSON may
/// describe a docking observation or contract, but raw data must not recreate
/// fresh physical-interlock or authenticated-identity state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceGateContext {
    pub authenticated_client_id: Option<String>,
    pub mechanical_capture_confirmed: bool,
    pub alignment_confirmed: bool,
    pub latch_confirmed: bool,
    pub electrical_isolation_confirmed: bool,
    pub local_interlocks_clear: bool,
    pub authority_permitted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceStartRefusal {
    MalformedObservation,
    MalformedContract,
    BindingMismatch,
    DockFaulted,
    DockNotServiceReady,
    MechanicalCaptureMissing,
    AlignmentMissing,
    LatchMissing,
    ElectricalIsolationMissing,
    IdentityNotAuthenticated,
    LocalInterlockBlocked,
    AuthorityDenied,
    EmptyServiceRequest,
    UnsupportedService,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceStartDecision {
    Permitted,
    Refused(ServiceStartRefusal),
}

/// Evaluate whether negotiated dock service may begin.
///
/// A `ServiceReady` wire observation is never sufficient by itself. Local
/// capture/alignment/latch/isolation facts, authenticated client identity,
/// physical interlocks and independent authority must all agree at point of use.
pub fn evaluate_service_start(
    contract: &MaritimeServiceContract,
    observation: &DockingObservation,
    requested_services: &BTreeSet<MaritimeServiceKind>,
    context: &ServiceGateContext,
) -> ServiceStartDecision {
    if observation.validate().is_err() {
        return ServiceStartDecision::Refused(ServiceStartRefusal::MalformedObservation);
    }
    if contract.validate().is_err() {
        return ServiceStartDecision::Refused(ServiceStartRefusal::MalformedContract);
    }
    if observation.dock_id != contract.dock_id
        || observation.client_platform_id != contract.client_platform_id
        || observation.interface_schema != contract.interface_schema
    {
        return ServiceStartDecision::Refused(ServiceStartRefusal::BindingMismatch);
    }
    if observation.faulted {
        return ServiceStartDecision::Refused(ServiceStartRefusal::DockFaulted);
    }
    if observation.phase != DockingPhase::ServiceReady {
        return ServiceStartDecision::Refused(ServiceStartRefusal::DockNotServiceReady);
    }
    if !context.mechanical_capture_confirmed {
        return ServiceStartDecision::Refused(ServiceStartRefusal::MechanicalCaptureMissing);
    }
    if !context.alignment_confirmed {
        return ServiceStartDecision::Refused(ServiceStartRefusal::AlignmentMissing);
    }
    if !context.latch_confirmed {
        return ServiceStartDecision::Refused(ServiceStartRefusal::LatchMissing);
    }
    if !context.electrical_isolation_confirmed {
        return ServiceStartDecision::Refused(ServiceStartRefusal::ElectricalIsolationMissing);
    }
    if context.authenticated_client_id.as_deref() != Some(contract.client_platform_id.as_str()) {
        return ServiceStartDecision::Refused(ServiceStartRefusal::IdentityNotAuthenticated);
    }
    if !context.local_interlocks_clear {
        return ServiceStartDecision::Refused(ServiceStartRefusal::LocalInterlockBlocked);
    }
    if !context.authority_permitted {
        return ServiceStartDecision::Refused(ServiceStartRefusal::AuthorityDenied);
    }
    if requested_services.is_empty() {
        return ServiceStartDecision::Refused(ServiceStartRefusal::EmptyServiceRequest);
    }
    if !requested_services.is_subset(&contract.allowed_services) {
        return ServiceStartDecision::Refused(ServiceStartRefusal::UnsupportedService);
    }

    ServiceStartDecision::Permitted
}

/// Fresh local facts required before the client may be physically released.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReleaseGateContext {
    pub service_complete: bool,
    pub electrical_service_isolated: bool,
    pub handling_volume_clear: bool,
    pub local_interlocks_clear: bool,
    pub release_evidence_present: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReleaseRefusal {
    MalformedObservation,
    DockFaulted,
    NotReleaseQualified,
    ServiceIncomplete,
    ElectricalServiceStillConnected,
    HandlingVolumeBlocked,
    LocalInterlockBlocked,
    MissingReleaseEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReleaseDecision {
    Permitted,
    Refused(ReleaseRefusal),
}

pub fn evaluate_release(
    observation: &DockingObservation,
    context: ReleaseGateContext,
) -> ReleaseDecision {
    if observation.validate().is_err() {
        return ReleaseDecision::Refused(ReleaseRefusal::MalformedObservation);
    }
    if observation.faulted {
        return ReleaseDecision::Refused(ReleaseRefusal::DockFaulted);
    }
    if observation.phase != DockingPhase::ReleaseQualified {
        return ReleaseDecision::Refused(ReleaseRefusal::NotReleaseQualified);
    }
    if !context.service_complete {
        return ReleaseDecision::Refused(ReleaseRefusal::ServiceIncomplete);
    }
    if !context.electrical_service_isolated {
        return ReleaseDecision::Refused(ReleaseRefusal::ElectricalServiceStillConnected);
    }
    if !context.handling_volume_clear {
        return ReleaseDecision::Refused(ReleaseRefusal::HandlingVolumeBlocked);
    }
    if !context.local_interlocks_clear {
        return ReleaseDecision::Refused(ReleaseRefusal::LocalInterlockBlocked);
    }
    if !context.release_evidence_present {
        return ReleaseDecision::Refused(ReleaseRefusal::MissingReleaseEvidence);
    }

    ReleaseDecision::Permitted
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation() -> DockingObservation {
        DockingObservation {
            dock_id: "dock-a".into(),
            host_platform_id: "tender-1".into(),
            client_platform_id: "auv-7".into(),
            interface_schema: "maritime-service-v1".into(),
            phase: DockingPhase::ServiceReady,
            faulted: false,
            evidence_binding: "evidence:dock-42".into(),
        }
    }

    fn contract() -> MaritimeServiceContract {
        MaritimeServiceContract {
            contract_id: "contract-1".into(),
            dock_id: "dock-a".into(),
            client_platform_id: "auv-7".into(),
            interface_schema: "maritime-service-v1".into(),
            allowed_services: BTreeSet::from([
                MaritimeServiceKind::ElectricalPower,
                MaritimeServiceKind::DataTransfer,
                MaritimeServiceKind::Diagnostics,
            ]),
            evidence_binding: "evidence:contract-9".into(),
        }
    }

    fn gate() -> ServiceGateContext {
        ServiceGateContext {
            authenticated_client_id: Some("auv-7".into()),
            mechanical_capture_confirmed: true,
            alignment_confirmed: true,
            latch_confirmed: true,
            electrical_isolation_confirmed: true,
            local_interlocks_clear: true,
            authority_permitted: true,
        }
    }

    fn requested() -> BTreeSet<MaritimeServiceKind> {
        BTreeSet::from([
            MaritimeServiceKind::ElectricalPower,
            MaritimeServiceKind::DataTransfer,
        ])
    }

    #[test]
    fn fully_qualified_service_can_begin() {
        assert_eq!(
            evaluate_service_start(&contract(), &observation(), &requested(), &gate()),
            ServiceStartDecision::Permitted
        );
    }

    #[test]
    fn service_ready_wire_state_cannot_replace_physical_interlocks() {
        let mut local = gate();
        local.latch_confirmed = false;
        assert_eq!(
            evaluate_service_start(&contract(), &observation(), &requested(), &local),
            ServiceStartDecision::Refused(ServiceStartRefusal::LatchMissing)
        );

        local.latch_confirmed = true;
        local.electrical_isolation_confirmed = false;
        assert_eq!(
            evaluate_service_start(&contract(), &observation(), &requested(), &local),
            ServiceStartDecision::Refused(ServiceStartRefusal::ElectricalIsolationMissing)
        );
    }

    #[test]
    fn exact_authenticated_client_binding_is_required() {
        let mut local = gate();
        local.authenticated_client_id = Some("auv-8".into());
        assert_eq!(
            evaluate_service_start(&contract(), &observation(), &requested(), &local),
            ServiceStartDecision::Refused(ServiceStartRefusal::IdentityNotAuthenticated)
        );
    }

    #[test]
    fn contract_observation_cross_binding_is_required() {
        let mut obs = observation();
        obs.dock_id = "dock-b".into();
        assert_eq!(
            evaluate_service_start(&contract(), &obs, &requested(), &gate()),
            ServiceStartDecision::Refused(ServiceStartRefusal::BindingMismatch)
        );
    }

    #[test]
    fn unsupported_or_empty_service_request_fails_closed() {
        assert_eq!(
            evaluate_service_start(&contract(), &observation(), &BTreeSet::new(), &gate()),
            ServiceStartDecision::Refused(ServiceStartRefusal::EmptyServiceRequest)
        );

        let unsupported = BTreeSet::from([MaritimeServiceKind::FluidService]);
        assert_eq!(
            evaluate_service_start(&contract(), &observation(), &unsupported, &gate()),
            ServiceStartDecision::Refused(ServiceStartRefusal::UnsupportedService)
        );
    }

    #[test]
    fn faulted_dock_never_starts_service() {
        let mut obs = observation();
        obs.faulted = true;
        assert_eq!(
            evaluate_service_start(&contract(), &obs, &requested(), &gate()),
            ServiceStartDecision::Refused(ServiceStartRefusal::DockFaulted)
        );
    }

    #[test]
    fn release_requires_independent_completion_and_isolation_facts() {
        let mut obs = observation();
        obs.phase = DockingPhase::ReleaseQualified;
        let good = ReleaseGateContext {
            service_complete: true,
            electrical_service_isolated: true,
            handling_volume_clear: true,
            local_interlocks_clear: true,
            release_evidence_present: true,
        };
        assert_eq!(evaluate_release(&obs, good), ReleaseDecision::Permitted);

        let still_connected = ReleaseGateContext {
            electrical_service_isolated: false,
            ..good
        };
        assert_eq!(
            evaluate_release(&obs, still_connected),
            ReleaseDecision::Refused(ReleaseRefusal::ElectricalServiceStillConnected)
        );
    }
}
