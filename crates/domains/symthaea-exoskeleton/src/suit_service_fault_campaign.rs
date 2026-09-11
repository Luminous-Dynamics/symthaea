// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic failure campaign for SX-019 suit servicing.
//!
//! These cases exercise the actual service-admission, suitport-state,
//! transactional-transfer, and safe-haven primitives. The campaign is
//! simulation evidence only; it exists to prove software containment
//! properties before any hardware interface is considered.

use serde::{Deserialize, Serialize};

use crate::plss::PlssReferenceTwin;
use crate::power::MultiBusPowerSystem;
use crate::resource_transfer::{
    apply_resource_transfer, MeteredAmount, ResourceTransferPermissions, ResourceTransferRequest,
};
use crate::safe_haven::{
    recommend_safe_haven, SafeHavenCandidate, SafeHavenKind, SafeHavenPolicy,
    SafeHavenRecommendation, SafeHavenSituation,
};
use crate::space_exosuit::ExosuitEvidenceLevel;
use crate::suit_service_interface::{
    SuitServiceCapability, SuitServiceKind, SuitServiceNodeAdvertisement, SuitServiceNodeKind,
    SuitServicePolicy,
};
use crate::suit_service_session::negotiate_resource_services;
use crate::suitport_state_machine::{SuitportEvidence, SuitportStateMachine};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuitServiceFaultCase {
    StaleServiceIdentity,
    LowIdentityConfidence,
    ServiceNodePowerLoss,
    CaptureLoss,
    SealDisagreement,
    PressureDisagreement,
    DecontaminationUnavailable,
    MeterDisagreement,
    InterruptedTransfer,
    SafeHavenUnavailable,
    CommunicationBlackout,
}

impl SuitServiceFaultCase {
    pub const ALL: [Self; 11] = [
        Self::StaleServiceIdentity,
        Self::LowIdentityConfidence,
        Self::ServiceNodePowerLoss,
        Self::CaptureLoss,
        Self::SealDisagreement,
        Self::PressureDisagreement,
        Self::DecontaminationUnavailable,
        Self::MeterDisagreement,
        Self::InterruptedTransfer,
        Self::SafeHavenUnavailable,
        Self::CommunicationBlackout,
    ];
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SuitServiceFaultOutcome {
    pub case: SuitServiceFaultCase,
    pub service_admitted: Option<bool>,
    pub pressure_boundary_permitted: Option<bool>,
    pub transfer_committed: Option<bool>,
    pub protected_survival_preserved: Option<bool>,
    pub safe_haven_recommendation: Option<SafeHavenRecommendation>,
    pub offline_local_operation_possible: Option<bool>,
    pub evidence: ExosuitEvidenceLevel,
}

pub fn run_service_fault_case(case: SuitServiceFaultCase) -> SuitServiceFaultOutcome {
    match case {
        SuitServiceFaultCase::StaleServiceIdentity => {
            let mut node = nominal_node();
            node.observation_age_s = 60.0;
            let result = negotiate_resource_services(
                &node,
                &nominal_request(),
                SuitServicePolicy::simulation_reference(),
            );
            base(case).with_service_admitted(result.all_required_services_admitted)
        }
        SuitServiceFaultCase::LowIdentityConfidence => {
            let mut node = nominal_node();
            node.identity_confidence = 0.5;
            let result = negotiate_resource_services(
                &node,
                &nominal_request(),
                SuitServicePolicy::simulation_reference(),
            );
            base(case).with_service_admitted(result.all_required_services_admitted)
        }
        SuitServiceFaultCase::ServiceNodePowerLoss => {
            let mut node = nominal_node();
            if let Some(capability) = node
                .capabilities
                .iter_mut()
                .find(|capability| capability.kind == SuitServiceKind::ElectricalCharge)
            {
                capability.available = false;
            }
            let result = negotiate_resource_services(
                &node,
                &nominal_request(),
                SuitServicePolicy::simulation_reference(),
            );
            base(case).with_service_admitted(result.all_required_services_admitted)
        }
        SuitServiceFaultCase::CaptureLoss => {
            let mut machine = SuitportStateMachine::simulation_reference();
            let nominal = SuitportEvidence::simulation_nominal();
            machine.advance(nominal);
            let mut failed = nominal;
            failed.mechanical_capture_locked = false;
            let result = machine.advance(failed);
            base(case).with_pressure_boundary(result.pressure_boundary_open_permitted)
        }
        SuitServiceFaultCase::SealDisagreement => {
            let mut machine = SuitportStateMachine::simulation_reference();
            let nominal = SuitportEvidence::simulation_nominal();
            machine.advance(nominal);
            let mut failed = nominal;
            failed.inner_seal_verified = false;
            let result = machine.advance(failed);
            base(case).with_pressure_boundary(result.pressure_boundary_open_permitted)
        }
        SuitServiceFaultCase::PressureDisagreement => {
            let mut machine = SuitportStateMachine::simulation_reference();
            let nominal = SuitportEvidence::simulation_nominal();
            machine.advance(nominal);
            machine.advance(nominal);
            let mut failed = nominal;
            failed.pressure_sensors_agree = false;
            let result = machine.advance(failed);
            base(case).with_pressure_boundary(result.pressure_boundary_open_permitted)
        }
        SuitServiceFaultCase::DecontaminationUnavailable => {
            let mut machine = SuitportStateMachine::simulation_reference();
            let nominal = SuitportEvidence::simulation_nominal();
            for _ in 0..4 {
                machine.advance(nominal);
            }
            let mut failed = nominal;
            failed.dust_decontamination_complete = false;
            let result = machine.advance(failed);
            base(case).with_pressure_boundary(result.pressure_boundary_open_permitted)
        }
        SuitServiceFaultCase::MeterDisagreement => {
            let mut power = depleted_power();
            let mut plss = depleted_plss();
            let before_power = power.config().survival.energy_wh;
            let before_o2 = plss.state().primary_o2_remaining_l;
            let mut request = nominal_request();
            request.primary_oxygen_l.source_amount = 100.0;
            request.primary_oxygen_l.suit_amount = 50.0;
            let result = apply_resource_transfer(
                &mut power,
                &mut plss,
                request,
                nominal_permissions(),
            );
            base(case)
                .with_transfer_committed(result.is_ok())
                .with_survival_preserved(
                    (power.config().survival.energy_wh - before_power).abs() < 1e-9
                        && (plss.state().primary_o2_remaining_l - before_o2).abs() < 1e-9,
                )
        }
        SuitServiceFaultCase::InterruptedTransfer => {
            let mut power = depleted_power();
            let mut plss = depleted_plss();
            let before = power.config().survival.energy_wh;
            let mut request = nominal_request();
            request.completion_fraction = 0.25;
            let receipt = apply_resource_transfer(
                &mut power,
                &mut plss,
                request,
                nominal_permissions(),
            );
            let (committed, preserved) = match receipt {
                Ok(receipt) => (
                    !receipt.transfer_complete && receipt.primary_oxygen_accepted_l > 0.0,
                    receipt.survival_energy_after_wh + 1e-9 >= before,
                ),
                Err(_) => (false, false),
            };
            base(case)
                .with_transfer_committed(committed)
                .with_survival_preserved(preserved)
        }
        SuitServiceFaultCase::SafeHavenUnavailable => {
            let mut situation = nominal_situation();
            situation.energetic_particle_alert = true;
            situation.base_is_radiation_shelter = false;
            situation.independent_suit_endurance_min = 30.0;
            situation.required_reserve_min = 10.0;
            let unavailable = SafeHavenCandidate {
                node_id: "rover-offline".into(),
                kind: SafeHavenKind::PressurizedRover,
                available: false,
                identity_verified: true,
                radiation_shelter: true,
                travel_time_upper_min: 5.0,
                travel_time_nominal_min: 4.0,
                route_confidence: 0.99,
                observation_age_s: 1.0,
                evidence: ExosuitEvidenceLevel::Simulation,
            };
            let decision = recommend_safe_haven(
                situation,
                &[unavailable],
                SafeHavenPolicy::simulation_reference(),
            );
            base(case).with_safe_haven(decision.recommendation)
        }
        SuitServiceFaultCase::CommunicationBlackout => {
            let mut situation = nominal_situation();
            situation.remaining_work_upper_min = 90.0;
            let local = SafeHavenCandidate {
                node_id: "local-rover".into(),
                kind: SafeHavenKind::PressurizedRover,
                available: true,
                identity_verified: true,
                radiation_shelter: true,
                travel_time_upper_min: 10.0,
                travel_time_nominal_min: 8.0,
                route_confidence: 0.99,
                observation_age_s: 2.0,
                evidence: ExosuitEvidenceLevel::Simulation,
            };
            let decision = recommend_safe_haven(
                situation,
                &[local],
                SafeHavenPolicy::simulation_reference(),
            );
            base(case)
                .with_safe_haven(decision.recommendation)
                .with_offline_local_operation(!decision.communication_required)
        }
    }
}

impl SuitServiceFaultOutcome {
    fn with_service_admitted(mut self, value: bool) -> Self {
        self.service_admitted = Some(value);
        self
    }

    fn with_pressure_boundary(mut self, value: bool) -> Self {
        self.pressure_boundary_permitted = Some(value);
        self
    }

    fn with_transfer_committed(mut self, value: bool) -> Self {
        self.transfer_committed = Some(value);
        self
    }

    fn with_survival_preserved(mut self, value: bool) -> Self {
        self.protected_survival_preserved = Some(value);
        self
    }

    fn with_safe_haven(mut self, value: SafeHavenRecommendation) -> Self {
        self.safe_haven_recommendation = Some(value);
        self
    }

    fn with_offline_local_operation(mut self, value: bool) -> Self {
        self.offline_local_operation_possible = Some(value);
        self
    }
}

fn base(case: SuitServiceFaultCase) -> SuitServiceFaultOutcome {
    SuitServiceFaultOutcome {
        case,
        service_admitted: None,
        pressure_boundary_permitted: None,
        transfer_committed: None,
        protected_survival_preserved: None,
        safe_haven_recommendation: None,
        offline_local_operation_possible: None,
        evidence: ExosuitEvidenceLevel::Simulation,
    }
}

fn metered(value: f64) -> MeteredAmount {
    MeteredAmount {
        source_amount: value,
        suit_amount: value,
        max_relative_disagreement: 0.02,
    }
}

fn nominal_request() -> ResourceTransferRequest {
    ResourceTransferRequest {
        duration_s: 600.0,
        electrical_energy_wh: metered(100.0),
        primary_oxygen_l: metered(100.0),
        secondary_oxygen_l: MeteredAmount::zero(),
        thermal_service_j: MeteredAmount::zero(),
        co2_regeneration_l: MeteredAmount::zero(),
        humidity_regeneration: MeteredAmount::zero(),
        completion_fraction: 1.0,
        evidence: ExosuitEvidenceLevel::Simulation,
    }
}

fn nominal_permissions() -> ResourceTransferPermissions {
    ResourceTransferPermissions {
        electrical_charge: true,
        oxygen_replenish: true,
        coolant_service: true,
        plss_regeneration: true,
    }
}

fn nominal_node() -> SuitServiceNodeAdvertisement {
    SuitServiceNodeAdvertisement {
        node_id: "service-rover".into(),
        kind: SuitServiceNodeKind::PressurizedRover,
        identity_verified: true,
        identity_confidence: 1.0,
        observation_age_s: 1.0,
        capabilities: vec![
            SuitServiceCapability {
                kind: SuitServiceKind::ElectricalCharge,
                available: true,
                max_rate: 1_000.0,
                confidence: 1.0,
                evidence: ExosuitEvidenceLevel::Simulation,
            },
            SuitServiceCapability {
                kind: SuitServiceKind::OxygenReplenish,
                available: true,
                max_rate: 100.0,
                confidence: 1.0,
                evidence: ExosuitEvidenceLevel::Simulation,
            },
        ],
        evidence: ExosuitEvidenceLevel::Simulation,
    }
}

fn depleted_power() -> MultiBusPowerSystem {
    let mut power = MultiBusPowerSystem::simulation_reference();
    power.config_mut_for_fault_injection().survival.energy_wh = 500.0;
    power
}

fn depleted_plss() -> PlssReferenceTwin {
    let mut plss = PlssReferenceTwin::simulation_reference();
    plss.state_mut_for_fault_injection().primary_o2_remaining_l = 1_000.0;
    plss
}

fn nominal_situation() -> SafeHavenSituation {
    SafeHavenSituation {
        independent_suit_endurance_min: 120.0,
        required_reserve_min: 20.0,
        return_to_base_upper_min: 40.0,
        remaining_work_upper_min: 30.0,
        base_is_radiation_shelter: true,
        energetic_particle_alert: false,
        evidence: ExosuitEvidenceLevel::Simulation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_declared_fault_cases_execute() {
        for case in SuitServiceFaultCase::ALL {
            let outcome = run_service_fault_case(case);
            assert_eq!(outcome.case, case);
            assert_eq!(outcome.evidence, ExosuitEvidenceLevel::Simulation);
        }
    }

    #[test]
    fn identity_and_node_power_faults_block_service_admission() {
        for case in [
            SuitServiceFaultCase::StaleServiceIdentity,
            SuitServiceFaultCase::LowIdentityConfidence,
            SuitServiceFaultCase::ServiceNodePowerLoss,
        ] {
            let outcome = run_service_fault_case(case);
            assert_eq!(outcome.service_admitted, Some(false));
        }
    }

    #[test]
    fn physical_docking_faults_never_open_pressure_boundary() {
        for case in [
            SuitServiceFaultCase::CaptureLoss,
            SuitServiceFaultCase::SealDisagreement,
            SuitServiceFaultCase::PressureDisagreement,
            SuitServiceFaultCase::DecontaminationUnavailable,
        ] {
            let outcome = run_service_fault_case(case);
            assert_eq!(outcome.pressure_boundary_permitted, Some(false));
        }
    }

    #[test]
    fn contradictory_meter_rolls_back_and_partial_transfer_is_accounted() {
        let mismatch = run_service_fault_case(SuitServiceFaultCase::MeterDisagreement);
        assert_eq!(mismatch.transfer_committed, Some(false));
        assert_eq!(mismatch.protected_survival_preserved, Some(true));

        let interrupted = run_service_fault_case(SuitServiceFaultCase::InterruptedTransfer);
        assert_eq!(interrupted.transfer_committed, Some(true));
        assert_eq!(interrupted.protected_survival_preserved, Some(true));
    }

    #[test]
    fn local_safe_haven_logic_remains_available_during_comm_blackout() {
        let outcome = run_service_fault_case(SuitServiceFaultCase::CommunicationBlackout);
        assert_eq!(
            outcome.safe_haven_recommendation,
            Some(SafeHavenRecommendation::UseNearestSafeHaven)
        );
        assert_eq!(outcome.offline_local_operation_possible, Some(true));
    }

    #[test]
    fn unavailable_shelter_is_not_invented() {
        let outcome = run_service_fault_case(SuitServiceFaultCase::SafeHavenUnavailable);
        assert_eq!(
            outcome.safe_haven_recommendation,
            Some(SafeHavenRecommendation::NoFeasibleSafeHaven)
        );
    }
}
