// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Derive the nominal per-frame temporal-assurance evidence contract.

use crate::delayed_observation::{ObservationPurpose, TimedObservation};
use crate::mission::SubterraneanMissionIntent;
use crate::mission_executive::MissionExecutive;
use crate::plan_freshness::{PlanBasis, RuntimeRevisions};
use crate::safety::{HazardAssessment, SubterraneanHazard};
use crate::temporal_assurance::{TEMPORAL_RUNTIME_FRAME_SCHEMA_VERSION, TemporalRuntimeFrame};
use crate::temporal_clock::{ClockDomain, ClockSample, ClockSourceId};
use crate::temporal_event::{CausalEvent, CausalEventId, CausalEventKind, TimeInterval};

/// Derive a world-state revision fingerprint for the current cycle from the
/// same inputs already available to the embodiment's control loop -- the
/// current step, the freshly assessed hazard, the mission executive (whose
/// tunnel graph stands in for topology and whose maintenance assessment
/// stands in for equipment calibration drift), and the currently requested
/// mission. Two calls with an unchanged world produce identical revisions;
/// any safety-relevant change (a new hazard, an edited tunnel graph, a
/// maintenance-state change, or a new requested mission) changes exactly the
/// corresponding field, which is what [`crate::plan_freshness::PlanFreshnessSupervisor`]
/// uses to detect a plan that no longer matches the world it was made for.
pub fn temporal_runtime_revisions(
    step: u64,
    hazard: HazardAssessment,
    mission_executive: &MissionExecutive,
    requested_mission: SubterraneanMissionIntent,
) -> RuntimeRevisions {
    RuntimeRevisions {
        state: step,
        hazard: hazard_revision(hazard),
        topology: topology_revision(mission_executive),
        calibration: calibration_revision(mission_executive),
        mission: mission_revision(requested_mission),
    }
}

fn hazard_revision(hazard: HazardAssessment) -> u64 {
    let severity_bucket = (hazard.severity.clamp(0.0, 1.0) * 100.0).round() as u64;
    hazard_ordinal(hazard.primary)
        .wrapping_mul(1_000)
        .wrapping_add(severity_bucket)
}

fn hazard_ordinal(hazard: SubterraneanHazard) -> u64 {
    match hazard {
        SubterraneanHazard::None => 0,
        SubterraneanHazard::Thermal => 1,
        SubterraneanHazard::Flood => 2,
        SubterraneanHazard::Gas => 3,
        SubterraneanHazard::RoofInstability => 4,
        SubterraneanHazard::EscapeLoss => 5,
        SubterraneanHazard::LocalizationLoss => 6,
        SubterraneanHazard::CommunicationsLoss => 7,
        SubterraneanHazard::BatteryCritical => 8,
        SubterraneanHazard::SpoilJam => 9,
        SubterraneanHazard::ReturnReserve => 10,
        SubterraneanHazard::TunnelConflict => 11,
        SubterraneanHazard::GeologicalUncertainty => 12,
        SubterraneanHazard::SensorFault => 13,
    }
}

fn topology_revision(mission_executive: &MissionExecutive) -> u64 {
    let graph = mission_executive.graph();
    let mut digest = (graph.nodes().len() as u64)
        .wrapping_mul(0x0000_0001_0000_01B3)
        .wrapping_add(graph.edges().len() as u64);
    for edge in graph.edges() {
        digest ^= (u64::from(edge.from.0) << 32) | u64::from(edge.to.0);
        digest = digest.rotate_left(13).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    digest
}

fn calibration_revision(mission_executive: &MissionExecutive) -> u64 {
    let assessment = mission_executive.maintenance().assessment();
    let health_bucket = (assessment.minimum_health.clamp(0.0, 1.0) * 1_000.0).round() as u64;
    health_bucket
        .wrapping_add(if assessment.maintenance_due {
            1_000_000
        } else {
            0
        })
        .wrapping_add(if assessment.mission_abort_required {
            2_000_000
        } else {
            0
        })
}

fn mission_revision(mission: SubterraneanMissionIntent) -> u64 {
    match mission {
        SubterraneanMissionIntent::Explore => 0,
        SubterraneanMissionIntent::ProbeAhead => 1,
        SubterraneanMissionIntent::FollowVein => 2,
        SubterraneanMissionIntent::ReturnHome => 3,
        SubterraneanMissionIntent::EmergencySurface => 4,
        SubterraneanMissionIntent::HoldPosition => 5,
        SubterraneanMissionIntent::YieldTunnel => 6,
        SubterraneanMissionIntent::MaintainRelay => 7,
        SubterraneanMissionIntent::AssistPeer => 8,
    }
}

#[derive(Debug, Clone)]
pub struct TemporalRuntimeInputs;

impl TemporalRuntimeInputs {
    pub fn derive(
        step: u64,
        current_control_time_ns: u64,
        dt_seconds: f32,
        revisions: RuntimeRevisions,
    ) -> TemporalRuntimeFrame {
        let increment = if dt_seconds.is_finite() && dt_seconds > 0.0 {
            (f64::from(dt_seconds) * 1_000_000_000.0).round().max(1.0) as u64
        } else {
            0
        };
        let expected_time_ns = current_control_time_ns.saturating_add(increment);
        let sequence = step.saturating_add(1);
        TemporalRuntimeFrame {
            schema_version: TEMPORAL_RUNTIME_FRAME_SCHEMA_VERSION,
            clock_samples: vec![ClockSample {
                source: ClockSourceId(0),
                domain: ClockDomain::Control,
                boot_epoch: 1,
                sequence,
                event_time_ns: expected_time_ns,
                uncertainty_ns: 1_000,
                received_step: step,
            }],
            observations: vec![TimedObservation {
                source: 0,
                purpose: ObservationPurpose::ImmediateControl,
                observed_time_ns: expected_time_ns,
                received_time_ns: expected_time_ns,
                uncertainty_ns: 1_000,
                freshness_limit_ns: increment.saturating_mul(4).max(1_000_000),
                sequence,
            }],
            events: vec![CausalEvent {
                id: CausalEventId {
                    source: 0,
                    boot_epoch: 1,
                    sequence,
                },
                kind: CausalEventKind::SensorObservation,
                interval: TimeInterval::point(expected_time_ns),
                observed_step: step,
                dependencies: Vec::new(),
                state_revision: revisions.state,
                payload_digest: temporal_payload_digest(step, revisions),
            }],
            plan: Some(PlanBasis {
                plan_id: sequence,
                created_step: step,
                expires_step: step,
                revisions,
                permits_productive_work: true,
            }),
            causes: Vec::new(),
            responses: Vec::new(),
        }
    }
}

fn temporal_payload_digest(step: u64, revisions: RuntimeRevisions) -> u64 {
    let mut digest = step.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for value in [
        revisions.state,
        revisions.hazard,
        revisions.topology,
        revisions.calibration,
        revisions.mission,
    ] {
        digest ^= value.wrapping_add(0xA076_1D64_78BD_642F);
        digest = digest.rotate_left(17).wrapping_mul(0xE703_7ED1_A0B4_28DB);
    }
    digest
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_frame_is_current_and_self_consistent() {
        let revisions = RuntimeRevisions {
            state: 5,
            hazard: 2,
            topology: 3,
            calibration: 4,
            mission: 1,
        };
        let frame = TemporalRuntimeInputs::derive(5, 25_000_000, 0.005, revisions);
        assert!(frame.validate());
        assert_eq!(frame.clock_samples[0].event_time_ns, 30_000_000);
        assert_eq!(frame.plan.unwrap().revisions, revisions);
    }

    #[test]
    fn revisions_are_stable_for_an_unchanged_world_and_change_with_hazard() {
        let executive = MissionExecutive::default();
        let a = temporal_runtime_revisions(
            3,
            HazardAssessment::clear(),
            &executive,
            SubterraneanMissionIntent::Explore,
        );
        let b = temporal_runtime_revisions(
            3,
            HazardAssessment::clear(),
            &executive,
            SubterraneanMissionIntent::Explore,
        );
        assert_eq!(a, b);

        let hazard = HazardAssessment {
            primary: SubterraneanHazard::Thermal,
            safety_level: crate::embodiment::MotorSafetyLevel::Yellow,
            severity: 0.4,
        };
        let c =
            temporal_runtime_revisions(3, hazard, &executive, SubterraneanMissionIntent::Explore);
        assert_ne!(a.hazard, c.hazard);
        assert_eq!(a.state, c.state);
        assert_eq!(a.topology, c.topology);
        assert_eq!(a.calibration, c.calibration);
        assert_eq!(a.mission, c.mission);
    }
}
