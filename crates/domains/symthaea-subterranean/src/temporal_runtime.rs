// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Derive the nominal per-frame temporal-assurance evidence contract.

use crate::delayed_observation::{ObservationPurpose, TimedObservation};
use crate::plan_freshness::{PlanBasis, RuntimeRevisions};
use crate::temporal_assurance::{
    TEMPORAL_RUNTIME_FRAME_SCHEMA_VERSION, TemporalRuntimeFrame,
};
use crate::temporal_clock::{ClockDomain, ClockSample, ClockSourceId};
use crate::temporal_event::{CausalEvent, CausalEventId, CausalEventKind, TimeInterval};

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
}
