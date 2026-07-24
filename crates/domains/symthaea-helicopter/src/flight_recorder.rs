// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic flight-data recording and replay-chain evidence.
//!
//! The recorder stores requested/applied commands, state, perturbations, fuel,
//! rotor energy, landing evidence, and ordered events. Every record extends a
//! deterministic FNV-1a replay chain. This detects accidental mutation and
//! segment discontinuity without claiming cryptographic tamper resistance.

use serde::{Deserialize, Serialize};

use crate::perturbations::PerturbationEffects;
use crate::powertrain::PowertrainState;
use crate::simulator::{LandingContact, SimpleHelicopterSimulator};
use crate::types::{HelicopterCommand, HelicopterState};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlightLogManifest {
    pub schema_version: String,
    pub scenario_id: String,
    pub controller_id: String,
    pub seed: u64,
    pub physics_hz: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlightFrame {
    pub sequence: u64,
    pub monotonic_time_s: f64,
    pub requested_command: HelicopterCommand,
    pub applied_command: HelicopterCommand,
    pub state: HelicopterState,
    pub perturbations: PerturbationEffects,
    pub powertrain: PowertrainState,
    pub rotor_kinetic_energy_j: f64,
    pub landing_contact: LandingContact,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FlightEventKind {
    AuthorityChanged {
        previous: String,
        current: String,
    },
    NavigationRejected {
        reason: String,
    },
    PerturbationChanged {
        description: String,
    },
    ReserveAction {
        action: String,
    },
    MissionDecisionAudit {
        status: String,
        violations: Vec<String>,
    },
    EvidenceSigned {
        key_id: String,
        signature_scheme: String,
    },
    RealtimeViolation {
        deadline_missed: bool,
        latency_exceeded: bool,
    },
    OperatorAnnotation {
        text: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlightEvent {
    pub sequence: u64,
    pub monotonic_time_s: f64,
    pub kind: FlightEventKind,
}

/// Unified append order. Frames and events remain in typed vectors while this
/// index preserves their exact interleaving for replay-chain verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlightRecordRef {
    Frame(usize),
    Event(usize),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlightSegmentSeal {
    pub schema_version: String,
    pub scenario_id: String,
    pub controller_id: String,
    pub segment_index: u32,
    pub parent_chain_tip: Option<String>,
    pub chain_tip: String,
    pub record_count: usize,
    pub first_sequence: u64,
    pub last_sequence: u64,
}

impl FlightSegmentSeal {
    pub fn follows(&self, previous: &Self) -> bool {
        self.schema_version == previous.schema_version
            && self.scenario_id == previous.scenario_id
            && self.controller_id == previous.controller_id
            && self.segment_index == previous.segment_index.saturating_add(1)
            && self.parent_chain_tip.as_deref() == Some(previous.chain_tip.as_str())
            && self.first_sequence > previous.last_sequence
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlightRecorderError {
    InvalidManifest,
    InvalidSegmentParent,
    NonFiniteFrame,
    SequenceDidNotIncrease,
    TimeWentBackwards,
    CapacityExceeded,
    EmptySegment,
    SerializationFailed,
    RecordOrderInvalid,
    EvidenceChainInvalid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlightRecorder {
    pub manifest: FlightLogManifest,
    pub frames: Vec<FlightFrame>,
    pub events: Vec<FlightEvent>,
    pub record_order: Vec<FlightRecordRef>,
    /// Per-record chain links aligned with `record_order`.
    pub record_chain_fnv1a64: Vec<u64>,
    pub segment_index: u32,
    pub parent_chain_tip: Option<String>,
    pub parent_last_sequence: Option<u64>,
    pub max_frames: usize,
}

impl FlightRecorder {
    pub fn new(
        manifest: FlightLogManifest,
        max_frames: usize,
    ) -> Result<Self, FlightRecorderError> {
        Self::new_internal(manifest, max_frames, 0, None, None)
    }

    pub fn new_segment(
        manifest: FlightLogManifest,
        max_frames: usize,
        previous: &FlightSegmentSeal,
    ) -> Result<Self, FlightRecorderError> {
        if manifest.schema_version != previous.schema_version
            || manifest.scenario_id != previous.scenario_id
            || manifest.controller_id != previous.controller_id
        {
            return Err(FlightRecorderError::InvalidSegmentParent);
        }
        Self::new_internal(
            manifest,
            max_frames,
            previous.segment_index.saturating_add(1),
            Some(previous.chain_tip.clone()),
            Some(previous.last_sequence),
        )
    }

    fn new_internal(
        manifest: FlightLogManifest,
        max_frames: usize,
        segment_index: u32,
        parent_chain_tip: Option<String>,
        parent_last_sequence: Option<u64>,
    ) -> Result<Self, FlightRecorderError> {
        if manifest.schema_version.trim().is_empty()
            || manifest.scenario_id.trim().is_empty()
            || manifest.controller_id.trim().is_empty()
            || !manifest.physics_hz.is_finite()
            || manifest.physics_hz <= 0.0
            || max_frames == 0
            || (segment_index == 0 && parent_chain_tip.is_some())
            || (segment_index > 0 && parent_chain_tip.is_none())
        {
            return Err(FlightRecorderError::InvalidManifest);
        }
        Ok(Self {
            manifest,
            frames: Vec::new(),
            events: Vec::new(),
            record_order: Vec::new(),
            record_chain_fnv1a64: Vec::new(),
            segment_index,
            parent_chain_tip,
            parent_last_sequence,
            max_frames,
        })
    }

    pub fn record_simulator_frame(
        &mut self,
        sequence: u64,
        monotonic_time_s: f64,
        requested_command: HelicopterCommand,
        simulator: &SimpleHelicopterSimulator,
    ) -> Result<(), FlightRecorderError> {
        let frame = FlightFrame {
            sequence,
            monotonic_time_s,
            requested_command,
            applied_command: simulator.applied_command(),
            state: simulator.state_snapshot(),
            perturbations: simulator.perturbation_effects(),
            powertrain: simulator.powertrain_state(),
            rotor_kinetic_energy_j: simulator.rotor_kinetic_energy_j(),
            landing_contact: simulator.landing_contact(),
        };
        self.record_frame(frame)
    }

    pub fn record_frame(&mut self, frame: FlightFrame) -> Result<(), FlightRecorderError> {
        if self.frames.len() >= self.max_frames {
            return Err(FlightRecorderError::CapacityExceeded);
        }
        if !frame.monotonic_time_s.is_finite()
            || !frame.rotor_kinetic_energy_j.is_finite()
            || !frame.state.is_finite()
            || !frame
                .requested_command
                .to_ctrl()
                .iter()
                .all(|value| value.is_finite())
            || !frame
                .applied_command
                .to_ctrl()
                .iter()
                .all(|value| value.is_finite())
        {
            return Err(FlightRecorderError::NonFiniteFrame);
        }
        self.validate_next_record(frame.sequence, frame.monotonic_time_s)?;
        let payload =
            serde_json::to_vec(&frame).map_err(|_| FlightRecorderError::SerializationFailed)?;
        let index = self.frames.len();
        let link = self.next_chain_link(b"frame", &payload)?;
        self.frames.push(frame);
        self.record_order.push(FlightRecordRef::Frame(index));
        self.record_chain_fnv1a64.push(link);
        Ok(())
    }

    pub fn record_event(&mut self, event: FlightEvent) -> Result<(), FlightRecorderError> {
        if !event.monotonic_time_s.is_finite() {
            return Err(FlightRecorderError::NonFiniteFrame);
        }
        self.validate_next_record(event.sequence, event.monotonic_time_s)?;
        let payload =
            serde_json::to_vec(&event).map_err(|_| FlightRecorderError::SerializationFailed)?;
        let index = self.events.len();
        let link = self.next_chain_link(b"event", &payload)?;
        self.events.push(event);
        self.record_order.push(FlightRecordRef::Event(index));
        self.record_chain_fnv1a64.push(link);
        Ok(())
    }

    fn validate_next_record(
        &self,
        sequence: u64,
        monotonic_time_s: f64,
    ) -> Result<(), FlightRecorderError> {
        if let Some((previous_sequence, previous_time)) = self.last_record_identity()? {
            if sequence <= previous_sequence {
                return Err(FlightRecorderError::SequenceDidNotIncrease);
            }
            if monotonic_time_s < previous_time {
                return Err(FlightRecorderError::TimeWentBackwards);
            }
        } else if self
            .parent_last_sequence
            .is_some_and(|previous| sequence <= previous)
        {
            return Err(FlightRecorderError::SequenceDidNotIncrease);
        }
        Ok(())
    }

    fn last_record_identity(&self) -> Result<Option<(u64, f64)>, FlightRecorderError> {
        match self.record_order.last().copied() {
            Some(FlightRecordRef::Frame(index)) => self
                .frames
                .get(index)
                .map(|record| Some((record.sequence, record.monotonic_time_s)))
                .ok_or(FlightRecorderError::RecordOrderInvalid),
            Some(FlightRecordRef::Event(index)) => self
                .events
                .get(index)
                .map(|record| Some((record.sequence, record.monotonic_time_s)))
                .ok_or(FlightRecorderError::RecordOrderInvalid),
            None => Ok(None),
        }
    }

    fn chain_seed(&self) -> Result<u64, FlightRecorderError> {
        let manifest = serde_json::to_vec(&self.manifest)
            .map_err(|_| FlightRecorderError::SerializationFailed)?;
        let mut hash = FNV_OFFSET;
        hash = fnv1a64_update(hash, b"symthaea-helicopter-flight-chain-v1");
        hash = fnv1a64_update(
            hash,
            self.parent_chain_tip
                .as_deref()
                .unwrap_or("genesis")
                .as_bytes(),
        );
        Ok(fnv1a64_update(hash, &manifest))
    }

    fn next_chain_link(&self, domain: &[u8], payload: &[u8]) -> Result<u64, FlightRecorderError> {
        let previous = self
            .record_chain_fnv1a64
            .last()
            .copied()
            .map(Ok)
            .unwrap_or_else(|| self.chain_seed())?;
        let mut hash = FNV_OFFSET;
        hash = fnv1a64_update(hash, &previous.to_le_bytes());
        hash = fnv1a64_update(hash, domain);
        Ok(fnv1a64_update(hash, payload))
    }

    pub fn verify_record_chain(&self) -> Result<(), FlightRecorderError> {
        if self.record_order.len() != self.record_chain_fnv1a64.len() {
            return Err(FlightRecorderError::RecordOrderInvalid);
        }
        let mut seen_frames = vec![false; self.frames.len()];
        let mut seen_events = vec![false; self.events.len()];
        let mut previous = self.chain_seed()?;
        let mut previous_identity: Option<(u64, f64)> = None;
        for (position, record_ref) in self.record_order.iter().copied().enumerate() {
            let (domain, payload, sequence, time) = match record_ref {
                FlightRecordRef::Frame(index) => {
                    let record = self
                        .frames
                        .get(index)
                        .ok_or(FlightRecorderError::RecordOrderInvalid)?;
                    if seen_frames[index] {
                        return Err(FlightRecorderError::RecordOrderInvalid);
                    }
                    seen_frames[index] = true;
                    (
                        b"frame".as_slice(),
                        serde_json::to_vec(record)
                            .map_err(|_| FlightRecorderError::SerializationFailed)?,
                        record.sequence,
                        record.monotonic_time_s,
                    )
                }
                FlightRecordRef::Event(index) => {
                    let record = self
                        .events
                        .get(index)
                        .ok_or(FlightRecorderError::RecordOrderInvalid)?;
                    if seen_events[index] {
                        return Err(FlightRecorderError::RecordOrderInvalid);
                    }
                    seen_events[index] = true;
                    (
                        b"event".as_slice(),
                        serde_json::to_vec(record)
                            .map_err(|_| FlightRecorderError::SerializationFailed)?,
                        record.sequence,
                        record.monotonic_time_s,
                    )
                }
            };
            if let Some((previous_sequence, previous_time)) = previous_identity {
                if sequence <= previous_sequence {
                    return Err(FlightRecorderError::SequenceDidNotIncrease);
                }
                if time < previous_time {
                    return Err(FlightRecorderError::TimeWentBackwards);
                }
            } else if self
                .parent_last_sequence
                .is_some_and(|parent| sequence <= parent)
            {
                return Err(FlightRecorderError::SequenceDidNotIncrease);
            }
            let mut expected = FNV_OFFSET;
            expected = fnv1a64_update(expected, &previous.to_le_bytes());
            expected = fnv1a64_update(expected, domain);
            expected = fnv1a64_update(expected, &payload);
            if self.record_chain_fnv1a64[position] != expected {
                return Err(FlightRecorderError::EvidenceChainInvalid);
            }
            previous = expected;
            previous_identity = Some((sequence, time));
        }
        if seen_frames.iter().any(|seen| !seen) || seen_events.iter().any(|seen| !seen) {
            return Err(FlightRecorderError::RecordOrderInvalid);
        }
        Ok(())
    }

    pub fn chain_tip_fnv1a64(&self) -> Result<String, FlightRecorderError> {
        let tip = self
            .record_chain_fnv1a64
            .last()
            .copied()
            .map(Ok)
            .unwrap_or_else(|| self.chain_seed())?;
        Ok(format!("fnv1a64-chain:{tip:016x}"))
    }

    pub fn seal_segment(&self) -> Result<FlightSegmentSeal, FlightRecorderError> {
        self.verify_record_chain()?;
        let Some((last_sequence, _)) = self.last_record_identity()? else {
            return Err(FlightRecorderError::EmptySegment);
        };
        let first_sequence = match self.record_order.first().copied() {
            Some(FlightRecordRef::Frame(index)) => self.frames[index].sequence,
            Some(FlightRecordRef::Event(index)) => self.events[index].sequence,
            None => return Err(FlightRecorderError::EmptySegment),
        };
        Ok(FlightSegmentSeal {
            schema_version: self.manifest.schema_version.clone(),
            scenario_id: self.manifest.scenario_id.clone(),
            controller_id: self.manifest.controller_id.clone(),
            segment_index: self.segment_index,
            parent_chain_tip: self.parent_chain_tip.clone(),
            chain_tip: self.chain_tip_fnv1a64()?,
            record_count: self.record_order.len(),
            first_sequence,
            last_sequence,
        })
    }

    /// Canonical for this schema: serde's deterministic struct-field order and
    /// vector order, with no maps in the evidence payload.
    pub fn canonical_json(&self) -> Result<Vec<u8>, FlightRecorderError> {
        serde_json::to_vec(self).map_err(|_| FlightRecorderError::SerializationFailed)
    }

    /// Stable non-cryptographic digest for complete artifact equality checks.
    pub fn evidence_digest_fnv1a64(&self) -> Result<String, FlightRecorderError> {
        let bytes = self.canonical_json()?;
        let hash = fnv1a64_update(FNV_OFFSET, &bytes);
        Ok(format!("fnv1a64:{hash:016x}"))
    }
}

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

fn fnv1a64_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::HelicopterPhysicsSimulator;

    fn manifest() -> FlightLogManifest {
        FlightLogManifest {
            schema_version: "symthaea-helicopter-flight-log-v1".to_string(),
            scenario_id: "hover-trim".to_string(),
            controller_id: "guidance-plus-hdc-residual".to_string(),
            seed: 7,
            physics_hz: 300.0,
        }
    }

    fn recorded_run() -> FlightRecorder {
        let mut recorder = FlightRecorder::new(manifest(), 10).unwrap();
        let mut simulator = SimpleHelicopterSimulator::new();
        let command = HelicopterCommand::hover();
        for sequence in 1..=3 {
            simulator.step(&command, 1.0 / 300.0);
            recorder
                .record_simulator_frame(sequence, sequence as f64 / 300.0, command, &simulator)
                .unwrap();
        }
        recorder
    }

    #[test]
    fn identical_runs_have_identical_digest_and_chain() {
        let a = recorded_run();
        let b = recorded_run();
        assert_eq!(a.canonical_json().unwrap(), b.canonical_json().unwrap());
        assert_eq!(
            a.evidence_digest_fnv1a64().unwrap(),
            b.evidence_digest_fnv1a64().unwrap()
        );
        assert_eq!(
            a.chain_tip_fnv1a64().unwrap(),
            b.chain_tip_fnv1a64().unwrap()
        );
        assert!(a.verify_record_chain().is_ok());
    }

    #[test]
    fn mutation_is_detected_by_replay_chain() {
        let mut recorder = recorded_run();
        recorder.frames[1].state.position[0] += 1.0;
        assert_eq!(
            recorder.verify_record_chain(),
            Err(FlightRecorderError::EvidenceChainInvalid)
        );
    }

    #[test]
    fn frames_and_events_share_global_sequence_order() {
        let mut recorder = FlightRecorder::new(manifest(), 10).unwrap();
        let simulator = SimpleHelicopterSimulator::new();
        recorder
            .record_simulator_frame(1, 0.0, HelicopterCommand::hover(), &simulator)
            .unwrap();
        recorder
            .record_event(FlightEvent {
                sequence: 2,
                monotonic_time_s: 0.1,
                kind: FlightEventKind::OperatorAnnotation {
                    text: "checkpoint".to_string(),
                },
            })
            .unwrap();
        assert!(recorder.verify_record_chain().is_ok());
        assert_eq!(recorder.record_order.len(), 2);
    }

    #[test]
    fn segment_seals_enforce_parent_continuity() {
        let first = recorded_run();
        let first_seal = first.seal_segment().unwrap();
        let mut second = FlightRecorder::new_segment(manifest(), 10, &first_seal).unwrap();
        let simulator = SimpleHelicopterSimulator::new();
        second
            .record_simulator_frame(4, 1.0, HelicopterCommand::hover(), &simulator)
            .unwrap();
        let second_seal = second.seal_segment().unwrap();
        assert!(second_seal.follows(&first_seal));
        assert_eq!(
            second.record_simulator_frame(3, 1.1, HelicopterCommand::hover(), &simulator),
            Err(FlightRecorderError::SequenceDidNotIncrease)
        );
    }

    #[test]
    fn capacity_is_bounded() {
        let mut recorder = FlightRecorder::new(manifest(), 1).unwrap();
        let simulator = SimpleHelicopterSimulator::new();
        recorder
            .record_simulator_frame(1, 0.0, HelicopterCommand::hover(), &simulator)
            .unwrap();
        assert_eq!(
            recorder.record_simulator_frame(2, 0.1, HelicopterCommand::hover(), &simulator),
            Err(FlightRecorderError::CapacityExceeded)
        );
    }
}
