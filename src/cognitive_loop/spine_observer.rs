// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! SPINE-000B Stage-A runtime observation substrate.
//!
//! This module is deliberately **not wired into `CognitiveLoopService` yet**.
//! It defines only fixed-size POD records, frozen numeric IDs, validation helpers,
//! and bounded non-allocating buffers for a later qualified observer tranche.
//! No hashing, serialization, I/O, wall-clock access, RNG, or callbacks belong here.

#![allow(dead_code)]

use core::mem::size_of;

pub(crate) const MANAGER_EVENT_CAPACITY: usize = 64;
pub(crate) const APPLICATION_EVENT_CAPACITY: usize = 32;
pub(crate) const GUARD_EVENT_CAPACITY: usize = 32;

pub(crate) mod execution_outcome {
    pub const SKIPPED_SCHEDULE: u8 = 0;
    pub const SKIPPED_HEALTH_DISABLED: u8 = 1;
    pub const EXECUTED_NEUTRAL: u8 = 2;
    pub const EXECUTED_NON_NEUTRAL: u8 = 3;
    pub const PANICKED_CAUGHT: u8 = 4;
    pub const FAILED_OTHER: u8 = 5;
}

pub(crate) mod urgency {
    pub const CRUISE: u8 = 0;
    pub const NORMAL: u8 = 1;
    pub const CRITICAL: u8 = 2;
}

pub(crate) mod application_source {
    pub const CONFIDENCE_DELTA: u8 = 0;
    pub const LR_MODULATION: u8 = 1;
    pub const EXPLORATION_DELTA: u8 = 2;
    pub const AROUSAL_DELTA: u8 = 3;
    pub const VALENCE_DELTA: u8 = 4;
    pub const FLAG: u8 = 5;
}

pub(crate) mod source_condition {
    pub const SCALAR_NON_IDENTITY: u8 = 0;
    pub const FLAG_SET: u8 = 1;
    pub const FLAG_CLEAR: u8 = 2;
}

pub(crate) mod state_change_status {
    pub const UNCHANGED: u8 = 0;
    pub const CHANGED: u8 = 1;
    pub const NOT_OBSERVED_AT_BOUNDARY: u8 = 2;
}

/// CanonicalValue tags from SPINE-000B C1. `NONE` is observer-local and means
/// that no stable canonical value exists at this application boundary.
pub(crate) mod value_kind {
    pub const NONE: u8 = 0;
    pub const F64_BITS: u8 = 1;
    pub const F32_BITS: u8 = 2;
    pub const U64: u8 = 3;
    pub const U32: u8 = 4;
    pub const BOOL: u8 = 5;
    pub const DIGEST32: u8 = 6;
}

/// C2 commits only predicates actually evaluated by production.
pub(crate) mod guard_outcome {
    pub const FALSE: u8 = 0;
    pub const TRUE: u8 = 1;
}

/// Stable manager IDs from SPINE_000B_MANAGER_ID_REGISTRY_V1.json.
pub(crate) mod manager_id {
    pub const DRIVE_MANAGER: u16 = 1;
    pub const MEMORY_MANAGER: u16 = 2;
    pub const LEARNING_MANAGER: u16 = 3;
    pub const MULTIMODAL_MANAGER: u16 = 4;
    pub const PERCEPTION_MANAGER: u16 = 5;
    pub const SWARM_MANAGER: u16 = 6;
    pub const MUSE_MANAGER: u16 = 7;
    pub const THERMODYNAMIC_MANAGER: u16 = 8;
    pub const SOUL_MANAGER: u16 = 9;
    pub const SPECTRUM_MANAGER: u16 = 10;
    pub const CPG_MANAGER: u16 = 11;
    pub const SPECTRAL_TWIN: u16 = 12;
    pub const THERAPEUTIC_MANAGER: u16 = 13;
    pub const FABRICATION_MANAGER: u16 = 14;
    pub const LANGUAGE_MANAGER: u16 = 15;
    pub const NEUROEVOLUTION: u16 = 16;
    pub const HYPERVISOR: u16 = 17;
    pub const VISION_MANAGER: u16 = 18;
    pub const REASONING_MANAGER: u16 = 19;
    pub const GOVERNANCE_MANAGER: u16 = 20;
    pub const GLYPH_MANAGER: u16 = 21;
    pub const TIME_MANAGER: u16 = 22;
    pub const TRUST_MANAGER: u16 = 23;
    pub const SOCIAL_FABRIC_MANAGER: u16 = 24;
    pub const SURVIVAL_MANAGER: u16 = 25;
    pub const MAX_V1: u16 = SURVIVAL_MANAGER;
}

/// Stable application IDs from SPINE_000B_OPERATION_OBSERVER_IDS_V1.json.
pub(crate) mod operation_id {
    pub const FEEDBACK_ADJUST_CONFIDENCE: u16 = 1;
    pub const FEEDBACK_SCALE_LR: u16 = 2;
    pub const FEEDBACK_ADJUST_EXPLORATION: u16 = 3;
    pub const EMOTION_ADD_CLAMPED_AROUSAL: u16 = 4;
    pub const EMOTION_ADD_CLAMPED_VALENCE: u16 = 5;
    pub const FEEDBACK_ADJUST_EXPLORATION_REQUEST: u16 = 6;
    pub const EPISODIC_MEMORY_CONSOLIDATE_RECENT: u16 = 7;
    pub const STATS_INCREMENT_ANOMALY_COUNT: u16 = 8;
    pub const FEEDBACK_SCALE_CONFIDENCE_ANOMALY: u16 = 9;
    pub const QUALITY_SET_SUBSYSTEM_VETO: u16 = 10;
    pub const FEEDBACK_SCALE_LR_REQUEST_REST: u16 = 11;
    pub const NETWORK_BROADCAST_SWARM_STATE: u16 = 12;
    pub const EMOTION_ADD_CLAMPED_URGENCY_AROUSAL: u16 = 13;
    pub const FEEDBACK_SCALE_EXPLORATION_URGENCY: u16 = 14;
    pub const QUALITY_SET_REQUEST_GEODESIC: u16 = 15;
    pub const VISION_SELECT_BEST_GEODESIC: u16 = 16;
    pub const VISION_POPULATE_MENTAL_MOVIE: u16 = 17;
    pub const QUALITY_CLEAR_REQUEST_GEODESIC: u16 = 18;
    pub const MAX_V1: u16 = QUALITY_CLEAR_REQUEST_GEODESIC;
}

/// Stable predicate IDs from SPINE_000B_RUNTIME_GUARD_OVERLAY_V1.json.
pub(crate) mod predicate_id {
    pub const VISION_BRIDGE_PRESENT: u16 = 1;
    pub const GEODESIC_PATH_NONEMPTY: u16 = 2;
    pub const DECODED_GEODESIC_FRAMES_NONEMPTY: u16 = 3;
    pub const NODE_ID_PRESENT: u16 = 4;
    pub const CONSCIOUSNESS_HV_PRESENT: u16 = 5;
    pub const INTENT_HV_PRESENT: u16 = 6;
    pub const NETWORK_SERVICE_PRESENT: u16 = 7;
    pub const MAX_V1: u16 = NETWORK_SERVICE_PRESENT;
}

/// Exact proposal bits emitted by one subsystem. These are raw bits, not qualified
/// values; N1 performs proposal-domain qualification later.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct ProposalBitsV2 {
    pub confidence_delta_bits: u64,
    pub lr_modulation_bits: u64,
    pub exploration_delta_bits: u64,
    pub arousal_delta_bits: u32,
    pub valence_delta_bits: u32,
    pub flags: u32,
    pub reserved: u32,
}

/// Raw Stage-A manager event. Fixed at 64 bytes on all supported Rust targets by
/// explicit primitive widths and padding; no pointers or `usize` enter the record.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct ManagerExecutionEventV1 {
    pub cycle_number: u64,
    pub proposal: ProposalBitsV2,
    pub manager_id: u16,
    pub sequence: u16,
    pub outcome: u8,
    pub urgency: u8,
    pub eligible: u8,
    pub emitted: u8,
    pub admitted: u8,
    pub proposal_present: u8,
    pub _padding: [u8; 6],
}

/// Raw Stage-A application event. `*_bits` are interpretation-free 64-bit slots;
/// `argument_kind` / `observation_kind` determine how deferred Stage B decodes them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct ApplicationEventV1 {
    pub cycle_number: u64,
    pub applied_argument_bits: u64,
    pub before_bits: u64,
    pub after_bits: u64,
    pub source_flag: u32,
    pub feature_profile_bits: u32,
    pub application_index: u16,
    pub operation_id: u16,
    pub source_tag: u8,
    pub source_condition: u8,
    pub argument_kind: u8,
    pub observation_kind: u8,
    pub applied: u8,
    pub state_change_status: u8,
    pub _padding: [u8; 6],
}

/// A guard witness exists only when production actually evaluated the predicate.
/// `outcome` accepts only FALSE/TRUE; absent downstream predicates are represented
/// by absence, not by fabricated NOT_EVALUATED events.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct GuardWitnessEventV1 {
    pub cycle_number: u64,
    pub witness_index: u16,
    pub predicate_id: u16,
    pub outcome: u8,
    pub _padding: [u8; 3],
}

const _: [(); 48] = [(); size_of::<ProposalBitsV2>()];
const _: [(); 64] = [(); size_of::<ManagerExecutionEventV1>()];
const _: [(); 56] = [(); size_of::<ApplicationEventV1>()];
const _: [(); 16] = [(); size_of::<GuardWitnessEventV1>()];
const _: [(); 64] = [(); MANAGER_EVENT_CAPACITY];
const _: [(); 32] = [(); APPLICATION_EVENT_CAPACITY];
const _: [(); 32] = [(); GUARD_EVENT_CAPACITY];

/// Bounded, allocation-free append buffer. Once full it records overflow and
/// refuses further events without modifying existing entries.
#[derive(Clone, Debug)]
pub(crate) struct FixedEventBuffer<T: Copy + Default, const N: usize> {
    events: [T; N],
    len: u16,
    overflowed: u8,
}

impl<T: Copy + Default, const N: usize> Default for FixedEventBuffer<T, N> {
    fn default() -> Self {
        assert!(N <= u16::MAX as usize, "SPINE fixed buffer capacity exceeds u16");
        Self {
            events: [T::default(); N],
            len: 0,
            overflowed: 0,
        }
    }
}

impl<T: Copy + Default, const N: usize> FixedEventBuffer<T, N> {
    #[inline]
    pub(crate) fn push(&mut self, event: T) -> bool {
        let index = self.len as usize;
        if index >= N {
            self.overflowed = 1;
            return false;
        }
        self.events[index] = event;
        self.len += 1;
        true
    }

    #[inline]
    pub(crate) fn as_slice(&self) -> &[T] {
        &self.events[..self.len as usize]
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.len as usize
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub(crate) fn overflowed(&self) -> bool {
        self.overflowed != 0
    }

    /// O(1) reset. Stale storage is inaccessible because `len` becomes zero.
    #[inline]
    pub(crate) fn clear(&mut self) {
        self.len = 0;
        self.overflowed = 0;
    }
}

pub(crate) type ManagerEventBuffer =
    FixedEventBuffer<ManagerExecutionEventV1, MANAGER_EVENT_CAPACITY>;
pub(crate) type ApplicationEventBuffer =
    FixedEventBuffer<ApplicationEventV1, APPLICATION_EVENT_CAPACITY>;
pub(crate) type GuardEventBuffer = FixedEventBuffer<GuardWitnessEventV1, GUARD_EVENT_CAPACITY>;

#[derive(Clone, Debug, Default)]
pub(crate) struct RuntimeCaptureBuffersV1 {
    pub managers: ManagerEventBuffer,
    pub applications: ApplicationEventBuffer,
    pub guards: GuardEventBuffer,
}

impl RuntimeCaptureBuffersV1 {
    #[inline]
    pub(crate) fn any_overflow(&self) -> bool {
        self.managers.overflowed() || self.applications.overflowed() || self.guards.overflowed()
    }

    #[inline]
    pub(crate) fn clear(&mut self) {
        self.managers.clear();
        self.applications.clear();
        self.guards.clear();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RawEventError {
    InvalidManagerId,
    InvalidOperationId,
    InvalidPredicateId,
    InvalidOutcome,
    InvalidUrgency,
    InvalidBool,
    InvalidSourceTag,
    InvalidSourceCondition,
    InvalidValueKind,
    InvalidStateChangeStatus,
    InvalidExecutionTruthTable,
    InvalidNeutralityLabel,
    InvalidGuardOutcome,
}

#[inline]
fn bool_byte(v: u8) -> bool {
    v <= 1
}

#[inline]
fn proposal_is_neutral(p: ProposalBitsV2) -> bool {
    // Deliberately matches current production `SubsystemOutput::is_neutral()`:
    // `_reserved` is not part of behavioral neutrality, while C2 still commits it.
    f64::from_bits(p.confidence_delta_bits) == 0.0
        && f64::from_bits(p.lr_modulation_bits) == 1.0
        && f64::from_bits(p.exploration_delta_bits) == 0.0
        && f32::from_bits(p.arousal_delta_bits) == 0.0
        && f32::from_bits(p.valence_delta_bits) == 0.0
        && p.flags == 0
}

pub(crate) fn validate_manager_event(event: &ManagerExecutionEventV1) -> Result<(), RawEventError> {
    use execution_outcome::*;
    if !(1..=manager_id::MAX_V1).contains(&event.manager_id) {
        return Err(RawEventError::InvalidManagerId);
    }
    if event.outcome > FAILED_OTHER {
        return Err(RawEventError::InvalidOutcome);
    }
    if event.urgency > urgency::CRITICAL {
        return Err(RawEventError::InvalidUrgency);
    }
    if !bool_byte(event.eligible)
        || !bool_byte(event.emitted)
        || !bool_byte(event.admitted)
        || !bool_byte(event.proposal_present)
    {
        return Err(RawEventError::InvalidBool);
    }

    let actual = (event.eligible, event.emitted, event.admitted, event.proposal_present);
    let expected = match event.outcome {
        SKIPPED_SCHEDULE => (0, 0, 0, 0),
        SKIPPED_HEALTH_DISABLED | PANICKED_CAUGHT | FAILED_OTHER => (1, 0, 0, 0),
        EXECUTED_NEUTRAL => (1, 1, 0, 1),
        EXECUTED_NON_NEUTRAL => (1, 1, 1, 1),
        _ => return Err(RawEventError::InvalidOutcome),
    };
    if actual != expected {
        return Err(RawEventError::InvalidExecutionTruthTable);
    }

    if event.proposal_present == 1 {
        let neutral = proposal_is_neutral(event.proposal);
        if (event.outcome == EXECUTED_NEUTRAL && !neutral)
            || (event.outcome == EXECUTED_NON_NEUTRAL && neutral)
        {
            return Err(RawEventError::InvalidNeutralityLabel);
        }
    }
    Ok(())
}

pub(crate) fn validate_application_event(event: &ApplicationEventV1) -> Result<(), RawEventError> {
    if !(1..=operation_id::MAX_V1).contains(&event.operation_id) {
        return Err(RawEventError::InvalidOperationId);
    }
    if event.source_tag > application_source::FLAG {
        return Err(RawEventError::InvalidSourceTag);
    }
    if event.source_condition > source_condition::FLAG_CLEAR {
        return Err(RawEventError::InvalidSourceCondition);
    }
    if event.argument_kind > value_kind::DIGEST32 || event.observation_kind > value_kind::DIGEST32 {
        return Err(RawEventError::InvalidValueKind);
    }
    if !bool_byte(event.applied) {
        return Err(RawEventError::InvalidBool);
    }
    if event.state_change_status > state_change_status::NOT_OBSERVED_AT_BOUNDARY {
        return Err(RawEventError::InvalidStateChangeStatus);
    }
    if event.source_tag == application_source::FLAG {
        if event.source_flag == 0
            || !matches!(event.source_condition, source_condition::FLAG_SET | source_condition::FLAG_CLEAR)
        {
            return Err(RawEventError::InvalidSourceCondition);
        }
    } else if event.source_flag != 0 || event.source_condition != source_condition::SCALAR_NON_IDENTITY {
        return Err(RawEventError::InvalidSourceCondition);
    }
    Ok(())
}

pub(crate) fn validate_guard_event(event: &GuardWitnessEventV1) -> Result<(), RawEventError> {
    if !(1..=predicate_id::MAX_V1).contains(&event.predicate_id) {
        return Err(RawEventError::InvalidPredicateId);
    }
    if !matches!(event.outcome, guard_outcome::FALSE | guard_outcome::TRUE) {
        return Err(RawEventError::InvalidGuardOutcome);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn neutral_proposal() -> ProposalBitsV2 {
        ProposalBitsV2 {
            confidence_delta_bits: 0.0f64.to_bits(),
            lr_modulation_bits: 1.0f64.to_bits(),
            exploration_delta_bits: 0.0f64.to_bits(),
            arousal_delta_bits: 0.0f32.to_bits(),
            valence_delta_bits: 0.0f32.to_bits(),
            flags: 0,
            reserved: 0,
        }
    }

    #[test]
    fn frozen_record_sizes() {
        assert_eq!(size_of::<ProposalBitsV2>(), 48);
        assert_eq!(size_of::<ManagerExecutionEventV1>(), 64);
        assert_eq!(size_of::<ApplicationEventV1>(), 56);
        assert_eq!(size_of::<GuardWitnessEventV1>(), 16);
        assert_eq!(MANAGER_EVENT_CAPACITY, 64);
        assert_eq!(APPLICATION_EVENT_CAPACITY, 32);
        assert_eq!(GUARD_EVENT_CAPACITY, 32);
    }

    #[test]
    fn fixed_buffer_overflow_never_overwrites() {
        let mut b: FixedEventBuffer<GuardWitnessEventV1, 2> = FixedEventBuffer::default();
        let first = GuardWitnessEventV1 { cycle_number: 1, witness_index: 0, predicate_id: 1, outcome: guard_outcome::TRUE, _padding: [0; 3] };
        let second = GuardWitnessEventV1 { cycle_number: 1, witness_index: 1, predicate_id: 2, outcome: guard_outcome::FALSE, _padding: [0; 3] };
        let rejected = GuardWitnessEventV1 { cycle_number: 1, witness_index: 2, predicate_id: 3, outcome: guard_outcome::TRUE, _padding: [0; 3] };
        assert!(b.push(first));
        assert!(b.push(second));
        assert!(!b.push(rejected));
        assert!(b.overflowed());
        assert_eq!(b.as_slice(), &[first, second]);
    }

    #[test]
    fn fixed_buffer_clear_is_constant_state_reset() {
        let mut b: FixedEventBuffer<GuardWitnessEventV1, 1> = FixedEventBuffer::default();
        assert!(b.push(GuardWitnessEventV1::default()));
        assert!(!b.push(GuardWitnessEventV1::default()));
        assert!(b.overflowed());
        b.clear();
        assert!(b.is_empty());
        assert!(!b.overflowed());
        assert!(b.push(GuardWitnessEventV1::default()));
    }

    #[test]
    fn manager_truth_table_and_reserved_only_neutrality() {
        let mut proposal = neutral_proposal();
        proposal.reserved = 0xA5A5_5A5A;
        let event = ManagerExecutionEventV1 {
            cycle_number: 7,
            proposal,
            manager_id: manager_id::DRIVE_MANAGER,
            sequence: 0,
            outcome: execution_outcome::EXECUTED_NEUTRAL,
            urgency: urgency::NORMAL,
            eligible: 1,
            emitted: 1,
            admitted: 0,
            proposal_present: 1,
            _padding: [0; 6],
        };
        assert_eq!(event.proposal.reserved, 0xA5A5_5A5A);
        assert_eq!(validate_manager_event(&event), Ok(()));

        let mut bad = event;
        bad.outcome = execution_outcome::EXECUTED_NON_NEUTRAL;
        bad.admitted = 1;
        assert_eq!(validate_manager_event(&bad), Err(RawEventError::InvalidNeutralityLabel));
    }

    #[test]
    fn invalid_raw_tags_fail_closed() {
        let mut manager = ManagerExecutionEventV1 {
            manager_id: 1,
            outcome: 99,
            ..Default::default()
        };
        assert_eq!(validate_manager_event(&manager), Err(RawEventError::InvalidOutcome));
        manager.outcome = execution_outcome::SKIPPED_SCHEDULE;
        manager.eligible = 2;
        assert_eq!(validate_manager_event(&manager), Err(RawEventError::InvalidBool));

        let guard = GuardWitnessEventV1 { predicate_id: 1, outcome: 2, ..Default::default() };
        assert_eq!(validate_guard_event(&guard), Err(RawEventError::InvalidGuardOutcome));
    }

    #[test]
    fn application_source_condition_rules_fail_closed() {
        let scalar = ApplicationEventV1 {
            operation_id: operation_id::FEEDBACK_ADJUST_CONFIDENCE,
            source_tag: application_source::CONFIDENCE_DELTA,
            source_condition: source_condition::SCALAR_NON_IDENTITY,
            argument_kind: value_kind::F32_BITS,
            observation_kind: value_kind::F64_BITS,
            applied: 1,
            state_change_status: state_change_status::CHANGED,
            ..Default::default()
        };
        assert_eq!(validate_application_event(&scalar), Ok(()));

        let mut bad_scalar = scalar;
        bad_scalar.source_flag = 1;
        assert_eq!(validate_application_event(&bad_scalar), Err(RawEventError::InvalidSourceCondition));

        let flag = ApplicationEventV1 {
            operation_id: operation_id::QUALITY_SET_SUBSYSTEM_VETO,
            source_tag: application_source::FLAG,
            source_flag: 1 << 3,
            source_condition: source_condition::FLAG_SET,
            argument_kind: value_kind::BOOL,
            observation_kind: value_kind::BOOL,
            applied: 1,
            state_change_status: state_change_status::CHANGED,
            ..Default::default()
        };
        assert_eq!(validate_application_event(&flag), Ok(()));
    }

    #[test]
    fn combined_buffers_keep_independent_overflow_state() {
        let mut all = RuntimeCaptureBuffersV1::default();
        for i in 0..MANAGER_EVENT_CAPACITY {
            let e = ManagerExecutionEventV1 { manager_id: 1, sequence: i as u16, ..Default::default() };
            assert!(all.managers.push(e));
        }
        assert!(!all.any_overflow());
        assert!(!all.managers.push(ManagerExecutionEventV1::default()));
        assert!(all.any_overflow());
        assert_eq!(all.applications.len(), 0);
        assert_eq!(all.guards.len(), 0);
        all.clear();
        assert!(!all.any_overflow());
    }
}
