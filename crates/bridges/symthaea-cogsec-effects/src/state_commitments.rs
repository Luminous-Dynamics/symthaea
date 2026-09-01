// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical resource-state commitments for first-hook ObserverOnly qualification.
//!
//! Graduation queue state is deliberately absent here. Its private owner,
//! `MemoryCoordinator`, mints `PendingGraduationCommitmentV1` directly so the
//! CogSec bridge cannot acquire raw queued memories merely to reconstruct a root.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use sha2::{Digest, Sha256};
use symthaea_cogsec::Digest32;
use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_memory::MemorySource;

use crate::{CognitiveEffectV1, WorkingMemoryItemView, effect_digest_v1};

const WM_STATE_DOMAIN_V1: &[u8] = b"SYMTHAEA_COGSEC_WM_STATE/v1";
const GOAL_STORE_STATE_DOMAIN_V1: &[u8] = b"SYMTHAEA_COGSEC_GOAL_STORE_STATE/v1";
const AFFECT_STATE_DOMAIN_V1: &[u8] = b"SYMTHAEA_COGSEC_AFFECT_STATE/v1";

/// Failure to construct a truthful canonical resource-state commitment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateCommitmentError {
    /// The legacy working-memory parallel arrays no longer describe one aligned owner state.
    WorkingMemoryParallelLengthMismatch {
        /// Number of HDC content entries.
        contents: usize,
        /// Number of arrival ticks.
        ticks: usize,
        /// Number of source entries.
        sources: usize,
        /// Number of legacy verification entries.
        verified: usize,
        /// Number of metadata entries.
        metadata: usize,
    },
}

impl fmt::Display for StateCommitmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkingMemoryParallelLengthMismatch {
                contents,
                ticks,
                sources,
                verified,
                metadata,
            } => write!(
                f,
                "working-memory parallel arrays differ in length: contents={contents}, ticks={ticks}, sources={sources}, verified={verified}, metadata={metadata}"
            ),
        }
    }
}

impl Error for StateCommitmentError {}

/// Read-only legacy goal record used to construct an ordered goal-store commitment.
#[derive(Debug, Clone, Copy)]
pub struct GoalRecordView<'a> {
    /// Exact goal identifier.
    pub id: &'a str,
    /// Exact goal description.
    pub description: &'a str,
    /// Exact goal embedding.
    pub embedding: &'a ContinuousHV,
    /// Exact stored priority.
    pub priority: f32,
    /// Exact stored progress.
    pub progress: f32,
    /// Exact stored active flag.
    pub is_active: bool,
}

/// Commit to the complete legacy working-memory owner state.
///
/// The current runtime stores one logical item across five parallel arrays. This
/// function rejects misaligned arrays rather than zipping/truncating them, then
/// commits to capacity, item count, order, HDC content, arrival tick, source,
/// legacy verification bit, and deterministically ordered metadata for every item.
pub(crate) fn working_memory_state_digest_v1(
    contents: &[ContinuousHV],
    ticks: &[u64],
    sources: &[MemorySource],
    verified: &[bool],
    metadata: &[HashMap<String, String>],
    capacity: usize,
) -> Result<Digest32, StateCommitmentError> {
    let len = contents.len();
    if ticks.len() != len
        || sources.len() != len
        || verified.len() != len
        || metadata.len() != len
    {
        return Err(StateCommitmentError::WorkingMemoryParallelLengthMismatch {
            contents: len,
            ticks: ticks.len(),
            sources: sources.len(),
            verified: verified.len(),
            metadata: metadata.len(),
        });
    }

    let mut out = StateWriter::with_domain(WM_STATE_DOMAIN_V1);
    out.u64(capacity as u64);
    out.u64(len as u64);

    for index in 0..len {
        let item = CognitiveEffectV1::working_memory_admit(
            WorkingMemoryItemView {
                content: &contents[index],
                arrival_tick: ticks[index],
                source: sources[index],
                verified: verified[index],
                metadata: &metadata[index],
            },
            index as u64,
        );
        out.digest(effect_digest_v1(&item));
    }

    Ok(sha256(&out.finish()))
}

/// Commit to an ordered legacy goal store.
///
/// Each record reuses the exact goal-effect field semantics, while this state
/// root also commits to the record's current store index. Reordering otherwise
/// identical goals therefore changes the resource root.
pub(crate) fn goal_store_state_digest_v1(goals: &[GoalRecordView<'_>]) -> Digest32 {
    let mut out = StateWriter::with_domain(GOAL_STORE_STATE_DOMAIN_V1);
    out.u64(goals.len() as u64);
    for (index, goal) in goals.iter().enumerate() {
        let record = CognitiveEffectV1::goal_activate(
            goal.id,
            goal.description,
            goal.embedding,
            goal.priority,
            goal.progress,
            goal.is_active,
        );
        out.u64(index as u64);
        out.digest(effect_digest_v1(&record));
    }
    sha256(&out.finish())
}

/// Commit to the first-hook affect resource (`emotional_valence`) exactly.
///
/// The v1 root is intentionally narrow. Widening the protected affect owner to
/// additional fields requires a new schema/domain rather than silently changing
/// the meaning of this root.
pub(crate) fn affect_state_digest_v1(emotional_valence: f32) -> Digest32 {
    let mut out = StateWriter::with_domain(AFFECT_STATE_DOMAIN_V1);
    out.u32(emotional_valence.to_bits());
    sha256(&out.finish())
}

fn sha256(bytes: &[u8]) -> Digest32 {
    let digest: [u8; 32] = Sha256::digest(bytes).into();
    Digest32(digest)
}

#[derive(Debug)]
struct StateWriter {
    bytes: Vec<u8>,
}

impl StateWriter {
    fn with_domain(domain: &[u8]) -> Self {
        let mut bytes = Vec::with_capacity(domain.len() + 1 + 128);
        bytes.extend_from_slice(domain);
        bytes.push(0);
        Self { bytes }
    }

    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn digest(&mut self, value: Digest32) {
        self.bytes.extend_from_slice(&value.0);
    }

    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hv(a: f32, b: f32) -> ContinuousHV {
        ContinuousHV::from_values(vec![a, b])
    }

    #[test]
    fn working_memory_root_rejects_parallel_array_misalignment() {
        let contents = vec![hv(0.1, 0.2)];
        let ticks = vec![];
        let sources = vec![MemorySource::Internal];
        let verified = vec![false];
        let metadata = vec![HashMap::new()];

        assert!(matches!(
            working_memory_state_digest_v1(
                &contents,
                &ticks,
                &sources,
                &verified,
                &metadata,
                4
            ),
            Err(StateCommitmentError::WorkingMemoryParallelLengthMismatch { .. })
        ));
    }

    #[test]
    fn working_memory_root_binds_capacity_order_and_arrival_tick() {
        let a = hv(0.1, 0.2);
        let b = hv(0.3, 0.4);
        let contents = vec![a.clone(), b.clone()];
        let reversed = vec![b, a];
        let ticks = vec![3, 7];
        let reversed_ticks = vec![7, 3];
        let sources = vec![MemorySource::Internal, MemorySource::UserInteraction];
        let reversed_sources = vec![MemorySource::UserInteraction, MemorySource::Internal];
        let verified = vec![false, true];
        let reversed_verified = vec![true, false];
        let metadata = vec![HashMap::new(), HashMap::new()];
        let reversed_metadata = metadata.clone();

        let base = working_memory_state_digest_v1(
            &contents,
            &ticks,
            &sources,
            &verified,
            &metadata,
            4,
        )
        .unwrap();
        let capacity_changed = working_memory_state_digest_v1(
            &contents,
            &ticks,
            &sources,
            &verified,
            &metadata,
            5,
        )
        .unwrap();
        let order_changed = working_memory_state_digest_v1(
            &reversed,
            &reversed_ticks,
            &reversed_sources,
            &reversed_verified,
            &reversed_metadata,
            4,
        )
        .unwrap();
        let tick_changed = working_memory_state_digest_v1(
            &contents,
            &[3, 8],
            &sources,
            &verified,
            &metadata,
            4,
        )
        .unwrap();

        assert_ne!(base, capacity_changed);
        assert_ne!(base, order_changed);
        assert_ne!(base, tick_changed);
    }

    #[test]
    fn goal_store_root_binds_order_and_record_fields() {
        let a = hv(0.1, 0.2);
        let b = hv(0.3, 0.4);
        let goals = [
            GoalRecordView {
                id: "goal_0",
                description: "a",
                embedding: &a,
                priority: 0.8,
                progress: 0.0,
                is_active: true,
            },
            GoalRecordView {
                id: "goal_1",
                description: "b",
                embedding: &b,
                priority: 0.7,
                progress: 0.1,
                is_active: true,
            },
        ];
        let reversed = [goals[1], goals[0]];
        let changed = [
            GoalRecordView {
                priority: 0.9,
                ..goals[0]
            },
            goals[1],
        ];

        let base = goal_store_state_digest_v1(&goals);
        assert_ne!(base, goal_store_state_digest_v1(&reversed));
        assert_ne!(base, goal_store_state_digest_v1(&changed));
    }

    #[test]
    fn affect_root_is_bit_exact() {
        assert_ne!(affect_state_digest_v1(0.0), affect_state_digest_v1(-0.0));
        assert_ne!(affect_state_digest_v1(0.2), affect_state_digest_v1(0.3));
    }
}
