// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Strongly typed wrappers around canonical CogSec commitments.
//!
//! The reference monitor intentionally accepts the generic [`Digest32`] type so
//! cryptography stays outside its logical TCB. Application code should not use
//! that genericity as permission to mix unrelated commitments. This module keeps
//! effect identity and protected-resource state identity distinct until the
//! trusted adapter deliberately unwraps them at the monitor boundary.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use symthaea_cogsec::Digest32;
use symthaea_core::hdc::LiquidHolocell;
use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_memory::{GraduationEvent, MemorySource};

use crate::{
    CognitiveEffectV1, GoalRecordView, StateCommitmentError, active_state_digest_v1,
    affect_state_digest_v1, effect_digest_v1, goal_store_state_digest_v1,
    graduation_queue_state_digest_v1, working_memory_state_digest_v1,
};

/// Canonical protected resource classes used by the first CogSec runtime tranche.
///
/// These names intentionally match the resource identities frozen by the shadow
/// runtime. The enum is commitment metadata only; it grants no authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CanonicalResourceV1 {
    /// Working-memory ordering/content/metadata owner.
    WorkingMemory,
    /// Liquid Holocell plus `current_thought` owner.
    ActiveCognitiveState,
    /// Ordered active-goal store.
    GoalStore,
    /// First-tranche affective state (`emotional_valence`).
    AffectiveState,
    /// Pending memory-graduation queue.
    GraduationQueue,
}

impl CanonicalResourceV1 {
    /// Stable resource identifier used by the CogSec shadow runtime.
    pub const fn resource_name(self) -> &'static str {
        match self {
            Self::WorkingMemory => "mind/working-memory",
            Self::ActiveCognitiveState => "mind/active-cognitive-state",
            Self::GoalStore => "mind/goals",
            Self::AffectiveState => "mind/affect",
            Self::GraduationQueue => "mind/memory/graduation",
        }
    }
}

/// Canonical commitment to one exact cognitive effect.
///
/// This is ordinary effect-identity data, not a permit, owner root, signature,
/// trusted fact, or authorization. The inner digest is private specifically so
/// application code has to make an explicit semantic conversion at the monitor
/// boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EffectCommitmentV1 {
    digest: Digest32,
}

impl EffectCommitmentV1 {
    /// Commit to one canonical effect.
    pub fn new(effect: &CognitiveEffectV1) -> Self {
        Self {
            digest: effect_digest_v1(effect),
        }
    }

    /// Borrow the generic monitor digest representation explicitly.
    pub const fn digest(&self) -> Digest32 {
        self.digest
    }

    /// Consume this typed wrapper and return the generic monitor digest.
    pub const fn into_digest(self) -> Digest32 {
        self.digest
    }
}

/// Canonical commitment to one named protected-resource state.
///
/// The resource tag prevents accidental substitution of, for example, a goal
/// store root where a working-memory root is expected. This type still does not
/// prove that the protected owner produced or endorsed the commitment; trusted
/// owner acquisition remains a separate adapter responsibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceStateCommitmentV1 {
    resource: CanonicalResourceV1,
    digest: Digest32,
}

impl ResourceStateCommitmentV1 {
    /// Canonical working-memory state commitment.
    pub fn working_memory(
        contents: &[ContinuousHV],
        ticks: &[u64],
        sources: &[MemorySource],
        verified: &[bool],
        metadata: &[HashMap<String, String>],
        capacity: usize,
    ) -> Result<Self, StateCommitmentError> {
        Ok(Self {
            resource: CanonicalResourceV1::WorkingMemory,
            digest: working_memory_state_digest_v1(
                contents, ticks, sources, verified, metadata, capacity,
            )?,
        })
    }

    /// Canonical active cognitive-state commitment.
    pub fn active_cognitive_state(
        holocell: &LiquidHolocell,
        current_thought: &ContinuousHV,
    ) -> Self {
        Self {
            resource: CanonicalResourceV1::ActiveCognitiveState,
            digest: active_state_digest_v1(holocell, current_thought),
        }
    }

    /// Canonical ordered goal-store commitment.
    pub fn goal_store(goals: &[GoalRecordView<'_>]) -> Self {
        Self {
            resource: CanonicalResourceV1::GoalStore,
            digest: goal_store_state_digest_v1(goals),
        }
    }

    /// Canonical first-tranche affect-state commitment.
    pub fn affective_state(emotional_valence: f32) -> Self {
        Self {
            resource: CanonicalResourceV1::AffectiveState,
            digest: affect_state_digest_v1(emotional_valence),
        }
    }

    /// Canonical commitment to an explicitly supplied ordered graduation queue.
    ///
    /// This does not bypass `MemoryCoordinator` privacy and therefore does not by
    /// itself establish owner provenance. Live runtime use must obtain the queue
    /// commitment from a narrow owner-side commitment seam rather than exporting
    /// private queue contents merely to call this helper.
    pub fn graduation_queue_reference(events: &[GraduationEvent]) -> Self {
        Self {
            resource: CanonicalResourceV1::GraduationQueue,
            digest: graduation_queue_state_digest_v1(events),
        }
    }

    /// Resource class committed by this root.
    pub const fn resource(&self) -> CanonicalResourceV1 {
        self.resource
    }

    /// Stable resource name committed by this root.
    pub const fn resource_name(&self) -> &'static str {
        self.resource.resource_name()
    }

    /// Return the generic digest only when the caller names the same resource.
    ///
    /// This is the preferred conversion at a trusted adapter boundary. It makes
    /// resource substitution an explicit error instead of relying on comments
    /// around otherwise interchangeable `Digest32` values.
    pub fn digest_for(
        &self,
        expected: CanonicalResourceV1,
    ) -> Result<Digest32, ResourceCommitmentMismatch> {
        if self.resource != expected {
            return Err(ResourceCommitmentMismatch {
                expected,
                observed: self.resource,
            });
        }
        Ok(self.digest)
    }
}

/// A typed resource-state commitment was supplied for the wrong protected owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceCommitmentMismatch {
    /// Resource requested by the trusted adapter.
    pub expected: CanonicalResourceV1,
    /// Resource carried by the supplied commitment.
    pub observed: CanonicalResourceV1,
}

impl fmt::Display for ResourceCommitmentMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "resource-state commitment is for {}, expected {}",
            self.observed.resource_name(),
            self.expected.resource_name()
        )
    }
}

impl Error for ResourceCommitmentMismatch {}

#[cfg(test)]
mod tests {
    use super::*;

    fn hv(a: f32, b: f32) -> ContinuousHV {
        ContinuousHV::from_values(vec![a, b])
    }

    #[test]
    fn resource_binding_rejects_cross_resource_substitution() {
        let commitment = ResourceStateCommitmentV1::affective_state(0.25);
        assert_eq!(
            commitment.digest_for(CanonicalResourceV1::WorkingMemory),
            Err(ResourceCommitmentMismatch {
                expected: CanonicalResourceV1::WorkingMemory,
                observed: CanonicalResourceV1::AffectiveState,
            })
        );
    }

    #[test]
    fn resource_binding_allows_exact_resource() {
        let commitment = ResourceStateCommitmentV1::affective_state(0.25);
        assert!(
            commitment
                .digest_for(CanonicalResourceV1::AffectiveState)
                .is_ok()
        );
    }

    #[test]
    fn working_memory_wrapper_preserves_alignment_failure() {
        let contents = vec![hv(0.1, 0.2)];
        let sources = vec![MemorySource::Internal];
        let verified = vec![false];
        let metadata = vec![HashMap::new()];

        assert!(matches!(
            ResourceStateCommitmentV1::working_memory(
                &contents,
                &[],
                &sources,
                &verified,
                &metadata,
                4,
            ),
            Err(StateCommitmentError::WorkingMemoryParallelLengthMismatch { .. })
        ));
    }
}
