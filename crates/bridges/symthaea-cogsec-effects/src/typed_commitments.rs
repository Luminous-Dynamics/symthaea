// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Strongly typed wrappers around canonical CogSec commitments.
//!
//! The reference monitor intentionally accepts the generic [`Digest32`] type so
//! cryptography stays outside its logical TCB. Application code should not use
//! that genericity as permission to mix unrelated commitments. This module keeps
//! effect identity, effect taxonomy, and protected-resource state identity bound
//! until the trusted adapter deliberately unwraps them at the monitor boundary.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use symthaea_cogsec::{Digest32, MutationKind};
use symthaea_core::hdc::LiquidHolocell;
use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_memory::{GraduationEvent, MemorySource};

use crate::{
    CognitiveEffectV1, GoalRecordView, StateCommitmentError, WorkingMemoryItemView,
    active_state_digest_v1, affect_state_digest_v1, effect_digest_v1,
    goal_store_state_digest_v1, graduation_queue_state_digest_v1,
    working_memory_state_digest_v1,
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

/// Exact semantic effect families represented by the v1 canonical encoder.
///
/// This is deliberately more precise than frozen K0 [`MutationKind`]. In
/// particular, replacement/eviction and active-state influence remain explicit
/// unresolved taxonomy classes rather than being coerced into a convenient K0
/// variant. That keeps #201 visible in the type system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CanonicalEffectClassV1 {
    /// Admit a WM item without eviction.
    WorkingMemoryAdmit,
    /// Admit a WM item while evicting/replacing an exact existing item.
    WorkingMemoryReplace,
    /// Enqueue one separate persistence/graduation candidate.
    GraduationEnqueue,
    /// Replace the active Holocell/current-thought state.
    ActiveStateReplace,
    /// Append one exact active goal record.
    GoalActivate,
    /// Apply one exact emotional-valence transition.
    AffectSet,
}

impl CanonicalEffectClassV1 {
    /// Protected resource materially changed by this exact effect class.
    pub const fn resource(self) -> CanonicalResourceV1 {
        match self {
            Self::WorkingMemoryAdmit | Self::WorkingMemoryReplace => {
                CanonicalResourceV1::WorkingMemory
            }
            Self::GraduationEnqueue => CanonicalResourceV1::GraduationQueue,
            Self::ActiveStateReplace => CanonicalResourceV1::ActiveCognitiveState,
            Self::GoalActivate => CanonicalResourceV1::GoalStore,
            Self::AffectSet => CanonicalResourceV1::AffectiveState,
        }
    }

    /// Exact frozen-K0 mutation mapping when one exists without semantic coercion.
    ///
    /// `None` is intentional for compound WM replacement and active-state
    /// influence. Those classes require the K0.1 taxonomy work in #201 before
    /// they can participate in a strong all-stage authorization claim.
    pub const fn k0_mutation_kind(self) -> Option<MutationKind> {
        match self {
            Self::WorkingMemoryAdmit => Some(MutationKind::WorkingMemoryAdmission),
            Self::WorkingMemoryReplace => None,
            Self::GraduationEnqueue => Some(MutationKind::PersistentMemoryCommit),
            Self::ActiveStateReplace => None,
            Self::GoalActivate => Some(MutationKind::GoalActivation),
            Self::AffectSet => Some(MutationKind::Affect),
        }
    }
}

/// Canonical commitment to one exact cognitive effect.
///
/// This is ordinary effect-identity data, not a permit, owner root, signature,
/// trusted fact, or authorization. Both the digest and precise effect class are
/// private so callers cannot manufacture inconsistent taxonomy/hash pairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EffectCommitmentV1 {
    class: CanonicalEffectClassV1,
    digest: Digest32,
}

impl EffectCommitmentV1 {
    fn from_effect(class: CanonicalEffectClassV1, effect: CognitiveEffectV1) -> Self {
        Self {
            class,
            digest: effect_digest_v1(&effect),
        }
    }

    /// Commit to an admission-only WM effect.
    pub fn working_memory_admit(item: WorkingMemoryItemView<'_>, insertion_index: u64) -> Self {
        Self::from_effect(
            CanonicalEffectClassV1::WorkingMemoryAdmit,
            CognitiveEffectV1::working_memory_admit(item, insertion_index),
        )
    }

    /// Commit to an exact compound WM replacement/eviction effect.
    pub fn working_memory_replace(
        admitted: WorkingMemoryItemView<'_>,
        admitted_index: u64,
        evicted: WorkingMemoryItemView<'_>,
        evicted_index: u64,
        evicted_steps_survived: u64,
    ) -> Self {
        Self::from_effect(
            CanonicalEffectClassV1::WorkingMemoryReplace,
            CognitiveEffectV1::working_memory_replace(
                admitted,
                admitted_index,
                evicted,
                evicted_index,
                evicted_steps_survived,
            ),
        )
    }

    /// Commit to one exact graduation-queue enqueue effect.
    #[allow(clippy::too_many_arguments)]
    pub fn graduation_enqueue(
        content: &ContinuousHV,
        label: impl Into<String>,
        steps_survived: u64,
        final_activation: f64,
        psi: f64,
        coherence: f64,
        source: MemorySource,
        is_verified: bool,
    ) -> Self {
        Self::from_effect(
            CanonicalEffectClassV1::GraduationEnqueue,
            CognitiveEffectV1::graduation_enqueue(
                content,
                label,
                steps_survived,
                final_activation,
                psi,
                coherence,
                source,
                is_verified,
            ),
        )
    }

    /// Commit to the exact active Holocell/current-thought replacement.
    pub fn active_state_replace(
        before_holocell: &LiquidHolocell,
        before_current_thought: &ContinuousHV,
        after_holocell: &LiquidHolocell,
        after_current_thought: &ContinuousHV,
    ) -> Self {
        Self::from_effect(
            CanonicalEffectClassV1::ActiveStateReplace,
            CognitiveEffectV1::active_state_replace(
                before_holocell,
                before_current_thought,
                after_holocell,
                after_current_thought,
            ),
        )
    }

    /// Commit to one exact goal-store append.
    pub fn goal_activate(
        goal_id: impl Into<String>,
        description: impl Into<String>,
        embedding: &ContinuousHV,
        priority: f32,
        progress: f32,
        is_active: bool,
    ) -> Self {
        Self::from_effect(
            CanonicalEffectClassV1::GoalActivate,
            CognitiveEffectV1::goal_activate(
                goal_id,
                description,
                embedding,
                priority,
                progress,
                is_active,
            ),
        )
    }

    /// Commit to one exact affect transition.
    pub fn affect_set(before: f32, delta: f32, after: f32) -> Self {
        Self::from_effect(
            CanonicalEffectClassV1::AffectSet,
            CognitiveEffectV1::affect_set(before, delta, after),
        )
    }

    /// Precise canonical effect class bound to this digest.
    pub const fn class(&self) -> CanonicalEffectClassV1 {
        self.class
    }

    /// Protected resource changed by this effect.
    pub const fn resource(&self) -> CanonicalResourceV1 {
        self.class.resource()
    }

    /// Exact frozen-K0 mutation class when one exists without semantic coercion.
    pub const fn k0_mutation_kind(&self) -> Option<MutationKind> {
        self.class.k0_mutation_kind()
    }

    /// Return the generic monitor digest explicitly.
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
    /// This is a **reference encoder**, not an owner-read API. Live runtime use
    /// must obtain the queue commitment through a narrow `MemoryCoordinator`
    /// owner-side commitment seam rather than exporting private queue contents.
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

/// One effect commitment paired with the exact protected-resource pre-state it targets.
///
/// Construction fails if the effect class and state commitment name different
/// resources. This remains ordinary canonical data; it is not an owner-issued
/// trusted fact until the trusted adapter independently establishes the state
/// commitment at the protected owner boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CanonicalTransitionCommitmentV1 {
    class: CanonicalEffectClassV1,
    resource: CanonicalResourceV1,
    effect_digest: Digest32,
    resource_state_digest: Digest32,
}

impl CanonicalTransitionCommitmentV1 {
    /// Bind one exact effect to a pre-state commitment for the same resource.
    pub fn bind(
        effect: EffectCommitmentV1,
        state: ResourceStateCommitmentV1,
    ) -> Result<Self, TransitionCommitmentMismatch> {
        let expected = effect.resource();
        if state.resource() != expected {
            return Err(TransitionCommitmentMismatch {
                effect_resource: expected,
                state_resource: state.resource(),
            });
        }
        Ok(Self {
            class: effect.class(),
            resource: expected,
            effect_digest: effect.digest(),
            resource_state_digest: state
                .digest_for(expected)
                .expect("resource equality checked above"),
        })
    }

    /// Exact effect class.
    pub const fn effect_class(&self) -> CanonicalEffectClassV1 {
        self.class
    }

    /// Exact protected resource.
    pub const fn resource(&self) -> CanonicalResourceV1 {
        self.resource
    }

    /// Frozen-K0 mutation kind if the exact effect has one.
    pub const fn k0_mutation_kind(&self) -> Option<MutationKind> {
        self.class.k0_mutation_kind()
    }

    /// Exact effect commitment for `MutationRequest::mutation_digest`.
    pub const fn effect_digest(&self) -> Digest32 {
        self.effect_digest
    }

    /// Exact pre-state commitment for the protected resource.
    pub const fn resource_state_digest(&self) -> Digest32 {
        self.resource_state_digest
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

/// Effect and pre-state commitments name different protected resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransitionCommitmentMismatch {
    /// Resource implied by the exact effect class.
    pub effect_resource: CanonicalResourceV1,
    /// Resource carried by the state commitment.
    pub state_resource: CanonicalResourceV1,
}

impl fmt::Display for TransitionCommitmentMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "effect targets {}, but state commitment is for {}",
            self.effect_resource.resource_name(),
            self.state_resource.resource_name()
        )
    }
}

impl Error for TransitionCommitmentMismatch {}

#[cfg(test)]
mod tests {
    use super::*;

    fn hv(a: f32, b: f32) -> ContinuousHV {
        ContinuousHV::from_values(vec![a, b])
    }

    #[test]
    fn unresolved_compound_and_active_effects_have_no_k0_mapping() {
        let metadata = HashMap::new();
        let admitted_hv = hv(0.1, 0.2);
        let evicted_hv = hv(0.3, 0.4);
        let admitted = WorkingMemoryItemView {
            content: &admitted_hv,
            arrival_tick: 7,
            source: MemorySource::UserInteraction,
            verified: false,
            metadata: &metadata,
        };
        let evicted = WorkingMemoryItemView {
            content: &evicted_hv,
            arrival_tick: 2,
            source: MemorySource::Internal,
            verified: false,
            metadata: &metadata,
        };
        let replacement = EffectCommitmentV1::working_memory_replace(admitted, 3, evicted, 0, 5);
        assert_eq!(replacement.k0_mutation_kind(), None);

        let before = LiquidHolocell::new(1);
        let mut after = before.clone();
        let input = hv(0.5, 0.6);
        after.step(&input, 0.1);
        let active = EffectCommitmentV1::active_state_replace(
            &before,
            &before.state,
            &after,
            &after.state,
        );
        assert_eq!(active.k0_mutation_kind(), None);
    }

    #[test]
    fn exact_k0_mappings_remain_explicit() {
        let metadata = HashMap::new();
        let item_hv = hv(0.1, 0.2);
        let item = WorkingMemoryItemView {
            content: &item_hv,
            arrival_tick: 7,
            source: MemorySource::UserInteraction,
            verified: false,
            metadata: &metadata,
        };
        assert_eq!(
            EffectCommitmentV1::working_memory_admit(item, 0).k0_mutation_kind(),
            Some(MutationKind::WorkingMemoryAdmission)
        );
        assert_eq!(
            EffectCommitmentV1::affect_set(0.0, 0.2, 0.06).k0_mutation_kind(),
            Some(MutationKind::Affect)
        );
    }

    #[test]
    fn transition_binding_rejects_effect_state_resource_mismatch() {
        let effect = EffectCommitmentV1::affect_set(0.0, 0.2, 0.06);
        let wrong_state = ResourceStateCommitmentV1::goal_store(&[]);
        assert_eq!(
            CanonicalTransitionCommitmentV1::bind(effect, wrong_state),
            Err(TransitionCommitmentMismatch {
                effect_resource: CanonicalResourceV1::AffectiveState,
                state_resource: CanonicalResourceV1::GoalStore,
            })
        );
    }

    #[test]
    fn transition_binding_preserves_effect_and_state_identity() {
        let effect = EffectCommitmentV1::affect_set(0.0, 0.2, 0.06);
        let effect_digest = effect.digest();
        let state = ResourceStateCommitmentV1::affective_state(0.0);
        let state_digest = state
            .digest_for(CanonicalResourceV1::AffectiveState)
            .unwrap();
        let transition = CanonicalTransitionCommitmentV1::bind(effect, state).unwrap();

        assert_eq!(transition.resource(), CanonicalResourceV1::AffectiveState);
        assert_eq!(transition.effect_digest(), effect_digest);
        assert_eq!(transition.resource_state_digest(), state_digest);
        assert_eq!(transition.k0_mutation_kind(), Some(MutationKind::Affect));
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
