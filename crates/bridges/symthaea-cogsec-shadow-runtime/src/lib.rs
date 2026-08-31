// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Owner-bound runtime bookkeeping for CogSec shadow mode.
//!
//! This crate deliberately does **not** own cognitive state and does not store
//! authorization or commit permits. It provides one optional observer object
//! that future `ContinuousMind` hooks can use to:
//!
//! - identify the protected owner being observed;
//! - track owner-issued `ResourceVersion`s after legacy mutations occur;
//! - append typed shadow evidence through one bounded ledger;
//! - keep qualification-run correlation separate from authority.
//!
//! Absence of this object is the default disabled state. Presence means
//! **audit/shadow observation only**; this crate has no enforcement mode.

#![forbid(unsafe_code)]

use symthaea_cogsec::ResourceId;
use symthaea_cogsec_evidence::{
    AppendError, EvidenceCompleteness, EvidenceLedgerSnapshot, EventId, LedgerInitError, LedgerStats,
    QualificationManifest, ResourceVersion, ShadowEventDraft, ShadowEventLedger,
};
use symthaea_evidence_plane::RunId;
use thiserror::Error;

const WORKING_MEMORY_RESOURCE: &str = "mind/working-memory";
const ACTIVE_COGNITIVE_STATE_RESOURCE: &str = "mind/active-cognitive-state";
const GOAL_STORE_RESOURCE: &str = "mind/goals";
const AFFECTIVE_STATE_RESOURCE: &str = "mind/affect";
const GRADUATION_BOUNDARY_RESOURCE: &str = "mind/memory/graduation";

/// First-tranche protected owner domains observed by CogSec shadow mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ShadowResource {
    /// Working-memory ordering/content/metadata owner.
    WorkingMemory,
    /// Holocell/current-thought active cognitive state owner.
    ActiveCognitiveState,
    /// Active goal store.
    GoalStore,
    /// Affective/emotional owner.
    AffectiveState,
    /// Working-memory eviction/graduation transition boundary.
    GraduationBoundary,
}

/// Stable resource identities for the first shadow-runtime tranche.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShadowResourceIds {
    working_memory: ResourceId,
    active_cognitive_state: ResourceId,
    goal_store: ResourceId,
    affective_state: ResourceId,
    graduation_boundary: ResourceId,
}

impl Default for ShadowResourceIds {
    fn default() -> Self {
        Self {
            working_memory: ResourceId(WORKING_MEMORY_RESOURCE.into()),
            active_cognitive_state: ResourceId(ACTIVE_COGNITIVE_STATE_RESOURCE.into()),
            goal_store: ResourceId(GOAL_STORE_RESOURCE.into()),
            affective_state: ResourceId(AFFECTIVE_STATE_RESOURCE.into()),
            graduation_boundary: ResourceId(GRADUATION_BOUNDARY_RESOURCE.into()),
        }
    }
}

impl ShadowResourceIds {
    /// Resource identity for one protected owner domain.
    pub fn get(&self, resource: ShadowResource) -> &ResourceId {
        match resource {
            ShadowResource::WorkingMemory => &self.working_memory,
            ShadowResource::ActiveCognitiveState => &self.active_cognitive_state,
            ShadowResource::GoalStore => &self.goal_store,
            ShadowResource::AffectiveState => &self.affective_state,
            ShadowResource::GraduationBoundary => &self.graduation_boundary,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VersionSlots {
    working_memory: ResourceVersion,
    active_cognitive_state: ResourceVersion,
    goal_store: ResourceVersion,
    affective_state: ResourceVersion,
    graduation_boundary: ResourceVersion,
}

impl VersionSlots {
    fn new(owner_epoch: u64) -> Self {
        let initial = ResourceVersion {
            owner_epoch,
            counter: 0,
        };
        Self {
            working_memory: initial,
            active_cognitive_state: initial,
            goal_store: initial,
            affective_state: initial,
            graduation_boundary: initial,
        }
    }

    fn get(&self, resource: ShadowResource) -> ResourceVersion {
        match resource {
            ShadowResource::WorkingMemory => self.working_memory,
            ShadowResource::ActiveCognitiveState => self.active_cognitive_state,
            ShadowResource::GoalStore => self.goal_store,
            ShadowResource::AffectiveState => self.affective_state,
            ShadowResource::GraduationBoundary => self.graduation_boundary,
        }
    }

    fn get_mut(&mut self, resource: ShadowResource) -> &mut ResourceVersion {
        match resource {
            ShadowResource::WorkingMemory => &mut self.working_memory,
            ShadowResource::ActiveCognitiveState => &mut self.active_cognitive_state,
            ShadowResource::GoalStore => &mut self.goal_store,
            ShadowResource::AffectiveState => &mut self.affective_state,
            ShadowResource::GraduationBoundary => &mut self.graduation_boundary,
        }
    }
}

/// Exact before/after freshness observation for one legacy owner transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceVersionTransition {
    /// Version before the legacy owner mutation.
    pub before: ResourceVersion,
    /// Version after the legacy owner mutation.
    pub after: ResourceVersion,
}

/// Failure to advance observer freshness state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum VersionObservationError {
    /// The per-owner mutation counter cannot advance further in this owner epoch.
    #[error("resource version exhausted for {resource:?}")]
    CounterExhausted {
        /// Protected resource whose counter overflowed.
        resource: ShadowResource,
    },
}

/// Failure to append through the owner-bound shadow evidence boundary.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ShadowAppendError {
    /// A qualification owner expects one run identity but the draft omitted it.
    #[error("shadow event is missing the owner-bound qualification run id")]
    MissingRunId,
    /// The draft carries a run identity different from the owner-bound run.
    #[error("shadow event run id does not match the owner-bound qualification run")]
    RunIdMismatch,
    /// The owner has no run grouping but the draft attempted to inject one.
    #[error("shadow event supplied an unexpected qualification run id")]
    UnexpectedRunId,
    /// The underlying bounded ledger rejected the event.
    #[error(transparent)]
    Ledger(#[from] AppendError),
}

/// Optional audit-only runtime observer for one `ContinuousMind` instance.
///
/// This holder is not a cognitive state owner. Its version counters advance
/// only **after** future hooks observe that legacy state actually mutated.
/// It contains no `MutationPermit`, `CommitPermit`, policy authority, or
/// enforcement switch.
#[derive(Debug)]
pub struct ShadowRuntimeOwner {
    run_id: Option<RunId>,
    resource_ids: ShadowResourceIds,
    versions: VersionSlots,
    ledger: ShadowEventLedger,
}

impl ShadowRuntimeOwner {
    /// Construct an audit-only owner with fully preallocated retained-event storage.
    ///
    /// `owner_epoch` must be changed when the protected owner is reconstructed or
    /// restored such that old freshness tokens must no longer compare equal.
    pub fn try_new_preallocated(
        ledger_epoch: u64,
        owner_epoch: u64,
        event_capacity: usize,
        manifest: QualificationManifest,
        run_id: Option<RunId>,
    ) -> Result<Self, LedgerInitError> {
        Ok(Self {
            run_id,
            resource_ids: ShadowResourceIds::default(),
            versions: VersionSlots::new(owner_epoch),
            ledger: ShadowEventLedger::try_new_preallocated(
                ledger_epoch,
                event_capacity,
                manifest,
            )?,
        })
    }

    /// Qualification grouping identity. This is never authorization authority.
    pub fn run_id(&self) -> Option<&RunId> {
        self.run_id.as_ref()
    }

    /// Stable protected resource identity for one owner domain.
    pub fn resource_id(&self, resource: ShadowResource) -> &ResourceId {
        self.resource_ids.get(resource)
    }

    /// Current owner-issued freshness version for one protected resource.
    pub fn resource_version(&self, resource: ShadowResource) -> ResourceVersion {
        self.versions.get(resource)
    }

    /// Observe one legacy owner mutation that has already committed.
    ///
    /// This method does not perform the mutation. It advances only the observer's
    /// freshness counter and returns the exact before/after pair to place on the
    /// corresponding `...Observed` event.
    pub fn observe_legacy_mutation(
        &mut self,
        resource: ShadowResource,
    ) -> Result<ResourceVersionTransition, VersionObservationError> {
        let slot = self.versions.get_mut(resource);
        let before = *slot;
        let counter = before
            .counter
            .checked_add(1)
            .ok_or(VersionObservationError::CounterExhausted { resource })?;
        let after = ResourceVersion {
            owner_epoch: before.owner_epoch,
            counter,
        };
        *slot = after;
        Ok(ResourceVersionTransition { before, after })
    }

    /// Observe that a proposed legacy transition did not mutate its protected owner.
    ///
    /// The freshness version deliberately remains unchanged.
    pub fn observe_legacy_noop(&self, resource: ShadowResource) -> ResourceVersionTransition {
        let version = self.versions.get(resource);
        ResourceVersionTransition {
            before: version,
            after: version,
        }
    }

    /// Append one event through the owner-bound evidence ledger.
    ///
    /// Run grouping is checked rather than silently rewritten. This prevents a
    /// future runtime hook from accidentally mixing two qualification scenarios
    /// into one owner while keeping `RunId` explicitly non-authoritative.
    pub fn try_append(&mut self, draft: ShadowEventDraft) -> Result<EventId, ShadowAppendError> {
        match (self.run_id.as_ref(), draft.run_id.as_ref()) {
            (Some(expected), Some(observed)) if expected != observed => {
                return Err(ShadowAppendError::RunIdMismatch);
            }
            (Some(_), None) => return Err(ShadowAppendError::MissingRunId),
            (None, Some(_)) => return Err(ShadowAppendError::UnexpectedRunId),
            _ => {}
        }
        Ok(self.ledger.try_append(draft)?)
    }

    /// Evidence completeness state observed by the bounded ledger.
    pub fn completeness(&self) -> EvidenceCompleteness {
        self.ledger.completeness()
    }

    /// Local append accounting.
    pub fn ledger_stats(&self) -> LedgerStats {
        self.ledger.stats()
    }

    /// Whether retained-event storage was fully reserved before construction returned.
    pub fn event_storage_fully_preallocated(&self) -> bool {
        self.ledger.event_storage_fully_preallocated()
    }

    /// Export ordinary portable evidence data for independent reconciliation.
    ///
    /// The returned snapshot is not authority and is not authenticated merely by
    /// being produced through this API.
    pub fn snapshot(&self) -> EvidenceLedgerSnapshot {
        self.ledger.snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use symthaea_cogsec_evidence::{
        EvidenceConfidentiality, IngressClass, PrincipalContext, ShadowEventKind,
        ShadowEventPayload,
    };

    fn manifest() -> QualificationManifest {
        QualificationManifest::new([ShadowEventKind::IngressObserved], [])
    }

    fn owner(run_id: Option<RunId>) -> ShadowRuntimeOwner {
        ShadowRuntimeOwner::try_new_preallocated(7, 11, 16, manifest(), run_id).unwrap()
    }

    fn ingress(run_id: Option<RunId>) -> ShadowEventDraft {
        ShadowEventDraft {
            kind: ShadowEventKind::IngressObserved,
            proposal_id: None,
            transaction_id: None,
            principals: PrincipalContext::default(),
            resource: None,
            resource_version_before: None,
            resource_version_after: None,
            state_root_before: None,
            state_root_after: None,
            policy_root: None,
            policy_epoch: None,
            authorization_epoch: None,
            revocation_epoch: None,
            causal_parents: BTreeSet::new(),
            cognitive_tick: None,
            run_id,
            confidentiality: EvidenceConfidentiality::LocalPrivate,
            payload: ShadowEventPayload::Ingress {
                ingress_class: IngressClass::LegacyUnclassified,
            },
        }
    }

    #[test]
    fn resource_versions_are_independent_and_start_in_one_owner_epoch() {
        let mut owner = owner(None);
        let wm_before = owner.resource_version(ShadowResource::WorkingMemory);
        let goal_before = owner.resource_version(ShadowResource::GoalStore);
        assert_eq!(wm_before.owner_epoch, 11);
        assert_eq!(wm_before.counter, 0);
        assert_eq!(goal_before, wm_before);

        let transition = owner
            .observe_legacy_mutation(ShadowResource::WorkingMemory)
            .unwrap();
        assert_eq!(transition.before.counter, 0);
        assert_eq!(transition.after.counter, 1);
        assert_eq!(
            owner.resource_version(ShadowResource::WorkingMemory).counter,
            1
        );
        assert_eq!(owner.resource_version(ShadowResource::GoalStore).counter, 0);
    }

    #[test]
    fn noop_observation_never_advances_resource_version() {
        let owner = owner(None);
        let transition = owner.observe_legacy_noop(ShadowResource::AffectiveState);
        assert_eq!(transition.before, transition.after);
        assert_eq!(transition.before.counter, 0);
    }

    #[test]
    fn resource_identity_is_stable_and_owner_specific() {
        let owner = owner(None);
        assert_eq!(
            owner.resource_id(ShadowResource::WorkingMemory).0,
            WORKING_MEMORY_RESOURCE
        );
        assert_eq!(
            owner.resource_id(ShadowResource::ActiveCognitiveState).0,
            ACTIVE_COGNITIVE_STATE_RESOURCE
        );
        assert_ne!(
            owner.resource_id(ShadowResource::GoalStore),
            owner.resource_id(ShadowResource::AffectiveState)
        );
    }

    #[test]
    fn event_storage_is_preallocated_before_runtime_use() {
        let owner = owner(None);
        assert!(owner.event_storage_fully_preallocated());
    }

    #[test]
    fn owner_bound_run_id_must_match_without_becoming_authority() {
        let expected = RunId::new("scenario-s0");
        let mut owner = owner(Some(expected.clone()));

        let missing = owner.try_append(ingress(None));
        assert_eq!(missing, Err(ShadowAppendError::MissingRunId));

        let mismatch = owner.try_append(ingress(Some(RunId::new("scenario-s1"))));
        assert_eq!(mismatch, Err(ShadowAppendError::RunIdMismatch));

        let event_id = owner.try_append(ingress(Some(expected))).unwrap();
        assert_eq!(event_id.ledger_epoch, 7);
        assert_eq!(event_id.sequence, 1);
    }

    #[test]
    fn counter_exhaustion_fails_without_wrapping_security_time() {
        let mut owner = owner(None);
        owner.versions.goal_store.counter = u64::MAX;
        let before = owner.resource_version(ShadowResource::GoalStore);

        let result = owner.observe_legacy_mutation(ShadowResource::GoalStore);
        assert_eq!(
            result,
            Err(VersionObservationError::CounterExhausted {
                resource: ShadowResource::GoalStore
            })
        );
        assert_eq!(owner.resource_version(ShadowResource::GoalStore), before);
    }
}
