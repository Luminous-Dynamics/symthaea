// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Audit-only runtime observation support for CogSec shadow mode.
//!
//! This crate deliberately does **not** own cognitive state and does not store
//! authorization or commit permits. It provides one optional observer object
//! that future `ContinuousMind` hooks can use to:
//!
//! - identify the protected resource being observed;
//! - append typed shadow evidence through one bounded ledger;
//! - keep qualification-run correlation separate from authority.
//!
//! The observer does **not** issue `ResourceVersion`. Early shadow continuity
//! comes from owner-issued evidence `EventId`s and explicit causal parents.
//! Authoritative `ResourceVersion` is reserved for the actual protected state
//! owner once that owner boundary is instrumented.
//!
//! Absence of this object is the default disabled state. Presence means
//! **audit/shadow observation only**; this crate has no enforcement mode.

#![forbid(unsafe_code)]

use symthaea_cogsec::ResourceId;
use symthaea_cogsec_evidence::{
    AppendError, EvidenceCompleteness, EvidenceLedgerSnapshot, EventId, LedgerInitError, LedgerStats,
    QualificationManifest, ShadowEventDraft, ShadowEventLedger,
};
use symthaea_evidence_plane::RunId;
use thiserror::Error;

const WORKING_MEMORY_RESOURCE: &str = "mind/working-memory";
const ACTIVE_COGNITIVE_STATE_RESOURCE: &str = "mind/active-cognitive-state";
const GOAL_STORE_RESOURCE: &str = "mind/goals";
const AFFECTIVE_STATE_RESOURCE: &str = "mind/affect";
const GRADUATION_BOUNDARY_RESOURCE: &str = "mind/memory/graduation";

/// First-tranche protected domains observed by CogSec shadow mode.
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
    /// Resource identity for one protected domain.
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

/// Failure to append through the observer-bound shadow evidence boundary.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ShadowAppendError {
    /// A qualification observer expects one run identity but the draft omitted it.
    #[error("shadow event is missing the observer-bound qualification run id")]
    MissingRunId,
    /// The draft carries a run identity different from the observer-bound run.
    #[error("shadow event run id does not match the observer-bound qualification run")]
    RunIdMismatch,
    /// The observer has no run grouping but the draft attempted to inject one.
    #[error("shadow event supplied an unexpected qualification run id")]
    UnexpectedRunId,
    /// The underlying bounded ledger rejected the event.
    #[error(transparent)]
    Ledger(#[from] AppendError),
}

/// Optional audit-only observer for one `ContinuousMind` instance.
///
/// This object owns evidence bookkeeping only. It deliberately has no mutable
/// cognitive-state reference and no method that issues authoritative state
/// freshness. `ResourceVersion` must come from the actual protected owner.
#[derive(Debug)]
pub struct ShadowRuntimeObserver {
    run_id: Option<RunId>,
    resource_ids: ShadowResourceIds,
    ledger: ShadowEventLedger,
}

impl ShadowRuntimeObserver {
    /// Construct an audit-only observer with fully preallocated retained-event storage.
    pub fn try_new_preallocated(
        ledger_epoch: u64,
        event_capacity: usize,
        manifest: QualificationManifest,
        run_id: Option<RunId>,
    ) -> Result<Self, LedgerInitError> {
        Ok(Self {
            run_id,
            resource_ids: ShadowResourceIds::default(),
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

    /// Stable protected resource identity for one observed domain.
    pub fn resource_id(&self, resource: ShadowResource) -> &ResourceId {
        self.resource_ids.get(resource)
    }

    /// Append one event through the observer-bound evidence ledger.
    ///
    /// Run grouping is checked rather than silently rewritten. The observer does
    /// not fill or reinterpret `resource_version_*`: those fields are reserved
    /// for values supplied by the actual protected state owner.
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

    fn observer(run_id: Option<RunId>) -> ShadowRuntimeObserver {
        ShadowRuntimeObserver::try_new_preallocated(7, 16, manifest(), run_id).unwrap()
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
    fn resource_identity_is_stable_and_domain_specific() {
        let observer = observer(None);
        assert_eq!(
            observer
                .resource_id(ShadowResource::WorkingMemory)
                .0
                .as_str(),
            WORKING_MEMORY_RESOURCE
        );
        assert_eq!(
            observer
                .resource_id(ShadowResource::ActiveCognitiveState)
                .0
                .as_str(),
            ACTIVE_COGNITIVE_STATE_RESOURCE
        );
        assert_ne!(
            observer.resource_id(ShadowResource::GoalStore),
            observer.resource_id(ShadowResource::AffectiveState)
        );
    }

    #[test]
    fn event_storage_is_preallocated_before_runtime_use() {
        let observer = observer(None);
        assert!(observer.event_storage_fully_preallocated());
    }

    #[test]
    fn observer_bound_run_id_must_match_without_becoming_authority() {
        let expected = RunId::new("scenario-s0");
        let mut observer = observer(Some(expected.clone()));

        let missing = observer.try_append(ingress(None));
        assert_eq!(missing, Err(ShadowAppendError::MissingRunId));

        let mismatch = observer.try_append(ingress(Some(RunId::new("scenario-s1"))));
        assert_eq!(mismatch, Err(ShadowAppendError::RunIdMismatch));

        let event_id = observer.try_append(ingress(Some(expected))).unwrap();
        assert_eq!(event_id.ledger_epoch, 7);
        assert_eq!(event_id.sequence, 1);
    }

    #[test]
    fn observer_does_not_populate_authoritative_resource_versions() {
        let mut observer = observer(None);
        let event_id = observer.try_append(ingress(None)).unwrap();
        let snapshot = observer.snapshot();
        let event = snapshot
            .events
            .iter()
            .find(|event| event.event_id == event_id)
            .unwrap();
        assert!(event.resource_version_before.is_none());
        assert!(event.resource_version_after.is_none());
    }
}
