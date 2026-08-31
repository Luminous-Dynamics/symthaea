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
//! - keep qualification-run correlation separate from authority;
//! - make the assurance profile explicit at construction time;
//! - bind each typed runtime stage to its canonical observed resource.
//!
//! The observer does **not** issue `ResourceVersion`. Early shadow continuity
//! comes from owner-issued evidence `EventId`s and explicit causal parents.
//! Authoritative `ResourceVersion` is reserved for the actual protected state
//! owner once that owner boundary is instrumented.
//!
//! `ObserverOnly` and `OwnerAware` are intentionally distinct profiles:
//! observer-only evidence must not carry authoritative resource versions,
//! whereas owner-aware evidence must use a manifest that requires them.
//!
//! Resource binding is attribution, not authorization. Knowing that a
//! `GoalActivationObserved` event belongs to `mind/goals` does not grant any
//! permission to mutate that resource.
//!
//! Absence of this object is the default disabled state. Presence means
//! **audit/shadow observation only**; this crate has no enforcement mode.

#![forbid(unsafe_code)]

use symthaea_cogsec::ResourceId;
use symthaea_cogsec_evidence::{
    AppendError, EvidenceCompleteness, EvidenceLedgerSnapshot, EventId, LedgerInitError, LedgerStats,
    QualificationManifest, ShadowEventDraft, ShadowEventKind, ShadowEventLedger,
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

impl ShadowResource {
    /// Canonical protected resource for a typed shadow stage.
    ///
    /// `IngressObserved` has no protected mutation resource and
    /// `EvidenceGapObserved` may describe loss from more than one resource, so
    /// those kinds intentionally return `None`.
    pub const fn for_event_kind(kind: ShadowEventKind) -> Option<Self> {
        match kind {
            ShadowEventKind::WorkingMemoryAdmissionEvaluated
            | ShadowEventKind::WorkingMemoryAdmissionObserved
            | ShadowEventKind::WorkingMemoryEvictionObserved
            | ShadowEventKind::DreamMergeEvaluated
            | ShadowEventKind::DreamMergeObserved => Some(Self::WorkingMemory),
            ShadowEventKind::GraduationEvaluated | ShadowEventKind::GraduationObserved => {
                Some(Self::GraduationBoundary)
            }
            ShadowEventKind::WorkingStateInfluenceEvaluated
            | ShadowEventKind::WorkingStateInfluenceObserved => Some(Self::ActiveCognitiveState),
            ShadowEventKind::GoalActivationEvaluated
            | ShadowEventKind::GoalActivationObserved => Some(Self::GoalStore),
            ShadowEventKind::AffectMutationEvaluated
            | ShadowEventKind::AffectMutationObserved => Some(Self::AffectiveState),
            ShadowEventKind::IngressObserved | ShadowEventKind::EvidenceGapObserved => None,
        }
    }
}

/// Explicit assurance profile for one shadow observer instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ShadowAssuranceProfile {
    /// Audit-only observation. Authoritative protected-owner freshness is not
    /// claimed and `resource_version_*` fields are forbidden on appended events.
    ObserverOnly,
    /// Owner-aware observation. The manifest must require owner-issued
    /// `ResourceVersion` on observed resource mutations. The observer records
    /// supplied versions but never creates them.
    OwnerAware,
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

/// Failure to construct a shadow observer with a contradictory assurance profile.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ShadowObserverInitError {
    /// Observer-only mode cannot claim owner-issued resource-version coverage.
    #[error("observer-only shadow profile cannot require authoritative resource versions")]
    ObserverOnlyRequiresResourceVersions,
    /// Owner-aware mode must require authoritative resource-version coverage.
    #[error("owner-aware shadow profile must require authoritative resource versions")]
    OwnerAwareOmitsResourceVersions,
    /// The bounded ledger could not establish its requested retained-event capacity.
    #[error(transparent)]
    Ledger(#[from] LedgerInitError),
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
    /// Observer-only mode may not carry authoritative owner freshness.
    #[error("observer-only shadow event {kind:?} supplied resource-version fields")]
    ObserverOnlyResourceVersionForbidden {
        /// Event kind that attempted to carry owner-version evidence.
        kind: ShadowEventKind,
    },
    /// A resource-bearing stage omitted its canonical protected resource.
    #[error("shadow event {kind:?} omitted canonical resource {expected:?}")]
    RequiredResourceMissing {
        /// Stage whose resource binding was missing.
        kind: ShadowEventKind,
        /// Canonical resource identity expected for the stage.
        expected: ResourceId,
    },
    /// A resource-bearing stage was attributed to the wrong protected resource.
    #[error(
        "shadow event {kind:?} targeted {observed:?}, expected canonical resource {expected:?}"
    )]
    ResourceBindingMismatch {
        /// Stage whose resource binding was inconsistent.
        kind: ShadowEventKind,
        /// Canonical resource identity expected for the stage.
        expected: ResourceId,
        /// Resource identity supplied by the caller.
        observed: ResourceId,
    },
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
    profile: ShadowAssuranceProfile,
    run_id: Option<RunId>,
    resource_ids: ShadowResourceIds,
    ledger: ShadowEventLedger,
}

impl ShadowRuntimeObserver {
    /// Construct an audit-only observer with fully preallocated retained-event storage.
    ///
    /// The profile and manifest must agree. This prevents an observer-only run
    /// from claiming owner-issued freshness and prevents an owner-aware run from
    /// silently omitting that qualification requirement.
    pub fn try_new_preallocated(
        profile: ShadowAssuranceProfile,
        ledger_epoch: u64,
        event_capacity: usize,
        manifest: QualificationManifest,
        run_id: Option<RunId>,
    ) -> Result<Self, ShadowObserverInitError> {
        match profile {
            ShadowAssuranceProfile::ObserverOnly if manifest.require_resource_versions => {
                return Err(ShadowObserverInitError::ObserverOnlyRequiresResourceVersions);
            }
            ShadowAssuranceProfile::OwnerAware if !manifest.require_resource_versions => {
                return Err(ShadowObserverInitError::OwnerAwareOmitsResourceVersions);
            }
            _ => {}
        }

        Ok(Self {
            profile,
            run_id,
            resource_ids: ShadowResourceIds::default(),
            ledger: ShadowEventLedger::try_new_preallocated(
                ledger_epoch,
                event_capacity,
                manifest,
            )?,
        })
    }

    /// Explicit assurance profile for this observer instance.
    pub const fn profile(&self) -> ShadowAssuranceProfile {
        self.profile
    }

    /// Qualification grouping identity. This is never authorization authority.
    pub fn run_id(&self) -> Option<&RunId> {
        self.run_id.as_ref()
    }

    /// Stable protected resource identity for one observed domain.
    pub fn resource_id(&self, resource: ShadowResource) -> &ResourceId {
        self.resource_ids.get(resource)
    }

    /// Canonical resource identity expected for one typed shadow stage.
    pub fn expected_resource_id(&self, kind: ShadowEventKind) -> Option<&ResourceId> {
        ShadowResource::for_event_kind(kind).map(|resource| self.resource_ids.get(resource))
    }

    fn validate_resource_binding(&self, draft: &ShadowEventDraft) -> Result<(), ShadowAppendError> {
        let Some(expected) = self.expected_resource_id(draft.kind) else {
            return Ok(());
        };

        match draft.resource.as_ref() {
            None => Err(ShadowAppendError::RequiredResourceMissing {
                kind: draft.kind,
                expected: expected.clone(),
            }),
            Some(observed) if observed != expected => {
                Err(ShadowAppendError::ResourceBindingMismatch {
                    kind: draft.kind,
                    expected: expected.clone(),
                    observed: observed.clone(),
                })
            }
            Some(_) => Ok(()),
        }
    }

    /// Append one event through the observer-bound evidence ledger.
    ///
    /// Run grouping is checked rather than silently rewritten. Resource-bearing
    /// stages must also target the canonical resource identity owned by this
    /// observer's local registry. This is an attribution check, not permission.
    ///
    /// In [`ShadowAssuranceProfile::ObserverOnly`] mode the append boundary also
    /// rejects any supplied `resource_version_*` values, preventing audit data
    /// from accidentally presenting itself as protected-owner freshness.
    ///
    /// In [`ShadowAssuranceProfile::OwnerAware`] mode owner-version fields are
    /// passed through unchanged; the underlying manifest/ledger validates the
    /// required version semantics for observed mutations.
    pub fn try_append(&mut self, draft: ShadowEventDraft) -> Result<EventId, ShadowAppendError> {
        match (self.run_id.as_ref(), draft.run_id.as_ref()) {
            (Some(expected), Some(observed)) if expected != observed => {
                return Err(ShadowAppendError::RunIdMismatch);
            }
            (Some(_), None) => return Err(ShadowAppendError::MissingRunId),
            (None, Some(_)) => return Err(ShadowAppendError::UnexpectedRunId),
            _ => {}
        }

        if self.profile == ShadowAssuranceProfile::ObserverOnly
            && (draft.resource_version_before.is_some() || draft.resource_version_after.is_some())
        {
            return Err(ShadowAppendError::ObserverOnlyResourceVersionForbidden {
                kind: draft.kind,
            });
        }

        self.validate_resource_binding(&draft)?;

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
        EvidenceConfidentiality, IngressClass, PrincipalContext, ResourceVersion,
        ShadowEventPayload,
    };

    fn observer_manifest() -> QualificationManifest {
        QualificationManifest::new([ShadowEventKind::IngressObserved], [])
    }

    fn owner_aware_manifest() -> QualificationManifest {
        QualificationManifest::new([ShadowEventKind::IngressObserved], [])
            .with_required_resource_versions(true)
    }

    fn observer(run_id: Option<RunId>) -> ShadowRuntimeObserver {
        ShadowRuntimeObserver::try_new_preallocated(
            ShadowAssuranceProfile::ObserverOnly,
            7,
            16,
            observer_manifest(),
            run_id,
        )
        .unwrap()
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

    fn mutation_observation(kind: ShadowEventKind, resource: Option<ResourceId>) -> ShadowEventDraft {
        ShadowEventDraft {
            kind,
            proposal_id: None,
            transaction_id: None,
            principals: PrincipalContext::default(),
            resource,
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
            run_id: None,
            confidentiality: EvidenceConfidentiality::LocalPrivate,
            payload: ShadowEventPayload::MutationObserved { applied: true },
        }
    }

    #[test]
    fn observer_only_profile_rejects_owner_version_requirement_at_init() {
        let result = ShadowRuntimeObserver::try_new_preallocated(
            ShadowAssuranceProfile::ObserverOnly,
            7,
            16,
            owner_aware_manifest(),
            None,
        );
        assert_eq!(
            result.unwrap_err(),
            ShadowObserverInitError::ObserverOnlyRequiresResourceVersions
        );
    }

    #[test]
    fn owner_aware_profile_requires_owner_version_requirement_at_init() {
        let result = ShadowRuntimeObserver::try_new_preallocated(
            ShadowAssuranceProfile::OwnerAware,
            7,
            16,
            observer_manifest(),
            None,
        );
        assert_eq!(
            result.unwrap_err(),
            ShadowObserverInitError::OwnerAwareOmitsResourceVersions
        );
    }

    #[test]
    fn assurance_profile_is_explicit_and_stable() {
        let observer = observer(None);
        assert_eq!(observer.profile(), ShadowAssuranceProfile::ObserverOnly);

        let owner_aware = ShadowRuntimeObserver::try_new_preallocated(
            ShadowAssuranceProfile::OwnerAware,
            8,
            16,
            owner_aware_manifest(),
            None,
        )
        .unwrap();
        assert_eq!(owner_aware.profile(), ShadowAssuranceProfile::OwnerAware);
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
    fn stage_resource_mapping_is_explicit_for_every_protected_stage() {
        let cases = [
            (
                ShadowEventKind::WorkingMemoryAdmissionEvaluated,
                ShadowResource::WorkingMemory,
            ),
            (
                ShadowEventKind::WorkingMemoryAdmissionObserved,
                ShadowResource::WorkingMemory,
            ),
            (
                ShadowEventKind::WorkingMemoryEvictionObserved,
                ShadowResource::WorkingMemory,
            ),
            (
                ShadowEventKind::GraduationEvaluated,
                ShadowResource::GraduationBoundary,
            ),
            (
                ShadowEventKind::GraduationObserved,
                ShadowResource::GraduationBoundary,
            ),
            (
                ShadowEventKind::WorkingStateInfluenceEvaluated,
                ShadowResource::ActiveCognitiveState,
            ),
            (
                ShadowEventKind::WorkingStateInfluenceObserved,
                ShadowResource::ActiveCognitiveState,
            ),
            (
                ShadowEventKind::GoalActivationEvaluated,
                ShadowResource::GoalStore,
            ),
            (
                ShadowEventKind::GoalActivationObserved,
                ShadowResource::GoalStore,
            ),
            (
                ShadowEventKind::AffectMutationEvaluated,
                ShadowResource::AffectiveState,
            ),
            (
                ShadowEventKind::AffectMutationObserved,
                ShadowResource::AffectiveState,
            ),
            (
                ShadowEventKind::DreamMergeEvaluated,
                ShadowResource::WorkingMemory,
            ),
            (
                ShadowEventKind::DreamMergeObserved,
                ShadowResource::WorkingMemory,
            ),
        ];

        for (kind, expected) in cases {
            assert_eq!(ShadowResource::for_event_kind(kind), Some(expected));
        }
        assert_eq!(
            ShadowResource::for_event_kind(ShadowEventKind::IngressObserved),
            None
        );
        assert_eq!(
            ShadowResource::for_event_kind(ShadowEventKind::EvidenceGapObserved),
            None
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
    fn observer_only_append_rejects_supplied_resource_versions() {
        let mut observer = observer(None);
        let mut draft = ingress(None);
        draft.resource_version_before = Some(ResourceVersion {
            owner_epoch: 3,
            counter: 9,
        });

        assert_eq!(
            observer.try_append(draft),
            Err(ShadowAppendError::ObserverOnlyResourceVersionForbidden {
                kind: ShadowEventKind::IngressObserved,
            })
        );
        assert_eq!(observer.ledger_stats().assigned_sequences, 0);
        assert!(observer.snapshot().events.is_empty());
    }

    #[test]
    fn missing_canonical_resource_is_rejected_before_event_id_allocation() {
        let mut observer = observer(None);
        let draft = mutation_observation(ShadowEventKind::GoalActivationObserved, None);

        assert_eq!(
            observer.try_append(draft),
            Err(ShadowAppendError::RequiredResourceMissing {
                kind: ShadowEventKind::GoalActivationObserved,
                expected: ResourceId(GOAL_STORE_RESOURCE.into()),
            })
        );
        assert_eq!(observer.ledger_stats().assigned_sequences, 0);
    }

    #[test]
    fn wrong_canonical_resource_is_rejected_before_event_id_allocation() {
        let mut observer = observer(None);
        let draft = mutation_observation(
            ShadowEventKind::GoalActivationObserved,
            Some(ResourceId(AFFECTIVE_STATE_RESOURCE.into())),
        );

        assert_eq!(
            observer.try_append(draft),
            Err(ShadowAppendError::ResourceBindingMismatch {
                kind: ShadowEventKind::GoalActivationObserved,
                expected: ResourceId(GOAL_STORE_RESOURCE.into()),
                observed: ResourceId(AFFECTIVE_STATE_RESOURCE.into()),
            })
        );
        assert_eq!(observer.ledger_stats().assigned_sequences, 0);
    }

    #[test]
    fn observer_only_append_does_not_populate_authoritative_resource_versions() {
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
