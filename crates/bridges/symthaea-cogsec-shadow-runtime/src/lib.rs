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
//! - bind each typed runtime stage to its canonical observed resource;
//! - require opaque same-monitor provenance for live evaluation events.
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
//! Live evaluation provenance is likewise separate from portable evidence.
//! A serializable `MutationReceiptRecord` is ordinary data; only an opaque
//! `MutationReceipt` accepted by the exact `ReferenceMonitor` may enter the
//! runtime ledger as an `...Evaluated` event through this adapter.
//!
//! Absence of this object is the default disabled state. Presence means
//! **audit/shadow observation only**; this crate has no enforcement mode.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use symthaea_cogsec::{MutationKind, MutationReceipt, ReferenceMonitor, ResourceId};
use symthaea_cogsec_evidence::{
    AppendError, CognitiveTick, EvidenceCompleteness, EvidenceConfidentiality,
    EvidenceLedgerSnapshot, EventId, LedgerInitError, LedgerStats, PrincipalContext, ProposalId,
    QualificationManifest, ResourceVersion, ShadowEventDraft, ShadowEventKind, ShadowEventLedger,
    ShadowEventPayload, TransactionId,
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

/// Correlation/context fields allowed when appending a live monitor evaluation.
///
/// Security facts already carried by the opaque monitor receipt are
/// intentionally absent. The runtime adapter derives resource, state root,
/// policy root/epochs, decision payload and exact evaluated effect commitment
/// from the receipt itself, avoiding two independently editable copies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShadowEvaluationDraft {
    /// Typed evaluation stage to record.
    pub kind: ShadowEventKind,
    /// Caller-visible proposal correlation only.
    pub proposal_id: Option<ProposalId>,
    /// Protected-owner transaction identity, if one exists.
    pub transaction_id: Option<TransactionId>,
    /// Role-explicit principal context. This remains evidence metadata; the
    /// authoritative evaluated subject comes from the monitor receipt.
    pub principals: PrincipalContext,
    /// Optional owner-supplied freshness before the transition. Forbidden in
    /// observer-only mode; never synthesized by this observer.
    pub resource_version_before: Option<ResourceVersion>,
    /// Optional owner-supplied freshness after the transition. Evaluation
    /// stages normally leave this absent; never synthesized by this observer.
    pub resource_version_after: Option<ResourceVersion>,
    /// Explicit causal predecessor set.
    pub causal_parents: BTreeSet<EventId>,
    /// Cognitive tick for correlation only.
    pub cognitive_tick: Option<CognitiveTick>,
    /// Qualification-run grouping label only.
    pub run_id: Option<RunId>,
    /// Confidentiality of the evidence record.
    pub confidentiality: EvidenceConfidentiality,
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
    /// Raw portable data may not claim that a live monitor evaluation occurred.
    #[error("evaluation event {kind:?} requires an opaque same-monitor receipt")]
    EvaluationRequiresOpaqueReceipt {
        /// Evaluation stage rejected from the generic append API.
        kind: ShadowEventKind,
    },
    /// The evaluation-specific API was called with a non-evaluation stage.
    #[error("shadow event {kind:?} is not an evaluation stage")]
    ExpectedEvaluationKind {
        /// Non-evaluation stage supplied to the evaluation API.
        kind: ShadowEventKind,
    },
    /// Opaque receipt belongs to another reference-monitor domain.
    #[error("opaque evaluation receipt belongs to another reference-monitor domain")]
    ForeignMonitorReceipt,
    /// A mapped evaluation stage disagrees with the mutation class in the
    /// opaque monitor receipt.
    #[error(
        "evaluation event {event_kind:?} expects mutation {expected:?}, receipt recorded {observed:?}"
    )]
    EvaluationMutationKindMismatch {
        /// Outer typed evaluation stage.
        event_kind: ShadowEventKind,
        /// Mutation class implied by the mapped stage.
        expected: MutationKind,
        /// Mutation class carried by the opaque receipt.
        observed: MutationKind,
    },
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

    fn expected_mutation_kind(kind: ShadowEventKind) -> Option<MutationKind> {
        match kind {
            ShadowEventKind::WorkingMemoryAdmissionEvaluated => {
                Some(MutationKind::WorkingMemoryAdmission)
            }
            ShadowEventKind::GraduationEvaluated => Some(MutationKind::PersistentMemoryCommit),
            ShadowEventKind::GoalActivationEvaluated => Some(MutationKind::GoalActivation),
            ShadowEventKind::AffectMutationEvaluated => Some(MutationKind::Affect),
            _ => None,
        }
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

    fn validate_common_draft(&self, draft: &ShadowEventDraft) -> Result<(), ShadowAppendError> {
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

        self.validate_resource_binding(draft)
    }

    fn try_append_prepared(
        &mut self,
        draft: ShadowEventDraft,
    ) -> Result<EventId, ShadowAppendError> {
        self.validate_common_draft(&draft)?;
        Ok(self.ledger.try_append(draft)?)
    }

    /// Append a non-evaluation event through the observer-bound evidence ledger.
    ///
    /// Portable application data is intentionally barred from creating live
    /// `...Evaluated` events. Call [`Self::try_append_evaluation`] with an opaque
    /// same-monitor receipt for that case.
    pub fn try_append(&mut self, draft: ShadowEventDraft) -> Result<EventId, ShadowAppendError> {
        if draft.kind.is_evaluation() {
            return Err(ShadowAppendError::EvaluationRequiresOpaqueReceipt {
                kind: draft.kind,
            });
        }
        self.try_append_prepared(draft)
    }

    /// Append one live evaluation using an opaque receipt from the exact
    /// reference-monitor domain that produced it.
    ///
    /// Security-relevant envelope fields are derived from the receipt rather
    /// than supplied independently by the caller:
    ///
    /// - resource identity;
    /// - observed pre-state root;
    /// - evaluated policy root/epoch;
    /// - authorization/revocation epochs;
    /// - portable evaluation payload, including the exact mutation digest and
    ///   monitor decision.
    ///
    /// The caller supplies only correlation/context fields through
    /// [`ShadowEvaluationDraft`]. In safe Rust, possessing a serialized
    /// `MutationReceiptRecord` is therefore insufficient to claim that the live
    /// monitor executed.
    pub fn try_append_evaluation(
        &mut self,
        monitor: &ReferenceMonitor,
        draft: ShadowEvaluationDraft,
        receipt: &MutationReceipt,
    ) -> Result<EventId, ShadowAppendError> {
        if !draft.kind.is_evaluation() {
            return Err(ShadowAppendError::ExpectedEvaluationKind { kind: draft.kind });
        }
        if !monitor.accepts_receipt(receipt) {
            return Err(ShadowAppendError::ForeignMonitorReceipt);
        }

        let record = receipt.record();
        if let Some(expected) = Self::expected_mutation_kind(draft.kind) {
            if record.kind != expected {
                return Err(ShadowAppendError::EvaluationMutationKindMismatch {
                    event_kind: draft.kind,
                    expected,
                    observed: record.kind,
                });
            }
        }

        let prepared = ShadowEventDraft {
            kind: draft.kind,
            proposal_id: draft.proposal_id,
            transaction_id: draft.transaction_id,
            principals: draft.principals,
            resource: Some(record.resource.clone()),
            resource_version_before: draft.resource_version_before,
            resource_version_after: draft.resource_version_after,
            state_root_before: Some(record.observed_resource_state_root),
            state_root_after: None,
            policy_root: Some(record.evaluated_policy_root),
            policy_epoch: Some(record.policy_epoch),
            authorization_epoch: Some(record.authorization_epoch),
            revocation_epoch: Some(record.revocation_epoch),
            causal_parents: draft.causal_parents,
            cognitive_tick: draft.cognitive_tick,
            run_id: draft.run_id,
            confidentiality: draft.confidentiality,
            payload: ShadowEventPayload::Evaluation {
                receipt: receipt.export_record(),
            },
        };

        self.try_append_prepared(prepared)
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
    use symthaea_cogsec::{
        CognitiveSecurityLabel, Consequence, ControlIntegrity, Digest32, MutationRequest,
        PolicyRule, PolicySnapshot, PrincipalId, TaintLevel,
    };
    use symthaea_cogsec_evidence::IngressClass;

    fn d(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

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

    fn evaluation_draft(kind: ShadowEventKind) -> ShadowEvaluationDraft {
        ShadowEvaluationDraft {
            kind,
            proposal_id: None,
            transaction_id: None,
            principals: PrincipalContext::default(),
            resource_version_before: None,
            resource_version_after: None,
            causal_parents: BTreeSet::new(),
            cognitive_tick: None,
            run_id: None,
            confidentiality: EvidenceConfidentiality::LocalPrivate,
        }
    }

    fn goal_monitor_receipt() -> (ReferenceMonitor, MutationReceipt) {
        let (monitor, authority) = ReferenceMonitor::bootstrap();
        let subject = PrincipalId("local-user".into());
        let resource = ResourceId(GOAL_STORE_RESOURCE.into());
        let input_label = CognitiveSecurityLabel::default();
        let request = MutationRequest {
            request_id: d(1),
            kind: MutationKind::GoalActivation,
            subject: subject.clone(),
            resource: resource.clone(),
            mutation_digest: d(2),
            expected_resource_state_root: d(8),
            expected_policy_root: d(9),
            input_label: input_label.clone(),
            consequence: Consequence::Low,
            sequence: 42,
        };
        let transition = authority.issue_transition(
            subject,
            request.kind,
            resource,
            request.mutation_digest,
            request.consequence,
            input_label,
            request.sequence,
        );
        let policy = authority.issue_policy(PolicySnapshot {
            root: d(9),
            epoch: 7,
            rules: vec![PolicyRule {
                kind: MutationKind::GoalActivation,
                minimum_control_integrity: ControlIntegrity::Untrusted,
                maximum_taint: TaintLevel::Revoked,
                capability_required: false,
            }],
        });
        let facts = authority
            .snapshot(&transition, d(8), &policy, 11, 13, &[])
            .unwrap();
        let receipt = monitor.receipt(&request, &facts, &policy).unwrap();
        (monitor, receipt)
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
    fn raw_portable_evaluation_record_cannot_claim_live_monitor_origin() {
        let (_monitor, receipt) = goal_monitor_receipt();
        let mut observer = observer(None);
        let mut draft = ingress(None);
        draft.kind = ShadowEventKind::GoalActivationEvaluated;
        draft.resource = Some(ResourceId(GOAL_STORE_RESOURCE.into()));
        draft.payload = ShadowEventPayload::Evaluation {
            receipt: receipt.export_record(),
        };

        assert_eq!(
            observer.try_append(draft),
            Err(ShadowAppendError::EvaluationRequiresOpaqueReceipt {
                kind: ShadowEventKind::GoalActivationEvaluated,
            })
        );
        assert_eq!(observer.ledger_stats().assigned_sequences, 0);
    }

    #[test]
    fn opaque_same_monitor_receipt_is_exported_with_derived_security_context() {
        let (monitor, receipt) = goal_monitor_receipt();
        let expected_record = receipt.export_record();
        let mut observer = observer(None);

        let event_id = observer
            .try_append_evaluation(
                &monitor,
                evaluation_draft(ShadowEventKind::GoalActivationEvaluated),
                &receipt,
            )
            .unwrap();

        let snapshot = observer.snapshot();
        let event = snapshot
            .events
            .iter()
            .find(|event| event.event_id == event_id)
            .unwrap();
        assert_eq!(event.resource, Some(ResourceId(GOAL_STORE_RESOURCE.into())));
        assert_eq!(event.state_root_before, Some(d(8)));
        assert_eq!(event.policy_root, Some(d(9)));
        assert_eq!(event.policy_epoch, Some(7));
        assert_eq!(event.authorization_epoch, Some(11));
        assert_eq!(event.revocation_epoch, Some(13));
        assert_eq!(
            event.payload,
            ShadowEventPayload::Evaluation {
                receipt: expected_record,
            }
        );
    }

    #[test]
    fn foreign_monitor_receipt_is_rejected_before_event_id_allocation() {
        let (monitor_a, _receipt_a) = goal_monitor_receipt();
        let (_monitor_b, receipt_b) = goal_monitor_receipt();
        let mut observer = observer(None);

        assert_eq!(
            observer.try_append_evaluation(
                &monitor_a,
                evaluation_draft(ShadowEventKind::GoalActivationEvaluated),
                &receipt_b,
            ),
            Err(ShadowAppendError::ForeignMonitorReceipt)
        );
        assert_eq!(observer.ledger_stats().assigned_sequences, 0);
    }

    #[test]
    fn mapped_evaluation_stage_must_match_receipt_mutation_kind() {
        let (monitor, receipt) = goal_monitor_receipt();
        let mut observer = observer(None);

        assert_eq!(
            observer.try_append_evaluation(
                &monitor,
                evaluation_draft(ShadowEventKind::AffectMutationEvaluated),
                &receipt,
            ),
            Err(ShadowAppendError::EvaluationMutationKindMismatch {
                event_kind: ShadowEventKind::AffectMutationEvaluated,
                expected: MutationKind::Affect,
                observed: MutationKind::GoalActivation,
            })
        );
        assert_eq!(observer.ledger_stats().assigned_sequences, 0);
    }

    #[test]
    fn non_evaluation_stage_is_rejected_by_opaque_receipt_api() {
        let (monitor, receipt) = goal_monitor_receipt();
        let mut observer = observer(None);

        assert_eq!(
            observer.try_append_evaluation(
                &monitor,
                evaluation_draft(ShadowEventKind::GoalActivationObserved),
                &receipt,
            ),
            Err(ShadowAppendError::ExpectedEvaluationKind {
                kind: ShadowEventKind::GoalActivationObserved,
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
