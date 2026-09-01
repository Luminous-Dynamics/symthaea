// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! One-use live effect pairing for CogSec shadow observation.
//!
//! This module strengthens the **in-process** observation API without pretending
//! the current portable event schema can independently re-prove effect equality
//! after export. A genuine monitor evaluation yields one non-cloneable pending
//! token carrying the exact evaluated effect commitment. The corresponding
//! observed mutation must consume that token and present the same digest.

use std::collections::BTreeSet;

use symthaea_cogsec::{
    CognitiveSecurityLabel, Consequence, ControlIntegrity, Digest32, MutationKind,
    MutationReceipt, MutationRequest, PolicyRule, PolicySnapshot, PrincipalId, ReferenceMonitor,
    ResourceId, TaintLevel,
};
use symthaea_cogsec_evidence::{
    CognitiveTick, EvidenceCompleteness, EvidenceConfidentiality, EvidenceLedgerSnapshot, EventId,
    LedgerStats, PrincipalContext, ProposalId, QualificationManifest, ResourceVersion,
    ShadowEventDraft, ShadowEventKind, ShadowEventPayload, TransactionId,
};
use symthaea_evidence_plane::RunId;
use thiserror::Error;

use crate::{
    ShadowAppendError, ShadowAssuranceProfile, ShadowEvaluationDraft, ShadowObserverInitError,
    ShadowResource, ShadowRuntimeObserver,
};

/// Context supplied after legacy code attempts the exact evaluated mutation.
///
/// `effect_digest` must be computed from the same canonical structured effect
/// representation used for monitor evaluation. It is checked against the
/// monitor receipt commitment before any observed event is appended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShadowObservedMutationDraft {
    /// Exact commitment to the effect legacy code attempted/applied.
    pub effect_digest: Digest32,
    /// Whether legacy code actually applied the protected mutation.
    pub applied: bool,
    /// Optional owner-issued freshness after the mutation. Observer-only mode
    /// will reject this through the underlying observer boundary.
    pub resource_version_after: Option<ResourceVersion>,
    /// Optional protected-state root after the mutation.
    pub state_root_after: Option<Digest32>,
}

/// One-use in-process proof that a specific monitor evaluation was appended and
/// is waiting for its corresponding observed legacy mutation.
///
/// The token is deliberately non-`Clone` and non-serde. Consuming it to append
/// an observed mutation gives the live adapter a one-evaluation/one-observation
/// shape without a replay table.
pub struct PendingObservedEffect {
    evaluation_event_id: EventId,
    observed_kind: ShadowEventKind,
    effect_digest: Digest32,
    proposal_id: Option<ProposalId>,
    transaction_id: Option<TransactionId>,
    principals: PrincipalContext,
    resource: ResourceId,
    resource_version_before: Option<ResourceVersion>,
    state_root_before: Option<Digest32>,
    policy_root: Option<Digest32>,
    policy_epoch: Option<u64>,
    authorization_epoch: Option<u64>,
    revocation_epoch: Option<u64>,
    cognitive_tick: Option<CognitiveTick>,
    run_id: Option<RunId>,
    confidentiality: EvidenceConfidentiality,
}

impl std::fmt::Debug for PendingObservedEffect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingObservedEffect")
            .field("evaluation_event_id", &self.evaluation_event_id)
            .field("observed_kind", &self.observed_kind)
            .field("resource", &self.resource)
            .finish_non_exhaustive()
    }
}

impl PendingObservedEffect {
    /// Event identity of the monitor evaluation this token must pair with.
    pub const fn evaluation_event_id(&self) -> EventId {
        self.evaluation_event_id
    }

    /// Typed observed stage that may consume this token.
    pub const fn observed_kind(&self) -> ShadowEventKind {
        self.observed_kind
    }
}

/// Failure at the effect-bound live observation façade.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EffectBoundAppendError {
    /// Underlying observer boundary rejected the event.
    #[error(transparent)]
    Observer(#[from] ShadowAppendError),
    /// A paired observed stage attempted to bypass its pending evaluation token.
    #[error("paired observed event {kind:?} requires its one-use pending evaluation token")]
    PairedObservationRequiresPendingEffect {
        /// Paired observed stage rejected from the generic path.
        kind: ShadowEventKind,
    },
    /// Evaluation stage does not map to a corresponding observed mutation stage.
    #[error("evaluation event {kind:?} has no corresponding observed mutation stage")]
    EvaluationHasNoObservedPair {
        /// Evaluation stage missing a pair.
        kind: ShadowEventKind,
    },
    /// Legacy code reported a different exact effect than the monitor evaluated.
    #[error("observed effect digest differs from the exact monitor-evaluated effect")]
    EffectDigestMismatch {
        /// Exact effect commitment recorded by the monitor receipt.
        expected: Digest32,
        /// Effect commitment reported at the legacy mutation boundary.
        observed: Digest32,
    },
}

/// Stricter first-integration façade around [`ShadowRuntimeObserver`].
///
/// The inner observer is intentionally private and cannot be extracted. Paired
/// `...Observed` stages therefore have to consume the token returned by
/// [`Self::try_append_evaluation`], while genuinely unpaired evidence such as
/// ingress, evidence gaps and the current transitional WM-eviction observation
/// can still use [`Self::try_append_unpaired`].
#[derive(Debug)]
pub struct EffectBoundShadowRuntimeObserver {
    inner: ShadowRuntimeObserver,
}

impl EffectBoundShadowRuntimeObserver {
    /// Construct the strict live shadow observer with fully preallocated event storage.
    pub fn try_new_preallocated(
        profile: ShadowAssuranceProfile,
        ledger_epoch: u64,
        event_capacity: usize,
        manifest: QualificationManifest,
        run_id: Option<RunId>,
    ) -> Result<Self, ShadowObserverInitError> {
        Ok(Self {
            inner: ShadowRuntimeObserver::try_new_preallocated(
                profile,
                ledger_epoch,
                event_capacity,
                manifest,
                run_id,
            )?,
        })
    }

    /// Current shadow assurance profile.
    pub const fn profile(&self) -> ShadowAssuranceProfile {
        self.inner.profile()
    }

    /// Qualification-run grouping identity only.
    pub fn run_id(&self) -> Option<&RunId> {
        self.inner.run_id()
    }

    /// Canonical resource identity for one protected shadow domain.
    pub fn resource_id(&self, resource: ShadowResource) -> &ResourceId {
        self.inner.resource_id(resource)
    }

    /// Canonical resource identity expected for one typed stage.
    pub fn expected_resource_id(&self, kind: ShadowEventKind) -> Option<&ResourceId> {
        self.inner.expected_resource_id(kind)
    }

    /// Evidence completeness observed by the bounded ledger.
    pub fn completeness(&self) -> EvidenceCompleteness {
        self.inner.completeness()
    }

    /// Local append accounting.
    pub fn ledger_stats(&self) -> LedgerStats {
        self.inner.ledger_stats()
    }

    /// Whether retained-event storage was fully preallocated.
    pub fn event_storage_fully_preallocated(&self) -> bool {
        self.inner.event_storage_fully_preallocated()
    }

    /// Export ordinary portable evidence for independent reconciliation.
    ///
    /// The current portable schema does **not** carry the observed effect digest,
    /// so export loses the stronger in-process digest-equality proof established
    /// by this façade. #232 remains open for that schema-level evolution.
    pub fn snapshot(&self) -> EvidenceLedgerSnapshot {
        self.inner.snapshot()
    }

    /// Append evidence that is not a paired legacy mutation.
    ///
    /// Paired `...Observed` stages are rejected here and must consume the
    /// [`PendingObservedEffect`] returned by the evaluation path.
    pub fn try_append_unpaired(
        &mut self,
        draft: ShadowEventDraft,
    ) -> Result<EventId, EffectBoundAppendError> {
        if draft.kind.expected_evaluation().is_some() {
            return Err(EffectBoundAppendError::PairedObservationRequiresPendingEffect {
                kind: draft.kind,
            });
        }
        self.inner.try_append(draft).map_err(EffectBoundAppendError::from)
    }

    /// Append one genuine monitor evaluation and return its one-use pending
    /// observed-effect token.
    ///
    /// The opaque monitor receipt itself is consumed by the underlying observer.
    /// Before handing it over, this façade captures the exact effect commitment
    /// and the security/correlation context needed to construct the paired
    /// observed event without accepting a second independently editable copy.
    pub fn try_append_evaluation(
        &mut self,
        monitor: &ReferenceMonitor,
        draft: ShadowEvaluationDraft,
        receipt: MutationReceipt,
    ) -> Result<PendingObservedEffect, EffectBoundAppendError> {
        let Some(observed_kind) = paired_observed_kind(draft.kind) else {
            return Err(EffectBoundAppendError::EvaluationHasNoObservedPair {
                kind: draft.kind,
            });
        };

        let record = receipt.record();
        let effect_digest = record.mutation_digest;
        let resource = record.resource.clone();
        let state_root_before = Some(record.observed_resource_state_root);
        let policy_root = Some(record.evaluated_policy_root);
        let policy_epoch = Some(record.policy_epoch);
        let authorization_epoch = Some(record.authorization_epoch);
        let revocation_epoch = Some(record.revocation_epoch);

        let proposal_id = draft.proposal_id;
        let transaction_id = draft.transaction_id;
        let principals = draft.principals.clone();
        let resource_version_before = draft.resource_version_before;
        let cognitive_tick = draft.cognitive_tick;
        let run_id = draft.run_id.clone();
        let confidentiality = draft.confidentiality;

        let evaluation_event_id = self
            .inner
            .try_append_evaluation(monitor, draft, receipt)?;

        Ok(PendingObservedEffect {
            evaluation_event_id,
            observed_kind,
            effect_digest,
            proposal_id,
            transaction_id,
            principals,
            resource,
            resource_version_before,
            state_root_before,
            policy_root,
            policy_epoch,
            authorization_epoch,
            revocation_epoch,
            cognitive_tick,
            run_id,
            confidentiality,
        })
    }

    /// Consume one pending evaluation token to append the corresponding observed
    /// legacy mutation.
    ///
    /// Digest mismatch fails before the underlying ledger receives an append,
    /// so no observed EventId is allocated. The pending token is still consumed;
    /// retry requires a fresh monitor evaluation rather than replaying the same
    /// evaluation evidence.
    pub fn try_append_observed_mutation(
        &mut self,
        pending: PendingObservedEffect,
        draft: ShadowObservedMutationDraft,
    ) -> Result<EventId, EffectBoundAppendError> {
        if draft.effect_digest != pending.effect_digest {
            return Err(EffectBoundAppendError::EffectDigestMismatch {
                expected: pending.effect_digest,
                observed: draft.effect_digest,
            });
        }

        let mut causal_parents = BTreeSet::new();
        causal_parents.insert(pending.evaluation_event_id);

        let observed = ShadowEventDraft {
            kind: pending.observed_kind,
            proposal_id: pending.proposal_id,
            transaction_id: pending.transaction_id,
            principals: pending.principals,
            resource: Some(pending.resource),
            resource_version_before: pending.resource_version_before,
            resource_version_after: draft.resource_version_after,
            state_root_before: pending.state_root_before,
            state_root_after: draft.state_root_after,
            policy_root: pending.policy_root,
            policy_epoch: pending.policy_epoch,
            authorization_epoch: pending.authorization_epoch,
            revocation_epoch: pending.revocation_epoch,
            causal_parents,
            cognitive_tick: pending.cognitive_tick,
            run_id: pending.run_id,
            confidentiality: pending.confidentiality,
            payload: ShadowEventPayload::MutationObserved {
                applied: draft.applied,
            },
        };

        self.inner
            .try_append(observed)
            .map_err(EffectBoundAppendError::from)
    }
}

fn paired_observed_kind(kind: ShadowEventKind) -> Option<ShadowEventKind> {
    match kind {
        ShadowEventKind::WorkingMemoryAdmissionEvaluated => {
            Some(ShadowEventKind::WorkingMemoryAdmissionObserved)
        }
        ShadowEventKind::GraduationEvaluated => Some(ShadowEventKind::GraduationObserved),
        ShadowEventKind::WorkingStateInfluenceEvaluated => {
            Some(ShadowEventKind::WorkingStateInfluenceObserved)
        }
        ShadowEventKind::GoalActivationEvaluated => Some(ShadowEventKind::GoalActivationObserved),
        ShadowEventKind::AffectMutationEvaluated => Some(ShadowEventKind::AffectMutationObserved),
        ShadowEventKind::DreamMergeEvaluated => Some(ShadowEventKind::DreamMergeObserved),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn observer() -> EffectBoundShadowRuntimeObserver {
        EffectBoundShadowRuntimeObserver::try_new_preallocated(
            ShadowAssuranceProfile::ObserverOnly,
            17,
            32,
            QualificationManifest::new([], []),
            None,
        )
        .unwrap()
    }

    fn goal_monitor_receipt() -> (ReferenceMonitor, MutationReceipt) {
        let (monitor, authority) = ReferenceMonitor::bootstrap();
        let subject = PrincipalId("local-user".into());
        let resource = ResourceId("mind/goals".into());
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

    fn goal_evaluation_draft() -> ShadowEvaluationDraft {
        ShadowEvaluationDraft {
            kind: ShadowEventKind::GoalActivationEvaluated,
            proposal_id: Some(ProposalId(d(30))),
            transaction_id: Some(TransactionId(d(31))),
            principals: PrincipalContext::default(),
            resource_version_before: None,
            resource_version_after: None,
            causal_parents: BTreeSet::new(),
            cognitive_tick: Some(CognitiveTick(77)),
            run_id: None,
            confidentiality: EvidenceConfidentiality::LocalPrivate,
        }
    }

    fn raw_goal_observation() -> ShadowEventDraft {
        ShadowEventDraft {
            kind: ShadowEventKind::GoalActivationObserved,
            proposal_id: None,
            transaction_id: None,
            principals: PrincipalContext::default(),
            resource: Some(ResourceId("mind/goals".into())),
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
    fn paired_observation_cannot_bypass_pending_effect_token() {
        let mut observer = observer();
        assert_eq!(
            observer.try_append_unpaired(raw_goal_observation()),
            Err(EffectBoundAppendError::PairedObservationRequiresPendingEffect {
                kind: ShadowEventKind::GoalActivationObserved,
            })
        );
        assert_eq!(observer.ledger_stats().assigned_sequences, 0);
    }

    #[test]
    fn matching_digest_consumes_pending_token_and_derives_pair_context() {
        let (monitor, receipt) = goal_monitor_receipt();
        let mut observer = observer();
        let pending = observer
            .try_append_evaluation(&monitor, goal_evaluation_draft(), receipt)
            .unwrap();
        let evaluation_event_id = pending.evaluation_event_id();

        let observed_event_id = observer
            .try_append_observed_mutation(
                pending,
                ShadowObservedMutationDraft {
                    effect_digest: d(2),
                    applied: true,
                    resource_version_after: None,
                    state_root_after: Some(d(10)),
                },
            )
            .unwrap();

        let snapshot = observer.snapshot();
        assert_eq!(snapshot.events.len(), 2);
        let observed = snapshot
            .events
            .iter()
            .find(|event| event.event_id == observed_event_id)
            .unwrap();
        assert_eq!(observed.kind, ShadowEventKind::GoalActivationObserved);
        assert_eq!(observed.resource, Some(ResourceId("mind/goals".into())));
        assert_eq!(observed.state_root_before, Some(d(8)));
        assert_eq!(observed.state_root_after, Some(d(10)));
        assert_eq!(observed.policy_root, Some(d(9)));
        assert_eq!(observed.policy_epoch, Some(7));
        assert_eq!(observed.authorization_epoch, Some(11));
        assert_eq!(observed.revocation_epoch, Some(13));
        assert_eq!(observed.proposal_id, Some(ProposalId(d(30))));
        assert_eq!(observed.transaction_id, Some(TransactionId(d(31))));
        assert_eq!(observed.cognitive_tick, Some(CognitiveTick(77)));
        assert_eq!(observed.causal_parents.len(), 1);
        assert!(observed.causal_parents.contains(&evaluation_event_id));
        assert_eq!(
            observed.payload,
            ShadowEventPayload::MutationObserved { applied: true }
        );
    }

    #[test]
    fn mismatched_effect_digest_is_rejected_before_observed_event_allocation() {
        let (monitor, receipt) = goal_monitor_receipt();
        let mut observer = observer();
        let pending = observer
            .try_append_evaluation(&monitor, goal_evaluation_draft(), receipt)
            .unwrap();

        assert_eq!(
            observer.try_append_observed_mutation(
                pending,
                ShadowObservedMutationDraft {
                    effect_digest: d(99),
                    applied: true,
                    resource_version_after: None,
                    state_root_after: None,
                },
            ),
            Err(EffectBoundAppendError::EffectDigestMismatch {
                expected: d(2),
                observed: d(99),
            })
        );

        assert_eq!(observer.ledger_stats().assigned_sequences, 1);
        assert_eq!(observer.snapshot().events.len(), 1);
    }

    #[test]
    fn paired_kind_is_derived_from_evaluation_token_not_observed_call_site() {
        let (monitor, receipt) = goal_monitor_receipt();
        let mut observer = observer();
        let pending = observer
            .try_append_evaluation(&monitor, goal_evaluation_draft(), receipt)
            .unwrap();
        assert_eq!(pending.observed_kind(), ShadowEventKind::GoalActivationObserved);
    }
}
