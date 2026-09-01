// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Automatic portable exact-effect production for strict CogSec shadow mode.
//!
//! The live effect-pairing façade proves equality in-process. This outer façade
//! records the same already-validated effect commitment at the moment the paired
//! observed event is successfully appended, eliminating a later caller-side
//! reconstruction step. Pending tokens are also sealed to the exact observer
//! instance that minted them.

use std::sync::Arc;

use symthaea_cogsec::{MutationReceipt, ReferenceMonitor, ResourceId};
use symthaea_cogsec_evidence::{
    EffectBoundEvidenceSnapshot, EvidenceCompleteness, EventId, LedgerStats, ObservedEffectBinding,
    QualificationManifest, ShadowEventDraft, ShadowEventKind,
};
use symthaea_evidence_plane::RunId;
use thiserror::Error;

use crate::{
    EffectBoundAppendError, EffectBoundShadowRuntimeObserver, PendingObservedEffect,
    ShadowAssuranceProfile, ShadowEvaluationDraft, ShadowObservedMutationDraft,
    ShadowObserverInitError, ShadowResource,
};

#[derive(Debug)]
struct ObserverDomainSeal;

/// Failure to construct the portable effect-producing observer.
#[derive(Debug, Error)]
pub enum PortableEffectObserverInitError {
    /// Exact-effect sidecar storage could not be reserved before runtime use.
    #[error("failed to reserve portable effect-binding capacity {capacity}")]
    BindingCapacityReservationFailed {
        /// Requested binding capacity.
        capacity: usize,
    },
    /// Underlying strict observer construction failed.
    #[error(transparent)]
    Observer(#[from] ShadowObserverInitError),
}

/// Failure while appending through the portable effect-producing observer.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PortableEffectAppendError {
    /// Pending effect token belongs to another observer instance.
    #[error("pending observed-effect token belongs to another shadow observer domain")]
    ForeignObserverPendingEffect,
    /// Underlying exact-effect observer rejected the append.
    #[error(transparent)]
    Observer(#[from] EffectBoundAppendError),
}

/// One-use pending observed-effect token bound to one exact observer instance.
///
/// The inner live pairing token is already non-`Clone` and non-serde. This
/// wrapper adds observer-instance affinity through a private pointer seal, so a
/// token minted by observer A cannot be consumed by observer B even when all
/// visible correlation fields happen to match.
pub struct PortablePendingObservedEffect {
    seal: Arc<ObserverDomainSeal>,
    inner: PendingObservedEffect,
}

impl std::fmt::Debug for PortablePendingObservedEffect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PortablePendingObservedEffect")
            .field("evaluation_event_id", &self.inner.evaluation_event_id())
            .field("observed_kind", &self.inner.observed_kind())
            .finish_non_exhaustive()
    }
}

impl PortablePendingObservedEffect {
    /// Event identity of the exact monitor evaluation awaiting observation.
    pub const fn evaluation_event_id(&self) -> EventId {
        self.inner.evaluation_event_id()
    }
}

/// Strict shadow observer that automatically produces portable exact-effect bindings.
///
/// This is the preferred pre-`ContinuousMind` integration surface after #245.
/// It keeps the lower observer private, records one sidecar entry only after a
/// paired observed mutation successfully appends, and exports the base ledger
/// and bindings together.
#[derive(Debug)]
pub struct PortableEffectBoundShadowRuntimeObserver {
    seal: Arc<ObserverDomainSeal>,
    inner: EffectBoundShadowRuntimeObserver,
    observed_effects: Vec<ObservedEffectBinding>,
    binding_capacity: usize,
}

impl PortableEffectBoundShadowRuntimeObserver {
    /// Construct a strict observer and reserve all portable sidecar storage before runtime use.
    pub fn try_new_preallocated(
        profile: ShadowAssuranceProfile,
        ledger_epoch: u64,
        event_capacity: usize,
        manifest: QualificationManifest,
        run_id: Option<RunId>,
    ) -> Result<Self, PortableEffectObserverInitError> {
        let mut observed_effects = Vec::new();
        observed_effects.try_reserve_exact(event_capacity).map_err(|_| {
            PortableEffectObserverInitError::BindingCapacityReservationFailed {
                capacity: event_capacity,
            }
        })?;

        Ok(Self {
            seal: Arc::new(ObserverDomainSeal),
            inner: EffectBoundShadowRuntimeObserver::try_new_preallocated(
                profile,
                ledger_epoch,
                event_capacity,
                manifest,
                run_id,
            )?,
            observed_effects,
            binding_capacity: event_capacity,
        })
    }

    /// Current assurance profile for this observer instance.
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

    /// Current evidence completeness from the underlying bounded event ledger.
    pub fn completeness(&self) -> EvidenceCompleteness {
        self.inner.completeness()
    }

    /// Whether the sidecar backing vector was reserved for the full declared event capacity.
    pub fn effect_binding_storage_fully_preallocated(&self) -> bool {
        self.observed_effects.capacity() >= self.binding_capacity
    }

    /// Current local append accounting from the underlying event ledger.
    pub fn ledger_stats(&self) -> LedgerStats {
        self.inner.ledger_stats()
    }

    /// Append evidence that is not a paired legacy mutation.
    ///
    /// This is the only generic live append surface exposed by the strict public
    /// observer. Paired `...Observed` stages are rejected by the inner
    /// effect-bound layer and must consume their pending token instead.
    pub fn try_append_unpaired(
        &mut self,
        draft: ShadowEventDraft,
    ) -> Result<EventId, PortableEffectAppendError> {
        self.inner.try_append_unpaired(draft).map_err(Into::into)
    }

    /// Append a genuine monitor evaluation and return an observer-domain-bound pending token.
    pub fn try_append_evaluation(
        &mut self,
        monitor: &ReferenceMonitor,
        draft: ShadowEvaluationDraft,
        receipt: MutationReceipt,
    ) -> Result<PortablePendingObservedEffect, PortableEffectAppendError> {
        let inner = self.inner.try_append_evaluation(monitor, draft, receipt)?;
        Ok(PortablePendingObservedEffect {
            seal: Arc::clone(&self.seal),
            inner,
        })
    }

    /// Consume one same-observer pending token and append its exact paired observation.
    ///
    /// The sidecar binding is added only after the underlying observed event was
    /// successfully retained. Digest mismatch or any other append failure cannot
    /// create an orphan portable effect binding.
    pub fn try_append_observed_mutation(
        &mut self,
        pending: PortablePendingObservedEffect,
        draft: ShadowObservedMutationDraft,
    ) -> Result<EventId, PortableEffectAppendError> {
        if !Arc::ptr_eq(&self.seal, &pending.seal) {
            return Err(PortableEffectAppendError::ForeignObserverPendingEffect);
        }

        let effect_digest = draft.effect_digest;
        let observed_event_id = self
            .inner
            .try_append_observed_mutation(pending.inner, draft)?;

        self.observed_effects.push(ObservedEffectBinding {
            observed_event_id,
            effect_digest,
        });
        Ok(observed_event_id)
    }

    /// Export base v1 shadow evidence plus the automatically produced effect sidecar.
    ///
    /// This remains ordinary serializable evidence; structural consistency does
    /// not authenticate the host that produced it.
    pub fn effect_bound_snapshot(&self) -> EffectBoundEvidenceSnapshot {
        EffectBoundEvidenceSnapshot::new(self.inner.snapshot(), self.observed_effects.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::BTreeSet;

    use symthaea_cogsec::{
        CognitiveSecurityLabel, Consequence, ControlIntegrity, Digest32, MutationKind,
        MutationRequest, PolicyRule, PolicySnapshot, PrincipalId, ResourceId, TaintLevel,
    };
    use symthaea_cogsec_evidence::{
        EvidenceConfidentiality, IngressClass, PrincipalContext, ShadowEventKind,
        ShadowEventPayload, validate_effect_bound_snapshot,
    };

    fn d(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn observer() -> PortableEffectBoundShadowRuntimeObserver {
        PortableEffectBoundShadowRuntimeObserver::try_new_preallocated(
            ShadowAssuranceProfile::ObserverOnly,
            23,
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

    fn ingress() -> ShadowEventDraft {
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
            run_id: None,
            confidentiality: EvidenceConfidentiality::LocalPrivate,
            payload: ShadowEventPayload::Ingress {
                ingress_class: IngressClass::LegacyUnclassified,
            },
        }
    }

    #[test]
    fn portable_binding_storage_is_preallocated() {
        assert!(observer().effect_binding_storage_fully_preallocated());
    }

    #[test]
    fn strict_public_facade_still_supports_unpaired_ingress() {
        let mut observer = observer();
        let event_id = observer.try_append_unpaired(ingress()).unwrap();
        assert_eq!(event_id.sequence, 1);
        assert_eq!(observer.ledger_stats().stored_events, 1);
    }

    #[test]
    fn successful_pair_automatically_exports_exact_effect_binding() {
        let (monitor, receipt) = goal_monitor_receipt();
        let mut observer = observer();
        let pending = observer
            .try_append_evaluation(&monitor, goal_evaluation_draft(), receipt)
            .unwrap();
        let observed_event_id = observer
            .try_append_observed_mutation(
                pending,
                ShadowObservedMutationDraft {
                    effect_digest: d(2),
                    applied: true,
                    resource_version_after: None,
                    state_root_after: None,
                },
            )
            .unwrap();

        let snapshot = observer.effect_bound_snapshot();
        assert_eq!(snapshot.observed_effects.len(), 1);
        assert_eq!(snapshot.observed_effects[0].observed_event_id, observed_event_id);
        assert_eq!(snapshot.observed_effects[0].effect_digest, d(2));
        assert!(validate_effect_bound_snapshot(&snapshot).effect_bindings_are_consistent());
    }

    #[test]
    fn failed_effect_match_does_not_create_orphan_sidecar_binding() {
        let (monitor, receipt) = goal_monitor_receipt();
        let mut observer = observer();
        let pending = observer
            .try_append_evaluation(&monitor, goal_evaluation_draft(), receipt)
            .unwrap();

        assert!(matches!(
            observer.try_append_observed_mutation(
                pending,
                ShadowObservedMutationDraft {
                    effect_digest: d(99),
                    applied: true,
                    resource_version_after: None,
                    state_root_after: None,
                },
            ),
            Err(PortableEffectAppendError::Observer(
                EffectBoundAppendError::EffectDigestMismatch { .. }
            ))
        ));

        let snapshot = observer.effect_bound_snapshot();
        assert!(snapshot.observed_effects.is_empty());
        assert_eq!(snapshot.base.events.len(), 1);
    }

    #[test]
    fn pending_token_cannot_cross_observer_domains() {
        let (monitor, receipt) = goal_monitor_receipt();
        let mut observer_a = observer();
        let mut observer_b = observer();
        let pending = observer_a
            .try_append_evaluation(&monitor, goal_evaluation_draft(), receipt)
            .unwrap();

        assert_eq!(
            observer_b.try_append_observed_mutation(
                pending,
                ShadowObservedMutationDraft {
                    effect_digest: d(2),
                    applied: true,
                    resource_version_after: None,
                    state_root_after: None,
                },
            ),
            Err(PortableEffectAppendError::ForeignObserverPendingEffect)
        );
        assert_eq!(observer_b.ledger_stats().assigned_sequences, 0);
        assert!(observer_b.effect_bound_snapshot().observed_effects.is_empty());
    }
}
