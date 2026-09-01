// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Portable exact-effect sidecar for CogSec shadow evidence.
//!
//! The v1 `ShadowEvent` schema intentionally remains unchanged. This additive
//! sidecar records the exact effect commitment associated with each paired
//! observed mutation so an independent verifier can re-check equality with the
//! direct evaluation parent's monitor receipt after export.
//!
//! These records are ordinary serializable evidence. They prove deterministic
//! structural consistency, not monitor/host authenticity; checkpoint/signature
//! provenance remains a separate assurance layer.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use symthaea_cogsec::{Digest32, ReceiptStage};

use crate::{EvidenceLedgerSnapshot, EventId, ShadowEventKind, ShadowEventPayload};

/// Schema version for the additive portable effect-binding sidecar.
pub const EFFECT_BINDING_SCHEMA_V1: u16 = 1;

/// Exact effect commitment claimed for one paired observed mutation event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservedEffectBinding {
    /// EventId of the paired `...Observed` mutation record.
    pub observed_event_id: EventId,
    /// Exact canonical effect commitment observed at the legacy mutation boundary.
    pub effect_digest: Digest32,
}

/// Portable shadow evidence plus exact observed-effect commitments.
///
/// This envelope does not authenticate either component. It allows an
/// independent verifier to deterministically re-check that every paired
/// observation carries the same effect commitment as its direct evaluation
/// parent's monitor receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectBoundEvidenceSnapshot {
    /// Sidecar schema version.
    pub schema_version: u16,
    /// Existing v1 event-ledger snapshot.
    pub base: EvidenceLedgerSnapshot,
    /// Exact effect commitments keyed by paired observed EventId.
    pub observed_effects: Vec<ObservedEffectBinding>,
}

impl EffectBoundEvidenceSnapshot {
    /// Construct the current portable effect-binding envelope.
    pub fn new(
        base: EvidenceLedgerSnapshot,
        observed_effects: Vec<ObservedEffectBinding>,
    ) -> Self {
        Self {
            schema_version: EFFECT_BINDING_SCHEMA_V1,
            base,
            observed_effects,
        }
    }
}

/// Deterministic contradiction in the portable effect-binding sidecar.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EffectBindingViolation {
    /// Sidecar schema version is unsupported.
    SnapshotSchemaMismatch {
        /// Version found in the portable envelope.
        found: u16,
    },
    /// More than one sidecar entry claims the same observed event.
    DuplicateObservedBinding {
        /// Observed EventId with duplicate commitments.
        observed_event_id: EventId,
    },
    /// Sidecar references an EventId absent from the base event ledger.
    UnknownObservedEvent {
        /// Missing EventId.
        observed_event_id: EventId,
    },
    /// Sidecar attempts to bind an event kind that is not a paired observed mutation.
    BindingForUnpairedEvent {
        /// EventId carrying the invalid binding.
        observed_event_id: EventId,
        /// Event kind found in the base ledger.
        kind: ShadowEventKind,
    },
    /// A paired observed mutation has no exact-effect sidecar entry.
    MissingObservedBinding {
        /// Paired observed EventId missing its commitment.
        observed_event_id: EventId,
    },
    /// Paired observation has no direct causal parent of the expected evaluation kind.
    MissingMatchingEvaluation {
        /// Paired observed EventId.
        observed_event_id: EventId,
        /// Evaluation stage required by the event taxonomy.
        expected: ShadowEventKind,
    },
    /// Paired observation has more than one direct causal parent of the expected evaluation kind.
    AmbiguousMatchingEvaluation {
        /// Paired observed EventId.
        observed_event_id: EventId,
        /// Evaluation stage required by the event taxonomy.
        expected: ShadowEventKind,
    },
    /// Matching evaluation event does not carry an evaluation payload.
    EvaluationPayloadMissing {
        /// Evaluation EventId with the malformed payload.
        evaluation_event_id: EventId,
    },
    /// Matching evaluation receipt is not an evaluation-stage receipt.
    EvaluationReceiptStageMismatch {
        /// Evaluation EventId with the inconsistent receipt stage.
        evaluation_event_id: EventId,
    },
    /// Observed effect commitment differs from the monitor-evaluated commitment.
    EffectDigestMismatch {
        /// Paired observed mutation EventId.
        observed_event_id: EventId,
        /// Direct matching evaluation EventId.
        evaluation_event_id: EventId,
        /// Commitment recorded in the portable observed-effect sidecar.
        observed: Digest32,
        /// Commitment recorded in the monitor evaluation receipt.
        evaluated: Digest32,
    },
}

/// Result of deterministic portable exact-effect verification.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectBindingReport {
    /// Hard structural/digest contradictions.
    pub violations: Vec<EffectBindingViolation>,
}

impl EffectBindingReport {
    /// Whether every paired observed mutation is bound to the exact effect its
    /// direct evaluation parent says the monitor evaluated.
    ///
    /// This is a structural consistency claim only. It does not authenticate
    /// the portable snapshot itself.
    pub fn effect_bindings_are_consistent(&self) -> bool {
        self.violations.is_empty()
    }
}

/// Independently validate the additive exact-effect sidecar.
///
/// Callers should also run the base v1 reconciliation/qualification layers;
/// this function deliberately does not duplicate all event-ledger, counter,
/// policy/resource, or monitor-origin checks.
pub fn validate_effect_bound_snapshot(snapshot: &EffectBoundEvidenceSnapshot) -> EffectBindingReport {
    let mut report = EffectBindingReport::default();

    if snapshot.schema_version != EFFECT_BINDING_SCHEMA_V1 {
        report
            .violations
            .push(EffectBindingViolation::SnapshotSchemaMismatch {
                found: snapshot.schema_version,
            });
    }

    let by_id: BTreeMap<EventId, _> = snapshot
        .base
        .events
        .iter()
        .map(|event| (event.event_id, event))
        .collect();

    let mut bindings = BTreeMap::<EventId, Digest32>::new();
    for binding in &snapshot.observed_effects {
        if bindings
            .insert(binding.observed_event_id, binding.effect_digest)
            .is_some()
        {
            report
                .violations
                .push(EffectBindingViolation::DuplicateObservedBinding {
                    observed_event_id: binding.observed_event_id,
                });
        }

        match by_id.get(&binding.observed_event_id) {
            None => report
                .violations
                .push(EffectBindingViolation::UnknownObservedEvent {
                    observed_event_id: binding.observed_event_id,
                }),
            Some(event) if event.kind.expected_evaluation().is_none() => report
                .violations
                .push(EffectBindingViolation::BindingForUnpairedEvent {
                    observed_event_id: binding.observed_event_id,
                    kind: event.kind,
                }),
            Some(_) => {}
        }
    }

    for observed_event in &snapshot.base.events {
        let Some(expected_evaluation_kind) = observed_event.kind.expected_evaluation() else {
            continue;
        };

        let Some(observed_digest) = bindings.get(&observed_event.event_id).copied() else {
            report
                .violations
                .push(EffectBindingViolation::MissingObservedBinding {
                    observed_event_id: observed_event.event_id,
                });
            continue;
        };

        let matching: Vec<_> = observed_event
            .causal_parents
            .iter()
            .filter_map(|parent_id| by_id.get(parent_id).copied())
            .filter(|event| event.kind == expected_evaluation_kind)
            .collect();

        let evaluation_event = match matching.as_slice() {
            [] => {
                report
                    .violations
                    .push(EffectBindingViolation::MissingMatchingEvaluation {
                        observed_event_id: observed_event.event_id,
                        expected: expected_evaluation_kind,
                    });
                continue;
            }
            [evaluation_event] => *evaluation_event,
            _ => {
                report
                    .violations
                    .push(EffectBindingViolation::AmbiguousMatchingEvaluation {
                        observed_event_id: observed_event.event_id,
                        expected: expected_evaluation_kind,
                    });
                continue;
            }
        };

        let ShadowEventPayload::Evaluation { receipt } = &evaluation_event.payload else {
            report
                .violations
                .push(EffectBindingViolation::EvaluationPayloadMissing {
                    evaluation_event_id: evaluation_event.event_id,
                });
            continue;
        };

        if receipt.stage != ReceiptStage::Evaluation {
            report
                .violations
                .push(EffectBindingViolation::EvaluationReceiptStageMismatch {
                    evaluation_event_id: evaluation_event.event_id,
                });
        }

        if observed_digest != receipt.mutation_digest {
            report
                .violations
                .push(EffectBindingViolation::EffectDigestMismatch {
                    observed_event_id: observed_event.event_id,
                    evaluation_event_id: evaluation_event.event_id,
                    observed: observed_digest,
                    evaluated: receipt.mutation_digest,
                });
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    use symthaea_cogsec::{
        CognitiveSecurityLabel, Consequence, DecisionOutcome, MutationKind, MutationReceiptRecord,
        PrincipalId, ReasonCode, ResourceId,
    };

    use crate::{
        EvidenceCompleteness, EvidenceConfidentiality, LedgerStats, PrincipalContext,
        QualificationManifest, SHADOW_EVENT_SCHEMA_V1, ShadowEvent,
    };

    fn d(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn receipt() -> MutationReceiptRecord {
        MutationReceiptRecord {
            stage: ReceiptStage::Evaluation,
            request_id: d(1),
            subject: PrincipalId("local-user".into()),
            kind: MutationKind::GoalActivation,
            resource: ResourceId("mind/goals".into()),
            mutation_digest: d(2),
            consequence: Consequence::High,
            input_label: CognitiveSecurityLabel::default(),
            expected_resource_state_root: d(3),
            observed_resource_state_root: d(3),
            expected_policy_root: d(4),
            evaluated_policy_root: d(4),
            trusted_policy_root: d(4),
            policy_epoch: 7,
            authorization_epoch: 11,
            revocation_epoch: 13,
            sequence: 42,
            capability_id: None,
            outcome: DecisionOutcome::Allow,
            reasons: Vec::<ReasonCode>::new(),
        }
    }

    fn evaluation_event() -> ShadowEvent {
        ShadowEvent {
            schema_version: SHADOW_EVENT_SCHEMA_V1,
            event_id: EventId {
                ledger_epoch: 1,
                sequence: 1,
            },
            proposal_id: None,
            transaction_id: None,
            principals: PrincipalContext::default(),
            kind: ShadowEventKind::GoalActivationEvaluated,
            resource: Some(ResourceId("mind/goals".into())),
            resource_version_before: None,
            resource_version_after: None,
            state_root_before: Some(d(3)),
            state_root_after: None,
            policy_root: Some(d(4)),
            policy_epoch: Some(7),
            authorization_epoch: Some(11),
            revocation_epoch: Some(13),
            causal_parents: BTreeSet::new(),
            cognitive_tick: None,
            run_id: None,
            confidentiality: EvidenceConfidentiality::LocalPrivate,
            payload: ShadowEventPayload::Evaluation { receipt: receipt() },
        }
    }

    fn observed_event() -> ShadowEvent {
        let mut parents = BTreeSet::new();
        parents.insert(EventId {
            ledger_epoch: 1,
            sequence: 1,
        });
        ShadowEvent {
            schema_version: SHADOW_EVENT_SCHEMA_V1,
            event_id: EventId {
                ledger_epoch: 1,
                sequence: 2,
            },
            proposal_id: None,
            transaction_id: None,
            principals: PrincipalContext::default(),
            kind: ShadowEventKind::GoalActivationObserved,
            resource: Some(ResourceId("mind/goals".into())),
            resource_version_before: None,
            resource_version_after: None,
            state_root_before: Some(d(3)),
            state_root_after: Some(d(5)),
            policy_root: Some(d(4)),
            policy_epoch: Some(7),
            authorization_epoch: Some(11),
            revocation_epoch: Some(13),
            causal_parents: parents,
            cognitive_tick: None,
            run_id: None,
            confidentiality: EvidenceConfidentiality::LocalPrivate,
            payload: ShadowEventPayload::MutationObserved { applied: true },
        }
    }

    fn base_snapshot(events: Vec<ShadowEvent>) -> EvidenceLedgerSnapshot {
        EvidenceLedgerSnapshot {
            schema_version: SHADOW_EVENT_SCHEMA_V1,
            ledger_epoch: 1,
            last_assigned_sequence: events.len() as u64,
            manifest: QualificationManifest::new([], []),
            completeness: EvidenceCompleteness::Complete,
            stats: LedgerStats {
                assigned_sequences: events.len() as u64,
                stored_events: events.len() as u64,
                ..LedgerStats::default()
            },
            events,
        }
    }

    fn bound_snapshot(effect_digest: Digest32) -> EffectBoundEvidenceSnapshot {
        EffectBoundEvidenceSnapshot::new(
            base_snapshot(vec![evaluation_event(), observed_event()]),
            vec![ObservedEffectBinding {
                observed_event_id: EventId {
                    ledger_epoch: 1,
                    sequence: 2,
                },
                effect_digest,
            }],
        )
    }

    #[test]
    fn exact_matching_effect_binding_qualifies() {
        let report = validate_effect_bound_snapshot(&bound_snapshot(d(2)));
        assert!(report.effect_bindings_are_consistent());
    }

    #[test]
    fn mismatched_effect_digest_is_a_hard_violation() {
        let report = validate_effect_bound_snapshot(&bound_snapshot(d(99)));
        assert!(report.violations.iter().any(|violation| matches!(
            violation,
            EffectBindingViolation::EffectDigestMismatch {
                observed: value,
                evaluated,
                ..
            } if *value == d(99) && *evaluated == d(2)
        )));
    }

    #[test]
    fn paired_observation_requires_exactly_one_sidecar_binding() {
        let snapshot = EffectBoundEvidenceSnapshot::new(
            base_snapshot(vec![evaluation_event(), observed_event()]),
            Vec::new(),
        );
        let report = validate_effect_bound_snapshot(&snapshot);
        assert!(report.violations.iter().any(|violation| matches!(
            violation,
            EffectBindingViolation::MissingObservedBinding { .. }
        )));
    }

    #[test]
    fn duplicate_binding_is_rejected() {
        let id = EventId {
            ledger_epoch: 1,
            sequence: 2,
        };
        let snapshot = EffectBoundEvidenceSnapshot::new(
            base_snapshot(vec![evaluation_event(), observed_event()]),
            vec![
                ObservedEffectBinding {
                    observed_event_id: id,
                    effect_digest: d(2),
                },
                ObservedEffectBinding {
                    observed_event_id: id,
                    effect_digest: d(2),
                },
            ],
        );
        let report = validate_effect_bound_snapshot(&snapshot);
        assert!(report.violations.iter().any(|violation| matches!(
            violation,
            EffectBindingViolation::DuplicateObservedBinding { .. }
        )));
    }

    #[test]
    fn binding_for_unpaired_event_is_rejected() {
        let evaluation = evaluation_event();
        let id = evaluation.event_id;
        let snapshot = EffectBoundEvidenceSnapshot::new(
            base_snapshot(vec![evaluation]),
            vec![ObservedEffectBinding {
                observed_event_id: id,
                effect_digest: d(2),
            }],
        );
        let report = validate_effect_bound_snapshot(&snapshot);
        assert!(report.violations.iter().any(|violation| matches!(
            violation,
            EffectBindingViolation::BindingForUnpairedEvent { .. }
        )));
    }

    #[test]
    fn portable_effect_binding_roundtrips_without_becoming_authority() {
        let snapshot = bound_snapshot(d(2));
        let encoded = serde_json::to_vec(&snapshot).unwrap();
        let decoded: EffectBoundEvidenceSnapshot = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, snapshot);
        assert!(validate_effect_bound_snapshot(&decoded).effect_bindings_are_consistent());
    }
}
