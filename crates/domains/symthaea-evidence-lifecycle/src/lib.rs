// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Time- and scope-aware lifecycle semantics for verified safety evidence.
//!
//! A receipt can be content-valid while no longer being applicable to the
//! current deployment. This crate binds receipts to an exact reviewed safety
//! contract and explicit validity interval, then evaluates revocation,
//! supersession, contradiction, and contradiction-resolution events.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use symthaea_formal_safety::{
    SafetyCase, SafetyEvidenceReceipt, StrictSafetyCaseReport, StrictSafetyCaseStatus,
    assess_strict_safety_case,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopedSafetyEvidenceReceipt {
    pub receipt: SafetyEvidenceReceipt,
    /// Exact `SafetyCase::contract_digest()` for which the evidence was reviewed.
    pub contract_digest: String,
    pub valid_from_ms: u64,
    pub valid_until_ms: u64,
    /// Deployment/configuration applicability evidence.
    pub applicability_refs: Vec<String>,
}

impl ScopedSafetyEvidenceReceipt {
    pub fn validate(&self) -> bool {
        self.receipt.validate()
            && !self.contract_digest.trim().is_empty()
            && self.valid_from_ms <= self.valid_until_ms
            && !self.applicability_refs.is_empty()
            && self
                .applicability_refs
                .iter()
                .all(|value| !value.trim().is_empty())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceLifecycleEventKind {
    Revoked { reason_ref: String },
    Superseded {
        by_receipt_id: String,
        reason_ref: String,
    },
    Contradicted {
        contradiction_id: String,
        contradiction_ref: String,
    },
    /// Clears the contradiction block after reviewed disposition, but does not
    /// reactivate the contradicted receipt.
    ContradictionResolved {
        contradiction_id: String,
        resolution_ref: String,
    },
}

impl EvidenceLifecycleEventKind {
    fn validate(&self, receipt_id: &str) -> bool {
        match self {
            Self::Revoked { reason_ref } => !reason_ref.trim().is_empty(),
            Self::Superseded {
                by_receipt_id,
                reason_ref,
            } => {
                !by_receipt_id.trim().is_empty()
                    && by_receipt_id != receipt_id
                    && !reason_ref.trim().is_empty()
            }
            Self::Contradicted {
                contradiction_id,
                contradiction_ref,
            } => !contradiction_id.trim().is_empty() && !contradiction_ref.trim().is_empty(),
            Self::ContradictionResolved {
                contradiction_id,
                resolution_ref,
            } => !contradiction_id.trim().is_empty() && !resolution_ref.trim().is_empty(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceLifecycleEvent {
    pub event_id: String,
    pub receipt_id: String,
    pub event_at_ms: u64,
    pub kind: EvidenceLifecycleEventKind,
}

impl EvidenceLifecycleEvent {
    pub fn validate(&self) -> bool {
        !self.event_id.trim().is_empty()
            && !self.receipt_id.trim().is_empty()
            && self.kind.validate(&self.receipt_id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReceiptExclusionReason {
    NotYetValid,
    Expired,
    Revoked,
    Superseded,
    Contradicted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExcludedSafetyReceipt {
    pub receipt_id: String,
    pub reason: ReceiptExclusionReason,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceLifecycleIssue {
    InvalidScopedReceipt(String),
    DuplicateReceiptId(String),
    ContractDigestMismatch {
        receipt_id: String,
        expected: String,
        observed: String,
    },
    ReceiptVerifiedInFuture {
        receipt_id: String,
        verified_at_ms: u64,
        assessed_at_ms: u64,
    },
    InvalidLifecycleEvent(String),
    DuplicateLifecycleEventId(String),
    UnknownLifecycleReceipt(String),
    FutureLifecycleEvent {
        event_id: String,
        event_at_ms: u64,
        assessed_at_ms: u64,
    },
    UnknownSupersedingReceipt {
        event_id: String,
        receipt_id: String,
    },
    SupersedingReceiptNotUsableAtEvent {
        event_id: String,
        receipt_id: String,
    },
    InvalidLifecycleSequence {
        event_id: String,
        receipt_id: String,
    },
    DuplicateContradictionId(String),
    UnknownContradictionResolution {
        event_id: String,
        contradiction_id: String,
    },
    UnresolvedContradiction {
        receipt_id: String,
        contradiction_id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LifecycleSafetyCaseReport {
    pub contract_digest: String,
    pub assessed_at_ms: u64,
    pub status: StrictSafetyCaseStatus,
    pub active_receipt_count: usize,
    pub excluded_receipts: Vec<ExcludedSafetyReceipt>,
    pub unresolved_contradiction_count: usize,
    pub issues: Vec<EvidenceLifecycleIssue>,
    pub strict_report: StrictSafetyCaseReport,
}

impl LifecycleSafetyCaseReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TerminalReceiptState {
    Revoked,
    Superseded,
    Contradicted,
}

#[derive(Debug, Default)]
struct ReceiptRuntimeState {
    terminal: Option<TerminalReceiptState>,
    open_contradictions: BTreeSet<String>,
}

/// Lifecycle-filter strict safety evidence at one trusted assessment time.
///
/// Expiry, revocation, and supersession remove receipts from the active set.
/// Another current receipt may still satisfy the obligation. An unresolved
/// contradiction is stronger: it blocks readiness even if another favorable
/// receipt exists, until a reviewed resolution is recorded.
pub fn assess_lifecycle_safety_case(
    safety_case: &SafetyCase,
    scoped_receipts: &[ScopedSafetyEvidenceReceipt],
    events: &[EvidenceLifecycleEvent],
    assessed_at_ms: u64,
) -> LifecycleSafetyCaseReport {
    let contract_digest = safety_case.contract_digest();
    let mut issues = Vec::new();
    let mut by_id = BTreeMap::<String, &ScopedSafetyEvidenceReceipt>::new();
    let mut invalid_receipts = BTreeSet::<String>::new();

    for scoped in scoped_receipts {
        let receipt_id = scoped.receipt.receipt_id.clone();
        if !scoped.validate() {
            issues.push(EvidenceLifecycleIssue::InvalidScopedReceipt(receipt_id.clone()));
            invalid_receipts.insert(receipt_id.clone());
        }
        if scoped.contract_digest != contract_digest {
            issues.push(EvidenceLifecycleIssue::ContractDigestMismatch {
                receipt_id: receipt_id.clone(),
                expected: contract_digest.clone(),
                observed: scoped.contract_digest.clone(),
            });
            invalid_receipts.insert(receipt_id.clone());
        }
        if scoped.receipt.verified_at_ms > assessed_at_ms {
            issues.push(EvidenceLifecycleIssue::ReceiptVerifiedInFuture {
                receipt_id: receipt_id.clone(),
                verified_at_ms: scoped.receipt.verified_at_ms,
                assessed_at_ms,
            });
            invalid_receipts.insert(receipt_id.clone());
        }
        if by_id.insert(receipt_id.clone(), scoped).is_some() {
            issues.push(EvidenceLifecycleIssue::DuplicateReceiptId(receipt_id.clone()));
            invalid_receipts.insert(receipt_id);
        }
    }

    let mut event_ids = BTreeSet::new();
    let mut contradiction_ids = BTreeSet::new();
    let mut valid_events = Vec::new();

    for event in events {
        if !event.validate() {
            issues.push(EvidenceLifecycleIssue::InvalidLifecycleEvent(
                event.event_id.clone(),
            ));
            continue;
        }
        if !event_ids.insert(event.event_id.clone()) {
            issues.push(EvidenceLifecycleIssue::DuplicateLifecycleEventId(
                event.event_id.clone(),
            ));
            continue;
        }
        if !by_id.contains_key(&event.receipt_id) {
            issues.push(EvidenceLifecycleIssue::UnknownLifecycleReceipt(
                event.receipt_id.clone(),
            ));
            continue;
        }
        if event.event_at_ms > assessed_at_ms {
            issues.push(EvidenceLifecycleIssue::FutureLifecycleEvent {
                event_id: event.event_id.clone(),
                event_at_ms: event.event_at_ms,
                assessed_at_ms,
            });
            continue;
        }
        if let EvidenceLifecycleEventKind::Contradicted {
            contradiction_id, ..
        } = &event.kind
        {
            if !contradiction_ids.insert(contradiction_id.clone()) {
                issues.push(EvidenceLifecycleIssue::DuplicateContradictionId(
                    contradiction_id.clone(),
                ));
                continue;
            }
        }
        valid_events.push(event);
    }

    valid_events.sort_by(|a, b| {
        a.event_at_ms
            .cmp(&b.event_at_ms)
            .then_with(|| a.event_id.cmp(&b.event_id))
    });

    let mut states = by_id
        .keys()
        .cloned()
        .map(|id| (id, ReceiptRuntimeState::default()))
        .collect::<BTreeMap<_, _>>();
    let mut contradiction_owner = BTreeMap::<String, String>::new();

    for event in valid_events {
        let Some(state) = states.get_mut(&event.receipt_id) else {
            continue;
        };
        match &event.kind {
            EvidenceLifecycleEventKind::Revoked { .. } => {
                if state.terminal.is_some() {
                    issues.push(EvidenceLifecycleIssue::InvalidLifecycleSequence {
                        event_id: event.event_id.clone(),
                        receipt_id: event.receipt_id.clone(),
                    });
                } else {
                    state.terminal = Some(TerminalReceiptState::Revoked);
                }
            }
            EvidenceLifecycleEventKind::Superseded { by_receipt_id, .. } => {
                if state.terminal.is_some() {
                    issues.push(EvidenceLifecycleIssue::InvalidLifecycleSequence {
                        event_id: event.event_id.clone(),
                        receipt_id: event.receipt_id.clone(),
                    });
                    continue;
                }
                let Some(replacement) = by_id.get(by_receipt_id) else {
                    issues.push(EvidenceLifecycleIssue::UnknownSupersedingReceipt {
                        event_id: event.event_id.clone(),
                        receipt_id: by_receipt_id.clone(),
                    });
                    continue;
                };
                if invalid_receipts.contains(by_receipt_id)
                    || replacement.receipt.verified_at_ms > event.event_at_ms
                    || event.event_at_ms < replacement.valid_from_ms
                    || event.event_at_ms > replacement.valid_until_ms
                {
                    issues.push(EvidenceLifecycleIssue::SupersedingReceiptNotUsableAtEvent {
                        event_id: event.event_id.clone(),
                        receipt_id: by_receipt_id.clone(),
                    });
                    continue;
                }
                state.terminal = Some(TerminalReceiptState::Superseded);
            }
            EvidenceLifecycleEventKind::Contradicted {
                contradiction_id, ..
            } => {
                if state.terminal.is_some() {
                    issues.push(EvidenceLifecycleIssue::InvalidLifecycleSequence {
                        event_id: event.event_id.clone(),
                        receipt_id: event.receipt_id.clone(),
                    });
                    continue;
                }
                state.terminal = Some(TerminalReceiptState::Contradicted);
                state.open_contradictions.insert(contradiction_id.clone());
                contradiction_owner.insert(contradiction_id.clone(), event.receipt_id.clone());
            }
            EvidenceLifecycleEventKind::ContradictionResolved {
                contradiction_id, ..
            } => {
                let Some(owner) = contradiction_owner.get(contradiction_id) else {
                    issues.push(EvidenceLifecycleIssue::UnknownContradictionResolution {
                        event_id: event.event_id.clone(),
                        contradiction_id: contradiction_id.clone(),
                    });
                    continue;
                };
                if owner != &event.receipt_id
                    || !state.open_contradictions.remove(contradiction_id)
                {
                    issues.push(EvidenceLifecycleIssue::UnknownContradictionResolution {
                        event_id: event.event_id.clone(),
                        contradiction_id: contradiction_id.clone(),
                    });
                }
            }
        }
    }

    let mut unresolved_contradiction_count = 0usize;
    for (receipt_id, state) in &states {
        for contradiction_id in &state.open_contradictions {
            unresolved_contradiction_count = unresolved_contradiction_count.saturating_add(1);
            issues.push(EvidenceLifecycleIssue::UnresolvedContradiction {
                receipt_id: receipt_id.clone(),
                contradiction_id: contradiction_id.clone(),
            });
        }
    }

    let mut active_receipts = Vec::new();
    let mut excluded_receipts = Vec::new();
    for (receipt_id, scoped) in &by_id {
        if invalid_receipts.contains(receipt_id) {
            continue;
        }
        let state = states
            .get(receipt_id)
            .expect("receipt state created from receipt id set");
        if let Some(terminal) = state.terminal {
            let reason = match terminal {
                TerminalReceiptState::Revoked => ReceiptExclusionReason::Revoked,
                TerminalReceiptState::Superseded => ReceiptExclusionReason::Superseded,
                TerminalReceiptState::Contradicted => ReceiptExclusionReason::Contradicted,
            };
            excluded_receipts.push(ExcludedSafetyReceipt {
                receipt_id: receipt_id.clone(),
                reason,
            });
            continue;
        }
        if assessed_at_ms < scoped.valid_from_ms {
            excluded_receipts.push(ExcludedSafetyReceipt {
                receipt_id: receipt_id.clone(),
                reason: ReceiptExclusionReason::NotYetValid,
            });
            continue;
        }
        if assessed_at_ms > scoped.valid_until_ms {
            excluded_receipts.push(ExcludedSafetyReceipt {
                receipt_id: receipt_id.clone(),
                reason: ReceiptExclusionReason::Expired,
            });
            continue;
        }
        active_receipts.push(scoped.receipt.clone());
    }

    excluded_receipts.sort_by(|a, b| a.receipt_id.cmp(&b.receipt_id));
    let strict_report = assess_strict_safety_case(safety_case, &active_receipts);
    let structurally_invalid = issues.iter().any(|issue| {
        matches!(
            issue,
            EvidenceLifecycleIssue::InvalidScopedReceipt(_)
                | EvidenceLifecycleIssue::DuplicateReceiptId(_)
                | EvidenceLifecycleIssue::ContractDigestMismatch { .. }
                | EvidenceLifecycleIssue::ReceiptVerifiedInFuture { .. }
                | EvidenceLifecycleIssue::InvalidLifecycleEvent(_)
                | EvidenceLifecycleIssue::DuplicateLifecycleEventId(_)
                | EvidenceLifecycleIssue::UnknownLifecycleReceipt(_)
                | EvidenceLifecycleIssue::FutureLifecycleEvent { .. }
                | EvidenceLifecycleIssue::UnknownSupersedingReceipt { .. }
                | EvidenceLifecycleIssue::SupersedingReceiptNotUsableAtEvent { .. }
                | EvidenceLifecycleIssue::InvalidLifecycleSequence { .. }
                | EvidenceLifecycleIssue::DuplicateContradictionId(_)
                | EvidenceLifecycleIssue::UnknownContradictionResolution { .. }
        )
    });
    let status = if structurally_invalid || strict_report.status == StrictSafetyCaseStatus::Invalid {
        StrictSafetyCaseStatus::Invalid
    } else if unresolved_contradiction_count > 0 {
        StrictSafetyCaseStatus::Blocked
    } else {
        strict_report.status
    };

    LifecycleSafetyCaseReport {
        contract_digest,
        assessed_at_ms,
        status,
        active_receipt_count: active_receipts.len(),
        excluded_receipts,
        unresolved_contradiction_count,
        issues,
        strict_report,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_formal_safety::{EvidenceKind, ProofObligation};

    fn safety_case(subject: &str) -> SafetyCase {
        let mut case = SafetyCase::new(subject);
        case.add_obligation(
            ProofObligation::new("controlled claim", EvidenceKind::Test).discharge("legacy:test"),
        );
        case
    }

    fn scoped(
        case: &SafetyCase,
        id: &str,
        valid_from_ms: u64,
        valid_until_ms: u64,
    ) -> ScopedSafetyEvidenceReceipt {
        let obligation = &case.obligations[0];
        ScopedSafetyEvidenceReceipt {
            receipt: SafetyEvidenceReceipt {
                receipt_id: id.into(),
                obligation_key: obligation.stable_key(),
                evidence_kind: obligation.expected_evidence,
                evidence_ref: format!("evidence:{id}"),
                evidence_digest: format!("blake3:{id}"),
                verifier_ref: "verifier:independent".into(),
                verified_at_ms: 100,
            },
            contract_digest: case.contract_digest(),
            valid_from_ms,
            valid_until_ms,
            applicability_refs: vec!["deployment-config:v1".into()],
        }
    }

    #[test]
    fn active_scoped_receipt_can_satisfy_strict_readiness() {
        let case = safety_case("harbor-a");
        let report = assess_lifecycle_safety_case(
            &case,
            &[scoped(&case, "r1", 100, 2_000)],
            &[],
            1_000,
        );
        assert_eq!(report.status, StrictSafetyCaseStatus::Ready);
        assert_eq!(report.active_receipt_count, 1);
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn receipt_cannot_cross_safety_contracts() {
        let case_a = safety_case("harbor-a");
        let case_b = safety_case("harbor-b");
        let report = assess_lifecycle_safety_case(
            &case_b,
            &[scoped(&case_a, "r1", 100, 2_000)],
            &[],
            1_000,
        );
        assert_eq!(report.status, StrictSafetyCaseStatus::Invalid);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            EvidenceLifecycleIssue::ContractDigestMismatch { .. }
        )));
    }

    #[test]
    fn expired_receipt_no_longer_satisfies_readiness() {
        let case = safety_case("harbor-a");
        let report = assess_lifecycle_safety_case(
            &case,
            &[scoped(&case, "r1", 100, 900)],
            &[],
            1_000,
        );
        assert_eq!(report.status, StrictSafetyCaseStatus::Blocked);
        assert_eq!(report.active_receipt_count, 0);
        assert_eq!(report.excluded_receipts[0].reason, ReceiptExclusionReason::Expired);
    }

    #[test]
    fn alternate_active_receipt_can_replace_expired_history() {
        let case = safety_case("harbor-a");
        let report = assess_lifecycle_safety_case(
            &case,
            &[
                scoped(&case, "old", 100, 900),
                scoped(&case, "new", 900, 2_000),
            ],
            &[],
            1_000,
        );
        assert_eq!(report.status, StrictSafetyCaseStatus::Ready);
        assert_eq!(report.active_receipt_count, 1);
    }

    #[test]
    fn supersession_can_move_readiness_to_new_receipt() {
        let case = safety_case("harbor-a");
        let old = scoped(&case, "old", 100, 2_000);
        let mut new = scoped(&case, "new", 400, 3_000);
        new.receipt.verified_at_ms = 400;
        let event = EvidenceLifecycleEvent {
            event_id: "ev1".into(),
            receipt_id: "old".into(),
            event_at_ms: 500,
            kind: EvidenceLifecycleEventKind::Superseded {
                by_receipt_id: "new".into(),
                reason_ref: "change:recalibration".into(),
            },
        };
        let report = assess_lifecycle_safety_case(&case, &[old, new], &[event], 1_000);
        assert_eq!(report.status, StrictSafetyCaseStatus::Ready);
        assert_eq!(report.active_receipt_count, 1);
    }

    #[test]
    fn unresolved_contradiction_blocks_even_with_alternate_favorable_receipt() {
        let case = safety_case("harbor-a");
        let event = EvidenceLifecycleEvent {
            event_id: "ev1".into(),
            receipt_id: "r1".into(),
            event_at_ms: 500,
            kind: EvidenceLifecycleEventKind::Contradicted {
                contradiction_id: "c1".into(),
                contradiction_ref: "incident:counterexample".into(),
            },
        };
        let report = assess_lifecycle_safety_case(
            &case,
            &[
                scoped(&case, "r1", 100, 2_000),
                scoped(&case, "r2", 100, 2_000),
            ],
            &[event],
            1_000,
        );
        assert_eq!(report.strict_report.status, StrictSafetyCaseStatus::Ready);
        assert_eq!(report.status, StrictSafetyCaseStatus::Blocked);
        assert_eq!(report.unresolved_contradiction_count, 1);
    }

    #[test]
    fn resolved_contradiction_does_not_reactivate_old_receipt() {
        let case = safety_case("harbor-a");
        let events = vec![
            EvidenceLifecycleEvent {
                event_id: "ev1".into(),
                receipt_id: "old".into(),
                event_at_ms: 500,
                kind: EvidenceLifecycleEventKind::Contradicted {
                    contradiction_id: "c1".into(),
                    contradiction_ref: "incident:counterexample".into(),
                },
            },
            EvidenceLifecycleEvent {
                event_id: "ev2".into(),
                receipt_id: "old".into(),
                event_at_ms: 700,
                kind: EvidenceLifecycleEventKind::ContradictionResolved {
                    contradiction_id: "c1".into(),
                    resolution_ref: "review:root-cause-and-retest".into(),
                },
            },
        ];
        let no_replacement = assess_lifecycle_safety_case(
            &case,
            &[scoped(&case, "old", 100, 2_000)],
            &events,
            1_000,
        );
        assert_eq!(no_replacement.unresolved_contradiction_count, 0);
        assert_eq!(no_replacement.status, StrictSafetyCaseStatus::Blocked);

        let with_replacement = assess_lifecycle_safety_case(
            &case,
            &[
                scoped(&case, "old", 100, 2_000),
                scoped(&case, "replacement", 700, 2_000),
            ],
            &events,
            1_000,
        );
        assert_eq!(with_replacement.status, StrictSafetyCaseStatus::Ready);
        assert!(with_replacement.excluded_receipts.iter().any(|entry| {
            entry.receipt_id == "old" && entry.reason == ReceiptExclusionReason::Contradicted
        }));
    }

    #[test]
    fn future_lifecycle_event_is_invalid_not_ignored() {
        let case = safety_case("harbor-a");
        let event = EvidenceLifecycleEvent {
            event_id: "future".into(),
            receipt_id: "r1".into(),
            event_at_ms: 2_000,
            kind: EvidenceLifecycleEventKind::Revoked {
                reason_ref: "future:revocation".into(),
            },
        };
        let report = assess_lifecycle_safety_case(
            &case,
            &[scoped(&case, "r1", 100, 3_000)],
            &[event],
            1_000,
        );
        assert_eq!(report.status, StrictSafetyCaseStatus::Invalid);
    }
}
