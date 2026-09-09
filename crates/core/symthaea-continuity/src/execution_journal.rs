// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Crash-reconstructed executor anti-replay journal.
//!
//! Session-local capability replay protection is insufficient after process or host
//! restart. This module reconstructs the spent-eligibility world from durable
//! pre-mutation intents plus optional result receipts.
//!
//! Core theorem:
//!
//! `DurableIntent != PhysicalSuccess`, but `DurableIntent == EligibilitySpent`.
//!
//! Therefore an intent without a receipt is not retry permission. It is an explicit
//! reconciliation obligation.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::execution_capability::{
    ExecutionAttemptId, ExecutionAttemptIntentV1, ExecutionAttemptOutcomeV1,
    ExecutionAttemptReceiptId, ExecutionAttemptReceiptV1, ExecutionCapabilityError,
};
use crate::trusted_commit_epoch::TrustedCommitEligibilityId;

const JOURNAL_DOMAIN: &[u8] = b"symthaea.continuity.execution-journal.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ExecutionJournalDigest([u8; 32]);

impl ExecutionJournalDigest {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JournalAttemptDispositionV1 {
    /// Intent exists and no validated receipt exists. Actual physical state must be
    /// reconciled before another transition involving the spent eligibility.
    AwaitingReconciliation,
    /// One validated result receipt exists for the attempt.
    Completed(ExecutionAttemptOutcomeV1),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JournalAttemptEntryV1 {
    attempt_id: ExecutionAttemptId,
    trusted_eligibility_id: TrustedCommitEligibilityId,
    receipt_id: Option<ExecutionAttemptReceiptId>,
    disposition: JournalAttemptDispositionV1,
}

impl JournalAttemptEntryV1 {
    pub fn attempt_id(&self) -> ExecutionAttemptId {
        self.attempt_id
    }

    pub fn trusted_eligibility_id(&self) -> TrustedCommitEligibilityId {
        self.trusted_eligibility_id
    }

    pub fn receipt_id(&self) -> Option<ExecutionAttemptReceiptId> {
        self.receipt_id
    }

    pub fn disposition(&self) -> JournalAttemptDispositionV1 {
        self.disposition
    }
}

/// Non-Serde reconstructed journal state. It is derived from persisted evidence and
/// should be rebuilt after restart rather than serialized as a second source of truth.
#[derive(Debug, Clone)]
pub struct ReconstructedExecutionJournalV1 {
    entries: BTreeMap<ExecutionAttemptId, JournalAttemptEntryV1>,
    spent_eligibilities: BTreeMap<TrustedCommitEligibilityId, ExecutionAttemptId>,
    digest: ExecutionJournalDigest,
}

impl ReconstructedExecutionJournalV1 {
    /// Reconstruct the exact executor journal from persisted pre-mutation intents and
    /// result receipts. Every artifact validates itself before entering the state.
    pub fn reconstruct(
        intents: &[ExecutionAttemptIntentV1],
        receipts: &[ExecutionAttemptReceiptV1],
    ) -> Result<Self, ExecutionJournalError> {
        let mut intent_refs = Vec::with_capacity(intents.len());
        for intent in intents {
            intent.validate()?;
            intent_refs.push(JournalIntentRef {
                attempt_id: intent.id(),
                trusted_eligibility_id: intent.trusted_eligibility_id(),
            });
        }

        let mut receipt_refs = Vec::with_capacity(receipts.len());
        for receipt in receipts {
            receipt.validate()?;
            receipt_refs.push(JournalReceiptRef {
                attempt_id: receipt.attempt_id(),
                receipt_id: receipt.id(),
                outcome: receipt.outcome(),
            });
        }

        reconstruct_from_refs(&intent_refs, &receipt_refs)
    }

    pub fn digest(&self) -> ExecutionJournalDigest {
        self.digest
    }

    pub fn entry(&self, attempt_id: ExecutionAttemptId) -> Option<&JournalAttemptEntryV1> {
        self.entries.get(&attempt_id)
    }

    pub fn attempt_for_eligibility(
        &self,
        eligibility_id: TrustedCommitEligibilityId,
    ) -> Option<ExecutionAttemptId> {
        self.spent_eligibilities.get(&eligibility_id).copied()
    }

    /// Once a durable intent exists, the eligibility is spent even if execution
    /// result is missing, failed, rolled back, or indeterminate.
    pub fn eligibility_is_spent(&self, eligibility_id: TrustedCommitEligibilityId) -> bool {
        self.spent_eligibilities.contains_key(&eligibility_id)
    }

    pub fn pending_reconciliation_attempts(&self) -> Vec<ExecutionAttemptId> {
        self.entries
            .values()
            .filter_map(|entry| {
                (entry.disposition == JournalAttemptDispositionV1::AwaitingReconciliation)
                    .then_some(entry.attempt_id)
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ExecutionJournalError {
    #[error(transparent)]
    Artifact(#[from] ExecutionCapabilityError),
    #[error("duplicate execution intent for attempt {attempt_id:?}")]
    DuplicateIntent { attempt_id: ExecutionAttemptId },
    #[error(
        "trusted eligibility {eligibility_id:?} appears in multiple durable attempts: {first_attempt:?} and {second_attempt:?}"
    )]
    EligibilityReplayConflict {
        eligibility_id: TrustedCommitEligibilityId,
        first_attempt: ExecutionAttemptId,
        second_attempt: ExecutionAttemptId,
    },
    #[error("execution receipt {receipt_id:?} has no corresponding durable intent for attempt {attempt_id:?}")]
    ReceiptWithoutIntent {
        receipt_id: ExecutionAttemptReceiptId,
        attempt_id: ExecutionAttemptId,
    },
    #[error("attempt {attempt_id:?} has more than one execution receipt")]
    DuplicateReceipt { attempt_id: ExecutionAttemptId },
}

#[derive(Debug, Clone, Copy)]
struct JournalIntentRef {
    attempt_id: ExecutionAttemptId,
    trusted_eligibility_id: TrustedCommitEligibilityId,
}

#[derive(Debug, Clone, Copy)]
struct JournalReceiptRef {
    attempt_id: ExecutionAttemptId,
    receipt_id: ExecutionAttemptReceiptId,
    outcome: ExecutionAttemptOutcomeV1,
}

fn reconstruct_from_refs(
    intents: &[JournalIntentRef],
    receipts: &[JournalReceiptRef],
) -> Result<ReconstructedExecutionJournalV1, ExecutionJournalError> {
    let mut entries = BTreeMap::<ExecutionAttemptId, JournalAttemptEntryV1>::new();
    let mut spent_eligibilities =
        BTreeMap::<TrustedCommitEligibilityId, ExecutionAttemptId>::new();

    for intent in intents {
        if entries.contains_key(&intent.attempt_id) {
            return Err(ExecutionJournalError::DuplicateIntent {
                attempt_id: intent.attempt_id,
            });
        }
        if let Some(first_attempt) = spent_eligibilities
            .insert(intent.trusted_eligibility_id, intent.attempt_id)
        {
            if first_attempt != intent.attempt_id {
                return Err(ExecutionJournalError::EligibilityReplayConflict {
                    eligibility_id: intent.trusted_eligibility_id,
                    first_attempt,
                    second_attempt: intent.attempt_id,
                });
            }
        }
        entries.insert(
            intent.attempt_id,
            JournalAttemptEntryV1 {
                attempt_id: intent.attempt_id,
                trusted_eligibility_id: intent.trusted_eligibility_id,
                receipt_id: None,
                disposition: JournalAttemptDispositionV1::AwaitingReconciliation,
            },
        );
    }

    for receipt in receipts {
        let Some(entry) = entries.get_mut(&receipt.attempt_id) else {
            return Err(ExecutionJournalError::ReceiptWithoutIntent {
                receipt_id: receipt.receipt_id,
                attempt_id: receipt.attempt_id,
            });
        };
        if entry.receipt_id.is_some() {
            return Err(ExecutionJournalError::DuplicateReceipt {
                attempt_id: receipt.attempt_id,
            });
        }
        entry.receipt_id = Some(receipt.receipt_id);
        entry.disposition = JournalAttemptDispositionV1::Completed(receipt.outcome);
    }

    let digest = ExecutionJournalDigest(hash_journal(&entries));
    Ok(ReconstructedExecutionJournalV1 {
        entries,
        spent_eligibilities,
        digest,
    })
}

fn hash_journal(entries: &BTreeMap<ExecutionAttemptId, JournalAttemptEntryV1>) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(JOURNAL_DOMAIN);
    hasher.update(&(entries.len() as u64).to_le_bytes());
    for entry in entries.values() {
        hasher.update(entry.attempt_id.as_bytes());
        hasher.update(entry.trusted_eligibility_id.as_bytes());
        match entry.receipt_id {
            Some(receipt_id) => {
                hasher.update(&[1]);
                hasher.update(receipt_id.as_bytes());
            }
            None => {
                hasher.update(&[0]);
            }
        }
        match entry.disposition {
            JournalAttemptDispositionV1::AwaitingReconciliation => {
                hasher.update(&[0]);
            }
            JournalAttemptDispositionV1::Completed(outcome) => {
                hasher.update(&[1]);
                hasher.update(&[outcome_tag(outcome)]);
            }
        }
    }
    *hasher.finalize().as_bytes()
}

fn outcome_tag(outcome: ExecutionAttemptOutcomeV1) -> u8 {
    match outcome {
        ExecutionAttemptOutcomeV1::Succeeded => 1,
        ExecutionAttemptOutcomeV1::Failed => 2,
        ExecutionAttemptOutcomeV1::RolledBack => 3,
        ExecutionAttemptOutcomeV1::Indeterminate => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attempt(seed: u8) -> ExecutionAttemptId {
        serde_json::from_value(serde_json::json!([seed; 32])).unwrap()
    }

    fn eligibility(seed: u8) -> TrustedCommitEligibilityId {
        serde_json::from_value(serde_json::json!([seed; 32])).unwrap()
    }

    fn receipt(seed: u8) -> ExecutionAttemptReceiptId {
        serde_json::from_value(serde_json::json!([seed; 32])).unwrap()
    }

    fn intent_ref(attempt_seed: u8, eligibility_seed: u8) -> JournalIntentRef {
        JournalIntentRef {
            attempt_id: attempt(attempt_seed),
            trusted_eligibility_id: eligibility(eligibility_seed),
        }
    }

    fn receipt_ref(
        attempt_seed: u8,
        receipt_seed: u8,
        outcome: ExecutionAttemptOutcomeV1,
    ) -> JournalReceiptRef {
        JournalReceiptRef {
            attempt_id: attempt(attempt_seed),
            receipt_id: receipt(receipt_seed),
            outcome,
        }
    }

    #[test]
    fn intent_without_receipt_is_spent_and_requires_reconciliation() {
        let state = reconstruct_from_refs(&[intent_ref(1, 9)], &[]).unwrap();
        assert!(state.eligibility_is_spent(eligibility(9)));
        assert_eq!(state.pending_reconciliation_attempts(), vec![attempt(1)]);
        assert_eq!(
            state.entry(attempt(1)).unwrap().disposition(),
            JournalAttemptDispositionV1::AwaitingReconciliation
        );
    }

    #[test]
    fn receipt_completes_exact_attempt() {
        let state = reconstruct_from_refs(
            &[intent_ref(1, 9)],
            &[receipt_ref(1, 2, ExecutionAttemptOutcomeV1::Succeeded)],
        )
        .unwrap();
        assert!(state.pending_reconciliation_attempts().is_empty());
        assert_eq!(
            state.entry(attempt(1)).unwrap().disposition(),
            JournalAttemptDispositionV1::Completed(ExecutionAttemptOutcomeV1::Succeeded)
        );
    }

    #[test]
    fn same_eligibility_in_second_attempt_is_replay_conflict() {
        assert_eq!(
            reconstruct_from_refs(&[intent_ref(1, 9), intent_ref(2, 9)], &[]).unwrap_err(),
            ExecutionJournalError::EligibilityReplayConflict {
                eligibility_id: eligibility(9),
                first_attempt: attempt(1),
                second_attempt: attempt(2),
            }
        );
    }

    #[test]
    fn receipt_without_durable_intent_fails_closed() {
        assert_eq!(
            reconstruct_from_refs(
                &[],
                &[receipt_ref(1, 2, ExecutionAttemptOutcomeV1::Succeeded)]
            )
            .unwrap_err(),
            ExecutionJournalError::ReceiptWithoutIntent {
                receipt_id: receipt(2),
                attempt_id: attempt(1),
            }
        );
    }

    #[test]
    fn duplicate_receipts_fail_closed_even_when_results_agree() {
        let result = reconstruct_from_refs(
            &[intent_ref(1, 9)],
            &[
                receipt_ref(1, 2, ExecutionAttemptOutcomeV1::Failed),
                receipt_ref(1, 3, ExecutionAttemptOutcomeV1::Failed),
            ],
        );
        assert_eq!(
            result.unwrap_err(),
            ExecutionJournalError::DuplicateReceipt {
                attempt_id: attempt(1)
            }
        );
    }

    #[test]
    fn journal_digest_commits_pending_vs_completed_world() {
        let pending = reconstruct_from_refs(&[intent_ref(1, 9)], &[]).unwrap();
        let completed = reconstruct_from_refs(
            &[intent_ref(1, 9)],
            &[receipt_ref(1, 2, ExecutionAttemptOutcomeV1::RolledBack)],
        )
        .unwrap();
        assert_ne!(pending.digest(), completed.digest());
    }
}
