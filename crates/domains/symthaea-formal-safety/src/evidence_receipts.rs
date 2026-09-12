// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reproducible evidence receipts for strict safety-case readiness.
//!
//! `ProofObligation::id` remains a runtime/workflow identity. Deployment-grade
//! evidence instead binds to a deterministic obligation key derived from the
//! controlled claim and expected evidence kind. This lets independent safety-case
//! instances refer to the same reviewed obligation without depending on random UUIDs.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{EvidenceKind, ObligationStatus, ProofObligation, SafetyCase};

const OBLIGATION_KEY_SCHEMA: &[u8] = b"symthaea-proof-obligation-v1\0";
const CONTRACT_DIGEST_SCHEMA: &[u8] = b"symthaea-safety-contract-v1\0";

impl ProofObligation {
    /// Deterministic content key for evidence binding.
    ///
    /// Runtime UUIDs are deliberately excluded. The key changes when either the
    /// controlled claim text or required evidence kind changes.
    pub fn stable_key(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(OBLIGATION_KEY_SCHEMA);
        hasher.update(self.claim.trim().as_bytes());
        hasher.update(b"\0");
        hasher.update(format!("{:?}", self.expected_evidence).as_bytes());
        format!("blake3:{}", hasher.finalize().to_hex())
    }
}

impl SafetyCase {
    /// Deterministic digest of the reviewed safety contract for this subject.
    ///
    /// Runtime case/obligation UUIDs and current discharge state are excluded.
    /// This is a digest of *what must be shown*, not whether it has been shown.
    pub fn contract_digest(&self) -> String {
        let mut keys = self
            .obligations
            .iter()
            .map(ProofObligation::stable_key)
            .collect::<Vec<_>>();
        keys.sort();

        let mut hasher = blake3::Hasher::new();
        hasher.update(CONTRACT_DIGEST_SCHEMA);
        hasher.update(self.subject.trim().as_bytes());
        hasher.update(b"\0");
        for key in keys {
            hasher.update(key.as_bytes());
            hasher.update(b"\0");
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    /// True only when strict receipt-based readiness succeeds.
    pub fn is_strictly_ready(&self, receipts: &[SafetyEvidenceReceipt]) -> bool {
        assess_strict_safety_case(self, receipts).status == StrictSafetyCaseStatus::Ready
    }
}

/// Evidence that has been independently verified for one stable obligation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetyEvidenceReceipt {
    /// Unique receipt identity inside the evidence lineage.
    pub receipt_id: String,
    /// `ProofObligation::stable_key()` of the reviewed obligation.
    pub obligation_key: String,
    /// Actual evidence kind supplied by the verifier.
    pub evidence_kind: EvidenceKind,
    /// Durable reference to the evidence object/run/report/proof.
    pub evidence_ref: String,
    /// Content digest of the referenced evidence object.
    pub evidence_digest: String,
    /// Identity/reference for the verifier or verification process.
    pub verifier_ref: String,
    /// Verification time supplied by the caller's trusted time domain.
    pub verified_at_ms: u64,
}

impl SafetyEvidenceReceipt {
    pub fn validate(&self) -> bool {
        !self.receipt_id.trim().is_empty()
            && !self.obligation_key.trim().is_empty()
            && !self.evidence_ref.trim().is_empty()
            && !self.evidence_digest.trim().is_empty()
            && !self.verifier_ref.trim().is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrictSafetyCaseStatus {
    /// The case structure/receipt set is malformed or internally inconsistent.
    Invalid,
    /// Structurally valid but at least one required obligation lacks acceptable evidence/workflow discharge.
    Blocked,
    /// Every obligation is discharged and has at least one matching verified receipt.
    Ready,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrictSafetyCaseIssue {
    EmptySubject,
    NoObligations,
    EmptyClaim { obligation_key: String },
    DuplicateObligationKey(String),
    InvalidReceipt(String),
    DuplicateReceiptId(String),
    UnknownObligationKey(String),
    EvidenceKindMismatch {
        receipt_id: String,
        expected: EvidenceKind,
        observed: EvidenceKind,
    },
    MissingEvidenceReceipt(String),
    ObligationNotDischarged {
        obligation_key: String,
        status: ObligationStatus,
    },
    ObligationFailed(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StrictSafetyCaseReport {
    pub contract_digest: String,
    pub status: StrictSafetyCaseStatus,
    pub obligation_count: usize,
    pub valid_receipt_count: usize,
    pub issues: Vec<StrictSafetyCaseIssue>,
}

impl StrictSafetyCaseReport {
    /// Readiness evidence is descriptive only and cannot authorize physical action.
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

/// Assess strict, reproducible readiness without mutating workflow state.
///
/// A legacy `evidence_ref` string stored on an obligation is not enough for this
/// assessment. Strict readiness requires a separate receipt binding the stable
/// obligation key to the actual evidence kind, content digest, and verifier.
pub fn assess_strict_safety_case(
    safety_case: &SafetyCase,
    receipts: &[SafetyEvidenceReceipt],
) -> StrictSafetyCaseReport {
    let mut issues = Vec::new();
    let mut obligations = BTreeMap::<String, &ProofObligation>::new();

    if safety_case.subject.trim().is_empty() {
        issues.push(StrictSafetyCaseIssue::EmptySubject);
    }
    if safety_case.obligations.is_empty() {
        issues.push(StrictSafetyCaseIssue::NoObligations);
    }

    for obligation in &safety_case.obligations {
        let key = obligation.stable_key();
        if obligation.claim.trim().is_empty() {
            issues.push(StrictSafetyCaseIssue::EmptyClaim {
                obligation_key: key.clone(),
            });
        }
        if obligations.insert(key.clone(), obligation).is_some() {
            issues.push(StrictSafetyCaseIssue::DuplicateObligationKey(key));
        }
    }

    let mut receipt_ids = BTreeSet::new();
    let mut receipt_counts = BTreeMap::<String, usize>::new();
    let mut valid_receipt_count = 0usize;

    for receipt in receipts {
        if !receipt.validate() {
            issues.push(StrictSafetyCaseIssue::InvalidReceipt(
                receipt.receipt_id.clone(),
            ));
            continue;
        }
        if !receipt_ids.insert(receipt.receipt_id.clone()) {
            issues.push(StrictSafetyCaseIssue::DuplicateReceiptId(
                receipt.receipt_id.clone(),
            ));
            continue;
        }
        let Some(obligation) = obligations.get(&receipt.obligation_key) else {
            issues.push(StrictSafetyCaseIssue::UnknownObligationKey(
                receipt.obligation_key.clone(),
            ));
            continue;
        };
        if receipt.evidence_kind != obligation.expected_evidence {
            issues.push(StrictSafetyCaseIssue::EvidenceKindMismatch {
                receipt_id: receipt.receipt_id.clone(),
                expected: obligation.expected_evidence,
                observed: receipt.evidence_kind,
            });
            continue;
        }
        *receipt_counts
            .entry(receipt.obligation_key.clone())
            .or_default() += 1;
        valid_receipt_count = valid_receipt_count.saturating_add(1);
    }

    for (key, obligation) in &obligations {
        if receipt_counts.get(key).copied().unwrap_or(0) == 0 {
            issues.push(StrictSafetyCaseIssue::MissingEvidenceReceipt(key.clone()));
        }
        match obligation.status {
            ObligationStatus::Discharged => {}
            ObligationStatus::Failed => {
                issues.push(StrictSafetyCaseIssue::ObligationFailed(key.clone()));
            }
            status => issues.push(StrictSafetyCaseIssue::ObligationNotDischarged {
                obligation_key: key.clone(),
                status,
            }),
        }
    }

    let invalid = issues.iter().any(|issue| {
        matches!(
            issue,
            StrictSafetyCaseIssue::EmptySubject
                | StrictSafetyCaseIssue::NoObligations
                | StrictSafetyCaseIssue::EmptyClaim { .. }
                | StrictSafetyCaseIssue::DuplicateObligationKey(_)
                | StrictSafetyCaseIssue::InvalidReceipt(_)
                | StrictSafetyCaseIssue::DuplicateReceiptId(_)
                | StrictSafetyCaseIssue::UnknownObligationKey(_)
                | StrictSafetyCaseIssue::EvidenceKindMismatch { .. }
        )
    });
    let status = if invalid {
        StrictSafetyCaseStatus::Invalid
    } else if issues.is_empty() {
        StrictSafetyCaseStatus::Ready
    } else {
        StrictSafetyCaseStatus::Blocked
    };

    StrictSafetyCaseReport {
        contract_digest: safety_case.contract_digest(),
        status,
        obligation_count: safety_case.obligations.len(),
        valid_receipt_count,
        issues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EvidenceKind, ProofObligation, SafetyCaseTemplate};

    fn receipt_for(obligation: &ProofObligation, id: &str) -> SafetyEvidenceReceipt {
        SafetyEvidenceReceipt {
            receipt_id: id.into(),
            obligation_key: obligation.stable_key(),
            evidence_kind: obligation.expected_evidence,
            evidence_ref: format!("evidence:{id}"),
            evidence_digest: format!("blake3:{id}"),
            verifier_ref: "verifier:test-harness".into(),
            verified_at_ms: 1_000,
        }
    }

    #[test]
    fn obligation_key_ignores_runtime_uuid() {
        let a = ProofObligation::new("same controlled claim", EvidenceKind::Test);
        let b = ProofObligation::new("same controlled claim", EvidenceKind::Test);
        assert_ne!(a.id, b.id);
        assert_eq!(a.stable_key(), b.stable_key());
    }

    #[test]
    fn contract_digest_is_reproducible_across_runtime_instances() {
        let a = SafetyCase::from_template("harbor-node", SafetyCaseTemplate::DomainAwareness);
        let b = SafetyCase::from_template("harbor-node", SafetyCaseTemplate::DomainAwareness);
        assert_ne!(a.id, b.id);
        assert_eq!(a.contract_digest(), b.contract_digest());
    }

    #[test]
    fn different_subject_changes_contract_digest() {
        let a = SafetyCase::from_template("harbor-a", SafetyCaseTemplate::DomainAwareness);
        let b = SafetyCase::from_template("harbor-b", SafetyCaseTemplate::DomainAwareness);
        assert_ne!(a.contract_digest(), b.contract_digest());
    }

    #[test]
    fn legacy_discharge_without_receipt_is_not_strictly_ready() {
        let obligation = ProofObligation::new("claim", EvidenceKind::Test).discharge("legacy:test");
        let mut case = SafetyCase::new("subject");
        case.add_obligation(obligation);
        let report = assess_strict_safety_case(&case, &[]);
        assert_eq!(report.status, StrictSafetyCaseStatus::Blocked);
        assert!(report
            .issues
            .iter()
            .any(|issue| matches!(issue, StrictSafetyCaseIssue::MissingEvidenceReceipt(_))));
    }

    #[test]
    fn matching_receipt_plus_discharge_is_ready() {
        let obligation = ProofObligation::new("claim", EvidenceKind::Test).discharge("legacy:test");
        let receipt = receipt_for(&obligation, "r1");
        let mut case = SafetyCase::new("subject");
        case.add_obligation(obligation);
        let report = assess_strict_safety_case(&case, &[receipt]);
        assert_eq!(report.status, StrictSafetyCaseStatus::Ready);
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn wrong_evidence_kind_is_invalid() {
        let obligation = ProofObligation::new("claim", EvidenceKind::FormalProof).discharge("proof:x");
        let mut receipt = receipt_for(&obligation, "r1");
        receipt.evidence_kind = EvidenceKind::Test;
        let mut case = SafetyCase::new("subject");
        case.add_obligation(obligation);
        let report = assess_strict_safety_case(&case, &[receipt]);
        assert_eq!(report.status, StrictSafetyCaseStatus::Invalid);
    }

    #[test]
    fn duplicate_receipt_id_cannot_double_count_evidence() {
        let obligation = ProofObligation::new("claim", EvidenceKind::Test).discharge("test:x");
        let receipt = receipt_for(&obligation, "r1");
        let mut case = SafetyCase::new("subject");
        case.add_obligation(obligation);
        let report = assess_strict_safety_case(&case, &[receipt.clone(), receipt]);
        assert_eq!(report.status, StrictSafetyCaseStatus::Invalid);
        assert_eq!(report.valid_receipt_count, 1);
    }
}
