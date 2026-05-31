// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Evidence Management Tests
//!
//! Tests for evidence submission, chain of custody, verification,
//! and integrity in the mycelix-justice dispute resolution system.

#[macro_use]
extern crate lazy_static;

#[cfg(test)]
mod evidence_submission_tests {
    use super::*;

    // ==========================================
    // Evidence Submission Tests
    // ==========================================

    #[test]
    fn test_submit_evidence_to_case() {
        let evidence = create_test_evidence(
            "evidence-001",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Document,
            "Contract document showing agreed terms",
        );

        let result = simulate_submit_evidence(evidence);
        assert!(result.is_ok(), "Submitting valid evidence should succeed");
    }

    #[test]
    fn test_submit_evidence_creates_link_to_case() {
        let evidence = create_test_evidence(
            "evidence-001",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Document,
            "Contract document",
        );

        simulate_submit_evidence(evidence).unwrap();

        let case_evidence = simulate_get_case_evidence("case-001");
        assert!(case_evidence.iter().any(|e| e.id == "evidence-001"));
    }

    #[test]
    fn test_submit_evidence_creates_link_to_submitter() {
        let evidence = create_test_evidence(
            "evidence-001",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Document,
            "Contract document",
        );

        simulate_submit_evidence(evidence).unwrap();

        let submitter_evidence = simulate_get_evidence_by_submitter("did:mycelix:alice");
        assert!(submitter_evidence.iter().any(|e| e.id == "evidence-001"));
    }

    #[test]
    fn test_submit_multiple_evidence_items() {
        let evidence1 = create_test_evidence(
            "ev-001",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Document,
            "Doc 1",
        );
        let evidence2 = create_test_evidence(
            "ev-002",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Transaction,
            "Transaction record",
        );
        let evidence3 = create_test_evidence(
            "ev-003",
            "case-001",
            "did:mycelix:bob",
            EvidenceType::Testimony,
            "Witness testimony",
        );

        simulate_submit_evidence(evidence1).unwrap();
        simulate_submit_evidence(evidence2).unwrap();
        simulate_submit_evidence(evidence3).unwrap();

        let case_evidence = simulate_get_case_evidence("case-001");
        assert_eq!(case_evidence.len(), 3);
    }

    #[test]
    fn test_submit_evidence_with_all_types() {
        // Document
        let doc = create_test_evidence(
            "ev-doc",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Document,
            "Document",
        );
        assert!(simulate_submit_evidence(doc).is_ok());

        // Transaction
        let txn = create_test_evidence(
            "ev-txn",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Transaction,
            "Transaction",
        );
        assert!(simulate_submit_evidence(txn).is_ok());

        // Communication
        let comm = create_test_evidence(
            "ev-comm",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Communication,
            "Chat logs",
        );
        assert!(simulate_submit_evidence(comm).is_ok());

        // Testimony
        let testimony = create_test_evidence(
            "ev-test",
            "case-001",
            "did:mycelix:witness",
            EvidenceType::Testimony,
            "Witness statement",
        );
        assert!(simulate_submit_evidence(testimony).is_ok());

        // Expert Opinion
        let expert = create_test_evidence(
            "ev-expert",
            "case-001",
            "did:mycelix:expert",
            EvidenceType::ExpertOpinion,
            "Expert analysis",
        );
        assert!(simulate_submit_evidence(expert).is_ok());

        // Media
        let media = create_test_evidence(
            "ev-media",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Media,
            "Photos",
        );
        assert!(simulate_submit_evidence(media).is_ok());
    }

    #[test]
    fn test_submit_encrypted_evidence() {
        let mut evidence = create_test_evidence(
            "evidence-encrypted",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Document,
            "Confidential document",
        );
        evidence.content.encrypted = true;
        evidence.content.key_reference = Some("key:case-001:evidence-encrypted".to_string());

        let result = simulate_submit_evidence(evidence);
        assert!(result.is_ok());

        let submitted = result.unwrap();
        assert!(submitted.content.encrypted);
        assert!(submitted.content.key_reference.is_some());
    }
}

#[cfg(test)]
mod chain_of_custody_tests {
    use super::*;

    // ==========================================
    // Chain of Custody Tests
    // ==========================================

    #[test]
    fn test_evidence_has_initial_custody_event() {
        let evidence = create_test_evidence(
            "evidence-001",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Document,
            "Contract document",
        );

        let result = simulate_submit_evidence(evidence);
        assert!(result.is_ok());

        let submitted = result.unwrap();
        assert!(
            !submitted.custody.is_empty(),
            "Evidence should have initial custody event"
        );
        assert_eq!(submitted.custody[0].action, CustodyAction::Submitted);
        assert_eq!(submitted.custody[0].actor, "did:mycelix:alice");
    }

    #[test]
    fn test_record_evidence_access() {
        let evidence = create_test_evidence(
            "evidence-001",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Document,
            "Doc",
        );
        simulate_submit_evidence(evidence).unwrap();

        let result = simulate_record_custody_event(
            "evidence-001",
            CustodyAction::Accessed,
            "did:mycelix:arbitrator1",
            Some("Accessed for review".to_string()),
        );

        assert!(result.is_ok());
        let updated = result.unwrap();
        assert!(
            updated.custody.iter().any(
                |c| c.action == CustodyAction::Accessed && c.actor == "did:mycelix:arbitrator1"
            )
        );
    }

    #[test]
    fn test_record_evidence_copy() {
        let evidence = create_test_evidence(
            "evidence-001",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Document,
            "Doc",
        );
        simulate_submit_evidence(evidence).unwrap();

        let result = simulate_record_custody_event(
            "evidence-001",
            CustodyAction::Copied,
            "did:mycelix:panel_system",
            Some("Copied for panel distribution".to_string()),
        );

        assert!(result.is_ok());
        let updated = result.unwrap();
        assert!(
            updated
                .custody
                .iter()
                .any(|c| c.action == CustodyAction::Copied)
        );
    }

    #[test]
    fn test_record_evidence_verified() {
        let evidence = create_test_evidence(
            "evidence-001",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Document,
            "Doc",
        );
        simulate_submit_evidence(evidence).unwrap();

        let result = simulate_record_custody_event(
            "evidence-001",
            CustodyAction::Verified,
            "did:mycelix:verifier",
            Some("Hash verified against original".to_string()),
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_record_evidence_challenged() {
        let evidence = create_test_evidence(
            "evidence-001",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Document,
            "Doc",
        );
        simulate_submit_evidence(evidence).unwrap();

        let result = simulate_record_custody_event(
            "evidence-001",
            CustodyAction::Challenged,
            "did:mycelix:bob",
            Some("Authenticity disputed".to_string()),
        );

        assert!(result.is_ok());
        let updated = result.unwrap();
        assert!(
            updated
                .custody
                .iter()
                .any(|c| c.action == CustodyAction::Challenged)
        );
    }

    #[test]
    fn test_seal_evidence() {
        let evidence = create_test_evidence(
            "evidence-001",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Document,
            "Doc",
        );
        simulate_submit_evidence(evidence).unwrap();

        let result = simulate_seal_evidence("evidence-001", "did:mycelix:arbitrator1");
        assert!(result.is_ok());

        let sealed = result.unwrap();
        assert!(sealed.sealed);
        assert!(
            sealed
                .custody
                .iter()
                .any(|c| c.action == CustodyAction::Sealed)
        );
    }

    #[test]
    fn test_sealed_evidence_cannot_be_modified() {
        let evidence = create_test_evidence(
            "evidence-001",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Document,
            "Doc",
        );
        simulate_submit_evidence(evidence).unwrap();
        simulate_seal_evidence("evidence-001", "did:mycelix:arbitrator1").unwrap();

        // Attempt to record another custody event should fail
        let result = simulate_record_custody_event(
            "evidence-001",
            CustodyAction::Accessed,
            "did:mycelix:bob",
            None,
        );

        assert!(
            result.is_err(),
            "Should not be able to modify sealed evidence"
        );
    }

    #[test]
    fn test_custody_chain_integrity() {
        let evidence = create_test_evidence(
            "evidence-001",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Document,
            "Doc",
        );
        simulate_submit_evidence(evidence).unwrap();

        // Multiple custody events
        simulate_record_custody_event(
            "evidence-001",
            CustodyAction::Accessed,
            "did:mycelix:arb1",
            None,
        )
        .unwrap();
        simulate_record_custody_event(
            "evidence-001",
            CustodyAction::Verified,
            "did:mycelix:verifier",
            None,
        )
        .unwrap();
        simulate_record_custody_event(
            "evidence-001",
            CustodyAction::Accessed,
            "did:mycelix:arb2",
            None,
        )
        .unwrap();

        let evidence = simulate_get_evidence("evidence-001").unwrap();

        // Verify chain is complete and ordered
        assert!(evidence.custody.len() >= 4);
        assert_eq!(evidence.custody[0].action, CustodyAction::Submitted); // First is always submission
    }
}

#[cfg(test)]
mod evidence_verification_tests {
    use super::*;

    // ==========================================
    // Evidence Verification Tests
    // ==========================================

    #[test]
    fn test_verify_evidence() {
        let evidence = create_test_evidence(
            "evidence-001",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Document,
            "Doc",
        );
        simulate_submit_evidence(evidence).unwrap();

        let verification = EvidenceVerificationInput {
            evidence_id: "evidence-001".to_string(),
            verifier: "did:mycelix:arbitrator1".to_string(),
            status: VerificationStatus::Verified,
            notes: Some("Document hash matches claimed content".to_string()),
        };

        let result = simulate_verify_evidence(verification);
        assert!(result.is_ok());
    }

    #[test]
    fn test_mark_evidence_pending_verification() {
        let evidence = create_test_evidence(
            "evidence-001",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Document,
            "Doc",
        );
        simulate_submit_evidence(evidence).unwrap();

        let verification = EvidenceVerificationInput {
            evidence_id: "evidence-001".to_string(),
            verifier: "did:mycelix:arbitrator1".to_string(),
            status: VerificationStatus::Pending,
            notes: Some("Awaiting expert review".to_string()),
        };

        let result = simulate_verify_evidence(verification);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reject_evidence_verification() {
        let evidence = create_test_evidence(
            "evidence-001",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Document,
            "Doc",
        );
        simulate_submit_evidence(evidence).unwrap();

        let verification = EvidenceVerificationInput {
            evidence_id: "evidence-001".to_string(),
            verifier: "did:mycelix:arbitrator1".to_string(),
            status: VerificationStatus::Rejected,
            notes: Some("Document appears to be altered".to_string()),
        };

        let result = simulate_verify_evidence(verification);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dispute_evidence() {
        let evidence = create_test_evidence(
            "evidence-001",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Document,
            "Doc",
        );
        simulate_submit_evidence(evidence).unwrap();

        let dispute = EvidenceDisputeInput {
            evidence_id: "evidence-001".to_string(),
            disputant: "did:mycelix:bob".to_string(),
            reason: "The document has been tampered with - metadata shows edits after filing date"
                .to_string(),
        };

        let result = simulate_dispute_evidence(dispute);
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_evidence_verifications() {
        let evidence = create_test_evidence(
            "evidence-001",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Document,
            "Doc",
        );
        simulate_submit_evidence(evidence).unwrap();

        // Multiple verifications from different arbitrators
        simulate_verify_evidence(EvidenceVerificationInput {
            evidence_id: "evidence-001".to_string(),
            verifier: "did:mycelix:arb1".to_string(),
            status: VerificationStatus::Verified,
            notes: None,
        })
        .unwrap();

        simulate_verify_evidence(EvidenceVerificationInput {
            evidence_id: "evidence-001".to_string(),
            verifier: "did:mycelix:arb2".to_string(),
            status: VerificationStatus::Verified,
            notes: None,
        })
        .unwrap();

        let verifications = simulate_get_evidence_verifications("evidence-001");
        assert_eq!(verifications.len(), 2);
    }

    #[test]
    fn test_get_evidence_disputes() {
        let evidence = create_test_evidence(
            "evidence-001",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Document,
            "Doc",
        );
        simulate_submit_evidence(evidence).unwrap();

        simulate_dispute_evidence(EvidenceDisputeInput {
            evidence_id: "evidence-001".to_string(),
            disputant: "did:mycelix:bob".to_string(),
            reason: "Forgery suspected".to_string(),
        })
        .unwrap();

        let disputes = simulate_get_evidence_disputes("evidence-001");
        assert_eq!(disputes.len(), 1);
        assert_eq!(disputes[0].disputant, "did:mycelix:bob");
    }

    #[test]
    fn test_disputed_evidence_updates_verification_status() {
        let evidence = create_test_evidence(
            "evidence-001",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Document,
            "Doc",
        );
        simulate_submit_evidence(evidence).unwrap();

        // First verify
        simulate_verify_evidence(EvidenceVerificationInput {
            evidence_id: "evidence-001".to_string(),
            verifier: "did:mycelix:arb1".to_string(),
            status: VerificationStatus::Verified,
            notes: None,
        })
        .unwrap();

        // Then dispute
        simulate_dispute_evidence(EvidenceDisputeInput {
            evidence_id: "evidence-001".to_string(),
            disputant: "did:mycelix:bob".to_string(),
            reason: "Dispute reason".to_string(),
        })
        .unwrap();

        let updated_evidence = simulate_get_evidence("evidence-001").unwrap();
        assert_eq!(
            updated_evidence.verification.status,
            VerificationStatus::Disputed
        );
    }
}

#[cfg(test)]
mod evidence_visibility_tests {
    use super::*;

    // ==========================================
    // Evidence Visibility/Access Control Tests
    // ==========================================

    #[test]
    fn test_evidence_visibility_all_parties() {
        let mut evidence = create_test_evidence(
            "evidence-001",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Document,
            "Doc",
        );
        evidence.visibility = EvidenceVisibility::AllParties;

        let result = simulate_submit_evidence(evidence);
        assert!(result.is_ok());

        // All parties should be able to access
        assert!(can_access_evidence("evidence-001", "did:mycelix:alice"));
        assert!(can_access_evidence("evidence-001", "did:mycelix:bob"));
        assert!(can_access_evidence(
            "evidence-001",
            "did:mycelix:arbitrator1"
        ));
    }

    #[test]
    fn test_evidence_visibility_adjudicators_only() {
        let mut evidence = create_test_evidence(
            "evidence-001",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Document,
            "Sensitive doc",
        );
        evidence.visibility = EvidenceVisibility::AdjudicatorsOnly;

        let result = simulate_submit_evidence(evidence);
        assert!(result.is_ok());

        // Only arbitrators should access
        assert!(can_access_evidence(
            "evidence-001",
            "did:mycelix:arbitrator1"
        ));
        assert!(!can_access_evidence("evidence-001", "did:mycelix:bob")); // Respondent cannot access
    }

    #[test]
    fn test_evidence_visibility_restricted() {
        let mut evidence = create_test_evidence(
            "evidence-001",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Document,
            "Restricted doc",
        );
        evidence.visibility = EvidenceVisibility::Restricted {
            parties: vec![
                "did:mycelix:alice".to_string(),
                "did:mycelix:arbitrator1".to_string(),
            ],
        };

        let result = simulate_submit_evidence(evidence);
        assert!(result.is_ok());

        // Only specified parties should access
        assert!(can_access_evidence("evidence-001", "did:mycelix:alice"));
        assert!(can_access_evidence(
            "evidence-001",
            "did:mycelix:arbitrator1"
        ));
        assert!(!can_access_evidence("evidence-001", "did:mycelix:bob"));
    }

    #[test]
    fn test_evidence_visibility_sealed() {
        let mut evidence = create_test_evidence(
            "evidence-001",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Document,
            "Sealed doc",
        );
        evidence.visibility = EvidenceVisibility::Sealed;

        let result = simulate_submit_evidence(evidence);
        assert!(result.is_ok());

        // No one should access sealed evidence except by court order
        assert!(!can_access_evidence("evidence-001", "did:mycelix:alice"));
        assert!(!can_access_evidence("evidence-001", "did:mycelix:bob"));
        assert!(!can_access_evidence(
            "evidence-001",
            "did:mycelix:arbitrator1"
        ));
    }
}

#[cfg(test)]
mod evidence_integrity_tests {
    use super::*;

    // ==========================================
    // Evidence Integrity Tests
    // ==========================================

    #[test]
    fn test_evidence_content_hash_immutable() {
        let evidence = create_test_evidence(
            "evidence-001",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Document,
            "Doc",
        );
        let original_hash = evidence.content.hash.clone();

        simulate_submit_evidence(evidence).unwrap();

        // Attempt to change content hash should fail
        let result = simulate_update_evidence_hash("evidence-001", "different_hash_xyz");
        assert!(result.is_err(), "Content hash should be immutable");

        let current = simulate_get_evidence("evidence-001").unwrap();
        assert_eq!(current.content.hash, original_hash);
    }

    #[test]
    fn test_evidence_requires_valid_hash_format() {
        let mut evidence = create_test_evidence(
            "evidence-001",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Document,
            "Doc",
        );
        evidence.content.hash = "".to_string(); // Empty hash

        let result = simulate_submit_evidence(evidence);
        assert!(
            result.is_err(),
            "Evidence with empty hash should be rejected"
        );
    }

    #[test]
    fn test_verify_evidence_hash_matches_content() {
        let evidence = create_test_evidence(
            "evidence-001",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Document,
            "Doc",
        );
        simulate_submit_evidence(evidence).unwrap();

        // Verify the hash integrity
        let result = verify_evidence_integrity(
            "evidence-001",
            "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG",
        );
        assert!(result.is_ok());
        assert!(result.unwrap()); // Hash matches
    }

    #[test]
    fn test_detect_evidence_tampering() {
        let evidence = create_test_evidence(
            "evidence-001",
            "case-001",
            "did:mycelix:alice",
            EvidenceType::Document,
            "Doc",
        );
        simulate_submit_evidence(evidence).unwrap();

        // Verify with wrong hash - should detect tampering
        let result = verify_evidence_integrity("evidence-001", "wrong_hash_abc123");
        assert!(result.is_ok());
        assert!(!result.unwrap()); // Hash does not match - tampering detected
    }
}

// ==========================================
// Test Helper Types and Functions
// ==========================================

#[derive(Clone, Debug, PartialEq)]
enum EvidenceType {
    Document,
    Transaction,
    Communication,
    Testimony,
    ExpertOpinion,
    Media,
    OnChainData { happ: String, entry_hash: String },
    External { source: String },
}

#[derive(Clone, Debug, PartialEq)]
enum VerificationStatus {
    Unverified,
    Pending,
    Verified,
    Disputed,
    Rejected,
}

#[derive(Clone, Debug, PartialEq)]
enum EvidenceVisibility {
    AllParties,
    AdjudicatorsOnly,
    Restricted { parties: Vec<String> },
    Sealed,
}

#[derive(Clone, Debug, PartialEq)]
enum CustodyAction {
    Submitted,
    Accessed,
    Copied,
    Verified,
    Challenged,
    Sealed,
    Released,
}

#[derive(Clone, Debug)]
struct EvidenceContent {
    hash: String,
    reference: String,
    mime_type: String,
    size: u64,
    encrypted: bool,
    key_reference: Option<String>,
}

#[derive(Clone, Debug)]
struct CustodyEvent {
    action: CustodyAction,
    actor: String,
    timestamp: MockTimestamp,
    notes: Option<String>,
}

#[derive(Clone, Debug)]
struct EvidenceVerificationState {
    status: VerificationStatus,
    verifier: Option<String>,
    method: Option<String>,
    verified_at: Option<MockTimestamp>,
    notes: Option<String>,
}

#[derive(Clone, Debug)]
struct TestEvidence {
    id: String,
    case_id: String,
    submitter: String,
    evidence_type: EvidenceType,
    content: EvidenceContent,
    description: String,
    custody: Vec<CustodyEvent>,
    verification: EvidenceVerificationState,
    visibility: EvidenceVisibility,
    sealed: bool,
}

#[derive(Clone, Debug)]
struct EvidenceVerificationInput {
    evidence_id: String,
    verifier: String,
    status: VerificationStatus,
    notes: Option<String>,
}

#[derive(Clone, Debug)]
struct EvidenceDisputeInput {
    evidence_id: String,
    disputant: String,
    reason: String,
}

#[derive(Clone, Debug)]
struct TestEvidenceDispute {
    id: String,
    evidence_id: String,
    disputant: String,
    reason: String,
    resolved: bool,
}

#[derive(Clone, Debug)]
struct TestEvidenceVerification {
    id: String,
    evidence_id: String,
    verifier: String,
    status: VerificationStatus,
    notes: Option<String>,
}

#[derive(Clone, Debug)]
struct MockTimestamp {
    micros: i64,
}

impl MockTimestamp {
    fn now() -> Self {
        MockTimestamp {
            micros: 1706000000000000,
        }
    }
}

// In-memory stores for testing
use std::collections::HashMap;
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref EVIDENCE_STORE: Mutex<HashMap<String, TestEvidence>> = Mutex::new(HashMap::new());
    static ref VERIFICATION_STORE: Mutex<Vec<TestEvidenceVerification>> = Mutex::new(Vec::new());
    static ref DISPUTE_STORE: Mutex<Vec<TestEvidenceDispute>> = Mutex::new(Vec::new());
    static ref ARBITRATORS: Mutex<Vec<String>> = Mutex::new(vec![
        "did:mycelix:arbitrator1".to_string(),
        "did:mycelix:arbitrator2".to_string(),
        "did:mycelix:arbitrator3".to_string(),
    ]);
}

fn create_test_evidence(
    id: &str,
    case_id: &str,
    submitter: &str,
    evidence_type: EvidenceType,
    description: &str,
) -> TestEvidence {
    TestEvidence {
        id: id.to_string(),
        case_id: case_id.to_string(),
        submitter: submitter.to_string(),
        evidence_type,
        content: EvidenceContent {
            hash: "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG".to_string(),
            reference: format!("ipfs://Qm{}", id),
            mime_type: "application/pdf".to_string(),
            size: 1024,
            encrypted: false,
            key_reference: None,
        },
        description: description.to_string(),
        custody: vec![],
        verification: EvidenceVerificationState {
            status: VerificationStatus::Unverified,
            verifier: None,
            method: None,
            verified_at: None,
            notes: None,
        },
        visibility: EvidenceVisibility::AllParties,
        sealed: false,
    }
}

fn simulate_submit_evidence(mut evidence: TestEvidence) -> Result<TestEvidence, String> {
    if evidence.content.hash.is_empty() {
        return Err("Evidence content hash required".to_string());
    }
    if !evidence.submitter.starts_with("did:") {
        return Err("Submitter must be a DID".to_string());
    }

    // Add initial custody event
    evidence.custody.push(CustodyEvent {
        action: CustodyAction::Submitted,
        actor: evidence.submitter.clone(),
        timestamp: MockTimestamp::now(),
        notes: Some("Initial submission".to_string()),
    });

    EVIDENCE_STORE
        .lock()
        .unwrap()
        .insert(evidence.id.clone(), evidence.clone());
    Ok(evidence)
}

fn simulate_get_evidence(id: &str) -> Option<TestEvidence> {
    EVIDENCE_STORE.lock().unwrap().get(id).cloned()
}

fn simulate_get_case_evidence(case_id: &str) -> Vec<TestEvidence> {
    EVIDENCE_STORE
        .lock()
        .unwrap()
        .values()
        .filter(|e| e.case_id == case_id)
        .cloned()
        .collect()
}

fn simulate_get_evidence_by_submitter(submitter: &str) -> Vec<TestEvidence> {
    EVIDENCE_STORE
        .lock()
        .unwrap()
        .values()
        .filter(|e| e.submitter == submitter)
        .cloned()
        .collect()
}

fn simulate_record_custody_event(
    evidence_id: &str,
    action: CustodyAction,
    actor: &str,
    notes: Option<String>,
) -> Result<TestEvidence, String> {
    let mut store = EVIDENCE_STORE.lock().unwrap();
    if let Some(evidence) = store.get_mut(evidence_id) {
        if evidence.sealed {
            return Err("Cannot modify sealed evidence".to_string());
        }
        evidence.custody.push(CustodyEvent {
            action,
            actor: actor.to_string(),
            timestamp: MockTimestamp::now(),
            notes,
        });
        Ok(evidence.clone())
    } else {
        Err("Evidence not found".to_string())
    }
}

fn simulate_seal_evidence(evidence_id: &str, sealer: &str) -> Result<TestEvidence, String> {
    let mut store = EVIDENCE_STORE.lock().unwrap();
    if let Some(evidence) = store.get_mut(evidence_id) {
        evidence.sealed = true;
        evidence.custody.push(CustodyEvent {
            action: CustodyAction::Sealed,
            actor: sealer.to_string(),
            timestamp: MockTimestamp::now(),
            notes: Some("Evidence sealed by arbitrator".to_string()),
        });
        Ok(evidence.clone())
    } else {
        Err("Evidence not found".to_string())
    }
}

fn simulate_verify_evidence(
    input: EvidenceVerificationInput,
) -> Result<TestEvidenceVerification, String> {
    let verification = TestEvidenceVerification {
        id: format!(
            "ver-{}-{}",
            input.evidence_id,
            VERIFICATION_STORE.lock().unwrap().len()
        ),
        evidence_id: input.evidence_id.clone(),
        verifier: input.verifier,
        status: input.status,
        notes: input.notes,
    };
    VERIFICATION_STORE
        .lock()
        .unwrap()
        .push(verification.clone());
    Ok(verification)
}

fn simulate_dispute_evidence(input: EvidenceDisputeInput) -> Result<TestEvidenceDispute, String> {
    let dispute = TestEvidenceDispute {
        id: format!(
            "disp-{}-{}",
            input.evidence_id,
            DISPUTE_STORE.lock().unwrap().len()
        ),
        evidence_id: input.evidence_id.clone(),
        disputant: input.disputant,
        reason: input.reason,
        resolved: false,
    };
    DISPUTE_STORE.lock().unwrap().push(dispute.clone());

    // Update evidence verification status to Disputed
    let mut store = EVIDENCE_STORE.lock().unwrap();
    if let Some(evidence) = store.get_mut(&input.evidence_id) {
        evidence.verification.status = VerificationStatus::Disputed;
    }

    Ok(dispute)
}

fn simulate_get_evidence_verifications(evidence_id: &str) -> Vec<TestEvidenceVerification> {
    VERIFICATION_STORE
        .lock()
        .unwrap()
        .iter()
        .filter(|v| v.evidence_id == evidence_id)
        .cloned()
        .collect()
}

fn simulate_get_evidence_disputes(evidence_id: &str) -> Vec<TestEvidenceDispute> {
    DISPUTE_STORE
        .lock()
        .unwrap()
        .iter()
        .filter(|d| d.evidence_id == evidence_id)
        .cloned()
        .collect()
}

fn can_access_evidence(evidence_id: &str, accessor: &str) -> bool {
    let store = EVIDENCE_STORE.lock().unwrap();
    let arbitrators = ARBITRATORS.lock().unwrap();

    if let Some(evidence) = store.get(evidence_id) {
        match &evidence.visibility {
            EvidenceVisibility::AllParties => true,
            EvidenceVisibility::AdjudicatorsOnly => arbitrators.contains(&accessor.to_string()),
            EvidenceVisibility::Restricted { parties } => parties.contains(&accessor.to_string()),
            EvidenceVisibility::Sealed => false,
        }
    } else {
        false
    }
}

fn simulate_update_evidence_hash(_evidence_id: &str, _new_hash: &str) -> Result<(), String> {
    // Content hash is immutable - always fail
    Err("Evidence content hash is immutable".to_string())
}

fn verify_evidence_integrity(evidence_id: &str, expected_hash: &str) -> Result<bool, String> {
    let store = EVIDENCE_STORE.lock().unwrap();
    if let Some(evidence) = store.get(evidence_id) {
        Ok(evidence.content.hash == expected_hash)
    } else {
        Err("Evidence not found".to_string())
    }
}
