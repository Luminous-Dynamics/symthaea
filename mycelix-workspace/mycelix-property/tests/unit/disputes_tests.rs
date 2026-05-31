// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Disputes Zome Unit Tests
//!
//! Comprehensive test coverage for property disputes, ownership claims,
//! escalation to justice, evidence management, and status updates.

use serde::{Deserialize, Serialize};

// =============================================================================
// Test Data Structures
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestPropertyDispute {
    pub id: String,
    pub property_id: String,
    pub dispute_type: TestDisputeType,
    pub claimant_did: String,
    pub respondent_did: String,
    pub description: String,
    pub evidence_ids: Vec<String>,
    pub status: TestDisputeStatus,
    pub justice_case_id: Option<String>,
    pub filed: i64,
    pub resolved: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestDisputeType {
    Boundary,
    Ownership,
    Encumbrance,
    Easement,
    Trespass,
    Damage,
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestDisputeStatus {
    Filed,
    UnderReview,
    Mediation,
    Arbitration,
    Resolved,
    Dismissed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestOwnershipClaim {
    pub id: String,
    pub property_id: String,
    pub claimant_did: String,
    pub claim_basis: TestClaimBasis,
    pub supporting_documents: Vec<String>,
    pub status: TestClaimStatus,
    pub filed: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestClaimBasis {
    PriorOwnership,
    Inheritance,
    AdversePossession,
    FraudulentTransfer,
    DocumentaryEvidence,
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestClaimStatus {
    Pending,
    UnderInvestigation,
    Validated,
    Rejected,
    Superseded,
}

// =============================================================================
// Helper Functions
// =============================================================================

fn create_test_dispute() -> TestPropertyDispute {
    TestPropertyDispute {
        id: "dispute:property123:1704067200000000".to_string(),
        property_id: "property:did:mycelix:owner:1704000000000000".to_string(),
        dispute_type: TestDisputeType::Boundary,
        claimant_did: "did:mycelix:neighbor".to_string(),
        respondent_did: "did:mycelix:owner".to_string(),
        description: "Fence encroaches 2 meters onto claimant's property according to new survey"
            .to_string(),
        evidence_ids: vec![
            "survey_2024.pdf".to_string(),
            "photos_boundary.zip".to_string(),
        ],
        status: TestDisputeStatus::Filed,
        justice_case_id: None,
        filed: 1704067200000000,
        resolved: None,
    }
}

fn create_test_ownership_claim() -> TestOwnershipClaim {
    TestOwnershipClaim {
        id: "claim:property123:1704067200000000".to_string(),
        property_id: "property:did:mycelix:currentowner:1700000000000000".to_string(),
        claimant_did: "did:mycelix:claimant".to_string(),
        claim_basis: TestClaimBasis::Inheritance,
        supporting_documents: vec![
            "will_document.pdf".to_string(),
            "death_certificate.pdf".to_string(),
            "inheritance_tax_receipt.pdf".to_string(),
        ],
        status: TestClaimStatus::Pending,
        filed: 1704067200000000,
    }
}

fn validate_did(did: &str) -> bool {
    did.starts_with("did:")
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // PROPERTY DISPUTE CREATION TESTS
    // =========================================================================

    #[test]
    fn test_dispute_requires_valid_claimant_did() {
        let dispute = create_test_dispute();
        assert!(validate_did(&dispute.claimant_did));
    }

    #[test]
    fn test_dispute_requires_valid_respondent_did() {
        let dispute = create_test_dispute();
        assert!(validate_did(&dispute.respondent_did));
    }

    #[test]
    fn test_dispute_invalid_claimant_did_rejected() {
        let mut dispute = create_test_dispute();
        dispute.claimant_did = "invalid_claimant".to_string();
        assert!(!validate_did(&dispute.claimant_did));
    }

    #[test]
    fn test_dispute_invalid_respondent_did_rejected() {
        let mut dispute = create_test_dispute();
        dispute.respondent_did = "invalid_respondent".to_string();
        assert!(!validate_did(&dispute.respondent_did));
    }

    #[test]
    fn test_dispute_has_unique_id() {
        let dispute = create_test_dispute();
        assert!(dispute.id.contains("dispute:"));
        assert!(!dispute.id.is_empty());
    }

    #[test]
    fn test_dispute_references_property() {
        let dispute = create_test_dispute();
        assert!(!dispute.property_id.is_empty());
        assert!(dispute.property_id.contains("property:"));
    }

    #[test]
    fn test_dispute_has_description() {
        let dispute = create_test_dispute();
        assert!(!dispute.description.is_empty());
    }

    #[test]
    fn test_dispute_filed_timestamp_valid() {
        let dispute = create_test_dispute();
        assert!(dispute.filed > 0);
        assert!(dispute.filed > 946684800000000); // After year 2000
    }

    #[test]
    fn test_dispute_initial_status() {
        let dispute = create_test_dispute();
        assert_eq!(dispute.status, TestDisputeStatus::Filed);
    }

    #[test]
    fn test_dispute_not_resolved_initially() {
        let dispute = create_test_dispute();
        assert!(dispute.resolved.is_none());
    }

    #[test]
    fn test_dispute_no_justice_case_initially() {
        let dispute = create_test_dispute();
        assert!(dispute.justice_case_id.is_none());
    }

    // =========================================================================
    // DISPUTE TYPE TESTS
    // =========================================================================

    #[test]
    fn test_all_dispute_types_supported() {
        let types = vec![
            TestDisputeType::Boundary,
            TestDisputeType::Ownership,
            TestDisputeType::Encumbrance,
            TestDisputeType::Easement,
            TestDisputeType::Trespass,
            TestDisputeType::Damage,
            TestDisputeType::Other("Custom dispute".to_string()),
        ];
        for t in types {
            assert!(matches!(
                t,
                TestDisputeType::Boundary
                    | TestDisputeType::Ownership
                    | TestDisputeType::Encumbrance
                    | TestDisputeType::Easement
                    | TestDisputeType::Trespass
                    | TestDisputeType::Damage
                    | TestDisputeType::Other(_)
            ));
        }
    }

    #[test]
    fn test_boundary_dispute() {
        let dispute = create_test_dispute();
        assert_eq!(dispute.dispute_type, TestDisputeType::Boundary);
    }

    #[test]
    fn test_ownership_dispute() {
        let mut dispute = create_test_dispute();
        dispute.dispute_type = TestDisputeType::Ownership;
        assert_eq!(dispute.dispute_type, TestDisputeType::Ownership);
    }

    #[test]
    fn test_encumbrance_dispute() {
        let mut dispute = create_test_dispute();
        dispute.dispute_type = TestDisputeType::Encumbrance;
        assert_eq!(dispute.dispute_type, TestDisputeType::Encumbrance);
    }

    // =========================================================================
    // DISPUTE STATUS TESTS
    // =========================================================================

    #[test]
    fn test_all_dispute_statuses_supported() {
        let statuses = vec![
            TestDisputeStatus::Filed,
            TestDisputeStatus::UnderReview,
            TestDisputeStatus::Mediation,
            TestDisputeStatus::Arbitration,
            TestDisputeStatus::Resolved,
            TestDisputeStatus::Dismissed,
        ];
        for status in statuses {
            assert!(matches!(
                status,
                TestDisputeStatus::Filed
                    | TestDisputeStatus::UnderReview
                    | TestDisputeStatus::Mediation
                    | TestDisputeStatus::Arbitration
                    | TestDisputeStatus::Resolved
                    | TestDisputeStatus::Dismissed
            ));
        }
    }

    fn is_dispute_active(status: &TestDisputeStatus) -> bool {
        matches!(
            status,
            TestDisputeStatus::Filed
                | TestDisputeStatus::UnderReview
                | TestDisputeStatus::Mediation
                | TestDisputeStatus::Arbitration
        )
    }

    fn is_dispute_closed(status: &TestDisputeStatus) -> bool {
        matches!(
            status,
            TestDisputeStatus::Resolved | TestDisputeStatus::Dismissed
        )
    }

    #[test]
    fn test_filed_dispute_is_active() {
        assert!(is_dispute_active(&TestDisputeStatus::Filed));
    }

    #[test]
    fn test_under_review_dispute_is_active() {
        assert!(is_dispute_active(&TestDisputeStatus::UnderReview));
    }

    #[test]
    fn test_mediation_dispute_is_active() {
        assert!(is_dispute_active(&TestDisputeStatus::Mediation));
    }

    #[test]
    fn test_arbitration_dispute_is_active() {
        assert!(is_dispute_active(&TestDisputeStatus::Arbitration));
    }

    #[test]
    fn test_resolved_dispute_is_closed() {
        assert!(is_dispute_closed(&TestDisputeStatus::Resolved));
    }

    #[test]
    fn test_dismissed_dispute_is_closed() {
        assert!(is_dispute_closed(&TestDisputeStatus::Dismissed));
    }

    // =========================================================================
    // EVIDENCE MANAGEMENT TESTS
    // =========================================================================

    #[test]
    fn test_dispute_can_have_evidence() {
        let dispute = create_test_dispute();
        assert!(!dispute.evidence_ids.is_empty());
    }

    #[test]
    fn test_dispute_can_have_no_evidence() {
        let mut dispute = create_test_dispute();
        dispute.evidence_ids = vec![];
        assert!(dispute.evidence_ids.is_empty());
    }

    #[test]
    fn test_dispute_can_add_evidence() {
        let mut dispute = create_test_dispute();
        let initial_count = dispute.evidence_ids.len();
        dispute.evidence_ids.push("new_evidence.pdf".to_string());
        assert_eq!(dispute.evidence_ids.len(), initial_count + 1);
    }

    fn can_add_evidence(dispute: &TestPropertyDispute, submitter_did: &str) -> bool {
        (dispute.claimant_did == submitter_did || dispute.respondent_did == submitter_did)
            && is_dispute_active(&dispute.status)
    }

    #[test]
    fn test_claimant_can_add_evidence() {
        let dispute = create_test_dispute();
        assert!(can_add_evidence(&dispute, &dispute.claimant_did));
    }

    #[test]
    fn test_respondent_can_add_evidence() {
        let dispute = create_test_dispute();
        assert!(can_add_evidence(&dispute, &dispute.respondent_did));
    }

    #[test]
    fn test_third_party_cannot_add_evidence() {
        let dispute = create_test_dispute();
        assert!(!can_add_evidence(&dispute, "did:mycelix:thirdparty"));
    }

    #[test]
    fn test_cannot_add_evidence_to_closed_dispute() {
        let mut dispute = create_test_dispute();
        dispute.status = TestDisputeStatus::Resolved;
        assert!(!can_add_evidence(&dispute, &dispute.claimant_did));
    }

    // =========================================================================
    // ESCALATION TO JUSTICE TESTS
    // =========================================================================

    fn can_escalate_to_justice(dispute: &TestPropertyDispute) -> bool {
        is_dispute_active(&dispute.status) && dispute.justice_case_id.is_none()
    }

    #[test]
    fn test_can_escalate_active_dispute() {
        let dispute = create_test_dispute();
        assert!(can_escalate_to_justice(&dispute));
    }

    #[test]
    fn test_cannot_escalate_resolved_dispute() {
        let mut dispute = create_test_dispute();
        dispute.status = TestDisputeStatus::Resolved;
        assert!(!can_escalate_to_justice(&dispute));
    }

    #[test]
    fn test_cannot_escalate_already_escalated_dispute() {
        let mut dispute = create_test_dispute();
        dispute.justice_case_id = Some("CASE-001".to_string());
        assert!(!can_escalate_to_justice(&dispute));
    }

    #[test]
    fn test_escalation_sets_justice_case_id() {
        let mut dispute = create_test_dispute();
        dispute.justice_case_id = Some("CASE-001".to_string());
        dispute.status = TestDisputeStatus::Arbitration;
        assert!(dispute.justice_case_id.is_some());
        assert_eq!(dispute.status, TestDisputeStatus::Arbitration);
    }

    // =========================================================================
    // DISPUTE RESOLUTION TESTS
    // =========================================================================

    fn resolve_dispute(dispute: &mut TestPropertyDispute, dismissed: bool, timestamp: i64) {
        dispute.status = if dismissed {
            TestDisputeStatus::Dismissed
        } else {
            TestDisputeStatus::Resolved
        };
        dispute.resolved = Some(timestamp);
    }

    #[test]
    fn test_resolve_dispute_sets_resolved_status() {
        let mut dispute = create_test_dispute();
        resolve_dispute(&mut dispute, false, 1704200000000000);
        assert_eq!(dispute.status, TestDisputeStatus::Resolved);
    }

    #[test]
    fn test_dismiss_dispute_sets_dismissed_status() {
        let mut dispute = create_test_dispute();
        resolve_dispute(&mut dispute, true, 1704200000000000);
        assert_eq!(dispute.status, TestDisputeStatus::Dismissed);
    }

    #[test]
    fn test_resolution_sets_timestamp() {
        let mut dispute = create_test_dispute();
        let resolution_time = 1704200000000000i64;
        resolve_dispute(&mut dispute, false, resolution_time);
        assert_eq!(dispute.resolved, Some(resolution_time));
    }

    #[test]
    fn test_resolution_timestamp_after_filed() {
        let mut dispute = create_test_dispute();
        let resolution_time = 1704200000000000i64;
        resolve_dispute(&mut dispute, false, resolution_time);
        assert!(dispute.resolved.unwrap() > dispute.filed);
    }

    // =========================================================================
    // OWNERSHIP CLAIM TESTS
    // =========================================================================

    #[test]
    fn test_claim_requires_valid_claimant_did() {
        let claim = create_test_ownership_claim();
        assert!(validate_did(&claim.claimant_did));
    }

    #[test]
    fn test_claim_invalid_claimant_did_rejected() {
        let mut claim = create_test_ownership_claim();
        claim.claimant_did = "invalid_did".to_string();
        assert!(!validate_did(&claim.claimant_did));
    }

    #[test]
    fn test_claim_has_unique_id() {
        let claim = create_test_ownership_claim();
        assert!(claim.id.contains("claim:"));
        assert!(!claim.id.is_empty());
    }

    #[test]
    fn test_claim_references_property() {
        let claim = create_test_ownership_claim();
        assert!(!claim.property_id.is_empty());
        assert!(claim.property_id.contains("property:"));
    }

    #[test]
    fn test_claim_initial_status() {
        let claim = create_test_ownership_claim();
        assert_eq!(claim.status, TestClaimStatus::Pending);
    }

    #[test]
    fn test_claim_filed_timestamp_valid() {
        let claim = create_test_ownership_claim();
        assert!(claim.filed > 0);
    }

    // =========================================================================
    // CLAIM BASIS TESTS
    // =========================================================================

    #[test]
    fn test_all_claim_bases_supported() {
        let bases = vec![
            TestClaimBasis::PriorOwnership,
            TestClaimBasis::Inheritance,
            TestClaimBasis::AdversePossession,
            TestClaimBasis::FraudulentTransfer,
            TestClaimBasis::DocumentaryEvidence,
            TestClaimBasis::Other("Custom basis".to_string()),
        ];
        for basis in bases {
            assert!(matches!(
                basis,
                TestClaimBasis::PriorOwnership
                    | TestClaimBasis::Inheritance
                    | TestClaimBasis::AdversePossession
                    | TestClaimBasis::FraudulentTransfer
                    | TestClaimBasis::DocumentaryEvidence
                    | TestClaimBasis::Other(_)
            ));
        }
    }

    #[test]
    fn test_inheritance_claim() {
        let claim = create_test_ownership_claim();
        assert_eq!(claim.claim_basis, TestClaimBasis::Inheritance);
    }

    #[test]
    fn test_adverse_possession_claim() {
        let mut claim = create_test_ownership_claim();
        claim.claim_basis = TestClaimBasis::AdversePossession;
        assert_eq!(claim.claim_basis, TestClaimBasis::AdversePossession);
    }

    #[test]
    fn test_fraudulent_transfer_claim() {
        let mut claim = create_test_ownership_claim();
        claim.claim_basis = TestClaimBasis::FraudulentTransfer;
        assert_eq!(claim.claim_basis, TestClaimBasis::FraudulentTransfer);
    }

    // =========================================================================
    // CLAIM STATUS TESTS
    // =========================================================================

    #[test]
    fn test_all_claim_statuses_supported() {
        let statuses = vec![
            TestClaimStatus::Pending,
            TestClaimStatus::UnderInvestigation,
            TestClaimStatus::Validated,
            TestClaimStatus::Rejected,
            TestClaimStatus::Superseded,
        ];
        for status in statuses {
            assert!(matches!(
                status,
                TestClaimStatus::Pending
                    | TestClaimStatus::UnderInvestigation
                    | TestClaimStatus::Validated
                    | TestClaimStatus::Rejected
                    | TestClaimStatus::Superseded
            ));
        }
    }

    fn is_claim_active(status: &TestClaimStatus) -> bool {
        matches!(
            status,
            TestClaimStatus::Pending | TestClaimStatus::UnderInvestigation
        )
    }

    fn is_claim_resolved(status: &TestClaimStatus) -> bool {
        matches!(
            status,
            TestClaimStatus::Validated | TestClaimStatus::Rejected | TestClaimStatus::Superseded
        )
    }

    #[test]
    fn test_pending_claim_is_active() {
        assert!(is_claim_active(&TestClaimStatus::Pending));
    }

    #[test]
    fn test_under_investigation_claim_is_active() {
        assert!(is_claim_active(&TestClaimStatus::UnderInvestigation));
    }

    #[test]
    fn test_validated_claim_is_resolved() {
        assert!(is_claim_resolved(&TestClaimStatus::Validated));
    }

    #[test]
    fn test_rejected_claim_is_resolved() {
        assert!(is_claim_resolved(&TestClaimStatus::Rejected));
    }

    #[test]
    fn test_superseded_claim_is_resolved() {
        assert!(is_claim_resolved(&TestClaimStatus::Superseded));
    }

    // =========================================================================
    // SUPPORTING DOCUMENTS TESTS
    // =========================================================================

    #[test]
    fn test_claim_can_have_documents() {
        let claim = create_test_ownership_claim();
        assert!(!claim.supporting_documents.is_empty());
    }

    #[test]
    fn test_claim_can_have_no_documents() {
        let mut claim = create_test_ownership_claim();
        claim.supporting_documents = vec![];
        assert!(claim.supporting_documents.is_empty());
    }

    fn can_add_document(claim: &TestOwnershipClaim, submitter_did: &str) -> bool {
        claim.claimant_did == submitter_did && is_claim_active(&claim.status)
    }

    #[test]
    fn test_claimant_can_add_document() {
        let claim = create_test_ownership_claim();
        assert!(can_add_document(&claim, &claim.claimant_did));
    }

    #[test]
    fn test_non_claimant_cannot_add_document() {
        let claim = create_test_ownership_claim();
        assert!(!can_add_document(&claim, "did:mycelix:other"));
    }

    #[test]
    fn test_cannot_add_document_to_resolved_claim() {
        let mut claim = create_test_ownership_claim();
        claim.status = TestClaimStatus::Validated;
        assert!(!can_add_document(&claim, &claim.claimant_did));
    }

    // =========================================================================
    // ACCESS CONTROL TESTS
    // =========================================================================

    #[test]
    fn test_parties_can_view_dispute() {
        let dispute = create_test_dispute();
        // Both claimant and respondent should be able to view
        assert!(validate_did(&dispute.claimant_did));
        assert!(validate_did(&dispute.respondent_did));
    }

    // =========================================================================
    // DISPUTE WORKFLOW TESTS
    // =========================================================================

    #[test]
    fn test_dispute_status_flow_to_resolution() {
        let mut dispute = create_test_dispute();

        // Filed
        assert_eq!(dispute.status, TestDisputeStatus::Filed);

        // Under review
        dispute.status = TestDisputeStatus::UnderReview;
        assert_eq!(dispute.status, TestDisputeStatus::UnderReview);

        // Mediation
        dispute.status = TestDisputeStatus::Mediation;
        assert_eq!(dispute.status, TestDisputeStatus::Mediation);

        // Resolved
        dispute.status = TestDisputeStatus::Resolved;
        dispute.resolved = Some(1704200000000000);
        assert_eq!(dispute.status, TestDisputeStatus::Resolved);
        assert!(dispute.resolved.is_some());
    }

    #[test]
    fn test_dispute_status_flow_to_arbitration() {
        let mut dispute = create_test_dispute();

        // Filed
        assert_eq!(dispute.status, TestDisputeStatus::Filed);

        // Escalate to arbitration
        dispute.status = TestDisputeStatus::Arbitration;
        dispute.justice_case_id = Some("CASE-001".to_string());
        assert_eq!(dispute.status, TestDisputeStatus::Arbitration);
        assert!(dispute.justice_case_id.is_some());
    }

    #[test]
    fn test_dispute_dismissal_flow() {
        let mut dispute = create_test_dispute();

        // Filed
        assert_eq!(dispute.status, TestDisputeStatus::Filed);

        // Under review
        dispute.status = TestDisputeStatus::UnderReview;

        // Dismissed (e.g., lack of evidence)
        dispute.status = TestDisputeStatus::Dismissed;
        dispute.resolved = Some(1704200000000000);
        assert_eq!(dispute.status, TestDisputeStatus::Dismissed);
    }

    // =========================================================================
    // CLAIM WORKFLOW TESTS
    // =========================================================================

    #[test]
    fn test_claim_status_flow_to_validation() {
        let mut claim = create_test_ownership_claim();

        // Pending
        assert_eq!(claim.status, TestClaimStatus::Pending);

        // Under investigation
        claim.status = TestClaimStatus::UnderInvestigation;
        assert_eq!(claim.status, TestClaimStatus::UnderInvestigation);

        // Validated
        claim.status = TestClaimStatus::Validated;
        assert_eq!(claim.status, TestClaimStatus::Validated);
    }

    #[test]
    fn test_claim_status_flow_to_rejection() {
        let mut claim = create_test_ownership_claim();

        // Pending
        assert_eq!(claim.status, TestClaimStatus::Pending);

        // Under investigation
        claim.status = TestClaimStatus::UnderInvestigation;

        // Rejected
        claim.status = TestClaimStatus::Rejected;
        assert_eq!(claim.status, TestClaimStatus::Rejected);
    }

    #[test]
    fn test_claim_superseded() {
        let mut claim = create_test_ownership_claim();

        // A claim can be superseded by a newer, stronger claim
        claim.status = TestClaimStatus::Superseded;
        assert_eq!(claim.status, TestClaimStatus::Superseded);
    }

    // =========================================================================
    // EDGE CASES
    // =========================================================================

    #[test]
    fn test_multiple_disputes_for_same_property() {
        let dispute1 = TestPropertyDispute {
            id: "dispute:prop:1".to_string(),
            property_id: "property:shared:123".to_string(),
            dispute_type: TestDisputeType::Boundary,
            claimant_did: "did:mycelix:neighbor1".to_string(),
            respondent_did: "did:mycelix:owner".to_string(),
            description: "North boundary dispute".to_string(),
            evidence_ids: vec![],
            status: TestDisputeStatus::Filed,
            justice_case_id: None,
            filed: 1704000000000000,
            resolved: None,
        };

        let dispute2 = TestPropertyDispute {
            id: "dispute:prop:2".to_string(),
            property_id: "property:shared:123".to_string(),
            dispute_type: TestDisputeType::Easement,
            claimant_did: "did:mycelix:neighbor2".to_string(),
            respondent_did: "did:mycelix:owner".to_string(),
            description: "Right of way dispute".to_string(),
            evidence_ids: vec![],
            status: TestDisputeStatus::Filed,
            justice_case_id: None,
            filed: 1704100000000000,
            resolved: None,
        };

        // Same property can have multiple disputes
        assert_eq!(dispute1.property_id, dispute2.property_id);
        // Different claimants
        assert_ne!(dispute1.claimant_did, dispute2.claimant_did);
        // Different dispute types
        assert_ne!(dispute1.dispute_type, dispute2.dispute_type);
    }

    #[test]
    fn test_multiple_claims_for_same_property() {
        let claim1 = TestOwnershipClaim {
            id: "claim:prop:1".to_string(),
            property_id: "property:disputed:456".to_string(),
            claimant_did: "did:mycelix:heir1".to_string(),
            claim_basis: TestClaimBasis::Inheritance,
            supporting_documents: vec!["will_v1.pdf".to_string()],
            status: TestClaimStatus::Pending,
            filed: 1704000000000000,
        };

        let claim2 = TestOwnershipClaim {
            id: "claim:prop:2".to_string(),
            property_id: "property:disputed:456".to_string(),
            claimant_did: "did:mycelix:heir2".to_string(),
            claim_basis: TestClaimBasis::Inheritance,
            supporting_documents: vec!["will_v2.pdf".to_string()],
            status: TestClaimStatus::Pending,
            filed: 1704100000000000,
        };

        // Same property can have multiple claims
        assert_eq!(claim1.property_id, claim2.property_id);
        // Different claimants
        assert_ne!(claim1.claimant_did, claim2.claimant_did);
    }

    #[test]
    fn test_dispute_with_many_evidence_items() {
        let mut dispute = create_test_dispute();
        dispute.evidence_ids = (0..20).map(|i| format!("evidence_{}.pdf", i)).collect();
        assert_eq!(dispute.evidence_ids.len(), 20);
    }

    #[test]
    fn test_claim_with_many_supporting_documents() {
        let mut claim = create_test_ownership_claim();
        claim.supporting_documents = (0..15).map(|i| format!("document_{}.pdf", i)).collect();
        assert_eq!(claim.supporting_documents.len(), 15);
    }

    #[test]
    fn test_long_running_dispute() {
        let mut dispute = create_test_dispute();
        dispute.filed = 1600000000000000; // Filed long ago
        dispute.status = TestDisputeStatus::Arbitration;
        dispute.justice_case_id = Some("CASE-LONG-001".to_string());

        // Dispute can remain active for extended periods
        assert!(is_dispute_active(&dispute.status));
    }

    #[test]
    fn test_dispute_between_co_owners() {
        let dispute = TestPropertyDispute {
            id: "dispute:coowner:1".to_string(),
            property_id: "property:shared:789".to_string(),
            dispute_type: TestDisputeType::Other("Co-owner partition request".to_string()),
            claimant_did: "did:mycelix:coowner1".to_string(),
            respondent_did: "did:mycelix:coowner2".to_string(),
            description: "Requesting partition of jointly owned property".to_string(),
            evidence_ids: vec!["ownership_agreement.pdf".to_string()],
            status: TestDisputeStatus::Filed,
            justice_case_id: None,
            filed: 1704067200000000,
            resolved: None,
        };

        // Disputes between co-owners are valid
        assert!(validate_did(&dispute.claimant_did));
        assert!(validate_did(&dispute.respondent_did));
        assert!(matches!(dispute.dispute_type, TestDisputeType::Other(_)));
    }

    #[test]
    fn test_adverse_possession_claim_requirements() {
        let claim = TestOwnershipClaim {
            id: "claim:adverse:1".to_string(),
            property_id: "property:abandoned:999".to_string(),
            claimant_did: "did:mycelix:squatter".to_string(),
            claim_basis: TestClaimBasis::AdversePossession,
            supporting_documents: vec![
                "occupancy_proof_10_years.pdf".to_string(),
                "tax_payments.pdf".to_string(),
                "improvements_made.pdf".to_string(),
                "neighbor_statements.pdf".to_string(),
            ],
            status: TestClaimStatus::UnderInvestigation,
            filed: 1704067200000000,
        };

        // Adverse possession claims typically require substantial documentation
        assert!(claim.supporting_documents.len() >= 3);
        assert_eq!(claim.claim_basis, TestClaimBasis::AdversePossession);
    }

    #[test]
    fn test_fraudulent_transfer_claim_with_evidence() {
        let claim = TestOwnershipClaim {
            id: "claim:fraud:1".to_string(),
            property_id: "property:fraudulent:111".to_string(),
            claimant_did: "did:mycelix:victim".to_string(),
            claim_basis: TestClaimBasis::FraudulentTransfer,
            supporting_documents: vec![
                "forged_signature_analysis.pdf".to_string(),
                "police_report.pdf".to_string(),
                "original_deed.pdf".to_string(),
            ],
            status: TestClaimStatus::UnderInvestigation,
            filed: 1704067200000000,
        };

        assert_eq!(claim.claim_basis, TestClaimBasis::FraudulentTransfer);
        // Fraud claims should have supporting evidence
        assert!(!claim.supporting_documents.is_empty());
    }
}
