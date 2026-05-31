// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! # Mycelix Justice - Sweettest Integration Tests
//!
//! Comprehensive integration tests using Holochain's sweettest framework.
//! Tests cover complaint filing, evidence submission, arbitration,
//! decisions, and appeals across the justice system.
//!
//! ## Running Tests
//!
//! ```bash
//! # Ensure the DNA bundle exists
//! ls mycelix-justice/dna/mycelix_justice.dna
//!
//! # Run tests (requires Holochain conductor via nix develop)
//! cargo test --test sweettest_integration -- --ignored
//! ```

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types (avoids importing zome crates / duplicate symbols)
// ============================================================================

// --- Complaint Mirror Types (from zomes/complaints/integrity) ---

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ComplaintType {
    ContractDispute,
    PropertyDispute,
    CredentialDispute,
    GovernanceDispute,
    FinancialDispute,
    Other(String),
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ComplaintStatus {
    Open,
    UnderReview,
    Filed,
    Mediation,
    Arbitration,
    Appeal,
    Resolved,
    Dismissed,
    Withdrawn,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Complaint {
    pub id: String,
    pub complainant: String,
    pub respondent: String,
    pub title: String,
    pub description: String,
    pub complaint_type: ComplaintType,
    pub status: ComplaintStatus,
    pub evidence_ids: Vec<String>,
    pub relief_sought: String,
    pub created: Timestamp,
    pub updated: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct UpdateComplaintStatusInput {
    pub complaint_id: String,
    pub new_status: ComplaintStatus,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SubmitResponseInput {
    pub complaint_id: String,
    pub respondent: String,
    pub content: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Response {
    pub id: String,
    pub complaint_id: String,
    pub respondent: String,
    pub content: String,
    pub submitted: Timestamp,
}

// --- Evidence Mirror Types (from zomes/evidence/integrity) ---

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum EvidenceType {
    Document,
    Testimony,
    Transaction,
    CrossReference,
    Media,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum VerificationStatus {
    Verified,
    Challenged,
    Rejected,
    Pending,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Evidence {
    pub id: String,
    pub complaint_id: String,
    pub submitter: String,
    pub evidence_type: EvidenceType,
    pub title: String,
    pub description: String,
    pub content_hash: String,
    pub encrypted_content: Option<String>,
    pub submitted: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EvidenceVerification {
    pub id: String,
    pub evidence_id: String,
    pub verifier: String,
    pub status: VerificationStatus,
    pub notes: Option<String>,
    pub verified_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VerifyEvidenceInput {
    pub evidence_id: String,
    pub verifier: String,
    pub status: VerificationStatus,
    pub notes: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DisputeEvidenceInput {
    pub evidence_id: String,
    pub disputant: String,
    pub reason: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EvidenceDispute {
    pub id: String,
    pub evidence_id: String,
    pub disputant: String,
    pub reason: String,
    pub created_at: Timestamp,
    pub resolved: bool,
}

// --- Arbitration Mirror Types (from dna/integrity) ---

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ArbitratorRole {
    Primary,
    PanelMember,
    Alternate,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Arbitrator {
    pub did: String,
    pub role: ArbitratorRole,
    pub selected_at: Timestamp,
    pub accepted: bool,
    pub recused: bool,
    pub recusal_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ArbitratorSelection {
    Random,
    MATLWeighted,
    PartyAgreed,
    ExpertiseBased { domain: String },
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ArbitrationStatus {
    PanelFormation,
    EvidenceReview,
    Hearing,
    Deliberation,
    DecisionDrafting,
    DecisionRendered,
    Appealed,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Arbitration {
    pub id: String,
    pub case_id: String,
    pub arbitrators: Vec<Arbitrator>,
    pub selection_method: ArbitratorSelection,
    pub status: ArbitrationStatus,
    pub deliberation_deadline: Option<Timestamp>,
    pub created_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct UpdateArbStatusInput {
    pub arbitration_hash: ActionHash,
    pub new_status: ArbitrationStatus,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ArbitratorResponseInput {
    pub arbitration_hash: ActionHash,
    pub arbitrator_did: String,
    pub accepted: bool,
    pub recused: bool,
    pub recusal_reason: Option<String>,
}

// --- Decision Mirror Types ---

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DecisionType {
    MeritsDecision,
    InterimDecision,
    DefaultDecision,
    ConsentDecision,
    Dismissal,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DecisionOutcome {
    ForComplainant,
    ForRespondent,
    SplitDecision,
    Dismissed,
    Settled,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum RemedyType {
    Compensation,
    Restitution,
    SpecificPerformance,
    Injunction,
    Apology,
    CommunityService,
    ReputationAdjustment,
    AccessRestriction,
    Education,
    RestorativeCircle,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Remedy {
    pub remedy_type: RemedyType,
    pub responsible_party: String,
    pub deadline: Option<Timestamp>,
    pub amount: Option<u128>,
    pub currency: Option<String>,
    pub description: String,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum VoteChoice {
    ForComplainant,
    ForRespondent,
    Abstain,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ArbitratorVote {
    pub arbitrator: String,
    pub vote: VoteChoice,
    pub timestamp: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DissentingOpinion {
    pub arbitrator: String,
    pub opinion: String,
    pub timestamp: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Decision {
    pub id: String,
    pub case_id: String,
    pub arbitration_id: String,
    pub decision_type: DecisionType,
    pub outcome: DecisionOutcome,
    pub reasoning: String,
    pub remedies: Vec<Remedy>,
    pub votes: Vec<ArbitratorVote>,
    pub dissents: Vec<DissentingOpinion>,
    pub rendered_at: Timestamp,
    pub appeal_deadline: Timestamp,
    pub finalized: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FinalizeDecisionInput {
    pub decision_hash: ActionHash,
}

// --- Appeal Mirror Types ---

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum AppealGround {
    ProceduralError,
    NewEvidence,
    LegalError,
    Bias,
    ExcessiveRemedy,
    InsufficientRemedy,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum AppealStatus {
    Filed,
    UnderReview,
    Granted,
    Denied,
    Remanded,
    Resolved,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Appeal {
    pub id: String,
    pub case_id: String,
    pub decision_id: String,
    pub appellant: String,
    pub grounds: Vec<AppealGround>,
    pub argument: String,
    pub status: AppealStatus,
    pub appeal_number: u8,
    pub created_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct UpdateAppealStatusInput {
    pub appeal_hash: ActionHash,
    pub new_status: AppealStatus,
}

// ============================================================================
// Test Utilities
// ============================================================================

/// Path to the pre-built DNA bundle
fn dna_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("dna")
        .join("mycelix_justice.dna")
}

async fn load_dna() -> DnaFile {
    SweetDnaFile::from_bundle(&dna_path())
        .await
        .expect("Failed to load DNA bundle - run 'hc dna pack dna/' first")
}

/// Decode an entry from a Record into a concrete type via MessagePack deserialization.
fn decode_entry<T: serde::de::DeserializeOwned>(record: &Record) -> Option<T> {
    match record.entry().as_option()? {
        Entry::App(bytes) => {
            let sb = SerializedBytes::from(bytes.to_owned());
            rmp_serde::from_slice(sb.bytes()).ok()
        }
        _ => None,
    }
}

// ============================================================================
// Complaint Tests
// ============================================================================

#[cfg(test)]
mod complaint_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // Requires Holochain conductor
    async fn test_file_complaint() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let now = Timestamp::now();
        let complaint = Complaint {
            id: "complaint-001".to_string(),
            complainant: "did:mycelix:alice".to_string(),
            respondent: "did:mycelix:bob".to_string(),
            title: "Contract breach on marketplace deal".to_string(),
            description: "Bob failed to deliver goods as agreed in listing #42".to_string(),
            complaint_type: ComplaintType::ContractDispute,
            status: ComplaintStatus::Open,
            evidence_ids: vec![],
            relief_sought: "Full refund of 500 credits".to_string(),
            created: now,
            updated: now,
        };

        let record: Record = conductor
            .call(
                &cell.zome("complaints"),
                "file_complaint",
                complaint.clone(),
            )
            .await;

        let filed: Complaint = decode_entry(&record).expect("Failed to decode complaint");

        assert_eq!(filed.id, "complaint-001", "Complaint ID must match");
        assert_eq!(
            filed.complainant, "did:mycelix:alice",
            "Complainant must match"
        );
        assert_eq!(filed.respondent, "did:mycelix:bob", "Respondent must match");
        assert_eq!(filed.status, ComplaintStatus::Open, "Status must be Open");
        assert_eq!(
            filed.complaint_type,
            ComplaintType::ContractDispute,
            "Type must match"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_complaints_by_party() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let now = Timestamp::now();
        let complaint = Complaint {
            id: "complaint-party-001".to_string(),
            complainant: "did:mycelix:charlie".to_string(),
            respondent: "did:mycelix:dave".to_string(),
            title: "Property boundary dispute".to_string(),
            description: "Disagreement about virtual property boundaries".to_string(),
            complaint_type: ComplaintType::PropertyDispute,
            status: ComplaintStatus::Open,
            evidence_ids: vec![],
            relief_sought: "Boundary realignment".to_string(),
            created: now,
            updated: now,
        };

        let _: Record = conductor
            .call(&cell.zome("complaints"), "file_complaint", complaint)
            .await;

        // Query as complainant
        let complaints: Vec<Record> = conductor
            .call(
                &cell.zome("complaints"),
                "get_complaints_by_party",
                "did:mycelix:charlie".to_string(),
            )
            .await;

        assert!(
            !complaints.is_empty(),
            "Should find complaints for complainant"
        );

        // Query as respondent
        let respondent_complaints: Vec<Record> = conductor
            .call(
                &cell.zome("complaints"),
                "get_complaints_by_party",
                "did:mycelix:dave".to_string(),
            )
            .await;

        assert!(
            !respondent_complaints.is_empty(),
            "Should find complaints for respondent"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_update_complaint_status() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let now = Timestamp::now();
        let complaint = Complaint {
            id: "complaint-status-001".to_string(),
            complainant: "did:mycelix:eve".to_string(),
            respondent: "did:mycelix:frank".to_string(),
            title: "Financial dispute".to_string(),
            description: "Discrepancy in payment amounts".to_string(),
            complaint_type: ComplaintType::FinancialDispute,
            status: ComplaintStatus::Open,
            evidence_ids: vec![],
            relief_sought: "Payment correction".to_string(),
            created: now,
            updated: now,
        };

        let _: Record = conductor
            .call(&cell.zome("complaints"), "file_complaint", complaint)
            .await;

        // Update status to UnderReview
        let update_input = UpdateComplaintStatusInput {
            complaint_id: "complaint-status-001".to_string(),
            new_status: ComplaintStatus::UnderReview,
        };

        let updated_record: Record = conductor
            .call(
                &cell.zome("complaints"),
                "update_complaint_status",
                update_input,
            )
            .await;

        let updated: Complaint = decode_entry(&updated_record).expect("Failed to decode");
        assert_eq!(
            updated.status,
            ComplaintStatus::UnderReview,
            "Status should be UnderReview"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_submit_response_to_complaint() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let now = Timestamp::now();
        let complaint = Complaint {
            id: "complaint-resp-001".to_string(),
            complainant: "did:mycelix:grace".to_string(),
            respondent: "did:mycelix:hank".to_string(),
            title: "Governance dispute".to_string(),
            description: "Voting irregularity".to_string(),
            complaint_type: ComplaintType::GovernanceDispute,
            status: ComplaintStatus::Open,
            evidence_ids: vec![],
            relief_sought: "Revote".to_string(),
            created: now,
            updated: now,
        };

        let _: Record = conductor
            .call(&cell.zome("complaints"), "file_complaint", complaint)
            .await;

        // Submit response
        let response_input = SubmitResponseInput {
            complaint_id: "complaint-resp-001".to_string(),
            respondent: "did:mycelix:hank".to_string(),
            content: "I dispute these claims. The voting was conducted properly.".to_string(),
        };

        let response_record: Record = conductor
            .call(&cell.zome("complaints"), "submit_response", response_input)
            .await;

        let response: Response = decode_entry(&response_record).expect("Failed to decode response");
        assert_eq!(
            response.complaint_id, "complaint-resp-001",
            "Complaint ID must match"
        );
        assert_eq!(
            response.respondent, "did:mycelix:hank",
            "Respondent must match"
        );

        // Get complaint responses
        let responses: Vec<Record> = conductor
            .call(
                &cell.zome("complaints"),
                "get_complaint_responses",
                "complaint-resp-001".to_string(),
            )
            .await;

        assert!(!responses.is_empty(), "Should have at least one response");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_open_complaints() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let now = Timestamp::now();

        // File two complaints
        for i in 0..2 {
            let complaint = Complaint {
                id: format!("complaint-open-{}", i),
                complainant: "did:mycelix:ivan".to_string(),
                respondent: "did:mycelix:judy".to_string(),
                title: format!("Open complaint {}", i),
                description: "Test complaint".to_string(),
                complaint_type: ComplaintType::Other("Test".to_string()),
                status: ComplaintStatus::Open,
                evidence_ids: vec![],
                relief_sought: "Resolution".to_string(),
                created: now,
                updated: now,
            };

            let _: Record = conductor
                .call(&cell.zome("complaints"), "file_complaint", complaint)
                .await;
        }

        let open: Vec<Record> = conductor
            .call(&cell.zome("complaints"), "get_open_complaints", ())
            .await;

        assert!(open.len() >= 2, "Should have at least 2 open complaints");
    }
}

// ============================================================================
// Evidence Tests
// ============================================================================

#[cfg(test)]
mod evidence_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_submit_evidence() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let now = Timestamp::now();
        let evidence = Evidence {
            id: "evidence-001".to_string(),
            complaint_id: "complaint-001".to_string(),
            submitter: "did:mycelix:alice".to_string(),
            evidence_type: EvidenceType::Document,
            title: "Transaction receipt".to_string(),
            description: "Receipt showing payment was made on time".to_string(),
            content_hash: "QmHash123abc".to_string(),
            encrypted_content: None,
            submitted: now,
        };

        let record: Record = conductor
            .call(&cell.zome("evidence"), "submit_evidence", evidence.clone())
            .await;

        let submitted: Evidence = decode_entry(&record).expect("Failed to decode evidence");

        assert_eq!(submitted.id, "evidence-001", "Evidence ID must match");
        assert_eq!(
            submitted.complaint_id, "complaint-001",
            "Complaint ID must match"
        );
        assert_eq!(
            submitted.submitter, "did:mycelix:alice",
            "Submitter must match"
        );
        assert_eq!(
            submitted.evidence_type,
            EvidenceType::Document,
            "Type must match"
        );
        assert_eq!(
            submitted.content_hash, "QmHash123abc",
            "Content hash must match"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_complaint_evidence() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let now = Timestamp::now();

        // Submit multiple evidence items for same complaint
        for i in 0..3 {
            let evidence = Evidence {
                id: format!("evidence-query-{}", i),
                complaint_id: "complaint-evidence-query".to_string(),
                submitter: "did:mycelix:alice".to_string(),
                evidence_type: EvidenceType::Document,
                title: format!("Evidence item {}", i),
                description: format!("Test evidence {}", i),
                content_hash: format!("QmHash{}", i),
                encrypted_content: None,
                submitted: now,
            };

            let _: Record = conductor
                .call(&cell.zome("evidence"), "submit_evidence", evidence)
                .await;
        }

        let complaint_evidence: Vec<Record> = conductor
            .call(
                &cell.zome("evidence"),
                "get_complaint_evidence",
                "complaint-evidence-query".to_string(),
            )
            .await;

        assert!(
            complaint_evidence.len() >= 3,
            "Should have at least 3 evidence items"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_verify_evidence() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let now = Timestamp::now();

        // First submit evidence
        let evidence = Evidence {
            id: "evidence-verify-001".to_string(),
            complaint_id: "complaint-verify".to_string(),
            submitter: "did:mycelix:alice".to_string(),
            evidence_type: EvidenceType::Transaction,
            title: "Transaction log".to_string(),
            description: "On-chain transaction record".to_string(),
            content_hash: "QmVerifyHash".to_string(),
            encrypted_content: None,
            submitted: now,
        };

        let _: Record = conductor
            .call(&cell.zome("evidence"), "submit_evidence", evidence)
            .await;

        // Verify the evidence
        let verify_input = VerifyEvidenceInput {
            evidence_id: "evidence-verify-001".to_string(),
            verifier: "did:mycelix:arbitrator-001".to_string(),
            status: VerificationStatus::Verified,
            notes: Some("Transaction confirmed on-chain".to_string()),
        };

        let verification_record: Record = conductor
            .call(&cell.zome("evidence"), "verify_evidence", verify_input)
            .await;

        let verification: EvidenceVerification =
            decode_entry(&verification_record).expect("Failed to decode verification");

        assert_eq!(
            verification.evidence_id, "evidence-verify-001",
            "Evidence ID must match"
        );
        assert_eq!(
            verification.status,
            VerificationStatus::Verified,
            "Status should be Verified"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_dispute_evidence() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let now = Timestamp::now();

        // Submit evidence
        let evidence = Evidence {
            id: "evidence-dispute-001".to_string(),
            complaint_id: "complaint-dispute".to_string(),
            submitter: "did:mycelix:alice".to_string(),
            evidence_type: EvidenceType::Testimony,
            title: "Witness statement".to_string(),
            description: "Statement from witness".to_string(),
            content_hash: "QmDisputeHash".to_string(),
            encrypted_content: None,
            submitted: now,
        };

        let _: Record = conductor
            .call(&cell.zome("evidence"), "submit_evidence", evidence)
            .await;

        // Dispute the evidence
        let dispute_input = DisputeEvidenceInput {
            evidence_id: "evidence-dispute-001".to_string(),
            disputant: "did:mycelix:bob".to_string(),
            reason: "The testimony contains factual inaccuracies".to_string(),
        };

        let dispute_record: Record = conductor
            .call(&cell.zome("evidence"), "dispute_evidence", dispute_input)
            .await;

        let dispute: EvidenceDispute =
            decode_entry(&dispute_record).expect("Failed to decode dispute");

        assert_eq!(
            dispute.evidence_id, "evidence-dispute-001",
            "Evidence ID must match"
        );
        assert_eq!(dispute.disputant, "did:mycelix:bob", "Disputant must match");
        assert!(!dispute.resolved, "Dispute should not be resolved yet");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_evidence_by_submitter() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let now = Timestamp::now();

        // Submit evidence from specific submitter
        let evidence = Evidence {
            id: "evidence-submitter-001".to_string(),
            complaint_id: "complaint-sub".to_string(),
            submitter: "did:mycelix:carol".to_string(),
            evidence_type: EvidenceType::Media,
            title: "Screenshot evidence".to_string(),
            description: "Screenshot of the issue".to_string(),
            content_hash: "QmSubmitterHash".to_string(),
            encrypted_content: None,
            submitted: now,
        };

        let _: Record = conductor
            .call(&cell.zome("evidence"), "submit_evidence", evidence)
            .await;

        let by_submitter: Vec<Record> = conductor
            .call(
                &cell.zome("evidence"),
                "get_evidence_by_submitter",
                "did:mycelix:carol".to_string(),
            )
            .await;

        assert!(
            !by_submitter.is_empty(),
            "Should find evidence by submitter"
        );
    }
}

// ============================================================================
// Arbitration Tests
// ============================================================================

#[cfg(test)]
mod arbitration_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_create_arbitration_panel() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let now = Timestamp::now();

        let arbitration = Arbitration {
            id: "arb-001".to_string(),
            case_id: "case-001".to_string(),
            arbitrators: vec![
                Arbitrator {
                    did: "did:mycelix:arb1".to_string(),
                    role: ArbitratorRole::Primary,
                    selected_at: now,
                    accepted: false,
                    recused: false,
                    recusal_reason: None,
                },
                Arbitrator {
                    did: "did:mycelix:arb2".to_string(),
                    role: ArbitratorRole::PanelMember,
                    selected_at: now,
                    accepted: false,
                    recused: false,
                    recusal_reason: None,
                },
                Arbitrator {
                    did: "did:mycelix:arb3".to_string(),
                    role: ArbitratorRole::PanelMember,
                    selected_at: now,
                    accepted: false,
                    recused: false,
                    recusal_reason: None,
                },
            ],
            selection_method: ArbitratorSelection::Random,
            status: ArbitrationStatus::PanelFormation,
            deliberation_deadline: None,
            created_at: now,
        };

        let record: Record = conductor
            .call(&cell.zome("arbitration"), "create_arbitration", arbitration)
            .await;

        let created: Arbitration = decode_entry(&record).expect("Failed to decode arbitration");

        assert_eq!(created.id, "arb-001", "Arbitration ID must match");
        assert_eq!(created.case_id, "case-001", "Case ID must match");
        assert_eq!(created.arbitrators.len(), 3, "Must have 3 arbitrators");
        assert_eq!(
            created.status,
            ArbitrationStatus::PanelFormation,
            "Status must be PanelFormation"
        );
        assert_eq!(
            created.selection_method,
            ArbitratorSelection::Random,
            "Selection method must match"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_update_arbitration_status() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let now = Timestamp::now();

        let arbitration = Arbitration {
            id: "arb-update-001".to_string(),
            case_id: "case-update-001".to_string(),
            arbitrators: vec![
                Arbitrator {
                    did: "did:mycelix:arb-u1".to_string(),
                    role: ArbitratorRole::Primary,
                    selected_at: now,
                    accepted: true,
                    recused: false,
                    recusal_reason: None,
                },
                Arbitrator {
                    did: "did:mycelix:arb-u2".to_string(),
                    role: ArbitratorRole::PanelMember,
                    selected_at: now,
                    accepted: true,
                    recused: false,
                    recusal_reason: None,
                },
                Arbitrator {
                    did: "did:mycelix:arb-u3".to_string(),
                    role: ArbitratorRole::PanelMember,
                    selected_at: now,
                    accepted: true,
                    recused: false,
                    recusal_reason: None,
                },
            ],
            selection_method: ArbitratorSelection::PartyAgreed,
            status: ArbitrationStatus::PanelFormation,
            deliberation_deadline: None,
            created_at: now,
        };

        let record: Record = conductor
            .call(&cell.zome("arbitration"), "create_arbitration", arbitration)
            .await;

        let arb_hash = record.action_address().clone();

        // Update status to EvidenceReview
        let update_input = UpdateArbStatusInput {
            arbitration_hash: arb_hash,
            new_status: ArbitrationStatus::EvidenceReview,
        };

        let updated_record: Record = conductor
            .call(
                &cell.zome("arbitration"),
                "update_arbitration_status",
                update_input,
            )
            .await;

        let updated: Arbitration = decode_entry(&updated_record).expect("Failed to decode");
        assert_eq!(
            updated.status,
            ArbitrationStatus::EvidenceReview,
            "Status should be EvidenceReview"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_render_decision() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let now = Timestamp::now();
        let appeal_deadline =
            Timestamp::from_micros(now.as_micros() as i64 + 30 * 24 * 3600 * 1_000_000);

        let decision = Decision {
            id: "decision-001".to_string(),
            case_id: "case-decision-001".to_string(),
            arbitration_id: "arb-decision-001".to_string(),
            decision_type: DecisionType::MeritsDecision,
            outcome: DecisionOutcome::ForComplainant,
            reasoning: "The evidence clearly shows a breach of contract.".to_string(),
            remedies: vec![Remedy {
                remedy_type: RemedyType::Compensation,
                responsible_party: "did:mycelix:respondent-001".to_string(),
                deadline: None,
                amount: Some(500),
                currency: Some("credits".to_string()),
                description: "Refund of purchase price".to_string(),
            }],
            votes: vec![
                ArbitratorVote {
                    arbitrator: "did:mycelix:arb1".to_string(),
                    vote: VoteChoice::ForComplainant,
                    timestamp: now,
                },
                ArbitratorVote {
                    arbitrator: "did:mycelix:arb2".to_string(),
                    vote: VoteChoice::ForComplainant,
                    timestamp: now,
                },
                ArbitratorVote {
                    arbitrator: "did:mycelix:arb3".to_string(),
                    vote: VoteChoice::ForRespondent,
                    timestamp: now,
                },
            ],
            dissents: vec![DissentingOpinion {
                arbitrator: "did:mycelix:arb3".to_string(),
                opinion: "I believe the evidence was insufficient.".to_string(),
                timestamp: now,
            }],
            rendered_at: now,
            appeal_deadline,
            finalized: false,
        };

        let record: Record = conductor
            .call(&cell.zome("arbitration"), "render_decision", decision)
            .await;

        let rendered: Decision = decode_entry(&record).expect("Failed to decode decision");

        assert_eq!(rendered.id, "decision-001", "Decision ID must match");
        assert_eq!(
            rendered.outcome,
            DecisionOutcome::ForComplainant,
            "Outcome must match"
        );
        assert_eq!(rendered.votes.len(), 3, "Must have 3 votes");
        assert_eq!(rendered.dissents.len(), 1, "Must have 1 dissent");
        assert!(!rendered.finalized, "Decision should not be finalized yet");
        assert_eq!(rendered.remedies.len(), 1, "Must have 1 remedy");
        assert_eq!(
            rendered.remedies[0].remedy_type,
            RemedyType::Compensation,
            "Remedy type must match"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_file_appeal() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let now = Timestamp::now();

        let appeal = Appeal {
            id: "appeal-001".to_string(),
            case_id: "case-appeal-001".to_string(),
            decision_id: "decision-appeal-001".to_string(),
            appellant: "did:mycelix:respondent-appeal".to_string(),
            grounds: vec![AppealGround::ProceduralError, AppealGround::NewEvidence],
            argument: "New evidence has surfaced that was not available during arbitration."
                .to_string(),
            status: AppealStatus::Filed,
            appeal_number: 1,
            created_at: now,
        };

        let record: Record = conductor
            .call(&cell.zome("arbitration"), "file_appeal", appeal)
            .await;

        let filed: Appeal = decode_entry(&record).expect("Failed to decode appeal");

        assert_eq!(filed.id, "appeal-001", "Appeal ID must match");
        assert_eq!(
            filed.appellant, "did:mycelix:respondent-appeal",
            "Appellant must match"
        );
        assert_eq!(filed.status, AppealStatus::Filed, "Status must be Filed");
        assert_eq!(filed.grounds.len(), 2, "Must have 2 grounds");
        assert_eq!(filed.appeal_number, 1, "Must be first appeal");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_case_arbitration() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let now = Timestamp::now();

        let arbitration = Arbitration {
            id: "arb-lookup-001".to_string(),
            case_id: "case-lookup-001".to_string(),
            arbitrators: vec![
                Arbitrator {
                    did: "did:mycelix:arb-l1".to_string(),
                    role: ArbitratorRole::Primary,
                    selected_at: now,
                    accepted: false,
                    recused: false,
                    recusal_reason: None,
                },
                Arbitrator {
                    did: "did:mycelix:arb-l2".to_string(),
                    role: ArbitratorRole::PanelMember,
                    selected_at: now,
                    accepted: false,
                    recused: false,
                    recusal_reason: None,
                },
                Arbitrator {
                    did: "did:mycelix:arb-l3".to_string(),
                    role: ArbitratorRole::PanelMember,
                    selected_at: now,
                    accepted: false,
                    recused: false,
                    recusal_reason: None,
                },
            ],
            selection_method: ArbitratorSelection::MATLWeighted,
            status: ArbitrationStatus::PanelFormation,
            deliberation_deadline: None,
            created_at: now,
        };

        let _: Record = conductor
            .call(&cell.zome("arbitration"), "create_arbitration", arbitration)
            .await;

        // Look up by case ID
        let found: Option<Record> = conductor
            .call(
                &cell.zome("arbitration"),
                "get_case_arbitration",
                "case-lookup-001".to_string(),
            )
            .await;

        assert!(found.is_some(), "Should find arbitration by case ID");

        let arb: Arbitration = decode_entry(&found.unwrap()).expect("Failed to decode");
        assert_eq!(arb.case_id, "case-lookup-001", "Case ID must match");
    }
}

// ============================================================================
// Full Justice Lifecycle Test
// ============================================================================

#[cfg(test)]
mod lifecycle_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_complete_justice_lifecycle() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let now = Timestamp::now();

        // 1. File complaint
        let complaint = Complaint {
            id: "lifecycle-complaint".to_string(),
            complainant: "did:mycelix:lifecycle-alice".to_string(),
            respondent: "did:mycelix:lifecycle-bob".to_string(),
            title: "Lifecycle test dispute".to_string(),
            description: "Full lifecycle test from complaint to resolution".to_string(),
            complaint_type: ComplaintType::ContractDispute,
            status: ComplaintStatus::Open,
            evidence_ids: vec![],
            relief_sought: "Full resolution".to_string(),
            created: now,
            updated: now,
        };

        let complaint_record: Record = conductor
            .call(&cell.zome("complaints"), "file_complaint", complaint)
            .await;

        let filed: Complaint = decode_entry(&complaint_record).expect("Failed to decode");
        assert_eq!(
            filed.status,
            ComplaintStatus::Open,
            "Complaint should be Open"
        );

        // 2. Submit evidence
        let evidence = Evidence {
            id: "lifecycle-evidence".to_string(),
            complaint_id: "lifecycle-complaint".to_string(),
            submitter: "did:mycelix:lifecycle-alice".to_string(),
            evidence_type: EvidenceType::Document,
            title: "Contract document".to_string(),
            description: "Original contract showing terms".to_string(),
            content_hash: "QmLifecycleHash".to_string(),
            encrypted_content: None,
            submitted: now,
        };

        let _: Record = conductor
            .call(&cell.zome("evidence"), "submit_evidence", evidence)
            .await;

        // 3. Submit response
        let response_input = SubmitResponseInput {
            complaint_id: "lifecycle-complaint".to_string(),
            respondent: "did:mycelix:lifecycle-bob".to_string(),
            content: "I acknowledge the contract but dispute the interpretation.".to_string(),
        };

        let _: Record = conductor
            .call(&cell.zome("complaints"), "submit_response", response_input)
            .await;

        // 4. Move to arbitration
        let update_input = UpdateComplaintStatusInput {
            complaint_id: "lifecycle-complaint".to_string(),
            new_status: ComplaintStatus::Arbitration,
        };

        let _: Record = conductor
            .call(
                &cell.zome("complaints"),
                "update_complaint_status",
                update_input,
            )
            .await;

        // 5. Create arbitration panel
        let arbitration = Arbitration {
            id: "lifecycle-arb".to_string(),
            case_id: "lifecycle-complaint".to_string(),
            arbitrators: vec![
                Arbitrator {
                    did: "did:mycelix:lifecycle-arb1".to_string(),
                    role: ArbitratorRole::Primary,
                    selected_at: now,
                    accepted: true,
                    recused: false,
                    recusal_reason: None,
                },
                Arbitrator {
                    did: "did:mycelix:lifecycle-arb2".to_string(),
                    role: ArbitratorRole::PanelMember,
                    selected_at: now,
                    accepted: true,
                    recused: false,
                    recusal_reason: None,
                },
                Arbitrator {
                    did: "did:mycelix:lifecycle-arb3".to_string(),
                    role: ArbitratorRole::PanelMember,
                    selected_at: now,
                    accepted: true,
                    recused: false,
                    recusal_reason: None,
                },
            ],
            selection_method: ArbitratorSelection::Random,
            status: ArbitrationStatus::PanelFormation,
            deliberation_deadline: None,
            created_at: now,
        };

        let arb_record: Record = conductor
            .call(&cell.zome("arbitration"), "create_arbitration", arbitration)
            .await;

        let created_arb: Arbitration = decode_entry(&arb_record).expect("Failed to decode arb");
        assert_eq!(
            created_arb.arbitrators.len(),
            3,
            "Panel must have 3 members"
        );

        // 6. Render decision
        let appeal_deadline =
            Timestamp::from_micros(now.as_micros() as i64 + 30 * 24 * 3600 * 1_000_000);
        let decision = Decision {
            id: "lifecycle-decision".to_string(),
            case_id: "lifecycle-complaint".to_string(),
            arbitration_id: "lifecycle-arb".to_string(),
            decision_type: DecisionType::MeritsDecision,
            outcome: DecisionOutcome::ForComplainant,
            reasoning: "Contract terms clearly support the complainant's position.".to_string(),
            remedies: vec![Remedy {
                remedy_type: RemedyType::Restitution,
                responsible_party: "did:mycelix:lifecycle-bob".to_string(),
                deadline: None,
                amount: Some(1000),
                currency: Some("credits".to_string()),
                description: "Return of goods or equivalent value".to_string(),
            }],
            votes: vec![
                ArbitratorVote {
                    arbitrator: "did:mycelix:lifecycle-arb1".to_string(),
                    vote: VoteChoice::ForComplainant,
                    timestamp: now,
                },
                ArbitratorVote {
                    arbitrator: "did:mycelix:lifecycle-arb2".to_string(),
                    vote: VoteChoice::ForComplainant,
                    timestamp: now,
                },
                ArbitratorVote {
                    arbitrator: "did:mycelix:lifecycle-arb3".to_string(),
                    vote: VoteChoice::ForRespondent,
                    timestamp: now,
                },
            ],
            dissents: vec![],
            rendered_at: now,
            appeal_deadline,
            finalized: false,
        };

        let decision_record: Record = conductor
            .call(&cell.zome("arbitration"), "render_decision", decision)
            .await;

        let rendered: Decision = decode_entry(&decision_record).expect("Failed to decode decision");
        assert_eq!(
            rendered.outcome,
            DecisionOutcome::ForComplainant,
            "Outcome should be for complainant"
        );

        // 7. Resolve complaint
        let resolve_input = UpdateComplaintStatusInput {
            complaint_id: "lifecycle-complaint".to_string(),
            new_status: ComplaintStatus::Resolved,
        };

        let resolved_record: Record = conductor
            .call(
                &cell.zome("complaints"),
                "update_complaint_status",
                resolve_input,
            )
            .await;

        let resolved: Complaint = decode_entry(&resolved_record).expect("Failed to decode");
        assert_eq!(
            resolved.status,
            ComplaintStatus::Resolved,
            "Complaint should be Resolved"
        );
    }
}
