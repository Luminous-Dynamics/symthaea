// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! # Mycelix Finance - Sweettest Integration Tests
//!
//! Comprehensive integration tests using Holochain's sweettest framework.
//! Tests cover CGC operations, credit scoring, lending, hearth/commons pool,
//! and TEND time exchange.
//!
//! ## Running Tests
//!
//! ```bash
//! # Ensure the DNA bundle exists
//! ls mycelix-finance/dna/mycelix_finance.dna
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

// --- CGC Mirror Types ---

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ContributionType {
    Education,
    CommunityOrganizing,
    CareWork,
    CreativeWork,
    TechnicalWork,
    GovernanceLabor,
    Stewardship,
    Gratitude,
    Custom(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GiftCgcInput {
    pub receiver_did: String,
    pub amount: u32,
    pub gratitude_message: String,
    pub contribution_type: ContributionType,
    pub cultural_alias: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AllocationStatus {
    pub member_did: String,
    pub cycle_id: String,
    pub total_allocated: u32,
    pub remaining: u32,
    pub gifted: u32,
    pub expires_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GiftRecord {
    pub id: String,
    pub giver_did: String,
    pub receiver_did: String,
    pub amount: u32,
    pub gratitude_message: String,
    pub contribution_type: ContributionType,
    pub timestamp: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RecognitionSummary {
    pub member_did: String,
    pub cycle_id: String,
    pub total_received: u32,
    pub unique_givers: u32,
    pub by_contribution_type: Vec<(String, u32)>,
    pub is_flagged: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RegisterAliasInput {
    pub alias: String,
    pub dao_did: String,
    pub meaning: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CulturalAlias {
    pub alias: String,
    pub dao_did: String,
    pub meaning: String,
    pub registered_at: Timestamp,
}

// --- Credit Scoring Mirror Types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateProfileInput {
    pub did: String,
    pub matl_score: f64,
    pub collateral_ratio: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreditProfile {
    pub did: String,
    pub matl_score: f64,
    pub payment_history_score: f64,
    pub collateral_ratio: f64,
    pub account_age_days: u64,
    pub activity_score: f64,
    pub computed_score: f64,
    pub last_updated: Timestamp,
    pub version: u32,
}

// --- Lending Mirror Types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RequestLoanInput {
    pub borrower_did: String,
    pub amount: f64,
    pub currency: String,
    pub term_days: u32,
    pub collateral_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum LoanStatus {
    Requested,
    Funded,
    Active,
    Repaid,
    Defaulted,
    Cancelled,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Loan {
    pub id: String,
    pub borrower_did: String,
    pub lender_did: String,
    pub principal: f64,
    pub currency: String,
    pub interest_rate: f64,
    pub term_days: u32,
    pub collateral_ids: Vec<String>,
    pub status: LoanStatus,
    pub created: Timestamp,
    pub funded: Option<Timestamp>,
    pub maturity: Option<Timestamp>,
    pub repaid: Option<Timestamp>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateOfferInput {
    pub lender_did: String,
    pub max_amount: f64,
    pub min_amount: f64,
    pub currency: String,
    pub base_interest_rate: f64,
    pub min_credit_score: f64,
    pub max_term_days: u32,
    pub collateral_required: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FundLoanInput {
    pub loan_id: String,
    pub lender_did: String,
    pub interest_rate: f64,
}

// --- Hearth Mirror Types ---

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ResourceType {
    Money,
    Time,
    Knowledge,
    Tools,
    Space,
    Custom(String),
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum NeedCategory {
    Emergency,
    Housing,
    Healthcare,
    Education,
    Food,
    Childcare,
    ElderCare,
    Infrastructure,
    Custom(String),
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PoolStatus {
    Proposed,
    Active,
    Paused,
    Dissolved,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum RequestStatus {
    Voting,
    Approved,
    Rejected,
    Disbursed,
    Cancelled,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreatePoolInput {
    pub name: String,
    pub description: String,
    pub dao_did: String,
    pub resource_type: ResourceType,
    pub cultural_alias: Option<String>,
    pub governance: Option<PoolGovernanceInput>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PoolGovernanceInput {
    pub allocation_threshold: Option<f32>,
    pub voting_period_hours: Option<u32>,
    pub contribution_weighted_voting: Option<bool>,
    pub min_civ_to_request: Option<f32>,
    pub max_allocation_percent: Option<f32>,
    pub require_justification: Option<bool>,
    pub request_cooldown_hours: Option<u32>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PoolSummary {
    pub id: String,
    pub name: String,
    pub description: String,
    pub dao_did: String,
    pub resource_type: ResourceType,
    pub total_resources: f64,
    pub contributor_count: u32,
    pub status: PoolStatus,
    pub pending_requests: u32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ContributeInput {
    pub pool_id: String,
    pub amount: f64,
    pub message: Option<String>,
    pub anonymous: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PoolContribution {
    pub id: String,
    pub pool_id: String,
    pub contributor_did: String,
    pub amount: f64,
    pub message: Option<String>,
    pub anonymous: bool,
    pub timestamp: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RequestAllocationInput {
    pub pool_id: String,
    pub recipient_did: Option<String>,
    pub amount: f64,
    pub justification: String,
    pub need_category: NeedCategory,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RequestSummary {
    pub id: String,
    pub pool_id: String,
    pub requester_did: String,
    pub recipient_did: String,
    pub amount: f64,
    pub justification: String,
    pub need_category: NeedCategory,
    pub status: RequestStatus,
    pub votes_for: u32,
    pub votes_against: u32,
    pub voting_deadline: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VoteInput {
    pub request_id: String,
    pub vote: bool,
    pub comment: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AllocationVote {
    pub request_id: String,
    pub voter_did: String,
    pub vote: bool,
    pub weight: f64,
    pub comment: Option<String>,
    pub timestamp: Timestamp,
}

// --- TEND Mirror Types ---

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ServiceCategory {
    Education,
    Healthcare,
    Technology,
    Creative,
    Manual,
    Administrative,
    Caregiving,
    Transport,
    Food,
    Custom(String),
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ExchangeStatus {
    Proposed,
    Confirmed,
    Disputed,
    Cancelled,
    Resolved,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RecordExchangeInput {
    pub receiver_did: String,
    pub hours: f32,
    pub service_description: String,
    pub service_category: ServiceCategory,
    pub cultural_alias: Option<String>,
    pub dao_did: String,
    pub service_date: Option<Timestamp>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ExchangeRecord {
    pub id: String,
    pub provider_did: String,
    pub receiver_did: String,
    pub hours: f32,
    pub service_description: String,
    pub service_category: ServiceCategory,
    pub status: ExchangeStatus,
    pub timestamp: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GetBalanceInput {
    pub member_did: String,
    pub dao_did: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BalanceInfo {
    pub member_did: String,
    pub dao_did: String,
    pub balance: i32,
    pub can_provide: bool,
    pub can_receive: bool,
    pub total_provided: f32,
    pub total_received: f32,
    pub exchange_count: u32,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Urgency {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateListingInput {
    pub dao_did: String,
    pub title: String,
    pub description: String,
    pub category: ServiceCategory,
    pub estimated_hours: Option<f32>,
    pub availability: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ServiceListing {
    pub id: String,
    pub provider_did: String,
    pub dao_did: String,
    pub title: String,
    pub description: String,
    pub category: ServiceCategory,
    pub estimated_hours: Option<f32>,
    pub availability: Option<String>,
    pub active: bool,
    pub created: Timestamp,
}

// ============================================================================
// Test Utilities
// ============================================================================

/// Path to the pre-built DNA bundle
fn dna_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("dna")
        .join("mycelix_finance.dna")
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
// CGC (Civic Gifting Credits) Tests
// ============================================================================

#[cfg(test)]
mod cgc_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // Requires Holochain conductor
    async fn test_get_or_create_allocation() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let member_did = format!("did:mycelix:{}", agent);

        let allocation: AllocationStatus = conductor
            .call(
                &cell.zome("cgc"),
                "get_or_create_allocation",
                member_did.clone(),
            )
            .await;

        assert_eq!(allocation.member_did, member_did, "Member DID must match");
        assert_eq!(
            allocation.total_allocated, 10,
            "Monthly allocation must be 10 CGC"
        );
        assert_eq!(
            allocation.remaining, 10,
            "New allocation should have 10 remaining"
        );
        assert_eq!(allocation.gifted, 0, "New allocation should have 0 gifted");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_gift_cgc_and_query() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;

        // Set up two agents
        let app1 = conductor
            .setup_app("giver-app", &[dna.clone()])
            .await
            .unwrap();
        let app2 = conductor.setup_app("receiver-app", &[dna]).await.unwrap();
        let cell1 = app1.cells()[0].clone();
        let agent1 = app1.agent().clone();
        let agent2 = app2.agent().clone();

        let giver_did = format!("did:mycelix:{}", agent1);
        let receiver_did = format!("did:mycelix:{}", agent2);

        // Ensure allocation exists
        let _: AllocationStatus = conductor
            .call(
                &cell1.zome("cgc"),
                "get_or_create_allocation",
                giver_did.clone(),
            )
            .await;

        // Gift CGC
        let gift_input = GiftCgcInput {
            receiver_did: receiver_did.clone(),
            amount: 3,
            gratitude_message: "Thank you for your mentoring!".to_string(),
            contribution_type: ContributionType::Education,
            cultural_alias: None,
        };

        let gift: GiftRecord = conductor
            .call(&cell1.zome("cgc"), "gift_cgc", gift_input)
            .await;

        assert_eq!(gift.giver_did, giver_did, "Giver DID must match");
        assert_eq!(gift.receiver_did, receiver_did, "Receiver DID must match");
        assert_eq!(gift.amount, 3, "Gift amount must be 3");
        assert_eq!(
            gift.contribution_type,
            ContributionType::Education,
            "Type must be Education"
        );

        // Check giver's allocation was decremented
        let updated_allocation: AllocationStatus = conductor
            .call(
                &cell1.zome("cgc"),
                "get_or_create_allocation",
                giver_did.clone(),
            )
            .await;

        assert_eq!(
            updated_allocation.remaining, 7,
            "Should have 7 remaining after gifting 3"
        );
        assert_eq!(updated_allocation.gifted, 3, "Should show 3 gifted");

        // Query gifts given
        let gifts_given: Vec<GiftRecord> = conductor
            .call(&cell1.zome("cgc"), "get_gifts_given", giver_did.clone())
            .await;

        assert!(
            !gifts_given.is_empty(),
            "Should have at least one gift given"
        );
        assert_eq!(
            gifts_given[0].receiver_did, receiver_did,
            "Gift receiver must match"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_recognition_summary() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;

        let app1 = conductor
            .setup_app("giver-app", &[dna.clone()])
            .await
            .unwrap();
        let app2 = conductor.setup_app("receiver-app", &[dna]).await.unwrap();
        let cell1 = app1.cells()[0].clone();
        let agent2 = app2.agent().clone();

        let receiver_did = format!("did:mycelix:{}", agent2);

        // Gift some CGC
        let gift_input = GiftCgcInput {
            receiver_did: receiver_did.clone(),
            amount: 5,
            gratitude_message: "Great technical work!".to_string(),
            contribution_type: ContributionType::TechnicalWork,
            cultural_alias: None,
        };

        let _: GiftRecord = conductor
            .call(&cell1.zome("cgc"), "gift_cgc", gift_input)
            .await;

        // Get recognition summary
        let summary: RecognitionSummary = conductor
            .call(
                &cell1.zome("cgc"),
                "get_recognition_summary",
                receiver_did.clone(),
            )
            .await;

        assert_eq!(summary.member_did, receiver_did, "Member DID must match");
        assert!(
            summary.total_received >= 5,
            "Should have received at least 5"
        );
        assert!(
            summary.unique_givers >= 1,
            "Should have at least 1 unique giver"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_register_cultural_alias() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let alias_input = RegisterAliasInput {
            alias: "SPARK".to_string(),
            dao_did: "did:mycelix:dao-123".to_string(),
            meaning: "Sparking ideas and innovation".to_string(),
        };

        let alias: CulturalAlias = conductor
            .call(&cell.zome("cgc"), "register_cultural_alias", alias_input)
            .await;

        assert_eq!(alias.alias, "SPARK", "Alias should be uppercased");
        assert_eq!(alias.dao_did, "did:mycelix:dao-123", "DAO DID must match");

        // Query aliases for DAO
        let aliases: Vec<CulturalAlias> = conductor
            .call(
                &cell.zome("cgc"),
                "get_dao_aliases",
                "did:mycelix:dao-123".to_string(),
            )
            .await;

        assert!(!aliases.is_empty(), "Should have at least one alias");
    }
}

// ============================================================================
// Credit Scoring Tests (Deprecated but still functional)
// ============================================================================

#[cfg(test)]
mod credit_scoring_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_create_and_get_credit_profile() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let input = CreateProfileInput {
            did: "did:mycelix:borrower-001".to_string(),
            matl_score: 0.85,
            collateral_ratio: 0.5,
        };

        let profile_record: Record = conductor
            .call(&cell.zome("credit_scoring"), "create_credit_profile", input)
            .await;

        let profile: CreditProfile =
            decode_entry(&profile_record).expect("Failed to decode credit profile");

        assert_eq!(profile.did, "did:mycelix:borrower-001", "DID must match");
        assert_eq!(profile.matl_score, 0.85, "MATL score must match");
        assert!(
            profile.computed_score > 0.0,
            "Computed score should be positive"
        );
        assert_eq!(profile.version, 1, "Initial version must be 1");

        // Get profile by DID
        let fetched: Option<Record> = conductor
            .call(
                &cell.zome("credit_scoring"),
                "get_credit_profile",
                "did:mycelix:borrower-001".to_string(),
            )
            .await;

        assert!(fetched.is_some(), "Should find profile by DID");
    }
}

// ============================================================================
// Lending Tests (Deprecated but still functional)
// ============================================================================

#[cfg(test)]
mod lending_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_request_loan() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let input = RequestLoanInput {
            borrower_did: "did:mycelix:borrower-001".to_string(),
            amount: 5000.0,
            currency: "USD".to_string(),
            term_days: 90,
            collateral_ids: vec![],
        };

        let loan_record: Record = conductor
            .call(&cell.zome("lending"), "request_loan", input)
            .await;

        let loan: Loan = decode_entry(&loan_record).expect("Failed to decode loan");

        assert_eq!(
            loan.borrower_did, "did:mycelix:borrower-001",
            "Borrower DID must match"
        );
        assert_eq!(loan.principal, 5000.0, "Principal must match");
        assert_eq!(loan.currency, "USD", "Currency must match");
        assert_eq!(loan.term_days, 90, "Term must match");
        assert_eq!(
            loan.status,
            LoanStatus::Requested,
            "Status must be Requested"
        );

        // Get borrower loans
        let loans: Vec<Record> = conductor
            .call(
                &cell.zome("lending"),
                "get_borrower_loans",
                "did:mycelix:borrower-001".to_string(),
            )
            .await;

        assert!(!loans.is_empty(), "Should find at least one loan");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_create_loan_offer() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let input = CreateOfferInput {
            lender_did: "did:mycelix:lender-001".to_string(),
            max_amount: 10000.0,
            min_amount: 100.0,
            currency: "USD".to_string(),
            base_interest_rate: 0.05,
            min_credit_score: 500.0,
            max_term_days: 365,
            collateral_required: false,
        };

        let offer_record: Record = conductor
            .call(&cell.zome("lending"), "create_loan_offer", input)
            .await;

        assert!(
            offer_record.entry().as_option().is_some(),
            "Offer record must have entry"
        );

        // Get active offers
        let offers: Vec<Record> = conductor
            .call(&cell.zome("lending"), "get_active_offers", ())
            .await;

        assert!(!offers.is_empty(), "Should have at least one active offer");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_loan_lifecycle_request_fund_repay() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // 1. Request loan
        let request_input = RequestLoanInput {
            borrower_did: "did:mycelix:borrower-lc".to_string(),
            amount: 2000.0,
            currency: "USD".to_string(),
            term_days: 30,
            collateral_ids: vec![],
        };

        let loan_record: Record = conductor
            .call(&cell.zome("lending"), "request_loan", request_input)
            .await;

        let loan: Loan = decode_entry(&loan_record).expect("Failed to decode loan");
        let loan_id = loan.id.clone();
        assert_eq!(
            loan.status,
            LoanStatus::Requested,
            "Loan should be Requested"
        );

        // 2. Fund loan
        let fund_input = FundLoanInput {
            loan_id: loan_id.clone(),
            lender_did: "did:mycelix:lender-lc".to_string(),
            interest_rate: 0.03,
        };

        let funded_record: Record = conductor
            .call(&cell.zome("lending"), "fund_loan", fund_input)
            .await;

        let funded_loan: Loan = decode_entry(&funded_record).expect("Failed to decode funded loan");
        assert_eq!(
            funded_loan.status,
            LoanStatus::Funded,
            "Loan should be Funded"
        );
        assert_eq!(
            funded_loan.lender_did, "did:mycelix:lender-lc",
            "Lender should be set"
        );

        // 3. Repay loan
        let repaid_record: Record = conductor
            .call(&cell.zome("lending"), "repay_loan", loan_id.clone())
            .await;

        let repaid_loan: Loan = decode_entry(&repaid_record).expect("Failed to decode repaid loan");
        assert_eq!(
            repaid_loan.status,
            LoanStatus::Repaid,
            "Loan should be Repaid"
        );
        assert!(
            repaid_loan.repaid.is_some(),
            "Repaid timestamp should be set"
        );
    }
}

// ============================================================================
// Hearth (Commons Pool) Tests
// ============================================================================

#[cfg(test)]
mod hearth_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_create_pool() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let input = CreatePoolInput {
            name: "Community Emergency Fund".to_string(),
            description: "Pool for community emergency needs".to_string(),
            dao_did: "did:mycelix:dao-test".to_string(),
            resource_type: ResourceType::Money,
            cultural_alias: Some("HEARTH".to_string()),
            governance: None,
        };

        let pool: PoolSummary = conductor
            .call(&cell.zome("hearth"), "create_pool", input)
            .await;

        assert_eq!(
            pool.name, "Community Emergency Fund",
            "Pool name must match"
        );
        assert_eq!(pool.dao_did, "did:mycelix:dao-test", "DAO DID must match");
        assert_eq!(
            pool.resource_type,
            ResourceType::Money,
            "Resource type must match"
        );
        assert_eq!(
            pool.status,
            PoolStatus::Proposed,
            "New pool should be Proposed"
        );
        assert_eq!(
            pool.total_resources, 0.0,
            "New pool should have no resources"
        );

        // Get DAO pools
        let pools: Vec<PoolSummary> = conductor
            .call(
                &cell.zome("hearth"),
                "get_dao_pools",
                "did:mycelix:dao-test".to_string(),
            )
            .await;

        assert!(!pools.is_empty(), "Should find at least one pool");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_contribute_to_pool() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Create pool first
        let pool_input = CreatePoolInput {
            name: "Test Pool".to_string(),
            description: "A test pool".to_string(),
            dao_did: "did:mycelix:dao-contrib".to_string(),
            resource_type: ResourceType::Money,
            cultural_alias: None,
            governance: None,
        };

        let pool: PoolSummary = conductor
            .call(&cell.zome("hearth"), "create_pool", pool_input)
            .await;

        // Contribute
        let contrib_input = ContributeInput {
            pool_id: pool.id.clone(),
            amount: 50.0,
            message: Some("Happy to help!".to_string()),
            anonymous: false,
        };

        let contribution: PoolContribution = conductor
            .call(&cell.zome("hearth"), "contribute", contrib_input)
            .await;

        assert_eq!(contribution.pool_id, pool.id, "Pool ID must match");
        assert_eq!(contribution.amount, 50.0, "Contribution amount must match");
        assert!(!contribution.anonymous, "Should not be anonymous");

        // Get pool contributions
        let contributions: Vec<PoolContribution> = conductor
            .call(
                &cell.zome("hearth"),
                "get_pool_contributions",
                pool.id.clone(),
            )
            .await;

        assert!(
            !contributions.is_empty(),
            "Should have at least one contribution"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_request_allocation_and_vote() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Create pool
        let pool_input = CreatePoolInput {
            name: "Allocation Test Pool".to_string(),
            description: "Pool for testing allocations".to_string(),
            dao_did: "did:mycelix:dao-alloc".to_string(),
            resource_type: ResourceType::Money,
            cultural_alias: None,
            governance: None,
        };

        let pool: PoolSummary = conductor
            .call(&cell.zome("hearth"), "create_pool", pool_input)
            .await;

        // Request allocation
        let request_input = RequestAllocationInput {
            pool_id: pool.id.clone(),
            recipient_did: None,
            amount: 25.0,
            justification: "Need help with medical expenses".to_string(),
            need_category: NeedCategory::Healthcare,
        };

        let request: RequestSummary = conductor
            .call(&cell.zome("hearth"), "request_allocation", request_input)
            .await;

        assert_eq!(request.pool_id, pool.id, "Pool ID must match");
        assert_eq!(request.amount, 25.0, "Requested amount must match");
        assert_eq!(
            request.status,
            RequestStatus::Voting,
            "Status should be Voting"
        );
        assert_eq!(
            request.need_category,
            NeedCategory::Healthcare,
            "Category must match"
        );

        // Vote on request
        let vote_input = VoteInput {
            request_id: request.id.clone(),
            vote: true,
            comment: Some("Fully support this request".to_string()),
        };

        let vote: AllocationVote = conductor
            .call(&cell.zome("hearth"), "vote_on_request", vote_input)
            .await;

        assert_eq!(vote.request_id, request.id, "Request ID must match");
        assert!(vote.vote, "Vote should be for");

        // Get pool requests
        let requests: Vec<RequestSummary> = conductor
            .call(&cell.zome("hearth"), "get_pool_requests", pool.id.clone())
            .await;

        assert!(!requests.is_empty(), "Should have at least one request");
    }
}

// ============================================================================
// TEND (Time Exchange) Tests
// ============================================================================

#[cfg(test)]
mod tend_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_record_exchange() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;

        let app1 = conductor
            .setup_app("provider-app", &[dna.clone()])
            .await
            .unwrap();
        let app2 = conductor.setup_app("receiver-app", &[dna]).await.unwrap();
        let cell1 = app1.cells()[0].clone();
        let agent2 = app2.agent().clone();

        let receiver_did = format!("did:mycelix:{}", agent2);

        let input = RecordExchangeInput {
            receiver_did: receiver_did.clone(),
            hours: 2.0,
            service_description: "Tutoring session on Rust programming".to_string(),
            service_category: ServiceCategory::Education,
            cultural_alias: None,
            dao_did: "did:mycelix:dao-tend".to_string(),
            service_date: None,
        };

        let exchange: ExchangeRecord = conductor
            .call(&cell1.zome("tend"), "record_exchange", input)
            .await;

        assert_eq!(exchange.receiver_did, receiver_did, "Receiver must match");
        assert_eq!(exchange.hours, 2.0, "Hours must match");
        assert_eq!(
            exchange.status,
            ExchangeStatus::Proposed,
            "Status should be Proposed"
        );
        assert_eq!(
            exchange.service_category,
            ServiceCategory::Education,
            "Category must match"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_balance() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let member_did = format!("did:mycelix:{}", agent);

        let input = GetBalanceInput {
            member_did: member_did.clone(),
            dao_did: "did:mycelix:dao-balance".to_string(),
        };

        let balance: BalanceInfo = conductor
            .call(&cell.zome("tend"), "get_balance", input)
            .await;

        assert_eq!(balance.member_did, member_did, "Member DID must match");
        assert_eq!(balance.balance, 0, "New balance should be 0");
        assert!(balance.can_provide, "Should be able to provide");
        assert!(balance.can_receive, "Should be able to receive");
        assert_eq!(balance.exchange_count, 0, "Should have 0 exchanges");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_create_service_listing() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let input = CreateListingInput {
            dao_did: "did:mycelix:dao-listing".to_string(),
            title: "Web Development Help".to_string(),
            description: "I can help with web development projects".to_string(),
            category: ServiceCategory::Technology,
            estimated_hours: Some(4.0),
            availability: Some("Weekdays 9-5".to_string()),
        };

        let listing: ServiceListing = conductor
            .call(&cell.zome("tend"), "create_listing", input)
            .await;

        assert_eq!(listing.title, "Web Development Help", "Title must match");
        assert_eq!(
            listing.category,
            ServiceCategory::Technology,
            "Category must match"
        );
        assert!(listing.active, "Listing should be active");

        // Get DAO listings
        let listings: Vec<ServiceListing> = conductor
            .call(
                &cell.zome("tend"),
                "get_dao_listings",
                "did:mycelix:dao-listing".to_string(),
            )
            .await;

        assert!(!listings.is_empty(), "Should have at least one listing");
    }
}

// ============================================================================
// Full Lifecycle Test
// ============================================================================

#[cfg(test)]
mod lifecycle_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_complete_finance_lifecycle() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;

        let app1 = conductor
            .setup_app("agent1-app", &[dna.clone()])
            .await
            .unwrap();
        let app2 = conductor.setup_app("agent2-app", &[dna]).await.unwrap();
        let cell1 = app1.cells()[0].clone();
        let agent1 = app1.agent().clone();
        let agent2 = app2.agent().clone();

        let did1 = format!("did:mycelix:{}", agent1);
        let did2 = format!("did:mycelix:{}", agent2);

        // 1. Check CGC allocation
        let alloc: AllocationStatus = conductor
            .call(&cell1.zome("cgc"), "get_or_create_allocation", did1.clone())
            .await;

        assert_eq!(alloc.total_allocated, 10, "Should have 10 CGC");

        // 2. Gift CGC to agent2
        let gift_input = GiftCgcInput {
            receiver_did: did2.clone(),
            amount: 2,
            gratitude_message: "Welcome to the community!".to_string(),
            contribution_type: ContributionType::Gratitude,
            cultural_alias: None,
        };

        let _: GiftRecord = conductor
            .call(&cell1.zome("cgc"), "gift_cgc", gift_input)
            .await;

        // 3. Create a commons pool
        let pool_input = CreatePoolInput {
            name: "Lifecycle Test Pool".to_string(),
            description: "Pool for lifecycle test".to_string(),
            dao_did: "did:mycelix:dao-lifecycle".to_string(),
            resource_type: ResourceType::Money,
            cultural_alias: None,
            governance: None,
        };

        let pool: PoolSummary = conductor
            .call(&cell1.zome("hearth"), "create_pool", pool_input)
            .await;

        assert!(!pool.id.is_empty(), "Pool should have an ID");

        // 4. Contribute to pool
        let contrib_input = ContributeInput {
            pool_id: pool.id.clone(),
            amount: 100.0,
            message: Some("Lifecycle contribution".to_string()),
            anonymous: false,
        };

        let _: PoolContribution = conductor
            .call(&cell1.zome("hearth"), "contribute", contrib_input)
            .await;

        // 5. Check TEND balance
        let balance_input = GetBalanceInput {
            member_did: did1.clone(),
            dao_did: "did:mycelix:dao-lifecycle".to_string(),
        };

        let balance: BalanceInfo = conductor
            .call(&cell1.zome("tend"), "get_balance", balance_input)
            .await;

        assert_eq!(balance.balance, 0, "Initial TEND balance should be 0");

        // 6. Record a time exchange
        let exchange_input = RecordExchangeInput {
            receiver_did: did2.clone(),
            hours: 1.5,
            service_description: "Community garden planning".to_string(),
            service_category: ServiceCategory::Manual,
            cultural_alias: None,
            dao_did: "did:mycelix:dao-lifecycle".to_string(),
            service_date: None,
        };

        let exchange: ExchangeRecord = conductor
            .call(&cell1.zome("tend"), "record_exchange", exchange_input)
            .await;

        assert_eq!(
            exchange.status,
            ExchangeStatus::Proposed,
            "Exchange should be Proposed"
        );
    }
}
