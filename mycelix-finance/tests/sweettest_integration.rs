//! # Mycelix Finance - Sweettest Integration Tests
//!
//! Comprehensive integration tests using Holochain's sweettest framework.
//! Tests cover the 3-currency model (MYCEL, SAP, TEND) across the new zome set:
//! payments, treasury, bridge, staking, tend, recognition.
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

use holochain::sweettest::*;
use holochain::prelude::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types (avoids importing zome crates / duplicate symbols)
// ============================================================================

// --- Recognition Mirror Types ---

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ContributionType {
    Technical,
    Community,
    Care,
    Governance,
    Creative,
    Education,
    General,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RecognizeMemberInput {
    pub recipient_did: String,
    pub contribution_type: ContributionType,
    pub cycle_id: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct InitializeMemberInput {
    pub member_did: String,
    pub is_apprentice: bool,
    pub mentor_did: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MemberMycelState {
    pub member_did: String,
    pub mycel_score: f64,
    pub participation: f64,
    pub recognition: f64,
    pub validation: f64,
    pub longevity: f64,
    pub active_months: u32,
    pub is_apprentice: bool,
    pub mentor_did: Option<String>,
    pub recognitions_given_this_cycle: u32,
    pub current_cycle_id: String,
    pub last_updated: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RecognitionEvent {
    pub recognizer_did: String,
    pub recipient_did: String,
    pub weight: f64,
    pub contribution_type: ContributionType,
    pub cycle_id: String,
    pub recognizer_mycel: f64,
    pub timestamp: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GetRecognitionsInput {
    pub member_did: String,
    pub cycle_id: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct UpdateMycelInput {
    pub member_did: String,
    pub participation: f64,
    pub recognition: f64,
    pub validation_override: Option<f64>,
    pub active_months: u32,
}

// --- Commons Pool Mirror Types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateCommonsPoolInput {
    pub dao_did: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ContributeToCommonsInput {
    pub commons_pool_id: String,
    pub contributor_did: String,
    pub amount: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CommonsPool {
    pub id: String,
    pub dao_did: String,
    pub inalienable_reserve: u64,
    pub available_balance: u64,
    pub demurrage_exempt: bool,
    pub created_at: Timestamp,
    pub last_activity: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RequestCommonsAllocationInput {
    pub commons_pool_id: String,
    pub requester_did: String,
    pub amount: u64,
    pub purpose: String,
}

// --- Collateral Stake Mirror Types ---

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum StakeStatus {
    Active,
    Unbonding,
    Withdrawn,
    Slashed,
    Jailed,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CollateralStake {
    pub id: String,
    pub staker_did: String,
    pub sap_amount: u64,
    pub mycel_score: f32,
    pub stake_weight: f32,
    pub staked_at: Timestamp,
    pub unbonding_until: Option<Timestamp>,
    pub status: StakeStatus,
    pub pending_rewards: u64,
    pub last_reward_claim: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateStakeInput {
    pub staker_did: String,
    pub sap_amount: u64,
}

// --- Payments Mirror Types ---

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PaymentType {
    Direct,
    TreasuryContribution(String),
    CommonsContribution(String),
    Escrow(String),
    Recurring(RecurringConfig),
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RecurringConfig {
    pub frequency_days: u32,
    pub end_date: Option<Timestamp>,
    pub remaining: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum TransferStatus {
    Pending,
    Processing,
    Completed,
    Failed(String),
    Cancelled,
    Refunded,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Payment {
    pub id: String,
    pub from_did: String,
    pub to_did: String,
    pub amount: u64,
    pub fee: u64,
    pub currency: String,
    pub payment_type: PaymentType,
    pub status: TransferStatus,
    pub memo: Option<String>,
    pub created: Timestamp,
    pub completed: Option<Timestamp>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SendPaymentInput {
    pub from_did: String,
    pub to_did: String,
    pub amount: u64,
    pub currency: String,
    pub payment_type: PaymentType,
    pub memo: Option<String>,
}

// --- Paginated API Input Types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PaginatedDaoInput {
    pub dao_did: String,
    pub limit: Option<usize>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GetPaymentHistoryInput {
    pub did: String,
    pub limit: Option<usize>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GetValidationScoreInput {
    pub member_did: String,
    pub limit: Option<usize>,
}

// --- TEND Mirror Types ---

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ServiceCategory {
    CareWork,
    HomeServices,
    FoodServices,
    Transportation,
    Education,
    GeneralAssistance,
    Administrative,
    Creative,
    TechSupport,
    Wellness,
    Gardening,
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
// TEND (Time Exchange) Tests
// ============================================================================

#[cfg(test)]
mod tend_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // Requires Holochain conductor
    async fn test_record_exchange() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;

        let app1 = conductor.setup_app("provider-app", &[dna.clone()]).await.unwrap();
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
        assert_eq!(exchange.status, ExchangeStatus::Proposed, "Status should be Proposed");
        assert_eq!(exchange.service_category, ServiceCategory::Education, "Category must match");
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
            category: ServiceCategory::TechSupport,
            estimated_hours: Some(4.0),
            availability: Some("Weekdays 9-5".to_string()),
        };

        let listing: ServiceListing = conductor
            .call(&cell.zome("tend"), "create_listing", input)
            .await;

        assert_eq!(listing.title, "Web Development Help", "Title must match");
        assert_eq!(listing.category, ServiceCategory::TechSupport, "Category must match");
        assert!(listing.active, "Listing should be active");

        // Get DAO listings
        let listings: Vec<ServiceListing> = conductor
            .call(&cell.zome("tend"), "get_dao_listings", PaginatedDaoInput {
                dao_did: "did:mycelix:dao-listing".to_string(),
                limit: None,
            })
            .await;

        assert!(!listings.is_empty(), "Should have at least one listing");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_service_category_variants() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;

        let app1 = conductor.setup_app("provider-app", &[dna.clone()]).await.unwrap();
        let app2 = conductor.setup_app("receiver-app", &[dna]).await.unwrap();
        let cell1 = app1.cells()[0].clone();
        let agent2 = app2.agent().clone();

        let receiver_did = format!("did:mycelix:{}", agent2);

        // Test CareWork category
        let input = RecordExchangeInput {
            receiver_did: receiver_did.clone(),
            hours: 1.0,
            service_description: "Babysitting for the afternoon".to_string(),
            service_category: ServiceCategory::CareWork,
            cultural_alias: None,
            dao_did: "did:mycelix:dao-cats".to_string(),
            service_date: None,
        };

        let exchange: ExchangeRecord = conductor
            .call(&cell1.zome("tend"), "record_exchange", input)
            .await;

        assert_eq!(exchange.service_category, ServiceCategory::CareWork, "Category must be CareWork");

        // Test Gardening category
        let input_garden = RecordExchangeInput {
            receiver_did: receiver_did.clone(),
            hours: 3.0,
            service_description: "Community garden planting".to_string(),
            service_category: ServiceCategory::Gardening,
            cultural_alias: None,
            dao_did: "did:mycelix:dao-cats".to_string(),
            service_date: None,
        };

        let exchange_garden: ExchangeRecord = conductor
            .call(&cell1.zome("tend"), "record_exchange", input_garden)
            .await;

        assert_eq!(
            exchange_garden.service_category,
            ServiceCategory::Gardening,
            "Category must be Gardening"
        );
    }
}

// ============================================================================
// Recognition (MYCEL) Tests
// ============================================================================

#[cfg(test)]
mod recognition_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // Requires Holochain conductor
    async fn test_initialize_member() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna.clone()]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let member_did = format!("did:mycelix:{}", agent);

        // Initialize as a full member (non-apprentice)
        let input = InitializeMemberInput {
            member_did: member_did.clone(),
            is_apprentice: false,
            mentor_did: None,
        };

        let record: Record = conductor
            .call(&cell.zome("recognition"), "initialize_member", input)
            .await;

        let state: MemberMycelState = decode_entry(&record)
            .expect("Failed to decode MemberMycelState");

        assert_eq!(state.member_did, member_did, "Member DID must match");
        assert_eq!(state.mycel_score, 0.3, "Full member initial MYCEL should be 0.3");
        assert!(!state.is_apprentice, "Should not be an apprentice");
        assert!(state.mentor_did.is_none(), "Full member should have no mentor");
        assert_eq!(state.active_months, 0, "New member should have 0 active months");

        // Initialize an apprentice
        let app2 = conductor.setup_app("apprentice-app", &[dna.clone()]).await.unwrap();
        let cell2 = app2.cells()[0].clone();
        let agent2 = app2.agent().clone();
        let apprentice_did = format!("did:mycelix:{}", agent2);

        let apprentice_input = InitializeMemberInput {
            member_did: apprentice_did.clone(),
            is_apprentice: true,
            mentor_did: Some(member_did.clone()),
        };

        let apprentice_record: Record = conductor
            .call(&cell2.zome("recognition"), "initialize_member", apprentice_input)
            .await;

        let apprentice_state: MemberMycelState = decode_entry(&apprentice_record)
            .expect("Failed to decode apprentice MemberMycelState");

        assert_eq!(apprentice_state.member_did, apprentice_did, "Apprentice DID must match");
        assert_eq!(apprentice_state.mycel_score, 0.1, "Apprentice initial MYCEL should be 0.1");
        assert!(apprentice_state.is_apprentice, "Should be an apprentice");
        assert_eq!(
            apprentice_state.mentor_did,
            Some(member_did),
            "Mentor DID must match"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_recognize_member() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;

        // Set up two agents
        let app1 = conductor.setup_app("recognizer-app", &[dna.clone()]).await.unwrap();
        let app2 = conductor.setup_app("recipient-app", &[dna]).await.unwrap();
        let cell1 = app1.cells()[0].clone();
        let agent1 = app1.agent().clone();
        let agent2 = app2.agent().clone();

        let recognizer_did = format!("did:mycelix:{}", agent1);
        let recipient_did = format!("did:mycelix:{}", agent2);

        // Initialize recognizer as full member (MYCEL 0.3, above MIN_MYCEL_TO_GIVE)
        let init_input = InitializeMemberInput {
            member_did: recognizer_did.clone(),
            is_apprentice: false,
            mentor_did: None,
        };

        let _: Record = conductor
            .call(&cell1.zome("recognition"), "initialize_member", init_input)
            .await;

        // Recognize the recipient
        let recognize_input = RecognizeMemberInput {
            recipient_did: recipient_did.clone(),
            contribution_type: ContributionType::Technical,
            cycle_id: "2026-02".to_string(),
        };

        let recognition_record: Record = conductor
            .call(&cell1.zome("recognition"), "recognize_member", recognize_input)
            .await;

        let event: RecognitionEvent = decode_entry(&recognition_record)
            .expect("Failed to decode RecognitionEvent");

        assert_eq!(event.recognizer_did, recognizer_did, "Recognizer DID must match");
        assert_eq!(event.recipient_did, recipient_did, "Recipient DID must match");
        assert_eq!(event.contribution_type, ContributionType::Technical, "Type must be Technical");
        assert_eq!(event.cycle_id, "2026-02", "Cycle ID must match");
        assert!(event.weight > 0.0, "Weight must be positive");
        assert_eq!(event.recognizer_mycel, 0.3, "Recognizer MYCEL should be 0.3");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_recognition_score_computation() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let member_did = format!("did:mycelix:{}", agent);

        // Initialize member
        let init_input = InitializeMemberInput {
            member_did: member_did.clone(),
            is_apprentice: false,
            mentor_did: None,
        };

        let _: Record = conductor
            .call(&cell.zome("recognition"), "initialize_member", init_input)
            .await;

        // Get initial MYCEL state
        let initial_state: MemberMycelState = conductor
            .call(&cell.zome("recognition"), "get_mycel_score", member_did.clone())
            .await;

        assert_eq!(initial_state.mycel_score, 0.3, "Initial MYCEL should be 0.3");

        // Update MYCEL score with component values
        // Composite = participation(0.8) * 0.40 + recognition(0.6) * 0.20
        //           + validation(0.5) * 0.20 + longevity(12/24=0.5) * 0.20
        // = 0.32 + 0.12 + 0.10 + 0.10 = 0.64
        let update_input = UpdateMycelInput {
            member_did: member_did.clone(),
            participation: 0.8,
            recognition: 0.6,
            validation_override: Some(0.5),
            active_months: 12,
        };

        let updated_state: MemberMycelState = conductor
            .call(&cell.zome("recognition"), "update_mycel_score", update_input)
            .await;

        // Verify the composite calculation
        let expected = 0.8 * 0.40 + 0.6 * 0.20 + 0.5 * 0.20 + 0.5 * 0.20;
        assert!(
            (updated_state.mycel_score - expected).abs() < 0.01,
            "MYCEL score ({}) should be approximately {} (P*40% + R*20% + V*20% + L*20%)",
            updated_state.mycel_score,
            expected
        );
        assert_eq!(updated_state.participation, 0.8, "Participation component must match");
        assert_eq!(updated_state.recognition, 0.6, "Recognition component must match");
        assert_eq!(updated_state.validation, 0.5, "Validation component must match");
        assert_eq!(updated_state.active_months, 12, "Active months must match");
    }
}

// ============================================================================
// Commons Pool Tests
// ============================================================================

#[cfg(test)]
mod commons_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // Requires Holochain conductor
    async fn test_create_commons_pool() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let input = CreateCommonsPoolInput {
            dao_did: "did:mycelix:dao-commons-test".to_string(),
        };

        let pool_record: Record = conductor
            .call(&cell.zome("treasury"), "create_commons_pool", input)
            .await;

        let pool: CommonsPool = decode_entry(&pool_record)
            .expect("Failed to decode CommonsPool");

        assert_eq!(pool.dao_did, "did:mycelix:dao-commons-test", "DAO DID must match");
        assert_eq!(pool.inalienable_reserve, 0, "New pool should have 0 inalienable reserve");
        assert_eq!(pool.available_balance, 0, "New pool should have 0 available balance");
        assert!(pool.demurrage_exempt, "Commons pool must be demurrage exempt");
        assert!(!pool.id.is_empty(), "Pool should have an ID");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_contribute_to_commons() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();
        let contributor_did = format!("did:mycelix:{}", agent);

        // Create commons pool
        let create_input = CreateCommonsPoolInput {
            dao_did: "did:mycelix:dao-contrib-test".to_string(),
        };

        let pool_record: Record = conductor
            .call(&cell.zome("treasury"), "create_commons_pool", create_input)
            .await;

        let pool: CommonsPool = decode_entry(&pool_record)
            .expect("Failed to decode CommonsPool");

        // Contribute 1000 SAP to commons pool
        let contrib_input = ContributeToCommonsInput {
            commons_pool_id: pool.id.clone(),
            contributor_did: contributor_did.clone(),
            amount: 1000,
        };

        let updated_record: Record = conductor
            .call(&cell.zome("treasury"), "contribute_to_commons", contrib_input)
            .await;

        let updated_pool: CommonsPool = decode_entry(&updated_record)
            .expect("Failed to decode updated CommonsPool");

        // Verify 25/75 split: 25% to inalienable reserve, 75% to available
        assert_eq!(
            updated_pool.inalienable_reserve, 250,
            "Inalienable reserve should be 25% of 1000 = 250"
        );
        assert_eq!(
            updated_pool.available_balance, 750,
            "Available balance should be 75% of 1000 = 750"
        );
        assert!(updated_pool.demurrage_exempt, "Pool must remain demurrage exempt");

        // Second contribution to verify accumulation
        let contrib_input_2 = ContributeToCommonsInput {
            commons_pool_id: pool.id.clone(),
            contributor_did: contributor_did.clone(),
            amount: 400,
        };

        let updated_record_2: Record = conductor
            .call(&cell.zome("treasury"), "contribute_to_commons", contrib_input_2)
            .await;

        let updated_pool_2: CommonsPool = decode_entry(&updated_record_2)
            .expect("Failed to decode second update");

        // 400 / 4 = 100 to reserve, 300 to available
        assert_eq!(
            updated_pool_2.inalienable_reserve, 350,
            "Inalienable reserve should accumulate: 250 + 100 = 350"
        );
        assert_eq!(
            updated_pool_2.available_balance, 1050,
            "Available balance should accumulate: 750 + 300 = 1050"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_inalienable_reserve_untouchable() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();
        let member_did = format!("did:mycelix:{}", agent);

        // Create and fund a commons pool
        let create_input = CreateCommonsPoolInput {
            dao_did: "did:mycelix:dao-reserve-test".to_string(),
        };

        let pool_record: Record = conductor
            .call(&cell.zome("treasury"), "create_commons_pool", create_input)
            .await;

        let pool: CommonsPool = decode_entry(&pool_record)
            .expect("Failed to decode CommonsPool");

        // Contribute 1000 SAP (250 reserve, 750 available)
        let contrib_input = ContributeToCommonsInput {
            commons_pool_id: pool.id.clone(),
            contributor_did: member_did.clone(),
            amount: 1000,
        };

        let _: Record = conductor
            .call(&cell.zome("treasury"), "contribute_to_commons", contrib_input)
            .await;

        // Request allocation within available balance -- should succeed
        let valid_request = RequestCommonsAllocationInput {
            commons_pool_id: pool.id.clone(),
            requester_did: member_did.clone(),
            amount: 500,
            purpose: "Community project funding".to_string(),
        };

        let alloc_record: Record = conductor
            .call(&cell.zome("treasury"), "request_allocation", valid_request)
            .await;

        let alloc_pool: CommonsPool = decode_entry(&alloc_record)
            .expect("Failed to decode allocation result");

        assert_eq!(
            alloc_pool.inalienable_reserve, 250,
            "Inalienable reserve must remain untouched at 250"
        );
        assert_eq!(
            alloc_pool.available_balance, 250,
            "Available balance should decrease: 750 - 500 = 250"
        );

        // Attempt to allocate more than available balance -- should fail
        // because it would require touching the inalienable reserve
        let invalid_request = RequestCommonsAllocationInput {
            commons_pool_id: pool.id.clone(),
            requester_did: member_did.clone(),
            amount: 300, // Only 250 available
            purpose: "Overreach".to_string(),
        };

        let result: Result<Record, _> = conductor
            .call_fallible(&cell.zome("treasury"), "request_allocation", invalid_request)
            .await;

        assert!(
            result.is_err(),
            "Allocating more than available balance must fail (inalienable reserve is untouchable)"
        );
    }
}

// ============================================================================
// Three-Currency Lifecycle Test (MYCEL -> SAP -> TEND)
// ============================================================================

#[cfg(test)]
mod three_currency_lifecycle {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // Requires Holochain conductor
    async fn test_complete_lifecycle() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;

        let app1 = conductor.setup_app("agent1-app", &[dna.clone()]).await.unwrap();
        let app2 = conductor.setup_app("agent2-app", &[dna]).await.unwrap();
        let cell1 = app1.cells()[0].clone();
        let agent1 = app1.agent().clone();
        let agent2 = app2.agent().clone();

        let did1 = format!("did:mycelix:{}", agent1);
        let did2 = format!("did:mycelix:{}", agent2);

        // ---- Step 1: Initialize MYCEL (soulbound reputation) ----
        let init_input = InitializeMemberInput {
            member_did: did1.clone(),
            is_apprentice: false,
            mentor_did: None,
        };

        let init_record: Record = conductor
            .call(&cell1.zome("recognition"), "initialize_member", init_input)
            .await;

        let mycel_state: MemberMycelState = decode_entry(&init_record)
            .expect("Failed to decode MemberMycelState");

        assert_eq!(mycel_state.mycel_score, 0.3, "Initial MYCEL score should be 0.3");
        assert!(!mycel_state.is_apprentice, "Should be a full member");

        // ---- Step 2: Recognize another member (MYCEL-weighted) ----
        let recognize_input = RecognizeMemberInput {
            recipient_did: did2.clone(),
            contribution_type: ContributionType::Community,
            cycle_id: "2026-02".to_string(),
        };

        let recognition_record: Record = conductor
            .call(&cell1.zome("recognition"), "recognize_member", recognize_input)
            .await;

        let event: RecognitionEvent = decode_entry(&recognition_record)
            .expect("Failed to decode RecognitionEvent");

        assert_eq!(event.recognizer_did, did1, "Recognizer should be agent1");
        assert_eq!(event.recipient_did, did2, "Recipient should be agent2");
        assert!(event.weight > 0.0, "Recognition weight must be positive");

        // ---- Step 3: Contribute SAP to commons pool ----
        let create_pool_input = CreateCommonsPoolInput {
            dao_did: "did:mycelix:dao-lifecycle".to_string(),
        };

        let pool_record: Record = conductor
            .call(&cell1.zome("treasury"), "create_commons_pool", create_pool_input)
            .await;

        let pool: CommonsPool = decode_entry(&pool_record)
            .expect("Failed to decode CommonsPool");

        assert!(!pool.id.is_empty(), "Pool should have an ID");
        assert!(pool.demurrage_exempt, "Commons pool must be demurrage exempt");

        // Contribute SAP to the commons pool
        let contrib_input = ContributeToCommonsInput {
            commons_pool_id: pool.id.clone(),
            contributor_did: did1.clone(),
            amount: 2000,
        };

        let contrib_record: Record = conductor
            .call(&cell1.zome("treasury"), "contribute_to_commons", contrib_input)
            .await;

        let funded_pool: CommonsPool = decode_entry(&contrib_record)
            .expect("Failed to decode funded CommonsPool");

        assert_eq!(funded_pool.inalienable_reserve, 500, "25% of 2000 = 500 to reserve");
        assert_eq!(funded_pool.available_balance, 1500, "75% of 2000 = 1500 available");

        // ---- Step 4: Record a TEND time exchange ----
        let exchange_input = RecordExchangeInput {
            receiver_did: did2.clone(),
            hours: 1.5,
            service_description: "Community garden planning session".to_string(),
            service_category: ServiceCategory::Gardening,
            cultural_alias: None,
            dao_did: "did:mycelix:dao-lifecycle".to_string(),
            service_date: None,
        };

        let exchange: ExchangeRecord = conductor
            .call(&cell1.zome("tend"), "record_exchange", exchange_input)
            .await;

        assert_eq!(exchange.status, ExchangeStatus::Proposed, "Exchange should be Proposed");
        assert_eq!(exchange.hours, 1.5, "Hours must match");
        assert_eq!(
            exchange.service_category,
            ServiceCategory::Gardening,
            "Category must be Gardening"
        );

        // ---- Step 5: Check TEND balance (should still be 0, exchange is Proposed) ----
        let balance_input = GetBalanceInput {
            member_did: did1.clone(),
            dao_did: "did:mycelix:dao-lifecycle".to_string(),
        };

        let balance: BalanceInfo = conductor
            .call(&cell1.zome("tend"), "get_balance", balance_input)
            .await;

        assert_eq!(balance.balance, 0, "TEND balance should be 0 (exchange not yet confirmed)");
        assert!(balance.can_provide, "Should be able to provide");
        assert!(balance.can_receive, "Should be able to receive");

        // ---- Initialize SAP balances for both agents (required before SAP transfer) ----
        let _: Record = conductor
            .call(&cell1.zome("payments"), "initialize_sap_balance", did1.clone())
            .await;

        let cell2 = app2.cells()[0].clone();
        let _: Record = conductor
            .call(&cell2.zome("payments"), "initialize_sap_balance", did2.clone())
            .await;

        // ---- Verify: payments only accept SAP or TEND ----
        let payment_input = SendPaymentInput {
            from_did: did1.clone(),
            to_did: did2.clone(),
            amount: 100_000_000,
            currency: "SAP".to_string(),
            payment_type: PaymentType::Direct,
            memo: Some("Lifecycle test SAP payment".to_string()),
        };

        let payment_record: Record = conductor
            .call(&cell1.zome("payments"), "send_payment", payment_input)
            .await;

        let payment: Payment = decode_entry(&payment_record)
            .expect("Failed to decode Payment");

        assert_eq!(payment.currency, "SAP", "Currency must be SAP");
        assert_eq!(payment.status, TransferStatus::Completed, "Payment should complete");
        assert_eq!(payment.amount, 100_000_000, "Amount must match");
    }
}
