// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! # Mycelix Care - Sweettest Integration Tests
//!
//! Uses pre-built .dna bundle instead of linking zome crates directly,
//! avoiding duplicate `__num_entry_types`/`__num_link_types` symbol collisions
//! from multiple integrity zomes.
//!
//! ## Running Tests
//!
//! ```bash
//! # Build the DNA first
//! cd mycelix-care && hc dna pack dna/
//!
//! # Run tests (require DNA bundle)
//! cargo test --test sweettest_integration -- --ignored
//! ```

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types for deserialization (avoids importing zome crates)
// ============================================================================

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ServiceCategory {
    Childcare,
    Eldercare,
    PetCare,
    Cooking,
    Cleaning,
    Gardening,
    Tutoring,
    TechSupport,
    Transportation,
    Companionship,
    HealthSupport,
    HomeRepair,
    LegalAdvice,
    Counseling,
    ArtMusic,
    LanguageHelp,
    Administrative,
    Other(String),
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum UrgencyLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ServiceOffer {
    pub provider: AgentPubKey,
    pub category: ServiceCategory,
    pub title: String,
    pub description: String,
    pub hours_available: f32,
    pub availability: String,
    pub location: String,
    pub skills_required: Vec<String>,
    pub active: bool,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ServiceRequest {
    pub requester: AgentPubKey,
    pub category: ServiceCategory,
    pub title: String,
    pub description: String,
    pub hours_needed: f32,
    pub preferred_schedule: String,
    pub location: String,
    pub urgency: UrgencyLevel,
    pub open: bool,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TimeExchange {
    pub offer_id: ActionHash,
    pub request_id: ActionHash,
    pub provider: AgentPubKey,
    pub recipient: AgentPubKey,
    pub hours: f32,
    pub category: ServiceCategory,
    pub completed_at: Timestamp,
    pub rating_provider: Option<u8>,
    pub rating_recipient: Option<u8>,
    pub notes: String,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TimeCredit {
    pub agent: AgentPubKey,
    pub balance: f64,
    pub total_earned: f64,
    pub total_spent: f64,
    pub updated_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompleteExchangeInput {
    pub offer_id: ActionHash,
    pub request_id: ActionHash,
    pub provider: AgentPubKey,
    pub recipient: AgentPubKey,
    pub hours: f32,
    pub category: ServiceCategory,
    pub notes: String,
}

// ============================================================================
// Test Utilities
// ============================================================================

fn dna_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("dna")
        .join("mycelix_care_dna.dna")
}

async fn load_dna() -> DnaFile {
    SweetDnaFile::from_bundle(&dna_path())
        .await
        .expect("Failed to load care DNA bundle")
}

fn decode_entry<T: serde::de::DeserializeOwned>(record: &Record) -> Option<T> {
    match record.entry().as_option()? {
        Entry::App(bytes) => {
            let sb = SerializedBytes::from(bytes.to_owned());
            rmp_serde::from_slice(sb.bytes()).ok()
        }
        _ => None,
    }
}

fn now_timestamp() -> Timestamp {
    Timestamp::from_micros(chrono::Utc::now().timestamp_micros())
}

fn make_offer(provider: &AgentPubKey, category: ServiceCategory) -> ServiceOffer {
    let now = now_timestamp();
    ServiceOffer {
        provider: provider.clone(),
        category,
        title: "Test Tutoring Service".to_string(),
        description: "Offering math tutoring for elementary students".to_string(),
        hours_available: 10.0,
        availability: "Weekday afternoons".to_string(),
        location: "Richardson, TX".to_string(),
        skills_required: vec!["math".to_string(), "patience".to_string()],
        active: true,
        created_at: now,
        updated_at: now,
    }
}

fn make_request(requester: &AgentPubKey, category: ServiceCategory) -> ServiceRequest {
    let now = now_timestamp();
    ServiceRequest {
        requester: requester.clone(),
        category,
        title: "Need tutoring help".to_string(),
        description: "Looking for math tutoring for my child".to_string(),
        hours_needed: 5.0,
        preferred_schedule: "Weekday evenings".to_string(),
        location: "Richardson, TX".to_string(),
        urgency: UrgencyLevel::Medium,
        open: true,
        created_at: now,
        updated_at: now,
    }
}

// ============================================================================
// Timebank Tests
// ============================================================================

#[cfg(test)]
mod timebank_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_create_and_get_offer() {
        println!("=== test_create_and_get_offer ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-care", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let offer = make_offer(&agent, ServiceCategory::Tutoring);

        let offer_record: Record = conductor
            .call(
                &cell.zome("timebank"),
                "create_service_offer",
                offer.clone(),
            )
            .await;

        let created: ServiceOffer = decode_entry(&offer_record).expect("No entry found");

        assert_eq!(created.title, "Test Tutoring Service");
        assert_eq!(created.provider, agent);
        assert!(created.active);

        println!("=== test_create_and_get_offer PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_all_active_offers() {
        println!("=== test_get_all_active_offers ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-care", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Create multiple offers in different categories
        let categories = vec![
            ServiceCategory::Tutoring,
            ServiceCategory::Gardening,
            ServiceCategory::Cooking,
        ];

        for cat in categories {
            let offer = make_offer(&agent, cat);
            let _: Record = conductor
                .call(&cell.zome("timebank"), "create_service_offer", offer)
                .await;
        }

        let active_offers: Vec<Record> = conductor
            .call(&cell.zome("timebank"), "get_all_active_offers", ())
            .await;

        assert!(
            active_offers.len() >= 3,
            "Should have at least 3 active offers"
        );

        println!("=== test_get_all_active_offers PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_create_service_request() {
        println!("=== test_create_service_request ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-care", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let request = make_request(&agent, ServiceCategory::Tutoring);

        let request_record: Record = conductor
            .call(
                &cell.zome("timebank"),
                "create_service_request",
                request.clone(),
            )
            .await;

        let created: ServiceRequest = decode_entry(&request_record).expect("No entry found");

        assert_eq!(created.title, "Need tutoring help");
        assert_eq!(created.requester, agent);
        assert!(created.open);

        // Verify it appears in open requests
        let open_requests: Vec<Record> = conductor
            .call(&cell.zome("timebank"), "get_all_open_requests", ())
            .await;

        assert!(
            open_requests.len() >= 1,
            "Should have at least 1 open request"
        );

        println!("=== test_create_service_request PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_initial_credit_balance() {
        println!("=== test_initial_credit_balance ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-care", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();

        let balance: TimeCredit = conductor
            .call(&cell.zome("timebank"), "get_my_balance", ())
            .await;

        // New agents start with 5.0 hour credit
        assert_eq!(balance.balance, 5.0, "Initial balance should be 5.0 hours");
        assert_eq!(balance.total_earned, 0.0);
        assert_eq!(balance.total_spent, 0.0);

        println!("=== test_initial_credit_balance PASSED ===\n");
    }
}

// ============================================================================
// Integration Flow Tests
// ============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_complete_care_exchange_flow() {
        println!("=== test_complete_care_exchange_flow ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app_alice = conductor
            .setup_app("app-alice", &[dna.clone()])
            .await
            .unwrap();
        let app_bob = conductor
            .setup_app("app-bob", &[dna.clone()])
            .await
            .unwrap();
        let alice_cell = app_alice.cells()[0].clone();
        let bob_cell = app_bob.cells()[0].clone();
        let alice = app_alice.agent().clone();
        let bob = app_bob.agent().clone();

        // 1. Alice creates a tutoring offer
        let offer = make_offer(&alice, ServiceCategory::Tutoring);
        let offer_record: Record = conductor
            .call(&alice_cell.zome("timebank"), "create_service_offer", offer)
            .await;
        let offer_hash = offer_record.action_address().clone();
        println!("  1. Alice created offer");

        // 2. Bob creates a tutoring request
        let request = make_request(&bob, ServiceCategory::Tutoring);
        let request_record: Record = conductor
            .call(
                &bob_cell.zome("timebank"),
                "create_service_request",
                request,
            )
            .await;
        let request_hash = request_record.action_address().clone();
        println!("  2. Bob created request");

        // 3. Complete the exchange (Alice provides, Bob receives)
        let exchange_input = CompleteExchangeInput {
            offer_id: offer_hash,
            request_id: request_hash,
            provider: alice.clone(),
            recipient: bob.clone(),
            hours: 3.0,
            category: ServiceCategory::Tutoring,
            notes: "Great tutoring session".to_string(),
        };

        let exchange_record: Record = conductor
            .call(
                &alice_cell.zome("timebank"),
                "complete_exchange",
                exchange_input,
            )
            .await;

        let exchange: TimeExchange = decode_entry(&exchange_record).expect("No entry found");
        assert_eq!(exchange.hours, 3.0);
        assert_eq!(exchange.provider, alice);
        assert_eq!(exchange.recipient, bob);
        println!("  3. Exchange completed: 3 hours tutoring");

        // 4. Verify Alice earned credits (5.0 initial + 3.0 earned = 8.0)
        let alice_balance: TimeCredit = conductor
            .call(&alice_cell.zome("timebank"), "get_my_balance", ())
            .await;
        assert!(
            alice_balance.balance >= 8.0,
            "Alice should have at least 8.0 hours (5 initial + 3 earned)"
        );
        println!("  4. Alice balance: {}", alice_balance.balance);

        // 5. Verify Bob spent credits (5.0 initial - 3.0 spent = 2.0)
        let bob_balance: TimeCredit = conductor
            .call(&bob_cell.zome("timebank"), "get_my_balance", ())
            .await;
        assert!(
            bob_balance.balance <= 2.0,
            "Bob should have at most 2.0 hours (5 initial - 3 spent)"
        );
        println!("  5. Bob balance: {}", bob_balance.balance);

        println!("=== test_complete_care_exchange_flow PASSED ===\n");
    }
}

// ============================================================================
// Unit Tests (No Conductor Required)
// ============================================================================

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_service_category_serialization() {
        let categories = vec![
            ServiceCategory::Childcare,
            ServiceCategory::Tutoring,
            ServiceCategory::Other("custom".to_string()),
        ];

        for cat in categories {
            let json = serde_json::to_string(&cat).expect("serialize");
            let deserialized: ServiceCategory = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(cat, deserialized);
        }
    }

    #[test]
    fn test_urgency_level_serialization() {
        let levels = vec![
            UrgencyLevel::Low,
            UrgencyLevel::Medium,
            UrgencyLevel::High,
            UrgencyLevel::Critical,
        ];

        for level in levels {
            let json = serde_json::to_string(&level).expect("serialize");
            let deserialized: UrgencyLevel = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(level, deserialized);
        }
    }
}
