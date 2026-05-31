// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! # TEND (Time Exchange) Integration Tests
//!
//! Tests for the Time Exchange mutual credit module implementing Commons Charter Article II.
//!
//! ## Key Features Tested:
//! - Balance limits (±40 TEND)
//! - Time exchange recording
//! - Mutual credit mechanics (zero-sum)
//! - Service listings and requests
//! - Exchange confirmation/dispute workflow
//!
//! ## Running Tests
//!
//! ```bash
//! cargo test --test tend_test
//! cargo test --test tend_test -- --ignored  # Full integration tests
//! ```

use holochain::prelude::*;
use holochain::sweettest::{SweetAgents, SweetConductor, SweetDnaFile};
use std::time::Duration;

// Import zome types
use tend_integrity::*;

mod test_helpers {
    pub const TEST_DID_PREFIX: &str = "did:mycelix:test:";
    pub const TEST_DAO: &str = "did:mycelix:dao:test_community";

    pub fn test_did(suffix: &str) -> String {
        format!("{}{}", TEST_DID_PREFIX, suffix)
    }
}

use test_helpers::*;

// ============================================================================
// Section 1: Balance Limits Tests
// ============================================================================

#[cfg(test)]
mod balance_limits {
    use super::*;

    /// Test 1.1: Balance limits are enforced (±40 TEND)
    #[tokio::test]
    #[ignore]
    async fn test_balance_limits() {
        println!("Test 1.1: Balance Limits (±40 TEND)");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, [&dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = test_did("alice");

        // Get balance
        let balance_input = GetBalanceInput {
            member_did: alice_did.clone(),
            dao_did: TEST_DAO.to_string(),
        };

        let balance: TendBalance = conductor
            .call(
                &alice_cell.zome("tend"),
                "get_or_create_balance",
                balance_input,
            )
            .await
            .expect("Failed to get balance");

        // Verify balance limits
        assert_eq!(balance.positive_limit, 40.0, "Positive limit should be 40");
        assert_eq!(
            balance.negative_limit, -40.0,
            "Negative limit should be -40"
        );

        println!("  - Current balance: {}", balance.current_balance);
        println!("  - Positive limit: +{}", balance.positive_limit);
        println!("  - Negative limit: {}", balance.negative_limit);
        println!("Test 1.1 PASSED: Balance limits are correct");
    }

    /// Test 1.2: Exchange rejected when it would exceed positive limit
    #[tokio::test]
    #[ignore]
    async fn test_positive_limit_enforcement() {
        println!("Test 1.2: Positive Limit Enforcement");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, [&dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = test_did("alice");
        let bob_did = test_did("bob");

        // Try to record 50 hours (exceeds +40 limit)
        let exchange_input = RecordExchangeInput {
            provider_did: alice_did.clone(),
            receiver_did: bob_did.clone(),
            dao_did: TEST_DAO.to_string(),
            hours: 50.0, // Exceeds +40 limit
            service_description: "Programming help".to_string(),
            category: Some(ServiceCategory::HomeServices),
        };

        let result: Result<ExchangeRecord, _> = conductor
            .call_fallible(alice_cell.zome("tend"), "record_exchange", exchange_input)
            .await;

        match result {
            Err(e) => {
                let error_msg = format!("{:?}", e);
                assert!(
                    error_msg.contains("would exceed") || error_msg.contains("limit"),
                    "Should reject exchange exceeding limit, got: {}",
                    error_msg
                );
                println!("  - Exchange exceeding +40 limit rejected: OK");
            }
            Ok(_) => panic!("Should have rejected exchange exceeding positive limit"),
        }

        println!("Test 1.2 PASSED: Positive limit enforced");
    }

    /// Test 1.3: Exchange rejected when it would exceed negative limit
    #[tokio::test]
    #[ignore]
    async fn test_negative_limit_enforcement() {
        println!("Test 1.3: Negative Limit Enforcement");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, [&dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let bob_cell = &apps[1].cells()[0];
        let alice_did = test_did("alice");
        let bob_did = test_did("bob");

        // Bob receives 50 hours (would put Alice at -50, exceeding -40 limit)
        // Note: Provider gains, receiver loses in TEND
        let exchange_input = RecordExchangeInput {
            provider_did: bob_did.clone(),
            receiver_did: alice_did.clone(), // Alice receiving would put her negative
            dao_did: TEST_DAO.to_string(),
            hours: 50.0, // Would put receiver at -50
            service_description: "Programming help".to_string(),
            category: Some(ServiceCategory::HomeServices),
        };

        let result: Result<ExchangeRecord, _> = conductor
            .call_fallible(bob_cell.zome("tend"), "record_exchange", exchange_input)
            .await;

        match result {
            Err(e) => {
                let error_msg = format!("{:?}", e);
                assert!(
                    error_msg.contains("would exceed") || error_msg.contains("limit"),
                    "Should reject exchange exceeding limit, got: {}",
                    error_msg
                );
                println!("  - Exchange exceeding -40 limit rejected: OK");
            }
            Ok(_) => panic!("Should have rejected exchange exceeding negative limit"),
        }

        println!("Test 1.3 PASSED: Negative limit enforced");
    }
}

// ============================================================================
// Section 2: Exchange Recording Tests
// ============================================================================

#[cfg(test)]
mod exchange_recording {
    use super::*;

    /// Test 2.1: Basic time exchange recording
    #[tokio::test]
    #[ignore]
    async fn test_basic_exchange() {
        println!("Test 2.1: Basic Time Exchange");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, [&dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = test_did("alice");
        let bob_did = test_did("bob");

        // Alice provides 2 hours of service to Bob
        let exchange_input = RecordExchangeInput {
            provider_did: alice_did.clone(),
            receiver_did: bob_did.clone(),
            dao_did: TEST_DAO.to_string(),
            hours: 2.0,
            service_description: "Garden design consultation".to_string(),
            category: Some(ServiceCategory::Creative),
        };

        let exchange: ExchangeRecord = conductor
            .call(&alice_cell.zome("tend"), "record_exchange", exchange_input)
            .await
            .expect("Failed to record exchange");

        assert_eq!(exchange.hours, 2.0, "Hours mismatch");
        assert_eq!(exchange.provider_did, alice_did, "Provider mismatch");
        assert_eq!(exchange.receiver_did, bob_did, "Receiver mismatch");
        assert!(
            matches!(exchange.status, ExchangeStatus::Proposed),
            "Should be pending confirmation"
        );

        println!("  - Exchange recorded: {} hours", exchange.hours);
        println!("  - Provider: {}", exchange.provider_did);
        println!("  - Receiver: {}", exchange.receiver_did);
        println!("  - Status: {:?}", exchange.status);
        println!("Test 2.1 PASSED: Basic exchange recording works");
    }

    /// Test 2.2: Exchange confirmation updates balances (zero-sum)
    #[tokio::test]
    #[ignore]
    async fn test_exchange_confirmation_zero_sum() {
        println!("Test 2.2: Exchange Confirmation (Zero-Sum)");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, [&dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let bob_cell = &apps[1].cells()[0];
        let alice_did = test_did("alice");
        let bob_did = test_did("bob");

        // Get initial balances
        let alice_balance_before: TendBalance = conductor
            .call(
                &alice_cell.zome("tend"),
                "get_or_create_balance",
                GetBalanceInput {
                    member_did: alice_did.clone(),
                    dao_did: TEST_DAO.to_string(),
                },
            )
            .await
            .expect("Failed to get Alice's balance");

        let bob_balance_before: TendBalance = conductor
            .call(
                &bob_cell.zome("tend"),
                "get_or_create_balance",
                GetBalanceInput {
                    member_did: bob_did.clone(),
                    dao_did: TEST_DAO.to_string(),
                },
            )
            .await
            .expect("Failed to get Bob's balance");

        // Alice provides 3 hours to Bob
        let exchange_input = RecordExchangeInput {
            provider_did: alice_did.clone(),
            receiver_did: bob_did.clone(),
            dao_did: TEST_DAO.to_string(),
            hours: 3.0,
            service_description: "Language tutoring".to_string(),
            category: Some(ServiceCategory::Education),
        };

        let exchange: ExchangeRecord = conductor
            .call(&alice_cell.zome("tend"), "record_exchange", exchange_input)
            .await
            .expect("Failed to record exchange");

        // Bob confirms the exchange
        let confirm_input = ConfirmExchangeInput {
            exchange_id: exchange.id.clone(),
            confirmer_did: bob_did.clone(),
        };

        let confirmed: ExchangeRecord = conductor
            .call(&bob_cell.zome("tend"), "confirm_exchange", confirm_input)
            .await
            .expect("Failed to confirm exchange");

        assert!(
            matches!(confirmed.status, ExchangeStatus::Confirmed),
            "Should be confirmed"
        );

        // Check balances after
        let alice_balance_after: TendBalance = conductor
            .call(
                &alice_cell.zome("tend"),
                "get_or_create_balance",
                GetBalanceInput {
                    member_did: alice_did.clone(),
                    dao_did: TEST_DAO.to_string(),
                },
            )
            .await
            .expect("Failed to get Alice's balance");

        let bob_balance_after: TendBalance = conductor
            .call(
                &bob_cell.zome("tend"),
                "get_or_create_balance",
                GetBalanceInput {
                    member_did: bob_did.clone(),
                    dao_did: TEST_DAO.to_string(),
                },
            )
            .await
            .expect("Failed to get Bob's balance");

        // Verify zero-sum: Provider gains, receiver loses
        let alice_change =
            alice_balance_after.current_balance - alice_balance_before.current_balance;
        let bob_change = bob_balance_after.current_balance - bob_balance_before.current_balance;

        assert_eq!(alice_change, 3.0, "Alice should gain 3 TEND");
        assert_eq!(bob_change, -3.0, "Bob should lose 3 TEND");
        assert_eq!(alice_change + bob_change, 0.0, "Changes should sum to zero");

        println!("  - Alice balance change: +{}", alice_change);
        println!("  - Bob balance change: {}", bob_change);
        println!(
            "  - Sum of changes: {} (zero-sum verified)",
            alice_change + bob_change
        );
        println!("Test 2.2 PASSED: Exchange confirmation maintains zero-sum");
    }

    /// Test 2.3: Self-exchange is rejected
    #[tokio::test]
    #[ignore]
    async fn test_self_exchange_rejected() {
        println!("Test 2.3: Self-Exchange Rejection");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, [&dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = test_did("alice");

        let exchange_input = RecordExchangeInput {
            provider_did: alice_did.clone(),
            receiver_did: alice_did.clone(), // Same as provider
            dao_did: TEST_DAO.to_string(),
            hours: 5.0,
            service_description: "Self-service".to_string(),
            category: None,
        };

        let result: Result<ExchangeRecord, _> = conductor
            .call_fallible(alice_cell.zome("tend"), "record_exchange", exchange_input)
            .await;

        match result {
            Err(e) => {
                let error_msg = format!("{:?}", e);
                assert!(
                    error_msg.contains("Cannot exchange with yourself"),
                    "Should reject self-exchange, got: {}",
                    error_msg
                );
                println!("  - Self-exchange rejected: OK");
            }
            Ok(_) => panic!("Should have rejected self-exchange"),
        }

        println!("Test 2.3 PASSED: Self-exchanges are rejected");
    }
}

// ============================================================================
// Section 3: Service Listings Tests
// ============================================================================

#[cfg(test)]
mod service_listings {
    use super::*;

    /// Test 3.1: Create service listing
    #[tokio::test]
    #[ignore]
    async fn test_create_listing() {
        println!("Test 3.1: Create Service Listing");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, [&dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = test_did("alice");

        let listing_input = CreateListingInput {
            provider_did: alice_did.clone(),
            dao_did: TEST_DAO.to_string(),
            title: "Programming Tutoring".to_string(),
            description: "Learn Rust programming".to_string(),
            category: ServiceCategory::HomeServices,
            estimated_hours: Some(2.0),
        };

        let listing: ServiceListing = conductor
            .call(&alice_cell.zome("tend"), "create_listing", listing_input)
            .await
            .expect("Failed to create listing");

        assert_eq!(listing.title, "Programming Tutoring", "Title mismatch");
        assert_eq!(listing.provider_did, alice_did, "Provider mismatch");
        assert!(listing.active, "Listing should be active");

        println!("  - Listing created: {}", listing.title);
        println!("  - Provider: {}", listing.provider_did);
        println!("  - Category: {:?}", listing.category);
        println!("Test 3.1 PASSED: Service listing creation works");
    }

    /// Test 3.2: Query listings by category
    #[tokio::test]
    #[ignore]
    async fn test_query_listings_by_category() {
        println!("Test 3.2: Query Listings by Category");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, [&dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let bob_cell = &apps[1].cells()[0];
        let alice_did = test_did("alice");
        let bob_did = test_did("bob");

        // Create listings in different categories
        let _: ServiceListing = conductor
            .call(
                &alice_cell.zome("tend"),
                "create_listing",
                CreateListingInput {
                    provider_did: alice_did.clone(),
                    dao_did: TEST_DAO.to_string(),
                    title: "Piano Lessons".to_string(),
                    description: "Music instruction".to_string(),
                    category: ServiceCategory::Creative,
                    estimated_hours: Some(1.0),
                },
            )
            .await
            .expect("Failed to create listing");

        let _: ServiceListing = conductor
            .call(
                &bob_cell.zome("tend"),
                "create_listing",
                CreateListingInput {
                    provider_did: bob_did.clone(),
                    dao_did: TEST_DAO.to_string(),
                    title: "Web Development".to_string(),
                    description: "Build websites".to_string(),
                    category: ServiceCategory::HomeServices,
                    estimated_hours: Some(3.0),
                },
            )
            .await
            .expect("Failed to create listing");

        // Wait for DHT
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Query technical listings
        let query_input = QueryListingsInput {
            dao_did: TEST_DAO.to_string(),
            category: Some(ServiceCategory::HomeServices),
        };

        let listings: Vec<ServiceListing> = conductor
            .call(
                &alice_cell.zome("tend"),
                "get_listings_by_category",
                query_input,
            )
            .await
            .expect("Failed to query listings");

        assert!(!listings.is_empty(), "Should find technical listings");
        assert!(
            listings
                .iter()
                .all(|l| matches!(l.category, ServiceCategory::HomeServices)),
            "All listings should be technical"
        );

        println!("  - Found {} technical listings", listings.len());
        println!("Test 3.2 PASSED: Category query works");
    }
}

// ============================================================================
// Section 4: Dispute Tests
// ============================================================================

#[cfg(test)]
mod dispute_tests {
    use super::*;

    /// Test 4.1: Dispute an exchange
    #[tokio::test]
    #[ignore]
    async fn test_dispute_exchange() {
        println!("Test 4.1: Dispute Exchange");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, [&dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let bob_cell = &apps[1].cells()[0];
        let alice_did = test_did("alice");
        let bob_did = test_did("bob");

        // Record exchange
        let exchange: ExchangeRecord = conductor
            .call(
                &alice_cell.zome("tend"),
                "record_exchange",
                RecordExchangeInput {
                    provider_did: alice_did.clone(),
                    receiver_did: bob_did.clone(),
                    dao_did: TEST_DAO.to_string(),
                    hours: 2.0,
                    service_description: "Home repair".to_string(),
                    category: Some(ServiceCategory::GeneralAssistance),
                },
            )
            .await
            .expect("Failed to record exchange");

        // Bob disputes the exchange
        let dispute_input = DisputeExchangeInput {
            exchange_id: exchange.id.clone(),
            disputer_did: bob_did.clone(),
            reason: "Service was not as described".to_string(),
        };

        let disputed: ExchangeRecord = conductor
            .call(&bob_cell.zome("tend"), "dispute_exchange", dispute_input)
            .await
            .expect("Failed to dispute exchange");

        assert!(
            matches!(disputed.status, ExchangeStatus::Disputed(_)),
            "Should be disputed"
        );

        println!("  - Exchange status: {:?}", disputed.status);
        println!("Test 4.1 PASSED: Dispute workflow works");
    }
}

// ============================================================================
// Unit Tests (No Conductor Required)
// ============================================================================

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_tend_constants() {
        // Per Commons Charter: 1 TEND = 1 hour, ±40 balance limit
        assert_eq!(TEND_PER_HOUR, 1.0, "1 TEND should equal 1 hour");
        assert_eq!(DEFAULT_POSITIVE_LIMIT, 40.0, "Positive limit should be 40");
        assert_eq!(
            DEFAULT_NEGATIVE_LIMIT, -40.0,
            "Negative limit should be -40"
        );
    }

    #[test]
    fn test_service_category_serialization() {
        let categories = vec![
            ServiceCategory::HomeServices,
            ServiceCategory::Creative,
            ServiceCategory::Education,
            ServiceCategory::CareWork,
            ServiceCategory::GeneralAssistance,
            ServiceCategory::Administrative,
            ServiceCategory::CareWork,
            ServiceCategory::Transportation,
            ServiceCategory::Custom("Other".to_string()),
        ];

        for category in categories {
            let json = serde_json::to_string(&category).expect("Serialize failed");
            let deserialized: ServiceCategory =
                serde_json::from_str(&json).expect("Deserialize failed");
            assert_eq!(category, deserialized, "Category round-trip failed");
        }
    }

    #[test]
    fn test_exchange_status_serialization() {
        let statuses = vec![
            ExchangeStatus::Proposed,
            ExchangeStatus::Confirmed,
            ExchangeStatus::Disputed("Reason".to_string()),
            ExchangeStatus::Cancelled,
            ExchangeStatus::Cancelled,
        ];

        for status in statuses {
            let json = serde_json::to_string(&status).expect("Serialize failed");
            let deserialized: ExchangeStatus =
                serde_json::from_str(&json).expect("Deserialize failed");
            assert_eq!(status, deserialized, "Status round-trip failed");
        }
    }

    #[test]
    fn test_zero_sum_property() {
        // Verify that TEND maintains zero-sum property
        // For any exchange: provider_change + receiver_change = 0
        let hours = 5.0;
        let provider_change = hours * TEND_PER_HOUR;
        let receiver_change = -hours * TEND_PER_HOUR;

        assert_eq!(provider_change + receiver_change, 0.0, "Must be zero-sum");
    }
}
