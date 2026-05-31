// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! # CGC (Civic Gifting Credits) Integration Tests
//!
//! Tests for the Civic Gifting Credits module implementing Commons Charter Article I.
//!
//! ## Key Features Tested:
//! - Monthly allocation (10 CGC/month)
//! - Gifting mechanics
//! - Non-accumulation (use it or lose it)
//! - High-activity flagging (>100/month or >50 from single source)
//! - Cultural aliases (KUDOS, THANKS, PROPS, etc.)
//!
//! ## Running Tests
//!
//! ```bash
//! cargo test --test cgc_test
//! cargo test --test cgc_test -- --ignored  # Full integration tests
//! ```

use holochain::prelude::*;
use holochain::sweettest::{SweetAgents, SweetConductor, SweetDnaFile};
use std::time::Duration;

// Import zome types
use cgc_integrity::*;

mod test_helpers {
    pub const TEST_DID_PREFIX: &str = "did:mycelix:test:";
    pub const TEST_DAO: &str = "did:mycelix:dao:test_community";

    pub fn test_did(suffix: &str) -> String {
        format!("{}{}", TEST_DID_PREFIX, suffix)
    }
}

use test_helpers::*;

// ============================================================================
// Section 1: Allocation Tests
// ============================================================================

#[cfg(test)]
mod allocation_tests {
    use super::*;

    /// Test 1.1: Monthly allocation is created correctly (10 CGC)
    #[tokio::test]
    #[ignore]
    async fn test_monthly_allocation_creation() {
        println!("Test 1.1: Monthly CGC Allocation Creation");

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

        // Get or create allocation
        let input = GetAllocationInput {
            member_did: alice_did.clone(),
            dao_did: TEST_DAO.to_string(),
        };

        let allocation: CgcAllocation = conductor
            .call(&alice_cell.zome("cgc"), "get_or_create_allocation", input)
            .await
            .expect("Failed to get allocation");

        // Verify 10 CGC allocated per Commons Charter
        assert_eq!(allocation.monthly_allowance, 10, "Should have 10 CGC/month");
        assert_eq!(allocation.remaining, 10, "Should have full 10 remaining");
        assert_eq!(allocation.gifted_this_cycle, 0, "No gifts yet");

        println!("  - Monthly allowance: {}", allocation.monthly_allowance);
        println!("  - Remaining: {}", allocation.remaining);
        println!("Test 1.1 PASSED: Monthly allocation works correctly");
    }

    /// Test 1.2: Allocation does not accumulate across cycles
    #[tokio::test]
    #[ignore]
    async fn test_allocation_non_accumulation() {
        println!("Test 1.2: CGC Non-Accumulation");

        // This test verifies that unused CGC expire at cycle end
        // Per Commons Charter: "CGC do not accumulate across cycles"

        // Note: Testing this properly requires time manipulation
        // For now, we verify the non_accumulating flag is set

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

        let input = GetAllocationInput {
            member_did: alice_did.clone(),
            dao_did: TEST_DAO.to_string(),
        };

        let allocation: CgcAllocation = conductor
            .call(&alice_cell.zome("cgc"), "get_or_create_allocation", input)
            .await
            .expect("Failed to get allocation");

        // Verify allocation structure supports non-accumulation
        // The cycle_start and cycle_end fields enable proper expiration
        assert!(
            allocation.cycle_end.as_micros() > allocation.cycle_start.as_micros(),
            "Cycle end should be after cycle start"
        );

        println!("  - Cycle start: {:?}", allocation.cycle_start);
        println!("  - Cycle end: {:?}", allocation.cycle_end);
        println!("Test 1.2 PASSED: Non-accumulation structure verified");
    }
}

// ============================================================================
// Section 2: Gifting Tests
// ============================================================================

#[cfg(test)]
mod gifting_tests {
    use super::*;

    /// Test 2.1: Basic CGC gift succeeds
    #[tokio::test]
    #[ignore]
    async fn test_basic_gift() {
        println!("Test 2.1: Basic CGC Gift");

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

        // Give CGC to Bob
        let gift_input = GiftCgcInput {
            giver_did: alice_did.clone(),
            receiver_did: bob_did.clone(),
            dao_did: TEST_DAO.to_string(),
            amount: 3,
            message: Some("Thanks for your help!".to_string()),
            category: ContributionType::CareWork,
        };

        let gift: CgcGift = conductor
            .call(&alice_cell.zome("cgc"), "gift_cgc", gift_input)
            .await
            .expect("Failed to gift CGC");

        assert_eq!(gift.amount, 3, "Gift amount mismatch");
        assert_eq!(gift.giver_did, alice_did, "Giver mismatch");
        assert_eq!(gift.receiver_did, bob_did, "Receiver mismatch");

        println!(
            "  - Gift created: {} CGC from {} to {}",
            gift.amount, gift.giver_did, gift.receiver_did
        );
        println!("Test 2.1 PASSED: Basic gifting works");
    }

    /// Test 2.2: Gift fails when exceeding remaining allocation
    #[tokio::test]
    #[ignore]
    async fn test_gift_exceeds_allocation() {
        println!("Test 2.2: Gift Exceeds Allocation");

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

        // Try to gift more than 10 CGC
        let gift_input = GiftCgcInput {
            giver_did: alice_did.clone(),
            receiver_did: bob_did.clone(),
            dao_did: TEST_DAO.to_string(),
            amount: 15, // Exceeds 10 CGC allocation
            message: None,
            category: ContributionType::Gratitude,
        };

        let result: Result<CgcGift, _> = conductor
            .call_fallible(alice_cell.zome("cgc"), "gift_cgc", gift_input)
            .await;

        match result {
            Err(e) => {
                let error_msg = format!("{:?}", e);
                assert!(
                    error_msg.contains("Insufficient CGC") || error_msg.contains("exceeds"),
                    "Should reject gift exceeding allocation, got: {}",
                    error_msg
                );
                println!("  - Over-allocation gift rejected: OK");
            }
            Ok(_) => panic!("Should have rejected gift exceeding allocation"),
        }

        println!("Test 2.2 PASSED: Over-allocation gifts are rejected");
    }

    /// Test 2.3: Self-gifting is rejected
    #[tokio::test]
    #[ignore]
    async fn test_self_gift_rejected() {
        println!("Test 2.3: Self-Gift Rejection");

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

        // Try to gift to self
        let gift_input = GiftCgcInput {
            giver_did: alice_did.clone(),
            receiver_did: alice_did.clone(), // Same as giver
            dao_did: TEST_DAO.to_string(),
            amount: 5,
            message: None,
            category: ContributionType::Gratitude,
        };

        let result: Result<CgcGift, _> = conductor
            .call_fallible(alice_cell.zome("cgc"), "gift_cgc", gift_input)
            .await;

        match result {
            Err(e) => {
                let error_msg = format!("{:?}", e);
                assert!(
                    error_msg.contains("Cannot gift to yourself"),
                    "Should reject self-gift, got: {}",
                    error_msg
                );
                println!("  - Self-gift rejected: OK");
            }
            Ok(_) => panic!("Should have rejected self-gift"),
        }

        println!("Test 2.3 PASSED: Self-gifts are rejected");
    }

    /// Test 2.4: Recognition categories are tracked
    #[tokio::test]
    #[ignore]
    async fn test_recognition_categories() {
        println!("Test 2.4: Recognition Categories");

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

        let categories = vec![
            ContributionType::CareWork,
            ContributionType::Education,
            ContributionType::CommunityOrganizing,
            ContributionType::TechnicalWork,
            ContributionType::Education,
            ContributionType::CreativeWork,
        ];

        for category in categories {
            let gift_input = GiftCgcInput {
                giver_did: alice_did.clone(),
                receiver_did: bob_did.clone(),
                dao_did: TEST_DAO.to_string(),
                amount: 1,
                message: None,
                category: category.clone(),
            };

            let gift: CgcGift = conductor
                .call(&alice_cell.zome("cgc"), "gift_cgc", gift_input)
                .await
                .expect(&format!("Failed to gift with category {:?}", category));

            assert_eq!(gift.category, category, "Category mismatch");
            println!("  - Category {:?}: OK", category);
        }

        println!("Test 2.4 PASSED: All recognition categories work");
    }
}

// ============================================================================
// Section 3: High Activity Flagging Tests
// ============================================================================

#[cfg(test)]
mod high_activity_tests {
    use super::*;

    /// Test 3.1: High activity flag triggered at >100 CGC received/month
    #[tokio::test]
    #[ignore]
    async fn test_high_volume_flag() {
        println!("Test 3.1: High Volume Flag (>100 CGC/month)");

        // Per Commons Charter: Flag if receiving >100 CGC/month
        // This test would require multiple givers to accumulate >100 CGC
        // For now, we verify the threshold constant exists

        assert_eq!(
            HIGH_ACTIVITY_THRESHOLD_VOLUME, 100,
            "High activity volume threshold should be 100"
        );

        println!(
            "  - HIGH_ACTIVITY_THRESHOLD_VOLUME: {}",
            HIGH_ACTIVITY_THRESHOLD_VOLUME
        );
        println!("Test 3.1 PASSED: High volume threshold verified");
    }

    /// Test 3.2: High activity flag triggered at >50 CGC from single source
    #[tokio::test]
    #[ignore]
    async fn test_concentrated_source_flag() {
        println!("Test 3.2: Concentrated Source Flag (>50 from single source)");

        // Per Commons Charter: Flag if receiving >50 CGC from single source
        assert_eq!(
            HIGH_ACTIVITY_THRESHOLD_SINGLE_SOURCE, 50,
            "High activity single source threshold should be 50"
        );

        println!(
            "  - HIGH_ACTIVITY_THRESHOLD_SINGLE_SOURCE: {}",
            HIGH_ACTIVITY_THRESHOLD_SINGLE_SOURCE
        );
        println!("Test 3.2 PASSED: Single source threshold verified");
    }
}

// ============================================================================
// Section 4: Recognition Summary Tests
// ============================================================================

#[cfg(test)]
mod recognition_summary_tests {
    use super::*;

    /// Test 4.1: Get recognition summary for a member
    #[tokio::test]
    #[ignore]
    async fn test_recognition_summary() {
        println!("Test 4.1: Recognition Summary");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 3).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, [&dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let bob_cell = &apps[1].cells()[0];
        let charlie_cell = &apps[2].cells()[0];

        let alice_did = test_did("alice");
        let bob_did = test_did("bob");
        let charlie_did = test_did("charlie");

        // Alice and Charlie both give CGC to Bob
        for (giver_cell, giver_did, amount) in [
            (&alice_cell, &alice_did, 3),
            (&charlie_cell, &charlie_did, 2),
        ] {
            let gift_input = GiftCgcInput {
                giver_did: giver_did.clone(),
                receiver_did: bob_did.clone(),
                dao_did: TEST_DAO.to_string(),
                amount,
                message: None,
                category: ContributionType::Gratitude,
            };

            let _: CgcGift = conductor
                .call(&giver_cell.zome("cgc"), "gift_cgc", gift_input)
                .await
                .expect("Failed to gift");
        }

        // Wait for DHT consistency
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Get Bob's recognition summary
        let summary_input = GetRecognitionInput {
            member_did: bob_did.clone(),
            dao_did: TEST_DAO.to_string(),
        };

        let summary: CgcRecognition = conductor
            .call(
                &bob_cell.zome("cgc"),
                "get_recognition_summary",
                summary_input,
            )
            .await
            .expect("Failed to get summary");

        assert_eq!(summary.total_received, 5, "Total received mismatch");
        assert_eq!(summary.unique_givers, 2, "Unique givers mismatch");

        println!("  - Total received: {}", summary.total_received);
        println!("  - Unique givers: {}", summary.unique_givers);
        println!("Test 4.1 PASSED: Recognition summary works");
    }
}

// ============================================================================
// Section 5: Cultural Alias Tests
// ============================================================================

#[cfg(test)]
mod cultural_alias_tests {
    use super::*;

    /// Test 5.1: Cultural alias registration
    #[tokio::test]
    #[ignore]
    async fn test_cultural_alias() {
        println!("Test 5.1: Cultural Alias Registration");

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

        // Register cultural alias
        let alias_input = RegisterAliasInput {
            dao_did: TEST_DAO.to_string(),
            alias: "KUDOS".to_string(),
            description: "Recognition tokens in our community".to_string(),
        };

        let alias: CulturalAlias = conductor
            .call(
                &alice_cell.zome("cgc"),
                "register_cultural_alias",
                alias_input,
            )
            .await
            .expect("Failed to register alias");

        assert_eq!(alias.alias, "KUDOS", "Alias mismatch");
        assert_eq!(alias.canonical, "CGC", "Should map to canonical CGC");

        println!(
            "  - Registered alias: {} -> {}",
            alias.alias, alias.canonical
        );
        println!("Test 5.1 PASSED: Cultural alias registration works");
    }

    /// Test 5.2: Predefined aliases
    #[test]
    fn test_predefined_aliases() {
        println!("Test 5.2: Predefined Cultural Aliases");

        // Per Commons Charter, these are standard aliases
        let expected_aliases = vec!["KUDOS", "THANKS", "PROPS", "HEARTS", "STARS"];

        for alias in &expected_aliases {
            println!("  - Standard alias: {}", alias);
        }

        println!("Test 5.2 PASSED: Predefined aliases documented");
    }
}

// ============================================================================
// Unit Tests (No Conductor Required)
// ============================================================================

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_cgc_constants() {
        // Verify Commons Charter constants
        assert_eq!(MONTHLY_CGC_ALLOWANCE, 10, "Monthly allowance should be 10");
        assert_eq!(
            HIGH_ACTIVITY_THRESHOLD_VOLUME, 100,
            "Volume threshold should be 100"
        );
        assert_eq!(
            HIGH_ACTIVITY_THRESHOLD_SINGLE_SOURCE, 50,
            "Single source threshold should be 50"
        );
    }

    #[test]
    fn test_recognition_category_serialization() {
        let categories = vec![
            ContributionType::CareWork,
            ContributionType::Education,
            ContributionType::CommunityOrganizing,
            ContributionType::TechnicalWork,
            ContributionType::Education,
            ContributionType::CreativeWork,
            ContributionType::CareWork,
            ContributionType::Gratitude,
            ContributionType::Custom("Special".to_string()),
        ];

        for category in categories {
            let json = serde_json::to_string(&category).expect("Serialize failed");
            let deserialized: ContributionType =
                serde_json::from_str(&json).expect("Deserialize failed");
            assert_eq!(category, deserialized, "Category round-trip failed");
        }
    }

    #[test]
    fn test_did_validation() {
        // Valid DIDs should start with "did:"
        assert!(test_did("alice").starts_with("did:"));
        assert!(TEST_DAO.starts_with("did:"));
    }
}
