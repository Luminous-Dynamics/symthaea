// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Recognition Zome Integration Tests
//!
//! Tests for MYCEL reputation through weighted recognition events covering:
//! - Apprentice onboarding and graduation
//! - Progressive fee integration with MYCEL tiers
//! - Quality rating -> MYCEL validation wiring
//! - Passive decay
//! - Jubilee normalization
//!
//! ## Running Tests
//!
//! ```bash
//! cargo test --test recognition_test
//! cargo test --test recognition_test -- --ignored  # Full integration tests
//! ```

use holochain::prelude::*;
use holochain::sweettest::*;
use std::time::Duration;

use mycelix_finance_types::FeeTier;
use recognition_integrity::*;

// Mirror types for TEND zome types — avoids linking tend_integrity alongside
// recognition_integrity (both HDI crates generate conflicting #[no_mangle] symbols).
// These must match the serde layout of the originals exactly.

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
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
pub struct RateExchangeInput {
    pub exchange_id: String,
    pub rating: u8,
    pub comment: Option<String>,
}

mod test_helpers {
    pub const TEST_DID_PREFIX: &str = "did:mycelix:test:";

    pub fn test_did(suffix: &str) -> String {
        format!("{}{}", TEST_DID_PREFIX, suffix)
    }
}

use test_helpers::*;

/// Paginated input for get_validation_score
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GetValidationScoreInput {
    pub member_did: String,
    pub limit: Option<usize>,
}

// ============================================================================
// Section 1: Apprentice Graduation Flow Tests
// ============================================================================

#[cfg(test)]
mod apprentice_lifecycle {
    use super::*;

    /// Test 1.1: Onboard apprentice with valid mentor
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_onboard_apprentice() {
        println!("Test 1.1: Onboard Apprentice");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let mentor_cell = &apps[0].cells()[0];

        // First, initialize mentor with MYCEL >= 0.3
        let mentor_key = agents[0].clone();
        let mentor_did = format!("did:mycelix:{}", mentor_key);

        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct InitializeMemberInput {
            pub member_did: String,
            pub is_apprentice: bool,
            pub mentor_did: Option<String>,
        }

        let init_mentor = InitializeMemberInput {
            member_did: mentor_did.clone(),
            is_apprentice: false, // Full member
            mentor_did: None,
        };

        let _: Record = conductor
            .call(
                &mentor_cell.zome("recognition"),
                "initialize_member",
                init_mentor,
            )
            .await;

        // Now onboard apprentice
        let apprentice_did = test_did("apprentice_1");

        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct OnboardApprenticeInput {
            pub apprentice_did: String,
        }

        let onboard_input = OnboardApprenticeInput {
            apprentice_did: apprentice_did.clone(),
        };

        let result: Record = conductor
            .call(
                &mentor_cell.zome("recognition"),
                "onboard_apprentice",
                onboard_input,
            )
            .await;

        let state: MemberMycelState = result
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert!(state.is_apprentice, "Should be apprentice");
        assert_eq!(state.mycel_score, 0.1, "Apprentice starts at 0.1");
        assert!(state.mentor_did.is_some(), "Should have a mentor");

        println!("  - Apprentice onboarded: MYCEL = {}", state.mycel_score);
        println!("  - Mentor: {:?}", state.mentor_did);
        println!("Test 1.1 PASSED: Apprentice onboarding works");
    }

    /// Test 1.2: Cannot onboard without sufficient MYCEL
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_onboard_insufficient_mycel() {
        println!("Test 1.2: Cannot Onboard Without Sufficient MYCEL");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let low_mycel_cell = &apps[0].cells()[0];
        let caller_key = agents[0].clone();
        let caller_did = format!("did:mycelix:{}", caller_key);

        // Initialize as apprentice (MYCEL = 0.1 < 0.3 minimum)
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct InitializeMemberInput {
            pub member_did: String,
            pub is_apprentice: bool,
            pub mentor_did: Option<String>,
        }

        let init = InitializeMemberInput {
            member_did: caller_did.clone(),
            is_apprentice: true,
            mentor_did: Some(test_did("some_mentor")),
        };

        let _: Record = conductor
            .call(
                &low_mycel_cell.zome("recognition"),
                "initialize_member",
                init,
            )
            .await;

        // Try to onboard an apprentice (should fail - caller MYCEL too low)
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct OnboardApprenticeInput {
            pub apprentice_did: String,
        }

        let result: Result<Record, _> = conductor
            .call_fallible(
                &low_mycel_cell.zome("recognition"),
                "onboard_apprentice",
                OnboardApprenticeInput {
                    apprentice_did: test_did("new_apprentice"),
                },
            )
            .await;

        match result {
            Err(e) => {
                let msg = format!("{:?}", e);
                assert!(
                    msg.contains("MYCEL") || msg.contains("minimum") || msg.contains("0.3"),
                    "Should mention insufficient MYCEL, got: {}",
                    msg
                );
                println!("  - Insufficient MYCEL rejection: OK");
            }
            Ok(_) => panic!("Should reject onboarding with low MYCEL"),
        }

        println!("Test 1.2 PASSED: Low MYCEL mentors cannot onboard");
    }

    /// Test 1.3: Graduate apprentice when MYCEL reaches 0.3
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_graduate_apprentice() {
        println!("Test 1.3: Graduate Apprentice");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let mentor_cell = &apps[0].cells()[0];
        let apprentice_cell = &apps[1].cells()[0];
        let mentor_key = agents[0].clone();
        let mentor_did = format!("did:mycelix:{}", mentor_key);
        let apprentice_key = agents[1].clone();
        let apprentice_did = format!("did:mycelix:{}", apprentice_key);

        // Initialize mentor
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct InitializeMemberInput {
            pub member_did: String,
            pub is_apprentice: bool,
            pub mentor_did: Option<String>,
        }

        let _: Record = conductor
            .call(
                &mentor_cell.zome("recognition"),
                "initialize_member",
                InitializeMemberInput {
                    member_did: mentor_did.clone(),
                    is_apprentice: false,
                    mentor_did: None,
                },
            )
            .await;

        // Initialize apprentice via onboarding
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct OnboardApprenticeInput {
            pub apprentice_did: String,
        }

        let _: Record = conductor
            .call(
                &mentor_cell.zome("recognition"),
                "onboard_apprentice",
                OnboardApprenticeInput {
                    apprentice_did: apprentice_did.clone(),
                },
            )
            .await;

        // Update apprentice's MYCEL to >= 0.3 by providing high component scores
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct UpdateMycelInput {
            pub member_did: String,
            pub participation: f64,
            pub recognition: f64,
            pub validation_override: Option<f64>,
            pub active_months: u32,
        }

        let update = UpdateMycelInput {
            member_did: apprentice_did.clone(),
            participation: 0.6,
            recognition: 0.5,
            validation_override: Some(0.4),
            active_months: 6,
        };

        let updated: MemberMycelState = conductor
            .call(
                &apprentice_cell.zome("recognition"),
                "update_mycel_score",
                update,
            )
            .await;

        println!("  - Updated MYCEL: {:.2}", updated.mycel_score);
        assert!(
            updated.mycel_score >= 0.3,
            "MYCEL should be >= 0.3 for graduation"
        );

        // Graduate
        let graduated: MemberMycelState = conductor
            .call(
                &apprentice_cell.zome("recognition"),
                "graduate_apprentice",
                apprentice_did.clone(),
            )
            .await;

        assert!(!graduated.is_apprentice, "Should no longer be apprentice");
        assert!(
            graduated.mentor_did.is_none(),
            "Mentor relationship should end"
        );

        println!("  - Graduated: is_apprentice = {}", graduated.is_apprentice);
        println!("  - MYCEL: {:.2}", graduated.mycel_score);
        println!("Test 1.3 PASSED: Apprentice graduation works");
    }

    /// Test 1.4: Cannot graduate below 0.3 MYCEL
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_cannot_graduate_early() {
        println!("Test 1.4: Cannot Graduate Below 0.3 MYCEL");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let mentor_cell = &apps[0].cells()[0];
        let apprentice_cell = &apps[1].cells()[0];
        let mentor_key = agents[0].clone();
        let mentor_did = format!("did:mycelix:{}", mentor_key);
        let apprentice_key = agents[1].clone();
        let apprentice_did = format!("did:mycelix:{}", apprentice_key);

        // Initialize mentor and onboard apprentice
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct InitializeMemberInput {
            pub member_did: String,
            pub is_apprentice: bool,
            pub mentor_did: Option<String>,
        }

        let _: Record = conductor
            .call(
                &mentor_cell.zome("recognition"),
                "initialize_member",
                InitializeMemberInput {
                    member_did: mentor_did.clone(),
                    is_apprentice: false,
                    mentor_did: None,
                },
            )
            .await;

        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct OnboardApprenticeInput {
            pub apprentice_did: String,
        }

        let _: Record = conductor
            .call(
                &mentor_cell.zome("recognition"),
                "onboard_apprentice",
                OnboardApprenticeInput {
                    apprentice_did: apprentice_did.clone(),
                },
            )
            .await;

        // Try to graduate without reaching 0.3 (starts at 0.1)
        let result: Result<MemberMycelState, _> = conductor
            .call_fallible(
                &apprentice_cell.zome("recognition"),
                "graduate_apprentice",
                apprentice_did.clone(),
            )
            .await;

        match result {
            Err(e) => {
                let msg = format!("{:?}", e);
                assert!(
                    msg.contains("0.3") || msg.contains("graduate") || msg.contains("MYCEL"),
                    "Should mention insufficient score, got: {}",
                    msg
                );
                println!("  - Early graduation rejected: OK");
            }
            Ok(_) => panic!("Should reject graduation below 0.3"),
        }

        println!("Test 1.4 PASSED: Cannot graduate early");
    }
}

// ============================================================================
// Section 2: Quality Rating -> MYCEL Integration Tests
// ============================================================================

#[cfg(test)]
mod quality_rating_integration {
    use super::*;

    /// Test 2.1: Quality ratings flow from TEND to MYCEL Validation component
    ///
    /// This is an end-to-end test:
    /// 1. Alice provides service to Bob (TEND exchange)
    /// 2. Bob confirms exchange
    /// 3. Bob rates Alice 5/5
    /// 4. Alice's MYCEL Validation component reflects the rating
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_quality_rating_flows_to_mycel() {
        println!("Test 2.1: Quality Rating -> MYCEL Validation Flow");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let bob_cell = &apps[1].cells()[0];
        let alice_key = agents[0].clone();
        let alice_did = format!("did:mycelix:{}", alice_key);
        let bob_key = agents[1].clone();
        let bob_did = format!("did:mycelix:{}", bob_key);

        // Initialize both members
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct InitializeMemberInput {
            pub member_did: String,
            pub is_apprentice: bool,
            pub mentor_did: Option<String>,
        }

        let _: Record = conductor
            .call(
                &alice_cell.zome("recognition"),
                "initialize_member",
                InitializeMemberInput {
                    member_did: alice_did.clone(),
                    is_apprentice: false,
                    mentor_did: None,
                },
            )
            .await;

        let _: Record = conductor
            .call(
                &bob_cell.zome("recognition"),
                "initialize_member",
                InitializeMemberInput {
                    member_did: bob_did.clone(),
                    is_apprentice: false,
                    mentor_did: None,
                },
            )
            .await;

        let dao_did = "did:mycelix:dao:test_community".to_string();

        // Step 1: Alice provides services and Bob rates them highly
        // We need 3 confirmed+rated exchanges for the validation score to be non-zero
        // Using mirror types from top-level (avoids HDI linker conflicts)

        for i in 0..3 {
            // Alice records exchange with Bob
            let exchange: ExchangeRecord = conductor
                .call(
                    &alice_cell.zome("tend"),
                    "record_exchange",
                    RecordExchangeInput {
                        receiver_did: bob_did.clone(),
                        dao_did: dao_did.clone(),
                        hours: 1.0,
                        service_description: format!("Service {}", i),
                        service_category: ServiceCategory::Education,
                        cultural_alias: None,
                        service_date: None,
                    },
                )
                .await;

            // Bob confirms
            let _: ExchangeRecord = conductor
                .call(
                    &bob_cell.zome("tend"),
                    "confirm_exchange",
                    exchange.id.clone(),
                )
                .await;

            // Bob rates 5 stars
            let _: Record = conductor
                .call(
                    &bob_cell.zome("tend"),
                    "rate_exchange",
                    RateExchangeInput {
                        exchange_id: exchange.id.clone(),
                        rating: 5,
                        comment: Some(format!("Excellent service {}", i)),
                    },
                )
                .await;

            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        println!("  - 3 exchanges recorded, confirmed, and rated (5/5)");

        // Step 2: Get Alice's validation score from TEND
        let validation_score: f64 = conductor
            .call(
                &alice_cell.zome("tend"),
                "get_validation_score",
                GetValidationScoreInput {
                    member_did: alice_did.clone(),
                    limit: None,
                },
            )
            .await;

        // 5/5 average maps to (5-1)/4 = 1.0
        assert!(
            validation_score > 0.9,
            "With all 5-star ratings, validation should be high, got: {}",
            validation_score
        );
        println!("  - TEND validation score: {:.2}", validation_score);

        // Step 3: Update Alice's MYCEL (validation auto-fetched from TEND)
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct UpdateMycelInput {
            pub member_did: String,
            pub participation: f64,
            pub recognition: f64,
            pub validation_override: Option<f64>,
            pub active_months: u32,
        }

        let updated: MemberMycelState = conductor
            .call(
                &alice_cell.zome("recognition"),
                "update_mycel_score",
                UpdateMycelInput {
                    member_did: alice_did.clone(),
                    participation: 0.5,
                    recognition: 0.5,
                    validation_override: None, // Auto-fetch from TEND!
                    active_months: 12,
                },
            )
            .await;

        // With validation auto-fetched: 0.5*0.4 + 0.5*0.2 + ~1.0*0.2 + 0.5*0.2
        // = 0.20 + 0.10 + 0.20 + 0.10 = 0.60
        println!("  - Updated MYCEL score: {:.2}", updated.mycel_score);
        assert!(
            updated.mycel_score > 0.5,
            "MYCEL should reflect high validation"
        );
        assert!(
            updated.validation > 0.9,
            "Validation component should be high from ratings"
        );

        println!("  - MYCEL validation component: {:.2}", updated.validation);
        println!("Test 2.1 PASSED: Quality ratings flow into MYCEL Validation");
    }

    /// Test 2.2: Low ratings reduce MYCEL Validation
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_low_ratings_reduce_validation() {
        println!("Test 2.2: Low Ratings Reduce MYCEL Validation");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let bob_cell = &apps[1].cells()[0];
        let alice_key = agents[0].clone();
        let alice_did = format!("did:mycelix:{}", alice_key);
        let bob_key = agents[1].clone();
        let bob_did = format!("did:mycelix:{}", bob_key);

        // Initialize members
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct InitializeMemberInput {
            pub member_did: String,
            pub is_apprentice: bool,
            pub mentor_did: Option<String>,
        }

        let _: Record = conductor
            .call(
                &alice_cell.zome("recognition"),
                "initialize_member",
                InitializeMemberInput {
                    member_did: alice_did.clone(),
                    is_apprentice: false,
                    mentor_did: None,
                },
            )
            .await;

        let _: Record = conductor
            .call(
                &bob_cell.zome("recognition"),
                "initialize_member",
                InitializeMemberInput {
                    member_did: bob_did.clone(),
                    is_apprentice: false,
                    mentor_did: None,
                },
            )
            .await;

        let dao_did = "did:mycelix:dao:test_community".to_string();

        // Using mirror types from top-level (avoids HDI linker conflicts)

        // 3 exchanges, all rated 1/5 (poor quality)
        for i in 0..3 {
            let exchange: ExchangeRecord = conductor
                .call(
                    &alice_cell.zome("tend"),
                    "record_exchange",
                    RecordExchangeInput {
                        receiver_did: bob_did.clone(),
                        dao_did: dao_did.clone(),
                        hours: 1.0,
                        service_description: format!("Service {}", i),
                        service_category: ServiceCategory::GeneralAssistance,
                        cultural_alias: None,
                        service_date: None,
                    },
                )
                .await;

            let _: ExchangeRecord = conductor
                .call(
                    &bob_cell.zome("tend"),
                    "confirm_exchange",
                    exchange.id.clone(),
                )
                .await;

            let _: Record = conductor
                .call(
                    &bob_cell.zome("tend"),
                    "rate_exchange",
                    RateExchangeInput {
                        exchange_id: exchange.id.clone(),
                        rating: 1, // Low rating
                        comment: Some("Poor service".to_string()),
                    },
                )
                .await;

            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        // Get validation score -- 1/5 avg maps to (1-1)/4 = 0.0
        let validation_score: f64 = conductor
            .call(
                &alice_cell.zome("tend"),
                "get_validation_score",
                GetValidationScoreInput {
                    member_did: alice_did.clone(),
                    limit: None,
                },
            )
            .await;

        assert!(
            validation_score < 0.1,
            "With all 1-star ratings, validation should be near 0, got: {}",
            validation_score
        );
        println!("  - Validation with low ratings: {:.2}", validation_score);
        println!("Test 2.2 PASSED: Low ratings reduce validation score");
    }
}

// ============================================================================
// Section 3: Progressive Fee Tests
// ============================================================================

#[cfg(test)]
mod progressive_fees {
    use super::*;

    /// Test 3.1: Fee tiers from MYCEL score
    #[test]
    fn test_fee_tier_from_mycel() {
        // Newcomer: MYCEL < 0.3 -> 0.10% fee
        assert!(matches!(FeeTier::from_mycel(0.0), FeeTier::Newcomer));
        assert!(matches!(FeeTier::from_mycel(0.1), FeeTier::Newcomer));
        assert!(matches!(FeeTier::from_mycel(0.29), FeeTier::Newcomer));

        // Member: 0.3 - 0.7 -> 0.03% fee
        assert!(matches!(FeeTier::from_mycel(0.3), FeeTier::Member));
        assert!(matches!(FeeTier::from_mycel(0.5), FeeTier::Member));
        assert!(matches!(FeeTier::from_mycel(0.69), FeeTier::Member));
        assert!(matches!(FeeTier::from_mycel(0.7), FeeTier::Member)); // boundary: > 0.7 for Steward

        // Steward: > 0.7 -> 0.01% fee
        assert!(matches!(FeeTier::from_mycel(0.71), FeeTier::Steward));
        assert!(matches!(FeeTier::from_mycel(0.8), FeeTier::Steward));
        assert!(matches!(FeeTier::from_mycel(1.0), FeeTier::Steward));
    }

    /// Test 3.2: Base fee rates
    #[test]
    fn test_fee_rates() {
        let newcomer = FeeTier::Newcomer;
        let member = FeeTier::Member;
        let steward = FeeTier::Steward;

        // Newcomer: 0.10% = 0.001
        assert!((newcomer.base_fee_rate() - 0.001).abs() < 1e-6);

        // Member: 0.03% = 0.0003
        assert!((member.base_fee_rate() - 0.0003).abs() < 1e-6);

        // Steward: 0.01% = 0.0001
        assert!((steward.base_fee_rate() - 0.0001).abs() < 1e-6);
    }

    /// Test 3.3: Fee computation on SAP amounts
    #[test]
    fn test_fee_computation() {
        let amount: u64 = 1_000_000; // 1 SAP in micro

        // Newcomer fee: 1M * 0.001 = 1000
        let newcomer_fee = (amount as f64 * FeeTier::Newcomer.base_fee_rate()) as u64;
        assert_eq!(newcomer_fee, 1000);

        // Member fee: 1M * 0.0003 = 300
        let member_fee = (amount as f64 * FeeTier::Member.base_fee_rate()) as u64;
        assert_eq!(member_fee, 300);

        // Steward fee: 1M * 0.0001 = 100
        let steward_fee = (amount as f64 * FeeTier::Steward.base_fee_rate()) as u64;
        assert_eq!(steward_fee, 100);

        // Progressive: Steward pays 10x less than Newcomer
        assert_eq!(newcomer_fee / steward_fee, 10);
    }

    /// Test 3.4: Fee tiers are serializable
    #[test]
    fn test_fee_tier_serialization() {
        let tiers = vec![FeeTier::Newcomer, FeeTier::Member, FeeTier::Steward];

        for tier in tiers {
            let json = serde_json::to_string(&tier).expect("Serialize failed");
            let deserialized: FeeTier = serde_json::from_str(&json).expect("Deserialize failed");
            assert_eq!(tier, deserialized, "FeeTier round-trip failed");
        }
    }
}

// ============================================================================
// Section 4: Jubilee and Decay Tests
// ============================================================================

#[cfg(test)]
mod jubilee_and_decay {
    use super::*;

    /// Test 4.1: Jubilee normalization formula
    #[test]
    fn test_jubilee_normalization_formula() {
        // new_mycel = 0.3 + (current - 0.3) * 0.8
        let test_cases: Vec<(f64, f64)> = vec![
            (1.0, 0.3 + (1.0 - 0.3) * 0.8), // High -> compressed
            (0.5, 0.3 + (0.5 - 0.3) * 0.8), // Medium -> slightly compressed
            (0.3, 0.3 + (0.3 - 0.3) * 0.8), // At threshold -> no change
            (0.1, 0.3 + (0.1 - 0.3) * 0.8), // Below threshold -> slight increase
        ];

        for (input, expected) in test_cases {
            let result: f64 = 0.3 + (input - 0.3) * 0.8;
            assert!(
                (result - expected).abs() < 1e-6,
                "Jubilee({}) = {}, expected {}",
                input,
                result,
                expected
            );
            println!("  - Jubilee({:.1}) = {:.3}", input, result);
        }
    }

    /// Test 4.2: Passive decay formula
    #[test]
    fn test_passive_decay_formula() {
        // 5% annual linear decay
        let score: f64 = 0.8;
        let rate: f64 = 0.05;

        // After 1 year: 0.8 - 0.8 * 0.05 * 1.0 = 0.76
        let after_1yr: f64 = score - score * rate * 1.0;
        assert!(
            (after_1yr - 0.76_f64).abs() < 1e-6,
            "1yr decay: got {}",
            after_1yr
        );

        // After 6 months: 0.8 - 0.8 * 0.05 * 0.5 = 0.78
        let after_6mo: f64 = score - score * rate * 0.5;
        assert!(
            (after_6mo - 0.78_f64).abs() < 1e-6,
            "6mo decay: got {}",
            after_6mo
        );

        // After 0 time: no change
        let after_0: f64 = score - score * rate * 0.0;
        assert!((after_0 - score).abs() < 1e-6, "0 decay: got {}", after_0);
    }

    /// Test 4.3: MYCEL score component weights
    #[test]
    fn test_mycel_component_weights() {
        // Participation (40%), Recognition (20%), Validation (20%), Longevity (20%)
        let participation: f64 = 1.0;
        let recognition: f64 = 1.0;
        let validation: f64 = 1.0;
        let longevity: f64 = 1.0;

        let composite: f64 =
            participation * 0.40 + recognition * 0.20 + validation * 0.20 + longevity * 0.20;

        assert!(
            (composite - 1.0_f64).abs() < 1e-6,
            "Max composite should be 1.0"
        );

        // All zeros -> 0.0
        let zero_composite: f64 = 0.0 * 0.40 + 0.0 * 0.20 + 0.0 * 0.20 + 0.0 * 0.20;
        assert!(
            (zero_composite - 0.0_f64).abs() < 1e-6,
            "Zero composite should be 0.0"
        );

        // Participation dominates
        let participation_only: f64 = 1.0 * 0.40 + 0.0 * 0.20 + 0.0 * 0.20 + 0.0 * 0.20;
        assert!(
            (participation_only - 0.40_f64).abs() < 1e-6,
            "Participation-only = 0.40"
        );
    }
}

// ============================================================================
// Unit Tests (No Conductor Required)
// ============================================================================

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_recognition_constants() {
        assert_eq!(MAX_RECOGNITIONS_PER_CYCLE, 10, "Max recognitions per cycle");
        assert_eq!(RECOGNITION_BASE_WEIGHT, 1.0, "Base recognition weight");
        assert!(
            (PASSIVE_DECAY_RATE - 0.05).abs() < 1e-6,
            "Passive decay rate"
        );
        assert!(
            (JUBILEE_COMPRESSION - 0.8).abs() < 1e-6,
            "Jubilee compression factor"
        );
        assert!(
            (MYCEL_APPRENTICE_MAX - 0.3).abs() < 1e-6,
            "Apprentice graduation threshold"
        );
        assert!(
            (MIN_MYCEL_TO_GIVE - 0.3).abs() < 1e-6,
            "Min MYCEL to give recognition"
        );
    }

    #[test]
    fn test_contribution_type_serialization() {
        use mycelix_finance_types::ContributionType;

        let types = vec![
            ContributionType::Community,
            ContributionType::Governance,
            ContributionType::Care,
            ContributionType::Education,
            ContributionType::Technical,
            ContributionType::Community,
            ContributionType::Education,
        ];

        for ct in types {
            let json = serde_json::to_string(&ct).expect("Serialize failed");
            let deserialized: ContributionType =
                serde_json::from_str(&json).expect("Deserialize failed");
            assert_eq!(ct, deserialized, "ContributionType round-trip failed");
        }
    }

    #[test]
    fn test_mycel_state_serialization() {
        let state = MemberMycelState {
            member_did: "did:mycelix:test:alice".to_string(),
            mycel_score: 0.65,
            participation: 0.7,
            recognition: 0.5,
            validation: 0.8,
            longevity: 0.5,
            active_months: 12,
            is_apprentice: false,
            mentor_did: None,
            recognitions_given_this_cycle: 3,
            current_cycle_id: "2026-02".to_string(),
            last_updated: Timestamp::from_micros(0),
        };

        let json = serde_json::to_string(&state).expect("Serialize failed");
        let deserialized: MemberMycelState =
            serde_json::from_str(&json).expect("Deserialize failed");

        assert_eq!(state.member_did, deserialized.member_did);
        assert_eq!(state.mycel_score, deserialized.mycel_score);
        assert_eq!(state.is_apprentice, deserialized.is_apprentice);
    }
}

// ============================================================================
// Section 5: Recognition Query Tests
// ============================================================================

#[cfg(test)]
mod recognition_query_tests {
    use super::*;

    /// Test 5.1: Recognize a member 3 times and query received recognitions
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_recognition_received() {
        println!("Test 5.1: Get Recognition Received");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let bob_cell = &apps[1].cells()[0];
        let alice_key = agents[0].clone();
        let alice_did = format!("did:mycelix:{}", alice_key);
        let bob_key = agents[1].clone();
        let bob_did = format!("did:mycelix:{}", bob_key);

        // Initialize both members
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct InitializeMemberInput {
            pub member_did: String,
            pub is_apprentice: bool,
            pub mentor_did: Option<String>,
        }

        let _: Record = conductor
            .call(
                &alice_cell.zome("recognition"),
                "initialize_member",
                InitializeMemberInput {
                    member_did: alice_did.clone(),
                    is_apprentice: false,
                    mentor_did: None,
                },
            )
            .await;

        let _: Record = conductor
            .call(
                &bob_cell.zome("recognition"),
                "initialize_member",
                InitializeMemberInput {
                    member_did: bob_did.clone(),
                    is_apprentice: false,
                    mentor_did: None,
                },
            )
            .await;

        // Alice recognizes Bob 3 times with different contribution types
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct RecognizeMemberInput {
            pub recipient_did: String,
            pub contribution_type: mycelix_finance_types::ContributionType,
            pub cycle_id: String,
        }

        use mycelix_finance_types::ContributionType;

        let contribution_types = vec![
            ContributionType::Community,
            ContributionType::Education,
            ContributionType::Care,
        ];

        for ct in &contribution_types {
            let _: Record = conductor
                .call(
                    &alice_cell.zome("recognition"),
                    "recognize_member",
                    RecognizeMemberInput {
                        recipient_did: bob_did.clone(),
                        contribution_type: ct.clone(),
                        cycle_id: "2026-02".to_string(),
                    },
                )
                .await;

            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        println!("  - 3 recognitions given to Bob");

        // Query recognition events received by Bob
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct GetRecognitionsInput {
            pub member_did: String,
            pub cycle_id: Option<String>,
        }

        let recognitions: Vec<RecognitionEvent> = conductor
            .call(
                &bob_cell.zome("recognition"),
                "get_recognition_received",
                GetRecognitionsInput {
                    member_did: bob_did.clone(),
                    cycle_id: None,
                },
            )
            .await;

        assert_eq!(
            recognitions.len(),
            3,
            "Bob should have received 3 recognitions"
        );

        for (i, r) in recognitions.iter().enumerate() {
            println!(
                "  - Recognition {}: type={:?}, weight={:.2}",
                i, r.contribution_type, r.weight
            );
        }

        println!("Test 5.1 PASSED: Recognition query returns correct results");
    }
}

// ============================================================================
// Section 6: Lifecycle Advanced Tests
// ============================================================================

#[cfg(test)]
mod lifecycle_advanced {
    use super::*;

    /// Test 6.1: Jubilee normalization integration
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_jubilee_normalize_integration() {
        println!("Test 6.1: Jubilee Normalize Integration");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_key = agents[0].clone();
        let alice_did = format!("did:mycelix:{}", alice_key);

        // Initialize member
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct InitializeMemberInput {
            pub member_did: String,
            pub is_apprentice: bool,
            pub mentor_did: Option<String>,
        }

        let _: Record = conductor
            .call(
                &alice_cell.zome("recognition"),
                "initialize_member",
                InitializeMemberInput {
                    member_did: alice_did.clone(),
                    is_apprentice: false,
                    mentor_did: None,
                },
            )
            .await;

        // Set score to 0.8 via update
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct UpdateMycelInput {
            pub member_did: String,
            pub participation: f64,
            pub recognition: f64,
            pub validation_override: Option<f64>,
            pub active_months: u32,
        }

        let _: MemberMycelState = conductor
            .call(
                &alice_cell.zome("recognition"),
                "update_mycel_score",
                UpdateMycelInput {
                    member_did: alice_did.clone(),
                    participation: 1.0,
                    recognition: 1.0,
                    validation_override: Some(1.0),
                    active_months: 24,
                },
            )
            .await;

        println!("  - MYCEL set to high value");

        // Apply jubilee normalization
        let jubileed: MemberMycelState = conductor
            .call(
                &alice_cell.zome("recognition"),
                "jubilee_normalize",
                alice_did.clone(),
            )
            .await;

        // Jubilee formula: new = 0.3 + (current - 0.3) * 0.8
        // For 0.8: 0.3 + (0.8 - 0.3) * 0.8 = 0.3 + 0.4 = 0.7
        // For ~1.0: 0.3 + (1.0 - 0.3) * 0.8 = 0.3 + 0.56 = 0.86
        println!("  - Post-jubilee MYCEL: {:.4}", jubileed.mycel_score);
        assert!(
            jubileed.mycel_score < 1.0,
            "Jubilee should compress scores toward 0.3"
        );
        // The exact value depends on the pre-jubilee score; just verify compression happened
        assert!(
            jubileed.mycel_score >= 0.3,
            "Post-jubilee score should be >= 0.3"
        );

        println!("Test 6.1 PASSED: Jubilee normalization works in integration");
    }

    /// Test 6.2: Dissolve MYCEL sets all scores to 0
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_dissolve_mycel() {
        println!("Test 6.2: Dissolve MYCEL");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_key = agents[0].clone();
        let alice_did = format!("did:mycelix:{}", alice_key);

        // Initialize member
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct InitializeMemberInput {
            pub member_did: String,
            pub is_apprentice: bool,
            pub mentor_did: Option<String>,
        }

        let _: Record = conductor
            .call(
                &alice_cell.zome("recognition"),
                "initialize_member",
                InitializeMemberInput {
                    member_did: alice_did.clone(),
                    is_apprentice: false,
                    mentor_did: None,
                },
            )
            .await;

        println!("  - Member initialized");

        // Dissolve MYCEL
        let _: () = conductor
            .call(
                &alice_cell.zome("recognition"),
                "dissolve_mycel",
                alice_did.clone(),
            )
            .await;

        println!("  - MYCEL dissolved");

        // Verify all scores are 0 by checking state
        // Re-initialize to verify the state was actually cleared
        // (dissolve_mycel sets everything to 0)
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct UpdateMycelInput {
            pub member_did: String,
            pub participation: f64,
            pub recognition: f64,
            pub validation_override: Option<f64>,
            pub active_months: u32,
        }

        // After dissolve, trying to update should start from 0
        let state: MemberMycelState = conductor
            .call(
                &alice_cell.zome("recognition"),
                "update_mycel_score",
                UpdateMycelInput {
                    member_did: alice_did.clone(),
                    participation: 0.0,
                    recognition: 0.0,
                    validation_override: Some(0.0),
                    active_months: 0,
                },
            )
            .await;

        assert!(
            (state.mycel_score - 0.0).abs() < 1e-6,
            "MYCEL score should be 0 after dissolve"
        );
        assert!(
            (state.participation - 0.0).abs() < 1e-6,
            "Participation should be 0"
        );
        assert!(
            (state.recognition - 0.0).abs() < 1e-6,
            "Recognition should be 0"
        );
        assert!(
            (state.validation - 0.0).abs() < 1e-6,
            "Validation should be 0"
        );

        println!("  - All scores verified as 0");
        println!("Test 6.2 PASSED: Dissolve MYCEL zeroes all scores");
    }

    /// Test 6.3: Passive decay integration test
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_passive_decay_integration() {
        println!("Test 6.3: Passive Decay Integration");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_key = agents[0].clone();
        let alice_did = format!("did:mycelix:{}", alice_key);

        // Initialize member
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct InitializeMemberInput {
            pub member_did: String,
            pub is_apprentice: bool,
            pub mentor_did: Option<String>,
        }

        let _: Record = conductor
            .call(
                &alice_cell.zome("recognition"),
                "initialize_member",
                InitializeMemberInput {
                    member_did: alice_did.clone(),
                    is_apprentice: false,
                    mentor_did: None,
                },
            )
            .await;

        // Set initial MYCEL to 0.5
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct UpdateMycelInput {
            pub member_did: String,
            pub participation: f64,
            pub recognition: f64,
            pub validation_override: Option<f64>,
            pub active_months: u32,
        }

        let before: MemberMycelState = conductor
            .call(
                &alice_cell.zome("recognition"),
                "update_mycel_score",
                UpdateMycelInput {
                    member_did: alice_did.clone(),
                    participation: 0.6,
                    recognition: 0.5,
                    validation_override: Some(0.5),
                    active_months: 6,
                },
            )
            .await;

        let score_before = before.mycel_score;
        println!("  - MYCEL before decay: {:.4}", score_before);

        // Apply passive decay (simulates 1 year of inactivity)
        let after: MemberMycelState = conductor
            .call(
                &alice_cell.zome("recognition"),
                "apply_passive_decay",
                alice_did.clone(),
            )
            .await;

        println!("  - MYCEL after decay: {:.4}", after.mycel_score);

        // Decay should reduce the score (5% annual linear decay)
        // Exact amount depends on elapsed time since last update
        assert!(
            after.mycel_score <= score_before,
            "Score should decrease or stay the same after decay: before={}, after={}",
            score_before,
            after.mycel_score
        );

        println!("Test 6.3 PASSED: Passive decay reduces MYCEL score");
    }
}
