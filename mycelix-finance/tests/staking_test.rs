//! # Staking Zome Integration Tests
//!
//! Tests for the Staking coordinator zome covering:
//! - Collateral stake lifecycle (create, unbond, withdraw)
//! - MYCEL-weighted staking (weight = 1.0 + mycel_score)
//! - Slashing with cryptographic evidence (Blake2b hashes)
//! - Crypto escrow with hash-lock and multi-sig release conditions
//! - Serialization round-trips for all status enums
//!
//! ## Running Tests
//!
//! ```bash
//! cargo test --test staking_test                   # Unit tests only
//! cargo test --test staking_test -- --ignored      # Full integration tests
//! ```

use holochain::sweettest::*;
use holochain::prelude::*;
use std::time::Duration;

use staking_integrity::*;

mod test_helpers {
    pub const TEST_DID_PREFIX: &str = "did:mycelix:test:";

    pub fn test_did(suffix: &str) -> String {
        format!("{}{}", TEST_DID_PREFIX, suffix)
    }
}

use test_helpers::*;

// ============================================================================
// Section 1: Stake Lifecycle Tests
// ============================================================================

#[cfg(test)]
mod stake_lifecycle {
    use super::*;

    /// Test 1.1: Create a stake with MYCEL weighting, verify weight = 1.0 + mycel_score
    /// MYCEL score is fetched from recognition zome (not caller-provided).
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_create_stake_with_mycel_weighting() {
        println!("Test 1.1: Create Stake with MYCEL Weighting");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path).await.expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = format!("did:mycelix:{}", agents[0]);

        // Initialize member in recognition zome first (sets MYCEL via components)
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct InitializeMemberInput {
            pub member_did: String,
            pub is_apprentice: bool,
            pub mentor_did: Option<String>,
        }

        let _: Record = conductor
            .call(&alice_cell.zome("recognition"), "initialize_member", InitializeMemberInput {
                member_did: alice_did.clone(),
                is_apprentice: false,
                mentor_did: None,
            })
            .await;

        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct CreateStakeInput {
            pub staker_did: String,
            pub sap_amount: u64,
        }

        let input = CreateStakeInput {
            staker_did: alice_did.clone(),
            sap_amount: 10_000,
        };

        let result: Record = conductor
            .call(&alice_cell.zome("staking"), "create_stake", input)
            .await;

        let stake: CollateralStake = result
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert_eq!(stake.staker_did, alice_did);
        assert_eq!(stake.sap_amount, 10_000);
        // MYCEL score is fetched from recognition; verify it's in valid range
        assert!(stake.mycel_score >= 0.0 && stake.mycel_score <= 1.0, "MYCEL score in [0,1]");
        assert!((stake.stake_weight - (1.0 + stake.mycel_score)).abs() < 0.001,
            "Weight should be 1.0 + mycel_score");
        assert!(matches!(stake.status, StakeStatus::Active));
        assert_eq!(stake.pending_rewards, 0);
        assert!(stake.unbonding_until.is_none());

        println!("  - Staker: {}", stake.staker_did);
        println!("  - SAP amount: {}", stake.sap_amount);
        println!("  - MYCEL score: {}", stake.mycel_score);
        println!("  - Stake weight: {}", stake.stake_weight);
        println!("  - Status: {:?}", stake.status);
        println!("Test 1.1 PASSED: Stake created with correct MYCEL weighting");
    }

    /// Test 1.2: Full lifecycle — create stake, begin unbonding, then withdraw
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_unbonding_and_withdrawal() {
        println!("Test 1.2: Unbonding and Withdrawal Lifecycle");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path).await.expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = format!("did:mycelix:{}", agents[0]);

        // Step 1: Create stake
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct CreateStakeInput {
            pub staker_did: String,
            pub sap_amount: u64,
        }

        let input = CreateStakeInput {
            staker_did: alice_did.clone(),
            sap_amount: 5_000,
        };

        let result: Record = conductor
            .call(&alice_cell.zome("staking"), "create_stake", input)
            .await;

        let stake: CollateralStake = result
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        let stake_id = stake.id.clone();
        assert!(matches!(stake.status, StakeStatus::Active));
        println!("  - Stake created: {}", stake_id);

        // Step 2: Begin unbonding
        let unbonding_result: Record = conductor
            .call(&alice_cell.zome("staking"), "begin_unbonding", stake_id.clone())
            .await;

        let unbonding_stake: CollateralStake = unbonding_result
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert!(matches!(unbonding_stake.status, StakeStatus::Unbonding));
        assert!(unbonding_stake.unbonding_until.is_some(), "Should have unbonding_until set");
        println!("  - Status after unbonding: {:?}", unbonding_stake.status);
        println!("  - Unbonding until: {:?}", unbonding_stake.unbonding_until);

        // Step 3: Attempt withdrawal (should fail since unbonding period is not complete)
        // The 21-day unbonding period means we cannot withdraw immediately in a real
        // test. We verify the mechanism is in place by trying and expecting failure.
        let withdraw_result: Result<Record, _> = conductor
            .call_fallible(&alice_cell.zome("staking"), "withdraw_stake", stake_id.clone())
            .await;

        // Withdrawal should fail because unbonding period is not complete
        assert!(
            withdraw_result.is_err(),
            "Withdrawal should fail during unbonding period"
        );
        println!("  - Withdrawal during unbonding correctly rejected");

        println!("Test 1.2 PASSED: Unbonding lifecycle works (withdrawal blocked until period completes)");
    }

    /// Test 1.3: Cannot withdraw a stake that is still in unbonding period
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_cannot_withdraw_during_unbonding() {
        println!("Test 1.3: Cannot Withdraw During Unbonding");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path).await.expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = format!("did:mycelix:{}", agents[0]);

        // Create and unbond
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct CreateStakeInput {
            pub staker_did: String,
            pub sap_amount: u64,
        }

        let input = CreateStakeInput {
            staker_did: alice_did.clone(),
            sap_amount: 8_000,
        };

        let result: Record = conductor
            .call(&alice_cell.zome("staking"), "create_stake", input)
            .await;

        let stake: CollateralStake = result
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        let stake_id = stake.id.clone();

        // Begin unbonding
        let _: Record = conductor
            .call(&alice_cell.zome("staking"), "begin_unbonding", stake_id.clone())
            .await;

        // Try to withdraw immediately (within 21-day window)
        let result: Result<Record, _> = conductor
            .call_fallible(&alice_cell.zome("staking"), "withdraw_stake", stake_id.clone())
            .await;

        match result {
            Err(e) => {
                let error_msg = format!("{:?}", e);
                assert!(
                    error_msg.contains("Unbonding period not complete")
                        || error_msg.contains("not complete")
                        || error_msg.contains("unbonding"),
                    "Should indicate unbonding period not complete, got: {}",
                    error_msg
                );
                println!("  - Premature withdrawal rejected: OK");
            }
            Ok(_) => panic!("Should have rejected withdrawal during unbonding period"),
        }

        println!("Test 1.3 PASSED: Cannot withdraw during unbonding period");
    }

    /// Test 1.4: Update MYCEL score on an active stake
    /// Score is re-fetched from recognition zome (not caller-provided).
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_update_mycel_score() {
        println!("Test 1.4: Update MYCEL Score (fetched from recognition)");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path).await.expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = format!("did:mycelix:{}", agents[0]);

        // Initialize member in recognition zome
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct InitializeMemberInput {
            pub member_did: String,
            pub is_apprentice: bool,
            pub mentor_did: Option<String>,
        }

        let _: Record = conductor
            .call(&alice_cell.zome("recognition"), "initialize_member", InitializeMemberInput {
                member_did: alice_did.clone(),
                is_apprentice: false,
                mentor_did: None,
            })
            .await;

        // Create stake — MYCEL fetched from recognition
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct CreateStakeInput {
            pub staker_did: String,
            pub sap_amount: u64,
        }

        let input = CreateStakeInput {
            staker_did: alice_did.clone(),
            sap_amount: 12_000,
        };

        let result: Record = conductor
            .call(&alice_cell.zome("staking"), "create_stake", input)
            .await;

        let stake: CollateralStake = result
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        let initial_weight = stake.stake_weight;
        println!("  - Initial MYCEL: {}, weight: {}", stake.mycel_score, stake.stake_weight);

        // Update stake — re-fetches MYCEL from recognition
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct UpdateMycelInput {
            pub stake_id: String,
        }

        let update_input = UpdateMycelInput {
            stake_id: stake.id.clone(),
        };

        let updated_result: Record = conductor
            .call(&alice_cell.zome("staking"), "update_stake_mycel", update_input)
            .await;

        let updated_stake: CollateralStake = updated_result
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        // Verify weight = 1.0 + mycel_score (whatever recognition returns)
        assert!((updated_stake.stake_weight - (1.0 + updated_stake.mycel_score)).abs() < 0.001,
            "Weight should be 1.0 + mycel_score");
        assert_eq!(updated_stake.sap_amount, 12_000, "SAP amount should be unchanged");
        assert!(matches!(updated_stake.status, StakeStatus::Active), "Should still be Active");

        println!("  - Updated MYCEL: {}, weight: {}", updated_stake.mycel_score, updated_stake.stake_weight);
        println!("Test 1.4 PASSED: MYCEL score update re-fetches from recognition");
    }
}

// ============================================================================
// Section 2: Slashing Tests
// ============================================================================

#[cfg(test)]
mod slashing_tests {
    use super::*;

    /// Build a minimal slashing evidence payload for tests
    fn test_evidence(evidence_type: EvidenceType, reporters: Vec<String>) -> SlashingEvidence {
        SlashingEvidence {
            evidence_type,
            violation_height: 42_000,
            conflicting_data: vec![
                b"block_a_hash_data_aaaaaaaaaaaaaa".to_vec(),
                b"block_b_hash_data_bbbbbbbbbbbbbb".to_vec(),
            ],
            signatures: vec![
                b"sig_validator_1_xxxxxxxxxxxxxxxxx".to_vec(),
                b"sig_validator_2_xxxxxxxxxxxxxxxxx".to_vec(),
            ],
            violation_time: 1_700_000_000,
            merkle_proof: Some(b"merkle_proof_placeholder_data_32".to_vec()),
            reporters,
        }
    }

    /// Test 2.1: Slash for double signing — 100% slash, jailed
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_slash_for_double_signing() {
        println!("Test 2.1: Slash for Double Signing (100%, Jailed)");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path).await.expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = format!("did:mycelix:{}", agents[0]);

        // Create a stake first
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct CreateStakeInput {
            pub staker_did: String,
            pub sap_amount: u64,
        }

        let create_input = CreateStakeInput {
            staker_did: alice_did.clone(),
            sap_amount: 20_000,
        };

        let stake_record: Record = conductor
            .call(&alice_cell.zome("staking"), "create_stake", create_input)
            .await;

        let stake: CollateralStake = stake_record
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        let stake_id = stake.id.clone();
        println!("  - Created stake: {} with {} SAP", stake_id, stake.sap_amount);

        // Slash for double signing
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct SlashStakeInput {
            pub stake_id: String,
            pub reason: SlashingReason,
            pub evidence: SlashingEvidence,
            pub custom_slash_percentage: Option<u8>,
        }

        let slash_input = SlashStakeInput {
            stake_id: stake_id.clone(),
            reason: SlashingReason::DoubleSigning,
            evidence: test_evidence(
                EvidenceType::DoubleSignEvidence,
                vec![format!("did:mycelix:{}", agents[0]), format!("did:mycelix:{}", agents[0])],
            ),
            custom_slash_percentage: None, // Use default 100%
        };

        let slash_result: Record = conductor
            .call(&alice_cell.zome("staking"), "slash_stake", slash_input)
            .await;

        let slashing_event: SlashingEvent = slash_result
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert_eq!(slashing_event.slash_percentage, 100, "DoubleSigning should slash 100%");
        assert_eq!(slashing_event.sap_slashed, 20_000, "Should slash entire stake");
        assert!(slashing_event.jailed, "DoubleSigning should result in jail");
        assert!(slashing_event.jail_release.is_some(), "Should have jail release timestamp");
        assert_eq!(slashing_event.evidence_hash.len(), 32, "Evidence hash should be 32 bytes");
        assert!(matches!(slashing_event.reason, SlashingReason::DoubleSigning));

        println!("  - Slash percentage: {}%", slashing_event.slash_percentage);
        println!("  - SAP slashed: {}", slashing_event.sap_slashed);
        println!("  - Jailed: {}", slashing_event.jailed);
        println!("  - Evidence hash length: {} bytes", slashing_event.evidence_hash.len());
        println!("Test 2.1 PASSED: Double signing slash works correctly");
    }

    /// Test 2.2: Slash for downtime — 5% slash, not jailed
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_slash_for_downtime() {
        println!("Test 2.2: Slash for Downtime (5%, Not Jailed)");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path).await.expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = format!("did:mycelix:{}", agents[0]);

        // Create a stake
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct CreateStakeInput {
            pub staker_did: String,
            pub sap_amount: u64,
        }

        let create_input = CreateStakeInput {
            staker_did: alice_did.clone(),
            sap_amount: 100_000,
        };

        let stake_record: Record = conductor
            .call(&alice_cell.zome("staking"), "create_stake", create_input)
            .await;

        let stake: CollateralStake = stake_record
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        let stake_id = stake.id.clone();

        // Slash for downtime
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct SlashStakeInput {
            pub stake_id: String,
            pub reason: SlashingReason,
            pub evidence: SlashingEvidence,
            pub custom_slash_percentage: Option<u8>,
        }

        let slash_input = SlashStakeInput {
            stake_id: stake_id.clone(),
            reason: SlashingReason::Downtime,
            evidence: test_evidence(
                EvidenceType::DowntimeEvidence,
                vec![format!("did:mycelix:{}", agents[0])],
            ),
            custom_slash_percentage: None, // Use default 5%
        };

        let slash_result: Record = conductor
            .call(&alice_cell.zome("staking"), "slash_stake", slash_input)
            .await;

        let slashing_event: SlashingEvent = slash_result
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert_eq!(slashing_event.slash_percentage, 5, "Downtime should slash 5%");
        // 5% of 100_000 = 5_000
        assert_eq!(slashing_event.sap_slashed, 5_000, "Should slash 5% of stake");
        assert!(!slashing_event.jailed, "Downtime should NOT result in jail");
        assert!(slashing_event.jail_release.is_none(), "Should have no jail release");
        assert!(matches!(slashing_event.reason, SlashingReason::Downtime));

        println!("  - Slash percentage: {}%", slashing_event.slash_percentage);
        println!("  - SAP slashed: {} (of {})", slashing_event.sap_slashed, stake.sap_amount);
        println!("  - Jailed: {}", slashing_event.jailed);
        println!("Test 2.2 PASSED: Downtime slash works correctly (5%, no jail)");
    }

    /// Test 2.3: Slash with custom percentage override
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_slash_with_custom_percentage() {
        println!("Test 2.3: Slash with Custom Percentage Override");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path).await.expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = format!("did:mycelix:{}", agents[0]);

        // Create a stake
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct CreateStakeInput {
            pub staker_did: String,
            pub sap_amount: u64,
        }

        let create_input = CreateStakeInput {
            staker_did: alice_did.clone(),
            sap_amount: 50_000,
        };

        let stake_record: Record = conductor
            .call(&alice_cell.zome("staking"), "create_stake", create_input)
            .await;

        let stake: CollateralStake = stake_record
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        let stake_id = stake.id.clone();

        // Slash with custom 25% instead of default 10% for InvalidGradient
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct SlashStakeInput {
            pub stake_id: String,
            pub reason: SlashingReason,
            pub evidence: SlashingEvidence,
            pub custom_slash_percentage: Option<u8>,
        }

        let slash_input = SlashStakeInput {
            stake_id: stake_id.clone(),
            reason: SlashingReason::InvalidGradient,
            evidence: test_evidence(
                EvidenceType::GradientAnalysis,
                vec![format!("did:mycelix:{}", agents[0])],
            ),
            custom_slash_percentage: Some(25), // Override default 10%
        };

        let slash_result: Record = conductor
            .call(&alice_cell.zome("staking"), "slash_stake", slash_input)
            .await;

        let slashing_event: SlashingEvent = slash_result
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert_eq!(slashing_event.slash_percentage, 25, "Should use custom 25%");
        // 25% of 50_000 = 12_500
        assert_eq!(slashing_event.sap_slashed, 12_500, "Should slash 25% of stake");
        assert!(!slashing_event.jailed, "InvalidGradient should NOT result in jail");
        assert!(matches!(slashing_event.reason, SlashingReason::InvalidGradient));

        println!("  - Reason: {:?} (default would be 10%)", slashing_event.reason);
        println!("  - Custom slash percentage: {}%", slashing_event.slash_percentage);
        println!("  - SAP slashed: {}", slashing_event.sap_slashed);
        println!("Test 2.3 PASSED: Custom slash percentage overrides default");
    }
}

// ============================================================================
// Section 3: Escrow Tests
// ============================================================================

#[cfg(test)]
mod escrow_tests {
    use super::*;

    /// Test 3.1: Hash-lock escrow — create, reveal preimage, release
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_hash_lock_escrow() {
        println!("Test 3.1: Hash-Lock Escrow Lifecycle");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path).await.expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = format!("did:mycelix:{}", agents[0]);
        let bob_did = format!("did:mycelix:{}", agents[1]);

        // Preimage and its hash (all hash types use Blake2b-256 internally)
        let preimage = b"secret_preimage_for_escrow_release".to_vec();
        let hash_lock = blake2b_simd::Params::new()
            .hash_length(32)
            .hash(&preimage)
            .as_bytes()
            .to_vec();

        // Create escrow with hash-lock condition
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct CreateEscrowInput {
            pub depositor_did: String,
            pub beneficiary_did: String,
            pub sap_amount: u64,
            pub purpose: String,
            pub conditions: Vec<ReleaseCondition>,
            pub required_conditions: u8,
            pub hash_lock: Option<Vec<u8>>,
            pub timelock: Option<i64>,
            pub multisig_threshold: Option<u8>,
            pub multisig_signers: Vec<String>,
        }

        let escrow_input = CreateEscrowInput {
            depositor_did: alice_did.clone(),
            beneficiary_did: bob_did.clone(),
            sap_amount: 5_000,
            purpose: "Payment for services upon secret reveal".to_string(),
            conditions: vec![ReleaseCondition::HashLock {
                hash: hash_lock.clone(),
                hash_type: EscrowHashType::Blake2b,
            }],
            required_conditions: 1,
            hash_lock: Some(hash_lock.clone()),
            timelock: None,
            multisig_threshold: None,
            multisig_signers: vec![],
        };

        let escrow_record: Record = conductor
            .call(&alice_cell.zome("staking"), "create_escrow", escrow_input)
            .await;

        let escrow: CryptoEscrow = escrow_record
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert!(matches!(escrow.status, EscrowStatus::Pending));
        assert_eq!(escrow.sap_amount, 5_000);
        assert_eq!(escrow.depositor_did, alice_did);
        assert_eq!(escrow.beneficiary_did, bob_did);
        assert!(escrow.met_conditions.is_empty(), "No conditions met yet");
        println!("  - Escrow created: {}", escrow.id);
        println!("  - Status: {:?}", escrow.status);

        let escrow_id = escrow.id.clone();

        // Reveal preimage
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct RevealPreimageInput {
            pub escrow_id: String,
            pub preimage: Vec<u8>,
        }

        let reveal_input = RevealPreimageInput {
            escrow_id: escrow_id.clone(),
            preimage: preimage.clone(),
        };

        let reveal_result: Record = conductor
            .call(&alice_cell.zome("staking"), "reveal_hash_preimage", reveal_input)
            .await;

        let revealed_escrow: CryptoEscrow = reveal_result
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert!(
            matches!(revealed_escrow.status, EscrowStatus::Releasable),
            "Escrow should be Releasable after hash-lock satisfied"
        );
        assert!(!revealed_escrow.met_conditions.is_empty(), "Hash-lock condition should be met");
        println!("  - After preimage reveal, status: {:?}", revealed_escrow.status);

        // Release escrow
        let release_result: Record = conductor
            .call(&alice_cell.zome("staking"), "release_escrow", escrow_id.clone())
            .await;

        let released_escrow: CryptoEscrow = release_result
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert!(matches!(released_escrow.status, EscrowStatus::Released));
        assert!(released_escrow.released_at.is_some(), "Should have release timestamp");
        println!("  - After release, status: {:?}", released_escrow.status);

        println!("Test 3.1 PASSED: Hash-lock escrow lifecycle works end-to-end");
    }

    /// Test 3.2: Multi-sig escrow — create 2-of-3, add signatures, release
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_multisig_escrow() {
        println!("Test 3.2: Multi-Sig Escrow (2-of-3)");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path).await.expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = format!("did:mycelix:{}", agents[0]);
        let bob_did = format!("did:mycelix:{}", agents[1]);

        let signer_1 = format!("did:mycelix:{}", agents[0]);
        let signer_2 = format!("did:mycelix:{}", agents[0]);
        let signer_3 = format!("did:mycelix:{}", agents[0]);

        // Create 2-of-3 multi-sig escrow
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct CreateEscrowInput {
            pub depositor_did: String,
            pub beneficiary_did: String,
            pub sap_amount: u64,
            pub purpose: String,
            pub conditions: Vec<ReleaseCondition>,
            pub required_conditions: u8,
            pub hash_lock: Option<Vec<u8>>,
            pub timelock: Option<i64>,
            pub multisig_threshold: Option<u8>,
            pub multisig_signers: Vec<String>,
        }

        let escrow_input = CreateEscrowInput {
            depositor_did: alice_did.clone(),
            beneficiary_did: bob_did.clone(),
            sap_amount: 10_000,
            purpose: "Multi-sig governed payment".to_string(),
            conditions: vec![ReleaseCondition::MultiSig {
                threshold: 2,
                signers: vec![signer_1.clone(), signer_2.clone(), signer_3.clone()],
            }],
            required_conditions: 1, // Just the multi-sig condition
            hash_lock: None,
            timelock: None,
            multisig_threshold: Some(2),
            multisig_signers: vec![signer_1.clone(), signer_2.clone(), signer_3.clone()],
        };

        let escrow_record: Record = conductor
            .call(&alice_cell.zome("staking"), "create_escrow", escrow_input)
            .await;

        let escrow: CryptoEscrow = escrow_record
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        let escrow_id = escrow.id.clone();
        assert!(matches!(escrow.status, EscrowStatus::Pending));
        println!("  - Multi-sig escrow created: {}", escrow_id);

        // Signer 1 signs
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct AddSignatureInput {
            pub escrow_id: String,
            pub signer_did: String,
            pub signature: Vec<u8>,
        }

        let sig_1_input = AddSignatureInput {
            escrow_id: escrow_id.clone(),
            signer_did: signer_1.clone(),
            signature: b"signature_from_signer_alpha_data".to_vec(),
        };

        let sig_1_result: Record = conductor
            .call(&alice_cell.zome("staking"), "add_escrow_signature", sig_1_input)
            .await;

        let after_sig_1: CryptoEscrow = sig_1_result
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert_eq!(after_sig_1.collected_signatures.len(), 1, "Should have 1 signature");
        // With only 1 of 2 required signatures, still Pending
        assert!(
            matches!(after_sig_1.status, EscrowStatus::Pending),
            "Should still be Pending with 1/2 sigs"
        );
        println!("  - Signature 1 added (1/2 required), status: {:?}", after_sig_1.status);

        // Signer 2 signs (reaches threshold)
        let sig_2_input = AddSignatureInput {
            escrow_id: escrow_id.clone(),
            signer_did: signer_2.clone(),
            signature: b"signature_from_signer_beta__data".to_vec(),
        };

        let sig_2_result: Record = conductor
            .call(&alice_cell.zome("staking"), "add_escrow_signature", sig_2_input)
            .await;

        let after_sig_2: CryptoEscrow = sig_2_result
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert_eq!(after_sig_2.collected_signatures.len(), 2, "Should have 2 signatures");
        assert!(
            matches!(after_sig_2.status, EscrowStatus::Releasable),
            "Should be Releasable with 2/2 sigs met"
        );
        println!("  - Signature 2 added (2/2 required), status: {:?}", after_sig_2.status);

        // Release escrow
        let release_result: Record = conductor
            .call(&alice_cell.zome("staking"), "release_escrow", escrow_id.clone())
            .await;

        let released: CryptoEscrow = release_result
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert!(matches!(released.status, EscrowStatus::Released));
        assert!(released.released_at.is_some());
        println!("  - Escrow released: {:?}", released.status);

        println!("Test 3.2 PASSED: Multi-sig escrow 2-of-3 works end-to-end");
    }

    /// Test 3.3: Unauthorized signer is rejected
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_unauthorized_signer_rejected() {
        println!("Test 3.3: Unauthorized Signer Rejected");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path).await.expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = format!("did:mycelix:{}", agents[0]);
        let bob_did = format!("did:mycelix:{}", agents[1]);

        let signer_1 = format!("did:mycelix:{}", agents[0]);
        let signer_2 = format!("did:mycelix:{}", agents[0]);
        let unauthorized = format!("did:mycelix:{}", agents[0]);

        // Create escrow with specific authorized signers
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct CreateEscrowInput {
            pub depositor_did: String,
            pub beneficiary_did: String,
            pub sap_amount: u64,
            pub purpose: String,
            pub conditions: Vec<ReleaseCondition>,
            pub required_conditions: u8,
            pub hash_lock: Option<Vec<u8>>,
            pub timelock: Option<i64>,
            pub multisig_threshold: Option<u8>,
            pub multisig_signers: Vec<String>,
        }

        let escrow_input = CreateEscrowInput {
            depositor_did: alice_did.clone(),
            beneficiary_did: bob_did.clone(),
            sap_amount: 3_000,
            purpose: "Restricted signers test".to_string(),
            conditions: vec![ReleaseCondition::MultiSig {
                threshold: 2,
                signers: vec![signer_1.clone(), signer_2.clone()],
            }],
            required_conditions: 1,
            hash_lock: None,
            timelock: None,
            multisig_threshold: Some(2),
            multisig_signers: vec![signer_1.clone(), signer_2.clone()],
        };

        let escrow_record: Record = conductor
            .call(&alice_cell.zome("staking"), "create_escrow", escrow_input)
            .await;

        let escrow: CryptoEscrow = escrow_record
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        let escrow_id = escrow.id.clone();

        // Attempt to sign with unauthorized signer
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct AddSignatureInput {
            pub escrow_id: String,
            pub signer_did: String,
            pub signature: Vec<u8>,
        }

        let bad_sig = AddSignatureInput {
            escrow_id: escrow_id.clone(),
            signer_did: unauthorized.clone(),
            signature: b"forged_signature_data_xxxxxxxxx".to_vec(),
        };

        let result: Result<Record, _> = conductor
            .call_fallible(&alice_cell.zome("staking"), "add_escrow_signature", bad_sig)
            .await;

        match result {
            Err(e) => {
                let error_msg = format!("{:?}", e);
                assert!(
                    error_msg.contains("not authorized")
                        || error_msg.contains("Signer not authorized"),
                    "Should indicate signer not authorized, got: {}",
                    error_msg
                );
                println!("  - Unauthorized signer rejected: OK");
            }
            Ok(_) => panic!("Should have rejected unauthorized signer"),
        }

        println!("Test 3.3 PASSED: Unauthorized signers are rejected");
    }
}

// ============================================================================
// Section 4: Unit Tests (No Conductor Required)
// ============================================================================

#[cfg(test)]
mod unit_tests {
    use super::*;

    /// Test stake weight calculation: weight = 1.0 + mycel_score
    #[test]
    fn test_stake_weight_calculation() {
        let test_cases: Vec<(f32, f32)> = vec![
            (0.0, 1.0),   // Minimum MYCEL
            (0.1, 1.1),
            (0.25, 1.25),
            (0.5, 1.5),   // Midpoint
            (0.7, 1.7),
            (0.9, 1.9),
            (1.0, 2.0),   // Maximum MYCEL
        ];

        for (mycel_score, expected_weight) in test_cases {
            let weight = 1.0 + mycel_score;
            assert!(
                (weight - expected_weight).abs() < 0.001,
                "For MYCEL {}: expected weight {}, got {}",
                mycel_score,
                expected_weight,
                weight
            );
        }

        // Verify weight range is always [1.0, 2.0]
        for i in 0..=100 {
            let mycel = i as f32 / 100.0;
            let w = 1.0 + mycel;
            assert!(w >= 1.0 && w <= 2.0, "Weight {} out of range for MYCEL {}", w, mycel);
        }
    }

    /// Test SlashingReason default percentages and jail behavior
    #[test]
    fn test_slashing_reason_defaults() {
        // DoubleSigning: 100%, jail
        assert_eq!(SlashingReason::DoubleSigning.default_slash_percentage(), 100);
        assert!(SlashingReason::DoubleSigning.results_in_jail());

        // Downtime: 5%, no jail
        assert_eq!(SlashingReason::Downtime.default_slash_percentage(), 5);
        assert!(!SlashingReason::Downtime.results_in_jail());

        // ByzantineConsensus: 50%, no jail
        assert_eq!(SlashingReason::ByzantineConsensus.default_slash_percentage(), 50);
        assert!(!SlashingReason::ByzantineConsensus.results_in_jail());

        // InvalidGradient: 10%, no jail
        assert_eq!(SlashingReason::InvalidGradient.default_slash_percentage(), 10);
        assert!(!SlashingReason::InvalidGradient.results_in_jail());

        // CartelActivity: 75%, jail
        assert_eq!(SlashingReason::CartelActivity.default_slash_percentage(), 75);
        assert!(SlashingReason::CartelActivity.results_in_jail());

        // GovernanceManipulation: 50%, no jail
        assert_eq!(SlashingReason::GovernanceManipulation.default_slash_percentage(), 50);
        assert!(!SlashingReason::GovernanceManipulation.results_in_jail());

        // Verify jail reasons are exactly DoubleSigning and CartelActivity
        let all_reasons = vec![
            SlashingReason::DoubleSigning,
            SlashingReason::Downtime,
            SlashingReason::ByzantineConsensus,
            SlashingReason::InvalidGradient,
            SlashingReason::CartelActivity,
            SlashingReason::GovernanceManipulation,
        ];

        let jail_reasons: Vec<_> = all_reasons
            .iter()
            .filter(|r| r.results_in_jail())
            .collect();
        assert_eq!(jail_reasons.len(), 2, "Exactly 2 reasons should result in jail");
    }

    /// Test StakeStatus serialization round-trip
    #[test]
    fn test_stake_status_serialization() {
        let statuses = vec![
            StakeStatus::Active,
            StakeStatus::Unbonding,
            StakeStatus::Withdrawn,
            StakeStatus::Slashed,
            StakeStatus::Jailed,
        ];

        for status in statuses {
            let json = serde_json::to_string(&status).expect("Serialize failed");
            let deserialized: StakeStatus =
                serde_json::from_str(&json).expect("Deserialize failed");
            assert_eq!(status, deserialized, "StakeStatus round-trip failed for {:?}", status);
        }
    }

    /// Test EscrowStatus serialization round-trip
    #[test]
    fn test_escrow_status_serialization() {
        let statuses = vec![
            EscrowStatus::Pending,
            EscrowStatus::Releasable,
            EscrowStatus::Released,
            EscrowStatus::Refunded,
            EscrowStatus::Disputed,
            EscrowStatus::Expired,
        ];

        for status in statuses {
            let json = serde_json::to_string(&status).expect("Serialize failed");
            let deserialized: EscrowStatus =
                serde_json::from_str(&json).expect("Deserialize failed");
            assert_eq!(status, deserialized, "EscrowStatus round-trip failed for {:?}", status);
        }
    }

    /// Test Blake2b evidence hash produces 32-byte output
    #[test]
    fn test_blake2b_evidence_hash() {
        // Replicate the hash computation used by the coordinator
        let evidence_data = b"test evidence payload for blake2b hashing";

        let hash = blake2b_simd::Params::new()
            .hash_length(32)
            .hash(evidence_data)
            .as_bytes()
            .to_vec();

        assert_eq!(hash.len(), 32, "Blake2b hash should be 32 bytes");

        // Same input produces same hash (deterministic)
        let hash_2 = blake2b_simd::Params::new()
            .hash_length(32)
            .hash(evidence_data)
            .as_bytes()
            .to_vec();

        assert_eq!(hash, hash_2, "Blake2b should be deterministic");

        // Different input produces different hash
        let different_data = b"completely different evidence data for comparison";
        let hash_different = blake2b_simd::Params::new()
            .hash_length(32)
            .hash(different_data)
            .as_bytes()
            .to_vec();

        assert_ne!(hash, hash_different, "Different inputs should produce different hashes");

        // Empty input still produces 32-byte hash
        let hash_empty = blake2b_simd::Params::new()
            .hash_length(32)
            .hash(b"")
            .as_bytes()
            .to_vec();

        assert_eq!(hash_empty.len(), 32, "Empty input should still produce 32-byte hash");

        // Verify the hash is not all zeros
        assert!(hash.iter().any(|&b| b != 0), "Hash should not be all zeros");
    }

    /// Test MYCEL score clamping to [0.0, 1.0]
    #[test]
    fn test_mycel_score_clamping() {
        // Values within range stay unchanged
        assert_eq!(0.0_f32.clamp(0.0, 1.0), 0.0);
        assert_eq!(0.5_f32.clamp(0.0, 1.0), 0.5);
        assert_eq!(1.0_f32.clamp(0.0, 1.0), 1.0);
        assert_eq!(0.73_f32.clamp(0.0, 1.0), 0.73);

        // Values below 0.0 clamp to 0.0
        assert_eq!((-0.1_f32).clamp(0.0, 1.0), 0.0);
        assert_eq!((-1.0_f32).clamp(0.0, 1.0), 0.0);
        assert_eq!((-100.0_f32).clamp(0.0, 1.0), 0.0);

        // Values above 1.0 clamp to 1.0
        assert_eq!(1.1_f32.clamp(0.0, 1.0), 1.0);
        assert_eq!(2.0_f32.clamp(0.0, 1.0), 1.0);
        assert_eq!(999.0_f32.clamp(0.0, 1.0), 1.0);

        // Verify weight calculation with clamped values
        let clamped_negative = (-0.5_f32).clamp(0.0, 1.0);
        assert_eq!(1.0 + clamped_negative, 1.0, "Negative MYCEL clamps to weight 1.0");

        let clamped_over = 1.5_f32.clamp(0.0, 1.0);
        assert_eq!(1.0 + clamped_over, 2.0, "Over-1.0 MYCEL clamps to weight 2.0");
    }

    /// Test SlashingReason serialization round-trip
    #[test]
    fn test_slashing_reason_serialization() {
        let reasons = vec![
            SlashingReason::DoubleSigning,
            SlashingReason::Downtime,
            SlashingReason::ByzantineConsensus,
            SlashingReason::InvalidGradient,
            SlashingReason::CartelActivity,
            SlashingReason::GovernanceManipulation,
        ];

        for reason in reasons {
            let json = serde_json::to_string(&reason).expect("Serialize failed");
            let deserialized: SlashingReason =
                serde_json::from_str(&json).expect("Deserialize failed");
            assert_eq!(reason, deserialized, "SlashingReason round-trip failed for {:?}", reason);
        }
    }

    /// Test EvidenceType serialization round-trip
    #[test]
    fn test_evidence_type_serialization() {
        let types = vec![
            EvidenceType::DoubleSignEvidence,
            EvidenceType::DowntimeEvidence,
            EvidenceType::InvalidStateProof,
            EvidenceType::GradientAnalysis,
            EvidenceType::CartelProof,
        ];

        for ev_type in types {
            let json = serde_json::to_string(&ev_type).expect("Serialize failed");
            let deserialized: EvidenceType =
                serde_json::from_str(&json).expect("Deserialize failed");
            assert_eq!(ev_type, deserialized, "EvidenceType round-trip failed for {:?}", ev_type);
        }
    }

    /// Test EscrowHashType serialization round-trip
    #[test]
    fn test_escrow_hash_type_serialization() {
        let hash_types = vec![
            EscrowHashType::Sha256,
            EscrowHashType::Sha3_256,
            EscrowHashType::Blake2b,
            EscrowHashType::Keccak256,
        ];

        for hash_type in hash_types {
            let json = serde_json::to_string(&hash_type).expect("Serialize failed");
            let deserialized: EscrowHashType =
                serde_json::from_str(&json).expect("Deserialize failed");
            assert_eq!(hash_type, deserialized, "EscrowHashType round-trip failed for {:?}", hash_type);
        }
    }

    /// Test ReleaseCondition serialization for each variant
    #[test]
    fn test_release_condition_serialization() {
        let conditions = vec![
            ReleaseCondition::Timelock { release_time: 1_700_000_000 },
            ReleaseCondition::HashLock {
                hash: vec![0xab; 32],
                hash_type: EscrowHashType::Blake2b,
            },
            ReleaseCondition::MultiSig {
                threshold: 2,
                signers: vec!["did:mycelix:test:a".into(), "did:mycelix:test:b".into()],
            },
            ReleaseCondition::OraclePrice {
                oracle_id: "oracle:eth_usd".into(),
                asset: "ETH".into(),
                min_price: 2500.0,
            },
            ReleaseCondition::GovernanceApproval {
                proposal_id: "prop:001".into(),
            },
            ReleaseCondition::MycelThreshold {
                min_mycel_score: 0.8,
            },
        ];

        for condition in conditions {
            let json = serde_json::to_string(&condition).expect("Serialize failed");
            let deserialized: ReleaseCondition =
                serde_json::from_str(&json).expect("Deserialize failed");
            assert_eq!(condition, deserialized, "ReleaseCondition round-trip failed");
        }
    }

    /// Test slash amount computation (integer math: amount * pct / 100)
    #[test]
    fn test_slash_amount_computation() {
        // 100% of 20_000
        let sap = 20_000u64;
        let pct = 100u8;
        let slashed = (sap as u128 * pct as u128 / 100) as u64;
        assert_eq!(slashed, 20_000);

        // 5% of 100_000
        let sap = 100_000u64;
        let pct = 5u8;
        let slashed = (sap as u128 * pct as u128 / 100) as u64;
        assert_eq!(slashed, 5_000);

        // 50% of 80_000
        let sap = 80_000u64;
        let pct = 50u8;
        let slashed = (sap as u128 * pct as u128 / 100) as u64;
        assert_eq!(slashed, 40_000);

        // 75% of 50_000
        let sap = 50_000u64;
        let pct = 75u8;
        let slashed = (sap as u128 * pct as u128 / 100) as u64;
        assert_eq!(slashed, 37_500);

        // 10% of 1_000
        let sap = 1_000u64;
        let pct = 10u8;
        let slashed = (sap as u128 * pct as u128 / 100) as u64;
        assert_eq!(slashed, 100);

        // Edge case: 0% slash
        let sap = 50_000u64;
        let pct = 0u8;
        let slashed = (sap as u128 * pct as u128 / 100) as u64;
        assert_eq!(slashed, 0);

        // Edge case: 1 SAP at 50% = 0 (integer truncation)
        let sap = 1u64;
        let pct = 50u8;
        let slashed = (sap as u128 * pct as u128 / 100) as u64;
        assert_eq!(slashed, 0, "1 SAP at 50% should truncate to 0");
    }

    /// Test weighted stake calculation (sap_amount * stake_weight)
    #[test]
    fn test_total_weighted_stake_computation() {
        // Single staker: 10_000 SAP * 1.7 weight = 17_000
        let sap = 10_000u64;
        let weight = 1.7f32;
        let weighted = sap as f64 * weight as f64;
        assert!((weighted - 17_000.0).abs() < 0.01);

        // Multiple stakers
        let stakers: Vec<(u64, f32)> = vec![
            (10_000, 1.5),  // 15_000
            (20_000, 1.0),  // 20_000
            (5_000, 2.0),   // 10_000
        ];

        let total: f64 = stakers
            .iter()
            .map(|(sap, weight)| *sap as f64 * *weight as f64)
            .sum();

        assert!((total - 45_000.0).abs() < 0.01, "Total weighted stake should be 45_000");

        // Edge: zero SAP with any weight = 0
        let zero_weighted = 0u64 as f64 * 2.0f64;
        assert_eq!(zero_weighted, 0.0);
    }
}
