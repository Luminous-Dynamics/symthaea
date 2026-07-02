// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Bridge Zome Integration Tests
//!
//! Tests for the Finance Bridge zome covering:
//! - Collateral bridge deposits (ETH/USDC -> SAP)
//! - Collateral redemption
//! - Rate limiting (5% of vault per day per member)
//! - Validation (DID format, collateral type, amounts)
//! - Cross-hApp payments
//!
//! ## Running Tests
//!
//! ```bash
//! cargo test --test bridge_test
//! cargo test --test bridge_test -- --ignored  # Full integration tests
//! ```

use holochain::prelude::*;
use holochain::sweettest::*;
use std::time::Duration;

use finance_bridge_integrity::*;
use finance_wire_types::{
    AssetType as WireAssetType, DepositCollateralInput, GetPaymentHistoryInput,
    ProcessPaymentInput, RegisterCollateralInput,
};

mod test_helpers {
    pub const TEST_DID_PREFIX: &str = "did:mycelix:test:";

    pub fn test_did(suffix: &str) -> String {
        format!("{}{}", TEST_DID_PREFIX, suffix)
    }
}

use test_helpers::*;

// ============================================================================
// Section 1: Collateral Bridge Deposit Tests
// ============================================================================

#[cfg(test)]
mod bridge_deposits {
    use super::*;

    /// Test 1.1: Basic ETH deposit mints SAP
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_eth_deposit() {
        println!("Test 1.1: ETH Deposit -> SAP Minting");

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
        let alice_did = format!("did:mycelix:{}", agents[0]);

        let input = DepositCollateralInput {
            depositor_did: alice_did.clone(),
            collateral_type: "ETH".to_string(),
            collateral_amount: 1_000_000, // 1 ETH in micro-units
            oracle_rate: 2000.0,
        };

        let result: Record = conductor
            .call(
                &alice_cell.zome("finance_bridge"),
                "deposit_collateral",
                input,
            )
            .await;

        let deposit: CollateralBridgeDeposit = result
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert_eq!(deposit.collateral_type, "ETH");
        assert_eq!(deposit.collateral_amount, 1_000_000);
        assert_eq!(deposit.sap_minted, 2_000_000_000); // 1M * 2000
        assert_eq!(deposit.depositor_did, alice_did);
        assert!(matches!(deposit.status, BridgeDepositStatus::Pending));

        println!(
            "  - ETH deposited: {} micro-units",
            deposit.collateral_amount
        );
        println!("  - SAP minted: {}", deposit.sap_minted);
        println!("Test 1.1 PASSED: ETH deposit mints correct SAP amount");
    }

    /// Test 1.2: USDC deposit mints SAP
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_usdc_deposit() {
        println!("Test 1.2: USDC Deposit -> SAP Minting");

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

        let input = DepositCollateralInput {
            depositor_did: format!("did:mycelix:{}", agents[0]),
            collateral_type: "USDC".to_string(),
            collateral_amount: 500_000_000, // 500 USDC in micro-units
            oracle_rate: 1.0,               // 1:1 USDC:SAP
        };

        let result: Record = conductor
            .call(
                &alice_cell.zome("finance_bridge"),
                "deposit_collateral",
                input,
            )
            .await;

        let deposit: CollateralBridgeDeposit = result
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert_eq!(deposit.collateral_type, "USDC");
        assert_eq!(deposit.sap_minted, 500_000_000);

        println!("  - USDC deposited: {}", deposit.collateral_amount);
        println!("  - SAP minted: {}", deposit.sap_minted);
        println!("Test 1.2 PASSED: USDC deposit works");
    }

    /// Test 1.3: Invalid collateral type rejected
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_invalid_collateral_type_rejected() {
        println!("Test 1.3: Invalid Collateral Type Rejected");

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

        let input = DepositCollateralInput {
            depositor_did: format!("did:mycelix:{}", agents[0]),
            collateral_type: "BTC".to_string(), // Invalid - only ETH and USDC
            collateral_amount: 1000,
            oracle_rate: 50000.0,
        };

        let result: Result<Record, _> = conductor
            .call_fallible(
                &alice_cell.zome("finance_bridge"),
                "deposit_collateral",
                input,
            )
            .await;

        assert!(result.is_err(), "BTC collateral should be rejected");
        println!("  - BTC collateral rejected: OK");
        println!("Test 1.3 PASSED: Invalid collateral types are rejected");
    }

    /// Test 1.4: Invalid DID rejected
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_invalid_did_rejected() {
        println!("Test 1.4: Invalid Depositor DID Rejected");

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

        let input = DepositCollateralInput {
            depositor_did: "invalid_did".to_string(), // No "did:" prefix
            collateral_type: "ETH".to_string(),
            collateral_amount: 1000,
            oracle_rate: 2000.0,
        };

        let result: Result<Record, _> = conductor
            .call_fallible(
                &alice_cell.zome("finance_bridge"),
                "deposit_collateral",
                input,
            )
            .await;

        assert!(result.is_err(), "Invalid DID should be rejected");
        println!("  - Invalid DID rejected: OK");
        println!("Test 1.4 PASSED: Invalid DIDs are rejected");
    }

    /// Test 1.5: Zero amount rejected
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_zero_amount_rejected() {
        println!("Test 1.5: Zero Amount Rejected");

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

        let input = DepositCollateralInput {
            depositor_did: format!("did:mycelix:{}", agents[0]),
            collateral_type: "ETH".to_string(),
            collateral_amount: 0, // Zero amount
            oracle_rate: 2000.0,
        };

        let result: Result<Record, _> = conductor
            .call_fallible(
                &alice_cell.zome("finance_bridge"),
                "deposit_collateral",
                input,
            )
            .await;

        assert!(result.is_err(), "Zero amount should be rejected");
        println!("  - Zero amount rejected: OK");
        println!("Test 1.5 PASSED: Zero amounts are rejected");
    }
}

// ============================================================================
// Section 2: Cross-hApp Payment Tests
// ============================================================================

#[cfg(test)]
mod cross_happ_payments {
    use super::*;

    /// Test 2.1: Basic cross-hApp payment
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_cross_happ_payment() {
        println!("Test 2.1: Cross-hApp Payment");

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

        let input = ProcessPaymentInput {
            source_happ: "mycelix-property".to_string(),
            from_did: format!("did:mycelix:{}", agents[0]),
            to_did: format!("did:mycelix:{}", agents[1]),
            amount: 500,
            currency: "SAP".to_string(),
            reference: "property:rent:2026-02".to_string(),
        };

        let result: Record = conductor
            .call(&alice_cell.zome("finance_bridge"), "process_payment", input)
            .await;

        let payment: CrossHappPayment = result
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert_eq!(payment.source_happ, "mycelix-property");
        assert_eq!(payment.amount, 500);
        assert_eq!(payment.currency, "SAP");

        println!("  - Cross-hApp payment processed: {} SAP", payment.amount);
        println!("  - Source hApp: {}", payment.source_happ);
        println!("Test 2.1 PASSED: Cross-hApp payment works");
    }

    /// Test 2.2: Non-SAP currency rejected for cross-hApp payment
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_non_sap_cross_happ_rejected() {
        println!("Test 2.2: Non-SAP Cross-hApp Payment Rejected");

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

        let input = ProcessPaymentInput {
            source_happ: "mycelix-property".to_string(),
            from_did: format!("did:mycelix:{}", agents[0]),
            to_did: format!("did:mycelix:{}", agents[1]),
            amount: 100,
            currency: "TEND".to_string(), // Invalid for cross-hApp
            reference: "test".to_string(),
        };

        let result: Result<Record, _> = conductor
            .call_fallible(&alice_cell.zome("finance_bridge"), "process_payment", input)
            .await;

        assert!(
            result.is_err(),
            "Non-SAP cross-hApp payment should be rejected"
        );
        println!("  - TEND cross-hApp payment rejected: OK");
        println!("Test 2.2 PASSED: Non-SAP currencies rejected for cross-hApp payments");
    }
}

// ============================================================================
// Section 3: Finance Event Tests
// ============================================================================

#[cfg(test)]
mod finance_events {
    use super::*;

    /// Test 3.1: Broadcast and retrieve finance events
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_broadcast_finance_event() {
        println!("Test 3.1: Broadcast Finance Event");

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

        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct BroadcastFinanceEventInput {
            pub event_type: FinanceEventType,
            pub subject_did: String,
            pub amount: Option<u64>,
            pub payload: String,
        }

        let input = BroadcastFinanceEventInput {
            event_type: FinanceEventType::CommonsContributed,
            subject_did: format!("did:mycelix:{}", agents[0]),
            amount: Some(1000),
            payload: serde_json::json!({"pool": "commons:local:1"}).to_string(),
        };

        let result: Record = conductor
            .call(
                &alice_cell.zome("finance_bridge"),
                "broadcast_finance_event",
                input,
            )
            .await;

        let event: FinanceBridgeEvent = result
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert!(matches!(
            event.event_type,
            FinanceEventType::CommonsContributed
        ));
        assert_eq!(event.amount, Some(1000));

        println!("  - Event type: {:?}", event.event_type);
        println!("  - Amount: {:?}", event.amount);
        println!("Test 3.1 PASSED: Finance event broadcast works");
    }
}

// ============================================================================
// Unit Tests (No Conductor Required)
// ============================================================================

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_bridge_deposit_status_serialization() {
        let statuses = vec![
            BridgeDepositStatus::Pending,
            BridgeDepositStatus::Confirmed,
            BridgeDepositStatus::Redeemed,
            BridgeDepositStatus::Failed,
        ];

        for status in statuses {
            let json = serde_json::to_string(&status).expect("Serialize failed");
            let deserialized: BridgeDepositStatus =
                serde_json::from_str(&json).expect("Deserialize failed");
            assert_eq!(status, deserialized, "Status round-trip failed");
        }
    }

    #[test]
    fn test_payment_status_serialization() {
        let statuses = vec![
            PaymentStatus::Pending,
            PaymentStatus::Processing,
            PaymentStatus::Completed,
            PaymentStatus::Failed,
            PaymentStatus::Refunded,
            PaymentStatus::Disputed,
        ];

        for status in statuses {
            let json = serde_json::to_string(&status).expect("Serialize failed");
            let deserialized: PaymentStatus =
                serde_json::from_str(&json).expect("Deserialize failed");
            assert_eq!(status, deserialized, "Status round-trip failed");
        }
    }

    #[test]
    fn test_finance_event_type_serialization() {
        let event_types = vec![
            FinanceEventType::PaymentCompleted,
            FinanceEventType::CollateralPledged,
            FinanceEventType::CollateralReleased,
            FinanceEventType::CollateralDeposited,
            FinanceEventType::CollateralRedeemed,
            FinanceEventType::CommonsContributed,
        ];

        for event_type in event_types {
            let json = serde_json::to_string(&event_type).expect("Serialize failed");
            let deserialized: FinanceEventType =
                serde_json::from_str(&json).expect("Deserialize failed");
            assert_eq!(event_type, deserialized, "EventType round-trip failed");
        }
    }

    #[test]
    fn test_asset_type_serialization() {
        let asset_types = vec![
            AssetType::RealEstate,
            AssetType::Vehicle,
            AssetType::Cryptocurrency,
            AssetType::EnergyAsset,
            AssetType::Equipment,
            AssetType::Other("Carbon Credit".to_string()),
        ];

        for asset_type in asset_types {
            let json = serde_json::to_string(&asset_type).expect("Serialize failed");
            let deserialized: AssetType = serde_json::from_str(&json).expect("Deserialize failed");
            assert_eq!(asset_type, deserialized, "AssetType round-trip failed");
        }
    }

    #[test]
    fn test_collateral_status_serialization() {
        let statuses = vec![
            CollateralStatus::Available,
            CollateralStatus::Pledged,
            CollateralStatus::Frozen,
            CollateralStatus::Released,
        ];

        for status in statuses {
            let json = serde_json::to_string(&status).expect("Serialize failed");
            let deserialized: CollateralStatus =
                serde_json::from_str(&json).expect("Deserialize failed");
            assert_eq!(status, deserialized, "CollateralStatus round-trip failed");
        }
    }

    #[test]
    fn test_sap_minting_computation() {
        // ETH at 2000 SAP/ETH rate
        let collateral_amount: u64 = 1_000_000; // 1 ETH in micro
        let oracle_rate: f64 = 2000.0;
        let sap_minted = (collateral_amount as f64 * oracle_rate) as u64;
        assert_eq!(sap_minted, 2_000_000_000);

        // USDC at 1:1 rate
        let usdc_amount: u64 = 100_000;
        let usdc_rate: f64 = 1.0;
        let usdc_sap = (usdc_amount as f64 * usdc_rate) as u64;
        assert_eq!(usdc_sap, 100_000);

        // Zero collateral => zero SAP
        let zero_sap = (0u64 as f64 * 2000.0) as u64;
        assert_eq!(zero_sap, 0);
    }

    #[test]
    fn test_rate_limit_computation() {
        // 5% of vault per day
        let vault_total: u64 = 1_000_000;
        let daily_limit = (vault_total as f64 * 0.05) as u64;
        assert_eq!(daily_limit, 50_000);

        // Edge case: empty vault allows first deposit (bootstrap)
        let empty_vault: u64 = 0;
        let empty_limit = (empty_vault as f64 * 0.05) as u64;
        assert_eq!(empty_limit, 0); // But code has bootstrap bypass
    }
}

// ============================================================================
// Section 4: Collateral Registration Tests
// ============================================================================

#[cfg(test)]
mod collateral_registration {
    use super::*;

    /// Test 4.1: Register a RealEstate collateral asset
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_register_collateral() {
        println!("Test 4.1: Register Collateral");

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
        let alice_did = format!("did:mycelix:{}", agents[0]);

        let input = RegisterCollateralInput {
            owner_did: alice_did.clone(),
            source_happ: "mycelix-property".to_string(),
            asset_type: WireAssetType::RealEstate,
            asset_id: "property:lot:42".to_string(),
            value_estimate: 500_000_000, // 500k SAP in micro-units
            currency: "SAP".to_string(),
        };

        let result: Record = conductor
            .call(
                &alice_cell.zome("finance_bridge"),
                "register_collateral",
                input,
            )
            .await;

        let collateral: CollateralRegistration = result
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert_eq!(collateral.owner_did, alice_did, "Owner DID mismatch");
        assert!(
            matches!(collateral.asset_type, AssetType::RealEstate),
            "Asset type should be RealEstate"
        );
        assert_eq!(collateral.asset_id, "property:lot:42", "Asset ID mismatch");
        assert_eq!(
            collateral.value_estimate, 500_000_000,
            "Value estimate mismatch"
        );
        assert!(
            matches!(collateral.status, CollateralStatus::Available),
            "Status should be Available"
        );

        println!("  - Collateral registered: {}", collateral.id);
        println!("  - Asset type: {:?}", collateral.asset_type);
        println!("  - Asset ID: {}", collateral.asset_id);
        println!("  - Value: {}", collateral.value_estimate);
        println!("Test 4.1 PASSED: Collateral registration works");
    }

    /// Test 4.2: Invalid collateral registration with empty asset_id
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_invalid_collateral_registration() {
        println!("Test 4.2: Invalid Collateral Registration");

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

        let input = RegisterCollateralInput {
            owner_did: format!("did:mycelix:{}", agents[0]),
            source_happ: "mycelix-property".to_string(),
            asset_type: WireAssetType::RealEstate,
            asset_id: "".to_string(), // Empty asset_id - should fail validation
            value_estimate: 100_000,
            currency: "SAP".to_string(),
        };

        let result: Result<Record, _> = conductor
            .call_fallible(
                &alice_cell.zome("finance_bridge"),
                "register_collateral",
                input,
            )
            .await;

        match result {
            Err(e) => {
                let error_msg = format!("{:?}", e);
                assert!(
                    error_msg.contains("asset")
                        || error_msg.contains("empty")
                        || error_msg.contains("Invalid")
                        || error_msg.contains("validation"),
                    "Should reject empty asset_id, got: {}",
                    error_msg
                );
                println!("  - Empty asset_id rejected: OK");
            }
            Ok(_) => panic!("Should have rejected registration with empty asset_id"),
        }

        println!("Test 4.2 PASSED: Invalid collateral registration is rejected");
    }
}

// ============================================================================
// Section 5: Redemption Tests
// ============================================================================

#[cfg(test)]
mod redemption_tests {
    use super::*;

    /// Test 5.1: Deposit ETH and redeem collateral
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_redeem_collateral() {
        println!("Test 5.1: Redeem Collateral");

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
        let alice_did = format!("did:mycelix:{}", agents[0]);

        let deposit_input = DepositCollateralInput {
            depositor_did: alice_did.clone(),
            collateral_type: "ETH".to_string(),
            collateral_amount: 500_000, // 0.5 ETH in micro-units
            oracle_rate: 2000.0,
        };

        let deposit_record: Record = conductor
            .call(
                &alice_cell.zome("finance_bridge"),
                "deposit_collateral",
                deposit_input,
            )
            .await;

        let deposit: CollateralBridgeDeposit = deposit_record
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        let deposit_id = deposit.id.clone();
        println!("  - Deposit ID: {}", deposit_id);
        println!("  - SAP minted: {}", deposit.sap_minted);

        // Redeem the collateral
        let redeemed_record: Record = conductor
            .call(
                &alice_cell.zome("finance_bridge"),
                "redeem_collateral",
                deposit_id.clone(),
            )
            .await;

        let redeemed: CollateralBridgeDeposit = redeemed_record
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert!(
            matches!(redeemed.status, BridgeDepositStatus::Redeemed),
            "Redeemed deposit should have Redeemed status"
        );

        println!("  - Redemption status: {:?}", redeemed.status);
        println!("Test 5.1 PASSED: Collateral redemption works");
    }

    /// Test 5.2: Deposit rate limit enforcement
    ///
    /// Makes an initial deposit (bootstraps vault), confirms it, makes a second
    /// deposit, then attempts a third that would exceed 5% of vault value in 24h.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_deposit_rate_limit_enforcement() {
        println!("Test 5.2: Deposit Rate Limit Enforcement");

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
        let alice_did = format!("did:mycelix:{}", agents[0]);

        // First deposit: bootstraps the vault (empty vault bypasses rate limit)
        let input1 = DepositCollateralInput {
            depositor_did: alice_did.clone(),
            collateral_type: "ETH".to_string(),
            collateral_amount: 1_000_000,
            oracle_rate: 2000.0,
        };

        let record1: Record = conductor
            .call(
                &alice_cell.zome("finance_bridge"),
                "deposit_collateral",
                input1,
            )
            .await;

        let deposit1: CollateralBridgeDeposit = record1
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");
        println!(
            "  - First deposit (bootstrap): {} SAP minted",
            deposit1.sap_minted
        );

        // Confirm the first deposit so it counts toward vault total
        let _: Record = conductor
            .call(
                &alice_cell.zome("finance_bridge"),
                "confirm_deposit",
                deposit1.id.clone(),
            )
            .await;
        println!("  - First deposit confirmed (vault now has confirmed SAP)");

        // Second deposit: should succeed if within 5% of vault
        let input2 = DepositCollateralInput {
            depositor_did: alice_did.clone(),
            collateral_type: "USDC".to_string(),
            collateral_amount: 10_000, // Small amount
            oracle_rate: 1.0,
        };

        let _: Record = conductor
            .call(
                &alice_cell.zome("finance_bridge"),
                "deposit_collateral",
                input2,
            )
            .await;
        println!("  - Second deposit succeeded (within daily limit)");

        // Third deposit: attempt a huge amount that exceeds 5% of vault
        // Vault is ~2B SAP (from first deposit), 5% = ~100M SAP.
        // Attempt to deposit much more than that.
        let input3 = DepositCollateralInput {
            depositor_did: alice_did.clone(),
            collateral_type: "ETH".to_string(),
            collateral_amount: 10_000_000, // 10 ETH at 2000 rate = 20B SAP (far exceeds 5%)
            oracle_rate: 2000.0,
        };

        let result: Result<Record, _> = conductor
            .call_fallible(
                &alice_cell.zome("finance_bridge"),
                "deposit_collateral",
                input3,
            )
            .await;

        match result {
            Err(e) => {
                let error_msg = format!("{:?}", e);
                assert!(
                    error_msg.contains("Rate limit exceeded") || error_msg.contains("rate limit"),
                    "Should fail with rate limit error, got: {}",
                    error_msg
                );
                println!("  - Third deposit rejected (rate limit exceeded): OK");
            }
            Ok(_) => panic!("Third deposit should have been rejected by rate limit"),
        }

        println!("Test 5.2 PASSED: Deposit rate limit enforcement works");
    }

    /// Test 5.3: Confirm deposit lifecycle
    ///
    /// Deposit -> Confirm -> Verify status transitions. Then try confirming
    /// again, which should fail because only Pending deposits can be confirmed.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_confirm_deposit_lifecycle() {
        println!("Test 5.3: Confirm Deposit Lifecycle");

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
        let alice_did = format!("did:mycelix:{}", agents[0]);

        // Create a deposit
        let input = DepositCollateralInput {
            depositor_did: alice_did.clone(),
            collateral_type: "ETH".to_string(),
            collateral_amount: 500_000,
            oracle_rate: 2000.0,
        };

        let record: Record = conductor
            .call(
                &alice_cell.zome("finance_bridge"),
                "deposit_collateral",
                input,
            )
            .await;

        let deposit: CollateralBridgeDeposit = record
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert!(
            matches!(deposit.status, BridgeDepositStatus::Pending),
            "New deposit should be Pending"
        );
        println!("  - Deposit created: {} (Pending)", deposit.id);

        // Confirm the deposit
        let confirmed_record: Record = conductor
            .call(
                &alice_cell.zome("finance_bridge"),
                "confirm_deposit",
                deposit.id.clone(),
            )
            .await;

        let confirmed: CollateralBridgeDeposit = confirmed_record
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert!(
            matches!(confirmed.status, BridgeDepositStatus::Confirmed),
            "Confirmed deposit should have Confirmed status"
        );
        assert!(
            confirmed.completed_at.is_some(),
            "Should have completion timestamp"
        );
        println!("  - Deposit confirmed: {:?}", confirmed.status);

        // Try to confirm again -- should fail
        let result: Result<Record, _> = conductor
            .call_fallible(
                &alice_cell.zome("finance_bridge"),
                "confirm_deposit",
                deposit.id.clone(),
            )
            .await;

        match result {
            Err(e) => {
                let error_msg = format!("{:?}", e);
                assert!(
                    error_msg.contains("only Pending deposits can be confirmed")
                        || error_msg.contains("Confirmed")
                        || error_msg.contains("Pending"),
                    "Should reject re-confirmation, got: {}",
                    error_msg
                );
                println!("  - Re-confirmation rejected: OK");
            }
            Ok(_) => panic!("Should have rejected confirming an already-confirmed deposit"),
        }

        println!("Test 5.3 PASSED: Confirm deposit lifecycle works correctly");
    }

    /// Test 5.4: Redeem unconfirmed deposit
    ///
    /// Create a deposit (Pending status), attempt to redeem it.
    /// Should fail because only confirmed deposits can be redeemed.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_redeem_unconfirmed_deposit() {
        println!("Test 5.4: Redeem Unconfirmed Deposit");

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
        let alice_did = format!("did:mycelix:{}", agents[0]);

        // Create a deposit (stays Pending -- do NOT confirm)
        let input = DepositCollateralInput {
            depositor_did: alice_did.clone(),
            collateral_type: "USDC".to_string(),
            collateral_amount: 100_000,
            oracle_rate: 1.0,
        };

        let record: Record = conductor
            .call(
                &alice_cell.zome("finance_bridge"),
                "deposit_collateral",
                input,
            )
            .await;

        let deposit: CollateralBridgeDeposit = record
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert!(
            matches!(deposit.status, BridgeDepositStatus::Pending),
            "Deposit should be Pending"
        );
        println!("  - Deposit created: {} (Pending)", deposit.id);

        // Attempt to redeem the Pending deposit -- should fail
        let result: Result<Record, _> = conductor
            .call_fallible(
                &alice_cell.zome("finance_bridge"),
                "redeem_collateral",
                deposit.id.clone(),
            )
            .await;

        match result {
            Err(e) => {
                let error_msg = format!("{:?}", e);
                assert!(
                    error_msg.contains("Only confirmed deposits can be redeemed")
                        || error_msg.contains("confirmed")
                        || error_msg.contains("Confirmed"),
                    "Should reject redeeming unconfirmed deposit, got: {}",
                    error_msg
                );
                println!("  - Redeem of Pending deposit rejected: OK");
            }
            Ok(_) => panic!("Should have rejected redeeming an unconfirmed deposit"),
        }

        println!("Test 5.4 PASSED: Unconfirmed deposits cannot be redeemed");
    }

    /// Test 5.5: Make 3 cross-hApp payments and query payment history
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_payment_history() {
        println!("Test 5.2: Payment History");

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
        let alice_did = format!("did:mycelix:{}", agents[0]);
        let bob_did = format!("did:mycelix:{}", agents[1]);

        // Make 3 cross-hApp payments
        for i in 0..3 {
            let input = ProcessPaymentInput {
                source_happ: "mycelix-property".to_string(),
                from_did: alice_did.clone(),
                to_did: bob_did.clone(),
                amount: 100 * (i + 1),
                currency: "SAP".to_string(),
                reference: format!("property:rent:2026-0{}", i + 1),
            };

            let _: Record = conductor
                .call(&alice_cell.zome("finance_bridge"), "process_payment", input)
                .await;

            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        println!("  - 3 cross-hApp payments processed");

        // Wait for DHT consistency
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Query payment history for Alice
        let history: Vec<Record> = conductor
            .call(
                &alice_cell.zome("finance_bridge"),
                "get_payment_history",
                GetPaymentHistoryInput {
                    did: alice_did.clone(),
                    limit: None,
                },
            )
            .await;

        assert_eq!(history.len(), 3, "Should have 3 payments in history");

        println!("  - Payment history count: {}", history.len());
        println!("Test 5.2 PASSED: Payment history returns correct count");
    }
}
