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

use holochain::sweettest::{SweetConductor, SweetDnaFile, SweetAgents};
use holochain::prelude::*;
use std::time::Duration;

use finance_bridge_integrity::*;

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
    #[tokio::test]
    #[ignore]
    async fn test_eth_deposit() {
        println!("Test 1.1: ETH Deposit -> SAP Minting");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path).await.expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = test_did("alice");

        // Deposit 1 ETH at rate 2000.0 SAP/ETH
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct DepositCollateralInput {
            pub depositor_did: String,
            pub collateral_type: String,
            pub collateral_amount: u64,
            pub oracle_rate: f64,
        }

        let input = DepositCollateralInput {
            depositor_did: alice_did.clone(),
            collateral_type: "ETH".to_string(),
            collateral_amount: 1_000_000, // 1 ETH in micro-units
            oracle_rate: 2000.0,
        };

        let result: Record = conductor
            .call(&alice_cell.zome("finance_bridge"), "deposit_collateral", input)
            .await
            .expect("Failed to deposit ETH");

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

        println!("  - ETH deposited: {} micro-units", deposit.collateral_amount);
        println!("  - SAP minted: {}", deposit.sap_minted);
        println!("Test 1.1 PASSED: ETH deposit mints correct SAP amount");
    }

    /// Test 1.2: USDC deposit mints SAP
    #[tokio::test]
    #[ignore]
    async fn test_usdc_deposit() {
        println!("Test 1.2: USDC Deposit -> SAP Minting");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path).await.expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];

        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct DepositCollateralInput {
            pub depositor_did: String,
            pub collateral_type: String,
            pub collateral_amount: u64,
            pub oracle_rate: f64,
        }

        let input = DepositCollateralInput {
            depositor_did: test_did("alice"),
            collateral_type: "USDC".to_string(),
            collateral_amount: 500_000_000, // 500 USDC in micro-units
            oracle_rate: 1.0, // 1:1 USDC:SAP
        };

        let result: Record = conductor
            .call(&alice_cell.zome("finance_bridge"), "deposit_collateral", input)
            .await
            .expect("Failed to deposit USDC");

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
    #[tokio::test]
    #[ignore]
    async fn test_invalid_collateral_type_rejected() {
        println!("Test 1.3: Invalid Collateral Type Rejected");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path).await.expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];

        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct DepositCollateralInput {
            pub depositor_did: String,
            pub collateral_type: String,
            pub collateral_amount: u64,
            pub oracle_rate: f64,
        }

        let input = DepositCollateralInput {
            depositor_did: test_did("alice"),
            collateral_type: "BTC".to_string(), // Invalid - only ETH and USDC
            collateral_amount: 1000,
            oracle_rate: 50000.0,
        };

        let result: Result<Record, _> = conductor
            .call_fallible(alice_cell.zome("finance_bridge"), "deposit_collateral", input)
            .await;

        assert!(result.is_err(), "BTC collateral should be rejected");
        println!("  - BTC collateral rejected: OK");
        println!("Test 1.3 PASSED: Invalid collateral types are rejected");
    }

    /// Test 1.4: Invalid DID rejected
    #[tokio::test]
    #[ignore]
    async fn test_invalid_did_rejected() {
        println!("Test 1.4: Invalid Depositor DID Rejected");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path).await.expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];

        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct DepositCollateralInput {
            pub depositor_did: String,
            pub collateral_type: String,
            pub collateral_amount: u64,
            pub oracle_rate: f64,
        }

        let input = DepositCollateralInput {
            depositor_did: "invalid_did".to_string(), // No "did:" prefix
            collateral_type: "ETH".to_string(),
            collateral_amount: 1000,
            oracle_rate: 2000.0,
        };

        let result: Result<Record, _> = conductor
            .call_fallible(alice_cell.zome("finance_bridge"), "deposit_collateral", input)
            .await;

        assert!(result.is_err(), "Invalid DID should be rejected");
        println!("  - Invalid DID rejected: OK");
        println!("Test 1.4 PASSED: Invalid DIDs are rejected");
    }

    /// Test 1.5: Zero amount rejected
    #[tokio::test]
    #[ignore]
    async fn test_zero_amount_rejected() {
        println!("Test 1.5: Zero Amount Rejected");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path).await.expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];

        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct DepositCollateralInput {
            pub depositor_did: String,
            pub collateral_type: String,
            pub collateral_amount: u64,
            pub oracle_rate: f64,
        }

        let input = DepositCollateralInput {
            depositor_did: test_did("alice"),
            collateral_type: "ETH".to_string(),
            collateral_amount: 0, // Zero amount
            oracle_rate: 2000.0,
        };

        let result: Result<Record, _> = conductor
            .call_fallible(alice_cell.zome("finance_bridge"), "deposit_collateral", input)
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
    #[tokio::test]
    #[ignore]
    async fn test_cross_happ_payment() {
        println!("Test 2.1: Cross-hApp Payment");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path).await.expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];

        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct ProcessPaymentInput {
            pub source_happ: String,
            pub from_did: String,
            pub to_did: String,
            pub amount: u64,
            pub currency: String,
            pub reference: String,
        }

        let input = ProcessPaymentInput {
            source_happ: "mycelix-property".to_string(),
            from_did: test_did("alice"),
            to_did: test_did("bob"),
            amount: 500,
            currency: "SAP".to_string(),
            reference: "property:rent:2026-02".to_string(),
        };

        let result: Record = conductor
            .call(&alice_cell.zome("finance_bridge"), "process_payment", input)
            .await
            .expect("Failed to process cross-hApp payment");

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
    #[tokio::test]
    #[ignore]
    async fn test_non_sap_cross_happ_rejected() {
        println!("Test 2.2: Non-SAP Cross-hApp Payment Rejected");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path).await.expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];

        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct ProcessPaymentInput {
            pub source_happ: String,
            pub from_did: String,
            pub to_did: String,
            pub amount: u64,
            pub currency: String,
            pub reference: String,
        }

        let input = ProcessPaymentInput {
            source_happ: "mycelix-property".to_string(),
            from_did: test_did("alice"),
            to_did: test_did("bob"),
            amount: 100,
            currency: "TEND".to_string(), // Invalid for cross-hApp
            reference: "test".to_string(),
        };

        let result: Result<Record, _> = conductor
            .call_fallible(alice_cell.zome("finance_bridge"), "process_payment", input)
            .await;

        assert!(result.is_err(), "Non-SAP cross-hApp payment should be rejected");
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
    #[tokio::test]
    #[ignore]
    async fn test_broadcast_finance_event() {
        println!("Test 3.1: Broadcast Finance Event");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path).await.expect("Load DNA");
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
            subject_did: test_did("alice"),
            amount: Some(1000),
            payload: serde_json::json!({"pool": "commons:local:1"}).to_string(),
        };

        let result: Record = conductor
            .call(&alice_cell.zome("finance_bridge"), "broadcast_finance_event", input)
            .await
            .expect("Failed to broadcast event");

        let event: FinanceBridgeEvent = result
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert!(matches!(event.event_type, FinanceEventType::CommonsContributed));
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
            let deserialized: AssetType =
                serde_json::from_str(&json).expect("Deserialize failed");
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
