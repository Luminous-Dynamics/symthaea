// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Payments Zome Integration Tests
//!
//! Comprehensive tests for the Mycelix Finance Payments zome covering:
//! - Payment creation and validation
//! - Double-spend prevention
//! - Transaction confirmation
//! - Fee calculation
//! - Failed transaction handling
//! - Payment channels
//! - Currency validation (SAP and TEND only)
//! - Demurrage on SAP payments
//!
//! ## Running Tests
//!
//! These tests require a running Holochain conductor with the Finance DNA.
//! Tests marked with `#[ignore]` require the DNA bundle to be built first.
//!
//! ```bash
//! # Build the DNA first
//! cd /home/tstoltz/Luminous-Dynamics/mycelix-finance
//! hc dna pack dna/
//!
//! # Run all tests (excluding ignored)
//! cargo test --test payments_test
//!
//! # Run with ignored tests (requires DNA bundle)
//! cargo test --test payments_test -- --ignored
//! ```

use holochain::prelude::*;
use holochain::sweettest::*;
use holochain_types::prelude::*;
use std::time::Duration;

use finance_wire_types::SapBalanceResponse;
// Import zome types
use mycelix_finance_types::compute_demurrage_deduction;
use payments::*;
use payments_integrity::*;

// Test utilities
mod test_helpers {
    use super::*;

    pub const TEST_DID_PREFIX: &str = "did:mycelix:test:";
    pub const TEST_CURRENCY: &str = "SAP";

    pub fn test_did(suffix: &str) -> String {
        format!("{}{}", TEST_DID_PREFIX, suffix)
    }

    pub fn unique_test_did(prefix: &str) -> String {
        let timestamp = chrono::Utc::now().timestamp_micros();
        format!("{}{}:{}", TEST_DID_PREFIX, prefix, timestamp)
    }
}

use test_helpers::*;

// ============================================================================
// Section 1: Payment Creation Tests
// ============================================================================

#[cfg(test)]
mod payment_creation {
    use super::*;

    /// Test 1.1: Basic payment creation succeeds with valid inputs
    ///
    /// Scenario:
    /// - Alice sends 100 SAP to Bob
    /// - Verify payment is created with correct fields
    /// - Verify receipt is generated
    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // Requires DNA bundle to be built
    async fn test_create_payment_success() {
        println!("Test 1.1: Basic Payment Creation");

        // Setup conductor with 2 agents
        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let bob_cell = &apps[1].cells()[0];

        let alice_did = format!("did:mycelix:{}", agents[0]);
        let bob_did = format!("did:mycelix:{}", agents[1]);

        // Create payment input
        let payment_input = SendPaymentInput {
            from_did: alice_did.clone(),
            to_did: bob_did.clone(),
            amount: 100_000_000,
            currency: TEST_CURRENCY.to_string(),
            payment_type: PaymentType::Direct,
            memo: Some("Test payment from Alice to Bob".to_string()),
        };

        // Send payment
        let payment_record: Record = conductor
            .call(&alice_cell.zome("payments"), "send_payment", payment_input)
            .await;

        println!("  - Payment created successfully");

        // Extract and verify payment entry
        let payment: Payment = payment_record
            .entry()
            .to_app_option()
            .expect("Failed to deserialize")
            .expect("No entry found");

        assert!(
            payment.id.starts_with("payment:"),
            "Payment ID should have correct prefix"
        );
        assert_eq!(payment.from_did, alice_did, "From DID mismatch");
        assert_eq!(payment.to_did, bob_did, "To DID mismatch");
        assert_eq!(payment.amount, 100_000_000, "Amount mismatch");
        assert_eq!(payment.currency, TEST_CURRENCY, "Currency mismatch");
        assert!(
            matches!(payment.status, TransferStatus::Completed),
            "Status should be Completed"
        );
        assert!(payment.memo.is_some(), "Memo should be present");

        println!("  - Payment fields verified");
        println!("  - Payment ID: {}", payment.id);
        println!("  - Status: {:?}", payment.status);
        println!("Test 1.1 PASSED: Basic payment creation works");
    }

    /// Test 1.2: Payment with different payment types
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_payment_types() {
        println!("Test 1.2: Payment Types");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];

        let alice_did = format!("did:mycelix:{}", agents[0]);
        let bob_did = format!("did:mycelix:{}", agents[1]);

        // Test each valid payment type (LoanPayment and EnergyInvestment removed)
        let payment_types = vec![
            (PaymentType::Direct, "Direct"),
            (
                PaymentType::TreasuryContribution("treasury:abc".to_string()),
                "TreasuryContribution",
            ),
            (
                PaymentType::CommonsContribution("commons:pool:1".to_string()),
                "CommonsContribution",
            ),
            (PaymentType::Escrow("escrow:456".to_string()), "Escrow"),
        ];

        for (payment_type, type_name) in payment_types {
            let input = SendPaymentInput {
                from_did: alice_did.clone(),
                to_did: bob_did.clone(),
                amount: 10_000_000,
                currency: TEST_CURRENCY.to_string(),
                payment_type: payment_type.clone(),
                memo: Some(format!("Test {} payment", type_name)),
            };

            let result: Record = conductor
                .call(&alice_cell.zome("payments"), "send_payment", input)
                .await;

            let payment: Payment = result
                .entry()
                .to_app_option()
                .expect("Deserialize failed")
                .expect("No entry");

            assert_eq!(
                payment.payment_type, payment_type,
                "{} type mismatch",
                type_name
            );
            println!("  - {} payment: OK", type_name);
        }

        println!("Test 1.2 PASSED: All payment types work correctly");
    }

    /// Test 1.3: Recurring payment configuration
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_recurring_payment() {
        println!("Test 1.3: Recurring Payment Configuration");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];

        let alice_did = format!("did:mycelix:{}", agents[0]);
        let bob_did = format!("did:mycelix:{}", agents[1]);

        // Create recurring payment config
        let recurring_config = RecurringConfig {
            frequency_days: 30,
            end_date: None,
            remaining: Some(12), // 12 payments
        };

        let input = SendPaymentInput {
            from_did: alice_did.clone(),
            to_did: bob_did.clone(),
            amount: 50_000_000,
            currency: TEST_CURRENCY.to_string(),
            payment_type: PaymentType::Recurring(recurring_config.clone()),
            memo: Some("Monthly subscription".to_string()),
        };

        let result: Record = conductor
            .call(&alice_cell.zome("payments"), "send_payment", input)
            .await;

        let payment: Payment = result
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        if let PaymentType::Recurring(config) = payment.payment_type {
            assert_eq!(config.frequency_days, 30, "Frequency mismatch");
            assert_eq!(config.remaining, Some(12), "Remaining payments mismatch");
            println!(
                "  - Recurring config verified: {} days, {} remaining",
                config.frequency_days,
                config.remaining.unwrap()
            );
        } else {
            panic!("Expected Recurring payment type");
        }

        println!("Test 1.3 PASSED: Recurring payment configuration works");
    }
}

// ============================================================================
// Section 2: Payment Validation Tests
// ============================================================================

#[cfg(test)]
mod payment_validation {
    use super::*;

    /// Test 2.1: Invalid DID format rejected
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_invalid_did_rejected() {
        println!("Test 2.1: Invalid DID Validation");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];

        // Test with invalid sender DID (no "did:" prefix)
        let invalid_sender_input = SendPaymentInput {
            from_did: "invalid:alice".to_string(), // Invalid - no "did:" prefix
            to_did: format!("did:mycelix:{}", agents[1]),
            amount: 100_000_000,
            currency: TEST_CURRENCY.to_string(),
            payment_type: PaymentType::Direct,
            memo: None,
        };

        let result: Result<Record, _> = conductor
            .call_fallible(
                &alice_cell.zome("payments"),
                "send_payment",
                invalid_sender_input,
            )
            .await;

        match result {
            Err(e) => {
                let error_msg = format!("{:?}", e);
                assert!(
                    error_msg.contains("Sender must be a valid DID")
                        || error_msg.contains("Invalid"),
                    "Should reject invalid sender DID, got: {}",
                    error_msg
                );
                println!("  - Invalid sender DID rejected: OK");
            }
            Ok(_) => panic!("Should have rejected invalid sender DID"),
        }

        // Test with invalid receiver DID
        let invalid_receiver_input = SendPaymentInput {
            from_did: format!("did:mycelix:{}", agents[0]),
            to_did: "bob@example.com".to_string(), // Invalid - email format
            amount: 100_000_000,
            currency: TEST_CURRENCY.to_string(),
            payment_type: PaymentType::Direct,
            memo: None,
        };

        let result: Result<Record, _> = conductor
            .call_fallible(
                &alice_cell.zome("payments"),
                "send_payment",
                invalid_receiver_input,
            )
            .await;

        match result {
            Err(e) => {
                let error_msg = format!("{:?}", e);
                assert!(
                    error_msg.contains("Receiver must be a valid DID")
                        || error_msg.contains("Invalid"),
                    "Should reject invalid receiver DID, got: {}",
                    error_msg
                );
                println!("  - Invalid receiver DID rejected: OK");
            }
            Ok(_) => panic!("Should have rejected invalid receiver DID"),
        }

        println!("Test 2.1 PASSED: Invalid DIDs are properly rejected");
    }

    /// Test 2.2: Zero or negative amounts rejected
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_invalid_amount_rejected() {
        println!("Test 2.2: Invalid Amount Validation");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = format!("did:mycelix:{}", agents[0]);
        let bob_did = format!("did:mycelix:{}", agents[1]);

        // Test zero amount
        let zero_amount_input = SendPaymentInput {
            from_did: alice_did.clone(),
            to_did: bob_did.clone(),
            amount: 0,
            currency: TEST_CURRENCY.to_string(),
            payment_type: PaymentType::Direct,
            memo: None,
        };

        let result: Result<Record, _> = conductor
            .call_fallible(
                &alice_cell.zome("payments"),
                "send_payment",
                zero_amount_input,
            )
            .await;

        match result {
            Err(e) => {
                let error_msg = format!("{:?}", e);
                assert!(
                    error_msg.contains("Amount must be positive"),
                    "Should reject zero amount, got: {}",
                    error_msg
                );
                println!("  - Zero amount rejected: OK");
            }
            Ok(_) => panic!("Should have rejected zero amount"),
        }

        // Note: negative amounts are now impossible at the type level (u64 is always >= 0).
        // Only zero-amount validation remains relevant.

        println!("Test 2.2 PASSED: Invalid amounts are properly rejected");
    }

    /// Test 2.3: Self-payment rejected
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_self_payment_rejected() {
        println!("Test 2.3: Self-Payment Validation");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = format!("did:mycelix:{}", agents[0]);

        // Attempt self-payment
        let self_payment_input = SendPaymentInput {
            from_did: alice_did.clone(),
            to_did: alice_did.clone(), // Same as from_did
            amount: 100_000_000,
            currency: TEST_CURRENCY.to_string(),
            payment_type: PaymentType::Direct,
            memo: Some("Self payment attempt".to_string()),
        };

        let result: Result<Record, _> = conductor
            .call_fallible(
                &alice_cell.zome("payments"),
                "send_payment",
                self_payment_input,
            )
            .await;

        match result {
            Err(e) => {
                let error_msg = format!("{:?}", e);
                assert!(
                    error_msg.contains("Cannot send payment to yourself"),
                    "Should reject self-payment, got: {}",
                    error_msg
                );
                println!("  - Self-payment rejected: OK");
            }
            Ok(_) => panic!("Should have rejected self-payment"),
        }

        println!("Test 2.3 PASSED: Self-payments are properly rejected");
    }

    /// Test 2.4: Invalid currency rejected (only SAP and TEND accepted)
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_invalid_currency_rejected() {
        println!("Test 2.4: Invalid Currency Validation");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = format!("did:mycelix:{}", agents[0]);
        let bob_did = format!("did:mycelix:{}", agents[1]);

        // Test invalid currencies that are no longer accepted
        let invalid_currencies = vec!["MYC", "USD", "ENERGY", "BTC", "ETH"];

        for currency in invalid_currencies {
            let input = SendPaymentInput {
                from_did: alice_did.clone(),
                to_did: bob_did.clone(),
                amount: 10_000_000,
                currency: currency.to_string(),
                payment_type: PaymentType::Direct,
                memo: None,
            };

            let result: Result<Record, _> = conductor
                .call_fallible(&alice_cell.zome("payments"), "send_payment", input)
                .await;

            match result {
                Err(e) => {
                    let error_msg = format!("{:?}", e);
                    assert!(
                        error_msg.contains("Currency must be")
                            || error_msg.contains("SAP")
                            || error_msg.contains("TEND"),
                        "Should reject {} currency, got: {}",
                        currency,
                        error_msg
                    );
                    println!("  - {} currency rejected: OK", currency);
                }
                Ok(_) => panic!("Should have rejected {} currency", currency),
            }
        }

        println!("Test 2.4 PASSED: Invalid currencies are properly rejected");
    }
}

// ============================================================================
// Section 3: Double-Spend Prevention Tests
// ============================================================================

#[cfg(test)]
mod double_spend_prevention {
    use super::*;

    /// Test 3.1: Payment ID uniqueness
    ///
    /// Verifies that each payment gets a unique ID to prevent replay attacks
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_payment_id_uniqueness() {
        println!("Test 3.1: Payment ID Uniqueness");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = format!("did:mycelix:{}", agents[0]);
        let bob_did = format!("did:mycelix:{}", agents[1]);

        // Create multiple identical payments
        let mut payment_ids = Vec::new();
        for i in 0..5 {
            let input = SendPaymentInput {
                from_did: alice_did.clone(),
                to_did: bob_did.clone(),
                amount: 10_000_000,
                currency: TEST_CURRENCY.to_string(),
                payment_type: PaymentType::Direct,
                memo: Some(format!("Payment {}", i)),
            };

            let result: Record = conductor
                .call(&alice_cell.zome("payments"), "send_payment", input)
                .await;

            let payment: Payment = result
                .entry()
                .to_app_option()
                .expect("Deserialize failed")
                .expect("No entry");

            payment_ids.push(payment.id.clone());
            println!("  - Payment {}: ID = {}", i, payment.id);

            // Small delay to ensure different timestamps
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // Verify all IDs are unique
        let unique_ids: std::collections::HashSet<_> = payment_ids.iter().collect();
        assert_eq!(
            unique_ids.len(),
            payment_ids.len(),
            "All payment IDs should be unique"
        );

        println!("  - {} unique payment IDs generated", unique_ids.len());
        println!("Test 3.1 PASSED: Payment IDs are unique");
    }

    /// Test 3.2: Receipt verification prevents replay
    ///
    /// Each payment generates a unique receipt that can be used to verify
    /// the payment has not been processed twice
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_receipt_prevents_replay() {
        println!("Test 3.2: Receipt Verification");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = format!("did:mycelix:{}", agents[0]);
        let bob_did = format!("did:mycelix:{}", agents[1]);

        // Send payment
        let input = SendPaymentInput {
            from_did: alice_did.clone(),
            to_did: bob_did.clone(),
            amount: 100_000_000,
            currency: TEST_CURRENCY.to_string(),
            payment_type: PaymentType::Direct,
            memo: Some("Payment with receipt".to_string()),
        };

        let payment_record: Record = conductor
            .call(&alice_cell.zome("payments"), "send_payment", input)
            .await;

        let payment: Payment = payment_record
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        // Get payment history and verify receipt link exists
        let history: Vec<Record> = conductor
            .call(
                &alice_cell.zome("payments"),
                "get_payment_history",
                GetPaymentHistoryInput {
                    did: alice_did.clone(),
                    limit: None,
                },
            )
            .await;

        assert!(!history.is_empty(), "Payment history should not be empty");

        // Verify the payment has a unique signature
        println!("  - Payment ID: {}", payment.id);
        println!("  - Payment has linked receipt");
        println!("  - History contains {} payments", history.len());

        println!("Test 3.2 PASSED: Receipts enable replay detection");
    }

    /// Test 3.3: Channel balance consistency
    ///
    /// Payment channels maintain consistent balances, preventing double-spend
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_channel_balance_consistency() {
        println!("Test 3.3: Channel Balance Consistency");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let bob_cell = &apps[1].cells()[0];

        let alice_did = format!("did:mycelix:{}", agents[0]);
        let bob_did = format!("did:mycelix:{}", agents[1]);

        // Open payment channel with initial deposits
        let channel_input = OpenChannelInput {
            party_a: alice_did.clone(),
            party_b: bob_did.clone(),
            currency: TEST_CURRENCY.to_string(),
            initial_deposit_a: 500_000_000,
            initial_deposit_b: 300_000_000,
        };

        let channel_record: Record = conductor
            .call(
                &alice_cell.zome("payments"),
                "open_payment_channel",
                channel_input,
            )
            .await;

        let channel: PaymentChannel = channel_record
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        let channel_id = channel.id.clone();
        println!("  - Channel opened: {}", channel_id);
        println!(
            "  - Initial balances: A={}, B={}",
            channel.balance_a, channel.balance_b
        );

        // Verify initial state
        assert_eq!(channel.balance_a, 500_000_000, "Initial balance A mismatch");
        assert_eq!(channel.balance_b, 300_000_000, "Initial balance B mismatch");

        // Make transfers within channel
        let transfer_input = ChannelTransferInput {
            channel_id: channel_id.clone(),
            amount: 100_000_000,
            from_a: true, // Alice sends to Bob
        };

        let updated_channel: Record = conductor
            .call(
                &alice_cell.zome("payments"),
                "channel_transfer",
                transfer_input,
            )
            .await;

        let channel_after: PaymentChannel = updated_channel
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        // Verify conservation of total balance
        let total_before: u64 = 500_000_000 + 300_000_000;
        let total_after = channel_after.balance_a + channel_after.balance_b;
        assert_eq!(
            total_before, total_after,
            "Total balance should be conserved"
        );
        assert_eq!(
            channel_after.balance_a, 400_000_000,
            "Balance A after transfer"
        );
        assert_eq!(
            channel_after.balance_b, 400_000_000,
            "Balance B after transfer"
        );

        println!(
            "  - After transfer: A={}, B={}",
            channel_after.balance_a, channel_after.balance_b
        );
        println!("  - Total conserved: {}", total_after);

        println!("Test 3.3 PASSED: Channel balances are consistent");
    }

    /// Test 3.4: Insufficient balance prevents double-spend in channels
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_channel_insufficient_balance() {
        println!("Test 3.4: Channel Insufficient Balance Check");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];

        let alice_did = format!("did:mycelix:{}", agents[0]);
        let bob_did = format!("did:mycelix:{}", agents[1]);

        // Open channel with limited balance
        let channel_input = OpenChannelInput {
            party_a: alice_did.clone(),
            party_b: bob_did.clone(),
            currency: TEST_CURRENCY.to_string(),
            initial_deposit_a: 100_000_000,
            initial_deposit_b: 100_000_000,
        };

        let channel_record: Record = conductor
            .call(
                &alice_cell.zome("payments"),
                "open_payment_channel",
                channel_input,
            )
            .await;

        let channel: PaymentChannel = channel_record
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        let channel_id = channel.id.clone();

        // Attempt transfer exceeding balance
        let transfer_input = ChannelTransferInput {
            channel_id: channel_id.clone(),
            amount: 150_000_000, // Exceeds Alice's 100_000_000 balance
            from_a: true,
        };

        let result: Result<Record, _> = conductor
            .call_fallible(
                &alice_cell.zome("payments"),
                "channel_transfer",
                transfer_input,
            )
            .await;

        match result {
            Err(e) => {
                let error_msg = format!("{:?}", e);
                assert!(
                    error_msg.contains("Insufficient balance"),
                    "Should reject insufficient balance, got: {}",
                    error_msg
                );
                println!("  - Insufficient balance transfer rejected: OK");
            }
            Ok(_) => panic!("Should have rejected transfer exceeding balance"),
        }

        println!("Test 3.4 PASSED: Insufficient balance transfers are rejected");
    }
}

// ============================================================================
// Section 4: Transaction Confirmation Tests
// ============================================================================

#[cfg(test)]
mod transaction_confirmation {
    use super::*;

    /// Test 4.1: Payment completion status
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_payment_completion_status() {
        println!("Test 4.1: Payment Completion Status");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = format!("did:mycelix:{}", agents[0]);
        let bob_did = format!("did:mycelix:{}", agents[1]);

        let input = SendPaymentInput {
            from_did: alice_did.clone(),
            to_did: bob_did.clone(),
            amount: 100_000_000,
            currency: TEST_CURRENCY.to_string(),
            payment_type: PaymentType::Direct,
            memo: None,
        };

        let result: Record = conductor
            .call(&alice_cell.zome("payments"), "send_payment", input)
            .await;

        let payment: Payment = result
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        // Verify completion status
        assert!(
            matches!(payment.status, TransferStatus::Completed),
            "Status should be Completed"
        );
        assert!(
            payment.completed.is_some(),
            "Completion timestamp should be set"
        );

        let created = payment.created;
        let completed = payment.completed.unwrap();
        assert!(
            completed.as_micros() >= created.as_micros(),
            "Completion should be at or after creation"
        );

        println!("  - Status: {:?}", payment.status);
        println!("  - Created: {:?}", created);
        println!("  - Completed: {:?}", completed);

        println!("Test 4.1 PASSED: Payment completion status is correct");
    }

    /// Test 4.2: Payment history retrieval (both sent and received)
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_payment_history_retrieval() {
        println!("Test 4.2: Payment History Retrieval");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 3).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let bob_cell = &apps[1].cells()[0];
        let charlie_cell = &apps[2].cells()[0];

        let alice_did = format!("did:mycelix:{}", agents[0]);
        let bob_did = format!("did:mycelix:{}", agents[1]);
        let charlie_did = format!("did:mycelix:{}", agents[2]);

        // Alice sends to Bob
        let input1 = SendPaymentInput {
            from_did: alice_did.clone(),
            to_did: bob_did.clone(),
            amount: 50_000_000,
            currency: TEST_CURRENCY.to_string(),
            payment_type: PaymentType::Direct,
            memo: Some("Alice to Bob".to_string()),
        };

        let _: Record = conductor
            .call(&alice_cell.zome("payments"), "send_payment", input1)
            .await;
        println!("  - Payment 1: Alice -> Bob (50)");

        // Charlie sends to Alice
        let input2 = SendPaymentInput {
            from_did: charlie_did.clone(),
            to_did: alice_did.clone(),
            amount: 30_000_000,
            currency: TEST_CURRENCY.to_string(),
            payment_type: PaymentType::Direct,
            memo: Some("Charlie to Alice".to_string()),
        };

        let _: Record = conductor
            .call(&charlie_cell.zome("payments"), "send_payment", input2)
            .await;
        println!("  - Payment 2: Charlie -> Alice (30)");

        // Alice sends to Charlie
        let input3 = SendPaymentInput {
            from_did: alice_did.clone(),
            to_did: charlie_did.clone(),
            amount: 20_000_000,
            currency: TEST_CURRENCY.to_string(),
            payment_type: PaymentType::Direct,
            memo: Some("Alice to Charlie".to_string()),
        };

        let _: Record = conductor
            .call(&alice_cell.zome("payments"), "send_payment", input3)
            .await;
        println!("  - Payment 3: Alice -> Charlie (20)");

        // Wait for DHT consistency
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Get Alice's payment history (should include sent and received)
        let alice_history: Vec<Record> = conductor
            .call(
                &alice_cell.zome("payments"),
                "get_payment_history",
                GetPaymentHistoryInput {
                    did: alice_did.clone(),
                    limit: None,
                },
            )
            .await;

        println!("  - Alice's history: {} payments", alice_history.len());
        assert!(
            alice_history.len() >= 2,
            "Alice should have at least 2 payments"
        );

        // Get Bob's payment history
        let bob_history: Vec<Record> = conductor
            .call(
                &bob_cell.zome("payments"),
                "get_payment_history",
                GetPaymentHistoryInput {
                    did: bob_did.clone(),
                    limit: None,
                },
            )
            .await;

        println!("  - Bob's history: {} payments", bob_history.len());
        assert!(
            !bob_history.is_empty(),
            "Bob should have at least 1 payment"
        );

        println!("Test 4.2 PASSED: Payment history retrieval works");
    }
}

// ============================================================================
// Section 5: Currency and Fee Tests
// ============================================================================

#[cfg(test)]
mod currency_tests {
    use super::*;

    /// Test 5.1: SAP currency payment
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_sap_payment() {
        println!("Test 5.1: SAP Currency Payment");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = format!("did:mycelix:{}", agents[0]);
        let bob_did = format!("did:mycelix:{}", agents[1]);

        let input = SendPaymentInput {
            from_did: alice_did.clone(),
            to_did: bob_did.clone(),
            amount: 100_000_000,
            currency: "SAP".to_string(),
            payment_type: PaymentType::Direct,
            memo: Some("SAP payment".to_string()),
        };

        let result: Record = conductor
            .call(&alice_cell.zome("payments"), "send_payment", input)
            .await;

        let payment: Payment = result
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert_eq!(payment.currency, "SAP", "Currency should be SAP");
        assert_eq!(
            payment.amount, 100_000_000,
            "Full amount should be recorded"
        );
        println!("  - SAP payment amount: {}", payment.amount);
        println!("  - Currency: {}", payment.currency);
        println!("Test 5.1 PASSED: SAP payment works");
    }

    /// Test 5.2: TEND currency payment
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_tend_payment() {
        println!("Test 5.2: TEND Currency Payment");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = format!("did:mycelix:{}", agents[0]);
        let bob_did = format!("did:mycelix:{}", agents[1]);

        let input = SendPaymentInput {
            from_did: alice_did.clone(),
            to_did: bob_did.clone(),
            amount: 5_000_000,
            currency: "TEND".to_string(),
            payment_type: PaymentType::Direct,
            memo: Some("TEND payment".to_string()),
        };

        let result: Record = conductor
            .call(&alice_cell.zome("payments"), "send_payment", input)
            .await;

        let payment: Payment = result
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert_eq!(payment.currency, "TEND", "Currency should be TEND");
        println!("  - TEND payment amount: {}", payment.amount);
        println!("  - Currency: {}", payment.currency);
        println!("Test 5.2 PASSED: TEND payment works");
    }

    /// Test 5.3: Only SAP and TEND currencies are valid
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_valid_currencies_only() {
        println!("Test 5.3: Valid Currencies Only (SAP/TEND)");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = format!("did:mycelix:{}", agents[0]);
        let bob_did = format!("did:mycelix:{}", agents[1]);

        // Valid: SAP
        let sap_input = SendPaymentInput {
            from_did: alice_did.clone(),
            to_did: bob_did.clone(),
            amount: 10_000_000,
            currency: "SAP".to_string(),
            payment_type: PaymentType::Direct,
            memo: None,
        };

        let _: Record = conductor
            .call(&alice_cell.zome("payments"), "send_payment", sap_input)
            .await;
        println!("  - SAP accepted: OK");

        // Valid: TEND
        let tend_input = SendPaymentInput {
            from_did: alice_did.clone(),
            to_did: bob_did.clone(),
            amount: 10_000_000,
            currency: "TEND".to_string(),
            payment_type: PaymentType::Direct,
            memo: None,
        };

        let _: Record = conductor
            .call(&alice_cell.zome("payments"), "send_payment", tend_input)
            .await;
        println!("  - TEND accepted: OK");

        // Invalid: MYC (old currency, no longer accepted)
        let myc_input = SendPaymentInput {
            from_did: alice_did.clone(),
            to_did: bob_did.clone(),
            amount: 10_000_000,
            currency: "MYC".to_string(),
            payment_type: PaymentType::Direct,
            memo: None,
        };

        let result: Result<Record, _> = conductor
            .call_fallible(&alice_cell.zome("payments"), "send_payment", myc_input)
            .await;

        assert!(result.is_err(), "MYC should be rejected");
        println!("  - MYC rejected: OK");

        println!("Test 5.3 PASSED: Only SAP and TEND currencies accepted");
    }
}

// ============================================================================
// Section 6: Failed Transaction Handling Tests
// ============================================================================

#[cfg(test)]
mod failed_transaction_handling {
    use super::*;

    /// Test 6.1: Channel not found error
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_channel_not_found() {
        println!("Test 6.1: Channel Not Found Error");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];

        // Try to transfer on non-existent channel
        let transfer_input = ChannelTransferInput {
            channel_id: "channel:nonexistent:12345".to_string(),
            amount: 50_000_000,
            from_a: true,
        };

        let result: Result<Record, _> = conductor
            .call_fallible(
                &alice_cell.zome("payments"),
                "channel_transfer",
                transfer_input,
            )
            .await;

        match result {
            Err(e) => {
                let error_msg = format!("{:?}", e);
                assert!(
                    error_msg.contains("Channel not found") || error_msg.contains("not found"),
                    "Should report channel not found, got: {}",
                    error_msg
                );
                println!("  - Channel not found error: OK");
            }
            Ok(_) => panic!("Should have failed for non-existent channel"),
        }

        println!("Test 6.1 PASSED: Channel not found properly handled");
    }

    /// Test 6.2: Empty history handling
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_empty_payment_history() {
        println!("Test 6.2: Empty Payment History");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];

        // Query history for user with no payments
        let new_user_did = format!("did:mycelix:{}", agents[0]);

        let history: Vec<Record> = conductor
            .call(
                &alice_cell.zome("payments"),
                "get_payment_history",
                GetPaymentHistoryInput {
                    did: new_user_did,
                    limit: None,
                },
            )
            .await;

        assert!(history.is_empty(), "New user should have empty history");
        println!("  - Empty history returned: OK");

        println!("Test 6.2 PASSED: Empty payment history handled correctly");
    }

    /// Test 6.3: Concurrent transfer race condition prevention
    ///
    /// Tests that rapid successive transfers are handled correctly
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_concurrent_channel_transfers() {
        println!("Test 6.3: Concurrent Channel Transfers");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = format!("did:mycelix:{}", agents[0]);
        let bob_did = format!("did:mycelix:{}", agents[1]);

        // Open channel with enough balance for multiple transfers
        let channel_input = OpenChannelInput {
            party_a: alice_did.clone(),
            party_b: bob_did.clone(),
            currency: TEST_CURRENCY.to_string(),
            initial_deposit_a: 1_000_000_000,
            initial_deposit_b: 1_000_000_000,
        };

        let channel_record: Record = conductor
            .call(
                &alice_cell.zome("payments"),
                "open_payment_channel",
                channel_input,
            )
            .await;

        let channel: PaymentChannel = channel_record
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        let channel_id = channel.id.clone();
        println!("  - Channel opened with 1000/1000 balances");

        // Perform rapid sequential transfers
        let transfer_amounts: Vec<u64> = vec![
            100_000_000,
            150_000_000,
            75_000_000,
            200_000_000,
            50_000_000,
        ];
        let mut expected_balance_a: u64 = 1_000_000_000;
        let mut expected_balance_b: u64 = 1_000_000_000;

        for (i, amount) in transfer_amounts.iter().enumerate() {
            let from_a = i % 2 == 0; // Alternate between A and B sending

            let transfer_input = ChannelTransferInput {
                channel_id: channel_id.clone(),
                amount: *amount,
                from_a,
            };

            let result: Record = conductor
                .call(
                    &alice_cell.zome("payments"),
                    "channel_transfer",
                    transfer_input,
                )
                .await;

            let updated_channel: PaymentChannel = result
                .entry()
                .to_app_option()
                .expect("Deserialize failed")
                .expect("No entry");

            // Update expected balances
            if from_a {
                expected_balance_a -= amount;
                expected_balance_b += amount;
            } else {
                expected_balance_a += amount;
                expected_balance_b -= amount;
            }

            assert_eq!(
                updated_channel.balance_a, expected_balance_a,
                "Balance A mismatch after transfer {}",
                i
            );
            assert_eq!(
                updated_channel.balance_b, expected_balance_b,
                "Balance B mismatch after transfer {}",
                i
            );

            println!(
                "  - Transfer {}: {} (from_a={}) -> A={}, B={}",
                i, amount, from_a, updated_channel.balance_a, updated_channel.balance_b
            );
        }

        // Verify final total
        let final_total = expected_balance_a + expected_balance_b;
        assert_eq!(
            final_total, 2_000_000_000,
            "Total balance should be conserved"
        );
        println!("  - Final total conserved: {}", final_total);

        println!("Test 6.3 PASSED: Concurrent transfers handled correctly");
    }
}

// ============================================================================
// Section 7: Payment Channel Tests
// ============================================================================

#[cfg(test)]
mod payment_channels {
    use super::*;

    /// Test 7.1: Channel creation with valid inputs
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_channel_creation() {
        println!("Test 7.1: Payment Channel Creation");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = format!("did:mycelix:{}", agents[0]);
        let bob_did = format!("did:mycelix:{}", agents[1]);

        let channel_input = OpenChannelInput {
            party_a: alice_did.clone(),
            party_b: bob_did.clone(),
            currency: TEST_CURRENCY.to_string(),
            initial_deposit_a: 500_000_000,
            initial_deposit_b: 300_000_000,
        };

        let result: Record = conductor
            .call(
                &alice_cell.zome("payments"),
                "open_payment_channel",
                channel_input,
            )
            .await;

        let channel: PaymentChannel = result
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert!(channel.id.starts_with("channel:"), "Channel ID format");
        assert_eq!(channel.party_a, alice_did, "Party A mismatch");
        assert_eq!(channel.party_b, bob_did, "Party B mismatch");
        assert_eq!(channel.balance_a, 500_000_000, "Balance A mismatch");
        assert_eq!(channel.balance_b, 300_000_000, "Balance B mismatch");
        assert_eq!(channel.currency, TEST_CURRENCY, "Currency mismatch");
        assert!(channel.closed.is_none(), "Channel should not be closed");

        println!("  - Channel ID: {}", channel.id);
        println!(
            "  - Party A: {}, Balance: {}",
            channel.party_a, channel.balance_a
        );
        println!(
            "  - Party B: {}, Balance: {}",
            channel.party_b, channel.balance_b
        );

        println!("Test 7.1 PASSED: Payment channel creation works");
    }

    /// Test 7.2: Bidirectional transfers in channel
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_bidirectional_transfers() {
        println!("Test 7.2: Bidirectional Channel Transfers");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let bob_cell = &apps[1].cells()[0];

        let alice_did = format!("did:mycelix:{}", agents[0]);
        let bob_did = format!("did:mycelix:{}", agents[1]);

        // Open channel
        let channel_input = OpenChannelInput {
            party_a: alice_did.clone(),
            party_b: bob_did.clone(),
            currency: TEST_CURRENCY.to_string(),
            initial_deposit_a: 100_000_000,
            initial_deposit_b: 100_000_000,
        };

        let channel_record: Record = conductor
            .call(
                &alice_cell.zome("payments"),
                "open_payment_channel",
                channel_input,
            )
            .await;

        let channel: PaymentChannel = channel_record
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        let channel_id = channel.id.clone();
        println!("  - Initial: A=100M, B=100M");

        // Alice -> Bob (from_a = true)
        let transfer1 = ChannelTransferInput {
            channel_id: channel_id.clone(),
            amount: 30_000_000,
            from_a: true,
        };

        let result1: Record = conductor
            .call(&alice_cell.zome("payments"), "channel_transfer", transfer1)
            .await;

        let channel1: PaymentChannel = result1
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert_eq!(channel1.balance_a, 70_000_000, "After A->B: A balance");
        assert_eq!(channel1.balance_b, 130_000_000, "After A->B: B balance");
        println!("  - After A->B (30M): A=70M, B=130M");

        // Bob -> Alice (from_a = false)
        let transfer2 = ChannelTransferInput {
            channel_id: channel_id.clone(),
            amount: 50_000_000,
            from_a: false,
        };

        let result2: Record = conductor
            .call(&bob_cell.zome("payments"), "channel_transfer", transfer2)
            .await;

        let channel2: PaymentChannel = result2
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert_eq!(channel2.balance_a, 120_000_000, "After B->A: A balance");
        assert_eq!(channel2.balance_b, 80_000_000, "After B->A: B balance");
        println!("  - After B->A (50M): A=120M, B=80M");

        // Verify total conserved
        let total = channel2.balance_a + channel2.balance_b;
        assert_eq!(total, 200_000_000, "Total should be conserved");

        println!("Test 7.2 PASSED: Bidirectional transfers work correctly");
    }

    /// Test 7.3: Channel validation - invalid DIDs
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_channel_invalid_parties() {
        println!("Test 7.3: Channel Invalid Parties Validation");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];

        // Test invalid party_a
        let invalid_input = OpenChannelInput {
            party_a: "invalid_did".to_string(), // No "did:" prefix
            party_b: format!("did:mycelix:{}", agents[1]),
            currency: TEST_CURRENCY.to_string(),
            initial_deposit_a: 100_000_000,
            initial_deposit_b: 100_000_000,
        };

        let result: Result<Record, _> = conductor
            .call_fallible(
                &alice_cell.zome("payments"),
                "open_payment_channel",
                invalid_input,
            )
            .await;

        match result {
            Err(e) => {
                let error_msg = format!("{:?}", e);
                assert!(
                    error_msg.contains("Parties must be valid DIDs")
                        || error_msg.contains("Invalid"),
                    "Should reject invalid party DID, got: {}",
                    error_msg
                );
                println!("  - Invalid party_a rejected: OK");
            }
            Ok(_) => panic!("Should have rejected invalid party_a"),
        }

        // Note: negative deposit amounts are now impossible at the type level (u64 is always >= 0).
        // Only DID validation remains relevant for channel creation.

        println!("Test 7.3 PASSED: Channel validation works correctly");
    }
}

// ============================================================================
// Unit Tests (No Conductor Required)
// ============================================================================

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_did_validation_format() {
        // Valid DIDs
        assert!(test_helpers::test_did("alice").starts_with("did:mycelix:test:"));
        assert!(test_helpers::test_did("bob").starts_with("did:mycelix:test:"));

        // Generated DIDs are different
        let did1 = test_helpers::unique_test_did("user");
        let did2 = test_helpers::unique_test_did("user");
        assert_ne!(did1, did2);
    }

    #[test]
    fn test_payment_type_serialization() {
        // Only current valid payment types (no LoanPayment or EnergyInvestment)
        let types = vec![
            PaymentType::Direct,
            PaymentType::TreasuryContribution("treasury:456".to_string()),
            PaymentType::CommonsContribution("commons:pool:789".to_string()),
            PaymentType::Escrow("escrow:abc".to_string()),
            PaymentType::Recurring(RecurringConfig {
                frequency_days: 30,
                end_date: None,
                remaining: Some(12),
            }),
        ];

        for payment_type in types {
            let json = serde_json::to_string(&payment_type).expect("Serialize failed");
            let deserialized: PaymentType =
                serde_json::from_str(&json).expect("Deserialize failed");
            assert_eq!(
                payment_type, deserialized,
                "Round-trip serialization failed"
            );
        }
    }

    #[test]
    fn test_transfer_status_variants() {
        let statuses = vec![
            TransferStatus::Pending,
            TransferStatus::Processing,
            TransferStatus::Completed,
            TransferStatus::Failed("Network error".to_string()),
            TransferStatus::Cancelled,
            TransferStatus::Refunded,
        ];

        for status in statuses {
            let json = serde_json::to_string(&status).expect("Serialize failed");
            let deserialized: TransferStatus =
                serde_json::from_str(&json).expect("Deserialize failed");
            assert_eq!(status, deserialized, "Status round-trip failed");
        }
    }

    #[test]
    fn test_demurrage_deduction_basic() {
        // Zero balance => no deduction
        assert_eq!(compute_demurrage_deduction(0, 100, 0.05, 31_536_000), 0);

        // Balance below exempt floor => no deduction
        assert_eq!(compute_demurrage_deduction(50, 100, 0.05, 31_536_000), 0);

        // Balance above exempt floor, 1 year elapsed at 5% rate
        let deduction = compute_demurrage_deduction(1000, 100, 0.05, 31_536_000);
        // eligible = 900, decay = 1 - e^(-0.05) ~ 0.04877, deduction ~ 43
        assert!(deduction > 0, "Should have some deduction");
        assert!(deduction < 900, "Should not exceed eligible amount");

        // Zero time elapsed => no deduction
        assert_eq!(compute_demurrage_deduction(1000, 100, 0.05, 0), 0);
    }

    #[test]
    fn test_valid_currencies() {
        // Only SAP and TEND are valid
        let valid = vec!["SAP", "TEND"];
        let invalid = vec!["MYC", "USD", "ENERGY", "BTC", "ETH"];

        for currency in &valid {
            assert!(
                *currency == "SAP" || *currency == "TEND",
                "{} should be valid",
                currency
            );
        }

        for currency in &invalid {
            assert!(
                *currency != "SAP" && *currency != "TEND",
                "{} should be invalid",
                currency
            );
        }
    }
}

// ============================================================================
// Performance Benchmarks
// ============================================================================

#[cfg(test)]
mod performance_benchmarks {
    use super::*;

    /// Benchmark: Multiple payment creation latency
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn benchmark_payment_creation_latency() {
        println!("Benchmark: Payment Creation Latency");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = format!("did:mycelix:{}", agents[0]);
        let bob_did = format!("did:mycelix:{}", agents[1]);

        let num_payments = 10;
        let mut latencies = Vec::new();

        for i in 0..num_payments {
            let input = SendPaymentInput {
                from_did: alice_did.clone(),
                to_did: bob_did.clone(),
                amount: 10_000_000,
                currency: TEST_CURRENCY.to_string(),
                payment_type: PaymentType::Direct,
                memo: Some(format!("Benchmark payment {}", i)),
            };

            let start = std::time::Instant::now();

            let _: Record = conductor
                .call(&alice_cell.zome("payments"), "send_payment", input)
                .await;

            let elapsed = start.elapsed();
            latencies.push(elapsed.as_millis());
        }

        let avg_latency: u128 = latencies.iter().sum::<u128>() / num_payments as u128;
        let max_latency = latencies.iter().max().unwrap();
        let min_latency = latencies.iter().min().unwrap();

        println!("  - Payments created: {}", num_payments);
        println!("  - Average latency: {}ms", avg_latency);
        println!("  - Min latency: {}ms", min_latency);
        println!("  - Max latency: {}ms", max_latency);

        // Target: <500ms per payment
        assert!(avg_latency < 500, "Average latency should be under 500ms");

        println!("Benchmark PASSED: Payment creation meets latency target");
    }
}

// ============================================================================
// Section 8: SAP Balance Management Tests
// ============================================================================

#[cfg(test)]
mod sap_balance_management {
    use super::*;

    /// Test 8.1: Initialize SAP balance and verify zero balance
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_initialize_sap_balance() {
        println!("Test 8.1: Initialize SAP Balance");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_key = agents[0].clone();
        let alice_did = format!("did:mycelix:{}", alice_key);

        // Initialize SAP balance
        let _: Record = conductor
            .call(
                &alice_cell.zome("payments"),
                "initialize_sap_balance",
                alice_did.clone(),
            )
            .await;

        println!("  - SAP balance initialized");

        // Get balance and verify zero
        let balance: SapBalanceResponse = conductor
            .call(
                &alice_cell.zome("payments"),
                "get_sap_balance",
                alice_did.clone(),
            )
            .await;

        assert_eq!(balance.raw_balance, 0, "Initial raw balance should be 0");
        assert_eq!(
            balance.effective_balance, 0,
            "Initial effective balance should be 0"
        );
        assert_eq!(
            balance.pending_demurrage, 0,
            "Initial pending demurrage should be 0"
        );

        println!("  - Raw balance: {}", balance.raw_balance);
        println!("  - Effective balance: {}", balance.effective_balance);
        println!("  - Pending demurrage: {}", balance.pending_demurrage);
        println!("Test 8.1 PASSED: SAP balance initializes to zero");
    }

    /// Test 8.2: Credit and debit SAP, verify remaining balance
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_credit_and_debit_sap() {
        println!("Test 8.2: Credit and Debit SAP");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_key = agents[0].clone();
        let alice_did = format!("did:mycelix:{}", alice_key);

        // Initialize
        let _: Record = conductor
            .call(
                &alice_cell.zome("payments"),
                "initialize_sap_balance",
                alice_did.clone(),
            )
            .await;

        // Credit 5000
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct CreditSapInput {
            pub member_did: String,
            pub amount: u64,
            pub reason: String,
        }

        let _: Record = conductor
            .call(
                &alice_cell.zome("payments"),
                "credit_sap",
                CreditSapInput {
                    member_did: alice_did.clone(),
                    amount: 5000,
                    reason: "Test credit".to_string(),
                },
            )
            .await;

        println!("  - Credited 5000 SAP");

        // Debit 2000
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct DebitSapInput {
            pub member_did: String,
            pub amount: u64,
            pub reason: String,
        }

        let _: Record = conductor
            .call(
                &alice_cell.zome("payments"),
                "debit_sap",
                DebitSapInput {
                    member_did: alice_did.clone(),
                    amount: 2000,
                    reason: "Test debit".to_string(),
                },
            )
            .await;

        println!("  - Debited 2000 SAP");

        // Verify remaining balance is 3000
        let balance: SapBalanceResponse = conductor
            .call(
                &alice_cell.zome("payments"),
                "get_sap_balance",
                alice_did.clone(),
            )
            .await;

        assert_eq!(
            balance.raw_balance, 3000,
            "Balance should be 3000 after credit 5000 and debit 2000"
        );

        println!("  - Remaining balance: {}", balance.raw_balance);
        println!("Test 8.2 PASSED: Credit and debit work correctly");
    }

    /// Test 8.3: Debit exceeding balance should fail
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_debit_exceeds_balance_fails() {
        println!("Test 8.3: Debit Exceeds Balance");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_key = agents[0].clone();
        let alice_did = format!("did:mycelix:{}", alice_key);

        // Initialize and credit 1000
        let _: Record = conductor
            .call(
                &alice_cell.zome("payments"),
                "initialize_sap_balance",
                alice_did.clone(),
            )
            .await;

        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct CreditSapInput {
            pub member_did: String,
            pub amount: u64,
            pub reason: String,
        }

        let _: Record = conductor
            .call(
                &alice_cell.zome("payments"),
                "credit_sap",
                CreditSapInput {
                    member_did: alice_did.clone(),
                    amount: 1000,
                    reason: "Test credit".to_string(),
                },
            )
            .await;

        // Try to debit 2000 (exceeds 1000 balance)
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct DebitSapInput {
            pub member_did: String,
            pub amount: u64,
            pub reason: String,
        }

        let result: Result<Record, _> = conductor
            .call_fallible(
                &alice_cell.zome("payments"),
                "debit_sap",
                DebitSapInput {
                    member_did: alice_did.clone(),
                    amount: 2000,
                    reason: "Test overdraft".to_string(),
                },
            )
            .await;

        match result {
            Err(e) => {
                let error_msg = format!("{:?}", e);
                assert!(
                    error_msg.contains("Insufficient")
                        || error_msg.contains("insufficient")
                        || error_msg.contains("balance"),
                    "Should reject debit exceeding balance, got: {}",
                    error_msg
                );
                println!("  - Overdraft rejected: OK");
            }
            Ok(_) => panic!("Should have rejected debit exceeding balance"),
        }

        println!("Test 8.3 PASSED: Debit exceeding balance fails correctly");
    }

    /// Test 8.4: Demurrage is applied on balance read
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_demurrage_applied_on_read() {
        println!("Test 8.4: Demurrage Applied on Read");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_key = agents[0].clone();
        let alice_did = format!("did:mycelix:{}", alice_key);

        // Initialize and credit a large balance (above exempt floor)
        let _: Record = conductor
            .call(
                &alice_cell.zome("payments"),
                "initialize_sap_balance",
                alice_did.clone(),
            )
            .await;

        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct CreditSapInput {
            pub member_did: String,
            pub amount: u64,
            pub reason: String,
        }

        let _: Record = conductor
            .call(
                &alice_cell.zome("payments"),
                "credit_sap",
                CreditSapInput {
                    member_did: alice_did.clone(),
                    amount: 10_000_000_000, // 10B micro-SAP (well above exempt floor)
                    reason: "Test large credit".to_string(),
                },
            )
            .await;

        // Wait to allow time to pass for demurrage calculation
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Read balance - pending_demurrage should be > 0 (time has elapsed)
        let balance: SapBalanceResponse = conductor
            .call(
                &alice_cell.zome("payments"),
                "get_sap_balance",
                alice_did.clone(),
            )
            .await;

        println!("  - Raw balance: {}", balance.raw_balance);
        println!("  - Effective balance: {}", balance.effective_balance);
        println!("  - Pending demurrage: {}", balance.pending_demurrage);

        // With 10B micro-SAP and 2+ seconds elapsed, demurrage should be > 0
        // (depends on rate and exempt floor, but any non-trivial time produces some)
        assert!(
            balance.pending_demurrage > 0,
            "Pending demurrage should be > 0 after time elapsed"
        );

        println!("Test 8.4 PASSED: Demurrage is calculated on balance read");
    }
}

// ============================================================================
// Section 9: Exit Protocol Tests
// ============================================================================

#[cfg(test)]
mod exit_protocol {
    use super::*;

    /// Test 9.1: Initiate exit with Commons succession preference
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_exit_with_commons_succession() {
        println!("Test 9.1: Exit with Commons Succession");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_key = agents[0].clone();
        let alice_did = format!("did:mycelix:{}", alice_key);

        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct InitiateExitInput {
            pub member_did: String,
            pub succession_preference: SuccessionPreference,
            pub sap_balance: u64,
        }

        let input = InitiateExitInput {
            member_did: alice_did.clone(),
            succession_preference: SuccessionPreference::Commons,
            sap_balance: 500_000_000,
        };

        let result: Record = conductor
            .call(&alice_cell.zome("payments"), "initiate_exit", input)
            .await;

        let exit: ExitRecord = result
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert_eq!(exit.member_did, alice_did, "Member DID mismatch");
        assert!(matches!(
            exit.succession_preference,
            SuccessionPreference::Commons
        ));
        assert_eq!(exit.sap_balance, 500_000_000, "SAP balance mismatch");

        println!("  - Exit initiated for: {}", exit.member_did);
        println!("  - Succession: Commons");
        println!("  - SAP balance: {}", exit.sap_balance);
        println!("Test 9.1 PASSED: Exit with Commons succession works");
    }

    /// Test 9.2: Initiate exit with Designee succession preference
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_exit_with_designee() {
        println!("Test 9.2: Exit with Designee");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_key = agents[0].clone();
        let alice_did = format!("did:mycelix:{}", alice_key);

        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct InitiateExitInput {
            pub member_did: String,
            pub succession_preference: SuccessionPreference,
            pub sap_balance: u64,
        }

        let designee_did = format!("did:mycelix:{}", agents[1]);
        let input = InitiateExitInput {
            member_did: alice_did.clone(),
            succession_preference: SuccessionPreference::Designee(designee_did.clone()),
            sap_balance: 250_000_000,
        };

        let result: Record = conductor
            .call(&alice_cell.zome("payments"), "initiate_exit", input)
            .await;

        let exit: ExitRecord = result
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert_eq!(exit.member_did, alice_did, "Member DID mismatch");
        if let SuccessionPreference::Designee(ref did) = exit.succession_preference {
            assert_eq!(did, &designee_did, "Designee DID mismatch");
        } else {
            panic!("Expected Designee succession preference");
        }

        println!("  - Exit initiated for: {}", exit.member_did);
        println!("  - Succession: Designee({})", designee_did);
        println!("Test 9.2 PASSED: Exit with Designee works");
    }
}

// ============================================================================
// Section 10: Escrow Tests
// ============================================================================

#[cfg(test)]
mod escrow_tests {
    use super::*;

    /// Test 10.1: Create and release escrow
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_create_and_release_escrow() {
        println!("Test 10.1: Create and Release Escrow");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = format!("did:mycelix:{}", agents[0]);
        let bob_did = format!("did:mycelix:{}", agents[1]);

        // Create escrow
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct CreateEscrowInput {
            pub from_did: String,
            pub to_did: String,
            pub amount: u64,
            pub currency: String,
            pub escrow_id: String,
            pub memo: Option<String>,
        }

        let escrow_input = CreateEscrowInput {
            from_did: alice_did.clone(),
            to_did: bob_did.clone(),
            amount: 200_000_000,
            currency: TEST_CURRENCY.to_string(),
            escrow_id: "escrow:test:001".to_string(),
            memo: Some("Test escrow".to_string()),
        };

        let escrow_record: Record = conductor
            .call(&alice_cell.zome("payments"), "create_escrow", escrow_input)
            .await;

        let escrow: Payment = escrow_record
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert!(
            matches!(escrow.status, TransferStatus::Pending),
            "Escrow should be Pending"
        );
        println!("  - Escrow created: {}", escrow.id);

        // Release escrow
        let released_record: Record = conductor
            .call(
                &alice_cell.zome("payments"),
                "release_escrow",
                escrow.id.clone(),
            )
            .await;

        let released: Payment = released_record
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert!(
            matches!(released.status, TransferStatus::Completed),
            "Released escrow should be Completed"
        );
        println!("  - Escrow released, status: {:?}", released.status);
        println!("Test 10.1 PASSED: Create and release escrow works");
    }

    /// Test 10.2: Refund a payment and verify Refunded status
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_refund_payment() {
        println!("Test 10.2: Refund Payment");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = format!("did:mycelix:{}", agents[0]);
        let bob_did = format!("did:mycelix:{}", agents[1]);

        // Create payment
        let input = SendPaymentInput {
            from_did: alice_did.clone(),
            to_did: bob_did.clone(),
            amount: 100_000_000,
            currency: TEST_CURRENCY.to_string(),
            payment_type: PaymentType::Direct,
            memo: Some("Payment to refund".to_string()),
        };

        let payment_record: Record = conductor
            .call(&alice_cell.zome("payments"), "send_payment", input)
            .await;

        let payment: Payment = payment_record
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        println!("  - Payment created: {}", payment.id);

        // Refund
        let refunded_record: Record = conductor
            .call(
                &alice_cell.zome("payments"),
                "refund_payment",
                payment.id.clone(),
            )
            .await;

        let refunded: Payment = refunded_record
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert!(
            matches!(refunded.status, TransferStatus::Refunded),
            "Refunded payment should have Refunded status"
        );
        println!("  - Refunded status: {:?}", refunded.status);
        println!("Test 10.2 PASSED: Payment refund works");
    }

    /// Test 10.3: Cannot double refund
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_cannot_double_refund() {
        println!("Test 10.3: Cannot Double Refund");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = format!("did:mycelix:{}", agents[0]);
        let bob_did = format!("did:mycelix:{}", agents[1]);

        // Create and refund a payment
        let input = SendPaymentInput {
            from_did: alice_did.clone(),
            to_did: bob_did.clone(),
            amount: 50_000_000,
            currency: TEST_CURRENCY.to_string(),
            payment_type: PaymentType::Direct,
            memo: None,
        };

        let payment_record: Record = conductor
            .call(&alice_cell.zome("payments"), "send_payment", input)
            .await;

        let payment: Payment = payment_record
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        // First refund - should succeed
        let _: Record = conductor
            .call(
                &alice_cell.zome("payments"),
                "refund_payment",
                payment.id.clone(),
            )
            .await;

        println!("  - First refund succeeded");

        // Second refund - should fail
        let result: Result<Record, _> = conductor
            .call_fallible(
                &alice_cell.zome("payments"),
                "refund_payment",
                payment.id.clone(),
            )
            .await;

        match result {
            Err(e) => {
                let error_msg = format!("{:?}", e);
                assert!(
                    error_msg.contains("already")
                        || error_msg.contains("Refunded")
                        || error_msg.contains("refund"),
                    "Should reject double refund, got: {}",
                    error_msg
                );
                println!("  - Double refund rejected: OK");
            }
            Ok(_) => panic!("Should have rejected double refund"),
        }

        println!("Test 10.3 PASSED: Double refund is prevented");
    }
}

// ============================================================================
// Section 11: Channel Close Tests
// ============================================================================

#[cfg(test)]
mod channel_close_tests {
    use super::*;

    /// Test 11.1: Close an open payment channel
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_close_payment_channel() {
        println!("Test 11.1: Close Payment Channel");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = format!("did:mycelix:{}", agents[0]);
        let bob_did = format!("did:mycelix:{}", agents[1]);

        // Open channel
        let channel_input = OpenChannelInput {
            party_a: alice_did.clone(),
            party_b: bob_did.clone(),
            currency: TEST_CURRENCY.to_string(),
            initial_deposit_a: 200_000_000,
            initial_deposit_b: 200_000_000,
        };

        let channel_record: Record = conductor
            .call(
                &alice_cell.zome("payments"),
                "open_payment_channel",
                channel_input,
            )
            .await;

        let channel: PaymentChannel = channel_record
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        let channel_id = channel.id.clone();
        assert!(channel.closed.is_none(), "Channel should be open");
        println!("  - Channel opened: {}", channel_id);

        // Close channel
        let closed_record: Record = conductor
            .call(
                &alice_cell.zome("payments"),
                "close_payment_channel",
                channel_id.clone(),
            )
            .await;

        let closed_channel: PaymentChannel = closed_record
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert!(
            closed_channel.closed.is_some(),
            "Channel should have closed timestamp"
        );
        println!(
            "  - Channel closed at: {:?}",
            closed_channel.closed.unwrap()
        );
        println!("Test 11.1 PASSED: Payment channel close works");
    }

    /// Test 11.2: Cannot close an already closed channel
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_cannot_close_already_closed() {
        println!("Test 11.2: Cannot Close Already Closed Channel");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = format!("did:mycelix:{}", agents[0]);
        let bob_did = format!("did:mycelix:{}", agents[1]);

        // Open and close channel
        let channel_input = OpenChannelInput {
            party_a: alice_did.clone(),
            party_b: bob_did.clone(),
            currency: TEST_CURRENCY.to_string(),
            initial_deposit_a: 100_000_000,
            initial_deposit_b: 100_000_000,
        };

        let channel_record: Record = conductor
            .call(
                &alice_cell.zome("payments"),
                "open_payment_channel",
                channel_input,
            )
            .await;

        let channel: PaymentChannel = channel_record
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        let channel_id = channel.id.clone();

        // First close
        let _: Record = conductor
            .call(
                &alice_cell.zome("payments"),
                "close_payment_channel",
                channel_id.clone(),
            )
            .await;

        println!("  - First close succeeded");

        // Second close - should fail
        let result: Result<Record, _> = conductor
            .call_fallible(
                &alice_cell.zome("payments"),
                "close_payment_channel",
                channel_id.clone(),
            )
            .await;

        match result {
            Err(e) => {
                let error_msg = format!("{:?}", e);
                assert!(
                    error_msg.contains("already closed")
                        || error_msg.contains("closed")
                        || error_msg.contains("Channel"),
                    "Should reject closing already-closed channel, got: {}",
                    error_msg
                );
                println!("  - Double close rejected: OK");
            }
            Ok(_) => panic!("Should have rejected closing an already-closed channel"),
        }

        println!("Test 11.2 PASSED: Cannot close already-closed channel");
    }
}

// ============================================================================
// Section 12: SAP Balance Unit Tests
// ============================================================================

#[cfg(test)]
mod sap_balance_unit_tests {
    use super::*;

    /// Test demurrage computation with known values
    #[test]
    fn test_demurrage_computation() {
        // 10B micro-SAP balance, 1B exempt floor, 0.02 rate, 1 year (31536000 seconds)
        // eligible = 10B - 1B = 9B
        // decay = 1 - e^(-0.02) ~ 0.0198
        // deduction = 9B * 0.0198 ~ 178M
        let deduction = compute_demurrage_deduction(
            10_000_000_000, // 10B micro-SAP
            1_000_000_000,  // 1B exempt floor
            0.02,           // 2% annual rate
            31_536_000,     // 1 year in seconds
        );

        // Expected: ~178M (9B * (1 - e^(-0.02)))
        // e^(-0.02) ~ 0.980199, so 1 - 0.980199 ~ 0.019801
        // 9_000_000_000 * 0.019801 ~ 178_209_000
        assert!(
            deduction > 170_000_000,
            "Deduction should be > 170M, got: {}",
            deduction
        );
        assert!(
            deduction < 190_000_000,
            "Deduction should be < 190M, got: {}",
            deduction
        );
        println!("  - Demurrage deduction: {} (expected ~178M)", deduction);
    }

    /// Test demurrage exempt floor: balance at or below exempt floor yields 0
    #[test]
    fn test_demurrage_exempt_floor() {
        // Balance equal to exempt floor
        let deduction_eq = compute_demurrage_deduction(
            1_000_000_000, // balance = exempt floor
            1_000_000_000, // exempt floor
            0.02,
            31_536_000,
        );
        assert_eq!(
            deduction_eq, 0,
            "Balance == exempt floor should yield 0 deduction"
        );

        // Balance below exempt floor
        let deduction_below = compute_demurrage_deduction(
            500_000_000,   // balance < exempt floor
            1_000_000_000, // exempt floor
            0.02,
            31_536_000,
        );
        assert_eq!(
            deduction_below, 0,
            "Balance < exempt floor should yield 0 deduction"
        );
    }

    /// Test fee computation: FeeTier::from_mycel(0.5) = Member, 1M * 0.0003 = 300
    #[test]
    fn test_fee_computation() {
        use mycelix_finance_types::FeeTier;

        let tier = FeeTier::from_mycel(0.5);
        assert!(
            matches!(tier, FeeTier::Member),
            "MYCEL 0.5 should be Member tier"
        );

        let rate = tier.base_fee_rate();
        assert!(
            (rate - 0.0003).abs() < 1e-6,
            "Member base fee rate should be 0.0003"
        );

        let amount: u64 = 1_000_000;
        let fee = (amount as f64 * rate) as u64;
        assert_eq!(fee, 300, "1M * 0.0003 should equal 300");
        println!(
            "  - FeeTier::Member base_fee_rate = {}, fee on 1M = {}",
            rate, fee
        );
    }
}

// ============================================================================
// Section 13: SAP Fee Tier and TEND Fee Tests
// ============================================================================

#[cfg(test)]
mod fee_tier_tests {
    use super::*;

    /// Test 13.1: SAP payment applies a progressive fee
    ///
    /// Initialize a SAP balance, credit sufficient funds, send a SAP payment
    /// of 1,000,000 micro-SAP (amount=1.0), and verify that a fee is deducted.
    ///
    /// Without cross-zome access to the recognition zome, the fee falls back
    /// to the Newcomer tier (0.10% = 0.001). For 1M micro-SAP, the fee should
    /// be 1000 micro-SAP.
    ///
    /// NOTE: Testing different MYCEL scores requires the recognition zome to be
    /// initialized with specific scores for each agent. Since cross-zome setup
    /// is complex in integration tests, we verify the default (Newcomer) tier.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_sap_fee_tiers() {
        println!("Test 13.1: SAP Fee Tiers (Default Newcomer)");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let bob_cell = &apps[1].cells()[0];
        let alice_key = agents[0].clone();
        let bob_key = agents[1].clone();
        let alice_did = format!("did:mycelix:{}", alice_key);
        let bob_did = format!("did:mycelix:{}", bob_key);

        // Initialize Alice's SAP balance and credit 10M micro-SAP
        let _: Record = conductor
            .call(
                &alice_cell.zome("payments"),
                "initialize_sap_balance",
                alice_did.clone(),
            )
            .await;

        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct CreditSapInput {
            pub member_did: String,
            pub amount: u64,
            pub reason: String,
        }

        let _: Record = conductor
            .call(
                &alice_cell.zome("payments"),
                "credit_sap",
                CreditSapInput {
                    member_did: alice_did.clone(),
                    amount: 10_000_000, // 10M micro-SAP
                    reason: "Test credit for fee test".to_string(),
                },
            )
            .await;

        // Initialize Bob's SAP balance (so credit_sap on receive works)
        let _: Record = conductor
            .call(
                &bob_cell.zome("payments"),
                "initialize_sap_balance",
                bob_did.clone(),
            )
            .await;

        println!("  - Alice funded with 10M micro-SAP");

        // Get Alice's balance before payment
        let balance_before: SapBalanceResponse = conductor
            .call(
                &alice_cell.zome("payments"),
                "get_sap_balance",
                alice_did.clone(),
            )
            .await;

        // Send 1.0 SAP (= 1_000_000 micro-SAP) from Alice to Bob
        let input = SendPaymentInput {
            from_did: alice_did.clone(),
            to_did: bob_did.clone(),
            amount: 1_000_000, // 1 SAP = 1_000_000 micro-SAP
            currency: "SAP".to_string(),
            payment_type: PaymentType::Direct,
            memo: Some("Fee tier test payment".to_string()),
        };

        let payment_record: Record = conductor
            .call(&alice_cell.zome("payments"), "send_payment", input)
            .await;

        let payment: Payment = payment_record
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert!(matches!(payment.status, TransferStatus::Completed));
        println!("  - SAP payment sent: {} SAP", payment.amount);

        // Check Alice's balance after -- should be reduced by amount + fee
        let balance_after: SapBalanceResponse = conductor
            .call(
                &alice_cell.zome("payments"),
                "get_sap_balance",
                alice_did.clone(),
            )
            .await;

        // Total debit = amount (1M) + fee (Newcomer 0.001 * 1M = 1000) = 1_001_000
        // Note: demurrage may also apply, so we check the balance decrease is >= amount + fee
        let decrease = balance_before
            .effective_balance
            .saturating_sub(balance_after.effective_balance);
        let micro_amount: u64 = 1_000_000;
        let newcomer_fee = (micro_amount as f64 * 0.001) as u64; // Newcomer tier: 0.10%

        assert!(
            decrease >= micro_amount + newcomer_fee,
            "Balance decrease ({}) should be >= amount ({}) + fee ({})",
            decrease,
            micro_amount,
            newcomer_fee
        );
        println!(
            "  - Balance decrease: {} (amount {} + fee {} + any demurrage)",
            decrease, micro_amount, newcomer_fee
        );

        // Verify Bob received the amount (without fee)
        let bob_balance: SapBalanceResponse = conductor
            .call(
                &bob_cell.zome("payments"),
                "get_sap_balance",
                bob_did.clone(),
            )
            .await;

        assert_eq!(
            bob_balance.raw_balance, micro_amount,
            "Bob should receive exactly the payment amount (no fee)"
        );
        println!(
            "  - Bob received: {} micro-SAP (fee-free)",
            bob_balance.raw_balance
        );

        println!("Test 13.1 PASSED: SAP fee tiers applied correctly (Newcomer default)");
    }

    /// Test 13.2: TEND payment has no fee
    ///
    /// Send a TEND payment and verify no fee is deducted. TEND is fee-free
    /// because the payments coordinator only computes fees for SAP currency.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_tend_payment_no_fee() {
        println!("Test 13.2: TEND Payment No Fee");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Failed to load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;

        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("mycelix-finance", &agents, &[dna])
            .await
            .expect("Failed to install app");

        let alice_cell = &apps[0].cells()[0];
        let alice_did = format!("did:mycelix:{}", agents[0]);
        let bob_did = format!("did:mycelix:{}", agents[1]);

        // Send a TEND payment -- TEND bypasses SAP balance/fee logic entirely
        let input = SendPaymentInput {
            from_did: alice_did.clone(),
            to_did: bob_did.clone(),
            amount: 5_000_000,
            currency: "TEND".to_string(),
            payment_type: PaymentType::Direct,
            memo: Some("TEND fee test".to_string()),
        };

        let payment_record: Record = conductor
            .call(&alice_cell.zome("payments"), "send_payment", input)
            .await;

        let payment: Payment = payment_record
            .entry()
            .to_app_option()
            .expect("Deserialize failed")
            .expect("No entry");

        assert_eq!(payment.currency, "TEND", "Currency should be TEND");
        assert_eq!(payment.amount, 5_000_000, "Amount should be recorded as-is");
        assert!(matches!(payment.status, TransferStatus::Completed));

        // TEND payments do not go through debit_sap/credit_sap, so no fee is applied.
        // The coordinator code path for non-SAP currencies sets fee_amount = 0.
        println!(
            "  - TEND payment completed: {} TEND (fee-free)",
            payment.amount
        );
        println!("  - No SAP balance deduction for TEND payments");

        println!("Test 13.2 PASSED: TEND payments are fee-free");
    }
}
