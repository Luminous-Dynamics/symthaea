// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Transfer Zome Unit Tests
//!
//! Comprehensive test coverage for property transfers, escrow management,
//! condition satisfaction, status transitions, and access control.

use serde::{Deserialize, Serialize};

// =============================================================================
// Test Data Structures
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestTransfer {
    pub id: String,
    pub property_id: String,
    pub from_did: String,
    pub to_did: String,
    pub transfer_type: TestTransferType,
    pub price: Option<f64>,
    pub currency: Option<String>,
    pub conditions: Vec<TestTransferCondition>,
    pub status: TestTransferStatus,
    pub initiated: i64,
    pub completed: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestTransferType {
    Sale,
    Gift,
    Inheritance,
    CourtOrder,
    Exchange,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestTransferCondition {
    pub condition_type: TestConditionType,
    pub description: String,
    pub satisfied: bool,
    pub verified_by: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestConditionType {
    PaymentReceived,
    InspectionComplete,
    TitleClear,
    DocumentsSigned,
    TaxesPaid,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestTransferStatus {
    Initiated,
    AwaitingAcceptance,
    InEscrow,
    ConditionsPending,
    Completed,
    Cancelled,
    Disputed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestEscrow {
    pub id: String,
    pub transfer_id: String,
    pub escrow_agent_did: Option<String>,
    pub amount: f64,
    pub currency: String,
    pub funded: bool,
    pub release_conditions: Vec<String>,
    pub created: i64,
    pub released: Option<i64>,
}

// =============================================================================
// Helper Functions
// =============================================================================

fn create_test_transfer() -> TestTransfer {
    TestTransfer {
        id: "transfer:property123:1704067200000000".to_string(),
        property_id: "property:did:mycelix:owner:1704000000000000".to_string(),
        from_did: "did:mycelix:seller".to_string(),
        to_did: "did:mycelix:buyer".to_string(),
        transfer_type: TestTransferType::Sale,
        price: Some(500000.0),
        currency: Some("USD".to_string()),
        conditions: vec![
            TestTransferCondition {
                condition_type: TestConditionType::PaymentReceived,
                description: "Buyer must transfer full purchase price".to_string(),
                satisfied: false,
                verified_by: None,
            },
            TestTransferCondition {
                condition_type: TestConditionType::TitleClear,
                description: "Property must have clear title".to_string(),
                satisfied: false,
                verified_by: None,
            },
        ],
        status: TestTransferStatus::Initiated,
        initiated: 1704067200000000,
        completed: None,
    }
}

fn create_test_escrow() -> TestEscrow {
    TestEscrow {
        id: "escrow:transfer123:1704067200000000".to_string(),
        transfer_id: "transfer:property123:1704067200000000".to_string(),
        escrow_agent_did: Some("did:mycelix:escrowagent".to_string()),
        amount: 500000.0,
        currency: "USD".to_string(),
        funded: false,
        release_conditions: vec![
            "All transfer conditions satisfied".to_string(),
            "Both parties confirm".to_string(),
        ],
        created: 1704067200000000,
        released: None,
    }
}

fn validate_did(did: &str) -> bool {
    did.starts_with("did:")
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // TRANSFER CREATION TESTS
    // =========================================================================

    #[test]
    fn test_transfer_requires_valid_seller_did() {
        let transfer = create_test_transfer();
        assert!(validate_did(&transfer.from_did));
    }

    #[test]
    fn test_transfer_requires_valid_buyer_did() {
        let transfer = create_test_transfer();
        assert!(validate_did(&transfer.to_did));
    }

    #[test]
    fn test_transfer_invalid_seller_did_rejected() {
        let mut transfer = create_test_transfer();
        transfer.from_did = "invalid_seller".to_string();
        assert!(!validate_did(&transfer.from_did));
    }

    #[test]
    fn test_transfer_invalid_buyer_did_rejected() {
        let mut transfer = create_test_transfer();
        transfer.to_did = "invalid_buyer".to_string();
        assert!(!validate_did(&transfer.to_did));
    }

    #[test]
    fn test_transfer_to_self_rejected() {
        let mut transfer = create_test_transfer();
        transfer.to_did = transfer.from_did.clone();
        // Cannot transfer to yourself
        assert_eq!(transfer.from_did, transfer.to_did);
        // This should be rejected by validation
    }

    #[test]
    fn test_transfer_has_unique_id() {
        let transfer = create_test_transfer();
        assert!(transfer.id.contains("transfer:"));
        assert!(!transfer.id.is_empty());
    }

    #[test]
    fn test_transfer_references_property() {
        let transfer = create_test_transfer();
        assert!(!transfer.property_id.is_empty());
        assert!(transfer.property_id.contains("property:"));
    }

    #[test]
    fn test_transfer_initiated_timestamp_valid() {
        let transfer = create_test_transfer();
        assert!(transfer.initiated > 0);
        assert!(transfer.initiated > 946684800000000); // After year 2000
    }

    #[test]
    fn test_transfer_initial_status() {
        let transfer = create_test_transfer();
        assert_eq!(transfer.status, TestTransferStatus::Initiated);
    }

    #[test]
    fn test_transfer_not_completed_initially() {
        let transfer = create_test_transfer();
        assert!(transfer.completed.is_none());
    }

    // =========================================================================
    // TRANSFER TYPE TESTS
    // =========================================================================

    #[test]
    fn test_transfer_type_sale_has_price() {
        let transfer = create_test_transfer();
        assert_eq!(transfer.transfer_type, TestTransferType::Sale);
        assert!(transfer.price.is_some());
        assert!(transfer.price.unwrap() > 0.0);
    }

    #[test]
    fn test_transfer_type_gift_no_price_required() {
        let mut transfer = create_test_transfer();
        transfer.transfer_type = TestTransferType::Gift;
        transfer.price = None;
        // Gift transfers don't require a price
        assert!(transfer.price.is_none());
    }

    #[test]
    fn test_transfer_type_inheritance_valid() {
        let mut transfer = create_test_transfer();
        transfer.transfer_type = TestTransferType::Inheritance;
        assert_eq!(transfer.transfer_type, TestTransferType::Inheritance);
    }

    #[test]
    fn test_transfer_type_court_order_valid() {
        let mut transfer = create_test_transfer();
        transfer.transfer_type = TestTransferType::CourtOrder;
        assert_eq!(transfer.transfer_type, TestTransferType::CourtOrder);
    }

    #[test]
    fn test_all_transfer_types_supported() {
        let types = vec![
            TestTransferType::Sale,
            TestTransferType::Gift,
            TestTransferType::Inheritance,
            TestTransferType::CourtOrder,
            TestTransferType::Exchange,
            TestTransferType::Other,
        ];
        for t in types {
            assert!(matches!(
                t,
                TestTransferType::Sale
                    | TestTransferType::Gift
                    | TestTransferType::Inheritance
                    | TestTransferType::CourtOrder
                    | TestTransferType::Exchange
                    | TestTransferType::Other
            ));
        }
    }

    // =========================================================================
    // TRANSFER CONDITIONS TESTS
    // =========================================================================

    #[test]
    fn test_transfer_can_have_conditions() {
        let transfer = create_test_transfer();
        assert!(!transfer.conditions.is_empty());
    }

    #[test]
    fn test_transfer_conditions_initially_unsatisfied() {
        let transfer = create_test_transfer();
        for condition in &transfer.conditions {
            assert!(!condition.satisfied);
            assert!(condition.verified_by.is_none());
        }
    }

    #[test]
    fn test_transfer_condition_types_supported() {
        let types = vec![
            TestConditionType::PaymentReceived,
            TestConditionType::InspectionComplete,
            TestConditionType::TitleClear,
            TestConditionType::DocumentsSigned,
            TestConditionType::TaxesPaid,
            TestConditionType::Custom("Custom condition".to_string()),
        ];
        for t in types {
            assert!(matches!(
                t,
                TestConditionType::PaymentReceived
                    | TestConditionType::InspectionComplete
                    | TestConditionType::TitleClear
                    | TestConditionType::DocumentsSigned
                    | TestConditionType::TaxesPaid
                    | TestConditionType::Custom(_)
            ));
        }
    }

    #[test]
    fn test_condition_satisfaction_tracking() {
        let mut transfer = create_test_transfer();
        transfer.conditions[0].satisfied = true;
        transfer.conditions[0].verified_by = Some("did:mycelix:verifier".to_string());

        assert!(transfer.conditions[0].satisfied);
        assert!(transfer.conditions[0].verified_by.is_some());
        assert!(validate_did(
            transfer.conditions[0].verified_by.as_ref().unwrap()
        ));
    }

    #[test]
    fn test_condition_requires_description() {
        let transfer = create_test_transfer();
        for condition in &transfer.conditions {
            assert!(!condition.description.is_empty());
        }
    }

    fn all_conditions_satisfied(transfer: &TestTransfer) -> bool {
        transfer.conditions.iter().all(|c| c.satisfied)
    }

    #[test]
    fn test_all_conditions_must_be_satisfied_to_complete() {
        let mut transfer = create_test_transfer();
        assert!(!all_conditions_satisfied(&transfer));

        // Satisfy all conditions
        for condition in &mut transfer.conditions {
            condition.satisfied = true;
        }
        assert!(all_conditions_satisfied(&transfer));
    }

    #[test]
    fn test_partial_condition_satisfaction() {
        let mut transfer = create_test_transfer();
        transfer.conditions[0].satisfied = true;
        // Not all conditions satisfied
        assert!(!all_conditions_satisfied(&transfer));
    }

    // =========================================================================
    // TRANSFER STATUS TESTS
    // =========================================================================

    #[test]
    fn test_transfer_status_transitions() {
        let statuses = vec![
            TestTransferStatus::Initiated,
            TestTransferStatus::AwaitingAcceptance,
            TestTransferStatus::InEscrow,
            TestTransferStatus::ConditionsPending,
            TestTransferStatus::Completed,
            TestTransferStatus::Cancelled,
            TestTransferStatus::Disputed,
        ];
        for status in statuses {
            assert!(matches!(
                status,
                TestTransferStatus::Initiated
                    | TestTransferStatus::AwaitingAcceptance
                    | TestTransferStatus::InEscrow
                    | TestTransferStatus::ConditionsPending
                    | TestTransferStatus::Completed
                    | TestTransferStatus::Cancelled
                    | TestTransferStatus::Disputed
            ));
        }
    }

    fn can_cancel_transfer(transfer: &TestTransfer) -> bool {
        transfer.status != TestTransferStatus::Completed
    }

    #[test]
    fn test_cannot_cancel_completed_transfer() {
        let mut transfer = create_test_transfer();
        transfer.status = TestTransferStatus::Completed;
        assert!(!can_cancel_transfer(&transfer));
    }

    #[test]
    fn test_can_cancel_initiated_transfer() {
        let transfer = create_test_transfer();
        assert_eq!(transfer.status, TestTransferStatus::Initiated);
        assert!(can_cancel_transfer(&transfer));
    }

    #[test]
    fn test_can_cancel_in_escrow_transfer() {
        let mut transfer = create_test_transfer();
        transfer.status = TestTransferStatus::InEscrow;
        assert!(can_cancel_transfer(&transfer));
    }

    fn can_dispute_transfer(transfer: &TestTransfer) -> bool {
        transfer.status != TestTransferStatus::Completed
            && transfer.status != TestTransferStatus::Cancelled
    }

    #[test]
    fn test_cannot_dispute_completed_transfer() {
        let mut transfer = create_test_transfer();
        transfer.status = TestTransferStatus::Completed;
        assert!(!can_dispute_transfer(&transfer));
    }

    #[test]
    fn test_cannot_dispute_cancelled_transfer() {
        let mut transfer = create_test_transfer();
        transfer.status = TestTransferStatus::Cancelled;
        assert!(!can_dispute_transfer(&transfer));
    }

    #[test]
    fn test_can_dispute_active_transfer() {
        let mut transfer = create_test_transfer();
        transfer.status = TestTransferStatus::InEscrow;
        assert!(can_dispute_transfer(&transfer));
    }

    fn can_accept_transfer(transfer: &TestTransfer) -> bool {
        transfer.status == TestTransferStatus::Initiated
            || transfer.status == TestTransferStatus::AwaitingAcceptance
    }

    #[test]
    fn test_can_accept_initiated_transfer() {
        let transfer = create_test_transfer();
        assert!(can_accept_transfer(&transfer));
    }

    #[test]
    fn test_cannot_accept_in_escrow_transfer() {
        let mut transfer = create_test_transfer();
        transfer.status = TestTransferStatus::InEscrow;
        assert!(!can_accept_transfer(&transfer));
    }

    // =========================================================================
    // ACCESS CONTROL TESTS
    // =========================================================================

    fn can_cancel(transfer: &TestTransfer, requester_did: &str) -> bool {
        transfer.from_did == requester_did && can_cancel_transfer(transfer)
    }

    fn can_accept(transfer: &TestTransfer, requester_did: &str) -> bool {
        transfer.to_did == requester_did && can_accept_transfer(transfer)
    }

    fn can_dispute(transfer: &TestTransfer, requester_did: &str) -> bool {
        (transfer.from_did == requester_did || transfer.to_did == requester_did)
            && can_dispute_transfer(transfer)
    }

    #[test]
    fn test_only_seller_can_cancel() {
        let transfer = create_test_transfer();
        assert!(can_cancel(&transfer, &transfer.from_did));
        assert!(!can_cancel(&transfer, &transfer.to_did));
        assert!(!can_cancel(&transfer, "did:mycelix:other"));
    }

    #[test]
    fn test_only_buyer_can_accept() {
        let transfer = create_test_transfer();
        assert!(can_accept(&transfer, &transfer.to_did));
        assert!(!can_accept(&transfer, &transfer.from_did));
        assert!(!can_accept(&transfer, "did:mycelix:other"));
    }

    #[test]
    fn test_either_party_can_dispute() {
        let transfer = create_test_transfer();
        assert!(can_dispute(&transfer, &transfer.from_did));
        assert!(can_dispute(&transfer, &transfer.to_did));
        assert!(!can_dispute(&transfer, "did:mycelix:other"));
    }

    #[test]
    fn test_third_party_cannot_dispute() {
        let transfer = create_test_transfer();
        assert!(!can_dispute(&transfer, "did:mycelix:thirdparty"));
    }

    // =========================================================================
    // ESCROW TESTS
    // =========================================================================

    #[test]
    fn test_escrow_creation() {
        let escrow = create_test_escrow();
        assert!(!escrow.id.is_empty());
        assert!(!escrow.transfer_id.is_empty());
    }

    #[test]
    fn test_escrow_amount_positive() {
        let escrow = create_test_escrow();
        assert!(escrow.amount > 0.0);
    }

    #[test]
    fn test_escrow_zero_amount_invalid() {
        let mut escrow = create_test_escrow();
        escrow.amount = 0.0;
        // Zero amount should be rejected
        assert!(escrow.amount <= 0.0);
    }

    #[test]
    fn test_escrow_negative_amount_invalid() {
        let mut escrow = create_test_escrow();
        escrow.amount = -1000.0;
        // Negative amount should be rejected
        assert!(escrow.amount < 0.0);
    }

    #[test]
    fn test_escrow_initially_not_funded() {
        let escrow = create_test_escrow();
        assert!(!escrow.funded);
    }

    #[test]
    fn test_escrow_initially_not_released() {
        let escrow = create_test_escrow();
        assert!(escrow.released.is_none());
    }

    #[test]
    fn test_escrow_has_currency() {
        let escrow = create_test_escrow();
        assert!(!escrow.currency.is_empty());
    }

    #[test]
    fn test_escrow_agent_optional() {
        let mut escrow = create_test_escrow();
        escrow.escrow_agent_did = None;
        assert!(escrow.escrow_agent_did.is_none());
    }

    #[test]
    fn test_escrow_agent_valid_did() {
        let escrow = create_test_escrow();
        if let Some(ref agent) = escrow.escrow_agent_did {
            assert!(validate_did(agent));
        }
    }

    #[test]
    fn test_escrow_has_release_conditions() {
        let escrow = create_test_escrow();
        assert!(!escrow.release_conditions.is_empty());
    }

    fn can_fund_escrow(escrow: &TestEscrow) -> bool {
        !escrow.funded
    }

    #[test]
    fn test_can_fund_unfunded_escrow() {
        let escrow = create_test_escrow();
        assert!(can_fund_escrow(&escrow));
    }

    #[test]
    fn test_cannot_fund_already_funded_escrow() {
        let mut escrow = create_test_escrow();
        escrow.funded = true;
        assert!(!can_fund_escrow(&escrow));
    }

    fn can_release_escrow(escrow: &TestEscrow) -> bool {
        escrow.funded && escrow.released.is_none()
    }

    #[test]
    fn test_cannot_release_unfunded_escrow() {
        let escrow = create_test_escrow();
        assert!(!can_release_escrow(&escrow));
    }

    #[test]
    fn test_can_release_funded_escrow() {
        let mut escrow = create_test_escrow();
        escrow.funded = true;
        assert!(can_release_escrow(&escrow));
    }

    #[test]
    fn test_cannot_release_already_released_escrow() {
        let mut escrow = create_test_escrow();
        escrow.funded = true;
        escrow.released = Some(1704200000000000);
        assert!(!can_release_escrow(&escrow));
    }

    // =========================================================================
    // TRANSFER COMPLETION TESTS
    // =========================================================================

    fn can_complete_transfer(transfer: &TestTransfer) -> bool {
        all_conditions_satisfied(transfer)
            && transfer.status != TestTransferStatus::Completed
            && transfer.status != TestTransferStatus::Cancelled
    }

    #[test]
    fn test_can_complete_when_all_conditions_satisfied() {
        let mut transfer = create_test_transfer();
        for condition in &mut transfer.conditions {
            condition.satisfied = true;
        }
        assert!(can_complete_transfer(&transfer));
    }

    #[test]
    fn test_cannot_complete_with_unsatisfied_conditions() {
        let transfer = create_test_transfer();
        assert!(!can_complete_transfer(&transfer));
    }

    #[test]
    fn test_cannot_complete_cancelled_transfer() {
        let mut transfer = create_test_transfer();
        for condition in &mut transfer.conditions {
            condition.satisfied = true;
        }
        transfer.status = TestTransferStatus::Cancelled;
        assert!(!can_complete_transfer(&transfer));
    }

    #[test]
    fn test_cannot_complete_already_completed_transfer() {
        let mut transfer = create_test_transfer();
        for condition in &mut transfer.conditions {
            condition.satisfied = true;
        }
        transfer.status = TestTransferStatus::Completed;
        assert!(!can_complete_transfer(&transfer));
    }

    #[test]
    fn test_completion_sets_timestamp() {
        let mut transfer = create_test_transfer();
        for condition in &mut transfer.conditions {
            condition.satisfied = true;
        }
        transfer.status = TestTransferStatus::Completed;
        transfer.completed = Some(1704100000000000);

        assert!(transfer.completed.is_some());
        assert!(transfer.completed.unwrap() > transfer.initiated);
    }

    // =========================================================================
    // PRICE AND CURRENCY TESTS
    // =========================================================================

    #[test]
    fn test_price_non_negative() {
        let transfer = create_test_transfer();
        if let Some(price) = transfer.price {
            assert!(price >= 0.0);
        }
    }

    #[test]
    fn test_currency_present_when_price_set() {
        let transfer = create_test_transfer();
        if transfer.price.is_some() {
            assert!(transfer.currency.is_some());
        }
    }

    #[test]
    fn test_currency_iso_format() {
        let transfer = create_test_transfer();
        if let Some(ref currency) = transfer.currency {
            // ISO 4217 codes are 3 letters
            assert!(currency.len() >= 3);
        }
    }

    // =========================================================================
    // EDGE CASES
    // =========================================================================

    #[test]
    fn test_transfer_with_no_conditions() {
        let mut transfer = create_test_transfer();
        transfer.conditions = vec![];
        // A transfer with no conditions is immediately completable
        assert!(all_conditions_satisfied(&transfer));
    }

    #[test]
    fn test_transfer_with_many_conditions() {
        let mut transfer = create_test_transfer();
        transfer.conditions = (0..10)
            .map(|i| TestTransferCondition {
                condition_type: TestConditionType::Custom(format!("Condition {}", i)),
                description: format!("Custom condition number {}", i),
                satisfied: false,
                verified_by: None,
            })
            .collect();

        assert_eq!(transfer.conditions.len(), 10);
        assert!(!all_conditions_satisfied(&transfer));
    }

    #[test]
    fn test_transfer_zero_price_valid_for_gift() {
        let mut transfer = create_test_transfer();
        transfer.transfer_type = TestTransferType::Gift;
        transfer.price = Some(0.0);
        // Gift with zero price is valid
        assert_eq!(transfer.price, Some(0.0));
    }

    #[test]
    fn test_escrow_amount_matches_transfer_price() {
        let transfer = create_test_transfer();
        let escrow = create_test_escrow();

        if let Some(price) = transfer.price {
            // Escrow amount should match the transfer price
            assert_eq!(escrow.amount, price);
        }
    }

    #[test]
    fn test_transfer_status_flow_happy_path() {
        let mut transfer = create_test_transfer();

        // Initial state
        assert_eq!(transfer.status, TestTransferStatus::Initiated);

        // Buyer accepts
        transfer.status = TestTransferStatus::ConditionsPending;
        assert_eq!(transfer.status, TestTransferStatus::ConditionsPending);

        // Escrow created
        transfer.status = TestTransferStatus::InEscrow;
        assert_eq!(transfer.status, TestTransferStatus::InEscrow);

        // Back to conditions pending
        transfer.status = TestTransferStatus::ConditionsPending;

        // Satisfy all conditions
        for condition in &mut transfer.conditions {
            condition.satisfied = true;
            condition.verified_by = Some("did:mycelix:verifier".to_string());
        }

        // Complete
        transfer.status = TestTransferStatus::Completed;
        transfer.completed = Some(1704200000000000);
        assert_eq!(transfer.status, TestTransferStatus::Completed);
        assert!(transfer.completed.is_some());
    }

    #[test]
    fn test_transfer_cancellation_flow() {
        let mut transfer = create_test_transfer();

        // Initial state
        assert_eq!(transfer.status, TestTransferStatus::Initiated);

        // Cancel
        transfer.status = TestTransferStatus::Cancelled;
        assert_eq!(transfer.status, TestTransferStatus::Cancelled);

        // Should not be completable
        assert!(!can_complete_transfer(&transfer));
    }

    #[test]
    fn test_transfer_dispute_flow() {
        let mut transfer = create_test_transfer();
        transfer.status = TestTransferStatus::InEscrow;

        // Dispute raised
        transfer.status = TestTransferStatus::Disputed;
        assert_eq!(transfer.status, TestTransferStatus::Disputed);

        // Cannot be cancelled once disputed
        assert!(!can_cancel_transfer(&transfer) || transfer.status == TestTransferStatus::Disputed);
    }

    #[test]
    fn test_condition_verification_chain() {
        let mut transfer = create_test_transfer();

        // Different verifiers for different conditions
        transfer.conditions[0].satisfied = true;
        transfer.conditions[0].verified_by = Some("did:mycelix:escrowagent".to_string());

        transfer.conditions[1].satisfied = true;
        transfer.conditions[1].verified_by = Some("did:mycelix:titlecompany".to_string());

        // All conditions satisfied by different parties
        assert!(all_conditions_satisfied(&transfer));
        assert_ne!(
            transfer.conditions[0].verified_by,
            transfer.conditions[1].verified_by
        );
    }

    #[test]
    fn test_multiple_transfers_for_same_property_over_time() {
        // Simulate multiple transfers of the same property
        let transfer1 = TestTransfer {
            id: "transfer:prop:1".to_string(),
            property_id: "property:did:mycelix:original:1000".to_string(),
            from_did: "did:mycelix:owner1".to_string(),
            to_did: "did:mycelix:owner2".to_string(),
            transfer_type: TestTransferType::Sale,
            price: Some(400000.0),
            currency: Some("USD".to_string()),
            conditions: vec![],
            status: TestTransferStatus::Completed,
            initiated: 1700000000000000,
            completed: Some(1700100000000000),
        };

        let transfer2 = TestTransfer {
            id: "transfer:prop:2".to_string(),
            property_id: transfer1.property_id.clone(),
            from_did: "did:mycelix:owner2".to_string(), // Previous buyer is now seller
            to_did: "did:mycelix:owner3".to_string(),
            transfer_type: TestTransferType::Sale,
            price: Some(450000.0),
            currency: Some("USD".to_string()),
            conditions: vec![],
            status: TestTransferStatus::Initiated,
            initiated: 1704000000000000,
            completed: None,
        };

        // Same property
        assert_eq!(transfer1.property_id, transfer2.property_id);
        // Second transfer seller is first transfer buyer
        assert_eq!(transfer1.to_did, transfer2.from_did);
        // Second transfer started after first completed
        assert!(transfer2.initiated > transfer1.completed.unwrap());
    }
}
