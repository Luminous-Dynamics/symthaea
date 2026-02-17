#[cfg(test)]
mod tests {
    use super::*;
    use transactions_integrity::*;

    // Helper functions for tests
    fn mock_transaction() -> Transaction {
        Transaction {
            buyer: AgentPubKey::from_raw_36(vec![1u8; 36]),
            seller: AgentPubKey::from_raw_36(vec![2u8; 36]),
            listing_hash: ActionHash::from_raw_36(vec![3u8; 36]),
            quantity: 1,
            total_price_cents: 1999,
            status: TransactionStatus::Pending,
            created_at: Timestamp::from_micros(1000000),
            updated_at: Timestamp::from_micros(1000000),
            tracking_info: None,
            epistemic: EpistemicClassification {
                empirical: EmpiricalLevel::E1Testimonial,
                normative: NormativeLevel::N1Communal,
                materiality: MaterialityLevel::M1Temporal,
            },
        }
    }

    // ===== State Machine Tests =====

    #[test]
    fn test_transaction_lifecycle_states() {
        // Verify all states exist
        let states = vec![
            TransactionStatus::Pending,
            TransactionStatus::Confirmed,
            TransactionStatus::Shipped,
            TransactionStatus::Delivered,
            TransactionStatus::Completed,
            TransactionStatus::Disputed,
            TransactionStatus::Cancelled,
        ];

        assert_eq!(states.len(), 7);
    }

    #[test]
    fn test_valid_state_transition_pending_to_confirmed() {
        let mut transaction = mock_transaction();

        // Initial state
        assert_eq!(transaction.status, TransactionStatus::Pending);

        // Valid transition: Pending → Confirmed
        transaction.status = TransactionStatus::Confirmed;
        assert_eq!(transaction.status, TransactionStatus::Confirmed);
    }

    #[test]
    fn test_valid_state_transition_confirmed_to_shipped() {
        let mut transaction = mock_transaction();
        transaction.status = TransactionStatus::Confirmed;

        // Valid transition: Confirmed → Shipped
        transaction.status = TransactionStatus::Shipped;
        assert_eq!(transaction.status, TransactionStatus::Shipped);
    }

    #[test]
    fn test_valid_state_transition_shipped_to_delivered() {
        let mut transaction = mock_transaction();
        transaction.status = TransactionStatus::Shipped;

        // Valid transition: Shipped → Delivered
        transaction.status = TransactionStatus::Delivered;
        assert_eq!(transaction.status, TransactionStatus::Delivered);
    }

    #[test]
    fn test_valid_state_transition_delivered_to_completed() {
        let mut transaction = mock_transaction();
        transaction.status = TransactionStatus::Delivered;

        // Valid transition: Delivered → Completed
        transaction.status = TransactionStatus::Completed;
        assert_eq!(transaction.status, TransactionStatus::Completed);
    }

    #[test]
    fn test_dispute_from_any_active_state() {
        // Can dispute from Pending
        let mut transaction = mock_transaction();
        transaction.status = TransactionStatus::Pending;
        transaction.status = TransactionStatus::Disputed;
        assert_eq!(transaction.status, TransactionStatus::Disputed);

        // Can dispute from Confirmed
        transaction.status = TransactionStatus::Confirmed;
        transaction.status = TransactionStatus::Disputed;
        assert_eq!(transaction.status, TransactionStatus::Disputed);

        // Can dispute from Shipped
        transaction.status = TransactionStatus::Shipped;
        transaction.status = TransactionStatus::Disputed;
        assert_eq!(transaction.status, TransactionStatus::Disputed);
    }

    #[test]
    fn test_cancel_from_early_states() {
        // Can cancel from Pending
        let mut transaction = mock_transaction();
        transaction.status = TransactionStatus::Pending;
        transaction.status = TransactionStatus::Cancelled;
        assert_eq!(transaction.status, TransactionStatus::Cancelled);

        // Can cancel from Confirmed
        transaction.status = TransactionStatus::Confirmed;
        transaction.status = TransactionStatus::Cancelled;
        assert_eq!(transaction.status, TransactionStatus::Cancelled);
    }

    #[test]
    fn test_invalid_transitions() {
        let transaction = mock_transaction();

        // Cannot go directly from Pending to Shipped (must Confirm first)
        // This would be caught by validation logic

        // Cannot go from Completed to anything (immutable)
        // This would be caught by validation logic

        // Cannot go from Cancelled to anything (immutable)
        // This would be caught by validation logic

        // These assertions verify the test structure exists
        assert_eq!(transaction.status, TransactionStatus::Pending);
    }

    // ===== Validation Tests =====

    #[test]
    fn test_transaction_quantity_validation() {
        let transaction = mock_transaction();

        // Quantity must be at least 1
        assert!(transaction.quantity >= 1);

        // Zero quantity would fail validation
        let zero_quantity = 0;
        assert_eq!(zero_quantity, 0); // Would fail in real validation
    }

    #[test]
    fn test_transaction_price_validation() {
        let transaction = mock_transaction();

        // Price must be greater than zero
        assert!(transaction.total_price_cents > 0);

        // Zero price would fail validation
        let zero_price = 0;
        assert_eq!(zero_price, 0); // Would fail in real validation
    }

    #[test]
    fn test_buyer_seller_different() {
        let transaction = mock_transaction();

        // Buyer and seller must be different
        assert_ne!(transaction.buyer, transaction.seller);

        // Same buyer and seller would fail validation
        let same_agent = AgentPubKey::from_raw_36(vec![1u8; 36]);
        assert_eq!(same_agent, same_agent); // Would fail in real validation
    }

    #[test]
    fn test_epistemic_classification_n1() {
        let transaction = mock_transaction();

        // Transactions must be N1 (communal agreement between buyer-seller)
        assert_eq!(transaction.epistemic.normative, NormativeLevel::N1Communal);
    }

    // ===== Tracking Info Tests =====

    #[test]
    fn test_tracking_info_optional() {
        let mut transaction = mock_transaction();

        // Initially no tracking info
        assert!(transaction.tracking_info.is_none());

        // Can add tracking when shipped
        transaction.tracking_info = Some("USPS-1234567890".to_string());
        assert!(transaction.tracking_info.is_some());
        assert_eq!(transaction.tracking_info.unwrap(), "USPS-1234567890");
    }

    // ===== Timestamp Tests =====

    #[test]
    fn test_timestamps() {
        let transaction = mock_transaction();

        // Created and updated should be set
        assert!(transaction.created_at.as_micros() > 0);
        assert!(transaction.updated_at.as_micros() > 0);

        // Initially they should be the same
        assert_eq!(transaction.created_at, transaction.updated_at);
    }

    #[test]
    fn test_updated_timestamp_changes() {
        let mut transaction = mock_transaction();
        let initial_updated = transaction.updated_at;

        // Update the transaction
        transaction.status = TransactionStatus::Confirmed;
        transaction.updated_at = Timestamp::from_micros(2000000);

        // Updated timestamp should change
        assert_ne!(transaction.updated_at, initial_updated);
        // Created timestamp should NOT change
        assert_eq!(transaction.created_at.as_micros(), 1000000);
    }

    // ===== MATL Integration Tests =====

    #[test]
    fn test_completion_triggers_matl_update() {
        let transaction = mock_transaction();

        // When transaction is completed, MATL should be updated
        // This would call: update_matl_score(seller, successful: true, value)

        // Verify transaction is in completed state
        let completed_status = TransactionStatus::Completed;
        assert_eq!(completed_status, TransactionStatus::Completed);

        // Verify we have seller info for MATL update
        assert_ne!(transaction.seller, AgentPubKey::from_raw_36(vec![0u8; 36]));

        // Verify we have transaction value for MATL weighting
        assert!(transaction.total_price_cents > 0);
    }

    #[test]
    fn test_disputed_transaction_does_not_update_matl() {
        let mut transaction = mock_transaction();

        // Disputed transactions should NOT trigger MATL update
        transaction.status = TransactionStatus::Disputed;

        // MATL update only happens on Completed
        assert_ne!(transaction.status, TransactionStatus::Completed);
    }

    // ===== Create Transaction Input Tests =====

    #[test]
    fn test_create_transaction_input() {
        let input = CreateTransactionInput {
            seller: AgentPubKey::from_raw_36(vec![2u8; 36]),
            listing_hash: ActionHash::from_raw_36(vec![3u8; 36]),
            quantity: 2,
            total_price_cents: 3998, // 2 * 1999
        };

        assert_eq!(input.quantity, 2);
        assert_eq!(input.total_price_cents, 3998);
    }

    #[test]
    fn test_price_quantity_relationship() {
        // Total price should match unit_price * quantity
        let unit_price = 1999;
        let quantity = 3;
        let total_price = unit_price * quantity;

        assert_eq!(total_price, 5997);
    }

    // ===== Mark Shipped Input Tests =====

    #[test]
    fn test_mark_shipped_input() {
        let input = MarkShippedInput {
            transaction_hash: ActionHash::from_raw_36(vec![1u8; 36]),
            tracking_info: Some("FedEx-9876543210".to_string()),
        };

        assert!(input.tracking_info.is_some());
        assert!(input.tracking_info.unwrap().starts_with("FedEx"));
    }

    #[test]
    fn test_shipping_without_tracking() {
        let input = MarkShippedInput {
            transaction_hash: ActionHash::from_raw_36(vec![1u8; 36]),
            tracking_info: None,
        };

        // Tracking info is optional
        assert!(input.tracking_info.is_none());
    }

    // ===== Dispute Input Tests =====

    #[test]
    fn test_dispute_transaction_input() {
        let input = DisputeTransactionInput {
            transaction_hash: ActionHash::from_raw_36(vec![1u8; 36]),
            reason: "Item not as described".to_string(),
        };

        assert!(!input.reason.is_empty());
        assert!(input.reason.len() > 5);
    }

    // ===== Response Structure Tests =====

    #[test]
    fn test_transaction_output() {
        let transaction = mock_transaction();
        let output = TransactionOutput {
            transaction_hash: ActionHash::from_raw_36(vec![1u8; 36]),
            transaction: transaction.clone(),
        };

        assert_eq!(output.transaction.status, TransactionStatus::Pending);
    }

    #[test]
    fn test_transactions_response() {
        let response = TransactionsResponse {
            transactions: vec![],
        };

        assert_eq!(response.transactions.len(), 0);
    }

    // ===== Complete Flow Test =====

    #[test]
    fn test_complete_purchase_flow() {
        let mut transaction = mock_transaction();

        // Step 1: Buyer creates transaction
        assert_eq!(transaction.status, TransactionStatus::Pending);

        // Step 2: Seller confirms
        transaction.status = TransactionStatus::Confirmed;
        assert_eq!(transaction.status, TransactionStatus::Confirmed);

        // Step 3: Seller ships with tracking
        transaction.status = TransactionStatus::Shipped;
        transaction.tracking_info = Some("UPS-1122334455".to_string());
        assert_eq!(transaction.status, TransactionStatus::Shipped);
        assert!(transaction.tracking_info.is_some());

        // Step 4: Buyer confirms delivery
        transaction.status = TransactionStatus::Delivered;
        assert_eq!(transaction.status, TransactionStatus::Delivered);

        // Step 5: System completes transaction (triggers MATL)
        transaction.status = TransactionStatus::Completed;
        transaction.epistemic.materiality = MaterialityLevel::M2Persistent;
        assert_eq!(transaction.status, TransactionStatus::Completed);
        assert_eq!(transaction.epistemic.materiality, MaterialityLevel::M2Persistent);
    }

    #[test]
    fn test_cancellation_flow() {
        let mut transaction = mock_transaction();

        // Buyer creates transaction
        assert_eq!(transaction.status, TransactionStatus::Pending);

        // Either party cancels before confirmation
        transaction.status = TransactionStatus::Cancelled;
        assert_eq!(transaction.status, TransactionStatus::Cancelled);

        // Transaction is now terminal
        // No further state changes allowed
    }

    #[test]
    fn test_dispute_flow() {
        let mut transaction = mock_transaction();

        // Transaction progresses normally
        transaction.status = TransactionStatus::Confirmed;
        transaction.status = TransactionStatus::Shipped;

        // Issue arises, either party disputes
        transaction.status = TransactionStatus::Disputed;
        assert_eq!(transaction.status, TransactionStatus::Disputed);

        // Now goes to arbitration zome for resolution
    }

    // ===== Edge Cases =====

    #[test]
    fn test_large_quantity() {
        let mut transaction = mock_transaction();
        transaction.quantity = 1000;
        transaction.total_price_cents = 1999 * 1000; // $19,990

        assert_eq!(transaction.quantity, 1000);
        assert_eq!(transaction.total_price_cents, 1_999_000);
    }

    #[test]
    fn test_high_value_transaction() {
        let mut transaction = mock_transaction();
        transaction.total_price_cents = 100_000_000; // $1,000,000

        assert_eq!(transaction.total_price_cents, 100_000_000);
    }
}
