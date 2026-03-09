//! Property Transfer Integrity Zome
use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Transfer {
    pub id: String,
    pub property_id: String,
    pub from_did: String,
    pub to_did: String,
    pub transfer_type: TransferType,
    pub price: Option<f64>,
    pub currency: Option<String>,
    pub conditions: Vec<TransferCondition>,
    pub status: TransferStatus,
    pub initiated: Timestamp,
    pub completed: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TransferType {
    Sale,
    Gift,
    Inheritance,
    CourtOrder,
    Exchange,
    Other,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TransferCondition {
    pub condition_type: ConditionType,
    pub description: String,
    pub satisfied: bool,
    pub verified_by: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ConditionType {
    PaymentReceived,
    InspectionComplete,
    TitleClear,
    DocumentsSigned,
    TaxesPaid,
    Custom(String),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TransferStatus {
    Initiated,
    AwaitingAcceptance,
    InEscrow,
    ConditionsPending,
    Completed,
    Cancelled,
    Disputed,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Escrow {
    pub id: String,
    pub transfer_id: String,
    pub escrow_agent_did: Option<String>,
    pub amount: f64,
    pub currency: String,
    pub funded: bool,
    pub release_conditions: Vec<String>,
    pub created: Timestamp,
    pub released: Option<Timestamp>,
}

/// Anchor entry for deterministic link bases from strings
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Transfer(Transfer),
    Escrow(Escrow),
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
}

#[hdk_link_types]
pub enum LinkTypes {
    PropertyToTransfers,
    SellerToTransfers,
    BuyerToTransfers,
    TransferToEscrow,
}

/// Genesis self-check
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Main validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Transfer(transfer) => {
                    validate_create_transfer(EntryCreationAction::Create(action), transfer)
                }
                EntryTypes::Escrow(escrow) => {
                    validate_create_escrow(EntryCreationAction::Create(action), escrow)
                }
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::Transfer(transfer) => validate_update_transfer(action, transfer),
                EntryTypes::Escrow(escrow) => validate_update_escrow(action, escrow),
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, tag, .. } => match link_type {
            LinkTypes::PropertyToTransfers => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "PropertyToTransfers link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::SellerToTransfers => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "SellerToTransfers link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::BuyerToTransfers => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "BuyerToTransfers link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::TransferToEscrow => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "TransferToEscrow link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        },
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_transfer(
    _action: EntryCreationAction,
    transfer: Transfer,
) -> ExternResult<ValidateCallbackResult> {
    // --- Empty string checks (required fields) ---
    if transfer.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Transfer id cannot be empty".into(),
        ));
    }
    if transfer.property_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Transfer property_id cannot be empty".into(),
        ));
    }
    // --- String length limits ---
    if transfer.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Transfer id too long (max 256 chars)".into(),
        ));
    }
    if transfer.property_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Transfer property_id too long (max 256 chars)".into(),
        ));
    }
    if let Some(ref currency) = transfer.currency {
        if currency.len() > 16 {
            return Ok(ValidateCallbackResult::Invalid(
                "Transfer currency too long (max 16 chars)".into(),
            ));
        }
    }
    // --- Existing validation ---
    if !transfer.from_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Seller must be a valid DID".into(),
        ));
    }
    if !transfer.to_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Buyer must be a valid DID".into(),
        ));
    }
    if transfer.from_did == transfer.to_did {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot transfer to yourself".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_transfer(
    _action: Update,
    _transfer: Transfer,
) -> ExternResult<ValidateCallbackResult> {
    // Status can be updated
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_escrow(
    _action: EntryCreationAction,
    escrow: Escrow,
) -> ExternResult<ValidateCallbackResult> {
    // --- Empty string checks (required fields) ---
    if escrow.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Escrow id cannot be empty".into(),
        ));
    }
    if escrow.transfer_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Escrow transfer_id cannot be empty".into(),
        ));
    }
    // --- String length limits ---
    if escrow.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Escrow id too long (max 256 chars)".into(),
        ));
    }
    if escrow.transfer_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Escrow transfer_id too long (max 256 chars)".into(),
        ));
    }
    if escrow.currency.len() > 16 {
        return Ok(ValidateCallbackResult::Invalid(
            "Escrow currency too long (max 16 chars)".into(),
        ));
    }
    for condition in &escrow.release_conditions {
        if condition.len() > 4096 {
            return Ok(ValidateCallbackResult::Invalid(
                "Escrow release condition too long (max 4096 chars)".into(),
            ));
        }
    }
    // --- Existing validation ---
    if !escrow.amount.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Escrow amount must be a finite number".into(),
        ));
    }
    if escrow.amount <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Escrow amount must be positive".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_escrow(_action: Update, escrow: Escrow) -> ExternResult<ValidateCallbackResult> {
    if !escrow.amount.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Escrow amount must be a finite number".into(),
        ));
    }
    if escrow.amount <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Escrow amount must be positive".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Factory Functions
    // ========================================================================

    fn create_test_timestamp() -> Timestamp {
        Timestamp::from_micros(1_000_000)
    }

    fn create_test_transfer(from_did: &str, to_did: &str) -> Transfer {
        Transfer {
            id: "transfer-001".to_string(),
            property_id: "property-123".to_string(),
            from_did: from_did.to_string(),
            to_did: to_did.to_string(),
            transfer_type: TransferType::Sale,
            price: Some(250_000.0),
            currency: Some("USD".to_string()),
            conditions: vec![],
            status: TransferStatus::Initiated,
            initiated: create_test_timestamp(),
            completed: None,
        }
    }

    fn create_test_transfer_with_conditions() -> Transfer {
        Transfer {
            id: "transfer-002".to_string(),
            property_id: "property-456".to_string(),
            from_did: "did:key:seller123".to_string(),
            to_did: "did:key:buyer456".to_string(),
            transfer_type: TransferType::Sale,
            price: Some(500_000.0),
            currency: Some("EUR".to_string()),
            conditions: vec![
                TransferCondition {
                    condition_type: ConditionType::PaymentReceived,
                    description: "Full payment received".to_string(),
                    satisfied: false,
                    verified_by: None,
                },
                TransferCondition {
                    condition_type: ConditionType::InspectionComplete,
                    description: "Home inspection completed".to_string(),
                    satisfied: false,
                    verified_by: Some("did:key:inspector789".to_string()),
                },
            ],
            status: TransferStatus::ConditionsPending,
            initiated: create_test_timestamp(),
            completed: None,
        }
    }

    fn create_test_escrow(amount: f64) -> Escrow {
        Escrow {
            id: "escrow-001".to_string(),
            transfer_id: "transfer-001".to_string(),
            escrow_agent_did: Some("did:key:agent123".to_string()),
            amount,
            currency: "USD".to_string(),
            funded: false,
            release_conditions: vec!["Payment verified".to_string()],
            created: create_test_timestamp(),
            released: None,
        }
    }

    fn create_test_entry_creation_action() -> EntryCreationAction {
        EntryCreationAction::Create(Create {
            author: AgentPubKey::from_raw_36(vec![0u8; 36]),
            timestamp: create_test_timestamp(),
            action_seq: 0,
            prev_action: ActionHash::from_raw_36(vec![0u8; 36]),
            entry_type: EntryType::App(AppEntryDef {
                entry_index: 0.into(),
                zome_index: 0.into(),
                visibility: EntryVisibility::Public,
            }),
            entry_hash: EntryHash::from_raw_36(vec![0u8; 36]),
            weight: Default::default(),
        })
    }

    fn create_test_update_action() -> Update {
        Update {
            author: AgentPubKey::from_raw_36(vec![0u8; 36]),
            timestamp: create_test_timestamp(),
            action_seq: 1,
            prev_action: ActionHash::from_raw_36(vec![0u8; 36]),
            entry_type: EntryType::App(AppEntryDef {
                entry_index: 0.into(),
                zome_index: 0.into(),
                visibility: EntryVisibility::Public,
            }),
            entry_hash: EntryHash::from_raw_36(vec![0u8; 36]),
            original_action_address: ActionHash::from_raw_36(vec![0u8; 36]),
            original_entry_address: EntryHash::from_raw_36(vec![0u8; 36]),
            weight: Default::default(),
        }
    }

    // ========================================================================
    // Transfer DID Validation Tests
    // ========================================================================

    #[test]
    fn test_transfer_valid_dids() {
        let transfer = create_test_transfer("did:key:seller123", "did:key:buyer456");
        let result = validate_create_transfer(create_test_entry_creation_action(), transfer);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_transfer_valid_did_web() {
        let transfer = create_test_transfer("did:web:example.com", "did:web:buyer.com");
        let result = validate_create_transfer(create_test_entry_creation_action(), transfer);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_transfer_valid_did_pkh() {
        let transfer = create_test_transfer("did:pkh:eth:0x123", "did:pkh:btc:bc1abc");
        let result = validate_create_transfer(create_test_entry_creation_action(), transfer);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_transfer_invalid_from_did_empty() {
        let transfer = create_test_transfer("", "did:key:buyer456");
        let result = validate_create_transfer(create_test_entry_creation_action(), transfer);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Seller must be a valid DID"));
            }
            _ => panic!("Expected invalid result"),
        }
    }

    #[test]
    fn test_transfer_invalid_from_did_no_prefix() {
        let transfer = create_test_transfer("key:seller123", "did:key:buyer456");
        let result = validate_create_transfer(create_test_entry_creation_action(), transfer);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Seller must be a valid DID"));
            }
            _ => panic!("Expected invalid result"),
        }
    }

    #[test]
    fn test_transfer_invalid_from_did_plain_string() {
        let transfer = create_test_transfer("seller123", "did:key:buyer456");
        let result = validate_create_transfer(create_test_entry_creation_action(), transfer);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Seller must be a valid DID"));
            }
            _ => panic!("Expected invalid result"),
        }
    }

    #[test]
    fn test_transfer_invalid_to_did_empty() {
        let transfer = create_test_transfer("did:key:seller123", "");
        let result = validate_create_transfer(create_test_entry_creation_action(), transfer);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Buyer must be a valid DID"));
            }
            _ => panic!("Expected invalid result"),
        }
    }

    #[test]
    fn test_transfer_invalid_to_did_no_prefix() {
        let transfer = create_test_transfer("did:key:seller123", "key:buyer456");
        let result = validate_create_transfer(create_test_entry_creation_action(), transfer);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Buyer must be a valid DID"));
            }
            _ => panic!("Expected invalid result"),
        }
    }

    #[test]
    fn test_transfer_invalid_to_did_plain_string() {
        let transfer = create_test_transfer("did:key:seller123", "buyer456");
        let result = validate_create_transfer(create_test_entry_creation_action(), transfer);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Buyer must be a valid DID"));
            }
            _ => panic!("Expected invalid result"),
        }
    }

    #[test]
    fn test_transfer_invalid_both_dids_missing_prefix() {
        let transfer = create_test_transfer("seller123", "buyer456");
        let result = validate_create_transfer(create_test_entry_creation_action(), transfer);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Seller must be a valid DID"));
            }
            _ => panic!("Expected invalid result"),
        }
    }

    // ========================================================================
    // Self-Transfer Prevention Tests
    // ========================================================================

    #[test]
    fn test_transfer_self_transfer_same_did() {
        let transfer = create_test_transfer("did:key:same123", "did:key:same123");
        let result = validate_create_transfer(create_test_entry_creation_action(), transfer);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Cannot transfer to yourself"));
            }
            _ => panic!("Expected invalid result"),
        }
    }

    #[test]
    fn test_transfer_self_transfer_different_methods() {
        // Even different DID methods with same identifier should be caught
        let transfer = create_test_transfer("did:key:abc123", "did:key:abc123");
        let result = validate_create_transfer(create_test_entry_creation_action(), transfer);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Cannot transfer to yourself"));
            }
            _ => panic!("Expected invalid result"),
        }
    }

    #[test]
    fn test_transfer_not_self_transfer_similar_dids() {
        // Similar but different DIDs should be allowed
        let transfer = create_test_transfer("did:key:abc123", "did:key:abc124");
        let result = validate_create_transfer(create_test_entry_creation_action(), transfer);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    // ========================================================================
    // Transfer Update Validation Tests
    // ========================================================================

    #[test]
    fn test_transfer_update_always_valid() {
        let transfer = create_test_transfer("did:key:seller123", "did:key:buyer456");
        let result = validate_update_transfer(create_test_update_action(), transfer);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_transfer_update_invalid_dids_allowed() {
        // Update validation doesn't check DIDs
        let transfer = create_test_transfer("invalid", "also-invalid");
        let result = validate_update_transfer(create_test_update_action(), transfer);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_transfer_update_self_transfer_allowed() {
        // Update validation doesn't check self-transfer
        let transfer = create_test_transfer("did:key:same", "did:key:same");
        let result = validate_update_transfer(create_test_update_action(), transfer);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    // ========================================================================
    // Escrow Amount Validation Tests
    // ========================================================================

    #[test]
    fn test_escrow_valid_amount_large() {
        let escrow = create_test_escrow(500_000.0);
        let result = validate_create_escrow(create_test_entry_creation_action(), escrow);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_escrow_valid_amount_small() {
        let escrow = create_test_escrow(0.01);
        let result = validate_create_escrow(create_test_entry_creation_action(), escrow);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_escrow_valid_amount_one_cent() {
        let escrow = create_test_escrow(0.01);
        let result = validate_create_escrow(create_test_entry_creation_action(), escrow);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_escrow_valid_amount_fractional() {
        let escrow = create_test_escrow(123.45);
        let result = validate_create_escrow(create_test_entry_creation_action(), escrow);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_escrow_invalid_amount_zero() {
        let escrow = create_test_escrow(0.0);
        let result = validate_create_escrow(create_test_entry_creation_action(), escrow);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Escrow amount must be positive"));
            }
            _ => panic!("Expected invalid result"),
        }
    }

    #[test]
    fn test_escrow_invalid_amount_negative() {
        let escrow = create_test_escrow(-1.0);
        let result = validate_create_escrow(create_test_entry_creation_action(), escrow);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Escrow amount must be positive"));
            }
            _ => panic!("Expected invalid result"),
        }
    }

    #[test]
    fn test_escrow_invalid_amount_large_negative() {
        let escrow = create_test_escrow(-1_000_000.0);
        let result = validate_create_escrow(create_test_entry_creation_action(), escrow);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Escrow amount must be positive"));
            }
            _ => panic!("Expected invalid result"),
        }
    }

    // ========================================================================
    // Escrow Update Validation Tests
    // ========================================================================

    #[test]
    fn test_escrow_update_valid_amount() {
        let escrow = create_test_escrow(250_000.0);
        let result = validate_update_escrow(create_test_update_action(), escrow);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_escrow_update_invalid_amount_zero() {
        let escrow = create_test_escrow(0.0);
        let result = validate_update_escrow(create_test_update_action(), escrow);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Escrow amount must be positive"));
            }
            _ => panic!("Expected invalid result"),
        }
    }

    #[test]
    fn test_escrow_update_invalid_amount_negative() {
        let escrow = create_test_escrow(-100.0);
        let result = validate_update_escrow(create_test_update_action(), escrow);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Escrow amount must be positive"));
            }
            _ => panic!("Expected invalid result"),
        }
    }

    // ========================================================================
    // TransferType Serde Roundtrip Tests
    // ========================================================================

    #[test]
    fn test_transfer_type_sale_serde() {
        let transfer_type = TransferType::Sale;
        let serialized = serde_json::to_string(&transfer_type).unwrap();
        let deserialized: TransferType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(transfer_type, deserialized);
    }

    #[test]
    fn test_transfer_type_gift_serde() {
        let transfer_type = TransferType::Gift;
        let serialized = serde_json::to_string(&transfer_type).unwrap();
        let deserialized: TransferType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(transfer_type, deserialized);
    }

    #[test]
    fn test_transfer_type_inheritance_serde() {
        let transfer_type = TransferType::Inheritance;
        let serialized = serde_json::to_string(&transfer_type).unwrap();
        let deserialized: TransferType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(transfer_type, deserialized);
    }

    #[test]
    fn test_transfer_type_court_order_serde() {
        let transfer_type = TransferType::CourtOrder;
        let serialized = serde_json::to_string(&transfer_type).unwrap();
        let deserialized: TransferType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(transfer_type, deserialized);
    }

    #[test]
    fn test_transfer_type_exchange_serde() {
        let transfer_type = TransferType::Exchange;
        let serialized = serde_json::to_string(&transfer_type).unwrap();
        let deserialized: TransferType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(transfer_type, deserialized);
    }

    #[test]
    fn test_transfer_type_other_serde() {
        let transfer_type = TransferType::Other;
        let serialized = serde_json::to_string(&transfer_type).unwrap();
        let deserialized: TransferType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(transfer_type, deserialized);
    }

    // ========================================================================
    // ConditionType Serde Roundtrip Tests
    // ========================================================================

    #[test]
    fn test_condition_type_payment_received_serde() {
        let condition_type = ConditionType::PaymentReceived;
        let serialized = serde_json::to_string(&condition_type).unwrap();
        let deserialized: ConditionType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(condition_type, deserialized);
    }

    #[test]
    fn test_condition_type_inspection_complete_serde() {
        let condition_type = ConditionType::InspectionComplete;
        let serialized = serde_json::to_string(&condition_type).unwrap();
        let deserialized: ConditionType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(condition_type, deserialized);
    }

    #[test]
    fn test_condition_type_title_clear_serde() {
        let condition_type = ConditionType::TitleClear;
        let serialized = serde_json::to_string(&condition_type).unwrap();
        let deserialized: ConditionType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(condition_type, deserialized);
    }

    #[test]
    fn test_condition_type_documents_signed_serde() {
        let condition_type = ConditionType::DocumentsSigned;
        let serialized = serde_json::to_string(&condition_type).unwrap();
        let deserialized: ConditionType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(condition_type, deserialized);
    }

    #[test]
    fn test_condition_type_taxes_paid_serde() {
        let condition_type = ConditionType::TaxesPaid;
        let serialized = serde_json::to_string(&condition_type).unwrap();
        let deserialized: ConditionType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(condition_type, deserialized);
    }

    #[test]
    fn test_condition_type_custom_serde() {
        let condition_type = ConditionType::Custom("Environmental clearance".to_string());
        let serialized = serde_json::to_string(&condition_type).unwrap();
        let deserialized: ConditionType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(condition_type, deserialized);
    }

    // ========================================================================
    // TransferStatus Serde Roundtrip Tests
    // ========================================================================

    #[test]
    fn test_transfer_status_initiated_serde() {
        let status = TransferStatus::Initiated;
        let serialized = serde_json::to_string(&status).unwrap();
        let deserialized: TransferStatus = serde_json::from_str(&serialized).unwrap();
        assert_eq!(status, deserialized);
    }

    #[test]
    fn test_transfer_status_awaiting_acceptance_serde() {
        let status = TransferStatus::AwaitingAcceptance;
        let serialized = serde_json::to_string(&status).unwrap();
        let deserialized: TransferStatus = serde_json::from_str(&serialized).unwrap();
        assert_eq!(status, deserialized);
    }

    #[test]
    fn test_transfer_status_in_escrow_serde() {
        let status = TransferStatus::InEscrow;
        let serialized = serde_json::to_string(&status).unwrap();
        let deserialized: TransferStatus = serde_json::from_str(&serialized).unwrap();
        assert_eq!(status, deserialized);
    }

    #[test]
    fn test_transfer_status_conditions_pending_serde() {
        let status = TransferStatus::ConditionsPending;
        let serialized = serde_json::to_string(&status).unwrap();
        let deserialized: TransferStatus = serde_json::from_str(&serialized).unwrap();
        assert_eq!(status, deserialized);
    }

    #[test]
    fn test_transfer_status_completed_serde() {
        let status = TransferStatus::Completed;
        let serialized = serde_json::to_string(&status).unwrap();
        let deserialized: TransferStatus = serde_json::from_str(&serialized).unwrap();
        assert_eq!(status, deserialized);
    }

    #[test]
    fn test_transfer_status_cancelled_serde() {
        let status = TransferStatus::Cancelled;
        let serialized = serde_json::to_string(&status).unwrap();
        let deserialized: TransferStatus = serde_json::from_str(&serialized).unwrap();
        assert_eq!(status, deserialized);
    }

    #[test]
    fn test_transfer_status_disputed_serde() {
        let status = TransferStatus::Disputed;
        let serialized = serde_json::to_string(&status).unwrap();
        let deserialized: TransferStatus = serde_json::from_str(&serialized).unwrap();
        assert_eq!(status, deserialized);
    }

    // ========================================================================
    // Optional Fields Tests
    // ========================================================================

    #[test]
    fn test_transfer_with_none_price() {
        let mut transfer = create_test_transfer("did:key:seller", "did:key:buyer");
        transfer.price = None;
        assert!(transfer.price.is_none());
    }

    #[test]
    fn test_transfer_with_none_currency() {
        let mut transfer = create_test_transfer("did:key:seller", "did:key:buyer");
        transfer.currency = None;
        assert!(transfer.currency.is_none());
    }

    #[test]
    fn test_transfer_with_none_completed() {
        let transfer = create_test_transfer("did:key:seller", "did:key:buyer");
        assert!(transfer.completed.is_none());
    }

    #[test]
    fn test_transfer_with_some_completed() {
        let mut transfer = create_test_transfer("did:key:seller", "did:key:buyer");
        transfer.completed = Some(Timestamp::from_micros(2_000_000));
        assert!(transfer.completed.is_some());
    }

    #[test]
    fn test_escrow_with_none_agent_did() {
        let mut escrow = create_test_escrow(100_000.0);
        escrow.escrow_agent_did = None;
        assert!(escrow.escrow_agent_did.is_none());
    }

    #[test]
    fn test_escrow_with_some_agent_did() {
        let escrow = create_test_escrow(100_000.0);
        assert!(escrow.escrow_agent_did.is_some());
    }

    #[test]
    fn test_escrow_with_none_released() {
        let escrow = create_test_escrow(100_000.0);
        assert!(escrow.released.is_none());
    }

    #[test]
    fn test_escrow_with_some_released() {
        let mut escrow = create_test_escrow(100_000.0);
        escrow.released = Some(Timestamp::from_micros(3_000_000));
        assert!(escrow.released.is_some());
    }

    // ========================================================================
    // TransferCondition Tests
    // ========================================================================

    #[test]
    fn test_transfer_condition_creation() {
        let condition = TransferCondition {
            condition_type: ConditionType::PaymentReceived,
            description: "Payment in full".to_string(),
            satisfied: true,
            verified_by: Some("did:key:verifier".to_string()),
        };
        assert_eq!(condition.condition_type, ConditionType::PaymentReceived);
        assert!(condition.satisfied);
        assert!(condition.verified_by.is_some());
    }

    #[test]
    fn test_transfer_condition_with_none_verifier() {
        let condition = TransferCondition {
            condition_type: ConditionType::TitleClear,
            description: "Title search pending".to_string(),
            satisfied: false,
            verified_by: None,
        };
        assert!(!condition.satisfied);
        assert!(condition.verified_by.is_none());
    }

    #[test]
    fn test_transfer_with_multiple_conditions() {
        let transfer = create_test_transfer_with_conditions();
        assert_eq!(transfer.conditions.len(), 2);
        assert_eq!(
            transfer.conditions[0].condition_type,
            ConditionType::PaymentReceived
        );
        assert_eq!(
            transfer.conditions[1].condition_type,
            ConditionType::InspectionComplete
        );
    }

    #[test]
    fn test_transfer_with_empty_conditions() {
        let transfer = create_test_transfer("did:key:seller", "did:key:buyer");
        assert_eq!(transfer.conditions.len(), 0);
    }

    // ========================================================================
    // Edge Cases and Integration Tests
    // ========================================================================

    #[test]
    fn test_transfer_all_types() {
        let types = vec![
            TransferType::Sale,
            TransferType::Gift,
            TransferType::Inheritance,
            TransferType::CourtOrder,
            TransferType::Exchange,
            TransferType::Other,
        ];

        for transfer_type in types {
            let mut transfer = create_test_transfer("did:key:seller", "did:key:buyer");
            transfer.transfer_type = transfer_type.clone();
            let result = validate_create_transfer(create_test_entry_creation_action(), transfer);
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
        }
    }

    #[test]
    fn test_transfer_all_statuses() {
        let statuses = vec![
            TransferStatus::Initiated,
            TransferStatus::AwaitingAcceptance,
            TransferStatus::InEscrow,
            TransferStatus::ConditionsPending,
            TransferStatus::Completed,
            TransferStatus::Cancelled,
            TransferStatus::Disputed,
        ];

        for status in statuses {
            let mut transfer = create_test_transfer("did:key:seller", "did:key:buyer");
            transfer.status = status.clone();
            let result = validate_create_transfer(create_test_entry_creation_action(), transfer);
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
        }
    }

    #[test]
    fn test_escrow_with_multiple_release_conditions() {
        let mut escrow = create_test_escrow(500_000.0);
        escrow.release_conditions = vec![
            "Payment verified".to_string(),
            "Title clear".to_string(),
            "Documents signed".to_string(),
        ];
        assert_eq!(escrow.release_conditions.len(), 3);
    }

    #[test]
    fn test_escrow_with_empty_release_conditions() {
        let mut escrow = create_test_escrow(100_000.0);
        escrow.release_conditions = vec![];
        assert_eq!(escrow.release_conditions.len(), 0);
    }

    #[test]
    fn test_escrow_funded_flag() {
        let mut escrow = create_test_escrow(250_000.0);
        assert!(!escrow.funded);
        escrow.funded = true;
        assert!(escrow.funded);
    }

    #[test]
    fn test_anchor_creation() {
        let anchor = Anchor("property-transfers".to_string());
        assert_eq!(anchor.0, "property-transfers");
    }

    #[test]
    fn test_transfer_condition_custom_type() {
        let condition = TransferCondition {
            condition_type: ConditionType::Custom("HOA approval".to_string()),
            description: "Homeowners association must approve".to_string(),
            satisfied: false,
            verified_by: None,
        };
        match condition.condition_type {
            ConditionType::Custom(ref s) => assert_eq!(s, "HOA approval"),
            _ => panic!("Expected Custom condition type"),
        }
    }

    #[test]
    fn test_transfer_serde_roundtrip() {
        let transfer = create_test_transfer_with_conditions();
        let serialized = serde_json::to_string(&transfer).unwrap();
        let deserialized: Transfer = serde_json::from_str(&serialized).unwrap();
        assert_eq!(transfer.id, deserialized.id);
        assert_eq!(transfer.from_did, deserialized.from_did);
        assert_eq!(transfer.to_did, deserialized.to_did);
        assert_eq!(transfer.conditions.len(), deserialized.conditions.len());
    }

    #[test]
    fn test_escrow_serde_roundtrip() {
        let escrow = create_test_escrow(123_456.78);
        let serialized = serde_json::to_string(&escrow).unwrap();
        let deserialized: Escrow = serde_json::from_str(&serialized).unwrap();
        assert_eq!(escrow.id, deserialized.id);
        assert_eq!(escrow.amount, deserialized.amount);
        assert_eq!(escrow.currency, deserialized.currency);
    }

    // ========================================================================
    // String Length & Empty Validation Tests
    // ========================================================================

    // --- Transfer ---

    #[test]
    fn test_transfer_empty_id() {
        let mut transfer = create_test_transfer("did:key:seller", "did:key:buyer");
        transfer.id = "".to_string();
        let result = validate_create_transfer(create_test_entry_creation_action(), transfer);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_transfer_whitespace_id() {
        let mut transfer = create_test_transfer("did:key:seller", "did:key:buyer");
        transfer.id = "   ".to_string();
        let result = validate_create_transfer(create_test_entry_creation_action(), transfer);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_transfer_id_too_long() {
        let mut transfer = create_test_transfer("did:key:seller", "did:key:buyer");
        transfer.id = "x".repeat(257);
        let result = validate_create_transfer(create_test_entry_creation_action(), transfer);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_transfer_id_at_limit() {
        let mut transfer = create_test_transfer("did:key:seller", "did:key:buyer");
        transfer.id = "x".repeat(64);
        let result = validate_create_transfer(create_test_entry_creation_action(), transfer);
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_transfer_empty_property_id() {
        let mut transfer = create_test_transfer("did:key:seller", "did:key:buyer");
        transfer.property_id = "".to_string();
        let result = validate_create_transfer(create_test_entry_creation_action(), transfer);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_transfer_property_id_too_long() {
        let mut transfer = create_test_transfer("did:key:seller", "did:key:buyer");
        transfer.property_id = "x".repeat(257);
        let result = validate_create_transfer(create_test_entry_creation_action(), transfer);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_transfer_currency_too_long() {
        let mut transfer = create_test_transfer("did:key:seller", "did:key:buyer");
        transfer.currency = Some("x".repeat(17));
        let result = validate_create_transfer(create_test_entry_creation_action(), transfer);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_transfer_currency_at_limit() {
        let mut transfer = create_test_transfer("did:key:seller", "did:key:buyer");
        transfer.currency = Some("x".repeat(16));
        let result = validate_create_transfer(create_test_entry_creation_action(), transfer);
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    // --- Escrow ---

    #[test]
    fn test_escrow_empty_id() {
        let mut escrow = create_test_escrow(1000.0);
        escrow.id = "".to_string();
        let result = validate_create_escrow(create_test_entry_creation_action(), escrow);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_escrow_whitespace_id() {
        let mut escrow = create_test_escrow(1000.0);
        escrow.id = "   ".to_string();
        let result = validate_create_escrow(create_test_entry_creation_action(), escrow);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_escrow_id_too_long() {
        let mut escrow = create_test_escrow(1000.0);
        escrow.id = "x".repeat(257);
        let result = validate_create_escrow(create_test_entry_creation_action(), escrow);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_escrow_empty_transfer_id() {
        let mut escrow = create_test_escrow(1000.0);
        escrow.transfer_id = "".to_string();
        let result = validate_create_escrow(create_test_entry_creation_action(), escrow);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_escrow_transfer_id_too_long() {
        let mut escrow = create_test_escrow(1000.0);
        escrow.transfer_id = "x".repeat(257);
        let result = validate_create_escrow(create_test_entry_creation_action(), escrow);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_escrow_currency_too_long() {
        let mut escrow = create_test_escrow(1000.0);
        escrow.currency = "x".repeat(17);
        let result = validate_create_escrow(create_test_entry_creation_action(), escrow);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_escrow_release_condition_too_long() {
        let mut escrow = create_test_escrow(1000.0);
        escrow.release_conditions = vec!["x".repeat(4097)];
        let result = validate_create_escrow(create_test_entry_creation_action(), escrow);
        assert!(matches!(
            result.unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_escrow_release_condition_at_limit() {
        let mut escrow = create_test_escrow(1000.0);
        escrow.release_conditions = vec!["x".repeat(4096)];
        let result = validate_create_escrow(create_test_entry_creation_action(), escrow);
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }
}
