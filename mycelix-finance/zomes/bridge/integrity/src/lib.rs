//! Finance Bridge Integrity Zome
//! Updated to use HDI 0.7 patterns with FlatOp validation
//!
//! Entry types for cross-hApp payment processing, collateral management,
//! and collateral bridge deposits for SAP minting.

use hdi::prelude::*;

// =============================================================================
// STRING LENGTH LIMITS — Prevent DHT bloat attacks
// =============================================================================

/// Maximum length for DID strings
const MAX_DID_LEN: usize = 256;
/// Maximum length for reference/ID strings
const MAX_REFERENCE_LEN: usize = 1024;
/// Maximum length for event payloads
const MAX_PAYLOAD_LEN: usize = 4096;
/// Maximum length for hApp ID strings
const MAX_HAPP_ID_LEN: usize = 256;

/// Cross-hApp payment request
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CrossHappPayment {
    pub id: String,
    pub source_happ: String,
    pub from_did: String,
    pub to_did: String,
    pub amount: u64,
    pub currency: String,
    pub reference: String,
    pub status: PaymentStatus,
    pub created_at: Timestamp,
    pub completed_at: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PaymentStatus {
    Pending,
    Processing,
    Completed,
    Failed,
    Refunded,
    Disputed,
}

/// Collateral registration from property hApp
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CollateralRegistration {
    pub id: String,
    pub owner_did: String,
    pub asset_type: AssetType,
    pub asset_id: String,
    pub source_happ: String,
    pub value_estimate: u64,
    pub currency: String,
    pub status: CollateralStatus,
    pub registered_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AssetType {
    RealEstate,
    Vehicle,
    Cryptocurrency,
    EnergyAsset,
    Equipment,
    Other(String),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum CollateralStatus {
    Available,
    Pledged,
    Frozen,
    Released,
}

/// Finance event for broadcasting
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct FinanceBridgeEvent {
    pub id: String,
    pub event_type: FinanceEventType,
    pub subject_did: String,
    pub amount: Option<u64>,
    pub payload: String,
    pub source_happ: String,
    pub timestamp: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum FinanceEventType {
    PaymentCompleted,
    CollateralPledged,
    CollateralReleased,
    CollateralDeposited,
    CollateralRedeemed,
    CommonsContributed,
    /// SAP minted via governance proposal or initial distribution
    SapMinted,
}

/// Collateral bridge deposit for minting SAP from external collateral
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CollateralBridgeDeposit {
    pub id: String,
    pub depositor_did: String,
    pub collateral_type: String, // "ETH" or "USDC"
    pub collateral_amount: u64,
    pub sap_minted: u64,
    pub oracle_rate: f64,
    pub status: BridgeDepositStatus,
    pub created_at: Timestamp,
    pub completed_at: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum BridgeDepositStatus {
    Pending,
    Confirmed,
    Redeemed,
    Failed,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    CrossHappPayment(CrossHappPayment),
    CollateralRegistration(CollateralRegistration),
    FinanceBridgeEvent(FinanceBridgeEvent),
    CollateralBridgeDeposit(CollateralBridgeDeposit),
}

#[hdk_link_types]
pub enum LinkTypes {
    DidToPayments,
    HappToPayments,
    CollateralRegistry,
    RecentEvents,
    DidToDeposits,
    DepositIdToDeposit,
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
                EntryTypes::CrossHappPayment(payment) => {
                    validate_create_cross_happ_payment(EntryCreationAction::Create(action), payment)
                }
                EntryTypes::CollateralRegistration(collateral) => {
                    validate_create_collateral_registration(
                        EntryCreationAction::Create(action),
                        collateral,
                    )
                }
                EntryTypes::FinanceBridgeEvent(event) => {
                    validate_create_finance_bridge_event(EntryCreationAction::Create(action), event)
                }
                EntryTypes::CollateralBridgeDeposit(deposit) => {
                    validate_create_collateral_bridge_deposit(
                        EntryCreationAction::Create(action),
                        deposit,
                    )
                }
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::CrossHappPayment(payment) => {
                    validate_update_cross_happ_payment(action, payment)
                }
                EntryTypes::CollateralRegistration(collateral) => {
                    validate_update_collateral_registration(action, collateral)
                }
                EntryTypes::FinanceBridgeEvent(_) => Ok(ValidateCallbackResult::Invalid(
                    "Finance events cannot be updated".into(),
                )),
                EntryTypes::CollateralBridgeDeposit(deposit) => {
                    validate_update_collateral_bridge_deposit(action, deposit)
                }
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address,
            target_address,
            ..
        } => match link_type {
            LinkTypes::DidToPayments | LinkTypes::DidToDeposits => {
                if base_address.as_ref().len() != 39 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "Link base must be a valid agent pubkey".into(),
                    ));
                }
                if target_address.as_ref().len() != 39 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "Link target must be a valid entry hash".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::HappToPayments | LinkTypes::CollateralRegistry => {
                if base_address.as_ref().len() != 39 || target_address.as_ref().len() != 39 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "Link must connect valid hashes".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::RecentEvents => {
                if target_address.as_ref().len() != 39 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "Link target must be a valid entry hash".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::DepositIdToDeposit => {
                if target_address.as_ref().len() != 39 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "Link target must be a valid action hash".into(),
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

fn validate_create_cross_happ_payment(
    _action: EntryCreationAction,
    payment: CrossHappPayment,
) -> ExternResult<ValidateCallbackResult> {
    // String length checks — prevent DHT bloat
    if payment.from_did.len() > MAX_DID_LEN || payment.to_did.len() > MAX_DID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "DID exceeds maximum length".into(),
        ));
    }
    if payment.source_happ.len() > MAX_HAPP_ID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Source hApp ID exceeds maximum length".into(),
        ));
    }
    if payment.reference.len() > MAX_REFERENCE_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Reference exceeds maximum length".into(),
        ));
    }
    if payment.id.len() > MAX_REFERENCE_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Payment ID exceeds maximum length".into(),
        ));
    }

    if payment.amount == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Amount must be positive".into(),
        ));
    }
    if payment.currency != "SAP" {
        return Ok(ValidateCallbackResult::Invalid(
            "CrossHappPayment currency must be SAP".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_cross_happ_payment(
    _action: Update,
    _payment: CrossHappPayment,
) -> ExternResult<ValidateCallbackResult> {
    // Status can be updated
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_collateral_registration(
    _action: EntryCreationAction,
    collateral: CollateralRegistration,
) -> ExternResult<ValidateCallbackResult> {
    // String length checks — prevent DHT bloat
    if collateral.owner_did.len() > MAX_DID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "DID exceeds maximum length".into(),
        ));
    }
    if collateral.asset_id.len() > MAX_REFERENCE_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Asset ID exceeds maximum length".into(),
        ));
    }
    if collateral.id.len() > MAX_REFERENCE_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Registration ID exceeds maximum length".into(),
        ));
    }
    if collateral.source_happ.len() > MAX_HAPP_ID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Source hApp ID exceeds maximum length".into(),
        ));
    }

    // Validate owner DID format
    if !collateral.owner_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Owner must be a valid DID".into(),
        ));
    }
    // Validate positive value estimate
    if collateral.value_estimate == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Value estimate must be positive".into(),
        ));
    }
    // Validate asset_id is non-empty
    if collateral.asset_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Asset ID is required".into(),
        ));
    }
    // Validate source_happ is non-empty
    if collateral.source_happ.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Source hApp is required".into(),
        ));
    }
    // Validate currency is non-empty
    if collateral.currency.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Currency is required".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_collateral_registration(
    _action: Update,
    collateral: CollateralRegistration,
) -> ExternResult<ValidateCallbackResult> {
    // Re-validate core invariants on update
    if !collateral.owner_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Owner must be a valid DID".into(),
        ));
    }
    if collateral.value_estimate == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Value estimate must be positive".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_finance_bridge_event(
    _action: EntryCreationAction,
    event: FinanceBridgeEvent,
) -> ExternResult<ValidateCallbackResult> {
    // String length checks — prevent DHT bloat
    if event.payload.len() > MAX_PAYLOAD_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Event payload exceeds maximum length".into(),
        ));
    }
    if event.subject_did.len() > MAX_DID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "DID exceeds maximum length".into(),
        ));
    }
    if event.source_happ.len() > MAX_HAPP_ID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Source hApp ID exceeds maximum length".into(),
        ));
    }
    if event.id.len() > MAX_REFERENCE_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Event ID exceeds maximum length".into(),
        ));
    }

    if event.source_happ.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Source hApp required".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_collateral_bridge_deposit(
    _action: EntryCreationAction,
    deposit: CollateralBridgeDeposit,
) -> ExternResult<ValidateCallbackResult> {
    // String length checks — prevent DHT bloat
    if deposit.depositor_did.len() > MAX_DID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "DID exceeds maximum length".into(),
        ));
    }
    if deposit.collateral_type.len() > MAX_HAPP_ID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Collateral type exceeds maximum length".into(),
        ));
    }
    if deposit.id.len() > MAX_REFERENCE_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Deposit ID exceeds maximum length".into(),
        ));
    }

    if !deposit.depositor_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Depositor must be a valid DID".into(),
        ));
    }
    if deposit.collateral_amount == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Collateral amount must be positive".into(),
        ));
    }
    if deposit.sap_minted == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "SAP minted must be positive".into(),
        ));
    }
    if deposit.oracle_rate <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Oracle rate must be positive".into(),
        ));
    }
    // Validate collateral type is supported
    if deposit.collateral_type != "ETH" && deposit.collateral_type != "USDC" {
        return Ok(ValidateCallbackResult::Invalid(
            "Collateral type must be ETH or USDC".into(),
        ));
    }
    // Validate sap_minted is consistent with collateral and oracle rate (allow +-1 for rounding)
    let expected = (deposit.collateral_amount as f64 * deposit.oracle_rate) as u64;
    if deposit.sap_minted < expected.saturating_sub(1)
        || deposit.sap_minted > expected.saturating_add(1)
    {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "SAP minted ({}) is inconsistent with collateral ({}) * oracle rate ({:.6}), expected ~{}",
            deposit.sap_minted, deposit.collateral_amount, deposit.oracle_rate, expected
        )));
    }
    // New deposits must start in Pending status
    if deposit.status != BridgeDepositStatus::Pending {
        return Ok(ValidateCallbackResult::Invalid(
            "New collateral bridge deposits must start with Pending status".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_collateral_bridge_deposit(
    _action: Update,
    deposit: CollateralBridgeDeposit,
) -> ExternResult<ValidateCallbackResult> {
    // Validate the deposit DID is still valid
    if !deposit.depositor_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Depositor must be a valid DID".into(),
        ));
    }
    // Core field invariants must still hold on update
    if deposit.collateral_amount == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Collateral amount must be positive".into(),
        ));
    }
    if deposit.sap_minted == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "SAP minted must be positive".into(),
        ));
    }
    if deposit.oracle_rate <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Oracle rate must be positive".into(),
        ));
    }
    if deposit.collateral_type != "ETH" && deposit.collateral_type != "USDC" {
        return Ok(ValidateCallbackResult::Invalid(
            "Collateral type must be ETH or USDC".into(),
        ));
    }
    // Status transition validation: cannot transition back to Pending.
    // Valid transitions are: Pending->Confirmed, Confirmed->Redeemed, Pending->Failed.
    // Full transition validation requires fetching the original entry, which is not
    // available in integrity validation. We enforce that updated status is never Pending
    // (a deposit cannot revert to Pending once it has progressed).
    if deposit.status == BridgeDepositStatus::Pending {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot transition deposit status back to Pending".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
