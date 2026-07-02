#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Finance Bridge Integrity Zome
//! Updated to use HDI 0.7 patterns with FlatOp validation
//!
//! Entry types for cross-hApp payment processing, collateral management,
//! and collateral bridge deposits for SAP minting.

use hdi::prelude::*;
use mycelix_bridge_entry_types::CrossClusterNotification;

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
    CovenantCreated,
    CovenantReleased,
    CollateralHealthUpdated,
    FiatDeposited,
    EnergyCertificateCreated,
    AgriAssetRegistered,
    MultiCollateralCreated,
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

/// Maximum length for JSON-serialized component lists
const MAX_COMPONENTS_JSON_LEN: usize = 16384;

/// Covenant on registered collateral (prevents transfer/sale while active).
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Covenant {
    pub id: String,
    pub collateral_id: String,
    pub restriction: String, // Serialized CovenantRestriction
    pub beneficiary_did: String,
    pub created_at: Timestamp,
    pub expires_at: Option<Timestamp>,
    pub released: bool,
    pub released_by: Option<String>,
    pub released_at: Option<Timestamp>,
}

/// Collateral health status (LTV monitoring).
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CollateralHealth {
    pub collateral_id: String,
    pub current_value: u64,
    pub obligation_amount: u64,
    pub ltv_ratio: f64,
    pub status: String, // "Healthy", "Warning", "MarginCall", "Liquidation"
    pub computed_at: Timestamp,
}

/// Fiat bridge deposit (one-way inbound: fiat -> SAP).
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct FiatBridgeDeposit {
    pub id: String,
    pub depositor_did: String,
    pub fiat_currency: String,
    pub fiat_amount: u64,
    pub sap_minted: u64,
    pub exchange_rate: f64,
    pub verifier_did: String,
    pub status: String, // "Pending", "Verified", "Rejected"
    pub external_reference: String,
    pub created_at: Timestamp,
    pub verified_at: Option<Timestamp>,
}

/// Energy production certificate (collateral-eligible).
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EnergyCertificate {
    pub id: String,
    pub project_id: String,
    pub source: String, // Energy source type
    pub kwh_produced: f64,
    pub period_start: Timestamp,
    pub period_end: Timestamp,
    pub location_lat: f64,
    pub location_lon: f64,
    pub producer_did: String,
    pub verifier_did: Option<String>,
    pub status: String, // "Pending", "Verified", "Rejected", "Pledged", "Retired"
    pub terra_atlas_id: Option<String>,
    pub sap_value: Option<u64>,
    pub created_at: Timestamp,
}

/// Agricultural asset registration (collateral-eligible).
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AgriculturalAsset {
    pub id: String,
    pub asset_type: String, // Serialized AgriAssetType
    pub quantity_kg: f64,
    pub location_lat: f64,
    pub location_lon: f64,
    pub production_date: Timestamp,
    pub viability_duration_micros: Option<i64>,
    pub producer_did: String,
    pub verifier_did: Option<String>,
    pub status: String, // "Pending", "Verified", "Rejected", "Pledged", "Consumed"
    pub sap_value: Option<u64>,
    pub created_at: Timestamp,
}

/// Multi-collateral position (basket of assets).
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MultiCollateralPosition {
    pub id: String,
    pub holder_did: String,
    pub components_json: String, // JSON-serialized Vec<CollateralComponent>
    pub aggregate_value: u64,
    pub aggregate_obligation: u64,
    pub diversification_bonus: f64,
    pub effective_ltv: f64,
    pub status: String, // Health status
    pub created_at: Timestamp,
    pub last_revalued_at: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    CrossHappPayment(CrossHappPayment),
    CollateralRegistration(CollateralRegistration),
    FinanceBridgeEvent(FinanceBridgeEvent),
    CollateralBridgeDeposit(CollateralBridgeDeposit),
    Covenant(Covenant),
    CollateralHealth(CollateralHealth),
    FiatBridgeDeposit(FiatBridgeDeposit),
    EnergyCertificate(EnergyCertificate),
    AgriculturalAsset(AgriculturalAsset),
    MultiCollateralPosition(MultiCollateralPosition),
    Notification(CrossClusterNotification),
}

#[hdk_link_types]
pub enum LinkTypes {
    DidToPayments,
    HappToPayments,
    CollateralRegistry,
    RecentEvents,
    DidToDeposits,
    DepositIdToDeposit,
    CollateralToCovenants,
    CollateralToHealth,
    FiatDepositRegistry,
    EnergyCertificateRegistry,
    AgriAssetRegistry,
    MultiCollateralRegistry,
    HolderToPositions,
    CovenantIdToCovenant,
    AgentToNotification,
    AllNotifications,
    NotificationSubscription,
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
                EntryTypes::Covenant(covenant) => {
                    validate_create_covenant(EntryCreationAction::Create(action), covenant)
                }
                EntryTypes::CollateralHealth(health) => {
                    validate_create_collateral_health(EntryCreationAction::Create(action), health)
                }
                EntryTypes::FiatBridgeDeposit(fiat) => {
                    validate_create_fiat_bridge_deposit(EntryCreationAction::Create(action), fiat)
                }
                EntryTypes::EnergyCertificate(cert) => {
                    validate_create_energy_certificate(EntryCreationAction::Create(action), cert)
                }
                EntryTypes::AgriculturalAsset(asset) => {
                    validate_create_agricultural_asset(EntryCreationAction::Create(action), asset)
                }
                EntryTypes::MultiCollateralPosition(pos) => {
                    validate_create_multi_collateral_position(
                        EntryCreationAction::Create(action),
                        pos,
                    )
                }
                EntryTypes::Notification(n) => {
                    mycelix_bridge_entry_types::validate_notification(&n)
                        .map(|()| ValidateCallbackResult::Valid)
                        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e)))
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
                EntryTypes::Covenant(covenant) => validate_update_covenant(action, covenant),
                EntryTypes::CollateralHealth(_) => {
                    // Health entries are append-only snapshots; updates are always valid
                    Ok(ValidateCallbackResult::Valid)
                }
                EntryTypes::FiatBridgeDeposit(fiat) => {
                    validate_update_fiat_bridge_deposit(action, fiat)
                }
                EntryTypes::EnergyCertificate(cert) => {
                    validate_update_energy_certificate(action, cert)
                }
                EntryTypes::AgriculturalAsset(asset) => {
                    validate_update_agricultural_asset(action, asset)
                }
                EntryTypes::MultiCollateralPosition(pos) => {
                    validate_update_multi_collateral_position(action, pos)
                }
                EntryTypes::Notification(_) => Ok(ValidateCallbackResult::Invalid(
                    "Notifications cannot be updated".into(),
                )),
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
            LinkTypes::DepositIdToDeposit | LinkTypes::CovenantIdToCovenant => {
                if target_address.as_ref().len() != 39 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "Link target must be a valid action hash".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::CollateralToCovenants
            | LinkTypes::CollateralToHealth
            | LinkTypes::FiatDepositRegistry
            | LinkTypes::EnergyCertificateRegistry
            | LinkTypes::AgriAssetRegistry
            | LinkTypes::MultiCollateralRegistry
            | LinkTypes::HolderToPositions => {
                if base_address.as_ref().len() != 39 || target_address.as_ref().len() != 39 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "Link must connect valid hashes".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentToNotification
            | LinkTypes::AllNotifications
            | LinkTypes::NotificationSubscription => Ok(ValidateCallbackResult::Valid),
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
    if !deposit.oracle_rate.is_finite() || deposit.oracle_rate <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Oracle rate must be a finite positive number".into(),
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
    if !deposit.oracle_rate.is_finite() || deposit.oracle_rate <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Oracle rate must be a finite positive number".into(),
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

// =============================================================================
// Covenant validation
// =============================================================================

fn validate_create_covenant(
    _action: EntryCreationAction,
    covenant: Covenant,
) -> ExternResult<ValidateCallbackResult> {
    if covenant.id.is_empty() || covenant.id.len() > MAX_REFERENCE_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Covenant ID invalid".into(),
        ));
    }
    if covenant.collateral_id.is_empty() || covenant.collateral_id.len() > MAX_REFERENCE_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Collateral ID invalid".into(),
        ));
    }
    if !covenant.beneficiary_did.starts_with("did:") || covenant.beneficiary_did.len() > MAX_DID_LEN
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Beneficiary DID invalid".into(),
        ));
    }
    if covenant.restriction.is_empty() || covenant.restriction.len() > MAX_PAYLOAD_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Restriction must be non-empty and within size limits".into(),
        ));
    }
    // New covenants must not be pre-released
    if covenant.released {
        return Ok(ValidateCallbackResult::Invalid(
            "New covenants cannot be pre-released".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_covenant(
    _action: Update,
    covenant: Covenant,
) -> ExternResult<ValidateCallbackResult> {
    // Core invariants must hold
    if !covenant.beneficiary_did.starts_with("did:") || covenant.beneficiary_did.len() > MAX_DID_LEN
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Beneficiary DID invalid".into(),
        ));
    }
    if covenant.id.is_empty() || covenant.id.len() > MAX_REFERENCE_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Covenant ID invalid".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

// =============================================================================
// CollateralHealth validation
// =============================================================================

fn validate_create_collateral_health(
    _action: EntryCreationAction,
    health: CollateralHealth,
) -> ExternResult<ValidateCallbackResult> {
    if health.collateral_id.is_empty() || health.collateral_id.len() > MAX_REFERENCE_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Collateral ID invalid".into(),
        ));
    }
    if !health.ltv_ratio.is_finite() || health.ltv_ratio < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "LTV ratio must be a finite non-negative number".into(),
        ));
    }
    let valid_statuses = ["Healthy", "Warning", "MarginCall", "Liquidation"];
    if !valid_statuses.contains(&health.status.as_str()) {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Invalid health status: {}. Must be one of: {:?}",
            health.status, valid_statuses
        )));
    }
    Ok(ValidateCallbackResult::Valid)
}

// =============================================================================
// FiatBridgeDeposit validation
// =============================================================================

fn validate_create_fiat_bridge_deposit(
    _action: EntryCreationAction,
    fiat: FiatBridgeDeposit,
) -> ExternResult<ValidateCallbackResult> {
    if fiat.id.is_empty() || fiat.id.len() > MAX_REFERENCE_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Fiat deposit ID invalid".into(),
        ));
    }
    if !fiat.depositor_did.starts_with("did:") || fiat.depositor_did.len() > MAX_DID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Depositor DID invalid".into(),
        ));
    }
    if !fiat.verifier_did.starts_with("did:") || fiat.verifier_did.len() > MAX_DID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Verifier DID invalid".into(),
        ));
    }
    if fiat.fiat_amount == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Fiat amount must be positive".into(),
        ));
    }
    if fiat.fiat_currency.is_empty() || fiat.fiat_currency.len() > 8 {
        return Ok(ValidateCallbackResult::Invalid(
            "Fiat currency must be 1-8 characters".into(),
        ));
    }
    if !fiat.exchange_rate.is_finite() || fiat.exchange_rate <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Exchange rate must be a finite positive number".into(),
        ));
    }
    if fiat.external_reference.len() > MAX_REFERENCE_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "External reference exceeds maximum length".into(),
        ));
    }
    // New fiat deposits must start as Pending
    if fiat.status != "Pending" {
        return Ok(ValidateCallbackResult::Invalid(
            "New fiat deposits must start with Pending status".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_fiat_bridge_deposit(
    _action: Update,
    fiat: FiatBridgeDeposit,
) -> ExternResult<ValidateCallbackResult> {
    if !fiat.depositor_did.starts_with("did:") || fiat.depositor_did.len() > MAX_DID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Depositor DID invalid".into(),
        ));
    }
    // Cannot revert to Pending
    if fiat.status == "Pending" {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot transition fiat deposit status back to Pending".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

// =============================================================================
// EnergyCertificate validation
// =============================================================================

fn validate_create_energy_certificate(
    _action: EntryCreationAction,
    cert: EnergyCertificate,
) -> ExternResult<ValidateCallbackResult> {
    if cert.id.is_empty() || cert.id.len() > MAX_REFERENCE_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Certificate ID invalid".into(),
        ));
    }
    if cert.project_id.is_empty() || cert.project_id.len() > MAX_REFERENCE_LEN {
        return Ok(ValidateCallbackResult::Invalid("Project ID invalid".into()));
    }
    if !cert.producer_did.starts_with("did:") || cert.producer_did.len() > MAX_DID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Producer DID invalid".into(),
        ));
    }
    if !cert.kwh_produced.is_finite() || cert.kwh_produced <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "kWh produced must be a finite positive number".into(),
        ));
    }
    if !cert.location_lat.is_finite() || cert.location_lat < -90.0 || cert.location_lat > 90.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Latitude must be between -90 and 90".into(),
        ));
    }
    if !cert.location_lon.is_finite() || cert.location_lon < -180.0 || cert.location_lon > 180.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Longitude must be between -180 and 180".into(),
        ));
    }
    if cert.source.is_empty() || cert.source.len() > MAX_REFERENCE_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Energy source invalid".into(),
        ));
    }
    if cert.status != "Pending" {
        return Ok(ValidateCallbackResult::Invalid(
            "New energy certificates must start with Pending status".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_energy_certificate(
    _action: Update,
    cert: EnergyCertificate,
) -> ExternResult<ValidateCallbackResult> {
    if !cert.producer_did.starts_with("did:") || cert.producer_did.len() > MAX_DID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Producer DID invalid".into(),
        ));
    }
    // Cannot revert to Pending
    if cert.status == "Pending" {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot transition certificate status back to Pending".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

// =============================================================================
// AgriculturalAsset validation
// =============================================================================

fn validate_create_agricultural_asset(
    _action: EntryCreationAction,
    asset: AgriculturalAsset,
) -> ExternResult<ValidateCallbackResult> {
    if asset.id.is_empty() || asset.id.len() > MAX_REFERENCE_LEN {
        return Ok(ValidateCallbackResult::Invalid("Asset ID invalid".into()));
    }
    if !asset.producer_did.starts_with("did:") || asset.producer_did.len() > MAX_DID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Producer DID invalid".into(),
        ));
    }
    if asset.asset_type.is_empty() || asset.asset_type.len() > MAX_REFERENCE_LEN {
        return Ok(ValidateCallbackResult::Invalid("Asset type invalid".into()));
    }
    if !asset.quantity_kg.is_finite() || asset.quantity_kg <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Quantity must be a finite positive number".into(),
        ));
    }
    if !asset.location_lat.is_finite() || asset.location_lat < -90.0 || asset.location_lat > 90.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Latitude must be between -90 and 90".into(),
        ));
    }
    if !asset.location_lon.is_finite() || asset.location_lon < -180.0 || asset.location_lon > 180.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Longitude must be between -180 and 180".into(),
        ));
    }
    if asset.status != "Pending" {
        return Ok(ValidateCallbackResult::Invalid(
            "New agricultural assets must start with Pending status".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_agricultural_asset(
    _action: Update,
    asset: AgriculturalAsset,
) -> ExternResult<ValidateCallbackResult> {
    if !asset.producer_did.starts_with("did:") || asset.producer_did.len() > MAX_DID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Producer DID invalid".into(),
        ));
    }
    if asset.status == "Pending" {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot transition asset status back to Pending".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

// =============================================================================
// MultiCollateralPosition validation
// =============================================================================

fn validate_create_multi_collateral_position(
    _action: EntryCreationAction,
    pos: MultiCollateralPosition,
) -> ExternResult<ValidateCallbackResult> {
    if pos.id.is_empty() || pos.id.len() > MAX_REFERENCE_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Position ID invalid".into(),
        ));
    }
    if !pos.holder_did.starts_with("did:") || pos.holder_did.len() > MAX_DID_LEN {
        return Ok(ValidateCallbackResult::Invalid("Holder DID invalid".into()));
    }
    if pos.components_json.is_empty() || pos.components_json.len() > MAX_COMPONENTS_JSON_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Components JSON must be non-empty and within size limits".into(),
        ));
    }
    if !pos.diversification_bonus.is_finite()
        || pos.diversification_bonus < 0.0
        || pos.diversification_bonus > 1.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Diversification bonus must be between 0.0 and 1.0".into(),
        ));
    }
    if !pos.effective_ltv.is_finite() || pos.effective_ltv < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Effective LTV must be a finite non-negative number".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_multi_collateral_position(
    _action: Update,
    pos: MultiCollateralPosition,
) -> ExternResult<ValidateCallbackResult> {
    if !pos.holder_did.starts_with("did:") || pos.holder_did.len() > MAX_DID_LEN {
        return Ok(ValidateCallbackResult::Invalid("Holder DID invalid".into()));
    }
    if !pos.effective_ltv.is_finite() || pos.effective_ltv < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Effective LTV must be a finite non-negative number".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(micros: i64) -> Timestamp {
        Timestamp::from_micros(micros)
    }

    fn make_create() -> Create {
        Create {
            author: AgentPubKey::from_raw_36(vec![0; 36]),
            timestamp: ts(1_000_000),
            action_seq: 0,
            prev_action: ActionHash::from_raw_36(vec![0; 36]),
            entry_type: EntryType::CapClaim,
            entry_hash: EntryHash::from_raw_36(vec![0; 36]),
            weight: Default::default(),
        }
    }

    fn make_update() -> Update {
        Update {
            author: AgentPubKey::from_raw_36(vec![0; 36]),
            timestamp: ts(2_000_000),
            action_seq: 1,
            prev_action: ActionHash::from_raw_36(vec![0; 36]),
            original_action_address: ActionHash::from_raw_36(vec![0; 36]),
            original_entry_address: EntryHash::from_raw_36(vec![0; 36]),
            entry_type: EntryType::CapClaim,
            entry_hash: EntryHash::from_raw_36(vec![0; 36]),
            weight: Default::default(),
        }
    }

    // =========================================================================
    // Helpers: valid entry constructors
    // =========================================================================

    fn valid_payment() -> CrossHappPayment {
        CrossHappPayment {
            id: "pay:test:001".into(),
            source_happ: "commons".into(),
            from_did: "did:mycelix:alice".into(),
            to_did: "did:mycelix:bob".into(),
            amount: 100,
            currency: "SAP".into(),
            reference: "invoice-42".into(),
            status: PaymentStatus::Pending,
            created_at: ts(1_000_000),
            completed_at: None,
        }
    }

    fn valid_collateral() -> CollateralRegistration {
        CollateralRegistration {
            id: "col:test:001".into(),
            owner_did: "did:mycelix:alice".into(),
            asset_type: AssetType::RealEstate,
            asset_id: "property-123".into(),
            source_happ: "commons".into(),
            value_estimate: 500_000,
            currency: "SAP".into(),
            status: CollateralStatus::Available,
            registered_at: ts(1_000_000),
        }
    }

    fn valid_event() -> FinanceBridgeEvent {
        FinanceBridgeEvent {
            id: "evt:test:001".into(),
            event_type: FinanceEventType::PaymentCompleted,
            subject_did: "did:mycelix:alice".into(),
            amount: Some(100),
            payload: r#"{"tx":"abc"}"#.into(),
            source_happ: "commons".into(),
            timestamp: ts(1_000_000),
        }
    }

    fn valid_deposit() -> CollateralBridgeDeposit {
        CollateralBridgeDeposit {
            id: "dep:test:001".into(),
            depositor_did: "did:mycelix:alice".into(),
            collateral_type: "ETH".into(),
            collateral_amount: 1000,
            sap_minted: 2000,
            oracle_rate: 2.0,
            status: BridgeDepositStatus::Pending,
            created_at: ts(1_000_000),
            completed_at: None,
        }
    }

    // =========================================================================
    // CrossHappPayment — create
    // =========================================================================

    #[test]
    fn test_payment_create_valid() {
        let result = validate_create_cross_happ_payment(
            EntryCreationAction::Create(make_create()),
            valid_payment(),
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_payment_rejects_zero_amount() {
        let mut p = valid_payment();
        p.amount = 0;
        let result =
            validate_create_cross_happ_payment(EntryCreationAction::Create(make_create()), p)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_payment_rejects_non_sap_currency() {
        let mut p = valid_payment();
        p.currency = "USD".into();
        let result =
            validate_create_cross_happ_payment(EntryCreationAction::Create(make_create()), p)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_payment_rejects_oversized_from_did() {
        let mut p = valid_payment();
        p.from_did = "d".repeat(MAX_DID_LEN + 1);
        let result =
            validate_create_cross_happ_payment(EntryCreationAction::Create(make_create()), p)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_payment_rejects_oversized_to_did() {
        let mut p = valid_payment();
        p.to_did = "d".repeat(MAX_DID_LEN + 1);
        let result =
            validate_create_cross_happ_payment(EntryCreationAction::Create(make_create()), p)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_payment_rejects_oversized_source_happ() {
        let mut p = valid_payment();
        p.source_happ = "h".repeat(MAX_HAPP_ID_LEN + 1);
        let result =
            validate_create_cross_happ_payment(EntryCreationAction::Create(make_create()), p)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_payment_rejects_oversized_reference() {
        let mut p = valid_payment();
        p.reference = "r".repeat(MAX_REFERENCE_LEN + 1);
        let result =
            validate_create_cross_happ_payment(EntryCreationAction::Create(make_create()), p)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_payment_rejects_oversized_id() {
        let mut p = valid_payment();
        p.id = "i".repeat(MAX_REFERENCE_LEN + 1);
        let result =
            validate_create_cross_happ_payment(EntryCreationAction::Create(make_create()), p)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // =========================================================================
    // CrossHappPayment — update (permissive, always Valid)
    // =========================================================================

    #[test]
    fn test_payment_update_valid() {
        let result = validate_update_cross_happ_payment(make_update(), valid_payment()).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    // =========================================================================
    // CollateralRegistration — create
    // =========================================================================

    #[test]
    fn test_collateral_create_valid() {
        let result = validate_create_collateral_registration(
            EntryCreationAction::Create(make_create()),
            valid_collateral(),
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_collateral_rejects_missing_did_prefix() {
        let mut c = valid_collateral();
        c.owner_did = "alice".into();
        let result =
            validate_create_collateral_registration(EntryCreationAction::Create(make_create()), c)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_collateral_rejects_zero_value_estimate() {
        let mut c = valid_collateral();
        c.value_estimate = 0;
        let result =
            validate_create_collateral_registration(EntryCreationAction::Create(make_create()), c)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_collateral_rejects_empty_asset_id() {
        let mut c = valid_collateral();
        c.asset_id = "".into();
        let result =
            validate_create_collateral_registration(EntryCreationAction::Create(make_create()), c)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_collateral_rejects_empty_source_happ() {
        let mut c = valid_collateral();
        c.source_happ = "".into();
        let result =
            validate_create_collateral_registration(EntryCreationAction::Create(make_create()), c)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_collateral_rejects_empty_currency() {
        let mut c = valid_collateral();
        c.currency = "".into();
        let result =
            validate_create_collateral_registration(EntryCreationAction::Create(make_create()), c)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_collateral_rejects_oversized_owner_did() {
        let mut c = valid_collateral();
        c.owner_did = "d".repeat(MAX_DID_LEN + 1);
        let result =
            validate_create_collateral_registration(EntryCreationAction::Create(make_create()), c)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_collateral_rejects_oversized_asset_id() {
        let mut c = valid_collateral();
        c.asset_id = "a".repeat(MAX_REFERENCE_LEN + 1);
        let result =
            validate_create_collateral_registration(EntryCreationAction::Create(make_create()), c)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_collateral_rejects_oversized_id() {
        let mut c = valid_collateral();
        c.id = "i".repeat(MAX_REFERENCE_LEN + 1);
        let result =
            validate_create_collateral_registration(EntryCreationAction::Create(make_create()), c)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_collateral_rejects_oversized_source_happ() {
        let mut c = valid_collateral();
        c.source_happ = "h".repeat(MAX_HAPP_ID_LEN + 1);
        let result =
            validate_create_collateral_registration(EntryCreationAction::Create(make_create()), c)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // =========================================================================
    // CollateralRegistration — update
    // =========================================================================

    #[test]
    fn test_collateral_update_valid() {
        let result =
            validate_update_collateral_registration(make_update(), valid_collateral()).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_collateral_update_rejects_invalid_did() {
        let mut c = valid_collateral();
        c.owner_did = "no-prefix".into();
        let result = validate_update_collateral_registration(make_update(), c).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_collateral_update_rejects_zero_value() {
        let mut c = valid_collateral();
        c.value_estimate = 0;
        let result = validate_update_collateral_registration(make_update(), c).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // =========================================================================
    // FinanceBridgeEvent — create
    // =========================================================================

    #[test]
    fn test_event_create_valid() {
        let result = validate_create_finance_bridge_event(
            EntryCreationAction::Create(make_create()),
            valid_event(),
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_event_rejects_empty_source_happ() {
        let mut e = valid_event();
        e.source_happ = "".into();
        let result =
            validate_create_finance_bridge_event(EntryCreationAction::Create(make_create()), e)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_event_rejects_oversized_payload() {
        let mut e = valid_event();
        e.payload = "x".repeat(MAX_PAYLOAD_LEN + 1);
        let result =
            validate_create_finance_bridge_event(EntryCreationAction::Create(make_create()), e)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_event_rejects_oversized_subject_did() {
        let mut e = valid_event();
        e.subject_did = "d".repeat(MAX_DID_LEN + 1);
        let result =
            validate_create_finance_bridge_event(EntryCreationAction::Create(make_create()), e)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_event_rejects_oversized_source_happ() {
        let mut e = valid_event();
        e.source_happ = "h".repeat(MAX_HAPP_ID_LEN + 1);
        let result =
            validate_create_finance_bridge_event(EntryCreationAction::Create(make_create()), e)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_event_rejects_oversized_id() {
        let mut e = valid_event();
        e.id = "i".repeat(MAX_REFERENCE_LEN + 1);
        let result =
            validate_create_finance_bridge_event(EntryCreationAction::Create(make_create()), e)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // =========================================================================
    // FinanceBridgeEvent — update (immutable, always Invalid)
    // =========================================================================

    // Note: update rejection is handled inline in validate() match arm,
    // not via a standalone function. Tested via the match arm returning Invalid.

    // =========================================================================
    // CollateralBridgeDeposit — create
    // =========================================================================

    #[test]
    fn test_deposit_create_valid() {
        let result = validate_create_collateral_bridge_deposit(
            EntryCreationAction::Create(make_create()),
            valid_deposit(),
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_deposit_create_valid_usdc() {
        let mut d = valid_deposit();
        d.collateral_type = "USDC".into();
        d.oracle_rate = 1.0;
        d.sap_minted = 1000;
        let result = validate_create_collateral_bridge_deposit(
            EntryCreationAction::Create(make_create()),
            d,
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_deposit_rejects_missing_did_prefix() {
        let mut d = valid_deposit();
        d.depositor_did = "alice".into();
        let result = validate_create_collateral_bridge_deposit(
            EntryCreationAction::Create(make_create()),
            d,
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_deposit_rejects_zero_collateral_amount() {
        let mut d = valid_deposit();
        d.collateral_amount = 0;
        let result = validate_create_collateral_bridge_deposit(
            EntryCreationAction::Create(make_create()),
            d,
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_deposit_rejects_zero_sap_minted() {
        let mut d = valid_deposit();
        d.sap_minted = 0;
        let result = validate_create_collateral_bridge_deposit(
            EntryCreationAction::Create(make_create()),
            d,
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_deposit_rejects_nan_oracle_rate() {
        let mut d = valid_deposit();
        d.oracle_rate = f64::NAN;
        let result = validate_create_collateral_bridge_deposit(
            EntryCreationAction::Create(make_create()),
            d,
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_deposit_rejects_infinity_oracle_rate() {
        let mut d = valid_deposit();
        d.oracle_rate = f64::INFINITY;
        let result = validate_create_collateral_bridge_deposit(
            EntryCreationAction::Create(make_create()),
            d,
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_deposit_rejects_neg_infinity_oracle_rate() {
        let mut d = valid_deposit();
        d.oracle_rate = f64::NEG_INFINITY;
        let result = validate_create_collateral_bridge_deposit(
            EntryCreationAction::Create(make_create()),
            d,
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_deposit_rejects_zero_oracle_rate() {
        let mut d = valid_deposit();
        d.oracle_rate = 0.0;
        let result = validate_create_collateral_bridge_deposit(
            EntryCreationAction::Create(make_create()),
            d,
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_deposit_rejects_negative_oracle_rate() {
        let mut d = valid_deposit();
        d.oracle_rate = -1.0;
        let result = validate_create_collateral_bridge_deposit(
            EntryCreationAction::Create(make_create()),
            d,
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_deposit_rejects_unsupported_collateral_type() {
        let mut d = valid_deposit();
        d.collateral_type = "BTC".into();
        let result = validate_create_collateral_bridge_deposit(
            EntryCreationAction::Create(make_create()),
            d,
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_deposit_rejects_inconsistent_sap_minted() {
        let mut d = valid_deposit();
        // oracle_rate=2.0, collateral=1000, expected sap=2000, but set to 9999
        d.sap_minted = 9999;
        let result = validate_create_collateral_bridge_deposit(
            EntryCreationAction::Create(make_create()),
            d,
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_deposit_allows_rounding_tolerance() {
        // expected = (1000 * 2.0) as u64 = 2000, allow +-1
        let mut d = valid_deposit();
        d.sap_minted = 2001; // within tolerance
        let result = validate_create_collateral_bridge_deposit(
            EntryCreationAction::Create(make_create()),
            d,
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));

        let mut d2 = valid_deposit();
        d2.sap_minted = 1999; // within tolerance
        let result2 = validate_create_collateral_bridge_deposit(
            EntryCreationAction::Create(make_create()),
            d2,
        )
        .unwrap();
        assert!(matches!(result2, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_deposit_rejects_non_pending_status() {
        let mut d = valid_deposit();
        d.status = BridgeDepositStatus::Confirmed;
        let result = validate_create_collateral_bridge_deposit(
            EntryCreationAction::Create(make_create()),
            d,
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_deposit_rejects_oversized_did() {
        let mut d = valid_deposit();
        d.depositor_did = "d".repeat(MAX_DID_LEN + 1);
        let result = validate_create_collateral_bridge_deposit(
            EntryCreationAction::Create(make_create()),
            d,
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_deposit_rejects_oversized_collateral_type() {
        let mut d = valid_deposit();
        d.collateral_type = "x".repeat(MAX_HAPP_ID_LEN + 1);
        let result = validate_create_collateral_bridge_deposit(
            EntryCreationAction::Create(make_create()),
            d,
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_deposit_rejects_oversized_id() {
        let mut d = valid_deposit();
        d.id = "i".repeat(MAX_REFERENCE_LEN + 1);
        let result = validate_create_collateral_bridge_deposit(
            EntryCreationAction::Create(make_create()),
            d,
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // =========================================================================
    // CollateralBridgeDeposit — update
    // =========================================================================

    #[test]
    fn test_deposit_update_valid_confirmed() {
        let mut d = valid_deposit();
        d.status = BridgeDepositStatus::Confirmed;
        let result = validate_update_collateral_bridge_deposit(make_update(), d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_deposit_update_valid_redeemed() {
        let mut d = valid_deposit();
        d.status = BridgeDepositStatus::Redeemed;
        let result = validate_update_collateral_bridge_deposit(make_update(), d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_deposit_update_valid_failed() {
        let mut d = valid_deposit();
        d.status = BridgeDepositStatus::Failed;
        let result = validate_update_collateral_bridge_deposit(make_update(), d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_deposit_update_rejects_pending_status() {
        let mut d = valid_deposit();
        d.status = BridgeDepositStatus::Pending;
        let result = validate_update_collateral_bridge_deposit(make_update(), d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_deposit_update_rejects_invalid_did() {
        let mut d = valid_deposit();
        d.status = BridgeDepositStatus::Confirmed;
        d.depositor_did = "no-prefix".into();
        let result = validate_update_collateral_bridge_deposit(make_update(), d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_deposit_update_rejects_zero_collateral() {
        let mut d = valid_deposit();
        d.status = BridgeDepositStatus::Confirmed;
        d.collateral_amount = 0;
        let result = validate_update_collateral_bridge_deposit(make_update(), d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_deposit_update_rejects_zero_sap_minted() {
        let mut d = valid_deposit();
        d.status = BridgeDepositStatus::Confirmed;
        d.sap_minted = 0;
        let result = validate_update_collateral_bridge_deposit(make_update(), d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_deposit_update_rejects_nan_oracle_rate() {
        let mut d = valid_deposit();
        d.status = BridgeDepositStatus::Confirmed;
        d.oracle_rate = f64::NAN;
        let result = validate_update_collateral_bridge_deposit(make_update(), d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_deposit_update_rejects_unsupported_collateral_type() {
        let mut d = valid_deposit();
        d.status = BridgeDepositStatus::Confirmed;
        d.collateral_type = "DOGE".into();
        let result = validate_update_collateral_bridge_deposit(make_update(), d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // =========================================================================
    // Covenant — create
    // =========================================================================

    fn valid_covenant() -> Covenant {
        Covenant {
            id: "cov:test:001".into(),
            collateral_id: "col:test:001".into(),
            restriction: r#"{"type":"NoTransfer"}"#.into(),
            beneficiary_did: "did:mycelix:alice".into(),
            created_at: ts(1_000_000),
            expires_at: None,
            released: false,
            released_by: None,
            released_at: None,
        }
    }

    #[test]
    fn test_covenant_create_valid() {
        let result =
            validate_create_covenant(EntryCreationAction::Create(make_create()), valid_covenant())
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_covenant_rejects_empty_id() {
        let mut c = valid_covenant();
        c.id = "".into();
        let result =
            validate_create_covenant(EntryCreationAction::Create(make_create()), c).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_covenant_rejects_empty_collateral_id() {
        let mut c = valid_covenant();
        c.collateral_id = "".into();
        let result =
            validate_create_covenant(EntryCreationAction::Create(make_create()), c).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_covenant_rejects_invalid_beneficiary_did() {
        let mut c = valid_covenant();
        c.beneficiary_did = "alice".into();
        let result =
            validate_create_covenant(EntryCreationAction::Create(make_create()), c).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_covenant_rejects_pre_released() {
        let mut c = valid_covenant();
        c.released = true;
        let result =
            validate_create_covenant(EntryCreationAction::Create(make_create()), c).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_covenant_rejects_empty_restriction() {
        let mut c = valid_covenant();
        c.restriction = "".into();
        let result =
            validate_create_covenant(EntryCreationAction::Create(make_create()), c).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_covenant_update_valid() {
        let mut c = valid_covenant();
        c.released = true;
        let result = validate_update_covenant(make_update(), c).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_covenant_update_rejects_invalid_did() {
        let mut c = valid_covenant();
        c.beneficiary_did = "no-prefix".into();
        let result = validate_update_covenant(make_update(), c).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // =========================================================================
    // CollateralHealth — create
    // =========================================================================

    fn valid_health() -> CollateralHealth {
        CollateralHealth {
            collateral_id: "col:test:001".into(),
            current_value: 100_000,
            obligation_amount: 50_000,
            ltv_ratio: 0.5,
            status: "Healthy".into(),
            computed_at: ts(1_000_000),
        }
    }

    #[test]
    fn test_health_create_valid() {
        let result = validate_create_collateral_health(
            EntryCreationAction::Create(make_create()),
            valid_health(),
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_health_rejects_empty_collateral_id() {
        let mut h = valid_health();
        h.collateral_id = "".into();
        let result =
            validate_create_collateral_health(EntryCreationAction::Create(make_create()), h)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_health_rejects_nan_ltv() {
        let mut h = valid_health();
        h.ltv_ratio = f64::NAN;
        let result =
            validate_create_collateral_health(EntryCreationAction::Create(make_create()), h)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_health_rejects_negative_ltv() {
        let mut h = valid_health();
        h.ltv_ratio = -0.1;
        let result =
            validate_create_collateral_health(EntryCreationAction::Create(make_create()), h)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_health_rejects_invalid_status() {
        let mut h = valid_health();
        h.status = "Critical".into();
        let result =
            validate_create_collateral_health(EntryCreationAction::Create(make_create()), h)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // =========================================================================
    // FiatBridgeDeposit — create
    // =========================================================================

    fn valid_fiat_deposit() -> FiatBridgeDeposit {
        FiatBridgeDeposit {
            id: "fiat:test:001".into(),
            depositor_did: "did:mycelix:alice".into(),
            fiat_currency: "USD".into(),
            fiat_amount: 10_000,
            sap_minted: 10_000,
            exchange_rate: 1.0,
            verifier_did: "did:mycelix:verifier".into(),
            status: "Pending".into(),
            external_reference: "wire-ref-123".into(),
            created_at: ts(1_000_000),
            verified_at: None,
        }
    }

    #[test]
    fn test_fiat_deposit_create_valid() {
        let result = validate_create_fiat_bridge_deposit(
            EntryCreationAction::Create(make_create()),
            valid_fiat_deposit(),
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_fiat_deposit_rejects_zero_amount() {
        let mut f = valid_fiat_deposit();
        f.fiat_amount = 0;
        let result =
            validate_create_fiat_bridge_deposit(EntryCreationAction::Create(make_create()), f)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_fiat_deposit_rejects_invalid_depositor_did() {
        let mut f = valid_fiat_deposit();
        f.depositor_did = "alice".into();
        let result =
            validate_create_fiat_bridge_deposit(EntryCreationAction::Create(make_create()), f)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_fiat_deposit_rejects_invalid_verifier_did() {
        let mut f = valid_fiat_deposit();
        f.verifier_did = "verifier".into();
        let result =
            validate_create_fiat_bridge_deposit(EntryCreationAction::Create(make_create()), f)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_fiat_deposit_rejects_non_pending_status() {
        let mut f = valid_fiat_deposit();
        f.status = "Verified".into();
        let result =
            validate_create_fiat_bridge_deposit(EntryCreationAction::Create(make_create()), f)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_fiat_deposit_rejects_invalid_exchange_rate() {
        let mut f = valid_fiat_deposit();
        f.exchange_rate = -1.0;
        let result =
            validate_create_fiat_bridge_deposit(EntryCreationAction::Create(make_create()), f)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_fiat_deposit_update_rejects_pending_revert() {
        let mut f = valid_fiat_deposit();
        f.status = "Pending".into();
        let result = validate_update_fiat_bridge_deposit(make_update(), f).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // =========================================================================
    // EnergyCertificate — create
    // =========================================================================

    fn valid_energy_cert() -> EnergyCertificate {
        EnergyCertificate {
            id: "cert:test:001".into(),
            project_id: "proj:solar:001".into(),
            source: "Solar".into(),
            kwh_produced: 1500.0,
            period_start: ts(1_000_000),
            period_end: ts(2_000_000),
            location_lat: 32.95,
            location_lon: -96.73,
            producer_did: "did:mycelix:producer".into(),
            verifier_did: None,
            status: "Pending".into(),
            terra_atlas_id: None,
            sap_value: None,
            created_at: ts(1_000_000),
        }
    }

    #[test]
    fn test_energy_cert_create_valid() {
        let result = validate_create_energy_certificate(
            EntryCreationAction::Create(make_create()),
            valid_energy_cert(),
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_energy_cert_rejects_zero_kwh() {
        let mut c = valid_energy_cert();
        c.kwh_produced = 0.0;
        let result =
            validate_create_energy_certificate(EntryCreationAction::Create(make_create()), c)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_energy_cert_rejects_invalid_lat() {
        let mut c = valid_energy_cert();
        c.location_lat = 91.0;
        let result =
            validate_create_energy_certificate(EntryCreationAction::Create(make_create()), c)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_energy_cert_rejects_invalid_lon() {
        let mut c = valid_energy_cert();
        c.location_lon = -181.0;
        let result =
            validate_create_energy_certificate(EntryCreationAction::Create(make_create()), c)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_energy_cert_rejects_invalid_producer_did() {
        let mut c = valid_energy_cert();
        c.producer_did = "producer".into();
        let result =
            validate_create_energy_certificate(EntryCreationAction::Create(make_create()), c)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_energy_cert_rejects_non_pending_status() {
        let mut c = valid_energy_cert();
        c.status = "Verified".into();
        let result =
            validate_create_energy_certificate(EntryCreationAction::Create(make_create()), c)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_energy_cert_update_rejects_pending_revert() {
        let mut c = valid_energy_cert();
        c.status = "Pending".into();
        let result = validate_update_energy_certificate(make_update(), c).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // =========================================================================
    // AgriculturalAsset — create
    // =========================================================================

    fn valid_agri_asset() -> AgriculturalAsset {
        AgriculturalAsset {
            id: "agri:test:001".into(),
            asset_type: "Grain".into(),
            quantity_kg: 5000.0,
            location_lat: 33.0,
            location_lon: -97.0,
            production_date: ts(1_000_000),
            viability_duration_micros: Some(90 * DAY_MICROS_CONST),
            producer_did: "did:mycelix:farmer".into(),
            verifier_did: None,
            status: "Pending".into(),
            sap_value: None,
            created_at: ts(1_000_000),
        }
    }

    /// 24 hours in microseconds (test constant)
    const DAY_MICROS_CONST: i64 = 24 * 60 * 60 * 1_000_000;

    #[test]
    fn test_agri_asset_create_valid() {
        let result = validate_create_agricultural_asset(
            EntryCreationAction::Create(make_create()),
            valid_agri_asset(),
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_agri_asset_rejects_zero_quantity() {
        let mut a = valid_agri_asset();
        a.quantity_kg = 0.0;
        let result =
            validate_create_agricultural_asset(EntryCreationAction::Create(make_create()), a)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_agri_asset_rejects_invalid_producer_did() {
        let mut a = valid_agri_asset();
        a.producer_did = "farmer".into();
        let result =
            validate_create_agricultural_asset(EntryCreationAction::Create(make_create()), a)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_agri_asset_rejects_non_pending_status() {
        let mut a = valid_agri_asset();
        a.status = "Verified".into();
        let result =
            validate_create_agricultural_asset(EntryCreationAction::Create(make_create()), a)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_agri_asset_update_rejects_pending_revert() {
        let mut a = valid_agri_asset();
        a.status = "Pending".into();
        let result = validate_update_agricultural_asset(make_update(), a).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // =========================================================================
    // MultiCollateralPosition — create
    // =========================================================================

    fn valid_multi_position() -> MultiCollateralPosition {
        MultiCollateralPosition {
            id: "mcp:test:001".into(),
            holder_did: "did:mycelix:alice".into(),
            components_json: r#"[{"type":"ETH","amount":1000}]"#.into(),
            aggregate_value: 100_000,
            aggregate_obligation: 50_000,
            diversification_bonus: 0.05,
            effective_ltv: 0.5,
            status: "Healthy".into(),
            created_at: ts(1_000_000),
            last_revalued_at: ts(1_000_000),
        }
    }

    #[test]
    fn test_multi_position_create_valid() {
        let result = validate_create_multi_collateral_position(
            EntryCreationAction::Create(make_create()),
            valid_multi_position(),
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_multi_position_rejects_empty_id() {
        let mut p = valid_multi_position();
        p.id = "".into();
        let result = validate_create_multi_collateral_position(
            EntryCreationAction::Create(make_create()),
            p,
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_multi_position_rejects_invalid_holder_did() {
        let mut p = valid_multi_position();
        p.holder_did = "alice".into();
        let result = validate_create_multi_collateral_position(
            EntryCreationAction::Create(make_create()),
            p,
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_multi_position_rejects_empty_components() {
        let mut p = valid_multi_position();
        p.components_json = "".into();
        let result = validate_create_multi_collateral_position(
            EntryCreationAction::Create(make_create()),
            p,
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_multi_position_rejects_invalid_diversification_bonus() {
        let mut p = valid_multi_position();
        p.diversification_bonus = 1.5;
        let result = validate_create_multi_collateral_position(
            EntryCreationAction::Create(make_create()),
            p,
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_multi_position_rejects_negative_ltv() {
        let mut p = valid_multi_position();
        p.effective_ltv = -0.1;
        let result = validate_create_multi_collateral_position(
            EntryCreationAction::Create(make_create()),
            p,
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_multi_position_update_valid() {
        let result =
            validate_update_multi_collateral_position(make_update(), valid_multi_position())
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_multi_position_update_rejects_invalid_did() {
        let mut p = valid_multi_position();
        p.holder_did = "no-prefix".into();
        let result = validate_update_multi_collateral_position(make_update(), p).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }
}
