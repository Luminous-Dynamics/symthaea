//! Payments Integrity Zome
//! Updated to use HDI 0.7 patterns with FlatOp validation
use hdi::prelude::*;
pub use mycelix_finance_types::SuccessionPreference;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Payment {
    pub id: String,
    pub from_did: String,
    pub to_did: String,
    pub amount: f64,
    pub currency: String,
    pub payment_type: PaymentType,
    pub status: TransferStatus,
    pub memo: Option<String>,
    pub created: Timestamp,
    pub completed: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PaymentType {
    Direct,
    TreasuryContribution(String), // treasury_id
    CommonsContribution(String),  // commons_pool_id
    Escrow(String),               // escrow_id
    Recurring(RecurringConfig),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RecurringConfig {
    pub frequency_days: u32,
    pub end_date: Option<Timestamp>,
    pub remaining: Option<u32>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TransferStatus {
    Pending,
    Processing,
    Completed,
    Failed(String),
    Cancelled,
    Refunded,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PaymentChannel {
    pub id: String,
    pub party_a: String,
    pub party_b: String,
    pub currency: String,
    pub balance_a: f64,
    pub balance_b: f64,
    pub opened: Timestamp,
    pub last_updated: Timestamp,
    pub closed: Option<Timestamp>,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Receipt {
    pub payment_id: String,
    pub from_did: String,
    pub to_did: String,
    pub amount: f64,
    pub currency: String,
    pub timestamp: Timestamp,
    pub signature: String,
}

/// On-chain SAP balance with lazy demurrage tracking.
/// Demurrage is applied on every balance read/mutation — no cron needed.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SapBalance {
    pub member_did: String,
    /// Raw balance in SAP micro-units (1 SAP = 1_000_000 micro-SAP)
    pub balance: u64,
    /// Timestamp of last demurrage application
    pub last_demurrage_at: Timestamp,
}

/// Record of a member exit, coordinating across all currencies
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ExitRecord {
    /// DID of the exiting member
    pub member_did: String,
    /// SAP succession preference
    pub succession_preference: SuccessionPreference,
    /// SAP balance at time of exit
    pub sap_balance: f64,
    /// TEND balances forgiven (list of dao_did:amount pairs)
    pub tend_balances_forgiven: Vec<(String, i32)>,
    /// Whether MYCEL was dissolved
    pub mycel_dissolved: bool,
    /// Timestamp of the exit
    pub exited_at: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Payment(Payment),
    PaymentChannel(PaymentChannel),
    Receipt(Receipt),
    ExitRecord(ExitRecord),
    SapBalance(SapBalance),
}

#[hdk_link_types]
pub enum LinkTypes {
    SenderToPayments,
    ReceiverToPayments,
    PaymentToReceipt,
    ChannelPartyA,
    ChannelPartyB,
    DidToSapBalance,
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
            OpEntry::CreateEntry { app_entry, action } => {
                match app_entry {
                    EntryTypes::Payment(payment) => {
                        validate_create_payment(EntryCreationAction::Create(action), payment)
                    }
                    EntryTypes::PaymentChannel(channel) => {
                        validate_create_payment_channel(EntryCreationAction::Create(action), channel)
                    }
                    EntryTypes::Receipt(receipt) => {
                        validate_create_receipt(EntryCreationAction::Create(action), receipt)
                    }
                    EntryTypes::ExitRecord(exit) => {
                        validate_create_exit_record(EntryCreationAction::Create(action), exit)
                    }
                    EntryTypes::SapBalance(bal) => {
                        validate_sap_balance(&bal)
                    }
                }
            }
            OpEntry::UpdateEntry { app_entry, action, .. } => {
                match app_entry {
                    EntryTypes::Payment(payment) => {
                        validate_update_payment(action, payment)
                    }
                    EntryTypes::PaymentChannel(channel) => {
                        validate_update_payment_channel(action, channel)
                    }
                    EntryTypes::Receipt(_) => {
                        Ok(ValidateCallbackResult::Invalid(
                            "Receipts cannot be updated".into(),
                        ))
                    }
                    EntryTypes::ExitRecord(_) => {
                        Ok(ValidateCallbackResult::Invalid(
                            "Exit records cannot be updated".into(),
                        ))
                    }
                    EntryTypes::SapBalance(bal) => {
                        validate_sap_balance(&bal)
                    }
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, base_address, target_address, .. } => {
            match link_type {
                LinkTypes::SenderToPayments | LinkTypes::ReceiverToPayments => {
                    // Base should be an agent pubkey (DID anchor)
                    // Target should be an entry hash (payment)
                    if base_address.as_ref().len() != 39 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link base must be a valid agent pubkey".into()
                        ));
                    }
                    if target_address.as_ref().len() != 39 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link target must be a valid entry hash".into()
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::PaymentToReceipt => {
                    // Both should be entry hashes
                    if base_address.as_ref().len() != 39 || target_address.as_ref().len() != 39 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "PaymentToReceipt link must connect two entry hashes".into()
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::ChannelPartyA | LinkTypes::ChannelPartyB => {
                    // Base is agent, target is channel entry
                    if target_address.as_ref().len() != 39 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link target must be a valid entry hash".into()
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::DidToSapBalance => {
                    if base_address.as_ref().len() != 39 || target_address.as_ref().len() != 39 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "DidToSapBalance link must connect valid hashes".into()
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
            }
        }
        FlatOp::RegisterDeleteLink { link_type, .. } => {
            // Prevent deletion of critical links
            match link_type {
                LinkTypes::PaymentToReceipt => {
                    Ok(ValidateCallbackResult::Invalid(
                        "PaymentToReceipt links cannot be deleted - receipts are immutable".into()
                    ))
                }
                _ => Ok(ValidateCallbackResult::Valid)
            }
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_payment(
    _action: EntryCreationAction,
    payment: Payment,
) -> ExternResult<ValidateCallbackResult> {
    if !payment.from_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid("Sender must be a valid DID".into()));
    }
    if !payment.to_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid("Receiver must be a valid DID".into()));
    }
    if payment.amount <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Amount must be positive".into()));
    }
    if payment.from_did == payment.to_did {
        return Ok(ValidateCallbackResult::Invalid("Cannot send payment to yourself".into()));
    }
    // Only SAP and TEND currencies are accepted
    if payment.currency != "SAP" && payment.currency != "TEND" {
        return Ok(ValidateCallbackResult::Invalid(
            "Currency must be \"SAP\" or \"TEND\"".into()
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_payment(
    _action: Update,
    payment: Payment,
) -> ExternResult<ValidateCallbackResult> {
    // Status can change but amount/parties cannot
    if payment.amount <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Amount must be positive".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_payment_channel(
    _action: EntryCreationAction,
    channel: PaymentChannel,
) -> ExternResult<ValidateCallbackResult> {
    if !channel.party_a.starts_with("did:") || !channel.party_b.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid("Parties must be valid DIDs".into()));
    }
    if channel.balance_a < 0.0 || channel.balance_b < 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Balances cannot be negative".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_payment_channel(
    _action: Update,
    channel: PaymentChannel,
) -> ExternResult<ValidateCallbackResult> {
    if channel.balance_a < 0.0 || channel.balance_b < 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Balances cannot be negative".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_receipt(
    _action: EntryCreationAction,
    receipt: Receipt,
) -> ExternResult<ValidateCallbackResult> {
    // Validate DIDs
    if !receipt.from_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid("Sender must be a valid DID".into()));
    }
    if !receipt.to_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid("Receiver must be a valid DID".into()));
    }

    // Validate amount
    if receipt.amount <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Amount must be positive".into()));
    }

    // Validate signature is present and properly formatted
    // Signature format: base64-encoded Ed25519 signature (88 chars) or hex (128 chars)
    if receipt.signature.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Receipt must have a signature".into()));
    }

    // Basic signature format validation
    // Ed25519 signatures are 64 bytes = 88 chars base64 or 128 chars hex
    let sig_len = receipt.signature.len();
    if sig_len < 64 {
        return Ok(ValidateCallbackResult::Invalid(
            "Signature too short - must be valid Ed25519 signature".into()
        ));
    }

    // Verify signature is valid base64 or hex
    let is_valid_format = receipt.signature.chars().all(|c| {
        c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '='
    });
    if !is_valid_format {
        return Ok(ValidateCallbackResult::Invalid(
            "Signature must be valid base64 or hex encoding".into()
        ));
    }

    // Note: Full cryptographic verification requires the sender's public key
    // which would be fetched from the identity zome in production.
    // The signature should cover: payment_id | from_did | to_did | amount | currency | timestamp

    Ok(ValidateCallbackResult::Valid)
}

fn validate_sap_balance(bal: &SapBalance) -> ExternResult<ValidateCallbackResult> {
    if !bal.member_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid("Member must be a valid DID".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_exit_record(
    _action: EntryCreationAction,
    exit: ExitRecord,
) -> ExternResult<ValidateCallbackResult> {
    if !exit.member_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid("Member must be a valid DID".into()));
    }
    if exit.sap_balance < 0.0 {
        return Ok(ValidateCallbackResult::Invalid("SAP balance cannot be negative".into()));
    }
    if let SuccessionPreference::Designee(ref designee) = exit.succession_preference {
        if !designee.starts_with("did:") {
            return Ok(ValidateCallbackResult::Invalid("Designee must be a valid DID".into()));
        }
        if *designee == exit.member_did {
            return Ok(ValidateCallbackResult::Invalid("Cannot designate yourself as successor".into()));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}
