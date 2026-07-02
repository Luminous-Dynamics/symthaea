// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Payments Integrity Zome - Entry types for payment processing
use hdi::prelude::*;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PaymentStatus { Pending, Authorized, Captured, Completed, Failed, Refunded, Disputed }

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PaymentMethod { BankTransfer, CreditCard, Crypto, Escrow, LetterOfCredit }

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Payment {
    pub po_hash: ActionHash,
    pub amount: u64,
    pub currency: String,
    pub method: PaymentMethod,
    pub status: PaymentStatus,
    pub payer: AgentPubKey,
    pub payee: AgentPubKey,
    pub reference: String,
    pub created_at: Timestamp,
    pub completed_at: Option<Timestamp>,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Invoice {
    pub po_hash: ActionHash,
    pub invoice_number: String,
    pub amount: u64,
    pub currency: String,
    pub due_date: Timestamp,
    pub issued_by: AgentPubKey,
    pub issued_to: AgentPubKey,
    pub line_items: Vec<InvoiceItem>,
    pub created_at: Timestamp,
    pub paid_at: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct InvoiceItem {
    pub description: String,
    pub quantity: u64,
    pub unit_price: u64,
    pub total: u64,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EscrowAccount {
    pub po_hash: ActionHash,
    pub amount: u64,
    pub currency: String,
    pub buyer: AgentPubKey,
    pub seller: AgentPubKey,
    pub arbiter: Option<AgentPubKey>,
    pub release_conditions: Vec<String>,
    pub funded_at: Option<Timestamp>,
    pub released_at: Option<Timestamp>,
    pub created_at: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Payment(Payment),
    Invoice(Invoice),
    EscrowAccount(EscrowAccount),
}

#[hdk_link_types]
pub enum LinkTypes { PoToPayments, PoToInvoices, PoToEscrow, PayerToPayments, PayeeToPayments }

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(OpEntry::CreateEntry { app_entry, .. }) => {
            match app_entry {
                EntryTypes::Payment(p) => {
                    if p.amount == 0 { return Ok(ValidateCallbackResult::Invalid("Payment amount must be positive".into())); }
                    if p.currency.is_empty() { return Ok(ValidateCallbackResult::Invalid("Currency required".into())); }
                    Ok(ValidateCallbackResult::Valid)
                }
                EntryTypes::Invoice(i) => {
                    if i.invoice_number.is_empty() { return Ok(ValidateCallbackResult::Invalid("Invoice number required".into())); }
                    if i.amount == 0 { return Ok(ValidateCallbackResult::Invalid("Invoice amount must be positive".into())); }
                    Ok(ValidateCallbackResult::Valid)
                }
                _ => Ok(ValidateCallbackResult::Valid),
            }
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
