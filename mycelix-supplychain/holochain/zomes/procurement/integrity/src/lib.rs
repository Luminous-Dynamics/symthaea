// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Procurement Integrity Zome
//!
//! Entry types and validation for supply chain procurement.
//! Handles purchase orders, suppliers, and RFQs.
//! Holochain 0.6 compatible (hdi 0.7)

use hdi::prelude::*;

/// Purchase Order status
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PurchaseOrderStatus {
    Draft,
    Submitted,
    Approved,
    Rejected,
    Sent,
    Acknowledged,
    PartiallyReceived,
    Received,
    Cancelled,
    Closed,
}

/// Purchase Order entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PurchaseOrder {
    pub po_number: String,
    pub supplier: AgentPubKey,
    pub buyer: AgentPubKey,
    pub items: Vec<PurchaseOrderItem>,
    pub total_amount: u64,
    pub currency: String,
    pub status: PurchaseOrderStatus,
    pub created_at: Timestamp,
    pub due_date: Option<Timestamp>,
    pub notes: Option<String>,
    pub trust_score_at_creation: Option<u32>,
}

/// Purchase Order line item
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PurchaseOrderItem {
    pub item_code: String,
    pub description: String,
    pub quantity: u64,
    pub unit_price: u64,
    pub unit: String,
}

/// Supplier profile entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SupplierProfile {
    pub agent: AgentPubKey,
    pub company_name: String,
    pub contact_email: String,
    pub contact_phone: Option<String>,
    pub address: Option<String>,
    pub categories: Vec<String>,
    pub certifications: Vec<String>,
    pub payment_terms: String,
    pub lead_time_days: u32,
    pub minimum_order_value: u64,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
}

/// Request for Quotation
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RequestForQuotation {
    pub rfq_number: String,
    pub buyer: AgentPubKey,
    pub items: Vec<RfqItem>,
    pub deadline: Timestamp,
    pub notes: Option<String>,
    pub status: RfqStatus,
    pub created_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RfqItem {
    pub description: String,
    pub quantity: u64,
    pub unit: String,
    pub specifications: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RfqStatus {
    Open,
    Closed,
    Awarded,
    Cancelled,
}

/// Quotation response to RFQ
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Quotation {
    pub rfq_hash: ActionHash,
    pub supplier: AgentPubKey,
    pub items: Vec<QuotationItem>,
    pub total_amount: u64,
    pub currency: String,
    pub valid_until: Timestamp,
    pub lead_time_days: u32,
    pub notes: Option<String>,
    pub submitted_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct QuotationItem {
    pub description: String,
    pub quantity: u64,
    pub unit_price: u64,
    pub unit: String,
}

/// Entry types for this zome
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    PurchaseOrder(PurchaseOrder),
    SupplierProfile(SupplierProfile),
    RequestForQuotation(RequestForQuotation),
    Quotation(Quotation),
}

/// Link types for this zome
#[hdk_link_types]
pub enum LinkTypes {
    BuyerToPurchaseOrders,
    SupplierToPurchaseOrders,
    AgentToSupplierProfile,
    BuyerToRfqs,
    RfqToQuotations,
    SupplierToQuotations,
    AllSuppliers,
    CategoryToSuppliers,
}

/// Validate entries and links
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(OpEntry::CreateEntry { app_entry, .. }) => {
            match app_entry {
                EntryTypes::PurchaseOrder(po) => validate_purchase_order(&po),
                EntryTypes::SupplierProfile(profile) => validate_supplier_profile(&profile),
                EntryTypes::RequestForQuotation(rfq) => validate_rfq(&rfq),
                EntryTypes::Quotation(quote) => validate_quotation(&quote),
            }
        }
        FlatOp::StoreEntry(OpEntry::UpdateEntry { app_entry, .. }) => {
            match app_entry {
                EntryTypes::PurchaseOrder(po) => validate_purchase_order(&po),
                EntryTypes::SupplierProfile(profile) => validate_supplier_profile(&profile),
                EntryTypes::RequestForQuotation(rfq) => validate_rfq(&rfq),
                EntryTypes::Quotation(quote) => validate_quotation(&quote),
            }
        }
        FlatOp::RegisterCreateLink { link_type, .. } => {
            match link_type {
                LinkTypes::BuyerToPurchaseOrders => Ok(ValidateCallbackResult::Valid),
                LinkTypes::SupplierToPurchaseOrders => Ok(ValidateCallbackResult::Valid),
                LinkTypes::AgentToSupplierProfile => Ok(ValidateCallbackResult::Valid),
                LinkTypes::BuyerToRfqs => Ok(ValidateCallbackResult::Valid),
                LinkTypes::RfqToQuotations => Ok(ValidateCallbackResult::Valid),
                LinkTypes::SupplierToQuotations => Ok(ValidateCallbackResult::Valid),
                LinkTypes::AllSuppliers => Ok(ValidateCallbackResult::Valid),
                LinkTypes::CategoryToSuppliers => Ok(ValidateCallbackResult::Valid),
            }
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_purchase_order(po: &PurchaseOrder) -> ExternResult<ValidateCallbackResult> {
    if po.po_number.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "PO number cannot be empty".to_string(),
        ));
    }
    if po.items.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "PO must have at least one item".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_supplier_profile(profile: &SupplierProfile) -> ExternResult<ValidateCallbackResult> {
    if profile.company_name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Company name cannot be empty".to_string(),
        ));
    }
    if profile.contact_email.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Contact email cannot be empty".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_rfq(rfq: &RequestForQuotation) -> ExternResult<ValidateCallbackResult> {
    if rfq.rfq_number.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "RFQ number cannot be empty".to_string(),
        ));
    }
    if rfq.items.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "RFQ must have at least one item".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_quotation(quote: &Quotation) -> ExternResult<ValidateCallbackResult> {
    if quote.items.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Quotation must have at least one item".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
