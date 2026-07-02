// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Financial data models
//!
//! Core data structures for general ledger, invoices, bills, and payments.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;

/// General Ledger Account
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct GlAccount {
    pub id: Uuid,
    pub account_number: String,
    pub account_name: String,
    pub account_type: AccountType,
    pub parent_account_id: Option<Uuid>,
    pub is_active: bool,
    pub currency: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Account types following standard accounting classification
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "account_type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AccountType {
    Asset,
    Liability,
    Equity,
    Revenue,
    Expense,
}

/// Journal Entry for double-entry bookkeeping
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct JournalEntry {
    pub id: Uuid,
    pub entry_number: String,
    pub entry_date: DateTime<Utc>,
    pub description: String,
    pub reference: Option<String>,
    pub status: EntryStatus,
    /// Hash of all line items for tamper detection
    pub lines_hash: String,
    /// Link to DKG claim for auditability
    pub claim_id: Option<String>,
    pub created_by: Uuid,
    pub created_at: DateTime<Utc>,
    pub posted_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "entry_status", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EntryStatus {
    Draft,
    Posted,
    Reversed,
    Voided,
}

/// Individual line in a journal entry
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct JournalLine {
    pub id: Uuid,
    pub entry_id: Uuid,
    pub line_number: i32,
    pub account_id: Uuid,
    pub debit_amount: Option<rust_decimal::Decimal>,
    pub credit_amount: Option<rust_decimal::Decimal>,
    pub description: Option<String>,
    pub dimension1: Option<String>, // For cost center, department, etc.
    pub dimension2: Option<String>,
    pub dimension3: Option<String>,
}

/// Journal entry with its lines (for API responses)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalEntryWithLines {
    #[serde(flatten)]
    pub entry: JournalEntry,
    pub lines: Vec<JournalLine>,
}

/// Request to create a journal entry
#[derive(Debug, Clone, Deserialize)]
pub struct CreateJournalEntryRequest {
    pub description: String,
    pub reference: Option<String>,
    pub lines: Vec<JournalLineRequest>,
}

/// Individual line in a journal entry request
#[derive(Debug, Clone, Deserialize)]
pub struct JournalLineRequest {
    pub account_id: Uuid,
    pub debit_amount: Option<rust_decimal::Decimal>,
    pub credit_amount: Option<rust_decimal::Decimal>,
    pub description: Option<String>,
}

/// Query parameters for listing journal entries
#[derive(Debug, Clone, Deserialize)]
pub struct JournalEntryQuery {
    pub status: Option<String>,
    pub limit: Option<i32>,
    pub offset: Option<i32>,
}

/// Customer invoice
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Invoice {
    pub id: Uuid,
    pub invoice_number: String,
    pub customer_id: Uuid,
    pub invoice_date: DateTime<Utc>,
    pub due_date: DateTime<Utc>,
    pub currency: String,
    pub subtotal: rust_decimal::Decimal,
    pub tax_amount: rust_decimal::Decimal,
    pub total_amount: rust_decimal::Decimal,
    pub status: InvoiceStatus,
    /// Link to journal entry when posted
    pub journal_entry_id: Option<Uuid>,
    /// Link to DKG claim
    pub claim_id: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "invoice_status", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum InvoiceStatus {
    Draft,
    Sent,
    Paid,
    PartiallyPaid,
    Overdue,
    Cancelled,
}

/// Line item on an invoice
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct InvoiceLine {
    pub id: Uuid,
    pub invoice_id: Uuid,
    pub line_number: i32,
    pub description: String,
    pub quantity: rust_decimal::Decimal,
    pub unit_price: rust_decimal::Decimal,
    pub line_total: rust_decimal::Decimal,
    pub tax_rate: Option<rust_decimal::Decimal>,
    pub tax_amount: Option<rust_decimal::Decimal>,
    /// Link to inventory item or service
    pub item_id: Option<Uuid>,
}

/// Vendor bill (accounts payable)
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Bill {
    pub id: Uuid,
    pub bill_number: String,
    pub vendor_id: Uuid,
    pub bill_date: DateTime<Utc>,
    pub due_date: DateTime<Utc>,
    pub currency: String,
    pub subtotal: rust_decimal::Decimal,
    pub tax_amount: rust_decimal::Decimal,
    pub total_amount: rust_decimal::Decimal,
    pub status: BillStatus,
    pub journal_entry_id: Option<Uuid>,
    pub claim_id: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "bill_status", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BillStatus {
    Draft,
    Approved,
    Paid,
    PartiallyPaid,
    Cancelled,
}

/// Line item on a bill
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct BillLine {
    pub id: Uuid,
    pub bill_id: Uuid,
    pub line_number: i32,
    pub description: String,
    pub quantity: rust_decimal::Decimal,
    pub unit_price: rust_decimal::Decimal,
    pub line_total: rust_decimal::Decimal,
    pub tax_rate: Option<rust_decimal::Decimal>,
    pub tax_amount: Option<rust_decimal::Decimal>,
    pub expense_account_id: Option<Uuid>,
}

/// Payment record (can be for invoice or bill)
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Payment {
    pub id: Uuid,
    pub payment_number: String,
    pub payment_type: PaymentType,
    pub payment_date: DateTime<Utc>,
    pub amount: rust_decimal::Decimal,
    pub currency: String,
    pub payment_method: PaymentMethod,
    pub reference: Option<String>,
    /// Invoice ID if this is a receivable payment
    pub invoice_id: Option<Uuid>,
    /// Bill ID if this is a payable payment
    pub bill_id: Option<Uuid>,
    pub journal_entry_id: Option<Uuid>,
    pub claim_id: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "payment_type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PaymentType {
    Receivable,
    Payable,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "payment_method", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PaymentMethod {
    Cash,
    Check,
    BankTransfer,
    CreditCard,
    Crypto,
    Other,
}

/// Request to create a new GL account
#[derive(Debug, Deserialize)]
pub struct CreateGlAccountRequest {
    pub account_number: String,
    pub account_name: String,
    pub account_type: AccountType,
    pub parent_account_id: Option<Uuid>,
    pub currency: String,
}

/// Request to create a new invoice
#[derive(Debug, Deserialize)]
pub struct CreateInvoiceRequest {
    pub customer_id: Uuid,
    pub invoice_date: DateTime<Utc>,
    pub due_date: DateTime<Utc>,
    pub currency: String,
    pub lines: Vec<CreateInvoiceLineRequest>,
}

#[derive(Debug, Deserialize)]
pub struct CreateInvoiceLineRequest {
    pub description: String,
    pub quantity: rust_decimal::Decimal,
    pub unit_price: rust_decimal::Decimal,
    pub tax_rate: Option<rust_decimal::Decimal>,
    pub item_id: Option<Uuid>,
}

/// Request to create a new bill
#[derive(Debug, Deserialize)]
pub struct CreateBillRequest {
    pub vendor_id: Uuid,
    pub bill_date: DateTime<Utc>,
    pub due_date: DateTime<Utc>,
    pub currency: String,
    pub lines: Vec<CreateBillLineRequest>,
}

#[derive(Debug, Deserialize)]
pub struct CreateBillLineRequest {
    pub description: String,
    pub quantity: rust_decimal::Decimal,
    pub unit_price: rust_decimal::Decimal,
    pub tax_rate: Option<rust_decimal::Decimal>,
    pub expense_account_id: Option<Uuid>,
}

/// Request to create a payment
#[derive(Debug, Deserialize)]
pub struct CreatePaymentRequest {
    pub payment_type: PaymentType,
    pub payment_date: DateTime<Utc>,
    pub amount: rust_decimal::Decimal,
    pub currency: String,
    pub payment_method: PaymentMethod,
    pub reference: Option<String>,
    pub invoice_id: Option<Uuid>,
    pub bill_id: Option<Uuid>,
}
