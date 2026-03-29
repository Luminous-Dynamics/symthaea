// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Finance module API endpoints
//!
//! REST API for GL, invoices, bills, and payments.

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use chrono::{Datelike, Utc};
use serde::Deserialize;
use serde_json::json;
use uuid::Uuid;

use super::models::*;
use super::ledger::LedgerService;
use super::invoicing::InvoicingService;
use super::payments::PaymentService;
use super::reporting::ReportingService;

/// Finance API state
#[derive(Clone)]
pub struct FinState {
    pub ledger: LedgerService,
    pub invoicing: InvoicingService,
    pub payments: PaymentService,
    pub reporting: ReportingService,
}

/// Query parameters for report date ranges
#[derive(Debug, Deserialize)]
pub struct ReportDateQuery {
    pub start_date: Option<chrono::DateTime<Utc>>,
    pub end_date: Option<chrono::DateTime<Utc>>,
    pub as_of_date: Option<chrono::DateTime<Utc>>,
}

/// Create finance module router
pub fn router(state: FinState) -> Router {
    Router::new()
        // GL Accounts
        .route("/v1/fin/accounts", post(create_gl_account))
        .route("/v1/fin/accounts", get(list_gl_accounts))
        .route("/v1/fin/accounts/:id", get(get_gl_account))
        // Journal Entries
        .route("/v1/fin/journal-entries", post(create_journal_entry))
        .route("/v1/fin/journal-entries", get(list_journal_entries))
        .route("/v1/fin/journal-entries/:id", get(get_journal_entry))
        .route("/v1/fin/journal-entries/:id/post", post(post_journal_entry))
        // Invoices
        .route("/v1/fin/invoices", post(create_invoice))
        .route("/v1/fin/invoices", get(list_invoices))
        .route("/v1/fin/invoices/:id", get(get_invoice))
        .route("/v1/fin/invoices/:id/send", post(send_invoice))
        // Bills
        .route("/v1/fin/bills", post(create_bill))
        .route("/v1/fin/bills", get(list_bills))
        .route("/v1/fin/bills/:id", get(get_bill))
        .route("/v1/fin/bills/:id/approve", post(approve_bill))
        // Payments
        .route("/v1/fin/payments", post(create_payment))
        .route("/v1/fin/payments", get(list_payments))
        .route("/v1/fin/payments/:id", get(get_payment))
        // Reports
        .route("/v1/fin/reports/trial-balance", get(trial_balance))
        .route("/v1/fin/reports/income-statement", get(income_statement))
        .route("/v1/fin/reports/balance-sheet", get(balance_sheet))
        .with_state(state)
}

// ============================================================================
// GL Account Handlers
// ============================================================================

async fn create_gl_account(
    State(state): State<FinState>,
    Json(req): Json<CreateGlAccountRequest>,
) -> impl IntoResponse {
    match state.ledger.create_account(req).await {
        Ok(account) => (StatusCode::CREATED, Json(account)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn list_gl_accounts(State(state): State<FinState>) -> impl IntoResponse {
    match state.ledger.list_accounts().await {
        Ok(accounts) => (StatusCode::OK, Json(accounts)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn get_gl_account(
    State(state): State<FinState>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.ledger.get_account(id).await {
        Ok(Some(account)) => (StatusCode::OK, Json(account)).into_response(),
        Ok(None) => (StatusCode::NOT_FOUND, Json(json!({ "error": "Account not found" }))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

// ============================================================================
// Journal Entry Handlers
// ============================================================================

async fn create_journal_entry(
    State(state): State<FinState>,
    Json(req): Json<CreateJournalEntryRequest>,
) -> impl IntoResponse {
    // Convert the request lines to the tuple format expected by the ledger service
    let lines: Vec<(Uuid, Option<rust_decimal::Decimal>, Option<rust_decimal::Decimal>, Option<String>)> = req
        .lines
        .iter()
        .map(|line| {
            (
                line.account_id,
                line.debit_amount,
                line.credit_amount,
                line.description.clone(),
            )
        })
        .collect();

    // Use a placeholder user ID - in production this would come from auth context
    let created_by = Uuid::nil();

    match state
        .ledger
        .create_journal_entry(req.description, req.reference, created_by, lines)
        .await
    {
        Ok(entry) => (StatusCode::CREATED, Json(json!(entry))).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn list_journal_entries(
    State(state): State<FinState>,
    Query(query): Query<JournalEntryQuery>,
) -> impl IntoResponse {
    // Parse status filter if provided
    let status = query.status.as_ref().and_then(|s| {
        match s.to_uppercase().as_str() {
            "DRAFT" => Some(EntryStatus::Draft),
            "POSTED" => Some(EntryStatus::Posted),
            "REVERSED" => Some(EntryStatus::Reversed),
            "VOIDED" => Some(EntryStatus::Voided),
            _ => None,
        }
    });

    let limit = query.limit.unwrap_or(100);
    let offset = query.offset.unwrap_or(0);

    match state.ledger.list_journal_entries(status, limit, offset).await {
        Ok(entries) => (StatusCode::OK, Json(json!({ "entries": entries }))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn get_journal_entry(
    State(state): State<FinState>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.ledger.get_journal_entry(id).await {
        Ok(Some(entry)) => (StatusCode::OK, Json(json!(entry))).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "Journal entry not found" })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn post_journal_entry(
    State(state): State<FinState>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.ledger.post_journal_entry(id).await {
        Ok(entry) => (StatusCode::OK, Json(json!(entry))).into_response(),
        Err(e) => {
            let error_msg = e.to_string();
            if error_msg.contains("no rows") {
                (
                    StatusCode::NOT_FOUND,
                    Json(json!({ "error": "Journal entry not found or already posted" })),
                )
                    .into_response()
            } else {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": error_msg })),
                )
                    .into_response()
            }
        }
    }
}

// ============================================================================
// Invoice Handlers
// ============================================================================

async fn create_invoice(
    State(state): State<FinState>,
    Json(req): Json<CreateInvoiceRequest>,
) -> impl IntoResponse {
    match state.invoicing.create_invoice(req).await {
        Ok(invoice) => (StatusCode::CREATED, Json(invoice)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn list_invoices(State(state): State<FinState>) -> impl IntoResponse {
    match state.invoicing.list_invoices().await {
        Ok(invoices) => (StatusCode::OK, Json(invoices)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn get_invoice(
    State(state): State<FinState>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.invoicing.get_invoice(id).await {
        Ok(Some(invoice)) => (StatusCode::OK, Json(invoice)).into_response(),
        Ok(None) => (StatusCode::NOT_FOUND, Json(json!({ "error": "Invoice not found" }))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn send_invoice(
    State(state): State<FinState>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.invoicing.send_invoice(id).await {
        Ok(invoice) => (StatusCode::OK, Json(invoice)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

// ============================================================================
// Bill Handlers
// ============================================================================

async fn create_bill(
    State(state): State<FinState>,
    Json(req): Json<CreateBillRequest>,
) -> impl IntoResponse {
    match state.invoicing.create_bill(req).await {
        Ok(bill) => (StatusCode::CREATED, Json(bill)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn list_bills(State(state): State<FinState>) -> impl IntoResponse {
    match state.invoicing.list_bills().await {
        Ok(bills) => (StatusCode::OK, Json(json!({ "bills": bills }))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn get_bill(
    State(state): State<FinState>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.invoicing.get_bill(id).await {
        Ok(Some(bill)) => (StatusCode::OK, Json(bill)).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "Bill not found" })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn approve_bill(
    State(state): State<FinState>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.invoicing.approve_bill(id).await {
        Ok(bill) => (StatusCode::OK, Json(bill)).into_response(),
        Err(e) => {
            let error_msg = e.to_string();
            if error_msg.contains("no rows") {
                (
                    StatusCode::NOT_FOUND,
                    Json(json!({ "error": "Bill not found or not in draft status" })),
                )
                    .into_response()
            } else {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": error_msg })),
                )
                    .into_response()
            }
        }
    }
}

// ============================================================================
// Payment Handlers
// ============================================================================

async fn create_payment(
    State(state): State<FinState>,
    Json(req): Json<CreatePaymentRequest>,
) -> impl IntoResponse {
    match state.payments.create_payment(req).await {
        Ok(payment) => (StatusCode::CREATED, Json(payment)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn list_payments(State(state): State<FinState>) -> impl IntoResponse {
    match state.payments.list_payments().await {
        Ok(payments) => (StatusCode::OK, Json(payments)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn get_payment(
    State(state): State<FinState>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.payments.get_payment(id).await {
        Ok(Some(payment)) => (StatusCode::OK, Json(payment)).into_response(),
        Ok(None) => (StatusCode::NOT_FOUND, Json(json!({ "error": "Payment not found" }))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

// ============================================================================
// Report Handlers
// ============================================================================

async fn trial_balance(
    State(state): State<FinState>,
    Query(query): Query<ReportDateQuery>,
) -> impl IntoResponse {
    let as_of_date = query.as_of_date.unwrap_or_else(Utc::now);

    match state.reporting.trial_balance(as_of_date).await {
        Ok(report) => (StatusCode::OK, Json(json!(report))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn income_statement(
    State(state): State<FinState>,
    Query(query): Query<ReportDateQuery>,
) -> impl IntoResponse {
    // Default to current month if no dates provided
    let now = Utc::now();
    let end_date = query.end_date.unwrap_or(now);
    let start_date = query.start_date.unwrap_or_else(|| {
        // Default to first day of current month
        chrono::DateTime::from_naive_utc_and_tz(
            chrono::NaiveDate::from_ymd_opt(now.year(), now.month(), 1)
                .unwrap_or(now.date_naive())
                .and_hms_opt(0, 0, 0)
                .unwrap(),
            Utc,
        )
    });

    match state.reporting.income_statement(start_date, end_date).await {
        Ok(report) => (StatusCode::OK, Json(json!(report))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn balance_sheet(
    State(state): State<FinState>,
    Query(query): Query<ReportDateQuery>,
) -> impl IntoResponse {
    let as_of_date = query.as_of_date.unwrap_or_else(Utc::now);

    match state.reporting.balance_sheet(as_of_date).await {
        Ok(report) => (StatusCode::OK, Json(json!(report))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}
