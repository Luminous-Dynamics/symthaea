// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bank Reconciliation API endpoints

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post, put},
    Router,
};
use chrono::NaiveDate;
use serde::Deserialize;
use serde_json::json;
use sqlx::PgPool;
use uuid::Uuid;

use super::reconciliation::{MatchType, ReconciliationService};

/// Reconciliation API state
#[derive(Clone)]
pub struct ReconciliationState {
    pub service: ReconciliationService,
}

impl ReconciliationState {
    pub fn new(pool: PgPool) -> Self {
        Self {
            service: ReconciliationService::new(pool),
        }
    }
}

/// Create reconciliation router
pub fn reconciliation_router(state: ReconciliationState) -> Router {
    Router::new()
        // Bank accounts
        .route("/v1/fin/bank-accounts", post(create_bank_account))
        .route("/v1/fin/bank-accounts", get(list_bank_accounts))
        .route("/v1/fin/bank-accounts/:id", get(get_bank_account))
        .route("/v1/fin/bank-accounts/:id/link-gl", put(link_gl_account))
        .route("/v1/fin/bank-accounts/:id/sync", post(sync_transactions))
        // Transactions
        .route("/v1/fin/bank-transactions", get(list_transactions))
        .route("/v1/fin/bank-transactions/unmatched", get(list_unmatched))
        .route("/v1/fin/bank-transactions/:id/match", post(match_transaction))
        .route("/v1/fin/bank-transactions/:id/ignore", post(ignore_transaction))
        // Auto-matching
        .route("/v1/fin/reconciliation/auto-match", post(auto_match))
        .route("/v1/fin/reconciliation/stats", get(get_stats))
        .with_state(state)
}

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct CreateBankAccountRequest {
    pub tenant_id: Uuid,
    pub plaid_account_id: String,
    pub plaid_item_id: String,
    pub account_name: String,
    pub account_type: String,
    pub account_subtype: Option<String>,
    pub mask: Option<String>,
    pub institution_name: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct LinkGlAccountRequest {
    pub gl_account_id: Uuid,
}

#[derive(Debug, Deserialize)]
pub struct MatchTransactionRequest {
    pub match_type: String,  // INVOICE, BILL, PAYMENT
    pub match_id: Uuid,
    pub user_id: Uuid,
}

#[derive(Debug, Deserialize)]
pub struct IgnoreTransactionRequest {
    pub reason: String,
    pub user_id: Uuid,
}

#[derive(Debug, Deserialize)]
pub struct AutoMatchRequest {
    pub tenant_id: Uuid,
}

#[derive(Debug, Deserialize)]
pub struct ListTransactionsQuery {
    pub tenant_id: Uuid,
    pub bank_account_id: Option<Uuid>,
    pub status: Option<String>,
    pub from_date: Option<NaiveDate>,
    pub to_date: Option<NaiveDate>,
}

#[derive(Debug, Deserialize)]
pub struct StatsQuery {
    pub tenant_id: Uuid,
    pub bank_account_id: Option<Uuid>,
}

// ============================================================================
// Handlers
// ============================================================================

async fn create_bank_account(
    State(state): State<ReconciliationState>,
    Json(req): Json<CreateBankAccountRequest>,
) -> impl IntoResponse {
    match state.service.create_bank_account(
        req.tenant_id,
        &req.plaid_account_id,
        &req.plaid_item_id,
        &req.account_name,
        &req.account_type,
        req.account_subtype.as_deref(),
        req.mask.as_deref(),
        req.institution_name.as_deref(),
    ).await {
        Ok(account) => (StatusCode::CREATED, Json(json!(account))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ).into_response(),
    }
}

async fn list_bank_accounts(
    State(state): State<ReconciliationState>,
    Query(query): Query<ListTransactionsQuery>,
) -> impl IntoResponse {
    match state.service.list_bank_accounts(query.tenant_id).await {
        Ok(accounts) => (
            StatusCode::OK,
            Json(json!({
                "accounts": accounts,
                "tenant_id": query.tenant_id,
                "count": accounts.len()
            })),
        ).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ).into_response(),
    }
}

async fn get_bank_account(
    State(state): State<ReconciliationState>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.service.get_bank_account_by_id(id).await {
        Ok(Some(account)) => (StatusCode::OK, Json(json!(account))).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("Bank account {} not found", id) })),
        ).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ).into_response(),
    }
}

async fn link_gl_account(
    State(state): State<ReconciliationState>,
    Path(bank_account_id): Path<Uuid>,
    Json(req): Json<LinkGlAccountRequest>,
) -> impl IntoResponse {
    match state.service.link_gl_account(bank_account_id, req.gl_account_id).await {
        Ok(()) => (StatusCode::OK, Json(json!({ "success": true }))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ).into_response(),
    }
}

async fn sync_transactions(
    State(state): State<ReconciliationState>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    // Check if Plaid is configured
    if !crate::integrations::plaid::PlaidConfig::is_configured() {
        return (
            StatusCode::OK,
            Json(json!({
                "bank_account_id": id,
                "imported": 0,
                "updated": 0,
                "message": "Plaid integration not configured. Set PLAID_CLIENT_ID and PLAID_SECRET environment variables."
            })),
        ).into_response();
    }

    // Get the access token for this bank account
    let access_token = match state.service.get_plaid_access_token(id).await {
        Ok(Some(token)) => token,
        Ok(None) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": "Bank account is not linked to Plaid. Complete the Plaid Link flow first."
                })),
            ).into_response();
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ).into_response();
        }
    };

    // Create Plaid client and sync transactions
    let plaid_client = match crate::integrations::plaid::PlaidClient::new() {
        Ok(client) => client,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("Failed to create Plaid client: {}", e) })),
            ).into_response();
        }
    };

    // Get the current sync cursor (if any) from the bank account
    // For simplicity, we'll start fresh each time - in production you'd store the cursor
    match plaid_client.sync_transactions(&access_token, None).await {
        Ok(sync_response) => {
            // Import the transactions
            let transactions = sync_response.added;
            match state.service.import_transactions(id, transactions).await {
                Ok(result) => {
                    // Update the cursor for next sync
                    let _ = state.service.update_sync_cursor(id, &sync_response.next_cursor).await;

                    (
                        StatusCode::OK,
                        Json(json!({
                            "bank_account_id": id,
                            "imported": result.imported,
                            "updated": result.updated,
                            "skipped": result.skipped,
                            "has_more": sync_response.has_more,
                            "message": "Transaction sync completed successfully"
                        })),
                    ).into_response()
                }
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": format!("Failed to import transactions: {}", e) })),
                ).into_response(),
            }
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("Plaid sync failed: {}", e) })),
        ).into_response(),
    }
}

async fn list_transactions(
    State(state): State<ReconciliationState>,
    Query(query): Query<ListTransactionsQuery>,
) -> impl IntoResponse {
    match state.service.list_transactions(
        query.tenant_id,
        query.bank_account_id,
        query.status.as_deref(),
        query.from_date,
        query.to_date,
    ).await {
        Ok(transactions) => (
            StatusCode::OK,
            Json(json!({
                "transactions": transactions,
                "count": transactions.len(),
                "tenant_id": query.tenant_id,
                "filters": {
                    "bank_account_id": query.bank_account_id,
                    "status": query.status,
                    "from_date": query.from_date,
                    "to_date": query.to_date
                }
            })),
        ).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ).into_response(),
    }
}

async fn list_unmatched(
    State(state): State<ReconciliationState>,
    Query(query): Query<ListTransactionsQuery>,
) -> impl IntoResponse {
    match state.service.get_unmatched_transactions(query.tenant_id).await {
        Ok(transactions) => (StatusCode::OK, Json(json!({ "transactions": transactions }))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ).into_response(),
    }
}

async fn match_transaction(
    State(state): State<ReconciliationState>,
    Path(transaction_id): Path<Uuid>,
    Json(req): Json<MatchTransactionRequest>,
) -> impl IntoResponse {
    let match_type = match req.match_type.to_uppercase().as_str() {
        "INVOICE" => MatchType::Invoice,
        "BILL" => MatchType::Bill,
        "PAYMENT" => MatchType::Payment,
        _ => return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "Invalid match type. Must be INVOICE, BILL, or PAYMENT" })),
        ).into_response(),
    };

    match state.service.manual_match(transaction_id, match_type, req.match_id, req.user_id).await {
        Ok(()) => (StatusCode::OK, Json(json!({ "success": true, "transaction_id": transaction_id }))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ).into_response(),
    }
}

async fn ignore_transaction(
    State(state): State<ReconciliationState>,
    Path(transaction_id): Path<Uuid>,
    Json(req): Json<IgnoreTransactionRequest>,
) -> impl IntoResponse {
    match state.service.ignore_transaction(transaction_id, &req.reason, req.user_id).await {
        Ok(()) => (StatusCode::OK, Json(json!({ "success": true, "transaction_id": transaction_id }))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ).into_response(),
    }
}

async fn auto_match(
    State(state): State<ReconciliationState>,
    Json(req): Json<AutoMatchRequest>,
) -> impl IntoResponse {
    match state.service.auto_match(req.tenant_id).await {
        Ok(suggestions) => (
            StatusCode::OK,
            Json(json!({
                "suggestions": suggestions,
                "count": suggestions.len()
            })),
        ).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ).into_response(),
    }
}

async fn get_stats(
    State(state): State<ReconciliationState>,
    Query(query): Query<StatsQuery>,
) -> impl IntoResponse {
    match state.service.get_stats(query.tenant_id, query.bank_account_id).await {
        Ok(stats) => (StatusCode::OK, Json(json!(stats))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ).into_response(),
    }
}
