// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-Currency API endpoints

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use chrono::NaiveDate;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sqlx::PgPool;
use uuid::Uuid;

use super::currency::{CurrencyService, RateType};

/// Currency API state
#[derive(Clone)]
pub struct CurrencyState {
    pub service: CurrencyService,
}

impl CurrencyState {
    pub fn new(pool: PgPool) -> Self {
        Self {
            service: CurrencyService::new(pool),
        }
    }
}

/// Create currency router
pub fn currency_router(state: CurrencyState) -> Router {
    Router::new()
        // Currencies
        .route("/v1/fin/currencies", get(list_currencies))
        .route("/v1/fin/currencies/:code", get(get_currency))
        // Tenant config
        .route("/v1/fin/currency-config", get(get_config))
        .route("/v1/fin/currency-config", post(init_config))
        .route("/v1/fin/currency-config/enable/:code", post(enable_currency))
        .route("/v1/fin/currency-config/disable/:code", post(disable_currency))
        // Exchange rates
        .route("/v1/fin/exchange-rates", get(list_rates))
        .route("/v1/fin/exchange-rates", post(set_rate))
        .route("/v1/fin/exchange-rates/current", get(get_rate))
        // Conversion
        .route("/v1/fin/convert", post(convert_amount))
        .with_state(state)
}

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct CurrencyTenantQuery {
    pub tenant_id: Uuid,
}

#[derive(Debug, Deserialize)]
pub struct InitConfigRequest {
    pub tenant_id: Uuid,
    pub base_currency: String,
}

#[derive(Debug, Deserialize)]
pub struct SetRateRequest {
    pub tenant_id: Uuid,
    pub from_currency: String,
    pub to_currency: String,
    pub rate: Decimal,
    pub rate_date: Option<NaiveDate>,
    pub rate_type: Option<String>,
    pub source: Option<String>,
    pub user_id: Option<Uuid>,
}

#[derive(Debug, Deserialize)]
pub struct GetRateQuery {
    pub tenant_id: Uuid,
    pub from_currency: String,
    pub to_currency: String,
    pub date: Option<NaiveDate>,
}

#[derive(Debug, Deserialize)]
pub struct ListRatesQuery {
    pub tenant_id: Uuid,
    pub from_date: NaiveDate,
    pub to_date: NaiveDate,
}

#[derive(Debug, Deserialize)]
pub struct ConvertRequest {
    pub tenant_id: Uuid,
    pub amount: Decimal,
    pub from_currency: String,
    pub to_currency: String,
    pub date: Option<NaiveDate>,
}

#[derive(Debug, Serialize)]
pub struct ConvertResponse {
    pub original_amount: Decimal,
    pub original_currency: String,
    pub converted_amount: Decimal,
    pub target_currency: String,
    pub exchange_rate: Decimal,
    pub date: NaiveDate,
}

// ============================================================================
// Handlers
// ============================================================================

async fn list_currencies(
    State(state): State<CurrencyState>,
) -> impl IntoResponse {
    match state.service.list_currencies().await {
        Ok(currencies) => (StatusCode::OK, Json(json!({ "currencies": currencies }))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ).into_response(),
    }
}

async fn get_currency(
    State(state): State<CurrencyState>,
    Path(code): Path<String>,
) -> impl IntoResponse {
    match state.service.get_currency(&code).await {
        Ok(currency) => (StatusCode::OK, Json(json!(currency))).into_response(),
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": e.to_string() })),
        ).into_response(),
    }
}

async fn get_config(
    State(state): State<CurrencyState>,
    Query(query): Query<CurrencyTenantQuery>,
) -> impl IntoResponse {
    match state.service.get_config(query.tenant_id).await {
        Ok(config) => (StatusCode::OK, Json(json!(config))).into_response(),
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": e.to_string() })),
        ).into_response(),
    }
}

async fn init_config(
    State(state): State<CurrencyState>,
    Json(req): Json<InitConfigRequest>,
) -> impl IntoResponse {
    match state.service.init_config(req.tenant_id, &req.base_currency).await {
        Ok(config) => (StatusCode::CREATED, Json(json!(config))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ).into_response(),
    }
}

async fn enable_currency(
    State(state): State<CurrencyState>,
    Path(code): Path<String>,
    Query(query): Query<CurrencyTenantQuery>,
) -> impl IntoResponse {
    match state.service.enable_currency(query.tenant_id, &code).await {
        Ok(()) => (StatusCode::OK, Json(json!({ "enabled": code }))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ).into_response(),
    }
}

async fn disable_currency(
    State(state): State<CurrencyState>,
    Path(code): Path<String>,
    Query(query): Query<CurrencyTenantQuery>,
) -> impl IntoResponse {
    match state.service.disable_currency(query.tenant_id, &code).await {
        Ok(()) => (StatusCode::OK, Json(json!({ "disabled": code }))).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": e.to_string() })),
        ).into_response(),
    }
}

async fn list_rates(
    State(state): State<CurrencyState>,
    Query(query): Query<ListRatesQuery>,
) -> impl IntoResponse {
    match state.service.list_rates(query.tenant_id, query.from_date, query.to_date).await {
        Ok(rates) => (StatusCode::OK, Json(json!({ "rates": rates }))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ).into_response(),
    }
}

async fn set_rate(
    State(state): State<CurrencyState>,
    Json(req): Json<SetRateRequest>,
) -> impl IntoResponse {
    let rate_date = req.rate_date.unwrap_or_else(|| chrono::Utc::now().date_naive());
    let rate_type = match req.rate_type.as_deref() {
        Some("CUSTOM") => RateType::Custom,
        Some("BUDGET") => RateType::Budget,
        _ => RateType::Market,
    };

    match state.service.set_rate(
        req.tenant_id,
        &req.from_currency,
        &req.to_currency,
        req.rate,
        rate_date,
        rate_type,
        req.source.as_deref(),
        req.user_id,
    ).await {
        Ok(rate) => (StatusCode::CREATED, Json(json!(rate))).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": e.to_string() })),
        ).into_response(),
    }
}

async fn get_rate(
    State(state): State<CurrencyState>,
    Query(query): Query<GetRateQuery>,
) -> impl IntoResponse {
    match state.service.get_rate(
        query.tenant_id,
        &query.from_currency,
        &query.to_currency,
        query.date,
    ).await {
        Ok(rate) => (StatusCode::OK, Json(json!(rate))).into_response(),
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": e.to_string() })),
        ).into_response(),
    }
}

async fn convert_amount(
    State(state): State<CurrencyState>,
    Json(req): Json<ConvertRequest>,
) -> impl IntoResponse {
    let date = req.date.unwrap_or_else(|| chrono::Utc::now().date_naive());

    match state.service.convert(
        req.tenant_id,
        req.amount,
        &req.from_currency,
        &req.to_currency,
        Some(date),
    ).await {
        Ok(result) => {
            let response = ConvertResponse {
                original_amount: req.amount,
                original_currency: req.from_currency,
                converted_amount: result.amount,
                target_currency: result.currency,
                exchange_rate: result.exchange_rate.unwrap_or(Decimal::ONE),
                date,
            };
            (StatusCode::OK, Json(json!(response))).into_response()
        }
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": e.to_string() })),
        ).into_response(),
    }
}
