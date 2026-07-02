// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CRM API Endpoints

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post, put},
    Json, Router,
};
use rust_decimal::Decimal;
use serde::Deserialize;
use serde_json::json;
use sqlx::PgPool;
use uuid::Uuid;

use super::{
    contacts::{ContactService, CreateAccountRequest, CreateContactRequest},
    leads::{CreateLeadRequest, LeadService, LeadStatus},
    opportunities::{CreateOpportunityRequest, OpportunityService},
    activities::{ActivityService, CreateActivityRequest, LogCallRequest},
};

// ============================================================================
// State
// ============================================================================

#[derive(Clone)]
pub struct CrmState {
    pub contact_service: ContactService,
    pub lead_service: LeadService,
    pub opportunity_service: OpportunityService,
    pub activity_service: ActivityService,
}

impl CrmState {
    pub fn new(pool: PgPool) -> Self {
        Self {
            contact_service: ContactService::new(pool.clone()),
            lead_service: LeadService::new(pool.clone()),
            opportunity_service: OpportunityService::new(pool.clone()),
            activity_service: ActivityService::new(pool),
        }
    }
}

// ============================================================================
// Router
// ============================================================================

pub fn crm_router(state: CrmState) -> Router {
    Router::new()
        // Accounts
        .route("/v1/crm/accounts", get(list_accounts))
        .route("/v1/crm/accounts", post(create_account))
        .route("/v1/crm/accounts/:id", get(get_account))
        .route("/v1/crm/accounts/:id", put(update_account))
        .route("/v1/crm/accounts/:id/contacts", get(get_account_contacts))
        // Contacts
        .route("/v1/crm/contacts", get(list_contacts))
        .route("/v1/crm/contacts", post(create_contact))
        .route("/v1/crm/contacts/:id", get(get_contact))
        .route("/v1/crm/contacts/search", get(search_contacts))
        // Leads
        .route("/v1/crm/leads", get(list_leads))
        .route("/v1/crm/leads", post(create_lead))
        .route("/v1/crm/leads/:id", get(get_lead))
        .route("/v1/crm/leads/:id/status", put(update_lead_status))
        .route("/v1/crm/leads/:id/score", put(update_lead_score))
        .route("/v1/crm/leads/:id/convert", post(convert_lead))
        .route("/v1/crm/leads/stats", get(get_lead_stats))
        // Opportunities
        .route("/v1/crm/opportunities", get(list_opportunities))
        .route("/v1/crm/opportunities", post(create_opportunity))
        .route("/v1/crm/opportunities/:id", get(get_opportunity))
        .route("/v1/crm/opportunities/:id/stage", put(update_opportunity_stage))
        .route("/v1/crm/opportunities/:id/amount", put(update_opportunity_amount))
        .route("/v1/crm/opportunities/pipeline", get(get_pipeline_summary))
        .route("/v1/crm/opportunities/closing-soon", get(get_closing_soon))
        // Activities
        .route("/v1/crm/activities", get(list_activities))
        .route("/v1/crm/activities", post(create_activity))
        .route("/v1/crm/activities/:id", get(get_activity))
        .route("/v1/crm/activities/:id/complete", post(complete_activity))
        .route("/v1/crm/activities/tasks", get(get_open_tasks))
        .route("/v1/crm/activities/events", get(get_upcoming_events))
        .route("/v1/crm/activities/log-call", post(log_call))
        .route("/v1/crm/activities/timeline", get(get_timeline))
        .with_state(state)
}

// ============================================================================
// Query Parameters
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct TenantQuery {
    pub tenant_id: Uuid,
}

#[derive(Debug, Deserialize)]
pub struct ListQuery {
    pub tenant_id: Uuid,
    pub limit: Option<i32>,
    pub offset: Option<i32>,
}

#[derive(Debug, Deserialize)]
pub struct AccountListQuery {
    pub tenant_id: Uuid,
    pub account_type: Option<String>,
    pub search: Option<String>,
    pub limit: Option<i32>,
    pub offset: Option<i32>,
}

#[derive(Debug, Deserialize)]
pub struct ContactListQuery {
    pub tenant_id: Uuid,
    pub account_id: Option<Uuid>,
    pub search: Option<String>,
    pub limit: Option<i32>,
    pub offset: Option<i32>,
}

#[derive(Debug, Deserialize)]
pub struct LeadListQuery {
    pub tenant_id: Uuid,
    pub status: Option<String>,
    pub owner_id: Option<Uuid>,
    pub limit: Option<i32>,
    pub offset: Option<i32>,
}

#[derive(Debug, Deserialize)]
pub struct OpportunityListQuery {
    pub tenant_id: Uuid,
    pub stage: Option<String>,
    pub owner_id: Option<Uuid>,
    pub is_closed: Option<bool>,
    pub limit: Option<i32>,
    pub offset: Option<i32>,
}

#[derive(Debug, Deserialize)]
pub struct ActivityListQuery {
    pub tenant_id: Uuid,
    pub account_id: Option<Uuid>,
    pub contact_id: Option<Uuid>,
    pub lead_id: Option<Uuid>,
    pub opportunity_id: Option<Uuid>,
    pub limit: Option<i32>,
}

#[derive(Debug, Deserialize)]
pub struct UserQuery {
    pub tenant_id: Uuid,
    pub user_id: Uuid,
}

#[derive(Debug, Deserialize)]
pub struct TimelineQuery {
    pub tenant_id: Uuid,
    pub account_id: Option<Uuid>,
    pub contact_id: Option<Uuid>,
    pub limit: Option<i32>,
}

// ============================================================================
// Account Handlers
// ============================================================================

async fn list_accounts(
    State(state): State<CrmState>,
    Query(query): Query<AccountListQuery>,
) -> impl IntoResponse {
    match state.contact_service.list_accounts(
        query.tenant_id,
        query.account_type.as_deref(),
        query.search.as_deref(),
        query.limit.unwrap_or(50),
        query.offset.unwrap_or(0),
    ).await {
        Ok(accounts) => (StatusCode::OK, Json(json!({ "accounts": accounts }))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

async fn create_account(
    State(state): State<CrmState>,
    Query(query): Query<TenantQuery>,
    Json(req): Json<CreateAccountRequest>,
) -> impl IntoResponse {
    match state.contact_service.create_account(query.tenant_id, req).await {
        Ok(account) => (StatusCode::CREATED, Json(json!(account))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

async fn get_account(
    State(state): State<CrmState>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.contact_service.get_account(id).await {
        Ok(account) => (StatusCode::OK, Json(json!(account))).into_response(),
        Err(e) => (StatusCode::NOT_FOUND, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct UpdateAccountRequest {
    pub name: Option<String>,
    pub industry: Option<String>,
    pub website: Option<String>,
    pub phone: Option<String>,
    pub email: Option<String>,
}

async fn update_account(
    State(state): State<CrmState>,
    Path(id): Path<Uuid>,
    Json(req): Json<UpdateAccountRequest>,
) -> impl IntoResponse {
    match state.contact_service.update_account(
        id,
        req.name.as_deref(),
        req.industry.as_deref(),
        req.website.as_deref(),
        req.phone.as_deref(),
        req.email.as_deref(),
    ).await {
        Ok(account) => (StatusCode::OK, Json(json!(account))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

async fn get_account_contacts(
    State(state): State<CrmState>,
    Path(account_id): Path<Uuid>,
) -> impl IntoResponse {
    match state.contact_service.get_account_contacts(account_id).await {
        Ok(contacts) => (StatusCode::OK, Json(json!({ "contacts": contacts }))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

// ============================================================================
// Contact Handlers
// ============================================================================

async fn list_contacts(
    State(state): State<CrmState>,
    Query(query): Query<ContactListQuery>,
) -> impl IntoResponse {
    match state.contact_service.list_contacts(
        query.tenant_id,
        query.account_id,
        query.search.as_deref(),
        query.limit.unwrap_or(50),
        query.offset.unwrap_or(0),
    ).await {
        Ok(contacts) => (StatusCode::OK, Json(json!({ "contacts": contacts }))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

async fn create_contact(
    State(state): State<CrmState>,
    Query(query): Query<TenantQuery>,
    Json(req): Json<CreateContactRequest>,
) -> impl IntoResponse {
    match state.contact_service.create_contact(query.tenant_id, req).await {
        Ok(contact) => (StatusCode::CREATED, Json(json!(contact))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

async fn get_contact(
    State(state): State<CrmState>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.contact_service.get_contact(id).await {
        Ok(contact) => (StatusCode::OK, Json(json!(contact))).into_response(),
        Err(e) => (StatusCode::NOT_FOUND, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct SearchQuery {
    pub tenant_id: Uuid,
    pub q: String,
    pub limit: Option<i32>,
}

async fn search_contacts(
    State(state): State<CrmState>,
    Query(query): Query<SearchQuery>,
) -> impl IntoResponse {
    match state.contact_service.search_contacts(query.tenant_id, &query.q, query.limit.unwrap_or(20)).await {
        Ok(contacts) => (StatusCode::OK, Json(json!({ "contacts": contacts }))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

// ============================================================================
// Lead Handlers
// ============================================================================

async fn list_leads(
    State(state): State<CrmState>,
    Query(query): Query<LeadListQuery>,
) -> impl IntoResponse {
    match state.lead_service.list_leads(
        query.tenant_id,
        query.status.as_deref(),
        query.owner_id,
        query.limit.unwrap_or(50),
        query.offset.unwrap_or(0),
    ).await {
        Ok(leads) => (StatusCode::OK, Json(json!({ "leads": leads }))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

async fn create_lead(
    State(state): State<CrmState>,
    Query(query): Query<TenantQuery>,
    Json(req): Json<CreateLeadRequest>,
) -> impl IntoResponse {
    match state.lead_service.create_lead(query.tenant_id, req).await {
        Ok(lead) => (StatusCode::CREATED, Json(json!(lead))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

async fn get_lead(
    State(state): State<CrmState>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.lead_service.get_lead(id).await {
        Ok(lead) => (StatusCode::OK, Json(json!(lead))).into_response(),
        Err(e) => (StatusCode::NOT_FOUND, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct UpdateLeadStatusRequest {
    pub status: String,
}

async fn update_lead_status(
    State(state): State<CrmState>,
    Path(id): Path<Uuid>,
    Json(req): Json<UpdateLeadStatusRequest>,
) -> impl IntoResponse {
    let status = match req.status.as_str() {
        "NEW" => LeadStatus::New,
        "CONTACTED" => LeadStatus::Contacted,
        "QUALIFIED" => LeadStatus::Qualified,
        "UNQUALIFIED" => LeadStatus::Unqualified,
        "CONVERTED" => LeadStatus::Converted,
        _ => return (StatusCode::BAD_REQUEST, Json(json!({ "error": "Invalid status" }))).into_response(),
    };

    match state.lead_service.update_status(id, status).await {
        Ok(lead) => (StatusCode::OK, Json(json!(lead))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct UpdateLeadScoreRequest {
    pub score: i32,
}

async fn update_lead_score(
    State(state): State<CrmState>,
    Path(id): Path<Uuid>,
    Json(req): Json<UpdateLeadScoreRequest>,
) -> impl IntoResponse {
    match state.lead_service.update_score(id, req.score).await {
        Ok(lead) => (StatusCode::OK, Json(json!(lead))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct ConvertLeadRequest {
    pub create_opportunity: Option<bool>,
    pub opportunity_name: Option<String>,
    pub opportunity_amount: Option<Decimal>,
}

async fn convert_lead(
    State(state): State<CrmState>,
    Path(id): Path<Uuid>,
    Json(req): Json<ConvertLeadRequest>,
) -> impl IntoResponse {
    match state.lead_service.convert_lead(
        id,
        req.create_opportunity.unwrap_or(true),
        req.opportunity_name.as_deref(),
        req.opportunity_amount,
    ).await {
        Ok(result) => (StatusCode::OK, Json(json!(result))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

async fn get_lead_stats(
    State(state): State<CrmState>,
    Query(query): Query<TenantQuery>,
) -> impl IntoResponse {
    match state.lead_service.get_stats(query.tenant_id).await {
        Ok(stats) => (StatusCode::OK, Json(json!(stats))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

// ============================================================================
// Opportunity Handlers
// ============================================================================

async fn list_opportunities(
    State(state): State<CrmState>,
    Query(query): Query<OpportunityListQuery>,
) -> impl IntoResponse {
    match state.opportunity_service.list_opportunities(
        query.tenant_id,
        query.stage.as_deref(),
        query.owner_id,
        query.is_closed,
        query.limit.unwrap_or(50),
        query.offset.unwrap_or(0),
    ).await {
        Ok(opps) => (StatusCode::OK, Json(json!({ "opportunities": opps }))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

async fn create_opportunity(
    State(state): State<CrmState>,
    Query(query): Query<TenantQuery>,
    Json(req): Json<CreateOpportunityRequest>,
) -> impl IntoResponse {
    match state.opportunity_service.create_opportunity(query.tenant_id, req).await {
        Ok(opp) => (StatusCode::CREATED, Json(json!(opp))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

async fn get_opportunity(
    State(state): State<CrmState>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.opportunity_service.get_opportunity(id).await {
        Ok(opp) => (StatusCode::OK, Json(json!(opp))).into_response(),
        Err(e) => (StatusCode::NOT_FOUND, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct UpdateStageRequest {
    pub stage: String,
}

async fn update_opportunity_stage(
    State(state): State<CrmState>,
    Path(id): Path<Uuid>,
    Json(req): Json<UpdateStageRequest>,
) -> impl IntoResponse {
    match state.opportunity_service.update_stage(id, &req.stage).await {
        Ok(opp) => (StatusCode::OK, Json(json!(opp))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct UpdateAmountRequest {
    pub amount: Decimal,
}

async fn update_opportunity_amount(
    State(state): State<CrmState>,
    Path(id): Path<Uuid>,
    Json(req): Json<UpdateAmountRequest>,
) -> impl IntoResponse {
    match state.opportunity_service.update_amount(id, req.amount).await {
        Ok(opp) => (StatusCode::OK, Json(json!(opp))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

async fn get_pipeline_summary(
    State(state): State<CrmState>,
    Query(query): Query<TenantQuery>,
) -> impl IntoResponse {
    match state.opportunity_service.get_pipeline_summary(query.tenant_id).await {
        Ok(summary) => (StatusCode::OK, Json(json!(summary))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

async fn get_closing_soon(
    State(state): State<CrmState>,
    Query(query): Query<TenantQuery>,
) -> impl IntoResponse {
    match state.opportunity_service.get_closing_this_month(query.tenant_id).await {
        Ok(opps) => (StatusCode::OK, Json(json!({ "opportunities": opps }))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

// ============================================================================
// Activity Handlers
// ============================================================================

async fn list_activities(
    State(state): State<CrmState>,
    Query(query): Query<ActivityListQuery>,
) -> impl IntoResponse {
    match state.activity_service.list_for_record(
        query.tenant_id,
        query.account_id,
        query.contact_id,
        query.lead_id,
        query.opportunity_id,
        query.limit.unwrap_or(50),
    ).await {
        Ok(activities) => (StatusCode::OK, Json(json!({ "activities": activities }))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

async fn create_activity(
    State(state): State<CrmState>,
    Query(query): Query<UserQuery>,
    Json(req): Json<CreateActivityRequest>,
) -> impl IntoResponse {
    match state.activity_service.create_activity(query.tenant_id, query.user_id, req).await {
        Ok(activity) => (StatusCode::CREATED, Json(json!(activity))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

async fn get_activity(
    State(state): State<CrmState>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.activity_service.get_activity(id).await {
        Ok(activity) => (StatusCode::OK, Json(json!(activity))).into_response(),
        Err(e) => (StatusCode::NOT_FOUND, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

async fn complete_activity(
    State(state): State<CrmState>,
    Path(id): Path<Uuid>,
    Query(query): Query<UserQuery>,
) -> impl IntoResponse {
    match state.activity_service.complete_activity(id, query.user_id).await {
        Ok(activity) => (StatusCode::OK, Json(json!(activity))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

async fn get_open_tasks(
    State(state): State<CrmState>,
    Query(query): Query<UserQuery>,
) -> impl IntoResponse {
    match state.activity_service.get_open_tasks(query.tenant_id, query.user_id).await {
        Ok(tasks) => (StatusCode::OK, Json(json!({ "tasks": tasks }))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct EventsQuery {
    pub tenant_id: Uuid,
    pub user_id: Uuid,
    pub days: Option<i32>,
}

async fn get_upcoming_events(
    State(state): State<CrmState>,
    Query(query): Query<EventsQuery>,
) -> impl IntoResponse {
    match state.activity_service.get_upcoming_events(query.tenant_id, query.user_id, query.days.unwrap_or(7)).await {
        Ok(events) => (StatusCode::OK, Json(json!({ "events": events }))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

async fn log_call(
    State(state): State<CrmState>,
    Query(query): Query<UserQuery>,
    Json(req): Json<LogCallRequest>,
) -> impl IntoResponse {
    match state.activity_service.log_call(query.tenant_id, query.user_id, req).await {
        Ok(activity) => (StatusCode::CREATED, Json(json!(activity))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

async fn get_timeline(
    State(state): State<CrmState>,
    Query(query): Query<TimelineQuery>,
) -> impl IntoResponse {
    match state.activity_service.get_timeline(
        query.tenant_id,
        query.account_id,
        query.contact_id,
        query.limit.unwrap_or(50),
    ).await {
        Ok(activities) => (StatusCode::OK, Json(json!({ "timeline": activities }))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}
