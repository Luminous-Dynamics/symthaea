// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HR API Endpoints
//!
//! REST API for Human Resources management.

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post, put},
    Json, Router,
};
use chrono::NaiveDate;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;

use crate::auth::Claims;

use super::{
    employees::{CreateEmployeeRequest, EmployeeService},
    departments::{CreateDepartmentRequest, DepartmentService},
    leave::{CreateLeaveRequest, LeaveService},
    payroll::{CreatePayRunRequest, PayrollService},
    HrError,
};

/// Get tenant_id from claims or return error
fn get_tenant_id(claims: &Claims) -> Result<Uuid, HrError> {
    claims.tenant_id.ok_or_else(|| HrError::Validation("Tenant ID required".into()))
}

// ============================================================================
// State
// ============================================================================

/// HR API state
#[derive(Clone)]
pub struct HrState {
    pub employee_service: EmployeeService,
    pub department_service: DepartmentService,
    pub leave_service: LeaveService,
    pub payroll_service: PayrollService,
}

impl HrState {
    pub fn new(pool: PgPool) -> Self {
        Self {
            employee_service: EmployeeService::new(pool.clone()),
            department_service: DepartmentService::new(pool.clone()),
            leave_service: LeaveService::new(pool.clone()),
            payroll_service: PayrollService::new(pool),
        }
    }
}

// ============================================================================
// Query Parameters
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct EmployeeQuery {
    pub department_id: Option<Uuid>,
    pub status: Option<String>,
    pub search: Option<String>,
    pub limit: Option<i32>,
    pub offset: Option<i32>,
}

#[derive(Debug, Deserialize)]
pub struct LeaveQuery {
    pub year: Option<i32>,
}

#[derive(Debug, Deserialize)]
pub struct PayRunQuery {
    pub year: Option<i32>,
}

#[derive(Debug, Deserialize)]
pub struct DateQuery {
    pub date: Option<NaiveDate>,
}

// ============================================================================
// Request Bodies
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct UpdateEmployeeStatus {
    pub status: String,
    pub end_date: Option<NaiveDate>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateLeaveBalance {
    pub annual_leave: Option<Decimal>,
    pub sick_leave: Option<Decimal>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateDepartment {
    pub name: Option<String>,
    pub description: Option<String>,
    pub manager_id: Option<Uuid>,
}

#[derive(Debug, Deserialize)]
pub struct LeaveApproval {
    pub approver_id: Uuid,
}

#[derive(Debug, Deserialize)]
pub struct LeaveRejection {
    pub approver_id: Uuid,
    pub reason: String,
}

#[derive(Debug, Deserialize)]
pub struct PayRunApproval {
    pub approver_id: Uuid,
}

// ============================================================================
// Response Types
// ============================================================================

#[derive(Serialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
}

impl<T: Serialize> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
        }
    }

    pub fn error(msg: impl Into<String>) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(msg.into()),
        }
    }
}

// ============================================================================
// Error Handling
// ============================================================================

impl IntoResponse for HrError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match self {
            HrError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            HrError::Validation(msg) => (StatusCode::BAD_REQUEST, msg),
            HrError::Database(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        };

        let body = Json(ApiResponse::<()>::error(message));
        (status, body).into_response()
    }
}

// ============================================================================
// Employee Endpoints
// ============================================================================

/// Create a new employee
async fn create_employee(
    State(state): State<HrState>,
    claims: Claims,
    Json(req): Json<CreateEmployeeRequest>,
) -> Result<impl IntoResponse, HrError> {
    let tenant_id = get_tenant_id(&claims)?;
    let employee = state.employee_service.create_employee(tenant_id, req).await?;
    Ok(Json(ApiResponse::success(employee)))
}

/// Get an employee by ID
async fn get_employee(
    State(state): State<HrState>,
    Path(id): Path<Uuid>,
) -> Result<impl IntoResponse, HrError> {
    let employee = state.employee_service.get_employee(id).await?;
    Ok(Json(ApiResponse::success(employee)))
}

/// List employees
async fn list_employees(
    State(state): State<HrState>,
    claims: Claims,
    Query(query): Query<EmployeeQuery>,
) -> Result<impl IntoResponse, HrError> {
    let tenant_id = get_tenant_id(&claims)?;
    let employees = state
        .employee_service
        .list_employees(
            tenant_id,
            query.department_id,
            query.status.as_deref(),
            query.search.as_deref(),
            query.limit.unwrap_or(50),
            query.offset.unwrap_or(0),
        )
        .await?;
    Ok(Json(ApiResponse::success(employees)))
}

/// Get direct reports for a manager
async fn get_direct_reports(
    State(state): State<HrState>,
    Path(manager_id): Path<Uuid>,
) -> Result<impl IntoResponse, HrError> {
    let reports = state.employee_service.get_direct_reports(manager_id).await?;
    Ok(Json(ApiResponse::success(reports)))
}

/// Update employee status
async fn update_employee_status(
    State(state): State<HrState>,
    Path(id): Path<Uuid>,
    Json(req): Json<UpdateEmployeeStatus>,
) -> Result<impl IntoResponse, HrError> {
    let employee = state
        .employee_service
        .update_status(id, &req.status, req.end_date)
        .await?;
    Ok(Json(ApiResponse::success(employee)))
}

/// Update employee leave balance
async fn update_employee_leave_balance(
    State(state): State<HrState>,
    Path(id): Path<Uuid>,
    Json(req): Json<UpdateLeaveBalance>,
) -> Result<impl IntoResponse, HrError> {
    let employee = state
        .employee_service
        .update_leave_balance(id, req.annual_leave, req.sick_leave)
        .await?;
    Ok(Json(ApiResponse::success(employee)))
}

/// Get headcount statistics
async fn get_headcount_stats(
    State(state): State<HrState>,
    claims: Claims,
) -> Result<impl IntoResponse, HrError> {
    let tenant_id = get_tenant_id(&claims)?;
    let stats = state.employee_service.get_headcount_stats(tenant_id).await?;
    Ok(Json(ApiResponse::success(stats)))
}

// ============================================================================
// Department Endpoints
// ============================================================================

/// Create a new department
async fn create_department(
    State(state): State<HrState>,
    claims: Claims,
    Json(req): Json<CreateDepartmentRequest>,
) -> Result<impl IntoResponse, HrError> {
    let tenant_id = get_tenant_id(&claims)?;
    let department = state
        .department_service
        .create_department(tenant_id, req)
        .await?;
    Ok(Json(ApiResponse::success(department)))
}

/// Get a department by ID
async fn get_department(
    State(state): State<HrState>,
    Path(id): Path<Uuid>,
) -> Result<impl IntoResponse, HrError> {
    let department = state.department_service.get_department(id).await?;
    Ok(Json(ApiResponse::success(department)))
}

/// List departments
async fn list_departments(
    State(state): State<HrState>,
    claims: Claims,
) -> Result<impl IntoResponse, HrError> {
    let tenant_id = get_tenant_id(&claims)?;
    let departments = state.department_service.list_departments(tenant_id).await?;
    Ok(Json(ApiResponse::success(departments)))
}

/// List departments with employee counts
async fn list_departments_with_counts(
    State(state): State<HrState>,
    claims: Claims,
) -> Result<impl IntoResponse, HrError> {
    let tenant_id = get_tenant_id(&claims)?;
    let departments = state.department_service.list_with_counts(tenant_id).await?;
    Ok(Json(ApiResponse::success(departments)))
}

/// Get child departments
async fn get_child_departments(
    State(state): State<HrState>,
    Path(parent_id): Path<Uuid>,
) -> Result<impl IntoResponse, HrError> {
    let children = state.department_service.get_children(parent_id).await?;
    Ok(Json(ApiResponse::success(children)))
}

/// Update a department
async fn update_department(
    State(state): State<HrState>,
    Path(id): Path<Uuid>,
    Json(req): Json<UpdateDepartment>,
) -> Result<impl IntoResponse, HrError> {
    let department = state
        .department_service
        .update_department(id, req.name.as_deref(), req.description.as_deref(), req.manager_id)
        .await?;
    Ok(Json(ApiResponse::success(department)))
}

/// Get organization chart
async fn get_org_chart(
    State(state): State<HrState>,
    claims: Claims,
) -> Result<impl IntoResponse, HrError> {
    let tenant_id = get_tenant_id(&claims)?;
    let chart = state.department_service.get_org_chart(tenant_id).await?;
    Ok(Json(ApiResponse::success(chart)))
}

// ============================================================================
// Leave Endpoints
// ============================================================================

/// Create a leave request
async fn create_leave_request(
    State(state): State<HrState>,
    claims: Claims,
    Json(req): Json<CreateLeaveRequest>,
) -> Result<impl IntoResponse, HrError> {
    let tenant_id = get_tenant_id(&claims)?;
    let request = state.leave_service.create_leave_request(tenant_id, req).await?;
    Ok(Json(ApiResponse::success(request)))
}

/// Get a leave request by ID
async fn get_leave_request(
    State(state): State<HrState>,
    Path(id): Path<Uuid>,
) -> Result<impl IntoResponse, HrError> {
    let request = state.leave_service.get_leave_request(id).await?;
    Ok(Json(ApiResponse::success(request)))
}

/// List leave requests for an employee
async fn list_employee_leave_requests(
    State(state): State<HrState>,
    Path(employee_id): Path<Uuid>,
    Query(query): Query<LeaveQuery>,
) -> Result<impl IntoResponse, HrError> {
    let requests = state
        .leave_service
        .list_for_employee(employee_id, query.year)
        .await?;
    Ok(Json(ApiResponse::success(requests)))
}

/// List pending leave requests for a manager
async fn list_pending_leave_requests(
    State(state): State<HrState>,
    claims: Claims,
    Path(manager_id): Path<Uuid>,
) -> Result<impl IntoResponse, HrError> {
    let tenant_id = get_tenant_id(&claims)?;
    let requests = state
        .leave_service
        .list_pending_for_manager(tenant_id, manager_id)
        .await?;
    Ok(Json(ApiResponse::success(requests)))
}

/// Approve a leave request
async fn approve_leave_request(
    State(state): State<HrState>,
    Path(id): Path<Uuid>,
    Json(req): Json<LeaveApproval>,
) -> Result<impl IntoResponse, HrError> {
    let request = state.leave_service.approve(id, req.approver_id).await?;
    Ok(Json(ApiResponse::success(request)))
}

/// Reject a leave request
async fn reject_leave_request(
    State(state): State<HrState>,
    Path(id): Path<Uuid>,
    Json(req): Json<LeaveRejection>,
) -> Result<impl IntoResponse, HrError> {
    let request = state
        .leave_service
        .reject(id, req.approver_id, &req.reason)
        .await?;
    Ok(Json(ApiResponse::success(request)))
}

/// Cancel a leave request
async fn cancel_leave_request(
    State(state): State<HrState>,
    Path(id): Path<Uuid>,
) -> Result<impl IntoResponse, HrError> {
    let request = state.leave_service.cancel(id).await?;
    Ok(Json(ApiResponse::success(request)))
}

/// Get leave balance for an employee
async fn get_leave_balance(
    State(state): State<HrState>,
    Path(employee_id): Path<Uuid>,
) -> Result<impl IntoResponse, HrError> {
    let balance = state.leave_service.get_balance(employee_id).await?;
    Ok(Json(ApiResponse::success(balance)))
}

/// Get who's out on a date
async fn get_whos_out(
    State(state): State<HrState>,
    claims: Claims,
    Query(query): Query<DateQuery>,
) -> Result<impl IntoResponse, HrError> {
    let tenant_id = get_tenant_id(&claims)?;
    let date = query.date.unwrap_or_else(|| chrono::Utc::now().date_naive());
    let out = state.leave_service.get_whos_out(tenant_id, date).await?;
    Ok(Json(ApiResponse::success(out)))
}

// ============================================================================
// Payroll Endpoints
// ============================================================================

/// Create a pay run
async fn create_pay_run(
    State(state): State<HrState>,
    claims: Claims,
    Json(req): Json<CreatePayRunRequest>,
) -> Result<impl IntoResponse, HrError> {
    let tenant_id = get_tenant_id(&claims)?;
    let run = state.payroll_service.create_pay_run(tenant_id, req).await?;
    Ok(Json(ApiResponse::success(run)))
}

/// Get a pay run by ID
async fn get_pay_run(
    State(state): State<HrState>,
    Path(id): Path<Uuid>,
) -> Result<impl IntoResponse, HrError> {
    let run = state.payroll_service.get_pay_run(id).await?;
    Ok(Json(ApiResponse::success(run)))
}

/// List pay runs
async fn list_pay_runs(
    State(state): State<HrState>,
    claims: Claims,
    Query(query): Query<PayRunQuery>,
) -> Result<impl IntoResponse, HrError> {
    let tenant_id = get_tenant_id(&claims)?;
    let runs = state.payroll_service.list_pay_runs(tenant_id, query.year).await?;
    Ok(Json(ApiResponse::success(runs)))
}

/// Generate pay stubs for a pay run
async fn generate_pay_stubs(
    State(state): State<HrState>,
    Path(pay_run_id): Path<Uuid>,
) -> Result<impl IntoResponse, HrError> {
    let stubs = state.payroll_service.generate_pay_stubs(pay_run_id).await?;
    Ok(Json(ApiResponse::success(stubs)))
}

/// Get pay stubs for a pay run
async fn get_pay_run_stubs(
    State(state): State<HrState>,
    Path(pay_run_id): Path<Uuid>,
) -> Result<impl IntoResponse, HrError> {
    let stubs = state.payroll_service.get_stubs_for_run(pay_run_id).await?;
    Ok(Json(ApiResponse::success(stubs)))
}

/// Get pay stubs for an employee
async fn get_employee_pay_stubs(
    State(state): State<HrState>,
    Path(employee_id): Path<Uuid>,
    Query(query): Query<LeaveQuery>,
) -> Result<impl IntoResponse, HrError> {
    let stubs = state
        .payroll_service
        .get_stubs_for_employee(employee_id, query.year)
        .await?;
    Ok(Json(ApiResponse::success(stubs)))
}

/// Approve a pay run
async fn approve_pay_run(
    State(state): State<HrState>,
    Path(id): Path<Uuid>,
    Json(req): Json<PayRunApproval>,
) -> Result<impl IntoResponse, HrError> {
    let run = state.payroll_service.approve_pay_run(id, req.approver_id).await?;
    Ok(Json(ApiResponse::success(run)))
}

/// Process a pay run
async fn process_pay_run(
    State(state): State<HrState>,
    Path(id): Path<Uuid>,
) -> Result<impl IntoResponse, HrError> {
    let run = state.payroll_service.process_pay_run(id).await?;
    Ok(Json(ApiResponse::success(run)))
}

/// Get payroll statistics
async fn get_payroll_stats(
    State(state): State<HrState>,
    claims: Claims,
) -> Result<impl IntoResponse, HrError> {
    let tenant_id = get_tenant_id(&claims)?;
    let stats = state.payroll_service.get_stats(tenant_id).await?;
    Ok(Json(ApiResponse::success(stats)))
}

// ============================================================================
// Router
// ============================================================================

/// Create HR router
pub fn hr_router(pool: PgPool) -> Router {
    let state = HrState::new(pool);

    Router::new()
        // Employee routes
        .route("/employees", post(create_employee))
        .route("/employees", get(list_employees))
        .route("/employees/stats", get(get_headcount_stats))
        .route("/employees/:id", get(get_employee))
        .route("/employees/:id/status", put(update_employee_status))
        .route("/employees/:id/leave-balance", put(update_employee_leave_balance))
        .route("/employees/:id/direct-reports", get(get_direct_reports))
        .route("/employees/:id/pay-stubs", get(get_employee_pay_stubs))
        .route("/employees/:id/leave-requests", get(list_employee_leave_requests))
        // Department routes
        .route("/departments", post(create_department))
        .route("/departments", get(list_departments))
        .route("/departments/with-counts", get(list_departments_with_counts))
        .route("/departments/org-chart", get(get_org_chart))
        .route("/departments/:id", get(get_department))
        .route("/departments/:id", put(update_department))
        .route("/departments/:id/children", get(get_child_departments))
        // Leave routes
        .route("/leave-requests", post(create_leave_request))
        .route("/leave-requests/whos-out", get(get_whos_out))
        .route("/leave-requests/:id", get(get_leave_request))
        .route("/leave-requests/:id/approve", post(approve_leave_request))
        .route("/leave-requests/:id/reject", post(reject_leave_request))
        .route("/leave-requests/:id/cancel", post(cancel_leave_request))
        .route("/leave-balance/:employee_id", get(get_leave_balance))
        .route("/managers/:manager_id/pending-leave", get(list_pending_leave_requests))
        // Payroll routes
        .route("/pay-runs", post(create_pay_run))
        .route("/pay-runs", get(list_pay_runs))
        .route("/pay-runs/stats", get(get_payroll_stats))
        .route("/pay-runs/:id", get(get_pay_run))
        .route("/pay-runs/:id/generate-stubs", post(generate_pay_stubs))
        .route("/pay-runs/:id/stubs", get(get_pay_run_stubs))
        .route("/pay-runs/:id/approve", post(approve_pay_run))
        .route("/pay-runs/:id/process", post(process_pay_run))
        .with_state(state)
}
