// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Leave/Time-Off Management
//!
//! Track employee time-off requests and balances.

use chrono::{DateTime, NaiveDate, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::{FromRow, PgPool};
use uuid::Uuid;

use super::HrError;

// ============================================================================
// Types
// ============================================================================

/// A leave request
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct LeaveRequest {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub employee_id: Uuid,
    pub leave_type: String,  // ANNUAL, SICK, PERSONAL, UNPAID, PARENTAL, BEREAVEMENT
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
    pub total_days: Decimal,
    pub reason: Option<String>,
    pub status: String,  // PENDING, APPROVED, REJECTED, CANCELLED
    pub approved_by: Option<Uuid>,
    pub approved_at: Option<DateTime<Utc>>,
    pub rejection_reason: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Leave type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LeaveType {
    Annual,
    Sick,
    Personal,
    Unpaid,
    Parental,
    Bereavement,
}

impl LeaveType {
    pub fn as_str(&self) -> &'static str {
        match self {
            LeaveType::Annual => "ANNUAL",
            LeaveType::Sick => "SICK",
            LeaveType::Personal => "PERSONAL",
            LeaveType::Unpaid => "UNPAID",
            LeaveType::Parental => "PARENTAL",
            LeaveType::Bereavement => "BEREAVEMENT",
        }
    }
}

/// Create leave request
#[derive(Debug, Deserialize)]
pub struct CreateLeaveRequest {
    pub employee_id: Uuid,
    pub leave_type: String,
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
    pub reason: Option<String>,
}

/// Leave balance
#[derive(Debug, Serialize)]
pub struct LeaveBalance {
    pub employee_id: Uuid,
    pub annual_entitled: Decimal,
    pub annual_taken: Decimal,
    pub annual_remaining: Decimal,
    pub sick_entitled: Decimal,
    pub sick_taken: Decimal,
    pub sick_remaining: Decimal,
}

// ============================================================================
// Service
// ============================================================================

/// Leave Management Service
#[derive(Clone)]
pub struct LeaveService {
    pool: PgPool,
}

impl LeaveService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Calculate days between dates (excluding weekends - simplified)
    fn calculate_days(&self, start: NaiveDate, end: NaiveDate) -> Decimal {
        let days = (end - start).num_days() + 1;
        Decimal::from(days)
    }

    /// Create a new leave request
    pub async fn create_leave_request(
        &self,
        tenant_id: Uuid,
        req: CreateLeaveRequest,
    ) -> Result<LeaveRequest, HrError> {
        let id = Uuid::new_v4();
        let total_days = self.calculate_days(req.start_date, req.end_date);

        sqlx::query(
            "INSERT INTO hr_leave_requests (
                id, tenant_id, employee_id, leave_type, start_date, end_date,
                total_days, reason
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
        )
        .bind(id)
        .bind(tenant_id)
        .bind(req.employee_id)
        .bind(&req.leave_type)
        .bind(req.start_date)
        .bind(req.end_date)
        .bind(total_days)
        .bind(&req.reason)
        .execute(&self.pool)
        .await
        .map_err(HrError::Database)?;

        self.get_leave_request(id).await
    }

    /// Get a leave request by ID
    pub async fn get_leave_request(&self, id: Uuid) -> Result<LeaveRequest, HrError> {
        sqlx::query_as::<_, LeaveRequest>("SELECT * FROM hr_leave_requests WHERE id = $1")
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(HrError::Database)?
            .ok_or_else(|| HrError::NotFound("Leave request not found".into()))
    }

    /// List leave requests for an employee
    pub async fn list_for_employee(
        &self,
        employee_id: Uuid,
        year: Option<i32>,
    ) -> Result<Vec<LeaveRequest>, HrError> {
        let requests = if let Some(y) = year {
            sqlx::query_as::<_, LeaveRequest>(
                "SELECT * FROM hr_leave_requests
                 WHERE employee_id = $1
                   AND EXTRACT(YEAR FROM start_date) = $2
                 ORDER BY start_date DESC",
            )
            .bind(employee_id)
            .bind(y)
            .fetch_all(&self.pool)
            .await
        } else {
            sqlx::query_as::<_, LeaveRequest>(
                "SELECT * FROM hr_leave_requests
                 WHERE employee_id = $1
                 ORDER BY start_date DESC",
            )
            .bind(employee_id)
            .fetch_all(&self.pool)
            .await
        }
        .map_err(HrError::Database)?;

        Ok(requests)
    }

    /// List pending requests for approval
    pub async fn list_pending_for_manager(
        &self,
        tenant_id: Uuid,
        manager_id: Uuid,
    ) -> Result<Vec<LeaveRequest>, HrError> {
        let requests = sqlx::query_as::<_, LeaveRequest>(
            "SELECT lr.* FROM hr_leave_requests lr
             JOIN hr_employees e ON lr.employee_id = e.id
             WHERE lr.tenant_id = $1
               AND e.manager_id = $2
               AND lr.status = 'PENDING'
             ORDER BY lr.start_date",
        )
        .bind(tenant_id)
        .bind(manager_id)
        .fetch_all(&self.pool)
        .await
        .map_err(HrError::Database)?;

        Ok(requests)
    }

    /// Approve a leave request
    pub async fn approve(
        &self,
        id: Uuid,
        approver_id: Uuid,
    ) -> Result<LeaveRequest, HrError> {
        let request = self.get_leave_request(id).await?;

        // Update request status
        sqlx::query(
            "UPDATE hr_leave_requests SET
                status = 'APPROVED',
                approved_by = $2,
                approved_at = NOW(),
                updated_at = NOW()
             WHERE id = $1",
        )
        .bind(id)
        .bind(approver_id)
        .execute(&self.pool)
        .await
        .map_err(HrError::Database)?;

        // Deduct from balance — use separate static queries to avoid dynamic SQL
        match request.leave_type.as_str() {
            "SICK" => {
                sqlx::query(
                    "UPDATE hr_employees SET sick_leave_balance = sick_leave_balance - $2, updated_at = NOW() WHERE id = $1",
                )
                .bind(request.employee_id)
                .bind(request.total_days)
                .execute(&self.pool)
                .await
                .map_err(HrError::Database)?;
            }
            _ => {
                sqlx::query(
                    "UPDATE hr_employees SET annual_leave_balance = annual_leave_balance - $2, updated_at = NOW() WHERE id = $1",
                )
                .bind(request.employee_id)
                .bind(request.total_days)
                .execute(&self.pool)
                .await
                .map_err(HrError::Database)?;
            }
        }

        self.get_leave_request(id).await
    }

    /// Reject a leave request
    pub async fn reject(
        &self,
        id: Uuid,
        approver_id: Uuid,
        reason: &str,
    ) -> Result<LeaveRequest, HrError> {
        sqlx::query(
            "UPDATE hr_leave_requests SET
                status = 'REJECTED',
                approved_by = $2,
                approved_at = NOW(),
                rejection_reason = $3,
                updated_at = NOW()
             WHERE id = $1",
        )
        .bind(id)
        .bind(approver_id)
        .bind(reason)
        .execute(&self.pool)
        .await
        .map_err(HrError::Database)?;

        self.get_leave_request(id).await
    }

    /// Cancel a leave request
    pub async fn cancel(&self, id: Uuid) -> Result<LeaveRequest, HrError> {
        let request = self.get_leave_request(id).await?;

        // If already approved, restore balance — use separate static queries to avoid dynamic SQL
        if request.status == "APPROVED" {
            match request.leave_type.as_str() {
                "SICK" => {
                    sqlx::query(
                        "UPDATE hr_employees SET sick_leave_balance = sick_leave_balance + $2, updated_at = NOW() WHERE id = $1",
                    )
                    .bind(request.employee_id)
                    .bind(request.total_days)
                    .execute(&self.pool)
                    .await
                    .map_err(HrError::Database)?;
                }
                _ => {
                    sqlx::query(
                        "UPDATE hr_employees SET annual_leave_balance = annual_leave_balance + $2, updated_at = NOW() WHERE id = $1",
                    )
                    .bind(request.employee_id)
                    .bind(request.total_days)
                    .execute(&self.pool)
                    .await
                    .map_err(HrError::Database)?;
                }
            }
        }

        sqlx::query(
            "UPDATE hr_leave_requests SET status = 'CANCELLED', updated_at = NOW() WHERE id = $1",
        )
        .bind(id)
        .execute(&self.pool)
        .await
        .map_err(HrError::Database)?;

        self.get_leave_request(id).await
    }

    /// Get leave balance for an employee
    pub async fn get_balance(&self, employee_id: Uuid) -> Result<LeaveBalance, HrError> {
        let (annual_balance, sick_balance): (Decimal, Decimal) = sqlx::query_as(
            "SELECT annual_leave_balance, sick_leave_balance
             FROM hr_employees WHERE id = $1",
        )
        .bind(employee_id)
        .fetch_one(&self.pool)
        .await
        .map_err(HrError::Database)?;

        // Default entitlements (would be configurable per policy)
        let annual_entitled = Decimal::from(20);
        let sick_entitled = Decimal::from(10);

        Ok(LeaveBalance {
            employee_id,
            annual_entitled,
            annual_taken: annual_entitled - annual_balance,
            annual_remaining: annual_balance,
            sick_entitled,
            sick_taken: sick_entitled - sick_balance,
            sick_remaining: sick_balance,
        })
    }

    /// Get who's out on a specific date
    pub async fn get_whos_out(
        &self,
        tenant_id: Uuid,
        date: NaiveDate,
    ) -> Result<Vec<LeaveRequest>, HrError> {
        let requests = sqlx::query_as::<_, LeaveRequest>(
            "SELECT * FROM hr_leave_requests
             WHERE tenant_id = $1
               AND status = 'APPROVED'
               AND start_date <= $2
               AND end_date >= $2
             ORDER BY start_date",
        )
        .bind(tenant_id)
        .bind(date)
        .fetch_all(&self.pool)
        .await
        .map_err(HrError::Database)?;

        Ok(requests)
    }
}
