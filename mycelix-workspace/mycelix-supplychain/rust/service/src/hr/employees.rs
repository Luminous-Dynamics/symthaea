// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Employee Management
//!
//! Core employee records, personal information, and employment details.

use chrono::{DateTime, NaiveDate, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::{FromRow, PgPool};
use uuid::Uuid;

// ============================================================================
// Types
// ============================================================================

/// An employee record
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Employee {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub employee_number: String,
    pub user_id: Option<Uuid>,  // Link to auth user
    // Personal info
    pub first_name: String,
    pub last_name: String,
    pub preferred_name: Option<String>,
    pub email: String,
    pub personal_email: Option<String>,
    pub phone: Option<String>,
    pub mobile: Option<String>,
    pub date_of_birth: Option<NaiveDate>,
    pub gender: Option<String>,
    // Address
    pub address_line1: Option<String>,
    pub address_line2: Option<String>,
    pub city: Option<String>,
    pub state: Option<String>,
    pub postal_code: Option<String>,
    pub country: Option<String>,
    // Emergency contact
    pub emergency_contact_name: Option<String>,
    pub emergency_contact_phone: Option<String>,
    pub emergency_contact_relation: Option<String>,
    // Employment
    pub department_id: Option<Uuid>,
    pub job_title: Option<String>,
    pub manager_id: Option<Uuid>,
    pub employment_type: String,  // FULL_TIME, PART_TIME, CONTRACTOR, INTERN
    pub employment_status: String,  // ACTIVE, ON_LEAVE, TERMINATED, PENDING
    pub start_date: NaiveDate,
    pub end_date: Option<NaiveDate>,
    pub probation_end_date: Option<NaiveDate>,
    // Compensation
    pub salary_type: String,  // HOURLY, SALARY, COMMISSION
    pub base_salary: Option<Decimal>,
    pub salary_currency: String,
    pub pay_frequency: String,  // WEEKLY, BIWEEKLY, MONTHLY
    // Time off
    pub annual_leave_balance: Decimal,
    pub sick_leave_balance: Decimal,
    // Metadata
    pub tags: Vec<String>,
    pub custom_fields: Option<serde_json::Value>,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Employment type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmploymentType {
    FullTime,
    PartTime,
    Contractor,
    Intern,
}

impl EmploymentType {
    pub fn as_str(&self) -> &'static str {
        match self {
            EmploymentType::FullTime => "FULL_TIME",
            EmploymentType::PartTime => "PART_TIME",
            EmploymentType::Contractor => "CONTRACTOR",
            EmploymentType::Intern => "INTERN",
        }
    }
}

/// Create employee request
#[derive(Debug, Deserialize)]
pub struct CreateEmployeeRequest {
    pub first_name: String,
    pub last_name: String,
    pub email: String,
    pub phone: Option<String>,
    pub department_id: Option<Uuid>,
    pub job_title: Option<String>,
    pub manager_id: Option<Uuid>,
    pub employment_type: String,
    pub start_date: NaiveDate,
    pub salary_type: Option<String>,
    pub base_salary: Option<Decimal>,
    pub salary_currency: Option<String>,
}

// ============================================================================
// Service
// ============================================================================

/// Employee Management Service
#[derive(Clone)]
pub struct EmployeeService {
    pool: PgPool,
}

impl EmployeeService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Generate employee number
    async fn generate_employee_number(&self, tenant_id: Uuid) -> Result<String, HrError> {
        let count: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM hr_employees WHERE tenant_id = $1",
        )
        .bind(tenant_id)
        .fetch_one(&self.pool)
        .await
        .map_err(HrError::Database)?;

        Ok(format!("EMP{:06}", count.0 + 1))
    }

    /// Create a new employee
    pub async fn create_employee(
        &self,
        tenant_id: Uuid,
        req: CreateEmployeeRequest,
    ) -> Result<Employee, HrError> {
        let id = Uuid::new_v4();
        let employee_number = self.generate_employee_number(tenant_id).await?;
        let salary_type = req.salary_type.unwrap_or_else(|| "SALARY".to_string());
        let salary_currency = req.salary_currency.unwrap_or_else(|| "USD".to_string());

        sqlx::query(
            "INSERT INTO hr_employees (
                id, tenant_id, employee_number, first_name, last_name, email, phone,
                department_id, job_title, manager_id, employment_type, start_date,
                salary_type, base_salary, salary_currency
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)",
        )
        .bind(id)
        .bind(tenant_id)
        .bind(&employee_number)
        .bind(&req.first_name)
        .bind(&req.last_name)
        .bind(&req.email)
        .bind(&req.phone)
        .bind(req.department_id)
        .bind(&req.job_title)
        .bind(req.manager_id)
        .bind(&req.employment_type)
        .bind(req.start_date)
        .bind(&salary_type)
        .bind(req.base_salary)
        .bind(&salary_currency)
        .execute(&self.pool)
        .await
        .map_err(HrError::Database)?;

        self.get_employee(id).await
    }

    /// Get an employee by ID
    pub async fn get_employee(&self, id: Uuid) -> Result<Employee, HrError> {
        sqlx::query_as::<_, Employee>("SELECT * FROM hr_employees WHERE id = $1")
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(HrError::Database)?
            .ok_or_else(|| HrError::NotFound("Employee not found".into()))
    }

    /// Get employee by user ID
    pub async fn get_by_user_id(&self, user_id: Uuid) -> Result<Employee, HrError> {
        sqlx::query_as::<_, Employee>("SELECT * FROM hr_employees WHERE user_id = $1")
            .bind(user_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(HrError::Database)?
            .ok_or_else(|| HrError::NotFound("Employee not found".into()))
    }

    /// List employees for a tenant
    pub async fn list_employees(
        &self,
        tenant_id: Uuid,
        department_id: Option<Uuid>,
        status: Option<&str>,
        search: Option<&str>,
        limit: i32,
        offset: i32,
    ) -> Result<Vec<Employee>, HrError> {
        let employees = match (department_id, status, search) {
            (Some(dept), _, _) => {
                sqlx::query_as::<_, Employee>(
                    "SELECT * FROM hr_employees
                     WHERE tenant_id = $1 AND department_id = $2 AND is_active = true
                     ORDER BY last_name, first_name LIMIT $3 OFFSET $4",
                )
                .bind(tenant_id)
                .bind(dept)
                .bind(limit)
                .bind(offset)
                .fetch_all(&self.pool)
                .await
            }
            (None, Some(s), _) => {
                sqlx::query_as::<_, Employee>(
                    "SELECT * FROM hr_employees
                     WHERE tenant_id = $1 AND employment_status = $2
                     ORDER BY last_name, first_name LIMIT $3 OFFSET $4",
                )
                .bind(tenant_id)
                .bind(s)
                .bind(limit)
                .bind(offset)
                .fetch_all(&self.pool)
                .await
            }
            (None, None, Some(q)) => {
                let pattern = format!("%{}%", q);
                sqlx::query_as::<_, Employee>(
                    "SELECT * FROM hr_employees
                     WHERE tenant_id = $1
                       AND (first_name ILIKE $2 OR last_name ILIKE $2 OR email ILIKE $2)
                     ORDER BY last_name, first_name LIMIT $3 OFFSET $4",
                )
                .bind(tenant_id)
                .bind(&pattern)
                .bind(limit)
                .bind(offset)
                .fetch_all(&self.pool)
                .await
            }
            (None, None, None) => {
                sqlx::query_as::<_, Employee>(
                    "SELECT * FROM hr_employees
                     WHERE tenant_id = $1 AND is_active = true
                     ORDER BY last_name, first_name LIMIT $2 OFFSET $3",
                )
                .bind(tenant_id)
                .bind(limit)
                .bind(offset)
                .fetch_all(&self.pool)
                .await
            }
        }
        .map_err(HrError::Database)?;

        Ok(employees)
    }

    /// Get direct reports for a manager
    pub async fn get_direct_reports(&self, manager_id: Uuid) -> Result<Vec<Employee>, HrError> {
        let employees = sqlx::query_as::<_, Employee>(
            "SELECT * FROM hr_employees
             WHERE manager_id = $1 AND is_active = true
             ORDER BY last_name, first_name",
        )
        .bind(manager_id)
        .fetch_all(&self.pool)
        .await
        .map_err(HrError::Database)?;

        Ok(employees)
    }

    /// Update employment status
    pub async fn update_status(
        &self,
        id: Uuid,
        status: &str,
        end_date: Option<NaiveDate>,
    ) -> Result<Employee, HrError> {
        let is_active = status == "ACTIVE";

        sqlx::query(
            "UPDATE hr_employees SET
                employment_status = $2, end_date = $3, is_active = $4, updated_at = NOW()
             WHERE id = $1",
        )
        .bind(id)
        .bind(status)
        .bind(end_date)
        .bind(is_active)
        .execute(&self.pool)
        .await
        .map_err(HrError::Database)?;

        self.get_employee(id).await
    }

    /// Update employee's leave balance
    pub async fn update_leave_balance(
        &self,
        id: Uuid,
        annual_leave: Option<Decimal>,
        sick_leave: Option<Decimal>,
    ) -> Result<Employee, HrError> {
        sqlx::query(
            "UPDATE hr_employees SET
                annual_leave_balance = COALESCE($2, annual_leave_balance),
                sick_leave_balance = COALESCE($3, sick_leave_balance),
                updated_at = NOW()
             WHERE id = $1",
        )
        .bind(id)
        .bind(annual_leave)
        .bind(sick_leave)
        .execute(&self.pool)
        .await
        .map_err(HrError::Database)?;

        self.get_employee(id).await
    }

    /// Get employee headcount stats
    pub async fn get_headcount_stats(&self, tenant_id: Uuid) -> Result<HeadcountStats, HrError> {
        let (total, active, on_leave, contractors): (i64, i64, i64, i64) = sqlx::query_as(
            "SELECT
                COUNT(*),
                COUNT(*) FILTER (WHERE employment_status = 'ACTIVE'),
                COUNT(*) FILTER (WHERE employment_status = 'ON_LEAVE'),
                COUNT(*) FILTER (WHERE employment_type = 'CONTRACTOR')
             FROM hr_employees WHERE tenant_id = $1",
        )
        .bind(tenant_id)
        .fetch_one(&self.pool)
        .await
        .map_err(HrError::Database)?;

        Ok(HeadcountStats {
            total,
            active,
            on_leave,
            contractors,
        })
    }
}

/// Headcount statistics
#[derive(Debug, Serialize)]
pub struct HeadcountStats {
    pub total: i64,
    pub active: i64,
    pub on_leave: i64,
    pub contractors: i64,
}

// ============================================================================
// Errors
// ============================================================================

/// HR errors
#[derive(Debug, thiserror::Error)]
pub enum HrError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Validation error: {0}")]
    Validation(String),
}
