// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Payroll Management
//!
//! Basic payroll tracking and pay run management.

use chrono::{DateTime, NaiveDate, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::{FromRow, PgPool};
use uuid::Uuid;

use super::HrError;

// ============================================================================
// Types
// ============================================================================

/// A pay run (payroll batch)
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct PayRun {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub pay_period_start: NaiveDate,
    pub pay_period_end: NaiveDate,
    pub pay_date: NaiveDate,
    pub status: String,  // DRAFT, APPROVED, PROCESSING, COMPLETED, CANCELLED
    pub total_gross: Decimal,
    pub total_deductions: Decimal,
    pub total_net: Decimal,
    pub employee_count: i32,
    pub notes: Option<String>,
    pub approved_by: Option<Uuid>,
    pub approved_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// A pay stub (individual employee payment)
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct PayStub {
    pub id: Uuid,
    pub pay_run_id: Uuid,
    pub employee_id: Uuid,
    pub pay_period_start: NaiveDate,
    pub pay_period_end: NaiveDate,
    // Earnings
    pub base_salary: Decimal,
    pub overtime_hours: Decimal,
    pub overtime_pay: Decimal,
    pub bonus: Decimal,
    pub commission: Decimal,
    pub other_earnings: Decimal,
    pub gross_pay: Decimal,
    // Deductions
    pub tax_federal: Decimal,
    pub tax_state: Decimal,
    pub tax_local: Decimal,
    pub social_security: Decimal,
    pub medicare: Decimal,
    pub health_insurance: Decimal,
    pub retirement_401k: Decimal,
    pub other_deductions: Decimal,
    pub total_deductions: Decimal,
    // Net
    pub net_pay: Decimal,
    pub payment_method: String,  // DIRECT_DEPOSIT, CHECK
    pub created_at: DateTime<Utc>,
}

/// Create pay run request
#[derive(Debug, Deserialize)]
pub struct CreatePayRunRequest {
    pub pay_period_start: NaiveDate,
    pub pay_period_end: NaiveDate,
    pub pay_date: NaiveDate,
    pub notes: Option<String>,
}

/// Payroll summary statistics
#[derive(Debug, Serialize)]
pub struct PayrollStats {
    pub total_payroll_ytd: Decimal,
    pub average_salary: Decimal,
    pub total_employees: i64,
    pub last_pay_run_date: Option<NaiveDate>,
    pub next_pay_run_date: Option<NaiveDate>,
}

// ============================================================================
// Service
// ============================================================================

/// Payroll Management Service
#[derive(Clone)]
pub struct PayrollService {
    pool: PgPool,
}

impl PayrollService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Create a new pay run
    pub async fn create_pay_run(
        &self,
        tenant_id: Uuid,
        req: CreatePayRunRequest,
    ) -> Result<PayRun, HrError> {
        let id = Uuid::new_v4();

        sqlx::query(
            "INSERT INTO hr_pay_runs (
                id, tenant_id, pay_period_start, pay_period_end, pay_date, notes
            ) VALUES ($1, $2, $3, $4, $5, $6)",
        )
        .bind(id)
        .bind(tenant_id)
        .bind(req.pay_period_start)
        .bind(req.pay_period_end)
        .bind(req.pay_date)
        .bind(&req.notes)
        .execute(&self.pool)
        .await
        .map_err(HrError::Database)?;

        self.get_pay_run(id).await
    }

    /// Get a pay run by ID
    pub async fn get_pay_run(&self, id: Uuid) -> Result<PayRun, HrError> {
        sqlx::query_as::<_, PayRun>("SELECT * FROM hr_pay_runs WHERE id = $1")
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(HrError::Database)?
            .ok_or_else(|| HrError::NotFound("Pay run not found".into()))
    }

    /// List pay runs for a tenant
    pub async fn list_pay_runs(
        &self,
        tenant_id: Uuid,
        year: Option<i32>,
    ) -> Result<Vec<PayRun>, HrError> {
        let runs = if let Some(y) = year {
            sqlx::query_as::<_, PayRun>(
                "SELECT * FROM hr_pay_runs
                 WHERE tenant_id = $1 AND EXTRACT(YEAR FROM pay_date) = $2
                 ORDER BY pay_date DESC",
            )
            .bind(tenant_id)
            .bind(y)
            .fetch_all(&self.pool)
            .await
        } else {
            sqlx::query_as::<_, PayRun>(
                "SELECT * FROM hr_pay_runs
                 WHERE tenant_id = $1
                 ORDER BY pay_date DESC
                 LIMIT 24",
            )
            .bind(tenant_id)
            .fetch_all(&self.pool)
            .await
        }
        .map_err(HrError::Database)?;

        Ok(runs)
    }

    /// Generate pay stubs for a pay run
    pub async fn generate_pay_stubs(&self, pay_run_id: Uuid) -> Result<Vec<PayStub>, HrError> {
        let pay_run = self.get_pay_run(pay_run_id).await?;

        // Get all active employees for the tenant
        let employees: Vec<(Uuid, Decimal, String)> = sqlx::query_as(
            "SELECT id, COALESCE(base_salary, 0), salary_type
             FROM hr_employees
             WHERE tenant_id = $1 AND is_active = true AND employment_status = 'ACTIVE'",
        )
        .bind(pay_run.tenant_id)
        .fetch_all(&self.pool)
        .await
        .map_err(HrError::Database)?;

        let mut stubs = Vec::new();

        for (employee_id, annual_salary, salary_type) in employees {
            // Calculate pay for the period (simplified - monthly)
            let base_pay = if salary_type == "HOURLY" {
                // Assume 160 hours/month for hourly
                annual_salary * Decimal::from(160)
            } else {
                // Monthly salary
                annual_salary / Decimal::from(12)
            };

            // Simplified tax calculations (US federal approximations)
            let tax_federal = base_pay * Decimal::new(22, 2); // 22%
            let tax_state = base_pay * Decimal::new(5, 2);    // 5%
            let social_security = base_pay * Decimal::new(62, 3); // 6.2%
            let medicare = base_pay * Decimal::new(145, 4);   // 1.45%
            let health = Decimal::from(200);                   // Flat $200
            let retirement = base_pay * Decimal::new(6, 2);   // 6% 401k

            let total_deductions = tax_federal + tax_state + social_security
                + medicare + health + retirement;
            let net_pay = base_pay - total_deductions;

            let stub_id = Uuid::new_v4();

            sqlx::query(
                "INSERT INTO hr_pay_stubs (
                    id, pay_run_id, employee_id, pay_period_start, pay_period_end,
                    base_salary, gross_pay, tax_federal, tax_state, social_security,
                    medicare, health_insurance, retirement_401k, total_deductions, net_pay
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)",
            )
            .bind(stub_id)
            .bind(pay_run_id)
            .bind(employee_id)
            .bind(pay_run.pay_period_start)
            .bind(pay_run.pay_period_end)
            .bind(base_pay)
            .bind(base_pay)
            .bind(tax_federal)
            .bind(tax_state)
            .bind(social_security)
            .bind(medicare)
            .bind(health)
            .bind(retirement)
            .bind(total_deductions)
            .bind(net_pay)
            .execute(&self.pool)
            .await
            .map_err(HrError::Database)?;

            stubs.push(self.get_pay_stub(stub_id).await?);
        }

        // Update pay run totals
        let (total_gross, total_deductions, total_net, count): (Decimal, Decimal, Decimal, i64) =
            sqlx::query_as(
                "SELECT COALESCE(SUM(gross_pay), 0), COALESCE(SUM(total_deductions), 0),
                        COALESCE(SUM(net_pay), 0), COUNT(*)
                 FROM hr_pay_stubs WHERE pay_run_id = $1",
            )
            .bind(pay_run_id)
            .fetch_one(&self.pool)
            .await
            .map_err(HrError::Database)?;

        sqlx::query(
            "UPDATE hr_pay_runs SET
                total_gross = $2, total_deductions = $3, total_net = $4,
                employee_count = $5, updated_at = NOW()
             WHERE id = $1",
        )
        .bind(pay_run_id)
        .bind(total_gross)
        .bind(total_deductions)
        .bind(total_net)
        .bind(count as i32)
        .execute(&self.pool)
        .await
        .map_err(HrError::Database)?;

        Ok(stubs)
    }

    /// Get a pay stub by ID
    pub async fn get_pay_stub(&self, id: Uuid) -> Result<PayStub, HrError> {
        sqlx::query_as::<_, PayStub>("SELECT * FROM hr_pay_stubs WHERE id = $1")
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(HrError::Database)?
            .ok_or_else(|| HrError::NotFound("Pay stub not found".into()))
    }

    /// Get pay stubs for a pay run
    pub async fn get_stubs_for_run(&self, pay_run_id: Uuid) -> Result<Vec<PayStub>, HrError> {
        let stubs = sqlx::query_as::<_, PayStub>(
            "SELECT * FROM hr_pay_stubs WHERE pay_run_id = $1 ORDER BY employee_id",
        )
        .bind(pay_run_id)
        .fetch_all(&self.pool)
        .await
        .map_err(HrError::Database)?;

        Ok(stubs)
    }

    /// Get pay stubs for an employee
    pub async fn get_stubs_for_employee(
        &self,
        employee_id: Uuid,
        year: Option<i32>,
    ) -> Result<Vec<PayStub>, HrError> {
        let stubs = if let Some(y) = year {
            sqlx::query_as::<_, PayStub>(
                "SELECT * FROM hr_pay_stubs
                 WHERE employee_id = $1 AND EXTRACT(YEAR FROM pay_period_start) = $2
                 ORDER BY pay_period_start DESC",
            )
            .bind(employee_id)
            .bind(y)
            .fetch_all(&self.pool)
            .await
        } else {
            sqlx::query_as::<_, PayStub>(
                "SELECT * FROM hr_pay_stubs
                 WHERE employee_id = $1
                 ORDER BY pay_period_start DESC
                 LIMIT 12",
            )
            .bind(employee_id)
            .fetch_all(&self.pool)
            .await
        }
        .map_err(HrError::Database)?;

        Ok(stubs)
    }

    /// Approve a pay run
    pub async fn approve_pay_run(
        &self,
        id: Uuid,
        approver_id: Uuid,
    ) -> Result<PayRun, HrError> {
        sqlx::query(
            "UPDATE hr_pay_runs SET
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

        self.get_pay_run(id).await
    }

    /// Process a pay run (mark as completed)
    pub async fn process_pay_run(&self, id: Uuid) -> Result<PayRun, HrError> {
        let pay_run = self.get_pay_run(id).await?;

        if pay_run.status != "APPROVED" {
            return Err(HrError::Validation(
                "Pay run must be approved before processing".into(),
            ));
        }

        sqlx::query(
            "UPDATE hr_pay_runs SET status = 'COMPLETED', updated_at = NOW() WHERE id = $1",
        )
        .bind(id)
        .execute(&self.pool)
        .await
        .map_err(HrError::Database)?;

        self.get_pay_run(id).await
    }

    /// Get payroll statistics
    pub async fn get_stats(&self, tenant_id: Uuid) -> Result<PayrollStats, HrError> {
        let (total_ytd,): (Decimal,) = sqlx::query_as(
            "SELECT COALESCE(SUM(total_gross), 0)
             FROM hr_pay_runs
             WHERE tenant_id = $1
               AND status = 'COMPLETED'
               AND EXTRACT(YEAR FROM pay_date) = EXTRACT(YEAR FROM CURRENT_DATE)",
        )
        .bind(tenant_id)
        .fetch_one(&self.pool)
        .await
        .map_err(HrError::Database)?;

        let (avg_salary, emp_count): (Option<Decimal>, i64) = sqlx::query_as(
            "SELECT AVG(base_salary), COUNT(*)
             FROM hr_employees
             WHERE tenant_id = $1 AND is_active = true",
        )
        .bind(tenant_id)
        .fetch_one(&self.pool)
        .await
        .map_err(HrError::Database)?;

        let last_run: Option<NaiveDate> = sqlx::query_scalar(
            "SELECT MAX(pay_date) FROM hr_pay_runs
             WHERE tenant_id = $1 AND status = 'COMPLETED'",
        )
        .bind(tenant_id)
        .fetch_one(&self.pool)
        .await
        .map_err(HrError::Database)?;

        Ok(PayrollStats {
            total_payroll_ytd: total_ytd,
            average_salary: avg_salary.unwrap_or_default(),
            total_employees: emp_count,
            last_pay_run_date: last_run,
            next_pay_run_date: None, // Would calculate based on pay schedule
        })
    }
}
