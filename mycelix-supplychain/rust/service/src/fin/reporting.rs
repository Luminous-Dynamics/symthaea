// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Financial reporting service
//!
//! Generates standard financial reports: trial balance, income statement, balance sheet.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{FromRow, PgPool};

use super::models::AccountType;

/// Service for generating financial reports
#[derive(Clone)]
pub struct ReportingService {
    pool: PgPool,
}

/// Trial Balance Report
#[derive(Debug, Serialize, Deserialize)]
pub struct TrialBalance {
    pub as_of_date: DateTime<Utc>,
    pub accounts: Vec<TrialBalanceAccount>,
    pub total_debits: rust_decimal::Decimal,
    pub total_credits: rust_decimal::Decimal,
}

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct TrialBalanceAccount {
    pub account_number: String,
    pub account_name: String,
    pub account_type: AccountType,
    pub debit_balance: rust_decimal::Decimal,
    pub credit_balance: rust_decimal::Decimal,
}

/// Income Statement (Profit & Loss)
#[derive(Debug, Serialize, Deserialize)]
pub struct IncomeStatement {
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub revenue: Vec<ReportLine>,
    pub expenses: Vec<ReportLine>,
    pub total_revenue: rust_decimal::Decimal,
    pub total_expenses: rust_decimal::Decimal,
    pub net_income: rust_decimal::Decimal,
}

/// Balance Sheet
#[derive(Debug, Serialize, Deserialize)]
pub struct BalanceSheet {
    pub as_of_date: DateTime<Utc>,
    pub assets: Vec<ReportLine>,
    pub liabilities: Vec<ReportLine>,
    pub equity: Vec<ReportLine>,
    pub total_assets: rust_decimal::Decimal,
    pub total_liabilities: rust_decimal::Decimal,
    pub total_equity: rust_decimal::Decimal,
}

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct ReportLine {
    pub account_number: String,
    pub account_name: String,
    pub amount: rust_decimal::Decimal,
}

impl ReportingService {
    /// Create a new reporting service
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Generate trial balance as of a specific date
    pub async fn trial_balance(
        &self,
        as_of_date: DateTime<Utc>,
    ) -> Result<TrialBalance, Box<dyn std::error::Error>> {
        let accounts: Vec<TrialBalanceAccount> = sqlx::query_as::<_, TrialBalanceAccount>(
            r#"
            SELECT
                a.account_number,
                a.account_name,
                a.account_type,
                COALESCE(SUM(jl.debit_amount), 0) as debit_balance,
                COALESCE(SUM(jl.credit_amount), 0) as credit_balance
            FROM gl_accounts a
            LEFT JOIN journal_lines jl ON a.id = jl.account_id
            LEFT JOIN journal_entries je ON jl.entry_id = je.id
            WHERE a.is_active = true
              AND (je.status = 'POSTED' OR je.id IS NULL)
              AND (je.entry_date <= $1 OR je.id IS NULL)
            GROUP BY a.id, a.account_number, a.account_name, a.account_type
            ORDER BY a.account_number
            "#,
        )
        .bind(as_of_date)
        .fetch_all(&self.pool)
        .await?;

        let total_debits: rust_decimal::Decimal = accounts
            .iter()
            .map(|a| a.debit_balance)
            .sum();

        let total_credits: rust_decimal::Decimal = accounts
            .iter()
            .map(|a| a.credit_balance)
            .sum();

        Ok(TrialBalance {
            as_of_date,
            accounts,
            total_debits,
            total_credits,
        })
    }

    /// Generate income statement for a date range
    pub async fn income_statement(
        &self,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<IncomeStatement, Box<dyn std::error::Error>> {
        // Get revenue accounts (credit balance)
        let revenue: Vec<ReportLine> = sqlx::query_as::<_, ReportLine>(
            r#"
            SELECT
                a.account_number,
                a.account_name,
                COALESCE(SUM(jl.credit_amount), 0) - COALESCE(SUM(jl.debit_amount), 0) as amount
            FROM gl_accounts a
            LEFT JOIN journal_lines jl ON a.id = jl.account_id
            LEFT JOIN journal_entries je ON jl.entry_id = je.id
            WHERE a.account_type = 'REVENUE'
              AND a.is_active = true
              AND je.status = 'POSTED'
              AND je.entry_date BETWEEN $1 AND $2
            GROUP BY a.id, a.account_number, a.account_name
            HAVING COALESCE(SUM(jl.credit_amount), 0) - COALESCE(SUM(jl.debit_amount), 0) != 0
            ORDER BY a.account_number
            "#,
        )
        .bind(start_date)
        .bind(end_date)
        .fetch_all(&self.pool)
        .await?;

        // Get expense accounts (debit balance)
        let expenses: Vec<ReportLine> = sqlx::query_as::<_, ReportLine>(
            r#"
            SELECT
                a.account_number,
                a.account_name,
                COALESCE(SUM(jl.debit_amount), 0) - COALESCE(SUM(jl.credit_amount), 0) as amount
            FROM gl_accounts a
            LEFT JOIN journal_lines jl ON a.id = jl.account_id
            LEFT JOIN journal_entries je ON jl.entry_id = je.id
            WHERE a.account_type = 'EXPENSE'
              AND a.is_active = true
              AND je.status = 'POSTED'
              AND je.entry_date BETWEEN $1 AND $2
            GROUP BY a.id, a.account_number, a.account_name
            HAVING COALESCE(SUM(jl.debit_amount), 0) - COALESCE(SUM(jl.credit_amount), 0) != 0
            ORDER BY a.account_number
            "#,
        )
        .bind(start_date)
        .bind(end_date)
        .fetch_all(&self.pool)
        .await?;

        let total_revenue: rust_decimal::Decimal = revenue.iter().map(|r| r.amount).sum();
        let total_expenses: rust_decimal::Decimal = expenses.iter().map(|e| e.amount).sum();
        let net_income = total_revenue - total_expenses;

        Ok(IncomeStatement {
            start_date,
            end_date,
            revenue,
            expenses,
            total_revenue,
            total_expenses,
            net_income,
        })
    }

    /// Generate balance sheet as of a specific date
    pub async fn balance_sheet(
        &self,
        as_of_date: DateTime<Utc>,
    ) -> Result<BalanceSheet, Box<dyn std::error::Error>> {
        // Get asset accounts (debit balance)
        let assets: Vec<ReportLine> = sqlx::query_as::<_, ReportLine>(
            r#"
            SELECT
                a.account_number,
                a.account_name,
                COALESCE(SUM(jl.debit_amount), 0) - COALESCE(SUM(jl.credit_amount), 0) as amount
            FROM gl_accounts a
            LEFT JOIN journal_lines jl ON a.id = jl.account_id
            LEFT JOIN journal_entries je ON jl.entry_id = je.id
            WHERE a.account_type = 'ASSET'
              AND a.is_active = true
              AND (je.status = 'POSTED' OR je.id IS NULL)
              AND (je.entry_date <= $1 OR je.id IS NULL)
            GROUP BY a.id, a.account_number, a.account_name
            HAVING COALESCE(SUM(jl.debit_amount), 0) - COALESCE(SUM(jl.credit_amount), 0) != 0
            ORDER BY a.account_number
            "#,
        )
        .bind(as_of_date)
        .fetch_all(&self.pool)
        .await?;

        // Get liability accounts (credit balance)
        let liabilities: Vec<ReportLine> = sqlx::query_as::<_, ReportLine>(
            r#"
            SELECT
                a.account_number,
                a.account_name,
                COALESCE(SUM(jl.credit_amount), 0) - COALESCE(SUM(jl.debit_amount), 0) as amount
            FROM gl_accounts a
            LEFT JOIN journal_lines jl ON a.id = jl.account_id
            LEFT JOIN journal_entries je ON jl.entry_id = je.id
            WHERE a.account_type = 'LIABILITY'
              AND a.is_active = true
              AND (je.status = 'POSTED' OR je.id IS NULL)
              AND (je.entry_date <= $1 OR je.id IS NULL)
            GROUP BY a.id, a.account_number, a.account_name
            HAVING COALESCE(SUM(jl.credit_amount), 0) - COALESCE(SUM(jl.debit_amount), 0) != 0
            ORDER BY a.account_number
            "#,
        )
        .bind(as_of_date)
        .fetch_all(&self.pool)
        .await?;

        // Get equity accounts (credit balance)
        let equity: Vec<ReportLine> = sqlx::query_as::<_, ReportLine>(
            r#"
            SELECT
                a.account_number,
                a.account_name,
                COALESCE(SUM(jl.credit_amount), 0) - COALESCE(SUM(jl.debit_amount), 0) as amount
            FROM gl_accounts a
            LEFT JOIN journal_lines jl ON a.id = jl.account_id
            LEFT JOIN journal_entries je ON jl.entry_id = je.id
            WHERE a.account_type = 'EQUITY'
              AND a.is_active = true
              AND (je.status = 'POSTED' OR je.id IS NULL)
              AND (je.entry_date <= $1 OR je.id IS NULL)
            GROUP BY a.id, a.account_number, a.account_name
            HAVING COALESCE(SUM(jl.credit_amount), 0) - COALESCE(SUM(jl.debit_amount), 0) != 0
            ORDER BY a.account_number
            "#,
        )
        .bind(as_of_date)
        .fetch_all(&self.pool)
        .await?;

        let total_assets: rust_decimal::Decimal = assets.iter().map(|a| a.amount).sum();
        let total_liabilities: rust_decimal::Decimal = liabilities.iter().map(|l| l.amount).sum();
        let total_equity: rust_decimal::Decimal = equity.iter().map(|e| e.amount).sum();

        Ok(BalanceSheet {
            as_of_date,
            assets,
            liabilities,
            equity,
            total_assets,
            total_liabilities,
            total_equity,
        })
    }
}
