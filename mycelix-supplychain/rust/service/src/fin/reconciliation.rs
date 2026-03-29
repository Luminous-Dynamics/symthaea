// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bank Reconciliation Module
//!
//! Provides automated and manual matching of bank transactions
//! to invoices, bills, and payments.

use chrono::{DateTime, NaiveDate, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::{FromRow, PgPool};
use uuid::Uuid;

use crate::integrations::plaid::PlaidTransaction;

/// Bank account linked via Plaid
#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct BankAccount {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub plaid_account_id: Option<String>,
    pub plaid_item_id: Option<String>,
    pub institution_name: Option<String>,
    pub account_name: String,
    pub account_type: String,
    pub account_subtype: Option<String>,
    pub mask: Option<String>,
    pub gl_account_id: Option<Uuid>,
    pub current_balance: Option<Decimal>,
    pub available_balance: Option<Decimal>,
    pub balance_currency: String,
    pub last_sync_at: Option<DateTime<Utc>>,
    pub sync_status: String,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
}

/// Imported bank transaction
#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct BankTransaction {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub bank_account_id: Uuid,
    pub external_transaction_id: String,
    pub transaction_date: NaiveDate,
    pub posted_date: Option<NaiveDate>,
    pub amount: Decimal,
    pub currency: String,
    pub merchant_name: Option<String>,
    pub description: Option<String>,
    pub category: Option<String>,
    pub is_pending: bool,
    pub payment_channel: Option<String>,
    pub reconciliation_status: String,
    pub matched_at: Option<DateTime<Utc>>,
    pub match_confidence: Option<Decimal>,
    pub matched_invoice_id: Option<Uuid>,
    pub matched_bill_id: Option<Uuid>,
    pub matched_payment_id: Option<Uuid>,
    pub created_at: DateTime<Utc>,
}

/// Suggested match for review
#[derive(Debug, Serialize, Deserialize)]
pub struct MatchSuggestion {
    pub bank_transaction_id: Uuid,
    pub suggested_type: MatchType,
    pub suggested_id: Uuid,
    pub confidence: f64,
    pub match_reasons: Vec<String>,
}

/// Type of document to match
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MatchType {
    Invoice,
    Bill,
    Payment,
}

/// Reconciliation statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct ReconciliationStats {
    pub total_transactions: i64,
    pub matched_count: i64,
    pub unmatched_count: i64,
    pub ignored_count: i64,
    pub pending_count: i64,
    pub auto_match_rate: f64,
}

/// Bank Reconciliation Service
#[derive(Clone)]
pub struct ReconciliationService {
    pool: PgPool,
}

impl ReconciliationService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Import transactions from Plaid
    pub async fn import_transactions(
        &self,
        bank_account_id: Uuid,
        transactions: Vec<PlaidTransaction>,
    ) -> Result<ImportResult, ReconciliationError> {
        let account = self.get_bank_account(bank_account_id).await?;
        let mut imported = 0;
        let mut updated = 0;
        let skipped = 0;

        for txn in transactions {
            // Check if transaction already exists
            let existing: Option<(Uuid,)> = sqlx::query_as(
                "SELECT id FROM fin_bank_transactions
                 WHERE tenant_id = $1 AND external_transaction_id = $2"
            )
            .bind(&account.tenant_id)
            .bind(&txn.transaction_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(ReconciliationError::Database)?;

            if let Some((id,)) = existing {
                // Update existing transaction if it was pending and now posted
                sqlx::query(
                    "UPDATE fin_bank_transactions
                     SET is_pending = $2, posted_date = $3, updated_at = NOW()
                     WHERE id = $1"
                )
                .bind(id)
                .bind(txn.pending)
                .bind(txn.date.parse::<NaiveDate>().ok())
                .execute(&self.pool)
                .await
                .map_err(ReconciliationError::Database)?;

                updated += 1;
            } else {
                // Insert new transaction
                sqlx::query(
                    "INSERT INTO fin_bank_transactions (
                        tenant_id, bank_account_id, external_transaction_id,
                        transaction_date, amount, currency, merchant_name,
                        description, category, is_pending, payment_channel
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)"
                )
                .bind(&account.tenant_id)
                .bind(bank_account_id)
                .bind(&txn.transaction_id)
                .bind(txn.date.parse::<NaiveDate>().unwrap_or(Utc::now().date_naive()))
                .bind(Decimal::try_from(txn.amount).unwrap_or_default())
                .bind(txn.iso_currency_code.unwrap_or_else(|| "USD".to_string()))
                .bind(&txn.merchant_name)
                .bind(&txn.name)
                .bind(txn.category.as_ref().and_then(|c| c.first()).cloned())
                .bind(txn.pending)
                .bind(&txn.payment_channel)
                .execute(&self.pool)
                .await
                .map_err(ReconciliationError::Database)?;

                imported += 1;
            }
        }

        Ok(ImportResult {
            imported,
            updated,
            skipped,
        })
    }

    /// Auto-match transactions using intelligent matching
    pub async fn auto_match(&self, tenant_id: Uuid) -> Result<Vec<MatchSuggestion>, ReconciliationError> {
        let unmatched = self.get_unmatched_transactions(tenant_id).await?;
        let mut suggestions = Vec::new();

        for txn in unmatched {
            if let Some(suggestion) = self.find_best_match(&txn).await? {
                suggestions.push(suggestion);
            }
        }

        Ok(suggestions)
    }

    /// Find the best match for a transaction
    async fn find_best_match(&self, txn: &BankTransaction) -> Result<Option<MatchSuggestion>, ReconciliationError> {
        let mut best_match: Option<MatchSuggestion> = None;
        let amount = txn.amount.abs();

        // For credits (positive amounts), look for invoice payments
        if txn.amount > Decimal::ZERO {
            // Find invoices with matching amount
            let invoices: Vec<(Uuid, Decimal, String)> = sqlx::query_as(
                "SELECT id, total_amount, invoice_number
                 FROM fin_invoices
                 WHERE tenant_id = $1
                 AND status IN ('SENT', 'OVERDUE')
                 AND ABS(total_amount - $2) < 0.01
                 ORDER BY due_date"
            )
            .bind(&txn.tenant_id)
            .bind(&amount)
            .fetch_all(&self.pool)
            .await
            .map_err(ReconciliationError::Database)?;

            for (invoice_id, inv_amount, inv_number) in invoices {
                let confidence = self.calculate_match_confidence(txn, &inv_amount, Some(&inv_number));
                if confidence > 70.0 && (best_match.is_none() || confidence > best_match.as_ref().unwrap().confidence) {
                    best_match = Some(MatchSuggestion {
                        bank_transaction_id: txn.id,
                        suggested_type: MatchType::Invoice,
                        suggested_id: invoice_id,
                        confidence,
                        match_reasons: vec![
                            format!("Amount matches: ${}", amount),
                            format!("Invoice: {}", inv_number),
                        ],
                    });
                }
            }
        }

        // For debits (negative amounts), look for bill payments
        if txn.amount < Decimal::ZERO {
            let bills: Vec<(Uuid, Decimal, String)> = sqlx::query_as(
                "SELECT id, total_amount, bill_number
                 FROM fin_bills
                 WHERE tenant_id = $1
                 AND status IN ('APPROVED', 'PENDING')
                 AND ABS(total_amount - $2) < 0.01
                 ORDER BY due_date"
            )
            .bind(&txn.tenant_id)
            .bind(&amount)
            .fetch_all(&self.pool)
            .await
            .map_err(ReconciliationError::Database)?;

            for (bill_id, bill_amount, bill_number) in bills {
                let confidence = self.calculate_match_confidence(txn, &bill_amount, Some(&bill_number));
                if confidence > 70.0 && (best_match.is_none() || confidence > best_match.as_ref().unwrap().confidence) {
                    best_match = Some(MatchSuggestion {
                        bank_transaction_id: txn.id,
                        suggested_type: MatchType::Bill,
                        suggested_id: bill_id,
                        confidence,
                        match_reasons: vec![
                            format!("Amount matches: ${}", amount),
                            format!("Bill: {}", bill_number),
                        ],
                    });
                }
            }
        }

        Ok(best_match)
    }

    /// Calculate match confidence score
    fn calculate_match_confidence(&self, txn: &BankTransaction, doc_amount: &Decimal, doc_number: Option<&str>) -> f64 {
        let mut score: f64 = 0.0;

        // Exact amount match = 50 points
        if (txn.amount.abs() - doc_amount.abs()).abs() < Decimal::new(1, 2) {
            score += 50.0;
        }

        // Amount within 1% = 30 points
        let diff_pct = ((txn.amount.abs() - doc_amount.abs()) / doc_amount.abs() * Decimal::from(100)).abs();
        if diff_pct < Decimal::from(1) {
            score += 30.0;
        }

        // Description contains document number = 30 points
        if let (Some(desc), Some(num)) = (&txn.description, doc_number) {
            if desc.to_lowercase().contains(&num.to_lowercase()) {
                score += 30.0;
            }
        }

        // Merchant name might contain invoice reference = 20 points
        if let (Some(merchant), Some(num)) = (&txn.merchant_name, doc_number) {
            if merchant.to_lowercase().contains(&num.to_lowercase()) {
                score += 20.0;
            }
        }

        score.min(100.0)
    }

    /// Manually match a transaction
    pub async fn manual_match(
        &self,
        transaction_id: Uuid,
        match_type: MatchType,
        match_id: Uuid,
        user_id: Uuid,
    ) -> Result<(), ReconciliationError> {
        let (invoice_id, bill_id, payment_id) = match match_type {
            MatchType::Invoice => (Some(match_id), None, None),
            MatchType::Bill => (None, Some(match_id), None),
            MatchType::Payment => (None, None, Some(match_id)),
        };

        sqlx::query(
            "UPDATE fin_bank_transactions
             SET reconciliation_status = 'MATCHED',
                 matched_at = NOW(),
                 matched_by = $2,
                 match_confidence = 100,
                 matched_invoice_id = $3,
                 matched_bill_id = $4,
                 matched_payment_id = $5,
                 updated_at = NOW()
             WHERE id = $1"
        )
        .bind(transaction_id)
        .bind(user_id)
        .bind(invoice_id)
        .bind(bill_id)
        .bind(payment_id)
        .execute(&self.pool)
        .await
        .map_err(ReconciliationError::Database)?;

        Ok(())
    }

    /// Ignore a transaction
    pub async fn ignore_transaction(
        &self,
        transaction_id: Uuid,
        reason: &str,
        user_id: Uuid,
    ) -> Result<(), ReconciliationError> {
        sqlx::query(
            "UPDATE fin_bank_transactions
             SET reconciliation_status = 'IGNORED',
                 ignore_reason = $2,
                 matched_by = $3,
                 matched_at = NOW(),
                 updated_at = NOW()
             WHERE id = $1"
        )
        .bind(transaction_id)
        .bind(reason)
        .bind(user_id)
        .execute(&self.pool)
        .await
        .map_err(ReconciliationError::Database)?;

        Ok(())
    }

    /// Get unmatched transactions
    pub async fn get_unmatched_transactions(&self, tenant_id: Uuid) -> Result<Vec<BankTransaction>, ReconciliationError> {
        let transactions = sqlx::query_as::<_, BankTransaction>(
            "SELECT * FROM fin_bank_transactions
             WHERE tenant_id = $1 AND reconciliation_status = 'UNMATCHED'
             ORDER BY transaction_date DESC"
        )
        .bind(tenant_id)
        .fetch_all(&self.pool)
        .await
        .map_err(ReconciliationError::Database)?;

        Ok(transactions)
    }

    /// Get reconciliation statistics
    pub async fn get_stats(&self, tenant_id: Uuid, bank_account_id: Option<Uuid>) -> Result<ReconciliationStats, ReconciliationError> {
        let (total, matched, unmatched, ignored, pending): (i64, i64, i64, i64, i64) = if let Some(account_id) = bank_account_id {
            sqlx::query_as(
                "SELECT
                    COUNT(*),
                    COUNT(*) FILTER (WHERE reconciliation_status = 'MATCHED'),
                    COUNT(*) FILTER (WHERE reconciliation_status = 'UNMATCHED'),
                    COUNT(*) FILTER (WHERE reconciliation_status = 'IGNORED'),
                    COUNT(*) FILTER (WHERE is_pending = true)
                 FROM fin_bank_transactions
                 WHERE tenant_id = $1 AND bank_account_id = $2"
            )
            .bind(tenant_id)
            .bind(account_id)
            .fetch_one(&self.pool)
            .await
            .map_err(ReconciliationError::Database)?
        } else {
            sqlx::query_as(
                "SELECT
                    COUNT(*),
                    COUNT(*) FILTER (WHERE reconciliation_status = 'MATCHED'),
                    COUNT(*) FILTER (WHERE reconciliation_status = 'UNMATCHED'),
                    COUNT(*) FILTER (WHERE reconciliation_status = 'IGNORED'),
                    COUNT(*) FILTER (WHERE is_pending = true)
                 FROM fin_bank_transactions
                 WHERE tenant_id = $1"
            )
            .bind(tenant_id)
            .fetch_one(&self.pool)
            .await
            .map_err(ReconciliationError::Database)?
        };

        let auto_match_rate = if total > 0 {
            (matched as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        Ok(ReconciliationStats {
            total_transactions: total,
            matched_count: matched,
            unmatched_count: unmatched,
            ignored_count: ignored,
            pending_count: pending,
            auto_match_rate,
        })
    }

    /// Get a bank account by ID
    async fn get_bank_account(&self, id: Uuid) -> Result<BankAccount, ReconciliationError> {
        sqlx::query_as::<_, BankAccount>(
            "SELECT * FROM fin_bank_accounts WHERE id = $1"
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(ReconciliationError::Database)?
        .ok_or(ReconciliationError::NotFound("Bank account not found".into()))
    }

    /// Link a bank account to a GL account
    pub async fn link_gl_account(
        &self,
        bank_account_id: Uuid,
        gl_account_id: Uuid,
    ) -> Result<(), ReconciliationError> {
        sqlx::query(
            "UPDATE fin_bank_accounts
             SET gl_account_id = $2, updated_at = NOW()
             WHERE id = $1"
        )
        .bind(bank_account_id)
        .bind(gl_account_id)
        .execute(&self.pool)
        .await
        .map_err(ReconciliationError::Database)?;

        Ok(())
    }

    /// Create a bank account from Plaid data
    pub async fn create_bank_account(
        &self,
        tenant_id: Uuid,
        plaid_account_id: &str,
        plaid_item_id: &str,
        account_name: &str,
        account_type: &str,
        account_subtype: Option<&str>,
        mask: Option<&str>,
        institution_name: Option<&str>,
    ) -> Result<BankAccount, ReconciliationError> {
        let id = Uuid::new_v4();

        sqlx::query(
            "INSERT INTO fin_bank_accounts (
                id, tenant_id, plaid_account_id, plaid_item_id,
                account_name, account_type, account_subtype, mask,
                institution_name
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)"
        )
        .bind(id)
        .bind(tenant_id)
        .bind(plaid_account_id)
        .bind(plaid_item_id)
        .bind(account_name)
        .bind(account_type)
        .bind(account_subtype)
        .bind(mask)
        .bind(institution_name)
        .execute(&self.pool)
        .await
        .map_err(ReconciliationError::Database)?;

        self.get_bank_account(id).await
    }

    /// List all bank accounts for a tenant
    pub async fn list_bank_accounts(&self, tenant_id: Uuid) -> Result<Vec<BankAccount>, ReconciliationError> {
        let accounts = sqlx::query_as::<_, BankAccount>(
            "SELECT * FROM fin_bank_accounts WHERE tenant_id = $1 AND is_active = true ORDER BY account_name"
        )
        .bind(tenant_id)
        .fetch_all(&self.pool)
        .await
        .map_err(ReconciliationError::Database)?;

        Ok(accounts)
    }

    /// Get a bank account by ID (public version)
    pub async fn get_bank_account_by_id(&self, id: Uuid) -> Result<Option<BankAccount>, ReconciliationError> {
        let account = sqlx::query_as::<_, BankAccount>(
            "SELECT * FROM fin_bank_accounts WHERE id = $1"
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(ReconciliationError::Database)?;

        Ok(account)
    }

    /// List transactions with optional filters
    pub async fn list_transactions(
        &self,
        tenant_id: Uuid,
        bank_account_id: Option<Uuid>,
        status: Option<&str>,
        from_date: Option<NaiveDate>,
        to_date: Option<NaiveDate>,
    ) -> Result<Vec<BankTransaction>, ReconciliationError> {
        let mut query = String::from(
            "SELECT * FROM fin_bank_transactions WHERE tenant_id = $1"
        );
        let mut param_count = 1;

        if bank_account_id.is_some() {
            param_count += 1;
            query.push_str(&format!(" AND bank_account_id = ${}", param_count));
        }

        if status.is_some() {
            param_count += 1;
            query.push_str(&format!(" AND reconciliation_status = ${}", param_count));
        }

        if from_date.is_some() {
            param_count += 1;
            query.push_str(&format!(" AND transaction_date >= ${}", param_count));
        }

        if to_date.is_some() {
            param_count += 1;
            query.push_str(&format!(" AND transaction_date <= ${}", param_count));
        }

        query.push_str(" ORDER BY transaction_date DESC, created_at DESC LIMIT 500");

        // Build the query dynamically
        let mut q = sqlx::query_as::<_, BankTransaction>(&query).bind(tenant_id);

        if let Some(account_id) = bank_account_id {
            q = q.bind(account_id);
        }

        if let Some(s) = status {
            q = q.bind(s);
        }

        if let Some(d) = from_date {
            q = q.bind(d);
        }

        if let Some(d) = to_date {
            q = q.bind(d);
        }

        let transactions = q.fetch_all(&self.pool).await.map_err(ReconciliationError::Database)?;

        Ok(transactions)
    }

    /// Get the Plaid access token for a bank account (stored securely)
    pub async fn get_plaid_access_token(&self, bank_account_id: Uuid) -> Result<Option<String>, ReconciliationError> {
        let result: Option<(Option<String>,)> = sqlx::query_as(
            "SELECT plaid_access_token FROM fin_bank_accounts WHERE id = $1"
        )
        .bind(bank_account_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(ReconciliationError::Database)?;

        Ok(result.and_then(|(token,)| token))
    }

    /// Update the sync cursor for a bank account
    pub async fn update_sync_cursor(
        &self,
        bank_account_id: Uuid,
        cursor: &str,
    ) -> Result<(), ReconciliationError> {
        sqlx::query(
            "UPDATE fin_bank_accounts SET sync_cursor = $2, last_sync_at = NOW(), sync_status = 'OK', updated_at = NOW() WHERE id = $1"
        )
        .bind(bank_account_id)
        .bind(cursor)
        .execute(&self.pool)
        .await
        .map_err(ReconciliationError::Database)?;

        Ok(())
    }
}

/// Import result
#[derive(Debug, Serialize)]
pub struct ImportResult {
    pub imported: usize,
    pub updated: usize,
    pub skipped: usize,
}

/// Reconciliation error
#[derive(Debug, thiserror::Error)]
pub enum ReconciliationError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Integration error: {0}")]
    Integration(String),
}
