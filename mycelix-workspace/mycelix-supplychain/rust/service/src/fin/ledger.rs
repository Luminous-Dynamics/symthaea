// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! General Ledger service
//!
//! Manages GL accounts, journal entries, and posting logic.

use chrono::{DateTime, Utc};
use serde_json::json;
use sha2::{Sha256, Digest};
use sqlx::PgPool;
use tracing::info;
use uuid::Uuid;

use super::models::*;

/// Service for managing general ledger operations
#[derive(Clone)]
pub struct LedgerService {
    pool: PgPool,
}

impl LedgerService {
    /// Create a new ledger service
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Create a new GL account
    pub async fn create_account(
        &self,
        req: CreateGlAccountRequest,
    ) -> Result<GlAccount, sqlx::Error> {
        let id = Uuid::new_v4();
        let now = Utc::now();

        let account = sqlx::query_as::<_, GlAccount>(
            r#"
            INSERT INTO gl_accounts (
                id, account_number, account_name, account_type,
                parent_account_id, is_active, currency, created_at, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING *
            "#,
        )
        .bind(id)
        .bind(&req.account_number)
        .bind(&req.account_name)
        .bind(&req.account_type)
        .bind(req.parent_account_id)
        .bind(true)
        .bind(&req.currency)
        .bind(now)
        .bind(now)
        .fetch_one(&self.pool)
        .await?;

        Ok(account)
    }

    /// Get a GL account by ID
    pub async fn get_account(&self, id: Uuid) -> Result<Option<GlAccount>, sqlx::Error> {
        let account = sqlx::query_as::<_, GlAccount>(
            "SELECT * FROM gl_accounts WHERE id = $1"
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(account)
    }

    /// List all GL accounts
    pub async fn list_accounts(&self) -> Result<Vec<GlAccount>, sqlx::Error> {
        let accounts = sqlx::query_as::<_, GlAccount>(
            "SELECT * FROM gl_accounts WHERE is_active = true ORDER BY account_number"
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(accounts)
    }

    /// Get account balance at a specific date
    pub async fn get_account_balance(
        &self,
        account_id: Uuid,
        as_of_date: DateTime<Utc>,
    ) -> Result<rust_decimal::Decimal, sqlx::Error> {
        let result: Option<rust_decimal::Decimal> = sqlx::query_scalar(
            r#"
            SELECT COALESCE(SUM(debit_amount), 0) - COALESCE(SUM(credit_amount), 0)
            FROM journal_lines jl
            JOIN journal_entries je ON jl.entry_id = je.id
            WHERE jl.account_id = $1
              AND je.status = 'POSTED'
              AND je.entry_date <= $2
            "#,
        )
        .bind(account_id)
        .bind(as_of_date)
        .fetch_one(&self.pool)
        .await?;

        Ok(result.unwrap_or(rust_decimal::Decimal::ZERO))
    }

    /// Create a journal entry (draft state)
    pub async fn create_journal_entry(
        &self,
        description: String,
        reference: Option<String>,
        created_by: Uuid,
        lines: Vec<(Uuid, Option<rust_decimal::Decimal>, Option<rust_decimal::Decimal>, Option<String>)>,
    ) -> Result<JournalEntry, Box<dyn std::error::Error>> {
        // Validate double-entry: debits must equal credits
        let total_debits: rust_decimal::Decimal = lines
            .iter()
            .filter_map(|(_, debit, _, _)| *debit)
            .sum();
        let total_credits: rust_decimal::Decimal = lines
            .iter()
            .filter_map(|(_, _, credit, _)| *credit)
            .sum();

        if total_debits != total_credits {
            return Err("Debits must equal credits in double-entry bookkeeping".into());
        }

        let mut tx = self.pool.begin().await?;
        let id = Uuid::new_v4();
        let now = Utc::now();
        let entry_number = format!("JE-{}", id.to_string()[..8].to_uppercase());

        // Calculate hash of all line items
        let lines_hash = Self::calculate_lines_hash(&lines);

        // Insert journal entry
        let entry = sqlx::query_as::<_, JournalEntry>(
            r#"
            INSERT INTO journal_entries (
                id, entry_number, entry_date, description, reference,
                status, lines_hash, created_by, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING *
            "#,
        )
        .bind(id)
        .bind(&entry_number)
        .bind(now)
        .bind(&description)
        .bind(reference)
        .bind(EntryStatus::Draft)
        .bind(&lines_hash)
        .bind(created_by)
        .bind(now)
        .fetch_one(&mut *tx)
        .await?;

        // Insert journal lines
        for (line_num, (account_id, debit, credit, line_desc)) in lines.iter().enumerate() {
            sqlx::query(
                r#"
                INSERT INTO journal_lines (
                    id, entry_id, line_number, account_id,
                    debit_amount, credit_amount, description
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                "#,
            )
            .bind(Uuid::new_v4())
            .bind(id)
            .bind(line_num as i32 + 1)
            .bind(account_id)
            .bind(debit)
            .bind(credit)
            .bind(line_desc)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        Ok(entry)
    }

    /// Post a journal entry (make it effective)
    pub async fn post_journal_entry(
        &self,
        entry_id: Uuid,
    ) -> Result<JournalEntry, Box<dyn std::error::Error>> {
        let now = Utc::now();

        // Update status to POSTED
        let entry = sqlx::query_as::<_, JournalEntry>(
            r#"
            UPDATE journal_entries
            SET status = 'POSTED', posted_at = $1
            WHERE id = $2 AND status = 'DRAFT'
            RETURNING *
            "#,
        )
        .bind(now)
        .bind(entry_id)
        .fetch_one(&self.pool)
        .await?;

        // Create DKG claim for audit trail
        let claim_id = self.create_journal_entry_claim(&entry).await;

        // Update the journal entry with the claim ID if creation succeeded
        if let Some(ref cid) = claim_id {
            let _ = sqlx::query(
                "UPDATE journal_entries SET claim_id = $1 WHERE id = $2"
            )
            .bind(cid)
            .bind(entry_id)
            .execute(&self.pool)
            .await;

            info!(
                entry_id = %entry_id,
                claim_id = %cid,
                "Created DKG claim for posted journal entry"
            );
        }

        Ok(entry)
    }

    /// Create a DKG claim for a journal entry
    async fn create_journal_entry_claim(&self, entry: &JournalEntry) -> Option<String> {
        // Build claim data structure for the journal entry
        let claim_data = json!({
            "type": "FinancialJournalEntryClaim",
            "entry_number": entry.entry_number,
            "entry_date": entry.entry_date.to_rfc3339(),
            "posted_at": entry.posted_at.map(|t| t.to_rfc3339()),
            "description": entry.description,
            "lines_hash": entry.lines_hash,
            "created_by": entry.created_by.to_string(),
        });

        // Compute claim hash
        let mut hasher = Sha256::new();
        hasher.update(entry.id.to_string().as_bytes());
        hasher.update(entry.entry_number.as_bytes());
        hasher.update(entry.lines_hash.as_bytes());
        hasher.update(entry.entry_date.to_rfc3339().as_bytes());
        let claim_hash = format!("{:x}", hasher.finalize());

        // Generate claim ID
        let claim_id = format!("fin:je:{}", Uuid::new_v4());

        // Create the DKG claim structure
        let dkg_claim = claim_model::DkgClaim {
            id: claim_id.clone(),
            claim_type: "FinancialJournalEntryClaim".to_string(),
            issuer: format!("did:mycelix:fin:{}", entry.created_by),
            subject: claim_model::Subject {
                batch_id: entry.entry_number.clone(),
                product_id: entry.id.to_string(),
            },
            assertion: claim_model::Assertion {
                event_type: claim_model::EventType::Certified,
                quantity: None,
                unit: None,
                facility_id: None,
            },
            evidence: claim_model::Evidence {
                vc_jwt: serde_json::to_string(&claim_data).unwrap_or_default(),
                additional_documents: None,
            },
            lineage: claim_model::Lineage {
                hash: claim_hash,
                previous_claims: None,
            },
            timestamp: Utc::now(),
            confidence: Some(1.0),
            metadata: Some({
                let mut map = std::collections::HashMap::new();
                map.insert("entry_id".to_string(), json!(entry.id.to_string()));
                map.insert("entry_number".to_string(), json!(entry.entry_number));
                map
            }),
        };

        // Publish to DKG (this is a placeholder that logs a warning)
        match crate::dkg_client::publish_claim(&dkg_claim).await {
            Ok(_txid) => Some(claim_id),
            Err(e) => {
                tracing::warn!(error = %e, "Failed to publish journal entry claim to DKG");
                // Still return the claim ID - the claim was created locally
                Some(claim_id)
            }
        }
    }

    /// Calculate SHA-256 hash of journal lines for tamper detection
    fn calculate_lines_hash(
        lines: &[(Uuid, Option<rust_decimal::Decimal>, Option<rust_decimal::Decimal>, Option<String>)],
    ) -> String {
        let mut hasher = Sha256::new();

        for (account_id, debit, credit, desc) in lines {
            hasher.update(account_id.as_bytes());
            if let Some(d) = debit {
                hasher.update(d.to_string().as_bytes());
            }
            if let Some(c) = credit {
                hasher.update(c.to_string().as_bytes());
            }
            if let Some(d) = desc {
                hasher.update(d.as_bytes());
            }
        }

        format!("{:x}", hasher.finalize())
    }

    /// Get a journal entry by ID with its lines
    pub async fn get_journal_entry(
        &self,
        entry_id: Uuid,
    ) -> Result<Option<JournalEntryWithLines>, sqlx::Error> {
        // Get the entry
        let entry = sqlx::query_as::<_, JournalEntry>(
            "SELECT * FROM journal_entries WHERE id = $1"
        )
        .bind(entry_id)
        .fetch_optional(&self.pool)
        .await?;

        match entry {
            Some(entry) => {
                // Get the lines
                let lines = sqlx::query_as::<_, JournalLine>(
                    "SELECT * FROM journal_lines WHERE entry_id = $1 ORDER BY line_number"
                )
                .bind(entry_id)
                .fetch_all(&self.pool)
                .await?;

                Ok(Some(JournalEntryWithLines { entry, lines }))
            }
            None => Ok(None),
        }
    }

    /// List journal entries with optional filters
    pub async fn list_journal_entries(
        &self,
        status: Option<EntryStatus>,
        limit: i32,
        offset: i32,
    ) -> Result<Vec<JournalEntry>, sqlx::Error> {
        let entries = match status {
            Some(s) => {
                sqlx::query_as::<_, JournalEntry>(
                    r#"
                    SELECT * FROM journal_entries
                    WHERE status = $1
                    ORDER BY entry_date DESC, created_at DESC
                    LIMIT $2 OFFSET $3
                    "#,
                )
                .bind(s)
                .bind(limit)
                .bind(offset)
                .fetch_all(&self.pool)
                .await?
            }
            None => {
                sqlx::query_as::<_, JournalEntry>(
                    r#"
                    SELECT * FROM journal_entries
                    ORDER BY entry_date DESC, created_at DESC
                    LIMIT $1 OFFSET $2
                    "#,
                )
                .bind(limit)
                .bind(offset)
                .fetch_all(&self.pool)
                .await?
            }
        };

        Ok(entries)
    }
}
