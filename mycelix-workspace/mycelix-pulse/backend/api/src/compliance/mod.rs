// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Compliance & Archival
//!
//! Legal hold, retention policies, eDiscovery, and audit logging

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;
use std::collections::HashMap;

// ============================================================================
// Retention Policy Service
// ============================================================================

pub struct RetentionService {
    pool: PgPool,
}

impl RetentionService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Create a retention policy
    pub async fn create_policy(
        &self,
        org_id: Uuid,
        policy: RetentionPolicyInput,
    ) -> Result<RetentionPolicy, ComplianceError> {
        let policy_id = Uuid::new_v4();

        sqlx::query(
            r#"
            INSERT INTO retention_policies (id, org_id, name, description, retention_days,
                                           applies_to, action, enabled, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
            "#,
        )
        .bind(policy_id)
        .bind(org_id)
        .bind(&policy.name)
        .bind(&policy.description)
        .bind(policy.retention_days)
        .bind(serde_json::to_value(&policy.applies_to).unwrap())
        .bind(policy.action.to_string())
        .bind(policy.enabled)
        .execute(&self.pool)
        .await
        .map_err(|e| ComplianceError::Database(e.to_string()))?;

        Ok(RetentionPolicy {
            id: policy_id,
            org_id,
            name: policy.name,
            description: policy.description,
            retention_days: policy.retention_days,
            applies_to: policy.applies_to,
            action: policy.action,
            enabled: policy.enabled,
            created_at: Utc::now(),
        })
    }

    /// Get all policies for an organization
    pub async fn get_policies(&self, org_id: Uuid) -> Result<Vec<RetentionPolicy>, ComplianceError> {
        let policies: Vec<RetentionPolicy> = sqlx::query_as(
            r#"
            SELECT id, org_id, name, description, retention_days, applies_to, action, enabled, created_at
            FROM retention_policies
            WHERE org_id = $1
            ORDER BY created_at DESC
            "#,
        )
        .bind(org_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| ComplianceError::Database(e.to_string()))?;

        Ok(policies)
    }

    /// Apply retention policies (run as scheduled job)
    pub async fn apply_policies(&self, org_id: Uuid) -> Result<RetentionResult, ComplianceError> {
        let policies = self.get_policies(org_id).await?;
        let mut result = RetentionResult::default();

        for policy in policies.iter().filter(|p| p.enabled) {
            let cutoff_date = Utc::now() - Duration::days(policy.retention_days as i64);

            // Build query based on applies_to criteria
            let affected = match policy.action {
                RetentionAction::Delete => {
                    self.delete_matching_emails(org_id, &policy.applies_to, cutoff_date)
                        .await?
                }
                RetentionAction::Archive => {
                    self.archive_matching_emails(org_id, &policy.applies_to, cutoff_date)
                        .await?
                }
                RetentionAction::Move { ref folder } => {
                    self.move_matching_emails(org_id, &policy.applies_to, cutoff_date, folder)
                        .await?
                }
            };

            result.policies_applied += 1;
            result.emails_affected += affected;
        }

        Ok(result)
    }

    async fn delete_matching_emails(
        &self,
        org_id: Uuid,
        applies_to: &RetentionAppliesTo,
        cutoff: DateTime<Utc>,
    ) -> Result<i64, ComplianceError> {
        // Don't delete emails under legal hold
        let result = sqlx::query(
            r#"
            DELETE FROM emails e
            USING users u
            WHERE e.user_id = u.id
            AND u.org_id = $1
            AND e.received_at < $2
            AND e.folder = ANY($3)
            AND NOT EXISTS (
                SELECT 1 FROM legal_holds lh
                WHERE lh.org_id = $1
                AND lh.status = 'active'
                AND e.received_at BETWEEN lh.start_date AND COALESCE(lh.end_date, NOW())
            )
            "#,
        )
        .bind(org_id)
        .bind(cutoff)
        .bind(&applies_to.folders)
        .execute(&self.pool)
        .await
        .map_err(|e| ComplianceError::Database(e.to_string()))?;

        Ok(result.rows_affected() as i64)
    }

    async fn archive_matching_emails(
        &self,
        org_id: Uuid,
        applies_to: &RetentionAppliesTo,
        cutoff: DateTime<Utc>,
    ) -> Result<i64, ComplianceError> {
        let result = sqlx::query(
            r#"
            UPDATE emails e
            SET folder = 'Archive', archived_at = NOW()
            FROM users u
            WHERE e.user_id = u.id
            AND u.org_id = $1
            AND e.received_at < $2
            AND e.folder = ANY($3)
            AND e.folder != 'Archive'
            "#,
        )
        .bind(org_id)
        .bind(cutoff)
        .bind(&applies_to.folders)
        .execute(&self.pool)
        .await
        .map_err(|e| ComplianceError::Database(e.to_string()))?;

        Ok(result.rows_affected() as i64)
    }

    async fn move_matching_emails(
        &self,
        org_id: Uuid,
        applies_to: &RetentionAppliesTo,
        cutoff: DateTime<Utc>,
        target_folder: &str,
    ) -> Result<i64, ComplianceError> {
        let result = sqlx::query(
            r#"
            UPDATE emails e
            SET folder = $4
            FROM users u
            WHERE e.user_id = u.id
            AND u.org_id = $1
            AND e.received_at < $2
            AND e.folder = ANY($3)
            "#,
        )
        .bind(org_id)
        .bind(cutoff)
        .bind(&applies_to.folders)
        .bind(target_folder)
        .execute(&self.pool)
        .await
        .map_err(|e| ComplianceError::Database(e.to_string()))?;

        Ok(result.rows_affected() as i64)
    }
}

// ============================================================================
// Legal Hold Service
// ============================================================================

pub struct LegalHoldService {
    pool: PgPool,
}

impl LegalHoldService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Create a legal hold
    pub async fn create_hold(
        &self,
        org_id: Uuid,
        hold: LegalHoldInput,
        created_by: Uuid,
    ) -> Result<LegalHold, ComplianceError> {
        let hold_id = Uuid::new_v4();

        sqlx::query(
            r#"
            INSERT INTO legal_holds (id, org_id, name, description, matter_id,
                                     custodians, start_date, end_date, status,
                                     created_by, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'active', $9, NOW())
            "#,
        )
        .bind(hold_id)
        .bind(org_id)
        .bind(&hold.name)
        .bind(&hold.description)
        .bind(&hold.matter_id)
        .bind(&hold.custodians)
        .bind(hold.start_date)
        .bind(hold.end_date)
        .bind(created_by)
        .execute(&self.pool)
        .await
        .map_err(|e| ComplianceError::Database(e.to_string()))?;

        // Log audit event
        self.audit_log(org_id, created_by, AuditAction::LegalHoldCreated, hold_id)
            .await?;

        Ok(LegalHold {
            id: hold_id,
            org_id,
            name: hold.name,
            description: hold.description,
            matter_id: hold.matter_id,
            custodians: hold.custodians,
            start_date: hold.start_date,
            end_date: hold.end_date,
            status: HoldStatus::Active,
            created_by,
            created_at: Utc::now(),
        })
    }

    /// Release a legal hold
    pub async fn release_hold(
        &self,
        hold_id: Uuid,
        released_by: Uuid,
    ) -> Result<(), ComplianceError> {
        let hold: (Uuid,) = sqlx::query_as(
            "SELECT org_id FROM legal_holds WHERE id = $1",
        )
        .bind(hold_id)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| ComplianceError::Database(e.to_string()))?;

        sqlx::query(
            "UPDATE legal_holds SET status = 'released', released_at = NOW(), released_by = $2 WHERE id = $1",
        )
        .bind(hold_id)
        .bind(released_by)
        .execute(&self.pool)
        .await
        .map_err(|e| ComplianceError::Database(e.to_string()))?;

        self.audit_log(hold.0, released_by, AuditAction::LegalHoldReleased, hold_id)
            .await?;

        Ok(())
    }

    /// Get all legal holds
    pub async fn get_holds(&self, org_id: Uuid) -> Result<Vec<LegalHold>, ComplianceError> {
        let holds: Vec<LegalHold> = sqlx::query_as(
            r#"
            SELECT id, org_id, name, description, matter_id, custodians,
                   start_date, end_date, status, created_by, created_at
            FROM legal_holds
            WHERE org_id = $1
            ORDER BY created_at DESC
            "#,
        )
        .bind(org_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| ComplianceError::Database(e.to_string()))?;

        Ok(holds)
    }

    /// Check if email is under legal hold
    pub async fn is_under_hold(
        &self,
        org_id: Uuid,
        email_date: DateTime<Utc>,
        user_id: Uuid,
    ) -> Result<bool, ComplianceError> {
        let result: Option<(bool,)> = sqlx::query_as(
            r#"
            SELECT true FROM legal_holds
            WHERE org_id = $1
            AND status = 'active'
            AND $2 BETWEEN start_date AND COALESCE(end_date, NOW() + INTERVAL '100 years')
            AND ($3 = ANY(custodians) OR array_length(custodians, 1) IS NULL)
            LIMIT 1
            "#,
        )
        .bind(org_id)
        .bind(email_date)
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| ComplianceError::Database(e.to_string()))?;

        Ok(result.is_some())
    }

    async fn audit_log(
        &self,
        org_id: Uuid,
        user_id: Uuid,
        action: AuditAction,
        reference_id: Uuid,
    ) -> Result<(), ComplianceError> {
        sqlx::query(
            r#"
            INSERT INTO audit_log (id, org_id, user_id, action, reference_id, created_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
            "#,
        )
        .bind(Uuid::new_v4())
        .bind(org_id)
        .bind(user_id)
        .bind(action.to_string())
        .bind(reference_id)
        .execute(&self.pool)
        .await
        .map_err(|e| ComplianceError::Database(e.to_string()))?;

        Ok(())
    }
}

// ============================================================================
// eDiscovery Service
// ============================================================================

pub struct EDiscoveryService {
    pool: PgPool,
}

impl EDiscoveryService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Create an eDiscovery search
    pub async fn create_search(
        &self,
        org_id: Uuid,
        search: EDiscoverySearchInput,
        created_by: Uuid,
    ) -> Result<EDiscoverySearch, ComplianceError> {
        let search_id = Uuid::new_v4();

        sqlx::query(
            r#"
            INSERT INTO ediscovery_searches (id, org_id, name, description, matter_id,
                                             query, date_from, date_to, custodians,
                                             status, created_by, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'pending', $10, NOW())
            "#,
        )
        .bind(search_id)
        .bind(org_id)
        .bind(&search.name)
        .bind(&search.description)
        .bind(&search.matter_id)
        .bind(&search.query)
        .bind(search.date_from)
        .bind(search.date_to)
        .bind(&search.custodians)
        .bind(created_by)
        .execute(&self.pool)
        .await
        .map_err(|e| ComplianceError::Database(e.to_string()))?;

        Ok(EDiscoverySearch {
            id: search_id,
            org_id,
            name: search.name,
            description: search.description,
            matter_id: search.matter_id,
            query: search.query,
            date_from: search.date_from,
            date_to: search.date_to,
            custodians: search.custodians,
            status: SearchStatus::Pending,
            result_count: None,
            created_by,
            created_at: Utc::now(),
        })
    }

    /// Execute an eDiscovery search
    pub async fn execute_search(
        &self,
        search_id: Uuid,
    ) -> Result<i64, ComplianceError> {
        // Get search parameters
        let search: EDiscoverySearch = sqlx::query_as(
            "SELECT * FROM ediscovery_searches WHERE id = $1",
        )
        .bind(search_id)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| ComplianceError::Database(e.to_string()))?;

        // Update status to running
        sqlx::query("UPDATE ediscovery_searches SET status = 'running' WHERE id = $1")
            .bind(search_id)
            .execute(&self.pool)
            .await
            .ok();

        // Build and execute search query
        let results = sqlx::query(
            r#"
            INSERT INTO ediscovery_results (search_id, email_id)
            SELECT $1, e.id
            FROM emails e
            JOIN users u ON u.id = e.user_id
            WHERE u.org_id = $2
            AND ($3::text IS NULL OR e.body_text ILIKE '%' || $3 || '%' OR e.subject ILIKE '%' || $3 || '%')
            AND ($4::timestamp IS NULL OR e.received_at >= $4)
            AND ($5::timestamp IS NULL OR e.received_at <= $5)
            AND ($6::uuid[] IS NULL OR e.user_id = ANY($6))
            "#,
        )
        .bind(search_id)
        .bind(search.org_id)
        .bind(&search.query)
        .bind(search.date_from)
        .bind(search.date_to)
        .bind(&search.custodians)
        .execute(&self.pool)
        .await
        .map_err(|e| ComplianceError::Database(e.to_string()))?;

        let count = results.rows_affected() as i64;

        // Update search with results
        sqlx::query(
            "UPDATE ediscovery_searches SET status = 'completed', result_count = $2, completed_at = NOW() WHERE id = $1",
        )
        .bind(search_id)
        .bind(count)
        .execute(&self.pool)
        .await
        .ok();

        Ok(count)
    }

    /// Export search results
    pub async fn export_results(
        &self,
        search_id: Uuid,
        format: ExportFormat,
    ) -> Result<ExportJob, ComplianceError> {
        let job_id = Uuid::new_v4();

        sqlx::query(
            r#"
            INSERT INTO export_jobs (id, search_id, format, status, created_at)
            VALUES ($1, $2, $3, 'pending', NOW())
            "#,
        )
        .bind(job_id)
        .bind(search_id)
        .bind(format.to_string())
        .execute(&self.pool)
        .await
        .map_err(|e| ComplianceError::Database(e.to_string()))?;

        // Spawn background export job
        let pool = self.pool.clone();
        tokio::spawn(async move {
            // Export logic would go here
            // Update job status when complete
        });

        Ok(ExportJob {
            id: job_id,
            search_id,
            format,
            status: ExportStatus::Pending,
            download_url: None,
            created_at: Utc::now(),
            completed_at: None,
        })
    }

    /// Get search results
    pub async fn get_results(
        &self,
        search_id: Uuid,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<SearchResult>, ComplianceError> {
        let results: Vec<SearchResult> = sqlx::query_as(
            r#"
            SELECT e.id as email_id, e.from_address, e.to_addresses, e.subject,
                   e.received_at, e.folder, u.email as custodian_email
            FROM ediscovery_results er
            JOIN emails e ON e.id = er.email_id
            JOIN users u ON u.id = e.user_id
            WHERE er.search_id = $1
            ORDER BY e.received_at DESC
            LIMIT $2 OFFSET $3
            "#,
        )
        .bind(search_id)
        .bind(limit)
        .bind(offset)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| ComplianceError::Database(e.to_string()))?;

        Ok(results)
    }
}

// ============================================================================
// Audit Log Service
// ============================================================================

pub struct AuditService {
    pool: PgPool,
}

impl AuditService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Log an audit event
    pub async fn log(
        &self,
        org_id: Uuid,
        user_id: Uuid,
        action: AuditAction,
        details: serde_json::Value,
    ) -> Result<(), ComplianceError> {
        sqlx::query(
            r#"
            INSERT INTO audit_log (id, org_id, user_id, action, details, ip_address, user_agent, created_at)
            VALUES ($1, $2, $3, $4, $5, NULL, NULL, NOW())
            "#,
        )
        .bind(Uuid::new_v4())
        .bind(org_id)
        .bind(user_id)
        .bind(action.to_string())
        .bind(details)
        .execute(&self.pool)
        .await
        .map_err(|e| ComplianceError::Database(e.to_string()))?;

        Ok(())
    }

    /// Query audit log
    pub async fn query(
        &self,
        org_id: Uuid,
        filter: AuditFilter,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<AuditEntry>, ComplianceError> {
        let entries: Vec<AuditEntry> = sqlx::query_as(
            r#"
            SELECT al.id, al.org_id, al.user_id, al.action, al.details,
                   al.ip_address, al.user_agent, al.created_at,
                   u.name as user_name, u.email as user_email
            FROM audit_log al
            LEFT JOIN users u ON u.id = al.user_id
            WHERE al.org_id = $1
            AND ($4::uuid IS NULL OR al.user_id = $4)
            AND ($5::text IS NULL OR al.action = $5)
            AND ($6::timestamp IS NULL OR al.created_at >= $6)
            AND ($7::timestamp IS NULL OR al.created_at <= $7)
            ORDER BY al.created_at DESC
            LIMIT $2 OFFSET $3
            "#,
        )
        .bind(org_id)
        .bind(limit)
        .bind(offset)
        .bind(filter.user_id)
        .bind(filter.action)
        .bind(filter.date_from)
        .bind(filter.date_to)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| ComplianceError::Database(e.to_string()))?;

        Ok(entries)
    }

    /// Export audit log
    pub async fn export(
        &self,
        org_id: Uuid,
        filter: AuditFilter,
    ) -> Result<String, ComplianceError> {
        let entries = self.query(org_id, filter, 100000, 0).await?;

        // Generate CSV
        let mut csv = String::from("Timestamp,User,Action,Details,IP Address\n");
        for entry in entries {
            csv.push_str(&format!(
                "{},{},{},{},{}\n",
                entry.created_at,
                entry.user_email.unwrap_or_default(),
                entry.action,
                serde_json::to_string(&entry.details).unwrap_or_default(),
                entry.ip_address.unwrap_or_default()
            ));
        }

        Ok(csv)
    }
}

// ============================================================================
// GDPR Data Export
// ============================================================================

pub struct GDPRService {
    pool: PgPool,
}

impl GDPRService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Export all user data (GDPR right to data portability)
    pub async fn export_user_data(&self, user_id: Uuid) -> Result<UserDataExport, ComplianceError> {
        // Get user profile
        let profile: Option<serde_json::Value> = sqlx::query_scalar(
            "SELECT row_to_json(u) FROM users u WHERE id = $1",
        )
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| ComplianceError::Database(e.to_string()))?;

        // Get emails count
        let email_count: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM emails WHERE user_id = $1",
        )
        .bind(user_id)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| ComplianceError::Database(e.to_string()))?;

        // Get contacts count
        let contact_count: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM contacts WHERE user_id = $1",
        )
        .bind(user_id)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| ComplianceError::Database(e.to_string()))?;

        Ok(UserDataExport {
            user_id,
            profile,
            email_count: email_count.0,
            contact_count: contact_count.0,
            export_date: Utc::now(),
        })
    }

    /// Delete all user data (GDPR right to erasure)
    pub async fn delete_user_data(&self, user_id: Uuid) -> Result<DeletionReport, ComplianceError> {
        let mut report = DeletionReport::default();

        // Delete emails
        let result = sqlx::query("DELETE FROM emails WHERE user_id = $1")
            .bind(user_id)
            .execute(&self.pool)
            .await
            .map_err(|e| ComplianceError::Database(e.to_string()))?;
        report.emails_deleted = result.rows_affected() as i64;

        // Delete contacts
        let result = sqlx::query("DELETE FROM contacts WHERE user_id = $1")
            .bind(user_id)
            .execute(&self.pool)
            .await
            .map_err(|e| ComplianceError::Database(e.to_string()))?;
        report.contacts_deleted = result.rows_affected() as i64;

        // Anonymize user record (keep for audit purposes)
        sqlx::query(
            r#"
            UPDATE users SET
                email = 'deleted_' || id::text || '@deleted.local',
                name = 'Deleted User',
                deleted_at = NOW()
            WHERE id = $1
            "#,
        )
        .bind(user_id)
        .execute(&self.pool)
        .await
        .map_err(|e| ComplianceError::Database(e.to_string()))?;

        report.user_anonymized = true;
        report.deletion_date = Utc::now();

        Ok(report)
    }
}

// ============================================================================
// Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicyInput {
    pub name: String,
    pub description: Option<String>,
    pub retention_days: i32,
    pub applies_to: RetentionAppliesTo,
    pub action: RetentionAction,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionAppliesTo {
    pub folders: Vec<String>,
    pub labels: Vec<String>,
    pub exclude_starred: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetentionAction {
    Delete,
    Archive,
    Move { folder: String },
}

impl ToString for RetentionAction {
    fn to_string(&self) -> String {
        match self {
            RetentionAction::Delete => "delete".to_string(),
            RetentionAction::Archive => "archive".to_string(),
            RetentionAction::Move { folder } => format!("move:{}", folder),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct RetentionPolicy {
    pub id: Uuid,
    pub org_id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub retention_days: i32,
    pub applies_to: serde_json::Value,
    pub action: String,
    pub enabled: bool,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetentionResult {
    pub policies_applied: i32,
    pub emails_affected: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegalHoldInput {
    pub name: String,
    pub description: Option<String>,
    pub matter_id: Option<String>,
    pub custodians: Vec<Uuid>,
    pub start_date: DateTime<Utc>,
    pub end_date: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct LegalHold {
    pub id: Uuid,
    pub org_id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub matter_id: Option<String>,
    pub custodians: Vec<Uuid>,
    pub start_date: DateTime<Utc>,
    pub end_date: Option<DateTime<Utc>>,
    pub status: HoldStatus,
    pub created_by: Uuid,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "text")]
pub enum HoldStatus {
    Active,
    Released,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EDiscoverySearchInput {
    pub name: String,
    pub description: Option<String>,
    pub matter_id: Option<String>,
    pub query: String,
    pub date_from: Option<DateTime<Utc>>,
    pub date_to: Option<DateTime<Utc>>,
    pub custodians: Vec<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct EDiscoverySearch {
    pub id: Uuid,
    pub org_id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub matter_id: Option<String>,
    pub query: String,
    pub date_from: Option<DateTime<Utc>>,
    pub date_to: Option<DateTime<Utc>>,
    pub custodians: Vec<Uuid>,
    pub status: SearchStatus,
    pub result_count: Option<i64>,
    pub created_by: Uuid,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "text")]
pub enum SearchStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct SearchResult {
    pub email_id: Uuid,
    pub from_address: String,
    pub to_addresses: Vec<String>,
    pub subject: String,
    pub received_at: DateTime<Utc>,
    pub folder: String,
    pub custodian_email: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ExportFormat {
    PST,
    MBOX,
    EML,
    PDF,
}

impl ToString for ExportFormat {
    fn to_string(&self) -> String {
        match self {
            ExportFormat::PST => "pst".to_string(),
            ExportFormat::MBOX => "mbox".to_string(),
            ExportFormat::EML => "eml".to_string(),
            ExportFormat::PDF => "pdf".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportJob {
    pub id: Uuid,
    pub search_id: Uuid,
    pub format: ExportFormat,
    pub status: ExportStatus,
    pub download_url: Option<String>,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ExportStatus {
    Pending,
    Processing,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditAction {
    Login,
    Logout,
    EmailRead,
    EmailSent,
    EmailDeleted,
    SettingsChanged,
    LegalHoldCreated,
    LegalHoldReleased,
    SearchExecuted,
    DataExported,
}

impl ToString for AuditAction {
    fn to_string(&self) -> String {
        match self {
            AuditAction::Login => "login".to_string(),
            AuditAction::Logout => "logout".to_string(),
            AuditAction::EmailRead => "email_read".to_string(),
            AuditAction::EmailSent => "email_sent".to_string(),
            AuditAction::EmailDeleted => "email_deleted".to_string(),
            AuditAction::SettingsChanged => "settings_changed".to_string(),
            AuditAction::LegalHoldCreated => "legal_hold_created".to_string(),
            AuditAction::LegalHoldReleased => "legal_hold_released".to_string(),
            AuditAction::SearchExecuted => "search_executed".to_string(),
            AuditAction::DataExported => "data_exported".to_string(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuditFilter {
    pub user_id: Option<Uuid>,
    pub action: Option<String>,
    pub date_from: Option<DateTime<Utc>>,
    pub date_to: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct AuditEntry {
    pub id: Uuid,
    pub org_id: Uuid,
    pub user_id: Uuid,
    pub action: String,
    pub details: serde_json::Value,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub created_at: DateTime<Utc>,
    pub user_name: Option<String>,
    pub user_email: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserDataExport {
    pub user_id: Uuid,
    pub profile: Option<serde_json::Value>,
    pub email_count: i64,
    pub contact_count: i64,
    pub export_date: DateTime<Utc>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeletionReport {
    pub emails_deleted: i64,
    pub contacts_deleted: i64,
    pub user_anonymized: bool,
    pub deletion_date: DateTime<Utc>,
}

#[derive(Debug, thiserror::Error)]
pub enum ComplianceError {
    #[error("Database error: {0}")]
    Database(String),
    #[error("Not authorized")]
    Unauthorized,
    #[error("Policy not found")]
    PolicyNotFound,
    #[error("Hold not found")]
    HoldNotFound,
}
