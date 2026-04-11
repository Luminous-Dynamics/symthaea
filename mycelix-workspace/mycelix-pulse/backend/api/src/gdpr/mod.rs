// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! GDPR Compliance
//!
//! Data export, deletion, consent management, and retention policies

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::io::Write;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tracing::{info, warn, error};
use uuid::Uuid;
use zip::write::FileOptions;
use zip::ZipWriter;

/// GDPR request types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "gdpr_request_type", rename_all = "lowercase")]
pub enum GdprRequestType {
    Export,
    Deletion,
    Rectification,
    Restriction,
}

/// GDPR request status
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "gdpr_request_status", rename_all = "lowercase")]
pub enum GdprRequestStatus {
    Pending,
    Processing,
    Completed,
    Failed,
    Cancelled,
}

/// GDPR request record
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct GdprRequest {
    pub id: Uuid,
    pub user_id: Uuid,
    pub request_type: GdprRequestType,
    pub status: GdprRequestStatus,
    pub reason: Option<String>,
    pub download_url: Option<String>,
    pub download_expires_at: Option<DateTime<Utc>>,
    pub error_message: Option<String>,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

/// User consent record
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct UserConsent {
    pub id: Uuid,
    pub user_id: Uuid,
    pub consent_type: String,
    pub granted: bool,
    pub granted_at: Option<DateTime<Utc>>,
    pub revoked_at: Option<DateTime<Utc>>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
}

/// Data retention policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    pub name: String,
    pub description: String,
    pub table_name: String,
    pub retention_days: u32,
    pub action: RetentionAction,
    pub enabled: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RetentionAction {
    Delete,
    Archive,
    Anonymize,
}

/// GDPR service
pub struct GdprService {
    pool: PgPool,
    export_path: String,
}

impl GdprService {
    pub fn new(pool: PgPool, export_path: &str) -> Self {
        Self {
            pool,
            export_path: export_path.to_string(),
        }
    }

    // ============ DATA EXPORT ============

    /// Request data export
    pub async fn request_export(&self, user_id: Uuid) -> Result<GdprRequest, GdprError> {
        let id = Uuid::new_v4();

        let request = sqlx::query_as::<_, GdprRequest>(
            r#"
            INSERT INTO gdpr_requests (id, user_id, request_type, status, created_at)
            VALUES ($1, $2, 'export', 'pending', NOW())
            RETURNING *
            "#,
        )
        .bind(id)
        .bind(user_id)
        .fetch_one(&self.pool)
        .await
        .map_err(GdprError::Database)?;

        info!(request_id = %id, user_id = %user_id, "GDPR export requested");

        Ok(request)
    }

    /// Process data export
    pub async fn process_export(&self, request_id: Uuid) -> Result<String, GdprError> {
        // Get request
        let request = sqlx::query_as::<_, GdprRequest>(
            "SELECT * FROM gdpr_requests WHERE id = $1",
        )
        .bind(request_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(GdprError::Database)?
        .ok_or(GdprError::NotFound)?;

        // Mark as processing
        sqlx::query("UPDATE gdpr_requests SET status = 'processing' WHERE id = $1")
            .bind(request_id)
            .execute(&self.pool)
            .await
            .map_err(GdprError::Database)?;

        let user_id = request.user_id;

        // Collect all user data
        let export_data = self.collect_user_data(user_id).await?;

        // Create ZIP archive
        let filename = format!("export_{}_{}.zip", user_id, Utc::now().timestamp());
        let filepath = format!("{}/{}", self.export_path, filename);

        self.create_export_archive(&filepath, &export_data).await?;

        // Generate download URL (expires in 7 days)
        let download_url = format!("/api/gdpr/exports/{}", filename);
        let expires_at = Utc::now() + Duration::days(7);

        // Update request
        sqlx::query(
            r#"
            UPDATE gdpr_requests
            SET status = 'completed',
                download_url = $2,
                download_expires_at = $3,
                completed_at = NOW()
            WHERE id = $1
            "#,
        )
        .bind(request_id)
        .bind(&download_url)
        .bind(expires_at)
        .execute(&self.pool)
        .await
        .map_err(GdprError::Database)?;

        info!(request_id = %request_id, "GDPR export completed");

        Ok(download_url)
    }

    /// Collect all user data for export
    async fn collect_user_data(&self, user_id: Uuid) -> Result<ExportData, GdprError> {
        // User profile
        let profile: Option<serde_json::Value> = sqlx::query_scalar(
            "SELECT row_to_json(u) FROM users u WHERE id = $1",
        )
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(GdprError::Database)?;

        // Emails (limited to prevent huge exports)
        let emails: Vec<serde_json::Value> = sqlx::query_scalar(
            r#"
            SELECT json_agg(e) FROM (
                SELECT id, subject, body_text, from_address, to_addresses,
                       is_read, is_starred, labels, received_at, sent_at
                FROM emails
                WHERE user_id = $1
                ORDER BY received_at DESC
                LIMIT 10000
            ) e
            "#,
        )
        .bind(user_id)
        .fetch_all(&self.pool)
        .await
        .map_err(GdprError::Database)?;

        // Contacts
        let contacts: Vec<serde_json::Value> = sqlx::query_scalar(
            r#"
            SELECT json_agg(c) FROM (
                SELECT email, display_name, notes, trust_score, interaction_count
                FROM contacts
                WHERE user_id = $1
            ) c
            "#,
        )
        .bind(user_id)
        .fetch_all(&self.pool)
        .await
        .map_err(GdprError::Database)?;

        // Consents
        let consents: Vec<serde_json::Value> = sqlx::query_scalar(
            "SELECT json_agg(uc) FROM user_consents uc WHERE user_id = $1",
        )
        .bind(user_id)
        .fetch_all(&self.pool)
        .await
        .map_err(GdprError::Database)?;

        // Activity logs
        let activity: Vec<serde_json::Value> = sqlx::query_scalar(
            r#"
            SELECT json_agg(a) FROM (
                SELECT action, resource_type, timestamp, ip_address
                FROM audit_logs
                WHERE user_id = $1
                ORDER BY timestamp DESC
                LIMIT 1000
            ) a
            "#,
        )
        .bind(user_id)
        .fetch_all(&self.pool)
        .await
        .map_err(GdprError::Database)?;

        Ok(ExportData {
            profile,
            emails,
            contacts,
            consents,
            activity,
            exported_at: Utc::now(),
        })
    }

    /// Create export ZIP archive
    async fn create_export_archive(
        &self,
        filepath: &str,
        data: &ExportData,
    ) -> Result<(), GdprError> {
        let file = std::fs::File::create(filepath)
            .map_err(|e| GdprError::Io(e.to_string()))?;

        let mut zip = ZipWriter::new(file);
        let options = FileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);

        // Write each section as JSON file
        if let Some(profile) = &data.profile {
            zip.start_file("profile.json", options)
                .map_err(|e| GdprError::Io(e.to_string()))?;
            zip.write_all(serde_json::to_string_pretty(profile).unwrap().as_bytes())
                .map_err(|e| GdprError::Io(e.to_string()))?;
        }

        zip.start_file("emails.json", options)
            .map_err(|e| GdprError::Io(e.to_string()))?;
        zip.write_all(serde_json::to_string_pretty(&data.emails).unwrap().as_bytes())
            .map_err(|e| GdprError::Io(e.to_string()))?;

        zip.start_file("contacts.json", options)
            .map_err(|e| GdprError::Io(e.to_string()))?;
        zip.write_all(serde_json::to_string_pretty(&data.contacts).unwrap().as_bytes())
            .map_err(|e| GdprError::Io(e.to_string()))?;

        zip.start_file("consents.json", options)
            .map_err(|e| GdprError::Io(e.to_string()))?;
        zip.write_all(serde_json::to_string_pretty(&data.consents).unwrap().as_bytes())
            .map_err(|e| GdprError::Io(e.to_string()))?;

        zip.start_file("activity.json", options)
            .map_err(|e| GdprError::Io(e.to_string()))?;
        zip.write_all(serde_json::to_string_pretty(&data.activity).unwrap().as_bytes())
            .map_err(|e| GdprError::Io(e.to_string()))?;

        // Metadata
        let metadata = serde_json::json!({
            "exported_at": data.exported_at,
            "format_version": "1.0",
            "service": "Mycelix Mail"
        });
        zip.start_file("metadata.json", options)
            .map_err(|e| GdprError::Io(e.to_string()))?;
        zip.write_all(serde_json::to_string_pretty(&metadata).unwrap().as_bytes())
            .map_err(|e| GdprError::Io(e.to_string()))?;

        zip.finish().map_err(|e| GdprError::Io(e.to_string()))?;

        Ok(())
    }

    // ============ RIGHT TO DELETION ============

    /// Request account deletion
    pub async fn request_deletion(
        &self,
        user_id: Uuid,
        reason: Option<&str>,
    ) -> Result<GdprRequest, GdprError> {
        let id = Uuid::new_v4();

        let request = sqlx::query_as::<_, GdprRequest>(
            r#"
            INSERT INTO gdpr_requests (id, user_id, request_type, status, reason, created_at)
            VALUES ($1, $2, 'deletion', 'pending', $3, NOW())
            RETURNING *
            "#,
        )
        .bind(id)
        .bind(user_id)
        .bind(reason)
        .fetch_one(&self.pool)
        .await
        .map_err(GdprError::Database)?;

        info!(request_id = %id, user_id = %user_id, "GDPR deletion requested");

        Ok(request)
    }

    /// Process account deletion
    pub async fn process_deletion(&self, request_id: Uuid) -> Result<(), GdprError> {
        let request = sqlx::query_as::<_, GdprRequest>(
            "SELECT * FROM gdpr_requests WHERE id = $1",
        )
        .bind(request_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(GdprError::Database)?
        .ok_or(GdprError::NotFound)?;

        let user_id = request.user_id;

        // Mark as processing
        sqlx::query("UPDATE gdpr_requests SET status = 'processing' WHERE id = $1")
            .bind(request_id)
            .execute(&self.pool)
            .await
            .map_err(GdprError::Database)?;

        // Delete in order (respecting foreign keys)
        let tables = [
            "audit_logs",
            "mail_merge_recipients",
            "mail_merge_jobs",
            "email_templates",
            "scheduled_emails",
            "attachments",
            "emails",
            "contacts",
            "trust_attestations",
            "user_consents",
            "tenant_members",
        ];

        for table in tables {
            let query = format!("DELETE FROM {} WHERE user_id = $1", table);
            sqlx::query(&query)
                .bind(user_id)
                .execute(&self.pool)
                .await
                .map_err(GdprError::Database)?;
        }

        // Finally delete user (anonymize rather than hard delete for audit)
        sqlx::query(
            r#"
            UPDATE users
            SET email = CONCAT('deleted_', id::text, '@deleted.local'),
                name = 'Deleted User',
                password_hash = NULL,
                is_deleted = true,
                deleted_at = NOW()
            WHERE id = $1
            "#,
        )
        .bind(user_id)
        .execute(&self.pool)
        .await
        .map_err(GdprError::Database)?;

        // Mark request complete
        sqlx::query(
            "UPDATE gdpr_requests SET status = 'completed', completed_at = NOW() WHERE id = $1",
        )
        .bind(request_id)
        .execute(&self.pool)
        .await
        .map_err(GdprError::Database)?;

        info!(request_id = %request_id, user_id = %user_id, "GDPR deletion completed");

        Ok(())
    }

    // ============ CONSENT MANAGEMENT ============

    /// Record user consent
    pub async fn record_consent(
        &self,
        user_id: Uuid,
        consent_type: &str,
        granted: bool,
        ip_address: Option<&str>,
        user_agent: Option<&str>,
    ) -> Result<UserConsent, GdprError> {
        let id = Uuid::new_v4();

        let consent = sqlx::query_as::<_, UserConsent>(
            r#"
            INSERT INTO user_consents (
                id, user_id, consent_type, granted,
                granted_at, ip_address, user_agent
            )
            VALUES ($1, $2, $3, $4, CASE WHEN $4 THEN NOW() ELSE NULL END, $5, $6)
            ON CONFLICT (user_id, consent_type)
            DO UPDATE SET
                granted = EXCLUDED.granted,
                granted_at = CASE WHEN EXCLUDED.granted THEN NOW() ELSE user_consents.granted_at END,
                revoked_at = CASE WHEN NOT EXCLUDED.granted THEN NOW() ELSE NULL END,
                ip_address = EXCLUDED.ip_address,
                user_agent = EXCLUDED.user_agent
            RETURNING *
            "#,
        )
        .bind(id)
        .bind(user_id)
        .bind(consent_type)
        .bind(granted)
        .bind(ip_address)
        .bind(user_agent)
        .fetch_one(&self.pool)
        .await
        .map_err(GdprError::Database)?;

        info!(
            user_id = %user_id,
            consent_type = consent_type,
            granted = granted,
            "Consent recorded"
        );

        Ok(consent)
    }

    /// Get user consents
    pub async fn get_user_consents(&self, user_id: Uuid) -> Result<Vec<UserConsent>, GdprError> {
        sqlx::query_as::<_, UserConsent>(
            "SELECT * FROM user_consents WHERE user_id = $1",
        )
        .bind(user_id)
        .fetch_all(&self.pool)
        .await
        .map_err(GdprError::Database)
    }

    /// Check if user has consent
    pub async fn has_consent(
        &self,
        user_id: Uuid,
        consent_type: &str,
    ) -> Result<bool, GdprError> {
        let granted: Option<bool> = sqlx::query_scalar(
            "SELECT granted FROM user_consents WHERE user_id = $1 AND consent_type = $2",
        )
        .bind(user_id)
        .bind(consent_type)
        .fetch_optional(&self.pool)
        .await
        .map_err(GdprError::Database)?;

        Ok(granted.unwrap_or(false))
    }

    // ============ DATA RETENTION ============

    /// Apply retention policies
    pub async fn apply_retention_policies(&self) -> Result<RetentionResult, GdprError> {
        let policies = self.get_retention_policies();
        let mut result = RetentionResult::default();

        for policy in policies {
            if !policy.enabled {
                continue;
            }

            let cutoff = Utc::now() - Duration::days(policy.retention_days as i64);

            match policy.action {
                RetentionAction::Delete => {
                    let query = format!(
                        "DELETE FROM {} WHERE created_at < $1",
                        policy.table_name
                    );
                    let deleted = sqlx::query(&query)
                        .bind(cutoff)
                        .execute(&self.pool)
                        .await
                        .map_err(GdprError::Database)?;

                    result.deleted += deleted.rows_affected() as u64;
                }
                RetentionAction::Archive => {
                    // Move to archive table
                    let query = format!(
                        r#"
                        WITH moved AS (
                            DELETE FROM {}
                            WHERE created_at < $1
                            RETURNING *
                        )
                        INSERT INTO {}_archive SELECT * FROM moved
                        "#,
                        policy.table_name, policy.table_name
                    );
                    let archived = sqlx::query(&query)
                        .bind(cutoff)
                        .execute(&self.pool)
                        .await
                        .map_err(GdprError::Database)?;

                    result.archived += archived.rows_affected() as u64;
                }
                RetentionAction::Anonymize => {
                    // Anonymize personal data
                    let query = format!(
                        r#"
                        UPDATE {}
                        SET email = CONCAT('anon_', id::text),
                            name = 'Anonymous'
                        WHERE created_at < $1 AND email NOT LIKE 'anon_%'
                        "#,
                        policy.table_name
                    );
                    let anonymized = sqlx::query(&query)
                        .bind(cutoff)
                        .execute(&self.pool)
                        .await
                        .map_err(GdprError::Database)?;

                    result.anonymized += anonymized.rows_affected() as u64;
                }
            }
        }

        info!(
            deleted = result.deleted,
            archived = result.archived,
            anonymized = result.anonymized,
            "Retention policies applied"
        );

        Ok(result)
    }

    /// Get retention policies
    fn get_retention_policies(&self) -> Vec<RetentionPolicy> {
        vec![
            RetentionPolicy {
                name: "Audit Log Retention".to_string(),
                description: "Delete audit logs older than 2 years".to_string(),
                table_name: "audit_logs".to_string(),
                retention_days: 730,
                action: RetentionAction::Delete,
                enabled: true,
            },
            RetentionPolicy {
                name: "Deleted Emails Purge".to_string(),
                description: "Permanently delete trashed emails after 30 days".to_string(),
                table_name: "emails".to_string(),
                retention_days: 30,
                action: RetentionAction::Delete,
                enabled: true,
            },
            RetentionPolicy {
                name: "Session Archive".to_string(),
                description: "Archive sessions older than 90 days".to_string(),
                table_name: "sessions".to_string(),
                retention_days: 90,
                action: RetentionAction::Archive,
                enabled: true,
            },
        ]
    }
}

#[derive(Debug, Clone, Serialize)]
struct ExportData {
    profile: Option<serde_json::Value>,
    emails: Vec<serde_json::Value>,
    contacts: Vec<serde_json::Value>,
    consents: Vec<serde_json::Value>,
    activity: Vec<serde_json::Value>,
    exported_at: DateTime<Utc>,
}

#[derive(Debug, Default)]
pub struct RetentionResult {
    pub deleted: u64,
    pub archived: u64,
    pub anonymized: u64,
}

/// GDPR errors
#[derive(Debug)]
pub enum GdprError {
    NotFound,
    Database(sqlx::Error),
    Io(String),
}

impl std::fmt::Display for GdprError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => write!(f, "Request not found"),
            Self::Database(e) => write!(f, "Database error: {}", e),
            Self::Io(msg) => write!(f, "IO error: {}", msg),
        }
    }
}

impl std::error::Error for GdprError {}

/// GDPR migrations
pub const GDPR_MIGRATION: &str = r#"
DO $$ BEGIN
    CREATE TYPE gdpr_request_type AS ENUM ('export', 'deletion', 'rectification', 'restriction');
EXCEPTION WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE gdpr_request_status AS ENUM ('pending', 'processing', 'completed', 'failed', 'cancelled');
EXCEPTION WHEN duplicate_object THEN null;
END $$;

CREATE TABLE IF NOT EXISTS gdpr_requests (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id),
    request_type gdpr_request_type NOT NULL,
    status gdpr_request_status NOT NULL DEFAULT 'pending',
    reason TEXT,
    download_url TEXT,
    download_expires_at TIMESTAMPTZ,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_gdpr_requests_user ON gdpr_requests(user_id);
CREATE INDEX IF NOT EXISTS idx_gdpr_requests_status ON gdpr_requests(status);

CREATE TABLE IF NOT EXISTS user_consents (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    consent_type VARCHAR(100) NOT NULL,
    granted BOOLEAN NOT NULL DEFAULT false,
    granted_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ,
    ip_address VARCHAR(45),
    user_agent TEXT,
    UNIQUE (user_id, consent_type)
);

CREATE INDEX IF NOT EXISTS idx_user_consents_user ON user_consents(user_id);

-- Add is_deleted and deleted_at to users table
ALTER TABLE users ADD COLUMN IF NOT EXISTS is_deleted BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE users ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;
"#;
