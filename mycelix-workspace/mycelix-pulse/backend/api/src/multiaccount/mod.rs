// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Multi-Account & Unified Inbox Module
 *
 * Multiple email accounts, unified view, send-as, cross-account search
 */

use sqlx::PgPool;
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Debug, thiserror::Error)]
pub enum AccountError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("Account not found")]
    NotFound,
    #[error("Account limit exceeded")]
    LimitExceeded,
    #[error("Invalid credentials")]
    InvalidCredentials,
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    #[error("Account already exists")]
    AlreadyExists,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmailAccount {
    pub id: Uuid,
    pub user_id: Uuid,
    pub email_address: String,
    pub display_name: Option<String>,
    pub account_type: AccountType,
    pub provider: Option<String>,
    pub is_primary: bool,
    pub is_enabled: bool,
    pub color: Option<String>,
    pub sync_settings: SyncSettings,
    pub last_sync_at: Option<DateTime<Utc>>,
    pub sync_status: SyncStatus,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, sqlx::Type)]
#[sqlx(type_name = "account_type", rename_all = "snake_case")]
pub enum AccountType {
    Imap,
    Exchange,
    Gmail,
    Outlook,
    Custom,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, sqlx::Type)]
#[sqlx(type_name = "sync_status", rename_all = "snake_case")]
pub enum SyncStatus {
    Active,
    Syncing,
    Error,
    Paused,
    Disconnected,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SyncSettings {
    pub sync_period_days: i32,
    pub sync_folders: Vec<String>,
    pub exclude_folders: Vec<String>,
    pub sync_interval_minutes: i32,
    pub download_attachments: bool,
    pub max_attachment_size_mb: i32,
}

impl Default for SyncSettings {
    fn default() -> Self {
        Self {
            sync_period_days: 30,
            sync_folders: vec!["INBOX".to_string(), "Sent".to_string()],
            exclude_folders: vec!["Spam".to_string(), "Trash".to_string()],
            sync_interval_minutes: 5,
            download_attachments: true,
            max_attachment_size_mb: 25,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AccountCredentials {
    pub imap_host: Option<String>,
    pub imap_port: Option<i32>,
    pub imap_username: Option<String>,
    pub imap_password: Option<String>,
    pub smtp_host: Option<String>,
    pub smtp_port: Option<i32>,
    pub smtp_username: Option<String>,
    pub smtp_password: Option<String>,
    pub oauth_token: Option<String>,
    pub oauth_refresh_token: Option<String>,
    pub oauth_expires_at: Option<DateTime<Utc>>,
    pub use_ssl: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AddAccountInput {
    pub email_address: String,
    pub display_name: Option<String>,
    pub account_type: AccountType,
    pub credentials: AccountCredentials,
    pub color: Option<String>,
    pub sync_settings: Option<SyncSettings>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SendAsIdentity {
    pub id: Uuid,
    pub account_id: Uuid,
    pub email_address: String,
    pub display_name: Option<String>,
    pub signature_id: Option<Uuid>,
    pub is_default: bool,
    pub reply_to: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UnifiedEmail {
    pub id: Uuid,
    pub account_id: Uuid,
    pub account_email: String,
    pub account_color: Option<String>,
    pub subject: String,
    pub from_address: String,
    pub from_name: Option<String>,
    pub received_at: DateTime<Utc>,
    pub is_read: bool,
    pub is_starred: bool,
    pub has_attachments: bool,
    pub folder: String,
    pub snippet: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AccountStats {
    pub account_id: Uuid,
    pub email_address: String,
    pub total_emails: i64,
    pub unread_count: i64,
    pub storage_used_bytes: i64,
    pub last_sync_at: Option<DateTime<Utc>>,
    pub sync_status: SyncStatus,
}

pub struct MultiAccountService {
    pool: PgPool,
}

impl MultiAccountService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn add_account(&self, user_id: Uuid, input: AddAccountInput) -> Result<EmailAccount, AccountError> {
        // Check account limit
        let count: i64 = sqlx::query_scalar!("SELECT COUNT(*) FROM email_accounts WHERE user_id = $1", user_id)
            .fetch_one(&self.pool).await?.unwrap_or(0);
        if count >= 10 { return Err(AccountError::LimitExceeded); }

        // Check for duplicate
        let exists: bool = sqlx::query_scalar!("SELECT EXISTS(SELECT 1 FROM email_accounts WHERE user_id = $1 AND email_address = $2)", user_id, input.email_address)
            .fetch_one(&self.pool).await?.unwrap_or(false);
        if exists { return Err(AccountError::AlreadyExists); }

        // Test connection
        self.test_connection(&input.account_type, &input.credentials).await?;

        let id = Uuid::new_v4();
        let is_primary = count == 0;
        let sync_settings = input.sync_settings.unwrap_or_default();
        let sync_json = serde_json::to_value(&sync_settings).unwrap();
        let creds_json = serde_json::to_value(&input.credentials).unwrap();

        sqlx::query!(
            r#"INSERT INTO email_accounts (id, user_id, email_address, display_name, account_type, is_primary, is_enabled, color, sync_settings, credentials, sync_status, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, true, $7, $8, $9, 'active', NOW())"#,
            id, user_id, input.email_address, input.display_name, input.account_type as AccountType,
            is_primary, input.color, sync_json, creds_json
        ).execute(&self.pool).await?;

        self.get_account(user_id, id).await
    }

    pub async fn get_account(&self, user_id: Uuid, account_id: Uuid) -> Result<EmailAccount, AccountError> {
        let row = sqlx::query!(
            r#"SELECT id, user_id, email_address, display_name, account_type as "account_type: AccountType",
                      provider, is_primary, is_enabled, color, sync_settings, last_sync_at,
                      sync_status as "sync_status: SyncStatus", created_at
               FROM email_accounts WHERE id = $1 AND user_id = $2"#,
            account_id, user_id
        ).fetch_optional(&self.pool).await?.ok_or(AccountError::NotFound)?;

        let sync_settings: SyncSettings = serde_json::from_value(row.sync_settings).unwrap_or_default();

        Ok(EmailAccount {
            id: row.id, user_id: row.user_id, email_address: row.email_address,
            display_name: row.display_name, account_type: row.account_type, provider: row.provider,
            is_primary: row.is_primary, is_enabled: row.is_enabled, color: row.color,
            sync_settings, last_sync_at: row.last_sync_at, sync_status: row.sync_status,
            created_at: row.created_at,
        })
    }

    pub async fn list_accounts(&self, user_id: Uuid) -> Result<Vec<EmailAccount>, AccountError> {
        let rows = sqlx::query!(
            r#"SELECT id, user_id, email_address, display_name, account_type as "account_type: AccountType",
                      provider, is_primary, is_enabled, color, sync_settings, last_sync_at,
                      sync_status as "sync_status: SyncStatus", created_at
               FROM email_accounts WHERE user_id = $1 ORDER BY is_primary DESC, created_at"#,
            user_id
        ).fetch_all(&self.pool).await?;

        Ok(rows.into_iter().map(|r| {
            let sync_settings: SyncSettings = serde_json::from_value(r.sync_settings).unwrap_or_default();
            EmailAccount {
                id: r.id, user_id: r.user_id, email_address: r.email_address,
                display_name: r.display_name, account_type: r.account_type, provider: r.provider,
                is_primary: r.is_primary, is_enabled: r.is_enabled, color: r.color,
                sync_settings, last_sync_at: r.last_sync_at, sync_status: r.sync_status,
                created_at: r.created_at,
            }
        }).collect())
    }

    pub async fn set_primary(&self, user_id: Uuid, account_id: Uuid) -> Result<(), AccountError> {
        sqlx::query!("UPDATE email_accounts SET is_primary = false WHERE user_id = $1", user_id).execute(&self.pool).await?;
        let result = sqlx::query!("UPDATE email_accounts SET is_primary = true WHERE id = $1 AND user_id = $2", account_id, user_id).execute(&self.pool).await?;
        if result.rows_affected() == 0 { return Err(AccountError::NotFound); }
        Ok(())
    }

    pub async fn toggle_account(&self, user_id: Uuid, account_id: Uuid, is_enabled: bool) -> Result<(), AccountError> {
        let result = sqlx::query!("UPDATE email_accounts SET is_enabled = $3 WHERE id = $1 AND user_id = $2", account_id, user_id, is_enabled).execute(&self.pool).await?;
        if result.rows_affected() == 0 { return Err(AccountError::NotFound); }
        Ok(())
    }

    pub async fn update_sync_settings(&self, user_id: Uuid, account_id: Uuid, settings: SyncSettings) -> Result<(), AccountError> {
        let json = serde_json::to_value(&settings).unwrap();
        let result = sqlx::query!("UPDATE email_accounts SET sync_settings = $3 WHERE id = $1 AND user_id = $2", account_id, user_id, json).execute(&self.pool).await?;
        if result.rows_affected() == 0 { return Err(AccountError::NotFound); }
        Ok(())
    }

    pub async fn delete_account(&self, user_id: Uuid, account_id: Uuid) -> Result<(), AccountError> {
        // Delete associated emails
        sqlx::query!("DELETE FROM emails WHERE account_id = $1 AND user_id = $2", account_id, user_id).execute(&self.pool).await?;

        let result = sqlx::query!("DELETE FROM email_accounts WHERE id = $1 AND user_id = $2", account_id, user_id).execute(&self.pool).await?;
        if result.rows_affected() == 0 { return Err(AccountError::NotFound); }
        Ok(())
    }

    pub async fn get_account_stats(&self, user_id: Uuid) -> Result<Vec<AccountStats>, AccountError> {
        let rows = sqlx::query!(
            r#"SELECT a.id, a.email_address, a.last_sync_at, a.sync_status as "sync_status: SyncStatus",
                      COALESCE(COUNT(e.id), 0) as total_emails,
                      COALESCE(SUM(CASE WHEN e.is_read = false THEN 1 ELSE 0 END), 0) as unread_count,
                      COALESCE(SUM(e.size_bytes), 0) as storage_used
               FROM email_accounts a
               LEFT JOIN emails e ON e.account_id = a.id
               WHERE a.user_id = $1
               GROUP BY a.id"#,
            user_id
        ).fetch_all(&self.pool).await?;

        Ok(rows.into_iter().map(|r| AccountStats {
            account_id: r.id, email_address: r.email_address,
            total_emails: r.total_emails.unwrap_or(0), unread_count: r.unread_count.unwrap_or(0),
            storage_used_bytes: r.storage_used.unwrap_or(0), last_sync_at: r.last_sync_at,
            sync_status: r.sync_status,
        }).collect())
    }

    async fn test_connection(&self, account_type: &AccountType, credentials: &AccountCredentials) -> Result<(), AccountError> {
        match account_type {
            AccountType::Imap => {
                let host = credentials.imap_host.as_ref().ok_or(AccountError::InvalidCredentials)?;
                let port = credentials.imap_port.unwrap_or(993);
                // In production: actually test IMAP connection
                if host.is_empty() { return Err(AccountError::InvalidCredentials); }
                Ok(())
            }
            AccountType::Gmail | AccountType::Outlook => {
                if credentials.oauth_token.is_none() { return Err(AccountError::InvalidCredentials); }
                Ok(())
            }
            _ => Ok(())
        }
    }

    pub async fn mark_sync_started(&self, account_id: Uuid) -> Result<(), AccountError> {
        sqlx::query!("UPDATE email_accounts SET sync_status = 'syncing' WHERE id = $1", account_id).execute(&self.pool).await?;
        Ok(())
    }

    pub async fn mark_sync_completed(&self, account_id: Uuid) -> Result<(), AccountError> {
        sqlx::query!("UPDATE email_accounts SET sync_status = 'active', last_sync_at = NOW() WHERE id = $1", account_id).execute(&self.pool).await?;
        Ok(())
    }

    pub async fn mark_sync_failed(&self, account_id: Uuid, error: &str) -> Result<(), AccountError> {
        sqlx::query!("UPDATE email_accounts SET sync_status = 'error' WHERE id = $1", account_id).execute(&self.pool).await?;
        Ok(())
    }
}

pub struct UnifiedInboxService {
    pool: PgPool,
}

impl UnifiedInboxService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn get_unified_inbox(&self, user_id: Uuid, limit: i32, offset: i32, folder: Option<&str>) -> Result<Vec<UnifiedEmail>, AccountError> {
        let rows = sqlx::query!(
            r#"SELECT e.id, e.account_id, a.email_address as account_email, a.color as account_color,
                      e.subject, e.from_address, e.from_name, e.received_at, e.is_read, e.is_starred,
                      e.has_attachments, e.folder, LEFT(e.body_text, 200) as snippet
               FROM emails e
               JOIN email_accounts a ON e.account_id = a.id
               WHERE e.user_id = $1 AND a.is_enabled = true AND ($4::text IS NULL OR e.folder = $4)
               ORDER BY e.received_at DESC
               LIMIT $2 OFFSET $3"#,
            user_id, limit as i64, offset as i64, folder
        ).fetch_all(&self.pool).await?;

        Ok(rows.into_iter().map(|r| UnifiedEmail {
            id: r.id, account_id: r.account_id, account_email: r.account_email,
            account_color: r.account_color, subject: r.subject, from_address: r.from_address,
            from_name: r.from_name, received_at: r.received_at, is_read: r.is_read,
            is_starred: r.is_starred, has_attachments: r.has_attachments.unwrap_or(false),
            folder: r.folder, snippet: r.snippet,
        }).collect())
    }

    pub async fn get_unread_counts(&self, user_id: Uuid) -> Result<Vec<(Uuid, String, i64)>, AccountError> {
        let rows = sqlx::query!(
            r#"SELECT a.id, a.email_address, COUNT(e.id) as unread_count
               FROM email_accounts a
               LEFT JOIN emails e ON e.account_id = a.id AND e.is_read = false AND e.folder = 'INBOX'
               WHERE a.user_id = $1 AND a.is_enabled = true
               GROUP BY a.id"#,
            user_id
        ).fetch_all(&self.pool).await?;

        Ok(rows.into_iter().map(|r| (r.id, r.email_address, r.unread_count.unwrap_or(0))).collect())
    }

    pub async fn search_all_accounts(&self, user_id: Uuid, query: &str, limit: i32) -> Result<Vec<UnifiedEmail>, AccountError> {
        let search_pattern = format!("%{}%", query.to_lowercase());

        let rows = sqlx::query!(
            r#"SELECT e.id, e.account_id, a.email_address as account_email, a.color as account_color,
                      e.subject, e.from_address, e.from_name, e.received_at, e.is_read, e.is_starred,
                      e.has_attachments, e.folder, LEFT(e.body_text, 200) as snippet
               FROM emails e
               JOIN email_accounts a ON e.account_id = a.id
               WHERE e.user_id = $1 AND a.is_enabled = true
                 AND (LOWER(e.subject) LIKE $2 OR LOWER(e.from_address) LIKE $2 OR LOWER(e.body_text) LIKE $2)
               ORDER BY e.received_at DESC
               LIMIT $3"#,
            user_id, search_pattern, limit as i64
        ).fetch_all(&self.pool).await?;

        Ok(rows.into_iter().map(|r| UnifiedEmail {
            id: r.id, account_id: r.account_id, account_email: r.account_email,
            account_color: r.account_color, subject: r.subject, from_address: r.from_address,
            from_name: r.from_name, received_at: r.received_at, is_read: r.is_read,
            is_starred: r.is_starred, has_attachments: r.has_attachments.unwrap_or(false),
            folder: r.folder, snippet: r.snippet,
        }).collect())
    }
}

pub struct SendAsService {
    pool: PgPool,
}

impl SendAsService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn add_identity(&self, user_id: Uuid, account_id: Uuid, email_address: String, display_name: Option<String>, signature_id: Option<Uuid>, reply_to: Option<String>) -> Result<SendAsIdentity, AccountError> {
        // Verify account ownership
        let account_exists: bool = sqlx::query_scalar!("SELECT EXISTS(SELECT 1 FROM email_accounts WHERE id = $1 AND user_id = $2)", account_id, user_id)
            .fetch_one(&self.pool).await?.unwrap_or(false);
        if !account_exists { return Err(AccountError::NotFound); }

        let id = Uuid::new_v4();
        let is_default = sqlx::query_scalar!("SELECT COUNT(*) = 0 FROM send_as_identities WHERE account_id = $1", account_id)
            .fetch_one(&self.pool).await?.unwrap_or(true);

        sqlx::query!("INSERT INTO send_as_identities (id, account_id, email_address, display_name, signature_id, is_default, reply_to, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())",
            id, account_id, email_address, display_name, signature_id, is_default, reply_to
        ).execute(&self.pool).await?;

        Ok(SendAsIdentity { id, account_id, email_address, display_name, signature_id, is_default, reply_to, created_at: Utc::now() })
    }

    pub async fn list_identities(&self, user_id: Uuid) -> Result<Vec<SendAsIdentity>, AccountError> {
        let rows = sqlx::query!(
            r#"SELECT i.id, i.account_id, i.email_address, i.display_name, i.signature_id, i.is_default, i.reply_to, i.created_at
               FROM send_as_identities i
               JOIN email_accounts a ON i.account_id = a.id
               WHERE a.user_id = $1
               ORDER BY i.is_default DESC, i.email_address"#,
            user_id
        ).fetch_all(&self.pool).await?;

        Ok(rows.into_iter().map(|r| SendAsIdentity {
            id: r.id, account_id: r.account_id, email_address: r.email_address,
            display_name: r.display_name, signature_id: r.signature_id, is_default: r.is_default,
            reply_to: r.reply_to, created_at: r.created_at,
        }).collect())
    }

    pub async fn set_default(&self, user_id: Uuid, account_id: Uuid, identity_id: Uuid) -> Result<(), AccountError> {
        // Verify ownership
        let exists: bool = sqlx::query_scalar!(
            r#"SELECT EXISTS(SELECT 1 FROM send_as_identities i JOIN email_accounts a ON i.account_id = a.id WHERE i.id = $1 AND a.user_id = $2)"#,
            identity_id, user_id
        ).fetch_one(&self.pool).await?.unwrap_or(false);
        if !exists { return Err(AccountError::NotFound); }

        sqlx::query!("UPDATE send_as_identities SET is_default = false WHERE account_id = $1", account_id).execute(&self.pool).await?;
        sqlx::query!("UPDATE send_as_identities SET is_default = true WHERE id = $1", identity_id).execute(&self.pool).await?;
        Ok(())
    }

    pub async fn delete_identity(&self, user_id: Uuid, identity_id: Uuid) -> Result<(), AccountError> {
        let result = sqlx::query!(
            r#"DELETE FROM send_as_identities WHERE id = $1 AND account_id IN (SELECT id FROM email_accounts WHERE user_id = $2)"#,
            identity_id, user_id
        ).execute(&self.pool).await?;
        if result.rows_affected() == 0 { return Err(AccountError::NotFound); }
        Ok(())
    }

    pub async fn get_default_identity(&self, user_id: Uuid, account_id: Uuid) -> Result<Option<SendAsIdentity>, AccountError> {
        let row = sqlx::query!(
            r#"SELECT i.id, i.account_id, i.email_address, i.display_name, i.signature_id, i.is_default, i.reply_to, i.created_at
               FROM send_as_identities i
               JOIN email_accounts a ON i.account_id = a.id
               WHERE i.account_id = $1 AND a.user_id = $2 AND i.is_default = true"#,
            account_id, user_id
        ).fetch_optional(&self.pool).await?;

        Ok(row.map(|r| SendAsIdentity {
            id: r.id, account_id: r.account_id, email_address: r.email_address,
            display_name: r.display_name, signature_id: r.signature_id, is_default: r.is_default,
            reply_to: r.reply_to, created_at: r.created_at,
        }))
    }
}
