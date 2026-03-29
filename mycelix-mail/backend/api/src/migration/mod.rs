// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Migration & Import Tools
//!
//! Import emails from Gmail, Outlook, MBOX/EML files

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;
use std::collections::HashMap;

// ============================================================================
// Migration Service
// ============================================================================

pub struct MigrationService {
    pool: PgPool,
}

impl MigrationService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Start a new migration job
    pub async fn start_migration(
        &self,
        user_id: Uuid,
        source: MigrationSource,
    ) -> Result<MigrationJob, MigrationError> {
        let job_id = Uuid::new_v4();

        sqlx::query(
            "INSERT INTO migration_jobs (id, user_id, source_type, status, created_at) VALUES ($1, $2, $3, 'pending', NOW())",
        )
        .bind(job_id)
        .bind(user_id)
        .bind(source.to_string())
        .execute(&self.pool)
        .await
        .map_err(|e| MigrationError::Database(e.to_string()))?;

        Ok(MigrationJob {
            id: job_id,
            user_id,
            source_type: source.to_string(),
            status: JobStatus::Pending,
            total_items: 0,
            processed_items: 0,
            failed_items: 0,
            current_phase: "Starting".to_string(),
            error_log: Vec::new(),
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
        })
    }

    /// Get migration job status
    pub async fn get_job(&self, job_id: Uuid) -> Result<MigrationJob, MigrationError> {
        let job: MigrationJob = sqlx::query_as(
            "SELECT * FROM migration_jobs WHERE id = $1",
        )
        .bind(job_id)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| MigrationError::Database(e.to_string()))?;

        Ok(job)
    }

    /// Cancel a running migration
    pub async fn cancel_migration(&self, job_id: Uuid) -> Result<(), MigrationError> {
        sqlx::query("UPDATE migration_jobs SET status = 'cancelled' WHERE id = $1 AND status = 'running'")
            .bind(job_id)
            .execute(&self.pool)
            .await
            .map_err(|e| MigrationError::Database(e.to_string()))?;

        Ok(())
    }
}

// ============================================================================
// Gmail Importer
// ============================================================================

pub struct GmailImporter {
    pool: PgPool,
}

impl GmailImporter {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Start Gmail import using OAuth
    pub async fn start_import(
        &self,
        user_id: Uuid,
        access_token: &str,
        options: GmailImportOptions,
    ) -> Result<MigrationJob, MigrationError> {
        let migration_service = MigrationService::new(self.pool.clone());
        let job = migration_service.start_migration(user_id, MigrationSource::Gmail).await?;

        let pool = self.pool.clone();
        let job_id = job.id;
        let token = access_token.to_string();

        tokio::spawn(async move {
            // Would use Gmail API to fetch and import emails
            sqlx::query("UPDATE migration_jobs SET status = 'running', started_at = NOW() WHERE id = $1")
                .bind(job_id)
                .execute(&pool)
                .await
                .ok();

            // Import logic here...

            sqlx::query("UPDATE migration_jobs SET status = 'completed', completed_at = NOW() WHERE id = $1")
                .bind(job_id)
                .execute(&pool)
                .await
                .ok();
        });

        Ok(job)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GmailImportOptions {
    pub include_spam: bool,
    pub include_trash: bool,
    pub label_mapping: HashMap<String, String>,
    pub date_from: Option<DateTime<Utc>>,
    pub date_to: Option<DateTime<Utc>>,
}

// ============================================================================
// Outlook Importer
// ============================================================================

pub struct OutlookImporter {
    pool: PgPool,
}

impl OutlookImporter {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Start Outlook import using OAuth
    pub async fn start_import(
        &self,
        user_id: Uuid,
        access_token: &str,
        options: OutlookImportOptions,
    ) -> Result<MigrationJob, MigrationError> {
        let migration_service = MigrationService::new(self.pool.clone());
        let job = migration_service.start_migration(user_id, MigrationSource::Outlook).await?;

        // Similar to Gmail, spawn background task
        Ok(job)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlookImportOptions {
    pub include_calendar: bool,
    pub include_contacts: bool,
    pub folder_mapping: HashMap<String, String>,
    pub date_from: Option<DateTime<Utc>>,
    pub date_to: Option<DateTime<Utc>>,
}

// ============================================================================
// File Importer (MBOX/EML)
// ============================================================================

pub struct FileImporter {
    pool: PgPool,
}

impl FileImporter {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Import from MBOX file
    pub async fn import_mbox(
        &self,
        user_id: Uuid,
        file_path: &str,
        target_folder: &str,
    ) -> Result<MigrationJob, MigrationError> {
        let migration_service = MigrationService::new(self.pool.clone());
        let job = migration_service.start_migration(user_id, MigrationSource::MBOX).await?;

        let pool = self.pool.clone();
        let job_id = job.id;
        let path = file_path.to_string();
        let folder = target_folder.to_string();

        tokio::spawn(async move {
            sqlx::query("UPDATE migration_jobs SET status = 'running', started_at = NOW() WHERE id = $1")
                .bind(job_id)
                .execute(&pool)
                .await
                .ok();

            // Parse MBOX file and import emails
            match tokio::fs::read_to_string(&path).await {
                Ok(content) => {
                    let emails: Vec<&str> = content.split("\nFrom ").collect();
                    let total = emails.len() as i64;

                    sqlx::query("UPDATE migration_jobs SET total_items = $2 WHERE id = $1")
                        .bind(job_id)
                        .bind(total)
                        .execute(&pool)
                        .await
                        .ok();

                    // Process each email...

                    sqlx::query("UPDATE migration_jobs SET status = 'completed', completed_at = NOW() WHERE id = $1")
                        .bind(job_id)
                        .execute(&pool)
                        .await
                        .ok();
                }
                Err(e) => {
                    sqlx::query("UPDATE migration_jobs SET status = 'failed', error_log = array_append(error_log, $2) WHERE id = $1")
                        .bind(job_id)
                        .bind(e.to_string())
                        .execute(&pool)
                        .await
                        .ok();
                }
            }
        });

        Ok(job)
    }

    /// Import single EML file
    pub async fn import_eml(
        &self,
        user_id: Uuid,
        file_content: &[u8],
        target_folder: &str,
    ) -> Result<Uuid, MigrationError> {
        let content = String::from_utf8_lossy(file_content);
        let parsed = self.parse_email(&content)?;

        let email_id = Uuid::new_v4();

        sqlx::query(
            r#"
            INSERT INTO emails (id, user_id, message_id, from_address, to_addresses,
                               subject, body_text, received_at, folder, is_read, imported_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, true, NOW())
            "#,
        )
        .bind(email_id)
        .bind(user_id)
        .bind(&parsed.message_id)
        .bind(&parsed.from_address)
        .bind(&parsed.to_addresses)
        .bind(&parsed.subject)
        .bind(&parsed.body_text)
        .bind(parsed.date)
        .bind(target_folder)
        .execute(&self.pool)
        .await
        .map_err(|e| MigrationError::Database(e.to_string()))?;

        Ok(email_id)
    }

    fn parse_email(&self, content: &str) -> Result<ParsedEmail, MigrationError> {
        let mut headers = HashMap::new();
        let mut in_headers = true;
        let mut body = String::new();

        for line in content.lines() {
            if in_headers {
                if line.is_empty() {
                    in_headers = false;
                    continue;
                }
                if let Some((key, value)) = line.split_once(':') {
                    headers.insert(key.trim().to_lowercase(), value.trim().to_string());
                }
            } else {
                body.push_str(line);
                body.push('\n');
            }
        }

        Ok(ParsedEmail {
            message_id: headers.get("message-id").cloned().unwrap_or_default(),
            from_address: headers.get("from").cloned().unwrap_or_default(),
            to_addresses: headers.get("to").map(|s| vec![s.clone()]).unwrap_or_default(),
            subject: headers.get("subject").cloned().unwrap_or_default(),
            body_text: body,
            date: headers
                .get("date")
                .and_then(|d| DateTime::parse_from_rfc2822(d).ok())
                .map(|d| d.with_timezone(&Utc))
                .unwrap_or_else(Utc::now),
        })
    }
}

#[derive(Debug, Clone)]
struct ParsedEmail {
    message_id: String,
    from_address: String,
    to_addresses: Vec<String>,
    subject: String,
    body_text: String,
    date: DateTime<Utc>,
}

// ============================================================================
// Types
// ============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MigrationSource {
    Gmail,
    Outlook,
    MBOX,
    EML,
    IMAP,
}

impl ToString for MigrationSource {
    fn to_string(&self) -> String {
        match self {
            MigrationSource::Gmail => "gmail".to_string(),
            MigrationSource::Outlook => "outlook".to_string(),
            MigrationSource::MBOX => "mbox".to_string(),
            MigrationSource::EML => "eml".to_string(),
            MigrationSource::IMAP => "imap".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "text")]
pub enum JobStatus {
    Pending,
    Running,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct MigrationJob {
    pub id: Uuid,
    pub user_id: Uuid,
    pub source_type: String,
    pub status: JobStatus,
    pub total_items: i64,
    pub processed_items: i64,
    pub failed_items: i64,
    pub current_phase: String,
    pub error_log: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

// ============================================================================
// ProtonMail Importer
// ============================================================================

pub struct ProtonMailImporter {
    pool: PgPool,
}

impl ProtonMailImporter {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Import from ProtonMail export (MBOX format from ProtonMail)
    pub async fn start_import(
        &self,
        user_id: Uuid,
        export_path: &str,
        options: ProtonMailImportOptions,
    ) -> Result<MigrationJob, MigrationError> {
        let migration_service = MigrationService::new(self.pool.clone());
        let job = migration_service.start_migration(user_id, MigrationSource::ProtonMail).await?;

        let pool = self.pool.clone();
        let job_id = job.id;
        let path = export_path.to_string();
        let opts = options.clone();

        tokio::spawn(async move {
            sqlx::query("UPDATE migration_jobs SET status = 'running', started_at = NOW(), current_phase = 'Scanning export' WHERE id = $1")
                .bind(job_id)
                .execute(&pool)
                .await
                .ok();

            // ProtonMail exports are typically MBOX with PGP-encrypted messages
            // Process export directory structure
            let import_result = Self::process_export(&pool, job_id, user_id, &path, &opts).await;

            match import_result {
                Ok(stats) => {
                    sqlx::query(
                        "UPDATE migration_jobs SET status = 'completed', completed_at = NOW(), processed_items = $2, failed_items = $3 WHERE id = $1"
                    )
                    .bind(job_id)
                    .bind(stats.processed)
                    .bind(stats.failed)
                    .execute(&pool)
                    .await
                    .ok();
                }
                Err(e) => {
                    sqlx::query("UPDATE migration_jobs SET status = 'failed', error_log = array_append(error_log, $2) WHERE id = $1")
                        .bind(job_id)
                        .bind(e.to_string())
                        .execute(&pool)
                        .await
                        .ok();
                }
            }
        });

        Ok(job)
    }

    async fn process_export(
        pool: &PgPool,
        job_id: Uuid,
        user_id: Uuid,
        path: &str,
        options: &ProtonMailImportOptions,
    ) -> Result<ImportStats, MigrationError> {
        let mut stats = ImportStats::default();

        // Read directory structure
        let entries = tokio::fs::read_dir(path)
            .await
            .map_err(|e| MigrationError::FileError(e.to_string()))?;

        let mut entries = entries;
        while let Some(entry) = entries.next_entry().await.map_err(|e| MigrationError::FileError(e.to_string()))? {
            let file_path = entry.path();

            if file_path.extension().map_or(false, |e| e == "mbox") {
                let folder_name = file_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("imported");

                // Update phase
                sqlx::query("UPDATE migration_jobs SET current_phase = $2 WHERE id = $1")
                    .bind(job_id)
                    .bind(format!("Importing {}", folder_name))
                    .execute(pool)
                    .await
                    .ok();

                // Process MBOX file
                match tokio::fs::read_to_string(&file_path).await {
                    Ok(content) => {
                        let emails: Vec<&str> = content.split("\nFrom ").collect();
                        stats.total += emails.len() as i64;

                        for email_content in emails {
                            if email_content.trim().is_empty() {
                                continue;
                            }

                            // Import email (simplified)
                            stats.processed += 1;
                        }
                    }
                    Err(e) => {
                        stats.failed += 1;
                        tracing::warn!("Failed to read {}: {}", file_path.display(), e);
                    }
                }
            }
        }

        // Import contacts if requested
        if options.include_contacts {
            sqlx::query("UPDATE migration_jobs SET current_phase = 'Importing contacts' WHERE id = $1")
                .bind(job_id)
                .execute(pool)
                .await
                .ok();

            // Process contacts vCard export
        }

        Ok(stats)
    }
}

#[derive(Debug, Clone, Default)]
struct ImportStats {
    total: i64,
    processed: i64,
    failed: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtonMailImportOptions {
    pub include_contacts: bool,
    pub include_calendar: bool,
    pub decrypt_with_key: Option<String>,
    pub folder_mapping: HashMap<String, String>,
}

// ============================================================================
// IMAP Importer (Generic)
// ============================================================================

pub struct ImapImporter {
    pool: PgPool,
}

impl ImapImporter {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Import from any IMAP server
    pub async fn start_import(
        &self,
        user_id: Uuid,
        config: ImapImportConfig,
    ) -> Result<MigrationJob, MigrationError> {
        let migration_service = MigrationService::new(self.pool.clone());
        let job = migration_service.start_migration(user_id, MigrationSource::IMAP).await?;

        let pool = self.pool.clone();
        let job_id = job.id;
        let cfg = config.clone();

        tokio::spawn(async move {
            sqlx::query("UPDATE migration_jobs SET status = 'running', started_at = NOW(), current_phase = 'Connecting to IMAP server' WHERE id = $1")
                .bind(job_id)
                .execute(&pool)
                .await
                .ok();

            // IMAP import logic would go here
            // 1. Connect to IMAP server
            // 2. List mailboxes
            // 3. Fetch emails from each mailbox
            // 4. Store in local database

            sqlx::query("UPDATE migration_jobs SET status = 'completed', completed_at = NOW() WHERE id = $1")
                .bind(job_id)
                .execute(&pool)
                .await
                .ok();
        });

        Ok(job)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImapImportConfig {
    pub host: String,
    pub port: u16,
    pub username: String,
    pub password: String,
    pub use_ssl: bool,
    pub folders: Option<Vec<String>>,
    pub date_from: Option<DateTime<Utc>>,
    pub date_to: Option<DateTime<Utc>>,
    pub batch_size: Option<usize>,
}

// ============================================================================
// Apple Mail Importer
// ============================================================================

pub struct AppleMailImporter {
    pool: PgPool,
}

impl AppleMailImporter {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Import from Apple Mail .emlx files
    pub async fn start_import(
        &self,
        user_id: Uuid,
        mail_directory: &str,
    ) -> Result<MigrationJob, MigrationError> {
        let migration_service = MigrationService::new(self.pool.clone());
        let job = migration_service.start_migration(user_id, MigrationSource::AppleMail).await?;

        // Apple Mail stores emails as .emlx files in ~/Library/Mail
        // Process directory structure

        Ok(job)
    }
}

// ============================================================================
// Thunderbird Importer
// ============================================================================

pub struct ThunderbirdImporter {
    pool: PgPool,
}

impl ThunderbirdImporter {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Import from Thunderbird profile
    pub async fn start_import(
        &self,
        user_id: Uuid,
        profile_path: &str,
    ) -> Result<MigrationJob, MigrationError> {
        let migration_service = MigrationService::new(self.pool.clone());
        let job = migration_service.start_migration(user_id, MigrationSource::Thunderbird).await?;

        // Thunderbird uses MBOX format in profile directory
        // Parse Local Folders and ImapMail directories

        Ok(job)
    }
}

// ============================================================================
// Migration API Endpoints
// ============================================================================

use axum::{
    extract::{Path, State, Multipart},
    response::Json,
    http::StatusCode,
};

pub async fn list_migrations(
    State(pool): State<PgPool>,
    user_id: Uuid,
) -> Result<Json<Vec<MigrationJob>>, (StatusCode, String)> {
    let jobs: Vec<MigrationJob> = sqlx::query_as(
        "SELECT * FROM migration_jobs WHERE user_id = $1 ORDER BY created_at DESC LIMIT 50"
    )
    .bind(user_id)
    .fetch_all(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(jobs))
}

pub async fn get_migration_status(
    State(pool): State<PgPool>,
    Path(job_id): Path<Uuid>,
) -> Result<Json<MigrationJob>, (StatusCode, String)> {
    let service = MigrationService::new(pool);
    let job = service.get_job(job_id)
        .await
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    Ok(Json(job))
}

pub async fn cancel_migration(
    State(pool): State<PgPool>,
    Path(job_id): Path<Uuid>,
) -> Result<StatusCode, (StatusCode, String)> {
    let service = MigrationService::new(pool);
    service.cancel_migration(job_id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(StatusCode::OK)
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MigrationSource {
    Gmail,
    Outlook,
    ProtonMail,
    MBOX,
    EML,
    IMAP,
    AppleMail,
    Thunderbird,
}

impl ToString for MigrationSource {
    fn to_string(&self) -> String {
        match self {
            MigrationSource::Gmail => "gmail".to_string(),
            MigrationSource::Outlook => "outlook".to_string(),
            MigrationSource::ProtonMail => "protonmail".to_string(),
            MigrationSource::MBOX => "mbox".to_string(),
            MigrationSource::EML => "eml".to_string(),
            MigrationSource::IMAP => "imap".to_string(),
            MigrationSource::AppleMail => "applemail".to_string(),
            MigrationSource::Thunderbird => "thunderbird".to_string(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MigrationError {
    #[error("Database error: {0}")]
    Database(String),
    #[error("API error: {0}")]
    ApiError(String),
    #[error("File error: {0}")]
    FileError(String),
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Authentication error: {0}")]
    AuthError(String),
    #[error("Connection error: {0}")]
    ConnectionError(String),
}
