// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Attachment Manager
//!
//! Cloud storage integration, attachment search, and storage quotas

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;
use std::io::Read;
use tokio::io::AsyncReadExt;

// ============================================================================
// Attachment Service
// ============================================================================

pub struct AttachmentService {
    pool: PgPool,
    storage: Box<dyn StorageBackend + Send + Sync>,
}

impl AttachmentService {
    pub fn new(pool: PgPool, storage: Box<dyn StorageBackend + Send + Sync>) -> Self {
        Self { pool, storage }
    }

    /// Upload an attachment
    pub async fn upload(
        &self,
        user_id: Uuid,
        email_id: Uuid,
        file: AttachmentUpload,
    ) -> Result<Attachment, AttachmentError> {
        // Check quota
        let quota = self.get_quota_usage(user_id).await?;
        if quota.used_bytes + file.size > quota.total_bytes {
            return Err(AttachmentError::QuotaExceeded);
        }

        let id = Uuid::new_v4();
        let storage_key = format!("{}/{}/{}", user_id, email_id, id);

        // Upload to storage backend
        self.storage.put(&storage_key, &file.content).await?;

        // Calculate content hash for deduplication
        let content_hash = self.hash_content(&file.content);

        sqlx::query(
            r#"
            INSERT INTO attachments (id, user_id, email_id, filename, content_type, size_bytes, storage_key, content_hash, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
            "#
        )
        .bind(id)
        .bind(user_id)
        .bind(email_id)
        .bind(&file.filename)
        .bind(&file.content_type)
        .bind(file.size as i64)
        .bind(&storage_key)
        .bind(&content_hash)
        .execute(&self.pool)
        .await
        .map_err(|e| AttachmentError::Database(e.to_string()))?;

        Ok(Attachment {
            id,
            user_id,
            email_id,
            filename: file.filename,
            content_type: file.content_type,
            size_bytes: file.size as i64,
            storage_key,
            content_hash,
            created_at: Utc::now(),
        })
    }

    /// Download an attachment
    pub async fn download(
        &self,
        user_id: Uuid,
        attachment_id: Uuid,
    ) -> Result<AttachmentDownload, AttachmentError> {
        let attachment: Attachment = sqlx::query_as(
            "SELECT * FROM attachments WHERE id = $1 AND user_id = $2"
        )
        .bind(attachment_id)
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| AttachmentError::Database(e.to_string()))?
        .ok_or(AttachmentError::NotFound)?;

        let content = self.storage.get(&attachment.storage_key).await?;

        Ok(AttachmentDownload {
            filename: attachment.filename,
            content_type: attachment.content_type,
            content,
        })
    }

    /// Delete an attachment
    pub async fn delete(
        &self,
        user_id: Uuid,
        attachment_id: Uuid,
    ) -> Result<(), AttachmentError> {
        let attachment: Option<Attachment> = sqlx::query_as(
            "SELECT * FROM attachments WHERE id = $1 AND user_id = $2"
        )
        .bind(attachment_id)
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| AttachmentError::Database(e.to_string()))?;

        if let Some(att) = attachment {
            self.storage.delete(&att.storage_key).await?;

            sqlx::query("DELETE FROM attachments WHERE id = $1")
                .bind(attachment_id)
                .execute(&self.pool)
                .await
                .map_err(|e| AttachmentError::Database(e.to_string()))?;
        }

        Ok(())
    }

    /// Search attachments by filename
    pub async fn search(
        &self,
        user_id: Uuid,
        query: AttachmentSearchQuery,
    ) -> Result<Vec<AttachmentSearchResult>, AttachmentError> {
        let mut sql = String::from(
            r#"
            SELECT a.*, e.subject as email_subject, e.from_address as email_from
            FROM attachments a
            JOIN emails e ON a.email_id = e.id
            WHERE a.user_id = $1
            "#
        );

        if let Some(filename) = &query.filename {
            sql.push_str(&format!(" AND LOWER(a.filename) LIKE LOWER('%{}%')", filename));
        }

        if let Some(content_type) = &query.content_type {
            sql.push_str(&format!(" AND a.content_type LIKE '{}%'", content_type));
        }

        if let Some(min_size) = query.min_size {
            sql.push_str(&format!(" AND a.size_bytes >= {}", min_size));
        }

        if let Some(max_size) = query.max_size {
            sql.push_str(&format!(" AND a.size_bytes <= {}", max_size));
        }

        if let Some(after) = &query.after {
            sql.push_str(&format!(" AND a.created_at >= '{}'", after.to_rfc3339()));
        }

        sql.push_str(" ORDER BY a.created_at DESC");
        sql.push_str(&format!(" LIMIT {} OFFSET {}", query.limit.unwrap_or(50), query.offset.unwrap_or(0)));

        let results: Vec<AttachmentSearchResult> = sqlx::query_as(&sql)
            .bind(user_id)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| AttachmentError::Database(e.to_string()))?;

        Ok(results)
    }

    /// Get attachments for an email
    pub async fn get_email_attachments(
        &self,
        user_id: Uuid,
        email_id: Uuid,
    ) -> Result<Vec<Attachment>, AttachmentError> {
        let attachments: Vec<Attachment> = sqlx::query_as(
            "SELECT * FROM attachments WHERE user_id = $1 AND email_id = $2 ORDER BY filename"
        )
        .bind(user_id)
        .bind(email_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| AttachmentError::Database(e.to_string()))?;

        Ok(attachments)
    }

    /// Generate a shareable link for an attachment
    pub async fn create_share_link(
        &self,
        user_id: Uuid,
        attachment_id: Uuid,
        expires_in: chrono::Duration,
    ) -> Result<ShareLink, AttachmentError> {
        let attachment: Attachment = sqlx::query_as(
            "SELECT * FROM attachments WHERE id = $1 AND user_id = $2"
        )
        .bind(attachment_id)
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| AttachmentError::Database(e.to_string()))?
        .ok_or(AttachmentError::NotFound)?;

        let link_id = Uuid::new_v4();
        let token = self.generate_token();
        let expires_at = Utc::now() + expires_in;

        sqlx::query(
            r#"
            INSERT INTO attachment_share_links (id, attachment_id, token, expires_at, created_at)
            VALUES ($1, $2, $3, $4, NOW())
            "#
        )
        .bind(link_id)
        .bind(attachment_id)
        .bind(&token)
        .bind(expires_at)
        .execute(&self.pool)
        .await
        .map_err(|e| AttachmentError::Database(e.to_string()))?;

        Ok(ShareLink {
            id: link_id,
            attachment_id,
            token,
            expires_at,
            download_count: 0,
        })
    }

    fn hash_content(&self, content: &[u8]) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(content);
        hex::encode(hasher.finalize())
    }

    fn generate_token(&self) -> String {
        use rand::Rng;
        let bytes: [u8; 32] = rand::thread_rng().gen();
        base64::encode_config(bytes, base64::URL_SAFE_NO_PAD)
    }

    /// Get quota usage for a user
    pub async fn get_quota_usage(&self, user_id: Uuid) -> Result<QuotaUsage, AttachmentError> {
        let used: (i64,) = sqlx::query_as(
            "SELECT COALESCE(SUM(size_bytes), 0) FROM attachments WHERE user_id = $1"
        )
        .bind(user_id)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| AttachmentError::Database(e.to_string()))?;

        let quota: Option<(i64,)> = sqlx::query_as(
            "SELECT storage_quota_bytes FROM users WHERE id = $1"
        )
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| AttachmentError::Database(e.to_string()))?;

        let total_bytes = quota.map(|(q,)| q).unwrap_or(5 * 1024 * 1024 * 1024); // 5GB default

        Ok(QuotaUsage {
            user_id,
            used_bytes: used.0,
            total_bytes,
            percentage_used: (used.0 as f64 / total_bytes as f64) * 100.0,
        })
    }
}

// ============================================================================
// Large File Service (for files > threshold)
// ============================================================================

pub struct LargeFileService {
    pool: PgPool,
    storage: Box<dyn StorageBackend + Send + Sync>,
    threshold_bytes: i64,
}

impl LargeFileService {
    pub fn new(pool: PgPool, storage: Box<dyn StorageBackend + Send + Sync>, threshold_mb: i64) -> Self {
        Self {
            pool,
            storage,
            threshold_bytes: threshold_mb * 1024 * 1024,
        }
    }

    /// Check if a file should use large file upload
    pub fn should_use_large_file(&self, size: i64) -> bool {
        size > self.threshold_bytes
    }

    /// Upload large file and return a link instead of embedding
    pub async fn upload_large_file(
        &self,
        user_id: Uuid,
        file: AttachmentUpload,
    ) -> Result<LargeFileLink, AttachmentError> {
        let id = Uuid::new_v4();
        let storage_key = format!("large/{}/{}", user_id, id);

        // Upload to storage
        self.storage.put(&storage_key, &file.content).await?;

        let expires_at = Utc::now() + chrono::Duration::days(30);

        sqlx::query(
            r#"
            INSERT INTO large_files (id, user_id, filename, content_type, size_bytes, storage_key, expires_at, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
            "#
        )
        .bind(id)
        .bind(user_id)
        .bind(&file.filename)
        .bind(&file.content_type)
        .bind(file.size as i64)
        .bind(&storage_key)
        .bind(expires_at)
        .execute(&self.pool)
        .await
        .map_err(|e| AttachmentError::Database(e.to_string()))?;

        Ok(LargeFileLink {
            id,
            filename: file.filename,
            size_bytes: file.size as i64,
            download_url: format!("/api/files/large/{}", id),
            expires_at,
        })
    }
}

// ============================================================================
// Storage Backends
// ============================================================================

#[async_trait::async_trait]
pub trait StorageBackend {
    async fn put(&self, key: &str, content: &[u8]) -> Result<(), AttachmentError>;
    async fn get(&self, key: &str) -> Result<Vec<u8>, AttachmentError>;
    async fn delete(&self, key: &str) -> Result<(), AttachmentError>;
    async fn exists(&self, key: &str) -> Result<bool, AttachmentError>;
}

/// Local filesystem storage
pub struct LocalStorage {
    base_path: String,
}

impl LocalStorage {
    pub fn new(base_path: &str) -> Self {
        Self { base_path: base_path.to_string() }
    }
}

#[async_trait::async_trait]
impl StorageBackend for LocalStorage {
    async fn put(&self, key: &str, content: &[u8]) -> Result<(), AttachmentError> {
        let path = format!("{}/{}", self.base_path, key);
        if let Some(parent) = std::path::Path::new(&path).parent() {
            tokio::fs::create_dir_all(parent).await
                .map_err(|e| AttachmentError::Storage(e.to_string()))?;
        }
        tokio::fs::write(&path, content).await
            .map_err(|e| AttachmentError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn get(&self, key: &str) -> Result<Vec<u8>, AttachmentError> {
        let path = format!("{}/{}", self.base_path, key);
        tokio::fs::read(&path).await
            .map_err(|e| AttachmentError::Storage(e.to_string()))
    }

    async fn delete(&self, key: &str) -> Result<(), AttachmentError> {
        let path = format!("{}/{}", self.base_path, key);
        tokio::fs::remove_file(&path).await
            .map_err(|e| AttachmentError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn exists(&self, key: &str) -> Result<bool, AttachmentError> {
        let path = format!("{}/{}", self.base_path, key);
        Ok(std::path::Path::new(&path).exists())
    }
}

/// S3-compatible storage
pub struct S3Storage {
    bucket: String,
    client: aws_sdk_s3::Client,
}

impl S3Storage {
    pub async fn new(bucket: &str, config: aws_config::SdkConfig) -> Self {
        Self {
            bucket: bucket.to_string(),
            client: aws_sdk_s3::Client::new(&config),
        }
    }
}

#[async_trait::async_trait]
impl StorageBackend for S3Storage {
    async fn put(&self, key: &str, content: &[u8]) -> Result<(), AttachmentError> {
        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(key)
            .body(content.to_vec().into())
            .send()
            .await
            .map_err(|e| AttachmentError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn get(&self, key: &str) -> Result<Vec<u8>, AttachmentError> {
        let response = self.client
            .get_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| AttachmentError::Storage(e.to_string()))?;

        let bytes = response.body.collect().await
            .map_err(|e| AttachmentError::Storage(e.to_string()))?;
        Ok(bytes.to_vec())
    }

    async fn delete(&self, key: &str) -> Result<(), AttachmentError> {
        self.client
            .delete_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| AttachmentError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn exists(&self, key: &str) -> Result<bool, AttachmentError> {
        match self.client
            .head_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
        {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }
}

// ============================================================================
// Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Attachment {
    pub id: Uuid,
    pub user_id: Uuid,
    pub email_id: Uuid,
    pub filename: String,
    pub content_type: String,
    pub size_bytes: i64,
    pub storage_key: String,
    pub content_hash: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct AttachmentUpload {
    pub filename: String,
    pub content_type: String,
    pub content: Vec<u8>,
    pub size: usize,
}

#[derive(Debug, Clone)]
pub struct AttachmentDownload {
    pub filename: String,
    pub content_type: String,
    pub content: Vec<u8>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AttachmentSearchQuery {
    pub filename: Option<String>,
    pub content_type: Option<String>,
    pub min_size: Option<i64>,
    pub max_size: Option<i64>,
    pub after: Option<DateTime<Utc>>,
    pub limit: Option<i64>,
    pub offset: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct AttachmentSearchResult {
    pub id: Uuid,
    pub filename: String,
    pub content_type: String,
    pub size_bytes: i64,
    pub created_at: DateTime<Utc>,
    pub email_id: Uuid,
    pub email_subject: String,
    pub email_from: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShareLink {
    pub id: Uuid,
    pub attachment_id: Uuid,
    pub token: String,
    pub expires_at: DateTime<Utc>,
    pub download_count: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LargeFileLink {
    pub id: Uuid,
    pub filename: String,
    pub size_bytes: i64,
    pub download_url: String,
    pub expires_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotaUsage {
    pub user_id: Uuid,
    pub used_bytes: i64,
    pub total_bytes: i64,
    pub percentage_used: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum AttachmentError {
    #[error("Database error: {0}")]
    Database(String),
    #[error("Storage error: {0}")]
    Storage(String),
    #[error("Attachment not found")]
    NotFound,
    #[error("Storage quota exceeded")]
    QuotaExceeded,
    #[error("File too large")]
    FileTooLarge,
}
