// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Scheduled Send & Undo Send
//!
//! Time-delayed email delivery with cancellation support

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use tokio::sync::mpsc;
use tracing::{info, warn, error};
use uuid::Uuid;

/// Scheduled email status
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "scheduled_status", rename_all = "lowercase")]
pub enum ScheduledStatus {
    Pending,
    Sending,
    Sent,
    Cancelled,
    Failed,
}

/// Scheduled email record
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct ScheduledEmail {
    pub id: Uuid,
    pub email_id: Uuid,
    pub user_id: Uuid,
    pub tenant_id: Option<Uuid>,
    pub scheduled_at: DateTime<Utc>,
    pub timezone: String,
    pub status: ScheduledStatus,
    pub retry_count: i32,
    pub last_error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Undo send window configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UndoSendConfig {
    /// Delay in seconds before actually sending (0 = disabled)
    pub delay_seconds: u32,
    /// Maximum undo window (even if user hasn't viewed confirmation)
    pub max_delay_seconds: u32,
}

impl Default for UndoSendConfig {
    fn default() -> Self {
        Self {
            delay_seconds: 10,
            max_delay_seconds: 30,
        }
    }
}

/// Scheduled email service
pub struct ScheduledEmailService {
    pool: PgPool,
    undo_config: UndoSendConfig,
}

impl ScheduledEmailService {
    pub fn new(pool: PgPool, undo_config: UndoSendConfig) -> Self {
        Self { pool, undo_config }
    }

    /// Schedule an email for future delivery
    pub async fn schedule_email(
        &self,
        email_id: Uuid,
        user_id: Uuid,
        tenant_id: Option<Uuid>,
        send_at: DateTime<Utc>,
        timezone: &str,
    ) -> Result<ScheduledEmail, sqlx::Error> {
        let id = Uuid::new_v4();

        let scheduled = sqlx::query_as::<_, ScheduledEmail>(
            r#"
            INSERT INTO scheduled_emails (
                id, email_id, user_id, tenant_id, scheduled_at, timezone,
                status, retry_count, created_at, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, 'pending', 0, NOW(), NOW())
            RETURNING *
            "#,
        )
        .bind(id)
        .bind(email_id)
        .bind(user_id)
        .bind(tenant_id)
        .bind(send_at)
        .bind(timezone)
        .fetch_one(&self.pool)
        .await?;

        info!(
            scheduled_id = %id,
            email_id = %email_id,
            send_at = %send_at,
            "Email scheduled"
        );

        Ok(scheduled)
    }

    /// Queue email with undo delay
    pub async fn queue_with_undo(
        &self,
        email_id: Uuid,
        user_id: Uuid,
        tenant_id: Option<Uuid>,
    ) -> Result<UndoHandle, sqlx::Error> {
        let send_at = Utc::now() + Duration::seconds(self.undo_config.delay_seconds as i64);

        let scheduled = self.schedule_email(
            email_id,
            user_id,
            tenant_id,
            send_at,
            "UTC",
        ).await?;

        Ok(UndoHandle {
            scheduled_id: scheduled.id,
            email_id,
            expires_at: send_at,
            can_undo: true,
        })
    }

    /// Cancel a scheduled email (undo send)
    pub async fn cancel_scheduled(
        &self,
        scheduled_id: Uuid,
        user_id: Uuid,
    ) -> Result<bool, ScheduledError> {
        // Verify ownership and status
        let scheduled = sqlx::query_as::<_, ScheduledEmail>(
            "SELECT * FROM scheduled_emails WHERE id = $1 AND user_id = $2",
        )
        .bind(scheduled_id)
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(ScheduledError::Database)?
        .ok_or(ScheduledError::NotFound)?;

        // Can only cancel pending emails
        if scheduled.status != ScheduledStatus::Pending {
            return Err(ScheduledError::CannotCancel(format!(
                "Email status is {:?}",
                scheduled.status
            )));
        }

        // Check if still within undo window
        if Utc::now() >= scheduled.scheduled_at {
            return Err(ScheduledError::CannotCancel(
                "Undo window has expired".to_string()
            ));
        }

        // Cancel
        sqlx::query(
            r#"
            UPDATE scheduled_emails
            SET status = 'cancelled', updated_at = NOW()
            WHERE id = $1
            "#,
        )
        .bind(scheduled_id)
        .execute(&self.pool)
        .await
        .map_err(ScheduledError::Database)?;

        info!(scheduled_id = %scheduled_id, "Scheduled email cancelled");

        Ok(true)
    }

    /// Reschedule an email
    pub async fn reschedule(
        &self,
        scheduled_id: Uuid,
        user_id: Uuid,
        new_send_at: DateTime<Utc>,
    ) -> Result<ScheduledEmail, ScheduledError> {
        // Verify ownership and status
        let scheduled = sqlx::query_as::<_, ScheduledEmail>(
            "SELECT * FROM scheduled_emails WHERE id = $1 AND user_id = $2",
        )
        .bind(scheduled_id)
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(ScheduledError::Database)?
        .ok_or(ScheduledError::NotFound)?;

        if scheduled.status != ScheduledStatus::Pending {
            return Err(ScheduledError::CannotCancel(format!(
                "Cannot reschedule email with status {:?}",
                scheduled.status
            )));
        }

        // Validate new time is in the future
        if new_send_at <= Utc::now() {
            return Err(ScheduledError::InvalidTime(
                "Scheduled time must be in the future".to_string()
            ));
        }

        let updated = sqlx::query_as::<_, ScheduledEmail>(
            r#"
            UPDATE scheduled_emails
            SET scheduled_at = $2, updated_at = NOW()
            WHERE id = $1
            RETURNING *
            "#,
        )
        .bind(scheduled_id)
        .bind(new_send_at)
        .fetch_one(&self.pool)
        .await
        .map_err(ScheduledError::Database)?;

        info!(
            scheduled_id = %scheduled_id,
            new_send_at = %new_send_at,
            "Email rescheduled"
        );

        Ok(updated)
    }

    /// Get pending emails due for sending
    pub async fn get_due_emails(&self, limit: i32) -> Result<Vec<ScheduledEmail>, sqlx::Error> {
        sqlx::query_as::<_, ScheduledEmail>(
            r#"
            SELECT * FROM scheduled_emails
            WHERE status = 'pending'
              AND scheduled_at <= NOW()
            ORDER BY scheduled_at ASC
            LIMIT $1
            FOR UPDATE SKIP LOCKED
            "#,
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await
    }

    /// Mark email as sending
    pub async fn mark_sending(&self, scheduled_id: Uuid) -> Result<(), sqlx::Error> {
        sqlx::query(
            r#"
            UPDATE scheduled_emails
            SET status = 'sending', updated_at = NOW()
            WHERE id = $1
            "#,
        )
        .bind(scheduled_id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Mark email as sent
    pub async fn mark_sent(&self, scheduled_id: Uuid) -> Result<(), sqlx::Error> {
        sqlx::query(
            r#"
            UPDATE scheduled_emails
            SET status = 'sent', updated_at = NOW()
            WHERE id = $1
            "#,
        )
        .bind(scheduled_id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Mark email as failed
    pub async fn mark_failed(
        &self,
        scheduled_id: Uuid,
        error: &str,
    ) -> Result<(), sqlx::Error> {
        sqlx::query(
            r#"
            UPDATE scheduled_emails
            SET status = 'failed',
                retry_count = retry_count + 1,
                last_error = $2,
                updated_at = NOW()
            WHERE id = $1
            "#,
        )
        .bind(scheduled_id)
        .bind(error)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Get user's scheduled emails
    pub async fn get_user_scheduled(
        &self,
        user_id: Uuid,
        include_sent: bool,
    ) -> Result<Vec<ScheduledEmail>, sqlx::Error> {
        let query = if include_sent {
            r#"
            SELECT * FROM scheduled_emails
            WHERE user_id = $1
            ORDER BY scheduled_at DESC
            LIMIT 100
            "#
        } else {
            r#"
            SELECT * FROM scheduled_emails
            WHERE user_id = $1 AND status = 'pending'
            ORDER BY scheduled_at ASC
            "#
        };

        sqlx::query_as::<_, ScheduledEmail>(query)
            .bind(user_id)
            .fetch_all(&self.pool)
            .await
    }
}

/// Handle for undo send
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UndoHandle {
    pub scheduled_id: Uuid,
    pub email_id: Uuid,
    pub expires_at: DateTime<Utc>,
    pub can_undo: bool,
}

impl UndoHandle {
    pub fn seconds_remaining(&self) -> i64 {
        (self.expires_at - Utc::now()).num_seconds().max(0)
    }

    pub fn is_expired(&self) -> bool {
        Utc::now() >= self.expires_at
    }
}

/// Scheduled email errors
#[derive(Debug)]
pub enum ScheduledError {
    NotFound,
    CannotCancel(String),
    InvalidTime(String),
    Database(sqlx::Error),
}

impl std::fmt::Display for ScheduledError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => write!(f, "Scheduled email not found"),
            Self::CannotCancel(msg) => write!(f, "Cannot cancel: {}", msg),
            Self::InvalidTime(msg) => write!(f, "Invalid time: {}", msg),
            Self::Database(e) => write!(f, "Database error: {}", e),
        }
    }
}

impl std::error::Error for ScheduledError {}

/// Background worker for processing scheduled emails
pub async fn scheduled_email_worker(
    pool: PgPool,
    mut shutdown: mpsc::Receiver<()>,
) {
    let service = ScheduledEmailService::new(pool.clone(), UndoSendConfig::default());

    info!("Scheduled email worker started");

    loop {
        tokio::select! {
            _ = shutdown.recv() => {
                info!("Scheduled email worker shutting down");
                break;
            }
            _ = tokio::time::sleep(std::time::Duration::from_secs(1)) => {
                // Process due emails
                match service.get_due_emails(10).await {
                    Ok(emails) => {
                        for scheduled in emails {
                            if let Err(e) = process_scheduled_email(&service, &pool, scheduled).await {
                                error!(error = %e, "Failed to process scheduled email");
                            }
                        }
                    }
                    Err(e) => {
                        error!(error = %e, "Failed to fetch due emails");
                    }
                }
            }
        }
    }
}

async fn process_scheduled_email(
    service: &ScheduledEmailService,
    pool: &PgPool,
    scheduled: ScheduledEmail,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::db::models::EmailStatus;
    use crate::email::transport::{OutgoingEmail, SmtpConfig, SmtpTransport};

    service.mark_sending(scheduled.id).await?;

    info!(
        scheduled_id = %scheduled.id,
        email_id = %scheduled.email_id,
        "Processing scheduled email"
    );

    // Fetch the email details from the database
    let email = sqlx::query_as::<_, EmailRow>(
        r#"
        SELECT id, from_address, to_addresses, cc_addresses, bcc_addresses,
               subject, body_text, body_html, message_id, in_reply_to, status
        FROM emails
        WHERE id = $1
        "#,
    )
    .bind(scheduled.email_id)
    .fetch_optional(pool)
    .await?
    .ok_or_else(|| format!("Email {} not found", scheduled.email_id))?;

    // Verify email is in a sendable state
    if email.status != "draft" && email.status != "queued" {
        warn!(
            email_id = %scheduled.email_id,
            status = %email.status,
            "Email is not in sendable state"
        );
        service.mark_failed(scheduled.id, &format!("Email status is {}", email.status)).await?;
        return Ok(());
    }

    // Build outgoing email
    let outgoing = OutgoingEmail {
        id: email.id.to_string(),
        from: email.from_address,
        to: email.to_addresses,
        cc: email.cc_addresses,
        bcc: email.bcc_addresses,
        subject: email.subject,
        body_text: email.body_text,
        body_html: email.body_html,
        reply_to: None,
        message_id: email.message_id,
        in_reply_to: email.in_reply_to,
        attachments: vec![], // Would need to fetch attachments separately
        is_encrypted: false,
    };

    // Initialize SMTP transport and send
    let smtp_config = SmtpConfig::default();
    let transport = SmtpTransport::new(smtp_config)?;

    match transport.send(&outgoing).await {
        Ok(response) => {
            info!(
                scheduled_id = %scheduled.id,
                email_id = %scheduled.email_id,
                response = %response,
                "Email sent successfully"
            );

            // Update email status in database
            sqlx::query(
                r#"
                UPDATE emails
                SET status = 'sent', sent_at = NOW(), updated_at = NOW()
                WHERE id = $1
                "#,
            )
            .bind(scheduled.email_id)
            .execute(pool)
            .await?;

            service.mark_sent(scheduled.id).await?;
        }
        Err(e) => {
            error!(
                scheduled_id = %scheduled.id,
                email_id = %scheduled.email_id,
                error = %e,
                "Failed to send email"
            );

            // Update email status to failed
            sqlx::query(
                r#"
                UPDATE emails
                SET status = 'failed', updated_at = NOW()
                WHERE id = $1
                "#,
            )
            .bind(scheduled.email_id)
            .execute(pool)
            .await?;

            service.mark_failed(scheduled.id, &e.to_string()).await?;
        }
    }

    Ok(())
}

/// Helper struct for fetching email data
#[derive(sqlx::FromRow)]
struct EmailRow {
    id: Uuid,
    from_address: String,
    to_addresses: Vec<String>,
    cc_addresses: Vec<String>,
    bcc_addresses: Vec<String>,
    subject: String,
    body_text: Option<String>,
    body_html: Option<String>,
    message_id: String,
    in_reply_to: Option<String>,
    status: String,
}

/// Migration for scheduled emails
pub const SCHEDULED_EMAIL_MIGRATION: &str = r#"
DO $$ BEGIN
    CREATE TYPE scheduled_status AS ENUM ('pending', 'sending', 'sent', 'cancelled', 'failed');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

CREATE TABLE IF NOT EXISTS scheduled_emails (
    id UUID PRIMARY KEY,
    email_id UUID NOT NULL REFERENCES emails(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    scheduled_at TIMESTAMPTZ NOT NULL,
    timezone VARCHAR(50) NOT NULL DEFAULT 'UTC',
    status scheduled_status NOT NULL DEFAULT 'pending',
    retry_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scheduled_emails_status_time
    ON scheduled_emails(status, scheduled_at)
    WHERE status = 'pending';

CREATE INDEX IF NOT EXISTS idx_scheduled_emails_user
    ON scheduled_emails(user_id, status);
"#;
