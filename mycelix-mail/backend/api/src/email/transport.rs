// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Email Transport Layer
//!
//! SMTP/IMAP integration for sending and receiving emails

use async_trait::async_trait;
use lettre::{
    message::{header::ContentType, Mailbox, MultiPart, SinglePart},
    transport::smtp::{authentication::Credentials, PoolConfig},
    AsyncSmtpTransport, AsyncTransport, Message, Tokio1Executor,
};
use std::time::Duration;
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

#[derive(Error, Debug)]
pub enum TransportError {
    #[error("SMTP error: {0}")]
    Smtp(#[from] lettre::transport::smtp::Error),

    #[error("Message build error: {0}")]
    MessageBuild(#[from] lettre::error::Error),

    #[error("Address parse error: {0}")]
    AddressParse(#[from] lettre::address::AddressError),

    #[error("Queue error: {0}")]
    Queue(String),

    #[error("Connection error: {0}")]
    Connection(String),
}

pub type TransportResult<T> = Result<T, TransportError>;

/// SMTP configuration
#[derive(Clone, Debug)]
pub struct SmtpConfig {
    pub host: String,
    pub port: u16,
    pub username: String,
    pub password: String,
    pub from_address: String,
    pub from_name: String,
    pub use_tls: bool,
    pub pool_size: u32,
    pub timeout_secs: u64,
}

impl Default for SmtpConfig {
    fn default() -> Self {
        Self {
            host: std::env::var("SMTP_HOST").unwrap_or_else(|_| "localhost".to_string()),
            port: std::env::var("SMTP_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(587),
            username: std::env::var("SMTP_USERNAME").unwrap_or_default(),
            password: std::env::var("SMTP_PASSWORD").unwrap_or_default(),
            from_address: std::env::var("SMTP_FROM_ADDRESS")
                .unwrap_or_else(|_| "noreply@mycelix.mail".to_string()),
            from_name: std::env::var("SMTP_FROM_NAME")
                .unwrap_or_else(|_| "Mycelix Mail".to_string()),
            use_tls: true,
            pool_size: 4,
            timeout_secs: 30,
        }
    }
}

/// Email message to send
#[derive(Clone, Debug)]
pub struct OutgoingEmail {
    pub id: String,
    pub from: String,
    pub to: Vec<String>,
    pub cc: Vec<String>,
    pub bcc: Vec<String>,
    pub subject: String,
    pub body_text: Option<String>,
    pub body_html: Option<String>,
    pub reply_to: Option<String>,
    pub message_id: String,
    pub in_reply_to: Option<String>,
    pub attachments: Vec<EmailAttachment>,
    pub is_encrypted: bool,
}

#[derive(Clone, Debug)]
pub struct EmailAttachment {
    pub filename: String,
    pub content_type: String,
    pub data: Vec<u8>,
}

/// Delivery status
#[derive(Clone, Debug)]
pub enum DeliveryStatus {
    Queued,
    Sending,
    Sent { message_id: String },
    Failed { error: String, retries: u32 },
    Bounced { reason: String },
}

/// SMTP transport for sending emails
pub struct SmtpTransport {
    mailer: AsyncSmtpTransport<Tokio1Executor>,
    config: SmtpConfig,
}

impl SmtpTransport {
    /// Create a new SMTP transport
    pub fn new(config: SmtpConfig) -> TransportResult<Self> {
        let creds = Credentials::new(config.username.clone(), config.password.clone());

        let mailer = if config.use_tls {
            AsyncSmtpTransport::<Tokio1Executor>::starttls_relay(&config.host)?
        } else {
            AsyncSmtpTransport::<Tokio1Executor>::relay(&config.host)?
        }
        .credentials(creds)
        .port(config.port)
        .timeout(Some(Duration::from_secs(config.timeout_secs)))
        .pool_config(PoolConfig::new().max_size(config.pool_size))
        .build();

        Ok(Self { mailer, config })
    }

    /// Send an email
    pub async fn send(&self, email: &OutgoingEmail) -> TransportResult<String> {
        let from_mailbox: Mailbox = format!("{} <{}>", self.config.from_name, email.from)
            .parse()
            .map_err(|e| TransportError::AddressParse(e))?;

        let mut builder = Message::builder()
            .from(from_mailbox)
            .subject(&email.subject)
            .message_id(Some(email.message_id.clone()));

        // Add recipients
        for to in &email.to {
            let mailbox: Mailbox = to.parse()?;
            builder = builder.to(mailbox);
        }

        for cc in &email.cc {
            let mailbox: Mailbox = cc.parse()?;
            builder = builder.cc(mailbox);
        }

        for bcc in &email.bcc {
            let mailbox: Mailbox = bcc.parse()?;
            builder = builder.bcc(mailbox);
        }

        // Reply-To header
        if let Some(reply_to) = &email.reply_to {
            let mailbox: Mailbox = reply_to.parse()?;
            builder = builder.reply_to(mailbox);
        }

        // In-Reply-To header
        if let Some(in_reply_to) = &email.in_reply_to {
            builder = builder.in_reply_to(in_reply_to.clone());
        }

        // Build message body
        let message = if email.attachments.is_empty() {
            // Simple message
            if let Some(html) = &email.body_html {
                if let Some(text) = &email.body_text {
                    builder.multipart(
                        MultiPart::alternative()
                            .singlepart(SinglePart::plain(text.clone()))
                            .singlepart(SinglePart::html(html.clone())),
                    )?
                } else {
                    builder.header(ContentType::TEXT_HTML).body(html.clone())?
                }
            } else {
                builder.body(email.body_text.clone().unwrap_or_default())?
            }
        } else {
            // Message with attachments
            let mut multipart = MultiPart::mixed();

            // Add body
            if let Some(html) = &email.body_html {
                if let Some(text) = &email.body_text {
                    multipart = multipart.multipart(
                        MultiPart::alternative()
                            .singlepart(SinglePart::plain(text.clone()))
                            .singlepart(SinglePart::html(html.clone())),
                    );
                } else {
                    multipart = multipart.singlepart(SinglePart::html(html.clone()));
                }
            } else if let Some(text) = &email.body_text {
                multipart = multipart.singlepart(SinglePart::plain(text.clone()));
            }

            // Add attachments
            for attachment in &email.attachments {
                let content_type: ContentType = attachment
                    .content_type
                    .parse()
                    .unwrap_or(ContentType::APPLICATION_OCTET_STREAM);

                multipart = multipart.singlepart(
                    SinglePart::builder()
                        .header(content_type)
                        .header(lettre::message::header::ContentDisposition::attachment(
                            &attachment.filename,
                        ))
                        .body(attachment.data.clone()),
                );
            }

            builder.multipart(multipart)?
        };

        // Send
        let response = self.mailer.send(message).await?;

        info!(
            email_id = %email.id,
            message_id = %email.message_id,
            "Email sent successfully"
        );

        Ok(response.message().map(|m| m.to_string()).unwrap_or_default())
    }

    /// Check connection health
    pub async fn health_check(&self) -> TransportResult<bool> {
        self.mailer.test_connection().await.map_err(|e| {
            error!("SMTP health check failed: {}", e);
            TransportError::Connection(e.to_string())
        })
    }
}

/// Email queue for async sending with retries
pub struct EmailQueue {
    sender: mpsc::Sender<QueuedEmail>,
}

#[derive(Clone)]
struct QueuedEmail {
    email: OutgoingEmail,
    retries: u32,
    callback: Option<String>, // Callback URL for delivery status
}

/// Email delivery status repository for database updates
pub struct DeliveryStatusRepository {
    pool: sqlx::PgPool,
}

impl DeliveryStatusRepository {
    pub fn new(pool: sqlx::PgPool) -> Self {
        Self { pool }
    }

    /// Update email status to delivered/sent
    pub async fn mark_delivered(&self, email_id: &str, smtp_response: &str) -> Result<(), sqlx::Error> {
        // Parse email_id as UUID
        let id = match uuid::Uuid::parse_str(email_id) {
            Ok(id) => id,
            Err(_) => {
                warn!(email_id = %email_id, "Invalid email ID format, skipping status update");
                return Ok(());
            }
        };

        sqlx::query(
            r#"
            UPDATE emails
            SET status = 'delivered',
                sent_at = NOW(),
                updated_at = NOW()
            WHERE id = $1
            "#,
        )
        .bind(id)
        .execute(&self.pool)
        .await?;

        // Log delivery event for analytics
        sqlx::query(
            r#"
            INSERT INTO email_delivery_events (email_id, event_type, smtp_response, created_at)
            VALUES ($1, 'delivered', $2, NOW())
            ON CONFLICT DO NOTHING
            "#,
        )
        .bind(id)
        .bind(smtp_response)
        .execute(&self.pool)
        .await
        .ok(); // Ignore if table doesn't exist

        info!(email_id = %email_id, "Delivery status updated to 'delivered'");
        Ok(())
    }

    /// Update email status to failed
    pub async fn mark_failed(&self, email_id: &str, error: &str, retry_count: u32) -> Result<(), sqlx::Error> {
        let id = match uuid::Uuid::parse_str(email_id) {
            Ok(id) => id,
            Err(_) => {
                warn!(email_id = %email_id, "Invalid email ID format, skipping status update");
                return Ok(());
            }
        };

        sqlx::query(
            r#"
            UPDATE emails
            SET status = 'failed',
                updated_at = NOW()
            WHERE id = $1
            "#,
        )
        .bind(id)
        .execute(&self.pool)
        .await?;

        // Log failure event for debugging
        sqlx::query(
            r#"
            INSERT INTO email_delivery_events (email_id, event_type, error_message, retry_count, created_at)
            VALUES ($1, 'failed', $2, $3, NOW())
            ON CONFLICT DO NOTHING
            "#,
        )
        .bind(id)
        .bind(error)
        .bind(retry_count as i32)
        .execute(&self.pool)
        .await
        .ok(); // Ignore if table doesn't exist

        error!(email_id = %email_id, error = %error, "Delivery status updated to 'failed'");
        Ok(())
    }

    /// Update email status to bounced
    pub async fn mark_bounced(&self, email_id: &str, reason: &str) -> Result<(), sqlx::Error> {
        let id = match uuid::Uuid::parse_str(email_id) {
            Ok(id) => id,
            Err(_) => return Ok(()),
        };

        sqlx::query(
            r#"
            UPDATE emails
            SET status = 'bounced',
                updated_at = NOW()
            WHERE id = $1
            "#,
        )
        .bind(id)
        .execute(&self.pool)
        .await?;

        info!(email_id = %email_id, reason = %reason, "Delivery status updated to 'bounced'");
        Ok(())
    }
}

impl EmailQueue {
    /// Create a new email queue with background worker
    pub fn new(transport: SmtpTransport, max_retries: u32) -> Self {
        Self::new_with_db(transport, max_retries, None)
    }

    /// Create a new email queue with database support for delivery tracking
    pub fn new_with_db(transport: SmtpTransport, max_retries: u32, db_pool: Option<sqlx::PgPool>) -> Self {
        let (sender, mut receiver) = mpsc::channel::<QueuedEmail>(1000);
        let delivery_repo = db_pool.map(DeliveryStatusRepository::new);

        // Spawn background worker
        tokio::spawn(async move {
            while let Some(mut queued) = receiver.recv().await {
                match transport.send(&queued.email).await {
                    Ok(response) => {
                        info!(
                            email_id = %queued.email.id,
                            "Email delivered: {}",
                            response
                        );
                        // Update delivery status in database
                        if let Some(ref repo) = delivery_repo {
                            if let Err(e) = repo.mark_delivered(&queued.email.id, &response).await {
                                error!(
                                    email_id = %queued.email.id,
                                    error = %e,
                                    "Failed to update delivery status"
                                );
                            }
                        }
                    }
                    Err(e) => {
                        queued.retries += 1;
                        if queued.retries <= max_retries {
                            warn!(
                                email_id = %queued.email.id,
                                retry = queued.retries,
                                "Email send failed, retrying: {}",
                                e
                            );
                            // Exponential backoff
                            tokio::time::sleep(Duration::from_secs(2u64.pow(queued.retries))).await;
                            // Re-queue (in production, use a proper retry mechanism)
                        } else {
                            error!(
                                email_id = %queued.email.id,
                                "Email send failed after {} retries: {}",
                                max_retries,
                                e
                            );
                            // Mark as failed in database
                            if let Some(ref repo) = delivery_repo {
                                if let Err(db_err) = repo.mark_failed(
                                    &queued.email.id,
                                    &e.to_string(),
                                    queued.retries,
                                ).await {
                                    error!(
                                        email_id = %queued.email.id,
                                        error = %db_err,
                                        "Failed to update failure status"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        });

        Self { sender }
    }

    /// Queue an email for sending
    pub async fn enqueue(&self, email: OutgoingEmail) -> TransportResult<()> {
        self.sender
            .send(QueuedEmail {
                email,
                retries: 0,
                callback: None,
            })
            .await
            .map_err(|e| TransportError::Queue(e.to_string()))
    }

    /// Queue with delivery callback
    pub async fn enqueue_with_callback(
        &self,
        email: OutgoingEmail,
        callback_url: String,
    ) -> TransportResult<()> {
        self.sender
            .send(QueuedEmail {
                email,
                retries: 0,
                callback: Some(callback_url),
            })
            .await
            .map_err(|e| TransportError::Queue(e.to_string()))
    }
}

/// Bounce handler for processing bounced emails
pub struct BounceHandler;

impl BounceHandler {
    /// Parse bounce notification
    pub fn parse_bounce(raw_email: &str) -> Option<BounceInfo> {
        // Parse DSN (Delivery Status Notification) format
        // This is a simplified implementation
        if raw_email.contains("Delivery Status Notification")
            || raw_email.contains("Mail Delivery Failed")
        {
            // Extract original recipient and reason
            let recipient = Self::extract_header(raw_email, "Final-Recipient:");
            let diagnostic = Self::extract_header(raw_email, "Diagnostic-Code:");

            return Some(BounceInfo {
                recipient: recipient.unwrap_or_default(),
                reason: diagnostic.unwrap_or_else(|| "Unknown bounce reason".to_string()),
                is_permanent: raw_email.contains("5.") || raw_email.contains("permanent"),
            });
        }

        None
    }

    fn extract_header(email: &str, header: &str) -> Option<String> {
        email.lines().find(|l| l.starts_with(header)).map(|l| {
            l.trim_start_matches(header)
                .trim()
                .to_string()
        })
    }
}

#[derive(Debug)]
pub struct BounceInfo {
    pub recipient: String,
    pub reason: String,
    pub is_permanent: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_smtp_config() {
        let config = SmtpConfig::default();
        assert_eq!(config.port, 587);
        assert!(config.use_tls);
    }

    #[test]
    fn test_bounce_detection() {
        let bounce_email = r#"
Subject: Delivery Status Notification (Failure)
Final-Recipient: rfc822; invalid@example.com
Diagnostic-Code: smtp; 550 5.1.1 User unknown
        "#;

        let bounce = BounceHandler::parse_bounce(bounce_email);
        assert!(bounce.is_some());

        let info = bounce.unwrap();
        assert!(info.recipient.contains("invalid@example.com"));
        assert!(info.is_permanent);
    }
}
