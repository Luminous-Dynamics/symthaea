// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Webhook System
//!
//! Event-driven webhooks for third-party integrations

use chrono::{DateTime, Duration, Utc};
use hmac::{Hmac, Mac};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use sqlx::PgPool;
use tokio::sync::mpsc;
use tracing::{error, info, warn};
use uuid::Uuid;

type HmacSha256 = Hmac<Sha256>;

/// Webhook event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "webhook_event_type", rename_all = "snake_case")]
pub enum WebhookEventType {
    EmailReceived,
    EmailSent,
    EmailRead,
    EmailDeleted,
    EmailBounced,
    EmailSpam,
    ContactCreated,
    ContactUpdated,
    TrustScoreChanged,
    AttestationCreated,
    UserCreated,
    UserUpdated,
}

impl WebhookEventType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::EmailReceived => "email.received",
            Self::EmailSent => "email.sent",
            Self::EmailRead => "email.read",
            Self::EmailDeleted => "email.deleted",
            Self::EmailBounced => "email.bounced",
            Self::EmailSpam => "email.spam",
            Self::ContactCreated => "contact.created",
            Self::ContactUpdated => "contact.updated",
            Self::TrustScoreChanged => "trust.score_changed",
            Self::AttestationCreated => "trust.attestation_created",
            Self::UserCreated => "user.created",
            Self::UserUpdated => "user.updated",
        }
    }
}

/// Webhook subscription
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct WebhookSubscription {
    pub id: Uuid,
    pub user_id: Uuid,
    pub tenant_id: Option<Uuid>,
    pub name: String,
    pub url: String,
    pub secret: String,
    pub events: Vec<String>,
    pub is_active: bool,
    pub retry_count: i32,
    pub last_triggered_at: Option<DateTime<Utc>>,
    pub last_success_at: Option<DateTime<Utc>>,
    pub last_failure_at: Option<DateTime<Utc>>,
    pub failure_count: i32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Webhook delivery record
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct WebhookDelivery {
    pub id: Uuid,
    pub subscription_id: Uuid,
    pub event_type: String,
    pub payload: serde_json::Value,
    pub status: DeliveryStatus,
    pub response_status: Option<i32>,
    pub response_body: Option<String>,
    pub attempt_count: i32,
    pub next_retry_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub delivered_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "delivery_status", rename_all = "lowercase")]
pub enum DeliveryStatus {
    Pending,
    Delivered,
    Failed,
    Retrying,
}

/// Webhook payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookPayload {
    pub id: String,
    pub event_type: String,
    pub timestamp: DateTime<Utc>,
    pub data: serde_json::Value,
}

/// Webhook service
pub struct WebhookService {
    pool: PgPool,
    client: Client,
    sender: mpsc::Sender<WebhookDeliveryTask>,
}

struct WebhookDeliveryTask {
    delivery_id: Uuid,
    subscription: WebhookSubscription,
    payload: WebhookPayload,
}

impl WebhookService {
    pub fn new(pool: PgPool) -> (Self, mpsc::Receiver<WebhookDeliveryTask>) {
        let (sender, receiver) = mpsc::channel(1000);
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        (Self { pool, client, sender }, receiver)
    }

    /// Create a new webhook subscription
    pub async fn create_subscription(
        &self,
        user_id: Uuid,
        tenant_id: Option<Uuid>,
        name: &str,
        url: &str,
        events: Vec<WebhookEventType>,
    ) -> Result<WebhookSubscription, WebhookError> {
        // Validate URL
        if !url.starts_with("https://") {
            return Err(WebhookError::Validation(
                "Webhook URL must use HTTPS".to_string(),
            ));
        }

        // Generate secret for signature verification
        let secret = generate_secret();
        let event_strings: Vec<String> = events.iter().map(|e| e.as_str().to_string()).collect();

        let subscription = sqlx::query_as::<_, WebhookSubscription>(
            r#"
            INSERT INTO webhook_subscriptions (
                id, user_id, tenant_id, name, url, secret, events,
                is_active, retry_count, failure_count, created_at, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, true, 3, 0, NOW(), NOW())
            RETURNING *
            "#,
        )
        .bind(Uuid::new_v4())
        .bind(user_id)
        .bind(tenant_id)
        .bind(name)
        .bind(url)
        .bind(&secret)
        .bind(&event_strings)
        .fetch_one(&self.pool)
        .await
        .map_err(WebhookError::Database)?;

        info!(
            subscription_id = %subscription.id,
            url = %url,
            events = ?event_strings,
            "Webhook subscription created"
        );

        Ok(subscription)
    }

    /// Trigger webhooks for an event
    pub async fn trigger(
        &self,
        event_type: WebhookEventType,
        tenant_id: Option<Uuid>,
        data: serde_json::Value,
    ) -> Result<usize, WebhookError> {
        let event_str = event_type.as_str();

        // Find matching subscriptions
        let subscriptions = sqlx::query_as::<_, WebhookSubscription>(
            r#"
            SELECT * FROM webhook_subscriptions
            WHERE is_active = true
              AND ($1::uuid IS NULL OR tenant_id IS NULL OR tenant_id = $1)
              AND $2 = ANY(events)
              AND failure_count < 10
            "#,
        )
        .bind(tenant_id)
        .bind(event_str)
        .fetch_all(&self.pool)
        .await
        .map_err(WebhookError::Database)?;

        let count = subscriptions.len();

        for subscription in subscriptions {
            let payload = WebhookPayload {
                id: Uuid::new_v4().to_string(),
                event_type: event_str.to_string(),
                timestamp: Utc::now(),
                data: data.clone(),
            };

            // Create delivery record
            let delivery_id = Uuid::new_v4();
            sqlx::query(
                r#"
                INSERT INTO webhook_deliveries (
                    id, subscription_id, event_type, payload,
                    status, attempt_count, created_at
                )
                VALUES ($1, $2, $3, $4, 'pending', 0, NOW())
                "#,
            )
            .bind(delivery_id)
            .bind(subscription.id)
            .bind(event_str)
            .bind(serde_json::to_value(&payload).unwrap())
            .execute(&self.pool)
            .await
            .map_err(WebhookError::Database)?;

            // Queue delivery
            let task = WebhookDeliveryTask {
                delivery_id,
                subscription,
                payload,
            };

            if let Err(e) = self.sender.send(task).await {
                error!(error = %e, "Failed to queue webhook delivery");
            }
        }

        Ok(count)
    }

    /// Deliver a webhook
    pub async fn deliver(&self, task: WebhookDeliveryTask) -> Result<(), WebhookError> {
        let payload_json = serde_json::to_string(&task.payload)
            .map_err(|e| WebhookError::Serialization(e.to_string()))?;

        // Generate signature
        let signature = sign_payload(&payload_json, &task.subscription.secret);

        // Send request
        let response = self
            .client
            .post(&task.subscription.url)
            .header("Content-Type", "application/json")
            .header("X-Webhook-Signature", &signature)
            .header("X-Webhook-Event", &task.payload.event_type)
            .header("X-Webhook-ID", &task.payload.id)
            .body(payload_json)
            .send()
            .await;

        match response {
            Ok(resp) => {
                let status = resp.status().as_u16() as i32;
                let body = resp.text().await.ok();

                if status >= 200 && status < 300 {
                    // Success
                    sqlx::query(
                        r#"
                        UPDATE webhook_deliveries
                        SET status = 'delivered',
                            response_status = $2,
                            response_body = $3,
                            attempt_count = attempt_count + 1,
                            delivered_at = NOW()
                        WHERE id = $1
                        "#,
                    )
                    .bind(task.delivery_id)
                    .bind(status)
                    .bind(&body)
                    .execute(&self.pool)
                    .await
                    .map_err(WebhookError::Database)?;

                    // Update subscription success
                    sqlx::query(
                        r#"
                        UPDATE webhook_subscriptions
                        SET last_triggered_at = NOW(),
                            last_success_at = NOW(),
                            failure_count = 0,
                            updated_at = NOW()
                        WHERE id = $1
                        "#,
                    )
                    .bind(task.subscription.id)
                    .execute(&self.pool)
                    .await
                    .map_err(WebhookError::Database)?;

                    info!(
                        delivery_id = %task.delivery_id,
                        subscription_id = %task.subscription.id,
                        "Webhook delivered successfully"
                    );

                    Ok(())
                } else {
                    // Failed with non-2xx status
                    self.handle_failure(
                        &task,
                        status,
                        body.as_deref(),
                    ).await
                }
            }
            Err(e) => {
                // Network or timeout error
                self.handle_failure(&task, 0, Some(&e.to_string())).await
            }
        }
    }

    async fn handle_failure(
        &self,
        task: &WebhookDeliveryTask,
        status: i32,
        error: Option<&str>,
    ) -> Result<(), WebhookError> {
        // Get current attempt count
        let delivery: WebhookDelivery = sqlx::query_as(
            "SELECT * FROM webhook_deliveries WHERE id = $1",
        )
        .bind(task.delivery_id)
        .fetch_one(&self.pool)
        .await
        .map_err(WebhookError::Database)?;

        let new_attempt = delivery.attempt_count + 1;

        if new_attempt < task.subscription.retry_count {
            // Schedule retry with exponential backoff
            let delay = Duration::seconds(2_i64.pow(new_attempt as u32) * 60);
            let next_retry = Utc::now() + delay;

            sqlx::query(
                r#"
                UPDATE webhook_deliveries
                SET status = 'retrying',
                    response_status = $2,
                    response_body = $3,
                    attempt_count = $4,
                    next_retry_at = $5
                WHERE id = $1
                "#,
            )
            .bind(task.delivery_id)
            .bind(status)
            .bind(error)
            .bind(new_attempt)
            .bind(next_retry)
            .execute(&self.pool)
            .await
            .map_err(WebhookError::Database)?;

            warn!(
                delivery_id = %task.delivery_id,
                attempt = new_attempt,
                next_retry = %next_retry,
                "Webhook delivery failed, scheduling retry"
            );
        } else {
            // Max retries exceeded
            sqlx::query(
                r#"
                UPDATE webhook_deliveries
                SET status = 'failed',
                    response_status = $2,
                    response_body = $3,
                    attempt_count = $4
                WHERE id = $1
                "#,
            )
            .bind(task.delivery_id)
            .bind(status)
            .bind(error)
            .bind(new_attempt)
            .execute(&self.pool)
            .await
            .map_err(WebhookError::Database)?;

            // Increment subscription failure count
            sqlx::query(
                r#"
                UPDATE webhook_subscriptions
                SET last_triggered_at = NOW(),
                    last_failure_at = NOW(),
                    failure_count = failure_count + 1,
                    updated_at = NOW()
                WHERE id = $1
                "#,
            )
            .bind(task.subscription.id)
            .execute(&self.pool)
            .await
            .map_err(WebhookError::Database)?;

            error!(
                delivery_id = %task.delivery_id,
                subscription_id = %task.subscription.id,
                "Webhook delivery failed permanently"
            );
        }

        Ok(())
    }

    /// Get subscription by ID
    pub async fn get_subscription(
        &self,
        subscription_id: Uuid,
        user_id: Uuid,
    ) -> Result<Option<WebhookSubscription>, sqlx::Error> {
        sqlx::query_as::<_, WebhookSubscription>(
            "SELECT * FROM webhook_subscriptions WHERE id = $1 AND user_id = $2",
        )
        .bind(subscription_id)
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await
    }

    /// List user's subscriptions
    pub async fn list_subscriptions(
        &self,
        user_id: Uuid,
    ) -> Result<Vec<WebhookSubscription>, sqlx::Error> {
        sqlx::query_as::<_, WebhookSubscription>(
            "SELECT * FROM webhook_subscriptions WHERE user_id = $1 ORDER BY created_at DESC",
        )
        .bind(user_id)
        .fetch_all(&self.pool)
        .await
    }

    /// Delete subscription
    pub async fn delete_subscription(
        &self,
        subscription_id: Uuid,
        user_id: Uuid,
    ) -> Result<bool, sqlx::Error> {
        let result = sqlx::query(
            "DELETE FROM webhook_subscriptions WHERE id = $1 AND user_id = $2",
        )
        .bind(subscription_id)
        .bind(user_id)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected() > 0)
    }

    /// Get delivery history for a subscription
    pub async fn get_deliveries(
        &self,
        subscription_id: Uuid,
        limit: i32,
    ) -> Result<Vec<WebhookDelivery>, sqlx::Error> {
        sqlx::query_as::<_, WebhookDelivery>(
            r#"
            SELECT * FROM webhook_deliveries
            WHERE subscription_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            "#,
        )
        .bind(subscription_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await
    }
}

/// Generate a random secret for webhook signing
fn generate_secret() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let bytes: Vec<u8> = (0..32).map(|_| rng.gen()).collect();
    hex::encode(bytes)
}

/// Sign payload with HMAC-SHA256
fn sign_payload(payload: &str, secret: &str) -> String {
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
        .expect("HMAC can take key of any size");
    mac.update(payload.as_bytes());
    let result = mac.finalize();
    format!("sha256={}", hex::encode(result.into_bytes()))
}

/// Verify webhook signature
pub fn verify_signature(payload: &str, signature: &str, secret: &str) -> bool {
    let expected = sign_payload(payload, secret);
    // Constant-time comparison
    use subtle::ConstantTimeEq;
    expected.as_bytes().ct_eq(signature.as_bytes()).into()
}

/// Webhook errors
#[derive(Debug)]
pub enum WebhookError {
    Validation(String),
    Serialization(String),
    Database(sqlx::Error),
    Delivery(String),
}

impl std::fmt::Display for WebhookError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Validation(msg) => write!(f, "Validation error: {}", msg),
            Self::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            Self::Database(e) => write!(f, "Database error: {}", e),
            Self::Delivery(msg) => write!(f, "Delivery error: {}", msg),
        }
    }
}

impl std::error::Error for WebhookError {}

/// Background worker for webhook delivery
pub async fn webhook_worker(
    service: WebhookService,
    mut receiver: mpsc::Receiver<WebhookDeliveryTask>,
    mut shutdown: mpsc::Receiver<()>,
) {
    info!("Webhook worker started");

    loop {
        tokio::select! {
            _ = shutdown.recv() => {
                info!("Webhook worker shutting down");
                break;
            }
            Some(task) = receiver.recv() => {
                if let Err(e) = service.deliver(task).await {
                    error!(error = %e, "Webhook delivery error");
                }
            }
        }
    }
}

/// Retry worker for failed deliveries
pub async fn webhook_retry_worker(
    pool: PgPool,
    sender: mpsc::Sender<WebhookDeliveryTask>,
    mut shutdown: mpsc::Receiver<()>,
) {
    info!("Webhook retry worker started");

    loop {
        tokio::select! {
            _ = shutdown.recv() => {
                info!("Webhook retry worker shutting down");
                break;
            }
            _ = tokio::time::sleep(std::time::Duration::from_secs(60)) => {
                // Find deliveries ready for retry
                let deliveries: Vec<WebhookDelivery> = sqlx::query_as(
                    r#"
                    SELECT * FROM webhook_deliveries
                    WHERE status = 'retrying'
                      AND next_retry_at <= NOW()
                    LIMIT 100
                    "#,
                )
                .fetch_all(&pool)
                .await
                .unwrap_or_default();

                for delivery in deliveries {
                    // Get subscription
                    if let Ok(Some(subscription)) = sqlx::query_as::<_, WebhookSubscription>(
                        "SELECT * FROM webhook_subscriptions WHERE id = $1",
                    )
                    .bind(delivery.subscription_id)
                    .fetch_optional(&pool)
                    .await
                    {
                        let payload: WebhookPayload = serde_json::from_value(delivery.payload)
                            .unwrap_or_else(|_| WebhookPayload {
                                id: delivery.id.to_string(),
                                event_type: delivery.event_type,
                                timestamp: Utc::now(),
                                data: serde_json::json!({}),
                            });

                        let task = WebhookDeliveryTask {
                            delivery_id: delivery.id,
                            subscription,
                            payload,
                        };

                        if let Err(e) = sender.send(task).await {
                            error!(error = %e, "Failed to queue webhook retry");
                        }
                    }
                }
            }
        }
    }
}

/// Migration for webhooks
pub const WEBHOOK_MIGRATION: &str = r#"
DO $$ BEGIN
    CREATE TYPE webhook_event_type AS ENUM (
        'email_received', 'email_sent', 'email_read', 'email_deleted',
        'email_bounced', 'email_spam', 'contact_created', 'contact_updated',
        'trust_score_changed', 'attestation_created', 'user_created', 'user_updated'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE delivery_status AS ENUM ('pending', 'delivered', 'failed', 'retrying');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

CREATE TABLE IF NOT EXISTS webhook_subscriptions (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    url TEXT NOT NULL,
    secret VARCHAR(64) NOT NULL,
    events TEXT[] NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT true,
    retry_count INTEGER NOT NULL DEFAULT 3,
    last_triggered_at TIMESTAMPTZ,
    last_success_at TIMESTAMPTZ,
    last_failure_at TIMESTAMPTZ,
    failure_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS webhook_deliveries (
    id UUID PRIMARY KEY,
    subscription_id UUID NOT NULL REFERENCES webhook_subscriptions(id) ON DELETE CASCADE,
    event_type VARCHAR(100) NOT NULL,
    payload JSONB NOT NULL,
    status delivery_status NOT NULL DEFAULT 'pending',
    response_status INTEGER,
    response_body TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    next_retry_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    delivered_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_webhook_subscriptions_user ON webhook_subscriptions(user_id);
CREATE INDEX IF NOT EXISTS idx_webhook_subscriptions_tenant ON webhook_subscriptions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_webhook_subscriptions_active ON webhook_subscriptions(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_subscription ON webhook_deliveries(subscription_id);
CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_retry ON webhook_deliveries(status, next_retry_at) WHERE status = 'retrying';
"#;
