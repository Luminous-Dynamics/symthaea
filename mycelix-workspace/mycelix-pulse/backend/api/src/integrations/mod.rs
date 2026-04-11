// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integrations Hub
//!
//! Webhooks, API keys, and external service connectors

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use sqlx::PgPool;
use uuid::Uuid;
use std::collections::HashMap;
use hmac::{Hmac, Mac};
use sha2::Sha256;

// ============================================================================
// Webhook Service
// ============================================================================

pub struct WebhookService {
    pool: PgPool,
    http_client: reqwest::Client,
}

impl WebhookService {
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool,
            http_client: reqwest::Client::new(),
        }
    }

    /// Register a new webhook endpoint
    pub async fn create_webhook(
        &self,
        user_id: Uuid,
        webhook: WebhookInput,
    ) -> Result<Webhook, IntegrationError> {
        let id = Uuid::new_v4();
        let secret = self.generate_secret();

        sqlx::query(
            r#"
            INSERT INTO webhooks (id, user_id, name, url, secret, events, is_active, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, true, NOW())
            "#
        )
        .bind(id)
        .bind(user_id)
        .bind(&webhook.name)
        .bind(&webhook.url)
        .bind(&secret)
        .bind(&webhook.events)
        .execute(&self.pool)
        .await
        .map_err(|e| IntegrationError::Database(e.to_string()))?;

        Ok(Webhook {
            id,
            user_id,
            name: webhook.name,
            url: webhook.url,
            secret,
            events: webhook.events,
            is_active: true,
            created_at: Utc::now(),
            last_triggered_at: None,
            failure_count: 0,
        })
    }

    /// Trigger webhooks for an event
    pub async fn trigger_event(
        &self,
        user_id: Uuid,
        event: WebhookEvent,
        payload: JsonValue,
    ) -> Result<Vec<WebhookDelivery>, IntegrationError> {
        let webhooks: Vec<Webhook> = sqlx::query_as(
            r#"
            SELECT * FROM webhooks
            WHERE user_id = $1 AND is_active = true AND $2 = ANY(events)
            "#
        )
        .bind(user_id)
        .bind(event.to_string())
        .fetch_all(&self.pool)
        .await
        .map_err(|e| IntegrationError::Database(e.to_string()))?;

        let mut deliveries = Vec::new();

        for webhook in webhooks {
            let delivery = self.deliver_webhook(&webhook, &event, &payload).await;
            deliveries.push(delivery);
        }

        Ok(deliveries)
    }

    async fn deliver_webhook(
        &self,
        webhook: &Webhook,
        event: &WebhookEvent,
        payload: &JsonValue,
    ) -> WebhookDelivery {
        let delivery_id = Uuid::new_v4();
        let timestamp = Utc::now().timestamp();
        let signature = self.sign_payload(&webhook.secret, payload, timestamp);

        let body = serde_json::json!({
            "event": event.to_string(),
            "timestamp": timestamp,
            "payload": payload,
        });

        let result = self.http_client
            .post(&webhook.url)
            .header("Content-Type", "application/json")
            .header("X-Webhook-Id", webhook.id.to_string())
            .header("X-Webhook-Signature", &signature)
            .header("X-Webhook-Timestamp", timestamp.to_string())
            .json(&body)
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .await;

        let (status_code, response_body, success) = match result {
            Ok(response) => {
                let status = response.status().as_u16() as i32;
                let body = response.text().await.unwrap_or_default();
                let success = status >= 200 && status < 300;
                (status, body, success)
            }
            Err(e) => (0, e.to_string(), false),
        };

        // Log delivery
        sqlx::query(
            r#"
            INSERT INTO webhook_deliveries (id, webhook_id, event, payload, status_code, response_body, success, delivered_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
            "#
        )
        .bind(delivery_id)
        .bind(webhook.id)
        .bind(event.to_string())
        .bind(payload)
        .bind(status_code)
        .bind(&response_body)
        .bind(success)
        .execute(&self.pool)
        .await
        .ok();

        // Update webhook stats
        if success {
            sqlx::query("UPDATE webhooks SET last_triggered_at = NOW(), failure_count = 0 WHERE id = $1")
                .bind(webhook.id)
                .execute(&self.pool)
                .await
                .ok();
        } else {
            sqlx::query("UPDATE webhooks SET failure_count = failure_count + 1 WHERE id = $1")
                .bind(webhook.id)
                .execute(&self.pool)
                .await
                .ok();
        }

        WebhookDelivery {
            id: delivery_id,
            webhook_id: webhook.id,
            event: event.clone(),
            status_code,
            success,
            delivered_at: Utc::now(),
        }
    }

    fn generate_secret(&self) -> String {
        use rand::Rng;
        let bytes: [u8; 32] = rand::thread_rng().gen();
        hex::encode(bytes)
    }

    fn sign_payload(&self, secret: &str, payload: &JsonValue, timestamp: i64) -> String {
        let message = format!("{}.{}", timestamp, serde_json::to_string(payload).unwrap_or_default());
        let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes()).unwrap();
        mac.update(message.as_bytes());
        let result = mac.finalize();
        format!("sha256={}", hex::encode(result.into_bytes()))
    }

    /// Get webhooks for a user
    pub async fn get_webhooks(&self, user_id: Uuid) -> Result<Vec<Webhook>, IntegrationError> {
        let webhooks: Vec<Webhook> = sqlx::query_as(
            "SELECT * FROM webhooks WHERE user_id = $1 ORDER BY created_at DESC"
        )
        .bind(user_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| IntegrationError::Database(e.to_string()))?;

        Ok(webhooks)
    }

    /// Delete a webhook
    pub async fn delete_webhook(&self, user_id: Uuid, webhook_id: Uuid) -> Result<(), IntegrationError> {
        sqlx::query("DELETE FROM webhooks WHERE id = $1 AND user_id = $2")
            .bind(webhook_id)
            .bind(user_id)
            .execute(&self.pool)
            .await
            .map_err(|e| IntegrationError::Database(e.to_string()))?;

        Ok(())
    }
}

// ============================================================================
// API Key Service
// ============================================================================

pub struct ApiKeyService {
    pool: PgPool,
}

impl ApiKeyService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Create a new API key
    pub async fn create_api_key(
        &self,
        user_id: Uuid,
        name: &str,
        scopes: Vec<ApiScope>,
        expires_at: Option<DateTime<Utc>>,
    ) -> Result<ApiKeyCreated, IntegrationError> {
        let id = Uuid::new_v4();
        let key = self.generate_api_key();
        let key_hash = self.hash_key(&key);
        let key_prefix = &key[..12];

        sqlx::query(
            r#"
            INSERT INTO api_keys (id, user_id, name, key_hash, key_prefix, scopes, expires_at, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
            "#
        )
        .bind(id)
        .bind(user_id)
        .bind(name)
        .bind(&key_hash)
        .bind(key_prefix)
        .bind(&scopes.iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .bind(expires_at)
        .execute(&self.pool)
        .await
        .map_err(|e| IntegrationError::Database(e.to_string()))?;

        Ok(ApiKeyCreated {
            id,
            key, // Only returned once at creation
            name: name.to_string(),
            scopes,
            expires_at,
            created_at: Utc::now(),
        })
    }

    /// Validate an API key and return associated user
    pub async fn validate_key(&self, key: &str) -> Result<ApiKeyValidation, IntegrationError> {
        let key_hash = self.hash_key(key);

        let api_key: Option<ApiKey> = sqlx::query_as(
            r#"
            SELECT * FROM api_keys
            WHERE key_hash = $1 AND is_revoked = false
            AND (expires_at IS NULL OR expires_at > NOW())
            "#
        )
        .bind(&key_hash)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| IntegrationError::Database(e.to_string()))?;

        match api_key {
            Some(key) => {
                // Update last used
                sqlx::query("UPDATE api_keys SET last_used_at = NOW() WHERE id = $1")
                    .bind(key.id)
                    .execute(&self.pool)
                    .await
                    .ok();

                Ok(ApiKeyValidation {
                    valid: true,
                    user_id: Some(key.user_id),
                    scopes: key.scopes.iter().filter_map(|s| s.parse().ok()).collect(),
                })
            }
            None => Ok(ApiKeyValidation {
                valid: false,
                user_id: None,
                scopes: Vec::new(),
            }),
        }
    }

    /// Revoke an API key
    pub async fn revoke_key(&self, user_id: Uuid, key_id: Uuid) -> Result<(), IntegrationError> {
        sqlx::query("UPDATE api_keys SET is_revoked = true WHERE id = $1 AND user_id = $2")
            .bind(key_id)
            .bind(user_id)
            .execute(&self.pool)
            .await
            .map_err(|e| IntegrationError::Database(e.to_string()))?;

        Ok(())
    }

    /// List API keys for a user
    pub async fn list_keys(&self, user_id: Uuid) -> Result<Vec<ApiKey>, IntegrationError> {
        let keys: Vec<ApiKey> = sqlx::query_as(
            "SELECT * FROM api_keys WHERE user_id = $1 AND is_revoked = false ORDER BY created_at DESC"
        )
        .bind(user_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| IntegrationError::Database(e.to_string()))?;

        Ok(keys)
    }

    fn generate_api_key(&self) -> String {
        use rand::Rng;
        let bytes: [u8; 32] = rand::thread_rng().gen();
        format!("myx_{}", base64::encode(bytes).replace(['+', '/', '='], ""))
    }

    fn hash_key(&self, key: &str) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(key.as_bytes());
        hex::encode(hasher.finalize())
    }
}

// ============================================================================
// Connector Service (Slack, Discord, etc.)
// ============================================================================

pub struct ConnectorService {
    pool: PgPool,
    http_client: reqwest::Client,
}

impl ConnectorService {
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool,
            http_client: reqwest::Client::new(),
        }
    }

    /// Add a new connector
    pub async fn add_connector(
        &self,
        user_id: Uuid,
        connector: ConnectorInput,
    ) -> Result<Connector, IntegrationError> {
        let id = Uuid::new_v4();

        sqlx::query(
            r#"
            INSERT INTO connectors (id, user_id, connector_type, name, config, triggers, is_active, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, true, NOW())
            "#
        )
        .bind(id)
        .bind(user_id)
        .bind(connector.connector_type.to_string())
        .bind(&connector.name)
        .bind(&connector.config)
        .bind(&connector.triggers.iter().map(|t| t.to_string()).collect::<Vec<_>>())
        .execute(&self.pool)
        .await
        .map_err(|e| IntegrationError::Database(e.to_string()))?;

        Ok(Connector {
            id,
            user_id,
            connector_type: connector.connector_type,
            name: connector.name,
            config: connector.config,
            triggers: connector.triggers,
            is_active: true,
            created_at: Utc::now(),
        })
    }

    /// Send notification via connector
    pub async fn notify(
        &self,
        user_id: Uuid,
        trigger: ConnectorTrigger,
        message: &str,
    ) -> Result<(), IntegrationError> {
        let connectors: Vec<Connector> = sqlx::query_as(
            r#"
            SELECT * FROM connectors
            WHERE user_id = $1 AND is_active = true AND $2 = ANY(triggers)
            "#
        )
        .bind(user_id)
        .bind(trigger.to_string())
        .fetch_all(&self.pool)
        .await
        .map_err(|e| IntegrationError::Database(e.to_string()))?;

        for connector in connectors {
            match connector.connector_type {
                ConnectorType::Slack => {
                    self.send_slack_notification(&connector.config, message).await?;
                }
                ConnectorType::Discord => {
                    self.send_discord_notification(&connector.config, message).await?;
                }
                ConnectorType::Teams => {
                    self.send_teams_notification(&connector.config, message).await?;
                }
                ConnectorType::Email => {
                    // Internal email notification
                }
                ConnectorType::Custom => {
                    // Custom webhook
                    if let Some(url) = connector.config.get("url").and_then(|v| v.as_str()) {
                        self.http_client
                            .post(url)
                            .json(&serde_json::json!({ "message": message }))
                            .send()
                            .await
                            .ok();
                    }
                }
            }
        }

        Ok(())
    }

    async fn send_slack_notification(&self, config: &JsonValue, message: &str) -> Result<(), IntegrationError> {
        let webhook_url = config.get("webhook_url")
            .and_then(|v| v.as_str())
            .ok_or(IntegrationError::Configuration("Missing Slack webhook URL".to_string()))?;

        self.http_client
            .post(webhook_url)
            .json(&serde_json::json!({
                "text": message,
            }))
            .send()
            .await
            .map_err(|e| IntegrationError::ExternalApi(e.to_string()))?;

        Ok(())
    }

    async fn send_discord_notification(&self, config: &JsonValue, message: &str) -> Result<(), IntegrationError> {
        let webhook_url = config.get("webhook_url")
            .and_then(|v| v.as_str())
            .ok_or(IntegrationError::Configuration("Missing Discord webhook URL".to_string()))?;

        self.http_client
            .post(webhook_url)
            .json(&serde_json::json!({
                "content": message,
            }))
            .send()
            .await
            .map_err(|e| IntegrationError::ExternalApi(e.to_string()))?;

        Ok(())
    }

    async fn send_teams_notification(&self, config: &JsonValue, message: &str) -> Result<(), IntegrationError> {
        let webhook_url = config.get("webhook_url")
            .and_then(|v| v.as_str())
            .ok_or(IntegrationError::Configuration("Missing Teams webhook URL".to_string()))?;

        self.http_client
            .post(webhook_url)
            .json(&serde_json::json!({
                "@type": "MessageCard",
                "text": message,
            }))
            .send()
            .await
            .map_err(|e| IntegrationError::ExternalApi(e.to_string()))?;

        Ok(())
    }

    /// Get connectors for a user
    pub async fn get_connectors(&self, user_id: Uuid) -> Result<Vec<Connector>, IntegrationError> {
        let connectors: Vec<Connector> = sqlx::query_as(
            "SELECT * FROM connectors WHERE user_id = $1 ORDER BY created_at DESC"
        )
        .bind(user_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| IntegrationError::Database(e.to_string()))?;

        Ok(connectors)
    }

    /// Delete a connector
    pub async fn delete_connector(&self, user_id: Uuid, connector_id: Uuid) -> Result<(), IntegrationError> {
        sqlx::query("DELETE FROM connectors WHERE id = $1 AND user_id = $2")
            .bind(connector_id)
            .bind(user_id)
            .execute(&self.pool)
            .await
            .map_err(|e| IntegrationError::Database(e.to_string()))?;

        Ok(())
    }
}

// ============================================================================
// Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookInput {
    pub name: String,
    pub url: String,
    pub events: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Webhook {
    pub id: Uuid,
    pub user_id: Uuid,
    pub name: String,
    pub url: String,
    pub secret: String,
    pub events: Vec<String>,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
    pub last_triggered_at: Option<DateTime<Utc>>,
    pub failure_count: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WebhookEvent {
    EmailReceived,
    EmailSent,
    EmailReplied,
    EmailForwarded,
    EmailArchived,
    EmailDeleted,
    ContactCreated,
    ContactUpdated,
    CalendarEventCreated,
    AssignmentCreated,
    SlaBreached,
}

impl ToString for WebhookEvent {
    fn to_string(&self) -> String {
        match self {
            WebhookEvent::EmailReceived => "email.received".to_string(),
            WebhookEvent::EmailSent => "email.sent".to_string(),
            WebhookEvent::EmailReplied => "email.replied".to_string(),
            WebhookEvent::EmailForwarded => "email.forwarded".to_string(),
            WebhookEvent::EmailArchived => "email.archived".to_string(),
            WebhookEvent::EmailDeleted => "email.deleted".to_string(),
            WebhookEvent::ContactCreated => "contact.created".to_string(),
            WebhookEvent::ContactUpdated => "contact.updated".to_string(),
            WebhookEvent::CalendarEventCreated => "calendar.event_created".to_string(),
            WebhookEvent::AssignmentCreated => "assignment.created".to_string(),
            WebhookEvent::SlaBreached => "sla.breached".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookDelivery {
    pub id: Uuid,
    pub webhook_id: Uuid,
    pub event: WebhookEvent,
    pub status_code: i32,
    pub success: bool,
    pub delivered_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct ApiKey {
    pub id: Uuid,
    pub user_id: Uuid,
    pub name: String,
    pub key_hash: String,
    pub key_prefix: String,
    pub scopes: Vec<String>,
    pub expires_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub last_used_at: Option<DateTime<Utc>>,
    pub is_revoked: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyCreated {
    pub id: Uuid,
    pub key: String,
    pub name: String,
    pub scopes: Vec<ApiScope>,
    pub expires_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyValidation {
    pub valid: bool,
    pub user_id: Option<Uuid>,
    pub scopes: Vec<ApiScope>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ApiScope {
    EmailRead,
    EmailWrite,
    EmailSend,
    ContactsRead,
    ContactsWrite,
    CalendarRead,
    CalendarWrite,
    SettingsRead,
    SettingsWrite,
    WebhooksManage,
    Full,
}

impl ToString for ApiScope {
    fn to_string(&self) -> String {
        match self {
            ApiScope::EmailRead => "email:read".to_string(),
            ApiScope::EmailWrite => "email:write".to_string(),
            ApiScope::EmailSend => "email:send".to_string(),
            ApiScope::ContactsRead => "contacts:read".to_string(),
            ApiScope::ContactsWrite => "contacts:write".to_string(),
            ApiScope::CalendarRead => "calendar:read".to_string(),
            ApiScope::CalendarWrite => "calendar:write".to_string(),
            ApiScope::SettingsRead => "settings:read".to_string(),
            ApiScope::SettingsWrite => "settings:write".to_string(),
            ApiScope::WebhooksManage => "webhooks:manage".to_string(),
            ApiScope::Full => "full".to_string(),
        }
    }
}

impl std::str::FromStr for ApiScope {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "email:read" => Ok(ApiScope::EmailRead),
            "email:write" => Ok(ApiScope::EmailWrite),
            "email:send" => Ok(ApiScope::EmailSend),
            "contacts:read" => Ok(ApiScope::ContactsRead),
            "contacts:write" => Ok(ApiScope::ContactsWrite),
            "calendar:read" => Ok(ApiScope::CalendarRead),
            "calendar:write" => Ok(ApiScope::CalendarWrite),
            "settings:read" => Ok(ApiScope::SettingsRead),
            "settings:write" => Ok(ApiScope::SettingsWrite),
            "webhooks:manage" => Ok(ApiScope::WebhooksManage),
            "full" => Ok(ApiScope::Full),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorInput {
    pub connector_type: ConnectorType,
    pub name: String,
    pub config: JsonValue,
    pub triggers: Vec<ConnectorTrigger>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Connector {
    pub id: Uuid,
    pub user_id: Uuid,
    pub connector_type: ConnectorType,
    pub name: String,
    pub config: JsonValue,
    pub triggers: Vec<ConnectorTrigger>,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "text")]
pub enum ConnectorType {
    Slack,
    Discord,
    Teams,
    Email,
    Custom,
}

impl ToString for ConnectorType {
    fn to_string(&self) -> String {
        match self {
            ConnectorType::Slack => "slack".to_string(),
            ConnectorType::Discord => "discord".to_string(),
            ConnectorType::Teams => "teams".to_string(),
            ConnectorType::Email => "email".to_string(),
            ConnectorType::Custom => "custom".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ConnectorTrigger {
    NewEmail,
    UrgentEmail,
    VipEmail,
    AssignedEmail,
    SlaWarning,
    DailyDigest,
}

impl ToString for ConnectorTrigger {
    fn to_string(&self) -> String {
        match self {
            ConnectorTrigger::NewEmail => "new_email".to_string(),
            ConnectorTrigger::UrgentEmail => "urgent_email".to_string(),
            ConnectorTrigger::VipEmail => "vip_email".to_string(),
            ConnectorTrigger::AssignedEmail => "assigned_email".to_string(),
            ConnectorTrigger::SlaWarning => "sla_warning".to_string(),
            ConnectorTrigger::DailyDigest => "daily_digest".to_string(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum IntegrationError {
    #[error("Database error: {0}")]
    Database(String),
    #[error("Configuration error: {0}")]
    Configuration(String),
    #[error("External API error: {0}")]
    ExternalApi(String),
    #[error("Invalid API key")]
    InvalidApiKey,
}
