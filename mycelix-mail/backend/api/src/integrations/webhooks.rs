// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Webhook System for Mycelix Mail
//!
//! Allows external integrations to receive real-time notifications
//! about email events via HTTP callbacks.

use std::collections::HashMap;
use std::sync::Arc;
use chrono::{DateTime, Utc};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Webhook event types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum WebhookEventType {
    // Email events
    EmailReceived,
    EmailSent,
    EmailOpened,
    EmailClicked,
    EmailBounced,
    EmailComplaint,
    EmailUnsubscribed,

    // Thread events
    ThreadCreated,
    ThreadUpdated,
    ThreadResolved,

    // Trust events
    TrustScoreChanged,
    ContactVerified,

    // Account events
    UserCreated,
    UserUpdated,
    UserDeleted,

    // Team events
    SharedInboxMessage,
    AssignmentChanged,
    CommentAdded,

    // Integration events
    SyncCompleted,
    RuleTriggered,
}

/// Webhook subscription
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookSubscription {
    pub id: Uuid,
    pub user_id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub url: String,
    pub secret: String, // For HMAC signing
    pub events: Vec<WebhookEventType>,
    pub enabled: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,

    // Filtering
    pub filters: Option<WebhookFilters>,

    // Retry configuration
    pub max_retries: u32,
    pub retry_delay_secs: u32,

    // Stats
    pub total_deliveries: u64,
    pub successful_deliveries: u64,
    pub failed_deliveries: u64,
    pub last_delivery_at: Option<DateTime<Utc>>,
    pub last_error: Option<String>,
}

/// Filters to narrow down which events trigger the webhook
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WebhookFilters {
    pub sender_patterns: Option<Vec<String>>,
    pub recipient_patterns: Option<Vec<String>>,
    pub subject_patterns: Option<Vec<String>>,
    pub folders: Option<Vec<String>>,
    pub labels: Option<Vec<String>>,
    pub min_trust_score: Option<f64>,
}

/// A webhook delivery attempt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookDelivery {
    pub id: Uuid,
    pub subscription_id: Uuid,
    pub event_type: WebhookEventType,
    pub payload: serde_json::Value,
    pub status: DeliveryStatus,
    pub attempt_count: u32,
    pub created_at: DateTime<Utc>,
    pub delivered_at: Option<DateTime<Utc>>,
    pub response_status: Option<u16>,
    pub response_body: Option<String>,
    pub error: Option<String>,
    pub next_retry_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DeliveryStatus {
    Pending,
    Delivering,
    Delivered,
    Failed,
    Exhausted, // All retries failed
}

/// Webhook payload structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookPayload {
    pub id: String,
    pub event: String,
    pub timestamp: DateTime<Utc>,
    pub data: serde_json::Value,
    pub metadata: WebhookMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookMetadata {
    pub subscription_id: String,
    pub attempt_number: u32,
    pub user_id: String,
}

/// Webhook service
pub struct WebhookService {
    subscriptions: Arc<RwLock<HashMap<Uuid, WebhookSubscription>>>,
    deliveries: Arc<RwLock<Vec<WebhookDelivery>>>,
    http_client: reqwest::Client,
}

impl WebhookService {
    pub fn new() -> Self {
        Self {
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
            deliveries: Arc::new(RwLock::new(Vec::new())),
            http_client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap(),
        }
    }

    /// Create a new webhook subscription
    pub async fn create_subscription(
        &self,
        user_id: Uuid,
        name: String,
        url: String,
        events: Vec<WebhookEventType>,
    ) -> Result<WebhookSubscription, WebhookError> {
        // Validate URL
        if !url.starts_with("https://") && !url.starts_with("http://localhost") {
            return Err(WebhookError::InvalidUrl("HTTPS required".to_string()));
        }

        // Generate secret for signing
        let secret = Self::generate_secret();

        let subscription = WebhookSubscription {
            id: Uuid::new_v4(),
            user_id,
            name,
            description: None,
            url,
            secret,
            events,
            enabled: true,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            filters: None,
            max_retries: 3,
            retry_delay_secs: 60,
            total_deliveries: 0,
            successful_deliveries: 0,
            failed_deliveries: 0,
            last_delivery_at: None,
            last_error: None,
        };

        self.subscriptions.write().await
            .insert(subscription.id, subscription.clone());

        Ok(subscription)
    }

    /// Update subscription
    pub async fn update_subscription(
        &self,
        subscription_id: Uuid,
        user_id: Uuid,
        updates: SubscriptionUpdate,
    ) -> Result<WebhookSubscription, WebhookError> {
        let mut subs = self.subscriptions.write().await;

        let sub = subs.get_mut(&subscription_id)
            .ok_or(WebhookError::NotFound(subscription_id))?;

        if sub.user_id != user_id {
            return Err(WebhookError::Unauthorized);
        }

        if let Some(name) = updates.name {
            sub.name = name;
        }
        if let Some(url) = updates.url {
            if !url.starts_with("https://") && !url.starts_with("http://localhost") {
                return Err(WebhookError::InvalidUrl("HTTPS required".to_string()));
            }
            sub.url = url;
        }
        if let Some(events) = updates.events {
            sub.events = events;
        }
        if let Some(enabled) = updates.enabled {
            sub.enabled = enabled;
        }
        if let Some(filters) = updates.filters {
            sub.filters = Some(filters);
        }

        sub.updated_at = Utc::now();

        Ok(sub.clone())
    }

    /// Delete subscription
    pub async fn delete_subscription(
        &self,
        subscription_id: Uuid,
        user_id: Uuid,
    ) -> Result<(), WebhookError> {
        let mut subs = self.subscriptions.write().await;

        let sub = subs.get(&subscription_id)
            .ok_or(WebhookError::NotFound(subscription_id))?;

        if sub.user_id != user_id {
            return Err(WebhookError::Unauthorized);
        }

        subs.remove(&subscription_id);
        Ok(())
    }

    /// Rotate webhook secret
    pub async fn rotate_secret(
        &self,
        subscription_id: Uuid,
        user_id: Uuid,
    ) -> Result<String, WebhookError> {
        let mut subs = self.subscriptions.write().await;

        let sub = subs.get_mut(&subscription_id)
            .ok_or(WebhookError::NotFound(subscription_id))?;

        if sub.user_id != user_id {
            return Err(WebhookError::Unauthorized);
        }

        let new_secret = Self::generate_secret();
        sub.secret = new_secret.clone();
        sub.updated_at = Utc::now();

        Ok(new_secret)
    }

    /// Trigger webhooks for an event
    pub async fn trigger(
        &self,
        event_type: WebhookEventType,
        user_id: Uuid,
        data: serde_json::Value,
    ) {
        let subs = self.subscriptions.read().await;

        let matching: Vec<_> = subs.values()
            .filter(|s| {
                s.enabled &&
                s.user_id == user_id &&
                s.events.contains(&event_type)
            })
            .cloned()
            .collect();

        drop(subs);

        for subscription in matching {
            // Check filters
            if !self.matches_filters(&subscription.filters, &data) {
                continue;
            }

            // Queue delivery
            self.queue_delivery(subscription.id, event_type.clone(), data.clone()).await;
        }
    }

    /// Check if event data matches subscription filters
    fn matches_filters(&self, filters: &Option<WebhookFilters>, data: &serde_json::Value) -> bool {
        let filters = match filters {
            Some(f) => f,
            None => return true,
        };

        // Check sender pattern
        if let Some(ref patterns) = filters.sender_patterns {
            if let Some(sender) = data.get("from").and_then(|v| v.as_str()) {
                let matches = patterns.iter().any(|p| sender.contains(p));
                if !matches {
                    return false;
                }
            }
        }

        // Check folder
        if let Some(ref folders) = filters.folders {
            if let Some(folder) = data.get("folder").and_then(|v| v.as_str()) {
                if !folders.iter().any(|f| f == folder) {
                    return false;
                }
            }
        }

        // Check trust score
        if let Some(min_score) = filters.min_trust_score {
            if let Some(score) = data.get("trust_score").and_then(|v| v.as_f64()) {
                if score < min_score {
                    return false;
                }
            }
        }

        true
    }

    /// Queue a delivery
    async fn queue_delivery(
        &self,
        subscription_id: Uuid,
        event_type: WebhookEventType,
        data: serde_json::Value,
    ) {
        let delivery = WebhookDelivery {
            id: Uuid::new_v4(),
            subscription_id,
            event_type,
            payload: data,
            status: DeliveryStatus::Pending,
            attempt_count: 0,
            created_at: Utc::now(),
            delivered_at: None,
            response_status: None,
            response_body: None,
            error: None,
            next_retry_at: None,
        };

        self.deliveries.write().await.push(delivery);
    }

    /// Process pending deliveries
    pub async fn process_deliveries(&self) {
        let now = Utc::now();
        let mut deliveries = self.deliveries.write().await;

        for delivery in deliveries.iter_mut() {
            if delivery.status != DeliveryStatus::Pending {
                continue;
            }

            if let Some(retry_at) = delivery.next_retry_at {
                if retry_at > now {
                    continue;
                }
            }

            // Get subscription
            let subs = self.subscriptions.read().await;
            let subscription = match subs.get(&delivery.subscription_id) {
                Some(s) => s.clone(),
                None => continue,
            };
            drop(subs);

            // Attempt delivery
            delivery.status = DeliveryStatus::Delivering;
            delivery.attempt_count += 1;

            let result = self.deliver(&subscription, delivery).await;

            // Update stats
            let mut subs = self.subscriptions.write().await;
            if let Some(sub) = subs.get_mut(&subscription.id) {
                sub.total_deliveries += 1;
                sub.last_delivery_at = Some(Utc::now());

                match result {
                    Ok(_) => {
                        delivery.status = DeliveryStatus::Delivered;
                        delivery.delivered_at = Some(Utc::now());
                        sub.successful_deliveries += 1;
                    }
                    Err(e) => {
                        let error = e.to_string();
                        delivery.error = Some(error.clone());
                        sub.last_error = Some(error);

                        if delivery.attempt_count >= sub.max_retries {
                            delivery.status = DeliveryStatus::Exhausted;
                            sub.failed_deliveries += 1;
                        } else {
                            delivery.status = DeliveryStatus::Pending;
                            let delay = sub.retry_delay_secs * (2u32.pow(delivery.attempt_count - 1));
                            delivery.next_retry_at = Some(now + chrono::Duration::seconds(delay as i64));
                        }
                    }
                }
            }
        }
    }

    /// Deliver a webhook
    async fn deliver(
        &self,
        subscription: &WebhookSubscription,
        delivery: &mut WebhookDelivery,
    ) -> Result<(), WebhookError> {
        let payload = WebhookPayload {
            id: delivery.id.to_string(),
            event: format!("{:?}", delivery.event_type),
            timestamp: Utc::now(),
            data: delivery.payload.clone(),
            metadata: WebhookMetadata {
                subscription_id: subscription.id.to_string(),
                attempt_number: delivery.attempt_count,
                user_id: subscription.user_id.to_string(),
            },
        };

        let body = serde_json::to_string(&payload)
            .map_err(|e| WebhookError::SerializationError(e.to_string()))?;

        // Calculate HMAC signature
        let signature = self.calculate_signature(&subscription.secret, &body);

        let response = self.http_client
            .post(&subscription.url)
            .header("Content-Type", "application/json")
            .header("X-Webhook-Signature", format!("sha256={}", signature))
            .header("X-Webhook-Event", format!("{:?}", delivery.event_type))
            .header("X-Webhook-Delivery-Id", delivery.id.to_string())
            .header("User-Agent", "Mycelix-Mail-Webhooks/1.0")
            .body(body)
            .send()
            .await
            .map_err(|e| WebhookError::DeliveryFailed(e.to_string()))?;

        let status = response.status().as_u16();
        delivery.response_status = Some(status);

        let response_body = response.text().await.unwrap_or_default();
        delivery.response_body = Some(response_body.chars().take(1000).collect());

        if status >= 200 && status < 300 {
            Ok(())
        } else {
            Err(WebhookError::DeliveryFailed(format!("HTTP {}", status)))
        }
    }

    /// Calculate HMAC-SHA256 signature
    fn calculate_signature(&self, secret: &str, payload: &str) -> String {
        type HmacSha256 = Hmac<Sha256>;

        let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(payload.as_bytes());

        hex::encode(mac.finalize().into_bytes())
    }

    /// Generate a random secret
    fn generate_secret() -> String {
        use rand::Rng;
        let bytes: [u8; 32] = rand::thread_rng().gen();
        hex::encode(bytes)
    }

    /// Get delivery history for a subscription
    pub async fn get_delivery_history(
        &self,
        subscription_id: Uuid,
        limit: usize,
    ) -> Vec<WebhookDelivery> {
        let deliveries = self.deliveries.read().await;

        deliveries.iter()
            .filter(|d| d.subscription_id == subscription_id)
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    /// Test webhook endpoint
    pub async fn test_webhook(
        &self,
        subscription_id: Uuid,
        user_id: Uuid,
    ) -> Result<TestResult, WebhookError> {
        let subs = self.subscriptions.read().await;
        let subscription = subs.get(&subscription_id)
            .ok_or(WebhookError::NotFound(subscription_id))?;

        if subscription.user_id != user_id {
            return Err(WebhookError::Unauthorized);
        }

        let subscription = subscription.clone();
        drop(subs);

        let test_payload = serde_json::json!({
            "test": true,
            "message": "This is a test webhook from Mycelix Mail"
        });

        let mut delivery = WebhookDelivery {
            id: Uuid::new_v4(),
            subscription_id,
            event_type: WebhookEventType::EmailReceived,
            payload: test_payload,
            status: DeliveryStatus::Delivering,
            attempt_count: 1,
            created_at: Utc::now(),
            delivered_at: None,
            response_status: None,
            response_body: None,
            error: None,
            next_retry_at: None,
        };

        let result = self.deliver(&subscription, &mut delivery).await;

        Ok(TestResult {
            success: result.is_ok(),
            response_status: delivery.response_status,
            response_body: delivery.response_body,
            error: delivery.error,
            duration_ms: 0, // Would need to measure
        })
    }
}

impl Default for WebhookService {
    fn default() -> Self {
        Self::new()
    }
}

/// Subscription update request
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SubscriptionUpdate {
    pub name: Option<String>,
    pub url: Option<String>,
    pub events: Option<Vec<WebhookEventType>>,
    pub enabled: Option<bool>,
    pub filters: Option<WebhookFilters>,
}

/// Test webhook result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub success: bool,
    pub response_status: Option<u16>,
    pub response_body: Option<String>,
    pub error: Option<String>,
    pub duration_ms: u64,
}

/// Webhook errors
#[derive(Debug, Clone)]
pub enum WebhookError {
    NotFound(Uuid),
    Unauthorized,
    InvalidUrl(String),
    DeliveryFailed(String),
    SerializationError(String),
}

impl std::fmt::Display for WebhookError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "Webhook not found: {}", id),
            Self::Unauthorized => write!(f, "Unauthorized"),
            Self::InvalidUrl(msg) => write!(f, "Invalid URL: {}", msg),
            Self::DeliveryFailed(e) => write!(f, "Delivery failed: {}", e),
            Self::SerializationError(e) => write!(f, "Serialization error: {}", e),
        }
    }
}

impl std::error::Error for WebhookError {}
