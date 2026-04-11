// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Developer Platform & Extensibility Module
//!
//! Provides plugin API, webhooks, custom actions, email scripting,
//! marketplace, and OAuth app authorization.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use sqlx::PgPool;
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum PlatformError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("Plugin not found: {0}")]
    PluginNotFound(Uuid),
    #[error("Webhook delivery failed: {0}")]
    WebhookFailed(String),
    #[error("Script execution error: {0}")]
    ScriptError(String),
    #[error("Invalid manifest: {0}")]
    InvalidManifest(String),
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    #[error("Rate limit exceeded")]
    RateLimitExceeded,
}

// ============================================================================
// Plugin API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plugin {
    pub id: Uuid,
    pub name: String,
    pub slug: String,
    pub version: String,
    pub description: String,
    pub author: PluginAuthor,
    pub icon_url: Option<String>,
    pub homepage_url: Option<String>,
    pub manifest: PluginManifest,
    pub permissions: Vec<PluginPermission>,
    pub status: PluginStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginAuthor {
    pub name: String,
    pub email: Option<String>,
    pub url: Option<String>,
    pub verified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    pub entry_point: String,
    pub api_version: String,
    pub min_app_version: Option<String>,
    pub hooks: Vec<PluginHook>,
    pub actions: Vec<PluginAction>,
    pub settings_schema: Option<JsonValue>,
    pub sidebar_panels: Vec<SidebarPanel>,
    pub compose_extensions: Vec<ComposeExtension>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginHook {
    pub event: HookEvent,
    pub handler: String,
    pub priority: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HookEvent {
    BeforeSend,
    AfterSend,
    OnReceive,
    OnOpen,
    OnArchive,
    OnDelete,
    OnLabel,
    OnCompose,
    OnReply,
    OnForward,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginAction {
    pub id: String,
    pub label: String,
    pub icon: Option<String>,
    pub handler: String,
    pub location: ActionLocation,
    pub shortcut: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionLocation {
    Toolbar,
    ContextMenu,
    ComposeToolbar,
    QuickActions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidebarPanel {
    pub id: String,
    pub title: String,
    pub icon: Option<String>,
    pub component: String,
    pub default_width: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposeExtension {
    pub id: String,
    pub label: String,
    pub position: ComposePosition,
    pub component: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComposePosition {
    BeforeRecipients,
    AfterRecipients,
    BeforeBody,
    AfterBody,
    Footer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginPermission {
    ReadEmails,
    WriteEmails,
    SendEmails,
    ReadContacts,
    WriteContacts,
    ReadCalendar,
    WriteCalendar,
    ReadAttachments,
    WriteAttachments,
    AccessNetwork,
    StorageLocal,
    StorageSync,
    Notifications,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PluginStatus {
    Active,
    Disabled,
    Suspended,
    PendingReview,
}

pub struct PluginService {
    pool: PgPool,
}

impl PluginService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn register_plugin(
        &self,
        manifest_json: &str,
    ) -> Result<Plugin, PlatformError> {
        let manifest: PluginManifest = serde_json::from_str(manifest_json)
            .map_err(|e| PlatformError::InvalidManifest(e.to_string()))?;

        let plugin_id = Uuid::new_v4();

        // Validate manifest
        self.validate_manifest(&manifest)?;

        sqlx::query!(
            r#"
            INSERT INTO plugins (id, name, slug, version, description, author,
                               manifest, permissions, status, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW(), NOW())
            "#,
            plugin_id,
            "New Plugin", // Would come from manifest
            "new-plugin",
            "1.0.0",
            "Plugin description",
            serde_json::to_value(&PluginAuthor {
                name: "Unknown".to_string(),
                email: None,
                url: None,
                verified: false,
            }).unwrap(),
            serde_json::to_value(&manifest).unwrap(),
            &[] as &[String],
            serde_json::to_string(&PluginStatus::PendingReview).unwrap()
        )
        .execute(&self.pool)
        .await?;

        self.get_plugin(plugin_id).await
    }

    fn validate_manifest(&self, manifest: &PluginManifest) -> Result<(), PlatformError> {
        if manifest.entry_point.is_empty() {
            return Err(PlatformError::InvalidManifest(
                "Entry point is required".to_string(),
            ));
        }
        if manifest.api_version.is_empty() {
            return Err(PlatformError::InvalidManifest(
                "API version is required".to_string(),
            ));
        }
        Ok(())
    }

    pub async fn get_plugin(&self, plugin_id: Uuid) -> Result<Plugin, PlatformError> {
        let row = sqlx::query!(
            r#"
            SELECT id, name, slug, version, description, author, icon_url,
                   homepage_url, manifest, permissions, status, created_at, updated_at
            FROM plugins
            WHERE id = $1
            "#,
            plugin_id
        )
        .fetch_optional(&self.pool)
        .await?
        .ok_or(PlatformError::PluginNotFound(plugin_id))?;

        Ok(Plugin {
            id: row.id,
            name: row.name,
            slug: row.slug,
            version: row.version,
            description: row.description,
            author: serde_json::from_value(row.author).unwrap_or(PluginAuthor {
                name: "Unknown".to_string(),
                email: None,
                url: None,
                verified: false,
            }),
            icon_url: row.icon_url,
            homepage_url: row.homepage_url,
            manifest: serde_json::from_value(row.manifest).unwrap(),
            permissions: row.permissions.iter()
                .filter_map(|p| serde_json::from_str(p).ok())
                .collect(),
            status: serde_json::from_str(&row.status).unwrap_or(PluginStatus::Disabled),
            created_at: row.created_at,
            updated_at: row.updated_at,
        })
    }

    pub async fn install_plugin(
        &self,
        user_id: Uuid,
        plugin_id: Uuid,
    ) -> Result<PluginInstallation, PlatformError> {
        let plugin = self.get_plugin(plugin_id).await?;

        if plugin.status != PluginStatus::Active {
            return Err(PlatformError::PermissionDenied(
                "Plugin is not available for installation".to_string(),
            ));
        }

        sqlx::query!(
            r#"
            INSERT INTO user_plugins (user_id, plugin_id, installed_at, enabled, settings)
            VALUES ($1, $2, NOW(), true, '{}')
            ON CONFLICT (user_id, plugin_id) DO UPDATE SET
                enabled = true
            "#,
            user_id,
            plugin_id
        )
        .execute(&self.pool)
        .await?;

        Ok(PluginInstallation {
            plugin_id,
            plugin_name: plugin.name,
            installed_at: Utc::now(),
            enabled: true,
            settings: JsonValue::Object(serde_json::Map::new()),
        })
    }

    pub async fn uninstall_plugin(
        &self,
        user_id: Uuid,
        plugin_id: Uuid,
    ) -> Result<(), PlatformError> {
        sqlx::query!(
            r#"
            DELETE FROM user_plugins
            WHERE user_id = $1 AND plugin_id = $2
            "#,
            user_id,
            plugin_id
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_user_plugins(&self, user_id: Uuid) -> Result<Vec<PluginInstallation>, PlatformError> {
        let installations = sqlx::query!(
            r#"
            SELECT up.plugin_id, p.name as plugin_name, up.installed_at,
                   up.enabled, up.settings
            FROM user_plugins up
            JOIN plugins p ON up.plugin_id = p.id
            WHERE up.user_id = $1
            ORDER BY up.installed_at DESC
            "#,
            user_id
        )
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|row| PluginInstallation {
            plugin_id: row.plugin_id,
            plugin_name: row.plugin_name,
            installed_at: row.installed_at,
            enabled: row.enabled,
            settings: row.settings,
        })
        .collect();

        Ok(installations)
    }

    pub async fn update_plugin_settings(
        &self,
        user_id: Uuid,
        plugin_id: Uuid,
        settings: JsonValue,
    ) -> Result<(), PlatformError> {
        sqlx::query!(
            r#"
            UPDATE user_plugins
            SET settings = $1
            WHERE user_id = $2 AND plugin_id = $3
            "#,
            settings,
            user_id,
            plugin_id
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn execute_hook(
        &self,
        user_id: Uuid,
        event: HookEvent,
        context: JsonValue,
    ) -> Result<Vec<HookResult>, PlatformError> {
        let plugins = sqlx::query!(
            r#"
            SELECT p.id, p.manifest, up.settings
            FROM user_plugins up
            JOIN plugins p ON up.plugin_id = p.id
            WHERE up.user_id = $1 AND up.enabled = true AND p.status = 'Active'
            "#,
            user_id
        )
        .fetch_all(&self.pool)
        .await?;

        let mut results = Vec::new();

        for plugin in plugins {
            let manifest: PluginManifest = serde_json::from_value(plugin.manifest).unwrap();

            for hook in &manifest.hooks {
                if std::mem::discriminant(&hook.event) == std::mem::discriminant(&event) {
                    // Execute hook (in production, this would run sandboxed JS)
                    results.push(HookResult {
                        plugin_id: plugin.id,
                        success: true,
                        output: None,
                        error: None,
                    });
                }
            }
        }

        Ok(results)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginInstallation {
    pub plugin_id: Uuid,
    pub plugin_name: String,
    pub installed_at: DateTime<Utc>,
    pub enabled: bool,
    pub settings: JsonValue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookResult {
    pub plugin_id: Uuid,
    pub success: bool,
    pub output: Option<JsonValue>,
    pub error: Option<String>,
}

// ============================================================================
// Webhooks
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Webhook {
    pub id: Uuid,
    pub user_id: Uuid,
    pub name: String,
    pub url: String,
    pub secret: String,
    pub events: Vec<WebhookEvent>,
    pub enabled: bool,
    pub created_at: DateTime<Utc>,
    pub last_triggered: Option<DateTime<Utc>>,
    pub failure_count: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WebhookEvent {
    EmailReceived,
    EmailSent,
    EmailOpened,
    EmailArchived,
    EmailDeleted,
    ContactCreated,
    ContactUpdated,
    LabelCreated,
    AttachmentUploaded,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookDelivery {
    pub id: Uuid,
    pub webhook_id: Uuid,
    pub event: WebhookEvent,
    pub payload: JsonValue,
    pub response_status: Option<i32>,
    pub response_body: Option<String>,
    pub delivered_at: DateTime<Utc>,
    pub success: bool,
    pub retry_count: i32,
}

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

    pub async fn create_webhook(
        &self,
        user_id: Uuid,
        name: &str,
        url: &str,
        events: Vec<WebhookEvent>,
    ) -> Result<Webhook, PlatformError> {
        let secret = generate_webhook_secret();

        let webhook = sqlx::query_as!(
            Webhook,
            r#"
            INSERT INTO webhooks (user_id, name, url, secret, events, enabled, created_at)
            VALUES ($1, $2, $3, $4, $5, true, NOW())
            RETURNING id, user_id, name, url, secret, events as "events: Vec<String>",
                      enabled, created_at, last_triggered, failure_count
            "#,
            user_id,
            name,
            url,
            secret,
            &events.iter().map(|e| serde_json::to_string(e).unwrap()).collect::<Vec<_>>()
        )
        .fetch_one(&self.pool)
        .await
        .map(|row| Webhook {
            id: row.id,
            user_id: row.user_id,
            name: row.name,
            url: row.url,
            secret: row.secret,
            events: row.events.iter()
                .filter_map(|e| serde_json::from_str(e).ok())
                .collect(),
            enabled: row.enabled,
            created_at: row.created_at,
            last_triggered: row.last_triggered,
            failure_count: row.failure_count,
        })?;

        Ok(webhook)
    }

    pub async fn get_webhooks(&self, user_id: Uuid) -> Result<Vec<Webhook>, PlatformError> {
        let webhooks = sqlx::query!(
            r#"
            SELECT id, user_id, name, url, secret, events, enabled,
                   created_at, last_triggered, failure_count
            FROM webhooks
            WHERE user_id = $1
            ORDER BY created_at DESC
            "#,
            user_id
        )
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|row| Webhook {
            id: row.id,
            user_id: row.user_id,
            name: row.name,
            url: row.url,
            secret: row.secret,
            events: row.events.iter()
                .filter_map(|e| serde_json::from_str(e).ok())
                .collect(),
            enabled: row.enabled,
            created_at: row.created_at,
            last_triggered: row.last_triggered,
            failure_count: row.failure_count,
        })
        .collect();

        Ok(webhooks)
    }

    pub async fn delete_webhook(&self, user_id: Uuid, webhook_id: Uuid) -> Result<(), PlatformError> {
        sqlx::query!(
            "DELETE FROM webhooks WHERE id = $1 AND user_id = $2",
            webhook_id,
            user_id
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn trigger_webhooks(
        &self,
        user_id: Uuid,
        event: WebhookEvent,
        payload: JsonValue,
    ) -> Result<Vec<WebhookDelivery>, PlatformError> {
        let webhooks = sqlx::query!(
            r#"
            SELECT id, url, secret
            FROM webhooks
            WHERE user_id = $1 AND enabled = true AND failure_count < 10
            "#,
            user_id
        )
        .fetch_all(&self.pool)
        .await?;

        let mut deliveries = Vec::new();

        for webhook in webhooks {
            // Check if webhook subscribes to this event
            let subscribed = sqlx::query_scalar!(
                "SELECT $1 = ANY(events) FROM webhooks WHERE id = $2",
                serde_json::to_string(&event).unwrap(),
                webhook.id
            )
            .fetch_one(&self.pool)
            .await?
            .unwrap_or(false);

            if !subscribed {
                continue;
            }

            let delivery = self
                .deliver_webhook(webhook.id, &webhook.url, &webhook.secret, &event, &payload)
                .await;

            deliveries.push(delivery);
        }

        Ok(deliveries)
    }

    async fn deliver_webhook(
        &self,
        webhook_id: Uuid,
        url: &str,
        secret: &str,
        event: &WebhookEvent,
        payload: &JsonValue,
    ) -> WebhookDelivery {
        let signature = compute_signature(secret, payload);

        let result = self
            .http_client
            .post(url)
            .header("Content-Type", "application/json")
            .header("X-Webhook-Signature", &signature)
            .header("X-Webhook-Event", serde_json::to_string(event).unwrap())
            .json(payload)
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .await;

        let (success, status, body) = match result {
            Ok(response) => {
                let status = response.status().as_u16() as i32;
                let body = response.text().await.ok();
                (status >= 200 && status < 300, Some(status), body)
            }
            Err(e) => (false, None, Some(e.to_string())),
        };

        // Update webhook stats
        if success {
            let _ = sqlx::query!(
                r#"
                UPDATE webhooks
                SET last_triggered = NOW(), failure_count = 0
                WHERE id = $1
                "#,
                webhook_id
            )
            .execute(&self.pool)
            .await;
        } else {
            let _ = sqlx::query!(
                r#"
                UPDATE webhooks
                SET failure_count = failure_count + 1
                WHERE id = $1
                "#,
                webhook_id
            )
            .execute(&self.pool)
            .await;
        }

        // Log delivery
        let delivery_id = Uuid::new_v4();
        let _ = sqlx::query!(
            r#"
            INSERT INTO webhook_deliveries (id, webhook_id, event, payload,
                                           response_status, response_body, success, delivered_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
            "#,
            delivery_id,
            webhook_id,
            serde_json::to_string(event).unwrap(),
            payload,
            status,
            body.as_deref(),
            success
        )
        .execute(&self.pool)
        .await;

        WebhookDelivery {
            id: delivery_id,
            webhook_id,
            event: event.clone(),
            payload: payload.clone(),
            response_status: status,
            response_body: body,
            delivered_at: Utc::now(),
            success,
            retry_count: 0,
        }
    }

    pub async fn get_delivery_history(
        &self,
        webhook_id: Uuid,
        limit: i32,
    ) -> Result<Vec<WebhookDelivery>, PlatformError> {
        let deliveries = sqlx::query!(
            r#"
            SELECT id, webhook_id, event, payload, response_status,
                   response_body, success, delivered_at, retry_count
            FROM webhook_deliveries
            WHERE webhook_id = $1
            ORDER BY delivered_at DESC
            LIMIT $2
            "#,
            webhook_id,
            limit as i64
        )
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|row| WebhookDelivery {
            id: row.id,
            webhook_id: row.webhook_id,
            event: serde_json::from_str(&row.event).unwrap_or(WebhookEvent::EmailReceived),
            payload: row.payload,
            response_status: row.response_status,
            response_body: row.response_body,
            delivered_at: row.delivered_at,
            success: row.success,
            retry_count: row.retry_count,
        })
        .collect();

        Ok(deliveries)
    }
}

fn generate_webhook_secret() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let bytes: Vec<u8> = (0..32).map(|_| rng.gen()).collect();
    hex::encode(bytes)
}

fn compute_signature(secret: &str, payload: &JsonValue) -> String {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;

    let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes()).unwrap();
    mac.update(serde_json::to_string(payload).unwrap().as_bytes());
    let result = mac.finalize();
    format!("sha256={}", hex::encode(result.into_bytes()))
}

// ============================================================================
// Custom Actions
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomAction {
    pub id: Uuid,
    pub user_id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub icon: Option<String>,
    pub script: String,
    pub location: ActionLocation,
    pub shortcut: Option<String>,
    pub enabled: bool,
    pub created_at: DateTime<Utc>,
}

pub struct CustomActionService {
    pool: PgPool,
}

impl CustomActionService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn create_action(
        &self,
        user_id: Uuid,
        name: &str,
        script: &str,
        location: ActionLocation,
        description: Option<&str>,
        shortcut: Option<&str>,
    ) -> Result<CustomAction, PlatformError> {
        // Validate script (basic check)
        self.validate_script(script)?;

        let action = sqlx::query_as!(
            CustomAction,
            r#"
            INSERT INTO custom_actions (user_id, name, description, script, location,
                                       shortcut, enabled, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, true, NOW())
            RETURNING id, user_id, name, description, icon, script,
                      location as "location: String", shortcut, enabled, created_at
            "#,
            user_id,
            name,
            description,
            script,
            serde_json::to_string(&location).unwrap(),
            shortcut
        )
        .fetch_one(&self.pool)
        .await
        .map(|row| CustomAction {
            id: row.id,
            user_id: row.user_id,
            name: row.name,
            description: row.description,
            icon: row.icon,
            script: row.script,
            location: serde_json::from_str(&row.location).unwrap_or(ActionLocation::Toolbar),
            shortcut: row.shortcut,
            enabled: row.enabled,
            created_at: row.created_at,
        })?;

        Ok(action)
    }

    fn validate_script(&self, script: &str) -> Result<(), PlatformError> {
        // Check for dangerous patterns
        let forbidden = ["eval(", "Function(", "require(", "import(", "fetch("];
        for pattern in forbidden {
            if script.contains(pattern) {
                return Err(PlatformError::ScriptError(format!(
                    "Script contains forbidden pattern: {}",
                    pattern
                )));
            }
        }
        Ok(())
    }

    pub async fn get_actions(&self, user_id: Uuid) -> Result<Vec<CustomAction>, PlatformError> {
        let actions = sqlx::query!(
            r#"
            SELECT id, user_id, name, description, icon, script,
                   location, shortcut, enabled, created_at
            FROM custom_actions
            WHERE user_id = $1
            ORDER BY name
            "#,
            user_id
        )
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|row| CustomAction {
            id: row.id,
            user_id: row.user_id,
            name: row.name,
            description: row.description,
            icon: row.icon,
            script: row.script,
            location: serde_json::from_str(&row.location).unwrap_or(ActionLocation::Toolbar),
            shortcut: row.shortcut,
            enabled: row.enabled,
            created_at: row.created_at,
        })
        .collect();

        Ok(actions)
    }

    pub async fn execute_action(
        &self,
        action_id: Uuid,
        user_id: Uuid,
        context: JsonValue,
    ) -> Result<ActionResult, PlatformError> {
        let action = sqlx::query!(
            "SELECT script FROM custom_actions WHERE id = $1 AND user_id = $2 AND enabled = true",
            action_id,
            user_id
        )
        .fetch_optional(&self.pool)
        .await?
        .ok_or(PlatformError::PermissionDenied(
            "Action not found or disabled".to_string(),
        ))?;

        // In production, execute script in sandboxed environment
        // For now, return success placeholder
        Ok(ActionResult {
            success: true,
            output: Some(JsonValue::String("Action executed".to_string())),
            error: None,
            execution_time_ms: 10,
        })
    }

    pub async fn delete_action(&self, action_id: Uuid, user_id: Uuid) -> Result<(), PlatformError> {
        sqlx::query!(
            "DELETE FROM custom_actions WHERE id = $1 AND user_id = $2",
            action_id,
            user_id
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionResult {
    pub success: bool,
    pub output: Option<JsonValue>,
    pub error: Option<String>,
    pub execution_time_ms: i64,
}

// ============================================================================
// OAuth Apps
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthApp {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub client_id: String,
    pub redirect_uris: Vec<String>,
    pub scopes: Vec<String>,
    pub owner_id: Uuid,
    pub created_at: DateTime<Utc>,
    pub verified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthAuthorization {
    pub id: Uuid,
    pub user_id: Uuid,
    pub app_id: Uuid,
    pub app_name: String,
    pub scopes: Vec<String>,
    pub authorized_at: DateTime<Utc>,
    pub last_used: Option<DateTime<Utc>>,
}

pub struct OAuthService {
    pool: PgPool,
}

impl OAuthService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn create_app(
        &self,
        owner_id: Uuid,
        name: &str,
        redirect_uris: Vec<String>,
        scopes: Vec<String>,
        description: Option<&str>,
    ) -> Result<(OAuthApp, String), PlatformError> {
        let client_id = generate_client_id();
        let client_secret = generate_client_secret();

        let app = sqlx::query_as!(
            OAuthApp,
            r#"
            INSERT INTO oauth_apps (owner_id, name, description, client_id, client_secret_hash,
                                   redirect_uris, scopes, created_at, verified)
            VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), false)
            RETURNING id, name, description, client_id,
                      redirect_uris as "redirect_uris: Vec<String>",
                      scopes as "scopes: Vec<String>",
                      owner_id, created_at, verified
            "#,
            owner_id,
            name,
            description,
            client_id,
            hash_secret(&client_secret),
            &redirect_uris,
            &scopes
        )
        .fetch_one(&self.pool)
        .await?;

        Ok((app, client_secret))
    }

    pub async fn get_app(&self, client_id: &str) -> Result<OAuthApp, PlatformError> {
        sqlx::query_as!(
            OAuthApp,
            r#"
            SELECT id, name, description, client_id,
                   redirect_uris as "redirect_uris: Vec<String>",
                   scopes as "scopes: Vec<String>",
                   owner_id, created_at, verified
            FROM oauth_apps
            WHERE client_id = $1
            "#,
            client_id
        )
        .fetch_optional(&self.pool)
        .await?
        .ok_or(PlatformError::PluginNotFound(Uuid::nil()))
    }

    pub async fn authorize_app(
        &self,
        user_id: Uuid,
        app_id: Uuid,
        scopes: Vec<String>,
    ) -> Result<String, PlatformError> {
        let auth_code = generate_auth_code();

        sqlx::query!(
            r#"
            INSERT INTO oauth_authorizations (user_id, app_id, scopes, auth_code,
                                             auth_code_expires, authorized_at)
            VALUES ($1, $2, $3, $4, NOW() + INTERVAL '10 minutes', NOW())
            ON CONFLICT (user_id, app_id) DO UPDATE SET
                scopes = EXCLUDED.scopes,
                auth_code = EXCLUDED.auth_code,
                auth_code_expires = EXCLUDED.auth_code_expires
            "#,
            user_id,
            app_id,
            &scopes,
            hash_secret(&auth_code)
        )
        .execute(&self.pool)
        .await?;

        Ok(auth_code)
    }

    pub async fn exchange_code(
        &self,
        client_id: &str,
        client_secret: &str,
        code: &str,
    ) -> Result<(String, String), PlatformError> {
        // Verify client credentials
        let app = sqlx::query!(
            "SELECT id, client_secret_hash FROM oauth_apps WHERE client_id = $1",
            client_id
        )
        .fetch_optional(&self.pool)
        .await?
        .ok_or(PlatformError::PermissionDenied("Invalid client".to_string()))?;

        if !verify_secret(client_secret, &app.client_secret_hash) {
            return Err(PlatformError::PermissionDenied(
                "Invalid client secret".to_string(),
            ));
        }

        // Find and verify auth code
        let auth = sqlx::query!(
            r#"
            SELECT user_id, scopes
            FROM oauth_authorizations
            WHERE app_id = $1 AND auth_code = $2 AND auth_code_expires > NOW()
            "#,
            app.id,
            hash_secret(code)
        )
        .fetch_optional(&self.pool)
        .await?
        .ok_or(PlatformError::PermissionDenied(
            "Invalid or expired code".to_string(),
        ))?;

        // Generate tokens
        let access_token = generate_access_token();
        let refresh_token = generate_refresh_token();

        sqlx::query!(
            r#"
            UPDATE oauth_authorizations
            SET access_token_hash = $1,
                refresh_token_hash = $2,
                auth_code = NULL,
                last_used = NOW()
            WHERE app_id = $3 AND user_id = $4
            "#,
            hash_secret(&access_token),
            hash_secret(&refresh_token),
            app.id,
            auth.user_id
        )
        .execute(&self.pool)
        .await?;

        Ok((access_token, refresh_token))
    }

    pub async fn get_user_authorizations(
        &self,
        user_id: Uuid,
    ) -> Result<Vec<OAuthAuthorization>, PlatformError> {
        let auths = sqlx::query!(
            r#"
            SELECT a.id, a.user_id, a.app_id, o.name as app_name,
                   a.scopes as "scopes: Vec<String>", a.authorized_at, a.last_used
            FROM oauth_authorizations a
            JOIN oauth_apps o ON a.app_id = o.id
            WHERE a.user_id = $1
            ORDER BY a.authorized_at DESC
            "#,
            user_id
        )
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|row| OAuthAuthorization {
            id: row.id,
            user_id: row.user_id,
            app_id: row.app_id,
            app_name: row.app_name,
            scopes: row.scopes,
            authorized_at: row.authorized_at,
            last_used: row.last_used,
        })
        .collect();

        Ok(auths)
    }

    pub async fn revoke_authorization(
        &self,
        user_id: Uuid,
        app_id: Uuid,
    ) -> Result<(), PlatformError> {
        sqlx::query!(
            "DELETE FROM oauth_authorizations WHERE user_id = $1 AND app_id = $2",
            user_id,
            app_id
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }
}

fn generate_client_id() -> String {
    format!("mc_{}", Uuid::new_v4().to_string().replace("-", ""))
}

fn generate_client_secret() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let bytes: Vec<u8> = (0..32).map(|_| rng.gen()).collect();
    format!("mcs_{}", hex::encode(bytes))
}

fn generate_auth_code() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let bytes: Vec<u8> = (0..24).map(|_| rng.gen()).collect();
    hex::encode(bytes)
}

fn generate_access_token() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let bytes: Vec<u8> = (0..32).map(|_| rng.gen()).collect();
    format!("mca_{}", hex::encode(bytes))
}

fn generate_refresh_token() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let bytes: Vec<u8> = (0..32).map(|_| rng.gen()).collect();
    format!("mcr_{}", hex::encode(bytes))
}

fn hash_secret(secret: &str) -> String {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(secret.as_bytes());
    hex::encode(hasher.finalize())
}

fn verify_secret(secret: &str, hash: &str) -> bool {
    hash_secret(secret) == hash
}

// ============================================================================
// Marketplace
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketplaceListing {
    pub plugin: Plugin,
    pub install_count: i64,
    pub rating: f32,
    pub review_count: i32,
    pub categories: Vec<String>,
    pub screenshots: Vec<String>,
    pub pricing: PluginPricing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginPricing {
    Free,
    Paid { price_cents: i32, currency: String },
    Subscription { monthly_cents: i32, currency: String },
}

pub struct MarketplaceService {
    pool: PgPool,
}

impl MarketplaceService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn search(
        &self,
        query: Option<&str>,
        category: Option<&str>,
        sort: MarketplaceSort,
        limit: i32,
        offset: i32,
    ) -> Result<Vec<MarketplaceListing>, PlatformError> {
        let listings = sqlx::query!(
            r#"
            SELECT p.*, ml.install_count, ml.rating, ml.review_count,
                   ml.categories as "categories: Vec<String>",
                   ml.screenshots as "screenshots: Vec<String>",
                   ml.pricing
            FROM plugins p
            JOIN marketplace_listings ml ON p.id = ml.plugin_id
            WHERE p.status = 'Active'
            AND ($1::text IS NULL OR p.name ILIKE '%' || $1 || '%' OR p.description ILIKE '%' || $1 || '%')
            AND ($2::text IS NULL OR $2 = ANY(ml.categories))
            ORDER BY
                CASE WHEN $3 = 'popular' THEN ml.install_count END DESC,
                CASE WHEN $3 = 'rating' THEN ml.rating END DESC,
                CASE WHEN $3 = 'recent' THEN p.created_at END DESC
            LIMIT $4 OFFSET $5
            "#,
            query,
            category,
            sort.to_string(),
            limit as i64,
            offset as i64
        )
        .fetch_all(&self.pool)
        .await?;

        // Convert to MarketplaceListing (simplified for brevity)
        Ok(vec![])
    }

    pub async fn get_featured(&self) -> Result<Vec<MarketplaceListing>, PlatformError> {
        // Return featured plugins
        Ok(vec![])
    }

    pub async fn submit_review(
        &self,
        user_id: Uuid,
        plugin_id: Uuid,
        rating: i32,
        review: &str,
    ) -> Result<(), PlatformError> {
        if rating < 1 || rating > 5 {
            return Err(PlatformError::InvalidManifest(
                "Rating must be between 1 and 5".to_string(),
            ));
        }

        sqlx::query!(
            r#"
            INSERT INTO plugin_reviews (plugin_id, user_id, rating, review, created_at)
            VALUES ($1, $2, $3, $4, NOW())
            ON CONFLICT (plugin_id, user_id) DO UPDATE SET
                rating = EXCLUDED.rating,
                review = EXCLUDED.review,
                updated_at = NOW()
            "#,
            plugin_id,
            user_id,
            rating,
            review
        )
        .execute(&self.pool)
        .await?;

        // Update aggregate rating
        sqlx::query!(
            r#"
            UPDATE marketplace_listings
            SET rating = (SELECT AVG(rating) FROM plugin_reviews WHERE plugin_id = $1),
                review_count = (SELECT COUNT(*) FROM plugin_reviews WHERE plugin_id = $1)
            WHERE plugin_id = $1
            "#,
            plugin_id
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketplaceSort {
    Popular,
    Rating,
    Recent,
}

impl ToString for MarketplaceSort {
    fn to_string(&self) -> String {
        match self {
            MarketplaceSort::Popular => "popular".to_string(),
            MarketplaceSort::Rating => "rating".to_string(),
            MarketplaceSort::Recent => "recent".to_string(),
        }
    }
}
