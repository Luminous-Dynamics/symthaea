// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Plugin System for Mycelix Mail
//!
//! Extensible plugin architecture supporting:
//! - Action plugins (email processing, filtering)
//! - Integration plugins (external services)
//! - UI plugins (custom views, widgets)
//! - Trust plugins (custom attestation types)

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

// ============================================================================
// Plugin Traits
// ============================================================================

/// Core trait that all plugins must implement
#[async_trait]
pub trait Plugin: Send + Sync {
    /// Unique identifier for the plugin
    fn id(&self) -> &str;

    /// Human-readable name
    fn name(&self) -> &str;

    /// Plugin version
    fn version(&self) -> &str;

    /// Plugin description
    fn description(&self) -> &str;

    /// Plugin author
    fn author(&self) -> &str;

    /// Initialize the plugin
    async fn initialize(&mut self, config: PluginConfig) -> Result<(), PluginError>;

    /// Shutdown the plugin
    async fn shutdown(&mut self) -> Result<(), PluginError>;

    /// Get plugin capabilities
    fn capabilities(&self) -> Vec<PluginCapability>;

    /// Health check
    async fn health_check(&self) -> PluginHealth;
}

/// Email processing plugin
#[async_trait]
pub trait EmailPlugin: Plugin {
    /// Process incoming email
    async fn process_incoming(&self, email: &mut EmailContext) -> Result<EmailAction, PluginError>;

    /// Process outgoing email
    async fn process_outgoing(&self, email: &mut EmailContext) -> Result<EmailAction, PluginError>;

    /// Get priority (lower = earlier in chain)
    fn priority(&self) -> i32 {
        100
    }
}

/// Integration plugin for external services
#[async_trait]
pub trait IntegrationPlugin: Plugin {
    /// Get available actions
    fn actions(&self) -> Vec<IntegrationAction>;

    /// Execute an action
    async fn execute(&self, action: &str, params: Value) -> Result<Value, PluginError>;

    /// Handle webhook
    async fn handle_webhook(&self, payload: Value) -> Result<Value, PluginError>;

    /// Get OAuth configuration if needed
    fn oauth_config(&self) -> Option<OAuthConfig> {
        None
    }
}

/// Trust/attestation plugin
#[async_trait]
pub trait TrustPlugin: Plugin {
    /// Get custom attestation types
    fn attestation_types(&self) -> Vec<AttestationType>;

    /// Validate an attestation
    async fn validate_attestation(&self, attestation: &Value) -> Result<bool, PluginError>;

    /// Calculate trust contribution
    async fn calculate_trust(&self, attestations: &[Value]) -> Result<f64, PluginError>;
}

/// UI extension plugin
pub trait UIPlugin: Plugin {
    /// Get UI components to register
    fn components(&self) -> Vec<UIComponent>;

    /// Get custom routes
    fn routes(&self) -> Vec<UIRoute>;

    /// Get sidebar items
    fn sidebar_items(&self) -> Vec<SidebarItem>;
}

// ============================================================================
// Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    pub settings: HashMap<String, Value>,
    pub secrets: HashMap<String, String>,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    pub id: String,
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub license: String,
    pub homepage: Option<String>,
    pub repository: Option<String>,
    pub capabilities: Vec<PluginCapability>,
    pub permissions: Vec<PluginPermission>,
    pub settings_schema: Option<Value>,
    pub dependencies: Vec<PluginDependency>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PluginCapability {
    EmailProcessing,
    Integration,
    Trust,
    UI,
    Storage,
    Notifications,
    Scheduling,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginPermission {
    ReadEmails,
    WriteEmails,
    ReadContacts,
    WriteContacts,
    ReadCalendar,
    WriteCalendar,
    ReadTrust,
    WriteTrust,
    ExternalNetwork,
    Storage,
    Notifications,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginDependency {
    pub plugin_id: String,
    pub version_requirement: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginHealth {
    pub healthy: bool,
    pub message: Option<String>,
    pub last_check: chrono::DateTime<chrono::Utc>,
    pub metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailContext {
    pub id: String,
    pub from: String,
    pub to: Vec<String>,
    pub cc: Vec<String>,
    pub subject: String,
    pub body_text: String,
    pub body_html: Option<String>,
    pub headers: HashMap<String, String>,
    pub attachments: Vec<AttachmentInfo>,
    pub metadata: HashMap<String, Value>,
    pub trust_score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttachmentInfo {
    pub id: String,
    pub filename: String,
    pub content_type: String,
    pub size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmailAction {
    Continue,
    Skip,
    Reject(String),
    Quarantine(String),
    Modify(EmailContext),
    AddLabel(String),
    RemoveLabel(String),
    Forward(String),
    AutoReply(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationAction {
    pub id: String,
    pub name: String,
    pub description: String,
    pub parameters: Vec<ActionParameter>,
    pub returns: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionParameter {
    pub name: String,
    pub param_type: String,
    pub required: bool,
    pub description: String,
    pub default: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthConfig {
    pub authorization_url: String,
    pub token_url: String,
    pub scopes: Vec<String>,
    pub client_id_setting: String,
    pub client_secret_setting: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationType {
    pub id: String,
    pub name: String,
    pub description: String,
    pub schema: Value,
    pub trust_weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UIComponent {
    pub id: String,
    pub name: String,
    pub component_type: UIComponentType,
    pub mount_point: String,
    pub props_schema: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UIComponentType {
    Widget,
    Panel,
    Modal,
    Toolbar,
    Setting,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UIRoute {
    pub path: String,
    pub component: String,
    pub title: String,
    pub icon: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidebarItem {
    pub id: String,
    pub label: String,
    pub icon: String,
    pub route: String,
    pub badge: Option<String>,
    pub order: i32,
}

#[derive(Debug, thiserror::Error)]
pub enum PluginError {
    #[error("Plugin initialization failed: {0}")]
    InitializationError(String),
    #[error("Plugin not found: {0}")]
    NotFound(String),
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    #[error("Configuration error: {0}")]
    ConfigError(String),
    #[error("Execution error: {0}")]
    ExecutionError(String),
    #[error("Dependency error: {0}")]
    DependencyError(String),
    #[error("Internal error: {0}")]
    InternalError(String),
}

// ============================================================================
// Plugin Manager
// ============================================================================

pub struct PluginManager {
    plugins: RwLock<HashMap<String, Arc<dyn Plugin>>>,
    email_plugins: RwLock<Vec<Arc<dyn EmailPlugin>>>,
    integration_plugins: RwLock<HashMap<String, Arc<dyn IntegrationPlugin>>>,
    trust_plugins: RwLock<Vec<Arc<dyn TrustPlugin>>>,
    configs: RwLock<HashMap<String, PluginConfig>>,
}

impl PluginManager {
    pub fn new() -> Self {
        Self {
            plugins: RwLock::new(HashMap::new()),
            email_plugins: RwLock::new(Vec::new()),
            integration_plugins: RwLock::new(HashMap::new()),
            trust_plugins: RwLock::new(Vec::new()),
            configs: RwLock::new(HashMap::new()),
        }
    }

    /// Register a plugin
    pub async fn register<P: Plugin + 'static>(&self, plugin: P) -> Result<(), PluginError> {
        let id = plugin.id().to_string();

        // Store in main registry
        let plugin_arc: Arc<dyn Plugin> = Arc::new(plugin);
        self.plugins.write().await.insert(id.clone(), plugin_arc);

        Ok(())
    }

    /// Register an email processing plugin
    pub async fn register_email_plugin<P: EmailPlugin + 'static>(
        &self,
        plugin: P,
    ) -> Result<(), PluginError> {
        let plugin_arc: Arc<dyn EmailPlugin> = Arc::new(plugin);

        // Insert sorted by priority
        let mut plugins = self.email_plugins.write().await;
        let priority = plugin_arc.priority();
        let pos = plugins
            .iter()
            .position(|p| p.priority() > priority)
            .unwrap_or(plugins.len());
        plugins.insert(pos, plugin_arc);

        Ok(())
    }

    /// Register an integration plugin
    pub async fn register_integration_plugin<P: IntegrationPlugin + 'static>(
        &self,
        plugin: P,
    ) -> Result<(), PluginError> {
        let id = plugin.id().to_string();
        let plugin_arc: Arc<dyn IntegrationPlugin> = Arc::new(plugin);
        self.integration_plugins.write().await.insert(id, plugin_arc);
        Ok(())
    }

    /// Register a trust plugin
    pub async fn register_trust_plugin<P: TrustPlugin + 'static>(
        &self,
        plugin: P,
    ) -> Result<(), PluginError> {
        let plugin_arc: Arc<dyn TrustPlugin> = Arc::new(plugin);
        self.trust_plugins.write().await.push(plugin_arc);
        Ok(())
    }

    /// Initialize all plugins
    pub async fn initialize_all(&self) -> Result<(), PluginError> {
        let configs = self.configs.read().await;

        for (id, plugin) in self.plugins.write().await.iter_mut() {
            let config = configs.get(id).cloned().unwrap_or(PluginConfig {
                settings: HashMap::new(),
                secrets: HashMap::new(),
                enabled: true,
            });

            if config.enabled {
                // Would need interior mutability pattern here
                // Arc::get_mut(plugin).unwrap().initialize(config).await?;
            }
        }

        Ok(())
    }

    /// Process email through plugin chain
    pub async fn process_incoming_email(
        &self,
        email: &mut EmailContext,
    ) -> Result<EmailAction, PluginError> {
        let plugins = self.email_plugins.read().await;

        for plugin in plugins.iter() {
            match plugin.process_incoming(email).await? {
                EmailAction::Continue => continue,
                action => return Ok(action),
            }
        }

        Ok(EmailAction::Continue)
    }

    /// Process outgoing email through plugin chain
    pub async fn process_outgoing_email(
        &self,
        email: &mut EmailContext,
    ) -> Result<EmailAction, PluginError> {
        let plugins = self.email_plugins.read().await;

        for plugin in plugins.iter() {
            match plugin.process_outgoing(email).await? {
                EmailAction::Continue => continue,
                action => return Ok(action),
            }
        }

        Ok(EmailAction::Continue)
    }

    /// Execute integration action
    pub async fn execute_integration(
        &self,
        plugin_id: &str,
        action: &str,
        params: Value,
    ) -> Result<Value, PluginError> {
        let plugins = self.integration_plugins.read().await;

        let plugin = plugins
            .get(plugin_id)
            .ok_or_else(|| PluginError::NotFound(plugin_id.to_string()))?;

        plugin.execute(action, params).await
    }

    /// Calculate aggregate trust score
    pub async fn calculate_trust(&self, attestations: &[Value]) -> Result<f64, PluginError> {
        let plugins = self.trust_plugins.read().await;

        if plugins.is_empty() {
            return Ok(0.0);
        }

        let mut total_score = 0.0;
        let mut total_weight = 0.0;

        for plugin in plugins.iter() {
            let score = plugin.calculate_trust(attestations).await?;
            // Could use weighted average based on plugin configuration
            total_score += score;
            total_weight += 1.0;
        }

        Ok(if total_weight > 0.0 {
            total_score / total_weight
        } else {
            0.0
        })
    }

    /// Get all plugin health statuses
    pub async fn health_check_all(&self) -> HashMap<String, PluginHealth> {
        let mut results = HashMap::new();
        let plugins = self.plugins.read().await;

        for (id, plugin) in plugins.iter() {
            results.insert(id.clone(), plugin.health_check().await);
        }

        results
    }

    /// List all registered plugins
    pub async fn list_plugins(&self) -> Vec<PluginInfo> {
        let plugins = self.plugins.read().await;

        plugins
            .iter()
            .map(|(id, plugin)| PluginInfo {
                id: id.clone(),
                name: plugin.name().to_string(),
                version: plugin.version().to_string(),
                description: plugin.description().to_string(),
                capabilities: plugin.capabilities(),
            })
            .collect()
    }

    /// Get plugin configuration
    pub async fn get_config(&self, plugin_id: &str) -> Option<PluginConfig> {
        self.configs.read().await.get(plugin_id).cloned()
    }

    /// Update plugin configuration
    pub async fn update_config(
        &self,
        plugin_id: &str,
        config: PluginConfig,
    ) -> Result<(), PluginError> {
        self.configs.write().await.insert(plugin_id.to_string(), config);
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginInfo {
    pub id: String,
    pub name: String,
    pub version: String,
    pub description: String,
    pub capabilities: Vec<PluginCapability>,
}

// ============================================================================
// Built-in Plugins
// ============================================================================

/// Spam filter plugin
pub struct SpamFilterPlugin {
    threshold: f64,
}

#[async_trait]
impl Plugin for SpamFilterPlugin {
    fn id(&self) -> &str {
        "builtin.spam-filter"
    }
    fn name(&self) -> &str {
        "Spam Filter"
    }
    fn version(&self) -> &str {
        "1.0.0"
    }
    fn description(&self) -> &str {
        "Built-in spam detection and filtering"
    }
    fn author(&self) -> &str {
        "Mycelix Team"
    }

    async fn initialize(&mut self, config: PluginConfig) -> Result<(), PluginError> {
        if let Some(threshold) = config.settings.get("threshold") {
            self.threshold = threshold.as_f64().unwrap_or(0.7);
        }
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<(), PluginError> {
        Ok(())
    }

    fn capabilities(&self) -> Vec<PluginCapability> {
        vec![PluginCapability::EmailProcessing]
    }

    async fn health_check(&self) -> PluginHealth {
        PluginHealth {
            healthy: true,
            message: None,
            last_check: chrono::Utc::now(),
            metrics: HashMap::new(),
        }
    }
}

#[async_trait]
impl EmailPlugin for SpamFilterPlugin {
    async fn process_incoming(&self, email: &mut EmailContext) -> Result<EmailAction, PluginError> {
        // Simple spam detection (would use ML in production)
        let spam_score = calculate_spam_score(email);

        email.metadata.insert(
            "spam_score".to_string(),
            serde_json::json!(spam_score),
        );

        if spam_score > self.threshold {
            Ok(EmailAction::Quarantine(format!(
                "Spam detected (score: {:.2})",
                spam_score
            )))
        } else {
            Ok(EmailAction::Continue)
        }
    }

    async fn process_outgoing(&self, _email: &mut EmailContext) -> Result<EmailAction, PluginError> {
        Ok(EmailAction::Continue)
    }

    fn priority(&self) -> i32 {
        10 // Run early
    }
}

fn calculate_spam_score(email: &EmailContext) -> f64 {
    let mut score = 0.0;
    let text = format!("{} {}", email.subject, email.body_text).to_lowercase();

    // Simple keyword detection
    let spam_keywords = [
        "buy now", "limited time", "act now", "free", "winner",
        "congratulations", "urgent", "click here",
    ];

    for keyword in spam_keywords {
        if text.contains(keyword) {
            score += 0.15;
        }
    }

    // Check for excessive caps
    let caps_ratio = email.subject.chars().filter(|c| c.is_uppercase()).count() as f64
        / email.subject.len().max(1) as f64;
    if caps_ratio > 0.5 {
        score += 0.2;
    }

    score.min(1.0)
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}
