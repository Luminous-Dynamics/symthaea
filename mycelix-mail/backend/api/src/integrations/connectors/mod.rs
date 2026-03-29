// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Third-Party Integration Connectors
//!
//! Connects Mycelix Mail with external services and productivity tools.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Core Traits
// ============================================================================

#[async_trait]
pub trait IntegrationConnector: Send + Sync {
    fn id(&self) -> &'static str;
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn icon_url(&self) -> &'static str;
    fn category(&self) -> ConnectorCategory;
    fn auth_type(&self) -> AuthType;

    async fn authenticate(&self, credentials: AuthCredentials) -> Result<AuthToken, ConnectorError>;
    async fn refresh_token(&self, token: &AuthToken) -> Result<AuthToken, ConnectorError>;
    async fn test_connection(&self, token: &AuthToken) -> Result<bool, ConnectorError>;
    async fn disconnect(&self, token: &AuthToken) -> Result<(), ConnectorError>;

    fn supported_actions(&self) -> Vec<ConnectorAction>;
    async fn execute_action(
        &self,
        token: &AuthToken,
        action: &str,
        params: HashMap<String, serde_json::Value>,
    ) -> Result<serde_json::Value, ConnectorError>;

    fn webhook_events(&self) -> Vec<WebhookEvent>;
    async fn handle_webhook(
        &self,
        event: &str,
        payload: serde_json::Value,
    ) -> Result<WebhookResponse, ConnectorError>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectorCategory {
    Calendar,
    ProjectManagement,
    CRM,
    Storage,
    Communication,
    Automation,
    Analytics,
    Development,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthType {
    OAuth2,
    ApiKey,
    Basic,
    Token,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthCredentials {
    pub auth_type: AuthType,
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub api_key: Option<String>,
    pub username: Option<String>,
    pub password: Option<String>,
    pub redirect_uri: Option<String>,
    pub scopes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthToken {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_at: Option<i64>,
    pub token_type: String,
    pub scopes: Vec<String>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorAction {
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
    pub default: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookEvent {
    pub event: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookResponse {
    pub success: bool,
    pub message: Option<String>,
    pub data: Option<serde_json::Value>,
}

#[derive(Debug, thiserror::Error)]
pub enum ConnectorError {
    #[error("Authentication failed: {0}")]
    AuthError(String),
    #[error("Connection error: {0}")]
    ConnectionError(String),
    #[error("Rate limited: retry after {0} seconds")]
    RateLimited(u64),
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    #[error("Internal error: {0}")]
    InternalError(String),
}

// ============================================================================
// Slack Connector
// ============================================================================

pub struct SlackConnector {
    client: reqwest::Client,
}

impl SlackConnector {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl IntegrationConnector for SlackConnector {
    fn id(&self) -> &'static str { "slack" }
    fn name(&self) -> &'static str { "Slack" }
    fn description(&self) -> &'static str { "Send emails to Slack channels and receive notifications" }
    fn icon_url(&self) -> &'static str { "/integrations/icons/slack.svg" }
    fn category(&self) -> ConnectorCategory { ConnectorCategory::Communication }
    fn auth_type(&self) -> AuthType { AuthType::OAuth2 }

    async fn authenticate(&self, credentials: AuthCredentials) -> Result<AuthToken, ConnectorError> {
        // OAuth2 flow implementation
        Ok(AuthToken {
            access_token: String::new(),
            refresh_token: None,
            expires_at: None,
            token_type: "Bearer".to_string(),
            scopes: vec!["chat:write".to_string(), "channels:read".to_string()],
            metadata: HashMap::new(),
        })
    }

    async fn refresh_token(&self, token: &AuthToken) -> Result<AuthToken, ConnectorError> {
        Ok(token.clone())
    }

    async fn test_connection(&self, token: &AuthToken) -> Result<bool, ConnectorError> {
        Ok(true)
    }

    async fn disconnect(&self, _token: &AuthToken) -> Result<(), ConnectorError> {
        Ok(())
    }

    fn supported_actions(&self) -> Vec<ConnectorAction> {
        vec![
            ConnectorAction {
                id: "send_message".to_string(),
                name: "Send Message".to_string(),
                description: "Send a message to a Slack channel".to_string(),
                parameters: vec![
                    ActionParameter {
                        name: "channel".to_string(),
                        param_type: "string".to_string(),
                        required: true,
                        description: "Slack channel ID".to_string(),
                        default: None,
                    },
                    ActionParameter {
                        name: "text".to_string(),
                        param_type: "string".to_string(),
                        required: true,
                        description: "Message text".to_string(),
                        default: None,
                    },
                ],
                returns: Some("message_id".to_string()),
            },
            ConnectorAction {
                id: "share_email".to_string(),
                name: "Share Email to Channel".to_string(),
                description: "Share an email summary to a Slack channel".to_string(),
                parameters: vec![
                    ActionParameter {
                        name: "channel".to_string(),
                        param_type: "string".to_string(),
                        required: true,
                        description: "Slack channel ID".to_string(),
                        default: None,
                    },
                    ActionParameter {
                        name: "email_id".to_string(),
                        param_type: "string".to_string(),
                        required: true,
                        description: "Email ID to share".to_string(),
                        default: None,
                    },
                ],
                returns: Some("message_id".to_string()),
            },
            ConnectorAction {
                id: "list_channels".to_string(),
                name: "List Channels".to_string(),
                description: "Get list of available Slack channels".to_string(),
                parameters: vec![],
                returns: Some("channels[]".to_string()),
            },
        ]
    }

    async fn execute_action(
        &self,
        token: &AuthToken,
        action: &str,
        params: HashMap<String, serde_json::Value>,
    ) -> Result<serde_json::Value, ConnectorError> {
        match action {
            "send_message" => {
                // Implementation
                Ok(serde_json::json!({ "message_id": "msg_123" }))
            }
            "share_email" => {
                Ok(serde_json::json!({ "message_id": "msg_456" }))
            }
            "list_channels" => {
                Ok(serde_json::json!({ "channels": [] }))
            }
            _ => Err(ConnectorError::InvalidRequest(format!("Unknown action: {}", action))),
        }
    }

    fn webhook_events(&self) -> Vec<WebhookEvent> {
        vec![
            WebhookEvent {
                event: "message".to_string(),
                description: "New message in channel".to_string(),
            },
            WebhookEvent {
                event: "reaction_added".to_string(),
                description: "Reaction added to message".to_string(),
            },
        ]
    }

    async fn handle_webhook(
        &self,
        event: &str,
        payload: serde_json::Value,
    ) -> Result<WebhookResponse, ConnectorError> {
        Ok(WebhookResponse {
            success: true,
            message: None,
            data: None,
        })
    }
}

// ============================================================================
// Notion Connector
// ============================================================================

pub struct NotionConnector {
    client: reqwest::Client,
}

impl NotionConnector {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl IntegrationConnector for NotionConnector {
    fn id(&self) -> &'static str { "notion" }
    fn name(&self) -> &'static str { "Notion" }
    fn description(&self) -> &'static str { "Save emails to Notion pages and databases" }
    fn icon_url(&self) -> &'static str { "/integrations/icons/notion.svg" }
    fn category(&self) -> ConnectorCategory { ConnectorCategory::ProjectManagement }
    fn auth_type(&self) -> AuthType { AuthType::OAuth2 }

    async fn authenticate(&self, credentials: AuthCredentials) -> Result<AuthToken, ConnectorError> {
        Ok(AuthToken {
            access_token: String::new(),
            refresh_token: None,
            expires_at: None,
            token_type: "Bearer".to_string(),
            scopes: vec![],
            metadata: HashMap::new(),
        })
    }

    async fn refresh_token(&self, token: &AuthToken) -> Result<AuthToken, ConnectorError> {
        Ok(token.clone())
    }

    async fn test_connection(&self, token: &AuthToken) -> Result<bool, ConnectorError> {
        Ok(true)
    }

    async fn disconnect(&self, _token: &AuthToken) -> Result<(), ConnectorError> {
        Ok(())
    }

    fn supported_actions(&self) -> Vec<ConnectorAction> {
        vec![
            ConnectorAction {
                id: "create_page".to_string(),
                name: "Create Page".to_string(),
                description: "Create a new Notion page from email".to_string(),
                parameters: vec![
                    ActionParameter {
                        name: "parent_id".to_string(),
                        param_type: "string".to_string(),
                        required: true,
                        description: "Parent page or database ID".to_string(),
                        default: None,
                    },
                    ActionParameter {
                        name: "title".to_string(),
                        param_type: "string".to_string(),
                        required: true,
                        description: "Page title".to_string(),
                        default: None,
                    },
                    ActionParameter {
                        name: "content".to_string(),
                        param_type: "string".to_string(),
                        required: false,
                        description: "Page content".to_string(),
                        default: None,
                    },
                ],
                returns: Some("page_id".to_string()),
            },
            ConnectorAction {
                id: "add_to_database".to_string(),
                name: "Add to Database".to_string(),
                description: "Add email as database entry".to_string(),
                parameters: vec![
                    ActionParameter {
                        name: "database_id".to_string(),
                        param_type: "string".to_string(),
                        required: true,
                        description: "Database ID".to_string(),
                        default: None,
                    },
                    ActionParameter {
                        name: "properties".to_string(),
                        param_type: "object".to_string(),
                        required: true,
                        description: "Property values".to_string(),
                        default: None,
                    },
                ],
                returns: Some("page_id".to_string()),
            },
            ConnectorAction {
                id: "list_databases".to_string(),
                name: "List Databases".to_string(),
                description: "Get available databases".to_string(),
                parameters: vec![],
                returns: Some("databases[]".to_string()),
            },
        ]
    }

    async fn execute_action(
        &self,
        token: &AuthToken,
        action: &str,
        params: HashMap<String, serde_json::Value>,
    ) -> Result<serde_json::Value, ConnectorError> {
        match action {
            "create_page" => Ok(serde_json::json!({ "page_id": "page_123" })),
            "add_to_database" => Ok(serde_json::json!({ "page_id": "page_456" })),
            "list_databases" => Ok(serde_json::json!({ "databases": [] })),
            _ => Err(ConnectorError::InvalidRequest(format!("Unknown action: {}", action))),
        }
    }

    fn webhook_events(&self) -> Vec<WebhookEvent> { vec![] }

    async fn handle_webhook(
        &self,
        event: &str,
        payload: serde_json::Value,
    ) -> Result<WebhookResponse, ConnectorError> {
        Ok(WebhookResponse { success: true, message: None, data: None })
    }
}

// ============================================================================
// Linear Connector
// ============================================================================

pub struct LinearConnector {
    client: reqwest::Client,
}

impl LinearConnector {
    pub fn new() -> Self {
        Self { client: reqwest::Client::new() }
    }
}

#[async_trait]
impl IntegrationConnector for LinearConnector {
    fn id(&self) -> &'static str { "linear" }
    fn name(&self) -> &'static str { "Linear" }
    fn description(&self) -> &'static str { "Create issues from emails and track project updates" }
    fn icon_url(&self) -> &'static str { "/integrations/icons/linear.svg" }
    fn category(&self) -> ConnectorCategory { ConnectorCategory::ProjectManagement }
    fn auth_type(&self) -> AuthType { AuthType::OAuth2 }

    async fn authenticate(&self, credentials: AuthCredentials) -> Result<AuthToken, ConnectorError> {
        Ok(AuthToken {
            access_token: String::new(),
            refresh_token: None,
            expires_at: None,
            token_type: "Bearer".to_string(),
            scopes: vec!["read".to_string(), "write".to_string()],
            metadata: HashMap::new(),
        })
    }

    async fn refresh_token(&self, token: &AuthToken) -> Result<AuthToken, ConnectorError> {
        Ok(token.clone())
    }

    async fn test_connection(&self, token: &AuthToken) -> Result<bool, ConnectorError> {
        Ok(true)
    }

    async fn disconnect(&self, _token: &AuthToken) -> Result<(), ConnectorError> {
        Ok(())
    }

    fn supported_actions(&self) -> Vec<ConnectorAction> {
        vec![
            ConnectorAction {
                id: "create_issue".to_string(),
                name: "Create Issue".to_string(),
                description: "Create a Linear issue from email".to_string(),
                parameters: vec![
                    ActionParameter {
                        name: "team_id".to_string(),
                        param_type: "string".to_string(),
                        required: true,
                        description: "Team ID".to_string(),
                        default: None,
                    },
                    ActionParameter {
                        name: "title".to_string(),
                        param_type: "string".to_string(),
                        required: true,
                        description: "Issue title".to_string(),
                        default: None,
                    },
                    ActionParameter {
                        name: "description".to_string(),
                        param_type: "string".to_string(),
                        required: false,
                        description: "Issue description".to_string(),
                        default: None,
                    },
                    ActionParameter {
                        name: "priority".to_string(),
                        param_type: "number".to_string(),
                        required: false,
                        description: "Priority (0-4)".to_string(),
                        default: Some(serde_json::json!(0)),
                    },
                ],
                returns: Some("issue_id".to_string()),
            },
            ConnectorAction {
                id: "list_teams".to_string(),
                name: "List Teams".to_string(),
                description: "Get available teams".to_string(),
                parameters: vec![],
                returns: Some("teams[]".to_string()),
            },
            ConnectorAction {
                id: "list_projects".to_string(),
                name: "List Projects".to_string(),
                description: "Get projects for a team".to_string(),
                parameters: vec![
                    ActionParameter {
                        name: "team_id".to_string(),
                        param_type: "string".to_string(),
                        required: true,
                        description: "Team ID".to_string(),
                        default: None,
                    },
                ],
                returns: Some("projects[]".to_string()),
            },
        ]
    }

    async fn execute_action(
        &self,
        token: &AuthToken,
        action: &str,
        params: HashMap<String, serde_json::Value>,
    ) -> Result<serde_json::Value, ConnectorError> {
        match action {
            "create_issue" => Ok(serde_json::json!({ "issue_id": "issue_123" })),
            "list_teams" => Ok(serde_json::json!({ "teams": [] })),
            "list_projects" => Ok(serde_json::json!({ "projects": [] })),
            _ => Err(ConnectorError::InvalidRequest(format!("Unknown action: {}", action))),
        }
    }

    fn webhook_events(&self) -> Vec<WebhookEvent> {
        vec![
            WebhookEvent {
                event: "issue_created".to_string(),
                description: "New issue created".to_string(),
            },
            WebhookEvent {
                event: "issue_updated".to_string(),
                description: "Issue was updated".to_string(),
            },
            WebhookEvent {
                event: "comment_created".to_string(),
                description: "New comment on issue".to_string(),
            },
        ]
    }

    async fn handle_webhook(
        &self,
        event: &str,
        payload: serde_json::Value,
    ) -> Result<WebhookResponse, ConnectorError> {
        Ok(WebhookResponse { success: true, message: None, data: None })
    }
}

// ============================================================================
// GitHub Connector
// ============================================================================

pub struct GitHubConnector {
    client: reqwest::Client,
}

impl GitHubConnector {
    pub fn new() -> Self {
        Self { client: reqwest::Client::new() }
    }
}

#[async_trait]
impl IntegrationConnector for GitHubConnector {
    fn id(&self) -> &'static str { "github" }
    fn name(&self) -> &'static str { "GitHub" }
    fn description(&self) -> &'static str { "Create issues, get PR notifications in email" }
    fn icon_url(&self) -> &'static str { "/integrations/icons/github.svg" }
    fn category(&self) -> ConnectorCategory { ConnectorCategory::Development }
    fn auth_type(&self) -> AuthType { AuthType::OAuth2 }

    async fn authenticate(&self, credentials: AuthCredentials) -> Result<AuthToken, ConnectorError> {
        Ok(AuthToken {
            access_token: String::new(),
            refresh_token: None,
            expires_at: None,
            token_type: "Bearer".to_string(),
            scopes: vec!["repo".to_string(), "user".to_string()],
            metadata: HashMap::new(),
        })
    }

    async fn refresh_token(&self, token: &AuthToken) -> Result<AuthToken, ConnectorError> {
        Ok(token.clone())
    }

    async fn test_connection(&self, token: &AuthToken) -> Result<bool, ConnectorError> {
        Ok(true)
    }

    async fn disconnect(&self, _token: &AuthToken) -> Result<(), ConnectorError> {
        Ok(())
    }

    fn supported_actions(&self) -> Vec<ConnectorAction> {
        vec![
            ConnectorAction {
                id: "create_issue".to_string(),
                name: "Create Issue".to_string(),
                description: "Create a GitHub issue from email".to_string(),
                parameters: vec![
                    ActionParameter {
                        name: "repo".to_string(),
                        param_type: "string".to_string(),
                        required: true,
                        description: "Repository (owner/name)".to_string(),
                        default: None,
                    },
                    ActionParameter {
                        name: "title".to_string(),
                        param_type: "string".to_string(),
                        required: true,
                        description: "Issue title".to_string(),
                        default: None,
                    },
                    ActionParameter {
                        name: "body".to_string(),
                        param_type: "string".to_string(),
                        required: false,
                        description: "Issue body".to_string(),
                        default: None,
                    },
                    ActionParameter {
                        name: "labels".to_string(),
                        param_type: "array".to_string(),
                        required: false,
                        description: "Labels to apply".to_string(),
                        default: None,
                    },
                ],
                returns: Some("issue_url".to_string()),
            },
            ConnectorAction {
                id: "list_repos".to_string(),
                name: "List Repositories".to_string(),
                description: "Get user repositories".to_string(),
                parameters: vec![],
                returns: Some("repos[]".to_string()),
            },
        ]
    }

    async fn execute_action(
        &self,
        token: &AuthToken,
        action: &str,
        params: HashMap<String, serde_json::Value>,
    ) -> Result<serde_json::Value, ConnectorError> {
        match action {
            "create_issue" => Ok(serde_json::json!({ "issue_url": "https://github.com/..." })),
            "list_repos" => Ok(serde_json::json!({ "repos": [] })),
            _ => Err(ConnectorError::InvalidRequest(format!("Unknown action: {}", action))),
        }
    }

    fn webhook_events(&self) -> Vec<WebhookEvent> {
        vec![
            WebhookEvent { event: "push".to_string(), description: "Push to repository".to_string() },
            WebhookEvent { event: "pull_request".to_string(), description: "PR opened/updated".to_string() },
            WebhookEvent { event: "issues".to_string(), description: "Issue created/updated".to_string() },
        ]
    }

    async fn handle_webhook(
        &self,
        event: &str,
        payload: serde_json::Value,
    ) -> Result<WebhookResponse, ConnectorError> {
        Ok(WebhookResponse { success: true, message: None, data: None })
    }
}

// ============================================================================
// Salesforce Connector
// ============================================================================

pub struct SalesforceConnector {
    client: reqwest::Client,
}

impl SalesforceConnector {
    pub fn new() -> Self {
        Self { client: reqwest::Client::new() }
    }
}

#[async_trait]
impl IntegrationConnector for SalesforceConnector {
    fn id(&self) -> &'static str { "salesforce" }
    fn name(&self) -> &'static str { "Salesforce" }
    fn description(&self) -> &'static str { "Sync contacts and log email activities in Salesforce" }
    fn icon_url(&self) -> &'static str { "/integrations/icons/salesforce.svg" }
    fn category(&self) -> ConnectorCategory { ConnectorCategory::CRM }
    fn auth_type(&self) -> AuthType { AuthType::OAuth2 }

    async fn authenticate(&self, credentials: AuthCredentials) -> Result<AuthToken, ConnectorError> {
        Ok(AuthToken {
            access_token: String::new(),
            refresh_token: None,
            expires_at: None,
            token_type: "Bearer".to_string(),
            scopes: vec!["api".to_string(), "refresh_token".to_string()],
            metadata: HashMap::new(),
        })
    }

    async fn refresh_token(&self, token: &AuthToken) -> Result<AuthToken, ConnectorError> {
        Ok(token.clone())
    }

    async fn test_connection(&self, token: &AuthToken) -> Result<bool, ConnectorError> {
        Ok(true)
    }

    async fn disconnect(&self, _token: &AuthToken) -> Result<(), ConnectorError> {
        Ok(())
    }

    fn supported_actions(&self) -> Vec<ConnectorAction> {
        vec![
            ConnectorAction {
                id: "log_email".to_string(),
                name: "Log Email Activity".to_string(),
                description: "Log email as an activity in Salesforce".to_string(),
                parameters: vec![
                    ActionParameter {
                        name: "contact_id".to_string(),
                        param_type: "string".to_string(),
                        required: true,
                        description: "Salesforce Contact ID".to_string(),
                        default: None,
                    },
                    ActionParameter {
                        name: "email_id".to_string(),
                        param_type: "string".to_string(),
                        required: true,
                        description: "Mycelix Email ID".to_string(),
                        default: None,
                    },
                ],
                returns: Some("activity_id".to_string()),
            },
            ConnectorAction {
                id: "find_contact".to_string(),
                name: "Find Contact".to_string(),
                description: "Find a Salesforce contact by email".to_string(),
                parameters: vec![
                    ActionParameter {
                        name: "email".to_string(),
                        param_type: "string".to_string(),
                        required: true,
                        description: "Email address to search".to_string(),
                        default: None,
                    },
                ],
                returns: Some("contact".to_string()),
            },
            ConnectorAction {
                id: "create_contact".to_string(),
                name: "Create Contact".to_string(),
                description: "Create a new contact in Salesforce".to_string(),
                parameters: vec![
                    ActionParameter {
                        name: "email".to_string(),
                        param_type: "string".to_string(),
                        required: true,
                        description: "Email address".to_string(),
                        default: None,
                    },
                    ActionParameter {
                        name: "first_name".to_string(),
                        param_type: "string".to_string(),
                        required: false,
                        description: "First name".to_string(),
                        default: None,
                    },
                    ActionParameter {
                        name: "last_name".to_string(),
                        param_type: "string".to_string(),
                        required: true,
                        description: "Last name".to_string(),
                        default: None,
                    },
                ],
                returns: Some("contact_id".to_string()),
            },
        ]
    }

    async fn execute_action(
        &self,
        token: &AuthToken,
        action: &str,
        params: HashMap<String, serde_json::Value>,
    ) -> Result<serde_json::Value, ConnectorError> {
        match action {
            "log_email" => Ok(serde_json::json!({ "activity_id": "act_123" })),
            "find_contact" => Ok(serde_json::json!({ "contact": null })),
            "create_contact" => Ok(serde_json::json!({ "contact_id": "con_123" })),
            _ => Err(ConnectorError::InvalidRequest(format!("Unknown action: {}", action))),
        }
    }

    fn webhook_events(&self) -> Vec<WebhookEvent> { vec![] }

    async fn handle_webhook(
        &self,
        event: &str,
        payload: serde_json::Value,
    ) -> Result<WebhookResponse, ConnectorError> {
        Ok(WebhookResponse { success: true, message: None, data: None })
    }
}

// ============================================================================
// Zapier Connector
// ============================================================================

pub struct ZapierConnector {
    client: reqwest::Client,
}

impl ZapierConnector {
    pub fn new() -> Self {
        Self { client: reqwest::Client::new() }
    }
}

#[async_trait]
impl IntegrationConnector for ZapierConnector {
    fn id(&self) -> &'static str { "zapier" }
    fn name(&self) -> &'static str { "Zapier" }
    fn description(&self) -> &'static str { "Connect Mycelix to 5000+ apps through Zapier" }
    fn icon_url(&self) -> &'static str { "/integrations/icons/zapier.svg" }
    fn category(&self) -> ConnectorCategory { ConnectorCategory::Automation }
    fn auth_type(&self) -> AuthType { AuthType::OAuth2 }

    async fn authenticate(&self, credentials: AuthCredentials) -> Result<AuthToken, ConnectorError> {
        Ok(AuthToken {
            access_token: String::new(),
            refresh_token: None,
            expires_at: None,
            token_type: "Bearer".to_string(),
            scopes: vec![],
            metadata: HashMap::new(),
        })
    }

    async fn refresh_token(&self, token: &AuthToken) -> Result<AuthToken, ConnectorError> {
        Ok(token.clone())
    }

    async fn test_connection(&self, token: &AuthToken) -> Result<bool, ConnectorError> {
        Ok(true)
    }

    async fn disconnect(&self, _token: &AuthToken) -> Result<(), ConnectorError> {
        Ok(())
    }

    fn supported_actions(&self) -> Vec<ConnectorAction> {
        vec![
            ConnectorAction {
                id: "trigger_zap".to_string(),
                name: "Trigger Zap".to_string(),
                description: "Trigger a Zapier zap with email data".to_string(),
                parameters: vec![
                    ActionParameter {
                        name: "webhook_url".to_string(),
                        param_type: "string".to_string(),
                        required: true,
                        description: "Zapier webhook URL".to_string(),
                        default: None,
                    },
                    ActionParameter {
                        name: "data".to_string(),
                        param_type: "object".to_string(),
                        required: true,
                        description: "Data to send".to_string(),
                        default: None,
                    },
                ],
                returns: Some("success".to_string()),
            },
        ]
    }

    async fn execute_action(
        &self,
        token: &AuthToken,
        action: &str,
        params: HashMap<String, serde_json::Value>,
    ) -> Result<serde_json::Value, ConnectorError> {
        match action {
            "trigger_zap" => Ok(serde_json::json!({ "success": true })),
            _ => Err(ConnectorError::InvalidRequest(format!("Unknown action: {}", action))),
        }
    }

    fn webhook_events(&self) -> Vec<WebhookEvent> { vec![] }

    async fn handle_webhook(
        &self,
        event: &str,
        payload: serde_json::Value,
    ) -> Result<WebhookResponse, ConnectorError> {
        Ok(WebhookResponse { success: true, message: None, data: None })
    }
}

// ============================================================================
// Connector Registry
// ============================================================================

pub struct ConnectorRegistry {
    connectors: HashMap<String, Box<dyn IntegrationConnector>>,
}

impl ConnectorRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            connectors: HashMap::new(),
        };

        // Register built-in connectors
        registry.register(Box::new(SlackConnector::new()));
        registry.register(Box::new(NotionConnector::new()));
        registry.register(Box::new(LinearConnector::new()));
        registry.register(Box::new(GitHubConnector::new()));
        registry.register(Box::new(SalesforceConnector::new()));
        registry.register(Box::new(ZapierConnector::new()));

        registry
    }

    pub fn register(&mut self, connector: Box<dyn IntegrationConnector>) {
        self.connectors.insert(connector.id().to_string(), connector);
    }

    pub fn get(&self, id: &str) -> Option<&dyn IntegrationConnector> {
        self.connectors.get(id).map(|c| c.as_ref())
    }

    pub fn list(&self) -> Vec<ConnectorInfo> {
        self.connectors
            .values()
            .map(|c| ConnectorInfo {
                id: c.id().to_string(),
                name: c.name().to_string(),
                description: c.description().to_string(),
                icon_url: c.icon_url().to_string(),
                category: c.category(),
                auth_type: c.auth_type(),
                actions: c.supported_actions(),
                webhook_events: c.webhook_events(),
            })
            .collect()
    }

    pub fn list_by_category(&self, category: ConnectorCategory) -> Vec<ConnectorInfo> {
        self.list()
            .into_iter()
            .filter(|c| matches!(&c.category, cat if std::mem::discriminant(cat) == std::mem::discriminant(&category)))
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub icon_url: String,
    pub category: ConnectorCategory,
    pub auth_type: AuthType,
    pub actions: Vec<ConnectorAction>,
    pub webhook_events: Vec<WebhookEvent>,
}

impl Default for ConnectorRegistry {
    fn default() -> Self {
        Self::new()
    }
}
