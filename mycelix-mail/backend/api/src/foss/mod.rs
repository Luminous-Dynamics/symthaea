// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Track Y: FOSS Integration Hub
//!
//! Deep integrations with open-source tools: Nextcloud, Matrix, Jitsi,
//! Vikunja, Keycloak, n8n, Meilisearch, Gitea, Joplin, Bitwarden, CalDAV/CardDAV.

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;
use std::collections::HashMap;

// ============================================================================
// Integration Registry
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Integration {
    pub id: Uuid,
    pub user_id: Uuid,
    pub integration_type: IntegrationType,
    pub name: String,
    pub config: IntegrationConfig,
    pub status: IntegrationStatus,
    pub last_sync: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntegrationType {
    Nextcloud,
    Matrix,
    Jitsi,
    Vikunja,
    OpenProject,
    Keycloak,
    Authentik,
    N8n,
    Huginn,
    Meilisearch,
    Typesense,
    Gitea,
    Forgejo,
    GitLab,
    Joplin,
    StandardNotes,
    Bitwarden,
    Vaultwarden,
    CalDAV,
    CardDAV,
    XMPP,
    LDAP,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub username: Option<String>,
    pub password_ref: Option<String>,  // Reference to secret in vault
    pub oauth_token: Option<String>,
    pub refresh_token: Option<String>,
    pub extra: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntegrationStatus {
    Active,
    Inactive,
    Error { message: String },
    Syncing,
    NeedsReauth,
}

pub struct IntegrationRegistry {
    pool: PgPool,
}

impl IntegrationRegistry {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn register(&self, integration: Integration) -> Result<Integration> {
        sqlx::query!(
            r#"
            INSERT INTO integrations (id, user_id, integration_type, name, config, status, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, NOW())
            "#,
            integration.id,
            integration.user_id,
            serde_json::to_string(&integration.integration_type)?,
            integration.name,
            serde_json::to_value(&integration.config)?,
            serde_json::to_string(&integration.status)?
        )
        .execute(&self.pool)
        .await?;

        Ok(integration)
    }

    pub async fn get(&self, integration_id: Uuid) -> Result<Integration> {
        let row = sqlx::query!(
            "SELECT * FROM integrations WHERE id = $1",
            integration_id
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(Integration {
            id: row.id,
            user_id: row.user_id,
            integration_type: serde_json::from_str(&row.integration_type)?,
            name: row.name,
            config: serde_json::from_value(row.config)?,
            status: serde_json::from_str(&row.status)?,
            last_sync: row.last_sync,
            created_at: row.created_at,
        })
    }

    pub async fn list_for_user(&self, user_id: Uuid) -> Result<Vec<Integration>> {
        let rows = sqlx::query!(
            "SELECT * FROM integrations WHERE user_id = $1 ORDER BY name",
            user_id
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().map(|r| Integration {
            id: r.id,
            user_id: r.user_id,
            integration_type: serde_json::from_str(&r.integration_type).unwrap_or(IntegrationType::Nextcloud),
            name: r.name,
            config: serde_json::from_value(r.config).unwrap_or_default(),
            status: serde_json::from_str(&r.status).unwrap_or(IntegrationStatus::Inactive),
            last_sync: r.last_sync,
            created_at: r.created_at,
        }).collect())
    }

    pub async fn update_status(&self, integration_id: Uuid, status: IntegrationStatus) -> Result<()> {
        sqlx::query!(
            "UPDATE integrations SET status = $1, last_sync = NOW() WHERE id = $2",
            serde_json::to_string(&status)?,
            integration_id
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }
}

// ============================================================================
// Nextcloud Integration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NextcloudConfig {
    pub base_url: String,
    pub username: String,
    pub app_password: String,
    pub sync_calendar: bool,
    pub sync_contacts: bool,
    pub sync_files: bool,
    pub sync_tasks: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NextcloudCalendar {
    pub id: String,
    pub display_name: String,
    pub color: String,
    pub ctag: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NextcloudEvent {
    pub uid: String,
    pub calendar_id: String,
    pub summary: String,
    pub description: Option<String>,
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    pub location: Option<String>,
    pub attendees: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NextcloudContact {
    pub uid: String,
    pub full_name: String,
    pub emails: Vec<String>,
    pub phones: Vec<String>,
    pub organization: Option<String>,
    pub photo_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NextcloudTask {
    pub uid: String,
    pub list_id: String,
    pub summary: String,
    pub description: Option<String>,
    pub due: Option<DateTime<Utc>>,
    pub priority: u8,
    pub completed: bool,
}

pub struct NextcloudIntegration {
    config: NextcloudConfig,
    client: reqwest::Client,
}

impl NextcloudIntegration {
    pub fn new(config: NextcloudConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    pub async fn test_connection(&self) -> Result<bool> {
        let response = self.client
            .get(format!("{}/ocs/v2.php/cloud/user", self.config.base_url))
            .basic_auth(&self.config.username, Some(&self.config.app_password))
            .header("OCS-APIRequest", "true")
            .send()
            .await?;

        Ok(response.status().is_success())
    }

    pub async fn get_calendars(&self) -> Result<Vec<NextcloudCalendar>> {
        let dav_url = format!("{}/remote.php/dav/calendars/{}/", self.config.base_url, self.config.username);

        let response = self.client
            .request(reqwest::Method::from_bytes(b"PROPFIND")?, &dav_url)
            .basic_auth(&self.config.username, Some(&self.config.app_password))
            .header("Depth", "1")
            .body(r#"<?xml version="1.0"?>
                <d:propfind xmlns:d="DAV:" xmlns:cs="http://calendarserver.org/ns/" xmlns:c="urn:ietf:params:xml:ns:caldav">
                    <d:prop>
                        <d:displayname/>
                        <cs:getctag/>
                        <x1:calendar-color xmlns:x1="http://apple.com/ns/ical/"/>
                    </d:prop>
                </d:propfind>"#)
            .send()
            .await?;

        // Parse WebDAV response
        let _body = response.text().await?;

        // Would parse XML properly
        Ok(vec![])
    }

    pub async fn create_event_from_email(&self, email_id: Uuid, calendar_id: &str, event: NextcloudEvent) -> Result<String> {
        let ical = self.event_to_ical(&event);

        let event_url = format!(
            "{}/remote.php/dav/calendars/{}/{}/{}.ics",
            self.config.base_url,
            self.config.username,
            calendar_id,
            event.uid
        );

        self.client
            .put(&event_url)
            .basic_auth(&self.config.username, Some(&self.config.app_password))
            .header("Content-Type", "text/calendar")
            .body(ical)
            .send()
            .await?;

        Ok(event.uid)
    }

    fn event_to_ical(&self, event: &NextcloudEvent) -> String {
        format!(
            r#"BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Mycelix-Mail//EN
BEGIN:VEVENT
UID:{}
DTSTAMP:{}
DTSTART:{}
DTEND:{}
SUMMARY:{}
DESCRIPTION:{}
LOCATION:{}
END:VEVENT
END:VCALENDAR"#,
            event.uid,
            Utc::now().format("%Y%m%dT%H%M%SZ"),
            event.start.format("%Y%m%dT%H%M%SZ"),
            event.end.format("%Y%m%dT%H%M%SZ"),
            event.summary,
            event.description.as_deref().unwrap_or(""),
            event.location.as_deref().unwrap_or("")
        )
    }

    pub async fn get_contacts(&self) -> Result<Vec<NextcloudContact>> {
        let dav_url = format!("{}/remote.php/dav/addressbooks/users/{}/contacts/", self.config.base_url, self.config.username);

        let response = self.client
            .request(reqwest::Method::from_bytes(b"PROPFIND")?, &dav_url)
            .basic_auth(&self.config.username, Some(&self.config.app_password))
            .header("Depth", "1")
            .send()
            .await?;

        let _body = response.text().await?;

        // Would parse vCard responses
        Ok(vec![])
    }

    pub async fn save_email_as_note(&self, subject: &str, body: &str, folder: &str) -> Result<String> {
        let notes_url = format!(
            "{}/remote.php/webdav/{}/{}.md",
            self.config.base_url,
            folder,
            slugify(subject)
        );

        self.client
            .put(&notes_url)
            .basic_auth(&self.config.username, Some(&self.config.app_password))
            .header("Content-Type", "text/markdown")
            .body(format!("# {}\n\n{}", subject, body))
            .send()
            .await?;

        Ok(notes_url)
    }
}

fn slugify(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

// ============================================================================
// Matrix Integration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixConfig {
    pub homeserver: String,
    pub user_id: String,
    pub access_token: String,
    pub notification_room: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixMessage {
    pub room_id: String,
    pub event_id: Option<String>,
    pub body: String,
    pub formatted_body: Option<String>,
    pub msg_type: String,
}

pub struct MatrixIntegration {
    config: MatrixConfig,
    client: reqwest::Client,
}

impl MatrixIntegration {
    pub fn new(config: MatrixConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    pub async fn send_message(&self, room_id: &str, message: &str) -> Result<String> {
        let txn_id = Uuid::new_v4().to_string();
        let url = format!(
            "{}/_matrix/client/v3/rooms/{}/send/m.room.message/{}",
            self.config.homeserver, room_id, txn_id
        );

        let response = self.client
            .put(&url)
            .header("Authorization", format!("Bearer {}", self.config.access_token))
            .json(&serde_json::json!({
                "msgtype": "m.text",
                "body": message
            }))
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        Ok(response["event_id"].as_str().unwrap_or("").to_string())
    }

    pub async fn send_email_notification(&self, room_id: &str, subject: &str, from: &str, preview: &str) -> Result<String> {
        let formatted = format!(
            r#"<h4>📧 New Email</h4>
<p><strong>From:</strong> {}</p>
<p><strong>Subject:</strong> {}</p>
<blockquote>{}</blockquote>"#,
            from, subject, preview
        );

        let plain = format!("📧 New Email\nFrom: {}\nSubject: {}\n\n{}", from, subject, preview);

        let txn_id = Uuid::new_v4().to_string();
        let url = format!(
            "{}/_matrix/client/v3/rooms/{}/send/m.room.message/{}",
            self.config.homeserver, room_id, txn_id
        );

        let response = self.client
            .put(&url)
            .header("Authorization", format!("Bearer {}", self.config.access_token))
            .json(&serde_json::json!({
                "msgtype": "m.text",
                "body": plain,
                "format": "org.matrix.custom.html",
                "formatted_body": formatted
            }))
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        Ok(response["event_id"].as_str().unwrap_or("").to_string())
    }

    pub async fn get_rooms(&self) -> Result<Vec<MatrixRoom>> {
        let url = format!("{}/_matrix/client/v3/joined_rooms", self.config.homeserver);

        let response = self.client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.config.access_token))
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        let room_ids: Vec<String> = response["joined_rooms"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();

        let mut rooms = Vec::new();
        for room_id in room_ids {
            if let Ok(room) = self.get_room_info(&room_id).await {
                rooms.push(room);
            }
        }

        Ok(rooms)
    }

    async fn get_room_info(&self, room_id: &str) -> Result<MatrixRoom> {
        let url = format!(
            "{}/_matrix/client/v3/rooms/{}/state/m.room.name",
            self.config.homeserver, room_id
        );

        let response = self.client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.config.access_token))
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        Ok(MatrixRoom {
            room_id: room_id.to_string(),
            name: response["name"].as_str().map(String::from),
            topic: None,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixRoom {
    pub room_id: String,
    pub name: Option<String>,
    pub topic: Option<String>,
}

// ============================================================================
// Jitsi Integration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JitsiConfig {
    pub server_url: String,
    pub jwt_app_id: Option<String>,
    pub jwt_secret: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JitsiMeeting {
    pub room_name: String,
    pub display_name: String,
    pub url: String,
    pub jwt: Option<String>,
}

pub struct JitsiIntegration {
    config: JitsiConfig,
}

impl JitsiIntegration {
    pub fn new(config: JitsiConfig) -> Self {
        Self { config }
    }

    pub fn create_meeting_from_email(&self, subject: &str, user_name: &str) -> JitsiMeeting {
        let room_name = format!("{}-{}", slugify(subject), Uuid::new_v4().to_string()[..8].to_string());

        let url = format!("{}/{}", self.config.server_url, room_name);

        let jwt = if self.config.jwt_app_id.is_some() && self.config.jwt_secret.is_some() {
            Some(self.generate_jwt(&room_name, user_name))
        } else {
            None
        };

        JitsiMeeting {
            room_name: room_name.clone(),
            display_name: user_name.to_string(),
            url: if let Some(ref token) = jwt {
                format!("{}?jwt={}", url, token)
            } else {
                url
            },
            jwt,
        }
    }

    fn generate_jwt(&self, room_name: &str, user_name: &str) -> String {
        // Would use jsonwebtoken crate
        // Simplified for now
        format!("jwt_token_for_{}_{}", room_name, user_name)
    }
}

// ============================================================================
// Vikunja Integration (Task Management)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VikunjaConfig {
    pub base_url: String,
    pub api_token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VikunjaTask {
    pub id: Option<i64>,
    pub title: String,
    pub description: Option<String>,
    pub done: bool,
    pub due_date: Option<DateTime<Utc>>,
    pub priority: i32,
    pub project_id: i64,
    pub labels: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VikunjaProject {
    pub id: i64,
    pub title: String,
    pub description: Option<String>,
}

pub struct VikunjaIntegration {
    config: VikunjaConfig,
    client: reqwest::Client,
}

impl VikunjaIntegration {
    pub fn new(config: VikunjaConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    pub async fn get_projects(&self) -> Result<Vec<VikunjaProject>> {
        let response = self.client
            .get(format!("{}/api/v1/projects", self.config.base_url))
            .header("Authorization", format!("Bearer {}", self.config.api_token))
            .send()
            .await?
            .json::<Vec<VikunjaProject>>()
            .await?;

        Ok(response)
    }

    pub async fn create_task(&self, task: VikunjaTask) -> Result<VikunjaTask> {
        let response = self.client
            .put(format!("{}/api/v1/projects/{}/tasks", self.config.base_url, task.project_id))
            .header("Authorization", format!("Bearer {}", self.config.api_token))
            .json(&task)
            .send()
            .await?
            .json::<VikunjaTask>()
            .await?;

        Ok(response)
    }

    pub async fn create_task_from_email(&self, project_id: i64, subject: &str, body: &str, due: Option<DateTime<Utc>>) -> Result<VikunjaTask> {
        let task = VikunjaTask {
            id: None,
            title: subject.to_string(),
            description: Some(body.chars().take(1000).collect()),
            done: false,
            due_date: due,
            priority: 0,
            project_id,
            labels: vec!["from-email".to_string()],
        };

        self.create_task(task).await
    }
}

// ============================================================================
// Keycloak/Authentik SSO Integration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OIDCConfig {
    pub provider: OIDCProvider,
    pub client_id: String,
    pub client_secret: String,
    pub issuer_url: String,
    pub redirect_uri: String,
    pub scopes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OIDCProvider {
    Keycloak,
    Authentik,
    Generic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OIDCToken {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: u64,
    pub refresh_token: Option<String>,
    pub id_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OIDCUserInfo {
    pub sub: String,
    pub email: Option<String>,
    pub name: Option<String>,
    pub preferred_username: Option<String>,
    pub groups: Option<Vec<String>>,
}

pub struct OIDCIntegration {
    config: OIDCConfig,
    client: reqwest::Client,
}

impl OIDCIntegration {
    pub fn new(config: OIDCConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    pub fn get_authorization_url(&self, state: &str) -> String {
        let scopes = self.config.scopes.join(" ");
        format!(
            "{}/protocol/openid-connect/auth?client_id={}&redirect_uri={}&response_type=code&scope={}&state={}",
            self.config.issuer_url,
            self.config.client_id,
            urlencoding::encode(&self.config.redirect_uri),
            urlencoding::encode(&scopes),
            state
        )
    }

    pub async fn exchange_code(&self, code: &str) -> Result<OIDCToken> {
        let response = self.client
            .post(format!("{}/protocol/openid-connect/token", self.config.issuer_url))
            .form(&[
                ("grant_type", "authorization_code"),
                ("client_id", &self.config.client_id),
                ("client_secret", &self.config.client_secret),
                ("code", code),
                ("redirect_uri", &self.config.redirect_uri),
            ])
            .send()
            .await?
            .json::<OIDCToken>()
            .await?;

        Ok(response)
    }

    pub async fn get_user_info(&self, access_token: &str) -> Result<OIDCUserInfo> {
        let response = self.client
            .get(format!("{}/protocol/openid-connect/userinfo", self.config.issuer_url))
            .header("Authorization", format!("Bearer {}", access_token))
            .send()
            .await?
            .json::<OIDCUserInfo>()
            .await?;

        Ok(response)
    }

    pub async fn refresh_token(&self, refresh_token: &str) -> Result<OIDCToken> {
        let response = self.client
            .post(format!("{}/protocol/openid-connect/token", self.config.issuer_url))
            .form(&[
                ("grant_type", "refresh_token"),
                ("client_id", &self.config.client_id),
                ("client_secret", &self.config.client_secret),
                ("refresh_token", refresh_token),
            ])
            .send()
            .await?
            .json::<OIDCToken>()
            .await?;

        Ok(response)
    }
}

// ============================================================================
// n8n/Huginn Automation Integration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct N8nConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub webhook_secret: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct N8nWebhook {
    pub id: Uuid,
    pub name: String,
    pub webhook_url: String,
    pub trigger_events: Vec<EmailEvent>,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmailEvent {
    Received,
    Sent,
    Read,
    Replied,
    Forwarded,
    Archived,
    Deleted,
    LabelAdded,
    LabelRemoved,
}

pub struct N8nIntegration {
    config: N8nConfig,
    client: reqwest::Client,
}

impl N8nIntegration {
    pub fn new(config: N8nConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    pub async fn trigger_webhook(&self, webhook_url: &str, payload: serde_json::Value) -> Result<()> {
        let signature = self.sign_payload(&payload)?;

        self.client
            .post(webhook_url)
            .header("X-Webhook-Signature", signature)
            .json(&payload)
            .send()
            .await?;

        Ok(())
    }

    fn sign_payload(&self, payload: &serde_json::Value) -> Result<String> {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;

        type HmacSha256 = Hmac<Sha256>;

        let mut mac = HmacSha256::new_from_slice(self.config.webhook_secret.as_bytes())?;
        mac.update(serde_json::to_string(payload)?.as_bytes());

        let result = mac.finalize();
        Ok(hex::encode(result.into_bytes()))
    }

    pub async fn send_email_event(&self, webhook: &N8nWebhook, event: EmailEvent, email_data: serde_json::Value) -> Result<()> {
        if !webhook.active || !webhook.trigger_events.contains(&event) {
            return Ok(());
        }

        let payload = serde_json::json!({
            "event": serde_json::to_string(&event)?,
            "timestamp": Utc::now().to_rfc3339(),
            "email": email_data,
        });

        self.trigger_webhook(&webhook.webhook_url, payload).await
    }
}

// ============================================================================
// Meilisearch/Typesense Integration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeilisearchConfig {
    pub host: String,
    pub api_key: Option<String>,
    pub index_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchDocument {
    pub id: String,
    pub user_id: String,
    pub subject: String,
    pub body: String,
    pub sender: String,
    pub recipients: Vec<String>,
    pub folder: String,
    pub labels: Vec<String>,
    pub received_at: i64,
    pub has_attachments: bool,
}

pub struct MeilisearchIntegration {
    config: MeilisearchConfig,
    client: reqwest::Client,
}

impl MeilisearchIntegration {
    pub fn new(config: MeilisearchConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    pub async fn index_email(&self, doc: SearchDocument) -> Result<()> {
        let mut request = self.client
            .post(format!("{}/indexes/{}/documents", self.config.host, self.config.index_name));

        if let Some(ref key) = self.config.api_key {
            request = request.header("Authorization", format!("Bearer {}", key));
        }

        request.json(&vec![doc]).send().await?;

        Ok(())
    }

    pub async fn search(&self, query: &str, user_id: &str, limit: u32) -> Result<Vec<SearchDocument>> {
        let mut request = self.client
            .post(format!("{}/indexes/{}/search", self.config.host, self.config.index_name));

        if let Some(ref key) = self.config.api_key {
            request = request.header("Authorization", format!("Bearer {}", key));
        }

        let response = request
            .json(&serde_json::json!({
                "q": query,
                "filter": format!("user_id = '{}'", user_id),
                "limit": limit,
            }))
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        let hits: Vec<SearchDocument> = response["hits"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|h| serde_json::from_value(h.clone()).ok())
            .collect();

        Ok(hits)
    }

    pub async fn delete_email(&self, email_id: &str) -> Result<()> {
        let mut request = self.client
            .delete(format!("{}/indexes/{}/documents/{}", self.config.host, self.config.index_name, email_id));

        if let Some(ref key) = self.config.api_key {
            request = request.header("Authorization", format!("Bearer {}", key));
        }

        request.send().await?;

        Ok(())
    }
}

// ============================================================================
// Gitea/Forgejo Integration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GiteaConfig {
    pub base_url: String,
    pub access_token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GiteaIssue {
    pub id: Option<i64>,
    pub title: String,
    pub body: String,
    pub state: String,
    pub labels: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GiteaNotification {
    pub id: i64,
    pub repository: String,
    pub subject_type: String,
    pub subject_title: String,
    pub updated_at: DateTime<Utc>,
}

pub struct GiteaIntegration {
    config: GiteaConfig,
    client: reqwest::Client,
}

impl GiteaIntegration {
    pub fn new(config: GiteaConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    pub async fn get_notifications(&self) -> Result<Vec<GiteaNotification>> {
        let response = self.client
            .get(format!("{}/api/v1/notifications", self.config.base_url))
            .header("Authorization", format!("token {}", self.config.access_token))
            .send()
            .await?
            .json::<Vec<GiteaNotification>>()
            .await?;

        Ok(response)
    }

    pub async fn create_issue(&self, owner: &str, repo: &str, issue: GiteaIssue) -> Result<GiteaIssue> {
        let response = self.client
            .post(format!("{}/api/v1/repos/{}/{}/issues", self.config.base_url, owner, repo))
            .header("Authorization", format!("token {}", self.config.access_token))
            .json(&issue)
            .send()
            .await?
            .json::<GiteaIssue>()
            .await?;

        Ok(response)
    }

    pub async fn create_issue_from_email(&self, owner: &str, repo: &str, subject: &str, body: &str) -> Result<GiteaIssue> {
        let issue = GiteaIssue {
            id: None,
            title: subject.to_string(),
            body: format!("Created from email:\n\n{}", body),
            state: "open".to_string(),
            labels: vec!["from-email".to_string()],
        };

        self.create_issue(owner, repo, issue).await
    }
}

// ============================================================================
// Joplin/Standard Notes Integration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoplinConfig {
    pub base_url: String,  // Usually http://localhost:41184
    pub token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoplinNote {
    pub id: Option<String>,
    pub parent_id: String,
    pub title: String,
    pub body: String,
    pub is_todo: bool,
    pub todo_completed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoplinFolder {
    pub id: String,
    pub title: String,
    pub parent_id: Option<String>,
}

pub struct JoplinIntegration {
    config: JoplinConfig,
    client: reqwest::Client,
}

impl JoplinIntegration {
    pub fn new(config: JoplinConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    pub async fn get_folders(&self) -> Result<Vec<JoplinFolder>> {
        let response = self.client
            .get(format!("{}/folders?token={}", self.config.base_url, self.config.token))
            .send()
            .await?
            .json::<JoplinResponse<JoplinFolder>>()
            .await?;

        Ok(response.items)
    }

    pub async fn create_note(&self, note: JoplinNote) -> Result<JoplinNote> {
        let response = self.client
            .post(format!("{}/notes?token={}", self.config.base_url, self.config.token))
            .json(&note)
            .send()
            .await?
            .json::<JoplinNote>()
            .await?;

        Ok(response)
    }

    pub async fn save_email_as_note(&self, folder_id: &str, subject: &str, body: &str, as_todo: bool) -> Result<JoplinNote> {
        let note = JoplinNote {
            id: None,
            parent_id: folder_id.to_string(),
            title: subject.to_string(),
            body: body.to_string(),
            is_todo: as_todo,
            todo_completed: false,
        };

        self.create_note(note).await
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JoplinResponse<T> {
    items: Vec<T>,
    has_more: bool,
}

// ============================================================================
// CalDAV/CardDAV (Generic)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DAVConfig {
    pub calendar_url: Option<String>,
    pub contacts_url: Option<String>,
    pub username: String,
    pub password: String,
}

pub struct DAVIntegration {
    config: DAVConfig,
    client: reqwest::Client,
}

impl DAVIntegration {
    pub fn new(config: DAVConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    pub async fn sync_contacts(&self) -> Result<Vec<NextcloudContact>> {
        if let Some(ref url) = self.config.contacts_url {
            let response = self.client
                .request(reqwest::Method::from_bytes(b"PROPFIND")?, url)
                .basic_auth(&self.config.username, Some(&self.config.password))
                .header("Depth", "1")
                .header("Content-Type", "application/xml")
                .body(r#"<?xml version="1.0"?>
                    <d:propfind xmlns:d="DAV:" xmlns:card="urn:ietf:params:xml:ns:carddav">
                        <d:prop>
                            <d:getetag/>
                            <card:address-data/>
                        </d:prop>
                    </d:propfind>"#)
                .send()
                .await?;

            let _body = response.text().await?;
            // Would parse vCard data
        }

        Ok(vec![])
    }

    pub async fn sync_calendar(&self) -> Result<Vec<NextcloudEvent>> {
        if let Some(ref url) = self.config.calendar_url {
            let response = self.client
                .request(reqwest::Method::from_bytes(b"PROPFIND")?, url)
                .basic_auth(&self.config.username, Some(&self.config.password))
                .header("Depth", "1")
                .header("Content-Type", "application/xml")
                .body(r#"<?xml version="1.0"?>
                    <d:propfind xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">
                        <d:prop>
                            <d:getetag/>
                            <c:calendar-data/>
                        </d:prop>
                    </d:propfind>"#)
                .send()
                .await?;

            let _body = response.text().await?;
            // Would parse iCal data
        }

        Ok(vec![])
    }
}

// ============================================================================
// Bitwarden/Vaultwarden Integration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitwardenConfig {
    pub server_url: String,
    pub client_id: String,
    pub client_secret: String,
}

pub struct BitwardenIntegration {
    config: BitwardenConfig,
    client: reqwest::Client,
    access_token: Option<String>,
}

impl BitwardenIntegration {
    pub fn new(config: BitwardenConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
            access_token: None,
        }
    }

    pub async fn authenticate(&mut self) -> Result<()> {
        let response = self.client
            .post(format!("{}/identity/connect/token", self.config.server_url))
            .form(&[
                ("grant_type", "client_credentials"),
                ("client_id", &self.config.client_id),
                ("client_secret", &self.config.client_secret),
                ("scope", "api"),
            ])
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        self.access_token = response["access_token"].as_str().map(String::from);
        Ok(())
    }

    pub async fn get_secret(&self, secret_id: &str) -> Result<String> {
        let token = self.access_token.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Not authenticated"))?;

        let response = self.client
            .get(format!("{}/api/secrets/{}", self.config.server_url, secret_id))
            .header("Authorization", format!("Bearer {}", token))
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        Ok(response["value"].as_str().unwrap_or("").to_string())
    }
}

// ============================================================================
// FOSS Integration Service (Main Orchestrator)
// ============================================================================

pub struct FossIntegrationService {
    pool: PgPool,
    registry: IntegrationRegistry,
}

impl FossIntegrationService {
    pub fn new(pool: PgPool) -> Self {
        Self {
            registry: IntegrationRegistry::new(pool.clone()),
            pool,
        }
    }

    pub async fn notify_email_received(&self, user_id: Uuid, email_data: serde_json::Value) -> Result<()> {
        let integrations = self.registry.list_for_user(user_id).await?;

        for integration in integrations {
            if !matches!(integration.status, IntegrationStatus::Active) {
                continue;
            }

            match integration.integration_type {
                IntegrationType::Matrix => {
                    if let Ok(config) = serde_json::from_value::<MatrixConfig>(serde_json::to_value(&integration.config)?) {
                        let matrix = MatrixIntegration::new(config.clone());
                        if let Some(room_id) = &config.notification_room {
                            let subject = email_data["subject"].as_str().unwrap_or("(no subject)");
                            let from = email_data["from"].as_str().unwrap_or("unknown");
                            let preview = email_data["preview"].as_str().unwrap_or("");
                            let _ = matrix.send_email_notification(room_id, subject, from, preview).await;
                        }
                    }
                }
                IntegrationType::N8n | IntegrationType::Huginn => {
                    // Trigger automation webhooks
                    if let Ok(config) = serde_json::from_value::<N8nConfig>(serde_json::to_value(&integration.config)?) {
                        let n8n = N8nIntegration::new(config);
                        // Would look up registered webhooks and trigger them
                    }
                }
                IntegrationType::Meilisearch | IntegrationType::Typesense => {
                    // Index email for search
                    if let Ok(config) = serde_json::from_value::<MeilisearchConfig>(serde_json::to_value(&integration.config)?) {
                        let meili = MeilisearchIntegration::new(config);
                        let doc = SearchDocument {
                            id: email_data["id"].as_str().unwrap_or("").to_string(),
                            user_id: user_id.to_string(),
                            subject: email_data["subject"].as_str().unwrap_or("").to_string(),
                            body: email_data["body"].as_str().unwrap_or("").to_string(),
                            sender: email_data["from"].as_str().unwrap_or("").to_string(),
                            recipients: vec![],
                            folder: "Inbox".to_string(),
                            labels: vec![],
                            received_at: Utc::now().timestamp(),
                            has_attachments: false,
                        };
                        let _ = meili.index_email(doc).await;
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }

    pub async fn create_meeting_from_email(&self, user_id: Uuid, email_id: Uuid) -> Result<JitsiMeeting> {
        let email = sqlx::query!(
            "SELECT subject, sender FROM emails WHERE id = $1 AND user_id = $2",
            email_id,
            user_id
        )
        .fetch_one(&self.pool)
        .await?;

        let integrations = self.registry.list_for_user(user_id).await?;

        let jitsi_integration = integrations.iter()
            .find(|i| matches!(i.integration_type, IntegrationType::Jitsi));

        if let Some(integration) = jitsi_integration {
            let config: JitsiConfig = serde_json::from_value(serde_json::to_value(&integration.config)?)?;
            let jitsi = JitsiIntegration::new(config);
            let meeting = jitsi.create_meeting_from_email(
                &email.subject.unwrap_or_default(),
                &email.sender
            );
            return Ok(meeting);
        }

        // Fallback to default Jitsi server
        let jitsi = JitsiIntegration::new(JitsiConfig {
            server_url: "https://meet.jit.si".to_string(),
            jwt_app_id: None,
            jwt_secret: None,
        });

        Ok(jitsi.create_meeting_from_email(
            &email.subject.unwrap_or_default(),
            &email.sender
        ))
    }

    pub async fn create_task_from_email(&self, user_id: Uuid, email_id: Uuid, project_id: i64) -> Result<VikunjaTask> {
        let email = sqlx::query!(
            "SELECT subject, body FROM emails WHERE id = $1 AND user_id = $2",
            email_id,
            user_id
        )
        .fetch_one(&self.pool)
        .await?;

        let integrations = self.registry.list_for_user(user_id).await?;

        let vikunja_integration = integrations.iter()
            .find(|i| matches!(i.integration_type, IntegrationType::Vikunja));

        if let Some(integration) = vikunja_integration {
            let config: VikunjaConfig = serde_json::from_value(serde_json::to_value(&integration.config)?)?;
            let vikunja = VikunjaIntegration::new(config);
            return vikunja.create_task_from_email(
                project_id,
                &email.subject.unwrap_or_default(),
                &email.body,
                None
            ).await;
        }

        Err(anyhow::anyhow!("No Vikunja integration configured"))
    }

    pub async fn save_email_to_notes(&self, user_id: Uuid, email_id: Uuid, folder_id: &str) -> Result<String> {
        let email = sqlx::query!(
            "SELECT subject, body FROM emails WHERE id = $1 AND user_id = $2",
            email_id,
            user_id
        )
        .fetch_one(&self.pool)
        .await?;

        let integrations = self.registry.list_for_user(user_id).await?;

        // Try Joplin first
        if let Some(integration) = integrations.iter().find(|i| matches!(i.integration_type, IntegrationType::Joplin)) {
            let config: JoplinConfig = serde_json::from_value(serde_json::to_value(&integration.config)?)?;
            let joplin = JoplinIntegration::new(config);
            let note = joplin.save_email_as_note(
                folder_id,
                &email.subject.unwrap_or_default(),
                &email.body,
                false
            ).await?;
            return Ok(note.id.unwrap_or_default());
        }

        // Try Nextcloud
        if let Some(integration) = integrations.iter().find(|i| matches!(i.integration_type, IntegrationType::Nextcloud)) {
            let config: NextcloudConfig = serde_json::from_value(serde_json::to_value(&integration.config)?)?;
            let nextcloud = NextcloudIntegration::new(config);
            return nextcloud.save_email_as_note(
                &email.subject.unwrap_or_default(),
                &email.body,
                folder_id
            ).await;
        }

        Err(anyhow::anyhow!("No notes integration configured"))
    }
}
