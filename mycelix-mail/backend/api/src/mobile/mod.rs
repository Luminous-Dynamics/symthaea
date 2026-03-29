// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Track W: Offline & Mobile Excellence
//!
//! Progressive Web App support, offline capability, smart prefetch,
//! mobile gestures, wearable support, and cross-device continuity.

use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;
use std::collections::HashMap;

// ============================================================================
// PWA & Offline Support
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OfflineCapability {
    pub user_id: Uuid,
    pub device_id: String,
    pub sync_status: SyncStatus,
    pub last_sync: DateTime<Utc>,
    pub pending_actions: Vec<PendingAction>,
    pub cached_emails: u32,
    pub cache_size_bytes: u64,
    pub prefetch_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncStatus {
    Synced,
    Syncing,
    Pending,
    Conflict,
    Offline,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingAction {
    pub id: Uuid,
    pub action_type: OfflineActionType,
    pub email_id: Option<Uuid>,
    pub data: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub retry_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OfflineActionType {
    MarkRead,
    MarkUnread,
    Star,
    Unstar,
    Archive,
    Delete,
    Move { folder: String },
    Label { label: String, add: bool },
    Reply { draft_id: Uuid },
    Compose { draft_id: Uuid },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncManifest {
    pub version: u64,
    pub emails: Vec<EmailSyncRecord>,
    pub folders: Vec<FolderSyncRecord>,
    pub labels: Vec<LabelSyncRecord>,
    pub settings: serde_json::Value,
    pub generated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailSyncRecord {
    pub id: Uuid,
    pub version: u64,
    pub folder: String,
    pub is_read: bool,
    pub is_starred: bool,
    pub labels: Vec<String>,
    pub updated_at: DateTime<Utc>,
    pub deleted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FolderSyncRecord {
    pub name: String,
    pub unread_count: u32,
    pub total_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelSyncRecord {
    pub name: String,
    pub color: String,
    pub count: u32,
}

pub struct OfflineSyncService {
    pool: PgPool,
}

impl OfflineSyncService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn get_sync_manifest(&self, user_id: Uuid, since_version: Option<u64>) -> Result<SyncManifest> {
        let current_version = self.get_current_version(user_id).await?;

        let emails = if let Some(since) = since_version {
            // Delta sync - only changed emails
            sqlx::query_as!(
                EmailSyncRow,
                r#"
                SELECT id, version, folder, is_read, is_starred, labels as "labels!: Vec<String>", updated_at, deleted
                FROM emails
                WHERE user_id = $1 AND version > $2
                ORDER BY version ASC
                LIMIT 1000
                "#,
                user_id,
                since as i64
            )
            .fetch_all(&self.pool)
            .await?
        } else {
            // Full sync
            sqlx::query_as!(
                EmailSyncRow,
                r#"
                SELECT id, version, folder, is_read, is_starred, labels as "labels!: Vec<String>", updated_at, deleted
                FROM emails
                WHERE user_id = $1 AND deleted = false
                ORDER BY received_at DESC
                LIMIT 500
                "#,
                user_id
            )
            .fetch_all(&self.pool)
            .await?
        };

        let folders = sqlx::query!(
            r#"
            SELECT
                folder as name,
                COUNT(*) FILTER (WHERE is_read = false) as unread_count,
                COUNT(*) as total_count
            FROM emails
            WHERE user_id = $1 AND deleted = false
            GROUP BY folder
            "#,
            user_id
        )
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|r| FolderSyncRecord {
            name: r.name,
            unread_count: r.unread_count.unwrap_or(0) as u32,
            total_count: r.total_count.unwrap_or(0) as u32,
        })
        .collect();

        let labels = sqlx::query!(
            r#"
            SELECT unnest(labels) as name, COUNT(*) as count
            FROM emails
            WHERE user_id = $1 AND deleted = false
            GROUP BY name
            "#,
            user_id
        )
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .filter_map(|r| r.name.map(|n| LabelSyncRecord {
            name: n,
            color: "#808080".to_string(),
            count: r.count.unwrap_or(0) as u32,
        }))
        .collect();

        Ok(SyncManifest {
            version: current_version,
            emails: emails.into_iter().map(|e| EmailSyncRecord {
                id: e.id,
                version: e.version as u64,
                folder: e.folder,
                is_read: e.is_read,
                is_starred: e.is_starred,
                labels: e.labels,
                updated_at: e.updated_at,
                deleted: e.deleted,
            }).collect(),
            folders,
            labels,
            settings: serde_json::json!({}),
            generated_at: Utc::now(),
        })
    }

    async fn get_current_version(&self, user_id: Uuid) -> Result<u64> {
        let result = sqlx::query!(
            "SELECT COALESCE(MAX(version), 0) as version FROM emails WHERE user_id = $1",
            user_id
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(result.version.unwrap_or(0) as u64)
    }

    pub async fn apply_offline_actions(&self, user_id: Uuid, actions: Vec<PendingAction>) -> Result<Vec<ActionResult>> {
        let mut results = Vec::new();

        for action in actions {
            let result = self.apply_single_action(user_id, &action).await;
            results.push(ActionResult {
                action_id: action.id,
                success: result.is_ok(),
                error: result.err().map(|e| e.to_string()),
                new_version: None,
            });
        }

        Ok(results)
    }

    async fn apply_single_action(&self, user_id: Uuid, action: &PendingAction) -> Result<()> {
        match &action.action_type {
            OfflineActionType::MarkRead => {
                if let Some(email_id) = action.email_id {
                    sqlx::query!(
                        "UPDATE emails SET is_read = true, version = version + 1, updated_at = NOW() WHERE id = $1 AND user_id = $2",
                        email_id,
                        user_id
                    )
                    .execute(&self.pool)
                    .await?;
                }
            }
            OfflineActionType::MarkUnread => {
                if let Some(email_id) = action.email_id {
                    sqlx::query!(
                        "UPDATE emails SET is_read = false, version = version + 1, updated_at = NOW() WHERE id = $1 AND user_id = $2",
                        email_id,
                        user_id
                    )
                    .execute(&self.pool)
                    .await?;
                }
            }
            OfflineActionType::Star => {
                if let Some(email_id) = action.email_id {
                    sqlx::query!(
                        "UPDATE emails SET is_starred = true, version = version + 1, updated_at = NOW() WHERE id = $1 AND user_id = $2",
                        email_id,
                        user_id
                    )
                    .execute(&self.pool)
                    .await?;
                }
            }
            OfflineActionType::Unstar => {
                if let Some(email_id) = action.email_id {
                    sqlx::query!(
                        "UPDATE emails SET is_starred = false, version = version + 1, updated_at = NOW() WHERE id = $1 AND user_id = $2",
                        email_id,
                        user_id
                    )
                    .execute(&self.pool)
                    .await?;
                }
            }
            OfflineActionType::Archive => {
                if let Some(email_id) = action.email_id {
                    sqlx::query!(
                        "UPDATE emails SET folder = 'Archive', version = version + 1, updated_at = NOW() WHERE id = $1 AND user_id = $2",
                        email_id,
                        user_id
                    )
                    .execute(&self.pool)
                    .await?;
                }
            }
            OfflineActionType::Delete => {
                if let Some(email_id) = action.email_id {
                    sqlx::query!(
                        "UPDATE emails SET folder = 'Trash', deleted = true, version = version + 1, updated_at = NOW() WHERE id = $1 AND user_id = $2",
                        email_id,
                        user_id
                    )
                    .execute(&self.pool)
                    .await?;
                }
            }
            OfflineActionType::Move { folder } => {
                if let Some(email_id) = action.email_id {
                    sqlx::query!(
                        "UPDATE emails SET folder = $1, version = version + 1, updated_at = NOW() WHERE id = $2 AND user_id = $3",
                        folder,
                        email_id,
                        user_id
                    )
                    .execute(&self.pool)
                    .await?;
                }
            }
            OfflineActionType::Label { label, add } => {
                if let Some(email_id) = action.email_id {
                    if *add {
                        sqlx::query!(
                            "UPDATE emails SET labels = array_append(labels, $1), version = version + 1, updated_at = NOW() WHERE id = $2 AND user_id = $3",
                            label,
                            email_id,
                            user_id
                        )
                        .execute(&self.pool)
                        .await?;
                    } else {
                        sqlx::query!(
                            "UPDATE emails SET labels = array_remove(labels, $1), version = version + 1, updated_at = NOW() WHERE id = $2 AND user_id = $3",
                            label,
                            email_id,
                            user_id
                        )
                        .execute(&self.pool)
                        .await?;
                    }
                }
            }
            _ => {}
        }

        Ok(())
    }

    pub async fn get_email_content(&self, email_id: Uuid) -> Result<EmailContent> {
        let email = sqlx::query_as!(
            EmailContentRow,
            r#"
            SELECT id, subject, body, sender, recipients as "recipients!: Vec<String>",
                   received_at, attachments, headers
            FROM emails
            WHERE id = $1
            "#,
            email_id
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(EmailContent {
            id: email.id,
            subject: email.subject,
            body: email.body,
            sender: email.sender,
            recipients: email.recipients,
            received_at: email.received_at,
            attachments: serde_json::from_value(email.attachments.unwrap_or_default())?,
            headers: serde_json::from_value(email.headers.unwrap_or_default())?,
        })
    }
}

#[derive(Debug)]
struct EmailSyncRow {
    id: Uuid,
    version: i64,
    folder: String,
    is_read: bool,
    is_starred: bool,
    labels: Vec<String>,
    updated_at: DateTime<Utc>,
    deleted: bool,
}

#[derive(Debug)]
struct EmailContentRow {
    id: Uuid,
    subject: Option<String>,
    body: String,
    sender: String,
    recipients: Vec<String>,
    received_at: DateTime<Utc>,
    attachments: Option<serde_json::Value>,
    headers: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailContent {
    pub id: Uuid,
    pub subject: Option<String>,
    pub body: String,
    pub sender: String,
    pub recipients: Vec<String>,
    pub received_at: DateTime<Utc>,
    pub attachments: Vec<AttachmentInfo>,
    pub headers: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttachmentInfo {
    pub id: Uuid,
    pub filename: String,
    pub mime_type: String,
    pub size: u64,
    pub cached: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionResult {
    pub action_id: Uuid,
    pub success: bool,
    pub error: Option<String>,
    pub new_version: Option<u64>,
}

// ============================================================================
// Smart Prefetch
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefetchConfig {
    pub user_id: Uuid,
    pub enabled: bool,
    pub max_emails: u32,
    pub max_cache_mb: u32,
    pub prefetch_attachments: bool,
    pub attachment_size_limit_mb: u32,
    pub prefetch_threads: bool,
    pub priority_senders: Vec<String>,
    pub priority_labels: Vec<String>,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            user_id: Uuid::nil(),
            enabled: true,
            max_emails: 200,
            max_cache_mb: 100,
            prefetch_attachments: true,
            attachment_size_limit_mb: 10,
            prefetch_threads: true,
            priority_senders: Vec::new(),
            priority_labels: vec!["important".to_string(), "urgent".to_string()],
        }
    }
}

pub struct PrefetchService {
    pool: PgPool,
}

impl PrefetchService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn predict_emails_to_cache(&self, user_id: Uuid) -> Result<Vec<Uuid>> {
        let config = self.get_config(user_id).await?;

        // Use ML-based prediction to determine which emails user will likely access
        // Factors: recency, sender importance, thread activity, labels

        let predictions = sqlx::query!(
            r#"
            WITH sender_importance AS (
                SELECT sender, COUNT(*) as interaction_count,
                       AVG(CASE WHEN is_read THEN 1.0 ELSE 0.0 END) as read_rate
                FROM emails
                WHERE user_id = $1 AND received_at > NOW() - INTERVAL '30 days'
                GROUP BY sender
            ),
            email_scores AS (
                SELECT e.id,
                    (CASE WHEN e.is_starred THEN 3.0 ELSE 0.0 END) +
                    (CASE WHEN e.is_read = false THEN 2.0 ELSE 0.0 END) +
                    (CASE WHEN e.received_at > NOW() - INTERVAL '1 day' THEN 5.0
                          WHEN e.received_at > NOW() - INTERVAL '7 days' THEN 3.0
                          ELSE 1.0 END) +
                    COALESCE(si.interaction_count / 10.0, 0) as score
                FROM emails e
                LEFT JOIN sender_importance si ON e.sender = si.sender
                WHERE e.user_id = $1 AND e.folder = 'Inbox'
            )
            SELECT id FROM email_scores
            ORDER BY score DESC
            LIMIT $2
            "#,
            user_id,
            config.max_emails as i64
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(predictions.into_iter().map(|r| r.id).collect())
    }

    pub async fn get_config(&self, user_id: Uuid) -> Result<PrefetchConfig> {
        let config = sqlx::query!(
            "SELECT * FROM prefetch_configs WHERE user_id = $1",
            user_id
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(config.map(|c| PrefetchConfig {
            user_id: c.user_id,
            enabled: c.enabled,
            max_emails: c.max_emails as u32,
            max_cache_mb: c.max_cache_mb as u32,
            prefetch_attachments: c.prefetch_attachments,
            attachment_size_limit_mb: c.attachment_size_limit_mb as u32,
            prefetch_threads: c.prefetch_threads,
            priority_senders: c.priority_senders,
            priority_labels: c.priority_labels,
        }).unwrap_or_else(|| PrefetchConfig { user_id, ..Default::default() }))
    }
}

// ============================================================================
// Mobile Gestures
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GestureConfig {
    pub user_id: Uuid,
    pub swipe_left: GestureAction,
    pub swipe_right: GestureAction,
    pub swipe_long_left: GestureAction,
    pub swipe_long_right: GestureAction,
    pub double_tap: GestureAction,
    pub long_press: GestureAction,
    pub shake: GestureAction,
    pub pull_down: GestureAction,
}

impl Default for GestureConfig {
    fn default() -> Self {
        Self {
            user_id: Uuid::nil(),
            swipe_left: GestureAction::Delete,
            swipe_right: GestureAction::Archive,
            swipe_long_left: GestureAction::Spam,
            swipe_long_right: GestureAction::Snooze { hours: 3 },
            double_tap: GestureAction::ToggleStar,
            long_press: GestureAction::SelectMultiple,
            shake: GestureAction::Undo,
            pull_down: GestureAction::Refresh,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GestureAction {
    Archive,
    Delete,
    Spam,
    MarkRead,
    MarkUnread,
    ToggleStar,
    Snooze { hours: u32 },
    Move { folder: String },
    Label { label: String },
    Reply,
    Forward,
    Undo,
    Refresh,
    SelectMultiple,
    QuickReply { template: String },
    Custom { action_id: String },
}

pub struct GestureService {
    pool: PgPool,
}

impl GestureService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn get_config(&self, user_id: Uuid) -> Result<GestureConfig> {
        let config = sqlx::query!(
            "SELECT * FROM gesture_configs WHERE user_id = $1",
            user_id
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(config.map(|c| GestureConfig {
            user_id: c.user_id,
            swipe_left: serde_json::from_value(c.swipe_left).unwrap_or(GestureAction::Delete),
            swipe_right: serde_json::from_value(c.swipe_right).unwrap_or(GestureAction::Archive),
            swipe_long_left: serde_json::from_value(c.swipe_long_left).unwrap_or(GestureAction::Spam),
            swipe_long_right: serde_json::from_value(c.swipe_long_right).unwrap_or(GestureAction::Snooze { hours: 3 }),
            double_tap: serde_json::from_value(c.double_tap).unwrap_or(GestureAction::ToggleStar),
            long_press: serde_json::from_value(c.long_press).unwrap_or(GestureAction::SelectMultiple),
            shake: serde_json::from_value(c.shake).unwrap_or(GestureAction::Undo),
            pull_down: serde_json::from_value(c.pull_down).unwrap_or(GestureAction::Refresh),
        }).unwrap_or_else(|| GestureConfig { user_id, ..Default::default() }))
    }

    pub async fn save_config(&self, config: &GestureConfig) -> Result<()> {
        sqlx::query!(
            r#"
            INSERT INTO gesture_configs (user_id, swipe_left, swipe_right, swipe_long_left, swipe_long_right, double_tap, long_press, shake, pull_down)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (user_id) DO UPDATE SET
                swipe_left = EXCLUDED.swipe_left,
                swipe_right = EXCLUDED.swipe_right,
                swipe_long_left = EXCLUDED.swipe_long_left,
                swipe_long_right = EXCLUDED.swipe_long_right,
                double_tap = EXCLUDED.double_tap,
                long_press = EXCLUDED.long_press,
                shake = EXCLUDED.shake,
                pull_down = EXCLUDED.pull_down
            "#,
            config.user_id,
            serde_json::to_value(&config.swipe_left)?,
            serde_json::to_value(&config.swipe_right)?,
            serde_json::to_value(&config.swipe_long_left)?,
            serde_json::to_value(&config.swipe_long_right)?,
            serde_json::to_value(&config.double_tap)?,
            serde_json::to_value(&config.long_press)?,
            serde_json::to_value(&config.shake)?,
            serde_json::to_value(&config.pull_down)?
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn execute_gesture(&self, user_id: Uuid, email_id: Uuid, gesture: &GestureAction) -> Result<()> {
        match gesture {
            GestureAction::Archive => {
                sqlx::query!(
                    "UPDATE emails SET folder = 'Archive' WHERE id = $1 AND user_id = $2",
                    email_id,
                    user_id
                )
                .execute(&self.pool)
                .await?;
            }
            GestureAction::Delete => {
                sqlx::query!(
                    "UPDATE emails SET folder = 'Trash' WHERE id = $1 AND user_id = $2",
                    email_id,
                    user_id
                )
                .execute(&self.pool)
                .await?;
            }
            GestureAction::Spam => {
                sqlx::query!(
                    "UPDATE emails SET folder = 'Spam' WHERE id = $1 AND user_id = $2",
                    email_id,
                    user_id
                )
                .execute(&self.pool)
                .await?;
            }
            GestureAction::MarkRead => {
                sqlx::query!(
                    "UPDATE emails SET is_read = true WHERE id = $1 AND user_id = $2",
                    email_id,
                    user_id
                )
                .execute(&self.pool)
                .await?;
            }
            GestureAction::MarkUnread => {
                sqlx::query!(
                    "UPDATE emails SET is_read = false WHERE id = $1 AND user_id = $2",
                    email_id,
                    user_id
                )
                .execute(&self.pool)
                .await?;
            }
            GestureAction::ToggleStar => {
                sqlx::query!(
                    "UPDATE emails SET is_starred = NOT is_starred WHERE id = $1 AND user_id = $2",
                    email_id,
                    user_id
                )
                .execute(&self.pool)
                .await?;
            }
            GestureAction::Snooze { hours } => {
                let snooze_until = Utc::now() + Duration::hours(*hours as i64);
                sqlx::query!(
                    "UPDATE emails SET snoozed_until = $1, folder = 'Snoozed' WHERE id = $2 AND user_id = $3",
                    snooze_until,
                    email_id,
                    user_id
                )
                .execute(&self.pool)
                .await?;
            }
            GestureAction::Move { folder } => {
                sqlx::query!(
                    "UPDATE emails SET folder = $1 WHERE id = $2 AND user_id = $3",
                    folder,
                    email_id,
                    user_id
                )
                .execute(&self.pool)
                .await?;
            }
            GestureAction::Label { label } => {
                sqlx::query!(
                    "UPDATE emails SET labels = array_append(labels, $1) WHERE id = $2 AND user_id = $3 AND NOT ($1 = ANY(labels))",
                    label,
                    email_id,
                    user_id
                )
                .execute(&self.pool)
                .await?;
            }
            _ => {}
        }

        Ok(())
    }
}

// ============================================================================
// Wearable Support
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WearableDevice {
    pub id: Uuid,
    pub user_id: Uuid,
    pub device_type: WearableType,
    pub device_name: String,
    pub connected: bool,
    pub last_sync: Option<DateTime<Utc>>,
    pub capabilities: Vec<WearableCapability>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WearableType {
    AppleWatch,
    WearOS,
    Fitbit,
    Garmin,
    Samsung,
    Generic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WearableCapability {
    Notifications,
    QuickReplies,
    VoiceDictation,
    Haptics,
    Complications,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WearableNotification {
    pub email_id: Uuid,
    pub subject: String,
    pub sender_name: String,
    pub preview: String,
    pub importance: NotificationImportance,
    pub quick_replies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationImportance {
    Critical,
    High,
    Normal,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WearableConfig {
    pub user_id: Uuid,
    pub notifications_enabled: bool,
    pub notification_importance_filter: NotificationImportance,
    pub quick_replies: Vec<String>,
    pub haptic_feedback: bool,
    pub show_sender_avatar: bool,
    pub max_preview_length: u32,
}

impl Default for WearableConfig {
    fn default() -> Self {
        Self {
            user_id: Uuid::nil(),
            notifications_enabled: true,
            notification_importance_filter: NotificationImportance::Normal,
            quick_replies: vec![
                "Thanks!".to_string(),
                "Got it.".to_string(),
                "I'll look into it.".to_string(),
                "Can we discuss later?".to_string(),
                "On my way.".to_string(),
            ],
            haptic_feedback: true,
            show_sender_avatar: true,
            max_preview_length: 100,
        }
    }
}

pub struct WearableService {
    pool: PgPool,
}

impl WearableService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn register_device(&self, user_id: Uuid, device: WearableDevice) -> Result<WearableDevice> {
        sqlx::query!(
            r#"
            INSERT INTO wearable_devices (id, user_id, device_type, device_name, connected, capabilities)
            VALUES ($1, $2, $3, $4, true, $5)
            ON CONFLICT (id) DO UPDATE SET
                device_name = EXCLUDED.device_name,
                connected = true,
                last_sync = NOW()
            "#,
            device.id,
            user_id,
            serde_json::to_string(&device.device_type)?,
            device.device_name,
            serde_json::to_value(&device.capabilities)?
        )
        .execute(&self.pool)
        .await?;

        Ok(device)
    }

    pub async fn prepare_notification(&self, user_id: Uuid, email_id: Uuid) -> Result<WearableNotification> {
        let config = self.get_config(user_id).await?;

        let email = sqlx::query!(
            r#"
            SELECT subject, sender, body, is_starred,
                   EXISTS(SELECT 1 FROM vip_contacts WHERE user_id = $1 AND email = emails.sender) as is_vip
            FROM emails
            WHERE id = $2 AND user_id = $1
            "#,
            user_id,
            email_id
        )
        .fetch_one(&self.pool)
        .await?;

        let importance = if email.is_starred {
            NotificationImportance::High
        } else if email.is_vip.unwrap_or(false) {
            NotificationImportance::High
        } else {
            NotificationImportance::Normal
        };

        let preview: String = email.body.chars()
            .take(config.max_preview_length as usize)
            .collect();

        Ok(WearableNotification {
            email_id,
            subject: email.subject.unwrap_or_default(),
            sender_name: email.sender,
            preview,
            importance,
            quick_replies: config.quick_replies,
        })
    }

    pub async fn send_quick_reply(&self, user_id: Uuid, email_id: Uuid, reply_text: &str) -> Result<Uuid> {
        let original = sqlx::query!(
            "SELECT sender, subject, thread_id FROM emails WHERE id = $1 AND user_id = $2",
            email_id,
            user_id
        )
        .fetch_one(&self.pool)
        .await?;

        let reply_id = Uuid::new_v4();
        let subject = format!("Re: {}", original.subject.unwrap_or_default());

        sqlx::query!(
            r#"
            INSERT INTO emails (id, user_id, subject, body, sender, recipients, folder, thread_id, in_reply_to, sent_at, is_read)
            VALUES ($1, $2, $3, $4, (SELECT email FROM users WHERE id = $2), ARRAY[$5], 'Sent', $6, $7, NOW(), true)
            "#,
            reply_id,
            user_id,
            subject,
            reply_text,
            original.sender,
            original.thread_id,
            email_id
        )
        .execute(&self.pool)
        .await?;

        Ok(reply_id)
    }

    pub async fn get_config(&self, user_id: Uuid) -> Result<WearableConfig> {
        let config = sqlx::query!(
            "SELECT * FROM wearable_configs WHERE user_id = $1",
            user_id
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(config.map(|c| WearableConfig {
            user_id: c.user_id,
            notifications_enabled: c.notifications_enabled,
            notification_importance_filter: serde_json::from_value(c.notification_importance_filter).unwrap_or(NotificationImportance::Normal),
            quick_replies: c.quick_replies,
            haptic_feedback: c.haptic_feedback,
            show_sender_avatar: c.show_sender_avatar,
            max_preview_length: c.max_preview_length as u32,
        }).unwrap_or_else(|| WearableConfig { user_id, ..Default::default() }))
    }
}

// ============================================================================
// Cross-Device Continuity
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceSession {
    pub id: Uuid,
    pub user_id: Uuid,
    pub device_id: String,
    pub device_type: DeviceType,
    pub device_name: String,
    pub current_activity: Option<ActivityState>,
    pub last_active: DateTime<Utc>,
    pub push_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceType {
    Desktop,
    Laptop,
    Tablet,
    Phone,
    Watch,
    Browser,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityState {
    pub activity_type: ActivityType,
    pub email_id: Option<Uuid>,
    pub draft_id: Option<Uuid>,
    pub scroll_position: Option<f32>,
    pub cursor_position: Option<u32>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivityType {
    ReadingEmail,
    ComposingEmail,
    ViewingFolder { folder: String },
    Searching { query: String },
    ViewingThread { thread_id: Uuid },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffData {
    pub from_device: String,
    pub to_device: String,
    pub activity: ActivityState,
    pub context: serde_json::Value,
}

pub struct ContinuityService {
    pool: PgPool,
}

impl ContinuityService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn register_device(&self, session: DeviceSession) -> Result<DeviceSession> {
        sqlx::query!(
            r#"
            INSERT INTO device_sessions (id, user_id, device_id, device_type, device_name, last_active, push_token)
            VALUES ($1, $2, $3, $4, $5, NOW(), $6)
            ON CONFLICT (device_id) DO UPDATE SET
                device_name = EXCLUDED.device_name,
                last_active = NOW(),
                push_token = EXCLUDED.push_token
            "#,
            session.id,
            session.user_id,
            session.device_id,
            serde_json::to_string(&session.device_type)?,
            session.device_name,
            session.push_token
        )
        .execute(&self.pool)
        .await?;

        Ok(session)
    }

    pub async fn update_activity(&self, device_id: &str, activity: ActivityState) -> Result<()> {
        sqlx::query!(
            r#"
            UPDATE device_sessions
            SET current_activity = $1, last_active = NOW()
            WHERE device_id = $2
            "#,
            serde_json::to_value(&activity)?,
            device_id
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_active_sessions(&self, user_id: Uuid) -> Result<Vec<DeviceSession>> {
        let sessions = sqlx::query!(
            r#"
            SELECT id, user_id, device_id, device_type, device_name, current_activity, last_active, push_token
            FROM device_sessions
            WHERE user_id = $1 AND last_active > NOW() - INTERVAL '15 minutes'
            ORDER BY last_active DESC
            "#,
            user_id
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(sessions.into_iter().map(|s| DeviceSession {
            id: s.id,
            user_id: s.user_id,
            device_id: s.device_id,
            device_type: serde_json::from_str(&s.device_type).unwrap_or(DeviceType::Browser),
            device_name: s.device_name,
            current_activity: s.current_activity.and_then(|a| serde_json::from_value(a).ok()),
            last_active: s.last_active,
            push_token: s.push_token,
        }).collect())
    }

    pub async fn initiate_handoff(&self, user_id: Uuid, from_device: &str, to_device: &str) -> Result<HandoffData> {
        let from_session = sqlx::query!(
            "SELECT current_activity FROM device_sessions WHERE device_id = $1 AND user_id = $2",
            from_device,
            user_id
        )
        .fetch_one(&self.pool)
        .await?;

        let activity: ActivityState = serde_json::from_value(
            from_session.current_activity.unwrap_or_default()
        )?;

        // Prepare context based on activity type
        let context = match &activity.activity_type {
            ActivityType::ComposingEmail => {
                if let Some(draft_id) = activity.draft_id {
                    let draft = sqlx::query!(
                        "SELECT subject, body, recipients FROM drafts WHERE id = $1",
                        draft_id
                    )
                    .fetch_optional(&self.pool)
                    .await?;

                    serde_json::json!({
                        "draft": draft,
                        "cursor_position": activity.cursor_position,
                    })
                } else {
                    serde_json::json!({})
                }
            }
            ActivityType::ReadingEmail => {
                serde_json::json!({
                    "email_id": activity.email_id,
                    "scroll_position": activity.scroll_position,
                })
            }
            _ => serde_json::json!({}),
        };

        let handoff = HandoffData {
            from_device: from_device.to_string(),
            to_device: to_device.to_string(),
            activity,
            context,
        };

        // Notify target device
        self.notify_device(to_device, &handoff).await?;

        Ok(handoff)
    }

    async fn notify_device(&self, device_id: &str, handoff: &HandoffData) -> Result<()> {
        let session = sqlx::query!(
            "SELECT push_token FROM device_sessions WHERE device_id = $1",
            device_id
        )
        .fetch_optional(&self.pool)
        .await?;

        if let Some(s) = session {
            if let Some(token) = s.push_token {
                // Would send push notification
                // self.send_push_notification(&token, handoff).await?;
            }
        }

        Ok(())
    }
}

// ============================================================================
// Bandwidth Optimization
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthConfig {
    pub user_id: Uuid,
    pub data_saver_mode: bool,
    pub lazy_load_images: bool,
    pub compress_attachments: bool,
    pub image_quality: ImageQuality,
    pub max_inline_image_kb: u32,
    pub stream_large_attachments: bool,
    pub prefetch_on_wifi_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageQuality {
    Original,
    High,
    Medium,
    Low,
    None,
}

impl Default for BandwidthConfig {
    fn default() -> Self {
        Self {
            user_id: Uuid::nil(),
            data_saver_mode: false,
            lazy_load_images: true,
            compress_attachments: false,
            image_quality: ImageQuality::High,
            max_inline_image_kb: 100,
            stream_large_attachments: true,
            prefetch_on_wifi_only: true,
        }
    }
}

pub struct BandwidthOptimizer {
    pool: PgPool,
}

impl BandwidthOptimizer {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn get_config(&self, user_id: Uuid) -> Result<BandwidthConfig> {
        let config = sqlx::query!(
            "SELECT * FROM bandwidth_configs WHERE user_id = $1",
            user_id
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(config.map(|c| BandwidthConfig {
            user_id: c.user_id,
            data_saver_mode: c.data_saver_mode,
            lazy_load_images: c.lazy_load_images,
            compress_attachments: c.compress_attachments,
            image_quality: serde_json::from_value(c.image_quality).unwrap_or(ImageQuality::High),
            max_inline_image_kb: c.max_inline_image_kb as u32,
            stream_large_attachments: c.stream_large_attachments,
            prefetch_on_wifi_only: c.prefetch_on_wifi_only,
        }).unwrap_or_else(|| BandwidthConfig { user_id, ..Default::default() }))
    }

    pub async fn optimize_email_content(&self, user_id: Uuid, email_id: Uuid, connection_type: ConnectionType) -> Result<OptimizedContent> {
        let config = self.get_config(user_id).await?;

        let email = sqlx::query!(
            "SELECT body, attachments FROM emails WHERE id = $1",
            email_id
        )
        .fetch_one(&self.pool)
        .await?;

        let mut body = email.body.clone();
        let attachments: Vec<AttachmentInfo> = serde_json::from_value(
            email.attachments.unwrap_or_default()
        ).unwrap_or_default();

        // Apply data saver optimizations
        if config.data_saver_mode || matches!(connection_type, ConnectionType::Cellular2G | ConnectionType::Cellular3G) {
            // Strip images from body
            if matches!(config.image_quality, ImageQuality::None) {
                body = self.strip_images(&body);
            }

            // Replace image URLs with placeholders
            if config.lazy_load_images {
                body = self.add_lazy_loading(&body);
            }
        }

        let optimized_attachments: Vec<OptimizedAttachment> = attachments.into_iter()
            .map(|a| {
                let should_stream = config.stream_large_attachments && a.size > 1024 * 1024;
                OptimizedAttachment {
                    id: a.id,
                    filename: a.filename,
                    mime_type: a.mime_type,
                    size: a.size,
                    stream_url: if should_stream { Some(format!("/api/attachments/{}/stream", a.id)) } else { None },
                    thumbnail_url: if a.mime_type.starts_with("image/") {
                        Some(format!("/api/attachments/{}/thumbnail", a.id))
                    } else { None },
                }
            })
            .collect();

        Ok(OptimizedContent {
            body,
            attachments: optimized_attachments,
            data_saver_applied: config.data_saver_mode,
            estimated_size_kb: (body.len() / 1024) as u32,
        })
    }

    fn strip_images(&self, html: &str) -> String {
        // Simple regex to remove img tags
        let re = regex::Regex::new(r"<img[^>]*>").unwrap();
        re.replace_all(html, "[Image removed - Data Saver mode]").to_string()
    }

    fn add_lazy_loading(&self, html: &str) -> String {
        // Add loading="lazy" to img tags
        let re = regex::Regex::new(r"<img ").unwrap();
        re.replace_all(html, "<img loading=\"lazy\" ").to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectionType {
    Wifi,
    Ethernet,
    Cellular5G,
    Cellular4G,
    Cellular3G,
    Cellular2G,
    Offline,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedContent {
    pub body: String,
    pub attachments: Vec<OptimizedAttachment>,
    pub data_saver_applied: bool,
    pub estimated_size_kb: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedAttachment {
    pub id: Uuid,
    pub filename: String,
    pub mime_type: String,
    pub size: u64,
    pub stream_url: Option<String>,
    pub thumbnail_url: Option<String>,
}
