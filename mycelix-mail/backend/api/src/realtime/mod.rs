// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Real-time Collaboration Module
//!
//! Provides live draft editing, presence indicators, collaborative threads,
//! shared drafts, comment threads, and version history.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use sqlx::PgPool;
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum RealtimeError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("Draft not found: {0}")]
    DraftNotFound(Uuid),
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    #[error("Conflict detected: {0}")]
    ConflictDetected(String),
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
}

// ============================================================================
// Live Draft Editing
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborativeDraft {
    pub id: Uuid,
    pub owner_id: Uuid,
    pub subject: String,
    pub body: String,
    pub recipients: Vec<String>,
    pub cc: Vec<String>,
    pub bcc: Vec<String>,
    pub collaborators: Vec<DraftCollaborator>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub version: i32,
    pub is_locked: bool,
    pub locked_by: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DraftCollaborator {
    pub user_id: Uuid,
    pub user_name: String,
    pub user_email: String,
    pub permission: CollaboratorPermission,
    pub added_at: DateTime<Utc>,
    pub last_active: Option<DateTime<Utc>>,
    pub cursor_position: Option<CursorPosition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollaboratorPermission {
    View,
    Comment,
    Edit,
    Admin,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CursorPosition {
    pub field: String, // "subject", "body", etc.
    pub offset: i32,
    pub selection_end: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DraftOperation {
    pub id: Uuid,
    pub draft_id: Uuid,
    pub user_id: Uuid,
    pub operation_type: OperationType,
    pub field: String,
    pub position: i32,
    pub content: Option<String>,
    pub length: Option<i32>,
    pub timestamp: DateTime<Utc>,
    pub version: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationType {
    Insert,
    Delete,
    Replace,
    SetField,
}

pub struct LiveDraftService {
    pool: PgPool,
}

impl LiveDraftService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn create_collaborative_draft(
        &self,
        owner_id: Uuid,
        subject: Option<&str>,
    ) -> Result<CollaborativeDraft, RealtimeError> {
        let draft_id = Uuid::new_v4();

        sqlx::query!(
            r#"
            INSERT INTO collaborative_drafts (id, owner_id, subject, body, recipients,
                                             cc, bcc, version, created_at, updated_at)
            VALUES ($1, $2, $3, '', '[]', '[]', '[]', 1, NOW(), NOW())
            "#,
            draft_id,
            owner_id,
            subject.unwrap_or("")
        )
        .execute(&self.pool)
        .await?;

        // Add owner as admin collaborator
        sqlx::query!(
            r#"
            INSERT INTO draft_collaborators (draft_id, user_id, permission, added_at)
            VALUES ($1, $2, $3, NOW())
            "#,
            draft_id,
            owner_id,
            serde_json::to_string(&CollaboratorPermission::Admin).unwrap()
        )
        .execute(&self.pool)
        .await?;

        self.get_draft(draft_id).await
    }

    pub async fn get_draft(&self, draft_id: Uuid) -> Result<CollaborativeDraft, RealtimeError> {
        let draft = sqlx::query!(
            r#"
            SELECT id, owner_id, subject, body, recipients, cc, bcc,
                   version, created_at, updated_at, is_locked, locked_by
            FROM collaborative_drafts
            WHERE id = $1
            "#,
            draft_id
        )
        .fetch_optional(&self.pool)
        .await?
        .ok_or(RealtimeError::DraftNotFound(draft_id))?;

        let collaborators = self.get_collaborators(draft_id).await?;

        Ok(CollaborativeDraft {
            id: draft.id,
            owner_id: draft.owner_id,
            subject: draft.subject,
            body: draft.body,
            recipients: serde_json::from_value(draft.recipients).unwrap_or_default(),
            cc: serde_json::from_value(draft.cc).unwrap_or_default(),
            bcc: serde_json::from_value(draft.bcc).unwrap_or_default(),
            collaborators,
            created_at: draft.created_at,
            updated_at: draft.updated_at,
            version: draft.version,
            is_locked: draft.is_locked,
            locked_by: draft.locked_by,
        })
    }

    pub async fn get_collaborators(
        &self,
        draft_id: Uuid,
    ) -> Result<Vec<DraftCollaborator>, RealtimeError> {
        let collaborators = sqlx::query!(
            r#"
            SELECT c.user_id, u.name as user_name, u.email as user_email,
                   c.permission, c.added_at, c.last_active,
                   c.cursor_field, c.cursor_offset, c.cursor_selection_end
            FROM draft_collaborators c
            JOIN users u ON c.user_id = u.id
            WHERE c.draft_id = $1
            "#,
            draft_id
        )
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|row| DraftCollaborator {
            user_id: row.user_id,
            user_name: row.user_name,
            user_email: row.user_email,
            permission: serde_json::from_str(&row.permission)
                .unwrap_or(CollaboratorPermission::View),
            added_at: row.added_at,
            last_active: row.last_active,
            cursor_position: row.cursor_field.map(|field| CursorPosition {
                field,
                offset: row.cursor_offset.unwrap_or(0),
                selection_end: row.cursor_selection_end,
            }),
        })
        .collect();

        Ok(collaborators)
    }

    pub async fn add_collaborator(
        &self,
        draft_id: Uuid,
        user_id: Uuid,
        added_by: Uuid,
        permission: CollaboratorPermission,
    ) -> Result<(), RealtimeError> {
        // Verify requester has admin permission
        self.check_permission(draft_id, added_by, CollaboratorPermission::Admin)
            .await?;

        sqlx::query!(
            r#"
            INSERT INTO draft_collaborators (draft_id, user_id, permission, added_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (draft_id, user_id) DO UPDATE SET
                permission = EXCLUDED.permission
            "#,
            draft_id,
            user_id,
            serde_json::to_string(&permission).unwrap()
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn apply_operation(
        &self,
        draft_id: Uuid,
        user_id: Uuid,
        operation: DraftOperation,
    ) -> Result<i32, RealtimeError> {
        // Verify edit permission
        self.check_permission(draft_id, user_id, CollaboratorPermission::Edit)
            .await?;

        // Check for conflicts
        let current_version = sqlx::query_scalar!(
            "SELECT version FROM collaborative_drafts WHERE id = $1",
            draft_id
        )
        .fetch_one(&self.pool)
        .await?;

        if operation.version != current_version {
            return Err(RealtimeError::ConflictDetected(format!(
                "Expected version {}, got {}",
                current_version, operation.version
            )));
        }

        // Apply operation
        match operation.operation_type {
            OperationType::Insert => {
                self.apply_insert(draft_id, &operation).await?;
            }
            OperationType::Delete => {
                self.apply_delete(draft_id, &operation).await?;
            }
            OperationType::Replace => {
                self.apply_replace(draft_id, &operation).await?;
            }
            OperationType::SetField => {
                self.apply_set_field(draft_id, &operation).await?;
            }
        }

        // Increment version
        let new_version = sqlx::query_scalar!(
            r#"
            UPDATE collaborative_drafts
            SET version = version + 1, updated_at = NOW()
            WHERE id = $1
            RETURNING version
            "#,
            draft_id
        )
        .fetch_one(&self.pool)
        .await?;

        // Store operation in history
        sqlx::query!(
            r#"
            INSERT INTO draft_operations (id, draft_id, user_id, operation_type,
                                         field, position, content, length, version, timestamp)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
            "#,
            operation.id,
            draft_id,
            user_id,
            serde_json::to_string(&operation.operation_type).unwrap(),
            operation.field,
            operation.position,
            operation.content,
            operation.length,
            new_version
        )
        .execute(&self.pool)
        .await?;

        Ok(new_version)
    }

    async fn apply_insert(
        &self,
        draft_id: Uuid,
        op: &DraftOperation,
    ) -> Result<(), RealtimeError> {
        let content = op.content.as_deref().unwrap_or("");

        match op.field.as_str() {
            "body" => {
                sqlx::query!(
                    r#"
                    UPDATE collaborative_drafts
                    SET body = CONCAT(
                        LEFT(body, $1),
                        $2,
                        SUBSTRING(body FROM $1 + 1)
                    )
                    WHERE id = $3
                    "#,
                    op.position,
                    content,
                    draft_id
                )
                .execute(&self.pool)
                .await?;
            }
            "subject" => {
                sqlx::query!(
                    r#"
                    UPDATE collaborative_drafts
                    SET subject = CONCAT(
                        LEFT(subject, $1),
                        $2,
                        SUBSTRING(subject FROM $1 + 1)
                    )
                    WHERE id = $3
                    "#,
                    op.position,
                    content,
                    draft_id
                )
                .execute(&self.pool)
                .await?;
            }
            _ => {}
        }

        Ok(())
    }

    async fn apply_delete(
        &self,
        draft_id: Uuid,
        op: &DraftOperation,
    ) -> Result<(), RealtimeError> {
        let length = op.length.unwrap_or(1);

        match op.field.as_str() {
            "body" => {
                sqlx::query!(
                    r#"
                    UPDATE collaborative_drafts
                    SET body = CONCAT(
                        LEFT(body, $1),
                        SUBSTRING(body FROM $1 + $2 + 1)
                    )
                    WHERE id = $3
                    "#,
                    op.position,
                    length,
                    draft_id
                )
                .execute(&self.pool)
                .await?;
            }
            "subject" => {
                sqlx::query!(
                    r#"
                    UPDATE collaborative_drafts
                    SET subject = CONCAT(
                        LEFT(subject, $1),
                        SUBSTRING(subject FROM $1 + $2 + 1)
                    )
                    WHERE id = $3
                    "#,
                    op.position,
                    length,
                    draft_id
                )
                .execute(&self.pool)
                .await?;
            }
            _ => {}
        }

        Ok(())
    }

    async fn apply_replace(
        &self,
        draft_id: Uuid,
        op: &DraftOperation,
    ) -> Result<(), RealtimeError> {
        let content = op.content.as_deref().unwrap_or("");
        let length = op.length.unwrap_or(0);

        match op.field.as_str() {
            "body" => {
                sqlx::query!(
                    r#"
                    UPDATE collaborative_drafts
                    SET body = CONCAT(
                        LEFT(body, $1),
                        $2,
                        SUBSTRING(body FROM $1 + $3 + 1)
                    )
                    WHERE id = $4
                    "#,
                    op.position,
                    content,
                    length,
                    draft_id
                )
                .execute(&self.pool)
                .await?;
            }
            _ => {}
        }

        Ok(())
    }

    async fn apply_set_field(
        &self,
        draft_id: Uuid,
        op: &DraftOperation,
    ) -> Result<(), RealtimeError> {
        let content = op.content.as_deref().unwrap_or("");

        match op.field.as_str() {
            "subject" => {
                sqlx::query!(
                    "UPDATE collaborative_drafts SET subject = $1 WHERE id = $2",
                    content,
                    draft_id
                )
                .execute(&self.pool)
                .await?;
            }
            "body" => {
                sqlx::query!(
                    "UPDATE collaborative_drafts SET body = $1 WHERE id = $2",
                    content,
                    draft_id
                )
                .execute(&self.pool)
                .await?;
            }
            "recipients" => {
                let recipients: Vec<String> = serde_json::from_str(content).unwrap_or_default();
                sqlx::query!(
                    "UPDATE collaborative_drafts SET recipients = $1 WHERE id = $2",
                    serde_json::to_value(&recipients).unwrap(),
                    draft_id
                )
                .execute(&self.pool)
                .await?;
            }
            _ => {}
        }

        Ok(())
    }

    async fn check_permission(
        &self,
        draft_id: Uuid,
        user_id: Uuid,
        required: CollaboratorPermission,
    ) -> Result<(), RealtimeError> {
        let permission = sqlx::query_scalar!(
            "SELECT permission FROM draft_collaborators WHERE draft_id = $1 AND user_id = $2",
            draft_id,
            user_id
        )
        .fetch_optional(&self.pool)
        .await?
        .ok_or(RealtimeError::PermissionDenied(
            "Not a collaborator on this draft".to_string(),
        ))?;

        let has_permission = match required {
            CollaboratorPermission::View => true,
            CollaboratorPermission::Comment => {
                matches!(
                    serde_json::from_str::<CollaboratorPermission>(&permission),
                    Ok(CollaboratorPermission::Comment)
                        | Ok(CollaboratorPermission::Edit)
                        | Ok(CollaboratorPermission::Admin)
                )
            }
            CollaboratorPermission::Edit => {
                matches!(
                    serde_json::from_str::<CollaboratorPermission>(&permission),
                    Ok(CollaboratorPermission::Edit) | Ok(CollaboratorPermission::Admin)
                )
            }
            CollaboratorPermission::Admin => {
                matches!(
                    serde_json::from_str::<CollaboratorPermission>(&permission),
                    Ok(CollaboratorPermission::Admin)
                )
            }
        };

        if !has_permission {
            return Err(RealtimeError::PermissionDenied(format!(
                "Requires {:?} permission",
                required
            )));
        }

        Ok(())
    }

    pub async fn update_cursor(
        &self,
        draft_id: Uuid,
        user_id: Uuid,
        cursor: CursorPosition,
    ) -> Result<(), RealtimeError> {
        sqlx::query!(
            r#"
            UPDATE draft_collaborators
            SET cursor_field = $1, cursor_offset = $2, cursor_selection_end = $3,
                last_active = NOW()
            WHERE draft_id = $4 AND user_id = $5
            "#,
            cursor.field,
            cursor.offset,
            cursor.selection_end,
            draft_id,
            user_id
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }
}

// ============================================================================
// Presence Indicators
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPresence {
    pub user_id: Uuid,
    pub user_name: String,
    pub status: PresenceStatus,
    pub current_view: Option<CurrentView>,
    pub last_active: DateTime<Utc>,
    pub typing_in: Option<Uuid>, // draft_id if currently typing
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PresenceStatus {
    Online,
    Away,
    DoNotDisturb,
    Offline,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurrentView {
    pub view_type: ViewType,
    pub resource_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViewType {
    Inbox,
    Email,
    Draft,
    Compose,
    Settings,
    Search,
}

pub struct PresenceService {
    pool: PgPool,
}

impl PresenceService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn update_presence(
        &self,
        user_id: Uuid,
        status: PresenceStatus,
        current_view: Option<CurrentView>,
    ) -> Result<(), RealtimeError> {
        sqlx::query!(
            r#"
            INSERT INTO user_presence (user_id, status, current_view, last_active)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (user_id) DO UPDATE SET
                status = EXCLUDED.status,
                current_view = EXCLUDED.current_view,
                last_active = NOW()
            "#,
            user_id,
            serde_json::to_string(&status).unwrap(),
            current_view.map(|v| serde_json::to_value(&v).unwrap())
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn set_typing(
        &self,
        user_id: Uuid,
        draft_id: Option<Uuid>,
    ) -> Result<(), RealtimeError> {
        sqlx::query!(
            r#"
            UPDATE user_presence
            SET typing_in = $1, last_active = NOW()
            WHERE user_id = $2
            "#,
            draft_id,
            user_id
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_team_presence(&self, user_id: Uuid) -> Result<Vec<UserPresence>, RealtimeError> {
        // Get presence for users in same team/shared mailboxes
        let presence = sqlx::query!(
            r#"
            SELECT DISTINCT p.user_id, u.name as user_name, p.status,
                   p.current_view, p.last_active, p.typing_in
            FROM user_presence p
            JOIN users u ON p.user_id = u.id
            WHERE p.last_active > NOW() - INTERVAL '15 minutes'
            AND p.user_id != $1
            AND (
                p.user_id IN (
                    SELECT DISTINCT mem2.user_id
                    FROM shared_mailbox_members mem1
                    JOIN shared_mailbox_members mem2 ON mem1.mailbox_id = mem2.mailbox_id
                    WHERE mem1.user_id = $1
                )
                OR p.user_id IN (
                    SELECT user_id FROM team_members WHERE team_id IN (
                        SELECT team_id FROM team_members WHERE user_id = $1
                    )
                )
            )
            ORDER BY p.last_active DESC
            "#,
            user_id
        )
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|row| UserPresence {
            user_id: row.user_id,
            user_name: row.user_name,
            status: serde_json::from_str(&row.status).unwrap_or(PresenceStatus::Offline),
            current_view: row.current_view.and_then(|v| serde_json::from_value(v).ok()),
            last_active: row.last_active,
            typing_in: row.typing_in,
        })
        .collect();

        Ok(presence)
    }

    pub async fn get_draft_collaborator_presence(
        &self,
        draft_id: Uuid,
    ) -> Result<Vec<UserPresence>, RealtimeError> {
        let presence = sqlx::query!(
            r#"
            SELECT p.user_id, u.name as user_name, p.status,
                   p.current_view, p.last_active, p.typing_in
            FROM user_presence p
            JOIN users u ON p.user_id = u.id
            JOIN draft_collaborators dc ON p.user_id = dc.user_id
            WHERE dc.draft_id = $1 AND p.last_active > NOW() - INTERVAL '5 minutes'
            "#,
            draft_id
        )
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|row| UserPresence {
            user_id: row.user_id,
            user_name: row.user_name,
            status: serde_json::from_str(&row.status).unwrap_or(PresenceStatus::Offline),
            current_view: row.current_view.and_then(|v| serde_json::from_value(v).ok()),
            last_active: row.last_active,
            typing_in: row.typing_in,
        })
        .collect();

        Ok(presence)
    }
}

// ============================================================================
// Collaborative Threads
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborativeThread {
    pub id: Uuid,
    pub email_id: Uuid,
    pub created_by: Uuid,
    pub created_at: DateTime<Utc>,
    pub participants: Vec<Uuid>,
    pub message_count: i32,
    pub last_message_at: Option<DateTime<Utc>>,
    pub is_resolved: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadMessage {
    pub id: Uuid,
    pub thread_id: Uuid,
    pub author_id: Uuid,
    pub author_name: String,
    pub content: String,
    pub created_at: DateTime<Utc>,
    pub edited_at: Option<DateTime<Utc>>,
    pub reactions: Vec<MessageReaction>,
    pub attachments: Vec<ThreadAttachment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageReaction {
    pub emoji: String,
    pub user_ids: Vec<Uuid>,
    pub count: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadAttachment {
    pub id: Uuid,
    pub filename: String,
    pub size: i64,
    pub mime_type: String,
}

pub struct ThreadService {
    pool: PgPool,
}

impl ThreadService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn create_thread(
        &self,
        email_id: Uuid,
        created_by: Uuid,
        initial_message: &str,
    ) -> Result<CollaborativeThread, RealtimeError> {
        let thread_id = Uuid::new_v4();

        sqlx::query!(
            r#"
            INSERT INTO collaborative_threads (id, email_id, created_by, participants, created_at)
            VALUES ($1, $2, $3, ARRAY[$3], NOW())
            "#,
            thread_id,
            email_id,
            created_by
        )
        .execute(&self.pool)
        .await?;

        // Add initial message
        self.add_message(thread_id, created_by, initial_message, vec![]).await?;

        self.get_thread(thread_id).await
    }

    pub async fn get_thread(&self, thread_id: Uuid) -> Result<CollaborativeThread, RealtimeError> {
        let thread = sqlx::query!(
            r#"
            SELECT id, email_id, created_by, created_at,
                   participants as "participants: Vec<Uuid>",
                   message_count, last_message_at, is_resolved
            FROM collaborative_threads
            WHERE id = $1
            "#,
            thread_id
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(CollaborativeThread {
            id: thread.id,
            email_id: thread.email_id,
            created_by: thread.created_by,
            created_at: thread.created_at,
            participants: thread.participants,
            message_count: thread.message_count,
            last_message_at: thread.last_message_at,
            is_resolved: thread.is_resolved,
        })
    }

    pub async fn get_threads_for_email(
        &self,
        email_id: Uuid,
    ) -> Result<Vec<CollaborativeThread>, RealtimeError> {
        let threads = sqlx::query!(
            r#"
            SELECT id, email_id, created_by, created_at,
                   participants as "participants: Vec<Uuid>",
                   message_count, last_message_at, is_resolved
            FROM collaborative_threads
            WHERE email_id = $1
            ORDER BY created_at DESC
            "#,
            email_id
        )
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|row| CollaborativeThread {
            id: row.id,
            email_id: row.email_id,
            created_by: row.created_by,
            created_at: row.created_at,
            participants: row.participants,
            message_count: row.message_count,
            last_message_at: row.last_message_at,
            is_resolved: row.is_resolved,
        })
        .collect();

        Ok(threads)
    }

    pub async fn add_message(
        &self,
        thread_id: Uuid,
        author_id: Uuid,
        content: &str,
        attachments: Vec<ThreadAttachment>,
    ) -> Result<ThreadMessage, RealtimeError> {
        let message_id = Uuid::new_v4();

        sqlx::query!(
            r#"
            INSERT INTO thread_messages (id, thread_id, author_id, content, attachments, created_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
            "#,
            message_id,
            thread_id,
            author_id,
            content,
            serde_json::to_value(&attachments).unwrap()
        )
        .execute(&self.pool)
        .await?;

        // Update thread stats
        sqlx::query!(
            r#"
            UPDATE collaborative_threads
            SET message_count = message_count + 1,
                last_message_at = NOW(),
                participants = ARRAY(SELECT DISTINCT unnest(participants || ARRAY[$1]))
            WHERE id = $2
            "#,
            author_id,
            thread_id
        )
        .execute(&self.pool)
        .await?;

        self.get_message(message_id).await
    }

    pub async fn get_message(&self, message_id: Uuid) -> Result<ThreadMessage, RealtimeError> {
        let msg = sqlx::query!(
            r#"
            SELECT m.id, m.thread_id, m.author_id, u.name as author_name,
                   m.content, m.created_at, m.edited_at, m.attachments
            FROM thread_messages m
            JOIN users u ON m.author_id = u.id
            WHERE m.id = $1
            "#,
            message_id
        )
        .fetch_one(&self.pool)
        .await?;

        let reactions = self.get_message_reactions(message_id).await?;

        Ok(ThreadMessage {
            id: msg.id,
            thread_id: msg.thread_id,
            author_id: msg.author_id,
            author_name: msg.author_name,
            content: msg.content,
            created_at: msg.created_at,
            edited_at: msg.edited_at,
            reactions,
            attachments: serde_json::from_value(msg.attachments).unwrap_or_default(),
        })
    }

    pub async fn get_messages(
        &self,
        thread_id: Uuid,
        limit: i32,
        before: Option<DateTime<Utc>>,
    ) -> Result<Vec<ThreadMessage>, RealtimeError> {
        let messages = sqlx::query!(
            r#"
            SELECT m.id, m.thread_id, m.author_id, u.name as author_name,
                   m.content, m.created_at, m.edited_at, m.attachments
            FROM thread_messages m
            JOIN users u ON m.author_id = u.id
            WHERE m.thread_id = $1
            AND ($2::timestamptz IS NULL OR m.created_at < $2)
            ORDER BY m.created_at DESC
            LIMIT $3
            "#,
            thread_id,
            before,
            limit as i64
        )
        .fetch_all(&self.pool)
        .await?;

        let mut result = Vec::new();
        for msg in messages {
            let reactions = self.get_message_reactions(msg.id).await?;
            result.push(ThreadMessage {
                id: msg.id,
                thread_id: msg.thread_id,
                author_id: msg.author_id,
                author_name: msg.author_name,
                content: msg.content,
                created_at: msg.created_at,
                edited_at: msg.edited_at,
                reactions,
                attachments: serde_json::from_value(msg.attachments).unwrap_or_default(),
            });
        }

        Ok(result)
    }

    pub async fn add_reaction(
        &self,
        message_id: Uuid,
        user_id: Uuid,
        emoji: &str,
    ) -> Result<(), RealtimeError> {
        sqlx::query!(
            r#"
            INSERT INTO message_reactions (message_id, user_id, emoji, created_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (message_id, user_id, emoji) DO NOTHING
            "#,
            message_id,
            user_id,
            emoji
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn remove_reaction(
        &self,
        message_id: Uuid,
        user_id: Uuid,
        emoji: &str,
    ) -> Result<(), RealtimeError> {
        sqlx::query!(
            "DELETE FROM message_reactions WHERE message_id = $1 AND user_id = $2 AND emoji = $3",
            message_id,
            user_id,
            emoji
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    async fn get_message_reactions(
        &self,
        message_id: Uuid,
    ) -> Result<Vec<MessageReaction>, RealtimeError> {
        let reactions = sqlx::query!(
            r#"
            SELECT emoji, ARRAY_AGG(user_id) as "user_ids!: Vec<Uuid>", COUNT(*) as count
            FROM message_reactions
            WHERE message_id = $1
            GROUP BY emoji
            "#,
            message_id
        )
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|row| MessageReaction {
            emoji: row.emoji,
            user_ids: row.user_ids,
            count: row.count.unwrap_or(0) as i32,
        })
        .collect();

        Ok(reactions)
    }

    pub async fn resolve_thread(&self, thread_id: Uuid, user_id: Uuid) -> Result<(), RealtimeError> {
        sqlx::query!(
            r#"
            UPDATE collaborative_threads
            SET is_resolved = true, resolved_by = $1, resolved_at = NOW()
            WHERE id = $2
            "#,
            user_id,
            thread_id
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }
}

// ============================================================================
// Version History
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DraftVersion {
    pub id: Uuid,
    pub draft_id: Uuid,
    pub version: i32,
    pub subject: String,
    pub body: String,
    pub created_by: Uuid,
    pub created_by_name: String,
    pub created_at: DateTime<Utc>,
    pub change_summary: Option<String>,
}

pub struct VersionHistoryService {
    pool: PgPool,
}

impl VersionHistoryService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn save_version(
        &self,
        draft_id: Uuid,
        user_id: Uuid,
        change_summary: Option<&str>,
    ) -> Result<DraftVersion, RealtimeError> {
        let draft = sqlx::query!(
            "SELECT subject, body, version FROM collaborative_drafts WHERE id = $1",
            draft_id
        )
        .fetch_one(&self.pool)
        .await?;

        let version_id = Uuid::new_v4();

        sqlx::query!(
            r#"
            INSERT INTO draft_versions (id, draft_id, version, subject, body,
                                       created_by, change_summary, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
            "#,
            version_id,
            draft_id,
            draft.version,
            draft.subject,
            draft.body,
            user_id,
            change_summary
        )
        .execute(&self.pool)
        .await?;

        self.get_version(version_id).await
    }

    pub async fn get_version(&self, version_id: Uuid) -> Result<DraftVersion, RealtimeError> {
        let version = sqlx::query!(
            r#"
            SELECT v.id, v.draft_id, v.version, v.subject, v.body,
                   v.created_by, u.name as created_by_name, v.created_at, v.change_summary
            FROM draft_versions v
            JOIN users u ON v.created_by = u.id
            WHERE v.id = $1
            "#,
            version_id
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(DraftVersion {
            id: version.id,
            draft_id: version.draft_id,
            version: version.version,
            subject: version.subject,
            body: version.body,
            created_by: version.created_by,
            created_by_name: version.created_by_name,
            created_at: version.created_at,
            change_summary: version.change_summary,
        })
    }

    pub async fn get_history(
        &self,
        draft_id: Uuid,
        limit: i32,
    ) -> Result<Vec<DraftVersion>, RealtimeError> {
        let versions = sqlx::query!(
            r#"
            SELECT v.id, v.draft_id, v.version, v.subject, v.body,
                   v.created_by, u.name as created_by_name, v.created_at, v.change_summary
            FROM draft_versions v
            JOIN users u ON v.created_by = u.id
            WHERE v.draft_id = $1
            ORDER BY v.version DESC
            LIMIT $2
            "#,
            draft_id,
            limit as i64
        )
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|row| DraftVersion {
            id: row.id,
            draft_id: row.draft_id,
            version: row.version,
            subject: row.subject,
            body: row.body,
            created_by: row.created_by,
            created_by_name: row.created_by_name,
            created_at: row.created_at,
            change_summary: row.change_summary,
        })
        .collect();

        Ok(versions)
    }

    pub async fn restore_version(
        &self,
        version_id: Uuid,
        user_id: Uuid,
    ) -> Result<i32, RealtimeError> {
        let version = self.get_version(version_id).await?;

        // Save current state as new version first
        self.save_version(version.draft_id, user_id, Some("Before restore")).await?;

        // Restore the old version
        let new_version = sqlx::query_scalar!(
            r#"
            UPDATE collaborative_drafts
            SET subject = $1, body = $2, version = version + 1, updated_at = NOW()
            WHERE id = $3
            RETURNING version
            "#,
            version.subject,
            version.body,
            version.draft_id
        )
        .fetch_one(&self.pool)
        .await?;

        // Save restored state
        self.save_version(
            version.draft_id,
            user_id,
            Some(&format!("Restored from version {}", version.version)),
        )
        .await?;

        Ok(new_version)
    }

    pub async fn compare_versions(
        &self,
        version_a: Uuid,
        version_b: Uuid,
    ) -> Result<VersionDiff, RealtimeError> {
        let a = self.get_version(version_a).await?;
        let b = self.get_version(version_b).await?;

        Ok(VersionDiff {
            version_a: a.version,
            version_b: b.version,
            subject_changed: a.subject != b.subject,
            body_diff: compute_diff(&a.body, &b.body),
            created_at_a: a.created_at,
            created_at_b: b.created_at,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionDiff {
    pub version_a: i32,
    pub version_b: i32,
    pub subject_changed: bool,
    pub body_diff: Vec<DiffChunk>,
    pub created_at_a: DateTime<Utc>,
    pub created_at_b: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffChunk {
    pub diff_type: DiffType,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiffType {
    Equal,
    Insert,
    Delete,
}

fn compute_diff(old: &str, new: &str) -> Vec<DiffChunk> {
    // Simple word-based diff (in production, use a proper diff library)
    let old_words: Vec<&str> = old.split_whitespace().collect();
    let new_words: Vec<&str> = new.split_whitespace().collect();

    let mut result = Vec::new();
    let mut i = 0;
    let mut j = 0;

    while i < old_words.len() || j < new_words.len() {
        if i >= old_words.len() {
            result.push(DiffChunk {
                diff_type: DiffType::Insert,
                content: new_words[j..].join(" "),
            });
            break;
        }
        if j >= new_words.len() {
            result.push(DiffChunk {
                diff_type: DiffType::Delete,
                content: old_words[i..].join(" "),
            });
            break;
        }

        if old_words[i] == new_words[j] {
            result.push(DiffChunk {
                diff_type: DiffType::Equal,
                content: old_words[i].to_string(),
            });
            i += 1;
            j += 1;
        } else {
            result.push(DiffChunk {
                diff_type: DiffType::Delete,
                content: old_words[i].to_string(),
            });
            result.push(DiffChunk {
                diff_type: DiffType::Insert,
                content: new_words[j].to_string(),
            });
            i += 1;
            j += 1;
        }
    }

    result
}
