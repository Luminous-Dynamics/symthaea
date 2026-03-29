// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Team Collaboration & Shared Mailboxes Module
//!
//! Provides shared inbox access, internal notes, @mentions,
//! collision detection, email assignment, and team activity tracking.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum TeamError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("Shared mailbox not found: {0}")]
    MailboxNotFound(Uuid),
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    #[error("User not found: {0}")]
    UserNotFound(Uuid),
    #[error("Assignment conflict: {0}")]
    AssignmentConflict(String),
}

// ============================================================================
// Shared Mailboxes
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedMailbox {
    pub id: Uuid,
    pub name: String,
    pub email_address: String,
    pub description: Option<String>,
    pub team_id: Uuid,
    pub created_at: DateTime<Utc>,
    pub settings: SharedMailboxSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedMailboxSettings {
    pub auto_assign: bool,
    pub round_robin: bool,
    pub notify_all_members: bool,
    pub require_assignment: bool,
    pub sla_hours: Option<i32>,
    pub default_folder_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedMailboxMember {
    pub user_id: Uuid,
    pub user_name: String,
    pub user_email: String,
    pub role: MailboxRole,
    pub added_at: DateTime<Utc>,
    pub added_by: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MailboxRole {
    Owner,
    Admin,
    Member,
    ReadOnly,
}

pub struct SharedMailboxService {
    pool: PgPool,
}

impl SharedMailboxService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn create_mailbox(
        &self,
        team_id: Uuid,
        creator_id: Uuid,
        name: &str,
        email_address: &str,
        description: Option<&str>,
    ) -> Result<SharedMailbox, TeamError> {
        let settings = SharedMailboxSettings {
            auto_assign: false,
            round_robin: false,
            notify_all_members: true,
            require_assignment: false,
            sla_hours: None,
            default_folder_id: None,
        };

        let mailbox = sqlx::query_as!(
            SharedMailbox,
            r#"
            INSERT INTO shared_mailboxes (team_id, name, email_address, description,
                                         settings, created_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
            RETURNING id, name, email_address, description, team_id, created_at,
                      settings as "settings: sqlx::types::Json<SharedMailboxSettings>"
            "#,
            team_id,
            name,
            email_address,
            description,
            sqlx::types::Json(&settings) as _
        )
        .fetch_one(&self.pool)
        .await
        .map(|row| SharedMailbox {
            id: row.id,
            name: row.name,
            email_address: row.email_address,
            description: row.description,
            team_id: row.team_id,
            created_at: row.created_at,
            settings,
        })?;

        // Add creator as owner
        self.add_member(mailbox.id, creator_id, creator_id, MailboxRole::Owner).await?;

        Ok(mailbox)
    }

    pub async fn get_mailbox(&self, mailbox_id: Uuid) -> Result<SharedMailbox, TeamError> {
        sqlx::query!(
            r#"
            SELECT id, name, email_address, description, team_id, created_at, settings
            FROM shared_mailboxes
            WHERE id = $1
            "#,
            mailbox_id
        )
        .fetch_optional(&self.pool)
        .await?
        .map(|row| SharedMailbox {
            id: row.id,
            name: row.name,
            email_address: row.email_address,
            description: row.description,
            team_id: row.team_id,
            created_at: row.created_at,
            settings: serde_json::from_value(row.settings).unwrap_or_default(),
        })
        .ok_or(TeamError::MailboxNotFound(mailbox_id))
    }

    pub async fn list_user_mailboxes(&self, user_id: Uuid) -> Result<Vec<SharedMailbox>, TeamError> {
        let mailboxes = sqlx::query!(
            r#"
            SELECT m.id, m.name, m.email_address, m.description, m.team_id,
                   m.created_at, m.settings
            FROM shared_mailboxes m
            JOIN shared_mailbox_members mem ON m.id = mem.mailbox_id
            WHERE mem.user_id = $1
            ORDER BY m.name
            "#,
            user_id
        )
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|row| SharedMailbox {
            id: row.id,
            name: row.name,
            email_address: row.email_address,
            description: row.description,
            team_id: row.team_id,
            created_at: row.created_at,
            settings: serde_json::from_value(row.settings).unwrap_or_default(),
        })
        .collect();

        Ok(mailboxes)
    }

    pub async fn add_member(
        &self,
        mailbox_id: Uuid,
        user_id: Uuid,
        added_by: Uuid,
        role: MailboxRole,
    ) -> Result<(), TeamError> {
        sqlx::query!(
            r#"
            INSERT INTO shared_mailbox_members (mailbox_id, user_id, role, added_at, added_by)
            VALUES ($1, $2, $3, NOW(), $4)
            ON CONFLICT (mailbox_id, user_id) DO UPDATE SET
                role = EXCLUDED.role
            "#,
            mailbox_id,
            user_id,
            serde_json::to_string(&role).unwrap(),
            added_by
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_members(&self, mailbox_id: Uuid) -> Result<Vec<SharedMailboxMember>, TeamError> {
        let members = sqlx::query!(
            r#"
            SELECT mem.user_id, u.name as user_name, u.email as user_email,
                   mem.role, mem.added_at, mem.added_by
            FROM shared_mailbox_members mem
            JOIN users u ON mem.user_id = u.id
            WHERE mem.mailbox_id = $1
            ORDER BY mem.added_at
            "#,
            mailbox_id
        )
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|row| SharedMailboxMember {
            user_id: row.user_id,
            user_name: row.user_name,
            user_email: row.user_email,
            role: serde_json::from_str(&row.role).unwrap_or(MailboxRole::Member),
            added_at: row.added_at,
            added_by: row.added_by,
        })
        .collect();

        Ok(members)
    }

    pub async fn check_access(
        &self,
        mailbox_id: Uuid,
        user_id: Uuid,
    ) -> Result<MailboxRole, TeamError> {
        let member = sqlx::query!(
            r#"
            SELECT role
            FROM shared_mailbox_members
            WHERE mailbox_id = $1 AND user_id = $2
            "#,
            mailbox_id,
            user_id
        )
        .fetch_optional(&self.pool)
        .await?;

        member
            .map(|m| serde_json::from_str(&m.role).unwrap_or(MailboxRole::ReadOnly))
            .ok_or(TeamError::PermissionDenied(
                "User is not a member of this mailbox".to_string(),
            ))
    }

    pub async fn get_mailbox_stats(&self, mailbox_id: Uuid) -> Result<MailboxStats, TeamError> {
        let stats = sqlx::query!(
            r#"
            SELECT
                COUNT(*) FILTER (WHERE NOT is_read) as unread_count,
                COUNT(*) FILTER (WHERE assigned_to IS NULL AND NOT is_read) as unassigned_count,
                COUNT(*) FILTER (WHERE assigned_to IS NOT NULL AND NOT is_resolved) as in_progress_count,
                COUNT(*) FILTER (WHERE is_resolved) as resolved_today,
                AVG(EXTRACT(EPOCH FROM (resolved_at - received_at)) / 3600)
                    FILTER (WHERE resolved_at IS NOT NULL) as avg_resolution_hours
            FROM shared_mailbox_emails
            WHERE mailbox_id = $1 AND received_at > NOW() - INTERVAL '24 hours'
            "#,
            mailbox_id
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(MailboxStats {
            unread_count: stats.unread_count.unwrap_or(0) as i32,
            unassigned_count: stats.unassigned_count.unwrap_or(0) as i32,
            in_progress_count: stats.in_progress_count.unwrap_or(0) as i32,
            resolved_today: stats.resolved_today.unwrap_or(0) as i32,
            avg_resolution_hours: stats.avg_resolution_hours.map(|h| h as f32),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MailboxStats {
    pub unread_count: i32,
    pub unassigned_count: i32,
    pub in_progress_count: i32,
    pub resolved_today: i32,
    pub avg_resolution_hours: Option<f32>,
}

// ============================================================================
// Internal Notes
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternalNote {
    pub id: Uuid,
    pub email_id: Uuid,
    pub author_id: Uuid,
    pub author_name: String,
    pub content: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: Option<DateTime<Utc>>,
    pub mentions: Vec<Uuid>,
    pub is_pinned: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateNoteInput {
    pub content: String,
    pub mentions: Vec<Uuid>,
    pub is_pinned: bool,
}

pub struct NotesService {
    pool: PgPool,
}

impl NotesService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn create_note(
        &self,
        email_id: Uuid,
        author_id: Uuid,
        input: CreateNoteInput,
    ) -> Result<InternalNote, TeamError> {
        let note_id = Uuid::new_v4();

        sqlx::query!(
            r#"
            INSERT INTO internal_notes (id, email_id, author_id, content, mentions,
                                       is_pinned, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, NOW())
            "#,
            note_id,
            email_id,
            author_id,
            input.content,
            &input.mentions,
            input.is_pinned
        )
        .execute(&self.pool)
        .await?;

        // Create mention notifications
        for user_id in &input.mentions {
            self.create_mention_notification(note_id, *user_id, author_id).await?;
        }

        self.get_note(note_id).await
    }

    pub async fn get_note(&self, note_id: Uuid) -> Result<InternalNote, TeamError> {
        let note = sqlx::query!(
            r#"
            SELECT n.id, n.email_id, n.author_id, u.name as author_name,
                   n.content, n.created_at, n.updated_at, n.mentions, n.is_pinned
            FROM internal_notes n
            JOIN users u ON n.author_id = u.id
            WHERE n.id = $1
            "#,
            note_id
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(InternalNote {
            id: note.id,
            email_id: note.email_id,
            author_id: note.author_id,
            author_name: note.author_name,
            content: note.content,
            created_at: note.created_at,
            updated_at: note.updated_at,
            mentions: note.mentions,
            is_pinned: note.is_pinned,
        })
    }

    pub async fn get_notes_for_email(&self, email_id: Uuid) -> Result<Vec<InternalNote>, TeamError> {
        let notes = sqlx::query!(
            r#"
            SELECT n.id, n.email_id, n.author_id, u.name as author_name,
                   n.content, n.created_at, n.updated_at, n.mentions, n.is_pinned
            FROM internal_notes n
            JOIN users u ON n.author_id = u.id
            WHERE n.email_id = $1
            ORDER BY n.is_pinned DESC, n.created_at DESC
            "#,
            email_id
        )
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|row| InternalNote {
            id: row.id,
            email_id: row.email_id,
            author_id: row.author_id,
            author_name: row.author_name,
            content: row.content,
            created_at: row.created_at,
            updated_at: row.updated_at,
            mentions: row.mentions,
            is_pinned: row.is_pinned,
        })
        .collect();

        Ok(notes)
    }

    pub async fn update_note(
        &self,
        note_id: Uuid,
        user_id: Uuid,
        content: &str,
    ) -> Result<InternalNote, TeamError> {
        // Verify ownership
        let note = sqlx::query!(
            "SELECT author_id FROM internal_notes WHERE id = $1",
            note_id
        )
        .fetch_one(&self.pool)
        .await?;

        if note.author_id != user_id {
            return Err(TeamError::PermissionDenied(
                "Only the author can edit this note".to_string(),
            ));
        }

        sqlx::query!(
            r#"
            UPDATE internal_notes
            SET content = $1, updated_at = NOW()
            WHERE id = $2
            "#,
            content,
            note_id
        )
        .execute(&self.pool)
        .await?;

        self.get_note(note_id).await
    }

    pub async fn delete_note(&self, note_id: Uuid, user_id: Uuid) -> Result<(), TeamError> {
        let result = sqlx::query!(
            r#"
            DELETE FROM internal_notes
            WHERE id = $1 AND author_id = $2
            "#,
            note_id,
            user_id
        )
        .execute(&self.pool)
        .await?;

        if result.rows_affected() == 0 {
            return Err(TeamError::PermissionDenied(
                "Cannot delete note: not found or not the author".to_string(),
            ));
        }

        Ok(())
    }

    async fn create_mention_notification(
        &self,
        note_id: Uuid,
        mentioned_user_id: Uuid,
        author_id: Uuid,
    ) -> Result<(), TeamError> {
        sqlx::query!(
            r#"
            INSERT INTO mention_notifications (note_id, user_id, mentioned_by, created_at)
            VALUES ($1, $2, $3, NOW())
            "#,
            note_id,
            mentioned_user_id,
            author_id
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_unread_mentions(&self, user_id: Uuid) -> Result<Vec<MentionNotification>, TeamError> {
        let mentions = sqlx::query!(
            r#"
            SELECT mn.id, mn.note_id, n.email_id, n.content as note_content,
                   u.name as mentioned_by_name, mn.created_at, mn.read_at
            FROM mention_notifications mn
            JOIN internal_notes n ON mn.note_id = n.id
            JOIN users u ON mn.mentioned_by = u.id
            WHERE mn.user_id = $1 AND mn.read_at IS NULL
            ORDER BY mn.created_at DESC
            "#,
            user_id
        )
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|row| MentionNotification {
            id: row.id,
            note_id: row.note_id,
            email_id: row.email_id,
            note_preview: row.note_content.chars().take(100).collect(),
            mentioned_by_name: row.mentioned_by_name,
            created_at: row.created_at,
            read_at: row.read_at,
        })
        .collect();

        Ok(mentions)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MentionNotification {
    pub id: Uuid,
    pub note_id: Uuid,
    pub email_id: Uuid,
    pub note_preview: String,
    pub mentioned_by_name: String,
    pub created_at: DateTime<Utc>,
    pub read_at: Option<DateTime<Utc>>,
}

// ============================================================================
// Collision Detection
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailPresence {
    pub email_id: Uuid,
    pub user_id: Uuid,
    pub user_name: String,
    pub activity: PresenceActivity,
    pub started_at: DateTime<Utc>,
    pub last_seen: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PresenceActivity {
    Viewing,
    Replying,
    Forwarding,
    Editing,
}

pub struct CollisionService {
    pool: PgPool,
}

impl CollisionService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn set_presence(
        &self,
        email_id: Uuid,
        user_id: Uuid,
        activity: PresenceActivity,
    ) -> Result<(), TeamError> {
        sqlx::query!(
            r#"
            INSERT INTO email_presence (email_id, user_id, activity, started_at, last_seen)
            VALUES ($1, $2, $3, NOW(), NOW())
            ON CONFLICT (email_id, user_id) DO UPDATE SET
                activity = EXCLUDED.activity,
                last_seen = NOW()
            "#,
            email_id,
            user_id,
            serde_json::to_string(&activity).unwrap()
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn clear_presence(&self, email_id: Uuid, user_id: Uuid) -> Result<(), TeamError> {
        sqlx::query!(
            r#"
            DELETE FROM email_presence
            WHERE email_id = $1 AND user_id = $2
            "#,
            email_id,
            user_id
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_presence(&self, email_id: Uuid) -> Result<Vec<EmailPresence>, TeamError> {
        // Clean up stale presence (older than 5 minutes)
        sqlx::query!(
            r#"
            DELETE FROM email_presence
            WHERE last_seen < NOW() - INTERVAL '5 minutes'
            "#
        )
        .execute(&self.pool)
        .await?;

        let presence = sqlx::query!(
            r#"
            SELECT p.email_id, p.user_id, u.name as user_name,
                   p.activity, p.started_at, p.last_seen
            FROM email_presence p
            JOIN users u ON p.user_id = u.id
            WHERE p.email_id = $1
            "#,
            email_id
        )
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|row| EmailPresence {
            email_id: row.email_id,
            user_id: row.user_id,
            user_name: row.user_name,
            activity: serde_json::from_str(&row.activity).unwrap_or(PresenceActivity::Viewing),
            started_at: row.started_at,
            last_seen: row.last_seen,
        })
        .collect();

        Ok(presence)
    }

    pub async fn check_collision(
        &self,
        email_id: Uuid,
        user_id: Uuid,
        intended_activity: PresenceActivity,
    ) -> Result<Option<CollisionWarning>, TeamError> {
        let others = self.get_presence(email_id).await?;

        let conflicting: Vec<_> = others
            .into_iter()
            .filter(|p| p.user_id != user_id)
            .filter(|p| {
                // Check for conflicting activities
                matches!(
                    (&p.activity, &intended_activity),
                    (PresenceActivity::Replying, PresenceActivity::Replying)
                        | (PresenceActivity::Replying, PresenceActivity::Forwarding)
                        | (PresenceActivity::Forwarding, PresenceActivity::Replying)
                        | (PresenceActivity::Editing, PresenceActivity::Editing)
                )
            })
            .collect();

        if conflicting.is_empty() {
            Ok(None)
        } else {
            Ok(Some(CollisionWarning {
                email_id,
                conflicting_users: conflicting
                    .iter()
                    .map(|p| ConflictingUser {
                        user_id: p.user_id,
                        user_name: p.user_name.clone(),
                        activity: p.activity.clone(),
                        since: p.started_at,
                    })
                    .collect(),
                message: format!(
                    "{} is already {} this email",
                    conflicting[0].user_name,
                    activity_verb(&conflicting[0].activity)
                ),
            }))
        }
    }
}

fn activity_verb(activity: &PresenceActivity) -> &'static str {
    match activity {
        PresenceActivity::Viewing => "viewing",
        PresenceActivity::Replying => "replying to",
        PresenceActivity::Forwarding => "forwarding",
        PresenceActivity::Editing => "editing",
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollisionWarning {
    pub email_id: Uuid,
    pub conflicting_users: Vec<ConflictingUser>,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictingUser {
    pub user_id: Uuid,
    pub user_name: String,
    pub activity: PresenceActivity,
    pub since: DateTime<Utc>,
}

// ============================================================================
// Email Assignment
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailAssignment {
    pub email_id: Uuid,
    pub assigned_to: Uuid,
    pub assigned_to_name: String,
    pub assigned_by: Uuid,
    pub assigned_at: DateTime<Utc>,
    pub status: AssignmentStatus,
    pub due_at: Option<DateTime<Utc>>,
    pub priority: AssignmentPriority,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssignmentStatus {
    Pending,
    InProgress,
    OnHold,
    Resolved,
    Escalated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssignmentPriority {
    Low,
    Normal,
    High,
    Urgent,
}

pub struct AssignmentService {
    pool: PgPool,
}

impl AssignmentService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn assign_email(
        &self,
        email_id: Uuid,
        assigned_to: Uuid,
        assigned_by: Uuid,
        priority: AssignmentPriority,
        due_at: Option<DateTime<Utc>>,
    ) -> Result<EmailAssignment, TeamError> {
        // Check if already assigned
        let existing = sqlx::query!(
            r#"
            SELECT assigned_to, status
            FROM email_assignments
            WHERE email_id = $1 AND status NOT IN ('Resolved')
            "#,
            email_id
        )
        .fetch_optional(&self.pool)
        .await?;

        if let Some(current) = existing {
            if current.assigned_to != assigned_to {
                // Log reassignment
                sqlx::query!(
                    r#"
                    INSERT INTO assignment_history (email_id, from_user, to_user,
                                                   changed_by, changed_at)
                    VALUES ($1, $2, $3, $4, NOW())
                    "#,
                    email_id,
                    current.assigned_to,
                    assigned_to,
                    assigned_by
                )
                .execute(&self.pool)
                .await?;
            }
        }

        sqlx::query!(
            r#"
            INSERT INTO email_assignments (email_id, assigned_to, assigned_by,
                                          status, priority, due_at, assigned_at)
            VALUES ($1, $2, $3, $4, $5, $6, NOW())
            ON CONFLICT (email_id) DO UPDATE SET
                assigned_to = EXCLUDED.assigned_to,
                assigned_by = EXCLUDED.assigned_by,
                priority = EXCLUDED.priority,
                due_at = EXCLUDED.due_at,
                assigned_at = NOW()
            "#,
            email_id,
            assigned_to,
            assigned_by,
            serde_json::to_string(&AssignmentStatus::Pending).unwrap(),
            serde_json::to_string(&priority).unwrap(),
            due_at
        )
        .execute(&self.pool)
        .await?;

        self.get_assignment(email_id).await
    }

    pub async fn get_assignment(&self, email_id: Uuid) -> Result<EmailAssignment, TeamError> {
        let assignment = sqlx::query!(
            r#"
            SELECT a.email_id, a.assigned_to, u.name as assigned_to_name,
                   a.assigned_by, a.assigned_at, a.status, a.due_at, a.priority
            FROM email_assignments a
            JOIN users u ON a.assigned_to = u.id
            WHERE a.email_id = $1
            "#,
            email_id
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(EmailAssignment {
            email_id: assignment.email_id,
            assigned_to: assignment.assigned_to,
            assigned_to_name: assignment.assigned_to_name,
            assigned_by: assignment.assigned_by,
            assigned_at: assignment.assigned_at,
            status: serde_json::from_str(&assignment.status).unwrap_or(AssignmentStatus::Pending),
            due_at: assignment.due_at,
            priority: serde_json::from_str(&assignment.priority).unwrap_or(AssignmentPriority::Normal),
        })
    }

    pub async fn get_user_assignments(
        &self,
        user_id: Uuid,
        include_resolved: bool,
    ) -> Result<Vec<EmailAssignment>, TeamError> {
        let assignments = sqlx::query!(
            r#"
            SELECT a.email_id, a.assigned_to, u.name as assigned_to_name,
                   a.assigned_by, a.assigned_at, a.status, a.due_at, a.priority
            FROM email_assignments a
            JOIN users u ON a.assigned_to = u.id
            WHERE a.assigned_to = $1
            AND ($2 OR a.status != 'Resolved')
            ORDER BY
                CASE a.priority
                    WHEN 'Urgent' THEN 1
                    WHEN 'High' THEN 2
                    WHEN 'Normal' THEN 3
                    ELSE 4
                END,
                a.due_at NULLS LAST,
                a.assigned_at
            "#,
            user_id,
            include_resolved
        )
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|row| EmailAssignment {
            email_id: row.email_id,
            assigned_to: row.assigned_to,
            assigned_to_name: row.assigned_to_name,
            assigned_by: row.assigned_by,
            assigned_at: row.assigned_at,
            status: serde_json::from_str(&row.status).unwrap_or(AssignmentStatus::Pending),
            due_at: row.due_at,
            priority: serde_json::from_str(&row.priority).unwrap_or(AssignmentPriority::Normal),
        })
        .collect();

        Ok(assignments)
    }

    pub async fn update_status(
        &self,
        email_id: Uuid,
        user_id: Uuid,
        status: AssignmentStatus,
    ) -> Result<EmailAssignment, TeamError> {
        sqlx::query!(
            r#"
            UPDATE email_assignments
            SET status = $1,
                resolved_at = CASE WHEN $1 = 'Resolved' THEN NOW() ELSE NULL END
            WHERE email_id = $2 AND assigned_to = $3
            "#,
            serde_json::to_string(&status).unwrap(),
            email_id,
            user_id
        )
        .execute(&self.pool)
        .await?;

        self.get_assignment(email_id).await
    }

    pub async fn auto_assign(
        &self,
        mailbox_id: Uuid,
        email_id: Uuid,
    ) -> Result<Option<EmailAssignment>, TeamError> {
        // Get mailbox settings
        let mailbox = sqlx::query!(
            "SELECT settings FROM shared_mailboxes WHERE id = $1",
            mailbox_id
        )
        .fetch_one(&self.pool)
        .await?;

        let settings: SharedMailboxSettings =
            serde_json::from_value(mailbox.settings).unwrap_or_default();

        if !settings.auto_assign {
            return Ok(None);
        }

        // Find next available member (round robin or least loaded)
        let assignee = if settings.round_robin {
            // Round robin: get member with oldest last assignment
            sqlx::query_scalar!(
                r#"
                SELECT mem.user_id
                FROM shared_mailbox_members mem
                LEFT JOIN (
                    SELECT assigned_to, MAX(assigned_at) as last_assigned
                    FROM email_assignments
                    GROUP BY assigned_to
                ) a ON mem.user_id = a.assigned_to
                WHERE mem.mailbox_id = $1 AND mem.role != 'ReadOnly'
                ORDER BY a.last_assigned NULLS FIRST
                LIMIT 1
                "#,
                mailbox_id
            )
            .fetch_optional(&self.pool)
            .await?
        } else {
            // Least loaded: get member with fewest open assignments
            sqlx::query_scalar!(
                r#"
                SELECT mem.user_id
                FROM shared_mailbox_members mem
                LEFT JOIN (
                    SELECT assigned_to, COUNT(*) as open_count
                    FROM email_assignments
                    WHERE status NOT IN ('Resolved')
                    GROUP BY assigned_to
                ) a ON mem.user_id = a.assigned_to
                WHERE mem.mailbox_id = $1 AND mem.role != 'ReadOnly'
                ORDER BY COALESCE(a.open_count, 0)
                LIMIT 1
                "#,
                mailbox_id
            )
            .fetch_optional(&self.pool)
            .await?
        };

        if let Some(Some(user_id)) = assignee {
            let assignment = self
                .assign_email(
                    email_id,
                    user_id,
                    user_id, // System auto-assign
                    AssignmentPriority::Normal,
                    settings.sla_hours.map(|h| Utc::now() + chrono::Duration::hours(h as i64)),
                )
                .await?;
            Ok(Some(assignment))
        } else {
            Ok(None)
        }
    }
}

// ============================================================================
// Team Activity Feed
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamActivity {
    pub id: Uuid,
    pub mailbox_id: Uuid,
    pub user_id: Uuid,
    pub user_name: String,
    pub activity_type: ActivityType,
    pub email_id: Option<Uuid>,
    pub email_subject: Option<String>,
    pub details: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivityType {
    EmailReceived,
    EmailSent,
    EmailAssigned,
    EmailReassigned,
    EmailResolved,
    NoteAdded,
    MentionCreated,
    StatusChanged,
    MemberJoined,
    MemberLeft,
}

pub struct ActivityService {
    pool: PgPool,
}

impl ActivityService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn log_activity(
        &self,
        mailbox_id: Uuid,
        user_id: Uuid,
        activity_type: ActivityType,
        email_id: Option<Uuid>,
        details: Option<&str>,
    ) -> Result<(), TeamError> {
        sqlx::query!(
            r#"
            INSERT INTO team_activity (mailbox_id, user_id, activity_type,
                                      email_id, details, created_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
            "#,
            mailbox_id,
            user_id,
            serde_json::to_string(&activity_type).unwrap(),
            email_id,
            details
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_activity(
        &self,
        mailbox_id: Uuid,
        limit: i32,
        offset: i32,
    ) -> Result<Vec<TeamActivity>, TeamError> {
        let activities = sqlx::query!(
            r#"
            SELECT a.id, a.mailbox_id, a.user_id, u.name as user_name,
                   a.activity_type, a.email_id, e.subject as email_subject,
                   a.details, a.created_at
            FROM team_activity a
            JOIN users u ON a.user_id = u.id
            LEFT JOIN emails e ON a.email_id = e.id
            WHERE a.mailbox_id = $1
            ORDER BY a.created_at DESC
            LIMIT $2 OFFSET $3
            "#,
            mailbox_id,
            limit as i64,
            offset as i64
        )
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|row| TeamActivity {
            id: row.id,
            mailbox_id: row.mailbox_id,
            user_id: row.user_id,
            user_name: row.user_name,
            activity_type: serde_json::from_str(&row.activity_type)
                .unwrap_or(ActivityType::EmailReceived),
            email_id: row.email_id,
            email_subject: row.email_subject,
            details: row.details,
            created_at: row.created_at,
        })
        .collect();

        Ok(activities)
    }

    pub async fn get_user_activity(
        &self,
        user_id: Uuid,
        limit: i32,
    ) -> Result<Vec<TeamActivity>, TeamError> {
        let activities = sqlx::query!(
            r#"
            SELECT a.id, a.mailbox_id, a.user_id, u.name as user_name,
                   a.activity_type, a.email_id, e.subject as email_subject,
                   a.details, a.created_at
            FROM team_activity a
            JOIN users u ON a.user_id = u.id
            LEFT JOIN emails e ON a.email_id = e.id
            WHERE a.user_id = $1
            ORDER BY a.created_at DESC
            LIMIT $2
            "#,
            user_id,
            limit as i64
        )
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|row| TeamActivity {
            id: row.id,
            mailbox_id: row.mailbox_id,
            user_id: row.user_id,
            user_name: row.user_name,
            activity_type: serde_json::from_str(&row.activity_type)
                .unwrap_or(ActivityType::EmailReceived),
            email_id: row.email_id,
            email_subject: row.email_subject,
            details: row.details,
            created_at: row.created_at,
        })
        .collect();

        Ok(activities)
    }
}

impl Default for SharedMailboxSettings {
    fn default() -> Self {
        Self {
            auto_assign: false,
            round_robin: false,
            notify_all_members: true,
            require_assignment: false,
            sla_hours: None,
            default_folder_id: None,
        }
    }
}
