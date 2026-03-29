// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Team Collaboration Features
//!
//! Shared inboxes, email assignment, internal notes, and team communication

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;

// ============================================================================
// Shared Inbox Service
// ============================================================================

pub struct SharedInboxService {
    pool: PgPool,
}

impl SharedInboxService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Create a shared inbox
    pub async fn create_shared_inbox(
        &self,
        org_id: Uuid,
        config: SharedInboxConfig,
    ) -> Result<SharedInbox, CollaborationError> {
        let inbox_id = Uuid::new_v4();

        sqlx::query(
            r#"
            INSERT INTO shared_inboxes (id, org_id, name, email_address, description,
                                        auto_assign, round_robin, sla_hours, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
            "#,
        )
        .bind(inbox_id)
        .bind(org_id)
        .bind(&config.name)
        .bind(&config.email_address)
        .bind(&config.description)
        .bind(config.auto_assign)
        .bind(config.round_robin)
        .bind(config.sla_hours)
        .execute(&self.pool)
        .await
        .map_err(|e| CollaborationError::Database(e.to_string()))?;

        Ok(SharedInbox {
            id: inbox_id,
            org_id,
            name: config.name,
            email_address: config.email_address,
            description: config.description,
            auto_assign: config.auto_assign,
            round_robin: config.round_robin,
            sla_hours: config.sla_hours,
            member_count: 0,
            unassigned_count: 0,
            created_at: Utc::now(),
        })
    }

    /// Add member to shared inbox
    pub async fn add_member(
        &self,
        inbox_id: Uuid,
        user_id: Uuid,
        role: InboxRole,
    ) -> Result<(), CollaborationError> {
        sqlx::query(
            r#"
            INSERT INTO shared_inbox_members (inbox_id, user_id, role, added_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (inbox_id, user_id) DO UPDATE SET role = $3
            "#,
        )
        .bind(inbox_id)
        .bind(user_id)
        .bind(role.to_string())
        .execute(&self.pool)
        .await
        .map_err(|e| CollaborationError::Database(e.to_string()))?;

        Ok(())
    }

    /// Get shared inbox statistics
    pub async fn get_stats(&self, inbox_id: Uuid) -> Result<InboxStats, CollaborationError> {
        let stats: InboxStats = sqlx::query_as(
            r#"
            SELECT
                COUNT(*) as total_emails,
                COUNT(*) FILTER (WHERE assigned_to IS NULL) as unassigned,
                COUNT(*) FILTER (WHERE status = 'open') as open,
                COUNT(*) FILTER (WHERE status = 'pending') as pending,
                COUNT(*) FILTER (WHERE status = 'resolved') as resolved,
                COUNT(*) FILTER (WHERE sla_due_at < NOW() AND status != 'resolved') as overdue,
                AVG(EXTRACT(EPOCH FROM (resolved_at - created_at)) / 3600)
                    FILTER (WHERE resolved_at IS NOT NULL) as avg_resolution_hours
            FROM shared_inbox_emails
            WHERE inbox_id = $1
            "#,
        )
        .bind(inbox_id)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| CollaborationError::Database(e.to_string()))?;

        Ok(stats)
    }
}

// ============================================================================
// Assignment Service
// ============================================================================

pub struct AssignmentService {
    pool: PgPool,
}

impl AssignmentService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Assign email to a user
    pub async fn assign(
        &self,
        inbox_id: Uuid,
        email_id: Uuid,
        assignee_id: Uuid,
        assigned_by: Uuid,
    ) -> Result<(), CollaborationError> {
        sqlx::query(
            r#"
            UPDATE shared_inbox_emails
            SET assigned_to = $1, assigned_at = NOW(), assigned_by = $2, status = 'open'
            WHERE email_id = $3 AND inbox_id = $4
            "#,
        )
        .bind(assignee_id)
        .bind(assigned_by)
        .bind(email_id)
        .bind(inbox_id)
        .execute(&self.pool)
        .await
        .map_err(|e| CollaborationError::Database(e.to_string()))?;

        Ok(())
    }

    /// Auto-assign using round-robin
    pub async fn auto_assign_round_robin(
        &self,
        inbox_id: Uuid,
        email_id: Uuid,
    ) -> Result<Uuid, CollaborationError> {
        let next_assignee: Option<(Uuid,)> = sqlx::query_as(
            r#"
            SELECT user_id FROM shared_inbox_members
            WHERE inbox_id = $1 AND is_available = true
            ORDER BY last_assigned_at ASC NULLS FIRST
            LIMIT 1
            "#,
        )
        .bind(inbox_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| CollaborationError::Database(e.to_string()))?;

        let assignee_id = next_assignee
            .ok_or(CollaborationError::NoAvailableAssignee)?
            .0;

        sqlx::query(
            r#"
            UPDATE shared_inbox_emails
            SET assigned_to = $1, assigned_at = NOW(), status = 'open'
            WHERE email_id = $2 AND inbox_id = $3
            "#,
        )
        .bind(assignee_id)
        .bind(email_id)
        .bind(inbox_id)
        .execute(&self.pool)
        .await
        .map_err(|e| CollaborationError::Database(e.to_string()))?;

        sqlx::query(
            "UPDATE shared_inbox_members SET last_assigned_at = NOW() WHERE inbox_id = $1 AND user_id = $2",
        )
        .bind(inbox_id)
        .bind(assignee_id)
        .execute(&self.pool)
        .await
        .ok();

        Ok(assignee_id)
    }

    /// Update email status
    pub async fn update_status(
        &self,
        inbox_id: Uuid,
        email_id: Uuid,
        status: EmailStatus,
    ) -> Result<(), CollaborationError> {
        let resolved_at = if status == EmailStatus::Resolved {
            Some(Utc::now())
        } else {
            None
        };

        sqlx::query(
            "UPDATE shared_inbox_emails SET status = $1, resolved_at = $2 WHERE email_id = $3 AND inbox_id = $4",
        )
        .bind(status.to_string())
        .bind(resolved_at)
        .bind(email_id)
        .bind(inbox_id)
        .execute(&self.pool)
        .await
        .map_err(|e| CollaborationError::Database(e.to_string()))?;

        Ok(())
    }
}

// ============================================================================
// Collision Detection
// ============================================================================

pub struct CollisionDetector {
    pool: PgPool,
}

impl CollisionDetector {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Mark user as viewing an email, returns other viewer if collision
    pub async fn start_viewing(
        &self,
        email_id: Uuid,
        user_id: Uuid,
    ) -> Result<Option<Uuid>, CollaborationError> {
        let current_viewer: Option<(Uuid, DateTime<Utc>)> = sqlx::query_as(
            r#"
            SELECT viewing_by, viewing_since FROM shared_inbox_emails
            WHERE email_id = $1 AND viewing_by IS NOT NULL
            AND viewing_since > NOW() - INTERVAL '5 minutes'
            "#,
        )
        .bind(email_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| CollaborationError::Database(e.to_string()))?;

        if let Some((viewer, _)) = current_viewer {
            if viewer != user_id {
                return Ok(Some(viewer));
            }
        }

        sqlx::query(
            "UPDATE shared_inbox_emails SET viewing_by = $1, viewing_since = NOW() WHERE email_id = $2",
        )
        .bind(user_id)
        .bind(email_id)
        .execute(&self.pool)
        .await
        .map_err(|e| CollaborationError::Database(e.to_string()))?;

        Ok(None)
    }

    /// Check if someone is composing a reply
    pub async fn start_replying(
        &self,
        email_id: Uuid,
        user_id: Uuid,
    ) -> Result<Option<Uuid>, CollaborationError> {
        let current_replier: Option<(Uuid,)> = sqlx::query_as(
            r#"
            SELECT replying_by FROM shared_inbox_emails
            WHERE email_id = $1 AND replying_by IS NOT NULL
            AND replying_since > NOW() - INTERVAL '10 minutes'
            "#,
        )
        .bind(email_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| CollaborationError::Database(e.to_string()))?;

        if let Some((replier,)) = current_replier {
            if replier != user_id {
                return Ok(Some(replier));
            }
        }

        sqlx::query(
            "UPDATE shared_inbox_emails SET replying_by = $1, replying_since = NOW() WHERE email_id = $2",
        )
        .bind(user_id)
        .bind(email_id)
        .execute(&self.pool)
        .await
        .map_err(|e| CollaborationError::Database(e.to_string()))?;

        Ok(None)
    }
}

// ============================================================================
// Internal Notes
// ============================================================================

pub struct InternalNotes {
    pool: PgPool,
}

impl InternalNotes {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Add internal note to an email
    pub async fn add_note(
        &self,
        email_id: Uuid,
        user_id: Uuid,
        content: &str,
        mentions: &[Uuid],
    ) -> Result<Note, CollaborationError> {
        let note_id = Uuid::new_v4();

        sqlx::query(
            r#"
            INSERT INTO internal_notes (id, email_id, user_id, content, mentions, created_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
            "#,
        )
        .bind(note_id)
        .bind(email_id)
        .bind(user_id)
        .bind(content)
        .bind(mentions)
        .execute(&self.pool)
        .await
        .map_err(|e| CollaborationError::Database(e.to_string()))?;

        Ok(Note {
            id: note_id,
            email_id,
            user_id,
            content: content.to_string(),
            mentions: mentions.to_vec(),
            created_at: Utc::now(),
        })
    }

    /// Get notes for an email
    pub async fn get_notes(&self, email_id: Uuid) -> Result<Vec<NoteWithAuthor>, CollaborationError> {
        let notes: Vec<NoteWithAuthor> = sqlx::query_as(
            r#"
            SELECT n.id, n.email_id, n.user_id, n.content, n.mentions, n.created_at,
                   u.name as author_name, u.avatar_url as author_avatar
            FROM internal_notes n
            JOIN users u ON u.id = n.user_id
            WHERE n.email_id = $1
            ORDER BY n.created_at ASC
            "#,
        )
        .bind(email_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CollaborationError::Database(e.to_string()))?;

        Ok(notes)
    }
}

// ============================================================================
// SLA Tracking
// ============================================================================

pub struct SLATracker {
    pool: PgPool,
}

impl SLATracker {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Get emails approaching SLA breach
    pub async fn get_approaching_breach(
        &self,
        inbox_id: Uuid,
        within_hours: i32,
    ) -> Result<Vec<SLAWarning>, CollaborationError> {
        let warnings: Vec<SLAWarning> = sqlx::query_as(
            r#"
            SELECT se.email_id, e.subject, se.assigned_to, se.sla_due_at,
                   EXTRACT(EPOCH FROM (se.sla_due_at - NOW())) / 3600 as hours_remaining
            FROM shared_inbox_emails se
            JOIN emails e ON e.id = se.email_id
            WHERE se.inbox_id = $1
            AND se.status != 'resolved'
            AND se.sla_due_at IS NOT NULL
            AND se.sla_due_at < NOW() + INTERVAL '1 hour' * $2
            ORDER BY se.sla_due_at ASC
            "#,
        )
        .bind(inbox_id)
        .bind(within_hours)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CollaborationError::Database(e.to_string()))?;

        Ok(warnings)
    }

    /// Escalate overdue emails
    pub async fn escalate_overdue(&self, inbox_id: Uuid) -> Result<Vec<Uuid>, CollaborationError> {
        let overdue: Vec<(Uuid,)> = sqlx::query_as(
            r#"
            SELECT email_id FROM shared_inbox_emails
            WHERE inbox_id = $1 AND status != 'resolved'
            AND sla_due_at < NOW() AND escalated = false
            "#,
        )
        .bind(inbox_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CollaborationError::Database(e.to_string()))?;

        let email_ids: Vec<Uuid> = overdue.into_iter().map(|(id,)| id).collect();

        if !email_ids.is_empty() {
            sqlx::query(
                "UPDATE shared_inbox_emails SET escalated = true, priority = 3 WHERE email_id = ANY($1)",
            )
            .bind(&email_ids)
            .execute(&self.pool)
            .await
            .map_err(|e| CollaborationError::Database(e.to_string()))?;
        }

        Ok(email_ids)
    }
}

// ============================================================================
// Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedInboxConfig {
    pub name: String,
    pub email_address: String,
    pub description: Option<String>,
    pub auto_assign: bool,
    pub round_robin: bool,
    pub sla_hours: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedInbox {
    pub id: Uuid,
    pub org_id: Uuid,
    pub name: String,
    pub email_address: String,
    pub description: Option<String>,
    pub auto_assign: bool,
    pub round_robin: bool,
    pub sla_hours: Option<i32>,
    pub member_count: i32,
    pub unassigned_count: i32,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum InboxRole {
    Admin,
    Member,
    Viewer,
}

impl ToString for InboxRole {
    fn to_string(&self) -> String {
        match self {
            InboxRole::Admin => "admin".to_string(),
            InboxRole::Member => "member".to_string(),
            InboxRole::Viewer => "viewer".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct InboxStats {
    pub total_emails: i64,
    pub unassigned: i64,
    pub open: i64,
    pub pending: i64,
    pub resolved: i64,
    pub overdue: i64,
    pub avg_resolution_hours: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EmailStatus {
    Open,
    Pending,
    Resolved,
    Closed,
}

impl ToString for EmailStatus {
    fn to_string(&self) -> String {
        match self {
            EmailStatus::Open => "open".to_string(),
            EmailStatus::Pending => "pending".to_string(),
            EmailStatus::Resolved => "resolved".to_string(),
            EmailStatus::Closed => "closed".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Note {
    pub id: Uuid,
    pub email_id: Uuid,
    pub user_id: Uuid,
    pub content: String,
    pub mentions: Vec<Uuid>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct NoteWithAuthor {
    pub id: Uuid,
    pub email_id: Uuid,
    pub user_id: Uuid,
    pub content: String,
    pub mentions: Vec<Uuid>,
    pub created_at: DateTime<Utc>,
    pub author_name: String,
    pub author_avatar: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct SLAWarning {
    pub email_id: Uuid,
    pub subject: String,
    pub assigned_to: Option<Uuid>,
    pub sla_due_at: DateTime<Utc>,
    pub hours_remaining: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum CollaborationError {
    #[error("Database error: {0}")]
    Database(String),
    #[error("Already assigned to another user")]
    AlreadyAssigned,
    #[error("No available assignee")]
    NoAvailableAssignee,
}
