// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Activity Tracking
//!
//! Log and track all interactions with contacts and accounts.

use chrono::{DateTime, NaiveDate, NaiveTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{FromRow, PgPool};
use uuid::Uuid;

use super::CrmError;

// ============================================================================
// Types
// ============================================================================

/// An activity (task, event, call, email, meeting)
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Activity {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub activity_type: String, // TASK, EVENT, CALL, EMAIL, MEETING, NOTE
    pub subject: String,
    pub description: Option<String>,
    pub status: String, // NOT_STARTED, IN_PROGRESS, COMPLETED, DEFERRED, CANCELLED
    pub priority: String, // LOW, NORMAL, HIGH, URGENT
    pub due_date: Option<NaiveDate>,
    pub due_time: Option<NaiveTime>,
    pub start_date: Option<NaiveDate>,
    pub start_time: Option<NaiveTime>,
    pub end_date: Option<NaiveDate>,
    pub end_time: Option<NaiveTime>,
    pub duration_minutes: Option<i32>,
    pub is_all_day: bool,
    pub location: Option<String>,
    // Related records
    pub account_id: Option<Uuid>,
    pub contact_id: Option<Uuid>,
    pub lead_id: Option<Uuid>,
    pub opportunity_id: Option<Uuid>,
    // Assignment
    pub owner_id: Option<Uuid>,
    pub assigned_to_id: Option<Uuid>,
    // Call/Email specific
    pub call_direction: Option<String>, // INBOUND, OUTBOUND
    pub call_result: Option<String>,
    pub email_message_id: Option<String>,
    // Completion
    pub completed_at: Option<DateTime<Utc>>,
    pub completed_by: Option<Uuid>,
    // Metadata
    pub is_reminder_set: bool,
    pub reminder_datetime: Option<DateTime<Utc>>,
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Activity type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivityType {
    Task,
    Event,
    Call,
    Email,
    Meeting,
    Note,
}

impl ActivityType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ActivityType::Task => "TASK",
            ActivityType::Event => "EVENT",
            ActivityType::Call => "CALL",
            ActivityType::Email => "EMAIL",
            ActivityType::Meeting => "MEETING",
            ActivityType::Note => "NOTE",
        }
    }
}

/// Create activity request
#[derive(Debug, Deserialize)]
pub struct CreateActivityRequest {
    pub activity_type: String,
    pub subject: String,
    pub description: Option<String>,
    pub priority: Option<String>,
    pub due_date: Option<NaiveDate>,
    pub due_time: Option<NaiveTime>,
    pub start_date: Option<NaiveDate>,
    pub start_time: Option<NaiveTime>,
    pub duration_minutes: Option<i32>,
    pub is_all_day: Option<bool>,
    pub location: Option<String>,
    pub account_id: Option<Uuid>,
    pub contact_id: Option<Uuid>,
    pub lead_id: Option<Uuid>,
    pub opportunity_id: Option<Uuid>,
    pub assigned_to_id: Option<Uuid>,
    pub tags: Option<Vec<String>>,
}

/// Log call request
#[derive(Debug, Deserialize)]
pub struct LogCallRequest {
    pub subject: String,
    pub description: Option<String>,
    pub contact_id: Option<Uuid>,
    pub account_id: Option<Uuid>,
    pub lead_id: Option<Uuid>,
    pub opportunity_id: Option<Uuid>,
    pub direction: String, // INBOUND, OUTBOUND
    pub result: Option<String>,
    pub duration_minutes: Option<i32>,
}

// ============================================================================
// Service
// ============================================================================

/// Activity Tracking Service
#[derive(Clone)]
pub struct ActivityService {
    pool: PgPool,
}

impl ActivityService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Create a new activity
    pub async fn create_activity(
        &self,
        tenant_id: Uuid,
        owner_id: Uuid,
        req: CreateActivityRequest,
    ) -> Result<Activity, CrmError> {
        let id = Uuid::new_v4();
        let tags = req.tags.unwrap_or_default();
        let priority = req.priority.unwrap_or_else(|| "NORMAL".to_string());
        let is_all_day = req.is_all_day.unwrap_or(false);

        sqlx::query(
            "INSERT INTO crm_activities (
                id, tenant_id, activity_type, subject, description, priority,
                due_date, due_time, start_date, start_time, duration_minutes,
                is_all_day, location, account_id, contact_id, lead_id, opportunity_id,
                owner_id, assigned_to_id, tags
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)",
        )
        .bind(id)
        .bind(tenant_id)
        .bind(&req.activity_type)
        .bind(&req.subject)
        .bind(&req.description)
        .bind(&priority)
        .bind(req.due_date)
        .bind(req.due_time)
        .bind(req.start_date)
        .bind(req.start_time)
        .bind(req.duration_minutes)
        .bind(is_all_day)
        .bind(&req.location)
        .bind(req.account_id)
        .bind(req.contact_id)
        .bind(req.lead_id)
        .bind(req.opportunity_id)
        .bind(owner_id)
        .bind(req.assigned_to_id.or(Some(owner_id)))
        .bind(&tags)
        .execute(&self.pool)
        .await
        .map_err(CrmError::Database)?;

        self.get_activity(id).await
    }

    /// Get an activity by ID
    pub async fn get_activity(&self, id: Uuid) -> Result<Activity, CrmError> {
        sqlx::query_as::<_, Activity>("SELECT * FROM crm_activities WHERE id = $1")
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(CrmError::Database)?
            .ok_or_else(|| CrmError::NotFound("Activity not found".into()))
    }

    /// List activities for a record
    pub async fn list_for_record(
        &self,
        tenant_id: Uuid,
        account_id: Option<Uuid>,
        contact_id: Option<Uuid>,
        lead_id: Option<Uuid>,
        opportunity_id: Option<Uuid>,
        limit: i32,
    ) -> Result<Vec<Activity>, CrmError> {
        let activities = if let Some(aid) = account_id {
            sqlx::query_as::<_, Activity>(
                "SELECT * FROM crm_activities
                 WHERE tenant_id = $1 AND account_id = $2
                 ORDER BY created_at DESC LIMIT $3",
            )
            .bind(tenant_id)
            .bind(aid)
            .bind(limit)
            .fetch_all(&self.pool)
            .await
        } else if let Some(cid) = contact_id {
            sqlx::query_as::<_, Activity>(
                "SELECT * FROM crm_activities
                 WHERE tenant_id = $1 AND contact_id = $2
                 ORDER BY created_at DESC LIMIT $3",
            )
            .bind(tenant_id)
            .bind(cid)
            .bind(limit)
            .fetch_all(&self.pool)
            .await
        } else if let Some(lid) = lead_id {
            sqlx::query_as::<_, Activity>(
                "SELECT * FROM crm_activities
                 WHERE tenant_id = $1 AND lead_id = $2
                 ORDER BY created_at DESC LIMIT $3",
            )
            .bind(tenant_id)
            .bind(lid)
            .bind(limit)
            .fetch_all(&self.pool)
            .await
        } else if let Some(oid) = opportunity_id {
            sqlx::query_as::<_, Activity>(
                "SELECT * FROM crm_activities
                 WHERE tenant_id = $1 AND opportunity_id = $2
                 ORDER BY created_at DESC LIMIT $3",
            )
            .bind(tenant_id)
            .bind(oid)
            .bind(limit)
            .fetch_all(&self.pool)
            .await
        } else {
            sqlx::query_as::<_, Activity>(
                "SELECT * FROM crm_activities
                 WHERE tenant_id = $1
                 ORDER BY created_at DESC LIMIT $2",
            )
            .bind(tenant_id)
            .bind(limit)
            .fetch_all(&self.pool)
            .await
        }
        .map_err(CrmError::Database)?;

        Ok(activities)
    }

    /// Get open tasks for user
    pub async fn get_open_tasks(
        &self,
        tenant_id: Uuid,
        user_id: Uuid,
    ) -> Result<Vec<Activity>, CrmError> {
        let tasks = sqlx::query_as::<_, Activity>(
            "SELECT * FROM crm_activities
             WHERE tenant_id = $1 AND assigned_to_id = $2
               AND activity_type = 'TASK'
               AND status NOT IN ('COMPLETED', 'CANCELLED')
             ORDER BY due_date, priority DESC",
        )
        .bind(tenant_id)
        .bind(user_id)
        .fetch_all(&self.pool)
        .await
        .map_err(CrmError::Database)?;

        Ok(tasks)
    }

    /// Get upcoming events
    pub async fn get_upcoming_events(
        &self,
        tenant_id: Uuid,
        user_id: Uuid,
        days: i32,
    ) -> Result<Vec<Activity>, CrmError> {
        let events = sqlx::query_as::<_, Activity>(
            "SELECT * FROM crm_activities
             WHERE tenant_id = $1 AND assigned_to_id = $2
               AND activity_type IN ('EVENT', 'MEETING')
               AND start_date >= CURRENT_DATE
               AND start_date <= CURRENT_DATE + $3 * INTERVAL '1 day'
             ORDER BY start_date, start_time",
        )
        .bind(tenant_id)
        .bind(user_id)
        .bind(days)
        .fetch_all(&self.pool)
        .await
        .map_err(CrmError::Database)?;

        Ok(events)
    }

    /// Log a phone call
    pub async fn log_call(
        &self,
        tenant_id: Uuid,
        user_id: Uuid,
        req: LogCallRequest,
    ) -> Result<Activity, CrmError> {
        let id = Uuid::new_v4();

        sqlx::query(
            "INSERT INTO crm_activities (
                id, tenant_id, activity_type, subject, description, status,
                contact_id, account_id, lead_id, opportunity_id,
                call_direction, call_result, duration_minutes,
                owner_id, assigned_to_id, completed_at, completed_by
            ) VALUES ($1, $2, 'CALL', $3, $4, 'COMPLETED', $5, $6, $7, $8, $9, $10, $11, $12, $12, NOW(), $12)",
        )
        .bind(id)
        .bind(tenant_id)
        .bind(&req.subject)
        .bind(&req.description)
        .bind(req.contact_id)
        .bind(req.account_id)
        .bind(req.lead_id)
        .bind(req.opportunity_id)
        .bind(&req.direction)
        .bind(&req.result)
        .bind(req.duration_minutes)
        .bind(user_id)
        .execute(&self.pool)
        .await
        .map_err(CrmError::Database)?;

        self.get_activity(id).await
    }

    /// Complete an activity
    pub async fn complete_activity(
        &self,
        id: Uuid,
        user_id: Uuid,
    ) -> Result<Activity, CrmError> {
        sqlx::query(
            "UPDATE crm_activities SET
                status = 'COMPLETED',
                completed_at = NOW(),
                completed_by = $2,
                updated_at = NOW()
             WHERE id = $1",
        )
        .bind(id)
        .bind(user_id)
        .execute(&self.pool)
        .await
        .map_err(CrmError::Database)?;

        self.get_activity(id).await
    }

    /// Get activity timeline for a record
    pub async fn get_timeline(
        &self,
        tenant_id: Uuid,
        account_id: Option<Uuid>,
        contact_id: Option<Uuid>,
        limit: i32,
    ) -> Result<Vec<Activity>, CrmError> {
        let activities = if let Some(aid) = account_id {
            sqlx::query_as::<_, Activity>(
                "SELECT * FROM crm_activities
                 WHERE tenant_id = $1
                   AND (account_id = $2 OR contact_id IN (
                       SELECT id FROM crm_contacts WHERE account_id = $2
                   ))
                 ORDER BY created_at DESC LIMIT $3",
            )
            .bind(tenant_id)
            .bind(aid)
            .bind(limit)
            .fetch_all(&self.pool)
            .await
        } else if let Some(cid) = contact_id {
            sqlx::query_as::<_, Activity>(
                "SELECT * FROM crm_activities
                 WHERE tenant_id = $1 AND contact_id = $2
                 ORDER BY created_at DESC LIMIT $3",
            )
            .bind(tenant_id)
            .bind(cid)
            .bind(limit)
            .fetch_all(&self.pool)
            .await
        } else {
            return Err(CrmError::Validation("Account or contact ID required".into()));
        }
        .map_err(CrmError::Database)?;

        Ok(activities)
    }
}
