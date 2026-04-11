// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Scheduling & Automation Module
 *
 * Send later, recurring emails, auto-responders, and rules engine
 */

use sqlx::PgPool;
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration, Weekday, NaiveTime};
use serde_json::Value as JsonValue;

#[derive(Debug, thiserror::Error)]
pub enum SchedulingError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("Not found")]
    NotFound,
    #[error("Invalid schedule: {0}")]
    InvalidSchedule(String),
    #[error("Past date not allowed")]
    PastDate,
    #[error("Rule limit exceeded")]
    RuleLimitExceeded,
    #[error("Invalid condition: {0}")]
    InvalidCondition(String),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ScheduledEmail {
    pub id: Uuid,
    pub user_id: Uuid,
    pub draft_id: Uuid,
    pub scheduled_for: DateTime<Utc>,
    pub status: ScheduleStatus,
    pub retry_count: i32,
    pub last_error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub sent_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, sqlx::Type, PartialEq)]
#[sqlx(type_name = "schedule_status", rename_all = "snake_case")]
pub enum ScheduleStatus { Pending, Sending, Sent, Failed, Cancelled }

#[derive(Debug, Serialize, Deserialize)]
pub struct RecurringEmail {
    pub id: Uuid,
    pub user_id: Uuid,
    pub name: String,
    pub to_addresses: Vec<String>,
    pub subject: String,
    pub body_html: String,
    pub schedule: RecurrenceSchedule,
    pub next_send_at: Option<DateTime<Utc>>,
    pub is_active: bool,
    pub send_count: i32,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RecurrenceSchedule {
    pub frequency: RecurrenceFrequency,
    pub interval: i32,
    pub days_of_week: Option<Vec<Weekday>>,
    pub time_of_day: NaiveTime,
    pub timezone: String,
    pub start_date: DateTime<Utc>,
    pub end_date: Option<DateTime<Utc>>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum RecurrenceFrequency { Daily, Weekly, BiWeekly, Monthly, Yearly }

#[derive(Debug, Serialize, Deserialize)]
pub struct AutoResponder {
    pub id: Uuid,
    pub user_id: Uuid,
    pub name: String,
    pub responder_type: ResponderType,
    pub subject: String,
    pub body_html: String,
    pub start_date: Option<DateTime<Utc>>,
    pub end_date: Option<DateTime<Utc>>,
    pub is_active: bool,
    pub exclude_contacts: bool,
    pub exclude_domains: Vec<String>,
    pub response_count: i32,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, sqlx::Type)]
#[sqlx(type_name = "responder_type", rename_all = "snake_case")]
pub enum ResponderType { Vacation, OutOfOffice, Acknowledgment, Custom }

#[derive(Debug, Serialize, Deserialize)]
pub struct EmailRule {
    pub id: Uuid,
    pub user_id: Uuid,
    pub name: String,
    pub priority: i32,
    pub conditions: Vec<RuleCondition>,
    pub condition_logic: ConditionLogic,
    pub actions: Vec<RuleAction>,
    pub is_active: bool,
    pub match_count: i32,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum ConditionLogic { And, Or }

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RuleCondition {
    pub field: ConditionField,
    pub operator: ConditionOperator,
    pub value: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum ConditionField { From, To, Subject, Body, HasAttachment, IsRead, Folder }

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum ConditionOperator { Equals, NotEquals, Contains, NotContains, StartsWith, EndsWith, Matches }

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RuleAction {
    pub action_type: ActionType,
    pub params: JsonValue,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum ActionType { MoveTo, AddLabel, MarkAsRead, Star, Archive, Delete, Forward, Notify, Skip }

#[derive(Debug, Serialize, Deserialize)]
pub struct FollowUpReminder {
    pub id: Uuid,
    pub user_id: Uuid,
    pub email_id: Uuid,
    pub remind_at: DateTime<Utc>,
    pub note: Option<String>,
    pub is_completed: bool,
    pub created_at: DateTime<Utc>,
}

pub struct SchedulingService {
    pool: PgPool,
}

impl SchedulingService {
    pub fn new(pool: PgPool) -> Self { Self { pool } }

    pub async fn schedule_email(&self, user_id: Uuid, draft_id: Uuid, scheduled_for: DateTime<Utc>) -> Result<ScheduledEmail, SchedulingError> {
        if scheduled_for <= Utc::now() { return Err(SchedulingError::PastDate); }

        let draft_exists: bool = sqlx::query_scalar!("SELECT EXISTS(SELECT 1 FROM drafts WHERE id = $1 AND user_id = $2)", draft_id, user_id)
            .fetch_one(&self.pool).await?.unwrap_or(false);
        if !draft_exists { return Err(SchedulingError::NotFound); }

        let id = Uuid::new_v4();
        sqlx::query!("INSERT INTO scheduled_emails (id, user_id, draft_id, scheduled_for, status, retry_count, created_at) VALUES ($1, $2, $3, $4, 'pending', 0, NOW())",
            id, user_id, draft_id, scheduled_for
        ).execute(&self.pool).await?;

        self.get_scheduled_email(user_id, id).await
    }

    pub async fn get_scheduled_email(&self, user_id: Uuid, id: Uuid) -> Result<ScheduledEmail, SchedulingError> {
        let row = sqlx::query!("SELECT id, user_id, draft_id, scheduled_for, status as \"status: ScheduleStatus\", retry_count, last_error, created_at, sent_at FROM scheduled_emails WHERE id = $1 AND user_id = $2", id, user_id)
            .fetch_optional(&self.pool).await?.ok_or(SchedulingError::NotFound)?;
        Ok(ScheduledEmail { id: row.id, user_id: row.user_id, draft_id: row.draft_id, scheduled_for: row.scheduled_for, status: row.status, retry_count: row.retry_count, last_error: row.last_error, created_at: row.created_at, sent_at: row.sent_at })
    }

    pub async fn list_scheduled(&self, user_id: Uuid) -> Result<Vec<ScheduledEmail>, SchedulingError> {
        let rows = sqlx::query!("SELECT id, user_id, draft_id, scheduled_for, status as \"status: ScheduleStatus\", retry_count, last_error, created_at, sent_at FROM scheduled_emails WHERE user_id = $1 AND status = 'pending' ORDER BY scheduled_for", user_id)
            .fetch_all(&self.pool).await?;
        Ok(rows.into_iter().map(|r| ScheduledEmail { id: r.id, user_id: r.user_id, draft_id: r.draft_id, scheduled_for: r.scheduled_for, status: r.status, retry_count: r.retry_count, last_error: r.last_error, created_at: r.created_at, sent_at: r.sent_at }).collect())
    }

    pub async fn cancel_scheduled(&self, user_id: Uuid, id: Uuid) -> Result<(), SchedulingError> {
        let result = sqlx::query!("UPDATE scheduled_emails SET status = 'cancelled' WHERE id = $1 AND user_id = $2 AND status = 'pending'", id, user_id)
            .execute(&self.pool).await?;
        if result.rows_affected() == 0 { return Err(SchedulingError::NotFound); }
        Ok(())
    }

    pub async fn get_due_emails(&self) -> Result<Vec<ScheduledEmail>, SchedulingError> {
        let rows = sqlx::query!("SELECT id, user_id, draft_id, scheduled_for, status as \"status: ScheduleStatus\", retry_count, last_error, created_at, sent_at FROM scheduled_emails WHERE status = 'pending' AND scheduled_for <= NOW() LIMIT 100")
            .fetch_all(&self.pool).await?;
        Ok(rows.into_iter().map(|r| ScheduledEmail { id: r.id, user_id: r.user_id, draft_id: r.draft_id, scheduled_for: r.scheduled_for, status: r.status, retry_count: r.retry_count, last_error: r.last_error, created_at: r.created_at, sent_at: r.sent_at }).collect())
    }

    pub async fn mark_sent(&self, id: Uuid) -> Result<(), SchedulingError> {
        sqlx::query!("UPDATE scheduled_emails SET status = 'sent', sent_at = NOW() WHERE id = $1", id).execute(&self.pool).await?;
        Ok(())
    }

    pub async fn mark_failed(&self, id: Uuid, error: &str) -> Result<(), SchedulingError> {
        sqlx::query!("UPDATE scheduled_emails SET status = CASE WHEN retry_count >= 3 THEN 'failed' ELSE 'pending' END, retry_count = retry_count + 1, last_error = $2 WHERE id = $1", id, error)
            .execute(&self.pool).await?;
        Ok(())
    }
}

pub struct AutoResponderService {
    pool: PgPool,
}

impl AutoResponderService {
    pub fn new(pool: PgPool) -> Self { Self { pool } }

    pub async fn create(&self, user_id: Uuid, name: String, responder_type: ResponderType, subject: String, body_html: String, start_date: Option<DateTime<Utc>>, end_date: Option<DateTime<Utc>>) -> Result<AutoResponder, SchedulingError> {
        let id = Uuid::new_v4();
        sqlx::query!("INSERT INTO auto_responders (id, user_id, name, responder_type, subject, body_html, start_date, end_date, is_active, exclude_contacts, exclude_domains, response_count, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, true, true, '{}', 0, NOW())",
            id, user_id, name, responder_type as ResponderType, subject, body_html, start_date, end_date
        ).execute(&self.pool).await?;

        self.get(user_id, id).await
    }

    pub async fn get(&self, user_id: Uuid, id: Uuid) -> Result<AutoResponder, SchedulingError> {
        let row = sqlx::query!("SELECT id, user_id, name, responder_type as \"responder_type: ResponderType\", subject, body_html, start_date, end_date, is_active, exclude_contacts, exclude_domains, response_count, created_at FROM auto_responders WHERE id = $1 AND user_id = $2", id, user_id)
            .fetch_optional(&self.pool).await?.ok_or(SchedulingError::NotFound)?;
        Ok(AutoResponder { id: row.id, user_id: row.user_id, name: row.name, responder_type: row.responder_type, subject: row.subject, body_html: row.body_html, start_date: row.start_date, end_date: row.end_date, is_active: row.is_active, exclude_contacts: row.exclude_contacts, exclude_domains: row.exclude_domains, response_count: row.response_count, created_at: row.created_at })
    }

    pub async fn list(&self, user_id: Uuid) -> Result<Vec<AutoResponder>, SchedulingError> {
        let rows = sqlx::query!("SELECT id, user_id, name, responder_type as \"responder_type: ResponderType\", subject, body_html, start_date, end_date, is_active, exclude_contacts, exclude_domains, response_count, created_at FROM auto_responders WHERE user_id = $1", user_id)
            .fetch_all(&self.pool).await?;
        Ok(rows.into_iter().map(|r| AutoResponder { id: r.id, user_id: r.user_id, name: r.name, responder_type: r.responder_type, subject: r.subject, body_html: r.body_html, start_date: r.start_date, end_date: r.end_date, is_active: r.is_active, exclude_contacts: r.exclude_contacts, exclude_domains: r.exclude_domains, response_count: r.response_count, created_at: r.created_at }).collect())
    }

    pub async fn toggle(&self, user_id: Uuid, id: Uuid, is_active: bool) -> Result<(), SchedulingError> {
        let result = sqlx::query!("UPDATE auto_responders SET is_active = $3 WHERE id = $1 AND user_id = $2", id, user_id, is_active).execute(&self.pool).await?;
        if result.rows_affected() == 0 { return Err(SchedulingError::NotFound); }
        Ok(())
    }

    pub async fn should_respond(&self, user_id: Uuid, sender_email: &str) -> Result<Option<AutoResponder>, SchedulingError> {
        let responders = sqlx::query!("SELECT id, user_id, name, responder_type as \"responder_type: ResponderType\", subject, body_html, start_date, end_date, is_active, exclude_contacts, exclude_domains, response_count, created_at FROM auto_responders WHERE user_id = $1 AND is_active = true AND (start_date IS NULL OR start_date <= NOW()) AND (end_date IS NULL OR end_date >= NOW())", user_id)
            .fetch_all(&self.pool).await?;

        if responders.is_empty() { return Ok(None); }

        let recently_responded: bool = sqlx::query_scalar!("SELECT EXISTS(SELECT 1 FROM auto_response_log WHERE user_id = $1 AND sender_email = $2 AND responded_at > NOW() - INTERVAL '24 hours')", user_id, sender_email)
            .fetch_one(&self.pool).await?.unwrap_or(false);
        if recently_responded { return Ok(None); }

        for r in responders {
            let domain = sender_email.split('@').last().unwrap_or("");
            if r.exclude_domains.iter().any(|d| d == domain) { continue; }

            if r.exclude_contacts {
                let is_contact: bool = sqlx::query_scalar!("SELECT EXISTS(SELECT 1 FROM contacts WHERE user_id = $1 AND email = $2)", user_id, sender_email)
                    .fetch_one(&self.pool).await?.unwrap_or(false);
                if is_contact { continue; }
            }

            return Ok(Some(AutoResponder { id: r.id, user_id: r.user_id, name: r.name, responder_type: r.responder_type, subject: r.subject, body_html: r.body_html, start_date: r.start_date, end_date: r.end_date, is_active: r.is_active, exclude_contacts: r.exclude_contacts, exclude_domains: r.exclude_domains, response_count: r.response_count, created_at: r.created_at }));
        }
        Ok(None)
    }

    pub async fn log_response(&self, user_id: Uuid, responder_id: Uuid, sender_email: &str) -> Result<(), SchedulingError> {
        sqlx::query!("INSERT INTO auto_response_log (user_id, responder_id, sender_email, responded_at) VALUES ($1, $2, $3, NOW())", user_id, responder_id, sender_email)
            .execute(&self.pool).await?;
        sqlx::query!("UPDATE auto_responders SET response_count = response_count + 1 WHERE id = $1", responder_id)
            .execute(&self.pool).await?;
        Ok(())
    }

    pub async fn delete(&self, user_id: Uuid, id: Uuid) -> Result<(), SchedulingError> {
        let result = sqlx::query!("DELETE FROM auto_responders WHERE id = $1 AND user_id = $2", id, user_id).execute(&self.pool).await?;
        if result.rows_affected() == 0 { return Err(SchedulingError::NotFound); }
        Ok(())
    }
}

pub struct RulesEngine {
    pool: PgPool,
}

impl RulesEngine {
    pub fn new(pool: PgPool) -> Self { Self { pool } }

    pub async fn create_rule(&self, user_id: Uuid, name: String, conditions: Vec<RuleCondition>, condition_logic: ConditionLogic, actions: Vec<RuleAction>, priority: i32) -> Result<EmailRule, SchedulingError> {
        let count: i64 = sqlx::query_scalar!("SELECT COUNT(*) FROM email_rules WHERE user_id = $1", user_id)
            .fetch_one(&self.pool).await?.unwrap_or(0);
        if count >= 100 { return Err(SchedulingError::RuleLimitExceeded); }

        let id = Uuid::new_v4();
        let conditions_json = serde_json::to_value(&conditions).map_err(|e| SchedulingError::InvalidCondition(e.to_string()))?;
        let actions_json = serde_json::to_value(&actions).map_err(|e| SchedulingError::InvalidCondition(e.to_string()))?;
        let logic_str = format!("{:?}", condition_logic);

        sqlx::query!("INSERT INTO email_rules (id, user_id, name, priority, conditions, condition_logic, actions, is_active, match_count, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7, true, 0, NOW())",
            id, user_id, name, priority, conditions_json, logic_str, actions_json
        ).execute(&self.pool).await?;

        self.get_rule(user_id, id).await
    }

    pub async fn get_rule(&self, user_id: Uuid, id: Uuid) -> Result<EmailRule, SchedulingError> {
        let row = sqlx::query!("SELECT id, user_id, name, priority, conditions, condition_logic, actions, is_active, match_count, created_at FROM email_rules WHERE id = $1 AND user_id = $2", id, user_id)
            .fetch_optional(&self.pool).await?.ok_or(SchedulingError::NotFound)?;

        let conditions: Vec<RuleCondition> = serde_json::from_value(row.conditions).unwrap_or_default();
        let actions: Vec<RuleAction> = serde_json::from_value(row.actions).unwrap_or_default();
        let condition_logic = if row.condition_logic == "Or" { ConditionLogic::Or } else { ConditionLogic::And };

        Ok(EmailRule { id: row.id, user_id: row.user_id, name: row.name, priority: row.priority, conditions, condition_logic, actions, is_active: row.is_active, match_count: row.match_count, created_at: row.created_at })
    }

    pub async fn list_rules(&self, user_id: Uuid) -> Result<Vec<EmailRule>, SchedulingError> {
        let rows = sqlx::query!("SELECT id, user_id, name, priority, conditions, condition_logic, actions, is_active, match_count, created_at FROM email_rules WHERE user_id = $1 ORDER BY priority DESC", user_id)
            .fetch_all(&self.pool).await?;

        Ok(rows.into_iter().map(|r| {
            let conditions: Vec<RuleCondition> = serde_json::from_value(r.conditions).unwrap_or_default();
            let actions: Vec<RuleAction> = serde_json::from_value(r.actions).unwrap_or_default();
            let condition_logic = if r.condition_logic == "Or" { ConditionLogic::Or } else { ConditionLogic::And };
            EmailRule { id: r.id, user_id: r.user_id, name: r.name, priority: r.priority, conditions, condition_logic, actions, is_active: r.is_active, match_count: r.match_count, created_at: r.created_at }
        }).collect())
    }

    pub async fn apply_rules(&self, user_id: Uuid, email: &EmailForRules) -> Result<Vec<RuleAction>, SchedulingError> {
        let rules = self.list_rules(user_id).await?;
        let mut result = Vec::new();

        for rule in rules {
            if !rule.is_active { continue; }
            if self.matches_rule(&rule, email) {
                sqlx::query!("UPDATE email_rules SET match_count = match_count + 1 WHERE id = $1", rule.id).execute(&self.pool).await?;
                if rule.actions.iter().any(|a| matches!(a.action_type, ActionType::Skip)) { break; }
                result.extend(rule.actions);
            }
        }
        Ok(result)
    }

    fn matches_rule(&self, rule: &EmailRule, email: &EmailForRules) -> bool {
        let results: Vec<bool> = rule.conditions.iter().map(|c| self.evaluate_condition(c, email)).collect();
        match rule.condition_logic {
            ConditionLogic::And => results.iter().all(|&r| r),
            ConditionLogic::Or => results.iter().any(|&r| r),
        }
    }

    fn evaluate_condition(&self, condition: &RuleCondition, email: &EmailForRules) -> bool {
        let value = match &condition.field {
            ConditionField::From => &email.from_address,
            ConditionField::To => &email.to_addresses.join(", "),
            ConditionField::Subject => &email.subject,
            ConditionField::Body => &email.body_text,
            ConditionField::HasAttachment => return email.has_attachments == (condition.value == "true"),
            ConditionField::IsRead => return email.is_read == (condition.value == "true"),
            ConditionField::Folder => &email.folder,
        };

        match condition.operator {
            ConditionOperator::Equals => value.to_lowercase() == condition.value.to_lowercase(),
            ConditionOperator::NotEquals => value.to_lowercase() != condition.value.to_lowercase(),
            ConditionOperator::Contains => value.to_lowercase().contains(&condition.value.to_lowercase()),
            ConditionOperator::NotContains => !value.to_lowercase().contains(&condition.value.to_lowercase()),
            ConditionOperator::StartsWith => value.to_lowercase().starts_with(&condition.value.to_lowercase()),
            ConditionOperator::EndsWith => value.to_lowercase().ends_with(&condition.value.to_lowercase()),
            ConditionOperator::Matches => regex::Regex::new(&condition.value).map(|r| r.is_match(value)).unwrap_or(false),
        }
    }

    pub async fn toggle_rule(&self, user_id: Uuid, id: Uuid, is_active: bool) -> Result<(), SchedulingError> {
        let result = sqlx::query!("UPDATE email_rules SET is_active = $3 WHERE id = $1 AND user_id = $2", id, user_id, is_active).execute(&self.pool).await?;
        if result.rows_affected() == 0 { return Err(SchedulingError::NotFound); }
        Ok(())
    }

    pub async fn delete_rule(&self, user_id: Uuid, id: Uuid) -> Result<(), SchedulingError> {
        let result = sqlx::query!("DELETE FROM email_rules WHERE id = $1 AND user_id = $2", id, user_id).execute(&self.pool).await?;
        if result.rows_affected() == 0 { return Err(SchedulingError::NotFound); }
        Ok(())
    }
}

#[derive(Debug)]
pub struct EmailForRules {
    pub from_address: String,
    pub to_addresses: Vec<String>,
    pub subject: String,
    pub body_text: String,
    pub has_attachments: bool,
    pub is_read: bool,
    pub folder: String,
}

pub struct ReminderService {
    pool: PgPool,
}

impl ReminderService {
    pub fn new(pool: PgPool) -> Self { Self { pool } }

    pub async fn create(&self, user_id: Uuid, email_id: Uuid, remind_at: DateTime<Utc>, note: Option<String>) -> Result<FollowUpReminder, SchedulingError> {
        let id = Uuid::new_v4();
        sqlx::query!("INSERT INTO follow_up_reminders (id, user_id, email_id, remind_at, note, is_completed, created_at) VALUES ($1, $2, $3, $4, $5, false, NOW())",
            id, user_id, email_id, remind_at, note
        ).execute(&self.pool).await?;

        Ok(FollowUpReminder { id, user_id, email_id, remind_at, note, is_completed: false, created_at: Utc::now() })
    }

    pub async fn get_due(&self, user_id: Uuid) -> Result<Vec<FollowUpReminder>, SchedulingError> {
        let rows = sqlx::query!("SELECT id, user_id, email_id, remind_at, note, is_completed, created_at FROM follow_up_reminders WHERE user_id = $1 AND remind_at <= NOW() AND is_completed = false", user_id)
            .fetch_all(&self.pool).await?;
        Ok(rows.into_iter().map(|r| FollowUpReminder { id: r.id, user_id: r.user_id, email_id: r.email_id, remind_at: r.remind_at, note: r.note, is_completed: r.is_completed, created_at: r.created_at }).collect())
    }

    pub async fn complete(&self, user_id: Uuid, id: Uuid) -> Result<(), SchedulingError> {
        let result = sqlx::query!("UPDATE follow_up_reminders SET is_completed = true WHERE id = $1 AND user_id = $2", id, user_id).execute(&self.pool).await?;
        if result.rows_affected() == 0 { return Err(SchedulingError::NotFound); }
        Ok(())
    }

    pub async fn snooze(&self, user_id: Uuid, id: Uuid, new_time: DateTime<Utc>) -> Result<(), SchedulingError> {
        let result = sqlx::query!("UPDATE follow_up_reminders SET remind_at = $3 WHERE id = $1 AND user_id = $2", id, user_id, new_time).execute(&self.pool).await?;
        if result.rows_affected() == 0 { return Err(SchedulingError::NotFound); }
        Ok(())
    }
}
