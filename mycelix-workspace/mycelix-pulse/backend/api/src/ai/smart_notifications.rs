// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Smart Notifications
//!
//! Intelligent notification filtering - only alert for important emails

use chrono::{DateTime, Utc, Timelike};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;
use std::collections::HashMap;

// ============================================================================
// Smart Notification Service
// ============================================================================

pub struct SmartNotificationService {
    pool: PgPool,
    importance_scorer: ImportanceScorer,
}

impl SmartNotificationService {
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool,
            importance_scorer: ImportanceScorer::new(),
        }
    }

    /// Determine if an email should trigger a notification
    pub async fn should_notify(
        &self,
        user_id: Uuid,
        email: &IncomingEmail,
    ) -> Result<NotificationDecision, NotificationError> {
        // Get user's notification preferences
        let prefs = self.get_user_preferences(user_id).await?;

        // Check if in focus mode
        if prefs.focus_mode_enabled {
            if let Some(until) = prefs.focus_mode_until {
                if Utc::now() < until {
                    return Ok(NotificationDecision {
                        should_notify: false,
                        reason: NotificationReason::FocusMode,
                        importance_score: 0.0,
                        suggested_action: Some(SuggestedAction::QueueForLater),
                    });
                }
            }
        }

        // Check quiet hours
        if self.is_quiet_hours(&prefs) {
            return Ok(NotificationDecision {
                should_notify: false,
                reason: NotificationReason::QuietHours,
                importance_score: 0.0,
                suggested_action: Some(SuggestedAction::QueueForLater),
            });
        }

        // Calculate importance score
        let importance = self.importance_scorer.score(user_id, email, &self.pool).await?;

        // Check against threshold
        let threshold = prefs.importance_threshold.unwrap_or(0.5);

        if importance.score >= threshold {
            // Check for batching
            if prefs.batch_notifications && importance.score < 0.8 {
                return Ok(NotificationDecision {
                    should_notify: false,
                    reason: NotificationReason::Batched,
                    importance_score: importance.score,
                    suggested_action: Some(SuggestedAction::BatchWithOthers),
                });
            }

            Ok(NotificationDecision {
                should_notify: true,
                reason: NotificationReason::Important(importance.factors.clone()),
                importance_score: importance.score,
                suggested_action: None,
            })
        } else {
            Ok(NotificationDecision {
                should_notify: false,
                reason: NotificationReason::BelowThreshold,
                importance_score: importance.score,
                suggested_action: Some(SuggestedAction::SilentDelivery),
            })
        }
    }

    /// Get batched notifications for user
    pub async fn get_batch_summary(
        &self,
        user_id: Uuid,
    ) -> Result<BatchSummary, NotificationError> {
        let pending: Vec<PendingNotificationRow> = sqlx::query_as(
            r#"
            SELECT e.id, e.subject, e.from_address, e.received_at, n.importance_score
            FROM pending_notifications n
            JOIN emails e ON e.id = n.email_id
            WHERE n.user_id = $1 AND n.sent = false
            ORDER BY n.importance_score DESC
            LIMIT 20
            "#,
        )
        .bind(user_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| NotificationError::Database(e.to_string()))?;

        if pending.is_empty() {
            return Ok(BatchSummary {
                count: 0,
                top_emails: vec![],
                summary: "No new emails".to_string(),
            });
        }

        let summary = if pending.len() == 1 {
            format!("1 new email from {}", pending[0].from_address)
        } else {
            let top_senders: Vec<_> = pending.iter()
                .take(3)
                .map(|e| e.from_address.clone())
                .collect();
            format!("{} new emails including from {}", pending.len(), top_senders.join(", "))
        };

        Ok(BatchSummary {
            count: pending.len(),
            top_emails: pending.into_iter().map(|e| BatchedEmail {
                id: e.id,
                subject: e.subject,
                from: e.from_address,
                importance: e.importance_score,
            }).collect(),
            summary,
        })
    }

    /// Update user notification preferences
    pub async fn update_preferences(
        &self,
        user_id: Uuid,
        prefs: NotificationPreferences,
    ) -> Result<(), NotificationError> {
        sqlx::query(
            r#"
            INSERT INTO notification_preferences (
                user_id, importance_threshold, batch_notifications,
                quiet_hours_start, quiet_hours_end, focus_mode_enabled,
                focus_mode_until, vip_senders, muted_senders, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
            ON CONFLICT (user_id) DO UPDATE SET
                importance_threshold = EXCLUDED.importance_threshold,
                batch_notifications = EXCLUDED.batch_notifications,
                quiet_hours_start = EXCLUDED.quiet_hours_start,
                quiet_hours_end = EXCLUDED.quiet_hours_end,
                focus_mode_enabled = EXCLUDED.focus_mode_enabled,
                focus_mode_until = EXCLUDED.focus_mode_until,
                vip_senders = EXCLUDED.vip_senders,
                muted_senders = EXCLUDED.muted_senders,
                updated_at = NOW()
            "#,
        )
        .bind(user_id)
        .bind(prefs.importance_threshold)
        .bind(prefs.batch_notifications)
        .bind(prefs.quiet_hours_start)
        .bind(prefs.quiet_hours_end)
        .bind(prefs.focus_mode_enabled)
        .bind(prefs.focus_mode_until)
        .bind(&prefs.vip_senders)
        .bind(&prefs.muted_senders)
        .execute(&self.pool)
        .await
        .map_err(|e| NotificationError::Database(e.to_string()))?;

        Ok(())
    }

    /// Enable focus mode
    pub async fn enable_focus_mode(
        &self,
        user_id: Uuid,
        duration_minutes: i64,
    ) -> Result<DateTime<Utc>, NotificationError> {
        let until = Utc::now() + chrono::Duration::minutes(duration_minutes);

        sqlx::query(
            "UPDATE notification_preferences SET focus_mode_enabled = true, focus_mode_until = $2 WHERE user_id = $1",
        )
        .bind(user_id)
        .bind(until)
        .execute(&self.pool)
        .await
        .map_err(|e| NotificationError::Database(e.to_string()))?;

        Ok(until)
    }

    /// Disable focus mode
    pub async fn disable_focus_mode(&self, user_id: Uuid) -> Result<(), NotificationError> {
        sqlx::query(
            "UPDATE notification_preferences SET focus_mode_enabled = false, focus_mode_until = NULL WHERE user_id = $1",
        )
        .bind(user_id)
        .execute(&self.pool)
        .await
        .map_err(|e| NotificationError::Database(e.to_string()))?;

        Ok(())
    }

    /// Learn from user feedback
    pub async fn record_feedback(
        &self,
        user_id: Uuid,
        email_id: Uuid,
        feedback: NotificationFeedback,
    ) -> Result<(), NotificationError> {
        sqlx::query(
            r#"
            INSERT INTO notification_feedback (id, user_id, email_id, feedback_type, created_at)
            VALUES ($1, $2, $3, $4, NOW())
            "#,
        )
        .bind(Uuid::new_v4())
        .bind(user_id)
        .bind(email_id)
        .bind(format!("{:?}", feedback))
        .execute(&self.pool)
        .await
        .map_err(|e| NotificationError::Database(e.to_string()))?;

        // Update importance model based on feedback
        self.importance_scorer.learn_from_feedback(user_id, email_id, feedback, &self.pool).await?;

        Ok(())
    }

    // Private helpers

    async fn get_user_preferences(&self, user_id: Uuid) -> Result<NotificationPreferences, NotificationError> {
        let prefs: Option<PreferencesRow> = sqlx::query_as(
            r#"
            SELECT importance_threshold, batch_notifications, quiet_hours_start,
                   quiet_hours_end, focus_mode_enabled, focus_mode_until,
                   vip_senders, muted_senders
            FROM notification_preferences
            WHERE user_id = $1
            "#,
        )
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| NotificationError::Database(e.to_string()))?;

        Ok(prefs.map(|p| NotificationPreferences {
            importance_threshold: p.importance_threshold,
            batch_notifications: p.batch_notifications,
            quiet_hours_start: p.quiet_hours_start,
            quiet_hours_end: p.quiet_hours_end,
            focus_mode_enabled: p.focus_mode_enabled,
            focus_mode_until: p.focus_mode_until,
            vip_senders: p.vip_senders,
            muted_senders: p.muted_senders,
        }).unwrap_or_default())
    }

    fn is_quiet_hours(&self, prefs: &NotificationPreferences) -> bool {
        if let (Some(start), Some(end)) = (prefs.quiet_hours_start, prefs.quiet_hours_end) {
            let now = Utc::now();
            let current_hour = now.hour() as i32;

            if start <= end {
                // Same day range (e.g., 9-17)
                current_hour >= start && current_hour < end
            } else {
                // Overnight range (e.g., 22-7)
                current_hour >= start || current_hour < end
            }
        } else {
            false
        }
    }
}

// ============================================================================
// Importance Scorer
// ============================================================================

struct ImportanceScorer;

impl ImportanceScorer {
    fn new() -> Self {
        Self
    }

    async fn score(
        &self,
        user_id: Uuid,
        email: &IncomingEmail,
        pool: &PgPool,
    ) -> Result<ImportanceScore, NotificationError> {
        let mut score = 0.0;
        let mut factors = Vec::new();

        // Factor 1: Sender trust score
        let sender_trust = self.get_sender_trust(user_id, &email.from, pool).await?;
        if sender_trust > 0.8 {
            score += 0.3;
            factors.push(ImportanceFactor::HighTrustSender(sender_trust));
        } else if sender_trust > 0.5 {
            score += 0.1;
        }

        // Factor 2: VIP sender
        if self.is_vip_sender(user_id, &email.from, pool).await? {
            score += 0.4;
            factors.push(ImportanceFactor::VIPSender);
        }

        // Factor 3: Direct recipient (not CC)
        if email.to.iter().any(|t| t.contains(&email.from)) {
            score += 0.1;
            factors.push(ImportanceFactor::DirectRecipient);
        }

        // Factor 4: Reply to user's email
        if email.in_reply_to.is_some() {
            if self.is_reply_to_user(user_id, &email.in_reply_to, pool).await? {
                score += 0.3;
                factors.push(ImportanceFactor::ReplyToMe);
            }
        }

        // Factor 5: Urgency keywords
        let urgency = self.detect_urgency(&email.subject, &email.body_preview);
        if urgency > 0.7 {
            score += 0.2;
            factors.push(ImportanceFactor::UrgentKeywords);
        }

        // Factor 6: Mentions user's name
        if self.mentions_user(user_id, &email.body_preview, pool).await? {
            score += 0.15;
            factors.push(ImportanceFactor::MentionsMe);
        }

        // Factor 7: Calendar invite
        if email.has_calendar_invite {
            score += 0.2;
            factors.push(ImportanceFactor::CalendarInvite);
        }

        // Muted sender check (overrides everything)
        if self.is_muted_sender(user_id, &email.from, pool).await? {
            return Ok(ImportanceScore {
                score: 0.0,
                factors: vec![ImportanceFactor::MutedSender],
            });
        }

        Ok(ImportanceScore {
            score: score.min(1.0),
            factors,
        })
    }

    async fn learn_from_feedback(
        &self,
        user_id: Uuid,
        email_id: Uuid,
        feedback: NotificationFeedback,
        pool: &PgPool,
    ) -> Result<(), NotificationError> {
        // Would update ML model or rule weights based on feedback
        // For now, just adjust VIP/muted lists
        match feedback {
            NotificationFeedback::AlwaysNotify => {
                // Add sender to VIP list
            }
            NotificationFeedback::NeverNotify => {
                // Add sender to muted list
            }
            _ => {}
        }
        Ok(())
    }

    async fn get_sender_trust(&self, user_id: Uuid, sender: &str, pool: &PgPool) -> Result<f64, NotificationError> {
        let result: Option<(f64,)> = sqlx::query_as(
            "SELECT trust_score FROM contacts WHERE user_id = $1 AND email = $2",
        )
        .bind(user_id)
        .bind(sender)
        .fetch_optional(pool)
        .await
        .map_err(|e| NotificationError::Database(e.to_string()))?;

        Ok(result.map(|r| r.0).unwrap_or(0.5))
    }

    async fn is_vip_sender(&self, user_id: Uuid, sender: &str, pool: &PgPool) -> Result<bool, NotificationError> {
        let result: Option<(Vec<String>,)> = sqlx::query_as(
            "SELECT vip_senders FROM notification_preferences WHERE user_id = $1",
        )
        .bind(user_id)
        .fetch_optional(pool)
        .await
        .map_err(|e| NotificationError::Database(e.to_string()))?;

        Ok(result.map(|r| r.0.contains(&sender.to_string())).unwrap_or(false))
    }

    async fn is_muted_sender(&self, user_id: Uuid, sender: &str, pool: &PgPool) -> Result<bool, NotificationError> {
        let result: Option<(Vec<String>,)> = sqlx::query_as(
            "SELECT muted_senders FROM notification_preferences WHERE user_id = $1",
        )
        .bind(user_id)
        .fetch_optional(pool)
        .await
        .map_err(|e| NotificationError::Database(e.to_string()))?;

        Ok(result.map(|r| r.0.contains(&sender.to_string())).unwrap_or(false))
    }

    async fn is_reply_to_user(&self, user_id: Uuid, in_reply_to: &Option<String>, pool: &PgPool) -> Result<bool, NotificationError> {
        if let Some(message_id) = in_reply_to {
            let exists: Option<(i64,)> = sqlx::query_as(
                "SELECT 1 FROM emails WHERE user_id = $1 AND message_id = $2 AND is_outgoing = true",
            )
            .bind(user_id)
            .bind(message_id)
            .fetch_optional(pool)
            .await
            .map_err(|e| NotificationError::Database(e.to_string()))?;

            Ok(exists.is_some())
        } else {
            Ok(false)
        }
    }

    async fn mentions_user(&self, user_id: Uuid, text: &str, pool: &PgPool) -> Result<bool, NotificationError> {
        let user: Option<(Option<String>,)> = sqlx::query_as(
            "SELECT name FROM users WHERE id = $1",
        )
        .bind(user_id)
        .fetch_optional(pool)
        .await
        .map_err(|e| NotificationError::Database(e.to_string()))?;

        if let Some((Some(name),)) = user {
            Ok(text.to_lowercase().contains(&name.to_lowercase()))
        } else {
            Ok(false)
        }
    }

    fn detect_urgency(&self, subject: &str, body: &str) -> f64 {
        let text = format!("{} {}", subject, body).to_lowercase();
        let urgent_keywords = [
            "urgent", "asap", "immediately", "critical", "emergency",
            "deadline", "today", "right away", "time sensitive"
        ];

        let matches = urgent_keywords.iter()
            .filter(|k| text.contains(*k))
            .count();

        (matches as f64 / 3.0).min(1.0)
    }
}

// ============================================================================
// Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncomingEmail {
    pub id: Uuid,
    pub from: String,
    pub to: Vec<String>,
    pub subject: String,
    pub body_preview: String,
    pub in_reply_to: Option<String>,
    pub has_attachment: bool,
    pub has_calendar_invite: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationDecision {
    pub should_notify: bool,
    pub reason: NotificationReason,
    pub importance_score: f64,
    pub suggested_action: Option<SuggestedAction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationReason {
    Important(Vec<ImportanceFactor>),
    BelowThreshold,
    FocusMode,
    QuietHours,
    Batched,
    MutedSender,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImportanceFactor {
    HighTrustSender(f64),
    VIPSender,
    DirectRecipient,
    ReplyToMe,
    UrgentKeywords,
    MentionsMe,
    CalendarInvite,
    MutedSender,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuggestedAction {
    SilentDelivery,
    BatchWithOthers,
    QueueForLater,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportanceScore {
    pub score: f64,
    pub factors: Vec<ImportanceFactor>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NotificationPreferences {
    pub importance_threshold: Option<f64>,
    pub batch_notifications: bool,
    pub quiet_hours_start: Option<i32>,
    pub quiet_hours_end: Option<i32>,
    pub focus_mode_enabled: bool,
    pub focus_mode_until: Option<DateTime<Utc>>,
    pub vip_senders: Vec<String>,
    pub muted_senders: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NotificationFeedback {
    Helpful,
    NotHelpful,
    AlwaysNotify,
    NeverNotify,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSummary {
    pub count: usize,
    pub top_emails: Vec<BatchedEmail>,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchedEmail {
    pub id: Uuid,
    pub subject: String,
    pub from: String,
    pub importance: f64,
}

#[derive(Debug, sqlx::FromRow)]
struct PreferencesRow {
    importance_threshold: Option<f64>,
    batch_notifications: bool,
    quiet_hours_start: Option<i32>,
    quiet_hours_end: Option<i32>,
    focus_mode_enabled: bool,
    focus_mode_until: Option<DateTime<Utc>>,
    vip_senders: Vec<String>,
    muted_senders: Vec<String>,
}

#[derive(Debug, sqlx::FromRow)]
struct PendingNotificationRow {
    id: Uuid,
    subject: String,
    from_address: String,
    received_at: DateTime<Utc>,
    importance_score: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum NotificationError {
    #[error("Database error: {0}")]
    Database(String),
}
