// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Email Scheduling & Undo Send for Mycelix Mail
//!
//! Schedule emails for future delivery and provide an undo window
//! before emails are actually sent.

use std::collections::HashMap;
use std::sync::Arc;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Status of a scheduled email
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ScheduledEmailStatus {
    /// In the undo window, can still be cancelled
    PendingConfirmation,
    /// Confirmed, waiting for scheduled time
    Scheduled,
    /// Currently being sent
    Sending,
    /// Successfully sent
    Sent,
    /// Cancelled by user
    Cancelled,
    /// Failed to send
    Failed,
}

/// A scheduled email
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledEmail {
    pub id: Uuid,
    pub user_id: Uuid,
    pub status: ScheduledEmailStatus,

    // Email content
    pub to: Vec<String>,
    pub cc: Vec<String>,
    pub bcc: Vec<String>,
    pub subject: String,
    pub body_text: String,
    pub body_html: Option<String>,
    pub attachment_ids: Vec<Uuid>,
    pub reply_to_id: Option<Uuid>,

    // Scheduling
    pub created_at: DateTime<Utc>,
    pub undo_until: DateTime<Utc>,
    pub scheduled_for: DateTime<Utc>,
    pub sent_at: Option<DateTime<Utc>>,
    pub cancelled_at: Option<DateTime<Utc>>,

    // Retry info
    pub attempt_count: u32,
    pub max_attempts: u32,
    pub last_attempt_at: Option<DateTime<Utc>>,
    pub last_error: Option<String>,

    // Metadata
    pub timezone: String,
    pub send_later_reason: Option<String>,
}

/// Configuration for email scheduling
#[derive(Debug, Clone)]
pub struct SchedulingConfig {
    /// Default undo window in seconds (e.g., 10 seconds)
    pub default_undo_window_secs: u32,
    /// Maximum scheduled delay in days
    pub max_schedule_days: u32,
    /// Maximum retry attempts
    pub max_retry_attempts: u32,
    /// Retry delay base (exponential backoff)
    pub retry_delay_base_secs: u32,
}

impl Default for SchedulingConfig {
    fn default() -> Self {
        Self {
            default_undo_window_secs: 10,
            max_schedule_days: 365,
            max_retry_attempts: 3,
            retry_delay_base_secs: 60,
        }
    }
}

/// Email scheduling service
pub struct EmailScheduler {
    config: SchedulingConfig,
    scheduled_emails: Arc<RwLock<HashMap<Uuid, ScheduledEmail>>>,
    user_undo_settings: Arc<RwLock<HashMap<Uuid, u32>>>, // user_id -> undo window in seconds
}

impl EmailScheduler {
    pub fn new(config: SchedulingConfig) -> Self {
        Self {
            config,
            scheduled_emails: Arc::new(RwLock::new(HashMap::new())),
            user_undo_settings: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Schedule an email for immediate send (with undo window)
    pub async fn send_with_undo(
        &self,
        user_id: Uuid,
        email: EmailDraft,
    ) -> Result<ScheduledEmail, SchedulingError> {
        let undo_window = self.get_user_undo_window(user_id).await;
        let now = Utc::now();

        let scheduled = ScheduledEmail {
            id: Uuid::new_v4(),
            user_id,
            status: ScheduledEmailStatus::PendingConfirmation,
            to: email.to,
            cc: email.cc.unwrap_or_default(),
            bcc: email.bcc.unwrap_or_default(),
            subject: email.subject,
            body_text: email.body_text,
            body_html: email.body_html,
            attachment_ids: email.attachment_ids.unwrap_or_default(),
            reply_to_id: email.reply_to_id,
            created_at: now,
            undo_until: now + Duration::seconds(undo_window as i64),
            scheduled_for: now + Duration::seconds(undo_window as i64),
            sent_at: None,
            cancelled_at: None,
            attempt_count: 0,
            max_attempts: self.config.max_retry_attempts,
            last_attempt_at: None,
            last_error: None,
            timezone: email.timezone.unwrap_or_else(|| "UTC".to_string()),
            send_later_reason: None,
        };

        self.scheduled_emails.write().await.insert(scheduled.id, scheduled.clone());

        Ok(scheduled)
    }

    /// Schedule an email for future delivery
    pub async fn schedule_send(
        &self,
        user_id: Uuid,
        email: EmailDraft,
        send_at: DateTime<Utc>,
    ) -> Result<ScheduledEmail, SchedulingError> {
        let now = Utc::now();

        // Validate schedule time
        if send_at <= now {
            return Err(SchedulingError::InvalidScheduleTime(
                "Cannot schedule email in the past".to_string()
            ));
        }

        let max_future = now + Duration::days(self.config.max_schedule_days as i64);
        if send_at > max_future {
            return Err(SchedulingError::InvalidScheduleTime(
                format!("Cannot schedule more than {} days in advance", self.config.max_schedule_days)
            ));
        }

        let scheduled = ScheduledEmail {
            id: Uuid::new_v4(),
            user_id,
            status: ScheduledEmailStatus::Scheduled,
            to: email.to,
            cc: email.cc.unwrap_or_default(),
            bcc: email.bcc.unwrap_or_default(),
            subject: email.subject,
            body_text: email.body_text,
            body_html: email.body_html,
            attachment_ids: email.attachment_ids.unwrap_or_default(),
            reply_to_id: email.reply_to_id,
            created_at: now,
            undo_until: now, // No undo window for scheduled emails
            scheduled_for: send_at,
            sent_at: None,
            cancelled_at: None,
            attempt_count: 0,
            max_attempts: self.config.max_retry_attempts,
            last_attempt_at: None,
            last_error: None,
            timezone: email.timezone.unwrap_or_else(|| "UTC".to_string()),
            send_later_reason: email.send_later_reason,
        };

        self.scheduled_emails.write().await.insert(scheduled.id, scheduled.clone());

        Ok(scheduled)
    }

    /// Cancel/undo a scheduled email
    pub async fn cancel(&self, user_id: Uuid, email_id: Uuid) -> Result<(), SchedulingError> {
        let mut emails = self.scheduled_emails.write().await;

        let email = emails.get_mut(&email_id)
            .ok_or(SchedulingError::NotFound(email_id))?;

        if email.user_id != user_id {
            return Err(SchedulingError::Unauthorized);
        }

        match email.status {
            ScheduledEmailStatus::PendingConfirmation |
            ScheduledEmailStatus::Scheduled => {
                email.status = ScheduledEmailStatus::Cancelled;
                email.cancelled_at = Some(Utc::now());
                Ok(())
            }
            ScheduledEmailStatus::Sending => {
                Err(SchedulingError::AlreadySending)
            }
            ScheduledEmailStatus::Sent => {
                Err(SchedulingError::AlreadySent)
            }
            ScheduledEmailStatus::Cancelled => {
                Err(SchedulingError::AlreadyCancelled)
            }
            ScheduledEmailStatus::Failed => {
                // Allow cancelling failed emails
                email.status = ScheduledEmailStatus::Cancelled;
                email.cancelled_at = Some(Utc::now());
                Ok(())
            }
        }
    }

    /// Reschedule an email
    pub async fn reschedule(
        &self,
        user_id: Uuid,
        email_id: Uuid,
        new_time: DateTime<Utc>,
    ) -> Result<ScheduledEmail, SchedulingError> {
        let mut emails = self.scheduled_emails.write().await;

        let email = emails.get_mut(&email_id)
            .ok_or(SchedulingError::NotFound(email_id))?;

        if email.user_id != user_id {
            return Err(SchedulingError::Unauthorized);
        }

        if !matches!(email.status,
            ScheduledEmailStatus::PendingConfirmation |
            ScheduledEmailStatus::Scheduled
        ) {
            return Err(SchedulingError::CannotReschedule);
        }

        let now = Utc::now();
        if new_time <= now {
            return Err(SchedulingError::InvalidScheduleTime(
                "Cannot reschedule to past time".to_string()
            ));
        }

        email.scheduled_for = new_time;
        email.status = ScheduledEmailStatus::Scheduled;

        Ok(email.clone())
    }

    /// Get emails ready to send
    pub async fn get_ready_to_send(&self) -> Vec<ScheduledEmail> {
        let now = Utc::now();
        let emails = self.scheduled_emails.read().await;

        emails.values()
            .filter(|e| {
                (e.status == ScheduledEmailStatus::Scheduled ||
                 e.status == ScheduledEmailStatus::PendingConfirmation) &&
                e.scheduled_for <= now
            })
            .cloned()
            .collect()
    }

    /// Get emails in undo window for a user
    pub async fn get_pending_undo(&self, user_id: Uuid) -> Vec<ScheduledEmail> {
        let now = Utc::now();
        let emails = self.scheduled_emails.read().await;

        emails.values()
            .filter(|e| {
                e.user_id == user_id &&
                e.status == ScheduledEmailStatus::PendingConfirmation &&
                e.undo_until > now
            })
            .cloned()
            .collect()
    }

    /// Get scheduled emails for a user
    pub async fn get_user_scheduled(&self, user_id: Uuid) -> Vec<ScheduledEmail> {
        let emails = self.scheduled_emails.read().await;

        emails.values()
            .filter(|e| {
                e.user_id == user_id &&
                matches!(e.status,
                    ScheduledEmailStatus::Scheduled |
                    ScheduledEmailStatus::PendingConfirmation
                )
            })
            .cloned()
            .collect()
    }

    /// Mark email as sending
    pub async fn mark_sending(&self, email_id: Uuid) -> Result<(), SchedulingError> {
        let mut emails = self.scheduled_emails.write().await;

        let email = emails.get_mut(&email_id)
            .ok_or(SchedulingError::NotFound(email_id))?;

        email.status = ScheduledEmailStatus::Sending;
        email.last_attempt_at = Some(Utc::now());
        email.attempt_count += 1;

        Ok(())
    }

    /// Mark email as sent
    pub async fn mark_sent(&self, email_id: Uuid) -> Result<(), SchedulingError> {
        let mut emails = self.scheduled_emails.write().await;

        let email = emails.get_mut(&email_id)
            .ok_or(SchedulingError::NotFound(email_id))?;

        email.status = ScheduledEmailStatus::Sent;
        email.sent_at = Some(Utc::now());

        Ok(())
    }

    /// Mark email as failed
    pub async fn mark_failed(
        &self,
        email_id: Uuid,
        error: String,
    ) -> Result<bool, SchedulingError> {
        let mut emails = self.scheduled_emails.write().await;

        let email = emails.get_mut(&email_id)
            .ok_or(SchedulingError::NotFound(email_id))?;

        email.last_error = Some(error);

        if email.attempt_count >= email.max_attempts {
            email.status = ScheduledEmailStatus::Failed;
            Ok(false) // No more retries
        } else {
            // Schedule retry with exponential backoff
            let delay = self.config.retry_delay_base_secs * (2u32.pow(email.attempt_count - 1));
            email.scheduled_for = Utc::now() + Duration::seconds(delay as i64);
            email.status = ScheduledEmailStatus::Scheduled;
            Ok(true) // Will retry
        }
    }

    /// Set user's undo window preference
    pub async fn set_user_undo_window(&self, user_id: Uuid, seconds: u32) {
        self.user_undo_settings.write().await.insert(user_id, seconds);
    }

    /// Get user's undo window (in seconds)
    async fn get_user_undo_window(&self, user_id: Uuid) -> u32 {
        self.user_undo_settings.read().await
            .get(&user_id)
            .copied()
            .unwrap_or(self.config.default_undo_window_secs)
    }

    /// Get remaining undo time for an email (in seconds)
    pub async fn get_remaining_undo_time(&self, email_id: Uuid) -> Option<i64> {
        let emails = self.scheduled_emails.read().await;

        emails.get(&email_id).and_then(|e| {
            if e.status == ScheduledEmailStatus::PendingConfirmation {
                let remaining = (e.undo_until - Utc::now()).num_seconds();
                if remaining > 0 {
                    Some(remaining)
                } else {
                    None
                }
            } else {
                None
            }
        })
    }

    /// Clean up old sent/cancelled emails
    pub async fn cleanup_old(&self, older_than_days: u32) {
        let cutoff = Utc::now() - Duration::days(older_than_days as i64);
        let mut emails = self.scheduled_emails.write().await;

        emails.retain(|_, e| {
            match e.status {
                ScheduledEmailStatus::Sent |
                ScheduledEmailStatus::Cancelled |
                ScheduledEmailStatus::Failed => {
                    e.created_at > cutoff
                }
                _ => true
            }
        });
    }
}

/// Email draft for scheduling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailDraft {
    pub to: Vec<String>,
    pub cc: Option<Vec<String>>,
    pub bcc: Option<Vec<String>>,
    pub subject: String,
    pub body_text: String,
    pub body_html: Option<String>,
    pub attachment_ids: Option<Vec<Uuid>>,
    pub reply_to_id: Option<Uuid>,
    pub timezone: Option<String>,
    pub send_later_reason: Option<String>,
}

/// Scheduling errors
#[derive(Debug, Clone)]
pub enum SchedulingError {
    NotFound(Uuid),
    Unauthorized,
    InvalidScheduleTime(String),
    AlreadySending,
    AlreadySent,
    AlreadyCancelled,
    CannotReschedule,
    SendFailed(String),
}

impl std::fmt::Display for SchedulingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "Scheduled email not found: {}", id),
            Self::Unauthorized => write!(f, "Not authorized to modify this email"),
            Self::InvalidScheduleTime(msg) => write!(f, "Invalid schedule time: {}", msg),
            Self::AlreadySending => write!(f, "Email is already being sent"),
            Self::AlreadySent => write!(f, "Email has already been sent"),
            Self::AlreadyCancelled => write!(f, "Email has already been cancelled"),
            Self::CannotReschedule => write!(f, "Cannot reschedule this email"),
            Self::SendFailed(e) => write!(f, "Failed to send: {}", e),
        }
    }
}

impl std::error::Error for SchedulingError {}

/// Smart send time suggestions
pub struct SendTimeSuggester;

impl SendTimeSuggester {
    /// Suggest optimal send times based on recipient timezone and engagement patterns
    pub fn suggest_send_times(
        recipient_timezone: &str,
        _engagement_history: Option<&[DateTime<Utc>]>,
    ) -> Vec<SuggestedSendTime> {
        // In a real implementation, this would analyze:
        // 1. Recipient's typical reading times
        // 2. Historical open rates by time
        // 3. Timezone considerations
        // 4. Business hours

        let tz = chrono_tz::Tz::from_str_insensitive(recipient_timezone)
            .unwrap_or(chrono_tz::UTC);

        let now = Utc::now().with_timezone(&tz);
        let today = now.date_naive();

        vec![
            SuggestedSendTime {
                time: today.and_hms_opt(9, 0, 0).unwrap().and_utc(),
                reason: "Start of business day - high open rates".to_string(),
                confidence: 0.85,
            },
            SuggestedSendTime {
                time: today.and_hms_opt(10, 30, 0).unwrap().and_utc(),
                reason: "Mid-morning - after inbox clearing".to_string(),
                confidence: 0.80,
            },
            SuggestedSendTime {
                time: today.and_hms_opt(14, 0, 0).unwrap().and_utc(),
                reason: "After lunch - email catch-up time".to_string(),
                confidence: 0.75,
            },
        ]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestedSendTime {
    pub time: DateTime<Utc>,
    pub reason: String,
    pub confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_send_with_undo() {
        let scheduler = EmailScheduler::new(SchedulingConfig {
            default_undo_window_secs: 10,
            ..Default::default()
        });

        let user_id = Uuid::new_v4();
        let draft = EmailDraft {
            to: vec!["recipient@example.com".to_string()],
            cc: None,
            bcc: None,
            subject: "Test".to_string(),
            body_text: "Hello".to_string(),
            body_html: None,
            attachment_ids: None,
            reply_to_id: None,
            timezone: None,
            send_later_reason: None,
        };

        let scheduled = scheduler.send_with_undo(user_id, draft).await.unwrap();

        assert_eq!(scheduled.status, ScheduledEmailStatus::PendingConfirmation);
        assert!(scheduled.undo_until > Utc::now());
    }

    #[tokio::test]
    async fn test_cancel_email() {
        let scheduler = EmailScheduler::new(SchedulingConfig::default());
        let user_id = Uuid::new_v4();

        let draft = EmailDraft {
            to: vec!["test@example.com".to_string()],
            cc: None, bcc: None,
            subject: "Test".to_string(),
            body_text: "Test".to_string(),
            body_html: None,
            attachment_ids: None,
            reply_to_id: None,
            timezone: None,
            send_later_reason: None,
        };

        let scheduled = scheduler.send_with_undo(user_id, draft).await.unwrap();

        scheduler.cancel(user_id, scheduled.id).await.unwrap();

        let emails = scheduler.scheduled_emails.read().await;
        assert_eq!(emails.get(&scheduled.id).unwrap().status, ScheduledEmailStatus::Cancelled);
    }
}
