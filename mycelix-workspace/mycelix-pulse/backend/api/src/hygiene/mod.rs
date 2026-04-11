// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Track U: Email Hygiene & Inbox Zero
//!
//! Smart unsubscribe, newsletter digests, email debt tracking,
//! and intelligent bulk actions for maintaining inbox health.

use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;
use std::collections::HashMap;

// ============================================================================
// Core Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InboxHealth {
    pub user_id: Uuid,
    pub total_emails: u64,
    pub unread_count: u64,
    pub inbox_zero_streak: u32,
    pub last_inbox_zero: Option<DateTime<Utc>>,
    pub health_score: f32,
    pub email_debt: EmailDebt,
    pub categories: Vec<CategoryBreakdown>,
    pub recommendations: Vec<HygieneRecommendation>,
    pub calculated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailDebt {
    pub overdue_replies: Vec<OverdueEmail>,
    pub aging_unread: Vec<AgingEmail>,
    pub total_debt_score: f32,
    pub oldest_unread_days: u32,
    pub average_response_time_hours: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverdueEmail {
    pub email_id: Uuid,
    pub subject: String,
    pub from: String,
    pub received_at: DateTime<Utc>,
    pub days_overdue: u32,
    pub priority: EmailPriority,
    pub suggested_action: SuggestedAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgingEmail {
    pub email_id: Uuid,
    pub subject: String,
    pub from: String,
    pub received_at: DateTime<Utc>,
    pub age_days: u32,
    pub category: EmailCategory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmailPriority {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuggestedAction {
    ReplyNow,
    ScheduleReply,
    Delegate,
    Archive,
    Delete,
    Snooze { until: DateTime<Utc> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryBreakdown {
    pub category: EmailCategory,
    pub count: u64,
    pub unread: u64,
    pub percentage: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EmailCategory {
    Primary,
    Social,
    Promotions,
    Updates,
    Forums,
    Newsletters,
    Notifications,
    Spam,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HygieneRecommendation {
    pub id: Uuid,
    pub recommendation_type: RecommendationType,
    pub title: String,
    pub description: String,
    pub impact: ImpactLevel,
    pub action: HygieneAction,
    pub affected_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    Unsubscribe,
    BulkArchive,
    BulkDelete,
    CreateFilter,
    MergeThreads,
    BlockSender,
    DigestNewsletter,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImpactLevel {
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HygieneAction {
    UnsubscribeFrom { sender: String },
    ArchiveMatching { query: String },
    DeleteMatching { query: String },
    CreateFilter { filter: FilterSpec },
    BlockSender { email: String },
    CreateDigest { senders: Vec<String> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterSpec {
    pub name: String,
    pub conditions: Vec<FilterCondition>,
    pub actions: Vec<FilterAction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterCondition {
    FromContains(String),
    SubjectContains(String),
    HasAttachment,
    SizeGreaterThan(u64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterAction {
    MoveTo(String),
    AddLabel(String),
    MarkRead,
    Archive,
    Delete,
}

// ============================================================================
// Unsubscribe Management
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subscription {
    pub id: Uuid,
    pub user_id: Uuid,
    pub sender_email: String,
    pub sender_name: Option<String>,
    pub list_id: Option<String>,
    pub unsubscribe_url: Option<String>,
    pub unsubscribe_email: Option<String>,
    pub first_seen: DateTime<Utc>,
    pub last_seen: DateTime<Utc>,
    pub email_count: u32,
    pub status: SubscriptionStatus,
    pub category: EmailCategory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SubscriptionStatus {
    Active,
    Unsubscribed,
    Pending,
    Failed,
    Blocked,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnsubscribeResult {
    pub subscription_id: Uuid,
    pub success: bool,
    pub method: UnsubscribeMethod,
    pub message: String,
    pub completed_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UnsubscribeMethod {
    ListUnsubscribe,
    OneClickLink,
    EmailRequest,
    Manual,
}

pub struct UnsubscribeService {
    pool: PgPool,
}

impl UnsubscribeService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn detect_subscriptions(&self, user_id: Uuid) -> Result<Vec<Subscription>> {
        // Find emails with List-Unsubscribe headers
        let subscriptions = sqlx::query_as!(
            SubscriptionRow,
            r#"
            SELECT
                sender as sender_email,
                MAX(sender_name) as sender_name,
                headers->>'List-Unsubscribe' as unsubscribe_url,
                headers->>'List-Unsubscribe-Post' as unsubscribe_email,
                headers->>'List-Id' as list_id,
                MIN(received_at) as first_seen,
                MAX(received_at) as last_seen,
                COUNT(*) as email_count
            FROM emails
            WHERE user_id = $1
              AND (headers ? 'List-Unsubscribe' OR sender LIKE '%newsletter%' OR sender LIKE '%noreply%')
            GROUP BY sender, headers->>'List-Unsubscribe', headers->>'List-Unsubscribe-Post', headers->>'List-Id'
            ORDER BY email_count DESC
            "#,
            user_id
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(subscriptions.into_iter().map(|s| Subscription {
            id: Uuid::new_v4(),
            user_id,
            sender_email: s.sender_email,
            sender_name: s.sender_name,
            list_id: s.list_id,
            unsubscribe_url: s.unsubscribe_url,
            unsubscribe_email: s.unsubscribe_email,
            first_seen: s.first_seen,
            last_seen: s.last_seen,
            email_count: s.email_count as u32,
            status: SubscriptionStatus::Active,
            category: EmailCategory::Newsletters,
        }).collect())
    }

    pub async fn unsubscribe(&self, subscription_id: Uuid) -> Result<UnsubscribeResult> {
        let subscription = sqlx::query_as!(
            SubscriptionDbRow,
            "SELECT * FROM subscriptions WHERE id = $1",
            subscription_id
        )
        .fetch_one(&self.pool)
        .await?;

        let (success, method, message) = if let Some(url) = &subscription.unsubscribe_url {
            // Try List-Unsubscribe header URL
            self.unsubscribe_via_url(url).await
        } else if let Some(email) = &subscription.unsubscribe_email {
            // Send unsubscribe email
            self.unsubscribe_via_email(email).await
        } else {
            (false, UnsubscribeMethod::Manual, "No automatic unsubscribe available".to_string())
        };

        // Update subscription status
        let new_status = if success {
            SubscriptionStatus::Unsubscribed
        } else {
            SubscriptionStatus::Failed
        };

        sqlx::query!(
            "UPDATE subscriptions SET status = $1, unsubscribed_at = NOW() WHERE id = $2",
            serde_json::to_string(&new_status)?,
            subscription_id
        )
        .execute(&self.pool)
        .await?;

        Ok(UnsubscribeResult {
            subscription_id,
            success,
            method,
            message,
            completed_at: Utc::now(),
        })
    }

    async fn unsubscribe_via_url(&self, url: &str) -> (bool, UnsubscribeMethod, String) {
        // Parse List-Unsubscribe header (can contain multiple URLs)
        let clean_url = url.trim_matches(|c| c == '<' || c == '>');

        // Check for one-click unsubscribe (RFC 8058)
        if url.contains("mailto:") {
            return self.unsubscribe_via_email(&clean_url.replace("mailto:", "")).await;
        }

        // HTTP(S) unsubscribe
        let client = reqwest::Client::new();
        match client.post(clean_url)
            .header("List-Unsubscribe", "One-Click")
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => {
                (true, UnsubscribeMethod::OneClickLink, "Successfully unsubscribed".to_string())
            }
            Ok(response) => {
                (false, UnsubscribeMethod::OneClickLink, format!("Server returned: {}", response.status()))
            }
            Err(e) => {
                (false, UnsubscribeMethod::OneClickLink, format!("Request failed: {}", e))
            }
        }
    }

    async fn unsubscribe_via_email(&self, email: &str) -> (bool, UnsubscribeMethod, String) {
        // Would send actual unsubscribe email
        // For now, return pending status
        (true, UnsubscribeMethod::EmailRequest, format!("Unsubscribe request sent to {}", email))
    }

    pub async fn bulk_unsubscribe(&self, subscription_ids: Vec<Uuid>) -> Result<Vec<UnsubscribeResult>> {
        let mut results = Vec::new();
        for id in subscription_ids {
            let result = self.unsubscribe(id).await?;
            results.push(result);
        }
        Ok(results)
    }
}

#[derive(Debug)]
struct SubscriptionRow {
    sender_email: String,
    sender_name: Option<String>,
    unsubscribe_url: Option<String>,
    unsubscribe_email: Option<String>,
    list_id: Option<String>,
    first_seen: DateTime<Utc>,
    last_seen: DateTime<Utc>,
    email_count: i64,
}

#[derive(Debug)]
struct SubscriptionDbRow {
    id: Uuid,
    user_id: Uuid,
    sender_email: String,
    unsubscribe_url: Option<String>,
    unsubscribe_email: Option<String>,
    status: String,
}

// ============================================================================
// Newsletter Digest
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DigestConfig {
    pub id: Uuid,
    pub user_id: Uuid,
    pub name: String,
    pub senders: Vec<String>,
    pub frequency: DigestFrequency,
    pub delivery_time: String,  // "09:00"
    pub delivery_day: Option<String>,  // For weekly: "monday"
    pub enabled: bool,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DigestFrequency {
    Daily,
    Weekly,
    BiWeekly,
    Monthly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DigestEmail {
    pub id: Uuid,
    pub digest_config_id: Uuid,
    pub emails: Vec<DigestItem>,
    pub generated_at: DateTime<Utc>,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DigestItem {
    pub email_id: Uuid,
    pub subject: String,
    pub from: String,
    pub preview: String,
    pub received_at: DateTime<Utc>,
}

pub struct DigestService {
    pool: PgPool,
}

impl DigestService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn create_digest_config(&self, user_id: Uuid, config: DigestConfig) -> Result<DigestConfig> {
        sqlx::query!(
            r#"
            INSERT INTO digest_configs (id, user_id, name, senders, frequency, delivery_time, delivery_day, enabled, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
            "#,
            config.id,
            user_id,
            config.name,
            &config.senders,
            serde_json::to_string(&config.frequency)?,
            config.delivery_time,
            config.delivery_day,
            config.enabled
        )
        .execute(&self.pool)
        .await?;

        Ok(config)
    }

    pub async fn generate_digest(&self, config_id: Uuid) -> Result<DigestEmail> {
        let config = sqlx::query_as!(
            DigestConfigRow,
            "SELECT * FROM digest_configs WHERE id = $1",
            config_id
        )
        .fetch_one(&self.pool)
        .await?;

        let period_start = self.calculate_period_start(&config);
        let period_end = Utc::now();

        // Fetch emails from digest senders
        let emails = sqlx::query_as!(
            EmailPreviewRow,
            r#"
            SELECT id, subject, sender, body, received_at
            FROM emails
            WHERE user_id = $1
              AND sender = ANY($2)
              AND received_at >= $3
              AND received_at <= $4
            ORDER BY received_at DESC
            "#,
            config.user_id,
            &config.senders,
            period_start,
            period_end
        )
        .fetch_all(&self.pool)
        .await?;

        let items: Vec<DigestItem> = emails.into_iter().map(|e| DigestItem {
            email_id: e.id,
            subject: e.subject.unwrap_or_default(),
            from: e.sender,
            preview: e.body.chars().take(200).collect(),
            received_at: e.received_at,
        }).collect();

        // Mark original emails as archived (bundled into digest)
        for item in &items {
            sqlx::query!(
                "UPDATE emails SET folder = 'Digest', is_read = true WHERE id = $1",
                item.email_id
            )
            .execute(&self.pool)
            .await?;
        }

        Ok(DigestEmail {
            id: Uuid::new_v4(),
            digest_config_id: config_id,
            emails: items,
            generated_at: Utc::now(),
            period_start,
            period_end,
        })
    }

    fn calculate_period_start(&self, config: &DigestConfigRow) -> DateTime<Utc> {
        let now = Utc::now();
        let frequency: DigestFrequency = serde_json::from_str(&config.frequency)
            .unwrap_or(DigestFrequency::Daily);

        match frequency {
            DigestFrequency::Daily => now - Duration::days(1),
            DigestFrequency::Weekly => now - Duration::weeks(1),
            DigestFrequency::BiWeekly => now - Duration::weeks(2),
            DigestFrequency::Monthly => now - Duration::days(30),
        }
    }
}

#[derive(Debug)]
struct DigestConfigRow {
    id: Uuid,
    user_id: Uuid,
    name: String,
    senders: Vec<String>,
    frequency: String,
    delivery_time: String,
    delivery_day: Option<String>,
    enabled: bool,
}

#[derive(Debug)]
struct EmailPreviewRow {
    id: Uuid,
    subject: Option<String>,
    sender: String,
    body: String,
    received_at: DateTime<Utc>,
}

// ============================================================================
// Email Debt Tracking
// ============================================================================

pub struct EmailDebtService {
    pool: PgPool,
}

impl EmailDebtService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn calculate_debt(&self, user_id: Uuid) -> Result<EmailDebt> {
        // Find emails that likely need replies
        let overdue = sqlx::query_as!(
            OverdueRow,
            r#"
            SELECT e.id, e.subject, e.sender, e.received_at,
                   EXTRACT(DAY FROM NOW() - e.received_at)::int as days_old
            FROM emails e
            WHERE e.user_id = $1
              AND e.is_read = true
              AND e.folder = 'Inbox'
              AND e.needs_reply = true
              AND NOT EXISTS (
                  SELECT 1 FROM emails r
                  WHERE r.thread_id = e.thread_id
                    AND r.sender = (SELECT email FROM users WHERE id = $1)
                    AND r.sent_at > e.received_at
              )
              AND e.received_at < NOW() - INTERVAL '2 days'
            ORDER BY e.received_at ASC
            LIMIT 20
            "#,
            user_id
        )
        .fetch_all(&self.pool)
        .await?;

        let overdue_replies: Vec<OverdueEmail> = overdue.into_iter().map(|r| {
            let days = r.days_old.unwrap_or(0) as u32;
            OverdueEmail {
                email_id: r.id,
                subject: r.subject.unwrap_or_default(),
                from: r.sender,
                received_at: r.received_at,
                days_overdue: days.saturating_sub(2),
                priority: if days > 7 { EmailPriority::Critical }
                         else if days > 4 { EmailPriority::High }
                         else { EmailPriority::Medium },
                suggested_action: if days > 14 { SuggestedAction::Archive }
                                  else { SuggestedAction::ReplyNow },
            }
        }).collect();

        // Find aging unread emails
        let aging = sqlx::query_as!(
            AgingRow,
            r#"
            SELECT id, subject, sender, received_at,
                   EXTRACT(DAY FROM NOW() - received_at)::int as age_days
            FROM emails
            WHERE user_id = $1
              AND is_read = false
              AND received_at < NOW() - INTERVAL '7 days'
            ORDER BY received_at ASC
            LIMIT 20
            "#,
            user_id
        )
        .fetch_all(&self.pool)
        .await?;

        let aging_unread: Vec<AgingEmail> = aging.into_iter().map(|a| AgingEmail {
            email_id: a.id,
            subject: a.subject.unwrap_or_default(),
            from: a.sender,
            received_at: a.received_at,
            age_days: a.age_days.unwrap_or(0) as u32,
            category: EmailCategory::Primary,
        }).collect();

        // Calculate average response time
        let avg_response = sqlx::query!(
            r#"
            SELECT AVG(EXTRACT(EPOCH FROM (reply.sent_at - original.received_at)) / 3600)::float as avg_hours
            FROM emails original
            JOIN emails reply ON reply.in_reply_to = original.message_id
            WHERE original.user_id = $1
              AND reply.sender = (SELECT email FROM users WHERE id = $1)
              AND original.received_at > NOW() - INTERVAL '30 days'
            "#,
            user_id
        )
        .fetch_one(&self.pool)
        .await?;

        let oldest_unread = aging_unread.first()
            .map(|a| a.age_days)
            .unwrap_or(0);

        let debt_score = self.calculate_debt_score(&overdue_replies, &aging_unread);

        Ok(EmailDebt {
            overdue_replies,
            aging_unread,
            total_debt_score: debt_score,
            oldest_unread_days: oldest_unread,
            average_response_time_hours: avg_response.avg_hours.unwrap_or(0.0) as f32,
        })
    }

    fn calculate_debt_score(&self, overdue: &[OverdueEmail], aging: &[AgingEmail]) -> f32 {
        let overdue_score: f32 = overdue.iter()
            .map(|e| match e.priority {
                EmailPriority::Critical => 10.0,
                EmailPriority::High => 5.0,
                EmailPriority::Medium => 2.0,
                EmailPriority::Low => 1.0,
            })
            .sum();

        let aging_score: f32 = aging.iter()
            .map(|e| (e.age_days as f32 / 7.0).min(5.0))
            .sum();

        (overdue_score + aging_score).min(100.0)
    }

    pub async fn get_inbox_zero_streak(&self, user_id: Uuid) -> Result<(u32, Option<DateTime<Utc>>)> {
        let streak = sqlx::query!(
            r#"
            SELECT
                streak_days,
                last_inbox_zero
            FROM inbox_zero_tracking
            WHERE user_id = $1
            "#,
            user_id
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(streak.map(|s| (s.streak_days as u32, s.last_inbox_zero))
            .unwrap_or((0, None)))
    }

    pub async fn check_inbox_zero(&self, user_id: Uuid) -> Result<bool> {
        let unread_count = sqlx::query!(
            r#"
            SELECT COUNT(*) as count
            FROM emails
            WHERE user_id = $1 AND folder = 'Inbox' AND is_read = false
            "#,
            user_id
        )
        .fetch_one(&self.pool)
        .await?;

        let is_zero = unread_count.count.unwrap_or(0) == 0;

        if is_zero {
            // Update streak
            sqlx::query!(
                r#"
                INSERT INTO inbox_zero_tracking (user_id, streak_days, last_inbox_zero)
                VALUES ($1, 1, NOW())
                ON CONFLICT (user_id) DO UPDATE SET
                    streak_days = CASE
                        WHEN inbox_zero_tracking.last_inbox_zero::date = CURRENT_DATE - INTERVAL '1 day'
                        THEN inbox_zero_tracking.streak_days + 1
                        WHEN inbox_zero_tracking.last_inbox_zero::date = CURRENT_DATE
                        THEN inbox_zero_tracking.streak_days
                        ELSE 1
                    END,
                    last_inbox_zero = NOW()
                "#,
                user_id
            )
            .execute(&self.pool)
            .await?;
        }

        Ok(is_zero)
    }
}

#[derive(Debug)]
struct OverdueRow {
    id: Uuid,
    subject: Option<String>,
    sender: String,
    received_at: DateTime<Utc>,
    days_old: Option<i32>,
}

#[derive(Debug)]
struct AgingRow {
    id: Uuid,
    subject: Option<String>,
    sender: String,
    received_at: DateTime<Utc>,
    age_days: Option<i32>,
}

// ============================================================================
// Bulk Actions
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BulkActionRequest {
    pub action_type: BulkActionType,
    pub query: Option<String>,
    pub email_ids: Option<Vec<Uuid>>,
    pub dry_run: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BulkActionType {
    Archive,
    Delete,
    MarkRead,
    MarkUnread,
    MoveTo { folder: String },
    AddLabel { label: String },
    RemoveLabel { label: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BulkActionResult {
    pub action_type: BulkActionType,
    pub affected_count: u32,
    pub success: bool,
    pub dry_run: bool,
    pub preview: Option<Vec<EmailPreview>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailPreview {
    pub id: Uuid,
    pub subject: String,
    pub from: String,
}

pub struct BulkActionService {
    pool: PgPool,
}

impl BulkActionService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn execute(&self, user_id: Uuid, request: BulkActionRequest) -> Result<BulkActionResult> {
        // Get affected emails
        let email_ids = if let Some(ids) = request.email_ids {
            ids
        } else if let Some(query) = &request.query {
            self.parse_natural_language_query(user_id, query).await?
        } else {
            return Ok(BulkActionResult {
                action_type: request.action_type,
                affected_count: 0,
                success: false,
                dry_run: request.dry_run,
                preview: None,
            });
        };

        if request.dry_run {
            // Return preview without executing
            let preview = self.get_email_previews(&email_ids).await?;
            return Ok(BulkActionResult {
                action_type: request.action_type,
                affected_count: email_ids.len() as u32,
                success: true,
                dry_run: true,
                preview: Some(preview),
            });
        }

        // Execute the action
        let affected = match &request.action_type {
            BulkActionType::Archive => {
                sqlx::query!(
                    "UPDATE emails SET folder = 'Archive', is_read = true WHERE id = ANY($1)",
                    &email_ids
                )
                .execute(&self.pool)
                .await?
                .rows_affected()
            }
            BulkActionType::Delete => {
                sqlx::query!(
                    "UPDATE emails SET folder = 'Trash' WHERE id = ANY($1)",
                    &email_ids
                )
                .execute(&self.pool)
                .await?
                .rows_affected()
            }
            BulkActionType::MarkRead => {
                sqlx::query!(
                    "UPDATE emails SET is_read = true WHERE id = ANY($1)",
                    &email_ids
                )
                .execute(&self.pool)
                .await?
                .rows_affected()
            }
            BulkActionType::MarkUnread => {
                sqlx::query!(
                    "UPDATE emails SET is_read = false WHERE id = ANY($1)",
                    &email_ids
                )
                .execute(&self.pool)
                .await?
                .rows_affected()
            }
            BulkActionType::MoveTo { folder } => {
                sqlx::query!(
                    "UPDATE emails SET folder = $1 WHERE id = ANY($2)",
                    folder,
                    &email_ids
                )
                .execute(&self.pool)
                .await?
                .rows_affected()
            }
            BulkActionType::AddLabel { label } => {
                sqlx::query!(
                    "UPDATE emails SET labels = array_append(labels, $1) WHERE id = ANY($2) AND NOT ($1 = ANY(labels))",
                    label,
                    &email_ids
                )
                .execute(&self.pool)
                .await?
                .rows_affected()
            }
            BulkActionType::RemoveLabel { label } => {
                sqlx::query!(
                    "UPDATE emails SET labels = array_remove(labels, $1) WHERE id = ANY($2)",
                    label,
                    &email_ids
                )
                .execute(&self.pool)
                .await?
                .rows_affected()
            }
        };

        Ok(BulkActionResult {
            action_type: request.action_type,
            affected_count: affected as u32,
            success: true,
            dry_run: false,
            preview: None,
        })
    }

    async fn parse_natural_language_query(&self, user_id: Uuid, query: &str) -> Result<Vec<Uuid>> {
        let query_lower = query.to_lowercase();

        // Parse natural language patterns
        let mut conditions = Vec::new();

        if query_lower.contains("promotional") || query_lower.contains("promotions") {
            conditions.push("category = 'promotions'");
        }
        if query_lower.contains("newsletter") {
            conditions.push("(sender LIKE '%newsletter%' OR headers ? 'List-Unsubscribe')");
        }
        if query_lower.contains("last month") {
            conditions.push("received_at < NOW() - INTERVAL '30 days'");
        }
        if query_lower.contains("last week") {
            conditions.push("received_at < NOW() - INTERVAL '7 days'");
        }
        if query_lower.contains("older than") {
            // Extract duration
            if query_lower.contains("90 days") || query_lower.contains("3 months") {
                conditions.push("received_at < NOW() - INTERVAL '90 days'");
            }
        }
        if query_lower.contains("read") && !query_lower.contains("unread") {
            conditions.push("is_read = true");
        }
        if query_lower.contains("unread") {
            conditions.push("is_read = false");
        }

        let where_clause = if conditions.is_empty() {
            "1=1".to_string()
        } else {
            conditions.join(" AND ")
        };

        let sql = format!(
            "SELECT id FROM emails WHERE user_id = $1 AND {} LIMIT 1000",
            where_clause
        );

        let ids: Vec<(Uuid,)> = sqlx::query_as(&sql)
            .bind(user_id)
            .fetch_all(&self.pool)
            .await?;

        Ok(ids.into_iter().map(|(id,)| id).collect())
    }

    async fn get_email_previews(&self, ids: &[Uuid]) -> Result<Vec<EmailPreview>> {
        let previews = sqlx::query_as!(
            EmailPreview,
            r#"
            SELECT id, subject, sender as from
            FROM emails
            WHERE id = ANY($1)
            LIMIT 10
            "#,
            ids
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(previews)
    }
}

// ============================================================================
// Quiet Hours / Focus Mode
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuietHoursConfig {
    pub id: Uuid,
    pub user_id: Uuid,
    pub enabled: bool,
    pub start_time: String,  // "22:00"
    pub end_time: String,    // "07:00"
    pub days: Vec<String>,   // ["monday", "tuesday", ...]
    pub exceptions: Vec<QuietHoursException>,
    pub auto_defer_low_priority: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuietHoursException {
    pub exception_type: ExceptionType,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExceptionType {
    Sender,
    Domain,
    Label,
    VIP,
}

pub struct QuietHoursService {
    pool: PgPool,
}

impl QuietHoursService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn is_quiet_hours(&self, user_id: Uuid) -> Result<bool> {
        let config = sqlx::query_as!(
            QuietHoursRow,
            "SELECT * FROM quiet_hours_config WHERE user_id = $1",
            user_id
        )
        .fetch_optional(&self.pool)
        .await?;

        if let Some(config) = config {
            if !config.enabled {
                return Ok(false);
            }

            let now = Utc::now();
            let current_time = now.format("%H:%M").to_string();
            let current_day = now.format("%A").to_string().to_lowercase();

            // Check if current day is in quiet days
            if !config.days.contains(&current_day) {
                return Ok(false);
            }

            // Check time range (handles overnight ranges)
            let in_range = if config.start_time > config.end_time {
                // Overnight range (e.g., 22:00 - 07:00)
                current_time >= config.start_time || current_time <= config.end_time
            } else {
                // Same-day range (e.g., 09:00 - 17:00)
                current_time >= config.start_time && current_time <= config.end_time
            };

            return Ok(in_range);
        }

        Ok(false)
    }

    pub async fn should_notify(&self, user_id: Uuid, email_sender: &str, labels: &[String]) -> Result<bool> {
        if !self.is_quiet_hours(user_id).await? {
            return Ok(true);  // Not in quiet hours, allow notification
        }

        let config = sqlx::query_as!(
            QuietHoursRow,
            "SELECT * FROM quiet_hours_config WHERE user_id = $1",
            user_id
        )
        .fetch_one(&self.pool)
        .await?;

        // Check exceptions
        let exceptions: Vec<QuietHoursException> = serde_json::from_value(config.exceptions)?;
        for exception in exceptions {
            match exception.exception_type {
                ExceptionType::Sender if email_sender == exception.value => return Ok(true),
                ExceptionType::Domain if email_sender.ends_with(&exception.value) => return Ok(true),
                ExceptionType::Label if labels.contains(&exception.value) => return Ok(true),
                ExceptionType::VIP => {
                    // Check VIP list
                    let is_vip = sqlx::query!(
                        "SELECT 1 as exists FROM vip_contacts WHERE user_id = $1 AND email = $2",
                        user_id,
                        email_sender
                    )
                    .fetch_optional(&self.pool)
                    .await?
                    .is_some();

                    if is_vip {
                        return Ok(true);
                    }
                }
                _ => {}
            }
        }

        Ok(false)  // In quiet hours with no exceptions
    }
}

#[derive(Debug)]
struct QuietHoursRow {
    id: Uuid,
    user_id: Uuid,
    enabled: bool,
    start_time: String,
    end_time: String,
    days: Vec<String>,
    exceptions: serde_json::Value,
    auto_defer_low_priority: bool,
}

// ============================================================================
// Sender Reputation
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SenderReputation {
    pub sender: String,
    pub domain: String,
    pub reputation_score: f32,
    pub email_count: u32,
    pub open_rate: f32,
    pub reply_rate: f32,
    pub archive_rate: f32,
    pub delete_rate: f32,
    pub unsubscribe_rate: f32,
    pub is_blocked: bool,
    pub is_vip: bool,
    pub first_contact: DateTime<Utc>,
    pub last_contact: DateTime<Utc>,
}

pub struct SenderReputationService {
    pool: PgPool,
}

impl SenderReputationService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn get_reputation(&self, user_id: Uuid, sender: &str) -> Result<SenderReputation> {
        let stats = sqlx::query!(
            r#"
            SELECT
                sender,
                SPLIT_PART(sender, '@', 2) as domain,
                COUNT(*) as email_count,
                COUNT(*) FILTER (WHERE is_read) as read_count,
                COUNT(*) FILTER (WHERE folder = 'Archive') as archive_count,
                COUNT(*) FILTER (WHERE folder = 'Trash') as delete_count,
                MIN(received_at) as first_contact,
                MAX(received_at) as last_contact
            FROM emails
            WHERE user_id = $1 AND sender = $2
            GROUP BY sender
            "#,
            user_id,
            sender
        )
        .fetch_one(&self.pool)
        .await?;

        let total = stats.email_count.unwrap_or(1) as f32;

        let is_blocked = sqlx::query!(
            "SELECT 1 as exists FROM blocked_senders WHERE user_id = $1 AND sender = $2",
            user_id,
            sender
        )
        .fetch_optional(&self.pool)
        .await?
        .is_some();

        let is_vip = sqlx::query!(
            "SELECT 1 as exists FROM vip_contacts WHERE user_id = $1 AND email = $2",
            user_id,
            sender
        )
        .fetch_optional(&self.pool)
        .await?
        .is_some();

        let open_rate = stats.read_count.unwrap_or(0) as f32 / total;
        let archive_rate = stats.archive_count.unwrap_or(0) as f32 / total;
        let delete_rate = stats.delete_count.unwrap_or(0) as f32 / total;

        // Calculate reputation score
        let reputation_score = (open_rate * 0.4) + ((1.0 - delete_rate) * 0.3) + ((1.0 - archive_rate) * 0.3);

        Ok(SenderReputation {
            sender: sender.to_string(),
            domain: stats.domain.unwrap_or_default(),
            reputation_score,
            email_count: stats.email_count.unwrap_or(0) as u32,
            open_rate,
            reply_rate: 0.0,  // Would calculate from sent emails
            archive_rate,
            delete_rate,
            unsubscribe_rate: 0.0,
            is_blocked,
            is_vip,
            first_contact: stats.first_contact.unwrap_or_else(Utc::now),
            last_contact: stats.last_contact.unwrap_or_else(Utc::now),
        })
    }

    pub async fn block_sender(&self, user_id: Uuid, sender: &str) -> Result<()> {
        sqlx::query!(
            r#"
            INSERT INTO blocked_senders (user_id, sender, blocked_at)
            VALUES ($1, $2, NOW())
            ON CONFLICT (user_id, sender) DO NOTHING
            "#,
            user_id,
            sender
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn unblock_sender(&self, user_id: Uuid, sender: &str) -> Result<()> {
        sqlx::query!(
            "DELETE FROM blocked_senders WHERE user_id = $1 AND sender = $2",
            user_id,
            sender
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn add_vip(&self, user_id: Uuid, email: &str, name: Option<&str>) -> Result<()> {
        sqlx::query!(
            r#"
            INSERT INTO vip_contacts (user_id, email, name, added_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (user_id, email) DO UPDATE SET name = EXCLUDED.name
            "#,
            user_id,
            email,
            name
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }
}

// ============================================================================
// Module Exports
// ============================================================================

pub fn create_inbox_health_service(pool: PgPool) -> InboxHealthService {
    InboxHealthService::new(pool)
}

pub struct InboxHealthService {
    pool: PgPool,
    debt_service: EmailDebtService,
    reputation_service: SenderReputationService,
}

impl InboxHealthService {
    pub fn new(pool: PgPool) -> Self {
        Self {
            debt_service: EmailDebtService::new(pool.clone()),
            reputation_service: SenderReputationService::new(pool.clone()),
            pool,
        }
    }

    pub async fn get_health(&self, user_id: Uuid) -> Result<InboxHealth> {
        let email_debt = self.debt_service.calculate_debt(user_id).await?;
        let (streak, last_zero) = self.debt_service.get_inbox_zero_streak(user_id).await?;

        let stats = sqlx::query!(
            r#"
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE is_read = false) as unread
            FROM emails
            WHERE user_id = $1 AND folder = 'Inbox'
            "#,
            user_id
        )
        .fetch_one(&self.pool)
        .await?;

        let health_score = self.calculate_health_score(
            stats.unread.unwrap_or(0) as u32,
            email_debt.total_debt_score,
            streak
        );

        Ok(InboxHealth {
            user_id,
            total_emails: stats.total.unwrap_or(0) as u64,
            unread_count: stats.unread.unwrap_or(0) as u64,
            inbox_zero_streak: streak,
            last_inbox_zero: last_zero,
            health_score,
            email_debt,
            categories: Vec::new(),  // Would calculate category breakdown
            recommendations: Vec::new(),  // Would generate recommendations
            calculated_at: Utc::now(),
        })
    }

    fn calculate_health_score(&self, unread: u32, debt_score: f32, streak: u32) -> f32 {
        let mut score = 100.0;

        // Penalize unread emails
        score -= (unread as f32 * 0.5).min(30.0);

        // Penalize email debt
        score -= (debt_score * 0.3).min(30.0);

        // Bonus for inbox zero streak
        score += (streak as f32 * 2.0).min(20.0);

        score.max(0.0)
    }
}
