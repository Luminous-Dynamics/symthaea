// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Track CF: Slack/Teams Notification Bridge
// Real-time notifications to collaboration platforms

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Supported notification platforms
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NotificationPlatform {
    Slack,
    MicrosoftTeams,
    Discord,
    Mattermost,
    CustomWebhook,
}

/// Notification channel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationChannel {
    pub id: Uuid,
    pub user_id: Uuid,
    pub platform: NotificationPlatform,
    pub name: String,
    pub webhook_url: String,
    pub enabled: bool,
    pub rules: Vec<NotificationRule>,
    pub settings: ChannelSettings,
    pub created_at: DateTime<Utc>,
    pub last_notification: Option<DateTime<Utc>>,
}

/// Channel-specific settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelSettings {
    /// Bot token for API access (Slack/Discord)
    pub bot_token: Option<String>,
    /// Channel/room ID for API calls
    pub channel_id: Option<String>,
    /// Custom bot name
    pub bot_name: Option<String>,
    /// Custom bot icon URL
    pub bot_icon: Option<String>,
    /// Enable rich formatting
    pub rich_formatting: bool,
    /// Include email preview
    pub include_preview: bool,
    /// Maximum preview length
    pub max_preview_length: usize,
    /// Quiet hours (no notifications)
    pub quiet_hours: Option<QuietHours>,
    /// Rate limit (max notifications per hour)
    pub rate_limit: Option<u32>,
}

impl Default for ChannelSettings {
    fn default() -> Self {
        Self {
            bot_token: None,
            channel_id: None,
            bot_name: Some("Mycelix Mail".to_string()),
            bot_icon: None,
            rich_formatting: true,
            include_preview: true,
            max_preview_length: 200,
            quiet_hours: None,
            rate_limit: Some(60),
        }
    }
}

/// Quiet hours configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuietHours {
    pub start_hour: u8,
    pub end_hour: u8,
    pub timezone: String,
    pub weekends_only: bool,
}

/// Notification trigger rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationRule {
    pub id: Uuid,
    pub name: String,
    pub enabled: bool,
    pub trigger: NotificationTrigger,
    pub filters: NotificationFilters,
    pub priority: NotificationPriority,
    pub template: Option<String>,
}

/// What triggers the notification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NotificationTrigger {
    /// New email received
    EmailReceived,
    /// Email from high trust sender
    HighTrustEmail { min_score: f64 },
    /// Email from VIP/important sender
    VipEmail,
    /// Email with specific labels
    LabeledEmail { labels: Vec<String> },
    /// Email matching search query
    SearchMatch { query: String },
    /// Calendar event reminder
    CalendarReminder { minutes_before: u32 },
    /// Task deadline approaching
    TaskDeadline { hours_before: u32 },
    /// Security alert (suspicious login, etc.)
    SecurityAlert,
    /// Daily digest
    DailyDigest { hour: u8 },
    /// Weekly summary
    WeeklySummary { day: String, hour: u8 },
}

/// Filters to refine notifications
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NotificationFilters {
    pub from_domains: Option<Vec<String>>,
    pub from_emails: Option<Vec<String>>,
    pub exclude_domains: Option<Vec<String>>,
    pub subject_contains: Option<Vec<String>>,
    pub min_trust_score: Option<f64>,
    pub has_attachments: Option<bool>,
    pub is_reply: Option<bool>,
}

/// Notification urgency
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NotificationPriority {
    Low,
    Normal,
    High,
    Urgent,
}

/// Notification payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationPayload {
    pub id: Uuid,
    pub channel_id: Uuid,
    pub trigger: NotificationTrigger,
    pub title: String,
    pub message: String,
    pub preview: Option<String>,
    pub url: Option<String>,
    pub metadata: HashMap<String, String>,
    pub priority: NotificationPriority,
    pub created_at: DateTime<Utc>,
}

/// Delivery result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryResult {
    pub notification_id: Uuid,
    pub channel_id: Uuid,
    pub success: bool,
    pub platform_response: Option<String>,
    pub error: Option<String>,
    pub delivered_at: DateTime<Utc>,
}

/// Notification service
pub struct NotificationService {
    channels: HashMap<Uuid, NotificationChannel>,
    history: Vec<DeliveryResult>,
    rate_counters: HashMap<Uuid, RateCounter>,
}

struct RateCounter {
    count: u32,
    window_start: DateTime<Utc>,
}

impl Default for NotificationService {
    fn default() -> Self {
        Self::new()
    }
}

impl NotificationService {
    pub fn new() -> Self {
        Self {
            channels: HashMap::new(),
            history: Vec::new(),
            rate_counters: HashMap::new(),
        }
    }

    /// Create a new notification channel
    pub async fn create_channel(
        &mut self,
        user_id: Uuid,
        platform: NotificationPlatform,
        name: String,
        webhook_url: String,
        settings: Option<ChannelSettings>,
    ) -> Result<NotificationChannel, NotificationError> {
        // Validate webhook URL
        self.validate_webhook_url(&webhook_url, &platform)?;

        let channel = NotificationChannel {
            id: Uuid::new_v4(),
            user_id,
            platform,
            name,
            webhook_url,
            enabled: true,
            rules: Vec::new(),
            settings: settings.unwrap_or_default(),
            created_at: Utc::now(),
            last_notification: None,
        };

        self.channels.insert(channel.id, channel.clone());

        // Send test notification
        self.send_test_notification(channel.id).await?;

        Ok(channel)
    }

    /// Validate webhook URL format for platform
    fn validate_webhook_url(&self, url: &str, platform: &NotificationPlatform) -> Result<(), NotificationError> {
        match platform {
            NotificationPlatform::Slack => {
                if !url.starts_with("https://hooks.slack.com/") {
                    return Err(NotificationError::InvalidWebhookUrl(
                        "Slack webhooks must start with https://hooks.slack.com/".to_string()
                    ));
                }
            }
            NotificationPlatform::MicrosoftTeams => {
                if !url.contains("webhook.office.com") && !url.contains("microsoft.com") {
                    return Err(NotificationError::InvalidWebhookUrl(
                        "Teams webhooks must be from webhook.office.com or microsoft.com".to_string()
                    ));
                }
            }
            NotificationPlatform::Discord => {
                if !url.starts_with("https://discord.com/api/webhooks/") &&
                   !url.starts_with("https://discordapp.com/api/webhooks/") {
                    return Err(NotificationError::InvalidWebhookUrl(
                        "Discord webhooks must start with https://discord.com/api/webhooks/".to_string()
                    ));
                }
            }
            NotificationPlatform::Mattermost => {
                if !url.contains("/hooks/") {
                    return Err(NotificationError::InvalidWebhookUrl(
                        "Mattermost webhooks must contain /hooks/".to_string()
                    ));
                }
            }
            NotificationPlatform::CustomWebhook => {
                if !url.starts_with("https://") {
                    return Err(NotificationError::InvalidWebhookUrl(
                        "Custom webhooks must use HTTPS".to_string()
                    ));
                }
            }
        }
        Ok(())
    }

    /// Add a notification rule to a channel
    pub async fn add_rule(
        &mut self,
        channel_id: Uuid,
        name: String,
        trigger: NotificationTrigger,
        filters: NotificationFilters,
        priority: NotificationPriority,
    ) -> Result<NotificationRule, NotificationError> {
        let channel = self.channels.get_mut(&channel_id)
            .ok_or(NotificationError::ChannelNotFound)?;

        let rule = NotificationRule {
            id: Uuid::new_v4(),
            name,
            enabled: true,
            trigger,
            filters,
            priority,
            template: None,
        };

        channel.rules.push(rule.clone());

        Ok(rule)
    }

    /// Remove a rule
    pub async fn remove_rule(&mut self, channel_id: Uuid, rule_id: Uuid) -> Result<(), NotificationError> {
        let channel = self.channels.get_mut(&channel_id)
            .ok_or(NotificationError::ChannelNotFound)?;

        channel.rules.retain(|r| r.id != rule_id);
        Ok(())
    }

    /// Process incoming email for notifications
    pub async fn process_email_notification(
        &mut self,
        user_id: Uuid,
        email_from: &str,
        email_subject: &str,
        email_preview: &str,
        email_id: Uuid,
        trust_score: f64,
        labels: &[String],
    ) -> Vec<DeliveryResult> {
        let mut results = Vec::new();

        // Find all channels for this user
        let channel_ids: Vec<Uuid> = self.channels.iter()
            .filter(|(_, c)| c.user_id == user_id && c.enabled)
            .map(|(id, _)| *id)
            .collect();

        for channel_id in channel_ids {
            let channel = match self.channels.get(&channel_id) {
                Some(c) => c.clone(),
                None => continue,
            };

            // Check each rule
            for rule in &channel.rules {
                if !rule.enabled {
                    continue;
                }

                // Check if rule matches
                if !self.rule_matches(
                    &rule.trigger,
                    &rule.filters,
                    email_from,
                    email_subject,
                    trust_score,
                    labels,
                ) {
                    continue;
                }

                // Check quiet hours
                if self.is_quiet_hours(&channel.settings) {
                    continue;
                }

                // Check rate limit
                if !self.check_rate_limit(channel_id, &channel.settings) {
                    continue;
                }

                // Build notification
                let payload = NotificationPayload {
                    id: Uuid::new_v4(),
                    channel_id,
                    trigger: rule.trigger.clone(),
                    title: format!("New email from {}", email_from),
                    message: email_subject.to_string(),
                    preview: if channel.settings.include_preview {
                        Some(self.truncate_preview(email_preview, channel.settings.max_preview_length))
                    } else {
                        None
                    },
                    url: Some(format!("/mail/inbox/{}", email_id)),
                    metadata: HashMap::from([
                        ("from".to_string(), email_from.to_string()),
                        ("trust_score".to_string(), trust_score.to_string()),
                    ]),
                    priority: rule.priority.clone(),
                    created_at: Utc::now(),
                };

                // Send notification
                let result = self.send_notification(&channel, payload).await;
                results.push(result);
            }
        }

        results
    }

    /// Check if a rule matches the email
    fn rule_matches(
        &self,
        trigger: &NotificationTrigger,
        filters: &NotificationFilters,
        from: &str,
        subject: &str,
        trust_score: f64,
        labels: &[String],
    ) -> bool {
        // Check trigger type
        let trigger_matches = match trigger {
            NotificationTrigger::EmailReceived => true,
            NotificationTrigger::HighTrustEmail { min_score } => trust_score >= *min_score,
            NotificationTrigger::LabeledEmail { labels: required } => {
                required.iter().any(|l| labels.contains(l))
            }
            NotificationTrigger::VipEmail => {
                labels.contains(&"vip".to_string()) || labels.contains(&"important".to_string())
            }
            NotificationTrigger::SearchMatch { query } => {
                let query_lower = query.to_lowercase();
                subject.to_lowercase().contains(&query_lower) ||
                from.to_lowercase().contains(&query_lower)
            }
            _ => false,
        };

        if !trigger_matches {
            return false;
        }

        // Apply filters
        let from_lower = from.to_lowercase();
        let subject_lower = subject.to_lowercase();

        // Domain filter
        if let Some(domains) = &filters.from_domains {
            let from_domain = from.split('@').last().unwrap_or("");
            if !domains.iter().any(|d| from_domain.eq_ignore_ascii_case(d)) {
                return false;
            }
        }

        // Exclude domains
        if let Some(domains) = &filters.exclude_domains {
            let from_domain = from.split('@').last().unwrap_or("");
            if domains.iter().any(|d| from_domain.eq_ignore_ascii_case(d)) {
                return false;
            }
        }

        // Email filter
        if let Some(emails) = &filters.from_emails {
            if !emails.iter().any(|e| from_lower.eq_ignore_ascii_case(e)) {
                return false;
            }
        }

        // Subject contains
        if let Some(terms) = &filters.subject_contains {
            if !terms.iter().any(|t| subject_lower.contains(&t.to_lowercase())) {
                return false;
            }
        }

        // Minimum trust score
        if let Some(min_score) = filters.min_trust_score {
            if trust_score < min_score {
                return false;
            }
        }

        true
    }

    /// Check if currently in quiet hours
    fn is_quiet_hours(&self, settings: &ChannelSettings) -> bool {
        if let Some(quiet) = &settings.quiet_hours {
            let now = Utc::now();
            let hour = now.hour() as u8;

            // Simple check (doesn't account for timezone properly)
            if quiet.start_hour < quiet.end_hour {
                hour >= quiet.start_hour && hour < quiet.end_hour
            } else {
                // Spans midnight
                hour >= quiet.start_hour || hour < quiet.end_hour
            }
        } else {
            false
        }
    }

    /// Check rate limit
    fn check_rate_limit(&mut self, channel_id: Uuid, settings: &ChannelSettings) -> bool {
        if let Some(limit) = settings.rate_limit {
            let now = Utc::now();
            let counter = self.rate_counters.entry(channel_id).or_insert(RateCounter {
                count: 0,
                window_start: now,
            });

            // Reset window if hour has passed
            if (now - counter.window_start).num_hours() >= 1 {
                counter.count = 0;
                counter.window_start = now;
            }

            if counter.count >= limit {
                return false;
            }

            counter.count += 1;
            true
        } else {
            true
        }
    }

    /// Truncate preview text
    fn truncate_preview(&self, text: &str, max_len: usize) -> String {
        if text.len() <= max_len {
            text.to_string()
        } else {
            format!("{}...", &text[..max_len])
        }
    }

    /// Send notification to platform
    async fn send_notification(
        &mut self,
        channel: &NotificationChannel,
        payload: NotificationPayload,
    ) -> DeliveryResult {
        let body = match channel.platform {
            NotificationPlatform::Slack => self.build_slack_payload(&payload, &channel.settings),
            NotificationPlatform::MicrosoftTeams => self.build_teams_payload(&payload, &channel.settings),
            NotificationPlatform::Discord => self.build_discord_payload(&payload, &channel.settings),
            NotificationPlatform::Mattermost => self.build_mattermost_payload(&payload, &channel.settings),
            NotificationPlatform::CustomWebhook => self.build_generic_payload(&payload),
        };

        // In production: HTTP POST to webhook_url
        // Simulated success for now
        let result = DeliveryResult {
            notification_id: payload.id,
            channel_id: channel.id,
            success: true,
            platform_response: Some("ok".to_string()),
            error: None,
            delivered_at: Utc::now(),
        };

        // Update last notification time
        if let Some(ch) = self.channels.get_mut(&channel.id) {
            ch.last_notification = Some(Utc::now());
        }

        self.history.push(result.clone());

        result
    }

    /// Build Slack Block Kit payload
    fn build_slack_payload(&self, payload: &NotificationPayload, settings: &ChannelSettings) -> serde_json::Value {
        let mut blocks = vec![
            serde_json::json!({
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": payload.title,
                    "emoji": true
                }
            }),
            serde_json::json!({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": format!("*{}*", payload.message)
                }
            }),
        ];

        if let Some(preview) = &payload.preview {
            blocks.push(serde_json::json!({
                "type": "section",
                "text": {
                    "type": "plain_text",
                    "text": preview
                }
            }));
        }

        if let Some(url) = &payload.url {
            blocks.push(serde_json::json!({
                "type": "actions",
                "elements": [{
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Open in Mycelix"
                    },
                    "url": url,
                    "action_id": "open_email"
                }]
            }));
        }

        let mut result = serde_json::json!({
            "blocks": blocks
        });

        if let Some(name) = &settings.bot_name {
            result["username"] = serde_json::json!(name);
        }
        if let Some(icon) = &settings.bot_icon {
            result["icon_url"] = serde_json::json!(icon);
        }

        result
    }

    /// Build Microsoft Teams Adaptive Card payload
    fn build_teams_payload(&self, payload: &NotificationPayload, settings: &ChannelSettings) -> serde_json::Value {
        let mut body = vec![
            serde_json::json!({
                "type": "TextBlock",
                "size": "Medium",
                "weight": "Bolder",
                "text": payload.title
            }),
            serde_json::json!({
                "type": "TextBlock",
                "text": payload.message,
                "wrap": true
            }),
        ];

        if let Some(preview) = &payload.preview {
            body.push(serde_json::json!({
                "type": "TextBlock",
                "text": preview,
                "wrap": true,
                "isSubtle": true
            }));
        }

        let mut actions = Vec::new();
        if let Some(url) = &payload.url {
            actions.push(serde_json::json!({
                "type": "Action.OpenUrl",
                "title": "Open in Mycelix",
                "url": url
            }));
        }

        serde_json::json!({
            "type": "message",
            "attachments": [{
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": body,
                    "actions": actions
                }
            }]
        })
    }

    /// Build Discord embed payload
    fn build_discord_payload(&self, payload: &NotificationPayload, settings: &ChannelSettings) -> serde_json::Value {
        let color = match payload.priority {
            NotificationPriority::Urgent => 15158332, // Red
            NotificationPriority::High => 15105570,   // Orange
            NotificationPriority::Normal => 3447003,  // Blue
            NotificationPriority::Low => 9807270,     // Gray
        };

        let mut embed = serde_json::json!({
            "title": payload.title,
            "description": payload.message,
            "color": color,
            "timestamp": payload.created_at.to_rfc3339()
        });

        if let Some(preview) = &payload.preview {
            embed["fields"] = serde_json::json!([{
                "name": "Preview",
                "value": preview,
                "inline": false
            }]);
        }

        if let Some(url) = &payload.url {
            embed["url"] = serde_json::json!(url);
        }

        let mut result = serde_json::json!({
            "embeds": [embed]
        });

        if let Some(name) = &settings.bot_name {
            result["username"] = serde_json::json!(name);
        }
        if let Some(icon) = &settings.bot_icon {
            result["avatar_url"] = serde_json::json!(icon);
        }

        result
    }

    /// Build Mattermost payload
    fn build_mattermost_payload(&self, payload: &NotificationPayload, settings: &ChannelSettings) -> serde_json::Value {
        let mut result = serde_json::json!({
            "text": format!("**{}**\n{}", payload.title, payload.message),
            "attachments": [{
                "fallback": payload.message,
                "title": payload.title,
                "text": payload.preview.as_deref().unwrap_or(""),
                "color": match payload.priority {
                    NotificationPriority::Urgent => "#FF0000",
                    NotificationPriority::High => "#FFA500",
                    NotificationPriority::Normal => "#0000FF",
                    NotificationPriority::Low => "#808080",
                }
            }]
        });

        if let Some(name) = &settings.bot_name {
            result["username"] = serde_json::json!(name);
        }
        if let Some(icon) = &settings.bot_icon {
            result["icon_url"] = serde_json::json!(icon);
        }

        result
    }

    /// Build generic webhook payload
    fn build_generic_payload(&self, payload: &NotificationPayload) -> serde_json::Value {
        serde_json::json!({
            "id": payload.id,
            "title": payload.title,
            "message": payload.message,
            "preview": payload.preview,
            "url": payload.url,
            "priority": format!("{:?}", payload.priority),
            "metadata": payload.metadata,
            "timestamp": payload.created_at.to_rfc3339()
        })
    }

    /// Send a test notification
    pub async fn send_test_notification(&mut self, channel_id: Uuid) -> Result<DeliveryResult, NotificationError> {
        let channel = self.channels.get(&channel_id)
            .ok_or(NotificationError::ChannelNotFound)?
            .clone();

        let payload = NotificationPayload {
            id: Uuid::new_v4(),
            channel_id,
            trigger: NotificationTrigger::EmailReceived,
            title: "Mycelix Mail Connected".to_string(),
            message: "Your notification channel is now active!".to_string(),
            preview: Some("This is a test notification from Mycelix Mail.".to_string()),
            url: None,
            metadata: HashMap::new(),
            priority: NotificationPriority::Normal,
            created_at: Utc::now(),
        };

        Ok(self.send_notification(&channel, payload).await)
    }

    /// Delete a channel
    pub async fn delete_channel(&mut self, channel_id: Uuid) -> Result<(), NotificationError> {
        self.channels.remove(&channel_id)
            .ok_or(NotificationError::ChannelNotFound)?;
        self.rate_counters.remove(&channel_id);
        Ok(())
    }

    /// Enable/disable a channel
    pub async fn set_channel_enabled(&mut self, channel_id: Uuid, enabled: bool) -> Result<(), NotificationError> {
        let channel = self.channels.get_mut(&channel_id)
            .ok_or(NotificationError::ChannelNotFound)?;
        channel.enabled = enabled;
        Ok(())
    }

    /// Get user's channels
    pub fn get_channels(&self, user_id: Uuid) -> Vec<&NotificationChannel> {
        self.channels.values()
            .filter(|c| c.user_id == user_id)
            .collect()
    }

    /// Get channel by ID
    pub fn get_channel(&self, channel_id: Uuid) -> Option<&NotificationChannel> {
        self.channels.get(&channel_id)
    }

    /// Get notification history
    pub fn get_history(&self, channel_id: Uuid, limit: usize) -> Vec<&DeliveryResult> {
        self.history.iter()
            .filter(|r| r.channel_id == channel_id)
            .rev()
            .take(limit)
            .collect()
    }

    /// Send daily digest
    pub async fn send_daily_digest(
        &mut self,
        user_id: Uuid,
        unread_count: u32,
        high_priority_count: u32,
        top_senders: Vec<(String, u32)>,
    ) -> Vec<DeliveryResult> {
        let mut results = Vec::new();

        let channel_ids: Vec<Uuid> = self.channels.iter()
            .filter(|(_, c)| c.user_id == user_id && c.enabled)
            .filter(|(_, c)| c.rules.iter().any(|r| matches!(r.trigger, NotificationTrigger::DailyDigest { .. })))
            .map(|(id, _)| *id)
            .collect();

        for channel_id in channel_ids {
            let channel = match self.channels.get(&channel_id) {
                Some(c) => c.clone(),
                None => continue,
            };

            let top_senders_text = top_senders.iter()
                .take(5)
                .map(|(sender, count)| format!("• {} ({} emails)", sender, count))
                .collect::<Vec<_>>()
                .join("\n");

            let payload = NotificationPayload {
                id: Uuid::new_v4(),
                channel_id,
                trigger: NotificationTrigger::DailyDigest { hour: 9 },
                title: "Daily Email Digest".to_string(),
                message: format!(
                    "You have {} unread emails ({} high priority)",
                    unread_count, high_priority_count
                ),
                preview: Some(format!("Top senders:\n{}", top_senders_text)),
                url: Some("/mail/inbox".to_string()),
                metadata: HashMap::from([
                    ("unread_count".to_string(), unread_count.to_string()),
                    ("high_priority_count".to_string(), high_priority_count.to_string()),
                ]),
                priority: NotificationPriority::Low,
                created_at: Utc::now(),
            };

            let result = self.send_notification(&channel, payload).await;
            results.push(result);
        }

        results
    }
}

/// Notification errors
#[derive(Debug, Clone)]
pub enum NotificationError {
    ChannelNotFound,
    InvalidWebhookUrl(String),
    DeliveryFailed(String),
    RateLimited,
    PlatformError(String),
}

impl std::fmt::Display for NotificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChannelNotFound => write!(f, "Notification channel not found"),
            Self::InvalidWebhookUrl(msg) => write!(f, "Invalid webhook URL: {}", msg),
            Self::DeliveryFailed(msg) => write!(f, "Delivery failed: {}", msg),
            Self::RateLimited => write!(f, "Rate limited"),
            Self::PlatformError(msg) => write!(f, "Platform error: {}", msg),
        }
    }
}

impl std::error::Error for NotificationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_slack_channel() {
        let mut service = NotificationService::new();
        let user_id = Uuid::new_v4();

        let channel = service.create_channel(
            user_id,
            NotificationPlatform::Slack,
            "My Slack".to_string(),
            "https://hooks.slack.com/services/T00/B00/XXX".to_string(),
            None,
        ).await.unwrap();

        assert_eq!(channel.platform, NotificationPlatform::Slack);
        assert!(channel.enabled);
    }

    #[tokio::test]
    async fn test_add_rule() {
        let mut service = NotificationService::new();
        let user_id = Uuid::new_v4();

        let channel = service.create_channel(
            user_id,
            NotificationPlatform::Discord,
            "My Discord".to_string(),
            "https://discord.com/api/webhooks/123/abc".to_string(),
            None,
        ).await.unwrap();

        let rule = service.add_rule(
            channel.id,
            "VIP Emails".to_string(),
            NotificationTrigger::HighTrustEmail { min_score: 0.8 },
            NotificationFilters::default(),
            NotificationPriority::High,
        ).await.unwrap();

        assert!(rule.enabled);

        let channel = service.get_channel(channel.id).unwrap();
        assert_eq!(channel.rules.len(), 1);
    }

    #[test]
    fn test_rule_matching() {
        let service = NotificationService::new();

        let trigger = NotificationTrigger::HighTrustEmail { min_score: 0.8 };
        let filters = NotificationFilters {
            from_domains: Some(vec!["example.com".to_string()]),
            ..Default::default()
        };

        // Should match
        assert!(service.rule_matches(
            &trigger,
            &filters,
            "alice@example.com",
            "Hello",
            0.9,
            &[]
        ));

        // Low trust score - no match
        assert!(!service.rule_matches(
            &trigger,
            &filters,
            "alice@example.com",
            "Hello",
            0.5,
            &[]
        ));

        // Wrong domain - no match
        assert!(!service.rule_matches(
            &trigger,
            &filters,
            "alice@other.com",
            "Hello",
            0.9,
            &[]
        ));
    }
}
