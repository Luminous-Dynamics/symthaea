// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * AI-Powered Features Module
 *
 * Smart compose, email summarization, priority inbox, sentiment analysis,
 * anomaly detection, and natural language search
 */

// AI feature submodules
pub mod anomaly;
pub mod categorization;
pub mod compose;
pub mod nl_search;

// Re-exports
pub use anomaly::{
    Anomaly as SecurityAnomaly, AnomalyDetector, AnomalyType,
    PhishingCheckInput, SecurityEvent, SecurityMonitorService, Severity,
};
pub use categorization::{
    CategoryPrediction, EmailCategory as IntelligentCategory, EmailFeatures,
    EmailIntelligence, EmailIntelligenceService, PriorityScore as IntelligentPriority,
};
pub use compose::{
    ComposeAssistant, ComposeSuggestion as SmartSuggestion, EmailContext,
    ReplyIntent, ReplyTone, SmartReply as IntelligentReply,
};
pub use nl_search::{
    DateRange, NLQueryParser, NLSearchService, ParsedQuery,
    SearchPrepResult, SearchQueryParams, SizeFilter, SortOrder,
};

use sqlx::PgPool;
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Debug, thiserror::Error)]
pub enum AiError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("AI service unavailable")]
    ServiceUnavailable,
    #[error("Model inference failed: {0}")]
    InferenceFailed(String),
    #[error("Rate limit exceeded")]
    RateLimitExceeded,
    #[error("Content too long")]
    ContentTooLong,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SmartComposeRequest {
    pub context: String,
    pub partial_text: String,
    pub tone: Option<ComposeTone>,
    pub max_suggestions: Option<u8>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum ComposeTone {
    Professional,
    Casual,
    Formal,
    Friendly,
    Concise,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CompletionSuggestion {
    pub text: String,
    pub confidence: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmailSummary {
    pub tldr: String,
    pub key_points: Vec<String>,
    pub action_items: Vec<ActionItem>,
    pub sentiment: Sentiment,
    pub detected_entities: Vec<Entity>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ActionItem {
    pub description: String,
    pub due_date: Option<String>,
    pub assignee: Option<String>,
    pub priority: Priority,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum Priority {
    Low,
    Medium,
    High,
    Urgent,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Sentiment {
    pub overall: SentimentType,
    pub score: f32,
    pub urgency: f32,
    pub formality: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum SentimentType {
    Positive,
    Neutral,
    Negative,
    Mixed,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Entity {
    pub entity_type: EntityType,
    pub value: String,
    pub start_pos: usize,
    pub end_pos: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum EntityType {
    Person,
    Organization,
    Date,
    Time,
    Money,
    Location,
    PhoneNumber,
    Email,
    Url,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PriorityScore {
    pub email_id: Uuid,
    pub score: f32,
    pub factors: Vec<PriorityFactor>,
    pub category: EmailCategory,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PriorityFactor {
    pub name: String,
    pub weight: f32,
    pub value: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, sqlx::Type)]
#[sqlx(type_name = "email_category", rename_all = "snake_case")]
pub enum EmailCategory {
    Primary,
    Social,
    Promotions,
    Updates,
    Forums,
    Newsletters,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SmartReply {
    pub suggestions: Vec<ReplySuggestion>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReplySuggestion {
    pub short_label: String,
    pub full_text: String,
    pub tone: ComposeTone,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ThreadSummary {
    pub participant_count: usize,
    pub message_count: usize,
    pub summary: String,
    pub timeline: Vec<ThreadEvent>,
    pub unresolved_questions: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ThreadEvent {
    pub timestamp: DateTime<Utc>,
    pub sender: String,
    pub action: String,
}

pub struct AiService {
    pool: PgPool,
    model_endpoint: String,
}

impl AiService {
    pub fn new(pool: PgPool, model_endpoint: String) -> Self {
        Self { pool, model_endpoint }
    }

    /// Generate smart compose suggestions
    pub async fn smart_compose(
        &self,
        user_id: Uuid,
        request: SmartComposeRequest,
    ) -> Result<Vec<CompletionSuggestion>, AiError> {
        self.check_rate_limit(user_id, "smart_compose").await?;

        let tone = request.tone.unwrap_or(ComposeTone::Professional);
        let max = request.max_suggestions.unwrap_or(3) as usize;

        let suggestions = self.generate_completions(&request.partial_text, tone, max).await?;
        self.log_ai_usage(user_id, "smart_compose", suggestions.len() as i32).await?;

        Ok(suggestions)
    }

    /// Summarize an email
    pub async fn summarize_email(
        &self,
        user_id: Uuid,
        email_id: Uuid,
    ) -> Result<EmailSummary, AiError> {
        let email = sqlx::query!(
            "SELECT subject, body_text FROM emails WHERE id = $1 AND user_id = $2",
            email_id,
            user_id
        )
        .fetch_optional(&self.pool)
        .await?
        .ok_or(AiError::InferenceFailed("Email not found".to_string()))?;

        let content = email.body_text.unwrap_or_default();

        if content.len() > 50000 {
            return Err(AiError::ContentTooLong);
        }

        // Check cache
        if let Some(cached) = self.get_cached_summary(email_id).await? {
            return Ok(cached);
        }

        let summary = self.generate_summary(&content).await?;
        let action_items = self.extract_action_items(&content).await?;
        let sentiment = self.analyze_sentiment(&content).await?;
        let entities = self.extract_entities(&content).await?;

        let result = EmailSummary {
            tldr: summary,
            key_points: self.extract_key_points(&content).await?,
            action_items,
            sentiment,
            detected_entities: entities,
        };

        self.cache_summary(email_id, &result).await?;
        Ok(result)
    }

    /// Calculate priority score for an email
    pub async fn calculate_priority(
        &self,
        user_id: Uuid,
        email_id: Uuid,
    ) -> Result<PriorityScore, AiError> {
        let email = sqlx::query!(
            "SELECT subject, body_text, from_address FROM emails WHERE id = $1 AND user_id = $2",
            email_id,
            user_id
        )
        .fetch_optional(&self.pool)
        .await?
        .ok_or(AiError::InferenceFailed("Email not found".to_string()))?;

        let mut factors = Vec::new();
        let mut total_score = 0.0;
        let content = email.body_text.as_deref().unwrap_or("");

        // Factor 1: Sender importance
        let sender_score = self.calculate_sender_importance(user_id, &email.from_address).await?;
        factors.push(PriorityFactor { name: "sender_importance".to_string(), weight: 0.3, value: sender_score });
        total_score += sender_score * 0.3;

        // Factor 2: Content urgency
        let urgency_score = self.calculate_urgency(content).await?;
        factors.push(PriorityFactor { name: "content_urgency".to_string(), weight: 0.4, value: urgency_score });
        total_score += urgency_score * 0.4;

        // Factor 3: Time sensitivity
        let time_score = self.calculate_time_sensitivity(content).await?;
        factors.push(PriorityFactor { name: "time_sensitivity".to_string(), weight: 0.3, value: time_score });
        total_score += time_score * 0.3;

        let category = self.categorize_email(content, &email.from_address).await?;

        Ok(PriorityScore { email_id, score: total_score, factors, category })
    }

    /// Generate smart reply suggestions
    pub async fn smart_reply(
        &self,
        user_id: Uuid,
        email_id: Uuid,
    ) -> Result<SmartReply, AiError> {
        let email = sqlx::query!(
            "SELECT body_text FROM emails WHERE id = $1 AND user_id = $2",
            email_id,
            user_id
        )
        .fetch_optional(&self.pool)
        .await?
        .ok_or(AiError::InferenceFailed("Email not found".to_string()))?;

        let content = email.body_text.as_deref().unwrap_or("");
        let is_question = content.contains('?');

        let suggestions = vec![
            ReplySuggestion {
                short_label: if is_question { "Yes" } else { "Thanks" }.to_string(),
                full_text: "Thank you for your email. I'll take care of this right away.".to_string(),
                tone: ComposeTone::Friendly,
            },
            ReplySuggestion {
                short_label: if is_question { "Let me check" } else { "Got it" }.to_string(),
                full_text: "Thank you for reaching out. I'll review this and get back to you shortly.".to_string(),
                tone: ComposeTone::Professional,
            },
            ReplySuggestion {
                short_label: if is_question { "No, sorry" } else { "Will do" }.to_string(),
                full_text: "I appreciate you thinking of me. I'll follow up on this soon.".to_string(),
                tone: ComposeTone::Professional,
            },
        ];

        Ok(SmartReply { suggestions })
    }

    /// Prioritize inbox
    pub async fn prioritize_inbox(&self, user_id: Uuid) -> Result<Vec<PriorityScore>, AiError> {
        let emails = sqlx::query!(
            "SELECT id FROM emails WHERE user_id = $1 AND is_read = false AND folder = 'inbox' LIMIT 50",
            user_id
        )
        .fetch_all(&self.pool)
        .await?;

        let mut scores = Vec::new();
        for email in emails {
            if let Ok(score) = self.calculate_priority(user_id, email.id).await {
                scores.push(score);
            }
        }

        scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scores)
    }

    // Private helpers

    async fn check_rate_limit(&self, user_id: Uuid, operation: &str) -> Result<(), AiError> {
        let count: i64 = sqlx::query_scalar!(
            "SELECT COUNT(*) FROM ai_usage_log WHERE user_id = $1 AND operation = $2 AND created_at > NOW() - INTERVAL '1 minute'",
            user_id, operation
        )
        .fetch_one(&self.pool)
        .await?
        .unwrap_or(0);

        if count > 60 { Err(AiError::RateLimitExceeded) } else { Ok(()) }
    }

    async fn log_ai_usage(&self, user_id: Uuid, operation: &str, count: i32) -> Result<(), AiError> {
        sqlx::query!("INSERT INTO ai_usage_log (user_id, operation, count, created_at) VALUES ($1, $2, $3, NOW())",
            user_id, operation, count)
            .execute(&self.pool).await?;
        Ok(())
    }

    async fn generate_completions(&self, partial: &str, tone: ComposeTone, max: usize) -> Result<Vec<CompletionSuggestion>, AiError> {
        let suggestions = match tone {
            ComposeTone::Professional => vec![
                "I hope this email finds you well.",
                "Thank you for your prompt response.",
                "Please let me know if you have any questions.",
            ],
            ComposeTone::Casual => vec![
                "Hope you're doing great!",
                "Thanks for getting back to me!",
                "Let me know what you think!",
            ],
            ComposeTone::Formal => vec![
                "I am writing to inquire about",
                "Please find attached the requested documents.",
                "I would appreciate your prompt attention to this matter.",
            ],
            _ => vec!["Thank you for your email."],
        };

        Ok(suggestions.into_iter().take(max)
            .map(|s| CompletionSuggestion { text: s.to_string(), confidence: 0.85 })
            .collect())
    }

    async fn generate_summary(&self, content: &str) -> Result<String, AiError> {
        let sentences: Vec<&str> = content.split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| s.len() > 20).take(3).collect();
        Ok(sentences.join(". ") + ".")
    }

    async fn extract_key_points(&self, content: &str) -> Result<Vec<String>, AiError> {
        Ok(content.lines()
            .filter(|l| l.trim().starts_with("- ") || l.trim().starts_with("* "))
            .map(|l| l.trim()[2..].to_string())
            .take(5).collect())
    }

    async fn extract_action_items(&self, content: &str) -> Result<Vec<ActionItem>, AiError> {
        let patterns = ["please", "could you", "need to", "must", "should"];
        Ok(content.lines()
            .filter(|l| patterns.iter().any(|p| l.to_lowercase().contains(p)))
            .map(|l| ActionItem { description: l.trim().to_string(), due_date: None, assignee: None, priority: Priority::Medium })
            .take(10).collect())
    }

    async fn analyze_sentiment(&self, content: &str) -> Result<Sentiment, AiError> {
        let lower = content.to_lowercase();
        let pos = ["thank", "great", "excellent", "happy"].iter().filter(|w| lower.contains(*w)).count();
        let neg = ["sorry", "unfortunately", "problem", "urgent"].iter().filter(|w| lower.contains(*w)).count();

        let (overall, score) = if pos > neg * 2 { (SentimentType::Positive, 0.8) }
            else if neg > pos * 2 { (SentimentType::Negative, -0.6) }
            else if pos > 0 && neg > 0 { (SentimentType::Mixed, 0.0) }
            else { (SentimentType::Neutral, 0.0) };

        let urgency = if lower.contains("urgent") || lower.contains("asap") { 0.9 }
            else if lower.contains("soon") { 0.6 } else { 0.3 };

        Ok(Sentiment { overall, score, urgency, formality: 0.5 })
    }

    async fn extract_entities(&self, content: &str) -> Result<Vec<Entity>, AiError> {
        let mut entities = Vec::new();
        let email_re = regex::Regex::new(r"[\w.-]+@[\w.-]+\.\w+").unwrap();
        for cap in email_re.find_iter(content) {
            entities.push(Entity { entity_type: EntityType::Email, value: cap.as_str().to_string(), start_pos: cap.start(), end_pos: cap.end() });
        }
        Ok(entities)
    }

    async fn get_cached_summary(&self, email_id: Uuid) -> Result<Option<EmailSummary>, AiError> {
        let cached = sqlx::query!("SELECT summary_json FROM email_summaries WHERE email_id = $1 AND created_at > NOW() - INTERVAL '24 hours'", email_id)
            .fetch_optional(&self.pool).await?;
        match cached {
            Some(row) => Ok(serde_json::from_value(row.summary_json).ok()),
            None => Ok(None),
        }
    }

    async fn cache_summary(&self, email_id: Uuid, summary: &EmailSummary) -> Result<(), AiError> {
        let json = serde_json::to_value(summary).map_err(|e| AiError::InferenceFailed(e.to_string()))?;
        sqlx::query!("INSERT INTO email_summaries (email_id, summary_json, created_at) VALUES ($1, $2, NOW()) ON CONFLICT (email_id) DO UPDATE SET summary_json = $2, created_at = NOW()", email_id, json)
            .execute(&self.pool).await?;
        Ok(())
    }

    async fn calculate_sender_importance(&self, user_id: Uuid, sender: &str) -> Result<f32, AiError> {
        let stats = sqlx::query!("SELECT COUNT(*) as total, COUNT(*) FILTER (WHERE is_read) as read_count FROM emails WHERE user_id = $1 AND from_address = $2", user_id, sender)
            .fetch_one(&self.pool).await?;
        let total = stats.total.unwrap_or(0) as f32;
        if total == 0.0 { return Ok(0.5); }
        Ok((stats.read_count.unwrap_or(0) as f32 / total).min(1.0))
    }

    async fn calculate_urgency(&self, content: &str) -> Result<f32, AiError> {
        let lower = content.to_lowercase();
        let keywords = [("urgent", 0.9), ("asap", 0.85), ("immediately", 0.8), ("critical", 0.85), ("deadline", 0.7)];
        Ok(keywords.iter().filter(|(kw, _)| lower.contains(kw)).map(|(_, s)| *s).fold(0.0f32, |a, b| a.max(b)))
    }

    async fn calculate_time_sensitivity(&self, content: &str) -> Result<f32, AiError> {
        let lower = content.to_lowercase();
        let has_time = ["meeting", "call", "deadline", "tomorrow"].iter().any(|w| lower.contains(w));
        Ok(if has_time { 0.7 } else { 0.2 })
    }

    async fn categorize_email(&self, content: &str, sender: &str) -> Result<EmailCategory, AiError> {
        let lower = content.to_lowercase();
        let sender_lower = sender.to_lowercase();

        if ["facebook", "twitter", "linkedin"].iter().any(|s| sender_lower.contains(s)) { return Ok(EmailCategory::Social); }
        if lower.contains("unsubscribe") || lower.contains("% off") { return Ok(EmailCategory::Promotions); }
        if lower.contains("order") || lower.contains("shipped") { return Ok(EmailCategory::Updates); }
        if lower.contains("newsletter") { return Ok(EmailCategory::Newsletters); }
        Ok(EmailCategory::Primary)
    }
}
