// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Track S: AI Copilot & Summarization
//!
//! Intelligent email assistance powered by local and federated AI models.
//! Supports thread summarization, action extraction, smart replies, and more.

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;

// ============================================================================
// Core Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadSummary {
    pub thread_id: Uuid,
    pub summary: String,
    pub key_points: Vec<String>,
    pub participants: Vec<ParticipantSummary>,
    pub timeline: Vec<TimelineEvent>,
    pub word_count_original: u32,
    pub word_count_summary: u32,
    pub compression_ratio: f32,
    pub generated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipantSummary {
    pub email: String,
    pub name: Option<String>,
    pub message_count: u32,
    pub sentiment: SentimentScore,
    pub key_contributions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEvent {
    pub timestamp: DateTime<Utc>,
    pub event_type: TimelineEventType,
    pub description: String,
    pub message_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimelineEventType {
    ThreadStarted,
    DecisionMade,
    QuestionAsked,
    QuestionAnswered,
    ActionItemCreated,
    DeadlineMentioned,
    ParticipantJoined,
    TopicChanged,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionItem {
    pub id: Uuid,
    pub email_id: Uuid,
    pub thread_id: Option<Uuid>,
    pub description: String,
    pub assignee: Option<String>,
    pub deadline: Option<DateTime<Utc>>,
    pub priority: ActionPriority,
    pub status: ActionStatus,
    pub confidence: f32,
    pub source_text: String,
    pub extracted_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionPriority {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionStatus {
    Pending,
    InProgress,
    Completed,
    Dismissed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartReply {
    pub id: Uuid,
    pub email_id: Uuid,
    pub reply_type: ReplyType,
    pub content: String,
    pub tone: ToneType,
    pub confidence: f32,
    pub personalization_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplyType {
    Acknowledgment,
    Acceptance,
    Decline,
    RequestInfo,
    ProvideInfo,
    ScheduleMeeting,
    Delegate,
    FollowUp,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToneType {
    Formal,
    Professional,
    Friendly,
    Casual,
    Urgent,
    Apologetic,
    Grateful,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingExtraction {
    pub id: Uuid,
    pub email_id: Uuid,
    pub title: Option<String>,
    pub proposed_times: Vec<ProposedTime>,
    pub location: Option<MeetingLocation>,
    pub attendees: Vec<String>,
    pub agenda: Option<String>,
    pub duration_minutes: Option<u32>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposedTime {
    pub start: DateTime<Utc>,
    pub end: Option<DateTime<Utc>>,
    pub timezone: Option<String>,
    pub is_flexible: bool,
    pub source_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeetingLocation {
    Physical { address: String, room: Option<String> },
    Virtual { platform: String, link: Option<String> },
    Hybrid { address: String, link: String },
    ToBeDetermined,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentScore {
    pub overall: f32,  // -1.0 to 1.0
    pub urgency: f32,  // 0.0 to 1.0
    pub formality: f32,  // 0.0 to 1.0
    pub positivity: f32,  // 0.0 to 1.0
    pub emotions: Vec<DetectedEmotion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedEmotion {
    pub emotion: EmotionType,
    pub intensity: f32,
    pub source_phrases: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmotionType {
    Joy,
    Gratitude,
    Excitement,
    Concern,
    Frustration,
    Disappointment,
    Confusion,
    Neutral,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToneAnalysis {
    pub email_id: Uuid,
    pub detected_tone: ToneType,
    pub suggested_tone: Option<ToneType>,
    pub issues: Vec<ToneIssue>,
    pub improvements: Vec<ToneImprovement>,
    pub readability_score: f32,
    pub clarity_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToneIssue {
    pub issue_type: ToneIssueType,
    pub description: String,
    pub location: TextSpan,
    pub severity: IssueSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToneIssueType {
    TooFormal,
    TooCasual,
    Ambiguous,
    PotentiallyOffensive,
    Passive,
    TooLong,
    Jargon,
    GrammarIssue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueSeverity {
    Critical,
    Warning,
    Suggestion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToneImprovement {
    pub location: TextSpan,
    pub original: String,
    pub suggestion: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSpan {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Translation {
    pub id: Uuid,
    pub email_id: Uuid,
    pub source_language: String,
    pub target_language: String,
    pub original_text: String,
    pub translated_text: String,
    pub cultural_notes: Vec<CulturalNote>,
    pub confidence: f32,
    pub translated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CulturalNote {
    pub note_type: CulturalNoteType,
    pub description: String,
    pub suggestion: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CulturalNoteType {
    FormalityDifference,
    Idiom,
    BusinessEtiquette,
    TimeZoneConsideration,
    HolidayAwareness,
    NameConvention,
}

// ============================================================================
// AI Model Configuration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiModelConfig {
    pub provider: AiProvider,
    pub model_id: String,
    pub endpoint: Option<String>,
    pub api_key_ref: Option<String>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub features: Vec<AiFeature>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AiProvider {
    Local { engine: LocalEngine },
    Ollama { host: String },
    OpenAiCompatible { base_url: String },
    Anthropic,
    Custom { name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LocalEngine {
    LlamaCpp,
    Candle,
    Onnx,
    MlcLlm,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AiFeature {
    Summarization,
    ActionExtraction,
    SmartReply,
    MeetingExtraction,
    SentimentAnalysis,
    ToneAnalysis,
    Translation,
}

// ============================================================================
// Services
// ============================================================================

pub struct SummarizationService {
    pool: PgPool,
    model_config: AiModelConfig,
}

impl SummarizationService {
    pub fn new(pool: PgPool, model_config: AiModelConfig) -> Self {
        Self { pool, model_config }
    }

    pub async fn summarize_thread(&self, thread_id: Uuid) -> Result<ThreadSummary> {
        // Fetch all emails in thread
        let emails = sqlx::query_as!(
            EmailContent,
            r#"
            SELECT id, subject, body, sender, recipients, sent_at
            FROM emails
            WHERE thread_id = $1
            ORDER BY sent_at ASC
            "#,
            thread_id
        )
        .fetch_all(&self.pool)
        .await?;

        // Calculate original word count
        let word_count_original: u32 = emails.iter()
            .map(|e| e.body.split_whitespace().count() as u32)
            .sum();

        // Build context for summarization
        let context = self.build_thread_context(&emails);

        // Generate summary using AI model
        let ai_response = self.call_ai_model(&context, "summarize_thread").await?;

        // Parse AI response into structured summary
        let summary = self.parse_summary_response(&ai_response, &emails)?;

        let word_count_summary = summary.summary.split_whitespace().count() as u32;

        // Store summary
        sqlx::query!(
            r#"
            INSERT INTO thread_summaries (thread_id, summary, key_points, generated_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (thread_id) DO UPDATE SET
                summary = EXCLUDED.summary,
                key_points = EXCLUDED.key_points,
                generated_at = NOW()
            "#,
            thread_id,
            summary.summary,
            &summary.key_points
        )
        .execute(&self.pool)
        .await?;

        Ok(ThreadSummary {
            thread_id,
            summary: summary.summary,
            key_points: summary.key_points,
            participants: self.extract_participants(&emails).await?,
            timeline: summary.timeline,
            word_count_original,
            word_count_summary,
            compression_ratio: word_count_summary as f32 / word_count_original as f32,
            generated_at: Utc::now(),
        })
    }

    pub async fn summarize_email(&self, email_id: Uuid) -> Result<String> {
        let email = sqlx::query_as!(
            EmailContent,
            "SELECT id, subject, body, sender, recipients, sent_at FROM emails WHERE id = $1",
            email_id
        )
        .fetch_one(&self.pool)
        .await?;

        let prompt = format!(
            "Summarize this email in 2-3 sentences:\n\nSubject: {}\n\n{}",
            email.subject.unwrap_or_default(),
            email.body
        );

        self.call_ai_model(&prompt, "summarize_single").await
    }

    fn build_thread_context(&self, emails: &[EmailContent]) -> String {
        emails.iter()
            .map(|e| format!(
                "From: {}\nDate: {}\nSubject: {}\n\n{}\n\n---\n",
                e.sender,
                e.sent_at.format("%Y-%m-%d %H:%M"),
                e.subject.as_deref().unwrap_or("(no subject)"),
                e.body
            ))
            .collect()
    }

    async fn call_ai_model(&self, prompt: &str, task: &str) -> Result<String> {
        match &self.model_config.provider {
            AiProvider::Ollama { host } => {
                self.call_ollama(host, prompt, task).await
            }
            AiProvider::Local { engine } => {
                self.call_local_model(engine, prompt, task).await
            }
            _ => {
                // Fallback to extractive summarization
                Ok(self.extractive_summarize(prompt))
            }
        }
    }

    async fn call_ollama(&self, host: &str, prompt: &str, _task: &str) -> Result<String> {
        let client = reqwest::Client::new();
        let response = client
            .post(format!("{}/api/generate", host))
            .json(&serde_json::json!({
                "model": self.model_config.model_id,
                "prompt": prompt,
                "stream": false,
                "options": {
                    "temperature": self.model_config.temperature,
                    "num_predict": self.model_config.max_tokens
                }
            }))
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        Ok(response["response"].as_str().unwrap_or("").to_string())
    }

    async fn call_local_model(&self, _engine: &LocalEngine, prompt: &str, _task: &str) -> Result<String> {
        // Local model inference - would integrate with llama.cpp, candle, etc.
        Ok(self.extractive_summarize(prompt))
    }

    fn extractive_summarize(&self, text: &str) -> String {
        // Simple extractive summarization as fallback
        let sentences: Vec<&str> = text.split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .collect();

        // Take first and last meaningful sentences
        let mut summary = Vec::new();
        if let Some(first) = sentences.first() {
            summary.push(first.trim());
        }
        if sentences.len() > 2 {
            if let Some(last) = sentences.last() {
                summary.push(last.trim());
            }
        }

        summary.join(". ") + "."
    }

    fn parse_summary_response(&self, response: &str, _emails: &[EmailContent]) -> Result<ParsedSummary> {
        Ok(ParsedSummary {
            summary: response.to_string(),
            key_points: self.extract_key_points(response),
            timeline: Vec::new(),
        })
    }

    fn extract_key_points(&self, text: &str) -> Vec<String> {
        text.lines()
            .filter(|line| line.starts_with("- ") || line.starts_with("• "))
            .map(|line| line.trim_start_matches(|c| c == '-' || c == '•' || c == ' ').to_string())
            .collect()
    }

    async fn extract_participants(&self, emails: &[EmailContent]) -> Result<Vec<ParticipantSummary>> {
        let mut participants: std::collections::HashMap<String, ParticipantSummary> =
            std::collections::HashMap::new();

        for email in emails {
            let entry = participants.entry(email.sender.clone()).or_insert(ParticipantSummary {
                email: email.sender.clone(),
                name: None,
                message_count: 0,
                sentiment: SentimentScore {
                    overall: 0.0,
                    urgency: 0.0,
                    formality: 0.5,
                    positivity: 0.5,
                    emotions: Vec::new(),
                },
                key_contributions: Vec::new(),
            });
            entry.message_count += 1;
        }

        Ok(participants.into_values().collect())
    }
}

struct ParsedSummary {
    summary: String,
    key_points: Vec<String>,
    timeline: Vec<TimelineEvent>,
}

#[derive(Debug)]
struct EmailContent {
    id: Uuid,
    subject: Option<String>,
    body: String,
    sender: String,
    recipients: Vec<String>,
    sent_at: DateTime<Utc>,
}

pub struct ActionExtractionService {
    pool: PgPool,
    model_config: AiModelConfig,
}

impl ActionExtractionService {
    pub fn new(pool: PgPool, model_config: AiModelConfig) -> Self {
        Self { pool, model_config }
    }

    pub async fn extract_actions(&self, email_id: Uuid) -> Result<Vec<ActionItem>> {
        let email = sqlx::query!(
            "SELECT body, thread_id FROM emails WHERE id = $1",
            email_id
        )
        .fetch_one(&self.pool)
        .await?;

        let actions = self.detect_actions(&email.body).await?;

        // Store extracted actions
        for action in &actions {
            sqlx::query!(
                r#"
                INSERT INTO action_items (id, email_id, thread_id, description, assignee, deadline, priority, status, confidence, source_text, extracted_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW())
                "#,
                action.id,
                email_id,
                email.thread_id,
                action.description,
                action.assignee,
                action.deadline,
                serde_json::to_string(&action.priority)?,
                serde_json::to_string(&action.status)?,
                action.confidence,
                action.source_text
            )
            .execute(&self.pool)
            .await?;
        }

        Ok(actions)
    }

    async fn detect_actions(&self, text: &str) -> Result<Vec<ActionItem>> {
        let mut actions = Vec::new();

        // Pattern-based action detection
        let action_patterns = [
            (r"(?i)please\s+(\w+)", ActionPriority::Medium),
            (r"(?i)could you\s+(\w+)", ActionPriority::Medium),
            (r"(?i)need(?:s)?\s+to\s+(\w+)", ActionPriority::High),
            (r"(?i)must\s+(\w+)", ActionPriority::Critical),
            (r"(?i)deadline[:\s]+([^.]+)", ActionPriority::High),
            (r"(?i)by\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", ActionPriority::High),
            (r"(?i)asap", ActionPriority::Critical),
            (r"(?i)urgent", ActionPriority::Critical),
        ];

        for (pattern, priority) in action_patterns {
            let re = regex::Regex::new(pattern)?;
            for cap in re.captures_iter(text) {
                if let Some(matched) = cap.get(0) {
                    actions.push(ActionItem {
                        id: Uuid::new_v4(),
                        email_id: Uuid::nil(),  // Will be set by caller
                        thread_id: None,
                        description: self.expand_action_context(text, matched.start(), matched.end()),
                        assignee: self.detect_assignee(text, matched.start()),
                        deadline: self.detect_deadline(text),
                        priority: priority.clone(),
                        status: ActionStatus::Pending,
                        confidence: 0.75,
                        source_text: matched.as_str().to_string(),
                        extracted_at: Utc::now(),
                    });
                }
            }
        }

        Ok(actions)
    }

    fn expand_action_context(&self, text: &str, start: usize, end: usize) -> String {
        // Get surrounding sentence for context
        let before = &text[..start];
        let after = &text[end..];

        let sentence_start = before.rfind(|c| c == '.' || c == '!' || c == '?')
            .map(|i| i + 1)
            .unwrap_or(0);

        let sentence_end = after.find(|c| c == '.' || c == '!' || c == '?')
            .map(|i| end + i + 1)
            .unwrap_or(text.len());

        text[sentence_start..sentence_end].trim().to_string()
    }

    fn detect_assignee(&self, text: &str, position: usize) -> Option<String> {
        // Look for @ mentions or names near the action
        let context = &text[position.saturating_sub(50)..];
        let re = regex::Regex::new(r"@(\w+)|(\b[A-Z][a-z]+\s+[A-Z][a-z]+\b)").ok()?;

        re.captures(context)
            .and_then(|cap| cap.get(1).or(cap.get(2)))
            .map(|m| m.as_str().to_string())
    }

    fn detect_deadline(&self, text: &str) -> Option<DateTime<Utc>> {
        // Simple deadline detection - would use more sophisticated NLP in production
        let patterns = [
            r"(?i)by\s+(\d{1,2}/\d{1,2}/\d{2,4})",
            r"(?i)by\s+(\d{4}-\d{2}-\d{2})",
            r"(?i)due\s+(\d{1,2}/\d{1,2}/\d{2,4})",
        ];

        for pattern in patterns {
            if let Ok(re) = regex::Regex::new(pattern) {
                if let Some(cap) = re.captures(text) {
                    if let Some(date_str) = cap.get(1) {
                        // Parse date - simplified
                        return None;  // Would parse properly
                    }
                }
            }
        }
        None
    }

    pub async fn get_pending_actions(&self, user_id: Uuid) -> Result<Vec<ActionItem>> {
        let actions = sqlx::query_as!(
            ActionItemRow,
            r#"
            SELECT ai.id, ai.email_id, ai.thread_id, ai.description, ai.assignee,
                   ai.deadline, ai.priority, ai.status, ai.confidence, ai.source_text, ai.extracted_at
            FROM action_items ai
            JOIN emails e ON ai.email_id = e.id
            WHERE e.user_id = $1 AND ai.status = 'Pending'
            ORDER BY ai.deadline ASC NULLS LAST, ai.extracted_at DESC
            "#,
            user_id
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(actions.into_iter().map(|r| r.into()).collect())
    }
}

#[derive(Debug)]
struct ActionItemRow {
    id: Uuid,
    email_id: Uuid,
    thread_id: Option<Uuid>,
    description: String,
    assignee: Option<String>,
    deadline: Option<DateTime<Utc>>,
    priority: String,
    status: String,
    confidence: f32,
    source_text: String,
    extracted_at: DateTime<Utc>,
}

impl From<ActionItemRow> for ActionItem {
    fn from(row: ActionItemRow) -> Self {
        ActionItem {
            id: row.id,
            email_id: row.email_id,
            thread_id: row.thread_id,
            description: row.description,
            assignee: row.assignee,
            deadline: row.deadline,
            priority: serde_json::from_str(&row.priority).unwrap_or(ActionPriority::Medium),
            status: serde_json::from_str(&row.status).unwrap_or(ActionStatus::Pending),
            confidence: row.confidence,
            source_text: row.source_text,
            extracted_at: row.extracted_at,
        }
    }
}

pub struct SmartReplyService {
    pool: PgPool,
    model_config: AiModelConfig,
}

impl SmartReplyService {
    pub fn new(pool: PgPool, model_config: AiModelConfig) -> Self {
        Self { pool, model_config }
    }

    pub async fn generate_replies(&self, email_id: Uuid, count: u8) -> Result<Vec<SmartReply>> {
        let email = sqlx::query!(
            "SELECT subject, body, sender FROM emails WHERE id = $1",
            email_id
        )
        .fetch_one(&self.pool)
        .await?;

        // Analyze email intent
        let intent = self.detect_intent(&email.body)?;

        // Generate contextual replies
        let mut replies = Vec::new();
        let reply_types = self.get_appropriate_reply_types(&intent);

        for (i, reply_type) in reply_types.into_iter().take(count as usize).enumerate() {
            let content = self.generate_reply_content(&email.body, &reply_type).await?;
            replies.push(SmartReply {
                id: Uuid::new_v4(),
                email_id,
                reply_type,
                content,
                tone: ToneType::Professional,
                confidence: 0.8 - (i as f32 * 0.1),
                personalization_score: 0.7,
            });
        }

        Ok(replies)
    }

    fn detect_intent(&self, body: &str) -> Result<EmailIntent> {
        let body_lower = body.to_lowercase();

        if body_lower.contains("meeting") || body_lower.contains("schedule") || body_lower.contains("call") {
            Ok(EmailIntent::MeetingRequest)
        } else if body_lower.contains("?") {
            Ok(EmailIntent::Question)
        } else if body_lower.contains("please") || body_lower.contains("could you") {
            Ok(EmailIntent::Request)
        } else if body_lower.contains("thank") || body_lower.contains("appreciate") {
            Ok(EmailIntent::Gratitude)
        } else if body_lower.contains("update") || body_lower.contains("status") {
            Ok(EmailIntent::StatusUpdate)
        } else {
            Ok(EmailIntent::Informational)
        }
    }

    fn get_appropriate_reply_types(&self, intent: &EmailIntent) -> Vec<ReplyType> {
        match intent {
            EmailIntent::MeetingRequest => vec![
                ReplyType::Acceptance,
                ReplyType::Decline,
                ReplyType::RequestInfo,
            ],
            EmailIntent::Question => vec![
                ReplyType::ProvideInfo,
                ReplyType::RequestInfo,
                ReplyType::Delegate,
            ],
            EmailIntent::Request => vec![
                ReplyType::Acceptance,
                ReplyType::Decline,
                ReplyType::RequestInfo,
            ],
            EmailIntent::Gratitude => vec![
                ReplyType::Acknowledgment,
            ],
            EmailIntent::StatusUpdate => vec![
                ReplyType::Acknowledgment,
                ReplyType::RequestInfo,
                ReplyType::FollowUp,
            ],
            EmailIntent::Informational => vec![
                ReplyType::Acknowledgment,
                ReplyType::RequestInfo,
            ],
        }
    }

    async fn generate_reply_content(&self, _original: &str, reply_type: &ReplyType) -> Result<String> {
        // Generate contextual reply - would use AI model in production
        let content = match reply_type {
            ReplyType::Acknowledgment => "Thank you for your email. I've received this and will review it shortly.",
            ReplyType::Acceptance => "Thank you for reaching out. I'd be happy to help with this. Let me know the next steps.",
            ReplyType::Decline => "Thank you for thinking of me, but unfortunately I won't be able to help with this at the moment.",
            ReplyType::RequestInfo => "Thank you for your email. Could you please provide more details about this?",
            ReplyType::ProvideInfo => "Here's the information you requested:",
            ReplyType::ScheduleMeeting => "I'd be happy to schedule a meeting. Here are some times that work for me:",
            ReplyType::Delegate => "I'm forwarding this to the appropriate person who can better assist you.",
            ReplyType::FollowUp => "I wanted to follow up on this. Please let me know if you need anything else.",
            ReplyType::Custom => "",
        };

        Ok(content.to_string())
    }
}

#[derive(Debug)]
enum EmailIntent {
    MeetingRequest,
    Question,
    Request,
    Gratitude,
    StatusUpdate,
    Informational,
}

pub struct MeetingExtractionService {
    pool: PgPool,
}

impl MeetingExtractionService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn extract_meeting(&self, email_id: Uuid) -> Result<Option<MeetingExtraction>> {
        let email = sqlx::query!(
            "SELECT subject, body FROM emails WHERE id = $1",
            email_id
        )
        .fetch_one(&self.pool)
        .await?;

        // Check if this looks like a meeting request
        let body_lower = email.body.to_lowercase();
        if !body_lower.contains("meeting") &&
           !body_lower.contains("schedule") &&
           !body_lower.contains("call") &&
           !body_lower.contains("zoom") &&
           !body_lower.contains("teams") {
            return Ok(None);
        }

        let proposed_times = self.extract_times(&email.body)?;
        let location = self.extract_location(&email.body)?;
        let attendees = self.extract_attendees(&email.body)?;

        Ok(Some(MeetingExtraction {
            id: Uuid::new_v4(),
            email_id,
            title: email.subject,
            proposed_times,
            location: Some(location),
            attendees,
            agenda: self.extract_agenda(&email.body),
            duration_minutes: self.extract_duration(&email.body),
            confidence: 0.75,
        }))
    }

    fn extract_times(&self, text: &str) -> Result<Vec<ProposedTime>> {
        let mut times = Vec::new();

        // Pattern for common time formats
        let time_patterns = [
            r"(\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?)",
            r"(\d{1,2}\s*(?:am|pm|AM|PM))",
            r"(tomorrow|today|monday|tuesday|wednesday|thursday|friday)\s+at\s+(\d{1,2})",
        ];

        for pattern in time_patterns {
            if let Ok(re) = regex::Regex::new(pattern) {
                for cap in re.captures_iter(text) {
                    if let Some(matched) = cap.get(0) {
                        times.push(ProposedTime {
                            start: Utc::now(),  // Would parse properly
                            end: None,
                            timezone: None,
                            is_flexible: text.to_lowercase().contains("flexible"),
                            source_text: matched.as_str().to_string(),
                        });
                    }
                }
            }
        }

        Ok(times)
    }

    fn extract_location(&self, text: &str) -> Result<MeetingLocation> {
        let text_lower = text.to_lowercase();

        if text_lower.contains("zoom") {
            Ok(MeetingLocation::Virtual {
                platform: "Zoom".to_string(),
                link: self.extract_url(text, "zoom")
            })
        } else if text_lower.contains("teams") || text_lower.contains("microsoft teams") {
            Ok(MeetingLocation::Virtual {
                platform: "Microsoft Teams".to_string(),
                link: self.extract_url(text, "teams")
            })
        } else if text_lower.contains("meet.google") || text_lower.contains("google meet") {
            Ok(MeetingLocation::Virtual {
                platform: "Google Meet".to_string(),
                link: self.extract_url(text, "meet.google")
            })
        } else if text_lower.contains("jitsi") {
            Ok(MeetingLocation::Virtual {
                platform: "Jitsi".to_string(),
                link: self.extract_url(text, "jitsi")
            })
        } else {
            Ok(MeetingLocation::ToBeDetermined)
        }
    }

    fn extract_url(&self, text: &str, platform: &str) -> Option<String> {
        let re = regex::Regex::new(r"https?://[^\s]+").ok()?;
        re.find_iter(text)
            .find(|m| m.as_str().to_lowercase().contains(platform))
            .map(|m| m.as_str().to_string())
    }

    fn extract_attendees(&self, text: &str) -> Result<Vec<String>> {
        let mut attendees = Vec::new();

        // Extract email addresses
        let email_re = regex::Regex::new(r"[\w.+-]+@[\w.-]+\.\w+")?;
        for cap in email_re.find_iter(text) {
            attendees.push(cap.as_str().to_string());
        }

        Ok(attendees)
    }

    fn extract_agenda(&self, text: &str) -> Option<String> {
        let text_lower = text.to_lowercase();
        if let Some(idx) = text_lower.find("agenda") {
            let start = idx + 6;
            let end = text[start..].find('\n')
                .map(|i| start + i)
                .unwrap_or(text.len().min(start + 200));
            Some(text[start..end].trim().to_string())
        } else {
            None
        }
    }

    fn extract_duration(&self, text: &str) -> Option<u32> {
        let re = regex::Regex::new(r"(\d+)\s*(?:min(?:ute)?s?|hour?s?)").ok()?;
        re.captures(text).and_then(|cap| {
            cap.get(1).and_then(|m| m.as_str().parse().ok())
        })
    }

    pub async fn create_calendar_event(&self, extraction: &MeetingExtraction) -> Result<CalendarEvent> {
        // Would integrate with CalDAV or Nextcloud Calendar
        Ok(CalendarEvent {
            id: Uuid::new_v4(),
            title: extraction.title.clone().unwrap_or_else(|| "Meeting".to_string()),
            start: extraction.proposed_times.first()
                .map(|t| t.start)
                .unwrap_or_else(Utc::now),
            end: None,
            location: extraction.location.clone(),
            attendees: extraction.attendees.clone(),
            source_email_id: extraction.email_id,
        })
    }
}

#[derive(Debug, Serialize)]
pub struct CalendarEvent {
    pub id: Uuid,
    pub title: String,
    pub start: DateTime<Utc>,
    pub end: Option<DateTime<Utc>>,
    pub location: Option<MeetingLocation>,
    pub attendees: Vec<String>,
    pub source_email_id: Uuid,
}

pub struct ToneAnalysisService {
    pool: PgPool,
}

impl ToneAnalysisService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn analyze_draft(&self, content: &str) -> Result<ToneAnalysis> {
        let detected_tone = self.detect_tone(content);
        let issues = self.find_issues(content);
        let improvements = self.suggest_improvements(content, &issues);

        Ok(ToneAnalysis {
            email_id: Uuid::nil(),
            detected_tone,
            suggested_tone: None,
            issues,
            improvements,
            readability_score: self.calculate_readability(content),
            clarity_score: self.calculate_clarity(content),
        })
    }

    fn detect_tone(&self, content: &str) -> ToneType {
        let content_lower = content.to_lowercase();

        // Count formal vs informal indicators
        let formal_words = ["sincerely", "regards", "respectfully", "pursuant", "hereby", "enclosed"];
        let casual_words = ["hey", "hi", "thanks!", "cheers", "cool", "awesome"];

        let formal_count: usize = formal_words.iter()
            .filter(|w| content_lower.contains(*w))
            .count();
        let casual_count: usize = casual_words.iter()
            .filter(|w| content_lower.contains(*w))
            .count();

        if content_lower.contains("urgent") || content_lower.contains("asap") {
            ToneType::Urgent
        } else if content_lower.contains("sorry") || content_lower.contains("apologize") {
            ToneType::Apologetic
        } else if content_lower.contains("thank") || content_lower.contains("grateful") {
            ToneType::Grateful
        } else if formal_count > casual_count {
            ToneType::Formal
        } else if casual_count > formal_count {
            ToneType::Casual
        } else {
            ToneType::Professional
        }
    }

    fn find_issues(&self, content: &str) -> Vec<ToneIssue> {
        let mut issues = Vec::new();

        // Check for passive voice
        let passive_indicators = ["was done", "has been", "will be done", "is being"];
        for indicator in passive_indicators {
            if let Some(pos) = content.to_lowercase().find(indicator) {
                issues.push(ToneIssue {
                    issue_type: ToneIssueType::Passive,
                    description: "Consider using active voice for clarity".to_string(),
                    location: TextSpan { start: pos, end: pos + indicator.len() },
                    severity: IssueSeverity::Suggestion,
                });
            }
        }

        // Check for jargon
        let jargon = ["synergy", "leverage", "bandwidth", "circle back", "deep dive"];
        for word in jargon {
            if let Some(pos) = content.to_lowercase().find(word) {
                issues.push(ToneIssue {
                    issue_type: ToneIssueType::Jargon,
                    description: format!("'{}' may be unclear to some readers", word),
                    location: TextSpan { start: pos, end: pos + word.len() },
                    severity: IssueSeverity::Suggestion,
                });
            }
        }

        // Check for overly long sentences
        for (i, sentence) in content.split(|c| c == '.' || c == '!' || c == '?').enumerate() {
            let word_count = sentence.split_whitespace().count();
            if word_count > 30 {
                issues.push(ToneIssue {
                    issue_type: ToneIssueType::TooLong,
                    description: format!("Sentence {} has {} words - consider breaking it up", i + 1, word_count),
                    location: TextSpan { start: 0, end: 0 },
                    severity: IssueSeverity::Warning,
                });
            }
        }

        issues
    }

    fn suggest_improvements(&self, content: &str, issues: &[ToneIssue]) -> Vec<ToneImprovement> {
        let mut improvements = Vec::new();

        for issue in issues {
            if matches!(issue.issue_type, ToneIssueType::Jargon) {
                let original = &content[issue.location.start..issue.location.end];
                let suggestion = match original.to_lowercase().as_str() {
                    "synergy" => "collaboration",
                    "leverage" => "use",
                    "bandwidth" => "capacity",
                    "circle back" => "follow up",
                    "deep dive" => "detailed analysis",
                    _ => continue,
                };
                improvements.push(ToneImprovement {
                    location: issue.location.clone(),
                    original: original.to_string(),
                    suggestion: suggestion.to_string(),
                    reason: "More accessible language".to_string(),
                });
            }
        }

        improvements
    }

    fn calculate_readability(&self, content: &str) -> f32 {
        // Simplified Flesch-Kincaid
        let words: Vec<&str> = content.split_whitespace().collect();
        let sentences = content.matches(|c| c == '.' || c == '!' || c == '?').count().max(1);
        let syllables: usize = words.iter().map(|w| self.count_syllables(w)).sum();

        let words_per_sentence = words.len() as f32 / sentences as f32;
        let syllables_per_word = syllables as f32 / words.len().max(1) as f32;

        // Flesch Reading Ease (normalized to 0-1)
        let score = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word);
        (score / 100.0).clamp(0.0, 1.0)
    }

    fn count_syllables(&self, word: &str) -> usize {
        let vowels = ['a', 'e', 'i', 'o', 'u'];
        let word_lower = word.to_lowercase();
        let mut count = 0;
        let mut prev_was_vowel = false;

        for c in word_lower.chars() {
            let is_vowel = vowels.contains(&c);
            if is_vowel && !prev_was_vowel {
                count += 1;
            }
            prev_was_vowel = is_vowel;
        }

        count.max(1)
    }

    fn calculate_clarity(&self, content: &str) -> f32 {
        let words: Vec<&str> = content.split_whitespace().collect();
        let avg_word_length: f32 = words.iter()
            .map(|w| w.len() as f32)
            .sum::<f32>() / words.len().max(1) as f32;

        // Shorter average word length = higher clarity (normalized)
        (1.0 - (avg_word_length - 4.0) / 10.0).clamp(0.0, 1.0)
    }
}

pub struct TranslationService {
    pool: PgPool,
    model_config: AiModelConfig,
}

impl TranslationService {
    pub fn new(pool: PgPool, model_config: AiModelConfig) -> Self {
        Self { pool, model_config }
    }

    pub async fn translate(&self, email_id: Uuid, target_language: &str) -> Result<Translation> {
        let email = sqlx::query!(
            "SELECT body FROM emails WHERE id = $1",
            email_id
        )
        .fetch_one(&self.pool)
        .await?;

        let source_language = self.detect_language(&email.body)?;
        let translated_text = self.perform_translation(&email.body, &source_language, target_language).await?;
        let cultural_notes = self.generate_cultural_notes(&source_language, target_language);

        let translation = Translation {
            id: Uuid::new_v4(),
            email_id,
            source_language: source_language.clone(),
            target_language: target_language.to_string(),
            original_text: email.body.clone(),
            translated_text,
            cultural_notes,
            confidence: 0.85,
            translated_at: Utc::now(),
        };

        // Cache translation
        sqlx::query!(
            r#"
            INSERT INTO email_translations (id, email_id, source_language, target_language, translated_text, translated_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
            "#,
            translation.id,
            email_id,
            source_language,
            target_language,
            translation.translated_text
        )
        .execute(&self.pool)
        .await?;

        Ok(translation)
    }

    fn detect_language(&self, text: &str) -> Result<String> {
        // Simple language detection based on character patterns
        // Would use a proper library like whatlang in production
        let text_lower = text.to_lowercase();

        if text_lower.chars().any(|c| matches!(c, 'ñ' | '¿' | '¡')) {
            Ok("es".to_string())
        } else if text_lower.chars().any(|c| matches!(c, 'ß' | 'ü' | 'ö' | 'ä')) {
            Ok("de".to_string())
        } else if text_lower.chars().any(|c| matches!(c, 'é' | 'è' | 'ê' | 'ç')) {
            Ok("fr".to_string())
        } else if text_lower.chars().any(|c| c >= '\u{4e00}' && c <= '\u{9fff}') {
            Ok("zh".to_string())
        } else if text_lower.chars().any(|c| c >= '\u{3040}' && c <= '\u{309f}') {
            Ok("ja".to_string())
        } else {
            Ok("en".to_string())
        }
    }

    async fn perform_translation(&self, text: &str, from: &str, to: &str) -> Result<String> {
        // Would integrate with LibreTranslate, Argos Translate, or AI model
        // Placeholder that returns original text with language note
        Ok(format!("[Translated from {} to {}]: {}", from, to, text))
    }

    fn generate_cultural_notes(&self, from: &str, to: &str) -> Vec<CulturalNote> {
        let mut notes = Vec::new();

        // Add relevant cultural notes based on language pair
        if from == "en" && to == "ja" {
            notes.push(CulturalNote {
                note_type: CulturalNoteType::FormalityDifference,
                description: "Japanese email typically requires honorific language (keigo)".to_string(),
                suggestion: Some("Consider using more formal expressions".to_string()),
            });
            notes.push(CulturalNote {
                note_type: CulturalNoteType::BusinessEtiquette,
                description: "Starting with a seasonal greeting is common in Japanese business emails".to_string(),
                suggestion: None,
            });
        }

        if from == "en" && to == "de" {
            notes.push(CulturalNote {
                note_type: CulturalNoteType::FormalityDifference,
                description: "German business email typically uses 'Sie' (formal you)".to_string(),
                suggestion: Some("Use formal address unless you know the recipient well".to_string()),
            });
        }

        notes
    }

    pub async fn get_supported_languages(&self) -> Vec<LanguageInfo> {
        vec![
            LanguageInfo { code: "en".to_string(), name: "English".to_string(), native_name: "English".to_string() },
            LanguageInfo { code: "es".to_string(), name: "Spanish".to_string(), native_name: "Español".to_string() },
            LanguageInfo { code: "fr".to_string(), name: "French".to_string(), native_name: "Français".to_string() },
            LanguageInfo { code: "de".to_string(), name: "German".to_string(), native_name: "Deutsch".to_string() },
            LanguageInfo { code: "ja".to_string(), name: "Japanese".to_string(), native_name: "日本語".to_string() },
            LanguageInfo { code: "zh".to_string(), name: "Chinese".to_string(), native_name: "中文".to_string() },
            LanguageInfo { code: "pt".to_string(), name: "Portuguese".to_string(), native_name: "Português".to_string() },
            LanguageInfo { code: "it".to_string(), name: "Italian".to_string(), native_name: "Italiano".to_string() },
            LanguageInfo { code: "ru".to_string(), name: "Russian".to_string(), native_name: "Русский".to_string() },
            LanguageInfo { code: "ko".to_string(), name: "Korean".to_string(), native_name: "한국어".to_string() },
        ]
    }
}

#[derive(Debug, Serialize)]
pub struct LanguageInfo {
    pub code: String,
    pub name: String,
    pub native_name: String,
}

// ============================================================================
// Module Exports
// ============================================================================

pub fn create_summarization_service(pool: PgPool) -> SummarizationService {
    let config = AiModelConfig {
        provider: AiProvider::Ollama { host: "http://localhost:11434".to_string() },
        model_id: "llama3.2".to_string(),
        endpoint: None,
        api_key_ref: None,
        max_tokens: 2048,
        temperature: 0.7,
        features: vec![AiFeature::Summarization],
    };
    SummarizationService::new(pool, config)
}

pub fn create_action_extraction_service(pool: PgPool) -> ActionExtractionService {
    let config = AiModelConfig {
        provider: AiProvider::Ollama { host: "http://localhost:11434".to_string() },
        model_id: "llama3.2".to_string(),
        endpoint: None,
        api_key_ref: None,
        max_tokens: 1024,
        temperature: 0.3,
        features: vec![AiFeature::ActionExtraction],
    };
    ActionExtractionService::new(pool, config)
}

pub fn create_smart_reply_service(pool: PgPool) -> SmartReplyService {
    let config = AiModelConfig {
        provider: AiProvider::Ollama { host: "http://localhost:11434".to_string() },
        model_id: "llama3.2".to_string(),
        endpoint: None,
        api_key_ref: None,
        max_tokens: 512,
        temperature: 0.8,
        features: vec![AiFeature::SmartReply],
    };
    SmartReplyService::new(pool, config)
}
