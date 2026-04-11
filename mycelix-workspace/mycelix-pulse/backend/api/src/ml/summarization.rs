// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Email Summarization
//!
//! AI-powered email and thread summarization

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Email summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailSummary {
    pub email_id: Uuid,
    pub one_liner: String,
    pub key_points: Vec<String>,
    pub action_items: Vec<ActionItem>,
    pub sentiment: Sentiment,
    pub topics: Vec<String>,
    pub urgency: Urgency,
    pub generated_at: chrono::DateTime<chrono::Utc>,
}

/// Thread summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadSummary {
    pub thread_id: Uuid,
    pub subject: String,
    pub participant_count: usize,
    pub email_count: usize,
    pub summary: String,
    pub key_decisions: Vec<String>,
    pub open_questions: Vec<String>,
    pub action_items: Vec<ActionItem>,
    pub timeline: Vec<TimelineEvent>,
    pub sentiment_trend: Vec<(chrono::DateTime<chrono::Utc>, Sentiment)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionItem {
    pub description: String,
    pub assignee: Option<String>,
    pub due_date: Option<chrono::NaiveDate>,
    pub priority: ActionPriority,
    pub status: ActionStatus,
    pub source_email_id: Uuid,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ActionPriority {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ActionStatus {
    Pending,
    InProgress,
    Completed,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Sentiment {
    VeryNegative,
    Negative,
    Neutral,
    Positive,
    VeryPositive,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Urgency {
    Low,
    Normal,
    High,
    Urgent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEvent {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub event_type: TimelineEventType,
    pub description: String,
    pub actor: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimelineEventType {
    ThreadStarted,
    DecisionMade,
    QuestionAsked,
    QuestionAnswered,
    ActionAssigned,
    ActionCompleted,
    ParticipantJoined,
    TopicChanged,
}

/// Summarization provider trait
#[async_trait]
pub trait SummarizationProvider: Send + Sync {
    async fn summarize_email(&self, content: &EmailContent) -> Result<EmailSummary, SummarizationError>;
    async fn summarize_thread(&self, emails: &[EmailContent]) -> Result<ThreadSummary, SummarizationError>;
}

#[derive(Debug, Clone)]
pub struct EmailContent {
    pub id: Uuid,
    pub from: String,
    pub to: Vec<String>,
    pub subject: String,
    pub body: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Local summarization using extractive methods
pub struct LocalSummarizer {
    // Sentence importance weights
    position_weight: f64,
    keyword_weight: f64,
    length_weight: f64,
}

impl Default for LocalSummarizer {
    fn default() -> Self {
        Self {
            position_weight: 0.3,
            keyword_weight: 0.5,
            length_weight: 0.2,
        }
    }
}

impl LocalSummarizer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Extract key sentences using TF-IDF inspired scoring
    fn extract_key_sentences(&self, text: &str, max_sentences: usize) -> Vec<String> {
        let sentences: Vec<&str> = text
            .split(|c| c == '.' || c == '!' || c == '?')
            .map(|s| s.trim())
            .filter(|s| s.len() > 20)
            .collect();

        if sentences.is_empty() {
            return vec![text.chars().take(200).collect()];
        }

        // Calculate word frequencies
        let mut word_freq: HashMap<String, usize> = HashMap::new();
        for sentence in &sentences {
            for word in sentence.split_whitespace() {
                let word = word.to_lowercase();
                if word.len() > 3 {
                    *word_freq.entry(word).or_insert(0) += 1;
                }
            }
        }

        // Score sentences
        let mut scored: Vec<(usize, f64, &str)> = sentences
            .iter()
            .enumerate()
            .map(|(idx, sentence)| {
                let position_score = 1.0 - (idx as f64 / sentences.len() as f64);

                let keyword_score: f64 = sentence
                    .split_whitespace()
                    .filter_map(|w| word_freq.get(&w.to_lowercase()))
                    .map(|&f| f as f64)
                    .sum::<f64>()
                    / sentence.split_whitespace().count().max(1) as f64;

                let length_score = (sentence.len() as f64 / 100.0).min(1.0);

                let total = self.position_weight * position_score
                    + self.keyword_weight * keyword_score
                    + self.length_weight * length_score;

                (idx, total, *sentence)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        scored
            .into_iter()
            .take(max_sentences)
            .map(|(_, _, s)| format!("{}.", s.trim_end_matches('.')))
            .collect()
    }

    /// Detect sentiment from text
    fn detect_sentiment(&self, text: &str) -> Sentiment {
        let text_lower = text.to_lowercase();

        let positive_words = [
            "thank", "great", "excellent", "happy", "pleased", "wonderful",
            "appreciate", "good", "fantastic", "amazing", "love", "helpful",
        ];
        let negative_words = [
            "sorry", "unfortunately", "problem", "issue", "disappointed",
            "frustrated", "urgent", "asap", "critical", "failed", "error",
        ];

        let positive_count = positive_words.iter()
            .filter(|w| text_lower.contains(*w))
            .count();
        let negative_count = negative_words.iter()
            .filter(|w| text_lower.contains(*w))
            .count();

        match (positive_count, negative_count) {
            (p, n) if p > n + 2 => Sentiment::VeryPositive,
            (p, n) if p > n => Sentiment::Positive,
            (p, n) if n > p + 2 => Sentiment::VeryNegative,
            (p, n) if n > p => Sentiment::Negative,
            _ => Sentiment::Neutral,
        }
    }

    /// Detect urgency from text
    fn detect_urgency(&self, text: &str) -> Urgency {
        let text_lower = text.to_lowercase();

        let urgent_indicators = [
            "urgent", "asap", "immediately", "critical", "emergency",
            "right away", "time sensitive", "deadline",
        ];
        let high_indicators = [
            "important", "priority", "soon", "today", "eod", "by tomorrow",
        ];

        if urgent_indicators.iter().any(|w| text_lower.contains(w)) {
            Urgency::Urgent
        } else if high_indicators.iter().any(|w| text_lower.contains(w)) {
            Urgency::High
        } else {
            Urgency::Normal
        }
    }

    /// Extract action items from text
    fn extract_action_items(&self, text: &str, email_id: Uuid) -> Vec<ActionItem> {
        let mut actions = Vec::new();

        let action_patterns = [
            "please", "could you", "can you", "need to", "must",
            "should", "will you", "todo", "action:", "action item",
        ];

        for sentence in text.split(|c| c == '.' || c == '!' || c == '?') {
            let sentence_lower = sentence.to_lowercase();

            if action_patterns.iter().any(|p| sentence_lower.contains(p)) {
                // Extract potential assignee
                let assignee = self.extract_assignee(&sentence_lower);

                // Extract potential due date
                let due_date = self.extract_due_date(&sentence_lower);

                actions.push(ActionItem {
                    description: sentence.trim().to_string(),
                    assignee,
                    due_date,
                    priority: if sentence_lower.contains("urgent") || sentence_lower.contains("asap") {
                        ActionPriority::High
                    } else {
                        ActionPriority::Medium
                    },
                    status: ActionStatus::Pending,
                    source_email_id: email_id,
                });
            }
        }

        actions
    }

    fn extract_assignee(&self, text: &str) -> Option<String> {
        // Simple pattern: "@name" or "assign to name"
        if let Some(at_pos) = text.find('@') {
            let after_at = &text[at_pos + 1..];
            let name: String = after_at
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                return Some(name);
            }
        }
        None
    }

    fn extract_due_date(&self, text: &str) -> Option<chrono::NaiveDate> {
        let today = chrono::Utc::now().date_naive();

        if text.contains("today") || text.contains("eod") {
            return Some(today);
        }
        if text.contains("tomorrow") {
            return Some(today + chrono::Duration::days(1));
        }
        if text.contains("next week") {
            return Some(today + chrono::Duration::weeks(1));
        }
        if text.contains("by friday") {
            // Find next Friday
            let days_until_friday = (5 - today.weekday().num_days_from_monday() as i64 + 7) % 7;
            return Some(today + chrono::Duration::days(if days_until_friday == 0 { 7 } else { days_until_friday }));
        }

        None
    }

    /// Extract topics from text
    fn extract_topics(&self, text: &str) -> Vec<String> {
        let mut word_freq: HashMap<String, usize> = HashMap::new();

        // Common stop words to ignore
        let stop_words = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can",
            "this", "that", "these", "those", "i", "you", "he", "she",
            "it", "we", "they", "what", "which", "who", "when", "where",
            "why", "how", "all", "each", "every", "both", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only",
            "own", "same", "so", "than", "too", "very", "just", "and",
            "but", "if", "or", "because", "as", "until", "while", "of",
            "at", "by", "for", "with", "about", "against", "between",
            "into", "through", "during", "before", "after", "above",
            "below", "to", "from", "up", "down", "in", "out", "on", "off",
        ];

        for word in text.split_whitespace() {
            let word = word.to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>();

            if word.len() > 4 && !stop_words.contains(&word.as_str()) {
                *word_freq.entry(word).or_insert(0) += 1;
            }
        }

        let mut topics: Vec<(String, usize)> = word_freq.into_iter().collect();
        topics.sort_by(|a, b| b.1.cmp(&a.1));

        topics.into_iter()
            .take(5)
            .map(|(word, _)| word)
            .collect()
    }
}

#[async_trait]
impl SummarizationProvider for LocalSummarizer {
    async fn summarize_email(&self, content: &EmailContent) -> Result<EmailSummary, SummarizationError> {
        let key_sentences = self.extract_key_sentences(&content.body, 3);
        let one_liner = key_sentences.first()
            .cloned()
            .unwrap_or_else(|| content.subject.clone());

        let sentiment = self.detect_sentiment(&content.body);
        let urgency = self.detect_urgency(&format!("{} {}", content.subject, content.body));
        let action_items = self.extract_action_items(&content.body, content.id);
        let topics = self.extract_topics(&content.body);

        Ok(EmailSummary {
            email_id: content.id,
            one_liner,
            key_points: key_sentences,
            action_items,
            sentiment,
            topics,
            urgency,
            generated_at: chrono::Utc::now(),
        })
    }

    async fn summarize_thread(&self, emails: &[EmailContent]) -> Result<ThreadSummary, SummarizationError> {
        if emails.is_empty() {
            return Err(SummarizationError::EmptyInput);
        }

        let thread_id = emails[0].id;
        let subject = emails[0].subject.clone();

        // Combine all email bodies
        let combined_text: String = emails
            .iter()
            .map(|e| e.body.clone())
            .collect::<Vec<_>>()
            .join("\n\n");

        let summary = self.extract_key_sentences(&combined_text, 5).join(" ");

        // Collect all action items
        let action_items: Vec<ActionItem> = emails
            .iter()
            .flat_map(|e| self.extract_action_items(&e.body, e.id))
            .collect();

        // Extract unique participants
        let mut participants: std::collections::HashSet<String> = std::collections::HashSet::new();
        for email in emails {
            participants.insert(email.from.clone());
            participants.extend(email.to.iter().cloned());
        }

        // Sentiment trend
        let sentiment_trend: Vec<(chrono::DateTime<chrono::Utc>, Sentiment)> = emails
            .iter()
            .map(|e| (e.timestamp, self.detect_sentiment(&e.body)))
            .collect();

        Ok(ThreadSummary {
            thread_id,
            subject,
            participant_count: participants.len(),
            email_count: emails.len(),
            summary,
            key_decisions: Vec::new(),  // Would need more sophisticated NLP
            open_questions: Vec::new(), // Would need more sophisticated NLP
            action_items,
            timeline: Vec::new(),
            sentiment_trend,
        })
    }
}

/// LLM-based summarization (external API)
pub struct LLMSummarizer {
    api_endpoint: String,
    api_key: String,
    model: String,
}

impl LLMSummarizer {
    pub fn new(api_endpoint: String, api_key: String, model: String) -> Self {
        Self {
            api_endpoint,
            api_key,
            model,
        }
    }
}

#[async_trait]
impl SummarizationProvider for LLMSummarizer {
    async fn summarize_email(&self, content: &EmailContent) -> Result<EmailSummary, SummarizationError> {
        // This would call an external LLM API
        // For now, fall back to local summarization
        let local = LocalSummarizer::new();
        local.summarize_email(content).await
    }

    async fn summarize_thread(&self, emails: &[EmailContent]) -> Result<ThreadSummary, SummarizationError> {
        let local = LocalSummarizer::new();
        local.summarize_thread(emails).await
    }
}

#[derive(Debug)]
pub enum SummarizationError {
    EmptyInput,
    ApiError(String),
    ParseError(String),
}

impl std::fmt::Display for SummarizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "Empty input provided"),
            Self::ApiError(msg) => write!(f, "API error: {}", msg),
            Self::ParseError(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl std::error::Error for SummarizationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_local_summarization() {
        let summarizer = LocalSummarizer::new();

        let content = EmailContent {
            id: Uuid::new_v4(),
            from: "sender@example.com".to_string(),
            to: vec!["recipient@example.com".to_string()],
            subject: "Project Update".to_string(),
            body: "Hi team, I wanted to provide an update on the project. \
                   We have completed the first phase successfully. \
                   Please review the attached documents by tomorrow. \
                   The next milestone is due next week. \
                   Thank you for your hard work!".to_string(),
            timestamp: chrono::Utc::now(),
        };

        let summary = summarizer.summarize_email(&content).await.unwrap();

        assert!(!summary.one_liner.is_empty());
        assert!(summary.urgency == Urgency::High); // "by tomorrow"
        assert!(summary.sentiment == Sentiment::Positive); // "Thank you"
    }
}
