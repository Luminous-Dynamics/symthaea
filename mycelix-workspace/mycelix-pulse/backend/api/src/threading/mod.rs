// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Email Threading & Conversation Intelligence
//!
//! Thread grouping, conversation view, and AI-powered thread analysis

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;
use std::collections::{HashMap, HashSet};

// ============================================================================
// Threading Service
// ============================================================================

pub struct ThreadingService {
    pool: PgPool,
}

impl ThreadingService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Build thread from email headers (References, In-Reply-To)
    pub async fn process_email_threading(
        &self,
        email_id: Uuid,
        headers: &EmailHeaders,
    ) -> Result<Uuid, ThreadingError> {
        // Try to find existing thread
        let thread_id = if let Some(ref in_reply_to) = headers.in_reply_to {
            // Find thread by In-Reply-To message ID
            self.find_thread_by_message_id(in_reply_to).await?
        } else if !headers.references.is_empty() {
            // Find thread by any reference
            self.find_thread_by_references(&headers.references).await?
        } else {
            None
        };

        let thread_id = match thread_id {
            Some(id) => {
                // Add email to existing thread
                self.add_to_thread(id, email_id, headers).await?;
                id
            }
            None => {
                // Create new thread
                self.create_thread(email_id, headers).await?
            }
        };

        // Update thread metadata
        self.update_thread_metadata(thread_id).await?;

        Ok(thread_id)
    }

    async fn find_thread_by_message_id(
        &self,
        message_id: &str,
    ) -> Result<Option<Uuid>, ThreadingError> {
        let result: Option<(Uuid,)> = sqlx::query_as(
            "SELECT thread_id FROM emails WHERE message_id = $1 AND thread_id IS NOT NULL LIMIT 1",
        )
        .bind(message_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| ThreadingError::Database(e.to_string()))?;

        Ok(result.map(|(id,)| id))
    }

    async fn find_thread_by_references(
        &self,
        references: &[String],
    ) -> Result<Option<Uuid>, ThreadingError> {
        for ref_id in references.iter().rev() {
            if let Some(thread_id) = self.find_thread_by_message_id(ref_id).await? {
                return Ok(Some(thread_id));
            }
        }
        Ok(None)
    }

    async fn create_thread(
        &self,
        email_id: Uuid,
        headers: &EmailHeaders,
    ) -> Result<Uuid, ThreadingError> {
        let thread_id = Uuid::new_v4();

        sqlx::query(
            r#"
            INSERT INTO email_threads (id, subject, participant_count, message_count,
                                       first_message_at, last_message_at, created_at)
            VALUES ($1, $2, 1, 1, $3, $3, NOW())
            "#,
        )
        .bind(thread_id)
        .bind(&headers.subject)
        .bind(headers.date)
        .execute(&self.pool)
        .await
        .map_err(|e| ThreadingError::Database(e.to_string()))?;

        // Link email to thread
        sqlx::query("UPDATE emails SET thread_id = $1, thread_position = 0 WHERE id = $2")
            .bind(thread_id)
            .bind(email_id)
            .execute(&self.pool)
            .await
            .map_err(|e| ThreadingError::Database(e.to_string()))?;

        Ok(thread_id)
    }

    async fn add_to_thread(
        &self,
        thread_id: Uuid,
        email_id: Uuid,
        headers: &EmailHeaders,
    ) -> Result<(), ThreadingError> {
        // Get current position count
        let count: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM emails WHERE thread_id = $1",
        )
        .bind(thread_id)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| ThreadingError::Database(e.to_string()))?;

        sqlx::query("UPDATE emails SET thread_id = $1, thread_position = $2 WHERE id = $3")
            .bind(thread_id)
            .bind(count.0 as i32)
            .bind(email_id)
            .execute(&self.pool)
            .await
            .map_err(|e| ThreadingError::Database(e.to_string()))?;

        Ok(())
    }

    async fn update_thread_metadata(&self, thread_id: Uuid) -> Result<(), ThreadingError> {
        sqlx::query(
            r#"
            UPDATE email_threads SET
                message_count = (SELECT COUNT(*) FROM emails WHERE thread_id = $1),
                participant_count = (
                    SELECT COUNT(DISTINCT from_address) FROM emails WHERE thread_id = $1
                ),
                last_message_at = (
                    SELECT MAX(received_at) FROM emails WHERE thread_id = $1
                ),
                has_unread = EXISTS (
                    SELECT 1 FROM emails WHERE thread_id = $1 AND is_read = false
                )
            WHERE id = $1
            "#,
        )
        .bind(thread_id)
        .execute(&self.pool)
        .await
        .map_err(|e| ThreadingError::Database(e.to_string()))?;

        Ok(())
    }

    /// Get conversation view for a thread
    pub async fn get_conversation(
        &self,
        user_id: Uuid,
        thread_id: Uuid,
    ) -> Result<Conversation, ThreadingError> {
        let thread: ThreadRow = sqlx::query_as(
            r#"
            SELECT id, subject, participant_count, message_count,
                   first_message_at, last_message_at, has_unread
            FROM email_threads WHERE id = $1
            "#,
        )
        .bind(thread_id)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| ThreadingError::Database(e.to_string()))?;

        let messages: Vec<ThreadMessage> = sqlx::query_as(
            r#"
            SELECT id, from_address, from_name, to_addresses, cc_addresses,
                   subject, body_text, body_html, received_at, is_read,
                   has_attachments, thread_position
            FROM emails
            WHERE thread_id = $1 AND user_id = $2
            ORDER BY thread_position ASC
            "#,
        )
        .bind(thread_id)
        .bind(user_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| ThreadingError::Database(e.to_string()))?;

        // Extract unique participants
        let mut participants = HashSet::new();
        for msg in &messages {
            participants.insert(msg.from_address.clone());
        }

        Ok(Conversation {
            thread_id,
            subject: thread.subject,
            participants: participants.into_iter().collect(),
            message_count: thread.message_count,
            messages,
            first_message_at: thread.first_message_at,
            last_message_at: thread.last_message_at,
            has_unread: thread.has_unread,
        })
    }

    /// Get threaded inbox view
    pub async fn get_threaded_inbox(
        &self,
        user_id: Uuid,
        folder: &str,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<ThreadSummary>, ThreadingError> {
        let threads: Vec<ThreadSummary> = sqlx::query_as(
            r#"
            SELECT DISTINCT ON (t.id)
                t.id as thread_id,
                t.subject,
                t.message_count,
                t.participant_count,
                t.last_message_at,
                t.has_unread,
                e.from_address as last_sender,
                e.from_name as last_sender_name,
                LEFT(e.body_text, 200) as preview
            FROM email_threads t
            JOIN emails e ON e.thread_id = t.id
            WHERE e.user_id = $1 AND e.folder = $2
            ORDER BY t.id, e.received_at DESC
            LIMIT $3 OFFSET $4
            "#,
        )
        .bind(user_id)
        .bind(folder)
        .bind(limit)
        .bind(offset)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| ThreadingError::Database(e.to_string()))?;

        Ok(threads)
    }

    /// Split a thread at a specific message
    pub async fn split_thread(
        &self,
        user_id: Uuid,
        email_id: Uuid,
    ) -> Result<Uuid, ThreadingError> {
        // Get the email and its current thread
        let email: (Uuid, i32, String) = sqlx::query_as(
            "SELECT thread_id, thread_position, subject FROM emails WHERE id = $1 AND user_id = $2",
        )
        .bind(email_id)
        .bind(user_id)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| ThreadingError::Database(e.to_string()))?;

        let (old_thread_id, position, subject) = email;

        // Create new thread
        let new_thread_id = Uuid::new_v4();

        sqlx::query(
            r#"
            INSERT INTO email_threads (id, subject, created_at)
            VALUES ($1, $2, NOW())
            "#,
        )
        .bind(new_thread_id)
        .bind(&subject)
        .execute(&self.pool)
        .await
        .map_err(|e| ThreadingError::Database(e.to_string()))?;

        // Move this email and all subsequent ones to new thread
        sqlx::query(
            r#"
            UPDATE emails
            SET thread_id = $1, thread_position = thread_position - $2
            WHERE thread_id = $3 AND thread_position >= $2
            "#,
        )
        .bind(new_thread_id)
        .bind(position)
        .bind(old_thread_id)
        .execute(&self.pool)
        .await
        .map_err(|e| ThreadingError::Database(e.to_string()))?;

        // Update both thread metadata
        self.update_thread_metadata(old_thread_id).await?;
        self.update_thread_metadata(new_thread_id).await?;

        Ok(new_thread_id)
    }

    /// Merge two threads together
    pub async fn merge_threads(
        &self,
        user_id: Uuid,
        source_thread_id: Uuid,
        target_thread_id: Uuid,
    ) -> Result<(), ThreadingError> {
        // Get current max position in target thread
        let max_pos: (i32,) = sqlx::query_as(
            "SELECT COALESCE(MAX(thread_position), -1) FROM emails WHERE thread_id = $1",
        )
        .bind(target_thread_id)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| ThreadingError::Database(e.to_string()))?;

        // Move all emails from source to target
        sqlx::query(
            r#"
            UPDATE emails
            SET thread_id = $1, thread_position = thread_position + $2 + 1
            WHERE thread_id = $3
            "#,
        )
        .bind(target_thread_id)
        .bind(max_pos.0)
        .bind(source_thread_id)
        .execute(&self.pool)
        .await
        .map_err(|e| ThreadingError::Database(e.to_string()))?;

        // Delete source thread
        sqlx::query("DELETE FROM email_threads WHERE id = $1")
            .bind(source_thread_id)
            .execute(&self.pool)
            .await
            .map_err(|e| ThreadingError::Database(e.to_string()))?;

        // Update target thread metadata
        self.update_thread_metadata(target_thread_id).await?;

        Ok(())
    }
}

// ============================================================================
// Conversation Intelligence
// ============================================================================

pub struct ConversationIntelligence {
    pool: PgPool,
}

impl ConversationIntelligence {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Generate AI summary of a thread
    pub async fn summarize_thread(
        &self,
        thread_id: Uuid,
    ) -> Result<ThreadSummaryAI, ThreadingError> {
        let messages: Vec<(String, String, DateTime<Utc>)> = sqlx::query_as(
            r#"
            SELECT from_address, body_text, received_at
            FROM emails WHERE thread_id = $1
            ORDER BY thread_position ASC
            "#,
        )
        .bind(thread_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| ThreadingError::Database(e.to_string()))?;

        // Build context for summarization
        let context: String = messages
            .iter()
            .map(|(from, body, date)| {
                format!(
                    "From: {} ({})\n{}\n---",
                    from,
                    date.format("%Y-%m-%d %H:%M"),
                    body.chars().take(500).collect::<String>()
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        // Would call LLM here for actual summarization
        let summary = format!(
            "Thread with {} messages discussing the topic.",
            messages.len()
        );

        // Extract key points (would use NLP)
        let key_points = vec![
            "Initial request made".to_string(),
            "Discussion of options".to_string(),
            "Decision pending".to_string(),
        ];

        // Detect action items
        let action_items = self.extract_action_items(&messages).await;

        // Analyze sentiment
        let sentiment = self.analyze_sentiment(&messages);

        Ok(ThreadSummaryAI {
            thread_id,
            summary,
            key_points,
            action_items,
            sentiment,
            generated_at: Utc::now(),
        })
    }

    async fn extract_action_items(
        &self,
        messages: &[(String, String, DateTime<Utc>)],
    ) -> Vec<ActionItem> {
        let mut items = Vec::new();

        // Simple pattern matching (would use NLP in production)
        let action_patterns = [
            "please send",
            "can you",
            "could you",
            "need to",
            "action required",
            "todo:",
            "follow up",
            "deadline",
        ];

        for (from, body, date) in messages {
            let body_lower = body.to_lowercase();
            for pattern in &action_patterns {
                if body_lower.contains(pattern) {
                    // Extract the sentence containing the pattern
                    if let Some(sentence) = body
                        .split('.')
                        .find(|s| s.to_lowercase().contains(pattern))
                    {
                        items.push(ActionItem {
                            text: sentence.trim().to_string(),
                            assigned_to: None,
                            due_date: None,
                            completed: false,
                            source_email_date: *date,
                        });
                    }
                }
            }
        }

        items
    }

    fn analyze_sentiment(
        &self,
        messages: &[(String, String, DateTime<Utc>)],
    ) -> ThreadSentiment {
        // Simple sentiment analysis (would use ML model)
        let mut positive_count = 0;
        let mut negative_count = 0;

        let positive_words = ["thank", "great", "excellent", "happy", "pleased", "appreciate"];
        let negative_words = ["urgent", "problem", "issue", "disappointed", "concern", "unfortunately"];

        for (_, body, _) in messages {
            let body_lower = body.to_lowercase();
            for word in &positive_words {
                if body_lower.contains(word) {
                    positive_count += 1;
                }
            }
            for word in &negative_words {
                if body_lower.contains(word) {
                    negative_count += 1;
                }
            }
        }

        if positive_count > negative_count * 2 {
            ThreadSentiment::Positive
        } else if negative_count > positive_count * 2 {
            ThreadSentiment::Negative
        } else {
            ThreadSentiment::Neutral
        }
    }

    /// Detect if thread needs response
    pub async fn needs_response(&self, thread_id: Uuid, user_email: &str) -> Result<bool, ThreadingError> {
        let last_message: Option<(String,)> = sqlx::query_as(
            r#"
            SELECT from_address FROM emails
            WHERE thread_id = $1
            ORDER BY received_at DESC LIMIT 1
            "#,
        )
        .bind(thread_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| ThreadingError::Database(e.to_string()))?;

        // Thread needs response if last message is not from user
        Ok(last_message.map(|(from,)| from != user_email).unwrap_or(false))
    }

    /// Find related threads by topic similarity
    pub async fn find_related_threads(
        &self,
        user_id: Uuid,
        thread_id: Uuid,
        limit: i64,
    ) -> Result<Vec<RelatedThread>, ThreadingError> {
        // Get thread subject for comparison
        let subject: (String,) = sqlx::query_as(
            "SELECT subject FROM email_threads WHERE id = $1",
        )
        .bind(thread_id)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| ThreadingError::Database(e.to_string()))?;

        // Simple keyword matching (would use embeddings in production)
        let keywords: Vec<&str> = subject.0
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .take(5)
            .collect();

        if keywords.is_empty() {
            return Ok(Vec::new());
        }

        let pattern = keywords.join("|");

        let related: Vec<RelatedThread> = sqlx::query_as(
            r#"
            SELECT DISTINCT t.id, t.subject, t.last_message_at
            FROM email_threads t
            JOIN emails e ON e.thread_id = t.id
            WHERE e.user_id = $1 AND t.id != $2
            AND t.subject ~* $3
            ORDER BY t.last_message_at DESC
            LIMIT $4
            "#,
        )
        .bind(user_id)
        .bind(thread_id)
        .bind(&pattern)
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| ThreadingError::Database(e.to_string()))?;

        Ok(related)
    }
}

// ============================================================================
// Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailHeaders {
    pub message_id: String,
    pub in_reply_to: Option<String>,
    pub references: Vec<String>,
    pub subject: String,
    pub date: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct ThreadRow {
    pub id: Uuid,
    pub subject: String,
    pub participant_count: i32,
    pub message_count: i32,
    pub first_message_at: DateTime<Utc>,
    pub last_message_at: DateTime<Utc>,
    pub has_unread: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub thread_id: Uuid,
    pub subject: String,
    pub participants: Vec<String>,
    pub message_count: i32,
    pub messages: Vec<ThreadMessage>,
    pub first_message_at: DateTime<Utc>,
    pub last_message_at: DateTime<Utc>,
    pub has_unread: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct ThreadMessage {
    pub id: Uuid,
    pub from_address: String,
    pub from_name: Option<String>,
    pub to_addresses: Vec<String>,
    pub cc_addresses: Vec<String>,
    pub subject: String,
    pub body_text: String,
    pub body_html: Option<String>,
    pub received_at: DateTime<Utc>,
    pub is_read: bool,
    pub has_attachments: bool,
    pub thread_position: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct ThreadSummary {
    pub thread_id: Uuid,
    pub subject: String,
    pub message_count: i32,
    pub participant_count: i32,
    pub last_message_at: DateTime<Utc>,
    pub has_unread: bool,
    pub last_sender: String,
    pub last_sender_name: Option<String>,
    pub preview: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadSummaryAI {
    pub thread_id: Uuid,
    pub summary: String,
    pub key_points: Vec<String>,
    pub action_items: Vec<ActionItem>,
    pub sentiment: ThreadSentiment,
    pub generated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionItem {
    pub text: String,
    pub assigned_to: Option<String>,
    pub due_date: Option<DateTime<Utc>>,
    pub completed: bool,
    pub source_email_date: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ThreadSentiment {
    Positive,
    Neutral,
    Negative,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct RelatedThread {
    pub id: Uuid,
    pub subject: String,
    pub last_message_at: DateTime<Utc>,
}

#[derive(Debug, thiserror::Error)]
pub enum ThreadingError {
    #[error("Database error: {0}")]
    Database(String),
    #[error("Thread not found")]
    NotFound,
}
