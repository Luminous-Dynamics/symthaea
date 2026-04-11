// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! AI Copilot
//!
//! Conversational email assistant with natural language understanding

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;
use std::collections::HashMap;

// ============================================================================
// Copilot Service
// ============================================================================

pub struct CopilotService {
    pool: PgPool,
    llm_provider: Box<dyn LLMProvider>,
    context_manager: ContextManager,
}

#[async_trait]
pub trait LLMProvider: Send + Sync {
    async fn complete(&self, prompt: &str, options: CompletionOptions) -> Result<String, CopilotError>;
    async fn chat(&self, messages: Vec<ChatMessage>, options: CompletionOptions) -> Result<String, CopilotError>;
    async fn embed(&self, text: &str) -> Result<Vec<f32>, CopilotError>;
}

#[derive(Debug, Clone)]
pub struct CompletionOptions {
    pub max_tokens: u32,
    pub temperature: f32,
    pub stop_sequences: Vec<String>,
}

impl Default for CompletionOptions {
    fn default() -> Self {
        Self {
            max_tokens: 1024,
            temperature: 0.7,
            stop_sequences: vec![],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl CopilotService {
    pub fn new(pool: PgPool, llm_provider: Box<dyn LLMProvider>) -> Self {
        Self {
            pool,
            llm_provider,
            context_manager: ContextManager::new(),
        }
    }

    /// Process a natural language command
    pub async fn process_command(
        &self,
        user_id: Uuid,
        command: &str,
        conversation_id: Option<Uuid>,
    ) -> Result<CopilotResponse, CopilotError> {
        // Parse intent from command
        let intent = self.parse_intent(command).await?;

        // Execute based on intent
        let response = match intent {
            CopilotIntent::SummarizeUnread => {
                self.handle_summarize_unread(user_id).await?
            }
            CopilotIntent::SummarizeEmail { email_id } => {
                self.handle_summarize_email(user_id, email_id).await?
            }
            CopilotIntent::DraftReply { email_id, instructions } => {
                self.handle_draft_reply(user_id, email_id, &instructions).await?
            }
            CopilotIntent::ComposeEmail { to, subject, instructions } => {
                self.handle_compose_email(user_id, to, subject, &instructions).await?
            }
            CopilotIntent::SearchEmails { query } => {
                self.handle_search_emails(user_id, &query).await?
            }
            CopilotIntent::ScheduleFollowUp { email_id, when } => {
                self.handle_schedule_followup(user_id, email_id, when).await?
            }
            CopilotIntent::GetPriority => {
                self.handle_get_priority(user_id).await?
            }
            CopilotIntent::Unknown { query } => {
                self.handle_general_query(user_id, &query, conversation_id).await?
            }
        };

        // Store conversation history
        if let Some(conv_id) = conversation_id {
            self.store_conversation(conv_id, user_id, command, &response).await?;
        }

        Ok(response)
    }

    /// Parse user intent from natural language
    async fn parse_intent(&self, command: &str) -> Result<CopilotIntent, CopilotError> {
        let command_lower = command.to_lowercase();

        // Pattern matching for common intents
        if command_lower.contains("summarize") && command_lower.contains("unread") {
            return Ok(CopilotIntent::SummarizeUnread);
        }

        if command_lower.contains("summarize") || command_lower.contains("summary") {
            if let Some(email_id) = self.extract_email_id(&command_lower) {
                return Ok(CopilotIntent::SummarizeEmail { email_id });
            }
        }

        if command_lower.contains("reply") || command_lower.contains("respond") {
            if let Some(email_id) = self.extract_email_id(&command_lower) {
                let instructions = self.extract_instructions(command);
                return Ok(CopilotIntent::DraftReply { email_id, instructions });
            }
        }

        if command_lower.starts_with("compose") || command_lower.starts_with("write") ||
           command_lower.starts_with("draft") || command_lower.contains("email to") {
            let (to, subject, instructions) = self.extract_compose_params(command);
            return Ok(CopilotIntent::ComposeEmail { to, subject, instructions });
        }

        if command_lower.starts_with("search") || command_lower.starts_with("find") {
            let query = command.trim_start_matches(|c: char| !c.is_alphabetic())
                .trim_start_matches("search")
                .trim_start_matches("find")
                .trim()
                .to_string();
            return Ok(CopilotIntent::SearchEmails { query });
        }

        if command_lower.contains("follow up") || command_lower.contains("remind") {
            if let Some(email_id) = self.extract_email_id(&command_lower) {
                let when = self.extract_time_reference(&command_lower);
                return Ok(CopilotIntent::ScheduleFollowUp { email_id, when });
            }
        }

        if command_lower.contains("priority") || command_lower.contains("important") ||
           command_lower.contains("urgent") {
            return Ok(CopilotIntent::GetPriority);
        }

        // Fall back to general query
        Ok(CopilotIntent::Unknown { query: command.to_string() })
    }

    fn extract_email_id(&self, _text: &str) -> Option<Uuid> {
        // Would extract email ID from context or explicit mention
        None
    }

    fn extract_instructions(&self, text: &str) -> String {
        // Extract the instruction part after "reply" or "respond"
        text.split_once("reply")
            .or_else(|| text.split_once("respond"))
            .map(|(_, rest)| rest.trim().to_string())
            .unwrap_or_default()
    }

    fn extract_compose_params(&self, text: &str) -> (Option<String>, Option<String>, String) {
        // Simple extraction - would use NLP in production
        let to = if text.contains(" to ") {
            text.split(" to ")
                .nth(1)
                .and_then(|s| s.split_whitespace().next())
                .map(|s| s.to_string())
        } else {
            None
        };

        let subject = if text.contains("about") {
            text.split("about")
                .nth(1)
                .map(|s| s.trim().to_string())
        } else {
            None
        };

        (to, subject, text.to_string())
    }

    fn extract_time_reference(&self, text: &str) -> Option<DateTime<Utc>> {
        // Simple time extraction
        if text.contains("tomorrow") {
            Some(Utc::now() + chrono::Duration::days(1))
        } else if text.contains("next week") {
            Some(Utc::now() + chrono::Duration::weeks(1))
        } else if text.contains("in an hour") || text.contains("in 1 hour") {
            Some(Utc::now() + chrono::Duration::hours(1))
        } else {
            None
        }
    }

    // ========================================================================
    // Intent Handlers
    // ========================================================================

    async fn handle_summarize_unread(&self, user_id: Uuid) -> Result<CopilotResponse, CopilotError> {
        // Get unread emails
        let unread: Vec<EmailSummaryRow> = sqlx::query_as(
            r#"
            SELECT id, subject, from_address, body_text, received_at
            FROM emails
            WHERE user_id = $1 AND is_read = false AND folder = 'inbox'
            ORDER BY received_at DESC
            LIMIT 20
            "#,
        )
        .bind(user_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CopilotError::Database(e.to_string()))?;

        if unread.is_empty() {
            return Ok(CopilotResponse {
                message: "You have no unread emails. Inbox zero achieved! 🎉".to_string(),
                actions: vec![],
                data: None,
            });
        }

        // Build summary prompt
        let email_list: String = unread
            .iter()
            .map(|e| format!("- From: {}\n  Subject: {}\n  Preview: {}",
                e.from_address,
                e.subject,
                e.body_text.chars().take(100).collect::<String>()
            ))
            .collect::<Vec<_>>()
            .join("\n\n");

        let prompt = format!(
            "Summarize these {} unread emails concisely. Group by topic/sender if relevant:\n\n{}",
            unread.len(),
            email_list
        );

        let summary = self.llm_provider.complete(&prompt, CompletionOptions {
            max_tokens: 500,
            temperature: 0.3,
            ..Default::default()
        }).await?;

        Ok(CopilotResponse {
            message: format!("You have {} unread emails:\n\n{}", unread.len(), summary),
            actions: vec![
                CopilotAction::ViewEmails { filter: "is:unread".to_string() },
            ],
            data: Some(serde_json::json!({
                "unread_count": unread.len(),
                "email_ids": unread.iter().map(|e| e.id).collect::<Vec<_>>()
            })),
        })
    }

    async fn handle_summarize_email(
        &self,
        user_id: Uuid,
        email_id: Uuid,
    ) -> Result<CopilotResponse, CopilotError> {
        let email: EmailSummaryRow = sqlx::query_as(
            "SELECT id, subject, from_address, body_text, received_at FROM emails WHERE id = $1 AND user_id = $2",
        )
        .bind(email_id)
        .bind(user_id)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| CopilotError::Database(e.to_string()))?;

        let prompt = format!(
            "Summarize this email in 2-3 sentences. Identify key points, action items, and deadlines:\n\nFrom: {}\nSubject: {}\n\n{}",
            email.from_address, email.subject, email.body_text
        );

        let summary = self.llm_provider.complete(&prompt, CompletionOptions {
            max_tokens: 200,
            temperature: 0.3,
            ..Default::default()
        }).await?;

        Ok(CopilotResponse {
            message: summary,
            actions: vec![
                CopilotAction::ViewEmail { email_id },
                CopilotAction::ReplyToEmail { email_id },
            ],
            data: None,
        })
    }

    async fn handle_draft_reply(
        &self,
        user_id: Uuid,
        email_id: Uuid,
        instructions: &str,
    ) -> Result<CopilotResponse, CopilotError> {
        let email: EmailSummaryRow = sqlx::query_as(
            "SELECT id, subject, from_address, body_text, received_at FROM emails WHERE id = $1 AND user_id = $2",
        )
        .bind(email_id)
        .bind(user_id)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| CopilotError::Database(e.to_string()))?;

        let user_context = self.get_user_context(user_id).await?;

        let prompt = format!(
            r#"Draft a professional email reply.

Original email from {}:
Subject: {}
---
{}
---

User's instructions: {}

User context: {}

Write a clear, professional reply. Match the tone of the original. Be concise."#,
            email.from_address,
            email.subject,
            email.body_text.chars().take(1000).collect::<String>(),
            if instructions.is_empty() { "Write an appropriate response" } else { instructions },
            user_context
        );

        let draft = self.llm_provider.complete(&prompt, CompletionOptions {
            max_tokens: 500,
            temperature: 0.7,
            ..Default::default()
        }).await?;

        Ok(CopilotResponse {
            message: format!("Here's a draft reply:\n\n{}", draft),
            actions: vec![
                CopilotAction::UseDraft {
                    email_id: Some(email_id),
                    draft: draft.clone(),
                },
                CopilotAction::EditDraft { draft },
            ],
            data: None,
        })
    }

    async fn handle_compose_email(
        &self,
        user_id: Uuid,
        to: Option<String>,
        subject: Option<String>,
        instructions: &str,
    ) -> Result<CopilotResponse, CopilotError> {
        let user_context = self.get_user_context(user_id).await?;

        let prompt = format!(
            r#"Compose a professional email.

To: {}
Subject: {}
Instructions: {}

User context: {}

Write a clear, professional email. Be concise and direct."#,
            to.as_deref().unwrap_or("[recipient]"),
            subject.as_deref().unwrap_or("[subject to be determined]"),
            instructions,
            user_context
        );

        let draft = self.llm_provider.complete(&prompt, CompletionOptions {
            max_tokens: 500,
            temperature: 0.7,
            ..Default::default()
        }).await?;

        Ok(CopilotResponse {
            message: format!("Here's your draft:\n\n{}", draft),
            actions: vec![
                CopilotAction::OpenCompose {
                    to,
                    subject,
                    body: Some(draft.clone()),
                },
                CopilotAction::EditDraft { draft },
            ],
            data: None,
        })
    }

    async fn handle_search_emails(
        &self,
        user_id: Uuid,
        query: &str,
    ) -> Result<CopilotResponse, CopilotError> {
        // Convert natural language to search query
        let search_query = self.nl_to_search_query(query).await?;

        Ok(CopilotResponse {
            message: format!("Searching for: {}", search_query),
            actions: vec![
                CopilotAction::Search { query: search_query },
            ],
            data: None,
        })
    }

    async fn handle_schedule_followup(
        &self,
        user_id: Uuid,
        email_id: Uuid,
        when: Option<DateTime<Utc>>,
    ) -> Result<CopilotResponse, CopilotError> {
        let reminder_time = when.unwrap_or_else(|| Utc::now() + chrono::Duration::days(1));

        // Create reminder
        sqlx::query(
            r#"
            INSERT INTO reminders (id, user_id, email_id, remind_at, message, created_at)
            VALUES ($1, $2, $3, $4, 'Follow up on this email', NOW())
            "#,
        )
        .bind(Uuid::new_v4())
        .bind(user_id)
        .bind(email_id)
        .bind(reminder_time)
        .execute(&self.pool)
        .await
        .map_err(|e| CopilotError::Database(e.to_string()))?;

        Ok(CopilotResponse {
            message: format!("I'll remind you to follow up on {}",
                reminder_time.format("%B %d at %H:%M")),
            actions: vec![],
            data: Some(serde_json::json!({
                "reminder_time": reminder_time
            })),
        })
    }

    async fn handle_get_priority(
        &self,
        user_id: Uuid,
    ) -> Result<CopilotResponse, CopilotError> {
        // Get high priority emails
        let priority_emails: Vec<PriorityEmailRow> = sqlx::query_as(
            r#"
            SELECT e.id, e.subject, e.from_address, e.received_at,
                   c.trust_score
            FROM emails e
            LEFT JOIN contacts c ON c.email = e.from_address AND c.user_id = e.user_id
            WHERE e.user_id = $1 AND e.is_read = false AND e.folder = 'inbox'
            ORDER BY COALESCE(c.trust_score, 0.5) DESC, e.received_at DESC
            LIMIT 5
            "#,
        )
        .bind(user_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CopilotError::Database(e.to_string()))?;

        if priority_emails.is_empty() {
            return Ok(CopilotResponse {
                message: "No unread emails to prioritize.".to_string(),
                actions: vec![],
                data: None,
            });
        }

        let list: String = priority_emails
            .iter()
            .enumerate()
            .map(|(i, e)| format!(
                "{}. **{}** from {}\n   Trust: {:.0}%",
                i + 1,
                e.subject,
                e.from_address,
                e.trust_score.unwrap_or(0.5) * 100.0
            ))
            .collect::<Vec<_>>()
            .join("\n\n");

        Ok(CopilotResponse {
            message: format!("Here are your priority emails:\n\n{}", list),
            actions: priority_emails.iter().map(|e|
                CopilotAction::ViewEmail { email_id: e.id }
            ).collect(),
            data: None,
        })
    }

    async fn handle_general_query(
        &self,
        user_id: Uuid,
        query: &str,
        conversation_id: Option<Uuid>,
    ) -> Result<CopilotResponse, CopilotError> {
        // Get conversation history if available
        let history = if let Some(conv_id) = conversation_id {
            self.get_conversation_history(conv_id).await?
        } else {
            vec![]
        };

        let mut messages = vec![
            ChatMessage {
                role: ChatRole::System,
                content: r#"You are a helpful email assistant for Mycelix Mail. You can help users:
- Summarize emails and threads
- Draft replies and new emails
- Search for emails
- Manage their inbox
- Schedule follow-ups and reminders

Be concise and helpful. If you need to perform an action, describe what you would do."#.to_string(),
            }
        ];

        messages.extend(history);
        messages.push(ChatMessage {
            role: ChatRole::User,
            content: query.to_string(),
        });

        let response = self.llm_provider.chat(messages, CompletionOptions {
            max_tokens: 500,
            temperature: 0.7,
            ..Default::default()
        }).await?;

        Ok(CopilotResponse {
            message: response,
            actions: vec![],
            data: None,
        })
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    async fn get_user_context(&self, user_id: Uuid) -> Result<String, CopilotError> {
        let user: Option<UserContextRow> = sqlx::query_as(
            "SELECT name, email FROM users WHERE id = $1",
        )
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| CopilotError::Database(e.to_string()))?;

        Ok(user.map(|u| format!("User: {} <{}>", u.name.unwrap_or_default(), u.email))
            .unwrap_or_default())
    }

    async fn nl_to_search_query(&self, natural_query: &str) -> Result<String, CopilotError> {
        // Convert natural language to search syntax
        let prompt = format!(
            r#"Convert this natural language query to email search syntax:
"{}"

Use operators like: from:, to:, subject:, has:attachment, is:unread, after:, before:
Return only the search query, nothing else."#,
            natural_query
        );

        self.llm_provider.complete(&prompt, CompletionOptions {
            max_tokens: 100,
            temperature: 0.1,
            ..Default::default()
        }).await
    }

    async fn get_conversation_history(&self, conv_id: Uuid) -> Result<Vec<ChatMessage>, CopilotError> {
        let messages: Vec<ConversationRow> = sqlx::query_as(
            r#"
            SELECT role, content FROM copilot_conversations
            WHERE conversation_id = $1
            ORDER BY created_at
            LIMIT 10
            "#,
        )
        .bind(conv_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CopilotError::Database(e.to_string()))?;

        Ok(messages.into_iter().map(|m| ChatMessage {
            role: if m.role == "user" { ChatRole::User } else { ChatRole::Assistant },
            content: m.content,
        }).collect())
    }

    async fn store_conversation(
        &self,
        conv_id: Uuid,
        user_id: Uuid,
        user_message: &str,
        response: &CopilotResponse,
    ) -> Result<(), CopilotError> {
        // Store user message
        sqlx::query(
            "INSERT INTO copilot_conversations (id, conversation_id, user_id, role, content, created_at)
             VALUES ($1, $2, $3, 'user', $4, NOW())",
        )
        .bind(Uuid::new_v4())
        .bind(conv_id)
        .bind(user_id)
        .bind(user_message)
        .execute(&self.pool)
        .await
        .map_err(|e| CopilotError::Database(e.to_string()))?;

        // Store assistant response
        sqlx::query(
            "INSERT INTO copilot_conversations (id, conversation_id, user_id, role, content, created_at)
             VALUES ($1, $2, $3, 'assistant', $4, NOW())",
        )
        .bind(Uuid::new_v4())
        .bind(conv_id)
        .bind(user_id)
        .bind(&response.message)
        .execute(&self.pool)
        .await
        .map_err(|e| CopilotError::Database(e.to_string()))?;

        Ok(())
    }
}

// ============================================================================
// Types
// ============================================================================

#[derive(Debug, Clone)]
pub enum CopilotIntent {
    SummarizeUnread,
    SummarizeEmail { email_id: Uuid },
    DraftReply { email_id: Uuid, instructions: String },
    ComposeEmail { to: Option<String>, subject: Option<String>, instructions: String },
    SearchEmails { query: String },
    ScheduleFollowUp { email_id: Uuid, when: Option<DateTime<Utc>> },
    GetPriority,
    Unknown { query: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopilotResponse {
    pub message: String,
    pub actions: Vec<CopilotAction>,
    pub data: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CopilotAction {
    ViewEmail { email_id: Uuid },
    ViewEmails { filter: String },
    ReplyToEmail { email_id: Uuid },
    OpenCompose { to: Option<String>, subject: Option<String>, body: Option<String> },
    UseDraft { email_id: Option<Uuid>, draft: String },
    EditDraft { draft: String },
    Search { query: String },
}

struct ContextManager {
    // Would manage conversation context, user preferences, etc.
}

impl ContextManager {
    fn new() -> Self {
        Self {}
    }
}

#[derive(Debug, sqlx::FromRow)]
struct EmailSummaryRow {
    id: Uuid,
    subject: String,
    from_address: String,
    body_text: String,
    received_at: DateTime<Utc>,
}

#[derive(Debug, sqlx::FromRow)]
struct PriorityEmailRow {
    id: Uuid,
    subject: String,
    from_address: String,
    received_at: DateTime<Utc>,
    trust_score: Option<f64>,
}

#[derive(Debug, sqlx::FromRow)]
struct UserContextRow {
    name: Option<String>,
    email: String,
}

#[derive(Debug, sqlx::FromRow)]
struct ConversationRow {
    role: String,
    content: String,
}

#[derive(Debug, thiserror::Error)]
pub enum CopilotError {
    #[error("Database error: {0}")]
    Database(String),
    #[error("LLM error: {0}")]
    LLM(String),
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
}
