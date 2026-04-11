// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Smart Compose
//!
//! AI-powered writing suggestions and autocomplete

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;
use std::collections::HashMap;

use super::copilot::{LLMProvider, CompletionOptions, CopilotError};

// ============================================================================
// Smart Compose Service
// ============================================================================

pub struct SmartComposeService {
    pool: PgPool,
    llm_provider: Box<dyn LLMProvider>,
    template_engine: TemplateEngine,
}

impl SmartComposeService {
    pub fn new(pool: PgPool, llm_provider: Box<dyn LLMProvider>) -> Self {
        Self {
            pool,
            llm_provider,
            template_engine: TemplateEngine::new(),
        }
    }

    /// Get autocomplete suggestions for current input
    pub async fn get_suggestions(
        &self,
        user_id: Uuid,
        context: ComposeContext,
    ) -> Result<Vec<Suggestion>, CopilotError> {
        let mut suggestions = Vec::new();

        // Get phrase completions
        if let Some(phrase_suggestions) = self.get_phrase_completions(&context).await? {
            suggestions.extend(phrase_suggestions);
        }

        // Get smart suggestions based on context
        if context.cursor_position > 20 {
            if let Some(smart_suggestions) = self.get_smart_suggestions(user_id, &context).await? {
                suggestions.extend(smart_suggestions);
            }
        }

        // Limit and dedupe
        suggestions.truncate(5);
        Ok(suggestions)
    }

    /// Get inline autocomplete (single suggestion)
    pub async fn get_inline_completion(
        &self,
        user_id: Uuid,
        context: ComposeContext,
    ) -> Result<Option<String>, CopilotError> {
        // Only suggest if user paused typing (would be handled by frontend debounce)
        if context.current_text.len() < 10 {
            return Ok(None);
        }

        // Get the last incomplete sentence/phrase
        let last_text = context.current_text
            .lines()
            .last()
            .unwrap_or(&context.current_text);

        if last_text.len() < 5 {
            return Ok(None);
        }

        let prompt = format!(
            r#"Complete this email text naturally. Only provide the completion, not the original text.

Email context:
To: {}
Subject: {}

Current text:
{}

Complete the current sentence/thought:"#,
            context.to.as_deref().unwrap_or(""),
            context.subject.as_deref().unwrap_or(""),
            context.current_text
        );

        let completion = self.llm_provider.complete(&prompt, CompletionOptions {
            max_tokens: 50,
            temperature: 0.3,
            stop_sequences: vec!["\n".to_string(), ".".to_string()],
        }).await?;

        if completion.trim().is_empty() {
            return Ok(None);
        }

        Ok(Some(completion.trim().to_string()))
    }

    /// Improve/rewrite selected text
    pub async fn improve_text(
        &self,
        user_id: Uuid,
        text: &str,
        style: ImprovementStyle,
    ) -> Result<String, CopilotError> {
        let instruction = match style {
            ImprovementStyle::MoreFormal => "Make this more formal and professional",
            ImprovementStyle::MoreCasual => "Make this more casual and friendly",
            ImprovementStyle::Shorter => "Make this more concise while keeping the meaning",
            ImprovementStyle::Longer => "Expand this with more detail",
            ImprovementStyle::FixGrammar => "Fix any grammar and spelling errors",
            ImprovementStyle::Clearer => "Make this clearer and easier to understand",
        };

        let prompt = format!(
            r#"{}. Return only the improved text, nothing else.

Original text:
{}

Improved text:"#,
            instruction, text
        );

        self.llm_provider.complete(&prompt, CompletionOptions {
            max_tokens: (text.len() * 2).min(1000) as u32,
            temperature: 0.5,
            ..Default::default()
        }).await
    }

    /// Generate email from bullet points
    pub async fn expand_bullets(
        &self,
        user_id: Uuid,
        context: ComposeContext,
        bullets: &[String],
    ) -> Result<String, CopilotError> {
        let user_style = self.get_user_writing_style(user_id).await?;

        let prompt = format!(
            r#"Write a professional email based on these bullet points.

To: {}
Subject: {}

Key points:
{}

Writing style: {}

Write a complete, well-structured email:"#,
            context.to.as_deref().unwrap_or("[recipient]"),
            context.subject.as_deref().unwrap_or("[subject]"),
            bullets.iter().map(|b| format!("• {}", b)).collect::<Vec<_>>().join("\n"),
            user_style
        );

        self.llm_provider.complete(&prompt, CompletionOptions {
            max_tokens: 500,
            temperature: 0.7,
            ..Default::default()
        }).await
    }

    /// Get smart reply suggestions
    pub async fn get_smart_replies(
        &self,
        user_id: Uuid,
        email_id: Uuid,
    ) -> Result<Vec<SmartReply>, CopilotError> {
        let email: EmailRow = sqlx::query_as(
            "SELECT subject, from_address, body_text FROM emails WHERE id = $1 AND user_id = $2",
        )
        .bind(email_id)
        .bind(user_id)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| CopilotError::Database(e.to_string()))?;

        let prompt = format!(
            r#"Generate 3 short reply options for this email. Each should be 1-2 sentences.

From: {}
Subject: {}
---
{}
---

Provide exactly 3 replies, one per line, labeled as:
POSITIVE: [positive/accepting response]
NEUTRAL: [neutral/informative response]
NEGATIVE: [declining/negative response]"#,
            email.from_address,
            email.subject,
            email.body_text.chars().take(500).collect::<String>()
        );

        let response = self.llm_provider.complete(&prompt, CompletionOptions {
            max_tokens: 200,
            temperature: 0.7,
            ..Default::default()
        }).await?;

        // Parse responses
        let mut replies = Vec::new();
        for line in response.lines() {
            if let Some((label, text)) = line.split_once(':') {
                let sentiment = match label.trim().to_uppercase().as_str() {
                    "POSITIVE" => ReplySentiment::Positive,
                    "NEGATIVE" => ReplySentiment::Negative,
                    _ => ReplySentiment::Neutral,
                };
                replies.push(SmartReply {
                    text: text.trim().to_string(),
                    sentiment,
                });
            }
        }

        Ok(replies)
    }

    /// Get templates for common email types
    pub async fn get_templates(
        &self,
        user_id: Uuid,
        category: TemplateCategory,
    ) -> Result<Vec<EmailTemplate>, CopilotError> {
        self.template_engine.get_templates(category)
    }

    /// Apply template with variable substitution
    pub async fn apply_template(
        &self,
        user_id: Uuid,
        template_id: &str,
        variables: HashMap<String, String>,
    ) -> Result<String, CopilotError> {
        self.template_engine.apply(template_id, &variables)
    }

    // ========================================================================
    // Private Helpers
    // ========================================================================

    async fn get_phrase_completions(
        &self,
        context: &ComposeContext,
    ) -> Result<Option<Vec<Suggestion>>, CopilotError> {
        let last_words: String = context.current_text
            .split_whitespace()
            .rev()
            .take(3)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();

        // Common phrase completions
        let completions: HashMap<&str, Vec<&str>> = HashMap::from([
            ("thank you for", vec!["your time", "your help", "getting back to me", "your quick response"]),
            ("i wanted to", vec!["follow up on", "check in about", "let you know", "ask about"]),
            ("please let me know", vec!["if you have any questions", "if this works for you", "your thoughts"]),
            ("looking forward to", vec!["hearing from you", "your response", "working together", "meeting you"]),
            ("i hope this", vec!["helps", "email finds you well", "makes sense", "answers your question"]),
            ("as discussed", vec!["in our meeting", "earlier", "on the call", "previously"]),
            ("i am writing to", vec!["follow up", "inquire about", "request", "confirm"]),
            ("could you please", vec!["send me", "let me know", "confirm", "provide"]),
        ]);

        for (phrase, options) in completions {
            if last_words.ends_with(phrase) {
                return Ok(Some(
                    options.iter().map(|o| Suggestion {
                        text: format!(" {}", o),
                        suggestion_type: SuggestionType::PhraseCompletion,
                        confidence: 0.9,
                    }).collect()
                ));
            }
        }

        Ok(None)
    }

    async fn get_smart_suggestions(
        &self,
        user_id: Uuid,
        context: &ComposeContext,
    ) -> Result<Option<Vec<Suggestion>>, CopilotError> {
        // Would use LLM for context-aware suggestions
        Ok(None)
    }

    async fn get_user_writing_style(&self, user_id: Uuid) -> Result<String, CopilotError> {
        // Analyze user's previous emails to determine style
        // For now, return default
        Ok("Professional, friendly, and concise".to_string())
    }
}

// ============================================================================
// Template Engine
// ============================================================================

struct TemplateEngine {
    templates: HashMap<String, EmailTemplate>,
}

impl TemplateEngine {
    fn new() -> Self {
        let mut templates = HashMap::new();

        // Built-in templates
        templates.insert("meeting_request".to_string(), EmailTemplate {
            id: "meeting_request".to_string(),
            name: "Meeting Request".to_string(),
            category: TemplateCategory::Meeting,
            subject: "Meeting Request: {{topic}}".to_string(),
            body: r#"Hi {{name}},

I hope this message finds you well. I would like to schedule a meeting to discuss {{topic}}.

Would you be available {{time_suggestion}}? Please let me know what works best for your schedule.

Best regards"#.to_string(),
            variables: vec!["name".to_string(), "topic".to_string(), "time_suggestion".to_string()],
        });

        templates.insert("follow_up".to_string(), EmailTemplate {
            id: "follow_up".to_string(),
            name: "Follow Up".to_string(),
            category: TemplateCategory::FollowUp,
            subject: "Following Up: {{topic}}".to_string(),
            body: r#"Hi {{name}},

I wanted to follow up on {{topic}} from our previous conversation.

{{additional_context}}

Please let me know if you have any updates or if there's anything else you need from me.

Best regards"#.to_string(),
            variables: vec!["name".to_string(), "topic".to_string(), "additional_context".to_string()],
        });

        templates.insert("thank_you".to_string(), EmailTemplate {
            id: "thank_you".to_string(),
            name: "Thank You".to_string(),
            category: TemplateCategory::ThankYou,
            subject: "Thank You".to_string(),
            body: r#"Hi {{name}},

Thank you so much for {{reason}}. I really appreciate {{specific_detail}}.

{{next_steps}}

Best regards"#.to_string(),
            variables: vec!["name".to_string(), "reason".to_string(), "specific_detail".to_string(), "next_steps".to_string()],
        });

        templates.insert("introduction".to_string(), EmailTemplate {
            id: "introduction".to_string(),
            name: "Introduction".to_string(),
            category: TemplateCategory::Introduction,
            subject: "Introduction: {{your_name}} - {{context}}".to_string(),
            body: r#"Hi {{name}},

My name is {{your_name}}, and I'm reaching out because {{reason}}.

{{background}}

I would love to {{call_to_action}}.

Looking forward to connecting.

Best regards"#.to_string(),
            variables: vec!["name".to_string(), "your_name".to_string(), "reason".to_string(), "background".to_string(), "call_to_action".to_string(), "context".to_string()],
        });

        Self { templates }
    }

    fn get_templates(&self, category: TemplateCategory) -> Result<Vec<EmailTemplate>, CopilotError> {
        Ok(self.templates
            .values()
            .filter(|t| t.category == category || matches!(category, TemplateCategory::All))
            .cloned()
            .collect())
    }

    fn apply(&self, template_id: &str, variables: &HashMap<String, String>) -> Result<String, CopilotError> {
        let template = self.templates.get(template_id)
            .ok_or_else(|| CopilotError::InvalidRequest("Template not found".to_string()))?;

        let mut result = template.body.clone();
        for (key, value) in variables {
            result = result.replace(&format!("{{{{{}}}}}", key), value);
        }

        Ok(result)
    }
}

// ============================================================================
// Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposeContext {
    pub to: Option<String>,
    pub cc: Option<String>,
    pub subject: Option<String>,
    pub current_text: String,
    pub cursor_position: usize,
    pub reply_to_email_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Suggestion {
    pub text: String,
    pub suggestion_type: SuggestionType,
    pub confidence: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SuggestionType {
    PhraseCompletion,
    SentenceCompletion,
    SmartSuggestion,
    Template,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ImprovementStyle {
    MoreFormal,
    MoreCasual,
    Shorter,
    Longer,
    FixGrammar,
    Clearer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartReply {
    pub text: String,
    pub sentiment: ReplySentiment,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ReplySentiment {
    Positive,
    Neutral,
    Negative,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailTemplate {
    pub id: String,
    pub name: String,
    pub category: TemplateCategory,
    pub subject: String,
    pub body: String,
    pub variables: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TemplateCategory {
    All,
    Meeting,
    FollowUp,
    ThankYou,
    Introduction,
    Request,
    Announcement,
    Custom,
}

#[derive(Debug, sqlx::FromRow)]
struct EmailRow {
    subject: String,
    from_address: String,
    body_text: String,
}
