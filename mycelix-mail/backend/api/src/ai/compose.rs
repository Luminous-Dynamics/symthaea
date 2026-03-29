// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! AI Compose Suggestions & Smart Replies
//!
//! Generates contextual email suggestions and smart reply options.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartReply {
    pub id: String,
    pub text: String,
    pub tone: ReplyTone,
    pub intent: ReplyIntent,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReplyTone {
    Professional,
    Friendly,
    Casual,
    Formal,
    Apologetic,
    Grateful,
    Urgent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReplyIntent {
    Confirm,
    Decline,
    RequestInfo,
    ProvideInfo,
    Acknowledge,
    FollowUp,
    Apologize,
    ThankYou,
    Schedule,
    Delegate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposeSuggestion {
    pub suggestion_type: SuggestionType,
    pub text: String,
    pub position: InsertPosition,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuggestionType {
    SubjectLine,
    Opening,
    Body,
    Closing,
    Signature,
    Autocomplete,
    GrammarFix,
    ToneAdjustment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InsertPosition {
    Replace { start: usize, end: usize },
    Insert { position: usize },
    Append,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailContext {
    pub original_subject: Option<String>,
    pub original_body: Option<String>,
    pub original_sender: Option<String>,
    pub thread_messages: Vec<ThreadMessage>,
    pub recipient_relationship: RecipientRelationship,
    pub user_writing_style: WritingStyle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadMessage {
    pub from: String,
    pub body: String,
    pub date: String,
    pub is_from_user: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RecipientRelationship {
    Colleague,
    Manager,
    DirectReport,
    Client,
    Vendor,
    Friend,
    Acquaintance,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WritingStyle {
    pub formality: f32,       // 0 = casual, 1 = formal
    pub verbosity: f32,       // 0 = brief, 1 = detailed
    pub emoji_usage: f32,     // 0 = never, 1 = frequent
    pub greeting_style: GreetingStyle,
    pub closing_style: ClosingStyle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GreetingStyle {
    None,
    Simple,      // "Hi John,"
    Formal,      // "Dear Mr. Smith,"
    Friendly,    // "Hey John!"
    Professional, // "Hello John,"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClosingStyle {
    None,
    Simple,      // "Thanks,"
    Formal,      // "Sincerely,"
    Friendly,    // "Cheers,"
    Professional, // "Best regards,"
}

impl Default for WritingStyle {
    fn default() -> Self {
        Self {
            formality: 0.6,
            verbosity: 0.5,
            emoji_usage: 0.1,
            greeting_style: GreetingStyle::Professional,
            closing_style: ClosingStyle::Professional,
        }
    }
}

// ============================================================================
// Smart Reply Generator
// ============================================================================

pub struct SmartReplyGenerator {
    templates: HashMap<String, Vec<ReplyTemplate>>,
}

#[derive(Debug, Clone)]
struct ReplyTemplate {
    pattern: String,
    replies: Vec<(String, ReplyTone, ReplyIntent)>,
}

impl SmartReplyGenerator {
    pub fn new() -> Self {
        let mut templates = HashMap::new();

        // Meeting-related patterns
        templates.insert("meeting".to_string(), vec![
            ReplyTemplate {
                pattern: "can we meet|schedule a meeting|set up a call".into(),
                replies: vec![
                    ("Sure, I'm available. When works best for you?".into(), ReplyTone::Professional, ReplyIntent::Confirm),
                    ("Yes, let me check my calendar and get back to you.".into(), ReplyTone::Professional, ReplyIntent::Confirm),
                    ("I'd be happy to meet. How about tomorrow afternoon?".into(), ReplyTone::Friendly, ReplyIntent::Schedule),
                    ("Unfortunately I'm not available this week. Can we look at next week?".into(), ReplyTone::Professional, ReplyIntent::Decline),
                ],
            },
            ReplyTemplate {
                pattern: "meeting invite|calendar invite".into(),
                replies: vec![
                    ("I'll be there!".into(), ReplyTone::Friendly, ReplyIntent::Confirm),
                    ("Thank you for the invite. I've accepted.".into(), ReplyTone::Professional, ReplyIntent::Confirm),
                    ("I have a conflict at that time. Could we reschedule?".into(), ReplyTone::Professional, ReplyIntent::Decline),
                ],
            },
        ]);

        // Request-related patterns
        templates.insert("request".to_string(), vec![
            ReplyTemplate {
                pattern: "can you|could you|would you|please".into(),
                replies: vec![
                    ("Sure, I'll take care of it.".into(), ReplyTone::Professional, ReplyIntent::Confirm),
                    ("I'll get this done today.".into(), ReplyTone::Professional, ReplyIntent::Confirm),
                    ("Of course! I'll have it ready by end of day.".into(), ReplyTone::Friendly, ReplyIntent::Confirm),
                    ("I need a bit more information to proceed. Could you clarify...".into(), ReplyTone::Professional, ReplyIntent::RequestInfo),
                ],
            },
        ]);

        // Thank you patterns
        templates.insert("thanks".to_string(), vec![
            ReplyTemplate {
                pattern: "thank you|thanks|appreciate".into(),
                replies: vec![
                    ("You're welcome!".into(), ReplyTone::Friendly, ReplyIntent::Acknowledge),
                    ("Happy to help!".into(), ReplyTone::Friendly, ReplyIntent::Acknowledge),
                    ("Glad I could assist.".into(), ReplyTone::Professional, ReplyIntent::Acknowledge),
                    ("No problem at all!".into(), ReplyTone::Casual, ReplyIntent::Acknowledge),
                ],
            },
        ]);

        // Question patterns
        templates.insert("question".to_string(), vec![
            ReplyTemplate {
                pattern: "\\?|what|how|when|where|why|who".into(),
                replies: vec![
                    ("Good question. Let me look into this and get back to you.".into(), ReplyTone::Professional, ReplyIntent::FollowUp),
                    ("I'll find out and follow up shortly.".into(), ReplyTone::Professional, ReplyIntent::FollowUp),
                    ("I can answer that! Here's what I know...".into(), ReplyTone::Friendly, ReplyIntent::ProvideInfo),
                ],
            },
        ]);

        // Status update patterns
        templates.insert("status".to_string(), vec![
            ReplyTemplate {
                pattern: "status|update|progress|how.*(going|coming)".into(),
                replies: vec![
                    ("Thanks for checking in. Here's where we are...".into(), ReplyTone::Professional, ReplyIntent::ProvideInfo),
                    ("Good news - we're on track to finish by the deadline.".into(), ReplyTone::Professional, ReplyIntent::ProvideInfo),
                    ("I'll send over a detailed update by end of day.".into(), ReplyTone::Professional, ReplyIntent::FollowUp),
                ],
            },
        ]);

        // Confirmation patterns
        templates.insert("confirm".to_string(), vec![
            ReplyTemplate {
                pattern: "confirm|verify|correct|right".into(),
                replies: vec![
                    ("Yes, that's correct!".into(), ReplyTone::Professional, ReplyIntent::Confirm),
                    ("Confirmed!".into(), ReplyTone::Casual, ReplyIntent::Confirm),
                    ("That's right. Let me know if you need anything else.".into(), ReplyTone::Professional, ReplyIntent::Confirm),
                ],
            },
        ]);

        // Apology/issue patterns
        templates.insert("issue".to_string(), vec![
            ReplyTemplate {
                pattern: "sorry|apologize|issue|problem|bug|error".into(),
                replies: vec![
                    ("No worries at all!".into(), ReplyTone::Friendly, ReplyIntent::Acknowledge),
                    ("Thanks for letting me know. I'll look into it right away.".into(), ReplyTone::Professional, ReplyIntent::Acknowledge),
                    ("I understand. Let's figure out how to resolve this.".into(), ReplyTone::Professional, ReplyIntent::FollowUp),
                ],
            },
        ]);

        Self { templates }
    }

    pub fn generate(&self, context: &EmailContext) -> Vec<SmartReply> {
        let mut replies = Vec::new();
        let body = context.original_body.as_deref().unwrap_or("").to_lowercase();

        // Check each template category
        for (category, category_templates) in &self.templates {
            for template in category_templates {
                if self.matches_pattern(&body, &template.pattern) {
                    for (text, tone, intent) in &template.replies {
                        // Adjust text based on relationship
                        let adjusted_text = self.adjust_for_relationship(text, &context.recipient_relationship);

                        replies.push(SmartReply {
                            id: format!("{}-{}", category, replies.len()),
                            text: adjusted_text,
                            tone: tone.clone(),
                            intent: intent.clone(),
                            confidence: 0.8,
                        });
                    }
                    break; // Only use first matching template per category
                }
            }
        }

        // Sort by confidence and limit
        replies.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        replies.truncate(5);

        // If no matches, provide generic replies
        if replies.is_empty() {
            replies = self.generate_generic_replies(context);
        }

        replies
    }

    fn matches_pattern(&self, text: &str, pattern: &str) -> bool {
        let patterns: Vec<&str> = pattern.split('|').collect();
        patterns.iter().any(|p| text.contains(p))
    }

    fn adjust_for_relationship(&self, text: &str, relationship: &RecipientRelationship) -> String {
        match relationship {
            RecipientRelationship::Manager | RecipientRelationship::Client => {
                // More formal
                text.replace("Sure,", "Certainly,")
                    .replace("No problem", "Of course")
                    .replace("!", ".")
            }
            RecipientRelationship::Friend => {
                // More casual
                text.replace("Certainly", "Sure")
                    .replace("I would be happy to", "Happy to")
            }
            _ => text.to_string(),
        }
    }

    fn generate_generic_replies(&self, context: &EmailContext) -> Vec<SmartReply> {
        vec![
            SmartReply {
                id: "generic-1".into(),
                text: "Thanks for your email. I'll review and get back to you shortly.".into(),
                tone: ReplyTone::Professional,
                intent: ReplyIntent::Acknowledge,
                confidence: 0.6,
            },
            SmartReply {
                id: "generic-2".into(),
                text: "Got it, thanks!".into(),
                tone: ReplyTone::Casual,
                intent: ReplyIntent::Acknowledge,
                confidence: 0.5,
            },
            SmartReply {
                id: "generic-3".into(),
                text: "I'll look into this and follow up.".into(),
                tone: ReplyTone::Professional,
                intent: ReplyIntent::FollowUp,
                confidence: 0.5,
            },
        ]
    }
}

// ============================================================================
// Compose Suggestion Generator
// ============================================================================

pub struct ComposeSuggestionGenerator {
    common_phrases: HashMap<String, Vec<String>>,
    greetings: Vec<(GreetingStyle, Vec<String>)>,
    closings: Vec<(ClosingStyle, Vec<String>)>,
}

impl ComposeSuggestionGenerator {
    pub fn new() -> Self {
        let mut common_phrases = HashMap::new();

        common_phrases.insert("follow_up".into(), vec![
            "Following up on our previous conversation...".into(),
            "Just checking in on...".into(),
            "I wanted to follow up regarding...".into(),
            "Circling back on...".into(),
        ]);

        common_phrases.insert("introduction".into(), vec![
            "I hope this email finds you well.".into(),
            "I hope you're having a great week.".into(),
            "I wanted to reach out about...".into(),
            "I'm writing to inquire about...".into(),
        ]);

        common_phrases.insert("request".into(), vec![
            "Would you be able to...".into(),
            "I was wondering if you could...".into(),
            "Could you please...".into(),
            "I'd appreciate it if you could...".into(),
        ]);

        common_phrases.insert("availability".into(), vec![
            "Please let me know what works best for you.".into(),
            "I'm flexible and can adjust to your schedule.".into(),
            "Let me know your availability.".into(),
            "What times work on your end?".into(),
        ]);

        let greetings = vec![
            (GreetingStyle::Simple, vec!["Hi".into(), "Hello".into()]),
            (GreetingStyle::Formal, vec!["Dear".into(), "Good morning".into(), "Good afternoon".into()]),
            (GreetingStyle::Friendly, vec!["Hey".into(), "Hi there".into()]),
            (GreetingStyle::Professional, vec!["Hello".into(), "Hi".into(), "Good day".into()]),
        ];

        let closings = vec![
            (ClosingStyle::Simple, vec!["Thanks".into(), "Thank you".into()]),
            (ClosingStyle::Formal, vec!["Sincerely".into(), "Respectfully".into(), "Kind regards".into()]),
            (ClosingStyle::Friendly, vec!["Cheers".into(), "Take care".into(), "All the best".into()]),
            (ClosingStyle::Professional, vec!["Best regards".into(), "Best".into(), "Regards".into()]),
        ];

        Self {
            common_phrases,
            greetings,
            closings,
        }
    }

    pub fn suggest_subject(&self, context: &EmailContext) -> Vec<ComposeSuggestion> {
        let mut suggestions = Vec::new();

        if let Some(original) = &context.original_subject {
            // Reply subject
            if !original.starts_with("Re:") {
                suggestions.push(ComposeSuggestion {
                    suggestion_type: SuggestionType::SubjectLine,
                    text: format!("Re: {}", original),
                    position: InsertPosition::Replace { start: 0, end: 0 },
                    confidence: 0.9,
                });
            }

            // Forward subject
            suggestions.push(ComposeSuggestion {
                suggestion_type: SuggestionType::SubjectLine,
                text: format!("Fwd: {}", original.trim_start_matches("Re: ")),
                position: InsertPosition::Replace { start: 0, end: 0 },
                confidence: 0.7,
            });
        }

        suggestions
    }

    pub fn suggest_opening(&self, context: &EmailContext, recipient_name: Option<&str>) -> Vec<ComposeSuggestion> {
        let mut suggestions = Vec::new();

        // Find appropriate greetings for style
        for (style, greetings) in &self.greetings {
            if *style == context.user_writing_style.greeting_style {
                for greeting in greetings {
                    let text = if let Some(name) = recipient_name {
                        format!("{} {},\n\n", greeting, name)
                    } else {
                        format!("{},\n\n", greeting)
                    };

                    suggestions.push(ComposeSuggestion {
                        suggestion_type: SuggestionType::Opening,
                        text,
                        position: InsertPosition::Insert { position: 0 },
                        confidence: 0.8,
                    });
                }
            }
        }

        // Add intro phrases
        if let Some(phrases) = self.common_phrases.get("introduction") {
            for phrase in phrases.iter().take(2) {
                suggestions.push(ComposeSuggestion {
                    suggestion_type: SuggestionType::Opening,
                    text: format!("{}\n\n", phrase),
                    position: InsertPosition::Append,
                    confidence: 0.6,
                });
            }
        }

        suggestions
    }

    pub fn suggest_closing(&self, context: &EmailContext, user_name: &str) -> Vec<ComposeSuggestion> {
        let mut suggestions = Vec::new();

        for (style, closings) in &self.closings {
            if *style == context.user_writing_style.closing_style {
                for closing in closings {
                    suggestions.push(ComposeSuggestion {
                        suggestion_type: SuggestionType::Closing,
                        text: format!("\n\n{},\n{}", closing, user_name),
                        position: InsertPosition::Append,
                        confidence: 0.8,
                    });
                }
            }
        }

        suggestions
    }

    pub fn autocomplete(&self, current_text: &str, cursor_position: usize) -> Vec<ComposeSuggestion> {
        let mut suggestions = Vec::new();

        // Get text before cursor
        let text_before = &current_text[..cursor_position.min(current_text.len())];
        let last_words: Vec<&str> = text_before.split_whitespace().rev().take(3).collect();

        // Check for common phrase triggers
        let trigger = last_words.iter().rev().cloned().collect::<Vec<_>>().join(" ").to_lowercase();

        // Phrase completions
        let completions: Vec<(&str, &str)> = vec![
            ("following up", " on our previous conversation"),
            ("i wanted to", " reach out regarding"),
            ("please let me", " know if you have any questions"),
            ("i look forward", " to hearing from you"),
            ("thank you for", " your time"),
            ("as per our", " discussion"),
            ("attached please", " find"),
            ("don't hesitate to", " reach out if you need anything"),
            ("i appreciate your", " help with this"),
            ("at your earliest", " convenience"),
        ];

        for (trigger_phrase, completion) in completions {
            if trigger.ends_with(trigger_phrase) {
                suggestions.push(ComposeSuggestion {
                    suggestion_type: SuggestionType::Autocomplete,
                    text: completion.to_string(),
                    position: InsertPosition::Insert { position: cursor_position },
                    confidence: 0.85,
                });
            }
        }

        suggestions
    }
}

// ============================================================================
// Combined Compose Assistant
// ============================================================================

pub struct ComposeAssistant {
    reply_generator: SmartReplyGenerator,
    suggestion_generator: ComposeSuggestionGenerator,
}

impl ComposeAssistant {
    pub fn new() -> Self {
        Self {
            reply_generator: SmartReplyGenerator::new(),
            suggestion_generator: ComposeSuggestionGenerator::new(),
        }
    }

    pub fn get_smart_replies(&self, context: &EmailContext) -> Vec<SmartReply> {
        self.reply_generator.generate(context)
    }

    pub fn get_suggestions(
        &self,
        context: &EmailContext,
        current_text: &str,
        cursor_position: usize,
        recipient_name: Option<&str>,
        user_name: &str,
    ) -> Vec<ComposeSuggestion> {
        let mut suggestions = Vec::new();

        // Subject suggestions if empty
        if current_text.is_empty() {
            suggestions.extend(self.suggestion_generator.suggest_subject(context));
            suggestions.extend(self.suggestion_generator.suggest_opening(context, recipient_name));
        }

        // Autocomplete
        if !current_text.is_empty() {
            suggestions.extend(self.suggestion_generator.autocomplete(current_text, cursor_position));
        }

        // Closing suggestions near end
        if current_text.len() > 100 && !current_text.contains("regards") && !current_text.contains("thanks") {
            suggestions.extend(self.suggestion_generator.suggest_closing(context, user_name));
        }

        suggestions
    }
}

impl Default for ComposeAssistant {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for SmartReplyGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ComposeSuggestionGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smart_reply_generation() {
        let generator = SmartReplyGenerator::new();
        let context = EmailContext {
            original_subject: Some("Meeting Request".into()),
            original_body: Some("Can we schedule a meeting to discuss the project?".into()),
            original_sender: Some("colleague@company.com".into()),
            thread_messages: vec![],
            recipient_relationship: RecipientRelationship::Colleague,
            user_writing_style: WritingStyle::default(),
        };

        let replies = generator.generate(&context);
        assert!(!replies.is_empty());
        assert!(replies.iter().any(|r| r.intent == ReplyIntent::Confirm || r.intent == ReplyIntent::Schedule));
    }

    #[test]
    fn test_autocomplete() {
        let generator = ComposeSuggestionGenerator::new();
        let text = "Thank you for";
        let suggestions = generator.autocomplete(text, text.len());
        assert!(!suggestions.is_empty());
    }
}
