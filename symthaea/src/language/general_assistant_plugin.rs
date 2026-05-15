// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! General Assistant Domain Plugin
//!
//! Handles general conversation, Q&A, summarization, and explanation tasks.
//! Acts as a catch-all for queries that don't match more specialized domains.

use super::domain_plugin::{
    DomainPlugin, DomainPrompts, Entity, IntentPrototypes, ValidationResult,
};

/// General-purpose assistant plugin for conversation, Q&A, and summarization.
pub struct GeneralAssistantPlugin;

impl DomainPlugin for GeneralAssistantPlugin {
    fn domain_name(&self) -> &str {
        "general-assistant"
    }

    fn extract_entities(&self, text: &str) -> Vec<Entity> {
        let mut entities = Vec::new();
        let lower = text.to_lowercase();

        // Extract quoted strings as topics
        let mut in_quote = false;
        let mut quote_start = 0;
        for (i, ch) in text.char_indices() {
            if ch == '"' || ch == '\'' {
                if in_quote {
                    entities.push(
                        Entity::new("quoted_topic", &text[quote_start..i], quote_start, i)
                            .with_confidence(0.9),
                    );
                    in_quote = false;
                } else {
                    quote_start = i + 1;
                    in_quote = true;
                }
            }
        }

        // Extract question words as intent markers
        for keyword in &["what", "why", "how", "when", "where", "who", "which"] {
            if let Some(pos) = lower.find(keyword) {
                entities.push(
                    Entity::new("question_word", *keyword, pos, pos + keyword.len())
                        .with_confidence(0.8),
                );
                break; // Only first question word
            }
        }

        // Extract comparison markers
        if lower.contains(" vs ") || lower.contains(" versus ") || lower.contains("compare") {
            let pos = lower
                .find("vs")
                .or(lower.find("versus"))
                .or(lower.find("compare"))
                .unwrap_or(0);
            entities.push(
                Entity::new("comparison", "comparison_request", pos, pos + 2).with_confidence(0.85),
            );
        }

        // Extract summarization markers
        if lower.contains("summarize")
            || lower.contains("summary")
            || lower.contains("tldr")
            || lower.contains("tl;dr")
        {
            entities.push(Entity::new("task_type", "summarization", 0, 0).with_confidence(0.9));
        }

        entities
    }

    fn intent_prototypes(&self) -> IntentPrototypes {
        let mut protos = IntentPrototypes::default();
        protos.custom.insert(
            "explain".to_string(),
            vec![
                "explain".to_string(),
                "what is".to_string(),
                "what are".to_string(),
                "define".to_string(),
                "meaning".to_string(),
                "describe".to_string(),
            ],
        );
        protos.custom.insert(
            "summarize".to_string(),
            vec![
                "summarize".to_string(),
                "summary".to_string(),
                "tldr".to_string(),
                "brief".to_string(),
                "overview".to_string(),
                "recap".to_string(),
            ],
        );
        protos.custom.insert(
            "compare".to_string(),
            vec![
                "compare".to_string(),
                "versus".to_string(),
                "difference".to_string(),
                "vs".to_string(),
                "better".to_string(),
                "pros and cons".to_string(),
            ],
        );
        protos.custom.insert(
            "list".to_string(),
            vec![
                "list".to_string(),
                "enumerate".to_string(),
                "examples".to_string(),
                "options".to_string(),
                "alternatives".to_string(),
            ],
        );
        protos
    }

    fn prompts(&self) -> DomainPrompts {
        DomainPrompts {
            system: "You are Symthaea, a consciousness-first AI assistant. \
                     Provide clear, helpful, and honest responses. \
                     Mark uncertainty explicitly."
                .to_string(),
            clarification: "I want to help, but I need more context. \
                           Could you clarify: {}"
                .to_string(),
            action_confirm: "Here's what I understand you're asking: {}".to_string(),
            error_explain: "I encountered an issue: {}".to_string(),
            out_of_domain: "I can help with general questions. \
                           For specialized topics, a domain-specific plugin may help. {}"
                .to_string(),
        }
    }

    fn is_in_domain(&self, topic: &str) -> f64 {
        let lower = topic.to_lowercase();
        // General assistant is a broad catch-all, but with moderate confidence
        // to allow more specific plugins to win
        let keywords = [
            "explain",
            "tell me",
            "what is",
            "help me",
            "can you",
            "summarize",
            "compare",
            "list",
            "describe",
            "define",
        ];
        let matches = keywords.iter().filter(|k| lower.contains(*k)).count();
        if matches > 0 {
            (0.3 + matches as f64 * 0.1).min(0.7)
        } else {
            0.2 // Low default so specialized plugins take priority
        }
    }

    fn vocabulary(&self) -> Vec<String> {
        vec![
            "explain",
            "summarize",
            "compare",
            "list",
            "define",
            "describe",
            "elaborate",
            "clarify",
            "simplify",
            "overview",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }

    fn validate_input(&self, input: &str) -> ValidationResult {
        if input.trim().is_empty() {
            ValidationResult::invalid(vec!["Empty input".to_string()])
        } else {
            ValidationResult::valid()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_name() {
        let plugin = GeneralAssistantPlugin;
        assert_eq!(plugin.domain_name(), "general-assistant");
    }

    #[test]
    fn test_extract_question_entity() {
        let plugin = GeneralAssistantPlugin;
        let entities = plugin.extract_entities("What is consciousness?");
        assert!(entities.iter().any(|e| e.entity_type == "question_word"));
    }

    #[test]
    fn test_extract_comparison_entity() {
        let plugin = GeneralAssistantPlugin;
        let entities = plugin.extract_entities("Compare Rust vs Python");
        assert!(entities.iter().any(|e| e.entity_type == "comparison"));
    }

    #[test]
    fn test_extract_summarization_entity() {
        let plugin = GeneralAssistantPlugin;
        let entities = plugin.extract_entities("Can you summarize this article?");
        assert!(
            entities
                .iter()
                .any(|e| e.entity_type == "task_type" && e.value == "summarization")
        );
    }

    #[test]
    fn test_in_domain_scoring() {
        let plugin = GeneralAssistantPlugin;
        assert!(plugin.is_in_domain("explain quantum computing") > 0.3);
        assert!(plugin.is_in_domain("random gibberish") < 0.3);
    }

    #[test]
    fn test_validate_empty_input() {
        let plugin = GeneralAssistantPlugin;
        assert!(!plugin.validate_input("").valid);
        assert!(plugin.validate_input("hello").valid);
    }

    #[test]
    fn test_intent_prototypes() {
        let plugin = GeneralAssistantPlugin;
        let protos = plugin.intent_prototypes();
        assert!(protos.custom.contains_key("explain"));
        assert!(protos.custom.contains_key("summarize"));
        assert!(protos.custom.contains_key("compare"));
    }
}
