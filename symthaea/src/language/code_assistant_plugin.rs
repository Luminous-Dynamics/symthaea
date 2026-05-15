// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Code Assistant Domain Plugin
//!
//! Multi-language code analysis, debugging, refactoring, and generation.
//! Extracts functions, classes, errors, and stack traces from user input.

use super::domain_plugin::{
    DomainPlugin, DomainPrompts, Entity, ErrorDiagnosis, IntentPrototypes, RiskLevel,
    ValidationResult,
};
use std::collections::HashMap;

/// Code assistant plugin for multi-language code analysis and generation.
pub struct CodeAssistantPlugin;

impl DomainPlugin for CodeAssistantPlugin {
    fn domain_name(&self) -> &str {
        "code-assistant"
    }

    fn extract_entities(&self, text: &str) -> Vec<Entity> {
        let mut entities = Vec::new();

        // Extract language mentions
        let languages = [
            "rust",
            "python",
            "javascript",
            "typescript",
            "go",
            "java",
            "c++",
            "c#",
            "ruby",
            "swift",
            "kotlin",
            "haskell",
            "nix",
            "bash",
            "shell",
            "sql",
            "html",
            "css",
        ];
        let lower = text.to_lowercase();
        for lang in &languages {
            if let Some(pos) = lower.find(lang) {
                entities.push(
                    Entity::new("language", *lang, pos, pos + lang.len()).with_confidence(0.9),
                );
            }
        }

        // Extract function/method names (word followed by parentheses)
        let re_func =
            regex::Regex::new(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(").expect("valid regex literal");
        for cap in re_func.captures_iter(text) {
            if let Some(m) = cap.get(1) {
                let name = m.as_str();
                // Skip common words that happen to precede parens
                if ![
                    "if", "for", "while", "match", "return", "fn", "def", "function", "class",
                ]
                .contains(&name)
                {
                    entities.push(
                        Entity::new("function", name, m.start(), m.end()).with_confidence(0.7),
                    );
                }
            }
        }

        // Extract error patterns
        if lower.contains("error") || lower.contains("exception") || lower.contains("panic") {
            // Try to extract error type (e.g., "TypeError: ..." or "error[E0308]")
            let re_err = regex::Regex::new(
                r"(?i)((?:error|exception|panic|TypeError|ValueError|RuntimeError)\S*)",
            )
            .expect("valid regex literal");
            for cap in re_err.captures_iter(text) {
                if let Some(m) = cap.get(1) {
                    entities.push(
                        Entity::new("error_type", m.as_str(), m.start(), m.end())
                            .with_confidence(0.85),
                    );
                }
            }
        }

        // Extract file paths
        let re_path = regex::Regex::new(r"(?:^|\s)((?:[a-zA-Z_./][a-zA-Z0-9_./\-]*)?\.(?:rs|py|js|ts|go|java|cpp|c|h|rb|swift|kt|hs|nix|sh|sql))").expect("valid regex literal");
        for cap in re_path.captures_iter(text) {
            if let Some(m) = cap.get(1) {
                entities.push(
                    Entity::new("file_path", m.as_str(), m.start(), m.end()).with_confidence(0.8),
                );
            }
        }

        // Extract line numbers (e.g., "line 42", "L42", ":42:")
        let re_line = regex::Regex::new(r"(?:line\s+|:)(\d+)").expect("valid regex literal");
        for cap in re_line.captures_iter(text) {
            if let Some(m) = cap.get(1) {
                entities.push(
                    Entity::new("line_number", m.as_str(), m.start(), m.end())
                        .with_confidence(0.75),
                );
            }
        }

        entities
    }

    fn intent_prototypes(&self) -> IntentPrototypes {
        let mut custom = HashMap::new();
        custom.insert(
            "debug".to_string(),
            vec![
                "debug".to_string(),
                "fix".to_string(),
                "error".to_string(),
                "bug".to_string(),
                "broken".to_string(),
                "crash".to_string(),
                "not working".to_string(),
                "fails".to_string(),
            ],
        );
        custom.insert(
            "refactor".to_string(),
            vec![
                "refactor".to_string(),
                "restructure".to_string(),
                "clean up".to_string(),
                "reorganize".to_string(),
                "simplify".to_string(),
                "extract".to_string(),
            ],
        );
        custom.insert(
            "explain_code".to_string(),
            vec![
                "explain".to_string(),
                "what does".to_string(),
                "how does".to_string(),
                "understand".to_string(),
                "walk through".to_string(),
                "trace".to_string(),
            ],
        );
        custom.insert(
            "generate".to_string(),
            vec![
                "generate".to_string(),
                "create".to_string(),
                "write".to_string(),
                "implement".to_string(),
                "build".to_string(),
                "code".to_string(),
            ],
        );
        IntentPrototypes {
            command: vec![
                "fix",
                "debug",
                "refactor",
                "optimize",
                "write",
                "generate",
                "implement",
                "add",
                "remove",
                "rename",
                "extract",
                "inline",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            custom,
            ..Default::default()
        }
    }

    fn prompts(&self) -> DomainPrompts {
        DomainPrompts {
            system: "You are Symthaea, a consciousness-first AI with deep programming expertise. \
                     Provide precise, idiomatic code. Explain your reasoning. \
                     Prioritize correctness and clarity over cleverness."
                .to_string(),
            clarification: "I need more context about your code. \
                           Which language and what's the expected behavior? {}"
                .to_string(),
            action_confirm: "I'll help with your code: {}".to_string(),
            error_explain: "I found an issue in your code: {}".to_string(),
            out_of_domain: "This seems to be a non-coding question. {}".to_string(),
        }
    }

    fn diagnose_error(&self, error_text: &str) -> Option<ErrorDiagnosis> {
        let lower = error_text.to_lowercase();

        // Rust-specific errors
        if lower.contains("error[e0") {
            let suggestion = if lower.contains("e0308") {
                Some("Type mismatch: check that the expected and found types align.".to_string())
            } else if lower.contains("e0382") {
                Some("Use after move: consider cloning or borrowing instead.".to_string())
            } else if lower.contains("e0502") || lower.contains("e0499") {
                Some("Borrow checker error: review mutable vs immutable references.".to_string())
            } else {
                None
            };
            return Some(ErrorDiagnosis {
                category: "compilation".to_string(),
                error_type: "rust_compiler_error".to_string(),
                description: "Rust compiler error detected.".to_string(),
                suggestion,
                confidence: 0.9,
                risk_level: RiskLevel::Medium,
                location: None,
            });
        }

        // Python errors
        if lower.contains("traceback")
            || lower.contains("typeerror")
            || lower.contains("valueerror")
        {
            return Some(ErrorDiagnosis {
                category: "runtime".to_string(),
                error_type: "python_exception".to_string(),
                description: "Python runtime exception detected.".to_string(),
                suggestion: Some(
                    "Check the traceback for the source file and line number.".to_string(),
                ),
                confidence: 0.85,
                risk_level: RiskLevel::Medium,
                location: None,
            });
        }

        // JavaScript/TypeScript errors
        if lower.contains("uncaught")
            || lower.contains("undefined is not")
            || lower.contains("cannot read property")
        {
            return Some(ErrorDiagnosis {
                category: "runtime".to_string(),
                error_type: "js_runtime_error".to_string(),
                description: "JavaScript runtime error detected.".to_string(),
                suggestion: Some(
                    "Check for null/undefined values before property access.".to_string(),
                ),
                confidence: 0.8,
                risk_level: RiskLevel::Medium,
                location: None,
            });
        }

        None
    }

    fn is_in_domain(&self, topic: &str) -> f64 {
        let lower = topic.to_lowercase();
        let code_keywords = [
            "code",
            "function",
            "variable",
            "class",
            "method",
            "compile",
            "error",
            "bug",
            "debug",
            "refactor",
            "implement",
            "api",
            "algorithm",
            "data structure",
            "test",
            "unit test",
            "syntax",
            "rust",
            "python",
            "javascript",
            "typescript",
        ];
        let matches = code_keywords.iter().filter(|k| lower.contains(*k)).count();
        // Check for code-like patterns (backticks, brackets, semicolons)
        let code_chars = text_has_code_patterns(&lower);

        let keyword_score = (matches as f64 * 0.15).min(0.7);
        let pattern_score = if code_chars { 0.2 } else { 0.0 };
        (keyword_score + pattern_score).min(0.95)
    }

    fn vocabulary(&self) -> Vec<String> {
        vec![
            "function",
            "variable",
            "class",
            "method",
            "interface",
            "struct",
            "enum",
            "trait",
            "module",
            "package",
            "import",
            "return",
            "async",
            "await",
            "error",
            "exception",
            "debug",
            "compile",
            "runtime",
            "stack trace",
            "breakpoint",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }

    fn validate_input(&self, input: &str) -> ValidationResult {
        if input.trim().is_empty() {
            return ValidationResult::invalid(vec!["Empty input".to_string()]);
        }
        let mut result = ValidationResult::valid();
        if input.len() > 50000 {
            result
                .warnings
                .push("Very long input may exceed context limits.".to_string());
        }
        result
    }
}

/// Check if text contains code-like patterns.
fn text_has_code_patterns(text: &str) -> bool {
    text.contains("```")
        || text.contains("fn ")
        || text.contains("def ")
        || text.contains("class ")
        || text.contains("function ")
        || (text.contains('{') && text.contains('}'))
        || text.contains("->")
        || text.contains("=>")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_name() {
        let plugin = CodeAssistantPlugin;
        assert_eq!(plugin.domain_name(), "code-assistant");
    }

    #[test]
    fn test_extract_language_entity() {
        let plugin = CodeAssistantPlugin;
        let entities = plugin.extract_entities("How do I write a function in Rust?");
        assert!(
            entities
                .iter()
                .any(|e| e.entity_type == "language" && e.value == "rust")
        );
    }

    #[test]
    fn test_extract_function_entity() {
        let plugin = CodeAssistantPlugin;
        let entities = plugin.extract_entities("The calculate_phi() function is broken");
        assert!(
            entities
                .iter()
                .any(|e| e.entity_type == "function" && e.value == "calculate_phi")
        );
    }

    #[test]
    fn test_extract_error_entity() {
        let plugin = CodeAssistantPlugin;
        let entities = plugin.extract_entities("I get error[E0308]: mismatched types");
        assert!(entities.iter().any(|e| e.entity_type == "error_type"));
    }

    #[test]
    fn test_extract_file_path() {
        let plugin = CodeAssistantPlugin;
        let entities = plugin.extract_entities("The bug is in src/main.rs");
        assert!(entities.iter().any(|e| e.entity_type == "file_path"));
    }

    #[test]
    fn test_diagnose_rust_error() {
        let plugin = CodeAssistantPlugin;
        let diag = plugin.diagnose_error("error[E0308]: mismatched types");
        assert!(diag.is_some());
        assert_eq!(diag.unwrap().error_type, "rust_compiler_error");
    }

    #[test]
    fn test_diagnose_python_error() {
        let plugin = CodeAssistantPlugin;
        let diag = plugin.diagnose_error("Traceback (most recent call last): TypeError");
        assert!(diag.is_some());
        assert_eq!(diag.unwrap().error_type, "python_exception");
    }

    #[test]
    fn test_in_domain_scoring() {
        let plugin = CodeAssistantPlugin;
        assert!(plugin.is_in_domain("debug this rust function") > 0.3);
        assert!(plugin.is_in_domain("what's the weather") < 0.3);
    }

    #[test]
    fn test_code_pattern_detection() {
        assert!(text_has_code_patterns("fn main() { }"));
        assert!(text_has_code_patterns("def hello():"));
        assert!(!text_has_code_patterns("hello world"));
    }
}
