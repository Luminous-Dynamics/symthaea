// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Code Domain Plugin — Registers code understanding as a DomainPlugin
//!
//! This plugin integrates the code understanding pipeline into Symthaea's
//! existing domain plugin system. When the input looks like code (not just
//! discussion *about* code), this plugin activates and routes through
//! the HDC+CfC code understanding pipeline.
//!
//! # Domain Detection
//!
//! Supersedes `ProgrammingPlugin` for actual code inputs:
//! - ProgrammingPlugin: handles discussion about programming concepts
//! - CodeDomainPlugin: handles actual code snippets, generation requests

use super::domain_plugin::{
    DomainPlugin, DomainPrompts, Entity, IntentPrototypes, ValidationResult,
};

/// Code markers that indicate actual code (not just discussion about code)
const CODE_MARKERS: &[&str] = &[
    "```",
    "fn ",
    "def ",
    "class ",
    "struct ",
    "impl ",
    "pub fn",
    "let ",
    "const ",
    "import ",
    "from ",
    "use ",
    "mod ",
    "if __name__",
    "#[derive",
    "#![",
    "async fn",
    "pub struct",
    "pub enum",
    "trait ",
    "type ",
    "match ",
    "enum ",
];

/// Keywords that indicate a code generation/understanding request
const CODE_REQUEST_KEYWORDS: &[&str] = &[
    "write a function",
    "generate code",
    "write code",
    "implement",
    "create a function",
    "create a struct",
    "create a class",
    "write a test",
    "refactor",
    "code review",
    "what does this code",
    "explain this code",
    "debug this",
    "fix this code",
    "write a module",
    "generate a",
    "scaffold",
];

/// Plugin for code understanding and generation via HDC+CfC pipeline
pub struct CodeDomainPlugin;

impl DomainPlugin for CodeDomainPlugin {
    fn domain_name(&self) -> &str {
        "code"
    }

    fn extract_entities(&self, text: &str) -> Vec<Entity> {
        let mut entities = Vec::new();

        // Detect language
        if let Some((lang, start, end)) = detect_language_mention(text) {
            entities.push(Entity::new("language", lang, start, end).with_confidence(0.9));
        }

        // Detect code blocks
        let mut search_start = 0;
        while let Some(block_start) = text[search_start..].find("```") {
            let abs_start = search_start + block_start;
            let after_fence = abs_start + 3;
            if let Some(block_end) = text[after_fence..].find("```") {
                let abs_end = after_fence + block_end + 3;
                let block_content = &text[after_fence..after_fence + block_end];
                let first_line = block_content.lines().next().unwrap_or("").trim();

                // Language tag on the opening fence
                let lang_tag =
                    if !first_line.is_empty() && first_line.len() < 20 && !first_line.contains(' ')
                    {
                        first_line
                    } else {
                        "unknown"
                    };

                entities.push(
                    Entity::new("code_block", lang_tag, abs_start, abs_end)
                        .with_confidence(0.95)
                        .with_metadata("language", lang_tag),
                );
                search_start = abs_end;
            } else {
                break;
            }
        }

        // Detect function/type names mentioned with backticks
        let mut tick_start = 0;
        while let Some(start) = text[tick_start..].find('`') {
            let abs_start = tick_start + start;
            if abs_start + 1 >= text.len() {
                break;
            }
            // Skip triple backticks
            if text[abs_start..].starts_with("```") {
                tick_start = abs_start + 3;
                continue;
            }
            if let Some(end) = text[abs_start + 1..].find('`') {
                let abs_end = abs_start + 1 + end + 1;
                let name = &text[abs_start + 1..abs_start + 1 + end];
                if !name.is_empty() && name.len() < 80 && !name.contains('\n') {
                    entities.push(
                        Entity::new("code_reference", name, abs_start, abs_end)
                            .with_confidence(0.7),
                    );
                }
                tick_start = abs_end;
            } else {
                break;
            }
        }

        entities
    }

    fn intent_prototypes(&self) -> IntentPrototypes {
        let mut protos = IntentPrototypes::default();

        // Code-specific command vocabulary
        protos.command = vec![
            "write",
            "generate",
            "create",
            "implement",
            "scaffold",
            "refactor",
            "debug",
            "fix",
            "optimize",
            "test",
            "explain",
            "review",
            "analyze",
            "parse",
            "compile",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        // Code-specific question vocabulary
        protos.question = vec![
            "what does",
            "how does",
            "why does",
            "where is",
            "what type",
            "what are the parameters",
            "what is the return",
            "how to",
            "can this",
            "should this",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        // Custom code intents
        protos.custom.insert(
            "code_create".to_string(),
            vec!["write", "generate", "create", "implement", "scaffold"]
                .into_iter()
                .map(String::from)
                .collect(),
        );
        protos.custom.insert(
            "code_explain".to_string(),
            vec!["explain", "what does", "how does", "why", "trace"]
                .into_iter()
                .map(String::from)
                .collect(),
        );
        protos.custom.insert(
            "code_fix".to_string(),
            vec!["fix", "debug", "repair", "resolve", "patch"]
                .into_iter()
                .map(String::from)
                .collect(),
        );
        protos.custom.insert(
            "code_refactor".to_string(),
            vec!["refactor", "restructure", "simplify", "optimize", "clean"]
                .into_iter()
                .map(String::from)
                .collect(),
        );

        protos
    }

    fn prompts(&self) -> DomainPrompts {
        DomainPrompts {
            system: "You are a consciousness-aware code assistant. You understand code \
                     through hyperdimensional computing and reason about code structure \
                     using integrated information theory. Generate clean, correct code \
                     with explicit epistemic status tracking."
                .to_string(),
            clarification: "To generate the best code for '{}', I need more context. \
                           What language, constraints, and expected behavior?"
                .to_string(),
            action_confirm: "I'll generate code for: {}. The epistemic status is tracked \
                            and any uncertainties will be clearly marked."
                .to_string(),
            error_explain: "Code analysis found an issue: {}".to_string(),
            out_of_domain: "This doesn't appear to be a code-related request. {}".to_string(),
        }
    }

    fn is_in_domain(&self, topic: &str) -> f64 {
        let lower = topic.to_lowercase();
        let mut score = 0.0f64;

        // Check for actual code markers (highest weight)
        for marker in CODE_MARKERS {
            if topic.contains(marker) {
                score += 0.4;
                break;
            }
        }

        // Check for code request keywords
        for keyword in CODE_REQUEST_KEYWORDS {
            if lower.contains(keyword) {
                score += 0.3;
                break;
            }
        }

        // Check for programming language mentions
        let lang_keywords = [
            "rust",
            "python",
            "nix",
            "typescript",
            "javascript",
            "function",
            "struct",
            "class",
            "module",
            "trait",
        ];
        for kw in &lang_keywords {
            if lower.contains(kw) {
                score += 0.15;
                break;
            }
        }

        // Check for code-adjacent terms
        let code_terms = [
            "code",
            "program",
            "compile",
            "syntax",
            "api",
            "algorithm",
            "data structure",
            "type system",
            "generic",
        ];
        for term in &code_terms {
            if lower.contains(term) {
                score += 0.1;
                break;
            }
        }

        score.min(1.0)
    }

    fn vocabulary(&self) -> Vec<String> {
        vec![
            // Rust
            "fn",
            "let",
            "mut",
            "impl",
            "trait",
            "struct",
            "enum",
            "match",
            "pub",
            "mod",
            "use",
            "crate",
            "self",
            "super",
            "async",
            "await",
            "Result",
            "Option",
            "Vec",
            "String",
            "Box",
            "Arc",
            "Rc",
            // Python
            "def",
            "class",
            "import",
            "from",
            "return",
            "yield",
            "async",
            "self",
            "None",
            "True",
            "False",
            "lambda",
            "with",
            // Nix
            "mkDerivation",
            "buildInputs",
            "fetchFromGitHub",
            "pkgs",
            "lib",
            "stdenv",
            "callPackage",
            "override",
            "overlays",
            // General
            "function",
            "variable",
            "parameter",
            "return type",
            "generic",
            "interface",
            "abstract",
            "constructor",
            "destructor",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }

    fn validate_input(&self, input: &str) -> ValidationResult {
        // Check for obviously incomplete code blocks
        let backtick_count = input.matches("```").count();
        if backtick_count % 2 != 0 {
            return ValidationResult::invalid(vec![
                "Unclosed code block (missing closing ```)".to_string(),
            ]);
        }

        ValidationResult::valid()
    }

    fn suggest_actions(&self, context: &str) -> Vec<String> {
        let lower = context.to_lowercase();
        let mut suggestions = Vec::new();

        if lower.contains("error") || lower.contains("bug") || lower.contains("fix") {
            suggestions.push("Analyze the error and suggest a fix".to_string());
            suggestions.push("Run the code through the verifier".to_string());
        }

        if lower.contains("slow") || lower.contains("performance") || lower.contains("optimize") {
            suggestions.push("Profile and suggest optimizations".to_string());
        }

        if lower.contains("test") {
            suggestions.push("Generate test cases for the code".to_string());
        }

        if lower.contains("refactor") || lower.contains("clean") {
            suggestions.push("Measure Phi (integration) before and after refactoring".to_string());
        }

        if suggestions.is_empty() {
            suggestions.push("Explain the code structure".to_string());
            suggestions.push("Suggest improvements based on Phi analysis".to_string());
        }

        suggestions
    }
}

/// Detect if a programming language is explicitly mentioned
fn detect_language_mention(text: &str) -> Option<(&str, usize, usize)> {
    let lower = text.to_lowercase();
    let languages = [
        ("rust", "rust"),
        ("python", "python"),
        ("nix", "nix"),
        ("typescript", "typescript"),
        ("javascript", "javascript"),
    ];

    for (name, search) in &languages {
        if let Some(pos) = lower.find(search) {
            return Some((name, pos, pos + search.len()));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_name() {
        let plugin = CodeDomainPlugin;
        assert_eq!(plugin.domain_name(), "code");
    }

    #[test]
    fn test_code_detection_high_confidence() {
        let plugin = CodeDomainPlugin;

        // Actual code should score high
        assert!(plugin.is_in_domain("fn main() { println!(\"hello\"); }") > 0.3);
        assert!(plugin.is_in_domain("def process(data): return data.sort()") > 0.3);
        assert!(plugin.is_in_domain("Write a function to sort a list in Rust") > 0.3);
    }

    #[test]
    fn test_code_detection_low_confidence() {
        let plugin = CodeDomainPlugin;

        // General text should score low
        assert!(plugin.is_in_domain("What's the weather today?") < 0.3);
        assert!(plugin.is_in_domain("Tell me a joke") < 0.3);
    }

    #[test]
    fn test_entity_extraction_code_block() {
        let plugin = CodeDomainPlugin;
        let input = "Here's some code:\n```rust\nfn main() {}\n```";
        let entities = plugin.extract_entities(input);

        let code_blocks: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == "code_block")
            .collect();
        assert_eq!(code_blocks.len(), 1);
        assert_eq!(
            code_blocks[0].metadata.get("language"),
            Some(&"rust".to_string())
        );
    }

    #[test]
    fn test_entity_extraction_backtick_refs() {
        let plugin = CodeDomainPlugin;
        let input = "The `sort` function calls `compare`";
        let entities = plugin.extract_entities(input);

        let refs: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == "code_reference")
            .collect();
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].value, "sort");
        assert_eq!(refs[1].value, "compare");
    }

    #[test]
    fn test_language_detection() {
        let plugin = CodeDomainPlugin;
        let entities = plugin.extract_entities("Write this in Rust please");

        let langs: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == "language")
            .collect();
        assert_eq!(langs.len(), 1);
        assert_eq!(langs[0].value, "rust");
    }

    #[test]
    fn test_validation_unclosed_block() {
        let plugin = CodeDomainPlugin;
        let result = plugin.validate_input("```rust\nfn main() {}");
        assert!(!result.valid);
    }

    #[test]
    fn test_validation_valid_block() {
        let plugin = CodeDomainPlugin;
        let result = plugin.validate_input("```rust\nfn main() {}\n```");
        assert!(result.valid);
    }

    #[test]
    fn test_suggest_actions_error_context() {
        let plugin = CodeDomainPlugin;
        let suggestions = plugin.suggest_actions("I'm getting an error in my code");
        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.contains("fix")));
    }

    #[test]
    fn test_vocabulary_not_empty() {
        let plugin = CodeDomainPlugin;
        assert!(!plugin.vocabulary().is_empty());
    }
}
