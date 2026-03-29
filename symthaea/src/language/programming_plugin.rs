// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Programming Domain Plugin
//!
//! Provides domain-specific language processing for programming and software
//! development content: language detection, framework recognition, code pattern
//! extraction, and programming-oriented intent prototypes.

use std::collections::HashMap;

use super::domain_plugin::{
    DomainPlugin, DomainPrompts, Entity, ErrorDiagnosis as DomainErrorDiagnosis, IntentPrototypes,
    RiskLevel, ValidationResult,
};

// ============================================================================
// PROGRAMMING KEYWORDS & PATTERNS
// ============================================================================

/// Keywords that indicate programming domain content
const PROGRAMMING_KEYWORDS: &[&str] = &[
    "code",
    "function",
    "variable",
    "class",
    "method",
    "object",
    "array",
    "string",
    "integer",
    "boolean",
    "float",
    "type",
    "compile",
    "compiler",
    "runtime",
    "debug",
    "debugger",
    "bug",
    "error",
    "exception",
    "stack trace",
    "segfault",
    "syntax",
    "semantic",
    "parser",
    "lexer",
    "token",
    "api",
    "endpoint",
    "request",
    "response",
    "http",
    "database",
    "query",
    "sql",
    "schema",
    "migration",
    "test",
    "unit test",
    "integration test",
    "mock",
    "assert",
    "git",
    "commit",
    "branch",
    "merge",
    "rebase",
    "pull request",
    "deploy",
    "ci/cd",
    "pipeline",
    "docker",
    "container",
    "algorithm",
    "data structure",
    "recursion",
    "iteration",
    "loop",
    "conditional",
    "if",
    "else",
    "switch",
    "match",
    "import",
    "module",
    "package",
    "library",
    "framework",
    "refactor",
    "optimize",
    "performance",
    "benchmark",
    "memory",
    "pointer",
    "reference",
    "ownership",
    "borrow",
    "async",
    "await",
    "thread",
    "concurrency",
    "parallel",
    "interface",
    "trait",
    "generic",
    "template",
    "polymorphism",
    "inheritance",
    "composition",
    "design pattern",
    "singleton",
    "callback",
    "promise",
    "future",
    "closure",
    "lambda",
    "ide",
    "linter",
    "formatter",
    "build system",
    "makefile",
    "source code",
    "repository",
    "dependency",
    "crate",
    "gem",
    "npm",
    "pip",
    "cargo",
    "maven",
    "gradle",
];

/// Programming languages for entity extraction
const LANGUAGES: &[&str] = &[
    "rust",
    "python",
    "javascript",
    "typescript",
    "java",
    "c++",
    "c#",
    "go",
    "golang",
    "ruby",
    "php",
    "swift",
    "kotlin",
    "scala",
    "haskell",
    "elixir",
    "erlang",
    "clojure",
    "lisp",
    "lua",
    "perl",
    "r",
    "matlab",
    "julia",
    "fortran",
    "assembly",
    "sql",
    "html",
    "css",
    "bash",
    "shell",
    "powershell",
    "zig",
    "nim",
    "ocaml",
    "f#",
    "dart",
];

/// Common frameworks and tools
const FRAMEWORKS: &[&str] = &[
    "react",
    "angular",
    "vue",
    "svelte",
    "next.js",
    "nuxt",
    "django",
    "flask",
    "fastapi",
    "express",
    "spring",
    "rails",
    "laravel",
    "phoenix",
    "actix",
    "axum",
    "rocket",
    "tokio",
    "tensorflow",
    "pytorch",
    "pandas",
    "numpy",
    "webpack",
    "vite",
    "rollup",
    "esbuild",
    "turbopack",
    "kubernetes",
    "terraform",
    "ansible",
    "jenkins",
    "github actions",
    "postgres",
    "mysql",
    "mongodb",
    "redis",
    "elasticsearch",
    "graphql",
    "rest",
    "grpc",
    "websocket",
    "oauth",
];

/// Common programming error patterns
const ERROR_PATTERNS: &[(&str, &str, &str, RiskLevel)] = &[
    (
        "null pointer",
        "null_reference",
        "Attempted to use a null/nil reference",
        RiskLevel::Medium,
    ),
    (
        "segmentation fault",
        "segfault",
        "Memory access violation; check pointer arithmetic and bounds",
        RiskLevel::High,
    ),
    (
        "stack overflow",
        "stack_overflow",
        "Recursive call depth exceeded; check for infinite recursion",
        RiskLevel::Medium,
    ),
    (
        "out of memory",
        "oom",
        "Process ran out of memory; reduce allocations or increase limits",
        RiskLevel::High,
    ),
    (
        "type mismatch",
        "type_error",
        "Incompatible types in expression or assignment",
        RiskLevel::Low,
    ),
    (
        "undefined is not",
        "undefined_access",
        "Attempted to access property on undefined/null value",
        RiskLevel::Low,
    ),
    (
        "cannot find module",
        "missing_module",
        "Module or package not found; check imports and installed dependencies",
        RiskLevel::Low,
    ),
    (
        "permission denied",
        "permission",
        "Insufficient permissions for file or network operation",
        RiskLevel::Medium,
    ),
    (
        "connection refused",
        "connection_error",
        "Could not connect to server; check the address and whether the service is running",
        RiskLevel::Low,
    ),
    (
        "syntax error",
        "syntax_error",
        "Invalid syntax; check for missing brackets, semicolons, or typos",
        RiskLevel::Safe,
    ),
    (
        "index out of bounds",
        "index_error",
        "Array/list index exceeds valid range",
        RiskLevel::Medium,
    ),
    (
        "deadlock",
        "deadlock",
        "Two or more threads are waiting on each other; review lock ordering",
        RiskLevel::High,
    ),
    (
        "race condition",
        "race_condition",
        "Concurrent access to shared state without proper synchronization",
        RiskLevel::High,
    ),
    (
        "borrow checker",
        "borrow_error",
        "Rust borrow checker violation; review ownership and lifetime annotations",
        RiskLevel::Safe,
    ),
    (
        "lifetime",
        "lifetime_error",
        "Lifetime mismatch; ensure references do not outlive their referents",
        RiskLevel::Safe,
    ),
];

// ============================================================================
// PROGRAMMING DOMAIN PLUGIN
// ============================================================================

/// Programming domain plugin for Symthaea's language processing system
///
/// Extracts programming languages, frameworks, code patterns, and error
/// information from user input. Provides intent prototypes for common
/// programming tasks like debugging, code review, and refactoring.
pub struct ProgrammingPlugin;

impl DomainPlugin for ProgrammingPlugin {
    fn domain_name(&self) -> &str {
        "programming"
    }

    fn extract_entities(&self, text: &str) -> Vec<Entity> {
        let mut entities = Vec::new();
        let lower = text.to_lowercase();

        // Extract programming language mentions
        for lang in LANGUAGES {
            // Use word-boundary-aware matching to avoid false positives
            // (e.g., "rust" in "frustrating" or "r" in "are")
            let min_len = if lang.len() <= 2 { 0 } else { 1 }; // Skip very short langs unless exact
            if lang.len() <= 2 && min_len == 0 {
                // For 1-2 char language names (r, c#, f#, go), require exact word match
                for (idx, _) in lower.match_indices(lang) {
                    let before_ok = idx == 0 || !lower.as_bytes()[idx - 1].is_ascii_alphanumeric();
                    let after_ok = idx + lang.len() >= lower.len()
                        || !lower.as_bytes()[idx + lang.len()].is_ascii_alphanumeric();
                    if before_ok && after_ok {
                        entities.push(
                            Entity::new("language", *lang, idx, idx + lang.len())
                                .with_confidence(0.7)
                                .with_metadata("category", "programming_language"),
                        );
                    }
                }
            } else {
                for (idx, _) in lower.match_indices(lang) {
                    let before_ok = idx == 0 || !lower.as_bytes()[idx - 1].is_ascii_alphanumeric();
                    let after_ok = idx + lang.len() >= lower.len()
                        || !lower.as_bytes()[idx + lang.len()].is_ascii_alphanumeric();
                    if before_ok && after_ok {
                        entities.push(
                            Entity::new("language", *lang, idx, idx + lang.len())
                                .with_confidence(0.9)
                                .with_metadata("category", "programming_language"),
                        );
                    }
                }
            }
        }

        // Extract framework mentions
        for fw in FRAMEWORKS {
            for (idx, _) in lower.match_indices(fw) {
                let before_ok = idx == 0 || !lower.as_bytes()[idx - 1].is_ascii_alphanumeric();
                let after_ok = idx + fw.len() >= lower.len()
                    || !lower.as_bytes()[idx + fw.len()].is_ascii_alphanumeric();
                if before_ok && after_ok {
                    entities.push(
                        Entity::new("framework", *fw, idx, idx + fw.len())
                            .with_confidence(0.85)
                            .with_metadata("category", "framework"),
                    );
                }
            }
        }

        // Extract code-like patterns (function calls, methods)
        // Pattern: word followed by parentheses, e.g., "foo()" or "bar(x, y)"
        let mut pos = 0;
        let bytes = text.as_bytes();
        while pos < bytes.len() {
            if bytes[pos] == b'(' {
                // Look backwards for identifier
                let paren_pos = pos;
                if pos > 0 {
                    let mut id_start = pos;
                    while id_start > 0
                        && (bytes[id_start - 1].is_ascii_alphanumeric()
                            || bytes[id_start - 1] == b'_'
                            || bytes[id_start - 1] == b'.')
                    {
                        id_start -= 1;
                    }
                    if id_start < paren_pos {
                        let identifier = &text[id_start..paren_pos];
                        if identifier.len() >= 2 && !identifier.chars().all(|c| c.is_ascii_digit())
                        {
                            entities.push(
                                Entity::new("code_pattern", identifier, id_start, paren_pos)
                                    .with_confidence(0.7)
                                    .with_metadata("pattern_type", "function_call"),
                            );
                        }
                    }
                }
            }
            pos += 1;
        }

        // Extract error-related entities
        let error_terms = ["error", "exception", "warning", "panic", "crash", "bug"];
        for term in &error_terms {
            for (idx, _) in lower.match_indices(term) {
                let before_ok = idx == 0 || !lower.as_bytes()[idx - 1].is_ascii_alphanumeric();
                let after_ok = idx + term.len() >= lower.len()
                    || !lower.as_bytes()[idx + term.len()].is_ascii_alphanumeric();
                if before_ok && after_ok {
                    entities.push(
                        Entity::new("error_indicator", *term, idx, idx + term.len())
                            .with_confidence(0.8)
                            .with_metadata("category", "error"),
                    );
                }
            }
        }

        entities
    }

    fn intent_prototypes(&self) -> IntentPrototypes {
        // Custom intents for programming
        let mut custom = HashMap::new();
        custom.insert(
            "debug".to_string(),
            vec![
                "debug".to_string(),
                "fix bug".to_string(),
                "investigate".to_string(),
                "trace".to_string(),
                "breakpoint".to_string(),
                "stack trace".to_string(),
                "reproduce".to_string(),
            ],
        );
        custom.insert(
            "explain_code".to_string(),
            vec![
                "explain".to_string(),
                "what does this do".to_string(),
                "how does this work".to_string(),
                "walk through".to_string(),
                "step by step".to_string(),
            ],
        );
        custom.insert(
            "write_code".to_string(),
            vec![
                "write".to_string(),
                "implement".to_string(),
                "create a function".to_string(),
                "generate".to_string(),
                "scaffold".to_string(),
                "boilerplate".to_string(),
            ],
        );
        custom.insert(
            "refactor".to_string(),
            vec![
                "refactor".to_string(),
                "clean up".to_string(),
                "simplify".to_string(),
                "restructure".to_string(),
                "extract method".to_string(),
                "rename".to_string(),
            ],
        );

        IntentPrototypes {
            // Programming-specific command intents
            command: vec![
                "write",
                "create",
                "implement",
                "build",
                "compile",
                "run",
                "execute",
                "test",
                "debug",
                "deploy",
                "install",
                "update",
                "refactor",
                "optimize",
                "fix",
                "format",
                "lint",
                "review",
                "merge",
                "rebase",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            // Programming-specific question intents
            question: vec![
                "how to",
                "what does",
                "why does",
                "when should",
                "what is the best way",
                "how do I",
                "can I",
                "what is the difference",
                "which is better",
                "how to fix",
                "what causes",
                "how to debug",
                "what library",
                "what framework",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            // Programming-specific complaint intents
            complaint: vec![
                "not compiling",
                "crashes",
                "segfault",
                "memory leak",
                "slow",
                "doesn't work",
                "returns wrong",
                "type error",
                "null pointer",
                "stack overflow",
                "infinite loop",
                "race condition",
                "deadlock",
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
            system: "You are a programming assistant with expertise across \
                     multiple languages and frameworks. Provide clear, \
                     well-structured code with explanations. Follow best \
                     practices and idiomatic patterns for each language."
                .to_string(),
            clarification: "Could you provide more details about the code? \
                           Specifically: '{}'"
                .to_string(),
            action_confirm: "I'll help you with this programming task: {}".to_string(),
            error_explain: "I found a programming issue: {}. \
                           Here's how to fix it."
                .to_string(),
            out_of_domain: "That doesn't seem to be a programming question. \
                           I'm specialized in software development. {}"
                .to_string(),
        }
    }

    fn diagnose_error(&self, error_text: &str) -> Option<DomainErrorDiagnosis> {
        let lower = error_text.to_lowercase();

        for (pattern, error_type, description, risk_level) in ERROR_PATTERNS {
            if lower.contains(pattern) {
                return Some(DomainErrorDiagnosis {
                    category: "programming".to_string(),
                    error_type: error_type.to_string(),
                    description: description.to_string(),
                    suggestion: Some(generate_fix_suggestion(error_type)),
                    confidence: 0.8,
                    risk_level: *risk_level,
                    location: extract_code_location(&lower),
                });
            }
        }

        // Check for common compiler/runtime error formats
        // Pattern: "file:line:col: error: message"
        if lower.contains("error:") || lower.contains("error[") {
            return Some(DomainErrorDiagnosis {
                category: "programming".to_string(),
                error_type: "compilation_error".to_string(),
                description: format!(
                    "Compilation error: {}",
                    error_text.lines().next().unwrap_or("unknown")
                ),
                suggestion: Some(
                    "Read the error message carefully; the compiler often suggests a fix"
                        .to_string(),
                ),
                confidence: 0.7,
                risk_level: RiskLevel::Safe,
                location: extract_code_location(&lower),
            });
        }

        None
    }

    fn validate_input(&self, input: &str) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut suggestions = Vec::new();

        // Check for balanced delimiters
        let mut paren_depth: i32 = 0;
        let mut bracket_depth: i32 = 0;
        let mut brace_depth: i32 = 0;

        for ch in input.chars() {
            match ch {
                '(' => paren_depth += 1,
                ')' => paren_depth -= 1,
                '[' => bracket_depth += 1,
                ']' => bracket_depth -= 1,
                '{' => brace_depth += 1,
                '}' => brace_depth -= 1,
                _ => {}
            }
            if paren_depth < 0 || bracket_depth < 0 || brace_depth < 0 {
                errors.push("Unexpected closing delimiter without matching opener".to_string());
                break;
            }
        }

        if paren_depth != 0 {
            errors.push(format!("Mismatched parentheses (depth: {paren_depth})"));
        }
        if bracket_depth != 0 {
            errors.push(format!("Mismatched brackets (depth: {bracket_depth})"));
        }
        if brace_depth != 0 {
            errors.push(format!("Mismatched braces (depth: {brace_depth})"));
        }

        // Check for common code smells in short snippets
        if input.contains("TODO") || input.contains("FIXME") || input.contains("HACK") {
            warnings.push("Code contains TODO/FIXME/HACK markers".to_string());
        }

        if input.contains("unwrap()") {
            warnings
                .push("Using unwrap() can panic at runtime; consider using ? or match".to_string());
            suggestions.push("Replace unwrap() with proper error handling".to_string());
        }

        if errors.is_empty() {
            let mut result = ValidationResult::valid();
            result.warnings = warnings;
            result.suggestions = suggestions;
            result
        } else {
            let mut result = ValidationResult::invalid(errors);
            result.warnings = warnings;
            result.suggestions = suggestions;
            result
        }
    }

    fn is_in_domain(&self, topic: &str) -> f64 {
        let lower = topic.to_lowercase();
        let mut score: f64 = 0.0;
        let mut keyword_matches = 0;

        // Check for programming keywords
        for keyword in PROGRAMMING_KEYWORDS {
            if keyword.len() < 3 {
                continue; // Skip very short keywords
            }
            if lower.contains(keyword) {
                keyword_matches += 1;
                // Strong indicators
                if [
                    "code",
                    "function",
                    "compile",
                    "debug",
                    "algorithm",
                    "refactor",
                    "repository",
                    "api",
                    "deploy",
                    "test",
                ]
                .contains(keyword)
                {
                    score += 0.3;
                } else {
                    score += 0.1;
                }
            }
        }

        // Check for programming language mentions
        for lang in LANGUAGES {
            if lang.len() <= 2 {
                continue; // Skip ambiguous short names for domain detection
            }
            if lower.contains(lang) {
                score += 0.3;
                keyword_matches += 1;
            }
        }

        // Check for framework mentions
        for fw in FRAMEWORKS {
            if lower.contains(fw) {
                score += 0.25;
                keyword_matches += 1;
            }
        }

        // Check for code-like patterns
        if lower.contains("()")
            || lower.contains("{}")
            || lower.contains("[]")
            || lower.contains("->")
            || lower.contains("=>")
            || lower.contains("::")
        {
            score += 0.2;
        }

        if keyword_matches > 0 {
            (0.55 + score).min(1.0)
        } else {
            0.1
        }
    }

    fn vocabulary(&self) -> Vec<String> {
        let mut vocab: Vec<String> = PROGRAMMING_KEYWORDS.iter().map(|s| s.to_string()).collect();
        vocab.extend(LANGUAGES.iter().map(|s| s.to_string()));
        vocab.extend(FRAMEWORKS.iter().map(|s| s.to_string()));
        vocab
    }

    fn suggest_actions(&self, context: &str) -> Vec<String> {
        let lower = context.to_lowercase();
        let mut actions = Vec::new();

        if lower.contains("error") || lower.contains("bug") || lower.contains("crash") {
            actions.push("Add logging/print statements around the error location".to_string());
            actions.push("Check the stack trace for the root cause".to_string());
            actions.push("Write a minimal reproduction case".to_string());
        }
        if lower.contains("slow") || lower.contains("performance") {
            actions.push("Profile the code to find bottlenecks".to_string());
            actions.push("Check for unnecessary allocations or copies".to_string());
            actions.push("Consider caching frequently computed results".to_string());
        }
        if lower.contains("test") {
            actions.push("Write tests for edge cases and error conditions".to_string());
            actions.push("Ensure test isolation (no shared mutable state)".to_string());
        }
        if lower.contains("deploy") {
            actions.push("Run the full test suite before deploying".to_string());
            actions.push("Use a staging environment for validation".to_string());
            actions.push("Prepare a rollback plan".to_string());
        }
        if lower.contains("refactor") {
            actions.push("Ensure existing tests pass before and after".to_string());
            actions.push("Make small, incremental changes".to_string());
            actions.push("Use version control to track each step".to_string());
        }

        actions
    }
}

/// Generate a fix suggestion based on error type
fn generate_fix_suggestion(error_type: &str) -> String {
    match error_type {
        "null_reference" => "Check for null/nil before accessing the value, or use Option/Optional types".to_string(),
        "segfault" => "Run with a memory debugger (e.g., valgrind, AddressSanitizer) to locate the invalid access".to_string(),
        "stack_overflow" => "Add a base case to recursive functions, or convert to an iterative approach".to_string(),
        "oom" => "Profile memory usage, look for leaks, and consider streaming or pagination".to_string(),
        "type_error" => "Check variable types and add explicit type annotations or casts".to_string(),
        "undefined_access" => "Add null/undefined checks, or use optional chaining (?.)".to_string(),
        "missing_module" => "Run 'npm install', 'pip install', or 'cargo build' to install dependencies".to_string(),
        "permission" => "Check file permissions and user privileges; use sudo if necessary".to_string(),
        "connection_error" => "Verify the server is running and the address/port are correct".to_string(),
        "syntax_error" => "Review the line indicated in the error for typos or missing delimiters".to_string(),
        "index_error" => "Add bounds checking before array access, or use safe access methods".to_string(),
        "deadlock" => "Review lock acquisition order; use lock-free data structures if possible".to_string(),
        "race_condition" => "Use mutexes, channels, or atomic operations to synchronize access".to_string(),
        "borrow_error" => "Restructure code to avoid conflicting borrows; consider cloning or using Rc/Arc".to_string(),
        "lifetime_error" => "Add explicit lifetime annotations and ensure references do not outlive owned data".to_string(),
        _ => "Review the error message and consult the language documentation".to_string(),
    }
}

/// Extract code location from error text
fn extract_code_location(error_text: &str) -> Option<super::domain_plugin::ErrorLocation> {
    // Common patterns: "file.rs:42:5" or "at line 42" or "File \"file.py\", line 42"
    for line in error_text.lines() {
        // Pattern: "path:line:col"
        let parts: Vec<&str> = line.splitn(4, ':').collect();
        if parts.len() >= 3 {
            if let (Some(_), Ok(line_num)) = (
                parts[0].strip_prefix(' ').or(Some(parts[0])),
                parts[1].trim().parse::<usize>(),
            ) {
                let col = parts.get(2).and_then(|c| c.trim().parse().ok());
                return Some(super::domain_plugin::ErrorLocation {
                    file: Some(parts[0].trim().to_string()),
                    line: Some(line_num),
                    column: col,
                    context: Some(line.to_string()),
                });
            }
        }
    }
    None
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_name() {
        let plugin = ProgrammingPlugin;
        assert_eq!(plugin.domain_name(), "programming");
    }

    #[test]
    fn test_language_extraction() {
        let plugin = ProgrammingPlugin;

        let entities = plugin.extract_entities("I'm writing a Rust function that calls Python");
        let languages: Vec<&Entity> = entities
            .iter()
            .filter(|e| e.entity_type == "language")
            .collect();
        assert!(languages.iter().any(|e| e.value == "rust"));
        assert!(languages.iter().any(|e| e.value == "python"));
    }

    #[test]
    fn test_framework_extraction() {
        let plugin = ProgrammingPlugin;

        let entities = plugin.extract_entities("Build a REST API with actix and postgres");
        let frameworks: Vec<&Entity> = entities
            .iter()
            .filter(|e| e.entity_type == "framework")
            .collect();
        assert!(frameworks.iter().any(|e| e.value == "actix"));
        assert!(frameworks.iter().any(|e| e.value == "postgres"));
    }

    #[test]
    fn test_domain_detection() {
        let plugin = ProgrammingPlugin;

        assert!(plugin.is_in_domain("How do I debug a Rust function?") > 0.6);
        assert!(plugin.is_in_domain("Write a Python API with FastAPI") > 0.6);
        assert!(plugin.is_in_domain("What is the weather today?") < 0.3);
    }

    #[test]
    fn test_error_diagnosis() {
        let plugin = ProgrammingPlugin;

        let diag = plugin.diagnose_error(
            "thread 'main' panicked at 'index out of bounds: len is 3 but index is 5'",
        );
        assert!(diag.is_some());
        let diag = diag.unwrap();
        assert_eq!(diag.error_type, "index_error");

        let diag = plugin.diagnose_error("error: borrow checker says cannot borrow as mutable");
        assert!(diag.is_some());
    }

    #[test]
    fn test_code_pattern_extraction() {
        let plugin = ProgrammingPlugin;

        let entities = plugin.extract_entities("Call fetch_data() and process_result(x, y)");
        let patterns: Vec<&Entity> = entities
            .iter()
            .filter(|e| e.entity_type == "code_pattern")
            .collect();
        assert!(patterns.iter().any(|e| e.value == "fetch_data"));
        assert!(patterns.iter().any(|e| e.value == "process_result"));
    }

    #[test]
    fn test_validation() {
        let plugin = ProgrammingPlugin;

        let result = plugin.validate_input("fn main() { println!(\"hello\"); }");
        assert!(result.valid);

        let result = plugin.validate_input("fn main() { println!(\"hello\")");
        assert!(!result.valid);
    }

    #[test]
    fn test_unwrap_warning() {
        let plugin = ProgrammingPlugin;
        let result = plugin.validate_input("let x = foo().unwrap();");
        assert!(result.valid);
        assert!(!result.warnings.is_empty());
    }
}
