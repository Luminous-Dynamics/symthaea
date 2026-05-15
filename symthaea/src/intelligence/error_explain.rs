// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! H2: Semantic Error Explanations
//!
//! Uses HDC to provide human-readable explanations for NixOS errors.

use std::collections::HashMap;

/// Category of error
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    /// Syntax error in Nix expression
    Syntax,
    /// Type mismatch
    Type,
    /// Missing attribute/option
    Missing,
    /// Undefined variable/reference
    Undefined,
    /// Infinite recursion
    Recursion,
    /// Evaluation error
    Evaluation,
    /// Build failure
    Build,
    /// Dependency conflict
    Dependency,
    /// Permission/access error
    Permission,
    /// Network/fetch error
    Network,
    /// Hash mismatch
    Hash,
    /// Configuration conflict
    Configuration,
    /// Unknown error
    Unknown,
}

impl ErrorCategory {
    /// Classify error from message
    pub fn classify(message: &str) -> Self {
        let lower = message.to_lowercase();

        if lower.contains("syntax error") || lower.contains("unexpected") {
            ErrorCategory::Syntax
        } else if lower.contains("type")
            && (lower.contains("mismatch") || lower.contains("expected"))
        {
            ErrorCategory::Type
        } else if lower.contains("attribute") && lower.contains("missing") {
            ErrorCategory::Missing
        } else if lower.contains("undefined variable") || lower.contains("not found in") {
            ErrorCategory::Undefined
        } else if lower.contains("infinite recursion") {
            ErrorCategory::Recursion
        } else if lower.contains("evaluation") && lower.contains("error") {
            ErrorCategory::Evaluation
        } else if lower.contains("build") && lower.contains("failed") {
            ErrorCategory::Build
        } else if lower.contains("conflict") && lower.contains("depend") {
            ErrorCategory::Dependency
        } else if lower.contains("permission") || lower.contains("access denied") {
            ErrorCategory::Permission
        } else if lower.contains("network") || lower.contains("fetch") || lower.contains("download")
        {
            ErrorCategory::Network
        } else if lower.contains("hash mismatch") || lower.contains("sha256") {
            ErrorCategory::Hash
        } else if lower.contains("conflict") || lower.contains("option") {
            ErrorCategory::Configuration
        } else {
            ErrorCategory::Unknown
        }
    }

    /// Get category icon
    pub fn icon(&self) -> &'static str {
        match self {
            ErrorCategory::Syntax => "📝",
            ErrorCategory::Type => "🔤",
            ErrorCategory::Missing => "❓",
            ErrorCategory::Undefined => "🔍",
            ErrorCategory::Recursion => "🔄",
            ErrorCategory::Evaluation => "⚙️",
            ErrorCategory::Build => "🔨",
            ErrorCategory::Dependency => "📦",
            ErrorCategory::Permission => "🔒",
            ErrorCategory::Network => "🌐",
            ErrorCategory::Hash => "#️⃣",
            ErrorCategory::Configuration => "⚙️",
            ErrorCategory::Unknown => "❌",
        }
    }
}

/// A semantic error explanation
#[derive(Debug, Clone)]
pub struct ErrorExplanation {
    /// Error category
    pub category: ErrorCategory,
    /// Original error message
    pub original: String,
    /// Human-readable summary
    pub summary: String,
    /// Detailed explanation
    pub explanation: String,
    /// Possible causes
    pub causes: Vec<String>,
    /// Suggested fixes
    pub suggestions: Vec<String>,
    /// Related documentation links
    pub docs: Vec<String>,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
}

impl ErrorExplanation {
    /// Create a new explanation
    fn new(category: ErrorCategory, original: impl Into<String>) -> Self {
        Self {
            category,
            original: original.into(),
            summary: String::new(),
            explanation: String::new(),
            causes: Vec::new(),
            suggestions: Vec::new(),
            docs: Vec::new(),
            confidence: 0.5,
        }
    }
}

/// Semantic error explainer
pub struct SemanticErrorExplainer {
    /// Known error patterns
    patterns: Vec<ErrorPattern>,
    /// Explanation cache
    cache: HashMap<String, ErrorExplanation>,
}

struct ErrorPattern {
    /// Pattern to match (regex-like)
    pattern: String,
    /// Category for this pattern
    category: ErrorCategory,
    /// Summary template
    summary: String,
    /// Explanation template
    explanation: String,
    /// Suggestions
    suggestions: Vec<String>,
    /// Confidence when matched
    confidence: f32,
}

impl SemanticErrorExplainer {
    /// Create new explainer
    pub fn new() -> Self {
        let mut explainer = Self {
            patterns: Vec::new(),
            cache: HashMap::new(),
        };
        explainer.init_patterns();
        explainer
    }

    fn init_patterns(&mut self) {
        // Syntax errors
        self.add_pattern(ErrorPattern {
            pattern: "syntax error, unexpected".to_string(),
            category: ErrorCategory::Syntax,
            summary: "Nix syntax error".to_string(),
            explanation: "The Nix parser encountered unexpected syntax. This usually means a missing bracket, semicolon, or other punctuation.".to_string(),
            suggestions: vec![
                "Check for missing semicolons at the end of let-bindings".to_string(),
                "Ensure all brackets, braces, and parentheses are balanced".to_string(),
                "Look for missing commas in attribute sets or lists".to_string(),
            ],
            confidence: 0.9,
        });

        // Type errors
        self.add_pattern(ErrorPattern {
            pattern: "value is a".to_string(),
            category: ErrorCategory::Type,
            summary: "Type mismatch in Nix expression".to_string(),
            explanation: "A value of one type was used where another type was expected."
                .to_string(),
            suggestions: vec![
                "Check that you're not passing a string where a list is expected".to_string(),
                "Ensure function arguments have the correct types".to_string(),
                "Verify option types match their declarations".to_string(),
            ],
            confidence: 0.85,
        });

        // Missing attribute
        self.add_pattern(ErrorPattern {
            pattern: "attribute .* missing".to_string(),
            category: ErrorCategory::Missing,
            summary: "Required attribute not found".to_string(),
            explanation: "An attribute was expected but not provided. This often happens when a module requires an option that wasn't set.".to_string(),
            suggestions: vec![
                "Check the attribute name for typos".to_string(),
                "Verify the option path is correct".to_string(),
                "Add the missing attribute to your configuration".to_string(),
            ],
            confidence: 0.9,
        });

        // Undefined variable
        self.add_pattern(ErrorPattern {
            pattern: "undefined variable".to_string(),
            category: ErrorCategory::Undefined,
            summary: "Reference to undefined variable".to_string(),
            explanation: "A variable was referenced that doesn't exist in the current scope."
                .to_string(),
            suggestions: vec![
                "Check variable name for typos".to_string(),
                "Ensure the variable is defined in the correct let-binding or function argument"
                    .to_string(),
                "If using a package, make sure it's available in your pkgs argument".to_string(),
            ],
            confidence: 0.95,
        });

        // Infinite recursion
        self.add_pattern(ErrorPattern {
            pattern: "infinite recursion".to_string(),
            category: ErrorCategory::Recursion,
            summary: "Infinite recursion detected".to_string(),
            explanation:
                "The configuration has a circular dependency where an option depends on itself."
                    .to_string(),
            suggestions: vec![
                "Check for options that reference themselves".to_string(),
                "Use mkDefault or mkForce to break cycles".to_string(),
                "Review module imports for circular dependencies".to_string(),
            ],
            confidence: 0.95,
        });

        // Build failure
        self.add_pattern(ErrorPattern {
            pattern: "builder .* failed".to_string(),
            category: ErrorCategory::Build,
            summary: "Package build failed".to_string(),
            explanation: "A package failed to compile or build. This could be due to missing dependencies, incompatible versions, or build script errors.".to_string(),
            suggestions: vec![
                "Check the build log for specific errors".to_string(),
                "Try updating nixpkgs to get a fixed version".to_string(),
                "Search for known issues in the nixpkgs repository".to_string(),
            ],
            confidence: 0.8,
        });

        // Hash mismatch
        self.add_pattern(ErrorPattern {
            pattern: "hash mismatch".to_string(),
            category: ErrorCategory::Hash,
            summary: "Source hash doesn't match".to_string(),
            explanation: "The downloaded source has a different hash than expected. This usually means the upstream source changed.".to_string(),
            suggestions: vec![
                "Update the hash in your derivation".to_string(),
                "Run 'nix-prefetch-url' to get the new hash".to_string(),
                "Check if upstream made a release update".to_string(),
            ],
            confidence: 0.95,
        });

        // Network error
        self.add_pattern(ErrorPattern {
            pattern: "unable to download".to_string(),
            category: ErrorCategory::Network,
            summary: "Network fetch failed".to_string(),
            explanation: "Nix couldn't download a required file from the network.".to_string(),
            suggestions: vec![
                "Check your internet connection".to_string(),
                "Verify the URL is correct and accessible".to_string(),
                "Try adding a substitute or cache".to_string(),
            ],
            confidence: 0.9,
        });

        // Permission error
        self.add_pattern(ErrorPattern {
            pattern: "permission denied".to_string(),
            category: ErrorCategory::Permission,
            summary: "Permission denied".to_string(),
            explanation: "The operation requires elevated privileges or the file/directory has restricted permissions.".to_string(),
            suggestions: vec![
                "Run with sudo for system operations".to_string(),
                "Check file permissions".to_string(),
                "Verify nix-daemon is running if using multi-user mode".to_string(),
            ],
            confidence: 0.9,
        });

        // Option collision
        self.add_pattern(ErrorPattern {
            pattern: "The option .* has conflicting".to_string(),
            category: ErrorCategory::Configuration,
            summary: "Configuration option conflict".to_string(),
            explanation: "Multiple modules are trying to set the same option to different values."
                .to_string(),
            suggestions: vec![
                "Use mkForce to override the conflicting value".to_string(),
                "Use mkDefault to set a lower-priority default".to_string(),
                "Remove one of the conflicting definitions".to_string(),
            ],
            confidence: 0.9,
        });
    }

    fn add_pattern(&mut self, pattern: ErrorPattern) {
        self.patterns.push(pattern);
    }

    /// Explain an error
    pub fn explain(&mut self, error_message: &str) -> ErrorExplanation {
        // Check cache
        if let Some(cached) = self.cache.get(error_message) {
            return cached.clone();
        }

        let lower = error_message.to_lowercase();

        // Find matching pattern
        let mut best_match: Option<&ErrorPattern> = None;
        let mut best_score = 0.0f32;

        for pattern in &self.patterns {
            if lower.contains(&pattern.pattern.to_lowercase()) {
                let score = if lower.is_empty() {
                    0.0
                } else {
                    pattern.confidence * (pattern.pattern.len() as f32 / lower.len() as f32)
                };
                if score > best_score {
                    best_score = score;
                    best_match = Some(pattern);
                }
            }
        }

        let explanation = if let Some(pattern) = best_match {
            self.create_explanation_from_pattern(error_message, pattern)
        } else {
            self.create_generic_explanation(error_message)
        };

        // Cache result
        self.cache
            .insert(error_message.to_string(), explanation.clone());

        explanation
    }

    fn create_explanation_from_pattern(
        &self,
        original: &str,
        pattern: &ErrorPattern,
    ) -> ErrorExplanation {
        let mut explanation = ErrorExplanation::new(pattern.category.clone(), original);

        explanation.summary = pattern.summary.clone();
        explanation.explanation = pattern.explanation.clone();
        explanation.suggestions = pattern.suggestions.clone();
        explanation.confidence = pattern.confidence;

        // Extract specific details from error message
        explanation.causes = self.extract_causes(original, &pattern.category);

        explanation
    }

    fn create_generic_explanation(&self, original: &str) -> ErrorExplanation {
        let category = ErrorCategory::classify(original);

        let mut explanation = ErrorExplanation::new(category.clone(), original);

        explanation.summary = match category {
            ErrorCategory::Syntax => "Syntax error in Nix expression".to_string(),
            ErrorCategory::Type => "Type mismatch in Nix expression".to_string(),
            ErrorCategory::Missing => "Missing required attribute".to_string(),
            ErrorCategory::Undefined => "Undefined reference".to_string(),
            ErrorCategory::Recursion => "Infinite recursion detected".to_string(),
            ErrorCategory::Evaluation => "Evaluation error".to_string(),
            ErrorCategory::Build => "Build failure".to_string(),
            ErrorCategory::Dependency => "Dependency conflict".to_string(),
            ErrorCategory::Permission => "Permission error".to_string(),
            ErrorCategory::Network => "Network error".to_string(),
            ErrorCategory::Hash => "Hash mismatch".to_string(),
            ErrorCategory::Configuration => "Configuration conflict".to_string(),
            ErrorCategory::Unknown => "Error occurred".to_string(),
        };

        explanation.explanation =
            "Unable to determine specific cause. Please review the full error message.".to_string();
        explanation
            .suggestions
            .push("Check the NixOS manual for more information".to_string());
        explanation.confidence = 0.3;

        explanation
    }

    fn extract_causes(&self, message: &str, category: &ErrorCategory) -> Vec<String> {
        let mut causes = Vec::new();

        match category {
            ErrorCategory::Missing => {
                // Try to extract attribute name
                if let Some(start) = message.find("attribute '") {
                    let rest = &message[start + 11..];
                    if let Some(end) = rest.find('\'') {
                        let attr = &rest[..end];
                        causes.push(format!("Missing attribute: {}", attr));
                        
                        // Suggest similar attributes
                        let similar = self.find_similar_attributes(attr);
                        if !similar.is_empty() {
                            causes.push(format!("Did you mean: {}", similar.join(", ")));
                        }
                    }
                }
            }
            ErrorCategory::Undefined => {
                // Try to extract variable name
                if let Some(start) = message.find("variable '") {
                    let rest = &message[start + 10..];
                    if let Some(end) = rest.find('\'') {
                        causes.push(format!("Undefined variable: {}", &rest[..end]));
                    }
                }
            }
            ErrorCategory::Type => {
                // Try to extract type info
                if message.contains("value is a") {
                    if let Some(start) = message.find("value is a ") {
                        let rest = &message[start + 11..];
                        if let Some(end) = rest.find(char::is_whitespace) {
                            causes.push(format!("Actual type: {}", &rest[..end]));
                        }
                    }
                }
            }
            _ => {}
        }

        causes
    }

    /// Format explanation for display
    pub fn format(&self, explanation: &ErrorExplanation) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "{} {}\n",
            explanation.category.icon(),
            explanation.summary
        ));
        output.push_str(&format!(
            "Confidence: {:.0}%\n\n",
            explanation.confidence * 100.0
        ));

        output.push_str(&format!("{}\n\n", explanation.explanation));

        if !explanation.causes.is_empty() {
            output.push_str("Possible causes:\n");
            for cause in &explanation.causes {
                output.push_str(&format!("  • {cause}\n"));
            }
            output.push('\n');
        }

        if !explanation.suggestions.is_empty() {
            output.push_str("Suggestions:\n");
            for suggestion in &explanation.suggestions {
                output.push_str(&format!("  → {suggestion}\n"));
            }
        }

        output
    }
}

impl Default for SemanticErrorExplainer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_classification() {
        assert_eq!(
            ErrorCategory::classify("syntax error, unexpected ';'"),
            ErrorCategory::Syntax
        );

        assert_eq!(
            ErrorCategory::classify("undefined variable 'pkgs'"),
            ErrorCategory::Undefined
        );

        assert_eq!(
            ErrorCategory::classify("hash mismatch in fixed-output derivation"),
            ErrorCategory::Hash
        );
    }

    #[test]
    fn test_explanation() {
        let mut explainer = SemanticErrorExplainer::new();

        let explanation = explainer.explain("undefined variable 'firefox'");

        assert_eq!(explanation.category, ErrorCategory::Undefined);
        assert!(!explanation.suggestions.is_empty());
        assert!(explanation.confidence > 0.5);
    }

    #[test]
    fn test_format() {
        let explainer = SemanticErrorExplainer::new();
        let explanation = ErrorExplanation {
            category: ErrorCategory::Syntax,
            original: "syntax error".to_string(),
            summary: "Test summary".to_string(),
            explanation: "Test explanation".to_string(),
            causes: vec!["Cause 1".to_string()],
            suggestions: vec!["Fix 1".to_string()],
            docs: vec![],
            confidence: 0.9,
        };

        let formatted = explainer.format(&explanation);
        assert!(formatted.contains("Test summary"));
        assert!(formatted.contains("90%"));
    }
}
