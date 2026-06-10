// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! H4: Auto-Fix Suggestions
//!
//! Intelligent correction hints for NixOS configuration issues.

use std::collections::HashMap;

/// A fix that can be applied
#[derive(Debug, Clone)]
pub struct Fix {
    /// Fix identifier
    pub id: u64,
    /// Description of what the fix does
    pub description: String,
    /// File to modify
    pub file: String,
    /// Line number (if applicable)
    pub line: Option<usize>,
    /// Original text
    pub original: String,
    /// Replacement text
    pub replacement: String,
    /// Confidence (0.0 - 1.0)
    pub confidence: f32,
    /// Whether fix is safe to auto-apply
    pub safe: bool,
}

impl Fix {
    /// Create a new fix
    pub fn new(description: impl Into<String>) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

        Self {
            id: COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            description: description.into(),
            file: String::new(),
            line: None,
            original: String::new(),
            replacement: String::new(),
            confidence: 0.5,
            safe: false,
        }
    }

    /// Set file
    pub fn in_file(mut self, file: impl Into<String>) -> Self {
        self.file = file.into();
        self
    }

    /// Set line
    pub fn at_line(mut self, line: usize) -> Self {
        self.line = Some(line);
        self
    }

    /// Set original/replacement
    pub fn replace(mut self, original: impl Into<String>, replacement: impl Into<String>) -> Self {
        self.original = original.into();
        self.replacement = replacement.into();
        self
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Mark as safe to auto-apply
    pub fn safe_to_apply(mut self) -> Self {
        self.safe = true;
        self
    }

    /// Generate unified diff
    pub fn diff(&self) -> String {
        let mut diff = String::new();

        if let Some(line) = self.line {
            diff.push_str(&format!("@@ -{line} +{line} @@\n"));
        }

        for orig_line in self.original.lines() {
            diff.push_str(&format!("-{orig_line}\n"));
        }

        for repl_line in self.replacement.lines() {
            diff.push_str(&format!("+{repl_line}\n"));
        }

        diff
    }
}

/// A fix suggestion with context
#[derive(Debug, Clone)]
pub struct FixSuggestion {
    /// Primary fix
    pub fix: Fix,
    /// Alternative fixes
    pub alternatives: Vec<Fix>,
    /// Explanation of why this fix is suggested
    pub rationale: String,
    /// Related documentation
    pub docs: Vec<String>,
}

impl FixSuggestion {
    /// Create from a fix
    pub fn from_fix(fix: Fix) -> Self {
        Self {
            fix,
            alternatives: Vec::new(),
            rationale: String::new(),
            docs: Vec::new(),
        }
    }

    /// Add alternative fix
    pub fn with_alternative(mut self, alt: Fix) -> Self {
        self.alternatives.push(alt);
        self
    }

    /// Set rationale
    pub fn with_rationale(mut self, rationale: impl Into<String>) -> Self {
        self.rationale = rationale.into();
        self
    }

    /// Add documentation link
    pub fn with_doc(mut self, doc: impl Into<String>) -> Self {
        self.docs.push(doc.into());
        self
    }
}

/// Auto-fix engine
pub struct AutoFixer {
    /// Fix patterns
    patterns: Vec<FixPattern>,
    /// Generated suggestions
    suggestions: Vec<FixSuggestion>,
    /// Known option type mappings
    option_types: HashMap<String, NixType>,
}

struct FixPattern {
    /// Error pattern to match
    error_pattern: String,
    /// Generator function name
    generator: FixGenerator,
}

#[derive(Clone)]
enum FixGenerator {
    /// Fix missing semicolon
    MissingSemicolon,
    /// Fix undefined variable
    UndefinedVariable,
    /// Fix option collision
    OptionCollision,
    /// Fix deprecated option
    DeprecatedOption,
    /// Fix type mismatch
    TypeMismatch,
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // RESERVED(future): NixOS type inference variants
enum NixType {
    Bool,
    String,
    Int,
    List,
    AttrsOf(Box<NixType>),
}

impl AutoFixer {
    /// Create new auto-fixer
    pub fn new() -> Self {
        let mut fixer = Self {
            patterns: Vec::new(),
            suggestions: Vec::new(),
            option_types: HashMap::new(),
        };
        fixer.init_patterns();
        fixer.init_option_types();
        fixer
    }

    fn init_patterns(&mut self) {
        self.patterns.push(FixPattern {
            error_pattern: "syntax error, unexpected".to_string(),
            generator: FixGenerator::MissingSemicolon,
        });

        self.patterns.push(FixPattern {
            error_pattern: "undefined variable".to_string(),
            generator: FixGenerator::UndefinedVariable,
        });

        self.patterns.push(FixPattern {
            error_pattern: "has conflicting".to_string(),
            generator: FixGenerator::OptionCollision,
        });

        self.patterns.push(FixPattern {
            error_pattern: "is deprecated".to_string(),
            generator: FixGenerator::DeprecatedOption,
        });

        self.patterns.push(FixPattern {
            error_pattern: "value is a".to_string(),
            generator: FixGenerator::TypeMismatch,
        });
    }

    fn init_option_types(&mut self) {
        // Common option types
        self.option_types
            .insert("services.*.enable".to_string(), NixType::Bool);
        self.option_types
            .insert("environment.systemPackages".to_string(), NixType::List);
        self.option_types
            .insert("networking.hostName".to_string(), NixType::String);
        self.option_types
            .insert("services.*.port".to_string(), NixType::Int);
        self.option_types.insert(
            "users.users.*".to_string(),
            NixType::AttrsOf(Box::new(NixType::String)),
        );
    }

    /// Analyze error and generate fix suggestions
    pub fn analyze(&mut self, error: &str, context: Option<&str>) -> Vec<FixSuggestion> {
        self.suggestions.clear();
        let lower = error.to_lowercase();

        // Collect matching generators first to avoid borrow conflict
        let matching_generators: Vec<FixGenerator> = self
            .patterns
            .iter()
            .filter(|p| lower.contains(&p.error_pattern))
            .map(|p| p.generator.clone())
            .collect();

        for generator in matching_generators {
            self.generate_fix(&generator, error, context);
        }

        self.suggestions.clone()
    }

    fn generate_fix(&mut self, generator: &FixGenerator, error: &str, context: Option<&str>) {
        match generator {
            FixGenerator::MissingSemicolon => {
                self.fix_missing_semicolon(error, context);
            }
            FixGenerator::UndefinedVariable => {
                self.fix_undefined_variable(error, context);
            }
            FixGenerator::OptionCollision => {
                self.fix_option_collision(error, context);
            }
            FixGenerator::TypeMismatch => {
                self.fix_type_mismatch(error, context);
            }
            FixGenerator::DeprecatedOption => {
                self.fix_deprecated_option(error, context);
            }
        }
    }

    fn fix_missing_semicolon(&mut self, _error: &str, context: Option<&str>) {
        // Try to identify where semicolon is missing
        if let Some(ctx) = context {
            // Look for let-bindings without semicolons
            let lines: Vec<&str> = ctx.lines().collect();
            for (i, line) in lines.iter().enumerate() {
                let trimmed = line.trim();
                if trimmed.contains('=') && !trimmed.ends_with(';') && !trimmed.ends_with('{') {
                    let fix = Fix::new("Add missing semicolon")
                        .at_line(i + 1)
                        .replace(line.to_string(), format!("{line};"))
                        .with_confidence(0.8)
                        .safe_to_apply();

                    self.suggestions
                        .push(FixSuggestion::from_fix(fix).with_rationale(
                            "Let-bindings and attribute definitions require semicolons",
                        ));
                }
            }
        }
    }

    fn fix_undefined_variable(&mut self, error: &str, _context: Option<&str>) {
        // Extract variable name
        if let Some(start) = error.find("undefined variable '") {
            let rest = &error[start + 20..];
            if let Some(end) = rest.find('\'') {
                let var_name = &rest[..end];

                // Common package fixes
                let common_fixes: HashMap<&str, (&str, &str)> = [
                    ("pkgs", ("Add pkgs to function arguments", "{ pkgs, ... }:")),
                    ("lib", ("Add lib to function arguments", "{ lib, ... }:")),
                    (
                        "config",
                        ("Add config to function arguments", "{ config, ... }:"),
                    ),
                ]
                .into_iter()
                .collect();

                if let Some((desc, replacement)) = common_fixes.get(var_name) {
                    let fix = Fix::new(*desc)
                        .replace("{ ... }:", *replacement)
                        .with_confidence(0.9)
                        .safe_to_apply();

                    self.suggestions
                        .push(FixSuggestion::from_fix(fix).with_rationale(format!(
                            "'{var_name}' is a standard NixOS module argument"
                        )));
                } else {
                    // Might be a package
                    let fix = Fix::new(format!("Add {var_name} to packages"))
                        .replace(
                            "environment.systemPackages = with pkgs; [",
                            format!("environment.systemPackages = with pkgs; [\n    {var_name}"),
                        )
                        .with_confidence(0.6);

                    self.suggestions.push(
                        FixSuggestion::from_fix(fix)
                            .with_rationale("Variable might be a package reference"),
                    );
                }
            }
        }
    }

    fn fix_option_collision(&mut self, error: &str, _context: Option<&str>) {
        // Extract option path
        if let Some(start) = error.find("option '") {
            let rest = &error[start + 8..];
            if let Some(end) = rest.find('\'') {
                let option_path = &rest[..end];

                // Suggest mkForce
                let fix = Fix::new(format!("Use mkForce to override {option_path}"))
                    .replace(
                        format!("{option_path} ="),
                        format!("{option_path} = lib.mkForce"),
                    )
                    .with_confidence(0.85);

                let alt = Fix::new("Use mkDefault for lower priority".to_string())
                    .replace(
                        format!("{option_path} ="),
                        format!("{option_path} = lib.mkDefault"),
                    )
                    .with_confidence(0.7);

                self.suggestions.push(
                    FixSuggestion::from_fix(fix)
                        .with_alternative(alt)
                        .with_rationale("Options can be overridden using mkForce or given lower priority with mkDefault")
                );
            }
        }
    }

    fn fix_type_mismatch(&mut self, error: &str, _context: Option<&str>) {
        // Extract type info
        let actual_type = if error.contains("value is a string") {
            "string"
        } else if error.contains("value is a list") {
            "list"
        } else if error.contains("value is a set") {
            "set"
        } else if error.contains("value is a boolean") {
            "boolean"
        } else {
            return;
        };

        // Generate fix based on type
        match actual_type {
            "string" => {
                if error.contains("expected a list") {
                    let fix = Fix::new("Convert string to singleton list")
                        .replace("\"value\"", "[ \"value\" ]")
                        .with_confidence(0.7);

                    self.suggestions.push(
                        FixSuggestion::from_fix(fix).with_rationale("Wrap the string in a list"),
                    );
                }
            }
            "boolean" => {
                if error.contains("expected a string") {
                    let fix = Fix::new("Convert boolean to string")
                        .replace("true", "\"true\"")
                        .with_confidence(0.6);

                    self.suggestions.push(
                        FixSuggestion::from_fix(fix)
                            .with_rationale("Convert boolean to string representation"),
                    );
                }
            }
            _ => {}
        }
    }

    fn fix_deprecated_option(&mut self, error: &str, _context: Option<&str>) {
        // Common deprecated option mappings
        let mappings: HashMap<&str, &str> = [
            (
                "services.xserver.displayManager.auto",
                "services.displayManager.autoLogin",
            ),
            ("services.xserver.layout", "services.xserver.xkb.layout"),
            (
                "services.xserver.xkbVariant",
                "services.xserver.xkb.variant",
            ),
            (
                "services.xserver.xkbOptions",
                "services.xserver.xkb.options",
            ),
            ("networking.useDHCP", "networking.interfaces.<name>.useDHCP"),
        ]
        .into_iter()
        .collect();

        for (old, new) in &mappings {
            if error.contains(old) {
                let fix = Fix::new("Replace deprecated option".to_string())
                    .replace(*old, *new)
                    .with_confidence(0.95)
                    .safe_to_apply();

                self.suggestions.push(
                    FixSuggestion::from_fix(fix)
                        .with_rationale(format!("'{old}' is deprecated in favor of '{new}'")),
                );
                break;
            }
        }
    }

    /// Get suggestions for common package name typos
    pub fn suggest_package_fix(&self, typo: &str) -> Vec<(String, f32)> {
        let common_packages: HashMap<&str, &str> = [
            ("firefox", "firefox"),
            ("firefoz", "firefox"),
            ("firefo", "firefox"),
            ("chromium", "chromium"),
            ("chrome", "chromium"),
            ("vscode", "vscode"),
            ("code", "vscode"),
            ("vim", "vim"),
            ("neovim", "neovim"),
            ("nvim", "neovim"),
            ("git", "git"),
            ("github", "gh"),
            ("python", "python3"),
            ("python3", "python3"),
            ("nodejs", "nodejs"),
            ("node", "nodejs"),
            ("npm", "nodejs"),
            ("rust", "rustc"),
            ("cargo", "cargo"),
            ("go", "go"),
            ("golang", "go"),
        ]
        .into_iter()
        .collect();

        let mut suggestions = Vec::new();
        let lower = typo.to_lowercase();

        for (variant, correct) in &common_packages {
            if *variant == lower {
                suggestions.push((correct.to_string(), 1.0));
            } else if levenshtein(&lower, variant) <= 2 {
                let distance = levenshtein(&lower, variant);
                let confidence = 1.0 - (distance as f32 / variant.len().max(lower.len()) as f32);
                suggestions.push((correct.to_string(), confidence));
            }
        }

        // Sort by confidence
        suggestions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        suggestions.dedup_by(|a, b| a.0 == b.0);

        suggestions
    }

    /// Format suggestions for display
    pub fn format(&self) -> String {
        if self.suggestions.is_empty() {
            return "No fixes suggested.".to_string();
        }

        let mut output = String::new();

        for (i, suggestion) in self.suggestions.iter().enumerate() {
            output.push_str(&format!(
                "{}. {} (confidence: {:.0}%){}\n",
                i + 1,
                suggestion.fix.description,
                suggestion.fix.confidence * 100.0,
                if suggestion.fix.safe { " [safe]" } else { "" }
            ));

            if !suggestion.rationale.is_empty() {
                output.push_str(&format!("   Reason: {}\n", suggestion.rationale));
            }

            if !suggestion.fix.original.is_empty() {
                output.push_str("   Change:\n");
                output.push_str(&format!("   - {}\n", suggestion.fix.original));
                output.push_str(&format!("   + {}\n", suggestion.fix.replacement));
            }

            for (j, alt) in suggestion.alternatives.iter().enumerate() {
                output.push_str(&format!(
                    "   Alt {}: {} ({:.0}%)\n",
                    j + 1,
                    alt.description,
                    alt.confidence * 100.0
                ));
            }

            output.push('\n');
        }

        output
    }
}

impl Default for AutoFixer {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple Levenshtein distance
fn levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut matrix = vec![vec![0; n + 1]; m + 1];

    for i in 0..=m {
        matrix[i][0] = i;
    }
    for j in 0..=n {
        matrix[0][j] = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            matrix[i][j] = (matrix[i - 1][j] + 1)
                .min(matrix[i][j - 1] + 1)
                .min(matrix[i - 1][j - 1] + cost);
        }
    }

    matrix[m][n]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fix_creation() {
        let fix = Fix::new("Test fix")
            .in_file("test.nix")
            .at_line(10)
            .replace("old", "new")
            .with_confidence(0.9)
            .safe_to_apply();

        assert!(fix.safe);
        assert_eq!(fix.confidence, 0.9);
        assert_eq!(fix.line, Some(10));
    }

    #[test]
    fn test_undefined_variable_fix() {
        let mut fixer = AutoFixer::new();

        let suggestions = fixer.analyze("undefined variable 'pkgs'", None);

        assert!(!suggestions.is_empty());
        assert!(suggestions[0].fix.description.contains("pkgs"));
    }

    #[test]
    fn test_package_typo() {
        let fixer = AutoFixer::new();

        let suggestions = fixer.suggest_package_fix("firefoz");

        assert!(!suggestions.is_empty());
        assert_eq!(suggestions[0].0, "firefox");
    }

    #[test]
    fn test_levenshtein() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", "abc"), 0);
    }
}