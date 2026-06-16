// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Inline Error Explanation for Shell
//!
//! Provides real-time error diagnosis and explanation when Nix commands fail.
//! Uses the NixErrorDiagnoser to parse error output and provide:
//! - Human-readable explanations
//! - Likely causes
//! - Suggested fixes
//! - Affected configuration paths

use crate::language::{ErrorDiagnosis, FixRiskLevel, NixErrorCategory, NixErrorDiagnoser};

/// Error explanation result for shell display
#[derive(Debug, Clone)]
pub struct ErrorExplanation {
    /// Category of the error
    pub category: String,

    /// Specific error type
    pub error_type: String,

    /// One-line summary
    pub summary: String,

    /// Human-readable explanation (2-3 sentences)
    pub explanation: String,

    /// Likely causes (ordered by probability)
    pub causes: Vec<String>,

    /// Suggested fixes with risk levels
    pub fixes: Vec<ShellFix>,

    /// Location where error occurred (file:line)
    pub location: Option<String>,

    /// Affected configuration paths
    pub affected_paths: Vec<String>,

    /// Confidence in the diagnosis (0-100%)
    pub confidence: u8,

    /// Icon for the category
    pub icon: &'static str,

    /// Color hint for the category
    pub color: ErrorColor,
}

/// Suggested fix with shell-specific formatting
#[derive(Debug, Clone)]
pub struct ShellFix {
    /// Description of the fix
    pub description: String,

    /// Command to run (if applicable)
    pub command: Option<String>,

    /// Risk level
    pub risk: FixRisk,

    /// Whether this is the primary/recommended fix
    pub primary: bool,
}

/// Risk level for fixes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixRisk {
    Safe,     // No side effects
    Low,      // Minor side effects, reversible
    Medium,   // May require confirmation
    High,     // Significant changes
    Critical, // Could affect system stability
}

impl FixRisk {
    pub fn icon(&self) -> &'static str {
        match self {
            Self::Safe => "✓",
            Self::Low => "○",
            Self::Medium => "△",
            Self::High => "⚠",
            Self::Critical => "⛔",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Safe => "safe",
            Self::Low => "low risk",
            Self::Medium => "medium risk",
            Self::High => "high risk",
            Self::Critical => "critical",
        }
    }
}

impl From<FixRiskLevel> for FixRisk {
    fn from(level: FixRiskLevel) -> Self {
        match level {
            FixRiskLevel::Safe => Self::Safe,
            FixRiskLevel::Low => Self::Low,
            FixRiskLevel::Medium => Self::Medium,
            FixRiskLevel::High => Self::High,
            FixRiskLevel::Critical => Self::Critical,
        }
    }
}

/// Color hint for error display
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorColor {
    Red,     // Critical/evaluation errors
    Yellow,  // Warnings, build issues
    Blue,    // Info, missing resources
    Magenta, // Permission errors
    Cyan,    // Flake-specific
    White,   // Unknown
}

/// Shell error explainer
pub struct ErrorExplainer {
    diagnoser: NixErrorDiagnoser,
}

impl Default for ErrorExplainer {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorExplainer {
    pub fn new() -> Self {
        Self {
            diagnoser: NixErrorDiagnoser::new(),
        }
    }

    /// Check if output looks like an error
    pub fn is_error_output(output: &str) -> bool {
        let lower = output.to_lowercase();
        lower.contains("error:")
            || lower.contains("failed")
            || lower.contains("cannot ")
            || lower.contains("permission denied")
            || lower.contains("assertion")
            || lower.contains("undefined")
            || lower.contains("syntax error")
            || lower.contains("hash mismatch")
            || lower.contains("infinite recursion")
    }

    /// Explain an error from command output
    pub fn explain(&self, error_output: &str) -> ErrorExplanation {
        let diagnosis = self.diagnoser.diagnose(error_output);
        self.format_diagnosis(diagnosis)
    }

    /// Format diagnosis for shell display
    fn format_diagnosis(&self, diagnosis: ErrorDiagnosis) -> ErrorExplanation {
        let (icon, color) = self.category_style(diagnosis.category);

        let fixes: Vec<ShellFix> = diagnosis
            .fixes
            .iter()
            .map(|f| ShellFix {
                description: f.description.clone(),
                command: f.command.clone(),
                risk: f.risk.into(),
                primary: f.primary,
            })
            .collect();

        ErrorExplanation {
            category: diagnosis.category.name().to_string(),
            error_type: diagnosis.error_type.name().to_string(),
            summary: self.generate_summary(&diagnosis),
            explanation: diagnosis.explanation,
            causes: diagnosis.likely_causes,
            fixes,
            location: diagnosis.location,
            affected_paths: diagnosis.affected_configs,
            confidence: (diagnosis.confidence * 100.0) as u8,
            icon,
            color,
        }
    }

    /// Get icon and color for error category
    fn category_style(&self, category: NixErrorCategory) -> (&'static str, ErrorColor) {
        match category {
            NixErrorCategory::Evaluation => ("❌", ErrorColor::Red),
            NixErrorCategory::Build => ("🔨", ErrorColor::Yellow),
            NixErrorCategory::Conflict => ("⚡", ErrorColor::Yellow),
            NixErrorCategory::MissingResource => ("🔍", ErrorColor::Blue),
            NixErrorCategory::Permission => ("🔒", ErrorColor::Magenta),
            NixErrorCategory::Flake => ("❄️", ErrorColor::Cyan),
            NixErrorCategory::Unknown => ("❓", ErrorColor::White),
        }
    }

    /// Generate a one-line summary
    fn generate_summary(&self, diagnosis: &ErrorDiagnosis) -> String {
        let location_hint = diagnosis
            .location
            .as_ref()
            .map(|l| format!(" at {l}"))
            .unwrap_or_default();

        format!(
            "{} {}{}",
            diagnosis.category.name(),
            diagnosis.error_type.name(),
            location_hint
        )
    }

    /// Format for terminal output (multiple lines)
    pub fn format_for_terminal(&self, explanation: &ErrorExplanation) -> Vec<String> {
        let mut lines = Vec::new();

        // Header
        lines.push(format!(
            "{} {} ({} confidence: {}%)",
            explanation.icon, explanation.summary, explanation.category, explanation.confidence
        ));

        // Explanation
        if !explanation.explanation.is_empty() {
            lines.push(format!("   {}", explanation.explanation));
        }

        // Causes
        if !explanation.causes.is_empty() {
            lines.push("   Likely causes:".to_string());
            for (i, cause) in explanation.causes.iter().take(3).enumerate() {
                lines.push(format!("   {}. {}", i + 1, cause));
            }
        }

        // Suggested fixes
        if !explanation.fixes.is_empty() {
            lines.push("   Suggested fixes:".to_string());
            for fix in &explanation.fixes {
                let risk_badge = format!("[{}]", fix.risk.label());
                let primary_marker = if fix.primary { "★ " } else { "  " };
                lines.push(format!(
                    "   {}{} {} {}",
                    primary_marker,
                    fix.risk.icon(),
                    fix.description,
                    risk_badge
                ));

                if let Some(ref cmd) = fix.command {
                    lines.push(format!("      → {cmd}"));
                }
            }
        }

        // Location
        if let Some(ref loc) = explanation.location {
            lines.push(format!("   Location: {loc}"));
        }

        // Affected paths
        if !explanation.affected_paths.is_empty() {
            lines.push(format!(
                "   Affected: {}",
                explanation.affected_paths.join(", ")
            ));
        }

        lines
    }

    /// Get just the fix commands for quick action
    pub fn get_fix_commands(&self, explanation: &ErrorExplanation) -> Vec<String> {
        explanation
            .fixes
            .iter()
            .filter_map(|f| f.command.clone())
            .collect()
    }
}

/// Quick error check for common patterns
pub fn quick_error_check(output: &str) -> Option<(&'static str, &'static str)> {
    let lower = output.to_lowercase();

    if lower.contains("infinite recursion") {
        Some((
            "Infinite Recursion",
            "Check for circular module imports or self-references",
        ))
    } else if lower.contains("undefined variable") {
        Some((
            "Undefined Variable",
            "Variable is not in scope - check imports and let bindings",
        ))
    } else if lower.contains("attribute") && lower.contains("missing") {
        Some(("Missing Attribute", "Required attribute not found in set"))
    } else if lower.contains("hash mismatch") {
        Some((
            "Hash Mismatch",
            "Source hash doesn't match - run nix flake update",
        ))
    } else if lower.contains("permission denied") {
        Some(("Permission Error", "Try with sudo or check file ownership"))
    } else if lower.contains("connection refused") || lower.contains("fetch failed") {
        Some((
            "Network Error",
            "Check internet connection or try --offline",
        ))
    } else if lower.contains("syntax error") {
        Some((
            "Syntax Error",
            "Check for missing semicolons, brackets, or quotes",
        ))
    } else if lower.contains("assertion") && lower.contains("failed") {
        Some((
            "Assertion Failed",
            "Configuration check failed - review option values",
        ))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_error_output() {
        assert!(ErrorExplainer::is_error_output(
            "error: undefined variable 'foo'"
        ));
        assert!(ErrorExplainer::is_error_output("build failed"));
        assert!(!ErrorExplainer::is_error_output("Build succeeded!"));
    }

    #[test]
    fn test_quick_error_check() {
        let result = quick_error_check("error: undefined variable 'pkgs'");
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, "Undefined Variable");
    }

    #[test]
    fn test_explain_error() {
        let explainer = ErrorExplainer::new();
        let explanation = explainer
            .explain("error: undefined variable 'foo' at /etc/nixos/configuration.nix:42:5");

        assert!(!explanation.summary.is_empty());
        assert_eq!(explanation.color, ErrorColor::Red); // Evaluation error
    }
}
