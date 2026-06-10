// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Code Health Scanner
//!
//! A comprehensive code quality analyzer that uses multiple factors:
//! - Phi (integration) - how tightly coupled is the code?
//! - Complexity - cyclomatic complexity, nesting depth
//! - Cohesion - do functions/types belong together?
//! - Coverage - how complete is the module?
//! - Primitives - Code tier primitive analysis
//!
//! Phi is ONE factor among several, providing a consciousness-inspired
//! measure of information integration in code structure.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use symthaea_core::hdc::ContinuousHV;

use crate::consciousness::code_primitives::{CodeOperation, CodePrimitiveExecutor};
use crate::hdc::code_encoder::CodeHDEncoder;
use crate::language::code_parser::{Entity, EntityKind, ParsedCode};

/// Health score for a single module
#[derive(Debug, Clone)]
pub struct ModuleHealth {
    /// Module path
    pub path: PathBuf,

    /// Overall health score (0.0 - 1.0, higher is better)
    pub overall_score: f32,

    /// Individual factor scores
    pub factors: HealthFactors,

    /// Specific issues found
    pub issues: Vec<HealthIssue>,

    /// Suggested improvements
    pub suggestions: Vec<HealthSuggestion>,
}

/// Individual health factors
#[derive(Debug, Clone)]
pub struct HealthFactors {
    /// Phi (integration) score - how tightly integrated is the code?
    /// Based on HDC similarity between entities.
    pub phi: f32,

    /// Complexity score - inverse of cyclomatic complexity
    /// Lower complexity = higher score
    pub complexity: f32,

    /// Cohesion score - do entities belong together?
    /// High cohesion = entities are semantically related
    pub cohesion: f32,

    /// Entity density - ratio of meaningful entities to code size
    pub density: f32,

    /// Structure score - good use of types, functions, modules
    pub structure: f32,

    /// Primitive phi - integration score from Code tier primitives
    pub primitive_psi: f32,
}

impl Default for HealthFactors {
    fn default() -> Self {
        Self {
            phi: 0.5,
            complexity: 0.5,
            cohesion: 0.5,
            density: 0.5,
            structure: 0.5,
            primitive_psi: 0.5,
        }
    }
}

impl HealthFactors {
    /// Compute weighted overall score
    pub fn overall(&self, weights: &HealthWeights) -> f32 {
        let total_weight = weights.phi
            + weights.complexity
            + weights.cohesion
            + weights.density
            + weights.structure
            + weights.primitive_psi;

        if total_weight == 0.0 {
            return 0.5;
        }

        (self.phi * weights.phi
            + self.complexity * weights.complexity
            + self.cohesion * weights.cohesion
            + self.density * weights.density
            + self.structure * weights.structure
            + self.primitive_psi * weights.primitive_psi)
            / total_weight
    }
}

/// Weights for health factors
#[derive(Debug, Clone)]
pub struct HealthWeights {
    pub phi: f32,
    pub complexity: f32,
    pub cohesion: f32,
    pub density: f32,
    pub structure: f32,
    pub primitive_psi: f32,
}

impl Default for HealthWeights {
    fn default() -> Self {
        Self {
            phi: 1.0,           // Integration matters
            complexity: 1.5,    // Complexity is important
            cohesion: 1.2,      // Cohesion matters for maintainability
            density: 0.5,       // Less important
            structure: 1.0,     // Good structure helps
            primitive_psi: 0.8, // Primitive integration is useful
        }
    }
}

/// Severity of a health issue
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueSeverity {
    /// Minor issue, low priority
    Low,
    /// Moderate issue, should be addressed
    Medium,
    /// Significant issue, needs attention
    High,
    /// Critical issue, should be fixed soon
    Critical,
}

/// A specific health issue
#[derive(Debug, Clone)]
pub struct HealthIssue {
    pub severity: IssueSeverity,
    pub category: IssueCategory,
    pub message: String,
    pub location: Option<String>,
}

/// Category of health issue
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueCategory {
    /// Low integration (Phi)
    LowIntegration,
    /// High complexity
    HighComplexity,
    /// Poor cohesion
    PoorCohesion,
    /// Missing structure (types, modules)
    MissingStructure,
    /// Dead code or unused entities
    DeadCode,
    /// Large file/module
    TooLarge,
    /// Other
    Other,
}

/// A suggested improvement
#[derive(Debug, Clone)]
pub struct HealthSuggestion {
    pub priority: u8, // 1-5, 1 is highest priority
    pub action: SuggestedAction,
    pub description: String,
    pub expected_improvement: f32, // Expected score improvement
}

/// Suggested action to improve health
#[derive(Debug, Clone)]
pub enum SuggestedAction {
    /// Extract functionality into a new module
    ExtractModule { suggested_name: String },
    /// Extract a trait from implementations
    ExtractTrait { suggested_name: String },
    /// Split a large function
    SplitFunction { target: String },
    /// Add type definitions
    AddTypes,
    /// Reduce nesting depth
    ReduceNesting { target: String },
    /// Remove dead code
    RemoveDeadCode { target: String },
    /// Add documentation
    AddDocumentation { target: String },
    /// General refactoring
    Refactor { description: String },
}

/// Code Health Scanner
pub struct CodeHealthScanner {
    encoder: CodeHDEncoder,
    primitive_executor: CodePrimitiveExecutor,
    weights: HealthWeights,
}

impl CodeHealthScanner {
    /// Create a new code health scanner
    pub fn new(dim: usize) -> Self {
        Self {
            encoder: CodeHDEncoder::new(dim),
            primitive_executor: CodePrimitiveExecutor::new(dim),
            weights: HealthWeights::default(),
        }
    }

    /// Create with custom weights
    pub fn with_weights(dim: usize, weights: HealthWeights) -> Self {
        Self {
            encoder: CodeHDEncoder::new(dim),
            primitive_executor: CodePrimitiveExecutor::new(dim),
            weights,
        }
    }

    /// Scan a single module for health
    pub fn scan_module(&self, path: &Path, parsed: &ParsedCode) -> ModuleHealth {
        let factors = self.compute_factors(parsed);
        let issues = self.detect_issues(parsed, &factors);
        let suggestions = self.generate_suggestions(&issues, &factors);
        let overall_score = factors.overall(&self.weights);

        ModuleHealth {
            path: path.to_path_buf(),
            overall_score,
            factors,
            issues,
            suggestions,
        }
    }

    /// Quick scan - just compute overall score
    pub fn quick_scan(&self, parsed: &ParsedCode) -> f32 {
        let factors = self.compute_factors(parsed);
        factors.overall(&self.weights)
    }

    /// Compute all health factors
    fn compute_factors(&self, parsed: &ParsedCode) -> HealthFactors {
        HealthFactors {
            phi: self.compute_phi(parsed),
            complexity: self.compute_complexity_score(parsed),
            cohesion: self.compute_cohesion(parsed),
            density: self.compute_density(parsed),
            structure: self.compute_structure_score(parsed),
            primitive_psi: self.compute_primitive_psi(),
        }
    }

    /// Compute Phi (integration) score
    ///
    /// Uses HDC similarity between entity encodings as a proxy for
    /// information integration. High similarity = tightly integrated.
    fn compute_phi(&self, parsed: &ParsedCode) -> f32 {
        let entities = parsed.all_entities();
        if entities.len() < 2 {
            return 1.0; // Single entity is maximally integrated with itself
        }

        // Encode all entities
        let hvs: Vec<ContinuousHV> = entities
            .iter()
            .map(|e| self.encoder.encode_entity(e))
            .collect();

        // Compute average pairwise similarity
        let mut total_sim = 0.0f32;
        let mut count = 0;

        for i in 0..hvs.len() {
            for j in (i + 1)..hvs.len() {
                total_sim += hvs[i].similarity(&hvs[j]);
                count += 1;
            }
        }

        if count > 0 {
            (total_sim / count as f32).max(0.0)
        } else {
            0.5
        }
    }

    /// Compute complexity score (inverse of complexity)
    ///
    /// Estimates based on nesting depth and function count.
    fn compute_complexity_score(&self, parsed: &ParsedCode) -> f32 {
        let entities = parsed.all_entities();
        let func_count = entities
            .iter()
            .filter(|e| e.kind == EntityKind::Function)
            .count();

        // Estimate nesting from child depth
        let max_depth = entities
            .iter()
            .map(|e| self.entity_depth(e))
            .max()
            .unwrap_or(0);

        // Penalize high function count and deep nesting
        let func_penalty = (func_count as f32 / 20.0).min(1.0);
        let depth_penalty = (max_depth as f32 / 5.0).min(1.0);

        // Invert: lower complexity = higher score
        1.0 - (func_penalty * 0.5 + depth_penalty * 0.5)
    }

    /// Compute entity depth (how nested is it)
    fn entity_depth(&self, entity: &Entity) -> usize {
        if entity.children.is_empty() {
            0
        } else {
            1 + entity
                .children
                .iter()
                .map(|c| self.entity_depth(c))
                .max()
                .unwrap_or(0)
        }
    }

    /// Compute cohesion score
    ///
    /// High cohesion = entities are semantically related (similar names/purposes)
    fn compute_cohesion(&self, parsed: &ParsedCode) -> f32 {
        let entities = parsed.all_entities();
        if entities.len() < 2 {
            return 1.0;
        }

        // Check if entities share common naming patterns
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();

        // Find common prefixes/suffixes
        let common_patterns = self.find_common_patterns(&names);

        // More common patterns = higher cohesion
        let pattern_score = (common_patterns as f32 / names.len() as f32).min(1.0);

        // Also check entity type distribution
        let type_counts = self.count_entity_types(parsed);
        let type_balance = self.compute_type_balance(&type_counts);

        (pattern_score * 0.6 + type_balance * 0.4).max(0.0).min(1.0)
    }

    /// Find common naming patterns
    fn find_common_patterns(&self, names: &[&str]) -> usize {
        if names.len() < 2 {
            return 0;
        }

        let mut pattern_count = 0;

        // Check for common prefixes (3+ chars)
        for i in 0..names.len() {
            for j in (i + 1)..names.len() {
                let common_prefix = self.common_prefix_len(names[i], names[j]);
                if common_prefix >= 3 {
                    pattern_count += 1;
                }
            }
        }

        pattern_count
    }

    fn common_prefix_len(&self, a: &str, b: &str) -> usize {
        a.chars()
            .zip(b.chars())
            .take_while(|(ca, cb)| ca == cb)
            .count()
    }

    /// Count entity types
    fn count_entity_types(&self, parsed: &ParsedCode) -> HashMap<EntityKind, usize> {
        let mut counts = HashMap::new();
        for entity in parsed.all_entities() {
            *counts.entry(entity.kind).or_insert(0) += 1;
        }
        counts
    }

    /// Compute type balance (good mix of functions, types, etc.)
    fn compute_type_balance(&self, counts: &HashMap<EntityKind, usize>) -> f32 {
        let total: usize = counts.values().sum();
        if total == 0 {
            return 0.5;
        }

        // Ideal: mix of functions and types
        let func_ratio = *counts.get(&EntityKind::Function).unwrap_or(&0) as f32 / total as f32;
        let type_ratio = (*counts.get(&EntityKind::Struct).unwrap_or(&0)
            + *counts.get(&EntityKind::Enum).unwrap_or(&0)
            + *counts.get(&EntityKind::Trait).unwrap_or(&0)) as f32
            / total as f32;

        // Penalize extremes (all functions or all types)
        if func_ratio > 0.9 || type_ratio > 0.9 {
            0.5
        } else if func_ratio > 0.3 && type_ratio > 0.1 {
            0.9 // Good balance
        } else {
            0.7
        }
    }

    /// Compute entity density
    fn compute_density(&self, parsed: &ParsedCode) -> f32 {
        let entity_count = parsed.all_entities().len();
        let source_lines = parsed.source.lines().count().max(1);

        // Ideal: ~10-20 lines per entity
        let lines_per_entity = source_lines as f32 / entity_count.max(1) as f32;

        if lines_per_entity < 5.0 {
            0.7 // Too dense
        } else if lines_per_entity > 50.0 {
            0.6 // Too sparse
        } else {
            0.9 // Good density
        }
    }

    /// Compute structure score
    fn compute_structure_score(&self, parsed: &ParsedCode) -> f32 {
        let entities = parsed.all_entities();

        // Check for good practices
        let has_types = entities.iter().any(|e| {
            e.kind == EntityKind::Struct
                || e.kind == EntityKind::Enum
                || e.kind == EntityKind::Trait
        });

        let has_impl = entities.iter().any(|e| e.kind == EntityKind::TraitImpl);
        let has_tests = entities
            .iter()
            .any(|e| e.name.contains("test") || e.annotations.contains_key("test"));

        let mut score: f32 = 0.5;
        if has_types {
            score += 0.2;
        }
        if has_impl {
            score += 0.15;
        }
        if has_tests {
            score += 0.15;
        }

        score.min(1.0)
    }

    /// Compute primitive phi using Code tier
    fn compute_primitive_psi(&self) -> f32 {
        // Use Verify operation as it exercises multiple primitives
        let result = self.primitive_executor.execute(CodeOperation::Verify);
        result.phi
    }

    /// Detect health issues
    fn detect_issues(&self, parsed: &ParsedCode, factors: &HealthFactors) -> Vec<HealthIssue> {
        let mut issues = Vec::new();

        // Low integration
        if factors.phi < 0.3 {
            issues.push(HealthIssue {
                severity: IssueSeverity::Medium,
                category: IssueCategory::LowIntegration,
                message: format!(
                    "Low integration score ({:.2}) - entities may not belong together",
                    factors.phi
                ),
                location: None,
            });
        }

        // High complexity
        if factors.complexity < 0.4 {
            issues.push(HealthIssue {
                severity: IssueSeverity::High,
                category: IssueCategory::HighComplexity,
                message: "High complexity detected - consider splitting functions".to_string(),
                location: None,
            });
        }

        // Poor cohesion
        if factors.cohesion < 0.3 {
            issues.push(HealthIssue {
                severity: IssueSeverity::Medium,
                category: IssueCategory::PoorCohesion,
                message: "Poor cohesion - entities may not be related".to_string(),
                location: None,
            });
        }

        // Large file
        let line_count = parsed.source.lines().count();
        if line_count > 500 {
            issues.push(HealthIssue {
                severity: IssueSeverity::Low,
                category: IssueCategory::TooLarge,
                message: format!("Large file ({} lines) - consider splitting", line_count),
                location: None,
            });
        }

        // Missing structure
        if factors.structure < 0.5 {
            issues.push(HealthIssue {
                severity: IssueSeverity::Low,
                category: IssueCategory::MissingStructure,
                message: "Could benefit from more type definitions".to_string(),
                location: None,
            });
        }

        issues
    }

    /// Generate improvement suggestions
    fn generate_suggestions(
        &self,
        issues: &[HealthIssue],
        _factors: &HealthFactors,
    ) -> Vec<HealthSuggestion> {
        let mut suggestions = Vec::new();

        for issue in issues {
            match issue.category {
                IssueCategory::LowIntegration => {
                    suggestions.push(HealthSuggestion {
                        priority: 3,
                        action: SuggestedAction::ExtractModule {
                            suggested_name: "extracted".to_string(),
                        },
                        description: "Consider extracting unrelated functionality".to_string(),
                        expected_improvement: 0.15,
                    });
                }
                IssueCategory::HighComplexity => {
                    suggestions.push(HealthSuggestion {
                        priority: 2,
                        action: SuggestedAction::SplitFunction {
                            target: "complex functions".to_string(),
                        },
                        description: "Split complex functions into smaller units".to_string(),
                        expected_improvement: 0.2,
                    });
                }
                IssueCategory::TooLarge => {
                    suggestions.push(HealthSuggestion {
                        priority: 4,
                        action: SuggestedAction::ExtractModule {
                            suggested_name: "submodule".to_string(),
                        },
                        description: "Consider splitting into multiple files".to_string(),
                        expected_improvement: 0.1,
                    });
                }
                IssueCategory::MissingStructure => {
                    suggestions.push(HealthSuggestion {
                        priority: 5,
                        action: SuggestedAction::AddTypes,
                        description: "Add type definitions for better structure".to_string(),
                        expected_improvement: 0.1,
                    });
                }
                _ => {}
            }
        }

        // Sort by priority (lower = more important)
        suggestions.sort_by_key(|s| s.priority);
        suggestions
    }

    /// Scan multiple modules and return sorted by health (worst first)
    pub fn scan_codebase(&self, modules: &[(PathBuf, ParsedCode)]) -> Vec<ModuleHealth> {
        let mut results: Vec<ModuleHealth> = modules
            .iter()
            .map(|(path, parsed)| self.scan_module(path, parsed))
            .collect();

        // Sort by overall score (lowest/worst first)
        results.sort_by(|a, b| {
            a.overall_score
                .partial_cmp(&b.overall_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }

    /// Generate a summary report
    pub fn summary_report(&self, results: &[ModuleHealth]) -> HealthSummary {
        if results.is_empty() {
            return HealthSummary::default();
        }

        let avg_score: f32 =
            results.iter().map(|r| r.overall_score).sum::<f32>() / results.len() as f32;

        let critical_count = results
            .iter()
            .flat_map(|r| &r.issues)
            .filter(|i| i.severity == IssueSeverity::Critical)
            .count();

        let high_count = results
            .iter()
            .flat_map(|r| &r.issues)
            .filter(|i| i.severity == IssueSeverity::High)
            .count();

        let worst = results.first().cloned();
        let best = results.last().cloned();

        HealthSummary {
            module_count: results.len(),
            average_score: avg_score,
            critical_issues: critical_count,
            high_issues: high_count,
            worst_module: worst,
            best_module: best,
        }
    }
}

/// Summary of codebase health
#[derive(Debug, Clone, Default)]
pub struct HealthSummary {
    pub module_count: usize,
    pub average_score: f32,
    pub critical_issues: usize,
    pub high_issues: usize,
    pub worst_module: Option<ModuleHealth>,
    pub best_module: Option<ModuleHealth>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language::code_parser::Span;

    fn test_span() -> Span {
        Span {
            start_byte: 0,
            end_byte: 100,
            start_line: 0,
            start_col: 0,
            end_line: 10,
            end_col: 0,
        }
    }

    fn make_parsed(source: &str, funcs: &[&str], types: &[&str]) -> ParsedCode {
        let mut parsed = ParsedCode::new(source, "rust");
        for f in funcs {
            parsed
                .entities
                .push(Entity::new(EntityKind::Function, *f, test_span()));
        }
        for t in types {
            parsed
                .entities
                .push(Entity::new(EntityKind::Struct, *t, test_span()));
        }
        parsed
    }

    #[test]
    fn test_health_scanner_creation() {
        let scanner = CodeHealthScanner::new(512);
        assert_eq!(scanner.weights.phi, 1.0);
    }

    #[test]
    fn test_quick_scan() {
        let scanner = CodeHealthScanner::new(512);
        let source = "fn foo() {}\nfn bar() {}";
        let parsed = make_parsed(source, &["foo", "bar"], &[]);

        let score = scanner.quick_scan(&parsed);
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_module_scan() {
        let scanner = CodeHealthScanner::new(512);
        let source = "struct Parser {}\nimpl Parser { fn parse() {} }";
        let parsed = make_parsed(source, &["parse"], &["Parser"]);

        let health = scanner.scan_module(Path::new("parser.rs"), &parsed);
        assert!(health.overall_score > 0.0);
        assert_eq!(health.path, PathBuf::from("parser.rs"));
    }

    #[test]
    fn test_cohesive_naming() {
        let scanner = CodeHealthScanner::new(512);
        // Cohesive: all functions start with "parse_"
        let parsed = make_parsed(
            "",
            &["parse_input", "parse_output", "parse_config"],
            &["ParseResult"],
        );
        let factors = scanner.compute_factors(&parsed);

        // Should have higher cohesion due to naming patterns
        assert!(factors.cohesion >= 0.5);
    }

    #[test]
    fn test_issue_detection() {
        let scanner = CodeHealthScanner::new(512);
        // Create a large file with many functions
        let mut source = String::new();
        for i in 0..60 {
            source.push_str(&format!("fn func{}() {{}}\n", i));
        }
        let funcs: Vec<String> = (0..60).map(|i| format!("func{}", i)).collect();
        let func_refs: Vec<&str> = funcs.iter().map(|s| s.as_str()).collect();

        let parsed = make_parsed(&source.repeat(10), &func_refs, &[]);
        let health = scanner.scan_module(Path::new("large.rs"), &parsed);

        // Should detect some issues
        assert!(!health.issues.is_empty() || health.overall_score < 0.8);
    }
}
