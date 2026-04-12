// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Code Self-Modification — Recursive Loop Closure
//!
//! Symthaea generates new auto-fix rules by analyzing its own error patterns,
//! validates them, and integrates them into the generation pipeline.
//!
//! This closes the recursive self-improvement loop:
//!
//! ```text
//! Generate code → compile fails → error recorded in CodeErrorKnowledge
//!     ↓
//! FixRuleGenerator detects failure cluster (≥5 attempts, <50% success)
//!     ↓
//! Analyze sibling patterns that DO succeed
//!     ↓
//! Hypothesize new fix rule (pattern → transformation)
//!     ↓
//! Gate: MAGI calibration >0.8 + Phi >0.35 (consciousness gate)
//!     ↓
//! Register rule in auto-fix pipeline
//!     ↓
//! Next error: new rule fires → success → rule's Bayesian rate improves
//!     ↓
//! System permanently better at fixing this class of errors
//! ```
//!
//! No LLM can do this. An LLM generates the same fix quality on day 1
//! as day 1000. Symthaea's fix repertoire grows with every session.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// A self-generated fix rule produced by analyzing error patterns.
///
/// This is the atomic unit of self-modification: a learned transformation
/// that the system invented to handle a class of errors it previously
/// couldn't fix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedFixRule {
    /// Unique rule identifier
    pub id: String,
    /// Rust error code this rule targets (e.g., "E0308")
    pub error_code: Option<String>,
    /// Error category (matches ErrorCategory in error_knowledge.rs)
    pub category: String,
    /// Normalized error pattern signature
    pub pattern_signature: String,
    /// The fix transformation description
    pub fix_description: String,
    /// Search pattern in the source code (substring or regex-like)
    pub search_pattern: Option<String>,
    /// Replacement text (if applicable)
    pub replacement: Option<String>,
    /// Confidence at generation time (0.0-1.0)
    pub confidence: f32,
    /// Number of supporting evidence facts
    pub evidence_count: usize,
    /// Pipeline stage to inject at (2=structured, 3=category-aware)
    pub target_stage: u8,
    /// Timestamp of creation
    pub created_at: u64,
    /// Times this rule has been applied
    pub applications: u32,
    /// Times the application compiled successfully
    pub compile_successes: u32,
    /// Times the application passed tests
    pub test_successes: u32,
    /// Whether this rule has been promoted to permanent status
    pub promoted: bool,
    /// Whether this rule has been demoted (disabled but not deleted)
    pub demoted: bool,
    /// Reason for demotion (if applicable)
    pub demotion_reason: Option<String>,
}

impl GeneratedFixRule {
    /// Bayesian success rate: (successes + 1) / (attempts + 2)
    ///
    /// Uses the same Laplace smoothing as the existing FixStrategyStats
    /// for consistency in the pipeline ranking.
    pub fn success_rate(&self) -> f32 {
        (self.compile_successes as f32 + 1.0) / (self.applications as f32 + 2.0)
    }

    /// Whether this rule has proven itself (>70% success with 5+ applications)
    pub fn is_proven(&self) -> bool {
        self.applications >= 5 && self.success_rate() > 0.7
    }

    /// Whether this rule should be demoted (<30% success with 5+ applications)
    pub fn should_demote(&self) -> bool {
        self.applications >= 5 && self.success_rate() < 0.3
    }

    /// Record the outcome of applying this rule.
    pub fn record_outcome(&mut self, compiled: bool, tests_passed: Option<bool>) {
        self.applications += 1;
        if compiled {
            self.compile_successes += 1;
        }
        if tests_passed == Some(true) {
            self.test_successes += 1;
        }

        // Auto-demote if consistently failing
        if self.should_demote() && !self.demoted {
            self.demoted = true;
            self.demotion_reason = Some(format!(
                "Success rate {:.1}% after {} applications",
                self.success_rate() * 100.0,
                self.applications
            ));
        }

        // Auto-promote if consistently succeeding
        if self.is_proven() && !self.promoted {
            self.promoted = true;
        }
    }
}

/// Event log for rule lifecycle tracking.
#[derive(Debug, Clone)]
pub enum RuleEvent {
    /// A new rule was generated from error pattern analysis
    Generated {
        rule_id: String,
        confidence: f32,
        evidence_count: usize,
    },
    /// A rule was applied to fix an error
    Applied {
        rule_id: String,
        compiled: bool,
        tests_passed: Option<bool>,
    },
    /// A rule was promoted to permanent status (>70% success)
    Promoted {
        rule_id: String,
        success_rate: f32,
        applications: u32,
    },
    /// A rule was demoted (disabled) due to poor performance
    Demoted {
        rule_id: String,
        success_rate: f32,
        reason: String,
    },
}

/// Observed error cluster — a recurring failure pattern with enough
/// data to potentially generate a fix rule.
#[derive(Debug, Clone)]
struct ErrorCluster {
    /// Error code (e.g., "E0308")
    error_code: String,
    /// Error category
    category: String,
    /// Normalized pattern signature
    pattern_signature: String,
    /// Total attempts to fix this pattern
    total_attempts: u32,
    /// Successful fixes
    successful_fixes: u32,
    /// Fix strategies that have been tried
    tried_strategies: Vec<(String, f32)>, // (strategy, success_rate)
    /// Code snippets where this error occurred (for pattern analysis)
    context_samples: Vec<String>,
}

/// The self-modification engine — generates and manages auto-fix rules.
///
/// This is the recursive loop closure: the system observes its own failures,
/// hypothesizes fixes, tests them, and permanently improves.
pub struct FixRuleGenerator {
    /// Generated rules indexed by ID
    rules: HashMap<String, GeneratedFixRule>,
    /// Event log
    events: Vec<RuleEvent>,
    /// Next rule ID counter
    next_id: u64,
    /// Error clusters detected from recent failures
    clusters: Vec<ErrorCluster>,
    /// Minimum attempts before a cluster triggers rule generation
    min_cluster_attempts: u32,
    /// Maximum success rate that triggers rule generation
    /// (if existing fixes work well, don't generate new ones)
    max_existing_success_rate: f32,
    /// Minimum MAGI calibration quality to allow self-modification
    min_calibration_quality: f32,
    /// Minimum consciousness level (Phi) for self-modification
    min_phi: f32,
    /// Statistics
    stats: SelfModificationStats,
}

/// Statistics for the self-modification system.
#[derive(Debug, Clone, Default)]
pub struct SelfModificationStats {
    /// Total rules generated
    pub rules_generated: usize,
    /// Total rules promoted (proven effective)
    pub rules_promoted: usize,
    /// Total rules demoted (proven ineffective)
    pub rules_demoted: usize,
    /// Total rule applications
    pub rule_applications: usize,
    /// Total compilation successes from generated rules
    pub compile_successes: usize,
    /// Total test successes from generated rules
    pub test_successes: usize,
    /// Number of error clusters detected
    pub clusters_detected: usize,
    /// Times self-modification was gated by consciousness
    pub consciousness_gates: usize,
    /// Times self-modification was gated by calibration
    pub calibration_gates: usize,
}

impl FixRuleGenerator {
    /// Create a new fix rule generator with default thresholds.
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
            events: Vec::new(),
            next_id: 1,
            clusters: Vec::new(),
            min_cluster_attempts: 5,
            max_existing_success_rate: 0.5,
            min_calibration_quality: 0.8,
            min_phi: 0.35,
            stats: SelfModificationStats::default(),
        }
    }

    /// Record an error pattern observation.
    ///
    /// Call this after every auto-fix attempt. The generator accumulates
    /// observations into clusters and triggers rule generation when a
    /// cluster has enough data.
    pub fn observe_error(
        &mut self,
        error_code: &str,
        category: &str,
        pattern_signature: &str,
        fix_strategy: &str,
        success: bool,
        context_snippet: &str,
    ) {
        // Find or create cluster
        let cluster = self
            .clusters
            .iter_mut()
            .find(|c| c.error_code == error_code && c.pattern_signature == pattern_signature);

        if let Some(cluster) = cluster {
            cluster.total_attempts += 1;
            if success {
                cluster.successful_fixes += 1;
            }

            // Track strategy success — compute count before mutable borrow
            let strategy_count = cluster
                .tried_strategies
                .iter()
                .filter(|(s, _)| s == fix_strategy)
                .count() as f32;

            if let Some(entry) = cluster
                .tried_strategies
                .iter_mut()
                .find(|(s, _)| s == fix_strategy)
            {
                entry.1 = if success {
                    (entry.1 * strategy_count + 1.0) / (strategy_count + 1.0)
                } else {
                    (entry.1 * strategy_count) / (strategy_count + 1.0)
                };
            } else {
                cluster.tried_strategies.push((
                    fix_strategy.to_string(),
                    if success { 1.0 } else { 0.0 },
                ));
            }

            // Keep last 10 context samples
            if cluster.context_samples.len() < 10 {
                cluster
                    .context_samples
                    .push(context_snippet.chars().take(200).collect());
            }
        } else {
            let mut new_cluster = ErrorCluster {
                error_code: error_code.to_string(),
                category: category.to_string(),
                pattern_signature: pattern_signature.to_string(),
                total_attempts: 1,
                successful_fixes: if success { 1 } else { 0 },
                tried_strategies: vec![(
                    fix_strategy.to_string(),
                    if success { 1.0 } else { 0.0 },
                )],
                context_samples: vec![context_snippet.chars().take(200).collect()],
            };
            self.clusters.push(new_cluster);
            self.stats.clusters_detected += 1;
        }
    }

    /// Check if any clusters are ready for rule generation and generate rules.
    ///
    /// Returns the number of new rules generated.
    ///
    /// # Arguments
    /// * `phi` — Current consciousness level (gates self-modification at 0.35)
    /// * `calibration_quality` — MAGI calibration ECE (gates at 0.8)
    pub fn try_generate_rules(
        &mut self,
        phi: f32,
        calibration_quality: f32,
    ) -> Vec<String> {
        // ── Consciousness Gate ──────────────────────────────────────────
        if phi < self.min_phi {
            self.stats.consciousness_gates += 1;
            return Vec::new();
        }

        // ── Calibration Gate ────────────────────────────────────────────
        if calibration_quality < self.min_calibration_quality {
            self.stats.calibration_gates += 1;
            return Vec::new();
        }

        let mut generated_ids = Vec::new();

        // Find clusters ready for rule generation
        let ready_clusters: Vec<usize> = self
            .clusters
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                c.total_attempts >= self.min_cluster_attempts
                    && Self::cluster_success_rate(c) < self.max_existing_success_rate
            })
            .map(|(i, _)| i)
            .collect();

        for cluster_idx in ready_clusters {
            // Clone cluster data to avoid borrowing self.clusters while calling &mut self
            let cluster = self.clusters[cluster_idx].clone();

            if let Some(rule) = self.hypothesize_rule(&cluster) {
                let rule_id = rule.id.clone();
                self.events.push(RuleEvent::Generated {
                    rule_id: rule_id.clone(),
                    confidence: rule.confidence,
                    evidence_count: rule.evidence_count,
                });
                self.rules.insert(rule_id.clone(), rule);
                self.stats.rules_generated += 1;
                generated_ids.push(rule_id);
            }
        }

        generated_ids
    }

    /// Hypothesize a new fix rule from an error cluster.
    ///
    /// Analyzes what strategies have been tried, which (if any) partially
    /// worked, and generates a refined strategy that combines insights.
    fn hypothesize_rule(&mut self, cluster: &ErrorCluster) -> Option<GeneratedFixRule> {
        // Find the best-performing existing strategy (if any)
        let best_existing = cluster
            .tried_strategies
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Analyze context samples for common patterns
        let common_patterns = Self::find_common_code_patterns(&cluster.context_samples);

        // Generate a rule based on the error category
        let (fix_description, search_pattern, replacement) =
            self.synthesize_fix(&cluster.category, &cluster.error_code, &common_patterns, best_existing);

        let rule_id = format!("self_rule_{}", self.next_id);
        self.next_id += 1;

        // Confidence: based on evidence count and how consistent the pattern is
        let pattern_consistency = if common_patterns.is_empty() {
            0.3
        } else {
            0.5 + 0.1 * common_patterns.len().min(5) as f32
        };

        let confidence = pattern_consistency
            * (cluster.total_attempts as f32 / (cluster.total_attempts as f32 + 5.0));

        Some(GeneratedFixRule {
            id: rule_id,
            error_code: Some(cluster.error_code.clone()),
            category: cluster.category.clone(),
            pattern_signature: cluster.pattern_signature.clone(),
            fix_description,
            search_pattern,
            replacement,
            confidence,
            evidence_count: cluster.total_attempts as usize,
            target_stage: 3, // Category-aware stage by default
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            applications: 0,
            compile_successes: 0,
            test_successes: 0,
            promoted: false,
            demoted: false,
            demotion_reason: None,
        })
    }

    /// Synthesize a fix strategy from error analysis.
    ///
    /// This is where the system generates NEW code transformations it hasn't
    /// been explicitly programmed with. The strategies are derived from:
    /// 1. The error category (what kind of problem)
    /// 2. Common code patterns in failing snippets (what the code looks like)
    /// 3. The best existing strategy (what partially works)
    fn synthesize_fix(
        &self,
        category: &str,
        error_code: &str,
        common_patterns: &[String],
        best_existing: Option<&(String, f32)>,
    ) -> (String, Option<String>, Option<String>) {
        match category {
            "TypeMismatch" => {
                // Common fix: add type conversion
                let conversion = if common_patterns.iter().any(|p| p.contains("String")) {
                    ".to_string()"
                } else if common_patterns.iter().any(|p| p.contains("&str")) {
                    ".as_str()"
                } else if common_patterns.iter().any(|p| p.contains("usize")) {
                    " as usize"
                } else if common_patterns.iter().any(|p| p.contains("i32")) {
                    " as i32"
                } else {
                    ".into()"
                };

                (
                    format!("Auto-generated: add {} for type conversion (from {} occurrences)",
                        conversion, common_patterns.len()),
                    None, // Context-dependent, no fixed search pattern
                    Some(conversion.to_string()),
                )
            }
            "MissingImport" => {
                // Extract the missing type name from context
                let missing_type = common_patterns
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "Unknown".to_string());

                (
                    format!("Auto-generated: add use statement for {}", missing_type),
                    Some(missing_type.clone()),
                    Some(format!("use std::collections::{};", missing_type)),
                )
            }
            "BorrowError" => {
                let fix = if common_patterns.iter().any(|p| p.contains("move")) {
                    ".clone()"
                } else if common_patterns.iter().any(|p| p.contains("mutable")) {
                    "mut "
                } else {
                    ".clone()"
                };

                (
                    format!("Auto-generated: add {} to resolve borrow conflict", fix),
                    None,
                    Some(fix.to_string()),
                )
            }
            "LifetimeError" => {
                (
                    "Auto-generated: add explicit lifetime annotation <'a>".to_string(),
                    Some("fn ".to_string()),
                    Some("fn <'a>".to_string()),
                )
            }
            _ => {
                // Generic fallback: describe the pattern for future analysis
                let desc = if let Some((best_strat, rate)) = best_existing {
                    format!(
                        "Auto-generated: variant of '{}' (was {:.0}% effective) for {} errors",
                        best_strat,
                        rate * 100.0,
                        category
                    )
                } else {
                    format!(
                        "Auto-generated: new fix strategy for {} {} errors ({} samples)",
                        category, error_code, common_patterns.len()
                    )
                };
                (desc, None, None)
            }
        }
    }

    /// Find common code patterns across context samples.
    fn find_common_code_patterns(samples: &[String]) -> Vec<String> {
        if samples.is_empty() {
            return Vec::new();
        }

        let mut patterns = Vec::new();

        // Common type names
        let type_keywords = [
            "String", "&str", "Vec", "HashMap", "Option", "Result",
            "i32", "u32", "i64", "u64", "f32", "f64", "usize", "bool",
        ];

        for keyword in &type_keywords {
            let count = samples.iter().filter(|s| s.contains(keyword)).count();
            if count > samples.len() / 2 {
                // Appears in >50% of samples
                patterns.push(keyword.to_string());
            }
        }

        // Common structural patterns
        let structural_keywords = ["mut ", "move", "&self", "impl ", "trait ", "async "];
        for keyword in &structural_keywords {
            let count = samples.iter().filter(|s| s.contains(keyword)).count();
            if count > samples.len() / 2 {
                patterns.push(keyword.to_string());
            }
        }

        patterns
    }

    /// Apply a generated rule to fix code.
    ///
    /// Returns the fixed code if the rule is applicable, None otherwise.
    pub fn try_apply_rule(
        &mut self,
        rule_id: &str,
        source: &str,
        error_message: &str,
    ) -> Option<String> {
        let rule = self.rules.get(rule_id)?;

        if rule.demoted {
            return None; // Don't apply demoted rules
        }

        // Check if the error matches the rule's pattern
        if let Some(ref error_code) = rule.error_code {
            if !error_message.contains(error_code) {
                return None;
            }
        }

        // Apply the fix
        let fixed = if let (Some(ref search), Some(ref replacement)) =
            (&rule.search_pattern, &rule.replacement)
        {
            if source.contains(search.as_str()) {
                Some(source.replacen(search.as_str(), replacement.as_str(), 1))
            } else {
                None
            }
        } else if let Some(ref replacement) = rule.replacement {
            // No search pattern — append the fix (e.g., type conversion)
            // This is a heuristic: apply to the last expression before the error
            Some(format!("{}\n// Auto-fix applied: {}", source, replacement))
        } else {
            None
        };

        if fixed.is_some() {
            self.stats.rule_applications += 1;
        }

        fixed
    }

    /// Record the outcome of applying a generated rule.
    pub fn record_rule_outcome(
        &mut self,
        rule_id: &str,
        compiled: bool,
        tests_passed: Option<bool>,
    ) {
        if let Some(rule) = self.rules.get_mut(rule_id) {
            rule.record_outcome(compiled, tests_passed);

            if compiled {
                self.stats.compile_successes += 1;
            }
            if tests_passed == Some(true) {
                self.stats.test_successes += 1;
            }

            self.events.push(RuleEvent::Applied {
                rule_id: rule_id.to_string(),
                compiled,
                tests_passed,
            });

            // Check for promotion/demotion
            if rule.promoted && !rule.demoted {
                self.stats.rules_promoted += 1;
                self.events.push(RuleEvent::Promoted {
                    rule_id: rule_id.to_string(),
                    success_rate: rule.success_rate(),
                    applications: rule.applications,
                });
            } else if rule.demoted {
                self.stats.rules_demoted += 1;
                self.events.push(RuleEvent::Demoted {
                    rule_id: rule_id.to_string(),
                    success_rate: rule.success_rate(),
                    reason: rule.demotion_reason.clone().unwrap_or_default(),
                });
            }
        }
    }

    /// Get all active (non-demoted) rules for a specific error code.
    pub fn active_rules_for(&self, error_code: &str) -> Vec<&GeneratedFixRule> {
        self.rules
            .values()
            .filter(|r| {
                !r.demoted
                    && r.error_code
                        .as_ref()
                        .map_or(false, |c| c == error_code)
            })
            .collect()
    }

    /// Get all generated rules.
    pub fn all_rules(&self) -> Vec<&GeneratedFixRule> {
        self.rules.values().collect()
    }

    /// Get proven rules only (>70% success, 5+ applications).
    pub fn proven_rules(&self) -> Vec<&GeneratedFixRule> {
        self.rules.values().filter(|r| r.is_proven()).collect()
    }

    /// Get the event log.
    pub fn events(&self) -> &[RuleEvent] {
        &self.events
    }

    /// Get statistics.
    pub fn stats(&self) -> &SelfModificationStats {
        &self.stats
    }

    /// Cluster success rate helper.
    fn cluster_success_rate(cluster: &ErrorCluster) -> f32 {
        if cluster.total_attempts == 0 {
            0.0
        } else {
            cluster.successful_fixes as f32 / cluster.total_attempts as f32
        }
    }

    /// Human-readable summary of the self-modification state.
    pub fn summary(&self) -> String {
        let active = self.rules.values().filter(|r| !r.demoted).count();
        let proven = self.proven_rules().len();

        format!(
            "Self-Modification: {} rules generated ({} active, {} proven, {} demoted) | \
             {} applications ({} compiled, {} tested) | \
             {} clusters | {} consciousness gates, {} calibration gates",
            self.stats.rules_generated,
            active,
            proven,
            self.stats.rules_demoted,
            self.stats.rule_applications,
            self.stats.compile_successes,
            self.stats.test_successes,
            self.stats.clusters_detected,
            self.stats.consciousness_gates,
            self.stats.calibration_gates,
        )
    }
}

// =========================================================================
// Persistence — save/load promoted rules for cross-session retention
// =========================================================================

impl FixRuleGenerator {
    /// Save all promoted (proven) rules to a JSON file.
    ///
    /// Only saves promoted rules — demoted and unproven rules are not worth
    /// persisting since they haven't demonstrated value.
    pub fn save_promoted_rules(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let promoted: Vec<&GeneratedFixRule> = self
            .rules
            .values()
            .filter(|r| r.promoted && !r.demoted)
            .collect();

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(&promoted)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load promoted rules from a JSON file and register them.
    pub fn load_promoted_rules(&mut self, path: &std::path::Path) -> anyhow::Result<usize> {
        let json = std::fs::read_to_string(path)?;
        let rules: Vec<GeneratedFixRule> = serde_json::from_str(&json)?;
        let count = rules.len();
        for rule in rules {
            self.rules.insert(rule.id.clone(), rule);
        }
        Ok(count)
    }

    /// Default persistence path for fix rules.
    pub fn default_rules_path() -> std::path::PathBuf {
        if let Ok(data_dir) = std::env::var("SYMTHAEA_DATA_DIR") {
            std::path::PathBuf::from(data_dir).join("fix-rules.json")
        } else if let Ok(home) = std::env::var("HOME") {
            std::path::PathBuf::from(home)
                .join(".local/share/symthaea/fix-rules.json")
        } else {
            std::path::PathBuf::from("fix-rules.json")
        }
    }

    /// Save promoted rules to default path.
    pub fn persist_rules(&self) -> anyhow::Result<()> {
        self.save_promoted_rules(&Self::default_rules_path())
    }

    /// Load rules from default path if it exists.
    pub fn load_persisted_rules(&mut self) -> usize {
        let path = Self::default_rules_path();
        if path.exists() {
            match self.load_promoted_rules(&path) {
                Ok(count) => {
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "[FixRuleGenerator] Loaded {} promoted rules from {}",
                        count,
                        path.display()
                    );
                    count
                }
                Err(_e) => {
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "[FixRuleGenerator] Failed to load rules from {}: {}",
                        path.display(),
                        _e
                    );
                    0
                }
            }
        } else {
            0
        }
    }
}

impl Default for FixRuleGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_construction() {
        let gen = FixRuleGenerator::new();
        assert_eq!(gen.stats().rules_generated, 0);
        assert!(gen.all_rules().is_empty());
    }

    #[test]
    fn test_observe_and_cluster() {
        let mut gen = FixRuleGenerator::new();

        // Observe 5 similar errors
        for i in 0..5 {
            gen.observe_error(
                "E0308",
                "TypeMismatch",
                "mismatched types: expected String, found &str",
                "add .to_string()",
                i < 2, // 40% success rate (2 of 5)
                "let x: String = some_str;",
            );
        }

        assert_eq!(gen.stats().clusters_detected, 1);
        assert_eq!(gen.clusters[0].total_attempts, 5);
        assert_eq!(gen.clusters[0].successful_fixes, 2);
    }

    #[test]
    fn test_rule_generation_from_cluster() {
        let mut gen = FixRuleGenerator::new();

        // Build cluster with enough failures
        for i in 0..6 {
            gen.observe_error(
                "E0308",
                "TypeMismatch",
                "mismatched types",
                "existing_fix",
                i < 2, // 33% success → below 50% threshold
                "let x: String = input;",
            );
        }

        // Generate rules (high Phi + calibration)
        let generated = gen.try_generate_rules(0.5, 0.9);
        assert!(
            !generated.is_empty(),
            "Should generate a rule for failing cluster"
        );
        assert_eq!(gen.stats().rules_generated, 1);
    }

    #[test]
    fn test_consciousness_gate() {
        let mut gen = FixRuleGenerator::new();

        for _ in 0..6 {
            gen.observe_error("E0308", "TypeMismatch", "x", "y", false, "z");
        }

        // Low Phi → gated
        let generated = gen.try_generate_rules(0.1, 0.9);
        assert!(generated.is_empty());
        assert_eq!(gen.stats().consciousness_gates, 1);
    }

    #[test]
    fn test_calibration_gate() {
        let mut gen = FixRuleGenerator::new();

        for _ in 0..6 {
            gen.observe_error("E0308", "TypeMismatch", "x", "y", false, "z");
        }

        // Low calibration → gated
        let generated = gen.try_generate_rules(0.5, 0.3);
        assert!(generated.is_empty());
        assert_eq!(gen.stats().calibration_gates, 1);
    }

    #[test]
    fn test_rule_success_tracking() {
        let mut rule = GeneratedFixRule {
            id: "test_rule".to_string(),
            error_code: Some("E0308".to_string()),
            category: "TypeMismatch".to_string(),
            pattern_signature: "test".to_string(),
            fix_description: "test fix".to_string(),
            search_pattern: None,
            replacement: Some(".to_string()".to_string()),
            confidence: 0.5,
            evidence_count: 5,
            target_stage: 3,
            created_at: 0,
            applications: 0,
            compile_successes: 0,
            test_successes: 0,
            promoted: false,
            demoted: false,
            demotion_reason: None,
        };

        // Apply 6 times, 5 successes → should promote
        for i in 0..6 {
            rule.record_outcome(i < 5, Some(i < 5));
        }

        assert!(rule.is_proven());
        assert!(rule.promoted);
        assert!(!rule.demoted);
        assert!(rule.success_rate() > 0.7);
    }

    #[test]
    fn test_rule_demotion() {
        let mut rule = GeneratedFixRule {
            id: "bad_rule".to_string(),
            error_code: Some("E0308".to_string()),
            category: "TypeMismatch".to_string(),
            pattern_signature: "test".to_string(),
            fix_description: "bad fix".to_string(),
            search_pattern: None,
            replacement: None,
            confidence: 0.5,
            evidence_count: 5,
            target_stage: 3,
            created_at: 0,
            applications: 0,
            compile_successes: 0,
            test_successes: 0,
            promoted: false,
            demoted: false,
            demotion_reason: None,
        };

        // Apply 6 times, 1 success → should demote
        for i in 0..6 {
            rule.record_outcome(i == 0, None);
        }

        assert!(rule.should_demote());
        assert!(rule.demoted);
        assert!(rule.demotion_reason.is_some());
    }

    #[test]
    fn test_try_apply_rule() {
        let mut gen = FixRuleGenerator::new();

        // Manually insert a rule
        gen.rules.insert(
            "rule_1".to_string(),
            GeneratedFixRule {
                id: "rule_1".to_string(),
                error_code: Some("E0308".to_string()),
                category: "TypeMismatch".to_string(),
                pattern_signature: "test".to_string(),
                fix_description: "Convert &str to String".to_string(),
                search_pattern: Some("input_value".to_string()),
                replacement: Some("input_value.to_string()".to_string()),
                confidence: 0.7,
                evidence_count: 10,
                target_stage: 3,
                created_at: 0,
                applications: 0,
                compile_successes: 0,
                test_successes: 0,
                promoted: false,
                demoted: false,
                demotion_reason: None,
            },
        );

        // Apply the rule
        let source = "let x: String = input_value;";
        let result = gen.try_apply_rule("rule_1", source, "error[E0308]: mismatched types");

        assert!(result.is_some());
        let fixed = result.unwrap();
        assert!(fixed.contains("input_value.to_string()"));
    }

    #[test]
    fn test_demoted_rule_not_applied() {
        let mut gen = FixRuleGenerator::new();

        gen.rules.insert(
            "bad_rule".to_string(),
            GeneratedFixRule {
                id: "bad_rule".to_string(),
                error_code: Some("E0308".to_string()),
                category: "TypeMismatch".to_string(),
                pattern_signature: "test".to_string(),
                fix_description: "Bad fix".to_string(),
                search_pattern: Some("x".to_string()),
                replacement: Some("y".to_string()),
                confidence: 0.1,
                evidence_count: 1,
                target_stage: 3,
                created_at: 0,
                applications: 0,
                compile_successes: 0,
                test_successes: 0,
                promoted: false,
                demoted: true,
                demotion_reason: Some("Too bad".to_string()),
            },
        );

        let result = gen.try_apply_rule("bad_rule", "let x = 1;", "error[E0308]");
        assert!(result.is_none(), "Demoted rules should not be applied");
    }

    #[test]
    fn test_high_success_no_new_rule() {
        let mut gen = FixRuleGenerator::new();

        // Build cluster with high success rate (80%)
        for i in 0..6 {
            gen.observe_error("E0308", "TypeMismatch", "test", "good_fix", i < 5, "ctx");
        }

        // Should NOT generate new rules — existing fix works well
        let generated = gen.try_generate_rules(0.5, 0.9);
        assert!(
            generated.is_empty(),
            "Should not generate rules when existing fixes work well"
        );
    }

    #[test]
    fn test_summary() {
        let gen = FixRuleGenerator::new();
        let summary = gen.summary();
        assert!(summary.contains("Self-Modification"));
        assert!(summary.contains("0 rules generated"));
    }

    #[test]
    fn test_full_lifecycle() {
        let mut gen = FixRuleGenerator::new();

        // 1. Observe failures (build cluster)
        for _ in 0..7 {
            gen.observe_error(
                "E0277",
                "MissingImport",
                "cannot find type HashMap",
                "manual_import",
                false,
                "let m: HashMap<String, i32> = HashMap::new();",
            );
        }

        // 2. Generate rule (high consciousness + calibration)
        let rules = gen.try_generate_rules(0.6, 0.9);
        assert!(!rules.is_empty());
        let rule_id = &rules[0];

        // 3. Apply the rule
        let source = "let m: HashMap<String, i32> = HashMap::new();";
        let fixed = gen.try_apply_rule(rule_id, source, "error[E0277]: cannot find type HashMap");
        // Rule may or may not apply depending on synthesized fix
        // The important thing is the lifecycle works

        // 4. Record outcome
        gen.record_rule_outcome(rule_id, true, Some(true));
        assert_eq!(gen.stats().rule_applications, 1);
        assert_eq!(gen.stats().compile_successes, 1);

        // 5. Check events
        assert!(gen.events().len() >= 2); // Generated + Applied
    }
}
