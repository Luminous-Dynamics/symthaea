// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Code Error Knowledge Graph
//!
//! Semantic knowledge about compiler errors and fix strategies.
//! Complements the flat `CodingExperienceStore` with structured
//! error → fix mappings that include success rates and context.
//!
//! # Architecture
//!
//! ```text
//! CompileError → CodeErrorFact → CodeErrorKnowledge (graph)
//!                                     ↓
//!                        suggest_fix(error_code, context) → ranked strategies
//! ```
//!
//! Science: Case-Based Reasoning (Kolodner 1993) — retrieve similar past
//! problems, adapt their solutions to the current context.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// A recorded fact about a compiler error and the strategy that fixed it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeErrorFact {
    /// Rust error code (e.g., "E0308", "E0277")
    pub error_code: String,
    /// Error category for coarser matching
    pub category: ErrorCategory,
    /// Normalized error pattern (type names stripped)
    pub pattern_signature: String,
    /// The fix strategy that was applied
    pub fix_strategy: String,
    /// Whether the fix compiled successfully
    pub compiled: bool,
    /// Whether the fix passed tests (None if not tested)
    pub tests_passed: Option<bool>,
    /// Source code context (first 200 chars around the error)
    pub context_snippet: String,
}

/// Coarse error categories for similarity matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorCategory {
    TypeMismatch,
    MissingImport,
    BorrowError,
    LifetimeError,
    MissingImpl,
    UnusedCode,
    SyntaxError,
    UndeclaredGeneric,
    Other,
}

impl ErrorCategory {
    /// Parse from a rustc error code string.
    pub fn from_error_code(code: &str) -> Self {
        match code {
            "E0308" | "E0277" => Self::TypeMismatch,
            "E0433" | "E0432" => Self::MissingImport,
            "E0382" | "E0502" | "E0505" | "E0596" | "E0507" => Self::BorrowError,
            "E0106" | "E0621" => Self::LifetimeError,
            "E0412" => Self::UndeclaredGeneric,
            "E0601" => Self::SyntaxError,
            _ if code.starts_with("E02") => Self::MissingImpl,
            _ => Self::Other,
        }
    }

    /// Parse from a category string (for interop with code_executor).
    pub fn from_category_str(cat: &str) -> Self {
        match cat {
            "TypeMismatch" => Self::TypeMismatch,
            "MissingImport" => Self::MissingImport,
            "BorrowError" => Self::BorrowError,
            "LifetimeError" => Self::LifetimeError,
            "MissingImpl" => Self::MissingImpl,
            "UnusedCode" => Self::UnusedCode,
            "SyntaxError" => Self::SyntaxError,
            "UndeclaredGeneric" => Self::UndeclaredGeneric,
            _ => Self::Other,
        }
    }
}

/// A fix strategy with its success statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixStrategyStats {
    /// Strategy description (e.g., "add .into()", "add use statement")
    pub strategy: String,
    /// Number of times this strategy was applied
    pub attempts: u32,
    /// Number of times it compiled successfully
    pub compile_successes: u32,
    /// Number of times it passed tests
    pub test_successes: u32,
    /// Bayesian success rate: (successes + 1) / (attempts + 2)
    pub success_rate: f32,
}

impl FixStrategyStats {
    fn new(strategy: &str) -> Self {
        Self {
            strategy: strategy.to_string(),
            attempts: 0,
            compile_successes: 0,
            test_successes: 0,
            success_rate: 0.5, // prior
        }
    }

    fn record(&mut self, compiled: bool, tests_passed: Option<bool>) {
        self.attempts += 1;
        if compiled {
            self.compile_successes += 1;
        }
        if tests_passed == Some(true) {
            self.test_successes += 1;
        }
        // Bayesian: (successes + 1) / (attempts + 2)
        self.success_rate = (self.compile_successes as f32 + 1.0) / (self.attempts as f32 + 2.0);
    }
}

/// Serializable snapshot of the knowledge graph for persistence.
#[derive(Serialize, Deserialize)]
struct KnowledgeSnapshot {
    facts: Vec<CodeErrorFact>,
    total_facts: u64,
}

/// In-memory knowledge graph of error patterns → fix strategies.
///
/// Indexed by (error_code, category) for fast lookup. Each entry
/// maintains a ranked list of fix strategies with Bayesian success rates.
#[derive(Debug, Clone, Default)]
pub struct CodeErrorKnowledge {
    /// Error code → ranked fix strategies
    by_code: HashMap<String, Vec<FixStrategyStats>>,
    /// Error category → ranked fix strategies (coarser fallback)
    by_category: HashMap<ErrorCategory, Vec<FixStrategyStats>>,
    /// Total facts recorded
    total_facts: u64,
    /// All recorded facts (ring buffer, max 500)
    facts: Vec<CodeErrorFact>,
}

impl CodeErrorKnowledge {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a compiler error and the fix strategy that was applied.
    pub fn record_fix(&mut self, fact: CodeErrorFact) {
        // Update by-code stats
        let code_strategies = self.by_code.entry(fact.error_code.clone()).or_default();
        Self::update_strategy_list(
            code_strategies,
            &fact.fix_strategy,
            fact.compiled,
            fact.tests_passed,
        );

        // Update by-category stats
        let cat_strategies = self.by_category.entry(fact.category).or_default();
        Self::update_strategy_list(
            cat_strategies,
            &fact.fix_strategy,
            fact.compiled,
            fact.tests_passed,
        );

        // Ring buffer
        if self.facts.len() >= 500 {
            self.facts.remove(0);
        }
        self.facts.push(fact);
        self.total_facts += 1;
    }

    fn update_strategy_list(
        strategies: &mut Vec<FixStrategyStats>,
        strategy_name: &str,
        compiled: bool,
        tests_passed: Option<bool>,
    ) {
        if let Some(existing) = strategies.iter_mut().find(|s| s.strategy == strategy_name) {
            existing.record(compiled, tests_passed);
        } else {
            let mut stats = FixStrategyStats::new(strategy_name);
            stats.record(compiled, tests_passed);
            strategies.push(stats);
        }
        // Sort by success rate descending
        strategies.sort_by(|a, b| {
            b.success_rate
                .partial_cmp(&a.success_rate)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Suggest fix strategies for an error code, ranked by success rate.
    ///
    /// Returns strategies from both exact code match and category fallback,
    /// with exact matches ranked higher.
    pub fn suggest_fixes(
        &self,
        error_code: &str,
        category: ErrorCategory,
    ) -> Vec<&FixStrategyStats> {
        let mut suggestions: Vec<&FixStrategyStats> = Vec::new();

        // Exact code match (highest priority)
        if let Some(code_strategies) = self.by_code.get(error_code) {
            suggestions.extend(code_strategies.iter());
        }

        // Category fallback (lower priority, only if not already suggested)
        if let Some(cat_strategies) = self.by_category.get(&category) {
            for strategy in cat_strategies {
                if !suggestions.iter().any(|s| s.strategy == strategy.strategy) {
                    suggestions.push(strategy);
                }
            }
        }

        suggestions
    }

    /// Get the best fix strategy for an error code (if any with >50% success rate).
    pub fn best_fix(&self, error_code: &str, category: ErrorCategory) -> Option<&str> {
        let suggestions = self.suggest_fixes(error_code, category);
        suggestions
            .first()
            .filter(|s| s.success_rate > 0.5 && s.attempts >= 2)
            .map(|s| s.strategy.as_str())
    }

    /// Access the recent facts ring buffer (for backfill queries).
    pub fn recent_facts(&self) -> &[CodeErrorFact] {
        &self.facts
    }

    /// Total facts recorded.
    pub fn total_facts(&self) -> u64 {
        self.total_facts
    }

    /// Number of distinct error codes seen.
    pub fn distinct_codes(&self) -> usize {
        self.by_code.len()
    }

    /// Clear all knowledge (for testing).
    pub fn clear(&mut self) {
        self.by_code.clear();
        self.by_category.clear();
        self.facts.clear();
        self.total_facts = 0;
    }

    // =========================================================================
    // Persistence — save/load to JSON for cross-session knowledge retention
    // =========================================================================

    /// Save knowledge to a JSON file. Creates parent directories if needed.
    pub fn save_to_json(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let snapshot = KnowledgeSnapshot {
            facts: self.facts.clone(),
            total_facts: self.total_facts,
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(&snapshot)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load knowledge from a JSON file, replaying all facts to rebuild indices.
    pub fn load_from_json(path: &std::path::Path) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let snapshot: KnowledgeSnapshot = serde_json::from_str(&json)?;

        let mut knowledge = Self::new();
        for fact in snapshot.facts {
            knowledge.record_fix(fact);
        }
        // Restore total_facts counter (may be higher than facts.len() due to ring buffer)
        knowledge.total_facts = snapshot.total_facts;
        Ok(knowledge)
    }

    /// Default persistence path for error knowledge.
    pub fn default_path() -> std::path::PathBuf {
        if let Ok(data_dir) = std::env::var("SYMTHAEA_DATA_DIR") {
            std::path::PathBuf::from(data_dir).join("error-knowledge.json")
        } else if let Ok(home) = std::env::var("HOME") {
            std::path::PathBuf::from(home).join(".local/share/symthaea/error-knowledge.json")
        } else {
            std::path::PathBuf::from("error-knowledge.json")
        }
    }

    /// Save to the default path. Creates directories as needed.
    pub fn persist(&self) -> anyhow::Result<()> {
        self.save_to_json(&Self::default_path())
    }

    /// Load from the default path if it exists. Returns empty knowledge if not found.
    pub fn load_or_default() -> Self {
        let path = Self::default_path();
        if path.exists() {
            match Self::load_from_json(&path) {
                Ok(k) => {
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "[CodeErrorKnowledge] Loaded {} facts from {}",
                        k.facts.len(),
                        path.display()
                    );
                    k
                }
                Err(_e) => {
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "[CodeErrorKnowledge] Failed to load from {}: {}",
                        path.display(),
                        _e
                    );
                    Self::new()
                }
            }
        } else {
            Self::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_suggest() {
        let mut knowledge = CodeErrorKnowledge::new();

        // Record a successful fix for E0308
        knowledge.record_fix(CodeErrorFact {
            error_code: "E0308".to_string(),
            category: ErrorCategory::TypeMismatch,
            pattern_signature: "expected i32, found f64".to_string(),
            fix_strategy: "add .into()".to_string(),
            compiled: true,
            tests_passed: Some(true),
            context_snippet: "let x: i32 = y;".to_string(),
        });

        let suggestions = knowledge.suggest_fixes("E0308", ErrorCategory::TypeMismatch);
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0].strategy, "add .into()");
        assert!(suggestions[0].success_rate > 0.5);
    }

    #[test]
    fn test_bayesian_ranking() {
        let mut knowledge = CodeErrorKnowledge::new();

        // Strategy A: 3 successes out of 3
        for _ in 0..3 {
            knowledge.record_fix(CodeErrorFact {
                error_code: "E0308".to_string(),
                category: ErrorCategory::TypeMismatch,
                pattern_signature: "type mismatch".to_string(),
                fix_strategy: "add .into()".to_string(),
                compiled: true,
                tests_passed: Some(true),
                context_snippet: String::new(),
            });
        }

        // Strategy B: 1 success out of 3
        for i in 0..3 {
            knowledge.record_fix(CodeErrorFact {
                error_code: "E0308".to_string(),
                category: ErrorCategory::TypeMismatch,
                pattern_signature: "type mismatch".to_string(),
                fix_strategy: "add as i32".to_string(),
                compiled: i == 0,
                tests_passed: if i == 0 { Some(true) } else { Some(false) },
                context_snippet: String::new(),
            });
        }

        let suggestions = knowledge.suggest_fixes("E0308", ErrorCategory::TypeMismatch);
        assert_eq!(suggestions.len(), 2);
        // "add .into()" should rank higher (4/5=0.8 vs 2/5=0.4)
        assert_eq!(suggestions[0].strategy, "add .into()");
    }

    #[test]
    fn test_category_fallback() {
        let mut knowledge = CodeErrorKnowledge::new();

        // Record fix under E0382 (borrow error)
        knowledge.record_fix(CodeErrorFact {
            error_code: "E0382".to_string(),
            category: ErrorCategory::BorrowError,
            pattern_signature: "use of moved value".to_string(),
            fix_strategy: "add .clone()".to_string(),
            compiled: true,
            tests_passed: Some(true),
            context_snippet: String::new(),
        });

        // Query E0505 (different borrow error) — should get category fallback
        let suggestions = knowledge.suggest_fixes("E0505", ErrorCategory::BorrowError);
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0].strategy, "add .clone()");
    }

    #[test]
    fn test_best_fix_threshold() {
        let mut knowledge = CodeErrorKnowledge::new();

        // One attempt only — not enough evidence
        knowledge.record_fix(CodeErrorFact {
            error_code: "E0308".to_string(),
            category: ErrorCategory::TypeMismatch,
            pattern_signature: "type mismatch".to_string(),
            fix_strategy: "add .into()".to_string(),
            compiled: true,
            tests_passed: Some(true),
            context_snippet: String::new(),
        });

        // best_fix requires >= 2 attempts
        assert!(knowledge
            .best_fix("E0308", ErrorCategory::TypeMismatch)
            .is_none());

        // Second attempt
        knowledge.record_fix(CodeErrorFact {
            error_code: "E0308".to_string(),
            category: ErrorCategory::TypeMismatch,
            pattern_signature: "type mismatch".to_string(),
            fix_strategy: "add .into()".to_string(),
            compiled: true,
            tests_passed: Some(true),
            context_snippet: String::new(),
        });

        assert_eq!(
            knowledge.best_fix("E0308", ErrorCategory::TypeMismatch),
            Some("add .into()")
        );
    }

    #[test]
    fn test_error_category_from_code() {
        assert_eq!(
            ErrorCategory::from_error_code("E0308"),
            ErrorCategory::TypeMismatch
        );
        assert_eq!(
            ErrorCategory::from_error_code("E0382"),
            ErrorCategory::BorrowError
        );
        assert_eq!(
            ErrorCategory::from_error_code("E0106"),
            ErrorCategory::LifetimeError
        );
        assert_eq!(
            ErrorCategory::from_error_code("E0433"),
            ErrorCategory::MissingImport
        );
        assert_eq!(
            ErrorCategory::from_error_code("E9999"),
            ErrorCategory::Other
        );
    }

    #[test]
    fn test_ring_buffer_cap() {
        let mut knowledge = CodeErrorKnowledge::new();
        for i in 0..600 {
            knowledge.record_fix(CodeErrorFact {
                error_code: format!("E{:04}", i % 100),
                category: ErrorCategory::Other,
                pattern_signature: format!("pattern {}", i),
                fix_strategy: "some fix".to_string(),
                compiled: true,
                tests_passed: None,
                context_snippet: String::new(),
            });
        }
        assert!(knowledge.facts.len() <= 500);
        assert_eq!(knowledge.total_facts, 600);
    }
}
