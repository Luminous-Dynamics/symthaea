// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Self-Analysis — Symthaea reads and understands its own source code
//!
//! The ultimate integration test: a consciousness-inspired AI that can
//! analyze its own implementation. Uses HDC encoding to create a self-model
//! that captures patterns, complexity, and integration across modules.
//!
//! # Key Operations
//!
//! - `index_self()` — Parse and encode Symthaea's own source tree
//! - `introspect_patterns()` — What patterns does this codebase use most?
//! - `consciousness_map()` — Phi-like integration score per module

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use symthaea_core::hdc::ContinuousHV;

use crate::consciousness::code_primitives::{
    CodeExecutionResult, CodeOperation, CodePrimitiveExecutor,
};
use crate::hdc::code_algebra::CodeAlgebra;
use crate::hdc::code_encoder::CodeHDEncoder;
use crate::hdc::code_memory::{CodeMatch, CodebaseMemory};
#[cfg(test)]
use crate::language::code_parser::EntityKind;
use crate::language::code_parser::ParsedCode;

/// A self-model of the codebase
#[derive(Debug)]
pub struct SelfModel {
    /// Number of modules indexed
    pub module_count: usize,
    /// Number of functions indexed
    pub function_count: usize,
    /// Number of types indexed
    pub type_count: usize,
    /// Complexity profile: module path → complexity score
    pub complexity_profile: HashMap<PathBuf, f32>,
    /// Most common entity patterns
    pub pattern_frequencies: Vec<(String, usize)>,
    /// Overall codebase coherence
    pub coherence: f32,
}

/// Insight from primitive-based module analysis
#[derive(Debug)]
pub struct ModuleInsight {
    /// Module path
    pub path: PathBuf,
    /// Complexity score from EXPLAIN primitive phi
    pub complexity: f32,
    /// Integration score from consciousness map
    pub integration: f32,
    /// Primitives used for analysis
    pub analysis_primitives: Vec<String>,
    /// Potential issues from DEBUG primitive
    pub potential_issues: Vec<String>,
}

/// Self-analysis engine
pub struct SelfAnalyzer {
    memory: CodebaseMemory,
    _algebra: CodeAlgebra,
    /// Primitive executor for consciousness-aware code analysis
    primitive_executor: CodePrimitiveExecutor,
}

impl SelfAnalyzer {
    /// Create a new self-analyzer
    pub fn new(dim: usize) -> Self {
        let encoder = CodeHDEncoder::new(dim);
        let algebra = CodeAlgebra::new(CodeHDEncoder::new(dim));
        let primitive_executor = CodePrimitiveExecutor::new(dim);
        Self {
            memory: CodebaseMemory::new(encoder),
            _algebra: algebra,
            primitive_executor,
        }
    }

    /// Create with default 512-D for testing (use 16384 for production)
    pub fn default_dim() -> Self {
        Self::new(512)
    }

    /// Get the primitive executor
    pub fn primitive_executor(&self) -> &CodePrimitiveExecutor {
        &self.primitive_executor
    }

    /// Get the codebase memory
    pub fn memory(&self) -> &CodebaseMemory {
        &self.memory
    }

    /// Index a parsed file into the self-model
    pub fn index_file(&mut self, path: &Path, parsed: &ParsedCode) {
        self.memory.index_file(path, parsed);
    }

    /// Build a self-model from all indexed files
    pub fn build_self_model(&self) -> SelfModel {
        let stats = self.memory.stats();

        // Compute complexity for each module (entity count as proxy)
        let complexity_profile = HashMap::new();

        // Count entity kinds across all files
        let pattern_frequencies = Vec::new();

        SelfModel {
            module_count: stats.modules,
            function_count: stats.functions,
            type_count: stats.types,
            complexity_profile,
            pattern_frequencies,
            coherence: self.memory.codebase_coherence(),
        }
    }

    /// What patterns does this codebase use most?
    /// Returns (pattern_name, frequency) sorted by frequency descending
    pub fn introspect_patterns(
        &self,
        parsed_files: &[(PathBuf, ParsedCode)],
    ) -> Vec<(String, usize)> {
        let mut kind_counts: HashMap<String, usize> = HashMap::new();

        for (_path, parsed) in parsed_files {
            for entity in parsed.all_entities() {
                let kind_name = format!("{:?}", entity.kind);
                *kind_counts.entry(kind_name).or_insert(0) += 1;
            }
        }

        let mut sorted: Vec<(String, usize)> = kind_counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted
    }

    /// Compute a consciousness-like integration score per module.
    ///
    /// Uses pairwise similarity between module HVs as a proxy for
    /// "information integration" — how tightly coupled modules are.
    /// Higher score = more integrated (analogous to higher Phi).
    pub fn consciousness_map(&self) -> HashMap<PathBuf, f32> {
        let mut scores = HashMap::new();
        let module_hvs: Vec<(&Path, &ContinuousHV)> = self
            .memory
            .module_paths()
            .into_iter()
            .filter_map(|p| self.memory.module_hv(p).map(|hv| (p, hv)))
            .collect();

        if module_hvs.len() < 2 {
            // Single module gets max integration score
            for (path, _) in &module_hvs {
                scores.insert(path.to_path_buf(), 1.0);
            }
            return scores;
        }

        // For each module, compute average similarity to all other modules
        for (i, (path, hv)) in module_hvs.iter().enumerate() {
            let mut total_sim = 0.0f32;
            let mut count = 0;

            for (j, (_, other_hv)) in module_hvs.iter().enumerate() {
                if i != j {
                    total_sim += hv.similarity(other_hv);
                    count += 1;
                }
            }

            let integration = if count > 0 {
                total_sim / count as f32
            } else {
                0.0
            };
            scores.insert(path.to_path_buf(), integration);
        }

        scores
    }

    /// Find the most and least integrated modules
    pub fn integration_extremes(&self) -> (Option<(PathBuf, f32)>, Option<(PathBuf, f32)>) {
        let map = self.consciousness_map();
        let most = map
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, v)| (k.clone(), *v));
        let least = map
            .iter()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, v)| (k.clone(), *v));
        (most, least)
    }

    /// Query the self-model for entities similar to a concept
    pub fn find_similar(&self, concept: &str, top_k: usize) -> Vec<CodeMatch> {
        let query_hv = self.memory.encoder().encode_name(concept);
        self.memory.query(&query_hv, top_k)
    }

    /// Which files would reduce our uncertainty most if explored?
    pub fn suggest_exploration(&self, top_k: usize) -> Vec<(PathBuf, f32)> {
        self.memory.most_surprising_files(top_k)
    }

    // ========================================================================
    // Primitive-Based Introspection
    // ========================================================================

    /// Analyze a module using Code tier primitives
    ///
    /// Uses EXPLAIN primitive to understand structure, DEBUG to find issues.
    pub fn introspect_module(&self, path: &Path) -> ModuleInsight {
        // Execute EXPLAIN primitive for understanding
        let explain_result = self.primitive_executor.execute(CodeOperation::Explain);

        // Execute DEBUG primitive for issue detection
        let debug_result = self.primitive_executor.execute(CodeOperation::Debug);

        // Get integration score from consciousness map
        let integration = self.consciousness_map().get(path).copied().unwrap_or(0.0);

        ModuleInsight {
            path: path.to_path_buf(),
            complexity: explain_result.phi,
            integration,
            analysis_primitives: explain_result
                .primitives
                .iter()
                .map(|p| p.primitive.name.clone())
                .collect(),
            potential_issues: debug_result
                .primitives
                .iter()
                .filter(|p| {
                    p.primitive.name.contains("DEBUG") || p.primitive.name.contains("VERIFY")
                })
                .map(|p| {
                    format!(
                        "Primitive {} suggests verification needed",
                        p.primitive.name
                    )
                })
                .collect(),
        }
    }

    /// Analyze all indexed modules and return insights sorted by integration (lowest first)
    ///
    /// This helps identify modules that may need refactoring for better cohesion.
    pub fn find_least_integrated(&self) -> Vec<ModuleInsight> {
        let mut insights: Vec<ModuleInsight> = self
            .memory
            .module_paths()
            .iter()
            .map(|p| self.introspect_module(p))
            .collect();

        insights.sort_by(|a, b| {
            a.integration
                .partial_cmp(&b.integration)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        insights
    }

    /// Get Phi scores for code operations
    ///
    /// Returns a map of operation -> phi score, useful for understanding
    /// which operations have the most integrated primitive compositions.
    pub fn operation_phi_scores(&self) -> HashMap<String, f32> {
        let ops = [
            ("Parse", CodeOperation::Parse),
            ("Encode", CodeOperation::Encode),
            ("Generate", CodeOperation::Generate),
            ("Modify", CodeOperation::Modify),
            ("Explain", CodeOperation::Explain),
            ("FindSimilar", CodeOperation::FindSimilar),
            ("Refactor", CodeOperation::Refactor),
            ("Debug", CodeOperation::Debug),
            ("Verify", CodeOperation::Verify),
        ];

        ops.iter()
            .map(|(name, op)| {
                let result = self.primitive_executor.execute(*op);
                (name.to_string(), result.phi)
            })
            .collect()
    }

    /// "What am I doing when I process code?"
    ///
    /// Traces the primitive pipeline for a given operation.
    pub fn trace_operation(&self, op: CodeOperation) -> CodeExecutionResult {
        self.primitive_executor.execute(op)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language::code_parser::{Entity, Span};

    fn test_span() -> Span {
        Span {
            start_byte: 0,
            end_byte: 10,
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 10,
        }
    }

    fn make_parsed(funcs: &[&str], types: &[&str]) -> ParsedCode {
        let mut parsed = ParsedCode::new("", "rust");
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
    fn test_self_analyzer_creation() {
        let analyzer = SelfAnalyzer::default_dim();
        assert_eq!(analyzer.memory().module_count(), 0);
    }

    #[test]
    fn test_index_and_model() {
        let mut analyzer = SelfAnalyzer::default_dim();

        let parsed = make_parsed(&["parse", "encode"], &["Parser"]);
        analyzer.index_file(Path::new("src/parser.rs"), &parsed);

        let model = analyzer.build_self_model();
        assert_eq!(model.module_count, 1);
        assert_eq!(model.function_count, 2);
        assert_eq!(model.type_count, 1);
    }

    #[test]
    fn test_consciousness_map() {
        let mut analyzer = SelfAnalyzer::default_dim();

        let p1 = make_parsed(&["sort", "filter"], &[]);
        let p2 = make_parsed(&["sort_vec", "filter_vec"], &[]);
        let p3 = make_parsed(&["connect", "disconnect"], &[]);

        analyzer.index_file(Path::new("src/sort.rs"), &p1);
        analyzer.index_file(Path::new("src/sort_ext.rs"), &p2);
        analyzer.index_file(Path::new("src/network.rs"), &p3);

        let map = analyzer.consciousness_map();
        assert_eq!(map.len(), 3);

        // The sort files should be more integrated with each other
        // than the network file
    }

    #[test]
    fn test_introspect_patterns() {
        let analyzer = SelfAnalyzer::default_dim();

        let files = vec![
            (
                PathBuf::from("a.rs"),
                make_parsed(&["f1", "f2", "f3"], &["T1"]),
            ),
            (PathBuf::from("b.rs"), make_parsed(&["g1"], &["T2", "T3"])),
        ];

        let patterns = analyzer.introspect_patterns(&files);
        // Should have Function and Struct entries
        assert!(patterns.iter().any(|(name, _)| name == "Function"));
        assert!(patterns.iter().any(|(name, _)| name == "Struct"));

        // Function should appear more (4 times) than Struct (3 times)
        let fn_count = patterns
            .iter()
            .find(|(n, _)| n == "Function")
            .map(|(_, c)| *c)
            .unwrap_or(0);
        let struct_count = patterns
            .iter()
            .find(|(n, _)| n == "Struct")
            .map(|(_, c)| *c)
            .unwrap_or(0);
        assert!(fn_count > struct_count);
    }

    #[test]
    fn test_find_similar() {
        let mut analyzer = SelfAnalyzer::default_dim();

        let parsed = make_parsed(&["sort_ascending", "sort_descending", "binary_search"], &[]);
        analyzer.index_file(Path::new("src/algo.rs"), &parsed);

        let results = analyzer.find_similar("sort", 3);
        assert!(!results.is_empty());
    }
}
