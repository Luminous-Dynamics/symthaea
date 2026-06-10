// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Code HDC Memory — Project-level codebase index in hypervector space
//!
//! Stores an entire codebase as HDC memory for fast semantic retrieval.
//! Each file is parsed, encoded, and indexed at multiple granularities:
//! module-level, function-level, and type-level.
//!
//! # Architecture
//!
//! ```text
//! Codebase (files)
//!     ↓ index_file()
//! CodebaseMemory {
//!     modules: HashMap<PathBuf, ContinuousHV>,     // file-level
//!     functions: Vec<(path, name, ContinuousHV)>,   // function-level
//!     types: Vec<(path, name, ContinuousHV)>,       // type-level
//!     relationships: Vec<ContinuousHV>,             // inter-entity
//! }
//!     ↓ query()
//! Vec<CodeMatch> (ranked by cosine similarity)
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use symthaea_core::hdc::ContinuousHV;

use super::code_encoder::CodeHDEncoder;
use crate::language::code_parser::{Entity, EntityKind, EntityRelation, ParsedCode};

/// A match result from querying the codebase memory
#[derive(Debug, Clone)]
pub struct CodeMatch {
    /// File path where the match was found
    pub path: PathBuf,
    /// Name of the matched entity (function name, type name, etc.)
    pub name: String,
    /// Kind of entity
    pub kind: EntityKind,
    /// Cosine similarity to the query
    pub similarity: f32,
}

/// Entry in the function index
#[derive(Debug, Clone)]
struct FunctionEntry {
    path: PathBuf,
    name: String,
    hv: ContinuousHV,
}

/// Entry in the type index
#[derive(Debug, Clone)]
struct TypeEntry {
    path: PathBuf,
    name: String,
    kind: EntityKind,
    hv: ContinuousHV,
}

/// Tracks function call relationships for impact analysis.
///
/// Maintains a bidirectional index of caller→callee edges, enabling both
/// forward dependency queries ("what does this function call?") and reverse
/// impact analysis ("what breaks if I change this function?").
#[derive(Debug, Clone, Default)]
pub struct CallGraph {
    /// Function name → list of functions it calls
    callees: HashMap<String, Vec<String>>,
    /// Function name → list of functions that call it (reverse index)
    callers: HashMap<String, Vec<String>>,
}

impl CallGraph {
    /// Create a new empty call graph
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that `caller` invokes `callee`.
    ///
    /// Duplicate edges are ignored — each (caller, callee) pair is stored at
    /// most once.
    pub fn add_call(&mut self, caller: &str, callee: &str) {
        let callee_list = self.callees.entry(caller.to_string()).or_default();
        if !callee_list.contains(&callee.to_string()) {
            callee_list.push(callee.to_string());
        }

        let caller_list = self.callers.entry(callee.to_string()).or_default();
        if !caller_list.contains(&caller.to_string()) {
            caller_list.push(caller.to_string());
        }
    }

    /// Get functions called by `name`
    pub fn callees_of(&self, name: &str) -> &[String] {
        self.callees.get(name).map_or(&[], |v| v.as_slice())
    }

    /// Get functions that call `name` (impact analysis)
    pub fn callers_of(&self, name: &str) -> &[String] {
        self.callers.get(name).map_or(&[], |v| v.as_slice())
    }

    /// How many functions would be directly impacted by changing `name`?
    pub fn impact_count(&self, name: &str) -> usize {
        self.callers_of(name).len()
    }

    /// Transitive callers (recursive impact analysis, bounded by depth).
    ///
    /// Returns all functions that transitively depend on `name`, up to
    /// `max_depth` hops. Uses a visited set to prevent infinite recursion
    /// in the presence of cycles.
    pub fn transitive_callers(&self, name: &str, max_depth: usize) -> Vec<String> {
        let mut visited = std::collections::HashSet::new();
        let mut result = Vec::new();
        self.collect_transitive_callers(name, max_depth, 0, &mut visited, &mut result);
        result
    }

    /// Recursive helper for transitive caller collection.
    fn collect_transitive_callers(
        &self,
        name: &str,
        max_depth: usize,
        current_depth: usize,
        visited: &mut std::collections::HashSet<String>,
        result: &mut Vec<String>,
    ) {
        if current_depth >= max_depth {
            return;
        }
        for caller in self.callers_of(name) {
            if visited.insert(caller.clone()) {
                result.push(caller.clone());
                self.collect_transitive_callers(
                    caller,
                    max_depth,
                    current_depth + 1,
                    visited,
                    result,
                );
            }
        }
    }

    /// Total edges in the graph
    pub fn edge_count(&self) -> usize {
        self.callees.values().map(|v| v.len()).sum()
    }

    /// Clear the graph, removing all edges
    pub fn clear(&mut self) {
        self.callees.clear();
        self.callers.clear();
    }
}

/// Project-level HDC memory for fast semantic code retrieval
pub struct CodebaseMemory {
    encoder: CodeHDEncoder,
    /// Module-level hypervectors (one per file)
    modules: HashMap<PathBuf, ContinuousHV>,
    /// Function-level index
    functions: Vec<FunctionEntry>,
    /// Type-level index (structs, enums, traits, classes)
    types: Vec<TypeEntry>,
    /// Encoded relationship triples
    relationships: Vec<ContinuousHV>,
    /// Overall codebase vector (centroid of all modules)
    codebase_hv: Option<ContinuousHV>,
    /// Call graph for function dependency / impact analysis
    call_graph: CallGraph,
}

impl CodebaseMemory {
    /// Create a new empty codebase memory with the given encoder
    pub fn new(encoder: CodeHDEncoder) -> Self {
        Self {
            encoder,
            modules: HashMap::new(),
            functions: Vec::new(),
            types: Vec::new(),
            relationships: Vec::new(),
            codebase_hv: None,
            call_graph: CallGraph::new(),
        }
    }

    /// Create with default dimension encoder
    pub fn with_default_encoder() -> Self {
        Self::new(CodeHDEncoder::default_dim())
    }

    /// Get a reference to the encoder
    pub fn encoder(&self) -> &CodeHDEncoder {
        &self.encoder
    }

    /// Get the number of indexed modules
    pub fn module_count(&self) -> usize {
        self.modules.len()
    }

    /// Get the number of indexed functions
    pub fn function_count(&self) -> usize {
        self.functions.len()
    }

    /// Get the number of indexed types
    pub fn type_count(&self) -> usize {
        self.types.len()
    }

    /// Get the overall codebase hypervector (centroid of all modules)
    pub fn codebase_hv(&self) -> Option<&ContinuousHV> {
        self.codebase_hv.as_ref()
    }

    /// Get all module paths
    pub fn module_paths(&self) -> Vec<&Path> {
        self.modules.keys().map(|p| p.as_path()).collect()
    }

    /// Get a module's HV by path
    pub fn module_hv(&self, path: &Path) -> Option<&ContinuousHV> {
        self.modules.get(path)
    }

    // ========================================================================
    // Indexing
    // ========================================================================

    /// Index a single parsed file into memory
    pub fn index_file(&mut self, path: &Path, parsed: &ParsedCode) {
        let path_buf = path.to_path_buf();

        // Encode the full module
        let module_hv = self.encoder.encode_module(parsed);
        self.modules.insert(path_buf.clone(), module_hv);

        // Index individual entities
        for entity in parsed.all_entities() {
            match entity.kind {
                EntityKind::Function | EntityKind::Method => {
                    let hv = self.encoder.encode_function(entity);
                    self.functions.push(FunctionEntry {
                        path: path_buf.clone(),
                        name: entity.name.clone(),
                        hv,
                    });
                }
                EntityKind::Struct
                | EntityKind::Class
                | EntityKind::Enum
                | EntityKind::Trait
                | EntityKind::Interface
                | EntityKind::TypeAlias => {
                    let hv = self.encoder.encode_entity(entity);
                    self.types.push(TypeEntry {
                        path: path_buf.clone(),
                        name: entity.name.clone(),
                        kind: entity.kind,
                        hv,
                    });
                }
                _ => {}
            }
        }

        // Index relationships
        for rel in &parsed.structure.relations {
            let rel_hv =
                self.encoder
                    .encode_relationship_triple(&rel.source, rel.relation, &rel.target);
            self.relationships.push(rel_hv);
        }

        // Update codebase centroid
        self.update_codebase_hv();
    }

    /// Remove a file from the index
    pub fn remove_file(&mut self, path: &Path) {
        self.modules.remove(path);
        self.functions.retain(|f| f.path != path);
        self.types.retain(|t| t.path != path);
        self.update_codebase_hv();
    }

    /// Update (re-index) a file — removes old entries and re-indexes
    pub fn update_file(&mut self, path: &Path, parsed: &ParsedCode) {
        self.remove_file(path);
        self.index_file(path, parsed);
    }

    /// Update the overall codebase HV (centroid of all module HVs)
    fn update_codebase_hv(&mut self) {
        if self.modules.is_empty() {
            self.codebase_hv = None;
            return;
        }

        let module_hvs: Vec<ContinuousHV> = self.modules.values().cloned().collect();
        self.codebase_hv = Some(ContinuousHV::bundle_owned(&module_hvs));
    }

    // ========================================================================
    // Querying
    // ========================================================================

    /// Query for the top-k most similar code entities across all indexed files
    pub fn query(&self, intent_hv: &ContinuousHV, top_k: usize) -> Vec<CodeMatch> {
        let mut results = Vec::new();

        // Search functions
        for entry in &self.functions {
            let sim = intent_hv.similarity(&entry.hv);
            results.push(CodeMatch {
                path: entry.path.clone(),
                name: entry.name.clone(),
                kind: EntityKind::Function,
                similarity: sim,
            });
        }

        // Search types
        for entry in &self.types {
            let sim = intent_hv.similarity(&entry.hv);
            results.push(CodeMatch {
                path: entry.path.clone(),
                name: entry.name.clone(),
                kind: entry.kind,
                similarity: sim,
            });
        }

        // Sort by similarity descending
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);
        results
    }

    /// Query only functions
    pub fn query_functions(&self, intent_hv: &ContinuousHV, top_k: usize) -> Vec<CodeMatch> {
        let mut results: Vec<CodeMatch> = self
            .functions
            .iter()
            .map(|entry| CodeMatch {
                path: entry.path.clone(),
                name: entry.name.clone(),
                kind: EntityKind::Function,
                similarity: intent_hv.similarity(&entry.hv),
            })
            .collect();

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);
        results
    }

    /// Query only types
    pub fn query_types(&self, intent_hv: &ContinuousHV, top_k: usize) -> Vec<CodeMatch> {
        let mut results: Vec<CodeMatch> = self
            .types
            .iter()
            .map(|entry| CodeMatch {
                path: entry.path.clone(),
                name: entry.name.clone(),
                kind: entry.kind,
                similarity: intent_hv.similarity(&entry.hv),
            })
            .collect();

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);
        results
    }

    /// Query for the most similar module (file)
    pub fn query_modules(&self, intent_hv: &ContinuousHV, top_k: usize) -> Vec<(PathBuf, f32)> {
        let mut results: Vec<(PathBuf, f32)> = self
            .modules
            .iter()
            .map(|(path, hv)| (path.clone(), intent_hv.similarity(hv)))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Get all function HVs (for use in algebra operations like analogy)
    pub fn all_function_hvs(&self) -> Vec<&ContinuousHV> {
        self.functions.iter().map(|f| &f.hv).collect()
    }

    /// Get all type HVs
    pub fn all_type_hvs(&self) -> Vec<&ContinuousHV> {
        self.types.iter().map(|t| &t.hv).collect()
    }

    /// Get all module HVs
    pub fn all_module_hvs(&self) -> Vec<&ContinuousHV> {
        self.modules.values().collect()
    }

    // ========================================================================
    // Analysis
    // ========================================================================

    /// Compute surprise: how much does a file's HV differ from the codebase centroid?
    /// High surprise = file is very different from the norm (potentially interesting)
    pub fn compute_surprise(&self, file_hv: &ContinuousHV) -> f32 {
        match &self.codebase_hv {
            Some(centroid) => 1.0 - centroid.similarity(file_hv),
            None => 0.5, // neutral if no codebase indexed
        }
    }

    /// Find the most "surprising" (atypical) files in the codebase
    pub fn most_surprising_files(&self, top_k: usize) -> Vec<(PathBuf, f32)> {
        let centroid = match &self.codebase_hv {
            Some(c) => c,
            None => return Vec::new(),
        };

        let mut results: Vec<(PathBuf, f32)> = self
            .modules
            .iter()
            .map(|(path, hv)| {
                let surprise = 1.0 - centroid.similarity(hv);
                (path.clone(), surprise)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Compute overall codebase coherence (average pairwise module similarity)
    pub fn codebase_coherence(&self) -> f32 {
        let hvs: Vec<&ContinuousHV> = self.modules.values().collect();
        if hvs.len() < 2 {
            return 1.0;
        }

        let mut total_sim = 0.0f32;
        let mut count = 0u32;

        for i in 0..hvs.len() {
            for j in (i + 1)..hvs.len() {
                total_sim += hvs[i].similarity(hvs[j]);
                count += 1;
            }
        }

        if count > 0 {
            total_sim / count as f32
        } else {
            1.0
        }
    }

    /// Index an entire directory tree of source files.
    ///
    /// Walks the directory recursively, parsing .rs and .py files and indexing
    /// them into memory. Skips files in `target/`, `node_modules/`, `.git/`,
    /// and other build artifact directories.
    ///
    /// Returns the number of files successfully indexed.
    pub fn index_directory(&mut self, dir: &Path) -> usize {
        let mut count = 0;
        if !dir.is_dir() {
            return 0;
        }
        self.index_directory_recursive(dir, &mut count);
        count
    }

    /// Recursive directory walker for index_directory.
    fn index_directory_recursive(&mut self, dir: &Path, count: &mut usize) {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            // Skip build artifacts and hidden directories
            if path.is_dir() {
                if name_str.starts_with('.')
                    || name_str == "target"
                    || name_str == "node_modules"
                    || name_str == "venv"
                    || name_str == "__pycache__"
                {
                    continue;
                }
                self.index_directory_recursive(&path, count);
                continue;
            }

            // Only index Rust and Python files
            let ext = path.extension().and_then(|e| e.to_str());
            let language = match ext {
                Some("rs") => "rust",
                Some("py") => "python",
                _ => continue,
            };

            // Read and parse the file
            let source = match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(_) => continue,
            };

            // Skip very large files (>100KB) to avoid excessive memory use
            if source.len() > 100_000 {
                continue;
            }

            let parsed = ParsedCode::new(&source, language);
            if !parsed.entities.is_empty() {
                self.index_file(&path, &parsed);
                *count += 1;
            }
        }
    }

    // ========================================================================
    // Call Graph / Dependency Tracking
    // ========================================================================

    /// Index function calls from parsed code relationships.
    ///
    /// Call this after indexing functions to build the call graph. Each
    /// invocation records that `caller_name` calls every function in
    /// `called_functions`.
    pub fn index_calls(&mut self, caller_name: &str, called_functions: &[String]) {
        for callee in called_functions {
            self.call_graph.add_call(caller_name, callee);
        }
    }

    /// Get the call graph for impact analysis
    pub fn call_graph(&self) -> &CallGraph {
        &self.call_graph
    }

    /// Compute task complexity based on dependency depth.
    ///
    /// Returns 0.0–1.0 where higher = more callers = more complex to change.
    /// Formula: `(direct_callers / 10.0).min(1.0)`
    pub fn dependency_complexity(&self, function_name: &str) -> f32 {
        let callers = self.call_graph.callers_of(function_name).len() as f32;
        (callers / 10.0).min(1.0)
    }

    /// Get memory statistics
    pub fn stats(&self) -> CodebaseMemoryStats {
        CodebaseMemoryStats {
            modules: self.modules.len(),
            functions: self.functions.len(),
            types: self.types.len(),
            relationships: self.relationships.len(),
            has_codebase_hv: self.codebase_hv.is_some(),
        }
    }
}

/// Statistics about the codebase memory
#[derive(Debug, Clone)]
pub struct CodebaseMemoryStats {
    pub modules: usize,
    pub functions: usize,
    pub types: usize,
    pub relationships: usize,
    pub has_codebase_hv: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language::code_parser::Span;

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

    fn make_parsed_file(_name: &str, funcs: &[&str], types: &[&str]) -> ParsedCode {
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
    fn test_index_and_query() {
        let mut memory = CodebaseMemory::new(CodeHDEncoder::new(512));

        let parsed = make_parsed_file("main.rs", &["sort", "filter", "map"], &["Config"]);
        memory.index_file(Path::new("src/main.rs"), &parsed);

        assert_eq!(memory.module_count(), 1);
        assert_eq!(memory.function_count(), 3);
        assert_eq!(memory.type_count(), 1);

        // Query for something similar to "sort"
        let query = memory.encoder().encode_name("sort_list");
        let results = memory.query(&query, 5);
        assert!(!results.is_empty());
        // "sort" should be among top results
        assert!(results.iter().any(|m| m.name == "sort"));
    }

    #[test]
    fn test_multiple_files() {
        let mut memory = CodebaseMemory::new(CodeHDEncoder::new(512));

        let file1 = make_parsed_file("lib.rs", &["parse", "tokenize"], &["Parser"]);
        let file2 = make_parsed_file("main.rs", &["run", "setup"], &["App"]);

        memory.index_file(Path::new("src/lib.rs"), &file1);
        memory.index_file(Path::new("src/main.rs"), &file2);

        assert_eq!(memory.module_count(), 2);
        assert_eq!(memory.function_count(), 4);
        assert_eq!(memory.type_count(), 2);

        // Codebase HV should exist
        assert!(memory.codebase_hv().is_some());
    }

    #[test]
    fn test_update_file() {
        let mut memory = CodebaseMemory::new(CodeHDEncoder::new(512));

        let parsed_v1 = make_parsed_file("main.rs", &["old_func"], &[]);
        memory.index_file(Path::new("src/main.rs"), &parsed_v1);
        assert_eq!(memory.function_count(), 1);

        let parsed_v2 = make_parsed_file("main.rs", &["new_func", "another_func"], &[]);
        memory.update_file(Path::new("src/main.rs"), &parsed_v2);
        assert_eq!(memory.function_count(), 2);
    }

    #[test]
    fn test_remove_file() {
        let mut memory = CodebaseMemory::new(CodeHDEncoder::new(512));

        let parsed = make_parsed_file("main.rs", &["func_a"], &["TypeA"]);
        memory.index_file(Path::new("src/main.rs"), &parsed);
        assert_eq!(memory.module_count(), 1);

        memory.remove_file(Path::new("src/main.rs"));
        assert_eq!(memory.module_count(), 0);
        assert_eq!(memory.function_count(), 0);
        assert_eq!(memory.type_count(), 0);
    }

    #[test]
    fn test_query_functions() {
        let mut memory = CodebaseMemory::new(CodeHDEncoder::new(512));

        let parsed = make_parsed_file("main.rs", &["sort", "search", "validate"], &["Config"]);
        memory.index_file(Path::new("src/main.rs"), &parsed);

        let query = memory.encoder().encode_name("sorting");
        let results = memory.query_functions(&query, 2);
        assert_eq!(results.len(), 2);
        assert!(results[0].similarity >= results[1].similarity);
    }

    #[test]
    fn test_surprise_and_coherence() {
        let mut memory = CodebaseMemory::new(CodeHDEncoder::new(512));

        // Index several similar files
        for i in 0..3 {
            let parsed = make_parsed_file(&format!("file{i}.rs"), &["sort", "filter"], &[]);
            memory.index_file(Path::new(&format!("src/file{i}.rs")), &parsed);
        }

        // Coherence should be positive
        let coherence = memory.codebase_coherence();
        assert!(coherence > 0.0);

        // A file about the same things should have low surprise
        let same_hv = memory.encoder().encode_name("sort");
        let surprise = memory.compute_surprise(&same_hv);
        assert!(surprise < 1.0);
    }

    #[test]
    fn test_stats() {
        let mut memory = CodebaseMemory::new(CodeHDEncoder::new(512));
        let stats = memory.stats();
        assert_eq!(stats.modules, 0);
        assert!(!stats.has_codebase_hv);

        let parsed = make_parsed_file("test.rs", &["f1"], &["T1"]);
        memory.index_file(Path::new("test.rs"), &parsed);

        let stats = memory.stats();
        assert_eq!(stats.modules, 1);
        assert_eq!(stats.functions, 1);
        assert_eq!(stats.types, 1);
        assert!(stats.has_codebase_hv);
    }

    // ====================================================================
    // Call Graph tests
    // ====================================================================

    #[test]
    fn test_call_graph_basic() {
        let mut cg = CallGraph::new();
        cg.add_call("main", "parse");
        cg.add_call("main", "execute");
        cg.add_call("execute", "parse");

        // Forward edges
        assert_eq!(cg.callees_of("main"), &["parse", "execute"]);
        assert_eq!(cg.callees_of("execute"), &["parse"]);

        // Reverse edges
        assert_eq!(cg.callers_of("parse"), &["main", "execute"]);
        assert_eq!(cg.callers_of("execute"), &["main"]);

        // Empty lookups
        assert!(cg.callees_of("parse").is_empty());
        assert!(cg.callers_of("main").is_empty());

        // Deduplication: adding same edge again should be a no-op
        cg.add_call("main", "parse");
        assert_eq!(cg.callees_of("main").len(), 2);
        assert_eq!(cg.callers_of("parse").len(), 2);
    }

    #[test]
    fn test_call_graph_impact() {
        let mut cg = CallGraph::new();
        cg.add_call("a", "core_fn");
        cg.add_call("b", "core_fn");
        cg.add_call("c", "core_fn");
        cg.add_call("d", "core_fn");

        assert_eq!(cg.impact_count("core_fn"), 4);
        assert_eq!(cg.impact_count("a"), 0);
        assert_eq!(cg.impact_count("nonexistent"), 0);
    }

    #[test]
    fn test_call_graph_transitive() {
        let mut cg = CallGraph::new();
        // Chain: d → c → b → a → target
        cg.add_call("a", "target");
        cg.add_call("b", "a");
        cg.add_call("c", "b");
        cg.add_call("d", "c");

        // Depth 1: only direct callers
        let t1 = cg.transitive_callers("target", 1);
        assert_eq!(t1, vec!["a"]);

        // Depth 2: a, b
        let t2 = cg.transitive_callers("target", 2);
        assert_eq!(t2, vec!["a", "b"]);

        // Depth 10: all four
        let tall = cg.transitive_callers("target", 10);
        assert_eq!(tall.len(), 4);
        assert!(tall.contains(&"a".to_string()));
        assert!(tall.contains(&"d".to_string()));

        // Cycle safety: a → b → a (cycle)
        let mut cg2 = CallGraph::new();
        cg2.add_call("a", "b");
        cg2.add_call("b", "a");
        let tc = cg2.transitive_callers("b", 100);
        // Should contain "a" but not loop forever
        assert_eq!(tc, vec!["a", "b"]);
    }

    #[test]
    fn test_dependency_complexity() {
        let mut memory = CodebaseMemory::new(CodeHDEncoder::new(512));

        // No callers → 0.0
        assert_eq!(memory.dependency_complexity("unknown_fn"), 0.0);

        // 5 callers → 0.5
        for i in 0..5 {
            memory.index_calls(&format!("caller_{i}"), &["target".to_string()]);
        }
        assert!((memory.dependency_complexity("target") - 0.5).abs() < f32::EPSILON);

        // 15 callers → capped at 1.0
        for i in 5..15 {
            memory.index_calls(&format!("caller_{i}"), &["hot_fn".to_string()]);
        }
        assert!((memory.dependency_complexity("hot_fn") - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_call_graph_clear() {
        let mut cg = CallGraph::new();
        cg.add_call("a", "b");
        cg.add_call("b", "c");
        assert_eq!(cg.edge_count(), 2);

        cg.clear();
        assert_eq!(cg.edge_count(), 0);
        assert!(cg.callees_of("a").is_empty());
        assert!(cg.callers_of("b").is_empty());
    }
}
