// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! AST-backed repository map for code generation and repair.
//!
//! This layer makes the existing tree-sitter parsers and HDC code memory usable
//! by agents. It keeps exact source spans next to HDC-ranked symbols so retrieval
//! can return precise snippets instead of ungrounded bags of text.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::hdc::code_encoder::CodeHDEncoder;
use crate::hdc::code_memory::{CodeMatch, CodebaseMemory};

use super::code_executor::{CompileError, parse_structured_errors};
use super::code_generator::CodeContext;
use super::code_parser::{CodeDiagnostic, Entity, EntityKind, ParsedCode, Span};
use super::parser_registry::ParserRegistry;
use super::rust_lsp::LspLocation;

const MAX_INDEXED_FILE_BYTES: usize = 250_000;

/// A symbol definition with an exact source span.
#[derive(Debug, Clone)]
pub struct RepoSymbol {
    /// Defining file.
    pub path: PathBuf,
    /// Language detected by the parser.
    pub language: String,
    /// Symbol name.
    pub name: String,
    /// Entity kind from the parser.
    pub kind: EntityKind,
    /// Exact source span for the parsed entity.
    pub span: Span,
    /// Compact signature or first line for prompt context.
    pub signature: String,
    /// Full source snippet for the entity span.
    pub snippet: String,
}

/// Parsed source file retained by the repo map.
#[derive(Debug, Clone)]
pub struct RepoFile {
    /// File path.
    pub path: PathBuf,
    /// Detected language.
    pub language: String,
    /// Original source.
    pub source: String,
    /// Parsed representation.
    pub parsed: ParsedCode,
}

/// Aggregate indexing statistics.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RepoMapStats {
    pub files_indexed: usize,
    pub files_skipped: usize,
    pub parse_errors: usize,
    pub symbols_indexed: usize,
}

/// Structured compiler diagnostic attached to repository evidence.
#[derive(Debug, Clone)]
pub struct RepoDiagnostic {
    /// Parsed compiler diagnostic.
    pub error: CompileError,
    /// Best symbol containing or preceding the diagnostic span.
    pub symbol: Option<RepoSymbol>,
}

/// A repository map combining precise AST spans with HDC retrieval.
pub struct RepoMap {
    root: PathBuf,
    parser_registry: ParserRegistry,
    memory: CodebaseMemory,
    files: HashMap<PathBuf, RepoFile>,
    symbols: Vec<RepoSymbol>,
    diagnostics: Vec<(PathBuf, CodeDiagnostic)>,
}

impl RepoMap {
    /// Create an empty map rooted at `root`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            parser_registry: ParserRegistry::with_builtins(),
            memory: CodebaseMemory::new(CodeHDEncoder::default_dim()),
            files: HashMap::new(),
            symbols: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    /// Repository root used for recursive scans.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// HDC memory backing semantic retrieval.
    pub fn memory(&self) -> &CodebaseMemory {
        &self.memory
    }

    /// Parsed files retained in the map.
    pub fn files(&self) -> &HashMap<PathBuf, RepoFile> {
        &self.files
    }

    /// Indexed symbols with exact snippets.
    pub fn symbols(&self) -> &[RepoSymbol] {
        &self.symbols
    }

    /// Parse diagnostics accumulated while indexing.
    pub fn diagnostics(&self) -> &[(PathBuf, CodeDiagnostic)] {
        &self.diagnostics
    }

    /// Recursively scan the configured root.
    pub fn scan(&mut self) -> io::Result<RepoMapStats> {
        let root = self.root.clone();
        self.scan_path(&root)
    }

    /// Recursively scan a path into this map.
    pub fn scan_path(&mut self, path: &Path) -> io::Result<RepoMapStats> {
        let mut stats = RepoMapStats::default();
        self.scan_path_inner(path, &mut stats)?;
        stats.symbols_indexed = self.symbols.len();
        Ok(stats)
    }

    /// Index one source string. Useful for tests, editors, and incremental scans.
    pub fn index_source(
        &mut self,
        path: impl Into<PathBuf>,
        source: impl Into<String>,
    ) -> Result<(), CodeDiagnostic> {
        let path = path.into();
        let source = source.into();
        let filename = path.to_string_lossy();
        let parsed = self
            .parser_registry
            .parse(&source, None, Some(filename.as_ref()))?;

        self.remove_path_entries(&path);
        self.memory.index_file(&path, &parsed);
        self.collect_symbols(&path, &parsed);
        self.files.insert(
            path.clone(),
            RepoFile {
                path,
                language: parsed.language.clone(),
                source,
                parsed,
            },
        );
        Ok(())
    }

    /// Find symbols by exact name.
    pub fn find_symbol(&self, name: &str) -> Vec<&RepoSymbol> {
        self.symbols
            .iter()
            .filter(|symbol| symbol.name == name)
            .collect()
    }

    /// HDC-rank symbols for a text query, then attach exact spans/snippets.
    pub fn query_symbols(&self, query: &str, top_k: usize) -> Vec<RepoSymbolMatch> {
        let query_hv = self.memory.encoder().encode_name(query);
        self.memory
            .query(&query_hv, top_k.saturating_mul(3).max(top_k))
            .into_iter()
            .filter_map(|matched| self.attach_symbol(matched))
            .take(top_k)
            .collect()
    }

    /// Return a compact skeleton for a file.
    pub fn file_skeleton(&self, path: &Path) -> Option<String> {
        let mut rows: Vec<&RepoSymbol> = self
            .symbols
            .iter()
            .filter(|symbol| symbol.path == path)
            .collect();
        rows.sort_by_key(|symbol| symbol.span.start_byte);

        if rows.is_empty() {
            return None;
        }

        Some(
            rows.into_iter()
                .map(|symbol| {
                    format!(
                        "L{}-L{} {:?} {} :: {}",
                        symbol.span.start_line + 1,
                        symbol.span.end_line + 1,
                        symbol.kind,
                        symbol.name,
                        symbol.signature
                    )
                })
                .collect::<Vec<_>>()
                .join("\n"),
        )
    }

    /// Build generation context containing HDC memory and precise source snippets.
    pub fn code_context_for_query(&self, query: &str, top_k: usize) -> CodeContext<'_> {
        let source_files = self
            .query_symbols(query, top_k)
            .into_iter()
            .map(|matched| {
                (
                    format!(
                        "{}:{}:{}",
                        matched.symbol.path.display(),
                        matched.symbol.span.start_line + 1,
                        matched.symbol.name
                    ),
                    matched.symbol.snippet,
                )
            })
            .collect();

        CodeContext {
            memory: Some(&self.memory),
            source_files,
            ..CodeContext::default()
        }
    }

    /// Build generation context from compiler diagnostics.
    ///
    /// This turns rustc text into structured repair evidence: error hints plus
    /// exact symbol snippets near the diagnostic location. If no precise span
    /// match exists, it falls back to HDC query over the diagnostic text.
    pub fn code_context_for_compile_errors(
        &self,
        compile_errors: &[String],
        fallback_top_k: usize,
    ) -> CodeContext<'_> {
        let diagnostics = self.attach_compile_errors(compile_errors);
        let mut source_files = Vec::new();
        let mut error_hints = Vec::new();

        for diagnostic in &diagnostics {
            let code = diagnostic.error.code.as_deref().unwrap_or("rustc");
            let location = match (
                diagnostic.error.file.as_deref(),
                diagnostic.error.line,
                diagnostic.error.column,
            ) {
                (Some(file), Some(line), Some(column)) => format!("{file}:{line}:{column}"),
                (Some(file), Some(line), None) => format!("{file}:{line}"),
                (Some(file), None, _) => file.to_string(),
                _ => "unknown location".to_string(),
            };

            error_hints.push((
                format!("compile_error_{code}"),
                format!(
                    "{code} at {location}: {} [{:?}]",
                    diagnostic.error.message, diagnostic.error.category
                ),
            ));

            if let Some(symbol) = &diagnostic.symbol {
                push_unique_source_file(
                    &mut source_files,
                    format!(
                        "{}:{}:{}",
                        symbol.path.display(),
                        symbol.span.start_line + 1,
                        symbol.name
                    ),
                    symbol.snippet.clone(),
                );
            }
        }

        if source_files.len() < fallback_top_k {
            for matched in self.query_symbols(&compile_errors.join("\n"), fallback_top_k) {
                push_unique_source_file(
                    &mut source_files,
                    format!(
                        "{}:{}:{}",
                        matched.symbol.path.display(),
                        matched.symbol.span.start_line + 1,
                        matched.symbol.name
                    ),
                    matched.symbol.snippet,
                );
                if source_files.len() >= fallback_top_k {
                    break;
                }
            }
        }

        CodeContext {
            memory: Some(&self.memory),
            source_files,
            error_hints,
            ..CodeContext::default()
        }
    }

    /// Build generation context from LSP navigation results.
    ///
    /// LSP supplies type-aware locations; the repo map turns those locations
    /// back into exact AST snippets so downstream generation still receives
    /// compact, symbol-scoped evidence.
    pub fn code_context_for_lsp_locations(&self, locations: &[LspLocation]) -> CodeContext<'_> {
        let mut source_files = Vec::new();
        let mut error_hints = Vec::new();

        for location in locations {
            let Some(path) = location.path() else {
                continue;
            };
            let line = location.range.start.line as usize + 1;
            if let Some(symbol) = self.symbol_at_line(&path.to_string_lossy(), line) {
                push_unique_source_file(
                    &mut source_files,
                    format!(
                        "{}:{}:{}",
                        symbol.path.display(),
                        symbol.span.start_line + 1,
                        symbol.name
                    ),
                    symbol.snippet.clone(),
                );
            } else {
                error_hints.push((
                    format!("lsp_location_{}:{}", path.display(), line),
                    "LSP returned this location, but no indexed AST symbol covered it".to_string(),
                ));
            }
        }

        CodeContext {
            memory: Some(&self.memory),
            source_files,
            error_hints,
            ..CodeContext::default()
        }
    }

    /// Build generation context for a repository-scale engineering task (SWE-bench style).
    ///
    /// Analyzes the issue description, ranks relevant symbols using HDC, and
    /// uses the LSP (if provided) to expand the context to include definitions
    /// and references related to the most relevant symbols.
    pub fn code_context_for_issue(
        &self,
        issue_text: &str,
        mut lsp: Option<&mut crate::language::rust_lsp::RustAnalyzerClient>,
        top_k: usize,
    ) -> CodeContext<'_> {
        // 1. Initial semantic ranking via HDC
        let mut context = self.code_context_for_query(issue_text, top_k);

        // 2. LSP-driven context expansion (Active Sensing)
        if let Some(lsp) = lsp.as_mut() {
            let mut expanded_locations = Vec::new();

            // Map our ranked symbols to LSP locations to find related code
            for (label, _) in &context.source_files {
                // Label format: "path:line:name"
                if let Some((path_str, line_str)) =
                    label.rsplit_once(':').and_then(|(p, _)| p.rsplit_once(':'))
                {
                    if let Ok(line) = line_str.parse::<u32>() {
                        // Find references to the relevant symbols to understand the "blast radius"
                        let pos = crate::language::rust_lsp::LspPosition::new(line - 1, 0);
                        if let Ok(refs) = lsp.find_references(path_str, pos, true) {
                            expanded_locations.extend(refs);
                        }
                    }
                }
            }

            // 3. Merge expanded LSP context back into the RepoMap AST context
            if !expanded_locations.is_empty() {
                let lsp_context = self.code_context_for_lsp_locations(&expanded_locations);
                for (label, snippet) in lsp_context.source_files {
                    push_unique_source_file(&mut context.source_files, label, snippet);
                }
            }
        }

        context.issue_text = Some(issue_text.to_string());
        context
    }

    /// Parse and attach compiler diagnostics to repository symbols.
    pub fn attach_compile_errors(&self, compile_errors: &[String]) -> Vec<RepoDiagnostic> {
        parse_structured_errors(&compile_errors.join("\n"))
            .into_iter()
            .map(|error| {
                let symbol = self.symbol_for_compile_error(&error).cloned();
                RepoDiagnostic { error, symbol }
            })
            .collect()
    }

    fn scan_path_inner(&mut self, path: &Path, stats: &mut RepoMapStats) -> io::Result<()> {
        if path.is_file() {
            self.scan_file(path, stats);
            return Ok(());
        }

        if !path.is_dir() {
            stats.files_skipped += 1;
            return Ok(());
        }

        for entry in fs::read_dir(path)? {
            let entry = match entry {
                Ok(entry) => entry,
                Err(_) => {
                    stats.files_skipped += 1;
                    continue;
                }
            };
            let entry_path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();

            if entry_path.is_dir() {
                if should_skip_dir(&name) {
                    continue;
                }
                self.scan_path_inner(&entry_path, stats)?;
            } else {
                self.scan_file(&entry_path, stats);
            }
        }

        Ok(())
    }

    fn scan_file(&mut self, path: &Path, stats: &mut RepoMapStats) {
        if !is_supported_source_file(path) {
            stats.files_skipped += 1;
            return;
        }

        let source = match fs::read_to_string(path) {
            Ok(source) => source,
            Err(_) => {
                stats.files_skipped += 1;
                return;
            }
        };

        if source.len() > MAX_INDEXED_FILE_BYTES {
            stats.files_skipped += 1;
            return;
        }

        match self.index_source(path.to_path_buf(), source) {
            Ok(()) => stats.files_indexed += 1,
            Err(diagnostic) => {
                stats.parse_errors += 1;
                self.diagnostics.push((path.to_path_buf(), diagnostic));
            }
        }
    }

    fn remove_path_entries(&mut self, path: &Path) {
        self.files.remove(path);
        self.symbols.retain(|symbol| symbol.path != path);
        self.memory.remove_file(path);
    }

    fn collect_symbols(&mut self, path: &Path, parsed: &ParsedCode) {
        for entity in parsed.all_entities() {
            self.symbols
                .push(symbol_from_entity(path, &parsed.language, entity));
        }
    }

    fn attach_symbol(&self, matched: CodeMatch) -> Option<RepoSymbolMatch> {
        let symbol = self
            .symbols
            .iter()
            .find(|symbol| {
                symbol.path == matched.path
                    && symbol.name == matched.name
                    && symbol.kind == matched.kind
            })
            .cloned()?;
        Some(RepoSymbolMatch {
            symbol,
            similarity: matched.similarity,
        })
    }

    fn symbol_for_compile_error(&self, error: &CompileError) -> Option<&RepoSymbol> {
        if let (Some(file), Some(line)) = (error.file.as_deref(), error.line) {
            if let Some(symbol) = self.symbol_at_line(file, line) {
                return Some(symbol);
            }
        }

        for name in backtick_names(&error.message) {
            if let Some(symbol) = self.symbols.iter().find(|symbol| symbol.name == name) {
                return Some(symbol);
            }
        }

        None
    }

    fn symbol_at_line(&self, file: &str, line: usize) -> Option<&RepoSymbol> {
        let mut candidates: Vec<&RepoSymbol> = self
            .symbols
            .iter()
            .filter(|symbol| path_matches_rustc_file(&symbol.path, file))
            .collect();
        candidates.sort_by_key(|symbol| symbol.span.start_line);

        candidates
            .iter()
            .copied()
            .find(|symbol| {
                let start = symbol.span.start_line + 1;
                let end = symbol.span.end_line + 1;
                start <= line && line <= end
            })
            .or_else(|| {
                candidates
                    .into_iter()
                    .rev()
                    .find(|symbol| symbol.span.start_line < line)
            })
    }
}

/// HDC match with exact symbol data attached.
#[derive(Debug, Clone)]
pub struct RepoSymbolMatch {
    pub symbol: RepoSymbol,
    pub similarity: f32,
}

fn symbol_from_entity(path: &Path, language: &str, entity: &Entity) -> RepoSymbol {
    RepoSymbol {
        path: path.to_path_buf(),
        language: language.to_string(),
        name: entity.name.clone(),
        kind: entity.kind,
        span: entity.span.clone(),
        signature: compact_signature(&entity.source_text),
        snippet: entity.source_text.clone(),
    }
}

fn compact_signature(source: &str) -> String {
    let first_line = source.lines().next().unwrap_or("").trim();
    if let Some((head, _)) = first_line.split_once('{') {
        let head = head.trim();
        if !head.is_empty() {
            return head.to_string();
        }
    }
    first_line.chars().take(160).collect()
}

fn is_supported_source_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some("rs") | Some("py") | Some("nix")
    )
}

fn should_skip_dir(name: &str) -> bool {
    name.starts_with('.')
        || matches!(
            name,
            "target" | "node_modules" | "venv" | ".venv" | "__pycache__" | "dist" | "build"
        )
}

fn push_unique_source_file(files: &mut Vec<(String, String)>, label: String, snippet: String) {
    if files.iter().any(|(existing, _)| existing == &label) {
        return;
    }
    files.push((label, snippet));
}

fn path_matches_rustc_file(path: &Path, rustc_file: &str) -> bool {
    let rustc_path = Path::new(rustc_file);
    path == rustc_path
        || path.ends_with(rustc_path)
        || rustc_file.ends_with(&path.to_string_lossy().to_string())
}

fn backtick_names(message: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut rest = message;
    while let Some(start) = rest.find('`') {
        let after_start = &rest[start + 1..];
        let Some(end) = after_start.find('`') else {
            break;
        };
        let name = &after_start[..end];
        if !name.is_empty() {
            names.push(name.to_string());
        }
        rest = &after_start[end + 1..];
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_map_indexes_rust_symbols_with_precise_snippets() {
        let mut repo = RepoMap::new(".");
        repo.index_source(
            "src/lib.rs",
            r#"
pub struct Engine {
    pub cycles: usize,
}

pub fn run_engine(engine: &Engine) -> usize {
    engine.cycles + 1
}
"#,
        )
        .unwrap();

        let symbols = repo.find_symbol("run_engine");
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].kind, EntityKind::Function);
        assert!(symbols[0].snippet.contains("engine.cycles + 1"));
        assert!(symbols[0].span.start_line < symbols[0].span.end_line);

        let skeleton = repo.file_skeleton(Path::new("src/lib.rs")).unwrap();
        assert!(skeleton.contains("Struct Engine"));
        assert!(skeleton.contains("Function run_engine"));
    }

    #[test]
    fn repo_map_query_returns_hdc_ranked_exact_symbols() {
        let mut repo = RepoMap::new(".");
        repo.index_source(
            "src/math.rs",
            r#"
pub fn parse_number(input: &str) -> Option<i64> {
    input.parse().ok()
}

pub fn render_label(name: &str) -> String {
    format!("label:{name}")
}
"#,
        )
        .unwrap();

        let matches = repo.query_symbols("parse integer number", 3);
        assert!(matches.iter().any(|matched| {
            matched.symbol.name == "parse_number"
                && matched.symbol.snippet.contains("input.parse().ok()")
        }));
    }

    #[test]
    fn repo_map_builds_code_context_with_memory_and_snippets() {
        let mut repo = RepoMap::new(".");
        repo.index_source(
            "tools/app.py",
            r#"
def normalize_name(value):
    return value.strip().lower()
"#,
        )
        .unwrap();

        let context = repo.code_context_for_query("normalize name", 2);
        assert!(context.memory.is_some());
        assert_eq!(context.source_files.len(), 1);
        assert!(context.source_files[0].1.contains("normalize_name"));
    }

    #[test]
    fn repo_map_attaches_compile_errors_to_symbol_spans() {
        let mut repo = RepoMap::new(".");
        repo.index_source(
            "src/lib.rs",
            "pub fn normalize_name(name: &str) -> String {\n    name.trim()\n}\n",
        )
        .unwrap();
        let compile_errors = vec![
            "error[E0308]: mismatched types\n  --> src/lib.rs:2:5\n   |\n2 |     name.trim()\n   |     ^^^^^^^^^^^ expected `String`, found `&str`"
                .to_string(),
        ];

        let diagnostics = repo.attach_compile_errors(&compile_errors);

        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].error.code.as_deref(), Some("E0308"));
        assert_eq!(
            diagnostics[0]
                .symbol
                .as_ref()
                .map(|symbol| symbol.name.as_str()),
            Some("normalize_name")
        );
    }

    #[test]
    fn repo_map_compile_error_context_includes_hints_and_snippets() {
        let mut repo = RepoMap::new(".");
        repo.index_source(
            "src/config.rs",
            "pub struct EngineConfig {\n    pub enabled: bool,\n}\n",
        )
        .unwrap();
        let compile_errors =
            vec!["error[E0412]: cannot find type `EngineConfig` in this scope".to_string()];

        let context = repo.code_context_for_compile_errors(&compile_errors, 2);

        assert!(context.memory.is_some());
        assert!(
            context
                .error_hints
                .iter()
                .any(|(pattern, hint)| pattern == "compile_error_E0412"
                    && hint.contains("cannot find type"))
        );
        assert!(
            context
                .source_files
                .iter()
                .any(|(_, snippet)| snippet.contains("pub struct EngineConfig"))
        );
    }

    #[test]
    fn repo_map_turns_lsp_locations_into_precise_snippets() {
        let mut repo = RepoMap::new(".");
        repo.index_source(
            "/tmp/symthaea/src/lib.rs",
            "pub fn normalize_name(name: &str) -> String {\n    name.trim().to_lowercase()\n}\n",
        )
        .unwrap();
        let locations = vec![LspLocation {
            uri: "file:///tmp/symthaea/src/lib.rs".to_string(),
            range: super::super::rust_lsp::LspRange {
                start: super::super::rust_lsp::LspPosition::new(1, 8),
                end: super::super::rust_lsp::LspPosition::new(1, 12),
            },
        }];

        let context = repo.code_context_for_lsp_locations(&locations);

        assert!(context.memory.is_some());
        assert!(
            context
                .source_files
                .iter()
                .any(|(label, snippet)| label.contains("normalize_name")
                    && snippet.contains("to_lowercase"))
        );
    }
}
