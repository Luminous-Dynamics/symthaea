// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unified Code Parser Trait
//!
//! Provides a language-agnostic interface for parsing source code into ASTs,
//! extracting entities, and encoding to HDC space. Generalizes the pattern
//! from `nix_parser.rs` into a reusable trait for Rust, Python, Nix, and beyond.
//!
//! # Architecture
//!
//! ```text
//! Source Code → CodeParser::parse() → ParsedCode (AST + entities)
//!                                        ↓
//!                           CodeParser::to_hdv() → ContinuousHV (16,384D)
//! ```

use std::collections::HashMap;
use std::fmt;

use tree_sitter::Tree;

/// Span in source text (byte offsets + line/col positions)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    /// Start byte offset
    pub start_byte: usize,
    /// End byte offset
    pub end_byte: usize,
    /// Start line (0-indexed)
    pub start_line: usize,
    /// Start column (0-indexed)
    pub start_col: usize,
    /// End line (0-indexed)
    pub end_line: usize,
    /// End column (0-indexed)
    pub end_col: usize,
}

impl Span {
    /// Create a new span from tree-sitter node positions
    pub fn from_node(node: &tree_sitter::Node) -> Self {
        let start = node.start_position();
        let end = node.end_position();
        Self {
            start_byte: node.start_byte(),
            end_byte: node.end_byte(),
            start_line: start.row,
            start_col: start.column,
            end_line: end.row,
            end_col: end.column,
        }
    }

    /// Source text length in bytes
    pub fn len(&self) -> usize {
        self.end_byte.saturating_sub(self.start_byte)
    }

    /// Whether this span is empty
    pub fn is_empty(&self) -> bool {
        self.start_byte >= self.end_byte
    }
}

/// What kind of code entity this is
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityKind {
    // Common
    Function,
    Method,
    Struct,
    Class,
    Enum,
    Interface,
    Trait,
    TraitImpl,
    Module,
    Import,
    Variable,
    Constant,
    TypeAlias,
    Macro,
    Comment,
    DocComment,

    // Rust-specific
    Lifetime,
    GenericParam,
    UnsafeBlock,
    AsyncBlock,
    ClosureExpr,

    // Python-specific
    Decorator,
    Comprehension,
    WithStatement,
    MatchStatement,

    // Nix-specific
    Binding,
    AttrSet,
    LetExpression,
    WithExpression,
    FlakeInput,
    FlakeOutput,
    Derivation,

    // Fallback
    Other,
}

impl fmt::Display for EntityKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// A unified representation of an extracted entity, regardless of source (Code or Config).
#[derive(Debug, Clone)]
pub struct Entity {
    /// What kind of entity
    pub kind: EntityKind,
    /// Name of the entity (function name, struct name, etc.)
    pub name: String,
    /// Location in source
    pub span: Span,
    /// Raw source text of this entity
    pub source_text: String,
    /// Value associated with this entity (used primarily for configuration data)
    pub value: Option<String>,
    /// Child entities (e.g., methods inside a struct impl)
    pub children: Vec<Entity>,
    /// Arbitrary annotations (e.g., "visibility" → "pub", "async" → "true")
    pub annotations: HashMap<String, String>,
}

impl Entity {
    /// Create a new entity
    pub fn new(kind: EntityKind, name: impl Into<String>, span: Span) -> Self {
        Self {
            kind,
            name: name.into(),
            source_text: String::new(),
            span,
            value: None,
            children: Vec::new(),
            annotations: HashMap::new(),
        }
    }

    /// Set source text
    pub fn with_source(mut self, text: impl Into<String>) -> Self {
        self.source_text = text.into();
        self
    }

    /// Set an optional value (useful for config data)
    pub fn with_value(mut self, value: impl Into<String>) -> Self {
        self.value = Some(value.into());
        self
    }

    /// Add an annotation
    pub fn with_annotation(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.annotations.insert(key.into(), value.into());
        self
    }

    /// Add a child entity
    pub fn with_child(mut self, child: Entity) -> Self {
        self.children.push(child);
        self
    }

    /// Count total entities (self + all descendants)
    pub fn total_count(&self) -> usize {
        1 + self.children.iter().map(|c| c.total_count()).sum::<usize>()
    }
}

/// Relationships between entities (Code or Config)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Relation {
    /// A calls B
    Calls,
    /// A imports B
    Imports,
    /// A implements B (trait/interface)
    Implements,
    /// A extends/inherits from B
    Extends,
    /// A references B (uses a type)
    References,
    /// A contains B (structural nesting)
    Contains,
    /// A depends on B (general dependency)
    DependsOn,
    /// A is associated with B (generic relation)
    AssociatedWith,
}

/// A relationship between two entities
#[derive(Debug, Clone)]
pub struct EntityRelation {
    /// Source entity name
    pub source: String,
    /// Relation type
    pub relation: Relation,
    /// Target entity name
    pub target: String,
}

/// Structure information about code and configuration (dependency graph, call graph)
#[derive(Debug, Clone, Default)]
pub struct CodeStructure {
    /// Relationships found between entities
    pub relations: Vec<EntityRelation>,
    /// Module/namespace hierarchy
    pub module_path: Vec<String>,
}

/// Severity of a diagnostic
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
    Info,
    Hint,
}

/// A diagnostic message from parsing
#[derive(Debug, Clone)]
pub struct CodeDiagnostic {
    /// Severity level
    pub severity: DiagnosticSeverity,
    /// Human-readable message
    pub message: String,
    /// Location in source
    pub span: Option<Span>,
}

impl std::fmt::Display for CodeDiagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(ref span) = self.span {
            write!(
                f,
                "[{:?}] {} at {}:{}",
                self.severity,
                self.message,
                span.start_line + 1,
                span.start_col + 1
            )
        } else {
            write!(f, "[{:?}] {}", self.severity, self.message)
        }
    }
}

/// The result of parsing source code
#[derive(Debug, Clone)]
pub struct ParsedCode {
    /// Raw tree-sitter AST (if available)
    tree: Option<Tree>,
    /// Original source text
    pub source: String,
    /// Language name
    pub language: String,
    /// Extracted entities
    pub entities: Vec<Entity>,
    /// Structural relationships
    pub structure: CodeStructure,
    /// Diagnostics (errors, warnings)
    pub diagnostics: Vec<CodeDiagnostic>,
}

impl ParsedCode {
    /// Create a new ParsedCode
    pub fn new(source: impl Into<String>, language: impl Into<String>) -> Self {
        Self {
            tree: None,
            source: source.into(),
            language: language.into(),
            entities: Vec::new(),
            structure: CodeStructure::default(),
            diagnostics: Vec::new(),
        }
    }

    /// Set the tree-sitter AST
    pub fn with_tree(mut self, tree: Tree) -> Self {
        self.tree = Some(tree);
        self
    }

    /// Get the raw tree-sitter Tree
    pub fn tree(&self) -> Option<&Tree> {
        self.tree.as_ref()
    }

    /// Whether parsing produced errors
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.severity == DiagnosticSeverity::Error)
    }

    /// Get all entities of a specific kind
    pub fn entities_of_kind(&self, kind: EntityKind) -> Vec<&Entity> {
        self.entities.iter().filter(|e| e.kind == kind).collect()
    }

    /// Total number of entities (including nested)
    pub fn total_entity_count(&self) -> usize {
        self.entities.iter().map(|e| e.total_count()).sum()
    }

    /// Get source text for a span
    pub fn source_at(&self, span: &Span) -> &str {
        self.source
            .get(span.start_byte..span.end_byte)
            .unwrap_or("")
    }

    /// Iterate all entities recursively (depth-first)
    pub fn all_entities(&self) -> Vec<&Entity> {
        let mut result = Vec::new();
        fn collect<'a>(entity: &'a Entity, out: &mut Vec<&'a Entity>) {
            out.push(entity);
            for child in &entity.children {
                collect(child, out);
            }
        }
        for entity in &self.entities {
            collect(entity, &mut result);
        }
        result
    }
}

/// Represents a high-level, inferred semantic concept derived from the graph.
#[derive(Debug, Clone)]
pub struct SemanticConcept {
    /// The name of the concept (e.g., "Plasma Stabilization Loop", "User Authentication Flow")
    pub concept_name: String,
    /// A description of the concept's function
    pub description: String,
    /// The entities that form the core of this concept
    pub core_entities: Vec<String>,
    /// The relationship type that defines the concept (e.g., "CausalChain", "DataFlow")
    pub inferred_relation: Relation,
    /// The span where the concept is primarily defined
    pub span: Span,
}

/// Trait for performing deep semantic analysis on a parsed structure.
pub trait SemanticAnalyzer {
    /// Analyzes the parsed code and structure to infer high-level concepts.
    /// This is the core AGI reasoning step.
    fn analyze(&self, parsed: &ParsedCode) -> Vec<SemanticConcept>;

    /// Analyzes the configuration and structure to infer high-level operational modes.
    fn analyze_config(&self, configured: &ConfiguredCode) -> Vec<SemanticConcept>;
}

/// The unified code parser trait.
///
/// Implement this for each language to enable code understanding via
/// tree-sitter AST parsing → entity extraction → HDC encoding.
pub trait CodeParser: Send + Sync {
    /// Name of the language this parser handles
    fn language_name(&self) -> &str;

    /// Parse source code into a ParsedCode with entities and diagnostics
    fn parse(&mut self, source: &str) -> Result<ParsedCode, CodeDiagnostic>;

    /// Parse incrementally given a previous tree (for editor integration)
    fn parse_incremental(
        &mut self,
        source: &str,
        _old_tree: &Tree,
    ) -> Result<ParsedCode, CodeDiagnostic> {
        // Default: fall back to full parse
        self.parse(source)
    }

    /// Extract high-level entities from parsed code
    fn extract_entities(&self, parsed: &ParsedCode) -> Vec<Entity>;

    /// Detect errors and warnings beyond syntax (semantic analysis)
    fn detect_diagnostics(&self, parsed: &ParsedCode) -> Vec<CodeDiagnostic>;

    /// File extensions this parser handles
    fn file_extensions(&self) -> &[&str];
}

/// Helper: collect all tree-sitter error/missing nodes into diagnostics
pub fn collect_tree_errors(
    node: &tree_sitter::Node,
    source: &str,
    diagnostics: &mut Vec<CodeDiagnostic>,
) {
    if node.is_error() {
        let text: String = source[node.byte_range()].chars().take(50).collect();
        diagnostics.push(CodeDiagnostic {
            severity: DiagnosticSeverity::Error,
            message: format!("Syntax error near: {}", text),
            span: Some(Span::from_node(node)),
        });
    }

    if node.is_missing() {
        diagnostics.push(CodeDiagnostic {
            severity: DiagnosticSeverity::Error,
            message: format!("Missing expected token: {}", node.kind()),
            span: Some(Span::from_node(node)),
        });
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_tree_errors(&child, source, diagnostics);
    }
}

/// Helper: extract text from a tree-sitter node
pub fn node_text<'a>(node: &tree_sitter::Node, source: &'a str) -> &'a str {
    source.get(node.byte_range()).unwrap_or("")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_from_values() {
        let span = Span {
            start_byte: 0,
            end_byte: 10,
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 10,
        };
        assert_eq!(span.len(), 10);
        assert!(!span.is_empty());
    }

    #[test]
    fn test_entity_builder() {
        let span = Span {
            start_byte: 0,
            end_byte: 50,
            start_line: 0,
            start_col: 0,
            end_line: 2,
            end_col: 1,
        };
        let entity = Entity::new(EntityKind::Function, "my_func", span)
            .with_source("fn my_func() {}")
            .with_annotation("visibility", "pub");

        assert_eq!(entity.kind, EntityKind::Function);
        assert_eq!(entity.name, "my_func");
        assert_eq!(
            entity.annotations.get("visibility"),
            Some(&"pub".to_string())
        );
        assert_eq!(entity.total_count(), 1);
    }

    #[test]
    fn test_entity_with_children() {
        let span = Span {
            start_byte: 0,
            end_byte: 100,
            start_line: 0,
            start_col: 0,
            end_line: 10,
            end_col: 1,
        };
        let child_span = Span {
            start_byte: 10,
            end_byte: 50,
            start_line: 1,
            start_col: 4,
            end_line: 3,
            end_col: 5,
        };

        let entity = Entity::new(EntityKind::Struct, "MyStruct", span)
            .with_child(Entity::new(
                EntityKind::Method,
                "new",
                child_span.clone(),
            ))
            .with_child(Entity::new(EntityKind::Method, "run", child_span));

        assert_eq!(entity.total_count(), 3);
    }

    #[test]
    fn test_parsed_code_entities_of_kind() {
        let span = Span {
            start_byte: 0,
            end_byte: 10,
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 10,
        };

        let mut parsed = ParsedCode::new("fn a() {} fn b() {} struct C {}", "rust");
        parsed.entities.push(Entity::new(EntityKind::Function, "a", span.clone()));
        parsed.entities.push(Entity::new(EntityKind::Function, "b", span.clone()));
        parsed.entities.push(Entity::new(EntityKind::Struct, "C", span));

        assert_eq!(parsed.entities_of_kind(EntityKind::Function).len(), 2);
        assert_eq!(parsed.entities_of_kind(EntityKind::Struct).len(), 1);
        assert_eq!(parsed.entities_of_kind(EntityKind::Enum).len(), 0);
    }

    #[test]
    fn test_parsed_code_all_entities() {
        let span = Span {
            start_byte: 0,
            end_byte: 10,
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 10,
        };

        let mut parsed = ParsedCode::new("", "rust");
        parsed.entities.push(
            Entity::new(EntityKind::Struct, "S", span.clone())
                .with_child(Entity::new(EntityKind::Method, "m1", span.clone()))
                .with_child(Entity::new(EntityKind::Method, "m2", span)),
        );

        assert_eq!(parsed.all_entities().len(), 3);
        assert_eq!(parsed.total_entity_count(), 3);
    }
}
