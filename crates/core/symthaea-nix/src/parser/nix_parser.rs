// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tree-sitter based Nix Parser
//!
//! Provides proper AST-based parsing of Nix expressions and configurations
//! using tree-sitter-nix for accurate syntax analysis.
//!
//! ## Features
//!
//! - Full Nix expression parsing
//! - Attribute set extraction
//! - Option path resolution
//! - Import tracking
//! - Error recovery and reporting
//!
//! ## Example
//!
//! ```no_run
//! use symthaea_nix::parser::nix_parser::NixParser;
//!
//! let mut parser = NixParser::new();
//! let config = parser.parse(r#"
//!     { config, pkgs, ... }: {
//!       services.nginx.enable = true;
//!       environment.systemPackages = with pkgs; [ vim git ];
//!     }
//! "#).unwrap();
//!
//! for option in &config.options {
//!     println!("{} = {:?}", option.path, option.value);
//! }
//! ```

use std::collections::HashMap;
use tree_sitter::{Node, Parser, Tree};

/// Nix AST Parser using tree-sitter
pub struct NixParser {
    /// Internal tree-sitter parser (pub for shell module access)
    pub(crate) parser: Parser,
}

/// Parsed Nix configuration
#[derive(Debug, Clone)]
pub struct NixConfig {
    /// Extracted options (path -> value)
    pub options: Vec<NixOption>,

    /// Import statements found
    pub imports: Vec<String>,

    /// Module arguments (config, pkgs, lib, etc.)
    pub module_args: Vec<String>,

    /// Raw AST for advanced queries
    tree: Option<Tree>,

    /// Source text for node extraction
    source: String,

    /// Parse errors encountered
    pub errors: Vec<NixParseError>,
}

/// A single Nix option/attribute
#[derive(Debug, Clone)]
pub struct NixOption {
    /// Dot-separated path (e.g., "services.nginx.enable")
    pub path: String,

    /// The value as a string representation
    pub value: NixValue,

    /// Line number in source (1-indexed)
    pub line: usize,

    /// Column number in source (1-indexed)
    pub column: usize,

    /// Raw source text of the value
    pub raw_value: String,
}

/// Typed Nix value
#[derive(Debug, Clone)]
pub enum NixValue {
    /// Boolean (true/false)
    Bool(bool),

    /// Integer
    Int(i64),

    /// String (quoted or multiline)
    String(String),

    /// Path (/path/to/file or ./relative)
    Path(String),

    /// List of values [ ... ]
    List(Vec<NixValue>),

    /// Attribute set { ... }
    AttrSet(HashMap<String, NixValue>),

    /// Function application (e.g., pkgs.firefox)
    Apply(String),

    /// With expression (with pkgs; [...])
    With { scope: String, body: Box<NixValue> },

    /// Null value
    Null,

    /// Unknown/complex expression
    Expression(String),
}

/// Parse error information
#[derive(Debug, Clone)]
pub struct NixParseError {
    /// Error message
    pub message: String,

    /// Line number (1-indexed)
    pub line: usize,

    /// Column number (1-indexed)
    pub column: usize,

    /// Error severity
    pub severity: ErrorSeverity,
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    /// Syntax error that prevents parsing
    Error,

    /// Potential issue but parseable
    Warning,

    /// Informational only
    Info,
}

impl NixParser {
    /// Create a new Nix parser
    pub fn new() -> Self {
        let mut parser = Parser::new();
        // Set the Nix language for parsing
        parser
            .set_language(&tree_sitter_nix::LANGUAGE.into())
            .expect("Failed to set Nix language for tree-sitter parser");
        Self { parser }
    }

    /// Parse Nix source code
    pub fn parse(&mut self, source: &str) -> Result<NixConfig, NixParseError> {
        let tree = self
            .parser
            .parse(source, None)
            .ok_or_else(|| NixParseError {
                message: "Failed to parse Nix source".to_string(),
                line: 1,
                column: 1,
                severity: ErrorSeverity::Error,
            })?;

        let root = tree.root_node();
        let mut config = NixConfig {
            options: Vec::new(),
            imports: Vec::new(),
            module_args: Vec::new(),
            tree: Some(tree.clone()),
            source: source.to_string(),
            errors: Vec::new(),
        };

        // Collect syntax errors
        Self::collect_errors(&root, source, &mut config.errors);

        // Extract module arguments from function definition
        self.extract_module_args(&root, source, &mut config);

        // Extract imports
        self.extract_imports(&root, source, &mut config);

        // Extract options from attribute sets
        self.extract_options(&root, source, "", &mut config);

        Ok(config)
    }

    /// Collect syntax errors from tree
    fn collect_errors(node: &Node, source: &str, errors: &mut Vec<NixParseError>) {
        if node.is_error() {
            let start = node.start_position();
            errors.push(NixParseError {
                message: format!(
                    "Syntax error near: {}",
                    &source[node.byte_range()]
                        .chars()
                        .take(50)
                        .collect::<String>()
                ),
                line: start.row + 1,
                column: start.column + 1,
                severity: ErrorSeverity::Error,
            });
        }

        if node.is_missing() {
            let start = node.start_position();
            errors.push(NixParseError {
                message: format!("Missing expected token: {}", node.kind()),
                line: start.row + 1,
                column: start.column + 1,
                severity: ErrorSeverity::Error,
            });
        }

        // Recurse into children
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            Self::collect_errors(&child, source, errors);
        }
    }

    /// Extract module arguments from function definition
    fn extract_module_args(&self, node: &Node, source: &str, config: &mut NixConfig) {
        // Look for function definition pattern: { config, pkgs, ... }: body
        if node.kind() == "function_expression" {
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if child.kind() == "formals" {
                    let mut formal_cursor = child.walk();
                    for formal in child.children(&mut formal_cursor) {
                        if formal.kind() == "formal" {
                            // Extract identifier
                            if let Some(id) = formal.child_by_field_name("name") {
                                config.module_args.push(source[id.byte_range()].to_string());
                            }
                        } else if formal.kind() == "identifier" {
                            config
                                .module_args
                                .push(source[formal.byte_range()].to_string());
                        }
                    }
                }
            }
        }

        // Recurse to find function expressions
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.extract_module_args(&child, source, config);
        }
    }

    /// Extract import statements
    fn extract_imports(&self, node: &Node, source: &str, config: &mut NixConfig) {
        // Look for imports = [ ... ] pattern
        if node.kind() == "binding" {
            if let Some(attr_path) = node.child_by_field_name("attrpath") {
                let path_text = source[attr_path.byte_range()].to_string();
                if path_text == "imports" {
                    if let Some(value) = node.child_by_field_name("expression") {
                        self.extract_import_paths(&value, source, config);
                    }
                }
            }
        }

        // Recurse
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.extract_imports(&child, source, config);
        }
    }

    /// Extract paths from import list
    fn extract_import_paths(&self, node: &Node, source: &str, config: &mut NixConfig) {
        if node.kind() == "path_expression" || node.kind() == "path" {
            config.imports.push(source[node.byte_range()].to_string());
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.extract_import_paths(&child, source, config);
        }
    }

    /// Extract options from attribute sets
    fn extract_options(&self, node: &Node, source: &str, prefix: &str, config: &mut NixConfig) {
        if node.kind() == "binding" {
            if let Some(attr_path) = node.child_by_field_name("attrpath") {
                let path_text = source[attr_path.byte_range()].to_string();
                let full_path = if prefix.is_empty() {
                    path_text.clone()
                } else {
                    format!("{prefix}.{path_text}")
                };

                // Skip imports (handled separately)
                if path_text == "imports" {
                    return;
                }

                if let Some(value_node) = node.child_by_field_name("expression") {
                    let start = node.start_position();
                    let raw_value = source[value_node.byte_range()].to_string();
                    let value = self.parse_value(&value_node, source);

                    // If it's an attrset, recurse into it
                    if value_node.kind() == "attrset_expression" {
                        self.extract_options(&value_node, source, &full_path, config);
                    } else {
                        config.options.push(NixOption {
                            path: full_path,
                            value,
                            line: start.row + 1,
                            column: start.column + 1,
                            raw_value,
                        });
                    }
                }
            }
        }

        // Recurse into children
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.extract_options(&child, source, prefix, config);
        }
    }

    /// Parse a value node into NixValue
    fn parse_value(&self, node: &Node, source: &str) -> NixValue {
        match node.kind() {
            "integer_expression" => {
                let text = source[node.byte_range()].trim();
                text.parse::<i64>()
                    .map(NixValue::Int)
                    .unwrap_or(NixValue::Expression(text.to_string()))
            }

            "true" => NixValue::Bool(true),
            "false" => NixValue::Bool(false),
            "null" => NixValue::Null,

            "string_expression" | "indented_string_expression" => {
                let text = source[node.byte_range()].to_string();
                // Remove quotes
                let unquoted = text
                    .trim_matches('"')
                    .trim_start_matches("''")
                    .trim_end_matches("''")
                    .to_string();
                NixValue::String(unquoted)
            }

            "path_expression" | "path" => NixValue::Path(source[node.byte_range()].to_string()),

            "list_expression" => {
                let mut items = Vec::new();
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    if child.kind() != "[" && child.kind() != "]" {
                        items.push(self.parse_value(&child, source));
                    }
                }
                NixValue::List(items)
            }

            "attrset_expression" | "rec_attrset_expression" => {
                let mut attrs = HashMap::new();
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    if child.kind() == "binding" {
                        if let Some(path) = child.child_by_field_name("attrpath") {
                            let key = source[path.byte_range()].to_string();
                            if let Some(val) = child.child_by_field_name("expression") {
                                attrs.insert(key, self.parse_value(&val, source));
                            }
                        }
                    }
                }
                NixValue::AttrSet(attrs)
            }

            "with_expression" => {
                let scope = node
                    .child_by_field_name("environment")
                    .map(|n| source[n.byte_range()].to_string())
                    .unwrap_or_default();
                let body = node
                    .child_by_field_name("body")
                    .map(|n| self.parse_value(&n, source))
                    .unwrap_or(NixValue::Null);
                NixValue::With {
                    scope,
                    body: Box::new(body),
                }
            }

            "select_expression" | "apply_expression" => {
                NixValue::Apply(source[node.byte_range()].to_string())
            }

            // In tree-sitter-nix 0.3+, true/false/null are parsed as variable_expression
            // We need to check the text content to identify boolean/null literals
            "variable_expression" | "identifier" => {
                let text = source[node.byte_range()].trim();
                match text {
                    "true" => NixValue::Bool(true),
                    "false" => NixValue::Bool(false),
                    "null" => NixValue::Null,
                    _ => NixValue::Apply(text.to_string()),
                }
            }

            _ => NixValue::Expression(source[node.byte_range()].to_string()),
        }
    }

    /// Check if source has syntax errors
    pub fn has_errors(&mut self, source: &str) -> bool {
        match self.parser.parse(source, None) {
            Some(tree) => tree.root_node().has_error(),
            _ => true,
        }
    }

    /// Get syntax errors only (quick check)
    pub fn get_errors(&mut self, source: &str) -> Vec<NixParseError> {
        match self.parse(source) {
            Ok(config) => config.errors,
            Err(e) => vec![e],
        }
    }
}

impl Default for NixParser {
    fn default() -> Self {
        Self::new()
    }
}

impl NixConfig {
    /// Get all option paths
    pub fn option_paths(&self) -> Vec<&str> {
        self.options.iter().map(|o| o.path.as_str()).collect()
    }

    /// Find option by path
    pub fn get_option(&self, path: &str) -> Option<&NixOption> {
        self.options.iter().find(|o| o.path == path)
    }

    /// Check if an option exists
    pub fn has_option(&self, path: &str) -> bool {
        self.options.iter().any(|o| o.path == path)
    }

    /// Get all options under a prefix
    pub fn options_under(&self, prefix: &str) -> Vec<&NixOption> {
        self.options
            .iter()
            .filter(|o| o.path.starts_with(prefix))
            .collect()
    }

    /// Check if parse had errors
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    // ============================================
    // AST Traversal Methods
    // ============================================

    /// Get the raw tree-sitter Tree for advanced queries
    ///
    /// Returns None if parsing failed or tree is unavailable.
    pub fn tree(&self) -> Option<&Tree> {
        self.tree.as_ref()
    }

    /// Get the source text used for parsing
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Traverse all nodes in the AST in pre-order (depth-first)
    ///
    /// Returns an iterator that yields each node in the tree exactly once.
    /// Nodes are visited in pre-order: parent before children, left to right.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use symthaea_nix::parser::nix_parser::NixParser;
    ///
    /// let mut parser = NixParser::new();
    /// let config = parser.parse("{ x = 1; }").unwrap();
    /// for node in config.traverse_ast() {
    ///     println!("{}: {}", node.kind(), config.get_source_range(&node));
    /// }
    /// ```
    pub fn traverse_ast(&self) -> impl Iterator<Item = Node<'_>> {
        AstIterator::new(self.tree.as_ref())
    }

    /// Find all nodes of a specific kind (type)
    ///
    /// # Arguments
    ///
    /// * `kind` - The tree-sitter node type to search for (e.g., "binding", "identifier", "string_expression")
    ///
    /// # Example
    ///
    /// ```no_run
    /// use symthaea_nix::parser::nix_parser::NixParser;
    ///
    /// let mut parser = NixParser::new();
    /// let config = parser.parse("{ x = \"hello\"; }").unwrap();
    /// let strings = config.find_nodes("string_expression");
    /// for node in strings {
    ///     println!("String: {}", config.get_source_range(&node));
    /// }
    /// ```
    pub fn find_nodes(&self, kind: &str) -> Vec<Node<'_>> {
        self.traverse_ast()
            .filter(|node| node.kind() == kind)
            .collect()
    }

    /// Extract let bindings from the AST
    ///
    /// Returns a vector of (name, value) pairs for all let bindings found.
    /// This includes both `let x = ...;` and `let { x = ...; } in ...` forms.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use symthaea_nix::parser::nix_parser::NixParser;
    ///
    /// let mut parser = NixParser::new();
    /// let config = parser.parse(r#"
    ///     let
    ///       x = 1;
    ///       y = "hello";
    ///     in x + y
    /// "#).unwrap();
    ///
    /// for (name, value) in config.extract_bindings() {
    ///     println!("{} = {}", name, value);
    /// }
    /// ```
    pub fn extract_bindings(&self) -> Vec<(String, String)> {
        let mut bindings = Vec::new();

        // Look for let_expression nodes which contain bindings
        for node in self.traverse_ast() {
            if node.kind() == "let_expression" {
                // Inside let_expression, find the binding_set
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    if child.kind() == "binding_set" || child.kind() == "attrset_expression" {
                        self.extract_bindings_from_set(&child, &mut bindings);
                    }
                }
            }
            // Also capture top-level bindings in attrset expressions
            // (for module-style Nix like { x = 1; })
            else if node.kind() == "binding" {
                if let Some((name, value)) = self.extract_single_binding(&node) {
                    bindings.push((name, value));
                }
            }
        }

        bindings
    }

    /// Helper to extract bindings from a binding_set or attrset
    fn extract_bindings_from_set(&self, node: &Node<'_>, bindings: &mut Vec<(String, String)>) {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "binding" {
                if let Some((name, value)) = self.extract_single_binding(&child) {
                    bindings.push((name, value));
                }
            }
        }
    }

    /// Helper to extract a single binding's name and value
    fn extract_single_binding(&self, binding: &Node<'_>) -> Option<(String, String)> {
        let attr_path = binding.child_by_field_name("attrpath")?;
        let expression = binding.child_by_field_name("expression")?;

        let name = self.get_source_range(&attr_path).to_string();
        let value = self.get_source_range(&expression).to_string();

        Some((name, value))
    }

    /// Get the source text for a specific AST node
    ///
    /// # Arguments
    ///
    /// * `node` - A tree-sitter Node from this config's AST
    ///
    /// # Example
    ///
    /// ```no_run
    /// use symthaea_nix::parser::nix_parser::NixParser;
    ///
    /// let mut parser = NixParser::new();
    /// let config = parser.parse("{ x = 1; y = 2; }").unwrap();
    /// for node in config.find_nodes("identifier") {
    ///     let text = config.get_source_range(&node);
    ///     println!("Identifier: {}", text);
    /// }
    /// ```
    pub fn get_source_range(&self, node: &Node<'_>) -> &str {
        let range = node.byte_range();
        // Safety: byte_range should be valid for the source that was parsed
        self.source.get(range).unwrap_or("")
    }

    /// Get nodes at a specific line and column position
    ///
    /// Returns the deepest node that contains the given position.
    pub fn node_at_position(&self, line: usize, column: usize) -> Option<Node<'_>> {
        let tree = self.tree.as_ref()?;
        let root = tree.root_node();

        // tree-sitter uses 0-indexed positions
        let point = tree_sitter::Point::new(line.saturating_sub(1), column.saturating_sub(1));

        self.find_deepest_node_at_point(&root, point)
    }

    /// Helper to find deepest node containing a point
    fn find_deepest_node_at_point<'a>(
        &self,
        node: &Node<'a>,
        point: tree_sitter::Point,
    ) -> Option<Node<'a>> {
        let start = node.start_position();
        let end = node.end_position();

        // Check if point is within this node
        if point < start || point > end {
            return None;
        }

        // Try to find a more specific child
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if let Some(deeper) = self.find_deepest_node_at_point(&child, point) {
                return Some(deeper);
            }
        }

        // No child contains the point, return this node
        Some(*node)
    }

    /// Get all named children of a node (excludes syntax tokens like brackets, semicolons)
    pub fn named_children<'a>(&'a self, node: &'a Node<'a>) -> impl Iterator<Item = Node<'a>> {
        let mut cursor = node.walk();
        let children: Vec<_> = node
            .children(&mut cursor)
            .filter(|n| n.is_named())
            .collect();
        children.into_iter()
    }
}

/// Iterator for pre-order AST traversal
struct AstIterator<'a> {
    stack: Vec<Node<'a>>,
}

impl<'a> AstIterator<'a> {
    fn new(tree: Option<&'a Tree>) -> Self {
        let stack = match tree {
            Some(t) => vec![t.root_node()],
            None => Vec::new(),
        };
        Self { stack }
    }
}

impl<'a> Iterator for AstIterator<'a> {
    type Item = Node<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.stack.pop()?;

        // Push children in reverse order so they're visited left-to-right
        let mut cursor = node.walk();
        let children: Vec<_> = node.children(&mut cursor).collect();
        for child in children.into_iter().rev() {
            self.stack.push(child);
        }

        Some(node)
    }
}

impl NixValue {
    /// Check if value is truthy
    pub fn is_truthy(&self) -> bool {
        match self {
            NixValue::Bool(b) => *b,
            NixValue::Null => false,
            NixValue::String(s) => !s.is_empty(),
            NixValue::List(l) => !l.is_empty(),
            NixValue::AttrSet(a) => !a.is_empty(),
            _ => true,
        }
    }

    /// Get as boolean
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            NixValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Get as string
    pub fn as_string(&self) -> Option<&str> {
        match self {
            NixValue::String(s) => Some(s),
            NixValue::Path(p) => Some(p),
            _ => None,
        }
    }

    /// Get as list
    pub fn as_list(&self) -> Option<&[NixValue]> {
        match self {
            NixValue::List(l) => Some(l),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_config() {
        let mut parser = NixParser::new();
        let result = parser.parse(
            r#"
            { config, pkgs, ... }: {
                services.nginx.enable = true;
                networking.hostName = "myhost";
            }
        "#,
        );

        assert!(result.is_ok());
        let config = result.unwrap();

        assert!(config.module_args.contains(&"config".to_string()));
        assert!(config.module_args.contains(&"pkgs".to_string()));
    }

    #[test]
    fn test_parse_system_packages() {
        let mut parser = NixParser::new();
        let result = parser.parse(
            r#"
            { pkgs, ... }: {
                environment.systemPackages = with pkgs; [
                    vim
                    git
                    firefox
                ];
            }
        "#,
        );

        assert!(result.is_ok());
        let config = result.unwrap();

        let pkg_option = config.get_option("environment.systemPackages");
        assert!(pkg_option.is_some());
    }

    #[test]
    fn test_parse_imports() {
        let mut parser = NixParser::new();
        let result = parser.parse(
            r#"
            { ... }: {
                imports = [
                    ./hardware-configuration.nix
                    ./networking.nix
                ];
            }
        "#,
        );

        assert!(result.is_ok());
        let config = result.unwrap();

        assert!(!config.imports.is_empty());
    }

    #[test]
    fn test_error_detection() {
        let mut parser = NixParser::new();
        let result = parser.parse(
            r#"
            { ... }: {
                services.nginx.enable = true
                # Missing semicolon above
                networking.hostName = "test";
            }
        "#,
        );

        // Should still parse but report errors
        assert!(result.is_ok());
        let config = result.unwrap();
        assert!(config.has_errors());
    }

    #[test]
    fn test_value_types() {
        let mut parser = NixParser::new();
        let result = parser.parse(
            r#"
            {
                boolVal = true;
                intVal = 42;
                strVal = "hello";
                pathVal = ./path;
                nullVal = null;
            }
        "#,
        );

        assert!(result.is_ok(), "Parse should succeed");
        let config = result.unwrap();

        if let Some(opt) = config.get_option("boolVal") {
            assert!(matches!(opt.value, NixValue::Bool(true)));
        }

        if let Some(opt) = config.get_option("intVal") {
            assert!(matches!(opt.value, NixValue::Int(42)));
        }

        if let Some(opt) = config.get_option("nullVal") {
            assert!(matches!(opt.value, NixValue::Null));
        }
    }

    // ============================================
    // AST Traversal Tests
    // ============================================

    #[test]
    fn test_traverse_ast() {
        let mut parser = NixParser::new();
        let config = parser
            .parse(
                r#"
            { x = 1; y = 2; }
        "#,
            )
            .unwrap();

        // Traverse should yield multiple nodes
        let nodes: Vec<_> = config.traverse_ast().collect();
        assert!(!nodes.is_empty(), "traverse_ast should yield nodes");

        // Should include the root source_code node
        let kinds: Vec<_> = nodes.iter().map(|n| n.kind()).collect();
        assert!(
            kinds.contains(&"source_code") || kinds.contains(&"attrset_expression"),
            "Should contain root or attrset node"
        );
    }

    #[test]
    fn test_find_nodes_by_kind() {
        let mut parser = NixParser::new();
        let config = parser
            .parse(
                r#"
            {
                a = 1;
                b = 2;
                c = 3;
            }
        "#,
            )
            .unwrap();

        // Find all binding nodes
        let bindings = config.find_nodes("binding");
        assert_eq!(bindings.len(), 3, "Should find 3 bindings");

        // Find all integer expressions
        let integers = config.find_nodes("integer_expression");
        assert_eq!(integers.len(), 3, "Should find 3 integers");
    }

    #[test]
    fn test_extract_let_bindings() {
        let mut parser = NixParser::new();
        let config = parser
            .parse(
                r#"
            let
              x = 1;
              y = "hello";
              z = true;
            in x
        "#,
            )
            .unwrap();

        let bindings = config.extract_bindings();
        assert!(!bindings.is_empty(), "Should extract some bindings");

        // Check that we found x, y, z
        let names: Vec<_> = bindings.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"x"), "Should find binding 'x'");
        assert!(names.contains(&"y"), "Should find binding 'y'");
        assert!(names.contains(&"z"), "Should find binding 'z'");
    }

    #[test]
    fn test_get_source_range() {
        let mut parser = NixParser::new();
        let config = parser.parse(r#"{ greeting = "hello world"; }"#).unwrap();

        // Find string expressions
        let strings = config.find_nodes("string_expression");
        assert!(!strings.is_empty(), "Should find string expression");

        let source = config.get_source_range(&strings[0]);
        assert!(
            source.contains("hello world"),
            "Should extract string content"
        );
    }

    #[test]
    fn test_tree_and_source_accessors() {
        let mut parser = NixParser::new();
        let source_text = "{ x = 42; }";
        let config = parser.parse(source_text).unwrap();

        // Tree should be available
        assert!(config.tree().is_some(), "Tree should be present");

        // Source should match input
        assert_eq!(config.source(), source_text, "Source should match input");
    }

    #[test]
    fn test_node_at_position() {
        let mut parser = NixParser::new();
        // Line 1: { x = 42; }
        //          ^
        //          col 3 is 'x'
        let config = parser.parse("{ x = 42; }").unwrap();

        // Position at 'x' (line 1, col 3)
        let node = config.node_at_position(1, 3);
        assert!(node.is_some(), "Should find node at position");

        let node = node.unwrap();
        // Should be an identifier or attrpath
        assert!(
            node.kind() == "identifier" || node.kind() == "attrpath",
            "Node at x position should be identifier or attrpath, got: {}",
            node.kind()
        );
    }

    #[test]
    fn test_traverse_complex_nix() {
        let mut parser = NixParser::new();
        let config = parser
            .parse(
                r#"
            { config, pkgs, lib, ... }:

            let
              myPkg = pkgs.hello;
              enabled = true;
            in
            {
              services.nginx = {
                enable = enabled;
                virtualHosts."example.com" = {
                  root = "/var/www";
                };
              };

              environment.systemPackages = with pkgs; [
                vim
                git
                myPkg
              ];
            }
        "#,
            )
            .unwrap();

        // Should be able to traverse without panic
        let node_count = config.traverse_ast().count();
        assert!(
            node_count > 10,
            "Complex Nix should have many nodes, got {}",
            node_count
        );

        // Should find let expression
        let let_exprs = config.find_nodes("let_expression");
        assert!(!let_exprs.is_empty(), "Should find let expression");

        // Should find with expression
        let with_exprs = config.find_nodes("with_expression");
        assert!(!with_exprs.is_empty(), "Should find with expression");

        // Should find list expression
        let lists = config.find_nodes("list_expression");
        assert!(!lists.is_empty(), "Should find list expression");
    }

    #[test]
    fn test_empty_config_traversal() {
        let mut parser = NixParser::new();
        let config = parser.parse("{}").unwrap();

        // Should still traverse
        let nodes: Vec<_> = config.traverse_ast().collect();
        assert!(!nodes.is_empty(), "Even empty config should have nodes");

        // Find nodes should return empty for non-existent types
        let nonexistent = config.find_nodes("nonexistent_node_type");
        assert!(nonexistent.is_empty(), "Should find no nonexistent nodes");
    }

    // ── NixValue type conversion tests ──

    #[test]
    fn test_is_truthy() {
        assert!(NixValue::Bool(true).is_truthy());
        assert!(!NixValue::Bool(false).is_truthy());
        assert!(!NixValue::Null.is_truthy());
        assert!(!NixValue::String("".into()).is_truthy());
        assert!(NixValue::String("hello".into()).is_truthy());
        assert!(!NixValue::List(vec![]).is_truthy());
        assert!(NixValue::List(vec![NixValue::Int(1)]).is_truthy());
        assert!(NixValue::Int(42).is_truthy());
        assert!(NixValue::Path("/nix/store".into()).is_truthy());
        assert!(NixValue::Expression("pkgs.vim".into()).is_truthy());
    }

    #[test]
    fn test_as_bool() {
        assert_eq!(NixValue::Bool(true).as_bool(), Some(true));
        assert_eq!(NixValue::Bool(false).as_bool(), Some(false));
        assert_eq!(NixValue::Int(1).as_bool(), None);
        assert_eq!(NixValue::String("true".into()).as_bool(), None);
        assert_eq!(NixValue::Null.as_bool(), None);
    }

    #[test]
    fn test_as_string() {
        assert_eq!(NixValue::String("hello".into()).as_string(), Some("hello"));
        assert_eq!(
            NixValue::Path("/etc/nixos".into()).as_string(),
            Some("/etc/nixos")
        );
        assert_eq!(NixValue::Int(42).as_string(), None);
        assert_eq!(NixValue::Bool(true).as_string(), None);
        assert_eq!(NixValue::Null.as_string(), None);
    }

    #[test]
    fn test_as_list() {
        let items = vec![NixValue::Int(1), NixValue::Int(2)];
        let list = NixValue::List(items.clone());
        let result = list.as_list();
        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 2);

        assert!(NixValue::String("not a list".into()).as_list().is_none());
        assert!(NixValue::Null.as_list().is_none());

        // Empty list
        assert_eq!(NixValue::List(vec![]).as_list().unwrap().len(), 0);
    }

    #[test]
    fn test_is_truthy_attrset() {
        use std::collections::HashMap;
        assert!(!NixValue::AttrSet(HashMap::new()).is_truthy());
        let mut map = HashMap::new();
        map.insert("key".to_string(), NixValue::Bool(true));
        assert!(NixValue::AttrSet(map).is_truthy());
    }
}
