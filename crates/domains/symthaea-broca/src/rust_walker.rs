// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tree-sitter based AST walker for Rust. Implements LanguageWalker trait for GenericStructuralScorer.
//! Extracts structural elements with dotted paths for comparison (e.g. "Foo.bar").

use tree_sitter::{Node, Parser};

pub trait LanguageWalker {
    fn extract_elements(&mut self, code: &str) -> Vec<StructuralElement>;
    fn language_name(&self) -> &'static str;
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructuralElement {
    pub kind: String,
    pub dotted_path: String,
    pub value_hash: u64, // stable hash of literal value for exact matching
    pub line: usize,
}

pub struct RustWalker {
    parser: Parser,
}

impl Default for RustWalker {
    fn default() -> Self {
        Self::new()
    }
}

impl RustWalker {
    pub fn new() -> Self {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .expect("Failed to load tree-sitter-rust grammar");
        Self { parser }
    }
}

impl LanguageWalker for RustWalker {
    fn extract_elements(&mut self, code: &str) -> Vec<StructuralElement> {
        let tree = self.parser.parse(code, None).expect("Parse failed");
        let mut elements = Vec::new();
        Self::walk_node(&tree.root_node(), &mut elements, "", code.as_bytes());
        elements
    }

    fn language_name(&self) -> &'static str {
        "rust"
    }
}

impl RustWalker {
    fn walk_node(
        node: &Node,
        elements: &mut Vec<StructuralElement>,
        current_path: &str,
        source: &[u8],
    ) {
        let kind = node.kind().to_string();

        // Focus on high-value structural nodes for code comparison
        match kind.as_str() {
            "function_item" | "struct_item" | "enum_item" | "impl_item" | "mod_item"
            | "const_item" | "static_item" => {
                let name_node = node.child_by_field_name("name").or_else(|| node.child(0));
                if let Some(name) = name_node {
                    let name_text = name.utf8_text(source).unwrap_or("unknown");
                    let path = if current_path.is_empty() {
                        name_text.to_string()
                    } else {
                        format!("{}.{}", current_path, name_text)
                    };

                    let value_hash = Self::hash_node(node, source);

                    elements.push(StructuralElement {
                        kind: kind.clone(),
                        dotted_path: path.clone(),
                        value_hash,
                        line: node.start_position().row + 1,
                    });

                    // Recurse into children with updated path
                    for i in 0..node.child_count() {
                        if let Some(child) = node.child(i) {
                            Self::walk_node(&child, elements, &path, source);
                        }
                    }
                    return; // avoid double counting
                }
            }
            "field_declaration" => {
                if let Some(name) = node.child_by_field_name("name") {
                    let name_text = name.utf8_text(source).unwrap_or("");
                    let path = if current_path.is_empty() {
                        name_text.to_string()
                    } else {
                        format!("{}.{}", current_path, name_text)
                    };
                    elements.push(StructuralElement {
                        kind: "field".to_string(),
                        dotted_path: path,
                        value_hash: Self::hash_node(node, source),
                        line: node.start_position().row + 1,
                    });
                }
            }
            _ => {}
        }

        // Always recurse
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                Self::walk_node(&child, elements, current_path, source);
            }
        }
    }

    fn hash_node(node: &Node, source: &[u8]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        if let Ok(text) = node.utf8_text(source) {
            text.hash(&mut hasher);
        } else {
            node.kind().hash(&mut hasher);
        }
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_walker_basic() {
        let code = r#"
        pub fn hello() {}
        pub struct Foo { x: i32 }
        "#;
        let mut walker = RustWalker::new();
        let elems = walker.extract_elements(code);
        assert!(!elems.is_empty());
        assert!(elems.iter().any(|e| e.dotted_path.contains("hello")));
        assert!(elems.iter().any(|e| e.kind == "struct_item"));
    }
}
