// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tree-sitter walker for Python. Extracts functions, classes, etc. with dotted paths.
//! Compatible with GenericStructuralScorer for multi-lang code verification.

use tree_sitter::{Node, Parser};

pub use crate::rust_walker::{LanguageWalker, StructuralElement}; // reuse trait + struct

pub struct PythonWalker {
    parser: Parser,
}

impl PythonWalker {
    pub fn new() -> Self {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .expect("Failed to load tree-sitter-python grammar");
        Self { parser }
    }
}

impl LanguageWalker for PythonWalker {
    fn extract_elements(&mut self, code: &str) -> Vec<StructuralElement> {
        let tree = self.parser.parse(code, None).expect("Parse failed");
        let mut elements = Vec::new();
        Self::walk_node(&tree.root_node(), &mut elements, "", code.as_bytes());
        elements
    }

    fn language_name(&self) -> &'static str {
        "python"
    }
}

impl PythonWalker {
    fn walk_node(
        node: &Node,
        elements: &mut Vec<StructuralElement>,
        current_path: &str,
        source: &[u8],
    ) {
        let kind = node.kind();
        if kind == "function_definition" || kind == "class_definition" {
            if let Some(name) = node.child_by_field_name("name") {
                let name_text = name.utf8_text(source).unwrap_or("unknown");
                let path = if current_path.is_empty() {
                    name_text.to_string()
                } else {
                    format!("{}.{}", current_path, name_text)
                };
                elements.push(StructuralElement {
                    kind: kind.to_string(),
                    dotted_path: path.clone(),
                    value_hash: Self::hash_node(node, source),
                    line: node.start_position().row + 1,
                });
                for i in 0..node.child_count() {
                    if let Some(child) = node.child(i) {
                        Self::walk_node(&child, elements, &path, source);
                    }
                }
                return;
            }
        }
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
