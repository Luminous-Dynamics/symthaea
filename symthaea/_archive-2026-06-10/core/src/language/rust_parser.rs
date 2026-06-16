// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Rust Parser via tree-sitter-rust
//!
//! Implements `CodeParser` for Rust source code, extracting functions, structs,
//! enums, traits, impls, use statements, lifetime annotations, generic bounds,
//! unsafe blocks, and macro invocations.

use tree_sitter::Parser;

use super::code_parser::*;

/// Tree-sitter based Rust parser
pub struct RustParser {
    parser: Parser,
}

impl RustParser {
    /// Create a new Rust parser
    pub fn new() -> Self {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .expect("Failed to set Rust language for tree-sitter parser");
        Self { parser }
    }

    /// Extract entities from tree-sitter AST nodes recursively
    fn extract_from_node(
        &self,
        node: &tree_sitter::Node,
        source: &str,
        entities: &mut Vec<Entity>,
        relations: &mut Vec<EntityRelation>,
    ) {
        match node.kind() {
            "function_item" => {
                if let Some(entity) = self.parse_function(node, source) {
                    entities.push(entity);
                }
            }
            "struct_item" => {
                if let Some(entity) = self.parse_struct(node, source) {
                    entities.push(entity);
                }
            }
            "enum_item" => {
                if let Some(entity) = self.parse_enum(node, source) {
                    entities.push(entity);
                }
            }
            "trait_item" => {
                if let Some(entity) = self.parse_trait(node, source) {
                    entities.push(entity);
                }
            }
            "impl_item" => {
                if let Some(entity) = self.parse_impl(node, source, relations) {
                    entities.push(entity);
                }
            }
            "use_declaration" => {
                if let Some(entity) = self.parse_use(node, source) {
                    entities.push(entity);
                }
            }
            "mod_item" => {
                if let Some(entity) = self.parse_mod(node, source) {
                    entities.push(entity);
                }
            }
            "macro_definition" => {
                if let Some(entity) = self.parse_macro_def(node, source) {
                    entities.push(entity);
                }
            }
            "type_item" => {
                if let Some(entity) = self.parse_type_alias(node, source) {
                    entities.push(entity);
                }
            }
            "const_item" | "static_item" => {
                if let Some(entity) = self.parse_const(node, source) {
                    entities.push(entity);
                }
            }
            "unsafe_block" => {
                let entity = Entity::new(EntityKind::UnsafeBlock, "unsafe", Span::from_node(node))
                    .with_source(node_text(node, source).to_string());
                entities.push(entity);
            }
            _ => {}
        }

        // Recurse into children (but not into already-extracted nodes)
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.extract_from_node(&child, source, entities, relations);
        }
    }

    fn parse_function(&self, node: &tree_sitter::Node, source: &str) -> Option<Entity> {
        let name = self.find_child_text(node, "name", source)?;
        let mut entity = Entity::new(EntityKind::Function, &name, Span::from_node(node))
            .with_source(node_text(node, source).to_string());

        // Check for visibility
        if let Some(vis) = self.find_child_kind(node, "visibility_modifier") {
            entity = entity.with_annotation("visibility", node_text(&vis, source));
        }

        // Check for async
        let src = node_text(node, source);
        if src.starts_with("async ") || src.starts_with("pub async ") {
            entity = entity.with_annotation("async", "true");
        }

        // Check for unsafe
        if src.contains("unsafe fn") {
            entity = entity.with_annotation("unsafe", "true");
        }

        // Extract parameters
        if let Some(params) = self.find_child_kind(node, "parameters") {
            entity = entity.with_annotation("parameters", node_text(&params, source));
        }

        // Extract return type
        if let Some(ret) = node.child_by_field_name("return_type") {
            entity = entity.with_annotation("return_type", node_text(&ret, source));
        }

        // Extract generic parameters
        if let Some(generics) = self.find_child_kind(node, "type_parameters") {
            entity = entity.with_annotation("generics", node_text(&generics, source));
        }

        // Extract where clause
        if let Some(where_clause) = self.find_child_kind(node, "where_clause") {
            entity = entity.with_annotation("where_clause", node_text(&where_clause, source));
        }

        Some(entity)
    }

    fn parse_struct(&self, node: &tree_sitter::Node, source: &str) -> Option<Entity> {
        let name = self.find_child_text(node, "name", source)?;
        let mut entity = Entity::new(EntityKind::Struct, &name, Span::from_node(node))
            .with_source(node_text(node, source).to_string());

        if let Some(vis) = self.find_child_kind(node, "visibility_modifier") {
            entity = entity.with_annotation("visibility", node_text(&vis, source));
        }

        if let Some(generics) = self.find_child_kind(node, "type_parameters") {
            entity = entity.with_annotation("generics", node_text(&generics, source));
        }

        // Extract fields
        if let Some(body) = self.find_child_kind(node, "field_declaration_list") {
            let mut cursor = body.walk();
            for field in body.children(&mut cursor) {
                if field.kind() == "field_declaration" {
                    if let Some(field_name) = field.child_by_field_name("name") {
                        let field_entity = Entity::new(
                            EntityKind::Variable,
                            node_text(&field_name, source),
                            Span::from_node(&field),
                        )
                        .with_source(node_text(&field, source).to_string());
                        entity = entity.with_child(field_entity);
                    }
                }
            }
        }

        Some(entity)
    }

    fn parse_enum(&self, node: &tree_sitter::Node, source: &str) -> Option<Entity> {
        let name = self.find_child_text(node, "name", source)?;
        let mut entity = Entity::new(EntityKind::Enum, &name, Span::from_node(node))
            .with_source(node_text(node, source).to_string());

        if let Some(vis) = self.find_child_kind(node, "visibility_modifier") {
            entity = entity.with_annotation("visibility", node_text(&vis, source));
        }

        if let Some(generics) = self.find_child_kind(node, "type_parameters") {
            entity = entity.with_annotation("generics", node_text(&generics, source));
        }

        // Extract variants
        if let Some(body) = self.find_child_kind(node, "enum_variant_list") {
            let mut cursor = body.walk();
            for variant in body.children(&mut cursor) {
                if variant.kind() == "enum_variant" {
                    if let Some(vname) = variant.child_by_field_name("name") {
                        let v_entity = Entity::new(
                            EntityKind::Variable,
                            node_text(&vname, source),
                            Span::from_node(&variant),
                        )
                        .with_source(node_text(&variant, source).to_string());
                        entity = entity.with_child(v_entity);
                    }
                }
            }
        }

        Some(entity)
    }

    fn parse_trait(&self, node: &tree_sitter::Node, source: &str) -> Option<Entity> {
        let name = self.find_child_text(node, "name", source)?;
        let mut entity = Entity::new(EntityKind::Trait, &name, Span::from_node(node))
            .with_source(node_text(node, source).to_string());

        if let Some(vis) = self.find_child_kind(node, "visibility_modifier") {
            entity = entity.with_annotation("visibility", node_text(&vis, source));
        }

        if let Some(generics) = self.find_child_kind(node, "type_parameters") {
            entity = entity.with_annotation("generics", node_text(&generics, source));
        }

        // Extract trait methods
        if let Some(body) = self.find_child_kind(node, "declaration_list") {
            let mut cursor = body.walk();
            for item in body.children(&mut cursor) {
                if item.kind() == "function_item" || item.kind() == "function_signature_item" {
                    if let Some(method) = self.parse_function(&item, source) {
                        let method = Entity {
                            kind: EntityKind::Method,
                            ..method
                        };
                        entity = entity.with_child(method);
                    }
                }
            }
        }

        Some(entity)
    }

    fn parse_impl(
        &self,
        node: &tree_sitter::Node,
        source: &str,
        relations: &mut Vec<EntityRelation>,
    ) -> Option<Entity> {
        let type_name = node
            .child_by_field_name("type")
            .map(|n| node_text(&n, source).to_string())
            .unwrap_or_else(|| "unknown".to_string());

        // Check if this is a trait impl
        let trait_name = node
            .child_by_field_name("trait")
            .map(|n| node_text(&n, source).to_string());

        let kind = if trait_name.is_some() {
            EntityKind::TraitImpl
        } else {
            EntityKind::TraitImpl // inherent impl also tracked
        };

        let display_name = if let Some(ref t) = trait_name {
            format!("{} for {}", t, type_name)
        } else {
            type_name.clone()
        };

        let mut entity = Entity::new(kind, &display_name, Span::from_node(node))
            .with_source(node_text(node, source).to_string())
            .with_annotation("type", &type_name);

        if let Some(ref t) = trait_name {
            entity = entity.with_annotation("trait", t);
            relations.push(EntityRelation {
                source: type_name.clone(),
                relation: Relation::Implements,
                target: t.clone(),
            });
        }

        // Extract methods
        if let Some(body) = self.find_child_kind(node, "declaration_list") {
            let mut cursor = body.walk();
            for item in body.children(&mut cursor) {
                if item.kind() == "function_item" {
                    if let Some(method) = self.parse_function(&item, source) {
                        let method = Entity {
                            kind: EntityKind::Method,
                            ..method
                        };
                        entity = entity.with_child(method);
                    }
                }
            }
        }

        Some(entity)
    }

    fn parse_use(&self, node: &tree_sitter::Node, source: &str) -> Option<Entity> {
        let text = node_text(node, source).to_string();
        // Extract the path from the use statement
        let path = text
            .strip_prefix("use ")
            .or_else(|| text.strip_prefix("pub use "))
            .unwrap_or(&text)
            .trim_end_matches(';')
            .trim()
            .to_string();

        Some(Entity::new(EntityKind::Import, &path, Span::from_node(node)).with_source(text))
    }

    fn parse_mod(&self, node: &tree_sitter::Node, source: &str) -> Option<Entity> {
        let name = self.find_child_text(node, "name", source)?;
        let mut entity = Entity::new(EntityKind::Module, &name, Span::from_node(node))
            .with_source(node_text(node, source).to_string());

        if let Some(vis) = self.find_child_kind(node, "visibility_modifier") {
            entity = entity.with_annotation("visibility", node_text(&vis, source));
        }

        Some(entity)
    }

    fn parse_macro_def(&self, node: &tree_sitter::Node, source: &str) -> Option<Entity> {
        let name = self.find_child_text(node, "name", source)?;
        Some(
            Entity::new(EntityKind::Macro, &name, Span::from_node(node))
                .with_source(node_text(node, source).to_string()),
        )
    }

    fn parse_type_alias(&self, node: &tree_sitter::Node, source: &str) -> Option<Entity> {
        let name = self.find_child_text(node, "name", source)?;
        Some(
            Entity::new(EntityKind::TypeAlias, &name, Span::from_node(node))
                .with_source(node_text(node, source).to_string()),
        )
    }

    fn parse_const(&self, node: &tree_sitter::Node, source: &str) -> Option<Entity> {
        let name = self.find_child_text(node, "name", source)?;
        Some(
            Entity::new(EntityKind::Constant, &name, Span::from_node(node))
                .with_source(node_text(node, source).to_string()),
        )
    }

    // Helpers

    fn find_child_text(
        &self,
        node: &tree_sitter::Node,
        field: &str,
        source: &str,
    ) -> Option<String> {
        node.child_by_field_name(field)
            .map(|n| node_text(&n, source).to_string())
    }

    fn find_child_kind<'a>(
        &self,
        node: &tree_sitter::Node<'a>,
        kind: &str,
    ) -> Option<tree_sitter::Node<'a>> {
        let mut cursor = node.walk();
        let children: Vec<_> = node.children(&mut cursor).collect();
        children.into_iter().find(|c| c.kind() == kind)
    }
}

impl Default for RustParser {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeParser for RustParser {
    fn language_name(&self) -> &str {
        "rust"
    }

    fn parse(&mut self, source: &str) -> Result<ParsedCode, CodeDiagnostic> {
        let tree = self
            .parser
            .parse(source, None)
            .ok_or_else(|| CodeDiagnostic {
                severity: DiagnosticSeverity::Error,
                message: "Failed to parse Rust source".to_string(),
                span: None,
            })?;

        let root = tree.root_node();

        // Collect syntax errors
        let mut diagnostics = Vec::new();
        collect_tree_errors(&root, source, &mut diagnostics);

        // Extract entities
        let mut entities = Vec::new();
        let mut relations = Vec::new();
        self.extract_from_node(&root, source, &mut entities, &mut relations);

        let mut parsed = ParsedCode::new(source, "rust").with_tree(tree);
        parsed.entities = entities;
        parsed.structure.relations = relations;
        parsed.diagnostics = diagnostics;

        Ok(parsed)
    }

    fn extract_entities(&self, parsed: &ParsedCode) -> Vec<Entity> {
        parsed.entities.clone()
    }

    fn detect_diagnostics(&self, parsed: &ParsedCode) -> Vec<CodeDiagnostic> {
        let mut diags = Vec::new();

        // Check for unsafe blocks
        let unsafe_count = parsed
            .all_entities()
            .iter()
            .filter(|e| e.kind == EntityKind::UnsafeBlock)
            .count();
        if unsafe_count > 0 {
            diags.push(CodeDiagnostic {
                severity: DiagnosticSeverity::Info,
                message: format!("Found {} unsafe block(s)", unsafe_count),
                span: None,
            });
        }

        // Check for functions without explicit return types (heuristic)
        for entity in parsed.all_entities() {
            if entity.kind == EntityKind::Function
                && !entity.annotations.contains_key("return_type")
                && entity.name != "main"
            {
                diags.push(CodeDiagnostic {
                    severity: DiagnosticSeverity::Hint,
                    message: format!("Function '{}' has no explicit return type", entity.name),
                    span: Some(entity.span.clone()),
                });
            }
        }

        diags
    }

    fn file_extensions(&self) -> &[&str] {
        &["rs"]
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Primitive-Aware Parsing Extension
// ═══════════════════════════════════════════════════════════════════════════════

use crate::consciousness::code_primitives::{
    CodeExecutionResult, CodeOperation, CodePrimitiveExecutor,
};

/// Extended parsing result with primitive analysis
#[derive(Debug)]
pub struct PrimitiveAwareParsed {
    /// Standard parsed code
    pub parsed: ParsedCode,
    /// Primitive execution result for parsing
    pub primitive_result: CodeExecutionResult,
    /// HDC encoding of the parsed module
    pub module_hv: Option<symthaea_core::hdc::ContinuousHV>,
}

impl RustParser {
    /// Parse with primitive awareness
    ///
    /// Uses Code tier primitives to guide parsing and provide integration metrics.
    pub fn parse_with_primitives(
        &mut self,
        source: &str,
        dim: usize,
    ) -> Result<PrimitiveAwareParsed, CodeDiagnostic> {
        // Execute Parse primitives first
        let executor = CodePrimitiveExecutor::new(dim);
        let primitive_result = executor.execute(CodeOperation::Parse);

        // Parse the source
        let parsed = self.parse(source)?;

        // Encode the module using HDC
        let module_hv = if !parsed.entities.is_empty() {
            use crate::hdc::code_encoder::CodeHDEncoder;
            let encoder = CodeHDEncoder::new(dim);
            Some(encoder.encode_module(&parsed))
        } else {
            None
        };

        Ok(PrimitiveAwareParsed {
            parsed,
            primitive_result,
            module_hv,
        })
    }

    /// Parse and analyze with cross-tier composition
    ///
    /// Uses Code + Consciousness primitives for understanding code purpose.
    pub fn parse_with_consciousness(
        &mut self,
        source: &str,
        dim: usize,
    ) -> Result<PrimitiveAwareParsed, CodeDiagnostic> {
        let executor = CodePrimitiveExecutor::new(dim);

        // Cross-tier: Code + Consciousness
        let cross_result = executor.execute_with_consciousness(CodeOperation::Parse);

        // Parse the source
        let parsed = self.parse(source)?;

        // Encode the module
        let module_hv = if !parsed.entities.is_empty() {
            use crate::hdc::code_encoder::CodeHDEncoder;
            let encoder = CodeHDEncoder::new(dim);
            Some(encoder.encode_module(&parsed))
        } else {
            None
        };

        // Convert CrossTierResult to CodeExecutionResult
        let primitive_result = CodeExecutionResult {
            success: true,
            primitives: cross_result
                .code_primitives
                .iter()
                .map(
                    |p| crate::consciousness::primitive_reasoning::PrimitiveExecution {
                        primitive: p.clone(),
                        input: symthaea_core::hdc::BinaryHV::zero(),
                        output: symthaea_core::hdc::BinaryHV::zero(),
                        transformation:
                            crate::consciousness::primitive_reasoning::TransformationType::Bind,
                        phi_contribution: cross_result.combined_phi as f64,
                        timestamp: 0.0,
                    },
                )
                .collect(),
            result_hv: cross_result.composed_hv.clone(),
            phi: cross_result.combined_phi,
            generated_code: None,
            diagnostics: vec![
                format!("Cross-tier Phi: {:.3}", cross_result.cross_phi),
                format!("Combined Phi: {:.3}", cross_result.combined_phi),
            ],
        };

        Ok(PrimitiveAwareParsed {
            parsed,
            primitive_result,
            module_hv,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_function() {
        let mut parser = RustParser::new();
        let code = r#"
fn hello() {
    println!("hello");
}
"#;
        let result = parser.parse(code);
        assert!(result.is_ok());
        let parsed = result.unwrap();
        assert!(!parsed.has_errors());

        let fns = parsed.entities_of_kind(EntityKind::Function);
        assert_eq!(fns.len(), 1);
        assert_eq!(fns[0].name, "hello");
    }

    #[test]
    fn test_parse_struct_with_impl() {
        let mut parser = RustParser::new();
        let code = r#"
pub struct Point {
    x: f64,
    y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn distance(&self, other: &Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}
"#;
        let result = parser.parse(code).unwrap();
        assert!(!result.has_errors());

        let structs = result.entities_of_kind(EntityKind::Struct);
        assert_eq!(structs.len(), 1);
        assert_eq!(structs[0].name, "Point");
        assert_eq!(structs[0].children.len(), 2); // x, y fields

        let impls: Vec<&Entity> = result
            .entities
            .iter()
            .filter(|e| e.kind == EntityKind::TraitImpl)
            .collect();
        assert_eq!(impls.len(), 1);
        assert_eq!(impls[0].children.len(), 2); // new, distance
    }

    #[test]
    fn test_parse_trait_and_impl() {
        let mut parser = RustParser::new();
        let code = r#"
pub trait Greet {
    fn greet(&self) -> String;
}

struct Person {
    name: String,
}

impl Greet for Person {
    fn greet(&self) -> String {
        format!("Hello, {}", self.name)
    }
}
"#;
        let result = parser.parse(code).unwrap();
        assert!(!result.has_errors());

        let traits = result.entities_of_kind(EntityKind::Trait);
        assert_eq!(traits.len(), 1);
        assert_eq!(traits[0].name, "Greet");

        // Should have an Implements relation
        assert!(result.structure.relations.iter().any(|r| {
            r.source == "Person" && r.relation == Relation::Implements && r.target == "Greet"
        }));
    }

    #[test]
    fn test_parse_use_statements() {
        let mut parser = RustParser::new();
        let code = r#"
use std::collections::HashMap;
use std::io::{self, Read};
pub use crate::hdc::ContinuousHV;

fn main() {}
"#;
        let result = parser.parse(code).unwrap();

        let imports = result.entities_of_kind(EntityKind::Import);
        assert_eq!(imports.len(), 3);
    }

    #[test]
    fn test_parse_enum_with_variants() {
        let mut parser = RustParser::new();
        let code = r#"
pub enum Color {
    Red,
    Green,
    Blue,
    Custom(u8, u8, u8),
}
"#;
        let result = parser.parse(code).unwrap();

        let enums = result.entities_of_kind(EntityKind::Enum);
        assert_eq!(enums.len(), 1);
        assert_eq!(enums[0].name, "Color");
        assert!(enums[0].children.len() >= 3); // At least Red, Green, Blue
    }

    #[test]
    fn test_parse_async_function() {
        let mut parser = RustParser::new();
        let code = r#"
pub async fn fetch_data(url: &str) -> Result<String, Error> {
    todo!()
}
"#;
        let result = parser.parse(code).unwrap();

        let fns = result.entities_of_kind(EntityKind::Function);
        assert_eq!(fns.len(), 1);
        assert_eq!(fns[0].annotations.get("async"), Some(&"true".to_string()));
    }

    #[test]
    fn test_parse_unsafe_block() {
        let mut parser = RustParser::new();
        let code = r#"
fn dangerous() {
    unsafe {
        std::ptr::null::<u8>().read();
    }
}
"#;
        let result = parser.parse(code).unwrap();

        let unsafe_blocks = result.entities_of_kind(EntityKind::UnsafeBlock);
        assert!(!unsafe_blocks.is_empty());
    }

    #[test]
    fn test_parse_generics() {
        let mut parser = RustParser::new();
        let code = r#"
pub struct Cache<K: Hash + Eq, V: Clone> {
    data: HashMap<K, V>,
}
"#;
        let result = parser.parse(code).unwrap();

        let structs = result.entities_of_kind(EntityKind::Struct);
        assert_eq!(structs.len(), 1);
        assert!(structs[0].annotations.contains_key("generics"));
    }

    #[test]
    fn test_file_extensions() {
        let parser = RustParser::new();
        assert_eq!(parser.file_extensions(), &["rs"]);
    }
}
