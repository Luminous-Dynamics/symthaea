// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Nix CodeParser Adapter
//!
//! Wraps the existing `NixParser` to implement the unified `CodeParser` trait,
//! enabling Nix to participate in the multi-language code understanding pipeline.

use super::code_parser::*;
use super::nix_parser::{NixConfig, NixParser, NixValue};

/// Adapter that makes NixParser implement CodeParser
pub struct NixCodeParser {
    parser: NixParser,
}

impl NixCodeParser {
    /// Create a new Nix code parser
    pub fn new() -> Self {
        Self {
            parser: NixParser::new(),
        }
    }

    /// Convert NixConfig entities to Entity format
    fn convert_entities(&self, config: &NixConfig) -> Vec<Entity> {
        let mut entities = Vec::new();

        // Convert module args to parameters
        for arg in &config.module_args {
            let span = Span {
                start_byte: 0,
                end_byte: 0,
                start_line: 0,
                start_col: 0,
                end_line: 0,
                end_col: 0,
            };
            entities.push(
                Entity::new(EntityKind::Variable, arg, span).with_annotation("kind", "module_arg"),
            );
        }

        // Convert imports
        for import in &config.imports {
            let span = Span {
                start_byte: 0,
                end_byte: 0,
                start_line: 0,
                start_col: 0,
                end_line: 0,
                end_col: 0,
            };
            entities.push(Entity::new(EntityKind::Import, import, span));
        }

        // Convert options to bindings
        for option in &config.options {
            let span = Span {
                start_byte: 0,
                end_byte: 0,
                start_line: option.line.saturating_sub(1),
                start_col: option.column.saturating_sub(1),
                end_line: option.line.saturating_sub(1),
                end_col: option.column.saturating_sub(1) + option.raw_value.len(),
            };

            let kind = self.classify_nix_option(&option.path, &option.value);
            let mut entity = Entity::new(kind, &option.path, span)
                .with_source(option.raw_value.clone())
                .with_annotation(
                    "value_type",
                    &format!("{:?}", std::mem::discriminant(&option.value)),
                );

            // Detect flake-specific patterns
            if option.path.starts_with("inputs.") {
                entity = Entity {
                    kind: EntityKind::FlakeInput,
                    ..entity
                };
            } else if option.path.starts_with("outputs") {
                entity = Entity {
                    kind: EntityKind::FlakeOutput,
                    ..entity
                };
            }

            entities.push(entity);
        }

        // Detect flake structure from source
        if config.source().contains("inputs") && config.source().contains("outputs") {
            let span = Span {
                start_byte: 0,
                end_byte: config.source().len(),
                start_line: 0,
                start_col: 0,
                end_line: 0,
                end_col: 0,
            };
            entities.push(
                Entity::new(EntityKind::Module, "flake", span).with_annotation("type", "flake"),
            );
        }

        entities
    }

    fn classify_nix_option(&self, path: &str, value: &NixValue) -> EntityKind {
        // Check for well-known patterns
        if path.contains("enable") {
            return EntityKind::Binding;
        }
        if path.starts_with("services.")
            || path.starts_with("networking.")
            || path.starts_with("boot.")
            || path.starts_with("environment.")
        {
            return EntityKind::Binding;
        }

        match value {
            NixValue::AttrSet(_) => EntityKind::AttrSet,
            NixValue::With { .. } => EntityKind::WithExpression,
            _ => EntityKind::Binding,
        }
    }
}

impl Default for NixCodeParser {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeParser for NixCodeParser {
    fn language_name(&self) -> &str {
        "nix"
    }

    fn parse(&mut self, source: &str) -> Result<ParsedCode, CodeDiagnostic> {
        let config = self.parser.parse(source).map_err(|e| CodeDiagnostic {
            severity: DiagnosticSeverity::Error,
            message: e.message,
            span: Some(Span {
                start_byte: 0,
                end_byte: 0,
                start_line: e.line.saturating_sub(1),
                start_col: e.column.saturating_sub(1),
                end_line: e.line.saturating_sub(1),
                end_col: e.column.saturating_sub(1),
            }),
        })?;

        let entities = self.convert_entities(&config);

        // Convert NixParseErrors to CodeDiagnostics
        let diagnostics: Vec<CodeDiagnostic> = config
            .errors
            .iter()
            .map(|e| {
                use super::nix_parser::ErrorSeverity;
                CodeDiagnostic {
                    severity: match e.severity {
                        ErrorSeverity::Error => DiagnosticSeverity::Error,
                        ErrorSeverity::Warning => DiagnosticSeverity::Warning,
                        ErrorSeverity::Info => DiagnosticSeverity::Info,
                    },
                    message: e.message.clone(),
                    span: Some(Span {
                        start_byte: 0,
                        end_byte: 0,
                        start_line: e.line.saturating_sub(1),
                        start_col: e.column.saturating_sub(1),
                        end_line: e.line.saturating_sub(1),
                        end_col: e.column.saturating_sub(1),
                    }),
                }
            })
            .collect();

        let tree = config.tree().cloned();
        let mut parsed = ParsedCode::new(source, "nix");
        if let Some(tree) = tree {
            parsed = parsed.with_tree(tree);
        }
        parsed.entities = entities;
        parsed.diagnostics = diagnostics;

        Ok(parsed)
    }

    fn extract_entities(&self, parsed: &ParsedCode) -> Vec<Entity> {
        parsed.entities.clone()
    }

    fn detect_diagnostics(&self, parsed: &ParsedCode) -> Vec<CodeDiagnostic> {
        let mut diags = Vec::new();

        // Check for missing semicolons (common Nix mistake)
        let source = &parsed.source;
        let lines: Vec<&str> = source.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            // Heuristic: attribute binding without semicolon
            if trimmed.contains(" = ")
                && !trimmed.ends_with(';')
                && !trimmed.ends_with('{')
                && !trimmed.ends_with('[')
                && !trimmed.is_empty()
                && !trimmed.starts_with('#')
                && !trimmed.starts_with("let")
                && !trimmed.starts_with("in")
            {
                diags.push(CodeDiagnostic {
                    severity: DiagnosticSeverity::Warning,
                    message: format!("Possible missing semicolon on line {}", i + 1),
                    span: Some(Span {
                        start_byte: 0,
                        end_byte: 0,
                        start_line: i,
                        start_col: 0,
                        end_line: i,
                        end_col: trimmed.len(),
                    }),
                });
            }
        }

        diags
    }

    fn file_extensions(&self) -> &[&str] {
        &["nix"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_nix() {
        let mut parser = NixCodeParser::new();
        let code = r#"
{ config, pkgs, ... }: {
    services.nginx.enable = true;
    environment.systemPackages = with pkgs; [ vim git ];
}
"#;
        let result = parser.parse(code);
        assert!(result.is_ok());
        let parsed = result.unwrap();
        assert_eq!(parsed.language, "nix");
        assert!(!parsed.entities.is_empty());
    }

    #[test]
    fn test_nix_imports_extracted() {
        let mut parser = NixCodeParser::new();
        let code = r#"
{ ... }: {
    imports = [
        ./hardware.nix
        ./networking.nix
    ];
}
"#;
        let result = parser.parse(code).unwrap();
        let imports = result.entities_of_kind(EntityKind::Import);
        assert!(!imports.is_empty());
    }

    #[test]
    fn test_nix_module_args() {
        let mut parser = NixCodeParser::new();
        let code = r#"{ config, pkgs, lib, ... }: { }"#;
        let result = parser.parse(code).unwrap();

        let vars = result.entities_of_kind(EntityKind::Variable);
        let module_args: Vec<_> = vars
            .iter()
            .filter(|e| e.annotations.get("kind") == Some(&"module_arg".to_string()))
            .collect();
        assert!(module_args.len() >= 2); // config, pkgs at minimum
    }

    #[test]
    fn test_file_extensions() {
        let parser = NixCodeParser::new();
        assert_eq!(parser.file_extensions(), &["nix"]);
    }
}
