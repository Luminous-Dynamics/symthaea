// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Parser Registry — manages language parsers with auto-detection
//!
//! Provides a central registry for all code parsers, with automatic
//! language detection from file extensions and source content heuristics.

use std::collections::HashMap;

use super::code_parser::{CodeDiagnostic, CodeParser, DiagnosticSeverity, ParsedCode};

/// Registry of language-specific code parsers
pub struct ParserRegistry {
    parsers: HashMap<String, Box<dyn CodeParser>>,
    /// Map file extension → language name
    extension_map: HashMap<String, String>,
}

impl ParserRegistry {
    /// Create an empty registry
    pub fn new() -> Self {
        Self {
            parsers: HashMap::new(),
            extension_map: HashMap::new(),
        }
    }

    /// Create a registry with all built-in parsers (Rust, Python, Nix)
    #[cfg(feature = "code_generation")]
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();
        registry.register(Box::new(super::rust_parser::RustParser::new()));
        registry.register(Box::new(super::python_parser::PythonParser::new()));
        // Nix parser uses the CodeParser adapter
        registry.register(Box::new(super::nix_code_parser::NixCodeParser::new()));
        registry
    }

    /// Create a registry with built-in parsers (non-feature-gated version for testing)
    pub fn with_defaults() -> Self {
        Self::new()
    }

    /// Register a parser
    pub fn register(&mut self, parser: Box<dyn CodeParser>) {
        let name = parser.language_name().to_string();
        for ext in parser.file_extensions() {
            self.extension_map.insert(ext.to_string(), name.clone());
        }
        self.parsers.insert(name, parser);
    }

    /// List all registered languages
    pub fn languages(&self) -> Vec<&str> {
        self.parsers.keys().map(|s| s.as_str()).collect()
    }

    /// Detect language from source content and optional filename
    pub fn detect_language(&self, source: &str, filename: Option<&str>) -> Option<String> {
        // First: try file extension
        if let Some(fname) = filename {
            if let Some(ext) = fname.rsplit('.').next() {
                if let Some(lang) = self.extension_map.get(ext) {
                    return Some(lang.clone());
                }
            }
        }

        // Second: content-based heuristics
        let trimmed = source.trim();

        // Rust heuristics
        if trimmed.starts_with("use ")
            || trimmed.starts_with("fn ")
            || trimmed.starts_with("pub fn ")
            || trimmed.starts_with("pub struct ")
            || trimmed.starts_with("pub enum ")
            || trimmed.starts_with("mod ")
            || trimmed.starts_with("#![")
            || trimmed.contains("fn main()")
            || trimmed.contains("impl ")
        {
            if self.parsers.contains_key("rust") {
                return Some("rust".to_string());
            }
        }

        // Python heuristics
        if trimmed.starts_with("import ")
            || trimmed.starts_with("from ")
            || trimmed.starts_with("def ")
            || trimmed.starts_with("class ")
            || trimmed.starts_with("async def ")
            || trimmed.starts_with("#!/usr/bin/env python")
            || trimmed.starts_with("# -*- coding:")
        {
            if self.parsers.contains_key("python") {
                return Some("python".to_string());
            }
        }

        // Nix heuristics
        if trimmed.starts_with("{ ")
            || trimmed.starts_with("let\n")
            || trimmed.starts_with("let ")
            || trimmed.contains("mkDerivation")
            || trimmed.contains("pkgs.")
            || trimmed.contains("nixpkgs")
            || (trimmed.starts_with("{") && trimmed.contains("..."))
        {
            if self.parsers.contains_key("nix") {
                return Some("nix".to_string());
            }
        }

        None
    }

    /// Parse source code, auto-detecting language if not specified
    pub fn parse(
        &mut self,
        source: &str,
        language: Option<&str>,
        filename: Option<&str>,
    ) -> Result<ParsedCode, CodeDiagnostic> {
        let lang = match language {
            Some(l) => l.to_string(),
            None => self
                .detect_language(source, filename)
                .ok_or_else(|| CodeDiagnostic {
                    severity: DiagnosticSeverity::Error,
                    message: "Could not detect language".to_string(),
                    span: None,
                })?,
        };

        let parser = self.parsers.get_mut(&lang).ok_or_else(|| CodeDiagnostic {
            severity: DiagnosticSeverity::Error,
            message: format!("No parser registered for language '{}'", lang),
            span: None,
        })?;

        parser.parse(source)
    }
}

impl Default for ParserRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_language_by_extension() {
        let mut registry = ParserRegistry::new();
        // Register a mock parser
        registry.register(Box::new(super::super::rust_parser::RustParser::new()));
        registry.register(Box::new(super::super::python_parser::PythonParser::new()));

        assert_eq!(
            registry.detect_language("", Some("main.rs")),
            Some("rust".to_string())
        );
        assert_eq!(
            registry.detect_language("", Some("app.py")),
            Some("python".to_string())
        );
        assert_eq!(registry.detect_language("", Some("unknown.xyz")), None);
    }

    #[test]
    fn test_detect_language_by_content() {
        let mut registry = ParserRegistry::new();
        registry.register(Box::new(super::super::rust_parser::RustParser::new()));
        registry.register(Box::new(super::super::python_parser::PythonParser::new()));

        assert_eq!(
            registry.detect_language("fn main() { println!(\"hello\"); }", None),
            Some("rust".to_string())
        );
        assert_eq!(
            registry.detect_language("def hello():\n    print('hello')", None),
            Some("python".to_string())
        );
    }

    #[test]
    fn test_parse_with_auto_detection() {
        let mut registry = ParserRegistry::new();
        registry.register(Box::new(super::super::rust_parser::RustParser::new()));

        let result = registry.parse("fn hello() {}", None, Some("test.rs"));
        assert!(result.is_ok());
        assert_eq!(result.unwrap().language, "rust");
    }

    #[test]
    fn test_languages_list() {
        let mut registry = ParserRegistry::new();
        registry.register(Box::new(super::super::rust_parser::RustParser::new()));
        registry.register(Box::new(super::super::python_parser::PythonParser::new()));

        let langs = registry.languages();
        assert!(langs.contains(&"rust"));
        assert!(langs.contains(&"python"));
    }
}
