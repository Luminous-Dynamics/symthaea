// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unified Configuration Parser Trait
//!
//! Provides a language-agnostic interface for parsing structured configuration
//! data (e.g., YAML, JSON, custom DSLs) into a structured AST representation.
//! This complements `code_parser.rs` by handling non-code inputs.
//!
//! # Architecture
//!
//! ```text
//! Config Source → ConfigParser::parse() → ConfiguredCode (AST + entities)
//!                                         ↓
//!                           ConfigParser::to_hdv() → ContinuousHV (16,384D)
//! ```

use std::collections::HashMap;
use std::fmt;

/// Represents a structured configuration value.
#[derive(Debug, Clone)]
pub enum ConfigValue {
    String(String),
    Number(f64),
    Boolean(bool),
    List(Vec<ConfigValue>),
    Map(HashMap<String, ConfigValue>),
    // Placeholder for complex types like references or enums
    Custom(String),
}

impl fmt::Display for ConfigValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigValue::String(s) => write!(f, "\"{}\"", s),
            ConfigValue::Number(n) => write!(f, "{}", n),
            ConfigValue::Boolean(b) => write!(f, "{}", b),
            ConfigValue::List(v) => write!(f, "[...list of {} items...]", v.len()),
            ConfigValue::Map(m) => write!(f, "{{...map with {} keys...}}", m.len()),
            ConfigValue::Custom(s) => write!(f, "{}", s),
        }
    }
}

/// A structured entity extracted from configuration data.
#[derive(Debug, Clone)]
pub struct ConfigEntity {
    /// What kind of configuration entity
    pub kind: ConfigEntityKind,
    /// Name of the entity (e.g., 'simulation_params', 'initial_state')
    pub name: String,
    /// Location in source
    pub span: Span,
    /// Raw source text of this entity
    pub source_text: String,
    /// Value associated with this entity (e.g., the map or list)
    pub value: ConfigValue,
    /// Child entities (e.g., nested map entries)
    pub children: Vec<ConfigEntity>,
    /// Arbitrary annotations (e.g., "required" → "true", "default" → "1.0")
    pub annotations: HashMap<String, String>,
}

impl ConfigEntity {
    /// Create a new configuration entity
    pub fn new(kind: ConfigEntityKind, name: impl Into<String>, span: Span, value: ConfigValue) -> Self {
        Self {
            kind,
            name: name.into(),
            source_text: String::new(),
            span,
            value,
            children: Vec::new(),
            annotations: HashMap::new(),
        }
    }

    /// Set source text
    pub fn with_source(mut self, text: impl Into<String>) -> Self {
        self.source_text = text.into();
        self
    }

    /// Add an annotation
    pub fn with_annotation(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.annotations.insert(key.into(), value.into());
        self
    }

    /// Add a child entity
    pub fn with_child(mut self, child: ConfigEntity) -> Self {
        self.children.push(child);
        self
    }
}

/// What kind of configuration entity this is
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConfigEntityKind {
    // Common
    Root,
    Parameter,
    Definition,
    Reference,
    // Specific types
    SimulationInput,
    KnowledgeGraph,
    DSLStatement,
    // Fallback
    Other,
}

/// The unified configuration parser trait.
pub trait ConfigParser: Send + Sync {
    /// Name of the configuration format this parser handles (e.g., "YAML", "JSON", "DSL")
    fn format_name(&self) -> &str;

    /// Parse source configuration into a ConfiguredCode structure.
    fn parse(&mut self, source: &str) -> Result<ConfiguredCode, ConfigDiagnostic>;

    /// File extensions this parser handles
    fn file_extensions(&self) -> &[&str];
}

/// The result of parsing configuration data
#[derive(Debug, Clone)]
pub struct ConfiguredCode {
    /// Raw source text
    pub source: String,
    /// Language/Format name
    pub format: String,
    /// Extracted entities
    pub entities: Vec<ConfigEntity>,
    /// Structural relationships (e.g., which parameter depends on which definition)
    pub structure: CodeStructure, // Reusing CodeStructure for simplicity
    /// Diagnostics (errors, warnings)
    pub diagnostics: Vec<ConfigDiagnostic>,
}

// Reusing Span, CodeDiagnostic, CodeStructure, etc., from code_parser.rs
// (Assuming these structs are available or imported)
// For this stub, we assume they are available in the scope.
// In a real project, we would need to define or import them.
// Since this is a suggestion, we rely on the existing definitions.
