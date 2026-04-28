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

// Assuming Entity, Span, CodeStructure, CodeDiagnostic, and EntityKind are available
// from the common library module (code_parser.rs)
use crate::code_parser::{Entity, Span, CodeStructure, CodeDiagnostic, EntityKind, Relation, EntityRelation};


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

/// The result of parsing configuration data
#[derive(Debug, Clone)]
pub struct ConfiguredCode {
    /// Raw source text
    pub source: String,
    /// Language/Format name
    pub format: String,
    /// Extracted entities
    pub entities: Vec<Entity>,
    /// Structural relationships (e.g., which parameter depends on which definition)
    pub structure: CodeStructure, // Now uses the unified CodeStructure
    /// Diagnostics (errors, warnings)
    pub diagnostics: Vec<CodeDiagnostic>,
}
