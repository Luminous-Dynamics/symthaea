// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::PrimitiveTier;
use crate::hdc::binary_hv::BinaryHV;
use serde::{Deserialize, Serialize};

/// A primitive concept with its BinaryHV encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Primitive {
    /// Name of the primitive
    pub name: String,

    /// Tier in hierarchy
    pub tier: PrimitiveTier,

    /// Domain this primitive belongs to
    pub domain: String,

    /// BinaryHV encoding (embedded in domain manifold)
    pub encoding: BinaryHV,

    /// Mathematical/logical definition
    pub definition: String,

    /// Whether this is a base primitive or derived
    pub is_base: bool,

    /// If derived, the formula for deriving it
    pub derivation: Option<String>,
}

impl Primitive {
    /// Create a base primitive
    pub fn base(
        name: impl Into<String>,
        tier: PrimitiveTier,
        domain: impl Into<String>,
        encoding: BinaryHV,
        definition: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            tier,
            domain: domain.into(),
            encoding,
            definition: definition.into(),
            is_base: true,
            derivation: None,
        }
    }

    /// Create a derived primitive
    pub fn derived(
        name: impl Into<String>,
        tier: PrimitiveTier,
        domain: impl Into<String>,
        encoding: BinaryHV,
        definition: impl Into<String>,
        derivation: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            tier,
            domain: domain.into(),
            encoding,
            definition: definition.into(),
            is_base: false,
            derivation: Some(derivation.into()),
        }
    }
}

/// Binding grammar - rules for valid primitive combinations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingRule {
    /// Name of this rule
    pub name: String,

    /// Pattern: which primitive types can bind
    pub pattern: Vec<PrimitiveTier>,

    /// Result tier
    pub result_tier: PrimitiveTier,

    /// Example application
    pub example: String,
}

/// Result of a typed primitive operation
#[derive(Debug, Clone)]
pub struct PrimitiveResult {
    /// The resulting BinaryHV encoding
    pub encoding: BinaryHV,
    /// Description of the operation
    pub operation: String,
    /// Source primitives used
    pub source_primitives: Vec<String>,
}

impl PrimitiveResult {
    /// Find similar primitives to this result
    pub fn find_similar(
        &self,
        system: &super::PrimitiveSystem,
        top_k: usize,
    ) -> Vec<(String, f32)> {
        system.find_similar_to_encoding(&self.encoding, top_k)
    }
}

/// Errors from typed primitive operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrimitiveError {
    /// Primitive not found
    NotFound(String),
    /// Empty input
    EmptyInput,
    /// Invalid weight (zero or negative total)
    InvalidWeight,
}

impl std::fmt::Display for PrimitiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrimitiveError::NotFound(name) => write!(f, "primitive not found: {name}"),
            PrimitiveError::EmptyInput => write!(f, "operation requires at least one input"),
            PrimitiveError::InvalidWeight => write!(f, "weights must sum to positive value"),
        }
    }
}

impl std::error::Error for PrimitiveError {}
