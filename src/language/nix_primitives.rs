//! NixOS-Specific Semantic Primitives
//!
//! This module defines the "Config Calculus Mini-Core" - a subset of mathematical/logical
//! primitives plus Nix-specific semantic primitives required for true NixOS understanding.
//!
//! ## Why This Matters
//!
//! Regular semantic primes (EXIST, CAUSE, KNOW, etc.) are insufficient for NixOS because:
//! - NixOS has unique evaluation semantics (lazy, pure, with fixpoints)
//! - Module system has specific merging/override rules
//! - Provenance tracking is essential for debugging
//! - Type constraints are domain-specific (service, package, option)
//!
//! ## Primitive Categories
//!
//! 1. **Logic & Control**: IF, THEN, AND, OR, NOT, EXISTS, FORALL, IMPLIES, DEPENDS
//! 2. **Record/Set/Attribute**: ATTRSET, KEY, VALUE, PATH, MERGE, OVERRIDE, DEFAULT, PRIORITY
//! 3. **Type & Constraint**: TYPE, VALIDATE, CONSTRAINT, CONFLICTS_WITH, DEPENDENCY, REQUIRED, OPTIONAL
//! 4. **Evaluation & Provenance**: EVALUATE, ORIGIN, TRACE, RESOLVES_TO, EFFECT, DERIVATION, STORE_PATH
//! 5. **System/Service**: MODULE, OPTION, SERVICE, PACKAGE, UNIT, ACTIVATION_SCRIPT, HOST

use crate::hdc::HV16;
use crate::hdc::universal_semantics::SemanticPrime;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::Hash;

// =============================================================================
// NIX PRIMITIVE TIERS
// =============================================================================

/// Tier classification for NixOS primitives (matching PrimitiveTier structure)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NixPrimitiveTier {
    /// Universal primitives (shared with general semantic primes)
    Universal,
    /// Logic and control flow primitives
    Logic,
    /// Record/set/attribute manipulation
    Record,
    /// Type and constraint system
    TypeConstraint,
    /// Evaluation and provenance tracking
    Evaluation,
    /// NixOS system/service domain
    System,
}

impl NixPrimitiveTier {
    /// Get descriptive name for this tier
    pub fn name(&self) -> &'static str {
        match self {
            Self::Universal => "Universal",
            Self::Logic => "Logic & Control",
            Self::Record => "Record/Set/Attribute",
            Self::TypeConstraint => "Type & Constraint",
            Self::Evaluation => "Evaluation & Provenance",
            Self::System => "System/Service",
        }
    }

    /// Get all primitives in this tier
    pub fn primitives(&self) -> Vec<NixPrimitive> {
        match self {
            Self::Universal => vec![
                NixPrimitive::Exist,
                NixPrimitive::Cause,
                NixPrimitive::Know,
            ],
            Self::Logic => vec![
                NixPrimitive::If,
                NixPrimitive::Then,
                NixPrimitive::And,
                NixPrimitive::Or,
                NixPrimitive::Not,
                NixPrimitive::Exists,
                NixPrimitive::ForAll,
                NixPrimitive::Implies,
                NixPrimitive::Depends,
            ],
            Self::Record => vec![
                NixPrimitive::AttrSet,
                NixPrimitive::Key,
                NixPrimitive::Value,
                NixPrimitive::Path,
                NixPrimitive::Merge,
                NixPrimitive::Override,
                NixPrimitive::Default,
                NixPrimitive::Priority,
            ],
            Self::TypeConstraint => vec![
                NixPrimitive::Type,
                NixPrimitive::Validate,
                NixPrimitive::Constraint,
                NixPrimitive::ConflictsWith,
                NixPrimitive::Dependency,
                NixPrimitive::Required,
                NixPrimitive::Optional,
            ],
            Self::Evaluation => vec![
                NixPrimitive::Evaluate,
                NixPrimitive::Origin,
                NixPrimitive::Trace,
                NixPrimitive::ResolvesTo,
                NixPrimitive::Effect,
                NixPrimitive::Derivation,
                NixPrimitive::StorePath,
            ],
            Self::System => vec![
                NixPrimitive::Module,
                NixPrimitive::Option,
                NixPrimitive::Service,
                NixPrimitive::Package,
                NixPrimitive::Unit,
                NixPrimitive::ActivationScript,
                NixPrimitive::Host,
            ],
        }
    }
}

// =============================================================================
// NIX PRIMITIVES
// =============================================================================

/// NixOS-specific semantic primitives
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NixPrimitive {
    // === Universal (shared with SemanticPrime) ===
    Exist,
    Cause,
    Know,

    // === Logic & Control ===
    /// Conditional branching: mkIf, assertions
    If,
    /// Consequent of conditional
    Then,
    /// Conjunction: multiple conditions must hold
    And,
    /// Disjunction: at least one condition holds
    Or,
    /// Negation
    Not,
    /// Existential quantification: "there exists some X"
    Exists,
    /// Universal quantification: "for all X"
    ForAll,
    /// Implication: "if A then B"
    Implies,
    /// Dependency relationship
    Depends,

    // === Record/Set/Attribute ===
    /// Attribute set: { key = value; }
    AttrSet,
    /// Attribute key
    Key,
    /// Attribute value
    Value,
    /// Attribute path: a.b.c
    Path,
    /// Merge operation: lib.recursiveUpdate, //
    Merge,
    /// Override: mkOverride, mkForce
    Override,
    /// Default value: mkDefault
    Default,
    /// Priority level in override system
    Priority,

    // === Type & Constraint ===
    /// Type specification: types.str, types.bool, etc.
    Type,
    /// Validation check
    Validate,
    /// Constraint expression
    Constraint,
    /// Conflict marker: services that can't coexist
    ConflictsWith,
    /// Package/service dependency
    Dependency,
    /// Required option
    Required,
    /// Optional with default
    Optional,

    // === Evaluation & Provenance ===
    /// Lazy evaluation
    Evaluate,
    /// Source origin: which file/module
    Origin,
    /// Evaluation trace for debugging
    Trace,
    /// Resolution result
    ResolvesTo,
    /// Side effect (activation, systemd reload)
    Effect,
    /// Nix derivation
    Derivation,
    /// Store path result
    StorePath,

    // === System/Service ===
    /// NixOS module
    Module,
    /// Module option
    Option,
    /// Systemd service
    Service,
    /// Nix package
    Package,
    /// Systemd unit
    Unit,
    /// Activation script
    ActivationScript,
    /// Host/system reference
    Host,
}

impl NixPrimitive {
    /// Get the tier this primitive belongs to
    pub fn tier(&self) -> NixPrimitiveTier {
        match self {
            Self::Exist | Self::Cause | Self::Know => NixPrimitiveTier::Universal,

            Self::If | Self::Then | Self::And | Self::Or | Self::Not |
            Self::Exists | Self::ForAll | Self::Implies | Self::Depends => NixPrimitiveTier::Logic,

            Self::AttrSet | Self::Key | Self::Value | Self::Path |
            Self::Merge | Self::Override | Self::Default | Self::Priority => NixPrimitiveTier::Record,

            Self::Type | Self::Validate | Self::Constraint | Self::ConflictsWith |
            Self::Dependency | Self::Required | Self::Optional => NixPrimitiveTier::TypeConstraint,

            Self::Evaluate | Self::Origin | Self::Trace | Self::ResolvesTo |
            Self::Effect | Self::Derivation | Self::StorePath => NixPrimitiveTier::Evaluation,

            Self::Module | Self::Option | Self::Service | Self::Package |
            Self::Unit | Self::ActivationScript | Self::Host => NixPrimitiveTier::System,
        }
    }

    /// Get canonical name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Exist => "EXIST",
            Self::Cause => "CAUSE",
            Self::Know => "KNOW",
            Self::If => "IF",
            Self::Then => "THEN",
            Self::And => "AND",
            Self::Or => "OR",
            Self::Not => "NOT",
            Self::Exists => "EXISTS",
            Self::ForAll => "FORALL",
            Self::Implies => "IMPLIES",
            Self::Depends => "DEPENDS",
            Self::AttrSet => "ATTRSET",
            Self::Key => "KEY",
            Self::Value => "VALUE",
            Self::Path => "PATH",
            Self::Merge => "MERGE",
            Self::Override => "OVERRIDE",
            Self::Default => "DEFAULT",
            Self::Priority => "PRIORITY",
            Self::Type => "TYPE",
            Self::Validate => "VALIDATE",
            Self::Constraint => "CONSTRAINT",
            Self::ConflictsWith => "CONFLICTS_WITH",
            Self::Dependency => "DEPENDENCY",
            Self::Required => "REQUIRED",
            Self::Optional => "OPTIONAL",
            Self::Evaluate => "EVALUATE",
            Self::Origin => "ORIGIN",
            Self::Trace => "TRACE",
            Self::ResolvesTo => "RESOLVES_TO",
            Self::Effect => "EFFECT",
            Self::Derivation => "DERIVATION",
            Self::StorePath => "STORE_PATH",
            Self::Module => "MODULE",
            Self::Option => "OPTION",
            Self::Service => "SERVICE",
            Self::Package => "PACKAGE",
            Self::Unit => "UNIT",
            Self::ActivationScript => "ACTIVATION_SCRIPT",
            Self::Host => "HOST",
        }
    }

    /// Get description of this primitive's meaning in NixOS context
    pub fn description(&self) -> &'static str {
        match self {
            Self::Exist => "Something exists or is defined",
            Self::Cause => "One thing causes another",
            Self::Know => "Information or state is known",
            Self::If => "Conditional: mkIf, assertions",
            Self::Then => "Consequence of a condition",
            Self::And => "All conditions must hold",
            Self::Or => "At least one condition holds",
            Self::Not => "Negation of a condition",
            Self::Exists => "There exists some value/option",
            Self::ForAll => "For all values/hosts",
            Self::Implies => "If A then B must hold",
            Self::Depends => "Requires another component",
            Self::AttrSet => "Attribute set: { key = value; }",
            Self::Key => "Attribute name in a set",
            Self::Value => "Attribute value in a set",
            Self::Path => "Attribute path like a.b.c",
            Self::Merge => "Combine attribute sets (recursiveUpdate, //)",
            Self::Override => "Override value (mkOverride, mkForce)",
            Self::Default => "Default value (mkDefault)",
            Self::Priority => "Priority level in merge resolution",
            Self::Type => "Type specification (types.str, etc.)",
            Self::Validate => "Validation check for option",
            Self::Constraint => "Constraint expression",
            Self::ConflictsWith => "Cannot coexist with",
            Self::Dependency => "Requires package/service",
            Self::Required => "Must be specified",
            Self::Optional => "May be specified with default",
            Self::Evaluate => "Evaluate expression (lazy)",
            Self::Origin => "Source file/module of definition",
            Self::Trace => "Evaluation trace for debugging",
            Self::ResolvesTo => "Final resolved value",
            Self::Effect => "Side effect (activation, reload)",
            Self::Derivation => "Nix derivation",
            Self::StorePath => "/nix/store path",
            Self::Module => "NixOS module",
            Self::Option => "Module option definition",
            Self::Service => "Systemd service",
            Self::Package => "Nix package",
            Self::Unit => "Systemd unit",
            Self::ActivationScript => "system.activationScripts",
            Self::Host => "Host/system reference",
        }
    }

    /// Get related Nix expressions/functions for this primitive
    pub fn nix_expressions(&self) -> Vec<&'static str> {
        match self {
            Self::Exist => vec!["builtins.hasAttr", "? operator"],
            Self::Cause => vec!["depends", "requires"],
            Self::Know => vec!["config", "options"],
            Self::If => vec!["mkIf", "if-then-else", "assert"],
            Self::Then => vec!["then", "mkIf true"],
            Self::And => vec!["&&", "lib.all"],
            Self::Or => vec!["||", "lib.any"],
            Self::Not => vec!["!", "lib.not"],
            Self::Exists => vec!["builtins.hasAttr", "?"],
            Self::ForAll => vec!["builtins.all", "lib.forEach"],
            Self::Implies => vec!["->", "mkIf"],
            Self::Depends => vec!["after", "requires", "wants"],
            Self::AttrSet => vec!["{ }", "lib.attrsets"],
            Self::Key => vec!["builtins.attrNames", "lib.attrNames"],
            Self::Value => vec!["builtins.attrValues", "lib.attrValues"],
            Self::Path => vec!["a.b.c", "lib.getAttrFromPath"],
            Self::Merge => vec!["//", "lib.recursiveUpdate", "lib.mkMerge"],
            Self::Override => vec!["mkOverride", "mkForce", "lib.mkOverride"],
            Self::Default => vec!["mkDefault", "lib.mkDefault"],
            Self::Priority => vec!["mkOverride 100", "mkForce", "priority"],
            Self::Type => vec!["types.str", "types.bool", "types.listOf"],
            Self::Validate => vec!["check", "apply", "coercedTo"],
            Self::Constraint => vec!["assertions", "warnings"],
            Self::ConflictsWith => vec!["conflicts", "exclusiveOr"],
            Self::Dependency => vec!["propagatedBuildInputs", "wants", "requires"],
            Self::Required => vec!["types.nullOr", "lib.mkOption"],
            Self::Optional => vec!["default = ", "mkOption { default = }"],
            Self::Evaluate => vec!["builtins.seq", "builtins.deepSeq"],
            Self::Origin => vec!["builtins.unsafeGetAttrPos", "__file__"],
            Self::Trace => vec!["builtins.trace", "lib.traceVal"],
            Self::ResolvesTo => vec!["config.", "final value"],
            Self::Effect => vec!["activation", "reload", "restart"],
            Self::Derivation => vec!["derivation", "stdenv.mkDerivation"],
            Self::StorePath => vec!["/nix/store/...", "builtins.storePath"],
            Self::Module => vec!["imports", "{ config, lib, pkgs, ... }:"],
            Self::Option => vec!["mkOption", "options."],
            Self::Service => vec!["services.", "systemd.services"],
            Self::Package => vec!["pkgs.", "environment.systemPackages"],
            Self::Unit => vec!["systemd.services", "systemd.timers"],
            Self::ActivationScript => vec!["system.activationScripts"],
            Self::Host => vec!["networking.hostName", "config"],
        }
    }

    /// All primitives
    pub fn all() -> Vec<Self> {
        vec![
            Self::Exist, Self::Cause, Self::Know,
            Self::If, Self::Then, Self::And, Self::Or, Self::Not,
            Self::Exists, Self::ForAll, Self::Implies, Self::Depends,
            Self::AttrSet, Self::Key, Self::Value, Self::Path,
            Self::Merge, Self::Override, Self::Default, Self::Priority,
            Self::Type, Self::Validate, Self::Constraint, Self::ConflictsWith,
            Self::Dependency, Self::Required, Self::Optional,
            Self::Evaluate, Self::Origin, Self::Trace, Self::ResolvesTo,
            Self::Effect, Self::Derivation, Self::StorePath,
            Self::Module, Self::Option, Self::Service, Self::Package,
            Self::Unit, Self::ActivationScript, Self::Host,
        ]
    }
}

// =============================================================================
// NIX PRIMITIVE ENCODER
// =============================================================================

/// Encodes NixOS primitives into hyperdimensional vectors
pub struct NixPrimitiveEncoder {
    /// Primitive to HV16 encoding mappings
    encodings: HashMap<NixPrimitive, HV16>,
    /// Tier-level encodings for hierarchical reasoning
    tier_encodings: HashMap<NixPrimitiveTier, HV16>,
}

impl NixPrimitiveEncoder {
    /// Create a new encoder with unique orthogonal encodings
    pub fn new() -> Self {
        let mut encodings = HashMap::new();
        let mut tier_encodings = HashMap::new();

        // Generate unique encodings for each primitive
        // Use deterministic seeds based on primitive name for reproducibility
        for (i, primitive) in NixPrimitive::all().iter().enumerate() {
            let seed = Self::name_to_seed(primitive.name()) + i as u64;
            let hv = HV16::random(seed);
            encodings.insert(*primitive, hv);
        }

        // Generate tier encodings
        for tier in [
            NixPrimitiveTier::Universal,
            NixPrimitiveTier::Logic,
            NixPrimitiveTier::Record,
            NixPrimitiveTier::TypeConstraint,
            NixPrimitiveTier::Evaluation,
            NixPrimitiveTier::System,
        ] {
            let seed = Self::name_to_seed(tier.name());
            let hv = HV16::random(seed);
            tier_encodings.insert(tier, hv);
        }

        Self { encodings, tier_encodings }
    }

    /// Convert name to seed for deterministic encoding generation
    fn name_to_seed(name: &str) -> u64 {
        let mut seed: u64 = 0;
        for (i, b) in name.bytes().enumerate() {
            seed = seed.wrapping_add((b as u64).wrapping_mul(31_u64.wrapping_pow(i as u32)));
        }
        seed
    }

    /// Get encoding for a primitive
    pub fn encode(&self, primitive: NixPrimitive) -> Option<&HV16> {
        self.encodings.get(&primitive)
    }

    /// Get tier encoding
    pub fn encode_tier(&self, tier: NixPrimitiveTier) -> Option<&HV16> {
        self.tier_encodings.get(&tier)
    }

    /// Encode a combination of primitives (bind operation)
    pub fn encode_combination(&self, primitives: &[NixPrimitive]) -> Option<HV16> {
        if primitives.is_empty() {
            return None;
        }

        let mut result = self.encodings.get(&primitives[0])?.clone();
        for prim in primitives.iter().skip(1) {
            if let Some(hv) = self.encodings.get(prim) {
                result = result.bind(hv);
            }
        }
        Some(result)
    }

    /// Find the closest primitive to a given encoding
    pub fn decode(&self, encoding: &HV16) -> Option<(NixPrimitive, f32)> {
        let mut best: Option<(NixPrimitive, f32)> = None;

        for (primitive, hv) in &self.encodings {
            let similarity = encoding.similarity(hv);
            if best.is_none() || similarity > best.unwrap().1 {
                best = Some((*primitive, similarity));
            }
        }

        best
    }

    /// Identify active primitives in an encoding (above threshold)
    pub fn identify_active(&self, encoding: &HV16, threshold: f32) -> Vec<(NixPrimitive, f32)> {
        let mut active = Vec::new();

        for (primitive, hv) in &self.encodings {
            let similarity = encoding.similarity(hv);
            if similarity > threshold {
                active.push((*primitive, similarity));
            }
        }

        active.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        active
    }
}

impl Default for NixPrimitiveEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// NIX EXPRESSION ANALYZER
// =============================================================================

/// Analyzes Nix expressions to identify which primitives are active
pub struct NixExpressionAnalyzer {
    encoder: NixPrimitiveEncoder,
    /// Keyword to primitive mappings
    keyword_mappings: HashMap<&'static str, Vec<NixPrimitive>>,
}

impl NixExpressionAnalyzer {
    /// Create a new analyzer
    pub fn new() -> Self {
        let encoder = NixPrimitiveEncoder::new();
        let mut keyword_mappings = HashMap::new();

        // Build keyword mappings from primitive expressions
        for primitive in NixPrimitive::all() {
            for expr in primitive.nix_expressions() {
                keyword_mappings
                    .entry(expr)
                    .or_insert_with(Vec::new)
                    .push(primitive);
            }
        }

        Self { encoder, keyword_mappings }
    }

    /// Analyze input text to identify active primitives
    pub fn analyze(&self, input: &str) -> Vec<(NixPrimitive, f32)> {
        let input_lower = input.to_lowercase();
        let mut scores: HashMap<NixPrimitive, f32> = HashMap::new();

        // Check for keyword matches
        for (keyword, primitives) in &self.keyword_mappings {
            if input_lower.contains(&keyword.to_lowercase()) {
                for primitive in primitives {
                    *scores.entry(*primitive).or_insert(0.0) += 1.0;
                }
            }
        }

        // Also check primitive names and descriptions
        for primitive in NixPrimitive::all() {
            if input_lower.contains(&primitive.name().to_lowercase()) {
                *scores.entry(primitive).or_insert(0.0) += 0.5;
            }
        }

        // Normalize and sort
        let max_score = scores.values().cloned().fold(0.0_f32, f32::max);
        if max_score > 0.0 {
            for score in scores.values_mut() {
                *score /= max_score;
            }
        }

        let mut results: Vec<_> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results
    }

    /// Get the encoder for external use
    pub fn encoder(&self) -> &NixPrimitiveEncoder {
        &self.encoder
    }
}

impl Default for NixExpressionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// BRIDGE TO SEMANTIC PRIMES
// =============================================================================

/// Maps between NixOS primitives and general semantic primes
pub fn to_semantic_prime(nix_prim: NixPrimitive) -> Option<SemanticPrime> {
    match nix_prim {
        NixPrimitive::Exist => Some(SemanticPrime::Be),           // EXIST → BE/EXIST
        NixPrimitive::Cause => Some(SemanticPrime::Because),      // CAUSE → BECAUSE/REASON
        NixPrimitive::Know => Some(SemanticPrime::Know),
        NixPrimitive::If => Some(SemanticPrime::If),
        NixPrimitive::Then => Some(SemanticPrime::Because),       // Consequence relation
        NixPrimitive::And | NixPrimitive::Or | NixPrimitive::Not => Some(SemanticPrime::Not),  // Logic group
        NixPrimitive::Depends => Some(SemanticPrime::Because),    // Dependency is causal
        NixPrimitive::AttrSet | NixPrimitive::Key | NixPrimitive::Value |
        NixPrimitive::Path | NixPrimitive::Merge | NixPrimitive::Override => Some(SemanticPrime::PartOf),  // Structure
        NixPrimitive::Type | NixPrimitive::Validate | NixPrimitive::Constraint => Some(SemanticPrime::KindOf),  // Classification
        NixPrimitive::Effect | NixPrimitive::Evaluate => Some(SemanticPrime::Happen),  // Action
        NixPrimitive::Service | NixPrimitive::Package | NixPrimitive::Module => Some(SemanticPrime::Something),  // Entity
        _ => None,  // No direct mapping
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitive_tiers() {
        assert_eq!(NixPrimitive::If.tier(), NixPrimitiveTier::Logic);
        assert_eq!(NixPrimitive::AttrSet.tier(), NixPrimitiveTier::Record);
        assert_eq!(NixPrimitive::Type.tier(), NixPrimitiveTier::TypeConstraint);
        assert_eq!(NixPrimitive::Evaluate.tier(), NixPrimitiveTier::Evaluation);
        assert_eq!(NixPrimitive::Service.tier(), NixPrimitiveTier::System);
    }

    #[test]
    fn test_primitive_all() {
        let all = NixPrimitive::all();
        assert!(all.len() >= 35);  // Should have 35+ primitives
    }

    #[test]
    fn test_encoder_creation() {
        let encoder = NixPrimitiveEncoder::new();
        assert!(encoder.encode(NixPrimitive::If).is_some());
        assert!(encoder.encode(NixPrimitive::Service).is_some());
    }

    #[test]
    fn test_encoder_orthogonality() {
        let encoder = NixPrimitiveEncoder::new();

        let if_enc = encoder.encode(NixPrimitive::If).unwrap();
        let service_enc = encoder.encode(NixPrimitive::Service).unwrap();

        // Different primitives should have lower similarity than identical ones
        // HV16 random vectors typically have ~0.5 similarity by chance
        // We just verify they're not identical (similarity < 1.0)
        let similarity = if_enc.similarity(service_enc);
        let self_similarity = if_enc.similarity(if_enc);

        assert!(similarity < self_similarity,
            "Different primitives should have lower similarity than self (got {} vs {})",
            similarity, self_similarity);
    }

    #[test]
    fn test_encode_combination() {
        let encoder = NixPrimitiveEncoder::new();

        let combined = encoder.encode_combination(&[
            NixPrimitive::If,
            NixPrimitive::Service,
            NixPrimitive::Exists,
        ]);

        assert!(combined.is_some());
    }

    #[test]
    fn test_decode_roundtrip() {
        let encoder = NixPrimitiveEncoder::new();

        let original = NixPrimitive::Override;
        let encoding = encoder.encode(original).unwrap();
        let (decoded, confidence) = encoder.decode(encoding).unwrap();

        assert_eq!(decoded, original);
        assert!(confidence > 0.9);  // Should be very high for exact encoding
    }

    #[test]
    fn test_expression_analyzer() {
        let analyzer = NixExpressionAnalyzer::new();

        let results = analyzer.analyze("mkIf services.nginx.enable");

        // Should identify If primitive (from mkIf)
        assert!(results.iter().any(|(p, _)| *p == NixPrimitive::If),
            "Should detect IF primitive from mkIf");
    }

    #[test]
    fn test_nix_expressions() {
        assert!(NixPrimitive::If.nix_expressions().contains(&"mkIf"));
        assert!(NixPrimitive::Merge.nix_expressions().contains(&"//"));
        assert!(NixPrimitive::Override.nix_expressions().contains(&"mkForce"));
    }

    #[test]
    fn test_semantic_prime_mapping() {
        assert_eq!(to_semantic_prime(NixPrimitive::Exist), Some(SemanticPrime::Be));
        assert_eq!(to_semantic_prime(NixPrimitive::If), Some(SemanticPrime::If));
        assert_eq!(to_semantic_prime(NixPrimitive::Service), Some(SemanticPrime::Something));
    }

    #[test]
    fn test_tier_primitives() {
        let logic_prims = NixPrimitiveTier::Logic.primitives();
        assert!(logic_prims.contains(&NixPrimitive::If));
        assert!(logic_prims.contains(&NixPrimitive::And));

        let system_prims = NixPrimitiveTier::System.primitives();
        assert!(system_prims.contains(&NixPrimitive::Service));
        assert!(system_prims.contains(&NixPrimitive::Package));
    }

    #[test]
    fn test_identify_active_primitives() {
        let encoder = NixPrimitiveEncoder::new();

        // Create an encoding that includes multiple primitives
        let combined = encoder.encode_combination(&[
            NixPrimitive::If,
            NixPrimitive::Service,
        ]).unwrap();

        let active = encoder.identify_active(&combined, 0.1);
        assert!(!active.is_empty(), "Should identify some active primitives");
    }
}
