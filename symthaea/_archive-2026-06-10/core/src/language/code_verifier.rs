// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Round-Trip Code Verifier
//!
//! Consciousness loop: generate → parse → re-encode → compare.
//! Verifies that generated code matches the original intent in HDC space.
//!
//! # Verification Pipeline
//!
//! ```text
//! Generated source code
//!     ↓ parse (tree-sitter)
//! ParsedCode (AST)
//!     ↓ encode (CodeHDEncoder)
//! Generated HV (16,384D)
//!     ↓ cosine_similarity(intent_hv, generated_hv)
//! VerificationResult
//! ```

use symthaea_core::hdc::ContinuousHV;

use super::code_parser::{CodeDiagnostic, DiagnosticSeverity, ParsedCode};
use crate::hdc::code_encoder::CodeHDEncoder;

/// Default similarity threshold for verification to pass.
///
/// Previously 0.1 (accepted code with ~10% semantic similarity — far too permissive).
/// Raised to 0.7 to ensure generated code meaningfully matches intent in HDC space.
const DEFAULT_VERIFICATION_THRESHOLD: f32 = 0.7;

/// Verification policy controlling how strictly generated code is judged.
///
/// Different contexts warrant different thresholds:
/// - Standard: general code generation (0.7)
/// - SafetyCritical: code with side effects, unsafe, or deployment targets (0.85)
/// - Exploratory: creative/experimental generation where novelty is valued (0.5)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VerificationPolicy {
    /// General-purpose verification (threshold: 0.7)
    Standard,
    /// Strict verification for safety-critical code (threshold: 0.85)
    SafetyCritical,
    /// Relaxed verification for exploratory/creative generation (threshold: 0.5)
    Exploratory,
    /// Custom threshold for specialized use cases
    Custom(f32),
}

impl VerificationPolicy {
    /// Get the similarity threshold for this policy
    pub fn threshold(&self) -> f32 {
        match self {
            Self::Standard => 0.7,
            Self::SafetyCritical => 0.85,
            Self::Exploratory => 0.5,
            Self::Custom(t) => *t,
        }
    }
}

impl Default for VerificationPolicy {
    fn default() -> Self {
        Self::Standard
    }
}

/// Result of verifying generated code against intent
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Whether the generated code is syntactically valid
    pub syntactically_valid: bool,
    /// Cosine similarity between intent HV and generated code's HV
    pub semantic_similarity: f32,
    /// Whether the verification passes the threshold
    pub passes_threshold: bool,
    /// Syntax errors found in the generated code
    pub syntax_errors: Vec<CodeDiagnostic>,
    /// Warnings from parsing
    pub warnings: Vec<CodeDiagnostic>,
    /// Number of entities extracted from generated code
    pub entity_count: usize,
    /// Phase 5: Pre-compilation type/borrow warnings from static analysis.
    /// These are heuristic — not guaranteed correct — but catch common errors
    /// before running `cargo check`.
    pub type_warnings: Vec<String>,
}

impl VerificationResult {
    /// Check if the generated code is good enough
    pub fn is_acceptable(&self) -> bool {
        self.syntactically_valid && self.passes_threshold
    }

    /// Get a human-readable summary
    pub fn summary(&self) -> String {
        if self.is_acceptable() {
            format!(
                "Verification PASSED: syntax OK, similarity {:.3}, {} entities",
                self.semantic_similarity, self.entity_count
            )
        } else {
            let mut reasons = Vec::new();
            if !self.syntactically_valid {
                reasons.push(format!("{} syntax errors", self.syntax_errors.len()));
            }
            if !self.passes_threshold {
                reasons.push(format!(
                    "similarity {:.3} below threshold",
                    self.semantic_similarity
                ));
            }
            format!("Verification FAILED: {}", reasons.join(", "))
        }
    }
}

/// Round-trip code verifier
pub struct CodeVerifier {
    encoder: CodeHDEncoder,
    /// Minimum similarity for verification to pass
    threshold: f32,
    /// Active verification policy
    policy: VerificationPolicy,
}

impl CodeVerifier {
    /// Create a new verifier with the given encoder (uses Standard policy, threshold 0.7)
    pub fn new(encoder: CodeHDEncoder) -> Self {
        Self {
            encoder,
            threshold: DEFAULT_VERIFICATION_THRESHOLD,
            policy: VerificationPolicy::Standard,
        }
    }

    /// Create with a specific verification policy
    pub fn with_policy(encoder: CodeHDEncoder, policy: VerificationPolicy) -> Self {
        Self {
            encoder,
            threshold: policy.threshold(),
            policy,
        }
    }

    /// Create with a custom threshold (legacy API — prefer `with_policy`)
    pub fn with_threshold(encoder: CodeHDEncoder, threshold: f32) -> Self {
        Self {
            encoder,
            threshold,
            policy: VerificationPolicy::Custom(threshold),
        }
    }

    /// Get the current threshold
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Get the active verification policy
    pub fn policy(&self) -> VerificationPolicy {
        self.policy
    }

    /// Set a new threshold
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
        self.policy = VerificationPolicy::Custom(threshold);
    }

    /// Set a new verification policy
    pub fn set_policy(&mut self, policy: VerificationPolicy) {
        self.policy = policy;
        self.threshold = policy.threshold();
    }

    /// Verify that generated code matches intent in HDC space
    ///
    /// This is the core round-trip verification:
    /// 1. Parse the generated source code
    /// 2. Re-encode to HDC
    /// 3. Compare with original intent HV
    pub fn verify_against_intent(
        &self,
        parsed: &ParsedCode,
        intent_hv: &ContinuousHV,
    ) -> VerificationResult {
        // Check syntax
        let syntax_errors: Vec<CodeDiagnostic> = parsed
            .diagnostics
            .iter()
            .filter(|d| d.severity == DiagnosticSeverity::Error)
            .cloned()
            .collect();

        let warnings: Vec<CodeDiagnostic> = parsed
            .diagnostics
            .iter()
            .filter(|d| d.severity == DiagnosticSeverity::Warning)
            .cloned()
            .collect();

        let syntactically_valid = syntax_errors.is_empty();

        // Re-encode to HDC
        let generated_hv = self.encoder.encode_module(parsed);

        // Compute similarity
        let semantic_similarity = intent_hv.similarity(&generated_hv);

        // Entity count as a quality signal
        let entity_count = parsed.all_entities().len();

        VerificationResult {
            syntactically_valid,
            semantic_similarity,
            passes_threshold: semantic_similarity >= self.threshold,
            syntax_errors,
            warnings,
            entity_count,
            type_warnings: Vec::new(), // Populated by caller via validate_types()
        }
    }

    /// Verify that two code snapshots are semantically equivalent
    /// (useful for checking refactoring correctness)
    pub fn verify_equivalence(
        &self,
        original: &ParsedCode,
        transformed: &ParsedCode,
    ) -> EquivalenceResult {
        let original_hv = self.encoder.encode_module(original);
        let transformed_hv = self.encoder.encode_module(transformed);

        let similarity = original_hv.similarity(&transformed_hv);

        // Check structural properties
        let original_entities = original.all_entities();
        let transformed_entities = transformed.all_entities();

        let entity_count_diff =
            (original_entities.len() as i32 - transformed_entities.len() as i32).abs();

        EquivalenceResult {
            semantic_similarity: similarity,
            likely_equivalent: similarity > 0.8,
            entity_count_original: original_entities.len(),
            entity_count_transformed: transformed_entities.len(),
            entity_count_diff: entity_count_diff as usize,
        }
    }

    /// Quick syntax-only verification (no HDC comparison)
    pub fn verify_syntax(&self, parsed: &ParsedCode) -> bool {
        parsed
            .diagnostics
            .iter()
            .all(|d| d.severity != DiagnosticSeverity::Error)
    }
}

/// Result of checking semantic equivalence between two code versions
#[derive(Debug, Clone)]
pub struct EquivalenceResult {
    /// Cosine similarity between original and transformed code
    pub semantic_similarity: f32,
    /// Whether the codes are likely semantically equivalent
    pub likely_equivalent: bool,
    /// Number of entities in the original
    pub entity_count_original: usize,
    /// Number of entities in the transformed version
    pub entity_count_transformed: usize,
    /// Absolute difference in entity counts
    pub entity_count_diff: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language::code_parser::{Entity, EntityKind, Span};

    fn test_span() -> Span {
        Span {
            start_byte: 0,
            end_byte: 10,
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 10,
        }
    }

    #[test]
    fn test_verify_valid_code() {
        let encoder = CodeHDEncoder::new(512);
        let verifier = CodeVerifier::new(CodeHDEncoder::new(512));

        // Create a "generated" parsed code
        let mut parsed = ParsedCode::new("fn sort() {}", "rust");
        parsed
            .entities
            .push(Entity::new(EntityKind::Function, "sort", test_span()));

        // Create an intent HV by encoding "sort"
        let intent_hv = encoder.encode_name("sort");

        let result = verifier.verify_against_intent(&parsed, &intent_hv);
        assert!(result.syntactically_valid);
        assert_eq!(result.entity_count, 1);
        assert!(result.syntax_errors.is_empty());
    }

    #[test]
    fn test_verify_with_errors() {
        let encoder = CodeHDEncoder::new(512);
        let verifier = CodeVerifier::new(CodeHDEncoder::new(512));

        let mut parsed = ParsedCode::new("fn sort() {", "rust");
        parsed.diagnostics.push(CodeDiagnostic {
            severity: DiagnosticSeverity::Error,
            message: "Expected }".to_string(),
            span: None,
        });

        let intent_hv = encoder.encode_name("sort");
        let result = verifier.verify_against_intent(&parsed, &intent_hv);
        assert!(!result.syntactically_valid);
        assert!(!result.is_acceptable());
    }

    #[test]
    fn test_verify_equivalence() {
        let verifier = CodeVerifier::new(CodeHDEncoder::new(512));

        // Original
        let mut original = ParsedCode::new("fn a() {} fn b() {}", "rust");
        original
            .entities
            .push(Entity::new(EntityKind::Function, "a", test_span()));
        original
            .entities
            .push(Entity::new(EntityKind::Function, "b", test_span()));

        // Same code, same structure
        let mut transformed = ParsedCode::new("fn a() {} fn b() {}", "rust");
        transformed
            .entities
            .push(Entity::new(EntityKind::Function, "a", test_span()));
        transformed
            .entities
            .push(Entity::new(EntityKind::Function, "b", test_span()));

        let result = verifier.verify_equivalence(&original, &transformed);
        assert!(result.semantic_similarity > 0.9);
        assert!(result.likely_equivalent);
        assert_eq!(result.entity_count_diff, 0);
    }

    #[test]
    fn test_verify_different_code() {
        let verifier = CodeVerifier::new(CodeHDEncoder::new(512));

        let mut original = ParsedCode::new("fn sort() {}", "rust");
        original
            .entities
            .push(Entity::new(EntityKind::Function, "sort", test_span()));

        let mut transformed = ParsedCode::new("struct Config {}", "rust");
        transformed
            .entities
            .push(Entity::new(EntityKind::Struct, "Config", test_span()));

        let result = verifier.verify_equivalence(&original, &transformed);
        assert!(result.semantic_similarity < 0.9);
    }

    #[test]
    fn test_custom_threshold() {
        let mut verifier = CodeVerifier::new(CodeHDEncoder::new(512));
        assert!((verifier.threshold() - DEFAULT_VERIFICATION_THRESHOLD).abs() < 1e-6);

        verifier.set_threshold(0.5);
        assert!((verifier.threshold() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_default_threshold_is_strict() {
        // Verify we're using 0.7, not the old 0.1
        assert!((DEFAULT_VERIFICATION_THRESHOLD - 0.7).abs() < 1e-6);
        let verifier = CodeVerifier::new(CodeHDEncoder::new(512));
        assert!((verifier.threshold() - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_verification_policy_thresholds() {
        assert!((VerificationPolicy::Standard.threshold() - 0.7).abs() < 1e-6);
        assert!((VerificationPolicy::SafetyCritical.threshold() - 0.85).abs() < 1e-6);
        assert!((VerificationPolicy::Exploratory.threshold() - 0.5).abs() < 1e-6);
        assert!((VerificationPolicy::Custom(0.3).threshold() - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_verifier_with_policy() {
        let verifier =
            CodeVerifier::with_policy(CodeHDEncoder::new(512), VerificationPolicy::SafetyCritical);
        assert!((verifier.threshold() - 0.85).abs() < 1e-6);
        assert_eq!(verifier.policy(), VerificationPolicy::SafetyCritical);
    }

    #[test]
    fn test_set_policy_updates_threshold() {
        let mut verifier = CodeVerifier::new(CodeHDEncoder::new(512));
        assert!((verifier.threshold() - 0.7).abs() < 1e-6);

        verifier.set_policy(VerificationPolicy::Exploratory);
        assert!((verifier.threshold() - 0.5).abs() < 1e-6);
        assert_eq!(verifier.policy(), VerificationPolicy::Exploratory);
    }

    #[test]
    fn test_verification_summary() {
        let result = VerificationResult {
            syntactically_valid: true,
            semantic_similarity: 0.85,
            passes_threshold: true,
            syntax_errors: Vec::new(),
            warnings: Vec::new(),
            entity_count: 5,
            type_warnings: Vec::new(),
        };

        let summary = result.summary();
        assert!(summary.contains("PASSED"));
        assert!(summary.contains("0.850"));
    }
}
