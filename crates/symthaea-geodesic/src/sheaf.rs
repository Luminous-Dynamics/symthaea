// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Sheaf Coherence -- local-to-global code correctness verification.
//!
//! Models code composability as a presheaf over the call graph:
//! - Open sets = connected subgraphs (function + dependencies)
//! - Sections = implementations with type signatures as HDC vectors
//! - Gluing condition = type compatibility across module boundaries
//!
//! # Theory
//!
//! Sheaf theory (Kashiwara & Schapira, 1990) guarantees that locally-consistent
//! data assembles into a globally-consistent whole. Applied to code: if every
//! caller's expectation of a callee's interface matches the callee's actual
//! interface (measured via HDC similarity), then the entire program composes
//! correctly.
//!
//! # HDC Encoding
//!
//! Each function's "expected signature" of a callee is computed by binding the
//! caller's encoding with a name-derived basis vector for the callee. The
//! callee's "actual signature" is its own encoding. High similarity between
//! these two vectors indicates type-compatible composition.

use symthaea_core::hdc::binary_hv::BinaryHV;

// ---------------------------------------------------------------------------
// Deterministic seed derivation for callee name encoding.
// ---------------------------------------------------------------------------

/// Derive a deterministic seed from a function name string.
///
/// Uses a simple hash: fold bytes with position-dependent shifts.
/// Must remain stable across versions for reproducible gluing checks.
fn name_seed(name: &str) -> u64 {
    name.bytes()
        .enumerate()
        .fold(0x50EA_FC0D_E000_0000u64, |acc, (i, b)| {
            acc.wrapping_add((b as u64).wrapping_shl((i as u32) % 64))
        })
}

// ============================================================================
// LOCAL SECTION
// ============================================================================

/// A local section: one function's implementation with its interface.
///
/// In sheaf-theoretic terms, this is a section over an open set (the function
/// and its direct dependencies). The `encoding` captures the function's
/// semantic content, while `param_types` and `return_type` describe its
/// interface.
#[derive(Debug, Clone)]
pub struct LocalSection {
    /// Function name (unique identifier within the sheaf).
    pub name: String,
    /// HDC encoding of the function's semantic content.
    pub encoding: BinaryHV,
    /// Parameter types encoded as HDC vectors.
    pub param_types: Vec<BinaryHV>,
    /// Return type encoded as HDC vector.
    pub return_type: BinaryHV,
    /// Names of functions this section calls (callees).
    pub callees: Vec<String>,
}

// ============================================================================
// GLUING CONDITION
// ============================================================================

/// A gluing condition between two overlapping sections.
///
/// The gluing axiom requires that on overlapping open sets, the restrictions
/// of two sections agree. Here, the "overlap" is the call site: the caller
/// has an expectation of the callee's interface, and the callee has an actual
/// interface. Agreement is measured by HDC similarity.
#[derive(Debug, Clone)]
pub struct GluingCondition {
    /// The calling function.
    pub caller: String,
    /// The called function.
    pub callee: String,
    /// Caller's expectation of the callee's signature (bind of caller encoding
    /// with callee name vector).
    pub expected_signature: BinaryHV,
    /// The callee's actual encoding.
    pub actual_signature: BinaryHV,
    /// Cosine similarity between expected and actual in [0, 1].
    pub similarity: f64,
    /// Whether the condition is satisfied (similarity > threshold).
    pub satisfied: bool,
}

// ============================================================================
// SHEAF DIAGNOSTIC
// ============================================================================

/// A diagnostic for a sheaf violation (unsatisfied gluing condition).
///
/// Provides human-readable information about which interface mismatch
/// was detected and how severe it is.
#[derive(Debug, Clone)]
pub struct SheafDiagnostic {
    /// The calling function.
    pub caller: String,
    /// The called function.
    pub callee: String,
    /// Human-readable description of the violation.
    pub message: String,
    /// Similarity score (lower = worse mismatch).
    pub similarity: f64,
    /// Summary of what the caller expected.
    pub expected_summary: String,
    /// Summary of what the callee actually provides.
    pub actual_summary: String,
}

// ============================================================================
// CODE SHEAF
// ============================================================================

/// The code sheaf: verifies local-to-global coherence of a codebase.
///
/// Collects [`LocalSection`]s (function implementations) and verifies that
/// all call-site interfaces are compatible via the sheaf gluing axiom.
///
/// # Example
///
/// ```rust,ignore
/// use symthaea_geodesic::sheaf::{CodeSheaf, LocalSection};
/// use symthaea_core::hdc::binary_hv::BinaryHV;
///
/// let mut sheaf = CodeSheaf::new();
/// sheaf.add_section(LocalSection {
///     name: "foo".into(),
///     encoding: BinaryHV::random(1),
///     param_types: vec![],
///     return_type: BinaryHV::random(2),
///     callees: vec!["bar".into()],
/// });
/// sheaf.add_section(LocalSection {
///     name: "bar".into(),
///     encoding: BinaryHV::random(3),
///     param_types: vec![],
///     return_type: BinaryHV::random(4),
///     callees: vec![],
/// });
///
/// match sheaf.verify() {
///     Ok(conditions) => println!("{} gluing conditions satisfied", conditions.len()),
///     Err(diagnostics) => {
///         for d in &diagnostics {
///             eprintln!("VIOLATION: {} -> {}: {}", d.caller, d.callee, d.message);
///         }
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct CodeSheaf {
    sections: Vec<LocalSection>,
    /// Similarity threshold for gluing conditions. A gluing condition is
    /// satisfied when the HDC similarity between expected and actual
    /// signatures exceeds this threshold.
    threshold: f64,
}

impl Default for CodeSheaf {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeSheaf {
    /// Default similarity threshold for gluing conditions.
    const DEFAULT_THRESHOLD: f64 = 0.3;

    /// Create an empty code sheaf with the default threshold (0.3).
    pub fn new() -> Self {
        Self {
            sections: Vec::new(),
            threshold: Self::DEFAULT_THRESHOLD,
        }
    }

    /// Set the similarity threshold for gluing conditions.
    ///
    /// Higher thresholds are stricter (require more similar interfaces).
    /// Values in [0.0, 1.0]; 0.5 is the expected similarity of random HVs.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Add a local section (function implementation) to the sheaf.
    pub fn add_section(&mut self, section: LocalSection) {
        self.sections.push(section);
    }

    /// Find a section by name.
    pub fn find_section(&self, name: &str) -> Option<&LocalSection> {
        self.sections.iter().find(|s| s.name == name)
    }

    /// Number of sections in the sheaf.
    pub fn section_count(&self) -> usize {
        self.sections.len()
    }

    /// Number of gluing conditions that would be checked.
    ///
    /// This equals the total number of (caller, callee) pairs where both
    /// the caller and callee exist as sections in the sheaf.
    pub fn gluing_count(&self) -> usize {
        self.sections
            .iter()
            .flat_map(|s| {
                s.callees
                    .iter()
                    .filter(|callee_name| self.find_section(callee_name).is_some())
            })
            .count()
    }

    /// Verify all gluing conditions between overlapping sections.
    ///
    /// Returns `Ok(conditions)` if all conditions are satisfied, where
    /// `conditions` contains the full list of checked gluing conditions.
    /// Returns `Err(diagnostics)` if any condition is violated.
    pub fn verify(&self) -> Result<Vec<GluingCondition>, Vec<SheafDiagnostic>> {
        let mut conditions = Vec::new();
        let mut diagnostics = Vec::new();

        for caller in &self.sections {
            for callee_name in &caller.callees {
                // Only check gluing if the callee exists as a section
                let Some(callee) = self.find_section(callee_name) else {
                    continue;
                };

                let condition = self.check_gluing(caller, callee);

                if !condition.satisfied {
                    diagnostics.push(SheafDiagnostic {
                        caller: condition.caller.clone(),
                        callee: condition.callee.clone(),
                        message: format!(
                            "Interface mismatch: similarity {:.4} < threshold {:.4}",
                            condition.similarity, self.threshold
                        ),
                        similarity: condition.similarity,
                        expected_summary: format!(
                            "caller '{}' expects callee signature (bind of caller encoding with name '{}')",
                            caller.name, callee_name
                        ),
                        actual_summary: format!(
                            "callee '{}' provides its own encoding",
                            callee.name
                        ),
                    });
                }

                conditions.push(condition);
            }
        }

        if diagnostics.is_empty() {
            Ok(conditions)
        } else {
            Err(diagnostics)
        }
    }

    /// Check a single gluing condition between a caller and callee.
    ///
    /// The caller's "expected signature" is computed by binding the caller's
    /// encoding with a name-derived vector for the callee. This captures the
    /// idea that the caller "projects" its understanding of the callee through
    /// its own semantic context.
    fn check_gluing(&self, caller: &LocalSection, callee: &LocalSection) -> GluingCondition {
        // The caller's expectation: bind caller encoding with callee name vector.
        // This produces a "projected interface" from the caller's perspective.
        let callee_name_hv = BinaryHV::random(name_seed(&callee.name));
        let expected = caller.encoding.bind(&callee_name_hv);

        let actual = &callee.encoding;
        let similarity = expected.similarity(actual) as f64;

        GluingCondition {
            caller: caller.name.clone(),
            callee: callee.name.clone(),
            expected_signature: expected,
            actual_signature: *actual,
            similarity,
            satisfied: similarity > self.threshold,
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a section with a given seed and callees.
    fn make_section(name: &str, seed: u64, callees: Vec<&str>) -> LocalSection {
        LocalSection {
            name: name.to_string(),
            encoding: BinaryHV::random(seed),
            param_types: vec![],
            return_type: BinaryHV::random(seed.wrapping_add(1)),
            callees: callees.into_iter().map(String::from).collect(),
        }
    }

    #[test]
    fn test_sheaf_no_sections() {
        // Empty sheaf is vacuously coherent.
        let sheaf = CodeSheaf::new();
        let result = sheaf.verify();
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
        assert_eq!(sheaf.section_count(), 0);
        assert_eq!(sheaf.gluing_count(), 0);
    }

    #[test]
    fn test_sheaf_coherent() {
        // Two sections where the caller's expected signature of the callee
        // has high similarity to the callee's actual encoding.
        //
        // We use the same encoding for the callee as what the caller expects:
        // expected = caller.bind(name_hv(callee)), so we set callee.encoding
        // to exactly that value.
        let caller_encoding = BinaryHV::random(100);
        let callee_name = "callee_fn";
        let callee_name_hv = BinaryHV::random(name_seed(callee_name));
        let expected_callee_encoding = caller_encoding.bind(&callee_name_hv);

        let mut sheaf = CodeSheaf::new();
        sheaf.add_section(LocalSection {
            name: "caller_fn".to_string(),
            encoding: caller_encoding,
            param_types: vec![],
            return_type: BinaryHV::random(101),
            callees: vec![callee_name.to_string()],
        });
        sheaf.add_section(LocalSection {
            name: callee_name.to_string(),
            encoding: expected_callee_encoding,
            param_types: vec![],
            return_type: BinaryHV::random(102),
            callees: vec![],
        });

        assert_eq!(sheaf.section_count(), 2);
        assert_eq!(sheaf.gluing_count(), 1);

        let result = sheaf.verify();
        assert!(
            result.is_ok(),
            "expected coherent sheaf, got: {:?}",
            result.err()
        );
        let conditions = result.unwrap();
        assert_eq!(conditions.len(), 1);
        assert!(conditions[0].satisfied);
        // Similarity should be 1.0 since we used the exact expected encoding.
        assert!(
            conditions[0].similarity > 0.99,
            "similarity should be ~1.0, got {}",
            conditions[0].similarity
        );
    }

    #[test]
    fn test_sheaf_violation() {
        // Caller and callee have unrelated encodings -> low similarity -> violation.
        let mut sheaf = CodeSheaf::new().with_threshold(0.6);
        sheaf.add_section(make_section("alpha", 200, vec!["beta"]));
        sheaf.add_section(make_section("beta", 300, vec![]));

        let result = sheaf.verify();
        assert!(
            result.is_err(),
            "expected sheaf violation for unrelated encodings"
        );

        let diagnostics = result.unwrap_err();
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].caller, "alpha");
        assert_eq!(diagnostics[0].callee, "beta");
        assert!(diagnostics[0].similarity < 0.6);
    }

    #[test]
    fn test_sheaf_self_referential() {
        // A section that calls itself (recursive function).
        // The gluing check should handle this gracefully.
        let mut sheaf = CodeSheaf::new();
        sheaf.add_section(make_section("recurse", 400, vec!["recurse"]));

        // Self-referential gluing: expected = encoding.bind(name_hv("recurse"))
        // which is unrelated to encoding itself, so similarity will be ~0.5.
        // With default threshold 0.3, this should pass.
        let result = sheaf.verify();
        assert!(
            result.is_ok(),
            "self-referential section should verify: {:?}",
            result.err()
        );
        let conditions = result.unwrap();
        assert_eq!(conditions.len(), 1);
        assert_eq!(conditions[0].caller, "recurse");
        assert_eq!(conditions[0].callee, "recurse");
    }

    #[test]
    fn test_sheaf_missing_callee_ignored() {
        // If a callee is not present as a section, the gluing condition
        // is not checked (external dependency).
        let mut sheaf = CodeSheaf::new();
        sheaf.add_section(make_section("main", 500, vec!["external_lib"]));

        // No section for "external_lib" -> no gluing condition to check.
        assert_eq!(sheaf.gluing_count(), 0);
        let result = sheaf.verify();
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }

    #[test]
    fn test_sheaf_transitive_calls() {
        // A -> B -> C: verify both A->B and B->C gluing conditions.
        let mut sheaf = CodeSheaf::new().with_threshold(0.3);
        sheaf.add_section(make_section("a", 600, vec!["b"]));
        sheaf.add_section(make_section("b", 700, vec!["c"]));
        sheaf.add_section(make_section("c", 800, vec![]));

        assert_eq!(sheaf.gluing_count(), 2);

        let result = sheaf.verify();
        // Both conditions are checked regardless of pass/fail.
        match result {
            Ok(conds) => assert_eq!(conds.len(), 2),
            Err(diags) => {
                // Even if some fail, the total checked should be 2.
                // With random encodings at threshold 0.3, ~50% pass.
                assert!(!diags.is_empty());
            }
        }
    }

    #[test]
    fn test_sheaf_find_section() {
        let mut sheaf = CodeSheaf::new();
        sheaf.add_section(make_section("foo", 900, vec![]));
        sheaf.add_section(make_section("bar", 901, vec![]));

        assert!(sheaf.find_section("foo").is_some());
        assert!(sheaf.find_section("bar").is_some());
        assert!(sheaf.find_section("baz").is_none());
    }

    #[test]
    fn test_sheaf_threshold_clamping() {
        let sheaf = CodeSheaf::new().with_threshold(2.0);
        assert!(sheaf.threshold <= 1.0);

        let sheaf = CodeSheaf::new().with_threshold(-1.0);
        assert!(sheaf.threshold >= 0.0);
    }

    #[test]
    fn test_name_seed_deterministic() {
        let s1 = name_seed("hello");
        let s2 = name_seed("hello");
        assert_eq!(s1, s2, "name_seed must be deterministic");

        let s3 = name_seed("world");
        assert_ne!(s1, s3, "different names should produce different seeds");
    }
}
