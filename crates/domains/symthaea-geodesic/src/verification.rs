// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Verification Stack — 3-layer code quality checking.
//!
//! Layer 1 (Syntax): Lightweight bracket/keyword balance check (~1ms)
//! Layer 2 (Semantic): HDC re-encode + cosine similarity to intent (~1ms)
//! Layer 3 (Structural): PDG → Betti numbers vs target topology (~5ms)

use crate::pdg::ProgramDependenceGraph;
use crate::topology::{BettiNumbers, TopologicalFingerprint};
use symthaea_core::hdc::binary_hv::BinaryHV;

// ---------------------------------------------------------------------------
// Deterministic seed for semantic re-encoding of source code.
// Uses the same byte-folding approach as token_codebook::token_seed().
// ---------------------------------------------------------------------------
fn source_seed(source: &str) -> u64 {
    source
        .bytes()
        .enumerate()
        .fold(0u64, |acc, (i, b)| acc.wrapping_add((b as u64) << (i % 8)))
}

// ---------------------------------------------------------------------------
// Verification result
// ---------------------------------------------------------------------------

/// Result of the 3-layer verification.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Whether Layer 1 (syntax) passed.
    pub syntax_ok: bool,
    /// Syntax errors found (empty if `syntax_ok` is true).
    pub syntax_errors: Vec<String>,
    /// Cosine similarity between re-encoded source and intent HV (0.0–1.0).
    pub semantic_similarity: f64,
    /// Whether Layer 2 (semantic) passed the threshold.
    pub semantic_ok: bool,
    /// Whether Layer 3 (structural) passed all Betti checks.
    pub structural_ok: bool,
    /// Structural violations (empty if `structural_ok` is true).
    pub structural_violations: Vec<String>,
    /// Actual Betti numbers computed from the generated code.
    pub actual_betti: BettiNumbers,
    /// True only when all three layers pass.
    pub all_passed: bool,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Verify generated Rust source code against target topology.
///
/// Runs the 3-layer verification stack:
/// 1. **Syntax**: Bracket matching, keyword balance, string literal closure.
/// 2. **Semantic**: Re-encode source as [`BinaryHV`], compare to `intent_hv`.
///    Skipped (passes automatically) if `intent_hv` is `None`.
/// 3. **Structural**: Build PDG from source, compute Betti numbers, compare
///    against `target_betti`.
///
/// # Arguments
///
/// * `source` — Generated Rust source code to verify.
/// * `function_name` — Name of the function being verified (used for PDG).
/// * `target_betti` — Expected Betti numbers the code's topology must match.
/// * `intent_hv` — Optional intent encoding for semantic comparison.
/// * `semantic_threshold` — Minimum similarity for Layer 2 to pass.
pub fn verify_generated_code(
    source: &str,
    function_name: &str,
    target_betti: &BettiNumbers,
    intent_hv: Option<&BinaryHV>,
    semantic_threshold: f64,
) -> VerificationResult {
    // Layer 1: Syntax
    let (syntax_ok, syntax_errors) = check_syntax(source);

    // Layer 2: Semantic
    let (semantic_similarity, semantic_ok) = check_semantic(source, intent_hv, semantic_threshold);

    // Layer 3: Structural
    let (structural_ok, structural_violations, actual_betti) =
        check_structural(source, function_name, target_betti);

    let all_passed = syntax_ok && semantic_ok && structural_ok;

    VerificationResult {
        syntax_ok,
        syntax_errors,
        semantic_similarity,
        semantic_ok,
        structural_ok,
        structural_violations,
        actual_betti,
        all_passed,
    }
}

// ---------------------------------------------------------------------------
// Layer 1: Lightweight syntax check
// ---------------------------------------------------------------------------

/// Layer 1: Lightweight syntax check (bracket matching, keyword balance).
///
/// Checks:
/// - Balanced braces `{}`, parentheses `()`, and brackets `[]`.
/// - Every `fn` keyword is followed by an opening brace somewhere.
/// - Every `if` keyword is followed by an opening brace somewhere.
/// - No unclosed string literals within a single line.
fn check_syntax(source: &str) -> (bool, Vec<String>) {
    let mut errors: Vec<String> = Vec::new();

    // --- Bracket balance ---
    let mut brace_depth: i32 = 0; // {}
    let mut paren_depth: i32 = 0; // ()
    let mut bracket_depth: i32 = 0; // []

    let mut in_string = false;
    let mut in_line_comment = false;
    let mut prev_char = '\0';

    for (line_idx, line) in source.lines().enumerate() {
        let line_num = line_idx + 1;
        in_line_comment = false;
        // Check for unclosed string literals on this line
        let mut line_string_count = 0u32;

        for ch in line.chars() {
            if in_line_comment {
                break;
            }

            // Track string literals (skip escaped quotes)
            if ch == '"' && prev_char != '\\' {
                in_string = !in_string;
                line_string_count += 1;
            }

            if !in_string {
                // Detect line comments
                if ch == '/' && prev_char == '/' {
                    in_line_comment = true;
                    continue;
                }

                match ch {
                    '{' => brace_depth += 1,
                    '}' => {
                        brace_depth -= 1;
                        if brace_depth < 0 {
                            errors.push(format!("line {line_num}: unmatched closing brace '}}'"));
                        }
                    }
                    '(' => paren_depth += 1,
                    ')' => {
                        paren_depth -= 1;
                        if paren_depth < 0 {
                            errors.push(format!("line {line_num}: unmatched closing paren ')'"));
                        }
                    }
                    '[' => bracket_depth += 1,
                    ']' => {
                        bracket_depth -= 1;
                        if bracket_depth < 0 {
                            errors.push(format!("line {line_num}: unmatched closing bracket ']'"));
                        }
                    }
                    _ => {}
                }
            }

            prev_char = ch;
        }

        // At end of line, string should be closed (single-line string literals)
        if in_string && line_string_count % 2 != 0 {
            errors.push(format!("line {line_num}: unclosed string literal"));
            in_string = false; // reset for next line
        }
    }

    if brace_depth > 0 {
        errors.push(format!("{brace_depth} unclosed brace(s) '{{'"));
    }
    if paren_depth > 0 {
        errors.push(format!("{paren_depth} unclosed paren(s) '('"));
    }
    if bracket_depth > 0 {
        errors.push(format!("{bracket_depth} unclosed bracket(s) '['"));
    }

    // --- Keyword checks ---
    // Check that every `fn ` has a `{` somewhere after it on the same or later line.
    let lines: Vec<&str> = source.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        // Skip comments
        if trimmed.starts_with("//") {
            continue;
        }
        // Detect fn declarations
        if (trimmed.starts_with("fn ") || trimmed.contains(" fn ")) && !trimmed.contains("//") {
            // Check if there's a brace on this line or subsequent lines
            let has_brace = lines[i..]
                .iter()
                .take(5) // look up to 5 lines ahead
                .any(|l| l.contains('{'));
            if !has_brace {
                errors.push(format!(
                    "line {}: `fn` declaration without opening brace",
                    i + 1
                ));
            }
        }
        // Detect `if` without brace
        if (trimmed.starts_with("if ") || trimmed.starts_with("if(")) && !trimmed.contains("//") {
            let has_brace = lines[i..].iter().take(3).any(|l| l.contains('{'));
            if !has_brace {
                errors.push(format!(
                    "line {}: `if` expression without opening brace",
                    i + 1
                ));
            }
        }
    }

    let ok = errors.is_empty();
    (ok, errors)
}

// ---------------------------------------------------------------------------
// Layer 2: Semantic HDC similarity
// ---------------------------------------------------------------------------

/// Layer 2: Re-encode source as [`BinaryHV`] and compare to intent.
///
/// Uses the same byte-folding seed method as
/// [`token_codebook::token_seed()`](crate::token_codebook) to produce a
/// deterministic encoding of the source text.
///
/// If `intent_hv` is `None`, the check passes automatically with similarity 1.0.
fn check_semantic(source: &str, intent_hv: Option<&BinaryHV>, threshold: f64) -> (f64, bool) {
    let intent = match intent_hv {
        Some(hv) => hv,
        None => return (1.0, true), // No intent → skip semantic check
    };

    let source_hv = BinaryHV::random(source_seed(source));
    let similarity = source_hv.similarity(intent) as f64;
    let ok = similarity >= threshold;
    (similarity, ok)
}

// ---------------------------------------------------------------------------
// Layer 3: Structural (PDG → Betti numbers)
// ---------------------------------------------------------------------------

/// Layer 3: Build PDG, compute topology, check against target Betti numbers.
///
/// Builds a [`ProgramDependenceGraph`] from the source, converts to a
/// simplicial complex, computes [`TopologicalFingerprint`], and checks
/// each Betti number against the target.
fn check_structural(
    source: &str,
    function_name: &str,
    target: &BettiNumbers,
) -> (bool, Vec<String>, BettiNumbers) {
    let pdg = ProgramDependenceGraph::from_rust_source(source, function_name);
    let complex = pdg.to_simplicial_complex();
    let fingerprint = TopologicalFingerprint::from_complex(&complex);
    let actual = &fingerprint.betti;

    let mut violations = Vec::new();

    if actual.beta_0 != target.beta_0 {
        violations.push(format!(
            "beta_0: expected {}, got {}",
            target.beta_0, actual.beta_0
        ));
    }
    if actual.beta_1 != target.beta_1 {
        violations.push(format!(
            "beta_1: expected {}, got {}",
            target.beta_1, actual.beta_1
        ));
    }
    if actual.beta_2 != target.beta_2 {
        violations.push(format!(
            "beta_2: expected {}, got {}",
            target.beta_2, actual.beta_2
        ));
    }

    let ok = violations.is_empty();
    (ok, violations, actual.clone())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Valid Rust function: all 3 layers should pass.
    #[test]
    fn test_verify_valid_code() {
        // Simple linear function: 1 component, 0 cycles, 0 voids.
        let source = r#"
fn add(a: i32, b: i32) -> i32 {
    let result = a + b;
    return result;
}
"#;
        let target = BettiNumbers {
            beta_0: 1,
            beta_1: 0,
            beta_2: 0,
        };

        let result = verify_generated_code(source, "add", &target, None, 0.5);

        assert!(
            result.syntax_ok,
            "syntax should pass: {:?}",
            result.syntax_errors
        );
        assert!(
            result.semantic_ok,
            "semantic should pass (no intent → auto-pass)"
        );
        assert!(
            result.structural_ok,
            "structural should pass: {:?}",
            result.structural_violations
        );
        assert!(result.all_passed, "all layers should pass");
    }

    /// Unbalanced braces: syntax layer should fail.
    #[test]
    fn test_verify_syntax_error() {
        let source = r#"
fn broken() {
    let x = 1;
    if x > 0 {
        let y = 2;
    // missing closing brace
}
"#;
        let target = BettiNumbers {
            beta_0: 1,
            beta_1: 0,
            beta_2: 0,
        };

        let result = verify_generated_code(source, "broken", &target, None, 0.5);

        assert!(
            !result.syntax_ok,
            "syntax should fail for unbalanced braces"
        );
        assert!(!result.syntax_errors.is_empty());
        assert!(!result.all_passed, "all_passed should be false");
    }

    /// Code with 2 loops but target expects β₁=1: structural should fail.
    #[test]
    fn test_verify_topology_mismatch() {
        // Two loops → β₁ should be >= 2 (from loop back-edges).
        let source = r#"
fn two_loops(n: i32) {
    for i in 0..n {
        let a = i;
    }
    for j in 0..n {
        let b = j;
    }
}
"#;
        // Target expects only 1 cycle, but code has 2.
        let target = BettiNumbers {
            beta_0: 1,
            beta_1: 1,
            beta_2: 0,
        };

        let result = verify_generated_code(source, "two_loops", &target, None, 0.5);

        assert!(result.syntax_ok, "syntax should pass");
        // Structural should fail because β₁ mismatch
        assert!(
            !result.structural_ok,
            "structural should fail for topology mismatch: actual={:?}",
            result.actual_betti
        );
        assert!(
            result
                .structural_violations
                .iter()
                .any(|v| v.contains("beta_1")),
            "should report beta_1 violation"
        );
        assert!(!result.all_passed);
    }

    /// End-to-end test exercising all 3 layers with intent HV.
    #[test]
    fn test_verify_all_layers() {
        let source = r#"
fn loop_sum(n: i32) -> i32 {
    let mut total = 0;
    for i in 0..n {
        let step = i;
    }
    return total;
}
"#;
        // Build a PDG to figure out what Betti numbers this code actually has.
        let pdg = ProgramDependenceGraph::from_rust_source(source, "loop_sum");
        let complex = pdg.to_simplicial_complex();
        let fp = TopologicalFingerprint::from_complex(&complex);

        // Use the actual Betti numbers as target so structural passes.
        let target = fp.betti.clone();

        // Create an intent HV — use source_seed for a matching encoding.
        let intent_hv = BinaryHV::random(source_seed(source));

        let result = verify_generated_code(source, "loop_sum", &target, Some(&intent_hv), 0.5);

        assert!(result.syntax_ok, "syntax: {:?}", result.syntax_errors);
        // Semantic: source encoded with same seed → similarity 1.0
        assert!(
            result.semantic_similarity > 0.99,
            "semantic similarity should be ~1.0 for identical encoding, got {}",
            result.semantic_similarity
        );
        assert!(result.semantic_ok, "semantic should pass");
        assert!(
            result.structural_ok,
            "structural: {:?}",
            result.structural_violations
        );
        assert!(result.all_passed, "all layers should pass");
    }
}
