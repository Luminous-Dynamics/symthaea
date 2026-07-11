// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Spec conformance: the check axiom-auditing misses.
//!
//! The deepest failure mode of a self-authoring prover is not inventing an axiom
//! — it is proving a *subtly different* theorem than intended: adding a false
//! hypothesis that makes the goal vacuously true, or proving `True`. A clean
//! axiom report says nothing about whether the theorem *statement* is the one
//! you pinned. This module compares the proved statement against the intended
//! spec (whitespace-normalized).

/// Collapse all runs of whitespace to single spaces and trim — so cosmetic
/// formatting differences do not read as a spec mismatch.
pub fn normalize_statement(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// True iff the proved statement matches the pinned spec up to whitespace.
pub fn statement_matches(actual: &str, expected: &str) -> bool {
    normalize_statement(actual) == normalize_statement(expected)
}

/// True iff proof source text contains a `sorry`/`admit` token (word-boundary).
///
/// This is a cheap pre-check; the authoritative unproved-proof signal is
/// `sorryAx` in the `#print axioms` report (see [`crate::axioms`]).
pub fn source_contains_sorry(src: &str) -> bool {
    src.split(|c: char| !c.is_alphanumeric() && c != '_')
        .any(|tok| tok == "sorry" || tok == "admit")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whitespace_differences_still_match() {
        assert!(statement_matches("a + b = b + a", "a  +  b   =\n b + a"));
    }

    #[test]
    fn different_statements_do_not_match() {
        assert!(!statement_matches("a + b = b + a", "a + b = 0"));
        // The vacuous-truth trap: proving `True` is not proving the spec.
        assert!(!statement_matches("True", "forall x, P x"));
    }

    #[test]
    fn source_sorry_detection() {
        assert!(source_contains_sorry("theorem t : P := by sorry"));
        assert!(source_contains_sorry("  admit"));
        assert!(!source_contains_sorry(
            "theorem sorryless : P := by exact h"
        ));
    }
}
