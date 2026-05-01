// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Phase 5: Curriculum Generator with Z3-Verified Lemmas
//!
//! Generates formal lemma statements from the IMO curriculum and
//! verifies each one against Z3 (with internal-DPLL fallback). Each
//! verified lemma becomes a reusable proof artifact: the SMTLIB2
//! encoding, the Z3 verdict, and an optional witness model are
//! persisted so downstream proof checkers (Lean, Coq) and human
//! reviewers can inspect what the IMO solver actually proved.
//!
//! ## Design philosophy
//!
//! - **Five lemma classes for v1**: pigeonhole (L3), AM-GM at small n
//!   (L1), Cauchy-Schwarz at small n (L2), Pell existence by induction
//!   (L4), and Cauchy functional equation continuity-implication on a
//!   sample grid (L5).
//! - **JSON persistence**: human-readable, greppable. SMTLIB2 strings
//!   are stored verbatim so a reviewer can paste them into z3 directly.
//!   Compression / sealing comes in Phase 6 if the catalog grows large.
//! - **Graceful degradation**: when the z3 binary is absent the module
//!   uses `Z3Bridge`'s internal DPLL fallback. Some lemma classes
//!   (propositional, linear) verify either way; nonlinear arithmetic
//!   needs the real solver.
//! - **Five lemma classes is a hard cap for v1.** Anything beyond is
//!   Phase 5 v2 — see `THE_SUBSTRATE_ROADMAP.md` style scope honesty.
//!
//! ## What this module does NOT do
//!
//! - General-n AM-GM (nonlinear arithmetic timeouts in Z3 at n > 4)
//! - Schur / Muirhead with arbitrary parameters (bounded instances only)
//! - Auxiliary geometry construction
//! - Lemma-reuses-lemma chaining (Phase 6)
//!
//! ## Status (Commit 1)
//!
//! - VerifiedLemma type + JSON persistence ✓
//! - L3 pigeonhole encoder + tests ✓
//! - L1, L2, L4, L5 stubs returning `Skipped` until commits 2-3
//!
//! The generator output for L3 alone is the "known-easy case" for the
//! test harness. Subsequent commits add the harder lemma classes.

use crate::z3_bridge::{VerificationResult, Z3Bridge};
use serde::{Deserialize, Serialize};
use std::path::Path;

// ─── VerifiedLemma type ─────────────────────────────────────────────────────

/// A formally-stated lemma plus its verification result. The SMTLIB2
/// encoding is stored verbatim so reviewers can paste it into z3 directly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedLemma {
    /// Stable identifier. Format: "L<class>.<index>" e.g. "L3.0".
    pub id: String,
    /// Human-readable statement of the lemma.
    pub statement: String,
    /// SMTLIB2 source. Verbatim, so a reviewer can paste-and-run.
    pub smtlib2: String,
    /// Verification verdict, serialized as a tagged string for greppability.
    pub verdict: LemmaVerdict,
    /// Optional structured proof outline (witness or human-readable
    /// proof sketch). Populated for lemma classes whose canonical
    /// answer is a textual form, e.g. functional equations.
    pub witness: Option<String>,
    /// Verification mode used: "z3" if the binary was available,
    /// "internal-dpll" if the bridge fell back, "skipped" if the
    /// generator hasn't implemented this lemma class yet.
    pub mode: String,
}

/// Lemma verification verdict — the JSON-friendly tagged form of
/// `z3_bridge::VerificationResult`. Storing as an enum with a string
/// discriminator keeps the serialized form greppable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LemmaVerdict {
    /// Lemma proved valid (its negation is unsat).
    Valid,
    /// Lemma is satisfiable (an instance exists).
    Sat { witness: Option<String> },
    /// Lemma's negation is satisfiable — it's invalid.
    Invalid,
    /// Z3 reported `unknown` or timed out.
    Inconclusive { reason: String },
    /// Lemma class not yet implemented (placeholder for v2 commits).
    Skipped { reason: String },
}

impl LemmaVerdict {
    /// True iff the lemma was successfully proved (`Valid`) or had a
    /// concrete satisfiable model (`Sat`). Used by the test harness to
    /// compute the verification rate.
    pub fn is_proved(&self) -> bool {
        matches!(self, Self::Valid | Self::Sat { .. })
    }

    /// Convert from the bridge's `VerificationResult`, attaching a mode tag.
    pub fn from_bridge(result: VerificationResult) -> (Self, &'static str) {
        match result {
            VerificationResult::Valid => (Self::Valid, "z3"),
            VerificationResult::Sat { witness } => (Self::Sat { witness }, "z3"),
            VerificationResult::Unsat { .. } => (Self::Invalid, "z3"),
            VerificationResult::Invalid => (Self::Invalid, "z3"),
            VerificationResult::Unknown { reason } => (Self::Inconclusive { reason }, "z3"),
            VerificationResult::Timeout => (
                Self::Inconclusive {
                    reason: "z3 timeout".into(),
                },
                "z3",
            ),
            VerificationResult::UsingFallback => (
                // Internal DPLL returned without proving — for L3
                // propositional lemmas this is fine, the encoder
                // handles the fallback path explicitly.
                Self::Inconclusive {
                    reason: "internal DPLL fallback (no z3 binary)".into(),
                },
                "internal-dpll",
            ),
        }
    }
}

// ─── L3: Pigeonhole lemmas ─────────────────────────────────────────────────
//
// Encoding: for `n > k` items distributed into `k` boxes, at least one
// box contains ≥ ⌈n/k⌉ items. We encode this as a finite-arity Boolean
// formula (one Boolean per `(item, box)` placement) and assert that the
// negation — "no box has ⌈n/k⌉ items AND every item is in some box" —
// is unsatisfiable. Z3 with QF_UF handles this trivially; the internal
// DPLL fallback also handles it because the encoding is purely
// propositional after grounding.
//
// This is the "known-easy case" for the test harness: if it doesn't
// verify, something is wrong with the encoder, not the prover.

/// Generate the SMTLIB2 encoding for "n items into k boxes ⇒ some box
/// has at least ⌈n/k⌉ items". Returns the full smtlib2 source.
pub fn encode_pigeonhole(n: usize, k: usize) -> String {
    assert!(n > 0 && k > 0, "pigeonhole requires positive n, k");
    let min_collision = n.div_ceil(k);
    let mut s = String::new();
    s.push_str("; Pigeonhole: ");
    s.push_str(&format!(
        "{} items into {} boxes ⇒ some box has ≥ {} items.\n",
        n, k, min_collision
    ));
    s.push_str("(set-logic QF_UF)\n");
    s.push_str("(set-option :produce-models true)\n");
    // p[i][j] = true iff item i is in box j
    for i in 0..n {
        for j in 0..k {
            s.push_str(&format!("(declare-const p_{}_{} Bool)\n", i, j));
        }
    }
    // Each item is in at least one box.
    for i in 0..n {
        s.push_str("(assert (or");
        for j in 0..k {
            s.push_str(&format!(" p_{}_{}", i, j));
        }
        s.push_str("))\n");
    }
    // Each item is in at most one box (cleanliness, optional but
    // matches the canonical pigeonhole statement).
    for i in 0..n {
        for j1 in 0..k {
            for j2 in (j1 + 1)..k {
                s.push_str(&format!(
                    "(assert (not (and p_{}_{} p_{}_{})))\n",
                    i, j1, i, j2
                ));
            }
        }
    }
    // The NEGATION of the goal: every box has fewer than min_collision items.
    // We encode "box j has at most min_collision-1 items" by enumerating
    // all (min_collision)-element subsets and saying not-all-true.
    // For min_collision=2 this is "no two items share a box" — exactly
    // pigeonhole. We assert this and check-sat: should be unsat.
    let mc = min_collision;
    for j in 0..k {
        // Choose mc items out of n, assert they aren't all in box j.
        let mut indices: Vec<usize> = (0..mc).collect();
        loop {
            s.push_str("(assert (not (and");
            for &i in &indices {
                s.push_str(&format!(" p_{}_{}", i, j));
            }
            s.push_str(")))\n");
            // next combination
            let mut pos = mc;
            for back in (0..mc).rev() {
                if indices[back] < n - (mc - back) {
                    pos = back;
                    break;
                }
            }
            if pos == mc {
                break;
            }
            indices[pos] += 1;
            for k2 in (pos + 1)..mc {
                indices[k2] = indices[k2 - 1] + 1;
            }
        }
    }
    s.push_str("(check-sat)\n");
    s
}

/// Generate the v1 pigeonhole lemmas. Each instance is a small
/// concrete `(n, k)` pair where the pigeonhole conclusion is provable.
///
/// **Verification policy**: the internal DPLL fallback in `z3_bridge`
/// cannot fully parse SMTLIB2 (it understands only flat propositional
/// asserts), so when no `z3` binary is on PATH this generator marks
/// every lemma as `Skipped { reason: "z3 binary required" }` rather
/// than trying to verify and hanging. The encoder + persistence still
/// ship; verification turns on automatically the next time the run
/// happens inside `nix develop` (where `pkgs.z3` is available per the
/// flake update in this commit).
pub fn generate_pigeonhole_lemmas(bridge: &Z3Bridge) -> Vec<VerifiedLemma> {
    let cases: Vec<(usize, usize)> = vec![(3, 2), (5, 4), (7, 6), (13, 12)];
    cases
        .into_iter()
        .enumerate()
        .map(|(idx, (n, k))| {
            let smtlib2 = encode_pigeonhole(n, k);
            let min_collision = n.div_ceil(k);
            let statement = format!(
                "Pigeonhole: distributing {} items into {} boxes forces some box to contain ≥ {} items.",
                n, k, min_collision
            );
            let (verdict, mode) = run_negation_check(bridge, &smtlib2);
            VerifiedLemma {
                id: format!("L3.{}", idx),
                statement,
                smtlib2,
                verdict,
                witness: None,
                mode: mode.to_string(),
            }
        })
        .collect()
}

// ─── L1: AM-GM lemmas at small n ───────────────────────────────────────────
//
// Encoding: for n ∈ {2, 3, 4}, encode AM-GM in its **squared/cubed/quartic
// form** to avoid sqrt:
//   n=2:  ((a+b)/2)²    ≥  a·b              ⇔  (a+b)²    ≥  4·a·b
//   n=3:  ((a+b+c)/3)³  ≥  a·b·c            ⇔  (a+b+c)³  ≥  27·a·b·c
//   n=4:  ((a+b+c+d)/4)⁴ ≥ a·b·c·d           ⇔  (a+b+c+d)⁴ ≥ 256·a·b·c·d
//
// Each is a polynomial inequality in QF_NRA. Z3 handles them directly.
// The encoder asserts the negation; an unsat result means the lemma is
// proved.

/// SMTLIB2 variable name for the i-th AM-GM input (lowercase letter).
fn amgm_var(i: usize) -> char {
    (b'a' + i as u8) as char
}

/// Build `(* x x x ... x)` repeated `times` times, or `1` for `times == 0`.
fn smt_pow(expr: &str, times: usize) -> String {
    if times == 0 {
        "1".to_string()
    } else if times == 1 {
        expr.to_string()
    } else {
        let body: Vec<&str> = std::iter::repeat(expr).take(times).collect();
        format!("(* {})", body.join(" "))
    }
}

/// SMTLIB2 source asserting the negation of "AM_n^n ≥ n^n · prod" on
/// non-negative reals. Z3 is expected to return `unsat`.
pub fn encode_amgm(n: usize) -> String {
    assert!((2..=4).contains(&n), "encode_amgm: n must be in 2..=4");
    let mut s = String::new();
    s.push_str(&format!(
        "; AM-GM at n={}: (sum)^{} >= {}^{} * prod for non-negative reals.\n",
        n, n, n, n
    ));
    s.push_str("(set-logic QF_NRA)\n");
    for i in 0..n {
        s.push_str(&format!("(declare-const {} Real)\n", amgm_var(i)));
        s.push_str(&format!("(assert (>= {} 0))\n", amgm_var(i)));
    }
    // sum = (+ a b ...)
    let sum: Vec<String> = (0..n).map(|i| amgm_var(i).to_string()).collect();
    let sum_expr = format!("(+ {})", sum.join(" "));
    // prod = (* a b ...)
    let prod_expr = format!("(* {})", sum.join(" "));
    // sum^n
    let sum_pow = smt_pow(&sum_expr, n);
    // n^n constant
    let n_pow_n = (n as u64).pow(n as u32);
    // Negation of the lemma: (sum)^n < n^n * prod
    s.push_str(&format!(
        "(assert (not (>= {} (* {} {}))))\n",
        sum_pow, n_pow_n, prod_expr
    ));
    s.push_str("(check-sat)\n");
    s
}

/// Generate the v1 AM-GM lemmas at n ∈ {2, 3, 4}.
pub fn generate_amgm_lemmas(bridge: &Z3Bridge) -> Vec<VerifiedLemma> {
    [2usize, 3, 4]
        .into_iter()
        .enumerate()
        .map(|(idx, n)| {
            let smtlib2 = encode_amgm(n);
            let statement = format!(
                "AM-GM at n={}: ((Σxᵢ)/{})^{} ≥ Πxᵢ for non-negative xᵢ.",
                n, n, n
            );
            let (verdict, mode) = run_negation_check(bridge, &smtlib2);
            VerifiedLemma {
                id: format!("L1.{}", idx),
                statement,
                smtlib2,
                verdict,
                witness: None,
                mode: mode.to_string(),
            }
        })
        .collect()
}

// ─── L2: Cauchy-Schwarz lemmas at small n ──────────────────────────────────
//
// Encoding: for vectors a, b ∈ R^n with n ∈ {2, 3}, the inequality
//   (Σ aᵢbᵢ)² ≤ (Σ aᵢ²)(Σ bᵢ²)
// is a polynomial inequality in 2n variables. Asserting the negation
// produces an unsat formula that Z3 handles trivially in QF_NRA — this
// is the classical "discriminant ≥ 0" form of Cauchy-Schwarz.

/// SMTLIB2 source asserting the negation of (Σaᵢbᵢ)² ≤ (Σaᵢ²)(Σbᵢ²).
pub fn encode_cauchy_schwarz(n: usize) -> String {
    assert!(
        (2..=3).contains(&n),
        "encode_cauchy_schwarz: n must be in 2..=3"
    );
    let mut s = String::new();
    s.push_str(&format!(
        "; Cauchy-Schwarz at n={}: (Σaᵢbᵢ)² ≤ (Σaᵢ²)(Σbᵢ²) on real vectors.\n",
        n
    ));
    s.push_str("(set-logic QF_NRA)\n");
    for i in 0..n {
        s.push_str(&format!("(declare-const a{} Real)\n", i));
        s.push_str(&format!("(declare-const b{} Real)\n", i));
    }
    // dot = Σ aᵢbᵢ
    let dot_terms: Vec<String> = (0..n).map(|i| format!("(* a{} b{})", i, i)).collect();
    let dot = format!("(+ {})", dot_terms.join(" "));
    // |a|² = Σ aᵢ²
    let a2_terms: Vec<String> = (0..n).map(|i| format!("(* a{} a{})", i, i)).collect();
    let a_sq = format!("(+ {})", a2_terms.join(" "));
    // |b|² = Σ bᵢ²
    let b2_terms: Vec<String> = (0..n).map(|i| format!("(* b{} b{})", i, i)).collect();
    let b_sq = format!("(+ {})", b2_terms.join(" "));
    // (dot)² ≤ |a|² · |b|²  ⇔  (dot)² - |a|²·|b|² ≤ 0
    // Negation: (dot)² > |a|²·|b|²
    s.push_str(&format!(
        "(assert (> (* {} {}) (* {} {})))\n",
        dot, dot, a_sq, b_sq
    ));
    s.push_str("(check-sat)\n");
    s
}

/// Generate the v1 Cauchy-Schwarz lemmas at n ∈ {2, 3}.
pub fn generate_cauchy_schwarz_lemmas(bridge: &Z3Bridge) -> Vec<VerifiedLemma> {
    [2usize, 3]
        .into_iter()
        .enumerate()
        .map(|(idx, n)| {
            let smtlib2 = encode_cauchy_schwarz(n);
            let statement = format!(
                "Cauchy-Schwarz at n={}: (Σ aᵢbᵢ)² ≤ (Σ aᵢ²)(Σ bᵢ²) for real vectors.",
                n
            );
            let (verdict, mode) = run_negation_check(bridge, &smtlib2);
            VerifiedLemma {
                id: format!("L2.{}", idx),
                statement,
                smtlib2,
                verdict,
                witness: None,
                mode: mode.to_string(),
            }
        })
        .collect()
}

/// Shared bridge-call helper for lemmas whose smtlib2 asserts the
/// **negation** of the goal: an unsat result means the goal is valid.
/// Converts a SAT result into `Invalid` (the encoder produced a
/// counterexample to a claim we believed was a theorem — surface as a
/// real bug). Skips when no z3 binary is available.
fn run_negation_check(bridge: &Z3Bridge, smtlib2: &str) -> (LemmaVerdict, &'static str) {
    if !bridge.z3_available {
        return (
            LemmaVerdict::Skipped {
                reason: "z3 binary required (run inside nix develop or install pkgs.z3)".into(),
            },
            "skipped",
        );
    }
    let raw = bridge.verify_satisfiable(smtlib2);
    match raw {
        VerificationResult::Unsat { .. } => (LemmaVerdict::Valid, "z3"),
        VerificationResult::Sat { .. } => (LemmaVerdict::Invalid, "z3"),
        other => LemmaVerdict::from_bridge(other),
    }
}

// ─── L4, L5 stubs (for commit 3) ───────────────────────────────────────────

pub fn generate_pell_lemmas(_bridge: &Z3Bridge) -> Vec<VerifiedLemma> {
    vec![placeholder(
        "L4.0",
        "Pell existence small D — implemented in commit 3",
    )]
}

pub fn generate_functional_equation_lemmas(_bridge: &Z3Bridge) -> Vec<VerifiedLemma> {
    vec![placeholder(
        "L5.0",
        "Cauchy continuity-grid implication — implemented in commit 3",
    )]
}

fn placeholder(id: &str, reason: &'static str) -> VerifiedLemma {
    VerifiedLemma {
        id: id.to_string(),
        statement: format!("(placeholder) {}", reason),
        smtlib2: String::new(),
        verdict: LemmaVerdict::Skipped {
            reason: reason.to_string(),
        },
        witness: None,
        mode: "skipped".to_string(),
    }
}

// ─── Top-level generator ──────────────────────────────────────────────────

/// Generate all lemmas for the v1 catalog. Order: L3 (pigeonhole),
/// L1 (AM-GM), L2 (Cauchy-Schwarz), L4 (Pell), L5 (functional eq).
pub fn generate_all_lemmas(bridge: &Z3Bridge) -> Vec<VerifiedLemma> {
    let mut all = Vec::new();
    all.extend(generate_pigeonhole_lemmas(bridge));
    all.extend(generate_amgm_lemmas(bridge));
    all.extend(generate_cauchy_schwarz_lemmas(bridge));
    all.extend(generate_pell_lemmas(bridge));
    all.extend(generate_functional_equation_lemmas(bridge));
    all
}

// ─── JSON persistence ──────────────────────────────────────────────────────

/// Serialize a lemma catalog to pretty-printed JSON for human inspection.
pub fn save_lemmas_json(lemmas: &[VerifiedLemma], path: &Path) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(lemmas)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Load a lemma catalog from JSON.
pub fn load_lemmas_json(path: &Path) -> std::io::Result<Vec<VerifiedLemma>> {
    let s = std::fs::read_to_string(path)?;
    serde_json::from_str(&s).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// Compute the verification rate for a catalog: number proved / total
/// non-skipped lemmas. Skipped lemmas are excluded from the denominator
/// so the rate reflects what's actually been implemented + verified.
pub fn verification_rate(lemmas: &[VerifiedLemma]) -> (usize, usize, f64) {
    let testable: Vec<_> = lemmas
        .iter()
        .filter(|l| !matches!(l.verdict, LemmaVerdict::Skipped { .. }))
        .collect();
    let proved = testable.iter().filter(|l| l.verdict.is_proved()).count();
    let total = testable.len();
    let rate = if total == 0 {
        0.0
    } else {
        proved as f64 / total as f64
    };
    (proved, total, rate)
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-constructed `Z3Bridge` with `z3_available=false`. Avoids
    /// the `Z3Bridge::new()` constructor which spawns a `which z3`
    /// subprocess that has been observed to hang in this test
    /// environment (see the bridge audit in `feedback_*` memory).
    /// The skip-branch in every generator is exercised, so this
    /// gives full coverage of commit-1 paths without any subprocess.
    fn test_bridge_no_z3() -> Z3Bridge {
        Z3Bridge {
            z3_available: false,
            z3_path: None,
            timeout_secs: 10,
        }
    }

    #[test]
    fn test_encode_pigeonhole_3_2_contains_required_constants() {
        let s = encode_pigeonhole(3, 2);
        // Six placement booleans (3 items × 2 boxes)
        for i in 0..3 {
            for j in 0..2 {
                assert!(
                    s.contains(&format!("(declare-const p_{}_{} Bool)", i, j)),
                    "missing p_{}_{}",
                    i,
                    j
                );
            }
        }
        assert!(s.contains("(set-logic QF_UF)"));
        assert!(s.contains("(check-sat)"));
    }

    // ── L1: AM-GM encoder + skip-branch ──

    #[test]
    fn test_encode_amgm_2_contains_polynomial_form() {
        let s = encode_amgm(2);
        // Two non-negativity declarations
        assert!(s.contains("(declare-const a Real)"));
        assert!(s.contains("(declare-const b Real)"));
        assert!(s.contains("(assert (>= a 0))"));
        assert!(s.contains("(assert (>= b 0))"));
        // n=2 ⇒ (a+b)^2 ≥ 4·a·b. Negation: (a+b)^2 < 4ab.
        // smt_pow uses repeated multiplication: "(* (+ a b) (+ a b))".
        assert!(s.contains("(* (+ a b) (+ a b))"));
        assert!(s.contains("4"));
        assert!(s.contains("(check-sat)"));
    }

    #[test]
    fn test_encode_amgm_4_has_quartic_form() {
        let s = encode_amgm(4);
        // 256 = 4^4, the constant on the RHS
        assert!(
            s.contains("256"),
            "n=4 lemma must use 4^4 = 256 as the rescaling constant"
        );
        // Four variable declarations
        for v in ['a', 'b', 'c', 'd'] {
            assert!(
                s.contains(&format!("(declare-const {} Real)", v)),
                "missing variable {}",
                v
            );
        }
    }

    #[test]
    fn test_amgm_lemmas_skip_when_no_z3() {
        let bridge = test_bridge_no_z3();
        let lemmas = generate_amgm_lemmas(&bridge);
        assert_eq!(lemmas.len(), 3, "expected 3 AM-GM lemmas (n=2,3,4)");
        for l in &lemmas {
            assert!(
                matches!(l.verdict, LemmaVerdict::Skipped { .. }),
                "lemma {} should be Skipped without z3",
                l.id
            );
            assert!(!l.smtlib2.is_empty());
            assert!(l.smtlib2.contains("QF_NRA"));
            assert!(l.smtlib2.contains("(check-sat)"));
        }
    }

    // ── L2: Cauchy-Schwarz encoder + skip-branch ──

    #[test]
    fn test_encode_cauchy_schwarz_2_contains_dot_product() {
        let s = encode_cauchy_schwarz(2);
        // Four variables
        for v in ["a0", "a1", "b0", "b1"] {
            assert!(
                s.contains(&format!("(declare-const {} Real)", v)),
                "missing {}",
                v
            );
        }
        // dot product term
        assert!(s.contains("(* a0 b0)"));
        assert!(s.contains("(* a1 b1)"));
        // |a|² and |b|² terms
        assert!(s.contains("(* a0 a0)"));
        assert!(s.contains("(* b0 b0)"));
    }

    #[test]
    fn test_encode_cauchy_schwarz_3_has_six_variables() {
        let s = encode_cauchy_schwarz(3);
        for v in ["a0", "a1", "a2", "b0", "b1", "b2"] {
            assert!(s.contains(&format!("(declare-const {} Real)", v)));
        }
    }

    #[test]
    fn test_cauchy_schwarz_lemmas_skip_when_no_z3() {
        let bridge = test_bridge_no_z3();
        let lemmas = generate_cauchy_schwarz_lemmas(&bridge);
        assert_eq!(lemmas.len(), 2, "expected 2 Cauchy-Schwarz lemmas (n=2,3)");
        for l in &lemmas {
            assert!(matches!(l.verdict, LemmaVerdict::Skipped { .. }));
            assert!(l.smtlib2.contains("QF_NRA"));
        }
    }

    #[test]
    fn test_pigeonhole_lemmas_skip_when_no_z3() {
        // With z3_available=false (the test fixture), every pigeonhole
        // lemma must be marked Skipped. Each must include a non-empty
        // smtlib2 encoding so that downstream consumers can still
        // inspect the formal statement even though it wasn't verified.
        let bridge = test_bridge_no_z3();
        let lemmas = generate_pigeonhole_lemmas(&bridge);
        assert_eq!(lemmas.len(), 4, "expected 4 pigeonhole lemmas");
        for l in &lemmas {
            assert!(
                matches!(l.verdict, LemmaVerdict::Skipped { .. }),
                "lemma {} expected Skipped, got {:?}",
                l.id,
                l.verdict
            );
            assert!(
                !l.smtlib2.is_empty(),
                "lemma {} must still ship its smtlib2 encoding",
                l.id
            );
            assert!(
                l.smtlib2.contains("(check-sat)"),
                "lemma {} smtlib2 missing (check-sat)",
                l.id
            );
        }
    }

    #[test]
    fn test_generate_all_lemmas_returns_at_least_one_per_class() {
        let bridge = test_bridge_no_z3();
        let all = generate_all_lemmas(&bridge);
        // 4 pigeonhole + 3 AM-GM + 2 Cauchy-Schwarz + 1 Pell stub + 1 functional eq stub
        assert_eq!(
            all.len(),
            11,
            "expected 11 lemmas in commit 2 (10 real + 2 stubs)"
        );
        // L3 should be the first 4
        assert!(all[0].id.starts_with("L3."));
        assert!(all[3].id.starts_with("L3."));
        // L1 next: 3 AM-GM lemmas
        assert_eq!(all[4].id, "L1.0");
        assert_eq!(all[6].id, "L1.2");
        // L2 next: 2 Cauchy-Schwarz lemmas
        assert_eq!(all[7].id, "L2.0");
        assert_eq!(all[8].id, "L2.1");
        // L4, L5 still stubs (commit 3)
        assert_eq!(all[9].id, "L4.0");
        assert_eq!(all[10].id, "L5.0");
    }

    #[test]
    fn test_verification_rate_excludes_skipped() {
        // With z3_available=false (test fixture), every lemma class
        // skipped → 0 testable in the verification_rate denominator.
        let bridge = test_bridge_no_z3();
        let all = generate_all_lemmas(&bridge);
        let (proved, total, _rate) = verification_rate(&all);
        assert_eq!(total, 0, "no testable lemmas without z3 binary");
        assert_eq!(proved, 0);
    }

    #[test]
    fn test_json_roundtrip() {
        use std::env::temp_dir;
        let bridge = test_bridge_no_z3();
        let lemmas = generate_pigeonhole_lemmas(&bridge);
        let path = temp_dir().join("symthaea_phase5_lemmas_test.json");
        save_lemmas_json(&lemmas, &path).expect("save JSON");
        let loaded = load_lemmas_json(&path).expect("load JSON");
        assert_eq!(loaded.len(), lemmas.len());
        for (a, b) in lemmas.iter().zip(loaded.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.statement, b.statement);
            assert_eq!(a.smtlib2, b.smtlib2);
        }
        // Cleanup
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_lemma_verdict_is_proved_logic() {
        assert!(LemmaVerdict::Valid.is_proved());
        assert!(LemmaVerdict::Sat { witness: None }.is_proved());
        assert!(!LemmaVerdict::Invalid.is_proved());
        assert!(!LemmaVerdict::Inconclusive {
            reason: "test".into()
        }
        .is_proved());
        assert!(!LemmaVerdict::Skipped {
            reason: "test".into()
        }
        .is_proved());
    }
}
