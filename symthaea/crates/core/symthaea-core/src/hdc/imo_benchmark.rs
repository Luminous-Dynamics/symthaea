// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # IMO Benchmark — what can the primitive library actually solve?
//!
//! This module is the **falsification test** for Phases 1 through 4. It
//! takes a curated set of IMO-style problems and attempts to solve each
//! using the existing tactic library. The result is a report:
//!
//! - **Solved problems**: ground truth that the library works
//! - **Failed problems**: concrete evidence of what's missing
//!
//! The problems are deliberately stratified by expected difficulty:
//!
//! 1. **Trivial** — direct single-tactic application (≥ 90% expected solve)
//! 2. **Easy** — single tactic but requires correct parameter choice (≥ 75%)
//! 3. **Medium** — multi-tactic composition in sequence (≥ 50%)
//! 4. **Hard** — requires creative tactic selection, potentially failing
//!    because our tactic wrappers are parameterized (≥ 20%)
//!
//! This is **not** an automated prover — each problem is manually
//! formalized. The goal is to measure *what our primitives can handle*
//! when you tell them the right strategy, not to automate strategy
//! selection (that's Phase 4.5 + more).

use crate::hdc::barycentric::{centroid, circumcenter};
use crate::hdc::combinatorial::{
    find_linear_invariant, find_linear_monovariant, pigeonhole_apply, pigeonhole_min_max_bucket,
};
use crate::hdc::computational_geometry::Point2D;
use crate::hdc::diophantine::pell_equation;
use crate::hdc::inequalities::{amgm_holds, cauchy_schwarz_holds, schur_t1_holds};
use crate::hdc::number_theory::NumberTheoryEngine;
use crate::hdc::synthetic_geometry::{GeomPredicate, GeomState};

// ─── Benchmark result types ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BenchResult {
    /// Problem solved — the primitive library produced a verified answer.
    Solved,
    /// Problem attempted but not solved — the failure is recorded with
    /// a short explanation.
    Unsolved(String),
}

impl BenchResult {
    pub fn is_solved(&self) -> bool {
        matches!(self, BenchResult::Solved)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Difficulty {
    Trivial,
    Easy,
    Medium,
    Hard,
}

#[derive(Debug, Clone)]
pub struct BenchProblem {
    pub name: &'static str,
    pub difficulty: Difficulty,
    pub domain: &'static str,
    pub description: &'static str,
}

// ─── Problem 1: Pigeonhole on mod 6 (Trivial, Combinatorics) ────────────────
//
// "Among any 7 integers, some two leave the same remainder when divided by 6."
// Direct pigeonhole: 7 items, 6 boxes → at least one box has ≥ 2.

pub fn problem_01_pigeonhole_mod_6() -> (BenchProblem, BenchResult) {
    let p = BenchProblem {
        name: "Pigeonhole mod 6",
        difficulty: Difficulty::Trivial,
        domain: "Combinatorics",
        description: "Among any 7 integers, some two are congruent mod 6.",
    };
    // Direct applicator
    let min_max = pigeonhole_min_max_bucket(7, 6);
    if min_max >= 2 {
        // Also verify on a concrete witness
        let witnesses: Vec<i32> = vec![3, 14, 27, 100, 5, 18, 71];
        let res = pigeonhole_apply(&witnesses, |&n: &i32| (n % 6 + 6) % 6, 2);
        match res {
            Some(_) => (p, BenchResult::Solved),
            None => (p, BenchResult::Unsolved("witness check failed".into())),
        }
    } else {
        (
            p,
            BenchResult::Unsolved(format!("min_max = {}, not 2", min_max)),
        )
    }
}

// ─── Problem 2: Pell existence (Trivial, Number Theory) ────────────────────
//
// "Show that x² − 13y² = 1 has a positive-integer solution."

pub fn problem_02_pell_d13() -> (BenchProblem, BenchResult) {
    let p = BenchProblem {
        name: "Pell D=13 existence",
        difficulty: Difficulty::Trivial,
        domain: "Number Theory",
        description: "x² − 13y² = 1 has a positive-integer solution (in fact (649, 180)).",
    };
    match pell_equation(13) {
        Some(sol) => {
            let (x, y) = sol.fundamental;
            if x == 649 && y == 180 && sol.verify(x, y) {
                (p, BenchResult::Solved)
            } else {
                (
                    p,
                    BenchResult::Unsolved(format!("unexpected fundamental ({}, {})", x, y)),
                )
            }
        }
        None => (p, BenchResult::Unsolved("Pell returned None".into())),
    }
}

// ─── Problem 3: Quadratic Residue via Legendre (Easy, Number Theory) ───────
//
// "Show that 2 is a quadratic residue modulo every prime p ≡ ±1 (mod 8)."
//
// We test the specific cases p ∈ {7, 17, 23, 41} — primes ≡ ±1 (mod 8) — and
// verify (2 / p) = 1 for each.

pub fn problem_03_quadratic_residue_2() -> (BenchProblem, BenchResult) {
    let p = BenchProblem {
        name: "QR: 2 is a residue mod p ≡ ±1 (mod 8)",
        difficulty: Difficulty::Easy,
        domain: "Number Theory",
        description: "2 is a QR mod 7, 17, 23, 41 (all primes ≡ ±1 mod 8).",
    };
    let engine = NumberTheoryEngine::new();
    let primes_mod_1_or_minus_1 = [7i64, 17, 23, 41];
    let mut failures = Vec::new();
    for &pr in &primes_mod_1_or_minus_1 {
        if engine.legendre_symbol(2, pr) != 1 {
            failures.push(pr);
        }
    }
    if failures.is_empty() {
        (p, BenchResult::Solved)
    } else {
        (
            p,
            BenchResult::Unsolved(format!("Legendre (2/p) ≠ 1 for p ∈ {:?}", failures)),
        )
    }
}

// ─── Problem 4: Inscribed Angle via saturation (Medium, Geometry) ──────────
//
// "In a cyclic quadrilateral ABCD, ∠BAC = ∠BDC (angles subtending arc BC)."
//
// This exercises the full forward-saturation loop: register the Concyclic
// fact, run `saturate()`, verify the AngleEq derivation appears.

pub fn problem_04_inscribed_angle() -> (BenchProblem, BenchResult) {
    let p = BenchProblem {
        name: "Inscribed angle theorem",
        difficulty: Difficulty::Medium,
        domain: "Geometry",
        description: "In cyclic quadrilateral ABCD (on unit circle), ∠BAC = ∠BDC.",
    };
    let mut state = GeomState::new();
    state.add_point("A", Point2D::new(1.0, 0.0));
    state.add_point("B", Point2D::new(0.0, 1.0));
    state.add_point("C", Point2D::new(-1.0, 0.0));
    state.add_point("D", Point2D::new(0.0, -1.0));
    state.add_fact(GeomPredicate::Concyclic(
        "A".into(),
        "B".into(),
        "C".into(),
        "D".into(),
    ));
    state.saturate(10);
    let target = GeomPredicate::AngleEq(
        "B".into(),
        "A".into(),
        "C".into(),
        "B".into(),
        "D".into(),
        "C".into(),
    );
    if state.facts.contains(&target) && state.facts_consistent() {
        (p, BenchResult::Solved)
    } else {
        (
            p,
            BenchResult::Unsolved("saturation did not derive AngleEq(B,A,C,B,D,C)".into()),
        )
    }
}

// ─── Problem 5: AM-GM-HM chain on a concrete triple (Easy, Inequalities) ───
//
// "For positive reals {1, 2, 4}: HM ≤ GM ≤ AM. Verify numerically."

pub fn problem_05_amgm_chain() -> (BenchProblem, BenchResult) {
    let p = BenchProblem {
        name: "HM ≤ GM ≤ AM on {1, 2, 4}",
        difficulty: Difficulty::Easy,
        domain: "Inequalities",
        description: "Classical mean-inequality chain verified on a concrete triple.",
    };
    let xs = [1.0, 2.0, 4.0];
    let hm = xs.len() as f64 / xs.iter().map(|x| 1.0 / x).sum::<f64>();
    let gm = xs
        .iter()
        .copied()
        .product::<f64>()
        .powf(1.0 / xs.len() as f64);
    let am = xs.iter().sum::<f64>() / xs.len() as f64;
    if hm <= gm + 1e-9 && gm <= am + 1e-9 && amgm_holds(&xs) {
        (p, BenchResult::Solved)
    } else {
        (
            p,
            BenchResult::Unsolved(format!("HM={:.3}, GM={:.3}, AM={:.3}", hm, gm, am)),
        )
    }
}

// ─── Problem 6: Cauchy-Schwarz with ratio witness (Medium, Inequalities) ───
//
// "For vectors a = (1, 2, 3), b = (4, 5, 6), verify (Σaᵢbᵢ)² ≤ (Σaᵢ²)(Σbᵢ²)
//  and identify the equality-case gap."

pub fn problem_06_cauchy_schwarz_gap() -> (BenchProblem, BenchResult) {
    let p = BenchProblem {
        name: "Cauchy-Schwarz on (1,2,3), (4,5,6) with slack",
        difficulty: Difficulty::Easy,
        domain: "Inequalities",
        description: "Verify C-S and quantify slack (non-zero because vectors aren't proportional).",
    };
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];
    if !cauchy_schwarz_holds(&a, &b) {
        return (p, BenchResult::Unsolved("C-S violated".into()));
    }
    // Slack: (Σa²)(Σb²) − (Σab)² = 14·77 − 32² = 1078 − 1024 = 54
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let a2: f64 = a.iter().map(|x| x * x).sum();
    let b2: f64 = b.iter().map(|x| x * x).sum();
    let slack = a2 * b2 - dot * dot;
    if (slack - 54.0).abs() < 1e-9 {
        (p, BenchResult::Solved)
    } else {
        (
            p,
            BenchResult::Unsolved(format!("slack = {}, expected 54", slack)),
        )
    }
}

// ─── Problem 7: CRT + QR composition (Medium, Number Theory) ──────────────
//
// "Show there exists x with x ≡ 1 (mod 4), x ≡ 2 (mod 5), and x is a
//  quadratic non-residue mod 11."
//
// Requires composing CRT (existence) + Legendre (classification).

pub fn problem_07_crt_plus_qr() -> (BenchProblem, BenchResult) {
    let p = BenchProblem {
        name: "CRT + QR composition",
        difficulty: Difficulty::Medium,
        domain: "Number Theory",
        description: "∃x. x ≡ 1 (mod 4), x ≡ 2 (mod 5), (x/11) = −1.",
    };
    let engine = NumberTheoryEngine::new();
    let (x, m) = match engine.crt(&[(1, 4), (2, 5)]) {
        Some((x, m)) => (x, m),
        None => return (p, BenchResult::Unsolved("CRT failed".into())),
    };
    if x != 17 || m != 20 {
        return (
            p,
            BenchResult::Unsolved(format!("CRT got ({}, {}), expected (17, 20)", x, m)),
        );
    }
    let leg = engine.legendre_symbol(x, 11);
    if leg != -1 {
        return (
            p,
            BenchResult::Unsolved(format!("(17/11) = {}, expected -1", leg)),
        );
    }
    (p, BenchResult::Solved)
}

// ─── Problem 8: Chip-firing conservation + termination (Medium, Comb) ─────
//
// "A chip-firing game: transition (a, b) → (a−1, b+1). Prove (i) the chip
//  count a+b is conserved, (ii) the game terminates (a strictly decreases)."

pub fn problem_08_chip_firing() -> (BenchProblem, BenchResult) {
    let p = BenchProblem {
        name: "Chip-firing conservation + termination",
        difficulty: Difficulty::Medium,
        domain: "Combinatorics",
        description: "Sum invariant + monovariant on a discrete transition.",
    };
    let trajectory = vec![
        vec![5.0, 0.0],
        vec![4.0, 1.0],
        vec![3.0, 2.0],
        vec![2.0, 3.0],
        vec![1.0, 4.0],
        vec![0.0, 5.0],
    ];
    let invariant = find_linear_invariant(&trajectory);
    let monovariant = find_linear_monovariant(&trajectory, true);
    match (invariant, monovariant) {
        (Some(_), Some(_)) => (p, BenchResult::Solved),
        (None, _) => (p, BenchResult::Unsolved("no invariant found".into())),
        (_, None) => (p, BenchResult::Unsolved("no monovariant found".into())),
    }
}

// ─── Problem 9: Triangle classical centers (Easy, Geometry) ──────────────
//
// "For the 3-4-5 right triangle with vertices A=(0,0), B=(4,0), C=(0,3),
//  compute the circumcenter (midpoint of hypotenuse) and centroid."

pub fn problem_09_triangle_centers() -> (BenchProblem, BenchResult) {
    let p = BenchProblem {
        name: "3-4-5 triangle circumcenter + centroid",
        difficulty: Difficulty::Easy,
        domain: "Geometry",
        description: "Circumcenter at midpoint of hypotenuse (2, 1.5), centroid at (4/3, 1).",
    };
    let a = Point2D::new(0.0, 0.0);
    let b = Point2D::new(4.0, 0.0);
    let c = Point2D::new(0.0, 3.0);
    let o = circumcenter(&a, &b, &c);
    let g = centroid(&a, &b, &c);
    let o_ok = (o.x - 2.0).abs() < 1e-7 && (o.y - 1.5).abs() < 1e-7;
    let g_ok = (g.x - 4.0 / 3.0).abs() < 1e-7 && (g.y - 1.0).abs() < 1e-7;
    if o_ok && g_ok {
        (p, BenchResult::Solved)
    } else {
        (
            p,
            BenchResult::Unsolved(format!(
                "O = ({:.3}, {:.3}), G = ({:.3}, {:.3})",
                o.x, o.y, g.x, g.y
            )),
        )
    }
}

// ─── Problem 10: Schur-like symmetric inequality (Hard, Inequalities) ────
//
// "For non-negative reals (a, b, c), prove that
//      a(a − b)(a − c) + b(b − a)(b − c) + c(c − a)(c − b) ≥ 0
//  (Schur's inequality at t=1)."
//
// HARD because our numerical checker only verifies it at specific points.
// A full symbolic proof requires Phase 3B SOS decomposition.

pub fn problem_10_schur_numerical() -> (BenchProblem, BenchResult) {
    let p = BenchProblem {
        name: "Schur t=1 (numerical witness)",
        difficulty: Difficulty::Hard,
        domain: "Inequalities",
        description: "Numerically verify Schur on a dense sample grid.",
    };
    // Sample on a 5×5×5 grid in [0, 2]
    let mut failed = false;
    for i in 0..=4 {
        for j in 0..=4 {
            for k in 0..=4 {
                let a = i as f64 * 0.5;
                let b = j as f64 * 0.5;
                let c = k as f64 * 0.5;
                if !schur_t1_holds(a, b, c) {
                    failed = true;
                }
            }
        }
    }
    if failed {
        (
            p,
            BenchResult::Unsolved("Schur violated on grid — unexpected; primitive bug?".into()),
        )
    } else {
        // Numerical verification succeeded on 125 points. This is a
        // "soft" solve — it demonstrates the inequality on a dense
        // sample but is not a symbolic proof.
        (p, BenchResult::Solved)
    }
}

// ─── Problems 11-14: Functional equations (Phase 3C) ────────────────────────
//
// Each problem feeds a natural-language statement through
// `template_functional_equation` (the parser-level detector for the IMO NL
// pipeline) and verifies that the resulting `CurriculumProblem`
//
//   1. has the expected `EquationKind`,
//   2. solves to true via the curriculum dispatch, and
//   3. produces a non-empty canonical_answer() string.
//
// This is the end-to-end existence proof for Phase 3C: NL text → cascade
// detector → tactic-grade problem → solved + answer formatted.

fn fe_solve_check(
    name: &'static str,
    difficulty: Difficulty,
    description: &'static str,
    text: &str,
    expected_kind: crate::hdc::functional_equations::EquationKind,
) -> (BenchProblem, BenchResult) {
    use crate::hdc::curriculum::ProblemKind;
    use crate::hdc::imo_nl_parser::template_functional_equation;
    let p = BenchProblem {
        name,
        difficulty,
        domain: "Functional Equations",
        description,
    };
    let parsed = match template_functional_equation(text) {
        Some(q) => q,
        None => {
            return (
                p,
                BenchResult::Unsolved(format!("template did not match: \"{}\"", text)),
            );
        }
    };
    match &parsed.kind {
        ProblemKind::FunctionalEquationFindAll { kind } if *kind == expected_kind => {}
        ProblemKind::FunctionalEquationFindAll { kind } => {
            return (
                p,
                BenchResult::Unsolved(format!("expected {:?}, got {:?}", expected_kind, kind)),
            );
        }
        other => {
            return (
                p,
                BenchResult::Unsolved(format!("wrong ProblemKind: {:?}", other)),
            );
        }
    }
    if !parsed.solve() {
        return (p, BenchResult::Unsolved("solve() returned false".into()));
    }
    match parsed.canonical_answer() {
        Some(ans) if !ans.is_empty() => (p, BenchResult::Solved),
        _ => (p, BenchResult::Unsolved("no canonical_answer".into())),
    }
}

pub fn problem_11_cauchy_additive() -> (BenchProblem, BenchResult) {
    use crate::hdc::functional_equations::EquationKind;
    fe_solve_check(
        "Cauchy additive (continuous case)",
        Difficulty::Trivial,
        "Find all f: R → R with f(x+y) = f(x) + f(y).",
        "Find all functions f: R → R such that f(x+y) = f(x) + f(y) for all real x, y.",
        EquationKind::CauchyAdditive,
    )
}

pub fn problem_12_multiplicative() -> (BenchProblem, BenchResult) {
    use crate::hdc::functional_equations::EquationKind;
    fe_solve_check(
        "Multiplicative on positives",
        Difficulty::Easy,
        "Find all f: R → R with f(xy) = f(x)f(y) for positive reals.",
        "Find all functions f: R → R such that f(xy) = f(x)f(y) for all positive x, y.",
        EquationKind::Multiplicative,
    )
}

pub fn problem_13_exponential_paraphrased() -> (BenchProblem, BenchResult) {
    use crate::hdc::functional_equations::EquationKind;
    fe_solve_check(
        "Exponential law (alternate variables)",
        Difficulty::Medium,
        "Determine all f satisfying f(a+b) = f(a)f(b) — uses (a, b) instead of (x, y).",
        "Determine all functions f satisfying f(a+b) = f(a)f(b) for every real a, b.",
        EquationKind::Exponential,
    )
}

pub fn problem_14_involution_paraphrased() -> (BenchProblem, BenchResult) {
    use crate::hdc::functional_equations::EquationKind;
    fe_solve_check(
        "Involution (alternate variable)",
        Difficulty::Hard,
        "Find all f with f(f(t)) = t — uses t instead of x.",
        "Find all functions f: R → R such that f(f(t)) = t for every real t.",
        EquationKind::Involution,
    )
}

// ─── The benchmark runner ───────────────────────────────────────────────────

pub fn run_benchmark() -> Vec<(BenchProblem, BenchResult)> {
    vec![
        problem_01_pigeonhole_mod_6(),
        problem_02_pell_d13(),
        problem_03_quadratic_residue_2(),
        problem_04_inscribed_angle(),
        problem_05_amgm_chain(),
        problem_06_cauchy_schwarz_gap(),
        problem_07_crt_plus_qr(),
        problem_08_chip_firing(),
        problem_09_triangle_centers(),
        problem_10_schur_numerical(),
        problem_11_cauchy_additive(),
        problem_12_multiplicative(),
        problem_13_exponential_paraphrased(),
        problem_14_involution_paraphrased(),
    ]
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The IMO benchmark: run all 10 problems, print a report, and
    /// assert that at least 80% are solved. The 80% floor reflects that
    /// problems 1-9 are within direct library scope; problem 10 is a
    /// soft-solve (numerical only) and we explicitly allow failure.
    #[test]
    fn test_imo_benchmark_report() {
        let results = run_benchmark();

        eprintln!("\n════════════════════════════════════════════════════════════");
        eprintln!("  IMO BENCHMARK — WHAT CAN THE PRIMITIVE LIBRARY SOLVE?");
        eprintln!(
            "  {} problems across 4 difficulty tiers × 5 domains",
            results.len()
        );
        eprintln!("════════════════════════════════════════════════════════════");

        let mut by_difficulty: std::collections::HashMap<Difficulty, (usize, usize)> =
            std::collections::HashMap::new();
        let mut by_domain: std::collections::HashMap<&'static str, (usize, usize)> =
            std::collections::HashMap::new();
        let mut total_solved = 0usize;

        for (prob, res) in &results {
            let solved = res.is_solved();
            let marker = if solved { "✓" } else { "✗" };
            eprintln!(
                "  {} [{:?}] {} — {}",
                marker, prob.difficulty, prob.name, prob.description
            );
            if let BenchResult::Unsolved(why) = res {
                eprintln!("      reason: {}", why);
            }
            let (s, t) = by_difficulty.entry(prob.difficulty).or_insert((0, 0));
            *t += 1;
            if solved {
                *s += 1;
            }
            let (ds, dt) = by_domain.entry(prob.domain).or_insert((0, 0));
            *dt += 1;
            if solved {
                *ds += 1;
            }
            if solved {
                total_solved += 1;
            }
        }

        eprintln!("\n────────────────────────────────────────────────────────────");
        eprintln!("  SUMMARY BY DIFFICULTY");
        eprintln!("────────────────────────────────────────────────────────────");
        for diff in [
            Difficulty::Trivial,
            Difficulty::Easy,
            Difficulty::Medium,
            Difficulty::Hard,
        ] {
            if let Some(&(s, t)) = by_difficulty.get(&diff) {
                eprintln!("  {:7?}: {}/{} solved", diff, s, t);
            }
        }
        eprintln!("\n  SUMMARY BY DOMAIN");
        eprintln!("────────────────────────────────────────────────────────────");
        let mut domains: Vec<&&str> = by_domain.keys().collect();
        domains.sort();
        for d in domains {
            let (s, t) = by_domain[*d];
            eprintln!("  {:20}: {}/{} solved", d, s, t);
        }
        eprintln!(
            "\n  OVERALL: {}/{} solved ({:.1}%)",
            total_solved,
            results.len(),
            total_solved as f64 / results.len() as f64 * 100.0
        );
        eprintln!("════════════════════════════════════════════════════════════");

        // Require ≥ 80% solve rate — anything less would signal a real
        // capability regression.
        let solve_rate = total_solved as f64 / results.len() as f64;
        assert!(
            solve_rate >= 0.80,
            "benchmark solve rate {:.1}% < 80%",
            solve_rate * 100.0
        );
    }

    #[test]
    fn test_benchmark_all_problems_run_without_panic() {
        // Sanity check: every problem runs to a definite result, no
        // panics or infinite loops.
        let results = run_benchmark();
        assert_eq!(results.len(), 14);
        for (prob, res) in &results {
            assert!(
                matches!(res, BenchResult::Solved | BenchResult::Unsolved(_)),
                "problem {} returned unexpected result",
                prob.name
            );
        }
    }

    /// Phase 3C-specific assertion: all four functional equation
    /// problems must solve. This is the falsification test for the
    /// new domain — if any of them fails, either the parser detector
    /// regressed or the curriculum dispatch broke.
    #[test]
    fn test_benchmark_functional_equation_domain_all_solved() {
        let results = run_benchmark();
        let fe: Vec<_> = results
            .iter()
            .filter(|(p, _)| p.domain == "Functional Equations")
            .collect();
        assert_eq!(fe.len(), 4, "expected 4 functional equation problems");
        for (prob, res) in &fe {
            assert!(
                res.is_solved(),
                "functional equation problem '{}' failed: {:?}",
                prob.name,
                res
            );
        }
    }
}
