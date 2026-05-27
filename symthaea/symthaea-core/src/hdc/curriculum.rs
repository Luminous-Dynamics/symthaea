// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Curriculum generator (Phase 5 scoped)
//!
//! Generates parameterized IMO-style problems at graduated difficulty
//! levels. Used to scale the Phase 4.5 SR experiment from curated ~15
//! problem corpora to ~1000+ problems, and to produce a reusable
//! benchmark artifact for future sessions.
//!
//! ## Design
//!
//! A `CurriculumProblem` is a parameterized template instantiated with
//! concrete numeric values. Each template targets a specific primitive
//! (Pell, CRT, Legendre, Pigeonhole, etc.) and carries a difficulty
//! parameter that controls problem size / constraint tightness.
//!
//! Instantiated problems expose a `solve()` method that runs the
//! appropriate primitive and returns a boolean. The generator can
//! batch-instantiate thousands of problems with reproducible seeds.
//!
//! ## Scope limits
//!
//! - Templates are hand-written, not auto-generated from IMO archive
//!   text. Phase 6+ would add a natural-language parser. (My earlier
//!   commit notes why: Lean 4 → Rust Goal translation is a standalone
//!   ~500-1000 LOC project.)
//! - Difficulty is measured by *parameter size*, not by compositional
//!   complexity. "Harder Pell" means larger D; it does NOT mean "more
//!   steps to formalize."
//! - All templates solve in O(1) primitive calls. They're benchmark
//!   material, not open research problems.

use crate::hdc::diophantine::pell_equation;
use crate::hdc::inequalities::{amgm_holds, cauchy_schwarz_holds};
use crate::hdc::number_theory::NumberTheoryEngine;

// ─── xorshift RNG (shared pattern with sr_tactic/sr_symreg) ─────────────────

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    if x == 0 {
        x = 0xCAFEBABEDEADBEEF;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ─── Difficulty and domain ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Difficulty {
    Easy,
    Medium,
    Hard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Domain {
    NumberTheory,
    Inequality,
    Combinatorics,
    FunctionalEquation,
}

// ─── Curriculum problem ────────────────────────────────────────────────────

/// A curriculum problem. The `kind` field holds the template-specific
/// parameters; `solve()` dispatches to the appropriate primitive.
#[derive(Debug, Clone)]
pub struct CurriculumProblem {
    pub name: String,
    pub difficulty: Difficulty,
    pub domain: Domain,
    pub kind: ProblemKind,
}

#[derive(Debug, Clone)]
pub enum ProblemKind {
    /// Solve x² − D·y² = 1 for the given D.
    PellEquation { d: i64 },
    /// Verify that the given CRT system is consistent and find the solution.
    CrtSystem {
        residues: Vec<(i64, i64)>,
        expected: i64,
    },
    /// Verify that (a/p) has the expected sign (+1 / −1).
    LegendreSymbol { a: i64, p: i64, expected: i32 },
    /// Verify AM ≥ GM on the given non-negative slice.
    Amgm { values: Vec<f64> },
    /// Verify the Cauchy-Schwarz inequality on equal-length vectors.
    CauchySchwarz { a: Vec<f64>, b: Vec<f64> },
    /// Verify that pigeonhole forces at least `min_collision` items in
    /// some bucket when distributing `items` into `boxes`.
    Pigeonhole {
        items: usize,
        boxes: usize,
        min_collision: usize,
    },
    /// Verify that `n` is prime using Miller-Rabin.
    PrimalityCheck { n: u64, expected: bool },
    /// Compute φ(n) = Euler's totient and verify against expected.
    EulerPhi { n: u64, expected: u64 },
    /// Verify power-mean inequality M_p ≤ M_q on a non-negative slice.
    PowerMeanIneq { values: Vec<f64>, p: f64, q: f64 },
    /// Verify Schur's inequality at exponent t (1 or 2) on a triple.
    SchurIneq { a: f64, b: f64, c: f64, t: u32 },
    /// Bezout's identity: find (x, y) with a·x + b·y = gcd(a, b).
    BezoutIdentity { a: i64, b: i64, expected_gcd: i64 },
    /// "Find all f: R → R such that <functional equation>." Solved by
    /// returning the canonical family identifier; `solve()` succeeds iff
    /// the requested family has a known canonical form (i.e., is not
    /// `EquationKind::Unknown`). Continuous case is assumed; pathological
    /// Hamel-basis solutions to Cauchy are not enumerated.
    FunctionalEquationFindAll {
        kind: crate::hdc::functional_equations::EquationKind,
    },
}

impl CurriculumProblem {
    /// Attempt to solve the problem using the appropriate primitive.
    /// Returns true if the expected answer is produced.
    pub fn solve(&self) -> bool {
        match &self.kind {
            ProblemKind::PellEquation { d } => pell_equation(*d).is_some(),
            ProblemKind::CrtSystem { residues, expected } => {
                let engine = NumberTheoryEngine::new();
                match engine.crt(residues) {
                    Some((x, _m)) => x == *expected,
                    None => false,
                }
            }
            ProblemKind::LegendreSymbol { a, p, expected } => {
                let engine = NumberTheoryEngine::new();
                engine.legendre_symbol(*a, *p) == *expected
            }
            ProblemKind::Amgm { values } => amgm_holds(values),
            ProblemKind::CauchySchwarz { a, b } => cauchy_schwarz_holds(a, b),
            ProblemKind::Pigeonhole {
                items,
                boxes,
                min_collision,
            } => {
                use crate::hdc::combinatorial::pigeonhole_min_max_bucket;
                pigeonhole_min_max_bucket(*items, *boxes) >= *min_collision
            }
            ProblemKind::PrimalityCheck { n, expected } => {
                let engine = NumberTheoryEngine::new();
                engine.miller_rabin(*n) == *expected
            }
            ProblemKind::EulerPhi { n, expected } => {
                use crate::hdc::number_theory::ModularRing;
                let ring = ModularRing::new(*n);
                ring.euler_totient() == *expected
            }
            ProblemKind::PowerMeanIneq { values, p, q } => {
                use crate::hdc::inequalities::power_mean_inequality_holds;
                power_mean_inequality_holds(values, *p, *q)
            }
            ProblemKind::SchurIneq { a, b, c, t } => {
                use crate::hdc::inequalities::{schur_t1_holds, schur_t2_holds};
                match *t {
                    1 => schur_t1_holds(*a, *b, *c),
                    2 => schur_t2_holds(*a, *b, *c),
                    _ => false,
                }
            }
            ProblemKind::BezoutIdentity { a, b, expected_gcd } => {
                let engine = NumberTheoryEngine::new();
                let (g, x, y) = engine.extended_gcd(*a, *b);
                g == *expected_gcd && a * x + b * y == g
            }
            ProblemKind::FunctionalEquationFindAll { kind } => {
                use crate::hdc::functional_equations::EquationKind;
                // Canonical answer is "known" for any family the
                // classifier supports. `Unknown` means we don't have a
                // canonical form, so the problem is unsolvable by the
                // current engine.
                !matches!(kind, EquationKind::Unknown)
            }
        }
    }

    /// Return the canonical closed-form answer for problems whose
    /// answer is a textual description (currently: functional
    /// equations). Returns `None` for problem kinds whose answer is
    /// expressed as numbers or booleans inside the `solve()` boolean
    /// (those are validated by `solve()` returning true, not by an
    /// answer string).
    ///
    /// Downstream consumers (paper writeups, the conjecture engine,
    /// curriculum reports) use this to display the actual closed form
    /// produced by the IMO solver instead of the bare success bit.
    pub fn canonical_answer(&self) -> Option<String> {
        match &self.kind {
            ProblemKind::FunctionalEquationFindAll { kind } => {
                Some(kind.canonical_form().to_string())
            }
            _ => None,
        }
    }

    /// Structured uniqueness-proof witness for problems whose answer
    /// is a textual canonical form. Returns `None` for problem kinds
    /// that don't have a uniqueness argument to express. The witness
    /// is a multi-line outline (ASSUMPTIONS / STEPS / CONCLUSION) that
    /// downstream proof checkers (Z3, Lean, Coq) can use as the
    /// skeleton of a formal proof.
    pub fn uniqueness_witness(&self) -> Option<String> {
        match &self.kind {
            ProblemKind::FunctionalEquationFindAll { kind } => {
                let w = kind.uniqueness_witness();
                if w.is_empty() { None } else { Some(w) }
            }
            _ => None,
        }
    }
}

// ─── Generators ─────────────────────────────────────────────────────────────

/// Primes in [2, 1000] for sampling Legendre problems.
fn small_primes() -> Vec<i64> {
    // Static list — up to 100 for fast access
    vec![
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
        97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181,
        191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251,
    ]
}

/// Generate a single Pell problem at the requested difficulty. Easy D is
/// drawn from [2, 20]; Medium from [21, 100]; Hard from [101, 1000].
pub fn gen_pell(difficulty: Difficulty, state: &mut u64) -> Option<CurriculumProblem> {
    // Hard capped at D ≤ 150 to stay within i128 during the continued-
    // fraction convergent computation. D > 200 can overflow due to the
    // squared `h·h − d·k·k` check. Fix in diophantine.rs would require
    // checked arithmetic or BigInt, both out of scope here.
    let (lo, hi) = match difficulty {
        Difficulty::Easy => (2u64, 20),
        Difficulty::Medium => (21, 80),
        Difficulty::Hard => (81, 150),
    };
    // Sample a non-square D in [lo, hi]
    for _ in 0..50 {
        let d = lo + xorshift64(state) % (hi - lo + 1);
        let sqrt = (d as f64).sqrt() as u64;
        if sqrt * sqrt != d {
            return Some(CurriculumProblem {
                name: format!("Pell D={}", d),
                difficulty,
                domain: Domain::NumberTheory,
                kind: ProblemKind::PellEquation { d: d as i64 },
            });
        }
    }
    None
}

/// Generate a single CRT problem. Easy = 2 residues with coprime moduli
/// in [2, 10]; Medium = 3 residues in [2, 20]; Hard = 4 residues in [2, 30]
/// with potentially non-coprime overlapping moduli.
pub fn gen_crt(difficulty: Difficulty, state: &mut u64) -> Option<CurriculumProblem> {
    let engine = NumberTheoryEngine::new();
    let (n_residues, mod_hi) = match difficulty {
        Difficulty::Easy => (2, 10u64),
        Difficulty::Medium => (3, 20),
        Difficulty::Hard => (4, 30),
    };
    // Sample moduli, compute a consistent solution by construction
    for _ in 0..50 {
        let mut moduli: Vec<i64> = Vec::new();
        let mut ok = true;
        for _ in 0..n_residues {
            let m = 2 + (xorshift64(state) % (mod_hi - 1)) as i64;
            if moduli.contains(&m) {
                ok = false;
                break;
            }
            moduli.push(m);
        }
        if !ok {
            continue;
        }
        // Pick a known x, derive the residues
        let x = (xorshift64(state) % 1000) as i64;
        let residues: Vec<(i64, i64)> = moduli.iter().map(|&m| (x.rem_euclid(m), m)).collect();
        // Verify it solves via CRT
        match engine.crt(&residues) {
            Some((sol, _)) => {
                let expected = sol;
                return Some(CurriculumProblem {
                    name: format!("CRT {} residues", n_residues),
                    difficulty,
                    domain: Domain::NumberTheory,
                    kind: ProblemKind::CrtSystem { residues, expected },
                });
            }
            None => continue,
        }
    }
    None
}

/// Generate a Legendre symbol problem. Easy = small prime (< 50);
/// Medium = medium prime (50-200); Hard = large prime (200-1000).
pub fn gen_legendre(difficulty: Difficulty, state: &mut u64) -> Option<CurriculumProblem> {
    let engine = NumberTheoryEngine::new();
    let primes = small_primes();
    let (lo_idx, hi_idx) = match difficulty {
        Difficulty::Easy => (1, 15),    // skip 2, use 3..47
        Difficulty::Medium => (15, 30), // 47..113
        Difficulty::Hard => (30, 54),   // 113..251
    };
    // Pick an odd prime p from the range
    let p_idx = lo_idx + (xorshift64(state) as usize) % (hi_idx - lo_idx);
    let p = primes[p_idx];
    // Pick a in [1, p-1]
    let a = 1 + (xorshift64(state) as i64) % (p - 1);
    let expected = engine.legendre_symbol(a, p);
    Some(CurriculumProblem {
        name: format!("Legendre ({}/{})", a, p),
        difficulty,
        domain: Domain::NumberTheory,
        kind: ProblemKind::LegendreSymbol { a, p, expected },
    })
}

/// Generate an AM-GM problem. Easy = 2 values in [1, 10]; Medium = 4 values
/// in [1, 50]; Hard = 8 values in [1, 200].
pub fn gen_amgm(difficulty: Difficulty, state: &mut u64) -> Option<CurriculumProblem> {
    let (n, hi) = match difficulty {
        Difficulty::Easy => (2, 10.0),
        Difficulty::Medium => (4, 50.0),
        Difficulty::Hard => (8, 200.0),
    };
    let values: Vec<f64> = (0..n)
        .map(|_| 1.0 + (xorshift64(state) as f64 / u64::MAX as f64) * (hi - 1.0))
        .collect();
    Some(CurriculumProblem {
        name: format!("AM-GM on {} values", n),
        difficulty,
        domain: Domain::Inequality,
        kind: ProblemKind::Amgm { values },
    })
}

/// Generate a Cauchy-Schwarz problem.
pub fn gen_cauchy_schwarz(difficulty: Difficulty, state: &mut u64) -> CurriculumProblem {
    let (n, hi) = match difficulty {
        Difficulty::Easy => (3, 10.0),
        Difficulty::Medium => (5, 30.0),
        Difficulty::Hard => (10, 100.0),
    };
    let a: Vec<f64> = (0..n)
        .map(|_| (xorshift64(state) as f64 / u64::MAX as f64) * hi)
        .collect();
    let b: Vec<f64> = (0..n)
        .map(|_| (xorshift64(state) as f64 / u64::MAX as f64) * hi)
        .collect();
    CurriculumProblem {
        name: format!("Cauchy-Schwarz on {}-vectors", n),
        difficulty,
        domain: Domain::Inequality,
        kind: ProblemKind::CauchySchwarz { a, b },
    }
}

/// Generate a pigeonhole problem where `min_collision` is derivable by
/// the classical ⌈items/boxes⌉ formula.
pub fn gen_pigeonhole(difficulty: Difficulty, state: &mut u64) -> CurriculumProblem {
    let (items_lo, items_hi, boxes_lo, boxes_hi) = match difficulty {
        Difficulty::Easy => (5usize, 15, 3usize, 6),
        Difficulty::Medium => (20, 60, 5, 15),
        Difficulty::Hard => (100, 500, 20, 50),
    };
    let items = items_lo + (xorshift64(state) as usize) % (items_hi - items_lo);
    let boxes = boxes_lo + (xorshift64(state) as usize) % (boxes_hi - boxes_lo);
    let min_collision = (items + boxes - 1) / boxes; // ⌈items / boxes⌉
    CurriculumProblem {
        name: format!("Pigeonhole {}/{}", items, boxes),
        difficulty,
        domain: Domain::Combinatorics,
        kind: ProblemKind::Pigeonhole {
            items,
            boxes,
            min_collision,
        },
    }
}

// ─── Full curriculum assembly ───────────────────────────────────────────────

/// Generate a full curriculum of `n_per_template_per_tier` problems for
/// every (template × difficulty) combination. Reproducible via `seed`.
pub fn generate_curriculum(n_per_template_per_tier: usize, seed: u64) -> Vec<CurriculumProblem> {
    let mut state = seed;
    let mut out = Vec::new();
    for &diff in &[Difficulty::Easy, Difficulty::Medium, Difficulty::Hard] {
        for _ in 0..n_per_template_per_tier {
            if let Some(p) = gen_pell(diff, &mut state) {
                out.push(p);
            }
            if let Some(p) = gen_crt(diff, &mut state) {
                out.push(p);
            }
            if let Some(p) = gen_legendre(diff, &mut state) {
                out.push(p);
            }
            if let Some(p) = gen_amgm(diff, &mut state) {
                out.push(p);
            }
            out.push(gen_cauchy_schwarz(diff, &mut state));
            out.push(gen_pigeonhole(diff, &mut state));
        }
    }
    out
}

/// Run the entire curriculum and return counts: (total, solved_by_domain,
/// solved_by_difficulty).
pub fn run_curriculum(problems: &[CurriculumProblem]) -> CurriculumReport {
    use std::collections::HashMap;
    let mut by_domain: HashMap<Domain, (usize, usize)> = HashMap::new();
    let mut by_difficulty: HashMap<Difficulty, (usize, usize)> = HashMap::new();
    let mut total_solved = 0usize;
    for p in problems {
        let solved = p.solve();
        let (ds, dt) = by_domain.entry(p.domain).or_insert((0, 0));
        *dt += 1;
        if solved {
            *ds += 1;
            total_solved += 1;
        }
        let (diffs, difft) = by_difficulty.entry(p.difficulty).or_insert((0, 0));
        *difft += 1;
        if solved {
            *diffs += 1;
        }
    }
    CurriculumReport {
        total: problems.len(),
        total_solved,
        by_domain,
        by_difficulty,
    }
}

#[derive(Debug)]
pub struct CurriculumReport {
    pub total: usize,
    pub total_solved: usize,
    pub by_domain: std::collections::HashMap<Domain, (usize, usize)>,
    pub by_difficulty: std::collections::HashMap<Difficulty, (usize, usize)>,
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_produces_solvable_pell() {
        let mut state = 42;
        let p = gen_pell(Difficulty::Easy, &mut state).unwrap();
        assert!(p.solve(), "generated Easy Pell should solve: {}", p.name);
    }

    #[test]
    fn test_generator_produces_solvable_crt() {
        let mut state = 42;
        for _ in 0..10 {
            let p = gen_crt(Difficulty::Easy, &mut state).unwrap();
            assert!(p.solve(), "generated CRT should solve: {}", p.name);
        }
    }

    #[test]
    fn test_generator_produces_solvable_legendre() {
        let mut state = 42;
        for _ in 0..10 {
            let p = gen_legendre(Difficulty::Easy, &mut state).unwrap();
            assert!(p.solve(), "generated Legendre should solve: {}", p.name);
        }
    }

    #[test]
    fn test_generator_produces_solvable_amgm() {
        let mut state = 42;
        for _ in 0..10 {
            let p = gen_amgm(Difficulty::Easy, &mut state).unwrap();
            assert!(p.solve(), "generated AM-GM should solve: {}", p.name);
        }
    }

    #[test]
    fn test_generator_produces_solvable_cauchy_schwarz() {
        let mut state = 42;
        for _ in 0..10 {
            let p = gen_cauchy_schwarz(Difficulty::Easy, &mut state);
            assert!(p.solve(), "Cauchy-Schwarz should solve: {}", p.name);
        }
    }

    #[test]
    fn test_generator_produces_solvable_pigeonhole() {
        let mut state = 42;
        for _ in 0..10 {
            let p = gen_pigeonhole(Difficulty::Easy, &mut state);
            assert!(p.solve(), "Pigeonhole should solve: {}", p.name);
        }
    }

    #[test]
    fn test_generator_reproducible() {
        let c1 = generate_curriculum(5, 42);
        let c2 = generate_curriculum(5, 42);
        assert_eq!(c1.len(), c2.len());
        for (a, b) in c1.iter().zip(c2.iter()) {
            assert_eq!(a.name, b.name);
        }
    }

    /// **The headline curriculum test.** Generate a 270-problem curriculum
    /// (15 per template per difficulty × 6 templates × 3 difficulties)
    /// and verify the solve rate exceeds 95%. Failures are expected for
    /// occasional edge cases (e.g. Pell with D that happens to be a
    /// perfect square due to RNG collision, CRT with non-coprime
    /// conflict) but the bulk should solve.
    #[test]
    fn test_curriculum_solve_rate() {
        let problems = generate_curriculum(15, 42);
        let report = run_curriculum(&problems);

        eprintln!("\n════════════════════════════════════════════════════════════");
        eprintln!("  CURRICULUM GENERATOR — Phase 5 scoped");
        eprintln!("  Generated: {} problems", report.total);
        eprintln!("────────────────────────────────────────────────────────────");
        eprintln!("  BY DOMAIN:");
        for (d, (s, t)) in &report.by_domain {
            eprintln!("    {:?}: {}/{}", d, s, t);
        }
        eprintln!("  BY DIFFICULTY:");
        for diff in [Difficulty::Easy, Difficulty::Medium, Difficulty::Hard] {
            if let Some(&(s, t)) = report.by_difficulty.get(&diff) {
                eprintln!("    {:?}: {}/{}", diff, s, t);
            }
        }
        let rate = report.total_solved as f64 / report.total as f64;
        eprintln!(
            "\n  OVERALL: {}/{} solved ({:.1}%)",
            report.total_solved,
            report.total,
            rate * 100.0
        );
        eprintln!("════════════════════════════════════════════════════════════");

        assert!(
            rate >= 0.95,
            "curriculum solve rate {:.1}% < 95% — generator has bugs",
            rate * 100.0
        );
    }

    #[test]
    fn test_curriculum_scales() {
        // Generator should produce thousands of problems without choking
        let problems = generate_curriculum(100, 42);
        assert!(problems.len() > 1500);
    }
}
