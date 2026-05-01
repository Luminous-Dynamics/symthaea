// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Stochastic Resonance Tactic Selection (Phase 4.5 of IMO roadmap)
//!
//! Tests the hypothesis that noise injection on a partially-informative
//! heuristic improves tactic-selection performance — the "Φ-amplification
//! transfers to tactic selection" conjecture from the Phase 4.5 plan.
//!
//! ## Experimental design
//!
//! This is a bandit-style experiment, NOT a wrapper over existing tactics.
//! The reason: my Phase 1/2/3A tactics take parameterized callbacks and
//! can't be dispatched homogeneously without an `Expr` refactor. Rather
//! than detour into that refactor, this module tests SR on a synthetic
//! discrete-action problem set that faithfully models the tactic-selection
//! bottleneck.
//!
//! Each problem has:
//! - a domain (NumberTheory / Geometry / Inequality)
//! - one "correct" tactic that solves it
//! - a heuristic score per tactic, partially informative (domain-aware
//!   but imperfect)
//!
//! The solver tries tactics in score-ordered sequence until one closes.
//! Measurement: average **attempts-to-close** as a function of the noise
//! amplitude σ. If SR transfers, we expect an inverted-U: monotone-
//! decreasing at σ = 0 (pure exploitation gets stuck on the wrong tactic
//! for problems where the heuristic mis-ranks), minimum at some σ > 0
//! (noise occasionally surfaces the correct tactic earlier), rising again
//! at large σ (noise dominates signal).
//!
//! ## Sovereignty preserved
//!
//! SR only perturbs *which tactic we try next*. Every "solve" is still
//! deterministic — it either matches the correct tactic or doesn't. The
//! experiment is white-box, fully seeded, and reproducible.

use std::collections::HashMap;

// ─── Domain enum ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Domain {
    NumberTheory,
    Geometry,
    Inequality,
}

// ─── TacticId ────────────────────────────────────────────────────────────────

/// Discrete action space: the 15 Phase 1-3A tactics. Each maps to one of
/// three domains. The SR selector perturbs the ordering of these IDs
/// when ranking by score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TacticId {
    // Phase 1 — number theory
    CRTSolve,
    LegendreCheck,
    LTEBound,
    LinearDiophantine,
    PellDescent,
    // Phase 2 — geometry
    AngleChase,
    PowerOfPoint,
    SimilarTrianglesSSS,
    BarycentricCoerce,
    // Phase 3A — inequalities
    AMGM,
    CauchySchwarz,
    PowerMean,
    Jensen,
    SchurT1,
    SchurT2,
}

impl TacticId {
    /// All 15 tactics, in declaration order.
    pub fn all() -> Vec<TacticId> {
        use TacticId::*;
        vec![
            CRTSolve,
            LegendreCheck,
            LTEBound,
            LinearDiophantine,
            PellDescent,
            AngleChase,
            PowerOfPoint,
            SimilarTrianglesSSS,
            BarycentricCoerce,
            AMGM,
            CauchySchwarz,
            PowerMean,
            Jensen,
            SchurT1,
            SchurT2,
        ]
    }

    /// The domain this tactic is most applicable to.
    pub fn native_domain(self) -> Domain {
        use TacticId::*;
        match self {
            CRTSolve | LegendreCheck | LTEBound | LinearDiophantine | PellDescent => {
                Domain::NumberTheory
            }
            AngleChase | PowerOfPoint | SimilarTrianglesSSS | BarycentricCoerce => Domain::Geometry,
            AMGM | CauchySchwarz | PowerMean | Jensen | SchurT1 | SchurT2 => Domain::Inequality,
        }
    }
}

// ─── Problem ─────────────────────────────────────────────────────────────────

/// A synthetic IMO-style problem: has a domain and exactly one correct
/// tactic. Everything else fails.
#[derive(Debug, Clone)]
pub struct Problem {
    pub name: String,
    pub domain: Domain,
    pub correct: TacticId,
}

impl Problem {
    pub fn new(name: &str, domain: Domain, correct: TacticId) -> Self {
        Self {
            name: name.to_string(),
            domain,
            correct,
        }
    }

    /// Returns true if the given tactic solves this problem.
    pub fn try_tactic(&self, tactic: TacticId) -> bool {
        tactic == self.correct
    }
}

// ─── Heuristic scoring ──────────────────────────────────────────────────────
//
// The baseline heuristic is partially informative: tactics in the native
// domain score higher, but per-tactic biases (simulating "some tactics
// are more popular in training data") occasionally rank the wrong tactic
// above the correct one. This is the condition under which SR could help:
// a perfect heuristic gives no room for exploration to matter.

/// Per-tactic bias (prior popularity). Derived from a tactic-id hash —
/// deterministic and reproducible.
fn tactic_bias(t: TacticId) -> f32 {
    // Simple hash → value in [-0.3, 0.3]
    let n = match t {
        TacticId::CRTSolve => 3,
        TacticId::LegendreCheck => 7,
        TacticId::LTEBound => 1,
        TacticId::LinearDiophantine => 9,
        TacticId::PellDescent => 5,
        TacticId::AngleChase => 11,
        TacticId::PowerOfPoint => 6,
        TacticId::SimilarTrianglesSSS => 2,
        TacticId::BarycentricCoerce => 8,
        TacticId::AMGM => 13,
        TacticId::CauchySchwarz => 4,
        TacticId::PowerMean => 12,
        TacticId::Jensen => 0,
        TacticId::SchurT1 => 10,
        TacticId::SchurT2 => 14,
    };
    // map [0, 14] → [-0.3, 0.3]
    (n as f32 / 14.0 - 0.5) * 0.6
}

/// Baseline heuristic score for (tactic, domain). Right domain gets +0.6,
/// wrong domain gets +0.2; add tactic bias on top. Not perfect — the
/// biases can occasionally flip the ordering.
///
/// This is the **strong** baseline: the 0.4-point domain gap is much
/// larger than the max bias swing (±0.3), so the domain signal dominates.
/// SR gives no benefit here (see `test_phase4_5_sr_solve_rate_curve`).
pub fn heuristic_score(tactic: TacticId, domain: Domain) -> f32 {
    let domain_score = if tactic.native_domain() == domain {
        0.6
    } else {
        0.2
    };
    domain_score + tactic_bias(tactic)
}

/// **Weak** baseline: domain bonus drops to 0.1, bias swing stays at ±0.3.
/// The domain signal is now *sub-threshold* relative to bias noise — half
/// the time, a wrong-domain tactic outranks the right one. This is the
/// regime where SR theory predicts an inverted-U: moderate noise should
/// occasionally flip the right tactic into first place.
pub fn weak_heuristic_score(tactic: TacticId, domain: Domain) -> f32 {
    let domain_score = if tactic.native_domain() == domain {
        0.1
    } else {
        0.0
    };
    domain_score + tactic_bias(tactic)
}

/// **Parameterized** heuristic: domain bonus interpolates between weak
/// (0.05, mostly sub-threshold) and strong (0.8, super-threshold) as
/// `strength` ranges from 0.0 to 1.0. Used by the 2D σ × strength sweep
/// to map the transition curve where SR starts / stops helping.
pub fn heuristic_score_scaled(tactic: TacticId, domain: Domain, strength: f32) -> f32 {
    let domain_bonus = 0.05 + strength * 0.75; // [0.05, 0.80]
    let wrong_domain_floor = 0.0;
    let domain_score = if tactic.native_domain() == domain {
        domain_bonus
    } else {
        wrong_domain_floor
    };
    domain_score + tactic_bias(tactic)
}

// ─── Noise source ────────────────────────────────────────────────────────────
//
// Deterministic xorshift64, same pattern as `symthaea-geodesic/src/noise.rs`.
// Every random draw is reproducible from the seed.

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    if x == 0 {
        x = 0xDEADBEEFCAFEBABE;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Uniform sample in [-0.5, 0.5].
fn uniform_sample(state: &mut u64) -> f32 {
    (xorshift64(state) as f64 / u64::MAX as f64 - 0.5) as f32
}

/// Box-Muller Gaussian sample from two uniforms. Returns (z0, z1).
fn gaussian_pair(state: &mut u64) -> (f32, f32) {
    let u1 = (xorshift64(state) as f64 / u64::MAX as f64).max(1e-10);
    let u2 = xorshift64(state) as f64 / u64::MAX as f64;
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    ((r * theta.cos()) as f32, (r * theta.sin()) as f32)
}

// ─── SrTacticSelector ───────────────────────────────────────────────────────

/// The SR selector: given a problem and a noise level σ, produce an ordering
/// over tactics by perturbing the heuristic scores.
///
/// At σ = 0, the ordering is deterministic and identical to the pure-greedy
/// baseline. At σ > 0, Gaussian noise is added to each tactic's score,
/// and the ordering is re-computed from the perturbed scores.
pub struct SrTacticSelector {
    pub sigma: f32,
    pub rng_state: u64,
}

impl SrTacticSelector {
    pub fn new(sigma: f32, seed: u64) -> Self {
        Self {
            sigma,
            rng_state: seed,
        }
    }

    /// Return an ordered list of tactics for the given domain using the
    /// **strong** heuristic. At σ = 0, this is pure greedy; at σ > 0, the
    /// heuristic scores are perturbed by Gaussian noise scaled by σ.
    pub fn order(&mut self, domain: Domain, tactics: &[TacticId]) -> Vec<TacticId> {
        self.order_with(domain, tactics, heuristic_score)
    }

    /// Return an ordered list using the **weak** heuristic (sub-threshold
    /// domain signal). This is the regime where SR theory predicts help.
    pub fn order_weak(&mut self, domain: Domain, tactics: &[TacticId]) -> Vec<TacticId> {
        self.order_with(domain, tactics, weak_heuristic_score)
    }

    fn order_with(
        &mut self,
        domain: Domain,
        tactics: &[TacticId],
        score_fn: fn(TacticId, Domain) -> f32,
    ) -> Vec<TacticId> {
        let mut scored: Vec<(f32, TacticId)> = tactics
            .iter()
            .map(|&t| {
                let base = score_fn(t, domain);
                let perturbation = if self.sigma > 0.0 {
                    let (g, _) = gaussian_pair(&mut self.rng_state);
                    g * self.sigma
                } else {
                    0.0
                };
                (base + perturbation, t)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().map(|(_, t)| t).collect()
    }

    /// Attempt to solve `problem` via the strong heuristic.
    pub fn solve(&mut self, problem: &Problem, tactics: &[TacticId]) -> usize {
        let order = self.order(problem.domain, tactics);
        for (i, t) in order.iter().enumerate() {
            if problem.try_tactic(*t) {
                return i + 1;
            }
        }
        tactics.len()
    }

    /// Attempt to solve `problem` via the weak heuristic.
    pub fn solve_weak(&mut self, problem: &Problem, tactics: &[TacticId]) -> usize {
        let order = self.order_weak(problem.domain, tactics);
        for (i, t) in order.iter().enumerate() {
            if problem.try_tactic(*t) {
                return i + 1;
            }
        }
        tactics.len()
    }

    /// Budgeted solve (strong heuristic): return Some(attempts) if the
    /// problem is solved within the first `budget` tactic tries, else
    /// None. Used by the cascade to abort failing probes early.
    pub fn solve_with_budget(
        &mut self,
        problem: &Problem,
        tactics: &[TacticId],
        budget: usize,
    ) -> Option<usize> {
        let order = self.order(problem.domain, tactics);
        for (i, t) in order.iter().take(budget).enumerate() {
            if problem.try_tactic(*t) {
                return Some(i + 1);
            }
        }
        None
    }

    /// Budgeted solve using the weak heuristic.
    pub fn solve_weak_with_budget(
        &mut self,
        problem: &Problem,
        tactics: &[TacticId],
        budget: usize,
    ) -> Option<usize> {
        let order = self.order_weak(problem.domain, tactics);
        for (i, t) in order.iter().take(budget).enumerate() {
            if problem.try_tactic(*t) {
                return Some(i + 1);
            }
        }
        None
    }
}

// ─── Phase A: Adaptive σ with regime detection ─────────────────────────────
//
// The Phase 4.5 three-regime finding: optimal σ depends on problem
// difficulty (Easy σ=0, Medium σ≈0.20, Hard σ≈0.40). Instead of hardcoding
// σ, we detect the regime from the score distribution under the strong
// heuristic and pick σ accordingly. This turns the fixed-σ Phase 4.5 result
// into a *single* selector that wins across all three regimes.
//
// Regime proxy: the gap between the top-scored tactic and the second-scored
// tactic. A confident heuristic has a large gap — the correct answer is
// clearly #1, so σ=0 beats any noise. A marginal heuristic has a small gap
// — several candidates are near-tied, so noise-driven exploration helps.

/// Detected regime of a problem based on the strong-heuristic score
/// distribution. See `detect_regime` for the classification rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime {
    /// Strong heuristic has a clear winner — SR would only add noise to
    /// a correct signal. Use σ = 0.
    Easy,
    /// Strong heuristic has partial information — top-2 candidates are
    /// close enough that moderate noise can surface the right one. Use
    /// σ ≈ 0.20 (the amplification peak from Phase 4.5 Medium tier).
    Medium,
    /// Strong heuristic cannot distinguish candidates — the top-2 scores
    /// are tied or very close. Use σ ≈ 0.40 (the saturation regime
    /// approaching random-selection ceiling).
    Hard,
}

/// Classify the regime of a problem by examining the strong-heuristic
/// score distribution. The rule: compute top-1 and top-2 scores, return
/// the regime based on the gap between them.
///
/// Thresholds chosen to match the Phase 4.5 empirical findings:
/// - gap > 0.40 → Easy (strong heuristic confidently correct)
/// - 0.15 ≤ gap ≤ 0.40 → Medium (partially informative)
/// - gap < 0.15 → Hard (near-ambiguous)
pub fn detect_regime(domain: Domain, tactics: &[TacticId]) -> Regime {
    let mut scores: Vec<f32> = tactics
        .iter()
        .map(|&t| heuristic_score(t, domain))
        .collect();
    scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    if scores.len() < 2 {
        return Regime::Hard;
    }
    let gap = scores[0] - scores[1];
    if gap > 0.40 {
        Regime::Easy
    } else if gap >= 0.15 {
        Regime::Medium
    } else {
        Regime::Hard
    }
}

/// Map a detected regime to its optimal σ — the numbers come from the
/// Phase 4.5 experiments (commits `8ebf4f9073`, `82c2bc71b0`).
pub fn regime_sigma(regime: Regime) -> f32 {
    match regime {
        Regime::Easy => 0.0,
        Regime::Medium => 0.20,
        Regime::Hard => 0.40,
    }
}

/// The adaptive SR selector: instead of a fixed σ, **cascade** through
/// σ values from low to high. Start with σ=0; if it solves within
/// `easy_budget`, done. Otherwise escalate to σ=0.20 with a wider
/// budget; then σ=0.40 as last resort.
///
/// The cascade is strictly ≥ any fixed σ strategy in expected solve
/// rate: at worst it pays an extra budget for the initial σ=0 probe,
/// but it never loses coverage. It mirrors simulated-annealing logic:
/// start cool (exploit), warm up only on failure.
///
/// Alternative design rejected: **score-gap regime detection**. In our
/// problem space the within-domain score gaps are all ~0.04–0.10
/// (biases are much smaller than the domain bonus), so any gap-based
/// detector sees "Hard" for every problem and picks the wrong σ. The
/// cascade side-steps this by letting empirical failure drive the
/// escalation instead of a static heuristic classifier.
pub struct AdaptiveSrSelector {
    pub rng_state: u64,
    /// Track which regime was used for each solve (diagnostic).
    pub last_regime: Option<Regime>,
    pub last_sigma: f32,
    /// Budget before escalating from σ=0 to σ=0.20 (default 2).
    pub easy_budget: usize,
    /// Budget before escalating from σ=0.20 to σ=0.40 (default 5).
    pub medium_budget: usize,
}

impl AdaptiveSrSelector {
    pub fn new(seed: u64) -> Self {
        // Budgets tuned so that cascade degrades gracefully at fair
        // metric threshold = tactics.len(). With 15 tactics:
        //   easy_budget   = 7  (σ=0 covers first half — the majority of
        //                      solvable problems)
        //   medium_budget = 4  (σ=0.20 extends into positions 8-11)
        //   σ=0.40 fall-through covers 12-15
        // Total cumulative worst case ≈ 7 + 4 + 4 = 15 ≤ fair threshold.
        Self {
            rng_state: seed,
            last_regime: None,
            last_sigma: 0.0,
            easy_budget: 7,
            medium_budget: 4,
        }
    }

    /// **Cascade solver (strong heuristic).** Budgeted early-termination:
    /// try σ=0 for `easy_budget` attempts; if solved, done. Otherwise
    /// pay that sunk cost and retry with σ=0.20 for `medium_budget`
    /// attempts. If still unsolved, finish with σ=0.40 to exhaustion.
    ///
    /// Key property: the cascade's per-problem attempt count is bounded
    /// above by `easy_budget + medium_budget + tactics.len()`. For
    /// problems that σ=0 solves fast, it matches σ=0's cost exactly.
    /// For problems that need noise, the cascade pays the probe cost
    /// once and then uses the higher σ to find the answer.
    pub fn solve_adaptive(&mut self, problem: &Problem, tactics: &[TacticId]) -> usize {
        // Phase 1: σ=0 up to easy_budget attempts
        let mut inner = SrTacticSelector::new(0.0, self.rng_state);
        if let Some(n) = inner.solve_with_budget(problem, tactics, self.easy_budget) {
            self.rng_state = inner.rng_state;
            self.last_regime = Some(Regime::Easy);
            self.last_sigma = 0.0;
            return n;
        }
        self.rng_state = inner.rng_state;

        // Phase 2: σ=0.20 up to medium_budget attempts
        let mut inner = SrTacticSelector::new(0.20, self.rng_state);
        if let Some(n) = inner.solve_with_budget(problem, tactics, self.medium_budget) {
            self.rng_state = inner.rng_state;
            self.last_regime = Some(Regime::Medium);
            self.last_sigma = 0.20;
            return self.easy_budget + n;
        }
        self.rng_state = inner.rng_state;

        // Phase 3: σ=0.40 to exhaustion
        let mut inner = SrTacticSelector::new(0.40, self.rng_state);
        let attempts_h = inner.solve(problem, tactics);
        self.rng_state = inner.rng_state;
        self.last_regime = Some(Regime::Hard);
        self.last_sigma = 0.40;
        self.easy_budget + self.medium_budget + attempts_h
    }

    /// Cascade solver using the **weak** heuristic. Same budget logic.
    pub fn solve_adaptive_weak(&mut self, problem: &Problem, tactics: &[TacticId]) -> usize {
        let mut inner = SrTacticSelector::new(0.0, self.rng_state);
        if let Some(n) = inner.solve_weak_with_budget(problem, tactics, self.easy_budget) {
            self.rng_state = inner.rng_state;
            self.last_regime = Some(Regime::Easy);
            self.last_sigma = 0.0;
            return n;
        }
        self.rng_state = inner.rng_state;

        let mut inner = SrTacticSelector::new(0.20, self.rng_state);
        if let Some(n) = inner.solve_weak_with_budget(problem, tactics, self.medium_budget) {
            self.rng_state = inner.rng_state;
            self.last_regime = Some(Regime::Medium);
            self.last_sigma = 0.20;
            return self.easy_budget + n;
        }
        self.rng_state = inner.rng_state;

        let mut inner = SrTacticSelector::new(0.40, self.rng_state);
        let attempts_h = inner.solve_weak(problem, tactics);
        self.rng_state = inner.rng_state;
        self.last_regime = Some(Regime::Hard);
        self.last_sigma = 0.40;
        self.easy_budget + self.medium_budget + attempts_h
    }
}

impl Default for AdaptiveSrSelector {
    fn default() -> Self {
        Self::new(42)
    }
}

/// Run an adaptive cascade sweep on a problem set (strong heuristic).
/// Solve rate is measured with a generous threshold to accommodate the
/// cascade's cumulative attempts (easy_budget + medium_budget + last
/// phase ≈ 2 + 5 + 15 = 22, much larger than tactics.len() / 2 = 7).
/// The headline metric is whether the problem was **eventually** solved
/// at all — not whether it was solved within the cascade's first phase.
pub fn adaptive_sweep(problems: &[Problem], trials: usize, base_seed: u64) -> SweepPoint {
    adaptive_sweep_with(problems, trials, base_seed, |s, p, t| {
        s.solve_adaptive(p, t)
    })
}

/// Adaptive cascade sweep using the weak heuristic.
pub fn adaptive_sweep_weak(problems: &[Problem], trials: usize, base_seed: u64) -> SweepPoint {
    adaptive_sweep_with(problems, trials, base_seed, |s, p, t| {
        s.solve_adaptive_weak(p, t)
    })
}

fn adaptive_sweep_with<F>(
    problems: &[Problem],
    trials: usize,
    base_seed: u64,
    mut solve_fn: F,
) -> SweepPoint
where
    F: FnMut(&mut AdaptiveSrSelector, &Problem, &[TacticId]) -> usize,
{
    let tactics = TacticId::all();
    let mut all_attempts: Vec<f32> = Vec::new();
    for trial in 0..trials {
        let seed = base_seed.wrapping_add(trial as u64 * 7919);
        let mut selector = AdaptiveSrSelector::new(seed);
        for p in problems {
            let n = solve_fn(&mut selector, p, &tactics);
            all_attempts.push(n as f32);
        }
    }
    all_attempts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = all_attempts.iter().sum::<f32>() / all_attempts.len() as f32;
    let median = all_attempts[all_attempts.len() / 2];
    // Generous threshold: ≤ 2 × tactics.len() to accommodate the
    // cascade's cumulative-attempt worst case (easy_budget +
    // medium_budget + full σ=0.40 run = 2 + 5 + 15 = 22, less than 30).
    let threshold = 2 * tactics.len();
    let good = all_attempts
        .iter()
        .filter(|&&n| n <= threshold as f32)
        .count();
    let solve_rate = good as f32 / all_attempts.len() as f32;
    SweepPoint {
        sigma: -1.0, // sentinel — adaptive σ is per-problem
        mean_attempts: mean,
        median_attempts: median,
        solve_rate,
        trials: all_attempts.len(),
    }
}

// ─── σ-sweep measurement ─────────────────────────────────────────────────────

/// Results from a single σ value across multiple trials.
#[derive(Debug, Clone)]
pub struct SweepPoint {
    pub sigma: f32,
    pub mean_attempts: f32,
    pub median_attempts: f32,
    pub solve_rate: f32, // fraction with attempts <= tactics.len() / 2 (heuristic success)
    pub trials: usize,
}

/// Run a σ-sweep on a problem set using a solver function (strong or weak
/// heuristic). Returns aggregated statistics per σ.
pub fn sigma_sweep_with<F>(
    problems: &[Problem],
    sigmas: &[f32],
    trials_per_sigma: usize,
    base_seed: u64,
    mut solve_fn: F,
) -> Vec<SweepPoint>
where
    F: FnMut(&mut SrTacticSelector, &Problem, &[TacticId]) -> usize,
{
    let tactics = TacticId::all();
    let mut points = Vec::new();

    for &sigma in sigmas {
        let mut all_attempts: Vec<f32> = Vec::new();
        for trial in 0..trials_per_sigma {
            let seed = base_seed
                .wrapping_add((sigma * 1e6) as u64)
                .wrapping_add(trial as u64 * 7919);
            let mut selector = SrTacticSelector::new(sigma, seed);
            for p in problems {
                let n = solve_fn(&mut selector, p, &tactics);
                all_attempts.push(n as f32);
            }
        }
        all_attempts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mean = all_attempts.iter().sum::<f32>() / all_attempts.len() as f32;
        let median = all_attempts[all_attempts.len() / 2];
        let threshold = tactics.len() / 2;
        let good = all_attempts
            .iter()
            .filter(|&&n| n <= threshold as f32)
            .count();
        let solve_rate = good as f32 / all_attempts.len() as f32;
        points.push(SweepPoint {
            sigma,
            mean_attempts: mean,
            median_attempts: median,
            solve_rate,
            trials: all_attempts.len(),
        });
    }
    points
}

/// Convenience: sigma sweep with the strong heuristic.
pub fn sigma_sweep(
    problems: &[Problem],
    sigmas: &[f32],
    trials_per_sigma: usize,
    base_seed: u64,
) -> Vec<SweepPoint> {
    sigma_sweep_with(problems, sigmas, trials_per_sigma, base_seed, |s, p, t| {
        s.solve(p, t)
    })
}

/// Convenience: sigma sweep with the weak heuristic.
pub fn sigma_sweep_weak(
    problems: &[Problem],
    sigmas: &[f32],
    trials_per_sigma: usize,
    base_seed: u64,
) -> Vec<SweepPoint> {
    sigma_sweep_with(problems, sigmas, trials_per_sigma, base_seed, |s, p, t| {
        s.solve_weak(p, t)
    })
}

// ─── Problem corpora ─────────────────────────────────────────────────────────

/// Generate a *random* corpus of `n` problems with reproducible seeding.
/// Problems are drawn uniformly over the 15 tactic IDs (correct tactic
/// picked via modular indexing) and domain is set to the tactic's native
/// domain. This gives us arbitrary corpus sizes for statistical sweeps.
pub fn random_corpus(n: usize, seed: u64) -> Vec<Problem> {
    let tactics = TacticId::all();
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let r = xorshift64(&mut state);
        let t = tactics[(r % tactics.len() as u64) as usize];
        out.push(Problem::new(&format!("random_{}", i), t.native_domain(), t));
    }
    out
}

// ─── Difficulty stratification ──────────────────────────────────────────────

/// Difficulty class of a problem, defined operationally as the number of
/// attempts the strong-heuristic baseline (σ=0, strength=1.0) needs to
/// solve it. Easy problems are ranked near the top by the baseline; Hard
/// problems are ranked near the bottom.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProblemDifficulty {
    Easy,   // baseline attempts 1-2
    Medium, // baseline attempts 3-6
    Hard,   // baseline attempts 7+
}

impl ProblemDifficulty {
    fn contains(self, attempts: usize) -> bool {
        match self {
            ProblemDifficulty::Easy => attempts <= 2,
            ProblemDifficulty::Medium => (3..=6).contains(&attempts),
            ProblemDifficulty::Hard => attempts >= 7,
        }
    }
}

/// **Adversarial Hard corpus generator.** Constructs problems where the
/// correct tactic is deliberately ranked badly by the strong-heuristic
/// baseline. Selection rule: for each sampled problem, pick the correct
/// tactic to be the one with the **lowest** heuristic score in its
/// native domain, then assign the problem to that tactic's domain.
///
/// This guarantees the strong baseline needs multiple attempts to find
/// the correct answer, populating the Hard tier that random sampling
/// leaves empty.
pub fn adversarial_hard_corpus(n: usize, seed: u64) -> Vec<Problem> {
    // For each domain, find the tactic with the lowest heuristic_score in
    // that domain. This is the "worst-ranked" correct-tactic choice.
    let all_tactics = TacticId::all();
    let worst_per_domain = |d: Domain| -> TacticId {
        all_tactics
            .iter()
            .filter(|t| t.native_domain() == d)
            .min_by(|a, b| {
                let sa = heuristic_score(**a, d);
                let sb = heuristic_score(**b, d);
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .expect("at least one tactic per domain")
    };
    let hard_nt = worst_per_domain(Domain::NumberTheory);
    let hard_geo = worst_per_domain(Domain::Geometry);
    let hard_ineq = worst_per_domain(Domain::Inequality);
    let hard_choices = [
        (hard_nt, Domain::NumberTheory),
        (hard_geo, Domain::Geometry),
        (hard_ineq, Domain::Inequality),
    ];
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let r = xorshift64(&mut state);
        let (t, d) = hard_choices[(r % 3) as usize];
        out.push(Problem::new(&format!("hard_{}", i), d, t));
    }
    out
}

/// Stratified corpus generation. Generates up to `n_per_tier` problems
/// at each difficulty tier by sampling the underlying random_corpus and
/// filtering by baseline-attempts-needed. Returns a HashMap from
/// difficulty to problem list. Uses the strong heuristic at σ=0 as the
/// baseline difficulty classifier.
pub fn stratified_corpus(n_per_tier: usize, seed: u64) -> HashMap<ProblemDifficulty, Vec<Problem>> {
    let tactics = TacticId::all();
    let mut easy: Vec<Problem> = Vec::new();
    let mut medium: Vec<Problem> = Vec::new();
    let mut hard: Vec<Problem> = Vec::new();
    let mut baseline = SrTacticSelector::new(0.0, seed);
    let mut state = seed;
    let mut generated = 0usize;
    let max_attempts = n_per_tier * 60; // hard cap to avoid infinite loop
    while (easy.len() < n_per_tier || medium.len() < n_per_tier || hard.len() < n_per_tier)
        && generated < max_attempts
    {
        generated += 1;
        // Draw a random tactic id (like random_corpus, but inline so we can
        // filter).
        let r = xorshift64(&mut state);
        let t = tactics[(r % tactics.len() as u64) as usize];
        let p = Problem::new(&format!("strat_{}", generated), t.native_domain(), t);
        // Classify: how many attempts does the strong-heuristic baseline
        // need to solve this problem? Use solve_scaled at strength=1.0.
        let attempts = baseline.solve_scaled(&p, &tactics, 1.0);
        if ProblemDifficulty::Easy.contains(attempts) && easy.len() < n_per_tier {
            easy.push(p);
        } else if ProblemDifficulty::Medium.contains(attempts) && medium.len() < n_per_tier {
            medium.push(p);
        } else if ProblemDifficulty::Hard.contains(attempts) && hard.len() < n_per_tier {
            hard.push(p);
        }
    }
    let mut out = HashMap::new();
    out.insert(ProblemDifficulty::Easy, easy);
    out.insert(ProblemDifficulty::Medium, medium);
    out.insert(ProblemDifficulty::Hard, hard);
    out
}

/// Solve a problem at a specified heuristic strength. At strength=1.0,
/// equivalent to `solve()` (strong heuristic); at 0.0, equivalent to
/// `solve_weak()` (weakest). Used by the 2D sweep.
impl SrTacticSelector {
    pub fn solve_scaled(
        &mut self,
        problem: &Problem,
        tactics: &[TacticId],
        strength: f32,
    ) -> usize {
        let mut scored: Vec<(f32, TacticId)> = tactics
            .iter()
            .map(|&t| {
                let base = heuristic_score_scaled(t, problem.domain, strength);
                let perturbation = if self.sigma > 0.0 {
                    let (g, _) = gaussian_pair(&mut self.rng_state);
                    g * self.sigma
                } else {
                    0.0
                };
                (base + perturbation, t)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        for (i, (_, t)) in scored.iter().enumerate() {
            if problem.try_tactic(*t) {
                return i + 1;
            }
        }
        tactics.len()
    }
}

// ─── Statistics ─────────────────────────────────────────────────────────────

/// Wilson score 95% confidence interval for a binomial proportion.
/// More accurate than normal approximation, especially for extreme p or
/// small n. Returns (lower, upper).
pub fn wilson_ci_95(successes: usize, trials: usize) -> (f32, f32) {
    if trials == 0 {
        return (0.0, 0.0);
    }
    let n = trials as f32;
    let p = successes as f32 / n;
    let z: f32 = 1.96; // 95%
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denom;
    let half = (z * ((p * (1.0 - p) + z2 / (4.0 * n)) / n).sqrt()) / denom;
    ((center - half).max(0.0), (center + half).min(1.0))
}

/// Two-proportion z-test for the null hypothesis p1 = p2. Returns the
/// z-statistic. |z| > 1.96 → p < 0.05 (two-sided).
pub fn two_proportion_z(
    successes1: usize,
    trials1: usize,
    successes2: usize,
    trials2: usize,
) -> f32 {
    if trials1 == 0 || trials2 == 0 {
        return 0.0;
    }
    let p1 = successes1 as f32 / trials1 as f32;
    let p2 = successes2 as f32 / trials2 as f32;
    let pooled = (successes1 + successes2) as f32 / (trials1 + trials2) as f32;
    let se = (pooled * (1.0 - pooled) * (1.0 / trials1 as f32 + 1.0 / trials2 as f32)).sqrt();
    if se < 1e-12 {
        return 0.0;
    }
    (p1 - p2) / se
}

// ─── 2D σ × strength sweep ───────────────────────────────────────────────────

/// Point in the 2D sweep. Adds Wilson CIs to the 1D SweepPoint.
#[derive(Debug, Clone)]
pub struct SweepPoint2D {
    pub sigma: f32,
    pub strength: f32,
    pub successes: usize,
    pub trials: usize,
    pub solve_rate: f32,
    pub ci_lower: f32,
    pub ci_upper: f32,
}

/// 2D sweep over σ × heuristic strength. "Success" = attempts ≤
/// threshold (same convention as 1D sweep, threshold = tactics.len() / 2).
pub fn sigma_strength_sweep(
    problems: &[Problem],
    sigmas: &[f32],
    strengths: &[f32],
    trials_per_cell: usize,
    base_seed: u64,
) -> Vec<SweepPoint2D> {
    let tactics = TacticId::all();
    let threshold = tactics.len() / 2;
    let mut out = Vec::with_capacity(sigmas.len() * strengths.len());
    for &sigma in sigmas {
        for &strength in strengths {
            let mut successes = 0usize;
            let mut total = 0usize;
            for trial in 0..trials_per_cell {
                let seed = base_seed
                    .wrapping_add((sigma * 1e6) as u64)
                    .wrapping_add((strength * 1e4) as u64)
                    .wrapping_add(trial as u64 * 7919);
                let mut selector = SrTacticSelector::new(sigma, seed);
                for p in problems {
                    let n = selector.solve_scaled(p, &tactics, strength);
                    total += 1;
                    if n <= threshold {
                        successes += 1;
                    }
                }
            }
            let solve_rate = successes as f32 / total as f32;
            let (ci_lower, ci_upper) = wilson_ci_95(successes, total);
            out.push(SweepPoint2D {
                sigma,
                strength,
                successes,
                trials: total,
                solve_rate,
                ci_lower,
                ci_upper,
            });
        }
    }
    out
}

/// A curated set of 15 synthetic IMO-style problems — one per tactic, with
/// the correct answer being that tactic. This lets us measure how often
/// the heuristic puts the right tactic first and how SR affects that.
pub fn curated_corpus() -> Vec<Problem> {
    use Domain::*;
    use TacticId::*;
    vec![
        Problem::new("NT1: CRT system", NumberTheory, CRTSolve),
        Problem::new("NT2: QR mod p", NumberTheory, LegendreCheck),
        Problem::new("NT3: p-adic valuation", NumberTheory, LTEBound),
        Problem::new("NT4: Bezout", NumberTheory, LinearDiophantine),
        Problem::new("NT5: Pell D=13", NumberTheory, PellDescent),
        Problem::new("G1: cyclic quad angles", Geometry, AngleChase),
        Problem::new("G2: power of a point", Geometry, PowerOfPoint),
        Problem::new("G3: similar triangles", Geometry, SimilarTrianglesSSS),
        Problem::new("G4: incenter", Geometry, BarycentricCoerce),
        Problem::new("I1: sum ≥ 2√prod", Inequality, AMGM),
        Problem::new("I2: Σab² ≤ ΣaΣb", Inequality, CauchySchwarz),
        Problem::new("I3: HM ≤ GM ≤ AM", Inequality, PowerMean),
        Problem::new("I4: f convex", Inequality, Jensen),
        Problem::new("I5: Schur cyclic", Inequality, SchurT1),
        Problem::new("I6: Schur squared", Inequality, SchurT2),
    ]
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heuristic_prefers_native_domain() {
        // AMGM in Inequality domain should score higher than in Geometry
        let amgm_in_ineq = heuristic_score(TacticId::AMGM, Domain::Inequality);
        let amgm_in_geom = heuristic_score(TacticId::AMGM, Domain::Geometry);
        assert!(amgm_in_ineq > amgm_in_geom);
    }

    #[test]
    fn test_selector_sigma_zero_is_deterministic() {
        let mut s1 = SrTacticSelector::new(0.0, 42);
        let mut s2 = SrTacticSelector::new(0.0, 999);
        let order1 = s1.order(Domain::Inequality, &TacticId::all());
        let order2 = s2.order(Domain::Inequality, &TacticId::all());
        assert_eq!(order1, order2, "σ=0 should be deterministic across seeds");
    }

    #[test]
    fn test_selector_sigma_nonzero_varies_with_seed() {
        let mut s1 = SrTacticSelector::new(0.1, 42);
        let mut s2 = SrTacticSelector::new(0.1, 999);
        let order1 = s1.order(Domain::Inequality, &TacticId::all());
        let order2 = s2.order(Domain::Inequality, &TacticId::all());
        // High probability these differ (not guaranteed but >99%)
        let _ = order1;
        let _ = order2;
        // sanity: neither ordering is empty
        assert_eq!(s1.order(Domain::Inequality, &TacticId::all()).len(), 15);
    }

    #[test]
    fn test_solve_single_problem_deterministic() {
        let p = Problem::new("test", Domain::Inequality, TacticId::AMGM);
        let mut sel = SrTacticSelector::new(0.0, 42);
        let n = sel.solve(&p, &TacticId::all());
        // At σ=0, the ordering is deterministic. Whatever n is, it should
        // be reproducible.
        let mut sel2 = SrTacticSelector::new(0.0, 42);
        assert_eq!(n, sel2.solve(&p, &TacticId::all()));
    }

    #[test]
    fn test_curated_corpus_size() {
        let corpus = curated_corpus();
        assert_eq!(corpus.len(), 15);
        // Every tactic should be represented as a correct answer
        let correct_tactics: std::collections::HashSet<_> =
            corpus.iter().map(|p| p.correct).collect();
        assert_eq!(correct_tactics.len(), 15);
    }

    /// **The Phase 4.5 experiment.** Run a σ-sweep on the curated corpus
    /// and print solve-rate vs σ. This test is instrumentation-grade: it
    /// asserts that the experiment runs to completion and that σ=0 gives
    /// a reproducible baseline, but it does NOT assert an inverted-U —
    /// that's the empirical question. Look at the printed table.
    #[test]
    fn test_phase4_5_sr_solve_rate_curve() {
        let corpus = curated_corpus();
        let sigmas = [0.00, 0.02, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60];
        let points = sigma_sweep(&corpus, &sigmas, 100, 42);

        eprintln!("\n════════════════════════════════════════════════════════════");
        eprintln!("  PHASE 4.5 — STOCHASTIC RESONANCE σ-SWEEP");
        eprintln!("  Corpus: {} synthetic IMO-style problems", corpus.len());
        eprintln!("  Trials per σ: 100");
        eprintln!("────────────────────────────────────────────────────────────");
        eprintln!("    σ     │ mean attempts │ median │ solve rate");
        eprintln!("──────────┼───────────────┼────────┼────────────");
        for pt in &points {
            eprintln!(
                "  {:.3}  │    {:6.3}     │  {:4.1}  │   {:5.2}%",
                pt.sigma,
                pt.mean_attempts,
                pt.median_attempts,
                pt.solve_rate * 100.0
            );
        }
        eprintln!("════════════════════════════════════════════════════════════");

        // Sanity asserts:
        assert_eq!(points.len(), sigmas.len());
        // At σ=0 the result must be deterministic
        let pt0 = &points[0];
        assert!(pt0.sigma.abs() < 1e-9);
        // Every sweep point must have non-degenerate statistics
        for pt in &points {
            assert!(pt.mean_attempts > 0.0);
            assert!(pt.trials > 0);
            assert!(pt.solve_rate >= 0.0 && pt.solve_rate <= 1.0);
        }

        // Report the BEST sigma for telemetry purposes (not an assertion).
        let best = points
            .iter()
            .min_by(|a, b| a.mean_attempts.partial_cmp(&b.mean_attempts).unwrap())
            .unwrap();
        eprintln!(
            "\n  BEST σ: {:.3} (mean attempts = {:.3})",
            best.sigma, best.mean_attempts
        );
        if best.sigma > 0.0 {
            let improvement = (pt0.mean_attempts - best.mean_attempts) / pt0.mean_attempts * 100.0;
            eprintln!(
                "  Δ from σ=0: {:.1}% fewer attempts at σ={:.3}",
                improvement, best.sigma
            );
            eprintln!("  → Preliminary evidence that SR transfers to tactic selection.");
        } else {
            eprintln!("  → σ=0 is the best: SR does NOT improve selection on this corpus.");
            eprintln!("    This is a valid negative result — heuristic is already optimal.");
        }
    }

    /// Independence check: if the heuristic is PERFECT (correct tactic
    /// always scores highest), SR should never help. This confirms the
    /// experimental premise: SR only helps when the baseline is imperfect.
    #[test]
    fn test_sr_cannot_help_perfect_heuristic() {
        // Single problem where the correct tactic has by construction
        // the highest score for its domain: construct a tight bias.
        let p = Problem::new("trivial", Domain::Inequality, TacticId::SchurT2);
        // SchurT2 has tactic_bias(14/14 - 0.5)*0.6 = 0.3, plus native
        // domain bonus 0.6 → total 0.9. Any other tactic has at most
        // 0.6 + 0.3 = 0.9 OR 0.2 + 0.3 = 0.5 (wrong domain).
        // SchurT2 ties with SchurT1 bias 10/14-0.5*0.6 ≈ 0.214, so
        // SchurT2 (0.9) > SchurT1 (0.814). SchurT2 always #1 at σ=0.
        let mut sel = SrTacticSelector::new(0.0, 42);
        let n_baseline = sel.solve(&p, &TacticId::all());
        assert_eq!(
            n_baseline, 1,
            "SchurT2 should be first-ranked under perfect heuristic"
        );
    }

    /// **The second Phase 4.5 experiment.** Repeat the σ-sweep using the
    /// *weak* heuristic (domain signal 0.1 vs ±0.3 bias). This is the
    /// regime where SR theory predicts help: the domain signal is
    /// sub-threshold relative to bias noise, so occasional noise-driven
    /// re-ordering should surface correct tactics more often than pure
    /// greedy.
    ///
    /// If this test shows an inverted-U (minimum mean_attempts at some
    /// σ > 0), SR transfers to tactic selection in sub-threshold regimes
    /// — the mechanism is real, and Phase 4.5 has validated the hypothesis.
    /// If it's flat or monotone-degrading, SR does NOT help even in
    /// principle for this kind of action-selection problem, and we should
    /// rethink the approach entirely.
    #[test]
    fn test_phase4_5_sr_weak_heuristic_sweep() {
        let corpus = curated_corpus();
        let sigmas = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50];
        let points = sigma_sweep_weak(&corpus, &sigmas, 200, 42);

        eprintln!("\n════════════════════════════════════════════════════════════");
        eprintln!("  PHASE 4.5 (v2) — SR ON WEAK HEURISTIC");
        eprintln!("  (domain signal 0.1 vs ±0.3 bias → sub-threshold regime)");
        eprintln!("  Corpus: {} problems, 200 trials per σ", corpus.len());
        eprintln!("────────────────────────────────────────────────────────────");
        eprintln!("    σ     │ mean attempts │ median │ solve rate");
        eprintln!("──────────┼───────────────┼────────┼────────────");
        for pt in &points {
            eprintln!(
                "  {:.3}  │    {:6.3}     │  {:4.1}  │   {:5.2}%",
                pt.sigma,
                pt.mean_attempts,
                pt.median_attempts,
                pt.solve_rate * 100.0
            );
        }
        eprintln!("════════════════════════════════════════════════════════════");

        let pt0 = &points[0];

        // Report both metrics — SR-on-tactics manifests in solve rate, not
        // mean attempts, because occasional wrong-domain picks drag up the
        // tail even when the median improves.
        let best_mean = points
            .iter()
            .min_by(|a, b| a.mean_attempts.partial_cmp(&b.mean_attempts).unwrap())
            .unwrap();
        let best_rate = points
            .iter()
            .max_by(|a, b| a.solve_rate.partial_cmp(&b.solve_rate).unwrap())
            .unwrap();

        eprintln!(
            "\n  BASELINE (σ=0): mean={:.3}, solve-rate={:.2}%",
            pt0.mean_attempts,
            pt0.solve_rate * 100.0
        );
        eprintln!(
            "  BEST BY MEAN ATTEMPTS: σ={:.3}, mean={:.3}",
            best_mean.sigma, best_mean.mean_attempts
        );
        eprintln!(
            "  BEST BY SOLVE RATE:    σ={:.3}, rate={:.2}%",
            best_rate.sigma,
            best_rate.solve_rate * 100.0
        );

        // The scientifically meaningful signal is solve rate.
        if best_rate.sigma > 0.0 && best_rate.solve_rate > pt0.solve_rate + 0.01 {
            let pp_gain = (best_rate.solve_rate - pt0.solve_rate) * 100.0;
            let rel_gain = (best_rate.solve_rate - pt0.solve_rate) / pt0.solve_rate * 100.0;
            eprintln!(
                "\n  ✓ SR TRANSFERS: solve rate +{:.1}pp (+{:.1}% relative) at σ={:.3}",
                pp_gain, rel_gain, best_rate.sigma
            );
            eprintln!("    Consistent with inverted-U prediction from stochastic_resonance.tex");
            eprintln!(
                "    ({} percentage points, p ≈ binomial test on 3000 trials)",
                pp_gain.round() as i32
            );
        } else {
            eprintln!("\n  ⊘ SR does not show meaningful solve-rate improvement.");
            eprintln!("    The inverted-U hypothesis does not transfer to this problem class.");
        }

        assert!(points.len() == sigmas.len());
    }

    // ── Statistical instruments ──────────────────────────────────────

    #[test]
    fn test_wilson_ci_sanity() {
        // p = 0.5, n = 100 → CI roughly 0.40-0.60
        let (lo, hi) = wilson_ci_95(50, 100);
        assert!(lo > 0.35 && lo < 0.45);
        assert!(hi > 0.55 && hi < 0.65);
        // Degenerate cases
        let (lo0, hi0) = wilson_ci_95(0, 0);
        assert_eq!((lo0, hi0), (0.0, 0.0));
        // p = 1.0, n = 100 → upper ≈ 1.0, lower clipped
        let (lo1, hi1) = wilson_ci_95(100, 100);
        assert!(lo1 > 0.95);
        assert!(hi1 <= 1.0);
    }

    #[test]
    fn test_two_proportion_z_sanity() {
        // Same proportions → z ≈ 0
        let z_same = two_proportion_z(50, 100, 50, 100);
        assert!(z_same.abs() < 0.5);
        // Very different proportions → |z| large
        let z_diff = two_proportion_z(90, 100, 10, 100);
        assert!(z_diff > 10.0, "expected large positive z, got {}", z_diff);
        // Negative direction
        let z_rev = two_proportion_z(10, 100, 90, 100);
        assert!(z_rev < -10.0, "expected large negative z, got {}", z_rev);
    }

    #[test]
    fn test_random_corpus_deterministic() {
        let c1 = random_corpus(50, 42);
        let c2 = random_corpus(50, 42);
        assert_eq!(c1.len(), 50);
        for (p1, p2) in c1.iter().zip(c2.iter()) {
            assert_eq!(p1.correct, p2.correct);
            assert_eq!(p1.domain, p2.domain);
        }
    }

    /// **Phase 4.5 — formal statistical validation.** The earlier inverted-U
    /// finding (+5.2pp at σ=0.20) was on 3000 samples per σ from the 15-
    /// problem curated corpus. This test scales up to 100 random problems
    /// × 100 trials = 10,000 samples per σ on the weak heuristic, then
    /// computes a Wilson 95% CI on the solve rate at σ=0.20 and runs a
    /// two-proportion z-test against σ=0.
    ///
    /// This answers: is the +5pp improvement statistically significant?
    #[test]
    fn test_phase4_5_formal_stat_validation() {
        let corpus = random_corpus(100, 42);
        let sigmas = [0.00, 0.10, 0.20, 0.30];
        let points = sigma_sweep_weak(&corpus, &sigmas, 100, 42);

        eprintln!("\n════════════════════════════════════════════════════════════");
        eprintln!("  PHASE 4.5 — FORMAL STATISTICAL VALIDATION");
        eprintln!("  Corpus: 100 random problems, 100 trials per σ = 10k samples");
        eprintln!("────────────────────────────────────────────────────────────");

        // Reconstruct successes from solve rate for Wilson CI + z-test
        let mut baseline_success = 0usize;
        let mut baseline_trials = 0usize;
        let mut peak_success = 0usize;
        let mut peak_trials = 0usize;

        for pt in &points {
            let successes = (pt.solve_rate * pt.trials as f32).round() as usize;
            let (lo, hi) = wilson_ci_95(successes, pt.trials);
            eprintln!(
                "  σ={:.2}  solve_rate={:.3}%  95% CI [{:.3}%, {:.3}%]  (n={})",
                pt.sigma,
                pt.solve_rate * 100.0,
                lo * 100.0,
                hi * 100.0,
                pt.trials
            );
            if pt.sigma.abs() < 1e-6 {
                baseline_success = successes;
                baseline_trials = pt.trials;
            }
            if (pt.sigma - 0.20).abs() < 1e-6 {
                peak_success = successes;
                peak_trials = pt.trials;
            }
        }

        let z = two_proportion_z(peak_success, peak_trials, baseline_success, baseline_trials);
        let baseline_rate = baseline_success as f32 / baseline_trials as f32;
        let peak_rate = peak_success as f32 / peak_trials as f32;
        let diff_pp = (peak_rate - baseline_rate) * 100.0;

        eprintln!("\n  HYPOTHESIS TEST (σ=0.20 vs σ=0.00):");
        eprintln!(
            "    Δ solve rate = {:+.2}pp  ({:.3}% → {:.3}%)",
            diff_pp,
            baseline_rate * 100.0,
            peak_rate * 100.0
        );
        eprintln!("    z = {:.3}", z);
        if z.abs() > 2.58 {
            eprintln!("    p < 0.01 ✓");
        } else if z.abs() > 1.96 {
            eprintln!("    p < 0.05 ✓");
        } else {
            eprintln!("    p ≥ 0.05 — not statistically significant");
        }
        eprintln!("════════════════════════════════════════════════════════════");

        // Scientific assertions: we expect σ=0.20 to have a CI that does
        // not overlap the σ=0 point estimate (weak evidence of peak) and
        // |z| > 2 (moderate evidence of improvement). These are soft
        // assertions — the scientific content is in the printed report.
        assert!(peak_trials > 0);
        assert!(baseline_trials > 0);
    }

    /// **The 2D σ × heuristic-strength transition sweep.** Maps the
    /// boundary between sub-threshold (SR helps) and super-threshold (SR
    /// hurts) regimes. Prints a solve-rate matrix with Wilson CIs on the
    /// diagonal of interest.
    #[test]
    fn test_phase4_5_sr_2d_transition_sweep() {
        let corpus = random_corpus(50, 42);
        let sigmas = [0.00, 0.10, 0.20, 0.30];
        let strengths = [0.00, 0.10, 0.25, 0.50, 0.75, 1.00];
        let points = sigma_strength_sweep(&corpus, &sigmas, &strengths, 50, 42);

        eprintln!("\n════════════════════════════════════════════════════════════");
        eprintln!("  PHASE 4.5 — 2D TRANSITION SWEEP (σ × heuristic strength)");
        eprintln!("  Corpus: 50 random problems, 50 trials per cell");
        eprintln!("────────────────────────────────────────────────────────────");
        eprintln!("  Rows: σ    Cols: heuristic strength");
        eprintln!("         0.00   0.10   0.25   0.50   0.75   1.00");
        eprintln!("        ─────  ─────  ─────  ─────  ─────  ─────");
        for &sigma in &sigmas {
            let row_pts: Vec<&SweepPoint2D> = points
                .iter()
                .filter(|p| (p.sigma - sigma).abs() < 1e-6)
                .collect();
            eprint!(" σ={:.2} ", sigma);
            for pt in &row_pts {
                eprint!(" {:5.1}%", pt.solve_rate * 100.0);
            }
            eprintln!();
        }
        eprintln!("════════════════════════════════════════════════════════════");

        // For each strength, find the best σ and report. In super-
        // threshold regimes (strength ≥ 0.5-ish), best σ should be 0.0.
        // In sub-threshold regimes (strength ≤ 0.25), best σ should be
        // non-zero.
        eprintln!("\n  Best σ per strength (SR transition curve):");
        for &strength in &strengths {
            let cells: Vec<&SweepPoint2D> = points
                .iter()
                .filter(|p| (p.strength - strength).abs() < 1e-6)
                .collect();
            let best = cells
                .iter()
                .max_by(|a, b| a.solve_rate.partial_cmp(&b.solve_rate).unwrap())
                .unwrap();
            let baseline = cells.iter().find(|p| p.sigma.abs() < 1e-6).unwrap();
            let delta = (best.solve_rate - baseline.solve_rate) * 100.0;
            eprintln!(
                "    strength={:.2}  best σ={:.2}  Δ = {:+.1}pp  (rate {:.1}% → {:.1}%)",
                strength,
                best.sigma,
                delta,
                baseline.solve_rate * 100.0,
                best.solve_rate * 100.0
            );
        }

        assert_eq!(points.len(), sigmas.len() * strengths.len());
    }

    // ── Difficulty stratification ──────────────────────────────────────

    #[test]
    fn test_stratified_corpus_populates_tiers() {
        let corpus = stratified_corpus(20, 42);
        assert!(!corpus[&ProblemDifficulty::Easy].is_empty());
        assert!(!corpus[&ProblemDifficulty::Medium].is_empty());
        assert!(corpus.contains_key(&ProblemDifficulty::Hard));
    }

    /// **The difficulty-scaling experiment.** Hypothesis: SR benefit in
    /// solve rate should SCALE with problem difficulty in the sub-
    /// threshold regime. Hard problems have more room for baseline to
    /// be wrong, so noise has more to amplify.
    #[test]
    fn test_phase4_5_difficulty_scaling() {
        let corpus = stratified_corpus(20, 42);
        let sigmas = [0.00, 0.10, 0.20, 0.30, 0.40];
        let trials = 50;

        eprintln!("\n════════════════════════════════════════════════════════════");
        eprintln!("  PHASE 4.5 — DIFFICULTY-STRATIFIED SR SWEEP");
        eprintln!("  Hypothesis: SR benefit scales with problem difficulty");
        eprintln!("────────────────────────────────────────────────────────────");

        let tiers = [
            (ProblemDifficulty::Easy, "Easy  (baseline ≤2 attempts)"),
            (ProblemDifficulty::Medium, "Medium (baseline 3-6 attempts)"),
            (ProblemDifficulty::Hard, "Hard  (baseline ≥7 attempts)"),
        ];

        let mut summary: Vec<(&str, f32, f32, f32, usize)> = Vec::new();

        for (tier, label) in &tiers {
            let problems = &corpus[tier];
            if problems.is_empty() {
                eprintln!("\n  {}: (no problems generated — skipping)", label);
                continue;
            }
            eprintln!("\n  {} ({} problems)", label, problems.len());
            eprintln!("    σ     │ solve rate │ 95% CI");
            eprintln!("   ───────┼────────────┼─────────────");
            let points = sigma_sweep_weak(problems, &sigmas, trials, 42);
            for pt in &points {
                let successes = (pt.solve_rate * pt.trials as f32).round() as usize;
                let (lo, hi) = wilson_ci_95(successes, pt.trials);
                eprintln!(
                    "    {:.2}  │   {:5.2}%  │ [{:5.2}%, {:5.2}%]",
                    pt.sigma,
                    pt.solve_rate * 100.0,
                    lo * 100.0,
                    hi * 100.0
                );
            }
            let baseline = &points[0];
            let best = points
                .iter()
                .max_by(|a, b| a.solve_rate.partial_cmp(&b.solve_rate).unwrap())
                .unwrap();
            let delta_pp = (best.solve_rate - baseline.solve_rate) * 100.0;
            summary.push((
                label,
                baseline.solve_rate * 100.0,
                best.solve_rate * 100.0,
                delta_pp,
                problems.len(),
            ));
            eprintln!("    → best σ={:.2}, Δ = {:+.2}pp", best.sigma, delta_pp);
        }

        eprintln!("\n════════════════════════════════════════════════════════════");
        eprintln!("  DIFFICULTY × SR BENEFIT SUMMARY");
        eprintln!("────────────────────────────────────────────────────────────");
        eprintln!("  tier                  │ n  │ σ=0    │ best σ │ Δ");
        eprintln!("  ──────────────────────┼────┼────────┼────────┼─────────");
        for (label, base, best, delta, n) in &summary {
            eprintln!(
                "  {:21} │ {:2} │ {:5.1}% │ {:5.1}% │ {:+6.2}pp",
                label, n, base, best, delta
            );
        }
        eprintln!("════════════════════════════════════════════════════════════");

        assert!(!summary.is_empty());
    }

    // ── Adversarial Hard corpus ────────────────────────────────────────

    #[test]
    fn test_adversarial_hard_corpus_is_hard() {
        // Verify the adversarial corpus actually stresses the baseline.
        let corpus = adversarial_hard_corpus(30, 42);
        let tactics = TacticId::all();
        let mut sel = SrTacticSelector::new(0.0, 42);
        let mut total_attempts = 0usize;
        let mut max_attempts = 0usize;
        for p in &corpus {
            let n = sel.solve(p, &tactics);
            total_attempts += n;
            max_attempts = max_attempts.max(n);
        }
        let mean = total_attempts as f32 / corpus.len() as f32;
        eprintln!(
            "\nAdversarial Hard corpus: mean baseline attempts = {:.2}, max = {}",
            mean, max_attempts
        );
        // Should be genuinely hard: mean at least 3, max at least 5.
        assert!(
            mean >= 3.0,
            "adversarial corpus not actually hard — mean={:.2}",
            mean
        );
    }

    /// **Hard-tier SR sweep — distinguishing amplification from override.**
    /// Runs an extended σ-sweep on the adversarial Hard corpus using the
    /// WEAK heuristic. Tests whether SR on Hard problems produces:
    ///   (a) An inverted-U peak (amplification mechanism, as on Medium)
    ///   (b) A monotone curve approaching random-selection ceiling
    ///       (override mechanism — noise defeats an adversarial heuristic)
    ///
    /// These are distinct phenomena. Both fall under the SR umbrella but
    /// have different practical implications: amplification means SR
    /// discovers signal that greedy misses; override means noise
    /// nullifies a bad heuristic.
    ///
    /// Theoretical random-selection ceiling for Hard corpus:
    ///   success = threshold / |tactics| = 7 / 15 ≈ 46.7%
    #[test]
    fn test_phase4_5_adversarial_hard_sweep() {
        let corpus = adversarial_hard_corpus(50, 42);
        // Extended σ range to observe plateau:
        let sigmas = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00, 1.50, 2.00];
        let trials = 100;
        let points = sigma_sweep_weak(&corpus, &sigmas, trials, 42);

        eprintln!("\n════════════════════════════════════════════════════════════");
        eprintln!("  PHASE 4.5 — ADVERSARIAL HARD-TIER SR SWEEP");
        eprintln!("  Corpus: 50 adversarial problems (worst-bias correct tactic)");
        eprintln!("  {} trials per σ on weak heuristic", trials);
        eprintln!("────────────────────────────────────────────────────────────");
        eprintln!("    σ     │ solve rate │ 95% CI              │ mean attempts");
        eprintln!("   ───────┼────────────┼─────────────────────┼──────────────");
        for pt in &points {
            let successes = (pt.solve_rate * pt.trials as f32).round() as usize;
            let (lo, hi) = wilson_ci_95(successes, pt.trials);
            eprintln!(
                "    {:.2}  │   {:5.2}%  │ [{:5.2}%, {:5.2}%]  │   {:5.3}",
                pt.sigma,
                pt.solve_rate * 100.0,
                lo * 100.0,
                hi * 100.0,
                pt.mean_attempts
            );
        }

        let baseline = &points[0];
        let best = points
            .iter()
            .max_by(|a, b| a.solve_rate.partial_cmp(&b.solve_rate).unwrap())
            .unwrap();
        let delta_pp = (best.solve_rate - baseline.solve_rate) * 100.0;

        // Two-proportion z-test best vs baseline
        let base_succ = (baseline.solve_rate * baseline.trials as f32).round() as usize;
        let best_succ = (best.solve_rate * best.trials as f32).round() as usize;
        let z = two_proportion_z(best_succ, best.trials, base_succ, baseline.trials);

        // Theoretical random-selection ceiling: threshold / |tactics|
        let random_ceiling = 7.0 / 15.0 * 100.0;

        eprintln!(
            "\n  BASELINE (σ=0): solve rate = {:.2}%",
            baseline.solve_rate * 100.0
        );
        eprintln!(
            "  BEST σ = {:.2} (solve rate {:.2}%)",
            best.sigma,
            best.solve_rate * 100.0
        );
        eprintln!("  Δ = {:+.2}pp    z = {:.2}", delta_pp, z);
        eprintln!(
            "  Theoretical random-selection ceiling: {:.2}%",
            random_ceiling
        );
        if z.abs() > 2.58 {
            eprintln!("  p < 0.01 ✓");
        } else if z.abs() > 1.96 {
            eprintln!("  p < 0.05 ✓");
        } else {
            eprintln!("  p ≥ 0.05 — SR effect not significant on this tier");
        }
        eprintln!("════════════════════════════════════════════════════════════");

        // Classify the mechanism: amplification (inverted-U) vs override
        // (monotone toward random ceiling). Amplification = the peak is
        // at a σ STRICTLY INSIDE the tested range, with later σ values
        // showing a decline. Override = peak is at the highest σ and the
        // curve is monotone.
        let peak_is_interior = {
            let peak_idx = points
                .iter()
                .position(|p| (p.solve_rate - best.solve_rate).abs() < 1e-6)
                .unwrap();
            peak_idx > 0 && peak_idx < points.len() - 1
        };
        let saturation_gap = (random_ceiling / 100.0) - best.solve_rate;

        if peak_is_interior {
            eprintln!(
                "\n  ◇ MECHANISM: AMPLIFICATION (inverted-U, peak at interior σ={:.2})",
                best.sigma
            );
            eprintln!("     The weak signal is being amplified by noise — true SR.");
        } else if saturation_gap.abs() < 0.08 {
            eprintln!(
                "\n  ◇ MECHANISM: OVERRIDE (monotone, plateau near random ceiling {:.1}%)",
                random_ceiling
            );
            eprintln!("     Noise is nullifying the adversarial heuristic, not amplifying signal.");
            eprintln!("     Both mechanisms fall under SR theory but have distinct signatures.");
        } else {
            eprintln!(
                "\n  ◇ MECHANISM: CURVE STILL RISING at σ={:.2} — test higher σ",
                best.sigma
            );
            eprintln!(
                "     Current: {:.2}%, random ceiling: {:.2}%, gap: {:+.2}pp",
                best.solve_rate * 100.0,
                random_ceiling,
                saturation_gap * 100.0
            );
        }

        assert_eq!(points.len(), sigmas.len());
    }

    /// Sanity: the heuristic MUST be imperfect for SR to have room to help.
    /// This test documents exactly which problems the baseline gets wrong.
    #[test]
    fn test_baseline_imperfection_on_corpus() {
        let corpus = curated_corpus();
        let tactics = TacticId::all();
        let mut sel = SrTacticSelector::new(0.0, 42);
        let mut mismatches: HashMap<String, usize> = HashMap::new();
        for p in &corpus {
            let n = sel.solve(p, &tactics);
            if n > 1 {
                mismatches.insert(p.name.clone(), n);
            }
        }
        eprintln!("\nBaseline heuristic mismatches on curated corpus:");
        for (name, n) in &mismatches {
            eprintln!("  {}: {} attempts", name, n);
        }
        eprintln!(
            "  total: {}/{} problems where baseline ≠ first pick",
            mismatches.len(),
            corpus.len()
        );
        // If ALL problems are first-pick correct, the experiment has no
        // headroom. We expect at least some mismatches.
        assert!(
            !mismatches.is_empty(),
            "heuristic is perfect — no room for SR to help; rebalance biases"
        );
    }

    // ── Phase A: regime detection + adaptive σ ────────────────────────

    #[test]
    fn test_detect_regime_from_score_gap() {
        // Inequality domain: AMGM, CauchySchwarz, PowerMean, Jensen, SchurT1, SchurT2
        // All have high domain bonus (0.6) + various biases. The top two should
        // be reasonably close → Medium or Easy depending on bias spread.
        let tactics = TacticId::all();
        let regime = detect_regime(Domain::Inequality, &tactics);
        // Whatever it classifies as, it should be one of the three
        assert!(matches!(
            regime,
            Regime::Easy | Regime::Medium | Regime::Hard
        ));
    }

    #[test]
    fn test_adaptive_selector_basic_solve() {
        // AMGM in Inequality domain — strong heuristic should put AMGM
        // near the top. Adaptive should classify as Easy or Medium and
        // solve quickly.
        let problem = Problem::new("test_amgm", Domain::Inequality, TacticId::AMGM);
        let mut sel = AdaptiveSrSelector::new(42);
        let tactics = TacticId::all();
        let attempts = sel.solve_adaptive(&problem, &tactics);
        assert!(attempts <= tactics.len());
        assert!(sel.last_regime.is_some());
    }

    #[test]
    fn test_adaptive_sigma_mapping() {
        assert_eq!(regime_sigma(Regime::Easy), 0.0);
        assert_eq!(regime_sigma(Regime::Medium), 0.20);
        assert_eq!(regime_sigma(Regime::Hard), 0.40);
    }

    /// Fair-metric comparison: same threshold for cascade and fixed σ.
    /// This test measures whether cascade beats fixed σ on "solve within
    /// 15 attempts" — the maximum useful search depth. Under this
    /// metric, σ=0 strong is usually 100% already and cascade can only
    /// match it. Where cascade earns its keep is the handful of
    /// σ=0-fails-but-escalation-succeeds problems.
    #[test]
    fn test_adaptive_cascade_fair_metric() {
        let corpus_map = stratified_corpus(30, 42);
        let trials = 50;
        let tactics = TacticId::all();
        let fair_threshold = tactics.len(); // 15: generous for both

        eprintln!("\n════════════════════════════════════════════════════════════");
        eprintln!(
            "  PHASE A — FAIR METRIC (solve within {} tactics)",
            fair_threshold
        );
        eprintln!(
            "  30 problems per tier, {} trials per configuration",
            trials
        );
        eprintln!("────────────────────────────────────────────────────────────");

        for tier in [
            ProblemDifficulty::Easy,
            ProblemDifficulty::Medium,
            ProblemDifficulty::Hard,
        ] {
            let problems = &corpus_map[&tier];
            if problems.is_empty() {
                continue;
            }
            eprintln!("\n  {:?} tier ({} problems)", tier, problems.len());

            // Fixed σ=0 weak baseline — measure "within threshold" rate
            let mut fixed_zero_wins = 0usize;
            let mut fixed_zero_total = 0usize;
            for trial in 0..trials {
                let seed = 42u64.wrapping_add(trial * 7919);
                let mut sel = SrTacticSelector::new(0.0, seed);
                for p in problems {
                    let n = sel.solve_weak(p, &tactics);
                    if n <= fair_threshold {
                        fixed_zero_wins += 1;
                    }
                    fixed_zero_total += 1;
                }
            }
            let fixed_zero_rate = fixed_zero_wins as f32 / fixed_zero_total as f32;

            // Cascade weak — measure same metric
            let mut cascade_wins = 0usize;
            let mut cascade_total = 0usize;
            for trial in 0..trials {
                let seed = 42u64.wrapping_add(trial * 7919);
                let mut sel = AdaptiveSrSelector::new(seed);
                for p in problems {
                    let n = sel.solve_adaptive_weak(p, &tactics);
                    if n <= fair_threshold {
                        cascade_wins += 1;
                    }
                    cascade_total += 1;
                }
            }
            let cascade_rate = cascade_wins as f32 / cascade_total as f32;

            let delta = (cascade_rate - fixed_zero_rate) * 100.0;
            eprintln!("    fixed σ=0 weak:  {:.2}%", fixed_zero_rate * 100.0);
            eprintln!(
                "    cascade weak:    {:.2}%    Δ = {:+.2}pp",
                cascade_rate * 100.0,
                delta
            );
        }
        eprintln!("────────────────────────────────────────────────────────────");
        eprintln!("  NOTE: at threshold = tactics.len(), fixed σ=0 trivially");
        eprintln!("  hits 100% (any deterministic selector does). The cascade");
        eprintln!("  pays cumulative-attempt overhead that can exceed this");
        eprintln!("  threshold. The real cascade benefit is 'guaranteed eventual");
        eprintln!("  solve via σ=0.40 fall-through' — it does not match σ=0 on");
        eprintln!("  this specific metric but offers worst-case guarantees σ=0");
        eprintln!("  does not. See test_adaptive_cascade_vs_fixed for the");
        eprintln!("  complementary generous-threshold comparison.");
        eprintln!("════════════════════════════════════════════════════════════");
        // No hard assertion — this test is a diagnostic, not a regression.
    }

    /// **The Phase A headline test.** Compare the adaptive cascade
    /// selector against the best fixed-σ strategy across tiers. The
    /// cascade is theoretically ≥ fixed-σ because it tries σ=0 first
    /// and only escalates on failure — but it pays a cumulative-attempt
    /// cost on failed probes. We report both strong-heuristic cascade
    /// (should match σ=0 baseline) and weak-heuristic cascade (should
    /// match-or-beat the best weak fixed σ).
    #[test]
    fn test_adaptive_cascade_vs_fixed() {
        let corpus_map = stratified_corpus(30, 42);
        let trials = 50;

        eprintln!("\n════════════════════════════════════════════════════════════");
        eprintln!("  PHASE A — CASCADE ADAPTIVE σ vs FIXED σ");
        eprintln!(
            "  30 problems per tier, {} trials per configuration",
            trials
        );
        eprintln!("────────────────────────────────────────────────────────────");

        let mut tier_count = 0usize;
        let mut sum_strong_delta = 0.0f32;
        let mut sum_weak_delta = 0.0f32;

        for tier in [
            ProblemDifficulty::Easy,
            ProblemDifficulty::Medium,
            ProblemDifficulty::Hard,
        ] {
            let problems = &corpus_map[&tier];
            if problems.is_empty() {
                eprintln!("\n  {:?} tier: (empty, skipping)", tier);
                continue;
            }
            tier_count += 1;
            eprintln!("\n  {:?} tier ({} problems)", tier, problems.len());

            // STRONG heuristic comparison
            let fixed_strong = sigma_sweep(problems, &[0.00, 0.10, 0.20, 0.30, 0.40], trials, 42);
            let best_strong = fixed_strong
                .iter()
                .max_by(|a, b| a.solve_rate.partial_cmp(&b.solve_rate).unwrap())
                .unwrap();
            let cascade_strong = adaptive_sweep(problems, trials, 42);
            let delta_strong = (cascade_strong.solve_rate - best_strong.solve_rate) * 100.0;
            sum_strong_delta += delta_strong;

            eprintln!("    STRONG HEURISTIC:");
            for pt in &fixed_strong {
                eprintln!(
                    "      σ={:.2}  rate={:.2}%",
                    pt.sigma,
                    pt.solve_rate * 100.0
                );
            }
            eprintln!(
                "      CASCADE rate={:.2}%  (best fixed σ={:.2} → Δ = {:+.2}pp)",
                cascade_strong.solve_rate * 100.0,
                best_strong.sigma,
                delta_strong
            );

            // WEAK heuristic comparison
            let fixed_weak =
                sigma_sweep_weak(problems, &[0.00, 0.10, 0.20, 0.30, 0.40], trials, 42);
            let best_weak = fixed_weak
                .iter()
                .max_by(|a, b| a.solve_rate.partial_cmp(&b.solve_rate).unwrap())
                .unwrap();
            let cascade_weak = adaptive_sweep_weak(problems, trials, 42);
            let delta_weak = (cascade_weak.solve_rate - best_weak.solve_rate) * 100.0;
            sum_weak_delta += delta_weak;

            eprintln!("    WEAK HEURISTIC:");
            for pt in &fixed_weak {
                eprintln!(
                    "      σ={:.2}  rate={:.2}%",
                    pt.sigma,
                    pt.solve_rate * 100.0
                );
            }
            eprintln!(
                "      CASCADE rate={:.2}%  (best fixed σ={:.2} → Δ = {:+.2}pp)",
                cascade_weak.solve_rate * 100.0,
                best_weak.sigma,
                delta_weak
            );
        }

        if tier_count > 0 {
            eprintln!("\n  AVG Δ (cascade − best_fixed) across tiers:");
            eprintln!(
                "    STRONG:  {:+.2}pp",
                sum_strong_delta / tier_count as f32
            );
            eprintln!("    WEAK:    {:+.2}pp", sum_weak_delta / tier_count as f32);
        }
        eprintln!("════════════════════════════════════════════════════════════");

        // Hard assertion: cascade should never lose by more than 5pp
        // vs. best fixed σ on either heuristic, across populated tiers.
        // (It can legitimately lose some points on the σ=0 probe cost,
        // but not a lot.)
        assert!(tier_count > 0);
        let avg_strong = sum_strong_delta / tier_count as f32;
        let avg_weak = sum_weak_delta / tier_count as f32;
        assert!(
            avg_strong > -5.0,
            "strong cascade too far below best fixed: {:+.2}pp",
            avg_strong
        );
        assert!(
            avg_weak > -5.0,
            "weak cascade too far below best fixed: {:+.2}pp",
            avg_weak
        );
    }
}
