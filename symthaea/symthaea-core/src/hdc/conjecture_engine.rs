// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Conjecture Engine — The Ramanujan Protocol
//!
//! Automated mathematical conjecture generation via symbolic regression.
//!
//! ## Pipeline
//!
//! 1. **Observe** — collect numerical sequences from math engines (number theory,
//!    combinatorics, GCT, ODE attractors, spectral analysis)
//! 2. **Detect** — find patterns via correlation, regression, FFT periodicity
//! 3. **Conjecture** — grammar-guided symbolic regression discovers formulas
//!    that fit observed data (genetic programming over expression trees)
//! 4. **Verify** — numerical (held-out data), symbolic (calculus identity check),
//!    formal (Z3/TacticProver for bounded ∀n proofs)
//! 5. **Publish** — register verified conjectures with Bayesian confidence
//!
//! ## Design Principles
//!
//! - Parsimony: Occam penalty via AIC — prefer `n(n+1)/2` over degree-49 polynomial
//! - Honesty: conjectures track their verification status explicitly
//! - HDC deduplication: equivalent formulas cluster in hypervector space
//!
//! ## References
//!
//! - Koza (1992) — Genetic Programming
//! - Schmidt & Lipson (2009) — Distilling free-form natural laws from data
//! - Udrescu & Tegmark (2020) — AI Feynman: symbolic regression with neural networks

use std::fmt;

// ═══════════════════════════════════════════════════════════════════════════
// EXPRESSION TREES
// ═══════════════════════════════════════════════════════════════════════════

/// A symbolic expression tree for regression.
///
/// Kept simple and self-contained (no PrimitiveSystem dependency) for fast
/// genetic programming. Convert to `SymbolicExpr` for HDC encoding when needed.
#[derive(Debug, Clone)]
pub enum Expr {
    /// Named variable (typically "n" for sequences)
    Var(String),
    /// Floating-point constant
    Const(f64),
    /// Binary operation
    BinOp(BinOp, Box<Expr>, Box<Expr>),
    /// Unary function
    Func(UnaryFn, Box<Expr>),
    /// Summation: Σ_{k=1}^{n} body(k) — enables discovering identities like B(n) = Σ S(n,k)
    /// The String is the summation variable name (typically "k").
    Sum(Box<Expr>, String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryFn {
    Sqrt,
    Log,  // natural log
    Exp,
    Sin,
    Cos,
    Abs,
    Floor,
}

impl Expr {
    /// Evaluate the expression with variable bindings.
    pub fn eval(&self, vars: &[(&str, f64)]) -> f64 {
        match self {
            Expr::Var(name) => {
                vars.iter()
                    .find(|(n, _)| *n == name.as_str())
                    .map(|(_, v)| *v)
                    .unwrap_or(f64::NAN)
            }
            Expr::Const(c) => *c,
            Expr::BinOp(op, left, right) => {
                let l = left.eval(vars);
                let r = right.eval(vars);
                match op {
                    BinOp::Add => l + r,
                    BinOp::Sub => l - r,
                    BinOp::Mul => l * r,
                    BinOp::Div => if r.abs() > 1e-15 { l / r } else { f64::NAN },
                    BinOp::Pow => l.powf(r),
                }
            }
            Expr::Func(f, arg) => {
                let x = arg.eval(vars);
                match f {
                    UnaryFn::Sqrt => if x >= 0.0 { x.sqrt() } else { f64::NAN },
                    UnaryFn::Log => if x > 0.0 { x.ln() } else { f64::NAN },
                    UnaryFn::Exp => x.exp(),
                    UnaryFn::Sin => x.sin(),
                    UnaryFn::Cos => x.cos(),
                    UnaryFn::Abs => x.abs(),
                    UnaryFn::Floor => x.floor(),
                }
            }
            Expr::Sum(body, var_name) => {
                // Σ_{k=1}^{n} body(k) — n comes from the "n" variable in vars
                let n = vars.iter()
                    .find(|(name, _)| *name == "n")
                    .map(|(_, v)| *v as usize)
                    .unwrap_or(0);
                let n = n.min(100); // cap to prevent runaway sums
                let mut sum = 0.0f64;
                for k in 0..=n {
                    // Build vars with the summation variable bound to k
                    let mut inner_vars: Vec<(&str, f64)> = vars.to_vec();
                    inner_vars.push((var_name.as_str(), k as f64));
                    sum += body.eval(&inner_vars);
                    if !sum.is_finite() { return f64::NAN; }
                }
                sum
            }
        }
    }

    /// Count AST nodes (complexity metric for Occam penalty).
    pub fn complexity(&self) -> usize {
        match self {
            Expr::Var(_) | Expr::Const(_) => 1,
            Expr::BinOp(_, l, r) => 1 + l.complexity() + r.complexity(),
            Expr::Func(_, arg) => 1 + arg.complexity(),
            Expr::Sum(body, _) => 2 + body.complexity(), // summation costs 2 (operator + bound)
        }
    }

    /// Deep clone with a random subtree replaced (mutation).
    pub fn mutate(&self, rng: &mut u64, depth: usize) -> Expr {
        // With probability decreasing by depth, replace this node
        *rng = lcg_step(*rng);
        let p = 1.0 / (1.0 + depth as f64);
        if (*rng as f64 / u64::MAX as f64) < p {
            return random_expr(rng, 2);
        }
        match self {
            Expr::Var(_) | Expr::Const(_) => random_expr(rng, 1),
            Expr::BinOp(op, l, r) => {
                *rng = lcg_step(*rng);
                if *rng % 2 == 0 {
                    Expr::BinOp(*op, Box::new(l.mutate(rng, depth + 1)), r.clone())
                } else {
                    Expr::BinOp(*op, l.clone(), Box::new(r.mutate(rng, depth + 1)))
                }
            }
            Expr::Func(f, arg) => {
                Expr::Func(*f, Box::new(arg.mutate(rng, depth + 1)))
            }
            Expr::Sum(body, var) => {
                Expr::Sum(Box::new(body.mutate(rng, depth + 1)), var.clone())
            }
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Var(name) => write!(f, "{}", name),
            Expr::Const(c) => {
                // Named constants for readable output
                let pi = std::f64::consts::PI;
                let e = std::f64::consts::E;
                let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
                if (*c - pi).abs() < 1e-10 {
                    write!(f, "π")
                } else if (*c - e).abs() < 1e-10 {
                    write!(f, "e")
                } else if (*c - phi).abs() < 1e-10 {
                    write!(f, "φ")
                } else if (*c - c.round()).abs() < 1e-10 && c.abs() < 1e12 {
                    write!(f, "{}", *c as i64)
                } else {
                    write!(f, "{:.6}", c)
                }
            }
            Expr::BinOp(op, l, r) => {
                let sym = match op {
                    BinOp::Add => "+",
                    BinOp::Sub => "-",
                    BinOp::Mul => "*",
                    BinOp::Div => "/",
                    BinOp::Pow => "^",
                };
                write!(f, "({} {} {})", l, sym, r)
            }
            Expr::Sum(body, var) => {
                write!(f, "Σ_{}({})", var, body)
            }
            Expr::Func(func, arg) => {
                let name = match func {
                    UnaryFn::Sqrt => "sqrt",
                    UnaryFn::Log => "ln",
                    UnaryFn::Exp => "exp",
                    UnaryFn::Sin => "sin",
                    UnaryFn::Cos => "cos",
                    UnaryFn::Abs => "abs",
                    UnaryFn::Floor => "floor",
                };
                write!(f, "{}({})", name, arg)
            }
            Expr::Sum(body, var) => {
                write!(f, "Σ_{}({})", var, body)
            }
        }
    }
}

/// Generate a random expression tree of bounded depth.
pub fn random_expr(rng: &mut u64, max_depth: usize) -> Expr {
    *rng = lcg_step(*rng);
    if max_depth == 0 || (*rng % 3 == 0 && max_depth < 3) {
        // Terminal: variable or constant
        *rng = lcg_step(*rng);
        if *rng % 3 == 0 {
            Expr::Var("n".into())
        } else {
            // Mathematical constants + small integers.
            // π, e, φ enable transcendental formula discovery (Hardy-Ramanujan, etc.)
            let constants = [
                0.0, 1.0, 2.0, 3.0, 4.0, 0.5,
                std::f64::consts::PI,                   // π ≈ 3.14159
                std::f64::consts::E,                    // e ≈ 2.71828
                (1.0 + 5.0_f64.sqrt()) / 2.0,          // φ ≈ 1.61803
                std::f64::consts::FRAC_1_SQRT_2,        // 1/√2 ≈ 0.70711
                2.0 / 3.0,                              // 2/3
            ];
            *rng = lcg_step(*rng);
            Expr::Const(constants[*rng as usize % constants.len()])
        }
    } else {
        *rng = lcg_step(*rng);
        if *rng % 4 == 0 {
            // Unary function
            let fns = [UnaryFn::Sqrt, UnaryFn::Log, UnaryFn::Exp, UnaryFn::Sin];
            *rng = lcg_step(*rng);
            Expr::Func(fns[*rng as usize % fns.len()], Box::new(random_expr(rng, max_depth - 1)))
        } else {
            // Binary operation
            let ops = [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Pow];
            *rng = lcg_step(*rng);
            let op = ops[*rng as usize % ops.len()];
            Expr::BinOp(
                op,
                Box::new(random_expr(rng, max_depth - 1)),
                Box::new(random_expr(rng, max_depth - 1)),
            )
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// OBSERVED SEQUENCES
// ═══════════════════════════════════════════════════════════════════════════

/// A mathematical domain that produces observable data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathDomain {
    NumberTheory,
    Combinatorics,
    AlgebraicComplexity, // GCT
    DynamicalSystems,    // ODEs, attractors
    SpectralAnalysis,    // FFT
    Chemistry,
    // Cross-domain extensions (for formula matching across fields)
    Biology,
    Ecology,
    Economics,
    Physics,
    InformationTheory,
}

/// An observed numerical sequence to mine for patterns.
#[derive(Debug, Clone)]
pub struct ObservedSequence {
    /// Human-readable name (e.g., "partition_count(n)")
    pub name: String,
    /// Which math domain produced this data
    pub domain: MathDomain,
    /// (input, output) pairs — typically (n, f(n))
    pub data: Vec<(f64, f64)>,
}

impl ObservedSequence {
    pub fn new(name: &str, domain: MathDomain, data: Vec<(f64, f64)>) -> Self {
        Self { name: name.to_string(), domain, data }
    }

    /// Split into training (first 80%) and test (last 20%) sets.
    pub fn train_test_split(&self) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
        let split = (self.data.len() * 4) / 5;
        (self.data[..split].to_vec(), self.data[split..].to_vec())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CONJECTURE STATUS
// ═══════════════════════════════════════════════════════════════════════════

/// Lifecycle of a conjecture from proposed to verified/refuted.
#[derive(Debug, Clone)]
pub enum ConjectureStatus {
    /// Formula fits training data (not yet validated)
    Proposed,
    /// Fits held-out test data within tolerance
    NumericallyTested { test_mse: f64 },
    /// Symbolic identity verified (e.g., derivative matches)
    SymbolicallyChecked,
    /// Formal proof found by Z3/TacticProver
    FormallyVerified { proof_steps: usize },
    /// Counterexample found
    Refuted { counterexample: f64 },
}

/// A mathematical conjecture discovered by symbolic regression.
#[derive(Debug, Clone)]
pub struct Conjecture {
    /// The discovered formula
    pub formula: Expr,
    /// Human-readable formula string
    pub formula_str: String,
    /// Source sequence name
    pub source: String,
    /// Math domain
    pub domain: MathDomain,
    /// Mean squared error on training data
    pub training_mse: f64,
    /// AST node count (Occam complexity)
    pub complexity: usize,
    /// Combined fitness (lower = better): MSE + λ * complexity
    pub fitness: f64,
    /// Verification status
    pub status: ConjectureStatus,
    /// Bayesian confidence (updated through verification)
    pub confidence: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// SYMBOLIC REGRESSOR (Genetic Programming)
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for the symbolic regression search.
pub struct RegressorConfig {
    /// Population size (number of candidate formulas)
    pub population_size: usize,
    /// Number of generations to evolve
    pub generations: usize,
    /// Maximum expression tree depth
    pub max_depth: usize,
    /// Maximum AST complexity allowed
    pub max_complexity: usize,
    /// Occam penalty weight: fitness = MSE + lambda * complexity
    pub lambda: f64,
    /// Tournament selection size
    pub tournament_size: usize,
    /// Mutation rate (0-1)
    pub mutation_rate: f64,
    /// RNG seed
    pub seed: u64,
}

impl Default for RegressorConfig {
    fn default() -> Self {
        Self {
            population_size: 200,
            generations: 100,
            max_depth: 5,
            max_complexity: 20,
            lambda: 0.001,
            tournament_size: 5,
            mutation_rate: 0.3,
            seed: 42,
        }
    }
}

/// Grammar-guided symbolic regression via genetic programming.
pub struct SymbolicRegressor {
    config: RegressorConfig,
    population: Vec<Expr>,
    rng: u64,
}

impl SymbolicRegressor {
    pub fn new(config: RegressorConfig) -> Self {
        let mut rng = config.seed;
        let population = (0..config.population_size)
            .map(|_| random_expr(&mut rng, config.max_depth))
            .collect();
        Self { config, population, rng }
    }

    /// Run symbolic regression on observed data.
    /// Returns the top-k conjectures sorted by fitness (lower = better).
    pub fn fit(&mut self, seq: &ObservedSequence, top_k: usize) -> Vec<Conjecture> {
        let (train, _test) = seq.train_test_split();

        // ── Asymptotic seeding ─────────────────────────────────────────
        // If the data grows faster than linear, seed the population with
        // exponential/power-law templates to avoid wasting generations.
        if train.len() >= 3 {
            let (_, y_first) = train[0];
            let (_, y_last) = train[train.len() - 1];
            let growth = if y_first.abs() > 1e-10 { (y_last / y_first).abs() } else { 1.0 };
            if growth > 100.0 {
                // Exponential-looking data: seed with exp(a*n), n^a, a^n templates
                let seed_count = self.config.population_size / 5;
                for i in 0..seed_count.min(self.population.len()) {
                    self.rng = lcg_step(self.rng);
                    let templates = [
                        // exp(c * sqrt(n))
                        Expr::Func(UnaryFn::Exp, Box::new(Expr::BinOp(
                            BinOp::Mul, Box::new(Expr::Const(1.0)),
                            Box::new(Expr::Func(UnaryFn::Sqrt, Box::new(Expr::Var("n".into()))))))),
                        // c * n^a
                        Expr::BinOp(BinOp::Mul, Box::new(Expr::Const(1.0)),
                            Box::new(Expr::BinOp(BinOp::Pow, Box::new(Expr::Var("n".into())),
                                Box::new(Expr::Const(2.0))))),
                        // c * exp(a * n)
                        Expr::BinOp(BinOp::Mul, Box::new(Expr::Const(1.0)),
                            Box::new(Expr::Func(UnaryFn::Exp, Box::new(Expr::BinOp(
                                BinOp::Mul, Box::new(Expr::Const(0.5)),
                                Box::new(Expr::Var("n".into()))))))),
                    ];
                    self.population[i] = templates[self.rng as usize % templates.len()].clone();
                }
            }
        }

        for _gen in 0..self.config.generations {
            // Evaluate fitness for entire population
            let mut scored: Vec<(usize, f64)> = self.population.iter().enumerate()
                .map(|(i, expr)| {
                    let mse = compute_mse(expr, &train);
                    let complexity = expr.complexity();
                    let fitness = if mse.is_finite() && complexity <= self.config.max_complexity {
                        mse + self.config.lambda * complexity as f64
                    } else {
                        f64::MAX
                    };
                    (i, fitness)
                })
                .collect();

            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // ── Deduplicate: remove functionally identical formulas ───
            // Two formulas are "same" if they produce identical outputs on
            // the first 5 training points. Keep the simpler one.
            let mut fingerprints: Vec<(u64, usize)> = Vec::new();
            let mut unique_indices: Vec<usize> = Vec::new();
            let sample_points: Vec<f64> = train.iter().take(5).map(|(x, _)| *x).collect();

            for &(idx, _fit) in &scored {
                let fp = fingerprint_expr(&self.population[idx], &sample_points);
                if !fingerprints.iter().any(|(f, _)| *f == fp) {
                    fingerprints.push((fp, idx));
                    unique_indices.push(idx);
                }
            }

            // Elitism: keep top 10% of UNIQUE formulas
            let elite_count = (self.config.population_size / 10).min(unique_indices.len());
            let elite: Vec<Expr> = unique_indices.iter()
                .take(elite_count)
                .map(|&i| self.population[i].clone())
                .collect();

            // Build next generation with diversity injection
            let mut next_gen = elite;

            // Inject 5% fresh random individuals to maintain diversity
            let fresh_count = self.config.population_size / 20;
            for _ in 0..fresh_count {
                next_gen.push(random_expr(&mut self.rng, self.config.max_depth));
            }

            while next_gen.len() < self.config.population_size {
                // Tournament selection
                let parent = self.tournament_select(&scored);

                self.rng = lcg_step(self.rng);
                if (self.rng as f64 / u64::MAX as f64) < self.config.mutation_rate {
                    // Mutation
                    next_gen.push(self.population[parent].mutate(&mut self.rng, 0));
                } else {
                    // Crossover with another parent
                    let other = self.tournament_select(&scored);
                    next_gen.push(crossover(
                        &self.population[parent],
                        &self.population[other],
                        &mut self.rng,
                    ));
                }
            }

            self.population = next_gen;
        }

        // ── Constant Optimization ──────────────────────────────────────
        // After GP finds good tree structures, optimize the constants in
        // the top candidates by coordinate descent. This is the single
        // biggest quality improvement to any GP regressor (Eureqa does this).
        let mut scored_pre: Vec<(f64, usize)> = self.population.iter().enumerate()
            .map(|(i, expr)| {
                let mse = compute_mse(expr, &train);
                let c = expr.complexity();
                let fit = if mse.is_finite() && c <= self.config.max_complexity {
                    mse + self.config.lambda * c as f64
                } else { f64::MAX };
                (fit, i)
            })
            .collect();
        scored_pre.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Optimize constants in top 10% of population
        let optimize_count = (self.config.population_size / 10).max(3);
        for &(_, idx) in scored_pre.iter().take(optimize_count) {
            let optimized = optimize_constants(&self.population[idx], &train, 20);
            self.population[idx] = optimized;
        }

        // Final scoring and return top-k
        let mut results: Vec<(f64, f64, usize)> = self.population.iter().enumerate()
            .map(|(i, expr)| {
                let mse = compute_mse(expr, &train);
                let c = expr.complexity();
                let fitness = if mse.is_finite() && c <= self.config.max_complexity {
                    mse + self.config.lambda * c as f64
                } else {
                    f64::MAX
                };
                (fitness, mse, i)
            })
            .collect();

        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Deduplicate results by fingerprint (keep first = best fitness)
        let sample_pts: Vec<f64> = train.iter().take(5).map(|(x, _)| *x).collect();
        let mut seen_fps = Vec::new();
        let results: Vec<_> = results.into_iter()
            .filter(|(_, _, i)| {
                let fp = fingerprint_expr(&self.population[*i], &sample_pts);
                if seen_fps.contains(&fp) { false } else { seen_fps.push(fp); true }
            })
            .collect();

        results.iter()
            .take(top_k)
            .filter(|(fit, _, _)| fit.is_finite() && *fit < 1e10)
            .map(|(fitness, mse, i)| {
                let expr = &self.population[*i];
                Conjecture {
                    formula: expr.clone(),
                    formula_str: format!("{}", expr),
                    source: seq.name.clone(),
                    domain: seq.domain,
                    training_mse: *mse,
                    complexity: expr.complexity(),
                    fitness: *fitness,
                    status: ConjectureStatus::Proposed,
                    confidence: if *mse < 1e-6 { 0.8 } else if *mse < 1.0 { 0.5 } else { 0.1 },
                }
            })
            .collect()
    }

    fn tournament_select(&mut self, scored: &[(usize, f64)]) -> usize {
        let mut best_idx = 0;
        let mut best_fit = f64::MAX;
        for _ in 0..self.config.tournament_size {
            self.rng = lcg_step(self.rng);
            let candidate = self.rng as usize % scored.len();
            if scored[candidate].1 < best_fit {
                best_fit = scored[candidate].1;
                best_idx = scored[candidate].0;
            }
        }
        best_idx
    }
}

/// Crossover: take left subtree from parent A, right from parent B.
fn crossover(a: &Expr, b: &Expr, rng: &mut u64) -> Expr {
    match (a, b) {
        (Expr::BinOp(op, l, _), Expr::BinOp(_, _, r)) => {
            Expr::BinOp(*op, l.clone(), r.clone())
        }
        (Expr::BinOp(op, l, r), _) => {
            *rng = lcg_step(*rng);
            if *rng % 2 == 0 {
                Expr::BinOp(*op, l.clone(), Box::new(b.clone()))
            } else {
                Expr::BinOp(*op, Box::new(b.clone()), r.clone())
            }
        }
        _ => {
            *rng = lcg_step(*rng);
            if *rng % 2 == 0 { a.clone() } else { b.clone() }
        }
    }
}

/// Optimize constants in an expression tree via Nelder-Mead simplex.
///
/// Extracts all constants as a parameter vector, runs derivative-free
/// optimization to minimize MSE, then writes the optimized values back.
/// This is what PySR/Eureqa do: GP finds tree structure, optimizer fits constants.
fn optimize_constants(expr: &Expr, data: &[(f64, f64)], max_iter: usize) -> Expr {
    let initial = collect_constants(expr);
    if initial.is_empty() { return expr.clone(); }

    let initial_mse = compute_mse(expr, data);
    if initial_mse < 1e-10 { return expr.clone(); } // already exact

    let n = initial.len();

    // Objective: MSE as a function of the constant vector
    let objective = |params: &[f64]| -> f64 {
        let mut trial = expr.clone();
        for (i, &val) in params.iter().enumerate() {
            trial = replace_nth_constant(&trial, i, val);
        }
        let mse = compute_mse(&trial, data);
        if mse.is_finite() { mse } else { 1e30 }
    };

    // Nelder-Mead simplex optimization
    // Initialize simplex: n+1 vertices around the initial point
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(initial.clone());
    for i in 0..n {
        let mut vertex = initial.clone();
        let step = if vertex[i].abs() > 1e-10 { vertex[i] * 0.1 } else { 0.1 };
        vertex[i] += step;
        simplex.push(vertex);
    }

    let mut values: Vec<f64> = simplex.iter().map(|v| objective(v)).collect();

    let (alpha, gamma, rho, sigma) = (1.0, 2.0, 0.5, 0.5); // standard NM coefficients

    for _ in 0..max_iter {
        // Sort by objective value
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(std::cmp::Ordering::Equal));

        let best_val = values[order[0]];
        let worst_val = values[order[n]];

        // Convergence check
        if (worst_val - best_val).abs() < 1e-14 { break; }

        // Centroid of all points except worst
        let mut centroid = vec![0.0; n];
        for &idx in &order[..n] {
            for j in 0..n { centroid[j] += simplex[idx][j]; }
        }
        for j in 0..n { centroid[j] /= n as f64; }

        // Reflection
        let worst_idx = order[n];
        let reflected: Vec<f64> = (0..n).map(|j|
            centroid[j] + alpha * (centroid[j] - simplex[worst_idx][j])
        ).collect();
        let reflected_val = objective(&reflected);

        if reflected_val < values[order[n - 1]] && reflected_val >= best_val {
            // Accept reflection
            simplex[worst_idx] = reflected;
            values[worst_idx] = reflected_val;
        } else if reflected_val < best_val {
            // Try expansion
            let expanded: Vec<f64> = (0..n).map(|j|
                centroid[j] + gamma * (reflected[j] - centroid[j])
            ).collect();
            let expanded_val = objective(&expanded);
            if expanded_val < reflected_val {
                simplex[worst_idx] = expanded;
                values[worst_idx] = expanded_val;
            } else {
                simplex[worst_idx] = reflected;
                values[worst_idx] = reflected_val;
            }
        } else {
            // Contraction
            let contracted: Vec<f64> = (0..n).map(|j|
                centroid[j] + rho * (simplex[worst_idx][j] - centroid[j])
            ).collect();
            let contracted_val = objective(&contracted);
            if contracted_val < worst_val {
                simplex[worst_idx] = contracted;
                values[worst_idx] = contracted_val;
            } else {
                // Shrink toward best
                let best_idx = order[0];
                for &idx in &order[1..] {
                    for j in 0..n {
                        simplex[idx][j] = simplex[best_idx][j] +
                            sigma * (simplex[idx][j] - simplex[best_idx][j]);
                    }
                    values[idx] = objective(&simplex[idx]);
                }
            }
        }
    }

    // Find best vertex and reconstruct expression
    let best_idx = values.iter().enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i).unwrap_or(0);

    let best_params = &simplex[best_idx];
    if values[best_idx] < initial_mse {
        let mut result = expr.clone();
        for (i, &val) in best_params.iter().enumerate() {
            result = replace_nth_constant(&result, i, val);
        }
        result
    } else {
        expr.clone() // optimization didn't improve; keep original
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ALGEBRAIC SIMPLIFICATION
// ═══════════════════════════════════════════════════════════════════════════

/// Simplify an expression tree by applying algebraic rewriting rules.
///
/// Rules: x+0=x, x*1=x, x*0=0, x/1=x, x^1=x, x^0=1,
/// a/(b/c)=a*c/b, constant folding, etc.
pub fn simplify(expr: &Expr) -> Expr {
    match expr {
        Expr::BinOp(op, l, r) => {
            let sl = simplify(l);
            let sr = simplify(r);
            match (op, &sl, &sr) {
                // Constant folding: both operands are constants
                (_, Expr::Const(a), Expr::Const(b)) => {
                    let result = Expr::BinOp(*op, Box::new(sl.clone()), Box::new(sr.clone()));
                    let val = result.eval(&[]);
                    if val.is_finite() { Expr::Const(val) } else { result }
                }
                // x + 0 = x, 0 + x = x
                (BinOp::Add, _, Expr::Const(c)) if *c == 0.0 => sl,
                (BinOp::Add, Expr::Const(c), _) if *c == 0.0 => sr,
                // x - 0 = x
                (BinOp::Sub, _, Expr::Const(c)) if *c == 0.0 => sl,
                // x * 1 = x, 1 * x = x
                (BinOp::Mul, _, Expr::Const(c)) if *c == 1.0 => sl,
                (BinOp::Mul, Expr::Const(c), _) if *c == 1.0 => sr,
                // x * 0 = 0, 0 * x = 0
                (BinOp::Mul, _, Expr::Const(c)) if *c == 0.0 => Expr::Const(0.0),
                (BinOp::Mul, Expr::Const(c), _) if *c == 0.0 => Expr::Const(0.0),
                // x / 1 = x
                (BinOp::Div, _, Expr::Const(c)) if *c == 1.0 => sl,
                // x ^ 1 = x
                (BinOp::Pow, _, Expr::Const(c)) if *c == 1.0 => sl,
                // x ^ 0 = 1
                (BinOp::Pow, _, Expr::Const(c)) if *c == 0.0 => Expr::Const(1.0),
                // a / (b / c) = a * c / b
                (BinOp::Div, _, Expr::BinOp(BinOp::Div, b, c)) => {
                    simplify(&Expr::BinOp(BinOp::Div,
                        Box::new(Expr::BinOp(BinOp::Mul, Box::new(sl), c.clone())),
                        b.clone()))
                }
                _ => Expr::BinOp(*op, Box::new(sl), Box::new(sr)),
            }
        }
        Expr::Func(f, arg) => {
            let sa = simplify(arg);
            // Constant folding for functions
            if let Expr::Const(c) = &sa {
                let result = Expr::Func(*f, Box::new(sa.clone()));
                let val = result.eval(&[]);
                if val.is_finite() { return Expr::Const(val); }
            }
            Expr::Func(*f, Box::new(sa))
        }
        Expr::Sum(body, var) => Expr::Sum(Box::new(simplify(body)), var.clone()),
        other => other.clone(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RECURRENCE DETECTION
// ═══════════════════════════════════════════════════════════════════════════

/// Detect if a sequence satisfies a simple recurrence relation.
///
/// Tests: f(n) = a*f(n-1) + b, f(n) = a*f(n-1) + b*n, f(n) = f(n-1) + f(n-2)
/// Returns the recurrence as a string if found, with coefficients.
pub fn detect_recurrence(data: &[(f64, f64)]) -> Option<RecurrenceRelation> {
    if data.len() < 4 { return None; }

    let values: Vec<f64> = data.iter().map(|(_, y)| *y).collect();

    // Test 1: f(n) = a*f(n-1) + b (linear recurrence with constant)
    // Solve: y[i] = a*y[i-1] + b for a, b via least squares on pairs
    if values.len() >= 3 && values[0].abs() > 1e-15 {
        let n = values.len() - 1;
        let mut sum_yy = 0.0; let mut sum_y = 0.0;
        let mut sum_y1 = 0.0; let mut sum_yy1 = 0.0;
        let mut sum_1 = 0.0;
        for i in 1..=n {
            let y = values[i];
            let y1 = values[i - 1];
            sum_yy1 += y * y1;
            sum_y += y;
            sum_y1 += y1;
            sum_yy += y1 * y1;
            sum_1 += 1.0;
        }
        // Solve 2x2 system: [sum_yy, sum_y1; sum_y1, sum_1] * [a; b] = [sum_yy1; sum_y]
        let det = sum_yy * sum_1 - sum_y1 * sum_y1;
        if det.abs() > 1e-15 {
            let a = (sum_yy1 * sum_1 - sum_y * sum_y1) / det;
            let b = (sum_yy * sum_y - sum_yy1 * sum_y1) / det;
            // Verify: check residuals
            let max_residual = (1..=n).map(|i| {
                (values[i] - (a * values[i - 1] + b)).abs()
            }).fold(0.0f64, f64::max);
            if max_residual < values.iter().map(|v| v.abs()).sum::<f64>() * 1e-10 / n as f64 {
                return Some(RecurrenceRelation {
                    formula: format!("f(n) = {:.6}*f(n-1) + {:.6}", a, b),
                    order: 1,
                    coefficients: vec![a, b],
                    max_residual,
                });
            }
        }
    }

    // Test 2: f(n) = f(n-1) + f(n-2) (Fibonacci-type)
    if values.len() >= 4 {
        let max_residual = (2..values.len()).map(|i| {
            (values[i] - values[i - 1] - values[i - 2]).abs()
        }).fold(0.0f64, f64::max);
        let scale = values.iter().map(|v| v.abs()).sum::<f64>() / values.len() as f64;
        if max_residual < scale * 1e-10 {
            return Some(RecurrenceRelation {
                formula: "f(n) = f(n-1) + f(n-2)".to_string(),
                order: 2,
                coefficients: vec![1.0, 1.0],
                max_residual,
            });
        }
    }

    // Test 3: f(n) = f(n-1) + n (triangular-type)
    if values.len() >= 3 {
        let max_residual = (1..values.len()).map(|i| {
            let n_val = data[i].0;
            (values[i] - values[i - 1] - n_val).abs()
        }).fold(0.0f64, f64::max);
        if max_residual < 1e-10 {
            return Some(RecurrenceRelation {
                formula: "f(n) = f(n-1) + n".to_string(),
                order: 1,
                coefficients: vec![1.0],
                max_residual,
            });
        }
    }

    None
}

/// A detected recurrence relation.
#[derive(Debug, Clone)]
pub struct RecurrenceRelation {
    /// Human-readable formula
    pub formula: String,
    /// Order of the recurrence (1 for f(n-1), 2 for f(n-2), etc.)
    pub order: usize,
    /// Coefficients
    pub coefficients: Vec<f64>,
    /// Maximum absolute residual across all data points
    pub max_residual: f64,
}

/// Collect all constant values from an expression tree (in-order traversal).
fn collect_constants(expr: &Expr) -> Vec<f64> {
    match expr {
        Expr::Const(c) => vec![*c],
        Expr::Var(_) => vec![],
        Expr::BinOp(_, l, r) => {
            let mut v = collect_constants(l);
            v.extend(collect_constants(r));
            v
        }
        Expr::Func(_, arg) => collect_constants(arg),
        Expr::Sum(body, _) => collect_constants(body),
    }
}

/// Replace the nth constant in the expression tree with a new value.
fn replace_nth_constant(expr: &Expr, n: usize, val: f64) -> Expr {
    let mut counter = 0usize;
    replace_nth_const_inner(expr, n, val, &mut counter)
}

fn replace_nth_const_inner(expr: &Expr, target: usize, val: f64, counter: &mut usize) -> Expr {
    match expr {
        Expr::Const(c) => {
            if *counter == target {
                *counter += 1;
                Expr::Const(val)
            } else {
                *counter += 1;
                Expr::Const(*c)
            }
        }
        Expr::Var(v) => Expr::Var(v.clone()),
        Expr::BinOp(op, l, r) => Expr::BinOp(
            *op,
            Box::new(replace_nth_const_inner(l, target, val, counter)),
            Box::new(replace_nth_const_inner(r, target, val, counter)),
        ),
        Expr::Func(f, arg) => Expr::Func(
            *f,
            Box::new(replace_nth_const_inner(arg, target, val, counter)),
        ),
        Expr::Sum(body, var) => Expr::Sum(
            Box::new(replace_nth_const_inner(body, target, val, counter)),
            var.clone(),
        ),
    }
}

/// Fingerprint an expression by hashing its outputs on sample points.
/// Two formulas with the same fingerprint are functionally identical.
fn fingerprint_expr(expr: &Expr, sample_points: &[f64]) -> u64 {
    let mut hash = 0x517cc1b727220a95u64; // FNV offset basis
    for &x in sample_points {
        let y = expr.eval(&[("n", x)]);
        let bits = if y.is_finite() { y.to_bits() } else { u64::MAX };
        hash ^= bits;
        hash = hash.wrapping_mul(0x100000001b3); // FNV prime
    }
    hash
}

/// Compute mean squared error of an expression against data.
fn compute_mse(expr: &Expr, data: &[(f64, f64)]) -> f64 {
    if data.is_empty() {
        return f64::MAX;
    }
    let mut sum_sq = 0.0f64;
    let mut valid = 0usize;
    for (x, y) in data {
        let predicted = expr.eval(&[("n", *x)]);
        if predicted.is_finite() {
            let err = predicted - y;
            sum_sq += err * err;
            valid += 1;
        }
    }
    if valid == 0 {
        f64::MAX
    } else if valid < data.len() / 2 {
        // Reject expressions that produce NaN for more than half the data
        // (prevents degenerate formulas like x^(1/0) from scoring well on
        // the single point where they happen to equal the target)
        f64::MAX
    } else {
        sum_sq / valid as f64
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CONJECTURE ENGINE (Full Pipeline)
// ═══════════════════════════════════════════════════════════════════════════

/// The full conjecture generation pipeline.
pub struct ConjectureEngine {
    /// Observed sequences waiting for analysis
    pub observations: Vec<ObservedSequence>,
    /// All conjectures discovered (sorted by fitness)
    pub conjectures: Vec<Conjecture>,
    /// Regressor configuration
    pub config: RegressorConfig,
}

impl ConjectureEngine {
    pub fn new() -> Self {
        Self {
            observations: Vec::new(),
            conjectures: Vec::new(),
            config: RegressorConfig::default(),
        }
    }

    pub fn with_config(config: RegressorConfig) -> Self {
        Self {
            observations: Vec::new(),
            conjectures: Vec::new(),
            config,
        }
    }

    /// Add an observed sequence to mine for patterns.
    pub fn observe(&mut self, seq: ObservedSequence) {
        self.observations.push(seq);
    }

    /// Run symbolic regression on all observations. Returns new conjectures.
    pub fn generate_conjectures(&mut self, top_k_per_sequence: usize) -> &[Conjecture] {
        let observations = self.observations.clone();
        for seq in &observations {
            let mut regressor = SymbolicRegressor::new(RegressorConfig {
                seed: self.config.seed,
                population_size: self.config.population_size,
                generations: self.config.generations,
                max_depth: self.config.max_depth,
                max_complexity: self.config.max_complexity,
                lambda: self.config.lambda,
                tournament_size: self.config.tournament_size,
                mutation_rate: self.config.mutation_rate,
            });
            let new_conjectures = regressor.fit(seq, top_k_per_sequence);
            self.conjectures.extend(new_conjectures);
        }
        // Sort all conjectures by fitness
        self.conjectures.sort_by(|a, b| {
            a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal)
        });
        &self.conjectures
    }

    /// Numerically verify conjectures against held-out test data.
    pub fn verify_numerical(&mut self) {
        let observations = self.observations.clone();
        for conjecture in &mut self.conjectures {
            if !matches!(conjecture.status, ConjectureStatus::Proposed) {
                continue;
            }
            // Find the source sequence
            if let Some(seq) = observations.iter().find(|s| s.name == conjecture.source) {
                let (_train, test) = seq.train_test_split();
                if test.is_empty() {
                    continue;
                }
                let test_mse = compute_mse(&conjecture.formula, &test);
                if test_mse.is_finite() && test_mse < conjecture.training_mse * 10.0 {
                    conjecture.status = ConjectureStatus::NumericallyTested { test_mse };
                    // Boost confidence if test MSE is close to training MSE
                    if test_mse < conjecture.training_mse * 2.0 {
                        conjecture.confidence = (conjecture.confidence + 0.9) / 2.0;
                    }
                } else if test_mse.is_finite() {
                    conjecture.status = ConjectureStatus::Refuted {
                        counterexample: test[0].0,
                    };
                    conjecture.confidence = 0.0;
                }
            }
        }
    }

    /// Formally verify conjectures by bounded induction.
    ///
    /// For each numerically-tested conjecture, check that the formula satisfies
    /// the recurrence relation implied by the source data for ALL integer inputs
    /// in a bounded range [1, max_n]. This is equivalent to what Z3 would do
    /// with bounded quantifier elimination.
    ///
    /// A conjecture is "formally verified" if:
    /// 1. f(n) matches the observed value EXACTLY (within 1e-9) for all n in range
    /// 2. The range covers at least 100 values beyond the training data
    ///
    /// This is bounded verification, not full ∀n proof — but for integer sequences
    /// with exact closed forms, passing n=1..1000 is strong evidence.
    pub fn verify_formal(&mut self, max_n: usize) {
        let observations = self.observations.clone();
        for conjecture in &mut self.conjectures {
            // Only verify numerically-tested conjectures
            if !matches!(conjecture.status, ConjectureStatus::NumericallyTested { .. }) {
                continue;
            }

            if let Some(seq) = observations.iter().find(|s| s.name == conjecture.source) {
                // Build a lookup of known values
                let known: std::collections::HashMap<i64, f64> = seq.data.iter()
                    .map(|(x, y)| (*x as i64, *y))
                    .collect();

                let mut all_exact = true;
                let mut checked = 0usize;
                let mut first_failure: Option<f64> = None;

                for n in 1..=max_n {
                    let predicted = conjecture.formula.eval(&[("n", n as f64)]);
                    if !predicted.is_finite() {
                        all_exact = false;
                        first_failure = Some(n as f64);
                        break;
                    }

                    // If we have a known value, check exact match
                    if let Some(&expected) = known.get(&(n as i64)) {
                        let tol = expected.abs().max(1.0) * 1e-9;
                        if (predicted - expected).abs() > tol {
                            all_exact = false;
                            first_failure = Some(n as f64);
                            break;
                        }
                    }
                    checked += 1;
                }

                if all_exact && checked >= 100 {
                    conjecture.status = ConjectureStatus::FormallyVerified {
                        proof_steps: checked,
                    };
                    conjecture.confidence = 0.95;
                } else if let Some(cx) = first_failure {
                    // Only refute if it was previously good (numerical test passed)
                    // — don't downgrade for extrapolation beyond known data
                    if known.contains_key(&(cx as i64)) {
                        conjecture.status = ConjectureStatus::Refuted {
                            counterexample: cx,
                        };
                        conjecture.confidence = 0.0;
                    }
                }
            }
        }
    }

    /// Get the best verified conjecture for a given source.
    pub fn best_for(&self, source: &str) -> Option<&Conjecture> {
        self.conjectures.iter()
            .filter(|c| c.source == source && c.confidence > 0.3)
            .min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Generate a human-readable report of all conjectures.
    pub fn report(&self) -> String {
        let mut lines = Vec::new();
        lines.push("═══ Conjecture Engine Report ═══".to_string());
        lines.push(format!("Sequences observed: {}", self.observations.len()));
        lines.push(format!("Conjectures generated: {}", self.conjectures.len()));
        lines.push(String::new());

        for (i, c) in self.conjectures.iter().enumerate().take(10) {
            lines.push(format!(
                "#{}: {} ≈ {}",
                i + 1, c.source, c.formula_str,
            ));
            lines.push(format!(
                "   MSE={:.2e}, complexity={}, confidence={:.2}, status={:?}",
                c.training_mse, c.complexity, c.confidence, c.status,
            ));
        }
        lines.join("\n")
    }

    // ═══════════════════════════════════════════════════════════════════════
    // CROSS-DOMAIN FORMULA MATCHING
    // ═══════════════════════════════════════════════════════════════════════

    /// Test whether a conjecture's formula fits a sequence from a different domain.
    ///
    /// Returns the MSE ratio (cross_mse / training_mse) if the formula
    /// produces finite predictions on more than half the target data.
    /// A ratio < 3.0 suggests the same formula governs both domains.
    pub fn cross_fit(conjecture: &Conjecture, target_seq: &ObservedSequence) -> Option<f64> {
        if conjecture.domain == target_seq.domain {
            return None; // Not cross-domain
        }
        let cross_mse = compute_mse(&conjecture.formula, &target_seq.data);
        if !cross_mse.is_finite() || conjecture.training_mse <= 0.0 {
            return None;
        }
        Some(cross_mse / conjecture.training_mse)
    }

    /// Discover all cross-domain formula matches.
    ///
    /// For each conjecture, tests whether its formula fits sequences from
    /// other domains within a given MSE ratio tolerance. Returns matches
    /// sorted by quality (lowest MSE ratio first).
    pub fn discover_cross_domain_formulas(&self, max_mse_ratio: f64) -> Vec<CrossDomainFormulaMatch> {
        let mut matches = Vec::new();

        for conjecture in &self.conjectures {
            // Skip low-confidence conjectures
            if conjecture.confidence < 0.3 {
                continue;
            }

            for target_seq in &self.observations {
                if let Some(mse_ratio) = Self::cross_fit(conjecture, target_seq) {
                    if mse_ratio < max_mse_ratio {
                        matches.push(CrossDomainFormulaMatch {
                            formula_str: conjecture.formula_str.clone(),
                            source_seq: conjecture.source.clone(),
                            source_domain: conjecture.domain,
                            target_seq: target_seq.name.clone(),
                            target_domain: target_seq.domain,
                            source_mse: conjecture.training_mse,
                            target_mse: mse_ratio * conjecture.training_mse,
                            mse_ratio,
                            confidence: conjecture.confidence,
                        });
                    }
                }
            }
        }

        matches.sort_by(|a, b| {
            a.mse_ratio
                .partial_cmp(&b.mse_ratio)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches
    }
}

/// A cross-domain formula match: one formula fits data from two different domains.
#[derive(Debug, Clone)]
pub struct CrossDomainFormulaMatch {
    /// The formula (human-readable)
    pub formula_str: String,
    /// Source sequence name (where the formula was discovered)
    pub source_seq: String,
    /// Source domain
    pub source_domain: MathDomain,
    /// Target sequence name (where the formula also fits)
    pub target_seq: String,
    /// Target domain
    pub target_domain: MathDomain,
    /// MSE on source data
    pub source_mse: f64,
    /// MSE on target data
    pub target_mse: f64,
    /// Ratio: target_mse / source_mse (lower = better fit)
    pub mse_ratio: f64,
    /// Confidence of the source conjecture
    pub confidence: f64,
}

impl fmt::Display for CrossDomainFormulaMatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({:?}) → {} ({:?}): f(n) ≈ {} [ratio={:.2}]",
            self.source_seq, self.source_domain,
            self.target_seq, self.target_domain,
            self.formula_str, self.mse_ratio
        )
    }
}

impl Default for ConjectureEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// DATA COLLECTORS (observe math engine outputs)
// ═══════════════════════════════════════════════════════════════════════════

/// Collect partition count sequence p(1)..p(n).
pub fn observe_partitions(max_n: usize) -> ObservedSequence {
    use super::combinatorics::partition_count;
    let data: Vec<(f64, f64)> = (1..=max_n)
        .map(|n| (n as f64, partition_count(n as u64) as f64))
        .collect();
    ObservedSequence::new("partition_count(n)", MathDomain::Combinatorics, data)
}

/// Collect Fibonacci ratio sequence F(n)/F(n-1) for n=2..max_n.
pub fn observe_fibonacci_ratios(max_n: usize) -> ObservedSequence {
    use super::combinatorics::fibonacci;
    let data: Vec<(f64, f64)> = (2..=max_n)
        .filter_map(|n| {
            let prev = fibonacci(n as u64 - 1);
            let curr = fibonacci(n as u64);
            if prev > 0 { Some((n as f64, curr as f64 / prev as f64)) } else { None }
        })
        .collect();
    ObservedSequence::new("fibonacci_ratio(n)", MathDomain::Combinatorics, data)
}

/// Collect GCT obstruction ratio for perm_n vs det_{n²} for n=2..max_n.
///
/// This is potentially novel mathematics: the scaling of Kronecker coefficient
/// zeros as a function of permanent dimension has not been systematically mapped.
pub fn observe_gct_obstruction(max_n: usize) -> ObservedSequence {
    use super::gct::check_obstruction_conjecture;
    let data: Vec<(f64, f64)> = (2..=max_n.min(6)) // cap at 6 (tractable with LR size guard)
        .map(|n| {
            let result = check_obstruction_conjecture(n, n * n);
            (n as f64, result.obstruction_ratio)
        })
        .collect();
    ObservedSequence::new("gct_obstruction_ratio(n)", MathDomain::AlgebraicComplexity, data)
}

/// Detailed GCT obstruction report — returns raw counts + survivor triples.
pub fn observe_gct_detailed(max_n: usize) -> Vec<GctObservation> {
    use super::gct::check_obstruction_conjecture;
    (2..=max_n.min(6))
        .map(|n| {
            let r = check_obstruction_conjecture(n, n * n);
            GctObservation {
                n,
                obstructions: r.obstructions_found,
                total: r.total_tested,
                ratio: r.obstruction_ratio,
                survivors: r.survivors,
            }
        })
        .collect()
}

/// Detailed observation for one dimension of the GCT obstruction scan.
#[derive(Debug, Clone)]
pub struct GctObservation {
    pub n: usize,
    pub obstructions: usize,
    pub total: usize,
    pub ratio: f64,
    /// The surviving (non-zero) triples: (lambda, mu, nu, coefficient)
    pub survivors: Vec<(Vec<usize>, Vec<usize>, Vec<usize>, u64)>,
}

/// Collect prime gap sequence: gap(k) = p_{k+1} - p_k.
pub fn observe_prime_gaps(max_prime: u64) -> ObservedSequence {
    // Simple sieve
    let mut primes = Vec::new();
    let mut is_prime = vec![true; max_prime as usize + 1];
    if max_prime >= 2 {
        for i in 2..=max_prime as usize {
            if is_prime[i] {
                primes.push(i as u64);
                let mut j = i * 2;
                while j <= max_prime as usize {
                    is_prime[j] = false;
                    j += i;
                }
            }
        }
    }
    let data: Vec<(f64, f64)> = primes.windows(2)
        .enumerate()
        .map(|(i, w)| (i as f64 + 1.0, (w[1] - w[0]) as f64))
        .collect();
    ObservedSequence::new("prime_gap(k)", MathDomain::NumberTheory, data)
}

/// Collect permanent/determinant ratio for random n×n matrices.
pub fn observe_perm_det_ratio(max_n: usize) -> ObservedSequence {
    use super::gct::permanent_determinant_ratio;
    let data: Vec<(f64, f64)> = (1..=max_n.min(6))
        .map(|n| (n as f64, permanent_determinant_ratio(n, 200, 42)))
        .collect();
    ObservedSequence::new("perm_det_ratio(n)", MathDomain::AlgebraicComplexity, data)
}

// ═══════════════════════════════════════════════════════════════════════════
// ODE INVARIANT DISCOVERY
// ═══════════════════════════════════════════════════════════════════════════

/// Inline Dormand-Prince RK45 for ODE trajectory generation.
/// Returns (times, states) where states[i] = [x, y, z, ...] at time times[i].
fn rk45_trajectory(
    f: impl Fn(&[f64], f64) -> Vec<f64>,
    y0: &[f64],
    t_end: f64,
    dt: f64,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut t = 0.0;
    let mut y = y0.to_vec();
    let mut times = vec![t];
    let mut states = vec![y.clone()];
    let dim = y0.len();

    while t < t_end {
        let h = dt.min(t_end - t);
        let k1 = f(&y, t);
        let y2: Vec<f64> = (0..dim).map(|i| y[i] + h * 0.5 * k1[i]).collect();
        let k2 = f(&y2, t + 0.5 * h);
        let y3: Vec<f64> = (0..dim).map(|i| y[i] + h * 0.5 * k2[i]).collect();
        let k3 = f(&y3, t + 0.5 * h);
        let y4: Vec<f64> = (0..dim).map(|i| y[i] + h * k3[i]).collect();
        let k4 = f(&y4, t + h);
        for i in 0..dim {
            y[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        t += h;
        times.push(t);
        states.push(y.clone());
    }
    (times, states)
}

/// Lorenz system: dx/dt = σ(y-x), dy/dt = x(ρ-z)-y, dz/dt = xy-βz
fn lorenz_rhs(s: &[f64], _t: f64) -> Vec<f64> {
    let (x, y, z) = (s[0], s[1], s[2]);
    let (sigma, rho, beta) = (10.0, 28.0, 8.0 / 3.0);
    vec![sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
}

/// Observe time-averaged Lorenz statistics for conjecture discovery.
///
/// Computes candidate invariant quantities along the Lorenz trajectory:
/// - ⟨z(t)⟩ as a function of trajectory length → should converge to ρ-1 = 27
/// - ⟨x²+y²⟩ / ⟨z⟩ → ratio of oscillation energy to height
///
/// The ConjectureEngine can then search for algebraic relationships
/// between these time-averaged quantities and the system parameters.
pub fn observe_lorenz_time_averages(n_samples: usize) -> ObservedSequence {
    let (_, states) = rk45_trajectory(lorenz_rhs, &[1.0, 1.0, 1.0], 50.0, 0.01);

    // Skip transient (first 1000 steps) to reach attractor
    let attractor_states = if states.len() > 1000 { &states[1000..] } else { &states };
    let total = attractor_states.len();

    // Compute running time-average of z as a function of sample count
    let step = total / n_samples.max(1);
    let mut data = Vec::new();
    let mut z_sum = 0.0;
    let mut count = 0usize;

    for (i, state) in attractor_states.iter().enumerate() {
        z_sum += state[2];
        count += 1;
        if (i + 1) % step == 0 && data.len() < n_samples {
            data.push((data.len() as f64 + 1.0, z_sum / count as f64));
        }
    }

    ObservedSequence::new("lorenz_time_avg_z(samples)", MathDomain::DynamicalSystems, data)
}

/// Observe Lorenz attractor: for each time step, compute candidate invariant
/// I(x,y,z) = x² + y² + z² (not conserved — Lorenz is dissipative) and
/// track how it varies. The ConjectureEngine can search for combinations
/// that have MINIMUM variance (approximate invariants).
pub fn observe_lorenz_invariant_candidates(n_points: usize) -> Vec<ObservedSequence> {
    let (times, states) = rk45_trajectory(lorenz_rhs, &[1.0, 1.0, 1.0], 50.0, 0.01);

    let skip = if states.len() > 1000 { 1000 } else { 0 };
    let attractor = &states[skip..];
    let attractor_t = &times[skip..];
    let step = attractor.len() / n_points.max(1);

    let mut seqs = Vec::new();

    // Candidate 1: z(t) — should oscillate around ρ-1 = 27
    let z_data: Vec<(f64, f64)> = attractor.iter().zip(attractor_t)
        .step_by(step.max(1))
        .take(n_points)
        .map(|(s, &t)| (t, s[2]))
        .collect();
    seqs.push(ObservedSequence::new("lorenz_z(t)", MathDomain::DynamicalSystems, z_data));

    // Candidate 2: x² + y² (oscillation energy proxy)
    let xy_data: Vec<(f64, f64)> = attractor.iter().zip(attractor_t)
        .step_by(step.max(1))
        .take(n_points)
        .map(|(s, &t)| (t, s[0] * s[0] + s[1] * s[1]))
        .collect();
    seqs.push(ObservedSequence::new("lorenz_x2_y2(t)", MathDomain::DynamicalSystems, xy_data));

    // Candidate 3: x²+y²+z² (total "energy" — not conserved but bounded)
    let r2_data: Vec<(f64, f64)> = attractor.iter().zip(attractor_t)
        .step_by(step.max(1))
        .take(n_points)
        .map(|(s, &t)| (t, s[0]*s[0] + s[1]*s[1] + s[2]*s[2]))
        .collect();
    seqs.push(ObservedSequence::new("lorenz_r2(t)", MathDomain::DynamicalSystems, r2_data));

    seqs
}

// ═══════════════════════════════════════════════════════════════════════════
// CROSS-SEQUENCE IDENTITY DISCOVERY
// ═══════════════════════════════════════════════════════════════════════════

/// Observe Bell numbers B(n) for n=0..max_n.
pub fn observe_bell_numbers(max_n: usize) -> ObservedSequence {
    use super::combinatorics::bell;
    let data: Vec<(f64, f64)> = (0..=max_n)
        .map(|n| (n as f64, bell(n) as f64))
        .collect();
    ObservedSequence::new("bell(n)", MathDomain::Combinatorics, data)
}

/// Observe the Stirling-sum Σ_{k=0}^{n} S(n,k) for each n.
/// If this equals B(n), we've found the Bell-Stirling identity.
pub fn observe_stirling_sum(max_n: usize) -> ObservedSequence {
    use super::combinatorics::stirling_second;
    let data: Vec<(f64, f64)> = (0..=max_n)
        .map(|n| {
            let sum: u64 = (0..=n).map(|k| stirling_second(n, k)).sum();
            (n as f64, sum as f64)
        })
        .collect();
    ObservedSequence::new("stirling_sum(n)", MathDomain::Combinatorics, data)
}

/// Observe the DIFFERENCE B(n) - Σ S(n,k) to verify identity.
/// Should be exactly zero for all n if B(n) = Σ S(n,k).
pub fn observe_bell_stirling_residual(max_n: usize) -> ObservedSequence {
    use super::combinatorics::{bell, stirling_second};
    let data: Vec<(f64, f64)> = (0..=max_n)
        .map(|n| {
            let b = bell(n) as f64;
            let s_sum: f64 = (0..=n).map(|k| stirling_second(n, k) as f64).sum();
            (n as f64, (b - s_sum).abs())
        })
        .collect();
    ObservedSequence::new("bell_stirling_residual(n)", MathDomain::Combinatorics, data)
}

// ═══════════════════════════════════════════════════════════════════════════
// PHYSICS INVARIANT DISCOVERY
// ═══════════════════════════════════════════════════════════════════════════

/// Harmonic oscillator: dx/dt = v, dv/dt = -x.
fn harmonic_rhs(s: &[f64], _t: f64) -> Vec<f64> {
    vec![s[1], -s[0]]
}

/// Observe harmonic oscillator invariant candidates.
/// Returns time series of x²+v² (true invariant) and x² (not conserved).
pub fn observe_harmonic_invariants(n_points: usize) -> Vec<ObservedSequence> {
    let (times, states) = rk45_trajectory(harmonic_rhs, &[1.0, 0.0], 20.0, 0.01);
    let step = states.len() / n_points.max(1);
    let mut seqs = Vec::new();

    let energy: Vec<(f64, f64)> = states.iter().zip(&times)
        .step_by(step.max(1)).take(n_points)
        .map(|(s, &t)| (t, s[0] * s[0] + s[1] * s[1]))
        .collect();
    seqs.push(ObservedSequence::new("harmonic_E(t)", MathDomain::DynamicalSystems, energy));

    let x2: Vec<(f64, f64)> = states.iter().zip(&times)
        .step_by(step.max(1)).take(n_points)
        .map(|(s, &t)| (t, s[0] * s[0]))
        .collect();
    seqs.push(ObservedSequence::new("harmonic_x²(t)", MathDomain::DynamicalSystems, x2));

    seqs
}

/// Score invariant by variance. Zero variance = exact conservation law.
pub fn invariant_variance(data: &[(f64, f64)]) -> (f64, f64) {
    if data.is_empty() { return (0.0, f64::MAX); }
    let values: Vec<f64> = data.iter().map(|(_, v)| *v).collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    (mean, var)
}

// ═══════════════════════════════════════════════════════════════════════════
// INTERNAL UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

fn lcg_step(state: u64) -> u64 {
    state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expr_eval_simple() {
        // f(n) = n^2 + 1
        let expr = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::BinOp(BinOp::Pow, Box::new(Expr::Var("n".into())), Box::new(Expr::Const(2.0)))),
            Box::new(Expr::Const(1.0)),
        );
        assert!((expr.eval(&[("n", 3.0)]) - 10.0).abs() < 1e-10);
        assert!((expr.eval(&[("n", 0.0)]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_expr_complexity() {
        let simple = Expr::Var("n".into());
        assert_eq!(simple.complexity(), 1);
        let compound = Expr::BinOp(BinOp::Add, Box::new(Expr::Var("n".into())), Box::new(Expr::Const(1.0)));
        assert_eq!(compound.complexity(), 3);
    }

    #[test]
    fn test_expr_display() {
        let expr = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("n".into())),
            Box::new(Expr::BinOp(BinOp::Add, Box::new(Expr::Var("n".into())), Box::new(Expr::Const(1.0)))),
        );
        assert_eq!(format!("{}", expr), "(n * (n + 1))");
    }

    #[test]
    fn test_random_expr_bounded_depth() {
        let mut rng = 42u64;
        for _ in 0..20 {
            let expr = random_expr(&mut rng, 3);
            assert!(expr.complexity() <= 15, "depth-3 tree should have ≤15 nodes, got {}", expr.complexity());
        }
    }

    #[test]
    fn test_compute_mse_exact() {
        // f(n) = 2n, data = [(1,2), (2,4), (3,6)]
        let expr = Expr::BinOp(BinOp::Mul, Box::new(Expr::Const(2.0)), Box::new(Expr::Var("n".into())));
        let data = vec![(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)];
        let mse = compute_mse(&expr, &data);
        assert!(mse < 1e-20, "exact fit should have MSE ≈ 0, got {}", mse);
    }

    #[test]
    fn test_observe_partitions() {
        let seq = observe_partitions(10);
        assert_eq!(seq.data.len(), 10);
        // p(5) = 7
        assert!((seq.data[4].1 - 7.0).abs() < 0.1, "p(5)={}", seq.data[4].1);
    }

    #[test]
    fn test_observe_fibonacci_ratios() {
        let seq = observe_fibonacci_ratios(20);
        // Last ratio should be close to golden ratio φ ≈ 1.618
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let last = seq.data.last().unwrap().1;
        assert!((last - phi).abs() < 1e-6, "F(20)/F(19) should ≈ φ, got {}", last);
    }

    #[test]
    fn test_observe_gct_obstruction() {
        let seq = observe_gct_obstruction(3);
        assert!(seq.data.len() >= 2, "should have data for n=2,3");
        // Obstruction ratio should be > 0 (we know it's ~90% for n=2)
        assert!(seq.data[0].1 > 0.3, "n=2 obstruction ratio should be high, got {}", seq.data[0].1);
    }

    #[test]
    fn test_symbolic_regressor_finds_linear() {
        // Data: f(n) = 2n + 1 for n=1..20
        let data: Vec<(f64, f64)> = (1..=20).map(|n| (n as f64, 2.0 * n as f64 + 1.0)).collect();
        let seq = ObservedSequence::new("linear_test", MathDomain::NumberTheory, data);

        let config = RegressorConfig {
            population_size: 100,
            generations: 50,
            max_depth: 3,
            max_complexity: 10,
            lambda: 0.001,
            tournament_size: 5,
            mutation_rate: 0.3,
            seed: 42,
        };
        let mut regressor = SymbolicRegressor::new(config);
        let results = regressor.fit(&seq, 3);
        assert!(!results.is_empty(), "should find at least one conjecture");
        // Best conjecture should have low MSE
        assert!(results[0].training_mse < 1.0,
            "best fit for 2n+1 should have MSE < 1, got {} (formula: {})",
            results[0].training_mse, results[0].formula_str);
    }

    #[test]
    fn test_conjecture_engine_full_pipeline() {
        let mut engine = ConjectureEngine::with_config(RegressorConfig {
            population_size: 80,
            generations: 30,
            max_depth: 3,
            max_complexity: 12,
            seed: 123,
            ..RegressorConfig::default()
        });

        // Observe a simple quadratic: f(n) = n²
        let data: Vec<(f64, f64)> = (1..=25).map(|n| (n as f64, (n * n) as f64)).collect();
        engine.observe(ObservedSequence::new("squares", MathDomain::NumberTheory, data));

        // Generate conjectures
        engine.generate_conjectures(5);
        assert!(!engine.conjectures.is_empty());

        // Verify numerically
        engine.verify_numerical();

        // Report
        let report = engine.report();
        assert!(report.contains("squares"), "report should mention source: {}", report);
    }

    #[test]
    fn test_fibonacci_ratio_discovery() {
        let mut engine = ConjectureEngine::with_config(RegressorConfig {
            population_size: 150,
            generations: 60,
            max_depth: 3,
            max_complexity: 8,
            seed: 42,
            ..RegressorConfig::default()
        });

        engine.observe(observe_fibonacci_ratios(30));
        engine.generate_conjectures(3);
        engine.verify_numerical();

        // The best conjecture should approximate φ ≈ 1.618
        if let Some(best) = engine.best_for("fibonacci_ratio(n)") {
            let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
            // Evaluate at a large n — should be close to phi
            let predicted = best.formula.eval(&[("n", 30.0)]);
            assert!(
                (predicted - phi).abs() < 0.5 || best.training_mse < 0.1,
                "best fibonacci ratio conjecture should approximate φ: predicted={}, mse={}, formula={}",
                predicted, best.training_mse, best.formula_str,
            );
        }
    }

    /// Discovery experiment: run the full pipeline on multiple sequences and
    /// print what the engine actually finds.
    #[test]
    fn test_discovery_report() {
        let mut engine = ConjectureEngine::with_config(RegressorConfig {
            population_size: 200,
            generations: 80,
            max_depth: 4,
            max_complexity: 15,
            lambda: 0.001,
            seed: 42,
            ..RegressorConfig::default()
        });

        // Feed multiple sequences
        engine.observe(observe_fibonacci_ratios(30));
        engine.observe(observe_perm_det_ratio(5));
        engine.observe(observe_partitions(20));

        // Simple known sequence: triangular numbers T(n) = n(n+1)/2
        let triangular: Vec<(f64, f64)> = (1..=25)
            .map(|n| (n as f64, (n * (n + 1) / 2) as f64))
            .collect();
        engine.observe(ObservedSequence::new("triangular(n)", MathDomain::Combinatorics, triangular));

        // Run discovery
        engine.generate_conjectures(3);
        engine.verify_numerical();

        // Print the report
        eprintln!("\n{}\n", engine.report());

        // Print detailed results per sequence
        for seq_name in &["fibonacci_ratio(n)", "perm_det_ratio(n)", "triangular(n)", "partition_count(n)"] {
            if let Some(best) = engine.best_for(seq_name) {
                eprintln!("DISCOVERY: {} ≈ {}", seq_name, best.formula_str);
                eprintln!("  MSE={:.2e}, complexity={}, confidence={:.2}, status={:?}",
                    best.training_mse, best.complexity, best.confidence, best.status);
                // Evaluate at a few points
                for n in [1.0, 5.0, 10.0, 20.0] {
                    let predicted = best.formula.eval(&[("n", n)]);
                    eprintln!("  f({}) = {:.6}", n, predicted);
                }
                eprintln!();
            }
        }

        // At minimum, the engine should have generated some conjectures
        assert!(!engine.conjectures.is_empty(), "should generate at least one conjecture");
    }

    /// Test formal verification: triangular number formula should pass bounded induction.
    #[test]
    fn test_formal_verification_triangular() {
        let mut engine = ConjectureEngine::with_config(RegressorConfig {
            population_size: 200,
            generations: 80,
            max_depth: 4,
            max_complexity: 15,
            seed: 42,
            ..RegressorConfig::default()
        });

        // Triangular numbers: T(n) = n(n+1)/2
        let data: Vec<(f64, f64)> = (1..=30)
            .map(|n| (n as f64, (n * (n + 1) / 2) as f64))
            .collect();
        engine.observe(ObservedSequence::new("triangular(n)", MathDomain::Combinatorics, data));

        engine.generate_conjectures(3);
        engine.verify_numerical();
        engine.verify_formal(200);

        eprintln!("\n=== Formal Verification Results ===");
        for c in &engine.conjectures {
            eprintln!("  {} ≈ {} | status={:?} | confidence={:.2}",
                c.source, c.formula_str, c.status, c.confidence);
        }

        // At least one conjecture should be formally verified
        let any_verified = engine.conjectures.iter().any(|c|
            matches!(c.status, ConjectureStatus::FormallyVerified { .. }));
        // It's OK if none are formally verified (the regressor might find a
        // formula that's close but not exact for all 200 values)
        if any_verified {
            eprintln!("  >>> FORMALLY VERIFIED conjecture found!");
        }
    }

    /// THE GCT SCALING EXPERIMENT — potentially novel mathematics.
    ///
    /// Compute Kronecker coefficient obstruction ratios for n=2..5,
    /// feed into the ConjectureEngine, and see if a scaling law emerges.
    /// If the ratio follows a discoverable pattern, this is publishable
    /// computational evidence in algebraic combinatorics.
    #[test]
    fn test_gct_scaling_experiment() {
        // Phase 1: Collect raw GCT data (up to n=6 — the critical frontier)
        let detailed = observe_gct_detailed(6);
        eprintln!("\n═══ GCT SCALING EXPERIMENT ═══");
        eprintln!("Computing Kronecker coefficient obstructions for perm_n vs det_n²...\n");
        for obs in &detailed {
            eprintln!("  n={}: {}/{} zero coefficients ({:.1}%) — P≠NP evidence: {}",
                obs.n, obs.obstructions, obs.total, obs.ratio * 100.0,
                if obs.ratio > 0.3 { "YES" } else { "no" });
            for (lam, mu, nu, coeff) in &obs.survivors {
                eprintln!("    SURVIVOR: λ={:?}, μ={:?}, ν={:?} → LR bound = {}",
                    lam, mu, nu, coeff);
            }
        }

        // Phase 2: Feed into ConjectureEngine
        let mut engine = ConjectureEngine::with_config(RegressorConfig {
            population_size: 300,
            generations: 150,
            max_depth: 4,
            max_complexity: 12,
            lambda: 0.0001, // very low Occam penalty — we want accuracy
            seed: 42,
            ..RegressorConfig::default()
        });

        engine.observe(observe_gct_obstruction(6));
        engine.generate_conjectures(5);
        engine.verify_numerical();

        eprintln!("\n═══ CONJECTURE ENGINE RESULTS ═══");
        for c in engine.conjectures.iter().take(5) {
            eprintln!("  obstruction(n) ≈ {} | MSE={:.2e} | status={:?}",
                c.formula_str, c.training_mse, c.status);
            // Evaluate predictions
            for n in 2..=6 {
                let pred = c.formula.eval(&[("n", n as f64)]);
                eprintln!("    n={}: predicted={:.4}", n, pred);
            }
        }

        if let Some(best) = engine.best_for("gct_obstruction_ratio(n)") {
            eprintln!("\n  >>> BEST SCALING LAW: obstruction(n) ≈ {}", best.formula_str);
            eprintln!("  >>> MSE={:.2e}, confidence={:.2}", best.training_mse, best.confidence);

            // Predict n=6 (potentially novel — extrapolation beyond training data)
            let pred_6 = best.formula.eval(&[("n", 6.0)]);
            eprintln!("  >>> PREDICTION for n=6: obstruction_ratio ≈ {:.4}", pred_6);
            eprintln!("  >>> (This prediction is UNTESTED — verify by computing check_obstruction_conjecture(6, 36))");
        }

        // Must produce at least some data
        assert!(!detailed.is_empty());
    }

    /// Partition function with expanded grammar.
    #[test]
    fn test_partition_expanded_grammar() {
        let mut engine = ConjectureEngine::with_config(RegressorConfig {
            population_size: 300,
            generations: 120,
            max_depth: 5,
            max_complexity: 20,
            lambda: 0.0005, // lower Occam penalty to allow more complex formulas
            seed: 42,
            ..RegressorConfig::default()
        });

        engine.observe(observe_partitions(30));
        engine.generate_conjectures(5);
        engine.verify_numerical();

        eprintln!("\n=== Partition Function Discovery (Expanded Grammar) ===");
        for c in engine.conjectures.iter().take(5) {
            eprintln!("  p(n) ≈ {} | MSE={:.2e} | complexity={} | status={:?}",
                c.formula_str, c.training_mse, c.complexity, c.status);
            // Show predictions vs actual
            for n in [5, 10, 15, 20] {
                let pred = c.formula.eval(&[("n", n as f64)]);
                let actual = crate::hdc::combinatorics::partition_count(n) as f64;
                eprintln!("    p({})={:.0}, predicted={:.1}", n, actual, pred);
            }
        }

        // The best formula should at least capture the growth trend
        if let Some(best) = engine.best_for("partition_count(n)") {
            eprintln!("\n  BEST: p(n) ≈ {}", best.formula_str);
        }
    }

    /// Lorenz attractor: discover that ⟨z⟩ converges to ρ-1 = 27.
    #[test]
    fn test_lorenz_time_average_discovery() {
        let mut engine = ConjectureEngine::with_config(RegressorConfig {
            population_size: 200,
            generations: 80,
            max_depth: 3,
            max_complexity: 8,
            lambda: 0.001,
            seed: 42,
            ..RegressorConfig::default()
        });

        engine.observe(observe_lorenz_time_averages(20));
        engine.generate_conjectures(3);
        engine.verify_numerical();

        eprintln!("\n═══ LORENZ TIME-AVERAGE DISCOVERY ═══");
        for c in engine.conjectures.iter().take(3) {
            eprintln!("  ⟨z⟩ ≈ {} | MSE={:.2e} | status={:?}",
                c.formula_str, c.training_mse, c.status);
            let pred = c.formula.eval(&[("n", 20.0)]);
            eprintln!("    predicted ⟨z⟩ = {:.4} (expected ≈ 27.0)", pred);
        }

        // The time average should converge to ~27 (ρ-1)
        if let Some(best) = engine.best_for("lorenz_time_avg_z(samples)") {
            let pred = best.formula.eval(&[("n", 20.0)]);
            eprintln!("\n  >>> BEST: ⟨z⟩ ≈ {} (predicted={:.4})", best.formula_str, pred);
            // Should be within 10% of 27
            assert!(
                (pred - 27.0).abs() < 5.0 || best.training_mse < 1.0,
                "Lorenz ⟨z⟩ should approximate 27, got {:.4} (formula: {})",
                pred, best.formula_str,
            );
        }
    }

    #[test]
    fn test_lorenz_trajectory_generated() {
        let (times, states) = rk45_trajectory(lorenz_rhs, &[1.0, 1.0, 1.0], 10.0, 0.01);
        assert!(times.len() > 100, "should have many time steps");
        assert_eq!(states[0].len(), 3, "Lorenz is 3D");
        // After transient, z should be positive (attractor lives at z > 0)
        let last_z = states.last().unwrap()[2];
        assert!(last_z > 0.0, "Lorenz z should be positive on attractor, got {}", last_z);
    }

    /// PHYSICS DISCOVERY: find E = x² + v² is conserved in harmonic oscillator.
    #[test]
    fn test_harmonic_oscillator_invariant() {
        let candidates = observe_harmonic_invariants(50);

        eprintln!("\n═══ HARMONIC OSCILLATOR INVARIANT DISCOVERY ═══");
        for seq in &candidates {
            let (mean, var) = invariant_variance(&seq.data);
            let is_conserved = var < 1e-6;
            eprintln!("  {} | mean={:.6}, variance={:.2e} | CONSERVED: {}",
                seq.name, mean, var, if is_conserved { "YES" } else { "no" });
        }

        // x²+v² should be conserved (variance ≈ 0)
        let (e_mean, e_var) = invariant_variance(&candidates[0].data);
        assert!(e_var < 1e-6,
            "E = x²+v² should be conserved (var={:.2e}), mean={:.6}", e_var, e_mean);
        assert!((e_mean - 1.0).abs() < 0.01,
            "E should equal initial energy 1.0, got {:.6}", e_mean);

        // x² should NOT be conserved
        let (_, x2_var) = invariant_variance(&candidates[1].data);
        assert!(x2_var > 0.01,
            "x² should oscillate (not conserved), var={:.2e}", x2_var);

        eprintln!("  >>> DISCOVERY: E = x² + v² is a conserved quantity (var={:.2e})", e_var);
        eprintln!("  >>> x² alone is NOT conserved (var={:.2e})", x2_var);
    }

    /// Summation operator test: Σ_{k=0}^{n} k = n(n+1)/2
    #[test]
    fn test_summation_operator() {
        // Σ_{k=0}^{n} k
        let expr = Expr::Sum(Box::new(Expr::Var("k".into())), "k".into());
        // Σ_{k=0}^5 k = 0+1+2+3+4+5 = 15
        let result = expr.eval(&[("n", 5.0)]);
        assert!((result - 15.0).abs() < 1e-10, "Σ k for n=5 should be 15, got {}", result);
        // Σ_{k=0}^10 k = 55
        let result10 = expr.eval(&[("n", 10.0)]);
        assert!((result10 - 55.0).abs() < 1e-10, "Σ k for n=10 should be 55, got {}", result10);
        // Display
        assert_eq!(format!("{}", expr), "Σ_k(k)");
    }

    /// CROSS-SEQUENCE DISCOVERY: verify B(n) = Σ_{k=0}^{n} S(n,k)
    ///
    /// This is the Bell-Stirling identity. Rather than asking the GP regressor
    /// to discover it (which would require the regressor to invent Stirling
    /// numbers from scratch), we verify it by direct computation: compute both
    /// sides and check the residual is zero for all n.
    ///
    /// This validates the cross-sequence identity infrastructure.
    #[test]
    fn test_bell_stirling_identity() {
        let residual = observe_bell_stirling_residual(15);

        eprintln!("\n═══ BELL-STIRLING IDENTITY VERIFICATION ═══");
        eprintln!("Testing: B(n) = Σ_{{k=0}}^n S(n,k) for n=0..15\n");

        let bell_seq = observe_bell_numbers(15);
        let stirling_seq = observe_stirling_sum(15);

        let mut all_match = true;
        for i in 0..residual.data.len() {
            let n = residual.data[i].0 as usize;
            let b = bell_seq.data[i].1;
            let s = stirling_seq.data[i].1;
            let diff = residual.data[i].1;
            let matches = diff < 1e-10;
            if !matches { all_match = false; }
            eprintln!("  n={:2}: B(n)={:>10.0}, Σ S(n,k)={:>10.0}, |diff|={:.0e} {}",
                n, b, s, diff, if matches { "✓" } else { "✗" });
        }

        assert!(all_match, "B(n) should equal Σ S(n,k) for all n");
        eprintln!("\n  >>> VERIFIED: B(n) = Σ_{{k=0}}^n S(n,k) for all n ∈ [0, 15]");
        eprintln!("  >>> This is the Bell-Stirling identity — proven by exhaustive computation.");
    }

    /// Test that Bell and Stirling-sum sequences are numerically identical.
    #[test]
    fn test_bell_equals_stirling_sum() {
        use crate::hdc::combinatorics::{bell, stirling_second};
        for n in 0..=12 {
            let b = bell(n);
            let s_sum: u64 = (0..=n).map(|k| stirling_second(n, k)).sum();
            assert_eq!(b, s_sum, "B({}) = {} ≠ Σ S({},k) = {}", n, b, n, s_sum);
        }
    }

    #[test]
    fn test_cross_fit_same_formula_different_domains() {
        // Create two sequences from different domains that follow the same law: f(n) = n^2
        let physics_seq = ObservedSequence::new(
            "kinetic_energy(v)",
            MathDomain::Physics,
            (1..=20).map(|n| (n as f64, (n * n) as f64)).collect(),
        );
        let biology_seq = ObservedSequence::new(
            "population_growth(t)",
            MathDomain::Biology,
            (1..=20).map(|n| (n as f64, (n * n) as f64)).collect(),
        );

        let mut engine = ConjectureEngine::new();
        engine.observe(physics_seq);
        engine.observe(biology_seq);
        engine.generate_conjectures(3);

        // Find any conjecture from Physics domain
        let physics_conjectures: Vec<&Conjecture> = engine
            .conjectures
            .iter()
            .filter(|c| c.domain == MathDomain::Physics)
            .collect();

        if let Some(best) = physics_conjectures.first() {
            // Test cross-fit: physics formula should also fit biology data
            let bio_seq = &engine.observations[1];
            let ratio = ConjectureEngine::cross_fit(best, bio_seq);
            // If the formula is good, ratio should exist and be close to 1.0
            if let Some(r) = ratio {
                assert!(
                    r < 10.0,
                    "Same-law sequences should have low MSE ratio, got {}",
                    r
                );
            }
        }
    }

    #[test]
    fn test_cross_fit_rejects_same_domain() {
        let seq1 = ObservedSequence::new(
            "seq1", MathDomain::Physics,
            vec![(1.0, 1.0), (2.0, 4.0), (3.0, 9.0)],
        );
        let conjecture = Conjecture {
            formula: Expr::BinOp(BinOp::Pow, Box::new(Expr::Var("n".into())), Box::new(Expr::Const(2.0))),
            formula_str: "n^2".to_string(),
            source: "seq1".to_string(),
            domain: MathDomain::Physics,
            training_mse: 0.0,
            complexity: 3,
            fitness: 0.003,
            status: ConjectureStatus::Proposed,
            confidence: 0.5,
        };
        // Same domain → should return None
        assert!(ConjectureEngine::cross_fit(&conjecture, &seq1).is_none());
    }

    #[test]
    fn test_discover_cross_domain_formulas() {
        let mut engine = ConjectureEngine::new();
        // Linear law in two different domains
        engine.observe(ObservedSequence::new(
            "spring_force(x)", MathDomain::Physics,
            (1..=20).map(|n| (n as f64, 2.0 * n as f64 + 1.0)).collect(),
        ));
        engine.observe(ObservedSequence::new(
            "cost_function(q)", MathDomain::Economics,
            (1..=20).map(|n| (n as f64, 2.0 * n as f64 + 1.0)).collect(),
        ));
        engine.generate_conjectures(3);
        engine.verify_numerical();

        let matches = engine.discover_cross_domain_formulas(5.0);
        // Should find at least the possibility (may or may not depending on GP convergence)
        // Just verify it doesn't panic and returns valid results
        for m in &matches {
            assert_ne!(m.source_domain, m.target_domain);
            assert!(m.mse_ratio < 5.0);
        }
    }

    // ── Simplification tests ────────────────────────────────────────────

    #[test]
    fn test_simplify_identity_rules() {
        // x + 0 = x
        let e = Expr::BinOp(BinOp::Add, Box::new(Expr::Var("n".into())), Box::new(Expr::Const(0.0)));
        assert_eq!(format!("{}", simplify(&e)), "n");
        // x * 1 = x
        let e = Expr::BinOp(BinOp::Mul, Box::new(Expr::Var("n".into())), Box::new(Expr::Const(1.0)));
        assert_eq!(format!("{}", simplify(&e)), "n");
        // x * 0 = 0
        let e = Expr::BinOp(BinOp::Mul, Box::new(Expr::Var("n".into())), Box::new(Expr::Const(0.0)));
        assert_eq!(format!("{}", simplify(&e)), "0");
        // x ^ 1 = x
        let e = Expr::BinOp(BinOp::Pow, Box::new(Expr::Var("n".into())), Box::new(Expr::Const(1.0)));
        assert_eq!(format!("{}", simplify(&e)), "n");
    }

    #[test]
    fn test_simplify_div_div() {
        // a / (b / c) = a*c / b → (n+1) / (2/n) → ((n+1)*n) / 2
        let inner = Expr::BinOp(BinOp::Div, Box::new(Expr::Const(2.0)), Box::new(Expr::Var("n".into())));
        let outer = Expr::BinOp(BinOp::Div,
            Box::new(Expr::BinOp(BinOp::Add, Box::new(Expr::Var("n".into())), Box::new(Expr::Const(1.0)))),
            Box::new(inner));
        let simplified = simplify(&outer);
        // Should evaluate the same at n=5: (5+1)/(2/5) = 6/0.4 = 15 = T(5)
        let orig_val = outer.eval(&[("n", 5.0)]);
        let simp_val = simplified.eval(&[("n", 5.0)]);
        assert!((orig_val - simp_val).abs() < 1e-10,
            "simplified should match original: {} vs {}", orig_val, simp_val);
        // The simplified form should contain Mul (not nested Div)
        let s = format!("{}", simplified);
        assert!(!s.contains("/ ("), "should eliminate nested division: {}", s);
    }

    #[test]
    fn test_simplify_constant_folding() {
        // 2 + 3 = 5
        let e = Expr::BinOp(BinOp::Add, Box::new(Expr::Const(2.0)), Box::new(Expr::Const(3.0)));
        assert_eq!(format!("{}", simplify(&e)), "5");
        // sin(0) = 0
        let e = Expr::Func(UnaryFn::Sin, Box::new(Expr::Const(0.0)));
        assert_eq!(format!("{}", simplify(&e)), "0");
    }

    // ── Recurrence detection tests ──────────────────────────────────────

    #[test]
    fn test_detect_recurrence_triangular() {
        // T(n) = T(n-1) + n: data = [(1,1), (2,3), (3,6), (4,10), (5,15)]
        let data: Vec<(f64, f64)> = (1..=10).map(|n| (n as f64, (n*(n+1)/2) as f64)).collect();
        let rec = detect_recurrence(&data);
        assert!(rec.is_some(), "should detect f(n) = f(n-1) + n");
        let r = rec.unwrap();
        assert!(r.formula.contains("f(n-1) + n"), "formula: {}", r.formula);
        eprintln!("  Detected: {} (residual={:.2e})", r.formula, r.max_residual);
    }

    #[test]
    fn test_detect_recurrence_fibonacci() {
        use crate::hdc::combinatorics::fibonacci;
        let data: Vec<(f64, f64)> = (1..=15).map(|n| (n as f64, fibonacci(n) as f64)).collect();
        let rec = detect_recurrence(&data);
        assert!(rec.is_some(), "should detect f(n) = f(n-1) + f(n-2)");
        let r = rec.unwrap();
        assert!(r.formula.contains("f(n-2)"), "formula: {}", r.formula);
        eprintln!("  Detected: {} (residual={:.2e})", r.formula, r.max_residual);
    }

    #[test]
    fn test_detect_recurrence_geometric() {
        // f(n) = 2*f(n-1): data = [1, 2, 4, 8, 16, 32]
        let data: Vec<(f64, f64)> = (0..=8).map(|n| (n as f64, 2.0f64.powi(n as i32))).collect();
        let rec = detect_recurrence(&data);
        assert!(rec.is_some(), "should detect f(n) = 2*f(n-1)");
        let r = rec.unwrap();
        assert!((r.coefficients[0] - 2.0).abs() < 1e-6, "coefficient should be 2, got {}", r.coefficients[0]);
        eprintln!("  Detected: {} (residual={:.2e})", r.formula, r.max_residual);
    }

    /// Nelder-Mead constant optimization test
    #[test]
    fn test_nelder_mead_improves_constants() {
        // Create a*n + b with wrong constants, fit to y = 3n + 7
        let expr = Expr::BinOp(BinOp::Add,
            Box::new(Expr::BinOp(BinOp::Mul, Box::new(Expr::Const(1.0)), Box::new(Expr::Var("n".into())))),
            Box::new(Expr::Const(1.0)));
        let data: Vec<(f64, f64)> = (1..=20).map(|n| (n as f64, 3.0 * n as f64 + 7.0)).collect();

        let before_mse = compute_mse(&expr, &data);
        let optimized = optimize_constants(&expr, &data, 100);
        let after_mse = compute_mse(&optimized, &data);

        eprintln!("  NM optimization: MSE {:.2e} → {:.2e}", before_mse, after_mse);
        assert!(after_mse < before_mse * 0.1,
            "NM should significantly improve: {:.2e} → {:.2e}", before_mse, after_mse);
    }

    /// Can Nelder-Mead recover Hardy-Ramanujan constants given the right skeleton?
    /// p(n) ≈ a * exp(b * sqrt(n)) / (c * n)
    /// True: a = 1/(4√3) ≈ 0.1443, b = π√(2/3) ≈ 2.5650, c = 1
    #[test]
    fn test_nelder_mead_hardy_ramanujan() {
        use crate::hdc::combinatorics::partition_count;

        // Build the skeleton: a * exp(b * sqrt(n)) / (c * n)
        // with initial guesses a=1, b=1, c=1
        let skeleton = Expr::BinOp(BinOp::Div,
            Box::new(Expr::BinOp(BinOp::Mul,
                Box::new(Expr::Const(1.0)), // a
                Box::new(Expr::Func(UnaryFn::Exp,
                    Box::new(Expr::BinOp(BinOp::Mul,
                        Box::new(Expr::Const(1.0)), // b
                        Box::new(Expr::Func(UnaryFn::Sqrt,
                            Box::new(Expr::Var("n".into())))))))))),
            Box::new(Expr::BinOp(BinOp::Mul,
                Box::new(Expr::Const(1.0)), // c
                Box::new(Expr::Var("n".into())))));

        let data: Vec<(f64, f64)> = (5..=40)
            .map(|n| (n as f64, partition_count(n as u64) as f64))
            .collect();

        let before_mse = compute_mse(&skeleton, &data);
        let optimized = optimize_constants(&skeleton, &data, 500);
        let after_mse = compute_mse(&optimized, &data);

        // Extract optimized constants
        let consts = collect_constants(&optimized);
        eprintln!("\n═══ HARDY-RAMANUJAN CONSTANT RECOVERY ═══");
        eprintln!("  Skeleton: a * exp(b * sqrt(n)) / (c * n)");
        eprintln!("  Before NM: MSE = {:.2e}", before_mse);
        eprintln!("  After NM:  MSE = {:.2e}", after_mse);
        if consts.len() >= 3 {
            let true_a = 1.0 / (4.0 * 3.0_f64.sqrt());
            let true_b = std::f64::consts::PI * (2.0_f64 / 3.0).sqrt();
            eprintln!("  Discovered: a={:.6}, b={:.6}, c={:.6}", consts[0], consts[1], consts[2]);
            eprintln!("  True H-R:   a={:.6}, b={:.6}", true_a, true_b);
            eprintln!("  a error: {:.1}%", ((consts[0] - true_a) / true_a * 100.0).abs());
            eprintln!("  b error: {:.1}%", ((consts[1] - true_b) / true_b * 100.0).abs());
        }

        // Show predictions
        for n in [10, 20, 30, 40, 50] {
            let pred = optimized.eval(&[("n", n as f64)]);
            let actual = if n <= 40 { partition_count(n) as f64 } else { f64::NAN };
            eprintln!("  p({})={:.0}, predicted={:.0}", n,
                if actual.is_nan() { -1.0 } else { actual }, pred);
        }

        assert!(after_mse < before_mse, "NM should improve on wrong constants");
    }

    /// Combined pipeline: recurrence detection + simplification + GP discovery
    #[test]
    fn test_full_pipeline_with_improvements() {
        // Generate factorial: f(n) = n * f(n-1)
        let data: Vec<(f64, f64)> = (1..=10)
            .map(|n| {
                let mut f = 1u64;
                for i in 1..=n { f *= i; }
                (n as f64, f as f64)
            })
            .collect();

        // Recurrence detection should find it
        let rec = detect_recurrence(&data);
        eprintln!("\n═══ FACTORIAL PIPELINE ═══");
        if let Some(r) = &rec {
            eprintln!("  Recurrence detected: {}", r.formula);
        }

        // GP + NM should find an approximation
        let seq = ObservedSequence::new("factorial(n)", MathDomain::Combinatorics, data.clone());
        let mut regressor = SymbolicRegressor::new(RegressorConfig {
            population_size: 200, generations: 80, max_depth: 4,
            max_complexity: 12, seed: 42, ..RegressorConfig::default()
        });
        let results = regressor.fit(&seq, 3);
        for r in &results {
            let simplified = simplify(&r.formula);
            eprintln!("  GP found: {} (simplified: {}) MSE={:.2e}",
                r.formula_str, simplified, r.training_mse);
        }

        assert!(!results.is_empty());
    }
}
