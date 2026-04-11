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
        }
    }

    /// Count AST nodes (complexity metric for Occam penalty).
    pub fn complexity(&self) -> usize {
        match self {
            Expr::Var(_) | Expr::Const(_) => 1,
            Expr::BinOp(_, l, r) => 1 + l.complexity() + r.complexity(),
            Expr::Func(_, arg) => 1 + arg.complexity(),
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
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Var(name) => write!(f, "{}", name),
            Expr::Const(c) => {
                if (*c - c.round()).abs() < 1e-10 && c.abs() < 1e12 {
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
            // Common mathematical constants + small integers
            let constants = [1.0, 2.0, 3.0, 0.5, std::f64::consts::PI,
                             std::f64::consts::E, (1.0 + 5.0_f64.sqrt()) / 2.0]; // φ
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
    AlgebraicComplexity,  // GCT
    DynamicalSystems,     // ODEs, attractors
    SpectralAnalysis,     // FFT
    Chemistry,
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

            // Elitism: keep top 10%
            let elite_count = self.config.population_size / 10;
            let elite: Vec<Expr> = scored.iter()
                .take(elite_count)
                .map(|(i, _)| self.population[*i].clone())
                .collect();

            // Build next generation
            let mut next_gen = elite;
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
pub fn observe_gct_obstruction(max_n: usize) -> ObservedSequence {
    use super::gct::check_obstruction_conjecture;
    let data: Vec<(f64, f64)> = (2..=max_n.min(5)) // cap at 5 (tractable)
        .map(|n| {
            let result = check_obstruction_conjecture(n, n * n);
            (n as f64, result.obstruction_ratio)
        })
        .collect();
    ObservedSequence::new("gct_obstruction_ratio(n)", MathDomain::AlgebraicComplexity, data)
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
}
