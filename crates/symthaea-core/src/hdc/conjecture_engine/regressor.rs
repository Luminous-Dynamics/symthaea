use super::*;

/// Configuration for the symbolic regression search.
#[derive(Debug, Clone)]
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
    /// If true, skip macro-operator seeding even if macros are available.
    /// Used for cold-vs-primed benchmarking.
    pub disable_macro_seeds: bool,
    /// If true, remove Sin/Cos from the unary function set used by the
    /// multivariate autonomous GP (random_expr_multivar + mutate_multivar).
    /// Diagnostic flag for Ceiling-4 work: trig functions create
    /// low-variance degenerate fits (e.g. `cos(y³) * 0.11`) that crowd
    /// out Kepler-shaped primitives during PCR3BP discovery. Setting
    /// this to true forces the GP to seek non-trigonometric invariants.
    pub exclude_trig: bool,
    /// Number of trajectories (with perturbed initial conditions) to
    /// sample in the autonomous discoverer's fitness function.
    /// Default 1 preserves prior behavior. Values > 1 evaluate variance
    /// on each trajectory independently and take the MAX — an expression
    /// constant on only one orbit (accidental-constant-of-this-orbit)
    /// loses to a true conservation law constant on all orbits. This is
    /// the Session-21 fix for Ceiling 4.
    pub diverse_trajectory_count: usize,
    /// Session 24: probability per-child that, instead of standard
    /// mutation/crossover, the GP composes two distinct pinned priors
    /// via a random binary operation (Add/Sub/Mul). Targets the
    /// composition-limited ceiling identified by Session 23: crossover
    /// rarely picks complementary pinned primitives as both parents,
    /// so the GP finds single-term partials instead of compositions.
    /// Default 0.0 preserves prior behavior. Only fires when the caller
    /// supplies at least 2 priors via extra_seed_templates.
    pub prior_composition_rate: f64,
    /// Session 25: structural-richness reward. When less than 1.0,
    /// fitness is multiplied by `prior_fragment_bonus^k` where k is
    /// the count of caller-supplied priors that appear as exact
    /// subtrees in the expression. Lower is stronger: 0.5 halves
    /// fitness per matched prior, 0.1 cuts it by 10× per match.
    /// Default 1.0 (no bonus). Targets the Session-24 finding that
    /// the composition operator produces 2-piece composites but
    /// variance selection doesn't reward their structural richness.
    pub prior_fragment_bonus: f64,
    /// Session 29: gradient-orthogonality penalty. When > 1.0 and
    /// `known_invariants` is non-empty, each candidate's fitness is
    /// multiplied by this factor whenever its state-space gradient
    /// is highly parallel (mean |cos θ| > orthogonality_threshold)
    /// to any known invariant's gradient across sampled trajectory
    /// points. Catches tautological variants like `L·π`, `L+L`,
    /// `exp(L)`, etc — all of which have gradients parallel to `∇L`
    /// even when the functional form differs. This unblocks the
    /// multi-invariant discovery problem diagnosed in Session 28
    /// (ang_mom's 1e-29 variance floor shadows all other invariants
    /// in top-10 selection). Default 1.0 (no penalty).
    pub orthogonality_penalty: f64,
    /// Session 29: cosine threshold for orthogonality_penalty. Mean
    /// |cos(grad_E, grad_I_k)| above this triggers the penalty.
    /// 0.9 catches scalar rescalings and element-wise nonlinearities;
    /// 0.99 is strict; 0.5 is lax. Default 0.9.
    pub orthogonality_threshold: f64,
    /// Session 29: invariants already discovered in a previous pass.
    /// When provided together with `orthogonality_penalty > 1.0`,
    /// candidates whose state-space gradient is parallel to any of
    /// these get a fitness penalty, forcing discovery of structurally
    /// independent invariants. Default empty.
    pub known_invariants: Vec<Expr>,
    /// Session 30: use Lie-derivative variance instead of raw
    /// trajectory variance as the fitness metric. For a candidate
    /// `E(state)`, the Lie derivative along the flow is
    /// `L_f E = ∇E · f(state)` where `f` is the RHS of the ODE.
    /// True conservation laws satisfy `L_f E = 0` exactly (up to
    /// integration error), so the variance of `L_f E` along the
    /// trajectory is zero for genuine invariants. Gameable 1D
    /// near-constants like `y^6` have non-zero `L_f E` because the
    /// flow `f` has non-zero components in every direction, forcing
    /// any expression with non-trivial dependence to produce varying
    /// derivatives. This is the physics-correct fitness — it cannot
    /// be satisfied by finite-sample accidents.
    /// Requires the caller to pass `rhs` (the ODE function) as part
    /// of the autonomous-discovery API, which we already do. Default
    /// false (preserves the S19-S29 variance fitness).
    pub use_lie_fitness: bool,
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
            disable_macro_seeds: false,
            exclude_trig: false,
            diverse_trajectory_count: 1,
            prior_composition_rate: 0.0,
            prior_fragment_bonus: 1.0,
            orthogonality_penalty: 1.0,
            orthogonality_threshold: 0.9,
            known_invariants: Vec::new(),
            use_lie_fitness: false,
        }
    }
}

impl RegressorConfig {
    /// Preset tuned for autonomous multivariate invariant discovery.
    ///
    /// This is the configuration validated by the Ramanujan Protocol's
    /// twelve-session arc (Sessions 15-26, Apr 17 2026), including the
    /// S26 Kepler control experiment that recovered angular momentum
    /// verbatim in 5/5 seeds at machine-epsilon variance.
    ///
    /// Settings:
    /// - `exclude_trig: true` — drops Sin/Cos from the unary function
    ///   set. Session 19 showed trig functions produce low-variance
    ///   degenerate fits (e.g. `cos(y³)·c`) that crowd out Kepler-shaped
    ///   primitives during multivariate discovery.
    /// - `diverse_trajectory_count: 5` — fitness is evaluated as MAX
    ///   variance across 5 perturbed-IC orbits instead of one.
    ///   Session 21 showed this prevents "accidentally-near-constant
    ///   on this specific orbit" from beating true conservation laws.
    /// - `prior_composition_rate: 0.15` — 15% of children are
    ///   `op(prior_A, prior_B)` for random distinct pinned priors.
    ///   Session 24 produced the arc's first 2-piece composite
    ///   (`1/r₁ − 1/r_origin`) with this rate.
    /// - `prior_fragment_bonus: 0.5` — fitness is halved per pinned
    ///   prior appearing as exact subtree. Session 25 showed this
    ///   gives the best-reliability condition (4/5 survivors, lowest
    ///   variance).
    ///
    /// Callers can still override any field after construction.
    /// Leaves `population_size`, `generations`, `seed` etc. at the
    /// default values — callers should set these for their target.
    ///
    /// # Example
    /// ```no_run
    /// use symthaea_core::hdc::conjecture_engine::RegressorConfig;
    /// let cfg = RegressorConfig {
    ///     seed: 42,
    ///     population_size: 300,
    ///     generations: 100,
    ///     max_depth: 6,
    ///     max_complexity: 24,
    ///     ..RegressorConfig::for_autonomous_discovery()
    /// };
    /// ```
    pub fn for_autonomous_discovery() -> Self {
        Self {
            exclude_trig: true,
            diverse_trajectory_count: 5,
            prior_composition_rate: 0.15,
            prior_fragment_bonus: 0.5,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SeedSpecializationStats {
    pub variants_scored: usize,
    pub variants_seeded: usize,
    pub elapsed_ms: u128,
    pub exact_fit_found: bool,
}

/// Grammar-guided symbolic regression via genetic programming.
pub struct SymbolicRegressor {
    config: RegressorConfig,
    population: Vec<Expr>,
    rng: u64,
    /// Optional seed expressions from learned macro-operators (abstract thought).
    /// Injected into the initial population alongside growth-class templates.
    seed_macros: Vec<Expr>,
    /// Per-generation best fitness (lower = better). Collected during `fit()`.
    /// Enables cold-vs-primed convergence comparison and "generations-to-ε" analysis.
    fitness_history: Vec<f64>,
    /// Macro-subtree appearance counts in top-k formulas at end of `fit()`.
    /// Key: canonical string of a seed macro. Value: how many top-k formulas
    /// contained a subtree matching that macro. Used for causal analysis
    /// in the macro acceleration benchmark.
    macro_usage: std::collections::HashMap<String, u64>,
    seed_specialization_stats: SeedSpecializationStats,
}

impl SymbolicRegressor {
    pub fn new(config: RegressorConfig) -> Self {
        let mut rng = config.seed;
        let population = (0..config.population_size)
            .map(|_| random_expr(&mut rng, config.max_depth))
            .collect();
        Self {
            config,
            population,
            rng,
            seed_macros: Vec::new(),
            fitness_history: Vec::new(),
            macro_usage: std::collections::HashMap::new(),
            seed_specialization_stats: SeedSpecializationStats::default(),
        }
    }

    /// Access the per-generation best-fitness history from the most recent `fit()` call.
    ///
    /// Each element is the best fitness (lower = better) observed after that
    /// generation's evaluation. Use for convergence analysis and cold-vs-primed comparison.
    pub fn fitness_history(&self) -> &[f64] {
        &self.fitness_history
    }

    /// Access macro usage counts from the most recent `fit()` call.
    ///
    /// Returns a map from each seed macro's canonical string to the number
    /// of top-k formulas where a structurally matching subtree appeared.
    /// Zero means no top-k formula contained that macro's pattern.
    pub fn macro_usage(&self) -> &std::collections::HashMap<String, u64> {
        &self.macro_usage
    }

    pub fn seed_specialization_stats(&self) -> &SeedSpecializationStats {
        &self.seed_specialization_stats
    }

    /// Inject macro-operator templates into the initial population.
    ///
    /// Called by `ConjectureEngine::generate_conjectures` when abstract thought
    /// is enabled and grammar has promoted macros. Each macro is instantiated
    /// with random constants to explore parameter variations.
    pub fn set_seed_macros(&mut self, macros: Vec<Expr>) {
        self.seed_macros = macros;
    }

    /// Run symbolic regression on observed data.
    /// Returns the top-k conjectures sorted by fitness (lower = better).
    pub fn fit(&mut self, seq: &ObservedSequence, top_k: usize) -> Vec<Conjecture> {
        // Clear history and usage at the start of each fit — each call is independent
        self.fitness_history.clear();
        self.macro_usage.clear();
        self.seed_specialization_stats = SeedSpecializationStats::default();
        // Pre-seed macro_usage keys so 0 counts are visible (not absent)
        for macro_expr in &self.seed_macros {
            let canonical = macro_usage_key(macro_expr);
            self.macro_usage.entry(canonical).or_insert(0);
        }
        let (train, _test) = seq.train_test_split();

        // ── Log-space pre-transform ──────────────────────────────────
        // If data is all positive and grows exponentially, try fitting
        // in log-space first. This turns a*exp(b*√n) into ln(a)+b*√n
        // which is trivially discoverable by GP.
        let all_positive = train.iter().all(|(_, y)| *y > 0.0);
        let growth = if train.len() >= 2 && train[0].1.abs() > 1e-10 {
            (train.last().expect("nonempty after len check").1 / train[0].1).abs()
        } else {
            1.0
        };

        // Pinned log-space candidates: exp-wrapped formulas discovered in
        // log-space. These get inserted into the population AND tracked as
        // guaranteed-available candidates, so tournament selection can't
        // silently filter them out before we reach the top-k return. The
        // original "insert one copy into a random slot" approach failed
        // on `derangements`/`fubini`/`motzkin`/`stirling_sum` because the
        // exp-wrapped formula has training_mse orders of magnitude worse
        // than a GP-discovered polynomial approximation on the first few
        // points — it loses tournament on partial fits even when it would
        // dominate on the full sequence.
        let mut log_space_pinned: Vec<Expr> = Vec::new();
        if all_positive && growth > 50.0 {
            let log_train: Vec<(f64, f64)> = train.iter().map(|(x, y)| (*x, y.ln())).collect();
            let log_seq =
                ObservedSequence::new(&format!("log({})", seq.name), seq.domain, log_train.clone());

            // Run a quick GP fit in log-space
            let mut log_regressor = SymbolicRegressor::new(RegressorConfig {
                population_size: self.config.population_size / 2,
                generations: self.config.generations / 2,
                max_depth: self.config.max_depth.min(4),
                max_complexity: self.config.max_complexity.min(12),
                lambda: self.config.lambda,
                tournament_size: self.config.tournament_size,
                mutation_rate: self.config.mutation_rate,
                seed: self.config.seed.wrapping_add(777),
                disable_macro_seeds: self.config.disable_macro_seeds,
                exclude_trig: self.config.exclude_trig,
                diverse_trajectory_count: self.config.diverse_trajectory_count,
                prior_composition_rate: self.config.prior_composition_rate,
                prior_fragment_bonus: self.config.prior_fragment_bonus,
                orthogonality_penalty: self.config.orthogonality_penalty,
                orthogonality_threshold: self.config.orthogonality_threshold,
                known_invariants: self.config.known_invariants.clone(),
                use_lie_fitness: self.config.use_lie_fitness,
            });
            let log_results = log_regressor.fit(&log_seq, 2);

            // Tightened acceptance: log-space training_mse < 0.1 means the
            // predicted log differs from the real log by ≤ 0.1 on average,
            // i.e. the exp-wrapped formula is within a factor of ~1.1×
            // of the true value. That's a meaningful closed-form recovery.
            // The old threshold (< 1.0) accepted factor-of-e errors.
            for lr in &log_results {
                if lr.training_mse < 0.1 {
                    let exp_wrapped = Expr::Func(UnaryFn::Exp, Box::new(lr.formula.clone()));
                    // Keep the exp-wrapped formula as a pinned candidate:
                    // we'll re-evaluate it on the original training data
                    // after GP evolution and include it in the top-k if
                    // it beats the evolved population.
                    log_space_pinned.push(exp_wrapped.clone());
                    // Also inject into the population — 3 copies (was 1)
                    // so tournament selection has more chances to keep it.
                    for _ in 0..3 {
                        self.rng = lcg_step(self.rng);
                        let idx = self.rng as usize % self.population.len();
                        self.population[idx] = exp_wrapped.clone();
                    }
                }
            }
        }

        // ── Template library seeding (#4) ────────────────────────────
        // Analyze growth class and seed population with appropriate templates.
        // This replaces blind random initialization with informed structures.
        let growth_class = analyze_growth(&train);
        let templates = build_template_library(&growth_class);
        let seed_count = (self.config.population_size / 4).min(templates.len() * 3);
        for i in 0..seed_count.min(self.population.len()) {
            self.rng = lcg_step(self.rng);
            self.population[i] = templates[self.rng as usize % templates.len()].clone();
        }

        // ── Macro-operator seeding (abstract thought feedback loop) ──
        // Inject learned macro-operators discovered across previous runs.
        // Each macro replaces a slot in the post-template region of the population.
        // Skipped entirely when `disable_macro_seeds` is set (cold benchmark mode).
        //
        // IMPORTANT: we seed the FIRST copy of each macro verbatim and apply
        // mild mutation (depth ≥ 2) only to subsequent copies. The previous
        // code called `template.mutate(&mut rng, 0)` on every slot, but
        // `mutate(rng, 0)` has `p = 1/(1+0) = 1.0` → it ALWAYS replaces the
        // entire tree with `random_expr(rng, 2)`. That meant the macro
        // seeding was effectively "replace a population slot with a fresh
        // small random expression" — the macro itself was never placed.
        // Cold runs (which skip this loop and keep their template-library
        // slots) actually got MORE informative seeds than macro-primed runs,
        // which explains why the distance-kernel-variant curriculum-transfer
        // test showed `cold_mse < M₁_mse` for a shape the macro should help.
        if !self.seed_macros.is_empty() && !self.config.disable_macro_seeds {
            let specialization_start = Instant::now();
            let budget = SpecializationBudget::for_population(
                self.config.population_size,
                self.seed_macros.len(),
            );
            let mut specialized_variants = Vec::new();
            let mut exact_fit_found = false;
            for template in &self.seed_macros {
                for variant in seed_macro_variants(template)
                    .into_iter()
                    .take(budget.max_variants_per_macro)
                {
                    if specialized_variants.len() >= budget.max_total_variants {
                        break;
                    }
                    let optimized =
                        specialize_seed_constants(&variant, &train, budget.optimization_iters);
                    let mse = compute_mse(&optimized, &train);
                    let complexity = optimized.complexity();
                    if mse.is_finite() && complexity <= self.config.max_complexity {
                        exact_fit_found |= mse < 1e-10;
                        specialized_variants
                            .push((mse + self.config.lambda * complexity as f64, optimized));
                    }
                }
                if exact_fit_found && specialized_variants.len() >= self.seed_macros.len() {
                    break;
                }
            }
            specialized_variants
                .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let macro_seed_count =
                (self.config.population_size / 6).min(specialized_variants.len().max(1));
            self.seed_specialization_stats = SeedSpecializationStats {
                variants_scored: specialized_variants.len(),
                variants_seeded: macro_seed_count,
                elapsed_ms: specialization_start.elapsed().as_millis(),
                exact_fit_found,
            };
            let macro_start = seed_count;
            for i in 0..macro_seed_count {
                let slot = macro_start + i;
                if slot >= self.population.len() {
                    break;
                }
                // Seed the best pre-specialized variants first. These variants
                // are ephemeral: they improve generation-0 transfer without
                // polluting the permanent macro grammar.
                let seeded = if let Some((_, expr)) = specialized_variants.get(i) {
                    expr.clone()
                } else {
                    self.rng = lcg_step(self.rng);
                    let macro_idx = self.rng as usize % self.seed_macros.len();
                    self.seed_macros[macro_idx].mutate(&mut self.rng, 2)
                };
                self.population[slot] = seeded;
            }
        }

        // Near-perfect-fit threshold: candidates with MSE below this bar
        // dominate the ranking over any imperfect fit, regardless of
        // complexity. This prevents the Occam penalty from suppressing the
        // exact answer in favor of a simpler approximation. Without this,
        // a perfect-fit `1/sqrt(n² + 1)` (mse=0, complexity 8, fitness 0.008)
        // loses to an approximate `0.794/n` (mse=1e-3, complexity 3, fitness
        // 0.004) even though the former is the ground truth. The threshold
        // is set well below what any reasonable approximation can achieve
        // for a genuinely mismatched structural form.
        const NEAR_PERFECT_MSE: f64 = 1e-10;

        for _gen in 0..self.config.generations {
            // Evaluate fitness for entire population. We now track MSE AND
            // scalar fitness separately so the ranking can apply hierarchical
            // comparison: near-perfect fits always beat imperfect ones.
            let mut scored: Vec<(usize, f64, f64)> = self
                .population
                .iter()
                .enumerate()
                .map(|(i, expr)| {
                    let mse = compute_mse(expr, &train);
                    let complexity = expr.complexity();
                    let (fitness, mse_kept) =
                        if mse.is_finite() && complexity <= self.config.max_complexity {
                            (mse + self.config.lambda * complexity as f64, mse)
                        } else {
                            (f64::MAX, f64::MAX)
                        };
                    (i, fitness, mse_kept)
                })
                .collect();

            // Hierarchical sort: candidates below NEAR_PERFECT_MSE dominate
            // candidates above it. Within each tier, sort by scalar fitness
            // (Occam-penalized). This lets a perfect-fit high-complexity
            // template beat any imperfect fit regardless of simplicity, while
            // preserving the Occam penalty as the Pareto tiebreaker for
            // imperfect candidates (where it's actually informative).
            scored.sort_by(|a, b| {
                let a_perfect = a.2 < NEAR_PERFECT_MSE;
                let b_perfect = b.2 < NEAR_PERFECT_MSE;
                match (a_perfect, b_perfect) {
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                    _ => a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal),
                }
            });

            // Record best fitness this generation (for benchmarking / convergence analysis)
            if let Some(&(_, best_fit, _)) = scored.first() {
                self.fitness_history.push(best_fit);
            }

            // ── Deduplicate: remove functionally identical formulas ───
            // Two formulas are "same" if they produce identical outputs on
            // the first 5 training points. Keep the simpler one.
            let mut fingerprints: Vec<(u64, usize)> = Vec::new();
            let mut unique_indices: Vec<usize> = Vec::new();
            let sample_points: Vec<f64> = train.iter().take(5).map(|(x, _)| *x).collect();

            for &(idx, _fit, _mse) in &scored {
                let fp = fingerprint_expr(&self.population[idx], &sample_points);
                if !fingerprints.iter().any(|(f, _)| *f == fp) {
                    fingerprints.push((fp, idx));
                    unique_indices.push(idx);
                }
            }

            // Elitism: keep top 10% of UNIQUE formulas
            let elite_count = (self.config.population_size / 10).min(unique_indices.len());
            let elite: Vec<Expr> = unique_indices
                .iter()
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
        let mut scored_pre: Vec<(f64, usize)> = self
            .population
            .iter()
            .enumerate()
            .map(|(i, expr)| {
                let mse = compute_mse(expr, &train);
                let c = expr.complexity();
                let fit = if mse.is_finite() && c <= self.config.max_complexity {
                    mse + self.config.lambda * c as f64
                } else {
                    f64::MAX
                };
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

        // Final scoring and return top-k. Uses the same hierarchical rule as
        // the GP loop: near-perfect fits dominate imperfect ones, Occam
        // fitness is the tiebreaker within each tier.
        let mut results: Vec<(f64, f64, usize)> = self
            .population
            .iter()
            .enumerate()
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

        results.sort_by(|a, b| {
            let a_perfect = a.1 < NEAR_PERFECT_MSE;
            let b_perfect = b.1 < NEAR_PERFECT_MSE;
            match (a_perfect, b_perfect) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal),
            }
        });

        // Deduplicate results by fingerprint (keep first = best fitness)
        let sample_pts: Vec<f64> = train.iter().take(5).map(|(x, _)| *x).collect();
        let mut seen_fps = Vec::new();
        let results: Vec<_> = results
            .into_iter()
            .filter(|(_, _, i)| {
                let fp = fingerprint_expr(&self.population[*i], &sample_pts);
                if seen_fps.contains(&fp) {
                    false
                } else {
                    seen_fps.push(fp);
                    true
                }
            })
            .collect();

        // ── Macro usage tracking (abstract thought causal analysis) ──
        // For each top-k formula that will be returned, check whether any
        // of the seed macros appear as a structural subtree. Counts are
        // exposed via `macro_usage()` for cold-vs-primed causal analysis.
        if !self.seed_macros.is_empty() {
            let top_indices: Vec<usize> = results.iter().take(top_k).map(|(_, _, i)| *i).collect();
            for &idx in &top_indices {
                let expr = &self.population[idx];
                for macro_expr in &self.seed_macros {
                    if contains_structural_match(expr, macro_expr) {
                        let key = macro_usage_key(macro_expr);
                        *self.macro_usage.entry(key).or_insert(0) += 1;
                    }
                }
            }
        }

        // Build the GP-discovered top-k.
        let mut candidates: Vec<Conjecture> = results
            .iter()
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
                    confidence: if *mse < 1e-6 {
                        0.8
                    } else if *mse < 1.0 {
                        0.5
                    } else {
                        0.1
                    },
                    macro_promotion_tier: MacroPromotionTier::RecurrentNumerical,
                    eml_compiled: None,
                    eml_metrics: None,
                    eml_verified_real: None,
                    eml_real_domain: None,
                    eml_verified_complex: None,
                    eml_constructive_compiled: None,
                    eml_constructive_metrics: None,
                    eml_verified_constructive_real: None,
                }
            })
            .collect();

        // Merge in log-space-pinned candidates. These are exp-wrapped
        // formulas discovered in log-space that may have lost tournament
        // selection in the original space. Evaluate each on the original
        // training data; if its fitness beats the GP top-k's worst, it
        // displaces the worst and we re-sort. This is the fix for the
        // `derangements`/`fubini`/`motzkin`/`stirling_sum` cluster —
        // super-exponential sequences where the log-space fit is the
        // natural answer but tournament selection in linear space
        // preferred polynomial approximants on partial data.
        for pinned_expr in &log_space_pinned {
            let mse = compute_mse(pinned_expr, &train);
            if !mse.is_finite() {
                continue;
            }
            let complexity = pinned_expr.complexity();
            let fitness = mse + self.config.lambda * complexity as f64;
            let conj = Conjecture {
                formula: pinned_expr.clone(),
                formula_str: format!("{}", pinned_expr),
                source: seq.name.clone(),
                domain: seq.domain,
                training_mse: mse,
                complexity,
                fitness,
                status: ConjectureStatus::Proposed,
                confidence: if mse < 1e-6 {
                    0.8
                } else if mse < 1.0 {
                    0.5
                } else {
                    0.1
                },
                macro_promotion_tier: MacroPromotionTier::RecurrentNumerical,
                eml_compiled: None,
                eml_metrics: None,
                eml_verified_real: None,
                eml_real_domain: None,
                eml_verified_complex: None,
                eml_constructive_compiled: None,
                eml_constructive_metrics: None,
                eml_verified_constructive_real: None,
            };
            candidates.push(conj);
        }
        // Re-sort by fitness ascending (lower = better) and take top_k.
        candidates.sort_by(|a, b| {
            a.fitness
                .partial_cmp(&b.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(top_k);
        candidates
    }

    fn tournament_select(&mut self, scored: &[(usize, f64, f64)]) -> usize {
        // Tournament selection uses the same hierarchical rule as the top-level
        // sort: a near-perfect candidate (mse < NEAR_PERFECT_MSE) always beats
        // an imperfect one, regardless of Occam-penalized fitness. Within each
        // tier, lower fitness wins. This keeps the dominance relation
        // consistent between `scored.sort_by` and tournament reproduction.
        const NEAR_PERFECT_MSE: f64 = 1e-10;
        let mut best_idx = 0;
        let mut best_fit = f64::MAX;
        let mut best_is_perfect = false;
        for _ in 0..self.config.tournament_size {
            self.rng = lcg_step(self.rng);
            let candidate = self.rng as usize % scored.len();
            let (cand_idx, cand_fit, cand_mse) = scored[candidate];
            let cand_is_perfect = cand_mse < NEAR_PERFECT_MSE;
            let wins = match (best_is_perfect, cand_is_perfect) {
                (false, true) => true,
                (true, false) => false,
                _ => cand_fit < best_fit,
            };
            if wins {
                best_fit = cand_fit;
                best_idx = cand_idx;
                best_is_perfect = cand_is_perfect;
            }
        }
        best_idx
    }
}
