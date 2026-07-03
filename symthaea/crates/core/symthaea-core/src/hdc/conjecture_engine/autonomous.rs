use super::*;

#[derive(Debug)]
pub struct DiscoveredConservation {
    pub name: String,
    pub expression: String,
    pub variance: f64,
    pub mean_value: f64,
    pub symbolically_proven: bool,
}

fn build_invariant_candidates(
    var_names: &[&str],
) -> Vec<(String, Box<dyn Fn(&[f64]) -> f64>, SymExpr)> {
    let mut candidates: Vec<(String, Box<dyn Fn(&[f64]) -> f64>, SymExpr)> = Vec::new();
    let ndim = var_names.len();

    if ndim >= 2 {
        let (v0, v1) = (var_names[0], var_names[1]);
        candidates.push((
            format!("{}² + {}²", v0, v1),
            Box::new(|s: &[f64]| s[0] * s[0] + s[1] * s[1]),
            SymExpr::Add(
                Box::new(SymExpr::Pow(Box::new(SymExpr::Var(v0.into())), 2.0)),
                Box::new(SymExpr::Pow(Box::new(SymExpr::Var(v1.into())), 2.0)),
            ),
        ));
        candidates.push((
            format!("{}·{}", v0, v1),
            Box::new(|s: &[f64]| s[0] * s[1]),
            SymExpr::Mul(
                Box::new(SymExpr::Var(v0.into())),
                Box::new(SymExpr::Var(v1.into())),
            ),
        ));
        candidates.push((
            format!("{}²", v0),
            Box::new(|s: &[f64]| s[0] * s[0]),
            SymExpr::Pow(Box::new(SymExpr::Var(v0.into())), 2.0),
        ));
        candidates.push((
            format!("{}²", v1),
            Box::new(|s: &[f64]| s[1] * s[1]),
            SymExpr::Pow(Box::new(SymExpr::Var(v1.into())), 2.0),
        ));
        candidates.push((
            format!("{0} - ln({0}) + {1} - ln({1})", v0, v1),
            Box::new(|s: &[f64]| {
                if s[0] > 0.0 && s[1] > 0.0 {
                    s[0] - s[0].ln() + s[1] - s[1].ln()
                } else {
                    f64::NAN
                }
            }),
            SymExpr::Add(
                Box::new(SymExpr::Add(
                    Box::new(SymExpr::Var(v0.into())),
                    Box::new(SymExpr::Neg(Box::new(SymExpr::Log(Box::new(
                        SymExpr::Var(v0.into()),
                    ))))),
                )),
                Box::new(SymExpr::Add(
                    Box::new(SymExpr::Var(v1.into())),
                    Box::new(SymExpr::Neg(Box::new(SymExpr::Log(Box::new(
                        SymExpr::Var(v1.into()),
                    ))))),
                )),
            ),
        ));
    }

    if ndim >= 4 {
        let (x, y, vx, vy) = (var_names[0], var_names[1], var_names[2], var_names[3]);
        candidates.push((
            "½(vx² + vy²)".into(),
            Box::new(|s: &[f64]| 0.5 * (s[2] * s[2] + s[3] * s[3])),
            SymExpr::Mul(
                Box::new(SymExpr::Const(0.5)),
                Box::new(SymExpr::Add(
                    Box::new(SymExpr::Pow(Box::new(SymExpr::Var(vx.into())), 2.0)),
                    Box::new(SymExpr::Pow(Box::new(SymExpr::Var(vy.into())), 2.0)),
                )),
            ),
        ));
        candidates.push((
            "L = x·vy - y·vx".into(),
            Box::new(|s: &[f64]| s[0] * s[3] - s[1] * s[2]),
            SymExpr::Add(
                Box::new(SymExpr::Mul(
                    Box::new(SymExpr::Var(x.into())),
                    Box::new(SymExpr::Var(vy.into())),
                )),
                Box::new(SymExpr::Neg(Box::new(SymExpr::Mul(
                    Box::new(SymExpr::Var(y.into())),
                    Box::new(SymExpr::Var(vx.into())),
                )))),
            ),
        ));
        candidates.push((
            "E = ½v² - 1/r".into(),
            Box::new(|s: &[f64]| {
                let r = (s[0] * s[0] + s[1] * s[1]).sqrt();
                if r > 1e-10 {
                    0.5 * (s[2] * s[2] + s[3] * s[3]) - 1.0 / r
                } else {
                    f64::NAN
                }
            }),
            SymExpr::Add(
                Box::new(SymExpr::Mul(
                    Box::new(SymExpr::Const(0.5)),
                    Box::new(SymExpr::Add(
                        Box::new(SymExpr::Pow(Box::new(SymExpr::Var(vx.into())), 2.0)),
                        Box::new(SymExpr::Pow(Box::new(SymExpr::Var(vy.into())), 2.0)),
                    )),
                )),
                Box::new(SymExpr::Neg(Box::new(SymExpr::Pow(
                    Box::new(SymExpr::Add(
                        Box::new(SymExpr::Pow(Box::new(SymExpr::Var(x.into())), 2.0)),
                        Box::new(SymExpr::Pow(Box::new(SymExpr::Var(y.into())), 2.0)),
                    )),
                    -0.5,
                )))),
            ),
        ));
        candidates.push((
            "x²+y²+vx²+vy²".into(),
            Box::new(|s: &[f64]| s[0] * s[0] + s[1] * s[1] + s[2] * s[2] + s[3] * s[3]),
            SymExpr::Add(
                Box::new(SymExpr::Add(
                    Box::new(SymExpr::Pow(Box::new(SymExpr::Var(x.into())), 2.0)),
                    Box::new(SymExpr::Pow(Box::new(SymExpr::Var(y.into())), 2.0)),
                )),
                Box::new(SymExpr::Add(
                    Box::new(SymExpr::Pow(Box::new(SymExpr::Var(vx.into())), 2.0)),
                    Box::new(SymExpr::Pow(Box::new(SymExpr::Var(vy.into())), 2.0)),
                )),
            ),
        ));
        candidates.push((
            "r² = x²+y²".into(),
            Box::new(|s: &[f64]| s[0] * s[0] + s[1] * s[1]),
            SymExpr::Add(
                Box::new(SymExpr::Pow(Box::new(SymExpr::Var(x.into())), 2.0)),
                Box::new(SymExpr::Pow(Box::new(SymExpr::Var(y.into())), 2.0)),
            ),
        ));
    }

    candidates
}

pub fn discover_conservation_laws(
    rhs: fn(&[f64], f64) -> Vec<f64>,
    initial_state: &[f64],
    dynamics: &[(&str, SymExpr)],
    var_names: &[&str],
    t_max: f64,
    dt: f64,
) -> Vec<DiscoveredConservation> {
    let ndim = initial_state.len();
    assert_eq!(var_names.len(), ndim);

    let (_times, states) = rk45_trajectory(rhs, initial_state, t_max, dt);
    let n_samples = 100.min(states.len());
    let step = states.len() / n_samples.max(1);
    let candidates = build_invariant_candidates(var_names);

    let mut results = Vec::new();
    for (name, eval_fn, sym_expr) in &candidates {
        let values: Vec<f64> = states
            .iter()
            .step_by(step.max(1))
            .take(n_samples)
            .map(|s| eval_fn(s))
            .filter(|v| v.is_finite())
            .collect();
        if values.len() < 10 {
            continue;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let proven = if var < 1e-6 * mean.abs().max(1.0) {
            verify_conservation_symbolic(sym_expr, dynamics).is_conserved
        } else {
            false
        };
        results.push(DiscoveredConservation {
            name: name.to_string(),
            expression: format!("{}", sym_expr),
            variance: var,
            mean_value: mean,
            symbolically_proven: proven,
        });
    }
    results.sort_by(|a, b| {
        a.variance
            .partial_cmp(&b.variance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

pub fn discover_conservation_laws_with_custom(
    rhs: fn(&[f64], f64) -> Vec<f64>,
    initial_state: &[f64],
    dynamics: &[(&str, SymExpr)],
    var_names: &[&str],
    custom_candidates: Vec<(String, Box<dyn Fn(&[f64]) -> f64>)>,
    t_max: f64,
    dt: f64,
) -> Vec<DiscoveredConservation> {
    let ndim = initial_state.len();
    assert_eq!(var_names.len(), ndim);

    let (_times, states) = rk45_trajectory(rhs, initial_state, t_max, dt);
    let n_samples = 200.min(states.len());
    let step = states.len() / n_samples.max(1);

    let mut results = Vec::new();
    let auto_candidates = build_invariant_candidates(var_names);
    for (name, eval_fn, sym_expr) in &auto_candidates {
        let values: Vec<f64> = states
            .iter()
            .step_by(step.max(1))
            .take(n_samples)
            .map(|s| eval_fn(s))
            .filter(|v| v.is_finite())
            .collect();
        if values.len() < 10 {
            continue;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let proven = if var < 1e-6 * mean.abs().max(1.0) {
            verify_conservation_symbolic(sym_expr, dynamics).is_conserved
        } else {
            false
        };
        results.push(DiscoveredConservation {
            name: name.clone(),
            expression: format!("{}", sym_expr),
            variance: var,
            mean_value: mean,
            symbolically_proven: proven,
        });
    }
    for (name, eval_fn) in &custom_candidates {
        let values: Vec<f64> = states
            .iter()
            .step_by(step.max(1))
            .take(n_samples)
            .map(|s| eval_fn(s))
            .filter(|v| v.is_finite())
            .collect();
        if values.len() < 10 {
            continue;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        results.push(DiscoveredConservation {
            name: name.clone(),
            expression: name.clone(),
            variance: var,
            mean_value: mean,
            symbolically_proven: false,
        });
    }
    results.sort_by(|a, b| {
        a.variance
            .partial_cmp(&b.variance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

fn random_expr_multivar(
    rng: &mut u64,
    max_depth: usize,
    var_names: &[&str],
    exclude_trig: bool,
) -> Expr {
    *rng = lcg_step(*rng);
    if max_depth == 0 || (*rng % 3 == 0 && max_depth < 3) {
        *rng = lcg_step(*rng);
        if *rng % 3 == 0 {
            *rng = lcg_step(*rng);
            Expr::Var(var_names[*rng as usize % var_names.len()].to_string())
        } else {
            let constants = [
                0.0,
                0.5,
                1.0,
                2.0,
                3.0,
                -1.0,
                -0.5,
                std::f64::consts::PI,
                std::f64::consts::E,
            ];
            *rng = lcg_step(*rng);
            Expr::Const(constants[*rng as usize % constants.len()])
        }
    } else {
        *rng = lcg_step(*rng);
        if *rng % 5 == 0 {
            let fns: &[UnaryFn] = if exclude_trig {
                &[UnaryFn::Sqrt, UnaryFn::Log]
            } else {
                &[UnaryFn::Sqrt, UnaryFn::Log, UnaryFn::Sin, UnaryFn::Cos]
            };
            *rng = lcg_step(*rng);
            Expr::Func(
                fns[*rng as usize % fns.len()],
                Box::new(random_expr_multivar(
                    rng,
                    max_depth - 1,
                    var_names,
                    exclude_trig,
                )),
            )
        } else {
            let ops = [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Pow];
            *rng = lcg_step(*rng);
            let op = ops[*rng as usize % ops.len()];
            Expr::BinOp(
                op,
                Box::new(random_expr_multivar(
                    rng,
                    max_depth - 1,
                    var_names,
                    exclude_trig,
                )),
                Box::new(random_expr_multivar(
                    rng,
                    max_depth - 1,
                    var_names,
                    exclude_trig,
                )),
            )
        }
    }
}

fn mutate_multivar(
    expr: &Expr,
    rng: &mut u64,
    depth: usize,
    var_names: &[&str],
    exclude_trig: bool,
) -> Expr {
    *rng = lcg_step(*rng);
    let p = 1.0 / (1.0 + depth as f64);
    if (*rng as f64 / u64::MAX as f64) < p {
        return random_expr_multivar(rng, 2, var_names, exclude_trig);
    }
    match expr {
        Expr::Var(_) | Expr::Const(_) => random_expr_multivar(rng, 1, var_names, exclude_trig),
        Expr::BinOp(op, l, r) => {
            *rng = lcg_step(*rng);
            if *rng % 2 == 0 {
                Expr::BinOp(
                    *op,
                    Box::new(mutate_multivar(l, rng, depth + 1, var_names, exclude_trig)),
                    r.clone(),
                )
            } else {
                Expr::BinOp(
                    *op,
                    l.clone(),
                    Box::new(mutate_multivar(r, rng, depth + 1, var_names, exclude_trig)),
                )
            }
        }
        Expr::Func(f, arg) => Expr::Func(
            *f,
            Box::new(mutate_multivar(
                arg,
                rng,
                depth + 1,
                var_names,
                exclude_trig,
            )),
        ),
        Expr::Sum(body, var) => Expr::Sum(
            Box::new(mutate_multivar(
                body,
                rng,
                depth + 1,
                var_names,
                exclude_trig,
            )),
            var.clone(),
        ),
    }
}

pub(crate) fn compute_trajectory_variance(
    expr: &Expr,
    trajectory: &[Vec<f64>],
    var_names: &[&str],
) -> f64 {
    let mut values = Vec::with_capacity(trajectory.len());
    let mut nan_count = 0usize;
    for state in trajectory {
        let bindings: Vec<(&str, f64)> = var_names
            .iter()
            .zip(state.iter())
            .map(|(name, val)| (*name, *val))
            .collect();
        let v = expr.eval(&bindings);
        if v.is_finite() {
            values.push(v);
        } else {
            nan_count += 1;
        }
    }

    let total = trajectory.len();
    if total == 0 || values.len() < 10 {
        return f64::MAX;
    }
    if nan_count * 4 > total {
        return f64::MAX;
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if !mean.is_finite() || mean.abs() > 1e15 {
        return f64::MAX;
    }
    let max_abs = values.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    if max_abs < 1e-5 {
        return f64::MAX;
    }
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    if !var.is_finite() {
        return f64::MAX;
    }
    let nan_fraction = nan_count as f64 / total as f64;
    var * (1.0 + 4.0 * nan_fraction)
}

#[derive(Debug, Clone)]
pub struct AutonomousInvariant {
    pub formula: Expr,
    pub formula_str: String,
    pub variance: f64,
    pub mean_value: f64,
    pub complexity: usize,
    pub symbolically_proven: bool,
}

pub fn compose_top_k_invariants(
    invariants: &[AutonomousInvariant],
    rhs: fn(&[f64], f64) -> Vec<f64>,
    initial_state: &[f64],
    var_names: &[&str],
    t_max: f64,
    dt: f64,
    top_k: usize,
) -> Option<AutonomousInvariant> {
    let k = invariants.len().min(top_k);
    if k < 2 {
        return None;
    }
    let (_t, states) = rk45_trajectory(rhs, initial_state, t_max, dt);
    let n_samples = 200.min(states.len());
    if n_samples < 20 {
        return None;
    }
    let step = states.len() / n_samples.max(1);
    let trajectory: Vec<Vec<f64>> = states
        .iter()
        .step_by(step.max(1))
        .take(n_samples)
        .cloned()
        .collect();

    let baseline: f64 = invariants[..k]
        .iter()
        .map(|inv| lie_derivative_variance(&inv.formula, rhs, &trajectory, var_names))
        .filter(|v| v.is_finite())
        .fold(f64::INFINITY, f64::min);

    const OPS: [BinOp; 4] = [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div];
    const SCALES: [f64; 6] = [0.25, 0.5, 1.0, 2.0, -0.5, -1.0];

    let mut best: Option<(Expr, f64)> = None;
    for i in 0..k {
        for j in 0..k {
            if i == j {
                continue;
            }
            for &op in &OPS {
                for &c in &SCALES {
                    let scaled_i = Expr::BinOp(
                        BinOp::Mul,
                        Box::new(Expr::Const(c)),
                        Box::new(invariants[i].formula.clone()),
                    );
                    let composite = Expr::BinOp(
                        op,
                        Box::new(scaled_i),
                        Box::new(invariants[j].formula.clone()),
                    );
                    let var = lie_derivative_variance(&composite, rhs, &trajectory, var_names);
                    if !var.is_finite() {
                        continue;
                    }
                    if best.as_ref().is_none_or(|(_, v)| var < *v) {
                        best = Some((composite, var));
                    }
                }
            }
        }
    }

    let (expr, variance) = best?;
    if variance >= baseline {
        return None;
    }

    let mut sum = 0.0f64;
    let mut n = 0usize;
    for state in &trajectory {
        let bindings: Vec<(&str, f64)> = var_names
            .iter()
            .zip(state.iter())
            .map(|(name, val)| (*name, *val))
            .collect();
        let v = expr.eval(&bindings);
        if v.is_finite() {
            sum += v;
            n += 1;
        }
    }
    let mean_value = if n > 0 { sum / n as f64 } else { f64::NAN };
    let formula_str = format!("{}", expr);
    let complexity = expr.complexity();

    Some(AutonomousInvariant {
        formula: expr,
        formula_str,
        variance,
        mean_value,
        complexity,
        symbolically_proven: false,
    })
}

pub fn discover_invariants_autonomous(
    rhs: fn(&[f64], f64) -> Vec<f64>,
    initial_state: &[f64],
    var_names: &[&str],
    dynamics: Option<&[(&str, SymExpr)]>,
    config: &RegressorConfig,
    t_max: f64,
    dt: f64,
) -> Vec<AutonomousInvariant> {
    discover_invariants_autonomous_with_seed_templates(
        rhs,
        initial_state,
        var_names,
        dynamics,
        config,
        t_max,
        dt,
        &[],
    )
}

pub fn discover_invariants_autonomous_with_seed_templates(
    rhs: fn(&[f64], f64) -> Vec<f64>,
    initial_state: &[f64],
    var_names: &[&str],
    dynamics: Option<&[(&str, SymExpr)]>,
    config: &RegressorConfig,
    t_max: f64,
    dt: f64,
    extra_seed_templates: &[Expr],
) -> Vec<AutonomousInvariant> {
    let ndim = initial_state.len();
    assert_eq!(var_names.len(), ndim);

    let sample_one = |ic: &[f64]| -> Vec<Vec<f64>> {
        let (_t, states) = rk45_trajectory(rhs, ic, t_max, dt);
        let n_samples = 200.min(states.len());
        let step = states.len() / n_samples.max(1);
        states
            .iter()
            .step_by(step.max(1))
            .take(n_samples)
            .cloned()
            .collect()
    };
    let sampled = sample_one(initial_state);
    if sampled.len() < 20 {
        return Vec::new();
    }

    let mut sampled_orbits: Vec<Vec<Vec<f64>>> = vec![sampled.clone()];
    let diverse_n = config.diverse_trajectory_count.max(1);
    if diverse_n > 1 {
        let mut prng = config.seed.wrapping_mul(0x517cc1b727220a95);
        for _ in 1..diverse_n {
            prng = lcg_step(prng);
            let mut ic = initial_state.to_vec();
            for x in &mut ic {
                prng = lcg_step(prng);
                let u = (prng as f64 / u64::MAX as f64) * 2.0 - 1.0;
                let scale = x.abs().max(0.05);
                *x += u * 0.1 * scale;
            }
            let extra = sample_one(&ic);
            if extra.len() >= 20 {
                sampled_orbits.push(extra);
            }
        }
    }

    let pop_size = config.population_size;
    let generations = config.generations;
    let max_depth = config.max_depth;
    let max_complexity = config.max_complexity;

    let mut rng = config.seed;
    let exclude_trig = config.exclude_trig;
    let mut population: Vec<Expr> = (0..pop_size)
        .map(|_| {
            rng = lcg_step(rng);
            random_expr_multivar(&mut rng, max_depth.min(3), var_names, exclude_trig)
        })
        .collect();

    let mut seed_templates = build_invariant_templates(var_names);
    let mut seen_seed_templates: std::collections::HashSet<String> =
        seed_templates.iter().map(macro_usage_key).collect();
    for template in extra_seed_templates {
        if !expr_uses_only_vars(template, var_names) {
            continue;
        }
        let canonical = macro_usage_key(template);
        if seen_seed_templates.insert(canonical) {
            seed_templates.push(template.clone());
        }
    }
    for (i, template) in seed_templates.into_iter().enumerate() {
        if i < population.len() / 4 {
            population[i] = template;
        }
    }

    let pinned_priors: Vec<Expr> = extra_seed_templates
        .iter()
        .filter(|t| expr_uses_only_vars(t, var_names))
        .cloned()
        .collect();
    let pinned_prior_keys: Vec<String> = pinned_priors.iter().map(macro_usage_key).collect();
    let fragment_bonus = config.prior_fragment_bonus;
    let fragment_active =
        fragment_bonus > 0.0 && fragment_bonus < 1.0 && !pinned_prior_keys.is_empty();

    let orth_threshold_sin = (1.0_f64 - config.orthogonality_threshold.powi(2))
        .max(0.0)
        .sqrt();
    let orth_active = config.orthogonality_penalty > 1.0 && !config.known_invariants.is_empty();
    let orth_probe_points: Vec<Vec<f64>> = if orth_active {
        (0..8)
            .filter_map(|k| {
                let idx = (k * sampled.len()) / 8;
                sampled.get(idx).cloned()
            })
            .collect()
    } else {
        Vec::new()
    };
    let orth_bases: Vec<Vec<Vec<f64>>> = if orth_active {
        orth_probe_points
            .iter()
            .map(|state| {
                let grads: Vec<Vec<f64>> = config
                    .known_invariants
                    .iter()
                    .map(|inv| fd_gradient(inv, state, var_names))
                    .filter(|g| g.iter().all(|x| x.is_finite()))
                    .collect();
                gram_schmidt(&grads)
            })
            .collect()
    } else {
        Vec::new()
    };

    let nt_test_points: Vec<Vec<f64>> = {
        let mut points = Vec::with_capacity(8);
        for i in [0, sampled.len() / 4, sampled.len() / 2, sampled.len() - 1] {
            if i < sampled.len() {
                points.push(sampled[i].clone());
            }
        }
        let perturbations: [&[f64]; 4] = [
            &[1.7, 0.4, 0.6, 0.9][..],
            &[0.2, 1.5, 0.7, 0.3][..],
            &[1.1, 1.1, 1.1, 1.1][..],
            &[0.5, 0.5, 0.5, 0.5][..],
        ];
        for p in &perturbations {
            let mut state = vec![0.0; ndim];
            for i in 0..ndim {
                state[i] = p.get(i).copied().unwrap_or(1.0);
            }
            points.push(state);
        }
        points
    };

    for _gen in 0..generations {
        let fitnesses: Vec<f64> = population
            .iter()
            .map(|expr| {
                if expr.complexity() > max_complexity || expr.complexity() < 3 {
                    return f64::MAX;
                }

                let mut values = Vec::with_capacity(nt_test_points.len());
                for state in &nt_test_points {
                    let bindings: Vec<(&str, f64)> = var_names
                        .iter()
                        .zip(state.iter())
                        .map(|(name, val)| (*name, *val))
                        .collect();
                    let v = expr.eval(&bindings);
                    if !v.is_finite() {
                        return f64::MAX;
                    }
                    values.push(v);
                }
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let var =
                    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
                let spread = var.sqrt();
                let scale = mean.abs().max(1e-6);
                if spread / scale < 1e-6 {
                    return f64::MAX;
                }

                let mut worst = 0.0_f64;
                for orbit in &sampled_orbits {
                    let v = if config.use_lie_fitness {
                        lie_derivative_variance(expr, rhs, orbit, var_names)
                    } else {
                        compute_trajectory_variance(expr, orbit, var_names)
                    };
                    if !v.is_finite() {
                        return f64::MAX;
                    }
                    if v > worst {
                        worst = v;
                    }
                }
                if fragment_active {
                    let matches = count_prior_subtrees(expr, &pinned_prior_keys);
                    if matches > 0 {
                        worst *= fragment_bonus.powi(matches as i32);
                    }
                }
                if orth_active {
                    let mut orth_sum = 0.0;
                    let mut orth_n = 0usize;
                    for (state, basis) in orth_probe_points.iter().zip(orth_bases.iter()) {
                        if basis.is_empty() {
                            continue;
                        }
                        let g = fd_gradient(expr, state, var_names);
                        if g.iter().all(|x| x.is_finite()) {
                            orth_sum += orthogonal_fraction(&g, basis);
                            orth_n += 1;
                        }
                    }
                    if orth_n > 0 {
                        let mean_orth = orth_sum / orth_n as f64;
                        if mean_orth < orth_threshold_sin {
                            return f64::MAX;
                        }
                    }
                }
                worst
            })
            .collect();

        let mut new_pop = Vec::with_capacity(pop_size);
        let mut ranked: Vec<(usize, f64)> =
            fitnesses.iter().enumerate().map(|(i, f)| (i, *f)).collect();
        ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        for &(idx, _) in ranked.iter().take(5) {
            new_pop.push(population[idx].clone());
        }

        let max_pinned = pop_size / 4;
        let mut pinned_added = 0;
        for prior in &pinned_priors {
            if pinned_added >= max_pinned {
                break;
            }
            let key = macro_usage_key(prior);
            let present = new_pop.iter().any(|e| macro_usage_key(e) == key);
            if !present && new_pop.len() < pop_size {
                new_pop.push(prior.clone());
                pinned_added += 1;
            }
        }

        while new_pop.len() < pop_size {
            rng = lcg_step(rng);
            let a = rng as usize % pop_size;
            rng = lcg_step(rng);
            let b = rng as usize % pop_size;
            let winner = if fitnesses[a] < fitnesses[b] { a } else { b };

            rng = lcg_step(rng);
            let roll = rng as f64 / u64::MAX as f64;
            let child = if pinned_priors.len() >= 2 && roll < config.prior_composition_rate {
                rng = lcg_step(rng);
                let i = (rng as usize) % pinned_priors.len();
                rng = lcg_step(rng);
                let mut j = (rng as usize) % pinned_priors.len();
                if j == i {
                    j = (j + 1) % pinned_priors.len();
                }
                rng = lcg_step(rng);
                let ops = [BinOp::Add, BinOp::Sub, BinOp::Mul];
                let op = ops[(rng as usize) % ops.len()];
                Expr::BinOp(
                    op,
                    Box::new(pinned_priors[i].clone()),
                    Box::new(pinned_priors[j].clone()),
                )
            } else if roll < config.mutation_rate {
                mutate_multivar(&population[winner], &mut rng, 0, var_names, exclude_trig)
            } else {
                rng = lcg_step(rng);
                let c = rng as usize % pop_size;
                rng = lcg_step(rng);
                let d = rng as usize % pop_size;
                let other = if fitnesses[c] < fitnesses[d] { c } else { d };
                crossover(&population[winner], &population[other], &mut rng)
            };
            new_pop.push(child);
        }
        population = new_pop;
    }

    let mut scored: Vec<(usize, f64, f64)> = population
        .iter()
        .enumerate()
        .map(|(i, expr)| {
            if matches!(simplify(expr), Expr::Const(_)) {
                return (i, f64::MAX, 0.0);
            }
            let mut nt_vals = Vec::with_capacity(nt_test_points.len());
            for state in &nt_test_points {
                let bindings: Vec<(&str, f64)> = var_names
                    .iter()
                    .zip(state.iter())
                    .map(|(name, val)| (*name, *val))
                    .collect();
                let v = expr.eval(&bindings);
                if !v.is_finite() {
                    return (i, f64::MAX, 0.0);
                }
                nt_vals.push(v);
            }
            let nt_mean = nt_vals.iter().sum::<f64>() / nt_vals.len() as f64;
            let nt_spread = (nt_vals.iter().map(|v| (v - nt_mean).powi(2)).sum::<f64>()
                / nt_vals.len() as f64)
                .sqrt();
            let nt_scale = nt_mean.abs().max(1e-6);
            if nt_spread / nt_scale < 1e-6 {
                return (i, f64::MAX, 0.0);
            }

            if orth_active {
                let mut orth_sum = 0.0_f64;
                let mut orth_n = 0usize;
                for (state, basis) in orth_probe_points.iter().zip(orth_bases.iter()) {
                    if basis.is_empty() {
                        continue;
                    }
                    let g = fd_gradient(expr, state, var_names);
                    if g.iter().all(|x| x.is_finite()) {
                        orth_sum += orthogonal_fraction(&g, basis);
                        orth_n += 1;
                    }
                }
                if orth_n > 0 {
                    let mean_orth = orth_sum / orth_n as f64;
                    if mean_orth < orth_threshold_sin {
                        return (i, f64::MAX, 0.0);
                    }
                }
            }

            let var = sampled_orbits
                .iter()
                .map(|orbit| {
                    if config.use_lie_fitness {
                        lie_derivative_variance(expr, rhs, orbit, var_names)
                    } else {
                        compute_trajectory_variance(expr, orbit, var_names)
                    }
                })
                .filter(|v| v.is_finite())
                .fold(0.0_f64, f64::max);
            let mean = {
                let vals: Vec<f64> = sampled
                    .iter()
                    .map(|s| {
                        let bindings: Vec<(&str, f64)> = var_names
                            .iter()
                            .zip(s.iter())
                            .map(|(n, v)| (*n, *v))
                            .collect();
                        expr.eval(&bindings)
                    })
                    .filter(|v| v.is_finite())
                    .collect();
                if vals.is_empty() {
                    0.0
                } else {
                    vals.iter().sum::<f64>() / vals.len() as f64
                }
            };
            (i, var, mean)
        })
        .filter(|(_, var, _)| var.is_finite() && *var < 1e10)
        .collect();

    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut results = Vec::new();
    let mut seen_means: Vec<f64> = Vec::new();
    for &(idx, var, mean) in scored.iter().take(100) {
        let expr_simplified = simplify(&population[idx]);
        if matches!(expr_simplified, Expr::Const(_)) {
            continue;
        }

        let test_states: Vec<Vec<(&str, f64)>> = vec![
            var_names
                .iter()
                .enumerate()
                .map(|(i, n)| (*n, if i == 0 { 1.0 } else { 0.5 }))
                .collect(),
            var_names
                .iter()
                .enumerate()
                .map(|(i, n)| (*n, if i == 0 { 0.3 } else { 0.9 }))
                .collect(),
        ];
        let v0 = expr_simplified.eval(&test_states[0]);
        let v1 = expr_simplified.eval(&test_states[1]);
        if v0.is_finite() && v1.is_finite() && (v0 - v1).abs() < 1e-8 * v0.abs().max(1.0) {
            continue;
        }
        if !v0.is_finite() || !v1.is_finite() {
            continue;
        }

        let is_duplicate = seen_means
            .iter()
            .any(|m| (m - mean).abs() < mean.abs() * 0.01 + 1e-6);
        if is_duplicate {
            continue;
        }
        seen_means.push(mean);

        let expr = simplify(&population[idx]);
        let proven = if let Some(dyn_rules) = dynamics {
            if var < 1e-4 * mean.abs().max(1.0) {
                if let Some(sym) = expr_to_sym(&expr) {
                    verify_conservation_symbolic(&sym, dyn_rules).is_conserved
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        };

        results.push(AutonomousInvariant {
            formula_str: format!("{}", expr),
            formula: expr.clone(),
            variance: var,
            mean_value: mean,
            complexity: expr.complexity(),
            symbolically_proven: proven,
        });

        if results.len() >= 10 {
            break;
        }
    }

    results
}

fn build_invariant_templates(var_names: &[&str]) -> Vec<Expr> {
    let mut templates = Vec::new();
    let ndim = var_names.len();

    if ndim >= 2 {
        let mut sum = Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var(var_names[0].into())),
            Box::new(Expr::Const(2.0)),
        );
        for v in &var_names[1..] {
            sum = Expr::BinOp(
                BinOp::Add,
                Box::new(sum),
                Box::new(Expr::BinOp(
                    BinOp::Pow,
                    Box::new(Expr::Var(v.to_string())),
                    Box::new(Expr::Const(2.0)),
                )),
            );
        }
        templates.push(sum);
    }

    if ndim >= 4 {
        templates.push(Expr::BinOp(
            BinOp::Sub,
            Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Var(var_names[0].into())),
                Box::new(Expr::Var(var_names[3].into())),
            )),
            Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Var(var_names[1].into())),
                Box::new(Expr::Var(var_names[2].into())),
            )),
        ));

        let v2 = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var(var_names[2].into())),
                Box::new(Expr::Const(2.0)),
            )),
            Box::new(Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var(var_names[3].into())),
                Box::new(Expr::Const(2.0)),
            )),
        );
        let r = Expr::Func(
            UnaryFn::Sqrt,
            Box::new(Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::BinOp(
                    BinOp::Pow,
                    Box::new(Expr::Var(var_names[0].into())),
                    Box::new(Expr::Const(2.0)),
                )),
                Box::new(Expr::BinOp(
                    BinOp::Pow,
                    Box::new(Expr::Var(var_names[1].into())),
                    Box::new(Expr::Const(2.0)),
                )),
            )),
        );
        templates.push(Expr::BinOp(
            BinOp::Sub,
            Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Const(0.5)),
                Box::new(v2),
            )),
            Box::new(Expr::BinOp(
                BinOp::Div,
                Box::new(Expr::Const(1.0)),
                Box::new(r),
            )),
        ));

        let var0_sq = Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var(var_names[0].into())),
            Box::new(Expr::Const(2.0)),
        );
        let var1_sq = Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var(var_names[1].into())),
            Box::new(Expr::Const(2.0)),
        );
        let half_p2 = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Const(0.5)),
            Box::new(Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::BinOp(
                    BinOp::Pow,
                    Box::new(Expr::Var(var_names[2].into())),
                    Box::new(Expr::Const(2.0)),
                )),
                Box::new(Expr::BinOp(
                    BinOp::Pow,
                    Box::new(Expr::Var(var_names[3].into())),
                    Box::new(Expr::Const(2.0)),
                )),
            )),
        );
        let half_q2 = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Const(0.5)),
            Box::new(Expr::BinOp(
                BinOp::Add,
                Box::new(var0_sq.clone()),
                Box::new(var1_sq),
            )),
        );
        let coupling = Expr::BinOp(
            BinOp::Mul,
            Box::new(var0_sq),
            Box::new(Expr::Var(var_names[1].into())),
        );
        let cubic = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Const(-1.0 / 3.0)),
            Box::new(Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var(var_names[1].into())),
                Box::new(Expr::Const(3.0)),
            )),
        );
        templates.push(Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::BinOp(
                    BinOp::Add,
                    Box::new(half_p2),
                    Box::new(half_q2),
                )),
                Box::new(coupling),
            )),
            Box::new(cubic),
        ));

        let pos_sq = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var(var_names[0].into())),
                Box::new(Expr::Const(2.0)),
            )),
            Box::new(Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var(var_names[1].into())),
                Box::new(Expr::Const(2.0)),
            )),
        );
        let vel_sq = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var(var_names[2].into())),
                Box::new(Expr::Const(2.0)),
            )),
            Box::new(Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var(var_names[3].into())),
                Box::new(Expr::Const(2.0)),
            )),
        );
        templates.push(Expr::BinOp(
            BinOp::Sub,
            Box::new(pos_sq.clone()),
            Box::new(vel_sq.clone()),
        ));

        let half_vel2 = Expr::BinOp(BinOp::Mul, Box::new(Expr::Const(0.5)), Box::new(vel_sq));
        let cross_qq = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var(var_names[0].into())),
            Box::new(Expr::Var(var_names[1].into())),
        );
        templates.push(Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::BinOp(
                BinOp::Add,
                Box::new(half_vel2),
                Box::new(pos_sq),
            )),
            Box::new(cross_qq),
        ));
    }

    for v in var_names {
        templates.push(Expr::BinOp(
            BinOp::Sub,
            Box::new(Expr::Var(v.to_string())),
            Box::new(Expr::Func(UnaryFn::Log, Box::new(Expr::Var(v.to_string())))),
        ));
    }
    if ndim >= 2 {
        let (a, b) = (var_names[0], var_names[1]);
        templates.push(Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::BinOp(
                BinOp::Sub,
                Box::new(Expr::Var(a.into())),
                Box::new(Expr::Func(UnaryFn::Log, Box::new(Expr::Var(a.into())))),
            )),
            Box::new(Expr::BinOp(
                BinOp::Sub,
                Box::new(Expr::Var(b.into())),
                Box::new(Expr::Func(UnaryFn::Log, Box::new(Expr::Var(b.into())))),
            )),
        ));
        templates.push(Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Func(UnaryFn::Log, Box::new(Expr::Var(a.into())))),
            Box::new(Expr::Func(UnaryFn::Log, Box::new(Expr::Var(b.into())))),
        ));
        templates.push(Expr::BinOp(
            BinOp::Sub,
            Box::new(Expr::Func(UnaryFn::Log, Box::new(Expr::Var(a.into())))),
            Box::new(Expr::Func(UnaryFn::Log, Box::new(Expr::Var(b.into())))),
        ));
        templates.push(Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Var(a.into())),
                Box::new(Expr::Func(UnaryFn::Log, Box::new(Expr::Var(a.into())))),
            )),
            Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Var(b.into())),
                Box::new(Expr::Func(UnaryFn::Log, Box::new(Expr::Var(b.into())))),
            )),
        ));
    }

    for v in var_names {
        templates.push(Expr::Var(v.to_string()));
        templates.push(Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var(v.to_string())),
            Box::new(Expr::Const(2.0)),
        ));
    }

    templates
}

#[derive(Debug, Clone, PartialEq)]
pub enum SystemClassification {
    Conservative {
        num_invariants: usize,
        best_variance: f64,
    },
    Dissipative {
        best_variance: f64,
        lyapunov_candidate: Option<String>,
    },
    IntegrabilityTransition {
        low_energy_invariants: usize,
        high_energy_invariants: usize,
    },
}

#[derive(Debug)]
pub struct SystemAnalysis {
    pub classification: SystemClassification,
    pub invariants: Vec<AutonomousInvariant>,
    pub report: String,
}

pub fn analyze_system_autonomous(
    rhs: fn(&[f64], f64) -> Vec<f64>,
    initial_state: &[f64],
    var_names: &[&str],
    dynamics: Option<&[(&str, SymExpr)]>,
    config: &RegressorConfig,
    t_max: f64,
    dt: f64,
) -> SystemAnalysis {
    let invariants =
        discover_invariants_autonomous(rhs, initial_state, var_names, dynamics, config, t_max, dt);

    let conservation_threshold = 1e-6;
    let conserved: Vec<&AutonomousInvariant> = invariants
        .iter()
        .filter(|i| {
            let rel_var = i.variance / (i.mean_value * i.mean_value).max(1e-10);
            rel_var < conservation_threshold
        })
        .collect();

    if !conserved.is_empty() {
        let best_var = conserved[0].variance;
        let mut report = format!(
            "CONSERVATIVE SYSTEM: {} invariant(s) found\n",
            conserved.len()
        );
        for inv in &conserved {
            let proven = if inv.symbolically_proven {
                " [PROVEN]"
            } else {
                ""
            };
            report += &format!(
                "  {} (var={:.2e}){}\n",
                inv.formula_str, inv.variance, proven
            );
        }
        SystemAnalysis {
            classification: SystemClassification::Conservative {
                num_invariants: conserved.len(),
                best_variance: best_var,
            },
            invariants,
            report,
        }
    } else {
        let best_var = invariants.first().map(|i| i.variance).unwrap_or(f64::MAX);
        let lyapunov = find_lyapunov_candidate(rhs, initial_state, var_names, t_max, dt);
        let mut report = "DISSIPATIVE SYSTEM: no conservation law found\n".to_string();
        report += &format!(
            "  Best candidate variance: {:.2e} (threshold: {:.2e})\n",
            best_var, conservation_threshold
        );
        if let Some(ref ly) = lyapunov {
            report += &format!("  Lyapunov function candidate: {}\n", ly);
        } else {
            report += "  No Lyapunov function found in candidate set\n";
        }

        SystemAnalysis {
            classification: SystemClassification::Dissipative {
                best_variance: best_var,
                lyapunov_candidate: lyapunov,
            },
            invariants,
            report,
        }
    }
}

fn find_lyapunov_candidate(
    rhs: fn(&[f64], f64) -> Vec<f64>,
    initial_state: &[f64],
    var_names: &[&str],
    t_max: f64,
    dt: f64,
) -> Option<String> {
    let (_times, states) = rk45_trajectory(rhs, initial_state, t_max, dt);
    if states.len() < 100 {
        return None;
    }

    let v_values: Vec<f64> = states
        .iter()
        .map(|s| s.iter().map(|x| x * x).sum::<f64>())
        .collect();
    let last_quarter = &v_values[v_values.len() * 3 / 4..];
    let mean_last = last_quarter.iter().sum::<f64>() / last_quarter.len() as f64;
    let var_last = last_quarter
        .iter()
        .map(|v| (v - mean_last).powi(2))
        .sum::<f64>()
        / last_quarter.len() as f64;

    if mean_last.is_finite() && var_last < mean_last * mean_last * 0.5 {
        let names: Vec<&str> = var_names.to_vec();
        return Some(format!(
            "Σ{}ᵢ² → {:.2} (bounded attractor, not Lyapunov)",
            if names.len() <= 3 {
                names.join("²+")
            } else {
                "x".into()
            },
            mean_last
        ));
    }

    None
}
