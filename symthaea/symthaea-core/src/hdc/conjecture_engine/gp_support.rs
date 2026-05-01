// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use super::{BinOp, Expr, UnaryFn};

pub(crate) fn contains_structural_match(haystack: &Expr, needle: &Expr) -> bool {
    if shape_equal(haystack, needle) {
        return true;
    }
    match haystack {
        Expr::Var(_) | Expr::Const(_) => false,
        Expr::BinOp(_, l, r) => {
            contains_structural_match(l, needle) || contains_structural_match(r, needle)
        }
        Expr::Func(_, arg) => contains_structural_match(arg, needle),
        Expr::Sum(body, _) => contains_structural_match(body, needle),
    }
}

pub(crate) fn expr_uses_only_vars(expr: &Expr, allowed_vars: &[&str]) -> bool {
    match expr {
        Expr::Var(name) => allowed_vars.iter().any(|allowed| *allowed == name),
        Expr::Const(_) => true,
        Expr::BinOp(_, l, r) => {
            expr_uses_only_vars(l, allowed_vars) && expr_uses_only_vars(r, allowed_vars)
        }
        Expr::Func(_, arg) => expr_uses_only_vars(arg, allowed_vars),
        Expr::Sum(body, var) => {
            allowed_vars.iter().any(|allowed| *allowed == var)
                && expr_uses_only_vars(body, allowed_vars)
        }
    }
}

pub(crate) fn macro_usage_key(expr: &Expr) -> String {
    #[cfg(feature = "abstract_thought")]
    {
        crate::hdc::abstract_thought::expr_canonical_string(expr)
    }
    #[cfg(not(feature = "abstract_thought"))]
    {
        format!("{}", expr)
    }
}

pub(crate) fn lie_derivative_variance(
    expr: &Expr,
    rhs: fn(&[f64], f64) -> Vec<f64>,
    trajectory: &[Vec<f64>],
    var_names: &[&str],
) -> f64 {
    let mut lie_vals = Vec::with_capacity(trajectory.len());
    let mut grad_sq_vals = Vec::with_capacity(trajectory.len());

    for state in trajectory {
        let grad = fd_gradient(expr, state, var_names);
        if grad.iter().any(|g| !g.is_finite()) {
            return f64::MAX;
        }
        let flow = rhs(state, 0.0);
        if flow.len() != grad.len() {
            return f64::MAX;
        }
        let lie: f64 = grad.iter().zip(flow.iter()).map(|(g, f)| g * f).sum();
        let grad_sq: f64 = grad.iter().map(|g| g * g).sum();
        if !lie.is_finite() || !grad_sq.is_finite() {
            return f64::MAX;
        }
        lie_vals.push(lie);
        grad_sq_vals.push(grad_sq);
    }

    if lie_vals.is_empty() {
        return f64::MAX;
    }

    let mean_lie_sq = lie_vals.iter().map(|v| v * v).sum::<f64>() / lie_vals.len() as f64;
    let mean_grad_sq = grad_sq_vals.iter().sum::<f64>() / grad_sq_vals.len() as f64;
    const MIN_GRADIENT_MAG_SQ: f64 = 1e-12;
    if mean_grad_sq < MIN_GRADIENT_MAG_SQ {
        return f64::MAX;
    }
    mean_lie_sq / mean_grad_sq
}

pub(crate) fn fd_gradient(expr: &Expr, state: &[f64], var_names: &[&str]) -> Vec<f64> {
    const EPS: f64 = 1e-5;
    let mut grad = Vec::with_capacity(var_names.len());
    for i in 0..var_names.len() {
        let mut plus = state.to_vec();
        let mut minus = state.to_vec();
        plus[i] += EPS;
        minus[i] -= EPS;
        let bindings_plus: Vec<(&str, f64)> = var_names
            .iter()
            .zip(plus.iter())
            .map(|(n, v)| (*n, *v))
            .collect();
        let bindings_minus: Vec<(&str, f64)> = var_names
            .iter()
            .zip(minus.iter())
            .map(|(n, v)| (*n, *v))
            .collect();
        let f_plus = expr.eval(&bindings_plus);
        let f_minus = expr.eval(&bindings_minus);
        grad.push((f_plus - f_minus) / (2.0 * EPS));
    }
    grad
}

pub(crate) fn gram_schmidt(vectors: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for v in vectors {
        let mut u = v.clone();
        for b in &basis {
            let dot: f64 = u.iter().zip(b.iter()).map(|(a, c)| a * c).sum();
            for i in 0..u.len() {
                u[i] -= dot * b[i];
            }
        }
        let norm: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 && norm.is_finite() {
            for x in &mut u {
                *x /= norm;
            }
            basis.push(u);
        }
    }
    basis
}

pub(crate) fn orthogonal_fraction(g: &[f64], basis: &[Vec<f64>]) -> f64 {
    let total_sq: f64 = g.iter().map(|x| x * x).sum();
    if total_sq < 1e-30 {
        return 1.0;
    }
    let mut parallel_sq = 0.0;
    for b in basis {
        let dot: f64 = g.iter().zip(b.iter()).map(|(a, c)| a * c).sum();
        parallel_sq += dot * dot;
    }
    let orth_sq = (total_sq - parallel_sq).max(0.0);
    (orth_sq / total_sq).sqrt()
}

pub(crate) fn count_prior_subtrees(expr: &Expr, prior_keys: &[String]) -> usize {
    if prior_keys.is_empty() {
        return 0;
    }
    let mut count = 0;
    count_prior_inner(expr, prior_keys, &mut count);
    count
}

fn count_prior_inner(expr: &Expr, prior_keys: &[String], count: &mut usize) {
    let key = macro_usage_key(expr);
    if prior_keys.iter().any(|k| k == &key) {
        *count += 1;
        return;
    }
    match expr {
        Expr::BinOp(_, l, r) => {
            count_prior_inner(l, prior_keys, count);
            count_prior_inner(r, prior_keys, count);
        }
        Expr::Func(_, arg) => count_prior_inner(arg, prior_keys, count),
        Expr::Sum(body, _) => count_prior_inner(body, prior_keys, count),
        _ => {}
    }
}

fn shape_equal(a: &Expr, b: &Expr) -> bool {
    match (a, b) {
        (Expr::Var(na), Expr::Var(nb)) => na == nb,
        (Expr::Const(_), Expr::Const(_)) => true,
        (Expr::BinOp(opa, la, ra), Expr::BinOp(opb, lb, rb)) => {
            opa == opb && shape_equal(la, lb) && shape_equal(ra, rb)
        }
        (Expr::Func(fa, aa), Expr::Func(fb, ab)) => fa == fb && shape_equal(aa, ab),
        (Expr::Sum(ba, va), Expr::Sum(bb, vb)) => va == vb && shape_equal(ba, bb),
        _ => false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MacroRole {
    PolynomialPower,
    Reciprocal,
    DistanceKernelCore,
    AffineShift,
    UnaryWrapped,
    MultivariateInvariant,
    Unknown,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SpecializationBudget {
    pub(crate) max_variants_per_macro: usize,
    pub(crate) max_total_variants: usize,
    pub(crate) optimization_iters: usize,
}

impl SpecializationBudget {
    pub(crate) fn for_population(population_size: usize, macro_count: usize) -> Self {
        Self {
            max_variants_per_macro: 8,
            max_total_variants: (population_size / 3)
                .max(macro_count.saturating_mul(2))
                .max(8),
            optimization_iters: 100,
        }
    }
}

pub(crate) fn seed_macro_variants(template: &Expr) -> Vec<Expr> {
    let mut variants = Vec::new();
    push_unique_variant(&mut variants, template.clone());

    match classify_macro_role(template) {
        MacroRole::PolynomialPower | MacroRole::DistanceKernelCore => {
            push_distance_kernel_variants(&mut variants, template);
        }
        MacroRole::Reciprocal => {
            push_unique_variant(
                &mut variants,
                Expr::BinOp(
                    BinOp::Add,
                    Box::new(template.clone()),
                    Box::new(Expr::Const(1.0)),
                ),
            );
            push_unique_variant(
                &mut variants,
                Expr::Func(
                    UnaryFn::Sqrt,
                    Box::new(Expr::BinOp(
                        BinOp::Add,
                        Box::new(template.clone()),
                        Box::new(Expr::Const(1.0)),
                    )),
                ),
            );
        }
        MacroRole::AffineShift | MacroRole::UnaryWrapped | MacroRole::Unknown => {}
        MacroRole::MultivariateInvariant => {}
    }

    variants
}

fn classify_macro_role(expr: &Expr) -> MacroRole {
    if expr_var_count(expr) > 1 {
        return MacroRole::MultivariateInvariant;
    }
    match expr {
        Expr::BinOp(BinOp::Pow, base, exp)
            if matches!(base.as_ref(), Expr::Var(_)) && matches!(exp.as_ref(), Expr::Const(_)) =>
        {
            MacroRole::PolynomialPower
        }
        Expr::BinOp(BinOp::Div, l, r)
            if matches!(l.as_ref(), Expr::Const(_)) && expr_uses_only_vars(r, &["n"]) =>
        {
            MacroRole::Reciprocal
        }
        Expr::BinOp(BinOp::Add | BinOp::Sub, l, r)
            if is_polynomial_power(l) || is_polynomial_power(r) =>
        {
            MacroRole::DistanceKernelCore
        }
        Expr::BinOp(BinOp::Add | BinOp::Sub, _, _) => MacroRole::AffineShift,
        Expr::Func(_, _) => MacroRole::UnaryWrapped,
        _ => MacroRole::Unknown,
    }
}

fn expr_var_count(expr: &Expr) -> usize {
    let mut vars = Vec::<String>::new();
    collect_expr_var_names(expr, &mut vars);
    vars.sort();
    vars.dedup();
    vars.len()
}

fn collect_expr_var_names(expr: &Expr, vars: &mut Vec<String>) {
    match expr {
        Expr::Var(name) => vars.push(name.clone()),
        Expr::Const(_) => {}
        Expr::BinOp(_, l, r) => {
            collect_expr_var_names(l, vars);
            collect_expr_var_names(r, vars);
        }
        Expr::Func(_, arg) => collect_expr_var_names(arg, vars),
        Expr::Sum(body, var) => {
            vars.push(var.clone());
            collect_expr_var_names(body, vars);
        }
    }
}

fn is_polynomial_power(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::BinOp(BinOp::Pow, base, exp)
            if matches!(base.as_ref(), Expr::Var(_)) && matches!(exp.as_ref(), Expr::Const(_))
    )
}

fn push_distance_kernel_variants(variants: &mut Vec<Expr>, template: &Expr) {
    if expr_uses_only_vars(template, &["n"]) {
        push_unique_variant(
            variants,
            Expr::Func(UnaryFn::Sqrt, Box::new(template.clone())),
        );
        push_unique_variant(
            variants,
            Expr::BinOp(
                BinOp::Div,
                Box::new(Expr::Const(1.0)),
                Box::new(template.clone()),
            ),
        );
        push_unique_variant(
            variants,
            Expr::BinOp(
                BinOp::Div,
                Box::new(Expr::Const(1.0)),
                Box::new(Expr::Func(UnaryFn::Sqrt, Box::new(template.clone()))),
            ),
        );

        let shifted = Expr::BinOp(
            BinOp::Add,
            Box::new(template.clone()),
            Box::new(Expr::Const(1.0)),
        );
        push_unique_variant(variants, shifted.clone());
        push_unique_variant(
            variants,
            Expr::Func(UnaryFn::Sqrt, Box::new(shifted.clone())),
        );
        push_unique_variant(
            variants,
            Expr::BinOp(
                BinOp::Div,
                Box::new(Expr::Const(1.0)),
                Box::new(Expr::Func(UnaryFn::Sqrt, Box::new(shifted))),
            ),
        );
    }
}

fn push_unique_variant(variants: &mut Vec<Expr>, candidate: Expr) {
    let key = macro_usage_key(&candidate);
    if !variants.iter().any(|expr| macro_usage_key(expr) == key) {
        variants.push(candidate);
    }
}

pub(crate) fn specialize_seed_constants(expr: &Expr, data: &[(f64, f64)], max_iter: usize) -> Expr {
    let constants = collect_constants(expr);
    if constants.is_empty() {
        return expr.clone();
    }

    let mut best = expr.clone();
    let mut best_mse = compute_mse(&best, data);
    let grid = [
        -10.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 10.0,
    ];

    for _ in 0..2 {
        for i in 0..constants.len() {
            let current = collect_constants(&best).get(i).copied().unwrap_or(1.0);
            let mut candidates = grid.to_vec();
            candidates.extend([
                current,
                current * 0.5,
                current * 2.0,
                current + 1.0,
                current - 1.0,
            ]);
            for candidate in candidates {
                let trial = replace_nth_constant(&best, i, candidate);
                let mse = compute_mse(&trial, data);
                if mse.is_finite() && mse < best_mse {
                    best = trial;
                    best_mse = mse;
                }
            }
        }
    }

    optimize_constants(&best, data, max_iter)
}

pub(crate) fn crossover(a: &Expr, b: &Expr, rng: &mut u64) -> Expr {
    match (a, b) {
        (Expr::BinOp(op, l, _), Expr::BinOp(_, _, r)) => Expr::BinOp(*op, l.clone(), r.clone()),
        (Expr::BinOp(op, l, r), _) => {
            *rng = lcg_step(*rng);
            if (*rng & 1) == 0 {
                Expr::BinOp(*op, l.clone(), Box::new(b.clone()))
            } else {
                Expr::BinOp(*op, Box::new(b.clone()), r.clone())
            }
        }
        _ => {
            *rng = lcg_step(*rng);
            if (*rng & 1) == 0 {
                a.clone()
            } else {
                b.clone()
            }
        }
    }
}

pub(crate) fn optimize_constants(expr: &Expr, data: &[(f64, f64)], max_iter: usize) -> Expr {
    let initial = collect_constants(expr);
    if initial.is_empty() {
        return expr.clone();
    }
    let initial_mse = compute_mse(expr, data);
    if initial_mse < 1e-10 {
        return expr.clone();
    }

    let n = initial.len();
    let objective = |params: &[f64]| -> f64 {
        let mut trial = expr.clone();
        for (i, &val) in params.iter().enumerate() {
            trial = replace_nth_constant(&trial, i, val);
        }
        let mse = compute_mse(&trial, data);
        if mse.is_finite() {
            mse
        } else {
            1e30
        }
    };

    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(initial.clone());
    for i in 0..n {
        let mut vertex = initial.clone();
        let step = if vertex[i].abs() > 1e-10 {
            vertex[i] * 0.1
        } else {
            0.1
        };
        vertex[i] += step;
        simplex.push(vertex);
    }
    let mut values: Vec<f64> = simplex.iter().map(|v| objective(v)).collect();
    let (alpha, gamma, rho, sigma) = (1.0, 2.0, 0.5, 0.5);

    for _ in 0..max_iter {
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| {
            values[a]
                .partial_cmp(&values[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let best_val = values[order[0]];
        let worst_val = values[order[n]];
        if (worst_val - best_val).abs() < 1e-14 {
            break;
        }

        let mut centroid = vec![0.0; n];
        for &idx in &order[..n] {
            for (j, c) in centroid.iter_mut().enumerate() {
                *c += simplex[idx][j];
            }
        }
        for c in &mut centroid {
            *c /= n as f64;
        }

        let worst_idx = order[n];
        let reflected: Vec<f64> = (0..n)
            .map(|j| centroid[j] + alpha * (centroid[j] - simplex[worst_idx][j]))
            .collect();
        let reflected_val = objective(&reflected);

        if reflected_val < values[order[n - 1]] && reflected_val >= best_val {
            simplex[worst_idx] = reflected;
            values[worst_idx] = reflected_val;
        } else if reflected_val < best_val {
            let expanded: Vec<f64> = (0..n)
                .map(|j| centroid[j] + gamma * (reflected[j] - centroid[j]))
                .collect();
            let expanded_val = objective(&expanded);
            if expanded_val < reflected_val {
                simplex[worst_idx] = expanded;
                values[worst_idx] = expanded_val;
            } else {
                simplex[worst_idx] = reflected;
                values[worst_idx] = reflected_val;
            }
        } else {
            let contracted: Vec<f64> = (0..n)
                .map(|j| centroid[j] + rho * (simplex[worst_idx][j] - centroid[j]))
                .collect();
            let contracted_val = objective(&contracted);
            if contracted_val < worst_val {
                simplex[worst_idx] = contracted;
                values[worst_idx] = contracted_val;
            } else {
                let best_idx = order[0];
                for &idx in &order[1..] {
                    for j in 0..n {
                        simplex[idx][j] =
                            simplex[best_idx][j] + sigma * (simplex[idx][j] - simplex[best_idx][j]);
                    }
                    values[idx] = objective(&simplex[idx]);
                }
            }
        }
    }

    let best_idx = values
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let best_params = &simplex[best_idx];
    if values[best_idx] < initial_mse {
        let mut result = expr.clone();
        for (i, &val) in best_params.iter().enumerate() {
            result = replace_nth_constant(&result, i, val);
        }
        result
    } else {
        expr.clone()
    }
}

pub(crate) fn collect_constants(expr: &Expr) -> Vec<f64> {
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

pub(crate) fn fingerprint_expr(expr: &Expr, sample_points: &[f64]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for &x in sample_points {
        let y = expr.eval(&[("n", x)]);
        if y.is_finite() {
            ((y * 1e9).round() as i64).hash(&mut hasher);
        } else {
            i64::MIN.hash(&mut hasher);
        }
    }
    hasher.finish()
}

pub(crate) fn compute_mse(expr: &Expr, data: &[(f64, f64)]) -> f64 {
    if data.is_empty() {
        return f64::MAX;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for &(x, y) in data {
        let predicted = expr.eval(&[("n", x)]);
        if !predicted.is_finite() {
            return f64::MAX;
        }
        sum += (predicted - y).powi(2);
        count += 1;
    }
    if count == 0 {
        f64::MAX
    } else {
        sum / count as f64
    }
}

fn lcg_step(state: u64) -> u64 {
    state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}
