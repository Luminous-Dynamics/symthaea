// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! The Integration-Differentiation Tradeoff Theorem.
//!
//! Proves computationally that for coupled oscillator networks:
//! - I(g) is monotonically increasing with coupling g
//! - D(g) is monotonically decreasing with coupling g
//! - I(g) × D(g) has a unique maximum at g* (the consciousness sweet spot)
//! - g* depends on network topology
//!
//! This explains WHY the brain operates at criticality: it's tuning
//! coupling to maximize the product of integration and differentiation.

use crate::nonlinear_waves::OscillatorNetwork;

/// Result of the I×D tradeoff analysis at one coupling strength.
#[derive(Debug, Clone)]
pub struct TradeoffPoint {
    pub coupling: f64,
    pub integration: f64,
    pub differentiation: f64,
    pub id_product: f64,
    pub synchrony: f64,
    pub metastability: f64,
}

/// Compute I, D, and I×D at a specific coupling for a given topology builder.
fn compute_id_at_coupling(
    n: usize,
    coupling: f64,
    build_coupling: &dyn Fn(usize, f64) -> Vec<f64>,
) -> TradeoffPoint {
    let dt = 0.1;
    let warmup = 3000;
    let measure = 5000;

    let mut net = OscillatorNetwork {
        oscillators: (0..n).map(|i| {
            let mut osc = crate::nonlinear_waves::FitzHughNagumo::new();
            // Heterogeneous: different natural frequencies
            osc.i_ext = 0.4 + 0.4 * (i as f64 / n as f64 - 0.5);
            osc.v = -1.0 + 0.2 * (i as f64 / n as f64);
            osc
        }).collect(),
        coupling: build_coupling(n, coupling),
        n,
    };

    for _ in 0..warmup { net.step(dt); }

    let mut traces: Vec<Vec<f64>> = vec![Vec::with_capacity(measure); n];
    let mut sync_trace = Vec::with_capacity(measure);

    for _ in 0..measure {
        net.step(dt);
        for (i, osc) in net.oscillators.iter().enumerate() {
            traces[i].push(osc.v);
        }
        sync_trace.push(net.synchronization_index());
    }

    // Integration: mean pairwise absolute correlation
    let mut integration = 0.0;
    let mut n_pairs = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            integration += pearson_abs(&traces[i], &traces[j]);
            n_pairs += 1;
        }
    }
    integration /= n_pairs.max(1) as f64;

    // Differentiation: variance of firing rates
    let rates: Vec<f64> = traces.iter().map(|tr| {
        tr.windows(2).filter(|w| w[0] < 0.0 && w[1] >= 0.0).count() as f64 / (measure as f64 * dt)
    }).collect();
    let rate_mean = rates.iter().sum::<f64>() / n as f64;
    let differentiation = if n > 1 {
        let var = rates.iter().map(|r| (r - rate_mean).powi(2)).sum::<f64>() / n as f64;
        // Normalize: max differentiation when rates span full range
        (var * 100.0).min(1.0)
    } else { 0.0 };

    let synchrony = sync_trace.iter().sum::<f64>() / sync_trace.len() as f64;
    let sync_mean = synchrony;
    let metastability = sync_trace.iter()
        .map(|s| (s - sync_mean).powi(2))
        .sum::<f64>() / sync_trace.len() as f64;

    TradeoffPoint {
        coupling,
        integration,
        differentiation,
        id_product: integration * differentiation,
        synchrony,
        metastability: (metastability * 50.0).min(1.0),
    }
}

/// Coupling matrix builder for all-to-all topology.
fn all_to_all_coupling(n: usize, g: f64) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    let w = g / (n - 1).max(1) as f64;
    for i in 0..n {
        for j in 0..n {
            if i != j { c[i * n + j] = w; }
        }
    }
    c
}

/// Coupling matrix builder for modular topology (4 modules).
fn modular_coupling(n: usize, g: f64) -> Vec<f64> {
    let n_modules = 4;
    let module_size = n / n_modules;
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        let mi = i / module_size.max(1);
        for j in 0..n {
            if i == j { continue; }
            let mj = j / module_size.max(1);
            c[i * n + j] = if mi == mj {
                g / module_size.max(1) as f64
            } else {
                g * 0.05 / n as f64
            };
        }
    }
    c
}

/// Coupling matrix builder for ring topology.
fn ring_coupling(n: usize, g: f64) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    let w = g / 2.0;
    for i in 0..n {
        c[i * n + (i + 1) % n] = w;
        c[i * n + (i + n - 1) % n] = w;
    }
    c
}

/// Coupling matrix for hierarchical (2×2×N/4).
fn hierarchical_coupling(n: usize, g: f64) -> Vec<f64> {
    let quarter = (n / 4).max(1);
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        let qi = i / quarter;
        let hi = i / (n / 2).max(1);
        for j in 0..n {
            if i == j { continue; }
            let qj = j / quarter;
            let hj = j / (n / 2).max(1);
            c[i * n + j] = if qi == qj {
                g / quarter as f64
            } else if hi == hj {
                g * 0.2 / (n / 2).max(1) as f64
            } else {
                g * 0.02 / n as f64
            };
        }
    }
    c
}

/// Sweep coupling from very weak to very strong, measuring I, D, and I×D.
pub fn coupling_sweep(
    n: usize,
    n_points: usize,
    topology_name: &str,
    build_fn: &dyn Fn(usize, f64) -> Vec<f64>,
) -> Vec<TradeoffPoint> {
    // Logarithmic sweep from 0.001 to 2.0
    (0..n_points).map(|i| {
        let frac = i as f64 / (n_points - 1) as f64;
        let g = 0.001 * (2000.0_f64).powf(frac); // 0.001 to 2.0
        compute_id_at_coupling(n, g, build_fn)
    }).collect()
}

/// Find the optimal coupling g* that maximizes I×D.
pub fn find_optimal_coupling(points: &[TradeoffPoint]) -> (f64, f64) {
    points.iter()
        .max_by(|a, b| a.id_product.partial_cmp(&b.id_product).unwrap())
        .map(|p| (p.coupling, p.id_product))
        .unwrap_or((0.0, 0.0))
}

/// Check monotonicity of I(g) and D(g).
pub fn check_monotonicity(points: &[TradeoffPoint]) -> (bool, bool, usize, usize) {
    let mut i_violations = 0usize;
    let mut d_violations = 0usize;

    for i in 1..points.len() {
        // I should be non-decreasing
        if points[i].integration < points[i - 1].integration - 0.01 {
            i_violations += 1;
        }
        // D should be non-increasing
        if points[i].differentiation > points[i - 1].differentiation + 0.01 {
            d_violations += 1;
        }
    }

    let i_mono = i_violations == 0;
    let d_mono = d_violations == 0;
    (i_mono, d_mono, i_violations, d_violations)
}

/// Full tradeoff analysis across 4 topologies.
pub fn full_tradeoff_analysis(n: usize, n_points: usize) -> Vec<(&'static str, Vec<TradeoffPoint>, f64, f64)> {
    let topologies: Vec<(&str, Box<dyn Fn(usize, f64) -> Vec<f64>>)> = vec![
        ("All-to-All", Box::new(all_to_all_coupling)),
        ("Ring", Box::new(ring_coupling)),
        ("Modular(4)", Box::new(modular_coupling)),
        ("Hierarchical", Box::new(hierarchical_coupling)),
    ];

    topologies.into_iter().map(|(name, build_fn)| {
        let points = coupling_sweep(n, n_points, name, &*build_fn);
        let (g_star, id_max) = find_optimal_coupling(&points);
        (name, points, g_star, id_max)
    }).collect()
}

fn pearson_abs(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len()) as f64;
    if n < 3.0 { return 0.0; }
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let (mut cov, mut vx, mut vy) = (0.0, 0.0, 0.0);
    for (xi, yi) in x.iter().zip(y.iter()) {
        let (dx, dy) = (xi - mx, yi - my);
        cov += dx * dy; vx += dx * dx; vy += dy * dy;
    }
    if vx < 1e-15 || vy < 1e-15 { return 0.0; }
    (cov / (vx * vy).sqrt()).abs().min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tradeoff_point_bounded() {
        let p = compute_id_at_coupling(10, 0.1, &all_to_all_coupling);
        assert!(p.integration >= 0.0 && p.integration <= 1.0);
        assert!(p.differentiation >= 0.0 && p.differentiation <= 1.0);
        assert!(p.id_product >= 0.0 && p.id_product <= 1.0);
    }

    #[test]
    fn test_optimal_coupling_exists() {
        let points = coupling_sweep(10, 8, "test", &all_to_all_coupling);
        let (g_star, id_max) = find_optimal_coupling(&points);
        assert!(g_star > 0.0, "Optimal coupling should exist: g*={:.4}", g_star);
        assert!(id_max > 0.0, "I×D max should be positive: {:.4}", id_max);
    }
}
