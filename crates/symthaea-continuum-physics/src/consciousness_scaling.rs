// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness Scaling Laws — revised with independent metrics.
//!
//! V2: Fixed dead entropy metric, added genuinely independent measures,
//! replaced sync-derived redundant metrics.

use crate::nonlinear_waves::{FitzHughNagumo, OscillatorNetwork};

pub const N_THEORIES: usize = 8;

pub const THEORY_NAMES: [&str; N_THEORIES] = [
    "Synchrony",      // 0: Global sync order parameter r (GWT)
    "Integration",    // 1: Mean pairwise cross-correlation (IIT)
    "Differentiation",// 2: Variance of per-oscillator mean firing rates (Complexity)
    "Coherence",      // 3: Autocorrelation of global field at lag 10 (Orch-OR)
    "Metastability",  // 4: Variance of instantaneous sync index (HOT)
    "Entropy",        // 5: TEMPORAL entropy of individual oscillators, averaged (Thermo)
    "TransferEnt",    // 6: Directional info flow (Granger-like causality proxy)
    "MultiScale",     // 7: Ratio of local to global integration
];

#[derive(Debug, Clone)]
pub struct NetworkScores {
    pub label: String,
    pub n_oscillators: usize,
    pub coupling_strength: f64,
    pub scores: [f64; N_THEORIES],
}

/// Compute 8 INDEPENDENT consciousness metrics for a network.
pub fn compute_network_scores(
    n_oscillators: usize,
    coupling_strength: f64,
    i_ext: f64,
    label: &str,
) -> NetworkScores {
    let dt = 0.1;
    let warmup = 3000;
    let measure = 5000;

    let mut net = OscillatorNetwork::new(n_oscillators, coupling_strength);
    for osc in &mut net.oscillators {
        osc.i_ext = i_ext;
    }
    for _ in 0..warmup { net.step(dt); }

    let mut traces: Vec<Vec<f64>> = vec![Vec::with_capacity(measure); n_oscillators];
    let mut sync_trace = Vec::with_capacity(measure);

    for _ in 0..measure {
        net.step(dt);
        for (i, osc) in net.oscillators.iter().enumerate() {
            traces[i].push(osc.v);
        }
        sync_trace.push(net.synchronization_index());
    }

    let n = n_oscillators;

    // === 0: Synchrony — mean of synchronization order parameter ===
    let synchrony = sync_trace.iter().sum::<f64>() / sync_trace.len() as f64;

    // === 1: Integration — mean absolute pairwise cross-correlation ===
    let mut integration = 0.0;
    let mut n_pairs = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            integration += pearson_abs(&traces[i], &traces[j]);
            n_pairs += 1;
        }
    }
    integration /= n_pairs.max(1) as f64;

    // === 2: Differentiation — variance of per-oscillator firing rates ===
    // Firing rate = number of upward zero-crossings per unit time
    let rates: Vec<f64> = traces.iter().map(|tr| {
        let crossings = tr.windows(2).filter(|w| w[0] < 0.0 && w[1] >= 0.0).count();
        crossings as f64 / (measure as f64 * dt)
    }).collect();
    let rate_mean = rates.iter().sum::<f64>() / n as f64;
    let differentiation = if n > 1 {
        rates.iter().map(|r| (r - rate_mean).powi(2)).sum::<f64>() / n as f64
    } else { 0.0 };

    // === 3: Coherence — autocorrelation of global field at lag=20 ===
    let global: Vec<f64> = (0..measure)
        .map(|t| traces.iter().map(|tr| tr[t]).sum::<f64>() / n as f64)
        .collect();
    let coherence = autocorrelation(&global, 20).abs();

    // === 4: Metastability — variance of instantaneous sync index ===
    let sync_mean = sync_trace.iter().sum::<f64>() / sync_trace.len() as f64;
    let metastability = sync_trace.iter()
        .map(|s| (s - sync_mean).powi(2))
        .sum::<f64>() / sync_trace.len() as f64;

    // === 5: Entropy — TEMPORAL Shannon entropy of each oscillator, averaged ===
    // Bin the TIME SERIES of each oscillator into 30 bins over [-2.5, 2.5]
    let n_bins = 30;
    let v_min = -2.5;
    let v_max = 2.5;
    let bin_width = (v_max - v_min) / n_bins as f64;
    let mut mean_entropy = 0.0;
    for tr in &traces {
        let mut bins = vec![0usize; n_bins];
        for &v in tr {
            let idx = ((v - v_min) / bin_width).clamp(0.0, (n_bins - 1) as f64) as usize;
            bins[idx] += 1;
        }
        let total = tr.len() as f64;
        let h: f64 = bins.iter().filter(|&&b| b > 0)
            .map(|&b| { let p = b as f64 / total; -p * p.ln() })
            .sum();
        mean_entropy += h;
    }
    mean_entropy /= n as f64;
    let max_h = (n_bins as f64).ln();
    let entropy_norm = if max_h > 0.0 { mean_entropy / max_h } else { 0.0 };

    // === 6: Transfer Entropy — directional information flow proxy ===
    // Simplified: for each pair (i→j), check if past of i predicts future of j
    // better than past of j alone. Average over all directed pairs.
    let lag = 5;
    let mut transfer_sum = 0.0;
    let mut n_dir_pairs = 0usize;
    // Sample a subset for speed
    let step = (n / 10).max(1);
    for i in (0..n).step_by(step) {
        for j in (0..n).step_by(step) {
            if i == j { continue; }
            let te = transfer_entropy_proxy(&traces[i], &traces[j], lag);
            transfer_sum += te;
            n_dir_pairs += 1;
        }
    }
    let transfer_ent = transfer_sum / n_dir_pairs.max(1) as f64;

    // === 7: Multi-Scale Integration — local vs global ===
    // Local: average correlation within nearest-neighbor pairs
    // Global: average correlation between distant pairs (distance > n/3)
    let mut local_corr = 0.0;
    let mut global_corr = 0.0;
    let mut n_local = 0usize;
    let mut n_global = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = (j - i).min(n - j + i); // Circular distance
            let c = pearson_abs(&traces[i], &traces[j]);
            if dist <= 2 {
                local_corr += c;
                n_local += 1;
            }
            if dist > n / 3 {
                global_corr += c;
                n_global += 1;
            }
        }
    }
    local_corr /= n_local.max(1) as f64;
    global_corr /= n_global.max(1) as f64;
    // Multi-scale ratio: local much higher than global = modular; equal = homogeneous
    let multi_scale = if local_corr + global_corr > 1e-10 {
        (local_corr - global_corr) / (local_corr + global_corr)
    } else { 0.0 };

    let scores = [
        synchrony.clamp(0.0, 1.0),
        integration.clamp(0.0, 1.0),
        (differentiation * 50.0).min(1.0).clamp(0.0, 1.0),
        coherence.clamp(0.0, 1.0),
        (metastability * 50.0).min(1.0).clamp(0.0, 1.0),
        entropy_norm.clamp(0.0, 1.0),
        (transfer_ent * 20.0).min(1.0).clamp(0.0, 1.0),
        ((multi_scale + 1.0) / 2.0).clamp(0.0, 1.0), // Map [-1,1] → [0,1]
    ];

    NetworkScores {
        label: label.to_string(),
        n_oscillators,
        coupling_strength,
        scores,
    }
}

/// Transfer entropy proxy: does knowing x's past improve prediction of y's future?
/// TE(x→y) ∝ |corr(x[t-lag], y[t]) - corr(y[t-lag], y[t])|
fn transfer_entropy_proxy(x: &[f64], y: &[f64], lag: usize) -> f64 {
    if lag >= x.len() || lag >= y.len() { return 0.0; }
    let n = x.len() - lag;

    // corr(x_past, y_future)
    let x_past: Vec<f64> = x[..n].to_vec();
    let y_future: Vec<f64> = y[lag..].to_vec();
    let r_xy = pearson_abs(&x_past, &y_future);

    // corr(y_past, y_future) — baseline
    let y_past: Vec<f64> = y[..n].to_vec();
    let r_yy = pearson_abs(&y_past, &y_future);

    // TE proxy: how much does x improve prediction beyond y's own past?
    (r_xy - r_yy).max(0.0)
}

pub fn network_correlation_matrix(results: &[NetworkScores]) -> [[f64; N_THEORIES]; N_THEORIES] {
    let mut corr = [[0.0; N_THEORIES]; N_THEORIES];
    for i in 0..N_THEORIES {
        for j in i..N_THEORIES {
            let xi: Vec<f64> = results.iter().map(|r| r.scores[i]).collect();
            let xj: Vec<f64> = results.iter().map(|r| r.scores[j]).collect();
            let r = pearson_signed(&xi, &xj);
            corr[i][j] = r;
            corr[j][i] = r;
        }
    }
    corr
}

pub fn effective_dimensionality(corr: &[[f64; N_THEORIES]; N_THEORIES]) -> (usize, Vec<f64>) {
    let n = N_THEORIES;
    let mut a = vec![0.0; n * n];
    for i in 0..n { for j in 0..n { a[i * n + j] = corr[i][j]; } }

    let mut eigenvalues = Vec::new();
    for _ in 0..n {
        let mut v = vec![1.0 / (n as f64).sqrt(); n];
        for _ in 0..300 {
            let mut av = vec![0.0; n];
            for i in 0..n { for j in 0..n { av[i] += a[i * n + j] * v[j]; } }
            let norm = av.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 { break; }
            for i in 0..n { v[i] = av[i] / norm; }
        }
        let mut av = vec![0.0; n];
        for i in 0..n { for j in 0..n { av[i] += a[i * n + j] * v[j]; } }
        let lambda: f64 = v.iter().zip(av.iter()).map(|(vi, avi)| vi * avi).sum();
        eigenvalues.push(lambda.abs());
        for i in 0..n { for j in 0..n { a[i * n + j] -= lambda * v[i] * v[j]; } }
    }
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let total: f64 = eigenvalues.iter().sum();
    let threshold = 0.05 * total;
    let n_eff = eigenvalues.iter().filter(|&&e| e > threshold).count();
    (n_eff, eigenvalues)
}

pub fn scaling_experiment(sizes: &[usize], n_configs: usize) -> Vec<(usize, usize, Vec<f64>)> {
    sizes.iter().map(|&n| {
        let configs: Vec<NetworkScores> = (0..n_configs).map(|c| {
            let coupling = 0.05 + (c as f64 / n_configs as f64) * 0.95;
            let i_ext = 0.3 + (c as f64 / n_configs as f64) * 0.5;
            compute_network_scores(n, coupling, i_ext, &format!("N{}_c{}", n, c))
        }).collect();
        let corr = network_correlation_matrix(&configs);
        let (n_eff, eigenvalues) = effective_dimensionality(&corr);
        (n, n_eff, eigenvalues)
    }).collect()
}

fn pearson_abs(x: &[f64], y: &[f64]) -> f64 { pearson_signed(x, y).abs() }

fn pearson_signed(x: &[f64], y: &[f64]) -> f64 {
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
    let r = cov / (vx * vy).sqrt();
    if r.is_nan() { 0.0 } else { r.clamp(-1.0, 1.0) }
}

fn autocorrelation(x: &[f64], lag: usize) -> f64 {
    if lag >= x.len() { return 0.0; }
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let var: f64 = x.iter().map(|xi| (xi - mean).powi(2)).sum::<f64>() / x.len() as f64;
    if var < 1e-15 { return 0.0; }
    let acov: f64 = x[..x.len() - lag].iter().zip(x[lag..].iter())
        .map(|(a, b)| (a - mean) * (b - mean)).sum::<f64>() / (x.len() - lag) as f64;
    acov / var
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scores_bounded() {
        let s = compute_network_scores(5, 0.3, 0.5, "test");
        for (i, &v) in s.scores.iter().enumerate() {
            assert!(v >= 0.0 && v <= 1.0, "Theory {} = {:.4}", THEORY_NAMES[i], v);
        }
    }

    #[test]
    fn test_entropy_nonzero_for_oscillating() {
        let s = compute_network_scores(5, 0.3, 0.5, "test");
        assert!(s.scores[5] > 0.01, "Entropy should be nonzero for oscillating network: {:.4}", s.scores[5]);
    }

    #[test]
    fn test_scaling_runs() {
        let r = scaling_experiment(&[3, 5], 3);
        assert_eq!(r.len(), 2);
    }
}
