// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness Scaling Laws — How does theory dimensionality scale with system size?
//!
//! Uses coupled FitzHugh-Nagumo oscillator networks of increasing size to measure
//! when consciousness theories become distinguishable. At the molecular level,
//! 10 theories collapse into 2 dimensions. At what network size do they separate?
//!
//! This bridges the molecular paper (floor exists) with a characterization of
//! the floor: the critical system size for consciousness science.

use crate::nonlinear_waves::{FitzHughNagumo, OscillatorNetwork};
use crate::open_quantum::{DensityMatrix, LindbladSolver};

/// Number of consciousness theories evaluated on neural networks.
pub const N_THEORIES: usize = 8;

pub const THEORY_NAMES: [&str; N_THEORIES] = [
    "Synchrony",    // 0: Global synchronization (GWT analog)
    "Integration",  // 1: Mutual information between subsystems (IIT analog)
    "Complexity",   // 2: Variance of oscillator phases (differentiation)
    "Coherence",    // 3: Phase coherence (Orch-OR analog)
    "Metastable",   // 4: Number of metastable states (HOT analog)
    "Entropy",      // 5: Phase entropy (thermodynamic analog)
    "Dissipative",  // 6: Energy throughput (Prigogine analog)
    "Decoherence",  // 7: Rate of desynchronization (QDarwinism analog)
];

/// Consciousness scores for a neural network at one configuration.
#[derive(Debug, Clone)]
pub struct NetworkScores {
    pub label: String,
    pub n_oscillators: usize,
    pub coupling_strength: f64,
    pub scores: [f64; N_THEORIES],
}

/// Compute 8 consciousness-like metrics for a coupled oscillator network.
pub fn compute_network_scores(
    n_oscillators: usize,
    coupling_strength: f64,
    i_ext: f64,
    label: &str,
) -> NetworkScores {
    let dt = 0.1;
    let warmup = 2000;
    let measure = 3000;

    // Build and warm up the network
    let mut net = OscillatorNetwork::new(n_oscillators, coupling_strength);
    for osc in &mut net.oscillators {
        osc.i_ext = i_ext;
    }

    for _ in 0..warmup {
        net.step(dt);
    }

    // Collect measurement data
    let mut voltage_traces: Vec<Vec<f64>> = vec![Vec::with_capacity(measure); n_oscillators];
    let mut sync_trace = Vec::with_capacity(measure);

    for _ in 0..measure {
        net.step(dt);
        for (i, osc) in net.oscillators.iter().enumerate() {
            voltage_traces[i].push(osc.v);
        }
        sync_trace.push(net.synchronization_index());
    }

    // === Theory 0: Synchrony (GWT analog) ===
    // Mean synchronization index over measurement window
    let synchrony = sync_trace.iter().sum::<f64>() / sync_trace.len() as f64;

    // === Theory 1: Integration (IIT analog) ===
    // Cross-correlation between all oscillator pairs
    let mut integration = 0.0;
    let mut n_pairs = 0;
    for i in 0..n_oscillators {
        for j in (i + 1)..n_oscillators {
            let corr = cross_correlation(&voltage_traces[i], &voltage_traces[j]);
            integration += corr.abs();
            n_pairs += 1;
        }
    }
    integration = if n_pairs > 0 { integration / n_pairs as f64 } else { 0.0 };

    // === Theory 2: Complexity (Differentiation) ===
    // Variance of mean voltages across oscillators — higher = more differentiated
    let means: Vec<f64> = voltage_traces.iter()
        .map(|tr| tr.iter().sum::<f64>() / tr.len() as f64)
        .collect();
    let grand_mean = means.iter().sum::<f64>() / means.len() as f64;
    let complexity = means.iter()
        .map(|m| (m - grand_mean).powi(2))
        .sum::<f64>() / means.len() as f64;

    // === Theory 3: Coherence (Orch-OR analog) ===
    // Autocorrelation at lag=10 — measures persistence of oscillatory patterns
    let global: Vec<f64> = (0..measure)
        .map(|t| voltage_traces.iter().map(|tr| tr[t]).sum::<f64>() / n_oscillators as f64)
        .collect();
    let coherence = autocorrelation(&global, 10);

    // === Theory 4: Metastability (HOT analog) ===
    // Variance of the synchronization index — high variance = switching between states
    let sync_mean = sync_trace.iter().sum::<f64>() / sync_trace.len() as f64;
    let metastable = sync_trace.iter()
        .map(|s| (s - sync_mean).powi(2))
        .sum::<f64>() / sync_trace.len() as f64;

    // === Theory 5: Entropy (Thermodynamic analog) ===
    // Shannon entropy of the phase distribution (binned)
    let n_bins = 20;
    let mut bins = vec![0usize; n_bins];
    for osc in &net.oscillators {
        let phase = ((osc.v + 2.0) / 4.0).clamp(0.0, 0.9999); // Map [-2,2] → [0,1)
        let bin = (phase * n_bins as f64) as usize;
        bins[bin] += 1;
    }
    let n_total = n_oscillators as f64;
    let entropy: f64 = bins.iter()
        .filter(|&&b| b > 0)
        .map(|&b| {
            let p = b as f64 / n_total;
            -p * p.ln()
        })
        .sum();
    let max_entropy = (n_bins.min(n_oscillators) as f64).ln();
    let entropy_norm = if max_entropy > 0.0 { entropy / max_entropy } else { 0.0 };

    // === Theory 6: Dissipative (Energy throughput) ===
    // Mean absolute velocity (dv/dt proxy) — how much energy flows through the system
    let dissipative: f64 = voltage_traces.iter()
        .map(|tr| {
            tr.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f64>() / (tr.len() - 1) as f64
        })
        .sum::<f64>() / n_oscillators as f64;

    // === Theory 7: Decoherence (QDarwinism analog) ===
    // Rate at which synchronization decays — fast decay = easy classical observation
    // Measure sync variance over short windows
    let window = 50;
    let mut sync_variability = 0.0;
    let mut n_windows = 0;
    for chunk in sync_trace.chunks(window) {
        if chunk.len() < window { break; }
        let mean = chunk.iter().sum::<f64>() / chunk.len() as f64;
        let var: f64 = chunk.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / chunk.len() as f64;
        sync_variability += var;
        n_windows += 1;
    }
    let decoherence = if n_windows > 0 { sync_variability / n_windows as f64 } else { 0.0 };

    // Normalize all to [0, 1]
    let scores = [
        synchrony.clamp(0.0, 1.0),
        integration.clamp(0.0, 1.0),
        (complexity * 100.0).clamp(0.0, 1.0).min(1.0),  // Scale up small values
        coherence.clamp(0.0, 1.0),
        (metastable * 100.0).clamp(0.0, 1.0).min(1.0),  // Scale up
        entropy_norm.clamp(0.0, 1.0),
        (dissipative * 5.0).clamp(0.0, 1.0).min(1.0),   // Scale
        (decoherence * 1000.0).clamp(0.0, 1.0).min(1.0), // Scale up
    ];

    NetworkScores {
        label: label.to_string(),
        n_oscillators,
        coupling_strength,
        scores,
    }
}

/// Compute the theory correlation matrix across a set of network configurations.
pub fn network_correlation_matrix(results: &[NetworkScores]) -> [[f64; N_THEORIES]; N_THEORIES] {
    let mut corr = [[0.0; N_THEORIES]; N_THEORIES];
    for i in 0..N_THEORIES {
        for j in i..N_THEORIES {
            let xi: Vec<f64> = results.iter().map(|r| r.scores[i]).collect();
            let xj: Vec<f64> = results.iter().map(|r| r.scores[j]).collect();
            let r = pearson(&xi, &xj);
            corr[i][j] = r;
            corr[j][i] = r;
        }
    }
    corr
}

/// Effective dimensionality via eigendecomposition.
pub fn effective_dimensionality(corr: &[[f64; N_THEORIES]; N_THEORIES]) -> (usize, Vec<f64>) {
    let n = N_THEORIES;
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = corr[i][j];
        }
    }

    let mut eigenvalues = Vec::new();
    for _ in 0..n {
        let mut v = vec![1.0 / (n as f64).sqrt(); n];
        for _ in 0..200 {
            let mut av = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    av[i] += a[i * n + j] * v[j];
                }
            }
            let norm = av.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 { break; }
            for i in 0..n { v[i] = av[i] / norm; }
        }
        let mut av = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                av[i] += a[i * n + j] * v[j];
            }
        }
        let lambda: f64 = v.iter().zip(av.iter()).map(|(vi, avi)| vi * avi).sum();
        eigenvalues.push(lambda.abs());
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] -= lambda * v[i] * v[j];
            }
        }
    }

    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let total: f64 = eigenvalues.iter().sum();
    let threshold = 0.05 * total;
    let n_eff = eigenvalues.iter().filter(|&&e| e > threshold).count();
    (n_eff, eigenvalues)
}

/// Run the full scaling experiment: networks from 2 to max_n oscillators.
pub fn scaling_experiment(
    sizes: &[usize],
    n_configs_per_size: usize,
) -> Vec<(usize, usize, Vec<f64>)> {
    let mut results = Vec::new();

    for &n in sizes {
        let mut configs = Vec::new();

        for c in 0..n_configs_per_size {
            let coupling = 0.05 + (c as f64 / n_configs_per_size as f64) * 0.95;
            let i_ext = 0.3 + (c as f64 / n_configs_per_size as f64) * 0.5;
            let label = format!("N{}_c{}", n, c);
            configs.push(compute_network_scores(n, coupling, i_ext, &label));
        }

        let corr = network_correlation_matrix(&configs);
        let (n_eff, eigenvalues) = effective_dimensionality(&corr);
        results.push((n, n_eff, eigenvalues));
    }

    results
}

fn cross_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len()) as f64;
    if n < 2.0 { return 0.0; }
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mx;
        let dy = yi - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx < 1e-15 || vy < 1e-15 { return 0.0; }
    cov / (vx * vy).sqrt()
}

fn autocorrelation(x: &[f64], lag: usize) -> f64 {
    if lag >= x.len() { return 0.0; }
    let n = (x.len() - lag) as f64;
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let var: f64 = x.iter().map(|xi| (xi - mean).powi(2)).sum::<f64>() / x.len() as f64;
    if var < 1e-15 { return 0.0; }
    let acov: f64 = x[..x.len() - lag].iter().zip(x[lag..].iter())
        .map(|(a, b)| (a - mean) * (b - mean))
        .sum::<f64>() / n;
    acov / var
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 3.0 { return 0.0; }
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mx;
        let dy = yi - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx < 1e-15 || vy < 1e-15 { return 0.0; }
    let r = cov / (vx * vy).sqrt();
    if r.is_nan() { 0.0 } else { r.clamp(-1.0, 1.0) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_scores_bounded() {
        let scores = compute_network_scores(5, 0.3, 0.5, "test");
        for (i, &s) in scores.scores.iter().enumerate() {
            assert!(s >= 0.0 && s <= 1.0, "Theory {} score = {:.4}", i, s);
        }
    }

    #[test]
    fn test_scaling_runs() {
        let results = scaling_experiment(&[3, 5], 3);
        assert_eq!(results.len(), 2);
        for (n, dim, _) in &results {
            assert!(*dim <= N_THEORIES, "Dimensionality {} > N_THEORIES for n={}", dim, n);
        }
    }
}
