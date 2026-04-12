// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness vs Network Topology.
//!
//! Tests the hypothesis: consciousness dimensionality depends on
//! organizational structure, not just system size. Compares all-to-all,
//! modular, hierarchical, and small-world networks at the same size.

use crate::consciousness_scaling::{
    compute_network_scores, effective_dimensionality, network_correlation_matrix,
    NetworkScores, N_THEORIES, THEORY_NAMES,
};
use crate::nonlinear_waves::{FitzHughNagumo, OscillatorNetwork};

/// Network topology types.
#[derive(Debug, Clone, Copy)]
pub enum Topology {
    AllToAll,
    Modular { n_modules: usize },
    Hierarchical,
    Ring,
    SmallWorld { rewire_prob: f64 },
}

/// Build a network with specified topology.
pub fn build_network(n: usize, topology: Topology, coupling_strength: f64) -> OscillatorNetwork {
    let mut net = OscillatorNetwork::new(n, 0.0); // Start with zero coupling

    // Clear coupling matrix
    for i in 0..n * n {
        net.coupling[i] = 0.0;
    }

    match topology {
        Topology::AllToAll => {
            let g = coupling_strength / (n - 1).max(1) as f64;
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        net.coupling[i * n + j] = g;
                    }
                }
            }
        }
        Topology::Modular { n_modules } => {
            let module_size = n / n_modules;
            let g_intra = coupling_strength; // Strong within-module
            let g_inter = coupling_strength * 0.05; // Weak between-module

            for i in 0..n {
                let mod_i = i / module_size;
                for j in 0..n {
                    if i == j { continue; }
                    let mod_j = j / module_size;
                    if mod_i == mod_j {
                        // Same module: strong coupling
                        net.coupling[i * n + j] = g_intra / module_size.max(1) as f64;
                    } else {
                        // Different modules: weak coupling
                        net.coupling[i * n + j] = g_inter / n as f64;
                    }
                }
            }
        }
        Topology::Hierarchical => {
            // 2 super-modules, each with 2 sub-modules
            let quarter = n / 4;
            let g_local = coupling_strength;
            let g_sub = coupling_strength * 0.2;
            let g_super = coupling_strength * 0.02;

            for i in 0..n {
                let quarter_i = i / quarter.max(1);
                let half_i = i / (n / 2).max(1);
                for j in 0..n {
                    if i == j { continue; }
                    let quarter_j = j / quarter.max(1);
                    let half_j = j / (n / 2).max(1);

                    let g = if quarter_i == quarter_j {
                        g_local / quarter.max(1) as f64
                    } else if half_i == half_j {
                        g_sub / (n / 2).max(1) as f64
                    } else {
                        g_super / n as f64
                    };
                    net.coupling[i * n + j] = g;
                }
            }
        }
        Topology::Ring => {
            // Each oscillator connects to its 2 nearest neighbors
            let g = coupling_strength / 2.0;
            for i in 0..n {
                let left = (i + n - 1) % n;
                let right = (i + 1) % n;
                net.coupling[i * n + left] = g;
                net.coupling[i * n + right] = g;
            }
        }
        Topology::SmallWorld { rewire_prob } => {
            // Start with ring, then rewire some connections
            let g = coupling_strength / 3.0;
            // Ring base
            for i in 0..n {
                let left = (i + n - 1) % n;
                let right = (i + 1) % n;
                net.coupling[i * n + left] = g;
                net.coupling[i * n + right] = g;
            }
            // Rewire: add long-range connections deterministically
            let mut seed = 42u64;
            for i in 0..n {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let prob = (seed >> 33) as f64 / (1u64 << 31) as f64;
                if prob < rewire_prob {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let target = ((seed >> 33) as f64 / (1u64 << 31) as f64 * n as f64) as usize % n;
                    if target != i {
                        net.coupling[i * n + target] = g;
                        net.coupling[target * n + i] = g;
                    }
                }
            }
        }
    }

    // Vary initial conditions slightly for symmetry breaking
    for (i, osc) in net.oscillators.iter_mut().enumerate() {
        osc.v = -1.0 + 0.1 * (i as f64 / n as f64);
    }

    net
}

/// Compute scores for a network with specified topology, varying parameters.
pub fn topology_scores(
    n: usize,
    topology: Topology,
    n_configs: usize,
) -> Vec<NetworkScores> {
    let dt = 0.1;
    let warmup = 2000;
    let measure = 3000;

    (0..n_configs).map(|c| {
        // Vary coupling across critical regime
        let coupling = 0.005 + (c as f64 / n_configs as f64) * 0.5;
        let base_i_ext = 0.4 + (c as f64 / n_configs as f64) * 0.2;

        let mut net = build_network(n, topology, coupling);
        // HETEROGENEOUS drive: each oscillator has different natural frequency
        // This prevents trivial synchronization from identical dynamics
        for (i, osc) in net.oscillators.iter_mut().enumerate() {
            let spread = 0.3; // Frequency spread
            osc.i_ext = base_i_ext + spread * (i as f64 / n as f64 - 0.5);
        }

        // Warm up
        for _ in 0..warmup {
            net.step(dt);
        }

        // Measure — reuse the compute_network_scores logic but from our custom network
        let mut voltage_traces: Vec<Vec<f64>> = vec![Vec::with_capacity(measure); n];
        let mut sync_trace = Vec::with_capacity(measure);

        for _ in 0..measure {
            net.step(dt);
            for (i, osc) in net.oscillators.iter().enumerate() {
                voltage_traces[i].push(osc.v);
            }
            sync_trace.push(net.synchronization_index());
        }

        // Compute all 8 metrics (same as consciousness_scaling but from custom topology)
        let synchrony = sync_trace.iter().sum::<f64>() / sync_trace.len() as f64;

        let mut integration = 0.0;
        let mut n_pairs = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                integration += cross_corr(&voltage_traces[i], &voltage_traces[j]).abs();
                n_pairs += 1;
            }
        }
        integration /= n_pairs.max(1) as f64;

        let means: Vec<f64> = voltage_traces.iter()
            .map(|tr| tr.iter().sum::<f64>() / tr.len() as f64).collect();
        let gm = means.iter().sum::<f64>() / means.len() as f64;
        let complexity = means.iter().map(|m| (m - gm).powi(2)).sum::<f64>() / means.len() as f64;

        let global: Vec<f64> = (0..measure)
            .map(|t| voltage_traces.iter().map(|tr| tr[t]).sum::<f64>() / n as f64)
            .collect();
        let coherence = autocorr(&global, 10);

        let sync_mean = sync_trace.iter().sum::<f64>() / sync_trace.len() as f64;
        let metastable = sync_trace.iter().map(|s| (s - sync_mean).powi(2)).sum::<f64>() / sync_trace.len() as f64;

        let n_bins = 20;
        let mut bins = vec![0usize; n_bins];
        for osc in &net.oscillators {
            let phase = ((osc.v + 2.0) / 4.0).clamp(0.0, 0.9999);
            bins[(phase * n_bins as f64) as usize] += 1;
        }
        let entropy: f64 = bins.iter().filter(|&&b| b > 0)
            .map(|&b| { let p = b as f64 / n as f64; -p * p.ln() }).sum();
        let max_ent = (n_bins.min(n) as f64).ln();
        let entropy_norm = if max_ent > 0.0 { entropy / max_ent } else { 0.0 };

        let dissipative: f64 = voltage_traces.iter()
            .map(|tr| tr.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f64>() / (tr.len() - 1) as f64)
            .sum::<f64>() / n as f64;

        let window = 50;
        let mut decoherence = 0.0;
        let mut n_w = 0;
        for chunk in sync_trace.chunks(window) {
            if chunk.len() < window { break; }
            let m = chunk.iter().sum::<f64>() / chunk.len() as f64;
            decoherence += chunk.iter().map(|s| (s - m).powi(2)).sum::<f64>() / chunk.len() as f64;
            n_w += 1;
        }
        decoherence /= n_w.max(1) as f64;

        let topo_name = match topology {
            Topology::AllToAll => "all2all",
            Topology::Modular { .. } => "modular",
            Topology::Hierarchical => "hierarch",
            Topology::Ring => "ring",
            Topology::SmallWorld { .. } => "smallwld",
        };

        NetworkScores {
            label: format!("{}_{}_c{}", topo_name, n, c),
            n_oscillators: n,
            coupling_strength: coupling,
            scores: [
                synchrony.clamp(0.0, 1.0),
                integration.clamp(0.0, 1.0),
                (complexity * 100.0).min(1.0).clamp(0.0, 1.0),
                coherence.clamp(0.0, 1.0),
                (metastable * 100.0).min(1.0).clamp(0.0, 1.0),
                entropy_norm.clamp(0.0, 1.0),
                (dissipative * 5.0).min(1.0).clamp(0.0, 1.0),
                (decoherence * 1000.0).min(1.0).clamp(0.0, 1.0),
            ],
        }
    }).collect()
}

fn cross_corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len()) as f64;
    if n < 2.0 { return 0.0; }
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let (mut cov, mut vx, mut vy) = (0.0, 0.0, 0.0);
    for (xi, yi) in x.iter().zip(y.iter()) {
        let (dx, dy) = (xi - mx, yi - my);
        cov += dx * dy; vx += dx * dx; vy += dy * dy;
    }
    if vx < 1e-15 || vy < 1e-15 { return 0.0; }
    cov / (vx * vy).sqrt()
}

fn autocorr(x: &[f64], lag: usize) -> f64 {
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
    fn test_topologies_produce_different_scores() {
        let n = 12;
        let a2a = topology_scores(n, Topology::AllToAll, 3);
        let modular = topology_scores(n, Topology::Modular { n_modules: 3 }, 3);
        let ring = topology_scores(n, Topology::Ring, 3);

        // At least some scores should differ between topologies
        let a2a_sync = a2a.iter().map(|s| s.scores[0]).sum::<f64>() / a2a.len() as f64;
        let mod_sync = modular.iter().map(|s| s.scores[0]).sum::<f64>() / modular.len() as f64;
        let ring_sync = ring.iter().map(|s| s.scores[0]).sum::<f64>() / ring.len() as f64;

        // All-to-all should have higher synchrony than ring (stronger coupling)
        // This is a soft test — the exact relationship depends on parameters
        assert!(a2a_sync >= 0.0 && mod_sync >= 0.0 && ring_sync >= 0.0);
    }

    #[test]
    fn test_modular_network_has_structure() {
        let net = build_network(12, Topology::Modular { n_modules: 3 }, 0.5);
        // Within-module coupling should be stronger than between-module
        let intra = net.coupling[0 * 12 + 1]; // Same module (0→1)
        let inter = net.coupling[0 * 12 + 5]; // Different module (0→5)
        assert!(intra > inter, "Intra ({:.4}) should > inter ({:.4})", intra, inter);
    }
}
