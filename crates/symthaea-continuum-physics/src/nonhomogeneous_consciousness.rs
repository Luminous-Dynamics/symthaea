// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Non-Homogeneous Consciousness: testing the I×D theorem against reality.
//!
//! Extends the tradeoff analysis to:
//! 1. Small-world (Watts-Strogatz): tight clusters + random long-range bridges
//! 2. Scale-free (Barabási-Albert): power-law hubs ("plutocratic" topology)
//! 3. Mixed tiers: heterogeneous oscillator types (Observer→Guardian spectrum)
//!
//! The question: does hierarchy still win when networks are messy?

use crate::id_tradeoff::{TradeoffPoint};
use crate::nonlinear_waves::OscillatorNetwork;

// ── Topology Builders ───────────────────────────────────────────────────────

/// Watts-Strogatz small-world: ring + rewired long-range bridges.
/// k = number of nearest neighbors per side, p = rewiring probability.
pub fn build_small_world(n: usize, g: f64, k: usize, rewire_prob: f64) -> Vec<f64> {
    let mut coupling = vec![0.0; n * n];
    let w = g / (2 * k).max(1) as f64;

    // Start with k-nearest-neighbor ring
    for i in 0..n {
        for d in 1..=k {
            let right = (i + d) % n;
            let left = (i + n - d) % n;
            coupling[i * n + right] = w;
            coupling[i * n + left] = w;
        }
    }

    // Rewire with probability p (deterministic pseudo-random)
    let mut seed = 314159u64;
    for i in 0..n {
        for d in 1..=k {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let prob = (seed >> 33) as f64 / (1u64 << 31) as f64;
            if prob < rewire_prob {
                let right = (i + d) % n;
                // Remove old link
                coupling[i * n + right] = 0.0;
                // Add random long-range link
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let target = ((seed >> 33) as f64 / (1u64 << 31) as f64 * n as f64) as usize % n;
                if target != i {
                    coupling[i * n + target] = w;
                    coupling[target * n + i] = w; // Symmetric
                }
            }
        }
    }

    coupling
}

/// Barabási-Albert scale-free: preferential attachment creates power-law hubs.
/// m = number of links each new node adds (m=2 typical).
pub fn build_scale_free(n: usize, g: f64, m: usize) -> Vec<f64> {
    let mut coupling = vec![0.0; n * n];
    let mut degree = vec![0usize; n];

    // Start with a fully connected seed of m+1 nodes
    for i in 0..=m.min(n - 1) {
        for j in (i + 1)..=m.min(n - 1) {
            coupling[i * n + j] = 1.0;
            coupling[j * n + i] = 1.0;
            degree[i] += 1;
            degree[j] += 1;
        }
    }

    // Add remaining nodes with preferential attachment
    let mut seed = 271828u64;
    for new_node in (m + 1)..n {
        let total_degree: usize = degree.iter().sum();
        let mut links_added = 0;
        let mut attempts = 0;

        while links_added < m && attempts < n * 10 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = (seed >> 33) as f64 / (1u64 << 31) as f64;

            // Preferential attachment: probability ∝ degree
            let mut cumulative = 0.0;
            let mut target = 0;
            for j in 0..new_node {
                cumulative += degree[j] as f64 / total_degree.max(1) as f64;
                if r < cumulative {
                    target = j;
                    break;
                }
            }

            if coupling[new_node * n + target] == 0.0 && target != new_node {
                coupling[new_node * n + target] = 1.0;
                coupling[target * n + new_node] = 1.0;
                degree[new_node] += 1;
                degree[target] += 1;
                links_added += 1;
            }
            attempts += 1;
        }
    }

    // Normalize coupling weights by degree and scale by g
    for i in 0..n {
        let deg = degree[i].max(1) as f64;
        for j in 0..n {
            if coupling[i * n + j] > 0.0 {
                coupling[i * n + j] = g / deg;
            }
        }
    }

    coupling
}

/// Build hierarchical coupling (reused from id_tradeoff but as standalone fn).
pub fn build_hierarchical(n: usize, g: f64) -> Vec<f64> {
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

/// Build all-to-all (for comparison).
pub fn build_all_to_all(n: usize, g: f64) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    let w = g / (n - 1).max(1) as f64;
    for i in 0..n { for j in 0..n { if i != j { c[i * n + j] = w; } } }
    c
}

// ── I×D Measurement ─────────────────────────────────────────────────────────

/// Compute I, D, and I×D for a network with custom coupling and oscillator params.
pub fn compute_id(
    n: usize,
    coupling: &[f64],
    i_ext_values: &[f64], // Per-oscillator drive currents
) -> TradeoffPoint {
    let dt = 0.1;
    let warmup = 3000;
    let measure = 5000;

    let mut net = OscillatorNetwork {
        oscillators: (0..n).map(|i| {
            let mut osc = crate::nonlinear_waves::FitzHughNagumo::new();
            osc.i_ext = i_ext_values[i];
            osc.v = -1.0 + 0.2 * (i as f64 / n as f64);
            osc
        }).collect(),
        coupling: coupling.to_vec(),
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

    let mut integration = 0.0;
    let mut n_pairs = 0usize;
    for i in 0..n { for j in (i+1)..n {
        integration += pearson_abs(&traces[i], &traces[j]);
        n_pairs += 1;
    }}
    integration /= n_pairs.max(1) as f64;

    let rates: Vec<f64> = traces.iter().map(|tr| {
        tr.windows(2).filter(|w| w[0] < 0.0 && w[1] >= 0.0).count() as f64 / (measure as f64 * dt)
    }).collect();
    let rate_mean = rates.iter().sum::<f64>() / n as f64;
    let differentiation = if n > 1 {
        (rates.iter().map(|r| (r - rate_mean).powi(2)).sum::<f64>() / n as f64 * 100.0).min(1.0)
    } else { 0.0 };

    let synchrony = sync_trace.iter().sum::<f64>() / sync_trace.len() as f64;
    let sync_mean = synchrony;
    let metastability = (sync_trace.iter().map(|s| (s - sync_mean).powi(2)).sum::<f64>()
        / sync_trace.len() as f64 * 50.0).min(1.0);

    TradeoffPoint {
        coupling: 0.0, // Set by caller
        integration,
        differentiation,
        id_product: integration * differentiation,
        synchrony,
        metastability,
    }
}

/// Coupling sweep for a topology builder.
pub fn sweep(
    n: usize,
    n_points: usize,
    build_fn: &dyn Fn(usize, f64) -> Vec<f64>,
    i_ext_values: &[f64],
) -> Vec<TradeoffPoint> {
    (0..n_points).map(|i| {
        let frac = i as f64 / (n_points - 1) as f64;
        let g = 0.001 * (2000.0_f64).powf(frac);
        let coupling = build_fn(n, g);
        let mut point = compute_id(n, &coupling, i_ext_values);
        point.coupling = g;
        point
    }).collect()
}

// ── Tier System ─────────────────────────────────────────────────────────────

/// Five participation tiers with different intrinsic dynamics.
/// Observer: low drive, slow dynamics (I_ext = 0.33, barely oscillating)
/// Citizen: moderate drive (I_ext = 0.45)
/// Steward: standard drive (I_ext = 0.55)
/// Elder: high drive (I_ext = 0.65)
/// Guardian: very high drive, fast dynamics (I_ext = 0.75)
pub fn five_tier_drives(n: usize) -> Vec<f64> {
    let tier_fractions = [0.30, 0.30, 0.20, 0.15, 0.05]; // Observer, Citizen, Steward, Elder, Guardian
    let tier_drives = [0.33, 0.45, 0.55, 0.65, 0.75];

    let mut drives = Vec::with_capacity(n);
    let mut assigned = 0;
    for (t, &frac) in tier_fractions.iter().enumerate() {
        let count = if t == tier_fractions.len() - 1 {
            n - assigned
        } else {
            (n as f64 * frac).round() as usize
        };
        for _ in 0..count {
            drives.push(tier_drives[t]);
        }
        assigned += count;
    }
    // Truncate or pad
    drives.truncate(n);
    while drives.len() < n { drives.push(0.5); }
    drives
}

/// Uniform drives (for comparison).
pub fn uniform_drives(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.4 + 0.4 * (i as f64 / n as f64 - 0.5)).collect()
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
    fn test_small_world_builds() {
        let c = build_small_world(20, 0.5, 2, 0.3);
        assert_eq!(c.len(), 400);
        // Should have some non-zero entries
        assert!(c.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn test_scale_free_has_hubs() {
        let c = build_scale_free(20, 0.5, 2);
        // Compute degree per node
        let degrees: Vec<usize> = (0..20).map(|i|
            (0..20).filter(|&j| c[i * 20 + j] > 0.0).count()
        ).collect();
        let max_deg = *degrees.iter().max().unwrap();
        let min_deg = *degrees.iter().min().unwrap();
        // Scale-free should have heterogeneous degrees
        assert!(max_deg > min_deg, "Should have hubs: max={}, min={}", max_deg, min_deg);
    }

    #[test]
    fn test_five_tier_drives() {
        let drives = five_tier_drives(20);
        assert_eq!(drives.len(), 20);
        let min = drives.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = drives.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max > min, "Tiers should create diversity");
    }
}
