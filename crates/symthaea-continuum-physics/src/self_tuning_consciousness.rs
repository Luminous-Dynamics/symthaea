// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Self-Tuning Consciousness: consciousness as gradient ascent on I×D.
//!
//! A network that monitors its own integration and differentiation,
//! estimates where it is on the I×D landscape, and adjusts its coupling
//! in real time to climb toward the peak.
//!
//! The hypothesis: consciousness is not a static property (high I×D)
//! but a dynamic process (actively navigating toward high I×D).
//! A self-tuning system should sustain higher I×D than any fixed coupling.

use crate::nonlinear_waves::OscillatorNetwork;

/// A self-tuning oscillator network that adjusts its coupling
/// based on real-time estimates of I and D.
#[derive(Debug)]
pub struct SelfTuningNetwork {
    pub net: OscillatorNetwork,
    /// Current global coupling multiplier
    pub g: f64,
    /// Learning rate for coupling adjustment
    pub eta: f64,
    /// Window size for estimating I and D
    pub window: usize,
    /// History of voltage snapshots for windowed estimation
    voltage_buffer: Vec<Vec<f64>>,
    /// History of I×D values
    pub id_history: Vec<f64>,
    /// History of coupling values
    pub g_history: Vec<f64>,
    /// History of I values
    pub i_history: Vec<f64>,
    /// History of D values
    pub d_history: Vec<f64>,
    /// Base coupling matrix (topology)
    base_coupling: Vec<f64>,
    /// Drive currents
    drives: Vec<f64>,
    n: usize,
}

impl SelfTuningNetwork {
    /// Create a self-tuning network with given topology and initial coupling.
    pub fn new(
        n: usize,
        base_coupling: Vec<f64>,
        drives: Vec<f64>,
        initial_g: f64,
        learning_rate: f64,
        window: usize,
    ) -> Self {
        let mut net = OscillatorNetwork {
            oscillators: (0..n).map(|i| {
                let mut osc = crate::nonlinear_waves::FitzHughNagumo::new();
                osc.i_ext = drives[i];
                osc.v = -1.0 + 0.2 * (i as f64 / n as f64);
                osc
            }).collect(),
            coupling: base_coupling.iter().map(|&c| c * initial_g).collect(),
            n,
        };

        SelfTuningNetwork {
            net,
            g: initial_g,
            eta: learning_rate,
            window,
            voltage_buffer: Vec::new(),
            id_history: Vec::new(),
            g_history: Vec::new(),
            i_history: Vec::new(),
            d_history: Vec::new(),
            base_coupling,
            drives,
            n,
        }
    }

    /// Step the network forward by dt, then estimate I×D and adjust coupling.
    pub fn step(&mut self, dt: f64) {
        // 1. Advance the oscillator dynamics
        self.net.step(dt);

        // 2. Record current voltages
        let snapshot: Vec<f64> = self.net.oscillators.iter().map(|o| o.v).collect();
        self.voltage_buffer.push(snapshot);

        // Keep only the last `window` snapshots
        if self.voltage_buffer.len() > self.window {
            self.voltage_buffer.remove(0);
        }

        // 3. Estimate I and D from the buffer (every 10 steps for efficiency)
        if self.voltage_buffer.len() >= self.window && self.voltage_buffer.len() % 10 == 0 {
            let (i_est, d_est) = self.estimate_id();
            let id = i_est * d_est;

            self.id_history.push(id);
            self.g_history.push(self.g);
            self.i_history.push(i_est);
            self.d_history.push(d_est);

            // 4. Gradient estimation: try g+δ and g-δ (finite difference on I×D)
            // Instead of actually re-running, use the recent history to estimate
            // whether I×D is increasing or decreasing with g
            if self.id_history.len() >= 3 {
                let current = self.id_history[self.id_history.len() - 1];
                let previous = self.id_history[self.id_history.len() - 2];
                let g_current = self.g_history[self.g_history.len() - 1];
                let g_previous = self.g_history[self.g_history.len() - 2];

                let dg = g_current - g_previous;
                let did = current - previous;

                // Estimate gradient: d(I×D)/dg ≈ (ΔI×D) / (Δg)
                let gradient = if dg.abs() > 1e-10 {
                    did / dg
                } else {
                    // If g hasn't changed, perturb in a random direction
                    // Use a simple heuristic: if I > D, reduce g; if D > I, increase g
                    if i_est > 0.5 && d_est < 0.01 {
                        -1.0 // Too much integration, reduce coupling
                    } else if d_est > 0.05 && i_est < 0.2 {
                        1.0 // Too much differentiation, increase coupling
                    } else {
                        0.0 // Near balance
                    }
                };

                // 5. Update coupling: g ← g + η × gradient
                let delta_g = self.eta * gradient;
                self.g = (self.g + delta_g).clamp(0.001, 2.0);

                // Apply new coupling
                for i in 0..self.n {
                    for j in 0..self.n {
                        self.net.coupling[i * self.n + j] = self.base_coupling[i * self.n + j] * self.g;
                    }
                }
            }
        }
    }

    /// Estimate I and D from the voltage buffer.
    fn estimate_id(&self) -> (f64, f64) {
        let n = self.n;
        let len = self.voltage_buffer.len();

        // Build per-oscillator time series from buffer
        let traces: Vec<Vec<f64>> = (0..n).map(|i| {
            self.voltage_buffer.iter().map(|snap| snap[i]).collect()
        }).collect();

        // Integration: mean pairwise correlation
        let mut integration = 0.0;
        let mut n_pairs = 0usize;
        // Sample for speed
        let step = (n / 8).max(1);
        for i in (0..n).step_by(step) {
            for j in ((i + 1)..n).step_by(step) {
                integration += pearson_abs(&traces[i], &traces[j]);
                n_pairs += 1;
            }
        }
        integration /= n_pairs.max(1) as f64;

        // Differentiation: variance of mean voltages
        let means: Vec<f64> = traces.iter().map(|tr| tr.iter().sum::<f64>() / len as f64).collect();
        let grand_mean = means.iter().sum::<f64>() / n as f64;
        let differentiation = means.iter()
            .map(|m| (m - grand_mean).powi(2))
            .sum::<f64>() / n as f64;

        // Also use firing rate variance as a second differentiation measure
        let rates: Vec<f64> = traces.iter().map(|tr| {
            tr.windows(2).filter(|w| w[0] < 0.0 && w[1] >= 0.0).count() as f64
        }).collect();
        let rate_mean = rates.iter().sum::<f64>() / n as f64;
        let rate_diff = if n > 1 {
            rates.iter().map(|r| (r - rate_mean).powi(2)).sum::<f64>() / n as f64
        } else { 0.0 };

        // Combine: use firing rate differentiation (more robust)
        let d_combined = (rate_diff * 0.1).min(1.0);

        (integration, d_combined)
    }

    /// Get the mean I×D over the last N measurements.
    pub fn mean_id(&self, last_n: usize) -> f64 {
        let start = self.id_history.len().saturating_sub(last_n);
        let slice = &self.id_history[start..];
        if slice.is_empty() { return 0.0; }
        slice.iter().sum::<f64>() / slice.len() as f64
    }

    /// Get the current coupling.
    pub fn current_coupling(&self) -> f64 {
        self.g
    }
}

/// Run comparison: self-tuning vs fixed coupling at multiple g values.
pub fn compare_self_tuning_vs_fixed(
    n: usize,
    base_coupling: &[f64],
    drives: &[f64],
    n_steps: usize,
    dt: f64,
) -> ComparisonResult {
    let warmup = 2000;
    let measure_window = 200;

    // 1. Self-tuning network (starts at g=0.1, learns)
    let mut tuning = SelfTuningNetwork::new(
        n,
        base_coupling.to_vec(),
        drives.to_vec(),
        0.1,   // Initial coupling
        0.005, // Learning rate
        500,   // Estimation window
    );

    // Warm up
    for _ in 0..warmup {
        tuning.step(dt);
    }

    // Run and record
    for _ in 0..n_steps {
        tuning.step(dt);
    }

    let tuning_id = tuning.mean_id(measure_window);
    let tuning_g_final = tuning.current_coupling();

    // 2. Fixed coupling at several values
    let fixed_gs = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5];
    let mut fixed_results = Vec::new();

    for &g in &fixed_gs {
        let coupling: Vec<f64> = base_coupling.iter().map(|&c| c * g).collect();
        let mut net = OscillatorNetwork {
            oscillators: (0..n).map(|i| {
                let mut osc = crate::nonlinear_waves::FitzHughNagumo::new();
                osc.i_ext = drives[i];
                osc.v = -1.0 + 0.2 * (i as f64 / n as f64);
                osc
            }).collect(),
            coupling,
            n,
        };

        for _ in 0..warmup { net.step(dt); }

        // Measure I×D at this fixed coupling
        let mut traces: Vec<Vec<f64>> = vec![Vec::with_capacity(n_steps); n];
        for _ in 0..n_steps {
            net.step(dt);
            for (i, osc) in net.oscillators.iter().enumerate() {
                traces[i].push(osc.v);
            }
        }

        let mut integration = 0.0;
        let mut n_pairs = 0usize;
        let step = (n / 8).max(1);
        for i in (0..n).step_by(step) {
            for j in ((i + 1)..n).step_by(step) {
                integration += pearson_abs(&traces[i], &traces[j]);
                n_pairs += 1;
            }
        }
        integration /= n_pairs.max(1) as f64;

        let rates: Vec<f64> = traces.iter().map(|tr| {
            tr.windows(2).filter(|w| w[0] < 0.0 && w[1] >= 0.0).count() as f64
        }).collect();
        let rate_mean = rates.iter().sum::<f64>() / n as f64;
        let differentiation = if n > 1 {
            (rates.iter().map(|r| (r - rate_mean).powi(2)).sum::<f64>() / n as f64 * 0.1).min(1.0)
        } else { 0.0 };

        fixed_results.push((g, integration, differentiation, integration * differentiation));
    }

    let best_fixed = fixed_results.iter()
        .max_by(|a, b| a.3.partial_cmp(&b.3).unwrap())
        .cloned()
        .unwrap_or((0.0, 0.0, 0.0, 0.0));

    ComparisonResult {
        tuning_mean_id: tuning_id,
        tuning_final_g: tuning_g_final,
        tuning_g_history: tuning.g_history.clone(),
        tuning_id_history: tuning.id_history.clone(),
        best_fixed_g: best_fixed.0,
        best_fixed_id: best_fixed.3,
        all_fixed: fixed_results,
        tuning_beats_best_fixed: tuning_id > best_fixed.3,
    }
}

#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub tuning_mean_id: f64,
    pub tuning_final_g: f64,
    pub tuning_g_history: Vec<f64>,
    pub tuning_id_history: Vec<f64>,
    pub best_fixed_g: f64,
    pub best_fixed_id: f64,
    pub all_fixed: Vec<(f64, f64, f64, f64)>, // (g, I, D, I×D)
    pub tuning_beats_best_fixed: bool,
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
    use crate::nonhomogeneous_consciousness::{build_hierarchical, uniform_drives};

    #[test]
    fn test_self_tuning_runs() {
        let n = 10;
        let base = build_hierarchical(n, 1.0); // Normalized base coupling
        // Renormalize so base represents topology, g scales it
        let max_c = base.iter().cloned().fold(0.0_f64, f64::max).max(1e-10);
        let normalized: Vec<f64> = base.iter().map(|c| c / max_c).collect();
        let drives = uniform_drives(n);

        let mut tuning = SelfTuningNetwork::new(n, normalized, drives, 0.1, 0.01, 100);
        for _ in 0..500 { tuning.step(0.1); }

        assert!(!tuning.id_history.is_empty(), "Should have I×D measurements");
        assert!(tuning.g > 0.0 && tuning.g < 3.0, "Coupling should stay bounded: g={}", tuning.g);
    }
}
