// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Predictive Consciousness: active inference on the I×D landscape.
//!
//! The gradient-following self-tuner failed because it can't see the landscape.
//! This module gives the network an INTERNAL MODEL: separate estimates of
//! dI/dg and dD/dg, which it uses to predict where I×D peaks.
//!
//! If this model-based tuner finds the sweet spot and the gradient tuner doesn't,
//! that's a computational proof that consciousness requires prediction —
//! a derivation of the Free Energy Principle from the I×D landscape.

use crate::nonlinear_waves::OscillatorNetwork;

/// A network with an internal model of its own I×D landscape.
pub struct PredictiveNetwork {
    pub net: OscillatorNetwork,
    n: usize,
    pub g: f64,
    base_coupling: Vec<f64>,
    drives: Vec<f64>,
    window: usize,
    voltage_buffer: Vec<Vec<f64>>,

    // The internal model: estimates of I(g) and D(g) at different g values
    // Stored as (g, I, D) triples from exploration
    model: Vec<(f64, f64, f64)>,

    // Exploration state
    explore_phase: bool,
    explore_step: usize,
    explore_g_values: Vec<f64>,

    // History
    pub g_history: Vec<f64>,
    pub id_history: Vec<f64>,
    pub i_history: Vec<f64>,
    pub d_history: Vec<f64>,
    pub predicted_g_star: Vec<f64>,
}

impl PredictiveNetwork {
    pub fn new(
        n: usize,
        base_coupling: Vec<f64>,
        drives: Vec<f64>,
        window: usize,
    ) -> Self {
        let initial_g = 0.1;
        let net = OscillatorNetwork {
            oscillators: (0..n).map(|i| {
                let mut osc = crate::nonlinear_waves::FitzHughNagumo::new();
                osc.i_ext = drives[i];
                osc.v = -1.0 + 0.2 * (i as f64 / n as f64);
                osc
            }).collect(),
            coupling: base_coupling.iter().map(|&c| c * initial_g).collect(),
            n,
        };

        // Exploration: sample g at logarithmic intervals
        let explore_g_values: Vec<f64> = (0..8).map(|i| {
            0.003 * (300.0_f64).powf(i as f64 / 7.0) // 0.003 to 0.9
        }).collect();

        PredictiveNetwork {
            net, n,
            g: initial_g,
            base_coupling,
            drives,
            window,
            voltage_buffer: Vec::new(),
            model: Vec::new(),
            explore_phase: true,
            explore_step: 0,
            explore_g_values,
            g_history: Vec::new(),
            id_history: Vec::new(),
            i_history: Vec::new(),
            d_history: Vec::new(),
            predicted_g_star: Vec::new(),
        }
    }

    pub fn step(&mut self, dt: f64) {
        self.net.step(dt);

        let snapshot: Vec<f64> = self.net.oscillators.iter().map(|o| o.v).collect();
        self.voltage_buffer.push(snapshot);
        if self.voltage_buffer.len() > self.window {
            self.voltage_buffer.remove(0);
        }

        if self.voltage_buffer.len() < self.window { return; }

        // Measure every `window/2` steps
        if self.voltage_buffer.len() % (self.window / 2).max(1) != 0 { return; }

        let (i_est, d_est) = self.estimate_id();
        let id = i_est * d_est;

        self.g_history.push(self.g);
        self.id_history.push(id);
        self.i_history.push(i_est);
        self.d_history.push(d_est);

        if self.explore_phase {
            // PHASE 1: EXPLORATION — sample the landscape
            self.model.push((self.g, i_est, d_est));
            self.explore_step += 1;

            if self.explore_step < self.explore_g_values.len() {
                // Move to next exploration point
                self.g = self.explore_g_values[self.explore_step];
                self.apply_coupling();
                self.voltage_buffer.clear(); // Reset measurement buffer
            } else {
                // Exploration complete — switch to exploitation
                self.explore_phase = false;

                // Use the model to predict g*
                if let Some(g_star) = self.predict_optimal_coupling() {
                    self.predicted_g_star.push(g_star);
                    self.g = g_star;
                    self.apply_coupling();
                    self.voltage_buffer.clear();
                }
            }
        } else {
            // PHASE 2: EXPLOITATION — maintain predicted g*
            // Periodically re-estimate and correct
            self.model.push((self.g, i_est, d_est));

            // Every 20 measurements, re-predict g*
            if self.id_history.len() % 20 == 0 {
                if let Some(g_star) = self.predict_optimal_coupling() {
                    self.predicted_g_star.push(g_star);
                    // Slowly move toward predicted optimum
                    self.g = 0.8 * self.g + 0.2 * g_star;
                    self.apply_coupling();
                }
            }
        }
    }

    /// Predict the optimal coupling g* from the internal model.
    ///
    /// Fits separate models for I(g) and D(g), then finds where I×D peaks.
    fn predict_optimal_coupling(&self) -> Option<f64> {
        if self.model.len() < 3 { return None; }

        // Find g that maximizes I×D among ALL observed points
        let mut best_g = self.model[0].0;
        let mut best_id = 0.0_f64;

        for &(g, i, d) in &self.model {
            let id = i * d;
            if id > best_id {
                best_id = id;
                best_g = g;
            }
        }

        // Also try interpolated points between the two best
        let mut sorted: Vec<(f64, f64, f64)> = self.model.clone();
        sorted.sort_by(|a, b| {
            let id_a = a.1 * a.2;
            let id_b = b.1 * b.2;
            id_b.partial_cmp(&id_a).unwrap()
        });

        if sorted.len() >= 2 {
            let g1 = sorted[0].0;
            let g2 = sorted[1].0;
            // Try the midpoint
            let g_mid = (g1 + g2) / 2.0;
            if g_mid > 0.001 && g_mid < 2.0 {
                // The predicted g* is between the two best observed points
                best_g = g_mid;
            }
        }

        Some(best_g.clamp(0.001, 2.0))
    }

    fn apply_coupling(&mut self) {
        for i in 0..self.n {
            for j in 0..self.n {
                self.net.coupling[i * self.n + j] = self.base_coupling[i * self.n + j] * self.g;
            }
        }
    }

    fn estimate_id(&self) -> (f64, f64) {
        let n = self.n;
        let len = self.voltage_buffer.len();
        let traces: Vec<Vec<f64>> = (0..n).map(|i| {
            self.voltage_buffer.iter().map(|snap| snap[i]).collect()
        }).collect();

        // Integration
        let mut integration = 0.0;
        let mut n_pairs = 0usize;
        let step = (n / 6).max(1);
        for i in (0..n).step_by(step) {
            for j in ((i + 1)..n).step_by(step) {
                integration += pearson_abs(&traces[i], &traces[j]);
                n_pairs += 1;
            }
        }
        integration /= n_pairs.max(1) as f64;

        // Differentiation via firing rates
        let rates: Vec<f64> = traces.iter().map(|tr| {
            tr.windows(2).filter(|w| w[0] < 0.0 && w[1] >= 0.0).count() as f64
        }).collect();
        let rate_mean = rates.iter().sum::<f64>() / n as f64;
        let differentiation = if n > 1 {
            (rates.iter().map(|r| (r - rate_mean).powi(2)).sum::<f64>() / n as f64 * 0.1).min(1.0)
        } else { 0.0 };

        (integration, differentiation)
    }

    pub fn mean_id(&self, last_n: usize) -> f64 {
        let start = self.id_history.len().saturating_sub(last_n);
        let slice = &self.id_history[start..];
        if slice.is_empty() { return 0.0; }
        slice.iter().sum::<f64>() / slice.len() as f64
    }
}

/// Run the definitive comparison: predictive vs gradient vs fixed.
pub fn three_way_comparison(
    n: usize,
    base_coupling: &[f64],
    drives: &[f64],
    n_steps: usize,
    dt: f64,
) -> ThreeWayResult {
    let warmup = 1000;

    // 1. Predictive (model-based) network
    let mut predictive = PredictiveNetwork::new(
        n, base_coupling.to_vec(), drives.to_vec(), 400,
    );
    for _ in 0..warmup { predictive.step(dt); }
    for _ in 0..n_steps { predictive.step(dt); }
    let pred_id = predictive.mean_id(200);
    let pred_g = predictive.g;
    let pred_g_star = predictive.predicted_g_star.last().copied().unwrap_or(0.0);

    // 2. Gradient-following (the one that failed before)
    let mut gradient = crate::self_tuning_consciousness::SelfTuningNetwork::new(
        n, base_coupling.to_vec(), drives.to_vec(), 0.1, 0.005, 500,
    );
    for _ in 0..warmup { gradient.step(dt); }
    for _ in 0..n_steps { gradient.step(dt); }
    let grad_id = gradient.mean_id(200);
    let grad_g = gradient.current_coupling();

    // 3. Fixed coupling sweep
    let fixed_gs = [0.003, 0.008, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0];
    let mut best_fixed = (0.0, 0.0_f64);
    let mut all_fixed = Vec::new();

    for &g in &fixed_gs {
        let coupling: Vec<f64> = base_coupling.iter().map(|&c| c * g).collect();
        let mut net = OscillatorNetwork {
            oscillators: (0..n).map(|i| {
                let mut osc = crate::nonlinear_waves::FitzHughNagumo::new();
                osc.i_ext = drives[i];
                osc.v = -1.0 + 0.2 * (i as f64 / n as f64);
                osc
            }).collect(),
            coupling, n,
        };
        for _ in 0..warmup { net.step(dt); }

        let mut traces: Vec<Vec<f64>> = vec![Vec::with_capacity(n_steps); n];
        for _ in 0..n_steps {
            net.step(dt);
            for (i, osc) in net.oscillators.iter().enumerate() {
                traces[i].push(osc.v);
            }
        }

        let mut integration = 0.0;
        let mut np = 0usize;
        let step = (n / 6).max(1);
        for i in (0..n).step_by(step) {
            for j in ((i+1)..n).step_by(step) {
                integration += pearson_abs(&traces[i], &traces[j]);
                np += 1;
            }
        }
        integration /= np.max(1) as f64;

        let rates: Vec<f64> = traces.iter().map(|tr| {
            tr.windows(2).filter(|w| w[0] < 0.0 && w[1] >= 0.0).count() as f64
        }).collect();
        let rm = rates.iter().sum::<f64>() / n as f64;
        let diff = if n > 1 { (rates.iter().map(|r| (r - rm).powi(2)).sum::<f64>() / n as f64 * 0.1).min(1.0) } else { 0.0 };
        let id = integration * diff;

        all_fixed.push((g, integration, diff, id));
        if id > best_fixed.1 { best_fixed = (g, id); }
    }

    ThreeWayResult {
        predictive_id: pred_id,
        predictive_g: pred_g,
        predictive_g_star: pred_g_star,
        gradient_id: grad_id,
        gradient_g: grad_g,
        best_fixed_g: best_fixed.0,
        best_fixed_id: best_fixed.1,
        all_fixed,
        predictive_model_points: predictive.model.len(),
    }
}

#[derive(Debug)]
pub struct ThreeWayResult {
    pub predictive_id: f64,
    pub predictive_g: f64,
    pub predictive_g_star: f64,
    pub gradient_id: f64,
    pub gradient_g: f64,
    pub best_fixed_g: f64,
    pub best_fixed_id: f64,
    pub all_fixed: Vec<(f64, f64, f64, f64)>,
    pub predictive_model_points: usize,
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
    fn test_predictive_network_runs() {
        let n = 10;
        let raw = build_hierarchical(n, 1.0);
        let max_c = raw.iter().cloned().fold(0.0_f64, f64::max).max(1e-10);
        let base: Vec<f64> = raw.iter().map(|c| c / max_c).collect();
        let drives = uniform_drives(n);

        let mut pn = PredictiveNetwork::new(n, base, drives, 200);
        for _ in 0..2000 { pn.step(0.1); }

        assert!(!pn.id_history.is_empty());
        assert!(pn.g > 0.0 && pn.g < 3.0);
    }
}
