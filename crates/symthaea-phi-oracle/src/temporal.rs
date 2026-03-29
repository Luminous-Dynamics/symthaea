// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use symthaea_core::hdc::{ContinuousHV, HdcLtcUnifiedNeuron, UnifiedActivation, UnifiedConfig};

use crate::result::TemporalCoherence;

/// CV threshold below which a timescale is considered "coherent."
const COHERENCE_CV_THRESHOLD: f64 = 0.5;

/// Temporal coherence prober using CfC (Closed-form Continuous-time) dynamics.
///
/// The CfC neuron acts as a dynamical probe: we feed the same encoded
/// observations through multiple neurons at different timescales (`dt` values),
/// then measure the coefficient of variation (CV) of each neuron's output norm.
///
/// - **Low CV** at a given τ → the system is temporally coherent at that
///   timescale (the probe's output is stable).
/// - **High CV** → the system's dynamics don't align with that τ.
/// - The τ with minimum CV → dominant timescale of integration.
/// - Wide range of low-CV τ values → deep temporal integration.
pub struct TemporalProber {
    /// One CfC neuron per probe timescale.
    probes: Vec<CfCProbe>,
    /// The dt values being probed.
    tau_values: Vec<f64>,
}

struct CfCProbe {
    neuron: HdcLtcUnifiedNeuron,
    /// Running statistics for output norm.
    norms: Vec<f64>,
}

impl TemporalProber {
    /// Create a temporal prober with the given timescale probes.
    ///
    /// - `hv_dim`: dimension of input hypervectors.
    /// - `tau_values`: the `dt` values to probe (e.g., `[0.01, 0.1, 1.0, 10.0]`).
    /// - `seed`: deterministic seed.
    pub fn new(hv_dim: usize, tau_values: &[f64], seed: u64) -> Self {
        let probes = tau_values
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let config = UnifiedConfig {
                    dimension: hv_dim,
                    tau_base: 0.1,
                    backbone_tau: 0.5,
                    activation: UnifiedActivation::Tanh,
                    learning_rate: 0.0, // no learning — pure dynamics
                    momentum: 0.0,
                    weight_decay: 0.0,
                    gating_steepness: 1.0,
                    interp_bias: 0.0,
                    fourier_frequencies: vec![],
                    fourier_amplitude: 0.0,
                };
                CfCProbe {
                    neuron: HdcLtcUnifiedNeuron::new(config, seed + 5000 + i as u64),
                    norms: Vec::new(),
                }
            })
            .collect();

        Self {
            probes,
            tau_values: tau_values.to_vec(),
        }
    }

    /// Feed one encoded observation through all probes at their respective
    /// timescales. Each probe evolves with its own `dt` and records the
    /// output norm.
    pub fn observe(&mut self, encoded: &ContinuousHV) {
        for (probe, &tau) in self.probes.iter_mut().zip(&self.tau_values) {
            probe.neuron.evolve_closed_form(tau as f32, encoded);
            let norm = probe.neuron.state().norm() as f64;
            probe.norms.push(norm);
        }
    }

    /// Compute the temporal coherence profile from accumulated observations.
    ///
    /// Returns `None` if fewer than 3 observations have been recorded (need
    /// enough data for meaningful CV).
    pub fn compute(&self) -> Option<TemporalCoherence> {
        if self.probes.is_empty() || self.probes[0].norms.len() < 3 {
            return None;
        }

        let cv_by_tau: Vec<(f64, f64)> = self
            .tau_values
            .iter()
            .zip(&self.probes)
            .map(|(&tau, probe)| {
                let cv = coefficient_of_variation(&probe.norms);
                (tau, cv)
            })
            .collect();

        // Dominant timescale = tau with minimum CV
        let (dominant_timescale, _min_cv) = cv_by_tau
            .iter()
            .copied()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        // Temporal depth = range of tau values where CV < threshold
        let coherent_taus: Vec<f64> = cv_by_tau
            .iter()
            .filter(|(_, cv)| *cv < COHERENCE_CV_THRESHOLD)
            .map(|(tau, _)| *tau)
            .collect();

        let temporal_depth = if coherent_taus.len() >= 2 {
            let min = coherent_taus
                .iter()
                .copied()
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            let max = coherent_taus
                .iter()
                .copied()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            (max / min).ln() // log-ratio of timescale range
        } else if coherent_taus.len() == 1 {
            0.0 // single coherent timescale — no depth
        } else {
            0.0 // no coherent timescales
        };

        Some(TemporalCoherence {
            cv_by_tau,
            dominant_timescale,
            temporal_depth,
        })
    }

    /// Reset all probes for a new measurement window.
    pub fn reset(&mut self, hv_dim: usize, seed: u64) {
        for (i, probe) in self.probes.iter_mut().enumerate() {
            let config = UnifiedConfig {
                dimension: hv_dim,
                tau_base: 0.1,
                backbone_tau: 0.5,
                activation: UnifiedActivation::Tanh,
                learning_rate: 0.0,
                momentum: 0.0,
                weight_decay: 0.0,
                gating_steepness: 1.0,
                interp_bias: 0.0,
                fourier_frequencies: vec![],
                fourier_amplitude: 0.0,
            };
            probe.neuron = HdcLtcUnifiedNeuron::new(config, seed + 5000 + i as u64);
            probe.norms.clear();
        }
    }
}

/// Coefficient of variation: std_dev / mean. Returns 0.0 for empty or
/// all-zero data to avoid NaN.
fn coefficient_of_variation(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    if mean.abs() < 1e-12 {
        return 0.0;
    }
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    variance.sqrt() / mean.abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cv_constant_signal() {
        let values = vec![5.0; 20];
        let cv = coefficient_of_variation(&values);
        assert!(cv.abs() < 1e-10, "constant signal should have CV ~0");
    }

    #[test]
    fn test_cv_noisy_signal() {
        let values: Vec<f64> = (0..100).map(|i| 10.0 + (i as f64 * 0.37).sin()).collect();
        let cv = coefficient_of_variation(&values);
        assert!(
            cv > 0.0 && cv < 1.0,
            "moderate noise should have 0 < CV < 1, got {cv}"
        );
    }

    #[test]
    fn test_prober_needs_min_observations() {
        let prober = TemporalProber::new(64, &[0.1, 1.0], 42);
        assert!(prober.compute().is_none(), "should need >= 3 observations");
    }

    #[test]
    fn test_prober_produces_coherence() {
        let mut prober = TemporalProber::new(64, &[0.01, 0.1, 1.0, 10.0], 42);
        // Feed a slowly varying signal
        for i in 0..50 {
            let t = i as f64 * 0.1;
            let mut vals = vec![0.0f32; 64];
            for (j, v) in vals.iter_mut().enumerate() {
                *v = ((t + j as f64 * 0.01).sin()) as f32;
            }
            let hv = ContinuousHV::from_vec(vals);
            prober.observe(&hv);
        }
        let coherence = prober.compute().expect("should have enough data");
        assert_eq!(coherence.cv_by_tau.len(), 4);
        assert!(coherence.dominant_timescale > 0.0);
    }
}
