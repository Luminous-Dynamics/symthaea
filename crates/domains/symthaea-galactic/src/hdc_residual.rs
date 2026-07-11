// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! HDC+CfC+GLU residual regressor, adapted from `symthaea-nuclear`'s
//! `HdcMassPredictor` (`hdc_mass.rs`).
//!
//! Diagnostic purpose: train one regressor per gravity model on that model's
//! normalized residuals `(v_obs − v_model) / e_v_obs`. A correct physics
//! model leaves residuals with no learnable radius/galaxy-dependent
//! structure (held-out R² near 0, no better than the mean-predictor
//! baseline); a wrong model leaves structure a flexible learner can pick up.
//! This is exploratory — see the crate README for why held-out R² per model
//! is NOT directly comparable across models with different free-parameter
//! counts (the NFW-vs-0-param asymmetry).
//!
//! Design deviation from `hdc_mass.rs`: nuclear's neuron is trained with
//! `contrastive_update(target_hv, input_hv, lr)`, where `target_hv`
//! re-encodes the *known* binding energy on a dedicated channel — a
//! meaningful positive example. Our target is a bare scalar residual with no
//! natural re-encoding (there is no "residual channel" the encoder exposes,
//! deliberately, since the model doesn't know the answer at prediction
//! time). Rather than fabricate a target encoding, the CfC neuron here acts
//! as a **fixed random nonlinear reservoir**: it evolves the input through
//! HDC-LTC dynamics but is never updated. Only the GLU readout
//! (`w_gates`/`w_values`/`scales`/`biases`) is trained via gradient descent.
//! This is a legitimate, simpler architecture (reservoir computing), not an
//! incomplete port.

use crate::encoder::{GalaxyPointState, GalaxyStateEncoder};
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

const N_HEADS: usize = 1;
const N_EVOLVE_STEPS: usize = 3;
const EVOLVE_DTS: [f32; N_EVOLVE_STEPS] = [0.01, 0.1, 1.0];

/// L2 weight decay on the GLU readout (`w_gates`/`w_values`), applied per
/// example update: `w -= lr * (grad + WEIGHT_DECAY * w)`.
///
/// Added after a loss-curve probe (`loss_curve_probe_newtonian_20_epochs`)
/// showed training RMS converging cleanly (20.9 -> 17.6 over 20 epochs)
/// while held-out R^2 stayed near-zero or negative — consistent with
/// overfitting, since the readout has 2*HDC_DIMENSION=32,768 weights
/// against splits with only ~130 independent training galaxies (points
/// within a galaxy share the same luminosity/distance/inclination/
/// gas-fraction inputs, so the real degrees of freedom are closer to
/// galaxy count than point count).
///
/// **Honest result**: at this value, it did NOT fix the diagnostic. Rerun
/// with WEIGHT_DECAY=0.001 gave held-out R^2 = -0.0711 (n=437 test points),
/// no better than — arguably within noise of — the unregularized epochs=5
/// baseline (-0.0137). Kept anyway because it's sound practice for a model
/// this overparameterized relative to its independent training signal and
/// doesn't measurably hurt, but it should NOT be read as "the fix" — the
/// negative held-out R^2 across every model/epoch-count/regularization
/// combination tried so far (see README) looks like it needs a bigger
/// change (more capacity via multi-head GLU, richer features, or accepting
/// this diagnostic as a documented null result) rather than a tuning knob.
const WEIGHT_DECAY: f64 = 0.001;

/// A single training/evaluation example: encoder input + the target scalar
/// (normalized residual) it should predict.
#[derive(Debug, Clone)]
pub struct ResidualExample {
    pub state: GalaxyPointState,
    /// Normalized residual: (v_obs - v_model) / e_v_obs
    pub target: f64,
}

/// HDC+CfC+GLU regressor over normalized rotation-curve residuals.
pub struct HdcResidualRegressor {
    encoder: GalaxyStateEncoder,
    neuron: HdcLtcUnifiedNeuron,
    w_gates: Vec<Vec<f32>>,
    w_values: Vec<Vec<f32>>,
    scales: Vec<f64>,
    biases: Vec<f64>,
    head_weights: Vec<f64>,
}

impl HdcResidualRegressor {
    /// Create an untrained regressor. `seed_offset` decorrelates independent
    /// instances (e.g. one per gravity model, or per CV fold).
    pub fn new(seed_offset: u64) -> Self {
        let config = UnifiedConfig {
            tau_base: 1.0,
            backbone_tau: 2.0,
            dimension: HDC_DIMENSION,
            learning_rate: 0.005,
            momentum: 0.9,
            weight_decay: 0.0001,
            ..UnifiedConfig::default()
        };

        let mut w_gates = Vec::with_capacity(N_HEADS);
        let mut w_values = Vec::with_capacity(N_HEADS);
        for h in 0..N_HEADS {
            let gi = ContinuousHV::random(
                HDC_DIMENSION,
                0xDA7E_5741 + seed_offset * 100 + h as u64 * 7919,
            );
            let vi = ContinuousHV::random(
                HDC_DIMENSION,
                0xFA1E_5741 + seed_offset * 100 + h as u64 * 6263,
            );
            let mut wg = vec![0.0f32; HDC_DIMENSION];
            let mut wv = vec![0.0f32; HDC_DIMENSION];
            for i in 0..HDC_DIMENSION {
                wg[i] = gi.values[i] * 0.01;
                wv[i] = vi.values[i] * 0.01;
            }
            w_gates.push(wg);
            w_values.push(wv);
        }

        Self {
            encoder: GalaxyStateEncoder::new(),
            neuron: HdcLtcUnifiedNeuron::new(config, 0xCA1A_DEAD + seed_offset),
            w_gates,
            w_values,
            scales: vec![1.0; N_HEADS],
            biases: vec![0.0; N_HEADS],
            head_weights: vec![1.0 / N_HEADS as f64; N_HEADS],
        }
    }

    fn evolved_state(&self, input_hv: &ContinuousHV) -> ContinuousHV {
        let mut neuron = self.neuron.clone();
        neuron.set_state(input_hv.clone());
        for &dt in &EVOLVE_DTS {
            neuron.evolve_closed_form(dt, input_hv);
        }
        neuron.state().clone()
    }

    fn glu_forward(&self, state: &ContinuousHV) -> f64 {
        let mut total = 0.0;
        for h in 0..N_HEADS {
            let gate_dot: f64 = state
                .values
                .iter()
                .zip(self.w_gates[h].iter())
                .map(|(&s, &g)| s as f64 * g as f64)
                .sum();
            let value_dot: f64 = state
                .values
                .iter()
                .zip(self.w_values[h].iter())
                .map(|(&s, &v)| s as f64 * v as f64)
                .sum();
            let gate = 1.0 / (1.0 + (-gate_dot.clamp(-20.0, 20.0)).exp());
            let head_out = gate * (value_dot * self.scales[h] + self.biases[h]);
            total += self.head_weights[h] * head_out;
        }
        total
    }

    /// Predict the normalized residual for a point state.
    pub fn predict(&self, s: &GalaxyPointState) -> f64 {
        let input_hv = self.encoder.encode(s);
        let state = self.evolved_state(&input_hv);
        self.glu_forward(&state)
    }

    /// Train on a set of examples for `epochs` passes, returning the
    /// per-epoch training RMS (loss curve) for diagnostic use. Order is
    /// preserved as given by the caller (galaxy-then-radius ordering
    /// recommended, to mirror nuclear's isotope-chain-order training).
    pub fn train(&mut self, examples: &[ResidualExample], epochs: usize) -> Vec<f64> {
        let lr = 0.005;
        let mut epoch_rms = Vec::with_capacity(epochs);

        for _epoch in 0..epochs {
            let mut total_sq_error = 0.0;
            for ex in examples {
                let input_hv = self.encoder.encode(&ex.state);
                let mut neuron_copy = self.neuron.clone();
                for &dt in &EVOLVE_DTS {
                    neuron_copy.evolve_closed_form(dt, &input_hv);
                }
                let state = neuron_copy.state();

                let mut predicted = 0.0;
                let mut head_out = vec![0.0; N_HEADS];
                let mut head_gates = vec![0.0; N_HEADS];
                let mut head_value_dots = vec![0.0; N_HEADS];

                for h in 0..N_HEADS {
                    let gate_dot: f64 = state
                        .values
                        .iter()
                        .zip(self.w_gates[h].iter())
                        .map(|(&s, &g)| s as f64 * g as f64)
                        .sum();
                    let value_dot: f64 = state
                        .values
                        .iter()
                        .zip(self.w_values[h].iter())
                        .map(|(&s, &v)| s as f64 * v as f64)
                        .sum();
                    let gate = 1.0 / (1.0 + (-gate_dot.clamp(-20.0, 20.0)).exp());
                    let value = value_dot * self.scales[h] + self.biases[h];
                    let out = gate * value;

                    head_out[h] = out;
                    head_gates[h] = gate;
                    head_value_dots[h] = value_dot;
                    predicted += self.head_weights[h] * out;
                }

                let error = predicted - ex.target;
                total_sq_error += error * error;

                for h in 0..N_HEADS {
                    let gate = head_gates[h];
                    let value_dot = head_value_dots[h];
                    let w = self.head_weights[h];

                    let grad_value = lr * error * w * gate * self.scales[h];
                    let grad_gate =
                        lr * error * w * value_dot * self.scales[h] * gate * (1.0 - gate);

                    for i in 0..HDC_DIMENSION {
                        let s = state.values[i] as f64;
                        let decay_value = lr * WEIGHT_DECAY * self.w_values[h][i] as f64;
                        let decay_gate = lr * WEIGHT_DECAY * self.w_gates[h][i] as f64;
                        self.w_values[h][i] -= (grad_value * s + decay_value) as f32;
                        self.w_gates[h][i] -= (grad_gate * s + decay_gate) as f32;
                    }

                    self.scales[h] -= lr * error * w * gate * value_dot;
                    self.biases[h] -= lr * error * w * gate;
                    self.scales[h] = self.scales[h].clamp(-1000.0, 1000.0);
                    self.biases[h] = self.biases[h].clamp(-500.0, 500.0);

                    self.head_weights[h] -= lr * 0.1 * error * head_out[h];
                }

                let w_sum: f64 = self.head_weights.iter().map(|w| w.abs()).sum();
                if w_sum > 0.01 {
                    for w in &mut self.head_weights {
                        *w /= w_sum;
                    }
                }
            }
            let rms = (total_sq_error / examples.len().max(1) as f64).sqrt();
            epoch_rms.push(rms);
        }
        epoch_rms
    }

    /// Deterministic FNV-1a hash of a galaxy name → bucket for train/test
    /// splitting (mirrors `symthaea-muse`'s `fnv` helper).
    pub fn fnv1a(name: &str) -> u32 {
        let mut hash: u32 = 0x811c_9dc5;
        for b in name.as_bytes() {
            hash ^= *b as u32;
            hash = hash.wrapping_mul(0x0100_0193);
        }
        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_state(r: f64) -> GalaxyPointState {
        GalaxyPointState {
            r_kpc: r,
            v_gas: 20.0,
            v_disk: 60.0,
            v_bul: 0.0,
            sb_disk: 300.0,
            sb_bul: 0.0,
            luminosity_3p6: 5.0,
            distance_mpc: 10.0,
            inclination_deg: 60.0,
            gas_fraction: 0.2,
        }
    }

    #[test]
    fn untrained_predictor_returns_finite_output() {
        let reg = HdcResidualRegressor::new(0);
        let out = reg.predict(&dummy_state(3.0));
        assert!(out.is_finite());
    }

    #[test]
    fn training_reduces_error_on_constant_target() {
        let examples: Vec<ResidualExample> = (1..=10)
            .map(|i| ResidualExample {
                state: dummy_state(i as f64),
                target: 0.5,
            })
            .collect();
        let mut reg = HdcResidualRegressor::new(1);
        let before: f64 = examples
            .iter()
            .map(|e| (reg.predict(&e.state) - e.target).powi(2))
            .sum();
        reg.train(&examples, 15);
        let after: f64 = examples
            .iter()
            .map(|e| (reg.predict(&e.state) - e.target).powi(2))
            .sum();
        assert!(
            after < before,
            "training should reduce squared error: before={before} after={after}"
        );
    }

    #[test]
    fn fnv1a_is_deterministic_and_spreads_names() {
        assert_eq!(
            HdcResidualRegressor::fnv1a("NGC3198"),
            HdcResidualRegressor::fnv1a("NGC3198")
        );
        assert_ne!(
            HdcResidualRegressor::fnv1a("NGC3198"),
            HdcResidualRegressor::fnv1a("DDO154")
        );
    }

    /// Diagnostic probe (not a strict pass/fail gate): trains on the real
    /// Newtonian-baryonic residuals for many more epochs than the benchmark
    /// binary uses by default, printing the per-epoch RMS loss curve AND
    /// the held-out R^2 with a proper galaxy-level train/test split. This
    /// exists to answer two questions: (1) is training converging to a
    /// floor (real capacity/signal limit) or barely moving at all (a
    /// learning-rate/gradient-scale bug)? (2) does WEIGHT_DECAY (added
    /// after the first run of this probe showed clean training convergence
    /// alongside negative held-out R^2 -- the overfitting fingerprint,
    /// given 32,768 readout weights against ~130 independent training
    /// galaxies) actually close the train/test gap?
    #[test]
    #[ignore = "requires SPARC data: run scripts/download_sparc.sh"]
    fn loss_curve_probe_newtonian_20_epochs() {
        use crate::fit::r_squared;
        use crate::gravity_models::{Newtonian, RotationModel};
        use crate::sparc::load_sparc;
        use crate::validation::{galaxy_residual_examples, train_test_split};

        let galaxies =
            load_sparc(&crate::test_support::sparc_data_dir()).expect("load_sparc failed");
        let cut: Vec<_> = galaxies
            .into_iter()
            .filter(|g| g.quality <= 2 && g.inclination_deg >= 30.0)
            .collect();
        let cut_refs: Vec<_> = cut.iter().collect();
        let (train_galaxies, test_galaxies) = train_test_split(&cut_refs);

        let train_examples: Vec<ResidualExample> = train_galaxies
            .iter()
            .flat_map(|g| galaxy_residual_examples(g, &Newtonian.fit(g)))
            .collect();
        let test_examples: Vec<ResidualExample> = test_galaxies
            .iter()
            .flat_map(|g| galaxy_residual_examples(g, &Newtonian.fit(g)))
            .collect();
        assert!(train_examples.len() > 1000, "expected thousands of points");

        let mut reg = HdcResidualRegressor::new(0);
        let rms_curve = reg.train(&train_examples, 20);
        eprintln!("Newtonian training RMS by epoch: {rms_curve:?}");
        eprintln!(
            "first={:.4}, last={:.4}, ratio={:.4}",
            rms_curve[0],
            rms_curve.last().unwrap(),
            rms_curve.last().unwrap() / rms_curve[0]
        );

        let observed: Vec<f64> = test_examples.iter().map(|e| e.target).collect();
        let predicted: Vec<f64> = test_examples
            .iter()
            .map(|e| reg.predict(&e.state))
            .collect();
        let held_out_r2 = r_squared(&observed, &predicted);
        eprintln!(
            "held-out R^2 (WEIGHT_DECAY={WEIGHT_DECAY}): {held_out_r2:.4} (n_test={})",
            test_examples.len()
        );
    }
}
