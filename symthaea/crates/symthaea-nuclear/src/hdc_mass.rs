// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC+LTC Nuclear Mass Predictor — Symthaea-native approach.
//!
//! Uses Hyperdimensional Computing (16,384D ContinuousHV) with Liquid
//! Time-Constant neurons for nuclear binding energy prediction.
//!
//! Architecture: DZ baseline + HDC correction (same as RF approach)
//! - Encode (Z, N) → 16,384D input via NuclearStateEncoder
//! - Evolve HdcLtcUnifiedNeuron with CfC closed-form
//! - Decode via learned projection: correction = dot(state, decoder) × scale
//! - Train via contrastive learning (neuron) + gradient descent (decoder)
//!
//! This is genuinely novel: nobody has an HDC-encoded nuclear chart with
//! consciousness-coupled temporal dynamics for mass prediction.

use crate::ame2020::ame2020_reference_nuclei;
use crate::deformation::frdm_deformation;
use crate::discovery::MeasuredNucleus;
use crate::duflo_zuker::dz_binding_energy;
use crate::encoder::{NuclearState, NuclearStateEncoder};
use serde::{Deserialize, Serialize};
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

/// HDC mass prediction result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdcMassPrediction {
    /// Total predicted binding energy (MeV)
    pub binding_energy: f64,
    /// DZ baseline contribution (MeV)
    pub dz_baseline: f64,
    /// HDC correction (MeV)
    pub hdc_correction: f64,
    /// Binding energy per nucleon (MeV)
    pub ba: f64,
}

/// Number of GLU decoder heads (1 = single head, proven stable).
const N_HEADS: usize = 1;

/// Number of CfC evolution steps per prediction.
const N_EVOLVE_STEPS: usize = 3;

/// Evolution time steps (different timescales for multi-scale features).
const EVOLVE_DTS: [f32; N_EVOLVE_STEPS] = [0.01, 0.1, 1.0];

/// HDC+LTC nuclear mass predictor with multi-head GLU decoder.
///
/// Improvements over single-head linear:
/// 1. **Multi-head GLU** (4 gate-value pairs, averaged) — captures different
///    aspects of nuclear structure
/// 2. **Multi-scale CfC evolution** (dt=0.1, 1.0, 10.0) — fast dynamics,
///    equilibrium, and slow modes
/// 3. **Isotope chain sequential training** — feeds isotope chains in order
///    so the LTC neuron accumulates shell structure knowledge
pub struct HdcMassPredictor {
    encoder: NuclearStateEncoder,
    neuron: HdcLtcUnifiedNeuron,
    /// Per-head gate projections (N_HEADS × HDC_DIMENSION)
    w_gates: Vec<Vec<f32>>,
    /// Per-head value projections
    w_values: Vec<Vec<f32>>,
    /// Per-head scale factors
    scales: Vec<f64>,
    /// Per-head biases
    biases: Vec<f64>,
    /// Head blend weights (learned)
    head_weights: Vec<f64>,
}

impl HdcMassPredictor {
    /// Create and train the HDC predictor on AME2020 data.
    pub fn new() -> Self {
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
        let mut scales = vec![10.0; N_HEADS];
        let mut biases = vec![0.0; N_HEADS];

        for h in 0..N_HEADS {
            let gate_init = ContinuousHV::random(HDC_DIMENSION, 0xDA7E_0001 + h as u64 * 7919);
            let value_init = ContinuousHV::random(HDC_DIMENSION, 0xFA1E_0002 + h as u64 * 6263);
            let mut wg = vec![0.0f32; HDC_DIMENSION];
            let mut wv = vec![0.0f32; HDC_DIMENSION];
            for i in 0..HDC_DIMENSION {
                wg[i] = gate_init.values[i] * 0.1;
                wv[i] = value_init.values[i] * 0.1;
            }
            w_gates.push(wg);
            w_values.push(wv);
        }

        let mut predictor = Self {
            encoder: NuclearStateEncoder::new(),
            neuron: HdcLtcUnifiedNeuron::new(config, 0xA0C1_DEAD),
            w_gates,
            w_values,
            scales,
            biases,
            head_weights: vec![1.0 / N_HEADS as f64; N_HEADS],
        };

        // Train on AME2020
        let nuclei: Vec<_> = ame2020_reference_nuclei()
            .into_iter()
            .filter(|n| n.is_measured && n.z >= 3)
            .collect();
        predictor.train(&nuclei, 10); // Few epochs — chain ordering gives strong first pass

        predictor
    }

    /// Encode a nucleus as input HV (Z, N only — no BE in input).
    fn encode_input(&self, z: u16, n: u16) -> ContinuousHV {
        let (beta2, _) = frdm_deformation(z, n);
        self.encoder.encode(&NuclearState {
            z,
            n,
            binding_energy: 0.0, // Unknown — this is what we predict
            shell_correction: 0.0,
            deformation: beta2,
        })
    }

    /// Forward pass: encode → multi-scale evolve → multi-head GLU → DZ + correction.
    fn forward(&self, z: u16, n: u16) -> (f64, f64, f64) {
        let input_hv = self.encode_input(z, n);

        // Multi-scale CfC evolution: evolve at 3 timescales
        // Fix: seed neuron state from input (not zero) so gradients flow
        let mut neuron = self.neuron.clone();
        neuron.set_state(input_hv.clone());
        for &dt in &EVOLVE_DTS {
            neuron.evolve_closed_form(dt, &input_hv);
        }
        let state = neuron.state();

        // Multi-head GLU decode: each head computes its own correction
        let mut total_correction = 0.0;
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
            let head_correction = gate * (value_dot * self.scales[h] + self.biases[h]);
            total_correction += self.head_weights[h] * head_correction;
        }

        let dz = dz_binding_energy(z, n);
        let total = dz + total_correction;

        (total, dz, total_correction)
    }

    /// Train on a set of measured nuclei with isotope chain ordering.
    pub fn train(&mut self, nuclei: &[MeasuredNucleus], epochs: usize) {
        let lr = 0.005; // 10× increase to escape zero basin
        let lr_neuron = 0.002;
        let n = nuclei.len();

        // Sort by (Z, N) for isotope chain sequential training
        // This lets the LTC neuron accumulate knowledge along chains
        let mut sorted: Vec<_> = nuclei.to_vec();
        sorted.sort_by(|a, b| a.z.cmp(&b.z).then(a.n.cmp(&b.n)));

        for epoch in 0..epochs {
            let mut total_sq_error = 0.0;

            // Always use isotope chain order — LTC accumulates shell knowledge
            let indices: Vec<usize> = (0..n).collect();

            for &idx in &indices {
                let nuc = &sorted[idx];
                let dz = dz_binding_energy(nuc.z, nuc.n);
                let target_residual = nuc.binding_energy_mev - dz;

                // Forward: multi-scale evolution
                let input_hv = self.encode_input(nuc.z, nuc.n);
                let mut neuron_copy = self.neuron.clone();
                for &dt in &EVOLVE_DTS {
                    neuron_copy.evolve_closed_form(dt, &input_hv);
                }
                let state = neuron_copy.state();

                // Multi-head GLU decode
                let mut predicted_residual = 0.0;
                let mut head_corrections = vec![0.0; N_HEADS];
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
                    let corr = gate * value;

                    head_corrections[h] = corr;
                    head_gates[h] = gate;
                    head_value_dots[h] = value_dot;
                    predicted_residual += self.head_weights[h] * corr;
                }

                let error = predicted_residual - target_residual;
                total_sq_error += error * error;

                // Update each head
                for h in 0..N_HEADS {
                    let gate = head_gates[h];
                    let value_dot = head_value_dots[h];
                    let value = value_dot * self.scales[h] + self.biases[h];
                    let w = self.head_weights[h];

                    let grad_value = lr * error * w * gate * self.scales[h];
                    let grad_gate = lr * error * w * value * gate * (1.0 - gate);

                    for i in 0..HDC_DIMENSION {
                        let s = state.values[i] as f64;
                        self.w_values[h][i] -= (grad_value * s) as f32;
                        self.w_gates[h][i] -= (grad_gate * s) as f32;
                    }

                    self.scales[h] -= lr * error * w * gate * value_dot;
                    self.biases[h] -= lr * error * w * gate;
                    self.scales[h] = self.scales[h].clamp(-1000.0, 1000.0);
                    self.biases[h] = self.biases[h].clamp(-500.0, 500.0);

                    // Update head weight toward better-performing heads
                    self.head_weights[h] -= lr * 0.1 * error * head_corrections[h];
                }

                // Normalize head weights to sum to 1
                let w_sum: f64 = self.head_weights.iter().map(|w| w.abs()).sum();
                if w_sum > 0.01 {
                    for w in &mut self.head_weights {
                        *w /= w_sum;
                    }
                }

                // (clamping already done per-head above)

                // Update neuron via contrastive learning
                // Target: state that encodes the measured BE
                let (beta2, _) = frdm_deformation(nuc.z, nuc.n);
                let target_hv = self.encoder.encode(&NuclearState {
                    z: nuc.z,
                    n: nuc.n,
                    binding_energy: nuc.binding_energy_mev,
                    shell_correction: 0.0,
                    deformation: beta2,
                });
                self.neuron
                    .contrastive_update(&target_hv, &input_hv, lr_neuron as f32);
            }

            let rms = (total_sq_error / n as f64).sqrt();
            if epoch == 0 || epoch == epochs - 1 || (epoch + 1) % 10 == 0 {
                eprintln!("  HDC epoch {}: RMS = {:.2} MeV", epoch + 1, rms);
            }
        }
    }

    /// Predict binding energy for (Z, N).
    pub fn predict(&self, z: u16, n: u16) -> HdcMassPrediction {
        let (total, dz, correction) = self.forward(z, n);
        let a = (z + n) as f64;
        HdcMassPrediction {
            binding_energy: total,
            dz_baseline: dz,
            hdc_correction: correction,
            ba: if a > 0.0 { total / a } else { 0.0 },
        }
    }

    /// Create an untrained predictor, train on the given nuclei, and return it.
    ///
    /// Use this for validation splits where you control train/test partitioning.
    pub fn train_on(nuclei: &[MeasuredNucleus], epochs: usize, seed_offset: u64) -> Self {
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
                0xDA7E_0001 + seed_offset * 100 + h as u64 * 7919,
            );
            let vi = ContinuousHV::random(
                HDC_DIMENSION,
                0xFA1E_0002 + seed_offset * 100 + h as u64 * 6263,
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

        let mut predictor = Self {
            encoder: NuclearStateEncoder::new(),
            neuron: HdcLtcUnifiedNeuron::new(config, 0xA0C1_DEAD + seed_offset),
            w_gates,
            w_values,
            scales: vec![100.0; N_HEADS],
            biases: vec![0.0; N_HEADS],
            head_weights: vec![1.0 / N_HEADS as f64; N_HEADS],
        };

        predictor.train(nuclei, epochs);
        predictor
    }

    /// 5-fold cross-validation RMS.
    pub fn cross_validate() -> f64 {
        let nuclei: Vec<_> = ame2020_reference_nuclei()
            .into_iter()
            .filter(|n| n.is_measured && n.z >= 3)
            .collect();
        let n = nuclei.len();
        let fold_size = n / 5;
        let mut total_sq_error = 0.0;
        let mut total_count = 0;

        for fold in 0..5 {
            let test_start = fold * fold_size;
            let test_end = if fold == 4 { n } else { (fold + 1) * fold_size };

            // Train on everything except this fold
            let train_nuclei: Vec<_> = nuclei
                .iter()
                .enumerate()
                .filter(|(i, _)| *i < test_start || *i >= test_end)
                .map(|(_, n)| n.clone())
                .collect();

            let config = UnifiedConfig {
                tau_base: 1.0,
                backbone_tau: 2.0,
                dimension: HDC_DIMENSION,
                learning_rate: 0.005,
                ..UnifiedConfig::default()
            };

            let mut w_gates = Vec::with_capacity(N_HEADS);
            let mut w_values = Vec::with_capacity(N_HEADS);
            for h in 0..N_HEADS {
                let gi = ContinuousHV::random(
                    HDC_DIMENSION,
                    0xDA7E_0001 + fold as u64 * 100 + h as u64 * 7919,
                );
                let vi = ContinuousHV::random(
                    HDC_DIMENSION,
                    0xFA1E_0002 + fold as u64 * 100 + h as u64 * 6263,
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
            let mut predictor = HdcMassPredictor {
                encoder: NuclearStateEncoder::new(),
                neuron: HdcLtcUnifiedNeuron::new(config, 0xA0C1_DEAD + fold as u64),
                w_gates,
                w_values,
                scales: vec![100.0; N_HEADS],
                biases: vec![0.0; N_HEADS],
                head_weights: vec![1.0 / N_HEADS as f64; N_HEADS],
            };

            predictor.train(&train_nuclei, 8); // Few epochs for CV speed

            // Test on fold
            for i in test_start..test_end {
                let nuc = &nuclei[i];
                let (predicted, _, _) = predictor.forward(nuc.z, nuc.n);
                let error = predicted - nuc.binding_energy_mev;
                total_sq_error += error * error;
                total_count += 1;
            }
        }

        (total_sq_error / total_count as f64).sqrt()
    }
}

impl Default for HdcMassPredictor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdc_predictor_trains() {
        let predictor = HdcMassPredictor::new();
        // Should be able to predict Fe-56
        let pred = predictor.predict(26, 30);
        assert!(
            pred.binding_energy.is_finite(),
            "Fe-56 HDC prediction should be finite: {}",
            pred.binding_energy
        );
        assert!(
            pred.binding_energy > 100.0,
            "Fe-56 BE = {} should be > 100 MeV",
            pred.binding_energy
        );
    }

    #[test]
    fn test_hdc_prediction_structure() {
        let predictor = HdcMassPredictor::new();
        let pred = predictor.predict(82, 126); // Pb-208
        assert!(pred.dz_baseline > 0.0);
        assert!(pred.hdc_correction.is_finite());
        assert!(pred.ba > 0.0);
    }

    #[test]
    fn test_hdc_vs_rf_comparison() {
        let hdc = HdcMassPredictor::new();
        let rf = crate::ml_mass::MlMassPredictor::new();
        let nuclei = ame2020_reference_nuclei();

        let mut hdc_errors = Vec::new();
        let mut rf_errors = Vec::new();

        for nuc in &nuclei {
            if !nuc.is_measured || nuc.z < 3 {
                continue;
            }
            let hdc_pred = hdc.predict(nuc.z, nuc.n);
            let rf_pred = rf.predict(nuc.z, nuc.n);

            hdc_errors.push((hdc_pred.binding_energy - nuc.binding_energy_mev).powi(2));
            rf_errors.push((rf_pred.binding_energy - nuc.binding_energy_mev).powi(2));
        }

        let hdc_rms = (hdc_errors.iter().sum::<f64>() / hdc_errors.len() as f64).sqrt();
        let rf_rms = (rf_errors.iter().sum::<f64>() / rf_errors.len() as f64).sqrt();

        eprintln!(
            "HDC RMS: {:.2} MeV, RF RMS: {:.2} MeV (ratio: {:.2})",
            hdc_rms,
            rf_rms,
            hdc_rms / rf_rms
        );

        // Both should be finite and reasonable
        assert!(hdc_rms.is_finite() && hdc_rms > 0.0);
        assert!(rf_rms.is_finite() && rf_rms > 0.0);
    }

    #[test]
    fn test_hdc_cross_validation() {
        let cv = HdcMassPredictor::cross_validate();
        eprintln!("HDC 5-fold CV RMS: {:.2} MeV", cv);
        assert!(
            cv.is_finite() && cv > 0.0 && cv < 500.0,
            "HDC CV = {} should be reasonable",
            cv
        );
    }
}
