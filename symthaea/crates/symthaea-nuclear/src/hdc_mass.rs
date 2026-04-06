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

/// HDC+LTC nuclear mass predictor.
pub struct HdcMassPredictor {
    encoder: NuclearStateEncoder,
    neuron: HdcLtcUnifiedNeuron,
    /// Learned projection vector for decoding state → scalar correction
    decoder: Vec<f32>,
    /// Output scale factor
    scale: f64,
    /// Output bias
    bias: f64,
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

        let mut predictor = Self {
            encoder: NuclearStateEncoder::new(),
            neuron: HdcLtcUnifiedNeuron::new(config, 0xA0C1_DEAD),
            decoder: vec![0.0; HDC_DIMENSION],
            scale: 100.0, // Initial scale for corrections (~100 MeV range)
            bias: 0.0,
        };

        // Initialize decoder with small random values
        let init_hv = ContinuousHV::random(HDC_DIMENSION, 0xDEC0_DE01);
        for i in 0..HDC_DIMENSION {
            predictor.decoder[i] = init_hv.values[i] * 0.01;
        }

        // Train on AME2020
        let nuclei: Vec<_> = ame2020_reference_nuclei()
            .into_iter()
            .filter(|n| n.is_measured && n.z >= 3)
            .collect();
        predictor.train(&nuclei, 30);

        predictor
    }

    /// Encode a nucleus as input HV (Z, N only — no BE in input).
    fn encode_input(&self, z: u16, n: u16) -> ContinuousHV {
        let (beta2, _) = frdm_deformation(z, n);
        self.encoder.encode(&NuclearState {
            z,
            n,
            binding_energy: 0.0,   // Unknown — this is what we predict
            shell_correction: 0.0,
            deformation: beta2,
        })
    }

    /// Forward pass: encode → evolve → decode → DZ + correction.
    fn forward(&self, z: u16, n: u16) -> (f64, f64, f64) {
        let input_hv = self.encode_input(z, n);

        // Evolve the neuron (clone to avoid mutation)
        let mut neuron = self.neuron.clone();
        neuron.evolve_closed_form(1.0, &input_hv);
        let state = neuron.state();

        // Decode: dot product with learned decoder
        let dot: f64 = state
            .values
            .iter()
            .zip(self.decoder.iter())
            .map(|(&s, &d)| s as f64 * d as f64)
            .sum();

        let correction = dot * self.scale + self.bias;
        let dz = dz_binding_energy(z, n);
        let total = dz + correction;

        (total, dz, correction)
    }

    /// Train on a set of measured nuclei.
    pub fn train(&mut self, nuclei: &[MeasuredNucleus], epochs: usize) {
        let lr = 0.001;
        let lr_neuron = 0.0005;
        let n = nuclei.len();

        for epoch in 0..epochs {
            let mut total_sq_error = 0.0;

            // Shuffle order using epoch-dependent permutation
            let mut indices: Vec<usize> = (0..n).collect();
            let mut rng = (epoch as u64).wrapping_mul(6364136223846793005).wrapping_add(1);
            for i in (1..n).rev() {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let j = (rng >> 33) as usize % (i + 1);
                indices.swap(i, j);
            }

            for &idx in &indices {
                let nuc = &nuclei[idx];
                let dz = dz_binding_energy(nuc.z, nuc.n);
                let target_residual = nuc.binding_energy_mev - dz;

                // Forward
                let input_hv = self.encode_input(nuc.z, nuc.n);
                let mut neuron_copy = self.neuron.clone();
                neuron_copy.evolve_closed_form(1.0, &input_hv);
                let state = neuron_copy.state();

                // Decode
                let dot: f64 = state
                    .values
                    .iter()
                    .zip(self.decoder.iter())
                    .map(|(&s, &d)| s as f64 * d as f64)
                    .sum();
                let predicted_residual = dot * self.scale + self.bias;
                let error = predicted_residual - target_residual;
                total_sq_error += error * error;

                // Update decoder (gradient descent)
                let grad_scale = lr * error;
                for i in 0..HDC_DIMENSION {
                    self.decoder[i] -= (grad_scale * state.values[i] as f64 * self.scale) as f32;
                }

                // Update scale and bias
                self.scale -= lr * error * dot;
                self.bias -= lr * error;

                // Clamp to prevent divergence
                self.scale = self.scale.clamp(-1000.0, 1000.0);
                self.bias = self.bias.clamp(-500.0, 500.0);

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

            let mut predictor = HdcMassPredictor {
                encoder: NuclearStateEncoder::new(),
                neuron: HdcLtcUnifiedNeuron::new(config, 0xA0C1_DEAD + fold as u64),
                decoder: vec![0.0; HDC_DIMENSION],
                scale: 100.0,
                bias: 0.0,
            };

            let init_hv = ContinuousHV::random(HDC_DIMENSION, 0xDEC0_DE01 + fold as u64);
            for i in 0..HDC_DIMENSION {
                predictor.decoder[i] = init_hv.values[i] * 0.01;
            }

            predictor.train(&train_nuclei, 20); // Fewer epochs for CV speed

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

        let hdc_rms =
            (hdc_errors.iter().sum::<f64>() / hdc_errors.len() as f64).sqrt();
        let rf_rms =
            (rf_errors.iter().sum::<f64>() / rf_errors.len() as f64).sqrt();

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
