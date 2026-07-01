// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Crystal Reservoir: Bio-mimetic Spectro-Temporal Feature Extraction
//!
//! Replaces random projections with structured 2D Gabor filters modeled after
//! the mammalian auditory cortex's Spectro-Temporal Receptive Fields (STRFs).
//!
//! The key insight: we don't learn "what is a fricative" - the geometry
//! already encodes it. We only learn "this sound matches the fricative pattern."

use std::f32::consts::PI;

/// A 2D Gabor filter for spectro-temporal pattern detection
///
/// Models the receptive field of an auditory cortex neuron:
/// - Frequency selectivity (which mel bins respond)
/// - Temporal modulation (onset detection, steady-state, transitions)
#[derive(Clone, Debug)]
pub struct GaborFilter {
    /// Center frequency in mel bins (0 to n_mels-1)
    pub center_freq: f32,
    /// Spectral bandwidth (sigma in frequency dimension)
    pub spectral_sigma: f32,
    /// Temporal modulation rate (cycles per context window)
    pub temporal_rate: f32,
    /// Temporal envelope width (sigma in time dimension)
    pub temporal_sigma: f32,
    /// Phase offset (0 = cosine, PI/2 = sine)
    pub phase: f32,
}

impl GaborFilter {
    /// Generate the 2D weight patch for this filter
    ///
    /// The Gabor function: G(t, f) = envelope(t, f) * carrier(t, f)
    /// - envelope: 2D Gaussian centered at (t_center, f_center)
    /// - carrier: sinusoid in time dimension (captures temporal modulation)
    pub fn generate_patch(&self, time_steps: usize, freq_bins: usize) -> Vec<f32> {
        let mut patch = Vec::with_capacity(time_steps * freq_bins);

        let t_center = time_steps as f32 / 2.0;
        let f_center = self.center_freq;

        for t in 0..time_steps {
            for f in 0..freq_bins {
                let t_norm = (t as f32 - t_center) / self.temporal_sigma;
                let f_norm = (f as f32 - f_center) / self.spectral_sigma;

                // 2D Gaussian envelope
                let envelope = (-0.5 * (t_norm * t_norm + f_norm * f_norm)).exp();

                // Sinusoidal carrier (temporal modulation)
                let carrier = (2.0 * PI * self.temporal_rate * t_norm + self.phase).cos();

                patch.push(envelope * carrier);
            }
        }

        // Normalize to unit energy
        let energy: f32 = patch.iter().map(|x| x * x).sum();
        if energy > 1e-6 {
            let norm = 1.0 / energy.sqrt();
            for x in &mut patch {
                *x *= norm;
            }
        }

        patch
    }
}

/// The Crystal Reservoir: structured feature extraction via Gabor STRFs
///
/// Unlike random projections which "spray and pray", this uses a mathematically
/// generated basis set that tiles the spectro-temporal space systematically.
#[derive(Clone)]
pub struct CrystalReservoir {
    /// Filter weights: [n_filters × (context_frames × feature_dim)]
    pub filters: Vec<Vec<f32>>,
    /// Number of output features
    pub n_features: usize,
    /// Input context frames
    pub context_frames: usize,
    /// Input feature dimension per frame
    pub feature_dim: usize,
    /// Activation function
    pub activation: CrystalActivation,
}

#[derive(Clone, Copy, Debug)]
pub enum CrystalActivation {
    ReLU,
    HalfWaveRect, // max(0, x) - biologically plausible
    Sigmoid,
}

impl CrystalReservoir {
    /// Create a new Crystal Reservoir with Gabor STRF initialization
    ///
    /// # Arguments
    /// * `n_filters` - Number of output features (e.g., 512)
    /// * `context_frames` - Temporal context (e.g., 7 or 11)
    /// * `feature_dim` - Features per frame (e.g., 120 for mel+delta+deltadelta)
    /// * `n_mels` - Number of mel bins (e.g., 40)
    pub fn new(n_filters: usize, context_frames: usize, feature_dim: usize, n_mels: usize) -> Self {
        let filters = generate_gabor_bank(n_filters, context_frames, feature_dim, n_mels);

        Self {
            filters,
            n_features: n_filters,
            context_frames,
            feature_dim,
            activation: CrystalActivation::ReLU,
        }
    }

    /// Project input through the crystal filters
    ///
    /// # Arguments
    /// * `input` - Stacked context features [context_frames × feature_dim]
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = Vec::with_capacity(self.n_features);

        for filter in &self.filters {
            // Dot product with filter
            let mut activation: f32 = 0.0;
            for (i, &w) in filter.iter().enumerate() {
                if i < input.len() {
                    activation += w * input[i];
                }
            }

            // Apply activation
            let y = match self.activation {
                CrystalActivation::ReLU => activation.max(0.0),
                CrystalActivation::HalfWaveRect => activation.max(0.0),
                CrystalActivation::Sigmoid => 1.0 / (1.0 + (-activation).exp()),
            };

            output.push(y);
        }

        output
    }
}

/// Generate a bank of Gabor filters that tile the spectro-temporal space
///
/// The filters are organized to capture:
/// 1. **Static spectral** (low temporal rate): Formants, steady vowels
/// 2. **Fast transients** (high temporal rate): Plosives, onsets
/// 3. **Rising/falling** (medium rate, different phases): Diphthongs, transitions
fn generate_gabor_bank(
    n_filters: usize,
    context_frames: usize,
    feature_dim: usize,
    n_mels: usize,
) -> Vec<Vec<f32>> {
    let mut filters = Vec::with_capacity(n_filters);

    // Define the parameter ranges for systematic tiling

    // Center frequencies: tile across the mel spectrum
    // More filters in the speech-critical F1/F2 range (bins 5-25)
    let freq_centers: Vec<f32> = (0..n_mels).map(|i| i as f32).collect();

    // Spectral bandwidths: narrow (2 bins) to wide (8 bins)
    let spectral_sigmas = [2.0, 4.0, 6.0, 8.0];

    // Temporal rates: static (0) to fast transient (2.0 cycles/window)
    // 0.0 = DC (steady state), 0.5 = slow modulation, 1.0+ = fast
    let temporal_rates = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0];

    // Temporal widths: narrow (1 frame) to broad (full context)
    let temporal_sigmas = [1.0, 2.0, 3.0, (context_frames as f32) / 2.0];

    // Phases: cosine (0) and sine (PI/2) for quadrature pair
    let phases = [0.0, PI / 2.0];

    // Generate filters by systematic tiling
    let mut filter_idx = 0;

    'outer: for &freq in &freq_centers {
        for &spec_sigma in &spectral_sigmas {
            for &temp_rate in &temporal_rates {
                for &temp_sigma in &temporal_sigmas {
                    for &phase in &phases {
                        if filter_idx >= n_filters {
                            break 'outer;
                        }

                        let gabor = GaborFilter {
                            center_freq: freq,
                            spectral_sigma: spec_sigma,
                            temporal_rate: temp_rate,
                            temporal_sigma: temp_sigma,
                            phase,
                        };

                        // Generate the 2D patch for mel bins only
                        let mel_patch = gabor.generate_patch(context_frames, n_mels);

                        // Expand to full feature_dim (mel + delta + deltadelta)
                        // Apply same filter to each feature stream
                        let full_filter = expand_to_full_features(
                            &mel_patch,
                            context_frames,
                            n_mels,
                            feature_dim,
                        );

                        filters.push(full_filter);
                        filter_idx += 1;
                    }
                }
            }
        }
    }

    // If we need more filters, add with slight variations
    while filters.len() < n_filters {
        // Add random perturbations of existing filters for diversity
        let base_idx = filters.len() % filters.len().max(1);
        let mut new_filter = filters
            .get(base_idx)
            .cloned()
            .unwrap_or_else(|| vec![0.0; context_frames * feature_dim]);

        // Small random perturbation for diversity
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        filters.len().hash(&mut hasher);
        let seed = hasher.finish();

        for (i, x) in new_filter.iter_mut().enumerate() {
            let noise = ((seed.wrapping_add(i as u64) % 1000) as f32 / 1000.0 - 0.5) * 0.1;
            *x += noise;
        }

        filters.push(new_filter);
    }

    filters
}

/// Expand a mel-only filter to the full feature dimension
///
/// The input is a `[context_frames × n_mels]` patch.
/// We replicate it for delta and delta-delta channels with appropriate signs.
fn expand_to_full_features(
    mel_patch: &[f32],
    context_frames: usize,
    n_mels: usize,
    feature_dim: usize,
) -> Vec<f32> {
    let n_streams = feature_dim / n_mels; // Usually 3 (mel, delta, deltadelta)
    let mut full = vec![0.0; context_frames * feature_dim];

    for t in 0..context_frames {
        for m in 0..n_mels {
            let mel_val = mel_patch[t * n_mels + m];

            // Mel channel: direct copy
            full[t * feature_dim + m] = mel_val;

            if n_streams >= 2 {
                // Delta channel: temporal derivative emphasis
                // Use sign based on temporal position (before/after center)
                let t_sign = if t < context_frames / 2 { -1.0 } else { 1.0 };
                full[t * feature_dim + n_mels + m] = mel_val * t_sign * 0.5;
            }

            if n_streams >= 3 {
                // Delta-delta channel: acceleration/curvature
                let t_center = context_frames as f32 / 2.0;
                let t_dist = (t as f32 - t_center).abs() / t_center;
                full[t * feature_dim + 2 * n_mels + m] = mel_val * t_dist * 0.25;
            }
        }
    }

    full
}

/// Online Phoneme Prototype Classifier
///
/// Implements "Geometric Error Correction" - instead of batch ridge regression,
/// we maintain running prototype vectors for each phoneme class and update
/// them online with each new sample.
#[derive(Clone)]
pub struct OnlinePrototypeClassifier {
    /// Prototype vectors for each phoneme [n_classes × feature_dim]
    pub prototypes: Vec<Vec<f32>>,
    /// Sample counts per class (for weighted averaging)
    pub counts: Vec<usize>,
    /// Class labels
    pub labels: Vec<String>,
    /// Feature dimension
    pub feature_dim: usize,
}

impl OnlinePrototypeClassifier {
    /// Create a new classifier
    pub fn new(labels: Vec<String>, feature_dim: usize) -> Self {
        let n_classes = labels.len();
        Self {
            prototypes: vec![vec![0.0; feature_dim]; n_classes],
            counts: vec![0; n_classes],
            labels,
            feature_dim,
        }
    }

    /// Update prototype with a new observation (Geometric Error Correction)
    ///
    /// This is the "One-Shot" learning: we simply blend the observation
    /// into the running prototype using a decaying weight.
    pub fn update(&mut self, observation: &[f32], class_idx: usize) {
        if class_idx >= self.prototypes.len() || observation.len() != self.feature_dim {
            return;
        }

        self.counts[class_idx] += 1;
        let count = self.counts[class_idx] as f32;

        // Running average: prototype = (prototype * (n-1) + observation) / n
        // Equivalent to: prototype += (observation - prototype) / n
        let weight = 1.0 / count;

        let proto = &mut self.prototypes[class_idx];
        for (i, &obs) in observation.iter().enumerate() {
            proto[i] += weight * (obs - proto[i]);
        }
    }

    /// Classify an observation by finding the nearest prototype
    ///
    /// Returns (class_idx, similarity_score)
    pub fn classify(&self, observation: &[f32]) -> (usize, f32) {
        let mut best_idx = 0;
        let mut best_sim = f32::NEG_INFINITY;

        for (idx, proto) in self.prototypes.iter().enumerate() {
            if self.counts[idx] == 0 {
                continue; // Skip uninitialized prototypes
            }

            let sim = cosine_similarity(observation, proto);
            if sim > best_sim {
                best_sim = sim;
                best_idx = idx;
            }
        }

        (best_idx, best_sim)
    }

    /// Get logits (similarity scores) for all classes
    pub fn get_logits(&self, observation: &[f32]) -> Vec<f32> {
        self.prototypes
            .iter()
            .enumerate()
            .map(|(idx, proto)| {
                if self.counts[idx] == 0 {
                    f32::NEG_INFINITY
                } else {
                    cosine_similarity(observation, proto)
                }
            })
            .collect()
    }
}

/// Cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom > 1e-8 { dot / denom } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gabor_filter_generation() {
        let gabor = GaborFilter {
            center_freq: 20.0,
            spectral_sigma: 4.0,
            temporal_rate: 0.5,
            temporal_sigma: 2.0,
            phase: 0.0,
        };

        let patch = gabor.generate_patch(7, 40);
        assert_eq!(patch.len(), 7 * 40);

        // Check normalization
        let energy: f32 = patch.iter().map(|x| x * x).sum();
        assert!(
            (energy - 1.0).abs() < 0.01,
            "Energy should be ~1.0, got {}",
            energy
        );
    }

    #[test]
    fn test_crystal_reservoir() {
        let crystal = CrystalReservoir::new(256, 7, 120, 40);
        assert_eq!(crystal.filters.len(), 256);
        assert_eq!(crystal.filters[0].len(), 7 * 120);

        // Test forward pass
        let input = vec![0.1; 7 * 120];
        let output = crystal.forward(&input);
        assert_eq!(output.len(), 256);
    }

    #[test]
    fn test_online_prototype_classifier() {
        let labels = vec!["A".into(), "B".into(), "C".into()];
        let mut classifier = OnlinePrototypeClassifier::new(labels, 4);

        // Train with some samples
        classifier.update(&[1.0, 0.0, 0.0, 0.0], 0); // A
        classifier.update(&[0.9, 0.1, 0.0, 0.0], 0); // A
        classifier.update(&[0.0, 1.0, 0.0, 0.0], 1); // B
        classifier.update(&[0.0, 0.0, 1.0, 0.0], 2); // C

        // Classify
        let (class, sim) = classifier.classify(&[0.95, 0.05, 0.0, 0.0]);
        assert_eq!(class, 0, "Should classify as A");
        assert!(sim > 0.9, "Should have high similarity");
    }
}
