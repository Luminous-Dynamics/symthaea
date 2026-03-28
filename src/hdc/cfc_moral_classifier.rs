// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CfC-based moral classifier using HdcLtcUnifiedNetwork.
//!
//! Uses the unified HDC-LTC network for non-linear moral classification.
//! Text is encoded via `TextHdcEncoder`, projected through `HarmonyBasis`
//! into an 8D moral subspace, re-expanded to HDC space, evolved through
//! a 2-layer CfC network, and classified against learned prototypes.
//!
//! # Architecture
//!
//! ```text
//! Text → TextHdcEncoder → HarmonyBasis (project to 8D) → re-expand to HDC
//!      → HdcLtcUnifiedNetwork (2 layers × 3 neurons) → output HV
//!      → cosine similarity to class prototypes → verdict
//! ```

use std::sync::Arc;

use symthaea_core::hdc::hdc_ltc_unified::{
    HdcLtcUnifiedNetwork, UnifiedActivation, UnifiedConfig, UnifiedNetworkConfig,
};
use symthaea_core::hdc::unified_hv::ContinuousHV;

use super::harmony_basis::HarmonyBasis;
use super::moral_algebra::MoralVerdict;
use super::moral_prototypes::MoralLabel;
use super::moral_text_encoder::TextHdcEncoder;

/// Configuration for the CfC moral classifier.
#[derive(Debug, Clone)]
pub struct CfcMoralConfig {
    /// HDC dimension.
    pub dim: usize,
    /// Number of CfC evolution steps per forward pass.
    pub n_evolve_steps: usize,
    /// Time step for CfC evolution.
    pub dt: f32,
    /// Learning rate for contrastive updates.
    pub learning_rate: f32,
    /// Number of retraining epochs over the training set.
    pub n_retrain_epochs: usize,
}

/// Dimension for CfC neurons (small for speed — CfC operates in projected space).
const CFC_NEURON_DIM: usize = 256;

impl Default for CfcMoralConfig {
    fn default() -> Self {
        Self {
            dim: 4096,
            n_evolve_steps: 5,
            dt: 0.1,
            learning_rate: 0.001,
            n_retrain_epochs: 10,
        }
    }
}

/// CfC-based moral classifier.
///
/// Combines text encoding, harmony projection, and a 2-layer CfC network
/// for non-linear moral classification into Good / Neutral / Bad.
pub struct CfcMoralClassifier {
    encoder: TextHdcEncoder,
    basis: Arc<HarmonyBasis>,
    network: HdcLtcUnifiedNetwork,
    /// Learned class prototypes: [Good, Bad, Neutral]
    class_prototypes: Option<[ContinuousHV; 3]>,
    config: CfcMoralConfig,
}

impl CfcMoralClassifier {
    /// Create a new CfC moral classifier.
    ///
    /// Sets up a TextHdcEncoder with sentiment, and a 2-layer (3, 3)
    /// HdcLtcUnifiedNetwork with layer binding and skip connections.
    pub fn new(basis: Arc<HarmonyBasis>, dim: usize) -> Self {
        let encoder = TextHdcEncoder::with_sentiment(dim, 3, 0.5, 0.2);

        let neuron_config = UnifiedConfig {
            tau_base: 0.05,
            backbone_tau: 0.3,
            dimension: CFC_NEURON_DIM, // Small dimension for fast CfC evolution
            activation: UnifiedActivation::Tanh,
            learning_rate: 0.001,
            ..UnifiedConfig::default()
        };

        let net_config = UnifiedNetworkConfig {
            layer_sizes: vec![3, 3],
            neuron_config,
            use_layer_binding: true,
            skip_connections: true,
        };

        let network = HdcLtcUnifiedNetwork::new(net_config, 90000001);

        Self {
            encoder,
            basis,
            network,
            class_prototypes: None,
            config: CfcMoralConfig {
                dim,
                ..CfcMoralConfig::default()
            },
        }
    }

    /// Encode text into a morally-structured HDC vector.
    ///
    /// 1. Encode text via TextHdcEncoder
    /// 2. Project through HarmonyBasis to 8D coordinates
    /// 3. Build 44 features: 8 linear + 28 cross-terms + 8 squared
    /// 4. Project 44 features to CFC_NEURON_DIM via deterministic random matrix
    pub fn encode_input(&self, text: &str) -> ContinuousHV {
        let text_hv = self.encoder.encode(text);
        let coords = self.basis.project(&text_hv);

        // Build feature vector: 8 linear + 28 cross-terms + 8 squared = 44 features
        let mut features = Vec::with_capacity(44);
        // Linear terms
        for &c in &coords {
            features.push(c as f32);
        }
        // Cross-terms (i < j)
        for i in 0..8 {
            for j in (i + 1)..8 {
                features.push((coords[i] * coords[j]) as f32);
            }
        }
        // Squared terms
        for &c in &coords {
            features.push((c * c) as f32);
        }

        // Project 44 features to CFC_NEURON_DIM via deterministic random matrix
        let mut result = vec![0.0f32; CFC_NEURON_DIM];
        for (f_idx, &feat) in features.iter().enumerate() {
            let proj = ContinuousHV::random(CFC_NEURON_DIM, 95000000 + f_idx as u64);
            for j in 0..CFC_NEURON_DIM {
                result[j] += feat * proj.values[j];
            }
        }

        // L2-normalize
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut result {
                *v /= norm;
            }
        }

        ContinuousHV::from_vec(result)
    }

    /// Run the CfC network forward on an input HV.
    ///
    /// Resets the network, evolves for `n_evolve_steps`, and returns the output.
    pub fn forward(&mut self, input: &ContinuousHV) -> ContinuousHV {
        self.network.reset();
        for _ in 0..self.config.n_evolve_steps {
            self.network.evolve_closed_form(self.config.dt, input);
        }
        self.network.output()
    }

    /// Train the classifier on labeled samples.
    ///
    /// **Phase A**: Accumulate forward outputs per class to build initial prototypes.
    /// **Phase B**: For each epoch, iterate samples, contrastive-update misclassified
    /// neurons, and rebuild prototypes.
    pub fn train(&mut self, samples: &[(String, MoralLabel)]) {
        if samples.is_empty() {
            return;
        }

        // Cache encoded inputs — encoding is expensive (TextHdcEncoder at full dim)
        // and doesn't change between epochs.
        let encoded: Vec<(ContinuousHV, MoralLabel)> = samples
            .iter()
            .map(|(text, label)| (self.encode_input(text), *label))
            .collect();

        // Phase A: Build initial prototypes from forward passes
        let prototypes = self.build_prototypes_cached(&encoded);
        self.class_prototypes = Some(prototypes);

        // Phase B: Contrastive retraining
        for _epoch in 0..self.config.n_retrain_epochs {
            for (input, label) in &encoded {
                let output = self.forward(input);

                let protos = self.class_prototypes.as_ref().unwrap();
                let correct_idx = label_to_index(*label);

                // Find predicted class (argmax similarity)
                let sims: Vec<f32> = protos.iter().map(|p| output.similarity(p)).collect();
                let predicted_idx = sims
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                // If misclassified, apply contrastive update to ALL output neurons
                // and hidden layer for stronger gradient signal
                if predicted_idx != correct_idx {
                    let correct_proto = protos[correct_idx].clone();
                    let wrong_proto = protos[predicted_idx].clone();
                    let lr = self.config.learning_rate;

                    if let Some(layer) = self.network.layer_mut(1) {
                        let layer_len = layer.len();
                        for (n_idx, neuron) in layer.iter_mut().enumerate() {
                            if n_idx == correct_idx % layer_len {
                                neuron.contrastive_update(&correct_proto, &wrong_proto, lr);
                            } else if n_idx == predicted_idx % layer_len {
                                neuron.contrastive_update(&wrong_proto, &correct_proto, lr * 0.5);
                            }
                        }
                    }

                    // Also update hidden layer with a smaller learning rate
                    if let Some(hidden) = self.network.layer_mut(0) {
                        for neuron in hidden.iter_mut() {
                            neuron.contrastive_update(&correct_proto, &wrong_proto, lr * 0.1);
                        }
                    }
                }
            }

            // Rebuild prototypes after each epoch (using cached inputs)
            let protos = self.build_prototypes_cached(&encoded);
            self.class_prototypes = Some(protos);
        }
    }

    /// Classify a text string into a moral verdict with confidence.
    ///
    /// Returns `(MoralVerdict, confidence)` where confidence is the margin
    /// between the best and second-best class similarity, clamped to [0, 1].
    pub fn classify(&mut self, text: &str) -> (MoralVerdict, f32) {
        let input = self.encode_input(text);
        let output = self.forward(&input);

        let protos = match &self.class_prototypes {
            Some(p) => p,
            None => return (MoralVerdict::Neutral, 0.0),
        };

        let sims: Vec<f32> = protos.iter().map(|p| output.similarity(p)).collect();

        // Find best and second-best
        let mut indices: Vec<usize> = (0..3).collect();
        indices.sort_by(|&a, &b| {
            sims[b]
                .partial_cmp(&sims[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let best_idx = indices[0];
        let margin = (sims[indices[0]] - sims[indices[1]]).max(0.0).min(1.0);

        let verdict = match best_idx {
            0 => MoralVerdict::Good,
            1 => MoralVerdict::Bad,
            _ => MoralVerdict::Neutral,
        };

        (verdict, margin)
    }

    /// Whether the classifier has been trained (prototypes exist).
    pub fn is_trained(&self) -> bool {
        self.class_prototypes.is_some()
    }

    /// Build class prototypes from cached encoded inputs.
    fn build_prototypes_cached(
        &mut self,
        encoded: &[(ContinuousHV, MoralLabel)],
    ) -> [ContinuousHV; 3] {
        let neuron_dim = CFC_NEURON_DIM;
        let mut accumulators = [
            vec![0.0f32; neuron_dim],
            vec![0.0f32; neuron_dim],
            vec![0.0f32; neuron_dim],
        ];
        let mut counts = [0usize; 3];

        for (input, label) in encoded {
            let output = self.forward(input);
            let idx = label_to_index(*label);

            for (acc, &val) in accumulators[idx].iter_mut().zip(output.values.iter()) {
                *acc += val;
            }
            counts[idx] += 1;
        }

        // Normalize each prototype
        let mut protos: [ContinuousHV; 3] = [
            ContinuousHV::zero(neuron_dim),
            ContinuousHV::zero(neuron_dim),
            ContinuousHV::zero(neuron_dim),
        ];

        for (i, acc) in accumulators.iter().enumerate() {
            if counts[i] > 0 {
                let norm: f32 = acc.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    let normalized: Vec<f32> = acc.iter().map(|x| x / norm).collect();
                    protos[i] = ContinuousHV::from_vec(normalized);
                }
            }
        }

        protos
    }
}

/// Map a MoralLabel to a prototype index: Good=0, Bad=1, Neutral=2.
fn label_to_index(label: MoralLabel) -> usize {
    match label {
        MoralLabel::Good => 0,
        MoralLabel::Bad => 1,
        MoralLabel::Neutral => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_basis(dim: usize) -> Arc<HarmonyBasis> {
        Arc::new(HarmonyBasis::new(dim))
    }

    #[test]
    fn test_untrained_returns_neutral() {
        let basis = make_basis(4096);
        let mut clf = CfcMoralClassifier::new(basis, 4096);
        assert!(!clf.is_trained());

        let (verdict, confidence) = clf.classify("stealing is wrong");
        assert!(matches!(verdict, MoralVerdict::Neutral));
        assert!((confidence - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_train_classify_roundtrip() {
        let basis = make_basis(4096);
        let mut clf = CfcMoralClassifier::new(basis, 4096);

        let samples = vec![
            (
                "helping others is kind and generous".to_string(),
                MoralLabel::Good,
            ),
            ("caring for the sick is noble".to_string(), MoralLabel::Good),
            ("sharing food with the hungry".to_string(), MoralLabel::Good),
            ("volunteering at shelters".to_string(), MoralLabel::Good),
            (
                "stealing from the poor is cruel".to_string(),
                MoralLabel::Bad,
            ),
            ("bullying children is wrong".to_string(), MoralLabel::Bad),
            ("lying to exploit people".to_string(), MoralLabel::Bad),
            ("murdering innocents is evil".to_string(), MoralLabel::Bad),
            ("walking to the store".to_string(), MoralLabel::Neutral),
            (
                "the weather is cloudy today".to_string(),
                MoralLabel::Neutral,
            ),
            ("reading a book at home".to_string(), MoralLabel::Neutral),
            ("eating lunch at noon".to_string(), MoralLabel::Neutral),
        ];

        clf.train(&samples);
        assert!(clf.is_trained());

        // Verify a clear good example classifies correctly
        let (verdict, _confidence) = clf.classify("helping kind generous caring love");
        // The classifier should at least not return Bad for clearly good text
        assert!(
            !matches!(verdict, MoralVerdict::Bad),
            "Clearly good text should not classify as Bad"
        );
    }

    #[test]
    fn test_confidence_bounded() {
        let basis = make_basis(4096);
        let mut clf = CfcMoralClassifier::new(basis, 4096);

        let samples = vec![
            ("helping is good".to_string(), MoralLabel::Good),
            ("stealing is bad".to_string(), MoralLabel::Bad),
            ("walking outside".to_string(), MoralLabel::Neutral),
        ];

        clf.train(&samples);

        let (_verdict, confidence) = clf.classify("some random text about morality");
        assert!(
            (0.0..=1.0).contains(&confidence),
            "Confidence should be in [0, 1], got {}",
            confidence
        );
    }
}
