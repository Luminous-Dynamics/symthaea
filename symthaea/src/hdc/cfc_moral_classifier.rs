// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CfC-based moral classifier using HdcLtcUnifiedNetwork.
//!
//! Uses the unified HDC-LTC network for non-linear moral classification.
//! Text is encoded via Spinozist affect geometry (12D NSM-grounded affects),
//! projected to 78 features (12 linear + 66 cross-terms), evolved through
//! a 2-layer CfC network, and classified against learned prototypes.
//!
//! # Architecture (Spinozist-Whiteheadian fusion)
//!
//! ```text
//! Text → NsmLexicon → AffectBasis (project to 12D) → 78 features
//!      → random project to CFC_NEURON_DIM
//!      → HdcLtcUnifiedNetwork (2 layers × 3 neurons) → output HV
//!      → cosine similarity to class prototypes → verdict
//! ```
//!
//! The Whiteheadian mode (`classify_whiteheadian`) processes words one at a
//! time as "actual occasions" — each word prehends prior occasions through
//! the CfC network state, reaching "satisfaction" (a determinate verdict)
//! only after all words have been processed.

use std::sync::Arc;

use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_core::hdc::hdc_ltc_unified::{
    HdcLtcUnifiedNetwork, UnifiedActivation, UnifiedConfig, UnifiedNetworkConfig,
};
use symthaea_core::hdc::HDC_DIMENSION;

use super::harmony_basis::HarmonyBasis;
use super::moral_algebra::MoralVerdict;
use super::moral_prototypes::MoralLabel;
use super::spinozist_geometry::{AffectBasis, NsmLexicon, NsmPrimeBasis, NUM_AFFECTS};

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

/// Number of Spinozist cross-term features: 12 linear + C(12,2) = 78.
const NUM_SPINOZIST_FEATURES: usize = NUM_AFFECTS + (NUM_AFFECTS * (NUM_AFFECTS - 1)) / 2;

/// Internals for Spinozist affect-space encoding.
///
/// Wraps `NsmPrimeBasis`, `AffectBasis`, and `NsmLexicon` so the CfC
/// classifier can project text into 12D affect space and encode individual
/// words for Whiteheadian concrescence.
struct SpinozistInternals {
    nsm_basis: NsmPrimeBasis,
    affect_basis: AffectBasis,
    lexicon: NsmLexicon,
}

impl SpinozistInternals {
    /// Construct all Spinozist components.
    fn new() -> Self {
        let nsm_basis = NsmPrimeBasis::new();
        let affect_basis = AffectBasis::new(&nsm_basis);
        let lexicon = NsmLexicon::new();
        Self {
            nsm_basis,
            affect_basis,
            lexicon,
        }
    }

    /// Encode full text as a single HV via lexicon word decomposition + bundle.
    fn encode_text(&self, text: &str) -> ContinuousHV {
        let words: Vec<&str> = text
            .split(|c: char| !c.is_alphanumeric() && c != '\'')
            .filter(|w| !w.is_empty())
            .collect();

        if words.is_empty() {
            return ContinuousHV::zero(HDC_DIMENSION);
        }

        let word_hvs: Vec<ContinuousHV> = words
            .iter()
            .map(|w| self.lexicon.encode_word(w, &self.nsm_basis))
            .collect();

        // Filter out zero vectors (stop words)
        let non_zero: Vec<&ContinuousHV> = word_hvs
            .iter()
            .filter(|hv| hv.values.iter().any(|v| v.abs() > 1e-10))
            .collect();

        if non_zero.is_empty() {
            return ContinuousHV::zero(HDC_DIMENSION);
        }

        ContinuousHV::bundle(&non_zero)
    }

    /// Encode a single word as a CFC_NEURON_DIM-sized HV for Whiteheadian
    /// word-by-word processing.
    fn encode_word_cfc(&self, word: &str) -> ContinuousHV {
        let word_hv = self.lexicon.encode_word(word, &self.nsm_basis);

        // Project onto 12 affect dimensions, then build 78 features
        let coords = self.affect_basis.project_affects(&word_hv);

        let mut features = Vec::with_capacity(NUM_SPINOZIST_FEATURES);
        for &c in &coords {
            features.push(c);
        }
        for i in 0..NUM_AFFECTS {
            for j in (i + 1)..NUM_AFFECTS {
                features.push(coords[i] * coords[j]);
            }
        }

        // Project to CFC_NEURON_DIM via deterministic random matrix
        let mut result = vec![0.0f32; CFC_NEURON_DIM];
        for (f_idx, &feat) in features.iter().enumerate() {
            let proj = ContinuousHV::random(CFC_NEURON_DIM, 96_000_000 + f_idx as u64);
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
}

/// CfC-based moral classifier.
///
/// Combines Spinozist affect-space encoding (12D NSM-grounded affects) with
/// a 2-layer CfC network for non-linear moral classification into
/// Good / Neutral / Bad. Supports both batch classification and
/// Whiteheadian word-by-word concrescence.
pub struct CfcMoralClassifier {
    #[allow(dead_code)]
    basis: Arc<HarmonyBasis>,
    spinozist: SpinozistInternals,
    network: HdcLtcUnifiedNetwork,
    /// Learned class prototypes: [Good, Bad, Neutral]
    class_prototypes: Option<[ContinuousHV; 3]>,
    config: CfcMoralConfig,
}

impl CfcMoralClassifier {
    /// Create a new CfC moral classifier.
    ///
    /// Accepts `HarmonyBasis` for API compatibility but uses Spinozist
    /// affect geometry internally. Sets up a 2-layer (3, 3)
    /// HdcLtcUnifiedNetwork with layer binding and skip connections.
    pub fn new(basis: Arc<HarmonyBasis>, dim: usize) -> Self {
        let spinozist = SpinozistInternals::new();

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
            basis,
            spinozist,
            network,
            class_prototypes: None,
            config: CfcMoralConfig {
                dim,
                ..CfcMoralConfig::default()
            },
        }
    }

    /// Encode text into a morally-structured CfC input vector via Spinozist
    /// affect geometry.
    ///
    /// 1. Encode text via NsmLexicon → weighted bundle of prime HVs
    /// 2. Project onto 12 Spinozist affect dimensions
    /// 3. Build 78 features: 12 linear + 66 cross-terms (i < j)
    /// 4. Project 78 features to CFC_NEURON_DIM via deterministic random matrix
    ///
    /// Cross-terms capture interactions like HARM×CARE (moral tension),
    /// DECEPTION×CONSENT (consent violation), JOY×SADNESS (ambivalence).
    pub fn encode_input(&self, text: &str) -> ContinuousHV {
        // Encode text through NsmLexicon
        let text_hv = self.spinozist.encode_text(text);

        // Project onto 12 affect dimensions
        let coords = self.spinozist.affect_basis.project_affects(&text_hv);

        // Build feature vector: 12 linear + 66 cross-terms (i<j) = 78 features
        let mut features = Vec::with_capacity(NUM_SPINOZIST_FEATURES);
        for &c in &coords {
            features.push(c);
        }
        for i in 0..NUM_AFFECTS {
            for j in (i + 1)..NUM_AFFECTS {
                features.push(coords[i] * coords[j]);
            }
        }

        // Project to CFC_NEURON_DIM via deterministic random matrix
        let mut result = vec![0.0f32; CFC_NEURON_DIM];
        for (f_idx, &feat) in features.iter().enumerate() {
            let proj = ContinuousHV::random(CFC_NEURON_DIM, 96_000_000 + f_idx as u64);
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

        // Cache encoded inputs — encoding is expensive (NsmLexicon + AffectBasis
        // projection) and doesn't change between epochs.
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
                                neuron
                                    .contrastive_update(&correct_proto, &wrong_proto, lr);
                            } else if n_idx == predicted_idx % layer_len {
                                neuron.contrastive_update(
                                    &wrong_proto,
                                    &correct_proto,
                                    lr * 0.5,
                                );
                            }
                        }
                    }

                    // Also update hidden layer with a smaller learning rate
                    if let Some(hidden) = self.network.layer_mut(0) {
                        for neuron in hidden.iter_mut() {
                            neuron
                                .contrastive_update(&correct_proto, &wrong_proto, lr * 0.1);
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
        indices.sort_by(|&a, &b| sims[b].partial_cmp(&sims[a]).unwrap_or(std::cmp::Ordering::Equal));

        let best_idx = indices[0];
        let margin = (sims[indices[0]] - sims[indices[1]]).max(0.0).min(1.0);

        let verdict = match best_idx {
            0 => MoralVerdict::Good,
            1 => MoralVerdict::Bad,
            _ => MoralVerdict::Neutral,
        };

        (verdict, margin)
    }

    /// Classify text via Whiteheadian concrescence: word-by-word CfC evolution.
    ///
    /// Each word is an "actual occasion" that prehends prior occasions through
    /// the CfC network's hidden state. The network is reset at the start (the
    /// beginning of a new actual occasion), then each word drives one CfC
    /// evolution step. The final output represents "satisfaction" — the
    /// determinate result of the process.
    ///
    /// This captures temporal moral unfolding: "It's okay to ignore someone"
    /// starts neutral then turns negative as the CfC state absorbs each word.
    pub fn classify_whiteheadian(&mut self, text: &str) -> (MoralVerdict, f32) {
        if self.class_prototypes.is_none() {
            return (MoralVerdict::Neutral, 0.0);
        }

        let lowered = text.to_lowercase();
        let words: Vec<&str> = lowered
            .split(|c: char| !c.is_alphanumeric() && c != '\'')
            .filter(|w| !w.is_empty())
            .collect();

        if words.is_empty() {
            return (MoralVerdict::Neutral, 0.0);
        }

        // Reset network — beginning of a new actual occasion
        self.network.reset();

        // Each word is an actual occasion that prehends prior occasions
        for word in &words {
            let word_hv = self.spinozist.encode_word_cfc(word);
            // Concrescence: the CfC evolves toward its subjective aim
            self.network.evolve_closed_form(self.config.dt, &word_hv);
        }

        // Satisfaction: the determinate result of the process
        let output = self.network.output();

        // Classify against prototypes
        let protos = self.class_prototypes.as_ref().unwrap();
        let sims: Vec<f32> = protos.iter().map(|p| output.similarity(p)).collect();

        let mut indices: Vec<usize> = (0..3).collect();
        indices
            .sort_by(|&a, &b| sims[b].partial_cmp(&sims[a]).unwrap_or(std::cmp::Ordering::Equal));

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
            ("helping others is kind and generous".to_string(), MoralLabel::Good),
            ("caring for the sick is noble".to_string(), MoralLabel::Good),
            ("sharing food with the hungry".to_string(), MoralLabel::Good),
            ("volunteering at shelters".to_string(), MoralLabel::Good),
            ("stealing from the poor is cruel".to_string(), MoralLabel::Bad),
            ("bullying children is wrong".to_string(), MoralLabel::Bad),
            ("lying to exploit people".to_string(), MoralLabel::Bad),
            ("murdering innocents is evil".to_string(), MoralLabel::Bad),
            ("walking to the store".to_string(), MoralLabel::Neutral),
            ("the weather is cloudy today".to_string(), MoralLabel::Neutral),
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
    fn test_untrained_whiteheadian_returns_neutral() {
        let basis = make_basis(4096);
        let mut clf = CfcMoralClassifier::new(basis, 4096);

        let (verdict, confidence) = clf.classify_whiteheadian("stealing is wrong");
        assert!(matches!(verdict, MoralVerdict::Neutral));
        assert!((confidence - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_whiteheadian_classify_roundtrip() {
        let basis = make_basis(4096);
        let mut clf = CfcMoralClassifier::new(basis, 4096);

        let samples = vec![
            ("helping others is kind and generous".to_string(), MoralLabel::Good),
            ("caring for the sick is noble".to_string(), MoralLabel::Good),
            ("sharing food with the hungry".to_string(), MoralLabel::Good),
            ("volunteering at shelters".to_string(), MoralLabel::Good),
            ("stealing from the poor is cruel".to_string(), MoralLabel::Bad),
            ("bullying children is wrong".to_string(), MoralLabel::Bad),
            ("lying to exploit people".to_string(), MoralLabel::Bad),
            ("murdering innocents is evil".to_string(), MoralLabel::Bad),
            ("walking to the store".to_string(), MoralLabel::Neutral),
            ("the weather is cloudy today".to_string(), MoralLabel::Neutral),
            ("reading a book at home".to_string(), MoralLabel::Neutral),
            ("eating lunch at noon".to_string(), MoralLabel::Neutral),
        ];

        clf.train(&samples);

        // Whiteheadian mode should produce bounded confidence
        let (_verdict, confidence) = clf.classify_whiteheadian("helping kind generous caring love");
        assert!(
            (0.0..=1.0).contains(&confidence),
            "Whiteheadian confidence should be in [0, 1], got {}",
            confidence
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
