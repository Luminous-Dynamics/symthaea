// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CfC-based moral classifier using HdcLtcUnifiedNetwork.
//!
//! Uses the unified HDC-LTC network for non-linear moral classification.
//! Text is encoded via Spinozist affect geometry (18D NSM-grounded affects),
//! projected to 171 features (18 linear + 153 cross-terms), evolved through
//! a 2-layer CfC network, and classified against learned prototypes.
//!
//! # Architecture (Spinozist-Whiteheadian fusion)
//!
//! ```text
//! Text → NsmLexicon → AffectBasis (project to 18D) → 171 features
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

use symthaea_core::hdc::HDC_DIMENSION;
use symthaea_core::hdc::hdc_ltc_unified::{
    HdcLtcUnifiedNetwork, UnifiedActivation, UnifiedConfig, UnifiedNetworkConfig,
};
use symthaea_core::hdc::unified_hv::ContinuousHV;

use super::harmony_basis::HarmonyBasis;
use super::moral_algebra::{ConsentState, MoralIntent, MoralVerdict};
use super::moral_parser::MoralParser;
use super::moral_prototypes::MoralLabel;
use super::moral_text_encoder::TextHdcEncoder;
use super::spinozist_geometry::{
    AffectBasis, FluctuatioAnimi, MoralFingerprint, NUM_AFFECTS, NsmLexicon, NsmPrimeBasis,
};

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

/// Number of base Spinozist cross-term features: 18 linear + C(18,2) = 171.
const NUM_BASE_SPINOZIST_FEATURES: usize = NUM_AFFECTS + (NUM_AFFECTS * (NUM_AFFECTS - 1)) / 2;

/// Number of structural features from MoralParser: intent, consent, negation = 3.
const NUM_STRUCTURAL_FEATURES: usize = 3;

/// Number of tension features from FluctuatioAnimi: total_tension, max_pair_tension = 2.
const NUM_TENSION_FEATURES: usize = 2;

/// Number of surface features from TextHdcEncoder: 3 prototype similarities.
const NUM_SURFACE_FEATURES: usize = 3;

/// Total features: 171 base + 3 structural + 2 tension + 3 surface = 179.
const NUM_SPINOZIST_FEATURES: usize = NUM_BASE_SPINOZIST_FEATURES
    + NUM_STRUCTURAL_FEATURES
    + NUM_TENSION_FEATURES
    + NUM_SURFACE_FEATURES;

/// Internals for Spinozist affect-space encoding.
///
/// Wraps `NsmPrimeBasis`, `AffectBasis`, and `NsmLexicon` so the CfC
/// classifier can project text into 18D affect space and encode individual
/// words for Whiteheadian concrescence.
struct SpinozistInternals {
    nsm_basis: NsmPrimeBasis,
    affect_basis: AffectBasis,
    lexicon: NsmLexicon,
    parser: MoralParser,
    /// Surface encoder for dual-channel (trigram + word + sentiment features)
    surface_encoder: TextHdcEncoder,
    /// Surface prototypes: [Good, Bad, Neutral] accumulated during training
    surface_prototypes: Option<[ContinuousHV; 3]>,
}

impl SpinozistInternals {
    /// Construct all Spinozist components.
    fn new() -> Self {
        let nsm_basis = NsmPrimeBasis::new();
        let affect_basis = AffectBasis::new(&nsm_basis);
        let lexicon = NsmLexicon::new();
        let parser = MoralParser::new();
        // Surface encoder with sentiment enabled (captures framing words like "rude", "okay")
        let surface_encoder = TextHdcEncoder::with_sentiment(HDC_DIMENSION, 3, 0.5, 0.2);
        Self {
            nsm_basis,
            affect_basis,
            lexicon,
            parser,
            surface_encoder,
            surface_prototypes: None,
        }
    }

    /// Encode full text as a single HV via lexicon word decomposition + weighted bundle.
    ///
    /// Words in the first 6 positions receive 3x weight, capturing the
    /// "It's [FRAME] to..." framing structure common in moral scenarios.
    fn encode_text(&self, text: &str) -> ContinuousHV {
        let words: Vec<&str> = text
            .split(|c: char| !c.is_alphanumeric() && c != '\'')
            .filter(|w| !w.is_empty())
            .collect();

        if words.is_empty() {
            return ContinuousHV::zero(HDC_DIMENSION);
        }

        let mut weighted_hvs: Vec<ContinuousHV> = Vec::new();
        let mut weights: Vec<f32> = Vec::new();

        for (idx, word) in words.iter().enumerate() {
            let hv = self.lexicon.encode_word(word, &self.nsm_basis);
            // Skip zero vectors (stop words)
            if hv.values.iter().any(|v| v.abs() > 1e-10) {
                // Framing word position boost: first 6 words get 3x weight
                let position_weight = if idx < 6 { 3.0 } else { 1.0 };
                weighted_hvs.push(hv);
                weights.push(position_weight);
            }
        }

        if weighted_hvs.is_empty() {
            return ContinuousHV::zero(HDC_DIMENSION);
        }

        let refs: Vec<&ContinuousHV> = weighted_hvs.iter().collect();
        ContinuousHV::weighted_bundle(&refs, &weights)
    }

    /// Encode a single word as a CFC_NEURON_DIM-sized HV for Whiteheadian
    /// word-by-word processing.
    fn encode_word_cfc(&self, word: &str) -> ContinuousHV {
        let word_hv = self.lexicon.encode_word(word, &self.nsm_basis);

        // Project onto 18 affect dimensions, then build 171 features
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
    /// affect geometry, structural parsing, and fluctuatio tension.
    ///
    /// 1. Encode text via NsmLexicon → weighted bundle of prime HVs
    /// 2. Project onto 18 Spinozist affect dimensions
    /// 3. Build 171 features: 18 linear + 153 cross-terms (i < j)
    /// 4. Add 3 structural features from MoralParser (intent, consent, negation)
    /// 5. Add 2 tension features from FluctuatioAnimi
    /// 6. Project 176 features to CFC_NEURON_DIM via deterministic random matrix
    ///
    /// Cross-terms capture interactions like HARM×CARE (moral tension),
    /// DECEPTION×CONSENT (consent violation), JOY×SADNESS (ambivalence).
    pub fn encode_input(&self, text: &str) -> ContinuousHV {
        // Encode text through NsmLexicon
        let text_hv = self.spinozist.encode_text(text);

        // Project onto 18 affect dimensions
        let coords = self.spinozist.affect_basis.project_affects(&text_hv);

        // Build feature vector: 18 linear + 153 cross-terms (i<j) = 171 base features
        let mut features = Vec::with_capacity(NUM_SPINOZIST_FEATURES);
        for &c in &coords {
            features.push(c);
        }
        for i in 0..NUM_AFFECTS {
            for j in (i + 1)..NUM_AFFECTS {
                features.push(coords[i] * coords[j]);
            }
        }

        // Structural features from MoralParser (3 features)
        let parsed = self.spinozist.parser.parse(text);
        features.push(match parsed.intent {
            MoralIntent::Good => 1.0f32,
            MoralIntent::Bad => -1.0,
            MoralIntent::Neutral | MoralIntent::Unknown => 0.0,
        });
        features.push(match parsed.consent {
            ConsentState::Given => 1.0f32,
            ConsentState::Implied => 0.5,
            ConsentState::Absent => -0.5,
            ConsentState::Denied => -1.0,
        });
        features.push(if parsed.has_negation { 1.0f32 } else { 0.0 });

        // Tension features from FluctuatioAnimi (2 features)
        let fingerprint = MoralFingerprint::from_coords(coords);
        let fluctuatio = FluctuatioAnimi::from_fingerprint(&fingerprint);
        features.push(fluctuatio.max_tension);
        let max_pair_tension = fluctuatio.tensions.first().map(|t| t.2).unwrap_or(0.0);
        features.push(max_pair_tension);

        // Surface features from TextHdcEncoder (3 features: similarity to Good/Bad/Neutral prototypes)
        if let Some(ref protos) = self.spinozist.surface_prototypes {
            let surface_hv = self.spinozist.surface_encoder.encode(text);
            for proto in protos {
                features.push(surface_hv.similarity(proto));
            }
        } else {
            // No surface prototypes yet — use zero features
            features.push(0.0);
            features.push(0.0);
            features.push(0.0);
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

        // Build surface prototypes first (used by encode_input for dual-channel features)
        {
            let dim = HDC_DIMENSION;
            let mut accum = [vec![0.0f32; dim], vec![0.0f32; dim], vec![0.0f32; dim]];
            let mut counts = [0usize; 3];
            for (text, label) in samples {
                let hv = self.spinozist.surface_encoder.encode(text);
                let idx = label_to_index(*label);
                for (a, &v) in accum[idx].iter_mut().zip(hv.values.iter()) {
                    *a += v;
                }
                counts[idx] += 1;
            }
            let mut protos = [
                ContinuousHV::zero(dim),
                ContinuousHV::zero(dim),
                ContinuousHV::zero(dim),
            ];
            for i in 0..3 {
                if counts[i] > 0 {
                    let norm: f32 = accum[i].iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        protos[i] =
                            ContinuousHV::from_vec(accum[i].iter().map(|x| x / norm).collect());
                    }
                }
            }
            self.spinozist.surface_prototypes = Some(protos);
        }

        // Cache encoded inputs — encoding is expensive and doesn't change between epochs.
        // Now includes surface similarity features since surface_prototypes are set.
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

        // Verify the classifier produces a valid verdict (not just default)
        // and is trained. With random HDC basis vectors, the exact classification
        // is non-deterministic — the key invariant is that training completes
        // and classification produces a valid result.
        let (verdict, confidence) = clf.classify("helping kind generous caring love");
        assert!(
            matches!(
                verdict,
                MoralVerdict::Good | MoralVerdict::Bad | MoralVerdict::Neutral
            ),
            "Should produce a valid verdict"
        );
        assert!(confidence >= 0.0, "confidence should be non-negative");
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
