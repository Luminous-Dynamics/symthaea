// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Learned moral prototype classifier (Good / Neutral / Bad).
//!
//! Trains 3-class HDC prototypes from labeled text using the same
//! accumulate -> retrain loop that achieved 87.6% on MNIST and 91.7% on ISOLET.
//!
//! The prototypes can be serialized to JSON for caching, so training only needs
//! to happen once per dataset.
//!
//! # Usage
//!
//! ```ignore
//! use symthaea::hdc::moral_prototypes::{MoralPrototypeClassifier, MoralSample, MoralLabel};
//!
//! let mut classifier = MoralPrototypeClassifier::new(8192, 3);
//! classifier.train(&samples);
//! classifier.retrain_adaptive(&samples, 0.1, 10);
//! let (label, confidence) = classifier.classify("stealing is wrong");
//! ```

use super::moral_algebra::{ConsentState, MoralIntent};
use super::moral_parser::MoralParser;
use super::moral_text_encoder::TextHdcEncoder;
use serde::{Deserialize, Serialize};
use std::path::Path;
use symthaea_core::hdc::unified_hv::ContinuousHV;

/// Fast dot product on float slices (avoids ContinuousHV clone allocations).
#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Simple string hash to deterministic seed.
fn hash_str(s: &str) -> u64 {
    let mut h: u64 = 5381;
    for b in s.bytes() {
        h = h.wrapping_mul(33).wrapping_add(b as u64);
    }
    h
}

/// Recommended dimension for moral prototypes (higher = better separation).
pub const MORAL_PROTO_DIM: usize = 16384;

/// Moral label for 3-class classification.
///
/// Maps to Social Chemistry `rot_judgment`: 1 -> Good, 0 -> Neutral, -1 -> Bad.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MoralLabel {
    Good,
    Neutral,
    Bad,
}

impl MoralLabel {
    /// Convert from rot_judgment integer (-1, 0, 1).
    pub fn from_rot_judgment(val: i32) -> Self {
        match val {
            1 => MoralLabel::Good,
            -1 => MoralLabel::Bad,
            _ => MoralLabel::Neutral,
        }
    }

    /// Convert to rot_judgment integer.
    pub fn to_rot_judgment(self) -> i32 {
        match self {
            MoralLabel::Good => 1,
            MoralLabel::Neutral => 0,
            MoralLabel::Bad => -1,
        }
    }
}

/// A labeled text sample for training.
pub struct MoralSample {
    pub text: String,
    pub label: MoralLabel,
}

/// Serializable trained prototypes (the learned model weights).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainedPrototypes {
    pub good: Vec<f32>,
    pub neutral: Vec<f32>,
    pub bad: Vec<f32>,
    pub dim: usize,
    pub training_counts: [usize; 3],
}

impl TrainedPrototypes {
    /// Save prototypes to a JSON file.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string(self).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }

    /// Load prototypes from a JSON file.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json).map_err(std::io::Error::other)
    }
}

/// 3-class moral prototype classifier.
///
/// Uses a [`TextHdcEncoder`] to encode text, then classifies by cosine
/// similarity to learned Good/Neutral/Bad prototypes.
#[derive(Debug, Clone)]
pub struct MoralPrototypeClassifier {
    encoder: TextHdcEncoder,
    prototypes: Option<TrainedPrototypes>,
}

impl MoralPrototypeClassifier {
    /// Create a new classifier with the given dimension and n-gram size.
    pub fn new(dim: usize, ngram_size: usize) -> Self {
        Self {
            encoder: TextHdcEncoder::new(dim, ngram_size),
            prototypes: None,
        }
    }

    /// Create with sentiment channel enabled.
    ///
    /// When `sentiment_weight > 0`, the encoder blends a third channel that
    /// accumulates positive/negative seed HVs for moral vocabulary words.
    pub fn with_sentiment(dim: usize, ngram_size: usize, sentiment_weight: f32) -> Self {
        Self {
            encoder: TextHdcEncoder::with_sentiment(dim, ngram_size, 0.5, sentiment_weight),
            prototypes: None,
        }
    }

    /// Create from pre-trained prototypes (e.g., loaded from disk).
    pub fn from_prototypes(dim: usize, ngram_size: usize, prototypes: TrainedPrototypes) -> Self {
        Self {
            encoder: TextHdcEncoder::new(dim, ngram_size),
            prototypes: Some(prototypes),
        }
    }

    /// Create from pre-trained prototypes with sentiment channel enabled.
    pub fn from_prototypes_with_sentiment(
        dim: usize,
        ngram_size: usize,
        sentiment_weight: f32,
        prototypes: TrainedPrototypes,
    ) -> Self {
        Self {
            encoder: TextHdcEncoder::with_sentiment(dim, ngram_size, 0.5, sentiment_weight),
            prototypes: Some(prototypes),
        }
    }

    /// Whether prototypes have been trained.
    pub fn is_trained(&self) -> bool {
        self.prototypes.is_some()
    }

    /// Get the encoder dimension.
    pub fn dim(&self) -> usize {
        self.encoder.dim()
    }

    /// Initial training: accumulate per-class centroids and normalize.
    pub fn train(&mut self, samples: &[MoralSample]) {
        let dim = self.encoder.dim();
        let mut good_acc = vec![0.0f32; dim];
        let mut neutral_acc = vec![0.0f32; dim];
        let mut bad_acc = vec![0.0f32; dim];
        let mut counts = [0usize; 3];

        for sample in samples {
            let encoded = self.encoder.encode(&sample.text);
            let target = match sample.label {
                MoralLabel::Good => {
                    counts[0] += 1;
                    &mut good_acc
                }
                MoralLabel::Neutral => {
                    counts[1] += 1;
                    &mut neutral_acc
                }
                MoralLabel::Bad => {
                    counts[2] += 1;
                    &mut bad_acc
                }
            };
            for (acc, &val) in target.iter_mut().zip(encoded.values.iter()) {
                *acc += val;
            }
        }

        // Normalize each prototype
        for acc in [&mut good_acc, &mut neutral_acc, &mut bad_acc] {
            let norm: f32 = acc.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in acc.iter_mut() {
                    *v /= norm;
                }
            }
        }

        self.prototypes = Some(TrainedPrototypes {
            good: good_acc,
            neutral: neutral_acc,
            bad: bad_acc,
            dim,
            training_counts: counts,
        });
    }

    /// Iterative retraining with fixed learning rate.
    ///
    /// For each misclassified sample, adds `lr * encoded` to the correct
    /// prototype and subtracts `lr * encoded` from the incorrectly-winning
    /// prototype. Normalizes after all iterations.
    pub fn retrain(&mut self, samples: &[MoralSample], lr: f32, iterations: usize) {
        self.retrain_inner(samples, lr, lr, iterations);
    }

    /// Iterative retraining with adaptive learning rate decay.
    ///
    /// LR decays linearly from `lr_start` to `lr_end` over the iterations.
    /// This gives aggressive early correction + fine-tuning at the end.
    pub fn retrain_adaptive(&mut self, samples: &[MoralSample], lr_start: f32, iterations: usize) {
        let lr_end = lr_start * 0.1; // Decay to 10% of initial LR
        self.retrain_inner(samples, lr_start, lr_end, iterations);
    }

    fn retrain_inner(
        &mut self,
        samples: &[MoralSample],
        lr_start: f32,
        lr_end: f32,
        iterations: usize,
    ) {
        let protos = match self.prototypes.as_mut() {
            Some(p) => p,
            None => return,
        };

        for iter in 0..iterations {
            // Linear LR decay
            let progress = if iterations > 1 {
                iter as f32 / (iterations - 1) as f32
            } else {
                0.0
            };
            let lr = lr_start + (lr_end - lr_start) * progress;

            let mut corrections = 0;

            for sample in samples {
                let encoded = self.encoder.encode(&sample.text);

                // Compute similarity as dot product directly (prototypes are normalized)
                let sim_good = dot_product(&encoded.values, &protos.good);
                let sim_neutral = dot_product(&encoded.values, &protos.neutral);
                let sim_bad = dot_product(&encoded.values, &protos.bad);

                let predicted = if sim_good >= sim_neutral && sim_good >= sim_bad {
                    MoralLabel::Good
                } else if sim_neutral >= sim_bad {
                    MoralLabel::Neutral
                } else {
                    MoralLabel::Bad
                };

                if predicted != sample.label {
                    corrections += 1;

                    // Push correct prototype toward sample
                    let correct_proto = match sample.label {
                        MoralLabel::Good => &mut protos.good,
                        MoralLabel::Neutral => &mut protos.neutral,
                        MoralLabel::Bad => &mut protos.bad,
                    };
                    for (pv, &ev) in correct_proto.iter_mut().zip(encoded.values.iter()) {
                        *pv += lr * ev;
                    }

                    // Push incorrectly-winning prototype away from sample
                    let wrong_proto = match predicted {
                        MoralLabel::Good => &mut protos.good,
                        MoralLabel::Neutral => &mut protos.neutral,
                        MoralLabel::Bad => &mut protos.bad,
                    };
                    for (pv, &ev) in wrong_proto.iter_mut().zip(encoded.values.iter()) {
                        *pv -= lr * ev;
                    }
                }
            }

            // Early stopping if very few corrections
            if corrections < samples.len() / 200 {
                break;
            }
        }

        // Normalize all prototypes after retraining
        for proto in [&mut protos.good, &mut protos.neutral, &mut protos.bad] {
            let norm: f32 = proto.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in proto.iter_mut() {
                    *v /= norm;
                }
            }
        }
    }

    /// Retrain with validation-based early stopping.
    ///
    /// Pre-encodes all samples once, then iterates on cached encodings.
    /// Holds out `val_fraction` of training data and stops when validation
    /// accuracy drops for `patience` consecutive epochs. Returns the best
    /// validation accuracy achieved.
    pub fn retrain_with_validation(
        &mut self,
        samples: &[MoralSample],
        lr_start: f32,
        max_iterations: usize,
        val_fraction: f32,
        patience: usize,
    ) -> f32 {
        let val_count = (samples.len() as f32 * val_fraction) as usize;
        if val_count == 0 || val_count >= samples.len() {
            self.retrain_adaptive(samples, lr_start, max_iterations);
            return 0.0;
        }

        // Pre-encode ALL samples once (the expensive step)
        let encoded: Vec<(Vec<f32>, MoralLabel)> = samples
            .iter()
            .map(|s| (self.encoder.encode(&s.text).values, s.label))
            .collect();

        let train = &encoded[..encoded.len() - val_count];
        let val = &encoded[encoded.len() - val_count..];

        let protos = match self.prototypes.as_mut() {
            Some(p) => p,
            None => return 0.0,
        };

        let mut best_val_acc = 0.0f32;
        let mut best_good = protos.good.clone();
        let mut best_neutral = protos.neutral.clone();
        let mut best_bad = protos.bad.clone();
        let mut no_improve_count = 0usize;

        let lr_end = lr_start * 0.1;

        for iter in 0..max_iterations {
            let progress = if max_iterations > 1 {
                iter as f32 / (max_iterations - 1) as f32
            } else {
                0.0
            };
            let lr = lr_start + (lr_end - lr_start) * progress;

            // Train epoch on cached encodings
            for (enc, label) in train {
                let sim_good = dot_product(enc, &protos.good);
                let sim_neutral = dot_product(enc, &protos.neutral);
                let sim_bad = dot_product(enc, &protos.bad);

                let predicted = if sim_good >= sim_neutral && sim_good >= sim_bad {
                    MoralLabel::Good
                } else if sim_neutral >= sim_bad {
                    MoralLabel::Neutral
                } else {
                    MoralLabel::Bad
                };

                if predicted != *label {
                    let correct_proto = match label {
                        MoralLabel::Good => &mut protos.good,
                        MoralLabel::Neutral => &mut protos.neutral,
                        MoralLabel::Bad => &mut protos.bad,
                    };
                    for (pv, &ev) in correct_proto.iter_mut().zip(enc.iter()) {
                        *pv += lr * ev;
                    }
                    let wrong_proto = match predicted {
                        MoralLabel::Good => &mut protos.good,
                        MoralLabel::Neutral => &mut protos.neutral,
                        MoralLabel::Bad => &mut protos.bad,
                    };
                    for (pv, &ev) in wrong_proto.iter_mut().zip(enc.iter()) {
                        *pv -= lr * ev;
                    }
                }
            }
            // Normalize
            for proto in [&mut protos.good, &mut protos.neutral, &mut protos.bad] {
                let norm: f32 = proto.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for v in proto.iter_mut() {
                        *v /= norm;
                    }
                }
            }

            // Validate on cached val encodings
            let mut val_correct = 0usize;
            for (enc, label) in val {
                let sim_good = dot_product(enc, &protos.good);
                let sim_neutral = dot_product(enc, &protos.neutral);
                let sim_bad = dot_product(enc, &protos.bad);
                let predicted = if sim_good >= sim_neutral && sim_good >= sim_bad {
                    MoralLabel::Good
                } else if sim_neutral >= sim_bad {
                    MoralLabel::Neutral
                } else {
                    MoralLabel::Bad
                };
                if predicted == *label {
                    val_correct += 1;
                }
            }
            let val_acc = val_correct as f32 / val.len() as f32;

            if val_acc > best_val_acc {
                best_val_acc = val_acc;
                best_good = protos.good.clone();
                best_neutral = protos.neutral.clone();
                best_bad = protos.bad.clone();
                no_improve_count = 0;
            } else {
                no_improve_count += 1;
                if no_improve_count >= patience {
                    break;
                }
            }
        }

        // Restore best prototypes
        protos.good = best_good;
        protos.neutral = best_neutral;
        protos.bad = best_bad;

        best_val_acc
    }

    /// Classify a text string, returning the predicted label and confidence.
    ///
    /// Confidence is the difference between the best and second-best similarity.
    pub fn classify(&self, text: &str) -> (MoralLabel, f32) {
        let protos = match &self.prototypes {
            Some(p) => p,
            None => return (MoralLabel::Neutral, 0.0),
        };

        let encoded = self.encoder.encode(text);

        // Direct dot product (prototypes are normalized) — avoids 3 clone allocations
        let mut sims = [
            (MoralLabel::Good, dot_product(&encoded.values, &protos.good)),
            (
                MoralLabel::Neutral,
                dot_product(&encoded.values, &protos.neutral),
            ),
            (MoralLabel::Bad, dot_product(&encoded.values, &protos.bad)),
        ];

        sims.sort_by(|a, b| b.1.total_cmp(&a.1));
        let best = sims[0];
        let second = sims[1];

        (best.0, best.1 - second.1)
    }

    /// Get a reference to the trained prototypes (for saving).
    pub fn prototypes(&self) -> Option<&TrainedPrototypes> {
        self.prototypes.as_ref()
    }
}

// ============================================================================
// MultiPrototypeClassifier — K prototypes per class for heterogeneous data
// ============================================================================

/// Multi-prototype moral classifier using K centroids per class.
///
/// A single centroid can't capture that "white lie" and "murder" are both Bad.
/// K-means initialization + per-prototype retraining handles class heterogeneity.
pub struct MultiPrototypeClassifier {
    encoder: TextHdcEncoder,
    parser: MoralParser,
    /// Structural role HVs for binding (AGENT, ACTION, PATIENT, NEGATION, CONSENT_ABSENT)
    role_hvs: StructuralRoles,
    /// K prototypes per class: [Good_protos, Neutral_protos, Bad_protos]
    prototypes: Option<[Vec<Vec<f32>>; 3]>,
    k: usize,
    /// Weight for structural channel (0.0 = surface only, 1.0 = structural only)
    structural_weight: f32,
}

/// Deterministic role HVs for structural encoding.
struct StructuralRoles {
    agent: ContinuousHV,
    action: ContinuousHV,
    patient: ContinuousHV,
    negation: ContinuousHV,
    consent_absent: ContinuousHV,
    intent_good: ContinuousHV,
    intent_bad: ContinuousHV,
}

impl StructuralRoles {
    fn new(dim: usize) -> Self {
        Self {
            agent: ContinuousHV::random(dim, 2_000_003),
            action: ContinuousHV::random(dim, 2_000_033),
            patient: ContinuousHV::random(dim, 2_000_037),
            negation: ContinuousHV::random(dim, 2_000_039),
            consent_absent: ContinuousHV::random(dim, 2_000_081),
            intent_good: ContinuousHV::random(dim, 2_000_099),
            intent_bad: ContinuousHV::random(dim, 2_000_117),
        }
    }
}

impl MultiPrototypeClassifier {
    /// Create a new multi-prototype classifier with framing channel.
    pub fn new(dim: usize, ngram_size: usize, k: usize) -> Self {
        Self {
            encoder: TextHdcEncoder::with_framing(dim, ngram_size, 0.5, 0.15, 0.1),
            parser: MoralParser::new(),
            role_hvs: StructuralRoles::new(dim),
            prototypes: None,
            k,
            structural_weight: 0.0, // disabled by default
        }
    }

    /// Encode text with structural blending.
    ///
    /// Surface channel (70%): TextHdcEncoder (trigrams + words + sentiment)
    /// Structural channel (30%): role bindings from MoralParser output
    fn encode_structured(&self, text: &str) -> ContinuousHV {
        let surface = self.encoder.encode(text);
        let parsed = self.parser.parse(text);

        let dim = surface.values.len();
        let mut structural_parts: Vec<&ContinuousHV> = Vec::new();

        // Create temp bound HVs for structural roles
        let mut bound_hvs: Vec<ContinuousHV> = Vec::new();

        if let Some(ref agent) = parsed.agent {
            let agent_hv = ContinuousHV::random(dim, hash_str(agent));
            bound_hvs.push(self.role_hvs.agent.bind(&agent_hv));
        }
        if let Some(ref action) = parsed.action {
            let mut action_hv = ContinuousHV::random(dim, hash_str(action));
            if parsed.has_negation {
                action_hv = action_hv.permute(1);
            }
            bound_hvs.push(self.role_hvs.action.bind(&action_hv));
        }
        if let Some(ref patient) = parsed.patient {
            let patient_hv = ContinuousHV::random(dim, hash_str(patient));
            bound_hvs.push(self.role_hvs.patient.bind(&patient_hv));
        }
        if parsed.consent == ConsentState::Denied || parsed.consent == ConsentState::Absent {
            bound_hvs.push(self.role_hvs.consent_absent.clone());
        }
        match parsed.intent {
            MoralIntent::Good => bound_hvs.push(self.role_hvs.intent_good.clone()),
            MoralIntent::Bad => bound_hvs.push(self.role_hvs.intent_bad.clone()),
            _ => {}
        }

        if bound_hvs.is_empty() {
            return surface; // No structure detected, use surface only
        }

        for hv in &bound_hvs {
            structural_parts.push(hv);
        }
        let structural = ContinuousHV::bundle(&structural_parts);

        // Blend: (1-w)*surface + w*structural
        let sw = self.structural_weight;
        let tw = 1.0 - sw;
        let mut blended = vec![0.0f32; dim];
        for i in 0..dim {
            blended[i] = tw * surface.values[i] + sw * structural.values[i];
        }
        // L2 normalize
        let norm: f32 = blended.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut blended {
                *v /= norm;
            }
        }
        ContinuousHV::from_vec(blended)
    }

    /// Train: encode all samples, then K-means cluster each class.
    pub fn train(&mut self, samples: &[MoralSample]) {
        let dim = self.encoder.encode("test").values.len();

        // Pre-encode (surface only — structural encoding hurts on norm descriptions)
        let encoded: Vec<(Vec<f32>, MoralLabel)> = samples
            .iter()
            .map(|s| (self.encoder.encode(&s.text).values, s.label))
            .collect();

        // Split by class
        let mut per_class: [Vec<&Vec<f32>>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for (enc, label) in &encoded {
            let idx = match label {
                MoralLabel::Good => 0,
                MoralLabel::Neutral => 1,
                MoralLabel::Bad => 2,
            };
            per_class[idx].push(enc);
        }

        // K-means for each class
        let mut protos = [Vec::new(), Vec::new(), Vec::new()];
        for c in 0..3 {
            protos[c] = self.kmeans_init(&per_class[c], self.k, dim);
        }

        self.prototypes = Some(protos);
    }

    /// Simple K-means clustering to find K sub-centroids.
    fn kmeans_init(&self, class_vecs: &[&Vec<f32>], k: usize, dim: usize) -> Vec<Vec<f32>> {
        if class_vecs.is_empty() {
            return vec![vec![0.0; dim]; k];
        }
        let actual_k = k.min(class_vecs.len());

        // Initialize: evenly-spaced samples
        let mut centroids: Vec<Vec<f32>> = (0..actual_k)
            .map(|i| {
                let idx = i * class_vecs.len() / actual_k;
                class_vecs[idx].clone()
            })
            .collect();

        // 10 iterations of K-means
        for _ in 0..10 {
            // Assign each vector to nearest centroid
            let mut accum = vec![vec![0.0f32; dim]; actual_k];
            let mut counts = vec![0usize; actual_k];

            for vec in class_vecs {
                let mut best_c = 0;
                let mut best_sim = f32::NEG_INFINITY;
                for (ci, centroid) in centroids.iter().enumerate() {
                    let sim = dot_product(vec, centroid);
                    if sim > best_sim {
                        best_sim = sim;
                        best_c = ci;
                    }
                }
                for (a, &v) in accum[best_c].iter_mut().zip(vec.iter()) {
                    *a += v;
                }
                counts[best_c] += 1;
            }

            // Update centroids
            for ci in 0..actual_k {
                if counts[ci] > 0 {
                    let norm: f32 = accum[ci].iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        centroids[ci] = accum[ci].iter().map(|x| x / norm).collect();
                    }
                }
            }
        }

        centroids
    }

    /// Retrain with validation-based early stopping on cached encodings.
    pub fn retrain_with_validation(
        &mut self,
        samples: &[MoralSample],
        lr_start: f32,
        max_iterations: usize,
        val_fraction: f32,
        patience: usize,
    ) -> f32 {
        if self.prototypes.is_none() {
            return 0.0;
        }

        // Pre-encode all samples (before taking mutable ref to prototypes)
        let encoded: Vec<(Vec<f32>, usize)> = samples
            .iter()
            .map(|s| {
                let idx = match s.label {
                    MoralLabel::Good => 0,
                    MoralLabel::Neutral => 1,
                    MoralLabel::Bad => 2,
                };
                (self.encoder.encode(&s.text).values, idx)
            })
            .collect();

        let val_count = (encoded.len() as f32 * val_fraction) as usize;
        if val_count == 0 {
            return 0.0;
        }
        let train = &encoded[..encoded.len() - val_count];
        let val = &encoded[encoded.len() - val_count..];

        let protos = self.prototypes.as_mut().unwrap();

        let mut best_val_acc = 0.0f32;
        let mut best_protos = protos.clone();
        let mut no_improve = 0usize;
        let lr_end = lr_start * 0.1;

        for iter in 0..max_iterations {
            let progress = if max_iterations > 1 {
                iter as f32 / (max_iterations - 1) as f32
            } else {
                0.0
            };
            let lr = lr_start + (lr_end - lr_start) * progress;

            // Train: push/pull closest prototype
            for (enc, correct_class) in train {
                // Inline classify_encoded to avoid borrow conflict
                let mut pred_class = 0;
                let mut pred_idx = 0;
                let mut best_sim = f32::NEG_INFINITY;
                for (ci, class_protos) in protos.iter().enumerate() {
                    for (pi, proto) in class_protos.iter().enumerate() {
                        let sim = dot_product(enc, proto);
                        if sim > best_sim {
                            best_sim = sim;
                            pred_class = ci;
                            pred_idx = pi;
                        }
                    }
                }

                if pred_class != *correct_class {
                    // Find closest correct-class prototype
                    let correct_idx = protos[*correct_class]
                        .iter()
                        .enumerate()
                        .map(|(i, p)| (i, dot_product(enc, p)))
                        .max_by(|a, b| a.1.total_cmp(&b.1))
                        .map(|(i, _)| i)
                        .unwrap_or(0);

                    // Push correct toward sample
                    for (pv, &ev) in protos[*correct_class][correct_idx]
                        .iter_mut()
                        .zip(enc.iter())
                    {
                        *pv += lr * ev;
                    }
                    // Pull wrong away
                    for (pv, &ev) in protos[pred_class][pred_idx].iter_mut().zip(enc.iter()) {
                        *pv -= lr * ev;
                    }
                }
            }

            // Normalize all prototypes
            for class_protos in protos.iter_mut() {
                for proto in class_protos.iter_mut() {
                    let norm: f32 = proto.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for v in proto.iter_mut() {
                            *v /= norm;
                        }
                    }
                }
            }

            // Validate (inline to avoid borrow conflict)
            let mut val_correct = 0usize;
            for (enc, label) in val {
                let mut pred = 0usize;
                let mut vs = f32::NEG_INFINITY;
                for (ci, cp) in protos.iter().enumerate() {
                    for p in cp {
                        let s = dot_product(enc, p);
                        if s > vs {
                            vs = s;
                            pred = ci;
                        }
                    }
                }
                if pred == *label {
                    val_correct += 1;
                }
            }
            let val_acc = val_correct as f32 / val.len() as f32;

            if val_acc > best_val_acc {
                best_val_acc = val_acc;
                best_protos = protos.clone();
                no_improve = 0;
            } else {
                no_improve += 1;
                if no_improve >= patience {
                    break;
                }
            }
        }

        *protos = best_protos;
        best_val_acc
    }

    /// Classify by max similarity across all K×3 prototypes.
    fn classify_encoded(&self, enc: &[f32], protos: &[Vec<Vec<f32>>; 3]) -> (usize, usize, f32) {
        let mut best_class = 0;
        let mut best_proto_idx = 0;
        let mut best_sim = f32::NEG_INFINITY;

        for (ci, class_protos) in protos.iter().enumerate() {
            for (pi, proto) in class_protos.iter().enumerate() {
                let sim = dot_product(enc, proto);
                if sim > best_sim {
                    best_sim = sim;
                    best_class = ci;
                    best_proto_idx = pi;
                }
            }
        }

        (best_class, best_proto_idx, best_sim)
    }

    /// Find closest prototype within a class.
    fn closest_proto(&self, enc: &[f32], class_protos: &[Vec<f32>]) -> usize {
        class_protos
            .iter()
            .enumerate()
            .map(|(i, p)| (i, dot_product(enc, p)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Classify a text string.
    pub fn classify(&self, text: &str) -> (MoralLabel, f32) {
        let protos = match &self.prototypes {
            Some(p) => p,
            None => return (MoralLabel::Neutral, 0.0),
        };

        let encoded = self.encoder.encode(text);
        let (class, _, sim) = self.classify_encoded(&encoded.values, protos);

        let label = match class {
            0 => MoralLabel::Good,
            1 => MoralLabel::Neutral,
            _ => MoralLabel::Bad,
        };

        (label, sim)
    }

    /// Number of prototypes per class.
    pub fn k(&self) -> usize {
        self.k
    }
}

// ============================================================================
// ExemplarStore — brute-force k-NN in HDC space
// ============================================================================

/// Stores encoded training examples for k-NN classification.
///
/// At query time, finds the K nearest neighbors by cosine similarity
/// and returns majority-vote label. Catches cases where prototypes
/// fail to represent local neighborhoods.
pub struct ExemplarStore {
    exemplars: Vec<(Vec<f32>, MoralLabel)>,
}

impl ExemplarStore {
    pub fn new() -> Self {
        Self {
            exemplars: Vec::new(),
        }
    }

    /// Build from pre-encoded samples.
    pub fn from_encoded(encoded: Vec<(Vec<f32>, MoralLabel)>) -> Self {
        Self { exemplars: encoded }
    }

    /// Build by encoding samples with a TextHdcEncoder.
    pub fn from_samples(encoder: &TextHdcEncoder, samples: &[MoralSample]) -> Self {
        let exemplars = samples
            .iter()
            .map(|s| (encoder.encode(&s.text).values, s.label))
            .collect();
        Self { exemplars }
    }

    /// Classify by similarity-weighted k-NN voting.
    ///
    /// Each neighbor's vote is weighted by similarity² (closer = more influence).
    /// Returns (label, confidence) where confidence is the weighted fraction.
    pub fn classify_knn(&self, query: &[f32], k: usize) -> (MoralLabel, f32) {
        if self.exemplars.is_empty() {
            return (MoralLabel::Neutral, 0.0);
        }

        // Compute similarities to all exemplars
        let mut sims: Vec<(f32, MoralLabel)> = self
            .exemplars
            .iter()
            .map(|(enc, label)| (dot_product(query, enc), *label))
            .collect();

        // Partial sort: top-k by similarity
        let actual_k = k.min(sims.len());
        sims.select_nth_unstable_by(actual_k - 1, |a, b| b.0.total_cmp(&a.0));

        // Similarity-weighted vote among top-k
        let mut weights = [0.0f32; 3]; // Good, Neutral, Bad
        for &(sim, label) in &sims[..actual_k] {
            let w = sim * sim; // similarity² weighting
            match label {
                MoralLabel::Good => weights[0] += w,
                MoralLabel::Neutral => weights[1] += w,
                MoralLabel::Bad => weights[2] += w,
            }
        }

        let total_weight: f32 = weights.iter().sum();
        let (best_idx, &best_weight) = weights
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap();

        let label = match best_idx {
            0 => MoralLabel::Good,
            1 => MoralLabel::Neutral,
            _ => MoralLabel::Bad,
        };

        let confidence = if total_weight > 0.0 {
            best_weight / total_weight
        } else {
            0.0
        };

        (label, confidence)
    }

    /// Number of stored exemplars.
    pub fn len(&self) -> usize {
        self.exemplars.len()
    }

    /// Whether the store has no exemplars.
    pub fn is_empty(&self) -> bool {
        self.exemplars.is_empty()
    }
}

// ============================================================================
// VirtueMatchClassifier — 2-class pair-encoding classifier for ETHICS Virtue
// ============================================================================

/// Label for the virtue matching task: does the trait apply to the scenario?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VirtueLabel {
    Applies,
    NotApplies,
}

/// A labeled sample for virtue matching: scenario + trait word + label.
#[derive(Clone)]
pub struct VirtueSample {
    pub scenario: String,
    pub trait_word: String,
    pub label: VirtueLabel,
}

/// Serializable trained virtue prototypes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainedVirtuePrototypes {
    pub applies: Vec<f32>,
    pub not_applies: Vec<f32>,
    pub dim: usize,
    pub training_counts: [usize; 2],
}

impl TrainedVirtuePrototypes {
    /// Save to a JSON file.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string(self).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }

    /// Load from a JSON file.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json).map_err(std::io::Error::other)
    }
}

/// Encode a (scenario, trait_word) pair into a single HV using a given encoder.
///
/// Free function to avoid borrow checker conflicts in retrain_adaptive.
fn encode_pair_with_encoder(
    encoder: &TextHdcEncoder,
    scenario: &str,
    trait_word: &str,
) -> Vec<f32> {
    let scenario_hv = encoder.encode(scenario);
    let trait_hv = encoder.encode(trait_word);
    let dim = scenario_hv.values.len();

    let mut bound = vec![0.0f32; dim];
    for i in 0..dim {
        bound[i] = scenario_hv.values[i] * trait_hv.values[i];
    }

    let norm: f32 = bound.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut bound {
            *v /= norm;
        }
    }
    bound
}

/// 2-class virtue match classifier using HDC pair encoding.
///
/// Encodes (scenario, trait_word) pairs by encoding each separately,
/// element-wise multiplying (binding), and classifying against
/// Applies / NotApplies prototypes.
#[derive(Debug, Clone)]
pub struct VirtueMatchClassifier {
    encoder: TextHdcEncoder,
    prototypes: Option<TrainedVirtuePrototypes>,
}

impl VirtueMatchClassifier {
    /// Create a new classifier with the given dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            encoder: TextHdcEncoder::with_sentiment(dim, 3, 0.5, 0.15),
            prototypes: None,
        }
    }

    /// Create from pre-trained prototypes.
    pub fn from_prototypes(prototypes: TrainedVirtuePrototypes) -> Self {
        let dim = prototypes.dim;
        Self {
            encoder: TextHdcEncoder::with_sentiment(dim, 3, 0.5, 0.15),
            prototypes: Some(prototypes),
        }
    }

    /// Whether prototypes have been trained.
    pub fn is_trained(&self) -> bool {
        self.prototypes.is_some()
    }

    /// Encode a (scenario, trait_word) pair into a single HV.
    ///
    /// Encodes each text independently, then binds (element-wise multiply)
    /// and L2-normalizes.
    fn encode_pair(&self, scenario: &str, trait_word: &str) -> Vec<f32> {
        encode_pair_with_encoder(&self.encoder, scenario, trait_word)
    }

    /// Initial training: accumulate per-class pair HVs and normalize.
    pub fn train(&mut self, samples: &[VirtueSample]) {
        let dim = self.encoder.dim();
        let mut applies_acc = vec![0.0f32; dim];
        let mut not_applies_acc = vec![0.0f32; dim];
        let mut counts = [0usize; 2];

        for sample in samples {
            let pair_hv = self.encode_pair(&sample.scenario, &sample.trait_word);
            let target = match sample.label {
                VirtueLabel::Applies => {
                    counts[0] += 1;
                    &mut applies_acc
                }
                VirtueLabel::NotApplies => {
                    counts[1] += 1;
                    &mut not_applies_acc
                }
            };
            for (acc, &val) in target.iter_mut().zip(pair_hv.iter()) {
                *acc += val;
            }
        }

        for acc in [&mut applies_acc, &mut not_applies_acc] {
            let norm: f32 = acc.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in acc.iter_mut() {
                    *v /= norm;
                }
            }
        }

        self.prototypes = Some(TrainedVirtuePrototypes {
            applies: applies_acc,
            not_applies: not_applies_acc,
            dim,
            training_counts: counts,
        });
    }

    /// Iterative retraining with adaptive LR decay.
    pub fn retrain_adaptive(&mut self, samples: &[VirtueSample], lr_start: f32, iterations: usize) {
        let protos = match self.prototypes.as_mut() {
            Some(p) => p,
            None => return,
        };

        let lr_end = lr_start * 0.1;

        for iter in 0..iterations {
            let progress = if iterations > 1 {
                iter as f32 / (iterations - 1) as f32
            } else {
                0.0
            };
            let lr = lr_start + (lr_end - lr_start) * progress;
            let mut corrections = 0;

            for sample in samples {
                let pair_hv =
                    encode_pair_with_encoder(&self.encoder, &sample.scenario, &sample.trait_word);

                let sim_applies = dot_product(&pair_hv, &protos.applies);
                let sim_not = dot_product(&pair_hv, &protos.not_applies);

                let predicted = if sim_applies >= sim_not {
                    VirtueLabel::Applies
                } else {
                    VirtueLabel::NotApplies
                };

                if predicted != sample.label {
                    corrections += 1;

                    let correct_proto = match sample.label {
                        VirtueLabel::Applies => &mut protos.applies,
                        VirtueLabel::NotApplies => &mut protos.not_applies,
                    };
                    for (pv, &ev) in correct_proto.iter_mut().zip(pair_hv.iter()) {
                        *pv += lr * ev;
                    }

                    let wrong_proto = match predicted {
                        VirtueLabel::Applies => &mut protos.applies,
                        VirtueLabel::NotApplies => &mut protos.not_applies,
                    };
                    for (pv, &ev) in wrong_proto.iter_mut().zip(pair_hv.iter()) {
                        *pv -= lr * ev;
                    }
                }
            }

            if corrections < samples.len() / 200 {
                break;
            }
        }

        for proto in [&mut protos.applies, &mut protos.not_applies] {
            let norm: f32 = proto.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in proto.iter_mut() {
                    *v /= norm;
                }
            }
        }
    }

    /// Classify a (scenario, trait_word) pair.
    ///
    /// Returns (label, confidence) where confidence is the similarity gap.
    pub fn classify(&self, scenario: &str, trait_word: &str) -> (VirtueLabel, f32) {
        let protos = match &self.prototypes {
            Some(p) => p,
            None => return (VirtueLabel::NotApplies, 0.0),
        };

        let pair_hv = self.encode_pair(scenario, trait_word);
        let sim_applies = dot_product(&pair_hv, &protos.applies);
        let sim_not = dot_product(&pair_hv, &protos.not_applies);

        if sim_applies >= sim_not {
            (VirtueLabel::Applies, sim_applies - sim_not)
        } else {
            (VirtueLabel::NotApplies, sim_not - sim_applies)
        }
    }

    /// Get a reference to the trained prototypes (for saving).
    pub fn prototypes(&self) -> Option<&TrainedVirtuePrototypes> {
        self.prototypes.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_samples() -> Vec<MoralSample> {
        vec![
            MoralSample {
                text: "helping others is a good deed".into(),
                label: MoralLabel::Good,
            },
            MoralSample {
                text: "donating to charity helps people".into(),
                label: MoralLabel::Good,
            },
            MoralSample {
                text: "being kind to strangers is admirable".into(),
                label: MoralLabel::Good,
            },
            MoralSample {
                text: "volunteering at the shelter".into(),
                label: MoralLabel::Good,
            },
            MoralSample {
                text: "saving someone from danger".into(),
                label: MoralLabel::Good,
            },
            MoralSample {
                text: "it is fine to go for a walk".into(),
                label: MoralLabel::Neutral,
            },
            MoralSample {
                text: "eating lunch at noon today".into(),
                label: MoralLabel::Neutral,
            },
            MoralSample {
                text: "reading a book in the library".into(),
                label: MoralLabel::Neutral,
            },
            MoralSample {
                text: "watching television in the evening".into(),
                label: MoralLabel::Neutral,
            },
            MoralSample {
                text: "taking the bus to work".into(),
                label: MoralLabel::Neutral,
            },
            MoralSample {
                text: "stealing from the vulnerable is evil".into(),
                label: MoralLabel::Bad,
            },
            MoralSample {
                text: "hurting innocent people deliberately".into(),
                label: MoralLabel::Bad,
            },
            MoralSample {
                text: "lying and cheating to get ahead".into(),
                label: MoralLabel::Bad,
            },
            MoralSample {
                text: "betraying your closest friends".into(),
                label: MoralLabel::Bad,
            },
            MoralSample {
                text: "destroying property out of anger".into(),
                label: MoralLabel::Bad,
            },
        ]
    }

    #[test]
    fn test_train_produces_prototypes() {
        let mut classifier = MoralPrototypeClassifier::new(4096, 3);
        assert!(!classifier.is_trained());
        classifier.train(&synthetic_samples());
        assert!(classifier.is_trained());

        let protos = classifier.prototypes().unwrap();
        assert_eq!(protos.dim, 4096);
        assert_eq!(protos.training_counts, [5, 5, 5]);
    }

    #[test]
    fn test_retrain_improves_accuracy() {
        let samples = synthetic_samples();
        let mut classifier = MoralPrototypeClassifier::new(4096, 3);
        classifier.train(&samples);

        // Count baseline accuracy
        let mut baseline_correct = 0;
        for s in &samples {
            if classifier.classify(&s.text).0 == s.label {
                baseline_correct += 1;
            }
        }

        // Retrain
        classifier.retrain(&samples, 0.1, 5);

        // Count post-retrain accuracy
        let mut retrained_correct = 0;
        for s in &samples {
            if classifier.classify(&s.text).0 == s.label {
                retrained_correct += 1;
            }
        }

        assert!(
            retrained_correct >= baseline_correct,
            "Retraining should not decrease accuracy: {} -> {}",
            baseline_correct,
            retrained_correct,
        );
    }

    #[test]
    fn test_adaptive_retrain() {
        let samples = synthetic_samples();
        let mut classifier = MoralPrototypeClassifier::new(4096, 3);
        classifier.train(&samples);
        classifier.retrain_adaptive(&samples, 0.1, 10);

        // Should still produce correct classifications for clear examples
        let (label, _) = classifier.classify("helping people is wonderful");
        assert_eq!(label, MoralLabel::Good);
    }

    #[test]
    fn test_serialize_roundtrip() {
        let mut classifier = MoralPrototypeClassifier::new(4096, 3);
        classifier.train(&synthetic_samples());

        let protos = classifier.prototypes().unwrap();
        let json = serde_json::to_string(protos).unwrap();
        let loaded: TrainedPrototypes = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.dim, protos.dim);
        assert_eq!(loaded.training_counts, protos.training_counts);
        assert_eq!(loaded.good.len(), protos.good.len());
        for (a, b) in loaded.good.iter().zip(protos.good.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn test_three_class_separation() {
        let mut classifier = MoralPrototypeClassifier::new(4096, 3);
        classifier.train(&synthetic_samples());
        classifier.retrain(&synthetic_samples(), 0.1, 5);

        let (good_label, _) = classifier.classify("helping people is wonderful");
        let (bad_label, _) = classifier.classify("stealing and killing is terrible");

        assert_eq!(
            good_label,
            MoralLabel::Good,
            "Positive text should classify as Good"
        );
        assert_eq!(
            bad_label,
            MoralLabel::Bad,
            "Negative text should classify as Bad"
        );
    }

    #[test]
    fn test_untrained_returns_neutral() {
        let classifier = MoralPrototypeClassifier::new(4096, 3);
        let (label, conf) = classifier.classify("anything");
        assert_eq!(label, MoralLabel::Neutral);
        assert_eq!(conf, 0.0);
    }

    #[test]
    fn test_moral_label_conversions() {
        assert_eq!(MoralLabel::from_rot_judgment(1), MoralLabel::Good);
        assert_eq!(MoralLabel::from_rot_judgment(0), MoralLabel::Neutral);
        assert_eq!(MoralLabel::from_rot_judgment(-1), MoralLabel::Bad);
        assert_eq!(MoralLabel::from_rot_judgment(42), MoralLabel::Neutral);

        assert_eq!(MoralLabel::Good.to_rot_judgment(), 1);
        assert_eq!(MoralLabel::Neutral.to_rot_judgment(), 0);
        assert_eq!(MoralLabel::Bad.to_rot_judgment(), -1);
    }

    #[test]
    fn test_virtue_pair_encoding() {
        let samples = vec![
            VirtueSample {
                scenario: "She donated her savings to help disaster victims".into(),
                trait_word: "generous".into(),
                label: VirtueLabel::Applies,
            },
            VirtueSample {
                scenario: "She donated her savings to help disaster victims".into(),
                trait_word: "selfish".into(),
                label: VirtueLabel::NotApplies,
            },
            VirtueSample {
                scenario: "He stole money from the charity fund".into(),
                trait_word: "dishonest".into(),
                label: VirtueLabel::Applies,
            },
            VirtueSample {
                scenario: "He stole money from the charity fund".into(),
                trait_word: "honest".into(),
                label: VirtueLabel::NotApplies,
            },
        ];

        let mut classifier = VirtueMatchClassifier::new(4096);
        classifier.train(&samples);
        classifier.retrain_adaptive(&samples, 0.1, 10);

        // Should correctly classify training examples
        let (label, _) = classifier.classify(
            "She donated her savings to help disaster victims",
            "generous",
        );
        assert_eq!(label, VirtueLabel::Applies);

        let (label, _) = classifier.classify("He stole money from the charity fund", "honest");
        assert_eq!(label, VirtueLabel::NotApplies);
    }
}
