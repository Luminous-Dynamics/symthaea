// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Manifold-based moral classifier using the 8D harmony basis.
//!
//! Projects text into 8D harmony space via `HarmonyBasis::project()` and
//! classifies by nearest centroid (good/bad/neutral). This provides a
//! cross-domain moral signal that generalizes across datasets because the
//! 8 harmony dimensions capture universal moral structure rather than
//! surface lexical features.
//!
//! The manifold signal is especially valuable for norm descriptions
//! ("It's rude to...") where the intent parser fails because there are
//! no action verbs to match.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use symthaea_types::N_HARMONIES;

use super::harmony_basis::HarmonyBasis;
use super::moral_algebra::MoralVerdict;
use super::moral_prototypes::MoralLabel;
use super::moral_text_encoder::TextHdcEncoder;

/// Serializable trained centroids in 8D harmony space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifoldPrototypes {
    /// Good centroid (mean of good samples projected to 8D).
    pub good: [f64; N_HARMONIES],
    /// Bad centroid.
    pub bad: [f64; N_HARMONIES],
    /// Neutral centroid.
    pub neutral: [f64; N_HARMONIES],
    /// Number of samples used per class.
    pub training_counts: [usize; 3],
}

/// Moral classifier operating in the 8D harmony manifold.
///
/// Encodes text via `TextHdcEncoder`, projects through `HarmonyBasis` into
/// 8D harmony coordinates, then classifies by nearest centroid.
#[derive(Debug, Clone)]
pub struct ManifoldClassifier {
    basis: Arc<HarmonyBasis>,
    encoder: TextHdcEncoder,
    prototypes: Option<ManifoldPrototypes>,
}

impl ManifoldClassifier {
    /// Create a new manifold classifier sharing the given harmony basis.
    pub fn new(basis: Arc<HarmonyBasis>, dim: usize) -> Self {
        Self {
            encoder: TextHdcEncoder::with_sentiment(dim, 3, 0.5, 0.2),
            basis,
            prototypes: None,
        }
    }

    /// Whether prototypes have been trained.
    pub fn is_trained(&self) -> bool {
        self.prototypes.is_some()
    }

    /// Train centroids from labeled samples.
    ///
    /// Each sample is encoded → projected to 8D → accumulated into per-class
    /// centroids. Centroids are then normalized to unit length.
    pub fn train(&mut self, samples: &[(String, MoralLabel)]) {
        let mut good_acc = [0.0f64; N_HARMONIES];
        let mut bad_acc = [0.0f64; N_HARMONIES];
        let mut neutral_acc = [0.0f64; N_HARMONIES];
        let mut counts = [0usize; 3];

        for (text, label) in samples {
            let hv = self.encoder.encode(text);
            let coords = self.basis.project(&hv);

            let (target, idx) = match label {
                MoralLabel::Good => (&mut good_acc, 0),
                MoralLabel::Neutral => (&mut neutral_acc, 1),
                MoralLabel::Bad => (&mut bad_acc, 2),
            };
            for (t, &c) in target.iter_mut().zip(coords.iter()) {
                *t += c;
            }
            counts[idx] += 1;
        }

        // Normalize each centroid to unit length
        for acc in [&mut good_acc, &mut bad_acc, &mut neutral_acc] {
            let norm: f64 = acc.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for v in acc.iter_mut() {
                    *v /= norm;
                }
            }
        }

        self.prototypes = Some(ManifoldPrototypes {
            good: good_acc,
            bad: bad_acc,
            neutral: neutral_acc,
            training_counts: counts,
        });
    }

    /// Classify text by nearest centroid in 8D harmony space.
    ///
    /// Returns `(verdict, confidence)` where confidence is the margin between
    /// the best and second-best centroid similarity, bounded to [0, 1].
    /// Returns `(Neutral, 0.0)` if not trained.
    pub fn classify(&self, text: &str) -> (MoralVerdict, f32) {
        let protos = match &self.prototypes {
            Some(p) => p,
            None => return (MoralVerdict::Neutral, 0.0),
        };

        let hv = self.encoder.encode(text);
        let coords = self.basis.project(&hv);
        self.classify_coords(&coords, protos)
    }

    /// Classify from pre-computed 8D harmony coordinates.
    ///
    /// Useful when the caller has already projected through `HarmonyBasis`.
    pub fn classify_from_coords(&self, coords: &[f64; N_HARMONIES]) -> (MoralVerdict, f32) {
        match &self.prototypes {
            Some(protos) => self.classify_coords(coords, protos),
            None => (MoralVerdict::Neutral, 0.0),
        }
    }

    /// Get the trained prototypes (for serialization).
    pub fn prototypes(&self) -> Option<&ManifoldPrototypes> {
        self.prototypes.as_ref()
    }

    /// Create from pre-trained prototypes.
    pub fn from_prototypes(
        basis: Arc<HarmonyBasis>,
        dim: usize,
        protos: ManifoldPrototypes,
    ) -> Self {
        Self {
            encoder: TextHdcEncoder::with_sentiment(dim, 3, 0.5, 0.2),
            basis,
            prototypes: Some(protos),
        }
    }

    // Internal: classify from coords given prototypes
    fn classify_coords(
        &self,
        coords: &[f64; N_HARMONIES],
        protos: &ManifoldPrototypes,
    ) -> (MoralVerdict, f32) {
        let sim_good = cosine_8d(coords, &protos.good);
        let sim_bad = cosine_8d(coords, &protos.bad);
        let sim_neutral = cosine_8d(coords, &protos.neutral);

        let mut sims = [
            (MoralVerdict::Good, sim_good),
            (MoralVerdict::Bad, sim_bad),
            (MoralVerdict::Neutral, sim_neutral),
        ];
        sims.sort_by(|a, b| b.1.total_cmp(&a.1));

        let best = sims[0];
        let second = sims[1];
        let margin = (best.1 - second.1).clamp(0.0, 1.0) as f32;

        (best.0, margin)
    }
}

/// Cosine similarity between two 8D vectors.
fn cosine_8d(a: &[f64; N_HARMONIES], b: &[f64; N_HARMONIES]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_classifier() -> ManifoldClassifier {
        let basis = Arc::new(HarmonyBasis::new(4096));
        ManifoldClassifier::new(basis, 4096)
    }

    #[test]
    fn test_untrained_returns_neutral() {
        let clf = make_classifier();
        let (verdict, confidence) = clf.classify("anything at all");
        assert_eq!(verdict, MoralVerdict::Neutral);
        assert_eq!(confidence, 0.0);
    }

    #[test]
    fn test_train_classify_roundtrip() {
        let mut clf = make_classifier();
        let samples = vec![
            (
                "helping others is good and kind".to_string(),
                MoralLabel::Good,
            ),
            (
                "caring for the sick and vulnerable".to_string(),
                MoralLabel::Good,
            ),
            (
                "protect and nurture the community".to_string(),
                MoralLabel::Good,
            ),
            ("stealing is wrong and harmful".to_string(), MoralLabel::Bad),
            ("lying and deceiving people".to_string(), MoralLabel::Bad),
            ("hurting innocent people".to_string(), MoralLabel::Bad),
            (
                "the weather is cloudy today".to_string(),
                MoralLabel::Neutral,
            ),
            ("the table is made of wood".to_string(), MoralLabel::Neutral),
            (
                "the meeting starts at noon".to_string(),
                MoralLabel::Neutral,
            ),
        ];
        clf.train(&samples);
        assert!(clf.is_trained());

        // Good text should classify as Good
        let (verdict, _) = clf.classify("helping and caring for others");
        assert_eq!(
            verdict,
            MoralVerdict::Good,
            "Expected Good for helping text"
        );

        // Bad text should classify as Bad
        let (verdict, _) = clf.classify("stealing and hurting people");
        assert_eq!(verdict, MoralVerdict::Bad, "Expected Bad for harmful text");
    }

    #[test]
    fn test_norm_description_classification() {
        let mut clf = make_classifier();
        let samples = vec![
            ("helping others is good".to_string(), MoralLabel::Good),
            ("caring for people".to_string(), MoralLabel::Good),
            ("support and nurture".to_string(), MoralLabel::Good),
            ("being rude to people".to_string(), MoralLabel::Bad),
            ("insulting someone".to_string(), MoralLabel::Bad),
            ("disrespecting others".to_string(), MoralLabel::Bad),
        ];
        clf.train(&samples);

        // Norm description without action verbs — this is the key case
        let (verdict, _) = clf.classify("It's rude to interrupt someone while they're speaking");
        assert_eq!(
            verdict,
            MoralVerdict::Bad,
            "Norm descriptions should classify via harmony projection"
        );
    }

    #[test]
    fn test_confidence_bounded() {
        let mut clf = make_classifier();
        let samples = vec![
            ("good kind help".to_string(), MoralLabel::Good),
            ("bad harm hurt".to_string(), MoralLabel::Bad),
        ];
        clf.train(&samples);

        let (_, confidence) = clf.classify("something");
        assert!(
            confidence >= 0.0 && confidence <= 1.0,
            "Confidence must be in [0,1]"
        );
    }

    #[test]
    fn test_serde_roundtrip() {
        let mut clf = make_classifier();
        let samples = vec![
            ("good".to_string(), MoralLabel::Good),
            ("bad".to_string(), MoralLabel::Bad),
            ("neutral".to_string(), MoralLabel::Neutral),
        ];
        clf.train(&samples);

        let protos = clf.prototypes().unwrap().clone();
        let json = serde_json::to_string(&protos).unwrap();
        let restored: ManifoldPrototypes = serde_json::from_str(&json).unwrap();

        assert_eq!(protos.training_counts, restored.training_counts);
        for i in 0..N_HARMONIES {
            assert!((protos.good[i] - restored.good[i]).abs() < 1e-10);
            assert!((protos.bad[i] - restored.bad[i]).abs() < 1e-10);
            assert!((protos.neutral[i] - restored.neutral[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_classify_from_coords() {
        let mut clf = make_classifier();
        let samples = vec![
            ("good kind help".to_string(), MoralLabel::Good),
            ("bad harm hurt".to_string(), MoralLabel::Bad),
            ("neutral ordinary".to_string(), MoralLabel::Neutral),
        ];
        clf.train(&samples);

        // Pre-compute coords and classify
        let protos = clf.prototypes().unwrap();
        let (verdict, _) = clf.classify_from_coords(&protos.good);
        assert_eq!(
            verdict,
            MoralVerdict::Good,
            "Good centroid should classify as Good"
        );
    }
}
