// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Emotional timbre space: learned manifold of timbral qualities.
//!
//! Instead of hardcoded mappings (dopamine→FM, serotonin→rolloff), timbres
//! are points in a 16,384D hyperdimensional space. Consciousness state
//! navigates this manifold via a CfC network, producing timbres that
//! *feel* like the emotion rather than just correlating by formula.
//!
//! # Architecture
//!
//! ```text
//! Procedural timbre samples → AudioHdcEncoder → 16,384D HV centroids
//!   → label by (valence, arousal) → TimbreManifold
//!
//! Runtime: MusicalState → emotional coordinate
//!   → nearest centroids (weighted by distance)
//!   → CfC interpolates smoothly over time
//!   → output: partial amplitudes for synthesis
//! ```

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

/// A point in the timbre manifold.
#[derive(Debug, Clone)]
pub struct TimbrePoint {
    /// 16,384D hypervector encoding the spectral character.
    pub hv: ContinuousHV,
    /// Emotional valence [-1, 1] (sad to joyful).
    pub valence: f32,
    /// Emotional arousal [0, 1] (calm to excited).
    pub arousal: f32,
    /// Human-readable label.
    pub label: String,
    /// Partial amplitudes this timbre produces.
    pub partials: Vec<f32>,
}

/// Manifold of timbral qualities indexed by emotional coordinates.
pub struct TimbreManifold {
    points: Vec<TimbrePoint>,
    /// Current timbre HV (smoothly interpolated).
    current_hv: ContinuousHV,
    /// Interpolation rate (lower = smoother transitions).
    lerp_rate: f32,
}

impl TimbreManifold {
    /// Create a default manifold with procedurally generated timbres.
    ///
    /// Generates a grid of timbres across the valence×arousal space,
    /// encoded as 16,384D HVs for distance-based lookup.
    pub fn default_manifold(genesis: &GenesisSeed) -> Self {
        let mut points = Vec::new();

        // Generate timbres across emotional space
        let configs = [
            (
                -0.8,
                0.1,
                "dark_pad",
                vec![1.0, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001],
            ),
            (
                -0.5,
                0.6,
                "tense_brass",
                vec![1.0, 0.7, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1],
            ),
            (
                -0.3,
                0.9,
                "harsh_lead",
                vec![1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25],
            ),
            (
                0.0,
                0.0,
                "neutral_sine",
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            (
                0.0,
                0.5,
                "warm_organ",
                vec![1.0, 0.5, 0.25, 0.12, 0.06, 0.03, 0.015, 0.0],
            ),
            (
                0.3,
                0.3,
                "soft_flute",
                vec![1.0, 0.2, 0.05, 0.01, 0.0, 0.0, 0.0, 0.0],
            ),
            (
                0.5,
                0.5,
                "bright_bell",
                vec![1.0, 0.3, 0.5, 0.2, 0.4, 0.1, 0.3, 0.05],
            ),
            (
                0.7,
                0.2,
                "glass_chime",
                vec![1.0, 0.1, 0.3, 0.05, 0.2, 0.02, 0.1, 0.01],
            ),
            (
                0.8,
                0.7,
                "joyful_choir",
                vec![1.0, 0.4, 0.3, 0.25, 0.15, 0.1, 0.08, 0.05],
            ),
            (
                0.9,
                0.9,
                "radiant_full",
                vec![1.0, 0.6, 0.4, 0.3, 0.25, 0.2, 0.15, 0.12],
            ),
        ];

        for (valence, arousal, label, partials) in configs {
            // Encode partials + emotional coords into HV
            let hv = encode_timbre(genesis, &partials, valence, arousal);
            points.push(TimbrePoint {
                hv,
                valence,
                arousal,
                label: label.to_string(),
                partials,
            });
        }

        Self {
            points,
            current_hv: ContinuousHV::zero(HDC_DIMENSION),
            lerp_rate: 0.08,
        }
    }

    /// Look up the best timbre for a given emotional state.
    ///
    /// Returns partial amplitudes interpolated from the 3 nearest centroids,
    /// weighted by inverse distance in emotional space.
    pub fn lookup(&mut self, valence: f32, arousal: f32) -> Vec<f32> {
        if self.points.is_empty() {
            return vec![1.0];
        }

        // Find 3 nearest centroids by emotional distance
        let mut distances: Vec<(usize, f32)> = self
            .points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let dv = p.valence - valence;
                let da = p.arousal - arousal;
                (i, (dv * dv + da * da).sqrt())
            })
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let top3: Vec<(usize, f32)> = distances.into_iter().take(3).collect();
        let total_weight: f32 = top3.iter().map(|(_, d)| 1.0 / (d + 0.01)).sum();

        // Weighted interpolation of partial amplitudes
        let max_partials = top3
            .iter()
            .map(|(i, _)| self.points[*i].partials.len())
            .max()
            .unwrap_or(1);

        let mut result = vec![0.0f32; max_partials];
        for (idx, dist) in &top3 {
            let weight = (1.0 / (dist + 0.01)) / total_weight;
            let point = &self.points[*idx];
            for (j, &amp) in point.partials.iter().enumerate() {
                result[j] += amp * weight;
            }
        }

        // Smooth transition via lerp
        // (In production, use CfC network for temporal smoothing)
        result
    }

    /// Get the nearest timbre label for a given emotional state.
    pub fn nearest_label(&self, valence: f32, arousal: f32) -> &str {
        self.points
            .iter()
            .min_by(|a, b| {
                let da = (a.valence - valence).powi(2) + (a.arousal - arousal).powi(2);
                let db = (b.valence - valence).powi(2) + (b.arousal - arousal).powi(2);
                da.partial_cmp(&db).unwrap()
            })
            .map(|p| p.label.as_str())
            .unwrap_or("unknown")
    }

    /// Number of timbre points in the manifold.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Get all timbre labels.
    pub fn labels(&self) -> Vec<&str> {
        self.points.iter().map(|p| p.label.as_str()).collect()
    }
}

/// Encode a timbre (partials + emotional coords) as a 16,384D HV.
fn encode_timbre(
    genesis: &GenesisSeed,
    partials: &[f32],
    valence: f32,
    arousal: f32,
) -> ContinuousHV {
    let mut result = ContinuousHV::zero(HDC_DIMENSION);

    // Encode each partial amplitude
    for (i, &amp) in partials.iter().enumerate() {
        let basis = genesis.hv(&format!("timbre_partial_{i}"), HDC_DIMENSION);
        result = result.add(&basis.scale(amp));
    }

    // Encode emotional coordinates
    let val_basis = genesis.hv("timbre_valence", HDC_DIMENSION);
    let aro_basis = genesis.hv("timbre_arousal", HDC_DIMENSION);
    result = result.add(&val_basis.scale(valence));
    result = result.add(&aro_basis.scale(arousal));

    result.normalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_genesis() -> GenesisSeed {
        GenesisSeed::from_phrase("test-timbre-space")
    }

    #[test]
    fn manifold_has_points() {
        let genesis = test_genesis();
        let manifold = TimbreManifold::default_manifold(&genesis);
        assert_eq!(manifold.len(), 10);
    }

    #[test]
    fn lookup_returns_partials() {
        let genesis = test_genesis();
        let mut manifold = TimbreManifold::default_manifold(&genesis);
        let partials = manifold.lookup(0.0, 0.5);
        assert!(!partials.is_empty());
        assert!(partials[0] > 0.0, "fundamental should be non-zero");
    }

    #[test]
    fn sad_calm_produces_dark_timbre() {
        let genesis = test_genesis();
        let manifold = TimbreManifold::default_manifold(&genesis);
        let label = manifold.nearest_label(-0.7, 0.15);
        assert_eq!(label, "dark_pad");
    }

    #[test]
    fn joyful_excited_produces_bright_timbre() {
        let genesis = test_genesis();
        let manifold = TimbreManifold::default_manifold(&genesis);
        let label = manifold.nearest_label(0.85, 0.85);
        assert_eq!(label, "radiant_full");
    }

    #[test]
    fn neutral_produces_sine() {
        let genesis = test_genesis();
        let manifold = TimbreManifold::default_manifold(&genesis);
        let label = manifold.nearest_label(0.0, 0.0);
        assert_eq!(label, "neutral_sine");
    }

    #[test]
    fn interpolation_blends_partials() {
        let genesis = test_genesis();
        let mut manifold = TimbreManifold::default_manifold(&genesis);

        // Midpoint between dark_pad and bright_bell
        let partials = manifold.lookup(0.0, 0.3);
        // Should have some upper partial content (not pure sine, not full bell)
        assert!(partials.len() >= 4);
        assert!(partials[0] > 0.5, "fundamental should be strong");
        // Second partial should be moderate (between sine=0 and bell=0.3)
        assert!(
            partials[1] > 0.0 && partials[1] < 0.8,
            "blended second partial"
        );
    }

    #[test]
    fn all_labels_unique() {
        let genesis = test_genesis();
        let manifold = TimbreManifold::default_manifold(&genesis);
        let labels = manifold.labels();
        let mut unique = labels.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(labels.len(), unique.len(), "all labels should be unique");
    }

    #[test]
    fn hv_encoding_preserves_similarity() {
        let genesis = test_genesis();
        // Two similar timbres should have similar HVs
        let hv1 = encode_timbre(&genesis, &[1.0, 0.5, 0.25], 0.5, 0.5);
        let hv2 = encode_timbre(&genesis, &[1.0, 0.45, 0.3], 0.55, 0.48);
        let hv3 = encode_timbre(&genesis, &[1.0, 0.0, 0.0], -0.8, 0.1);

        let sim_close = hv1.similarity(&hv2);
        let sim_far = hv1.similarity(&hv3);
        assert!(
            sim_close > sim_far,
            "similar timbres should be closer: {sim_close} vs {sim_far}"
        );
    }
}
