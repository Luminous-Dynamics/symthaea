// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-modal synesthesia: bind visual and audio features via HDC.
//!
//! Maps visual properties (color, shape, texture) to audio parameters
//! (timbre, rhythm, harmony) using hyperdimensional binding operations.
//! Each visual-audio mapping is a quasi-orthogonal HDC binding that
//! preserves cross-modal similarity.
//!
//! # Mappings
//!
//! | Visual | Audio | Basis |
//! |--------|-------|-------|
//! | Hue (0-360°) | Pitch class (C-B) | Chromatic circle ↔ color wheel |
//! | Saturation | Timbre brightness | Vivid ↔ bright partials |
//! | Lightness | Dynamics/velocity | Light ↔ loud, dark ↔ soft |
//! | Shape (angular) | Rhythm (sharp onset) | Jagged ↔ staccato |
//! | Shape (round) | Rhythm (smooth) | Curved ↔ legato |
//! | Texture density | Harmonic density | Dense ↔ many partials |

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

/// Visual features for synesthetic mapping.
#[derive(Debug, Clone, Copy, Default)]
pub struct VisualFeatures {
    /// Hue [0, 1] (0=red, 0.33=green, 0.67=blue).
    pub hue: f32,
    /// Saturation [0, 1] (0=grey, 1=vivid).
    pub saturation: f32,
    /// Lightness [0, 1] (0=black, 1=white).
    pub lightness: f32,
    /// Angular shape factor [0, 1] (0=round, 1=jagged).
    pub angularity: f32,
    /// Texture density [0, 1] (0=sparse, 1=dense).
    pub density: f32,
    /// Motion speed [0, 1] (0=still, 1=fast).
    pub motion: f32,
}

/// Audio parameters derived from visual features.
#[derive(Debug, Clone, Copy)]
pub struct SynestheticAudioParams {
    /// Pitch class [0, 11] (C=0, C#=1, ..., B=11).
    pub pitch_class: usize,
    /// Brightness [0, 1] (timbre upper partial emphasis).
    pub brightness: f32,
    /// Velocity [0, 1] (dynamics from lightness).
    pub velocity: f32,
    /// Staccato factor [0, 1] (note shortening from angularity).
    pub staccato: f32,
    /// Harmonic density [0, 1] (number of active partials).
    pub harmonic_density: f32,
    /// Tempo scale [0.5, 2.0] (motion → speed).
    pub tempo_scale: f32,
}

/// Synesthetic encoder: binds visual features to audio HV via HDC operations.
pub struct SynestheticEncoder {
    hue_basis: Vec<ContinuousHV>, // 12 pitch-class basis HVs
    sat_basis: ContinuousHV,
    light_basis: ContinuousHV,
    angular_basis: ContinuousHV,
    density_basis: ContinuousHV,
    motion_basis: ContinuousHV,
}

impl SynestheticEncoder {
    /// Create encoder with genesis-seeded basis vectors.
    pub fn new(genesis: &GenesisSeed) -> Self {
        let hue_basis = (0..12)
            .map(|i| genesis.hv(&format!("synesthesia_hue_{i}"), HDC_DIMENSION))
            .collect();

        Self {
            hue_basis,
            sat_basis: genesis.hv("synesthesia_saturation", HDC_DIMENSION),
            light_basis: genesis.hv("synesthesia_lightness", HDC_DIMENSION),
            angular_basis: genesis.hv("synesthesia_angularity", HDC_DIMENSION),
            density_basis: genesis.hv("synesthesia_density", HDC_DIMENSION),
            motion_basis: genesis.hv("synesthesia_motion", HDC_DIMENSION),
        }
    }

    /// Encode visual features as a 16,384D HV.
    ///
    /// The resulting HV can be compared to audio HVs for cross-modal similarity.
    pub fn encode_visual(&self, vis: &VisualFeatures) -> ContinuousHV {
        // Hue → weighted blend of pitch-class basis vectors
        let hue_idx = ((vis.hue * 12.0) as usize).min(11);
        let hue_frac = vis.hue * 12.0 - hue_idx as f32;
        let next_idx = (hue_idx + 1) % 12;
        let hue_hv = self.hue_basis[hue_idx]
            .scale(1.0 - hue_frac)
            .add(&self.hue_basis[next_idx].scale(hue_frac));

        // Bind all features
        let mut result = hue_hv;
        result = result.add(&self.sat_basis.scale(vis.saturation));
        result = result.add(&self.light_basis.scale(vis.lightness));
        result = result.add(&self.angular_basis.scale(vis.angularity));
        result = result.add(&self.density_basis.scale(vis.density));
        result = result.add(&self.motion_basis.scale(vis.motion));

        result.normalize()
    }

    /// Map visual features to audio parameters (direct mapping, no HV).
    pub fn visual_to_audio(&self, vis: &VisualFeatures) -> SynestheticAudioParams {
        SynestheticAudioParams {
            // Hue → pitch class (color wheel = chromatic circle)
            pitch_class: ((vis.hue * 12.0) as usize).min(11),
            // Saturation → brightness (vivid = bright timbre)
            brightness: vis.saturation,
            // Lightness → velocity (light = loud, dark = soft)
            velocity: vis.lightness * 0.7 + 0.15,
            // Angularity → staccato (jagged = short notes)
            staccato: vis.angularity * 0.8,
            // Density → harmonic density
            harmonic_density: vis.density,
            // Motion → tempo scale
            tempo_scale: 0.5 + vis.motion * 1.5,
        }
    }

    /// Cross-modal similarity: how "similar" does a visual feature
    /// sound to an audio feature? Returns cosine similarity [-1, 1].
    pub fn cross_modal_similarity(&self, visual: &VisualFeatures, audio_hv: &ContinuousHV) -> f32 {
        let visual_hv = self.encode_visual(visual);
        visual_hv.similarity(audio_hv)
    }
}

/// Apply synesthetic modulation to a MusicalState.
pub fn apply_synesthetic_modulation(
    params: &SynestheticAudioParams,
    state: &mut crate::MusicalState,
    strength: f32,
) {
    let s = strength.clamp(0.0, 1.0);

    // Brightness → dopamine (timbre control)
    state.dopamine += s * 0.1 * (params.brightness - state.dopamine);
    // Velocity → consciousness level (dynamics)
    state.consciousness_level += s * 0.05 * (params.velocity - state.consciousness_level);
    // Staccato → arousal (note shortening = more energetic)
    state.arousal += s * 0.08 * (params.staccato - state.arousal);
    // Tempo scale → noradrenaline
    let ne_target = (params.tempo_scale - 1.0).clamp(0.0, 1.0);
    state.noradrenaline += s * 0.06 * (ne_target - state.noradrenaline);

    // Clamp
    state.dopamine = state.dopamine.clamp(0.0, 1.0);
    state.consciousness_level = state.consciousness_level.clamp(0.0, 1.0);
    state.arousal = state.arousal.clamp(0.0, 1.0);
    state.noradrenaline = state.noradrenaline.clamp(0.0, 1.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_genesis() -> GenesisSeed {
        GenesisSeed::from_phrase("test-synesthesia")
    }

    #[test]
    fn hue_maps_to_pitch_class() {
        let enc = SynestheticEncoder::new(&test_genesis());

        // Red (hue=0) → C (pitch_class=0)
        let red = VisualFeatures {
            hue: 0.0,
            ..Default::default()
        };
        assert_eq!(enc.visual_to_audio(&red).pitch_class, 0);

        // Blue (hue=0.67) → Ab (pitch_class=8)
        let blue = VisualFeatures {
            hue: 0.67,
            ..Default::default()
        };
        assert_eq!(enc.visual_to_audio(&blue).pitch_class, 8);
    }

    #[test]
    fn saturation_maps_to_brightness() {
        let enc = SynestheticEncoder::new(&test_genesis());
        let grey = VisualFeatures {
            saturation: 0.0,
            ..Default::default()
        };
        let vivid = VisualFeatures {
            saturation: 1.0,
            ..Default::default()
        };
        assert!(enc.visual_to_audio(&vivid).brightness > enc.visual_to_audio(&grey).brightness);
    }

    #[test]
    fn similar_colors_similar_hvs() {
        let enc = SynestheticEncoder::new(&test_genesis());
        let red1 = VisualFeatures {
            hue: 0.0,
            saturation: 0.5,
            lightness: 0.5,
            ..Default::default()
        };
        let red2 = VisualFeatures {
            hue: 0.02,
            saturation: 0.55,
            lightness: 0.48,
            ..Default::default()
        };
        let blue = VisualFeatures {
            hue: 0.67,
            saturation: 0.5,
            lightness: 0.5,
            ..Default::default()
        };

        let hv_r1 = enc.encode_visual(&red1);
        let hv_r2 = enc.encode_visual(&red2);
        let hv_b = enc.encode_visual(&blue);

        let sim_close = hv_r1.similarity(&hv_r2);
        let sim_far = hv_r1.similarity(&hv_b);
        assert!(
            sim_close > sim_far,
            "similar colors should be closer: {sim_close} vs {sim_far}"
        );
    }

    #[test]
    fn cross_modal_similarity_works() {
        let genesis = test_genesis();
        let enc = SynestheticEncoder::new(&genesis);
        let vis = VisualFeatures {
            hue: 0.0,
            saturation: 0.5,
            lightness: 0.5,
            ..Default::default()
        };
        let audio_hv = genesis.hv("some_audio", HDC_DIMENSION);
        let sim = enc.cross_modal_similarity(&vis, &audio_hv);
        assert!(sim.is_finite());
        assert!(sim >= -1.0 && sim <= 1.0);
    }

    #[test]
    fn apply_modulation_stays_bounded() {
        let params = SynestheticAudioParams {
            pitch_class: 5,
            brightness: 1.0,
            velocity: 1.0,
            staccato: 1.0,
            harmonic_density: 1.0,
            tempo_scale: 2.0,
        };
        let mut state = crate::MusicalState::default();
        apply_synesthetic_modulation(&params, &mut state, 1.0);
        assert!(state.dopamine <= 1.0 && state.dopamine >= 0.0);
        assert!(state.arousal <= 1.0 && state.arousal >= 0.0);
    }
}
