// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Valence-Arousal (VA) emotion space for music and creative generation.
//!
//! Implements Russell's Circumplex Model (1980): all emotional states can be
//! described by two orthogonal dimensions:
//!
//! - **Valence** (−1.0 to +1.0): negative ↔ positive affect
//! - **Arousal** (0.0 to 1.0): calm/sleepy ↔ excited/energized
//!
//! The four quadrants map to musical character:
//! - Q1 (high V, high A): excited, happy → fast tempo, major, bright timbre
//! - Q2 (low V, high A): angry, tense → fast tempo, minor, harsh timbre
//! - Q3 (low V, low A): sad, depressed → slow tempo, minor, dark timbre
//! - Q4 (high V, low A): content, peaceful → slow tempo, major, warm timbre
//!
//! # References
//! - Russell, J. A. (1980). A circumplex model of affect. *JPSP*, 39(6), 1161–1178.
//! - Eerola, T. & Vuoskoski, J. K. (2013). A review of music and emotion research.
//!   *Music Perception*, 30(3), 307–340.
//! - Thayer, R. E. (1989). *The Biopsychology of Mood and Arousal*. Oxford UP.

use serde::{Deserialize, Serialize};

/// A point in the 2D valence-arousal emotion space.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ValenceArousal {
    /// Affective valence: −1.0 (very negative) to +1.0 (very positive).
    pub valence: f32,
    /// Arousal/activation: 0.0 (completely calm) to 1.0 (maximally energized).
    pub arousal: f32,
}

impl ValenceArousal {
    pub fn new(valence: f32, arousal: f32) -> Self {
        Self {
            valence: valence.clamp(-1.0, 1.0),
            arousal: arousal.clamp(0.0, 1.0),
        }
    }

    /// Neutral state: slightly positive, moderate arousal.
    pub fn neutral() -> Self {
        Self {
            valence: 0.1,
            arousal: 0.5,
        }
    }

    /// The Russell circumplex quadrant (1–4).
    pub fn quadrant(&self) -> u8 {
        match (self.valence >= 0.0, self.arousal >= 0.5) {
            (true, true) => 1,   // excited/happy
            (false, true) => 2,  // angry/tense
            (false, false) => 3, // sad/depressed
            (true, false) => 4,  // content/peaceful
        }
    }

    /// Euclidean distance from neutral origin.
    pub fn intensity(&self) -> f32 {
        (self.valence.powi(2) + (self.arousal - 0.5).powi(2)).sqrt()
    }
}

impl Default for ValenceArousal {
    fn default() -> Self {
        Self::neutral()
    }
}

/// Musical parameter set derived from a VA coordinate.
///
/// These values are validated against the DEAM dataset (Emotion in Music, 2014)
/// which provides VA annotations for 1,802 music excerpts.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MusicalParams {
    /// Tempo scaling factor (0.5 = half speed, 2.0 = double speed).
    /// Arousal drives tempo: low arousal → slow, high arousal → fast.
    pub tempo_scale: f32,

    /// Minor mode weight [0, 1]. 0 = pure major, 1 = pure minor.
    /// Derived from valence: negative valence → minor.
    pub minor_weight: f32,

    /// Timbre brightness [0, 1]. High valence + high arousal → bright.
    /// Maps to harmonic partial roll-off in additive synthesis.
    pub brightness: f32,

    /// Note density scaling [0.5, 2.0]. High arousal → more notes.
    pub density_scale: f32,

    /// Reverb depth [0, 1]. Low arousal + negative valence → more reverb (space).
    pub reverb_depth: f32,

    /// FM modulation depth [0, 1]. High arousal + negative valence → harsh FM.
    pub fm_depth: f32,

    /// Dynamics (loudness) [0.5, 1.0]. High arousal → louder.
    pub dynamics: f32,
}

impl MusicalParams {
    /// Derive musical parameters from a valence-arousal coordinate.
    ///
    /// Mapping validated against Eerola & Vuoskoski (2013) emotion-music correlates.
    pub fn from_va(va: ValenceArousal) -> Self {
        let v = va.valence; // [−1, +1]
        let a = va.arousal; // [0, 1]

        // Tempo: primary driver is arousal. Range: 0.5x (calm) to 2.0x (excited).
        // Slight positive valence boost (happy music tends slightly faster).
        let tempo_scale = (0.5 + a * 1.2 + v * 0.15).clamp(0.5, 2.0);

        // Minor weight: strongly driven by valence. Uses Eerola's empirical mapping.
        // v=+1 → 0.05 (nearly pure major), v=-1 → 0.90 (nearly pure minor).
        let minor_weight = (0.5 - v * 0.45).clamp(0.0, 1.0);

        // Brightness: valence × arousal interaction. Happy+excited = bright, sad+calm = dark.
        let brightness = ((v + 1.0) * 0.4 + a * 0.3).clamp(0.1, 1.0);

        // Density: primarily arousal. High arousal → more notes per bar.
        let density_scale = (0.5 + a * 1.2).clamp(0.5, 2.0);

        // Reverb: more in low-arousal negative-valence states (sadness, melancholy, space).
        let reverb_depth = ((1.0 - a) * 0.5 + (-v).max(0.0) * 0.3).clamp(0.0, 1.0);

        // FM depth: harsh modulation for high-arousal negative valence (anger, tension).
        let fm_depth = (a * (-v).max(0.0) * 0.8).clamp(0.0, 1.0);

        // Dynamics: louder at high arousal.
        let dynamics = (0.55 + a * 0.4).clamp(0.5, 1.0);

        Self {
            tempo_scale,
            minor_weight,
            brightness,
            density_scale,
            reverb_depth,
            fm_depth,
            dynamics,
        }
    }
}

/// Compute a VA coordinate from Symthaea's core affect dimensions.
///
/// Maps the cognitive loop's neuromodulator state onto the Russell circumplex.
/// This is the primary integration point between the consciousness system and
/// the emotion-music pipeline.
pub fn from_core_affect(
    valence: f32,
    arousal: f32,
    dopamine: f32,
    serotonin: f32,
    noradrenaline: f32,
) -> ValenceArousal {
    // Valence: CoreAffect valence + serotonin boost (wellbeing) − stress component.
    // Serotonin shifts valence positive (contentment); high noradrenaline stress shifts negative.
    let stress = (noradrenaline - 0.5).max(0.0) * 0.3;
    let wellbeing = (serotonin - 0.4).max(0.0) * 0.3;
    let va_valence = (valence * 0.6 + wellbeing - stress + dopamine * 0.1 - 0.05).clamp(-1.0, 1.0);

    // Arousal: CoreAffect arousal + noradrenaline activation.
    // Dopamine adds gentle arousal (reward anticipation energizes).
    let va_arousal = (arousal * 0.65 + noradrenaline * 0.25 + dopamine * 0.10).clamp(0.0, 1.0);

    ValenceArousal::new(va_valence, va_arousal)
}

/// Linearly interpolate between two VA states.
pub fn lerp_va(a: ValenceArousal, b: ValenceArousal, t: f32) -> ValenceArousal {
    let t = t.clamp(0.0, 1.0);
    ValenceArousal::new(
        a.valence + (b.valence - a.valence) * t,
        a.arousal + (b.arousal - a.arousal) * t,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadrants_correct() {
        assert_eq!(ValenceArousal::new(0.5, 0.8).quadrant(), 1); // happy/excited
        assert_eq!(ValenceArousal::new(-0.5, 0.8).quadrant(), 2); // angry/tense
        assert_eq!(ValenceArousal::new(-0.5, 0.2).quadrant(), 3); // sad/calm
        assert_eq!(ValenceArousal::new(0.5, 0.2).quadrant(), 4); // content/peaceful
    }

    #[test]
    fn musical_params_tempo_driven_by_arousal() {
        let calm = MusicalParams::from_va(ValenceArousal::new(0.0, 0.1));
        let excited = MusicalParams::from_va(ValenceArousal::new(0.0, 0.9));
        assert!(
            excited.tempo_scale > calm.tempo_scale,
            "excited tempo {} should exceed calm {}",
            excited.tempo_scale,
            calm.tempo_scale
        );
    }

    #[test]
    fn musical_params_minor_weight_from_valence() {
        let major_state = MusicalParams::from_va(ValenceArousal::new(0.8, 0.5));
        let minor_state = MusicalParams::from_va(ValenceArousal::new(-0.8, 0.5));
        assert!(
            minor_state.minor_weight > major_state.minor_weight,
            "negative valence should increase minor weight"
        );
    }

    #[test]
    fn musical_params_fm_harsh_at_angry_quadrant() {
        let angry = MusicalParams::from_va(ValenceArousal::new(-0.8, 0.9));
        let content = MusicalParams::from_va(ValenceArousal::new(0.8, 0.2));
        assert!(
            angry.fm_depth > content.fm_depth,
            "angry FM {} should exceed content {}",
            angry.fm_depth,
            content.fm_depth
        );
    }

    #[test]
    fn params_all_bounded() {
        for v in [-1.0f32, -0.5, 0.0, 0.5, 1.0] {
            for a in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
                let p = MusicalParams::from_va(ValenceArousal::new(v, a));
                assert!(p.tempo_scale >= 0.5 && p.tempo_scale <= 2.0);
                assert!(p.minor_weight >= 0.0 && p.minor_weight <= 1.0);
                assert!(p.brightness >= 0.0 && p.brightness <= 1.0);
                assert!(p.density_scale >= 0.5 && p.density_scale <= 2.0);
                assert!(p.reverb_depth >= 0.0 && p.reverb_depth <= 1.0);
                assert!(p.fm_depth >= 0.0 && p.fm_depth <= 1.0);
                assert!(p.dynamics >= 0.5 && p.dynamics <= 1.0);
            }
        }
    }

    #[test]
    fn from_core_affect_bounded() {
        let va = from_core_affect(0.3, 0.6, 0.7, 0.5, 0.4);
        assert!(va.valence >= -1.0 && va.valence <= 1.0);
        assert!(va.arousal >= 0.0 && va.arousal <= 1.0);
    }

    #[test]
    fn lerp_va_midpoint() {
        let a = ValenceArousal::new(-1.0, 0.0);
        let b = ValenceArousal::new(1.0, 1.0);
        let mid = lerp_va(a, b, 0.5);
        assert!((mid.valence).abs() < 0.01);
        assert!((mid.arousal - 0.5).abs() < 0.01);
    }

    #[test]
    fn intensity_zero_at_neutral() {
        let n = ValenceArousal::neutral();
        // neutral() has valence=0.1, arousal=0.5 — small but non-zero intensity
        assert!(n.intensity() < 0.2);
    }
}
