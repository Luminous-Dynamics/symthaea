// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Taste space: continuous multi-dimensional representation of musical taste.
//!
//! Any listener is a point in a 10-dimensional space. The 7 named profiles
//! (Tristan, Classical, Metal, Jazz, Ambient, Pop, Universal) are fixed
//! landmarks. New profiles can be:
//! - Derived from Spotify data
//! - Interpolated between landmarks ("70% Jazz + 30% Metal")
//! - Positioned anywhere in the continuous space
//!
//! The taste space enables:
//! - Generating music for any taste, not just predefined genres
//! - Blending tastes for group listening
//! - Exploring the space of musical possibility
//! - Training the evolutionary tuner on diverse targets

use crate::taste_bench::TasteProfile;

/// A point in taste space — 10 continuous dimensions.
#[derive(Debug, Clone)]
pub struct TastePoint {
    /// How much repetition is desired [0=none, 1=lots].
    pub repetition_affinity: f32,
    /// How stepwise vs leaping [0=all leaps, 1=all steps].
    pub stepwise_preference: f32,
    /// How much harmonic complexity (thirds, chord tones) [0=simple, 1=rich].
    pub harmonic_richness: f32,
    /// Desired dynamic range [0=compressed, 1=extreme].
    pub dynamic_preference: f32,
    /// Preferred register height [0=bass, 0.5=mid, 1=treble].
    pub register_height: f32,
    /// Preferred pitch range width [0=narrow, 1=wide].
    pub range_width: f32,
    /// Tempo preference [0=slow, 1=fast].
    pub tempo_preference: f32,
    /// Rhythmic complexity [0=simple 4/4, 1=complex odd meters].
    pub rhythmic_complexity: f32,
    /// Timbral brightness [0=dark/warm, 1=bright/harsh].
    pub brightness: f32,
    /// Emotional intensity [0=gentle, 1=intense].
    pub intensity: f32,
}

impl TastePoint {
    /// Convert to a TasteProfile for benchmark scoring.
    pub fn to_profile(&self) -> TasteProfile {
        TasteProfile {
            target_repeated_pct: 5.0 + self.repetition_affinity * 25.0, // 5-30%
            target_steps_pct: 30.0 + self.stepwise_preference * 35.0,   // 30-65%
            target_thirds_pct: 8.0 + self.harmonic_richness * 15.0,     // 8-23%
            target_leaps_pct: 5.0 + (1.0 - self.stepwise_preference) * 20.0, // 5-25%
            target_ascending_pct: 50.0,                                 // always balanced
            min_unique_pitches: (10.0 + self.range_width * 20.0) as usize, // 10-30
            max_top_note_pct: 8.0 + (1.0 - self.range_width) * 15.0,    // 8-23%
            target_pitch_range: 8.0 + self.range_width * 16.0,          // 8-24 semitones
            target_mean_midi: 52.0 + self.register_height * 24.0,       // 52-76 (E3-E5)
            min_velocity_range: 20.0 + self.dynamic_preference * 80.0,  // 20-100
        }
    }

    /// Named landmarks in taste space.
    pub fn tristan() -> Self {
        Self {
            repetition_affinity: 0.15,
            stepwise_preference: 0.65,
            harmonic_richness: 0.55,
            dynamic_preference: 0.75,
            register_height: 0.55,
            range_width: 0.45,
            tempo_preference: 0.55,
            rhythmic_complexity: 0.75,
            brightness: 0.45,
            intensity: 0.65,
        }
    }

    pub fn classical() -> Self {
        Self {
            repetition_affinity: 0.10,
            stepwise_preference: 0.75,
            harmonic_richness: 0.65,
            dynamic_preference: 0.90,
            register_height: 0.50,
            range_width: 0.60,
            tempo_preference: 0.45,
            rhythmic_complexity: 0.30,
            brightness: 0.40,
            intensity: 0.50,
        }
    }

    pub fn metal() -> Self {
        Self {
            repetition_affinity: 0.20,
            stepwise_preference: 0.35,
            harmonic_richness: 0.30,
            dynamic_preference: 0.40,
            register_height: 0.25,
            range_width: 0.65,
            tempo_preference: 0.80,
            rhythmic_complexity: 0.70,
            brightness: 0.75,
            intensity: 0.90,
        }
    }

    pub fn jazz() -> Self {
        Self {
            repetition_affinity: 0.10,
            stepwise_preference: 0.55,
            harmonic_richness: 0.85,
            dynamic_preference: 0.70,
            register_height: 0.60,
            range_width: 0.55,
            tempo_preference: 0.50,
            rhythmic_complexity: 0.60,
            brightness: 0.50,
            intensity: 0.45,
        }
    }

    pub fn ambient() -> Self {
        Self {
            repetition_affinity: 0.70,
            stepwise_preference: 0.70,
            harmonic_richness: 0.20,
            dynamic_preference: 0.25,
            register_height: 0.45,
            range_width: 0.25,
            tempo_preference: 0.15,
            rhythmic_complexity: 0.10,
            brightness: 0.30,
            intensity: 0.15,
        }
    }

    pub fn pop() -> Self {
        Self {
            repetition_affinity: 0.45,
            stepwise_preference: 0.70,
            harmonic_richness: 0.35,
            dynamic_preference: 0.30,
            register_height: 0.55,
            range_width: 0.30,
            tempo_preference: 0.55,
            rhythmic_complexity: 0.15,
            brightness: 0.55,
            intensity: 0.50,
        }
    }

    pub fn hiphop() -> Self {
        Self {
            repetition_affinity: 0.55,
            stepwise_preference: 0.45,
            harmonic_richness: 0.40,
            dynamic_preference: 0.35,
            register_height: 0.35,
            range_width: 0.35,
            tempo_preference: 0.50,
            rhythmic_complexity: 0.45,
            brightness: 0.50,
            intensity: 0.60,
        }
    }

    pub fn edm() -> Self {
        Self {
            repetition_affinity: 0.65,
            stepwise_preference: 0.50,
            harmonic_richness: 0.30,
            dynamic_preference: 0.25,
            register_height: 0.50,
            range_width: 0.40,
            tempo_preference: 0.75,
            rhythmic_complexity: 0.35,
            brightness: 0.70,
            intensity: 0.80,
        }
    }

    /// Interpolate between two taste points.
    pub fn blend(a: &TastePoint, b: &TastePoint, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let s = 1.0 - t;
        Self {
            repetition_affinity: a.repetition_affinity * s + b.repetition_affinity * t,
            stepwise_preference: a.stepwise_preference * s + b.stepwise_preference * t,
            harmonic_richness: a.harmonic_richness * s + b.harmonic_richness * t,
            dynamic_preference: a.dynamic_preference * s + b.dynamic_preference * t,
            register_height: a.register_height * s + b.register_height * t,
            range_width: a.range_width * s + b.range_width * t,
            tempo_preference: a.tempo_preference * s + b.tempo_preference * t,
            rhythmic_complexity: a.rhythmic_complexity * s + b.rhythmic_complexity * t,
            brightness: a.brightness * s + b.brightness * t,
            intensity: a.intensity * s + b.intensity * t,
        }
    }

    /// Blend multiple taste points with weights.
    pub fn blend_multi(points: &[(&TastePoint, f32)]) -> Self {
        let total_weight: f32 = points.iter().map(|(_, w)| w).sum();
        if total_weight < 0.001 {
            return Self::tristan();
        }

        let mut result = Self {
            repetition_affinity: 0.0,
            stepwise_preference: 0.0,
            harmonic_richness: 0.0,
            dynamic_preference: 0.0,
            register_height: 0.0,
            range_width: 0.0,
            tempo_preference: 0.0,
            rhythmic_complexity: 0.0,
            brightness: 0.0,
            intensity: 0.0,
        };

        for (point, weight) in points {
            let w = weight / total_weight;
            result.repetition_affinity += point.repetition_affinity * w;
            result.stepwise_preference += point.stepwise_preference * w;
            result.harmonic_richness += point.harmonic_richness * w;
            result.dynamic_preference += point.dynamic_preference * w;
            result.register_height += point.register_height * w;
            result.range_width += point.range_width * w;
            result.tempo_preference += point.tempo_preference * w;
            result.rhythmic_complexity += point.rhythmic_complexity * w;
            result.brightness += point.brightness * w;
            result.intensity += point.intensity * w;
        }
        result
    }

    /// Distance between two taste points (Euclidean).
    pub fn distance(&self, other: &TastePoint) -> f32 {
        let d = [
            self.repetition_affinity - other.repetition_affinity,
            self.stepwise_preference - other.stepwise_preference,
            self.harmonic_richness - other.harmonic_richness,
            self.dynamic_preference - other.dynamic_preference,
            self.register_height - other.register_height,
            self.range_width - other.range_width,
            self.tempo_preference - other.tempo_preference,
            self.rhythmic_complexity - other.rhythmic_complexity,
            self.brightness - other.brightness,
            self.intensity - other.intensity,
        ];
        d.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// All named landmarks.
    pub fn landmarks() -> Vec<(&'static str, TastePoint)> {
        vec![
            ("Tristan", Self::tristan()),
            ("Classical", Self::classical()),
            ("Metal", Self::metal()),
            ("Jazz", Self::jazz()),
            ("Ambient", Self::ambient()),
            ("Pop", Self::pop()),
            ("Hip-Hop", Self::hiphop()),
            ("EDM", Self::edm()),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_midpoint() {
        let a = TastePoint::classical();
        let b = TastePoint::metal();
        let mid = TastePoint::blend(&a, &b, 0.5);
        // Midpoint should be between both
        assert!(mid.intensity > a.intensity && mid.intensity < b.intensity);
    }

    #[test]
    fn blend_endpoints() {
        let a = TastePoint::jazz();
        let b = TastePoint::ambient();
        let at_a = TastePoint::blend(&a, &b, 0.0);
        assert!((at_a.harmonic_richness - a.harmonic_richness).abs() < 0.01);
    }

    #[test]
    fn distance_self_is_zero() {
        let p = TastePoint::tristan();
        assert!(p.distance(&p) < 0.001);
    }

    #[test]
    fn distance_different_nonzero() {
        let a = TastePoint::metal();
        let b = TastePoint::ambient();
        assert!(
            a.distance(&b) > 0.5,
            "metal and ambient should be far apart"
        );
    }

    #[test]
    fn to_profile_reasonable() {
        let p = TastePoint::tristan().to_profile();
        assert!(p.target_steps_pct > 30.0 && p.target_steps_pct < 70.0);
        assert!(p.target_repeated_pct > 0.0 && p.target_repeated_pct < 40.0);
    }

    #[test]
    fn blend_multi_weighted() {
        let jazz = TastePoint::jazz();
        let metal = TastePoint::metal();
        // 70% jazz, 30% metal
        let blend = TastePoint::blend_multi(&[(&jazz, 0.7), (&metal, 0.3)]);
        // Should be closer to jazz
        assert!(blend.distance(&jazz) < blend.distance(&metal));
    }
}
