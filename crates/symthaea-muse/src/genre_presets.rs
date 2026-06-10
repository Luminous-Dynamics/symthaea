// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Genre presets: seed consciousness states for different musical styles.
//!
//! Derived from Tristan's Spotify analysis (5,972 liked songs).
//! Each preset configures the consciousness parameters to produce
//! music in a specific style.

use crate::MusicalState;

/// Available genre presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Genre {
    /// Dream Theater, Animals As Leaders: complex meters, high energy, virtuosic
    ProgMetal,
    /// Esperanza Spalding, Makaya McCraven: extended chords, groove, space
    Jazz,
    /// Pink Floyd, Steven Wilson: atmospheric, long builds, emotional depth
    ProgressiveAtmospheric,
    /// Vulfpeck, RHCP, Khruangbin: pocket, funk, laid-back groove
    FunkGroove,
    /// Shida Shahabi, Maria Chiara Argirò: textured, contemplative, minimal
    AmbientContemplative,
    /// Led Zeppelin, Hendrix: blues-rock, dynamic, raw
    ClassicRock,
    /// Pretty Lights, Tame Impala, Polo & Pan: electronic, psychedelic
    ElectronicPsych,
    /// Bob Marley, Jack Johnson: reggae-influenced, warm, relaxed
    IslandGroove,
    /// Bach, Beethoven, Chopin: formal structure, piano/strings, wide dynamics
    Classical,
}

impl Genre {
    /// Get the seed consciousness state for this genre.
    pub fn seed_state(&self) -> MusicalState {
        match self {
            Self::ProgMetal => MusicalState {
                consciousness_level: 0.85, // high complexity
                arousal: 0.75,             // energetic
                valence: 0.2,              // slightly dark
                dopamine: 0.7,             // bright, technical
                serotonin: 0.2,            // not laid-back
                noradrenaline: 0.6,        // edge, intensity
                prediction_error: 0.6,     // surprising, complex
                harmony_activations: [
                    0.4, 0.3, 0.5, 0.3, // moderate coherence, high creative
                    0.5, 0.4, 0.8, 0.1, // high evolutionary progression
                ],
            },
            Self::Jazz => MusicalState {
                consciousness_level: 0.7,
                arousal: 0.4,
                valence: 0.3,
                dopamine: 0.6,  // colorful harmony
                serotonin: 0.6, // warm, relaxed
                noradrenaline: 0.2,
                prediction_error: 0.4, // moderate surprise
                harmony_activations: [
                    0.6, 0.5, 0.4, 0.6, // high coherence, infinite play
                    0.4, 0.3, 0.5, 0.3,
                ],
            },
            Self::ProgressiveAtmospheric => MusicalState {
                consciousness_level: 0.6,
                arousal: 0.25, // slow build
                valence: 0.1,  // neutral to dark
                dopamine: 0.3,
                serotonin: 0.7, // warm, atmospheric
                noradrenaline: 0.15,
                prediction_error: 0.2, // gradual surprise
                harmony_activations: [
                    0.5, 0.3, 0.3, 0.2, 0.3, 0.4, 0.4, 0.5, // moderate stillness
                ],
            },
            Self::FunkGroove => MusicalState {
                consciousness_level: 0.5,
                arousal: 0.6,   // energetic but not aggressive
                valence: 0.5,   // positive
                dopamine: 0.8,  // bright, funky
                serotonin: 0.5, // warm groove
                noradrenaline: 0.2,
                prediction_error: 0.3, // pocket, not chaotic
                harmony_activations: [
                    0.7, 0.3, 0.3, 0.7, // high coherence + infinite play
                    0.4, 0.3, 0.3, 0.1,
                ],
            },
            Self::AmbientContemplative => MusicalState {
                consciousness_level: 0.4,
                arousal: 0.1,  // very calm
                valence: 0.15, // slightly melancholy
                dopamine: 0.15,
                serotonin: 0.8, // very warm
                noradrenaline: 0.05,
                prediction_error: 0.1, // minimal surprise
                harmony_activations: [
                    0.3, 0.2, 0.2, 0.1, 0.2, 0.3, 0.2, 0.8, // high stillness
                ],
            },
            Self::ClassicRock => MusicalState {
                consciousness_level: 0.55,
                arousal: 0.65,
                valence: 0.35,
                dopamine: 0.5,
                serotonin: 0.4,
                noradrenaline: 0.5, // raw energy
                prediction_error: 0.35,
                harmony_activations: [0.5, 0.4, 0.4, 0.3, 0.5, 0.4, 0.5, 0.15],
            },
            Self::ElectronicPsych => MusicalState {
                consciousness_level: 0.6,
                arousal: 0.5,
                valence: 0.4, // positive, dreamy
                dopamine: 0.6,
                serotonin: 0.5,
                noradrenaline: 0.3,
                prediction_error: 0.45, // psychedelic surprise
                harmony_activations: [
                    0.4, 0.5, 0.5, 0.5, // explorative
                    0.3, 0.4, 0.4, 0.3,
                ],
            },
            Self::Classical => MusicalState {
                consciousness_level: 0.65,
                arousal: 0.35,   // moderate, controlled
                valence: 0.15,   // slightly melancholy (minor key feel)
                dopamine: 0.3,   // not flashy
                serotonin: 0.55, // warm
                noradrenaline: 0.15,
                prediction_error: 0.25, // structured, some surprise
                harmony_activations: [
                    0.6, 0.4, 0.3, 0.2, // good coherence
                    0.3, 0.4, 0.5, 0.3, // moderate progression
                ],
            },
            Self::IslandGroove => MusicalState {
                consciousness_level: 0.4,
                arousal: 0.35,
                valence: 0.6, // warm, positive
                dopamine: 0.4,
                serotonin: 0.75, // very laid-back
                noradrenaline: 0.1,
                prediction_error: 0.15, // predictable, comforting
                harmony_activations: [
                    0.6, 0.3, 0.2, 0.5, // coherent + playful
                    0.3, 0.2, 0.2, 0.4,
                ],
            },
        }
    }

    /// Suggested tempo range for this genre.
    pub fn tempo_range(&self) -> (f32, f32) {
        match self {
            Self::ProgMetal => (120.0, 180.0),
            Self::Jazz => (80.0, 140.0),
            Self::ProgressiveAtmospheric => (60.0, 90.0),
            Self::FunkGroove => (90.0, 120.0),
            Self::AmbientContemplative => (55.0, 75.0),
            Self::ClassicRock => (100.0, 150.0),
            Self::ElectronicPsych => (100.0, 130.0),
            Self::Classical => (72.0, 120.0),
            Self::IslandGroove => (70.0, 95.0),
        }
    }

    /// Suggested duration in seconds.
    pub fn duration_secs(&self) -> f32 {
        match self {
            Self::ProgMetal => 360.0,              // 6 min (prog suites)
            Self::Jazz => 300.0,                   // 5 min
            Self::ProgressiveAtmospheric => 480.0, // 8 min (long builds)
            Self::FunkGroove => 240.0,             // 4 min
            Self::AmbientContemplative => 420.0,   // 7 min
            Self::ClassicRock => 270.0,            // 4.5 min
            Self::ElectronicPsych => 360.0,        // 6 min
            Self::Classical => 300.0,              // 5 min
            Self::IslandGroove => 240.0,           // 4 min
        }
    }

    pub fn all() -> &'static [Genre] {
        &[
            Self::ProgMetal,
            Self::Jazz,
            Self::ProgressiveAtmospheric,
            Self::FunkGroove,
            Self::AmbientContemplative,
            Self::ClassicRock,
            Self::ElectronicPsych,
            Self::IslandGroove,
            Self::Classical,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::ProgMetal => "Prog Metal",
            Self::Jazz => "Jazz",
            Self::ProgressiveAtmospheric => "Progressive Atmospheric",
            Self::FunkGroove => "Funk Groove",
            Self::AmbientContemplative => "Ambient Contemplative",
            Self::ClassicRock => "Classic Rock",
            Self::ElectronicPsych => "Electronic Psychedelic",
            Self::Classical => "Classical",
            Self::IslandGroove => "Island Groove",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_presets_valid() {
        for genre in Genre::all() {
            let state = genre.seed_state();
            assert!(state.consciousness_level >= 0.0 && state.consciousness_level <= 1.0);
            assert!(state.arousal >= 0.0 && state.arousal <= 1.0);
            let (lo, hi) = genre.tempo_range();
            assert!(lo < hi);
            assert!(genre.duration_secs() > 60.0);
        }
    }

    #[test]
    fn prog_metal_is_complex() {
        let state = Genre::ProgMetal.seed_state();
        assert!(state.consciousness_level > 0.7, "prog should be complex");
        assert!(state.prediction_error > 0.5, "prog should be surprising");
    }

    #[test]
    fn ambient_is_calm() {
        let state = Genre::AmbientContemplative.seed_state();
        assert!(state.arousal < 0.2, "ambient should be calm");
        assert!(
            state.harmony_activations[7] > 0.6,
            "ambient should have stillness"
        );
    }

    #[test]
    fn funk_has_groove() {
        let state = Genre::FunkGroove.seed_state();
        assert!(state.dopamine > 0.7, "funk should be bright/funky");
        assert!(state.prediction_error < 0.4, "funk should be in the pocket");
    }
}
