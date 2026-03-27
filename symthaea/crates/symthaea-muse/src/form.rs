// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Song form planning: temporal structure driven by the Eight Harmonies.
//!
//! Maps cognitive state to a sequence of sections (verse, chorus, bridge, etc.)
//! that tile the composition duration. The Eight Harmonies determine the form:
//!
//! | Dominant Harmony | Form | Character |
//! |------------------|------|-----------|
//! | EvolutionaryProgression | Verse-Chorus arc | Building narrative |
//! | SacredStillness | Through-composed | Continuous flow |
//! | InfinitePlay | Rondo (ABACAD) | Playful variation |
//! | Default | Binary (AB) | Balanced contrast |

use crate::structure::SectionType;
use crate::MusicalState;
use serde::{Deserialize, Serialize};

/// A section within a song form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Section {
    /// Section type (determines density, phrasing).
    pub section_type: SectionType,
    /// Start time in seconds.
    pub start_time: f32,
    /// Duration in seconds.
    pub duration: f32,
    /// Key shift in semitones from root (0 = no shift).
    pub key_shift: i32,
    /// Energy level [0, 1] — drives voice count and tempo modulation.
    pub energy_level: f32,
}

/// A complete song form: tiled sequence of sections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SongForm {
    /// Ordered sections tiling the full duration.
    pub sections: Vec<Section>,
    /// Total duration in seconds.
    pub total_duration: f32,
}

/// Plan a song form from cognitive state and total duration.
///
/// The form is determined by the dominant harmony activation. Sections tile
/// the duration exactly with no gaps or overlaps.
pub fn plan_form(state: &MusicalState, duration_secs: f32) -> SongForm {
    let progression = state.harmony_activations[6];
    let stillness = state.harmony_activations[7];
    let play = state.harmony_activations[3];

    let sections = if progression > 0.5 && progression >= stillness && progression >= play {
        // Verse-Chorus arc: A-B-A-B-C-B (narrative with refrain)
        plan_verse_chorus(state, duration_secs)
    } else if stillness > 0.5 && stillness >= play {
        // Through-composed: single flowing section (no repetition)
        plan_through_composed(state, duration_secs)
    } else if play > 0.5 {
        // Rondo: A-B-A-C-A (playful returns with variations)
        plan_rondo(state, duration_secs)
    } else {
        // Binary: A-B (balanced contrast)
        plan_binary(state, duration_secs)
    };

    SongForm {
        sections,
        total_duration: duration_secs,
    }
}

/// Verse-Chorus arc: developmental → climactic → developmental → climactic → exploratory → climactic
fn plan_verse_chorus(state: &MusicalState, dur: f32) -> Vec<Section> {
    let n = 6;
    let sec_dur = dur / n as f32;
    let types = [
        (SectionType::Developmental, 0, 0.4),  // verse
        (SectionType::Climactic, 0, 0.7),       // chorus
        (SectionType::Developmental, 2, 0.5),   // verse 2 (key shift)
        (SectionType::Climactic, 0, 0.8),       // chorus
        (SectionType::Exploratory, 5, 0.6),     // bridge
        (SectionType::Climactic, 0, 1.0),       // final chorus
    ];

    types
        .iter()
        .enumerate()
        .map(|(i, &(st, key, energy))| Section {
            section_type: st,
            start_time: i as f32 * sec_dur,
            duration: sec_dur,
            key_shift: key,
            energy_level: energy * state.arousal.max(0.3),
        })
        .collect()
}

/// Through-composed: ambient flowing texture (SacredStillness).
fn plan_through_composed(state: &MusicalState, dur: f32) -> Vec<Section> {
    // Gradual energy arc: low → mid → low
    let n = 3;
    let sec_dur = dur / n as f32;
    let energies = [0.2, 0.4, 0.2];

    (0..n)
        .map(|i| Section {
            section_type: SectionType::Ambient,
            start_time: i as f32 * sec_dur,
            duration: sec_dur,
            key_shift: 0,
            energy_level: energies[i] * state.consciousness_level.max(0.2),
        })
        .collect()
}

/// Rondo: A-B-A-C-A (InfinitePlay)
fn plan_rondo(state: &MusicalState, dur: f32) -> Vec<Section> {
    let n = 5;
    let sec_dur = dur / n as f32;
    let pattern = [
        (SectionType::Developmental, 0, 0.5),   // A
        (SectionType::Exploratory, 3, 0.6),      // B
        (SectionType::Developmental, 0, 0.5),    // A
        (SectionType::Exploratory, 5, 0.7),      // C
        (SectionType::Developmental, 0, 0.6),    // A
    ];

    pattern
        .iter()
        .enumerate()
        .map(|(i, &(st, key, energy))| Section {
            section_type: st,
            start_time: i as f32 * sec_dur,
            duration: sec_dur,
            key_shift: key,
            energy_level: energy * state.arousal.max(0.3),
        })
        .collect()
}

/// Binary: A-B (balanced contrast).
fn plan_binary(state: &MusicalState, dur: f32) -> Vec<Section> {
    let half = dur / 2.0;
    vec![
        Section {
            section_type: SectionType::Developmental,
            start_time: 0.0,
            duration: half,
            key_shift: 0,
            energy_level: 0.4 * state.arousal.max(0.3),
        },
        Section {
            section_type: SectionType::Climactic,
            start_time: half,
            duration: half,
            key_shift: 0,
            energy_level: 0.7 * state.arousal.max(0.3),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sections_tile_duration() {
        let state = MusicalState::default();
        let form = plan_form(&state, 10.0);
        let total: f32 = form.sections.iter().map(|s| s.duration).sum();
        assert!(
            (total - 10.0).abs() < 0.01,
            "sections should tile exactly: sum={total}"
        );
    }

    #[test]
    fn sections_no_gaps() {
        let state = MusicalState {
            harmony_activations: [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.8, 0.3],
            ..Default::default()
        };
        let form = plan_form(&state, 12.0);
        for w in form.sections.windows(2) {
            let end = w[0].start_time + w[0].duration;
            assert!(
                (end - w[1].start_time).abs() < 0.01,
                "gap between sections at t={end}"
            );
        }
    }

    #[test]
    fn progression_gives_verse_chorus() {
        let state = MusicalState {
            harmony_activations: [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.8, 0.1],
            arousal: 0.5,
            ..Default::default()
        };
        let form = plan_form(&state, 12.0);
        assert_eq!(form.sections.len(), 6, "verse-chorus should have 6 sections");
        assert!(
            form.sections.iter().filter(|s| s.section_type == SectionType::Climactic).count() >= 3,
            "should have at least 3 chorus sections"
        );
    }

    #[test]
    fn stillness_gives_through_composed() {
        let state = MusicalState {
            harmony_activations: [0.3, 0.3, 0.3, 0.1, 0.3, 0.3, 0.1, 0.8],
            ..Default::default()
        };
        let form = plan_form(&state, 10.0);
        assert!(
            form.sections.iter().all(|s| s.section_type == SectionType::Ambient),
            "through-composed should be all ambient"
        );
    }

    #[test]
    fn play_gives_rondo() {
        let state = MusicalState {
            harmony_activations: [0.3, 0.3, 0.3, 0.8, 0.3, 0.3, 0.1, 0.1],
            ..Default::default()
        };
        let form = plan_form(&state, 10.0);
        assert_eq!(form.sections.len(), 5, "rondo should have 5 sections (ABACA)");
    }

    #[test]
    fn default_gives_binary() {
        let state = MusicalState {
            harmony_activations: [0.3; 8],
            ..Default::default()
        };
        let form = plan_form(&state, 8.0);
        assert_eq!(form.sections.len(), 2, "binary form should have 2 sections");
    }

    #[test]
    fn energy_scales_with_arousal() {
        let low = MusicalState {
            arousal: 0.3,
            harmony_activations: [0.3; 8],
            ..Default::default()
        };
        let high = MusicalState {
            arousal: 0.9,
            harmony_activations: [0.3; 8],
            ..Default::default()
        };
        let form_low = plan_form(&low, 8.0);
        let form_high = plan_form(&high, 8.0);
        let avg_low: f32 =
            form_low.sections.iter().map(|s| s.energy_level).sum::<f32>() / form_low.sections.len() as f32;
        let avg_high: f32 =
            form_high.sections.iter().map(|s| s.energy_level).sum::<f32>() / form_high.sections.len() as f32;
        assert!(
            avg_high > avg_low,
            "high arousal ({avg_high}) should have more energy than low ({avg_low})"
        );
    }
}
