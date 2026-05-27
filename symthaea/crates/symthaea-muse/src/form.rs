// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Song form planning from the Eight Harmonies.
//!
//! Maps cognitive state to large-scale musical form (verse-chorus, rondo,
//! through-composed, binary) based on which harmonies are most active.

use serde::{Deserialize, Serialize};

use crate::MusicalState;
use crate::structure::SectionType;

/// Index of InfinitePlay in the 8-harmony activation array.
pub const HARMONY_PLAY: usize = 3;
/// Index of EvolutionaryProgression.
pub const HARMONY_PROGRESSION: usize = 6;
/// Index of SacredStillness.
pub const HARMONY_STILLNESS: usize = 7;

/// A section within a song form, with timing and structural metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Section {
    /// Musical section type (determines density and character).
    pub section_type: SectionType,
    /// Start time in seconds from the beginning of the piece.
    pub start_time: f32,
    /// Duration of this section in seconds.
    pub duration: f32,
    /// Key transposition in semitones from the base key.
    pub key_shift: i32,
    /// Energy level [0, 1] controlling dynamics and density.
    pub energy_level: f32,
}

/// A complete song form: an ordered list of sections tiling the piece.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SongForm {
    /// Ordered sections of the piece.
    pub sections: Vec<Section>,
    /// Total duration of the piece in seconds.
    pub total_duration: f32,
}

/// Plan a song form from cognitive state and desired duration.
///
/// The dominant harmony determines the overall form:
/// - Progression > 0.5: verse-chorus (6 sections)
/// - Stillness > 0.5: through-composed (3 ambient sections)
/// - Play > 0.5: rondo ABACA (5 sections)
/// - Default: binary AB (2 sections)
pub fn plan_form(state: &MusicalState, duration_secs: f32) -> SongForm {
    let progression = state.harmony_activations[HARMONY_PROGRESSION];
    let stillness = state.harmony_activations[HARMONY_STILLNESS];
    let play = state.harmony_activations[HARMONY_PLAY];
    let energy = state.arousal.max(0.3);

    let templates: Vec<(SectionType, i32, f32)> = if progression > 0.5 {
        // Verse-chorus: Dev-Climactic-Dev-Climactic-Exploratory-Climactic
        vec![
            (SectionType::Developmental, 0, energy * 0.7),
            (SectionType::Climactic, 0, energy),
            (SectionType::Developmental, 2, energy * 0.7),
            (SectionType::Climactic, 2, energy),
            (SectionType::Exploratory, -3, energy * 0.5),
            (SectionType::Climactic, 0, energy),
        ]
    } else if stillness > 0.5 {
        // Through-composed: 3 ambient sections
        vec![
            (SectionType::Ambient, 0, energy * 0.4),
            (SectionType::Ambient, -2, energy * 0.3),
            (SectionType::Ambient, 0, energy * 0.5),
        ]
    } else if play > 0.5 {
        // Rondo ABACA
        vec![
            (SectionType::Developmental, 0, energy * 0.8),
            (SectionType::Exploratory, 3, energy * 0.6),
            (SectionType::Developmental, 0, energy * 0.8),
            (SectionType::Exploratory, -2, energy * 0.6),
            (SectionType::Developmental, 0, energy * 0.9),
        ]
    } else {
        // Binary AB
        vec![
            (SectionType::Developmental, 0, energy * 0.7),
            (SectionType::Climactic, 0, energy),
        ]
    };

    let n = templates.len();
    let section_dur = duration_secs / n as f32;

    let sections = templates
        .into_iter()
        .enumerate()
        .map(|(i, (section_type, key_shift, energy_level))| Section {
            section_type,
            start_time: i as f32 * section_dur,
            duration: section_dur,
            key_shift,
            energy_level,
        })
        .collect();

    SongForm {
        sections,
        total_duration: duration_secs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sections_tile_exactly() {
        let state = MusicalState::default();
        let form = plan_form(&state, 10.0);
        let total: f32 = form.sections.iter().map(|s| s.duration).sum();
        assert!((total - 10.0).abs() < 1e-4, "total {total} != 10.0");
    }

    #[test]
    fn no_gaps_between_sections() {
        let state = MusicalState {
            harmony_activations: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0],
            ..Default::default()
        };
        let form = plan_form(&state, 12.0);
        for i in 1..form.sections.len() {
            let expected = form.sections[i - 1].start_time + form.sections[i - 1].duration;
            let actual = form.sections[i].start_time;
            assert!(
                (expected - actual).abs() < 1e-4,
                "gap between section {} and {}: {} vs {}",
                i - 1,
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    fn progression_produces_verse_chorus() {
        let state = MusicalState {
            harmony_activations: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0],
            ..Default::default()
        };
        let form = plan_form(&state, 12.0);
        assert_eq!(
            form.sections.len(),
            6,
            "verse-chorus should have 6 sections"
        );
    }

    #[test]
    fn stillness_produces_ambient() {
        let state = MusicalState {
            harmony_activations: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
            ..Default::default()
        };
        let form = plan_form(&state, 9.0);
        assert_eq!(form.sections.len(), 3);
        assert!(
            form.sections
                .iter()
                .all(|s| s.section_type == SectionType::Ambient)
        );
    }

    #[test]
    fn play_produces_rondo() {
        let state = MusicalState {
            harmony_activations: [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        };
        let form = plan_form(&state, 10.0);
        assert_eq!(form.sections.len(), 5, "rondo should have 5 sections");
    }

    #[test]
    fn default_produces_binary() {
        let state = MusicalState::default();
        let form = plan_form(&state, 8.0);
        assert_eq!(form.sections.len(), 2, "default should produce binary AB");
    }

    #[test]
    fn energy_scales_with_arousal() {
        let low = MusicalState {
            arousal: 0.1,
            ..Default::default()
        };
        let high = MusicalState {
            arousal: 0.9,
            ..Default::default()
        };
        let form_low = plan_form(&low, 4.0);
        let form_high = plan_form(&high, 4.0);
        // Both should produce binary; high arousal → higher energy
        let max_low = form_low
            .sections
            .iter()
            .map(|s| s.energy_level)
            .fold(0.0f32, f32::max);
        let max_high = form_high
            .sections
            .iter()
            .map(|s| s.energy_level)
            .fold(0.0f32, f32::max);
        assert!(
            max_high > max_low,
            "high arousal energy {max_high} should exceed low {max_low}"
        );
    }
}
