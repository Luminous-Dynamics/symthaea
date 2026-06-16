// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Higher-level musical form from the creativity system.
//!
//! Creative mode affects musical structure:
//! - Divergent: explore new melodic motifs
//! - Convergent: repeat and develop the best motif
//! - Incubation: ambient drone / sparse texture

use serde::{Deserialize, Serialize};

use crate::MusicalState;

/// Musical section type, derived from creative mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SectionType {
    /// Exploratory: many short varied phrases.
    Exploratory,
    /// Developmental: repeat and vary a motif.
    Developmental,
    /// Ambient: sparse, drone-like texture.
    Ambient,
    /// Climactic: building intensity.
    Climactic,
}

/// Determine the musical section type from cognitive state.
pub fn determine_section(state: &MusicalState) -> SectionType {
    let psi = state.consciousness_level;
    let arousal = state.arousal;
    let stillness = state.harmony_activations[7];
    let play = state.harmony_activations[3];
    let progression = state.harmony_activations[6];

    if stillness > 0.6 && arousal < 0.3 {
        SectionType::Ambient
    } else if play > 0.6 || (psi > 0.5 && state.prediction_error > 0.3) {
        SectionType::Exploratory
    } else if progression > 0.6 && arousal > 0.5 {
        SectionType::Climactic
    } else {
        SectionType::Developmental
    }
}

/// Compute the phrase length (in notes) for a section type.
pub fn phrase_length(section: SectionType, consciousness: f32) -> usize {
    let base = match section {
        SectionType::Exploratory => 4,
        SectionType::Developmental => 8,
        SectionType::Ambient => 2,
        SectionType::Climactic => 6,
    };

    // Higher consciousness → longer phrases
    let extension = (consciousness * 4.0) as usize;
    base + extension
}

/// Compute the note density multiplier for a section type.
/// Higher density = more notes per beat.
pub fn density_multiplier(section: SectionType) -> f32 {
    match section {
        SectionType::Exploratory => 1.5,
        SectionType::Developmental => 1.0,
        SectionType::Ambient => 0.3,
        SectionType::Climactic => 2.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stillness_produces_ambient() {
        let state = MusicalState {
            harmony_activations: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
            arousal: 0.1,
            ..Default::default()
        };
        assert_eq!(determine_section(&state), SectionType::Ambient);
    }

    #[test]
    fn playful_produces_exploratory() {
        let state = MusicalState {
            harmony_activations: [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        };
        assert_eq!(determine_section(&state), SectionType::Exploratory);
    }

    #[test]
    fn progression_arousal_produces_climactic() {
        let state = MusicalState {
            harmony_activations: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0],
            arousal: 0.7,
            ..Default::default()
        };
        assert_eq!(determine_section(&state), SectionType::Climactic);
    }

    #[test]
    fn phrase_length_scales_with_consciousness() {
        let short = phrase_length(SectionType::Developmental, 0.1);
        let long = phrase_length(SectionType::Developmental, 0.9);
        assert!(long > short);
    }

    #[test]
    fn ambient_low_density() {
        assert!(density_multiplier(SectionType::Ambient) < 1.0);
    }
}
