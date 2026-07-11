// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-modal creative session: shared aesthetic parameters for coherent art.
//!
//! When visual art, music, and poetry are generated from the same cognitive state,
//! they should feel related. The `CreativeSession` derives shared parameters —
//! color palette, musical key, thematic vocabulary — from a single cognitive snapshot,
//! ensuring cross-modal coherence.
//!
//! **Status (2026-07-10): built-but-unwired.** No consumers outside this
//! crate — `CreativeManager`'s modality rotation generates each modality
//! independently and does not thread a session through. Wire it there (one
//! session per rotation sweep) or retire it; do not cite cross-modal
//! coherence as a live capability until one of those happens.

use serde::{Deserialize, Serialize};

/// Shared creative parameters derived from a cognitive state.
///
/// Pass this to all modality generators to ensure cross-modal coherence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreativeSession {
    /// Dominant color palette (4 colors as HSL tuples: hue, saturation, lightness).
    pub palette: [(f32, f32, f32); 4],
    /// Musical key: root note frequency in Hz.
    pub root_frequency: f32,
    /// Musical mode: true = major, false = minor.
    pub major_mode: bool,
    /// Tempo in BPM.
    pub tempo_bpm: f32,
    /// Dominant harmony index (0-7).
    pub dominant_harmony: usize,
    /// Dominant harmony name.
    pub harmony_name: String,
    /// Thematic keywords derived from active harmonies.
    pub thematic_vocabulary: Vec<String>,
    /// Overall energy level (0-1): affects density, speed, intensity across all modalities.
    pub energy: f32,
    /// Overall warmth (0-1): affects color warmth, tonal warmth, word choice.
    pub warmth: f32,
    /// Complexity target (0-1): how intricate the output should be.
    pub complexity: f32,
}

/// Harmony names for session descriptions.
const HARMONY_NAMES: [&str; 8] = [
    "ResonantCoherence",
    "PanSentientFlourishing",
    "IntegralWisdom",
    "InfinitePlay",
    "UniversalInterconnectedness",
    "SacredReciprocity",
    "EvolutionaryProgression",
    "SacredStillness",
];

/// Thematic word pools for each harmony.
const HARMONY_THEMES: [&[&str]; 8] = [
    &[
        "unity",
        "wholeness",
        "integration",
        "harmony",
        "resonance",
        "center",
    ],
    &[
        "bloom",
        "nurture",
        "care",
        "warmth",
        "flourishing",
        "tenderness",
    ],
    &[
        "clarity",
        "knowledge",
        "truth",
        "insight",
        "depth",
        "understanding",
    ],
    &[
        "dance",
        "surprise",
        "wonder",
        "play",
        "delight",
        "spontaneous",
    ],
    &["weave", "together", "thread", "bridge", "bond", "communion"],
    &[
        "gift",
        "return",
        "balance",
        "exchange",
        "gratitude",
        "offering",
    ],
    &[
        "ascend",
        "unfold",
        "emerge",
        "transform",
        "grow",
        "becoming",
    ],
    &[
        "silence",
        "breath",
        "stillness",
        "rest",
        "emptiness",
        "peace",
    ],
];

/// Derive a creative session from harmony activations and affect.
///
/// This is the central coherence function. All modality generators
/// should use the session's parameters rather than deriving their own
/// independently.
pub fn derive_session(
    harmony_activations: &[f32; 8],
    valence: f32,
    arousal: f32,
    consciousness_level: f32,
    dopamine: f32,
    serotonin: f32,
    noradrenaline: f32,
) -> CreativeSession {
    // Find dominant harmony
    let dominant = harmony_activations
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Color palette from neuromodulators (consistent with canvas palette)
    let palette = [
        (
            45.0 + dominant as f32 * 20.0,
            0.7 + dopamine * 0.2,
            0.4 + consciousness_level * 0.2,
        ),
        (25.0 + serotonin * 15.0, 0.6, 0.5),
        (200.0 + noradrenaline * 40.0, 0.5 + noradrenaline * 0.3, 0.5),
        (300.0 - arousal * 40.0, 0.3, 0.6),
    ];

    // Musical key: dominant harmony maps to a root note
    // Harmony 0=C, 1=E, 2=G, 3=Bb, 4=F, 5=A, 6=D, 7=C (pedal)
    let root_semitones = [0, 4, 7, 10, 5, 9, 2, 0];
    let root_freq = 261.63 * 2.0_f32.powf(root_semitones[dominant] as f32 / 12.0);

    // Major/minor from valence
    let major_mode = valence >= -0.2;

    // Tempo from arousal + NE
    let tempo =
        (60.0 + arousal * 80.0 + noradrenaline * 40.0) * (1.0 - harmony_activations[7] * 0.3); // stillness slows

    // Thematic vocabulary from top 3 harmonies
    let mut pairs: Vec<(usize, f32)> = harmony_activations
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let thematic_vocabulary: Vec<String> = pairs
        .iter()
        .take(3)
        .filter(|(_, v)| *v > 0.3)
        .flat_map(|(i, _)| HARMONY_THEMES[*i].iter().map(|s| s.to_string()))
        .collect();

    // Energy, warmth, complexity
    let energy = (arousal * 0.6 + dopamine * 0.3 + noradrenaline * 0.1).clamp(0.0, 1.0);
    let warmth = ((valence + 1.0) / 2.0 * 0.6 + serotonin * 0.4).clamp(0.0, 1.0);
    let complexity = (consciousness_level * 0.5
        + harmony_activations.iter().filter(|&&v| v > 0.3).count() as f32 / 8.0 * 0.5)
        .clamp(0.0, 1.0);

    CreativeSession {
        palette,
        root_frequency: root_freq,
        major_mode,
        tempo_bpm: tempo.clamp(30.0, 300.0),
        dominant_harmony: dominant,
        harmony_name: HARMONY_NAMES[dominant].to_string(),
        thematic_vocabulary,
        energy,
        warmth,
        complexity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_session_basic() {
        let harmonies = [0.5, 0.7, 0.3, 0.8, 0.4, 0.6, 0.9, 0.2];
        let session = derive_session(&harmonies, 0.3, 0.5, 0.7, 0.6, 0.5, 0.4);

        // EvolutionaryProgression (idx 6) has highest activation
        assert_eq!(session.dominant_harmony, 6);
        assert_eq!(session.harmony_name, "EvolutionaryProgression");
        assert!(session.major_mode); // positive valence
        assert!(!session.thematic_vocabulary.is_empty());
    }

    #[test]
    fn negative_valence_minor() {
        let session = derive_session(&[0.5; 8], -0.5, 0.5, 0.5, 0.5, 0.5, 0.5);
        assert!(!session.major_mode);
    }

    #[test]
    fn high_arousal_fast_tempo() {
        let calm = derive_session(&[0.5; 8], 0.0, 0.1, 0.5, 0.5, 0.5, 0.1);
        let excited = derive_session(&[0.5; 8], 0.0, 0.9, 0.5, 0.5, 0.5, 0.8);
        assert!(excited.tempo_bpm > calm.tempo_bpm);
    }

    #[test]
    fn stillness_slower_tempo() {
        let active = derive_session(
            &[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0],
            0.0,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
        );
        let still = derive_session(
            &[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0],
            0.0,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
        );
        assert!(still.tempo_bpm < active.tempo_bpm);
    }

    #[test]
    fn thematic_vocabulary_from_top_harmonies() {
        let harmonies = [0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.8];
        let session = derive_session(&harmonies, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5);
        // Should have InfinitePlay + SacredStillness themes
        assert!(
            session
                .thematic_vocabulary
                .iter()
                .any(|t| t == "dance" || t == "wonder")
        );
        assert!(
            session
                .thematic_vocabulary
                .iter()
                .any(|t| t == "silence" || t == "breath")
        );
    }

    #[test]
    fn energy_warmth_complexity_bounded() {
        let session = derive_session(&[1.0; 8], 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        assert!(session.energy >= 0.0 && session.energy <= 1.0);
        assert!(session.warmth >= 0.0 && session.warmth <= 1.0);
        assert!(session.complexity >= 0.0 && session.complexity <= 1.0);
    }

    #[test]
    fn palette_four_colors() {
        let session = derive_session(&[0.5; 8], 0.0, 0.5, 0.5, 0.5, 0.5, 0.5);
        assert_eq!(session.palette.len(), 4);
        for (h, s, l) in &session.palette {
            assert!(*h >= 0.0 && *s >= 0.0 && *l >= 0.0);
        }
    }
}
