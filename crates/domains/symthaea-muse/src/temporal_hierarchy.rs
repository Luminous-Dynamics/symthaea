// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Multi-timescale FEP hierarchy for music: Friston's temporal depth.
//!
//! Three levels of active inference, each operating at a different timescale:
//! - **Note level** (~32ms): predict next audio frame, select note-level actions
//! - **Phrase level** (~4 bars): predict harmonic trajectory, select chord progressions
//! - **Section level** (~16 bars): predict structural arc, select verse/chorus/bridge
//!
//! Higher levels set priors for lower levels. Lower levels report prediction
//! errors upward. This creates hierarchical predictive coding for music.

use crate::MusicalState;
use crate::audio_feedback::AudioFeatures;
use crate::musical_inference::{MusicAction, MusicInferenceResult, MusicalInferenceEngine};

/// Timescale at which inference operates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Timescale {
    /// ~32ms: individual notes and audio frames.
    Note,
    /// ~4 bars (~8 seconds at 120bpm): harmonic phrases.
    Phrase,
    /// ~16 bars (~32 seconds): song sections.
    Section,
}

/// Multi-timescale inference result combining all three levels.
#[derive(Debug, Clone)]
pub struct HierarchicalResult {
    /// Note-level action (immediate synthesis control).
    pub note_action: MusicAction,
    /// Phrase-level structural suggestion.
    pub phrase_suggestion: PhraseSuggestion,
    /// Section-level arc.
    pub section_arc: SectionArc,
    /// Combined free energy across all levels.
    pub total_free_energy: f64,
    /// Per-level free energies.
    pub level_free_energies: [f64; 3],
}

/// Phrase-level structural suggestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhraseSuggestion {
    /// Continue current harmonic pattern.
    Continue,
    /// Develop: vary the motif.
    Develop,
    /// Contrast: introduce new material.
    Contrast,
    /// Cadence: prepare for resolution.
    Cadence,
}

/// Section-level structural arc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionArc {
    /// Building tension (verse / development).
    Build,
    /// Peak intensity (chorus / climax).
    Peak,
    /// Release and resolve (bridge / coda).
    Release,
    /// Maintain steady state.
    Sustain,
}

/// Multi-timescale FEP hierarchy.
pub struct TemporalHierarchy {
    /// Note-level agent (runs every chunk, ~32ms).
    note_agent: MusicalInferenceEngine,
    /// Phrase-level state: accumulated features over ~4 bars.
    phrase_accumulator: Vec<AudioFeatures>,
    phrase_length: usize,
    phrase_cycles: usize,
    /// Section-level state: accumulated phrase summaries.
    section_phrase_count: usize,
    section_phrases_per_section: usize,
    /// Current structural suggestions from higher levels.
    current_phrase: PhraseSuggestion,
    current_section: SectionArc,
    /// Prediction errors flowing upward.
    phrase_prediction_error: f64,
    section_prediction_error: f64,
}

impl TemporalHierarchy {
    /// Create a new 3-level hierarchy.
    ///
    /// `phrase_chunks`: chunks per phrase (~250 = ~8s at 32ms/chunk)
    /// `phrases_per_section`: phrases per section (~4)
    pub fn new(phrase_chunks: usize, phrases_per_section: usize) -> Self {
        Self {
            note_agent: MusicalInferenceEngine::new(),
            phrase_accumulator: Vec::with_capacity(phrase_chunks),
            phrase_length: phrase_chunks,
            phrase_cycles: 0,
            section_phrase_count: 0,
            section_phrases_per_section: phrases_per_section,
            current_phrase: PhraseSuggestion::Continue,
            current_section: SectionArc::Build,
            phrase_prediction_error: 0.0,
            section_prediction_error: 0.0,
        }
    }

    /// Run one hierarchical inference cycle (called every chunk).
    pub fn infer(&mut self, features: &AudioFeatures) -> HierarchicalResult {
        // ── Note level: runs every chunk ──
        let note_result = self.note_agent.infer(features);

        // Accumulate for phrase level
        self.phrase_accumulator.push(*features);
        self.phrase_cycles += 1;

        // ── Phrase level: runs every phrase_length chunks ──
        if self.phrase_cycles >= self.phrase_length {
            self.update_phrase_level();
            self.phrase_accumulator.clear();
            self.phrase_cycles = 0;
            self.section_phrase_count += 1;

            // ── Section level: runs every N phrases ──
            if self.section_phrase_count >= self.section_phrases_per_section {
                self.update_section_level();
                self.section_phrase_count = 0;
            }
        }

        HierarchicalResult {
            note_action: note_result.action,
            phrase_suggestion: self.current_phrase,
            section_arc: self.current_section,
            total_free_energy: note_result.free_energy
                + self.phrase_prediction_error * 0.5
                + self.section_prediction_error * 0.25,
            level_free_energies: [
                note_result.free_energy,
                self.phrase_prediction_error,
                self.section_prediction_error,
            ],
        }
    }

    /// Apply hierarchical result to musical state.
    /// Higher-level suggestions set priors for lower-level actions.
    pub fn apply(&self, result: &HierarchicalResult, state: &mut MusicalState) {
        // Note level: direct action
        self.note_agent.apply_action(
            &MusicInferenceResult {
                action: result.note_action,
                free_energy: result.level_free_energies[0],
                prediction_error: 0.0,
                surprise: 0.0,
                is_surprised: false,
                learning_rate_mod: 1.0,
                sensory_precision: 1.0,
                prior_precision: 1.0,
            },
            state,
        );

        // Phrase level: modulate harmonic context
        match result.phrase_suggestion {
            PhraseSuggestion::Continue => {}
            PhraseSuggestion::Develop => {
                state.harmony_activations[6] += 0.05; // EvolutionaryProgression
            }
            PhraseSuggestion::Contrast => {
                state.prediction_error = (state.prediction_error + 0.1).min(1.0);
            }
            PhraseSuggestion::Cadence => {
                state.harmony_activations[0] += 0.1; // ResonantCoherence
                state.prediction_error = (state.prediction_error - 0.1).max(0.0);
            }
        }

        // Section level: modulate energy arc
        match result.section_arc {
            SectionArc::Build => {
                state.arousal = (state.arousal + 0.02).min(1.0);
            }
            SectionArc::Peak => {
                state.dopamine = (state.dopamine + 0.03).min(1.0);
            }
            SectionArc::Release => {
                state.arousal = (state.arousal - 0.03).max(0.0);
                state.serotonin = (state.serotonin + 0.02).min(1.0);
            }
            SectionArc::Sustain => {}
        }
    }

    fn update_phrase_level(&mut self) {
        if self.phrase_accumulator.is_empty() {
            return;
        }

        // Compute phrase-level statistics
        let n = self.phrase_accumulator.len() as f32;
        let _avg_centroid: f32 = self
            .phrase_accumulator
            .iter()
            .map(|f| f.spectral_centroid)
            .sum::<f32>()
            / n;
        let avg_tension: f32 = self
            .phrase_accumulator
            .iter()
            .map(|f| f.harmonic_tension)
            .sum::<f32>()
            / n;
        let avg_flux: f32 = self
            .phrase_accumulator
            .iter()
            .map(|f| f.spectral_flux)
            .sum::<f32>()
            / n;
        let avg_energy: f32 = self
            .phrase_accumulator
            .iter()
            .map(|f| f.rms_energy)
            .sum::<f32>()
            / n;

        // Phrase suggestion based on accumulated features
        self.current_phrase = if avg_tension > 0.6 {
            PhraseSuggestion::Cadence // high tension → resolve
        } else if avg_flux > 0.5 {
            PhraseSuggestion::Contrast // high change → introduce contrast
        } else if avg_energy > 0.5 {
            PhraseSuggestion::Develop // moderate energy → develop
        } else {
            PhraseSuggestion::Continue // low energy → continue
        };

        // Phrase prediction error: how much the phrase deviated from expectation
        self.phrase_prediction_error = (avg_flux + avg_tension * 0.5) as f64;
    }

    fn update_section_level(&mut self) {
        // Section arc cycles: Build → Peak → Release → Sustain → Build ...
        self.current_section = match self.current_section {
            SectionArc::Build => SectionArc::Peak,
            SectionArc::Peak => SectionArc::Release,
            SectionArc::Release => SectionArc::Sustain,
            SectionArc::Sustain => SectionArc::Build,
        };

        // Section PE: based on phrase-level accumulated error
        self.section_prediction_error = self.phrase_prediction_error * 0.7;
    }

    /// Get current timescale states.
    pub fn phrase_suggestion(&self) -> PhraseSuggestion {
        self.current_phrase
    }
    pub fn section_arc(&self) -> SectionArc {
        self.current_section
    }
    pub fn phrase_cycle(&self) -> usize {
        self.phrase_cycles
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn steady_features() -> AudioFeatures {
        AudioFeatures {
            spectral_centroid: 0.4,
            spectral_flux: 0.1,
            rhythm_entropy: 0.3,
            harmonic_tension: 0.2,
            rms_energy: 0.4,
            zero_crossing_rate: 0.1,
        }
    }

    #[test]
    fn hierarchy_creates_and_runs() {
        let mut h = TemporalHierarchy::new(10, 4); // short phrases for testing
        for _ in 0..50 {
            let result = h.infer(&steady_features());
            assert!(result.total_free_energy.is_finite());
        }
    }

    #[test]
    fn phrase_level_triggers() {
        let mut h = TemporalHierarchy::new(5, 2); // 5 chunks per phrase
        for i in 0..12 {
            h.infer(&steady_features());
            if i == 4 {
                assert_eq!(h.phrase_cycle(), 0, "phrase should reset at boundary");
            }
        }
    }

    #[test]
    fn section_arc_cycles() {
        let mut h = TemporalHierarchy::new(3, 2); // 3 chunks/phrase, 2 phrases/section
        assert_eq!(h.section_arc(), SectionArc::Build);

        // Run through one full section
        for _ in 0..6 {
            h.infer(&steady_features());
        }
        assert_eq!(h.section_arc(), SectionArc::Peak);

        for _ in 0..6 {
            h.infer(&steady_features());
        }
        assert_eq!(h.section_arc(), SectionArc::Release);
    }

    #[test]
    fn high_tension_triggers_cadence() {
        let mut h = TemporalHierarchy::new(5, 4);
        let tense = AudioFeatures {
            harmonic_tension: 0.8,
            spectral_flux: 0.3,
            ..steady_features()
        };
        for _ in 0..5 {
            h.infer(&tense);
        }
        assert_eq!(h.phrase_suggestion(), PhraseSuggestion::Cadence);
    }

    #[test]
    fn apply_modulates_state() {
        let h = TemporalHierarchy::new(10, 4);
        let result = HierarchicalResult {
            note_action: MusicAction::ResolveTension,
            phrase_suggestion: PhraseSuggestion::Cadence,
            section_arc: SectionArc::Release,
            total_free_energy: 0.5,
            level_free_energies: [0.3, 0.15, 0.05],
        };
        let mut state = MusicalState::default();
        let orig_arousal = state.arousal;
        h.apply(&result, &mut state);
        // Release should decrease arousal
        assert!(
            state.arousal < orig_arousal,
            "Release arc should decrease arousal"
        );
        // Cadence should boost coherence
        assert!(
            state.harmony_activations[0] > 0.3,
            "Cadence should boost coherence"
        );
    }
}
