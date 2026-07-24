// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Score-side closure of the Symthaea–Muse prediction loop.
//!
//! A cognitive proposal predicts directional changes. The theory engine then
//! produces a valid alternative. This module compares the baseline and
//! alternative scores with deterministic symbolic measurements and records the
//! resulting evidence and channel-specific prediction error in `PieceRecipe`.
//!
//! These measurements do not replace renderer analysis or listening tests.

use crate::cognitive_bridge::SymbolicMeasurementEvidence;
use crate::piece_recipe::PieceRecipe;
use serde::{Deserialize, Serialize};
use symthaea_music_theory::{Duration, Score, profile_score, profile_score_region};

/// Errors produced while attaching symbolic outcome evidence to a recipe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClosedLoopError {
    DecisionNotFound(u32),
    InvalidRegion,
}

/// Measure the complete baseline and candidate scores.
pub fn measure_symbolic_outcome(
    baseline: &Score,
    candidate: &Score,
) -> SymbolicMeasurementEvidence {
    SymbolicMeasurementEvidence::new(profile_score(baseline), profile_score(candidate))
}

/// Measure matching score regions governed by one edit contract.
pub fn measure_symbolic_region(
    baseline: &Score,
    candidate: &Score,
    start: Duration,
    end: Duration,
) -> Result<SymbolicMeasurementEvidence, ClosedLoopError> {
    let baseline =
        profile_score_region(baseline, start, end).ok_or(ClosedLoopError::InvalidRegion)?;
    let candidate =
        profile_score_region(candidate, start, end).ok_or(ClosedLoopError::InvalidRegion)?;
    Ok(SymbolicMeasurementEvidence::new(baseline, candidate))
}

/// Measure a complete candidate and attach the evidence to a recipe decision.
pub fn record_symbolic_outcome(
    recipe: &mut PieceRecipe,
    sequence: u32,
    baseline: &Score,
    candidate: &Score,
) -> Result<SymbolicMeasurementEvidence, ClosedLoopError> {
    let evidence = measure_symbolic_outcome(baseline, candidate);
    attach_evidence(recipe, sequence, evidence.clone())?;
    Ok(evidence)
}

/// Measure a selected region and attach the evidence to a recipe decision.
pub fn record_symbolic_region(
    recipe: &mut PieceRecipe,
    sequence: u32,
    baseline: &Score,
    candidate: &Score,
    start: Duration,
    end: Duration,
) -> Result<SymbolicMeasurementEvidence, ClosedLoopError> {
    let evidence = measure_symbolic_region(baseline, candidate, start, end)?;
    attach_evidence(recipe, sequence, evidence.clone())?;
    Ok(evidence)
}

fn attach_evidence(
    recipe: &mut PieceRecipe,
    sequence: u32,
    evidence: SymbolicMeasurementEvidence,
) -> Result<(), ClosedLoopError> {
    let decision = recipe
        .cognitive_decisions
        .get_mut(sequence as usize)
        .filter(|decision| decision.sequence == sequence)
        .ok_or(ClosedLoopError::DecisionNotFound(sequence))?;
    decision.observe_symbolic(evidence);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_bridge::{
        ActionScope, CognitiveDecisionTrace, CognitiveSection, InferenceEvidence,
        PredictedMusicalOutcome, SymbolicAction, SymbolicActionProposal, SymbolicMusicObservation,
    };
    use crate::musical_inference::MusicAction;
    use crate::piece_recipe::RendererRecipe;
    use symthaea_music_theory::{
        CompositionSpec, Emphasis, Key, MusicalIntent, Pitch, PitchClass, ScoreNote, Style,
        VoiceRole,
    };

    fn note(pitch: Pitch, onset: Duration) -> ScoreNote {
        ScoreNote {
            pitch,
            onset,
            duration: Duration::quarter(),
            velocity: 0.7,
            role: VoiceRole::Melody,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        }
    }

    fn trace() -> CognitiveDecisionTrace {
        CognitiveDecisionTrace {
            observation: SymbolicMusicObservation {
                section: CognitiveSection::Development,
                active_goal: None,
                goal_urgency: 0.5,
                valence: 0.0,
                arousal: 0.5,
                prediction_error: 0.2,
                consciousness_level: 0.5,
                dominant_harmony: 0,
                dominant_harmony_activation: 0.7,
                pending_obligations: 0,
                overdue_obligations: Vec::new(),
                obligation_demands: Vec::new(),
                obligation_pressure: 0.0,
            },
            inference: InferenceEvidence {
                source_action: MusicAction::IncreaseComplexity,
                free_energy: 0.2,
                prediction_error: 0.2,
                surprise: 0.1,
                sensory_precision: 1.0,
                prior_precision: 1.0,
            },
            proposal: SymbolicActionProposal {
                action: SymbolicAction::IncreaseDensity,
                driving_obligation_id: None,
                supporting_obligation_ids: Vec::new(),
                deferred_obligation_ids: Vec::new(),
                scope: ActionScope::CurrentPhrase,
                preserve: Vec::new(),
                urgency: 0.5,
                confidence: 0.5,
                rationale: vec!["increase density".into()],
            },
            predicted_outcome: PredictedMusicalOutcome {
                tension_delta: 0.15,
                density_delta: 0.35,
                familiarity_delta: 0.0,
                tonal_displacement_delta: 0.0,
            },
        }
    }

    fn recipe(spec: CompositionSpec) -> PieceRecipe {
        let mut recipe = PieceRecipe::new(
            MusicalIntent::default(),
            spec,
            RendererRecipe::new("native", 48_000, "0.1.0", "0.1.0"),
        );
        recipe.record_decision(trace());
        recipe
    }

    #[test]
    fn denser_candidate_produces_positive_density_delta() {
        let mut baseline = Score::new(Key::major(PitchClass::C), 120.0, 4);
        baseline.push(ScoreNote {
            duration: Duration::whole(),
            ..note(Pitch::new(PitchClass::C, 4), Duration::zero())
        });
        let mut candidate = Score::new(Key::major(PitchClass::C), 120.0, 4);
        for beat in 0..4 {
            candidate.push(note(Pitch::new(PitchClass::C, 4), Duration::new(beat, 1)));
        }

        let evidence = measure_symbolic_outcome(&baseline, &candidate);
        assert!(evidence.observed_outcome.density_delta > 0.0);
    }

    #[test]
    fn recording_evidence_closes_the_recipe_decision_loop() {
        let mut baseline = Score::new(Key::major(PitchClass::C), 120.0, 4);
        baseline.push(ScoreNote {
            duration: Duration::whole(),
            ..note(Pitch::new(PitchClass::C, 4), Duration::zero())
        });
        let mut candidate = baseline.clone();
        candidate.push(ScoreNote {
            pitch: Pitch::new(PitchClass::FSHARP, 4),
            role: VoiceRole::Harmony,
            ..note(Pitch::new(PitchClass::FSHARP, 4), Duration::zero())
        });

        let mut recipe = recipe(Style::Classical.spec());
        let evidence = record_symbolic_outcome(&mut recipe, 0, &baseline, &candidate).unwrap();
        let decision = &recipe.cognitive_decisions[0];

        assert_eq!(decision.symbolic_measurement.as_ref(), Some(&evidence));
        assert_eq!(decision.observed_outcome, Some(evidence.observed_outcome));
        assert!(decision.prediction_error.is_some());
    }

    #[test]
    fn unknown_decision_is_rejected() {
        let score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        let mut recipe = recipe(Style::Classical.spec());
        assert_eq!(
            record_symbolic_outcome(&mut recipe, 7, &score, &score),
            Err(ClosedLoopError::DecisionNotFound(7))
        );
    }
}
