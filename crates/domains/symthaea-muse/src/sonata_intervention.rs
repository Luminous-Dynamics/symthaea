// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Narrow, auditable Sonata intervention for returning thematic material.
//!
//! This is deliberately not a general note-writing agent. It handles one
//! product path: a [`SymbolicAction::ReturnOpeningMaterial`] proposal over an
//! existing, plan-bearing Sonata realization. The theory plan identifies the
//! source and target regions; deterministic strategies produce several score
//! candidates; independent obligation verification and symbolic measurements
//! rank them through an explicit policy that is separate from outcome prediction.

use crate::adaptive_prediction::{
    AdaptiveLearningError, AdaptiveOutcomeModel, InterventionCalibrationEvidence,
    InterventionPredictionContext, PredictionContext,
};
use crate::closed_loop::{ClosedLoopError, measure_symbolic_region};
use crate::cognitive_bridge::{
    CognitiveDecisionTrace, CognitiveSection, SymbolicAction, SymbolicMeasurementEvidence,
};
use crate::intervention::{InterventionDescriptor, InterventionStrategy, ObligationClass};
use crate::musical_policy::{
    MusicalPolicyPreference, MusicalPolicySelection, PolicyCandidateEvidence,
    select_by_musical_policy,
};
use serde::{Deserialize, Serialize};
use symthaea_music_theory::{
    MotifReturnEvidence, ObligationKind, Pitch, ReturnTransformation, Score, ScoreNote,
    ScoreValidationConfig, SonataObligationEvidence, SonataPlan, SonataRealization,
    SonataSectionKind, TheoryValidationReport, VoiceRole, compare_melodic_regions, validate_score,
    verify_sonata_obligations,
};

/// Deterministic strategies exposed to the artist and retained in provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SonataReturnStrategy {
    KeepCurrent,
    LiteralRegister,
    OctaveAbove,
    OctaveBelow,
    ContourFromCurrentRegister,
}

/// One generated score and all evidence used to assess it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SonataInterventionCandidate {
    pub alternative_id: String,
    pub strategy: SonataReturnStrategy,
    pub descriptor: InterventionDescriptor,
    pub score: Score,
    pub motif_return: MotifReturnEvidence,
    pub verification: Vec<SonataObligationEvidence>,
    pub theory_validation: TheoryValidationReport,
    pub preserved_invariants: bool,
    pub measurement: SymbolicMeasurementEvidence,
    pub prediction: InterventionCalibrationEvidence,
}

impl SonataInterventionCandidate {
    pub fn hard_constraints_valid(&self) -> bool {
        self.theory_validation.valid
    }
}

/// Complete candidate batch and deterministic recommendation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SonataInterventionBatch {
    pub source_section: SonataSectionKind,
    pub target_section: SonataSectionKind,
    pub expected_transformation: ReturnTransformation,
    pub driving_obligation_id: Option<u64>,
    pub candidates: Vec<SonataInterventionCandidate>,
    pub selection: MusicalPolicySelection,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SonataInterventionError {
    UnsupportedAction,
    MissingSourceSection,
    MissingTargetSection,
    MissingSourceMelody,
    MissingTargetMelody,
    InvalidMeasurementRegion,
}

impl From<ClosedLoopError> for SonataInterventionError {
    fn from(value: ClosedLoopError) -> Self {
        match value {
            ClosedLoopError::InvalidRegion => Self::InvalidMeasurementRegion,
            ClosedLoopError::DecisionNotFound(_) => Self::InvalidMeasurementRegion,
        }
    }
}

/// Generate and rank several plan-preserving returns of Sonata material.
pub fn generate_and_rank_sonata_return(
    realization: &SonataRealization,
    trace: &CognitiveDecisionTrace,
) -> Result<SonataInterventionBatch, SonataInterventionError> {
    let context = PredictionContext::new(
        trace.proposal.action,
        trace.observation.section,
        "Sonata",
        "Sonata",
        realization.score.meter,
        crate::adaptive_prediction::TextureBand::from_active_voices(
            symthaea_music_theory::profile_score(&realization.score).active_voice_count,
        ),
    );
    generate_and_rank_sonata_return_with_model(
        realization,
        trace,
        &AdaptiveOutcomeModel::default(),
        context,
    )
}

pub fn generate_and_rank_sonata_return_with_model(
    realization: &SonataRealization,
    trace: &CognitiveDecisionTrace,
    model: &AdaptiveOutcomeModel,
    context: PredictionContext,
) -> Result<SonataInterventionBatch, SonataInterventionError> {
    if trace.proposal.action != SymbolicAction::ReturnOpeningMaterial
        || context.action != trace.proposal.action
    {
        return Err(SonataInterventionError::UnsupportedAction);
    }

    let (source_kind, target_kind, expected) = thematic_target(trace);
    let source = section(&realization.plan, source_kind)
        .ok_or(SonataInterventionError::MissingSourceSection)?;
    let target = section(&realization.plan, target_kind)
        .ok_or(SonataInterventionError::MissingTargetSection)?;
    let source_indices =
        melody_indices(&realization.score, source.start.beats(), source.end.beats());
    let target_indices =
        melody_indices(&realization.score, target.start.beats(), target.end.beats());
    if source_indices.is_empty() {
        return Err(SonataInterventionError::MissingSourceMelody);
    }
    if target_indices.is_empty() {
        return Err(SonataInterventionError::MissingTargetMelody);
    }

    let mut generated = Vec::new();
    generated.push((
        "current-score".to_owned(),
        SonataReturnStrategy::KeepCurrent,
        realization.score.clone(),
    ));
    for (id, strategy) in [
        ("literal-register", SonataReturnStrategy::LiteralRegister),
        ("octave-above", SonataReturnStrategy::OctaveAbove),
        ("octave-below", SonataReturnStrategy::OctaveBelow),
        (
            "contour-current-register",
            SonataReturnStrategy::ContourFromCurrentRegister,
        ),
    ] {
        generated.push((
            id.to_owned(),
            strategy,
            apply_return_strategy(
                &realization.score,
                &source_indices,
                &target_indices,
                strategy,
            ),
        ));
    }

    let baseline_profile =
        symthaea_music_theory::profile_score_region(&realization.score, target.start, target.end)
            .unwrap_or_default();
    let baseline_motif = compare_melodic_regions(
        &realization.score,
        source.start,
        source.end,
        target.start,
        target.end,
        expected,
    );
    let mut candidates = Vec::with_capacity(generated.len());
    let mut policy_evidence = Vec::with_capacity(generated.len());
    for (alternative_id, strategy, score) in generated {
        let motif_return = compare_melodic_regions(
            &score,
            source.start,
            source.end,
            target.start,
            target.end,
            expected,
        );
        let (resolution, verification) = verify_sonata_obligations(&score, &realization.plan);
        let target_verified = trace.proposal.driving_obligation_id.map(|id| {
            verification
                .iter()
                .find(|evidence| evidence.obligation_id == id)
                .is_some_and(|evidence| evidence.verified)
        });
        let pressure = resolution.pressure_at(score.total_beats);
        let preserved_invariants =
            preserves_score_skeleton_outside_target(&realization.score, &score, &target_indices);
        let measurement =
            measure_symbolic_region(&realization.score, &score, target.start, target.end)?;
        let descriptor = descriptor_for(
            strategy,
            trace,
            baseline_motif.overall_similarity,
            baseline_profile.tension,
            baseline_profile.density,
            target_indices.len(),
            score.notes.len(),
        );
        let prediction_context =
            InterventionPredictionContext::new(context.clone(), descriptor.clone());
        let prediction = model.predict_intervention(&prediction_context);
        let validation_config = ScoreValidationConfig {
            max_melodic_leap_semitones: 24,
            check_strong_beat_consonance: false,
            check_parallel_perfect_motion: false,
            ..ScoreValidationConfig::default()
        };
        let theory_validation = validate_score(&score, &validation_config);
        policy_evidence.push(PolicyCandidateEvidence {
            alternative_id: alternative_id.clone(),
            theory_validation: theory_validation.clone(),
            preserved_invariants,
            overdue_obligations_remaining: pressure.overdue_count,
            unresolved_obligations_remaining: resolution.unresolved().len(),
            obligation_pressure_remaining: pressure.weighted_pressure,
            target_obligation_verified: target_verified,
            motif_return_similarity: Some(motif_return.overall_similarity),
            prediction: prediction.clone(),
            observed_outcome: Some(measurement.observed_outcome),
        });
        candidates.push(SonataInterventionCandidate {
            alternative_id,
            strategy,
            descriptor,
            score,
            motif_return,
            verification,
            theory_validation,
            preserved_invariants,
            measurement,
            prediction,
        });
    }

    let selection = select_by_musical_policy(
        MusicalPolicyPreference::for_action(trace.proposal.action),
        &policy_evidence,
    );

    Ok(SonataInterventionBatch {
        source_section: source_kind,
        target_section: target_kind,
        expected_transformation: expected,
        driving_obligation_id: trace.proposal.driving_obligation_id,
        candidates,
        selection,
    })
}

/// Learn symbolic outcomes from every theory-valid generated alternative.
/// This updates only the world model; previewing never creates an artist
/// preference label.
pub fn learn_symbolic_outcomes_from_batch(
    model: &mut AdaptiveOutcomeModel,
    batch: &SonataInterventionBatch,
) -> Result<usize, AdaptiveLearningError> {
    let mut learned = 0;
    for candidate in &batch.candidates {
        if candidate.theory_validation.valid && candidate.preserved_invariants {
            model.observe_intervention(
                candidate.prediction.context.clone(),
                candidate.measurement.observed_outcome,
            )?;
            learned += 1;
        }
    }
    Ok(learned)
}

fn descriptor_for(
    strategy: SonataReturnStrategy,
    trace: &CognitiveDecisionTrace,
    baseline_motif_similarity: f32,
    baseline_tension: f32,
    baseline_density: f32,
    affected_note_count: usize,
    score_note_count: usize,
) -> InterventionDescriptor {
    let (generic_strategy, pitch_shift, strength) = match strategy {
        SonataReturnStrategy::KeepCurrent => (InterventionStrategy::KeepCurrent, 0, 0.0),
        SonataReturnStrategy::LiteralRegister => (InterventionStrategy::Literal, 0, 0.5),
        SonataReturnStrategy::OctaveAbove => (InterventionStrategy::RegisterShift, 12, 0.75),
        SonataReturnStrategy::OctaveBelow => (InterventionStrategy::RegisterShift, -12, 0.75),
        SonataReturnStrategy::ContourFromCurrentRegister => {
            (InterventionStrategy::ContourPreserving, 0, 0.6)
        }
    };
    let obligation = trace
        .proposal
        .driving_obligation_id
        .and_then(|id| {
            trace
                .observation
                .obligation_demands
                .iter()
                .find(|demand| demand.id == id)
        })
        .map(|demand| obligation_class(&demand.kind))
        .unwrap_or(ObligationClass::None);
    InterventionDescriptor::new(
        trace.proposal.action,
        generic_strategy,
        CognitiveSection::Exposition,
        CognitiveSection::Recapitulation,
        obligation,
        pitch_shift,
        1.0,
        strength,
        baseline_motif_similarity,
        baseline_tension,
        baseline_density,
        affected_note_count,
        score_note_count,
    )
}

fn obligation_class(kind: &ObligationKind) -> ObligationClass {
    match kind {
        ObligationKind::ReturnMotif { .. } => ObligationClass::ReturnMotif,
        ObligationKind::ReachKey { .. } => ObligationClass::ReachKey,
        ObligationKind::RestoreIdentity { .. } => ObligationClass::RestoreIdentity,
        ObligationKind::ResolveAlteredDegree { .. } => ObligationClass::ResolveAlteredDegree,
        ObligationKind::Cadence { .. } => ObligationClass::Cadence,
        ObligationKind::ReachClimax => ObligationClass::Climax,
        ObligationKind::EnterVoice { .. } => ObligationClass::VoiceEntry,
        ObligationKind::Custom { .. } => ObligationClass::None,
    }
}

fn thematic_target(
    trace: &CognitiveDecisionTrace,
) -> (SonataSectionKind, SonataSectionKind, ReturnTransformation) {
    let demand = trace.proposal.driving_obligation_id.and_then(|id| {
        trace
            .observation
            .obligation_demands
            .iter()
            .find(|demand| demand.id == id)
    });
    match demand.map(|demand| &demand.kind) {
        Some(ObligationKind::ReturnMotif {
            motif_id,
            transformation,
        }) if motif_id == "sonata.secondary" => (
            SonataSectionKind::ExpositionSecondary,
            SonataSectionKind::RecapitulationSecondary,
            *transformation,
        ),
        Some(ObligationKind::ReturnMotif { transformation, .. }) => (
            SonataSectionKind::ExpositionPrimary,
            SonataSectionKind::RecapitulationPrimary,
            *transformation,
        ),
        _ => (
            SonataSectionKind::ExpositionPrimary,
            SonataSectionKind::RecapitulationPrimary,
            ReturnTransformation::Literal,
        ),
    }
}

fn section(
    plan: &SonataPlan,
    kind: SonataSectionKind,
) -> Option<&symthaea_music_theory::PlannedSonataSection> {
    plan.sections.iter().find(|section| section.kind == kind)
}

fn melody_indices(score: &Score, start: f64, end: f64) -> Vec<usize> {
    let mut indices: Vec<_> = score
        .notes
        .iter()
        .enumerate()
        .filter(|(_, note)| {
            note.role == VoiceRole::Melody
                && note.onset.beats() >= start
                && note.onset.beats() < end
        })
        .map(|(index, _)| index)
        .collect();
    indices.sort_by(|left, right| {
        score.notes[*left]
            .onset
            .beats()
            .total_cmp(&score.notes[*right].onset.beats())
    });
    indices
}

fn apply_return_strategy(
    baseline: &Score,
    source_indices: &[usize],
    target_indices: &[usize],
    strategy: SonataReturnStrategy,
) -> Score {
    if strategy == SonataReturnStrategy::KeepCurrent {
        return baseline.clone();
    }
    let mut candidate = baseline.clone();
    let source_pitches: Vec<Pitch> = source_indices
        .iter()
        .map(|index| baseline.notes[*index].pitch)
        .collect();
    let current_target: Vec<Pitch> = target_indices
        .iter()
        .map(|index| baseline.notes[*index].pitch)
        .collect();

    let generated = match strategy {
        SonataReturnStrategy::KeepCurrent => current_target,
        SonataReturnStrategy::LiteralRegister => {
            resample_pitches(&source_pitches, target_indices.len())
        }
        SonataReturnStrategy::OctaveAbove => {
            resample_pitches(&source_pitches, target_indices.len())
                .into_iter()
                .map(|pitch| pitch.transpose(12))
                .collect()
        }
        SonataReturnStrategy::OctaveBelow => {
            resample_pitches(&source_pitches, target_indices.len())
                .into_iter()
                .map(|pitch| pitch.transpose(-12))
                .collect()
        }
        SonataReturnStrategy::ContourFromCurrentRegister => {
            contour_from_anchor(&source_pitches, &current_target)
        }
    };
    for (index, pitch) in target_indices.iter().zip(generated) {
        candidate.notes[*index].pitch = pitch;
    }
    candidate
}

fn resample_pitches(source: &[Pitch], target_len: usize) -> Vec<Pitch> {
    if source.is_empty() || target_len == 0 {
        return Vec::new();
    }
    if target_len == 1 {
        return vec![source[0]];
    }
    (0..target_len)
        .map(|index| {
            let source_index = index * (source.len() - 1) / (target_len - 1);
            source[source_index]
        })
        .collect()
}

fn contour_from_anchor(source: &[Pitch], current_target: &[Pitch]) -> Vec<Pitch> {
    if source.is_empty() || current_target.is_empty() {
        return Vec::new();
    }
    let source = resample_pitches(source, current_target.len());
    let mut result = Vec::with_capacity(current_target.len());
    result.push(current_target[0]);
    for pair in source.windows(2) {
        let interval = pair[0].semitones_to(pair[1]);
        let next = result.last().copied().unwrap().transpose(interval);
        result.push(next);
    }
    result
}

fn preserves_score_skeleton_outside_target(
    baseline: &Score,
    candidate: &Score,
    target_indices: &[usize],
) -> bool {
    if baseline.notes.len() != candidate.notes.len() {
        return false;
    }
    let mut target = vec![false; baseline.notes.len()];
    for index in target_indices {
        if let Some(value) = target.get_mut(*index) {
            *value = true;
        }
    }
    baseline
        .notes
        .iter()
        .zip(&candidate.notes)
        .enumerate()
        .all(|(index, (before, after))| {
            if target[index] {
                same_note_except_pitch(before, after)
            } else {
                before == after
            }
        })
}

fn same_note_except_pitch(left: &ScoreNote, right: &ScoreNote) -> bool {
    left.onset == right.onset
        && left.duration == right.duration
        && left.velocity == right.velocity
        && left.role == right.role
        && left.emphasis == right.emphasis
        && left.section_intensity == right.section_intensity
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_bridge::{
        ActionScope, CognitiveObligationDemand, CognitiveSection, InferenceEvidence,
        PredictedMusicalOutcome, PreserveInvariant, SymbolicActionProposal,
        SymbolicMusicObservation,
    };
    use crate::musical_inference::MusicAction;
    use symthaea_music_theory::{MusicalIntent, Style, compose_sonata_with_plan};

    fn return_trace(realization: &SonataRealization) -> CognitiveDecisionTrace {
        let obligation = realization
            .plan
            .obligations
            .items()
            .iter()
            .find(|item| {
                matches!(
                    &item.kind,
                    ObligationKind::ReturnMotif { motif_id, .. }
                        if motif_id == "sonata.primary"
                )
            })
            .unwrap();
        CognitiveDecisionTrace {
            observation: SymbolicMusicObservation {
                section: CognitiveSection::Recapitulation,
                active_goal: None,
                goal_urgency: 0.9,
                valence: 0.0,
                arousal: 0.5,
                prediction_error: 0.3,
                consciousness_level: 0.5,
                dominant_harmony: 0,
                dominant_harmony_activation: 0.8,
                pending_obligations: 1,
                overdue_obligations: vec![obligation.id],
                obligation_demands: vec![CognitiveObligationDemand {
                    id: obligation.id,
                    priority: obligation.priority,
                    due_by: obligation.due_by,
                    overdue: true,
                    kind: obligation.kind.clone(),
                }],
                obligation_pressure: 1.0,
            },
            inference: InferenceEvidence {
                source_action: MusicAction::RepeatMotif,
                free_energy: 0.2,
                prediction_error: 0.3,
                surprise: 0.2,
                sensory_precision: 1.0,
                prior_precision: 1.0,
            },
            proposal: SymbolicActionProposal {
                action: SymbolicAction::ReturnOpeningMaterial,
                driving_obligation_id: Some(obligation.id),
                supporting_obligation_ids: vec![obligation.id],
                deferred_obligation_ids: Vec::new(),
                scope: ActionScope::CurrentSection,
                preserve: vec![
                    PreserveInvariant::MotifIdentity,
                    PreserveInvariant::Meter,
                    PreserveInvariant::FormLength,
                    PreserveInvariant::Ending,
                ],
                urgency: 1.0,
                confidence: 0.7,
                rationale: vec!["restore the primary subject".into()],
            },
            predicted_outcome: PredictedMusicalOutcome {
                tension_delta: -0.2,
                density_delta: 0.0,
                familiarity_delta: 0.5,
                tonal_displacement_delta: -0.35,
            },
        }
    }

    #[test]
    fn several_plan_preserving_returns_are_generated_and_ranked() {
        let intent = MusicalIntent {
            seed: 41,
            bars: 4,
            ..MusicalIntent::default()
        };
        let spec = Style::Sonata.spec();
        let mut realization = compose_sonata_with_plan(&intent, &spec).unwrap();
        let target = section(&realization.plan, SonataSectionKind::RecapitulationPrimary).unwrap();
        let target_indices =
            melody_indices(&realization.score, target.start.beats(), target.end.beats());
        // Simulate an artist or generator weakening the return before Symthaea
        // proposes a repair.
        for index in &target_indices {
            realization.score.notes[*index].pitch =
                realization.score.notes[*index].pitch.transpose(1);
        }
        let (resolution, verification) =
            verify_sonata_obligations(&realization.score, &realization.plan);
        realization.resolution = resolution;
        realization.verification = verification;

        let trace = return_trace(&realization);
        let batch = generate_and_rank_sonata_return(&realization, &trace).unwrap();
        assert_eq!(batch.candidates.len(), 5);
        let selected = batch.selection.recommended_id.as_deref().unwrap();
        let selected = batch
            .candidates
            .iter()
            .find(|candidate| candidate.alternative_id == selected)
            .unwrap();
        assert!(selected.hard_constraints_valid());
        assert!(selected.preserved_invariants);
        assert!(selected.motif_return.overall_similarity >= 0.95);
        assert!(selected.verification.iter().any(|evidence| {
            evidence.obligation_id == trace.proposal.driving_obligation_id.unwrap()
                && evidence.verified
        }));
    }

    #[test]
    fn preview_batch_trains_world_model_from_all_valid_alternatives() {
        let intent = MusicalIntent {
            seed: 43,
            bars: 4,
            ..MusicalIntent::default()
        };
        let spec = Style::Sonata.spec();
        let realization = compose_sonata_with_plan(&intent, &spec).unwrap();
        let trace = return_trace(&realization);
        let batch = generate_and_rank_sonata_return(&realization, &trace).unwrap();
        let mut model = AdaptiveOutcomeModel::default();
        let learned = learn_symbolic_outcomes_from_batch(&mut model, &batch).unwrap();
        assert_eq!(
            learned,
            batch
                .candidates
                .iter()
                .filter(|candidate| {
                    candidate.theory_validation.valid && candidate.preserved_invariants
                })
                .count()
        );
        assert_eq!(model.intervention_contexts.len(), learned);
    }

    #[test]
    fn intervention_rejects_unrelated_actions() {
        let intent = MusicalIntent::default();
        let spec = Style::Sonata.spec();
        let realization = compose_sonata_with_plan(&intent, &spec).unwrap();
        let mut trace = return_trace(&realization);
        trace.proposal.action = SymbolicAction::IncreaseDensity;
        assert_eq!(
            generate_and_rank_sonata_return(&realization, &trace),
            Err(SonataInterventionError::UnsupportedAction)
        );
    }
}
