#![cfg(feature = "theory")]

use sha2::{Digest, Sha256};
use symthaea_muse::cognitive_bridge::{
    ActionScope, CognitiveDecisionTrace, CognitiveGoal, CognitiveSection, InferenceEvidence,
    SymbolicAction, SymbolicActionProposal, SymbolicMeasurementEvidence, SymbolicMusicObservation,
    default_predicted_outcome,
};
use symthaea_muse::cognitive_selection::{
    SymbolicAlternativeEvidence, select_symbolic_alternative,
    select_symbolic_alternative_from_inference,
};
use symthaea_muse::musical_inference::MusicAction;
use symthaea_music_theory::{
    Duration, Emphasis, Key, PartId, Pitch, PitchClass, Score, ScoreNote, VoiceRole,
    profile_score,
};

fn note(onset: Duration, duration: Duration) -> ScoreNote {
    ScoreNote {
        part: PartId::UNASSIGNED,
        pitch: Pitch::new(PitchClass::C, 4),
        onset,
        duration,
        velocity: 0.7,
        role: VoiceRole::Melody,
        emphasis: Emphasis::Normal,
        section_intensity: 1.0,
    }
}

fn sparse_score() -> Score {
    let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
    score.push(note(Duration::zero(), Duration::whole()));
    score
}

fn dense_score() -> Score {
    let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
    for beat in 0..4 {
        score.push(note(Duration::new(beat, 1), Duration::quarter()));
    }
    score
}

fn trace() -> CognitiveDecisionTrace {
    CognitiveDecisionTrace {
        observation: SymbolicMusicObservation {
            section: CognitiveSection::Recapitulation,
            active_goal: Some(CognitiveGoal::Recapitulate),
            goal_urgency: 1.0,
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
            action: SymbolicAction::ReturnOpeningMaterial,
            driving_obligation_id: None,
            supporting_obligation_ids: Vec::new(),
            deferred_obligation_ids: Vec::new(),
            scope: ActionScope::CurrentSection,
            preserve: Vec::new(),
            urgency: 1.0,
            confidence: 0.5,
            rationale: vec![
                "formal recapitulation action remains ReturnOpeningMaterial".into(),
            ],
        },
        predicted_outcome: default_predicted_outcome(SymbolicAction::ReturnOpeningMaterial),
    }
}

fn evidence(id: &str, baseline: &Score, candidate: &Score) -> SymbolicAlternativeEvidence {
    SymbolicAlternativeEvidence {
        alternative_id: id.into(),
        measurement: SymbolicMeasurementEvidence::new(
            profile_score(baseline),
            profile_score(candidate),
        ),
        hard_constraints_valid: true,
        preserved_invariants: true,
        overdue_obligations_remaining: 0,
        unresolved_obligations_remaining: 0,
        obligation_pressure_remaining: 0.0,
        target_obligation_verified: Some(true),
        motif_return_similarity: None,
    }
}

fn score_sha256(score: &Score) -> String {
    let bytes = serde_json::to_vec(score).expect("Score must serialize for identity evidence");
    format!("{:x}", Sha256::digest(bytes))
}

#[test]
fn temporal_source_preference_can_reach_a_different_concrete_score_identity() {
    let sparse = sparse_score();
    let dense = dense_score();
    let sparse_profile = profile_score(&sparse);
    let dense_profile = profile_score(&dense);
    assert!(dense_profile.density > sparse_profile.density);

    let alternatives = [
        evidence("unchanged-sparse", &sparse, &sparse),
        evidence("measured-denser", &sparse, &dense),
    ];
    let trace = trace();

    // The formal ReturnOpeningMaterial effect target prefers preserving the
    // unchanged score when all formal/theory gates are equal.
    let formal = select_symbolic_alternative(&trace, &alternatives);
    assert_eq!(formal.recommended_id.as_deref(), Some("unchanged-sparse"));

    // The formal action remains ReturnOpeningMaterial, but the actual FEP
    // source action (IncreaseComplexity) is now allowed to choose among the
    // already-valid alternatives and prefers the measured density increase.
    let cognitive = select_symbolic_alternative_from_inference(&trace, &alternatives);
    assert_eq!(
        cognitive.formal_proposal_action,
        SymbolicAction::ReturnOpeningMaterial
    );
    assert_eq!(cognitive.source_action, MusicAction::IncreaseComplexity);
    assert_eq!(
        cognitive.selection.recommended_id.as_deref(),
        Some("measured-denser")
    );

    let baseline_sha = score_sha256(&sparse);
    let selected_sha = match cognitive.selection.recommended_id.as_deref() {
        Some("unchanged-sparse") => score_sha256(&sparse),
        Some("measured-denser") => score_sha256(&dense),
        other => panic!("unexpected cognitive recommendation: {other:?}"),
    };
    assert_ne!(baseline_sha, selected_sha);
}

#[test]
fn alternative_id_divergence_is_not_score_divergence() {
    let score = dense_score();
    let alias_of_same_score = score.clone();

    // Future causal evidence must compare score content identity, not merely
    // candidate IDs. Two differently named alternatives can still be the
    // exact same music.
    assert_eq!(score_sha256(&score), score_sha256(&alias_of_same_score));
}
