use mel_music_edit_intent_grounding_harness::{
    verify_music_edit_intent_plan_v1, seal_source_observation_v1,
    EditParameterV1, GroundingEvidenceKindV1, IntentTargetWindowV1,
    MusicEditIntentPlanV1, MusicEditOperationV1, OccurrenceGroundingEvidenceV1,
    OccurrenceSelectorV1, PlanDispositionV1, SourceEventV1,
    SourceOccurrenceV1, SourceObservationV1, TemporalAnchorV1,
    TemporalAnchorKindV1, SOURCE_OBSERVATION_SCHEMA_V1, INTENT_PLAN_SCHEMA_V1,
};
use serde::Serialize;

fn h(value: u64) -> String {
    format!("{value:064x}")
}

fn event(id: &str, part: &str, onset: u64, duration: u64) -> SourceEventV1 {
    SourceEventV1 {
        event_id: id.to_owned(),
        part_id: part.to_owned(),
        onset_ticks: onset,
        duration_ticks: duration,
    }
}

fn source() -> Result<SourceObservationV1, Box<dyn std::error::Error>> {
    Ok(seal_source_observation_v1(SourceObservationV1 {
        schema_version: SOURCE_OBSERVATION_SCHEMA_V1.to_owned(),
        source_artifact_sha256: h(1),
        source_lineage_sha256: h(2),
        duration_ticks: 6_000,
        events: vec![
            event("bass-note-1", "bass", 100, 300),
            event("guitar-note-1", "guitar", 1_200, 300),
            event("guitar-note-2", "guitar", 4_100, 300),
            event("snare-hit-1", "drums", 4_400, 100),
            event("vocal-note-1", "vocal", 4_300, 200),
        ],
        anchors: vec![
            TemporalAnchorV1 {
                anchor_id: "bridge-1".to_owned(),
                kind: TemporalAnchorKindV1::Section,
                label: "bridge".to_owned(),
                start_tick: 2_000,
                end_tick: 3_000,
            },
            TemporalAnchorV1 {
                anchor_id: "chorus-1".to_owned(),
                kind: TemporalAnchorKindV1::Section,
                label: "first chorus".to_owned(),
                start_tick: 1_000,
                end_tick: 2_000,
            },
            TemporalAnchorV1 {
                anchor_id: "chorus-2".to_owned(),
                kind: TemporalAnchorKindV1::Section,
                label: "second chorus".to_owned(),
                start_tick: 4_000,
                end_tick: 5_000,
            },
        ],
        occurrences: vec![
            SourceOccurrenceV1 {
                occurrence_id: "guitar-entry-1".to_owned(),
                semantic_class_id: "guitar_entrance".to_owned(),
                start_tick: 1_100,
                end_tick: 1_600,
                event_ids: vec!["guitar-note-1".to_owned()],
                anchor_ids: vec!["chorus-1".to_owned()],
            },
            SourceOccurrenceV1 {
                occurrence_id: "guitar-entry-2".to_owned(),
                semantic_class_id: "guitar_entrance".to_owned(),
                start_tick: 4_100,
                end_tick: 4_500,
                event_ids: vec!["guitar-note-2".to_owned()],
                anchor_ids: vec!["chorus-2".to_owned()],
            },
            SourceOccurrenceV1 {
                occurrence_id: "snare-hit-1".to_owned(),
                semantic_class_id: "snare_hit".to_owned(),
                start_tick: 4_400,
                end_tick: 4_500,
                event_ids: vec!["snare-hit-1".to_owned()],
                anchor_ids: vec!["chorus-2".to_owned()],
            },
        ],
        observation_sha256: String::new(),
    })?)
}

#[derive(Serialize)]
struct Report {
    ready_plan_sha256: String,
    ready_plan_verified: bool,
    clarification_plan_sha256: String,
    clarification_plan_verified: bool,
    ready_selected_events: Vec<String>,
    clarification_candidates: Vec<String>,
    clarification_is_executable: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let observation = source()?;
    let ready = MusicEditIntentPlanV1 {
        schema_version: INTENT_PLAN_SCHEMA_V1.to_owned(),
        plan_id: "remove-second-guitar-entry".to_owned(),
        instruction_sha256: h(3),
        source_observation_sha256: observation.observation_sha256.clone(),
        disposition: PlanDispositionV1::Ready,
        operation: Some(MusicEditOperationV1::Remove),
        selector: Some(OccurrenceSelectorV1::NthMatching {
            semantic_class_id: "guitar_entrance".to_owned(),
            ordinal: 2,
        }),
        selected_occurrence_ids: vec!["guitar-entry-2".to_owned()],
        selected_event_ids: vec!["guitar-note-2".to_owned()],
        protected_event_ids: vec![
            "bass-note-1".to_owned(),
            "guitar-note-1".to_owned(),
            "snare-hit-1".to_owned(),
            "vocal-note-1".to_owned(),
        ],
        target_windows: vec![IntentTargetWindowV1 {
            occurrence_id: "guitar-entry-2".to_owned(),
            start_tick: 4_100,
            end_tick: 4_500,
        }],
        parameters: Vec::<EditParameterV1>::new(),
        grounding_evidence: vec![OccurrenceGroundingEvidenceV1 {
            occurrence_id: "guitar-entry-2".to_owned(),
            source_observation_sha256: observation.observation_sha256.clone(),
            evidence_kind: GroundingEvidenceKindV1::TemporalAnchor,
            confidence_bp: 9_700,
        }],
        clarification_candidates: Vec::new(),
        clarification_question: None,
        abstention_reason: None,
        confidence_bp: 9_700,
    };
    verify_music_edit_intent_plan_v1(&observation, &ready)?;
    let ready_hash = mel_music_edit_intent_grounding_harness::intent_plan_commitment_v1(&ready)?;

    let clarification = MusicEditIntentPlanV1 {
        schema_version: INTENT_PLAN_SCHEMA_V1.to_owned(),
        plan_id: "clarify-which-guitar-entry".to_owned(),
        instruction_sha256: h(4),
        source_observation_sha256: observation.observation_sha256.clone(),
        disposition: PlanDispositionV1::NeedsClarification,
        operation: Some(MusicEditOperationV1::Remove),
        selector: Some(OccurrenceSelectorV1::UniqueMatching {
            semantic_class_id: "guitar_entrance".to_owned(),
        }),
        selected_occurrence_ids: Vec::new(),
        selected_event_ids: Vec::new(),
        protected_event_ids: Vec::new(),
        target_windows: Vec::new(),
        parameters: Vec::new(),
        grounding_evidence: Vec::new(),
        clarification_candidates: vec![
            "guitar-entry-1".to_owned(),
            "guitar-entry-2".to_owned(),
        ],
        clarification_question: Some(
            "Which guitar entrance should I remove: the first chorus or second chorus?".to_owned()
        ),
        abstention_reason: None,
        confidence_bp: 7_000,
    };
    verify_music_edit_intent_plan_v1(&observation, &clarification)?;
    let clarification_hash = mel_music_edit_intent_grounding_harness::intent_plan_commitment_v1(&clarification)?;

    println!("{}", serde_json::to_string_pretty(&Report {
        ready_plan_sha256: ready_hash,
        ready_plan_verified: true,
        clarification_plan_sha256: clarification_hash,
        clarification_plan_verified: true,
        ready_selected_events: ready.selected_event_ids,
        clarification_candidates: clarification.clarification_candidates,
        clarification_is_executable: clarification.disposition == PlanDispositionV1::Ready,
    })?);
    Ok(())
}
