// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic known-answer checks for multi-family score orchestration.
//!
//! Run with:
//! cargo run -p symthaea-music-theory --example orchestration_family_contract_benchmark
//!
//! All data is synthetic. This verifies contract mechanics, not instrumental
//! idiom, renderer quality, cultural authenticity, or listener preference.

use serde_json::json;
use symthaea_music_theory::harmony::Key;
use symthaea_music_theory::orchestration::{
    InstrumentCapabilityV1, InstrumentKindV1, InstrumentOrchestrationPlanV1,
    InstrumentPartAssignmentV1, ORCHESTRATION_PLAN_SCHEMA_V1, SoundingRangeV1,
    validate_instrument_orchestration,
};
use symthaea_music_theory::pitch::{Pitch, PitchClass};
use symthaea_music_theory::rhythm::Duration;
use symthaea_music_theory::score::{Emphasis, PartId, Score, ScoreNote, VoiceRole};

fn build_score() -> Score {
    let mut score = Score::new(Key::major(PitchClass::C), 84.0, 4);
    for (part, midi, role) in [
        (PartId(1), 48, VoiceRole::Bass),
        (PartId(2), 60, VoiceRole::Harmony),
        (PartId(3), 55, VoiceRole::CounterMelody),
        (PartId(4), 76, VoiceRole::Melody),
    ] {
        score.push(ScoreNote {
            part,
            pitch: Pitch::from_midi(midi),
            onset: Duration::zero(),
            duration: Duration::whole(),
            velocity: 0.72,
            role,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        });
    }
    score
}

fn build_plan() -> InstrumentOrchestrationPlanV1 {
    InstrumentOrchestrationPlanV1 {
        schema: ORCHESTRATION_PLAN_SCHEMA_V1.into(),
        instruments: vec![
            InstrumentCapabilityV1 {
                id: "cello_solo".into(),
                kind: InstrumentKindV1::Cello,
                sounding_range: SoundingRangeV1 { lowest_midi: 36, highest_midi: 84 },
                max_polyphony: 4,
            },
            InstrumentCapabilityV1 {
                id: "piano".into(),
                kind: InstrumentKindV1::Piano,
                sounding_range: SoundingRangeV1 { lowest_midi: 21, highest_midi: 108 },
                max_polyphony: 10,
            },
            InstrumentCapabilityV1 {
                id: "french_horn".into(),
                kind: InstrumentKindV1::FrenchHorn,
                sounding_range: SoundingRangeV1 { lowest_midi: 34, highest_midi: 77 },
                max_polyphony: 1,
            },
            InstrumentCapabilityV1 {
                id: "harp".into(),
                kind: InstrumentKindV1::Harp,
                sounding_range: SoundingRangeV1 { lowest_midi: 24, highest_midi: 104 },
                max_polyphony: 8,
            },
        ],
        assignments: vec![
            InstrumentPartAssignmentV1 {
                part_id: 1,
                instrument_id: "cello_solo".into(),
                musical_function: "evolving_bass_theme".into(),
            },
            InstrumentPartAssignmentV1 {
                part_id: 2,
                instrument_id: "piano".into(),
                musical_function: "harmonic_motion".into(),
            },
            InstrumentPartAssignmentV1 {
                part_id: 3,
                instrument_id: "french_horn".into(),
                musical_function: "broad_counter_theme".into(),
            },
            InstrumentPartAssignmentV1 {
                part_id: 4,
                instrument_id: "harp".into(),
                musical_function: "resonant_arpeggiation".into(),
            },
        ],
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let score = build_score();
    let plan = build_plan();

    let positive = validate_instrument_orchestration(&score, &plan);

    let mut out_of_range_score = score.clone();
    out_of_range_score.notes[0].pitch = Pitch::from_midi(28);
    let out_of_range = validate_instrument_orchestration(&out_of_range_score, &plan);
    let out_of_range_rejected = out_of_range.issues.iter().any(|issue| {
        issue.code == symthaea_music_theory::orchestration::OrchestrationIssueCodeV1::NoteOutOfRange
    });

    let mut polyphony_score = score.clone();
    polyphony_score.push(ScoreNote {
        part: PartId(2),
        pitch: Pitch::from_midi(64),
        onset: Duration::zero(),
        duration: Duration::whole(),
        velocity: 0.6,
        role: VoiceRole::Harmony,
        emphasis: Emphasis::Normal,
        section_intensity: 1.0,
    });
    let mut polyphony_plan = plan.clone();
    polyphony_plan.instruments.iter_mut()
        .find(|instrument| instrument.id == "piano")
        .expect("fixture piano profile exists")
        .max_polyphony = 1;
    let polyphony = validate_instrument_orchestration(&polyphony_score, &polyphony_plan);
    let polyphony_rejected = polyphony.issues.iter().any(|issue| {
        issue.code == symthaea_music_theory::orchestration::OrchestrationIssueCodeV1::PolyphonyExceeded
    });

    let all_assertions_hold = positive.valid && out_of_range_rejected && polyphony_rejected;
    let result = json!({
        "schema": "melothaea-orchestration-family-contract-benchmark-v1",
        "status": if all_assertions_hold {
            "synthetic_known_answer_mechanics_pass"
        } else {
            "failed"
        },
        "claim_boundary": "synthetic symbolic contract mechanics only; no audio, idiom, cultural authenticity, or preference claim",
        "positive_multi_family_case": {
            "valid": positive.valid,
            "parts": positive.part_count,
            "instrument_instances": positive.assigned_instrument_count,
            "instrument_families": positive.represented_family_count,
            "issues": positive.issues
        },
        "negative_controls": {
            "out_of_range_note_rejected": out_of_range_rejected,
            "polyphony_overcommitment_rejected": polyphony_rejected
        },
        "all_assertions_hold": all_assertions_hold
    });
    println!("{}", serde_json::to_string_pretty(&result)?);
    if !all_assertions_hold {
        return Err("orchestration benchmark assertion failed".into());
    }
    Ok(())
}
