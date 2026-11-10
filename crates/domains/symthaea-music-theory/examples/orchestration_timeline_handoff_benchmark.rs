// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Synthetic known-answer mechanics for time-indexed instrument handoffs.
//!
//! This example validates a symbolic line transferred from cello to horn.
//! It does not compose, render, or evaluate a musical performance.

use symthaea_music_theory::harmony::Key;
use symthaea_music_theory::orchestration::{
    InstrumentCapabilityV1, InstrumentKindV1, SoundingRangeV1,
};
use symthaea_music_theory::orchestration_timeline::{
    HandoffKindV1, OrchestrationSegmentV1, OrchestrationTimelineV1,
    ORCHESTRATION_TIMELINE_SCHEMA_V1, ORCHESTRATION_SCORE_DIGEST_SCHEMA_V1,
    RationalBeatV1, SegmentHandoffV1,
    OrchestrationTimelineIssueCodeV1, validate_orchestration_timeline,
};
use symthaea_music_theory::pitch::{Pitch, PitchClass};
use symthaea_music_theory::rhythm::Duration;
use symthaea_music_theory::score::{Emphasis, PartId, Score, ScoreNote, VoiceRole};

fn note(midi: u8, onset: i64, duration: i64, role: VoiceRole) -> ScoreNote {
    ScoreNote {
        part: PartId(1),
        pitch: Pitch::from_midi(midi),
        onset: Duration::new(onset, 1),
        duration: Duration::new(duration, 1),
        velocity: 0.7,
        role,
        emphasis: Emphasis::Normal,
        section_intensity: 1.0,
    }
}

fn main() {
    let mut score = Score::new(Key::major(PitchClass::C), 96.0, 4);
    score.push(note(48, 0, 2, VoiceRole::Melody));
    score.push(note(60, 2, 2, VoiceRole::CounterMelody));

    let instruments = vec![
        InstrumentCapabilityV1 {
            id: "cello-solo".into(),
            kind: InstrumentKindV1::Cello,
            sounding_range: SoundingRangeV1 {
                lowest_midi: 36,
                highest_midi: 84,
            },
            max_polyphony: 1,
        },
        InstrumentCapabilityV1 {
            id: "horn-solo".into(),
            kind: InstrumentKindV1::FrenchHorn,
            sounding_range: SoundingRangeV1 {
                lowest_midi: 34,
                highest_midi: 77,
            },
            max_polyphony: 1,
        },
    ];

    // The digest is a shape-valid synthetic fixture, not a cryptographic
    // commitment to this score. Production callers must supply the digest
    // emitted by the score-demand/provenance layer.
    let plan = OrchestrationTimelineV1 {
        schema: ORCHESTRATION_TIMELINE_SCHEMA_V1.into(),
        score_digest_schema: ORCHESTRATION_SCORE_DIGEST_SCHEMA_V1.into(),
        score_sha256: "a".repeat(64),
        segments: vec![
            OrchestrationSegmentV1 {
                id: "cello-opening".into(),
                part_id: 1,
                instrument_id: "cello-solo".into(),
                start: RationalBeatV1 {
                    numerator: 0,
                    denominator: 1,
                },
                end: RationalBeatV1 {
                    numerator: 2,
                    denominator: 1,
                },
                musical_function: "opening-theme".into(),
                handoff: None,
            },
            OrchestrationSegmentV1 {
                id: "horn-response".into(),
                part_id: 1,
                instrument_id: "horn-solo".into(),
                start: RationalBeatV1 {
                    numerator: 2,
                    denominator: 1,
                },
                end: RationalBeatV1 {
                    numerator: 4,
                    denominator: 1,
                },
                musical_function: "transformed-theme".into(),
                handoff: Some(SegmentHandoffV1 {
                    from_segment_id: "cello-opening".into(),
                    kind: HandoffKindV1::Reattack,
                }),
            },
        ],
    };

    let accepted = validate_orchestration_timeline(&score, &plan, &instruments, &plan.score_sha256);
    assert!(accepted.valid, "valid handoff rejected: {:?}", accepted.issues);
    println!(
        "PASS synthetic cello -> horn handoff: {} persistent part, {} segments, {} instrument instances",
        accepted.part_count, accepted.segment_count, accepted.used_instrument_count
    );

    let mut crossing_score = score.clone();
    crossing_score.notes[0].duration = Duration::new(3, 1);
    let rejected = validate_orchestration_timeline(&crossing_score, &plan, &instruments, &plan.score_sha256);
    assert!(!rejected.valid, "note crossing handoff was accepted");
    assert!(
        rejected.issues.iter().any(|issue| {
            issue.code == OrchestrationTimelineIssueCodeV1::NoteCrossesHandoff
        }),
        "wrong rejection: {:?}",
        rejected.issues
    );
    println!("PASS negative control: note spanning the handoff is rejected");

    println!(
        "LIMIT: synthetic symbolic-contract mechanics only; no renderer, playability, or musical-quality claim."
    );
}
