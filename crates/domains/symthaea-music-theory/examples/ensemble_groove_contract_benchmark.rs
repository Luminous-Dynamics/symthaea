// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Offline, deterministic known-answer harness for ensemble groove contracts.
//!
//! Run with:
//! cargo run -p symthaea-music-theory --example ensemble_groove_contract_benchmark
//!
//! All patterns are synthetic software fixtures. This harness checks exact
//! mechanics for cross-cycle interlock, counterfactual collisions, inter-onset
//! space and directed cue/state/response. It is not a genre-recognition test,
//! audio-quality benchmark or cultural-authenticity evaluation.

use serde_json::json;
use symthaea_music_theory::ensemble_cues::{
    CueEventV1, CueResponseEventV1, CueStateChangeEventV1, CueStateV1,
    CueTransitionRuleV1, EnsembleCueGraphV1, EnsembleCueSequenceV1,
    ENSEMBLE_CUE_GRAPH_SCHEMA_V1, ENSEMBLE_CUE_SEQUENCE_SCHEMA_V1,
    verify_ensemble_cues,
};
use symthaea_music_theory::ensemble_groove::{
    EnsembleGrooveV1, GrooveConstraintKindV1, GrooveConstraintV1, GrooveEventV1,
    GrooveLayerV1, ENSEMBLE_GROOVE_SCHEMA_V1, analyze_ensemble_groove,
};
use symthaea_music_theory::groove_space::analyze_groove_space;

fn cross_cycle_fixture(independent_onsets: [u32; 2]) -> EnsembleGrooveV1 {
    EnsembleGrooveV1 {
        schema: ENSEMBLE_GROOVE_SCHEMA_V1.into(),
        ticks_per_beat: 24,
        layers: vec![
            GrooveLayerV1 {
                id: "timekeeper".into(),
                role: "reference_pulse".into(),
                cycle_ticks: 48,
                events: [0, 12, 24, 36]
                    .into_iter()
                    .map(|onset_tick| GrooveEventV1 {
                        onset_tick,
                        accent: if onset_tick == 0 { 3 } else { 1 },
                    })
                    .collect(),
            },
            GrooveLayerV1 {
                id: "independent_line".into(),
                role: "counter_rhythm".into(),
                cycle_ticks: 32,
                events: independent_onsets
                    .into_iter()
                    .map(|onset_tick| GrooveEventV1 {
                        onset_tick,
                        accent: 1,
                    })
                    .collect(),
            },
        ],
        constraints: vec![
            GrooveConstraintV1 {
                left_layer_id: "timekeeper".into(),
                right_layer_id: "independent_line".into(),
                rule: GrooveConstraintKindV1::NoCoincidentAttacks,
            },
            GrooveConstraintV1 {
                left_layer_id: "timekeeper".into(),
                right_layer_id: "independent_line".into(),
                rule: GrooveConstraintKindV1::MinimumCircularSeparation {
                    minimum_ticks: 2,
                },
            },
        ],
    }
}

fn one_layer(events: &[u32]) -> EnsembleGrooveV1 {
    EnsembleGrooveV1 {
        schema: ENSEMBLE_GROOVE_SCHEMA_V1.into(),
        ticks_per_beat: 24,
        layers: vec![GrooveLayerV1 {
            id: "primary".into(),
            role: "primary_rhythm".into(),
            cycle_ticks: 96,
            events: events
                .iter()
                .copied()
                .map(|onset_tick| GrooveEventV1 {
                    onset_tick,
                    accent: 1,
                })
                .collect(),
        }],
        constraints: vec![],
    }
}

fn cue_graph() -> EnsembleCueGraphV1 {
    EnsembleCueGraphV1 {
        schema: ENSEMBLE_CUE_GRAPH_SCHEMA_V1.into(),
        initial_state: "full".into(),
        states: vec![
            CueStateV1 { id: "full".into() },
            CueStateV1 { id: "soft".into() },
        ],
        layer_ids: vec!["captain_guitar".into(), "talking_drum".into()],
        transitions: vec![CueTransitionRuleV1 {
            from_state: "full".into(),
            cue_type: "soften".into(),
            source_layer_id: "captain_guitar".into(),
            to_state: "soft".into(),
            response_layer_id: "talking_drum".into(),
            response_action: "lower_intensity".into(),
            minimum_response_delay_ticks: 2,
            maximum_response_delay_ticks: 12,
            maximum_state_change_latency_ticks: 4,
        }],
    }
}

fn cue_sequence(include_response: bool) -> EnsembleCueSequenceV1 {
    EnsembleCueSequenceV1 {
        schema: ENSEMBLE_CUE_SEQUENCE_SCHEMA_V1.into(),
        horizon_ticks: 96,
        cues: vec![CueEventV1 {
            id: "cue_001".into(),
            cue_type: "soften".into(),
            source_layer_id: "captain_guitar".into(),
            onset_tick: 16,
        }],
        responses: if include_response {
            vec![CueResponseEventV1 {
                cue_id: "cue_001".into(),
                layer_id: "talking_drum".into(),
                action: "lower_intensity".into(),
                onset_tick: 20,
            }]
        } else {
            vec![]
        },
        state_changes: vec![CueStateChangeEventV1 {
            cue_id: "cue_001".into(),
            to_state: "soft".into(),
            onset_tick: 18,
        }],
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let valid_interlock = analyze_ensemble_groove(&cross_cycle_fixture([10, 26])).map_err(|issues| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{issues:?}")))?;
    assert_eq!(valid_interlock.supercycle_ticks, 96);
    assert!(valid_interlock.all_constraints_passed);
    assert_eq!(valid_interlock.pair_analyses[0].coincident_attack_pairs, 0);
    assert_eq!(
        valid_interlock.pair_analyses[0].minimum_circular_separation_ticks,
        Some(2)
    );

    let collision_control = analyze_ensemble_groove(&cross_cycle_fixture([8, 24])).map_err(|issues| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{issues:?}")))?;
    assert_eq!(
        collision_control.pair_analyses[0].coincident_attack_pairs,
        2
    );
    assert!(!collision_control.all_constraints_passed);

    let sparse_space = analyze_groove_space(&one_layer(&[0, 48]), 24).map_err(|issues| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{issues:?}")))?;
    let dense_space =
        analyze_groove_space(&one_layer(&[0, 12, 24, 36, 48, 60, 72, 84]), 24).map_err(|issues| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{issues:?}")))?;
    assert_eq!(
        sparse_space.layers[0].long_gap_span_basis_points,
        10_000
    );
    assert_eq!(
        dense_space.layers[0].long_gap_span_basis_points,
        0
    );
    assert_eq!(
        sparse_space.layers[0].inter_onset_gaps_ticks.iter().sum::<u64>(),
        96
    );
    assert_eq!(
        dense_space.layers[0].inter_onset_gaps_ticks.iter().sum::<u64>(),
        96
    );

    let cue_success = verify_ensemble_cues(&cue_graph(), &cue_sequence(true)).map_err(|issues| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{issues:?}")))?;
    let cue_missing_response = verify_ensemble_cues(&cue_graph(), &cue_sequence(false)).map_err(|issues| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{issues:?}")))?;
    assert!(cue_success.all_transitions_passed);
    assert!(!cue_missing_response.all_transitions_passed);
    assert_eq!(cue_success.final_state, "soft");
    assert_eq!(cue_missing_response.final_state, "soft");

    let report = json!({
        "schema": "melothaea-groove-contract-benchmark-v1",
        "status": "synthetic_known_answer_mechanics_only",
        "claim_limits": [
            "not_genre_recognition",
            "not_audio_quality",
            "not_authenticity",
            "not_listener_preference",
            "not_real_recording_analysis"
        ],
        "cross_cycle_interlock": {
            "passed": valid_interlock.all_constraints_passed,
            "supercycle_ticks": valid_interlock.supercycle_ticks,
            "pairwise_comparisons": valid_interlock.pair_analyses[0].possible_attack_pairs,
            "coincident_pairs": valid_interlock.pair_analyses[0].coincident_attack_pairs,
            "minimum_separation_ticks": valid_interlock.pair_analyses[0].minimum_circular_separation_ticks
        },
        "collision_negative_control": {
            "passed_as_expected_failure": !collision_control.all_constraints_passed,
            "coincident_pairs": collision_control.pair_analyses[0].coincident_attack_pairs
        },
        "inter_onset_space": {
            "sparse_long_gap_span_basis_points": sparse_space.layers[0].long_gap_span_basis_points,
            "dense_long_gap_span_basis_points": dense_space.layers[0].long_gap_span_basis_points,
            "sparse_gaps_ticks": sparse_space.layers[0].inter_onset_gaps_ticks,
            "dense_gaps_ticks": dense_space.layers[0].inter_onset_gaps_ticks
        },
        "cue_state_response": {
            "complete_chain_passed": cue_success.all_transitions_passed,
            "complete_chain_final_state": cue_success.final_state,
            "missing_response_failed_as_expected": !cue_missing_response.all_transitions_passed,
            "missing_response_kept_state_change": cue_missing_response.final_state
        }
    });

    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
