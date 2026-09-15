// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Confirmatory-grid leakage controls for PARADOX-002A.
//!
//! These tests bind two finite-grid properties that matter for later behavioral
//! interpretation:
//! 1. the matched noncausal auxiliary cue cannot deterministically reveal the
//!    first claim polarity within any confirmatory seed; and
//! 2. the causal context cue remains aligned between the visible-context C3
//!    fixture and hidden-context C6 fixture generated from the same latent trial.

use symthaea_paradox_fixtures::{
    CONFIRMATORY_SEEDS, CONFIRMATORY_TRIALS_PER_CONDITION, ClaimPolarity, Condition,
    FixtureGenerator,
};

const NONCAUSAL_CONDITIONS: [Condition; 5] = [
    Condition::CoherentControl,
    Condition::SurpriseOnly,
    Condition::TransientConflict,
    Condition::PersistentIrreducible,
    Condition::SelfReferentialConflict,
];

fn polarity_index(polarity: ClaimPolarity) -> usize {
    match polarity {
        ClaimPolarity::Proposition => 0,
        ClaimPolarity::Negation => 1,
    }
}

#[test]
fn matched_auxiliary_cue_has_no_deterministic_polarity_mapping_per_confirmatory_seed() {
    for condition in NONCAUSAL_CONDITIONS {
        for seed in CONFIRMATORY_SEEDS {
            let mut observed = Vec::with_capacity(CONFIRMATORY_TRIALS_PER_CONDITION);
            let mut distinct_cues = Vec::with_capacity(2);

            for trial_index in 0..CONFIRMATORY_TRIALS_PER_CONDITION {
                let fixture = FixtureGenerator::generate(condition, seed, trial_index)
                    .expect("confirmatory fixture must generate");
                let events = fixture.agent_view().events();
                let first_polarity = events[0]
                    .polarity()
                    .expect("first event must carry a claim polarity");
                let auxiliary = &events[3];

                assert_eq!(auxiliary.polarity(), None);
                assert_eq!(auxiliary.visible_context(), None);

                let cue = auxiliary.surface_cue();
                if !distinct_cues.contains(&cue) {
                    distinct_cues.push(cue);
                }
                observed.push((first_polarity, cue));
            }

            assert_eq!(
                distinct_cues.len(),
                2,
                "matched auxiliary cue must expose exactly two surface values for {condition:?}, seed {seed}"
            );

            let mut cells = [[0usize; 2]; 2];
            for (polarity, cue) in observed {
                let cue_index = distinct_cues
                    .iter()
                    .position(|candidate| *candidate == cue)
                    .expect("observed cue must be indexed");
                cells[polarity_index(polarity)][cue_index] += 1;
            }

            for (polarity_index, row) in cells.iter().enumerate() {
                for (cue_index, count) in row.iter().enumerate() {
                    assert!(
                        *count > 0,
                        "missing polarity/cue combination for {condition:?}, seed {seed}, polarity index {polarity_index}, cue index {cue_index}"
                    );
                }
            }
        }
    }
}

#[test]
fn c3_and_c6_share_causal_context_cue_without_sharing_visibility() {
    for seed in CONFIRMATORY_SEEDS {
        for trial_index in 0..CONFIRMATORY_TRIALS_PER_CONDITION {
            let c3 = FixtureGenerator::generate(Condition::PersistentResolvable, seed, trial_index)
                .expect("C3 confirmatory fixture must generate");
            let c6 = FixtureGenerator::generate(Condition::OntologyFailure, seed, trial_index)
                .expect("C6 confirmatory fixture must generate");

            let c3_auxiliary = &c3.agent_view().events()[3];
            let c6_auxiliary = &c6.agent_view().events()[3];

            assert_eq!(c3_auxiliary.surface_cue(), c6_auxiliary.surface_cue());
            assert!(c3_auxiliary.visible_context().is_some());
            assert!(c6_auxiliary.visible_context().is_none());
        }
    }
}
