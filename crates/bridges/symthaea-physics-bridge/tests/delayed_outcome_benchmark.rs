// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic paired benchmark for feature alarms with delayed outcome labels.
//!
//! This smoke benchmark deliberately contrasts benign covariate movement with a
//! harmful outcome change under the same alarm position, delay schedule, and
//! label-missingness pattern. It validates the outcome-monitor contract; it is
//! not a comparison against published SOTA implementations.

use symthaea_physics_bridge::delayed_outcome_validation::{
    DelayedOutcomeAssessment, DelayedOutcomeConfig, DelayedOutcomeMonitor, OutcomeAssociation,
    OutcomeDirection,
};

const ALARM_INDEX: u64 = 20;
const WINDOW_SIZE: u64 = 10;
const MIN_LABELED: usize = 8;
const MIN_COVERAGE: f64 = 0.8;
const MINIMUM_EFFECT: f64 = 0.5;

#[derive(Clone, Copy, Debug)]
enum World {
    BenignCovariateShift,
    HarmfulOutcomeShift,
}

#[derive(Debug)]
struct ReplayResult {
    final_assessment: DelayedOutcomeAssessment,
    first_confirmation_index: Option<u64>,
}

fn is_missing_label(observation_index: u64, missing_per_ten: u8) -> bool {
    observation_index % 10 < u64::from(missing_per_ten)
}

fn outcome_for(world: World, observation_index: u64) -> f64 {
    match world {
        World::BenignCovariateShift => 1.0,
        World::HarmfulOutcomeShift if observation_index >= ALARM_INDEX => 3.0,
        World::HarmfulOutcomeShift => 1.0,
    }
}

fn is_resolved(association: OutcomeAssociation) -> bool {
    matches!(
        association,
        OutcomeAssociation::Deteriorated
            | OutcomeAssociation::NoMaterialChange
            | OutcomeAssociation::Improved
    )
}

/// Replays event-time label delivery without exposing outcome values before
/// their arrival index. The current index's alarm is raised before that index's
/// label can be consumed by the monitor.
fn replay(
    world: World,
    label_delay: u64,
    missing_per_ten: u8,
) -> ReplayResult {
    let mut monitor = DelayedOutcomeMonitor::new(DelayedOutcomeConfig {
        window_size: WINDOW_SIZE,
        min_labeled_per_window: MIN_LABELED,
        min_labeled_fraction: MIN_COVERAGE,
        minimum_effect: MINIMUM_EFFECT,
        direction: OutcomeDirection::HigherIsWorse,
    })
    .expect("benchmark configuration is valid");

    let last_post_observation = ALARM_INDEX + WINDOW_SIZE - 1;
    let final_arrival_index = last_post_observation + label_delay;
    let mut first_confirmation_index = None;

    for current_index in 0..=final_arrival_index {
        monitor
            .observe(current_index)
            .expect("benchmark stream indices are consecutive");

        if current_index == ALARM_INDEX {
            monitor
                .raise_alarm(current_index)
                .expect("alarm is emitted at the current event-time cursor");
        }

        // Delayed labels enter only at their actual arrival cursor. In a real
        // online deployment, the receiver would supply this receipt as it arrives.
        for observation_index in 0..=current_index {
            if observation_index + label_delay != current_index {
                continue;
            }
            if is_missing_label(observation_index, missing_per_ten) {
                continue;
            }
            monitor
                .record_outcome(
                    observation_index,
                    current_index,
                    outcome_for(world, observation_index),
                )
                .expect("label receipt respects the event-time contract");
        }

        if current_index >= ALARM_INDEX && first_confirmation_index.is_none() {
            let assessment = monitor
                .assess(ALARM_INDEX)
                .expect("the alarm was registered");
            if is_resolved(assessment.association) {
                first_confirmation_index = Some(current_index);
            }
        }
    }

    ReplayResult {
        final_assessment: monitor
            .assess(ALARM_INDEX)
            .expect("the alarm was registered"),
        first_confirmation_index,
    }
}

#[test]
fn paired_benign_and_harmful_worlds_separate_alarm_from_outcome_confirmation() {
    let delays = [0_u64, 3, 10];
    // 0, 1, and 3 missing labels out of each ten-observation window.
    // The 30% regime intentionally fails the predeclared 80% coverage gate.
    let missing_regimes = [0_u8, 1, 3];

    let mut benign_confirmed_no_change = 0_usize;
    let mut benign_unresolved = 0_usize;
    let mut harmful_confirmed_deterioration = 0_usize;
    let mut harmful_unresolved = 0_usize;
    let mut naive_benign_false_confirmations = 0_usize;
    let mut cases = 0_usize;

    for delay in delays {
        for missing_per_ten in missing_regimes {
            let benign = replay(World::BenignCovariateShift, delay, missing_per_ten);
            let harmful = replay(World::HarmfulOutcomeShift, delay, missing_per_ten);
            cases += 2;

            let window_complete_index = ALARM_INDEX + WINDOW_SIZE - 1;
            let missing_leading_post_label = if missing_per_ten == 1 { 1 } else { 0 };
            let last_required_label_index =
                ALARM_INDEX + MIN_LABELED as u64 - 1 + delay + missing_leading_post_label;
            let expected_confirmation =
                Some(window_complete_index.max(last_required_label_index));
            match (missing_per_ten, benign.final_assessment.association) {
                (0 | 1, OutcomeAssociation::NoMaterialChange) => {
                    benign_confirmed_no_change += 1;
                    assert_eq!(benign.first_confirmation_index, expected_confirmation);
                }
                (3, OutcomeAssociation::InsufficientLabelCoverage) => {
                    benign_unresolved += 1;
                    assert_eq!(benign.first_confirmation_index, None);
                }
                (missing, observed) => {
                    panic!("unexpected benign state for missing={missing}/10: {observed:?}");
                }
            }

            match (missing_per_ten, harmful.final_assessment.association) {
                (0 | 1, OutcomeAssociation::Deteriorated) => {
                    harmful_confirmed_deterioration += 1;
                    assert_eq!(harmful.first_confirmation_index, expected_confirmation);
                }
                (3, OutcomeAssociation::InsufficientLabelCoverage) => {
                    harmful_unresolved += 1;
                    assert_eq!(harmful.first_confirmation_index, None);
                }
                (missing, observed) => {
                    panic!("unexpected harmful state for missing={missing}/10: {observed:?}");
                }
            }

            // The naive policy that equates every feature alarm with harm would
            // falsely confirm every benign world, independently of label delay.
            naive_benign_false_confirmations += 1;

            println!(
                "delay={delay:>2} missing={missing_per_ten}/10 \
                 benign={:?} harmful={:?} confirmation={:?}",
                benign.final_assessment.association,
                harmful.final_assessment.association,
                harmful.first_confirmation_index
            );
        }
    }

    println!(
        "paired_cases={cases} benign_no_change={benign_confirmed_no_change} \
         benign_unresolved={benign_unresolved}"
    );
    println!(
        "harmful_confirmed={harmful_confirmed_deterioration} \
         harmful_unresolved={harmful_unresolved} \
         naive_benign_false_confirmations={naive_benign_false_confirmations}"
    );

    assert_eq!(cases, 18);
    assert_eq!(benign_confirmed_no_change, 6);
    assert_eq!(benign_unresolved, 3);
    assert_eq!(harmful_confirmed_deterioration, 6);
    assert_eq!(harmful_unresolved, 3);
    assert_eq!(naive_benign_false_confirmations, 9);
}
