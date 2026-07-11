// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cognition-causality acceptance — thought must actually steer the arm.
//!
//! The 2026-07-08 cognition-ablation experiment (examples/cognition_ablation.rs)
//! measured that through a shipped bridge with genesis-random weights,
//! semantically opposite intents produce ZERO task-axis separation — the
//! thought vector was a perturbation seed, not a control signal. This test is
//! the acceptance gate for the fix (trainer→bridge weight transfer +
//! intent-conditioned curriculum): after `train_intent_weights` +
//! `install_weights`, opposite intents MUST separate on the task axis, and
//! the same protocol on an untrained bridge must separate far less.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use symthaea_manipulator::embodiment::ManipulatorEmbodiment;
use symthaea_manipulator::training::{intent_hv, train_intent_weights};
use symthaea_manipulator::types::ManipulatorConfig;

const STEPS: usize = 100;
const DT: f32 = 0.002;
const PHI_GREEN: f64 = 0.9;

/// Drive a bridge with a constant intent at Green and return the base-joint
/// angle displacement (the task axis for reach-left vs reach-right).
fn base_joint_displacement(bridge: &mut ManipulatorEmbodiment, intent: &ContinuousHV) -> f64 {
    let start = bridge.body_state().joint_angles[0];
    for _ in 0..STEPS {
        let r = bridge.step(intent, DT, PHI_GREEN);
        assert!(r.success, "simulation must stay finite");
    }
    bridge.body_state().joint_angles[0] - start
}

/// Run the left/right protocol on a fresh pair of bridges (identical genesis,
/// optionally with trained weights installed) and return the task-axis
/// separation (left displacement − right displacement). Gravity and any other
/// intent-independent dynamics cancel in the difference.
fn task_axis_separation(
    config: &ManipulatorConfig,
    weights: Option<&symthaea_manipulator::controller::ControllerWeights>,
) -> f64 {
    let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
    let left = intent_hv(&genesis, "reach_left");
    let right = intent_hv(&genesis, "reach_right");

    let mut bridge_l = ManipulatorEmbodiment::new(&genesis);
    let mut bridge_r = ManipulatorEmbodiment::new(&genesis);
    if let Some(w) = weights {
        bridge_l.install_weights(w).expect("weights must fit");
        bridge_r.install_weights(w).expect("weights must fit");
    }

    let disp_l = base_joint_displacement(&mut bridge_l, &left);
    let disp_r = base_joint_displacement(&mut bridge_r, &right);
    disp_l - disp_r
}

#[test]
fn trained_bridge_gives_thought_task_axis_authority() {
    // Use a learning rate suited to the short supervised curriculum (the
    // default 0.0005 is tuned for long reflex episodes).
    let mut config = ManipulatorConfig::default();
    config.learning_rate = 0.01;

    let weights = train_intent_weights(&config, 40);

    let trained_sep = task_axis_separation(&config, Some(&weights));
    let untrained_sep = task_axis_separation(&config, None);

    println!(
        "task-axis separation: trained {trained_sep:.6} rad, untrained {untrained_sep:.6} rad"
    );

    // 1. Direction: the curriculum maps reach_left → positive base torque,
    //    so the trained separation must be positive.
    assert!(
        trained_sep > 0.0,
        "trained left-vs-right separation must follow the curriculum's sign, got {trained_sep}"
    );

    // 2. Magnitude: cognition must dominate the untrained chaos floor —
    //    the ablation experiment's finding was untrained separation ≈ 0.
    assert!(
        trained_sep.abs() > 5.0 * untrained_sep.abs(),
        "trained separation ({trained_sep}) must dominate the untrained chaos floor \
         ({untrained_sep}) — otherwise thought still isn't a control signal"
    );
}

#[test]
fn safety_gating_survives_trained_weights() {
    // Trained cognitive authority must NOT leak through the Red-tier
    // GravityHold: at Red the output stays thought-independent even with
    // intent-trained weights installed.
    let mut config = ManipulatorConfig::default();
    config.learning_rate = 0.01;
    let weights = train_intent_weights(&config, 40);
    let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);

    let mut bridge_l = ManipulatorEmbodiment::new(&genesis);
    let mut bridge_r = ManipulatorEmbodiment::new(&genesis);
    bridge_l.install_weights(&weights).unwrap();
    bridge_r.install_weights(&weights).unwrap();

    let left = intent_hv(&genesis, "reach_left");
    let right = intent_hv(&genesis, "reach_right");

    for _ in 0..50 {
        let rl = bridge_l.step(&left, DT, 0.05); // Red
        let rr = bridge_r.step(&right, DT, 0.05);
        assert_eq!(
            rl.control_effort, rr.control_effort,
            "at Red, trained intents must have zero motor authority (GravityHold)"
        );
    }
}
