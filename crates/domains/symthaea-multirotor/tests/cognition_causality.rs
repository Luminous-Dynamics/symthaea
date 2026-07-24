// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cognition-causality acceptance — thought must actually steer the drone.
//!
//! The 2026-07-08 cognition-ablation experiment (examples/cognition_ablation.rs)
//! measured that through a shipped bridge with genesis-random weights,
//! semantically opposite intents (climb vs descend) produce ZERO task-axis
//! separation — the thought vector was a perturbation seed, not a control
//! signal. This test is the acceptance gate for the fix (trainer→bridge
//! controller transfer + intent-conditioned curriculum): after
//! `train_intent_controller` + `FlightEmbodiment::with_controller`, opposite
//! intents MUST separate on the vertical task axis, and the same protocol on
//! an untrained bridge must separate far less.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use symthaea_multirotor::embodiment::FlightEmbodiment;
use symthaea_multirotor::simulator::PhysicsSimulator;
use symthaea_multirotor::training::{intent_hv, train_intent_controller};
use symthaea_multirotor::types::FlightConfig;

const STEPS: usize = 250; // 0.5 s at the 500 Hz reflex rate
const DT: f32 = 0.002;
const PHI_GREEN: f64 = 0.9;

/// Drive a bridge with a constant intent at Green and return the altitude
/// displacement (the task axis for climb vs descend).
fn altitude_displacement(bridge: &mut FlightEmbodiment, intent: &ContinuousHV) -> f64 {
    let start = bridge.simulator().state().position[2];
    for _ in 0..STEPS {
        let r = bridge.step(intent, DT, PHI_GREEN);
        assert!(r.success, "simulation must stay finite");
    }
    bridge.simulator().state().position[2] - start
}

/// Run the climb/descend protocol on a fresh pair of bridges and return the
/// task-axis separation (climb displacement − descend displacement).
/// Gravity/drag and any other intent-independent dynamics cancel in the
/// difference.
fn task_axis_separation(config: &FlightConfig, trained: bool) -> f64 {
    let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
    let climb = intent_hv(&genesis, "climb");
    let descend = intent_hv(&genesis, "descend");

    let make_bridge = || {
        if trained {
            FlightEmbodiment::with_controller(&genesis, train_intent_controller(config, 40))
        } else {
            FlightEmbodiment::new(&genesis)
        }
    };

    let disp_climb = altitude_displacement(&mut make_bridge(), &climb);
    let disp_descend = altitude_displacement(&mut make_bridge(), &descend);
    disp_climb - disp_descend
}

#[test]
fn trained_bridge_gives_thought_task_axis_authority() {
    let config = FlightConfig::default();

    let trained_sep = task_axis_separation(&config, true);
    let untrained_sep = task_axis_separation(&config, false);

    println!("task-axis separation: trained {trained_sep:.4} m, untrained {untrained_sep:.4} m");

    // 1. Direction: climb must end above descend.
    assert!(
        trained_sep > 0.0,
        "trained climb-vs-descend separation must follow the curriculum's sign, \
         got {trained_sep}"
    );

    // 2. Magnitude: cognition must dominate the untrained chaos floor — the
    //    ablation experiment's finding was untrained separation ≈ 0.
    assert!(
        trained_sep.abs() > 5.0 * untrained_sep.abs(),
        "trained separation ({trained_sep}) must dominate the untrained chaos \
         floor ({untrained_sep}) — otherwise thought still isn't a control signal"
    );
}

#[test]
fn safety_gating_survives_trained_controller() {
    // Trained cognitive authority must NOT leak through the Red-tier
    // fallback chain: at Red, opposite trained intents must produce an
    // identical trajectory (thrust comes from the fallback state machine,
    // moments are zeroed).
    let config = FlightConfig::default();
    let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
    let controller = train_intent_controller(&config, 40);

    let mut bridge_a = FlightEmbodiment::with_controller(&genesis, controller.clone());
    let mut bridge_b = FlightEmbodiment::with_controller(&genesis, controller);
    let climb = intent_hv(&genesis, "climb");
    let descend = intent_hv(&genesis, "descend");

    for _ in 0..100 {
        let ra = bridge_a.step(&climb, DT, 0.05); // Red
        let rb = bridge_b.step(&descend, DT, 0.05);
        assert_eq!(
            ra.control_effort, rb.control_effort,
            "at Red, trained intents must have zero motor authority (fallback chain)"
        );
    }
    let alt_a = bridge_a.simulator().state().position[2];
    let alt_b = bridge_b.simulator().state().position[2];
    assert_eq!(
        alt_a, alt_b,
        "at Red, trajectories must be identical regardless of intent"
    );
}
