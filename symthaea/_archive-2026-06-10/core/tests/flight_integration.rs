// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration test for the `multirotor` feature gate.
//!
//! Validates that the multirotor crate is properly re-exported and usable
//! from the main symthaea crate.
//!
//! Run: `cargo test --features multirotor --test flight_integration`

use symthaea::multirotor::formation::{FormationController, FormationShape};
use symthaea::multirotor::{
    FlightConfig, FlightController, FlightTrainer, PhysicsSimulator, QuadrotorHdcEncoder,
    SimplePhysicsSimulator,
};
use symthaea::symthaea_core::genesis::GenesisSeed;

#[test]
fn test_multirotor_feature_reexports() {
    // Verify all key types are accessible through the main crate
    let config = FlightConfig::default();
    assert!(config.motor_hz > 0.0);
    assert!(config.cognitive_hz > 0.0);
    assert_eq!(config.cognitive_interval(), 20);
}

#[test]
fn test_multirotor_basic_training() {
    let config = FlightConfig {
        num_episodes: 3,
        steps_per_episode: 100,
        ..FlightConfig::default()
    };
    let mut trainer = FlightTrainer::new(config);
    let metrics = trainer.train();

    assert_eq!(metrics.len(), 3);
    for m in &metrics {
        assert!(m.avg_position_error.is_finite());
        assert!(m.hover_fraction >= 0.0 && m.hover_fraction <= 1.0);
    }
}

#[test]
fn test_flight_controller_from_main_crate() {
    let genesis = GenesisSeed::from_phrase("integration-test");
    let config = FlightConfig::default();
    let mut controller = FlightController::new(&genesis, &config);
    let mut encoder = QuadrotorHdcEncoder::new(&genesis, config.num_levels);
    let mut sim = SimplePhysicsSimulator::new();

    // Run a few steps
    for _ in 0..10 {
        let state = sim.state().clone();
        let sensor_hv = encoder.encode(&state);
        let cmd = controller.forward(&sensor_hv, 0.002);

        assert!(cmd.thrust.is_finite());
        sim.step(&cmd, 0.002);
    }
}

#[test]
fn test_formation_from_main_crate() {
    let shape = FormationShape::Circle { radius: 0.5 };
    let ctrl = FormationController::new(0, shape, 4);
    let sp = ctrl.compute_setpoint();
    assert_eq!(sp.agent_id, 0);
    assert!(sp.offset[0].is_finite());
}
