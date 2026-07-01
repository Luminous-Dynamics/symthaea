// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Energy budget tests for the AUV fuel tank system.
//!
//! Verifies that thrust depletes energy, exhausted battery enters drift mode,
//! and energy accounting is monotonic and bounded.

use symthaea_auv::simulator::{AuvPhysicsSimulator, SimpleAuvSimulator};
use symthaea_auv::types::AuvCommand;

#[test]
fn test_energy_depletes_with_thrust() {
    let mut sim = SimpleAuvSimulator::new();
    let initial_fraction = sim.energy_fraction();
    assert!((initial_fraction - 1.0).abs() < 1e-10, "Should start full");

    // Full thrust for 100 steps
    let cmd = AuvCommand::forward(1.0);
    for _ in 0..100 {
        sim.step(&cmd, 0.01);
    }

    assert!(
        sim.energy_fraction() < initial_fraction,
        "Energy should deplete under thrust: {:.4}",
        sim.energy_fraction()
    );
    assert!(sim.energy_consumed() > 0.0, "Should have consumed energy");
}

#[test]
fn test_zero_thrust_minimal_energy_use() {
    let mut sim = SimpleAuvSimulator::new();
    let cmd = AuvCommand::zero();

    for _ in 0..1000 {
        sim.step(&cmd, 0.01);
    }

    // Zero thrust should consume negligible energy
    assert!(
        sim.energy_fraction() > 0.99,
        "Zero thrust should barely deplete: {:.6}",
        sim.energy_fraction()
    );
}

#[test]
fn test_energy_exhaustion_enters_drift_mode() {
    // Small battery that will run out quickly
    let mut sim = SimpleAuvSimulator::with_energy_budget(10.0); // Only 10 joules

    let cmd = AuvCommand::forward(1.0);
    let mut exhausted_step = None;

    for i in 0..500 {
        sim.step(&cmd, 0.01);
        if sim.energy_exhausted() && exhausted_step.is_none() {
            exhausted_step = Some(i);
        }
        assert!(sim.state().is_finite(), "NaN at step {i}");
    }

    assert!(
        exhausted_step.is_some(),
        "10J battery should exhaust under full thrust"
    );
    assert!(sim.energy_exhausted());

    // After exhaustion, vehicle should drift (velocity decays via drag)
    // Run more steps and verify speed decreases
    let speed_at_exhaustion = sim.state().speed();
    for _ in 0..200 {
        sim.step(&cmd, 0.01); // Command still says thrust, but battery is dead
    }
    let speed_after_drift = sim.state().speed();

    assert!(
        speed_after_drift <= speed_at_exhaustion + 0.01,
        "Speed should decrease in drift: at_exhaust={speed_at_exhaustion:.4}, after={speed_after_drift:.4}"
    );
}

#[test]
fn test_energy_consumed_monotonic() {
    let mut sim = SimpleAuvSimulator::new();
    let cmd = AuvCommand::forward(0.5);
    let mut prev_consumed = 0.0;

    for i in 0..200 {
        sim.step(&cmd, 0.01);
        let consumed = sim.energy_consumed();
        assert!(
            consumed >= prev_consumed,
            "Energy consumed should be monotonic at step {i}: {consumed} < {prev_consumed}"
        );
        prev_consumed = consumed;
    }
}

#[test]
fn test_energy_fraction_bounded() {
    let mut sim = SimpleAuvSimulator::with_energy_budget(100.0);
    let cmd = AuvCommand::forward(1.0);

    for i in 0..1000 {
        sim.step(&cmd, 0.01);
        let frac = sim.energy_fraction();
        assert!(
            frac >= 0.0 && frac <= 1.0,
            "Energy fraction should be in [0,1] at step {i}: {frac}"
        );
    }
}

#[test]
fn test_reset_restores_energy() {
    let mut sim = SimpleAuvSimulator::new();
    let cmd = AuvCommand::forward(1.0);

    for _ in 0..100 {
        sim.step(&cmd, 0.01);
    }
    assert!(sim.energy_fraction() < 1.0);

    sim.reset(10.0);
    assert!(
        (sim.energy_fraction() - 1.0).abs() < 1e-10,
        "Reset should restore full energy"
    );
    assert!(
        sim.energy_consumed() < 1e-10,
        "Reset should zero consumed energy"
    );
}

#[test]
fn test_higher_thrust_depletes_faster() {
    let mut sim_low = SimpleAuvSimulator::new();
    let mut sim_high = SimpleAuvSimulator::new();

    let low_cmd = AuvCommand::forward(0.3);
    let high_cmd = AuvCommand::forward(1.0);

    for _ in 0..200 {
        sim_low.step(&low_cmd, 0.01);
        sim_high.step(&high_cmd, 0.01);
    }

    assert!(
        sim_high.energy_consumed() > sim_low.energy_consumed(),
        "Higher thrust should consume more energy: high={:.2} vs low={:.2}",
        sim_high.energy_consumed(),
        sim_low.energy_consumed()
    );
}
