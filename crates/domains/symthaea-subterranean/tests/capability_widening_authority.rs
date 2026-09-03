// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Adversarial public-surface gates for capability-restoration authority.
//!
//! Restrictive transitions may be easy. Restorative transitions must not gain
//! authority merely because a downstream caller repeats a method or supplies a
//! boolean.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};
use symthaea_subterranean::actuator_isolation::{
    ActuatorIsolationPolicy, ActuatorIsolationSupervisor, PhysicalActuator,
};
use symthaea_subterranean::degraded_operations::{
    DegradedMode, DegradedObservation, DegradedOperationsSupervisor,
};
use symthaea_subterranean::embodiment::{MotorSafetyLevel, SubterraneanEmbodiment};
use symthaea_subterranean::types::{SubterraneanCommand, SubterraneanState};

fn healthy_recovery_observation() -> DegradedObservation {
    DegradedObservation {
        operator_link_fresh: true,
        control_loop_healthy: true,
        checkpoint_valid: true,
        reboot_count_in_window: 0,
        battery_ratio: 0.9,
        return_feasible: true,
        at_surface_or_service_bay: true,
    }
}

#[test]
fn repeated_same_observation_must_not_satisfy_recovery_dwell() {
    let mut supervisor = DegradedOperationsSupervisor::default();
    let mut failed = healthy_recovery_observation();
    failed.checkpoint_valid = false;
    supervisor.update(failed);
    assert_eq!(supervisor.mode(), DegradedMode::RecoveryRequired);

    // Replaying one unchanged observation is not 200 distinct healthy runtime
    // steps. A future authority gate may reject the raw boolean entirely; while
    // this API exists, repeated calls at one logical instant must not accumulate
    // temporal recovery credit.
    let healthy = healthy_recovery_observation();
    for _ in 0..1_000 {
        assert!(
            !supervisor.authorize_recovery_clear(healthy, true),
            "replaying identical evidence satisfied a temporal recovery dwell"
        );
        assert_eq!(
            supervisor.mode(),
            DegradedMode::RecoveryRequired,
            "repeated calls widened degraded authority without new runtime evidence"
        );
    }
}

#[test]
fn public_safety_override_clear_must_not_widen_without_release_authority() {
    let genesis = GenesisSeed::from_phrase("capability widening safety override");
    let mut embodiment = SubterraneanEmbodiment::new(&genesis);
    let thought = ContinuousHV::random(HDC_DIMENSION, 81_001);

    embodiment.set_safety_override(MotorSafetyLevel::Red);
    embodiment.step(&thought, 0.005, 0.95);
    assert_eq!(embodiment.safety_level(), MotorSafetyLevel::Red);

    // A bare public clear is not a verified release from the authority source
    // that imposed the restriction. Until such a release exists, the effective
    // safety floor must remain Red.
    embodiment.clear_safety_override();
    embodiment.step(&thought, 0.005, 0.95);

    assert_eq!(
        embodiment.safety_level(),
        MotorSafetyLevel::Red,
        "public clear_safety_override widened safety authority without a release capability"
    );
}

#[test]
fn bare_service_call_must_not_instantly_restore_isolated_actuator_authority() {
    let mut supervisor = ActuatorIsolationSupervisor::new(ActuatorIsolationPolicy {
        mismatch_penalty: 0.25,
        mismatch_streak_limit: 4,
        isolation_threshold: 0.2,
        ..Default::default()
    });
    let mut command = SubterraneanCommand::zero();
    command.set_left_track(1.0);
    let state = SubterraneanState::home();

    for _ in 0..4 {
        supervisor.observe(&command, &state, &state);
    }
    assert!(
        supervisor
            .report()
            .is_isolated(PhysicalActuator::LeftTrack),
        "test setup must first establish a latched isolated actuator"
    );

    // Service intent is not evidence that maintenance succeeded. Reintegration
    // should require verified service authority plus post-service qualification
    // (or an equivalent future protocol), not instantly set health to nominal.
    supervisor.service(PhysicalActuator::LeftTrack);

    assert!(
        supervisor
            .report()
            .is_isolated(PhysicalActuator::LeftTrack),
        "bare service call immediately restored isolated actuator authority"
    );
}
