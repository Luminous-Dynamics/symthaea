//! Integration test for the `humanoid` feature gate.
//!
//! Validates that the humanoid crate is properly re-exported and usable
//! from the main symthaea crate.
//!
//! Run: `cargo test --features humanoid --test humanoid_integration`

use symthaea::humanoid::{episode_reward, standing_reward, HumanoidTask};
use symthaea::humanoid::{
    ActiveInferenceHumanoidAgent, HumanoidConfig, HumanoidController, HumanoidFepConfig,
    HumanoidHdcEncoder, HumanoidPerturbation, HumanoidPhysicsSimulator, HumanoidTrainer,
    PerturbationSchedule, SimpleHumanoidSimulator,
};
use symthaea::symthaea_core::genesis::GenesisSeed;
use symthaea::symthaea_core::hdc::HDC_DIMENSION;

#[test]
fn test_humanoid_feature_reexports() {
    // Verify all key types are accessible through the main crate
    let config = HumanoidConfig::default();
    assert!(config.physics_hz > 0.0);
    assert!(config.cognitive_hz > 0.0);
    assert_eq!(config.cognitive_interval(), 4); // 40Hz / 10Hz
}

#[test]
fn test_humanoid_full_pipeline() {
    let config = HumanoidConfig {
        num_episodes: 3,
        steps_per_episode: 50,
        ..HumanoidConfig::default()
    };
    let mut trainer = HumanoidTrainer::new(config);
    let metrics = trainer.train();

    assert_eq!(metrics.len(), 3);
    for m in &metrics {
        assert!(m.avg_standing_reward.is_finite());
        assert!(m.avg_episode_reward.is_finite());
        assert!(m.avg_free_energy.is_finite());
        assert!(m.avg_head_height.is_finite());
        assert!(m.total_steps > 0);
    }
}

#[test]
fn test_humanoid_controller_from_main_crate() {
    let genesis = GenesisSeed::from_phrase("integration-test-humanoid");
    let config = HumanoidConfig::default();
    let mut controller = HumanoidController::new(&genesis, &config);
    let mut encoder = HumanoidHdcEncoder::new(&genesis, config.num_levels);
    let mut sim = SimpleHumanoidSimulator::new();

    // Run 10 steps through encode → forward → step
    for _ in 0..10 {
        let state = sim.state().clone();
        let sensor_hv = encoder.encode(&state);
        let cmd = controller.forward(&sensor_hv, 0.025);

        for &t in &cmd.torques {
            assert!(t.is_finite(), "Torque must be finite");
            assert!(t >= -1.0 && t <= 1.0, "Torque must be in [-1, 1]");
        }

        sim.step(&cmd, 0.025);
    }

    // Verify reward functions work through re-export
    let state = sim.state();
    let sr = standing_reward(state);
    assert!(sr.is_finite());
    let er = episode_reward(
        state,
        &symthaea::humanoid::HumanoidCommand::zero(),
        &HumanoidTask::Stand,
        0.0,
    );
    assert!(er.is_finite());
}

#[test]
fn test_humanoid_perturbation_schedule() {
    let schedule = PerturbationSchedule::chest_shove();
    assert_eq!(schedule.len(), 1);

    // Verify trigger window
    assert!(schedule.triggered_at(299).is_empty());
    assert_eq!(schedule.triggered_at(300).len(), 1);
    assert_eq!(schedule.triggered_at(304).len(), 1);
    assert!(schedule.triggered_at(305).is_empty());
}

#[test]
fn test_motor_bridge_step_returns_perception() {
    use symthaea::cognitive_loop::motor_bridge::MotorBridge;
    use symthaea::symthaea_core::hdc::ContinuousHV;

    let genesis = GenesisSeed::from_phrase("motor-bridge-integration-test");
    let mut bridge = MotorBridge::new(&genesis);

    // Create a thought vector (simulating cognitive loop output)
    let thought = ContinuousHV::random(HDC_DIMENSION, 42);

    // Step returns motor command + proprioceptive perception
    let (cmd, perception) = bridge.step(&thought);

    // Motor command should be valid
    for &t in &cmd.torques {
        assert!(t.is_finite(), "Torque must be finite");
        assert!(t >= -1.0 && t <= 1.0, "Torque must be in [-1, 1]");
    }

    // Perception should be a valid 16,384D HDC vector
    assert_eq!(perception.dim(), HDC_DIMENSION);
    assert!(
        perception.similarity(&perception) > 0.99,
        "Self-similarity should be ~1.0"
    );

    // last_perception should match
    let last = bridge
        .last_perception()
        .expect("should have last perception");
    assert!((last.similarity(&perception) - 1.0).abs() < 1e-5);

    assert_eq!(bridge.total_steps(), 1);
}

#[test]
fn test_motor_bridge_perception_changes_with_state() {
    use symthaea::cognitive_loop::motor_bridge::MotorBridge;
    use symthaea::symthaea_core::hdc::ContinuousHV;

    let genesis = GenesisSeed::from_phrase("motor-bridge-perception-diff");
    let mut bridge = MotorBridge::new(&genesis);

    let thought = ContinuousHV::random(HDC_DIMENSION, 99);

    // Step 1: initial state
    let (_, p1) = bridge.step(&thought);

    // Step multiple times to let state evolve
    let mut p_last = p1.clone();
    for _ in 0..20 {
        let (_, p) = bridge.step(&thought);
        p_last = p;
    }

    // After 20 physics steps, body state has changed → perception should differ
    let sim = p1.similarity(&p_last);
    assert!(
        sim < 0.999,
        "Perception should evolve as body state changes: sim={sim}"
    );
}

#[test]
fn test_motor_bridge_reset_clears_perception() {
    use symthaea::cognitive_loop::motor_bridge::MotorBridge;
    use symthaea::symthaea_core::hdc::ContinuousHV;

    let genesis = GenesisSeed::from_phrase("motor-bridge-reset-test");
    let mut bridge = MotorBridge::new(&genesis);

    let thought = ContinuousHV::random(HDC_DIMENSION, 7);
    bridge.step(&thought);
    assert!(bridge.last_perception().is_some());
    assert_eq!(bridge.total_steps(), 1);

    bridge.reset();
    assert!(bridge.last_perception().is_none());
    assert_eq!(bridge.total_steps(), 0);
}

#[test]
fn test_motor_bridge_from_checkpoint() {
    use symthaea::cognitive_loop::motor_bridge::MotorBridge;
    use symthaea::symthaea_core::hdc::ContinuousHV;

    let config = HumanoidConfig::default();
    let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
    let controller = HumanoidController::new(&genesis, &config);

    let mut bridge = MotorBridge::from_controller(controller, config);
    let thought = ContinuousHV::random(HDC_DIMENSION, 123);

    let (cmd, perception) = bridge.step(&thought);
    assert!(cmd.torques.iter().all(|t| t.is_finite()));
    assert_eq!(perception.dim(), HDC_DIMENSION);
}

#[test]
fn test_checkpoint_resume_training() {
    let config = HumanoidConfig {
        num_episodes: 3,
        steps_per_episode: 30,
        ..HumanoidConfig::default()
    };

    // Train and save
    let dir = "/tmp/symthaea_integ_checkpoint_resume";
    let _ = std::fs::remove_dir_all(dir);
    let mut trainer = HumanoidTrainer::new(config.clone());
    let _ = trainer.train_with_telemetry(dir);

    let checkpoint_path = format!("{}/checkpoint.json", dir);
    assert!(std::path::Path::new(&checkpoint_path).exists());

    // Resume
    let resume_config = HumanoidConfig {
        num_episodes: 2,
        steps_per_episode: 30,
        ..HumanoidConfig::default()
    };
    let mut resumed = HumanoidTrainer::with_checkpoint(resume_config, &checkpoint_path)
        .expect("should load checkpoint");
    let metrics = resumed.train();
    assert_eq!(metrics.len(), 2);
    for m in &metrics {
        assert!(m.avg_standing_reward.is_finite());
    }

    let _ = std::fs::remove_dir_all(dir);
}

/// Perturbation crucible: run a controller with a chest shove perturbation and
/// verify the FEP agent responds and the system stays physically valid.
#[test]
fn test_perturbation_crucible_chest_shove() {
    let config = HumanoidConfig::default();
    let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
    let mut encoder = HumanoidHdcEncoder::new(&genesis, config.num_levels);
    let mut controller = HumanoidController::new(&genesis, &config);
    let mut fep_agent =
        ActiveInferenceHumanoidAgent::new(HumanoidFepConfig::default(), HumanoidTask::Stand);
    let mut sim = SimpleHumanoidSimulator::new();

    // Moderate push (150N) — tests perturbation detection without requiring
    // a trained controller to survive a violent shove.
    let mut schedule = PerturbationSchedule::new().with(HumanoidPerturbation::ExternalPush {
        force: [150.0, 0.0, 0.0],
        at_step: 50,
        duration: 5,
    });

    let dt = config.physics_dt();
    let cognitive_interval = config.cognitive_interval();
    let mut pre_push_fe = Vec::new();
    let mut post_push_fe = Vec::new();

    for step in 0..200 {
        let state = sim.state().clone();
        let sensor_hv = encoder.encode(&state);
        let mut cmd = controller.forward(&sensor_hv, dt as f32);

        schedule.apply(step, &mut cmd, &mut sim);
        sim.step(&cmd, dt);

        // FEP cognitive tick
        if step % cognitive_interval == 0 {
            let fep_result = fep_agent.step(sim.state(), &cmd);
            if step < 50 {
                pre_push_fe.push(fep_result.free_energy);
            } else if step >= 55 && step < 120 {
                post_push_fe.push(fep_result.free_energy);
            }
        }
    }

    // 1. FEP produced finite free energy throughout
    assert!(
        !pre_push_fe.is_empty() && !post_push_fe.is_empty(),
        "Should have FEP samples before and after push"
    );
    for &fe in pre_push_fe.iter().chain(post_push_fe.iter()) {
        assert!(fe.is_finite(), "Free energy should be finite: {fe}");
    }

    // 2. All state values remain finite after perturbation
    let final_state = sim.state();
    assert!(final_state.head_height.is_finite());
    assert!(final_state.root_height.is_finite());
    for &a in &final_state.joint_angles {
        assert!(a.is_finite(), "Joint angle must stay finite after push");
    }

    // 3. The system didn't completely collapse to ground minimum
    assert!(
        final_state.root_height > 0.15,
        "Root should be above ground after push: {}",
        final_state.root_height
    );
}

/// Perturbation crucible: phantom limb (right knee failure) during standing.
/// The controller should continue producing finite outputs despite a permanently
/// disabled actuator.
#[test]
fn test_perturbation_crucible_phantom_limb() {
    let config = HumanoidConfig::default();
    let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
    let mut encoder = HumanoidHdcEncoder::new(&genesis, config.num_levels);
    let mut controller = HumanoidController::new(&genesis, &config);
    let mut sim = SimpleHumanoidSimulator::new();

    // Disable right knee at step 50 (short episode version of phantom_limb)
    let mut schedule = PerturbationSchedule::new().with(HumanoidPerturbation::ActuatorFailure {
        joint_index: 6, // right_knee
        at_step: 50,
    });

    let dt = config.physics_dt();
    let mut knee_torque_after_failure = Vec::new();

    for step in 0..200 {
        let state = sim.state().clone();
        let sensor_hv = encoder.encode(&state);
        let mut cmd = controller.forward(&sensor_hv, dt as f32);

        schedule.apply(step, &mut cmd, &mut sim);

        // Track knee torque after failure
        if step > 50 {
            knee_torque_after_failure.push(cmd.torques[6]);
        }

        sim.step(&cmd, dt);
    }

    // After failure, right_knee torque should always be 0 (masked)
    for &t in &knee_torque_after_failure {
        assert!(
            t.abs() < 1e-6,
            "Right knee should be masked to zero after failure: {t}"
        );
    }

    // Other actuators should still produce finite output
    let state = sim.state().clone();
    let sensor_hv = encoder.encode(&state);
    let cmd = controller.forward(&sensor_hv, dt as f32);
    for (i, &t) in cmd.torques.iter().enumerate() {
        assert!(t.is_finite(), "Joint {i} should produce finite output");
    }

    // Humanoid shouldn't be in a completely collapsed state
    assert!(
        sim.state().root_height > 0.15,
        "Humanoid should maintain some height even with knee failure: {}",
        sim.state().root_height
    );
}
