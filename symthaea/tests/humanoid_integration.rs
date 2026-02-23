//! Integration test for the `humanoid` feature gate.
//!
//! Validates that the humanoid crate is properly re-exported and usable
//! from the main symthaea crate.
//!
//! Run: `cargo test --features humanoid --test humanoid_integration`

use symthaea::humanoid::{episode_reward, standing_reward, HumanoidTask};
use symthaea::humanoid::{
    HumanoidConfig, HumanoidController, HumanoidHdcEncoder, HumanoidPhysicsSimulator,
    HumanoidTrainer, PerturbationSchedule, SimpleHumanoidSimulator,
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
    assert!(perception.similarity(&perception) > 0.99, "Self-similarity should be ~1.0");

    // last_perception should match
    let last = bridge.last_perception().expect("should have last perception");
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
