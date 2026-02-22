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
