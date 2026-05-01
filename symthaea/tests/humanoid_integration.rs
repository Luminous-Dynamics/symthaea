// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for the `symthaea-humanoid` crate.
//!
//! Covers: encoder, controller, simulator, gait analyzer, reward functions,
//! FEP agent, perturbations, morphological transfer, and training pipeline.
//!
//! Run: `cargo test --features humanoid --test humanoid_integration`

use symthaea::humanoid::transfer::{ChannelMapping, MorphologicalTransfer};
use symthaea::humanoid::{
    clearance_reward, cot_efficiency_reward, episode_reward, pd_standing_baseline,
    pd_walking_baseline, standing_reward, ActiveInferenceHumanoidAgent, HumanoidCommand,
    HumanoidConfig, HumanoidController, HumanoidFepConfig, HumanoidHdcEncoder, HumanoidPdGains,
    HumanoidPerturbation, HumanoidPhysicsSimulator, HumanoidState, HumanoidTask, HumanoidTrainer,
    PerturbationSchedule, SimpleHumanoidSimulator, NUM_ACTUATORS,
};
use symthaea::symthaea_core::genesis::GenesisSeed;
use symthaea::symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

// ============================================================================
// 1. HumanoidHdcEncoder: finite, non-zero 16384D vectors from HumanoidState
// ============================================================================

#[test]
fn test_encoder_produces_finite_nonzero_16384d_from_standing() {
    let genesis = GenesisSeed::from_phrase("integ-encoder-test");
    let mut encoder = HumanoidHdcEncoder::new(&genesis, 32);
    let state = HumanoidState::standing();
    let hv = encoder.encode(&state);

    assert_eq!(
        hv.dim(),
        HDC_DIMENSION,
        "Encoder must produce 16384D vector"
    );
    assert!(
        hv.norm().is_finite() && hv.norm() > 0.0,
        "Encoded HV must be finite and non-zero, got norm={}",
        hv.norm()
    );
    // Self-similarity should be near 1.0
    let self_sim = hv.similarity(&hv);
    assert!(
        (self_sim - 1.0).abs() < 1e-4,
        "Self-similarity should be ~1.0, got {self_sim}"
    );
}

#[test]
fn test_encoder_produces_finite_nonzero_from_perturbed_state() {
    let genesis = GenesisSeed::from_phrase("integ-encoder-perturbed");
    let mut encoder = HumanoidHdcEncoder::new(&genesis, 32);

    // Construct a non-trivial state (partially fallen, moving)
    let mut state = HumanoidState::standing();
    state.root_height = 0.6;
    state.head_height = 0.7;
    state.torso_vertical = [0.3, 0.1, 0.85];
    state.root_angular_velocity = [3.0, -2.0, 1.0];
    state.root_linear_velocity = [0.5, 0.2, -0.3];
    state.joint_angles[0] = 0.5; // abdomen tilted
    state.joint_angles[5] = -0.3; // hip flexion
    state.joint_velocities[6] = 2.0;
    state.com_velocity = [0.5, 0.1, -0.1];

    let hv = encoder.encode(&state);
    assert_eq!(hv.dim(), HDC_DIMENSION);
    assert!(
        hv.norm().is_finite() && hv.norm() > 0.0,
        "Perturbed state encoding must be finite non-zero"
    );
}

#[test]
fn test_encoder_derivative_encoding_changes_output() {
    let genesis = GenesisSeed::from_phrase("integ-encoder-deriv");
    let mut encoder = HumanoidHdcEncoder::new(&genesis, 32);

    let state1 = HumanoidState::standing();
    let hv1 = encoder.encode(&state1);

    // Second encode with same state triggers derivative encoding (delta=0)
    let hv2 = encoder.encode(&state1);

    assert_eq!(hv1.dim(), HDC_DIMENSION);
    assert_eq!(hv2.dim(), HDC_DIMENSION);
    // First call has no derivative, second has derivative channels
    // They should differ because derivative encoding adds channels even if delta=0
    let sim = hv1.similarity(&hv2);
    assert!(
        sim < 0.999,
        "First vs second encode should differ due to derivative channels: sim={sim}"
    );
}

#[test]
fn test_encoder_different_states_produce_different_hvs() {
    let genesis = GenesisSeed::from_phrase("integ-encoder-diff");
    let mut encoder = HumanoidHdcEncoder::new(&genesis, 32);

    let standing = HumanoidState::standing();
    let hv_standing = encoder.encode(&standing);

    encoder.reset();

    let mut fallen = HumanoidState::standing();
    fallen.root_height = 0.2;
    fallen.head_height = 0.3;
    fallen.torso_vertical = [0.7, 0.0, 0.3];
    let hv_fallen = encoder.encode(&fallen);

    let sim = hv_standing.similarity(&hv_fallen);
    assert!(
        sim < 0.95,
        "Standing vs fallen should produce distinct encodings: sim={sim}"
    );
}

// ============================================================================
// 2. HumanoidController: forward pass produces 21D output, finite, tanh-bounded
// ============================================================================

#[test]
fn test_controller_forward_produces_21d_bounded_output() {
    let genesis = GenesisSeed::from_phrase("integ-ctrl-forward");
    let config = HumanoidConfig::default();
    let mut controller = HumanoidController::new(&genesis, &config);

    let sensor = ContinuousHV::random(HDC_DIMENSION, 42);
    let cmd = controller.forward(&sensor, 0.025);

    assert_eq!(
        cmd.torques.len(),
        NUM_ACTUATORS,
        "Controller must produce {} outputs",
        NUM_ACTUATORS
    );
    for (i, &t) in cmd.torques.iter().enumerate() {
        assert!(t.is_finite(), "Torque {i} must be finite, got {t}");
        assert!(
            t >= -1.0 && t <= 1.0,
            "Torque {i} must be tanh-bounded [-1,1], got {t}"
        );
    }
}

#[test]
fn test_controller_forward_multiple_steps_stay_bounded() {
    let genesis = GenesisSeed::from_phrase("integ-ctrl-multi");
    let config = HumanoidConfig::default();
    let mut controller = HumanoidController::new(&genesis, &config);

    // Run 50 forward steps with varying inputs
    for step in 0..50 {
        let sensor = ContinuousHV::random(HDC_DIMENSION, step as u64);
        let cmd = controller.forward(&sensor, 0.025);
        for (i, &t) in cmd.torques.iter().enumerate() {
            assert!(
                t.is_finite() && t >= -1.0 && t <= 1.0,
                "Step {step}, joint {i}: torque {t} out of bounds"
            );
        }
    }
}

// ============================================================================
// 3. Controller warmup reduces error
// ============================================================================

#[test]
fn test_controller_warmup_reduces_error() {
    let genesis = GenesisSeed::from_phrase("integ-ctrl-warmup");
    let config = HumanoidConfig::default();
    let mut encoder = HumanoidHdcEncoder::new(&genesis, config.num_levels);
    let pd_gains = HumanoidPdGains::default();

    // Generate 20 significantly perturbed (sensor_hv, target) training samples.
    // Use large joint deviations so PD baseline produces meaningful non-zero torques,
    // giving the warmup a clear gradient signal to learn from.
    let mut samples = Vec::with_capacity(20);
    for i in 0..20 {
        let mut state = HumanoidState::standing();
        // Significant perturbations so PD baseline produces non-trivial torques
        state.joint_angles[0] = 0.3 + 0.02 * i as f64; // abdomen tilt
        state.joint_angles[3] = -0.2 - 0.01 * i as f64; // hip flexion
        state.joint_angles[5] = 0.15 + 0.015 * i as f64; // knee
        state.joint_angles[8] = -0.1 + 0.01 * i as f64; // ankle
        state.joint_velocities[0] = 0.5;
        state.joint_velocities[3] = -0.3;

        let hv = encoder.encode(&state);
        let target = pd_standing_baseline(&state, &pd_gains);
        samples.push((hv, target));
        encoder.reset();
    }

    // Helper: compute average MSE over all samples
    let avg_mse = |ctrl: &mut HumanoidController| -> f32 {
        let mut total_err = 0.0f32;
        for (hv, target) in &samples {
            let cmd = ctrl.forward(hv, 0.025);
            let err: f32 = cmd
                .torques
                .iter()
                .zip(target.torques.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum();
            total_err += err;
        }
        ctrl.reset();
        total_err / samples.len() as f32
    };

    // Measure initial average error across all samples
    let mut controller = HumanoidController::new(&genesis, &config);
    let initial_err = avg_mse(&mut controller);

    // Warmup: 200 steps across the 20 samples
    controller.warmup(&samples, 200, 0.0005);

    // Measure post-warmup average error
    let post_err = avg_mse(&mut controller);

    assert!(
        post_err < initial_err,
        "Warmup should reduce average error: initial={initial_err:.4}, post={post_err:.4}"
    );
}

// ============================================================================
// 4. SimpleHumanoidSimulator step produces valid next state
// ============================================================================

#[test]
fn test_simulator_step_produces_valid_state() {
    let mut sim = SimpleHumanoidSimulator::new();
    let cmd = HumanoidCommand::zero();

    for _ in 0..20 {
        sim.step(&cmd, 0.025);
    }

    let state = sim.state();
    assert!(state.root_height.is_finite(), "root_height must be finite");
    assert!(state.head_height.is_finite(), "head_height must be finite");
    assert!(
        state.root_height > 0.0,
        "root_height must be positive: {}",
        state.root_height
    );
    assert!(
        state.head_height > 0.0,
        "head_height must be positive: {}",
        state.head_height
    );
    for (i, &a) in state.joint_angles.iter().enumerate() {
        assert!(a.is_finite(), "Joint angle {i} must be finite: {a}");
    }
    for (i, &v) in state.joint_velocities.iter().enumerate() {
        assert!(v.is_finite(), "Joint velocity {i} must be finite: {v}");
    }
    for (i, &q) in state.root_quaternion.iter().enumerate() {
        assert!(
            q.is_finite(),
            "Quaternion component {i} must be finite: {q}"
        );
    }
    // Quaternion should be approximately unit norm
    let qnorm = state
        .root_quaternion
        .iter()
        .map(|q| q * q)
        .sum::<f64>()
        .sqrt();
    assert!(
        (qnorm - 1.0).abs() < 0.01,
        "Quaternion should be unit norm: {qnorm}"
    );
}

#[test]
fn test_simulator_with_nonzero_torques_stays_finite() {
    let mut sim = SimpleHumanoidSimulator::new();
    // Apply moderate torques to all joints
    let cmd = HumanoidCommand {
        torques: [0.3; NUM_ACTUATORS],
    };

    for _ in 0..100 {
        sim.step(&cmd, 0.025);
    }

    let state = sim.state();
    assert!(state.root_height.is_finite());
    assert!(state.head_height.is_finite());
    assert!(state.timestamp.is_finite());
    assert!(state.timestamp > 0.0, "Time should advance");
}

#[test]
fn test_simulator_reset_restores_standing() {
    let mut sim = SimpleHumanoidSimulator::new();
    let cmd = HumanoidCommand {
        torques: [0.5; NUM_ACTUATORS],
    };

    // Disturb the state
    sim.apply_external_force([500.0, 200.0, 100.0]);
    for _ in 0..50 {
        sim.step(&cmd, 0.025);
    }

    sim.reset();
    let state = sim.state();
    assert!(
        (state.root_height - 1.3).abs() < 0.1,
        "Reset should restore root_height near 1.3: {}",
        state.root_height
    );
    assert!(
        state.horizontal_speed() < 1e-10,
        "Reset should zero horizontal speed"
    );
}

#[test]
fn test_simulator_external_force_affects_state() {
    let mut sim = SimpleHumanoidSimulator::new();
    let cmd = HumanoidCommand::zero();

    // Record initial state
    let initial_vx = sim.state().root_linear_velocity[0];

    sim.apply_external_force([1000.0, 0.0, 0.0]);
    sim.step(&cmd, 0.025);
    // Need additional steps for jitter buffer to show the effect
    sim.step(&cmd, 0.025);

    let state = sim.state();
    assert!(
        (state.root_linear_velocity[0] - initial_vx).abs() > 0.01,
        "External force should change velocity: vx={}",
        state.root_linear_velocity[0]
    );
}

// ============================================================================
// 5. GaitAnalyzer tracks foot contacts and computes finite metrics
// ============================================================================

#[test]
fn test_gait_analyzer_detects_swing_stance_transitions() {
    use symthaea::humanoid::GaitAnalyzer;

    let mut analyzer = GaitAnalyzer::new();

    // Simulate: ground(5 ticks) -> swing(10 ticks) -> ground(1 tick) = 1 stride
    for _ in 0..5 {
        let mut state = HumanoidState::standing();
        state.extremities[8] = 0.0; // right foot on ground
        analyzer.update(&state);
    }
    for _ in 0..10 {
        let mut state = HumanoidState::standing();
        state.extremities[8] = 0.08; // right foot in air
        analyzer.update(&state);
    }
    // Touchdown
    let mut state = HumanoidState::standing();
    state.extremities[8] = 0.0;
    analyzer.update(&state);

    assert_eq!(
        analyzer.num_strides(),
        1,
        "Should detect 1 stride from swing-to-stance"
    );

    let avg_cl = analyzer.avg_clearance();
    let min_cl = analyzer.min_clearance();
    assert!(avg_cl.is_finite(), "avg_clearance must be finite");
    assert!(min_cl.is_finite(), "min_clearance must be finite");
    assert!(
        (avg_cl - 0.08).abs() < 1e-6,
        "Max clearance should be 0.08: got {avg_cl}"
    );
}

#[test]
fn test_gait_analyzer_summary_all_finite() {
    use symthaea::humanoid::GaitAnalyzer;

    let mut analyzer = GaitAnalyzer::new();
    let mut state = HumanoidState::standing();

    // Create two right-foot swing/stance cycles with position and time
    // Cycle 1: ground -> swing -> touchdown
    state.extremities[8] = 0.0;
    analyzer.update_with_position(&state, [0.0, 0.0], 0.0);
    state.extremities[8] = 0.07;
    analyzer.update_with_position(&state, [0.1, 0.0], 0.1);
    state.extremities[8] = 0.0;
    analyzer.update_with_position(&state, [0.3, 0.0], 0.3);

    // Cycle 2: swing -> touchdown
    state.extremities[8] = 0.06;
    analyzer.update_with_position(&state, [0.5, 0.0], 0.5);
    state.extremities[8] = 0.0;
    analyzer.update_with_position(&state, [0.8, 0.0], 0.8);

    let summary = analyzer.summary();
    assert!(
        summary.avg_clearance.is_finite(),
        "avg_clearance must be finite"
    );
    assert!(
        summary.min_clearance.is_finite(),
        "min_clearance must be finite"
    );
    assert!(
        summary.avg_stride_length.is_finite(),
        "avg_stride_length must be finite"
    );
    assert!(
        summary.avg_cadence.is_finite(),
        "avg_cadence must be finite"
    );
    assert!(
        summary.gait_asymmetry.is_finite(),
        "gait_asymmetry must be finite"
    );
    assert!(
        summary.step_regularity.is_finite(),
        "step_regularity must be finite"
    );
    assert!(
        summary.foot_strike_quality.is_finite(),
        "foot_strike_quality must be finite"
    );
    assert!(
        summary.stride_count >= 2,
        "Should have at least 2 strides: got {}",
        summary.stride_count
    );
}

#[test]
fn test_gait_analyzer_reset_clears_state() {
    use symthaea::humanoid::GaitAnalyzer;

    let mut analyzer = GaitAnalyzer::new();

    // Generate some data
    let mut state = HumanoidState::standing();
    state.extremities[8] = 0.0;
    analyzer.update(&state);
    state.extremities[8] = 0.1;
    analyzer.update(&state);
    state.extremities[8] = 0.0;
    analyzer.update(&state);

    analyzer.reset();
    assert_eq!(analyzer.num_strides(), 0, "Reset should clear strides");
    assert!(
        analyzer.avg_clearance().abs() < 1e-10,
        "Reset should clear clearance"
    );
}

// ============================================================================
// 6. Standing reward: upright+still > tilted+moving
// ============================================================================

#[test]
fn test_standing_reward_upright_higher_than_tilted() {
    let upright = HumanoidState::standing(); // head=1.4, uprightness=1.0
    let reward_upright = standing_reward(&upright);

    let mut tilted = HumanoidState::standing();
    tilted.head_height = 0.8;
    tilted.torso_vertical = [0.4, 0.0, 0.6]; // uprightness = 0.6
    let reward_tilted = standing_reward(&tilted);

    assert!(
        reward_upright > reward_tilted,
        "Upright reward ({reward_upright:.4}) should exceed tilted ({reward_tilted:.4})"
    );
    assert!(
        reward_upright > 0.8,
        "Upright reward should be high: {reward_upright}"
    );
    assert!(
        reward_tilted < 0.5,
        "Tilted reward should be low: {reward_tilted}"
    );
}

#[test]
fn test_episode_reward_stand_zero_torque_high() {
    let state = HumanoidState::standing();
    let cmd = HumanoidCommand::zero();
    let reward = episode_reward(&state, &cmd, &HumanoidTask::Stand, 0.0);

    assert!(reward.is_finite(), "Episode reward must be finite");
    assert!(
        reward > 0.5,
        "Standing correctly with zero torque should give good reward: {reward}"
    );
}

#[test]
fn test_episode_reward_fallen_very_low() {
    let mut state = HumanoidState::standing();
    state.head_height = 0.2;
    state.torso_vertical = [0.0, 0.0, 0.1];
    let cmd = HumanoidCommand::zero();
    let reward = episode_reward(&state, &cmd, &HumanoidTask::Stand, 0.0);

    assert!(reward.is_finite());
    assert!(
        reward < 0.1,
        "Fallen state should give very low reward: {reward}"
    );
}

#[test]
fn test_clearance_and_cot_rewards_finite() {
    let state = HumanoidState::standing();
    let cmd = HumanoidCommand::zero();

    let cr = clearance_reward(&state);
    let cot = cot_efficiency_reward(&state, &cmd);

    assert!(cr.is_finite(), "Clearance reward must be finite");
    assert!(cot.is_finite(), "CoT reward must be finite");
    assert!(
        cr >= 0.0 && cr <= 1.0,
        "Clearance reward should be in [0,1]: {cr}"
    );
    assert!(
        cot >= 0.0 && cot <= 1.0,
        "CoT reward should be in [0,1]: {cot}"
    );
}

// ============================================================================
// 7. HumanoidTrainer::run_episode runs without panic (at least 5 steps)
// ============================================================================

#[test]
fn test_trainer_run_episode_no_panic() {
    let config = HumanoidConfig {
        num_episodes: 1,
        steps_per_episode: 10, // short episode
        ..HumanoidConfig::default()
    };
    let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
    let mut encoder = HumanoidHdcEncoder::new(&genesis, config.num_levels);
    let mut controller = HumanoidController::new(&genesis, &config);
    let mut fep_agent =
        ActiveInferenceHumanoidAgent::new(HumanoidFepConfig::default(), HumanoidTask::Stand);

    let trainer = HumanoidTrainer::new(config);
    let metrics = trainer.run_episode(&mut encoder, &mut controller, &mut fep_agent, 0);

    assert!(
        metrics.total_steps >= 5,
        "Episode should complete at least 5 steps: got {}",
        metrics.total_steps
    );
    assert!(
        metrics.avg_standing_reward.is_finite(),
        "avg_standing_reward must be finite"
    );
    assert!(
        metrics.avg_episode_reward.is_finite(),
        "avg_episode_reward must be finite"
    );
    assert!(
        metrics.avg_free_energy.is_finite(),
        "avg_free_energy must be finite"
    );
    assert!(
        metrics.avg_head_height.is_finite(),
        "avg_head_height must be finite"
    );
}

#[test]
fn test_trainer_full_pipeline_multiple_episodes() {
    let config = HumanoidConfig {
        num_episodes: 3,
        steps_per_episode: 30,
        ..HumanoidConfig::default()
    };
    let mut trainer = HumanoidTrainer::new(config);
    let metrics = trainer.train();

    assert_eq!(metrics.len(), 3, "Should produce 3 episode metrics");
    for (i, m) in metrics.iter().enumerate() {
        assert!(
            m.avg_standing_reward.is_finite(),
            "Episode {i}: avg_standing_reward must be finite"
        );
        assert!(
            m.avg_episode_reward.is_finite(),
            "Episode {i}: avg_episode_reward must be finite"
        );
        assert!(
            m.avg_free_energy.is_finite(),
            "Episode {i}: avg_free_energy must be finite"
        );
        assert!(
            m.avg_head_height.is_finite(),
            "Episode {i}: avg_head_height must be finite"
        );
        assert!(
            m.total_steps > 0,
            "Episode {i}: should complete at least 1 step"
        );
    }
}

// ============================================================================
// 8. FEP agent produces valid actions (tau/LR modulation)
// ============================================================================

#[test]
fn test_fep_agent_produces_valid_actions_standing() {
    let config = HumanoidFepConfig::default();
    let mut agent = ActiveInferenceHumanoidAgent::new(config, HumanoidTask::Stand);

    let state = HumanoidState::standing();
    let cmd = HumanoidCommand::zero();

    let result = agent.step(&state, &cmd);
    assert!(
        result.free_energy.is_finite(),
        "Free energy must be finite: {}",
        result.free_energy
    );
    assert!(
        result.prediction_error.is_finite(),
        "Prediction error must be finite"
    );
    assert!(
        result.tau_factor.is_finite(),
        "tau_factor must be finite: {}",
        result.tau_factor
    );
    assert!(
        result.learning_rate_factor.is_finite(),
        "learning_rate_factor must be finite"
    );
    assert!(
        result.prior_precision.is_finite(),
        "prior_precision must be finite"
    );
    // tau_factor should be one of the defined action values
    let valid_tau = [0.85_f32, 1.0, 1.15];
    assert!(
        valid_tau
            .iter()
            .any(|&v| (result.tau_factor - v).abs() < 1e-5),
        "tau_factor should be 0.85, 1.0, or 1.15: got {}",
        result.tau_factor
    );
}

#[test]
fn test_fep_agent_responds_to_falling_state() {
    let config = HumanoidFepConfig::default();
    let mut agent = ActiveInferenceHumanoidAgent::new(config, HumanoidTask::Stand);

    // Simulate a falling state (high angular momentum, low head height)
    let mut state = HumanoidState::standing();
    state.head_height = 0.5;
    state.torso_vertical = [0.5, 0.0, 0.5];
    state.root_angular_velocity = [5.0, 3.0, 1.0];

    let cmd = HumanoidCommand::zero();
    let result = agent.step(&state, &cmd);

    assert!(result.free_energy.is_finite());
    assert!(result.prediction_error.is_finite());
    // When falling, rule-based policy should select DropTau (0.85)
    assert!(
        (result.tau_factor - 0.85).abs() < 1e-5,
        "Falling state should trigger DropTau: got tau_factor={}",
        result.tau_factor
    );
}

#[test]
fn test_fep_agent_exploration_triggers_on_sustained_high_fe() {
    let config = HumanoidFepConfig {
        exploration_patience: 3,
        exploration_fe_threshold: 0.0, // Threshold of 0 means FE always exceeds
        ..HumanoidFepConfig::default()
    };
    let mut agent = ActiveInferenceHumanoidAgent::new(config, HumanoidTask::Stand);

    let state = HumanoidState::standing();
    let cmd = HumanoidCommand::zero();

    let mut exploration_triggered = false;
    for _ in 0..10 {
        let result = agent.step(&state, &cmd);
        if result.exploration_noise.is_some() {
            exploration_triggered = true;
            // Verify noise array has correct dimension
            let noise = result.exploration_noise.unwrap();
            assert_eq!(
                noise.len(),
                NUM_ACTUATORS,
                "Exploration noise must have {} entries",
                NUM_ACTUATORS
            );
            for &n in &noise {
                assert!(n.is_finite(), "Exploration noise must be finite");
            }
            break;
        }
    }
    assert!(
        exploration_triggered,
        "Exploration should trigger after patience ticks with low FE threshold"
    );
}

#[test]
fn test_fep_agent_reset_clears_state() {
    let config = HumanoidFepConfig::default();
    let mut agent = ActiveInferenceHumanoidAgent::new(config, HumanoidTask::Stand);

    let state = HumanoidState::standing();
    let cmd = HumanoidCommand::zero();
    agent.step(&state, &cmd);

    agent.reset();
    let fe = agent.current_free_energy();
    assert!(fe.abs() < 1e-6, "Reset should zero free energy: got {fe}");
}

// ============================================================================
// 9. Perturbation (ChestShove) produces non-zero force
// ============================================================================

#[test]
fn test_perturbation_chest_shove_applies_force() {
    let mut schedule = PerturbationSchedule::chest_shove();
    assert_eq!(schedule.len(), 1);

    // Verify trigger window: step 300..305
    assert!(schedule.triggered_at(299).is_empty());
    assert_eq!(schedule.triggered_at(300).len(), 1);
    assert_eq!(schedule.triggered_at(304).len(), 1);
    assert!(schedule.triggered_at(305).is_empty());

    // Apply the perturbation at step 300
    let mut sim = SimpleHumanoidSimulator::new();
    let mut cmd = HumanoidCommand::zero();

    // Before perturbation
    schedule.apply(299, &mut cmd, &mut sim);
    sim.step(&cmd, 0.025);
    let vel_before = sim.state().root_linear_velocity[0];

    // During perturbation (500N push)
    schedule.apply(300, &mut cmd, &mut sim);
    sim.step(&cmd, 0.025);
    schedule.apply(301, &mut cmd, &mut sim);
    sim.step(&cmd, 0.025);

    let vel_after = sim.state().root_linear_velocity[0];
    assert!(
        (vel_after - vel_before).abs() > 0.01,
        "Chest shove should change velocity: before={vel_before}, after={vel_after}"
    );
}

#[test]
fn test_perturbation_actuator_failure_masks_torque() {
    let mut schedule = PerturbationSchedule::new().with(HumanoidPerturbation::ActuatorFailure {
        joint_index: 6, // right knee
        at_step: 10,
    });

    let mut sim = SimpleHumanoidSimulator::new();
    let mut cmd = HumanoidCommand {
        torques: [0.5; NUM_ACTUATORS],
    };

    // Before failure: torque passes through
    schedule.apply(5, &mut cmd, &mut sim);
    assert!(
        (cmd.torques[6] - 0.5).abs() < 1e-6,
        "Before failure, torque should pass: {}",
        cmd.torques[6]
    );

    // At failure step: actuator disabled
    cmd.torques[6] = 0.5;
    schedule.apply(10, &mut cmd, &mut sim);
    assert!(
        cmd.torques[6].abs() < 1e-6,
        "At failure, torque must be zero: {}",
        cmd.torques[6]
    );

    // After failure: permanently disabled
    cmd.torques[6] = 0.8;
    schedule.apply(100, &mut cmd, &mut sim);
    assert!(
        cmd.torques[6].abs() < 1e-6,
        "After failure, torque must stay zero: {}",
        cmd.torques[6]
    );
    assert!(
        schedule.is_actuator_disabled(6),
        "Joint 6 should be disabled"
    );
}

#[test]
fn test_perturbation_external_push_has_nonzero_force() {
    let push = HumanoidPerturbation::ExternalPush {
        force: [500.0, 0.0, 0.0],
        at_step: 300,
        duration: 5,
    };

    match &push {
        HumanoidPerturbation::ExternalPush { force, .. } => {
            let magnitude =
                (force[0] * force[0] + force[1] * force[1] + force[2] * force[2]).sqrt();
            assert!(
                magnitude > 0.0,
                "ChestShove force must be non-zero: magnitude={magnitude}"
            );
            assert!(
                magnitude > 100.0,
                "ChestShove should be a significant force: {magnitude}N"
            );
        }
        _ => panic!("Expected ExternalPush variant"),
    }
}

// ============================================================================
// 10. Morphological transfer channel mapping has correct dimensions
// ============================================================================

#[test]
fn test_transfer_flight_to_humanoid_channel_mapping() {
    let transfer = MorphologicalTransfer::flight_to_humanoid();

    // Should have 11 shared channel pairs:
    // 1 altitude + 4 orientation + 3 linear vel + 3 angular vel
    assert_eq!(
        transfer.num_shared_channels(),
        11,
        "Flight-to-humanoid should map 11 channel pairs"
    );
    assert!(
        (transfer.blend_factor() - 0.5).abs() < 1e-6,
        "Default blend factor should be 0.5"
    );
}

#[test]
fn test_transfer_creates_valid_controller() {
    use symthaea::symthaea_core::hdc::{HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig};

    // Create a small "trained" flight network
    let flight_genesis = GenesisSeed::from_phrase("symthaea-multirotor-quadrotor");
    let flight_config = UnifiedNetworkConfig {
        layer_sizes: vec![4; 2], // 2x4 flight config
        neuron_config: UnifiedConfig {
            tau_base: 0.01,
            backbone_tau: 0.5,
            dimension: HDC_DIMENSION,
            learning_rate: 0.001,
            ..UnifiedConfig::default()
        },
        use_layer_binding: true,
        skip_connections: false,
    };
    let mut flight_network = HdcLtcUnifiedNetwork::from_genesis(flight_config, &flight_genesis);

    // "Train" the flight network by evolving with several inputs
    for i in 0..20 {
        let input = ContinuousHV::random(HDC_DIMENSION, i as u64);
        flight_network.evolve_closed_form(0.01, &input);
    }

    // Transfer to humanoid
    let transfer = MorphologicalTransfer::flight_to_humanoid();
    let humanoid_config = HumanoidConfig::default();
    let mut controller = transfer.initialize_humanoid(&flight_network, &humanoid_config);

    // Forward pass should produce valid output
    let sensor = ContinuousHV::random(HDC_DIMENSION, 42);
    let cmd = controller.forward(&sensor, 0.025);

    assert_eq!(cmd.torques.len(), NUM_ACTUATORS);
    for (i, &t) in cmd.torques.iter().enumerate() {
        assert!(
            t.is_finite(),
            "Transfer ctrl torque {i} must be finite: {t}"
        );
        assert!(
            t >= -1.0 && t <= 1.0,
            "Transfer ctrl torque {i} must be in [-1,1]: {t}"
        );
    }
}

#[test]
fn test_transfer_differs_from_random_init() {
    use symthaea::symthaea_core::hdc::{HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig};

    let flight_genesis = GenesisSeed::from_phrase("symthaea-multirotor-quadrotor");
    let flight_config = UnifiedNetworkConfig {
        layer_sizes: vec![4; 2],
        neuron_config: UnifiedConfig {
            tau_base: 0.01,
            backbone_tau: 0.5,
            dimension: HDC_DIMENSION,
            learning_rate: 0.001,
            ..UnifiedConfig::default()
        },
        use_layer_binding: true,
        skip_connections: false,
    };
    let mut flight_network = HdcLtcUnifiedNetwork::from_genesis(flight_config, &flight_genesis);

    for i in 0..20 {
        let input = ContinuousHV::random(HDC_DIMENSION, i as u64);
        flight_network.evolve_closed_form(0.01, &input);
    }

    let transfer = MorphologicalTransfer::flight_to_humanoid();
    let config = HumanoidConfig::default();
    let mut transfer_ctrl = transfer.initialize_humanoid(&flight_network, &config);

    let humanoid_genesis = GenesisSeed::from_phrase("symthaea-humanoid-dmc");
    let mut random_ctrl = HumanoidController::new(&humanoid_genesis, &config);

    // Run multiple forward steps to let weight differences accumulate
    let sensor = ContinuousHV::random(HDC_DIMENSION, 42);
    for _ in 0..10 {
        transfer_ctrl.forward(&sensor, 0.025);
        random_ctrl.forward(&sensor, 0.025);
    }

    // Network outputs should diverge
    let transfer_output = transfer_ctrl.network().output();
    let random_output = random_ctrl.network().output();
    let sim = transfer_output.similarity(&random_output);

    assert!(
        sim < 0.999,
        "Transfer-initialized network should diverge from random: sim={sim}"
    );
}

#[test]
fn test_custom_transfer_mapping_dimensions() {
    let source_genesis = GenesisSeed::from_phrase("test-source");
    let target_genesis = GenesisSeed::from_phrase("test-target");
    let mapping = vec![
        ChannelMapping {
            source_ch: 0,
            target_ch: 0,
            weight: 1.5,
        },
        ChannelMapping {
            source_ch: 3,
            target_ch: 1,
            weight: 2.0,
        },
        ChannelMapping {
            source_ch: 7,
            target_ch: 26,
            weight: 0.8,
        },
    ];

    let transfer = MorphologicalTransfer::new(source_genesis, target_genesis, mapping, 0.7);
    assert_eq!(transfer.num_shared_channels(), 3);
    assert!((transfer.blend_factor() - 0.7).abs() < 1e-6);
}

// ============================================================================
// Additional integration tests: full encode-forward-step loop
// ============================================================================

#[test]
fn test_full_encode_forward_step_loop() {
    let config = HumanoidConfig::default();
    let genesis = GenesisSeed::from_phrase("integ-full-loop");
    let mut encoder = HumanoidHdcEncoder::new(&genesis, config.num_levels);
    let mut controller = HumanoidController::new(&genesis, &config);
    let mut sim = SimpleHumanoidSimulator::new();

    // Run 20 closed-loop steps: encode -> forward -> step
    for _ in 0..20 {
        let state = sim.state().clone();
        let sensor_hv = encoder.encode(&state);
        let cmd = controller.forward(&sensor_hv, 0.025);

        // Every torque must be valid
        for &t in &cmd.torques {
            assert!(t.is_finite());
            assert!(t >= -1.0 && t <= 1.0);
        }

        sim.step(&cmd, 0.025);
    }

    // Final state should remain physical
    let final_state = sim.state();
    assert!(final_state.root_height > 0.0);
    assert!(final_state.head_height.is_finite());
}

#[test]
fn test_humanoid_feature_reexports() {
    // Verify all key types are accessible through the main crate
    let config = HumanoidConfig::default();
    assert!(config.physics_hz > 0.0);
    assert!(config.cognitive_hz > 0.0);
    assert_eq!(config.cognitive_interval(), 4); // 40Hz / 10Hz
}

#[test]
fn test_humanoid_state_to_channels_consistency() {
    let state = HumanoidState::standing();
    let channels = state.to_channels();
    assert_eq!(
        channels.len(),
        HumanoidState::num_channels(),
        "to_channels() length must match num_channels()"
    );
    assert_eq!(channels.len(), 72);
    // root_height at index 0
    assert!((channels[0] - 1.3).abs() < 1e-6);
    // All channels should be finite
    for (i, &ch) in channels.iter().enumerate() {
        assert!(ch.is_finite(), "Channel {i} must be finite: {ch}");
    }
}

// ============================================================================
// PD baseline tests
// ============================================================================

#[test]
fn test_pd_standing_baseline_zero_at_standing() {
    let state = HumanoidState::standing();
    let gains = HumanoidPdGains::default();
    let cmd = pd_standing_baseline(&state, &gains);

    // At standing (all joints at zero): all torques should be near zero
    for (i, &t) in cmd.torques.iter().enumerate() {
        assert!(
            t.abs() < 0.01,
            "Joint {i} at standing should have near-zero torque: {t}"
        );
    }
}

#[test]
fn test_pd_walking_baseline_cyclic_gait() {
    let state = HumanoidState::standing();
    let gains = HumanoidPdGains::default();

    let cmd_0 = pd_walking_baseline(&state, &gains, 0.0, 1.0);
    let cmd_quarter = pd_walking_baseline(&state, &gains, 0.25, 1.0);
    let cmd_half = pd_walking_baseline(&state, &gains, 0.5, 1.0);

    // Different phases should produce different torques
    let diff_0_quarter: f32 = cmd_0
        .torques
        .iter()
        .zip(cmd_quarter.torques.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    let diff_0_half: f32 = cmd_0
        .torques
        .iter()
        .zip(cmd_half.torques.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    assert!(
        diff_0_quarter > 0.01,
        "Phase 0 vs 0.25 should differ: sum_diff={diff_0_quarter}"
    );
    assert!(
        diff_0_half > 0.001,
        "Phase 0 vs 0.5 should differ: sum_diff={diff_0_half}"
    );
}

// ============================================================================
// Perturbation crucible: chest shove
// ============================================================================

#[test]
fn test_perturbation_crucible_chest_shove() {
    let config = HumanoidConfig::default();
    let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
    let mut encoder = HumanoidHdcEncoder::new(&genesis, config.num_levels);
    let mut controller = HumanoidController::new(&genesis, &config);
    let mut fep_agent =
        ActiveInferenceHumanoidAgent::new(HumanoidFepConfig::default(), HumanoidTask::Stand);
    let mut sim = SimpleHumanoidSimulator::new();

    // Moderate push (150N)
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

        if step % cognitive_interval == 0 {
            let fep_result = fep_agent.step(sim.state(), &cmd);
            if step < 50 {
                pre_push_fe.push(fep_result.free_energy);
            } else if step >= 55 && step < 120 {
                post_push_fe.push(fep_result.free_energy);
            }
        }
    }

    // All FE values should be finite
    assert!(!pre_push_fe.is_empty() && !post_push_fe.is_empty());
    for &fe in pre_push_fe.iter().chain(post_push_fe.iter()) {
        assert!(fe.is_finite(), "Free energy should be finite: {fe}");
    }

    // Final state should remain physical
    let final_state = sim.state();
    assert!(final_state.head_height.is_finite());
    assert!(final_state.root_height > 0.15, "Should stay above ground");
}

// ============================================================================
// Perturbation crucible: phantom limb
// ============================================================================

#[test]
fn test_perturbation_crucible_phantom_limb() {
    let config = HumanoidConfig::default();
    let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
    let mut encoder = HumanoidHdcEncoder::new(&genesis, config.num_levels);
    let mut controller = HumanoidController::new(&genesis, &config);
    let mut sim = SimpleHumanoidSimulator::new();

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

        if step > 50 {
            knee_torque_after_failure.push(cmd.torques[6]);
        }

        sim.step(&cmd, dt);
    }

    // Right knee should be masked to zero after failure
    for &t in &knee_torque_after_failure {
        assert!(
            t.abs() < 1e-6,
            "Right knee should be zero after failure: {t}"
        );
    }

    // Other actuators should still produce finite output
    let state = sim.state().clone();
    let sensor_hv = encoder.encode(&state);
    let cmd = controller.forward(&sensor_hv, dt as f32);
    for (i, &t) in cmd.torques.iter().enumerate() {
        assert!(t.is_finite(), "Joint {i} should produce finite output");
    }
}

// ============================================================================
// Motor bridge integration (cognitive loop <-> humanoid)
// ============================================================================

#[test]
fn test_motor_bridge_step_returns_perception() {
    use symthaea::cognitive_loop::motor_bridge::MotorBridge;

    let genesis = GenesisSeed::from_phrase("motor-bridge-integration-test");
    let mut bridge = MotorBridge::new(&genesis);

    let thought = ContinuousHV::random(HDC_DIMENSION, 42);
    let (cmd, perception) = bridge.step_direct(&thought);

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

    assert_eq!(bridge.total_steps(), 1);
}

#[test]
fn test_motor_bridge_perception_evolves() {
    use symthaea::cognitive_loop::motor_bridge::MotorBridge;

    let genesis = GenesisSeed::from_phrase("motor-bridge-perception-evolve");
    let mut bridge = MotorBridge::new(&genesis);

    let thought = ContinuousHV::random(HDC_DIMENSION, 99);

    // Initial perception
    let (_, p1) = bridge.step_direct(&thought);

    // Step multiple times to let state evolve
    let mut p_last = p1.clone();
    for _ in 0..20 {
        let (_, p) = bridge.step_direct(&thought);
        p_last = p;
    }

    // Perception should have changed as body state evolved
    let sim = p1.similarity(&p_last);
    assert!(
        sim < 0.999,
        "Perception should evolve as body state changes: sim={sim}"
    );
}

#[test]
fn test_motor_bridge_reset() {
    use symthaea::cognitive_loop::motor_bridge::MotorBridge;

    let genesis = GenesisSeed::from_phrase("motor-bridge-reset-test");
    let mut bridge = MotorBridge::new(&genesis);

    let thought = ContinuousHV::random(HDC_DIMENSION, 7);
    bridge.step_direct(&thought);
    assert!(bridge.last_perception().is_some());
    assert_eq!(bridge.total_steps(), 1);

    bridge.reset();
    assert!(bridge.last_perception().is_none());
    assert_eq!(bridge.total_steps(), 0);
}

// ============================================================================
// Checkpoint save/load roundtrip
// ============================================================================

#[test]
fn test_checkpoint_save_load_roundtrip() {
    let config = HumanoidConfig::default();
    let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
    let ctrl = HumanoidController::new(&genesis, &config);

    let path = "/tmp/symthaea_integ_humanoid_checkpoint.json";
    ctrl.save_checkpoint(path, &config).unwrap();

    let mut original = HumanoidController::new(&genesis, &config);
    let mut loaded = HumanoidController::load_checkpoint(path).unwrap();

    let sensor = ContinuousHV::random(HDC_DIMENSION, 42);
    let cmd_original = original.forward(&sensor, 0.025);
    let cmd_loaded = loaded.forward(&sensor, 0.025);

    for i in 0..NUM_ACTUATORS {
        assert!(
            (cmd_original.torques[i] - cmd_loaded.torques[i]).abs() < 1e-6,
            "Checkpoint roundtrip mismatch at joint {i}: {} vs {}",
            cmd_original.torques[i],
            cmd_loaded.torques[i]
        );
    }

    let _ = std::fs::remove_file(path);
}

#[test]
fn test_checkpoint_resume_training() {
    let config = HumanoidConfig {
        num_episodes: 3,
        steps_per_episode: 30,
        ..HumanoidConfig::default()
    };

    let dir = "/tmp/symthaea_integ_checkpoint_resume";
    let _ = std::fs::remove_dir_all(dir);
    let mut trainer = HumanoidTrainer::new(config.clone());
    let _ = trainer.train_with_telemetry(dir);

    let checkpoint_path = format!("{}/checkpoint.json", dir);
    assert!(std::path::Path::new(&checkpoint_path).exists());

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

// ============================================================================
// Perturbation schedule presets
// ============================================================================

#[test]
fn test_perturbation_schedule_presets() {
    let chest = PerturbationSchedule::chest_shove();
    assert_eq!(chest.len(), 1);

    let ice = PerturbationSchedule::ice_floor();
    assert_eq!(ice.len(), 1);

    let backpack = PerturbationSchedule::backpack();
    assert_eq!(backpack.len(), 1);

    let chaos = PerturbationSchedule::progressive_chaos();
    assert_eq!(chaos.len(), 3);

    let phantom = PerturbationSchedule::phantom_limb();
    assert_eq!(phantom.len(), 1);

    let empty = PerturbationSchedule::default();
    assert!(empty.is_empty());
}

// ============================================================================
// Domain randomization and noise models
// ============================================================================

#[test]
fn test_simulator_domain_randomization_changes_body() {
    let mut sim = SimpleHumanoidSimulator::new().with_domain_randomization(true);
    sim.reset_with_perturbation(0.5, 42);

    // After reset with domain randomization, body should still produce valid physics
    let cmd = HumanoidCommand::zero();
    for _ in 0..50 {
        sim.step(&cmd, 0.025);
    }

    let state = sim.state();
    assert!(state.root_height.is_finite());
    assert!(state.head_height.is_finite());
    assert!(state.root_height > 0.0);
}

#[test]
fn test_simulator_perturbation_changes_initial_state() {
    let mut sim = SimpleHumanoidSimulator::new();
    sim.reset_with_perturbation(1.0, 42);
    let any_nonzero = sim.state().joint_angles.iter().any(|a| a.abs() > 0.001);
    assert!(
        any_nonzero,
        "Perturbation should change at least one joint angle"
    );
}

// ============================================================================
// From-controller trainer
// ============================================================================

#[test]
fn test_motor_bridge_from_controller() {
    use symthaea::cognitive_loop::motor_bridge::MotorBridge;

    let config = HumanoidConfig::default();
    let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
    let controller = HumanoidController::new(&genesis, &config);

    let mut bridge = MotorBridge::from_controller(controller, config);
    let thought = ContinuousHV::random(HDC_DIMENSION, 123);

    let (cmd, perception) = bridge.step_direct(&thought);
    assert!(cmd.torques.iter().all(|t| t.is_finite()));
    assert_eq!(perception.dim(), HDC_DIMENSION);
}
