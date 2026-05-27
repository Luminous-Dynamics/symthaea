// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Symthaea Stress Test Suite
//!
//! Three batteries testing Symthaea's limits:
//! 1. **Morphological Degradation Frontier** — progressive joint failure, Phi correlation
//! 2. **Cost of Transport Profile** — energy efficiency across tasks vs biological baselines
//! 3. **Perturbation Recovery Time** — adaptability under chest shoves, ice, mass changes
//!
//! ```bash
//! cargo run --example symthaea_stress_test --release --features humanoid
//! ```

fn main() {
    #[cfg(not(feature = "humanoid"))]
    {
        eprintln!("ERROR: Requires --features humanoid");
        std::process::exit(1);
    }

    #[cfg(feature = "humanoid")]
    run_stress_test();
}

#[cfg(feature = "humanoid")]
fn run_stress_test() {
    use std::time::Instant;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Symthaea Stress Test Suite                                 ║");
    println!("║  Testing the limits of the Holographic Liquid Brain         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let start = Instant::now();

    battery_1_joint_degradation();
    battery_2_cost_of_transport();
    battery_3_perturbation_recovery();

    println!("━━━ Complete ━━━");
    println!(
        "  Total wall time: {:.1} min",
        start.elapsed().as_secs_f64() / 60.0
    );
}

// ═══════════════════════════════════════════════════════════════
// Battery 1: Morphological Degradation Frontier
// ═══════════════════════════════════════════════════════════════

#[cfg(feature = "humanoid")]
fn battery_1_joint_degradation() {
    use symthaea_core::genesis::GenesisSeed;
    use symthaea_humanoid::controller::HumanoidController;
    use symthaea_humanoid::encoder::HumanoidHdcEncoder;
    use symthaea_humanoid::fep_agent::{ActiveInferenceHumanoidAgent, HumanoidFepConfig};
    use symthaea_humanoid::simulator::{HumanoidPhysicsSimulator, SimpleHumanoidSimulator};
    use symthaea_humanoid::types::*;

    println!("━━━ Battery 1: Morphological Degradation Frontier ━━━");
    println!("  Progressive joint failure during standing. 9 joints disabled over 2000 steps.");
    println!();

    // Train controller
    let (mut controller, mut encoder, genesis) = train_controller("stress-degradation", 30);

    let mut sim = SimpleHumanoidSimulator::new();
    let mut fep =
        ActiveInferenceHumanoidAgent::new(HumanoidFepConfig::default(), HumanoidTask::Stand);

    // Joint failure schedule: arms → ankles → knees → hips
    let failures: Vec<(usize, usize, &str)> = vec![
        (200, 20, "left_elbow"),
        (400, 17, "right_elbow"),
        (600, 19, "left_shoulder2"),
        (800, 15, "right_shoulder1"),
        (1000, 8, "right_ankle_y"),
        (1200, 13, "left_ankle_x"),
        (1400, 6, "right_knee"), // ← expect cliff
        (1600, 12, "left_knee"), // ← expect catastrophic
        (1800, 5, "right_hip_y"),
    ];

    let mut disabled: Vec<usize> = Vec::new();
    let dt = 0.025;
    let total_steps = 2000;

    println!(
        "{:>6} {:>7} {:>12} {:>10} {:>10} {:>8} {:>12}",
        "Step", "Active", "StandReward", "HeadHt", "Upright", "PE", "Disabled"
    );

    let mut csv_rows: Vec<String> = Vec::new();
    csv_rows.push("step,active_joints,disabled_joint,standing_reward,head_height,uprightness,prediction_error,free_energy,fell".to_string());

    for step in 0..total_steps {
        // Check for new failure
        let mut just_disabled = "";
        for &(fail_step, joint, name) in &failures {
            if step == fail_step {
                disabled.push(joint);
                just_disabled = name;
            }
        }

        let state = sim.state().clone();
        let hv = encoder.encode(&state);
        let learned_cmd = controller.forward(&hv, dt as f32);

        // Combined system: PD baseline (safe mode) + HDC-LTC learned corrections
        // This is how a real robot would operate — PD provides stability,
        // the learned controller adds adaptivity
        let pd_gains = HumanoidPdGains::default();
        let pd_cmd = pd_standing_baseline(&state, &pd_gains);
        let pd_blend = 0.5; // 50% PD + 50% learned
        let mut cmd = HumanoidCommand::zero();
        for i in 0..cmd.torques.len().min(learned_cmd.torques.len()) {
            cmd.torques[i] =
                pd_blend * pd_cmd.torques[i] + (1.0 - pd_blend) * learned_cmd.torques[i];
            cmd.torques[i] = cmd.torques[i].clamp(-1.0, 1.0);
        }

        // Apply joint masking (phantom limb)
        for &joint in &disabled {
            if joint < cmd.torques.len() {
                cmd.torques[joint] = 0.0;
            }
        }

        sim.step(&cmd, dt);

        let s = sim.state();
        let standing = standing_reward_simple(s);
        let pe = encoder.prediction_error();
        let active = 21 - disabled.len();
        let fell = s.head_height < 0.5;

        // FEP tick every 4 steps
        let fe = if step % 4 == 0 {
            let enc_pe = if encoder.has_predictive_layer() {
                Some(pe)
            } else {
                None
            };
            let result = fep.step_with_encoder_pe(s, &cmd, enc_pe);
            result.free_energy
        } else {
            0.0
        };

        // Print every 200 steps or on failure events
        if step % 200 == 0 || !just_disabled.is_empty() {
            println!(
                "{:>6} {:>7} {:>12.4} {:>10.4} {:>10.4} {:>8.4} {:>12}",
                step,
                active,
                standing,
                s.head_height,
                s.torso_vertical[2],
                pe,
                if just_disabled.is_empty() {
                    "-"
                } else {
                    just_disabled
                }
            );
        }

        csv_rows.push(format!(
            "{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
            step,
            active,
            if just_disabled.is_empty() {
                "-".to_string()
            } else {
                just_disabled.to_string()
            },
            standing,
            s.head_height,
            s.torso_vertical[2],
            pe,
            fe,
            fell
        ));

        if fell && step > 200 {
            println!(
                "  *** FELL at step {} with {} joints active ***",
                step, active
            );
            // Continue running to show post-fall state
        }
    }

    // Write CSV
    let csv = csv_rows.join("\n");
    if let Err(e) = std::fs::create_dir_all("papers/stress_test") {
        eprintln!("  Warning: could not create output dir: {e}");
    }
    if let Err(e) = std::fs::write("papers/stress_test/joint_degradation.csv", &csv) {
        eprintln!("  Warning: could not write CSV: {e}");
    } else {
        println!(
            "  Output: papers/stress_test/joint_degradation.csv ({} rows)",
            csv_rows.len()
        );
    }
    println!();
}

// ═══════════════════════════════════════════════════════════════
// Battery 2: Cost of Transport Profile
// ═══════════════════════════════════════════════════════════════

#[cfg(feature = "humanoid")]
fn battery_2_cost_of_transport() {
    use symthaea_humanoid::training::HumanoidTrainer;
    use symthaea_humanoid::types::*;

    println!("━━━ Battery 2: Cost of Transport Profile ━━━");
    println!("  Energy efficiency across Stand/Walk/Run vs biological baselines.");
    println!();

    let mut csv_rows: Vec<String> = Vec::new();
    csv_rows.push(
        "task,episode,mean_power,cost_of_transport,control_effort,standing_reward,horizontal_speed"
            .to_string(),
    );

    for task in [HumanoidTask::Stand, HumanoidTask::Walk, HumanoidTask::Run] {
        let config = HumanoidConfig {
            num_episodes: 20,
            steps_per_episode: 1000,
            task,
            adaptive_curriculum: task != HumanoidTask::Stand,
            domain_randomization: false,
            genesis_phrase: format!("stress-cot-{:?}", task),
            ..HumanoidConfig::default()
        };

        println!("  Training {:?} (20 episodes)...", task);
        let mut trainer = HumanoidTrainer::new(config);
        let metrics = trainer.train();

        // Report last 5 episodes
        let last5: Vec<_> = metrics.iter().rev().take(5).collect();
        let mean_cot: f64 = last5.iter().map(|m| m.cost_of_transport).sum::<f64>() / 5.0;
        let mean_effort: f64 = last5
            .iter()
            .map(|m| m.avg_control_effort as f64)
            .sum::<f64>()
            / 5.0;
        let mean_reward: f64 = last5.iter().map(|m| m.avg_episode_reward).sum::<f64>() / 5.0;
        let mean_speed: f64 = last5.iter().map(|m| m.avg_horizontal_speed).sum::<f64>() / 5.0;

        println!(
            "    CoT={:.4}, Effort={:.4}, Reward={:.4}, Speed={:.4} m/s",
            mean_cot, mean_effort, mean_reward, mean_speed
        );

        for m in &metrics {
            csv_rows.push(format!(
                "{:?},{},{:.6},{:.6},{:.6},{:.6},{:.6}",
                task,
                m.episode,
                0.0,
                m.cost_of_transport,
                m.avg_control_effort,
                m.avg_standing_reward,
                m.avg_horizontal_speed
            ));
        }
    }

    // Biological baselines
    println!();
    println!("  Biological baselines:");
    println!("    Human walking CoT:  0.05");
    println!("    Human running CoT:  0.07");
    println!("    Atlas CoT:          ~1.5");
    println!("    Digit CoT:          ~1.0");
    println!("    Cassie CoT:         ~0.7");

    let csv = csv_rows.join("\n");
    let _ = std::fs::write("papers/stress_test/cost_of_transport.csv", &csv);
    println!(
        "  Output: papers/stress_test/cost_of_transport.csv ({} rows)",
        csv_rows.len()
    );
    println!();
}

// ═══════════════════════════════════════════════════════════════
// Battery 3: Perturbation Recovery Time
// ═══════════════════════════════════════════════════════════════

#[cfg(feature = "humanoid")]
fn battery_3_perturbation_recovery() {
    use symthaea_core::genesis::GenesisSeed;
    use symthaea_humanoid::controller::HumanoidController;
    use symthaea_humanoid::encoder::HumanoidHdcEncoder;
    use symthaea_humanoid::fep_agent::{ActiveInferenceHumanoidAgent, HumanoidFepConfig};
    use symthaea_humanoid::perturbations::{HumanoidPerturbation, PerturbationSchedule};
    use symthaea_humanoid::simulator::{HumanoidPhysicsSimulator, SimpleHumanoidSimulator};
    use symthaea_humanoid::types::*;

    println!("━━━ Battery 3: Perturbation Recovery Time ━━━");
    println!("  Chest shoves, ice floors, mass changes. Measure recovery steps.");
    println!();

    let (mut controller, mut encoder, genesis) = train_controller("stress-perturb", 30);

    // Define perturbation scenarios
    let scenarios: Vec<(&str, f64, HumanoidPerturbation)> = vec![
        // Chest shoves
        (
            "shove",
            100.0,
            HumanoidPerturbation::ExternalPush {
                force: [100.0, 0.0, 0.0],
                at_step: 300,
                duration: 5,
            },
        ),
        (
            "shove",
            250.0,
            HumanoidPerturbation::ExternalPush {
                force: [250.0, 0.0, 0.0],
                at_step: 300,
                duration: 5,
            },
        ),
        (
            "shove",
            500.0,
            HumanoidPerturbation::ExternalPush {
                force: [500.0, 0.0, 0.0],
                at_step: 300,
                duration: 5,
            },
        ),
        (
            "shove",
            1000.0,
            HumanoidPerturbation::ExternalPush {
                force: [1000.0, 0.0, 0.0],
                at_step: 300,
                duration: 5,
            },
        ),
        (
            "shove",
            2000.0,
            HumanoidPerturbation::ExternalPush {
                force: [2000.0, 0.0, 0.0],
                at_step: 300,
                duration: 5,
            },
        ),
        // Ice floor
        (
            "ice",
            0.8,
            HumanoidPerturbation::FloorFriction {
                friction: 0.8,
                at_step: 300,
            },
        ),
        (
            "ice",
            0.5,
            HumanoidPerturbation::FloorFriction {
                friction: 0.5,
                at_step: 300,
            },
        ),
        (
            "ice",
            0.3,
            HumanoidPerturbation::FloorFriction {
                friction: 0.3,
                at_step: 300,
            },
        ),
        (
            "ice",
            0.2,
            HumanoidPerturbation::FloorFriction {
                friction: 0.2,
                at_step: 300,
            },
        ),
        (
            "ice",
            0.1,
            HumanoidPerturbation::FloorFriction {
                friction: 0.1,
                at_step: 300,
            },
        ),
        // Mass change
        (
            "mass",
            1.1,
            HumanoidPerturbation::MassChange {
                body: "torso".to_string(),
                factor: 1.1,
                at_step: 300,
            },
        ),
        (
            "mass",
            1.2,
            HumanoidPerturbation::MassChange {
                body: "torso".to_string(),
                factor: 1.2,
                at_step: 300,
            },
        ),
        (
            "mass",
            1.3,
            HumanoidPerturbation::MassChange {
                body: "torso".to_string(),
                factor: 1.3,
                at_step: 300,
            },
        ),
        (
            "mass",
            1.5,
            HumanoidPerturbation::MassChange {
                body: "torso".to_string(),
                factor: 1.5,
                at_step: 300,
            },
        ),
        (
            "mass",
            2.0,
            HumanoidPerturbation::MassChange {
                body: "torso".to_string(),
                factor: 2.0,
                at_step: 300,
            },
        ),
    ];

    println!(
        "{:>8} {:>8} {:>12} {:>10} {:>14} {:>6}",
        "Type", "Severity", "PreReward", "MinReward", "RecoverySteps", "Fell"
    );

    let mut csv_rows: Vec<String> = Vec::new();
    csv_rows.push("perturbation_type,severity,pre_reward,min_reward,recovery_steps,final_reward,fell,pe_spike".to_string());

    let dt = 0.025;
    let total_steps = 1000;

    for (ptype, severity, perturbation) in &scenarios {
        let mut sim = SimpleHumanoidSimulator::new();
        let mut schedule = PerturbationSchedule::new().with(perturbation.clone());
        encoder.reset();
        controller.reset();

        let mut pre_rewards: Vec<f64> = Vec::new();
        let mut all_rewards: Vec<f64> = Vec::new();
        let mut max_pe: f32 = 0.0;
        let mut fell = false;

        for step in 0..total_steps {
            let state = sim.state().clone();
            let hv = encoder.encode(&state);
            let learned_cmd = controller.forward(&hv, dt as f32);

            // Combined: PD baseline + learned corrections
            let pd_gains = HumanoidPdGains::default();
            let pd_cmd = pd_standing_baseline(&state, &pd_gains);
            let pd_blend = 0.5;
            let mut cmd = HumanoidCommand::zero();
            for i in 0..cmd.torques.len().min(learned_cmd.torques.len()) {
                cmd.torques[i] =
                    pd_blend * pd_cmd.torques[i] + (1.0 - pd_blend) * learned_cmd.torques[i];
                cmd.torques[i] = cmd.torques[i].clamp(-1.0, 1.0);
            }

            schedule.apply(step, &mut cmd, &mut sim);

            sim.step(&cmd, dt);

            let s = sim.state();
            let reward = standing_reward_simple(s);
            let pe = encoder.prediction_error();
            if pe > max_pe {
                max_pe = pe;
            }

            if step < 300 {
                pre_rewards.push(reward);
            }
            all_rewards.push(reward);

            if s.head_height < 0.5 {
                fell = true;
            }
        }

        let pre_mean = if pre_rewards.is_empty() {
            0.0
        } else {
            pre_rewards.iter().sum::<f64>() / pre_rewards.len() as f64
        };
        let min_reward = all_rewards.iter().copied().fold(f64::MAX, f64::min);
        let final_reward = all_rewards.last().copied().unwrap_or(0.0);

        // Recovery: first step after 300 where reward > 50% of pre-mean
        let recovery_threshold = pre_mean * 0.5;
        let recovery_steps = all_rewards
            .iter()
            .enumerate()
            .skip(300)
            .position(|(_, &r)| r >= recovery_threshold)
            .map(|pos| pos);

        let recovery_str = recovery_steps
            .map(|s| format!("{}", s))
            .unwrap_or("never".to_string());

        println!(
            "{:>8} {:>8.1} {:>12.4} {:>10.4} {:>14} {:>6}",
            ptype,
            severity,
            pre_mean,
            min_reward,
            recovery_str,
            if fell { "YES" } else { "no" }
        );

        csv_rows.push(format!(
            "{},{:.2},{:.6},{:.6},{},{:.6},{},{:.6}",
            ptype,
            severity,
            pre_mean,
            min_reward,
            recovery_steps
                .map(|s| s.to_string())
                .unwrap_or("-1".to_string()),
            final_reward,
            fell,
            max_pe
        ));
    }

    let csv = csv_rows.join("\n");
    let _ = std::fs::write("papers/stress_test/perturbation_recovery.csv", &csv);
    println!(
        "  Output: papers/stress_test/perturbation_recovery.csv ({} rows)",
        csv_rows.len()
    );
    println!();
}

// ═══════════════════════════════════════════════════════════════
// Shared utilities
// ═══════════════════════════════════════════════════════════════

#[cfg(feature = "humanoid")]
fn train_controller(
    seed: &str,
    episodes: usize,
) -> (
    symthaea_humanoid::controller::HumanoidController,
    symthaea_humanoid::encoder::HumanoidHdcEncoder,
    symthaea_core::genesis::GenesisSeed,
) {
    use symthaea_core::genesis::GenesisSeed;
    use symthaea_humanoid::training::HumanoidTrainer;
    use symthaea_humanoid::types::*;

    let config = HumanoidConfig {
        num_episodes: episodes,
        steps_per_episode: 1000,
        task: HumanoidTask::Stand,
        adaptive_curriculum: false,
        domain_randomization: false,
        genesis_phrase: seed.to_string(),
        ..HumanoidConfig::default()
    };

    println!("  Training {} episodes (Stand, seed={})...", episodes, seed);
    let start = std::time::Instant::now();
    let mut trainer = HumanoidTrainer::new(config.clone());

    // Use train_and_extract() to get the ACTUALLY TRAINED controller + encoder
    let (metrics, controller, encoder) = trainer.train_and_extract();
    let elapsed = start.elapsed();

    if let Some(last) = metrics.last() {
        println!(
            "  Trained in {:.1}s — standing={:.4}, upright={:.4}",
            elapsed.as_secs_f64(),
            last.avg_standing_reward,
            last.avg_uprightness
        );
    }

    let genesis = GenesisSeed::from_phrase(seed);
    (controller, encoder, genesis)
}

#[cfg(feature = "humanoid")]
fn standing_reward_simple(state: &symthaea_humanoid::types::HumanoidState) -> f64 {
    let head_tol = if state.head_height >= 1.2 {
        1.0
    } else {
        (state.head_height / 1.2).max(0.0)
    };
    let upright = state.torso_vertical[2].max(0.0);
    head_tol * upright
}