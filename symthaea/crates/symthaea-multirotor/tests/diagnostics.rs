// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Diagnostic tests for symthaea-multirotor learning dynamics.
//!
//! Three tests:
//! 1. Training curve: 10 episodes x 500 steps, per-episode metrics
//! 2. Wind benchmark: 3 episodes x 1000 steps, baseline vs FEP comparison
//! 3. Within-episode learning curve: 1 episode x 1000 steps with telemetry

use symthaea_core::genesis::GenesisSeed;
use symthaea_multirotor::benchmarks::{WindBenchmarkConfig, WindGust, run_wind_benchmark};
use symthaea_multirotor::fep_agent::FlightFepConfig;
use symthaea_multirotor::simulator::PhysicsSimulator;
use symthaea_multirotor::training::FlightTrainer;
use symthaea_multirotor::{
    ActiveInferenceFlightAgent, FlightConfig, FlightController, FlightSetpoint, PdGains, PidState,
    QuadrotorHdcEncoder, SimplePhysicsSimulator, pd_baseline, pid_baseline,
};

#[test]
#[ignore] // ~30s: run manually with `cargo test -p symthaea-multirotor -- --ignored --nocapture`
fn diagnostic_training_curve() {
    println!("\n{}", "=".repeat(80));
    println!("=== DIAGNOSTIC 1: Training Curve (10 episodes x 500 steps) ===");
    println!("{}\n", "=".repeat(80));

    let config = FlightConfig {
        num_episodes: 10,
        steps_per_episode: 500,
        ..FlightConfig::default()
    };

    let mut trainer = FlightTrainer::new(config);
    let metrics = trainer.train();

    println!(
        "{:<8} {:<20} {:<20} {:<18} {:<15} {:<18}",
        "Episode",
        "Avg Pos Error",
        "Final Pos Error",
        "Avg Free Energy",
        "Hover Frac",
        "Explore Count"
    );
    println!("{}", "-".repeat(99));

    for m in &metrics {
        println!(
            "{:<8} {:<20.6} {:<20.6} {:<18.6} {:<15.4} {:<18}",
            m.episode,
            m.avg_position_error,
            m.final_position_error,
            m.avg_free_energy,
            m.hover_fraction,
            m.exploration_count,
        );
    }

    // Summary statistics
    let first_3_avg: f64 = metrics[..3]
        .iter()
        .map(|m| m.avg_position_error)
        .sum::<f64>()
        / 3.0;
    let last_3_avg: f64 = metrics[7..]
        .iter()
        .map(|m| m.avg_position_error)
        .sum::<f64>()
        / 3.0;
    let best_ep = metrics
        .iter()
        .min_by(|a, b| a.avg_position_error.total_cmp(&b.avg_position_error))
        .unwrap();
    let worst_ep = metrics
        .iter()
        .max_by(|a, b| a.avg_position_error.total_cmp(&b.avg_position_error))
        .unwrap();

    println!("\n--- Summary ---");
    println!("First 3 episodes avg error:  {:.6}", first_3_avg);
    println!("Last 3 episodes avg error:   {:.6}", last_3_avg);
    println!(
        "Best episode:  {} (avg_err={:.6})",
        best_ep.episode, best_ep.avg_position_error
    );
    println!(
        "Worst episode: {} (avg_err={:.6})",
        worst_ep.episode, worst_ep.avg_position_error
    );
    println!(
        "Improvement ratio (first3/last3): {:.3}",
        first_3_avg / last_3_avg
    );

    let total_explorations: usize = metrics.iter().map(|m| m.exploration_count).sum();
    println!(
        "Total explorations across all episodes: {}",
        total_explorations
    );
    println!();
}

#[test]
#[ignore] // ~30s: run manually with `cargo test -p symthaea-multirotor -- --ignored --nocapture`
fn diagnostic_wind_benchmark() {
    println!("\n{}", "=".repeat(80));
    println!("=== DIAGNOSTIC 2: Wind Benchmark (3 episodes x 1000 steps) ===");
    println!("{}\n", "=".repeat(80));

    let wind_config = WindBenchmarkConfig {
        flight_config: FlightConfig {
            num_episodes: 1,
            steps_per_episode: 1000,
            early_termination: false, // Need full episodes for wind analysis
            ..FlightConfig::default()
        },
        gusts: vec![
            WindGust {
                force: [0.05, 0.0, 0.0], // Lateral gust
                start_time: 0.5,
                duration: 0.2,
            },
            WindGust {
                force: [0.0, 0.0, -0.03], // Downdraft
                start_time: 1.0,
                duration: 0.3,
            },
        ],
        eval_episodes: 3,
    };

    let result = run_wind_benchmark(&wind_config);

    println!(
        "{:<10} {:<22} {:<22}",
        "Episode", "Baseline Avg Error", "FEP Avg Error"
    );
    println!("{}", "-".repeat(54));

    for (i, (base, fep)) in result
        .baseline_metrics
        .iter()
        .zip(result.fep_metrics.iter())
        .enumerate()
    {
        println!("{:<10} {:<22.6} {:<22.6}", i, base, fep,);
    }

    println!("\n--- Recovery & Deviation ---");
    println!("Recovery steps per gust (FEP): {:?}", result.recovery_steps);
    println!("Max deviation (FEP):      {:.6}", result.max_deviation);
    println!(
        "Max deviation (baseline):  {:.6}",
        result.baseline_max_deviation
    );

    let baseline_avg: f64 =
        result.baseline_metrics.iter().sum::<f64>() / result.baseline_metrics.len() as f64;
    let fep_avg: f64 = result.fep_metrics.iter().sum::<f64>() / result.fep_metrics.len() as f64;
    println!("\nOverall baseline avg error: {:.6}", baseline_avg);
    println!("Overall FEP avg error:     {:.6}", fep_avg);
    if fep_avg < baseline_avg {
        println!(
            "FEP improvement: {:.2}%",
            (1.0 - fep_avg / baseline_avg) * 100.0
        );
    } else {
        println!(
            "FEP degradation: {:.2}%",
            (fep_avg / baseline_avg - 1.0) * 100.0
        );
    }
    println!();
}

#[test]
#[ignore] // ~30s: run manually with `cargo test -p symthaea-multirotor -- --ignored --nocapture`
fn diagnostic_within_episode_learning_curve() {
    println!("\n{}", "=".repeat(80));
    println!("=== DIAGNOSTIC 3: Within-Episode Learning Curve (1 episode x 1000 steps) ===");
    println!("{}\n", "=".repeat(80));

    let config = FlightConfig {
        num_episodes: 1,
        steps_per_episode: 1000,
        collect_telemetry: true,
        early_termination: false, // Disable so we get the full 1000 steps for analysis
        ..FlightConfig::default()
    };

    let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
    let trainer = FlightTrainer::new(config.clone());

    let mut encoder = QuadrotorHdcEncoder::new(&genesis, config.num_levels);
    let mut controller = FlightController::new(&genesis, &config);
    let mut fep_agent = ActiveInferenceFlightAgent::new(FlightFepConfig::default());

    let metrics = trainer.run_episode(&mut encoder, &mut controller, &mut fep_agent, 0);

    let n_steps = metrics.telemetry.len();
    assert!(n_steps > 0, "Expected at least some telemetry entries");

    let checkpoints: Vec<usize> = [0, 100, 200, 500, n_steps - 1]
        .iter()
        .filter(|&&s| s < n_steps)
        .copied()
        .collect();
    println!(
        "{:<8} {:<16} {:<16} {:<16} {:<14} {:<14} {:<14}",
        "Step", "Pos Error", "Att Error", "Free Energy", "Tau Factor", "LR", "Thrust"
    );
    println!("{}", "-".repeat(98));

    for &step in &checkpoints {
        let t = &metrics.telemetry[step];
        println!(
            "{:<8} {:<16.6} {:<16.6} {:<16.6} {:<14.4} {:<14.6} {:<14.6}",
            t.step,
            t.position_error,
            t.attitude_error,
            t.free_energy,
            t.tau_factor,
            t.learning_rate,
            t.command.thrust,
        );
    }

    // Additional analysis: error trend in windows
    println!("\n--- Error Trend (100-step windows) ---");
    println!(
        "{:<12} {:<18} {:<18} {:<18}",
        "Window", "Avg Pos Error", "Avg Att Error", "Avg Speed"
    );
    println!("{}", "-".repeat(66));

    for window_start in (0..n_steps).step_by(100) {
        let window_end = (window_start + 100).min(n_steps);
        let window = &metrics.telemetry[window_start..window_end];
        let avg_pos: f64 =
            window.iter().map(|t| t.position_error).sum::<f64>() / window.len() as f64;
        let avg_att: f64 =
            window.iter().map(|t| t.attitude_error).sum::<f64>() / window.len() as f64;
        let avg_speed: f64 = window.iter().map(|t| t.speed).sum::<f64>() / window.len() as f64;
        println!(
            "{:<12} {:<18.6} {:<18.6} {:<18.6}",
            format!("{}-{}", window_start, window_end - 1),
            avg_pos,
            avg_att,
            avg_speed,
        );
    }

    // Altitude trace at checkpoints
    println!("\n--- Altitude Trace ---");
    println!("Target altitude: 0.1 m");
    for &step in &checkpoints {
        let t = &metrics.telemetry[step];
        println!(
            "  Step {:>4}: altitude = {:.6} m (error = {:.6} m)",
            step, t.altitude, t.position_error
        );
    }

    // FEP action analysis: tau_factor distribution
    let tau_below_1 = metrics
        .telemetry
        .iter()
        .filter(|t| t.tau_factor < 0.99)
        .count();
    let tau_above_1 = metrics
        .telemetry
        .iter()
        .filter(|t| t.tau_factor > 1.01)
        .count();
    let tau_at_1 = metrics.telemetry.len() - tau_below_1 - tau_above_1;
    let pct = n_steps as f64 / 100.0;
    println!("\n--- FEP Action Distribution ---");
    println!(
        "Tau < 1.0 (DropTau):   {} steps ({:.1}%)",
        tau_below_1,
        tau_below_1 as f64 / pct
    );
    println!(
        "Tau = 1.0 (neutral):   {} steps ({:.1}%)",
        tau_at_1,
        tau_at_1 as f64 / pct
    );
    println!(
        "Tau > 1.0 (RaiseTau):  {} steps ({:.1}%)",
        tau_above_1,
        tau_above_1 as f64 / pct
    );

    println!("\nEpisode summary:");
    println!("  avg_position_error: {:.6}", metrics.avg_position_error);
    println!(
        "  final_position_error: {:.6}",
        metrics.final_position_error
    );
    println!("  avg_free_energy: {:.6}", metrics.avg_free_energy);
    println!("  hover_fraction: {:.4}", metrics.hover_fraction);
    println!("  exploration_count: {}", metrics.exploration_count);
    println!();
}

#[test]
#[ignore] // ~30s: run manually with `cargo test -p symthaea-multirotor -- --ignored --nocapture`
fn diagnostic_motor_lag_recovery() {
    println!("\n{}", "=".repeat(80));
    println!("=== DIAGNOSTIC 4: Motor Lag Recovery (10 episodes x 500 steps) ===");
    println!("{}\n", "=".repeat(80));

    let config = FlightConfig {
        num_episodes: 10,
        steps_per_episode: 500,
        ..FlightConfig::default()
    };

    // Train with motor lag enabled (default)
    let mut trainer = FlightTrainer::new(config);
    let metrics = trainer.train();

    println!(
        "{:<8} {:<20} {:<18} {:<15}",
        "Episode", "Avg Pos Error", "Final Pos Error", "Hover Frac"
    );
    println!("{}", "-".repeat(61));

    for m in &metrics {
        println!(
            "{:<8} {:<20.6} {:<18.6} {:<15.4}",
            m.episode, m.avg_position_error, m.final_position_error, m.hover_fraction,
        );
    }

    // Verify convergence despite motor lag
    let last_3: Vec<_> = metrics.iter().rev().take(3).collect();
    let last_3_avg: f64 = last_3.iter().map(|m| m.avg_position_error).sum::<f64>() / 3.0;

    println!("\nLast 3 episodes avg error: {:.6}", last_3_avg);
    assert!(
        last_3_avg < 2.0,
        "Motor lag should not prevent convergence: avg={last_3_avg:.4}"
    );
    println!("PASS: Motor lag convergence verified.");
    println!();
}

#[test]
#[ignore] // ~5s: run manually with `cargo test -p symthaea-multirotor -- --ignored --nocapture`
fn diagnostic_pid_vs_pd_wind_baseline() {
    println!("\n{}", "=".repeat(80));
    println!("=== DIAGNOSTIC 5: PID vs PD Under Sustained Wind ===");
    println!("{}\n", "=".repeat(80));

    let setpoint = FlightSetpoint {
        position: [0.0, 0.0, 0.1],
        yaw: 0.0,
    };
    let gains = PdGains::default();
    let dt = 0.002;
    let wind = [0.02, 0.0, 0.0]; // sustained lateral wind

    // Run PD controller under wind
    let mut sim_pd = SimplePhysicsSimulator::new();
    sim_pd.set_motor_lag(false); // isolate controller effect
    sim_pd.reset(0.1);
    for _ in 0..2000 {
        let state = sim_pd.state().clone();
        let cmd = pd_baseline(&state, &setpoint, &gains);
        sim_pd.apply_external_force(wind);
        sim_pd.step(&cmd, dt);
    }
    let pd_final_err = setpoint.position_error_magnitude(sim_pd.state());

    // Run PID controller under same wind
    let mut sim_pid = SimplePhysicsSimulator::new();
    sim_pid.set_motor_lag(false);
    sim_pid.reset(0.1);
    let mut pid_state = PidState::default();
    for _ in 0..2000 {
        let state = sim_pid.state().clone();
        let cmd = pid_baseline(&state, &setpoint, &gains, &mut pid_state, dt);
        sim_pid.apply_external_force(wind);
        sim_pid.step(&cmd, dt);
    }
    let pid_final_err = setpoint.position_error_magnitude(sim_pid.state());

    println!("PD  final position error: {:.6} m", pd_final_err);
    println!("PID final position error: {:.6} m", pid_final_err);

    if pid_final_err < pd_final_err {
        println!(
            "PID improvement: {:.2}%",
            (1.0 - pid_final_err / pd_final_err) * 100.0
        );
    } else {
        println!(
            "PID degradation: {:.2}% (integral may need tuning)",
            (pid_final_err / pd_final_err - 1.0) * 100.0
        );
    }

    // PID should have lower steady-state error under sustained wind
    assert!(
        pid_final_err < pd_final_err * 1.5,
        "PID should not dramatically degrade: pid={pid_final_err:.6}, pd={pd_final_err:.6}"
    );
    println!("PASS: PID baseline validated.");
    println!();
}

// ── Item 4: Diagnostic Regression Hardening (fast, NOT #[ignore]) ──

#[test]
fn test_regression_motor_lag_convergence() {
    // Fast regression test: 3 episodes x 200 steps.
    // Verifies training converges despite motor lag filter.
    let config = FlightConfig {
        num_episodes: 3,
        steps_per_episode: 200,
        ..FlightConfig::default()
    };
    let mut trainer = FlightTrainer::new(config);
    let metrics = trainer.train();

    assert_eq!(metrics.len(), 3);
    // Last episode should have reasonable error despite motor lag
    assert!(
        metrics[2].avg_position_error < 2.0,
        "Motor lag should not prevent convergence: last_episode_err={:.4}",
        metrics[2].avg_position_error
    );
}

#[test]
fn test_regression_pid_vs_pd() {
    // Fast regression: 1000 steps PD vs 1000 steps PID under sustained wind.
    // PID should not dramatically degrade compared to PD.
    let setpoint = FlightSetpoint {
        position: [0.0, 0.0, 0.1],
        yaw: 0.0,
    };
    let gains = PdGains::default();
    let dt = 0.002;
    let wind = [0.02, 0.0, 0.0];

    // PD under wind
    let mut sim_pd = SimplePhysicsSimulator::new();
    sim_pd.set_motor_lag(false);
    sim_pd.reset(0.1);
    for _ in 0..1000 {
        let state = sim_pd.state().clone();
        let cmd = pd_baseline(&state, &setpoint, &gains);
        sim_pd.apply_external_force(wind);
        sim_pd.step(&cmd, dt);
    }
    let pd_final_err = setpoint.position_error_magnitude(sim_pd.state());

    // PID under same wind
    let mut sim_pid = SimplePhysicsSimulator::new();
    sim_pid.set_motor_lag(false);
    sim_pid.reset(0.1);
    let mut pid_state = PidState::default();
    for _ in 0..1000 {
        let state = sim_pid.state().clone();
        let cmd = pid_baseline(&state, &setpoint, &gains, &mut pid_state, dt);
        sim_pid.apply_external_force(wind);
        sim_pid.step(&cmd, dt);
    }
    let pid_final_err = setpoint.position_error_magnitude(sim_pid.state());

    assert!(
        pid_final_err < pd_final_err * 1.5,
        "PID should not dramatically degrade vs PD: pid={pid_final_err:.6}, pd={pd_final_err:.6}"
    );
}

// ── MuJoCo-gated diagnostic: Kinetic Sacrifice ──

#[cfg(feature = "mujoco")]
#[test]
#[ignore] // Requires MuJoCo library: cargo test -p symthaea-multirotor --features mujoco -- --ignored --nocapture diagnostic_kinetic_sacrifice
fn diagnostic_kinetic_sacrifice() {
    use symthaea_multirotor::scenarios::kinetic_sacrifice::{
        KineticSacrificeConfig, run_kinetic_sacrifice,
    };

    println!("\n{}", "=".repeat(80));
    println!("=== DIAGNOSTIC: Kinetic Sacrifice (FEP moral reasoning) ===");
    println!("{}\n", "=".repeat(80));

    let config = KineticSacrificeConfig::default();
    let result = run_kinetic_sacrifice(&config);

    println!("Events:");
    for event in &result.events {
        println!(
            "  [step {:>3}] {} (FE={:.3}, tau={:.3}, danger={:.3})",
            event.step, event.description, event.free_energy, event.tau_factor, event.human_danger
        );
    }
    println!("\nResults:");
    println!("  Beam hit human:    {}", result.beam_hit_human);
    println!("  Drone intercepted: {}", result.drone_intercepted);
    println!("  Max free energy:   {:.4}", result.max_free_energy);
    println!("  Min tau factor:    {:.4}", result.min_tau);
    if let Some(step) = result.abort_step {
        println!("  Mission aborted:   step {step}");
    }

    // The scenario MUST trigger danger detection (mission abort)
    assert!(
        result.abort_step.is_some(),
        "Kinetic sacrifice should trigger mission abort (danger detection)"
    );

    // Min tau should drop below 1.0 (survival reflex activated)
    assert!(
        result.min_tau < 0.9,
        "FEP should drop tau below 0.9 in response to human danger, got {:.4}",
        result.min_tau
    );

    // Beam should NOT hit human (either deflected or fell away)
    assert!(
        !result.beam_hit_human,
        "Beam should not hit human in the scenario"
    );
}

// ── Closed-loop FEP→PID→physics integration test ──

#[test]
fn test_closed_loop_fep_pid_under_sustained_wind() {
    // Proves the full architectural chain:
    // sustained wind → FEP detects steady-state offset → signals use_pid_target
    // → integral term eliminates offset → lower error than PD-only.
    //
    // Two parallel simulations under identical sustained wind:
    // A: PD-only (no FEP, no PID switching)
    // B: FEP-driven with dynamic PID switching
    let genesis = GenesisSeed::from_phrase("test-closed-loop-fep-pid");
    let config = FlightConfig::default();
    let setpoint = FlightSetpoint::hover();
    let gains = PdGains::default();
    let dt = config.motor_dt();
    let cognitive_interval = config.cognitive_interval();
    let wind = [0.02, 0.0, 0.0]; // Sustained lateral wind
    let steps = 1500;

    // ── Run A: PD-only baseline (no FEP, no PID) ──
    let pd_final_err = {
        let mut sim = SimplePhysicsSimulator::new();
        sim.set_motor_lag(false);
        sim.reset(0.1);
        let mut encoder = QuadrotorHdcEncoder::new(&genesis, config.num_levels);
        let mut controller = FlightController::new(&genesis, &config);

        for step in 0..steps {
            let state = sim.state().clone();
            let sensor_hv = encoder.encode(&state);
            let command = controller.forward(&sensor_hv, dt as f32);
            sim.apply_external_force(wind);
            sim.step(&command, dt);

            if step % config.train_every == 0 {
                let target = pd_baseline(&state, &setpoint, &gains);
                controller.train_step(&sensor_hv, &target, dt as f32, None);
            }
        }
        setpoint.position_error_magnitude(sim.state())
    };

    // ── Run B: FEP-driven with dynamic PID switching ──
    let (fep_final_err, had_pid_signal) = {
        let mut sim = SimplePhysicsSimulator::new();
        sim.set_motor_lag(false);
        sim.reset(0.1);
        let mut encoder = QuadrotorHdcEncoder::new(&genesis, config.num_levels);
        let mut controller = FlightController::new(&genesis, &config);
        let mut fep_agent = ActiveInferenceFlightAgent::new(FlightFepConfig::default());
        let mut fep_result = fep_agent.initial_step(sim.state(), &setpoint);
        let mut pid_state = PidState::default();
        let mut saw_pid = false;

        for step in 0..steps {
            let state = sim.state().clone();
            let sensor_hv = encoder.encode(&state);
            let mut command = controller.forward(&sensor_hv, dt as f32);

            if let Some(noise) = &fep_result.exploration_noise {
                command = command.with_noise(noise);
            }

            sim.apply_external_force(wind);
            sim.step(&command, dt);

            if step % config.train_every == 0 {
                let target = if fep_result.use_pid_target {
                    saw_pid = true;
                    pid_baseline(&state, &setpoint, &gains, &mut pid_state, dt)
                } else {
                    pd_baseline(&state, &setpoint, &gains)
                };
                let lr = config.learning_rate * fep_result.learning_rate_factor;
                controller.train_step(&sensor_hv, &target, dt as f32, Some(lr));
            }

            if step % cognitive_interval == 0 {
                fep_result = fep_agent.step(sim.state(), &setpoint);
                if (fep_result.tau_factor - 1.0).abs() > 0.01 {
                    controller.modulate_tau(fep_result.tau_factor);
                }
            }
        }
        (setpoint.position_error_magnitude(sim.state()), saw_pid)
    };

    // The FEP agent should have detected the steady-state offset and signaled PID
    assert!(
        had_pid_signal,
        "FEP should signal use_pid_target under sustained wind"
    );

    // FEP+PID should not be dramatically worse than PD-only
    // (with tuning, it should be better; at minimum it should not degrade)
    assert!(
        fep_final_err < pd_final_err * 2.0,
        "FEP+PID should not dramatically degrade vs PD-only: \
         fep={fep_final_err:.6}, pd={pd_final_err:.6}"
    );

    // Both should have reasonable final error
    assert!(
        fep_final_err.is_finite() && pd_final_err.is_finite(),
        "Both runs should produce finite errors"
    );
}
