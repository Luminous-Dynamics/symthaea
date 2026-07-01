// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # CMA-ES Walking: PD + Learned Gait Correction
//!
//! Standing is linear (PD optimal). Walking is NONLINEAR — requires
//! phase-dependent gait patterns. This is where HDC-LTC earns its keep.
//!
//! Optimizes: PD walking baseline + gain × learned_correction
//! Fitness: forward speed + standing reward - energy cost
//!
//! ```bash
//! cargo run --example cmaes_walking --release --features humanoid
//! ```

fn main() {
    #[cfg(not(feature = "humanoid"))]
    {
        eprintln!("ERROR: Requires --features humanoid");
        std::process::exit(1);
    }

    #[cfg(feature = "humanoid")]
    run();
}

#[cfg(feature = "humanoid")]
fn run() {
    use std::io::Write;
    use std::time::Instant;
    use symthaea_core::genesis::GenesisSeed;
    use symthaea_humanoid::controller::HumanoidController;
    use symthaea_humanoid::encoder::HumanoidHdcEncoder;
    use symthaea_humanoid::simulator::{HumanoidPhysicsSimulator, SimpleHumanoidSimulator};
    use symthaea_humanoid::types::*;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  CMA-ES Walking: PD + Learned Gait Correction               ║");
    println!("║  Where HDC-LTC temporal dynamics actually matter             ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let config = HumanoidConfig {
        genesis_phrase: "cmaes-walking-v1".to_string(),
        network_layers: 1,
        neurons_per_layer: 2,
        ..HumanoidConfig::default()
    };

    let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
    let pd_gains = HumanoidPdGains::default();

    let base_controller = HumanoidController::new(&genesis, &config);
    let (init_weights, init_bias) = base_controller.output_projection();
    let n_weights = init_weights.len();
    let n_bias = init_bias.len();
    let n_params = n_weights + n_bias + 1; // +1 for residual gain

    println!("  Parameters: {}", n_params);

    let pop_size = 24;
    let n_generations = 80;
    let eval_steps = 400; // 10 seconds at 40Hz
    let target_speed = 1.0; // 1 m/s walking
    let dt = 0.025;

    let mut mean: Vec<f64> = vec![0.0; n_params];
    mean[n_params - 1] = 0.15; // Start with moderate residual gain
    let mut sigma: f64 = 0.08;

    let mut step_sizes: Vec<f64> = vec![1.0; n_params];
    let mut path_s: Vec<f64> = vec![0.0; n_params];
    let c_s = 0.3;
    let c_c = 0.4 / (n_params as f64).sqrt();
    let damp = 1.0 + n_params as f64 / pop_size as f64;

    println!(
        "  Pop: {pop_size}, Gens: {n_generations}, Steps: {eval_steps}, Target: {target_speed} m/s"
    );
    println!();
    std::io::stdout().flush().unwrap_or(());

    let start = Instant::now();
    let mut best_ever_fitness = f64::NEG_INFINITY;
    let mut best_ever_params: Vec<f64> = mean.clone();

    println!(
        "{:>5} {:>10} {:>10} {:>10} {:>7} {:>7} {:>7}",
        "Gen", "BestFit", "MeanFit", "BestEver", "Sigma", "Gain", "Speed"
    );
    std::io::stdout().flush().unwrap_or(());

    for r#gen in 0..n_generations {
        let mut population: Vec<Vec<f64>> = Vec::with_capacity(pop_size);
        let mut fitnesses: Vec<f64> = Vec::with_capacity(pop_size);
        let mut best_speed = 0.0f64;

        for p in 0..pop_size {
            let mut candidate = vec![0.0f64; n_params];
            let mut rng = (r#gen * pop_size + p + 42) as u64;
            for i in 0..n_params {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let u1 = (rng as f64 / u64::MAX as f64).max(1e-10);
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let u2 = rng as f64 / u64::MAX as f64;
                let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                candidate[i] = mean[i] + sigma * step_sizes[i] * normal;
            }
            candidate[n_params - 1] = candidate[n_params - 1].clamp(0.0, 0.5);

            let (fitness, speed) = evaluate_walking(
                &candidate,
                n_weights,
                &genesis,
                &config,
                &pd_gains,
                eval_steps,
                dt,
                target_speed,
            );
            if speed > best_speed {
                best_speed = speed;
            }
            population.push(candidate);
            fitnesses.push(fitness);
        }

        let mut ranked: Vec<(usize, f64)> =
            fitnesses.iter().enumerate().map(|(i, &f)| (i, f)).collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let best_fitness = ranked[0].1;
        let mean_fitness: f64 = fitnesses.iter().sum::<f64>() / pop_size as f64;

        if best_fitness > best_ever_fitness {
            best_ever_fitness = best_fitness;
            best_ever_params = population[ranked[0].0].clone();
        }

        // CMA-ES update
        let n_elite = pop_size / 2;
        let mut new_mean = vec![0.0f64; n_params];
        let weight_sum: f64 = (1..=n_elite)
            .map(|i| (n_elite as f64 + 0.5 - i as f64).max(0.0))
            .sum();
        for rank in 0..n_elite {
            let idx = ranked[rank].0;
            let w = (n_elite as f64 + 0.5 - rank as f64).max(0.0) / weight_sum;
            for i in 0..n_params {
                new_mean[i] += w * population[idx][i];
            }
        }

        for i in 0..n_params {
            let diff = (new_mean[i] - mean[i]) / (sigma * step_sizes[i]).max(1e-10);
            path_s[i] = (1.0 - c_c) * path_s[i] + c_c.sqrt() * (2.0 - c_c).sqrt() * diff;
            step_sizes[i] *= (c_s / damp * (path_s[i].abs() / 0.7979 - 1.0)).exp();
            step_sizes[i] = step_sizes[i].clamp(0.01, 10.0);
        }

        let path_norm: f64 = path_s.iter().map(|p| p * p).sum::<f64>().sqrt();
        let expected_norm = (n_params as f64).sqrt() * 0.7979;
        sigma *= (c_s / damp * (path_norm / expected_norm - 1.0)).exp();
        sigma = sigma.clamp(0.001, 0.5);

        mean = new_mean;

        if r#gen % 5 == 0 || r#gen == n_generations - 1 {
            let gain = best_ever_params[n_params - 1];
            println!(
                "{:>5} {:>10.4} {:>10.4} {:>10.4} {:>7.4} {:>7.3} {:>7.3}",
                r#gen, best_fitness, mean_fitness, best_ever_fitness, sigma, gain, best_speed
            );
            std::io::stdout().flush().unwrap_or(());
        }
    }

    let elapsed = start.elapsed();
    println!();
    println!("━━━ Complete ({:.1} min) ━━━", elapsed.as_secs_f64() / 60.0);
    println!("  Best ever fitness: {:.4}", best_ever_fitness);
    let best_gain = best_ever_params[n_params - 1];
    println!("  Best gain: {:.4}", best_gain);

    // Final comparison: PD alone vs PD + learned
    println!();
    println!("━━━ Final Comparison: 1000 steps ━━━");

    let (pd_fit, pd_speed) = evaluate_walking(
        &vec![0.0; n_params],
        n_weights,
        &genesis,
        &config,
        &pd_gains,
        1000,
        dt,
        target_speed,
    );
    let (best_fit, best_speed) = evaluate_walking(
        &best_ever_params,
        n_weights,
        &genesis,
        &config,
        &pd_gains,
        1000,
        dt,
        target_speed,
    );

    println!(
        "  PD alone:      fitness={:.4}, speed={:.4} m/s",
        pd_fit, pd_speed
    );
    println!(
        "  PD + learned:  fitness={:.4}, speed={:.4} m/s (gain={:.3})",
        best_fit, best_speed, best_gain
    );

    if best_fit > pd_fit {
        let improvement = (best_fit - pd_fit) / pd_fit.abs().max(0.001) * 100.0;
        println!(
            "  *** LEARNED CORRECTION IMPROVES WALKING BY {:.1}% ***",
            improvement
        );
    } else {
        println!("  PD alone is still optimal for walking.");
    }
    std::io::stdout().flush().unwrap_or(());
}

#[cfg(feature = "humanoid")]
fn evaluate_walking(
    params: &[f64],
    n_weights: usize,
    genesis: &symthaea_core::genesis::GenesisSeed,
    config: &symthaea_humanoid::types::HumanoidConfig,
    pd_gains: &symthaea_humanoid::types::HumanoidPdGains,
    eval_steps: usize,
    dt: f64,
    target_speed: f64,
) -> (f64, f64) {
    use symthaea_humanoid::controller::HumanoidController;
    use symthaea_humanoid::encoder::HumanoidHdcEncoder;
    use symthaea_humanoid::simulator::{HumanoidPhysicsSimulator, SimpleHumanoidSimulator};
    use symthaea_humanoid::types::*;

    let n_bias = config.morphology.num_actuators();
    let n_params = params.len();
    let residual_gain = params[n_params - 1].clamp(0.0, 0.5) as f32;

    let mut controller = HumanoidController::new(genesis, config);
    let weights: Vec<f32> = params[..n_weights].iter().map(|&w| w as f32).collect();
    let bias: Vec<f32> = params[n_weights..n_weights + n_bias]
        .iter()
        .map(|&w| w as f32)
        .collect();
    controller.set_output_projection(&weights, &bias);

    let mut encoder = HumanoidHdcEncoder::new(genesis, config.num_levels);
    let mut sim = SimpleHumanoidSimulator::new().with_ground_contact(true);

    let mut total_reward = 0.0;
    let mut total_speed = 0.0;
    let gait_freq = 1.5; // Hz — natural walking cadence

    for step in 0..eval_steps {
        let state = sim.state().clone();
        let hv = encoder.encode(&state);
        let learned = controller.forward(&hv, dt as f32);

        // PD walking baseline with gait phase
        let gait_phase = (step as f64 * dt * gait_freq).fract();
        let pd = pd_walking_baseline(&state, pd_gains, gait_phase, target_speed);

        // Residual: PD + gain × learned
        let mut cmd = HumanoidCommand::zero();
        for i in 0..cmd.torques.len().min(pd.torques.len()) {
            cmd.torques[i] = (pd.torques[i] + residual_gain * learned.torques[i]).clamp(-1.0, 1.0);
        }
        sim.step(&cmd, dt);

        let s = sim.state();
        let speed = s.horizontal_speed();
        total_speed += speed;

        // Walking fitness: speed toward target + staying upright - energy
        let speed_reward = 1.0 - ((speed - target_speed).abs() / target_speed).min(1.0);
        let upright = s.torso_vertical[2].max(0.0);
        let head_ok = if s.head_height >= 1.0 {
            1.0
        } else {
            (s.head_height / 1.0).max(0.0)
        };
        let effort = cmd.control_effort() as f64 * 0.05;

        total_reward += speed_reward * upright * head_ok - effort;

        if s.head_height < 0.5 {
            total_reward -= (eval_steps - step) as f64 * 0.3;
            break;
        }
    }

    let mean_speed = total_speed / eval_steps as f64;
    (total_reward / eval_steps as f64, mean_speed)
}