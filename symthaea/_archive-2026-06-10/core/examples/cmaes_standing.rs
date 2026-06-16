// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # CMA-ES Independent Standing
//!
//! Evolutionary optimization of the output projection weights.
//! Gradient-free — bypasses the distribution shift problem entirely.
//! Evaluates each candidate by running the controller ALONE (no PD).
//!
//! Population: 32, Parameters: ~1.4K (bottleneck), Generations: 500
//!
//! ```bash
//! cargo run --example cmaes_standing --release --features humanoid
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
    use std::time::Instant;
    use symthaea_core::genesis::GenesisSeed;
    use symthaea_humanoid::controller::HumanoidController;
    use symthaea_humanoid::encoder::HumanoidHdcEncoder;
    use symthaea_humanoid::simulator::{HumanoidPhysicsSimulator, SimpleHumanoidSimulator};
    use symthaea_humanoid::types::*;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  CMA-ES Independent Standing                                ║");
    println!("║  Gradient-free optimization of 1.4K output weights          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let config = HumanoidConfig {
        num_episodes: 1,
        steps_per_episode: 500,
        task: HumanoidTask::Stand,
        genesis_phrase: "cmaes-standing-v1".to_string(),
        ..HumanoidConfig::default()
    };

    let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
    let base_controller = HumanoidController::new(&genesis, &config);

    // Get parameter dimensionality from the output projection
    let (init_weights, init_bias) = base_controller.output_projection();
    let n_weights = init_weights.len();
    let n_bias = init_bias.len();
    let n_params = n_weights + n_bias;

    println!(
        "  Parameters: {} weights + {} bias = {} total",
        n_weights, n_bias, n_params
    );

    // CMA-ES hyperparameters
    let pop_size = 16; // Smaller population for speed
    let n_generations = 100; // Quick test — scale up if converging
    let eval_steps = 300; // Steps per fitness evaluation (no PD!)
    let dt = 0.025;

    // Initial mean = current weights, initial sigma = 0.1
    let mut mean: Vec<f64> = init_weights
        .iter()
        .chain(init_bias.iter())
        .map(|&w| w as f64)
        .collect();
    let mut sigma: f64 = 0.1;

    // Diagonal CMA-ES (separable): per-parameter step sizes
    let mut step_sizes: Vec<f64> = vec![1.0; n_params];
    let mut path_s: Vec<f64> = vec![0.0; n_params]; // Evolution path

    // Learning rates
    let c_s = 0.3; // Step-size adaptation rate
    let c_c = 0.4 / (n_params as f64).sqrt(); // Path update rate
    let damp = 1.0 + n_params as f64 / pop_size as f64; // Damping

    println!("  Population: {pop_size}, Generations: {n_generations}, Eval steps: {eval_steps}");
    println!("  Initial sigma: {sigma}");
    println!();

    let start = Instant::now();
    let mut best_ever_fitness = f64::NEG_INFINITY;
    let mut best_ever_params: Vec<f64> = mean.clone();

    println!(
        "{:>5} {:>12} {:>12} {:>12} {:>8}",
        "Gen", "BestFit", "MeanFit", "BestEver", "Sigma"
    );

    for r#gen in 0..n_generations {
        // Sample population
        let mut population: Vec<Vec<f64>> = Vec::with_capacity(pop_size);
        let mut fitnesses: Vec<f64> = Vec::with_capacity(pop_size);

        for p in 0..pop_size {
            // Sample: x = mean + sigma * step_sizes * N(0,1)
            let mut candidate = vec![0.0f64; n_params];
            let mut rng_state = (r#gen * pop_size + p + 42) as u64;
            for i in 0..n_params {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                // Box-Muller for normal distribution
                let u1 = (rng_state as f64 / u64::MAX as f64).max(1e-10);
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                let u2 = rng_state as f64 / u64::MAX as f64;
                let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                candidate[i] = mean[i] + sigma * step_sizes[i] * normal;
            }

            // Evaluate fitness: run controller ALONE for eval_steps
            let fitness =
                evaluate_candidate(&candidate, n_weights, &genesis, &config, eval_steps, dt);

            population.push(candidate);
            fitnesses.push(fitness);
        }

        // Sort by fitness (descending)
        let mut ranked: Vec<(usize, f64)> =
            fitnesses.iter().enumerate().map(|(i, &f)| (i, f)).collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let best_fitness = ranked[0].1;
        let mean_fitness: f64 = fitnesses.iter().sum::<f64>() / pop_size as f64;

        if best_fitness > best_ever_fitness {
            best_ever_fitness = best_fitness;
            best_ever_params = population[ranked[0].0].clone();
        }

        // Update mean: weighted average of top 50%
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

        // Update evolution path and step sizes
        for i in 0..n_params {
            let diff = (new_mean[i] - mean[i]) / (sigma * step_sizes[i]).max(1e-10);
            path_s[i] = (1.0 - c_c) * path_s[i] + c_c.sqrt() * (2.0 - c_c).sqrt() * diff;
            step_sizes[i] *= (c_s / damp * (path_s[i].abs() / 0.7979 - 1.0)).exp(); // 0.7979 ≈ E[|N(0,1)|]
            step_sizes[i] = step_sizes[i].clamp(0.01, 10.0);
        }

        // Update sigma (global step size)
        let path_norm: f64 = path_s.iter().map(|p| p * p).sum::<f64>().sqrt();
        let expected_norm = (n_params as f64).sqrt() * 0.7979;
        sigma *= (c_s / damp * (path_norm / expected_norm - 1.0)).exp();
        sigma = sigma.clamp(0.001, 1.0);

        mean = new_mean;

        if r#gen % 10 == 0 || r#gen == n_generations - 1 {
            println!(
                "{:>5} {:>12.4} {:>12.4} {:>12.4} {:>8.5}",
                r#gen, best_fitness, mean_fitness, best_ever_fitness, sigma
            );
        }
    }

    let elapsed = start.elapsed();
    println!();
    println!(
        "━━━ CMA-ES Complete ({:.1} min) ━━━",
        elapsed.as_secs_f64() / 60.0
    );
    println!("  Best ever fitness: {:.4}", best_ever_fitness);
    println!("  Total evaluations: {}", n_generations * pop_size);

    // Final evaluation with best-ever parameters
    println!();
    println!("━━━ Final Evaluation (best-ever weights, 2000 steps, no PD) ━━━");

    let final_fitness =
        evaluate_candidate(&best_ever_params, n_weights, &genesis, &config, 2000, dt);

    // Detailed eval
    let mut controller = HumanoidController::new(&genesis, &config);
    let weights: Vec<f32> = best_ever_params[..n_weights]
        .iter()
        .map(|&w| w as f32)
        .collect();
    let bias: Vec<f32> = best_ever_params[n_weights..]
        .iter()
        .map(|&w| w as f32)
        .collect();
    controller.set_output_projection(&weights, &bias);

    let mut encoder = HumanoidHdcEncoder::new(&genesis, config.num_levels);
    let mut sim = SimpleHumanoidSimulator::new();

    let mut steps_standing = 0;
    let mut fell = false;
    let mut fall_step = 0;
    for step in 0..2000 {
        let state = sim.state().clone();
        let hv = encoder.encode(&state);
        let cmd = controller.forward(&hv, dt as f32);
        sim.step(&cmd, dt);
        let s = sim.state();
        if s.head_height >= 0.8 {
            steps_standing += 1;
        }
        if s.head_height < 0.5 && !fell {
            fell = true;
            fall_step = step;
        }
    }

    let standing_pct = steps_standing as f64 / 2000.0 * 100.0;
    println!("  Fitness: {:.4}", final_fitness);
    println!(
        "  Standing: {} / 2000 ({:.1}%)",
        steps_standing, standing_pct
    );
    if fell {
        println!("  Fell at step {}", fall_step);
    } else {
        println!("  *** DID NOT FALL — INDEPENDENT STANDING ACHIEVED ***");
    }
}

#[cfg(feature = "humanoid")]
fn evaluate_candidate(
    params: &[f64],
    n_weights: usize,
    genesis: &symthaea_core::genesis::GenesisSeed,
    config: &symthaea_humanoid::types::HumanoidConfig,
    eval_steps: usize,
    dt: f64,
) -> f64 {
    use symthaea_humanoid::controller::HumanoidController;
    use symthaea_humanoid::encoder::HumanoidHdcEncoder;
    use symthaea_humanoid::simulator::{HumanoidPhysicsSimulator, SimpleHumanoidSimulator};

    let mut controller = HumanoidController::new(genesis, config);
    let weights: Vec<f32> = params[..n_weights].iter().map(|&w| w as f32).collect();
    let bias: Vec<f32> = params[n_weights..].iter().map(|&w| w as f32).collect();
    controller.set_output_projection(&weights, &bias);

    let mut encoder = HumanoidHdcEncoder::new(genesis, config.num_levels);
    let mut sim = SimpleHumanoidSimulator::new();

    let mut total_reward = 0.0;
    for _step in 0..eval_steps {
        let state = sim.state().clone();
        let hv = encoder.encode(&state);
        let cmd = controller.forward(&hv, dt as f32);
        sim.step(&cmd, dt);

        let s = sim.state();
        // Reward: head height + uprightness + low control effort
        let head_reward = if s.head_height >= 1.2 {
            1.0
        } else {
            (s.head_height / 1.4).max(0.0)
        };
        let upright = s.torso_vertical[2].max(0.0);
        let effort_penalty = cmd.control_effort() as f64 * 0.1;
        total_reward += head_reward * upright - effort_penalty;

        // Early termination bonus: penalize falling
        if s.head_height < 0.5 {
            total_reward -= (eval_steps - _step) as f64 * 0.5; // Penalize remaining steps
            break;
        }
    }

    total_reward / eval_steps as f64
}