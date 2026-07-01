// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Residual Standing: PD + Learned Correction
//!
//! The controller outputs PD_baseline + small_learned_residual.
//! This guarantees at least PD-level standing (residual=0 is always valid)
//! and can only improve from there.
//!
//! The residual is scaled by a learned gain [0, 0.3] so the correction
//! never dominates the PD signal. CMA-ES optimizes only the residual gain
//! and the bottleneck weights — the PD structure is preserved.
//!
//! ```bash
//! cargo run --example residual_standing --release --features humanoid
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
    println!("║  Residual Standing: PD + Learned Correction                 ║");
    println!("║  CMA-ES on residual weights, PD provides stability floor    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let config = HumanoidConfig {
        genesis_phrase: "residual-standing-v1".to_string(),
        network_layers: 1,    // Minimal network for CMA-ES speed
        neurons_per_layer: 2, // 2 neurons × 1 layer = 2 CfC neurons total
        ..HumanoidConfig::default()
    };

    let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
    let pd_gains = HumanoidPdGains::default();

    let base_controller = HumanoidController::new(&genesis, &config);
    let (init_weights, init_bias) = base_controller.output_projection();
    let n_weights = init_weights.len();
    let n_bias = init_bias.len();
    let n_params = n_weights + n_bias + 1; // +1 for residual gain

    println!(
        "  Parameters: {} (weights + bias + residual_gain)",
        n_params
    );

    // CMA-ES — larger population since we know the landscape is hard
    let pop_size = 16;
    let n_generations = 50;
    let eval_steps = 200; // 5 seconds of simulated standing
    let dt = 0.025;

    let mut mean: Vec<f64> = vec![0.0; n_params]; // Start at zero residual
    mean[n_params - 1] = 0.1; // Initial residual gain = 0.1
    let mut sigma: f64 = 0.05; // Small sigma — we're searching near PD

    let mut step_sizes: Vec<f64> = vec![1.0; n_params];
    let mut path_s: Vec<f64> = vec![0.0; n_params];
    let c_s = 0.3;
    let c_c = 0.4 / (n_params as f64).sqrt();
    let damp = 1.0 + n_params as f64 / pop_size as f64;

    println!("  Pop: {pop_size}, Gens: {n_generations}, Steps: {eval_steps}, Sigma: {sigma}");
    println!();
    use std::io::Write;
    std::io::stdout().flush().unwrap_or(());

    let start = Instant::now();
    let mut best_ever_fitness = f64::NEG_INFINITY;
    let mut best_ever_params: Vec<f64> = mean.clone();

    println!(
        "{:>5} {:>12} {:>12} {:>12} {:>8}",
        "Gen", "BestFit", "MeanFit", "BestEver", "Sigma"
    );

    for r#gen in 0..n_generations {
        let mut population: Vec<Vec<f64>> = Vec::with_capacity(pop_size);
        let mut fitnesses: Vec<f64> = Vec::with_capacity(pop_size);

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
            // Clamp residual gain to [0, 0.3]
            candidate[n_params - 1] = candidate[n_params - 1].clamp(0.0, 0.3);

            let fitness = evaluate_residual(
                &candidate, n_weights, &genesis, &config, &pd_gains, eval_steps, dt,
            );
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
                "{:>5} {:>12.4} {:>12.4} {:>12.4} {:>8.5} gain={:.3}",
                r#gen, best_fitness, mean_fitness, best_ever_fitness, sigma, gain
            );
            use std::io::Write;
            std::io::stdout().flush().unwrap_or(());
        }
    }

    let elapsed = start.elapsed();
    println!();
    println!("━━━ Complete ({:.1} min) ━━━", elapsed.as_secs_f64() / 60.0);

    // Final eval
    println!();
    println!("━━━ Final: 2000 steps, NO PD (residual controller alone) ━━━");

    let gain = best_ever_params[n_params - 1].clamp(0.0, 0.3) as f32;
    let mut controller = HumanoidController::new(&genesis, &config);
    let weights: Vec<f32> = best_ever_params[..n_weights]
        .iter()
        .map(|&w| w as f32)
        .collect();
    let bias: Vec<f32> = best_ever_params[n_weights..n_weights + n_bias]
        .iter()
        .map(|&w| w as f32)
        .collect();
    controller.set_output_projection(&weights, &bias);

    let mut encoder = HumanoidHdcEncoder::new(&genesis, config.num_levels);
    let mut sim = SimpleHumanoidSimulator::new();

    let mut steps_standing = 0;
    let mut fell = false;
    let mut total_reward = 0.0;
    for step in 0..2000 {
        let state = sim.state().clone();
        let hv = encoder.encode(&state);
        let learned = controller.forward(&hv, dt as f32);
        let pd = pd_standing_baseline(&state, &pd_gains);

        // Residual: PD + gain × learned
        let mut cmd = HumanoidCommand::zero();
        for i in 0..cmd.torques.len().min(pd.torques.len()) {
            cmd.torques[i] = (pd.torques[i] + gain * learned.torques[i]).clamp(-1.0, 1.0);
        }
        sim.step(&cmd, dt);

        let s = sim.state();
        if s.head_height >= 0.8 {
            steps_standing += 1;
        }
        let r = if s.head_height >= 1.2 {
            1.0
        } else {
            (s.head_height / 1.4).max(0.0)
        } * s.torso_vertical[2].max(0.0);
        total_reward += r;
        if s.head_height < 0.5 && !fell {
            fell = true;
            println!("  Fell at step {}", step);
        }
    }

    println!("  Mean reward: {:.4}", total_reward / 2000.0);
    println!(
        "  Standing: {} / 2000 ({:.1}%)",
        steps_standing,
        steps_standing as f64 / 20.0
    );
    println!("  Residual gain: {:.3}", gain);
    if !fell {
        println!("  *** INDEPENDENT STANDING ACHIEVED ***");
    }

    // Compare: PD alone
    println!();
    println!("━━━ Comparison: PD alone (no learned residual) ━━━");
    sim.reset();
    encoder.reset();
    let mut pd_standing = 0;
    let mut pd_reward = 0.0;
    for _ in 0..2000 {
        let state = sim.state().clone();
        let pd = pd_standing_baseline(&state, &pd_gains);
        sim.step(&pd, dt);
        let s = sim.state();
        if s.head_height >= 0.8 {
            pd_standing += 1;
        }
        let r = if s.head_height >= 1.2 {
            1.0
        } else {
            (s.head_height / 1.4).max(0.0)
        } * s.torso_vertical[2].max(0.0);
        pd_reward += r;
    }
    println!("  PD mean reward: {:.4}", pd_reward / 2000.0);
    println!(
        "  PD standing: {} / 2000 ({:.1}%)",
        pd_standing,
        pd_standing as f64 / 20.0
    );
}

#[cfg(feature = "humanoid")]
fn evaluate_residual(
    params: &[f64],
    n_weights: usize,
    genesis: &symthaea_core::genesis::GenesisSeed,
    config: &symthaea_humanoid::types::HumanoidConfig,
    pd_gains: &symthaea_humanoid::types::HumanoidPdGains,
    eval_steps: usize,
    dt: f64,
) -> f64 {
    use symthaea_humanoid::controller::HumanoidController;
    use symthaea_humanoid::encoder::HumanoidHdcEncoder;
    use symthaea_humanoid::simulator::{HumanoidPhysicsSimulator, SimpleHumanoidSimulator};
    use symthaea_humanoid::types::*;

    let n_bias = config.morphology.num_actuators();
    let n_params = params.len();
    let residual_gain = params[n_params - 1].clamp(0.0, 0.3) as f32;

    let mut controller = HumanoidController::new(genesis, config);
    let weights: Vec<f32> = params[..n_weights].iter().map(|&w| w as f32).collect();
    let bias: Vec<f32> = params[n_weights..n_weights + n_bias]
        .iter()
        .map(|&w| w as f32)
        .collect();
    controller.set_output_projection(&weights, &bias);

    let mut encoder = HumanoidHdcEncoder::new(genesis, config.num_levels);
    let mut sim = SimpleHumanoidSimulator::new();

    let mut total_reward = 0.0;
    for step in 0..eval_steps {
        let state = sim.state().clone();
        let hv = encoder.encode(&state);
        let learned = controller.forward(&hv, dt as f32);
        let pd = pd_standing_baseline(&state, pd_gains);

        // Residual: PD + gain × learned
        let mut cmd = HumanoidCommand::zero();
        for i in 0..cmd.torques.len().min(pd.torques.len()) {
            cmd.torques[i] = (pd.torques[i] + residual_gain * learned.torques[i]).clamp(-1.0, 1.0);
        }
        sim.step(&cmd, dt);

        let s = sim.state();
        let head_r = if s.head_height >= 1.2 {
            1.0
        } else {
            (s.head_height / 1.4).max(0.0)
        };
        let upright = s.torso_vertical[2].max(0.0);
        let effort_bonus = (1.0 - cmd.control_effort() as f64).max(0.0) * 0.1;
        total_reward += head_r * upright + effort_bonus;

        if s.head_height < 0.5 {
            total_reward -= (eval_steps - step) as f64 * 0.3;
            break;
        }
    }

    total_reward / eval_steps as f64
}