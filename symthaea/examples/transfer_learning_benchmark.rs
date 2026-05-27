// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Transfer Learning Benchmark: Flight → Humanoid
//!
//! Tests whether flight stabilization dynamics transfer to bipedal standing
//! via HDC morphological projection (11 shared channels).
//!
//! Runs N seeds for both conditions (transfer-initialized vs random),
//! tracks learning curves, and measures episodes to standing mastery.
//!
//! ```bash
//! cargo run --example transfer_learning_benchmark --release --features humanoid
//! ```

fn main() {
    #[cfg(not(feature = "humanoid"))]
    {
        eprintln!("ERROR: Requires --features humanoid");
        std::process::exit(1);
    }

    #[cfg(feature = "humanoid")]
    run_benchmark();
}

#[cfg(feature = "humanoid")]
fn run_benchmark() {
    use symthaea_core::genesis::GenesisSeed;
    use symthaea_core::hdc::{
        ContinuousHV, HDC_DIMENSION, HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig,
    };
    use symthaea_humanoid::training::{EpisodeMetrics, HumanoidTrainer};
    use symthaea_humanoid::transfer::MorphologicalTransfer;
    use symthaea_humanoid::types::{HumanoidConfig, HumanoidTask};

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Transfer Learning Benchmark: Flight → Humanoid             ║");
    println!("║  HDC morphological projection via 11 shared channels        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let n_seeds = 3; // Use 3 for speed; increase to 5 for publication
    let n_episodes = 30; // Short training for benchmark
    let mastery_threshold = 0.5; // Standing reward threshold for "mastery"
    let mastery_streak = 3; // Consecutive episodes above threshold

    println!(
        "  Seeds: {n_seeds}, Episodes: {n_episodes}, Mastery: >{mastery_threshold} for {mastery_streak} consecutive"
    );
    println!();

    let mut transfer_curves: Vec<Vec<f64>> = Vec::new();
    let mut random_curves: Vec<Vec<f64>> = Vec::new();
    let mut transfer_mastery: Vec<Option<usize>> = Vec::new();
    let mut random_mastery: Vec<Option<usize>> = Vec::new();

    for seed in 0..n_seeds {
        println!("━━━ Seed {seed} ━━━");

        // Create a "trained" flight network (evolved with 50 random inputs)
        let flight_net = make_flight_network(seed as u64);

        // Transfer condition
        let config = HumanoidConfig {
            num_episodes: n_episodes,
            steps_per_episode: 500,
            task: HumanoidTask::Stand,
            adaptive_curriculum: false,
            domain_randomization: false,
            genesis_phrase: format!("transfer-bench-seed-{seed}"),
            ..HumanoidConfig::default()
        };

        let transfer = MorphologicalTransfer::flight_to_humanoid();
        let transfer_ctrl = transfer.initialize_humanoid(&flight_net, &config);
        let mut transfer_trainer = HumanoidTrainer::with_controller(config.clone(), transfer_ctrl);

        let t_start = std::time::Instant::now();
        let t_metrics = transfer_trainer.train();
        let t_time = t_start.elapsed();

        let t_rewards: Vec<f64> = t_metrics.iter().map(|m| m.avg_standing_reward).collect();
        let t_mastery = find_mastery(&t_rewards, mastery_threshold, mastery_streak);

        println!(
            "  Transfer: {:.2}s, final={:.4}, mastery={:?}",
            t_time.as_secs_f64(),
            t_rewards.last().unwrap_or(&0.0),
            t_mastery
        );

        // Random condition
        let r_config = HumanoidConfig {
            genesis_phrase: format!("random-bench-seed-{seed}"),
            ..config.clone()
        };
        let mut random_trainer = HumanoidTrainer::new(r_config);

        let r_start = std::time::Instant::now();
        let r_metrics = random_trainer.train();
        let r_time = r_start.elapsed();

        let r_rewards: Vec<f64> = r_metrics.iter().map(|m| m.avg_standing_reward).collect();
        let r_mastery = find_mastery(&r_rewards, mastery_threshold, mastery_streak);

        println!(
            "  Random:   {:.2}s, final={:.4}, mastery={:?}",
            r_time.as_secs_f64(),
            r_rewards.last().unwrap_or(&0.0),
            r_mastery
        );

        transfer_curves.push(t_rewards);
        random_curves.push(r_rewards);
        transfer_mastery.push(t_mastery);
        random_mastery.push(r_mastery);
        println!();
    }

    // ─── Summary ───
    println!("━━━ Summary ━━━");

    let t_final_mean: f64 = transfer_curves
        .iter()
        .map(|c| c.last().unwrap_or(&0.0))
        .sum::<f64>()
        / n_seeds as f64;
    let r_final_mean: f64 = random_curves
        .iter()
        .map(|c| c.last().unwrap_or(&0.0))
        .sum::<f64>()
        / n_seeds as f64;

    let t_mastery_mean = mastery_mean(&transfer_mastery);
    let r_mastery_mean = mastery_mean(&random_mastery);

    println!(
        "  Transfer: final_reward={:.4}, mastery_ep={}",
        t_final_mean,
        t_mastery_mean
            .map(|m| format!("{:.1}", m))
            .unwrap_or("never".to_string())
    );
    println!(
        "  Random:   final_reward={:.4}, mastery_ep={}",
        r_final_mean,
        r_mastery_mean
            .map(|m| format!("{:.1}", m))
            .unwrap_or("never".to_string())
    );

    if t_final_mean > r_final_mean {
        let adv = (t_final_mean - r_final_mean) / r_final_mean.max(0.001) * 100.0;
        println!("  Transfer advantage: +{:.1}% final reward", adv);
    } else {
        let adv = (r_final_mean - t_final_mean) / t_final_mean.max(0.001) * 100.0;
        println!("  Random advantage: +{:.1}% final reward", adv);
    }

    // ─── CSV ───
    println!();
    println!("━━━ CSV Output ━━━");
    println!("seed,condition,episode,standing_reward");
    for (seed, curve) in transfer_curves.iter().enumerate() {
        for (ep, reward) in curve.iter().enumerate() {
            println!("{seed},transfer,{ep},{reward:.6}");
        }
    }
    for (seed, curve) in random_curves.iter().enumerate() {
        for (ep, reward) in curve.iter().enumerate() {
            println!("{seed},random,{ep},{reward:.6}");
        }
    }
}

#[cfg(feature = "humanoid")]
fn make_flight_network(seed: u64) -> symthaea_core::hdc::HdcLtcUnifiedNetwork {
    use symthaea_core::genesis::GenesisSeed;
    use symthaea_core::hdc::*;

    let genesis = GenesisSeed::from_phrase(&format!("flight-bench-seed-{seed}"));
    let config = UnifiedNetworkConfig {
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

    let mut network = HdcLtcUnifiedNetwork::from_genesis(config, &genesis);

    // Simulate flight training: evolve with 50 pseudo-random inputs
    for i in 0..50 {
        let input = ContinuousHV::random(HDC_DIMENSION, seed * 1000 + i);
        network.evolve_closed_form(0.01, &input);
    }

    network
}

#[cfg(feature = "humanoid")]
fn find_mastery(rewards: &[f64], threshold: f64, streak_required: usize) -> Option<usize> {
    let mut streak = 0;
    for (i, &r) in rewards.iter().enumerate() {
        if r >= threshold {
            streak += 1;
            if streak >= streak_required {
                return Some(i + 1 - streak_required);
            }
        } else {
            streak = 0;
        }
    }
    None
}

#[cfg(feature = "humanoid")]
fn mastery_mean(masteries: &[Option<usize>]) -> Option<f64> {
    let valid: Vec<f64> = masteries
        .iter()
        .filter_map(|m| m.map(|v| v as f64))
        .collect();
    if valid.is_empty() {
        None
    } else {
        Some(valid.iter().sum::<f64>() / valid.len() as f64)
    }
}