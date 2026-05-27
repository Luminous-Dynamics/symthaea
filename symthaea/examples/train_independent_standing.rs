// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Train Independent Standing
//!
//! Focused training run: 200 episodes, Stand only, curriculum that
//! spends the majority of time at PD=0 to force independent standing.
//!
//! ```bash
//! cargo run --example train_independent_standing --release --features humanoid
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
    use symthaea_humanoid::fep_agent::{ActiveInferenceHumanoidAgent, HumanoidFepConfig};
    use symthaea_humanoid::simulator::{HumanoidPhysicsSimulator, SimpleHumanoidSimulator};
    use symthaea_humanoid::training::HumanoidTrainer;
    use symthaea_humanoid::types::*;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Training Independent Standing                              ║");
    println!("║  200 episodes, aggressive PD decay, Stand only              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Aggressive curriculum: PD decays to 0 by episode 60 (30%)
    // Remaining 140 episodes (70%) are at PD=0 — pure learned control
    let config = HumanoidConfig {
        num_episodes: 200,
        steps_per_episode: 1000,
        task: HumanoidTask::Stand,
        adaptive_curriculum: false, // Fixed schedule, controlled decay
        domain_randomization: false,
        progressive_noise: false,
        learning_rate: 0.001, // Higher LR for faster learning
        genesis_phrase: "independent-standing-v1".to_string(),
        ..HumanoidConfig::default()
    };

    println!(
        "  Config: {} episodes, LR={}, Stand only",
        config.num_episodes, config.learning_rate
    );
    println!("  Curriculum: PD 0.8→0 over first 60 episodes, then 140 at PD=0");
    println!();

    let start = Instant::now();
    let mut trainer = HumanoidTrainer::new(config.clone());
    // DAgger: every 10 episodes, run 200 steps without PD, train on OOD states
    let (metrics, controller, encoder) = trainer.train_dagger(10, 200);
    let train_time = start.elapsed();

    println!();
    println!(
        "━━━ Training Complete ({:.1} min) ━━━",
        train_time.as_secs_f64() / 60.0
    );

    // Report learning curve
    println!();
    println!(
        "{:>5} {:>12} {:>10} {:>10} {:>8}",
        "Ep", "StandReward", "HeadHt", "Upright", "Effort"
    );
    for (i, m) in metrics.iter().enumerate() {
        if i % 20 == 0 || i == metrics.len() - 1 {
            println!(
                "{:>5} {:>12.4} {:>10.4} {:>10.4} {:>8.4}",
                i,
                m.avg_standing_reward,
                m.avg_head_height,
                m.avg_uprightness,
                m.avg_control_effort
            );
        }
    }

    // Evaluate: run the TRAINED controller ALONE (no PD) for 1000 steps
    println!();
    println!("━━━ Evaluation: Controller ALONE (no PD baseline) ━━━");

    let mut sim = SimpleHumanoidSimulator::new();
    let mut eval_encoder = encoder;
    let mut eval_controller = controller;
    eval_encoder.reset();

    let dt = 0.025;
    let eval_steps = 2000;
    let mut total_reward = 0.0;
    let mut min_height = 2.0f64;
    let mut fell = false;
    let mut steps_standing = 0;

    for step in 0..eval_steps {
        let state = sim.state().clone();
        let hv = eval_encoder.encode(&state);
        let cmd = eval_controller.forward(&hv, dt as f32);
        sim.step(&cmd, dt);

        let s = sim.state();
        let reward = if s.head_height >= 1.2 {
            1.0
        } else {
            (s.head_height / 1.2).max(0.0)
        } * s.torso_vertical[2].max(0.0);
        total_reward += reward;
        if s.head_height < min_height {
            min_height = s.head_height;
        }
        if s.head_height >= 0.8 {
            steps_standing += 1;
        }
        if s.head_height < 0.5 && !fell {
            println!(
                "  *** FELL at step {} (head={:.3}) ***",
                step, s.head_height
            );
            fell = true;
        }
    }

    let mean_reward = total_reward / eval_steps as f64;
    let standing_pct = steps_standing as f64 / eval_steps as f64 * 100.0;

    println!();
    println!("  Results ({} steps, no PD):", eval_steps);
    println!("    Mean standing reward: {:.4}", mean_reward);
    println!("    Min head height:      {:.4} m", min_height);
    println!(
        "    Steps standing:       {} / {} ({:.1}%)",
        steps_standing, eval_steps, standing_pct
    );
    println!(
        "    Fell:                 {}",
        if fell { "YES" } else { "NO" }
    );

    // Also evaluate WITH PD blend for comparison
    println!();
    println!("━━━ Evaluation: Controller + PD Blend (50/50) ━━━");

    sim.reset();
    eval_encoder.reset();

    let pd_gains = HumanoidPdGains::default();
    let mut total_reward_pd = 0.0;
    let mut fell_pd = false;
    let mut steps_standing_pd = 0;

    for step in 0..eval_steps {
        let state = sim.state().clone();
        let hv = eval_encoder.encode(&state);
        let learned = eval_controller.forward(&hv, dt as f32);
        let pd = pd_standing_baseline(&state, &pd_gains);

        let mut cmd = HumanoidCommand::zero();
        for i in 0..cmd.torques.len().min(learned.torques.len()) {
            cmd.torques[i] = 0.5 * pd.torques[i] + 0.5 * learned.torques[i];
            cmd.torques[i] = cmd.torques[i].clamp(-1.0, 1.0);
        }
        sim.step(&cmd, dt);

        let s = sim.state();
        let reward = if s.head_height >= 1.2 {
            1.0
        } else {
            (s.head_height / 1.2).max(0.0)
        } * s.torso_vertical[2].max(0.0);
        total_reward_pd += reward;
        if s.head_height >= 0.8 {
            steps_standing_pd += 1;
        }
        if s.head_height < 0.5 && !fell_pd {
            fell_pd = true;
        }
    }

    let mean_reward_pd = total_reward_pd / eval_steps as f64;
    let standing_pct_pd = steps_standing_pd as f64 / eval_steps as f64 * 100.0;

    println!("  Results ({} steps, 50% PD blend):", eval_steps);
    println!("    Mean standing reward: {:.4}", mean_reward_pd);
    println!(
        "    Steps standing:       {} / {} ({:.1}%)",
        steps_standing_pd, eval_steps, standing_pct_pd
    );
    println!(
        "    Fell:                 {}",
        if fell_pd { "YES" } else { "NO" }
    );

    println!();
    println!("━━━ Summary ━━━");
    println!(
        "  Controller alone: {:.4} reward, {:.1}% standing, fell={}",
        mean_reward, standing_pct, fell
    );
    println!(
        "  With PD blend:    {:.4} reward, {:.1}% standing, fell={}",
        mean_reward_pd, standing_pct_pd, fell_pd
    );
}