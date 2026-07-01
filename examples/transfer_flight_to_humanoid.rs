// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Morphological transfer benchmark: flight → humanoid.
//!
//! Trains a flight controller on hover, then transfers learned temporal
//! dynamics to a humanoid controller. Compares transfer-initialized vs
//! random-initialized humanoid training.
//!
//! Run: `cargo run --features multirotor,humanoid --example transfer_flight_to_humanoid --release`

use symthaea::humanoid::HumanoidConfig;
use symthaea::humanoid::transfer::{MorphologicalTransfer, transfer_learning_comparison};
use symthaea::multirotor::{
    FlightConfig, FlightController, FlightState, QuadrotorCommand, QuadrotorHdcEncoder,
};
use symthaea::symthaea_core::genesis::GenesisSeed;
use symthaea::symthaea_core::hdc::ContinuousHV;

fn main() {
    println!("=== Morphological Transfer: Flight → Humanoid ===");
    println!();

    // ── Phase 1: Train a flight controller on hover ──
    println!("Phase 1: Training flight controller on hover...");
    let flight_genesis = GenesisSeed::from_phrase("symthaea-multirotor-quadrotor");
    let flight_config = FlightConfig::default();
    let mut flight_ctrl = FlightController::new(&flight_genesis, &flight_config);
    let mut flight_encoder = QuadrotorHdcEncoder::new(&flight_genesis, flight_config.num_levels);

    // Simulate hover training: feed encoder → evolve → train for 500 steps
    // (longer training produces more stable temporal dynamics for transfer)
    let hover_state = FlightState::hover(1.0);
    for step in 0..500 {
        let sensor_hv = flight_encoder.encode(&hover_state);
        let cmd = flight_ctrl.forward(&sensor_hv, 0.002);

        // Train toward hover target
        let target = QuadrotorCommand::hover();
        flight_ctrl.train_step(&sensor_hv, &target, 0.002, None);

        if step % 100 == 0 {
            println!(
                "  Step {step:>3}: thrust = {:.4}, moments = ({:.5}, {:.5}, {:.5})",
                cmd.thrust, cmd.roll_moment, cmd.pitch_moment, cmd.yaw_moment
            );
        }
    }
    println!(
        "  Flight training complete. Network stats: {:?}",
        flight_ctrl.stats()
    );
    println!();

    // ── Phase 2: Extract transfer concepts ──
    println!("Phase 2: Extracting morphological transfer concepts...");
    let transfer = MorphologicalTransfer::flight_to_humanoid();
    println!(
        "  Shared channels: {} (altitude, orientation, velocity)",
        transfer.num_shared_channels()
    );
    println!("  Blend factor: {}", transfer.blend_factor());

    let concepts = transfer.extract_concepts(flight_ctrl.network());
    println!(
        "  Extracted {} concept HVs from flight network",
        concepts.len()
    );

    // Check concept norms
    let avg_norm: f32 =
        concepts.iter().map(|(_, _, c)| c.norm()).sum::<f32>() / concepts.len() as f32;
    println!("  Average concept norm: {avg_norm:.4}");
    println!();

    // ── Phase 3: Transfer comparison benchmark ──
    println!("Phase 3: Transfer vs Random training comparison...");
    println!("  Training 10 episodes each (300 steps/episode)...");
    println!();

    let (transfer_metrics, random_metrics) =
        transfer_learning_comparison(flight_ctrl.network(), 10);

    println!(
        "{:<8} {:>12} {:>12} {:>12} {:>10}",
        "Episode", "Transfer R", "Random R", "Delta", "Head H (T)"
    );
    println!("{}", "-".repeat(58));

    for (tm, rm) in transfer_metrics.iter().zip(random_metrics.iter()) {
        let delta = tm.avg_standing_reward - rm.avg_standing_reward;
        let sign = if delta >= 0.0 { "+" } else { "" };
        println!(
            "{:<8} {:>12.4} {:>12.4} {:>11}{:.4} {:>10.3}",
            tm.episode,
            tm.avg_standing_reward,
            rm.avg_standing_reward,
            sign,
            delta,
            tm.avg_head_height,
        );
    }

    println!();

    // ── Summary ──
    let transfer_final = transfer_metrics
        .last()
        .map(|m| m.avg_standing_reward)
        .unwrap_or(0.0);
    let random_final = random_metrics
        .last()
        .map(|m| m.avg_standing_reward)
        .unwrap_or(0.0);
    let transfer_avg: f64 = transfer_metrics
        .iter()
        .map(|m| m.avg_standing_reward)
        .sum::<f64>()
        / transfer_metrics.len() as f64;
    let random_avg: f64 = random_metrics
        .iter()
        .map(|m| m.avg_standing_reward)
        .sum::<f64>()
        / random_metrics.len() as f64;

    println!("Summary:");
    println!("  Transfer avg standing reward: {transfer_avg:.4}");
    println!("  Random avg standing reward:   {random_avg:.4}");
    println!("  Transfer final reward:        {transfer_final:.4}");
    println!("  Random final reward:          {random_final:.4}");

    if transfer_avg > random_avg {
        let pct = ((transfer_avg - random_avg) / random_avg.max(1e-6)) * 100.0;
        println!("  Transfer advantage:           +{pct:.1}% (mean standing reward)");
    } else {
        println!("  No transfer advantage observed (may need more training epochs)");
    }

    // Check network divergence
    let humanoid_config = HumanoidConfig::default();
    let mut transfer_ctrl = transfer.initialize_humanoid(flight_ctrl.network(), &humanoid_config);
    let genesis = GenesisSeed::from_phrase("symthaea-humanoid-dmc");
    let mut random_ctrl = symthaea::humanoid::HumanoidController::new(&genesis, &humanoid_config);

    let test_input = ContinuousHV::random(16384, 42);
    for _ in 0..5 {
        transfer_ctrl.forward(&test_input, 0.025);
        random_ctrl.forward(&test_input, 0.025);
    }
    let sim = transfer_ctrl
        .network()
        .output()
        .similarity(&random_ctrl.network().output());
    println!("  Network state similarity:     {sim:.4} (1.0 = identical, <0.99 = diverged)");
}
