// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Symthaea Performance Test Suite
//!
//! Measures Symthaea's limits: HDC operation speed, encoding throughput,
//! training step latency, cognitive cycle rate, and scale characteristics.
//!
//! ```bash
//! cargo run --example performance_test_suite --release
//! # With humanoid training benchmark:
//! cargo run --example performance_test_suite --release --features humanoid
//! ```

use std::time::Instant;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Symthaea Performance Test Suite                            ║");
    println!("║  Testing the limits of the Holographic Liquid Brain         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let genesis = GenesisSeed::from_phrase("performance-test-suite");

    // ─── HDC Microbenchmarks ───
    println!("━━━ HDC Microbenchmarks (D={HDC_DIMENSION}) ━━━");

    bench_hdc_random(&genesis);
    bench_hdc_bind(&genesis);
    bench_hdc_similarity(&genesis);
    bench_hdc_bundle(&genesis);
    bench_hdc_weighted_bundle(&genesis);

    // ─── Encoding Benchmarks ───
    println!("\n━━━ Encoding Benchmarks ━━━");

    #[cfg(feature = "humanoid")]
    bench_encoding();

    #[cfg(not(feature = "humanoid"))]
    println!("  (skip: requires --features humanoid)");

    // ─── Training Benchmarks ───
    println!("\n━━━ Training Benchmarks ━━━");

    #[cfg(feature = "humanoid")]
    bench_training();

    #[cfg(not(feature = "humanoid"))]
    println!("  (skip: requires --features humanoid)");

    println!("\n━━━ Done ━━━");
}

fn bench_hdc_random(genesis: &GenesisSeed) {
    let n = 10_000;
    let start = Instant::now();
    for i in 0..n {
        let _ = genesis.hv(&format!("bench_{i}"), HDC_DIMENSION);
    }
    let elapsed = start.elapsed();
    let per_op = elapsed / n;
    println!(
        "  random({}D)     {:>8.0}ns/op  ({} ops in {:.2}ms)",
        HDC_DIMENSION,
        per_op.as_nanos(),
        n,
        elapsed.as_secs_f64() * 1000.0
    );
}

fn bench_hdc_bind(genesis: &GenesisSeed) {
    let a = genesis.hv("bind_a", HDC_DIMENSION);
    let b = genesis.hv("bind_b", HDC_DIMENSION);
    let n = 100_000;
    let start = Instant::now();
    for _ in 0..n {
        let _ = a.bind(&b);
    }
    let elapsed = start.elapsed();
    let per_op = elapsed / n;
    println!(
        "  bind({}D)       {:>8.0}ns/op  ({} ops in {:.2}ms)",
        HDC_DIMENSION,
        per_op.as_nanos(),
        n,
        elapsed.as_secs_f64() * 1000.0
    );
}

fn bench_hdc_similarity(genesis: &GenesisSeed) {
    let a = genesis.hv("sim_a", HDC_DIMENSION);
    let b = genesis.hv("sim_b", HDC_DIMENSION);
    let n = 100_000;
    let start = Instant::now();
    let mut sum = 0.0f32;
    for _ in 0..n {
        sum += a.similarity(&b);
    }
    let elapsed = start.elapsed();
    let per_op = elapsed / n;
    println!(
        "  similarity({}D) {:>8.0}ns/op  ({} ops in {:.2}ms) [sum={:.2}]",
        HDC_DIMENSION,
        per_op.as_nanos(),
        n,
        elapsed.as_secs_f64() * 1000.0,
        sum / n as f32
    );
}

fn bench_hdc_bundle(genesis: &GenesisSeed) {
    let hvs: Vec<ContinuousHV> = (0..5)
        .map(|i| genesis.hv(&format!("bundle_{i}"), HDC_DIMENSION))
        .collect();
    let refs: Vec<&ContinuousHV> = hvs.iter().collect();
    let n = 10_000;
    let start = Instant::now();
    for _ in 0..n {
        let _ = ContinuousHV::bundle(&refs);
    }
    let elapsed = start.elapsed();
    let per_op = elapsed / n;
    println!(
        "  bundle(5×{}D)  {:>8.0}ns/op  ({} ops in {:.2}ms)",
        HDC_DIMENSION,
        per_op.as_nanos(),
        n,
        elapsed.as_secs_f64() * 1000.0
    );
}

fn bench_hdc_weighted_bundle(genesis: &GenesisSeed) {
    let hvs: Vec<ContinuousHV> = (0..10)
        .map(|i| genesis.hv(&format!("wbundle_{i}"), HDC_DIMENSION))
        .collect();
    let refs: Vec<&ContinuousHV> = hvs.iter().collect();
    let weights: Vec<f32> = vec![1.0, 1.5, 2.0, 0.8, 1.0, 1.5, 2.0, 0.5, 1.0, 1.2];
    let n = 10_000;
    let start = Instant::now();
    for _ in 0..n {
        let _ = ContinuousHV::weighted_bundle(&refs, &weights);
    }
    let elapsed = start.elapsed();
    let per_op = elapsed / n;
    println!(
        "  weighted_bundle(10×{}D) {:>8.0}ns/op  ({} ops in {:.2}ms)",
        HDC_DIMENSION,
        per_op.as_nanos(),
        n,
        elapsed.as_secs_f64() * 1000.0
    );
}

#[cfg(feature = "humanoid")]
fn bench_encoding() {
    use symthaea_humanoid::encoder::HumanoidHdcEncoder;
    use symthaea_humanoid::morphology::HumanoidMorphology;
    use symthaea_humanoid::types::HumanoidState;

    let genesis = GenesisSeed::from_phrase("perf-test-encoder");

    // DMC21 encoding (72 channels)
    {
        let mut enc = HumanoidHdcEncoder::new(&genesis, 32);
        let state = HumanoidState::standing();
        // Warm up
        for _ in 0..10 {
            enc.encode(&state);
        }
        enc.reset();

        let n = 1_000;
        let start = Instant::now();
        for _ in 0..n {
            enc.encode(&state);
        }
        let elapsed = start.elapsed();
        let per_op = elapsed / n;
        println!(
            "  encode(DMC21, 72ch)       {:>8.0}us/state  ({} states in {:.2}ms)",
            per_op.as_micros(),
            n,
            elapsed.as_secs_f64() * 1000.0
        );
    }

    // Dexterous53 encoding (142 channels)
    {
        let mut enc = HumanoidHdcEncoder::new_for(&genesis, 32, HumanoidMorphology::Dexterous53);
        let state = HumanoidState::standing_for(HumanoidMorphology::Dexterous53);
        for _ in 0..10 {
            enc.encode(&state);
        }
        enc.reset();

        let n = 1_000;
        let start = Instant::now();
        for _ in 0..n {
            enc.encode(&state);
        }
        let elapsed = start.elapsed();
        let per_op = elapsed / n;
        println!(
            "  encode(Dex53, 142ch)      {:>8.0}us/state  ({} states in {:.2}ms)",
            per_op.as_micros(),
            n,
            elapsed.as_secs_f64() * 1000.0
        );
    }

    // Predictive encoding (72ch + LTC layer)
    {
        let mut enc = HumanoidHdcEncoder::new_predictive(&genesis, 32, HumanoidMorphology::Dmc21);
        let state = HumanoidState::standing();
        for _ in 0..10 {
            enc.encode(&state);
        }
        enc.reset();

        let n = 1_000;
        let start = Instant::now();
        for _ in 0..n {
            enc.encode(&state);
        }
        let elapsed = start.elapsed();
        let per_op = elapsed / n;
        println!(
            "  encode(predictive, 72ch)  {:>8.0}us/state  ({} states in {:.2}ms)",
            per_op.as_micros(),
            n,
            elapsed.as_secs_f64() * 1000.0
        );
        println!(
            "    PE={:.4}  confidence={:.4}",
            enc.prediction_error(),
            enc.confidence()
        );
    }
}

#[cfg(feature = "humanoid")]
fn bench_training() {
    use symthaea_humanoid::training::HumanoidTrainer;
    use symthaea_humanoid::types::{HumanoidConfig, HumanoidTask};

    // Quick training benchmark: 5 episodes Stand
    let config = HumanoidConfig {
        num_episodes: 5,
        steps_per_episode: 1000,
        task: HumanoidTask::Stand,
        adaptive_curriculum: false,
        collect_telemetry: false,
        domain_randomization: false,
        ..HumanoidConfig::default()
    };

    println!("  Training 5 episodes × 1000 steps (Stand, no DR)...");
    let mut trainer = HumanoidTrainer::new(config);
    let start = Instant::now();
    let metrics = trainer.train();
    let elapsed = start.elapsed();

    let total_steps = 5 * 1000;
    let steps_per_sec = total_steps as f64 / elapsed.as_secs_f64();
    let ms_per_step = elapsed.as_secs_f64() * 1000.0 / total_steps as f64;
    let episodes_per_hour = 5.0 / elapsed.as_secs_f64() * 3600.0;

    println!("    Wall time:       {:.2}s", elapsed.as_secs_f64());
    println!("    Steps/sec:       {:.0}", steps_per_sec);
    println!("    ms/step:         {:.3}", ms_per_step);
    println!("    Episodes/hour:   {:.0}", episodes_per_hour);

    if let Some(last) = metrics.last() {
        println!("    Final reward:    {:.4}", last.avg_episode_reward);
        println!("    Final standing:  {:.4}", last.avg_standing_reward);
        println!("    Final upright:   {:.4}", last.avg_uprightness);
    }

    // Extrapolate
    let est_200_ep = elapsed.as_secs_f64() * (200.0 / 5.0);
    println!(
        "    Est. 200 episodes: {:.1}s ({:.1} min)",
        est_200_ep,
        est_200_ep / 60.0
    );
}