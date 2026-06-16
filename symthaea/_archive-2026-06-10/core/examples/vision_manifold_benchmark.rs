// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Vision Manifold Benchmark
//!
//! Evaluates the vision manifold pipeline on synthetic but realistic video sequences:
//!
//! 1. **Static scene convergence** — How fast does prediction error drop?
//! 2. **Scene change detection** — Surprise spike amplitude + recovery time.
//! 3. **Alternating pattern learning** — Can the CfC learn A→B→A→B?
//! 4. **Multi-scale discrimination** — Do fine/coarse encoders capture different info?
//! 5. **Temporal prediction accuracy** — Multi-horizon error profiles.
//! 6. **Throughput** — Frames per second for encode + evolve.
//!
//! Usage: `cargo run --example vision_manifold_benchmark --features vision-manifold`

#[cfg(feature = "vision-manifold")]
fn main() {
    use std::time::Instant;
    use symthaea::perception::vision_manifold::{
        HorizonAccuracy, MultiScaleEncoder, VisionConfig, VisionManifold,
    };

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Symthaea v0.5.0 — Vision Manifold Benchmark");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let width = 128u32;
    let height = 128u32;
    let dt = 0.033f32; // 30fps

    // ── Synthetic video generators ───────────────────────────────────

    let solid_frame = |value: u8| -> Vec<u8> { vec![value; (width * height) as usize] };

    let gradient_frame = |seed: u8| -> Vec<u8> {
        (0..width * height)
            .map(|i| {
                let x = (i % width) as u8;
                let y = (i / width) as u8;
                x.wrapping_add(y).wrapping_add(seed)
            })
            .collect()
    };

    let checkerboard_frame = |block_size: u32, seed: u8| -> Vec<u8> {
        (0..width * height)
            .map(|i| {
                let x = i % width;
                let y = i / width;
                let checker = ((x / block_size) + (y / block_size)) % 2;
                if checker == 0 { seed } else { 255 - seed }
            })
            .collect()
    };

    let noise_frame = |seed: u64| -> Vec<u8> {
        let mut val = seed;
        (0..width * height)
            .map(|_| {
                // Simple LCG for reproducible noise
                val = val
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((val >> 33) & 0xFF) as u8
            })
            .collect()
    };

    // ═══════════════════════════════════════════════════════════════════
    // Benchmark 1: Static Scene Convergence
    // ═══════════════════════════════════════════════════════════════════
    println!("━━━ Benchmark 1: Static Scene Convergence ━━━━━━━━━━━━━━━━━━━━━━━");
    {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, width, height);
        let frame = gradient_frame(42);

        let mut errors = Vec::new();
        for _ in 0..100 {
            let tel = m.observe_frame(&frame, width, height, 1, dt);
            errors.push(tel.prediction_error);
        }

        let converged_error = errors[90..100].iter().sum::<f32>() / 10.0;
        let frames_to_converge = errors.iter().position(|&e| e < 0.05).unwrap_or(100);

        println!("  Frames to converge (<0.05 error): {frames_to_converge}");
        println!("  Converged error (last 10):        {converged_error:.6}");
        println!("  Final coherence:                  {:.4}", m.coherence());
        println!();
    }

    // ═══════════════════════════════════════════════════════════════════
    // Benchmark 2: Scene Change Detection
    // ═══════════════════════════════════════════════════════════════════
    println!("━━━ Benchmark 2: Scene Change Detection ━━━━━━━━━━━━━━━━━━━━━━━━━");
    {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, width, height);
        let scene_a = solid_frame(50);
        let scene_b = solid_frame(200);

        // Converge on A
        for _ in 0..50 {
            m.observe_frame(&scene_a, width, height, 1, dt);
        }
        let baseline_error = m.prediction_error();

        // Switch to B
        let tel = m.observe_frame(&scene_b, width, height, 1, dt);
        let spike_error = tel.prediction_error;
        let spike_salient = tel.num_salient_patches;

        // Recovery: how many frames to drop back below 0.1?
        let mut recovery_frames = 0;
        for i in 0..50 {
            m.observe_frame(&scene_b, width, height, 1, dt);
            if m.prediction_error() < 0.1 {
                recovery_frames = i + 1;
                break;
            }
        }

        println!("  Baseline error (after 50 frames):  {baseline_error:.6}");
        println!("  Spike error (scene change):        {spike_error:.6}");
        println!(
            "  Spike ratio:                       {:.1}×",
            spike_error / baseline_error.max(1e-8)
        );
        println!("  Salient patches at spike:          {spike_salient}");
        println!("  Recovery frames (<0.1):            {recovery_frames}");
        println!();
    }

    // ═══════════════════════════════════════════════════════════════════
    // Benchmark 3: Alternating Pattern Learning
    // ═══════════════════════════════════════════════════════════════════
    println!("━━━ Benchmark 3: Alternating Pattern Learning ━━━━━━━━━━━━━━━━━━━");
    {
        let mut cfg = VisionConfig::default();
        cfg.training.error_threshold = 0.05; // more aggressive training
        let mut m = VisionManifold::new(cfg, width, height);
        let frame_a = checkerboard_frame(8, 100);
        let frame_b = gradient_frame(0);

        let mut early_errors = Vec::new();
        let mut late_errors = Vec::new();

        for i in 0..200 {
            let frame = if i % 2 == 0 { &frame_a } else { &frame_b };
            let tel = m.observe_frame(frame, width, height, 1, dt);
            if i < 20 {
                early_errors.push(tel.prediction_error);
            }
            if i >= 180 {
                late_errors.push(tel.prediction_error);
            }
        }

        let early_mean: f32 = early_errors.iter().sum::<f32>() / early_errors.len() as f32;
        let late_mean: f32 = late_errors.iter().sum::<f32>() / late_errors.len() as f32;

        println!("  Early mean error (first 20):  {early_mean:.6}");
        println!("  Late mean error (last 20):    {late_mean:.6}");
        println!(
            "  Improvement ratio:            {:.2}×",
            early_mean / late_mean.max(1e-8)
        );
        println!("  Training steps:               {}", m.training_steps());
        println!();
    }

    // ═══════════════════════════════════════════════════════════════════
    // Benchmark 4: Multi-Scale Discrimination
    // ═══════════════════════════════════════════════════════════════════
    println!("━━━ Benchmark 4: Multi-Scale Discrimination ━━━━━━━━━━━━━━━━━━━━━");
    {
        let cfg = VisionConfig::default();
        let mut encoder = MultiScaleEncoder::new(&cfg, width, height);

        // Fine checkerboard (2px blocks) should differ from coarse (32px blocks)
        let fine_checker = checkerboard_frame(2, 128);
        let coarse_checker = checkerboard_frame(32, 128);
        let uniform = solid_frame(128);

        let (hv_fine, _, _) = encoder.encode_frame(&fine_checker, width, height, 1);
        let (hv_coarse, _, _) = encoder.encode_frame(&coarse_checker, width, height, 1);
        let (hv_uniform, _, _) = encoder.encode_frame(&uniform, width, height, 1);

        let sim_fine_coarse = hv_fine.similarity(&hv_coarse);
        let sim_fine_uniform = hv_fine.similarity(&hv_uniform);
        let sim_coarse_uniform = hv_coarse.similarity(&hv_uniform);

        println!("  Similarity(fine_check, coarse_check): {sim_fine_coarse:.4}");
        println!("  Similarity(fine_check, uniform):      {sim_fine_uniform:.4}");
        println!("  Similarity(coarse_check, uniform):    {sim_coarse_uniform:.4}");
        println!(
            "  Discriminating: {}",
            if sim_fine_coarse < 0.95 {
                "YES — different patterns produce different HVs"
            } else {
                "NO — patterns too similar"
            }
        );
        println!();
    }

    // ═══════════════════════════════════════════════════════════════════
    // Benchmark 5: Temporal Prediction Accuracy
    // ═══════════════════════════════════════════════════════════════════
    println!("━━━ Benchmark 5: Temporal Prediction Accuracy ━━━━━━━━━━━━━━━━━━━");
    {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, width, height);
        let frame = gradient_frame(77);

        // Converge
        for _ in 0..50 {
            m.observe_frame(&frame, width, height, 1, dt);
        }

        let acc: HorizonAccuracy = m.evaluate_horizons();
        for (i, label) in acc.labels.iter().enumerate() {
            println!(
                "  {:<14} ({:.3}s): error={:.6}",
                label, acc.horizons[i], acc.errors[i]
            );
        }
        println!();
    }

    // ═══════════════════════════════════════════════════════════════════
    // Benchmark 6: Throughput
    // ═══════════════════════════════════════════════════════════════════
    println!("━━━ Benchmark 6: Throughput ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, width, height);

        // Warmup
        let frame = noise_frame(12345);
        for _ in 0..10 {
            m.observe_frame(&frame, width, height, 1, dt);
        }

        // Benchmark
        let n_frames = 200;
        let t0 = Instant::now();
        let mut total_encode_us = 0u64;
        let mut total_evolve_us = 0u64;

        for i in 0..n_frames {
            let f = noise_frame(i as u64 * 7919);
            let tel = m.observe_frame(&f, width, height, 1, dt);
            total_encode_us += tel.encode_time_us;
            total_evolve_us += tel.evolve_time_us;
        }

        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let fps = n_frames as f64 / (total_ms / 1000.0);
        let encode_avg_us = total_encode_us as f64 / n_frames as f64;
        let evolve_avg_us = total_evolve_us as f64 / n_frames as f64;

        println!("  Resolution:     {width}×{height} (1 channel)");
        println!("  Frames:         {n_frames}");
        println!("  Total time:     {total_ms:.1} ms");
        println!("  Throughput:     {fps:.1} fps");
        println!("  Encode avg:     {encode_avg_us:.1} µs");
        println!("  Evolve avg:     {evolve_avg_us:.1} µs");
        println!(
            "  30fps budget:   {}",
            if fps >= 30.0 { "MET" } else { "MISSED" }
        );
        println!();
    }

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Benchmark complete.");
    println!("═══════════════════════════════════════════════════════════════════");
}

#[cfg(not(feature = "vision-manifold"))]
fn main() {
    eprintln!("This example requires the `vision-manifold` feature.");
    eprintln!("Run with: cargo run --example vision_manifold_benchmark --features vision-manifold");
}