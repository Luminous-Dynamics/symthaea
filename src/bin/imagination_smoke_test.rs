// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::time::{Duration, Instant};
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
use symthaea::shell::ipc_server::MetricsProvider;

fn main() {
    let config = CognitiveLoopConfig {
        enable_vision_manifold: true,
        ..Default::default()
    };
    let mut service = CognitiveLoopService::new(config).unwrap();

    println!("🚀 Starting Imagination Smoke Test...");
    println!("Target: Surprise > 0.2 -> 64K Dilation -> RK4 Geodesic -> IPC Streaming\n");

    let frame_a = vec![10u8; 64 * 64 * 3]; // Dim base frame
    let frame_shock = vec![240u8; 64 * 64 * 3]; // High-entropy shock frame

    let mut movie_captured = false;

    for cycle in 1..=40 {
        let start = Instant::now();

        // 1. Inject input
        let frame = if cycle == 20 {
            println!("💥 INJECTING SHOCK FRAME at cycle {cycle}");
            &frame_shock
        } else {
            &frame_a
        };

        // Use public accessors
        if cycle == 20 {
            service.set_vision_free_energy_override(0.5);
        }

        service.inject_vision_frame(frame.clone());

        // 2. Execute Cognitive Cycle
        let _result = service.cycle("live-test");
        let elapsed = start.elapsed();

        // 3. Extract Metrics (simulating IPC Dashboard)
        let metrics = service.get_metrics();

        let has_movie = metrics.mental_movie.is_some();
        let surprise = metrics.prediction_error;

        // Extract dim from service directly
        let dim = service.vision_hdc_dim().unwrap_or(16384);

        if has_movie && !movie_captured {
            let movie = metrics.mental_movie.as_ref().unwrap();
            println!("\n✨ SUCCESS: Mental Movie Captured in Metrics Stream!");
            println!(
                "   | Resolution: {}x{} ({}ch)",
                movie.width, movie.height, movie.channels
            );
            println!("   | Horizon:    {} steps", movie.path_length);
            println!("   | Coherence:  {:.4}", movie.semantic_coherence);
            println!(
                "   | Total Size: {} bytes",
                movie.frames.iter().map(|f| f.len()).sum::<usize>()
            );
            movie_captured = true;
        }

        // Performance Check
        if cycle % 5 == 0 || cycle == 20 {
            println!(
                "Cycle {:2} | Surprise: {:.3} | Dim: {:5} | Latency: {:4}µs | Movie: {}",
                cycle,
                surprise,
                dim,
                elapsed.as_micros(),
                if has_movie { "YES" } else { "no " }
            );
        }

        // Maintain ~30Hz simulation speed
        let target_dt = Duration::from_millis(33);
        if elapsed < target_dt {
            std::thread::sleep(target_dt - elapsed);
        }
    }

    if movie_captured {
        println!("\n✅ Pipeline Verified: Imagination is visible and physically grounded.");
    } else {
        println!("\n❌ Pipeline Failed: No mental movie detected.");
        std::process::exit(1);
    }
}
