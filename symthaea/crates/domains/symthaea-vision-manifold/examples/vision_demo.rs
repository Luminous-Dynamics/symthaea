// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Vision manifold demo: synthetic video with scene memory, motion, and telemetry.
//!
//! Creates a VisionManifold, feeds a moving stripe pattern, and demonstrates:
//! - Temporal coherence (prediction error drops for static scenes)
//! - Scene change detection (error spikes on transitions)
//! - Motion field detection (per-patch motion vectors)
//! - Scene memory recognition (revisiting earlier scenes)
//! - Health telemetry
//!
//! Run with: `cargo run -p symthaea-vision-manifold --example vision_demo`

use symthaea_vision_manifold::{ManifoldHealth, SceneMemory, VisionBridge, VisionConfig};

fn moving_stripe_frame(width: u32, height: u32, offset: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let stripe = ((x + offset) / 8) % 2;
            let gradient = (y * 255 / height.max(1)) as u8;
            pixels.push(if stripe == 0 {
                gradient
            } else {
                255 - gradient
            });
        }
    }
    pixels
}

fn solid_frame(width: u32, height: u32, value: u8) -> Vec<u8> {
    vec![value; (width * height) as usize]
}

/// Frame with a bright spot at (cx, cy) — simulates a moving object.
fn spot_frame(width: u32, height: u32, cx: u32, cy: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let dx = x as i32 - cx as i32;
            let dy = y as i32 - cy as i32;
            let dist_sq = dx * dx + dy * dy;
            let v = if dist_sq < 64 { 255u8 } else { 50u8 };
            pixels.push(v);
        }
    }
    pixels
}

fn main() {
    let width = 64u32;
    let height = 64u32;
    let dt = 0.033; // ~30fps

    println!("=== Vision Manifold Demo ===\n");
    println!("Frame size: {width}x{height}, dt={dt}s (~30fps)");
    println!("Pipeline: pixels -> PatchHdcEncoder -> CfC Manifold -> prediction + surprise\n");

    // Create bridge (wraps manifold + attention boost)
    let cfg = VisionConfig::default();
    let mut bridge = VisionBridge::new(cfg, width, height);

    // Scene memory for episodic recognition
    let mut memory = SceneMemory::new(8);

    // Phase 1: Moving stripe (30 frames)
    println!("--- Phase 1: Moving stripe (30 frames) ---");
    for i in 0..30 {
        let frame = moving_stripe_frame(width, height, i);
        let (hv, tel) = bridge.process_frame_with_telemetry(&frame, width, height, 1, dt);

        if i % 10 == 0 || i == 29 {
            println!(
                "  frame {:>2}: pred_err={:.4}, coherence={:.4}, motion={:.4}, salient={}, training={}",
                i,
                tel.prediction_error,
                tel.manifold_coherence,
                tel.motion_surprise,
                tel.num_salient_patches,
                tel.training_triggered,
            );
        }

        // Store scene at frame 15 (mid-stripe)
        if i == 15 {
            memory.remember(&hv, i as u64, vec![]);
            println!("  >> Stored scene landmark at frame {i}");
        }
    }

    // Phase 2: Scene change to solid (20 frames)
    println!("\n--- Phase 2: Scene change -> solid gray (20 frames) ---");
    for i in 30..50 {
        let frame = solid_frame(width, height, 128);
        let (hv, tel) = bridge.process_frame_with_telemetry(&frame, width, height, 1, dt);

        if i == 30 || i == 35 || i == 49 {
            println!(
                "  frame {:>2}: pred_err={:.4}, coherence={:.4}, motion={:.4}, salient={}",
                i,
                tel.prediction_error,
                tel.manifold_coherence,
                tel.motion_surprise,
                tel.num_salient_patches,
            );
        }

        // Store solid scene
        if i == 40 {
            memory.remember(&hv, i as u64, vec![]);
            println!("  >> Stored scene landmark at frame {i}");
        }
    }

    // Phase 3: Moving spot — demonstrates motion field detection
    println!("\n--- Phase 3: Moving spot (20 frames) ---");
    println!("  A bright spot moves rightward across the frame.");
    for i in 50..70 {
        let cx = 10 + (i - 50) * 2; // Move rightward
        let cy = 32;
        let frame = spot_frame(width, height, cx, cy);
        let (_, tel) = bridge.process_frame_with_telemetry(&frame, width, height, 1, dt);

        if i % 5 == 0 || i == 69 {
            // Show motion vectors summary
            let vectors = bridge.manifold().motion_vectors();
            let active_count = vectors
                .iter()
                .filter(|v| (v[0] * v[0] + v[1] * v[1]).sqrt() > 0.01)
                .count();
            let max_mag = bridge
                .manifold()
                .motion_saliency()
                .iter()
                .copied()
                .fold(0.0f32, f32::max);

            println!(
                "  frame {:>2}: spot@({cx},{cy}), motion_surprise={:.4}, mf_norm={:.4}, active_patches={}/{}, max_mag={:.4}",
                i,
                tel.motion_surprise,
                tel.motion_field_norm,
                active_count,
                vectors.len(),
                max_mag,
            );
        }
    }

    // Phase 4: Return to stripe (scene recognition)
    println!("\n--- Phase 4: Return to moving stripe (20 frames) ---");
    for i in 70..90 {
        let frame = moving_stripe_frame(width, height, i - 70);
        let (hv, tel) = bridge.process_frame_with_telemetry(&frame, width, height, 1, dt);

        // Try scene recognition
        if let Some(m) = memory.recognize(&hv, i as u64) {
            println!(
                "  frame {:>2}: RECOGNIZED scene {} (sim={:.4}, stored at frame {}, {} frames ago)",
                i, m.scene_id, m.similarity, m.stored_at_frame, m.frames_since_stored,
            );
        } else if i == 70 || i == 75 || i == 89 {
            println!(
                "  frame {:>2}: pred_err={:.4}, coherence={:.4}, motion={:.4}, no scene match",
                i, tel.prediction_error, tel.manifold_coherence, tel.motion_surprise,
            );
        }
    }

    // Health telemetry
    let health: ManifoldHealth = bridge.manifold().compute_health();
    println!("\n--- Manifold Health ---");
    println!("  total_frames:       {}", health.total_frames);
    println!("  training_steps:     {}", health.total_training_steps);
    println!("  weight_drift:       {:.4}", health.weight_drift);
    println!("  tau_value:          {:.4}", health.tau_value);
    println!("  encoder_entropy:    {:.4}", health.encoder_weight_entropy);
    println!("  training_frequency: {:.4}", health.training_frequency);
    println!("  pred_error:         {:.4}", health.mean_prediction_error);
    println!("  coherence:          {:.4}", health.mean_coherence);
    println!("  is_healthy:         {}", health.is_healthy);

    println!(
        "\n=== Demo complete: {} frames processed ===",
        bridge.frame_count()
    );
}
