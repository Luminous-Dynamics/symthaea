// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Vision Manifold Webcam Demo
//!
//! Captures frames from a `MockCameraSource` (or real camera with `--features webcam`),
//! processes them through the full vision pipeline (object tracking, working memory,
//! scene graph, imagination), and prints per-frame telemetry.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example vision_webcam_demo --features vision-manifold
//! ```
//!
//! ## What it demonstrates
//!
//! - Full P1-P6 vision pipeline in action
//! - Object identity persistence across frames
//! - Working memory bounded capacity
//! - Scene graph spatial relations
//! - Imagination-reality surprise
//! - Performance (ms/frame)

#[cfg(not(feature = "vision-manifold"))]
fn main() {
    eprintln!("This example requires: --features vision-manifold");
}

#[cfg(feature = "vision-manifold")]
fn main() {
    use symthaea::perception::vision_manifold::{
        VisionBridge, VisionConfig, VisionManifold,
    };
    use std::time::Instant;

    println!("=== Symthaea Vision Manifold Demo ===");
    println!("Processing 100 synthetic frames through the full P1-P6 pipeline\n");

    // Configure with all features
    let mut cfg = VisionConfig::default();
    cfg.enable_depth = true;
    cfg.enable_object_binding = true;
    cfg.enable_temporal_binding = true;

    let width = 64u32;
    let height = 64u32;
    let mut manifold = VisionManifold::new(cfg, width, height);
    manifold.enable_object_memory(16);
    manifold.enable_working_memory(4);
    manifold.enable_scene_graph();
    manifold.enable_scene_memory(16);

    let dt = 0.033; // ~30fps
    let mut total_ms = 0.0f64;

    for frame_idx in 0..100 {
        // Generate a synthetic scene that changes over time
        let frame = generate_scene(width, height, frame_idx);

        let t0 = Instant::now();
        let telemetry = manifold.observe_frame(&frame, width, height, 3, dt);
        let frame_ms = t0.elapsed().as_secs_f64() * 1000.0;
        total_ms += frame_ms;

        // Print telemetry every 10 frames
        if frame_idx % 10 == 0 || frame_idx == 99 {
            let obj_count = manifold
                .object_memory()
                .map_or(0, |m| m.len());
            let wm_load = manifold
                .working_memory()
                .map_or(0, |wm| wm.load());
            let sg_edges = manifold
                .scene_graph()
                .map_or(0, |sg| sg.num_edges());

            println!(
                "Frame {:3} | {:.1}ms | PE={:.3} | Coh={:.3} | ImgSurp={:.3} | \
                 Sal={} | Obj={} | WM={}/{} | SG={}e | motion={:.3}",
                telemetry.frame_sequence,
                frame_ms,
                telemetry.prediction_error,
                telemetry.manifold_coherence,
                telemetry.imagination_surprise,
                telemetry.num_salient_patches,
                obj_count,
                wm_load,
                4, // capacity
                sg_edges,
                telemetry.motion_surprise,
            );
        }
    }

    let avg_ms = total_ms / 100.0;
    let hz = 1000.0 / avg_ms;
    println!("\n=== Performance ===");
    println!("Average: {:.1}ms/frame ({:.0} Hz)", avg_ms, hz);
    println!(
        "Budget:  {}",
        if avg_ms < 50.0 {
            "FITS 20Hz cognitive loop ✓"
        } else {
            "EXCEEDS 20Hz budget ✗"
        }
    );

    // Dream ahead: predict 5 future frames
    let dreams = manifold.dream_ahead(5, dt);
    println!("\n=== Visual Imagination (5 steps) ===");
    for (i, dream) in dreams.iter().enumerate() {
        let sim_to_state = dream.similarity(manifold.state());
        println!(
            "Dream +{}: norm={:.3}, sim_to_current={:.3}",
            i + 1,
            dream.norm(),
            sim_to_state
        );
    }

    // Working memory contents
    if let Some(wm) = manifold.working_memory() {
        println!("\n=== Working Memory ({}/{}) ===", wm.load(), wm.capacity());
        for slot in wm.slots() {
            println!(
                "  Track #{}: sal={:.3}, pos=({},{}), held {} frames",
                slot.track_id,
                slot.saliency,
                slot.centroid_row,
                slot.centroid_col,
                manifold.frame_count() - slot.entered_at_frame,
            );
        }
    }

    // Scene graph
    if let Some(sg) = manifold.scene_graph() {
        println!("\n=== Scene Graph ({} edges) ===", sg.num_edges());
        for edge in sg.edges().iter().take(10) {
            println!(
                "  #{} {:?} #{} (hv_norm={:.3})",
                edge.subject_id,
                edge.relation,
                edge.object_id,
                edge.relation_hv.norm(),
            );
        }
    }

    // Broca bridge: scene description as relational triples
    let descriptions = manifold.describe_scene();
    if !descriptions.is_empty() {
        println!("\n=== Scene Description (Broca bridge) ===");
        for (subj, rel, obj) in descriptions.iter().take(8) {
            println!("  \"{subj}\" {rel} \"{obj}\"");
        }
    }

    // Dream replay: consolidate scene memories
    let replays = manifold.dream_replay(0.1, 3);
    if !replays.is_empty() {
        println!("\n=== Dream Replay ({} memories consolidated) ===", replays.len());
        for (i, replay) in replays.iter().enumerate() {
            println!(
                "  Memory {}: norm={:.3}, sim_to_current={:.3}",
                i,
                replay.norm(),
                replay.similarity(manifold.state()),
            );
        }
    }
}

#[cfg(feature = "vision-manifold")]
fn generate_scene(width: u32, height: u32, frame_idx: usize) -> Vec<u8> {
    let mut pixels = vec![0u8; (width * height * 3) as usize];
    let w = width as usize;
    let h = height as usize;

    // Background: slowly shifting gradient
    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * 3;
            let bg = ((x + frame_idx) % 64) as u8;
            pixels[base] = bg;
            pixels[base + 1] = bg;
            pixels[base + 2] = bg;
        }
    }

    // Object 1: Red square that moves rightward
    let obj1_x = (frame_idx * 2) % w;
    let obj1_y = h / 4;
    draw_rect(&mut pixels, w, obj1_x, obj1_y, 16, 16, [200, 40, 40]);

    // Object 2: Blue square that moves downward
    let obj2_x = w / 4;
    let obj2_y = (frame_idx * 3) % h;
    draw_rect(&mut pixels, w, obj2_x, obj2_y, 12, 12, [40, 40, 200]);

    // Object 3: Green square (stationary)
    draw_rect(&mut pixels, w, 3 * w / 4, 3 * h / 4, 20, 20, [40, 180, 40]);

    // Scene change at frame 50: add yellow flash
    if frame_idx >= 50 && frame_idx < 55 {
        for y in 0..h / 2 {
            for x in 0..w {
                let base = (y * w + x) * 3;
                pixels[base] = 255;
                pixels[base + 1] = 255;
                pixels[base + 2] = 0;
            }
        }
    }

    pixels
}

#[cfg(feature = "vision-manifold")]
fn draw_rect(pixels: &mut [u8], stride: usize, x: usize, y: usize, w: usize, h: usize, color: [u8; 3]) {
    for dy in 0..h {
        for dx in 0..w {
            let px = (x + dx) % stride;
            let py = y + dy;
            if py < pixels.len() / (stride * 3) {
                let base = (py * stride + px) * 3;
                if base + 2 < pixels.len() {
                    pixels[base] = color[0];
                    pixels[base + 1] = color[1];
                    pixels[base + 2] = color[2];
                }
            }
        }
    }
}
