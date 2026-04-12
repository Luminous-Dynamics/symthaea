// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MOT17 Benchmark — Object Tracking Validation on Real-World Video
//!
//! Processes MOT17 pedestrian tracking sequences through the P1-P8 vision
//! manifold and reports tracking metrics against ground truth.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --example benchmark_vision_mot --features vision-manifold -- \
//!   data/mot-sample/train/MOT17-02/ --frames 100
//! ```
//!
//! ## Metrics Reported
//!
//! - Track persistence: do ObjectMemory tracks survive across frames?
//! - Working memory utilization: does WM saturate at 4±1 (Cowan 2001)?
//! - Imagination accuracy: does surprise spike on scene changes?
//! - Scene graph density: meaningful spatial relations?
//! - Processing speed: frames per second

#[cfg(not(feature = "vision-manifold"))]
fn main() {
    eprintln!("Requires: --features vision-manifold");
}

#[cfg(feature = "vision-manifold")]
fn main() {
    use image::GenericImageView;
    use std::time::Instant;
    use symthaea::perception::vision_manifold::{VisionConfig, VisionManifold};

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: benchmark_vision_mot <mot-sequence-dir/> [--frames N]");
        eprintln!("  e.g.: data/mot-sample/train/MOT17-02/ --frames 100");
        std::process::exit(1);
    }

    let seq_dir = std::path::Path::new(&args[1]);
    let max_frames: usize = args
        .iter()
        .position(|a| a == "--frames")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    // Parse sequence info
    let seqinfo_path = seq_dir.join("seqinfo.ini");
    let seqinfo = std::fs::read_to_string(&seqinfo_path).expect("Missing seqinfo.ini");
    let seq_name = seqinfo
        .lines()
        .find(|l| l.starts_with("name="))
        .map(|l| l.trim_start_matches("name="))
        .unwrap_or("unknown");

    println!("=== Symthaea Vision — MOT Benchmark ===");
    println!("Sequence: {seq_name}");
    println!("Max frames: {max_frames}");

    // Parse ground truth: frame,id,x,y,w,h,conf,class,visibility
    let gt_path = seq_dir.join("gt/gt.txt");
    let gt_entries: Vec<GtEntry> = if gt_path.exists() {
        std::fs::read_to_string(&gt_path)
            .unwrap_or_default()
            .lines()
            .filter_map(|line| {
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() >= 9 {
                    Some(GtEntry {
                        frame: parts[0].parse().ok()?,
                        id: parts[1].parse().ok()?,
                        x: parts[2].parse().ok()?,
                        y: parts[3].parse().ok()?,
                        w: parts[4].parse().ok()?,
                        h: parts[5].parse().ok()?,
                        conf: parts[6].parse().unwrap_or(1.0),
                        class: parts[7].parse().unwrap_or(1),
                        visibility: parts[8].parse().unwrap_or(1.0),
                    })
                } else {
                    None
                }
            })
            .collect()
    } else {
        Vec::new()
    };
    println!("Ground truth: {} annotations", gt_entries.len());

    // Configure manifold
    let mut cfg = VisionConfig::default();
    cfg.enable_depth = true;
    cfg.enable_object_binding = true;
    cfg.enable_temporal_binding = true;

    let target_w = 64u32;
    let target_h = 64u32;
    let mut manifold = VisionManifold::new(cfg, target_w, target_h);
    manifold.enable_object_memory(16);
    manifold.enable_working_memory(4);
    manifold.enable_scene_graph();
    manifold.enable_scene_memory(16);

    let dt = 1.0 / 30.0; // 30fps
    let img_dir = seq_dir.join("img1");

    // Metrics accumulators
    let mut total_ms = 0.0f64;
    let mut pe_values: Vec<f32> = Vec::new();
    let mut img_surp_values: Vec<f32> = Vec::new();
    let mut wm_loads: Vec<usize> = Vec::new();
    let mut sg_edges: Vec<usize> = Vec::new();
    let mut obj_counts: Vec<usize> = Vec::new();
    let mut motion_values: Vec<f32> = Vec::new();
    let mut frames_processed = 0usize;

    println!("\nProcessing frames...\n");

    for frame_idx in 1..=max_frames {
        let img_path = img_dir.join(format!("{:06}.jpg", frame_idx));
        if !img_path.exists() {
            break;
        }

        let img = match image::open(&img_path) {
            Ok(img) => img,
            Err(_) => continue,
        };
        let resized = img.resize_exact(target_w, target_h, image::imageops::FilterType::Triangle);
        let rgb = resized.to_rgb8();
        let pixels: Vec<u8> = rgb.into_raw();

        let t0 = Instant::now();
        let tel = manifold.observe_frame(&pixels, target_w, target_h, 3, dt);
        let frame_ms = t0.elapsed().as_secs_f64() * 1000.0;
        total_ms += frame_ms;

        let obj_count = manifold.object_memory().map_or(0, |m| m.len());
        let wm_load = manifold.working_memory().map_or(0, |wm| wm.load());
        let sg_edge_count = manifold.scene_graph().map_or(0, |sg| sg.num_edges());

        pe_values.push(tel.prediction_error);
        img_surp_values.push(tel.imagination_surprise);
        wm_loads.push(wm_load);
        sg_edges.push(sg_edge_count);
        obj_counts.push(obj_count);
        motion_values.push(tel.motion_surprise);
        frames_processed += 1;

        // Print every 10 frames
        if frame_idx % 10 == 1 || frame_idx == max_frames {
            // Count GT objects in this frame
            let gt_objects = gt_entries.iter().filter(|e| e.frame == frame_idx as u32 && e.conf > 0.0).count();
            println!(
                "  Frame {:4} | {:.1}ms | PE={:.3} | ImgSurp={:.3} | Obj={:2} | WM={}/{} | SG={:2}e | GT={:2} | motion={:.3}",
                frame_idx, frame_ms,
                tel.prediction_error, tel.imagination_surprise,
                obj_count, wm_load, 4, sg_edge_count,
                gt_objects, tel.motion_surprise,
            );
        }
    }

    // Compute metrics
    let avg_ms = total_ms / frames_processed.max(1) as f64;
    let avg_pe: f32 = pe_values.iter().sum::<f32>() / pe_values.len().max(1) as f32;
    let avg_surp: f32 = img_surp_values.iter().sum::<f32>() / img_surp_values.len().max(1) as f32;
    let avg_wm: f32 = wm_loads.iter().sum::<usize>() as f32 / wm_loads.len().max(1) as f32;
    let avg_sg: f32 = sg_edges.iter().sum::<usize>() as f32 / sg_edges.len().max(1) as f32;
    let avg_obj: f32 = obj_counts.iter().sum::<usize>() as f32 / obj_counts.len().max(1) as f32;
    let max_wm = wm_loads.iter().copied().max().unwrap_or(0);
    let max_obj = obj_counts.iter().copied().max().unwrap_or(0);

    // Track persistence: how many frames does the average track survive?
    let track_persistence = if let Some(obj_mem) = manifold.object_memory() {
        let tracks = obj_mem.tracks();
        if tracks.is_empty() {
            0.0
        } else {
            tracks.iter().map(|t| t.track_length as f64).sum::<f64>() / tracks.len() as f64
        }
    } else {
        0.0
    };

    // Imagination accuracy: ratio of max to min surprise (novelty detection)
    let max_surp = img_surp_values.iter().copied().fold(0.0f32, f32::max);
    let min_surp = img_surp_values.iter().copied().filter(|s| *s > 0.0).fold(f32::MAX, f32::min);
    let novelty_ratio = if min_surp > 0.0 && min_surp < f32::MAX {
        max_surp / min_surp
    } else {
        0.0
    };

    // Scene descriptions
    let descriptions = manifold.describe_scene();

    println!("\n=== MOT Benchmark Results ===");
    println!("Sequence: {seq_name}");
    println!("Frames: {frames_processed}");
    println!();
    println!("--- Performance ---");
    println!("  Average:   {:.1}ms/frame ({:.0} Hz)", avg_ms, 1000.0 / avg_ms);
    println!("  Budget:    {}", if avg_ms < 50.0 { "FITS 20Hz ✓" } else { "EXCEEDS 20Hz ✗" });
    println!();
    println!("--- Object Tracking ---");
    println!("  Avg objects tracked: {:.1}", avg_obj);
    println!("  Max objects tracked: {}", max_obj);
    println!("  Track persistence:   {:.1} frames/track", track_persistence);
    println!();
    println!("--- Working Memory (Cowan 2001) ---");
    println!("  Avg WM load: {:.1}/{}", avg_wm, 4);
    println!("  Max WM load: {}/{}", max_wm, 4);
    println!("  Capacity saturated: {}", if max_wm >= 4 { "YES ✓" } else { "no" });
    println!();
    println!("--- Active Inference ---");
    println!("  Avg prediction error:     {:.4}", avg_pe);
    println!("  Avg imagination surprise: {:.4}", avg_surp);
    println!("  Novelty detection ratio:  {:.2}x", novelty_ratio);
    println!();
    println!("--- Scene Understanding ---");
    println!("  Avg scene graph edges: {:.1}", avg_sg);
    println!("  Scene descriptions:    {} relational triples", descriptions.len());
    if !descriptions.is_empty() {
        for (s, r, o) in descriptions.iter().take(5) {
            println!("    \"{s}\" {r} \"{o}\"");
        }
    }
    println!();

    // Ground truth comparison
    if !gt_entries.is_empty() {
        let gt_unique_ids: std::collections::HashSet<u32> =
            gt_entries.iter().map(|e| e.id).collect();
        let gt_total_objects = gt_unique_ids.len();
        let gt_frames: std::collections::HashSet<u32> =
            gt_entries.iter().map(|e| e.frame).collect();
        let gt_avg_objects_per_frame =
            gt_entries.len() as f32 / gt_frames.len().max(1) as f32;
        println!("--- Ground Truth Comparison ---");
        println!("  GT unique objects:       {}", gt_total_objects);
        println!("  GT avg objects/frame:    {:.1}", gt_avg_objects_per_frame);
        println!("  Manifold avg objects:    {:.1}", avg_obj);
        println!("  Coverage ratio:          {:.1}%", avg_obj / gt_avg_objects_per_frame * 100.0);
    }

    // Dream replay consolidation
    let replays = manifold.dream_replay(0.1, 3);
    if !replays.is_empty() {
        println!();
        println!("--- Dream Replay ({} memories) ---", replays.len());
        for (i, r) in replays.iter().take(3).enumerate() {
            println!(
                "  Memory {}: sim_to_current={:.4}",
                i,
                r.similarity(manifold.state()),
            );
        }
    }
}

#[cfg(feature = "vision-manifold")]
#[derive(Debug)]
struct GtEntry {
    frame: u32,
    id: u32,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    conf: f32,
    class: u32,
    visibility: f32,
}
