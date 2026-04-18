// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Live Camera Vision — process real images through the full vision manifold.
//!
//! Loads PNG images from a directory or captures from ADB, resizes to 64×64,
//! and feeds them through P1-P8 vision pipeline with full telemetry.
//!
//! ## Usage
//!
//! ```bash
//! # Process a single image:
//! cargo run --release --example vision_live_camera --features vision-manifold -- /tmp/pixel_screen.png
//!
//! # Process a directory of frames:
//! cargo run --release --example vision_live_camera --features vision-manifold -- /tmp/frames/
//! ```

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
        eprintln!("Usage: vision_live_camera <image.png | directory/>");
        eprintln!("  Processes real images through the P1-P8 vision manifold.");
        std::process::exit(1);
    }

    // Collect image paths
    let path = std::path::Path::new(&args[1]);
    let mut image_paths: Vec<std::path::PathBuf> = if path.is_dir() {
        let mut paths: Vec<_> = std::fs::read_dir(path)
            .expect("Cannot read directory")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .map_or(false, |ext| ext == "png" || ext == "jpg" || ext == "jpeg")
            })
            .collect();
        paths.sort();
        paths
    } else {
        vec![path.to_path_buf()]
    };

    if image_paths.is_empty() {
        eprintln!("No images found.");
        std::process::exit(1);
    }

    println!("=== Symthaea Vision ��� Live Camera Processing ===");
    println!("Images: {}", image_paths.len());

    // Configure manifold with all features
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

    let dt = 0.033;
    let mut total_ms = 0.0f64;

    for (i, img_path) in image_paths.iter().enumerate() {
        // Load and resize to 64×64
        let img = match image::open(img_path) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("  Skipping {}: {}", img_path.display(), e);
                continue;
            }
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
        let sg_edges = manifold.scene_graph().map_or(0, |sg| sg.num_edges());

        println!(
            "\nFrame {:3} [{}] | {:.1}ms",
            i + 1,
            img_path.file_name().unwrap_or_default().to_string_lossy(),
            frame_ms,
        );
        println!(
            "  PE={:.3} Coh={:.3} ImgSurp={:.3} motion={:.3}",
            tel.prediction_error, tel.manifold_coherence,
            tel.imagination_surprise, tel.motion_surprise,
        );
        println!(
            "  Objects={} WM={}/{} SG={}edges Salient={}",
            obj_count, wm_load, 4, sg_edges, tel.num_salient_patches,
        );

        // Scene description
        let desc = manifold.describe_scene();
        if !desc.is_empty() {
            print!("  Scene: ");
            for (j, (s, r, o)) in desc.iter().take(4).enumerate() {
                if j > 0 { print!(", "); }
                print!("{s} {r} {o}");
            }
            println!();
        }

        // Working memory
        if let Some(wm) = manifold.working_memory() {
            if wm.load() > 0 {
                print!("  Attending: ");
                for (j, slot) in wm.slots().iter().enumerate() {
                    if j > 0 { print!(", "); }
                    print!("#{} @({},{})", slot.track_id, slot.centroid_row, slot.centroid_col);
                }
                println!();
            }
        }
    }

    let n = image_paths.len().max(1);
    let avg_ms = total_ms / n as f64;
    println!("\n=== Summary ===");
    println!("Frames: {n}");
    println!("Average: {:.1}ms/frame ({:.0} Hz)", avg_ms, 1000.0 / avg_ms);
    println!(
        "Budget: {}",
        if avg_ms < 50.0 { "FITS 20Hz ✓" } else { "EXCEEDS 20Hz ✗" }
    );

    // Final imagination
    let dreams = manifold.dream_ahead(3, dt);
    println!("\nImagining 3 steps ahead...");
    for (i, d) in dreams.iter().enumerate() {
        println!(
            "  +{}: sim_to_now={:.4}",
            i + 1,
            d.similarity(manifold.state())
        );
    }
}
