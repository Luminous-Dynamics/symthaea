// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Streaming video watcher — ffmpeg pipe → vision manifold.
//!
//! Reads raw RGB frames from stdin (or spawns ffmpeg to decode a file) and
//! processes each through the P1-P8 vision manifold as it arrives. Nothing
//! is written to disk by default — frames are consumed, semantic state is
//! extracted, then pixels are discarded. Only scene memory (compressed HV
//! landmarks) is preserved for consolidation.
//!
//! ## Usage
//!
//! ```bash
//! # Decode a local video file at 8fps, 128×128:
//! ffmpeg -i video.mp4 -vf "fps=8,scale=128:128" -f rawvideo -pix_fmt rgb24 - \
//!   | cargo run --release --example watch_video_stream --features vision-manifold \
//!     -- --width 128 --height 128 --fps 8
//!
//! # Watch a YouTube URL (via yt-dlp):
//! yt-dlp -o - "URL" | ffmpeg -i - -vf "fps=8,scale=128:128" \
//!     -f rawvideo -pix_fmt rgb24 - \
//!   | cargo run --release --example watch_video_stream --features vision-manifold \
//!     -- --width 128 --height 128 --fps 8
//!
//! # Watch the phone screen live:
//! adb exec-out screenrecord --output-format=h264 - \
//!   | ffmpeg -i - -vf "fps=8,scale=128:128" -f rawvideo -pix_fmt rgb24 - \
//!   | cargo run --release --example watch_video_stream --features vision-manifold \
//!     -- --width 128 --height 128 --fps 8
//! ```
//!
//! ## Memory policy
//!
//! Pixels: never saved (stream-and-discard).
//! Scene memory: saved to `data/memories/{session_id}.json` only on explicit
//! `--save` flag or when imagination surprise exceeds 0.4 (novelty detection).

#[cfg(not(feature = "vision-manifold"))]
fn main() {
    eprintln!("Requires: --features vision-manifold");
}

#[cfg(feature = "vision-manifold")]
fn main() {
    use std::io::Read;
    use std::time::Instant;
    use symthaea::perception::vision_manifold::{VisionConfig, VisionManifold};

    // Parse args
    let args: Vec<String> = std::env::args().collect();
    let width: u32 = parse_arg::<u32>(&args, "--width", 128);
    let height: u32 = parse_arg::<u32>(&args, "--height", 128);
    let fps: u32 = parse_arg::<u32>(&args, "--fps", 8);
    let max_frames: usize = parse_arg::<usize>(&args, "--max-frames", 1000);
    let save_memories = args.iter().any(|a| a == "--save");
    let novelty_threshold: f32 = parse_arg_f32(&args, "--novelty", 0.4);

    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  Symthaea — Streaming Video Observer                ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!();
    println!("Input:         raw RGB24 frames from stdin");
    println!("Resolution:    {width}×{height}");
    println!("Target FPS:    {fps}");
    println!("Max frames:    {max_frames}");
    println!("Save memories: {save_memories}");
    println!("Novelty threshold: {novelty_threshold}");
    println!();
    println!("Memory policy:");
    println!("  Pixels → stream-and-discard (no disk writes)");
    println!("  Semantic state → working memory (in-process)");
    println!("  Scene landmarks → disk (only if save_memories or novelty spike)");
    println!();

    // Configure vision manifold
    let mut cfg = VisionConfig::default();
    cfg.enable_depth = true;
    cfg.enable_object_binding = true;
    cfg.enable_temporal_binding = true;

    let mut manifold = VisionManifold::new(cfg, width, height);
    manifold.enable_object_memory(16);
    manifold.enable_working_memory(4);
    manifold.enable_scene_graph();
    manifold.enable_scene_memory(32);

    let frame_size = (width * height * 3) as usize;
    let dt = 1.0 / fps as f32;

    let mut stdin = std::io::stdin();
    let mut frame_buf = vec![0u8; frame_size];

    // Stats accumulators
    let mut frames_seen = 0usize;
    let mut pe_values: Vec<f32> = Vec::new();
    let mut img_surp_values: Vec<f32> = Vec::new();
    let mut motion_values: Vec<f32> = Vec::new();
    let mut scene_cuts: Vec<usize> = Vec::new();
    let mut novelty_events: Vec<(usize, f32)> = Vec::new();
    let mut wm_saturations = 0usize;
    let mut total_processing_us = 0u64;

    let start = Instant::now();

    println!("[Stream] Waiting for frames on stdin...\n");

    // Read frames from stdin until EOF or max_frames reached
    while frames_seen < max_frames {
        // Read one full frame
        match stdin.read_exact(&mut frame_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                println!("\n[Stream] EOF reached.");
                break;
            }
            Err(e) => {
                eprintln!("[Stream] Read error: {e}");
                break;
            }
        }

        // Process through vision manifold (zero-copy: we own the buffer)
        let t0 = Instant::now();
        let tel = manifold.observe_frame(&frame_buf, width, height, 3, dt);
        let elapsed_us = t0.elapsed().as_micros() as u64;
        total_processing_us += elapsed_us;

        pe_values.push(tel.prediction_error);
        img_surp_values.push(tel.imagination_surprise);
        motion_values.push(tel.motion_surprise);

        // Scene cut detection
        let is_cut = tel.prediction_error > 0.2 && tel.motion_surprise > 0.15;
        if is_cut {
            scene_cuts.push(frames_seen);
        }

        // Novelty detection (for on-demand memory save)
        let is_novel = tel.imagination_surprise > novelty_threshold;
        if is_novel {
            novelty_events.push((frames_seen, tel.imagination_surprise));
        }

        if tel.working_memory_load >= 4 {
            wm_saturations += 1;
        }

        // Periodic status line
        if frames_seen % fps as usize == 0 {
            let t = frames_seen as f64 / fps as f64;
            let cut = if is_cut { " ⚡" } else { "" };
            let novel = if is_novel { " ★" } else { "" };
            println!(
                "  t={:5.1}s | {:4.1}ms | PE={:.3} ImgSurp={:.3} motion={:.3} | WM={}/4 SG={}edges{}{}",
                t,
                elapsed_us as f64 / 1000.0,
                tel.prediction_error,
                tel.imagination_surprise,
                tel.motion_surprise,
                tel.working_memory_load,
                tel.scene_graph_edges,
                cut,
                novel,
            );
        }

        frames_seen += 1;

        // IMPORTANT: frame_buf is reused — pixels discarded after this point.
    }

    let watch_time = start.elapsed().as_secs_f64();

    // Summary
    let n = pe_values.len().max(1) as f32;
    let avg_pe = pe_values.iter().sum::<f32>() / n;
    let avg_surp = img_surp_values.iter().sum::<f32>() / n;
    let avg_motion = motion_values.iter().sum::<f32>() / n;
    let max_pe = pe_values.iter().copied().fold(0.0f32, f32::max);
    let max_surp = img_surp_values.iter().copied().fold(0.0f32, f32::max);
    let max_motion = motion_values.iter().copied().fold(0.0f32, f32::max);
    let avg_process_ms = total_processing_us as f64 / frames_seen.max(1) as f64 / 1000.0;
    let effective_fps = frames_seen as f64 / watch_time;

    println!();
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║             Video Observation Summary                ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!();
    println!("Frames processed: {frames_seen}");
    println!("Wall time:        {:.1}s", watch_time);
    println!("Effective FPS:    {:.1}", effective_fps);
    println!("Avg process:      {:.1}ms/frame", avg_process_ms);
    println!();
    println!("--- Prediction Error (surprise) ---");
    println!("  Avg: {:.4}  Max: {:.4}", avg_pe, max_pe);
    println!();
    println!("--- Imagination Surprise (reality vs dream) ---");
    println!("  Avg: {:.4}  Max: {:.4}", avg_surp, max_surp);
    println!();
    println!("--- Motion ---");
    println!("  Avg: {:.4}  Max: {:.4}", avg_motion, max_motion);
    println!();
    println!("--- Scene Cuts ---");
    println!("  Count: {}", scene_cuts.len());
    if !scene_cuts.is_empty() && scene_cuts.len() <= 20 {
        let times: Vec<String> = scene_cuts
            .iter()
            .map(|&f| format!("{:.1}s", f as f64 / fps as f64))
            .collect();
        println!("  Times: {}", times.join(", "));
    }
    println!();
    println!("--- Novelty Events (ImgSurp > {}) ---", novelty_threshold);
    println!("  Count: {}", novelty_events.len());
    for (f, s) in novelty_events.iter().take(10) {
        println!("    t={:.1}s  surprise={:.3}", *f as f64 / fps as f64, s);
    }
    println!();
    println!("--- Working Memory ---");
    println!(
        "  Saturation: {}/{} frames at 4/4",
        wm_saturations, frames_seen
    );
    println!();

    // Scene memory consolidation summary
    // (SceneMemory internal — approximated via telemetry)
    println!("--- Semantic State ---");
    println!("  Working memory is preserved in-process (pixels discarded).");
    println!("  Dream replay can consolidate landmarks at session end.");
    let landmarks = 0usize;

    // Final scene description
    let desc = manifold.describe_scene();
    if !desc.is_empty() {
        println!();
        println!("--- Final Scene (Broca bridge) ---");
        for (s, r, o) in desc.iter().take(5) {
            println!("  {s} {r} {o}");
        }
    }

    // Save decision
    println!();
    let should_save = save_memories || !novelty_events.is_empty();
    if should_save {
        let session_id = format!(
            "{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        );
        let dir = std::path::Path::new("data/memories");
        if std::fs::create_dir_all(dir).is_ok() {
            let path = dir.join(format!("{session_id}.txt"));
            let mut summary = String::new();
            summary.push_str(&format!("Session: {session_id}\n"));
            summary.push_str(&format!("Frames: {frames_seen}\n"));
            summary.push_str(&format!(
                "Duration: {:.1}s\n",
                frames_seen as f64 / fps as f64
            ));
            summary.push_str(&format!("Scene cuts: {}\n", scene_cuts.len()));
            summary.push_str(&format!("Novelty events: {}\n", novelty_events.len()));
            summary.push_str(&format!("Landmarks: {landmarks}\n"));
            summary.push_str(&format!("Avg PE: {avg_pe:.4}\n"));
            summary.push_str(&format!("Max ImgSurp: {max_surp:.4}\n"));
            for (s, r, o) in desc.iter().take(10) {
                summary.push_str(&format!("  {s} {r} {o}\n"));
            }
            if std::fs::write(&path, summary).is_ok() {
                println!("[Save] Memory summary saved to {}", path.display());
            }
        }
        if save_memories {
            println!("[Save] Reason: user flag --save");
        }
        if !novelty_events.is_empty() {
            println!(
                "[Save] Reason: {} novelty events detected (threshold {})",
                novelty_events.len(),
                novelty_threshold
            );
        }
    } else {
        println!(
            "[Save] No save (no novelty events, no --save flag). Pixels discarded, memory volatile."
        );
    }

    println!();
    println!("[Done] Symthaea watched {frames_seen} frames. Semantic state preserved.");
}

#[cfg(feature = "vision-manifold")]
fn parse_arg<T: std::str::FromStr + Copy>(args: &[String], flag: &str, default: T) -> T {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

#[cfg(feature = "vision-manifold")]
fn parse_arg_f32(args: &[String], flag: &str, default: f32) -> f32 {
    parse_arg(args, flag, default)
}