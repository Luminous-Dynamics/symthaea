// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sovereign RDP codec benchmarks.
//!
//! Measures:
//! - HDC tile encoding time
//! - Full-frame change detection (510 tiles @ 1080p)
//! - Delta frame encoding
//! - X11 capture latency (with rdp-x11 feature)
//!
//! Run: `cargo bench --bench rdp_codec_bench`

use std::time::Instant;

use symthaea::swarm::rdp_capture::{ScreenCapture, TestCapture};
use symthaea::swarm::rdp_codec::HybridCodec;

fn make_frame(width: u32, height: u32, seed: u8) -> Vec<u8> {
    let n = width as usize * height as usize;
    let mut pixels = vec![0u8; n * 4];
    for i in 0..n {
        pixels[i * 4] = seed.wrapping_add((i % 256) as u8);
        pixels[i * 4 + 1] = seed.wrapping_add(((i / 256) % 256) as u8);
        pixels[i * 4 + 2] = 128;
        pixels[i * 4 + 3] = 255;
    }
    pixels
}

/// Benchmark: HDC change detection at 1080p (510 tiles).
/// Target: < 65μs (claimed), must be < 1ms for viability.
fn bench_change_detection_1080p() {
    let width = 1920u32;
    let height = 1080u32;
    let mut codec = HybridCodec::new(width, height, 42);

    let frame1 = make_frame(width, height, 0);
    let frame2 = make_frame(width, height, 1); // Slightly different

    // Warm up: establish baseline.
    codec.detect_changes(&frame1, width, height);

    // Measure change detection.
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        let (_changed, _sims) = codec.detect_changes(&frame2, width, height);
    }
    let elapsed = start.elapsed();
    let per_frame_us = elapsed.as_micros() as f64 / iterations as f64;

    let tile_count = codec.tile_cols as usize * codec.tile_rows as usize;
    println!(
        "HDC change detection ({}x{}, {} tiles): {:.1}μs/frame ({:.1}μs/tile)",
        width,
        height,
        tile_count,
        per_frame_us,
        per_frame_us / tile_count as f64,
    );

    assert!(
        per_frame_us < 50_000.0, // 50ms hard limit
        "Change detection too slow: {:.1}μs (must be < 50ms for 20Hz)",
        per_frame_us,
    );
}

/// Benchmark: Full frame encoding at 1080p.
fn bench_full_frame_encode_1080p() {
    let width = 1920u32;
    let height = 1080u32;
    let codec = HybridCodec::new(width, height, 42);
    let frame = make_frame(width, height, 0);

    let iterations = 10;
    let start = Instant::now();
    for i in 0..iterations {
        let _full = codec.encode_full_frame(
            &frame,
            width,
            height,
            i as u64,
            i as u64 * 200,
            0.5,
            "present",
        );
    }
    let elapsed = start.elapsed();
    let per_frame_ms = elapsed.as_millis() as f64 / iterations as f64;

    println!(
        "Full frame encode ({}x{}): {:.1}ms/frame",
        width, height, per_frame_ms,
    );
}

/// Benchmark: Delta frame encoding (sparse: 10% changed tiles).
fn bench_delta_frame_encode() {
    let width = 1920u32;
    let height = 1080u32;
    let mut codec = HybridCodec::new(width, height, 42);

    let frame1 = make_frame(width, height, 0);
    codec.detect_changes(&frame1, width, height); // Baseline

    // Create frame with ~10% changes (modify a rectangular region).
    let mut frame2 = frame1.clone();
    for y in 0..108 {
        // 10% of 1080
        for x in 0..1920 {
            let offset = (y * 1920 + x) * 4;
            frame2[offset] = 200;
        }
    }

    let iterations = 50;
    let start = Instant::now();
    for i in 0..iterations {
        let (changed, sims) = codec.detect_changes(&frame2, width, height);
        let _delta = codec.encode_delta_frame(
            &frame2,
            width,
            height,
            &changed,
            &sims,
            i as u64 + 1,
            i as u64,
            i as u64 * 200,
            0.5,
        );
    }
    let elapsed = start.elapsed();
    let per_frame_ms = elapsed.as_millis() as f64 / iterations as f64;

    println!(
        "Delta frame (detect + encode, {}x{}, ~10% change): {:.1}ms/frame",
        width, height, per_frame_ms,
    );
}

/// Benchmark: TestCapture latency.
fn bench_test_capture() {
    let mut capture = TestCapture::new(1920, 1080);

    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        let _frame = capture.capture();
    }
    let elapsed = start.elapsed();
    let per_frame_ms = elapsed.as_millis() as f64 / iterations as f64;

    println!("TestCapture (1920x1080): {:.1}ms/frame", per_frame_ms,);
}

/// Benchmark: X11 capture latency (only with rdp-x11 feature).
#[cfg(feature = "rdp-x11")]
fn bench_x11_capture() {
    use symthaea::swarm::rdp_capture::X11ShmCapture;

    match X11ShmCapture::new() {
        Ok(mut capture) => {
            let iterations = 50;
            let start = Instant::now();
            for _ in 0..iterations {
                let _frame = capture.capture();
            }
            let elapsed = start.elapsed();
            let per_frame_ms = elapsed.as_millis() as f64 / iterations as f64;

            println!(
                "X11 SHM capture ({}x{}): {:.1}ms/frame (target: <20ms)",
                capture.width(),
                capture.height(),
                per_frame_ms,
            );

            assert!(
                per_frame_ms < 20.0,
                "X11 capture {:.1}ms exceeds 20ms budget",
                per_frame_ms,
            );
        }
        Err(e) => {
            println!("X11 capture not available: {} (skipping benchmark)", e);
        }
    }
}

fn main() {
    println!("═══════════════════════════════════════════════════");
    println!("  Sovereign RDP Codec Benchmarks");
    println!("═══════════════════════════════════════════════════\n");

    bench_test_capture();
    bench_change_detection_1080p();
    bench_full_frame_encode_1080p();
    bench_delta_frame_encode();

    #[cfg(feature = "rdp-x11")]
    bench_x11_capture();

    println!("\n═══════════════════════════════════════════════════");
    println!("  Benchmarks complete.");
    println!("═══════════════════════════════════════════════════");
}
