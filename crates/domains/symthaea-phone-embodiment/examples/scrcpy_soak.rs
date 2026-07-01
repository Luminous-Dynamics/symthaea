// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phase I.B.6 sustain + soak harness.
//!
//! Drives a live `ScrcpyCaptureStream` for a configurable duration,
//! pulling frames as fast as the device will produce them, and emits the
//! v1.1 observability metrics every `report_interval` seconds plus a
//! final summary table.
//!
//! # Usage
//!
//! Sustain (default 60 s):
//!
//! ```text
//! cd symthaea
//! nix develop --command cargo run --release --example scrcpy_soak \
//!     --features scrcpy -- 41201FDJG000UM 60 720 10
//! ```
//!
//! 10-minute soak:
//!
//! ```text
//! nix develop --command cargo run --release --example scrcpy_soak \
//!     --features scrcpy -- 41201FDJG000UM 600 720 30
//! ```
//!
//! Args (positional, all optional):
//! 1. device serial (default: 41201FDJG000UM — Pixel 8 Pro)
//! 2. duration in seconds (default: 60)
//! 3. max_size for the encoder (default: 720; use 0 to omit and let
//!    the encoder pick native)
//! 4. report interval seconds (default: 10)
//!
//! # What gets measured (per the v1.1 roadmap observability spec)
//!
//! Per-window (printed every `report_interval` seconds):
//! - frames decoded in the window + cumulative
//! - effective fps for the window
//! - wire bytes in the window + cumulative + bytes/sec
//! - decode latency p50 / p95 / p99 in microseconds
//! - dropped count (next_frame returned Ok(None), i.e. read timeout)
//! - timeouts since last report
//!
//! Final summary:
//! - total frames, total duration, mean fps
//! - total wire bytes, mean bytes/sec
//! - global p50 / p95 / p99 decode latency
//! - peak per-frame wire size, smallest per-frame wire size
//! - sustained vs target fps (>= target ⇒ "PASS")
//!
//! # What this harness does NOT do (yet)
//!
//! - Memory growth detection (Phase I.B.7 task — needs procfs read)
//! - scrcpy-server restart counting (no restart logic in the current
//!   ScrcpyCaptureStream — chaos test in I.B.7 will exercise this)
//! - Replay-reject counting (we're not running the AEAD seal layer here)
//! - Broadcast queue depth (this harness is single-consumer)

use std::env;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use symthaea_phone_embodiment::scrcpy::stream::ScrcpyCaptureStream;
use symthaea_phone_embodiment::scrcpy::{ScrcpyOptions, VENDORED_JAR_NAME};

/// Sustain target: 30 fps is the v1.1 roadmap spec for the cognitive loop.
const TARGET_FPS: f64 = 30.0;

/// Acceptable margin below target — anything above 0.85 × target is "PASS".
const SUSTAIN_PASS_FRACTION: f64 = 0.85;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let serial = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "41201FDJG000UM".to_string());
    let duration_secs: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(60);
    let max_size: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(720);
    let report_interval: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(10);

    let crate_dir = env!("CARGO_MANIFEST_DIR");
    let jar = PathBuf::from(crate_dir)
        .join("vendor")
        .join(VENDORED_JAR_NAME);

    println!("=== Phase I.B.6 sustain/soak harness ===");
    println!("device serial   : {serial}");
    println!("duration        : {duration_secs} s");
    println!("max_size        : {max_size}");
    println!("report interval : {report_interval} s");
    println!("vendored JAR    : {}", jar.display());
    println!("target fps      : {TARGET_FPS}");
    println!(
        "sustain pass at : {} fps ({}% of target)",
        TARGET_FPS * SUSTAIN_PASS_FRACTION,
        (SUSTAIN_PASS_FRACTION * 100.0) as u32
    );
    println!();

    let mut opts = ScrcpyOptions::cybernetic_defaults(serial.clone(), 8401);
    if max_size > 0 {
        opts.extra_args.push(format!("max_size={max_size}"));
    }
    // Force a 1-second iframe interval. Without this, scrcpy v2.4
    // defaults to KEY_I_FRAME_INTERVAL = 15 seconds, which means a
    // static screen produces a burst of frames at each keyframe and
    // then silence for 15 seconds. The Phase I.B.6 sustain run caught
    // this on the first clean test.
    opts.extra_args
        .push("video_codec_options=i-frame-interval=1".to_string());

    println!("Launching scrcpy capture stream...");
    let mut stream = ScrcpyCaptureStream::launch(&jar, &opts)?;
    println!("  device       : {}", stream.device_meta().name);
    println!(
        "  encoder size : {}x{} ({:?})",
        stream.video_header().width,
        stream.video_header().height,
        stream.video_header().codec,
    );
    println!();

    // Cumulative state.
    let started_at = Instant::now();
    let mut total_frames: u64 = 0;
    let mut total_timeouts: u64 = 0;
    let mut all_latencies_us: Vec<u32> = Vec::with_capacity(duration_secs as usize * 60);
    let mut peak_wire_packet: u64 = 0;
    let mut min_wire_packet: u64 = u64::MAX;
    // Tracks the per-frame previous cumulative for peak/min computation.
    let mut prev_frame_wire_bytes_total: u64 = 0;
    let mut prev_frame_wire_packets_total: u64 = 0;
    // Tracks the cumulative AT THE LAST WINDOW REPORT, used to compute
    // per-window deltas. Distinct from the per-frame tracker above.
    let mut last_window_wire_bytes_total: u64 = 0;

    // Per-window state (reset every report_interval).
    let mut window_started_at = started_at;
    let mut window_frames: u64 = 0;
    let mut window_timeouts: u64 = 0;
    let mut window_latencies_us: Vec<u32> = Vec::new();
    let report_interval_dur = Duration::from_secs(report_interval);

    // Print the column header so log parsers can find numbers.
    // wire_KB / wire_KB/s come from the HEVC payload byte counter on
    // ScrcpyCaptureStream — distinct from the (constant-size) RGBA
    // output buffers.
    println!("    elapsed  frames  fps     wire_KB  wire_KB/s  p50us  p95us  p99us  drops");
    println!("    -------  ------  ------  -------  ---------  -----  -----  -----  -----");

    let target = Duration::from_secs(duration_secs);
    while started_at.elapsed() < target {
        let frame_start = Instant::now();
        match stream.next_frame() {
            Ok(Some(_frame)) => {
                let latency = frame_start.elapsed();
                let latency_us = latency.as_micros().min(u32::MAX as u128) as u32;
                total_frames += 1;
                all_latencies_us.push(latency_us);
                window_frames += 1;
                window_latencies_us.push(latency_us);

                // Track per-packet wire size deltas (the new wire-bytes
                // counter on the stream tracks cumulative HEVC payload
                // bytes, distinct from the fixed-size RGBA outputs).
                let cur_packets = stream.wire_packets_total();
                let cur_bytes = stream.wire_bytes_total();
                if cur_packets > prev_frame_wire_packets_total {
                    let new_bytes = cur_bytes - prev_frame_wire_bytes_total;
                    let new_packets = cur_packets - prev_frame_wire_packets_total;
                    let avg_packet = new_bytes / new_packets.max(1);
                    peak_wire_packet = peak_wire_packet.max(avg_packet);
                    if avg_packet > 0 {
                        min_wire_packet = min_wire_packet.min(avg_packet);
                    }
                }
                prev_frame_wire_bytes_total = cur_bytes;
                prev_frame_wire_packets_total = cur_packets;
            }
            Ok(None) => {
                total_timeouts += 1;
                window_timeouts += 1;
            }
            Err(e) => {
                eprintln!("Stream error after {} frames: {e}", total_frames);
                break;
            }
        }

        // Report on schedule.
        if window_started_at.elapsed() >= report_interval_dur {
            let cur_total_wire = stream.wire_bytes_total();
            print_window_row(
                started_at.elapsed(),
                window_frames,
                cur_total_wire,
                window_timeouts,
                &mut window_latencies_us,
                window_started_at.elapsed(),
                last_window_wire_bytes_total,
            );
            std::io::stdout().flush().ok();

            last_window_wire_bytes_total = cur_total_wire;
            window_started_at = Instant::now();
            window_frames = 0;
            window_timeouts = 0;
            window_latencies_us.clear();
        }
    }

    // Final window if any frames were captured after the last report tick.
    if window_frames > 0 || window_timeouts > 0 {
        let cur_total_wire = stream.wire_bytes_total();
        print_window_row(
            started_at.elapsed(),
            window_frames,
            cur_total_wire,
            window_timeouts,
            &mut window_latencies_us,
            window_started_at.elapsed(),
            last_window_wire_bytes_total,
        );
    }
    // Capture the FINAL totals from the stream before drop.
    let total_wire_bytes = stream.wire_bytes_total();
    let total_wire_packets = stream.wire_packets_total();

    let total_elapsed = started_at.elapsed();
    let mean_fps = total_frames as f64 / total_elapsed.as_secs_f64();
    let mean_bytes_per_sec = total_wire_bytes as f64 / total_elapsed.as_secs_f64();

    all_latencies_us.sort_unstable();
    let global_p50 = pct(&all_latencies_us, 50.0);
    let global_p95 = pct(&all_latencies_us, 95.0);
    let global_p99 = pct(&all_latencies_us, 99.0);

    let sustain_threshold = TARGET_FPS * SUSTAIN_PASS_FRACTION;
    let sustain_verdict = if mean_fps >= sustain_threshold {
        "PASS"
    } else {
        "FAIL"
    };

    println!();
    println!("=== Phase I.B.6 final summary ===");
    println!(
        "duration             : {:.2} s",
        total_elapsed.as_secs_f64()
    );
    println!("decoded frames       : {total_frames}");
    println!("mean fps             : {mean_fps:.2}");
    println!("target fps           : {TARGET_FPS:.0}");
    println!(
        "sustain threshold    : {sustain_threshold:.1} fps ({}% of target)",
        (SUSTAIN_PASS_FRACTION * 100.0) as u32
    );
    println!("sustain verdict      : {sustain_verdict}");
    println!();
    println!("wire packets read    : {total_wire_packets}");
    println!(
        "wire (HEVC) bytes    : {total_wire_bytes} ({} KB)",
        total_wire_bytes / 1024
    );
    println!(
        "mean wire throughput : {:.1} KB/s",
        mean_bytes_per_sec / 1024.0
    );
    println!("peak wire pkt bytes  : {peak_wire_packet}");
    println!(
        "min wire pkt bytes   : {}",
        if min_wire_packet == u64::MAX {
            0
        } else {
            min_wire_packet
        }
    );
    println!();
    println!("decode p50           : {global_p50} us");
    println!("decode p95           : {global_p95} us");
    println!("decode p99           : {global_p99} us");
    println!();
    println!("read timeouts        : {total_timeouts}");

    drop(stream);
    println!("\nStream torn down. Done.");

    // Exit code reflects sustain verdict so CI / scripts can grep it.
    if sustain_verdict == "PASS" {
        Ok(())
    } else {
        Err(format!("sustain FAIL: {mean_fps:.2} fps < {sustain_threshold:.1} fps").into())
    }
}

fn print_window_row(
    elapsed: Duration,
    frames: u64,
    cumulative_wire_bytes: u64,
    timeouts: u64,
    window_latencies_us: &mut Vec<u32>,
    window_dur: Duration,
    prev_cumulative_wire_bytes: u64,
) {
    let fps = if window_dur.as_secs_f64() > 0.0 {
        frames as f64 / window_dur.as_secs_f64()
    } else {
        0.0
    };
    let window_bytes = cumulative_wire_bytes.saturating_sub(prev_cumulative_wire_bytes);
    let kb = window_bytes as f64 / 1024.0;
    let kbps = if window_dur.as_secs_f64() > 0.0 {
        kb / window_dur.as_secs_f64()
    } else {
        0.0
    };
    window_latencies_us.sort_unstable();
    let p50 = pct(window_latencies_us, 50.0);
    let p95 = pct(window_latencies_us, 95.0);
    let p99 = pct(window_latencies_us, 99.0);
    println!(
        "    {:>7.1}s  {:>6}  {:>6.2}  {:>7.1}  {:>9.1}  {:>5}  {:>5}  {:>5}  {:>5}",
        elapsed.as_secs_f64(),
        frames,
        fps,
        kb,
        kbps,
        p50,
        p95,
        p99,
        timeouts,
    );
}

/// Percentile from a sorted slice. Returns 0 for empty input.
fn pct(sorted: &[u32], q: f64) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((q / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
