// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic CPU-cost probe for the exact Spore boot renderer.
//!
//! This is intentionally an evidence tool rather than a benchmark gate. CI uses
//! it to establish the cost of the complete organic + holographic + fidelity +
//! factual-identity stack before physical-host enablement. Hard performance
//! thresholds should only be introduced after representative 1080p/1440p data
//! exists on known hardware.

use std::cmp::Ordering;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use blake3::Hasher;
use symthaea_boot_ecology::{BootEcologyComposer, BootStateReceipt, MorphologyLineage};
use symthaea_quicken_fb::ecology_renderer::EcologyRenderer;

fn main() {
    if let Err(error) = run() {
        eprintln!("spore_render_probe: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let mut width = 640u32;
    let mut height = 360u32;
    let mut frames = 24u32;
    let mut output = PathBuf::from("spore-render-probe.json");

    let mut index = 1usize;
    while index < args.len() {
        match args[index].as_str() {
            "--width" => width = parse_number(&next_value(&args, &mut index, "--width")?, "width")?,
            "--height" => height = parse_number(&next_value(&args, &mut index, "--height")?, "height")?,
            "--frames" => frames = parse_number(&next_value(&args, &mut index, "--frames")?, "frames")?,
            "--out" => output = PathBuf::from(next_value(&args, &mut index, "--out")?),
            "--help" | "-h" => {
                eprintln!("spore_render_probe [--width N] [--height N] [--frames N] [--out FILE]");
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        index += 1;
    }

    if width == 0 || height == 0 {
        return Err("probe dimensions must be non-zero".to_string());
    }
    if frames < 2 {
        return Err("probe requires at least two frames".to_string());
    }

    let receipt = BootStateReceipt::first_boot([0x5a; 32]);
    let genome = BootEcologyComposer::compose(&receipt, &MorphologyLineage::default());
    let renderer = EcologyRenderer::new(width, height, genome);
    let duration_ms = renderer.total_duration_ms().max(1);
    let mut pixels = vec![0u32; width as usize * height as usize];

    // Warm the complete stack once so first-use allocation/cache noise is not
    // confused with steady rendering cost. Persistent renderer workspaces are
    // already allocated by construction.
    renderer.render_at(duration_ms / 3, &mut pixels);

    let mut samples_ms = Vec::with_capacity(frames as usize);
    let total_start = Instant::now();
    for frame in 0..frames {
        let elapsed_ms = ((duration_ms as u64 * frame as u64) / (frames - 1) as u64) as u32;
        let start = Instant::now();
        renderer.render_at(elapsed_ms, &mut pixels);
        samples_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
    }
    let wall_ms = total_start.elapsed().as_secs_f64() * 1_000.0;

    samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let sum_ms: f64 = samples_ms.iter().sum();
    let mean_ms = sum_ms / samples_ms.len() as f64;
    let p50_ms = percentile(&samples_ms, 0.50);
    let p95_ms = percentile(&samples_ms, 0.95);
    let max_ms = *samples_ms.last().unwrap_or(&0.0);

    let mut frame_hasher = Hasher::new();
    for pixel in &pixels {
        frame_hasher.update(&pixel.to_le_bytes());
    }

    let report = serde_json::json!({
        "schema": "spore-render-probe-v1",
        "renderer": "organic+holographic+fidelity+identity",
        "width": width,
        "height": height,
        "frames": frames,
        "sequence_duration_ms": duration_ms,
        "mean_frame_ms": mean_ms,
        "p50_frame_ms": p50_ms,
        "p95_frame_ms": p95_ms,
        "max_frame_ms": max_ms,
        "measured_wall_ms": wall_ms,
        "final_frame_blake3": frame_hasher.finalize().to_hex().to_string(),
        "policy": "evidence-only-no-performance-threshold"
    });

    fs::write(
        &output,
        serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("write {}: {error}", output.display()))?;

    println!("{}", serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?);
    Ok(())
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * quantile.clamp(0.0, 1.0)).round() as usize;
    sorted[index]
}

fn next_value(args: &[String], index: &mut usize, flag: &str) -> Result<String, String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_number<T>(value: &str, label: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse::<T>()
        .map_err(|error| format!("invalid {label} {value:?}: {error}"))
}
