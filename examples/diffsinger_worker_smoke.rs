// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! One-phrase Rust -> JSONL worker -> adapter smoke test.

use std::path::PathBuf;

use anyhow::{Context, Result};
use symthaea::voice::diffsinger::DiffSingerEngine;
use symthaea::voice::singing_engine::{SingingVoiceEngine, VocalPerformance};
use symthaea_muse::Note;

fn main() -> Result<()> {
    let worker = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: diffsinger_worker_smoke PATH_TO_WORKER")?;
    let melody = [Note {
        frequency: 220.0,
        start_time: 0.0,
        duration: 0.2,
        velocity: 0.75,
    }];
    let performance = VocalPerformance::from_melody("light", &melody, "en")?;
    let mut engine = DiffSingerEngine::spawn(&worker)?;
    let stem = engine.render(&performance)?;
    anyhow::ensure!(stem.sample_rate == 24_000, "unexpected fixture sample rate");
    anyhow::ensure!(!stem.samples.is_empty(), "worker returned empty audio");
    println!(
        "Rust -> worker -> adapter smoke: PASS ({} samples at {} Hz)",
        stem.samples.len(),
        stem.sample_rate
    );
    Ok(())
}
