// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Symthaea Background Distillation & Convergence Engine
//!
//! Monitored target: data/distillation_flywheel.jsonl
//! Accumulates, shuffles, and optimizes time-constant weights natively.

use rand::seq::SliceRandom;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::Command;
use std::time::Duration;

const DATA_PATH: &str = "data/distillation_flywheel.jsonl";
const RETRAIN_THRESHOLD: usize = 50;
const CHECK_INTERVAL_SECS: u64 = 10;

fn count_lines<P: AsRef<Path>>(path: P) -> usize {
    File::open(path)
        .map(|f| BufReader::new(f).lines().count())
        .unwrap_or(0)
}

fn interleave_and_shuffle_dataset<P: AsRef<Path> + Clone>(path: P) -> anyhow::Result<()> {
    println!(
        "[Flywheel] Conditioning and interleaving data layers to prevent subspace collapse..."
    );
    let file = File::open(path.clone())?;
    let reader = BufReader::new(file);
    let mut records: Vec<String> = reader.lines().filter_map(|l| l.ok()).collect();

    // Native in-place slice shuffling via rand thread-rng
    let mut rng = rand::thread_rng();
    records.shuffle(&mut rng);

    let mut file = OpenOptions::new().write(true).truncate(true).open(path)?;

    for record in records {
        writeln!(file, "{}", record)?;
    }
    println!("[Flywheel] Interleaved optimization formatting successfully applied.");
    Ok(())
}

fn trigger_background_retrain() {
    println!("\n[Flywheel] 🚀 Retrain threshold breached. Spawning background compiler...");

    // Spawn the training pass natively using a low-priority thread mask via 'nice'
    let mut child = Command::new("nice")
        .args([
            "-n",
            "19",
            "cargo",
            "test",
            "--release",
            "--lib",
            "language::algorithm_training::tests::test_cfc_training_converges",
            "--",
            "--nocapture",
        ])
        .env("CARGO_TARGET_DIR", "/tmp/symthaea-broca-host-release")
        .env("RUSTC_WRAPPER", "")
        .env("SCCACHE_DISABLE", "1")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("[Flywheel] Failed to initialize offline training thread configuration.");

    println!(
        "[Flywheel] Optimization pass spawned successfully under PID {}. Weights evolving offline.",
        child.id()
    );
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("===============================================================================");
    print!("   Symthaea Distillation & Convergence Daemon Initialized (Rust-Native 2026)   ");
    println!("===============================================================================");
    println!("Monitoring target: {}", DATA_PATH);
    println!(
        "Trigger window: Accumulation of every {} verified turns",
        RETRAIN_THRESHOLD
    );

    let mut last_count = count_lines(DATA_PATH);
    println!(
        "Initial baseline state: Dataset currently holds {} examples.",
        last_count
    );

    let mut interval = tokio::time::interval(Duration::from_secs(CHECK_INTERVAL_SECS));

    loop {
        interval.tick().await;
        let current_count = count_lines(DATA_PATH);

        if current_count >= last_count + RETRAIN_THRESHOLD {
            println!(
                "\n[Flywheel] Accumulation window closed ({} -> {}).",
                last_count, current_count
            );
            if let Err(e) = interleave_and_shuffle_dataset(DATA_PATH) {
                eprintln!("[Flywheel] Data conditioning failure: {}", e);
            } else {
                trigger_background_retrain();
            }
            last_count = current_count;
        }
    }
}
