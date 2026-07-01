// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Phase 2 M7.b — distillation trainer adapter.
//!
//! Loads harvester JSONL pairs (produced by the main crate's
//! `harvest_nix_distillation` example) and runs them through Broca's
//! existing training loop (`training::train`).
//!
//! The adapter's only job: bridge the harvester's 17-channel Nix intent
//! format into Broca's `TrainingPair { channels: Vec<f32>, target_text,
//! target_ids }`. Broca pads channels to `NUM_CHANNELS` (43 by default)
//! automatically, so the 17D intent vector lands at positions 0..17 and
//! the remaining 26 channels default to zero.
//!
//! Usage:
//!   cargo run --bin distill_nix_train                 # default
//!   cargo run --bin distill_nix_train -- --in  <path> # override input
//!   cargo run --bin distill_nix_train -- --out <path> # override output checkpoint
//!   cargo run --bin distill_nix_train -- --epochs 10  # more epochs
//!
//! Default paths:
//!   input:  ~/.cache/symthaea/distillation-pairs.jsonl
//!   output: ~/.cache/symthaea/broca-nix-distilled.mpk
//!
//! This smoke-tests the pipeline end-to-end. For real training runs
//! (large datasets, many epochs), invoke as a long-running process.

use std::path::PathBuf;

use serde::Deserialize;
use symthaea_broca::generator::{BrocaConfig, BrocaGenerator};
use symthaea_broca::tokenizer::BpeTokenizer;
use symthaea_broca::training::{TrainingConfig, TrainingDataset, TrainingPair, train};
use symthaea_core::genesis::GenesisSeed;

/// Shape of one line from the harvester output. Must stay in lockstep
/// with `examples/harvest_nix_distillation.rs::DistillPair` in the main
/// crate.
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct HarvestPair {
    substrate: String,
    prompt: String,
    channels: Vec<f32>,
    code: String,
    #[serde(default)]
    iterations: usize,
    #[serde(default)]
    repair_steps: usize,
    /// When true, this pair is a holdout reserved for generalization
    /// evaluation — the trainer silently skips it.
    #[serde(default)]
    holdout: bool,
}

fn default_in_path() -> PathBuf {
    home_path()
        .join(".cache")
        .join("symthaea")
        .join("distillation-pairs.jsonl")
}

fn default_out_path() -> PathBuf {
    home_path()
        .join(".cache")
        .join("symthaea")
        .join("broca-nix-distilled.mpk")
}

fn home_path() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string()))
}

fn parse_flag_path(flag: &str, default: PathBuf) -> PathBuf {
    let args: Vec<String> = std::env::args().collect();
    for w in args.windows(2) {
        if w[0] == flag {
            return PathBuf::from(&w[1]);
        }
    }
    default
}

fn parse_epochs() -> usize {
    let args: Vec<String> = std::env::args().collect();
    for w in args.windows(2) {
        if w[0] == "--epochs" {
            if let Ok(n) = w[1].parse::<usize>() {
                return n;
            }
        }
    }
    1 // smoke-test default
}

fn load_harvest_jsonl(path: &std::path::Path) -> Result<Vec<HarvestPair>, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("reading {}: {}", path.display(), e))?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let pair: HarvestPair =
            serde_json::from_str(line).map_err(|e| format!("parsing line {}: {}", i + 1, e))?;
        out.push(pair);
    }
    Ok(out)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let in_path = parse_flag_path("--in", default_in_path());
    let out_path = parse_flag_path("--out", default_out_path());
    let epochs = parse_epochs();

    println!("┌─────────────────────────────────────────────────────────");
    println!("│ Broca Nix Distillation Trainer (Phase 2 M7.b)");
    println!("│ Input:  {}", in_path.display());
    println!("│ Output: {}", out_path.display());
    println!("│ Epochs: {}", epochs);
    println!("└─────────────────────────────────────────────────────────");

    // ── Load harvester pairs ──
    let harvest = load_harvest_jsonl(&in_path)?;
    if harvest.is_empty() {
        eprintln!("✗ No training pairs found at {}.", in_path.display());
        eprintln!(
            "  Run `cargo run --features code_generation \\\n  \
             --example harvest_nix_distillation` in the main crate first."
        );
        std::process::exit(1);
    }
    println!("Loaded {} harvested pairs.", harvest.len());

    // Stratify by whether repair was needed — interesting telemetry.
    let (initial, repaired): (Vec<_>, Vec<_>) = harvest.iter().partition(|p| p.repair_steps == 0);
    println!(
        "  · {} initial-PASS pairs (no repair)\n  · {} repair-PASS pairs",
        initial.len(),
        repaired.len()
    );

    // Filter out holdout pairs — reserved for generalization eval.
    let holdout_count = harvest.iter().filter(|p| p.holdout).count();
    let harvest: Vec<HarvestPair> = harvest.into_iter().filter(|p| !p.holdout).collect();
    if holdout_count > 0 {
        println!(
            "  · {} holdout pairs skipped (training on {} only)",
            holdout_count,
            harvest.len()
        );
    }

    // ── Adapt to Broca's TrainingPair format ──
    // The harvester emits 17-channel intent vectors; Broca auto-pads to
    // NUM_CHANNELS at TrainingPair::to_thought_channels time, so no
    // reshaping is needed here. target_text = the Nix code (the output
    // Broca should learn to emit given those channels). target_ids left
    // empty — populated by dataset.tokenize_all after vocabulary setup.
    let pairs: Vec<TrainingPair> = harvest
        .iter()
        .map(|hp| TrainingPair {
            channels: hp.channels.clone(),
            target_text: hp.code.clone(),
            target_ids: Vec::new(),
            valence: 0.0,
            arousal: 0.5,
        })
        .collect();

    let mut dataset = TrainingDataset { pairs };

    // ── Construct Broca with the Nix-augmented vocabulary ──
    // Use `BrocaGenerator::with_tokenizer` (the same path
    // `from_checkpoint` uses) so the controller is sized to the
    // post-augmentation vocab. `add_nix_tokens()` grows the BPE
    // vocabulary with M5's NIX_TOKENS list: Nix keywords, attrpath
    // roots (`services.`, `hardware.`, etc.), call-site identifiers
    // (`mkShell`, `mkOption`), common idioms. Each becomes a single
    // token rather than a char-by-char BPE reconstruction — major
    // quality + speed win at training time.
    let genesis = GenesisSeed::from_phrase("symthaea-nix-distillation-m7d");
    // Use the Nix-only tokenizer (no Rust/Python keywords). The prior
    // tokenizer (default_minimal + add_nix_tokens) left CODE_TOKENS in
    // place, biasing emission toward Rust gibberish (`impl Into#[...`)
    // — see 2026-04-19 BENCHMARK entry. `for_nix_distillation` builds
    // the same vocab MINUS those tokens. Genesis phrase bumped to
    // -m7c so the new checkpoints can't be confused with m7b ones
    // trained against a different vocabulary.
    let tokenizer = BpeTokenizer::for_nix_distillation();
    println!(
        "Built Nix-only vocab (CODE_TOKENS excluded, NIX_TOKENS included): {} tokens.",
        tokenizer.vocab_size()
    );
    let mut generator = BrocaGenerator::with_tokenizer(&genesis, BrocaConfig::default(), tokenizer);
    println!(
        "Built BrocaGenerator. Controller sized to {} vocab entries.",
        generator.tokenizer().vocab_size()
    );

    dataset.tokenize_all(generator.tokenizer());
    println!("Tokenized {} pairs.", dataset.len());

    // ── Training config ──
    // Smoke-test defaults: fast, easy-to-diagnose. Real runs should
    // bump epochs to ~50-200 and reconsider learning rate.
    let config = TrainingConfig {
        epochs,
        learning_rate: 0.02,
        bptt_window: 32,
        grad_clip: 1.0,
        report_interval: 5,
        use_adam: true,
        warmup_fraction: 0.1,
        patience: 0, // disabled — smoke test
        enable_diagnostics: false,
        train_network: true,
        network_lr_scale: 0.2,
        embedding_target_norm: 0.0,
        ..Default::default()
    };

    println!(
        "\nTraining {} pairs for {} epoch(s)…",
        dataset.len(),
        epochs
    );
    let metrics = train(&mut generator, &dataset, &config);
    println!("\nTraining complete. Per-epoch metrics:");
    for (i, m) in metrics.iter().enumerate() {
        println!("  epoch {}: {:?}", i + 1, m);
    }

    // ── Save checkpoint ──
    // Checkpoint API takes granular fields rather than the full
    // TrainingConfig: (path, epoch, loss, adam_state, projection_weights,
    // liquid_mamba_config). Use None for fields the M7.b smoke path
    // doesn't populate.
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let final_epoch = metrics.len();
    // Pull the final epoch's average loss into the checkpoint metadata
    // so future reloads + retraining resume with a meaningful anchor.
    // Fall back to 0.0 if training produced zero epochs (shouldn't
    // happen — train() guarantees at least one pass).
    let final_loss = metrics.last().map(|m| m.avg_loss).unwrap_or(0.0);
    generator
        .save_checkpoint(&out_path, final_epoch, final_loss, None, None, None)
        .map_err(|e| format!("save_checkpoint: {}", e))?;
    println!("\n✓ Saved checkpoint → {}", out_path.display());
    println!("\nNext: load via `BrocaGenerator::from_checkpoint(...)` and generate");
    println!("      Nix from thought channels in consciousness-gated emission mode (M8).");

    Ok(())
}
