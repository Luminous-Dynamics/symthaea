// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Hold-out evaluation for distilled Broca (#1 of the
//! "make this even better" list).
//!
//! Reads the same distillation JSONL the trainer consumes, picks out
//! entries flagged `holdout: true`, regenerates for each held-out
//! prompt, and emits `{prompt, golden, generated}` JSONL for the
//! main-crate scorer to consume.
//!
//! Why this is the gating test: the trainer memorized 26 pairs down
//! to loss 0.39. We don't yet know whether that's generalization
//! (model learned the Nix distribution) or memorization (model
//! learned these 26 prompts). Holding out 6 pairs, training on 20,
//! and scoring generation on the 6 answers the question. If ≥1/6
//! held-out prompt passes the structural scorer, there's evidence of
//! generalization. If 0/6, we know the 20-pair training overfit
//! entirely and need more data / different approach.
//!
//! Two-phase split because broca's `ssm_language` feature conflicts
//! with the main crate's `broca_lite` default. This bin emits raw
//! generation results; a main-crate example (`nix_holdout_score`)
//! reads them + runs the AST scorer.
//!
//! Usage:
//!   cargo run --bin distill_nix_evaluate -p symthaea-broca -- \
//!       --checkpoint /tmp/broca-nix-aligned-25ep.mpk \
//!       --in /tmp/distill-aligned-split.jsonl \
//!       --out /tmp/holdout-generated.jsonl

use std::io::Write;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use symthaea_broca::encoder::ThoughtChannels;
use symthaea_broca::generator::{BrocaGenerator, SamplingStrategy};
use symthaea_core::genesis::GenesisSeed;

/// Matches the harvester's DistillPair — only fields we need.
#[derive(Debug, Deserialize)]
struct HarvestPair {
    prompt: String,
    intent: String,
    channels: Vec<f32>,
    code: String,
    #[serde(default)]
    holdout: bool,
}

/// What we write out for each held-out prompt. Main-crate scorer
/// reads this.
#[derive(Debug, Serialize)]
struct EvalRow {
    prompt: String,
    intent: String,
    /// Hand-written golden Nix for this prompt.
    golden: String,
    /// What Broca emitted given the same channels.
    generated: String,
    /// Byte length of generation — simple sanity signal.
    generated_bytes: usize,
}

fn parse_path(flag: &str, default: PathBuf) -> PathBuf {
    let args: Vec<String> = std::env::args().collect();
    for w in args.windows(2) {
        if w[0] == flag {
            return PathBuf::from(&w[1]);
        }
    }
    default
}

fn home_path() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string()))
}

fn pack_channels(channels: &[f32]) -> ThoughtChannels {
    let mut tc = ThoughtChannels::default();
    // Zero out Broca's default-intent bit (position 7 defaults to
    // 1.0 for Unknown). Same reason `distill_nix_generate` does this.
    for i in 0..8 {
        tc.channels[i] = 0.0;
    }
    let n = tc.channels.len().min(channels.len());
    tc.channels[..n].copy_from_slice(&channels[..n]);
    tc
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint = parse_path(
        "--checkpoint",
        home_path()
            .join(".cache")
            .join("symthaea")
            .join("broca-nix-distilled.mpk"),
    );
    let in_path = parse_path(
        "--in",
        home_path()
            .join(".cache")
            .join("symthaea")
            .join("distillation-pairs.jsonl"),
    );
    let out_path = parse_path(
        "--out",
        home_path()
            .join(".cache")
            .join("symthaea")
            .join("holdout-generated.jsonl"),
    );

    println!("┌─────────────────────────────────────────────────────────");
    println!("│ Hold-out Evaluation (#1)");
    println!("│ Checkpoint: {}", checkpoint.display());
    println!("│ Input:      {}", in_path.display());
    println!("│ Output:     {}", out_path.display());
    println!("└─────────────────────────────────────────────────────────");

    // ── Load the checkpoint ──
    // Genesis phrase must match the trainer's. `-m7c` is the Nix-only
    // tokenizer era (2026-04-19, for_nix_distillation). Older `-m7b`
    // checkpoints use default_minimal + add_nix_tokens with a different
    // vocabulary and can't be loaded here.
    let genesis = GenesisSeed::from_phrase("symthaea-nix-distillation-m7c");
    let (mut generator, _adam, _proj, _lm) = BrocaGenerator::from_checkpoint(&checkpoint, &genesis)
        .map_err(|e| format!("from_checkpoint: {}", e))?;
    // Match distill_nix_generate's sampling so the comparison is
    // apples-to-apples with the interactive demo.
    generator.set_sampling(SamplingStrategy::TopK {
        k: 20,
        temperature: 0.7,
    });
    generator.config_mut().repetition_penalty = 1.3;
    println!(
        "Loaded. Vocab: {} tokens. Sampling: TopK(20, 0.7), rep_penalty=1.3.",
        generator.tokenizer().vocab_size()
    );

    // ── Load the distillation JSONL and filter to holdouts ──
    let text = std::fs::read_to_string(&in_path)
        .map_err(|e| format!("reading {}: {}", in_path.display(), e))?;
    let mut holdouts: Vec<HarvestPair> = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let pair: HarvestPair =
            serde_json::from_str(line).map_err(|e| format!("parsing line {}: {}", i + 1, e))?;
        if pair.holdout {
            holdouts.push(pair);
        }
    }
    if holdouts.is_empty() {
        eprintln!("✗ No holdout entries found in {}.", in_path.display());
        eprintln!("  Re-run the harvester with `--holdout N` (e.g. --holdout 6).");
        std::process::exit(1);
    }
    println!("Found {} holdout prompt(s).", holdouts.len());

    // ── Generate + write ──
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(&out_path)?;
    let mut writer = std::io::BufWriter::new(file);

    for (i, pair) in holdouts.iter().enumerate() {
        let tc = pack_channels(&pair.channels);
        let result = generator.generate(&tc);
        let row = EvalRow {
            prompt: pair.prompt.clone(),
            intent: pair.intent.clone(),
            golden: pair.code.clone(),
            generated: result.text.clone(),
            generated_bytes: result.text.len(),
        };
        writeln!(writer, "{}", serde_json::to_string(&row)?)?;
        println!(
            "  [{}/{}] {} → {} bytes",
            i + 1,
            holdouts.len(),
            pair.prompt,
            result.text.len()
        );
    }
    writer.flush()?;

    println!(
        "\n✓ {} holdout generations written → {}",
        holdouts.len(),
        out_path.display()
    );
    println!("\nNext: run the main-crate scorer example");
    println!("  cargo run --features code_generation \\");
    println!(
        "      --example nix_holdout_score -- --in {}",
        out_path.display()
    );

    Ok(())
}
