// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Distill a full Broca checkpoint for Spore (WASM) deployment.
//!
//! The full v6-joint checkpoint is ~976MB (includes optimizer state, projection
//! weights, and Liquid-Mamba config). This tool strips everything unnecessary
//! for inference, optionally reduces vocabulary to the top-K most common tokens,
//! and produces a compact checkpoint suitable for browser delivery.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --bin distill-for-spore -- \
//!     --input data/broca-cfc-v6-joint.bin \
//!     --output data/broca-spore-distilled.bin \
//!     --vocab-size 2048
//! ```
//!
//! # What gets stripped
//!
//! - Adam optimizer state (`adam_state`) -> None
//! - Projection weights (`projection_weights`) -> None
//! - Liquid-Mamba config (`liquid_mamba_config`) -> None
//! - Vocabulary reduced to top-K tokens by frequency rank
//! - Token embeddings reduced to match vocabulary

use anyhow::{Context, Result};
use std::path::PathBuf;
use symthaea_broca::BrocaCheckpoint;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let mut input_path: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut target_vocab_size: Option<usize> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" | "-i" => {
                i += 1;
                input_path = Some(PathBuf::from(&args[i]));
            }
            "--output" | "-o" => {
                i += 1;
                output_path = Some(PathBuf::from(&args[i]));
            }
            "--vocab-size" | "-v" => {
                i += 1;
                target_vocab_size = Some(args[i].parse().context("--vocab-size must be a number")?);
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage: distill-for-spore --input <path> --output <path> [--vocab-size <n>]"
                );
                eprintln!();
                eprintln!("Distill a Broca checkpoint for WASM/Spore deployment.");
                eprintln!("Strips optimizer state and optionally reduces vocabulary.");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --input, -i       Path to full Broca checkpoint");
                eprintln!("  --output, -o      Path for distilled checkpoint");
                eprintln!("  --vocab-size, -v  Target vocabulary size (default: keep all)");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let input_path = input_path.context("--input is required")?;
    let output_path = output_path.context("--output is required")?;

    eprintln!("Loading checkpoint from: {}", input_path.display());
    let checkpoint = BrocaCheckpoint::load_from_file(&input_path)?;

    let original_vocab_size = checkpoint.token_embeddings.len();
    let original_merge_count = checkpoint.vocab.merges.len();
    eprintln!(
        "  Original: {} tokens, {} merges, epoch {}, loss {:.4}",
        original_vocab_size,
        original_merge_count,
        checkpoint.training_epoch,
        checkpoint.training_loss
    );
    eprintln!("  Has optimizer state: {}", checkpoint.adam_state.is_some());
    eprintln!(
        "  Has projection weights: {}",
        checkpoint.projection_weights.is_some()
    );
    eprintln!(
        "  Has Liquid-Mamba config: {}",
        checkpoint.liquid_mamba_config.is_some()
    );

    // Build distilled checkpoint
    let mut distilled = BrocaCheckpoint {
        version: checkpoint.version,
        token_embeddings: checkpoint.token_embeddings,
        network_state: checkpoint.network_state,
        vocab: checkpoint.vocab,
        config: checkpoint.config,
        training_epoch: checkpoint.training_epoch,
        training_loss: checkpoint.training_loss,
        // Strip training-only state
        adam_state: None,
        projection_weights: None,
        highway_weights: checkpoint.highway_weights,
        liquid_mamba_config: None,
        metadata: checkpoint.metadata,
        checksum: [0u8; 32],
    };

    // Optionally reduce vocabulary
    if let Some(target) = target_vocab_size {
        if target < original_vocab_size {
            eprintln!(
                "Reducing vocabulary: {} -> {} tokens",
                original_vocab_size, target
            );

            // Keep the first `target` tokens (BPE vocabularies are ordered by
            // frequency: byte-level tokens first, then most common merges).
            // Special tokens (BOS, EOS, UNK, PAD) are typically in the first
            // positions and will be preserved.
            distilled.token_embeddings.truncate(target);
            distilled.vocab.tokens.truncate(target);

            // Keep only merges that produce tokens within the vocabulary
            let retained: std::collections::HashSet<String> =
                distilled.vocab.tokens.iter().cloned().collect();
            distilled.vocab.merges.retain(|merge| {
                // A merge "a b" -> "ab" is only valid if "ab" is in our vocabulary
                let merged = format!("{}{}", merge.left, merge.right);
                retained.contains(&merged)
            });

            eprintln!(
                "  Retained {} merges (of {})",
                distilled.vocab.merges.len(),
                original_merge_count
            );

            // Update controller config to match
            distilled.config.controller.vocab_size = target;
        } else {
            eprintln!(
                "Target vocab size ({}) >= original ({}), keeping all",
                target, original_vocab_size
            );
        }
    }

    // Save distilled checkpoint
    eprintln!("Saving distilled checkpoint to: {}", output_path.display());
    distilled.save_to_file(&output_path)?;

    // Report size reduction
    let input_size = std::fs::metadata(&input_path)?.len();
    let output_size = std::fs::metadata(&output_path)?.len();
    let reduction = 100.0 * (1.0 - output_size as f64 / input_size as f64);
    eprintln!(
        "Size: {} MB -> {} MB ({:.1}% reduction)",
        input_size / (1024 * 1024),
        output_size / (1024 * 1024),
        reduction
    );

    eprintln!("Done.");
    Ok(())
}
