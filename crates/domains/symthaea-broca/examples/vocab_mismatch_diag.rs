// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Diagnostic: does `BrocaGenerator::from_checkpoint` reconstruct the SAME
//! tokenizer vocabulary the checkpoint was actually trained with?
//!
//! `BrocaCheckpoint::token_embeddings` (one row per vocab token) and
//! `BrocaCheckpoint::vocab` (the actual `VocabFile` used at training/save
//! time) are both stored in the checkpoint file. But
//! `BrocaGenerator::from_checkpoint_struct` never reads `checkpoint.vocab`
//! -- it calls `Self::new(genesis, checkpoint.config)`, which always
//! constructs a tokenizer via `BpeTokenizer::default_minimal()` (208
//! tokens), regardless of what vocabulary was actually used to train the
//! embeddings being restored. `BrocaCheckpointMetadata::vocab_hash`/
//! `vocab_size` are computed and saved but never read back anywhere to
//! validate this. If the checkpoint was ever trained with a different
//! vocabulary (e.g. `default_4k`'s 4,096-token vocab, which the crate's
//! own docs recommend for production), every token ID the network
//! predicts would decode through the WRONG vocabulary -- producing real,
//! grammatical-looking words that are simply the wrong words, exactly the
//! "individual real tokens, nonsensical sequence" pattern seen in
//! production's current output.

use anyhow::{Context, Result};
use symthaea_broca::checkpoint::BrocaCheckpoint;
use symthaea_broca::generator::BrocaGenerator;
use symthaea_core::genesis::GenesisSeed;

fn main() -> Result<()> {
    let checkpoint_path = std::env::args().nth(1).unwrap_or_else(|| {
        "crates/domains/symthaea-broca/data/models/broca-checkpoint-latest.bin".to_string()
    });

    let raw = BrocaCheckpoint::load_from_file_allow_checksum_mismatch(&checkpoint_path)
        .with_context(|| format!("loading raw checkpoint {checkpoint_path}"))?;

    println!("=== raw checkpoint on disk ===");
    println!("token_embeddings.len() = {}", raw.token_embeddings.len());
    println!("vocab.tokens.len()     = {}", raw.vocab.tokens.len());
    println!("vocab.merges.len()     = {}", raw.vocab.merges.len());
    println!("metadata.vocab_size    = {}", raw.metadata.vocab_size);
    println!("metadata.embedding_dim = {}", raw.metadata.embedding_dim);
    println!("metadata.channel_count = {}", raw.metadata.channel_count);
    println!("training_epoch         = {}", raw.training_epoch);
    println!("training_loss          = {}", raw.training_loss);
    println!("n_layers               = {}", raw.network_state.n_layers());
    for i in 0..raw.network_state.n_layers() {
        if let Some(layer) = raw.network_state.layer(i) {
            println!("  layer[{i}].len() (neurons) = {}", layer.len());
        }
    }
    println!(
        "metadata.vocab_hash    = {}",
        raw.metadata
            .vocab_hash
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    );
    println!("first 20 vocab tokens (as stored in the checkpoint):");
    for (i, t) in raw.vocab.tokens.iter().take(20).enumerate() {
        println!("  [{i}] {t:?}");
    }

    println!("\n=== tokenizer BrocaGenerator::from_checkpoint actually builds ===");
    let genesis = GenesisSeed::from_phrase("broca-training-default");
    let (generator, _, _, _) =
        BrocaGenerator::from_checkpoint_allow_checksum_mismatch(&checkpoint_path, &genesis)
            .with_context(|| format!("loading via BrocaGenerator {checkpoint_path}"))?;
    let tok = generator.tokenizer();
    println!("tokenizer.vocab_size() = {}", tok.vocab_size());
    println!("first 20 tokens from the LIVE tokenizer (by id):");
    for id in 0..20u32.min(tok.vocab_size() as u32) {
        println!("  [{id}] {:?}", tok.token_str(id));
    }

    let live_embeddings_len = generator.controller().token_embeddings().len();
    println!("\nlive controller().token_embeddings().len() = {live_embeddings_len}");

    println!("\n=== comparison ===");
    println!(
        "checkpoint vocab size {} vs live tokenizer vocab size {} -> {}",
        raw.vocab.tokens.len(),
        tok.vocab_size(),
        if raw.vocab.tokens.len() == tok.vocab_size() {
            "MATCH"
        } else {
            "MISMATCH"
        }
    );
    let mut first_diff = None;
    for (i, stored) in raw.vocab.tokens.iter().enumerate() {
        if i as u32 >= tok.vocab_size() as u32 {
            break;
        }
        let live = tok.token_str(i as u32);
        if live != *stored {
            first_diff = Some((i, stored.clone(), live.to_string()));
            break;
        }
    }
    match first_diff {
        Some((i, stored, live)) => {
            println!("first token-identity mismatch at id {i}: stored={stored:?} live={live:?}")
        }
        None => println!("no token-identity mismatch found in the overlapping range checked"),
    }

    Ok(())
}
