// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Diagnostic (2026-07-25, follow-up to the checkpoint vocab-restore fix in
//! generator.rs): post-fix generation from `broca-checkpoint-latest.bin` produces
//! real vocabulary, but samples for different thought intents share a repeated
//! low-diversity prefix ("lap website (district) plus ..." / "chew(-ory) rupture
//! ..." depending on genesis phrase) regardless of input. Hypothesis: classic
//! greedy-decoding mode collapse at the START of generation, where the thought-
//! conditioning signal is weak relative to the marginal token-frequency prior in
//! the first few steps -- NOT a training deficiency requiring a retrain. Tests
//! Greedy vs TopK (several k/temperature combos) vs TopP on the SAME checkpoint,
//! measuring first-3-token diversity across 8 distinct thought inputs.
//!
//! Run: cargo run --release --example probe_prefix_diversity -- <checkpoint-path> <genesis-phrase>

use std::collections::HashSet;

use symthaea_broca::generator::{BrocaGenerator, SamplingStrategy};
use symthaea_broca::training::generate_diverse_thoughts;
use symthaea_core::genesis::GenesisSeed;

fn first_n_words(text: &str, n: usize) -> String {
    text.split_whitespace()
        .take(n)
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() {
    let checkpoint_path = std::env::args().nth(1).unwrap_or_else(|| {
        "crates/domains/symthaea-broca/data/models/broca-checkpoint-latest.bin".to_string()
    });
    let genesis_phrase = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "symthaea luminous dynamics".to_string());
    eprintln!("checkpoint: {checkpoint_path}");
    eprintln!("genesis phrase: {genesis_phrase:?}");
    let genesis = GenesisSeed::from_phrase(&genesis_phrase);

    let (mut generator, ..) = BrocaGenerator::from_checkpoint(&checkpoint_path, &genesis)
        .unwrap_or_else(|e| panic!("Failed to load checkpoint '{checkpoint_path}': {e}"));

    let thoughts = generate_diverse_thoughts();
    let n = 8usize;
    let step = thoughts.len() / n.max(1);

    let strategies: Vec<(&str, SamplingStrategy)> = vec![
        ("greedy", SamplingStrategy::Greedy),
        (
            "topk-k5-t0.7",
            SamplingStrategy::TopK {
                k: 5,
                temperature: 0.7,
            },
        ),
        (
            "topk-k20-t1.0",
            SamplingStrategy::TopK {
                k: 20,
                temperature: 1.0,
            },
        ),
        (
            "topp-p0.9-t0.9",
            SamplingStrategy::TopP {
                p: 0.9,
                temperature: 0.9,
            },
        ),
    ];

    for (label, strategy) in strategies {
        generator.set_sampling(strategy);
        println!("\n=== {label} ===");
        let mut prefixes: HashSet<String> = HashSet::new();
        for i in 0..n.min(thoughts.len()) {
            let channels = &thoughts[(i * step) % thoughts.len()];
            let result = generator.generate(channels);
            let safe_end = (0..result.text.len().min(100))
                .rev()
                .find(|&b| result.text.is_char_boundary(b))
                .unwrap_or(0);
            let prefix3 = first_n_words(&result.text, 3);
            prefixes.insert(prefix3.clone());
            println!(
                "[{i:>2}] {:>3} tok | prefix3=\"{prefix3}\" | \"{}\"",
                result.num_tokens,
                &result.text[..safe_end]
            );
        }
        println!(
            "  unique first-3-word prefixes: {}/{}",
            prefixes.len(),
            n.min(thoughts.len())
        );
    }
}
