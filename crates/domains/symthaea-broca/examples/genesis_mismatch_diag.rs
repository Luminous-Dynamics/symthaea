// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Diagnostic: does loading a checkpoint with the WRONG genesis phrase
//! (mismatched vs. the phrase used during training) produce degenerate
//! output, even though `from_checkpoint` doesn't error?
//!
//! `BrocaGenerator::from_checkpoint_struct` restores `network_state` and
//! `token_embeddings` verbatim from the file, but constructs the
//! `ThoughtLanguageEncoder` fresh from whatever `GenesisSeed` the CALLER
//! passes in (`ThoughtLanguageEncoder::new(genesis)` derives its
//! base/level hypervectors purely from the genesis phrase). If that
//! phrase differs from what training used, the encoder produces a
//! completely different, unrelated `thought_hv` for the same channels --
//! structured noise from the trained network's point of view. The
//! checkpoint's blake3 checksum only covers file integrity, not genesis
//! provenance, so a mismatch loads silently with no error.
//!
//! broca-train's default genesis phrase is "broca-training-default".
//! broca-topic-coverage / broca-decoder-ab / broca_curriculum_sync all
//! default to "symthaea luminous dynamics" instead. No script in this
//! repo passes --genesis explicitly anywhere in the curriculum-bridge
//! pipeline, so if the checkpoint was ever trained/resumed under one
//! default and evaluated under the other, every evaluation would be
//! silently corrupted regardless of training quality.
//!
//! RESULT (2026-07-09, run against `broca-checkpoint-latest.bin`): this
//! hypothesis is **refuted**. Both genesis phrases produce equally
//! degenerate word-salad for the same channels -- neither is coherent.
//! Worse: feeding the model the EXACT channels from its own base-corpus
//! training examples (`train-combined-v8.jsonl`) also produces unrelated
//! word-salad, nowhere close to the known `target_text`, under both
//! phrases. This rules out genesis mismatch AND out-of-distribution
//! curriculum channels as the explanation -- the production checkpoint's
//! *direct-decoder* generation is fundamentally low-fluency right now,
//! even on its own in-distribution training data. This is a separate,
//! deeper standing issue from anything the semantic_hv bootstrap work
//! (Part 3/4 of the curriculum-bridge plan) was targeting: every
//! "candidate shows no improvement over baseline" gate failure this
//! session was comparing degenerate-vs-degenerate output, not measuring
//! whether semantic conditioning helps.

use anyhow::{Context, Result};
use serde::Deserialize;
use symthaea_broca::encoder::{NUM_CHANNELS, ThoughtChannels};
use symthaea_broca::generator::BrocaGenerator;
use symthaea_core::genesis::GenesisSeed;

#[derive(Debug, Deserialize)]
struct SamplePair {
    channels: Vec<f32>,
    target_text: String,
}

fn channels_from_raw(raw: &[f32]) -> ThoughtChannels {
    let mut ch = ThoughtChannels::default();
    let n = raw.len().min(NUM_CHANNELS);
    ch.channels[..n].copy_from_slice(&raw[..n]);
    ch
}

fn main() -> Result<()> {
    let checkpoint_path = std::env::args().nth(1).unwrap_or_else(|| {
        "crates/domains/symthaea-broca/data/models/broca-checkpoint-latest.bin".to_string()
    });
    let sample_pairs_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/dev/null".to_string());

    let phrases = ["broca-training-default", "symthaea luminous dynamics"];

    // A handful of representative channel vectors: default + a few intent
    // variants, matching what both the base corpus and curriculum holdout
    // use.
    let mut test_channels: Vec<(String, ThoughtChannels)> = vec![
        ("default".to_string(), ThoughtChannels::default()),
        ("intent(0)".to_string(), ThoughtChannels::with_intent(0)),
        ("intent(1)".to_string(), ThoughtChannels::with_intent(1)),
        ("intent(2)".to_string(), ThoughtChannels::with_intent(2)),
    ];

    // Real base-corpus (train-combined-v8.jsonl) channels+target_text, so we
    // can compare the model's IN-DISTRIBUTION reproduction quality against
    // its actual known training target, not just generic presets.
    let mut sample_targets: Vec<(String, String)> = Vec::new();
    if let Ok(contents) = std::fs::read_to_string(&sample_pairs_path) {
        for (i, line) in contents.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let pair: SamplePair = serde_json::from_str(line).context("parsing sample pair")?;
            let label = format!("base-corpus[{i}]");
            test_channels.push((label.clone(), channels_from_raw(&pair.channels)));
            sample_targets.push((label, pair.target_text));
        }
    }

    for phrase in phrases {
        println!("\n=== genesis phrase: {phrase:?} ===");
        let genesis = GenesisSeed::from_phrase(phrase);
        let (mut generator, _, _, _) =
            BrocaGenerator::from_checkpoint_allow_checksum_mismatch(&checkpoint_path, &genesis)
                .with_context(|| format!("loading checkpoint {checkpoint_path}"))?;
        generator.config_mut().bypass_gating = true;

        for (label, channels) in &test_channels {
            let result = generator.generate(channels);
            println!("  [{label}] generated: {:?}", result.text);
            if let Some((_, target)) = sample_targets.iter().find(|(l, _)| l == label) {
                println!("  [{label}] target:    {target:?}");
            }
        }
    }

    Ok(())
}
