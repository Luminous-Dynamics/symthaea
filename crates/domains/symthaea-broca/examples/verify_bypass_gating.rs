// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Diagnostic (2026-07-25, SYMTHAEA_COGNITION_IMPROVEMENT_PLAN_2026-07-21.md Tier 2.2
//! follow-up): the freshly-retrained broca-cfc-v9-gpu.bin checkpoint converged cleanly
//! (loss 12->2 over 20 epochs) but `--samples` output was near-total `<unk>` garbage.
//! Isolates whether the raw CfC decoder is broken, or whether the ~7-stage inference-time
//! logit gating stack (epistemic/epistemic-cube/emotional/NSM/...) is the culprit, by
//! generating the SAME thoughts with `bypass_gating` true vs false.
//!
//! Run: cargo run --release --example verify_bypass_gating -- <checkpoint-path>

use symthaea_broca::generator::BrocaGenerator;
use symthaea_broca::training::generate_diverse_thoughts;
use symthaea_core::genesis::GenesisSeed;

fn main() {
    let checkpoint_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "crates/domains/symthaea-broca/data/broca-cfc-v9-gpu.bin".to_string());
    let genesis_phrase = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "broca-training-default".to_string());
    eprintln!("genesis phrase: {genesis_phrase:?}");
    let genesis = GenesisSeed::from_phrase(&genesis_phrase);

    let (mut generator, ..) = BrocaGenerator::from_checkpoint(&checkpoint_path, &genesis)
        .unwrap_or_else(|e| panic!("Failed to load checkpoint '{checkpoint_path}': {e}"));

    let thoughts = generate_diverse_thoughts();
    let n = 8usize;
    let step = thoughts.len() / n.max(1);

    for bypass in [true, false] {
        generator.config_mut().bypass_gating = bypass;
        println!(
            "\n=== bypass_gating = {bypass} ({}) ===",
            if bypass { "RAW CfC" } else { "GATED" }
        );
        for i in 0..n.min(thoughts.len()) {
            let channels = &thoughts[(i * step) % thoughts.len()];
            let result = generator.generate(channels);
            println!(
                "[{i:>2}] {:>3} tok | \"{}\"",
                result.num_tokens,
                &result.text[..result.text.len().min(100)]
            );
        }
    }
}
