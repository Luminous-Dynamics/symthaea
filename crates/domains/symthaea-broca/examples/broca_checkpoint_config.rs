// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dump the `BrocaConfig` a checkpoint was saved with, plus its metadata.
//!
//! Written 2026-07-29 to rule out a specific confound on
//! `SYMTHAEA_BROCA_BASELINE_2026-07-29.md`: the eval path
//! (`CanonicalEvalCase`) has no `semantic_hv` field and therefore always passes `None`, while
//! the curriculum-training path (`broca_curriculum_sync.rs`) *does* compute one from objective
//! identity precisely because `build_channels()` "collapses to ~2 distinct vectors across a
//! real curriculum batch".
//!
//! If the checkpoint was trained with `enable_nsm_semantic = true` and real `semantic_hv`s,
//! then evaluating it without them is a **train/eval mismatch**, and the baseline's
//! "non-functional" verdict would be partly a harness artifact rather than a property of the
//! model. This prints the stored flags so that question is answered from the file itself
//! rather than inferred.
//!
//! ```bash
//! cargo run --release -p symthaea-broca --example broca_checkpoint_config -- [checkpoint]
//! ```

use symthaea_broca::checkpoint::BrocaCheckpoint;

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "crates/domains/symthaea-broca/data/models/broca-checkpoint-latest.bin".to_string()
    });
    eprintln!("loading {path} ...");
    let ck = BrocaCheckpoint::load_from_file_allow_checksum_mismatch(&path)?;

    println!("=== metadata ===");
    println!("  version           : {}", ck.version);
    println!("  training_epoch    : {}", ck.training_epoch);
    println!("  training_loss     : {}", ck.training_loss);
    println!("  vocab_size (meta) : {}", ck.metadata.vocab_size);
    println!("  vocab.tokens.len  : {}", ck.vocab.tokens.len());
    println!("  token_embeddings  : {}", ck.token_embeddings.len());
    println!("  embedding_dim     : {}", ck.metadata.embedding_dim);
    println!("  channel_count     : {}", ck.metadata.channel_count);
    println!(
        "  channel_schema_v  : {}",
        ck.metadata.channel_schema_version
    );
    println!("  backend           : {:?}", ck.metadata.backend);
    println!("  feature_set       : {:?}", ck.metadata.feature_set);

    let c = &ck.config;
    println!("\n=== the question this example exists to answer ===");
    println!("  enable_nsm_semantic   : {}", c.enable_nsm_semantic);
    println!("  nsm_semantic_alpha    : {}", c.nsm_semantic_alpha);
    println!("  enable_nsm_gate       : {}", c.enable_nsm_gate);
    println!(
        "  => {}",
        if c.enable_nsm_semantic {
            "TRAINED WITH semantic blending ON. Evaluating without semantic_hv is a \
             TRAIN/EVAL MISMATCH and the baseline is confounded."
        } else {
            "trained with semantic blending OFF -- the baseline evaluated it in the same \
             configuration it was trained in. No mismatch."
        }
    );

    println!("\n=== other generation-relevant flags ===");
    println!("  sampling                    : {:?}", c.sampling);
    println!(
        "  enable_coherence_feedback   : {}",
        c.enable_coherence_feedback
    );
    println!("  enable_semantic_veto        : {}", c.enable_semantic_veto);
    println!(
        "  enable_consciousness_gating : {}",
        c.enable_consciousness_gating
    );
    println!(
        "  enable_epistemic_cube_gate  : {}",
        c.enable_epistemic_cube_gate
    );
    println!("  bypass_gating               : {}", c.bypass_gating);
    println!(
        "  controller.vocab_size       : {}",
        c.controller.vocab_size
    );
    println!(
        "  controller.network_layers   : {}",
        c.controller.network_layers
    );
    println!(
        "  controller.neurons_per_layer: {}",
        c.controller.neurons_per_layer
    );
    Ok(())
}
