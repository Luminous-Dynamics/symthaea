// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Phase 2 M8 — Broca-generates-Nix smoke demo.
//!
//! Loads a distillation checkpoint (from `distill_nix_train`, M7.b),
//! constructs canonical `ThoughtChannels` for a handful of Nix intents,
//! runs `generate()`, prints what came out.
//!
//! Why broca-local bin, not main-crate example: the main crate's
//! default features include `broca_lite` (lightweight Spore-based
//! generation) which is mutually exclusive with `ssm_language` (full
//! CfC-HDC via this crate). Running an example with `ssm_language`
//! enabled requires `--no-default-features`, which in turn breaks
//! other lib.rs modules that assume defaults. Rather than untangle
//! the feature chain (scope for a later cleanup), we keep M8 as a
//! broca-internal bin that exercises only the Broca API — no main-
//! crate cross-compilation required.
//!
//! The 17D Nix intent vectors this demo uses are hand-coded copies of
//! what `symthaea::language::nix_broca_bridge::nix_channels_flat`
//! would produce for canonical prompts. Keep in lockstep with that
//! module's layout (channels 0-9 intent, 10-16 context scalars).
//!
//! Purpose: **prove the end-to-end wiring** (checkpoint → channels →
//! generator → text). A 1-epoch smoke-trained model will not produce
//! usable Nix; 50-200 epochs on GPU would. This demo verifies the
//! mechanism, not the trained quality.
//!
//! Usage:
//!   cargo run --bin distill_nix_generate -p symthaea-broca
//!   cargo run --bin distill_nix_generate -p symthaea-broca -- \
//!       --checkpoint /tmp/broca-nix-smoke.mpk

use std::path::PathBuf;

use symthaea_broca::encoder::ThoughtChannels;
use symthaea_broca::generator::{BrocaGenerator, SamplingStrategy};
use symthaea_core::genesis::GenesisSeed;

/// Broca-aligned intent slot (post-projection 10→8-way). Keep in
/// lockstep with `nix_broca_bridge::project_intent` in the main
/// crate.
#[allow(dead_code)]
enum BrocaIntentSlot {
    DevShell = 0,
    Service = 1,
    Hardware = 2,
    Desktop = 3,
    User = 4,
    Networking = 5,
    HomeManager = 6,
    Secrets = 7,
}

/// Build a Broca-aligned 43D ThoughtChannels for a canonical Nix
/// intent. Intent at positions 0-7 (one-hot), context scalars at
/// 24-27 (code channels), emotional/consciousness/epistemic blocks
/// left at zero. Matches the layout produced by
/// `nix_broca_bridge::nix_channels_as_broca` in the main crate.
///
/// Parameters:
/// - `intent`: projected 8-way Broca intent slot.
/// - `language_norm`: 0..1, where Rust=1/6, Python=2/6, etc.
/// - `has_hardware_flag`: 1.0 if the prompt mentions gpu/hardware.
fn make_broca_channels(
    intent: BrocaIntentSlot,
    language_norm: f32,
    has_hardware_flag: f32,
) -> ThoughtChannels {
    let mut tc = ThoughtChannels::default();
    // Broca's `ThoughtChannels::default()` sets position 7 to 1.0
    // (Unknown-intent default). Clear the 0-7 intent block before
    // setting our target — otherwise every generation gets BOTH the
    // requested intent AND Unknown as active, which muddies output.
    // This matches `nix_channels_as_broca` in the main crate, which
    // always produces a clean one-hot in 0-7.
    for i in 0..8 {
        tc.channels[i] = 0.0;
    }
    // 0..8 — intent one-hot.
    tc.channels[intent as usize] = 1.0;
    // 24..28 — code-channel-packed Nix context.
    tc.channels[24] = 0.0; // syntax_complexity — smoke demo leaves 0
    tc.channels[25] = 0.0; // has_extras — smoke demo leaves 0
    tc.channels[26] = language_norm; // algorithm_pattern = language / 6.0
    tc.channels[27] = has_hardware_flag * 0.5; // hardware = top bit of 4-flag pack
    tc
}

fn default_checkpoint() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home)
        .join(".cache")
        .join("symthaea")
        .join("broca-nix-distilled.mpk")
}

fn parse_checkpoint_path() -> PathBuf {
    let args: Vec<String> = std::env::args().collect();
    for w in args.windows(2) {
        if w[0] == "--checkpoint" {
            return PathBuf::from(&w[1]);
        }
    }
    default_checkpoint()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint = parse_checkpoint_path();

    println!("┌─────────────────────────────────────────────────────────");
    println!("│ Broca → Nix Smoke-Demo (Phase 2 M8)");
    println!("│ Checkpoint: {}", checkpoint.display());
    println!("└─────────────────────────────────────────────────────────");

    if !checkpoint.exists() {
        eprintln!("✗ Checkpoint not found at {}", checkpoint.display());
        eprintln!("  Run the M7.b trainer first:");
        eprintln!(
            "    cargo run --bin distill_nix_train -p symthaea-broca \\\n    \
             -- --out {} --epochs 10",
            checkpoint.display()
        );
        std::process::exit(1);
    }

    // from_checkpoint needs the genesis seed used at training time.
    // Must match the seed used by distill_nix_train.rs or the
    // hypervector table will decorrelate. The vocabulary is restored
    // from the checkpoint's `vocab` field (including any NIX_TOKENS
    // augmentation the trainer applied), so nothing extra to do here.
    let genesis = GenesisSeed::from_phrase("symthaea-nix-distillation-m7b");
    let (mut generator, _adam, _proj, _liquid_mamba_cfg) =
        BrocaGenerator::from_checkpoint(&checkpoint, &genesis)
            .map_err(|e| format!("from_checkpoint: {}", e))?;
    println!(
        "Loaded. Vocab: {} tokens (NIX_TOKENS restored from checkpoint).",
        generator.tokenizer().vocab_size()
    );

    // Override sampling: checkpoint was trained under `BrocaConfig::
    // default()` which uses `SamplingStrategy::Greedy` + no repetition
    // penalty. Greedy on an over-fit 26-pair model produces loops like
    // `services.services.` and `config config`. Swap to top-k + a
    // stronger repetition penalty so token diversity kicks in.
    generator.set_sampling(SamplingStrategy::TopK {
        k: 20,
        temperature: 0.7,
    });
    generator.config_mut().repetition_penalty = 1.3;
    println!("Sampling: TopK(k=20, temp=0.7), repetition_penalty=1.3");

    // Three canonical intents to demonstrate that (a) different
    // channels produce different outputs (signal propagation works)
    // and (b) the full emission pipeline runs without panics.
    // Language norm: rust = 1/6 ≈ 0.167 for the dev-shell case.
    let cases = [
        (
            "service: nginx",
            make_broca_channels(BrocaIntentSlot::Service, 0.0, 0.0),
        ),
        (
            "dev-shell: rust",
            make_broca_channels(BrocaIntentSlot::DevShell, 1.0 / 6.0, 0.0),
        ),
        (
            "hardware: nvidia",
            make_broca_channels(BrocaIntentSlot::Hardware, 0.0, 1.0),
        ),
    ];

    for (label, tc) in cases {
        println!("\n── {} ──", label);
        println!("  channels[0..8] (intent): {:?}", &tc.channels[..8]);
        println!(
            "  channels[24..28] (code/context): {:?}",
            &tc.channels[24..28]
        );
        let result = generator.generate(&tc);
        println!("  → {} bytes emitted", result.text.len());
        println!("  ▽ {}", result.text.replace('\n', "\n    "));
    }

    println!("\n── End of demo ──");
    println!(
        "Note: this checkpoint was trained {} epoch(s) on 26 pairs.",
        "~1"
    );
    println!("Real-quality Nix emission needs 50-200 epochs (GPU).");

    Ok(())
}