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
use symthaea_broca::generator::BrocaGenerator;
use symthaea_core::genesis::GenesisSeed;

/// NixIntent::ALL order — keep in lockstep with nix_codegen.rs.
/// Index of each variant in the 10D one-hot block.
#[allow(dead_code)]
enum NixIntentIdx {
    DevShell = 0,
    Service = 1,
    Hardware = 2,
    Desktop = 3,
    User = 4,
    Networking = 5,
    HomeManager = 6,
    Secrets = 7,
    FlakeTemplate = 8,
    Generic = 9,
}

/// Build a 17D Nix intent vector. Channels 0-9 are one-hot intent,
/// 10-16 are context scalars. Mirrors `nix_channels_as_slice` in the
/// main crate's nix_broca_bridge module.
fn make_channels_17(intent: NixIntentIdx, language: f32, has_hw: f32) -> [f32; 17] {
    let mut out = [0.0_f32; 17];
    out[intent as usize] = 1.0;
    out[10] = language;
    // [11]=item_count, [12]=has_extras, [13]=has_network_spec,
    // [14]=has_hardware, [15]=has_permission, [16]=has_wayland
    out[14] = has_hw;
    out
}

/// Pack the 17D intent vector into Broca's 43D ThoughtChannels. First
/// 17 positions carry the Nix signal; the remaining 26 default to
/// zero. The channel-semantics mismatch (Broca 0-7 = general intent,
/// our 0-9 = Nix intent) is acknowledged in the lib.rs docstring of
/// the main crate's bridge — full alignment is future work.
fn pack(flat: [f32; 17]) -> ThoughtChannels {
    let mut tc = ThoughtChannels::default();
    let n = tc.channels.len().min(flat.len());
    tc.channels[..n].copy_from_slice(&flat[..n]);
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
    // hypervector table will decorrelate.
    let genesis = GenesisSeed::from_phrase("symthaea-nix-distillation-m7b");
    let (mut generator, _adam, _proj, _liquid_mamba_cfg) =
        BrocaGenerator::from_checkpoint(&checkpoint, &genesis)
            .map_err(|e| format!("from_checkpoint: {}", e))?;
    println!(
        "Loaded. Vocab: {} tokens.",
        generator.tokenizer().vocab_size()
    );

    // Three canonical intents to demonstrate that (a) different
    // channels produce different outputs (signal propagation works)
    // and (b) the full emission pipeline runs without panics.
    let cases = [
        (
            "service: nginx",
            make_channels_17(NixIntentIdx::Service, 0.0, 0.0),
        ),
        (
            "dev-shell: rust",
            make_channels_17(NixIntentIdx::DevShell, 1.0, 0.0),
        ),
        (
            "hardware: nvidia",
            make_channels_17(NixIntentIdx::Hardware, 0.0, 1.0),
        ),
    ];

    for (label, flat) in cases {
        println!("\n── {} ──", label);
        println!("  channels[0..10] (intent): {:?}", &flat[..10]);
        println!("  channels[10..17] (context): {:?}", &flat[10..17]);
        let tc = pack(flat);
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
