// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Phase 2 M7 (harvester): generate (prompt, channels, code) training
//! pairs for Broca distillation by running the Phase 1 repair loop over
//! the golden-backed NixEval subset.
//!
//! Pipeline per prompt:
//! 1. `generate_nix_with_scorer_repair(prompt, golden, max_iters=5)`
//!    — lets the M2 scorer-in-the-loop close small gaps so the training
//!    set isn't starved by idiom-template misses.
//! 2. Skip any prompt where the final verdict failed (M7 trains on
//!    structurally-correct pairs only — the whole point is "PASS-
//!    filtered distillation").
//! 3. Compute the 17-channel intent bridge vector (`nix_channels_flat`).
//! 4. Emit one JSONL line: `{prompt, intent, channels, code,
//!    repair_steps, iterations}`.
//!
//! Default output: `~/.cache/symthaea/distillation-pairs.jsonl`.
//! Override via `--out <path>`.
//!
//! Usage:
//!   cargo run --features code_generation \
//!       --example harvest_nix_distillation
//!   cargo run --features code_generation \
//!       --example harvest_nix_distillation -- --out /tmp/pairs.jsonl
//!
//! Consumed by the M7 trainer (Broca distillation — next milestone).

use std::io::Write;
use std::path::PathBuf;

use symthaea::language::nix_broca_bridge::{nix_channels_flat, NIX_CHANNEL_COUNT};
use symthaea::language::nix_codegen::classify_nix_intent;
use symthaea::language::nix_eval_goldens::golden_for;
use symthaea::language::nix_repair::generate_nix_with_scorer_repair;

/// Candidate prompts — the 26 golden-backed subset from
/// `nix_eval_goldens`. Mirrored here (rather than imported from
/// `score_all_goldens`) so the harvester can log per-prompt progress.
const HARVEST_PROMPTS: &[&str] = &[
    "set up postgresql with pgvector",
    "configure postgresql service",
    "enable nginx web server",
    "enable redis cache server",
    "enable docker and add my user to the docker group",
    "set up ipfs kubo node",
    "enable tailscale VPN",
    "configure prometheus monitoring",
    "grafana dashboard server",
    "configure CUPS printing service",
    "enable systemd-resolved for DNS",
    "configure nvidia gpu drivers",
    "enable nvidia hardware acceleration",
    "configure intel hardware acceleration",
    "set up sway window manager",
    "enable kde plasma desktop environment",
    "enable hyprland wayland compositor",
    "set up hyprland with fonts",
    "set up gnome desktop environment",
    "open firewall ports 80 and 443",
    "open port 8080 in firewall",
    "open udp port 51820 for wireguard",
    "set time zone to Africa/Johannesburg",
    "set up a rust dev environment with rust-analyzer and mold",
    "rust dev shell with sccache and openssl",
    "set up a node development environment with typescript",
];

fn default_out_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home)
        .join(".cache")
        .join("symthaea")
        .join("distillation-pairs.jsonl")
}

fn parse_out_arg() -> PathBuf {
    let args: Vec<String> = std::env::args().collect();
    for w in args.windows(2) {
        if w[0] == "--out" {
            return PathBuf::from(&w[1]);
        }
    }
    default_out_path()
}

/// One training pair in on-disk form. Channels is a fixed-size array
/// rather than Vec<f32> so the downstream trainer doesn't need to
/// runtime-check width each load.
#[derive(serde::Serialize)]
struct DistillPair<'a> {
    prompt: &'a str,
    /// Debug-format of `NixIntent` — cheap string match for the trainer.
    intent: String,
    /// `NIX_CHANNEL_COUNT`-long float vector (17D by default).
    channels: [f32; NIX_CHANNEL_COUNT],
    /// Structurally PASS-verified generated Nix source.
    code: String,
    /// How many scorer→repair iterations the loop took before PASS.
    /// Zero means initial generation already passed (no repair needed).
    iterations: usize,
    /// Count of repair steps applied. Useful for stratifying training
    /// — zero-step pairs are "easy"; multi-step pairs taught the
    /// repair heuristics something.
    repair_steps: usize,
}

fn main() {
    let out_path = parse_out_arg();
    if let Some(parent) = out_path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            eprintln!(
                "✗ Cannot create output directory {}: {}",
                parent.display(),
                e
            );
            std::process::exit(1);
        }
    }
    let file = match std::fs::File::create(&out_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("✗ Cannot open {}: {}", out_path.display(), e);
            std::process::exit(1);
        }
    };
    let mut writer = std::io::BufWriter::new(file);

    println!("┌─────────────────────────────────────────────────────────");
    println!("│ Nix Distillation Harvester (Phase 2 M7)");
    println!("│ Output: {}", out_path.display());
    println!("│ Candidates: {} prompts", HARVEST_PROMPTS.len());
    println!("└─────────────────────────────────────────────────────────");

    let mut harvested = 0usize;
    let mut skipped_fail = 0usize;
    let mut skipped_no_golden = 0usize;
    let mut total_repair_steps = 0usize;
    let mut pairs_with_repair = 0usize;

    for prompt in HARVEST_PROMPTS {
        let Some(golden) = golden_for(prompt) else {
            println!("  · [skip, no golden] {}", prompt);
            skipped_no_golden += 1;
            continue;
        };
        let result = generate_nix_with_scorer_repair(prompt, golden, 5);
        if !result.verdict.pass() {
            println!("  ✗ [FAIL after {} iter(s)] {}", result.iterations, prompt);
            skipped_fail += 1;
            continue;
        }
        let intent = classify_nix_intent(&prompt.to_lowercase());
        let channels = nix_channels_flat(prompt);
        let pair = DistillPair {
            prompt,
            intent: format!("{intent:?}"),
            channels,
            code: result.code,
            iterations: result.iterations,
            repair_steps: result.steps.len(),
        };
        match serde_json::to_string(&pair) {
            Ok(line) => {
                if let Err(e) = writeln!(writer, "{}", line) {
                    eprintln!("✗ write error on {}: {}", prompt, e);
                    std::process::exit(1);
                }
            }
            Err(e) => {
                eprintln!("✗ serialization error on {}: {}", prompt, e);
                std::process::exit(1);
            }
        }
        let tag = if result.steps.is_empty() {
            "initial"
        } else {
            pairs_with_repair += 1;
            total_repair_steps += result.steps.len();
            "repaired"
        };
        println!(
            "  ✓ [{} iter(s), {} step(s), {}] {}",
            result.iterations,
            result.steps.len(),
            tag,
            prompt
        );
        harvested += 1;
    }

    if let Err(e) = writer.flush() {
        eprintln!("✗ flush error: {}", e);
        std::process::exit(1);
    }

    println!("\n╔═════════════════════════════════════════════════════════");
    println!("║ Harvested: {}/{} pairs", harvested, HARVEST_PROMPTS.len());
    println!(
        "║   of which {} required repair (total {} steps)",
        pairs_with_repair, total_repair_steps
    );
    println!("║ Skipped (FAIL after repair): {}", skipped_fail);
    println!("║ Skipped (no golden):          {}", skipped_no_golden);
    println!("║ Written to: {}", out_path.display());
    println!("╚═════════════════════════════════════════════════════════");
    // Exit 0 even with skipped entries — skips are real signal, not
    // harness failures. `--goldens-require-all-pass` is the strict
    // variant on the scorer side; harvester is deliberately permissive.
}
