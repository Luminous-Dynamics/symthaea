// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Epistemic-gate A/B demo (#2 of the "make this even better" list).
//!
//! Loads a trained checkpoint and runs the SAME channel input twice:
//! once with `enable_epistemic_gate = true`, once with `false`. Side-
//! by-side comparison of the emitted text shows whether the gate
//! actually changes generation behavior — the architectural claim
//! "consciousness-gated emission" becomes defensible or hollow
//! depending on the observed delta.
//!
//! Hypothesis: with gate ON, low-confidence tokens (unknown option
//! paths, rare vocab) get suppressed at logit level. With gate OFF,
//! the generator emits whatever the CfC-HDC controller selects
//! unconstrained.
//!
//! What this bin measures:
//! 1. **Byte-count delta**: do the two runs produce different-length
//!    outputs? (If identical, the gate is a no-op.)
//! 2. **Jaccard token overlap**: how much of the emitted token-set
//!    is shared between gate-on and gate-off? (Low overlap = gate is
//!    actively reshaping the emission.)
//! 3. **Tokens suppressed by the gate**: tokens present in gate-off
//!    but absent in gate-on. (These are the "would-have-hallucinated"
//!    emissions — prints them for qualitative inspection.)
//!
//! Usage:
//!   cargo run --bin distill_nix_gate_ab -p symthaea-broca -- \
//!       --checkpoint /tmp/broca-nix-aligned-25ep.mpk

use std::collections::HashSet;
use std::path::PathBuf;

use symthaea_broca::encoder::ThoughtChannels;
use symthaea_broca::generator::{BrocaGenerator, SamplingStrategy};
use symthaea_core::genesis::GenesisSeed;

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

fn make_channels(
    intent: BrocaIntentSlot,
    language_norm: f32,
    has_hardware_flag: f32,
) -> ThoughtChannels {
    let mut tc = ThoughtChannels::default();
    for i in 0..8 {
        tc.channels[i] = 0.0;
    }
    tc.channels[intent as usize] = 1.0;
    tc.channels[26] = language_norm;
    tc.channels[27] = has_hardware_flag * 0.5;
    tc
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

/// Split a generated string into whitespace-separated tokens, trimmed.
/// Not a real tokenizer — just enough for Jaccard overlap reporting.
fn tokens_of(text: &str) -> HashSet<String> {
    text.split_whitespace()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn home_path() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string()))
}

/// Set all gating flags on the loaded generator. Mutates in place;
/// caller can flip between calls with the same generator instance so
/// the weights stay constant.
fn set_gates(gen: &mut BrocaGenerator, enable: bool) {
    let cfg = gen.config_mut();
    cfg.enable_epistemic_gate = enable;
    cfg.enable_emotional_modulation = enable;
    cfg.enable_coherence_feedback = enable;
    cfg.enable_consciousness_gating = enable;
    cfg.enable_semantic_veto = enable;
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint = parse_path(
        "--checkpoint",
        home_path()
            .join(".cache")
            .join("symthaea")
            .join("broca-nix-distilled.mpk"),
    );
    println!("┌─────────────────────────────────────────────────────────");
    println!("│ Epistemic-gate A/B Demo (#2)");
    println!("│ Checkpoint: {}", checkpoint.display());
    println!("└─────────────────────────────────────────────────────────");
    if !checkpoint.exists() {
        eprintln!("✗ Checkpoint not found at {}", checkpoint.display());
        std::process::exit(1);
    }

    let genesis = GenesisSeed::from_phrase("symthaea-nix-distillation-m7b");
    let (mut generator, _a, _p, _l) = BrocaGenerator::from_checkpoint(&checkpoint, &genesis)
        .map_err(|e| format!("from_checkpoint: {}", e))?;

    // Deterministic sampling so the A/B comparison is pure-signal —
    // stochasticity would wash out the gate's effect.
    generator.set_sampling(SamplingStrategy::TopK {
        k: 20,
        temperature: 0.7,
    });
    // Seeded RNG → identical token-selection paths at identical logits.
    generator.config_mut().sampling_seed = Some(42);
    generator.config_mut().repetition_penalty = 1.3;

    let cases = [
        (
            "service: nginx",
            make_channels(BrocaIntentSlot::Service, 0.0, 0.0),
        ),
        (
            "dev-shell: rust",
            make_channels(BrocaIntentSlot::DevShell, 1.0 / 6.0, 0.0),
        ),
        (
            "hardware: nvidia",
            make_channels(BrocaIntentSlot::Hardware, 0.0, 1.0),
        ),
    ];

    for (label, tc) in cases {
        println!("\n═══════════ {} ═══════════", label);

        // A: gate ON
        set_gates(&mut generator, true);
        let on = generator.generate(&tc);
        let on_tokens = tokens_of(&on.text);

        // B: gate OFF
        set_gates(&mut generator, false);
        let off = generator.generate(&tc);
        let off_tokens = tokens_of(&off.text);

        // Measurements
        let intersection: HashSet<&String> = on_tokens.intersection(&off_tokens).collect();
        let union: HashSet<&String> = on_tokens.union(&off_tokens).collect();
        let jaccard = if union.is_empty() {
            1.0
        } else {
            intersection.len() as f32 / union.len() as f32
        };
        let suppressed: Vec<&String> = off_tokens.difference(&on_tokens).collect();
        let introduced: Vec<&String> = on_tokens.difference(&off_tokens).collect();

        println!(
            "▶ GATE ON  ({:>3} bytes, {:>3} unique tokens):",
            on.text.len(),
            on_tokens.len()
        );
        println!("  {}", on.text.replace('\n', " "));

        println!(
            "▶ GATE OFF ({:>3} bytes, {:>3} unique tokens):",
            off.text.len(),
            off_tokens.len()
        );
        println!("  {}", off.text.replace('\n', " "));

        println!("\n▶ Measurements:");
        println!("  Token Jaccard (on ∩ off) / (on ∪ off) = {:.2}", jaccard);
        println!(
            "  Suppressed by gate (in OFF, not ON): {}",
            suppressed.len()
        );
        if !suppressed.is_empty() && suppressed.len() <= 25 {
            let mut shown: Vec<String> = suppressed.iter().take(25).map(|s| (*s).clone()).collect();
            shown.sort();
            println!("    {}", shown.join(", "));
        }
        println!(
            "  Introduced by gate (in ON, not OFF): {}",
            introduced.len()
        );
        if !introduced.is_empty() && introduced.len() <= 25 {
            let mut shown: Vec<String> = introduced.iter().take(25).map(|s| (*s).clone()).collect();
            shown.sort();
            println!("    {}", shown.join(", "));
        }
    }

    println!("\n── Interpretation ──");
    println!("Jaccard ≈ 1.0 → gate is effectively a no-op at current");
    println!("              signal strength (expected for an undertrained");
    println!("              26-pair checkpoint).");
    println!("Jaccard < 0.8 → gate is materially reshaping emission.");
    println!("Suppressed tokens = what the gate told the model NOT to say.");
    Ok(())
}
