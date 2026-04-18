// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Compose eval benchmark (#3 of the "make this even better" list).
//!
//! Runs the Compose code generator across its golden-backed prompts,
//! scores each output with `compose_scorer::score`, prints a pass
//! rate. Mirrors `nix_eval_benchmark` in shape — enough to prove the
//! Nix pipeline's "scorer + golden + generator" pattern ports to a
//! second substrate.
//!
//! Usage:
//!   cargo run --features code_generation --example compose_eval_benchmark

use symthaea::language::compose_codegen::{
    compose_golden_for, compose_golden_prompts, generate_compose,
};
use symthaea::language::compose_scorer::score;

fn main() {
    println!("┌─────────────────────────────────────────────────────────");
    println!("│ Compose Eval Benchmark (#3 substrate-independence)");
    println!(
        "│ Golden-backed prompts: {}",
        compose_golden_prompts().len()
    );
    println!("└─────────────────────────────────────────────────────────");

    let mut pass = 0usize;
    for prompt in compose_golden_prompts() {
        let result = generate_compose(prompt);
        let golden = match compose_golden_for(prompt) {
            Some(g) => g,
            None => {
                println!("  · [no golden] {}", prompt);
                continue;
            }
        };
        let verdict = score(&result.code, golden);
        let mark = if verdict.pass() { "✓" } else { "✗" };
        if verdict.pass() {
            pass += 1;
        }
        println!(
            "  {} {:50} | {} ({:?} / {:?})",
            mark,
            prompt,
            verdict.summary(),
            result.intent,
            result.source
        );
        for p in verdict.missing_required.iter().take(2) {
            println!("        missing: {}", p);
        }
        for m in verdict.value_mismatches.iter().take(1) {
            println!(
                "        mismatch: {} (got {} want {})",
                m.path,
                m.got.display(),
                m.want.display()
            );
        }
    }

    let total = compose_golden_prompts().len();
    let rate = pass as f32 / total as f32 * 100.0;
    println!("\n╔═════════════════════════════════════════════════════════");
    println!("║ Compose pass: {}/{} ({:.0}%)", pass, total, rate);
    println!("╚═════════════════════════════════════════════════════════");
}
