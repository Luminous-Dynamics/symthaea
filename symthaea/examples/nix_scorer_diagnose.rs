// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Diagnose a single golden-backed prompt: dump the generated code and
//! show every attrpath the scorer extracted on both sides. Useful for
//! figuring out *why* a structural score failed — usually because the
//! generator uses a different (but semantically equivalent) syntactic
//! form than the hand-written golden.
//!
//! Run:
//!   cargo run --features code_generation --example nix_scorer_diagnose -- "enable nginx web server"

use symthaea::language::nix_codegen::generate_nix;
use symthaea::language::nix_eval_goldens::golden_for;
use symthaea::language::nix_scorer::score;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let prompt = if args.is_empty() {
        "enable nginx web server"
    } else {
        args[0].as_str()
    };

    let Some(golden) = golden_for(prompt) else {
        eprintln!("No golden for prompt: {prompt:?}");
        std::process::exit(2);
    };

    let result = generate_nix(prompt);

    println!("=== prompt ===");
    println!("{prompt}\n");
    println!("=== generated ===");
    println!("{}", result.code);
    println!("=== golden ===");
    println!("{golden}");
    println!("=== verdict ===");
    let verdict = score(&result.code, golden);
    println!("  pass:         {}", verdict.pass());
    println!("  jaccard:      {:.2}", verdict.path_jaccard);
    println!(
        "  mismatches:   {} (value type/content differs on same path)",
        verdict.value_mismatches.len()
    );
    for m in &verdict.value_mismatches {
        println!(
            "      {}: got {} / want {}",
            m.path,
            m.got.display(),
            m.want.display()
        );
    }
    println!(
        "  missing:      {} (golden has it, generated doesn't)",
        verdict.missing_required.len()
    );
    for p in &verdict.missing_required {
        println!("      {}", p);
    }
    println!(
        "  extraneous:   {} (generated has it, golden doesn't — warning only)",
        verdict.extraneous.len()
    );
    for p in verdict.extraneous.iter().take(10) {
        println!("      {}", p);
    }
}
