// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Nix Codegen CLI Demo
//!
//! Mirrors brain_codegen.rs but emits NixOS configuration fragments.
//! Pipeline: prompt → classify intent → match idiom → verify with
//! `nix-instantiate --parse`. Pure neurosymbolic, no LLM.
//!
//! Usage:
//!   cargo run --example nix_codegen --features code_generation -- \
//!       "set up a rust dev environment with rust-analyzer and mold"
//!
//! Without arguments, runs the 5-prompt user-authorized demo set.

use symthaea::language::nix_codegen::{NixGenSource, classify_nix_intent, generate_nix};

fn print_separator(title: &str) {
    println!("\n┌─────────────────────────────────────────────────────────");
    println!("│ {title}");
    println!("└─────────────────────────────────────────────────────────");
}

fn run_one(prompt: &str) -> bool {
    print_separator(&format!("PROMPT: {prompt}"));
    let intent = classify_nix_intent(&prompt.to_lowercase());
    println!("System 1: {:?}", intent);

    let result = generate_nix(prompt);

    println!("Source:    {:?}", result.source);
    println!("Iterations: {}", result.iterations);
    println!(
        "Parses:    {}",
        if result.parses { "✓ YES" } else { "✗ NO" }
    );

    println!("\n--- Generated Nix ---");
    println!("{}", result.code);

    if !result.parses {
        if let Some(err) = &result.last_error {
            println!("--- Parse error ---");
            println!("  {err}");
        }
    }

    result.parses
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    println!("┌─────────────────────────────────────────────────────────");
    println!("│ Symthaea Nix Codegen — Neurosymbolic NixOS Pipeline");
    println!("└─────────────────────────────────────────────────────────");
    println!("\nVerifier: nix-instantiate --parse");

    if args.len() == 1 {
        // Custom prompt from CLI
        let _ = run_one(&args[0]);
        return;
    }

    // User-authorized demo set
    let demos = [
        "set up a rust dev environment with rust-analyzer and mold",
        "enable docker and add my user to the docker group",
        "set up postgresql with pgvector",
        "configure a python data-science environment with jupyter and pandas",
        "enable wayland with sway and configure standard fonts",
    ];

    let mut parses = 0;
    for prompt in &demos {
        if run_one(prompt) {
            parses += 1;
        }
    }

    println!("\n╔═════════════════════════════════════════════════════════");
    println!(
        "║ FINAL: {}/{} parse via nix-instantiate",
        parses,
        demos.len()
    );
    println!("║ Pipeline: text → intent → idiom → nix-instantiate");
    println!("║ No LLM. Pure neurosymbolic Nix generation.");
    println!("╚═════════════════════════════════════════════════════════");
}
