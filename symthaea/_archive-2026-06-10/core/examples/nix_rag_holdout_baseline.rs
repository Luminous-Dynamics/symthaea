// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Baseline measurement for retrieval-augmented composition (RAG).
//!
//! Before designing new architecture, establish what the EXISTING
//! idiom+KG+repair pipeline (`generate_nix_with_self_repair`) scores
//! on the 13 held-out prompts that the Broca distillation path failed
//! (0/13 structural pass across every regime tested this session).
//!
//! If this pipeline hits ≥10/13, it validates the user's intuition
//! that the distillation was trying to replace something that already
//! works — and the path forward is to keep retrieval as the backbone
//! and use Broca for composition *inside* it, not instead of it.
//!
//! Takes no Broca checkpoint input. Loads the same 13-holdout JSONL
//! that distill_nix_evaluate has been using, runs the retrieval
//! pipeline per-prompt, scores each against the golden, reports
//! pass/fail plus the retrieval-derived attrpath.
//!
//! Usage:
//!   cargo run --features code_generation --release \
//!       --example nix_rag_holdout_baseline -- \
//!       --in ~/.cache/symthaea/combined-v2-58.jsonl

use serde::Deserialize;
use std::path::PathBuf;

use symthaea::language::nix_repair::generate_nix_with_self_repair;
use symthaea::language::nix_scorer::score;

#[derive(Debug, Deserialize)]
struct HoldoutPair {
    prompt: String,
    #[serde(default)]
    intent: String,
    code: String,
    #[serde(default)]
    holdout: bool,
}

fn parse_in() -> PathBuf {
    let args: Vec<String> = std::env::args().collect();
    for w in args.windows(2) {
        if w[0] == "--in" {
            return PathBuf::from(&w[1]);
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home)
        .join(".cache")
        .join("symthaea")
        .join("combined-v2-58.jsonl")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let in_path = parse_in();
    println!("┌─────────────────────────────────────────────────────────");
    println!("│ RAG baseline (retrieval + idiom + repair, no Broca)");
    println!("│ Input: {}", in_path.display());
    println!("└─────────────────────────────────────────────────────────");

    let text = std::fs::read_to_string(&in_path)?;
    let pairs: Vec<HoldoutPair> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(Box::<dyn std::error::Error>::from))
        .collect::<Result<_, _>>()?;

    let holdouts: Vec<&HoldoutPair> = pairs.iter().filter(|p| p.holdout).collect();
    println!("Found {} holdout prompts\n", holdouts.len());

    let mut pass = 0usize;
    let mut parse_valid = 0usize;
    let mut per_prompt: Vec<(String, bool, String)> = Vec::new();

    for (i, p) in holdouts.iter().enumerate() {
        let result = generate_nix_with_self_repair(&p.prompt, 5);
        let verdict = score(&result.code, &p.code);
        let passed = verdict.pass();
        let parses = verdict.parse_error.is_none();
        if passed {
            pass += 1;
        }
        if parses {
            parse_valid += 1;
        }
        let status = if passed {
            "PASS"
        } else if parses {
            "parse-ok, structural-fail"
        } else {
            "PARSE-ERR"
        };
        println!(
            "  [{}/{}] {} → {} ({} iter, {} step)",
            i + 1,
            holdouts.len(),
            p.prompt,
            status,
            result.iterations,
            result.steps.len()
        );
        if !passed && parses {
            // Show the pass-gating detail: missing paths + value mismatches.
            if !verdict.missing_required.is_empty() {
                println!(
                    "        missing_required: {}",
                    verdict.missing_required.join(", ")
                );
            }
            if !verdict.value_mismatches.is_empty() {
                let vm: Vec<String> = verdict
                    .value_mismatches
                    .iter()
                    .map(|m| {
                        format!(
                            "{}: got {} want {}",
                            m.path,
                            m.got.display(),
                            m.want.display()
                        )
                    })
                    .collect();
                println!("        value_mismatches: {}", vm.join("; "));
            }
        }
        per_prompt.push((p.prompt.clone(), passed, result.code));
    }

    println!("\n╔═════════════════════════════════════════════════════════");
    println!(
        "║ Structural pass: {}/{} ({:.0}%)",
        pass,
        holdouts.len(),
        100.0 * pass as f32 / holdouts.len() as f32
    );
    println!(
        "║ Parse-valid:     {}/{} ({:.0}%)",
        parse_valid,
        holdouts.len(),
        100.0 * parse_valid as f32 / holdouts.len() as f32
    );
    println!("╠═════════════════════════════════════════════════════════");
    println!("║ Compared to distill_nix_evaluate best-of-100 on same");
    println!("║ 13 holdout: parse 8/13, structural 0/13, keyword 2/13.");
    println!("║");
    println!("║ If RAG baseline ≫ Broca distillation, the research path");
    println!("║ is 'retrieve-then-compose' (use Broca INSIDE retrieval,");
    println!("║ not instead of it) rather than 'train bigger Broca'.");
    println!("╚═════════════════════════════════════════════════════════");

    Ok(())
}
