// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Hold-out generalization score (#1 of the "make this even better"
//! list, phase 2).
//!
//! Reads the JSONL produced by the broca-bin `distill_nix_evaluate`
//! (one `{prompt, intent, golden, generated, generated_bytes}` per
//! line), runs the structural scorer against each golden, reports
//! the pass rate.
//!
//! Two-phase because broca's `ssm_language` feature conflicts with
//! main-crate defaults (`broca_lite`). The broca-bin generates text;
//! this main-crate example scores it. Together they implement the
//! gating test: does the Broca-distilled model generalize to held-out
//! prompts it never saw during training?
//!
//! Usage:
//!   cargo run --features code_generation --example nix_holdout_score \
//!       -- --in /tmp/holdout-generated.jsonl

use serde::Deserialize;
use std::path::PathBuf;

use symthaea::language::nix_scorer::score;

#[derive(Debug, Deserialize)]
struct EvalRow {
    prompt: String,
    intent: String,
    golden: String,
    generated: String,
    #[serde(default)]
    generated_bytes: usize,
}

fn parse_in_path() -> PathBuf {
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
        .join("holdout-generated.jsonl")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let in_path = parse_in_path();
    println!("┌─────────────────────────────────────────────────────────");
    println!("│ Hold-out Generalization Score (#1)");
    println!("│ Input: {}", in_path.display());
    println!("└─────────────────────────────────────────────────────────");

    let text = std::fs::read_to_string(&in_path)
        .map_err(|e| format!("reading {}: {}", in_path.display(), e))?;
    let rows: Vec<EvalRow> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, line)| {
            serde_json::from_str::<EvalRow>(line).map_err(|e| format!("line {}: {}", i + 1, e))
        })
        .collect::<Result<_, _>>()?;

    if rows.is_empty() {
        eprintln!("✗ No rows in {}.", in_path.display());
        std::process::exit(1);
    }

    let mut pass = 0usize;
    let mut empty_gen = 0usize;
    for (i, r) in rows.iter().enumerate() {
        let verdict = score(&r.generated, &r.golden);
        let mark = if verdict.pass() { "✓" } else { "✗" };
        if verdict.pass() {
            pass += 1;
        }
        if r.generated.trim().is_empty() || r.generated_bytes < 5 {
            empty_gen += 1;
        }
        let summary = verdict.summary();
        println!(
            "  {} [{}/{:02}] {} ({})\n        {}",
            mark,
            i + 1,
            rows.len(),
            r.prompt,
            r.intent,
            summary
        );
        // Print up to 2 missing paths + 1 mismatch for diagnostics.
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

    let rate = (pass as f32 / rows.len() as f32) * 100.0;
    println!("\n╔═════════════════════════════════════════════════════════");
    println!("║ Hold-out pass: {}/{} ({:.0}%)", pass, rows.len(), rate);
    if empty_gen > 0 {
        println!(
            "║ ({} of {} outputs were effectively empty — the model emitted",
            empty_gen,
            rows.len()
        );
        println!("║  nothing or near-nothing for those prompts, which means it's");
        println!("║  failing LOUDLY rather than hallucinating — epistemic honesty.)");
    }
    println!("╠═════════════════════════════════════════════════════════");
    println!("║ Interpretation:");
    println!("║   0/N   = full memorization; no generalization");
    println!("║   1-2/N = weak generalization; research-claim-relevant");
    println!("║   ≥3/N  = strong generalization on this corpus");
    println!("╚═════════════════════════════════════════════════════════");

    Ok(())
}
