// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! NixEval Benchmark — 94 prompts, legacy-substring and structural scorers.
//!
//! Each problem specifies:
//! - prompt: natural language request
//! - expected_intent: what System 1 should classify as
//! - expected_substrings: must appear in the generated Nix (legacy scorer)
//! - forbidden_substrings: must NOT appear (catches wrong idioms)
//! - require_parse: whether nix-instantiate --parse must succeed
//!
//! The legacy four-way check (intent + parse + expected + forbidden) admits
//! false positives — e.g. `services.postgresql.enable = false; # pgvector
//! needed` passes when the required substrings are `postgresql`, `enable`,
//! `pgvector`. The `--structural` flag activates the AST-based scorer in
//! `language::nix_scorer`, which compares against golden references from
//! `language::nix_eval_goldens::golden_for`. Goldens are backfilled
//! incrementally; problems without a golden fall through to substring.
//!
//! Usage:
//!   cargo run --example nix_eval_benchmark --features code_generation
//!   cargo run --example nix_eval_benchmark --features code_generation -- --structural

use symthaea::language::nix_codegen::{generate_nix, NixIntent};
use symthaea::language::nix_eval_corpus::{problems, NixProblem};
use symthaea::language::nix_eval_goldens::{golden_count, golden_for, score_all_goldens};
use symthaea::language::nix_scorer::score as structural_score;
#[derive(Default)]
struct ScoreCard {
    total: usize,
    intent_correct: usize,
    parse_correct: usize,
    expected_present: usize,
    forbidden_absent: usize,
    full_pass: usize,
    by_intent: std::collections::HashMap<NixIntent, (usize, usize)>,

    // Structural-mode counters (only populated when --structural active).
    structural_scored: usize, // problems that had a golden + got scored
    structural_pass: usize,   // structural scorer said PASS
    structural_fallback_pass: usize, // no golden, fell through to substring and passed
    structural_fallback_total: usize,
}

/// Legacy four-way check. Substring-based; known-loose.
fn evaluate(problem: &NixProblem) -> (bool, bool, bool, bool, String) {
    let result = generate_nix(problem.prompt);
    let intent_ok = result.intent == problem.expected_intent;
    let parse_ok = if problem.require_parse {
        result.parses
    } else {
        true
    };

    let expected_ok = problem
        .expected_substrings
        .iter()
        .all(|s| result.code.contains(s));

    let forbidden_ok = problem
        .forbidden_substrings
        .iter()
        .all(|s| !result.code.contains(s));

    (intent_ok, parse_ok, expected_ok, forbidden_ok, result.code)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let structural_mode = args.iter().any(|a| a == "--structural");
    let goldens_only = args.iter().any(|a| a == "--goldens-only");

    // --goldens-only: fast subset run (6 prompts, seconds). Useful for
    // smoke-testing the scorer plumbing without paying the full 95-prompt
    // cost. Returns before loading the full corpus.
    if goldens_only {
        println!("┌─────────────────────────────────────────────────────────");
        println!(
            "│ NixEval Benchmark — GOLDENS-ONLY ({} prompts)",
            golden_count()
        );
        println!("│ Structural AST scorer on golden-backed subset");
        println!("└─────────────────────────────────────────────────────────");
        let results = score_all_goldens();
        let mut pass = 0;
        for (prompt, passed, summary) in &results {
            let mark = if *passed { "✓" } else { "✗" };
            if *passed {
                pass += 1;
            }
            println!("  {} {:60} | {}", mark, prompt, summary);
        }
        println!("\n╔═════════════════════════════════════════════════════════");
        println!(
            "║ Goldens-only pass: {}/{} ({:.0}%)",
            pass,
            results.len(),
            pass as f32 / results.len() as f32 * 100.0
        );
        println!("╚═════════════════════════════════════════════════════════");
        std::process::exit(if pass == results.len() { 0 } else { 1 });
    }

    let problems = problems();
    let mut card = ScoreCard::default();
    card.total = problems.len();

    println!("┌─────────────────────────────────────────────────────────");
    println!(
        "│ NixEval Benchmark — {} problems ({} with goldens)",
        problems.len(),
        golden_count()
    );
    println!(
        "│ Mode: {}",
        if structural_mode {
            "STRUCTURAL (AST) + legacy substring diff"
        } else {
            "LEGACY substring only (run with --structural to activate AST scorer)"
        }
    );
    println!("└─────────────────────────────────────────────────────────");

    for (i, p) in problems.iter().enumerate() {
        let (intent_ok, parse_ok, exp_ok, forbid_ok, code) = evaluate(p);

        if intent_ok {
            card.intent_correct += 1;
        }
        if parse_ok {
            card.parse_correct += 1;
        }
        if exp_ok {
            card.expected_present += 1;
        }
        if forbid_ok {
            card.forbidden_absent += 1;
        }
        let full = intent_ok && parse_ok && exp_ok && forbid_ok;
        if full {
            card.full_pass += 1;
        }

        // Structural scoring runs in parallel when enabled and a golden
        // exists. Absence of a golden is tracked as fallback-to-substring.
        if structural_mode {
            match golden_for(p.prompt) {
                Some(golden) => {
                    card.structural_scored += 1;
                    let verdict = structural_score(&code, golden);
                    if verdict.pass() && intent_ok {
                        card.structural_pass += 1;
                    } else if i < 20 {
                        // Print structural diagnostics for the first 20
                        // failures (keeps output readable on large corpora).
                        println!("       structural: {}", verdict.summary());
                        for m in verdict.value_mismatches.iter().take(3) {
                            println!(
                                "         mismatch {}: got {} want {}",
                                m.path,
                                m.got.display(),
                                m.want.display()
                            );
                        }
                        for miss in verdict.missing_required.iter().take(3) {
                            println!("         missing:  {miss}");
                        }
                    }
                }
                None => {
                    card.structural_fallback_total += 1;
                    if full {
                        card.structural_fallback_pass += 1;
                    }
                }
            }
        }

        let entry = card.by_intent.entry(p.expected_intent).or_insert((0, 0));
        entry.1 += 1;
        if full {
            entry.0 += 1;
        }

        let mark = if full { "✓" } else { "✗" };
        let detail = format!(
            "intent={} parse={} expected={} forbidden={}",
            if intent_ok { "✓" } else { "✗" },
            if parse_ok { "✓" } else { "✗" },
            if exp_ok { "✓" } else { "✗" },
            if forbid_ok { "✓" } else { "✗" },
        );
        println!("  {} #{:02} {:50} | {}", mark, i + 1, p.prompt, detail);

        if !full {
            // Show what was missing for diagnostics
            for s in p.expected_substrings.iter() {
                if !code.contains(s) {
                    println!("       missing: {s}");
                }
            }
            for s in p.forbidden_substrings.iter() {
                if code.contains(s) {
                    println!("       leaked:  {s}");
                }
            }
        }
    }

    let n = card.total as f32;
    println!("\n╔═════════════════════════════════════════════════════════");
    println!(
        "║ Intent classification:  {}/{} ({:.0}%)",
        card.intent_correct,
        card.total,
        card.intent_correct as f32 / n * 100.0
    );
    println!(
        "║ Parses successfully:    {}/{} ({:.0}%)",
        card.parse_correct,
        card.total,
        card.parse_correct as f32 / n * 100.0
    );
    println!(
        "║ Expected substrings:    {}/{} ({:.0}%)",
        card.expected_present,
        card.total,
        card.expected_present as f32 / n * 100.0
    );
    println!(
        "║ No forbidden leakage:   {}/{} ({:.0}%)",
        card.forbidden_absent,
        card.total,
        card.forbidden_absent as f32 / n * 100.0
    );
    println!(
        "║ FULL PASS (all 4):      {}/{} ({:.0}%)",
        card.full_pass,
        card.total,
        card.full_pass as f32 / n * 100.0
    );
    println!("╚═════════════════════════════════════════════════════════");

    println!("\nPer-intent full pass rate:");
    let mut intents: Vec<_> = card.by_intent.iter().collect();
    intents.sort_by_key(|(k, _)| format!("{:?}", k));
    for (intent, (passed, total)) in intents {
        println!(
            "  {:?}: {}/{} ({:.0}%)",
            intent,
            passed,
            total,
            *passed as f32 / *total as f32 * 100.0
        );
    }

    if structural_mode {
        println!("\n╔═════════════════════════════════════════════════════════");
        println!("║ STRUCTURAL SCORER REPORT (gold-backed subset)");
        println!("║");
        if card.structural_scored == 0 {
            println!("║ No goldens found for any prompt. Backfill some via");
            println!("║ `language::nix_eval_goldens::golden_for`.");
        } else {
            println!(
                "║ Golden-backed problems:  {}/{}",
                card.structural_scored, card.total
            );
            println!(
                "║ Structural PASS:         {}/{} ({:.0}%)",
                card.structural_pass,
                card.structural_scored,
                card.structural_pass as f32 / card.structural_scored as f32 * 100.0
            );
            println!(
                "║ Legacy fallback pass:    {}/{} (ungolded problems via substring)",
                card.structural_fallback_pass, card.structural_fallback_total
            );
            println!("║");
            println!("║ Structural PASS is the publishable honest number on the");
            println!("║ subset that has goldens. Substring fallback is legacy and");
            println!("║ overcounts — see nix_scorer docstring.");
        }
        println!("╚═════════════════════════════════════════════════════════");
    }
}
