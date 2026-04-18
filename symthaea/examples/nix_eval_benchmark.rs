// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! NixEval Benchmark — 20 prompts with structural-correctness assertions.
//!
//! Each problem specifies:
//! - prompt: natural language request
//! - expected_intent: what System 1 should classify as
//! - expected_substrings: must appear in the generated Nix
//! - forbidden_substrings: must NOT appear (catches wrong idioms)
//! - require_parse: whether nix-instantiate --parse must succeed
//!
//! Honest measurement of the hybrid pipeline on a curated test set.
//!
//! Run:
//!   cargo run --example nix_eval_benchmark --features code_generation

use symthaea::language::nix_codegen::{generate_nix, NixIntent};
use symthaea::language::nix_eval_corpus::{problems, NixProblem};
#[derive(Default)]
struct ScoreCard {
    total: usize,
    intent_correct: usize,
    parse_correct: usize,
    expected_present: usize,
    forbidden_absent: usize,
    full_pass: usize,
    by_intent: std::collections::HashMap<NixIntent, (usize, usize)>,
}

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
    let problems = problems();
    let mut card = ScoreCard::default();
    card.total = problems.len();

    println!("┌─────────────────────────────────────────────────────────");
    println!("│ NixEval Benchmark — {} problems", problems.len());
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
}
