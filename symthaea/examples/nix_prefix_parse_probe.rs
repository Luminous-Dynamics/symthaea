// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Diagnostic for feasibility of rnix-gated decoding (#1 of the
//! "make this even better" list, probe phase).
//!
//! For each held-out generation, find the longest byte-prefix that
//! rnix-parses without errors. Report:
//! - longest parseable prefix length
//! - what the prefix contains
//!
//! Interpretation:
//! - 0 bytes for all: the model emits parse-invalid from token 1.
//!   No gating strategy can save it with the current checkpoint;
//!   training-data scale (#3) or structure-aware loss (#2) is the
//!   only path.
//! - ≥30 bytes for some: the model CAN produce valid prefixes; per-
//!   token rnix-gated decoding could plausibly extend those prefixes
//!   into full valid outputs by masking invalid-continuation tokens.
//! - Full parse for some: we'd already have non-zero hold-out pass
//!   and our 0/9 would be wrong.
//!
//! Usage:
//!   cargo run --features code_generation --example nix_prefix_parse_probe \
//!       -- --in /tmp/holdout-gen.jsonl

use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct EvalRow {
    prompt: String,
    #[serde(default)]
    intent: String,
    generated: String,
    #[serde(default)]
    generated_bytes: usize,
}

fn parse_in() -> PathBuf {
    let args: Vec<String> = std::env::args().collect();
    for w in args.windows(2) {
        if w[0] == "--in" {
            return PathBuf::from(&w[1]);
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home).join(".cache/symthaea/holdout-generated.jsonl")
}

/// Find the longest byte-prefix of `s` that parses cleanly under
/// rnix::Root::parse. Linear search from full length down — for
/// ~250-byte strings this is fast enough. Returns 0 if nothing
/// parses.
fn longest_parseable_prefix(s: &str) -> usize {
    for end in (1..=s.len()).rev() {
        // Must break on a char boundary.
        if !s.is_char_boundary(end) {
            continue;
        }
        let prefix = &s[..end];
        let parse = rnix::Root::parse(prefix);
        if parse.errors().is_empty() {
            return end;
        }
    }
    0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let in_path = parse_in();
    println!("┌─────────────────────────────────────────────────────────");
    println!("│ Nix prefix-parseable probe (#1 diagnostic)");
    println!("│ Input: {}", in_path.display());
    println!("└─────────────────────────────────────────────────────────");

    let text = std::fs::read_to_string(&in_path)?;
    let rows: Vec<EvalRow> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<EvalRow>(l).map_err(Box::<dyn std::error::Error>::from))
        .collect::<Result<_, _>>()?;

    let mut total_longest = 0usize;
    let mut nonzero_count = 0usize;
    let mut full_parse_count = 0usize;
    for (i, r) in rows.iter().enumerate() {
        let longest = longest_parseable_prefix(&r.generated);
        total_longest += longest;
        if longest > 0 {
            nonzero_count += 1;
        }
        if longest == r.generated.len() {
            full_parse_count += 1;
        }
        let ratio = if r.generated.is_empty() {
            0.0
        } else {
            longest as f32 / r.generated.len() as f32 * 100.0
        };
        println!(
            "  [{}/{:02}] {:50} → {:3}/{:3} bytes parse ({:.0}%)",
            i + 1,
            rows.len(),
            &r.prompt,
            longest,
            r.generated.len(),
            ratio
        );
        if longest > 0 && longest < 100 {
            let shown = r.generated[..longest].replace('\n', "\\n");
            println!("        prefix: {:?}", shown);
        }
    }

    let avg = if rows.is_empty() {
        0.0
    } else {
        total_longest as f32 / rows.len() as f32
    };
    println!("\n╔═════════════════════════════════════════════════════════");
    println!("║ Summary:");
    println!(
        "║   {}/{} generations have ANY parseable prefix",
        nonzero_count,
        rows.len()
    );
    println!("║   {}/{} parse fully", full_parse_count, rows.len());
    println!("║   avg longest parseable prefix: {:.0} bytes", avg);
    println!("╠═════════════════════════════════════════════════════════");
    println!("║ Verdict:");
    if nonzero_count == 0 {
        println!("║   NO parseable prefixes. Model emits garbage from token 1.");
        println!("║   Per-token rnix-gated decoding cannot recover from zero");
        println!("║   signal — need training-scale (#3) or structure-aware loss");
        println!("║   (#2) instead.");
    } else if avg < 10.0 {
        println!("║   Nearly-zero parseable prefixes (avg < 10 bytes).");
        println!("║   Gating could produce trivially-valid outputs (e.g. `{{}}`)");
        println!("║   but not useful Nix. Same verdict: need more data / structure");
        println!("║   loss.");
    } else if full_parse_count > 0 {
        println!("║   Some outputs ALREADY parse fully — would've passed the scorer");
        println!("║   if it weren't for value mismatches or missing required paths.");
        println!("║   Hold-out eval's 0/9 was strict structural scoring, not parse.");
    } else {
        println!("║   Non-trivial parseable prefixes exist. Per-token rnix-gated");
        println!("║   decoding has a real chance of extending them into full");
        println!("║   parseable outputs. #1 is worth the full implementation.");
    }
    println!("╚═════════════════════════════════════════════════════════");

    Ok(())
}
