// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! RAG baseline + grounded compiler verdict (Option 2 foundation).
//!
//! Extends `nix_rag_holdout_baseline` by piping every generated output
//! through `nix-instantiate`. This gives us the TRUE semantic ceiling:
//! not "does it match the golden?" but "does nix actually accept it?".
//!
//! Three signals per prompt:
//! - **structural**: `nix_scorer::score(gen, golden).pass()` — matches
//!   golden attrpaths + values. Depends on having a golden.
//! - **parse**: `rnix::Root::parse(gen).errors().is_empty()` — syntactic.
//! - **grounded**: `cached_module_eval(gen)` → `NixVerdict::ParseOk` —
//!   nix-instantiate with full module evaluation succeeds. This is what
//!   the compiler says, independent of golden.
//!
//! A future RL training path would use the grounded signal as the
//! reward: every generation gets a deterministic pass/fail from the
//! actual nix toolchain. No cross-entropy guessing, no golden
//! dependency in production.
//!
//! Caveats:
//! - `nix-instantiate --parse` is fast (~5-50ms)
//! - full module eval (needed for services.X semantics) is slower
//!   (~200ms-2s per call) but cached via content-hash in
//!   `nix_eval_cache`. Subsequent runs with the same snippet hit cache.
//! - Some outputs look like modules but can't be evaluated in
//!   isolation (require external state). The grounded signal is
//!   conservative — it fires false-negatives, not false-positives.
//!
//! Usage:
//!   cargo run --features code_generation --release \
//!       --example nix_rag_grounded -- \
//!       --in ~/.cache/symthaea/combined-v2-58.jsonl

use serde::Deserialize;
use std::path::PathBuf;

use symthaea::language::nix_codegen::try_nix_module_eval;
use symthaea::language::nix_eval_cache::cached_module_eval;
use symthaea::language::nix_repair::generate_nix_with_self_repair;
use symthaea::language::nix_scorer::score;

#[derive(Debug, Deserialize)]
struct HoldoutPair {
    prompt: String,
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

/// Evaluate `code` via cached nix-instantiate. Returns Some(ok: bool,
/// message: String) if the snippet is shaped like a module and we got
/// a verdict; None if the snippet isn't module-shaped (can't
/// meaningfully eval in isolation).
fn grounded_eval(code: &str) -> Option<(bool, String)> {
    // Cache path first (content-hash keyed). If miss, fall through to
    // direct eval so the cache populates.
    if let Some(v) = cached_module_eval(code) {
        return Some((v.is_ok(), v.message()));
    }
    // Direct eval — also the miss-population path. try_nix_module_eval
    // returns None for non-module snippets.
    try_nix_module_eval(code).map(|v| (v.is_ok(), v.message()))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let in_path = parse_in();
    println!("┌─────────────────────────────────────────────────────────");
    println!("│ RAG + grounded compiler verdict (Option 2 foundation)");
    println!("│ Input: {}", in_path.display());
    println!("└─────────────────────────────────────────────────────────");

    let text = std::fs::read_to_string(&in_path)?;
    let pairs: Vec<HoldoutPair> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(Box::<dyn std::error::Error>::from))
        .collect::<Result<_, _>>()?;
    let holdouts: Vec<&HoldoutPair> = pairs.iter().filter(|p| p.holdout).collect();
    println!("Holdout prompts: {}\n", holdouts.len());

    let mut struct_pass = 0usize;
    let mut parse_ok = 0usize;
    let mut grounded_ok = 0usize;
    let mut grounded_na = 0usize;
    let mut grounded_fail_with_struct_pass = 0usize;

    println!("{:<55} | struct | parse | ground | nix-err", "prompt");
    println!("{}", "-".repeat(120));
    for p in &holdouts {
        let result = generate_nix_with_self_repair(&p.prompt, 5);
        let verdict = score(&result.code, &p.code);
        let struct_ok = verdict.pass();
        let syntax_ok = verdict.parse_error.is_none();
        let ground = grounded_eval(&result.code);

        if struct_ok {
            struct_pass += 1;
        }
        if syntax_ok {
            parse_ok += 1;
        }
        let (g_label, g_err) = match &ground {
            Some((true, _)) => {
                grounded_ok += 1;
                ("✓", String::new())
            }
            Some((false, msg)) => {
                if struct_ok {
                    grounded_fail_with_struct_pass += 1;
                }
                ("✗", msg.chars().take(60).collect())
            }
            None => {
                grounded_na += 1;
                ("-", String::new())
            }
        };
        let short = &p.prompt[..p.prompt.len().min(54)];
        println!(
            "{:<55} |   {}    |   {}   |   {}    | {}",
            short,
            if struct_ok { "✓" } else { "✗" },
            if syntax_ok { "✓" } else { "✗" },
            g_label,
            g_err
        );
    }

    println!("\n╔═════════════════════════════════════════════════════════");
    println!(
        "║ Structural pass: {}/{} ({:.0}%)",
        struct_pass,
        holdouts.len(),
        100.0 * struct_pass as f32 / holdouts.len() as f32
    );
    println!(
        "║ Parse-valid:     {}/{} ({:.0}%)",
        parse_ok,
        holdouts.len(),
        100.0 * parse_ok as f32 / holdouts.len() as f32
    );
    println!(
        "║ Grounded pass:   {}/{} ({:.0}% of N, {:.0}% of eligible)",
        grounded_ok,
        holdouts.len(),
        100.0 * grounded_ok as f32 / holdouts.len() as f32,
        if holdouts.len() == grounded_na {
            0.0
        } else {
            100.0 * grounded_ok as f32 / (holdouts.len() - grounded_na) as f32
        }
    );
    if grounded_na > 0 {
        println!(
            "║   ({} outputs not module-shaped — no grounded eval)",
            grounded_na
        );
    }
    if grounded_fail_with_struct_pass > 0 {
        println!(
            "║   {} outputs: struct-pass but nix-instantiate rejects",
            grounded_fail_with_struct_pass
        );
        println!("║     (structural scorer over-approves — the compiler is stricter)");
    }
    println!("╠═════════════════════════════════════════════════════════");
    println!("║ Interpretation:");
    println!("║   struct-pass  = matches the golden shape");
    println!("║   parse-valid  = syntactically well-formed Nix");
    println!("║   grounded-ok  = nix-instantiate accepts the snippet");
    println!("║");
    println!("║ The grounded signal is Option 2's RL reward candidate.");
    println!("║ It needs no golden, runs on every generation, and is");
    println!("║ harder to game than cross-entropy.");
    println!("╚═════════════════════════════════════════════════════════");

    Ok(())
}
