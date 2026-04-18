// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Corpus accept filter — turns scraper review queue(s) into harvester-
//! compatible training pairs. Mechanical filter, not a human review
//! replacement — pairs that survive this filter are *candidates* for
//! training, not automatically blessed.
//!
//! Intended use: preliminary retraining experiment that answers
//! "does ~50 pairs help at all vs. 26?" without the multi-week
//! human-review cycle. A positive signal motivates the full P5
//! review; a negative signal tells us grammar (not data volume) is
//! the 0/9 bottleneck.
//!
//! Filters applied (all hard-coded):
//! - drop `environment.variables` / `environment.shellAliases` /
//!   `environment.systemPackages` — current template prompts for these
//!   are weak ("install variables"); not useful until templates
//!   improve
//! - drop richness == 1 — trivial enable-only shape is already in the
//!   existing 26 pairs
//! - drop duplicate attrpaths (keeps highest richness)
//! - drop pairs whose prompt classifies as `NixIntent::Unknown` — no
//!   intent → garbage channel vector → no training signal
//! - sanity-check: golden must re-parse cleanly and the scorer must
//!   find the expected attrpath in it (rnix's own guarantee, but we
//!   verify before writing)
//!
//! Input: one or more review JSONL files (from nix_corpus_scrape).
//! Output: single JSONL matching `harvest_nix_distillation`'s
//! DistillPair shape, so the existing Broca trainer can ingest it
//! without code changes.
//!
//! Usage:
//!   cargo run --features code_generation --example nix_corpus_accept \
//!       -- --in /tmp/corpus-review.jsonl \
//!          --in /tmp/corpus-review-infra.jsonl \
//!          --out ~/.cache/symthaea/corpus-pairs.jsonl

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;

use symthaea::language::nix_broca_bridge::broca_channels_for_nix_prompt;
use symthaea::language::nix_codegen::{classify_nix_intent, NixIntent};
use symthaea::language::nix_scorer::attrpath_set_of;

/// Scraper output shape — kept in sync manually with nix_corpus_scrape's
/// CorpusCandidate.
#[derive(Debug, Deserialize)]
struct CorpusCandidate {
    prompt: String,
    golden: String,
    attrpath: String,
    source_file: String,
    #[serde(default)]
    source_line: usize,
    richness: usize,
}

/// Harvester-compatible training pair. Owned-string variant of the
/// reference-based `DistillPair` in `harvest_nix_distillation.rs`.
/// Trainer reads fields by name, not by type, so the owned version
/// serializes identically.
#[derive(Debug, Serialize)]
struct DistillPair {
    prompt: String,
    intent: String,
    channels: Vec<f32>,
    code: String,
    iterations: usize,
    repair_steps: usize,
    holdout: bool,
}

fn parse_ins() -> Vec<PathBuf> {
    let args: Vec<String> = std::env::args().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i + 1 < args.len() {
        if args[i] == "--in" {
            out.push(PathBuf::from(&args[i + 1]));
            i += 2;
        } else {
            i += 1;
        }
    }
    out
}

fn parse_out() -> PathBuf {
    let args: Vec<String> = std::env::args().collect();
    for w in args.windows(2) {
        if w[0] == "--out" {
            return PathBuf::from(&w[1]);
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home)
        .join(".cache")
        .join("symthaea")
        .join("corpus-pairs.jsonl")
}

/// Dropped-root blacklist. These appear in real configs but their
/// template prompts ("install variables", "install shellAliases") are
/// not useful training signal. Revisit when the template in
/// nix_corpus_scrape grows NLG for these shapes.
fn dropped_attrpath(path: &str) -> bool {
    matches!(
        path,
        "environment.variables"
            | "environment.shellAliases"
            | "environment.systemPackages"
            | "environment.sessionVariables"
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let inputs = parse_ins();
    if inputs.is_empty() {
        eprintln!("Usage: --in <path.jsonl> [--in ...] [--out <path>]");
        std::process::exit(1);
    }
    let out_path = parse_out();

    println!("┌─────────────────────────────────────────────────────────");
    println!("│ Symthaea corpus accept filter");
    for p in &inputs {
        println!("│ Input:  {}", p.display());
    }
    println!("│ Output: {}", out_path.display());
    println!("└─────────────────────────────────────────────────────────");

    // Load + dedupe.
    let mut by_attrpath: BTreeMap<String, CorpusCandidate> = BTreeMap::new();
    let mut total_in = 0usize;
    for p in &inputs {
        let text = std::fs::read_to_string(p)?;
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<CorpusCandidate>(line) {
                Ok(c) => {
                    total_in += 1;
                    // Prefer the richest entry per attrpath across all
                    // input files.
                    by_attrpath
                        .entry(c.attrpath.clone())
                        .and_modify(|existing| {
                            if c.richness > existing.richness {
                                *existing = CorpusCandidate {
                                    prompt: c.prompt.clone(),
                                    golden: c.golden.clone(),
                                    attrpath: c.attrpath.clone(),
                                    source_file: c.source_file.clone(),
                                    source_line: c.source_line,
                                    richness: c.richness,
                                };
                            }
                        })
                        .or_insert(c);
                }
                Err(e) => {
                    eprintln!("  × parse error: {e}");
                }
            }
        }
    }
    println!(
        "Loaded {} candidates, deduped to {} unique attrpaths",
        total_in,
        by_attrpath.len()
    );

    // Filter pipeline.
    let mut dropped_blacklist = 0usize;
    let mut dropped_richness = 0usize;
    let mut dropped_intent = 0usize;
    let mut dropped_attrpath_check = 0usize;
    let mut accepted: Vec<DistillPair> = Vec::new();

    for (_path, cand) in by_attrpath {
        if dropped_attrpath(&cand.attrpath) {
            dropped_blacklist += 1;
            continue;
        }
        if cand.richness < 2 {
            dropped_richness += 1;
            continue;
        }
        let intent = classify_nix_intent(&cand.prompt.to_lowercase());
        // Generic = classifier couldn't pin a specific intent from the
        // prompt. No training signal — skip.
        if matches!(intent, NixIntent::Generic) {
            dropped_intent += 1;
            continue;
        }
        // Sanity: the scorer must see the expected attrpath in the
        // golden. Guards against scraper bugs that emit a prompt
        // claiming X while the golden actually defines Y.
        let paths = attrpath_set_of(&cand.golden);
        let path_seen = paths
            .iter()
            .any(|p| p == &cand.attrpath || p.starts_with(&format!("{}.", cand.attrpath)));
        if !path_seen {
            dropped_attrpath_check += 1;
            continue;
        }

        let channels = broca_channels_for_nix_prompt(&cand.prompt);
        accepted.push(DistillPair {
            prompt: cand.prompt,
            intent: format!("{intent:?}"),
            channels: channels.to_vec(),
            code: cand.golden,
            iterations: 0, // from scraper: raw, no repair loop ran
            repair_steps: 0,
            holdout: false,
        });
    }

    println!("─── Filter results ───────────────────────────────────────");
    println!("  dropped (env.* blacklist):      {dropped_blacklist}");
    println!("  dropped (richness < 2):         {dropped_richness}");
    println!("  dropped (Unknown intent):       {dropped_intent}");
    println!("  dropped (attrpath sanity):      {dropped_attrpath_check}");
    println!("  accepted:                       {}", accepted.len());
    println!();

    // Intent distribution of accepted pairs — useful for gauging
    // balance (the existing 26 lean Service-heavy).
    let mut intent_hist: BTreeMap<String, usize> = BTreeMap::new();
    for p in &accepted {
        *intent_hist.entry(p.intent.clone()).or_insert(0) += 1;
    }
    println!("─── Accepted intent distribution ─────────────────────────");
    for (intent, count) in &intent_hist {
        println!("  {intent:>16} → {count}");
    }
    println!();

    // Write.
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut w = std::io::BufWriter::new(std::fs::File::create(&out_path)?);
    for p in &accepted {
        writeln!(w, "{}", serde_json::to_string(p)?)?;
    }
    w.flush()?;
    println!(
        "✓ Wrote {} training pairs to {}",
        accepted.len(),
        out_path.display()
    );
    println!();
    println!("╔═════════════════════════════════════════════════════════");
    println!("║ Next step: concat with existing distillation-pairs.jsonl");
    println!(
        "║ (26 from harvest_nix_distillation) to get a ~{}-pair corpus",
        26 + accepted.len()
    );
    println!("║ for a retraining experiment:");
    println!("║   cat ~/.cache/symthaea/distillation-pairs.jsonl \\");
    println!("║       {} \\", out_path.display());
    println!("║     > ~/.cache/symthaea/distillation-pairs-combined.jsonl");
    println!("╚═════════════════════════════════════════════════════════");

    Ok(())
}
