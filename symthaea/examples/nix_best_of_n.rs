// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Best-of-N selection over holdout generations, using rnix
//! parse-validity as the picker.
//!
//! Partial implementation of #1 on the coding-AI "make this even
//! better" list (rnix-gated decoding). True per-token gating would
//! mask invalid logits during sampling, but the broca generator
//! doesn't expose a per-token callback with rejection sampling — so
//! this is the post-hoc cousin: generate N candidates per prompt
//! (via `distill_nix_evaluate --samples N`), load them here, pick
//! the one with the longest parseable prefix.
//!
//! The 2026-04-19 multi-seed experiments showed large sampling
//! variance (44/105/55 bytes for the same checkpoint). If that
//! variance contains signal — which the outlier 105-byte prefix
//! suggests — best-of-N should produce a usable boost without any
//! retraining.
//!
//! Pipeline:
//! 1. `distill_nix_evaluate` writes N lines per prompt, each with a
//!    `sample_id` field.
//! 2. This example loads the N-sample JSONL, groups by prompt,
//!    scores each sample's generation via `longest_parseable_prefix`,
//!    and emits a one-line-per-prompt JSONL with the selected
//!    generation.
//! 3. Selected JSONL is a drop-in input to `nix_holdout_score` +
//!    `nix_prefix_parse_probe` for apples-to-apples comparison with
//!    single-sample baselines.
//!
//! Metrics printed:
//! - best prefix length vs single-sample median (variance harvest
//!   quantification)
//! - how many prompts have ANY parse-valid sample (could inform
//!   whether the full-parse 0/13 ceiling is sampling-variance-
//!   bounded vs architectural)
//!
//! Usage:
//!   # Phase 1 — generate N samples:
//!   LD_LIBRARY_PATH=/run/opengl-driver/lib \
//!   ./target/release/distill_nix_evaluate \
//!       --checkpoint <ckpt> --in <combined.jsonl> \
//!       --out /tmp/holdout-20samples.jsonl --samples 20
//!
//!   # Phase 2 — pick best per prompt:
//!   cargo run --features code_generation --release \
//!       --example nix_best_of_n -- \
//!       --in /tmp/holdout-20samples.jsonl \
//!       --out /tmp/holdout-bestof20.jsonl

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;

use symthaea::language::nix_scorer::attrpath_set_of;

#[derive(Debug, Deserialize)]
struct SampleRow {
    prompt: String,
    #[serde(default)]
    intent: String,
    golden: String,
    generated: String,
    #[serde(default)]
    generated_bytes: usize,
    #[serde(default)]
    sample_id: usize,
}

/// Matches distill_nix_evaluate's EvalRow shape exactly so downstream
/// tools (nix_holdout_score, nix_prefix_parse_probe) accept the output
/// without modification. sample_id is preserved as the winning index
/// for telemetry.
#[derive(Debug, Serialize)]
struct SelectedRow {
    prompt: String,
    intent: String,
    golden: String,
    generated: String,
    generated_bytes: usize,
    sample_id: usize,
}

fn parse_flag(flag: &str, default: Option<String>) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    for w in args.windows(2) {
        if w[0] == flag {
            return Some(w[1].clone());
        }
    }
    default
}

/// Longest byte-prefix of `s` that rnix parses without errors. Same
/// algorithm as `examples/nix_prefix_parse_probe.rs`, but returns
/// just the length. Linear downward scan from full length — fast
/// enough for ~250-byte outputs.
fn longest_parseable_prefix(s: &str) -> usize {
    for end in (1..=s.len()).rev() {
        if !s.is_char_boundary(end) {
            continue;
        }
        if rnix::Root::parse(&s[..end]).errors().is_empty() {
            return end;
        }
    }
    0
}

/// Score a single generation for "how well does it match the golden?".
/// Returns a tuple (primary, secondary) where primary orders samples
/// by the strongest signal first:
///   (full_parse_and_has_expected_path, longest_parseable_prefix)
/// - full_parse_and_has_expected_path = 1 if the whole generation
///   parses AND contains at least one path that matches a golden path
///   prefix (substring-match on the flattened paths). Else 0.
/// - longest_parseable_prefix = bytes, as tiebreaker.
fn score_sample(generated: &str, golden: &str) -> (u32, usize) {
    let longest = longest_parseable_prefix(generated);
    let full_parse_hit = if longest == generated.len() && !generated.is_empty() {
        // Check path overlap — prevents selecting an empty `{}` as
        // the "best" sample just because it parses.
        let gen_paths = attrpath_set_of(generated);
        let gold_paths = attrpath_set_of(golden);
        let overlap = gen_paths.iter().any(|gp| {
            gold_paths
                .iter()
                .any(|gold| gp == gold || gold.starts_with(&format!("{}.", gp)))
        });
        if overlap {
            1
        } else {
            0
        }
    } else {
        0
    };
    (full_parse_hit, longest)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let in_path = parse_flag("--in", None)
        .map(PathBuf::from)
        .ok_or("--in <path.jsonl> is required")?;
    let out_path = parse_flag("--out", None)
        .map(PathBuf::from)
        .ok_or("--out <path.jsonl> is required")?;

    println!("┌─────────────────────────────────────────────────────────");
    println!("│ Nix best-of-N picker (#1 partial — post-hoc rnix gating)");
    println!("│ Input:  {}", in_path.display());
    println!("│ Output: {}", out_path.display());
    println!("└─────────────────────────────────────────────────────────");

    // Load + group by prompt.
    let text = std::fs::read_to_string(&in_path)?;
    let mut by_prompt: BTreeMap<String, Vec<SampleRow>> = BTreeMap::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let row: SampleRow = serde_json::from_str(line)?;
        by_prompt.entry(row.prompt.clone()).or_default().push(row);
    }
    println!("Loaded {} prompts", by_prompt.len());
    let samples_per_prompt = by_prompt.values().next().map(|v| v.len()).unwrap_or(0);
    println!("Samples per prompt: {}\n", samples_per_prompt);

    // Per-prompt picker + telemetry.
    let mut selected: Vec<SelectedRow> = Vec::new();
    let mut per_prompt_gains: Vec<(String, usize, usize)> = Vec::new(); // (prompt, median, best)
    let mut full_parse_hits = 0usize;

    for (prompt, samples) in &by_prompt {
        let scored: Vec<(u32, usize, &SampleRow)> = samples
            .iter()
            .map(|s| {
                let (fp, pfx) = score_sample(&s.generated, &s.golden);
                (fp, pfx, s)
            })
            .collect();
        // Best by (full_parse_hit, prefix_length).
        let best = scored
            .iter()
            .max_by_key(|(fp, pfx, _)| (*fp, *pfx))
            .expect("at least one sample per prompt");

        // Median prefix across the N samples for gain measurement.
        let mut prefixes: Vec<usize> = scored.iter().map(|(_, p, _)| *p).collect();
        prefixes.sort_unstable();
        let median = prefixes[prefixes.len() / 2];

        per_prompt_gains.push((prompt.clone(), median, best.1));
        if best.0 == 1 {
            full_parse_hits += 1;
        }

        selected.push(SelectedRow {
            prompt: best.2.prompt.clone(),
            intent: best.2.intent.clone(),
            golden: best.2.golden.clone(),
            generated: best.2.generated.clone(),
            generated_bytes: best.2.generated_bytes,
            sample_id: best.2.sample_id,
        });
    }

    println!("─── Per-prompt prefix gain (median → best) ───────────────");
    for (p, med, best) in &per_prompt_gains {
        let gain = if *med > 0 {
            format!("{:+.0}%", 100.0 * (*best as f32 / *med as f32 - 1.0))
        } else {
            "N/A".to_string()
        };
        println!("  {:60} {:>3} → {:>3} bytes ({})", p, med, best, gain);
    }
    let total_median: usize = per_prompt_gains.iter().map(|(_, m, _)| m).sum();
    let total_best: usize = per_prompt_gains.iter().map(|(_, _, b)| b).sum();
    let avg_median = total_median as f32 / per_prompt_gains.len() as f32;
    let avg_best = total_best as f32 / per_prompt_gains.len() as f32;
    println!();
    println!("─── Aggregate ────────────────────────────────────────────");
    println!("  avg median prefix: {:.0} bytes", avg_median);
    println!("  avg best prefix:   {:.0} bytes", avg_best);
    let lift = if avg_median > 0.0 {
        format!("{:+.0}%", 100.0 * (avg_best / avg_median - 1.0))
    } else {
        "N/A".to_string()
    };
    println!("  best-of-N lift:    {}", lift);
    println!(
        "  full-parse-with-path-overlap: {}/{} prompts",
        full_parse_hits,
        per_prompt_gains.len()
    );
    println!();

    // Write selected rows.
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut w = std::io::BufWriter::new(std::fs::File::create(&out_path)?);
    for row in &selected {
        writeln!(w, "{}", serde_json::to_string(row)?)?;
    }
    w.flush()?;
    println!(
        "✓ Wrote {} selected rows to {}",
        selected.len(),
        out_path.display()
    );
    println!();
    println!("╔═════════════════════════════════════════════════════════");
    println!("║ Next: feed the selected JSONL to the existing scorers:");
    println!("║   cargo run --features code_generation --release \\");
    println!(
        "║       --example nix_holdout_score -- --in {}",
        out_path.display()
    );
    println!("║   cargo run --features code_generation --release \\");
    println!(
        "║       --example nix_prefix_parse_probe -- --in {}",
        out_path.display()
    );
    println!("╚═════════════════════════════════════════════════════════");

    Ok(())
}
