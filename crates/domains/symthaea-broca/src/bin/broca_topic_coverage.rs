// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! broca-topic-coverage: does a candidate checkpoint actually know the new material?
//!
//! Consumes a `.holdout.jsonl` file written by `broca-curriculum-sync`
//! (one text per curriculum objective, deliberately never included in
//! training) and generates the candidate checkpoint's own articulation of
//! each held-out objective's channels, side by side with the reference
//! text an LLM wrote for it.
//!
//! This exists because the checkpoint-compare regression gate
//! (`broca-checkpoint-compare`) only proves the candidate didn't get worse
//! on drift/hallucination/structured-validity metrics — it has no signal
//! for whether newly-learned material actually landed. This report gives
//! a human reviewing PROMOTION_READY.json concrete, readable evidence
//! ("here's what she can now say about X") instead of only abstract
//! regression-absence numbers.
//!
//! Usage:
//!   broca-topic-coverage --checkpoint candidate.bin --holdout batch.holdout.jsonl --json-out report.json

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::process;

use symthaea_broca::encoder::{NUM_CHANNELS, ThoughtChannels};
use symthaea_broca::generator::BrocaGenerator;
use symthaea_core::genesis::GenesisSeed;

#[derive(Debug, Clone, Deserialize)]
struct HoldoutEntry {
    objective_id: String,
    channels: Vec<f32>,
    reference_text: String,
}

#[derive(Debug, Serialize)]
struct CoverageCase {
    objective_id: String,
    reference_text: String,
    generated_text: String,
    generated_coherence: f32,
    veto_triggered: bool,
    /// Cheap, non-authoritative signal: fraction of the reference text's
    /// significant (>3 char) words that also appear in the generated text.
    /// Not a quality score — just a quick skim aid; the reference/generated
    /// pair is what a human should actually read.
    keyword_overlap: f32,
}

#[derive(Debug, Serialize)]
struct CoverageReport {
    schema_version: u32,
    checkpoint_path: String,
    num_objectives: usize,
    avg_keyword_overlap: f32,
    avg_generated_coherence: f32,
    veto_count: usize,
    cases: Vec<CoverageCase>,
}

struct Options {
    checkpoint_path: String,
    holdout_path: String,
    json_out: Option<String>,
    genesis_phrase: String,
    max_gen_tokens: usize,
    allow_checkpoint_recovery: bool,
}

fn parse_args(args: &[String]) -> Result<Options> {
    let mut checkpoint_path = None;
    let mut holdout_path = None;
    let mut json_out = None;
    let mut genesis_phrase = "symthaea luminous dynamics".to_string();
    let mut max_gen_tokens = 48;
    let mut allow_checkpoint_recovery = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--checkpoint" => {
                i += 1;
                checkpoint_path = Some(value(args, i, "--checkpoint")?.to_string());
            }
            "--holdout" => {
                i += 1;
                holdout_path = Some(value(args, i, "--holdout")?.to_string());
            }
            "--json-out" => {
                i += 1;
                json_out = Some(value(args, i, "--json-out")?.to_string());
            }
            "--genesis" => {
                i += 1;
                genesis_phrase = value(args, i, "--genesis")?.to_string();
            }
            "--max-gen-tokens" => {
                i += 1;
                max_gen_tokens = value(args, i, "--max-gen-tokens")?.parse()?;
            }
            "--allow-checkpoint-recovery" => {
                allow_checkpoint_recovery = true;
            }
            other => anyhow::bail!("unknown argument: {other}"),
        }
        i += 1;
    }

    Ok(Options {
        checkpoint_path: checkpoint_path.context("--checkpoint is required")?,
        holdout_path: holdout_path.context("--holdout is required")?,
        json_out,
        genesis_phrase,
        max_gen_tokens,
        allow_checkpoint_recovery,
    })
}

fn value<'a>(args: &'a [String], index: usize, flag: &str) -> Result<&'a str> {
    args.get(index)
        .map(|s| s.as_str())
        .with_context(|| format!("{flag} requires a value"))
}

fn channels_from_raw(raw: &[f32]) -> ThoughtChannels {
    let mut ch = ThoughtChannels::default();
    let n = raw.len().min(NUM_CHANNELS);
    ch.channels[..n].copy_from_slice(&raw[..n]);
    ch
}

fn keyword_overlap(reference: &str, generated: &str) -> f32 {
    let generated_lower = generated.to_lowercase();
    let words: Vec<&str> = reference
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.len() > 3)
        .collect();
    if words.is_empty() {
        return 0.0;
    }
    let hits = words
        .iter()
        .filter(|w| generated_lower.contains(&w.to_lowercase()))
        .count();
    hits as f32 / words.len() as f32
}

fn run(opts: Options) -> Result<()> {
    let holdout_contents = std::fs::read_to_string(&opts.holdout_path)
        .with_context(|| format!("reading holdout file {}", opts.holdout_path))?;
    let entries: Vec<HoldoutEntry> = holdout_contents
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| serde_json::from_str(line).context("parsing holdout entry"))
        .collect::<Result<Vec<_>>>()?;

    let genesis = GenesisSeed::from_phrase(&opts.genesis_phrase);
    let load_result = if opts.allow_checkpoint_recovery {
        BrocaGenerator::from_checkpoint_allow_checksum_mismatch(&opts.checkpoint_path, &genesis)
    } else {
        BrocaGenerator::from_checkpoint(&opts.checkpoint_path, &genesis)
    };
    let (mut generator, _, _, _) =
        load_result.with_context(|| format!("loading checkpoint {}", opts.checkpoint_path))?;
    generator.config_mut().gating.base_max_tokens = opts.max_gen_tokens;
    generator.config_mut().bypass_gating = true;

    let mut cases = Vec::with_capacity(entries.len());
    for entry in &entries {
        let channels = channels_from_raw(&entry.channels);
        let result = generator.generate(&channels);
        cases.push(CoverageCase {
            objective_id: entry.objective_id.clone(),
            reference_text: entry.reference_text.clone(),
            generated_text: result.text.clone(),
            generated_coherence: result.final_coherence,
            veto_triggered: result.veto_triggered,
            keyword_overlap: keyword_overlap(&entry.reference_text, &result.text),
        });
    }

    let num_objectives = cases.len();
    let avg_keyword_overlap = if num_objectives > 0 {
        cases.iter().map(|c| c.keyword_overlap).sum::<f32>() / num_objectives as f32
    } else {
        0.0
    };
    let avg_generated_coherence = if num_objectives > 0 {
        cases.iter().map(|c| c.generated_coherence).sum::<f32>() / num_objectives as f32
    } else {
        0.0
    };
    let veto_count = cases.iter().filter(|c| c.veto_triggered).count();

    let report = CoverageReport {
        schema_version: 1,
        checkpoint_path: opts.checkpoint_path.clone(),
        num_objectives,
        avg_keyword_overlap,
        avg_generated_coherence,
        veto_count,
        cases,
    };

    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = &opts.json_out {
        std::fs::write(path, json)?;
    } else {
        println!("{json}");
    }

    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let opts = match parse_args(&args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("Error: {e}");
            eprintln!(
                "Usage: broca-topic-coverage --checkpoint <path> --holdout <path> --json-out <path>"
            );
            process::exit(1);
        }
    };
    if let Err(e) = run(opts) {
        eprintln!("Error: {e:#}");
        process::exit(1);
    }
}
