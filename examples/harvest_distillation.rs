// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Phase 2 M7 (harvester): generate (prompt, channels, code) training
//! pairs for Broca distillation by running the structural repair loop
//! over golden-backed subsets (Nix, HCL, Compose).

use std::io::Write;
use std::path::PathBuf;
use std::str::FromStr;

use symthaea::language::compose_eval_goldens::{COMPOSE_HARVEST_PROMPTS, compose_golden_for};
use symthaea::language::hcl_eval_goldens::{HCL_HARVEST_PROMPTS, hcl_golden_for};
use symthaea::language::nix_eval_goldens::{
    HARVEST_PROMPTS as NIX_PROMPTS, golden_for as nix_golden_for,
};
use symthaea::language::rust_eval_goldens::{RUST_HARVEST_PROMPTS, rust_golden_for};
use symthaea::language::substrate::{Substrate, generate_with_repair};

/// One training pair in on-disk form.
#[derive(serde::Serialize)]
struct DistillPair<'a> {
    substrate: Substrate,
    prompt: &'a str,
    channels: Vec<f32>,
    code: String,
    iterations: usize,
    repair_steps: usize,
    holdout: bool,
}

fn default_out_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home)
        .join(".cache")
        .join("symthaea")
        .join("distillation-pairs.jsonl")
}

fn parse_args() -> (PathBuf, Substrate, usize, Option<PathBuf>) {
    let args: Vec<String> = std::env::args().collect();
    let mut out = default_out_path();
    let mut substrate = Substrate::Nix;
    let mut holdout = 0;
    let mut prompt_file = None;

    for w in args.windows(2) {
        if w[0] == "--out" {
            out = PathBuf::from(&w[1]);
        } else if w[0] == "--substrate" {
            substrate = Substrate::from_str(&w[1]).expect("valid substrate");
        } else if w[0] == "--holdout" {
            holdout = w[1].parse().expect("valid holdout count");
        } else if w[0] == "--prompt-file" {
            prompt_file = Some(PathBuf::from(&w[1]));
        }
    }
    (out, substrate, holdout, prompt_file)
}

fn is_holdout(prompt: &str, holdout_count: usize, total: usize) -> bool {
    if holdout_count == 0 || holdout_count >= total {
        return false;
    }
    let mut h: u64 = 14695981039346656037; // FNV-1a offset
    for b in prompt.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    (h % total as u64) < holdout_count as u64
}

fn main() {
    let (out_path, substrate, holdout_count, prompt_file) = parse_args();
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let file = std::fs::File::create(&out_path).expect("cannot create output file");
    let mut writer = std::io::BufWriter::new(file);

    let default_prompts: Vec<String> = match substrate {
        Substrate::Nix => NIX_PROMPTS.iter().map(|s| s.to_string()).collect(),
        Substrate::Hcl => HCL_HARVEST_PROMPTS.iter().map(|s| s.to_string()).collect(),
        Substrate::Compose => COMPOSE_HARVEST_PROMPTS
            .iter()
            .map(|s| s.to_string())
            .collect(),
        Substrate::Rust => RUST_HARVEST_PROMPTS.iter().map(|s| s.to_string()).collect(),
        Substrate::Python => Vec::new(),
    };

    let mut external_prompts = Vec::new();
    if let Some(path) = prompt_file {
        let content = std::fs::read_to_string(path).expect("cannot read prompt file");
        for line in content.lines() {
            if !line.trim().is_empty() {
                external_prompts.push(line.trim().to_string());
            }
        }
    }

    let prompts: Vec<String> = if external_prompts.is_empty() {
        default_prompts
    } else {
        external_prompts
    };

    println!("┌─────────────────────────────────────────────────────────");
    println!("│ Multi-Substrate Distillation Harvester");
    println!("│ Substrate:  {}", substrate.name());
    println!("│ Output:     {}", out_path.display());
    println!("│ Candidates: {} prompts", prompts.len());
    println!("└─────────────────────────────────────────────────────────");

    let mut harvested = 0;
    for prompt in &prompts {
        let golden = match substrate {
            Substrate::Nix => nix_golden_for(prompt),
            Substrate::Hcl => hcl_golden_for(prompt),
            Substrate::Compose => compose_golden_for(prompt),
            Substrate::Rust => rust_golden_for(prompt),
            Substrate::Python => None,
        };

        let result = generate_with_repair(substrate, prompt, golden, 5);
        if !result.verdict.pass {
            println!("  ✗ [FAIL] {}", prompt);
            continue;
        }

        let holdout = is_holdout(prompt, holdout_count, prompts.len());
        let pair = DistillPair {
            substrate,
            prompt,
            channels: result.channels,
            code: result.code,
            iterations: result.iterations,
            repair_steps: result.repair_steps,
            holdout,
        };

        let line = serde_json::to_string(&pair).unwrap();
        writeln!(writer, "{}", line).unwrap();
        println!("  ✓ [{} iters] {}", result.iterations, prompt);
        harvested += 1;
    }

    writer.flush().unwrap();
    println!(
        "\nHarvested {}/{} pairs for {}",
        harvested,
        prompts.len(),
        substrate.name()
    );
}
