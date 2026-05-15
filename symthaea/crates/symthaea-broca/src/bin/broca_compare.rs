// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! broca-compare: A/B comparison of CfC-HDC vs Liquid-Mamba generation.
//!
//! Loads an eval dataset, generates from both paths on the same inputs,
//! and produces a side-by-side quality comparison report.
//!
//! Usage:
//!   broca-compare --data eval.jsonl [--max-samples 20] [--genesis broca]
//!   broca-compare --data eval.jsonl --checkpoint broca-cfc.bin --projection broca-proj.bin
//!   broca-compare --data eval.jsonl --verbose

use std::process;
use std::time::Instant;

use symthaea_broca::evaluation::{english_word_ratio, english_word_ratio_mamba};
use symthaea_broca::generator::{BrocaConfig, BrocaGenerator};
use symthaea_broca::liquid_mamba::{LiquidMambaConfig, LiquidMambaGenerator};
use symthaea_broca::training::TrainingDataset;

use symthaea_core::genesis::GenesisSeed;

const INTENT_NAMES: [&str; 8] = [
    "Acknowledge",
    "Answer",
    "Clarify",
    "Propose",
    "Uncertainty",
    "Reflect",
    "Continue",
    "Unknown",
];

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".parse().unwrap()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let opts = match parse_args(&args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("Error: {e}");
            print_usage();
            process::exit(1);
        }
    };

    // Load dataset
    tracing::info!(path = %opts.data_path, "Loading evaluation data");
    let dataset = match TrainingDataset::from_jsonl(&opts.data_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to load data from '{}': {e}", opts.data_path);
            process::exit(1);
        }
    };

    if dataset.is_empty() {
        eprintln!("Dataset is empty");
        process::exit(1);
    }

    let sample_count = dataset.len().min(opts.max_samples);
    tracing::info!(
        total = dataset.len(),
        using = sample_count,
        "Dataset loaded"
    );

    let genesis = GenesisSeed::from_phrase(&opts.genesis);

    // --- CfC-HDC Generator ---
    tracing::info!("Initializing CfC-HDC generator");
    let mut cfc_gen = BrocaGenerator::new(&genesis, BrocaConfig::default());

    // Load checkpoint if provided
    if let Some(ref ckpt_path) = opts.cfc_checkpoint {
        tracing::info!(path = %ckpt_path, "Loading CfC-HDC checkpoint");
        match BrocaGenerator::from_checkpoint(ckpt_path, &genesis) {
            Ok((gen, _adam, _proj, _lm_config)) => {
                cfc_gen = gen;
                tracing::info!("CfC-HDC checkpoint loaded");
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to load CfC-HDC checkpoint, using random init");
            }
        }
    }

    // --- Liquid-Mamba Generator ---
    tracing::info!("Initializing Liquid-Mamba generator");
    let lm_config = LiquidMambaConfig {
        max_tokens: opts.max_gen_tokens,
        ..Default::default()
    };
    let mut lm_gen = match LiquidMambaGenerator::new(&genesis, lm_config) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed to create Liquid-Mamba generator: {e}");
            eprintln!("Ensure mamba-130m is available (will be downloaded on first use).");
            process::exit(1);
        }
    };

    // Load projection checkpoint if provided
    if let Some(ref proj_path) = opts.projection_checkpoint {
        tracing::info!(path = %proj_path, "Loading projection checkpoint");
        match symthaea_broca::checkpoint::ProjectionCheckpoint::load_from_file(proj_path) {
            Ok(ckpt) => {
                lm_gen
                    .projection_mut()
                    .load_weights(&ckpt.projection_weights);
                tracing::info!("Projection checkpoint loaded");
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to load projection checkpoint, using random init");
            }
        }
    }

    // --- Run A/B comparison ---
    println!("=== Broca A/B Comparison ===");
    println!("  Genesis:    {}", opts.genesis);
    println!("  Samples:    {sample_count}");
    println!("  Max tokens: {}", opts.max_gen_tokens);
    println!();

    let mut cfc_results = Vec::new();
    let mut lm_results = Vec::new();

    for (i, pair) in dataset.pairs.iter().take(sample_count).enumerate() {
        let channels = pair.to_thought_channels();
        let intent_idx = dominant_intent(&pair.channels);
        let intent_name = INTENT_NAMES[intent_idx];

        // CfC-HDC generation
        let cfc_start = Instant::now();
        let cfc_result = cfc_gen.generate(&channels);
        let cfc_elapsed = cfc_start.elapsed();

        // Liquid-Mamba generation
        let lm_start = Instant::now();
        let lm_result = lm_gen.generate(&channels);
        let lm_elapsed = lm_start.elapsed();

        let cfc_english = english_word_ratio(&cfc_result.token_ids, cfc_gen.tokenizer());
        let lm_english = english_word_ratio_mamba(&lm_result.token_ids, lm_gen.mamba());

        if opts.verbose {
            println!("--- Sample {}/{sample_count} [{intent_name}] ---", i + 1);
            if !pair.target_text.is_empty() {
                let target_preview: String = pair.target_text.chars().take(80).collect();
                println!("  Target:    {target_preview}...");
            }
            let cfc_preview: String = cfc_result.text.chars().take(80).collect();
            let lm_preview: String = lm_result.text.chars().take(80).collect();
            println!("  CfC-HDC:   {cfc_preview}");
            println!("  L-Mamba:   {lm_preview}");
            println!(
                "  CfC: {:.0}ms, eng={:.0}%, coh={:.3}, veto={}",
                cfc_elapsed.as_secs_f32() * 1000.0,
                cfc_english * 100.0,
                cfc_result.final_coherence,
                cfc_result.veto_triggered
            );
            println!(
                "  L-M: {:.0}ms, eng={:.0}%, coh={:.3}, pe={:.3}, veto={}",
                lm_elapsed.as_secs_f32() * 1000.0,
                lm_english * 100.0,
                lm_result.final_coherence,
                lm_result.semantic_pe,
                lm_result.veto_triggered
            );
            println!();
        }

        cfc_results.push(SampleResult {
            intent_idx,
            english_ratio: cfc_english,
            coherence: cfc_result.final_coherence,
            num_tokens: cfc_result.num_tokens,
            veto: cfc_result.veto_triggered,
            elapsed_ms: cfc_elapsed.as_secs_f32() * 1000.0,
        });

        lm_results.push(SampleResult {
            intent_idx,
            english_ratio: lm_english,
            coherence: lm_result.final_coherence,
            num_tokens: lm_result.num_tokens,
            veto: lm_result.veto_triggered,
            elapsed_ms: lm_elapsed.as_secs_f32() * 1000.0,
        });

        // Progress
        if (i + 1) % 10 == 0 || i + 1 == sample_count {
            tracing::info!(
                progress = i + 1,
                total = sample_count,
                "Comparison progress"
            );
        }
    }

    // --- Aggregate report ---
    print_comparison_report(&cfc_results, &lm_results);
}

struct SampleResult {
    intent_idx: usize,
    english_ratio: f32,
    coherence: f32,
    num_tokens: usize,
    veto: bool,
    elapsed_ms: f32,
}

fn dominant_intent(channels: &[f32]) -> usize {
    (0..8)
        .max_by(|&a, &b| channels[a].total_cmp(&channels[b]))
        .unwrap_or(7)
}

fn print_comparison_report(cfc: &[SampleResult], lm: &[SampleResult]) {
    let n = cfc.len() as f32;
    if n < 1.0 {
        println!("No samples to compare.");
        return;
    }

    let avg = |results: &[SampleResult], f: fn(&SampleResult) -> f32| -> f32 {
        results.iter().map(f).sum::<f32>() / n
    };

    let cfc_eng = avg(cfc, |r| r.english_ratio);
    let lm_eng = avg(lm, |r| r.english_ratio);
    let cfc_coh = avg(cfc, |r| r.coherence);
    let lm_coh = avg(lm, |r| r.coherence);
    let cfc_tok = avg(cfc, |r| r.num_tokens as f32);
    let lm_tok = avg(lm, |r| r.num_tokens as f32);
    let cfc_ms = avg(cfc, |r| r.elapsed_ms);
    let lm_ms = avg(lm, |r| r.elapsed_ms);
    let cfc_veto = cfc.iter().filter(|r| r.veto).count();
    let lm_veto = lm.iter().filter(|r| r.veto).count();

    println!("=== Comparison Report ({} samples) ===\n", cfc.len());
    println!(
        "{:<22} {:>12} {:>12} {:>10}",
        "Metric", "CfC-HDC", "L-Mamba", "Winner"
    );
    println!(
        "{:<22} {:>12} {:>12} {:>10}",
        "------", "-------", "-------", "------"
    );

    // English ratio
    let eng_winner = if lm_eng > cfc_eng + 0.01 {
        "L-Mamba"
    } else if cfc_eng > lm_eng + 0.01 {
        "CfC-HDC"
    } else {
        "Tie"
    };
    println!(
        "{:<22} {:>11.1}% {:>11.1}% {:>10}",
        "English ratio",
        cfc_eng * 100.0,
        lm_eng * 100.0,
        eng_winner
    );

    // Coherence
    let coh_winner = if lm_coh > cfc_coh + 0.01 {
        "L-Mamba"
    } else if cfc_coh > lm_coh + 0.01 {
        "CfC-HDC"
    } else {
        "Tie"
    };
    println!(
        "{:<22} {:>12.3} {:>12.3} {:>10}",
        "Avg coherence", cfc_coh, lm_coh, coh_winner
    );

    // Tokens
    println!(
        "{:<22} {:>12.1} {:>12.1} {:>10}",
        "Avg tokens", cfc_tok, lm_tok, ""
    );

    // Speed
    let speed_winner = if cfc_ms < lm_ms * 0.9 {
        "CfC-HDC"
    } else if lm_ms < cfc_ms * 0.9 {
        "L-Mamba"
    } else {
        "Tie"
    };
    println!(
        "{:<22} {:>10.0}ms {:>10.0}ms {:>10}",
        "Avg latency", cfc_ms, lm_ms, speed_winner
    );

    // Vetos
    println!(
        "{:<22} {:>12} {:>12} {:>10}",
        "Veto count", cfc_veto, lm_veto, ""
    );

    // Per-intent breakdown
    println!("\n--- Per-Intent Breakdown ---");
    println!(
        "{:<14} {:>8} {:>8} {:>8} {:>8}",
        "Intent", "CfC Eng%", "LM Eng%", "CfC Coh", "LM Coh"
    );
    println!(
        "{:<14} {:>8} {:>8} {:>8} {:>8}",
        "------", "--------", "-------", "-------", "------"
    );

    for (intent_idx, name) in INTENT_NAMES.iter().enumerate() {
        let cfc_intent: Vec<&SampleResult> =
            cfc.iter().filter(|r| r.intent_idx == intent_idx).collect();
        let lm_intent: Vec<&SampleResult> =
            lm.iter().filter(|r| r.intent_idx == intent_idx).collect();

        if cfc_intent.is_empty() {
            continue;
        }

        let cn = cfc_intent.len() as f32;
        let cfc_ie: f32 = cfc_intent.iter().map(|r| r.english_ratio).sum::<f32>() / cn;
        let lm_ie: f32 = lm_intent.iter().map(|r| r.english_ratio).sum::<f32>() / cn;
        let cfc_ic: f32 = cfc_intent.iter().map(|r| r.coherence).sum::<f32>() / cn;
        let lm_ic: f32 = lm_intent.iter().map(|r| r.coherence).sum::<f32>() / cn;

        println!(
            "{:<14} {:>7.1}% {:>7.1}% {:>8.3} {:>8.3}",
            format!("{name} ({})", cfc_intent.len()),
            cfc_ie * 100.0,
            lm_ie * 100.0,
            cfc_ic,
            lm_ic
        );
    }

    // Overall verdict
    println!("\n--- Verdict ---");
    let mut cfc_wins = 0u32;
    let mut lm_wins = 0u32;
    if cfc_eng > lm_eng + 0.01 {
        cfc_wins += 1;
    } else if lm_eng > cfc_eng + 0.01 {
        lm_wins += 1;
    }
    if cfc_coh > lm_coh + 0.01 {
        cfc_wins += 1;
    } else if lm_coh > cfc_coh + 0.01 {
        lm_wins += 1;
    }
    if cfc_ms < lm_ms * 0.9 {
        cfc_wins += 1;
    } else if lm_ms < cfc_ms * 0.9 {
        lm_wins += 1;
    }

    if cfc_wins > lm_wins {
        println!("  CfC-HDC leads ({cfc_wins} wins vs {lm_wins})");
    } else if lm_wins > cfc_wins {
        println!("  Liquid-Mamba leads ({lm_wins} wins vs {cfc_wins})");
    } else {
        println!("  Tied ({cfc_wins} wins each)");
    }
    println!("  Note: With untrained projections, CfC-HDC and L-Mamba produce");
    println!("  random tokens. Train both paths first for meaningful comparison.");
}

struct CompareOpts {
    data_path: String,
    max_samples: usize,
    max_gen_tokens: usize,
    genesis: String,
    cfc_checkpoint: Option<String>,
    projection_checkpoint: Option<String>,
    verbose: bool,
}

fn parse_args(args: &[String]) -> Result<CompareOpts, String> {
    let mut opts = CompareOpts {
        data_path: String::new(),
        max_samples: 20,
        max_gen_tokens: 64,
        genesis: "broca".to_string(),
        cfc_checkpoint: None,
        projection_checkpoint: None,
        verbose: false,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data" | "-d" => {
                i += 1;
                opts.data_path = args.get(i).cloned().ok_or("--data requires a path")?;
            }
            "--max-samples" | "-n" => {
                i += 1;
                opts.max_samples = args
                    .get(i)
                    .ok_or("--max-samples requires a number")?
                    .parse()
                    .map_err(|_| "--max-samples must be a positive integer")?;
            }
            "--max-tokens" => {
                i += 1;
                opts.max_gen_tokens = args
                    .get(i)
                    .ok_or("--max-tokens requires a number")?
                    .parse()
                    .map_err(|_| "--max-tokens must be a positive integer")?;
            }
            "--genesis" | "-g" => {
                i += 1;
                opts.genesis = args.get(i).cloned().ok_or("--genesis requires a phrase")?;
            }
            "--checkpoint" => {
                i += 1;
                opts.cfc_checkpoint =
                    Some(args.get(i).cloned().ok_or("--checkpoint requires a path")?);
            }
            "--projection" => {
                i += 1;
                opts.projection_checkpoint =
                    Some(args.get(i).cloned().ok_or("--projection requires a path")?);
            }
            "--verbose" | "-v" => {
                opts.verbose = true;
            }
            "--help" | "-h" => {
                print_usage();
                process::exit(0);
            }
            arg => return Err(format!("Unknown argument: {arg}")),
        }
        i += 1;
    }

    if opts.data_path.is_empty() {
        return Err("--data is required".to_string());
    }

    Ok(opts)
}

fn print_usage() {
    eprintln!("Usage: broca-compare [OPTIONS]");
    eprintln!();
    eprintln!("Compares CfC-HDC and Liquid-Mamba generation on the same inputs.");
    eprintln!();
    eprintln!("Required:");
    eprintln!("  --data, -d PATH      Evaluation JSONL file path");
    eprintln!();
    eprintln!("Optional:");
    eprintln!("  --max-samples, -n N  Max samples to compare (default: 20)");
    eprintln!("  --max-tokens N       Max tokens per generation (default: 64)");
    eprintln!("  --genesis, -g PHRASE Genesis seed phrase (default: broca)");
    eprintln!("  --checkpoint PATH    CfC-HDC checkpoint (.bin)");
    eprintln!("  --projection PATH    Projection checkpoint (.bin)");
    eprintln!("  --verbose, -v        Show per-sample generation output");
    eprintln!("  --help, -h           Show this help message");
}
