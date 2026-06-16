// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! broca-collect: Distillation data collection from an Ollama LLM.
//!
//! Generates diverse ThoughtChannels covering the full intent × epistemic × emotion
//! combinatorial space, constructs system prompts, queries an Ollama model, and
//! records (channels, response) pairs as JSONL for Broca training.
//!
//! Multi-model mode: --model2 queries a second model for each thought, writing both
//! responses as separate training pairs. This produces diverse teacher distributions
//! and doubles the effective dataset size.
//!
//! Usage:
//!   broca-collect --model gemma4:e2b --output data/distillation.jsonl
//!   broca-collect --model gemma4:e2b --model2 mistral:7b --output data/distillation.jsonl
//!   broca-collect --model gemma4:e2b --output data/distillation.jsonl --repeats 3 --augment
//!   broca-collect --model gemma4:e2b --output data/distillation.jsonl --temperature 0.7

use std::io::Write;
use std::process;

use symthaea_broca::encoder::ThoughtChannels;
use symthaea_broca::tokenizer::BpeTokenizer;
use symthaea_broca::training::{TrainingPair, generate_diverse_thoughts, thought_to_prompt};

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

    let tokenizer = BpeTokenizer::default_4k();

    // Generate the full curriculum of diverse thoughts
    let base_thoughts = generate_diverse_thoughts();
    tracing::info!(
        configs = base_thoughts.len(),
        repeats = opts.repeats,
        total = base_thoughts.len() * opts.repeats,
        "Generating distillation data"
    );

    // Open output file
    let mut file = match std::fs::File::create(&opts.output_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to create output file '{}': {e}", opts.output_path);
            process::exit(1);
        }
    };

    let mut success_count = 0usize;
    let mut error_count = 0usize;
    let mut model2_success = 0usize;
    let mut model2_errors = 0usize;
    let mut intent_counts = [0usize; 8]; // Per-intent success counts
    let total = base_thoughts.len() * opts.repeats;
    let models: Vec<&str> = {
        let mut v = vec![opts.model.as_str()];
        if let Some(ref m2) = opts.model2 {
            v.push(m2.as_str());
        }
        v
    };

    if models.len() > 1 {
        tracing::info!(
            model1 = %models[0],
            model2 = %models[1],
            "Multi-model distillation enabled"
        );
    }

    for (idx, base_channels) in base_thoughts.iter().enumerate() {
        for repeat in 0..opts.repeats {
            let channels = if opts.augment && repeat > 0 {
                augment_channels(base_channels, repeat, opts.curriculum)
            } else {
                *base_channels
            };

            let prompt = thought_to_prompt(&channels);
            let system_prompt = build_system_prompt(&channels);
            // Query each model and write separate training pairs
            for (model_idx, model_name) in models.iter().enumerate() {
                let max_tokens = intent_max_tokens(&channels, model_name);
                match query_ollama(
                    &opts.ollama_url,
                    model_name,
                    &system_prompt,
                    &prompt,
                    opts.temperature,
                    max_tokens,
                ) {
                    Ok(response) => {
                        let response = response.trim().to_string();
                        if response.is_empty() {
                            if model_idx == 0 {
                                error_count += 1;
                            } else {
                                model2_errors += 1;
                            }
                            continue;
                        }

                        let pair = TrainingPair::new(channels, response, &tokenizer);
                        match serde_json::to_string(&pair) {
                            Ok(line) => {
                                if let Err(e) = writeln!(file, "{line}") {
                                    eprintln!("Write error: {e}");
                                    if model_idx == 0 {
                                        error_count += 1;
                                    } else {
                                        model2_errors += 1;
                                    }
                                } else {
                                    if model_idx == 0 {
                                        success_count += 1;
                                        let di = dominant_intent(&channels);
                                        intent_counts[di] += 1;
                                    } else {
                                        model2_success += 1;
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, model = %model_name, "Failed to serialize pair");
                                if model_idx == 0 {
                                    error_count += 1;
                                } else {
                                    model2_errors += 1;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            idx = idx,
                            repeat = repeat,
                            model = %model_name,
                            error = %e,
                            "Ollama query failed"
                        );
                        if model_idx == 0 {
                            error_count += 1;
                        } else {
                            model2_errors += 1;
                        }
                    }
                }
            }

            let progress = idx * opts.repeats + repeat + 1;
            if progress % 50 == 0 || progress == total {
                tracing::info!(
                    progress = progress,
                    total = total,
                    success = success_count + model2_success,
                    errors = error_count + model2_errors,
                    "Collection progress"
                );
            }
        }
    }

    // Flush
    if let Err(e) = file.flush() {
        eprintln!("Failed to flush output: {e}");
    }

    let intent_names = [
        "Acknowledge",
        "Answer",
        "Query",
        "Suggest",
        "Wonder",
        "Reflect",
        "Continue",
        "General",
    ];

    println!("\n--- Collection Summary ---");
    println!("  Model 1:       {}", opts.model);
    if let Some(ref m2) = opts.model2 {
        println!("  Model 2:       {m2}");
    }
    println!("  Total configs: {}", base_thoughts.len());
    println!("  Repeats:       {}", opts.repeats);
    println!(
        "  Curriculum:    {}",
        if opts.curriculum { "yes" } else { "no" }
    );
    println!("  Model 1 ok:    {success_count}");
    println!("  Model 1 err:   {error_count}");
    if opts.model2.is_some() {
        println!("  Model 2 ok:    {model2_success}");
        println!("  Model 2 err:   {model2_errors}");
        println!("  Total pairs:   {}", success_count + model2_success);
    }
    println!("  Output:        {}", opts.output_path);
    println!("\n  Per-intent breakdown (model 1):");
    for (i, name) in intent_names.iter().enumerate() {
        println!("    {name:<12} {}", intent_counts[i]);
    }
}

/// Build a system prompt appropriate for the thought's epistemic status and intent.
fn build_system_prompt(channels: &ThoughtChannels) -> String {
    let epistemic_idx = (channels.epistemic_ordinal() as usize).min(4);
    let epistemic_instruction = match epistemic_idx {
        0 => "Respond with certainty and directness.",
        1 => "Respond with reasonable confidence, noting that you believe this is likely correct.",
        2 => {
            "Respond while acknowledging some uncertainty. Use hedging language like 'perhaps', 'it seems', 'I believe'."
        }
        3 => {
            "Respond while clearly acknowledging you are uncertain. Use phrases like 'I'm not sure', 'it's unclear'."
        }
        _ => "Acknowledge this is outside your area of knowledge. Be honest about limitations.",
    };

    let intent_idx = (0..8)
        .max_by(|&a, &b| channels.channels[a].total_cmp(&channels.channels[b]))
        .unwrap_or(7);

    let intent_instruction = match intent_idx {
        0 => "Acknowledge the user's statement warmly.",
        1 => "Provide a direct answer to the implied question.",
        2 => "Ask a clarifying question or rephrase for clarity.",
        3 => "Propose a suggestion or course of action.",
        4 => "Express thoughtful uncertainty about the topic.",
        5 => "Reflect on the topic with depth and nuance.",
        6 => "Continue an ongoing train of thought naturally.",
        _ => "Respond naturally and helpfully.",
    };

    let warmth = channels.warmth();
    let tone = if warmth > 0.7 {
        "Use a warm, empathetic tone."
    } else if warmth < 0.3 {
        "Use a measured, professional tone."
    } else {
        "Use a balanced, friendly tone."
    };

    format!(
        "You are a thoughtful AI assistant. {epistemic_instruction} {intent_instruction} {tone}\n\
         Keep your response concise (1-3 sentences). Do not use markdown formatting."
    )
}

/// Add noise to channels for data augmentation.
///
/// In curriculum mode, noise scales up with repeat index (progressive difficulty):
/// repeat 0 = no noise, repeat N = N × base_noise.
fn augment_channels(base: &ThoughtChannels, repeat: usize, curriculum: bool) -> ThoughtChannels {
    let mut channels = *base;
    let noise_scale = if curriculum {
        // Progressive: 0.03 × repeat (e.g., 0.03, 0.06, 0.09, ...)
        repeat as f32 * 0.03
    } else {
        0.05 // Fixed augmentation noise
    };
    let seed = repeat as f32;
    for i in 8..20 {
        // Skip intent one-hot (0..8), perturb continuous channels
        let noise = ((i as f32 * 7.3 + seed * 13.7).sin()) * noise_scale;
        channels.channels[i] = (channels.channels[i] + noise).clamp(-1.0, 10.0);
    }
    channels
}

/// Per-intent target response length (num_predict tokens).
///
/// Short intents (acknowledge, query) get fewer tokens to avoid padding.
/// Long intents (reflect, continue) get more tokens for complete thoughts.
///
/// Models with thinking/reasoning mode (e.g., gemma4:e2b) consume ~250 tokens
/// on internal chain-of-thought before producing visible output. The `model`
/// parameter adds a thinking budget for such models.
fn intent_max_tokens(channels: &ThoughtChannels, model: &str) -> usize {
    let intent_idx = (0..8)
        .max_by(|&a, &b| channels.channels[a].total_cmp(&channels.channels[b]))
        .unwrap_or(7);
    let base = match intent_idx {
        0 => 64,  // Acknowledge — short and warm
        1 => 96,  // Answer — moderate detail
        2 => 80,  // Query — concise question
        3 => 96,  // Suggest — actionable proposal
        4 => 80,  // Wonder — brief uncertainty
        5 => 128, // Reflect — room for depth
        6 => 128, // Continue — extended thought
        _ => 96,  // Fallback
    };
    // Gemma 4 models use 300-400 thinking tokens before visible output.
    // Budget 500 to ensure all intents produce visible text.
    let thinking_budget = if model.contains("gemma4") { 500 } else { 0 };
    base + thinking_budget
}

/// Dominant intent index (0..7) from channels.
fn dominant_intent(channels: &ThoughtChannels) -> usize {
    (0..8)
        .max_by(|&a, &b| channels.channels[a].total_cmp(&channels.channels[b]))
        .unwrap_or(7)
}

/// Query Ollama's /api/generate endpoint.
fn query_ollama(
    base_url: &str,
    model: &str,
    system_prompt: &str,
    prompt: &str,
    temperature: f32,
    max_tokens: usize,
) -> Result<String, String> {
    let url = format!("{base_url}/api/generate");
    let body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": false,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    });

    let response = ureq::post(&url)
        .set("Content-Type", "application/json")
        .send_string(&body.to_string())
        .map_err(|e| format!("HTTP request failed: {e}"))?;

    let response_body = response
        .into_string()
        .map_err(|e| format!("Failed to read response: {e}"))?;

    let parsed: serde_json::Value =
        serde_json::from_str(&response_body).map_err(|e| format!("Failed to parse JSON: {e}"))?;

    parsed["response"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| "No 'response' field in Ollama output".to_string())
}

struct CollectOpts {
    model: String,
    /// Optional second model for multi-model distillation.
    model2: Option<String>,
    output_path: String,
    repeats: usize,
    augment: bool,
    temperature: f32,
    ollama_url: String,
    /// Curriculum mode: progressive augmentation + per-intent difficulty scaling.
    curriculum: bool,
}

fn parse_args(args: &[String]) -> Result<CollectOpts, String> {
    let mut opts = CollectOpts {
        model: String::new(),
        model2: None,
        output_path: String::new(),
        repeats: 1,
        augment: false,
        temperature: 0.7,
        ollama_url: "http://localhost:11434".to_string(),
        curriculum: false,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                i += 1;
                opts.model = args.get(i).cloned().ok_or("--model requires a value")?;
            }
            "--model2" => {
                i += 1;
                opts.model2 = Some(args.get(i).cloned().ok_or("--model2 requires a value")?);
            }
            "--output" | "-o" => {
                i += 1;
                opts.output_path = args.get(i).cloned().ok_or("--output requires a path")?;
            }
            "--repeats" => {
                i += 1;
                opts.repeats = args
                    .get(i)
                    .ok_or("--repeats requires a number")?
                    .parse()
                    .map_err(|_| "--repeats must be a positive integer")?;
            }
            "--augment" => {
                opts.augment = true;
            }
            "--curriculum" => {
                opts.curriculum = true;
                opts.augment = true; // curriculum implies augment
            }
            "--temperature" => {
                i += 1;
                opts.temperature = args
                    .get(i)
                    .ok_or("--temperature requires a number")?
                    .parse()
                    .map_err(|_| "--temperature must be a float")?;
            }
            "--ollama-url" => {
                i += 1;
                opts.ollama_url = args.get(i).cloned().ok_or("--ollama-url requires a URL")?;
            }
            "--help" | "-h" => {
                print_usage();
                process::exit(0);
            }
            arg => return Err(format!("Unknown argument: {arg}")),
        }
        i += 1;
    }

    if opts.model.is_empty() {
        return Err("--model is required".to_string());
    }
    if opts.output_path.is_empty() {
        return Err("--output is required".to_string());
    }

    Ok(opts)
}

fn print_usage() {
    eprintln!("Usage: broca-collect [OPTIONS]");
    eprintln!();
    eprintln!("Required:");
    eprintln!("  --model, -m NAME     Ollama model name (e.g., gemma4:e2b)");
    eprintln!("  --output, -o PATH    Output JSONL file path");
    eprintln!();
    eprintln!("Optional:");
    eprintln!("  --model2 NAME        Second model for multi-model distillation");
    eprintln!("  --repeats N          Repeat each config N times (default: 1)");
    eprintln!("  --augment            Add ±0.05 channel noise on repeat > 0");
    eprintln!("  --curriculum         Progressive augmentation (implies --augment)");
    eprintln!("  --temperature F      Ollama sampling temperature (default: 0.7)");
    eprintln!("  --ollama-url URL     Ollama API URL (default: http://localhost:11434)");
    eprintln!("  --help, -h           Show this help message");
}
