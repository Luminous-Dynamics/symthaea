//! broca-collect: Distillation data collection from an Ollama LLM.
//!
//! Generates diverse ThoughtChannels covering the full intent × epistemic × emotion
//! combinatorial space, constructs system prompts, queries an Ollama model, and
//! records (channels, response) pairs as JSONL for Broca training.
//!
//! Usage:
//!   broca-collect --model gemma3:1b --output data/distillation.jsonl
//!   broca-collect --model gemma3:1b --output data/distillation.jsonl --repeats 3 --augment
//!   broca-collect --model gemma3:1b --output data/distillation.jsonl --temperature 0.7

use std::io::Write;
use std::process;

use symthaea_broca::encoder::ThoughtChannels;
use symthaea_broca::training::{generate_diverse_thoughts, thought_to_prompt, TrainingPair};
use symthaea_broca::tokenizer::BpeTokenizer;

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
    let total = base_thoughts.len() * opts.repeats;

    for (idx, base_channels) in base_thoughts.iter().enumerate() {
        for repeat in 0..opts.repeats {
            let channels = if opts.augment && repeat > 0 {
                augment_channels(base_channels, repeat)
            } else {
                *base_channels
            };

            let prompt = thought_to_prompt(&channels);
            let system_prompt = build_system_prompt(&channels);

            match query_ollama(&opts.ollama_url, &opts.model, &system_prompt, &prompt, opts.temperature) {
                Ok(response) => {
                    let response = response.trim().to_string();
                    if response.is_empty() {
                        error_count += 1;
                        continue;
                    }

                    let pair = TrainingPair::new(channels, response, &tokenizer);
                    match serde_json::to_string(&pair) {
                        Ok(line) => {
                            if let Err(e) = writeln!(file, "{line}") {
                                eprintln!("Write error: {e}");
                                error_count += 1;
                            } else {
                                success_count += 1;
                            }
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "Failed to serialize pair");
                            error_count += 1;
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        idx = idx,
                        repeat = repeat,
                        error = %e,
                        "Ollama query failed"
                    );
                    error_count += 1;
                }
            }

            let progress = idx * opts.repeats + repeat + 1;
            if progress % 50 == 0 || progress == total {
                tracing::info!(
                    progress = progress,
                    total = total,
                    success = success_count,
                    errors = error_count,
                    "Collection progress"
                );
            }
        }
    }

    // Flush
    if let Err(e) = file.flush() {
        eprintln!("Failed to flush output: {e}");
    }

    println!("\n--- Collection Summary ---");
    println!("  Total configs: {}", base_thoughts.len());
    println!("  Repeats:       {}", opts.repeats);
    println!("  Successful:    {success_count}");
    println!("  Errors:        {error_count}");
    println!("  Output:        {}", opts.output_path);
}

/// Build a system prompt appropriate for the thought's epistemic status and intent.
fn build_system_prompt(channels: &ThoughtChannels) -> String {
    let epistemic_idx = (channels.epistemic_ordinal() as usize).min(4);
    let epistemic_instruction = match epistemic_idx {
        0 => "Respond with certainty and directness.",
        1 => "Respond with reasonable confidence, noting that you believe this is likely correct.",
        2 => "Respond while acknowledging some uncertainty. Use hedging language like 'perhaps', 'it seems', 'I believe'.",
        3 => "Respond while clearly acknowledging you are uncertain. Use phrases like 'I'm not sure', 'it's unclear'.",
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

/// Add small noise to channels for data augmentation.
fn augment_channels(base: &ThoughtChannels, repeat: usize) -> ThoughtChannels {
    let mut channels = *base;
    // Deterministic noise based on repeat index
    let seed = repeat as f32 * 0.05;
    for i in 8..20 {
        // Skip intent one-hot (0..8), perturb continuous channels
        let noise = ((i as f32 * 7.3 + seed * 13.7).sin()) * 0.05;
        channels.channels[i] = (channels.channels[i] + noise).clamp(-1.0, 10.0);
    }
    channels
}

/// Query Ollama's /api/generate endpoint.
fn query_ollama(
    base_url: &str,
    model: &str,
    system_prompt: &str,
    prompt: &str,
    temperature: f32,
) -> Result<String, String> {
    let url = format!("{base_url}/api/generate");
    let body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": false,
        "options": {
            "temperature": temperature,
            "num_predict": 128,
        }
    });

    let response = ureq::post(&url)
        .set("Content-Type", "application/json")
        .send_string(&body.to_string())
        .map_err(|e| format!("HTTP request failed: {e}"))?;

    let response_body = response
        .into_string()
        .map_err(|e| format!("Failed to read response: {e}"))?;

    let parsed: serde_json::Value = serde_json::from_str(&response_body)
        .map_err(|e| format!("Failed to parse JSON: {e}"))?;

    parsed["response"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| "No 'response' field in Ollama output".to_string())
}

struct CollectOpts {
    model: String,
    output_path: String,
    repeats: usize,
    augment: bool,
    temperature: f32,
    ollama_url: String,
}

fn parse_args(args: &[String]) -> Result<CollectOpts, String> {
    let mut opts = CollectOpts {
        model: String::new(),
        output_path: String::new(),
        repeats: 1,
        augment: false,
        temperature: 0.7,
        ollama_url: "http://localhost:11434".to_string(),
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                i += 1;
                opts.model = args.get(i).cloned().ok_or("--model requires a value")?;
            }
            "--output" | "-o" => {
                i += 1;
                opts.output_path = args.get(i).cloned().ok_or("--output requires a path")?;
            }
            "--repeats" => {
                i += 1;
                opts.repeats = args.get(i)
                    .ok_or("--repeats requires a number")?
                    .parse()
                    .map_err(|_| "--repeats must be a positive integer")?;
            }
            "--augment" => {
                opts.augment = true;
            }
            "--temperature" => {
                i += 1;
                opts.temperature = args.get(i)
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
    eprintln!("  --model, -m NAME     Ollama model name (e.g., gemma3:1b)");
    eprintln!("  --output, -o PATH    Output JSONL file path");
    eprintln!();
    eprintln!("Optional:");
    eprintln!("  --repeats N          Repeat each config N times (default: 1)");
    eprintln!("  --augment            Add ±0.05 channel noise on repeat > 0");
    eprintln!("  --temperature F      Ollama sampling temperature (default: 0.7)");
    eprintln!("  --ollama-url URL     Ollama API URL (default: http://localhost:11434)");
    eprintln!("  --help, -h           Show this help message");
}
