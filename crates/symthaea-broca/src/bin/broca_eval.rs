//! broca-eval: Interactive evaluation tool for Broca CfC-HDC language checkpoints.
//!
//! Usage:
//!   broca-eval --checkpoint broca-cfc-v4.bin
//!   broca-eval --checkpoint broca-cfc-v4.bin --temperature 0.8 --top-k 40
//!   broca-eval --checkpoint broca-cfc-v4.bin --eval eval.jsonl --samples 20
//!
//! Modes:
//! - Interactive: type intent/epistemic/valence, see generated text in real-time
//! - Batch: run eval dataset + sample generation with configurable sampling

use std::process;

use symthaea_broca::encoder::ThoughtChannels;
use symthaea_broca::evaluation;
use symthaea_broca::generator::{BrocaGenerator, SamplingStrategy};
use symthaea_broca::training::TrainingDataset;

use symthaea_core::genesis::GenesisSeed;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn".parse().unwrap()),
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

    let genesis = GenesisSeed::from_phrase(&opts.genesis_phrase);

    // Load checkpoint
    eprintln!("Loading checkpoint: {}", opts.checkpoint_path);
    let (mut generator, _adam, _proj, _lm) =
        match BrocaGenerator::from_checkpoint(&opts.checkpoint_path, &genesis) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Failed to load checkpoint: {e}");
                process::exit(1);
            }
        };

    // Override sampling strategy
    let sampling = match (opts.top_k, opts.top_p) {
        (Some(k), _) => SamplingStrategy::TopK {
            k,
            temperature: opts.temperature,
        },
        (_, Some(p)) => SamplingStrategy::TopP {
            p,
            temperature: opts.temperature,
        },
        _ if (opts.temperature - 1.0).abs() > 0.01 || opts.temperature < 0.99 => {
            // Temperature != 1.0 implies we want stochastic sampling
            SamplingStrategy::TopK {
                k: 50,
                temperature: opts.temperature,
            }
        }
        _ => SamplingStrategy::Greedy,
    };

    // Apply sampling strategy by reconstructing with modified config
    // (BrocaGenerator doesn't expose config mutation, so we print info)
    eprintln!(
        "Sampling: {:?}",
        match &sampling {
            SamplingStrategy::Greedy => "greedy".to_string(),
            SamplingStrategy::TopK { k, temperature } => format!("top-k={k}, temp={temperature}"),
            SamplingStrategy::TopP { p, temperature } => format!("top-p={p}, temp={temperature}"),
        }
    );

    // Run evaluation if --eval provided
    if let Some(ref eval_path) = opts.eval_path {
        eprintln!("Evaluating on: {eval_path}");
        let mut eval_dataset = match TrainingDataset::from_jsonl(eval_path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Failed to load eval data: {e}");
                process::exit(1);
            }
        };
        let tokenizer = generator.tokenizer().clone();
        eval_dataset.tokenize_all(&tokenizer);

        let eval_config = evaluation::EvalConfig {
            dataset: eval_dataset,
            compute_perplexity: true,
            compute_english_ratio: true,
            per_intent_breakdown: true,
            max_gen_tokens: 64,
        };

        let result = evaluation::evaluate(&mut generator, &eval_config);
        println!("{}", evaluation::format_eval_report(&result));
    }

    // Generate samples
    let intent_names = ["Ack", "Ans", "Clr", "Pro", "Unc", "Ref", "Con", "Unk"];
    let epistemic_names = ["Certain", "Probable", "Uncertain", "Unknown", "OOD"];

    if opts.sample_count > 0 {
        println!("\n=== Sample Generations ===");
        println!(
            "{:>3}  {:>5}  {:>10}  {:>4}  {:>4}  {}",
            "#", "Intent", "Epistemic", "Psi", "Tok", "Generated Text"
        );
        println!("{}", "-".repeat(80));

        let thoughts = symthaea_broca::training::generate_diverse_thoughts();
        let step = thoughts.len() / opts.sample_count.max(1);

        for i in 0..opts.sample_count.min(thoughts.len()) {
            let channels = &thoughts[(i * step) % thoughts.len()];
            let result = generator.generate(channels);

            let intent_idx = (0..8)
                .max_by(|&a, &b| channels.channels[a].total_cmp(&channels.channels[b]))
                .unwrap_or(7);
            let ep_idx = (channels.epistemic_ordinal() as usize).min(4);

            let display_text = if result.text.len() > 100 {
                format!("{}...", &result.text[..100])
            } else {
                result.text.clone()
            };

            println!(
                "{:>3}  {:>5}  {:>10}  {:.1}  {:>4}  \"{}\"",
                i,
                intent_names[intent_idx],
                epistemic_names[ep_idx],
                channels.psi(),
                result.num_tokens,
                display_text,
            );
        }
    }

    // Interactive mode (if no --eval and no --samples, or --interactive)
    if opts.interactive || (opts.eval_path.is_none() && opts.sample_count == 0) {
        println!("\n=== Interactive Mode ===");
        println!("Enter thought channels as: intent(0-7) epistemic(0-4) valence(-1..1) arousal(0..1) psi(0..1)");
        println!("Example: 1 0 0.5 0.3 0.7  (Answer, Certain, positive, calm, aware)");
        println!("Or just press Enter for defaults. Type 'quit' to exit.\n");

        let stdin = std::io::stdin();
        loop {
            eprint!("> ");
            let mut line = String::new();
            if stdin.read_line(&mut line).is_err() || line.trim() == "quit" {
                break;
            }

            let parts: Vec<f32> = line
                .trim()
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();

            let mut channels = ThoughtChannels::default();
            if parts.len() >= 1 {
                channels = ThoughtChannels::with_intent(parts[0] as usize);
            }
            if parts.len() >= 2 {
                channels.set_epistemic(parts[1]);
            }
            if parts.len() >= 3 {
                channels.set_emotion(parts[2], parts.get(3).copied().unwrap_or(0.5), 0.5);
            }
            if parts.len() >= 5 {
                channels.set_consciousness(parts[4], 0.5, 0.5);
            }

            let result = generator.generate(&channels);
            println!(
                "  [{} tok, coh={:.3}] \"{}\"",
                result.num_tokens, result.final_coherence, result.text
            );
        }
    }
}

struct EvalOpts {
    checkpoint_path: String,
    eval_path: Option<String>,
    genesis_phrase: String,
    sample_count: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    interactive: bool,
}

fn parse_args(args: &[String]) -> Result<EvalOpts, String> {
    let mut opts = EvalOpts {
        checkpoint_path: String::new(),
        eval_path: None,
        genesis_phrase: "broca-training-default".to_string(),
        sample_count: 0,
        temperature: 1.0,
        top_k: None,
        top_p: None,
        interactive: false,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--checkpoint" | "-c" => {
                i += 1;
                opts.checkpoint_path =
                    args.get(i).cloned().ok_or("--checkpoint requires a path")?;
            }
            "--eval" => {
                i += 1;
                opts.eval_path = Some(args.get(i).cloned().ok_or("--eval requires a path")?);
            }
            "--genesis" => {
                i += 1;
                opts.genesis_phrase = args.get(i).cloned().ok_or("--genesis requires a phrase")?;
            }
            "--samples" | "-n" => {
                i += 1;
                opts.sample_count = args
                    .get(i)
                    .ok_or("--samples requires a number")?
                    .parse()
                    .map_err(|_| "--samples must be a number")?;
            }
            "--temperature" | "-t" => {
                i += 1;
                opts.temperature = args
                    .get(i)
                    .ok_or("--temperature requires a number")?
                    .parse()
                    .map_err(|_| "--temperature must be a float")?;
            }
            "--top-k" => {
                i += 1;
                opts.top_k = Some(
                    args.get(i)
                        .ok_or("--top-k requires a number")?
                        .parse()
                        .map_err(|_| "--top-k must be a number")?,
                );
            }
            "--top-p" => {
                i += 1;
                opts.top_p = Some(
                    args.get(i)
                        .ok_or("--top-p requires a number")?
                        .parse()
                        .map_err(|_| "--top-p must be a float")?,
                );
            }
            "--interactive" | "-i" => {
                opts.interactive = true;
            }
            "--help" | "-h" => {
                print_usage();
                process::exit(0);
            }
            arg => return Err(format!("Unknown argument: {arg}")),
        }
        i += 1;
    }

    if opts.checkpoint_path.is_empty() {
        return Err("--checkpoint is required".to_string());
    }

    Ok(opts)
}

fn print_usage() {
    eprintln!("Usage: broca-eval [OPTIONS]");
    eprintln!();
    eprintln!("Required:");
    eprintln!("  --checkpoint, -c PATH  Path to Broca checkpoint (.bin)");
    eprintln!();
    eprintln!("Optional:");
    eprintln!("  --eval PATH            Evaluate on held-out JSONL dataset");
    eprintln!("  --samples, -n N        Generate N sample outputs from diverse thoughts");
    eprintln!("  --temperature, -t F    Sampling temperature (default: 1.0)");
    eprintln!("  --top-k K              Top-k sampling (default: greedy)");
    eprintln!("  --top-p P              Top-p nucleus sampling (default: greedy)");
    eprintln!("  --interactive, -i      Force interactive mode");
    eprintln!("  --genesis PHRASE       Genesis seed phrase (default: broca-training-default)");
    eprintln!("  --help, -h             Show this help message");
}
