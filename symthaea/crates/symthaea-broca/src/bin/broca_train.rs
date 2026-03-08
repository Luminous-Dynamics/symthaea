//! broca-train: Command-line training tool for the Broca SSM language center.
//!
//! Usage:
//!   broca-train --data training.jsonl --epochs 100 --lr 0.001 --output broca.bin
//!   broca-train --data training.jsonl --resume prev.bin --epochs 50
//!   broca-train --data training.jsonl --epochs 200 --eval eval.jsonl --diagnostics
//!
//! The data file should be JSONL with one TrainingPair per line:
//!   {"channels": [0.0, ...20 floats...], "target_text": "hello world"}

use std::process;

use symthaea_broca::evaluation;
use symthaea_broca::generator::{BrocaConfig, BrocaGenerator};
use symthaea_broca::training::{train_with_adam, TrainingConfig, TrainingDataset};

use symthaea_core::genesis::GenesisSeed;

fn main() {
    // Initialize tracing
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

    // Handle --curriculum: generate curriculum JSONL and exit
    if let Some(ref curriculum_path) = opts.generate_curriculum {
        let tokenizer = symthaea_broca::BpeTokenizer::default_minimal();
        match symthaea_broca::training::write_curriculum_jsonl(curriculum_path, &tokenizer) {
            Ok(count) => {
                println!("Generated curriculum with {count} pairs → {curriculum_path}");
                process::exit(0);
            }
            Err(e) => {
                eprintln!("Failed to generate curriculum: {e}");
                process::exit(1);
            }
        }
    }

    // Load dataset
    tracing::info!(path = %opts.data_path, "Loading training data");
    let mut dataset = match TrainingDataset::from_jsonl(&opts.data_path) {
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

    tracing::info!(pairs = dataset.len(), "Dataset loaded");

    // Genesis seed for deterministic initialization
    let genesis = GenesisSeed::from_phrase(&opts.genesis_phrase);

    // Create or resume generator
    let (mut generator, adam_state) = if let Some(ref resume_path) = opts.resume_path {
        tracing::info!(path = %resume_path, "Resuming from checkpoint");
        match BrocaGenerator::from_checkpoint(resume_path, &genesis) {
            Ok((gen, adam, _proj, _lm_config)) => (gen, adam),
            Err(e) => {
                eprintln!("Failed to load checkpoint '{}': {e}", resume_path);
                process::exit(1);
            }
        }
    } else {
        let config = BrocaConfig::default();
        (BrocaGenerator::new(&genesis, config), None)
    };

    // Re-tokenize all pairs with the generator's BPE tokenizer.
    // The JSONL may have target_ids from a different tokenizer (e.g., GPT-NeoX
    // from broca-collect), so we always re-tokenize for CfC-HDC compatibility.
    let tokenizer = generator.tokenizer().clone();
    dataset.retokenize_all(&tokenizer);

    let train_config = TrainingConfig {
        epochs: opts.epochs,
        learning_rate: opts.learning_rate,
        bptt_window: opts.bptt_window,
        grad_clip: opts.grad_clip,
        report_interval: 1,
        use_adam: true,
        warmup_fraction: 0.1,
        patience: opts.patience,
        enable_diagnostics: opts.diagnostics,
    };

    tracing::info!(
        epochs = opts.epochs,
        lr = opts.learning_rate,
        bptt_window = opts.bptt_window,
        patience = opts.patience,
        use_adam = true,
        diagnostics = opts.diagnostics,
        "Starting training"
    );

    let (metrics, final_adam, diagnostics) =
        train_with_adam(&mut generator, &dataset, &train_config, adam_state);

    // Report results
    if let Some(last) = metrics.last() {
        tracing::info!(
            epochs_completed = metrics.len(),
            final_loss = last.avg_loss,
            total_tokens = last.num_tokens,
            "Training complete"
        );
    }

    // Print epoch summary
    println!("\n--- Training Summary ---");
    println!("{:>6}  {:>10}  {:>8}", "Epoch", "Avg Loss", "Tokens");
    for m in &metrics {
        println!("{:>6}  {:>10.6}  {:>8}", m.epoch, m.avg_loss, m.num_tokens);
    }

    // Print gradient diagnostics if enabled
    if let Some(diag) = diagnostics {
        println!();
        println!("{}", diag.format_summary());
    }

    // Save checkpoint
    let final_loss = metrics.last().map(|m| m.avg_loss).unwrap_or(0.0);
    let final_epoch = metrics.last().map(|m| m.epoch).unwrap_or(0);

    tracing::info!(path = %opts.output_path, "Saving checkpoint");
    if let Err(e) = generator.save_checkpoint(
        &opts.output_path,
        final_epoch,
        final_loss,
        final_adam,
        None, // No projection weights in standalone training
        None, // No L-SSM config in CfC-HDC training
    ) {
        eprintln!("Failed to save checkpoint: {e}");
        process::exit(1);
    }

    println!("\nCheckpoint saved to: {}", opts.output_path);

    // Run evaluation if --eval provided
    if let Some(ref eval_path) = opts.eval_path {
        tracing::info!(path = %eval_path, "Loading evaluation data");
        let mut eval_dataset = match TrainingDataset::from_jsonl(eval_path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Failed to load eval data from '{}': {e}", eval_path);
                process::exit(1);
            }
        };
        let eval_tokenizer = generator.tokenizer().clone();
        eval_dataset.tokenize_all(&eval_tokenizer);

        let eval_config = evaluation::EvalConfig {
            dataset: eval_dataset,
            compute_perplexity: true,
            compute_english_ratio: true,
            per_intent_breakdown: true,
            max_gen_tokens: 64,
        };

        let result = evaluation::evaluate(&mut generator, &eval_config);
        println!();
        println!("{}", evaluation::format_eval_report(&result));
    }

    // Generate sample outputs if --samples N
    if opts.sample_count > 0 {
        println!("\n--- Sample Generations ---");
        let thoughts = symthaea_broca::training::generate_diverse_thoughts();
        let step = thoughts.len() / opts.sample_count.max(1);
        for i in 0..opts.sample_count.min(thoughts.len()) {
            let channels = &thoughts[(i * step) % thoughts.len()];
            let result = generator.generate(&symthaea_broca::ThoughtChannels {
                channels: channels.channels,
            });
            let intent = (0..8)
                .max_by(|&a, &b| channels.channels[a].total_cmp(&channels.channels[b]))
                .unwrap_or(7);
            let intent_names = [
                "Ack", "Ans", "Clr", "Pro", "Unc", "Ref", "Con", "Unk",
            ];
            println!(
                "[{:>3}] intent={} psi={:.1} | {:>3} tok | \"{}\"",
                i,
                intent_names[intent],
                channels.psi(),
                result.num_tokens,
                &result.text[..result.text.len().min(80)]
            );
        }
    }
}

struct TrainOpts {
    data_path: String,
    output_path: String,
    resume_path: Option<String>,
    eval_path: Option<String>,
    epochs: usize,
    learning_rate: f32,
    bptt_window: usize,
    grad_clip: f32,
    patience: usize,
    diagnostics: bool,
    genesis_phrase: String,
    /// Generate a curriculum JSONL and exit (no training).
    generate_curriculum: Option<String>,
    /// After training, generate sample text from N random thoughts.
    sample_count: usize,
}

fn parse_args(args: &[String]) -> Result<TrainOpts, String> {
    let mut opts = TrainOpts {
        data_path: String::new(),
        output_path: "broca.bin".to_string(),
        resume_path: None,
        eval_path: None,
        epochs: 100,
        learning_rate: 0.001,
        bptt_window: 16,
        grad_clip: 1.0,
        patience: 0,
        diagnostics: false,
        genesis_phrase: "broca-training-default".to_string(),
        generate_curriculum: None,
        sample_count: 0,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data" | "-d" => {
                i += 1;
                opts.data_path = args.get(i).cloned().ok_or("--data requires a path")?;
            }
            "--output" | "-o" => {
                i += 1;
                opts.output_path = args.get(i).cloned().ok_or("--output requires a path")?;
            }
            "--resume" | "-r" => {
                i += 1;
                opts.resume_path = Some(args.get(i).cloned().ok_or("--resume requires a path")?);
            }
            "--eval" => {
                i += 1;
                opts.eval_path = Some(args.get(i).cloned().ok_or("--eval requires a path")?);
            }
            "--epochs" | "-e" => {
                i += 1;
                opts.epochs = args
                    .get(i)
                    .ok_or("--epochs requires a number")?
                    .parse()
                    .map_err(|_| "--epochs must be a positive integer")?;
            }
            "--lr" => {
                i += 1;
                opts.learning_rate = args
                    .get(i)
                    .ok_or("--lr requires a number")?
                    .parse()
                    .map_err(|_| "--lr must be a float")?;
            }
            "--bptt-window" => {
                i += 1;
                opts.bptt_window = args
                    .get(i)
                    .ok_or("--bptt-window requires a number")?
                    .parse()
                    .map_err(|_| "--bptt-window must be a positive integer")?;
            }
            "--grad-clip" => {
                i += 1;
                opts.grad_clip = args
                    .get(i)
                    .ok_or("--grad-clip requires a number")?
                    .parse()
                    .map_err(|_| "--grad-clip must be a float")?;
            }
            "--patience" => {
                i += 1;
                opts.patience = args
                    .get(i)
                    .ok_or("--patience requires a number")?
                    .parse()
                    .map_err(|_| "--patience must be a non-negative integer")?;
            }
            "--diagnostics" => {
                opts.diagnostics = true;
            }
            "--genesis" => {
                i += 1;
                opts.genesis_phrase = args.get(i).cloned().ok_or("--genesis requires a phrase")?;
            }
            "--curriculum" => {
                i += 1;
                opts.generate_curriculum =
                    Some(args.get(i).cloned().ok_or("--curriculum requires a path")?);
            }
            "--samples" => {
                i += 1;
                opts.sample_count = args
                    .get(i)
                    .ok_or("--samples requires a number")?
                    .parse()
                    .map_err(|_| "--samples must be a non-negative integer")?;
            }
            "--help" | "-h" => {
                print_usage();
                process::exit(0);
            }
            arg => return Err(format!("Unknown argument: {arg}")),
        }
        i += 1;
    }

    if opts.data_path.is_empty() && opts.generate_curriculum.is_none() {
        return Err("--data is required (or use --curriculum to generate data)".to_string());
    }

    Ok(opts)
}

fn print_usage() {
    eprintln!("Usage: broca-train [OPTIONS]");
    eprintln!();
    eprintln!("Required:");
    eprintln!("  --data, -d PATH      JSONL training data file");
    eprintln!();
    eprintln!("Optional:");
    eprintln!("  --output, -o PATH    Output checkpoint file (default: broca.bin)");
    eprintln!("  --resume, -r PATH    Resume from existing checkpoint");
    eprintln!("  --eval PATH          Held-out JSONL for post-training evaluation");
    eprintln!("  --epochs, -e N       Number of training epochs (default: 100)");
    eprintln!("  --lr RATE            Learning rate (default: 0.001)");
    eprintln!("  --bptt-window N      BPTT truncation window (default: 16)");
    eprintln!("  --grad-clip THRESH   Gradient clipping threshold (default: 1.0)");
    eprintln!("  --patience N         Early stopping patience, 0=disabled (default: 0)");
    eprintln!("  --diagnostics        Enable gradient flow diagnostics");
    eprintln!("  --genesis PHRASE     Genesis seed phrase (default: broca-training-default)");
    eprintln!("  --curriculum PATH    Generate curriculum JSONL (1800 diverse pairs) and exit");
    eprintln!("  --samples N          Generate N sample outputs after training");
    eprintln!("  --help, -h           Show this help message");
}
