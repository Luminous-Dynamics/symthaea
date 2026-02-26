//! broca-train: Command-line training tool for the Broca SSM language center.
//!
//! Usage:
//!   broca-train --data training.jsonl --epochs 100 --lr 0.001 --output broca.bin
//!   broca-train --data training.jsonl --resume prev.bin --epochs 50
//!
//! The data file should be JSONL with one TrainingPair per line:
//!   {"channels": [0.0, ...20 floats...], "target_text": "hello world"}

use std::process;

use symthaea_broca::generator::{BrocaConfig, BrocaGenerator};
use symthaea_broca::training::{TrainingConfig, TrainingDataset, train_with_adam};

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
            Ok((gen, adam, _proj)) => (gen, adam),
            Err(e) => {
                eprintln!("Failed to load checkpoint '{}': {e}", resume_path);
                process::exit(1);
            }
        }
    } else {
        let config = BrocaConfig::default();
        (BrocaGenerator::new(&genesis, config), None)
    };

    // Tokenize all pairs with the generator's tokenizer
    let tokenizer = generator.tokenizer().clone();
    dataset.tokenize_all(&tokenizer);

    let train_config = TrainingConfig {
        epochs: opts.epochs,
        learning_rate: opts.learning_rate,
        bptt_window: opts.bptt_window,
        grad_clip: opts.grad_clip,
        report_interval: 1,
        use_adam: true,
        warmup_fraction: 0.1,
        patience: opts.patience,
    };

    tracing::info!(
        epochs = opts.epochs,
        lr = opts.learning_rate,
        bptt_window = opts.bptt_window,
        patience = opts.patience,
        use_adam = true,
        "Starting training"
    );

    let (metrics, final_adam) = train_with_adam(
        &mut generator,
        &dataset,
        &train_config,
        adam_state,
    );

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
    ) {
        eprintln!("Failed to save checkpoint: {e}");
        process::exit(1);
    }

    println!("\nCheckpoint saved to: {}", opts.output_path);
}

struct TrainOpts {
    data_path: String,
    output_path: String,
    resume_path: Option<String>,
    epochs: usize,
    learning_rate: f32,
    bptt_window: usize,
    grad_clip: f32,
    patience: usize,
    genesis_phrase: String,
}

fn parse_args(args: &[String]) -> Result<TrainOpts, String> {
    let mut opts = TrainOpts {
        data_path: String::new(),
        output_path: "broca.bin".to_string(),
        resume_path: None,
        epochs: 100,
        learning_rate: 0.001,
        bptt_window: 16,
        grad_clip: 1.0,
        patience: 0,
        genesis_phrase: "broca-training-default".to_string(),
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
            "--epochs" | "-e" => {
                i += 1;
                opts.epochs = args.get(i)
                    .ok_or("--epochs requires a number")?
                    .parse()
                    .map_err(|_| "--epochs must be a positive integer")?;
            }
            "--lr" => {
                i += 1;
                opts.learning_rate = args.get(i)
                    .ok_or("--lr requires a number")?
                    .parse()
                    .map_err(|_| "--lr must be a float")?;
            }
            "--bptt-window" => {
                i += 1;
                opts.bptt_window = args.get(i)
                    .ok_or("--bptt-window requires a number")?
                    .parse()
                    .map_err(|_| "--bptt-window must be a positive integer")?;
            }
            "--grad-clip" => {
                i += 1;
                opts.grad_clip = args.get(i)
                    .ok_or("--grad-clip requires a number")?
                    .parse()
                    .map_err(|_| "--grad-clip must be a float")?;
            }
            "--patience" => {
                i += 1;
                opts.patience = args.get(i)
                    .ok_or("--patience requires a number")?
                    .parse()
                    .map_err(|_| "--patience must be a non-negative integer")?;
            }
            "--genesis" => {
                i += 1;
                opts.genesis_phrase = args.get(i).cloned().ok_or("--genesis requires a phrase")?;
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
    eprintln!("Usage: broca-train [OPTIONS]");
    eprintln!();
    eprintln!("Required:");
    eprintln!("  --data, -d PATH      JSONL training data file");
    eprintln!();
    eprintln!("Optional:");
    eprintln!("  --output, -o PATH    Output checkpoint file (default: broca.bin)");
    eprintln!("  --resume, -r PATH    Resume from existing checkpoint");
    eprintln!("  --epochs, -e N       Number of training epochs (default: 100)");
    eprintln!("  --lr RATE            Learning rate (default: 0.001)");
    eprintln!("  --bptt-window N      BPTT truncation window (default: 16)");
    eprintln!("  --grad-clip THRESH   Gradient clipping threshold (default: 1.0)");
    eprintln!("  --patience N         Early stopping patience, 0=disabled (default: 0)");
    eprintln!("  --genesis PHRASE     Genesis seed phrase (default: broca-training-default)");
    eprintln!("  --help, -h           Show this help message");
}
