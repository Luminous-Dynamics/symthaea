// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Train a Broca checkpoint on Soma's embodied vocabulary.
//!
//! Produces a checkpoint that `BrocaSoma::load_checkpoint()` can load, so Soma
//! generates phone-aware language instead of generic tokens.
//!
//! Usage:
//!   cargo run -p symthaea-soma --release --bin train_soma_broca --features broca-full -- \
//!     --corpus data/soma_corpus.jsonl \
//!     --output data/soma_broca_checkpoint.bin \
//!     --epochs 50

use std::process;

use symthaea_broca::generator::{BrocaConfig, BrocaGenerator, SamplingStrategy};
use symthaea_broca::training::{
    CurriculumSchedule, TrainingConfig, TrainingDataset, train_with_adam,
};
use symthaea_broca::{LanguageControllerConfig, ThoughtChannels};
use symthaea_core::genesis::GenesisSeed;

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

    // Load corpus
    tracing::info!(path = %opts.corpus_path, "Loading Soma training corpus");
    let mut dataset = match TrainingDataset::from_jsonl(&opts.corpus_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to load corpus '{}': {e}", opts.corpus_path);
            process::exit(1);
        }
    };

    if dataset.is_empty() {
        eprintln!("Corpus is empty");
        process::exit(1);
    }

    tracing::info!(pairs = dataset.len(), "Corpus loaded");

    // Genesis seed matching BrocaSoma::new()
    let genesis = GenesisSeed::from_phrase("symthaea-soma");

    // Create generator with Soma-tuned config (matching BrocaSoma::new())
    let (mut generator, adam_state) = if let Some(ref resume_path) = opts.resume_path {
        tracing::info!(path = %resume_path, "Resuming from checkpoint");
        match BrocaGenerator::from_checkpoint(resume_path, &genesis) {
            Ok((r#gen, adam, _proj, _lm)) => (r#gen, adam),
            Err(e) => {
                eprintln!("Failed to load checkpoint '{}': {e}", resume_path);
                process::exit(1);
            }
        }
    } else {
        let config = BrocaConfig {
            controller: LanguageControllerConfig {
                network_layers: 3,
                neurons_per_layer: 8,
                dt_per_token: 0.02,
                logit_scale: 20.0,
                ..Default::default()
            },
            sampling: SamplingStrategy::TopK {
                k: 40,
                temperature: 0.7,
            },
            enable_epistemic_gate: true,
            enable_emotional_modulation: true,
            enable_coherence_feedback: true,
            enable_consciousness_gating: true,
            enable_semantic_veto: true,
            ..Default::default()
        };
        // Use 4K tokenizer for richer vocabulary coverage
        (BrocaGenerator::new_4k(&genesis, config), None)
    };

    // Re-tokenize all pairs with the generator's BPE tokenizer
    let tokenizer = generator.tokenizer().clone();
    dataset.retokenize_all(&tokenizer);

    let train_config = TrainingConfig {
        epochs: opts.epochs,
        learning_rate: opts.learning_rate,
        bptt_window: 16,
        grad_clip: 1.0,
        report_interval: 1,
        use_adam: true,
        warmup_fraction: 0.1,
        patience: opts.patience,
        enable_diagnostics: opts.diagnostics,
        train_network: true,
        network_lr_scale: 0.3,
        embedding_target_norm: 128.0,
        negative_samples: 0,
        carry_state: 0.3, // Carry CfC state 30% of the time for coherent multi-sentence flow
        network_warmup_epochs: opts.network_warmup,
        validation_dataset: None,
        curriculum: CurriculumSchedule::LengthAscending,
        enable_smoke_test: true,
        ..Default::default()
    };

    tracing::info!(
        epochs = opts.epochs,
        lr = opts.learning_rate,
        pairs = dataset.len(),
        patience = opts.patience,
        network_warmup = opts.network_warmup,
        "Starting Soma Broca training"
    );

    let (metrics, final_adam, diagnostics, anomaly_report, validation) =
        train_with_adam(&mut generator, &dataset, &train_config, adam_state);

    // Print epoch summary
    println!("\n--- Soma Broca Training Summary ---");
    println!("{:>6}  {:>10}  {:>8}", "Epoch", "Avg Loss", "Tokens");
    for m in &metrics {
        println!("{:>6}  {:>10.6}  {:>8}", m.epoch, m.avg_loss, m.num_tokens);
    }

    // Print gradient diagnostics if enabled
    if let Some(diag) = diagnostics {
        println!();
        println!("{}", diag.format_summary());
    }

    // Print anomaly report
    if let Some(ref report) = anomaly_report {
        println!("\n--- Anomaly Report ---");
        println!("  Anomalous epochs: {}", report.anomalous_epoch_count);
        println!("  Early stopped:    {}", report.anomaly_early_stopped);
    }

    // Print smoke test results
    if let Some(ref val) = validation {
        println!("\n--- Smoke Test ---");
        println!("  Result: {}", if val.passed { "PASSED" } else { "FAILED" });
        println!("  Mean coherence: {:.4}", val.mean_coherence);
    }

    // Save checkpoint
    let final_loss = metrics.last().map(|m| m.avg_loss).unwrap_or(0.0);
    let final_epoch = metrics.last().map(|m| m.epoch).unwrap_or(0);

    tracing::info!(path = %opts.output_path, epoch = final_epoch, loss = final_loss, "Saving Soma Broca checkpoint");
    if let Err(e) = generator.save_checkpoint(
        &opts.output_path,
        final_epoch,
        final_loss,
        final_adam,
        None,
        None,
    ) {
        eprintln!("Failed to save checkpoint: {e}");
        process::exit(1);
    }

    println!("\nCheckpoint saved to: {}", opts.output_path);

    // Generate sample outputs
    if opts.sample_count > 0 {
        println!("\n--- Sample Generations (Soma Embodied) ---");
        let sample_configs: Vec<(&str, ThoughtChannels)> = vec![
            ("high-consciousness", {
                let mut tc = ThoughtChannels::with_intent(5); // reflect
                tc.set_consciousness(0.9, 0.8, 0.85);
                tc.set_emotion(0.3, 0.3, 0.6);
                tc.set_epistemic(1.0);
                tc.set_context(0.2, 0.7, 0.3, 0.8);
                tc
            }),
            ("motion-walking", {
                let mut tc = ThoughtChannels::with_intent(0); // acknowledge
                tc.set_consciousness(0.6, 0.5, 0.6);
                tc.set_emotion(0.3, 0.5, 0.5);
                tc.set_epistemic(1.0);
                tc.set_context(0.3, 0.5, 0.3, 0.5);
                tc
            }),
            ("screen-surprise", {
                let mut tc = ThoughtChannels::with_intent(1); // answer
                tc.set_consciousness(0.6, 0.5, 0.6);
                tc.set_emotion(0.4, 0.7, 0.4);
                tc.set_epistemic(1.0);
                tc.set_context(0.5, 0.5, 0.4, 0.6);
                tc
            }),
            ("dreaming", {
                let mut tc = ThoughtChannels::with_intent(5); // reflect
                tc.set_consciousness(0.2, 0.15, 0.2);
                tc.set_emotion(0.0, 0.1, 0.4);
                tc.set_epistemic(1.0);
                tc.set_context(0.0, 0.3, 0.3, 0.2);
                tc
            }),
            ("epistemic-humble", {
                let mut tc = ThoughtChannels::with_intent(4); // uncertainty
                tc.set_consciousness(0.5, 0.4, 0.5);
                tc.set_emotion(0.1, 0.3, 0.4);
                tc.set_epistemic(1.0);
                tc.set_context(0.1, 0.4, 0.3, 0.4);
                tc
            }),
            ("high-oxytocin", {
                let mut tc = ThoughtChannels::with_intent(5); // reflect
                tc.set_consciousness(0.7, 0.6, 0.7);
                tc.set_emotion(0.4, 0.3, 0.9);
                tc.set_epistemic(1.0);
                tc.set_context(0.1, 0.6, 0.2, 0.7);
                tc
            }),
        ];

        for (label, channels) in &sample_configs {
            let result = generator.generate(channels);
            println!(
                "[{:>18}] {:>3} tok | \"{}\"",
                label,
                result.num_tokens,
                &result.text[..result.text.len().min(80)]
            );
        }
    }
}

struct TrainOpts {
    corpus_path: String,
    output_path: String,
    resume_path: Option<String>,
    epochs: usize,
    learning_rate: f32,
    patience: usize,
    diagnostics: bool,
    sample_count: usize,
    network_warmup: usize,
}

fn parse_args(args: &[String]) -> Result<TrainOpts, String> {
    let mut opts = TrainOpts {
        corpus_path: String::new(),
        output_path: "data/soma_broca_checkpoint.bin".to_string(),
        resume_path: None,
        epochs: 50,
        learning_rate: 0.001,
        patience: 10,
        diagnostics: false,
        sample_count: 6,
        network_warmup: 5,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--corpus" | "-c" => {
                i += 1;
                opts.corpus_path = args.get(i).cloned().ok_or("--corpus requires a path")?;
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
            "--patience" => {
                i += 1;
                opts.patience = args
                    .get(i)
                    .ok_or("--patience requires a number")?
                    .parse()
                    .map_err(|_| "--patience must be a non-negative integer")?;
            }
            "--network-warmup" => {
                i += 1;
                opts.network_warmup = args
                    .get(i)
                    .ok_or("--network-warmup requires a number")?
                    .parse()
                    .map_err(|_| "--network-warmup must be a non-negative integer")?;
            }
            "--diagnostics" => {
                opts.diagnostics = true;
            }
            "--samples" => {
                i += 1;
                opts.sample_count = args
                    .get(i)
                    .ok_or("--samples requires a number")?
                    .parse()
                    .map_err(|_| "--samples must be a non-negative integer")?;
            }
            "--no-samples" => {
                opts.sample_count = 0;
            }
            "--help" | "-h" => {
                print_usage();
                process::exit(0);
            }
            arg => return Err(format!("Unknown argument: {arg}")),
        }
        i += 1;
    }

    if opts.corpus_path.is_empty() {
        return Err("--corpus is required".to_string());
    }

    Ok(opts)
}

fn print_usage() {
    eprintln!("Usage: train_soma_broca [OPTIONS]");
    eprintln!();
    eprintln!("Train a Broca checkpoint on Soma's embodied vocabulary.");
    eprintln!();
    eprintln!("Required:");
    eprintln!("  --corpus, -c PATH    JSONL corpus file (soma_corpus.jsonl)");
    eprintln!();
    eprintln!("Optional:");
    eprintln!("  --output, -o PATH    Output checkpoint (default: data/soma_broca_checkpoint.bin)");
    eprintln!("  --resume, -r PATH    Resume training from existing checkpoint");
    eprintln!("  --epochs, -e N       Number of training epochs (default: 50)");
    eprintln!("  --lr RATE            Learning rate (default: 0.001)");
    eprintln!("  --patience N         Early stopping patience (default: 10)");
    eprintln!("  --network-warmup N   Embedding-only epochs before CfC BPTT (default: 5)");
    eprintln!("  --diagnostics        Enable gradient flow diagnostics");
    eprintln!("  --samples N          Generate N sample outputs after training (default: 6)");
    eprintln!("  --no-samples         Skip sample generation");
    eprintln!("  --help, -h           Show this help message");
}
