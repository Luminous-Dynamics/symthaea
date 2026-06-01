// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
use symthaea_broca::generator::{BrocaConfig, BrocaGenerator, SamplingStrategy};
use symthaea_broca::training::{
    CurriculumSchedule, TrainingConfig, TrainingDataset, train_with_adam,
};

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
        let style = match symthaea_broca::training::CurriculumStyle::parse(&opts.curriculum_style) {
            Ok(style) => style,
            Err(e) => {
                eprintln!("Invalid --curriculum-style: {e}");
                process::exit(2);
            }
        };
        match symthaea_broca::training::write_curriculum_jsonl_with_style(
            curriculum_path,
            &tokenizer,
            style,
        ) {
            Ok(count) => {
                println!(
                    "Generated {} curriculum with {count} pairs → {curriculum_path}",
                    opts.curriculum_style
                );
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
    let (mut generator, mut adam_state) = if let Some(ref resume_path) = opts.resume_path {
        tracing::info!(path = %resume_path, "Resuming from checkpoint");
        let load_result = if opts.allow_checksum_mismatch {
            BrocaGenerator::from_checkpoint_allow_checksum_mismatch(resume_path, &genesis)
        } else {
            BrocaGenerator::from_checkpoint(resume_path, &genesis)
        };
        match load_result {
            Ok((loaded_generator, adam, _proj, _lm_config)) => (loaded_generator, adam),
            Err(e) => {
                eprintln!("Failed to load checkpoint '{}': {e}", resume_path);
                process::exit(1);
            }
        }
    } else {
        let mut config = BrocaConfig::default();
        config.controller.network_layers = opts.network_layers;
        config.controller.neurons_per_layer = opts.neurons_per_layer;
        (BrocaGenerator::new_4k(&genesis, config), None)
    };

    // Re-tokenize all pairs with the generator's BPE tokenizer.
    // The JSONL may have target_ids from a different tokenizer (e.g., GPT-NeoX
    // from broca-collect), so we always re-tokenize for CfC-HDC compatibility.
    let tokenizer = generator.tokenizer().clone();
    dataset.retokenize_all(&tokenizer);
    let unk_target_tokens = dataset
        .pairs
        .iter()
        .flat_map(|pair| pair.target_ids.iter())
        .filter(|&&id| id == tokenizer.unk_id)
        .count();
    let total_target_tokens: usize = dataset.pairs.iter().map(|pair| pair.target_ids.len()).sum();
    if total_target_tokens > 0 {
        tracing::info!(
            unk_target_tokens,
            total_target_tokens,
            unk_target_rate = unk_target_tokens as f32 / total_target_tokens as f32,
            "Retokenized training target unknown-token rate"
        );
    }

    // Load validation dataset for early stopping if --eval was provided
    let validation_dataset = opts.eval_path.as_ref().and_then(|eval_path| {
        match TrainingDataset::from_jsonl(eval_path) {
            Ok(mut val) => {
                val.retokenize_all(&tokenizer);
                tracing::info!(pairs = val.len(), "Validation dataset loaded for early stopping");
                Some(val)
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to load validation dataset, using training loss for early stopping");
                None
            }
        }
    });

    // Discard checkpoint Adam state if --fresh-adam
    if opts.fresh_adam {
        tracing::info!("Discarding checkpoint Adam state (--fresh-adam)");
        adam_state = None;
    }

    // Auto-generate best checkpoint path: output.bin → output.best.bin
    let best_path = if opts.best_checkpoint_path.is_empty() && opts.patience > 0 {
        if let Some(stem) = opts.output_path.strip_suffix(".bin") {
            format!("{stem}.best.bin")
        } else {
            format!("{}.best", opts.output_path)
        }
    } else {
        opts.best_checkpoint_path.clone()
    };

    let train_config = TrainingConfig {
        epochs: opts.epochs,
        learning_rate: opts.learning_rate,
        bptt_window: opts.bptt_window,
        grad_clip: opts.grad_clip,
        report_interval: 1,
        progress: true,
        use_adam: true,
        warmup_fraction: 0.1,
        patience: opts.patience,
        enable_diagnostics: opts.diagnostics,
        train_network: opts.train_network,
        network_lr_scale: opts.network_lr_scale,
        embedding_target_norm: opts.embedding_target_norm,
        negative_samples: opts.negative_samples,
        carry_state: opts.carry_state,
        network_warmup_epochs: opts.network_warmup_epochs,
        validation_dataset,
        curriculum: CurriculumSchedule::LengthAscending,
        freeze_embeddings: opts.freeze_embeddings,
        coherence_alignment_weight: opts.coherence_alignment_weight,
        coherence_alignment_start_weight: opts.coherence_alignment_start,
        merge_token_loss_weight: opts.merge_token_loss_weight,
        enable_fusion_during_training: opts.enable_fusion,
        fusion_warmup_epochs: opts.fusion_warmup_epochs,
        contrastive_weight: opts.contrastive_weight,
        contrastive_margin: opts.contrastive_margin,
        scheduled_sampling_max: opts.scheduled_sampling,
        label_smoothing: opts.label_smoothing,
        thought_logit_aux_weight: opts.thought_logit_aux_weight,
        logit_anchor_weight: opts.logit_anchor_weight,
        top_token_anticollapse_weight: opts.top_token_anticollapse_weight,
        top_token_anticollapse_margin: opts.top_token_anticollapse_margin,
        unknown_token_penalty_weight: opts.unknown_token_penalty_weight,
        unknown_token_penalty_margin: opts.unknown_token_penalty_margin,
        best_checkpoint_path: best_path,
        hidden_dropout: opts.hidden_dropout,
        adaptive_veto_target: opts.adaptive_veto_target,
        veto_warmup_epochs: opts.veto_warmup_epochs,
        enable_soft_veto_during_training: opts.enable_soft_veto_training,
        ..Default::default()
    };

    // Optionally disable thought-seeded CfC state
    if opts.no_thought_seed {
        generator
            .controller_mut()
            .config_mut()
            .enable_thought_seeding = false;
    }
    if opts.thought_logit_residual_weight > 0.0 {
        generator
            .controller_mut()
            .config_mut()
            .thought_logit_residual_weight = opts.thought_logit_residual_weight.clamp(0.0, 1.0);
    }

    tracing::info!(
        epochs = opts.epochs,
        lr = opts.learning_rate,
        bptt_window = opts.bptt_window,
        patience = opts.patience,
        use_adam = true,
        train_network = opts.train_network,
        network_lr_scale = opts.network_lr_scale,
        embedding_norm = opts.embedding_target_norm,
        negative_samples = opts.negative_samples,
        carry_state = opts.carry_state,
        network_warmup_epochs = opts.network_warmup_epochs,
        freeze_embeddings = opts.freeze_embeddings,
        diagnostics = opts.diagnostics,
        "Starting training"
    );

    let (metrics, final_adam, diagnostics, anomaly_report, validation) =
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

    // Print anomaly report if anomaly response was active
    if let Some(ref report) = anomaly_report {
        println!("\n--- Anomaly Report ---");
        println!("  Final LR multiplier: {:.4}", report.final_lr_multiplier);
        println!("  Final grad clip:     {:.4}", report.final_grad_clip);
        println!("  Anomalous epochs:    {}", report.anomalous_epoch_count);
        println!("  Early stopped:       {}", report.anomaly_early_stopped);
        println!(
            "  Forced CfC training: {}",
            report.plateau_forced_network_training
        );
        for (epoch, anomalies) in &report.epoch_anomalies {
            for a in anomalies {
                println!("  epoch {epoch}: {a}");
            }
        }
    }

    // Print smoke test results if enabled
    if let Some(ref val) = validation {
        println!("\n--- Smoke Test ---");
        println!("  Result: {}", if val.passed { "PASSED" } else { "FAILED" });
        println!("  Mean coherence: {:.4}", val.mean_coherence);
        for (intent, coh) in &val.intent_coherences {
            println!("  intent {intent}: coh={coh:.4}");
        }
        if !val.failed_intents.is_empty() {
            println!("  Failed intents: {:?}", val.failed_intents);
        }
    }

    // Save checkpoint
    let final_loss = metrics.last().map(|m| m.avg_loss).unwrap_or(0.0);
    let final_epoch = metrics.last().map(|m| m.epoch).unwrap_or(0);

    tracing::info!(path = %opts.output_path, "Saving checkpoint");
    let checkpoint_adam = if opts.no_save_adam { None } else { final_adam };
    if let Err(e) = generator.save_checkpoint(
        &opts.output_path,
        final_epoch,
        final_loss,
        checkpoint_adam,
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
            eval_limit: 0,
            progress: true,
            compute_contrastive_intent: true,
        };

        let result = evaluation::evaluate(&mut generator, &eval_config);
        println!();
        println!("{}", evaluation::format_eval_report(&result));
    }

    // Override sampling strategy for sample generation if --temperature/--top-k
    if opts.sample_top_k > 0 && opts.sample_temperature > 0.0 {
        generator.set_sampling(SamplingStrategy::TopK {
            k: opts.sample_top_k,
            temperature: opts.sample_temperature,
        });
    }

    // Generate sample outputs if --samples N
    if opts.sample_count > 0 {
        println!("\n--- Sample Generations ---");
        let thoughts = symthaea_broca::training::generate_diverse_thoughts();
        let step = thoughts.len() / opts.sample_count.max(1);
        for i in 0..opts.sample_count.min(thoughts.len()) {
            let channels = &thoughts[(i * step) % thoughts.len()];
            let result = generator.generate(channels);
            let intent = (0..8)
                .max_by(|&a, &b| channels.channels[a].total_cmp(&channels.channels[b]))
                .unwrap_or(7);
            let intent_names = ["Ack", "Ans", "Clr", "Pro", "Unc", "Ref", "Con", "Unk"];
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
    /// Curriculum target style for --curriculum (structured or prose).
    curriculum_style: String,
    /// After training, generate sample text from N random thoughts.
    sample_count: usize,
    /// Train CfC network weights via BPTT (default: true).
    train_network: bool,
    /// CfC network LR scale relative to embedding LR (default: 0.3).
    network_lr_scale: f32,
    /// Target embedding L2 norm, 0 = disabled (default: 128.0).
    embedding_target_norm: f32,
    /// Negative samples for sampled softmax, 0 = full (default: 0).
    negative_samples: usize,
    /// Probability of carrying CfC state between pairs (default: 0.0).
    carry_state: f32,
    /// Embedding-only epochs before enabling CfC BPTT (default: 0).
    network_warmup_epochs: usize,
    /// Discard checkpoint Adam optimizer state on resume (default: false).
    fresh_adam: bool,
    /// Freeze embedding weights, only train CfC network via BPTT (default: false).
    freeze_embeddings: bool,
    /// Temperature for sample generation (default: 1.0).
    sample_temperature: f32,
    /// Top-k for sample generation, 0 = use config default (default: 0).
    sample_top_k: usize,
    /// Coherence alignment loss weight (default: 0.0 = off).
    coherence_alignment_weight: f32,
    /// Starting weight for curriculum alignment annealing (default: 0.0 = constant).
    coherence_alignment_start: f32,
    /// Merge-token loss bias weight (default: 1.5).
    merge_token_loss_weight: f32,
    /// Enable fusion flags during training (compositional logits, adaptive dt, adaptive alpha).
    enable_fusion: bool,
    /// Epochs of raw CfC training before fusion activates (default: 0 = immediate).
    fusion_warmup_epochs: usize,
    /// Disable thought-seeded CfC initial state.
    no_thought_seed: bool,
    /// Contrastive intent loss weight (default: 0.0 = off).
    contrastive_weight: f32,
    /// Contrastive margin (default: 0.0).
    contrastive_margin: f32,
    /// Scheduled sampling max probability (default: 0.0 = pure teacher forcing).
    /// Linearly anneals from 0 to this value over training epochs.
    scheduled_sampling: f32,
    /// Label smoothing epsilon (default: 0.0 = disabled).
    label_smoothing: f32,
    /// Thought-to-logit auxiliary loss weight (default: 0.0 = disabled).
    thought_logit_aux_weight: f32,
    /// KL-style logit distribution anchor weight (default: 0.0 = disabled).
    logit_anchor_weight: f32,
    /// Wrong-argmax anti-collapse margin loss weight (default: 0.0 = disabled).
    top_token_anticollapse_weight: f32,
    /// Required target-vs-wrong-top logit margin for anti-collapse (default: 0.0).
    top_token_anticollapse_margin: f32,
    /// `<unk>` margin penalty loss weight (default: 0.0 = disabled).
    unknown_token_penalty_weight: f32,
    /// Required target-vs-`<unk>` logit margin for penalty (default: 0.0).
    unknown_token_penalty_margin: f32,
    /// Direct thought-logit residual blend during decoding (default: 0.0).
    thought_logit_residual_weight: f32,
    /// Save best checkpoint path (auto-generated from output if not specified).
    best_checkpoint_path: String,
    /// Do not store Adam optimizer state in the final checkpoint.
    no_save_adam: bool,
    /// Hidden state dropout rate (default: 0.0 = disabled).
    hidden_dropout: f32,
    /// CfC network layers (default: 3). Only used for fresh training (not --resume).
    network_layers: usize,
    /// CfC neurons per layer (default: 8). Only used for fresh training (not --resume).
    neurons_per_layer: usize,
    /// Adaptive veto target threshold (default: 0.0 = disabled).
    /// When > 0, veto_threshold ramps from 0.0 to this value over the final
    /// veto_warmup_epochs, teaching the CfC to satisfy the veto coherence gate.
    adaptive_veto_target: f32,
    /// Epochs over which to ramp veto threshold (default: 10).
    veto_warmup_epochs: usize,
    /// Enable soft veto during training (default: false).
    enable_soft_veto_training: bool,
    /// Allow loading a checkpoint with a checksum mismatch for explicit recovery.
    allow_checksum_mismatch: bool,
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
        curriculum_style: "structured".to_string(),
        sample_count: 0,
        train_network: true,
        network_lr_scale: 0.3,
        embedding_target_norm: 128.0,
        negative_samples: 0,
        carry_state: 0.0,
        network_warmup_epochs: 0,
        fresh_adam: false,
        freeze_embeddings: false,
        sample_temperature: 1.0,
        sample_top_k: 0,
        coherence_alignment_weight: 0.0,
        coherence_alignment_start: 0.0,
        merge_token_loss_weight: 1.5,
        enable_fusion: false,
        fusion_warmup_epochs: 0,
        no_thought_seed: false,
        contrastive_weight: 0.0,
        contrastive_margin: 0.0,
        scheduled_sampling: 0.0,
        label_smoothing: 0.0,
        thought_logit_aux_weight: 0.0,
        logit_anchor_weight: 0.0,
        top_token_anticollapse_weight: 0.0,
        top_token_anticollapse_margin: 0.0,
        unknown_token_penalty_weight: 0.0,
        unknown_token_penalty_margin: 0.0,
        thought_logit_residual_weight: 0.0,
        best_checkpoint_path: String::new(),
        no_save_adam: false,
        hidden_dropout: 0.0,
        network_layers: 3,
        neurons_per_layer: 8,
        adaptive_veto_target: 0.0,
        veto_warmup_epochs: 10,
        enable_soft_veto_training: false,
        allow_checksum_mismatch: false,
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
            "--curriculum-style" => {
                i += 1;
                opts.curriculum_style = args
                    .get(i)
                    .cloned()
                    .ok_or("--curriculum-style requires structured or prose")?;
            }
            "--samples" => {
                i += 1;
                opts.sample_count = args
                    .get(i)
                    .ok_or("--samples requires a number")?
                    .parse()
                    .map_err(|_| "--samples must be a non-negative integer")?;
            }
            "--no-network-train" => {
                opts.train_network = false;
            }
            "--network-lr-scale" => {
                i += 1;
                opts.network_lr_scale = args
                    .get(i)
                    .ok_or("--network-lr-scale requires a number")?
                    .parse()
                    .map_err(|_| "--network-lr-scale must be a float")?;
            }
            "--embedding-norm" => {
                i += 1;
                opts.embedding_target_norm = args
                    .get(i)
                    .ok_or("--embedding-norm requires a number")?
                    .parse()
                    .map_err(|_| "--embedding-norm must be a float (0 to disable)")?;
            }
            "--negative-samples" | "--neg-samples" => {
                i += 1;
                opts.negative_samples = args
                    .get(i)
                    .ok_or("--negative-samples requires a number")?
                    .parse()
                    .map_err(|_| "--negative-samples must be a non-negative integer")?;
            }
            "--carry-state" => {
                i += 1;
                opts.carry_state = args
                    .get(i)
                    .ok_or("--carry-state requires a number")?
                    .parse()
                    .map_err(|_| "--carry-state must be a float 0.0-1.0")?;
            }
            "--network-warmup" => {
                i += 1;
                opts.network_warmup_epochs = args
                    .get(i)
                    .ok_or("--network-warmup requires a number")?
                    .parse()
                    .map_err(|_| "--network-warmup must be a non-negative integer")?;
            }
            "--fresh-adam" => {
                opts.fresh_adam = true;
            }
            "--no-save-adam" => {
                opts.no_save_adam = true;
            }
            "--freeze-embeddings" => {
                opts.freeze_embeddings = true;
            }
            "--temperature" => {
                i += 1;
                opts.sample_temperature = args
                    .get(i)
                    .ok_or("--temperature requires a number")?
                    .parse()
                    .map_err(|_| "--temperature must be a float")?;
            }
            "--top-k" => {
                i += 1;
                opts.sample_top_k = args
                    .get(i)
                    .ok_or("--top-k requires a number")?
                    .parse()
                    .map_err(|_| "--top-k must be a positive integer")?;
            }
            "--coherence-alignment" => {
                i += 1;
                opts.coherence_alignment_weight = args
                    .get(i)
                    .ok_or("--coherence-alignment requires a number")?
                    .parse()
                    .map_err(|_| "--coherence-alignment must be a float")?;
            }
            "--alignment-start" => {
                i += 1;
                opts.coherence_alignment_start = args
                    .get(i)
                    .ok_or("--alignment-start requires a number")?
                    .parse()
                    .map_err(|_| "--alignment-start must be a float")?;
            }
            "--merge-bias" => {
                i += 1;
                opts.merge_token_loss_weight = args
                    .get(i)
                    .ok_or("--merge-bias requires a number")?
                    .parse()
                    .map_err(|_| "--merge-bias must be a float")?;
            }
            "--no-thought-seed" => {
                opts.no_thought_seed = true;
            }
            "--contrastive" => {
                i += 1;
                opts.contrastive_weight = args
                    .get(i)
                    .ok_or("--contrastive requires a number")?
                    .parse()
                    .map_err(|_| "--contrastive must be a float")?;
            }
            "--contrastive-margin" => {
                i += 1;
                opts.contrastive_margin = args
                    .get(i)
                    .ok_or("--contrastive-margin requires a number")?
                    .parse()
                    .map_err(|_| "--contrastive-margin must be a float")?;
            }
            "--scheduled-sampling" => {
                i += 1;
                opts.scheduled_sampling = args
                    .get(i)
                    .ok_or("--scheduled-sampling requires a number")?
                    .parse()
                    .map_err(|_| "--scheduled-sampling must be a float")?;
            }
            "--label-smoothing" => {
                i += 1;
                opts.label_smoothing = args
                    .get(i)
                    .ok_or("--label-smoothing requires a number")?
                    .parse()
                    .map_err(|_| "--label-smoothing must be a float")?;
            }
            "--thought-logit-aux" => {
                i += 1;
                opts.thought_logit_aux_weight = args
                    .get(i)
                    .ok_or("--thought-logit-aux requires a number")?
                    .parse()
                    .map_err(|_| "--thought-logit-aux must be a float")?;
            }
            "--logit-anchor" => {
                i += 1;
                opts.logit_anchor_weight = args
                    .get(i)
                    .ok_or("--logit-anchor requires a number")?
                    .parse()
                    .map_err(|_| "--logit-anchor must be a float")?;
            }
            "--top-token-anticollapse" => {
                i += 1;
                opts.top_token_anticollapse_weight = args
                    .get(i)
                    .ok_or("--top-token-anticollapse requires a number")?
                    .parse()
                    .map_err(|_| "--top-token-anticollapse must be a float")?;
            }
            "--top-token-margin" => {
                i += 1;
                opts.top_token_anticollapse_margin = args
                    .get(i)
                    .ok_or("--top-token-margin requires a number")?
                    .parse()
                    .map_err(|_| "--top-token-margin must be a float")?;
            }
            "--unk-token-penalty" => {
                i += 1;
                opts.unknown_token_penalty_weight = args
                    .get(i)
                    .ok_or("--unk-token-penalty requires a number")?
                    .parse()
                    .map_err(|_| "--unk-token-penalty must be a float")?;
            }
            "--unk-token-margin" => {
                i += 1;
                opts.unknown_token_penalty_margin = args
                    .get(i)
                    .ok_or("--unk-token-margin requires a number")?
                    .parse()
                    .map_err(|_| "--unk-token-margin must be a float")?;
            }
            "--thought-logit-residual" => {
                i += 1;
                opts.thought_logit_residual_weight = args
                    .get(i)
                    .ok_or("--thought-logit-residual requires a number")?
                    .parse()
                    .map_err(|_| "--thought-logit-residual must be a float")?;
            }
            "--hidden-dropout" => {
                i += 1;
                opts.hidden_dropout = args
                    .get(i)
                    .ok_or("--hidden-dropout requires a number")?
                    .parse()
                    .map_err(|_| "--hidden-dropout must be a float")?;
            }
            "--network-layers" => {
                i += 1;
                opts.network_layers = args
                    .get(i)
                    .ok_or("--network-layers requires a number")?
                    .parse()
                    .map_err(|_| "--network-layers must be a positive integer")?;
            }
            "--neurons-per-layer" => {
                i += 1;
                opts.neurons_per_layer = args
                    .get(i)
                    .ok_or("--neurons-per-layer requires a number")?
                    .parse()
                    .map_err(|_| "--neurons-per-layer must be a positive integer")?;
            }
            "--adaptive-veto" => {
                i += 1;
                opts.adaptive_veto_target = args
                    .get(i)
                    .ok_or("--adaptive-veto requires a float (e.g., 0.15)")?
                    .parse()
                    .map_err(|_| "--adaptive-veto must be a float")?;
            }
            "--veto-warmup" => {
                i += 1;
                opts.veto_warmup_epochs = args
                    .get(i)
                    .ok_or("--veto-warmup requires a number")?
                    .parse()
                    .map_err(|_| "--veto-warmup must be a positive integer")?;
            }
            "--soft-veto-training" => {
                opts.enable_soft_veto_training = true;
            }
            "--allow-checksum-mismatch" => {
                opts.allow_checksum_mismatch = true;
            }
            "--fusion" => {
                opts.enable_fusion = true;
            }
            "--fusion-warmup" => {
                i += 1;
                opts.fusion_warmup_epochs = args
                    .get(i)
                    .ok_or("--fusion-warmup requires a number")?
                    .parse()
                    .map_err(|_| "--fusion-warmup must be a non-negative integer")?;
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
    eprintln!(
        "  --curriculum-style S Target style for --curriculum: structured or prose (default: structured)"
    );
    eprintln!("  --samples N          Generate N sample outputs after training");
    eprintln!("  --no-network-train   Only train embeddings, not CfC network weights");
    eprintln!("  --network-lr-scale F CfC network LR scale (default: 0.3)");
    eprintln!("  --embedding-norm F   Target embedding L2 norm, 0=off (default: 128.0)");
    eprintln!("  --negative-samples N Sampled softmax negatives, 0=full (default: 0)");
    eprintln!("  --carry-state F      CfC state carry probability 0-1 (default: 0.0)");
    eprintln!("  --network-warmup N   Embedding-only epochs before CfC BPTT (default: 0)");
    eprintln!("  --fresh-adam          Discard checkpoint Adam state on resume");
    eprintln!("  --no-save-adam        Save model weights without Adam optimizer state");
    eprintln!("  --freeze-embeddings   Only train CfC network, keep embeddings frozen");
    eprintln!(
        "  --allow-checksum-mismatch  Recovery only: resume despite checkpoint checksum mismatch"
    );
    eprintln!("  --temperature F       Sampling temperature for --samples (default: 1.0)");
    eprintln!("  --top-k N             Top-k sampling for --samples (default: 0 = off)");
    eprintln!("  --coherence-alignment F  Coherence alignment loss weight (default: 0.0 = off)");
    eprintln!(
        "  --alignment-start F  Starting alignment weight for curriculum annealing (default: 0.0)"
    );
    eprintln!("  --merge-bias F       Merge-token loss weight (default: 1.5, 1.0 = off)");
    eprintln!("  --no-thought-seed    Disable thought-seeded CfC initial state");
    eprintln!("  --contrastive F      Contrastive intent loss weight (default: 0.0 = off)");
    eprintln!("  --contrastive-margin F  Margin for contrastive hinge (default: 0.0)");
    eprintln!("  --label-smoothing F  Label smoothing epsilon (default: 0.0 = off)");
    eprintln!(
        "  --thought-logit-aux F  Thought-to-logit auxiliary loss weight (default: 0.0 = off)"
    );
    eprintln!("  --logit-anchor F     KL-style logit anchor weight (default: 0.0 = off)");
    eprintln!(
        "  --top-token-anticollapse F  Penalize wrong argmax token dominance (default: 0.0 = off)"
    );
    eprintln!("  --top-token-margin F Required target-vs-wrong-top margin (default: 0.0)");
    eprintln!("  --unk-token-penalty F  Penalize <unk> outranking target (default: 0.0 = off)");
    eprintln!("  --unk-token-margin F   Required target-vs-<unk> margin (default: 0.0)");
    eprintln!(
        "  --thought-logit-residual F  Blend direct thought logits into decoder logits (default: 0.0 = off)"
    );
    eprintln!("  --scheduled-sampling F  Max scheduled sampling probability (default: 0.0 = off)");
    eprintln!("  --hidden-dropout F   CfC hidden state dropout rate (default: 0.0 = off)");
    eprintln!("  --network-layers N   CfC network layers (default: 3, fresh train only)");
    eprintln!("  --neurons-per-layer N  CfC neurons per layer (default: 8, fresh train only)");
    eprintln!(
        "  --adaptive-veto F    Ramp veto threshold to F over final --veto-warmup epochs (default: 0.0 = off)"
    );
    eprintln!("  --veto-warmup N      Epochs over which to ramp veto threshold (default: 10)");
    eprintln!("  --soft-veto-training Enable soft veto (partial CfC restore) during training");
    eprintln!("  --help, -h           Show this help message");
}
