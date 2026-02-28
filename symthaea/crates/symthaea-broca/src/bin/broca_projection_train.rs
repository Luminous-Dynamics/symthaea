//! broca-projection-train: Batch training for the HDC↔SSM projection bridge.
//!
//! Trains the 8.8M parameter projection (16,384D → 256D → 768D) that connects
//! Symthaea's HDC thought space to Mamba's SSM hidden space. Mamba stays frozen;
//! only the projection lens is trained.
//!
//! Usage:
//!   broca-projection-train --data train.jsonl [--eval eval.jsonl] [--epochs 5]
//!     [--lr 0.001] [--warm-start] [--diagnostics] [--output broca-projection.bin]
//!     [--model state-spaces/mamba-130m] [--genesis PHRASE]

use std::process;

use symthaea_broca::checkpoint::ProjectionCheckpoint;
use symthaea_broca::encoder::ThoughtChannels;
use symthaea_broca::liquid_mamba::{LiquidMambaConfig, LiquidMambaGenerator};
use symthaea_broca::training::TrainingDataset;

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

    // Load training dataset
    tracing::info!(path = %opts.data_path, "Loading training data");
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
    tracing::info!(pairs = dataset.len(), "Dataset loaded");

    // Genesis seed
    let genesis = GenesisSeed::from_phrase(&opts.genesis_phrase);

    // Build LiquidMambaGenerator
    let lm_config = LiquidMambaConfig {
        model_id: opts.model_id.clone(),
        max_tokens: 64,
        base_lr: opts.learning_rate,
        warmup_steps: opts.warmup_steps,
        accumulation_steps: opts.accumulation_steps,
        cosine_annealing_steps: opts.cosine_annealing_steps,
        deep_projection: opts.deep_projection,
        ..Default::default()
    };

    tracing::info!(model = %opts.model_id, "Loading Mamba model (this may download ~500MB on first run)");
    let mut gen = match LiquidMambaGenerator::new(&genesis, lm_config) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed to create LiquidMambaGenerator: {e}");
            process::exit(1);
        }
    };

    // Load existing projection weights if provided
    if let Some(ref resume_path) = opts.resume_path {
        tracing::info!(path = %resume_path, "Loading existing projection checkpoint");
        match ProjectionCheckpoint::load_from_file(resume_path) {
            Ok(ckpt) => {
                gen.projection_mut().load_weights(&ckpt.projection_weights);
                tracing::info!(epoch = ckpt.training_epoch, "Projection weights loaded");
            }
            Err(e) => {
                eprintln!(
                    "Failed to load projection checkpoint '{}': {e}",
                    resume_path
                );
                process::exit(1);
            }
        }
    }

    // Warm-start: compute principal directions from training data
    if opts.warm_start || opts.warm_start_bidirectional {
        let sample_hvs: Vec<_> = dataset
            .pairs
            .iter()
            .take(200) // Use up to 200 samples for covariance
            .map(|pair| {
                let channels = ThoughtChannels {
                    channels: pair.channels,
                };
                gen.encoder().encode(&channels)
            })
            .collect();
        if opts.warm_start_bidirectional {
            tracing::info!("Bidirectional warm-starting projection from training data");
            gen.projection_mut().warm_start_bidirectional(&sample_hvs);
        } else {
            tracing::info!("Warm-starting projection from training data covariance");
            gen.projection_mut().warm_start_from_samples(&sample_hvs);
        }
    }

    // Contrastive pretraining: learn to separate thoughts before distillation
    if opts.contrastive_pretrain_epochs > 0 {
        tracing::info!(
            epochs = opts.contrastive_pretrain_epochs,
            "Running contrastive pretraining"
        );
        let sample_hvs: Vec<_> = dataset
            .pairs
            .iter()
            .take(50) // Diverse subset — 50 thoughts → 1,225 pairs
            .map(|pair| {
                let channels = ThoughtChannels {
                    channels: pair.channels,
                };
                gen.encoder().encode(&channels)
            })
            .collect();
        let (avg_dist, recon_err) = gen.projection_mut().contrastive_pretrain(
            &sample_hvs,
            opts.contrastive_pretrain_epochs,
            opts.learning_rate * 0.5, // Lower LR for pretraining
        );
        tracing::info!(
            avg_distance = format!("{avg_dist:.4}"),
            recon_error = format!("{recon_err:.4}"),
            "Contrastive pretraining complete"
        );
    }

    // Training loop
    tracing::info!(
        epochs = opts.epochs,
        lr = opts.learning_rate,
        warm_start = opts.warm_start,
        diagnostics = opts.diagnostics,
        "Starting projection training"
    );

    let mut all_epoch_metrics = Vec::new();

    for epoch in 0..opts.epochs {
        let mut epoch_semantic_pe = 0.0f32;
        let mut epoch_coherence = 0.0f32;
        let mut epoch_vetos = 0usize;
        let num_samples = dataset.pairs.len();

        for (i, pair) in dataset.pairs.iter().enumerate() {
            let channels = ThoughtChannels {
                channels: pair.channels,
            };

            // Generate through full Mamba inference
            let result = gen.generate(&channels);

            // Distill step: update projection from reconstruction error
            gen.distill_step(&channels, &result);

            epoch_semantic_pe += result.semantic_pe;
            epoch_coherence += result.final_coherence;
            if result.veto_triggered {
                epoch_vetos += 1;
            }

            // Periodic projection health check
            if opts.diagnostics && (i + 1) % 50 == 0 && !result.output_hvs.is_empty() {
                gen.check_projection_health(&result.output_hvs);
                let rank = gen.projection().effective_rank(&result.output_hvs);
                tracing::info!(
                    epoch = epoch,
                    sample = i + 1,
                    semantic_pe = result.semantic_pe,
                    effective_rank = rank,
                    coherence = result.final_coherence,
                    "Health check"
                );
            }
        }

        let avg_pe = if num_samples > 0 {
            epoch_semantic_pe / num_samples as f32
        } else {
            1.0
        };
        let avg_coh = if num_samples > 0 {
            epoch_coherence / num_samples as f32
        } else {
            0.0
        };

        // Compute effective rank on a few samples
        let rank_samples: Vec<_> = dataset
            .pairs
            .iter()
            .take(10)
            .map(|p| {
                gen.encoder().encode(&ThoughtChannels {
                    channels: p.channels,
                })
            })
            .collect();
        let avg_rank = gen.projection().effective_rank(&rank_samples);

        let metrics = ProjectionEpochMetrics {
            epoch,
            avg_semantic_pe: avg_pe,
            avg_effective_rank: avg_rank,
            avg_coherence: avg_coh,
            num_vetos: epoch_vetos,
        };

        tracing::info!(
            epoch = epoch,
            avg_pe = format!("{:.4}", avg_pe),
            avg_rank = format!("{:.2}", avg_rank),
            avg_coherence = format!("{:.4}", avg_coh),
            vetos = epoch_vetos,
            "Epoch complete"
        );

        all_epoch_metrics.push(metrics);
    }

    // Print training summary
    println!("\n--- Projection Training Summary ---");
    println!(
        "{:>6}  {:>10}  {:>8}  {:>10}  {:>6}",
        "Epoch", "Semantic PE", "Rank", "Coherence", "Vetos"
    );
    for m in &all_epoch_metrics {
        println!(
            "{:>6}  {:>10.4}  {:>8.2}  {:>10.4}  {:>6}",
            m.epoch, m.avg_semantic_pe, m.avg_effective_rank, m.avg_coherence, m.num_vetos
        );
    }

    // Save projection checkpoint
    let final_epoch = all_epoch_metrics.last().map(|m| m.epoch).unwrap_or(0);
    let weights = gen.projection().flatten_weights();

    let mut checkpoint = ProjectionCheckpoint::new(
        weights,
        gen.config().hdc_dim,
        gen.config().bottleneck_dim,
        gen.config().ssm_dim,
        final_epoch,
        gen.projection().is_deep(),
        gen.projection().inner_dim(),
    );

    tracing::info!(path = %opts.output_path, "Saving projection checkpoint");
    if let Err(e) = checkpoint.save_to_file(&opts.output_path) {
        eprintln!("Failed to save projection checkpoint: {e}");
        process::exit(1);
    }
    println!("\nProjection checkpoint saved to: {}", opts.output_path);

    // Run evaluation if --eval provided
    if let Some(ref eval_path) = opts.eval_path {
        tracing::info!(path = %eval_path, "Loading evaluation data");
        let eval_dataset = match TrainingDataset::from_jsonl(eval_path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Failed to load eval data from '{}': {e}", eval_path);
                process::exit(1);
            }
        };

        let eval_config = symthaea_broca::evaluation::LiquidMambaEvalConfig {
            dataset: eval_dataset,
            compute_perplexity: true,
            compute_english_ratio: true,
            per_intent_breakdown: true,
            max_gen_tokens: 64,
            consciousness_gating_test: true,
        };

        tracing::info!("Running evaluation");
        let result = symthaea_broca::evaluation::evaluate_liquid_mamba(&mut gen, &eval_config);
        println!();
        println!(
            "{}",
            symthaea_broca::evaluation::format_liquid_mamba_eval_report(&result)
        );
    }
}

/// Per-epoch training metrics for projection training.
struct ProjectionEpochMetrics {
    epoch: usize,
    avg_semantic_pe: f32,
    avg_effective_rank: f32,
    avg_coherence: f32,
    num_vetos: usize,
}

struct ProjectionTrainOpts {
    data_path: String,
    output_path: String,
    resume_path: Option<String>,
    eval_path: Option<String>,
    model_id: String,
    epochs: usize,
    learning_rate: f32,
    warm_start: bool,
    warm_start_bidirectional: bool,
    diagnostics: bool,
    warmup_steps: usize,
    accumulation_steps: usize,
    cosine_annealing_steps: usize,
    contrastive_pretrain_epochs: usize,
    genesis_phrase: String,
    deep_projection: bool,
}

fn parse_args(args: &[String]) -> Result<ProjectionTrainOpts, String> {
    let mut opts = ProjectionTrainOpts {
        data_path: String::new(),
        output_path: "broca-projection.bin".to_string(),
        resume_path: None,
        eval_path: None,
        model_id: "state-spaces/mamba-130m".to_string(),
        epochs: 5,
        learning_rate: 0.001,
        warm_start: false,
        warm_start_bidirectional: false,
        diagnostics: false,
        warmup_steps: 100,
        accumulation_steps: 4,
        cosine_annealing_steps: 0,
        contrastive_pretrain_epochs: 0,
        genesis_phrase: "broca-projection-default".to_string(),
        deep_projection: false,
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
            "--model" => {
                i += 1;
                opts.model_id = args.get(i).cloned().ok_or("--model requires a model ID")?;
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
            "--warm-start" => {
                opts.warm_start = true;
            }
            "--warm-start-bidirectional" => {
                opts.warm_start_bidirectional = true;
            }
            "--cosine-annealing-steps" => {
                i += 1;
                opts.cosine_annealing_steps = args
                    .get(i)
                    .ok_or("--cosine-annealing-steps requires a number")?
                    .parse()
                    .map_err(|_| "--cosine-annealing-steps must be a positive integer")?;
            }
            "--contrastive-pretrain" => {
                i += 1;
                opts.contrastive_pretrain_epochs = args
                    .get(i)
                    .ok_or("--contrastive-pretrain requires a number")?
                    .parse()
                    .map_err(|_| "--contrastive-pretrain must be a positive integer")?;
            }
            "--diagnostics" => {
                opts.diagnostics = true;
            }
            "--warmup-steps" => {
                i += 1;
                opts.warmup_steps = args
                    .get(i)
                    .ok_or("--warmup-steps requires a number")?
                    .parse()
                    .map_err(|_| "--warmup-steps must be a positive integer")?;
            }
            "--accumulation-steps" => {
                i += 1;
                opts.accumulation_steps = args
                    .get(i)
                    .ok_or("--accumulation-steps requires a number")?
                    .parse()
                    .map_err(|_| "--accumulation-steps must be a positive integer")?;
            }
            "--genesis" => {
                i += 1;
                opts.genesis_phrase = args.get(i).cloned().ok_or("--genesis requires a phrase")?;
            }
            "--deep-projection" => {
                opts.deep_projection = true;
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
    eprintln!("Usage: broca-projection-train [OPTIONS]");
    eprintln!();
    eprintln!("Required:");
    eprintln!("  --data, -d PATH        JSONL training data file");
    eprintln!();
    eprintln!("Optional:");
    eprintln!("  --output, -o PATH      Output checkpoint file (default: broca-projection.bin)");
    eprintln!("  --resume, -r PATH      Load existing projection weights to continue training");
    eprintln!("  --eval PATH            Held-out JSONL for post-training evaluation");
    eprintln!("  --model ID             HuggingFace model ID (default: state-spaces/mamba-130m)");
    eprintln!("  --epochs, -e N         Number of training epochs (default: 5)");
    eprintln!("  --lr RATE              Learning rate (default: 0.001)");
    eprintln!("  --warm-start                PCA warm-start from training data covariance");
    eprintln!(
        "  --warm-start-bidirectional  Bidirectional warm-start (forward + backward projection)"
    );
    eprintln!("  --contrastive-pretrain N   Contrastive pretraining epochs before distillation");
    eprintln!("  --diagnostics               Enable periodic projection health logging");
    eprintln!("  --warmup-steps N            LR warmup steps (default: 100)");
    eprintln!("  --accumulation-steps N      Gradient accumulation steps (default: 4)");
    eprintln!("  --cosine-annealing-steps N  Cosine annealing total steps (default: 0 = disabled)");
    eprintln!(
        "  --genesis PHRASE            Genesis seed phrase (default: broca-projection-default)"
    );
    eprintln!("  --deep-projection           Use deep double-bottleneck projection (256→128→256)");
    eprintln!("  --help, -h                  Show this help message");
}
