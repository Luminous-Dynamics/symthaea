// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
//!
//! Eval-only mode (skip training):
//!   broca-projection-train --eval-only --resume checkpoint.bin --eval eval.jsonl

use std::io::Write;
use std::process;

use symthaea_broca::checkpoint::{AdamState, BrocaCheckpoint, ProjectionCheckpoint};

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

    // Genesis seed
    let genesis = GenesisSeed::from_phrase(&opts.genesis_phrase);

    // Build LiquidMambaGenerator
    // Build gating config with optional hedging boost override
    let mut gating_config = symthaea_broca::gating::GatingConfig::default();
    if let Some(boost) = opts.hedging_boost {
        gating_config.unknown_hedging_boost = boost;
        gating_config.uncertain_hedging_boost = boost * 0.5; // proportional
        tracing::info!(
            unknown_boost = boost,
            uncertain_boost = boost * 0.5,
            "Hedging boost override"
        );
    }

    let lm_config = LiquidMambaConfig {
        model_id: opts.model_id.clone(),
        max_tokens: opts.max_gen_tokens,
        base_lr: opts.learning_rate,
        warmup_steps: opts.warmup_steps,
        accumulation_steps: opts.accumulation_steps,
        cosine_annealing_steps: opts.cosine_annealing_steps,
        deep_projection: opts.deep_projection,
        temporal_projection: opts.temporal_projection,
        learned_pos_enc: opts.learned_pos_enc,
        temporal_stride: opts.temporal_stride,
        temporal_directional_loss: opts.directional_loss,
        temporal_smoothness_weight: opts.smoothness_weight,
        temporal_rank_reg_weight: opts.rank_reg_weight,
        temporal_learned_attention: opts.learned_attention,
        grad_clip: opts.grad_clip,
        temporal_chunk_dim: opts.chunk_size,
        temporal_num_groups: opts.num_groups,
        temporal_anticollapse_weight: opts.anticollapse,
        temporal_anticollapse_threshold: opts.anticollapse_threshold,
        temporal_rotate_grad_position: !opts.no_rotate_grad,
        temporal_adapter: opts.adapter,
        lora_rank: opts.lora_rank,
        lora_alpha: opts.lora_alpha,
        lora_lr: opts.lora_lr,
        embedding_stats_path: opts.embedding_stats.clone(),
        e2e_grad_chunks: opts.e2e_grad_chunks,
        orthogonality_weight: opts.orthogonality_weight,
        orthogonality_samples: opts.orthogonality_samples,
        gating_config,
        temperature: opts.temperature.unwrap_or(0.8),
        top_k: opts.top_k.unwrap_or(40),
        enable_semantic_attractor: opts.semantic_attractor,
        semantic_attractor_strength: opts.semantic_attractor_strength,
        semantic_attractor_top_k: opts.semantic_attractor_top_k,
        semantic_attractor_max_adjustment: opts.semantic_attractor_max_adjustment,
        semantic_attractor_normalize: true,
        // Disable runtime safety features during batch training:
        // - consciousness_gating: the PE > 0.8 gate in distill_step() blocks all
        //   gradient flow when projection is untrained (PE ≈ 1.0 always).
        //   Also gates recovery noise injection, which fights gradient updates.
        // - veto: truncates generation to ~3 tokens, starving distillation of signal
        enable_consciousness_gating: false,
        enable_veto: false,
        ..Default::default()
    };

    tracing::info!(model = %opts.model_id, "Loading Mamba model (this may download ~500MB on first run)");
    let mut r#gen = match LiquidMambaGenerator::new(&genesis, lm_config) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed to create LiquidMambaGenerator: {e}");
            process::exit(1);
        }
    };

    // Load existing projection weights if provided
    let mut resumed_diagnostics_snapshot = None;
    if let Some(ref resume_path) = opts.resume_path {
        tracing::info!(path = %resume_path, "Loading existing projection checkpoint");
        match ProjectionCheckpoint::load_from_file(resume_path) {
            Ok(ckpt) => {
                r#gen
                    .projection_mut()
                    .load_weights(&ckpt.projection_weights);
                // Load temporal weights if present
                if ckpt.temporal {
                    if let (Some(tw), Some(tp)) =
                        (&ckpt.temporal_weights, r#gen.temporal_proj_mut())
                    {
                        tp.load_weights(tw);
                        tracing::info!(
                            chunk_dim = ckpt.chunk_dim,
                            num_chunks = ckpt.num_chunks,
                            "Temporal projection weights loaded"
                        );
                    }
                }
                tracing::info!(epoch = ckpt.training_epoch, "Projection weights loaded");
                // Log diagnostics snapshot if present in the checkpoint
                if let Some(ref snap) = ckpt.diagnostics_snapshot {
                    tracing::info!(
                        total_steps = snap.total_steps,
                        clip_count = snap.clip_count,
                        last_norm_down = format!("{:.6}", snap.last_norm_down),
                        last_norm_up = format!("{:.6}", snap.last_norm_up),
                        collapse_detected = snap.collapse_detected,
                        "Restored diagnostics snapshot from checkpoint"
                    );
                }
                resumed_diagnostics_snapshot = ckpt.diagnostics_snapshot;
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

    // Enable gradient diagnostics if requested
    if opts.diagnostics {
        r#gen.enable_diagnostics();
        // Restore cumulative counters from checkpoint so resumed training
        // reports cumulative total_steps and clip_count
        if let Some(ref snap) = resumed_diagnostics_snapshot {
            if let Some(diag) = r#gen.projection_diagnostics_mut() {
                diag.restore_from_snapshot(snap);
                tracing::info!(
                    total_steps = snap.total_steps,
                    clip_count = snap.clip_count,
                    "Diagnostics counters restored from checkpoint"
                );
            }
        }
    }

    // ─── Eval-only mode: skip training, jump straight to evaluation ───
    if opts.eval_only {
        let eval_path = opts.eval_path.as_ref().unwrap(); // validated in parse_args
        run_evaluation(&mut r#gen, eval_path);
        return;
    }

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

    // Warm-start: compute principal directions from training data
    if opts.warm_start || opts.warm_start_bidirectional {
        let sample_hvs: Vec<_> = dataset
            .pairs
            .iter()
            .take(200) // Use up to 200 samples for covariance
            .map(|pair| {
                let channels = pair.to_thought_channels();
                r#gen.encoder().encode(&channels)
            })
            .collect();

        // Warm-start temporal projection if active
        if let Some(tp) = r#gen.temporal_proj_mut() {
            tracing::info!("Warm-starting temporal projection from training data PCA");
            tp.warm_start_from_samples(&sample_hvs);
        }

        // Also warm-start spatial projection
        if opts.warm_start_bidirectional {
            tracing::info!("Bidirectional warm-starting projection from training data");
            r#gen.projection_mut().warm_start_bidirectional(&sample_hvs);
        } else {
            tracing::info!("Warm-starting projection from training data covariance");
            r#gen.projection_mut().warm_start_from_samples(&sample_hvs);
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
                let channels = pair.to_thought_channels();
                r#gen.encoder().encode(&channels)
            })
            .collect();
        let (avg_dist, recon_err) = r#gen.projection_mut().contrastive_pretrain(
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
        grad_clip = opts.grad_clip,
        max_gen_tokens = opts.max_gen_tokens,
        warm_start = opts.warm_start,
        diagnostics = opts.diagnostics,
        temporal = opts.temporal_projection,
        learned_pos_enc = opts.learned_pos_enc,
        directional_loss = opts.directional_loss,
        num_groups = opts.num_groups,
        anticollapse = opts.anticollapse,
        adapter = opts.adapter,
        lora_rank = opts.lora_rank,
        rotate_grad = !opts.no_rotate_grad,
        e2e_grad_chunks = opts.e2e_grad_chunks,
        embedding_stats = opts.embedding_stats.is_some(),
        "Starting projection training"
    );

    let mut all_epoch_metrics = Vec::new();
    let mut best_pe = f32::MAX;
    let mut patience_counter = 0usize;
    let patience = opts.patience;

    for epoch in 0..opts.epochs {
        // Pos enc curriculum: unfreeze learned pos_enc after N epochs
        if opts.pos_enc_unfreeze_epoch > 0 && opts.learned_pos_enc {
            if let Some(tp) = r#gen.temporal_proj_mut() {
                if epoch < opts.pos_enc_unfreeze_epoch {
                    tp.set_learned_pos_enc(false);
                    if epoch == 0 {
                        tracing::info!(
                            unfreeze_at = opts.pos_enc_unfreeze_epoch,
                            "Pos enc curriculum: frozen for initial epochs"
                        );
                    }
                } else {
                    if !tp.learned_pos_enc() {
                        tracing::info!(epoch, "Pos enc curriculum: unfreezing learned pos_enc");
                    }
                    tp.set_learned_pos_enc(true);
                }
            }
        }

        let mut epoch_semantic_pe = 0.0f32;
        let mut epoch_coherence = 0.0f32;
        let mut epoch_effective_rank = 0.0f32;
        let mut epoch_vetos = 0usize;
        let num_samples = dataset.pairs.len();

        for (i, pair) in dataset.pairs.iter().enumerate() {
            let channels = pair.to_thought_channels();

            // Generate through full Mamba inference
            let result = r#gen.generate(&channels);

            // Distill step: update projection using Local FEP (Predictive Coding)
            let lr = r#gen.compute_lr();
            let thought_hv = r#gen.encoder().encode(&channels);
            r#gen
                .local_fep_distill(&thought_hv, &result.token_ids, lr)
                .unwrap();

            if (i + 1) % 10 == 0 {
                tracing::info!(epoch, sample = i + 1, "Sample processed");
            }

            epoch_semantic_pe += result.semantic_pe;
            epoch_coherence += result.final_coherence;
            let rank = if let Some(tp) = r#gen.temporal_proj() {
                tp.effective_rank(&result.output_hvs)
            } else {
                r#gen.projection.effective_rank(&result.output_hvs)
            };
            epoch_effective_rank += rank;

            if result.veto_triggered {
                epoch_vetos += 1;
            }

            // Periodic progress report (rank computed without injecting noise)
            if opts.diagnostics && (i + 1) % 50 == 0 && !result.output_hvs.is_empty() {
                tracing::info!(
                    epoch = epoch,
                    sample = i + 1,
                    semantic_pe = result.semantic_pe,
                    effective_rank = rank,
                    coherence = result.final_coherence,
                    "Health check"
                );
                // Unbuffered progress (tracing stderr is internally buffered)
                let phase = if result.semantic_pe < 0.5 {
                    "mamba"
                } else {
                    "roundtrip"
                };
                let _ = writeln!(
                    std::io::stderr(),
                    "[progress] epoch={epoch} sample={}/{num_samples} pe={:.4} rank={rank:.1} coh={:.4} phase={phase}",
                    i + 1,
                    result.semantic_pe,
                    result.final_coherence,
                );
                std::io::stderr().flush().ok();
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
        let avg_rank = if num_samples > 0 {
            epoch_effective_rank / num_samples as f32
        } else {
            0.0
        };

        // Compute effective rank on a few samples
        let rank_samples: Vec<_> = dataset
            .pairs
            .iter()
            .take(10)
            .map(|p| r#gen.encoder().encode(&p.to_thought_channels()))
            .collect();
        let final_rank = if let Some(tp) = r#gen.temporal_proj() {
            tp.effective_rank(&rank_samples)
        } else {
            r#gen.projection.effective_rank(&rank_samples)
        };

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
        // Unbuffered epoch summary with gradient norms
        let grad_info = if let Some(diag) = r#gen.projection_diagnostics() {
            let snap = diag.snapshot();
            let clip_rate = if snap.total_steps > 0 {
                snap.clip_count as f32 / snap.total_steps as f32 * 100.0
            } else {
                0.0
            };
            format!(
                " grad_up={:.1} grad_down={:.1} clip={:.0}%",
                snap.last_norm_up, snap.last_norm_down, clip_rate,
            )
        } else {
            String::new()
        };
        let _ = writeln!(
            std::io::stderr(),
            "[epoch] {epoch}/{} avg_pe={avg_pe:.4} rank={avg_rank:.2} coh={avg_coh:.4} vetos={epoch_vetos}{grad_info}",
            opts.epochs,
        );
        std::io::stderr().flush().ok();

        // Save best-PE checkpoint (prevents training regression from losing best weights)
        if avg_pe < best_pe - 1e-4 {
            best_pe = avg_pe;
            patience_counter = 0;
            let best_path = format!("{}.best", opts.output_path);
            let weights = r#gen.projection.flatten_weights();
            let mut ckpt = if let Some(tp) = r#gen.temporal_proj() {
                let num_groups = tp.num_groups();
                let has_adapter = tp.has_adapter();
                if num_groups > 1 || has_adapter {
                    ProjectionCheckpoint::new_temporal_with_groups(
                        weights,
                        tp.flatten_weights(),
                        r#gen.config.hdc_dim,
                        r#gen.config.bottleneck_dim,
                        r#gen.config.ssm_dim,
                        epoch,
                        tp.chunk_size(),
                        tp.num_chunks(),
                        num_groups,
                        has_adapter,
                    )
                } else {
                    ProjectionCheckpoint::new_temporal(
                        weights,
                        tp.flatten_weights(),
                        r#gen.config.hdc_dim,
                        r#gen.config.bottleneck_dim,
                        r#gen.config.ssm_dim,
                        epoch,
                        tp.chunk_size(),
                        tp.num_chunks(),
                    )
                }
            } else {
                ProjectionCheckpoint::new(
                    weights,
                    r#gen.config.hdc_dim,
                    r#gen.config.bottleneck_dim,
                    r#gen.config.ssm_dim,
                    epoch,
                    r#gen.projection.is_deep(),
                    r#gen.projection.inner_dim(),
                )
            };
            if let Some(diag) = r#gen.projection_diagnostics() {
                ckpt.diagnostics_snapshot = Some(diag.snapshot());
            }
            if let Err(e) = ckpt.save_to_file(&best_path) {
                eprintln!("Warning: failed to save best checkpoint: {e}");
            } else {
                let _ = writeln!(
                    std::io::stderr(),
                    "[best] New best PE={avg_pe:.4} at epoch {epoch}, saved to {best_path}"
                );
                std::io::stderr().flush().ok();
            }
        }

        all_epoch_metrics.push(metrics);

        // Early stopping: if PE hasn't improved for `patience` epochs, stop
        if avg_pe >= best_pe - 1e-4 {
            patience_counter += 1;
        }
        if patience > 0 && patience_counter >= patience {
            let _ = writeln!(
                std::io::stderr(),
                "[early-stop] No PE improvement for {patience} epochs (best={best_pe:.4}), stopping"
            );
            std::io::stderr().flush().ok();
            break;
        }
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

    // Print gradient diagnostics if enabled
    if let Some(diag) = r#gen.projection_diagnostics() {
        println!();
        println!("{}", diag.format_summary());
    }

    // Save projection checkpoint
    let final_epoch = all_epoch_metrics.last().map(|m| m.epoch).unwrap_or(0);
    let weights = r#gen.projection.flatten_weights();

    let mut checkpoint = if let Some(tp) = r#gen.temporal_proj() {
        let num_groups = tp.num_groups();
        let has_adapter = tp.has_adapter();
        if num_groups > 1 || has_adapter {
            ProjectionCheckpoint::new_temporal_with_groups(
                weights.clone(),
                tp.flatten_weights(),
                r#gen.config.hdc_dim,
                r#gen.config.bottleneck_dim,
                r#gen.config.ssm_dim,
                final_epoch,
                tp.chunk_size(),
                tp.num_chunks(),
                num_groups,
                has_adapter,
            )
        } else {
            ProjectionCheckpoint::new_temporal(
                weights.clone(),
                tp.flatten_weights(),
                r#gen.config.hdc_dim,
                r#gen.config.bottleneck_dim,
                r#gen.config.ssm_dim,
                final_epoch,
                tp.chunk_size(),
                tp.num_chunks(),
            )
        }
    } else {
        ProjectionCheckpoint::new(
            weights.clone(),
            r#gen.config.hdc_dim,
            r#gen.config.bottleneck_dim,
            r#gen.config.ssm_dim,
            final_epoch,
            r#gen.projection.is_deep(),
            r#gen.projection.inner_dim(),
        )
    };

    // Include full Broca state in the final optimized checkpoint
    let full_path = std::path::PathBuf::from("/tmp/broca-diag/temp-full.bin");
    r#gen
        .save_checkpoint(
            &full_path,
            final_epoch,
            0.0,
            None::<AdamState>,
            Some(weights.clone()),
            None,
        )
        .unwrap();
    let full_snapshot = BrocaCheckpoint::load_from_file(&full_path).unwrap();
    checkpoint.full_snapshot = Some(Box::new(full_snapshot));
    let _ = std::fs::remove_file(&full_path);

    // Persist gradient diagnostics snapshot if enabled
    if let Some(diag) = r#gen.projection_diagnostics() {
        checkpoint.diagnostics_snapshot = Some(diag.snapshot());
    }

    tracing::info!(path = %opts.output_path, "Saving projection checkpoint");
    if let Err(e) = checkpoint.save_to_file(&opts.output_path) {
        eprintln!("Failed to save projection checkpoint: {e}");
        process::exit(1);
    }
    println!("\nProjection checkpoint saved to: {}", opts.output_path);

    // Note best checkpoint
    let best_path = format!("{}.best", opts.output_path);
    if std::path::Path::new(&best_path).exists() {
        println!("Best checkpoint (PE={best_pe:.4}): {best_path}");
    }

    // Run evaluation if --eval provided
    if let Some(ref eval_path) = opts.eval_path {
        run_evaluation(&mut r#gen, eval_path);
    }
}

/// Run evaluation and print the formatted report.
fn run_evaluation(r#gen: &mut LiquidMambaGenerator, eval_path: &str) {
    tracing::info!(path = %eval_path, "Loading evaluation data");
    let eval_dataset = match TrainingDataset::from_jsonl(eval_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to load eval data from '{eval_path}': {e}");
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
    let result = symthaea_broca::evaluation::evaluate_liquid_mamba(r#gen, &eval_config);
    println!();
    println!(
        "{}",
        symthaea_broca::evaluation::format_liquid_mamba_eval_report(
            &result,
            &symthaea_broca::evaluation::QualityGateThresholds::default(),
        )
    );
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
    grad_clip: f32,
    warm_start: bool,
    warm_start_bidirectional: bool,
    diagnostics: bool,
    eval_only: bool,
    warmup_steps: usize,
    accumulation_steps: usize,
    cosine_annealing_steps: usize,
    contrastive_pretrain_epochs: usize,
    genesis_phrase: String,
    deep_projection: bool,
    temporal_projection: bool,
    learned_pos_enc: bool,
    temporal_stride: usize,
    directional_loss: bool,
    smoothness_weight: f32,
    rank_reg_weight: f32,
    pos_enc_unfreeze_epoch: usize,
    learned_attention: bool,
    max_gen_tokens: usize,
    // ─── Improvement A-F CLI options ───
    chunk_size: usize,
    num_groups: usize,
    anticollapse: f32,
    anticollapse_threshold: f32,
    adapter: bool,
    no_rotate_grad: bool,
    lora_rank: usize,
    lora_alpha: f32,
    lora_lr: f32,
    embedding_stats: Option<String>,
    e2e_grad_chunks: usize,
    orthogonality_weight: f32,
    orthogonality_samples: usize,
    // ─── Gate + sampling overrides ───
    hedging_boost: Option<f32>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    semantic_attractor: bool,
    semantic_attractor_strength: f32,
    semantic_attractor_top_k: usize,
    semantic_attractor_max_adjustment: f32,
    /// Early stopping patience: stop if PE doesn't improve for this many epochs (0=disabled).
    patience: usize,
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
        grad_clip: 100.0,
        warm_start: false,
        warm_start_bidirectional: false,
        diagnostics: false,
        eval_only: false,
        warmup_steps: 100,
        accumulation_steps: 4,
        cosine_annealing_steps: 0,
        contrastive_pretrain_epochs: 0,
        genesis_phrase: "broca-projection-default".to_string(),
        deep_projection: false,
        temporal_projection: false,
        learned_pos_enc: false,
        temporal_stride: 0,
        directional_loss: false,
        smoothness_weight: 0.0,
        rank_reg_weight: 0.0,
        pos_enc_unfreeze_epoch: 0,
        learned_attention: false,
        max_gen_tokens: 16,
        chunk_size: 0,
        num_groups: 1,
        anticollapse: 0.0,
        anticollapse_threshold: 0.9,
        adapter: false,
        no_rotate_grad: false,
        lora_rank: 0,
        lora_alpha: 1.0,
        lora_lr: 0.0001,
        embedding_stats: None,
        e2e_grad_chunks: 1,
        orthogonality_weight: 0.0,
        orthogonality_samples: 64,
        hedging_boost: None,
        temperature: None,
        top_k: None,
        semantic_attractor: true,
        semantic_attractor_strength: 0.5,
        semantic_attractor_top_k: 128,
        semantic_attractor_max_adjustment: 1.5,
        patience: 3,
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
            "--grad-clip" => {
                i += 1;
                opts.grad_clip = args
                    .get(i)
                    .ok_or("--grad-clip requires a number")?
                    .parse()
                    .map_err(|_| "--grad-clip must be a float")?;
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
            "--eval-only" => {
                opts.eval_only = true;
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
            "--temporal" => {
                opts.temporal_projection = true;
            }
            "--learned-pos-enc" => {
                opts.learned_pos_enc = true;
            }
            "--stride" => {
                i += 1;
                opts.temporal_stride = args
                    .get(i)
                    .ok_or("--stride requires a number")?
                    .parse()
                    .map_err(|_| "--stride must be a positive integer")?;
            }
            "--directional-loss" => {
                opts.directional_loss = true;
            }
            "--smoothness" => {
                i += 1;
                opts.smoothness_weight = args
                    .get(i)
                    .ok_or("--smoothness requires a number")?
                    .parse()
                    .map_err(|_| "--smoothness must be a float")?;
            }
            "--rank-reg" => {
                i += 1;
                opts.rank_reg_weight = args
                    .get(i)
                    .ok_or("--rank-reg requires a number")?
                    .parse()
                    .map_err(|_| "--rank-reg must be a float")?;
            }
            "--learned-attention" => {
                opts.learned_attention = true;
            }
            "--pos-enc-unfreeze" => {
                i += 1;
                opts.pos_enc_unfreeze_epoch = args
                    .get(i)
                    .ok_or("--pos-enc-unfreeze requires a number")?
                    .parse()
                    .map_err(|_| "--pos-enc-unfreeze must be a positive integer")?;
            }
            "--max-r#gen-tokens" => {
                i += 1;
                opts.max_gen_tokens = args
                    .get(i)
                    .ok_or("--max-r#gen-tokens requires a number")?
                    .parse()
                    .map_err(|_| "--max-r#gen-tokens must be a positive integer")?;
            }
            "--chunk-size" => {
                i += 1;
                opts.chunk_size = args
                    .get(i)
                    .ok_or("--chunk-size requires a number")?
                    .parse()
                    .map_err(|_| "--chunk-size must be a positive integer")?;
            }
            "--num-groups" => {
                i += 1;
                opts.num_groups = args
                    .get(i)
                    .ok_or("--num-groups requires a number")?
                    .parse()
                    .map_err(|_| "--num-groups must be a positive integer")?;
            }
            "--anticollapse" => {
                i += 1;
                opts.anticollapse = args
                    .get(i)
                    .ok_or("--anticollapse requires a number")?
                    .parse()
                    .map_err(|_| "--anticollapse must be a float")?;
            }
            "--anticollapse-threshold" => {
                i += 1;
                opts.anticollapse_threshold = args
                    .get(i)
                    .ok_or("--anticollapse-threshold requires a number")?
                    .parse()
                    .map_err(|_| "--anticollapse-threshold must be a float")?;
            }
            "--adapter" => {
                opts.adapter = true;
            }
            "--no-rotate-grad" => {
                opts.no_rotate_grad = true;
            }
            "--lora-rank" => {
                i += 1;
                opts.lora_rank = args
                    .get(i)
                    .ok_or("--lora-rank requires a number")?
                    .parse()
                    .map_err(|_| "--lora-rank must be a positive integer")?;
            }
            "--lora-alpha" => {
                i += 1;
                opts.lora_alpha = args
                    .get(i)
                    .ok_or("--lora-alpha requires a number")?
                    .parse()
                    .map_err(|_| "--lora-alpha must be a float")?;
            }
            "--lora-lr" => {
                i += 1;
                opts.lora_lr = args
                    .get(i)
                    .ok_or("--lora-lr requires a number")?
                    .parse()
                    .map_err(|_| "--lora-lr must be a float")?;
            }
            "--embedding-stats" => {
                i += 1;
                opts.embedding_stats = Some(
                    args.get(i)
                        .cloned()
                        .ok_or("--embedding-stats requires a path")?,
                );
            }
            "--e2e-grad-chunks" => {
                i += 1;
                opts.e2e_grad_chunks = args
                    .get(i)
                    .ok_or("--e2e-grad-chunks requires a number")?
                    .parse()
                    .map_err(|_| "--e2e-grad-chunks must be a positive integer")?;
            }
            "--hedging-boost" => {
                i += 1;
                opts.hedging_boost = Some(
                    args.get(i)
                        .ok_or("--hedging-boost requires a number")?
                        .parse()
                        .map_err(|_| "--hedging-boost must be a float")?,
                );
            }
            "--temperature" => {
                i += 1;
                opts.temperature = Some(
                    args.get(i)
                        .ok_or("--temperature requires a number")?
                        .parse()
                        .map_err(|_| "--temperature must be a float")?,
                );
            }
            "--top-k" => {
                i += 1;
                opts.top_k = Some(
                    args.get(i)
                        .ok_or("--top-k requires a number")?
                        .parse()
                        .map_err(|_| "--top-k must be a positive integer")?,
                );
            }
            "--no-semantic-attractor" => {
                opts.semantic_attractor = false;
            }
            "--semantic-attractor-strength" => {
                i += 1;
                opts.semantic_attractor_strength = args
                    .get(i)
                    .ok_or("--semantic-attractor-strength requires a number")?
                    .parse()
                    .map_err(|_| "--semantic-attractor-strength must be a float")?;
            }
            "--semantic-attractor-top-k" => {
                i += 1;
                opts.semantic_attractor_top_k = args
                    .get(i)
                    .ok_or("--semantic-attractor-top-k requires a number")?
                    .parse()
                    .map_err(|_| "--semantic-attractor-top-k must be a positive integer")?;
            }
            "--semantic-attractor-max-adjustment" => {
                i += 1;
                opts.semantic_attractor_max_adjustment = args
                    .get(i)
                    .ok_or("--semantic-attractor-max-adjustment requires a number")?
                    .parse()
                    .map_err(|_| "--semantic-attractor-max-adjustment must be a float")?;
            }
            "--patience" => {
                i += 1;
                opts.patience = args
                    .get(i)
                    .ok_or("--patience requires a number")?
                    .parse()
                    .map_err(|_| "--patience must be a non-negative integer")?;
            }
            "--orthogonality" | "--orth-weight" => {
                i += 1;
                opts.orthogonality_weight = args
                    .get(i)
                    .ok_or("--orthogonality requires a number")?
                    .parse()
                    .map_err(|_| "--orthogonality must be a float")?;
            }
            "--orth-samples" => {
                i += 1;
                opts.orthogonality_samples = args
                    .get(i)
                    .ok_or("--orth-samples requires a number")?
                    .parse()
                    .map_err(|_| "--orth-samples must be a positive integer")?;
            }
            "--help" | "-h" => {
                print_usage();
                process::exit(0);
            }
            arg => return Err(format!("Unknown argument: {arg}")),
        }
        i += 1;
    }

    if opts.data_path.is_empty() && !opts.eval_only {
        return Err("--data is required (unless --eval-only)".to_string());
    }

    if opts.eval_only {
        if opts.resume_path.is_none() {
            return Err("--eval-only requires --resume to load projection weights".to_string());
        }
        if opts.eval_path.is_none() {
            return Err("--eval-only requires --eval to specify evaluation data".to_string());
        }
    }

    Ok(opts)
}

fn print_usage() {
    eprintln!("Usage: broca-projection-train [OPTIONS]");
    eprintln!();
    eprintln!("Required:");
    eprintln!("  --data, -d PATH        JSONL training data file (not required with --eval-only)");
    eprintln!();
    eprintln!("Training options:");
    eprintln!("  --output, -o PATH      Output checkpoint file (default: broca-projection.bin)");
    eprintln!("  --resume, -r PATH      Load existing projection weights to continue training");
    eprintln!("  --epochs, -e N         Number of training epochs (default: 5)");
    eprintln!("  --lr RATE              Learning rate (default: 0.001)");
    eprintln!("  --grad-clip THRESH     Gradient clipping threshold (default: 100.0)");
    eprintln!("  --warm-start                PCA warm-start from training data covariance");
    eprintln!(
        "  --warm-start-bidirectional  Bidirectional warm-start (forward + backward projection)"
    );
    eprintln!("  --contrastive-pretrain N   Contrastive pretraining epochs before distillation");
    eprintln!("  --warmup-steps N            LR warmup steps (default: 100)");
    eprintln!("  --accumulation-steps N      Gradient accumulation steps (default: 4)");
    eprintln!("  --cosine-annealing-steps N  Cosine annealing total steps (default: 0 = disabled)");
    eprintln!(
        "  --genesis PHRASE            Genesis seed phrase (default: broca-projection-default)"
    );
    eprintln!(
        "  --deep-projection           Use deep double-bottleneck projection (256->128->256)"
    );
    eprintln!(
        "  --temporal                  Use temporal projection (64×256D chunks → continuous latent prompting)"
    );
    eprintln!(
        "  --learned-pos-enc           Use learned (trainable) positional encoding (requires --temporal)"
    );
    eprintln!(
        "  --stride N                  Chunk stride for temporal overlap (default: chunk_size=256, no overlap)"
    );
    eprintln!(
        "  --directional-loss          Use directional cosine loss instead of roundtrip (requires --temporal)"
    );
    eprintln!(
        "  --smoothness W              Temporal smoothness regularization weight (default: 0, try 0.01-0.1)"
    );
    eprintln!(
        "  --rank-reg W                Rank regularization weight for W_up decorrelation (default: 0, try 0.001-0.01)"
    );
    eprintln!(
        "  --learned-attention          Enable learned chunk attention weighting (requires --temporal)"
    );
    eprintln!(
        "  --pos-enc-unfreeze N        Epoch to unfreeze learned pos_enc (0 = always learned, requires --learned-pos-enc)"
    );
    eprintln!(
        "  --max-r#gen-tokens N          Max tokens per generation during training (default: 16)"
    );
    eprintln!();
    eprintln!("Architecture improvements (A-F):");
    eprintln!(
        "  --chunk-size N              Override chunk dimension for temporal projection (default: bottleneck_dim=256)"
    );
    eprintln!(
        "  --num-groups N              Number of per-group up-projection matrices (default: 1 = shared)"
    );
    eprintln!(
        "  --anticollapse W            Anti-collapse regularization weight (default: 0 = disabled, try 0.01-0.1)"
    );
    eprintln!(
        "  --anticollapse-threshold T  Cosine similarity threshold for anti-collapse (default: 0.9)"
    );
    eprintln!("  --adapter                   Enable adapter MLP after temporal up-projection");
    eprintln!(
        "  --no-rotate-grad            Disable rotating grad-chunk position (use all chunks for E2E)"
    );
    eprintln!("  --lora-rank N               LoRA rank for Mamba layer adaptation (0 = disabled)");
    eprintln!("  --lora-alpha A              LoRA alpha scaling factor (default: 1.0)");
    eprintln!("  --lora-lr LR                LoRA learning rate (default: 0.0001)");
    eprintln!(
        "  --embedding-stats PATH      Pre-computed embedding stats for manifold moment matching adapter init"
    );
    eprintln!(
        "  --e2e-grad-chunks K         Simultaneous E2E gradient positions per step (default: 1 = legacy rotating)"
    );
    eprintln!();
    eprintln!("Gate + sampling overrides:");
    eprintln!(
        "  --hedging-boost N      Override epistemic hedging boost (default: 5.0, Unknown=N, Uncertain=N/2)"
    );
    eprintln!("  --temperature T        Sampling temperature (default: 0.8)");
    eprintln!("  --top-k K              Top-k sampling (default: 40)");
    eprintln!(
        "  --no-semantic-attractor              Disable HDC semantic attractor during generation"
    );
    eprintln!(
        "  --semantic-attractor-strength F      Semantic attractor logit strength (default: 0.5)"
    );
    eprintln!("  --semantic-attractor-top-k K         Candidate count to inspect (default: 128)");
    eprintln!(
        "  --semantic-attractor-max-adjustment F  Max absolute logit adjustment (default: 1.5)"
    );
    eprintln!(
        "  --patience N           Early stopping: stop if PE doesn't improve for N epochs (default: 3, 0=disabled)"
    );
    eprintln!();
    eprintln!("Evaluation options:");
    eprintln!("  --eval PATH            Held-out JSONL for post-training evaluation");
    eprintln!(
        "  --eval-only            Skip training, run evaluation only (requires --resume + --eval)"
    );
    eprintln!();
    eprintln!("Diagnostics:");
    eprintln!("  --diagnostics          Enable gradient diagnostics and periodic health logging");
    eprintln!();
    eprintln!("Other:");
    eprintln!("  --model ID             HuggingFace model ID (default: state-spaces/mamba-130m)");
    eprintln!("  --help, -h             Show this help message");
}
