// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! broca-fusion-bench: paired A/B smoke benchmark for Broca initialization and fusion.
//!
//! With --epochs N, every configuration is trained independently on the same
//! training split, then evaluated on held-out phrases. Without --epochs, output
//! is an initialization-only smoke comparison, not a trained-model result.
//!
//! Usage:
//!   cargo run --release -p symthaea-broca --bin broca_fusion_bench -- --epochs 20
//!   cargo run --release -p symthaea-broca --bin broca_fusion_bench -- --epochs 20 --sweep
//!   GIT_COMMIT_SHA=$(git rev-parse HEAD) cargo run --release -p symthaea-broca --bin broca_fusion_bench -- --epochs 20 --json-out broca-smoke.json

use symthaea_broca::encoder::ThoughtChannels;
use symthaea_broca::evaluation::{self, EvalConfig, EvalResult};
use symthaea_broca::gating::GatingConfig;
use symthaea_broca::generator::{BrocaConfig, BrocaGenerator, SamplingStrategy};
use symthaea_broca::tokenizer::BpeTokenizer;
use symthaea_broca::training::{self, TrainingConfig, TrainingDataset, TrainingPair};

use serde::Serialize;
use sha2::{Digest, Sha256};
use symthaea_core::genesis::GenesisSeed;

const GENESIS_PHRASE: &str = "broca-fusion-bench-v3";

struct BenchRun {
    name: String,
    setup: Box<dyn Fn(&mut BrocaConfig)>,
}

#[derive(Debug, Serialize)]
struct BenchArmResult {
    name: String,
    evaluation: EvalResult,
}

#[derive(Debug, Serialize)]
struct FusionBenchReport {
    schema_version: u32,
    benchmark_id: &'static str,
    evidence_level: &'static str,
    evaluation_mode: &'static str,
    source_revision: Option<&'static str>,
    genesis_phrase: &'static str,
    tokenizer_vocab_size: usize,
    training_pairs: usize,
    held_out_pairs: usize,
    training_dataset_sha256: String,
    held_out_dataset_sha256: String,
    epochs_per_arm: usize,
    results: Vec<BenchArmResult>,
}

fn dataset_sha256(dataset: &TrainingDataset) -> String {
    let canonical = serde_json::to_vec(dataset)
        .expect("the in-memory benchmark dataset must serialize deterministically");
    format!("{:x}", Sha256::digest(canonical))
}

fn base_config() -> BrocaConfig {
    // New defaults have compositional_logits + algebraic_correction ON.
    // Override both OFF for a true baseline comparison.
    BrocaConfig {
        gating: GatingConfig {
            base_max_tokens: 32,
            enable_algebraic_correction: false,
            ..Default::default()
        },
        sampling: SamplingStrategy::Greedy,
        enable_coherence_feedback: true,
        enable_semantic_veto: false,
        ..Default::default()
    }
}

fn core_configs() -> Vec<BenchRun> {
    vec![
        BenchRun {
            name: "raw baseline (genesis tokens)".into(),
            setup: Box::new(|c| {
                c.controller.enable_compositional_logits = false;
                c.enable_semantic_token_embeddings = false;
            }),
        },
        BenchRun {
            name: "semantic token init only".into(),
            setup: Box::new(|c| {
                c.controller.enable_compositional_logits = false;
                c.enable_semantic_token_embeddings = true;
            }),
        },
        BenchRun {
            name: "semantic init + comp+alg".into(),
            setup: Box::new(|c| {
                c.controller.enable_compositional_logits = true;
                c.enable_semantic_token_embeddings = true;
                c.gating.enable_algebraic_correction = true;
            }),
        },
        BenchRun {
            name: "new defaults (comp+alg)".into(),
            setup: Box::new(|c| {
                c.controller.enable_compositional_logits = true;
                c.gating.enable_algebraic_correction = true;
            }),
        },
        BenchRun {
            name: "defaults + adaptive alpha".into(),
            setup: Box::new(|c| {
                c.controller.enable_compositional_logits = true;
                c.controller.adaptive_compositional_alpha = true;
                c.gating.enable_algebraic_correction = true;
            }),
        },
        BenchRun {
            name: "defaults + adaptive dt".into(),
            setup: Box::new(|c| {
                c.controller.enable_compositional_logits = true;
                c.controller.enable_adaptive_dt = true;
                c.gating.enable_algebraic_correction = true;
            }),
        },
        BenchRun {
            name: "ALL ON".into(),
            setup: Box::new(|c| {
                c.controller.enable_compositional_logits = true;
                c.controller.adaptive_compositional_alpha = true;
                c.controller.enable_adaptive_dt = true;
                c.gating.enable_algebraic_correction = true;
                c.gating.enable_soft_veto = true;
            }),
        },
    ]
}

fn alpha_sweep_configs() -> Vec<BenchRun> {
    [0.05f32, 0.10, 0.15, 0.20, 0.25, 0.30]
        .iter()
        .map(|&alpha| BenchRun {
            name: format!("alpha={alpha:.2}"),
            setup: Box::new(move |c| {
                c.controller.enable_compositional_logits = true;
                c.controller.compositional_alpha = alpha;
                c.gating.enable_algebraic_correction = true;
            }),
        })
        .collect()
}

fn make_datasets(tok: &BpeTokenizer) -> (TrainingDataset, TrainingDataset) {
    let mut training = TrainingDataset::default();
    let training_cases = [
        // (intent channel, epistemic ordinal, target phrase)
        (0, 1.0, "I understand"),
        (1, 1.0, "the answer is"),
        (2, 1.0, "let me clarify"),
        (3, 1.0, "perhaps we should"),
        (4, 2.0, "I am not certain"),
        (5, 1.0, "reflecting on it"),
        (6, 1.0, "continuing now"),
        (7, 3.0, "something new"),
    ];
    for (intent, epistemic, text) in training_cases {
        let mut channels = ThoughtChannels::with_intent(intent);
        channels.set_epistemic(epistemic);
        for _ in 0..3 {
            training.push(TrainingPair::new(channels, text.to_string(), tok));
        }
    }

    // Hold out different surface combinations. This remains a tiny smoke split,
    // not a general language-capability benchmark, but it avoids reporting
    // training-set perplexity as held-out generalization.
    let mut evaluation = TrainingDataset::default();
    let held_out_cases = [
        // Different surface forms preserve the same intent target as training.
        (0, 1.0, "I hear what you mean"),
        (1, 1.0, "a likely answer"),
        (2, 1.0, "could you clarify this"),
        (3, 1.0, "we should maybe wait"),
        (4, 2.0, "the result is uncertain"),
        (5, 1.0, "consider this again"),
        (6, 1.0, "please continue later"),
        (7, 3.0, "I cannot classify this request"),
    ];
    for (intent, epistemic, text) in held_out_cases {
        let mut channels = ThoughtChannels::with_intent(intent);
        channels.set_epistemic(epistemic);
        evaluation.push(TrainingPair::new(channels, text.to_string(), tok));
    }

    (training, evaluation)
}

fn run_eval(
    genesis: &GenesisSeed,
    bench: &BenchRun,
    training_dataset: &TrainingDataset,
    evaluation_dataset: &TrainingDataset,
    epochs: usize,
) -> EvalResult {
    let mut cfg = base_config();
    (bench.setup)(&mut cfg);
    let mut r#gen = BrocaGenerator::new(genesis, cfg);
    if epochs > 0 {
        let training_config = TrainingConfig {
            epochs,
            learning_rate: 0.01,
            bptt_window: 8,
            use_adam: true,
            train_network: true,
            network_lr_scale: 0.3,
            embedding_target_norm: 128.0,
            ..Default::default()
        };
        // Every arm receives the same data, epoch count and optimizer profile.
        // Initialization and enabled mechanisms are the intended differences.
        let _metrics = training::train(&mut r#gen, training_dataset, &training_config);
    }
    let eval_cfg = EvalConfig {
        dataset: evaluation_dataset.clone(),
        compute_perplexity: true,
        compute_english_ratio: true,
        per_intent_breakdown: false,
        max_gen_tokens: 20,
        eval_limit: 0,
        progress: false,
        compute_contrastive_intent: true,
    };
    evaluation::evaluate(&mut r#gen, &eval_cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_receipt_digests_are_stable_and_change_with_fixture_content() {
        let tokenizer = BpeTokenizer::default_minimal();
        let (training, evaluation) = make_datasets(&tokenizer);

        let training_digest = dataset_sha256(&training);
        assert_eq!(training_digest, dataset_sha256(&training));
        assert_ne!(training_digest, dataset_sha256(&evaluation));

        let mut modified = training.clone();
        modified.pairs[0].target_text.push_str(" changed");
        assert_ne!(training_digest, dataset_sha256(&modified));
    }

    #[test]
    fn train_and_held_out_fixtures_preserve_intent_and_epistemic_targets() {
        let tokenizer = BpeTokenizer::default_minimal();
        let (training, evaluation) = make_datasets(&tokenizer);

        assert_eq!(training.pairs.len(), 24);
        assert_eq!(evaluation.pairs.len(), 8);

        // Three repeated training examples per intent and one held-out example
        // per intent. Channel 4 is uncertainty, not certainty.
        for intent in 0..8 {
            let train_case = &training.pairs[intent * 3];
            let eval_case = &evaluation.pairs[intent];

            assert_eq!(
                train_case.channels[0..8]
                    .iter()
                    .position(|value| *value == 1.0),
                Some(intent),
                "training intent mismatch at index {intent}"
            );
            assert_eq!(
                eval_case.channels[0..8]
                    .iter()
                    .position(|value| *value == 1.0),
                Some(intent),
                "held-out intent mismatch at index {intent}"
            );
        }

        assert_eq!(training.pairs[4 * 3].channels[8], 2.0);
        assert_eq!(evaluation.pairs[4].channels[8], 2.0);
        assert_eq!(training.pairs[7 * 3].channels[8], 3.0);
        assert_eq!(evaluation.pairs[7].channels[8], 3.0);
        assert!(training.pairs[4 * 3].target_text.to_lowercase().contains("not certain"));
        assert!(evaluation.pairs[4].target_text.to_lowercase().contains("uncertain"));
    }
}

fn print_table(results: &[(&str, EvalResult)]) {
    println!(
        "\n{:<32} {:>8} {:>6} {:>8} {:>6}",
        "Config", "Perplx", "Eng%", "Coher", "Hall"
    );
    println!("{}", "-".repeat(66));
    for (name, r) in results {
        println!(
            "{:<32} {:>8.1} {:>.3} {:>+8.4} {:>.3}",
            &name[..name.len().min(32)],
            r.perplexity,
            r.english_word_ratio,
            r.avg_coherence,
            r.hallucination_rate.unwrap_or(0.0)
        );
    }
    if let Some((_, ref base)) = results.first() {
        println!("\nDelta from first row:");
        for (name, r) in results.iter().skip(1) {
            println!(
                "  {name}: ppl={:+.1} eng={:+.3} coh={:+.4}",
                r.perplexity - base.perplexity,
                r.english_word_ratio - base.english_word_ratio,
                r.avg_coherence - base.avg_coherence
            );
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let epochs: usize = args
        .iter()
        .position(|a| a == "--epochs")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let do_sweep = args.iter().any(|a| a == "--sweep");
    let do_fusion_train = args.iter().any(|a| a == "--fusion-train");

    let genesis = GenesisSeed::from_phrase(GENESIS_PHRASE);
    let tok = BpeTokenizer::default_minimal();
    let (training_dataset, evaluation_dataset) = make_datasets(&tok);
    let json_out = args
        .iter()
        .position(|arg| arg == "--json-out")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let mut report_arms = Vec::new();

    if epochs == 0 {
        eprintln!(
            "Initialization-only smoke comparison: use --epochs N for paired trained evaluation."
        );
    } else {
        eprintln!(
            "Paired training: {epochs} epochs per configuration; held-out phrases reserved for evaluation."
        );
    }

    eprintln!("=== Core configs ===");
    let mut results: Vec<(&str, EvalResult)> = Vec::new();
    let configs = core_configs();
    for run in &configs {
        eprint!("  {} ... ", run.name);
        let r = run_eval(
            &genesis,
            run,
            &training_dataset,
            &evaluation_dataset,
            epochs,
        );
        eprintln!(
            "ppl={:.1} eng={:.3} coh={:.4}",
            r.perplexity, r.english_word_ratio, r.avg_coherence
        );
        report_arms.push(BenchArmResult {
            name: run.name.clone(),
            evaluation: r.clone(),
        });
        results.push((&run.name, r));
    }
    print_table(&results);

    // Phase 2: Alpha sweep
    if do_sweep {
        eprintln!("\n=== Alpha sweep ===");
        let mut sweep_results: Vec<(&str, EvalResult)> = Vec::new();
        let sweep = alpha_sweep_configs();
        for run in &sweep {
            eprint!("  {} ... ", run.name);
            let r = run_eval(
                &genesis,
                run,
                &training_dataset,
                &evaluation_dataset,
                epochs,
            );
            eprintln!(
                "ppl={:.1} eng={:.3} coh={:.4}",
                r.perplexity, r.english_word_ratio, r.avg_coherence
            );
            report_arms.push(BenchArmResult {
                name: run.name.clone(),
                evaluation: r.clone(),
            });
            sweep_results.push((&run.name, r));
        }
        print_table(&sweep_results);
    }

    // Phase 3: Training-time fusion
    if do_fusion_train && epochs > 0 {
        eprintln!("\n=== Phase 3: Training WITH fusion flags ===");
        let mut cfg = base_config();
        cfg.controller.enable_compositional_logits = true;
        cfg.controller.adaptive_compositional_alpha = true;
        cfg.controller.enable_adaptive_dt = true;
        cfg.gating.enable_algebraic_correction = true;
        let mut r#gen = BrocaGenerator::new(&genesis, cfg);
        let tc = TrainingConfig {
            epochs,
            learning_rate: 0.01,
            bptt_window: 8,
            use_adam: true,
            train_network: true,
            network_lr_scale: 0.3,
            embedding_target_norm: 128.0,
            enable_fusion_during_training: true,
            fusion_warmup_epochs: epochs / 4, // First 25% without fusion
            ..Default::default()
        };
        let m = training::train(&mut r#gen, &training_dataset, &tc);
        eprintln!(
            "  Fusion-trained final loss: {:.4}",
            m.last().map(|m| m.avg_loss).unwrap_or(0.0)
        );
        if let Some(last) = m.last() {
            if let Some(dt) = last.adaptive_dt_mean {
                eprintln!(
                    "  Adaptive dt mean: {dt:.4} [{:.4}-{:.4}]",
                    last.adaptive_dt_min.unwrap_or(0.0),
                    last.adaptive_dt_max.unwrap_or(0.0)
                );
            }
        }

        // Eval the fusion-trained model
        let eval_cfg = EvalConfig {
            dataset: evaluation_dataset.clone(),
            compute_perplexity: true,
            compute_english_ratio: true,
            per_intent_breakdown: false,
            max_gen_tokens: 20,
            eval_limit: 0,
            progress: false,
            compute_contrastive_intent: true,
        };
        let r = evaluation::evaluate(&mut r#gen, &eval_cfg);
        eprintln!(
            "  Fusion-trained eval: ppl={:.1} eng={:.3} coh={:.4}",
            r.perplexity, r.english_word_ratio, r.avg_coherence
        );
        report_arms.push(BenchArmResult {
            name: "fusion-trained".to_string(),
            evaluation: r,
        });
    }

    if let Some(path) = json_out {
        let report = FusionBenchReport {
            schema_version: 1,
            benchmark_id: "broca-fusion-smoke-v1",
            evidence_level: "smoke",
            evaluation_mode: if epochs == 0 {
                "initialization_only"
            } else {
                "paired_training_held_out"
            },
            source_revision: option_env!("GIT_COMMIT_SHA"),
            genesis_phrase: GENESIS_PHRASE,
            tokenizer_vocab_size: tok.vocab_size(),
            training_pairs: training_dataset.pairs.len(),
            held_out_pairs: evaluation_dataset.pairs.len(),
            training_dataset_sha256: dataset_sha256(&training_dataset),
            held_out_dataset_sha256: dataset_sha256(&evaluation_dataset),
            epochs_per_arm: epochs,
            results: report_arms,
        };
        let json = serde_json::to_string_pretty(&report)
            .expect("the benchmark report must serialize");
        if let Err(error) = std::fs::write(&path, json) {
            eprintln!("failed to write benchmark receipt {path}: {error}");
            std::process::exit(2);
        }
        eprintln!("Wrote smoke benchmark receipt: {path}");
    }
}