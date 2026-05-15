// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! broca-fusion-bench: A/B benchmark for algebraic fusion flags.
//!
//! Usage:
//!   cargo run --release -p symthaea-broca --bin broca_fusion_bench -- --epochs 20
//!   cargo run --release -p symthaea-broca --bin broca_fusion_bench -- --epochs 20 --sweep

use symthaea_broca::encoder::ThoughtChannels;
use symthaea_broca::evaluation::{self, EvalConfig, EvalResult};
use symthaea_broca::gating::GatingConfig;
use symthaea_broca::generator::{BrocaConfig, BrocaGenerator, SamplingStrategy};
use symthaea_broca::tokenizer::BpeTokenizer;
use symthaea_broca::training::{self, TrainingConfig, TrainingDataset, TrainingPair};

use symthaea_core::genesis::GenesisSeed;

struct BenchRun {
    name: String,
    setup: Box<dyn Fn(&mut BrocaConfig)>,
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
            name: "raw baseline (all off)".into(),
            setup: Box::new(|c| {
                c.controller.enable_compositional_logits = false;
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

fn make_dataset(tok: &BpeTokenizer) -> TrainingDataset {
    let mut ds = TrainingDataset::default();
    let texts = [
        "I understand",
        "the answer is",
        "let me clarify",
        "perhaps we should",
        "this is certain",
        "reflecting on it",
        "continuing now",
        "something new",
    ];
    for (i, text) in texts.iter().enumerate() {
        let ch = ThoughtChannels::with_intent(i % 8);
        for _ in 0..3 {
            ds.push(TrainingPair::new(ch, text.to_string(), tok));
        }
    }
    ds
}

fn run_eval(genesis: &GenesisSeed, bench: &BenchRun, dataset: &TrainingDataset) -> EvalResult {
    let mut cfg = base_config();
    (bench.setup)(&mut cfg);
    let mut gen = BrocaGenerator::new(genesis, cfg);
    let eval_cfg = EvalConfig {
        dataset: dataset.clone(),
        compute_perplexity: true,
        compute_english_ratio: true,
        per_intent_breakdown: false,
        max_gen_tokens: 20,
        eval_limit: 0,
        progress: false,
        compute_contrastive_intent: true,
    };
    evaluation::evaluate(&mut gen, &eval_cfg)
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

    let genesis = GenesisSeed::from_phrase("broca-fusion-bench-v3");
    let tok = BpeTokenizer::default_minimal();
    let dataset = make_dataset(&tok);

    // Phase 1: Eval-only benchmark (train baseline, eval with flags)
    if epochs > 0 {
        eprintln!("=== Phase 1: Train baseline ({epochs} epochs), eval with flags ===");
        let cfg = base_config();
        let mut gen = BrocaGenerator::new(&genesis, cfg);
        let tc = TrainingConfig {
            epochs,
            learning_rate: 0.01,
            bptt_window: 8,
            use_adam: true,
            train_network: true,
            network_lr_scale: 0.3,
            embedding_target_norm: 128.0,
            ..Default::default()
        };
        let m = training::train(&mut gen, &dataset, &tc);
        eprintln!(
            "  Final loss: {:.4}\n",
            m.last().map(|m| m.avg_loss).unwrap_or(0.0)
        );
    }

    eprintln!("=== Core configs ===");
    let mut results: Vec<(&str, EvalResult)> = Vec::new();
    let configs = core_configs();
    for run in &configs {
        eprint!("  {} ... ", run.name);
        let r = run_eval(&genesis, run, &dataset);
        eprintln!(
            "ppl={:.1} eng={:.3} coh={:.4}",
            r.perplexity, r.english_word_ratio, r.avg_coherence
        );
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
            let r = run_eval(&genesis, run, &dataset);
            eprintln!(
                "ppl={:.1} eng={:.3} coh={:.4}",
                r.perplexity, r.english_word_ratio, r.avg_coherence
            );
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
        let mut gen = BrocaGenerator::new(&genesis, cfg);
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
        let m = training::train(&mut gen, &dataset, &tc);
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
            dataset: dataset.clone(),
            compute_perplexity: true,
            compute_english_ratio: true,
            per_intent_breakdown: false,
            max_gen_tokens: 20,
            eval_limit: 0,
            progress: false,
            compute_contrastive_intent: true,
        };
        let r = evaluation::evaluate(&mut gen, &eval_cfg);
        eprintln!(
            "  Fusion-trained eval: ppl={:.1} eng={:.3} coh={:.4}",
            r.perplexity, r.english_word_ratio, r.avg_coherence
        );
    }
}
