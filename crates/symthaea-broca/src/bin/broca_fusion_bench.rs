//! broca-fusion-bench: A/B benchmark for algebraic fusion flags.
//!
//! Usage:
//!   cargo run -p symthaea-broca --bin broca_fusion_bench
//!   cargo run -p symthaea-broca --bin broca_fusion_bench -- --epochs 20
//!   cargo run --release -p symthaea-broca --bin broca_fusion_bench -- --epochs 20

use symthaea_broca::controller::LanguageControllerConfig;
use symthaea_broca::encoder::ThoughtChannels;
use symthaea_broca::evaluation::{self, EvalConfig, EvalResult};
use symthaea_broca::gating::GatingConfig;
use symthaea_broca::generator::{BrocaConfig, BrocaGenerator, SamplingStrategy};
use symthaea_broca::tokenizer::BpeTokenizer;
use symthaea_broca::training::{self, TrainingConfig, TrainingDataset, TrainingPair};

use symthaea_core::genesis::GenesisSeed;

struct BenchRun {
    name: &'static str,
    setup: fn(&mut BrocaConfig),
}

fn configs() -> Vec<BenchRun> {
    vec![
        BenchRun {
            name: "baseline (all off)",
            setup: |_| {},
        },
        BenchRun {
            name: "compositional (α=0.15)",
            setup: |c| {
                c.controller.enable_compositional_logits = true;
            },
        },
        BenchRun {
            name: "adaptive alpha",
            setup: |c| {
                c.controller.enable_compositional_logits = true;
                c.controller.adaptive_compositional_alpha = true;
            },
        },
        BenchRun {
            name: "algebraic coherence",
            setup: |c| {
                c.gating.enable_algebraic_correction = true;
            },
        },
        BenchRun {
            name: "adaptive dt",
            setup: |c| {
                c.controller.enable_adaptive_dt = true;
            },
        },
        BenchRun {
            name: "soft veto",
            setup: |c| {
                c.gating.enable_soft_veto = true;
            },
        },
        BenchRun {
            name: "ALL ON",
            setup: |c| {
                c.controller.enable_compositional_logits = true;
                c.controller.adaptive_compositional_alpha = true;
                c.controller.enable_adaptive_dt = true;
                c.gating.enable_algebraic_correction = true;
                c.gating.enable_soft_veto = true;
            },
        },
    ]
}

fn base_config() -> BrocaConfig {
    BrocaConfig {
        controller: LanguageControllerConfig {
            network_layers: 3,
            neurons_per_layer: 8,
            ..Default::default()
        },
        gating: GatingConfig {
            base_max_tokens: 32,
            ..Default::default()
        },
        sampling: SamplingStrategy::Greedy,
        enable_coherence_feedback: true,
        enable_semantic_veto: false,
        ..Default::default()
    }
}

fn make_dataset(tok: &BpeTokenizer) -> TrainingDataset {
    let mut ds = TrainingDataset::default();
    let texts = [
        "I understand",
        "the answer is",
        "let me clarify",
        "perhaps we should",
    ];
    for (i, text) in texts.iter().enumerate() {
        let ch = ThoughtChannels::with_intent(i);
        for _ in 0..3 {
            ds.push(TrainingPair::new(ch, text.to_string(), tok));
        }
    }
    ds
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let epochs: usize = args
        .iter()
        .position(|a| a == "--epochs")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let genesis = GenesisSeed::from_phrase("broca-fusion-bench-v2");
    let tok = BpeTokenizer::default_minimal();
    let dataset = make_dataset(&tok);

    if epochs > 0 {
        eprintln!("Training baseline for {epochs} epochs...");
        let mut gen = BrocaGenerator::new(&genesis, base_config());
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
            "Training done. Final loss: {:.4}",
            m.last().map(|m| m.avg_loss).unwrap_or(0.0)
        );
    }

    let eval_cfg = EvalConfig {
        dataset: dataset.clone(),
        compute_perplexity: true,
        compute_english_ratio: true,
        per_intent_breakdown: false,
        max_gen_tokens: 20,
    };

    let mut results: Vec<(&str, EvalResult)> = Vec::new();
    for run in configs() {
        let mut cfg = base_config();
        (run.setup)(&mut cfg);
        let mut gen = BrocaGenerator::new(&genesis, cfg);
        eprint!("  {} ... ", run.name);
        let r = evaluation::evaluate(&mut gen, &eval_cfg);
        eprintln!(
            "ppl={:.1} eng={:.3} coh={:.4}",
            r.perplexity, r.english_word_ratio, r.avg_coherence
        );
        results.push((run.name, r));
    }

    println!(
        "\n{:<30} {:>8} {:>6} {:>8} {:>6}",
        "Config", "Perplx", "Eng%", "Coher", "Hall"
    );
    println!("{}", "-".repeat(62));
    for (name, r) in &results {
        println!(
            "{:<30} {:>8.1} {:>.3} {:>+8.4} {:>.3}",
            &name[..name.len().min(30)],
            r.perplexity,
            r.english_word_ratio,
            r.avg_coherence,
            r.hallucination_rate.unwrap_or(0.0)
        );
    }

    if let Some((_, ref base)) = results.first() {
        println!("\nDelta from baseline:");
        for (name, r) in results.iter().skip(1) {
            println!(
                "  {name}: ppl={:+.1} coh={:+.4}",
                r.perplexity - base.perplexity,
                r.avg_coherence - base.avg_coherence
            );
        }
    }
}
