//! broca-fusion-bench: A/B benchmark for algebraic fusion flags.
//!
//! Runs evaluation with each fusion flag individually and all combined,
//! comparing against a baseline (all flags off).
//!
//! Usage:
//!   broca-fusion-bench
//!   broca-fusion-bench --checkpoint broca-cfc-v4.bin
//!   broca-fusion-bench --genesis "custom-seed"
//!   broca-fusion-bench --epochs 5  (train before eval)

use symthaea_broca::controller::LanguageControllerConfig;
use symthaea_broca::encoder::ThoughtChannels;
use symthaea_broca::evaluation::{self, EvalConfig, EvalResult};
use symthaea_broca::gating::GatingConfig;
use symthaea_broca::generator::{BrocaConfig, BrocaGenerator, SamplingStrategy};
use symthaea_broca::training::{self, TrainingConfig, TrainingDataset, TrainingPair};

use symthaea_core::genesis::GenesisSeed;

/// A single benchmark configuration with a name and the flags to set.
struct BenchConfig {
    name: &'static str,
    compositional_logits: bool,
    compositional_alpha: f32,
    algebraic_correction: bool,
    algebraic_strength: f32,
    adaptive_dt: bool,
    soft_veto: bool,
    orthogonal_positions: bool,
    spectral_gating: bool,
}

fn baseline() -> BenchConfig {
    BenchConfig {
        name: "baseline (all off)",
        compositional_logits: false,
        compositional_alpha: 0.15,
        algebraic_correction: false,
        algebraic_strength: 0.3,
        adaptive_dt: false,
        soft_veto: false,
        orthogonal_positions: false,
        spectral_gating: false,
    }
}

fn all_configs() -> Vec<BenchConfig> {
    vec![
        baseline(),
        BenchConfig {
            name: "compositional logits (α=0.15)",
            compositional_logits: true,
            ..baseline()
        },
        BenchConfig {
            name: "algebraic coherence (s=0.3)",
            algebraic_correction: true,
            ..baseline()
        },
        BenchConfig {
            name: "adaptive dt (0.005-0.08)",
            adaptive_dt: true,
            ..baseline()
        },
        BenchConfig {
            name: "soft veto (α=0.5)",
            soft_veto: true,
            ..baseline()
        },
        BenchConfig {
            name: "orthogonal positions (512)",
            orthogonal_positions: true,
            ..baseline()
        },
        BenchConfig {
            name: "spectral gating (t=0.1)",
            spectral_gating: true,
            ..baseline()
        },
        BenchConfig {
            name: "ALL fusion flags ON",
            compositional_logits: true,
            algebraic_correction: true,
            adaptive_dt: true,
            soft_veto: true,
            orthogonal_positions: true,
            spectral_gating: true,
            ..baseline()
        },
    ]
}

fn apply_config(broca: &mut BrocaConfig, bench: &BenchConfig) {
    broca.controller.enable_compositional_logits = bench.compositional_logits;
    broca.controller.compositional_alpha = bench.compositional_alpha;
    broca.controller.enable_adaptive_dt = bench.adaptive_dt;
    broca.controller.enable_orthogonal_positions = bench.orthogonal_positions;
    broca.gating.enable_algebraic_correction = bench.algebraic_correction;
    broca.gating.algebraic_correction_strength = bench.algebraic_strength;
    broca.gating.enable_soft_veto = bench.soft_veto;
    broca.gating.enable_spectral_gating = bench.spectral_gating;
}

fn make_broca_config(bench: &BenchConfig) -> BrocaConfig {
    let mut config = BrocaConfig {
        controller: LanguageControllerConfig {
            network_layers: 3,
            neurons_per_layer: 8,
            ..Default::default()
        },
        gating: GatingConfig {
            base_max_tokens: 64,
            ..Default::default()
        },
        sampling: SamplingStrategy::Greedy,
        enable_coherence_feedback: true,
        enable_semantic_veto: true,
        ..Default::default()
    };
    apply_config(&mut config, bench);
    config
}

fn make_eval_dataset(tokenizer: &symthaea_broca::tokenizer::BpeTokenizer) -> TrainingDataset {
    let mut dataset = TrainingDataset::default();
    let intents = [0, 1, 2, 3, 4, 5, 6, 7];
    let texts = [
        "I understand",
        "the answer is clear",
        "let me clarify this point",
        "perhaps we should consider",
        "I am uncertain about this",
        "reflecting on the situation",
        "continuing from where we left",
        "something unexpected happened",
    ];
    for (&intent, text) in intents.iter().zip(texts.iter()) {
        let channels = ThoughtChannels::with_intent(intent);
        for _ in 0..3 {
            dataset.push(TrainingPair::new(channels, text.to_string(), tokenizer));
        }
    }
    dataset
}

fn run_eval(genesis: &GenesisSeed, bench: &BenchConfig, dataset: &TrainingDataset) -> EvalResult {
    let config = make_broca_config(bench);
    let mut gen = BrocaGenerator::new(genesis, config);

    let eval_config = EvalConfig {
        dataset: dataset.clone(),
        compute_perplexity: true,
        compute_english_ratio: true,
        per_intent_breakdown: false,
        max_gen_tokens: 32,
    };

    evaluation::evaluate(&mut gen, &eval_config)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let genesis_phrase = args
        .iter()
        .position(|a| a == "--genesis")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("broca-fusion-bench-v1");

    let train_epochs: usize = args
        .iter()
        .position(|a| a == "--epochs")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let genesis = GenesisSeed::from_phrase(genesis_phrase);

    // Build eval dataset from the minimal tokenizer
    let temp_gen = BrocaGenerator::new(&genesis, BrocaConfig::default());
    let dataset = make_eval_dataset(temp_gen.tokenizer());
    drop(temp_gen);

    // Optional: train first
    if train_epochs > 0 {
        eprintln!("Training baseline for {train_epochs} epochs...");
        let config = make_broca_config(&baseline());
        let mut gen = BrocaGenerator::new(&genesis, config);
        let train_config = TrainingConfig {
            epochs: train_epochs,
            learning_rate: 0.01,
            bptt_window: 8,
            use_adam: true,
            train_network: true,
            network_lr_scale: 0.3,
            embedding_target_norm: 128.0,
            ..Default::default()
        };
        let metrics = training::train(&mut gen, &dataset, &train_config);
        let final_loss = metrics.last().map(|m| m.avg_loss).unwrap_or(0.0);
        eprintln!("Training complete. Final loss: {final_loss:.4}");
    }

    // Run all configurations
    let configs = all_configs();
    let mut results: Vec<(&str, EvalResult)> = Vec::new();

    for bench in &configs {
        eprint!("  Evaluating: {} ... ", bench.name);
        let result = run_eval(&genesis, bench, &dataset);
        eprintln!(
            "ppl={:.2} eng={:.3} coh={:.4} hall={:.3}",
            result.perplexity,
            result.english_word_ratio,
            result.avg_coherence,
            result.hallucination_rate.unwrap_or(0.0),
        );
        results.push((bench.name, result));
    }

    // Print comparison table
    println!();
    println!("╔══════════════════════════════════════╦══════════╦═══════╦═════════╦══════╗");
    println!("║ Configuration                        ║ Perplxty ║ Eng%  ║ Coher.  ║ Hall ║");
    println!("╠══════════════════════════════════════╬══════════╬═══════╬═════════╬══════╣");

    for (name, result) in &results {
        println!(
            "║ {:<36} ║ {:>8.2} ║ {:.3} ║ {:>+7.4} ║ {:.2} ║",
            &name[..name.len().min(36)],
            result.perplexity,
            result.english_word_ratio,
            result.avg_coherence,
            result.hallucination_rate.unwrap_or(0.0),
        );
    }
    println!("╚══════════════════════════════════════╩══════════╩═══════╩═════════╩══════╝");

    // Delta from baseline
    if let Some((_, ref baseline_result)) = results.first() {
        println!();
        println!("Delta from baseline:");
        for (name, result) in results.iter().skip(1) {
            let ppl_delta = result.perplexity - baseline_result.perplexity;
            let coh_delta = result.avg_coherence - baseline_result.avg_coherence;
            let eng_delta = result.english_word_ratio - baseline_result.english_word_ratio;
            println!(
                "  {name}: ppl={ppl_delta:+.2} eng={eng_delta:+.3} coh={coh_delta:+.4}"
            );
        }
    }
}
