// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Paired smoke experiment for legacy and versioned Broca token initialization.
//!
//! The model architecture, genesis phrase per replicate, data, optimizer, and
//! evaluation are held constant. Only the token-embedding seed scheme changes.
//!
//! Usage:
//!   cargo run -p symthaea-broca --bin broca_seed_baseline_bench
//!   cargo run -p symthaea-broca --bin broca_seed_baseline_bench -- --epochs 20 --seed-replicates 5
//!   GIT_COMMIT_SHA=$(git rev-parse HEAD) cargo run -p symthaea-broca --bin broca_seed_baseline_bench --
//!       --epochs 20 --seed-replicates 5 --json-out seed-ab.json

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::error::Error;
use symthaea_broca::controller::{
    LanguageController, LanguageControllerConfig, TokenEmbeddingSeedScheme,
};
use symthaea_broca::encoder::ThoughtChannels;
use symthaea_broca::evaluation::{self, EvalConfig, EvalResult};
use symthaea_broca::gating::GatingConfig;
use symthaea_broca::generator::{BrocaConfig, BrocaGenerator, SamplingStrategy};
use symthaea_broca::tokenizer::BpeTokenizer;
use symthaea_broca::training::{self, TrainingConfig, TrainingDataset, TrainingPair};
use symthaea_core::genesis::GenesisSeed;

const TRAIN_REPEATS: usize = 3;
const GEOMETRY_ROWS: usize = 32;

#[derive(Debug, Clone, Copy, Serialize)]
struct FixtureCase {
    intent: usize,
    epistemic: f32,
    text: &'static str,
}

const TRAIN_CASES: [FixtureCase; 8] = [
    FixtureCase { intent: 0, epistemic: 1.0, text: "I understand" },
    FixtureCase { intent: 1, epistemic: 1.0, text: "the answer is" },
    FixtureCase { intent: 2, epistemic: 1.0, text: "let me clarify" },
    FixtureCase { intent: 3, epistemic: 1.0, text: "perhaps we should" },
    FixtureCase { intent: 4, epistemic: 2.0, text: "I am not certain" },
    FixtureCase { intent: 5, epistemic: 1.0, text: "reflecting on it" },
    FixtureCase { intent: 6, epistemic: 1.0, text: "continuing now" },
    FixtureCase { intent: 7, epistemic: 3.0, text: "something new" },
];

const HELD_OUT_CASES: [FixtureCase; 8] = [
    FixtureCase { intent: 0, epistemic: 1.0, text: "I hear what you mean" },
    FixtureCase { intent: 1, epistemic: 1.0, text: "a likely answer" },
    FixtureCase { intent: 2, epistemic: 1.0, text: "could you clarify this" },
    FixtureCase { intent: 3, epistemic: 1.0, text: "we should maybe wait" },
    FixtureCase { intent: 4, epistemic: 2.0, text: "the result is uncertain" },
    FixtureCase { intent: 5, epistemic: 1.0, text: "consider this again" },
    FixtureCase { intent: 6, epistemic: 1.0, text: "please continue later" },
    FixtureCase { intent: 7, epistemic: 3.0, text: "I cannot classify this request" },
];

#[derive(Debug, Serialize)]
struct EmbeddingGeometry {
    rows_sampled: usize,
    dimension: usize,
    finite_coordinate_count: usize,
    non_finite_coordinate_count: usize,
    coordinate_mean: f64,
    coordinate_rms: f64,
    mean_absolute_pairwise_cosine: f64,
    max_absolute_pairwise_cosine: f64,
}

#[derive(Debug, Serialize)]
struct ArmResult {
    seed_replicate: usize,
    seed_scheme: TokenEmbeddingSeedScheme,
    initial_geometry: EmbeddingGeometry,
    training_final_loss: Option<f32>,
    evaluation: EvalResult,
}

#[derive(Debug, Serialize)]
struct SeedBaselineReceipt {
    schema_version: u32,
    benchmark_id: &'static str,
    evidence_level: &'static str,
    evaluation_mode: &'static str,
    source_revision: Option<&'static str>,
    epochs_per_arm: usize,
    seed_replicates: usize,
    genesis_phrase_prefix: &'static str,
    tokenizer_vocab_size: usize,
    training_pairs: usize,
    held_out_pairs: usize,
    training_fixture_sha256: String,
    held_out_fixture_sha256: String,
    training_repeats_per_case: usize,
    optimizer: &'static str,
    results: Vec<ArmResult>,
}

fn digest<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).expect("fixed in-memory fixtures must serialize");
    format!("{:x}", Sha256::digest(bytes))
}

fn base_config() -> BrocaConfig {
    BrocaConfig {
        controller: LanguageControllerConfig {
            enable_compositional_logits: false,
            adaptive_compositional_alpha: false,
            enable_adaptive_dt: false,
            ..Default::default()
        },
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

fn make_dataset(
    cases: &[FixtureCase],
    tokenizer: &BpeTokenizer,
    repeats: usize,
) -> TrainingDataset {
    let mut dataset = TrainingDataset::default();
    for case in cases {
        let mut channels = ThoughtChannels::with_intent(case.intent);
        channels.set_epistemic(case.epistemic);
        for _ in 0..repeats {
            dataset.push(TrainingPair::new(channels, case.text.to_string(), tokenizer));
        }
    }
    dataset
}

fn embedding_geometry(
    genesis: &GenesisSeed,
    vocab_size: usize,
    scheme: TokenEmbeddingSeedScheme,
) -> EmbeddingGeometry {
    let config = LanguageControllerConfig { vocab_size, ..Default::default() };
    let controller =
        LanguageController::new_with_token_embedding_seed_scheme(genesis, &config, scheme);
    let rows: Vec<_> = controller.token_embeddings().iter().take(GEOMETRY_ROWS).collect();
    let dimension = rows.first().map(|row| row.values.len()).unwrap_or(0);
    let mut finite_coordinate_count = 0usize;
    let mut non_finite_coordinate_count = 0usize;
    let mut sum = 0.0f64;
    let mut square_sum = 0.0f64;

    for row in &rows {
        for &value in &row.values {
            if value.is_finite() {
                finite_coordinate_count += 1;
                sum += f64::from(value);
                square_sum += f64::from(value) * f64::from(value);
            } else {
                non_finite_coordinate_count += 1;
            }
        }
    }

    let coordinate_count = finite_coordinate_count.max(1) as f64;
    let coordinate_mean = sum / coordinate_count;
    let coordinate_rms = (square_sum / coordinate_count).sqrt();
    let mut pair_sum = 0.0f64;
    let mut pair_count = 0usize;
    let mut max_abs_pairwise_cosine = 0.0f64;
    for i in 0..rows.len() {
        for j in (i + 1)..rows.len() {
            let similarity = rows[i].similarity(rows[j]).abs() as f64;
            pair_sum += similarity;
            pair_count += 1;
            max_abs_pairwise_cosine = max_abs_pairwise_cosine.max(similarity);
        }
    }

    EmbeddingGeometry {
        rows_sampled: rows.len(),
        dimension,
        finite_coordinate_count,
        non_finite_coordinate_count,
        coordinate_mean,
        coordinate_rms,
        mean_absolute_pairwise_cosine: if pair_count == 0 {
            0.0
        } else {
            pair_sum / pair_count as f64
        },
        max_absolute_pairwise_cosine,
    }
}

fn run_arm(
    genesis: &GenesisSeed,
    seed_replicate: usize,
    scheme: TokenEmbeddingSeedScheme,
    training_dataset: &TrainingDataset,
    held_out_dataset: &TrainingDataset,
    epochs: usize,
) -> ArmResult {
    let mut config = base_config();
    config.controller.vocab_size = BpeTokenizer::default_minimal().vocab_size();
    let initial_geometry = embedding_geometry(genesis, config.controller.vocab_size, scheme);
    let mut generator =
        BrocaGenerator::new_with_token_embedding_seed_scheme(genesis, config, scheme);

    let training_final_loss = if epochs > 0 {
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
        training::train(&mut generator, training_dataset, &training_config)
            .last()
            .map(|metrics| metrics.avg_loss)
    } else {
        None
    };

    let eval_config = EvalConfig {
        dataset: held_out_dataset.clone(),
        compute_perplexity: true,
        compute_english_ratio: true,
        per_intent_breakdown: false,
        max_gen_tokens: 20,
        eval_limit: 0,
        progress: false,
        compute_contrastive_intent: true,
    };
    let evaluation = evaluation::evaluate(&mut generator, &eval_config);

    ArmResult {
        seed_replicate,
        seed_scheme: scheme,
        initial_geometry,
        training_final_loss,
        evaluation,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let epochs: usize = args.iter().position(|arg| arg == "--epochs")
        .and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(0);
    let seed_replicates: usize = args.iter().position(|arg| arg == "--seed-replicates")
        .and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(3);
    if seed_replicates == 0 {
        return Err("--seed-replicates must be at least 1".into());
    }
    let json_out = match args.iter().position(|arg| arg == "--json-out") {
        Some(i) => Some(args.get(i + 1).filter(|path| !path.trim().is_empty())
            .ok_or("--json-out requires a non-empty path")?.clone()),
        None => None,
    };

    let tokenizer = BpeTokenizer::default_minimal();
    let training_dataset = make_dataset(&TRAIN_CASES, &tokenizer, TRAIN_REPEATS);
    let held_out_dataset = make_dataset(&HELD_OUT_CASES, &tokenizer, 1);
    let mut results = Vec::new();

    for replicate in 0..seed_replicates {
        let phrase = format!("broca-baseline-ab::replicate::{replicate}");
        let genesis = GenesisSeed::from_phrase(&phrase);
        for scheme in [
            TokenEmbeddingSeedScheme::LegacyV1,
            TokenEmbeddingSeedScheme::DomainSeparatedV2,
        ] {
            eprintln!("replicate={replicate} scheme={scheme:?} epochs={epochs}");
            results.push(run_arm(
                &genesis, replicate, scheme, &training_dataset, &held_out_dataset, epochs,
            ));
        }
    }

    let receipt = SeedBaselineReceipt {
        schema_version: 1,
        benchmark_id: "broca-genesis-token-init-v2-smoke-v1",
        evidence_level: "smoke",
        evaluation_mode: if epochs == 0 {
            "initialization_only_smoke"
        } else {
            "paired_training_held_out_smoke"
        },
        source_revision: option_env!("GIT_COMMIT_SHA"),
        epochs_per_arm: epochs,
        seed_replicates,
        genesis_phrase_prefix: "broca-baseline-ab::replicate::",
        tokenizer_vocab_size: tokenizer.vocab_size(),
        training_pairs: training_dataset.pairs.len(),
        held_out_pairs: held_out_dataset.pairs.len(),
        training_fixture_sha256: digest(&(&TRAIN_CASES[..], TRAIN_REPEATS)),
        held_out_fixture_sha256: digest(&(&HELD_OUT_CASES[..], 1usize)),
        training_repeats_per_case: TRAIN_REPEATS,
        optimizer: "Adam; lr=0.01; bptt_window=8; network_lr_scale=0.3",
        results,
    };
    let json = serde_json::to_string_pretty(&receipt)?;
    if let Some(path) = json_out {
        std::fs::write(&path, &json)?;
        eprintln!("wrote receipt to {path}");
    }

    println!("{:<10} {:<22} {:>10} {:>9} {:>10} {:>11} {:>10}",
        "Seed", "Scheme", "Perplexity", "English", "Coherence", "Mean |cos|", "NonFinite");
    for result in &receipt.results {
        println!(
            "{:<10} {:<22} {:>10.2} {:>9.3} {:>10.4} {:>11.5} {:>10}",
            result.seed_replicate,
            format!("{:?}", result.seed_scheme),
            result.evaluation.perplexity,
            result.evaluation.english_word_ratio,
            result.evaluation.avg_coherence,
            result.initial_geometry.mean_absolute_pairwise_cosine,
            result.initial_geometry.non_finite_coordinate_count,
        );
    }
    println!();
    println!("{json}");
    Ok(())
}
