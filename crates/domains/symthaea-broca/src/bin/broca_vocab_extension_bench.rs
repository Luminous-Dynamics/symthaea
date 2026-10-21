// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Initialization-only smoke benchmark for safe vocabulary extension.
//!
//! Compares the current seeded-random append path with norm-matched contextual
//! initialization. It does not use semantic labels and makes no semantic-quality
//! claim; supplied context rows are controlled vector fixtures.
//!
//! Usage:
//!   cargo run -p symthaea-broca --bin broca-vocab-extension-bench
//!   GIT_COMMIT_SHA=$(git rev-parse HEAD) cargo run -p symthaea-broca --bin broca-vocab-extension-bench -- --json-out extension.json

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::error::Error;
use symthaea_broca::controller::{
    LanguageController, LanguageControllerConfig, TokenExtensionInitialization,
};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

const BASE_VOCAB_SIZE: usize = 64;
const NEW_TOKEN_COUNT: usize = 8;
const CONTEXTS: [[usize; 3]; NEW_TOKEN_COUNT] = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
    [13, 14, 15],
    [16, 17, 18],
    [19, 20, 21],
    [22, 23, 24],
];

#[derive(Debug, Serialize)]
struct ExtensionMetrics {
    strategy: &'static str,
    old_rows_bit_identical: bool,
    old_vocab_max_abs_logit_delta: f32,
    old_vocab_top1_preserved: bool,
    median_new_to_old_norm_ratio: f64,
    mean_context_cosine: f64,
    mean_abs_pairwise_new_cosine: f64,
    max_abs_pairwise_new_cosine: f64,
    new_token_probability_mass: f64,
    conditional_old_vocab_kl_nats: f64,
    finite_new_coordinate_count: usize,
    non_finite_new_coordinate_count: usize,
}

#[derive(Debug, Serialize)]
struct ExtensionReceipt {
    schema_version: u32,
    benchmark_id: &'static str,
    evidence_level: &'static str,
    evaluation_mode: &'static str,
    source_revision: Option<&'static str>,
    genesis_phrase: &'static str,
    seed_scheme: &'static str,
    embedding_dimension: usize,
    base_vocab_size: usize,
    appended_token_count: usize,
    context_index_digest_sha256: String,
    fixed_query_domain: &'static str,
    results: Vec<ExtensionMetrics>,
    claim_boundary: &'static str,
}

#[derive(Debug, Clone, Copy)]
enum BenchStrategy {
    SeededRandomV1,
    LegacyContextBundle,
    ContextBundleNormMatched,
    ContextBundleNormMatchedWithResidual { residual_weight: f32 },
}

impl BenchStrategy {
    fn label(self) -> &'static str {
        match self {
            Self::SeededRandomV1 => "seeded_random_v1",
            Self::LegacyContextBundle => "legacy_context_bundle",
            Self::ContextBundleNormMatched => "context_bundle_norm_matched",
            Self::ContextBundleNormMatchedWithResidual { .. } => {
                "context_bundle_norm_matched_residual_0_15"
            }
        }
    }
}

fn digest<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(value)?)))
}

fn stable_softmax(logits: &[f32]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max_logit.is_finite() {
        return vec![0.0; logits.len()];
    }
    let exp_values: Vec<f64> = logits
        .iter()
        .map(|logit| (f64::from(*logit) - f64::from(max_logit)).exp())
        .collect();
    let denominator: f64 = exp_values.iter().sum();
    if !denominator.is_finite() || denominator <= 0.0 {
        return vec![0.0; logits.len()];
    }
    exp_values.iter().map(|value| value / denominator).collect()
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

fn create_base_controller(genesis: &GenesisSeed) -> LanguageController {
    let config = LanguageControllerConfig {
        network_layers: 1,
        neurons_per_layer: 2,
        vocab_size: BASE_VOCAB_SIZE,
        max_seq_len: 32,
        ..Default::default()
    };
    LanguageController::new(genesis, &config)
}

fn run_arm(
    base: &LanguageController,
    genesis: &GenesisSeed,
    query: &ContinuousHV,
    baseline_logits: &[f32],
    strategy: BenchStrategy,
) -> Result<ExtensionMetrics, Box<dyn Error>> {
    let old_vocab_size = base.vocab_size();
    let old_rows: Vec<Vec<f32>> = base
        .token_embeddings()
        .iter()
        .map(|row| row.values.clone())
        .collect();
    let old_norms: Vec<f64> = base
        .token_embeddings()
        .iter()
        .map(|row| f64::from(row.norm()))
        .collect();
    let median_old_norm = median(old_norms.clone());

    let context_rows: Vec<Vec<ContinuousHV>> = CONTEXTS
        .iter()
        .map(|indices| {
            indices
                .iter()
                .map(|index| base.token_embeddings()[*index].clone())
                .collect()
        })
        .collect();

    let mut controller = base.clone();
    let mut new_indices = Vec::with_capacity(NEW_TOKEN_COUNT);
    for (new_index, rows) in context_rows.iter().enumerate() {
        let refs: Vec<&ContinuousHV> = rows.iter().collect();
        let token_name = format!("extension-smoke-{new_index}");
        let index = controller.vocab_size();
        match strategy {
            BenchStrategy::SeededRandomV1 => {
                controller.append_token_with_initialization(
                    genesis,
                    &token_name,
                    None,
                    TokenExtensionInitialization::SeededRandomV1,
                )?;
            }
            BenchStrategy::LegacyContextBundle => {
                // Exact control for the existing append_token(context_hvs) behavior.
                controller.append_token(genesis, &token_name, Some(&refs));
            }
            BenchStrategy::ContextBundleNormMatched => {
                controller.append_token_with_initialization(
                    genesis,
                    &token_name,
                    Some(&refs),
                    TokenExtensionInitialization::ContextBundleNormMatched,
                )?;
            }
            BenchStrategy::ContextBundleNormMatchedWithResidual { residual_weight } => {
                controller.append_token_with_initialization(
                    genesis,
                    &token_name,
                    Some(&refs),
                    TokenExtensionInitialization::ContextBundleNormMatchedWithResidual {
                        residual_weight,
                    },
                )?;
            }
        }
        new_indices.push(index);
    }

    let old_rows_bit_identical = controller.token_embeddings()[..old_vocab_size]
        .iter()
        .zip(old_rows.iter())
        .all(|(row, expected)| row.values == *expected);

    controller.reset();
    let extended_logits = controller.forward_step(query, 0, 0);
    let old_vocab_max_abs_logit_delta = baseline_logits
        .iter()
        .zip(extended_logits[..old_vocab_size].iter())
        .map(|(before, after)| (before - after).abs())
        .fold(0.0f32, f32::max);
    let base_top1 = baseline_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index);
    let extended_old_top1 = extended_logits[..old_vocab_size]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index);
    let old_vocab_top1_preserved = base_top1 == extended_old_top1;

    let base_probs = stable_softmax(baseline_logits);
    let extended_probs = stable_softmax(&extended_logits);
    let new_token_probability_mass: f64 = extended_probs[old_vocab_size..].iter().sum();
    let extended_old_mass = 1.0 - new_token_probability_mass;
    let conditional_old_vocab_kl_nats = if extended_old_mass > 0.0 {
        base_probs
            .iter()
            .zip(extended_probs[..old_vocab_size].iter())
            .filter(|(p, _)| **p > 0.0)
            .map(|(p, q)| {
                let conditional_q = *q / extended_old_mass;
                if conditional_q > 0.0 {
                    *p * (*p / conditional_q).ln()
                } else {
                    f64::INFINITY
                }
            })
            .sum()
    } else {
        f64::INFINITY
    };

    let new_rows: Vec<&ContinuousHV> = new_indices
        .iter()
        .map(|index| &controller.token_embeddings()[*index])
        .collect();
    let new_norm_ratio = median(
        new_rows
            .iter()
            .map(|row| f64::from(row.norm()) / median_old_norm.max(1e-12))
            .collect(),
    );
    let mut context_similarities = Vec::new();
    for (new_row, contexts) in new_rows.iter().zip(context_rows.iter()) {
        for context in contexts {
            context_similarities.push(f64::from(new_row.similarity(context)));
        }
    }

    let mut pairwise = Vec::new();
    for left in 0..new_rows.len() {
        for right in (left + 1)..new_rows.len() {
            pairwise.push(f64::from(new_rows[left].similarity(new_rows[right])).abs());
        }
    }

    let mut finite_new_coordinate_count = 0usize;
    let mut non_finite_new_coordinate_count = 0usize;
    for row in &new_rows {
        for value in &row.values {
            if value.is_finite() {
                finite_new_coordinate_count += 1;
            } else {
                non_finite_new_coordinate_count += 1;
            }
        }
    }

    Ok(ExtensionMetrics {
        strategy: strategy.label(),
        old_rows_bit_identical,
        old_vocab_max_abs_logit_delta,
        old_vocab_top1_preserved,
        median_new_to_old_norm_ratio: new_norm_ratio,
        mean_context_cosine: if context_similarities.is_empty() {
            0.0
        } else {
            context_similarities.iter().sum::<f64>() / context_similarities.len() as f64
        },
        mean_abs_pairwise_new_cosine: if pairwise.is_empty() {
            0.0
        } else {
            pairwise.iter().sum::<f64>() / pairwise.len() as f64
        },
        max_abs_pairwise_new_cosine: pairwise.iter().copied().fold(0.0, f64::max),
        new_token_probability_mass,
        conditional_old_vocab_kl_nats,
        finite_new_coordinate_count,
        non_finite_new_coordinate_count,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let json_out = match args.iter().position(|arg| arg == "--json-out") {
        Some(index) => Some(
            args.get(index + 1)
                .filter(|path| !path.trim().is_empty())
                .ok_or("--json-out requires a non-empty path")?
                .clone(),
        ),
        None => None,
    };

    let genesis = GenesisSeed::from_phrase("broca-vocab-extension-smoke-v1");
    let base = create_base_controller(&genesis);
    let mut base_for_logits = base.clone();
    let fixed_query_domain = "extension-smoke::fixed-query";
    let query = ContinuousHV::from_genesis(
        &genesis,
        fixed_query_domain,
        symthaea_core::hdc::HDC_DIMENSION,
    );
    base_for_logits.reset();
    let baseline_logits = base_for_logits.forward_step(&query, 0, 0);

    let results = vec![
        run_arm(
            &base,
            &genesis,
            &query,
            &baseline_logits,
            BenchStrategy::SeededRandomV1,
        )?,
        run_arm(
            &base,
            &genesis,
            &query,
            &baseline_logits,
            BenchStrategy::LegacyContextBundle,
        )?,
        run_arm(
            &base,
            &genesis,
            &query,
            &baseline_logits,
            BenchStrategy::ContextBundleNormMatched,
        )?,
        run_arm(
            &base,
            &genesis,
            &query,
            &baseline_logits,
            BenchStrategy::ContextBundleNormMatchedWithResidual {
                residual_weight: 0.15,
            },
        )?,
    ];

    let receipt = ExtensionReceipt {
        schema_version: 1,
        benchmark_id: "broca-vocab-extension-init-smoke-v1",
        evidence_level: "smoke",
        evaluation_mode: "initialization_only_fixed_query",
        source_revision: option_env!("GIT_COMMIT_SHA"),
        genesis_phrase: "broca-vocab-extension-smoke-v1",
        seed_scheme: "legacy_v1_for_compatibility",
        embedding_dimension: symthaea_core::hdc::HDC_DIMENSION,
        base_vocab_size: BASE_VOCAB_SIZE,
        appended_token_count: NEW_TOKEN_COUNT,
        context_index_digest_sha256: digest(&CONTEXTS)?,
        fixed_query_domain,
        results,
        claim_boundary: "Synthetic vector contexts only; no semantic labels, training, or SOTA claim.",
    };
    let json = serde_json::to_string_pretty(&receipt)?;
    if let Some(path) = json_out {
        std::fs::write(&path, &json)?;
        eprintln!("wrote receipt to {path}");
    }

    println!(
        "{:<42} {:>9} {:>13} {:>12} {:>12} {:>12}",
        "Strategy", "Rows kept", "Max Δ logit", "Norm ratio", "Context cos", "New mass"
    );
    for result in &receipt.results {
        println!(
            "{:<42} {:>9} {:>13.6} {:>12.4} {:>12.4} {:>12.6}",
            result.strategy,
            result.old_rows_bit_identical,
            result.old_vocab_max_abs_logit_delta,
            result.median_new_to_old_norm_ratio,
            result.mean_context_cosine,
            result.new_token_probability_mass,
        );
    }
    println!();
    println!("{json}");
    Ok(())
}
