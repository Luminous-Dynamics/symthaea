// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Synthetic, held-out benchmark for angular calibration of new vocabulary rows.
//!
//! This benchmark creates paired positive/protected query families around a
//! shared compositional context. It intentionally includes an ambiguity trap:
//! context-only initialization should activate on both the intended and protected
//! side, while contrastive calibration may separate them. This is a controlled
//! geometry test, not a semantic-quality or SOTA claim.
//!
//! Usage:
//!   cargo run -p symthaea-broca --bin broca-angular-calibration-bench
//!   cargo run -p symthaea-broca --bin broca-angular-calibration-bench -- --json-out angular.json

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::error::Error;
use symthaea_broca::controller::{LanguageController, LanguageControllerConfig};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

const BASE_VOCAB_SIZE: usize = 64;
const NEW_TOKEN_COUNT: usize = 8;
const TRAIN_QUERY_COUNT: usize = 8;
const HELDOUT_QUERY_COUNT: usize = 64;
const LOGIT_SCALE: f32 = 20.0;
const GENESIS_PHRASE: &str = "broca-angular-calibration-smoke-v1";
const MAX_TRAIN_FALSE_ACTIVATION_RATE: f64 = 0.01;
const CONFIDENCE_ALPHA: f64 = 0.05;
const LAMBDA_GRID: [f32; 11] = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0];

#[derive(Debug)]
struct Scenario {
    context: ContinuousHV,
    positive_train: Vec<ContinuousHV>,
    positive_heldout: Vec<ContinuousHV>,
    protected_train: Vec<ContinuousHV>,
    protected_heldout: Vec<ContinuousHV>,
}

#[derive(Debug, Clone, Copy)]
struct EvalCounts {
    positive_count: usize,
    positive_target_beats_old: usize,
    positive_target_is_global_top1: usize,
    protected_count: usize,
    protected_false_activations: usize,
    positive_margin_sum: f64,
    protected_safety_margin_sum: f64,
}

#[derive(Debug, Serialize, Clone)]
struct Evaluation {
    positive_contexts: usize,
    intended_token_beats_old_vocab_fraction: f64,
    intended_token_global_top1_fraction: f64,
    protected_contexts: usize,
    protected_false_activation_count: usize,
    protected_false_activation_rate: f64,
    protected_false_activation_upper_95: f64,
    mean_positive_margin_over_old_vocab: f64,
    mean_protected_old_best_margin_over_new_vocab: f64,
}

#[derive(Debug, Serialize)]
struct LambdaTrial {
    lambda: f32,
    train_positive_global_top1_fraction: f64,
    train_protected_false_activation_rate: f64,
    train_protected_false_activation_upper_95: f64,
    train_mean_positive_margin: f64,
}

#[derive(Debug, Serialize)]
struct StrategyResult {
    strategy: &'static str,
    heldout: Evaluation,
    mean_abs_pairwise_new_candidate_cosine: f64,
    max_abs_pairwise_new_candidate_cosine: f64,
}

#[derive(Debug, Serialize)]
struct Receipt {
    schema_version: u32,
    benchmark_id: &'static str,
    evidence_level: &'static str,
    evaluation_mode: &'static str,
    source_revision: Option<&'static str>,
    base_vocab_size: usize,
    appended_token_count: usize,
    embedding_dimension: usize,
    train_queries_per_class_and_side: usize,
    heldout_queries_per_class_and_side: usize,
    seed_scheme: &'static str,
    selected_lambda: f32,
    maximum_allowed_train_false_activation_rate: f64,
    selected_train_false_activation_rate: f64,
    selected_train_false_activation_upper_95: f64,
    calibration_constraint_met: bool,
    heldout_false_activation_confidence: &'static str,
    false_activation_bound_assumption: &'static str,
    lambda_trials: Vec<LambdaTrial>,
    results: Vec<StrategyResult>,
    fixture_spec_digest_sha256: String,
    claim_boundary: &'static str,
}


/// Log-sum-exp for two log probabilities.
fn log_add_exp(a: f64, b: f64) -> f64 {
    if !a.is_finite() {
        return b;
    }
    if !b.is_finite() {
        return a;
    }
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

/// Binomial CDF P[X <= observed] evaluated in log space.
fn binomial_cdf_log(observed: usize, trials: usize, probability: f64) -> f64 {
    if probability <= 0.0 {
        return 1.0;
    }
    if probability >= 1.0 {
        return if observed >= trials { 1.0 } else { 0.0 };
    }
    let mut log_probability = (trials as f64) * (-probability).ln_1p();
    let mut log_cdf = log_probability;
    for k in 1..=observed.min(trials) {
        log_probability += ((trials - k + 1) as f64).ln()
            - (k as f64).ln()
            + probability.ln()
            - (-probability).ln_1p();
        log_cdf = log_add_exp(log_cdf, log_probability);
    }
    log_cdf.exp().clamp(0.0, 1.0)
}

/// One-sided exact Clopper-Pearson upper confidence bound for a binomial rate.
fn clopper_pearson_upper(observed: usize, trials: usize, alpha: f64) -> f64 {
    if trials == 0 || observed >= trials {
        return 1.0;
    }
    if observed == 0 {
        return 1.0 - alpha.powf(1.0 / trials as f64);
    }
    let mut low = observed as f64 / trials as f64;
    let mut high = 1.0;
    for _ in 0..80 {
        let middle = (low + high) * 0.5;
        if binomial_cdf_log(observed, trials, middle) > alpha {
            low = middle;
        } else {
            high = middle;
        }
    }
    high
}

fn make_controller(genesis: &GenesisSeed) -> LanguageController {
    let config = LanguageControllerConfig {
        network_layers: 1,
        neurons_per_layer: 2,
        vocab_size: BASE_VOCAB_SIZE,
        max_seq_len: 32,
        logit_scale: LOGIT_SCALE,
        ..Default::default()
    };
    LanguageController::new(genesis, &config)
}

fn orthogonal_direction(
    genesis: &GenesisSeed,
    shared: &ContinuousHV,
    name: &str,
) -> ContinuousHV {
    let mut direction = ContinuousHV::from_genesis(
        genesis,
        &format!("broca-angular::feature::{name}"),
        HDC_DIMENSION,
    )
    .normalize();
    direction.add_scaled(shared, -direction.similarity(shared));
    direction.normalize()
}

fn query_around(
    genesis: &GenesisSeed,
    context: &ContinuousHV,
    feature: &ContinuousHV,
    sign: f32,
    name: &str,
    sample_index: usize,
) -> ContinuousHV {
    let mut query = context.clone();
    query.add_scaled(feature, sign);
    let noise = ContinuousHV::from_genesis(
        genesis,
        &format!("broca-angular::noise::{name}::{sample_index}"),
        HDC_DIMENSION,
    )
    .normalize();
    query.add_scaled(&noise, 0.25);
    query.normalize()
}

fn build_scenarios(base: &LanguageController, genesis: &GenesisSeed) -> Vec<Scenario> {
    (0..NEW_TOKEN_COUNT)
        .map(|concept| {
            let left = concept * 2;
            let right = left + 1;
            let contexts = [&base.token_embeddings()[left], &base.token_embeddings()[right]];
            let shared = ContinuousHV::bundle(&contexts).normalize();
            let feature = orthogonal_direction(genesis, &shared, &format!("{concept}"));

            let positive_train = (0..TRAIN_QUERY_COUNT)
                .map(|i| {
                    query_around(
                        genesis,
                        &shared,
                        &feature,
                        1.0,
                        &format!("positive-train-{concept}"),
                        i,
                    )
                })
                .collect();
            let positive_heldout = (0..HELDOUT_QUERY_COUNT)
                .map(|i| {
                    query_around(
                        genesis,
                        &shared,
                        &feature,
                        1.0,
                        &format!("positive-heldout-{concept}"),
                        i,
                    )
                })
                .collect();
            let protected_train = (0..TRAIN_QUERY_COUNT)
                .map(|i| {
                    query_around(
                        genesis,
                        &shared,
                        &feature,
                        -1.0,
                        &format!("protected-train-{concept}"),
                        i,
                    )
                })
                .collect();
            let protected_heldout = (0..HELDOUT_QUERY_COUNT)
                .map(|i| {
                    query_around(
                        genesis,
                        &shared,
                        &feature,
                        -1.0,
                        &format!("protected-heldout-{concept}"),
                        i,
                    )
                })
                .collect();

            Scenario {
                context: shared,
                positive_train,
                positive_heldout,
                protected_train,
                protected_heldout,
            }
        })
        .collect()
}

fn positive_centroid(samples: &[ContinuousHV]) -> ContinuousHV {
    let refs: Vec<&ContinuousHV> = samples.iter().collect();
    ContinuousHV::bundle(&refs).normalize()
}

fn context_with_residual(context: &ContinuousHV, residual: &ContinuousHV, weight: f32) -> ContinuousHV {
    let mut candidate = context.clone();
    candidate.add_scaled(residual, weight);
    candidate.normalize()
}

fn contrastive_candidate(
    scenario: &Scenario,
    lambda: f32,
    use_heldout: bool,
) -> ContinuousHV {
    let positives = if use_heldout {
        &scenario.positive_heldout
    } else {
        &scenario.positive_train
    };
    let protected = if use_heldout {
        &scenario.protected_heldout
    } else {
        &scenario.protected_train
    };
    let positive = positive_centroid(positives);
    let negative = positive_centroid(protected);
    let mut candidate = scenario.context.clone();
    candidate.add_scaled(&positive, lambda);
    candidate.add_scaled(&negative, -lambda);
    candidate.normalize()
}

fn strategy_candidates(
    scenarios: &[Scenario],
    genesis: &GenesisSeed,
    strategy: &str,
    lambda: f32,
) -> Vec<ContinuousHV> {
    scenarios
        .iter()
        .enumerate()
        .map(|(index, scenario)| match strategy {
            "seeded_random" => ContinuousHV::from_genesis(
                genesis,
                &format!("broca-angular::random-candidate::{index}"),
                HDC_DIMENSION,
            )
            .normalize(),
            "context_bundle" => scenario.context.clone(),
            "positive_centroid" => positive_centroid(&scenario.positive_train),
            "context_plus_residual_0_15" => {
                let residual = ContinuousHV::from_genesis(
                    genesis,
                    &format!("broca-angular::residual::{index}"),
                    HDC_DIMENSION,
                )
                .normalize();
                context_with_residual(&scenario.context, &residual, 0.15)
            }
            "contrastive_calibrated" => contrastive_candidate(scenario, lambda, false),
            _ => scenario.context.clone(),
        })
        .collect()
}

fn best(values: &[f32]) -> f32 {
    values.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

fn candidate_logits(
    query: &ContinuousHV,
    candidates: &[ContinuousHV],
) -> Vec<f32> {
    candidates
        .iter()
        .map(|candidate| LOGIT_SCALE * query.similarity(candidate))
        .collect()
}

fn evaluate(
    controller: &LanguageController,
    scenarios: &[Scenario],
    candidates: &[ContinuousHV],
    heldout: bool,
) -> (Evaluation, EvalCounts) {
    let mut counts = EvalCounts {
        positive_count: 0,
        positive_target_beats_old: 0,
        positive_target_is_global_top1: 0,
        protected_count: 0,
        protected_false_activations: 0,
        positive_margin_sum: 0.0,
        protected_safety_margin_sum: 0.0,
    };

    for (target, scenario) in scenarios.iter().enumerate() {
        let positives = if heldout {
            &scenario.positive_heldout
        } else {
            &scenario.positive_train
        };
        for query in positives {
            let old_logits = controller.compute_logits(query);
            let old_best = best(&old_logits);
            let new_logits = candidate_logits(query, candidates);
            let target_logit = new_logits[target];
            let best_new_other = new_logits
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != target)
                .map(|(_, logit)| *logit)
                .fold(f32::NEG_INFINITY, f32::max);
            counts.positive_count += 1;
            if target_logit > old_best {
                counts.positive_target_beats_old += 1;
            }
            if target_logit > old_best && target_logit >= best_new_other {
                counts.positive_target_is_global_top1 += 1;
            }
            counts.positive_margin_sum += f64::from(target_logit - old_best);
        }

        let protected = if heldout {
            &scenario.protected_heldout
        } else {
            &scenario.protected_train
        };
        for query in protected {
            let old_logits = controller.compute_logits(query);
            let old_best = best(&old_logits);
            let new_logits = candidate_logits(query, candidates);
            let new_best = best(&new_logits);
            counts.protected_count += 1;
            if new_best >= old_best {
                counts.protected_false_activations += 1;
            }
            counts.protected_safety_margin_sum += f64::from(old_best - new_best);
        }
    }

    let pos_n = counts.positive_count.max(1) as f64;
    let protected_n = counts.protected_count.max(1) as f64;
    let metrics = Evaluation {
        positive_contexts: counts.positive_count,
        intended_token_beats_old_vocab_fraction: counts.positive_target_beats_old as f64 / pos_n,
        intended_token_global_top1_fraction: counts.positive_target_is_global_top1 as f64 / pos_n,
        protected_contexts: counts.protected_count,
        protected_false_activation_count: counts.protected_false_activations,
        protected_false_activation_rate: counts.protected_false_activations as f64 / protected_n,
        protected_false_activation_upper_95: clopper_pearson_upper(
            counts.protected_false_activations,
            counts.protected_count,
            CONFIDENCE_ALPHA,
        ),
        mean_positive_margin_over_old_vocab: counts.positive_margin_sum / pos_n,
        mean_protected_old_best_margin_over_new_vocab: counts.protected_safety_margin_sum / protected_n,
    };
    (metrics, counts)
}

fn candidate_diversity(candidates: &[ContinuousHV]) -> (f64, f64) {
    let mut cosines = Vec::new();
    for left in 0..candidates.len() {
        for right in (left + 1)..candidates.len() {
            cosines.push(f64::from(candidates[left].similarity(&candidates[right])).abs());
        }
    }
    if cosines.is_empty() {
        return (0.0, 0.0);
    }
    let mean = cosines.iter().sum::<f64>() / cosines.len() as f64;
    let max = cosines.iter().copied().fold(0.0f64, f64::max);
    (mean, max)
}

fn digest_fixture() -> String {
    let fixture = [
        format!("schema=1;genesis={GENESIS_PHRASE};dimension={HDC_DIMENSION};base_vocab={BASE_VOCAB_SIZE}"),
        format!("new_tokens={NEW_TOKEN_COUNT};train={TRAIN_QUERY_COUNT}"),
        format!("heldout={HELDOUT_QUERY_COUNT};noise=0.25;logit_scale={LOGIT_SCALE};cp_alpha={CONFIDENCE_ALPHA}"),
        format!("lambda_grid={LAMBDA_GRID:?}"),
    ]
    .join(";");
    format!("{:x}", Sha256::digest(fixture.as_bytes()))
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

    let genesis = GenesisSeed::from_phrase(GENESIS_PHRASE);
    let controller = make_controller(&genesis);
    let scenarios = build_scenarios(&controller, &genesis);

    let mut trials = Vec::new();
    let mut chosen: Option<(f32, Evaluation, f64)> = None;
    for lambda in LAMBDA_GRID {
        let candidates = strategy_candidates(&scenarios, &genesis, "contrastive_calibrated", lambda);
        let (metrics, _) = evaluate(&controller, &scenarios, &candidates, false);
        trials.push(LambdaTrial {
            lambda,
            train_positive_global_top1_fraction: metrics.intended_token_global_top1_fraction,
            train_protected_false_activation_rate: metrics.protected_false_activation_rate,
            train_protected_false_activation_upper_95: metrics.protected_false_activation_upper_95,
            train_mean_positive_margin: metrics.mean_positive_margin_over_old_vocab,
        });
        let feasible = metrics.protected_false_activation_rate <= MAX_TRAIN_FALSE_ACTIVATION_RATE;
        let candidate_key = (lambda, metrics.clone(), metrics.protected_false_activation_rate);
        let should_replace = match chosen.as_ref() {
            None => true,
            Some((_, best_metrics, best_false_rate)) => {
                let best_feasible =
                    *best_false_rate <= MAX_TRAIN_FALSE_ACTIVATION_RATE;
                if feasible && !best_feasible {
                    true
                } else if feasible == best_feasible && feasible {
                    metrics.intended_token_global_top1_fraction
                        > best_metrics.intended_token_global_top1_fraction
                        || (metrics.intended_token_global_top1_fraction
                            == best_metrics.intended_token_global_top1_fraction
                            && metrics.mean_positive_margin_over_old_vocab
                                > best_metrics.mean_positive_margin_over_old_vocab)
                } else if !feasible && !best_feasible {
                    metrics.protected_false_activation_rate < *best_false_rate
                        || (metrics.protected_false_activation_rate == *best_false_rate
                            && metrics.intended_token_global_top1_fraction
                                > best_metrics.intended_token_global_top1_fraction)
                } else {
                    false
                }
            }
        };
        if should_replace {
            chosen = Some(candidate_key);
        }
    }
    let (selected_lambda, selected_train_metrics, _) =
        chosen.ok_or("lambda grid unexpectedly empty")?;

    let strategies = [
        ("seeded_random", "seeded_random"),
        ("context_bundle", "context_bundle"),
        ("positive_centroid", "positive_centroid"),
        ("context_plus_residual_0_15", "context_plus_residual_0_15"),
        ("contrastive_calibrated", "contrastive_calibrated"),
    ];
    let mut results = Vec::new();
    for (label, candidate_strategy) in strategies {
        let candidates = strategy_candidates(
            &scenarios,
            &genesis,
            candidate_strategy,
            selected_lambda,
        );
        let (heldout, _) = evaluate(&controller, &scenarios, &candidates, true);
        let (mean_abs_pairwise_new_candidate_cosine, max_abs_pairwise_new_candidate_cosine) =
            candidate_diversity(&candidates);
        results.push(StrategyResult {
            strategy: label,
            heldout,
            mean_abs_pairwise_new_candidate_cosine,
            max_abs_pairwise_new_candidate_cosine,
        });
    }

    let receipt = Receipt {
        schema_version: 1,
        benchmark_id: "broca-angular-calibration-synthetic-v1",
        evidence_level: "synthetic_geometry_smoke",
        evaluation_mode: "paired_positive_protected_with_heldout_noise",
        source_revision: option_env!("GIT_COMMIT_SHA"),
        base_vocab_size: BASE_VOCAB_SIZE,
        appended_token_count: NEW_TOKEN_COUNT,
        embedding_dimension: HDC_DIMENSION,
        train_queries_per_class_and_side: TRAIN_QUERY_COUNT,
        heldout_queries_per_class_and_side: HELDOUT_QUERY_COUNT,
        seed_scheme: "GenesisSeed::from_phrase with named deterministic vector domains",
        selected_lambda,
        maximum_allowed_train_false_activation_rate: MAX_TRAIN_FALSE_ACTIVATION_RATE,
        selected_train_false_activation_rate: selected_train_metrics.protected_false_activation_rate,
        selected_train_false_activation_upper_95: selected_train_metrics.protected_false_activation_upper_95,
        calibration_constraint_met: selected_train_metrics.protected_false_activation_rate
            <= MAX_TRAIN_FALSE_ACTIVATION_RATE,
        heldout_false_activation_confidence: "one-sided exact Clopper-Pearson 95% upper bound",
        false_activation_bound_assumption: concat!(
            "IID Bernoulli queries; concepts may be heterogeneous, so this is diagnostic, ",
            "not a distribution-free guarantee"
        ),
        lambda_trials: trials,
        results,
        fixture_spec_digest_sha256: digest_fixture(),
        claim_boundary: "Synthetic planted geometry only; no semantic labels, downstream training, or SOTA claim.",
    };

    let json = serde_json::to_string_pretty(&receipt)?;
    if let Some(path) = json_out {
        std::fs::write(&path, &json)?;
        eprintln!("wrote receipt to {path}");
    }
    println!("{json}");
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_false_activations_with_512_queries_has_sub_one_percent_iid_bound() {
        let upper = clopper_pearson_upper(0, 512, 0.05);
        assert!(upper > 0.005 && upper < 0.006);
        assert!(upper < 0.01);
    }

    #[test]
    fn one_false_activation_with_512_queries_still_has_sub_one_percent_iid_bound() {
        assert!(clopper_pearson_upper(1, 512, 0.05) < 0.01);
    }

    #[test]
    fn two_false_activations_with_512_queries_exceed_one_percent_iid_bound() {
        assert!(clopper_pearson_upper(2, 512, 0.05) > 0.01);
    }

    #[test]
    fn empty_or_all_false_activation_samples_fail_closed() {
        assert_eq!(clopper_pearson_upper(0, 0, 0.05), 1.0);
        assert_eq!(clopper_pearson_upper(10, 10, 0.05), 1.0);
    }
}
