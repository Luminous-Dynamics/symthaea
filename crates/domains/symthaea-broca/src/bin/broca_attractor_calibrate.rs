// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! broca-attractor-calibrate: measure whether HDC thought state ranks target
//! tokens above current Mamba distractors.
//!
//! This is intentionally report-first. It answers the control question we need
//! before deeper coupling: does the token-HDC back-projection make the intended
//! target token more aligned to the thought than the model's top local choices?

use std::collections::BTreeMap;
use std::fs;
use std::process;

use anyhow::{Context, Result};
use serde::Serialize;
use symthaea_broca::checkpoint::ProjectionCheckpoint;
use symthaea_broca::evaluation::{CanonicalEvalCase, CanonicalEvalDataset};
use symthaea_broca::liquid_mamba::{LiquidMambaConfig, LiquidMambaGenerator};
use symthaea_broca::training::TrainingPair;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::math::cosine_similarity_f32;

#[derive(Debug, Clone)]
struct Options {
    checkpoint_path: String,
    canonical_path: String,
    eval_limit: usize,
    json_out: Option<String>,
    dump_cases: Option<String>,
    genesis_phrase: String,
    model_id: String,
    top_k: usize,
    min_new_tokens: usize,
    allow_checkpoint_recovery: bool,
    min_positive_margin_rate: Option<f32>,
    min_avg_alignment_margin: Option<f32>,
}

#[derive(Debug, Serialize)]
struct CalibrationReport {
    checkpoint_path: String,
    canonical_path: String,
    model_id: String,
    top_k: usize,
    min_new_tokens: usize,
    aggregate: CalibrationAggregate,
    gate: CalibrationGateResult,
    categories: BTreeMap<String, CalibrationAggregate>,
    cases: Vec<CalibrationCaseResult>,
}

#[derive(Debug, Clone, Default, Serialize)]
struct CalibrationAggregate {
    num_cases: usize,
    avg_target_alignment: f32,
    avg_best_distractor_alignment: f32,
    avg_alignment_margin: f32,
    positive_margin_rate: f32,
    avg_target_alignment_rank: f32,
    target_logit_top_k_rate: f32,
    avg_target_logit_rank: f32,
    avg_target_probability: f32,
}

#[derive(Debug, Clone, Serialize)]
struct CalibrationGateResult {
    passed: bool,
    min_positive_margin_rate: Option<f32>,
    min_avg_alignment_margin: Option<f32>,
    failures: Vec<String>,
}

#[derive(Debug, Serialize)]
struct CalibrationCaseResult {
    case_index: usize,
    case_id: String,
    category: String,
    tags: Vec<String>,
    target_text: String,
    target_token_id: u32,
    target_token: String,
    target_alignment: f32,
    best_distractor_token_id: Option<u32>,
    best_distractor_token: Option<String>,
    best_distractor_alignment: f32,
    alignment_margin: f32,
    target_alignment_rank: usize,
    target_logit_rank: usize,
    target_probability: f32,
    target_logit_top_k: bool,
    top_logits: Vec<CalibrationCandidate>,
    alignment_candidates: Vec<CalibrationCandidate>,
}

#[derive(Debug, Clone, Serialize)]
struct CalibrationCandidate {
    rank: usize,
    token_id: u32,
    token: String,
    logit: f32,
    probability: f32,
    alignment: f32,
}

#[derive(Debug, Clone)]
struct ScoredToken {
    token_id: u32,
    logit: f32,
    probability: f32,
    alignment: f32,
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".parse().unwrap()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let opts = match parse_args(&args) {
        Ok(opts) => opts,
        Err(e) => {
            eprintln!("Error: {e}");
            print_usage();
            process::exit(1);
        }
    };

    if let Err(e) = run(opts) {
        eprintln!("Error: {e:#}");
        process::exit(1);
    }
}

fn run(opts: Options) -> Result<()> {
    let dataset = CanonicalEvalDataset::from_jsonl(&opts.canonical_path)?;
    let genesis = GenesisSeed::from_phrase(&opts.genesis_phrase);
    let checkpoint = if opts.allow_checkpoint_recovery {
        ProjectionCheckpoint::load_from_file_allow_checksum_mismatch(&opts.checkpoint_path)
    } else {
        ProjectionCheckpoint::load_from_file(&opts.checkpoint_path)
    }
    .with_context(|| format!("loading projection checkpoint {}", opts.checkpoint_path))?;

    let mut config = LiquidMambaConfig {
        model_id: opts.model_id.clone(),
        max_tokens: 1,
        min_new_tokens: opts.min_new_tokens,
        top_k: opts.top_k,
        enable_gating: false,
        enable_consciousness_gating: false,
        enable_veto: false,
        enable_semantic_attractor: false,
        ..Default::default()
    };
    config.temporal_projection = checkpoint.temporal || checkpoint.temporal_weights.is_some();
    config.deep_projection = checkpoint.deep;
    if checkpoint.chunk_dim > 0 {
        config.temporal_chunk_dim = checkpoint.chunk_dim;
    }
    if checkpoint.num_groups > 0 {
        config.temporal_num_groups = checkpoint.num_groups;
    }

    tracing::info!(
        checkpoint = %opts.checkpoint_path,
        top_k = opts.top_k,
        "Loading Liquid-Mamba attractor calibration harness"
    );
    let mut generator = LiquidMambaGenerator::new(&genesis, config)?;
    generator
        .projection_mut()
        .load_weights(&checkpoint.projection_weights);
    if checkpoint.temporal {
        if let (Some(weights), Some(temporal_proj)) = (
            checkpoint.temporal_weights.as_ref(),
            generator.temporal_proj_mut(),
        ) {
            temporal_proj.load_weights(weights);
        }
    }

    let cases = limited_cases(&dataset, opts.eval_limit);
    let mut results = Vec::with_capacity(cases.len());
    let mut dump_lines = String::new();
    for (case_index, case) in cases.iter().enumerate() {
        let result = calibrate_case(
            case_index,
            case,
            &mut generator,
            opts.top_k,
            opts.min_new_tokens,
        )
        .with_context(|| format!("calibrating case {}", case.id))?;
        if opts.dump_cases.is_some() {
            dump_lines.push_str(&serde_json::to_string(&result)?);
            dump_lines.push('\n');
        }
        results.push(result);
    }

    let aggregate = aggregate(&results);
    let gate = evaluate_gate(
        &aggregate,
        opts.min_positive_margin_rate,
        opts.min_avg_alignment_margin,
    );
    let report = CalibrationReport {
        checkpoint_path: opts.checkpoint_path,
        canonical_path: opts.canonical_path,
        model_id: opts.model_id,
        top_k: opts.top_k,
        min_new_tokens: opts.min_new_tokens,
        aggregate,
        gate: gate.clone(),
        categories: category_aggregates(&results),
        cases: results,
    };

    if let Some(path) = opts.dump_cases {
        fs::write(path, dump_lines)?;
    }
    if let Some(path) = opts.json_out {
        fs::write(path, serde_json::to_string_pretty(&report)?)?;
    } else {
        println!("{}", serde_json::to_string_pretty(&report)?);
    }

    if !gate.passed {
        anyhow::bail!(
            "attractor calibration gate failed: {}",
            gate.failures.join("; ")
        );
    }

    Ok(())
}

fn calibrate_case(
    case_index: usize,
    case: &CanonicalEvalCase,
    generator: &mut LiquidMambaGenerator,
    top_k: usize,
    min_new_tokens: usize,
) -> Result<CalibrationCaseResult> {
    let channels = case_channels(case);
    let thought_hv = generator.encoder().encode(&channels);
    let target_ids = generator.mamba.encode(&case.target_text)?;
    let Some(target_token_id) = first_content_token(generator, &target_ids)? else {
        anyhow::bail!("no content token in target text {:?}", case.target_text);
    };

    generator.mamba.reset();
    if generator.config.temporal_projection && generator.temporal_proj.is_some() {
        let sequence = generator
            .temporal_proj
            .as_ref()
            .unwrap()
            .project_to_ssm_sequence(&thought_hv);
        generator.mamba.inject_context_sequence(&sequence)?;
    } else {
        let ctx = generator.projection.project_to_ssm(&thought_hv);
        generator.mamba.inject_initial_context(&ctx)?;
    }

    let prev_token = generator.mamba.eos_token_id();
    let mut logits = generator.mamba.forward_one_token(prev_token)?;
    apply_prefix_guards(generator, &mut logits, 0, min_new_tokens);

    let probabilities = probabilities(&logits);
    let top_ids = top_logit_ids(&logits, top_k);
    let mut candidate_ids = top_ids.clone();
    if !candidate_ids.contains(&target_token_id) {
        candidate_ids.push(target_token_id);
    }

    let mut scored = Vec::with_capacity(candidate_ids.len());
    for token_id in candidate_ids {
        let alignment = token_alignment(generator, &thought_hv.values, token_id)?;
        let logit = logits
            .get(token_id as usize)
            .copied()
            .unwrap_or(f32::NEG_INFINITY);
        let probability = probabilities.get(token_id as usize).copied().unwrap_or(0.0);
        scored.push(ScoredToken {
            token_id,
            logit,
            probability,
            alignment,
        });
    }

    let target = scored
        .iter()
        .find(|candidate| candidate.token_id == target_token_id)
        .expect("target token was inserted into candidates");
    let target_alignment = target.alignment;
    let target_probability = target.probability;
    let target_logit_rank = logit_rank(&logits, target_token_id);
    let target_logit_top_k = top_ids.contains(&target_token_id);

    let best_distractor = scored
        .iter()
        .filter(|candidate| candidate.token_id != target_token_id)
        .max_by(|a, b| a.alignment.total_cmp(&b.alignment));
    let best_distractor_token_id = best_distractor.map(|c| c.token_id);
    let best_distractor_alignment = best_distractor.map(|c| c.alignment).unwrap_or(0.0);
    let best_distractor_token =
        best_distractor_token_id.map(|token_id| decode_token(generator, token_id));
    let alignment_margin = target_alignment - best_distractor_alignment;
    let target_alignment_rank = alignment_rank(&scored, target_token_id);

    let mut by_logit = scored.clone();
    by_logit.sort_by(|a, b| b.logit.total_cmp(&a.logit));
    let top_logits = candidates_for_report(generator, &by_logit, top_k)?;

    let mut by_alignment = scored;
    by_alignment.sort_by(|a, b| b.alignment.total_cmp(&a.alignment));
    let alignment_candidates = candidates_for_report(generator, &by_alignment, top_k)?;

    Ok(CalibrationCaseResult {
        case_index,
        case_id: case.id.clone(),
        category: case.category.clone(),
        tags: case.tags.clone(),
        target_text: case.target_text.clone(),
        target_token_id,
        target_token: decode_token(generator, target_token_id),
        target_alignment,
        best_distractor_token_id,
        best_distractor_token,
        best_distractor_alignment,
        alignment_margin,
        target_alignment_rank,
        target_logit_rank,
        target_probability,
        target_logit_top_k,
        top_logits,
        alignment_candidates,
    })
}

fn limited_cases(dataset: &CanonicalEvalDataset, eval_limit: usize) -> &[CanonicalEvalCase] {
    if eval_limit > 0 && eval_limit < dataset.cases.len() {
        &dataset.cases[..eval_limit]
    } else {
        &dataset.cases
    }
}

fn case_channels(case: &CanonicalEvalCase) -> symthaea_broca::encoder::ThoughtChannels {
    TrainingPair {
        channels: case.channels.clone(),
        target_text: case.target_text.clone(),
        target_ids: vec![],
        valence: 0.0,
        arousal: 0.5,
        ..Default::default()
    }
    .to_thought_channels()
}

fn first_content_token(
    generator: &mut LiquidMambaGenerator,
    token_ids: &[u32],
) -> Result<Option<u32>> {
    let eos = generator.mamba.eos_token_id();
    for &token_id in token_ids {
        if token_id == eos {
            continue;
        }
        let decoded = decode_token(generator, token_id);
        if !decoded.trim().is_empty() {
            return Ok(Some(token_id));
        }
    }
    Ok(None)
}

fn apply_prefix_guards(
    generator: &mut LiquidMambaGenerator,
    logits: &mut [f32],
    position: usize,
    min_new_tokens: usize,
) {
    if position >= min_new_tokens {
        return;
    }
    let eos = generator.mamba.eos_token_id();
    if let Some(logit) = logits.get_mut(eos as usize) {
        *logit = f32::NEG_INFINITY;
    }
    for text in ["\n", "\r", "\t", " "] {
        if let Ok(ids) = generator.mamba.encode(text) {
            if ids.len() == 1 {
                if let Some(logit) = logits.get_mut(ids[0] as usize) {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }
    }
}

fn token_alignment(
    generator: &mut LiquidMambaGenerator,
    thought_values: &[f32],
    token_id: u32,
) -> Result<f32> {
    let token_emb = generator.mamba.embedding_vector(token_id)?;
    let token_hv = if let Some(ref tp) = generator.temporal_proj {
        tp.project_to_hdc(&token_emb)
    } else {
        generator.projection.project_to_hdc(&token_emb)
    };
    Ok(cosine_similarity_f32(thought_values, &token_hv.values))
}

fn top_logit_ids(logits: &[f32], k: usize) -> Vec<u32> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|(_, left), (_, right)| right.total_cmp(left));
    indexed
        .into_iter()
        .take(k)
        .map(|(token_id, _)| token_id as u32)
        .collect()
}

fn probabilities(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .max_by(f32::total_cmp)
        .unwrap_or(0.0);
    let exp_values: Vec<f32> = logits
        .iter()
        .map(|&logit| {
            if logit.is_finite() {
                (logit - max_logit).exp()
            } else {
                0.0
            }
        })
        .collect();
    let sum = exp_values.iter().sum::<f32>();
    if sum <= 1e-20 {
        return vec![0.0; logits.len()];
    }
    exp_values.into_iter().map(|value| value / sum).collect()
}

fn logit_rank(logits: &[f32], target_token_id: u32) -> usize {
    let Some(&target_logit) = logits.get(target_token_id as usize) else {
        return logits.len() + 1;
    };
    1 + logits.iter().filter(|&&logit| logit > target_logit).count()
}

fn alignment_rank(candidates: &[ScoredToken], target_token_id: u32) -> usize {
    let Some(target_alignment) = candidates
        .iter()
        .find(|candidate| candidate.token_id == target_token_id)
        .map(|candidate| candidate.alignment)
    else {
        return candidates.len() + 1;
    };
    1 + candidates
        .iter()
        .filter(|candidate| candidate.alignment > target_alignment)
        .count()
}

fn candidates_for_report(
    generator: &mut LiquidMambaGenerator,
    candidates: &[ScoredToken],
    limit: usize,
) -> Result<Vec<CalibrationCandidate>> {
    let mut out = Vec::with_capacity(limit.min(candidates.len()));
    for (rank, candidate) in candidates.iter().take(limit).enumerate() {
        out.push(CalibrationCandidate {
            rank,
            token_id: candidate.token_id,
            token: decode_token(generator, candidate.token_id),
            logit: candidate.logit,
            probability: candidate.probability,
            alignment: candidate.alignment,
        });
    }
    Ok(out)
}

fn decode_token(generator: &mut LiquidMambaGenerator, token_id: u32) -> String {
    generator
        .mamba
        .decode_token(token_id)
        .unwrap_or_else(|_| format!("<token:{token_id}>"))
}

fn aggregate(results: &[CalibrationCaseResult]) -> CalibrationAggregate {
    let n = results.len();
    if n == 0 {
        return CalibrationAggregate::default();
    }
    let nf = n as f32;
    CalibrationAggregate {
        num_cases: n,
        avg_target_alignment: results.iter().map(|r| r.target_alignment).sum::<f32>() / nf,
        avg_best_distractor_alignment: results
            .iter()
            .map(|r| r.best_distractor_alignment)
            .sum::<f32>()
            / nf,
        avg_alignment_margin: results.iter().map(|r| r.alignment_margin).sum::<f32>() / nf,
        positive_margin_rate: results.iter().filter(|r| r.alignment_margin > 0.0).count() as f32
            / nf,
        avg_target_alignment_rank: results
            .iter()
            .map(|r| r.target_alignment_rank as f32)
            .sum::<f32>()
            / nf,
        target_logit_top_k_rate: results.iter().filter(|r| r.target_logit_top_k).count() as f32
            / nf,
        avg_target_logit_rank: results
            .iter()
            .map(|r| r.target_logit_rank as f32)
            .sum::<f32>()
            / nf,
        avg_target_probability: results.iter().map(|r| r.target_probability).sum::<f32>() / nf,
    }
}

fn category_aggregates(
    results: &[CalibrationCaseResult],
) -> BTreeMap<String, CalibrationAggregate> {
    let mut grouped: BTreeMap<String, Vec<CalibrationCaseResult>> = BTreeMap::new();
    for result in results {
        grouped
            .entry(result.category.clone())
            .or_default()
            .push(CalibrationCaseResult {
                case_index: result.case_index,
                case_id: result.case_id.clone(),
                category: result.category.clone(),
                tags: result.tags.clone(),
                target_text: result.target_text.clone(),
                target_token_id: result.target_token_id,
                target_token: result.target_token.clone(),
                target_alignment: result.target_alignment,
                best_distractor_token_id: result.best_distractor_token_id,
                best_distractor_token: result.best_distractor_token.clone(),
                best_distractor_alignment: result.best_distractor_alignment,
                alignment_margin: result.alignment_margin,
                target_alignment_rank: result.target_alignment_rank,
                target_logit_rank: result.target_logit_rank,
                target_probability: result.target_probability,
                target_logit_top_k: result.target_logit_top_k,
                top_logits: result.top_logits.clone(),
                alignment_candidates: result.alignment_candidates.clone(),
            });
    }
    grouped
        .into_iter()
        .map(|(category, cases)| (category, aggregate(&cases)))
        .collect()
}

fn evaluate_gate(
    aggregate: &CalibrationAggregate,
    min_positive_margin_rate: Option<f32>,
    min_avg_alignment_margin: Option<f32>,
) -> CalibrationGateResult {
    let mut failures = Vec::new();
    if let Some(threshold) = min_positive_margin_rate {
        if aggregate.positive_margin_rate < threshold {
            failures.push(format!(
                "positive_margin_rate {:.4} < {:.4}",
                aggregate.positive_margin_rate, threshold
            ));
        }
    }
    if let Some(threshold) = min_avg_alignment_margin {
        if aggregate.avg_alignment_margin < threshold {
            failures.push(format!(
                "avg_alignment_margin {:.6} < {:.6}",
                aggregate.avg_alignment_margin, threshold
            ));
        }
    }
    CalibrationGateResult {
        passed: failures.is_empty(),
        min_positive_margin_rate,
        min_avg_alignment_margin,
        failures,
    }
}

fn parse_args(args: &[String]) -> Result<Options> {
    let mut opts = Options {
        checkpoint_path: String::new(),
        canonical_path: "crates/symthaea-broca/tests/fixtures/eval-canonical-v1.jsonl".into(),
        eval_limit: 0,
        json_out: None,
        dump_cases: None,
        genesis_phrase: "symthaea luminous dynamics".into(),
        model_id: "state-spaces/mamba-130m".into(),
        top_k: env_usize("BROCA_ATTRACTOR_CAL_TOP_K", 32),
        min_new_tokens: env_usize("BROCA_ATTRACTOR_CAL_MIN_NEW_TOKENS", 1),
        allow_checkpoint_recovery: false,
        min_positive_margin_rate: env_optional_f32("BROCA_ATTRACTOR_CAL_MIN_POSITIVE_MARGIN_RATE"),
        min_avg_alignment_margin: env_optional_f32("BROCA_ATTRACTOR_CAL_MIN_AVG_ALIGNMENT_MARGIN"),
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--checkpoint" => {
                i += 1;
                opts.checkpoint_path = value(args, i, "--checkpoint")?.to_string();
            }
            "--canonical-eval" => {
                i += 1;
                opts.canonical_path = value(args, i, "--canonical-eval")?.to_string();
            }
            "--eval-limit" => {
                i += 1;
                opts.eval_limit = value(args, i, "--eval-limit")?.parse()?;
            }
            "--json-out" => {
                i += 1;
                opts.json_out = Some(value(args, i, "--json-out")?.to_string());
            }
            "--dump-cases" => {
                i += 1;
                opts.dump_cases = Some(value(args, i, "--dump-cases")?.to_string());
            }
            "--genesis" => {
                i += 1;
                opts.genesis_phrase = value(args, i, "--genesis")?.to_string();
            }
            "--model" => {
                i += 1;
                opts.model_id = value(args, i, "--model")?.to_string();
            }
            "--top-k" => {
                i += 1;
                opts.top_k = value(args, i, "--top-k")?.parse()?;
            }
            "--min-new-tokens" => {
                i += 1;
                opts.min_new_tokens = value(args, i, "--min-new-tokens")?.parse()?;
            }
            "--allow-checkpoint-recovery" => {
                opts.allow_checkpoint_recovery = true;
            }
            "--min-positive-margin-rate" => {
                i += 1;
                opts.min_positive_margin_rate =
                    Some(value(args, i, "--min-positive-margin-rate")?.parse()?);
            }
            "--min-avg-alignment-margin" => {
                i += 1;
                opts.min_avg_alignment_margin =
                    Some(value(args, i, "--min-avg-alignment-margin")?.parse()?);
            }
            "--help" | "-h" => {
                print_usage();
                process::exit(0);
            }
            other => anyhow::bail!("unknown argument {other}"),
        }
        i += 1;
    }

    if opts.checkpoint_path.is_empty() {
        anyhow::bail!("--checkpoint is required");
    }
    if opts.top_k == 0 {
        anyhow::bail!("--top-k must be greater than zero");
    }
    Ok(opts)
}

fn value<'a>(args: &'a [String], index: usize, flag: &str) -> Result<&'a str> {
    args.get(index)
        .map(String::as_str)
        .with_context(|| format!("{flag} requires a value"))
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_optional_f32(name: &str) -> Option<f32> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .and_then(|value| value.parse().ok())
}

fn print_usage() {
    eprintln!(
        "Usage: broca-attractor-calibrate --checkpoint PATH [--canonical-eval PATH] [--eval-limit N] [--top-k K] [--json-out PATH] [--dump-cases PATH] [--min-positive-margin-rate F] [--min-avg-alignment-margin F] [--allow-checkpoint-recovery]"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alignment_rank_places_target_above_weaker_candidates() {
        let candidates = vec![
            ScoredToken {
                token_id: 10,
                logit: 1.0,
                probability: 0.4,
                alignment: 0.1,
            },
            ScoredToken {
                token_id: 20,
                logit: 0.9,
                probability: 0.3,
                alignment: 0.5,
            },
            ScoredToken {
                token_id: 30,
                logit: 0.8,
                probability: 0.2,
                alignment: 0.2,
            },
        ];
        assert_eq!(alignment_rank(&candidates, 20), 1);
        assert_eq!(alignment_rank(&candidates, 30), 2);
    }

    #[test]
    fn logit_rank_uses_one_based_full_vocab_rank() {
        let logits = vec![0.2, 1.0, -0.5, 0.7];
        assert_eq!(logit_rank(&logits, 1), 1);
        assert_eq!(logit_rank(&logits, 3), 2);
        assert_eq!(logit_rank(&logits, 2), 4);
    }

    #[test]
    fn probabilities_ignore_masked_logits() {
        let probs = probabilities(&[0.0, f32::NEG_INFINITY, 0.0]);
        assert!((probs[0] - 0.5).abs() < 1e-6);
        assert_eq!(probs[1], 0.0);
        assert!((probs[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn calibration_gate_reports_failed_thresholds() {
        let aggregate = CalibrationAggregate {
            num_cases: 2,
            avg_alignment_margin: -0.01,
            positive_margin_rate: 0.25,
            ..Default::default()
        };
        let gate = evaluate_gate(&aggregate, Some(0.5), Some(0.0));
        assert!(!gate.passed);
        assert_eq!(gate.failures.len(), 2);
    }
}
