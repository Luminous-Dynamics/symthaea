// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! broca-liquid-eval: direct Liquid-Mamba canonical generation harness.
//!
//! This intentionally bypasses the full BrocaGenerator wrapper. It is used to
//! verify the projection bridge, CfC modulation, and semantic attractor in the
//! actual LiquidMambaGenerator::generate path.

use std::fs;
use std::process;

use anyhow::{Context, Result};
use serde::Serialize;
use symthaea_broca::checkpoint::ProjectionCheckpoint;
use symthaea_broca::evaluation::{CanonicalEvalCase, CanonicalEvalDataset};
use symthaea_broca::generator::{GenerationResult, GenerationStepLogits, GenerationTopLogit};
use symthaea_broca::liquid_mamba::{LiquidMambaConfig, LiquidMambaGenerator};
use symthaea_broca::training::TrainingPair;
use symthaea_core::genesis::GenesisSeed;

#[derive(Debug, Clone)]
struct Options {
    checkpoint_path: String,
    canonical_path: String,
    eval_limit: usize,
    max_gen_tokens: usize,
    min_new_tokens: usize,
    json_out: Option<String>,
    dump_generations: Option<String>,
    genesis_phrase: String,
    model_id: String,
    temperature: f32,
    top_k: usize,
    sampling_seed: Option<u64>,
    allow_checkpoint_recovery: bool,
    semantic_attractor: bool,
    semantic_attractor_strength: f32,
    semantic_attractor_top_k: usize,
    semantic_attractor_max_adjustment: f32,
    semantic_attractor_normalize: bool,
}

#[derive(Debug, Serialize)]
struct LiquidEvalReport {
    checkpoint_path: String,
    canonical_path: String,
    model_id: String,
    semantic_attractor: bool,
    semantic_attractor_strength: f32,
    semantic_attractor_top_k: usize,
    semantic_attractor_max_adjustment: f32,
    semantic_attractor_normalize: bool,
    temperature: f32,
    top_k: usize,
    sampling_seed: Option<u64>,
    min_new_tokens: usize,
    aggregate: LiquidEvalAggregate,
    cases: Vec<LiquidEvalCaseResult>,
}

#[derive(Debug, Default, Serialize)]
struct LiquidEvalAggregate {
    num_cases: usize,
    avg_coherence: f32,
    avg_semantic_pe: f32,
    hallucination_rate: f32,
    avg_entropy: f32,
    avg_max_probability: f32,
    avg_pre_attractor_entropy: Option<f32>,
    avg_pre_attractor_max_probability: Option<f32>,
    avg_cfc_delta_scale: Option<f32>,
    avg_cfc_b_scale: Option<f32>,
    avg_semantic_attractor_mean_adjustment: Option<f32>,
    avg_semantic_attractor_max_adjustment: Option<f32>,
    avg_selected_semantic_alignment: Option<f32>,
}

#[derive(Debug, Serialize)]
struct LiquidEvalCaseResult {
    case_index: usize,
    case_id: String,
    category: String,
    tags: Vec<String>,
    intent: String,
    target_text: String,
    generated_text: String,
    token_ids: Vec<u32>,
    decoded_tokens: Vec<String>,
    num_tokens: usize,
    eos_terminated: bool,
    hallucination_flag: bool,
    final_coherence: f32,
    semantic_pe: f32,
    logit_diagnostics: Vec<LiquidStepDiagnostic>,
}

#[derive(Debug, Serialize)]
struct LiquidStepDiagnostic {
    position: usize,
    selected_token_id: u32,
    selected_token: String,
    entropy: f32,
    max_probability: f32,
    pre_attractor_entropy: Option<f32>,
    pre_attractor_max_probability: Option<f32>,
    cfc_delta_scale: Option<f32>,
    cfc_b_scale: Option<f32>,
    semantic_attractor_mean_adjustment: Option<f32>,
    semantic_attractor_max_adjustment: Option<f32>,
    semantic_attractor_alignment_mean: Option<f32>,
    semantic_attractor_alignment_std: Option<f32>,
    selected_semantic_alignment: Option<f32>,
    top_k: Vec<LiquidTopLogit>,
}

#[derive(Debug, Serialize)]
struct LiquidTopLogit {
    rank: usize,
    token_id: u32,
    token: String,
    logit: f32,
    probability: f32,
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
    let mut config = LiquidMambaConfig {
        model_id: opts.model_id.clone(),
        max_tokens: opts.max_gen_tokens,
        min_new_tokens: opts.min_new_tokens,
        temperature: opts.temperature,
        top_k: opts.top_k,
        sampling_seed: opts.sampling_seed,
        enable_semantic_attractor: opts.semantic_attractor,
        semantic_attractor_strength: opts.semantic_attractor_strength,
        semantic_attractor_top_k: opts.semantic_attractor_top_k,
        semantic_attractor_max_adjustment: opts.semantic_attractor_max_adjustment,
        semantic_attractor_normalize: opts.semantic_attractor_normalize,
        enable_consciousness_gating: false,
        enable_veto: false,
        ..Default::default()
    };

    let checkpoint = if opts.allow_checkpoint_recovery {
        ProjectionCheckpoint::load_from_file_allow_checksum_mismatch(&opts.checkpoint_path)
    } else {
        ProjectionCheckpoint::load_from_file(&opts.checkpoint_path)
    }
    .with_context(|| format!("loading projection checkpoint {}", opts.checkpoint_path))?;

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
        semantic_attractor = opts.semantic_attractor,
        "Loading direct Liquid-Mamba evaluator"
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
        let channels = case_channels(case);
        let result = generator.generate(&channels);
        let case_result = case_result(case_index, case, &result, &mut generator);
        if opts.dump_generations.is_some() {
            dump_lines.push_str(&serde_json::to_string(&case_result)?);
            dump_lines.push('\n');
        }
        results.push(case_result);
    }

    let report = LiquidEvalReport {
        checkpoint_path: opts.checkpoint_path,
        canonical_path: opts.canonical_path,
        model_id: opts.model_id,
        semantic_attractor: opts.semantic_attractor,
        semantic_attractor_strength: opts.semantic_attractor_strength,
        semantic_attractor_top_k: opts.semantic_attractor_top_k,
        semantic_attractor_max_adjustment: opts.semantic_attractor_max_adjustment,
        semantic_attractor_normalize: opts.semantic_attractor_normalize,
        temperature: opts.temperature,
        top_k: opts.top_k,
        sampling_seed: opts.sampling_seed,
        min_new_tokens: opts.min_new_tokens,
        aggregate: aggregate(&results),
        cases: results,
    };

    if let Some(path) = opts.dump_generations {
        fs::write(path, dump_lines)?;
    }
    if let Some(path) = opts.json_out {
        fs::write(path, serde_json::to_string_pretty(&report)?)?;
    } else {
        println!("{}", serde_json::to_string_pretty(&report)?);
    }

    Ok(())
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

fn case_result(
    case_index: usize,
    case: &CanonicalEvalCase,
    result: &GenerationResult,
    generator: &mut LiquidMambaGenerator,
) -> LiquidEvalCaseResult {
    let decoded_tokens = result
        .token_ids
        .iter()
        .map(|&token_id| decode_token(generator, token_id))
        .collect();
    let logit_diagnostics = result
        .logit_diagnostics
        .iter()
        .map(|step| step_diagnostic(step, generator))
        .collect();

    LiquidEvalCaseResult {
        case_index,
        case_id: case.id.clone(),
        category: case.category.clone(),
        tags: case.tags.clone(),
        intent: active_intent_name(&case.channels).to_string(),
        target_text: case.target_text.clone(),
        generated_text: result.text.clone(),
        token_ids: result.token_ids.clone(),
        decoded_tokens,
        num_tokens: result.num_tokens,
        eos_terminated: result.eos_terminated,
        hallucination_flag: result.hallucination_flag,
        final_coherence: result.final_coherence,
        semantic_pe: result.semantic_pe,
        logit_diagnostics,
    }
}

fn step_diagnostic(
    step: &GenerationStepLogits,
    generator: &mut LiquidMambaGenerator,
) -> LiquidStepDiagnostic {
    LiquidStepDiagnostic {
        position: step.position,
        selected_token_id: step.selected_token_id,
        selected_token: decode_token(generator, step.selected_token_id),
        entropy: step.entropy,
        max_probability: step.max_probability,
        pre_attractor_entropy: step.pre_attractor_entropy,
        pre_attractor_max_probability: step.pre_attractor_max_probability,
        cfc_delta_scale: step.cfc_delta_scale,
        cfc_b_scale: step.cfc_b_scale,
        semantic_attractor_mean_adjustment: step.semantic_attractor_mean_adjustment,
        semantic_attractor_max_adjustment: step.semantic_attractor_max_adjustment,
        semantic_attractor_alignment_mean: step.semantic_attractor_alignment_mean,
        semantic_attractor_alignment_std: step.semantic_attractor_alignment_std,
        selected_semantic_alignment: step.selected_semantic_alignment,
        top_k: step
            .top_k
            .iter()
            .map(|top| top_logit(top, generator))
            .collect(),
    }
}

fn top_logit(top: &GenerationTopLogit, generator: &mut LiquidMambaGenerator) -> LiquidTopLogit {
    LiquidTopLogit {
        rank: top.rank,
        token_id: top.token_id,
        token: decode_token(generator, top.token_id),
        logit: top.logit,
        probability: top.probability,
    }
}

fn decode_token(generator: &mut LiquidMambaGenerator, token_id: u32) -> String {
    generator
        .mamba_mut()
        .decode_token(token_id)
        .unwrap_or_else(|_| format!("<token:{token_id}>"))
}

fn aggregate(results: &[LiquidEvalCaseResult]) -> LiquidEvalAggregate {
    let n = results.len().max(1) as f32;
    let mut aggregate = LiquidEvalAggregate {
        num_cases: results.len(),
        avg_coherence: results.iter().map(|r| r.final_coherence).sum::<f32>() / n,
        avg_semantic_pe: results.iter().map(|r| r.semantic_pe).sum::<f32>() / n,
        hallucination_rate: results.iter().filter(|r| r.hallucination_flag).count() as f32 / n,
        avg_entropy: avg_required_step(results, |s| s.entropy),
        avg_max_probability: avg_required_step(results, |s| s.max_probability),
        avg_pre_attractor_entropy: avg_optional_step(results, |s| s.pre_attractor_entropy),
        avg_pre_attractor_max_probability: avg_optional_step(results, |s| {
            s.pre_attractor_max_probability
        }),
        avg_cfc_delta_scale: avg_optional_step(results, |s| s.cfc_delta_scale),
        avg_cfc_b_scale: avg_optional_step(results, |s| s.cfc_b_scale),
        avg_semantic_attractor_mean_adjustment: avg_optional_step(results, |s| {
            s.semantic_attractor_mean_adjustment
        }),
        avg_semantic_attractor_max_adjustment: avg_optional_step(results, |s| {
            s.semantic_attractor_max_adjustment
        }),
        avg_selected_semantic_alignment: avg_optional_step(results, |s| {
            s.selected_semantic_alignment
        }),
    };
    if results.is_empty() {
        aggregate.avg_entropy = 0.0;
        aggregate.avg_max_probability = 0.0;
    }
    aggregate
}

fn avg_required_step<F>(results: &[LiquidEvalCaseResult], mut f: F) -> f32
where
    F: FnMut(&LiquidStepDiagnostic) -> f32,
{
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for step in results.iter().flat_map(|r| r.logit_diagnostics.iter()) {
        sum += f(step);
        count += 1;
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

fn avg_optional_step<F>(results: &[LiquidEvalCaseResult], mut f: F) -> Option<f32>
where
    F: FnMut(&LiquidStepDiagnostic) -> Option<f32>,
{
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for value in results
        .iter()
        .flat_map(|r| r.logit_diagnostics.iter())
        .filter_map(|s| f(s))
    {
        sum += value;
        count += 1;
    }
    (count > 0).then_some(sum / count as f32)
}

fn active_intent_name(channels: &[f32]) -> &'static str {
    let names = [
        "analyze", "create", "explain", "question", "answer", "reflect", "relate", "unknown",
    ];
    let idx = channels
        .iter()
        .take(names.len())
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(7);
    names[idx]
}

fn parse_args(args: &[String]) -> Result<Options> {
    let mut opts = Options {
        checkpoint_path: String::new(),
        canonical_path: "crates/symthaea-broca/tests/fixtures/eval-canonical-v1.jsonl".into(),
        eval_limit: 0,
        max_gen_tokens: 8,
        min_new_tokens: env_usize("BROCA_LIQUID_EVAL_MIN_NEW_TOKENS", 1),
        json_out: None,
        dump_generations: None,
        genesis_phrase: "symthaea luminous dynamics".into(),
        model_id: "state-spaces/mamba-130m".into(),
        temperature: env_f32("BROCA_LIQUID_EVAL_TEMPERATURE", 0.8),
        top_k: env_usize("BROCA_LIQUID_EVAL_TOP_K", 40),
        sampling_seed: env_u64("BROCA_LIQUID_EVAL_SAMPLING_SEED"),
        allow_checkpoint_recovery: false,
        semantic_attractor: env_bool("BROCA_LIQUID_EVAL_SEMANTIC_ATTRACTOR", false),
        semantic_attractor_strength: env_f32("BROCA_LIQUID_EVAL_SEMANTIC_ATTRACTOR_STRENGTH", 0.5),
        semantic_attractor_top_k: env_usize("BROCA_LIQUID_EVAL_SEMANTIC_ATTRACTOR_TOP_K", 128),
        semantic_attractor_max_adjustment: env_f32(
            "BROCA_LIQUID_EVAL_SEMANTIC_ATTRACTOR_MAX_ADJUSTMENT",
            1.5,
        ),
        semantic_attractor_normalize: env_bool(
            "BROCA_LIQUID_EVAL_SEMANTIC_ATTRACTOR_NORMALIZE",
            true,
        ),
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
            "--max-gen-tokens" | "--max-r#gen-tokens" => {
                i += 1;
                opts.max_gen_tokens = value(args, i, "--max-gen-tokens")?.parse()?;
            }
            "--min-new-tokens" => {
                i += 1;
                opts.min_new_tokens = value(args, i, "--min-new-tokens")?.parse()?;
            }
            "--json-out" => {
                i += 1;
                opts.json_out = Some(value(args, i, "--json-out")?.to_string());
            }
            "--dump-generations" => {
                i += 1;
                opts.dump_generations = Some(value(args, i, "--dump-generations")?.to_string());
            }
            "--genesis" => {
                i += 1;
                opts.genesis_phrase = value(args, i, "--genesis")?.to_string();
            }
            "--model" => {
                i += 1;
                opts.model_id = value(args, i, "--model")?.to_string();
            }
            "--temperature" => {
                i += 1;
                opts.temperature = value(args, i, "--temperature")?.parse()?;
            }
            "--top-k" => {
                i += 1;
                opts.top_k = value(args, i, "--top-k")?.parse()?;
            }
            "--sampling-seed" => {
                i += 1;
                opts.sampling_seed = Some(value(args, i, "--sampling-seed")?.parse()?);
            }
            "--allow-checkpoint-recovery" => {
                opts.allow_checkpoint_recovery = true;
            }
            "--semantic-attractor" => {
                opts.semantic_attractor = true;
            }
            "--no-semantic-attractor" => {
                opts.semantic_attractor = false;
            }
            "--semantic-attractor-strength" => {
                i += 1;
                opts.semantic_attractor_strength =
                    value(args, i, "--semantic-attractor-strength")?.parse()?;
            }
            "--semantic-attractor-top-k" => {
                i += 1;
                opts.semantic_attractor_top_k =
                    value(args, i, "--semantic-attractor-top-k")?.parse()?;
            }
            "--semantic-attractor-max-adjustment" => {
                i += 1;
                opts.semantic_attractor_max_adjustment =
                    value(args, i, "--semantic-attractor-max-adjustment")?.parse()?;
            }
            "--semantic-attractor-normalize" => {
                opts.semantic_attractor_normalize = true;
            }
            "--no-semantic-attractor-normalize" => {
                opts.semantic_attractor_normalize = false;
            }
            "-h" | "--help" => {
                print_usage();
                process::exit(0);
            }
            other => anyhow::bail!("unknown argument: {other}"),
        }
        i += 1;
    }

    if opts.checkpoint_path.is_empty() {
        anyhow::bail!("--checkpoint is required");
    }
    Ok(opts)
}

fn value<'a>(args: &'a [String], index: usize, name: &str) -> Result<&'a str> {
    args.get(index)
        .map(String::as_str)
        .with_context(|| format!("{name} requires a value"))
}

fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .and_then(|v| match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

fn print_usage() {
    eprintln!(
        "Usage: broca-liquid-eval --checkpoint projection.bin [--canonical-eval eval.jsonl] \\
         [--eval-limit N] [--max-gen-tokens N] [--min-new-tokens N] [--top-k N] [--json-out report.json] \\
         [--dump-generations dump.jsonl] [--allow-checkpoint-recovery] \\
         [--semantic-attractor|--no-semantic-attractor]"
    );
}
