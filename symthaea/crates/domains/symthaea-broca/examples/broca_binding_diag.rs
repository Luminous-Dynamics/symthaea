// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Compare Broca teacher-forced logits against free-rollout logits.
//!
//! This is intentionally an example target so it can be used during model
//! debugging without expanding the production CLI surface.

use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::Serialize;
use symthaea_broca::checkpoint::BrocaCheckpoint;
use symthaea_broca::encoder::ThoughtChannels;
use symthaea_broca::evaluation::{CanonicalEvalCase, CanonicalEvalDataset};
use symthaea_broca::generator::BrocaGenerator;
use symthaea_broca::training::TrainingPair;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

const TOP_K: usize = 8;
const DEFAULT_GENESIS_PHRASE: &str = "broca-training-default";

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn".parse().unwrap()),
        )
        .init();

    let opts = Opts::parse()?;
    let genesis = GenesisSeed::from_phrase(&opts.genesis_phrase);
    let (mut generator, _adam, _proj, _lm) = if opts.allow_checksum_mismatch {
        BrocaGenerator::from_checkpoint_allow_checksum_mismatch(&opts.checkpoint, &genesis)
    } else {
        BrocaGenerator::from_checkpoint(&opts.checkpoint, &genesis)
    }
    .with_context(|| format!("loading checkpoint {}", opts.checkpoint.display()))?;
    if opts.thought_logit_residual_weight > 0.0 {
        generator
            .controller_mut()
            .config_mut()
            .thought_logit_residual_weight = opts.thought_logit_residual_weight.clamp(0.0, 1.0);
    }

    let dataset = CanonicalEvalDataset::from_jsonl(
        opts.canonical_eval
            .to_str()
            .context("canonical eval path is not valid UTF-8")?,
    )?;
    let cases: Vec<_> = dataset
        .cases
        .iter()
        .take(if opts.eval_limit == 0 {
            usize::MAX
        } else {
            opts.eval_limit
        })
        .cloned()
        .collect();

    let target_token_ids = collect_target_token_ids(&generator, &cases);
    let checkpoint_delta = match &opts.baseline_checkpoint {
        Some(path) => {
            let baseline = load_checkpoint(path, opts.allow_checksum_mismatch)
                .with_context(|| format!("loading baseline checkpoint {}", path.display()))?;
            let candidate = load_checkpoint(&opts.checkpoint, opts.allow_checksum_mismatch)
                .with_context(|| {
                    format!("loading candidate checkpoint {}", opts.checkpoint.display())
                })?;
            Some(checkpoint_delta(&baseline, &candidate, &target_token_ids))
        }
        None => None,
    };

    let mut case_reports = Vec::with_capacity(cases.len());
    for case in &cases {
        case_reports.push(diagnose_case(&mut generator, case, opts.max_positions));
    }

    let report = BindingDiagReport {
        schema_version: 1,
        checkpoint: opts.checkpoint.display().to_string(),
        baseline_checkpoint: opts
            .baseline_checkpoint
            .as_ref()
            .map(|p| p.display().to_string()),
        genesis_phrase: opts.genesis_phrase,
        thought_logit_residual_weight: opts.thought_logit_residual_weight.clamp(0.0, 1.0),
        eval_limit: opts.eval_limit,
        max_positions: opts.max_positions,
        checkpoint_delta,
        cases: case_reports,
    };

    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = opts.json_out {
        std::fs::write(&path, json).with_context(|| format!("writing {}", path.display()))?;
    } else {
        println!("{json}");
    }

    Ok(())
}

#[derive(Debug)]
struct Opts {
    checkpoint: PathBuf,
    baseline_checkpoint: Option<PathBuf>,
    canonical_eval: PathBuf,
    json_out: Option<PathBuf>,
    eval_limit: usize,
    max_positions: usize,
    genesis_phrase: String,
    allow_checksum_mismatch: bool,
    thought_logit_residual_weight: f32,
}

impl Opts {
    fn parse() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let mut opts = Self {
            checkpoint: PathBuf::new(),
            baseline_checkpoint: None,
            canonical_eval: PathBuf::new(),
            json_out: None,
            eval_limit: 2,
            max_positions: 4,
            genesis_phrase: DEFAULT_GENESIS_PHRASE.to_string(),
            allow_checksum_mismatch: false,
            thought_logit_residual_weight: 0.0,
        };

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--checkpoint" => opts.checkpoint = next_path(&mut args, "--checkpoint")?,
                "--baseline-checkpoint" => {
                    opts.baseline_checkpoint = Some(next_path(&mut args, "--baseline-checkpoint")?)
                }
                "--canonical-eval" => {
                    opts.canonical_eval = next_path(&mut args, "--canonical-eval")?
                }
                "--json-out" => opts.json_out = Some(next_path(&mut args, "--json-out")?),
                "--eval-limit" => opts.eval_limit = next_usize(&mut args, "--eval-limit")?,
                "--max-positions" => opts.max_positions = next_usize(&mut args, "--max-positions")?,
                "--genesis" => opts.genesis_phrase = next_string(&mut args, "--genesis")?,
                "--allow-checksum-mismatch" => opts.allow_checksum_mismatch = true,
                "--thought-logit-residual" => {
                    opts.thought_logit_residual_weight =
                        next_f32(&mut args, "--thought-logit-residual")?
                }
                "-h" | "--help" => {
                    print_usage();
                    std::process::exit(0);
                }
                _ => anyhow::bail!("unknown argument {arg}"),
            }
        }

        if opts.checkpoint.as_os_str().is_empty() {
            anyhow::bail!("--checkpoint is required");
        }
        if opts.canonical_eval.as_os_str().is_empty() {
            anyhow::bail!("--canonical-eval is required");
        }
        if opts.max_positions == 0 {
            anyhow::bail!("--max-positions must be > 0");
        }
        Ok(opts)
    }
}

fn print_usage() {
    eprintln!(
        "Usage: cargo run -p symthaea-broca --example broca_binding_diag -- \\
         --checkpoint PATH --canonical-eval PATH [--baseline-checkpoint PATH] \\
         [--json-out PATH] [--eval-limit N] [--max-positions N] \\
         [--genesis PHRASE] [--thought-logit-residual F]\n\
         Default --genesis: {DEFAULT_GENESIS_PHRASE}"
    );
}

fn next_string(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    args.next()
        .with_context(|| format!("{flag} requires a value"))
}

fn next_path(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<PathBuf> {
    Ok(PathBuf::from(next_string(args, flag)?))
}

fn next_usize(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<usize> {
    let value = next_string(args, flag)?;
    value
        .parse()
        .with_context(|| format!("{flag} value {value:?} is not a usize"))
}

fn next_f32(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<f32> {
    let value = next_string(args, flag)?;
    value
        .parse()
        .with_context(|| format!("{flag} value {value:?} is not an f32"))
}

#[derive(Debug, Serialize)]
struct BindingDiagReport {
    schema_version: u32,
    checkpoint: String,
    baseline_checkpoint: Option<String>,
    genesis_phrase: String,
    thought_logit_residual_weight: f32,
    eval_limit: usize,
    max_positions: usize,
    checkpoint_delta: Option<CheckpointDelta>,
    cases: Vec<CaseDiag>,
}

#[derive(Debug, Serialize)]
struct CaseDiag {
    id: String,
    category: String,
    target_text: String,
    target_token_count: usize,
    teacher_forced: Vec<TokenStepDiag>,
    free_rollout: Vec<TokenStepDiag>,
}

#[derive(Debug, Serialize)]
struct TokenStepDiag {
    pos: usize,
    prev_token_id: u32,
    prev_token: String,
    target_token_id: Option<u32>,
    target_token: Option<String>,
    target_rank: Option<usize>,
    target_probability: Option<f32>,
    target_logit: Option<f32>,
    selected_token_id: u32,
    selected_token: String,
    entropy: f32,
    max_probability: f32,
    logit_mean: f32,
    logit_stddev: f32,
    top_k: Vec<TopToken>,
}

#[derive(Debug, Serialize)]
struct TopToken {
    rank: usize,
    token_id: u32,
    token: String,
    logit: f32,
    probability: f32,
}

#[derive(Debug, Serialize)]
struct CheckpointDelta {
    baseline_training_epoch: usize,
    candidate_training_epoch: usize,
    baseline_training_loss: f32,
    candidate_training_loss: f32,
    embedding_count_compared: usize,
    embedding_dim_compared: usize,
    mean_embedding_l2_delta: f32,
    max_embedding_l2_delta: f32,
    mean_embedding_cosine: f32,
    target_token_embedding_deltas: Vec<TokenEmbeddingDelta>,
    network_state_serialized_len_baseline: Option<usize>,
    network_state_serialized_len_candidate: Option<usize>,
    network_state_hash_baseline: Option<String>,
    network_state_hash_candidate: Option<String>,
    network_state_hash_equal: Option<bool>,
}

#[derive(Debug, Serialize)]
struct TokenEmbeddingDelta {
    token_id: u32,
    token: String,
    l2_delta: f32,
    cosine: f32,
}

fn diagnose_case(
    generator: &mut BrocaGenerator,
    case: &CanonicalEvalCase,
    max_positions: usize,
) -> CaseDiag {
    let channels = case_channels(case);
    let thought_hv = generator.encoder().encode(&channels);
    let target_ids = generator.tokenizer().encode(&case.target_text);
    let steps = max_positions.min(target_ids.len().max(1));

    let teacher_forced = rollout_steps(generator, &thought_hv, &target_ids, steps, true);
    let free_rollout = rollout_steps(generator, &thought_hv, &target_ids, steps, false);

    CaseDiag {
        id: case.id.clone(),
        category: case.category.clone(),
        target_text: case.target_text.clone(),
        target_token_count: target_ids.len(),
        teacher_forced,
        free_rollout,
    }
}

fn rollout_steps(
    generator: &mut BrocaGenerator,
    thought_hv: &ContinuousHV,
    target_ids: &[u32],
    steps: usize,
    teacher_forced: bool,
) -> Vec<TokenStepDiag> {
    let thought_id = generator.tokenizer().thought_id;
    generator.controller_mut().reset();
    generator.controller_mut().seed_from_thought(thought_hv);

    let mut prev_token = thought_id;
    let mut out = Vec::with_capacity(steps);
    for pos in 0..steps {
        let logits = generator
            .controller_mut()
            .forward_step(thought_hv, prev_token, pos);
        let target_token_id = target_ids.get(pos).copied();
        let diag = summarize_logits(generator, pos, prev_token, target_token_id, &logits);
        prev_token = if teacher_forced {
            target_token_id.unwrap_or(diag.selected_token_id)
        } else {
            diag.selected_token_id
        };
        out.push(diag);
    }
    out
}

fn summarize_logits(
    generator: &BrocaGenerator,
    pos: usize,
    prev_token_id: u32,
    target_token_id: Option<u32>,
    logits: &[f32],
) -> TokenStepDiag {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_values: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum_exp: f32 = exp_values.iter().sum();
    let probabilities: Vec<f32> = if sum_exp > 0.0 && sum_exp.is_finite() {
        exp_values.iter().map(|e| e / sum_exp).collect()
    } else {
        vec![0.0; logits.len()]
    };

    let entropy = probabilities
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();
    let (selected_idx, &selected_logit) = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap_or((0, &0.0));
    let selected_token_id = selected_idx as u32;
    let max_probability = probabilities.get(selected_idx).copied().unwrap_or(0.0);
    let mean = if logits.is_empty() {
        0.0
    } else {
        logits.iter().sum::<f32>() / logits.len() as f32
    };
    let variance = if logits.is_empty() {
        0.0
    } else {
        logits
            .iter()
            .map(|l| {
                let d = *l - mean;
                d * d
            })
            .sum::<f32>()
            / logits.len() as f32
    };

    let top_k = top_k_tokens(generator, logits, &probabilities, TOP_K);
    let target_logit = target_token_id.and_then(|id| logits.get(id as usize).copied());
    let target_probability = target_token_id.and_then(|id| probabilities.get(id as usize).copied());
    let target_rank = target_token_id.and_then(|id| {
        logits
            .get(id as usize)
            .map(|&target| 1 + logits.iter().filter(|&&logit| logit > target).count())
    });

    let _ = selected_logit;
    TokenStepDiag {
        pos,
        prev_token_id,
        prev_token: token_str(generator, prev_token_id),
        target_token_id,
        target_token: target_token_id.map(|id| token_str(generator, id)),
        target_rank,
        target_probability,
        target_logit,
        selected_token_id,
        selected_token: token_str(generator, selected_token_id),
        entropy,
        max_probability,
        logit_mean: mean,
        logit_stddev: variance.sqrt(),
        top_k,
    }
}

fn top_k_tokens(
    generator: &BrocaGenerator,
    logits: &[f32],
    probabilities: &[f32],
    k: usize,
) -> Vec<TopToken> {
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    indices
        .into_iter()
        .take(k)
        .enumerate()
        .map(|(rank, idx)| TopToken {
            rank: rank + 1,
            token_id: idx as u32,
            token: token_str(generator, idx as u32),
            logit: logits[idx],
            probability: probabilities.get(idx).copied().unwrap_or(0.0),
        })
        .collect()
}

fn collect_target_token_ids(
    generator: &BrocaGenerator,
    cases: &[CanonicalEvalCase],
) -> BTreeSet<u32> {
    cases
        .iter()
        .flat_map(|case| generator.tokenizer().encode(&case.target_text))
        .collect()
}

fn checkpoint_delta(
    baseline: &BrocaCheckpoint,
    candidate: &BrocaCheckpoint,
    target_token_ids: &BTreeSet<u32>,
) -> CheckpointDelta {
    let compared = baseline
        .token_embeddings
        .len()
        .min(candidate.token_embeddings.len());
    let dim = baseline
        .token_embeddings
        .first()
        .map(ContinuousHV::dim)
        .unwrap_or(0)
        .min(
            candidate
                .token_embeddings
                .first()
                .map(ContinuousHV::dim)
                .unwrap_or(0),
        );

    let mut total_l2 = 0.0;
    let mut max_l2 = 0.0f32;
    let mut total_cosine = 0.0;
    for i in 0..compared {
        let before = baseline.token_embeddings[i].as_slice();
        let after = candidate.token_embeddings[i].as_slice();
        let l2 = l2_delta(before, after, dim);
        total_l2 += l2;
        max_l2 = max_l2.max(l2);
        total_cosine += cosine(before, after, dim);
    }

    let target_token_embedding_deltas = target_token_ids
        .iter()
        .copied()
        .filter_map(|id| {
            let idx = id as usize;
            let before = baseline.token_embeddings.get(idx)?;
            let after = candidate.token_embeddings.get(idx)?;
            Some(TokenEmbeddingDelta {
                token_id: id,
                token: baseline
                    .vocab
                    .tokens
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| "<out-of-vocab>".to_string()),
                l2_delta: l2_delta(before.as_slice(), after.as_slice(), dim),
                cosine: cosine(before.as_slice(), after.as_slice(), dim),
            })
        })
        .collect();

    let baseline_network = bincode::serialize(&baseline.network_state).ok();
    let candidate_network = bincode::serialize(&candidate.network_state).ok();
    let baseline_hash = baseline_network
        .as_ref()
        .map(|bytes| blake3::hash(bytes).to_hex().to_string());
    let candidate_hash = candidate_network
        .as_ref()
        .map(|bytes| blake3::hash(bytes).to_hex().to_string());

    CheckpointDelta {
        baseline_training_epoch: baseline.training_epoch,
        candidate_training_epoch: candidate.training_epoch,
        baseline_training_loss: baseline.training_loss,
        candidate_training_loss: candidate.training_loss,
        embedding_count_compared: compared,
        embedding_dim_compared: dim,
        mean_embedding_l2_delta: if compared == 0 {
            0.0
        } else {
            total_l2 / compared as f32
        },
        max_embedding_l2_delta: max_l2,
        mean_embedding_cosine: if compared == 0 {
            0.0
        } else {
            total_cosine / compared as f32
        },
        target_token_embedding_deltas,
        network_state_serialized_len_baseline: baseline_network.as_ref().map(Vec::len),
        network_state_serialized_len_candidate: candidate_network.as_ref().map(Vec::len),
        network_state_hash_equal: baseline_hash
            .as_ref()
            .zip(candidate_hash.as_ref())
            .map(|(a, b)| a == b),
        network_state_hash_baseline: baseline_hash,
        network_state_hash_candidate: candidate_hash,
    }
}

fn load_checkpoint(path: &PathBuf, allow_checksum_mismatch: bool) -> Result<BrocaCheckpoint> {
    if allow_checksum_mismatch {
        BrocaCheckpoint::load_from_file_allow_checksum_mismatch(path)
    } else {
        BrocaCheckpoint::load_from_file(path)
    }
}

fn case_channels(case: &CanonicalEvalCase) -> ThoughtChannels {
    TrainingPair {
        channels: case.channels.clone(),
        target_text: case.target_text.clone(),
        target_ids: Vec::new(),
        valence: 0.0,
        arousal: 0.5,
    }
    .to_thought_channels()
}

fn token_str(generator: &BrocaGenerator, id: u32) -> String {
    generator.tokenizer().token_str(id).to_string()
}

fn l2_delta(before: &[f32], after: &[f32], dim: usize) -> f32 {
    before
        .iter()
        .zip(after.iter())
        .take(dim)
        .map(|(&a, &b)| {
            let d = b - a;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

fn cosine(before: &[f32], after: &[f32], dim: usize) -> f32 {
    let mut dot = 0.0;
    let mut before_norm = 0.0;
    let mut after_norm = 0.0;
    for (&a, &b) in before.iter().zip(after.iter()).take(dim) {
        dot += a * b;
        before_norm += a * a;
        after_norm += b * b;
    }
    if before_norm <= 0.0 || after_norm <= 0.0 {
        0.0
    } else {
        dot / (before_norm.sqrt() * after_norm.sqrt())
    }
}
