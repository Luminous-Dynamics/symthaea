// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! broca-eval: Interactive evaluation tool for Broca CfC-HDC language checkpoints.
//!
//! Usage:
//!   broca-eval --checkpoint broca-cfc-v4.bin
//!   broca-eval --checkpoint broca-cfc-v4.bin --temperature 0.8 --top-k 40
//!   broca-eval --checkpoint broca-cfc-v4.bin --eval eval.jsonl --samples 20
//!
//! Modes:
//! - Interactive: type intent/epistemic/valence, see generated text in real-time
//! - Batch: run eval dataset + sample generation with configurable sampling

use std::collections::HashMap;
use std::process::{self, Command};

use serde::Serialize;
use sha2::{Digest, Sha256};
use symthaea_broca::encoder::ThoughtChannels;
use symthaea_broca::evaluation;
use symthaea_broca::generator::{
    BrocaGenerator, GenerationResult, GenerationStepLogits, SamplingStrategy,
};
use symthaea_broca::training::{TrainingDataset, TrainingPair};

use symthaea_core::genesis::GenesisSeed;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn".parse().unwrap()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let opts = match parse_args(&args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("Error: {e}");
            print_usage();
            process::exit(1);
        }
    };

    let genesis = GenesisSeed::from_phrase(&opts.genesis_phrase);

    // Load checkpoint
    eprintln!("Loading checkpoint: {}", opts.checkpoint_path);
    let load_result = if opts.allow_checkpoint_recovery {
        BrocaGenerator::from_checkpoint_allow_checksum_mismatch(&opts.checkpoint_path, &genesis)
    } else {
        BrocaGenerator::from_checkpoint(&opts.checkpoint_path, &genesis)
    };
    let (mut generator, _adam, _proj, _lm) = match load_result {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to load checkpoint: {e}");
            process::exit(1);
        }
    };
    if opts.thought_logit_residual_weight > 0.0 {
        generator
            .controller_mut()
            .config_mut()
            .thought_logit_residual_weight = opts.thought_logit_residual_weight.clamp(0.0, 1.0);
    }

    // Override sampling strategy
    let sampling = match (opts.top_k, opts.top_p) {
        (Some(k), _) => SamplingStrategy::TopK {
            k,
            temperature: opts.temperature,
        },
        (_, Some(p)) => SamplingStrategy::TopP {
            p,
            temperature: opts.temperature,
        },
        _ if (opts.temperature - 1.0).abs() > 0.01 || opts.temperature < 0.99 => {
            // Temperature != 1.0 implies we want stochastic sampling
            SamplingStrategy::TopK {
                k: 50,
                temperature: opts.temperature,
            }
        }
        _ => SamplingStrategy::Greedy,
    };

    // Apply sampling strategy by reconstructing with modified config
    // (BrocaGenerator doesn't expose config mutation, so we print info)
    eprintln!(
        "Sampling: {:?}",
        match &sampling {
            SamplingStrategy::Greedy => "greedy".to_string(),
            SamplingStrategy::TopK { k, temperature } => format!("top-k={k}, temp={temperature}"),
            SamplingStrategy::TopP { p, temperature } => format!("top-p={p}, temp={temperature}"),
        }
    );

    // Run canonical quality suite if requested.
    if let Some(ref canonical_path) = opts.canonical_eval_path {
        eprintln!("Running canonical quality suite on: {canonical_path}");
        let canonical = match evaluation::CanonicalEvalDataset::from_jsonl(canonical_path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Failed to load canonical eval data: {e}");
                process::exit(1);
            }
        };
        let mut result = evaluation::evaluate_quality_suite(
            &mut generator,
            &canonical,
            opts.max_gen_tokens,
            opts.eval_limit,
            !opts.teacher_forced_only,
        );
        result.metadata = Some(build_quality_metadata(&opts));
        let failures = evaluation::check_quality_suite(&result, &opts.quality_thresholds);

        if let Some(ref json_out) = opts.json_output_path {
            let json = serde_json::to_string_pretty(&result).expect("quality report serializes");
            if let Err(e) = std::fs::write(json_out, json) {
                eprintln!("Failed to write JSON report '{json_out}': {e}");
                process::exit(1);
            }
        } else {
            println!(
                "{}",
                serde_json::to_string_pretty(&result).expect("quality report serializes")
            );
        }
        if let Some(ref dump_path) = opts.dump_generations_path {
            if opts.teacher_forced_only {
                eprintln!(
                    "--dump-generations requested with --teacher-forced-only; writing generated diagnostics anyway"
                );
            }
            if let Err(e) = dump_canonical_generations(
                &mut generator,
                &canonical,
                &opts.checkpoint_path,
                opts.eval_limit,
                opts.max_gen_tokens,
                dump_path,
            ) {
                eprintln!("Failed to write generation dump '{dump_path}': {e}");
                process::exit(1);
            }
        }

        if !opts.report_only && !failures.is_empty() {
            eprintln!("Canonical quality gate failed:");
            for failure in &failures {
                eprintln!(
                    "  {} observed={:.6} threshold={:.6}: {}",
                    failure.metric, failure.observed, failure.threshold, failure.message
                );
            }
            process::exit(2);
        } else if opts.report_only && !failures.is_empty() {
            eprintln!(
                "Canonical quality report has {} threshold failure(s); continuing due to --report-only",
                failures.len()
            );
        }
    }

    // Run evaluation if --eval provided
    if let Some(ref eval_path) = opts.eval_path {
        eprintln!("Evaluating on: {eval_path}");
        let mut eval_dataset = match TrainingDataset::from_jsonl(eval_path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Failed to load eval data: {e}");
                process::exit(1);
            }
        };
        let tokenizer = generator.tokenizer().clone();
        eval_dataset.tokenize_all(&tokenizer);

        let eval_config = evaluation::EvalConfig {
            dataset: eval_dataset,
            compute_perplexity: true,
            compute_english_ratio: true,
            per_intent_breakdown: true,
            max_gen_tokens: opts.max_gen_tokens,
            eval_limit: opts.eval_limit,
            progress: true,
            compute_contrastive_intent: true,
        };

        let result = evaluation::evaluate(&mut generator, &eval_config);
        if let Some(ref json_out) = opts.json_output_path {
            let json = serde_json::to_string_pretty(&result).expect("eval report serializes");
            if let Err(e) = std::fs::write(json_out, json) {
                eprintln!("Failed to write JSON report '{json_out}': {e}");
                process::exit(1);
            }
        } else {
            println!("{}", evaluation::format_eval_report(&result));
        }
    }

    // Generate samples
    let intent_names = ["Ack", "Ans", "Clr", "Pro", "Unc", "Ref", "Con", "Unk"];
    let epistemic_names = ["Certain", "Probable", "Uncertain", "Unknown", "OOD"];

    if opts.sample_count > 0 {
        println!("\n=== Sample Generations ===");
        println!(
            "{:>3}  {:>5}  {:>10}  {:>4}  {:>4}  {}",
            "#", "Intent", "Epistemic", "Psi", "Tok", "Generated Text"
        );
        println!("{}", "-".repeat(80));

        let thoughts = symthaea_broca::training::generate_diverse_thoughts();
        let step = thoughts.len() / opts.sample_count.max(1);

        for i in 0..opts.sample_count.min(thoughts.len()) {
            let channels = &thoughts[(i * step) % thoughts.len()];
            let result = generator.generate(channels);

            let intent_idx = (0..8)
                .max_by(|&a, &b| channels.channels[a].total_cmp(&channels.channels[b]))
                .unwrap_or(7);
            let ep_idx = (channels.epistemic_ordinal() as usize).min(4);

            let display_text = if result.text.len() > 100 {
                format!("{}...", &result.text[..100])
            } else {
                result.text.clone()
            };

            println!(
                "{:>3}  {:>5}  {:>10}  {:.1}  {:>4}  \"{}\"",
                i,
                intent_names[intent_idx],
                epistemic_names[ep_idx],
                channels.psi(),
                result.num_tokens,
                display_text,
            );
        }
    }

    // Interactive mode (if no --eval and no --samples, or --interactive)
    if opts.interactive
        || (opts.eval_path.is_none()
            && opts.canonical_eval_path.is_none()
            && opts.sample_count == 0
            && !opts.samples_requested)
    {
        println!("\n=== Interactive Mode ===");
        println!("Enter thought channels as: intent(0-7) epistemic(0-4) valence(-1..1) arousal(0..1) psi(0..1)");
        println!("Example: 1 0 0.5 0.3 0.7  (Answer, Certain, positive, calm, aware)");
        println!("Or just press Enter for defaults. Type 'quit' to exit.\n");

        let stdin = std::io::stdin();
        loop {
            eprint!("> ");
            let mut line = String::new();
            if stdin.read_line(&mut line).is_err() || line.trim() == "quit" {
                break;
            }

            let parts: Vec<f32> = line
                .trim()
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();

            let mut channels = ThoughtChannels::default();
            if parts.len() >= 1 {
                channels = ThoughtChannels::with_intent(parts[0] as usize);
            }
            if parts.len() >= 2 {
                channels.set_epistemic(parts[1]);
            }
            if parts.len() >= 3 {
                channels.set_emotion(parts[2], parts.get(3).copied().unwrap_or(0.5), 0.5);
            }
            if parts.len() >= 5 {
                channels.set_consciousness(parts[4], 0.5, 0.5);
            }

            let result = generator.generate(&channels);
            println!(
                "  [{} tok, coh={:.3}] \"{}\"",
                result.num_tokens, result.final_coherence, result.text
            );
        }
    }
}

fn build_quality_metadata(opts: &EvalOpts) -> evaluation::QualityRunMetadata {
    evaluation::QualityRunMetadata {
        backend: std::env::var("BROCA_EVAL_BACKEND").unwrap_or_else(|_| "unknown".to_string()),
        eval_lane: std::env::var("BROCA_EVAL_LANE").unwrap_or_else(|_| {
            if opts.teacher_forced_only {
                "fast".to_string()
            } else {
                "full".to_string()
            }
        }),
        checkpoint_path: opts.checkpoint_path.clone(),
        checkpoint_sha256: checkpoint_sha256(&opts.checkpoint_path),
        git_commit: git_commit(),
        feature_set: std::env::var("BROCA_EVAL_FEATURES")
            .map(|features| {
                features
                    .split(',')
                    .map(str::trim)
                    .filter(|feature| !feature.is_empty())
                    .map(ToOwned::to_owned)
                    .collect()
            })
            .unwrap_or_default(),
        train_recipe: std::env::var("BROCA_TRAIN_RECIPE").ok(),
        train_pair_count: parse_env_usize("BROCA_TRAIN_PAIR_COUNT"),
        train_pair_selection: std::env::var("BROCA_TRAIN_PAIR_SELECTION").ok(),
        train_epochs: parse_env_usize("BROCA_TRAIN_EPOCHS"),
        train_bptt_window: parse_env_usize("BROCA_TRAIN_BPTT_WINDOW"),
        train_negative_samples: parse_env_usize("BROCA_TRAIN_NEGATIVE_SAMPLES"),
        train_learning_rate: parse_env_f32("BROCA_TRAIN_LR"),
        train_network_lr_scale: parse_env_f32("BROCA_TRAIN_NETWORK_LR_SCALE"),
        train_network_layers: parse_env_usize("BROCA_TRAIN_NETWORK_LAYERS"),
        train_neurons_per_layer: parse_env_usize("BROCA_TRAIN_NEURONS_PER_LAYER"),
        train_coherence_alignment: parse_env_f32("BROCA_TRAIN_COHERENCE_ALIGNMENT"),
        train_alignment_start: parse_env_f32("BROCA_TRAIN_ALIGNMENT_START"),
        train_contrastive: parse_env_f32("BROCA_TRAIN_CONTRASTIVE"),
        train_contrastive_margin: parse_env_f32("BROCA_TRAIN_CONTRASTIVE_MARGIN"),
        train_scheduled_sampling: parse_env_f32("BROCA_TRAIN_SCHEDULED_SAMPLING"),
        train_label_smoothing: parse_env_f32("BROCA_TRAIN_LABEL_SMOOTHING"),
        train_thought_logit_aux: parse_env_f32("BROCA_TRAIN_THOUGHT_LOGIT_AUX"),
        train_thought_logit_residual: parse_env_f32("BROCA_TRAIN_THOUGHT_LOGIT_RESIDUAL"),
        train_merge_bias: parse_env_f32("BROCA_TRAIN_MERGE_BIAS"),
    }
}

fn checkpoint_sha256(path: &str) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    let digest = Sha256::digest(bytes);
    Some(hex::encode(digest))
}

fn git_commit() -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let commit = String::from_utf8(output.stdout).ok()?;
    let commit = commit.trim();
    (!commit.is_empty()).then(|| commit.to_string())
}

fn parse_env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
}

fn parse_env_f32(name: &str) -> Option<f32> {
    std::env::var(name).ok()?.parse().ok()
}

#[derive(Debug, Serialize)]
struct CanonicalGenerationDumpRecord {
    checkpoint_path: String,
    case_index: usize,
    case_id: String,
    category: String,
    tags: Vec<String>,
    intent: String,
    target_text: String,
    raw: GenerationDump,
    gated: GenerationDump,
}

#[derive(Debug, Serialize)]
struct GenerationDump {
    text: String,
    token_ids: Vec<u32>,
    token_count: usize,
    eos_terminated: bool,
    veto_triggered: bool,
    final_coherence: f32,
    long_coherence: f32,
    coherence_dynamics: Vec<f32>,
    logit_diagnostics: Vec<GenerationStepLogitDump>,
    hallucination_flag: bool,
    nsm_prime_coverage: f32,
    repeated_tokens: Vec<TokenFrequency>,
}

#[derive(Debug, Serialize)]
struct GenerationStepLogitDump {
    position: usize,
    selected_token_id: u32,
    selected_token: String,
    entropy: f32,
    max_probability: f32,
    top_k: Vec<GenerationTopLogitDump>,
}

#[derive(Debug, Serialize)]
struct GenerationTopLogitDump {
    rank: usize,
    token_id: u32,
    token: String,
    logit: f32,
    probability: f32,
}

#[derive(Debug, Serialize)]
struct TokenFrequency {
    token_id: u32,
    token: String,
    count: usize,
}

fn dump_canonical_generations(
    generator: &mut BrocaGenerator,
    dataset: &evaluation::CanonicalEvalDataset,
    checkpoint_path: &str,
    eval_limit: usize,
    max_gen_tokens: usize,
    output_path: &str,
) -> std::io::Result<()> {
    let mut out = String::new();
    let cases = limited_canonical_cases(dataset, eval_limit);
    let original_bypass = generator.config().bypass_gating;

    for (case_index, case) in cases.iter().enumerate() {
        let channels = canonical_case_channels(case);

        generator.config_mut().bypass_gating = true;
        let raw = generate_for_dump(generator, &channels, max_gen_tokens);

        generator.config_mut().bypass_gating = false;
        let gated = generate_for_dump(generator, &channels, max_gen_tokens);

        let record = CanonicalGenerationDumpRecord {
            checkpoint_path: checkpoint_path.to_string(),
            case_index,
            case_id: case.id.clone(),
            category: case.category.clone(),
            tags: case.tags.clone(),
            intent: active_intent_name(&case.channels).to_string(),
            target_text: case.target_text.clone(),
            raw: generation_dump(generator, raw),
            gated: generation_dump(generator, gated),
        };
        out.push_str(&serde_json::to_string(&record).expect("generation dump record serializes"));
        out.push('\n');
    }

    generator.config_mut().bypass_gating = original_bypass;
    std::fs::write(output_path, out)
}

fn limited_canonical_cases(
    dataset: &evaluation::CanonicalEvalDataset,
    eval_limit: usize,
) -> &[evaluation::CanonicalEvalCase] {
    if eval_limit > 0 && eval_limit < dataset.cases.len() {
        &dataset.cases[..eval_limit]
    } else {
        &dataset.cases
    }
}

fn canonical_case_channels(case: &evaluation::CanonicalEvalCase) -> ThoughtChannels {
    TrainingPair {
        channels: case.channels.clone(),
        target_text: case.target_text.clone(),
        target_ids: vec![],
        valence: 0.0,
        arousal: 0.5,
    }
    .to_thought_channels()
}

fn generate_for_dump(
    generator: &mut BrocaGenerator,
    channels: &ThoughtChannels,
    max_gen_tokens: usize,
) -> GenerationResult {
    let original_base_max_tokens = generator.config().gating.base_max_tokens;
    let original_consciousness_gating = generator.config().enable_consciousness_gating;
    {
        let config = generator.config_mut();
        config.gating.base_max_tokens = max_gen_tokens;
        config.enable_consciousness_gating = false;
    }
    let result = generator.generate(channels);
    {
        let config = generator.config_mut();
        config.gating.base_max_tokens = original_base_max_tokens;
        config.enable_consciousness_gating = original_consciousness_gating;
    }
    result
}

fn generation_dump(generator: &BrocaGenerator, result: GenerationResult) -> GenerationDump {
    GenerationDump {
        repeated_tokens: repeated_tokens(generator, &result.token_ids, 8),
        text: result.text,
        token_ids: result.token_ids,
        token_count: result.num_tokens,
        eos_terminated: result.eos_terminated,
        veto_triggered: result.veto_triggered,
        final_coherence: result.final_coherence,
        long_coherence: result.long_coherence,
        logit_diagnostics: logit_diagnostics_dump(generator, &result.logit_diagnostics),
        coherence_dynamics: result.coherence_dynamics,
        hallucination_flag: result.hallucination_flag,
        nsm_prime_coverage: result.nsm_prime_coverage,
    }
}

fn logit_diagnostics_dump(
    generator: &BrocaGenerator,
    diagnostics: &[GenerationStepLogits],
) -> Vec<GenerationStepLogitDump> {
    diagnostics
        .iter()
        .map(|step| GenerationStepLogitDump {
            position: step.position,
            selected_token_id: step.selected_token_id,
            selected_token: generator
                .tokenizer()
                .token_str(step.selected_token_id)
                .to_string(),
            entropy: step.entropy,
            max_probability: step.max_probability,
            top_k: step
                .top_k
                .iter()
                .map(|entry| GenerationTopLogitDump {
                    rank: entry.rank,
                    token_id: entry.token_id,
                    token: generator.tokenizer().token_str(entry.token_id).to_string(),
                    logit: entry.logit,
                    probability: entry.probability,
                })
                .collect(),
        })
        .collect()
}

fn repeated_tokens(
    generator: &BrocaGenerator,
    token_ids: &[u32],
    limit: usize,
) -> Vec<TokenFrequency> {
    let mut counts: HashMap<u32, usize> = HashMap::new();
    for &token_id in token_ids {
        *counts.entry(token_id).or_insert(0) += 1;
    }
    let mut counts: Vec<_> = counts.into_iter().filter(|(_, count)| *count > 1).collect();
    counts.sort_by(|(left_id, left_count), (right_id, right_count)| {
        right_count
            .cmp(left_count)
            .then_with(|| left_id.cmp(right_id))
    });
    counts
        .into_iter()
        .take(limit)
        .map(|(token_id, count)| TokenFrequency {
            token_id,
            token: generator.tokenizer().token_str(token_id).to_string(),
            count,
        })
        .collect()
}

fn active_intent_name(channels: &[f32]) -> &'static str {
    const INTENT_NAMES: [&str; 8] = [
        "Acknowledge",
        "Answer",
        "Clarify",
        "Propose",
        "Uncertainty",
        "Reflect",
        "Continue",
        "Unknown",
    ];
    if channels.len() < INTENT_NAMES.len() {
        return "Unknown";
    }
    let idx = (0..INTENT_NAMES.len())
        .max_by(|&a, &b| channels[a].total_cmp(&channels[b]))
        .unwrap_or(INTENT_NAMES.len() - 1);
    INTENT_NAMES[idx]
}

struct EvalOpts {
    checkpoint_path: String,
    eval_path: Option<String>,
    canonical_eval_path: Option<String>,
    json_output_path: Option<String>,
    dump_generations_path: Option<String>,
    genesis_phrase: String,
    sample_count: usize,
    samples_requested: bool,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    interactive: bool,
    eval_limit: usize,
    max_gen_tokens: usize,
    quality_thresholds: evaluation::CanonicalQualityThresholds,
    report_only: bool,
    teacher_forced_only: bool,
    allow_checkpoint_recovery: bool,
    thought_logit_residual_weight: f32,
}

fn parse_args(args: &[String]) -> Result<EvalOpts, String> {
    let mut opts = EvalOpts {
        checkpoint_path: String::new(),
        eval_path: None,
        canonical_eval_path: None,
        json_output_path: None,
        dump_generations_path: None,
        genesis_phrase: "broca-training-default".to_string(),
        sample_count: 0,
        samples_requested: false,
        temperature: 1.0,
        top_k: None,
        top_p: None,
        interactive: false,
        eval_limit: 0,
        max_gen_tokens: 64,
        quality_thresholds: evaluation::CanonicalQualityThresholds::default(),
        report_only: false,
        teacher_forced_only: false,
        allow_checkpoint_recovery: false,
        thought_logit_residual_weight: 0.0,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--checkpoint" | "-c" => {
                i += 1;
                opts.checkpoint_path =
                    args.get(i).cloned().ok_or("--checkpoint requires a path")?;
            }
            "--eval" => {
                i += 1;
                opts.eval_path = Some(args.get(i).cloned().ok_or("--eval requires a path")?);
            }
            "--canonical-eval" => {
                i += 1;
                opts.canonical_eval_path = Some(
                    args.get(i)
                        .cloned()
                        .ok_or("--canonical-eval requires a path")?,
                );
            }
            "--json-out" => {
                i += 1;
                opts.json_output_path =
                    Some(args.get(i).cloned().ok_or("--json-out requires a path")?);
            }
            "--dump-generations" => {
                i += 1;
                opts.dump_generations_path = Some(
                    args.get(i)
                        .cloned()
                        .ok_or("--dump-generations requires a path")?,
                );
            }
            "--genesis" => {
                i += 1;
                opts.genesis_phrase = args.get(i).cloned().ok_or("--genesis requires a phrase")?;
            }
            "--samples" | "-n" => {
                i += 1;
                opts.samples_requested = true;
                opts.sample_count = args
                    .get(i)
                    .ok_or("--samples requires a number")?
                    .parse()
                    .map_err(|_| "--samples must be a number")?;
            }
            "--temperature" | "-t" => {
                i += 1;
                opts.temperature = args
                    .get(i)
                    .ok_or("--temperature requires a number")?
                    .parse()
                    .map_err(|_| "--temperature must be a float")?;
            }
            "--top-k" => {
                i += 1;
                opts.top_k = Some(
                    args.get(i)
                        .ok_or("--top-k requires a number")?
                        .parse()
                        .map_err(|_| "--top-k must be a number")?,
                );
            }
            "--top-p" => {
                i += 1;
                opts.top_p = Some(
                    args.get(i)
                        .ok_or("--top-p requires a number")?
                        .parse()
                        .map_err(|_| "--top-p must be a float")?,
                );
            }
            "--interactive" | "-i" => {
                opts.interactive = true;
            }
            "--allow-checkpoint-recovery" => {
                opts.allow_checkpoint_recovery = true;
            }
            "--thought-logit-residual" => {
                i += 1;
                opts.thought_logit_residual_weight = args
                    .get(i)
                    .ok_or("--thought-logit-residual requires a number")?
                    .parse()
                    .map_err(|_| "--thought-logit-residual must be a float")?;
            }
            "--report-only" => {
                opts.report_only = true;
            }
            "--teacher-forced-only" => {
                opts.teacher_forced_only = true;
            }
            "--eval-limit" => {
                i += 1;
                opts.eval_limit = args
                    .get(i)
                    .ok_or("--eval-limit requires a number")?
                    .parse()
                    .map_err(|_| "--eval-limit must be a number")?;
            }
            "--max-gen-tokens" => {
                i += 1;
                opts.max_gen_tokens = args
                    .get(i)
                    .ok_or("--max-gen-tokens requires a number")?
                    .parse()
                    .map_err(|_| "--max-gen-tokens must be a number")?;
            }
            "--max-gated-perplexity" => {
                i += 1;
                opts.quality_thresholds.max_gated_perplexity = Some(
                    args.get(i)
                        .ok_or("--max-gated-perplexity requires a number")?
                        .parse()
                        .map_err(|_| "--max-gated-perplexity must be a float")?,
                );
            }
            "--min-gated-coherence" => {
                i += 1;
                opts.quality_thresholds.min_gated_coherence = Some(
                    args.get(i)
                        .ok_or("--min-gated-coherence requires a number")?
                        .parse()
                        .map_err(|_| "--min-gated-coherence must be a float")?,
                );
            }
            "--min-gated-english-ratio" => {
                i += 1;
                opts.quality_thresholds.min_gated_english_ratio = Some(
                    args.get(i)
                        .ok_or("--min-gated-english-ratio requires a number")?
                        .parse()
                        .map_err(|_| "--min-gated-english-ratio must be a float")?,
                );
            }
            "--max-gated-hallucination-rate" => {
                i += 1;
                opts.quality_thresholds.max_gated_hallucination_rate = Some(
                    args.get(i)
                        .ok_or("--max-gated-hallucination-rate requires a number")?
                        .parse()
                        .map_err(|_| "--max-gated-hallucination-rate must be a float")?,
                );
            }
            "--max-coherence-regression" => {
                i += 1;
                opts.quality_thresholds.max_coherence_regression = Some(
                    args.get(i)
                        .ok_or("--max-coherence-regression requires a number")?
                        .parse()
                        .map_err(|_| "--max-coherence-regression must be a float")?,
                );
            }
            "--max-target-overlap-regression" => {
                i += 1;
                opts.quality_thresholds.max_target_overlap_regression = Some(
                    args.get(i)
                        .ok_or("--max-target-overlap-regression requires a number")?
                        .parse()
                        .map_err(|_| "--max-target-overlap-regression must be a float")?,
                );
            }
            "--min-moral-refusal-rate" => {
                i += 1;
                opts.quality_thresholds.min_moral_refusal_rate = Some(
                    args.get(i)
                        .ok_or("--min-moral-refusal-rate requires a number")?
                        .parse()
                        .map_err(|_| "--min-moral-refusal-rate must be a float")?,
                );
            }
            "--max-code-sheaf-incoherence-rate" => {
                i += 1;
                opts.quality_thresholds.max_code_sheaf_incoherence_rate = Some(
                    args.get(i)
                        .ok_or("--max-code-sheaf-incoherence-rate requires a number")?
                        .parse()
                        .map_err(|_| "--max-code-sheaf-incoherence-rate must be a float")?,
                );
            }
            "--min-code-sheaf-function-coherence-rate" => {
                i += 1;
                opts.quality_thresholds
                    .min_code_sheaf_function_coherence_rate = Some(
                    args.get(i)
                        .ok_or("--min-code-sheaf-function-coherence-rate requires a number")?
                        .parse()
                        .map_err(|_| "--min-code-sheaf-function-coherence-rate must be a float")?,
                );
            }
            "--min-structured-output-validity-rate" => {
                i += 1;
                opts.quality_thresholds.min_structured_output_validity_rate = Some(
                    args.get(i)
                        .ok_or("--min-structured-output-validity-rate requires a number")?
                        .parse()
                        .map_err(|_| "--min-structured-output-validity-rate must be a float")?,
                );
            }
            "--help" | "-h" => {
                print_usage();
                process::exit(0);
            }
            arg => return Err(format!("Unknown argument: {arg}")),
        }
        i += 1;
    }

    if opts.checkpoint_path.is_empty() {
        return Err("--checkpoint is required".to_string());
    }

    Ok(opts)
}

fn print_usage() {
    eprintln!("Usage: broca-eval [OPTIONS]");
    eprintln!();
    eprintln!("Required:");
    eprintln!("  --checkpoint, -c PATH  Path to Broca checkpoint (.bin)");
    eprintln!();
    eprintln!("Optional:");
    eprintln!("  --eval PATH            Evaluate on held-out JSONL dataset");
    eprintln!("  --canonical-eval PATH  Run raw-vs-gated canonical quality suite");
    eprintln!("  --json-out PATH        Write eval result JSON instead of human report");
    eprintln!(
        "  --dump-generations PATH  Write raw/gated canonical generation diagnostics as JSONL"
    );
    eprintln!("  --eval-limit N         Limit eval pairs (default: all)");
    eprintln!("  --max-gen-tokens N     Max teacher-forced/generated tokens (default: 64)");
    eprintln!("  --report-only          Emit canonical quality JSON without applying thresholds");
    eprintln!("  --teacher-forced-only  Skip generation-heavy canonical metrics");
    eprintln!("  --max-gated-perplexity F        Fail canonical eval above this gated PPL");
    eprintln!("  --min-gated-coherence F         Fail canonical eval below this gated coherence");
    eprintln!(
        "  --min-gated-english-ratio F     Fail canonical eval below this gated English ratio"
    );
    eprintln!(
        "  --max-gated-hallucination-rate F  Fail canonical eval above this hallucination rate"
    );
    eprintln!("  --max-coherence-regression F    Fail if gating lowers coherence by more than F");
    eprintln!(
        "  --max-target-overlap-regression F  Fail if gating lowers target overlap by more than F"
    );
    eprintln!("  --min-moral-refusal-rate F       Fail if canonical moral refusal rate is below F");
    eprintln!(
        "  --max-code-sheaf-incoherence-rate F  Fail if canonical code sheaf incoherence exceeds F"
    );
    eprintln!(
        "  --min-code-sheaf-function-coherence-rate F  Fail if canonical code function coherence is below F"
    );
    eprintln!(
        "  --min-structured-output-validity-rate F  Fail if structured output validity is below F"
    );
    eprintln!("  --samples, -n N        Generate N sample outputs from diverse thoughts");
    eprintln!("  --temperature, -t F    Sampling temperature (default: 1.0)");
    eprintln!("  --top-k K              Top-k sampling (default: greedy)");
    eprintln!("  --top-p P              Top-p nucleus sampling (default: greedy)");
    eprintln!("  --interactive, -i      Force interactive mode");
    eprintln!("  --genesis PHRASE       Genesis seed phrase (default: broca-training-default)");
    eprintln!(
        "  --thought-logit-residual F  Blend direct thought logits into decoder logits (default: 0.0)"
    );
    eprintln!("  --allow-checkpoint-recovery  Load legacy/recovery checkpoints with explicit compatibility bypass");
    eprintln!("  --help, -h             Show this help message");
}
