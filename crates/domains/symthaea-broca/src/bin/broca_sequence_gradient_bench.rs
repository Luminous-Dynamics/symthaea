// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Compare repeated-prefix and single-unroll context gradients on Mamba-130M.
//!
//! This is a synthetic token/embedding microbenchmark for objective and gradient
//! parity plus runtime scaling. It is NOT a language-quality benchmark and does
//! not measure semantic task accuracy. The exact repeated-prefix baseline is
//! intentionally expensive as target length grows; keep target lengths small
//! until its runtime has been measured in the intended environment.

use anyhow::{bail, ensure, Context, Result};
use serde::Serialize;
use std::time::Instant;
use symthaea_broca::evaluation::cross_entropy_loss;
use symthaea_broca::mamba::{best_device, MambaBackend, MambaWrapper};

#[derive(Debug, Clone)]
struct Options {
    model_id: String,
    context_chunks: usize,
    target_tokens: usize,
    repeats: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            model_id: "state-spaces/mamba-130m".to_string(),
            context_chunks: 1,
            target_tokens: 2,
            repeats: 1,
        }
    }
}

#[derive(Debug, Serialize)]
struct MethodTiming {
    p50_ms: f64,
    p95_ms: f64,
    samples_ms: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct ParityReport {
    loss_abs_diff: f32,
    max_context_gradient_abs_diff: f32,
    gradient_coordinates_compared: usize,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    benchmark: &'static str,
    note: &'static str,
    model_id: String,
    device: String,
    context_chunks: usize,
    target_tokens: usize,
    repeats: usize,
    synthetic_target_ids: Vec<u32>,
    projected_context_dimension: usize,
    legacy_repeated_prefix_modeled_forward_steps: usize,
    single_unroll_modeled_forward_steps: usize,
    estimated_forward_step_reduction: f64,
    repeated_prefix: MethodTiming,
    single_unroll: MethodTiming,
    observed_p50_speedup: f64,
    single_unroll_mean_loss: f32,
    repeated_prefix_reference_loss: f32,
    parity: ParityReport,
}

#[derive(Debug)]
struct RunResult {
    elapsed_ms: f64,
    mean_loss: f32,
    context_gradients: Vec<Vec<f32>>,
}

fn parse_options() -> Result<Options> {
    let mut options = Options::default();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        let value = |name: &str, args: &mut std::iter::Skip<std::env::Args>| -> Result<String> {
            args.next()
                .with_context(|| format!("missing value after {name}"))
        };

        match arg.as_str() {
            "--model-id" => options.model_id = value(&arg, &mut args)?,
            "--context-chunks" => {
                options.context_chunks = value(&arg, &mut args)?
                    .parse()
                    .context("invalid --context-chunks value")?;
            }
            "--target-tokens" => {
                options.target_tokens = value(&arg, &mut args)?
                    .parse()
                    .context("invalid --target-tokens value")?;
            }
            "--repeats" => {
                options.repeats = value(&arg, &mut args)?
                    .parse()
                    .context("invalid --repeats value")?;
            }
            "--help" | "-h" => {
                println!(
                    "broca-sequence-gradient-bench [--model-id ID] [--context-chunks N]                      [--target-tokens N] [--repeats N]\n                     Defaults: context-chunks=1, target-tokens=2, repeats=1.\n                     Uses synthetic context embeddings and token IDs; this is not a task-quality benchmark."
                );
                std::process::exit(0);
            }
            _ => bail!("unknown argument: {arg}"),
        }
    }

    ensure!((1..=8).contains(&options.context_chunks), "--context-chunks must be in 1..=8");
    ensure!((1..=32).contains(&options.target_tokens), "--target-tokens must be in 1..=32");
    ensure!((1..=10).contains(&options.repeats), "--repeats must be in 1..=10");
    ensure!(!options.model_id.trim().is_empty(), "--model-id must not be empty");
    Ok(options)
}

fn synthetic_context(chunks: usize, d_model: usize) -> Vec<Vec<f32>> {
    (0..chunks)
        .map(|chunk| {
            (0..d_model)
                .map(|dim| {
                    let phase = (chunk * (d_model + 1) + dim + 1) as f32;
                    0.1 * (phase * 0.017).sin() + 0.03 * (phase * 0.031).cos()
                })
                .collect()
        })
        .collect()
}

fn synthetic_targets(count: usize, vocab_size: usize) -> Result<Vec<u32>> {
    // Deliberately synthetic deterministic IDs: this is a systems microbenchmark.
    ensure!(vocab_size > 128, "model vocabulary unexpectedly small");
    let targets: Vec<u32> = (0..count).map(|i| 7 + ((i * 17) % 113) as u32).collect();
    ensure!(
        targets.iter().all(|&target| (target as usize) < vocab_size),
        "synthetic target token outside model vocabulary"
    );
    Ok(targets)
}

fn reference_mean_loss(
    wrapper: &mut MambaWrapper,
    context: &[Vec<f32>],
    target_ids: &[u32],
) -> Result<f32> {
    ensure!(!target_ids.is_empty(), "target sequence must not be empty");
    wrapper.inject_context_sequence(context)?;
    let mut previous_token = wrapper.eos_token_id();
    let mut total_loss = 0.0f32;

    for (index, &target) in target_ids.iter().enumerate() {
        let (logits, _) = wrapper.forward_with_state(previous_token)?;
        ensure!(
            (target as usize) < logits.len(),
            "target token {target} outside logits vocabulary at index {index}"
        );
        ensure!(
            logits.iter().all(|value| value.is_finite()),
            "non-finite reference logits at target index {index}"
        );
        let token_loss = cross_entropy_loss(&logits, target as usize);
        ensure!(token_loss.is_finite(), "non-finite reference loss at target index {index}");
        total_loss += token_loss;
        previous_token = target;
    }

    Ok(total_loss / target_ids.len() as f32)
}

fn run_repeated_prefix(
    base: &MambaWrapper,
    context: &[Vec<f32>],
    target_ids: &[u32],
) -> Result<RunResult> {
    let mut backend = base.clone();
    let start = Instant::now();
    let d_model = backend.d_model();
    let mut gradient_sums = vec![vec![0.0f32; d_model]; context.len()];

    // Exact reference: for every target, replay its entire teacher prefix and
    // compute a gradient at every projected-context position. Accumulate each
    // per-target gradient divided by T to obtain the mean sequence objective.
    for (target_index, &target) in target_ids.iter().enumerate() {
        let mut teacher_tokens = Vec::with_capacity(target_index + 2);
        teacher_tokens.push(backend.eos_token_id());
        teacher_tokens.extend_from_slice(&target_ids[..target_index]);
        teacher_tokens.push(target);

        for context_position in 0..context.len() {
            let gradient = backend.compute_e2e_token_loss_at_isolated(
                context,
                &teacher_tokens,
                context_position,
            )?;
            ensure!(
                gradient.len() == d_model && gradient.iter().all(|value| value.is_finite()),
                "invalid repeated-prefix gradient at context position {context_position}, target {target_index}"
            );
            for (sum, value) in gradient_sums[context_position].iter_mut().zip(gradient.iter()) {
                *sum += *value / target_ids.len() as f32;
            }
        }
    }

    // The old path also needed a separate forward pass to report mean loss.
    let mean_loss = reference_mean_loss(&mut backend, context, target_ids)?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
    Ok(RunResult {
        elapsed_ms,
        mean_loss,
        context_gradients: gradient_sums,
    })
}

fn run_single_unroll(
    base: &MambaWrapper,
    context: &[Vec<f32>],
    target_ids: &[u32],
) -> Result<RunResult> {
    let mut backend = base.clone();
    let positions: Vec<usize> = (0..context.len()).collect();
    let start = Instant::now();
    let report = backend.compute_e2e_sequence_loss_and_gradients(context, target_ids, &positions)?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;

    let mut gradients = vec![vec![0.0f32; backend.d_model()]; context.len()];
    for (position, gradient) in report.context_gradients {
        ensure!(position < gradients.len(), "single-unroll gradient position out of range");
        ensure!(
            gradients[position].iter().all(|value| *value == 0.0),
            "single-unroll backend returned duplicate context gradient position {position}"
        );
        ensure!(
            gradient.len() == backend.d_model() && gradient.iter().all(|value| value.is_finite()),
            "invalid single-unroll gradient at context position {position}"
        );
        gradients[position] = gradient;
    }

    Ok(RunResult {
        elapsed_ms,
        mean_loss: report.mean_loss,
        context_gradients: gradients,
    })
}

fn percentile_ms(samples: &[f64], percentile: f64) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = (percentile.clamp(0.0, 1.0) * (sorted.len() - 1) as f64).ceil() as usize;
    sorted[rank]
}

fn timing(samples: &[RunResult]) -> MethodTiming {
    let elapsed: Vec<f64> = samples.iter().map(|sample| sample.elapsed_ms).collect();
    MethodTiming {
        p50_ms: percentile_ms(&elapsed, 0.50),
        p95_ms: percentile_ms(&elapsed, 0.95),
        samples_ms: elapsed,
    }
}

fn main() -> Result<()> {
    let options = parse_options()?;
    let mut base = MambaWrapper::load(&options.model_id, best_device())
        .with_context(|| format!("loading model {}", options.model_id))?;
    let device = format!("{:?}", base.device());
    let context = synthetic_context(options.context_chunks, base.d_model());
    let targets = synthetic_targets(options.target_tokens, base.vocab_size())?;

    // Warm each path once outside the reported samples; otherwise the first
    // path would pay one-time setup costs that bias comparisons in its favor or
    // against it. This can take time because the exact baseline is intentionally
    // expensive, so the input sizes default to a very small diagnostic.
    let _ = run_repeated_prefix(&base, &context, &targets)?;
    let _ = run_single_unroll(&base, &context, &targets)?;

    let mut repeated = Vec::with_capacity(options.repeats);
    let mut single = Vec::with_capacity(options.repeats);
    for _ in 0..options.repeats {
        repeated.push(run_repeated_prefix(&base, &context, &targets)?);
        single.push(run_single_unroll(&base, &context, &targets)?);
    }

    let reference = repeated.last().expect("repeats is positive");
    let optimized = single.last().expect("repeats is positive");
    let loss_abs_diff = (reference.mean_loss - optimized.mean_loss).abs();
    let mut max_gradient_diff = 0.0f32;
    let mut coordinates_compared = 0usize;
    for (reference_position, optimized_position) in reference
        .context_gradients
        .iter()
        .zip(optimized.context_gradients.iter())
    {
        ensure!(
            reference_position.len() == optimized_position.len(),
            "gradient shape mismatch between repeated-prefix and single-unroll methods"
        );
        for (&reference_value, &optimized_value) in
            reference_position.iter().zip(optimized_position.iter())
        {
            max_gradient_diff = max_gradient_diff.max((reference_value - optimized_value).abs());
            coordinates_compared += 1;
        }
    }

    let loss_tolerance = 1e-4 + reference.mean_loss.abs() * 1e-4;
    let gradient_tolerance = 5e-3;
    let parity = ParityReport {
        loss_abs_diff,
        max_context_gradient_abs_diff: max_gradient_diff,
        gradient_coordinates_compared: coordinates_compared,
        passed: loss_abs_diff <= loss_tolerance && max_gradient_diff <= gradient_tolerance,
    };

    let context_chunks = context.len();
    let target_tokens = targets.len();
    let repeated_steps = context_chunks * (context_chunks * target_tokens
        + target_tokens * (target_tokens + 1) / 2)
        + context_chunks
        + target_tokens;
    let single_steps = context_chunks + target_tokens;
    let report = BenchmarkReport {
        benchmark: "broca-sequence-gradient-microbenchmark",
        note: "Synthetic token/context microbenchmark only; not a language-quality score. Forward-step counts are algorithmic estimates, not instrumented kernel counts.",
        model_id: options.model_id,
        device,
        context_chunks,
        target_tokens,
        repeats: options.repeats,
        synthetic_target_ids: targets,
        projected_context_dimension: base.d_model(),
        legacy_repeated_prefix_modeled_forward_steps: repeated_steps,
        single_unroll_modeled_forward_steps: single_steps,
        estimated_forward_step_reduction: repeated_steps as f64 / single_steps as f64,
        repeated_prefix: timing(&repeated),
        single_unroll: timing(&single),
        observed_p50_speedup: timing(&repeated).p50_ms / timing(&single).p50_ms.max(f64::MIN_POSITIVE),
        single_unroll_mean_loss: optimized.mean_loss,
        repeated_prefix_reference_loss: reference.mean_loss,
        parity,
    };

    println!("{}", serde_json::to_string_pretty(&report)?);
    ensure!(
        report.parity.passed,
        "loss/gradient parity failed (loss abs diff {}, max gradient abs diff {})",
        report.parity.loss_abs_diff,
        report.parity.max_context_gradient_abs_diff
    );

    Ok(())
}
