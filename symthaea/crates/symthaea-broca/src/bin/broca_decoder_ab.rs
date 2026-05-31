// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! broca-decoder-ab: compare direct CfC/HDC, structured molecule, and optional
//! Liquid-Mamba translation paths on the same canonical thought inputs.

use std::process;

use anyhow::{Context, Result};
use serde::Serialize;
use symthaea_broca::decoder::{StructuredDecoder, StructuredReadout, StructuredRoleFill};
use symthaea_broca::evaluation::{CanonicalEvalCase, CanonicalEvalDataset};
use symthaea_broca::generator::{BrocaConfig, BrocaDecoderKind, BrocaGenerator, GenerationResult};
use symthaea_broca::training::TrainingPair;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

#[cfg(feature = "mamba-cpu")]
use symthaea_broca::checkpoint::ProjectionCheckpoint;
#[cfg(feature = "mamba-cpu")]
use symthaea_broca::liquid_mamba::{LiquidMambaConfig, LiquidMambaGenerator};

#[derive(Debug, Clone)]
struct Options {
    checkpoint_path: Option<String>,
    mamba_projection_checkpoint: Option<String>,
    canonical_path: String,
    json_out: Option<String>,
    eval_limit: usize,
    max_gen_tokens: usize,
    genesis_phrase: String,
    allow_checkpoint_recovery: bool,
    decoders: Vec<BrocaDecoderKind>,
    fail_on_gate: bool,
    max_direct_drift: f32,
    max_mamba_drift: f32,
    max_hallucination_rate: f32,
    min_structured_validity: f32,
    min_structured_required_role_rate: f32,
    include_structured_molecule: bool,
}

#[derive(Debug, Serialize)]
struct DecoderAbReport {
    schema_version: u32,
    evidence_level: String,
    checkpoint_path: Option<String>,
    mamba_projection_checkpoint: Option<String>,
    canonical_path: String,
    decoders: Vec<String>,
    aggregate: DecoderAbAggregate,
    gates: DecoderAbGates,
    cases: Vec<DecoderAbCase>,
}

#[derive(Debug, Default, Serialize)]
struct DecoderAbAggregate {
    num_cases: usize,
    avg_direct_coherence: Option<f32>,
    direct_hallucination_rate: Option<f32>,
    avg_direct_semantic_drift: Option<f32>,
    avg_structured_confidence: Option<f32>,
    avg_structured_intensity: Option<f32>,
    avg_structured_validity: Option<f32>,
    structured_required_role_rate: Option<f32>,
    avg_mamba_coherence: Option<f32>,
    mamba_hallucination_rate: Option<f32>,
    avg_mamba_semantic_drift: Option<f32>,
}

#[derive(Debug, Serialize)]
struct DecoderAbGates {
    passed: bool,
    failures: Vec<DecoderAbGateFailure>,
}

#[derive(Debug, Serialize)]
struct DecoderAbGateFailure {
    metric: String,
    observed: f32,
    threshold: f32,
}

#[derive(Debug, Serialize)]
struct DecoderAbCase {
    case_index: usize,
    case_id: String,
    category: String,
    target_text: String,
    direct: Option<DirectCase>,
    structured: Option<StructuredCase>,
    mamba: Option<MambaCase>,
}

#[derive(Debug, Serialize)]
struct StructuredCase {
    evidence_level: String,
    readout: StructuredReadoutReport,
    role_count: usize,
    required_roles_present: usize,
    has_required_roles: bool,
    validity: f32,
}

#[derive(Debug, Serialize)]
struct StructuredReadoutReport {
    decoder: String,
    intent: String,
    roles: Vec<StructuredRoleFill>,
    intensity: f32,
    confidence: f32,
    surface: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    molecule: Option<symthaea_core::hdc::universal_semantics::SemanticMolecule>,
}

#[derive(Debug, Serialize)]
struct DirectCase {
    evidence_level: String,
    text: String,
    token_ids: Vec<u32>,
    num_tokens: usize,
    final_coherence: f32,
    hallucination_flag: bool,
    eos_terminated: bool,
    semantic_drift: Option<f32>,
}

#[derive(Debug, Serialize)]
struct MambaCase {
    evidence_level: String,
    text: String,
    token_ids: Vec<u32>,
    num_tokens: usize,
    final_coherence: f32,
    hallucination_flag: bool,
    eos_terminated: bool,
    semantic_pe: Option<f32>,
    semantic_drift: Option<f32>,
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn".parse().unwrap()),
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
    let mut direct_generator = if wants(&opts, BrocaDecoderKind::Direct) {
        Some(load_direct_generator(&opts, &genesis)?)
    } else {
        None
    };
    if let Some(generator) = direct_generator.as_mut() {
        generator.config_mut().gating.base_max_tokens = opts.max_gen_tokens;
        generator.config_mut().decoder_kind = BrocaDecoderKind::Direct;
        generator.config_mut().bypass_gating = true;
        generator.config_mut().enable_semantic_veto = false;
        generator.config_mut().enable_coherence_feedback = false;
    }
    let structured_decoder =
        wants(&opts, BrocaDecoderKind::Structured).then(|| StructuredDecoder::new(&genesis));
    let mut mamba_generator = if wants(&opts, BrocaDecoderKind::Mamba) {
        Some(load_mamba_generator(&opts, &genesis)?)
    } else {
        None
    };

    let cases = limited_cases(&dataset, opts.eval_limit);
    let mut case_results = Vec::with_capacity(cases.len());
    for (case_index, case) in cases.iter().enumerate() {
        let channels = case_channels(case);

        let target_thought = direct_generator
            .as_ref()
            .map(|g| g.encoder().encode(&channels))
            .unwrap_or_else(|| ContinuousHV::zero(symthaea_core::hdc::HDC_DIMENSION));

        let direct = match direct_generator.as_mut() {
            Some(generator) => {
                let res = generator.generate(&channels);
                let token_embeddings = generator.controller().token_embeddings();
                let drift =
                    calculate_semantic_drift(&res.token_ids, &target_thought, token_embeddings);
                Some(direct_case(res, Some(drift)))
            }
            None => None,
        };
        let structured = structured_decoder.as_ref().map(|decoder| {
            let readout = decoder.decode(&channels);
            structured_case(readout, opts.include_structured_molecule)
        });

        let mamba = match mamba_generator.as_mut() {
            Some(generator) => {
                let result = generator.generate(&channels);
                let drift = direct_generator.as_ref().map(|dg| {
                    calculate_semantic_drift(
                        &result.token_ids,
                        &target_thought,
                        dg.controller().token_embeddings(),
                    )
                });
                Some(mamba_case(result, drift))
            }
            None => None,
        };
        case_results.push(DecoderAbCase {
            case_index,
            case_id: case.id.clone(),
            category: case.category.clone(),
            target_text: case.target_text.clone(),
            direct,
            structured,
            mamba,
        });
    }

    let aggregate = aggregate(&case_results);
    let gates = decoder_gates(&aggregate, &opts);
    let passed = gates.passed;
    let report = DecoderAbReport {
        schema_version: 1,
        evidence_level: "measured".to_string(),
        checkpoint_path: opts.checkpoint_path,
        mamba_projection_checkpoint: opts.mamba_projection_checkpoint,
        canonical_path: opts.canonical_path,
        decoders: opts
            .decoders
            .iter()
            .map(|decoder| decoder.as_str().to_string())
            .collect(),
        aggregate,
        gates,
        cases: case_results,
    };

    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = opts.json_out {
        std::fs::write(path, json)?;
    } else {
        println!("{json}");
    }
    if opts.fail_on_gate && !passed {
        anyhow::bail!("decoder A/B gates failed");
    }
    Ok(())
}

fn calculate_semantic_drift(
    token_ids: &[u32],
    target_thought: &ContinuousHV,
    token_embeddings: &[ContinuousHV],
) -> f32 {
    if token_ids.is_empty() {
        return 1.0;
    }
    let mut active_refs = Vec::with_capacity(token_ids.len());
    for &id in token_ids {
        if let Some(vector) = token_embeddings.get(id as usize) {
            active_refs.push(vector);
        }
    }
    if active_refs.is_empty() {
        return 1.0;
    }
    let utterance_centroid = ContinuousHV::bundle(&active_refs).normalize();
    let alignment = utterance_centroid
        .similarity(target_thought)
        .clamp(-1.0, 1.0);
    1.0 - alignment
}

fn load_direct_generator(opts: &Options, genesis: &GenesisSeed) -> Result<BrocaGenerator> {
    if let Some(path) = opts.checkpoint_path.as_deref() {
        let (generator, _, _, _) = if opts.allow_checkpoint_recovery {
            BrocaGenerator::from_checkpoint_allow_checksum_mismatch(path, genesis)
        } else {
            BrocaGenerator::from_checkpoint(path, genesis)
        }
        .with_context(|| format!("loading direct Broca checkpoint {path}"))?;
        Ok(generator)
    } else {
        let mut config = BrocaConfig::default();
        config.decoder_kind = BrocaDecoderKind::Direct;
        config.gating.base_max_tokens = opts.max_gen_tokens;
        Ok(BrocaGenerator::new_4k(genesis, config))
    }
}

#[cfg(feature = "mamba-cpu")]
fn load_mamba_generator(opts: &Options, genesis: &GenesisSeed) -> Result<LiquidMambaGenerator> {
    let path = opts
        .mamba_projection_checkpoint
        .as_deref()
        .context("--mamba-projection-checkpoint is required for mamba decoder")?;
    let checkpoint = if opts.allow_checkpoint_recovery {
        ProjectionCheckpoint::load_from_file_allow_checksum_mismatch(path)
    } else {
        ProjectionCheckpoint::load_from_file(path)
    }
    .with_context(|| format!("loading Mamba projection checkpoint {path}"))?;
    let mut config = LiquidMambaConfig {
        max_tokens: opts.max_gen_tokens,
        enable_consciousness_gating: false,
        enable_veto: false,
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
    let mut generator = LiquidMambaGenerator::new(genesis, config)?;
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
    Ok(generator)
}

#[cfg(not(feature = "mamba-cpu"))]
fn load_mamba_generator(_opts: &Options, _genesis: &GenesisSeed) -> Result<()> {
    anyhow::bail!("mamba decoder requires the mamba-cpu or mamba feature")
}

fn direct_case(result: GenerationResult, semantic_drift: Option<f32>) -> DirectCase {
    DirectCase {
        evidence_level: "measured".to_string(),
        text: result.text,
        token_ids: result.token_ids,
        num_tokens: result.num_tokens,
        final_coherence: result.final_coherence,
        hallucination_flag: result.hallucination_flag,
        eos_terminated: result.eos_terminated,
        semantic_drift,
    }
}

#[cfg(feature = "mamba-cpu")]
fn mamba_case(result: GenerationResult, semantic_drift: Option<f32>) -> MambaCase {
    MambaCase {
        evidence_level: "measured".to_string(),
        text: result.text,
        token_ids: result.token_ids,
        num_tokens: result.num_tokens,
        final_coherence: result.final_coherence,
        hallucination_flag: result.hallucination_flag,
        eos_terminated: result.eos_terminated,
        semantic_pe: Some(result.semantic_pe),
        semantic_drift,
    }
}

#[cfg(not(feature = "mamba-cpu"))]
fn mamba_case(_result: GenerationResult, _semantic_drift: Option<f32>) -> MambaCase {
    unreachable!("mamba result cannot be produced without mamba-cpu")
}

fn structured_case(readout: StructuredReadout, include_molecule: bool) -> StructuredCase {
    const REQUIRED_ROLES: &[&str] = &["AGENT", "ACTION", "PATIENT", "PREDICATE", "EVALUATOR"];

    let required_roles_present = REQUIRED_ROLES
        .iter()
        .filter(|role| readout.roles.iter().any(|fill| fill.role == **role))
        .count();
    let required_coverage = required_roles_present as f32 / REQUIRED_ROLES.len() as f32;
    let confidence = readout.confidence.clamp(0.0, 1.0);
    let intensity_present = if readout.intensity.is_finite() && readout.intensity > 0.0 {
        1.0
    } else {
        0.0
    };
    let validity =
        (0.6 * required_coverage + 0.3 * confidence + 0.1 * intensity_present).clamp(0.0, 1.0);
    let readout = StructuredReadoutReport {
        decoder: readout.decoder,
        intent: readout.intent,
        roles: readout.roles,
        intensity: readout.intensity,
        confidence: readout.confidence,
        surface: readout.surface,
        molecule: include_molecule.then_some(readout.molecule),
    };
    StructuredCase {
        evidence_level: "measured".to_string(),
        role_count: readout.roles.len(),
        required_roles_present,
        has_required_roles: required_roles_present == REQUIRED_ROLES.len(),
        validity,
        readout,
    }
}

fn aggregate(cases: &[DecoderAbCase]) -> DecoderAbAggregate {
    DecoderAbAggregate {
        num_cases: cases.len(),
        avg_direct_coherence: avg_direct(cases, |case| case.final_coherence),
        direct_hallucination_rate: rate_direct(cases, |case| case.hallucination_flag),
        avg_direct_semantic_drift: avg_direct(cases, |case| case.semantic_drift.unwrap_or(0.0)),
        avg_structured_confidence: avg_structured(cases, |case| case.readout.confidence),
        avg_structured_intensity: avg_structured(cases, |case| case.readout.intensity),
        avg_structured_validity: avg_structured(cases, |case| case.validity),
        structured_required_role_rate: rate_structured(cases, |case| case.has_required_roles),
        avg_mamba_coherence: avg_mamba(cases, |case| case.final_coherence),
        mamba_hallucination_rate: rate_mamba(cases, |case| case.hallucination_flag),
        avg_mamba_semantic_drift: avg_mamba(cases, |case| case.semantic_drift.unwrap_or(0.0)),
    }
}

fn decoder_gates(aggregate: &DecoderAbAggregate, opts: &Options) -> DecoderAbGates {
    let mut failures = Vec::new();
    check_max(
        &mut failures,
        "avg_direct_semantic_drift",
        aggregate.avg_direct_semantic_drift,
        opts.max_direct_drift,
    );
    check_max(
        &mut failures,
        "avg_mamba_semantic_drift",
        aggregate.avg_mamba_semantic_drift,
        opts.max_mamba_drift,
    );
    check_max(
        &mut failures,
        "direct_hallucination_rate",
        aggregate.direct_hallucination_rate,
        opts.max_hallucination_rate,
    );
    check_max(
        &mut failures,
        "mamba_hallucination_rate",
        aggregate.mamba_hallucination_rate,
        opts.max_hallucination_rate,
    );
    check_min(
        &mut failures,
        "avg_structured_validity",
        aggregate.avg_structured_validity,
        opts.min_structured_validity,
    );
    check_min(
        &mut failures,
        "structured_required_role_rate",
        aggregate.structured_required_role_rate,
        opts.min_structured_required_role_rate,
    );
    DecoderAbGates {
        passed: failures.is_empty(),
        failures,
    }
}

fn check_min(
    failures: &mut Vec<DecoderAbGateFailure>,
    metric: &str,
    observed: Option<f32>,
    threshold: f32,
) {
    if let Some(observed) = observed {
        if observed < threshold {
            failures.push(DecoderAbGateFailure {
                metric: metric.to_string(),
                observed,
                threshold,
            });
        }
    }
}

fn check_max(
    failures: &mut Vec<DecoderAbGateFailure>,
    metric: &str,
    observed: Option<f32>,
    threshold: f32,
) {
    if let Some(observed) = observed {
        if observed > threshold {
            failures.push(DecoderAbGateFailure {
                metric: metric.to_string(),
                observed,
                threshold,
            });
        }
    }
}

fn avg_direct<F>(cases: &[DecoderAbCase], mut f: F) -> Option<f32>
where
    F: FnMut(&DirectCase) -> f32,
{
    avg(cases
        .iter()
        .filter_map(|case| case.direct.as_ref())
        .map(|c| f(c)))
}

fn rate_direct<F>(cases: &[DecoderAbCase], mut f: F) -> Option<f32>
where
    F: FnMut(&DirectCase) -> bool,
{
    rate(
        cases
            .iter()
            .filter_map(|case| case.direct.as_ref())
            .map(|c| f(c)),
    )
}

fn avg_structured<F>(cases: &[DecoderAbCase], mut f: F) -> Option<f32>
where
    F: FnMut(&StructuredCase) -> f32,
{
    avg(cases
        .iter()
        .filter_map(|case| case.structured.as_ref())
        .map(|c| f(c)))
}

fn rate_structured<F>(cases: &[DecoderAbCase], mut f: F) -> Option<f32>
where
    F: FnMut(&StructuredCase) -> bool,
{
    rate(
        cases
            .iter()
            .filter_map(|case| case.structured.as_ref())
            .map(|c| f(c)),
    )
}

fn avg_mamba<F>(cases: &[DecoderAbCase], mut f: F) -> Option<f32>
where
    F: FnMut(&MambaCase) -> f32,
{
    avg(cases
        .iter()
        .filter_map(|case| case.mamba.as_ref())
        .map(|c| f(c)))
}

fn rate_mamba<F>(cases: &[DecoderAbCase], mut f: F) -> Option<f32>
where
    F: FnMut(&MambaCase) -> bool,
{
    rate(
        cases
            .iter()
            .filter_map(|case| case.mamba.as_ref())
            .map(|c| f(c)),
    )
}

fn avg<I>(values: I) -> Option<f32>
where
    I: Iterator<Item = f32>,
{
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        sum += value;
        count += 1;
    }
    (count > 0).then_some(sum / count as f32)
}

fn rate<I>(values: I) -> Option<f32>
where
    I: Iterator<Item = bool>,
{
    let mut count = 0usize;
    let mut yes = 0usize;
    for value in values {
        count += 1;
        if value {
            yes += 1;
        }
    }
    (count > 0).then_some(yes as f32 / count as f32)
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
    }
    .to_thought_channels()
}

fn wants(opts: &Options, decoder: BrocaDecoderKind) -> bool {
    opts.decoders.contains(&decoder)
        || (decoder == BrocaDecoderKind::Direct
            && opts.decoders.contains(&BrocaDecoderKind::Hybrid))
        || (decoder == BrocaDecoderKind::Structured
            && opts.decoders.contains(&BrocaDecoderKind::Hybrid))
}

fn parse_args(args: &[String]) -> Result<Options> {
    let mut opts = Options {
        checkpoint_path: None,
        mamba_projection_checkpoint: None,
        canonical_path: "crates/symthaea-broca/tests/fixtures/eval-canonical-v1.jsonl".into(),
        json_out: None,
        eval_limit: 0,
        max_gen_tokens: 32,
        genesis_phrase: "symthaea luminous dynamics".into(),
        allow_checkpoint_recovery: false,
        decoders: vec![BrocaDecoderKind::Direct, BrocaDecoderKind::Structured],
        fail_on_gate: false,
        max_direct_drift: 1.10,
        max_mamba_drift: 1.10,
        max_hallucination_rate: 1.0,
        min_structured_validity: 0.5,
        min_structured_required_role_rate: 1.0,
        include_structured_molecule: false,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--checkpoint" => {
                i += 1;
                opts.checkpoint_path = Some(value(args, i, "--checkpoint")?.to_string());
            }
            "--mamba-projection-checkpoint" => {
                i += 1;
                opts.mamba_projection_checkpoint =
                    Some(value(args, i, "--mamba-projection-checkpoint")?.to_string());
            }
            "--canonical-eval" => {
                i += 1;
                opts.canonical_path = value(args, i, "--canonical-eval")?.to_string();
            }
            "--json-out" => {
                i += 1;
                opts.json_out = Some(value(args, i, "--json-out")?.to_string());
            }
            "--eval-limit" => {
                i += 1;
                opts.eval_limit = value(args, i, "--eval-limit")?.parse()?;
            }
            "--max-r#gen-tokens" => {
                i += 1;
                opts.max_gen_tokens = value(args, i, "--max-r#gen-tokens")?.parse()?;
            }
            "--genesis" => {
                i += 1;
                opts.genesis_phrase = value(args, i, "--genesis")?.to_string();
            }
            "--decoder" | "--decoders" => {
                i += 1;
                opts.decoders = parse_decoders(value(args, i, "--decoder")?)?;
            }
            "--allow-checkpoint-recovery" => {
                opts.allow_checkpoint_recovery = true;
            }
            "--include-structured-molecule" => {
                opts.include_structured_molecule = true;
            }
            "--fail-on-gate" => {
                opts.fail_on_gate = true;
            }
            "--max-direct-drift" => {
                i += 1;
                opts.max_direct_drift = value(args, i, "--max-direct-drift")?.parse()?;
            }
            "--max-mamba-drift" => {
                i += 1;
                opts.max_mamba_drift = value(args, i, "--max-mamba-drift")?.parse()?;
            }
            "--max-hallucination-rate" => {
                i += 1;
                opts.max_hallucination_rate =
                    value(args, i, "--max-hallucination-rate")?.parse()?;
            }
            "--min-structured-validity" => {
                i += 1;
                opts.min_structured_validity =
                    value(args, i, "--min-structured-validity")?.parse()?;
            }
            "--min-structured-required-role-rate" => {
                i += 1;
                opts.min_structured_required_role_rate =
                    value(args, i, "--min-structured-required-role-rate")?.parse()?;
            }
            "--help" | "-h" => {
                print_usage();
                process::exit(0);
            }
            other => anyhow::bail!("unknown argument {other}"),
        }
        i += 1;
    }

    if opts.max_gen_tokens == 0 {
        anyhow::bail!("--max-r#gen-tokens must be greater than zero");
    }
    if !(0.0..=2.0).contains(&opts.max_direct_drift) || !(0.0..=2.0).contains(&opts.max_mamba_drift)
    {
        anyhow::bail!("semantic drift thresholds must be between 0.0 and 2.0");
    }
    if !(0.0..=1.0).contains(&opts.max_hallucination_rate) {
        anyhow::bail!("--max-hallucination-rate must be between 0.0 and 1.0");
    }
    if !(0.0..=1.0).contains(&opts.min_structured_validity)
        || !(0.0..=1.0).contains(&opts.min_structured_required_role_rate)
    {
        anyhow::bail!("structured thresholds must be between 0.0 and 1.0");
    }
    Ok(opts)
}

fn parse_decoders(value: &str) -> Result<Vec<BrocaDecoderKind>> {
    let mut decoders = Vec::new();
    for part in value.split(',') {
        let decoder = BrocaDecoderKind::parse(part)
            .with_context(|| format!("unknown decoder kind {part:?}"))?;
        if !decoders.contains(&decoder) {
            decoders.push(decoder);
        }
    }
    if decoders.is_empty() {
        anyhow::bail!("at least one decoder is required");
    }
    Ok(decoders)
}

fn value<'a>(args: &'a [String], index: usize, flag: &str) -> Result<&'a str> {
    args.get(index)
        .map(String::as_str)
        .with_context(|| format!("{flag} requires a value"))
}

fn print_usage() {
    eprintln!(
        "Usage: broca-decoder-ab [--checkpoint PATH] [--mamba-projection-checkpoint PATH] [--decoder direct,structured,mamba,hybrid] [--canonical-eval PATH] [--eval-limit N] [--max-r#gen-tokens N] [--json-out PATH] [--fail-on-gate] [--include-structured-molecule] [--max-direct-drift F] [--max-mamba-drift F] [--max-hallucination-rate F] [--min-structured-validity F] [--min-structured-required-role-rate F]"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_decoder_aliases() {
        let decoders = parse_decoders("direct,semantic").unwrap();
        assert_eq!(
            decoders,
            vec![BrocaDecoderKind::Direct, BrocaDecoderKind::Structured]
        );
    }

    #[test]
    fn aggregate_handles_missing_decoders() {
        let report = aggregate(&[DecoderAbCase {
            case_index: 0,
            case_id: "x".into(),
            category: "intent".into(),
            target_text: "hello".into(),
            direct: None,
            structured: None,
            mamba: None,
        }]);
        assert_eq!(report.num_cases, 1);
        assert!(report.avg_direct_coherence.is_none());
    }
}
