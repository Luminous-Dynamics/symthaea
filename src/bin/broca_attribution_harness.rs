// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic attribution harness for Broca/vocal-tract speech planning.
//!
//! This is a mechanism-attribution instrument, not a naturalness benchmark.
//! It compares a fixed phoneme schedule under isolated/composed prosody and
//! feedback conditions, then reports frame-level acoustic proxies and FEP telemetry.

use std::{collections::BTreeMap, fs, path::PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;
use symthaea::voice::broca_realization::{
    BrocaFramePosition, BrocaProsodyAdapter,
};
use symthaea::voice::vocal_tract_encoder::VoiceCognitiveState;
use symthaea_core::genesis::GenesisSeed;
use symthaea_broca::{SpeechPlan, StructuredDecoder, ThoughtChannels};
use symthaea_vocal_tract::{
    fep::VocalTractObservation,
    pipeline::{Intonation, PitchAccent, ProsodyContext, VocalTractPipeline},
};

const SAMPLE_RATE: u32 = 24_000;
const FRAME_RATE: u32 = 200;
const DT: f32 = 1.0 / FRAME_RATE as f32;
const FRAMES_PER_PHONEME: usize = 16;
const PHONEMES: [&str; 4] = ["B", "AE1", "T", "AH0"];

#[derive(Debug, Clone, Copy, Serialize)]
enum AttributionCondition {
    Neutral,
    VocalTractOnly,
    BrocaOnly,
    Composed,
    ComposedWithFeedback,
}

impl AttributionCondition {
    fn feedback_enabled(self) -> bool {
        matches!(self, Self::ComposedWithFeedback)
    }

    fn label(self) -> &'static str {
        match self {
            Self::Neutral => "neutral",
            Self::VocalTractOnly => "vocal-tract-only",
            Self::BrocaOnly => "broca-only",
            Self::Composed => "composed",
            Self::ComposedWithFeedback => "composed-feedback",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct FrameProxy {
    mean_f0: f32,
    f0_min: f32,
    f0_max: f32,
    f0_span: f32,
    mean_energy: f32,
    energy_span: f32,
    mean_f1: f32,
    mean_f2: f32,
    total_f0_variation: f32,
    total_formant_variation: f32,
    nonfinite_fields: usize,
    frames: usize,
}

#[derive(Debug, Clone, Serialize)]
struct FepTelemetry {
    events: usize,
    prediction_errors: Vec<f64>,
    selected_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ConditionResult {
    condition: String,
    feedback_enabled: bool,
    plan_surface: String,
    plan_rate_target: f32,
    rate_target_applied_by_pipeline: bool,
    frame_proxy: FrameProxy,
    fep: FepTelemetry,
    exact_repeat: bool,
    repeat_max_abs_delta: f32,
}

#[derive(Debug, Serialize)]
struct AttributionReport {
    schema_version: u32,
    evidence_level: &'static str,
    seed_phrase: &'static str,
    sample_rate: u32,
    frame_rate: u32,
    dt_seconds: f32,
    phonemes: Vec<&'static str>,
    frames_per_phoneme: usize,
    semantic_stage: &'static str,
    semantic_stage_note: &'static str,
    conditions: Vec<ConditionResult>,
    deltas_from_neutral: BTreeMap<String, FrameProxyDelta>,
    limitations: Vec<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
struct FrameProxyDelta {
    mean_f0_delta: f32,
    f0_span_delta: f32,
    mean_energy_delta: f32,
    energy_span_delta: f32,
    mean_f1_delta: f32,
    mean_f2_delta: f32,
    f0_variation_delta: f32,
    formant_variation_delta: f32,
}

#[derive(Debug, Clone, Copy)]
struct FrameCapture {
    f0: f32,
    energy: f32,
    f1: f32,
    f2: f32,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("broca-attribution-harness: {error:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let json_out = parse_json_out();
    let seed_phrase = "broca-attribution-harness-v1";
    let genesis = GenesisSeed::from_phrase(seed_phrase);
    let decoder = StructuredDecoder::new(&genesis);

    let mut channels = ThoughtChannels::with_intent(3);
    channels.set_epistemic(0.0);
    let readout = decoder.decode(&channels);
    let plan = SpeechPlan::from_readout(&channels, &readout);
    plan.validate().context("Broca speech plan validation")?;

    let conditions = [
        AttributionCondition::Neutral,
        AttributionCondition::VocalTractOnly,
        AttributionCondition::BrocaOnly,
        AttributionCondition::Composed,
        AttributionCondition::ComposedWithFeedback,
    ];

    let results = conditions
        .into_iter()
        .map(|condition| run_condition(&genesis, &plan, condition))
        .collect::<Result<Vec<_>>>()?;

    let neutral = results
        .iter()
        .find(|result| result.condition == "neutral")
        .context("neutral condition missing")?;

    let mut deltas_from_neutral = BTreeMap::new();
    for result in &results {
        deltas_from_neutral.insert(
            result.condition.clone(),
            proxy_delta(&result.frame_proxy, &neutral.frame_proxy),
        );
    }

    let report = AttributionReport {
        schema_version: 1,
        evidence_level: "deterministic-attribution-frame-proxy",
        seed_phrase,
        sample_rate: SAMPLE_RATE,
        frame_rate: FRAME_RATE,
        dt_seconds: DT,
        phonemes: PHONEMES.to_vec(),
        frames_per_phoneme: FRAMES_PER_PHONEME,
        semantic_stage: "SpeechPlan",
        semantic_stage_note:
            "This harness preserves the exact SpeechPlan surface but does not claim semantic-delivery success from frame metrics.",
        conditions: results,
        deltas_from_neutral,
        limitations: vec![
            "Frame proxies are mechanism-attribution measures, not human naturalness scores.",
            "Absolute intelligibility is not measured here; use the existing semantic-delivery and waveform/ASR gates separately.",
            "Broca rate_target is recorded but is not consumed by VocalTractPipeline timing in this harness.",
            "Feedback-enabled runs use a fixed synthetic VocalTractObservation and therefore test controller coupling, not real auditory self-hearing.",
            "The composed condition uses an explicit field-level precedence policy documented by compose_context().",
        ],
    };

    write_report(json_out.as_ref(), &report)?;
    println!(
        "broca attribution harness: {} conditions captured with exact deterministic repeat checks",
        report.conditions.len()
    );
    Ok(())
}

fn run_condition(
    genesis: &GenesisSeed,
    plan: &SpeechPlan,
    condition: AttributionCondition,
) -> Result<ConditionResult> {
    let first = capture(genesis, plan, condition)?;
    let second = capture(genesis, plan, condition)?;
    let (exact_repeat, repeat_max_abs_delta) = compare_captures(&first.0, &second.0);

    Ok(ConditionResult {
        condition: condition.label().to_string(),
        feedback_enabled: condition.feedback_enabled(),
        plan_surface: plan.grounding_surface(),
        plan_rate_target: plan.prosody.rate,
        rate_target_applied_by_pipeline: false,
        frame_proxy: summarize(&first.0),
        fep: first.1,
        exact_repeat,
        repeat_max_abs_delta,
    })
}

fn capture(
    genesis: &GenesisSeed,
    plan: &SpeechPlan,
    condition: AttributionCondition,
) -> Result<(Vec<FrameCapture>, FepTelemetry)> {
    let mut pipeline = VocalTractPipeline::new(genesis);
    let cognitive_state = VoiceCognitiveState::default();
    let adapter = BrocaProsodyAdapter::new(*plan);
    let mut frames = Vec::with_capacity(PHONEMES.len() * FRAMES_PER_PHONEME);
    let mut prediction_errors = Vec::new();
    let mut selected_actions = Vec::new();
    let feedback = VocalTractObservation {
        articulation_score: 1.0,
        formant_accuracy: 1.0,
        pitch_stability: 1.0,
        coarticulation_smoothness: 1.0,
        duration_accuracy: 1.0,
        energy_consistency: 1.0,
    };

    for (phoneme_index, phoneme) in PHONEMES.iter().enumerate() {
        let stress = if phoneme.ends_with('1') { 1 } else { 0 };
        let syllable_index = if phoneme_index < 3 { 0 } else { 1 };
        let is_syllable_onset = phoneme_index == 0 || phoneme_index == 3;

        for frame_index in 0..FRAMES_PER_PHONEME {
            let global_index = phoneme_index * FRAMES_PER_PHONEME + frame_index;
            let utterance_progress =
                global_index as f32 / (PHONEMES.len() * FRAMES_PER_PHONEME - 1) as f32;
            let phoneme_progress = frame_index as f32 / (FRAMES_PER_PHONEME - 1) as f32;
            let syllable_progress = if syllable_index == 0 {
                ((phoneme_index * FRAMES_PER_PHONEME + frame_index) as f32
                    / (3 * FRAMES_PER_PHONEME - 1) as f32)
                    .clamp(0.0, 1.0)
            } else {
                phoneme_progress;
            };

            let legacy = legacy_context(
                utterance_progress,
                phoneme_progress,
                stress,
                syllable_progress,
                is_syllable_onset,
            );
            let broca = adapter.context(
                120.0,
                BrocaFramePosition {
                    utterance_progress,
                    phoneme_progress,
                    stress,
                    phrase_index: 0,
                    phrase_progress: utterance_progress,
                    is_focus: false,
                    is_syllable_onset,
                    syllable_progress,
                    prev_source_type: None,
                    next_source_type: None,
                },
            );
            let context = match condition {
                AttributionCondition::Neutral => neutral_context(
                    utterance_progress,
                    phoneme_progress,
                    stress,
                    syllable_progress,
                    is_syllable_onset,
                ),
                AttributionCondition::VocalTractOnly => legacy,
                AttributionCondition::BrocaOnly => broca,
                AttributionCondition::Composed | AttributionCondition::ComposedWithFeedback => {
                    compose_context(legacy, broca)
                }
            };

            let metrics = if condition.feedback_enabled() {
                Some(&feedback)
            } else {
                None
            };
            let frame = pipeline.tick_with_prosody(
                &cognitive_state,
                metrics,
                DT,
                Some(phoneme.trim_end_matches(['0', '1', '2'])),
                &context,
            );

            if !frame.f0.is_finite()
                || !frame.energy.is_finite()
                || !frame.f1.is_finite()
                || !frame.f2.is_finite()
            {
                frames.push(FrameCapture {
                    f0: frame.f0,
                    energy: frame.energy,
                    f1: frame.f1,
                    f2: frame.f2,
                });
            } else {
                frames.push(FrameCapture {
                    f0: frame.f0,
                    energy: frame.energy,
                    f1: frame.f1,
                    f2: frame.f2,
                });
            }

            if global_index % 20 == 19 {
                if let Some(result) = pipeline.last_fep_result() {
                    prediction_errors.push(result.prediction_error);
                    selected_actions.push(format!("{:?}", result.action));
                }
            }
        }
    }

    Ok((
        frames,
        FepTelemetry {
            events: prediction_errors.len(),
            prediction_errors,
            selected_actions,
        },
    ))
}

fn neutral_context(
    utterance_progress: f32,
    phoneme_progress: f32,
    stress: u8,
    syllable_progress: f32,
    is_syllable_onset: bool,
) -> ProsodyContext {
    legacy_context(
        utterance_progress,
        phoneme_progress,
        stress,
        syllable_progress,
        is_syllable_onset,
    )
}

fn legacy_context(
    utterance_progress: f32,
    phoneme_progress: f32,
    stress: u8,
    syllable_progress: f32,
    is_syllable_onset: bool,
) -> ProsodyContext {
    ProsodyContext {
        utterance_progress,
        phoneme_progress,
        stress,
        base_f0: 120.0,
        arousal: 0.5,
        intonation: Intonation::Statement,
        phrase_index: 0,
        phrase_progress: utterance_progress,
        is_focus: false,
        pitch_accent: if stress == 1 {
            PitchAccent::High
        } else {
            PitchAccent::None
        },
        is_syllable_onset,
        syllable_progress,
        prev_source_type: None,
        next_source_type: None,
    }
}

fn compose_context(legacy: ProsodyContext, broca: ProsodyContext) -> ProsodyContext {
    // Explicit attribution policy:
    // - realization timing and syllable metadata remain scheduler/legacy-owned;
    // - Broca owns intonation, arousal, and pitch-accent intent;
    // - speaker base F0 remains legacy-owned;
    // - focus remains scheduler/phonological-owned.
    ProsodyContext {
        intonation: broca.intonation,
        arousal: broca.arousal,
        pitch_accent: broca.pitch_accent,
        ..legacy
    }
}

fn summarize(frames: &[FrameCapture]) -> FrameProxy {
    let finite = frames.iter().filter(|frame| {
        frame.f0.is_finite()
            && frame.energy.is_finite()
            && frame.f1.is_finite()
            && frame.f2.is_finite()
    });
    let values: Vec<_> = finite.collect();

    let f0_min = values.iter().map(|frame| frame.f0).fold(f32::INFINITY, f32::min);
    let f0_max = values.iter().map(|frame| frame.f0).fold(f32::NEG_INFINITY, f32::max);
    let energy_min = values.iter().map(|frame| frame.energy).fold(f32::INFINITY, f32::min);
    let energy_max = values.iter().map(|frame| frame.energy).fold(f32::NEG_INFINITY, f32::max);

    FrameProxy {
        mean_f0: mean(values.iter().map(|frame| frame.f0)),
        f0_min: if values.is_empty() { 0.0 } else { f0_min },
        f0_max: if values.is_empty() { 0.0 } else { f0_max },
        f0_span: if values.is_empty() { 0.0 } else { f0_max - f0_min },
        mean_energy: mean(values.iter().map(|frame| frame.energy)),
        energy_span: if values.is_empty() { 0.0 } else { energy_max - energy_min },
        mean_f1: mean(values.iter().map(|frame| frame.f1)),
        mean_f2: mean(values.iter().map(|frame| frame.f2)),
        total_f0_variation: total_variation(values.iter().map(|frame| frame.f0)),
        total_formant_variation: total_variation_pairs(
            values.iter().map(|frame| (frame.f1, frame.f2)),
        ),
        nonfinite_fields: frames.iter().map(|frame| {
            usize::from(!frame.f0.is_finite())
                + usize::from(!frame.energy.is_finite())
                + usize::from(!frame.f1.is_finite())
                + usize::from(!frame.f2.is_finite())
        }).sum(),
        frames: frames.len(),
    }
}

fn mean<I>(values: I) -> f32
where
    I: Iterator<Item = f32>,
{
    let values: Vec<f32> = values.collect();
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn total_variation<I>(values: I) -> f32
where
    I: Iterator<Item = f32>,
{
    let values: Vec<f32> = values.collect();
    values.windows(2).map(|pair| (pair[1] - pair[0]).abs()).sum()
}

fn total_variation_pairs<I>(values: I) -> f32
where
    I: Iterator<Item = (f32, f32)>,
{
    let values: Vec<(f32, f32)> = values.collect();
    values
        .windows(2)
        .map(|pair| (pair[1].0 - pair[0].0).abs() + (pair[1].1 - pair[0].1).abs())
        .sum()
}

fn proxy_delta(candidate: &FrameProxy, baseline: &FrameProxy) -> FrameProxyDelta {
    FrameProxyDelta {
        mean_f0_delta: candidate.mean_f0 - baseline.mean_f0,
        f0_span_delta: candidate.f0_span - baseline.f0_span,
        mean_energy_delta: candidate.mean_energy - baseline.mean_energy,
        energy_span_delta: candidate.energy_span - baseline.energy_span,
        mean_f1_delta: candidate.mean_f1 - baseline.mean_f1,
        mean_f2_delta: candidate.mean_f2 - baseline.mean_f2,
        f0_variation_delta: candidate.total_f0_variation - baseline.total_f0_variation,
        formant_variation_delta: candidate.total_formant_variation - baseline.total_formant_variation,
    }
}

fn compare_captures(
    first: &[FrameCapture],
    second: &[FrameCapture],
) -> (bool, f32) {
    if first.len() != second.len() {
        return (false, f32::INFINITY);
    }
    let mut max_delta = 0.0;
    let mut exact = true;
    for (a, b) in first.iter().zip(second.iter()) {
        for delta in [
            (a.f0 - b.f0).abs(),
            (a.energy - b.energy).abs(),
            (a.f1 - b.f1).abs(),
            (a.f2 - b.f2).abs(),
        ] {
            if delta != 0.0 {
                exact = false;
            }
            if delta > max_delta {
                max_delta = delta;
            }
        }
    }
    (exact, max_delta)
}

fn parse_json_out() -> Option<PathBuf> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--json-out" {
            return args.next().map(PathBuf::from);
        }
    }
    None
}

fn write_report(path: Option<&PathBuf>, report: &AttributionReport) -> Result<()> {
    if let Some(path) = path {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_vec_pretty(report)?)?;
    }
    Ok(())
}
