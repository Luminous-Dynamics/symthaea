// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Preregistered source-candidate-002 compatibility sweep for `/a/` at 220 Hz.

use std::{
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde_json::{Value, json};
use symthaea_vocal_tract::{
    CANDIDATE_WAVEGUIDE_VERSION, CandidateBranchedWaveguideV2, CandidateRendererStems,
    CandidateWaveguideConfig, CardinalVowel, GlottalSourceParameters, IdentityAnatomy,
    IdentityPhysiology, PhonationRegime, PhysicalTractFrame, VowelAnchorDecoder,
};

const SAMPLE_RATE: u32 = 48_000;
const FRAME_RATE: f32 = 200.0;
const F0_HZ: f32 = 220.0;
const SWEEP_SECONDS: f32 = 1.1;
const FINAL_SECONDS: f32 = 3.0;

#[derive(Clone)]
struct Evaluation {
    parameters: GlottalSourceParameters,
    eligible: bool,
    score: f32,
    summary: Value,
}

fn save_wav(path: &Path, samples: &[f32]) -> Result<()> {
    let data_size = (samples.len() * 2) as u32;
    let mut file = File::create(path)?;
    file.write_all(b"RIFF")?;
    file.write_all(&(36 + data_size).to_le_bytes())?;
    file.write_all(b"WAVEfmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&SAMPLE_RATE.to_le_bytes())?;
    file.write_all(&(SAMPLE_RATE * 2).to_le_bytes())?;
    file.write_all(&2u16.to_le_bytes())?;
    file.write_all(&16u16.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;
    for &sample in samples {
        let quantized = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        file.write_all(&quantized.to_le_bytes())?;
    }
    Ok(())
}

fn rms(samples: &[f32]) -> f32 {
    (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len().max(1) as f32).sqrt()
}

fn listening_copy(samples: &[f32]) -> Vec<f32> {
    let level = rms(samples).max(1e-9);
    let peak = samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0, f32::max)
        .max(1e-9);
    let gain = (0.1 / level).min(0.95 / peak);
    samples.iter().map(|sample| sample * gain).collect()
}

fn goertzel_power(samples: &[f32], frequency_hz: f32) -> f32 {
    let omega = std::f32::consts::TAU * frequency_hz / SAMPLE_RATE as f32;
    let coefficient = 2.0 * omega.cos();
    let mut previous = 0.0;
    let mut previous_two = 0.0;
    for (index, &sample) in samples.iter().enumerate() {
        let window = 0.5
            - 0.5
                * (std::f32::consts::TAU * index as f32
                    / samples.len().saturating_sub(1).max(1) as f32)
                    .cos();
        let current = sample * window + coefficient * previous - previous_two;
        previous_two = previous;
        previous = current;
    }
    (previous * previous + previous_two * previous_two - coefficient * previous * previous_two)
        .max(0.0)
}

fn db_ratio(numerator: f32, denominator: f32) -> f32 {
    20.0 * (numerator.max(1e-12) / denominator.max(1e-12)).log10()
}

fn summarize(stems: &CandidateRendererStems) -> Value {
    let samples = &stems.final_output;
    let start = (0.55 * SAMPLE_RATE as f32) as usize;
    let end = (start + 16_384).min(samples.len());
    let steady = &samples[start.min(end)..end];
    let powers: Vec<_> = (1..=16)
        .map(|harmonic| goertzel_power(steady, F0_HZ * harmonic as f32).max(1e-18))
        .collect();
    let h1 = powers.first().copied().unwrap_or(1e-18);
    let harmonics_db: Vec<_> = powers
        .iter()
        .map(|power| 10.0 * (power / h1).log10())
        .collect();
    let peak = samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0, f32::max);
    let peak_delta = samples
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).abs())
        .fold(0.0, f32::max);
    let periodic_rms = rms(&stems.glottal_source);
    let aspiration_rms = rms(&stems.aspiration);
    json!({
        "rms_dbfs": 20.0 * rms(samples).max(1e-12).log10(),
        "peak_dbfs": 20.0 * peak.max(1e-12).log10(),
        "peak_sample_delta": peak_delta,
        "clipped_sample_fraction": samples.iter().filter(|sample| sample.abs() >= 1.0).count() as f32
            / samples.len().max(1) as f32,
        "all_finite": samples.iter().all(|sample| sample.is_finite()),
        "aspiration_to_periodic_db": db_ratio(aspiration_rms, periodic_rms),
        "harmonic_db_relative_h1": harmonics_db
    })
}

fn static_frames(frame: &PhysicalTractFrame, seconds: f32) -> Vec<PhysicalTractFrame> {
    vec![frame.clone(); (seconds * FRAME_RATE) as usize]
}

fn render(
    frame: &PhysicalTractFrame,
    parameters: GlottalSourceParameters,
    seconds: f32,
) -> Result<(CandidateRendererStems, usize)> {
    let mut config = CandidateWaveguideConfig::default();
    config.source_parameters_override = Some(parameters);
    let mut renderer = CandidateBranchedWaveguideV2::new(config);
    let stems = renderer
        .render_frames(&static_frames(frame, seconds), FRAME_RATE)
        .map_err(|error| anyhow::anyhow!(error))?;
    let unintended = renderer
        .diagnostics()
        .reflection_episodes
        .iter()
        .filter(|episode| !episode.intended_closure)
        .count();
    Ok((stems, unintended))
}

fn distance_outside(value: f32, low: f32, high: f32) -> f32 {
    if value < low {
        low - value
    } else if value > high {
        value - high
    } else {
        0.0
    }
}

fn evaluate(frame: &PhysicalTractFrame, parameters: GlottalSourceParameters) -> Result<Evaluation> {
    let (stems, unintended_reflections) = render(frame, parameters, SWEEP_SECONDS)?;
    let summary = summarize(&stems);
    let harmonics = summary["harmonic_db_relative_h1"]
        .as_array()
        .context("harmonic array")?;
    let support_indices = [2usize, 3, 4, 5, 10];
    let harmonic_support = support_indices
        .iter()
        .map(|&index| harmonics[index].as_f64().unwrap_or(-120.0) as f32)
        .sum::<f32>()
        / support_indices.len() as f32;
    let rms_dbfs = summary["rms_dbfs"].as_f64().unwrap_or(-120.0) as f32;
    let aspiration_db = summary["aspiration_to_periodic_db"]
        .as_f64()
        .unwrap_or(120.0) as f32;
    let peak_delta = summary["peak_sample_delta"].as_f64().unwrap_or(1.0) as f32;
    let clipped = summary["clipped_sample_fraction"].as_f64().unwrap_or(1.0) as f32;
    let finite = summary["all_finite"].as_bool().unwrap_or(false);
    let eligible = finite && clipped == 0.0 && peak_delta <= 0.012 && unintended_reflections == 0;
    let score = harmonic_support
        - 0.5 * (-42.0 - rms_dbfs).max(0.0)
        - 0.5 * distance_outside(aspiration_db, -55.0, -20.0)
        - 2000.0 * (peak_delta - 0.012).max(0.0);
    Ok(Evaluation {
        parameters,
        eligible,
        score,
        summary: json!({
            "signal": summary,
            "unintended_reflection_episodes": unintended_reflections,
            "harmonic_support_score_db": harmonic_support
        }),
    })
}

fn main() -> Result<()> {
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("audio_output/covered_domain_a220_candidate002_v1"));
    std::fs::create_dir_all(&output)?;

    let anatomy = IdentityAnatomy::velvet();
    let physiology = IdentityPhysiology::default();
    let tract = VowelAnchorDecoder::default()
        .physical_frame(CardinalVowel::A, &anatomy, &physiology, F0_HZ)
        .map_err(|error| anyhow::anyhow!(error))?;

    let mut evaluations = Vec::new();
    for open_quotient in [0.50, 0.56, 0.62] {
        for spectral_tilt_keep in [0.12, 0.20, 0.28] {
            for closing_bias in [0.78, 0.95, 1.12] {
                evaluations.push(evaluate(
                    &tract,
                    GlottalSourceParameters::new(
                        open_quotient,
                        1.05,
                        0.76,
                        spectral_tilt_keep,
                        0.004,
                        closing_bias,
                    ),
                )?);
            }
        }
    }
    let winner = evaluations
        .iter()
        .filter(|evaluation| evaluation.eligible)
        .max_by(|left, right| left.score.total_cmp(&right.score))
        .context("no eligible candidate")?
        .clone();

    let candidate001 = PhonationRegime::Modal.parameters();
    let (candidate001_stems, _) = render(&tract, candidate001, FINAL_SECONDS)?;
    let (candidate002_stems, unintended_reflections) =
        render(&tract, winner.parameters, FINAL_SECONDS)?;
    for (id, stems) in [
        ("joint_candidate_001", &candidate001_stems),
        ("source_candidate_002", &candidate002_stems),
    ] {
        save_wav(&output.join(format!("{id}.raw.wav")), &stems.final_output)?;
        save_wav(
            &output.join(format!("{id}.listening_-20dbfs.wav")),
            &listening_copy(&stems.final_output),
        )?;
    }
    save_wav(
        &output.join("source_candidate_002.glottal_source.wav"),
        &listening_copy(&candidate002_stems.glottal_source),
    )?;
    save_wav(
        &output.join("source_candidate_002.aspiration.wav"),
        &listening_copy(&candidate002_stems.aspiration),
    )?;

    let report = json!({
        "campaign_version": "symthaea.vowel-truth.modal-a-220-source-candidate002.v1",
        "status": "executed_diagnostic_only",
        "renderer": CANDIDATE_WAVEGUIDE_VERSION,
        "target": {"vowel":"a", "f0_hz":F0_HZ, "phonation_intent":"modal"},
        "tract": "tract_candidate_a_001_anchor_v1",
        "grid_size": evaluations.len(),
        "selection_rule": "preregistered candidate002_campaign.json",
        "winner": {
            "parameters": winner.parameters,
            "score": winner.score,
            "sweep_summary": winner.summary,
            "full_duration_summary": summarize(&candidate002_stems),
            "unintended_reflection_episodes": unintended_reflections
        },
        "candidate001_full_duration_summary": summarize(&candidate001_stems),
        "sweep": evaluations.iter().map(|evaluation| json!({
            "parameters": evaluation.parameters,
            "eligible": evaluation.eligible,
            "score": evaluation.score,
            "summary": evaluation.summary
        })).collect::<Vec<_>>(),
        "human_audio_accessed": false,
        "validation_or_holdout_accessed": false,
        "residual_model_enabled": false,
        "human_calibration_claim": false,
        "promotion_status": "locked"
    });
    std::fs::write(
        output.join("candidate002_report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("wrote {}", output.display());
    Ok(())
}
