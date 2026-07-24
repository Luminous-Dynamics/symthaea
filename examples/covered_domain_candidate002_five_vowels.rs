// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Frozen candidate-002 generalization across five vowels at 220 Hz.

use std::{
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::Result;
use serde_json::{Value, json};
use symthaea::voice::singing_quality::analyze_render_cleanliness;
use symthaea_vocal_tract::{
    CandidateBranchedWaveguideV2, CandidateRendererStems, CandidateWaveguideConfig, CardinalVowel,
    GlottalSourceParameters, IdentityAnatomy, IdentityPhysiology, PhonationRegime,
    PhysicalTractFrame, VowelAnchorDecoder,
};

const SAMPLE_RATE: u32 = 48_000;
const FRAME_RATE: f32 = 200.0;
const F0_HZ: f32 = 220.0;
const DURATION_SECONDS: f32 = 2.0;

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

fn peak_delta(samples: &[f32]) -> f32 {
    samples
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).abs())
        .fold(0.0, f32::max)
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

fn render(
    frame: &PhysicalTractFrame,
    parameters: GlottalSourceParameters,
) -> Result<(CandidateRendererStems, usize)> {
    let mut config = CandidateWaveguideConfig::default();
    config.source_parameters_override = Some(parameters);
    let mut renderer = CandidateBranchedWaveguideV2::new(config);
    let frames = vec![frame.clone(); (DURATION_SECONDS * FRAME_RATE) as usize];
    let stems = renderer
        .render_frames(&frames, FRAME_RATE)
        .map_err(|error| anyhow::anyhow!(error))?;
    let unintended = renderer
        .diagnostics()
        .reflection_episodes
        .iter()
        .filter(|episode| !episode.intended_closure)
        .count();
    Ok((stems, unintended))
}

fn summarize(
    vowel: CardinalVowel,
    stems: &CandidateRendererStems,
    unintended_reflections: usize,
) -> Value {
    let samples = &stems.final_output;
    let start = (0.55 * SAMPLE_RATE as f32) as usize;
    let end = (start + 16_384).min(samples.len());
    let steady = &samples[start.min(end)..end];
    let h1 = goertzel_power(steady, F0_HZ).max(1e-18);
    let target = vowel.target();
    let nearest_harmonics: Vec<_> = target.formant_hz[..3]
        .iter()
        .map(|formant| {
            let harmonic = (*formant / F0_HZ).round().clamp(1.0, 16.0) as usize;
            let relative_db =
                10.0 * (goertzel_power(steady, harmonic as f32 * F0_HZ).max(1e-18) / h1).log10();
            json!({"formant_target_hz":formant,"harmonic":harmonic,"relative_db":relative_db})
        })
        .collect();
    let mean_support = nearest_harmonics
        .iter()
        .map(|item| item["relative_db"].as_f64().unwrap_or(-120.0) as f32)
        .sum::<f32>()
        / 3.0;
    let peak = samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0, f32::max);
    let final_peak_delta = peak_delta(samples);
    let contextual_cleanliness = analyze_render_cleanliness(samples, SAMPLE_RATE);
    let aspiration_db =
        20.0 * (rms(&stems.aspiration).max(1e-12) / rms(&stems.glottal_source).max(1e-12)).log10();
    json!({
        "rms_dbfs":20.0*rms(samples).max(1e-12).log10(),
        "peak_dbfs":20.0*peak.max(1e-12).log10(),
        "peak_sample_delta":final_peak_delta,
        "contextual_cleanliness":contextual_cleanliness,
        "clipped_sample_fraction":samples.iter().filter(|sample| sample.abs()>=1.0).count() as f32/samples.len().max(1) as f32,
        "all_finite":samples.iter().all(|sample|sample.is_finite()),
        "aspiration_to_periodic_db":aspiration_db,
        "unintended_reflection_episodes":unintended_reflections,
        "stem_peak_sample_delta": {
            "glottal_source": peak_delta(&stems.glottal_source),
            "aspiration": peak_delta(&stems.aspiration),
            "oral_output": peak_delta(&stems.oral_output),
            "nasal_output": peak_delta(&stems.nasal_output),
            "final_output": final_peak_delta
        },
        "nearest_formant_harmonics":nearest_harmonics,
        "mean_nearest_formant_harmonic_db":mean_support
    })
}

fn main() -> Result<()> {
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("audio_output/covered_domain_candidate002_five_vowels_v1")
        });
    std::fs::create_dir_all(&output)?;
    let anatomy = IdentityAnatomy::velvet();
    let physiology = IdentityPhysiology::default();
    let decoder = VowelAnchorDecoder::default();
    let candidate001 = PhonationRegime::Modal.parameters();
    let candidate002 = GlottalSourceParameters::new(0.50, 1.05, 0.76, 0.28, 0.004, 0.78);
    let mut cells = Vec::new();
    let mut campaign_pass = true;

    for vowel in CardinalVowel::ALL {
        let frame = decoder
            .physical_frame(vowel, &anatomy, &physiology, F0_HZ)
            .map_err(|error| anyhow::anyhow!(error))?;
        let (left, left_reflections) = render(&frame, candidate001)?;
        let (right, right_reflections) = render(&frame, candidate002)?;
        let left_summary = summarize(vowel, &left, left_reflections);
        let right_summary = summarize(vowel, &right, right_reflections);
        let regression_db = left_summary["mean_nearest_formant_harmonic_db"]
            .as_f64()
            .unwrap_or(-120.0) as f32
            - right_summary["mean_nearest_formant_harmonic_db"]
                .as_f64()
                .unwrap_or(-120.0) as f32;
        let physical_pass = right_summary["all_finite"].as_bool().unwrap_or(false)
            && right_summary["clipped_sample_fraction"]
                .as_f64()
                .unwrap_or(1.0)
                == 0.0
            && right_summary["peak_sample_delta"].as_f64().unwrap_or(1.0) <= 0.012
            && right_reflections == 0;
        let aspiration_db = right_summary["aspiration_to_periodic_db"]
            .as_f64()
            .unwrap_or(120.0);
        let cell_pass =
            physical_pass && regression_db <= 3.0 && (-55.0..=-20.0).contains(&aspiration_db);
        campaign_pass &= cell_pass;
        let name = vowel.ipa();
        save_wav(
            &output.join(format!("{name}_candidate001.raw.wav")),
            &left.final_output,
        )?;
        save_wav(
            &output.join(format!("{name}_candidate001.listening_-20dbfs.wav")),
            &listening_copy(&left.final_output),
        )?;
        save_wav(
            &output.join(format!("{name}_candidate002.raw.wav")),
            &right.final_output,
        )?;
        save_wav(
            &output.join(format!("{name}_candidate002.listening_-20dbfs.wav")),
            &listening_copy(&right.final_output),
        )?;
        cells.push(json!({
            "vowel":name,
            "candidate001":left_summary,
            "candidate002":right_summary,
            "candidate002_harmonic_regression_db":regression_db,
            "physical_pass":physical_pass,
            "cell_pass":cell_pass
        }));
    }
    let report = json!({
        "campaign_version":"symthaea.vowel-truth.candidate002-five-vowel-220.v1",
        "status":"executed_diagnostic_only",
        "source_candidate_parameters":candidate002,
        "per_vowel_source_tuning_performed":false,
        "cells":cells,
        "campaign_pass":campaign_pass,
        "human_audio_accessed":false,
        "validation_or_holdout_accessed":false,
        "human_calibration_claim":false,
        "promotion_status":"locked"
    });
    std::fs::write(
        output.join("five_vowel_report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("wrote {} pass={campaign_pass}", output.display());
    Ok(())
}
