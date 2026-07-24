// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Covered-domain modal /a/ at 220 Hz source-by-tract factorial campaign.

use std::{
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::Result;
use serde_json::{Value, json};
use symthaea_vocal_tract::{
    BranchedWaveguideConfig, BranchedWaveguideV2, CardinalVowel, GestureFrame, GesturePlanner,
    IdentityAnatomy, IdentityPhysiology, LfGlottalSource, PhonationRegime, PhysicalTractFrame,
    VowelAnchorDecoder,
};

const SAMPLE_RATE: u32 = 48_000;
const FRAME_RATE: f32 = 200.0;
const DURATION_SECONDS: f32 = 3.0;
const F0_HZ: f32 = 220.0;

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
        let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        file.write_all(&value.to_le_bytes())?;
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

fn signal_summary(samples: &[f32]) -> Value {
    let start = (0.75 * SAMPLE_RATE as f32) as usize;
    let end = (start + 16_384).min(samples.len());
    let steady = &samples[start.min(end)..end];
    let harmonic_power: Vec<_> = (1..=16)
        .map(|harmonic| goertzel_power(steady, F0_HZ * harmonic as f32).max(1e-18))
        .collect();
    let h1 = harmonic_power.first().copied().unwrap_or(1e-18);
    let harmonics_db: Vec<_> = harmonic_power
        .iter()
        .map(|power| 10.0 * (power / h1).log10())
        .collect();
    let mut band_power = [0.0f32; 4];
    for frequency_hz in (50..=20_000).step_by(25) {
        let power = goertzel_power(steady, frequency_hz as f32);
        let band = match frequency_hz {
            0..=999 => 0,
            1000..=3999 => 1,
            4000..=7999 => 2,
            _ => 3,
        };
        band_power[band] += power;
    }
    let total_band_power = band_power.iter().sum::<f32>().max(1e-18);
    let peak = samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0, f32::max);
    let peak_delta = samples
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).abs())
        .fold(0.0, f32::max);
    json!({
        "rms_dbfs": 20.0 * rms(samples).max(1e-12).log10(),
        "peak_dbfs": 20.0 * peak.max(1e-12).log10(),
        "peak_sample_delta": peak_delta,
        "clipped_sample_fraction": samples.iter().filter(|sample| sample.abs() >= 1.0).count() as f32
            / samples.len().max(1) as f32,
        "harmonic_db_relative_h1": harmonics_db,
        "diagnostic_band_energy_fraction": {
            "below_1khz": band_power[0] / total_band_power,
            "1_4khz": band_power[1] / total_band_power,
            "4_8khz": band_power[2] / total_band_power,
            "8_20khz": band_power[3] / total_band_power
        }
    })
}

fn static_frames(frame: &PhysicalTractFrame) -> Vec<PhysicalTractFrame> {
    vec![frame.clone(); (DURATION_SECONDS * FRAME_RATE) as usize]
}

fn render(frame: &PhysicalTractFrame, legacy_sparse_source: bool) -> Result<Vec<f32>> {
    let mut renderer = BranchedWaveguideV2::new(BranchedWaveguideConfig {
        source_regime: Some(PhonationRegime::Modal),
        legacy_sparse_source,
        ..Default::default()
    });
    Ok(renderer
        .render_frames(&static_frames(frame), FRAME_RATE)
        .map_err(|error| anyhow::anyhow!(error))?
        .final_output)
}

fn current_tract(
    anatomy: &IdentityAnatomy,
    physiology: &IdentityPhysiology,
) -> Result<PhysicalTractFrame> {
    let mut gesture = GestureFrame::default();
    gesture.f0_hz = F0_HZ;
    let mut planner = GesturePlanner::default();
    planner.use_calibrated_vowel_manifold = false;
    planner
        .realize(&gesture, anatomy, physiology, 1.0 / FRAME_RATE)
        .map_err(|error| anyhow::anyhow!(error))
}

fn source_candidate_stem() -> Vec<f32> {
    let mut source = LfGlottalSource::default();
    (0..(DURATION_SECONDS * SAMPLE_RATE as f32) as usize)
        .map(|_| {
            source
                .step(
                    F0_HZ,
                    0.78,
                    1.0,
                    PhonationRegime::Modal.parameters(),
                    SAMPLE_RATE as f32,
                )
                .filtered_source
        })
        .collect()
}

fn resonator(input: &[f32], frequency_hz: f32, bandwidth_hz: f32) -> Vec<f32> {
    let radius = (-std::f32::consts::PI * bandwidth_hz / SAMPLE_RATE as f32).exp();
    let angle = std::f32::consts::TAU * frequency_hz / SAMPLE_RATE as f32;
    let a1 = 2.0 * radius * angle.cos();
    let a2 = -(radius * radius);
    let gain = (1.0 - radius).max(1e-5);
    let mut y1 = 0.0;
    let mut y2 = 0.0;
    input
        .iter()
        .map(|&sample| {
            let output = gain * sample + a1 * y1 + a2 * y2;
            y2 = y1;
            y1 = output;
            output
        })
        .collect()
}

fn candidate_source_formant_control(source: &[f32]) -> Vec<f32> {
    let target = CardinalVowel::A.target();
    let first = resonator(source, target.formant_hz[0], 80.0);
    let second = resonator(&first, target.formant_hz[1], 100.0);
    resonator(&second, target.formant_hz[2], 140.0)
}

fn main() -> Result<()> {
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("audio_output/covered_domain_a220_factorial_v1"));
    std::fs::create_dir_all(&output)?;

    let anatomy = IdentityAnatomy::velvet();
    let physiology = IdentityPhysiology::default();
    let baseline_tract = current_tract(&anatomy, &physiology)?;
    let tract_candidate = VowelAnchorDecoder::default()
        .physical_frame(CardinalVowel::A, &anatomy, &physiology, F0_HZ)
        .map_err(|error| anyhow::anyhow!(error))?;

    let conditions = [
        ("baseline_a", &baseline_tract, true),
        ("source_candidate_001", &baseline_tract, false),
        ("tract_candidate_a_001", &tract_candidate, true),
        ("joint_candidate_001", &tract_candidate, false),
    ];
    let mut reports = Vec::new();
    for (id, tract, legacy_source) in conditions {
        let audio = render(tract, legacy_source)?;
        save_wav(&output.join(format!("{id}.raw.wav")), &audio)?;
        save_wav(
            &output.join(format!("{id}.listening_-20dbfs.wav")),
            &listening_copy(&audio),
        )?;
        reports.push(json!({
            "id": id,
            "source": if legacy_source {"frozen_legacy_sparse"} else {"source_candidate_001_lf"},
            "tract": if std::ptr::eq(tract, &baseline_tract) {"frozen_current"} else {"tract_candidate_a_001_anchor_v1"},
            "summary": signal_summary(&audio)
        }));
    }

    let source = source_candidate_stem();
    save_wav(
        &output.join("diagnostic_source_candidate_001_alone.wav"),
        &listening_copy(&source),
    )?;

    let mut impulse_renderer = BranchedWaveguideV2::new(BranchedWaveguideConfig::default());
    let impulse = impulse_renderer
        .render_impulse_response(&tract_candidate, 0.25)
        .map_err(|error| anyhow::anyhow!(error))?
        .final_output;
    save_wav(
        &output.join("diagnostic_tract_candidate_a_001_impulse.wav"),
        &listening_copy(&impulse),
    )?;

    let formant = candidate_source_formant_control(&source);
    save_wav(
        &output.join("diagnostic_source_candidate_001_formant_filter.wav"),
        &listening_copy(&formant),
    )?;

    let report = json!({
        "campaign_version": "symthaea.vowel-truth.modal-a-220-factorial.v1",
        "status": "executed_diagnostic_only",
        "target": {"vowel":"a", "f0_hz":F0_HZ, "phonation_intent":"modal"},
        "identity": {"kind":"synthetic", "anatomy":anatomy.name},
        "residual_model_enabled": false,
        "validation_or_holdout_access": false,
        "human_calibration_claim": false,
        "conditions": reports,
        "diagnostics": {
            "source_alone": signal_summary(&source),
            "tract_impulse_file": "diagnostic_tract_candidate_a_001_impulse.wav",
            "candidate_source_formant_filter_file": "diagnostic_source_candidate_001_formant_filter.wav",
            "candidate_source_formant_filter_summary": signal_summary(&formant),
            "bandwidth_claim": null,
            "note": "The rejected half-power bandwidth estimator is not used. No candidate is called calibrated."
        },
        "promotion_status": "locked"
    });
    std::fs::write(
        output.join("factorial_report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("wrote {}", output.display());
    Ok(())
}
