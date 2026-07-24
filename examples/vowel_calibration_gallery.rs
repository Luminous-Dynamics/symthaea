// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reproducible source/vowel calibration and 2x2 source-by-tract ablation pack.

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
    VowelAnchorDecoder, analyze_impulse_response, analyze_source_quality,
    speech::vocoder,
    types::{FormantFrame, SourceType},
};

const SAMPLE_RATE: u32 = 48_000;
const FRAME_RATE: f32 = 200.0;
const DURATION_SECONDS: f32 = 1.5;

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

fn listening_copy(samples: &[f32], target_rms_dbfs: f32) -> Vec<f32> {
    let rms = rms(samples);
    let peak = samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0, f32::max);
    let target = 10.0f32.powf(target_rms_dbfs / 20.0);
    let gain = (target / rms.max(1e-9)).min(0.95 / peak.max(1e-9));
    samples.iter().map(|sample| sample * gain).collect()
}

fn rms(samples: &[f32]) -> f32 {
    (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len().max(1) as f32).sqrt()
}

fn regime_name(regime: PhonationRegime) -> &'static str {
    match regime {
        PhonationRegime::Modal => "modal",
        PhonationRegime::Breathy => "breathy",
        PhonationRegime::Pressed => "pressed",
        PhonationRegime::Head => "head",
        PhonationRegime::Falsetto => "falsetto",
        PhonationRegime::Choral => "choral",
        PhonationRegime::Belt => "belt",
    }
}

fn static_frames(frame: &PhysicalTractFrame) -> Vec<PhysicalTractFrame> {
    vec![frame.clone(); (DURATION_SECONDS * FRAME_RATE) as usize]
}

fn procedural_frame(
    anatomy: &IdentityAnatomy,
    physiology: &IdentityPhysiology,
    f0_hz: f32,
) -> Result<PhysicalTractFrame> {
    let mut gesture = GestureFrame::default();
    gesture.f0_hz = f0_hz;
    let mut planner = GesturePlanner::default();
    planner.use_calibrated_vowel_manifold = false;
    planner
        .realize(&gesture, anatomy, physiology, 1.0 / FRAME_RATE)
        .map_err(|error| anyhow::anyhow!(error))
}

fn render_static(
    frame: &PhysicalTractFrame,
    regime: PhonationRegime,
    legacy_sparse_source: bool,
) -> Result<Vec<f32>> {
    let config = BranchedWaveguideConfig {
        source_regime: Some(regime),
        legacy_sparse_source,
        ..Default::default()
    };
    let mut renderer = BranchedWaveguideV2::new(config);
    Ok(renderer
        .render_frames(&static_frames(frame), FRAME_RATE)
        .map_err(|error| anyhow::anyhow!(error))?
        .final_output)
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

fn audio_summary(samples: &[f32], f0_hz: f32) -> Value {
    let onset = (0.30 * SAMPLE_RATE as f32) as usize;
    let available = &samples[onset.min(samples.len())..];
    let analysis_len = available.len().min(8192);
    let steady = &available[..analysis_len];
    let harmonic_power: Vec<_> = (1..=12)
        .map(|harmonic| goertzel_power(steady, f0_hz * harmonic as f32).max(1e-18))
        .collect();
    let h1 = harmonic_power.first().copied().unwrap_or(1e-18);
    let harmonic_db_relative_h1: Vec<_> = harmonic_power
        .iter()
        .map(|power| 10.0 * (power / h1).log10())
        .collect();
    let mut bands = [0.0f32; 4];
    for frequency in (50..=20_000).step_by(25) {
        let power = goertzel_power(steady, frequency as f32);
        let index = match frequency {
            0..=999 => 0,
            1000..=3999 => 1,
            4000..=7999 => 2,
            _ => 3,
        };
        bands[index] += power;
    }
    let total = bands.iter().sum::<f32>().max(1e-18);
    json!({
        "rms_dbfs": 20.0 * rms(samples).max(1e-12).log10(),
        "harmonic_db_relative_h1": harmonic_db_relative_h1,
        "band_energy_fraction": {
            "below_1khz": bands[0] / total,
            "1_4khz": bands[1] / total,
            "4_8khz": bands[2] / total,
            "8_20khz": bands[3] / total,
        }
    })
}

fn formant_reference() -> Vec<f32> {
    let count = (DURATION_SECONDS * FRAME_RATE) as usize;
    let target = CardinalVowel::A.target();
    let frames: Vec<_> = (0..count)
        .map(|index| FormantFrame {
            f1: target.formant_hz[0],
            f2: target.formant_hz[1],
            f3: target.formant_hz[2],
            b1: 80.0,
            b2: 100.0,
            b3: 140.0,
            f0: 220.0,
            energy: 0.72,
            voicing: 1.0,
            time: index as f32 / FRAME_RATE,
            source_type: SourceType::Vowel,
            nasal_zero_freq: 0.0,
            nasal_zero_bw: 0.0,
        })
        .collect();
    vocoder::synthesize(&frames, SAMPLE_RATE)
}

fn main() -> Result<()> {
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("audio_output/vowel_calibration"));
    let source_directory = output.join("source_regimes");
    let impulse_directory = output.join("impulse_responses");
    let vowel_directory = output.join("sustained_vowels");
    let factorial_directory = output.join("factorial_a_220hz");
    for directory in [
        &output,
        &source_directory,
        &impulse_directory,
        &vowel_directory,
        &factorial_directory,
    ] {
        std::fs::create_dir_all(directory)?;
    }

    let regimes = [
        PhonationRegime::Modal,
        PhonationRegime::Breathy,
        PhonationRegime::Pressed,
        PhonationRegime::Head,
        PhonationRegime::Falsetto,
        PhonationRegime::Choral,
        PhonationRegime::Belt,
    ];
    let mut source_reports = Vec::new();
    let mut all_source_quality_pass = true;
    for regime in regimes {
        let name = regime_name(regime);
        let mut source = LfGlottalSource::default();
        let mut flow = Vec::new();
        let mut derivative = Vec::new();
        let mut filtered = Vec::new();
        let mut aspiration = Vec::new();
        for _ in 0..(DURATION_SECONDS * SAMPLE_RATE as f32) as usize {
            let sample = source.step(220.0, 0.78, 1.0, regime.parameters(), SAMPLE_RATE as f32);
            flow.push(sample.raw_flow);
            derivative.push(sample.differentiated_flow);
            filtered.push(sample.filtered_source);
            aspiration.push(sample.aspiration);
        }
        let trim = (0.10 * SAMPLE_RATE as f32) as usize;
        let metrics = analyze_source_quality(
            &flow[trim..],
            &filtered[trim..],
            &aspiration[trim..],
            SAMPLE_RATE,
            220.0,
        );
        save_wav(
            &source_directory.join(format!("{name}_raw_flow.wav")),
            &flow,
        )?;
        save_wav(
            &source_directory.join(format!("{name}_flow_derivative.wav")),
            &derivative,
        )?;
        save_wav(
            &source_directory.join(format!("{name}_filtered_source.wav")),
            &listening_copy(&filtered, -20.0),
        )?;
        save_wav(
            &source_directory.join(format!("{name}_aspiration.wav")),
            &listening_copy(&aspiration, -28.0),
        )?;
        let quality_gate = metrics.quality_gate();
        all_source_quality_pass &= quality_gate.pass;
        source_reports.push(json!({
            "regime": name,
            "parameters": regime.parameters(),
            "metrics": metrics,
            "quality_gate": quality_gate
        }));
    }

    let anatomy = IdentityAnatomy::velvet();
    let physiology = IdentityPhysiology::default();
    let decoder = VowelAnchorDecoder::default();
    let mut calibration_reports = Vec::new();
    let mut all_vowel_targets_pass = true;
    for vowel in CardinalVowel::ALL {
        let frame = decoder
            .physical_frame(vowel, &anatomy, &physiology, 220.0)
            .map_err(|error| anyhow::anyhow!(error))?;
        let mut renderer = BranchedWaveguideV2::new(BranchedWaveguideConfig::default());
        let impulse = renderer
            .render_impulse_response(&frame, 0.20)
            .map_err(|error| anyhow::anyhow!(error))?
            .final_output;
        let calibration = analyze_impulse_response(vowel, &impulse, SAMPLE_RATE);
        all_vowel_targets_pass &= calibration.target_pass;
        save_wav(
            &impulse_directory.join(format!("{}_anchor_v1_impulse.wav", vowel.ipa())),
            &listening_copy(&impulse, -20.0),
        )?;
        calibration_reports.push(calibration);

        for f0_hz in [110.0, 165.0, 220.0, 330.0] {
            let frame = decoder
                .physical_frame(vowel, &anatomy, &physiology, f0_hz)
                .map_err(|error| anyhow::anyhow!(error))?;
            let audio = render_static(&frame, PhonationRegime::Modal, false)?;
            let stem = format!("{}_modal_{f0_hz:.0}hz", vowel.ipa());
            save_wav(&vowel_directory.join(format!("{stem}.wav")), &audio)?;
            save_wav(
                &vowel_directory.join(format!("{stem}.listening_-20dbfs.wav")),
                &listening_copy(&audio, -20.0),
            )?;
        }
    }

    let current = procedural_frame(&anatomy, &physiology, 220.0)?;
    let calibrated = decoder
        .physical_frame(CardinalVowel::A, &anatomy, &physiology, 220.0)
        .map_err(|error| anyhow::anyhow!(error))?;
    let factorial = [
        ("A_legacy_source_current_tract", &current, true),
        ("B_calibrated_source_current_tract", &current, false),
        ("C_legacy_source_calibrated_a", &calibrated, true),
        ("D_calibrated_source_calibrated_a", &calibrated, false),
    ];
    let mut factorial_reports = Vec::new();
    for (name, frame, legacy) in factorial {
        let audio = render_static(frame, PhonationRegime::Modal, legacy)?;
        save_wav(&factorial_directory.join(format!("{name}.wav")), &audio)?;
        save_wav(
            &factorial_directory.join(format!("{name}.listening_-20dbfs.wav")),
            &listening_copy(&audio, -20.0),
        )?;
        factorial_reports.push(json!({"name": name, "audio": audio_summary(&audio, 220.0)}));
    }
    let formant_reference = formant_reference();
    save_wav(
        &factorial_directory.join("reference_conventional_formant_a_220hz.wav"),
        &listening_copy(&formant_reference, -20.0),
    )?;

    let report = json!({
        "schema": "symthaea.vowel_calibration.v1",
        "sample_rate": SAMPLE_RATE,
        "duration_seconds": DURATION_SECONDS,
        "identity_anatomy": anatomy.name,
        "source_quality_pass": all_source_quality_pass,
        "source_regimes": source_reports,
        "all_vowel_targets_pass": all_vowel_targets_pass,
        "vowel_anchor_calibration": calibration_reports,
        "factorial_a_220hz": factorial_reports,
        "references": {
            "conventional_formant": "factorial_a_220hz/reference_conventional_formant_a_220hz.wav",
            "real_human_vowel": null,
            "real_human_note": "No locally licensed sustained human-vowel recording was found. Add a pitch/loudness-matched permissively licensed reference; do not substitute synthetic audio for this cell."
        },
        "interpretation": "Anchor v1 is experimental. It becomes production-calibrated only when its measured F1-F4 targets pass and blinded listening confirms a perceptual gain."
    });
    std::fs::write(
        output.join("vowel_calibration_report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("wrote {}", output.display());
    Ok(())
}
