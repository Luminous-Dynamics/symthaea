// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Audition procedural HDC/CfC functional-tract identities on one phrase.

use std::path::PathBuf;
use std::{fs::File, io::Write, path::Path};

use anyhow::Result;
use symthaea::voice::functional_singing::{FUNCTIONAL_MOTOR_RATE, FunctionalSingingEngine};
use symthaea::voice::singing_engine::{VocalPerformance, VocalStem};
use symthaea::voice::singing_quality::analyze_vocal_stem;
use symthaea_muse::Note;
use symthaea_vocal_tract::{
    ArticulatoryState, FunctionalTractConfig, FunctionalTractRenderer, FunctionalVoiceIdentity,
};

fn save_pcm16_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let data_size = (samples.len() * 2) as u32;
    let mut file = File::create(path)?;
    file.write_all(b"RIFF")?;
    file.write_all(&(36 + data_size).to_le_bytes())?;
    file.write_all(b"WAVEfmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&(sample_rate * 2).to_le_bytes())?;
    file.write_all(&2u16.to_le_bytes())?;
    file.write_all(&16u16.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;
    for &sample in samples {
        let pcm = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        file.write_all(&pcm.to_le_bytes())?;
    }
    Ok(())
}

fn load_pcm16_wav(path: &Path) -> Result<VocalStem> {
    let bytes = std::fs::read(path)?;
    anyhow::ensure!(
        bytes.len() >= 44 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "invalid WAV: {}",
        path.display()
    );
    let sample_rate = u32::from_le_bytes(bytes[24..28].try_into()?);
    let samples = bytes[44..]
        .chunks_exact(2)
        .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]) as f32 / i16::MAX as f32)
        .collect();
    Ok(VocalStem {
        samples,
        sample_rate,
        backend: "functional-tract-direct-existing".into(),
    })
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().collect();
    let output = args
        .iter()
        .skip(1)
        .find(|argument| !argument.to_string_lossy().starts_with("--"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("audio_output/functional_tract_gallery"));
    std::fs::create_dir_all(&output)?;

    let frequencies = [220.0, 246.94, 277.18, 329.63, 293.66, 277.18, 246.94, 220.0];
    let melody: Vec<_> = frequencies
        .iter()
        .enumerate()
        .map(|(index, &frequency)| Note {
            frequency,
            start_time: index as f32 * 0.42,
            duration: if index == frequencies.len() - 1 {
                0.9
            } else {
                0.42
            },
            velocity: 0.68 + 0.06 * (index as f32 / frequencies.len() as f32),
        })
        .collect();
    let performance =
        VocalPerformance::from_melody("softly now the stars are waking", &melody, "en")?;

    if args.iter().any(|argument| argument == "--analyze-only") {
        let objective: Vec<_> = ["silk", "luminous", "velvet"]
            .into_iter()
            .map(|name| -> Result<_> {
                let stem = load_pcm16_wav(&output.join(format!("{name}.wav")))?;
                Ok(serde_json::json!({
                    "identity": name,
                    "metrics": analyze_vocal_stem(&performance, &stem),
                }))
            })
            .collect::<Result<_>>()?;
        std::fs::write(
            output.join("objective_report.json"),
            serde_json::to_vec_pretty(&objective)?,
        )?;
        return Ok(());
    }

    let silk = FunctionalVoiceIdentity::symthaea();
    let mut luminous = silk.clone();
    luminous.tract_length_cm = 14.9;
    luminous.oral_scale = 1.08;
    luminous.glottal_rd = 1.72;
    luminous.open_quotient = 0.72;
    luminous.aspiration = 0.07;

    let mut velvet = silk.clone();
    velvet.tract_length_cm = 16.7;
    velvet.pharynx_scale = 1.08;
    velvet.glottal_rd = 1.35;
    velvet.open_quotient = 0.62;
    velvet.aspiration = 0.035;
    velvet.spectral_tilt = 0.66;

    let controls_path = output.join("controls.json");
    let states: Vec<ArticulatoryState> =
        if args.iter().any(|argument| argument == "--reuse-controls") {
            serde_json::from_slice(&std::fs::read(&controls_path)?)?
        } else {
            let mut motor = FunctionalSingingEngine::with_identity(48_000, silk.clone());
            motor.motor_states(&performance)?
        };
    std::fs::write(&controls_path, serde_json::to_vec_pretty(&states)?)?;
    let mut objective = Vec::new();
    for (name, identity) in [("silk", silk), ("luminous", luminous), ("velvet", velvet)] {
        let mut renderer = FunctionalTractRenderer::new(
            identity.clone(),
            FunctionalTractConfig {
                sample_rate: 48_000,
                ..Default::default()
            },
        );
        let samples = renderer.synthesize_states(&states, FUNCTIONAL_MOTOR_RATE);
        let physical_diagnostics = renderer.diagnostics().clone();
        let stem = VocalStem {
            samples,
            sample_rate: 48_000,
            backend: format!("functional-tract-direct-{name}"),
        };
        let path = output.join(format!("{name}.wav"));
        save_pcm16_wav(&path, &stem.samples, stem.sample_rate)?;
        std::fs::write(
            output.join(format!("{name}.identity.json")),
            serde_json::to_vec_pretty(&identity)?,
        )?;
        objective.push(serde_json::json!({
            "identity": name,
            "metrics": analyze_vocal_stem(&performance, &stem),
            "physical_diagnostics": physical_diagnostics,
        }));
        println!("{}", path.display());
    }
    std::fs::write(
        output.join("objective_report.json"),
        serde_json::to_vec_pretty(&objective)?,
    )?;
    Ok(())
}
