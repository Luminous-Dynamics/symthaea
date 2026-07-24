// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Isolated renderer, anatomy and planner experiments for vocal physiology v2.

use std::{
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::Result;
use symthaea::voice::{
    functional_singing::{FUNCTIONAL_MOTOR_RATE, FunctionalSingingEngine},
    singing_engine::{VocalPerformance, VocalStem},
    singing_quality::analyze_vocal_stem,
};
use symthaea_muse::Note;
use symthaea_vocal_tract::{
    ArticulatoryState, BranchedWaveguideConfig, BranchedWaveguideDiagnostics, BranchedWaveguideV2,
    FunctionalTractConfig, FunctionalTractRenderer, FunctionalVoiceIdentity, GestureFrame,
    GesturePlanner, IdentityAnatomy, IdentityPhysiology, PhysicalTractFrame, RendererStems,
    TransmissionLineReference, analyze_physiology_acoustics,
};

fn save_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
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
        file.write_all(
            &((sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16).to_le_bytes(),
        )?;
    }
    Ok(())
}

fn listening_copy(samples: &[f32], target_rms_dbfs: f32) -> Vec<f32> {
    let rms = (samples.iter().map(|sample| sample * sample).sum::<f32>()
        / samples.len().max(1) as f32)
        .sqrt();
    let target = 10.0f32.powf(target_rms_dbfs / 20.0);
    let peak = samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0, f32::max);
    let gain = (target / rms.max(1e-9)).min(0.95 / peak.max(1e-9));
    samples.iter().map(|sample| sample * gain).collect()
}

fn performance() -> Result<VocalPerformance> {
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
            velocity: 0.72,
        })
        .collect();
    VocalPerformance::from_melody("softly now the stars are waking", &melody, "en")
}

fn realize(
    gestures: &[GestureFrame],
    anatomy: &IdentityAnatomy,
    physiology: &IdentityPhysiology,
    maximum_rate: f32,
) -> Result<Vec<PhysicalTractFrame>> {
    let mut planner = GesturePlanner::default();
    planner.maximum_rate_per_second = maximum_rate;
    planner
        .realize_sequence(gestures, anatomy, physiology, 1.0 / FUNCTIONAL_MOTOR_RATE)
        .map_err(|error| anyhow::anyhow!(error))
}

fn baseline_states(frames: &[PhysicalTractFrame]) -> Vec<ArticulatoryState> {
    frames
        .iter()
        .map(|frame| {
            let area_cm2 = std::array::from_fn(|index| {
                let position = index as f32 * (frame.oral_area_cm2.len() - 1) as f32 / 23.0;
                let lower = position.floor() as usize;
                let upper = (lower + 1).min(frame.oral_area_cm2.len() - 1);
                let blend = position - lower as f32;
                frame.oral_area_cm2[lower]
                    + blend * (frame.oral_area_cm2[upper] - frame.oral_area_cm2[lower])
            });
            let constriction = frame.constrictions.first();
            ArticulatoryState {
                area_cm2,
                velum_opening: (frame.velopharyngeal_area_cm2 / 0.42).clamp(0.0, 1.0),
                glottal_opening: 1.0 - frame.glottal.adduction.get(),
                turbulence: constriction.map_or(0.0, |_| 0.55),
                turbulence_section: constriction
                    .map_or(12, |value| (value.location.get() * 22.0).round() as usize),
                f0: frame.glottal.f0_hz,
                energy: (frame.lung_pressure_pa / 900.0).clamp(0.0, 1.0),
                voicing: frame.glottal.voicing.get(),
            }
        })
        .collect()
}

fn metric_json(performance: &VocalPerformance, name: &str, samples: Vec<f32>) -> serde_json::Value {
    let stem = VocalStem {
        samples,
        sample_rate: 48_000,
        backend: name.into(),
    };
    serde_json::json!({"name": name, "metrics": analyze_vocal_stem(performance, &stem)})
}

fn save_stems(directory: &Path, stems: &RendererStems) -> Result<()> {
    for (name, samples) in [
        ("raw_glottal_flow", &stems.raw_glottal_flow),
        ("glottal_derivative", &stems.glottal_derivative),
        ("glottal_source", &stems.glottal_source),
        ("aspiration", &stems.aspiration),
        ("turbulence", &stems.turbulence),
        ("oral_output", &stems.oral_output),
        ("nasal_output", &stems.nasal_output),
        ("final", &stems.final_output),
    ] {
        save_wav(&directory.join(format!("{name}.wav")), samples, 48_000)?;
    }
    Ok(())
}

fn save_prefixed_stems(directory: &Path, prefix: &str, stems: &RendererStems) -> Result<()> {
    for (name, samples) in [
        ("raw_glottal_flow", &stems.raw_glottal_flow),
        ("glottal_derivative", &stems.glottal_derivative),
        ("glottal_source", &stems.glottal_source),
        ("aspiration", &stems.aspiration),
        ("turbulence", &stems.turbulence),
        ("oral_output", &stems.oral_output),
        ("nasal_output", &stems.nasal_output),
        ("final", &stems.final_output),
    ] {
        save_wav(
            &directory.join(format!("{prefix}_{name}.wav")),
            samples,
            48_000,
        )?;
    }
    Ok(())
}

fn diagnostics_summary(diagnostics: &BranchedWaveguideDiagnostics) -> serde_json::Value {
    serde_json::json!({
        "internal_sample_rate_min": diagnostics.internal_sample_rate_min,
        "internal_sample_rate_max": diagnostics.internal_sample_rate_max,
        "maximum_stored_energy": diagnostics.maximum_stored_energy,
        "maximum_source_free_energy_ratio": diagnostics.maximum_source_free_energy_ratio,
        "non_finite_samples": diagnostics.non_finite_samples,
        "reflection_sample_hits_recorded": diagnostics.reflection_events.len(),
        "reflection_episodes": diagnostics.reflection_episodes.len(),
        "intended_closure_episodes": diagnostics.reflection_episodes.iter().filter(|event| event.intended_closure).count(),
        "unintended_warning_episodes": diagnostics.reflection_episodes.iter().filter(|event| !event.intended_closure).count(),
        "oral_length_min_cm": diagnostics.oral_length_min_cm,
        "oral_length_max_cm": diagnostics.oral_length_max_cm,
        "maximum_velopharyngeal_area_cm2": diagnostics.maximum_velopharyngeal_area_cm2,
    })
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().collect();
    let output = args
        .iter()
        .skip(1)
        .find(|argument| !argument.to_string_lossy().starts_with("--"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("audio_output/vocal_physiology_v2"));
    std::fs::create_dir_all(&output)?;
    let performance = performance()?;
    let gesture_path = output.join("gestures.v1.json");
    let gestures: Vec<GestureFrame> = if args.iter().any(|argument| argument == "--reuse-gestures")
    {
        serde_json::from_slice(&std::fs::read(&gesture_path)?)?
    } else {
        FunctionalSingingEngine::new(48_000).gesture_frames(&performance)?
    };
    std::fs::write(&gesture_path, serde_json::to_vec_pretty(&gestures)?)?;

    let physiology = IdentityPhysiology::default();
    let velvet_frames = realize(&gestures, &IdentityAnatomy::velvet(), &physiology, 18.0)?;
    std::fs::write(
        output.join("velvet.physical.v1.json"),
        serde_json::to_vec_pretty(&velvet_frames)?,
    )?;

    let mut baseline = FunctionalTractRenderer::new(
        FunctionalVoiceIdentity::muse(),
        FunctionalTractConfig::default(),
    );
    let baseline_audio =
        baseline.synthesize_states(&baseline_states(&velvet_frames), FUNCTIONAL_MOTOR_RATE);
    save_wav(
        &output.join("renderer_same_anatomy_kl24.wav"),
        &baseline_audio,
        48_000,
    )?;

    let mut v2 = BranchedWaveguideV2::new(BranchedWaveguideConfig::default());
    let v2_stems = v2
        .render_frames(&velvet_frames, FUNCTIONAL_MOTOR_RATE)
        .map_err(|error| anyhow::anyhow!(error))?;
    save_stems(&output, &v2_stems)?;
    save_wav(
        &output.join("final.listening_-20dbfs.wav"),
        &listening_copy(&v2_stems.final_output, -20.0),
        48_000,
    )?;
    std::fs::write(
        output.join("renderer_v2.reflections.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "events": &v2.diagnostics().reflection_events,
            "episodes": &v2.diagnostics().reflection_episodes,
        }))?,
    )?;
    let mut reference = TransmissionLineReference::default();
    let reference_stems = reference
        .render_frames(&velvet_frames, FUNCTIONAL_MOTOR_RATE)
        .map_err(|error| anyhow::anyhow!(error))?;
    save_prefixed_stems(&output, "reference", &reference_stems)?;
    std::fs::write(
        output.join("renderer_reference.reflections.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "events": &reference.diagnostics().reflection_events,
            "episodes": &reference.diagnostics().reflection_episodes,
        }))?,
    )?;
    save_wav(
        &output.join("renderer_same_anatomy_reference.wav"),
        &reference_stems.final_output,
        48_000,
    )?;
    save_wav(
        &output.join("renderer_same_anatomy_reference.listening_-20dbfs.wav"),
        &listening_copy(&reference_stems.final_output, -20.0),
        48_000,
    )?;

    let mut identity_results = Vec::new();
    let mut identity_audio = Vec::new();
    for anatomy in [
        IdentityAnatomy::silk(),
        IdentityAnatomy::luminous(),
        IdentityAnatomy::velvet(),
    ] {
        let frames = realize(&gestures, &anatomy, &physiology, 18.0)?;
        let mut renderer = BranchedWaveguideV2::default();
        let stems = renderer
            .render_frames(&frames, FUNCTIONAL_MOTOR_RATE)
            .map_err(|error| anyhow::anyhow!(error))?;
        let filename = format!("identity_same_gesture_{}.wav", anatomy.name.to_lowercase());
        save_wav(&output.join(&filename), &stems.final_output, 48_000)?;
        save_wav(
            &output.join(filename.replace(".wav", ".listening_-20dbfs.wav")),
            &listening_copy(&stems.final_output, -20.0),
            48_000,
        )?;
        std::fs::write(
            output.join(format!(
                "identity_{}.reflections.json",
                anatomy.name.to_lowercase()
            )),
            serde_json::to_vec_pretty(&serde_json::json!({
                "events": &renderer.diagnostics().reflection_events,
                "episodes": &renderer.diagnostics().reflection_episodes,
            }))?,
        )?;
        identity_audio.push((anatomy.name.clone(), stems.final_output.clone()));
        identity_results.push(serde_json::json!({
            "identity": anatomy.name,
            "anatomy": anatomy,
            "metrics": metric_json(&performance, &filename, stems.final_output),
            "diagnostics": diagnostics_summary(renderer.diagnostics()),
        }));
    }
    let mut identity_separation = Vec::new();
    for left in 0..identity_audio.len() {
        for right in left + 1..identity_audio.len() {
            identity_separation.push(serde_json::json!({
                "left": &identity_audio[left].0,
                "right": &identity_audio[right].0,
                "comparison": TransmissionLineReference::compare(
                    &identity_audio[left].1,
                    &identity_audio[right].1,
                ),
            }));
        }
    }

    let mut planner_results = Vec::new();
    for rate in [8.0, 18.0, 40.0] {
        let frames = realize(&gestures, &IdentityAnatomy::velvet(), &physiology, rate)?;
        let mut renderer = BranchedWaveguideV2::default();
        let stems = renderer
            .render_frames(&frames, FUNCTIONAL_MOTOR_RATE)
            .map_err(|error| anyhow::anyhow!(error))?;
        let name = format!("planner_same_anatomy_rate_{rate:.0}.wav");
        save_wav(&output.join(&name), &stems.final_output, 48_000)?;
        planner_results.push(metric_json(&performance, &name, stems.final_output));
    }

    let report = serde_json::json!({
        "schema": "symthaea.vocal-physiology-isolation.v1",
        "gesture_cache": "intention-relative; no absolute tube geometry",
        "renderer_isolation": {
            "anatomy": "Velvet",
            "baseline_metrics": metric_json(&performance, "Kl24BaselineV1", baseline_audio),
            "v2_metrics": metric_json(&performance, "BranchedWaveguideV2", v2_stems.final_output.clone()),
            "reference_metrics": metric_json(&performance, "TransmissionLineReference", reference_stems.final_output.clone()),
            "v2_vs_reference": TransmissionLineReference::compare(
                &v2_stems.final_output,
                &reference_stems.final_output,
            ),
            "v2_acoustic_metrics": analyze_physiology_acoustics(
                &v2_stems,
                &velvet_frames,
                48_000,
                FUNCTIONAL_MOTOR_RATE,
            ),
            "reference_acoustic_metrics": analyze_physiology_acoustics(
                &reference_stems,
                &velvet_frames,
                48_000,
                FUNCTIONAL_MOTOR_RATE,
            ),
            "baseline_diagnostics": {
                "version": &baseline.diagnostics().baseline_version,
                "reflection_sample_hits": baseline.diagnostics().reflection_limit_hits,
                "reflection_events_recorded": baseline.diagnostics().reflection_events.len(),
                "velum_supported": baseline.diagnostics().velum_supported,
                "non_finite_samples": baseline.diagnostics().non_finite_samples,
            },
            "v2_diagnostics": diagnostics_summary(v2.diagnostics()),
            "reference_diagnostics": diagnostics_summary(reference.diagnostics()),
        },
        "identity_isolation": identity_results,
        "identity_separation": identity_separation,
        "planner_isolation": planner_results,
    });
    std::fs::write(
        output.join("isolation_report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", output.display());
    Ok(())
}
