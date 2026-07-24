// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Source-candidate-002-fixed tract stress-pair campaign.

use std::{
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use anyhow::Result;
use rustfft::{num_complex::Complex32, FftPlanner};
use serde_json::{json, Value};
use symthaea::voice::singing_quality::analyze_render_cleanliness;
use symthaea_vocal_tract::{
    BranchedWaveguideConfig, BranchedWaveguideV2, BranchedWaveguideV2Observed, CardinalVowel,
    GlottalSourceParameters, IdentityAnatomy, IdentityPhysiology, ObservedRender,
    PhysicalTractFrame, SiObservationCalibration, SiObservationStems, TractManifoldCandidateConfig,
    TractManifoldCandidateDecoder,
};

const SAMPLE_RATE: u32 = 48_000;
const FRAME_RATE: f32 = 200.0;
const F0_HZ: f32 = 220.0;
const DURATION_SECONDS: f32 = 2.0;
const IMPULSE_SECONDS: f32 = 0.18;
const SOURCE_002: GlottalSourceParameters =
    GlottalSourceParameters::new(0.50, 1.05, 0.76, 0.28, 0.004, 0.78);
const STRESS_VOWELS: [CardinalVowel; 3] = [CardinalVowel::I, CardinalVowel::O, CardinalVowel::U];

#[derive(Clone)]
struct StaticProfile {
    response: Vec<Complex32>,
    bin_hz: f32,
    f1_hz: f32,
    f2_hz: f32,
    f2_contrast_db: f32,
    response_rms_db: f32,
    finite: bool,
}

fn rms(samples: &[f32]) -> f32 {
    (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len().max(1) as f32).sqrt()
}

fn db(value: f32) -> f32 {
    20.0 * value.max(1e-20).log10()
}

fn save_wav(path: &Path, samples: &[f32]) -> Result<()> {
    let data_size = (samples.len() * 2) as u32;
    let mut file = BufWriter::new(File::create(path)?);
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

fn fft_response(samples: &[f32]) -> (Vec<Complex32>, f32) {
    let size = samples.len().next_power_of_two().max(16_384);
    let mut values = vec![Complex32::new(0.0, 0.0); size];
    for (value, sample) in values.iter_mut().zip(samples) {
        value.re = *sample / 0.1; // De-embed the controlled glottal impulse.
    }
    FftPlanner::new()
        .plan_fft_forward(size)
        .process(&mut values);
    (values, SAMPLE_RATE as f32 / size as f32)
}

fn peak_in(response: &[Complex32], bin_hz: f32, lower: f32, upper: f32) -> (f32, f32) {
    let first = (lower / bin_hz).ceil() as usize;
    let last = ((upper / bin_hz).floor() as usize).min(response.len() / 2 - 1);
    let (index, magnitude) = (first..=last)
        .map(|index| (index, response[index].norm()))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap_or((first, 0.0));
    (index as f32 * bin_hz, magnitude)
}

fn static_profile(frame: &PhysicalTractFrame, vowel: CardinalVowel) -> Result<StaticProfile> {
    let impulse = BranchedWaveguideV2::new(BranchedWaveguideConfig::default())
        .render_impulse_response(frame, IMPULSE_SECONDS)
        .map_err(|error| anyhow::anyhow!(error))?;
    let finite = impulse.final_output.iter().all(|sample| sample.is_finite());
    let response_rms_db = db(rms(&impulse.final_output));
    let (response, bin_hz) = fft_response(&impulse.final_output);
    let target = vowel.target();
    let boundaries = [
        150.0,
        0.5 * (target.formant_hz[0] + target.formant_hz[1]),
        0.5 * (target.formant_hz[1] + target.formant_hz[2]),
        0.5 * (target.formant_hz[2] + target.formant_hz[3]),
        4_500.0,
    ];
    let (f1_hz, _) = peak_in(&response, bin_hz, boundaries[0], boundaries[1]);
    let (f2_hz, f2_peak) = peak_in(&response, bin_hz, boundaries[1], boundaries[2]);
    let shoulder: Vec<_> = response
        .iter()
        .enumerate()
        .filter_map(|(index, value)| {
            let frequency = index as f32 * bin_hz;
            ((frequency - f2_hz).abs() >= 140.0
                && (frequency - f2_hz).abs() <= 320.0
                && frequency >= boundaries[1]
                && frequency < boundaries[2])
                .then_some(value.norm())
        })
        .collect();
    let shoulder_mean = shoulder.iter().sum::<f32>() / shoulder.len().max(1) as f32;
    Ok(StaticProfile {
        response,
        bin_hz,
        f1_hz,
        f2_hz,
        f2_contrast_db: db(f2_peak) - db(shoulder_mean),
        response_rms_db,
        finite,
    })
}

fn normalized_shape(profile: &StaticProfile) -> Vec<f32> {
    let first = (300.0 / profile.bin_hz).ceil() as usize;
    let last = (3_500.0 / profile.bin_hz).floor() as usize;
    let mut values: Vec<_> = profile.response[first..=last]
        .iter()
        .map(|value| db(value.norm()))
        .collect();
    let mean = values.iter().sum::<f32>() / values.len().max(1) as f32;
    for value in &mut values {
        *value -= mean;
    }
    values
}

fn shape_distance(left: &StaticProfile, right: &StaticProfile) -> f32 {
    let left = normalized_shape(left);
    let right = normalized_shape(right);
    (left
        .iter()
        .zip(right)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / left.len().max(1) as f32)
        .sqrt()
}

fn profile_set(frames: &[PhysicalTractFrame; 3]) -> Result<[StaticProfile; 3]> {
    Ok([
        static_profile(&frames[0], CardinalVowel::I)?,
        static_profile(&frames[1], CardinalVowel::O)?,
        static_profile(&frames[2], CardinalVowel::U)?,
    ])
}

fn set_metrics(profiles: &[StaticProfile; 3]) -> Value {
    let i_u_f2_separation_hz = (profiles[0].f2_hz - profiles[2].f2_hz).abs();
    let levels: Vec<_> = profiles
        .iter()
        .map(|profile| profile.response_rms_db)
        .collect();
    let level_spread = levels.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - levels.iter().copied().fold(f32::INFINITY, f32::min);
    json!({
        "i_u_f2_peak_separation_hz": i_u_f2_separation_hz,
        "i_u_transfer_shape_distance_db": shape_distance(&profiles[0], &profiles[2]),
        "o_u_transfer_shape_distance_db": shape_distance(&profiles[1], &profiles[2]),
        "mean_f2_local_contrast_db": profiles.iter().map(|profile| profile.f2_contrast_db).sum::<f32>() / 3.0,
        "static_response_level_spread_db": level_spread,
        "vowels": STRESS_VOWELS.iter().zip(profiles).map(|(vowel, profile)| json!({
            "vowel": vowel.ipa(), "f1_peak_hz": profile.f1_hz, "f2_peak_hz": profile.f2_hz,
            "f2_local_contrast_db": profile.f2_contrast_db,
            "static_response_rms_db": profile.response_rms_db,
            "finite": profile.finite
        })).collect::<Vec<_>>()
    })
}

fn frames_for(
    decoder: &TractManifoldCandidateDecoder,
    anatomy: &IdentityAnatomy,
    physiology: &IdentityPhysiology,
) -> Result<[PhysicalTractFrame; 3]> {
    Ok([
        decoder
            .physical_frame(CardinalVowel::I, anatomy, physiology, F0_HZ)
            .map_err(|error| anyhow::anyhow!(error))?,
        decoder
            .physical_frame(CardinalVowel::O, anatomy, physiology, F0_HZ)
            .map_err(|error| anyhow::anyhow!(error))?,
        decoder
            .physical_frame(CardinalVowel::U, anatomy, physiology, F0_HZ)
            .map_err(|error| anyhow::anyhow!(error))?,
    ])
}

fn metric(value: &Value, name: &str) -> f32 {
    value[name].as_f64().unwrap_or(f64::NEG_INFINITY) as f32
}

fn score(baseline: &Value, candidate: &Value, f1_shift: f32) -> f32 {
    (metric(candidate, "i_u_f2_peak_separation_hz") - metric(baseline, "i_u_f2_peak_separation_hz"))
        / 50.0
        + (metric(candidate, "i_u_transfer_shape_distance_db")
            - metric(baseline, "i_u_transfer_shape_distance_db"))
            / 0.25
        + (metric(candidate, "o_u_transfer_shape_distance_db")
            - metric(baseline, "o_u_transfer_shape_distance_db"))
            / 0.10
        + (metric(candidate, "mean_f2_local_contrast_db")
            - metric(baseline, "mean_f2_local_contrast_db"))
            / 1.0
        - (f1_shift - 150.0).max(0.0) / 25.0
        - (metric(candidate, "static_response_level_spread_db") - 12.0).max(0.0)
}

fn render_observed(frame: &PhysicalTractFrame) -> Result<(ObservedRender, usize)> {
    let mut renderer = BranchedWaveguideV2Observed::new_with_source_override(
        BranchedWaveguideConfig::default(),
        SiObservationCalibration::default(),
        SOURCE_002,
    );
    let frames = vec![frame.clone(); (DURATION_SECONDS * FRAME_RATE) as usize];
    let output = renderer
        .render_frames(&frames, FRAME_RATE)
        .map_err(|error| anyhow::anyhow!(error))?;
    let unintended = renderer
        .diagnostics()
        .reflection_episodes
        .iter()
        .filter(|episode| !episode.intended_closure)
        .count();
    Ok((output, unintended))
}

fn save_observations(path: &Path, observations: &SiObservationStems) -> Result<()> {
    let start = observations
        .glottal_volume_velocity_m3_s
        .len()
        .saturating_sub(16_384);
    let mut file = BufWriter::new(File::create(path)?);
    writeln!(
        file,
        "sample,ug_m3_s,pg_pa,poral_pa,ulip_m3_s,prad_oral_pa,pnasal_pa,unostril_m3_s,prad_nasal_pa,prad_total_pa"
    )?;
    for index in start..observations.glottal_volume_velocity_m3_s.len() {
        writeln!(
            file,
            "{},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e}",
            index,
            observations.glottal_volume_velocity_m3_s[index],
            observations.glottal_tract_pressure_pa[index],
            observations.oral_pre_radiation_pressure_pa[index],
            observations.lip_volume_velocity_m3_s[index],
            observations.radiated_oral_pressure_pa_at_1m[index],
            observations.nasal_pre_radiation_pressure_pa[index],
            observations.nostril_volume_velocity_m3_s[index],
            observations.radiated_nasal_pressure_pa_at_1m[index],
            observations.radiated_total_pressure_pa_at_1m[index]
        )?;
    }
    Ok(())
}

fn dft(samples: &[f32], frequency: f32) -> Complex32 {
    samples
        .iter()
        .enumerate()
        .fold(Complex32::new(0.0, 0.0), |sum, (index, sample)| {
            let phase = -std::f32::consts::TAU * frequency * index as f32 / SAMPLE_RATE as f32;
            sum + Complex32::from_polar(*sample, phase)
        })
}

fn save_harmonic_transfer(path: &Path, observations: &SiObservationStems) -> Result<Vec<Value>> {
    let start = observations
        .glottal_volume_velocity_m3_s
        .len()
        .saturating_sub(16_384);
    let ug = &observations.glottal_volume_velocity_m3_s[start..];
    let lip = &observations.lip_volume_velocity_m3_s[start..];
    let radiated = &observations.radiated_oral_pressure_pa_at_1m[start..];
    let mut file = BufWriter::new(File::create(path)?);
    writeln!(
        file,
        "harmonic,frequency_hz,h_ul_over_ug_re,h_ul_over_ug_im,h_rad_over_ul_re,h_rad_over_ul_im,h_total_re,h_total_im,h_total_db"
    )?;
    let mut records = Vec::new();
    for harmonic in 1..=18 {
        let frequency = harmonic as f32 * F0_HZ;
        let source = dft(ug, frequency);
        let lip_value = dft(lip, frequency);
        let pressure = dft(radiated, frequency);
        let tract = lip_value / source;
        let radiation = pressure / lip_value;
        let total = pressure / source;
        writeln!(
            file,
            "{harmonic},{frequency:.3},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.6}",
            tract.re,
            tract.im,
            radiation.re,
            radiation.im,
            total.re,
            total.im,
            db(total.norm())
        )?;
        records.push(json!({"harmonic":harmonic,"frequency_hz":frequency,"h_ul_over_ug":[tract.re,tract.im],"h_rad_over_ul":[radiation.re,radiation.im],"h_total":[total.re,total.im],"h_total_db":db(total.norm())}));
    }
    Ok(records)
}

fn save_dense_response(path: &Path, profile: &StaticProfile) -> Result<()> {
    let mut file = BufWriter::new(File::create(path)?);
    writeln!(file, "frequency_hz,real,imag,magnitude_db")?;
    let first = (100.0 / profile.bin_hz).ceil() as usize;
    let last = (4_500.0 / profile.bin_hz).floor() as usize;
    for index in first..=last {
        let value = profile.response[index];
        writeln!(
            file,
            "{:.6},{:.9e},{:.9e},{:.6}",
            index as f32 * profile.bin_hz,
            value.re,
            value.im,
            db(value.norm())
        )?;
    }
    Ok(())
}

fn condition_summary(
    id: &str,
    vowel: CardinalVowel,
    frame: &PhysicalTractFrame,
    render: &ObservedRender,
    unintended: usize,
    output: &Path,
) -> Result<Value> {
    let prefix = format!("{}_{}", vowel.ipa(), id);
    save_wav(
        &output.join(format!("{prefix}.raw.wav")),
        &render.stems.final_output,
    )?;
    save_wav(
        &output.join(format!("{prefix}.listening_-20dbfs.wav")),
        &listening_copy(&render.stems.final_output),
    )?;
    save_wav(
        &output.join(format!("{prefix}.glottal_source.wav")),
        &render.stems.glottal_source,
    )?;
    save_wav(
        &output.join(format!("{prefix}.aspiration.wav")),
        &render.stems.aspiration,
    )?;
    save_wav(
        &output.join(format!("{prefix}.oral_output.wav")),
        &render.stems.oral_output,
    )?;
    save_wav(
        &output.join(format!("{prefix}.nasal_output.wav")),
        &render.stems.nasal_output,
    )?;
    save_observations(
        &output.join(format!("{prefix}.si_observations.csv")),
        &render.observations,
    )?;
    let transfer = save_harmonic_transfer(
        &output.join(format!("{prefix}.harmonic_transfer.csv")),
        &render.observations,
    )?;
    let cleanliness = analyze_render_cleanliness(&render.stems.final_output, SAMPLE_RATE);
    let pressure = &render.observations.radiated_total_pressure_pa_at_1m;
    let mean_square_pressure =
        pressure.iter().map(|value| value * value).sum::<f32>() / pressure.len().max(1) as f32;
    Ok(json!({
        "id": id,
        "vowel": vowel.ipa(),
        "rms_dbfs": db(rms(&render.stems.final_output)),
        "diagnostic_radiated_intensity_w_m2": mean_square_pressure / (1.204 * 343.0),
        "absolute_spl_claim": false,
        "cleanliness": cleanliness,
        "clipped_sample_fraction": render.stems.final_output.iter().filter(|sample| sample.abs() >= 1.0).count() as f32 / render.stems.final_output.len().max(1) as f32,
        "unintended_reflection_episodes": unintended,
        "oral_length_cm": frame.oral_length_cm,
        "oral_area_cm2": frame.oral_area_cm2,
        "source_normalized_harmonic_transfer": transfer
    }))
}

fn main() -> Result<()> {
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("audio_output/source_fixed_tract_stress_v1"));
    std::fs::create_dir_all(&output)?;
    let anatomy = IdentityAnatomy::velvet();
    let physiology = IdentityPhysiology::default();
    let baseline_decoder = TractManifoldCandidateDecoder::new(Default::default())
        .map_err(|error| anyhow::anyhow!(error))?;
    let baseline_frames = frames_for(&baseline_decoder, &anatomy, &physiology)?;
    let baseline_profiles = profile_set(&baseline_frames)?;
    let baseline_metrics = set_metrics(&baseline_profiles);

    let mut grid = Vec::new();
    let mut selected: Option<(
        f32,
        TractManifoldCandidateConfig,
        [PhysicalTractFrame; 3],
        [StaticProfile; 3],
        Value,
        f32,
    )> = None;
    for tongue in [0.12, 0.24, 0.36] {
        for back in [0.08, 0.18, 0.28] {
            for lips in [0.10, 0.22, 0.34] {
                for length in [0.02, 0.045, 0.07] {
                    let config = TractManifoldCandidateConfig {
                        tongue_constriction_strength: tongue,
                        back_constriction_strength: back,
                        lip_rounding_strength: lips,
                        protrusion_length_strength: length,
                    };
                    let decoder = TractManifoldCandidateDecoder::new(config)
                        .map_err(|error| anyhow::anyhow!(error))?;
                    let frames = frames_for(&decoder, &anatomy, &physiology)?;
                    let profiles = profile_set(&frames)?;
                    let metrics = set_metrics(&profiles);
                    let max_f1_shift = profiles
                        .iter()
                        .zip(&baseline_profiles)
                        .map(|(candidate, baseline)| (candidate.f1_hz - baseline.f1_hz).abs())
                        .fold(0.0, f32::max);
                    let eligible = profiles.iter().all(|profile| profile.finite)
                        && max_f1_shift <= 150.0
                        && metric(&metrics, "static_response_level_spread_db") <= 12.0;
                    let composite = score(&baseline_metrics, &metrics, max_f1_shift);
                    grid.push(json!({"config":config,"eligible":eligible,"score":composite,"max_f1_shift_hz":max_f1_shift,"metrics":metrics}));
                    if eligible && selected.as_ref().is_none_or(|best| composite > best.0) {
                        selected =
                            Some((composite, config, frames, profiles, metrics, max_f1_shift));
                    }
                }
            }
        }
    }
    let (
        selected_score,
        selected_config,
        candidate_frames,
        candidate_profiles,
        candidate_metrics,
        max_f1_shift,
    ) = selected.ok_or_else(|| anyhow::anyhow!("no physically eligible tract candidate"))?;

    let mut conditions = Vec::new();
    for (index, vowel) in STRESS_VOWELS.iter().copied().enumerate() {
        save_dense_response(
            &output.join(format!(
                "{}_baseline.static_complex_response.csv",
                vowel.ipa()
            )),
            &baseline_profiles[index],
        )?;
        save_dense_response(
            &output.join(format!(
                "{}_candidate.static_complex_response.csv",
                vowel.ipa()
            )),
            &candidate_profiles[index],
        )?;
        let baseline_impulse = BranchedWaveguideV2::default()
            .render_impulse_response(&baseline_frames[index], IMPULSE_SECONDS)
            .map_err(|error| anyhow::anyhow!(error))?;
        let candidate_impulse = BranchedWaveguideV2::default()
            .render_impulse_response(&candidate_frames[index], IMPULSE_SECONDS)
            .map_err(|error| anyhow::anyhow!(error))?;
        save_wav(
            &output.join(format!("{}_baseline.impulse.wav", vowel.ipa())),
            &baseline_impulse.final_output,
        )?;
        save_wav(
            &output.join(format!("{}_candidate.impulse.wav", vowel.ipa())),
            &candidate_impulse.final_output,
        )?;
        let (baseline_render, baseline_reflections) = render_observed(&baseline_frames[index])?;
        let (candidate_render, candidate_reflections) = render_observed(&candidate_frames[index])?;
        conditions.push(condition_summary(
            "baseline",
            vowel,
            &baseline_frames[index],
            &baseline_render,
            baseline_reflections,
            &output,
        )?);
        conditions.push(condition_summary(
            "candidate",
            vowel,
            &candidate_frames[index],
            &candidate_render,
            candidate_reflections,
            &output,
        )?);
    }
    let candidate_levels: Vec<_> = conditions
        .iter()
        .filter(|condition| condition["id"] == "candidate")
        .map(|condition| condition["rms_dbfs"].as_f64().unwrap_or(-120.0) as f32)
        .collect();
    let periodic_level_spread = candidate_levels
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
        - candidate_levels
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
    let separation_gains = json!({
        "i_u_f2_peak_separation_gain_hz": metric(&candidate_metrics,"i_u_f2_peak_separation_hz") - metric(&baseline_metrics,"i_u_f2_peak_separation_hz"),
        "i_u_transfer_shape_distance_gain_db": metric(&candidate_metrics,"i_u_transfer_shape_distance_db") - metric(&baseline_metrics,"i_u_transfer_shape_distance_db"),
        "o_u_transfer_shape_distance_gain_db": metric(&candidate_metrics,"o_u_transfer_shape_distance_db") - metric(&baseline_metrics,"o_u_transfer_shape_distance_db")
    });
    let candidate_clean = conditions
        .iter()
        .filter(|condition| condition["id"] == "candidate")
        .all(|condition| {
            condition["cleanliness"]["render_cleanliness_pass"]
                .as_bool()
                .unwrap_or(false)
                && condition["clipped_sample_fraction"].as_f64().unwrap_or(1.0) == 0.0
                && condition["unintended_reflection_episodes"]
                    .as_u64()
                    .unwrap_or(1)
                    == 0
        });
    let campaign_pass = candidate_clean
        && separation_gains["i_u_f2_peak_separation_gain_hz"]
            .as_f64()
            .unwrap_or(-1.0)
            >= 50.0
        && separation_gains["i_u_transfer_shape_distance_gain_db"]
            .as_f64()
            .unwrap_or(-1.0)
            >= 0.25
        && separation_gains["o_u_transfer_shape_distance_gain_db"]
            .as_f64()
            .unwrap_or(-1.0)
            >= 0.10
        && max_f1_shift <= 150.0
        && periodic_level_spread <= 12.0;
    let report = json!({
        "campaign_version":"symthaea.vowel-truth.source-fixed-tract-stress.v1",
        "status":"executed_diagnostic_only",
        "source_candidate_002_parameters":SOURCE_002,
        "source_changed":false,
        "per_vowel_parameter_tuning_performed":false,
        "grid_cells_executed":grid.len(),
        "selected_score":selected_score,
        "selected_tract_config":selected_config,
        "baseline_static_metrics":baseline_metrics,
        "candidate_static_metrics":candidate_metrics,
        "separation_gains":separation_gains,
        "max_f1_shift_hz":max_f1_shift,
        "candidate_periodic_raw_rms_spread_db":periodic_level_spread,
        "pairwise_transfer_shape_distance_db":{
            "baseline":{"i_o":shape_distance(&baseline_profiles[0],&baseline_profiles[1]),"i_u":shape_distance(&baseline_profiles[0],&baseline_profiles[2]),"o_u":shape_distance(&baseline_profiles[1],&baseline_profiles[2])},
            "candidate":{"i_o":shape_distance(&candidate_profiles[0],&candidate_profiles[1]),"i_u":shape_distance(&candidate_profiles[0],&candidate_profiles[2]),"o_u":shape_distance(&candidate_profiles[1],&candidate_profiles[2])}
        },
        "conditions":conditions,
        "grid":grid,
        "campaign_pass":campaign_pass,
        "human_audio_accessed":false,
        "validation_or_holdout_accessed":false,
        "human_calibration_claim":false,
        "promotion_status":"locked",
        "diagnostic_limitations":["Static complex responses use a controlled normalized glottal impulse and do not establish absolute SPL.","Pole/zero fitting is emitted by a separate validated physics-path diagnostic and is not a human-recording estimator."]
    });
    std::fs::write(
        output.join("tract_stress_report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!(
        "wrote {} pass={campaign_pass} selected={selected_config:?}",
        output.display()
    );
    Ok(())
}
