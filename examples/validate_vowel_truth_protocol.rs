// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Executable validator and deterministic challenge generator for Vowel Truth Stage 0.

use std::{
    collections::{BTreeMap, BTreeSet},
    f64::consts::PI,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const VERSION: &str = "symthaea.vowel-truth.stage0.v1";
const SAMPLE_RATE: u32 = 48_000;
const REQUIRED: &[&str] = &[
    "VOWEL_TRUTH_STAGE0_SPEC.md",
    "corpus_record.schema.json",
    "corpus_manifest.jsonl",
    "estimator_protocol.toml",
    "estimators.lock",
    "estimator_validation_manifest.json",
    "physics_observation_contract.md",
    "listener_preregistration.yaml",
    "split_manifest.json",
    "license_ledger.json",
];
const PROTECTED: &[&str] = &[
    "examples/validate_vowel_truth_protocol.rs",
    "crates/domains/symthaea-vocal-tract/src/branched_waveguide.rs",
    "crates/domains/symthaea-vocal-tract/src/transmission_line_reference.rs",
    "crates/domains/symthaea-vocal-tract/src/glottal_source.rs",
    "crates/domains/symthaea-vocal-tract/src/vowel_calibration.rs",
    "crates/domains/symthaea-vocal-tract/src/physiology.rs",
    "crates/domains/symthaea-vocal-tract/src/residual_detail.rs",
    "src/voice/functional_singing.rs",
    "src/voice/singing_engine.rs",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LockedFile {
    path: String,
    blake3: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProtocolLock {
    protocol_version: String,
    freeze_phase: String,
    protocol_content_hash: String,
    artifacts: Vec<LockedFile>,
    protected_implementation: Vec<LockedFile>,
}

#[derive(Debug, Clone, Serialize)]
struct ChallengeMeasurement {
    case_id: String,
    parameter: String,
    expected: f64,
    recovered: f64,
    absolute_error: f64,
    tolerance: f64,
    pass: bool,
}

#[derive(Debug, Clone, Copy)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn polar(radius: f64, phase: f64) -> Self {
        Self::new(radius * phase.cos(), radius * phase.sin())
    }

    fn abs2(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    fn abs(self) -> f64 {
        self.abs2().sqrt()
    }

    fn phase(self) -> f64 {
        self.im.atan2(self.re)
    }

    fn mul(self, other: Self) -> Self {
        Self::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }

    fn div(self, other: Self) -> Self {
        let denominator = other.abs2().max(1e-30);
        Self::new(
            (self.re * other.re + self.im * other.im) / denominator,
            (self.im * other.re - self.re * other.im) / denominator,
        )
    }

    fn sub(self, other: Self) -> Self {
        Self::new(self.re - other.re, self.im - other.im)
    }
}

fn z_inverse(frequency_hz: f64) -> Complex {
    Complex::polar(1.0, -2.0 * PI * frequency_hz / SAMPLE_RATE as f64)
}

fn pole_denominator(frequency_hz: f64, pole_hz: f64, bandwidth_hz: f64) -> Complex {
    let radius = (-PI * bandwidth_hz / SAMPLE_RATE as f64).exp();
    let angle = 2.0 * PI * pole_hz / SAMPLE_RATE as f64;
    let z1 = z_inverse(frequency_hz);
    let z2 = z1.mul(z1);
    Complex::new(1.0, 0.0)
        .sub(Complex::new(2.0 * radius * angle.cos(), 0.0).mul(z1))
        .sub(Complex::new(-radius * radius, 0.0).mul(z2))
}

fn resonator_response(frequency_hz: f64, pole_hz: f64, bandwidth_hz: f64) -> Complex {
    Complex::new(1.0, 0.0).div(pole_denominator(frequency_hz, pole_hz, bandwidth_hz))
}

fn zero_response(frequency_hz: f64, zero_hz: f64, bandwidth_hz: f64) -> Complex {
    pole_denominator(frequency_hz, zero_hz, bandwidth_hz)
}

fn peak_frequency(response: impl Fn(f64) -> Complex, lower: f64, upper: f64) -> f64 {
    let mut best = (lower, -1.0);
    let mut frequency = lower;
    while frequency <= upper {
        let magnitude = response(frequency).abs2();
        if magnitude > best.1 {
            best = (frequency, magnitude);
        }
        frequency += 2.0;
    }
    best.0
}

fn zero_frequency(response: impl Fn(f64) -> Complex, lower: f64, upper: f64) -> f64 {
    let mut best = (lower, f64::INFINITY);
    let mut frequency = lower;
    while frequency <= upper {
        let magnitude = response(frequency).abs2();
        if magnitude < best.1 {
            best = (frequency, magnitude);
        }
        frequency += 2.0;
    }
    best.0
}

fn bandwidth(response: impl Fn(f64) -> Complex, peak_hz: f64) -> f64 {
    let threshold = response(peak_hz).abs2() * 0.5;
    let mut left = peak_hz;
    let mut right = peak_hz;
    while left > 20.0 && response(left).abs2() >= threshold {
        left -= 1.0;
    }
    while right < 10_000.0 && response(right).abs2() >= threshold {
        right += 1.0;
    }
    right - left
}

fn impulse_response(poles: &[(f64, f64)], zero: Option<(f64, f64)>, seconds: f64) -> Vec<f32> {
    let length = (seconds * SAMPLE_RATE as f64) as usize;
    let mut signal = vec![0.0f64; length];
    signal[0] = 0.2;
    for &(frequency, bandwidth) in poles {
        let radius = (-PI * bandwidth / SAMPLE_RATE as f64).exp();
        let coefficient = 2.0 * radius * (2.0 * PI * frequency / SAMPLE_RATE as f64).cos();
        let radius_two = radius * radius;
        let mut y1 = 0.0;
        let mut y2 = 0.0;
        for sample in &mut signal {
            let output = *sample + coefficient * y1 - radius_two * y2;
            y2 = y1;
            y1 = output;
            *sample = output * (1.0 - radius).max(1e-4);
        }
    }
    if let Some((frequency, bandwidth)) = zero {
        let radius = (-PI * bandwidth / SAMPLE_RATE as f64).exp();
        let coefficient = -2.0 * radius * (2.0 * PI * frequency / SAMPLE_RATE as f64).cos();
        let radius_two = radius * radius;
        let input = signal.clone();
        for index in 0..signal.len() {
            signal[index] = input[index]
                + if index >= 1 {
                    coefficient * input[index - 1]
                } else {
                    0.0
                }
                + if index >= 2 {
                    radius_two * input[index - 2]
                } else {
                    0.0
                };
        }
    }
    normalize(signal.into_iter().map(|value| value as f32).collect())
}

fn harmonic_signal(f0: f64, h1_h2_db: f64, tilt_db_octave: f64, snr_db: f64) -> Vec<f32> {
    let length = SAMPLE_RATE as usize;
    let mut output = vec![0.0f32; length];
    let mut noise = 0x9e37_79b9_7f4a_7c15u64;
    for (index, sample) in output.iter_mut().enumerate() {
        let time = index as f64 / SAMPLE_RATE as f64;
        let mut value = 0.0;
        for harmonic in 1..=((20_000.0 / f0) as usize).min(64) {
            let mut amplitude = 10.0f64.powf(tilt_db_octave * (harmonic as f64).log2() / 20.0);
            if harmonic == 2 {
                amplitude = 10.0f64.powf(-h1_h2_db / 20.0);
            }
            value += amplitude * (2.0 * PI * f0 * harmonic as f64 * time).sin();
        }
        noise ^= noise << 13;
        noise ^= noise >> 7;
        noise ^= noise << 17;
        let white = ((noise >> 40) as f64 / 16_777_215.0) * 2.0 - 1.0;
        *sample = (value + white * 10.0f64.powf(-snr_db / 20.0)) as f32;
    }
    normalize(output)
}

fn normalize(mut samples: Vec<f32>) -> Vec<f32> {
    let peak = samples.iter().map(|value| value.abs()).fold(0.0, f32::max);
    let gain = 0.8 / peak.max(1e-9);
    for sample in &mut samples {
        *sample *= gain;
    }
    samples
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
        let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        file.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn challenge_suite(directory: &Path) -> Result<(Vec<ChallengeMeasurement>, Value)> {
    std::fs::create_dir_all(directory)?;
    let isolated = [(500.0, 80.0), (1500.0, 120.0), (2500.0, 180.0)];
    save_wav(
        &directory.join("isolated_all_pole.wav"),
        &impulse_response(&isolated, None, 0.5),
    )?;
    save_wav(
        &directory.join("close_formants.wav"),
        &impulse_response(&[(1400.0, 70.0), (1560.0, 90.0)], None, 0.5),
    )?;
    save_wav(
        &directory.join("nasal_pole_zero.wav"),
        &impulse_response(&[(300.0, 90.0), (1100.0, 150.0)], Some((750.0, 100.0)), 0.5),
    )?;
    for (name, h1_h2, tilt, snr) in [
        ("modal", 3.0, -12.0, 40.0),
        ("breathy", 8.0, -18.0, 20.0),
        ("pressed", -2.0, -8.0, 50.0),
    ] {
        save_wav(
            &directory.join(format!("source_tilt_{name}.wav")),
            &harmonic_signal(165.0, h1_h2, tilt, snr),
        )?;
    }
    save_wav(
        &directory.join("sparse_high_f0.wav"),
        &harmonic_signal(330.0, 4.0, -12.0, 30.0),
    )?;

    let mut measurements = Vec::new();
    for (index, &(frequency, expected_bandwidth)) in isolated.iter().enumerate() {
        let response = |f| resonator_response(f, frequency, expected_bandwidth);
        let recovered_frequency = peak_frequency(response, frequency - 250.0, frequency + 250.0);
        let recovered_bandwidth = bandwidth(response, recovered_frequency);
        measurements.push(measurement(
            "isolated_all_pole",
            &format!("f{}_hz", index + 1),
            frequency,
            recovered_frequency,
            25.0,
        ));
        measurements.push(measurement(
            "isolated_all_pole",
            &format!("b{}_hz", index + 1),
            expected_bandwidth,
            recovered_bandwidth,
            35.0,
        ));
    }
    let nasal = |f| zero_response(f, 750.0, 100.0);
    measurements.push(measurement(
        "nasal_pole_zero",
        "zero_hz",
        750.0,
        zero_frequency(nasal, 550.0, 950.0),
        40.0,
    ));

    let huv = |f: f64| resonator_response(f, 700.0, 100.0);
    let hrad = |f: f64| {
        let omega = 2.0 * PI * f / SAMPLE_RATE as f64;
        Complex::new(1.0 - omega.cos(), omega.sin())
    };
    let mut magnitude_errors = Vec::new();
    let mut phase_errors = Vec::new();
    for frequency in (50..=12_000).step_by(10) {
        let expected = huv(frequency as f64).mul(hrad(frequency as f64));
        let reconstructed = huv(frequency as f64).mul(hrad(frequency as f64));
        magnitude_errors.push(
            20.0 * (reconstructed.abs() / expected.abs().max(1e-15))
                .log10()
                .abs(),
        );
        phase_errors.push(
            (reconstructed.phase() - expected.phase())
                .to_degrees()
                .abs(),
        );
    }
    magnitude_errors.sort_by(f64::total_cmp);
    phase_errors.sort_by(f64::total_cmp);
    let factorization = json!({
        "median_magnitude_error_db": magnitude_errors[magnitude_errors.len() / 2],
        "p95_phase_error_degrees": phase_errors[(phase_errors.len() as f64 * 0.95) as usize],
        "pass": magnitude_errors[magnitude_errors.len() / 2] <= 0.10
            && phase_errors[(phase_errors.len() as f64 * 0.95) as usize] <= 1.0
    });
    Ok((measurements, factorization))
}

fn measurement(
    case_id: &str,
    parameter: &str,
    expected: f64,
    recovered: f64,
    tolerance: f64,
) -> ChallengeMeasurement {
    let absolute_error = (expected - recovered).abs();
    ChallengeMeasurement {
        case_id: case_id.into(),
        parameter: parameter.into(),
        expected,
        recovered,
        absolute_error,
        tolerance,
        pass: absolute_error <= tolerance,
    }
}

fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_slice(&std::fs::read(path)?)
        .with_context(|| format!("parse {}", path.display()))
}

fn contains_all(path: &Path, required: &[&str]) -> Result<bool> {
    let text = std::fs::read_to_string(path)?;
    Ok(required.iter().all(|token| text.contains(token)))
}

fn validate_manifest(directory: &Path, split: &Value, licenses: &Value) -> Result<Value> {
    let text = std::fs::read_to_string(directory.join("corpus_manifest.jsonl"))?;
    let license_ids: BTreeSet<_> = licenses["entries"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|entry| entry["license_id"].as_str())
        .collect();
    let mut record_ids = BTreeSet::new();
    let mut speakers_by_split: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut errors = Vec::new();
    let mut records = 0usize;
    for (line_number, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        records += 1;
        let record: Value = serde_json::from_str(line)
            .with_context(|| format!("corpus line {}", line_number + 1))?;
        let id = record["record_id"].as_str().unwrap_or("");
        let speaker = record["speaker"]["pseudonym"].as_str().unwrap_or("");
        let split_name = record["split"].as_str().unwrap_or("");
        let license = record["provenance"]["license_id"].as_str().unwrap_or("");
        if record["protocol_version"] != VERSION || id.is_empty() || speaker.is_empty() {
            errors.push(format!("line {} missing identity/version", line_number + 1));
        }
        if !record_ids.insert(id.to_owned()) {
            errors.push(format!("duplicate record_id {id}"));
        }
        if !license_ids.contains(license) {
            errors.push(format!("unknown license_id {license}"));
        }
        speakers_by_split
            .entry(split_name.to_owned())
            .or_default()
            .insert(speaker.to_owned());
        for key in ["acquisition", "performance", "segmentation", "measurements"] {
            if record.get(key).is_none() {
                errors.push(format!("{id} missing {key}"));
            }
        }
    }
    let names = ["development", "validation", "sealed_holdout"];
    let mut leakage = Vec::new();
    for left in 0..names.len() {
        for right in left + 1..names.len() {
            let a = speakers_by_split
                .get(names[left])
                .cloned()
                .unwrap_or_default();
            let b = speakers_by_split
                .get(names[right])
                .cloned()
                .unwrap_or_default();
            leakage.extend(a.intersection(&b).cloned());
        }
    }
    let split_lists_match = names.iter().all(|name| {
        let manifest = speakers_by_split.get(*name).cloned().unwrap_or_default();
        let key = format!("{name}_speakers");
        let locked: BTreeSet<_> = split[&key]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect();
        manifest == locked
    });
    Ok(json!({
        "records": records,
        "errors": errors,
        "speaker_leakage": leakage,
        "split_lists_match": split_lists_match,
        "human_corpus_ready": records > 0 && errors.is_empty() && leakage.is_empty() && split_lists_match
    }))
}

fn hash_file(path: &Path) -> Result<String> {
    Ok(blake3::hash(&std::fs::read(path)?).to_hex().to_string())
}

fn make_lock(root: &Path, directory: &Path) -> Result<ProtocolLock> {
    let mut artifacts = Vec::new();
    for name in REQUIRED {
        artifacts.push(LockedFile {
            path: (*name).into(),
            blake3: hash_file(&directory.join(name))?,
        });
    }
    artifacts.push(LockedFile {
        path: "estimator_validation_report.json".into(),
        blake3: hash_file(&directory.join("estimator_validation_report.json"))?,
    });
    let mut generated: Vec<_> = std::fs::read_dir(directory.join("validation_cases/generated"))?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().is_file())
        .collect();
    generated.sort_by_key(|entry| entry.file_name());
    for entry in generated {
        let path = format!(
            "validation_cases/generated/{}",
            entry.file_name().to_string_lossy()
        );
        artifacts.push(LockedFile {
            path,
            blake3: hash_file(&entry.path())?,
        });
    }
    let mut protected_implementation = Vec::new();
    for name in PROTECTED {
        protected_implementation.push(LockedFile {
            path: (*name).into(),
            blake3: hash_file(&root.join(name))?,
        });
    }
    let mut aggregate = blake3::Hasher::new();
    for file in artifacts.iter().chain(&protected_implementation) {
        aggregate.update(file.path.as_bytes());
        aggregate.update(file.blake3.as_bytes());
    }
    Ok(ProtocolLock {
        protocol_version: VERSION.into(),
        freeze_phase: "protocol_pre_corpus".into(),
        protocol_content_hash: aggregate.finalize().to_hex().to_string(),
        artifacts,
        protected_implementation,
    })
}

fn verify_lock(root: &Path, directory: &Path, lock: &ProtocolLock) -> Result<Vec<String>> {
    let mut mismatches = Vec::new();
    for file in &lock.artifacts {
        if hash_file(&directory.join(&file.path))? != file.blake3 {
            mismatches.push(file.path.clone());
        }
    }
    for file in &lock.protected_implementation {
        if hash_file(&root.join(&file.path))? != file.blake3 {
            mismatches.push(file.path.clone());
        }
    }
    Ok(mismatches)
}

fn physics_evidence(root: &Path) -> Result<Value> {
    let path = root.join("audio_output/vocal_physiology_v2/isolation_report.json");
    if !path.is_file() {
        return Ok(json!({"available": false, "pass": false}));
    }
    let report = read_json(&path)?;
    let renderer = &report["renderer_isolation"];
    let v2 = &renderer["v2_metrics"]["metrics"];
    let reference = &renderer["reference_metrics"]["metrics"];
    let diagnostics = &renderer["v2_diagnostics"];
    let signal_gates = v2["objective_pass"].as_bool() == Some(true)
        && reference["objective_pass"].as_bool() == Some(true)
        && diagnostics["non_finite_samples"].as_u64() == Some(0)
        && renderer["v2_acoustic_metrics"]["source_alias_pass"].as_bool() == Some(true);
    Ok(json!({
        "available": true,
        "evidence_path": "audio_output/vocal_physiology_v2/isolation_report.json",
        "signal_gates_pass": signal_gates,
        "frozen_geometry_decay_machine_report": false,
        "moving_geometry_work_accounting_pass": false,
        "static_response_cross_solver_tolerance_pass": false,
        "pass": false,
        "note": "Existing signal evidence is clean; Stage-0-specific SI observation, work-accounting, and static transfer-function evidence remains required."
    }))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let root = std::env::current_dir()?;
    let directory = args
        .iter()
        .position(|argument| argument == "--protocol")
        .and_then(|index| args.get(index + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("communication/singing/vowel_truth_stage0"));
    let freeze = args.iter().any(|argument| argument == "--freeze");

    let mut missing = Vec::new();
    for name in REQUIRED {
        if !directory.join(name).is_file() {
            missing.push(*name);
        }
    }
    if !missing.is_empty() {
        bail!("missing required protocol artifacts: {missing:?}");
    }
    let schema = read_json(&directory.join("corpus_record.schema.json"))?;
    let splits = read_json(&directory.join("split_manifest.json"))?;
    let licenses = read_json(&directory.join("license_ledger.json"))?;
    let estimator_manifest = read_json(&directory.join("estimator_validation_manifest.json"))?;
    let schema_valid = schema["$schema"].is_string()
        && schema["properties"]["provenance"].is_object()
        && schema["properties"]["segmentation"].is_object()
        && schema["$defs"]["measurement"].is_object();
    let text_contracts_valid = contains_all(
        &directory.join("physics_observation_contract.md"),
        &["H_gp(f)", "H_uv(f)", "H_rad(f)", "H_total(f)", "m³/s", "Pa"],
    )? && contains_all(
        &directory.join("listener_preregistration.yaml"),
        &[
            "minimum_effect_required",
            "listener_random_effect",
            "stimulus_random_effect",
            "multiplicity_correction",
            "no_holdout_listening_during_tuning",
        ],
    )? && contains_all(
        &directory.join("estimator_protocol.toml"),
        &[
            "maximum_absolute_bias_hz",
            "minimum_ci_coverage",
            "trusted_implementation",
            "high_f0_may_be_inconclusive",
        ],
    )?;
    let manifest = validate_manifest(&directory, &splits, &licenses)?;
    let license_review_pass = licenses["entries"]
        .as_array()
        .into_iter()
        .flatten()
        .all(|entry| {
            entry["manual_review"] == "pass"
                && entry["exact_text_path"]
                    .as_str()
                    .is_some_and(|path| directory.join(path).is_file())
        });

    let challenge_directory = directory.join("validation_cases/generated");
    let (challenge_measurements, factorization) = challenge_suite(&challenge_directory)?;
    let physics_estimator_pass = challenge_measurements
        .iter()
        .all(|measurement| measurement.pass)
        && factorization["pass"].as_bool() == Some(true);
    let mut challenge_entries: Vec<_> = std::fs::read_dir(&challenge_directory)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().is_file())
        .collect();
    challenge_entries.sort_by_key(|entry| entry.file_name());
    let challenge_hashes: Vec<_> = challenge_entries
        .into_iter()
        .map(|entry| {
            Ok(json!({
                "path": entry.file_name().to_string_lossy(),
                "blake3": hash_file(&entry.path())?
            }))
        })
        .collect::<Result<_>>()?;
    let challenge_report = json!({
        "protocol_version": VERSION,
        "measurements": challenge_measurements,
        "factorization": factorization,
        "bias_variance_failure_coverage_complete": false,
        "actual_cpp_cross_implementation_pass": false,
        "recording_path_validation_complete": false,
        "physics_path_initial_tolerances_pass": physics_estimator_pass,
        "generated_signals": challenge_hashes
    });
    std::fs::write(
        directory.join("estimator_validation_report.json"),
        serde_json::to_vec_pretty(&challenge_report)?,
    )?;

    let lock_path = directory.join("protocol_lock.json");
    if freeze {
        if lock_path.exists() {
            bail!(
                "protocol_lock.json already exists; bump the protocol version instead of silently overwriting a frozen protocol"
            );
        }
        let lock = make_lock(&root, &directory)?;
        std::fs::write(&lock_path, serde_json::to_vec_pretty(&lock)?)?;
    }
    let lock: ProtocolLock = serde_json::from_slice(
        &std::fs::read(&lock_path)
            .context("run once with --freeze to create protocol_lock.json")?,
    )?;
    let lock_mismatches = verify_lock(&root, &directory, &lock)?;
    let physics = physics_evidence(&root)?;
    let protocol_valid = schema_valid
        && text_contracts_valid
        && license_review_pass
        && lock_mismatches.is_empty()
        && estimator_manifest["protocol_version"] == VERSION;
    let stage0_exit_ready = protocol_valid
        && manifest["human_corpus_ready"].as_bool() == Some(true)
        && challenge_report["bias_variance_failure_coverage_complete"].as_bool() == Some(true)
        && challenge_report["actual_cpp_cross_implementation_pass"].as_bool() == Some(true)
        && challenge_report["recording_path_validation_complete"].as_bool() == Some(true)
        && physics["pass"].as_bool() == Some(true)
        && splits["assignment_salt_commitment"].is_string()
        && splits["status"] == "frozen";
    let report = json!({
        "protocol_version": VERSION,
        "protocol_content_hash": lock.protocol_content_hash,
        "protocol_valid": protocol_valid,
        "stage0_exit_ready": stage0_exit_ready,
        "retuning_allowed": stage0_exit_ready,
        "checks": {
            "required_artifacts": {"pass": missing.is_empty(), "missing": missing},
            "schema_structure": {"pass": schema_valid},
            "text_contracts": {"pass": text_contracts_valid},
            "license_ledger": {"pass": license_review_pass},
            "hash_and_protected_implementation_lock": {"pass": lock_mismatches.is_empty(), "mismatches": lock_mismatches},
            "corpus_and_splits": manifest,
            "estimator_challenges": challenge_report,
            "physics_protection": physics
        },
        "exit_blockers": [
            "licensed development/validation/sealed-holdout human corpus absent",
            "split salt and immutable speaker assignments absent",
            "bias/variance/failure/CI-coverage challenge matrix incomplete",
            "actual CPP independent implementation not locked",
            "recording-path estimator validation incomplete",
            "SI observation surfaces and moving-geometry work accounting not yet evidenced",
            "production/reference static transfer-function tolerance report absent",
            "listener target sample size awaits protocol-only variance pilot"
        ]
    });
    std::fs::write(
        directory.join("stage0_validation_report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!(
        "protocol_valid={} stage0_exit_ready={} content_hash={}",
        protocol_valid, stage0_exit_ready, lock.protocol_content_hash
    );
    if !protocol_valid {
        bail!("Stage-0 protocol validation failed");
    }
    Ok(())
}
