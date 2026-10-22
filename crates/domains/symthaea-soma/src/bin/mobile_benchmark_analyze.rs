// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Offline analyzer for paired sovereign mobile-device benchmark runs.
//!
//! Usage:
//!   cargo run -p symthaea-soma --release --bin mobile-benchmark-analyze -- experiment.json
//!   cargo run -p symthaea-soma --release --bin mobile-benchmark-analyze -- --exploratory experiment.json
//!
//! Standard qualification mode requires at least 30 paired observations per scenario.
//! Exploratory mode permits smaller samples for pipeline debugging, but can never
//! report a declared gate pass. Synthetic data is always diagnostic, never qualification
//! evidence. Energy gates are only evaluable for physical-device runs using calibrated
//! instrumentation and a non-empty calibration identity.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::process;

const MIN_REPETITIONS_PER_SCENARIO: usize = 30;
const EXPLORATORY_MIN_REPETITIONS_PER_SCENARIO: usize = 2;
const DEFAULT_BOOTSTRAP_REPETITIONS: usize = 10_000;
const DEFAULT_SEED: u64 = 0x534F_4D41_4D4F_4231;
const ENERGY_SAVINGS_TARGET: f64 = 0.20;
const MAX_SUCCESS_RATE_DROP: f64 = 0.02;
const MAX_P95_LATENCY_REGRESSION: f64 = 0.10;

#[derive(Debug, Clone, Deserialize)]
struct ExperimentRecord {
    schema_version: String,
    experiment_id: String,
    measurement_origin: String,
    scenario_corpus_sha256: String,
    scoring_code_sha256: String,
    raw_samples_sha256: String,
    baseline_config_sha256: String,
    candidate_config_sha256: String,
    device: DeviceInfo,
    samples: Vec<PairedSample>,
}

#[derive(Debug, Clone, Deserialize)]
struct DeviceInfo {
    model: String,
    soc: String,
    os_build: String,
    unit_token: String,
    runtime_id: String,
    application_build_sha256: String,
    energy_measurement_method: String,
    calibration_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct PairedSample {
    pair_id: String,
    scenario_id: String,
    baseline: ArmMeasurement,
    candidate: ArmMeasurement,
}

#[derive(Debug, Clone, Deserialize)]
struct ArmMeasurement {
    energy_joules: f64,
    success: bool,
    latency_ms: f64,
    severe_thermal_seconds: f64,
    privacy_violation_count: u32,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct ArmSummary {
    attempts: usize,
    successful_tasks: usize,
    success_rate: f64,
    total_energy_joules: f64,
    energy_joules_per_success: f64,
    p95_latency_ms: f64,
    total_severe_thermal_seconds: f64,
    privacy_violation_count: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct GateReport {
    physical_measurement_eligible: bool,
    /// The analyzer consumes metadata only; it does not authenticate receipts or meter records.
    evidence_receipt_verified: bool,
    energy_gate_evaluable: bool,
    energy_gate_pass: bool,
    utility_gate_pass: bool,
    p95_latency_gate_pass: bool,
    severe_thermal_gate_pass: bool,
    privacy_gate_pass: bool,
    /// All declared gates passed; this is not authenticated physical qualification.
    declared_gates_pass: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct AnalysisReport {
    schema_version: String,
    report_type: String,
    experiment_id: String,
    measurement_origin: String,
    device_model: String,
    device_soc: String,
    device_os_build: String,
    device_unit_token: String,
    runtime_id: String,
    application_build_sha256: String,
    energy_measurement_method: String,
    calibration_id: Option<String>,
    scenario_corpus_sha256: String,
    scoring_code_sha256: String,
    raw_samples_sha256: String,
    baseline_config_sha256: String,
    candidate_config_sha256: String,
    paired_sample_count: usize,
    scenario_count: usize,
    exploratory: bool,
    bootstrap_repetitions_requested: usize,
    bootstrap_repetitions_valid: usize,
    bootstrap_seed: u64,
    baseline: ArmSummary,
    candidate: ArmSummary,
    energy_savings_fraction: f64,
    energy_savings_ci95: [f64; 2],
    success_rate_difference_fraction: f64,
    success_rate_difference_percentage_points: f64,
    p95_latency_regression_fraction: f64,
    severe_thermal_seconds_difference: f64,
    privacy_violations_total: u64,
    gates: GateReport,
}

fn non_empty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_experiment(
    experiment: &ExperimentRecord,
    minimum_repetitions_per_scenario: usize,
) -> Result<BTreeMap<String, Vec<usize>>, String> {
    if experiment.schema_version != "1.0" {
        return Err(format!(
            "unsupported schema_version {:?}; expected 1.0",
            experiment.schema_version
        ));
    }
    if !non_empty(&experiment.experiment_id) {
        return Err("experiment_id must be non-empty".into());
    }

    for (name, value) in [
        ("scenario_corpus_sha256", &experiment.scenario_corpus_sha256),
        ("scoring_code_sha256", &experiment.scoring_code_sha256),
        ("raw_samples_sha256", &experiment.raw_samples_sha256),
        ("baseline_config_sha256", &experiment.baseline_config_sha256),
        ("candidate_config_sha256", &experiment.candidate_config_sha256),
    ] {
        if !is_sha256(value) {
            return Err(format!("{name} must be a 64-character hexadecimal SHA-256"));
        }
    }

    for (name, value) in [
        ("device.model", &experiment.device.model),
        ("device.soc", &experiment.device.soc),
        ("device.os_build", &experiment.device.os_build),
        ("device.unit_token", &experiment.device.unit_token),
        ("device.runtime_id", &experiment.device.runtime_id),
    ] {
        if !non_empty(value) {
            return Err(format!("{name} must be non-empty"));
        }
    }
    if !is_sha256(&experiment.device.application_build_sha256) {
        return Err("device.application_build_sha256 must be a 64-character hexadecimal SHA-256".into());
    }

    match experiment.measurement_origin.as_str() {
        "physical_device" | "synthetic_fixture" => {}
        other => {
            return Err(format!(
                "measurement_origin must be physical_device or synthetic_fixture, got {other:?}"
            ));
        }
    }
    match experiment.device.energy_measurement_method.as_str() {
        "calibrated_instrument" | "os_power_proxy" | "battery_percentage_estimate"
        | "synthetic_fixture" => {}
        other => {
            return Err(format!("unsupported energy_measurement_method {other:?}"));
        }
    }
    if experiment.measurement_origin == "synthetic_fixture"
        && experiment.device.energy_measurement_method != "synthetic_fixture"
    {
        return Err("synthetic_fixture origin must use synthetic_fixture measurement method".into());
    }
    if experiment.measurement_origin == "physical_device"
        && experiment.device.energy_measurement_method == "synthetic_fixture"
    {
        return Err("physical_device origin cannot use synthetic_fixture measurement method".into());
    }
    if experiment.device.energy_measurement_method == "calibrated_instrument"
        && experiment
            .device
            .calibration_id
            .as_deref()
            .is_none_or(|value| !non_empty(value))
    {
        return Err("calibrated_instrument requires a non-empty calibration_id".into());
    }

    if experiment.samples.is_empty() {
        return Err("samples must contain at least one paired observation".into());
    }

    let mut pair_ids = BTreeSet::new();
    let mut strata: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut baseline_successes = 0usize;
    let mut candidate_successes = 0usize;
    for (index, sample) in experiment.samples.iter().enumerate() {
        if !non_empty(&sample.pair_id) {
            return Err(format!("samples[{index}].pair_id must be non-empty"));
        }
        if !pair_ids.insert(sample.pair_id.as_str()) {
            return Err(format!("duplicate pair_id {:?}", sample.pair_id));
        }
        if !non_empty(&sample.scenario_id) {
            return Err(format!("samples[{index}].scenario_id must be non-empty"));
        }

        for (arm_name, arm) in [("baseline", &sample.baseline), ("candidate", &sample.candidate)] {
            if !arm.energy_joules.is_finite() || arm.energy_joules <= 0.0 {
                return Err(format!(
                    "samples[{index}].{arm_name}.energy_joules must be finite and > 0"
                ));
            }
            if !arm.latency_ms.is_finite() || arm.latency_ms <= 0.0 {
                return Err(format!(
                    "samples[{index}].{arm_name}.latency_ms must be finite and > 0"
                ));
            }
            if !arm.severe_thermal_seconds.is_finite() || arm.severe_thermal_seconds < 0.0 {
                return Err(format!(
                    "samples[{index}].{arm_name}.severe_thermal_seconds must be finite and >= 0"
                ));
            }
        }
        baseline_successes += if sample.baseline.success { 1 } else { 0 };
        candidate_successes += if sample.candidate.success { 1 } else { 0 };
        strata
            .entry(sample.scenario_id.clone())
            .or_default()
            .push(index);
    }

    if baseline_successes == 0 {
        return Err("baseline has zero successful tasks; energy per successful task is undefined".into());
    }
    if candidate_successes == 0 {
        return Err("candidate has zero successful tasks; energy per successful task is undefined".into());
    }

    for (scenario_id, indices) in &strata {
        if indices.len() < minimum_repetitions_per_scenario {
            return Err(format!(
                "scenario {scenario_id:?} has {} paired observations; needs at least {}",
                indices.len(),
                minimum_repetitions_per_scenario
            ));
        }
    }
    Ok(strata)
}

fn nearest_rank_percentile(sorted_values: &[f64], quantile: f64) -> Result<f64, String> {
    if sorted_values.is_empty() {
        return Err("cannot calculate percentile of an empty sample".into());
    }
    if !quantile.is_finite() || !(0.0..=1.0).contains(&quantile) {
        return Err("quantile must be finite and in [0, 1]".into());
    }
    let rank = ((quantile * sorted_values.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(sorted_values.len() - 1);
    Ok(sorted_values[rank])
}

fn summarize_arm(
    selected: &[&PairedSample],
    candidate: bool,
) -> Result<ArmSummary, String> {
    if selected.is_empty() {
        return Err("cannot summarize an empty sample".into());
    }
    let mut successful_tasks = 0usize;
    let mut total_energy_joules = 0.0f64;
    let mut total_severe_thermal_seconds = 0.0f64;
    let mut privacy_violation_count = 0u64;
    let mut latencies = Vec::with_capacity(selected.len());

    for paired in selected {
        let arm = if candidate {
            &paired.candidate
        } else {
            &paired.baseline
        };
        total_energy_joules += arm.energy_joules;
        successful_tasks += if arm.success { 1 } else { 0 };
        total_severe_thermal_seconds += arm.severe_thermal_seconds;
        privacy_violation_count += u64::from(arm.privacy_violation_count);
        latencies.push(arm.latency_ms);
    }

    if successful_tasks == 0 {
        return Err("energy per successful task is undefined when successful task count is zero".into());
    }
    if !total_energy_joules.is_finite() || total_energy_joules <= 0.0 {
        return Err("total energy must be finite and > 0".into());
    }

    latencies.sort_by(f64::total_cmp);
    Ok(ArmSummary {
        attempts: selected.len(),
        successful_tasks,
        success_rate: successful_tasks as f64 / selected.len() as f64,
        total_energy_joules,
        energy_joules_per_success: total_energy_joules / successful_tasks as f64,
        p95_latency_ms: nearest_rank_percentile(&latencies, 0.95)?,
        total_severe_thermal_seconds,
        privacy_violation_count,
    })
}

fn scenario_strata(samples: &[PairedSample]) -> BTreeMap<String, Vec<usize>> {
    let mut strata: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (index, sample) in samples.iter().enumerate() {
        strata
            .entry(sample.scenario_id.clone())
            .or_default()
            .push(index);
    }
    strata
}

#[derive(Debug, Clone)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { DEFAULT_SEED } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64*; deterministic resampling, not cryptographic randomness.
        let mut value = self.state;
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        self.state = value;
        value.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn index(&mut self, upper_exclusive: usize) -> usize {
        // Multiply-high mapping avoids the modulo bias from value % upper_exclusive.
        ((u128::from(self.next_u64()) * upper_exclusive as u128) >> 64) as usize
    }
}

fn stratified_resample(
    strata: &BTreeMap<String, Vec<usize>>,
    rng: &mut DeterministicRng,
) -> Vec<usize> {
    let mut selected = Vec::new();
    for indices in strata.values() {
        for _ in 0..indices.len() {
            selected.push(indices[rng.index(indices.len())]);
        }
    }
    selected
}

fn bootstrap_energy_savings(
    experiment: &ExperimentRecord,
    strata: &BTreeMap<String, Vec<usize>>,
    repetitions: usize,
    seed: u64,
) -> Result<Vec<f64>, String> {
    if repetitions < 100 {
        return Err("bootstrap repetitions must be >= 100".into());
    }
    let mut rng = DeterministicRng::new(seed);
    let mut distribution = Vec::with_capacity(repetitions);
    let attempt_limit = repetitions.saturating_mul(20).max(repetitions);
    let mut attempts = 0usize;

    while distribution.len() < repetitions && attempts < attempt_limit {
        attempts += 1;
        let indices = stratified_resample(strata, &mut rng);
        let selected: Vec<&PairedSample> = indices
            .iter()
            .map(|&index| &experiment.samples[index])
            .collect();
        let baseline = match summarize_arm(&selected, false) {
            Ok(summary) => summary,
            Err(_) => continue,
        };
        let candidate = match summarize_arm(&selected, true) {
            Ok(summary) => summary,
            Err(_) => continue,
        };
        if baseline.energy_joules_per_success <= 0.0 {
            continue;
        }
        let savings =
            1.0 - candidate.energy_joules_per_success / baseline.energy_joules_per_success;
        if savings.is_finite() {
            distribution.push(savings);
        }
    }

    if distribution.len() < repetitions {
        return Err(format!(
            "could produce only {} valid bootstrap replicates out of {repetitions}; success rates may be too low",
            distribution.len()
        ));
    }
    distribution.sort_by(f64::total_cmp);
    Ok(distribution)
}

fn analyze(
    experiment: &ExperimentRecord,
    bootstrap_repetitions: usize,
    seed: u64,
    exploratory: bool,
) -> Result<AnalysisReport, String> {
    let min_per_scenario = if exploratory {
        EXPLORATORY_MIN_REPETITIONS_PER_SCENARIO
    } else {
        MIN_REPETITIONS_PER_SCENARIO
    };
    let strata = validate_experiment(experiment, min_per_scenario)?;
    let selected: Vec<&PairedSample> = experiment.samples.iter().collect();
    let baseline = summarize_arm(&selected, false)?;
    let candidate = summarize_arm(&selected, true)?;

    let observed_savings =
        1.0 - candidate.energy_joules_per_success / baseline.energy_joules_per_success;
    let distribution = bootstrap_energy_savings(
        experiment,
        &strata,
        bootstrap_repetitions,
        seed,
    )?;
    let ci_low = nearest_rank_percentile(&distribution, 0.025)?;
    let ci_high = nearest_rank_percentile(&distribution, 0.975)?;
    let success_rate_difference = candidate.success_rate - baseline.success_rate;
    let latency_regression =
        candidate.p95_latency_ms / baseline.p95_latency_ms - 1.0;
    let thermal_difference =
        candidate.total_severe_thermal_seconds - baseline.total_severe_thermal_seconds;
    let privacy_violations_total =
        baseline.privacy_violation_count + candidate.privacy_violation_count;

    let physical_measurement_eligible =
        experiment.measurement_origin == "physical_device";
    let energy_gate_evaluable = physical_measurement_eligible
        && experiment.device.energy_measurement_method == "calibrated_instrument"
        && experiment
            .device
            .calibration_id
            .as_deref()
            .is_some_and(non_empty);
    let energy_gate_pass = energy_gate_evaluable
        && observed_savings >= ENERGY_SAVINGS_TARGET
        && ci_low > 0.0;
    let utility_gate_pass = success_rate_difference >= -MAX_SUCCESS_RATE_DROP;
    let p95_latency_gate_pass = latency_regression <= MAX_P95_LATENCY_REGRESSION;
    let severe_thermal_gate_pass = thermal_difference <= 0.0;
    let privacy_gate_pass = privacy_violations_total == 0;
    let declared_gates_pass = !exploratory
        && energy_gate_evaluable
        && energy_gate_pass
        && utility_gate_pass
        && p95_latency_gate_pass
        && severe_thermal_gate_pass
        && privacy_gate_pass;

    Ok(AnalysisReport {
        schema_version: "1.0".into(),
        report_type: "mobile_device_paired_analysis".into(),
        experiment_id: experiment.experiment_id.clone(),
        measurement_origin: experiment.measurement_origin.clone(),
        device_model: experiment.device.model.clone(),
        device_soc: experiment.device.soc.clone(),
        device_os_build: experiment.device.os_build.clone(),
        device_unit_token: experiment.device.unit_token.clone(),
        runtime_id: experiment.device.runtime_id.clone(),
        application_build_sha256: experiment.device.application_build_sha256.clone(),
        energy_measurement_method: experiment.device.energy_measurement_method.clone(),
        calibration_id: experiment.device.calibration_id.clone(),
        scenario_corpus_sha256: experiment.scenario_corpus_sha256.clone(),
        scoring_code_sha256: experiment.scoring_code_sha256.clone(),
        raw_samples_sha256: experiment.raw_samples_sha256.clone(),
        baseline_config_sha256: experiment.baseline_config_sha256.clone(),
        candidate_config_sha256: experiment.candidate_config_sha256.clone(),
        paired_sample_count: experiment.samples.len(),
        scenario_count: strata.len(),
        exploratory,
        bootstrap_repetitions_requested: bootstrap_repetitions,
        bootstrap_repetitions_valid: distribution.len(),
        bootstrap_seed: seed,
        baseline,
        candidate,
        energy_savings_fraction: observed_savings,
        energy_savings_ci95: [ci_low, ci_high],
        success_rate_difference_fraction: success_rate_difference,
        success_rate_difference_percentage_points: success_rate_difference * 100.0,
        p95_latency_regression_fraction: latency_regression,
        severe_thermal_seconds_difference: thermal_difference,
        privacy_violations_total,
        gates: GateReport {
            physical_measurement_eligible,
            evidence_receipt_verified: false,
            energy_gate_evaluable,
            energy_gate_pass,
            utility_gate_pass,
            p95_latency_gate_pass,
            severe_thermal_gate_pass,
            privacy_gate_pass,
            declared_gates_pass,
        },
    })
}

fn print_help() {
    println!(
        "mobile-benchmark-analyze

Analyze paired observations from the Sovereign Personal Device benchmark.

USAGE:
  mobile-benchmark-analyze [OPTIONS] <experiment.json>

OPTIONS:
  --bootstrap-reps N  Bootstrap replicates (default: 10000; minimum: 100)
  --seed N            Deterministic bootstrap seed
  --exploratory       Allow 2+ observations per scenario; declared_gates_pass is always false
  -h, --help          Print this help

Standard qualification requires >=30 paired observations per scenario.
Energy qualification requires physical-device data, calibrated instrumentation,
and a non-empty calibration_id. Synthetic/proxy measurements cannot qualify."
    );
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let mut path: Option<String> = None;
    let mut bootstrap_repetitions = DEFAULT_BOOTSTRAP_REPETITIONS;
    let mut seed = DEFAULT_SEED;
    let mut exploratory = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_help();
                return Ok(());
            }
            "--exploratory" => exploratory = true,
            "--bootstrap-reps" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--bootstrap-reps requires a value".to_string())?;
                bootstrap_repetitions = value
                    .parse::<usize>()
                    .map_err(|_| "--bootstrap-reps must be a positive integer".to_string())?;
                if !(100..=1_000_000).contains(&bootstrap_repetitions) {
                    return Err("--bootstrap-reps must be between 100 and 1000000".into());
                }
            }
            "--seed" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--seed requires a value".to_string())?;
                seed = value
                    .parse::<u64>()
                    .map_err(|_| "--seed must be an unsigned integer".to_string())?;
            }
            _ if arg.starts_with('-') => {
                return Err(format!("unknown option {arg:?}"));
            }
            _ => {
                if path.replace(arg).is_some() {
                    return Err("provide exactly one experiment JSON path".into());
                }
            }
        }
    }

    let path = path.ok_or_else(|| "missing experiment JSON path (try --help)".to_string())?;
    let input = fs::read_to_string(&path)
        .map_err(|error| format!("failed to read {path:?}: {error}"))?;
    let experiment: ExperimentRecord = serde_json::from_str(&input)
        .map_err(|error| format!("invalid experiment JSON: {error}"))?;
    let report = analyze(&experiment, bootstrap_repetitions, seed, exploratory)?;
    let json = serde_json::to_string_pretty(&report)
        .map_err(|error| format!("failed to serialize report: {error}"))?;
    println!("{json}");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mobile-benchmark-analyze: {error}");
        process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn experiment(
        measurement_origin: &str,
        energy_measurement_method: &str,
        repeats_per_scenario: usize,
        scenario_count: usize,
    ) -> ExperimentRecord {
        let samples = (0..scenario_count)
            .flat_map(|scenario_index| {
                (0..repeats_per_scenario).map(move |repeat| {
                    let baseline_energy = 4.0 + (repeat % 5) as f64 * 0.02;
                    let candidate_energy = baseline_energy * 0.70;
                    PairedSample {
                        pair_id: format!("pair-{scenario_index}-{repeat}"),
                        scenario_id: format!("scenario-{scenario_index}"),
                        baseline: ArmMeasurement {
                            energy_joules: baseline_energy,
                            success: true,
                            latency_ms: 100.0 + (repeat % 4) as f64,
                            severe_thermal_seconds: if repeat % 7 == 0 { 0.2 } else { 0.0 },
                            privacy_violation_count: 0,
                        },
                        candidate: ArmMeasurement {
                            energy_joules: candidate_energy,
                            success: true,
                            latency_ms: 101.0 + (repeat % 4) as f64,
                            severe_thermal_seconds: if repeat % 7 == 0 { 0.1 } else { 0.0 },
                            privacy_violation_count: 0,
                        },
                    }
                })
            })
            .collect();

        ExperimentRecord {
            schema_version: "1.0".into(),
            experiment_id: "test-experiment".into(),
            measurement_origin: measurement_origin.into(),
            scenario_corpus_sha256: sha('a'),
            scoring_code_sha256: sha('b'),
            raw_samples_sha256: sha('c'),
            baseline_config_sha256: sha('d'),
            candidate_config_sha256: sha('e'),
            device: DeviceInfo {
                model: "test-device".into(),
                soc: "test-soc".into(),
                os_build: "test-os".into(),
                unit_token: "unit-01".into(),
                runtime_id: "test-runtime".into(),
                application_build_sha256: sha('f'),
                energy_measurement_method: energy_measurement_method.into(),
                calibration_id: if energy_measurement_method == "calibrated_instrument" {
                    Some("calibration-2026-01".into())
                } else {
                    None
                },
            },
            samples,
        }
    }

    #[test]
    fn percentile_uses_nearest_rank_definition() {
        let values: Vec<f64> = (1..=20).map(|value| value as f64).collect();
        assert_eq!(nearest_rank_percentile(&values, 0.95).unwrap(), 19.0);
        assert_eq!(nearest_rank_percentile(&values, 0.0).unwrap(), 1.0);
    }

    #[test]
    fn resampling_preserves_each_scenario_sample_count() {
        let experiment = experiment("synthetic_fixture", "synthetic_fixture", 5, 3);
        let strata = scenario_strata(&experiment.samples);
        let resampled = stratified_resample(&strata, &mut DeterministicRng::new(17));
        assert_eq!(resampled.len(), experiment.samples.len());
        for indices in strata.values() {
            let selected_set: BTreeSet<usize> = indices.iter().copied().collect();
            let resampled_count = resampled
                .iter()
                .filter(|index| selected_set.contains(*index))
                .count();
            assert_eq!(resampled_count, indices.len());
        }
    }

    #[test]
    fn repeated_analysis_is_deterministic_for_same_seed() {
        let experiment = experiment("physical_device", "calibrated_instrument", 30, 2);
        let first = analyze(&experiment, 500, 91, false).unwrap();
        let second = analyze(&experiment, 500, 91, false).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn synthetic_experiment_can_never_qualify() {
        let experiment = experiment("synthetic_fixture", "synthetic_fixture", 30, 1);
        let report = analyze(&experiment, 500, 3, false).unwrap();
        assert!(!report.gates.physical_measurement_eligible);
        assert!(!report.gates.energy_gate_evaluable);
        assert!(!report.gates.declared_gates_pass);
        assert!(!report.gates.evidence_receipt_verified);
    }

    #[test]
    fn calibrated_energy_win_does_not_override_privacy_failure() {
        let mut experiment = experiment("physical_device", "calibrated_instrument", 30, 1);
        experiment.samples[0].candidate.privacy_violation_count = 1;
        let report = analyze(&experiment, 500, 5, false).unwrap();
        assert!(report.gates.energy_gate_evaluable);
        assert!(report.gates.energy_gate_pass);
        assert!(!report.gates.evidence_receipt_verified);
        assert!(!report.gates.privacy_gate_pass);
        assert!(!report.gates.declared_gates_pass);
    }

    #[test]
    fn zero_success_candidate_is_rejected() {
        let mut experiment = experiment("physical_device", "calibrated_instrument", 30, 1);
        for sample in &mut experiment.samples {
            sample.candidate.success = false;
        }
        let error = analyze(&experiment, 500, 5, false).unwrap_err();
        assert!(error.contains("candidate has zero successful tasks"));
    }

    #[test]
    fn duplicate_pair_id_is_rejected() {
        let mut experiment = experiment("synthetic_fixture", "synthetic_fixture", 3, 1);
        experiment.samples[1].pair_id = experiment.samples[0].pair_id.clone();
        let error = validate_experiment(&experiment, 2).unwrap_err();
        assert!(error.contains("duplicate pair_id"));
    }

    #[test]
    fn official_mode_requires_thirty_repetitions_per_scenario() {
        let experiment = experiment("synthetic_fixture", "synthetic_fixture", 29, 1);
        let error = analyze(&experiment, 500, 7, false).unwrap_err();
        assert!(error.contains("needs at least 30"));
    }

    #[test]
    fn energy_proxy_is_not_eligible_for_energy_claim() {
        let experiment = experiment("physical_device", "os_power_proxy", 30, 1);
        let report = analyze(&experiment, 500, 9, false).unwrap();
        assert!(report.gates.physical_measurement_eligible);
        assert!(!report.gates.energy_gate_evaluable);
        assert!(!report.gates.energy_gate_pass);
        assert!(!report.gates.declared_gates_pass);
    }
}
