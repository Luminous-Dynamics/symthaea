// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Energy Efficiency Measurement Benchmark
//!
//! Measures the computational cost of the HDC+CfC pipeline and compares
//! energy-per-inference against published transformer baselines.
//!
//! Run: `cargo run -p symthaea-physics --example energy_benchmark --release`

#![deny(unsafe_code)]

use serde::Serialize;
use std::fs;
use std::mem;
use std::path::{Path, PathBuf};
use std::time::Instant;
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};
use symthaea_physics::cmod_adapter::{
    CModHdcEncoder, DisruptionLabel, SensorNormalizer, SyntheticConfig, compute_statistics,
    generate_synthetic_data, to_cmod_plasma_sample,
};
use symthaea_physics::fusion_twin::{
    FusionDigitalTwin, PLASMA_HORIZON_LABELS, PLASMA_HORIZONS, PlasmaFepAgent,
    PlasmaMultiScalePredictor,
};

const WARMUP_ITERATIONS: usize = 100;
const ENCODING_SAMPLES: usize = 10_000;
const PREDICTION_ITERATIONS: usize = 1000;

// Illustrative power inputs for scenarios; not measured CPU power.
const POWER_ASSUMPTION_DESKTOP_W: f64 = 65.0;
const POWER_ASSUMPTION_LAPTOP_W: f64 = 15.0;

// Illustrative transformer assumptions; no common measurement protocol.
struct TransformerRef {
    name: &'static str,
    power_watts: f64,
    time_per_inference_ms: f64,
}

const TRANSFORMER_REFS: &[TransformerRef] = &[
    TransformerRef {
        name: "GPT-3 175B",
        power_watts: 355.0,
        time_per_inference_ms: 100.0,
    },
    TransformerRef {
        name: "Llama 7B",
        power_watts: 50.0,
        time_per_inference_ms: 30.0,
    },
    TransformerRef {
        name: "Mistral 7B",
        power_watts: 45.0,
        time_per_inference_ms: 25.0,
    },
];

#[derive(Debug, Clone, Serialize)]
struct ThroughputResult {
    label: String,
    total_operations: usize,
    total_time_ms: f64,
    ops_per_second: f64,
    us_per_op: f64,
    ci_95_lower_us: f64,
    ci_95_upper_us: f64,
}

#[derive(Debug, Clone, Serialize)]
struct MemoryResult {
    component: String,
    size_bytes: usize,
}

#[derive(Debug, Clone, Serialize)]
struct EnergyComparison {
    system: String,
    power_watts: f64,
    time_per_inference_ms: f64,
    energy_per_inference_joules: f64,
    modeled_ratio_vs_gpt3: f64,
    /// Evidence label: all values in this table are power × latency models.
    basis: String,
}

#[derive(Debug, Clone, Serialize)]
struct PackageEnergyMeasurement {
    counter_domains: Vec<String>,
    measurement_seconds: f64,
    package_energy_joules: f64,
    /// Counts timed calls plus the benchmark's warm-up calls.
    executed_calls_including_warmup: usize,
    /// Package energy divided by executed calls; includes background/system energy.
    package_energy_joules_per_call: f64,
    limitation: String,
}

#[derive(Debug, Clone)]
struct PackageEnergyCounter {
    name: String,
    directory: PathBuf,
}

#[derive(Debug, Clone)]
struct EnergyCounterReading {
    name: String,
    energy_uj: u64,
    max_energy_range_uj: u64,
}

#[derive(Debug, Clone, Serialize)]
struct PredictionThroughput {
    label: String,
    horizon_seconds: f64,
    predictions_per_second: f64,
    us_per_prediction: f64,
}

#[derive(Debug, Clone, Serialize)]
struct EnergyReport {
    hdc_dimension: usize,
    encoding_samples: usize,
    prediction_iterations: usize,
    hdc_encoding: ThroughputResult,
    cfc_predictions: Vec<PredictionThroughput>,
    full_pipeline: ThroughputResult,
    package_energy_measurement: Option<PackageEnergyMeasurement>,
    memory_usage: Vec<MemoryResult>,
    energy_comparison_desktop: Vec<EnergyComparison>,
    energy_comparison_laptop: Vec<EnergyComparison>,
    total_wall_clock_seconds: f64,
}

fn measure_with_stats(
    label: &str,
    iterations: usize,
    mut operation: impl FnMut(),
) -> ThroughputResult {
    // Warmup
    for _ in 0..WARMUP_ITERATIONS {
        operation();
    }

    // Collect per-iteration timings for statistics
    let batch_size = iterations.min(100);
    let num_batches = (iterations / batch_size).max(1);
    let mut batch_times_us = Vec::with_capacity(num_batches);

    let total_start = Instant::now();
    for _ in 0..num_batches {
        let batch_start = Instant::now();
        for _ in 0..batch_size {
            operation();
        }
        let batch_elapsed = batch_start.elapsed();
        batch_times_us.push(batch_elapsed.as_nanos() as f64 / 1000.0 / batch_size as f64);
    }
    let total_elapsed = total_start.elapsed();

    let total_ops = num_batches * batch_size;
    let total_time_ms = total_elapsed.as_secs_f64() * 1000.0;
    let ops_per_second = total_ops as f64 / total_elapsed.as_secs_f64();
    let us_per_op = total_elapsed.as_nanos() as f64 / 1000.0 / total_ops as f64;

    // Statistics on batch means
    let n = batch_times_us.len() as f64;
    let mean = batch_times_us.iter().sum::<f64>() / n;
    let variance = if n > 1.0 {
        batch_times_us
            .iter()
            .map(|t| (t - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0)
    } else {
        0.0
    };
    let std_dev = variance.sqrt();
    let ci_margin = 1.96 * std_dev / n.sqrt();

    ThroughputResult {
        label: label.to_string(),
        total_operations: total_ops,
        total_time_ms,
        ops_per_second,
        us_per_op,
        ci_95_lower_us: mean - ci_margin,
        ci_95_upper_us: mean + ci_margin,
    }
}

fn energy_per_inference(tdp_watts: f64, time_ms: f64) -> f64 {
    tdp_watts * time_ms / 1000.0 // joules
}

// Linux powercap/RAPL can expose CPU package energy counters. These readings
// are system-level observations, not perfectly isolated workload attribution.
fn discover_package_energy_counters() -> Vec<PackageEnergyCounter> {
    fn collect(path: &Path, depth: usize, found: &mut Vec<PackageEnergyCounter>) {
        if depth > 3 {
            return;
        }
        if let Ok(name) = fs::read_to_string(path.join("name")) {
            let name = name.trim().to_string();
            if name.to_ascii_lowercase().starts_with("package")
                && path.join("energy_uj").is_file()
                && path.join("max_energy_range_uj").is_file()
            {
                found.push(PackageEnergyCounter {
                    name,
                    directory: path.to_path_buf(),
                });
            }
        }
        if depth == 3 {
            return;
        }
        if let Ok(entries) = fs::read_dir(path) {
            for entry in entries.flatten() {
                let child = entry.path();
                if child.is_dir() {
                    collect(&child, depth + 1, found);
                }
            }
        }
    }

    let mut found = Vec::new();
    collect(Path::new("/sys/class/powercap"), 0, &mut found);
    found.sort_by(|a, b| a.directory.cmp(&b.directory));
    found.dedup_by(|a, b| a.directory == b.directory);
    found
}

fn read_package_energy_counters(
    counters: &[PackageEnergyCounter],
) -> Option<Vec<EnergyCounterReading>> {
    if counters.is_empty() {
        return None;
    }
    counters
        .iter()
        .map(|counter| {
            let energy_uj = fs::read_to_string(counter.directory.join("energy_uj"))
                .ok()?
                .trim()
                .parse::<u64>()
                .ok()?;
            let max_energy_range_uj = fs::read_to_string(
                counter.directory.join("max_energy_range_uj"),
            )
            .ok()?
            .trim()
            .parse::<u64>()
            .ok()?;
            if max_energy_range_uj == 0 || energy_uj > max_energy_range_uj {
                return None;
            }
            Some(EnergyCounterReading {
                name: counter.name.clone(),
                energy_uj,
                max_energy_range_uj,
            })
        })
        .collect()
}

fn package_energy_delta_uj(
    before: &[EnergyCounterReading],
    after: &[EnergyCounterReading],
) -> Option<u128> {
    if before.len() != after.len() {
        return None;
    }
    let mut total_delta = 0_u128;
    for (start, end) in before.iter().zip(after) {
        if start.name != end.name
            || start.max_energy_range_uj != end.max_energy_range_uj
            || start.energy_uj > start.max_energy_range_uj
            || end.energy_uj > end.max_energy_range_uj
        {
            return None;
        }
        let delta = if end.energy_uj >= start.energy_uj {
            end.energy_uj - start.energy_uj
        } else {
            // The kernel energy counter wraps at max_energy_range_uj.
            start.max_energy_range_uj - start.energy_uj + end.energy_uj
        };
        total_delta += u128::from(delta);
    }
    Some(total_delta)
}

fn build_package_energy_measurement(
    before: Option<Vec<EnergyCounterReading>>,
    after: Option<Vec<EnergyCounterReading>>,
    measurement_seconds: f64,
    executed_calls_including_warmup: usize,
) -> Option<PackageEnergyMeasurement> {
    if !measurement_seconds.is_finite()
        || measurement_seconds <= 0.0
        || executed_calls_including_warmup == 0
    {
        return None;
    }
    let before = before?;
    let after = after?;
    let energy_uj = package_energy_delta_uj(&before, &after)?;
    let package_energy_joules = energy_uj as f64 / 1_000_000.0;
    if !package_energy_joules.is_finite() {
        return None;
    }
    Some(PackageEnergyMeasurement {
        counter_domains: before.iter().map(|reading| reading.name.clone()).collect(),
        measurement_seconds,
        package_energy_joules,
        executed_calls_including_warmup,
        package_energy_joules_per_call: package_energy_joules
            / executed_calls_including_warmup as f64,
        limitation: "Observed CPU package energy during the benchmark window; includes operating-system/background activity and warm-up calls. Not a workload-isolated energy measurement.".into(),
    })
}

fn build_energy_comparisons(
    symthaea_pipeline_time_ms: f64,
    tdp: f64,
    tdp_label: &str,
) -> Vec<EnergyComparison> {
    let gpt3_energy = energy_per_inference(
        TRANSFORMER_REFS[0].power_watts,
        TRANSFORMER_REFS[0].time_per_inference_ms,
    );

    let mut comparisons = Vec::new();

    // Symthaea entry
    let sym_energy = energy_per_inference(tdp, symthaea_pipeline_time_ms);
    comparisons.push(EnergyComparison {
        system: format!("Symthaea HDC+CfC ({tdp_label})"),
        power_watts: tdp,
        time_per_inference_ms: symthaea_pipeline_time_ms,
        energy_per_inference_joules: sym_energy,
        modeled_ratio_vs_gpt3: sym_energy / gpt3_energy,
        basis: "Modeled from configured power assumption × measured pipeline latency; not metered energy.".into(),
    });

    // Transformer references
    for t in TRANSFORMER_REFS {
        let e = energy_per_inference(t.power_watts, t.time_per_inference_ms);
        comparisons.push(EnergyComparison {
            system: t.name.to_string(),
            power_watts: t.power_watts,
            time_per_inference_ms: t.time_per_inference_ms,
            energy_per_inference_joules: e,
            modeled_ratio_vs_gpt3: e / gpt3_energy,
            basis: "Illustrative power/latency assumptions; workload, hardware, and measurement comparability not established.".into(),
        });
    }

    comparisons
}

fn print_energy_table(comparisons: &[EnergyComparison]) {
    println!(
        "  {:<35} {:>10} {:>15} {:>18} {:>20}",
        "System", "Power (W)", "Time/Inf (ms)", "Modeled J/Inf", "Modeled ratio vs GPT-3"
    );
    println!("  {}", "-".repeat(106));
    for c in comparisons {
        println!(
            "  {:<35} {:>10.1} {:>15.3} {:>18.6} {:>20.6}",
            c.system,
            c.power_watts,
            c.time_per_inference_ms,
            c.energy_per_inference_joules,
            c.modeled_ratio_vs_gpt3,
        );
        println!("    Basis: {}", c.basis);
    }
}

fn main() {
    let total_start = Instant::now();

    println!("==========================================================");
    println!("  Energy Efficiency Measurement Benchmark");
    println!("  HDC Dimension: {}", HDC_DIMENSION);
    println!("==========================================================");

    // ── Generate synthetic data ────────────────────────────────────────
    println!("\n--- Generating synthetic plasma data ---");
    let config = SyntheticConfig {
        num_shots: 20,
        disruption_probability: 0.3,
        samples_per_shot: 500,
        sample_interval_ms: 1.0,
        seed: 42,
    };
    let shots = generate_synthetic_data(&config);
    let stats = compute_statistics(&shots);
    let normalizer = SensorNormalizer::from_stats(&stats);

    // Convert to CModPlasmaSamples
    let mut samples = Vec::new();
    for shot in &shots {
        for sample in &shot.samples {
            let plasma_sample = to_cmod_plasma_sample(sample, &normalizer, DisruptionLabel::Normal);
            samples.push(plasma_sample);
        }
    }
    // Take up to ENCODING_SAMPLES
    samples.truncate(ENCODING_SAMPLES);
    println!(
        "  Generated {} plasma samples from {} shots",
        samples.len(),
        shots.len()
    );

    // ── 1. HDC Encoding Throughput ─────────────────────────────────────
    println!("\n--- Phase 1: HDC Encoding Throughput ---");
    let encoder = CModHdcEncoder::default_encoder();

    let mut sample_idx = 0;
    let encoding_result = measure_with_stats("HDC Encoding", samples.len(), || {
        let s = &samples[sample_idx % samples.len()];
        let _ = encoder.encode(s);
        sample_idx += 1;
    });

    println!(
        "  Throughput: {:.0} samples/sec ({:.2} us/sample)",
        encoding_result.ops_per_second, encoding_result.us_per_op
    );
    println!(
        "  95% CI: [{:.2}, {:.2}] us/sample",
        encoding_result.ci_95_lower_us, encoding_result.ci_95_upper_us
    );
    println!("  Operations/sec: {:.2e}", encoding_result.ops_per_second);

    // ── 2. CfC Prediction Throughput ──────────────────────────────────
    println!("\n--- Phase 2: CfC Prediction Throughput ---");
    let predictor = PlasmaMultiScalePredictor::new();
    let input_hv = ContinuousHV::random(HDC_DIMENSION, 0xBE_0002);

    let mut pred_results = Vec::new();
    for (&horizon, &label) in PLASMA_HORIZONS.iter().zip(PLASMA_HORIZON_LABELS) {
        // Warmup
        for _ in 0..WARMUP_ITERATIONS {
            let _ = predictor.predict_at_horizon(&input_hv, horizon);
        }

        let start = Instant::now();
        for _ in 0..PREDICTION_ITERATIONS {
            let _ = predictor.predict_at_horizon(&input_hv, horizon);
        }
        let elapsed = start.elapsed();
        let us_per_pred = elapsed.as_nanos() as f64 / 1000.0 / PREDICTION_ITERATIONS as f64;
        let preds_per_sec = PREDICTION_ITERATIONS as f64 / elapsed.as_secs_f64();

        println!(
            "  {:<28} {:.2} us/pred  ({:.0} pred/s)",
            label, us_per_pred, preds_per_sec
        );

        pred_results.push(PredictionThroughput {
            label: label.to_string(),
            horizon_seconds: horizon as f64,
            predictions_per_second: preds_per_sec,
            us_per_prediction: us_per_pred,
        });
    }

    // ── 3. Full Pipeline Throughput ───────────────────────────────────
    println!("\n--- Phase 3: Full Pipeline (Encode + Predict + FEP Action) ---");
    let mut twin = FusionDigitalTwin::new();

    // Create PlasmaReadings from synthetic data for the twin
    use symthaea_physics::PlasmaReading;
    use symthaea_physics::PlasmaSensorType;

    let plasma_readings: Vec<Vec<PlasmaReading>> = samples
        .iter()
        .map(|s| {
            let sensors = &s.sensors;
            vec![
                PlasmaReading::new(
                    PlasmaSensorType::ElectronDensity,
                    sensors[1] as f64 * 2.0,
                    s.time_ms,
                ),
                PlasmaReading::new(
                    PlasmaSensorType::ElectronTemperature,
                    sensors[2] as f64 * 10.0,
                    s.time_ms,
                ),
                PlasmaReading::new(
                    PlasmaSensorType::RadiatedPower,
                    sensors[3] as f64 * 15.0,
                    s.time_ms,
                ),
                PlasmaReading::new(
                    PlasmaSensorType::SafetyFactor,
                    sensors[5] as f64 * 5.0,
                    s.time_ms,
                ),
            ]
        })
        .collect();

    // Set reference from first reading
    if !plasma_readings.is_empty() {
        twin.set_reference_shot(&plasma_readings[0]);
    }

    let pipeline_count = samples.len().min(5000);
    let mut reading_idx = 0;
    let package_energy_counters = discover_package_energy_counters();
    let package_energy_before = read_package_energy_counters(&package_energy_counters);
    let package_energy_window_start = Instant::now();
    let pipeline_result =
        measure_with_stats("Full Pipeline (encode+predict+FEP)", pipeline_count, || {
            let readings = &plasma_readings[reading_idx % plasma_readings.len()];
            let _ = twin.step(readings, 0.001);
            reading_idx += 1;
        });
    let package_energy_window_seconds = package_energy_window_start.elapsed().as_secs_f64();
    let package_energy_after = read_package_energy_counters(&package_energy_counters);
    let package_energy_measurement = build_package_energy_measurement(
        package_energy_before,
        package_energy_after,
        package_energy_window_seconds,
        pipeline_result.total_operations + WARMUP_ITERATIONS,
    );

    println!(
        "  Throughput: {:.0} inferences/sec ({:.2} us/inference)",
        pipeline_result.ops_per_second, pipeline_result.us_per_op
    );
    println!(
        "  95% CI: [{:.2}, {:.2}] us/inference",
        pipeline_result.ci_95_lower_us, pipeline_result.ci_95_upper_us
    );

    // ── 4. Memory Usage ──────────────────────────────────────────────
    println!("\n--- Phase 4: Memory Usage ---");

    let memory_results = vec![
        MemoryResult {
            component: "CModHdcEncoder".to_string(),
            size_bytes: mem::size_of::<CModHdcEncoder>(),
        },
        MemoryResult {
            component: "PlasmaMultiScalePredictor".to_string(),
            size_bytes: mem::size_of::<PlasmaMultiScalePredictor>(),
        },
        MemoryResult {
            component: "PlasmaFepAgent".to_string(),
            size_bytes: mem::size_of::<PlasmaFepAgent>(),
        },
        MemoryResult {
            component: "FusionDigitalTwin".to_string(),
            size_bytes: mem::size_of::<FusionDigitalTwin>(),
        },
        MemoryResult {
            component: format!("ContinuousHV ({}D)", HDC_DIMENSION),
            size_bytes: mem::size_of::<ContinuousHV>(),
        },
    ];

    println!(
        "  {:<40} {:>12} {:>12}",
        "Component", "Stack (B)", "Stack (KB)"
    );
    println!("  {}", "-".repeat(66));
    for m in &memory_results {
        println!(
            "  {:<40} {:>12} {:>12.2}",
            m.component,
            m.size_bytes,
            m.size_bytes as f64 / 1024.0,
        );
    }
    println!(
        "\n  Note: ContinuousHV heap allocation = {} * 4 = {} bytes ({:.1} KB)",
        HDC_DIMENSION,
        HDC_DIMENSION * 4,
        HDC_DIMENSION as f64 * 4.0 / 1024.0,
    );

    // ── 5. Energy Comparison ─────────────────────────────────────────
    let pipeline_time_ms = pipeline_result.us_per_op / 1000.0;

    println!("\n--- Phase 5: Energy Efficiency Comparison ---");
    println!("\n  Desktop power scenario ({POWER_ASSUMPTION_DESKTOP_W}W assumed):");
    let desktop_comparisons = build_energy_comparisons(pipeline_time_ms, POWER_ASSUMPTION_DESKTOP_W, "desktop");
    print_energy_table(&desktop_comparisons);

    println!("\n  Laptop power scenario ({POWER_ASSUMPTION_LAPTOP_W}W assumed):");
    let laptop_comparisons = build_energy_comparisons(pipeline_time_ms, POWER_ASSUMPTION_LAPTOP_W, "laptop");
    print_energy_table(&laptop_comparisons);

    // The power × latency values below are scenarios, not measured energy.
    let symthaea_desktop_energy = energy_per_inference(
        POWER_ASSUMPTION_DESKTOP_W,
        pipeline_time_ms,
    );
    let symthaea_laptop_energy = energy_per_inference(
        POWER_ASSUMPTION_LAPTOP_W,
        pipeline_time_ms,
    );

    println!("\\n==========================================================");
    println!("\n==========================================================");
    println!("==========================================================");
    println!(
        "  Pipeline time:                 {:.3} ms ({:.2} us)",
        pipeline_time_ms, pipeline_result.us_per_op
    );
    println!(
        "  Modeled energy @ {:.1} W:       {:.6} J/call (not metered)",
        POWER_ASSUMPTION_DESKTOP_W, symthaea_desktop_energy
    );
    println!(
        "  Modeled energy @ {:.1} W:       {:.6} J/call (not metered)",
        POWER_ASSUMPTION_LAPTOP_W, symthaea_laptop_energy
    );
    match &package_energy_measurement {
        Some(measurement) => {
            println!(
                "  Observed CPU package energy:    {:.6} J total over {:.3} s",
                measurement.package_energy_joules, measurement.measurement_seconds
            );
            println!(
                "  Package J/executed call:        {:.8} (includes warm-up/background)",
                measurement.package_energy_joules_per_call
            );
            println!(
                "  Counter domains:                {}",
                measurement.counter_domains.join(", ")
            );
        }
        None => println!(
            "  Observed CPU package energy:    unavailable (no readable Linux powercap/RAPL package counter)"
        ),
    }
    println!(
        "  Transformer comparison rows are illustrative power × latency assumptions, not verified apples-to-apples measurements."
    );
    println!(
        "  Encoding throughput:            {:.0} samples/sec",
        encoding_result.ops_per_second
    );
    println!(
        "  Pipeline throughput:            {:.0} inferences/sec",
        pipeline_result.ops_per_second
    );
    println!("==========================================================");

    let total_elapsed = total_start.elapsed();

    // ── JSON output ────────────────────────────────────────────────────
    let report = EnergyReport {
        hdc_dimension: HDC_DIMENSION,
        encoding_samples: samples.len(),
        prediction_iterations: PREDICTION_ITERATIONS,
        hdc_encoding: encoding_result,
        cfc_predictions: pred_results,
        full_pipeline: pipeline_result,
        package_energy_measurement,
        memory_usage: memory_results,
        energy_comparison_desktop: desktop_comparisons,
        energy_comparison_laptop: laptop_comparisons,
        total_wall_clock_seconds: total_elapsed.as_secs_f64(),
    };

    println!("\n--- JSON Output ---");
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("JSON serialization failed")
    );

    println!(
        "\nTotal benchmark wall-clock time: {:.2}s",
        total_elapsed.as_secs_f64()
    );
}


#[cfg(test)]
mod package_energy_counter_tests {
    use super::*;

    fn reading(name: &str, energy_uj: u64, max_energy_range_uj: u64) -> EnergyCounterReading {
        EnergyCounterReading {
            name: name.into(),
            energy_uj,
            max_energy_range_uj,
        }
    }

    #[test]
    fn energy_delta_handles_regular_counter_progress() {
        assert_eq!(
            package_energy_delta_uj(&[reading("package-0", 100, 1_000)], &[reading("package-0", 145, 1_000)]),
            Some(45)
        );
    }

    #[test]
    fn energy_delta_handles_one_counter_wrap() {
        assert_eq!(
            package_energy_delta_uj(&[reading("package-0", 990, 1_000)], &[reading("package-0", 20, 1_000)]),
            Some(30)
        );
    }

    #[test]
    fn energy_delta_rejects_changed_domains_or_ranges() {
        assert_eq!(
            package_energy_delta_uj(&[reading("package-0", 100, 1_000)], &[reading("package-1", 145, 1_000)]),
            None
        );
        assert_eq!(
            package_energy_delta_uj(&[reading("package-0", 100, 1_000)], &[reading("package-0", 145, 2_000)]),
            None
        );
    }

    #[test]
    fn direct_measurement_is_unavailable_without_counters() {
        assert!(build_package_energy_measurement(None, None, 1.0, 10).is_none());
    }

    #[test]
    fn direct_measurement_reports_package_energy_and_call_count() {
        let before = Some(vec![reading("package-0", 1_000, 10_000)]);
        let after = Some(vec![reading("package-0", 3_500, 10_000)]);
        let measurement = build_package_energy_measurement(before, after, 2.0, 25).unwrap();
        assert_eq!(measurement.package_energy_joules, 0.0025);
        assert_eq!(measurement.executed_calls_including_warmup, 25);
        assert!((measurement.package_energy_joules_per_call - 0.0001).abs() < 1e-12);
    }
}
