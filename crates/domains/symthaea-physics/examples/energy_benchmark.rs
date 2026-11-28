// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC+CfC throughput and scenario-energy diagnostic
//!
//! Measures wall-clock throughput and memory for the synthetic HDC+CfC
//! workload. The optional energy fields are scenario estimates formed from
//! assumed CPU power multiplied by measured latency. They are NOT hardware
//! energy measurements, do not represent whole-system energy, and must not be
//! compared with other models or systems.
//!
//! For hardware-backed energy receipts and matched-quality comparisons, use
//! the EnergyMeter and EnergyRunReceipt qualification path.
//!
//! Run: `cargo run -p symthaea-physics --example energy_benchmark --release`

#![deny(unsafe_code)]

use serde::Serialize;
use std::mem;
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

// Illustrative CPU-power scenarios only (watts). These values are not
// measurements, do not represent whole-system power, and are not TDP-derived
// claims about actual workload draw.
const ASSUMED_DESKTOP_CPU_POWER_WATTS: f64 = 65.0;
const ASSUMED_LAPTOP_CPU_POWER_WATTS: f64 = 15.0;

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
struct ScenarioEnergyEstimate {
    system_label: String,
    measurement_status: String,
    physical_scope: String,
    claim_ceiling: String,
    assumed_power_watts: f64,
    observed_pipeline_latency_ms: f64,
    scenario_energy_joules_not_measured: f64,
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
    memory_usage: Vec<MemoryResult>,
    scenario_energy_estimate_desktop: ScenarioEnergyEstimate,
    scenario_energy_estimate_laptop: ScenarioEnergyEstimate,
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

fn scenario_energy_from_assumed_power(assumed_power_watts: f64, observed_latency_ms: f64) -> f64 {
    // Dimensional scenario calculation: assumed W × measured s = scenario J.
    // This is not a meter reading and does not include unmeasured power domains.
    assumed_power_watts * observed_latency_ms / 1000.0
}

fn build_scenario_energy_estimate(
    observed_pipeline_latency_ms: f64,
    assumed_power_watts: f64,
    scenario_label: &str,
) -> ScenarioEnergyEstimate {
    ScenarioEnergyEstimate {
        system_label: scenario_label.to_string(),
        measurement_status: "SCENARIO_ESTIMATE_NOT_MEASURED".to_string(),
        physical_scope: "assumed CPU-power input only; no sensor coverage".to_string(),
        claim_ceiling: "not comparable to another system and not a hardware energy receipt"
            .to_string(),
        assumed_power_watts,
        observed_pipeline_latency_ms,
        scenario_energy_joules_not_measured: scenario_energy_from_assumed_power(
            assumed_power_watts,
            observed_pipeline_latency_ms,
        ),
    }
}

fn print_scenario_energy_estimate(estimate: &ScenarioEnergyEstimate) {
    println!("  Scenario: {}", estimate.system_label);
    println!(
        "  Assumed CPU power: {:.1} W (scenario input, not measured)",
        estimate.assumed_power_watts
    );
    println!(
        "  Observed pipeline latency: {:.3} ms",
        estimate.observed_pipeline_latency_ms
    );
    println!(
        "  Scenario energy: {:.6} J (NOT MEASURED; assumed power × observed latency)",
        estimate.scenario_energy_joules_not_measured
    );
}

fn main() {
    let total_start = Instant::now();

    println!("==========================================================");
    println!("  HDC+CfC Throughput and Scenario-Energy Diagnostic");
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
    let pipeline_result =
        measure_with_stats("Full Pipeline (encode+predict+FEP)", pipeline_count, || {
            let readings = &plasma_readings[reading_idx % plasma_readings.len()];
            let _ = twin.step(readings, 0.001);
            reading_idx += 1;
        });

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

    // ── 5. Scenario energy estimates (not hardware measurements) ─────
    let pipeline_time_ms = pipeline_result.us_per_op / 1000.0;

    println!("\n--- Phase 5: Scenario Energy Estimates (NOT MEASURED) ---");
    println!("These estimates multiply assumed CPU power by observed latency.");
    println!("They are not hardware-counter readings and are not comparable to other systems.");

    let desktop_estimate = build_scenario_energy_estimate(
        pipeline_time_ms,
        ASSUMED_DESKTOP_CPU_POWER_WATTS,
        "desktop CPU-power scenario",
    );
    print_scenario_energy_estimate(&desktop_estimate);

    let laptop_estimate = build_scenario_energy_estimate(
        pipeline_time_ms,
        ASSUMED_LAPTOP_CPU_POWER_WATTS,
        "laptop CPU-power scenario",
    );
    print_scenario_energy_estimate(&laptop_estimate);

    println!("\n==========================================================");
    println!("  THROUGHPUT AND SCENARIO-DIAGNOSTIC SUMMARY");
    println!("==========================================================");
    println!(
        "  Observed pipeline latency: {:.3} ms ({:.2} us)",
        pipeline_time_ms, pipeline_result.us_per_op
    );
    println!(
        "  Desktop scenario energy:   {:.6} J (not measured)",
        desktop_estimate.scenario_energy_joules_not_measured
    );
    println!(
        "  Laptop scenario energy:    {:.6} J (not measured)",
        laptop_estimate.scenario_energy_joules_not_measured
    );
    println!(
        "  Encoding throughput:       {:.0} samples/sec",
        encoding_result.ops_per_second
    );
    println!(
        "  Pipeline throughput:       {:.0} inferences/sec",
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
        memory_usage: memory_results,
        scenario_energy_estimate_desktop: desktop_estimate,
        scenario_energy_estimate_laptop: laptop_estimate,
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
mod scenario_energy_tests {
    use super::*;

    #[test]
    fn scenario_energy_arithmetic_preserves_watt_millisecond_units() {
        // 65 W × 100 ms = 6.5 J as a scenario calculation only.
        assert!((scenario_energy_from_assumed_power(65.0, 100.0) - 6.5).abs() < 1e-12);
    }

    #[test]
    fn zero_latency_produces_zero_scenario_energy() {
        assert_eq!(scenario_energy_from_assumed_power(65.0, 0.0), 0.0);
    }

    #[test]
    fn output_labels_the_energy_as_unmeasured() {
        let estimate = build_scenario_energy_estimate(100.0, 65.0, "test scenario");
        assert_eq!(
            estimate.measurement_status,
            "SCENARIO_ESTIMATE_NOT_MEASURED"
        );
        assert!(estimate.physical_scope.contains("no sensor coverage"));
        assert!(estimate.claim_ceiling.contains("not comparable"));
        assert_eq!(estimate.scenario_energy_joules_not_measured, 6.5);
    }
}
