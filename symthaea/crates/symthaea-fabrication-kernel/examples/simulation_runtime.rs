//! End-to-end manufacturing simulation pipeline.
//!
//! Drives the complete pipeline: design → slice (with infill) → G-code →
//! mock print → Cincinnati monitoring → QC → defect prediction → report.
//!
//! Run: cargo run -p symthaea-fabrication-kernel --example simulation_runtime

use symthaea_fabrication_kernel::{
    cincinnati_live::{CincinnatiMonitor, CincinnatiMonitorConfig, SensorReading},
    csg::{BooleanOp, CSGNode, Primitive, Transform3D},
    defect_prediction::{DefectPredictor, PredictionInput},
    infill::{InfillConfig, InfillPattern},
    mesh::resolve_to_mesh,
    printer_control::{MockPrinter, PrinterApi},
    slicer::{slice_mesh, SliceConfig},
    toolpath::{generate_gcode, ToolpathConfig},
};

// ── Constants ────────────────────────────────────────────────────────────

const TOTAL_STEPS: u32 = 100;
const ANOMALY_STEPS: &[u32] = &[30, 70];
const ANOMALY_SEVERITY: f64 = 8.0; // Z-score multiplier for anomaly injection.

// ── Helper functions ─────────────────────────────────────────────────────

/// Create a CSG design: cube with a cylindrical hole subtracted.
fn create_design() -> CSGNode {
    let cube = CSGNode::Transform {
        node: Box::new(CSGNode::Primitive(Primitive::Cube)),
        transform: Transform3D {
            scale: [20.0, 20.0, 10.0],
            ..Transform3D::default()
        },
    };
    let cylinder = CSGNode::Transform {
        node: Box::new(CSGNode::Primitive(Primitive::Cylinder)),
        transform: Transform3D {
            scale: [6.0, 6.0, 12.0],
            ..Transform3D::default()
        },
    };
    CSGNode::Boolean {
        op: BooleanOp::Subtract,
        left: Box::new(cube),
        right: Box::new(cylinder),
    }
}

/// Simulate sensor readings for a given step.
/// Returns 4-channel readings with noise and anomaly injection at specified steps.
fn simulate_sensor_readings(step: u32, total: u32) -> Vec<SensorReading> {
    let progress = step as f64 / total as f64;
    let is_anomaly = ANOMALY_STEPS.contains(&step);

    let noise = ((step as f64 * 7.3).sin() * 0.5).abs();
    let anomaly_boost = if is_anomaly { ANOMALY_SEVERITY } else { 0.0 };

    vec![
        SensorReading {
            channel: "temperature_nozzle".into(),
            value: 200.0 + progress * 10.0 + noise + anomaly_boost * 3.0,
            timestamp_ms: step as u64 * 100,
        },
        SensorReading {
            channel: "temperature_bed".into(),
            value: 60.0 + progress * 2.0 + noise * 0.3,
            timestamp_ms: step as u64 * 100,
        },
        SensorReading {
            channel: "vibration_x".into(),
            value: 0.5 + noise * 0.2 + anomaly_boost * 0.5,
            timestamp_ms: step as u64 * 100,
        },
        SensorReading {
            channel: "extrusion_pressure".into(),
            value: 45.0 + progress * 5.0 + noise + anomaly_boost * 2.0,
            timestamp_ms: step as u64 * 100,
        },
    ]
}

/// Simulate temperature ramp: nozzle from 25°C to 200°C, bed from 25°C to 60°C.
fn simulate_temperatures(step: u32) -> (f32, f32) {
    let t = (step as f32 / 10.0).min(1.0); // Ramp over first 10 steps.
    let nozzle = 25.0 + t * 175.0;
    let bed = 25.0 + t * 35.0;
    (nozzle, bed)
}

/// Quality check: quality = 1.0 - (anomaly_count * 0.05), clamped to [0, 1].
fn run_quality_check(anomaly_count: u64) -> f64 {
    (1.0 - anomaly_count as f64 * 0.05).clamp(0.0, 1.0)
}

/// Generate training data from simulation results.
fn generate_training_data(
    anomaly_count: u64,
    layer_count: u32,
    quality: f64,
) -> Vec<(PredictionInput, f64)> {
    let base_tolerance = if anomaly_count > 5 { 0.3 } else { 0.85 };
    let base_surface = if anomaly_count > 3 { 0.4 } else { 0.9 };

    (0..5)
        .map(|i| {
            let noise = (i as f64 * 0.03).sin().abs() * 0.05;
            (
                PredictionInput {
                    tolerance: (base_tolerance + noise).clamp(0.0, 1.0),
                    surface_quality: (base_surface - noise).clamp(0.0, 1.0),
                    throughput: 0.7 + noise,
                    energy_cost: 0.35 + noise * 0.5,
                    anomaly_count: anomaly_count as u32,
                    layer_count,
                },
                quality,
            )
        })
        .collect()
}

/// Report consciousness bridge mapping: manufacturing events → neuromodulator pathways.
fn report_consciousness_bridge(anomaly_count: u64, quality: f64) {
    println!("\n=== Phase F: Consciousness Bridge Mapping ===");
    println!("  Manufacturing events → neuromodulator pathways:");

    // Anomaly detection → NE (norepinephrine) alertness.
    let ne_boost = (anomaly_count as f64 * 0.1).min(1.0);
    println!(
        "  NE (alertness):     +{:.2} (from {} anomalies)",
        ne_boost, anomaly_count
    );

    // Quality outcome → DA (dopamine) reward.
    let da_signal = quality - 0.5; // Positive = reward, negative = punishment.
    println!(
        "  DA (reward):        {:+.2} (quality={:.2})",
        da_signal, quality
    );

    // Process stability → 5-HT (serotonin) regulation.
    let stability = 1.0 - (anomaly_count as f64 / TOTAL_STEPS as f64);
    println!(
        "  5-HT (regulation):  {:.2} (stability={:.2})",
        stability * 0.5,
        stability
    );

    // Swarm coordination → OT (oxytocin) bonding.
    println!("  OT (coordination):  0.30 (mock swarm signal)");
}

// ── Main pipeline ────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Symthaea Fabrication Kernel — Simulation Runtime           ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // ── Phase A: Design → Mesh → Slice → G-code ─────────────────────
    println!("=== Phase A: Design → Mesh → Slice (with infill) → G-code ===");

    let design = create_design();
    let mesh = resolve_to_mesh(&design);
    println!(
        "  Mesh: {} vertices, {} triangles",
        mesh.vertices.len(),
        mesh.indices.len()
    );

    let slice_config = SliceConfig {
        layer_height: 0.2,
        nozzle_diameter: 0.4,
        tolerance: 1e-3,
        infill: Some(InfillConfig {
            pattern: InfillPattern::Rectilinear,
            density: 0.2,
            angle_degrees: 45.0,
        }),
    };
    let layers = slice_mesh(&mesh, &slice_config);
    let total_contours: usize = layers
        .iter()
        .map(|l| l.outer_contours.len() + l.inner_contours.len())
        .sum();
    let total_infill: usize = layers.iter().map(|l| l.infill_lines.len()).sum();
    println!(
        "  Layers: {}, contours: {}, infill segments: {}",
        layers.len(),
        total_contours,
        total_infill
    );

    let toolpath_config = ToolpathConfig::default();
    let gcode = generate_gcode(&layers, &slice_config, &toolpath_config);
    println!(
        "  G-code: {} commands, {:.1} mm extrusion",
        gcode.command_count(),
        gcode.total_extrusion_mm
    );

    // ── Phase B: Submit to MockPrinter ───────────────────────────────
    println!("\n=== Phase B: Submit G-code to MockPrinter ===");
    let mut printer = MockPrinter::new();
    printer.connect().expect("mock connect");
    let gcode_str = gcode.to_gcode_string();
    let job_id = printer.submit_gcode(&gcode_str).expect("submit gcode");
    println!("  Job submitted: {}", job_id);
    println!("  G-code size: {} bytes", gcode_str.len());

    // ── Phase C: Simulate print with Cincinnati monitoring ───────────
    println!(
        "\n=== Phase C: Simulate {}-step print with monitoring ===",
        TOTAL_STEPS
    );
    let monitor_config = CincinnatiMonitorConfig::default();
    let mut monitor = CincinnatiMonitor::new(monitor_config);
    let mut anomaly_events: Vec<(u32, String)> = Vec::new();

    for step in 0..TOTAL_STEPS {
        let readings = simulate_sensor_readings(step, TOTAL_STEPS);
        for reading in readings {
            if let Some(alert) = monitor.ingest_reading(reading) {
                anomaly_events.push((
                    step,
                    format!(
                        "{}: severity={:.2}, z={:.1}",
                        alert.channel, alert.severity, alert.z_score
                    ),
                ));
            }
        }
        // Advance mock printer progress.
        printer.advance_progress(1.0 / TOTAL_STEPS as f32);
    }

    println!("  Total sensor readings: {}", monitor.history_len());
    println!("  Anomalies detected: {}", monitor.anomaly_count());
    for (step, desc) in &anomaly_events {
        println!("    Step {}: {}", step, desc);
    }

    // ── Phase D: QC check ────────────────────────────────────────────
    println!("\n=== Phase D: Quality Control ===");
    let quality = run_quality_check(monitor.anomaly_count());
    let pass = quality >= 0.7;
    println!("  Quality score: {:.2}", quality);
    println!("  Result: {}", if pass { "PASS ✓" } else { "FAIL ✗" });

    // ── Phase E: Train DefectPredictor ───────────────────────────────
    println!("\n=== Phase E: Defect Prediction Training ===");
    let mut predictor = DefectPredictor::new();
    let training_data =
        generate_training_data(monitor.anomaly_count(), layers.len() as u32, quality);

    for (input, actual) in &training_data {
        predictor.train(input, *actual);
    }
    println!("  Training samples: {}", predictor.training_count);

    // Predict on the last sample.
    let last_input = &training_data.last().unwrap().0;
    let prediction = predictor.predict(last_input);
    println!("  Predicted quality: {:.3}", prediction.predicted_quality);
    println!("  Confidence: {:.3}", prediction.confidence);
    if !prediction.risk_factors.is_empty() {
        println!("  Risk factors:");
        for rf in &prediction.risk_factors {
            println!(
                "    {}: contribution={:.3}, severity={:.2}",
                rf.name, rf.contribution, rf.severity
            );
        }
    }

    let importance = predictor.feature_importance();
    println!("  Feature importance:");
    for (name, weight) in &importance {
        println!("    {}: {:.4}", name, weight);
    }

    // ── Phase F: Consciousness bridge report ─────────────────────────
    report_consciousness_bridge(monitor.anomaly_count(), quality);

    // ── Phase G: Summary ─────────────────────────────────────────────
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  Simulation Complete                                        ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║  Mesh:       {:>6} triangles                               ║",
        mesh.indices.len()
    );
    println!(
        "║  Layers:     {:>6}                                         ║",
        layers.len()
    );
    println!(
        "║  Infill:     {:>6} segments                                ║",
        total_infill
    );
    println!(
        "║  G-code:     {:>6} commands                                ║",
        gcode.command_count()
    );
    println!(
        "║  Anomalies:  {:>6}                                         ║",
        monitor.anomaly_count()
    );
    println!(
        "║  Quality:    {:>6.2}                                        ║",
        quality
    );
    println!(
        "║  QC:         {:>6}                                         ║",
        if pass { "PASS" } else { "FAIL" }
    );
    println!(
        "║  Predicted:  {:>6.3}                                       ║",
        prediction.predicted_quality
    );
    println!("╚══════════════════════════════════════════════════════════════╝");
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_sensor_readings_count() {
        let readings = simulate_sensor_readings(50, 100);
        assert_eq!(readings.len(), 4, "Should produce 4 channel readings");
        for r in &readings {
            assert!(r.value.is_finite(), "Reading value must be finite");
        }
    }

    #[test]
    fn test_simulate_temperatures_ramp() {
        let (nozzle_0, bed_0) = simulate_temperatures(0);
        assert!(
            (nozzle_0 - 25.0).abs() < 0.1,
            "Nozzle should start at ~25°C"
        );
        assert!((bed_0 - 25.0).abs() < 0.1, "Bed should start at ~25°C");

        let (nozzle_end, bed_end) = simulate_temperatures(100);
        assert!(
            (nozzle_end - 200.0).abs() < 0.1,
            "Nozzle should reach ~200°C"
        );
        assert!((bed_end - 60.0).abs() < 0.1, "Bed should reach ~60°C");
    }

    #[test]
    fn test_quality_check_no_anomalies() {
        let q = run_quality_check(0);
        assert!((q - 1.0).abs() < 1e-6, "Zero anomalies → quality 1.0");

        let q5 = run_quality_check(5);
        assert!((q5 - 0.75).abs() < 1e-6, "5 anomalies → quality 0.75");

        let q_many = run_quality_check(100);
        assert!(q_many >= 0.0, "Quality must not go negative");
    }
}
