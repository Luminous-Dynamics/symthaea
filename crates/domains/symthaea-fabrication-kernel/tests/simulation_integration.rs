// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration test: end-to-end manufacturing simulation pipeline.
//!
//! Verifies the complete flow: CSG design → mesh → slice (with infill) →
//! G-code → mock print → Cincinnati monitoring → QC → online defect prediction.
//! Each phase boundary has assertions.

use symthaea_fabrication_kernel::{
    cincinnati_live::{CincinnatiMonitor, CincinnatiMonitorConfig, SensorReading},
    csg::{BooleanOp, CSGNode, Primitive, Transform3D},
    defect_prediction::{DefectPredictor, PredictionInput},
    infill::{InfillConfig, InfillPattern},
    mesh::resolve_to_mesh,
    printer_control::{MockPrinter, PrinterApi, PrinterStatus},
    slicer::{SliceConfig, slice_mesh},
    toolpath::{ToolpathConfig, generate_gcode},
};

#[test]
fn full_pipeline_design_to_qc() {
    // ── Phase A: Design → Mesh ──────────────────────────────────────
    let cube = CSGNode::Transform {
        node: Box::new(CSGNode::Primitive(Primitive::Cube)),
        transform: Transform3D {
            scale: [20.0, 20.0, 10.0],
            ..Transform3D::default()
        },
    };
    let hole = CSGNode::Transform {
        node: Box::new(CSGNode::Primitive(Primitive::Cylinder)),
        transform: Transform3D {
            scale: [6.0, 6.0, 12.0],
            ..Transform3D::default()
        },
    };
    let design = CSGNode::Boolean {
        op: BooleanOp::Subtract,
        left: Box::new(cube),
        right: Box::new(hole),
    };
    let mesh = resolve_to_mesh(&design);
    assert!(mesh.vertices.len() > 100, "Mesh should have many vertices");
    assert!(mesh.indices.len() > 100, "Mesh should have many triangles");

    // ── Phase B: Slice with infill ──────────────────────────────────
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
    assert!(layers.len() >= 30, "Should produce >= 30 layers");

    let total_infill: usize = layers.iter().map(|l| l.infill_lines.len()).sum();
    assert!(total_infill > 0, "Infill must be generated");

    // Verify alternating angles: even/odd layers should have different infill directions.
    if layers.len() >= 4 {
        let even_lines = &layers[0].infill_lines;
        let odd_lines = &layers[1].infill_lines;
        if !even_lines.is_empty() && !odd_lines.is_empty() {
            let even_dx = (even_lines[0].end.x - even_lines[0].start.x).abs();
            let even_dy = (even_lines[0].end.y - even_lines[0].start.y).abs();
            let odd_dx = (odd_lines[0].end.x - odd_lines[0].start.x).abs();
            let odd_dy = (odd_lines[0].end.y - odd_lines[0].start.y).abs();
            let even_ratio = even_dx / (even_dy + 1e-6);
            let odd_ratio = odd_dx / (odd_dy + 1e-6);
            assert!(
                (even_ratio - odd_ratio).abs() > 0.1,
                "Even/odd layers should have different infill angles (ratios: {:.2} vs {:.2})",
                even_ratio,
                odd_ratio
            );
        }
    }

    // ── Phase C: G-code generation ──────────────────────────────────
    let toolpath_config = ToolpathConfig::default();
    let gcode = generate_gcode(&layers, &slice_config, &toolpath_config);
    assert!(
        gcode.command_count() > 1000,
        "Complex design should have many G-code commands"
    );
    assert!(
        gcode.total_extrusion_mm > 10.0,
        "Should have significant extrusion"
    );

    let gcode_str = gcode.to_gcode_string();
    assert!(
        gcode_str.contains("Infill"),
        "G-code should contain infill section"
    );

    // ── Phase D: Mock print ─────────────────────────────────────────
    let mut printer = MockPrinter::new();
    printer.connect().unwrap();
    let job_id = printer.submit_gcode(&gcode_str).unwrap();
    assert!(job_id.starts_with("mock-job-"));

    // Advance to completion.
    printer.advance_progress(1.0);
    let status = printer.status().unwrap();
    assert!(
        matches!(status, PrinterStatus::Idle),
        "Printer should be idle after full progress, got {:?}",
        status
    );

    // ── Phase E: Cincinnati monitoring ──────────────────────────────
    let mut monitor = CincinnatiMonitor::new(CincinnatiMonitorConfig::default());

    // Build baseline with normal readings.
    for i in 0..20 {
        for ch in &[
            "temperature_nozzle",
            "temperature_bed",
            "vibration_x",
            "extrusion_pressure",
        ] {
            monitor.ingest_reading(SensorReading {
                channel: ch.to_string(),
                value: 200.0 + i as f64 * 0.5,
                timestamp_ms: i * 100,
            });
        }
    }
    let baseline_anomalies = monitor.anomaly_count();

    // Inject anomalous readings.
    for ch in &["temperature_nozzle", "vibration_x"] {
        monitor.ingest_reading(SensorReading {
            channel: ch.to_string(),
            value: 500.0, // Far outside normal range.
            timestamp_ms: 5000,
        });
    }
    assert!(
        monitor.anomaly_count() > baseline_anomalies,
        "Anomalous readings should be detected"
    );

    // ── Phase F: QC with online defect prediction ───────────────────
    let anomaly_count = monitor.anomaly_count();
    let quality = (1.0 - anomaly_count as f64 * 0.05).clamp(0.0, 1.0);
    assert!(quality > 0.0 && quality <= 1.0);

    let mut predictor = DefectPredictor::new();

    // Online training: feed samples incrementally.
    for i in 0..20 {
        let noise = (i as f64 * 0.1).sin().abs() * 0.05;
        let input = PredictionInput {
            tolerance: 0.8 + noise,
            surface_quality: 0.85 - noise,
            throughput: 0.7,
            energy_cost: 0.3,
            anomaly_count: anomaly_count as u32,
            layer_count: layers.len() as u32,
        };
        predictor.train(&input, quality);
    }

    assert_eq!(predictor.training_count, 20);
    let prediction = predictor.predict(&PredictionInput {
        tolerance: 0.8,
        surface_quality: 0.85,
        throughput: 0.7,
        energy_cost: 0.3,
        anomaly_count: anomaly_count as u32,
        layer_count: layers.len() as u32,
    });
    assert!(
        prediction.predicted_quality >= 0.0 && prediction.predicted_quality <= 1.0,
        "Prediction must be in [0, 1]"
    );
    assert!(
        prediction.confidence > 0.0,
        "After training, confidence should be > 0"
    );
}

#[test]
fn grid_infill_integration() {
    let mesh = resolve_to_mesh(&CSGNode::Primitive(Primitive::Cube));
    let config = SliceConfig {
        layer_height: 0.2,
        nozzle_diameter: 0.4,
        tolerance: 1e-3,
        infill: Some(InfillConfig {
            pattern: InfillPattern::Grid,
            density: 0.3,
            angle_degrees: 0.0,
        }),
    };
    let layers = slice_mesh(&mesh, &config);
    assert!(!layers.is_empty());

    // Grid should produce infill in every layer with contours.
    for layer in &layers {
        if !layer.outer_contours.is_empty() {
            assert!(
                !layer.infill_lines.is_empty(),
                "Grid infill should produce lines for layer at z={}",
                layer.z
            );
        }
    }
}

#[test]
fn online_defect_prediction_improves() {
    let mut predictor = DefectPredictor::new();

    // Phase 1: Train on good data.
    for _ in 0..50 {
        predictor.train(
            &PredictionInput {
                tolerance: 0.9,
                surface_quality: 0.95,
                throughput: 0.8,
                energy_cost: 0.2,
                anomaly_count: 0,
                layer_count: 100,
            },
            0.95,
        );
    }

    let good_pred = predictor.predict(&PredictionInput {
        tolerance: 0.9,
        surface_quality: 0.95,
        throughput: 0.8,
        energy_cost: 0.2,
        anomaly_count: 0,
        layer_count: 100,
    });

    let bad_pred = predictor.predict(&PredictionInput {
        tolerance: 0.1,
        surface_quality: 0.1,
        throughput: 0.2,
        energy_cost: 0.9,
        anomaly_count: 10,
        layer_count: 5,
    });

    assert!(
        good_pred.predicted_quality > bad_pred.predicted_quality,
        "After training, good input ({:.3}) should score higher than bad ({:.3})",
        good_pred.predicted_quality,
        bad_pred.predicted_quality
    );
    assert!(
        good_pred.confidence >= 0.5,
        "50 samples should give >= 0.5 confidence"
    );
}
